"""Stopping the updater while one of its commands runs.

The manager stops updated with SIGINT to the Python process alone and kills it 5 seconds later
(system/manager/process.py). The tests do the same to a child process that runs a command through updated.run()."""
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from openpilot.common.test import OpenpilotTestCase
from openpilot.system.updated import updated
from openpilot.system.updated.tests.helpers import GIT_ENV, git, locks, reset_process_state, running, tmp_root

MANAGER_KILLS_AFTER = 5.0   # s, system/manager/process.py: join_process(self.proc, 5), then SIGKILL
STOPPED = 42                # exit code of the child when run() ended with KeyboardInterrupt

# what the updater process is, for these tests: the SIGINT handler of main(), then one command
CHILD = r'''
import signal, sys, time
from pathlib import Path
from openpilot.system.updated import updated
updated.OVERLAY_INIT = Path(sys.argv[1])
signal.signal(signal.SIGINT, updated.request_stop)
try:
  print("ready", flush=True)
  if sys.argv[2] == "--wait":
    time.sleep(60)
  else:
    updated.run(sys.argv[3:], sys.argv[2])
except KeyboardInterrupt:
  sys.exit(42)
'''


def wait_for(condition, timeout: float = 20.0) -> None:
  end = time.monotonic() + timeout
  while not condition():
    assert time.monotonic() < end, "timed out waiting for the test's start condition"
    time.sleep(0.01)


class TestStop(OpenpilotTestCase):
  def setup_method(self):
    os.environ.update(GIT_ENV)
    reset_process_state(self)
    self.root = tmp_root(self)
    self.overlay_init = self.root / ".overlay_init"
    self.overlay_init.touch()
    self.pids = self.root / "pids"

  def _start(self, *args: str) -> subprocess.Popen:
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(self.overlay_init), *args], stdout=subprocess.PIPE, text=True)
    self._reap(child)
    assert child.stdout.readline().strip() == "ready"
    return child

  def _reap(self, proc: subprocess.Popen) -> None:
    self.addCleanup(proc.stdout.close)
    self.addCleanup(proc.wait)
    self.addCleanup(lambda: proc.poll() is None and os.kill(proc.pid, signal.SIGKILL))

  def _stop(self, child: subprocess.Popen) -> float:
    """what the manager does; returns the seconds until the process is gone"""
    t = time.monotonic()
    os.kill(child.pid, signal.SIGINT)
    child.wait(timeout=MANAGER_KILLS_AFTER)
    return time.monotonic() - t

  def _command_pids(self) -> list[int]:
    return [int(pid) for pid in self.pids.read_text().split()]

  def _repo_with_slow_checkout(self) -> Path:
    """a repository in which `git reset --hard` holds .git/index.lock until it is stopped: it has to write one file
    through a filter that never finishes (the same mechanism as the LFS filter on the device)"""
    repo = self.root / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / ".gitattributes").write_text("*.dat filter=slow\n")
    (repo / "big.dat").write_text("content\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "one filtered file")
    git(repo, "config", "filter.slow.smudge", f"sh -c 'sleep 60 & echo $$ $! > {self.pids}; wait'")
    (repo / "big.dat").unlink()
    return repo

  def test_stop_during_git_leaves_no_lock(self):
    repo = self._repo_with_slow_checkout()
    child = self._start(str(repo), "git", "reset", "--hard")
    wait_for(lambda: self.pids.exists() and len(self.pids.read_text().split()) == 2 and locks(repo))
    assert locks(repo) == [".git/index.lock"]
    filter_pids = self._command_pids()

    seconds = self._stop(child)

    assert child.returncode == STOPPED
    assert seconds < updated.STOP_TERM_WAIT, seconds                                   # git ends on SIGTERM at once
    assert locks(repo) == []
    assert not any(running(pid) for pid in filter_pids)             # git's children went with it: nothing left behind
    assert self.overlay_init.exists()                               # nothing was killed, so nothing to rebuild
    git(repo, "config", "--unset", "filter.slow.smudge")
    git(repo, "reset", "--hard")                                    # and the next git command works

  def test_control_python_default_stop_leaves_the_lock(self):
    # what subprocess.check_output() does on SIGINT (the updater's run() until now): SIGKILL to git alone
    repo = self._repo_with_slow_checkout()
    control = "import subprocess, sys\nprint('ready', flush=True)\nsubprocess.check_output(sys.argv[3:], cwd=sys.argv[2])"
    child = subprocess.Popen([sys.executable, "-c", control, "-", str(repo), "git", "reset", "--hard"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    self._reap(child)
    assert child.stdout.readline().strip() == "ready"
    wait_for(lambda: self.pids.exists() and len(self.pids.read_text().split()) == 2 and locks(repo))
    filter_pids = self._command_pids()
    for pid in filter_pids:
      self.addCleanup(lambda pid=pid: running(pid) and os.kill(pid, signal.SIGKILL))

    self._stop(child)

    assert locks(repo) == [".git/index.lock"]
    assert any(running(pid) for pid in filter_pids)                 # and git's children run on as orphans

  def test_stop_kills_what_ignores_sigterm_and_marks_the_staging_area(self):
    script = f"trap '' TERM; sleep 60 & echo $$ $! > {self.pids}; wait"
    child = self._start(str(self.root), "sh", "-c", script)
    wait_for(lambda: self.pids.exists() and len(self.pids.read_text().split()) == 2)
    command_pids = self._command_pids()

    seconds = self._stop(child)

    assert child.returncode == STOPPED
    assert updated.STOP_TERM_WAIT <= seconds < 0.8 * MANAGER_KILLS_AFTER, seconds
    assert not any(running(pid) for pid in command_pids)
    assert not self.overlay_init.exists()                           # marked for rebuild

  def test_stop_reaches_what_a_command_left_running(self):
    # the command itself has exited; a process it started still holds its output, so run() is still waiting
    child = self._start(str(self.root), "sh", "-c", f"sleep 60 & echo $! $! > {self.pids}")
    wait_for(lambda: self.pids.exists() and len(self.pids.read_text().split()) == 2)
    time.sleep(0.3)
    left_running = self._command_pids()
    assert all(running(pid) for pid in left_running)

    seconds = self._stop(child)

    assert child.returncode == STOPPED and seconds < updated.STOP_TERM_WAIT, (child.returncode, seconds)
    assert not any(running(pid) for pid in left_running)
    assert self.overlay_init.exists()

  def test_stop_outside_a_command_is_immediate(self):
    child = self._start("--wait")
    seconds = self._stop(child)
    assert child.returncode == STOPPED
    assert seconds < 1.0, seconds

  def test_stop_bound_is_under_the_managers(self):
    assert updated.STOP_POLL + updated.STOP_TERM_WAIT + updated.STOP_KILL_WAIT < 0.8 * MANAGER_KILLS_AFTER

  def test_handler_holds_the_interrupt_back_only_during_a_command(self):
    updated.ProcessState.in_command = True
    updated.request_stop(signal.SIGINT, None)
    assert updated.ProcessState.stop_requested
    updated.ProcessState.in_command = updated.ProcessState.stop_requested = False
    with self.assertRaises(KeyboardInterrupt):
      updated.request_stop(signal.SIGINT, None)

  def test_request_that_arrives_as_the_command_ends_still_stops(self):
    updated.ProcessState.stop_requested = True
    with self.assertRaises(KeyboardInterrupt):
      updated.run(["true"])
    assert not updated.ProcessState.in_command

  def _fake_sudo(self) -> None:
    """a `sudo` that just runs the command, first in PATH: the tests must not depend on the real one"""
    bin_dir = self.root / "bin"
    bin_dir.mkdir()
    (bin_dir / "sudo").write_text('#!/bin/sh\nexec "$@"\n')
    (bin_dir / "sudo").chmod(0o755)
    os.environ["PATH"] = f"{bin_dir}:{os.environ['PATH']}"

  def test_stop_during_a_real_sudo_command(self):
    # sudo accepts the signal and passes it on to its command (seen with sudo 1.9.15, as root and as another user).
    # Runs wherever sudo works without a password (`sudo -n true`): as root, and for a user set up like the device's.
    if shutil.which("sudo") is None or subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0:
      self.skipTest("needs a sudo that runs without a password")
    child = self._start(str(self.root), "sudo", "sh", "-c", f"echo $$ > {self.pids}; exec sleep 60")
    wait_for(lambda: self.pids.exists() and len(self.pids.read_text().split()) == 1)
    time.sleep(0.2)                                                 # sh has become sleep
    command_pids = self._command_pids()
    assert all(running(pid) for pid in command_pids)

    seconds = self._stop(child)

    assert child.returncode == STOPPED and seconds < updated.STOP_TERM_WAIT, (child.returncode, seconds)
    assert not any(running(pid) for pid in command_pids)            # nothing left
    assert self.overlay_init.exists()                               # nothing marked

  def test_sudo_command_that_cannot_be_signalled_is_left_after_the_bound(self, mocker):
    # the fallback: where sudo refuses the signal (EPERM), the stop does not wait for the command beyond the bound
    self._fake_sudo()
    mocker.patch.object(updated, "OVERLAY_INIT", self.overlay_init)
    mocker.patch.object(updated, "STOP_TERM_WAIT", 0.4)
    mocker.patch.object(updated, "STOP_KILL_WAIT", 0.2)
    refused = []

    def send_signal(proc, sig):                                     # a sudo that the updater's user may not signal
      refused.append((proc, sig))
      raise PermissionError(1, "Operation not permitted")
    mocker.patch.object(subprocess.Popen, "send_signal", send_signal)
    stop = threading.Timer(0.3, lambda: setattr(updated.ProcessState, "stop_requested", True))
    stop.start()
    self.addCleanup(stop.cancel)

    t = time.monotonic()
    with self.assertRaises(KeyboardInterrupt):
      updated.run(["sudo", "sleep", "60"])
    seconds = time.monotonic() - t

    proc = refused[0][0]
    self._reap(proc)
    assert [sig for _, sig in refused] == [signal.SIGTERM, signal.SIGKILL]
    assert seconds < 0.3 + updated.STOP_POLL + 0.4 + 0.2 + 0.5, seconds   # it did not wait for the command
    assert running(proc.pid)                                        # which is still there: it could not be ended
    assert not self.overlay_init.exists()                           # so the staging area is marked for rebuild
    assert not updated.ProcessState.overlay_built

  def test_only_sudo_commands_stay_in_the_updaters_session(self):
    self._fake_sudo()
    show = "import os; print(os.getpid(), os.getsid(0))"
    pid, session = updated.run([sys.executable, "-c", show]).split()
    assert pid == session                                           # its own session and process group
    pid, session = updated.run(["sudo", sys.executable, "-c", show]).split()
    assert int(session) == os.getsid(0) and pid != session          # started as before

  def test_run_keeps_the_contract_of_check_output(self):
    assert updated.run(["sh", "-c", "echo out; echo err >&2"]) == "out\nerr\n"
    assert updated.run(["pwd"], str(self.root)).strip() == str(self.root)
    failing = ["sh", "-c", "echo before; echo why >&2; exit 3"]
    with self.assertRaises(subprocess.CalledProcessError) as failed:
      updated.run(failing)
    assert (failed.exception.returncode, failed.exception.cmd, failed.exception.output) == (3, failing, "before\nwhy\n")
    with self.assertRaises(FileNotFoundError):
      updated.run(["no-such-command-for-the-updater-test"])
    assert not updated.ProcessState.in_command
