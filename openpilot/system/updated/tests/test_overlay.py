"""The staging overlay: an updated process builds on a staging area only if it built that staging area itself.

TestOverlayRule decides this without mounting anything (the commands are answered by a stand-in).
TestRealOverlay runs the whole updater, sudo and overlay mount included, where that is possible (root, overlayfs)."""
import os
import shutil
import signal
import subprocess
import sys
import time

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.updated import updated
from openpilot.system.updated.tests.helpers import (BRANCH, GIT_ENV, FakeCommands, Remotes, git, locks, reset_process_state, run_main,
                                                    running, tmp_root, use_dirs)

FIRST_WAIT, AFTER_FAILURE, AFTER_SUCCESS = 60, 5 * 60, 1.5 * 60 * 60
MANAGER_KILLS_AFTER = 5.0  # s, system/manager/process.py

# an updater process: updated.main() on the test's folders, with its first one-minute wait skipped
UPDATER_PROCESS = r'''
import sys
from pathlib import Path
from openpilot.system.updated import updated
from openpilot.system.updated.tests.helpers import updater_dirs, updater_settings
for name, value in updater_settings(updater_dirs(Path(sys.argv[1]), Path(sys.argv[2]))).items():
  setattr(updated, name, value)
updated.system_time_valid = lambda: True
updated.WaitTimeHelper.sleep = lambda self, t: None if t == 60 else sys.exit(f"unexpected wait of {t} s")
try:
  updated.main()
except KeyboardInterrupt:  # as the manager's launcher does
  sys.exit(42)
'''


class TestOverlayRule(OpenpilotTestCase):
  def setup_method(self):
    reset_process_state(self)

  def _device(self, mocker) -> FakeCommands:
    self.dirs = use_dirs(mocker, tmp_root(self))
    self.dirs.basedir.mkdir()
    self.commands = FakeCommands(self.dirs).install(mocker)
    return self.commands

  def _rebuilds(self) -> int:
    return len(self.commands.commands("sudo", "mount"))

  def _leave_staging_area_of_an_earlier_process(self):
    """everything the reuse shortcut looks at says "valid": mounted, BASEDIR/.overlay_init present and newer than .git"""
    updated.init_overlay()
    (self.dirs.merged / ".git").mkdir(exist_ok=True)
    (self.dirs.merged / ".git/index.lock").touch()                  # e.g. left by a git command that was killed
    self.commands.calls.clear()
    reset_process_state(self)                                       # a new process
    assert self.dirs.overlay_init.is_file() and os.path.ismount(str(self.dirs.merged))

  def test_new_process_rebuilds_a_staging_area_it_did_not_build(self, mocker):
    self._device(mocker)
    self._leave_staging_area_of_an_earlier_process()

    updated.init_overlay()

    cmds = [cmd for cmd, _ in self.commands.calls]
    assert cmds[:2] == [["sudo", "umount", "-l", str(self.dirs.merged)], ["sudo", "rm", "-rf", str(self.dirs.staging)]], cmds
    assert self._rebuilds() == 1 and self.commands.commands("find") == []
    assert not (self.dirs.merged / ".git/index.lock").exists()
    assert updated.ProcessState.overlay_built

  def test_same_process_reuses_the_staging_area_it_built(self, mocker):
    self._device(mocker)
    updated.init_overlay()
    self.commands.calls.clear()

    updated.init_overlay()

    assert [cmd[0] for cmd, _ in self.commands.calls] == ["find"]    # the check for changes in BASEDIR/.git, nothing else

  def test_same_process_rebuilds_when_the_install_changed(self, mocker):
    self._device(mocker)
    updated.init_overlay()
    real = self.commands.__call__
    mocker.patch.object(updated, "run", lambda cmd, cwd=None: "changed-file\n" if cmd[0] == "find" else real(cmd, cwd))

    updated.init_overlay()

    assert self._rebuilds() == 2

  def test_staging_area_whose_build_failed_is_not_reused(self, mocker):
    commands = self._device(mocker)
    commands.fail_on("git", "diff", times=1)
    with self.assertRaises(subprocess.CalledProcessError):
      updated.init_overlay()
    assert not updated.ProcessState.overlay_built
    assert self.dirs.overlay_init.is_file() and os.path.ismount(str(self.dirs.merged))   # it looks valid from outside

    updated.init_overlay()

    assert self._rebuilds() == 2

  def test_marked_staging_area_is_rebuilt(self, mocker):
    self._device(mocker)
    updated.init_overlay()
    updated.mark_overlay_for_rebuild()                               # what a forced kill does (test_stop.py)
    assert not self.dirs.overlay_init.exists()

    updated.init_overlay()

    assert self._rebuilds() == 2

  def test_each_updater_process_builds_its_own_staging_area(self, mocker):
    self._device(mocker)
    assert run_main(self, mocker, 3) == [FIRST_WAIT, AFTER_SUCCESS, AFTER_SUCCESS]
    assert self._rebuilds() == 1                                     # one process: built once, then reused

    self.commands.calls.clear()
    reset_process_state(self)                                        # the updater is stopped and started again (a drive)
    assert run_main(self, mocker, 3) == [FIRST_WAIT, AFTER_SUCCESS, AFTER_SUCCESS]
    assert self._rebuilds() == 1                                     # the new process: built once more


def overlay_possible(root) -> str | None:
  """None if this machine can run the updater's sudo and overlay mount in root, else the reason why not"""
  if os.geteuid() != 0 or shutil.which("sudo") is None:
    return "needs root and sudo"
  parts = [root / "probe" / name for name in ("lower", "upper", "work", "merged")]
  for part in parts:
    part.mkdir(parents=True)
  mount = subprocess.run(["sudo", "-n", "mount", "-t", "overlay", "-o", f"lowerdir={parts[0]},upperdir={parts[1]},workdir={parts[2]}", "none", str(parts[3])],
                         capture_output=True)
  if mount.returncode != 0:
    return "overlay filesystems cannot be mounted here"
  subprocess.run(["sudo", "-n", "umount", "-l", str(parts[3])], check=True)
  return None


class TestRealOverlay(OpenpilotTestCase):
  """updated.main() with nothing replaced but its folders and its wait: real sudo, real overlay mount, real git
  against local repositories whose history contains the unfetchable submodule pointer."""

  def setup_method(self):
    os.environ.update(GIT_ENV)
    reset_process_state(self)
    self.root = tmp_root(self)
    reason = overlay_possible(self.root)
    if reason is not None:
      self.skipTest(reason)
    self.remotes = Remotes(self.root)
    self.device = self.remotes.install(self.root / "openpilot")
    self.installed = git(self.device, "rev-parse", "HEAD")
    self.remotes.commit_main(self.remotes.missing_sub_commit())
    self.sub_head = self.remotes.commit_sub()
    self.head = self.remotes.commit_main(self.sub_head)
    self.params = Params()

  def _dirs(self, mocker):
    self.dirs = use_dirs(mocker, self.root, basedir=self.device)
    self.addCleanup(subprocess.run, ["sudo", "-n", "umount", "-l", str(self.dirs.merged)], capture_output=True)
    return self.dirs

  def _assert_update_is_ready(self):
    assert self.params.get("UpdateFailedCount") == 0
    assert self.params.get_bool("UpdateAvailable") and self.params.get("UpdaterLastFetchTime") is not None
    assert self.params.get("UpdaterNewDescription") == f"0.0.1 / {BRANCH} / {self.head[:7]} / " + self.params.get("UpdaterNewDescription").rsplit(" / ", 1)[1]
    assert (self.dirs.finalized / ".overlay_consistent").is_file()
    assert git(self.dirs.finalized, "rev-parse", "HEAD") == self.head
    assert git(self.dirs.finalized / "sub", "rev-parse", "HEAD") == self.sub_head
    assert git(self.dirs.finalized, "status", "--porcelain") == ""
    assert git(self.device, "rev-parse", "HEAD") == self.installed   # the install itself is untouched

  def test_check_download_finalize(self, mocker):
    self._dirs(mocker)
    assert run_main(self, mocker, 2) == [FIRST_WAIT, AFTER_SUCCESS]
    self._assert_update_is_ready()
    assert os.path.ismount(str(self.dirs.merged))
    assert self.params.get("UpdaterLastFailure") is None

  def test_new_process_is_not_stopped_by_a_lock_left_in_the_staging_area(self, mocker):
    self._dirs(mocker)
    assert run_main(self, mocker, 1) == [FIRST_WAIT]                 # a first process builds the staging area ...
    (self.dirs.merged / ".git/index.lock").touch()                  # ... and is killed in the middle of a git command
    assert locks(self.dirs.merged) == [".git/index.lock"]

    reset_process_state(self)                                        # the next process
    assert run_main(self, mocker, 2) == [FIRST_WAIT, AFTER_SUCCESS]
    self._assert_update_is_ready()
    assert locks(self.dirs.merged) == []
    assert self.params.get("UpdaterLastFailure") is None

  def test_stop_during_git_in_the_staging_area(self, mocker):
    # the new head has a file that git writes through a filter; on this device the filter never finishes, so
    # `git checkout` of the download stays in the staging area with .git/index.lock held (as with a slow LFS filter)
    pids = self.root / "pids"
    self.head = self.remotes.commit_main(self.sub_head, files={".gitattributes": "*.dat filter=slow\n", "big.dat": "content\n"})
    git(self.device, "config", "filter.slow.smudge", f"sh -c 'sleep 60 & echo $$ $! > {pids}; wait'")
    self._dirs(mocker)

    process = subprocess.Popen([sys.executable, "-c", UPDATER_PROCESS, str(self.root), str(self.device)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    self.addCleanup(process.wait)
    self.addCleanup(lambda: process.poll() is None and process.kill())
    end = time.monotonic() + 60
    while not (pids.exists() and len(pids.read_text().split()) == 2 and locks(self.dirs.merged)):
      assert process.poll() is None and time.monotonic() < end, "the updater did not reach the checkout"
      time.sleep(0.02)
    assert locks(self.dirs.merged) == [".git/index.lock"]
    filter_pids = [int(pid) for pid in pids.read_text().split()]

    t = time.monotonic()
    os.kill(process.pid, signal.SIGINT)                              # the manager's stop
    process.wait(timeout=MANAGER_KILLS_AFTER)
    seconds = time.monotonic() - t

    assert process.returncode == 42 and seconds < updated.STOP_TERM_WAIT, (process.returncode, seconds)
    assert locks(self.dirs.merged) == []
    assert not any(running(pid) for pid in filter_pids)
    assert self.dirs.overlay_init.is_file()                          # nothing had to be killed

    # the next updater process (the filter works again) completes the update
    git(self.device, "config", "--unset", "filter.slow.smudge")
    reset_process_state(self)
    assert run_main(self, mocker, 2) == [FIRST_WAIT, AFTER_SUCCESS]
    self._assert_update_is_ready()
    assert (self.dirs.finalized / "big.dat").read_text() == "content\n"
