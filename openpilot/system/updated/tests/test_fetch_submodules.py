"""The download fetches only the branch it was asked for.

A branch whose history holds a commit with a submodule pointer that cannot be fetched (the id names no commit), and
a fetchable pointer at its head: `git fetch origin <branch>`, the updater's command until now, fails on it, because git
also fetches what the submodule pointers of every new commit name. This is the failed update of 10 Nov 2026."""
import os
import subprocess

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.updated import updated
from openpilot.system.updated.tests.helpers import BRANCH, GIT_ENV, Remotes, git, reset_process_state, tmp_root, use_dirs


class TestFetchSubmodules(OpenpilotTestCase):
  def setup_method(self):
    os.environ.update(GIT_ENV)
    reset_process_state(self)
    self.root = tmp_root(self)
    self.remotes = Remotes(self.root)
    self.device = self.remotes.install(self.root / "device")                   # installed before the bad commit is pushed
    self.installed = git(self.device, "rev-parse", "HEAD")
    self.unfetchable = self.remotes.missing_sub_commit()
    self.remotes.commit_main(self.unfetchable)                                 # a commit in the history points to it
    self.sub_head = self.remotes.commit_sub()
    self.head = self.remotes.commit_main(self.sub_head)                        # the head points to a pushed commit

  def _staging_is_device(self, mocker):
    # no overlay here: the git steps run in the install itself, which is what the overlay shows them
    dirs = use_dirs(mocker, self.root, basedir=self.device)
    mocker.patch.object(updated, "OVERLAY_MERGED", str(self.device))
    return dirs

  def test_command_without_the_flag_fails(self):
    with self.assertRaises(subprocess.CalledProcessError) as failed:
      updated.run(["git", "fetch", "origin", BRANCH], str(self.device))
    assert f"not our ref {self.unfetchable}" in failed.exception.output, failed.exception.output
    assert "Errors during submodule fetch" in failed.exception.output, failed.exception.output

    # and this is what the updater keeps of it
    record = updated.failure_record("download", failed.exception)
    assert record["reason"].endswith(f"upload-pack: not our ref {self.unfetchable}"), record   # the first "fatal:" line
    assert record["cmd"] == f"git fetch origin {BRANCH}" and record["rc"] == 1, record

  def test_fetch_update_passes_and_ends_on_the_heads_pointers(self, mocker):
    dirs = self._staging_is_device(mocker)
    spy = mocker.patch.object(updated, "run", wraps=updated.run)

    updated.Updater().fetch_update()

    fetches = [c.args[0] for c in spy.call_args_list if c.args[0][:2] == ["git", "fetch"]]
    assert fetches == [["git", "fetch", "--no-recurse-submodules", "origin", BRANCH]], fetches
    assert git(self.device, "rev-parse", "HEAD") == self.head
    assert git(self.device / "sub", "rev-parse", "HEAD") == self.sub_head
    assert git(self.device, "status", "--porcelain") == ""
    # the finalized copy is complete and marked ready
    assert git(dirs.finalized, "rev-parse", "HEAD") == self.head
    assert git(dirs.finalized / "sub", "rev-parse", "HEAD") == self.sub_head
    assert (dirs.finalized / ".overlay_consistent").is_file()
    assert Params().get("UpdaterLastFailure") is None

  def test_same_steps_with_the_old_fetch_fail(self, mocker):
    # control: the only difference to the test above is the fetch command
    self._staging_is_device(mocker)
    real_run = updated.run

    def run_with_old_fetch(cmd, cwd=None):
      return real_run([arg for arg in cmd if arg != "--no-recurse-submodules"], cwd)
    mocker.patch.object(updated, "run", run_with_old_fetch)

    with self.assertRaises(subprocess.CalledProcessError) as failed:
      updated.Updater().fetch_update()
    assert failed.exception.cmd == ["git", "fetch", "origin", BRANCH]
    assert git(self.device, "rev-parse", "HEAD") == self.installed
