"""What the updater does with a failure: it keeps it (param UpdaterLastFailure), counts it, and after a failed
download tries exactly once more at once. updated.main() runs here with its commands answered by a stand-in."""
import datetime
import json
import os
import re
import signal
import subprocess
import time

from openpilot.common import params as params_module
from openpilot.common.params import Params, ParamKeyFlag
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.updated import updated
from openpilot.system.updated.tests.helpers import FakeCommands, reset_process_state, run_main, tmp_root, use_dirs

FIRST_WAIT, AFTER_FAILURE, AFTER_SUCCESS = 60, 5 * 60, 1.5 * 60 * 60


def manager_start(params: Params) -> None:
  """what system/manager/manager.py manager_init() clears at every boot"""
  for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
               ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON):
    params.clear_all(flag)


def stored_size() -> int:
  return os.path.getsize(Params().get_param_path("UpdaterLastFailure"))


class UpdaterLoopCase(OpenpilotTestCase):
  def setup_method(self):
    reset_process_state(self)
    self.params = Params()

  def new_process(self, mocker) -> FakeCommands:
    """a new updated process on a new, empty set of folders"""
    reset_process_state(self)
    self.dirs = use_dirs(mocker, tmp_root(self))
    self.dirs.basedir.mkdir()
    self.commands = FakeCommands(self.dirs).install(mocker)
    return self.commands


class TestFailureRecord(OpenpilotTestCase):
  def test_param_is_persistent_and_not_in_backups(self):
    header = os.path.join(os.path.dirname(params_module.__file__), "params_keys.h")
    with open(header) as f:
      flags = re.findall(r'\{"UpdaterLastFailure", \{([^,]*), (\w+)\}\}', f.read())
    assert flags == [("PERSISTENT", "JSON")], flags                 # no CLEAR_ON_*, no BACKUP, logged (no DONT_LOG)

    record = {"stage": "check", "reason": "kept"}
    Params().put("UpdaterLastFailure", record, block=True)
    manager_start(Params())
    assert Params().get("UpdaterLastFailure") == record

  def test_content_of_a_failed_command(self):
    output = "remote: Enumerating objects: 5, done.\nfatal: remote error: upload-pack: not our ref 1433c14\nfatal: the remote end hung up\n"
    e = subprocess.CalledProcessError(128, ["git", "fetch", "--no-recurse-submodules", "origin", "inster"], output=output)
    record = updated.failure_record("download", e)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", record.pop("time"))
    assert record == {
      "stage": "download", "cmd": "git fetch --no-recurse-submodules origin inster", "rc": 128,
      "reason": "remote error: upload-pack: not our ref 1433c14", "output": output.strip(),
    }

  def test_content_of_another_exception(self):
    record = updated.failure_record("staging", RuntimeError("base and overlay merge directories are on different filesystems"))
    assert (record["stage"], record["cmd"], record["rc"], record["output"]) == ("staging", "", None, "")
    assert record["reason"] == "RuntimeError: base and overlay merge directories are on different filesystems"

  def test_time_is_utc_whatever_the_devices_time_zone(self):
    before = os.environ.get("TZ")
    os.environ["TZ"] = "UPD-9"                                       # a zone nine hours ahead of UTC
    time.tzset()
    try:
      record = updated.failure_record("check", RuntimeError("x"))
    finally:
      os.environ.pop("TZ")
      if before is not None:
        os.environ["TZ"] = before
      time.tzset()
    recorded = datetime.datetime.strptime(record["time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.UTC)
    assert abs((datetime.datetime.now(datetime.UTC) - recorded).total_seconds()) < 60, record["time"]

  def test_reason(self):
    assert updated.failure_reason("warning: x\nerror: RPC failed; curl 56\nfatal: early EOF\n") == "RPC failed; curl 56"
    assert updated.failure_reason("fatal: unable to access 'https://host/repo/': Could not resolve host: host\n") == \
           "unable to access 'https://host/repo/': Could not resolve host: host"
    assert updated.failure_reason("first line\n\nlast line\n\n") == "last line"
    assert updated.failure_reason("") == ""
    e = subprocess.CalledProcessError(7, ["sudo", "mount"], output="")
    assert updated.failure_record("staging", e)["reason"] == "exit code 7"

  def test_stored_record_is_capped(self):
    lines = [f'line {i}: "quoted" \\ back\tslash caf\u00e9 \u2713 ' + "x" * 70 for i in range(3000)]
    e = subprocess.CalledProcessError(1, ["git", "fetch"] + ["argument"] * 200, output="\n".join(lines))
    first = updated.fit_failure_record(updated.failure_record("download", e))
    for record in (first, updated.fit_failure_record(updated.with_retry_outcome(first, updated.failure_record("finalize", e)))):
      assert len(json.dumps(record)) <= updated.FAILURE_RECORD_MAX_BYTES
      assert record["output"].endswith(lines[-1])                   # the last lines are what is kept
      assert record["output"].startswith("line ")                   # from the start of a line
      assert len(record["cmd"]) == 200 and record["reason"] == lines[-1][:160]
      Params().put("UpdaterLastFailure", record, block=True)
      assert stored_size() <= updated.FAILURE_RECORD_MAX_BYTES
      assert Params().get("UpdaterLastFailure") == record
    assert len(json.dumps(first)) > updated.FAILURE_RECORD_MAX_BYTES - 100   # and the room is used

  def test_cap_holds_without_any_output(self):
    e = RuntimeError("\u2713" * 5000)
    record = updated.fit_failure_record(updated.with_retry_outcome(updated.failure_record("check", e), updated.failure_record("check", e)))
    assert len(json.dumps(record)) <= updated.FAILURE_RECORD_MAX_BYTES

  def test_credentials_are_stripped(self):
    secrets = ["s3cretpassword", "ghp_" + "A1b2" * 9, "github_pat_" + "Zz9_" * 10, "eyJhbGciOiJIUzI1NiJ9.eyJpZGVudGl0eSI6IngifQ.c2lnbmF0dXJlMTIzNDU2"]
    output = "\n".join([
      f"fatal: unable to access 'https://someone:{secrets[0]}@example.com/owner/repo.git/': The requested URL returned error: 403",
      f"remote: https://x-access-token:{secrets[1]}@github.com/owner/repo",
      f"error: token {secrets[2]} was refused",
      f"> Authorization: Bearer {secrets[3]}",
      f"> authorization: Basic {secrets[0]}",
    ])
    cmd = ["git", "fetch", f"https://oauth2:{secrets[1]}@example.com/owner/repo.git", "inster"]
    for e in (subprocess.CalledProcessError(128, cmd, output=output), RuntimeError(output)):
      first = updated.failure_record("download", e)
      for record in (first, updated.with_retry_outcome(first, first)):
        text = json.dumps(record)
        for secret in secrets:
          assert secret not in text, (secret, text)
    record = updated.failure_record("download", subprocess.CalledProcessError(128, cmd, output=output))
    assert record["cmd"] == "git fetch https://example.com/owner/repo.git inster"
    assert record["reason"] == "unable to access 'https://example.com/owner/repo.git/': The requested URL returned error: 403"
    # an address without credentials is left alone
    assert updated.strip_secrets("git@github.com:owner/repo.git https://github.com/owner/repo") == "git@github.com:owner/repo.git https://github.com/owner/repo"


class TestFailureIsKept(UpdaterLoopCase):
  def test_kept_across_restart_and_success_replaced_by_next_failure(self, mocker):
    commands = self.new_process(mocker)
    commands.fail_on("git", "ls-remote", "--heads", output="fatal: unable to access 'https://example.com/repo/': Could not resolve host: example.com\n")
    assert run_main(self, mocker, 2) == [FIRST_WAIT, AFTER_FAILURE]
    record = self.params.get("UpdaterLastFailure")
    assert (record["stage"], record["cmd"], record["rc"]) == ("check", "git ls-remote --heads", 1), record
    assert record["reason"] == "unable to access 'https://example.com/repo/': Could not resolve host: example.com"
    assert "retry" not in record
    assert self.params.get("UpdateFailedCount") == 1

    # reboot: the count and the old exception text go, the record stays
    manager_start(self.params)
    assert self.params.get("UpdateFailedCount") is None and self.params.get("LastUpdateException") is None
    assert self.params.get("UpdaterLastFailure") == record

    # a new updater process whose attempts all succeed does not touch it
    self.new_process(mocker)
    assert run_main(self, mocker, 3) == [FIRST_WAIT, AFTER_SUCCESS, AFTER_SUCCESS]
    assert self.params.get("UpdateFailedCount") == 0 and self.params.get("LastUpdateTime") is not None
    assert self.params.get("UpdaterLastFailure") == record

    # the next failure replaces it
    commands = self.new_process(mocker)
    commands.fail_on("git", "diff", output="error: cannot read the index\n")
    run_main(self, mocker, 1)
    replaced = self.params.get("UpdaterLastFailure")
    assert (replaced["stage"], replaced["cmd"], replaced["reason"]) == ("staging", "git diff --submodule=diff", "cannot read the index")


class TestStageOfTheRecord(UpdaterLoopCase):
  def test_failure_in_finalize(self, mocker):
    commands = self.new_process(mocker)
    commands.fail_on("git", "reset", "--hard", cwd=self.dirs.finalized, output="fatal: finalize broke\n")
    assert run_main(self, mocker, 2) == [FIRST_WAIT, AFTER_FAILURE]
    record = self.params.get("UpdaterLastFailure")
    assert (record["stage"], record["cmd"], record["reason"]) == ("finalize", "git reset --hard", "finalize broke")
    assert (record["retry"], record["retry_stage"]) == ("failed", "finalize")
    assert len(commands.commands("git", "fetch")) == 2               # the download itself went through, twice

  def test_staging_failure_in_the_loop_after_a_download_attempt(self, mocker):
    commands = self.new_process(mocker)
    commands.fail_on("git", "fetch", times=2)                        # one attempt: download and its retry fail
    commands.fail_on("git", "diff", after=2, times=1)                # the next attempt cannot build its staging area
    seen = []
    waits = run_main(self, mocker, 4, during_wait=lambda n, wait_helper: seen.append(self.params.get("UpdaterLastFailure")))
    assert waits == [FIRST_WAIT, AFTER_FAILURE, AFTER_FAILURE, AFTER_SUCCESS]
    assert (seen[1]["stage"], seen[1]["retry_stage"]) == ("download", "download")
    assert (seen[2]["stage"], seen[2]["cmd"]) == ("staging", "git diff --submodule=diff") and "retry" not in seen[2]

  def test_failed_check_in_the_loop_after_a_download(self, mocker):
    commands = self.new_process(mocker)
    commands.fail_on("git", "ls-remote", "--heads", after=1, times=1)
    assert run_main(self, mocker, 3) == [FIRST_WAIT, AFTER_SUCCESS, AFTER_FAILURE]
    assert self.params.get("UpdaterLastFailure")["stage"] == "check"


class TestRecordThatCannotBeWritten(UpdaterLoopCase):
  """the record never changes what the updater does"""

  def _refuse_the_record(self, mocker):
    put = Params.put

    def refuse(params, key, *args, **kwargs):
      if key in ("UpdaterLastFailure", b"UpdaterLastFailure"):
        raise OSError(28, "No space left on device")
      return put(params, key, *args, **kwargs)
    mocker.patch.object(Params, "put", refuse)

  def test_successful_retry_is_still_a_success(self, mocker):
    commands = self.new_process(mocker)
    commands.fail_on("git", "fetch", times=1)
    self._refuse_the_record(mocker)
    assert run_main(self, mocker, 2) == [FIRST_WAIT, AFTER_SUCCESS]
    assert self.params.get("UpdateFailedCount") == 0 and self.params.get("LastUpdateTime") is not None
    assert self.params.get_bool("UpdateAvailable") and self.params.get("UpdaterLastFetchTime") is not None
    assert (self.dirs.finalized / ".overlay_consistent").is_file() and self.dirs.overlay_init.is_file()
    assert len(commands.commands("git", "fetch")) == 2
    assert self.params.get("UpdaterLastFailure") is None

  def test_failure_is_still_counted_and_retried_once(self, mocker):
    commands = self.new_process(mocker)
    commands.fail_on("git", "fetch")
    self._refuse_the_record(mocker)
    assert run_main(self, mocker, 2) == [FIRST_WAIT, AFTER_FAILURE]
    assert self.params.get("UpdateFailedCount") == 1 and self.params.get("LastUpdateTime") is None
    assert len(commands.commands("git", "fetch")) == 2


class TestEveryFailureCounts(UpdaterLoopCase):
  def _assert_counted_as_failure(self, count: int):
    assert self.params.get("UpdateFailedCount") == count
    assert self.params.get("LastUpdateTime") is None                # not reported as a successful check
    assert self.params.get("LastUpdateException") is not None

  def test_failure_in_init_overlay(self, mocker):
    commands = self.new_process(mocker)
    commands.fail_on("git", "diff")
    waits = run_main(self, mocker, 1)
    assert waits == [AFTER_FAILURE]                                  # not the 1.5 hours after a success
    self._assert_counted_as_failure(1)
    assert self.params.get("UpdaterLastFailure")["stage"] == "staging"

  def test_failures_in_init_overlay_go_on_counting(self, mocker):
    commands = self.new_process(mocker)
    commands.fail_on("git", "diff")
    assert run_main(self, mocker, 3) == [AFTER_FAILURE] * 3
    self._assert_counted_as_failure(3)

  def test_failure_in_init_overlay_that_is_not_a_command(self, mocker):
    self.new_process(mocker)
    mocker.patch.object(updated, "init_overlay", side_effect=RuntimeError("base and overlay merge directories are on different filesystems"))
    assert run_main(self, mocker, 1) == [AFTER_FAILURE]
    self._assert_counted_as_failure(1)
    assert self.params.get("UpdaterLastFailure")["reason"].startswith("RuntimeError: base and overlay")

  def test_failure_in_first_set_params(self, mocker):
    commands = self.new_process(mocker)
    commands.fail_on("git", "rev-parse", "--abbrev-ref", times=1)    # the first thing set_params asks git
    assert run_main(self, mocker, 1) == [AFTER_FAILURE]
    self._assert_counted_as_failure(1)
    assert self.params.get("UpdaterLastFailure")["stage"] == "check"

  def test_failure_after_counting_is_counted_once_and_success_resets(self, mocker):
    counts = []
    commands = self.new_process(mocker)
    commands.fail_on("git", "ls-remote", "--heads", times=2)
    waits = run_main(self, mocker, 4, during_wait=lambda n, wait_helper: counts.append(self.params.get("UpdateFailedCount")))
    assert waits == [FIRST_WAIT, AFTER_FAILURE, AFTER_FAILURE, AFTER_SUCCESS]
    assert counts == [0, 1, 2]                                       # after the first run and after each failed check
    assert self.params.get("UpdateFailedCount") == 0 and self.params.get("LastUpdateTime") is not None


class TestRetryOnce(UpdaterLoopCase):
  def _steps(self) -> list[str]:
    """fetches and staging builds in the order they happened"""
    names = {("git", "fetch"): "fetch", ("sudo", "mount"): "build staging"}
    return [names[tuple(cmd[:2])] for cmd, _ in self.commands.calls if tuple(cmd[:2]) in names]

  def test_failed_download_is_retried_once_on_a_rebuilt_staging_area(self, mocker):
    commands = self.new_process(mocker)
    commands.fail_on("git", "fetch", times=1, output="fatal: unable to access 'https://example.com/repo/': Failed to connect\n")
    assert run_main(self, mocker, 2) == [FIRST_WAIT, AFTER_SUCCESS]
    assert self._steps() == ["build staging", "fetch", "build staging", "fetch"]
    assert self.params.get("UpdateFailedCount") == 0
    assert self.params.get("UpdaterLastFetchTime") is not None and self.params.get_bool("UpdateAvailable")
    assert (self.dirs.finalized / ".overlay_consistent").is_file()

    record = self.params.get("UpdaterLastFailure")                   # the first failure is kept, with the retry's outcome
    assert (record["stage"], record["cmd"], record["retry"]) == ("download", "git fetch --no-recurse-submodules origin inster", "ok")
    assert record["reason"] == "unable to access 'https://example.com/repo/': Failed to connect"

  def test_second_failure_is_final_until_the_five_minute_wait(self, mocker):
    commands = self.new_process(mocker)
    commands.fail_on("git", "fetch")
    seen = []
    waits = run_main(self, mocker, 4, during_wait=lambda n, wait_helper: seen.append((self._steps(), self.params.get("UpdateFailedCount"))))
    assert waits == [FIRST_WAIT, AFTER_FAILURE, AFTER_FAILURE, AFTER_FAILURE]
    one_attempt = ["build staging", "fetch", "build staging", "fetch"]                # two tries, never a third
    assert seen == [(["build staging"], 0), (one_attempt, 1), (one_attempt * 2, 2)]
    assert self._steps() == one_attempt * 3 and self.params.get("UpdateFailedCount") == 3
    assert self.params.get("LastUpdateTime") is None and self.params.get("UpdaterLastFetchTime") is None

  def test_first_failure_is_the_one_kept_when_the_retry_fails_too(self, mocker):
    commands = self.new_process(mocker)
    commands.fail_on("git", "fetch", times=1, output="fatal: first\n")
    commands.fail_on("git", "diff", after=1, output="fatal: second\n")                # rebuilding the staging area fails
    assert run_main(self, mocker, 2) == [FIRST_WAIT, AFTER_FAILURE]
    record = self.params.get("UpdaterLastFailure")
    assert (record["stage"], record["cmd"], record["reason"]) == ("download", "git fetch --no-recurse-submodules origin inster", "first")
    assert (record["retry"], record["retry_stage"], record["retry_cmd"], record["retry_reason"]) == \
           ("failed", "staging", "git diff --submodule=diff", "second")
    assert record["output"] == "fatal: first"
    assert self.params.get("UpdateFailedCount") == 1

  def test_failed_check_is_not_retried(self, mocker):
    commands = self.new_process(mocker)
    commands.fail_on("git", "ls-remote", "--heads")
    assert run_main(self, mocker, 3) == [FIRST_WAIT, AFTER_FAILURE, AFTER_FAILURE]
    assert len(commands.commands("git", "ls-remote", "--heads")) == 2
    assert commands.commands("git", "fetch") == []

  def test_first_failure_is_kept_when_the_updater_is_stopped_during_the_retry(self, mocker):
    commands = self.new_process(mocker)
    commands.fail_on("git", "fetch", times=1)

    def stopped_in_second_fetch(cmd, cwd=None):
      if cmd[:2] == ["git", "fetch"] and len(commands.commands("git", "fetch")) == 1:
        raise KeyboardInterrupt
      return commands(cmd, cwd)
    mocker.patch.object(updated, "run", stopped_in_second_fetch)
    mocker.patch.object(updated.WaitTimeHelper, "sleep", lambda self, t: None)
    mocker.patch.object(updated, "system_time_valid", lambda: True)
    with self.assertRaises(KeyboardInterrupt):
      updated.main()
    record = self.params.get("UpdaterLastFailure")
    assert record["stage"] == "download" and "retry" not in record


class TestMainInstallsTheStopHandler(UpdaterLoopCase):
  def test_handler(self, mocker):
    self.new_process(mocker)
    run_main(self, mocker, 1)
    assert signal.getsignal(signal.SIGINT) is updated.request_stop
