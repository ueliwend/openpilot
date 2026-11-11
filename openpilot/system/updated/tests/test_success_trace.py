"""A successful check and download does what it did before the updater fix, command for command and param for param.

success_trace_0057d61.json is the record of updated.py as it was before the fix (0057d61) in the scenario below:
every command it ran and every param it wrote or removed, in order. The updater of this tree must produce the same
record, except for the one flag added to `git fetch`.

Scenario: a device on an old commit boots; the updater starts, builds its staging area and waits its first minute;
checks, downloads and finalizes the update; after its 1.5 hours the user presses CHECK; the updater checks again.

To record the scenario with any updated.py that is importable as openpilot.system.updated.updated:
  python3 -m openpilot.system.updated.tests.test_success_trace > trace.json"""
import json
import sys
import unittest
from pathlib import Path

from openpilot.common.test import OpenpilotTestCase
from openpilot.system.updated import updated
from openpilot.system.updated.tests.helpers import FakeCommands, Recorder, reset_process_state, run_main, tmp_root, use_dirs

GOLDEN = Path(__file__).with_name("success_trace_0057d61.json")
OLD_FETCH = ["git", "fetch", "origin", "inster"]
NEW_FETCH = ["git", "fetch", "--no-recurse-submodules", "origin", "inster"]


def record_success_trace(test, mocker) -> tuple[list, list]:
  reset_process_state(test)
  root = tmp_root(test)
  dirs = use_dirs(mocker, root)
  dirs.basedir.mkdir()
  commands = FakeCommands(dirs).install(mocker)
  recorder = Recorder(root).install(mocker, commands)

  def press_check_in_second_wait(n, wait_helper):
    if n == 2:
      wait_helper.user_request = updated.UserRequest.CHECK
  waits = run_main(test, mocker, 3, during_wait=press_check_in_second_wait)
  return recorder.trace, waits


class TestSuccessTrace(OpenpilotTestCase):
  def test_same_commands_and_params_as_before_the_fix(self, mocker):
    trace, waits = record_success_trace(self, mocker)
    assert waits == [60, 1.5 * 60 * 60, 1.5 * 60 * 60]

    with open(GOLDEN) as f:
      before = json.load(f)
    assert sum(entry == ["run", OLD_FETCH, "<root>/safe_staging/merged"] for entry in before) == 1
    expected = [["run", NEW_FETCH, entry[2]] if entry[:2] == ["run", OLD_FETCH] else entry for entry in before]
    assert json.loads(json.dumps(trace)) == expected
    assert not any("UpdaterLastFailure" in entry for entry in trace)


if __name__ == "__main__":
  class Record(OpenpilotTestCase):
    def test_record(self, mocker):
      trace, _ = record_success_trace(self, mocker)
      sys.stdout.write("[\n" + ",\n".join(json.dumps(entry) for entry in trace) + "\n]\n")

  result = unittest.TextTestRunner(stream=sys.stderr).run(unittest.defaultTestLoader.loadTestsFromTestCase(Record))
  sys.exit(not result.wasSuccessful())
