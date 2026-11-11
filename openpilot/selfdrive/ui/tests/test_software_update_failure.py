"""Software screen, Download row: what it says after a failed update attempt, and that it says what it always said
in every other state.

The layout is loaded on its own with small stand-ins for the widgets and for ui_state, so the test needs neither
a window nor raylib and never touches the real modules."""
import datetime
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

from openpilot.common.test import OpenpilotTestCase

SOFTWARE_PY = Path(__file__).resolve().parents[1] / "layouts/settings/software.py"

RECORD = {
  "time": "2026-11-10T18:38:02Z", "stage": "download", "cmd": "git fetch origin inster", "rc": 1,
  "reason": "remote error: upload-pack: not our ref 1433c1415d",
  "output": "Fetching submodule opendbc_repo\nfatal: remote error: upload-pack: not our ref 1433c1415d\nErrors during submodule fetch:\n\topendbc_repo",
}


class Action:
  def __init__(self, text):
    self.text, self.value, self.enabled = text(), "", True

  def set_text(self, text):
    self.text = text

  def set_value(self, value):
    self.value = value

  def set_enabled(self, enabled):
    self.enabled = enabled


class Item:
  """the parts of system/ui/widgets/list_view.py ListItem that the layout uses, with the same rules for the description"""

  def __init__(self, title=None, text=None, description=None, callback=None):
    self.action_item = Action(text) if callable(text) else Action(lambda: text)
    self.visible, self.description, self.description_visible = True, description or "", False

  def set_visible(self, visible):
    self.visible = visible

  def set_description(self, description):
    self.description = description or ""

  def _set_description_visible(self, visible):
    if self.description and self.description_visible != visible:
      self.description_visible = visible

  def tap(self):
    self._set_description_visible(not self.description_visible)


class FakeParams(dict):
  def get_bool(self, key):
    return bool(self.get(key))


def load_layout():
  """software.py as a module of its own, with stand-ins for everything it imports from the UI"""
  ui_state = types.SimpleNamespace(params=FakeParams(), is_onroad=lambda: False, is_offroad=lambda: True)
  stand_ins = {
    "openpilot.selfdrive.ui.ui_state": {"ui_state": ui_state},
    "openpilot.system.ui.lib.application": {"gui_app": types.SimpleNamespace(sunnypilot_ui=lambda: False, push_widget=lambda w: None)},
    "openpilot.system.ui.lib.multilang": {"tr": lambda s: s, "trn": lambda one, many, n: one if n == 1 else many},
    "openpilot.system.ui.widgets": {"Widget": type("Widget", (), {"__init__": lambda self: None}), "DialogResult": types.SimpleNamespace(CONFIRM=1)},
    "openpilot.system.ui.widgets.confirm_dialog": {"ConfirmDialog": object},
    "openpilot.system.ui.widgets.list_view": {"button_item": Item, "text_item": Item, "ListItem": Item},
    "openpilot.system.ui.widgets.option_dialog": {"MultiOptionDialog": object},
    "openpilot.system.ui.widgets.scroller_tici": {"Scroller": lambda items, **kwargs: types.SimpleNamespace(items=items)},
  }
  modules = {}
  for name, attributes in stand_ins.items():
    modules[name] = types.ModuleType(name)
    vars(modules[name]).update(attributes)
  spec = importlib.util.spec_from_file_location("software_layout_under_test", SOFTWARE_PY)
  module = importlib.util.module_from_spec(spec)
  with mock.patch.dict(sys.modules, modules):
    spec.loader.exec_module(module)
  return module, ui_state.params


class TestSoftwareUpdateFailure(OpenpilotTestCase):
  def setup_method(self):
    self.software, self.params = load_layout()
    self.layout = self.software.SoftwareLayout()
    self.row = self.layout._download_btn

  def _show(self, **params):
    self.params.clear()
    self.params.update(params)
    self.layout._update_state()
    return self.row.action_item.value, self.row.action_item.text, self.row.action_item.enabled, self.row.description

  def test_other_states_are_unchanged(self):
    hour_ago = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - datetime.timedelta(hours=1, minutes=5)
    assert self._show() == ("up to date, last checked never", "CHECK", True, "")
    assert self._show(UpdaterState="idle", UpdateFailedCount=0, LastUpdateTime=hour_ago) == ("up to date, last checked 1 hour ago", "CHECK", True, "")
    assert self._show(UpdaterState="idle", UpdaterFetchAvailable=True) == ("update available", "DOWNLOAD", True, "")
    assert self._show(UpdaterState="checking...") == ("checking...", "DOWNLOAD", False, "")
    assert self._show(UpdaterState="downloading...", UpdaterFetchAvailable=True) == ("downloading...", "DOWNLOAD", False, "")
    assert self._show(UpdaterState="finalizing update...", UpdaterFetchAvailable=True) == ("finalizing update...", "DOWNLOAD", False, "")
    # a record of an old failure is stored, but the last attempt did not fail: nothing of it is shown
    assert self._show(UpdaterState="idle", UpdaterFetchAvailable=True, UpdaterLastFailure=RECORD) == ("update available", "DOWNLOAD", True, "")
    assert self._show(UpdaterState="idle", UpdateFailedCount=0, UpdaterLastFailure=RECORD) == ("up to date, last checked never", "CHECK", True, "")
    self.row.tap()
    assert not self.row.description_visible                          # and there is nothing to open

  def test_failure_without_a_record_reads_as_before(self):
    assert self._show(UpdaterState="idle", UpdateFailedCount=1) == ("failed to check for update", "CHECK", True, "")
    assert self._show(UpdaterState="idle", UpdateFailedCount=3, UpdaterLastFailure="not a record") == ("failed to check for update", "CHECK", True, "")

  def test_failed_download(self):
    value, button, enabled, details = self._show(UpdaterState="idle", UpdateFailedCount=1, UpdaterFetchAvailable=True, UpdaterLastFailure=RECORD)
    assert (value, button, enabled) == ("download failed: remote error: upload-pack: not our ref 1433c1415d", "CHECK", True)
    assert details == "<br>".join([
      "download failed (download), 2026-11-10 18:38:02 UTC",
      "git fetch origin inster (exit code 1)",
      "Fetching submodule opendbc_repo",
      "fatal: remote error: upload-pack: not our ref 1433c1415d",
      "Errors during submodule fetch:",
      "\topendbc_repo",
    ])
    self.row.tap()
    assert self.row.description_visible                              # the stored text opens on the row

  def test_stage_decides_between_check_and_download(self):
    for stage, text in (("staging", "check failed"), ("check", "check failed"), ("download", "download failed"), ("finalize", "download failed")):
      value, _, _, details = self._show(UpdaterState="idle", UpdateFailedCount=2, UpdaterLastFailure={**RECORD, "stage": stage})
      assert value == f"{text}: {RECORD['reason']}" and details.startswith(f"{text} ({stage}), ")

  def test_failure_that_is_not_a_command(self):
    record = {"time": "2026-11-10T18:38:02Z", "stage": "staging", "cmd": "", "rc": None, "reason": "RuntimeError: not valid for overlay FS!", "output": ""}
    value, _, _, details = self._show(UpdaterState="idle", UpdateFailedCount=1, UpdaterLastFailure=record)
    assert value == "check failed: RuntimeError: not valid for overlay FS!"
    assert details == "check failed (staging), 2026-11-10 18:38:02 UTC<br>RuntimeError: not valid for overlay FS!"

  def test_retry_outcome_is_shown(self):
    failed = {**RECORD, "retry": "failed", "retry_stage": "download", "retry_cmd": "git fetch origin inster", "retry_reason": "the same again"}
    details = self._show(UpdaterState="idle", UpdateFailedCount=1, UpdaterLastFailure=failed)[3]
    assert details.endswith("<br>immediate retry: failed (download): git fetch origin inster<br>the same again")

  def test_text_cannot_turn_into_markup(self):
    record = {**RECORD, "output": "error: <b>bold</b> <br> x"}
    details = self._show(UpdaterState="idle", UpdateFailedCount=1, UpdaterLastFailure=record)[3]
    assert details.endswith("<br>error: (b)bold(/b) (br) x") and details.count("<") == details.count("<br>") == 2

  def test_open_details_close_when_the_failure_is_gone(self):
    self._show(UpdaterState="idle", UpdateFailedCount=1, UpdaterLastFailure=RECORD)
    self.row.tap()
    assert self.row.description_visible
    assert self._show(UpdaterState="checking...", UpdateFailedCount=1, UpdaterLastFailure=RECORD)[3] == ""
    assert not self.row.description_visible
    assert self._show(UpdaterState="idle", UpdateFailedCount=0, UpdaterLastFailure=RECORD) == ("up to date, last checked never", "CHECK", True, "")
    assert not self.row.description_visible
