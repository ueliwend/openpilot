"""The UI keeps every user setting while the last start's car was not recognised (owner, 21 Nov 2026).

The UI re-reads CarParamsPersistent five times a second and, in UIStateSP._enforce_constraints and in the settings panels,
removed or rewrote settings the car seemed to lack. After an unrecognised start (mock platform) that cleared Alpha
longitudinal, the blind-spot lane change delay, Custom ACC increments, both Smart Cruise Control settings and Speed
Limit "Assist". These tests run that code with the mock platform and with the Inster. pyray, the D-Bus library and the
UI application are replaced by mocks so the logic runs without a display."""
import importlib
import sys
import types
from types import SimpleNamespace
from unittest import mock

from openpilot.cereal import custom
from opendbc.car.structs import car
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.car.helpers import is_car_recognised
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.common import Mode

_STUBS = ("pyray", "raylib", "jeepney", "jeepney.io", "jeepney.io.blocking", "jeepney.io.threading", "jeepney.low_level",
          "jeepney.wrappers", "jeepney.bus_messages", "openpilot.system.ui.lib.application",
          "openpilot.selfdrive.ui.sunnypilot.layouts.settings.display", "openpilot.system.ui.sunnypilot.widgets.screen_saver")
_MODULES = {
  "ui_state_sp": "openpilot.selfdrive.ui.sunnypilot.ui_state",
  "developer": "openpilot.selfdrive.ui.layouts.settings.developer",
  "developer_mici": "openpilot.selfdrive.ui.mici.layouts.settings.developer",
  "cruise": "openpilot.selfdrive.ui.sunnypilot.layouts.settings.cruise",
  "lane_change": "openpilot.selfdrive.ui.sunnypilot.layouts.settings.steering_sub_layouts.lane_change_settings",
  "speed_limit": "openpilot.selfdrive.ui.sunnypilot.layouts.settings.cruise_sub_layouts.speed_limit_settings",
  "visuals": "openpilot.selfdrive.ui.sunnypilot.layouts.settings.visuals",
}


def _import_ui_modules():
  fake_ui_state = types.ModuleType("openpilot.selfdrive.ui.ui_state")
  fake_ui_state.ui_state = None
  fake_ui_state.UIStatus = mock.MagicMock()
  stubs = {name: mock.MagicMock() for name in _STUBS}
  stubs["openpilot.selfdrive.ui.ui_state"] = fake_ui_state
  with mock.patch.dict(sys.modules, stubs):
    return SimpleNamespace(**{k: importlib.import_module(v) for k, v in _MODULES.items()})


UI = _import_ui_modules()

SETTINGS = {
  "AlphaLongitudinalEnabled": True,
  "AutoLaneChangeBsmDelay": True,
  "AutoLaneChangeTimer": 1,
  "CustomAccIncrementsEnabled": True,
  "SmartCruiseControlVision": True,
  "SmartCruiseControlMap": True,
  "IntelligentCruiseButtonManagement": True,
  "SpeedLimitMode": int(Mode.assist),
  "SpeedLimitValueOffset": 3,
  "ChevronInfo": 4,
  "EnforceTorqueControl": True,
  "NeuralNetworkLateralControl": True,
  "LateralJerkTorqueController": True,
}
INSTER_FLAGS = 2030143872  # carParams.flags of every recognised start in the archive (incl. HAS_BSM), LOG


def mock_cp():
  # as logged on 00000069 (carParams): MOCK, brand mock, dashcamOnly, no longitudinal, no alpha
  return car.CarParams.new_message(brand="mock", carFingerprint="MOCK", dashcamOnly=True).as_reader()


def mock_cp_sp():
  return custom.CarParamsSP.new_message(pcmCruiseSpeed=True).as_reader()


def inster_cp(long: bool):
  return car.CarParams.new_message(brand="hyundai", carFingerprint="HYUNDAI_INSTER", flags=INSTER_FLAGS,
                                   alphaLongitudinalAvailable=True, openpilotLongitudinalControl=long, pcmCruise=not long).as_reader()


def inster_cp_sp():
  return custom.CarParamsSP.new_message(intelligentCruiseButtonManagementAvailable=True, pcmCruiseSpeed=True).as_reader()


def seed(params: Params, **overrides):
  for k, v in {**SETTINGS, **overrides}.items():
    if isinstance(v, bool):
      params.put_bool(k, v, block=True)
    else:
      params.put(k, v, block=True)


def snapshot(params: Params) -> dict:
  return {k: params.get(k) for k in SETTINGS}


def fake_ui_state(CP, CP_SP, has_long: bool, has_icbm: bool = False):
  return SimpleNamespace(params=Params(), CP=CP, CP_SP=CP_SP, has_longitudinal_control=has_long, has_icbm=has_icbm,
                         is_sp_release=False, is_release=False, car_recognised=is_car_recognised(CP), is_metric=True,
                         engaged=False, is_offroad=lambda: True, update_params=lambda: None)


def panel(cls, **attrs):
  """An instance of a settings panel without its widgets: every widget it touches is a MagicMock."""
  def __getattr__(self, name):
    if name.startswith("__"):
      raise AttributeError(name)
    m = mock.MagicMock()
    object.__setattr__(self, name, m)
    return m
  obj = object.__new__(type("Test" + cls.__name__, (cls,), {"__getattr__": __getattr__, "__abstractmethods__": frozenset()}))
  for k, v in attrs.items():
    object.__setattr__(obj, k, v)
  return obj


def ui_state_sp(CP, CP_SP, has_long: bool, has_icbm: bool = False):
  s = object.__new__(UI.ui_state_sp.UIStateSP)
  s.params = Params()
  s.CP, s.CP_SP, s.has_longitudinal_control, s.has_icbm = CP, CP_SP, has_long, has_icbm
  return s


class TestUIStateKeepsSettings(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()
    seed(self.params)

  def test_car_recognised_property(self):
    assert not ui_state_sp(None, None, False).car_recognised
    assert not ui_state_sp(mock_cp(), mock_cp_sp(), False).car_recognised
    assert ui_state_sp(inster_cp(True), inster_cp_sp(), True).car_recognised

  def test_unrecognised_start_changes_no_setting(self):
    before = snapshot(self.params)
    for _ in range(10):  # runs five times a second on the device
      ui_state_sp(mock_cp(), mock_cp_sp(), has_long=False)._enforce_constraints()
    assert snapshot(self.params) == before

  def test_no_car_params_changes_no_setting(self):
    before = snapshot(self.params)
    ui_state_sp(None, None, has_long=False)._enforce_constraints()
    assert snapshot(self.params) == before

  def test_recognised_inster_with_longitudinal_unchanged(self):
    seed(self.params, LateralJerkTorqueController=False, NeuralNetworkLateralControl=False)
    before = snapshot(self.params)
    ui_state_sp(inster_cp(True), inster_cp_sp(), has_long=True)._enforce_constraints()
    after = snapshot(self.params)
    assert after.pop("IntelligentCruiseButtonManagement") is None  # upstream: ICBM cleared while openpilot longitudinal is on
    before.pop("IntelligentCruiseButtonManagement")
    assert after == before

  def test_recognised_inster_without_longitudinal_keeps_upstream_rules(self):
    seed(self.params, IntelligentCruiseButtonManagement=False, LateralJerkTorqueController=False, NeuralNetworkLateralControl=False)
    ui_state_sp(inster_cp(False), inster_cp_sp(), has_long=False)._enforce_constraints()
    for k in ("CustomAccIncrementsEnabled", "SmartCruiseControlVision", "SmartCruiseControlMap"):
      assert self.params.get(k) is None, k
    assert self.params.get_bool("AlphaLongitudinalEnabled")  # alpha is available on the Inster
    assert self.params.get_bool("AutoLaneChangeBsmDelay")  # HAS_BSM set

  def test_recognised_car_without_bsm_or_alpha_keeps_upstream_rules(self):
    CP = car.CarParams.new_message(brand="hyundai", carFingerprint="HYUNDAI_INSTER", flags=0, alphaLongitudinalAvailable=False).as_reader()
    ui_state_sp(CP, inster_cp_sp(), has_long=False)._enforce_constraints()
    assert self.params.get("AlphaLongitudinalEnabled") is None
    assert self.params.get("AutoLaneChangeBsmDelay") is None
    assert not self.params.get_bool("NeuralNetworkLateralControl")  # conflict with LateralJerkTorqueController resolved


class TestSettingsPanelsKeepSettings(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()
    seed(self.params)

  def run_panels(self, ui):
    with mock.patch.object(UI.cruise, "ui_state", ui), mock.patch.object(UI.lane_change, "ui_state", ui), \
         mock.patch.object(UI.speed_limit, "ui_state", ui), mock.patch.object(UI.visuals, "ui_state", ui), \
         mock.patch.object(UI.developer, "ui_state", ui), mock.patch.object(UI.developer_mici, "ui_state", ui):
      panel(UI.cruise.CruiseLayout)._update_state()
      panel(UI.lane_change.LaneChangeSettingsLayout)._update_toggles()
      panel(UI.speed_limit.SpeedLimitSettingsLayout)._update_state()
      panel(UI.visuals.VisualsLayout, _params=Params(), _toggle_defs={})._update_state()
      panel(UI.developer.DeveloperLayout, _params=Params(), _is_release=False)._update_toggles()
      panel(UI.developer_mici.DeveloperLayoutMici)._update_toggles()

  def test_unrecognised_start_changes_no_setting(self):
    before = snapshot(self.params)
    self.run_panels(fake_ui_state(mock_cp(), mock_cp_sp(), has_long=False))
    assert snapshot(self.params) == before

  def test_recognised_inster_with_longitudinal_unchanged(self):
    before = snapshot(self.params)
    self.run_panels(fake_ui_state(inster_cp(True), inster_cp_sp(), has_long=True))
    after = snapshot(self.params)
    assert after.pop("IntelligentCruiseButtonManagement") is None  # upstream: ICBM cleared while openpilot longitudinal is on
    before.pop("IntelligentCruiseButtonManagement")
    assert after == before

  def test_recognised_inster_without_longitudinal_keeps_upstream_rules(self):
    seed(self.params, IntelligentCruiseButtonManagement=False)
    self.run_panels(fake_ui_state(inster_cp(False), inster_cp_sp(), has_long=False))
    assert self.params.get("SpeedLimitMode") == int(Mode.warning)
    assert self.params.get("ChevronInfo") == 0
    for k in ("CustomAccIncrementsEnabled", "SmartCruiseControlVision", "SmartCruiseControlMap"):
      assert self.params.get(k) is None, k
    assert self.params.get_bool("AlphaLongitudinalEnabled") and self.params.get_bool("AutoLaneChangeBsmDelay")

  def test_recognised_car_without_bsm_or_alpha_keeps_upstream_rules(self):
    CP = car.CarParams.new_message(brand="hyundai", carFingerprint="HYUNDAI_INSTER", flags=0, alphaLongitudinalAvailable=False).as_reader()
    for developer_panel in ("developer", "developer_mici", "lane_change"):
      seed(self.params)
      ui = fake_ui_state(CP, inster_cp_sp(), has_long=False)
      with mock.patch.object(UI.developer, "ui_state", ui), mock.patch.object(UI.developer_mici, "ui_state", ui), \
           mock.patch.object(UI.lane_change, "ui_state", ui):
        if developer_panel == "developer":
          panel(UI.developer.DeveloperLayout, _params=Params(), _is_release=False)._update_toggles()
          assert self.params.get("AlphaLongitudinalEnabled") is None
        elif developer_panel == "developer_mici":
          panel(UI.developer_mici.DeveloperLayoutMici)._update_toggles()
          assert self.params.get("AlphaLongitudinalEnabled") is None
        else:
          panel(UI.lane_change.LaneChangeSettingsLayout)._update_toggles()
          assert self.params.get("AutoLaneChangeBsmDelay") is None

  def test_release_branch_still_hides_and_clears_alpha(self):
    # not car-dependent (upstream): release branches clear the alpha longitudinal setting whatever the car
    ui = fake_ui_state(mock_cp(), mock_cp_sp(), has_long=False)
    with mock.patch.object(UI.developer, "ui_state", ui):
      panel(UI.developer.DeveloperLayout, _params=Params(), _is_release=True)._update_toggles()
    assert self.params.get("AlphaLongitudinalEnabled") is None
