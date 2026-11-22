"""A start without a recognised car keeps every user setting (owner, 21 Nov 2026).

Route 00000069 started with the mock platform (fingerprinting failed); during that start the cleanup code removed
AlphaLongitudinalEnabled, AutoLaneChangeBsmDelay, CustomAccIncrementsEnabled and both Smart Cruise Control settings and
rewrote Speed Limit "Assist" to "Warning", so the next start (0000006a, car recognised) ran without openpilot longitudinal.
These tests run the car-side paths (card -> sunnypilot setup_interfaces, plannerd -> SpeedLimitAssist, selfdrived) with
the mock platform and with the Inster, with and without openpilot longitudinal."""
from types import SimpleNamespace
from unittest import mock

from openpilot.cereal import custom, log
from opendbc.car import gen_empty_fingerprint
from opendbc.car.car_helpers import interfaces
from opendbc.car.structs import car
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.car.helpers import is_car_recognised, is_mock_car
from openpilot.sunnypilot.selfdrive.car import interfaces as sunnypilot_interfaces
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.common import Mode
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.helpers import set_speed_limit_assist_availability
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_assist import SpeedLimitAssist

# the owner's settings before the unrecognised start (initData of 00000068 / 00000069), plus settings that a recognised
# car's cleanup would change (torque controller conflict, ICBM, metrics below the chevron)
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
HYUNDAI_HAS_BSM_MSG = 0x58b  # non-CAN-FD Hyundai: HAS_BSM when this message is in the fingerprint (opendbc hyundai interface)


def seed_settings(params: Params) -> None:
  params.put_bool("IsReleaseSpBranch", False, block=True)
  for k, v in SETTINGS.items():
    if isinstance(v, bool):
      params.put_bool(k, v, block=True)
    else:
      params.put(k, v, block=True)


def snapshot(params: Params) -> dict:
  return {k: params.get(k) for k in SETTINGS}


def mock_car():
  """The platform card falls back to when fingerprinting fails (00000069: MOCK, brand mock, dashcamOnly)."""
  CarInterface = interfaces["MOCK"]
  CP = CarInterface.get_non_essential_params("MOCK")
  CP_SP = CarInterface.get_non_essential_params_sp(CP, "MOCK")
  return CarInterface(CP, CP_SP)


def inster(alpha_long: bool):
  """HYUNDAI_INSTER as card builds it: openpilot longitudinal = AlphaLongitudinalEnabled and alpha available."""
  CarInterface = interfaces["HYUNDAI_INSTER"]
  fingerprint = gen_empty_fingerprint()
  fingerprint[0][HYUNDAI_HAS_BSM_MSG] = 8
  CP = CarInterface.get_params("HYUNDAI_INSTER", fingerprint, [], alpha_long, False, False)
  CP_SP = CarInterface.get_params_sp(CP, "HYUNDAI_INSTER", fingerprint, [], alpha_long, False, False)
  return CarInterface(CP, CP_SP)


class TestCarRecognisedPredicate(OpenpilotTestCase):
  def test_mock_platform_is_not_recognised(self):
    CI = mock_car()
    assert CI.CP.brand == "mock" and CI.CP.carFingerprint == "MOCK" and CI.CP.dashcamOnly
    assert is_mock_car(CI.CP)
    assert not is_car_recognised(CI.CP)

  def test_no_or_empty_car_params_are_not_recognised(self):
    assert not is_car_recognised(None)
    assert not is_car_recognised(car.CarParams.new_message().as_reader())
    assert not is_car_recognised(car.CarParams.new_message(brand="mock", carFingerprint="MOCK").as_reader())
    assert not is_car_recognised(car.CarParams.new_message(brand="hyundai", carFingerprint="MOCK").as_reader())

  def test_inster_is_recognised(self):
    assert is_car_recognised(inster(True).CP)
    assert is_car_recognised(car.CarParams.new_message(brand="hyundai", carFingerprint="HYUNDAI_INSTER").as_reader())
    # a recognised car in dashcam mode (unsupported platform) still describes the car: upstream rules apply
    assert is_car_recognised(car.CarParams.new_message(brand="hyundai", carFingerprint="HYUNDAI_INSTER", dashcamOnly=True).as_reader())


class TestCardKeepsSettings(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()
    seed_settings(self.params)

  def test_unrecognised_start_changes_no_setting(self):
    CI = mock_car()
    assert CI.CP_SP.pcmCruiseSpeed and not CI.CP.openpilotLongitudinalControl  # as logged on 00000069
    before = snapshot(self.params)
    sunnypilot_interfaces.setup_interfaces(CI, self.params)
    assert snapshot(self.params) == before

  def test_speed_limit_assist_kept_on_unrecognised_start(self):
    CI = mock_car()
    assert not set_speed_limit_assist_availability(CI.CP, CI.CP_SP, self.params)
    assert self.params.get("SpeedLimitMode") == int(Mode.assist)
    sla = SpeedLimitAssist(CI.CP, CI.CP_SP)
    assert not sla.enabled  # not available: Speed Limit Assist does nothing on this start
    for frame in range(200):  # periodic re-check in plannerd
      sla.frame = frame
      sla.update_params()
    assert not sla.enabled
    assert self.params.get("SpeedLimitMode") == int(Mode.assist)

  def test_recognised_inster_with_longitudinal_unchanged(self):
    CI = inster(True)
    assert CI.CP.openpilotLongitudinalControl and CI.CP.alphaLongitudinalAvailable
    self.params.put_bool("LateralJerkTorqueController", False, block=True)
    self.params.put_bool("NeuralNetworkLateralControl", False, block=True)
    before = snapshot(self.params)
    sunnypilot_interfaces.setup_interfaces(CI, self.params)
    after = snapshot(self.params)
    # upstream rule kept: ICBM is cleared when openpilot longitudinal is on; nothing else changes
    assert after.pop("IntelligentCruiseButtonManagement") is None  # removed
    before.pop("IntelligentCruiseButtonManagement")
    assert after == before
    sla = SpeedLimitAssist(CI.CP, CI.CP_SP)
    assert sla.enabled and self.params.get("SpeedLimitMode") == int(Mode.assist)

  def test_recognised_inster_without_longitudinal_keeps_upstream_rules(self):
    # recognised car that really runs without openpilot longitudinal (owner turned Alpha off, ICBM off):
    # upstream behaviour is kept: Assist -> Warning, cruise features that need longitudinal or ICBM are cleared
    self.params.put_bool("IntelligentCruiseButtonManagement", False, block=True)
    CI = inster(False)
    assert not CI.CP.openpilotLongitudinalControl and CI.CP_SP.pcmCruiseSpeed
    sunnypilot_interfaces.setup_interfaces(CI, self.params)
    assert self.params.get("SpeedLimitMode") == int(Mode.warning)
    for k in ("CustomAccIncrementsEnabled", "SmartCruiseControlVision", "SmartCruiseControlMap"):
      assert self.params.get(k) is None, k  # removed
    # the car-side cleanup never touches these
    assert self.params.get_bool("AlphaLongitudinalEnabled") and self.params.get_bool("AutoLaneChangeBsmDelay")

  def test_recognised_car_conflicting_lateral_controllers_still_resolved(self):
    CI = inster(True)
    sunnypilot_interfaces.setup_interfaces(CI, self.params)
    assert not self.params.get_bool("LateralJerkTorqueController")
    assert not self.params.get_bool("NeuralNetworkLateralControl")

  def test_rivian_speed_limit_assist_still_rewritten(self):
    # upstream case kept for recognised cars: Speed Limit Assist is never allowed on Rivian
    CI = mock_car()
    CP = car.CarParams.new_message(brand="rivian", carFingerprint="RIVIAN_R1_GEN1", openpilotLongitudinalControl=True).as_reader()
    assert not set_speed_limit_assist_availability(CP, CI.CP_SP, self.params)
    assert self.params.get("SpeedLimitMode") == int(Mode.warning)


class TestSelfdrivedKeepsAlphaLongitudinal(OpenpilotTestCase):
  def make_selfdrived(self, CP):
    from openpilot.selfdrive.selfdrived import selfdrived
    build = SimpleNamespace(openpilot=SimpleNamespace(comma_remote=False, sunnypilot_remote=False), tested_channel=False)
    with mock.patch.object(selfdrived, "get_build_metadata", return_value=build), \
         mock.patch.object(selfdrived, "messaging"), mock.patch.object(selfdrived, "VisionIpcClient"):
      return selfdrived.SelfdriveD(CP.as_reader(), custom.CarParamsSP.new_message().as_reader())

  def setup_method(self):
    self.params = Params()
    seed_settings(self.params)

  def test_unrecognised_start_keeps_alpha_longitudinal(self):
    before = snapshot(self.params)
    sd = self.make_selfdrived(car.CarParams.new_message(brand="mock", carFingerprint="MOCK", dashcamOnly=True))
    assert snapshot(self.params) == before
    assert log.OnroadEvent.EventName.carUnrecognized in sd.events.static_events  # the start is still reported

  def test_recognised_car_without_alpha_still_clears_it(self):
    # upstream rule kept: a recognised car that has no alpha longitudinal clears the setting
    self.make_selfdrived(car.CarParams.new_message(brand="hyundai", carFingerprint="HYUNDAI_INSTER", alphaLongitudinalAvailable=False))
    assert self.params.get("AlphaLongitudinalEnabled") is None  # removed

  def test_recognised_inster_keeps_alpha_longitudinal(self):
    self.make_selfdrived(car.CarParams.new_message(brand="hyundai", carFingerprint="HYUNDAI_INSTER", alphaLongitudinalAvailable=True,
                                                   openpilotLongitudinalControl=True))
    assert self.params.get_bool("AlphaLongitudinalEnabled")
