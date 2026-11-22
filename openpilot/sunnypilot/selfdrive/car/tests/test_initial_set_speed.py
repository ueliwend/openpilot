"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Initial set speed at engagement (owner's decisions 16 / 21 Nov 2026): 1. Speed Limit = Assist and the car's displayed speed limit and the
map agree on the speed limit -> limit + Speed Limit Offset; 2. else the current speed when at or above Cruise -> Initial Max Speed
(fallback); 3. else that setting. Resume keeps the last set speed; nothing changes with the car's own cruise (pcmCruise).
"""
import types

import pytest

from opendbc.car.structs import car
from openpilot.cereal import custom
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.car.cruise import VCruiseHelper, V_CRUISE_INITIAL, V_CRUISE_MAX, V_CRUISE_MIN, V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.car.initial_set_speed import agreed_speed_limit, speed_limit_initial_set_speed, current_map_speed_limit, \
                                                             engagement_speed_limits, MAP_LIMIT_MAX_AGE
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.common import Mode, OffsetType

ButtonEvent = car.CarState.ButtonEvent
ButtonType = car.CarState.ButtonEvent.Type

KPH = CV.KPH_TO_MS
MPH = CV.MPH_TO_MS
PCT3 = (OffsetType.percentage, 3)  # the owner's Speed Limit Offset (% , 3)


def rule1(car_kph, map_kph, mode=Mode.assist, offset=PCT3, is_metric=True, unit=KPH):
  return speed_limit_initial_set_speed(car_kph * unit, map_kph * unit, mode, offset[0], offset[1], is_metric)


class TestRuleOne(OpenpilotTestCase):
  def test_agree_percent_offset(self):
    assert rule1(80, 80) == 82.          # 80 + 3 % = 82.4 -> 82 km/h
    assert rule1(50, 50) == 52.          # 51.5 -> 52, as Speed Limit Assist rounds the same sum
    assert rule1(120, 120) == 124.       # 123.6 -> 124
    assert rule1(30, 30) == 31.          # 30.9 -> 31

  def test_agree_fixed_and_no_offset(self):
    assert rule1(80, 80, offset=(OffsetType.fixed, 5)) == 85.
    assert rule1(80, 80, offset=(OffsetType.fixed, -5)) == 75.
    assert rule1(80, 80, offset=(OffsetType.off, 3)) == 80.
    assert rule1(80, 80, offset=(OffsetType.percentage, -10)) == 72.

  def test_only_with_assist(self):
    # owner, 21 Nov 2026: Info, Warning and Off -> max(current speed, Initial Max Speed)
    assert rule1(60, 60, mode=Mode.assist) == 62.
    for mode in (Mode.information, Mode.warning, Mode.off, 7):
      assert rule1(60, 60, mode=mode) is None, mode

  def test_missing_or_disputed(self):
    for car_kph, map_kph in [(80, 0), (0, 80), (0, 0), (80, 100), (100, 80), (80, 81)]:
      assert rule1(car_kph, map_kph) is None, (car_kph, map_kph)

  def test_agree_means_same_whole_display_unit(self):
    # the map's m/s value is not bit-equal to the car's: same whole km/h agrees, a different one does not
    assert agreed_speed_limit(80 * KPH, 80.3 * KPH, True) == pytest.approx(80 * KPH)
    assert agreed_speed_limit(80 * KPH, 80.6 * KPH, True) == 0.
    assert agreed_speed_limit(-1., 80 * KPH, True) == 0.

  def test_imperial(self):
    # 55 mph + 3 % = 56.65 -> 57 mph, kept as km/h to 0.1 (shown as 57 mph)
    v = rule1(55, 55, is_metric=False, unit=MPH)
    assert v == round(57 * CV.MPH_TO_KPH, 1)
    assert round(v * CV.KPH_TO_MPH) == 57
    assert rule1(55, 55, is_metric=False, unit=MPH, offset=(OffsetType.fixed, 5)) == round(60 * CV.MPH_TO_KPH, 1)
    assert rule1(55, 56, is_metric=False, unit=MPH) is None

  def test_unknown_offset_type_falls_back(self):
    assert rule1(80, 80, offset=(7, 3)) is None


def map_msg(kph, limit_valid=True, ahead_kph=0.):
  msg = custom.LiveMapDataSP.new_message()
  msg.speedLimit, msg.speedLimitValid = kph * KPH, limit_valid
  msg.speedLimitAhead, msg.speedLimitAheadValid, msg.speedLimitAheadDistance = ahead_kph * KPH, ahead_kph > 0, 50.
  return msg


class FakeSM:
  """the part of card's SubMaster that engagement_speed_limits reads"""
  def __init__(self, msg, valid=True, alive=True, recv_time=100.):
    self.data = {'liveMapDataSP': msg}
    self.valid, self.alive, self.recv_time = {'liveMapDataSP': valid}, {'liveMapDataSP': alive}, {'liveMapDataSP': recv_time}

  def __getitem__(self, s):
    return self.data[s]


class TestMapGates(OpenpilotTestCase):
  """the map's limit counts only when mapd marks it valid, the message is valid (gpsOK), alive and at most 2 s old"""
  def test_usable(self):
    assert current_map_speed_limit(map_msg(80), True, True, 0.9) == pytest.approx(80 * KPH)
    assert current_map_speed_limit(map_msg(80), True, True, MAP_LIMIT_MAX_AGE) == pytest.approx(80 * KPH)

  def test_message_invalid_gps_not_ok(self):
    assert current_map_speed_limit(map_msg(80), False, True, 0.5) == 0.

  def test_not_alive(self):
    assert current_map_speed_limit(map_msg(80), True, False, 0.5) == 0.

  def test_too_old(self):
    assert current_map_speed_limit(map_msg(80), True, True, MAP_LIMIT_MAX_AGE + 0.1) == 0.
    assert current_map_speed_limit(map_msg(80), True, True, 1e9) == 0.   # never received: recv_time 0

  def test_limit_invalid(self):
    assert current_map_speed_limit(map_msg(80, limit_valid=False), True, True, 0.5) == 0.

  def test_current_road_not_look_ahead(self):
    assert current_map_speed_limit(map_msg(80, ahead_kph=50), True, True, 0.5) == pytest.approx(80 * KPH)

  def test_engagement_inputs(self):
    CI = fake_ci(cluster=60)
    car_l, map_l = engagement_speed_limits(CI, FakeSM(map_msg(60), recv_time=100.), now=101.)
    assert (car_l, map_l) == (pytest.approx(60 * KPH), pytest.approx(60 * KPH))
    assert engagement_speed_limits(CI, FakeSM(map_msg(60), recv_time=100.), now=103.) == (pytest.approx(60 * KPH), 0.)
    assert engagement_speed_limits(CI, FakeSM(map_msg(60), valid=False), now=100.5)[1] == 0.
    assert engagement_speed_limits(CI, FakeSM(map_msg(60), alive=False), now=100.5)[1] == 0.
    assert engagement_speed_limits(fake_ci(cluster=0), FakeSM(map_msg(60)), now=100.5)[0] == 0.


def fake_ci(cluster=0.):
  """the part of a car interface engagement_speed_limits reads: opendbc's CarState.speed_limit_cluster (km/h here)"""
  return types.SimpleNamespace(CS=types.SimpleNamespace(speed_limit_cluster=cluster * KPH))


class TestClusterLimit(OpenpilotTestCase):
  """the car's limit for rule 1 is the limit shown in the cluster (opendbc speed_limit_cluster, LKAS12), not carStateSP's further
  fallback to the head unit's Navi_HU value (owner, 21 / 22 Nov 2026)"""
  def test_rule_one_needs_the_cluster_limit(self):
    # cluster blank (Navi_HU may be 80, so carStateSP 80) and map 80: no rule 1 -> current speed
    h = VCruiseHelper(car.CarParams(pcmCruise=False), custom.CarParamsSP(pcmCruiseSpeed=True))
    Params().put("SpeedLimitMode", int(Mode.assist), block=True)
    h.read_custom_set_speed_params()
    h.update_speed_limit_assist(True, custom.LongitudinalPlanSP.new_message())
    h.set_initial_speed_limits(*engagement_speed_limits(fake_ci(cluster=0), FakeSM(map_msg(80)), now=100.5))
    CS = car.CarState(vEgo=float(50 * KPH), cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.decelCruise, pressed=False)]
    h.update_v_cruise(CS, enabled=True, is_metric=True)
    h.initialize_v_cruise(CS, 20)
    assert h.v_cruise_kph == 50


class TestEngagement(OpenpilotTestCase):
  def setup_method(self):
    self.CP = car.CarParams(pcmCruise=False)
    self.CP_SP = custom.CarParamsSP(pcmCruiseSpeed=True)
    self.params = Params()
    self.params.put("SpeedLimitMode", int(Mode.assist), block=True)
    self.params.put("SpeedLimitOffsetType", int(OffsetType.percentage), block=True)
    self.params.put("SpeedLimitValueOffset", 3, block=True)
    self.h = VCruiseHelper(self.CP, self.CP_SP)

  def engage(self, v_kph, initial=20, car_kph=0., map_kph=0., button=ButtonType.decelCruise):
    """card.py order: update, then on the rising edge of enabled the engagement limits and initialize with the buttons of that step"""
    self.h.update_speed_limit_assist(True, custom.LongitudinalPlanSP.new_message())
    self.h.set_initial_speed_limits(car_kph * KPH, map_kph * KPH)
    CS = car.CarState(vEgo=float(v_kph * KPH), cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=button, pressed=False)]
    self.h.update_v_cruise(CS, enabled=True, is_metric=True)
    self.h.initialize_v_cruise(CS, initial)
    return self.h.v_cruise_kph

  # rule 1
  def test_limit_agree_faster_than_car(self):
    assert self.engage(35, car_kph=50, map_kph=50) == 52.

  def test_limit_agree_slower_than_car(self):
    assert self.engage(95, car_kph=80, map_kph=80) == 82.

  def test_limit_agree_below_minimum_setting(self):
    # rule 1 comes first, also below Initial Max Speed (here 50)
    assert self.engage(10, initial=50, car_kph=30, map_kph=30) == 31.

  # rules 2 / 3 when rule 1 does not apply
  def test_limit_missing_or_disputed_uses_current_speed(self):
    for car_kph, map_kph in [(80, 0), (0, 80), (80, 60), (0, 0)]:
      assert self.engage(95, car_kph=car_kph, map_kph=map_kph) == 95, (car_kph, map_kph)
      assert self.engage(10, car_kph=car_kph, map_kph=map_kph) == 20, (car_kph, map_kph)

  def test_speed_limit_not_assist_uses_current_speed(self):
    for mode in (Mode.off, Mode.information, Mode.warning):
      self.params.put("SpeedLimitMode", int(mode), block=True)
      self.h.read_custom_set_speed_params()
      assert self.engage(95, car_kph=80, map_kph=80) == 95, mode
      assert self.engage(35, car_kph=80, map_kph=80) == 35, mode
      assert self.engage(10, car_kph=80, map_kph=80) == 20, mode

  def test_unusable_map_uses_current_speed(self):
    # card passes 0 for a map limit that fails a gate (TestMapGates)
    assert self.engage(95, car_kph=80, map_kph=0) == 95

  def test_exactly_at_minimum(self):
    assert self.engage(20, initial=20) == 20
    assert self.engage(30, initial=30) == 30

  def test_below_minimum(self):
    assert self.engage(12, initial=20) == 20
    assert self.engage(0, initial=20) == 20
    assert self.engage(25, initial=30) == 30

  def test_minimum_twenty_now_possible(self):
    # upstream clipped every initial set speed to at least 40 km/h
    assert V_CRUISE_INITIAL == 20
    assert self.engage(5, initial=20) == 20

  def test_upper_bound(self):
    assert self.engage(160, initial=20) == V_CRUISE_MAX

  # resume keeps the last set speed, also when car and map agree
  def test_resume_keeps_last_set_speed(self):
    assert self.engage(95, car_kph=0, map_kph=0) == 95
    self.h.update_v_cruise(car.CarState(vEgo=float(70 * KPH), cruiseState={"available": True}), enabled=False, is_metric=True)
    assert self.engage(70, car_kph=80, map_kph=80, button=ButtonType.accelCruise) == 95

  def test_resume_without_earlier_set_speed_uses_rules(self):
    assert self.engage(70, car_kph=80, map_kph=80, button=ButtonType.accelCruise) == 82.

  def test_rule_one_within_button_limits(self):
    v = self.engage(10, car_kph=5, map_kph=5)
    assert V_CRUISE_MIN <= v <= V_CRUISE_MAX


class TestPcmCruiseUnchanged(OpenpilotTestCase):
  def test_pcm_cruise_set_speed_from_car(self):
    h = VCruiseHelper(car.CarParams(pcmCruise=True), custom.CarParamsSP(pcmCruiseSpeed=True))
    h.update_speed_limit_assist(True, custom.LongitudinalPlanSP.new_message())
    h.set_initial_speed_limits(80 * KPH, 80 * KPH)
    CS = car.CarState(vEgo=float(60 * KPH), cruiseState={"available": True, "speed": float(100 * KPH), "speedCluster": float(100 * KPH)})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.decelCruise, pressed=False)]
    h.update_v_cruise(CS, enabled=True, is_metric=True)
    h.initialize_v_cruise(CS, 20)
    assert h.v_cruise_kph == pytest.approx(100)

  def test_pcm_cruise_initialize_is_noop(self):
    h = VCruiseHelper(car.CarParams(pcmCruise=True), custom.CarParamsSP(pcmCruiseSpeed=True))
    h.update_speed_limit_assist(True, custom.LongitudinalPlanSP.new_message())
    h.set_initial_speed_limits(80 * KPH, 80 * KPH)
    h.initialize_v_cruise(car.CarState(vEgo=float(60 * KPH)), 20)
    assert h.v_cruise_kph == V_CRUISE_UNSET


class TestParams(OpenpilotTestCase):
  def test_default_twenty(self):
    assert Params().get_default_value("InitialCruiseSpeedKph") == 20

  def test_helper_never_writes_settings(self):
    # E12: no setting may be rewritten by a start; the helper only reads (an out-of-range offset type stays as stored)
    params = Params()
    stored = {"InitialCruiseSpeedKph": 40, "SpeedLimitMode": 3, "SpeedLimitOffsetType": 7, "SpeedLimitValueOffset": 99}
    for k, v in stored.items():
      params.put(k, v, block=True)
    h = VCruiseHelper(car.CarParams(pcmCruise=False), custom.CarParamsSP(pcmCruiseSpeed=True))
    h.read_custom_set_speed_params()
    h.update_speed_limit_assist(True, custom.LongitudinalPlanSP.new_message())
    h.set_initial_speed_limits(80 * KPH, 80 * KPH)
    CS = car.CarState(vEgo=float(50 * KPH), cruiseState={"available": True})
    CS.buttonEvents = [ButtonEvent(type=ButtonType.decelCruise, pressed=False)]
    h.update_v_cruise(CS, enabled=True, is_metric=True)
    h.initialize_v_cruise(CS, 40)
    assert h.v_cruise_kph == 50          # unknown offset type: no rule 1, current speed
    for k, v in stored.items():
      assert params.get(k) == v, k


class TestClusterLimitInsterInterface(OpenpilotTestCase):
  """engagement_speed_limits on the real Hyundai Inster car interface (opendbc's recorded Inster fingerprint and CAN, an LKAS12
  displayed limit and a Navi_HU limit packed onto the bus): it takes opendbc's speed_limit_cluster; the detailed cases are opendbc's
  sunnypilot/car/hyundai/tests/test_speed_limit_cluster.py"""
  def test_reads_cluster_limit(self):
    import json
    import lzma
    import os
    import opendbc.car.hyundai as hyundai_pkg
    from opendbc.can import CANPacker
    from opendbc.car import Bus
    from opendbc.car.can_definitions import CanData
    from opendbc.car.hyundai.interface import CarInterface
    from opendbc.car.hyundai.values import CAR, DBC
    from opendbc.sunnypilot.car.interfaces import setup_interfaces
    with lzma.open(os.path.join(os.path.dirname(hyundai_pkg.__file__), "tests", "inster_carstate_can.json.xz"), "rt") as f:
      data = json.load(f)
    fingerprint = {bus: {} for bus in range(8)}
    for bus, msgs in data["fingerprint"].items():
      fingerprint[int(bus)] = {int(addr): size for addr, size in msgs.items()}
    cycles = [(t, [CanData(addr, bytes.fromhex(dat), bus) for addr, dat, bus in frames]) for t, frames in data["cycles"]]
    CP = CarInterface.get_params(CAR.HYUNDAI_INSTER, fingerprint, [], True, False, False)
    CP_SP = CarInterface.get_params_sp(CP, CAR.HYUNDAI_INSTER, fingerprint, [], True, False, False)
    setup_interfaces(CarInterface, CP, CP_SP, [{"HyundaiLongitudinalTuning": 2}])
    CI = CarInterface(CP, CP_SP)
    packer = CANPacker(DBC[CAR.HYUNDAI_INSTER][Bus.pt])
    for t, frames in cycles[:50]:
      CI.update([(t, frames)])
    for i, (lkas12, nav, want_car, want_sp) in enumerate([(80, 60, 80, 80), (255, 60, 0, 60)]):
      t, frames = cycles[50 + i]
      _, CS_SP = CI.update([(t, frames + [packer.make_can_msg("LKAS12", 2, {"CF_Lkas_TsrSpeed_Display_Clu": lkas12}),
                                          packer.make_can_msg("Navi_HU", 0, {"SpeedLim_Nav_Clu": nav})])])
      car_l, _ = engagement_speed_limits(CI, FakeSM(map_msg(80)), now=100.5)
      assert car_l == pytest.approx(want_car * KPH), (lkas12, nav)
      assert CS_SP.speedLimit == pytest.approx(want_sp * KPH), (lkas12, nav)
