"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import copy
import math
import random
from unittest import mock

import numpy as np

from openpilot.cereal import log
from opendbc.car.structs import car
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI, INSTER_STEER_MAX, inster_steer_max
from opendbc.car.vehicle_model import VehicleModel
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP
from opendbc.sunnypilot.car.interfaces import setup_interfaces as opendbc_sunnypilot_setup_interfaces
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.sunnypilot.selfdrive.car import interfaces as sunnypilot_interfaces
from openpilot.sunnypilot.selfdrive.controls.lib import inster_lateral, latcontrol_torque_v0
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_LATERAL_ACCEL_NO_ROLL, clip_curvature
from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import A_NODES, CMD_RAW, CMD_RAW_EXIT, F0, FF_TABLE, FF_TABLE_EXIT, \
                                                                       INSTER_SWITCHES, SCALE_CLIP, STEER_MAX_RAW, V_NODES, \
                                                                       InsterStictionFriction, ff_lat_accel_equivalent, ff_torque, \
                                                                       high_torque_steer_max, is_inster_high_torque, \
                                                                       low_speed_kp_scale, max_lateral_accel_no_roll, table_torque, \
                                                                       TORQUE_CAP_DEFAULT, TORQUE_CAP_MAX, TORQUE_CAP_MIN, inster_torque_cap
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import InsterLowSpeedPID, LatControlTorque as LatControlTorqueV0

INSTER_KEYS = ("InsterStictionFriction", "InsterSpeedFeedforward")   # the step-2 switches
STEP3_KEYS = ("InsterExitTuning", "InsterLowSpeedP", "InsterHighTorqueLimits")
ALL_INSTER_KEYS = INSTER_KEYS + STEP3_KEYS
SPEEDS = [0.0, 2.0, 5.0, 6.1] + list(np.linspace(6.11, 30.83, 60)) + [31.0, 35.0, 45.0]
ACCELS = list(np.linspace(0.0, 4.0, 161))


def _term(x, h, v=20.0, rate=0.0, lat_delay=0.2, exit_tuning=False, wheel_rate=0.0):
  """the term with settled filters (first update after a reset) for intended wheel rate x (deg/s) and held torque h (raw)"""
  fr = InsterStictionFriction(DT_CTRL, exit_tuning=exit_tuning)
  return fr.update(v, x * lat_delay, 0.0, 0.0, lat_delay, rate, h, wheel_rate)


class TestInsterTable(OpenpilotTestCase):
  def test_cap_unchanged(self):
    # the table and the friction amplitudes are in raw units of the 384 cap: review them before changing INSTER_STEER_MAX
    assert STEER_MAX_RAW == INSTER_STEER_MAX

  def test_nodes(self):
    for j, v in enumerate(V_NODES):
      for i, a in enumerate(A_NODES):
        assert abs(table_torque(a, v) * STEER_MAX_RAW - CMD_RAW[j][i]) < 1e-9

  def test_zero_at_zero(self):
    for v in SPEEDS:
      assert table_torque(0.0, v) == 0.0
      assert ff_torque(0.0, v, 2.8) == 0.0
      assert ff_lat_accel_equivalent(0.0, v, 2.8) == 0.0

  def test_odd(self):
    for v in SPEEDS:
      for a in ACCELS:
        for f in (1.0, 2.8, 3.7):
          assert ff_torque(-a, v, f) == -ff_torque(a, v, f)

  def test_monotone_in_lateral_accel(self):
    for v in SPEEDS:
      t = [table_torque(a, v) for a in ACCELS]
      assert all(t1 >= t0 - 1e-12 for t0, t1 in zip(t[:-1], t[1:], strict=True))
      assert t[-1] > t[0]

  def test_non_increasing_with_speed(self):
    for a in ACCELS:
      t = [table_torque(a, v) for v in SPEEDS]
      assert all(t1 <= t0 + 1e-12 for t0, t1 in zip(t[:-1], t[1:], strict=True))

  def test_bounded(self):
    for v in SPEEDS:
      for a in ACCELS + [10.0, 100.0]:
        for f in (0.0, 0.5, 2.8, 10.0):
          assert abs(ff_torque(a, v, f)) <= 1.0
          assert abs(ff_torque(-a, v, f)) <= 1.0

  def test_clamped_outside_speed_nodes(self):
    for a in ACCELS:
      assert table_torque(a, 1.0) == table_torque(a, V_NODES[0])
      assert table_torque(a, 40.0) == table_torque(a, V_NODES[-1])

  def test_scale_clip(self):
    for v in (5.0, 12.0, 25.0):
      for a in (0.1, 0.6, 1.3, 2.2):
        assert ff_torque(a, v, F0) == table_torque(a, v)
        assert ff_torque(a, v, 3.0) == table_torque(a * (F0 / 3.0), v)             # inside the clip: scaled by F0 / factor
        assert ff_torque(a, v, 1.0) == table_torque(a * SCALE_CLIP[1], v)          # factor far below F0: at most +15 %
        assert ff_torque(a, v, 0.0) == table_torque(a * SCALE_CLIP[1], v)
        assert ff_torque(a, v, 10.0) == table_torque(a * SCALE_CLIP[0], v)         # factor far above F0: at most -15 %

  def test_lat_accel_equivalent(self):
    for v in (5.0, 20.0):
      for a in (-2.0, -0.3, 0.4, 1.7):
        for f in (2.0, 2.8, 3.7):
          assert ff_lat_accel_equivalent(a, v, f) == f * ff_torque(a, v, f)


class TestInsterStictionFriction(OpenpilotTestCase):
  def test_value_and_slope_zero_at_zero(self):
    for h in (-250.0, -20.0, 0.0, 20.0, 250.0):
      for v in (3.0, 10.0, 20.0):
        assert _term(0.0, h, v) == 0.0
        eps = 1e-5
        assert abs(_term(eps, h, v) / eps) < 1e-3 and abs(_term(-eps, h, v) / eps) < 1e-3

  def test_no_dead_zone(self):
    for x in (1e-3, 0.01, 0.5, 3.0, 20.0):
      for h in (-100.0, 0.0, 100.0):
        assert _term(x, h) > 0.0
        assert _term(-x, h) < 0.0

  def test_continuous_at_in_out_switch(self):
    # in / out amplitude switch: at x = 0 both branches give 0 with zero slope; across h = 0 both branches equal FRICTION_A
    for x in (-8.0, -1.0, 1.0, 8.0):
      assert abs(_term(x, 1e-6) - _term(x, -1e-6)) < 1e-6
    for h in (-120.0, 120.0):
      assert abs(_term(1e-5, h) - _term(-1e-5, h)) < 1e-6

  def test_bounded(self):
    rng = random.Random(3)
    fr = InsterStictionFriction(DT_CTRL)
    for _ in range(20000):
      t = fr.update(rng.uniform(0, 40), rng.uniform(-900, 900), rng.uniform(-900, 900), rng.uniform(-900, 900), rng.uniform(0.0, 0.5),
                    rng.uniform(0, 600), rng.uniform(-384, 384) * rng.choice([1, 10]))
      assert abs(t) <= inster_lateral.FRICTION_MAX

  def test_no_hysteresis(self):
    rng = random.Random(7)
    seq = [(rng.uniform(5, 30), rng.gauss(0, 30), rng.gauss(0, 30), rng.gauss(0, 30), 0.2, abs(rng.gauss(0, 20)), rng.gauss(0, 150))
           for _ in range(1500)]
    # the same input history gives the same output
    a, b = InsterStictionFriction(DT_CTRL), InsterStictionFriction(DT_CTRL)
    assert [a.update(*s) for s in seq] == [b.update(*s) for s in seq]
    # no latch: after different early histories the outputs converge once the inputs agree (fading memory of first-order filters)
    c = InsterStictionFriction(DT_CTRL)
    for s in seq[:300]:
      c.update(s[0], -s[1], s[2], s[3], s[4], s[5], -s[6])
    d = InsterStictionFriction(DT_CTRL)
    d.update(*seq[0])
    out_c = [c.update(*s) for s in seq[300:]]
    out_d = [d.update(*s) for s in seq[300:]]
    assert abs(out_c[0] - out_d[0]) > 1e-3
    assert max(abs(p - q) for p, q in zip(out_c[-200:], out_d[-200:], strict=True)) < 1e-6

  def test_reset_starts_settled(self):
    fr = InsterStictionFriction(DT_CTRL)
    first = fr.update(15.0, 6.0, 4.0, 3.5, 0.2, 5.0, 80.0)
    steady = [fr.update(15.0, 6.0, 4.0, 3.5, 0.2, 5.0, 80.0) for _ in range(5)]
    assert all(abs(s - first) < 1e-9 for s in steady)
    fr.reset()
    assert fr.update(15.0, 6.0, 4.0, 3.5, 0.2, 5.0, 80.0) == first


def _make_v0(car_name, on=(), after_setup=(), cap=None):
  params = Params()
  for key in ALL_INSTER_KEYS + ("LateralJerkTorqueController", "NeuralNetworkLateralControl"):
    params.put_bool(key, key in on, block=True)
  params.put("InsterTorqueCap", 384 if cap is None else cap, block=True)
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
  # as card: the car-side switches reach opendbc through initialize_params (InsterHighTorqueLimits -> HyundaiFlagsSP.INSTER_HIGH_TORQUE)
  opendbc_sunnypilot_setup_interfaces(CarInterface, CP, CP_SP, sunnypilot_interfaces.initialize_params(params))
  CI = CarInterface(CP, CP_SP)
  sunnypilot_interfaces.setup_interfaces(CI, params)
  for key in after_setup:
    params.put_bool(key, True, block=True)
  CP_SP = convert_to_capnp(CP_SP).as_reader()
  controller = LatControlTorqueV0(CP.as_reader(), CP_SP, CI, DT_CTRL)
  controller.test_cp_sp = CP_SP
  return controller, VehicleModel(CP)


def _run(controller, VM, n=1200):
  """synthetic drive: curves at changing speed, a driver press, an inactive stretch, safety-limited steps; returns every output"""
  out = []
  angle = 0.0
  for k in range(n):
    t = k * DT_CTRL
    CS = car.CarState.new_message()
    CS.vEgo = 4.0 + 12.0 * (1.0 + math.sin(0.15 * t))
    CS.vEgoRaw = CS.vEgo + 0.05 * math.sin(7.0 * t)
    CS.aEgo = 0.3 * math.cos(0.15 * t)
    desired_curvature = 0.012 * math.sin(0.8 * t) + 0.004 * math.sin(3.1 * t)
    target = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, 0.0))
    rate = (target - angle) * 4.0
    angle += rate * DT_CTRL
    CS.steeringAngleDeg = angle
    CS.steeringRateDeg = abs(rate)
    CS.steeringPressed = 400 <= k < 430
    params = log.VehicleParameters.new_message()
    params.roll = 0.02 * math.sin(0.05 * t)
    params.angleOffsetDeg = 0.4
    VM.update_params(1.0 + 0.05 * math.sin(0.01 * t), 13.8)
    if k % 100 == 0:
      controller.update_torque_parameters(3.0 + 0.2 * math.sin(0.02 * t), 0.05, 0.09)
      controller.extension.update_limits()
    active = not (700 <= k < 760)
    torque, _, lac_log = controller.update(active, CS, VM, params, k % 97 == 0, desired_curvature, None, k % 211 == 0, 0.2)
    out.append((torque, lac_log.p, lac_log.i, lac_log.f, lac_log.output, lac_log.error, lac_log.saturated, lac_log.active,
                CS.vEgoRaw))
  return out


class TestV0InsterSwitches(OpenpilotTestCase):
  def test_switches_off_never_reach_the_inster_code(self):
    controller, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER)
    assert not controller.inster_friction and not controller.inster_ff
    assert not (controller.inster_exit or controller.inster_low_speed_p or controller.inster_high_torque)
    assert not hasattr(controller, "inster_stiction") and type(controller.pid) is not InsterLowSpeedPID
    assert controller.steer_max == 1.0
    entered = AssertionError("Inster code path entered with all switches off")
    with mock.patch.object(latcontrol_torque_v0, "ff_lat_accel_equivalent", side_effect=entered), \
         mock.patch.object(LatControlTorqueV0, "_inster_friction_raw", side_effect=entered), \
         mock.patch.object(InsterStictionFriction, "update", side_effect=entered), \
         mock.patch.object(latcontrol_torque_v0, "high_torque_steer_max", side_effect=entered), \
         mock.patch.object(latcontrol_torque_v0, "low_speed_kp_scale", side_effect=entered):
      out = _run(controller, VM, 300)
    assert all(abs(o[0]) <= 1.0 for o in out)

  def test_other_car_ignores_switches(self):
    off, VM_off = _make_v0(HYUNDAI.HYUNDAI_KONA_EV)
    on, VM_on = _make_v0(HYUNDAI.HYUNDAI_KONA_EV, on=ALL_INSTER_KEYS)
    assert not on.inster_friction and not on.inster_ff
    assert not (on.inster_exit or on.inster_low_speed_p or on.inster_high_torque)
    assert _run(off, VM_off) == _run(on, VM_on)

  def test_nnlc_on_ignores_switches(self):
    off, VM_off = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=("NeuralNetworkLateralControl",))
    on, VM_on = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=ALL_INSTER_KEYS + ("NeuralNetworkLateralControl",))
    assert not on.inster_friction and not on.inster_ff
    assert not (on.inster_exit or on.inster_low_speed_p or on.inster_high_torque)
    assert not (on.test_cp_sp.flags & HyundaiFlagsSP.INSTER_HIGH_TORQUE)
    assert _run(off, VM_off) == _run(on, VM_on)

  def test_switches_change_the_output(self):
    base = _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER))
    for on in (("InsterStictionFriction",), ("InsterSpeedFeedforward",), INSTER_KEYS):
      controller, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      assert controller.inster_friction == ("InsterStictionFriction" in on)
      assert controller.inster_ff == ("InsterSpeedFeedforward" in on)
      out = _run(controller, VM)
      assert out != base
      assert all(abs(o[0]) <= 1.0 for o in out)
      assert all(o[0] == 0.0 and not o[7] for o in out[700:760])   # inactive: no output, as v0

  def test_friction_term_resets_when_inactive(self):
    controller, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=("InsterStictionFriction",))
    _run(controller, VM, 720)                      # ends inside the inactive stretch
    assert not controller.inster_stiction.initialized and controller.inster_last_output_torque == 0.0

  def test_jerk_controller_switched_off(self):
    # car setup: an Inster switch switches the Lateral Jerk Torque Controller off (other cars keep it)
    params = Params()
    _make_v0(HYUNDAI.HYUNDAI_INSTER, on=("InsterSpeedFeedforward", "LateralJerkTorqueController"))
    assert not params.get_bool("LateralJerkTorqueController")
    _make_v0(HYUNDAI.HYUNDAI_KONA_EV, on=("InsterSpeedFeedforward", "LateralJerkTorqueController"))
    assert params.get_bool("LateralJerkTorqueController")
    # controller: with the jerk param still on (set after the car setup), v0 keeps its own output
    ja, VM_ja = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, after_setup=("LateralJerkTorqueController",))
    assert not ja.extension._jerk_aware_enabled
    plain, VM_plain = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS)
    assert _run(ja, VM_ja) == _run(plain, VM_plain)

  def test_jerk_controller_switched_off_by_step3_switches(self):
    params = Params()
    for key in STEP3_KEYS:
      _make_v0(HYUNDAI.HYUNDAI_INSTER, on=(key, "LateralJerkTorqueController"))
      assert not params.get_bool("LateralJerkTorqueController"), key
      ja, _ = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=(key,), after_setup=("LateralJerkTorqueController",))
      assert not ja.extension._jerk_aware_enabled, key


class TestInsterExitTuning(OpenpilotTestCase):
  def test_exit_table(self):
    # the 22 km/h row pulled to the 32 km/h row from 1.0 m/s^2 (where it is higher); the other rows unchanged
    assert CMD_RAW_EXIT[0] == (0.0, 23.0, 57.1, 142.4, 146.2, 190.6, 242.5, 314.7, 384.0, 384.0, 384.0)
    assert CMD_RAW_EXIT[1:] == CMD_RAW[1:]
    for v in SPEEDS:
      t = [table_torque(a, v, FF_TABLE_EXIT) for a in ACCELS]
      assert all(t1 >= t0 - 1e-12 for t0, t1 in zip(t[:-1], t[1:], strict=True))
      for a in ACCELS:
        assert ff_torque(-a, v, 2.8, FF_TABLE_EXIT) == -ff_torque(a, v, 2.8, FF_TABLE_EXIT)
        assert 0.0 <= table_torque(a, v, FF_TABLE_EXIT) <= table_torque(a, v, FF_TABLE) <= 1.0
    for a in ACCELS:
      t = [table_torque(a, v, FF_TABLE_EXIT) for v in SPEEDS]
      assert all(t1 <= t0 + 1e-12 for t0, t1 in zip(t[:-1], t[1:], strict=True))
    # the default table argument is the step-2 table (unchanged)
    assert FF_TABLE is inster_lateral._TAB

  def test_exit_parameters(self):
    fr = InsterStictionFriction(DT_CTRL, exit_tuning=True)
    assert (fr.w_hi, fr.w_lo, fr.w_bp, fr.kd) == (3.0, 6.0, (5.56, 11.11), 2.0)
    assert fr.alpha_x == DT_CTRL / (0.02 + DT_CTRL)
    base = InsterStictionFriction(DT_CTRL)
    assert (base.w_hi, base.w_lo, base.w_bp, base.kd) == (4.5, 9.0, (8.33, 13.89), 0.0)
    assert base.alpha_x == DT_CTRL / (0.05 + DT_CTRL)

  def test_release_only_damping(self):
    # releasing (x h <= 0): - kd (wheel rate - plan rate); winding in (x h > 0): the wheel rate has no effect
    for v in (3.0, 8.0, 20.0):
      for x, h in ((5.0, -80.0), (-5.0, 80.0), (3.0, 0.0)):
        plan = x                                     # _term: theta_now = theta_measured = 0, so the plan rate is x
        assert abs(_term(x, h, v, exit_tuning=True, wheel_rate=plan + 10.0) - _term(x, h, v, exit_tuning=True, wheel_rate=plan)
                   + 20.0) < 1e-9
      for x, h in ((5.0, 80.0), (-5.0, -80.0)):
        assert _term(x, h, v, exit_tuning=True, wheel_rate=37.0) == _term(x, h, v, exit_tuning=True, wheel_rate=-11.0)
    # without the exit tuning the wheel rate is not used
    assert _term(5.0, -80.0, wheel_rate=50.0) == _term(5.0, -80.0, wheel_rate=0.0)

  def test_exit_needs_both_step2_switches(self):
    base, VM_base = _make_v0(HYUNDAI.HYUNDAI_INSTER)
    alone, VM_alone = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=("InsterExitTuning",))
    assert not alone.inster_exit
    assert _run(base, VM_base, 600) == _run(alone, VM_alone, 600)
    for step2 in (("InsterStictionFriction",), ("InsterSpeedFeedforward",)):
      ref, VM_ref = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=step2)
      c, VM_c = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=step2 + ("InsterExitTuning",))
      assert not c.inster_exit
      assert _run(ref, VM_ref, 600) == _run(c, VM_c, 600)

  def test_exit_with_both_step2_switches(self):
    ref, VM_ref = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS)
    c, VM_c = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS + ("InsterExitTuning",))
    assert c.inster_exit and c.inster_stiction.kd == 2.0 and c.inster_ff_table is FF_TABLE_EXIT
    out, out_ref = _run(c, VM_c), _run(ref, VM_ref)
    assert out != out_ref
    assert all(abs(o[0]) <= 1.0 for o in out)
    assert all(o[0] == 0.0 and not o[7] for o in out[700:760])


  def test_wheel_rate_sign_end_to_end(self):
    # v0 with the exit tuning, through update(): the wheel held to the right, then the plan unwinds and the steering angle (+ left)
    # turns left at 100 deg/s: the term gets a wheel rate of -100 deg/s (+ right, over the last 4 steps), and while the term releases
    # its damping pushes against the wheel's motion relative to the plan, - 2.0 raw per deg/s
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS + ("InsterExitTuning",))
    seen = []
    orig = c.inster_stiction.update

    def rec(*a):
      v, th_f, th_n, th_m, ld, rate, h, wheel_rate = a
      plan_rate = (th_f - th_n) / (ld if ld > DT_CTRL else DT_CTRL)
      probe = copy.deepcopy(c.inster_stiction.__dict__)
      t = orig(*a)
      ref = InsterStictionFriction(DT_CTRL, exit_tuning=True)
      ref.__dict__.update({k: val for k, val in probe.items() if k != 'update'})
      t_no_damping = InsterStictionFriction.update(ref, v, th_f, th_n, th_m, ld, rate, h, plan_rate)
      seen.append((wheel_rate, plan_rate, t - t_no_damping))
      return t
    c.inster_stiction.update = rec
    angle = -20.0
    for k in range(160):
      CS = car.CarState.new_message()
      CS.vEgo = CS.vEgoRaw = 10.0
      moving = 100 <= k < 120
      if moving:
        angle += 1.0                                  # 1 deg per 10 ms to the left
      CS.steeringAngleDeg = angle
      CS.steeringRateDeg = 100.0 if moving else 0.0
      desired_curvature = 0.01 if k < 100 else 0.0    # + right: held to the right, then the plan unwinds
      c.update(True, CS, VM, log.VehicleParameters.new_message(), False, desired_curvature, None, False, 0.2)
    assert all(w == 0.0 for w, _, _ in seen[:100])
    assert abs(seen[100][0] + 25.0) < 1e-9 and abs(seen[101][0] + 50.0) < 1e-9
    assert all(abs(w + 100.0) < 1e-9 for w, _, _ in seen[103:120])  # (angle(k) - angle(k-4)) / 0.04 s, sign flipped to + right
    damped = [(w - plan, d) for w, plan, d in seen if d != 0.0]
    assert len(damped) > 5
    assert all(rel * d < 0.0 for rel, d in damped)                   # opposes the motion relative to the plan
    assert all(abs(d + 2.0 * rel) < 1e-6 for rel, d in damped)


class TestInsterLowSpeedP(OpenpilotTestCase):
  def test_scale(self):
    for v, f in ((0.0, 0.75), (5.0, 0.75), (5.56, 0.75), ((5.56 + 8.33) / 2, 0.875), (8.33, 1.0), (30.0, 1.0)):
      assert abs(low_speed_kp_scale(v) - f) < 1e-12, v
    s = [low_speed_kp_scale(v / 10) for v in range(400)]
    assert all(b >= a for a, b in zip(s[:-1], s[1:], strict=True))

  def test_pid_gain(self):
    for on in ((), INSTER_KEYS):
      c, _ = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on + ("InsterLowSpeedP",))
      plain, _ = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      assert c.inster_low_speed_p and type(c.pid) is InsterLowSpeedPID
      for v in (0.5, 3.0, 5.56, 7.0, 8.33, 15.0, 30.0):
        c.pid.speed = plain.pid.speed = v
        assert c.pid.k_p == plain.pid.k_p * low_speed_kp_scale(v)
        assert c.pid.k_i == plain.pid.k_i
      assert (c.pid.pos_limit, c.pid.neg_limit) == (plain.pid.pos_limit, plain.pid.neg_limit)
      c2, VM_c2 = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on + ("InsterLowSpeedP",))
      p2, VM_p2 = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      out_c2, out_p2 = _run(c2, VM_c2, 3500), _run(p2, VM_p2, 3500)       # the synthetic drive is below 30 km/h around 25-38 s
      fast = [k for k, o in enumerate(out_p2) if o[8] > 8.5]
      first_slow = next(k for k, o in enumerate(out_p2) if o[8] < 8.33)
      assert out_c2[:first_slow] == out_p2[:first_slow] and fast[0] == 0
      assert out_c2[first_slow] != out_p2[first_slow]


class TestInsterHighTorque(OpenpilotTestCase):
  def test_helpers(self):
    for v in (0.0, 5.0, 7.0, 9.0, 11.11, 30.0):
      assert high_torque_steer_max(v) == inster_steer_max(v) / 384.0
      assert 1.0 < high_torque_steer_max(v) <= 450.0 / 384.0
    assert MAX_LATERAL_ACCEL_NO_ROLL == 3.0
    assert set(INSTER_SWITCHES) == set(ALL_INSTER_KEYS)

  def test_flag_only_for_the_inster_with_the_switch(self):
    for car_name, on, expected in ((HYUNDAI.HYUNDAI_INSTER, (), False), (HYUNDAI.HYUNDAI_INSTER, ("InsterHighTorqueLimits",), True),
                                   (HYUNDAI.HYUNDAI_INSTER, ("InsterHighTorqueLimits", "NeuralNetworkLateralControl"), False),
                                   (HYUNDAI.HYUNDAI_KONA_EV, ("InsterHighTorqueLimits",), False)):
      c, _ = _make_v0(car_name, on=on)
      assert bool(c.test_cp_sp.flags & HyundaiFlagsSP.INSTER_HIGH_TORQUE) == expected, (car_name, on)
      assert c.inster_high_torque == expected
      CP = interfaces[car_name].get_non_essential_params(car_name)
      assert is_inster_high_torque(CP, c.test_cp_sp) == expected
      assert max_lateral_accel_no_roll(c, MAX_LATERAL_ACCEL_NO_ROLL) == (3.3 if expected else 3.0)

  def test_ceiling_only_with_v0_running(self):
    # controlsd's choice of the lateral controller (ControlsExt.initialize_lateral_control): v0 (Enforce Torque Lateral Control off, or
    # on with tune v0.0) gets 3.3 with the flag; tune v1.0 (the upstream torque controller, output within 1.0) keeps 3.0
    from types import SimpleNamespace
    from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque as LatControlTorqueV1
    from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt
    params = Params()
    c, _ = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=("InsterHighTorqueLimits",))
    CarInterface = interfaces[HYUNDAI.HYUNDAI_INSTER]
    CP = CarInterface.get_non_essential_params(HYUNDAI.HYUNDAI_INSTER)
    CI = CarInterface(CP, CarInterface.get_non_essential_params_sp(CP, HYUNDAI.HYUNDAI_INSTER))
    try:
      for enforce, tune, expect_v0 in ((False, 0.0, True), (True, 0.0, True), (True, 1.0, False)):
        params.put_bool("EnforceTorqueControl", enforce, block=True)
        params.put("TorqueControlTune", tune, block=True)
        ext = SimpleNamespace(CP=CP.as_reader(), CP_SP=c.test_cp_sp, params=params)
        upstream = LatControlTorqueV1(CP.as_reader(), c.test_cp_sp, CI, DT_CTRL)
        LaC = ControlsExt.initialize_lateral_control(ext, upstream, CI, DT_CTRL)
        assert isinstance(LaC, LatControlTorqueV0) == expect_v0, (enforce, tune)
        assert max_lateral_accel_no_roll(LaC, MAX_LATERAL_ACCEL_NO_ROLL) == (3.3 if expect_v0 else 3.0), (enforce, tune)
        if not expect_v0:
          assert LaC is upstream and LaC.steer_max == 1.0
    finally:
      params.put_bool("EnforceTorqueControl", False, block=True)
      params.put("TorqueControlTune", 0.0, block=True)

  def test_initialize_params(self):
    params = Params()
    for switch, nnlc in ((False, False), (True, False), (True, True), (False, True)):
      params.put_bool("InsterHighTorqueLimits", switch, block=True)
      params.put_bool("NeuralNetworkLateralControl", nnlc, block=True)
      entries = {k: v for d in sunnypilot_interfaces.initialize_params(params) for k, v in d.items()}
      assert entries["InsterHighTorqueLimits"] == (switch and not nnlc)

  def test_output_follows_the_speed_table(self):
    for on in (("InsterHighTorqueLimits",), ALL_INSTER_KEYS):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on, cap=415)
      assert c.inster_high_torque and c.inster_torque_cap == 415
      c.inster_torque_cap = 450                   # test only, beyond the cap's range: the speed table still applies underneath
      out = _run(c, VM)
      limits = [inster_steer_max(o[8]) / 384.0 for o in out]
      assert all(abs(o[0]) <= lim + 1e-9 for o, lim in zip(out, limits, strict=True))
      assert max(abs(o[0]) for o in out) > 1.0                                  # it does go above 384 raw on this sequence
      assert any(abs(abs(o[0]) - lim) < 1e-9 and lim < 450.0 / 384.0 for o, lim in zip(out, limits, strict=True))
      # saturation is judged against the speed-dependent limit: never at |output| <= 1 when the limit is higher
      assert c.steer_max == limits[-1]
    # switch off: the same sequence stays within 1.0
    off, VM_off = _make_v0(HYUNDAI.HYUNDAI_INSTER)
    assert max(abs(o[0]) for o in _run(off, VM_off)) <= 1.0

  def test_clip_curvature_ceiling(self):
    rng = random.Random(11)
    for _ in range(2000):
      v, prev, new, roll = rng.uniform(0, 40), rng.uniform(-0.05, 0.05), rng.uniform(-0.25, 0.25), rng.uniform(-0.1, 0.1)
      assert clip_curvature(v, prev, new, roll) == clip_curvature(v, prev, new, roll, MAX_LATERAL_ACCEL_NO_ROLL)
    # at 20 m/s and no roll: 3.0 / 400 vs 3.3 / 400
    c30, lim30 = clip_curvature(20.0, 0.0085, 0.0086, 0.0)
    c33, lim33 = clip_curvature(20.0, 0.0085, 0.0086, 0.0, 3.3)
    assert lim30 and abs(c30 - 3.0 / 400) < 1e-12
    assert lim33 and abs(c33 - 3.3 / 400) < 1e-12


class TestInsterTorqueCap(OpenpilotTestCase):
  def test_parse_and_clamp(self):
    for value, cap in ((None, 384), ("abc", 384), (float("nan"), 384), (300, 384), (384, 384), (395, 395), ("405", 405), (400.4, 400),
                       (415, 415), (416, 415), (1000, 415)):
      assert inster_torque_cap(value) == cap, value
    assert (TORQUE_CAP_MIN, TORQUE_CAP_MAX, TORQUE_CAP_DEFAULT) == (384, 415, 384)
    assert min(inster_steer_max(v / 10) for v in range(500)) >= TORQUE_CAP_MAX + 15   # the car side / panda table is above any cap

  def test_min_with_the_table(self):
    for v in (0.0, 5.0, 8.0, 10.0, 12.0, 30.0):
      assert high_torque_steer_max(v) == inster_steer_max(v) / 384.0
      for cap in (384, 400, 415):
        assert high_torque_steer_max(v, cap) == min(inster_steer_max(v), cap) / 384.0 == cap / 384.0
      assert high_torque_steer_max(v, 445) == min(inster_steer_max(v), 445) / 384.0

  def test_default_cap_never_above_384(self):
    # switch on with the cap at its default 384: the request never exceeds 384 raw (also through the car-side rounding)
    for on in (("InsterHighTorqueLimits",), ALL_INSTER_KEYS):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      assert c.inster_high_torque and c.inster_torque_cap == 384
      out = _run(c, VM)
      assert max(int(round(abs(o[0]) * 384)) for o in out) == 384
      assert all(abs(o[0]) <= 1.0 + 1e-12 for o in out)
      assert c.steer_max == 1.0

  def test_cap_limits_output_and_saturation(self):
    for cap in (395, 415):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=("InsterHighTorqueLimits",), cap=cap)
      out = _run(c, VM)
      assert max(int(round(abs(o[0]) * 384)) for o in out) == cap       # reaches the cap, never more
      assert c.steer_max == cap / 384.0
      lim = c.pid.pos_limit / c.torque_params.latAccelFactor
      assert abs(lim - cap / 384.0) < 1e-6                               # PID limits follow the cap
      # saturation (steerSaturated) needs |output| at the cap, not at 1.0
      sat = [o for o in out if o[6]]
      assert sat and all(abs(abs(o[4]) - cap / 384.0) < 0.05 for o in sat[:1])

  def test_live_change(self):
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=("InsterHighTorqueLimits",), cap=415)
    out1 = _run(c, VM, 600)
    assert max(int(round(abs(o[0]) * 384)) for o in out1) == 415
    c.set_inster_torque_cap(390)                                          # lower: applies at the next update
    out2 = _run(c, VM, 600)
    assert max(int(round(abs(o[0]) * 384)) for o in out2) == 390 and c.steer_max == 390 / 384.0
    c.set_inster_torque_cap(999)                                          # clamped
    assert c.inster_torque_cap == 415
    off, _ = _make_v0(HYUNDAI.HYUNDAI_INSTER, cap=415)
    off.set_inster_torque_cap(400)                                        # ignored with the switch off
    assert not hasattr(off, "inster_torque_cap")

  def test_switch_off_ignores_the_cap(self):
    a, VM_a = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, cap=384)
    b, VM_b = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, cap=415)
    assert _run(a, VM_a) == _run(b, VM_b)
    k, VM_k = _make_v0(HYUNDAI.HYUNDAI_KONA_EV, on=("InsterHighTorqueLimits",), cap=415)
    assert not k.inster_high_torque and max(abs(o[0]) for o in _run(k, VM_k)) <= 1.0

  def test_controlsd_params_refresh(self):
    # controlsd's existing params refresh (ControlsExt.get_params_sp, every PARAMS_UPDATE_PERIOD) passes the cap to v0
    from types import SimpleNamespace
    from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt
    params = Params()
    c, _ = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=("InsterHighTorqueLimits",), cap=384)
    CP = interfaces[HYUNDAI.HYUNDAI_INSTER].get_non_essential_params(HYUNDAI.HYUNDAI_INSTER).as_reader()
    ext = SimpleNamespace(CP=CP, params=params, LaC=c, _param_update_time=0.0, lat_delay=0.2,
                          blinker_pause_lateral=SimpleNamespace(get_params=lambda: None))
    sm = {"lateralDelay": SimpleNamespace(lateralDelay=0.2)}
    params.put("InsterTorqueCap", 401, block=True)
    ControlsExt.get_params_sp(ext, sm)
    assert c.inster_torque_cap == 401
    params.put("InsterTorqueCap", 384, block=True)
    ControlsExt.get_params_sp(ext, sm)                                    # within the refresh period: not read again
    assert c.inster_torque_cap == 401
    ext._param_update_time = 0.0
    ControlsExt.get_params_sp(ext, sm)
    assert c.inster_torque_cap == 384

