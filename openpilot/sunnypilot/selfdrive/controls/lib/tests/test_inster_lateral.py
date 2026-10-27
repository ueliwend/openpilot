"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import random
from unittest import mock

import numpy as np

from openpilot.cereal import log
from opendbc.car.structs import car
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI, INSTER_STEER_FULL_SCALE, INSTER_STEER_MAX
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.params import Params
from openpilot.common.pid import PIDController
from openpilot.common.realtime import DT_CTRL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.sunnypilot.selfdrive.car import interfaces as sunnypilot_interfaces
from openpilot.sunnypilot.selfdrive.controls.lib import inster_lateral, latcontrol_torque_v0
from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import ANGLE_PI_G6_KI, ANGLE_PI_G6_KP, ANGLE_PI_G6_LOG_VERSION, \
                                                                       ANGLE_PI_G6_SWITCH
from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import A_NODES, ANGLE_PI_KI, ANGLE_PI_KP, ANGLE_PI_LOG_VERSION, CMD_RAW, F0, \
                                                                       SCALE_CLIP, STEER_MAX_RAW, V_NODES, InsterAnglePI, InsterStictionFriction, \
                                                                       angle_error, angle_pi_gain, ff_lat_accel_equivalent, ff_torque, \
                                                                       table_torque
from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import CRAWL_PER_M_RELEASE, CRAWL_PLAN_RELEASE_SPEED, CRAWL_PLAN_SPEED, \
                                                                       CRAWL_STRAIGHTEN_LENGTH, CRAWL_STRAIGHTEN_PER_M, CRAWL_TURN_IN_PER_M, \
                                                                       crawl_plan_curvature
from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorque as LatControlTorqueV0

STEP2_KEYS = ("InsterStictionFriction", "InsterSpeedFeedforward")
ANGLE_PI_KEY = "InsterAngleErrorPI"
INSTER_KEYS = STEP2_KEYS + (ANGLE_PI_KEY,)
SPEEDS = [0.0, 2.0, 5.0, 6.1] + list(np.linspace(6.11, 30.83, 60)) + [31.0, 35.0, 45.0]
ACCELS = list(np.linspace(0.0, 4.0, 161))


def _term(x, h, v=20.0, rate=0.0, lat_delay=0.2):
  """the term with settled filters (first update after a reset) for intended wheel rate x (deg/s) and held torque h (raw)"""
  fr = InsterStictionFriction(DT_CTRL)
  return fr.update(v, x * lat_delay, 0.0, 0.0, lat_delay, rate, h)


class TestInsterTable(OpenpilotTestCase):
  def test_full_scale_unchanged(self):
    # the table and the friction amplitudes are in raw units of the full scale 384 (actuators.torque 1.0): review them before
    # changing INSTER_STEER_FULL_SCALE; the cap INSTER_STEER_MAX (409) does not change their meaning
    assert STEER_MAX_RAW == INSTER_STEER_FULL_SCALE == 384
    assert INSTER_STEER_MAX == 409

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
          assert abs(ff_torque(a, v, f)) <= 409 / 384
          assert abs(ff_torque(-a, v, f)) <= 409 / 384

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

  def test_smooth_in_lateral_accel(self):
    # the rule the table is built and refit by (t2_table409): below the cap, neighbouring slopes in a differ by at most x2.5 up and
    # x0.4 down (0.01 tolerance: the cells are rounded to 0.1 raw)
    for row in CMD_RAW:
      s = [(row[i + 1] - row[i]) / (A_NODES[i + 1] - A_NODES[i]) for i in range(len(row) - 1)]
      for i in range(len(s) - 1):
        if row[i + 2] < INSTER_STEER_MAX and s[i] > 0.0:
          assert 0.4 - 0.01 <= s[i + 1] / s[i] <= 2.5 + 0.01, (row, i)

  def test_logged_learner_range_inside_scale_clip(self):
    # learned factor logged on the table drives 22-26 Oct 2026 (torqued, per minute): filtered 2.679 .. 3.064 (what controlsd passes),
    # raw 2.551 .. 3.123; the clip must not cut any of them, so the table follows the learner one to one in that range
    for f in (2.551, 2.679, 3.064, 3.123):
      assert SCALE_CLIP[0] < F0 / f < SCALE_CLIP[1]

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


def _make_v0(car_name, on=(), after_setup=()):
  params = Params()
  for key in INSTER_KEYS + (G6_KEY, "LateralJerkTorqueController", "NeuralNetworkLateralControl"):
    params.put_bool(key, key in on, block=True)
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
  CI = CarInterface(CP, CP_SP)
  sunnypilot_interfaces.setup_interfaces(CI, params)
  for key in after_setup:
    params.put_bool(key, True, block=True)
  CP_SP = convert_to_capnp(CP_SP)
  return LatControlTorqueV0(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL), VehicleModel(CP)


def _run(controller, VM, n=1200, extension_model=None):
  """synthetic drive: curves at changing speed, a driver press, an inactive stretch, safety-limited steps; returns every output"""
  out = []
  angle = 0.0
  for k in range(n):
    t = k * DT_CTRL
    CS = car.CarState.new_message()
    CS.vEgo = 4.0 + 12.0 * (1.0 + math.sin(0.15 * t))
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
    if extension_model is not None:
      # as controlsd: with torque parameters in use, extension.update_limits() runs every step (sets the shared PID to +-steer_max)
      controller.extension.update_limits()
      controller.extension.update_model_v2(extension_model(desired_curvature * CS.vEgo ** 2))
      controller.extension.update_lateral_lag(0.2)
    torque, _, lac_log = controller.update(active, CS, VM, params, k % 97 == 0, desired_curvature, None, k % 211 == 0, 0.2)
    out.append((torque, lac_log.p, lac_log.i, lac_log.f, lac_log.output, lac_log.error, lac_log.saturated, lac_log.active))
  return out


class TestV0InsterSwitches(OpenpilotTestCase):
  def test_switches_off_never_reach_the_inster_code(self):
    controller, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER)
    assert not controller.inster_friction and not controller.inster_ff and not controller.inster_angle_pi
    assert not hasattr(controller, "inster_stiction")
    assert type(controller.pid) is PIDController
    entered = AssertionError("Inster code path entered with all switches off")
    with mock.patch.object(latcontrol_torque_v0, "ff_lat_accel_equivalent", side_effect=entered), \
         mock.patch.object(LatControlTorqueV0, "_inster_friction_raw", side_effect=entered), \
         mock.patch.object(InsterStictionFriction, "update", side_effect=entered), \
         mock.patch.object(latcontrol_torque_v0, "angle_error", side_effect=entered), \
         mock.patch.object(InsterAnglePI, "update", side_effect=entered):
      _run(controller, VM, 300)

  def test_other_car_ignores_switches(self):
    off, VM_off = _make_v0(HYUNDAI.HYUNDAI_KONA_EV)
    on, VM_on = _make_v0(HYUNDAI.HYUNDAI_KONA_EV, on=INSTER_KEYS)
    assert not on.inster_friction and not on.inster_ff and not on.inster_angle_pi
    assert type(on.pid) is PIDController
    assert _run(off, VM_off) == _run(on, VM_on)

  def test_nnlc_on_ignores_switches(self):
    off, VM_off = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=("NeuralNetworkLateralControl",))
    on, VM_on = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS + ("NeuralNetworkLateralControl",))
    assert not on.inster_friction and not on.inster_ff and not on.inster_angle_pi
    assert type(on.pid) is PIDController
    assert _run(off, VM_off) == _run(on, VM_on)

  def test_switches_change_the_output(self):
    base = _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER))
    for on in (("InsterStictionFriction",), ("InsterSpeedFeedforward",), STEP2_KEYS, (ANGLE_PI_KEY,), INSTER_KEYS):
      controller, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      assert controller.inster_friction == ("InsterStictionFriction" in on)
      assert controller.inster_ff == ("InsterSpeedFeedforward" in on)
      assert controller.inster_angle_pi == (ANGLE_PI_KEY in on)
      out = _run(controller, VM)
      assert out != base
      assert all(abs(o[0]) <= 409 / 384 + 1e-12 for o in out)   # the Inster cap (409 raw, 1.0 = 384)
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
    _make_v0(HYUNDAI.HYUNDAI_INSTER, on=(ANGLE_PI_KEY, "LateralJerkTorqueController"))
    assert not params.get_bool("LateralJerkTorqueController")
    _make_v0(HYUNDAI.HYUNDAI_KONA_EV, on=("InsterSpeedFeedforward", "LateralJerkTorqueController"))
    assert params.get_bool("LateralJerkTorqueController")
    # controller: with the jerk param still on (set after the car setup), v0 keeps its own output
    ja, VM_ja = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, after_setup=("LateralJerkTorqueController",))
    assert not ja.extension._jerk_aware_enabled
    plain, VM_plain = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS)
    assert _run(ja, VM_ja) == _run(plain, VM_plain)


class _MV:
  """minimal modelV2 stand-in for the jerk-aware / NNLC look-ahead (acceleration.y, orientation.x)"""
  def __init__(self, ay):
    self.acceleration = type('X', (), {})()
    self.acceleration.y = [ay] * 33
    self.orientation = type('X', (), {})()
    self.orientation.x = [0.0] * 33
    self.orientation.y = [0.0] * 33


CAP = 409 / 384


class TestInsterSteerCap(OpenpilotTestCase):
  """The Inster's fixed cap of 409 raw with 1.0 = 384 raw: every lateral controller that can run on the Inster limits its output to
  409 / 384, other cars keep 1.0."""

  def test_steer_max(self):
    assert inster_lateral.INSTER_STEER_MAX_FRACTION == CAP
    for car_name, expected in ((HYUNDAI.HYUNDAI_INSTER, CAP), (HYUNDAI.HYUNDAI_KONA_EV, 1.0), (TOYOTA.TOYOTA_RAV4, 1.0)):
      CP = interfaces[car_name].get_non_essential_params(car_name)
      assert inster_lateral.lateral_steer_max(CP) == expected
    assert inster_lateral.lateral_steer_max(None) == 1.0

  def _check_capped(self, out, reach=True):
    raw = [int(round(abs(o[0]) * 384)) for o in out]
    assert max(abs(o[0]) for o in out) <= CAP + 1e-12
    assert max(raw) <= 409
    if reach:
      assert max(raw) == 409                                            # the sequence does drive it to the cap

  def test_v0_cap_and_saturation(self):
    for on in ((), ("InsterStictionFriction",), ("InsterSpeedFeedforward",), STEP2_KEYS, (ANGLE_PI_KEY,), INSTER_KEYS):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      assert c.steer_max == CAP
      out = _run(c, VM)
      self._check_capped(out)
      lim = c.pid.pos_limit / c.torque_params.latAccelFactor
      assert abs(lim - CAP) < 1e-6                                      # PID limits at 409 / 384
      sat = [o for o in out if o[6]]
      assert sat and all(abs(o[4]) > 1.0 for o in sat)                  # steerSaturated only when the output is above 1.0, at the cap

  def test_v0_other_car_keeps_one(self):
    c, VM = _make_v0(HYUNDAI.HYUNDAI_KONA_EV, on=INSTER_KEYS)
    assert c.steer_max == 1.0
    assert max(abs(o[0]) for o in _run(c, VM)) <= 1.0

  def test_upstream_torque_controller_v1(self):
    from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque as LatControlTorqueV1
    for car_name, cap in ((HYUNDAI.HYUNDAI_INSTER, CAP), (HYUNDAI.HYUNDAI_KONA_EV, 1.0)):
      params = Params()
      for key in INSTER_KEYS + (G6_KEY, "LateralJerkTorqueController", "NeuralNetworkLateralControl"):
        params.put_bool(key, False, block=True)
      CarInterface = interfaces[car_name]
      CP = CarInterface.get_non_essential_params(car_name)
      CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
      CI = CarInterface(CP, CP_SP)
      c = LatControlTorqueV1(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)
      assert c.steer_max == cap
      out = _run(c, VehicleModel(CP))
      assert max(abs(o[0]) for o in out) <= cap + 1e-12
      assert max(abs(o[0]) for o in out) > cap - 1e-6                   # reaches its limit on this sequence

  def test_extensions_jerk_aware_and_nnlc(self):
    # the Lateral Jerk Torque Controller and NNLC take the limits from lac_torque.steer_max: 409 / 384 on the Inster as well
    # (once controlsd has called extension.update_limits(), i.e. with torque parameters in use; the car side caps at 409 regardless)
    for on in (("LateralJerkTorqueController",), ("NeuralNetworkLateralControl",)):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      assert c.steer_max == CAP
      out = _run(c, VM, extension_model=_MV)
      assert max(abs(o[0]) for o in out) <= CAP + 1e-9, on
      assert max(abs(o[0]) for o in out) > 1.0, on

  def test_extensions_pid_in_torque_space_from_first_step(self):
    # without controlsd ever calling extension.update_limits() (torque parameters not in use, useParams off), the jerk-aware / NNLC
    # extensions' PID has torque-space limits +-409/384 from the first step on the Inster, so its anti-windup works there
    for on in (("LateralJerkTorqueController",), ("NeuralNetworkLateralControl",)):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      assert (c.pid.pos_limit, c.pid.neg_limit) == ((CAP, -CAP) if on[0] == "LateralJerkTorqueController" else (c.pid.pos_limit, c.pid.neg_limit))
      for k in range(400):
        CS = car.CarState.new_message()
        CS.vEgo = 12.0
        CS.steeringAngleDeg = -60.0 if k > 50 else 0.0
        c.extension.update_model_v2(_MV(4.0))
        c.extension.update_lateral_lag(0.2)
        c.update(True, CS, VM, log.VehicleParameters.new_message(), False, 0.03, None, False, 0.2)
        if c.extension._jerk_aware_enabled or c.extension._nnlc_enabled:
          assert (c.pid.pos_limit, c.pid.neg_limit) == (CAP, -CAP), (on, k)
          assert abs(c.pid.i) <= CAP + 1e-12, (on, k, c.pid.i)    # integrator bounded by the torque-space limit
      assert c.extension._jerk_aware_enabled or c.extension._nnlc_enabled, on

  def test_extensions_held_to_cap_with_stale_limits(self):
    # even before controlsd has set the shared PID to +-steer_max (limits still in lateral-acceleration units), the extensions'
    # output is held to 409 / 384 on the Inster; other cars are unchanged
    for car_name, cap in ((HYUNDAI.HYUNDAI_INSTER, CAP), (HYUNDAI.HYUNDAI_KONA_EV, None)):
      for on in (("LateralJerkTorqueController",), ("NeuralNetworkLateralControl",)):
        c, VM = _make_v0(car_name, on=on)
        assert c.extension._inster == (cap is not None)
        out = []
        for k in range(600):
          CS = car.CarState.new_message()
          CS.vEgo = 12.0
          CS.steeringAngleDeg = -60.0 if k > 50 else 0.0
          c.extension.update_model_v2(_MV(4.0))
          c.extension.update_lateral_lag(0.2)
          tq, _, lac_log = c.update(True, CS, VM, log.VehicleParameters.new_message(), False, 0.03, None, False, 0.2)
          out.append(abs(tq))
        if cap is not None:
          assert max(out) <= cap + 1e-12 and max(out) > 1.0, (car_name, on, max(out))

  def test_feedforward_and_friction_bounded(self):
    # neither step-2 contribution alone can ask for more than the cap: no table cell above INSTER_STEER_MAX (the table reaches 409
    # at high lateral acceleration), the friction term at most 200 raw; the controller's limit decides the rest
    assert max(c for row in CMD_RAW for c in row) == INSTER_STEER_MAX
    assert all(0.0 <= c <= INSTER_STEER_MAX for row in CMD_RAW for c in row)
    for v in SPEEDS:
      for a in ACCELS + [10.0]:
        for f in (1.0, 2.8, 3.7):
          assert abs(ff_torque(a, v, f)) * STEER_MAX_RAW <= INSTER_STEER_MAX + 1e-9
    assert inster_lateral.FRICTION_MAX < INSTER_STEER_MAX

  def test_ceiling(self):
    from openpilot.selfdrive.controls.lib.drive_helpers import MAX_LATERAL_ACCEL_NO_ROLL
    assert inster_lateral.INSTER_MAX_LATERAL_ACCEL_NO_ROLL == 3.3
    for car_name, expected in ((HYUNDAI.HYUNDAI_INSTER, inster_lateral.INSTER_MAX_LATERAL_ACCEL_NO_ROLL), (HYUNDAI.HYUNDAI_KONA_EV, 3.0)):
      CP = interfaces[car_name].get_non_essential_params(car_name)
      assert inster_lateral.max_lateral_accel_no_roll(CP, MAX_LATERAL_ACCEL_NO_ROLL) == expected

  def test_removed_switches(self):
    # the step-3 params are gone: unknown to Params (a stale file on a device is never read); the two step-2 switches and F remain
    params = Params()
    keys = {k.decode() if isinstance(k, bytes) else k for k in params.all_keys()}
    for key in ("InsterExitTuning", "InsterLowSpeedP", "InsterHighTorqueLimits", "InsterTorqueCap"):
      assert key not in keys
    assert inster_lateral.INSTER_SWITCHES == INSTER_KEYS
    assert set(INSTER_KEYS) <= keys
    assert G6_KEY in keys and G6_KEY not in inster_lateral.INSTER_SWITCHES          # G6: a sub-switch of InsterAngleErrorPI


# G2 (27 Oct 2026; replaces F's gains): the Nelder-Mead result's coefficients at full precision, the 6-significant-digit values the
# car code uses, and the anchor values at 0 / 10 / 55 / 115 / 160 km/h
G2_KP_SEARCH = (3.17476749408284, 0.6064020771465092, -0.011953117949879298)
G2_KI_SEARCH = (0.93525, 0.32978182120404387, -0.006770430057410106)
G2_KP = (3.17477, 0.606402, -0.0119531)
G2_KI = (0.93525, 0.329782, -0.00677043)
G2_ANCHOR_KMH = (0.0, 10.0, 55.0, 115.0, 160.0)
G2_KP_ANCHOR = (3.17477, 4.76699, 9.64927, 10.34843, 6.51491)
G2_KI_ANCHOR = (0.93525, 1.79907, 4.3933, 4.56108, 2.21854)
F_KP = (4.3447, 0.40366, -0.0072954)          # F's gains until 26 Oct 2026 (torqueState.version 100 / 101)
F_KI = (0.93525, 0.27948, -0.0057456)


def _cs(v, angle=0.0, pressed=False, rate=0.0):
  CS = car.CarState.new_message()
  CS.vEgo = v
  CS.steeringAngleDeg = angle
  CS.steeringRateDeg = rate
  CS.steeringPressed = pressed
  return CS


def _vp(roll=0.0, offset=0.0):
  params = log.VehicleParameters.new_message()
  params.roll = roll
  params.angleOffsetDeg = offset
  return params


class TestInsterAnglePI(OpenpilotTestCase):
  """switch InsterAngleErrorPI: v0's P and I replaced by "F" on the steering-wheel angle error"""

  def test_gains(self):
    assert ANGLE_PI_KP == G2_KP and ANGLE_PI_KI == G2_KI
    assert G2_KP == tuple(float(f"{c:.6g}") for c in G2_KP_SEARCH) and G2_KI == tuple(float(f"{c:.6g}") for c in G2_KI_SEARCH)
    assert (ANGLE_PI_KP, ANGLE_PI_KI) != (F_KP, F_KI)
    # the G2 table: kp 4.77 / 9.65 / 10.35 / 6.51, ki 1.80 / 4.39 / 4.56 / 2.22 at 10 / 55 / 115 / 160 km/h (no hold above 130 km/h)
    for kmh, kp, ki in zip(G2_ANCHOR_KMH, G2_KP_ANCHOR, G2_KI_ANCHOR, strict=True):
      assert abs(angle_pi_gain(ANGLE_PI_KP, kmh / 3.6) - kp) < 1e-5, kmh
      assert abs(angle_pi_gain(ANGLE_PI_KI, kmh / 3.6) - ki) < 1e-5, kmh
    # rounding to 6 significant digits: the gains stay within 1e-5 (relative) of the search's up to 160 km/h
    for v in np.linspace(0.0, 160 / 3.6, 1601):
      assert abs(angle_pi_gain(ANGLE_PI_KP, v) / angle_pi_gain(G2_KP_SEARCH, v) - 1.0) < 1e-5, v
      assert abs(angle_pi_gain(ANGLE_PI_KI, v) / angle_pi_gain(G2_KI_SEARCH, v) - 1.0) < 1e-5, v
    # never negative from 0 to 200 km/h; positive below the roots (ki 185.0 km/h, kp 199.9 km/h), clipped at 0 beyond them
    for v in np.linspace(0.0, 200 / 3.6, 2001):
      assert angle_pi_gain(ANGLE_PI_KP, v) >= 0.0 and angle_pi_gain(ANGLE_PI_KI, v) >= 0.0
      if v < 185 / 3.6:
        assert angle_pi_gain(ANGLE_PI_KP, v) > 0.0 and angle_pi_gain(ANGLE_PI_KI, v) > 0.0
    assert angle_pi_gain(ANGLE_PI_KP, 199.8 / 3.6) > 0.0 and angle_pi_gain(ANGLE_PI_KP, 200 / 3.6) == 0.0
    assert angle_pi_gain(ANGLE_PI_KI, 55.0) == 0.0 and angle_pi_gain(ANGLE_PI_KP, 55.0) > 0.0
    assert angle_pi_gain(ANGLE_PI_KP, 70.0) == 0.0
    pi = InsterAnglePI(rate=1 / DT_CTRL)
    for v in (0.0, 4.2, 15.3, 31.9):
      pi.speed = v
      assert pi.k_p == angle_pi_gain(G2_KP, v) and pi.k_i == angle_pi_gain(G2_KI, v) and pi.k_d == 0.0

  def test_angle_error_is_wheel_angle_error(self):
    # e = delta_plan - delta_meas (rad, + right): delta_plan from the inverse vehicle model incl. steer ratio, slip factor and roll,
    # delta_meas = steering angle - learned offset; equal to R(v) x (desired - v0's measured curvature)
    CP = interfaces[HYUNDAI.HYUNDAI_INSTER].get_non_essential_params(HYUNDAI.HYUNDAI_INSTER)
    VM = VehicleModel(CP)
    rng = random.Random(11)
    worst = 0.0
    for _ in range(5000):
      VM.update_params(rng.uniform(0.6, 1.4), rng.uniform(11.0, 17.0))
      v = rng.choice([0.0, rng.uniform(0.0, 3.0), rng.uniform(0.0, 45.0)])
      roll = rng.uniform(-0.1, 0.1)
      offset = rng.uniform(-3.0, 3.0)
      angle = rng.uniform(-400.0, 400.0)
      desired_curvature = rng.uniform(-0.2, 0.2)
      measured_curvature = -VM.calc_curvature(math.radians(angle - offset), v, roll)      # as v0
      delta_plan = -VM.get_steer_from_curvature(-desired_curvature, v, roll)              # + right
      delta_meas = -math.radians(angle - offset)                                          # + right
      expected = delta_plan - delta_meas
      e = angle_error(VM, v, desired_curvature, measured_curvature)
      worst = max(worst, abs(e - expected) / max(1.0, abs(expected)))
    assert worst < 1e-12, worst
    # sign: wheel 10 deg left of a straight plan -> the error asks for right (+)
    VM.update_params(1.0, 14.0)
    mc = -VM.calc_curvature(math.radians(10.0), 20.0, 0.0)
    assert abs(angle_error(VM, 20.0, 0.0, mc) - math.radians(10.0)) < 1e-12

  def _check_rule(self, c, CS, VM, params, steer_limited, desired_curvature, i_prev):
    """the F rule for the step just run: p = kp e, i += ki e dt unless frozen, anti-windup and clip as common/pid.py"""
    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    e = angle_error(VM, CS.vEgo, desired_curvature, measured_curvature)
    kp, ki = angle_pi_gain(G2_KP, CS.vEgo), angle_pi_gain(G2_KI, CS.vEgo)
    pid = c.pid
    assert pid.p == kp * e
    lim = pid.pos_limit
    if steer_limited or CS.steeringPressed:
      assert pid.i == i_prev
    else:
      i_new = i_prev + ki * DT_CTRL * e
      test = pid.p + i_new + pid.f
      hi = i_prev if test > lim else lim
      lo = i_prev if test < -lim else -lim
      assert pid.i == min(max(i_new, lo), hi)
    assert pid.d == 0.0
    out = pid.p + pid.i + pid.f
    assert pid.control == min(max(out, -lim), lim)
    return e

  def test_rule_in_v0(self):
    for on in ((ANGLE_PI_KEY,), INSTER_KEYS):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      assert c.inster_angle_pi and isinstance(c.pid, InsterAnglePI) and c.extension._pid is c.pid
      assert abs(c.pid.pos_limit / c.torque_params.latAccelFactor - CAP) < 1e-9
      rng = random.Random(5)
      angle = 0.0
      frozen = 0
      for k in range(3000):
        t = k * DT_CTRL
        v = max(0.0, 12.0 + 13.0 * math.sin(0.11 * t))
        dc = 0.02 * math.sin(0.7 * t) + 0.01 * math.sin(2.9 * t)
        target = math.degrees(VM.get_steer_from_curvature(-dc, v, 0.0))
        rate = (target - angle) * 3.0 + rng.gauss(0, 5.0)
        angle += rate * DT_CTRL
        CS = _cs(v, angle, pressed=(k % 700) < 25, rate=abs(rate))
        params = _vp(0.02 * math.sin(0.05 * t), 0.3)
        if k % 100 == 0:
          c.update_torque_parameters(2.8 + 0.3 * math.sin(0.02 * t), 0.04, 0.09)
        sl = (k % 41) == 0
        i_prev = c.pid.i
        tq, _, lac_log = c.update(True, CS, VM, params, sl, dc, None, False, 0.2)
        self._check_rule(c, CS, VM, params, sl, dc, i_prev)
        frozen += sl or CS.steeringPressed
        assert abs(tq) <= CAP + 1e-12
        assert tq == -c.pid.control / c.torque_params.latAccelFactor
        # logs: p / i in lateral-acceleration units, error stays v0's lateral-acceleration error
        assert lac_log.p == np.float32(c.pid.p) and lac_log.i == np.float32(c.pid.i)
        measured_curvature = -VM.calc_curvature(math.radians(angle - 0.3), v, params.roll)
        assert abs(lac_log.error - (dc - measured_curvature) * v ** 2) < 1e-3 * max(1.0, abs(lac_log.error))
      assert frozen > 50

  def test_integrator_from_standstill(self):
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS)
    params = _vp()
    for v in (0.0, 1.0, 4.9):
      c.pid.reset()
      i_values = []
      for _ in range(50):
        c.update(True, _cs(v, angle=-5.0), VM, params, False, 0.0, None, False, 0.2)   # wheel 5 deg right of a straight plan
        i_values.append(c.pid.i)
      v = _cs(v).vEgo                                                                         # as the controller sees it (float32)
      e = angle_error(VM, v, 0.0, -VM.calc_curvature(math.radians(-5.0), v, 0.0))
      assert e < 0.0 and abs(e - math.radians(-5.0)) < 1e-12                                 # the plan is left of the wheel: e < 0
      # integrates from the first step at every speed (v0 freezes below 5 m/s)
      assert abs(i_values[0] - angle_pi_gain(G2_KI, v) * DT_CTRL * e) < 1e-15
      assert abs(i_values[-1] - 50 * angle_pi_gain(G2_KI, v) * DT_CTRL * e) < 1e-12
    # v0 (switch off) keeps its freeze below 5 m/s
    off, VM_off = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=STEP2_KEYS)
    for _ in range(50):
      off.update(True, _cs(1.0, angle=-5.0), VM_off, params, False, 0.0, None, False, 0.2)
    assert off.pid.i == 0.0

  def test_freeze(self):
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS)
    params = _vp()
    for _ in range(20):
      c.update(True, _cs(10.0, angle=-5.0), VM, params, False, 0.0, None, False, 0.2)
    i0 = c.pid.i
    assert i0 < 0.0
    for _ in range(30):
      c.update(True, _cs(10.0, angle=-5.0, pressed=True), VM, params, False, 0.0, None, False, 0.2)
      assert c.pid.i == i0                                                                   # steeringPressed
      c.update(True, _cs(10.0, angle=-5.0), VM, params, True, 0.0, None, False, 0.2)
      assert c.pid.i == i0                                                                   # steer_limited_by_safety (cap / ramp)
    c.update(True, _cs(10.0, angle=-5.0), VM, params, False, 0.0, None, False, 0.2)
    assert c.pid.i < i0
    # inactive: no output, and F's integrator is reset (v0 keeps its own, see test_stale_integrator_after_inactive)
    assert c.pid.i != 0.0
    tq, _, lac_log = c.update(False, _cs(10.0, angle=-5.0), VM, params, False, 0.0, None, False, 0.2)
    assert tq == 0.0 and not lac_log.active and c.pid.i == 0.0 and c.pid.p == 0.0

  def test_stale_integrator_after_inactive(self):
    # review scenario (agents/f_review rv_check.py "stale"): 8 s active at 2 m/s in a slow turn (plan 0.02 1/m) with the wheel 8 deg
    # behind the plan winds F's integrator (v0 is frozen below 5 m/s); 10 s inactive (driver took over); re-engaged at 25 m/s on a
    # straight with the wheel on the plan. Without the reset F's first command was 170.6 raw towards the old curve; now ~0 as v0.
    first = {}
    for name, on in (("v0_step2", STEP2_KEYS), ("F", INSTER_KEYS)):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      for _ in range(800):
        plan = math.degrees(VM.get_steer_from_curvature(-0.02, 2.0, 0.0))
        c.update(True, _cs(2.0, plan - 8.0), VM, _vp(), False, 0.02, None, False, 0.2)
      i_wound = c.pid.i
      for _ in range(1000):
        c.update(False, _cs(10.0, 0.0), VM, _vp(), False, 0.0, None, False, 0.2)
      tq, _, lac_log = c.update(True, _cs(25.0, 0.0), VM, _vp(), False, 0.0, None, False, 0.2)
      first[name] = (i_wound, c.pid.i, tq * STEER_MAX_RAW, lac_log.version)
    assert first["v0_step2"][0] == 0.0 and first["v0_step2"][3] == 0
    assert first["F"][0] < -1.5                                     # F did wind at 2 m/s (-1.75 m/s^2 with G2, -1.64 with F's gains)
    assert abs(first["F"][1]) < 1e-3                                # reset: only the first step's I at the re-engagement
    assert abs(first["F"][2]) < 0.5 and abs(first["F"][2] - first["v0_step2"][2]) < 0.5   # ~0 raw, as v0 (was 170.6 raw)
    assert first["F"][3] == ANGLE_PI_LOG_VERSION

  def test_log_version(self):
    # F drives are identifiable in the logs: torqueState.version 102 with F on (G2 gains; 101 with F's gains and the crawl plan, 100
    # before it), 0 for v0 otherwise (upstream: 1)
    assert ANGLE_PI_LOG_VERSION == 102
    assert ANGLE_PI_LOG_VERSION not in (latcontrol_torque_v0.VERSION, 1)
    for on, expected in ((INSTER_KEYS, ANGLE_PI_LOG_VERSION), ((ANGLE_PI_KEY,), ANGLE_PI_LOG_VERSION), (STEP2_KEYS, 0), ((), 0),
                         (INSTER_KEYS + ("NeuralNetworkLateralControl",), 0)):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      for active in (True, False):
        _, _, lac_log = c.update(active, _cs(10.0, 1.0), VM, _vp(), False, 0.0, None, False, 0.2)
        assert lac_log.version == expected, (on, active)
    c, VM = _make_v0(HYUNDAI.HYUNDAI_KONA_EV, on=INSTER_KEYS)
    assert c.update(True, _cs(10.0, 1.0), VM, _vp(), False, 0.0, None, False, 0.2)[2].version == 0

  def test_anti_windup_and_limits(self):
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS)
    params = _vp()
    lim = c.pid.pos_limit
    # far off the plan: the output sits at the cap (409 raw)
    for _ in range(300):
      tq, _, _ = c.update(True, _cs(8.0, angle=-200.0), VM, params, False, 0.0, None, False, 0.2)
      assert abs(c.pid.i) <= lim + 1e-12 and abs(tq) <= CAP + 1e-12
    assert abs(tq - CAP) < 1e-12
    # P alone below the limit (wheel 0.3 rad left of a straight plan: e = +0.3, asks for right), P + I would pass it after ~2 s: I stops where the output
    # reaches the limit (common/pid.py) instead of winding up for the 10 s
    c.pid.reset()
    for _ in range(1000):
      tq, _, _ = c.update(True, _cs(8.0, angle=math.degrees(0.3)), VM, params, False, 0.0, None, False, 0.2)
      assert abs(tq) <= CAP + 1e-12
    p, i, f = c.pid.p, c.pid.i, c.pid.f
    assert 0.0 < p < lim and abs(tq + CAP) < 1e-9
    assert abs(p + i + f - lim) < 0.05                                                       # held at the limit, no wind-up
    # small error the other way: the output leaves the limit on the first step
    tq, _, _ = c.update(True, _cs(8.0, angle=-math.degrees(0.05)), VM, params, False, 0.0, None, False, 0.2)
    assert -CAP + 0.1 < tq

  def test_alone_replaces_only_p_and_i(self):
    # F with both step-2 switches off: v0's linear feed-forward and get_friction (on v0's lateral-acceleration error) stay
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=(ANGLE_PI_KEY,))
    assert not c.inster_friction and not c.inster_ff and c.inster_angle_pi
    base, VM_base = _make_v0(HYUNDAI.HYUNDAI_INSTER)
    with mock.patch.object(latcontrol_torque_v0, "get_friction", wraps=latcontrol_torque_v0.get_friction) as gf:
      a = _run(c, VM)
      assert gf.call_count > 0
    b = _run(base, VM_base)
    assert [o[3] for o in a] == [o[3] for o in b]          # f: the same feed-forward
    assert [o[5] for o in a] == [o[5] for o in b]          # error: v0's lateral-acceleration error
    assert [o[0] for o in a] != [o[0] for o in b]

  def test_off_keeps_step2_output(self):
    # F off: the step-2 combination runs exactly as before (also with the F param present but off)
    a = _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=STEP2_KEYS))
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=STEP2_KEYS)
    assert type(c.pid) is PIDController and not c.inster_angle_pi
    assert a == _run(c, VM)
    assert a != _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS))

  def test_error_is_steering_wheel_angle_through_update(self):
    # F's error is the STEERING-WHEEL ANGLE error (rad), not v0's curvature x v^2 error. Through the real update(): with a fresh
    # integrator, p = kp(v) e and i = ki(v) e dt, where e is built here directly from the carState / liveParameters values:
    #   delta_plan = VM.get_steer_from_curvature(-desired_curvature, v, roll)   (openpilot's +left angle frame; as latcontrol_pid /
    #                latcontrol_angle turn desired_curvature into a wheel angle, incl. steer ratio, slip factor and roll)
    #   delta_meas = radians(steeringAngleDeg - angleOffsetDeg)                (+left)
    #   e = -(delta_plan - delta_meas)   (+right, v0's internal frame) = -radians(latcontrol_pid's error) / 1 deg
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS)
    VM.update_params(0.93, 14.6)
    n = n_integrating = 0
    for kmh in (0.0, 10.0, 30.0, 60.0, 100.0, 130.0, 160.0):
      for desired_curvature in (0.004, -0.004, 0.0):          # both turn directions and straight
        for wheel_vs_plan in (2.0, -2.0, 0.5, -0.5):          # wheel left / right of the plan (deg)
          for roll, offset in ((0.05, 2.5), (-0.04, -1.5)):
            CS0 = _cs(kmh / 3.6)
            vp = _vp(roll, offset)
            v, roll32, off32 = CS0.vEgo, vp.roll, vp.angleOffsetDeg   # the float32 values the controller reads
            plan_deg = math.degrees(VM.get_steer_from_curvature(-desired_curvature, v, roll32)) + off32
            CS = _cs(kmh / 3.6, angle=plan_deg + wheel_vs_plan)
            delta_plan = VM.get_steer_from_curvature(-desired_curvature, v, roll32)
            delta_meas = math.radians(CS.steeringAngleDeg - off32)
            e = -(delta_plan - delta_meas)
            pid_error_deg = math.degrees(delta_plan) + off32 - CS.steeringAngleDeg   # latcontrol_pid's error (deg, +left)
            assert abs(e + math.radians(pid_error_deg)) < 1e-12
            assert (e > 0.0) == (wheel_vs_plan > 0.0)        # wheel left of the plan -> e > 0 -> steer right (v0-internal +)
            assert abs(e - math.radians(CS.steeringAngleDeg - plan_deg)) < 1e-6
            c.pid.reset()
            c.update(True, CS, VM, vp, False, desired_curvature, None, False, 0.2)
            kp, ki = angle_pi_gain(G2_KP, v), angle_pi_gain(G2_KI, v)
            assert abs(c.pid.p - kp * e) <= 1e-12 * max(1.0, abs(kp * e)), (kmh, desired_curvature, wheel_vs_plan, roll)
            i_new = ki * DT_CTRL * e                                  # first step from i = 0, then common/pid.py's anti-windup
            test, lim = c.pid.p + i_new + c.pid.f, c.pid.pos_limit
            expected_i = min(max(i_new, 0.0 if test < -lim else -lim), 0.0 if test > lim else lim)
            assert abs(c.pid.i - expected_i) <= 1e-14, (kmh, desired_curvature, wheel_vs_plan, roll)
            n += 1
            n_integrating += expected_i == i_new
    assert n == 7 * 3 * 4 * 2 and n_integrating > 0.8 * n

  def test_gain_units_at_100_kmh(self):
    # at 100 km/h a 1 deg wheel-angle error gives p = kp(27.8 m/s) x radians(1) m/s^2 and an I step of ki x radians(1) x dt, NOT
    # kp x v0's lateral-acceleration error; the gains use the speed of the current step (PIDController.update sets self.speed first)
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS)
    VM.update_params(1.0, 14.0)
    params = _vp()
    c.update(True, _cs(30 / 3.6, angle=1.0), VM, params, False, 0.0, None, False, 0.2)        # a step at another speed first
    c.pid.reset()
    CS = _cs(100 / 3.6, angle=1.0)                                                           # wheel 1 deg left of a straight plan
    _, _, lac_log = c.update(True, CS, VM, params, False, 0.0, None, False, 0.2)
    v = CS.vEgo
    kp, ki = angle_pi_gain(G2_KP, v), angle_pi_gain(G2_KI, v)
    assert abs(kp - 10.80) < 0.01 and abs(ki - 4.87) < 0.01                                 # G2 at 100 km/h
    assert abs(c.pid.p - kp * math.radians(1.0)) < 1e-12                                     # 0.173 m/s^2
    assert abs(c.pid.i - ki * math.radians(1.0) * DT_CTRL) < 1e-15
    # v0's lateral-acceleration error for the same 1 deg is (curvature error) x v^2: a different quantity
    lat_error = (0.0 + VM.calc_curvature(math.radians(1.0), v, 0.0)) * v ** 2
    assert abs(lac_log.error - lat_error) < 1e-5 and lat_error > 0.0
    assert abs(c.pid.p - kp * lat_error) > 0.5 * c.pid.p                                    # not kp x the lateral-acceleration error
    assert abs(c.pid.p / lac_log.error - kp * VM.get_steer_from_curvature(1.0, v, 0.0) / v ** 2) < 1e-5


# G6 (27 Oct 2026; alongside G2, sub-switch InsterAngleErrorPIG6): the Nelder-Mead result's coefficients at full precision, the
# 6-significant-digit values the car code uses, and the anchor values at 0 / 10 / 55 / 115 / 160 km/h
G6_KEY = "InsterAngleErrorPIG6"
G6_KP_SEARCH = (3.4164007100591722, 0.4107633656804734, -0.007152631668639052)
G6_KI_SEARCH = (0.93525, 0.4462301643956044, -0.009480687570989011)
G6_KP = (3.4164, 0.410763, -0.00715263)
G6_KI = (0.93525, 0.44623, -0.00948069)
G6_KP_ANCHOR = (3.4164, 4.50222, 8.02245, 9.23911, 7.54388)
G6_KI_ANCHOR = (0.93525, 2.10162, 5.53976, 5.51527, 2.04041)


class TestInsterAnglePIG6(OpenpilotTestCase):
  """sub-switch InsterAngleErrorPIG6: with InsterAngleErrorPI on, "F" runs the gain set G6 instead of G2 (torqueState.version 103)"""

  def test_gains(self):
    assert ANGLE_PI_G6_KP == G6_KP and ANGLE_PI_G6_KI == G6_KI and ANGLE_PI_G6_SWITCH == G6_KEY
    assert G6_KP == tuple(float(f"{c:.6g}") for c in G6_KP_SEARCH) and G6_KI == tuple(float(f"{c:.6g}") for c in G6_KI_SEARCH)
    for kmh, kp, ki in zip(G2_ANCHOR_KMH, G6_KP_ANCHOR, G6_KI_ANCHOR, strict=True):
      assert abs(angle_pi_gain(ANGLE_PI_G6_KP, kmh / 3.6) - kp) < 1e-5, kmh
      assert abs(angle_pi_gain(ANGLE_PI_G6_KI, kmh / 3.6) - ki) < 1e-5, kmh
    # rounding to 6 significant digits: within 1e-5 (relative) of the search's gains up to 160 km/h
    for v in np.linspace(0.0, 160 / 3.6, 1601):
      assert abs(angle_pi_gain(ANGLE_PI_G6_KP, v) / angle_pi_gain(G6_KP_SEARCH, v) - 1.0) < 1e-5, v
      assert abs(angle_pi_gain(ANGLE_PI_G6_KI, v) / angle_pi_gain(G6_KI_SEARCH, v) - 1.0) < 1e-5, v
    # never negative from 0 to 200 km/h: kp > 0 throughout (root 233 km/h), ki > 0 below its root (176.7 km/h), clipped at 0 above
    for v in np.linspace(0.0, 200 / 3.6, 2001):
      assert angle_pi_gain(ANGLE_PI_G6_KP, v) > 0.0 and angle_pi_gain(ANGLE_PI_G6_KI, v) >= 0.0
      if v < 176 / 3.6:
        assert angle_pi_gain(ANGLE_PI_G6_KI, v) > 0.0
    assert angle_pi_gain(ANGLE_PI_G6_KI, 177 / 3.6) == 0.0
    # less P, more I than G2 from 10 to 130 km/h
    for v in np.linspace(10 / 3.6, 130 / 3.6, 121):
      assert angle_pi_gain(ANGLE_PI_G6_KP, v) < angle_pi_gain(ANGLE_PI_KP, v)
      assert angle_pi_gain(ANGLE_PI_G6_KI, v) > angle_pi_gain(ANGLE_PI_KI, v)
    pi = InsterAnglePI(rate=1 / DT_CTRL, kp=ANGLE_PI_G6_KP, ki=ANGLE_PI_G6_KI)
    g2 = InsterAnglePI(rate=1 / DT_CTRL)
    for v in (0.0, 4.2, 15.3, 31.9):
      pi.speed = g2.speed = v
      assert pi.k_p == angle_pi_gain(G6_KP, v) and pi.k_i == angle_pi_gain(G6_KI, v) and pi.k_d == 0.0
      assert g2.k_p == angle_pi_gain(G2_KP, v) and g2.k_i == angle_pi_gain(G2_KI, v)

  def test_switch(self):
    for on in ((ANGLE_PI_KEY, G6_KEY), INSTER_KEYS + (G6_KEY,)):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      assert c.inster_angle_pi and c.inster_angle_pi_g6 and isinstance(c.pid, InsterAnglePI) and c.extension._pid is c.pid
      assert (c.pid.kp_coeffs, c.pid.ki_coeffs) == (G6_KP, G6_KI)
    for on in ((ANGLE_PI_KEY,), INSTER_KEYS):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      assert c.inster_angle_pi and not c.inster_angle_pi_g6 and (c.pid.kp_coeffs, c.pid.ki_coeffs) == (G2_KP, G2_KI)
    # G6 alone, with NNLC, or on another car: no effect (the run equals the one without the G6 param)
    for car_name, on in ((HYUNDAI.HYUNDAI_INSTER, ()), (HYUNDAI.HYUNDAI_INSTER, STEP2_KEYS),
                         (HYUNDAI.HYUNDAI_INSTER, INSTER_KEYS + ("NeuralNetworkLateralControl",)), (HYUNDAI.HYUNDAI_KONA_EV, INSTER_KEYS)):
      c, VM = _make_v0(car_name, on=on + (G6_KEY,))
      assert not c.inster_angle_pi_g6 and type(c.pid) is PIDController
      assert _run(c, VM) == _run(*_make_v0(car_name, on=on)), (car_name, on)
    # G6 changes the output vs G2
    assert _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS + (G6_KEY,))) != _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS))
    # not an Inster switch of its own: alone it leaves the Lateral Jerk Torque Controller on
    _make_v0(HYUNDAI.HYUNDAI_INSTER, on=(G6_KEY, "LateralJerkTorqueController"))
    assert Params().get_bool("LateralJerkTorqueController")

  def test_log_version(self):
    # G6 drives are identifiable in the logs: torqueState.version 103 with G6, 102 with G2, 0 for v0 (upstream: 1)
    assert (ANGLE_PI_G6_LOG_VERSION, ANGLE_PI_LOG_VERSION) == (103, 102)
    for on, expected in ((INSTER_KEYS + (G6_KEY,), 103), ((ANGLE_PI_KEY, G6_KEY), 103), (INSTER_KEYS, 102), ((ANGLE_PI_KEY,), 102),
                         (STEP2_KEYS + (G6_KEY,), 0), ((G6_KEY,), 0), (INSTER_KEYS + (G6_KEY, "NeuralNetworkLateralControl"), 0)):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      for active in (True, False):
        _, _, lac_log = c.update(active, _cs(10.0, 1.0), VM, _vp(), False, 0.0, None, False, 0.2)
        assert lac_log.version == expected, (on, active)

  def test_rule_through_update(self):
    # through the real update(), from a fresh integrator: p = kp_G6(v) e and i = ki_G6(v) e dt (then common/pid.py's anti-windup)
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS + (G6_KEY,))
    VM.update_params(1.0, 14.0)
    n_integrating = 0
    for kmh in (0.0, 10.0, 30.0, 60.0, 100.0, 130.0, 160.0):
      for wheel in (2.0, -2.0, 0.5):
        CS = _cs(kmh / 3.6, angle=wheel)
        v = CS.vEgo
        e = angle_error(VM, v, 0.0, -VM.calc_curvature(math.radians(wheel), v, 0.0))
        c.pid.reset()
        c.update(True, CS, VM, _vp(), False, 0.0, None, False, 0.2)
        kp, ki = angle_pi_gain(G6_KP, v), angle_pi_gain(G6_KI, v)
        assert abs(c.pid.p - kp * e) <= 1e-12 * max(1.0, abs(kp * e)), kmh
        i_new = ki * DT_CTRL * e
        test, lim = c.pid.p + i_new + c.pid.f, c.pid.pos_limit
        expected_i = min(max(i_new, 0.0 if test < -lim else -lim), 0.0 if test > lim else lim)
        assert abs(c.pid.i - expected_i) <= 1e-14, kmh
        n_integrating += expected_i == i_new
    assert n_integrating > 15


def _rate(c, v):
  """the rule's rate (1/m per metre) at speed v: c up to the release speed, opening linearly to max(c, release rate) at 1.4 m/s"""
  if v <= 1.0:
    return c
  return c + (v - 1.0) / 0.4 * (max(c, 1.0) - c)


def _crawl_sim(vs, model, k0, rule=True):
  """100 Hz as controlsd (rule, then clip_curvature with the Inster's ceiling); returns (s, desired, model) per step, the largest
  10 ms step of the desired curvature and the extra heading (rad) / lateral offset (m) vs following the model until the plan rejoins it"""
  des, s, h, y, mx, done, out = k0, 0.0, 0.0, 0.0, 0.0, False, []
  for v in vs:
    m = model(s)
    new = crawl_plan_curvature(v, des, m, DT_CTRL) if rule else m
    nd, _ = clip_curvature(v, des, new, 0.0, 3.3)
    mx = max(mx, abs(nd - des))
    des = nd
    s += v * DT_CTRL
    if not done:
      h += (des - m) * v * DT_CTRL
      y += h * v * DT_CTRL
      done = abs(des - m) < 1e-12
    out.append((s, des, m))
  return out, mx, h, y


class TestInsterCrawlPlan(OpenpilotTestCase):
  """controlsd with "F" on: below 1.4 m/s the plan moves by at most a rate per metre travelled"""

  def test_constants(self):
    assert (CRAWL_PLAN_SPEED, CRAWL_PLAN_RELEASE_SPEED) == (1.4, 1.0)
    assert (CRAWL_STRAIGHTEN_PER_M, CRAWL_STRAIGHTEN_LENGTH, CRAWL_TURN_IN_PER_M, CRAWL_PER_M_RELEASE) == (0.04, 2.0, 0.4, 1.0)
    assert ANGLE_PI_LOG_VERSION == 102                      # F drives with the crawl plan (G2 gains) are identifiable in the logs

  def test_unchanged_from_crawl_speed_up(self):
    rng = random.Random(1)
    for _ in range(20000):
      v = rng.choice([1.4, rng.uniform(1.4, 50.0)])
      prev, new = rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)
      assert crawl_plan_curvature(v, prev, new, DT_CTRL) == new

  def test_rule(self):
    # independent restatement: towards straight at (0.04 + |prev| / 2) per metre up to zero, then away from straight at 0.4 per metre
    rng = random.Random(3)
    for _ in range(50000):
      v = rng.choice([0.0, 1.0, rng.uniform(0.0, 1.4)])
      prev = rng.choice([0.0, rng.uniform(-0.2, 0.2), rng.uniform(-0.003, 0.003)])
      new = rng.choice([prev, -prev, rng.uniform(-0.2, 0.2), prev + rng.uniform(-1e-3, 1e-3)])
      out = crawl_plan_curvature(v, prev, new, DT_CTRL)
      ds = v * DT_CTRL
      k = prev
      if k != 0.0 and (new - k) * k < 0.0:
        target = new if new * k > 0.0 else 0.0
        c = _rate(0.04 + abs(k) / 2.0, v)
        if abs(target - k) > c * ds:
          expected = k - math.copysign(c * ds, k)
        else:
          ds -= abs(target - k) / c
          k = target
          expected = k if k == new else None
      else:
        expected = None
      if expected is None:
        step = _rate(0.4, v) * ds
        expected = k + max(-step, min(step, new - k))
      assert abs(out - expected) < 1e-15, (v, prev, new, out, expected)
      assert min(prev, new) - 1e-15 <= out <= max(prev, new) + 1e-15     # between the previous plan and the model's
      # never more than the rates allow in this step (no jump at zero or near straight)
      assert abs(out - prev) <= max(_rate(0.4, v), _rate(0.04 + abs(prev) / 2.0, v)) * v * DT_CTRL + 1e-15
    assert crawl_plan_curvature(0.0, 0.05, 0.0, DT_CTRL) == 0.05           # standstill: no travel, no change
    assert crawl_plan_curvature(0.5, 0.05, 0.05, DT_CTRL) == 0.05

  def test_release_continuous(self):
    # continuous in speed at the release speed; just below 1.4 m/s both rates are 1.0 1/m per metre (0.014 1/m per step)
    for prev, new in ((0.05, 0.0), (0.0, 0.05), (0.02, -0.05), (0.2, 0.0)):
      a, b = crawl_plan_curvature(1.0 - 1e-9, prev, new, DT_CTRL), crawl_plan_curvature(1.0 + 1e-9, prev, new, DT_CTRL)
      assert abs(a - b) < 1e-9, (prev, new)
    assert abs(crawl_plan_curvature(1.4 - 1e-9, 0.0, 0.1, DT_CTRL) - 0.014) < 1e-9
    assert abs(crawl_plan_curvature(1.4 - 1e-9, 0.05, 0.0, DT_CTRL) - (0.05 - 0.014)) < 1e-9

  def test_collapsed_plan_at_a_stop_in_a_turn(self):
    # LOG-like stop (owner's r97 stop at 1209 s): 0.068 1/m at 1.07 m/s; the model plan shrinks with v (floor) and straightens as the
    # car stops; afterwards it comes back and the turn tightens
    k_turn, prev, v = 0.068, 0.068, 1.4
    while v > 0.3:                                      # approach: 1.4 -> 0.3 m/s at 0.6 m/s^2
      prev = crawl_plan_curvature(v, prev, k_turn * min(v, 1.0) ** 1.5, DT_CTRL)
      v -= 0.6 * DT_CTRL
    # 1.6 m of travel below 1.4 m/s with the model plan going to 0.011: the proportional relaxation (worst-case bound) keeps 0.0226
    collapsed = k_turn * 0.3 ** 1.5
    assert 0.022 < prev < 0.023 and prev > 2 * collapsed, prev
    prev, v = prev, 0.31                                # drive-off from the held wheel (controlsd: measured curvature while inactive)
    for _ in range(300):
      model = 0.12 * min(v, 1.0)
      out = crawl_plan_curvature(v, prev, model, DT_CTRL)
      if v >= 1.4:
        assert out == model
      prev = out
      v += 0.8 * DT_CTRL
    assert prev == 0.12

  def test_worst_cases(self):
    # (1) creep out of a curve whose model plan goes straight at once: bounded extra heading / offset (proportional relaxation)
    for v in (0.5, 0.9):
      for k0, h_max, y_max in ((0.1, 4.1, 0.09), (0.2, 11.5, 0.36)):
        out, _, h, y = _crawl_sim([v] * 8000, lambda s: 0.0, k0)
        assert out[-1][1] == 0.0 and math.degrees(h) < h_max and y < y_max, (v, k0, math.degrees(h), y)
    # (2) S-bend at creep, +k -> -k over 2 m: no jump through zero (largest 10 ms step <= the turn-in rate x distance)
    for v in (0.5, 0.9):
      for k in (0.03, 0.05, 0.1):
        out, mx, h, y = _crawl_sim([v] * 4000, lambda s, k=k: k - 2 * k * min(1.0, s / 2.0), k)
        assert mx <= 0.4 * v * DT_CTRL + 1e-12 and abs(out[-1][1] + k) < 1e-12, (v, k, mx)
        assert y < 0.04
    # (3) stop before a turn with a small opposite residual: crosses zero soon after drive-off
    for kr, s_max in ((-0.003, 0.1), (-0.01, 0.25), (-0.03, 0.66)):
      vs = [min(0.3 + 0.5 * DT_CTRL * i, 3.0) for i in range(600)]
      out, _, _, _ = _crawl_sim(vs, lambda s: 0.08, kr)
      s_cross = next(s for s, d, _ in out if d > 0.0)
      assert s_cross < s_max, (kr, s_cross)
    # (4) stop straight, then a turn-in of 0 -> 0.08 1/m over 0.5 m: no delay vs the model's plan
    vs = [min(0.3 + 0.5 * DT_CTRL * i, 3.0) for i in range(600)]
    rule, _, _, _ = _crawl_sim(vs, lambda s: min(0.08, 0.16 * s), 0.0)
    plain, _, _, _ = _crawl_sim(vs, lambda s: min(0.08, 0.16 * s), 0.0, rule=False)
    assert all(abs(a[1] - b[1]) < 1e-12 for a, b in zip(rule, plain, strict=True))

  def test_controlsd_applies_the_rule_only_with_f(self):
    # controlsd's plan statement, executed from its source: model plan (or maneuver / measured) as before unless inster_crawl_plan;
    # with it, crawl_plan_curvature from the previous plan, or from the measured curvature while the driver steers
    import ast
    import importlib.util
    import textwrap
    import types
    spec = importlib.util.find_spec("openpilot.selfdrive.controls.controlsd")
    assert spec is not None and spec.origin is not None
    src = open(spec.origin).read()
    assert "self.inster_crawl_plan = bool(getattr(self.LaC, 'inster_angle_pi', False))" in src
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "state_control")
    node = next(n for n in fn.body if isinstance(n, ast.If) and "lateralManeuverPlan" in ast.get_source_segment(src, n.test))
    code = compile(textwrap.dedent(" " * node.col_offset + ast.get_source_segment(src, node)), "controlsd", "exec")
    rng = random.Random(4)
    for _ in range(4000):
      flag, lat, pressed, man = (rng.random() < 0.5 for _ in range(4))
      v = rng.choice([0.0, 0.3, rng.uniform(0.0, 1.5), rng.uniform(0.0, 40.0)])
      prev, model, meas, mplan = (rng.uniform(-0.2, 0.2) for _ in range(4))

      class SM(dict):
        valid = {"lateralManeuverPlan": man}
      ctl = types.SimpleNamespace(sm=SM(lateralManeuverPlan=types.SimpleNamespace(desiredCurvature=mplan)), curvature=meas,
                                  desired_curvature=prev, inster_crawl_plan=flag)
      env = {"self": ctl, "CC": types.SimpleNamespace(latActive=lat), "CS": types.SimpleNamespace(vEgo=v, steeringPressed=pressed),
             "model_v2": types.SimpleNamespace(action=types.SimpleNamespace(desiredCurvature=model)),
             "crawl_plan_curvature": crawl_plan_curvature, "DT_CTRL": DT_CTRL}
      exec(code, env)
      if man:
        expected = mplan if lat else meas
      elif not lat:
        expected = meas
      elif flag:
        expected = crawl_plan_curvature(v, meas if pressed else prev, model, DT_CTRL)
      else:
        expected = model
      assert env["new_desired_curvature"] == expected
