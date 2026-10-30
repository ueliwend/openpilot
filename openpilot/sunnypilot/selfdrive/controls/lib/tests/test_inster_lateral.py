"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import random
from unittest import mock

import numpy as np

from openpilot.cereal import log, messaging
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
                                                                       ANGLE_PI_G6_SWITCH, ANGLE_PI_D_TAU, ANGLE_PI_GAINS_LEGACY_P6, \
                                                                       ANGLE_PI_GAINS_PARAM, LATERAL_CONTROLLER_PARAM, LATERAL_PI, \
                                                                       LATERAL_PID, ANGLE_PI_P6_BASE_VERSION, ANGLE_PI_P6_KD, \
                                                                       ANGLE_PI_P6_KI, ANGLE_PI_P6_KP, ANGLE_PI_P6_LOG_VERSION, \
                                                                       ANGLE_PI_VERSION_D_BIT, angle_lateral_controller
from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import ANGLE_PI_PI_BASE_VERSION, ANGLE_PI_PI_KI, ANGLE_PI_PI_KP, \
                                                                       ANGLE_PI_PID_BASE_VERSION, ANGLE_PI_PID_KD, ANGLE_PI_PID_KI, \
                                                                       ANGLE_PI_PID_KP
from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import ANGLE_PI_VERSION_FREEZE_DIR_BIT, LIMIT_TOL_RAW, \
                                                                       InsterSteerLimitObserver, freeze_angle_integrator
from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import ANGLE_PI_VERSION_SOFT_P_BIT, SOFT_P_A0, SOFT_P_E0, SOFT_P_GAMMA, \
                                                                       SOFT_P_PARAM, SOFT_P_RHO, SOFT_P_V_MIN, angle_pi_log_version, \
                                                                       soft_p_sigma
from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import A_NODES, ANGLE_PI_VERSION_FIXED_FACTOR_BIT, CMD_RAW, F0, \
                                                                       INSTER_FIXED_FACTOR, STEER_MAX_RAW, V_NODES, InsterAnglePI, InsterStictionFriction, \
                                                                       angle_error, angle_pi_gain, ff_lat_accel_equivalent, ff_torque, \
                                                                       table_torque
from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import CRAWL_PER_M_RELEASE, CRAWL_PLAN_RELEASE_SPEED, CRAWL_PLAN_SPEED, \
                                                                       CRAWL_STRAIGHTEN_LENGTH, CRAWL_STRAIGHTEN_PER_M, CRAWL_TURN_IN_PER_M, \
                                                                       crawl_plan_curvature
from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import ANGLE_PI_VERSION_MPC_BIT, MPC_PARAM, MPC_PARAMS, \
                                                                       MPC_A_LIM, MPC_JERK_MAX, InsterMPC, ff_table_slope_raw, \
                                                                       mpc_gain_rows, mpc_tau, ANGLE_PI_MPC_KP, ANGLE_PI_MPC_KI, \
                                                                       ANGLE_PI_MPC_KD, ANGLE_PI_MPC_D_TAU
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorque as LatControlTorqueV0

STEP2_KEYS = ("InsterStictionFriction", "InsterSpeedFeedforward")
ANGLE_PI_KEY = "InsterAngleErrorPI"
INSTER_KEYS = STEP2_KEYS + (ANGLE_PI_KEY,)
CTRL_KEY = "InsterLateralController"         # controller type: 0 = PI (G6), 1 = PID (P6)
SPEEDS = [0.0, 2.0, 5.0, 6.1] + list(np.linspace(6.11, 30.83, 60)) + [31.0, 35.0, 45.0]
ACCELS = list(np.linspace(0.0, 4.0, 161))


class _ScriptedLimit:
  """test stand-in for InsterSteerLimitObserver: the limit direction seen by the n-th active step is direction_of(n), so the
  integration rule can be checked without the car-side ramp in the way (the observer itself: TestInsterIntegratorFreeze)"""

  def __init__(self, direction_of=lambda n: 0):
    self.direction_of = direction_of
    self.n = 0
    self.direction = direction_of(0)

  def reset(self):
    pass

  def update(self, torque, driver_torque):
    self.n += 1
    self.direction = self.direction_of(self.n)
    return self.direction


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
      assert ff_torque(0.0, v) == 0.0
      assert ff_lat_accel_equivalent(0.0, v, F0) == 0.0

  def test_odd(self):
    for v in SPEEDS:
      for a in ACCELS:
        assert ff_torque(-a, v) == -ff_torque(a, v)

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
        assert abs(ff_torque(a, v)) <= 409 / 384
        assert abs(ff_torque(-a, v)) <= 409 / 384

  def test_clamped_outside_speed_nodes(self):
    for a in ACCELS:
      assert table_torque(a, 1.0) == table_torque(a, V_NODES[0])
      assert table_torque(a, 40.0) == table_torque(a, V_NODES[-1])

  def test_table_read_unscaled(self):
    # 30 Oct 2026: the table is read as measured, no scaling by a learned factor
    for v in (5.0, 12.0, 25.0):
      for a in (0.1, 0.6, 1.3, 2.2):
        assert ff_torque(a, v) == table_torque(a, v)
        assert ff_torque(-a, v) == -table_torque(a, v)

  def test_smooth_in_lateral_accel(self):
    # the rule the table is built and refit by (t2_table409): below the cap, neighbouring slopes in a differ by at most x2.5 up and
    # x0.4 down (0.01 tolerance: the cells are rounded to 0.1 raw)
    for row in CMD_RAW:
      s = [(row[i + 1] - row[i]) / (A_NODES[i + 1] - A_NODES[i]) for i in range(len(row) - 1)]
      for i in range(len(s) - 1):
        if row[i + 2] < INSTER_STEER_MAX and s[i] > 0.0:
          assert 0.4 - 0.01 <= s[i + 1] / s[i] <= 2.5 + 0.01, (row, i)

  def test_fixed_factor_is_f0(self):
    # the Inster path's one conversion constant (the closed-loop simulator's factor)
    assert INSTER_FIXED_FACTOR == F0 == 2.8

  def test_lat_accel_equivalent(self):
    for v in (5.0, 20.0):
      for a in (-2.0, -0.3, 0.4, 1.7):
        for f in (2.0, 2.8, 3.7):
          assert ff_lat_accel_equivalent(a, v, f) == f * ff_torque(a, v)      # unit conversion only: the query is not scaled


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


def _make_v0(car_name, on=(), after_setup=(), ctrl=None, legacy_gains=None):
  """ctrl: controller-type selector InsterLateralController (None = unset: PI, or the legacy gain-set selector legacy_gains)"""
  params = Params()
  for key in INSTER_KEYS + (G6_KEY, O5_KEY, MPC_KEY, "LateralJerkTorqueController", "NeuralNetworkLateralControl"):
    params.put_bool(key, key in on, block=True)
  for key, value in ((CTRL_KEY, ctrl), (GAINS_KEY, legacy_gains)):
    if value is None:
      params.remove(key)
    else:
      params.put(key, value, block=True)
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
      for key in INSTER_KEYS + (G6_KEY, O5_KEY, MPC_KEY, "LateralJerkTorqueController", "NeuralNetworkLateralControl"):
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
        assert abs(ff_torque(a, v)) * STEER_MAX_RAW <= INSTER_STEER_MAX + 1e-9
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
    assert GAINS_KEY in keys and GAINS_KEY not in inster_lateral.INSTER_SWITCHES    # the gain-set selector that replaces it
    assert O5_KEY in keys and O5_KEY not in inster_lateral.INSTER_SWITCHES          # option 5: needs InsterAngleErrorPI


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
    # G2 is no longer selectable (29 Oct 2026: one option per controller type, PI = G6 or PID = P6): no G2 constants in the car
    # code, and InsterAnglePI has no default gain set; G2's coefficients (torqueState.version 102 drives) stay documented here
    assert not hasattr(inster_lateral, "ANGLE_PI_KP") and not hasattr(inster_lateral, "ANGLE_PI_KI")
    assert not hasattr(inster_lateral, "ANGLE_PI_LOG_VERSION") and not hasattr(inster_lateral, "angle_pi_gain_set")
    with self.assertRaises(TypeError):
      InsterAnglePI(rate=1 / DT_CTRL)
    assert G2_KP == tuple(float(f"{c:.6g}") for c in G2_KP_SEARCH) and G2_KI == tuple(float(f"{c:.6g}") for c in G2_KI_SEARCH)
    for kmh, kp, ki in zip(G2_ANCHOR_KMH, G2_KP_ANCHOR, G2_KI_ANCHOR, strict=True):
      assert abs(angle_pi_gain(G2_KP, kmh / 3.6) - kp) < 1e-5, kmh
      assert abs(angle_pi_gain(G2_KI, kmh / 3.6) - ki) < 1e-5, kmh
    # angle_pi_gain: c0 + c1 v + c2 v^2, clipped at 0 (never negative)
    for v in np.linspace(0.0, 200 / 3.6, 2001):
      assert angle_pi_gain(G2_KP, v) >= 0.0 and angle_pi_gain(G2_KI, v) >= 0.0
    assert angle_pi_gain(G2_KP, 199.8 / 3.6) > 0.0 and angle_pi_gain(G2_KP, 200 / 3.6) == 0.0
    assert angle_pi_gain(G2_KI, 55.0) == 0.0 and angle_pi_gain(G2_KP, 55.0) > 0.0
    # InsterAnglePI uses the coefficients it is given
    pi = InsterAnglePI(rate=1 / DT_CTRL, kp=G2_KP, ki=G2_KI)
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

  def _check_rule(self, c, CS, VM, params, limit_direction, desired_curvature, i_prev):
    """the F rule for the step just run: p = kp e, i += ki e dt unless frozen (pressed, or the last request held back by the car-side
    limiter in the direction of this update), anti-windup and clip as common/pid.py"""
    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    e = angle_error(VM, CS.vEgo, desired_curvature, measured_curvature)
    kp, ki = angle_pi_gain(CAR_PI_KP, CS.vEgo), angle_pi_gain(CAR_PI_KI, CS.vEgo)   # selector unset: PI
    pid = c.pid
    assert pid.p == kp * e
    lim = pid.pos_limit
    if freeze_angle_integrator(CS.steeringPressed, limit_direction, e):
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
      c.inster_limit_observer = _ScriptedLimit(lambda n: (0, 1, 0, -1, 0, 0, 1)[n % 7])
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
        sl = (k % 41) == 0                     # controlsd's flag: no longer freezes F (its integrator uses the limit direction)
        i_prev = c.pid.i
        ld = c.inster_limit_observer.direction
        tq, _, lac_log = c.update(True, CS, VM, params, sl, dc, None, False, 0.2)
        e = self._check_rule(c, CS, VM, params, ld, dc, i_prev)
        frozen += freeze_angle_integrator(CS.steeringPressed, ld, e)
        assert abs(tq) <= CAP + 1e-12
        assert c.torque_params.latAccelFactor == F0_STORED    # fixed factor: the learner's 2.5-3.1 above is not used
        assert tq == -c.pid.control / c.torque_params.latAccelFactor
        # logs: p / i in lateral-acceleration units, error stays v0's lateral-acceleration error
        assert lac_log.p == np.float32(c.pid.p) and lac_log.i == np.float32(c.pid.i)
        measured_curvature = -VM.calc_curvature(math.radians(angle - 0.3), v, params.roll)
        assert abs(lac_log.error - (dc - measured_curvature) * v ** 2) < 1e-3 * max(1.0, abs(lac_log.error))
      assert frozen > 50

  def test_integrator_from_standstill(self):
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS)
    c.inster_limit_observer = _ScriptedLimit()       # no speed rule, and no car-side limit in this check (see TestInsterIntegratorFreeze)
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
      assert abs(i_values[0] - angle_pi_gain(CAR_PI_KI, v) * DT_CTRL * e) < 1e-15
      assert abs(i_values[-1] - 50 * angle_pi_gain(CAR_PI_KI, v) * DT_CTRL * e) < 1e-12
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
    assert i0 < 0.0                                 # e < 0 (wheel right of a straight plan): i < 0, actuators.torque > 0 (left)
    script = _ScriptedLimit(lambda n: 0)
    c.inster_limit_observer = script
    for _ in range(30):
      c.update(True, _cs(10.0, angle=-5.0, pressed=True), VM, params, False, 0.0, None, False, 0.2)
      assert c.pid.i == i0                                                                   # steeringPressed
    # the last request held back towards + (left, the direction of this update): frozen; controlsd's flag alone: no freeze
    script.direction_of = lambda n: 1
    script.direction = 1
    for _ in range(30):
      c.update(True, _cs(10.0, angle=-5.0), VM, params, True, 0.0, None, False, 0.2)
      assert c.pid.i == i0
    script.direction_of = lambda n: 0
    script.direction = 0
    c.update(True, _cs(10.0, angle=-5.0), VM, params, True, 0.0, None, False, 0.2)
    assert c.pid.i < i0                                                                       # flag set, limiter quiet: integrates
    # held back towards - (right) while this update pushes left: integrates (only the limited direction is frozen)
    i1 = c.pid.i
    script.direction_of = lambda n: -1
    script.direction = -1
    c.update(True, _cs(10.0, angle=-5.0), VM, params, False, 0.0, None, False, 0.2)
    assert c.pid.i < i1
    # ... and unwinding through a limit in the other direction is allowed: wheel left of the plan (e > 0) with the push to the right
    i2 = c.pid.i
    c.update(True, _cs(10.0, angle=5.0), VM, params, False, 0.0, None, False, 0.2)
    assert c.pid.i == i2                          # e > 0 pushes right (-), the request was held back towards - : frozen
    script.direction_of = lambda n: 1
    script.direction = 1
    c.update(True, _cs(10.0, angle=5.0), VM, params, False, 0.0, None, False, 0.2)
    assert c.pid.i > i2                           # held back towards + while the update pushes right: unwinds
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
    assert first["F"][0] < -1.5                                     # F did wind at 2 m/s
    assert abs(first["F"][1]) < 1e-3                                # reset: only the first step's I at the re-engagement
    assert abs(first["F"][2]) < 0.5 and abs(first["F"][2] - first["v0_step2"][2]) < 0.5   # ~0 raw, as v0 (was 170.6 raw)
    assert first["F"][3] == angle_pi_log_version(LATERAL_PI, False)

  def test_log_version(self):
    # F drives are identifiable in the logs: torqueState.version 320105 with F on and the selector unset (PI = PI-30Oct base 105 +
    # 1000 x (freeze-rule bit 64 | fixed-factor bit 256); 30 Oct 2026 before the fixed factor 320105, 29 Oct 2026 with G6 64103; before
    # the freeze fix 103; older drives: 102 G2, 101 F's gains with the crawl plan, 100 before it), 0 for v0 otherwise (upstream: 1)
    assert ANGLE_PI_G6_LOG_VERSION == 103 and ANGLE_PI_VERSION_FREEZE_DIR_BIT == 64 and ANGLE_PI_VERSION_FIXED_FACTOR_BIT == 256
    assert ANGLE_PI_G6_LOG_VERSION not in (latcontrol_torque_v0.VERSION, 1)
    for on, expected in ((INSTER_KEYS, 320105), ((ANGLE_PI_KEY,), 320105), (STEP2_KEYS, 0), ((), 0),
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
    # F with both step-2 switches off: v0's linear feed-forward and get_friction (on v0's lateral-acceleration error) stay; compared
    # with v0 at the Inster path's fixed factor F0 (with a switch on, the learned factor is not used: TestInsterFixedFactor)
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=(ANGLE_PI_KEY,))
    assert not c.inster_friction and not c.inster_ff and c.inster_angle_pi
    base, VM_base = _make_v0(HYUNDAI.HYUNDAI_INSTER)
    base_update = base.update_torque_parameters
    base.update_torque_parameters = lambda factor, offset, friction: base_update(F0, offset, friction)
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
            c.inster_limit_observer.reset()                    # a fresh step: no request held back before it
            c.update(True, CS, VM, vp, False, desired_curvature, None, False, 0.2)
            kp, ki = angle_pi_gain(CAR_PI_KP, v), angle_pi_gain(CAR_PI_KI, v)
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
    c.inster_limit_observer.reset()
    CS = _cs(100 / 3.6, angle=1.0)                                                           # wheel 1 deg left of a straight plan
    _, _, lac_log = c.update(True, CS, VM, params, False, 0.0, None, False, 0.2)
    v = CS.vEgo
    kp, ki = angle_pi_gain(CAR_PI_KP, v), angle_pi_gain(CAR_PI_KI, v)
    assert abs(kp - 10.72) < 0.01 and abs(ki - 5.69) < 0.01                                  # the PI (PI-30Oct) at 100 km/h
    assert abs(c.pid.p - kp * math.radians(1.0)) < 1e-12                                     # 0.187 m/s^2
    assert abs(c.pid.i - ki * math.radians(1.0) * DT_CTRL) < 1e-15
    # v0's lateral-acceleration error for the same 1 deg is (curvature error) x v^2: a different quantity
    lat_error = (0.0 + VM.calc_curvature(math.radians(1.0), v, 0.0)) * v ** 2
    assert abs(lac_log.error - lat_error) < 1e-5 and lat_error > 0.0
    assert abs(c.pid.p - kp * lat_error) > 0.5 * c.pid.p                                    # not kp x the lateral-acceleration error
    assert abs(c.pid.p / lac_log.error - kp * VM.get_steer_from_curvature(1.0, v, 0.0) / v ** 2) < 1e-5


# G6 = the PI (27 Oct 2026 alongside G2 as sub-switch InsterAngleErrorPIG6; 29 Oct 2026 controller type PI, the default): the
# Nelder-Mead result's coefficients at full precision, the 6-significant-digit values the car code uses, and the anchor values at
# 0 / 10 / 55 / 115 / 160 km/h
G6_KEY = "InsterAngleErrorPIG6"
G6_KP_SEARCH = (3.4164007100591722, 0.4107633656804734, -0.007152631668639052)
G6_KI_SEARCH = (0.93525, 0.4462301643956044, -0.009480687570989011)
G6_KP = (3.4164, 0.410763, -0.00715263)
G6_KI = (0.93525, 0.44623, -0.00948069)
G6_KP_ANCHOR = (3.4164, 4.50222, 8.02245, 9.23911, 7.54388)
G6_KI_ANCHOR = (0.93525, 2.10162, 5.53976, 5.51527, 2.04041)


class TestInsterAnglePIG6(OpenpilotTestCase):
  """the PI = gains G6 (torqueState.version 103): the default with the selector unset, the legacy G6 bool or selector 0 (PI); the
  selector itself: TestInsterLateralController"""

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
      assert angle_pi_gain(ANGLE_PI_G6_KP, v) < angle_pi_gain(G2_KP, v)
      assert angle_pi_gain(ANGLE_PI_G6_KI, v) > angle_pi_gain(G2_KI, v)
    pi = InsterAnglePI(rate=1 / DT_CTRL, kp=ANGLE_PI_G6_KP, ki=ANGLE_PI_G6_KI)
    for v in (0.0, 4.2, 15.3, 31.9):
      pi.speed = v
      assert pi.k_p == angle_pi_gain(G6_KP, v) and pi.k_i == angle_pi_gain(G6_KI, v) and pi.k_d == 0.0

  def test_switch(self):
    # F on: the PI with the selector unset (also with the legacy G6 bool, before the migration) or at 0; never G2
    for on in ((ANGLE_PI_KEY,), INSTER_KEYS, (ANGLE_PI_KEY, G6_KEY), INSTER_KEYS + (G6_KEY,)):
      for ctrl in (None, LATERAL_PI):
        c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on, ctrl=ctrl)
        assert c.inster_angle_pi and not c.inster_angle_pi_d and isinstance(c.pid, InsterAnglePI) and c.extension._pid is c.pid
        assert (c.pid.kp_coeffs, c.pid.ki_coeffs, c.pid.kd_coeffs) == (CAR_PI_KP, CAR_PI_KI, None), (on, ctrl)
    # the legacy G6 bool changes nothing any more (PI either way)
    assert _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS + (G6_KEY,))) == _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS))
    # G6 bool alone, with NNLC, or on another car: no effect (the run equals the one without the G6 param)
    for car_name, on in ((HYUNDAI.HYUNDAI_INSTER, ()), (HYUNDAI.HYUNDAI_INSTER, STEP2_KEYS),
                         (HYUNDAI.HYUNDAI_INSTER, INSTER_KEYS + ("NeuralNetworkLateralControl",)), (HYUNDAI.HYUNDAI_KONA_EV, INSTER_KEYS)):
      c, VM = _make_v0(car_name, on=on + (G6_KEY,))
      assert not c.inster_angle_pi and type(c.pid) is PIDController
      assert _run(c, VM) == _run(*_make_v0(car_name, on=on)), (car_name, on)
    # not an Inster switch of its own: alone it leaves the Lateral Jerk Torque Controller on
    _make_v0(HYUNDAI.HYUNDAI_INSTER, on=(G6_KEY, "LateralJerkTorqueController"))
    assert Params().get_bool("LateralJerkTorqueController")

  def test_log_version(self):
    # PI drives are identifiable in the logs: torqueState.version 320105 (PI-30Oct base 105 + 1000 x (freeze-rule bit 64 | fixed-factor
    # bit 256); G6: 64103 on 29 Oct 2026, 103 before the freeze fix), 0 for v0
    # (upstream: 1); 102 (G2) is no longer produced
    assert ANGLE_PI_G6_LOG_VERSION == 103
    for on, expected in ((INSTER_KEYS + (G6_KEY,), 320105), ((ANGLE_PI_KEY, G6_KEY), 320105), (INSTER_KEYS, 320105), ((ANGLE_PI_KEY,), 320105),
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
        c.inster_limit_observer.reset()
        c.update(True, CS, VM, _vp(), False, 0.0, None, False, 0.2)
        kp, ki = angle_pi_gain(CAR_PI_KP, v), angle_pi_gain(CAR_PI_KI, v)
        assert abs(c.pid.p - kp * e) <= 1e-12 * max(1.0, abs(kp * e)), kmh
        i_new = ki * DT_CTRL * e
        test, lim = c.pid.p + i_new + c.pid.f, c.pid.pos_limit
        expected_i = min(max(i_new, 0.0 if test < -lim else -lim), 0.0 if test > lim else lim)
        assert abs(c.pid.i - expected_i) <= 1e-14, kmh
        n_integrating += expected_i == i_new
    assert n_integrating > 15


# P6 = the PID (27 Oct 2026 selector InsterAngleErrorPIGains = 2; 29 Oct 2026 controller type PID): the PID search's coefficients
# at full precision, the 6-significant-digit
# values the car code uses, and the anchor values at 0 / 10 / 55 / 115 / 160 km/h
GAINS_KEY = "InsterAngleErrorPIGains"       # legacy gain-set selector (0 G2, 1 G6, 2 P6), migrated to CTRL_KEY
P6_KP_SEARCH = (3.33059449704142, 0.5058280544378698, -0.009431306414201182)
P6_KI_SEARCH = (0.93525, 0.43611400879120865, -0.00917132202197802)
P6_KD_SEARCH = (0.36172704733727806, 0.0008542736094674528, -8.687034319526545e-06)
P6_KP = (3.33059, 0.505828, -0.00943131)
P6_KI = (0.93525, 0.436114, -0.00917132)
P6_KD = (0.361727, 0.000854274, -8.68703e-06)
P6_KP_ANCHOR = (3.33059, 4.66290, 8.85715, 9.86483, 7.18209)
P6_KI_ANCHOR = (0.93525, 2.07591, 5.45742, 5.50782, 2.20191)
P6_KD_ANCHOR = (0.361727, 0.364033, 0.372751, 0.380152, 0.382535)

# The gain sets the car runs since 30 Oct 2026 (owner: refined fits go to the car as soon as they exist): PI-30Oct / PID-30Oct = the
# refined plateau centroids (4 simulator members, +-1 % jitter) of the PI / PID refits under the large-error objective with the
# integrator-freeze rule of 29 Oct 2026; the search's coefficients at full precision and the 6-significant-digit values the car uses
CAR_PI_KP_SEARCH = (3.9714508767002017, 0.5786595102717313, -0.012083981418475878)
CAR_PI_KI_SEARCH = (0.9352500000000001, 0.4252513282125513, -0.009146448037856603)
CAR_PID_KP_SEARCH = (3.5363805804058277, 0.6377225397370454, -0.012945495431809693)
CAR_PID_KI_SEARCH = (0.93525, 0.472450086742904, -0.010106428772096797)
CAR_PID_KD_SEARCH = (0.3999999999999999, -0.004485739014260417, 4.623027432735433e-05)
CAR_PI_KP = (3.97145, 0.57866, -0.012084)
CAR_PI_KI = (0.93525, 0.425251, -0.00914645)
CAR_PID_KP = (3.53638, 0.637723, -0.0129455)
CAR_PID_KI = (0.93525, 0.47245, -0.0101064)
CAR_PID_KD = (0.4, -0.00448574, 4.62303e-05)


class TestInsterGainSets30Oct(OpenpilotTestCase):
  """the gain sets the car's PI / PID run since 30 Oct 2026 (PI-30Oct base version 105, PID-30Oct 106); G6 / P6 stay for older logs"""

  def test_coded_gains(self):
    assert (ANGLE_PI_PI_KP, ANGLE_PI_PI_KI) == (CAR_PI_KP, CAR_PI_KI)
    assert (ANGLE_PI_PID_KP, ANGLE_PI_PID_KI, ANGLE_PI_PID_KD, ANGLE_PI_D_TAU) == (CAR_PID_KP, CAR_PID_KI, CAR_PID_KD, 0.03)
    for full, coded, tol in ((CAR_PI_KP_SEARCH, CAR_PI_KP, 1e-5), (CAR_PI_KI_SEARCH, CAR_PI_KI, 2e-5), (CAR_PID_KP_SEARCH, CAR_PID_KP, 1e-5),
                             (CAR_PID_KI_SEARCH, CAR_PID_KI, 3e-5), (CAR_PID_KD_SEARCH, CAR_PID_KD, 1e-6)):
      assert coded == tuple(float(f"{c:.6g}") for c in full)
      for v in np.linspace(0.0, 160 / 3.6, 1601):                      # rounding: within tol relative up to 160 km/h
        assert abs(angle_pi_gain(coded, v) / angle_pi_gain(full, v) - 1.0) < tol, (coded, v)

  def test_never_negative(self):
    # kp > 0 below 194 km/h, ki > 0 below 174 km/h (clipped at 0 above), kd > 0 everywhere
    for v in np.linspace(0.0, 200 / 3.6, 2001):
      for c in (CAR_PI_KP, CAR_PI_KI, CAR_PID_KP, CAR_PID_KI, CAR_PID_KD):
        assert angle_pi_gain(c, v) >= 0.0
      assert angle_pi_gain(CAR_PID_KD, v) > 0.0
      if v < 174 / 3.6:
        assert min(angle_pi_gain(c, v) for c in (CAR_PI_KP, CAR_PI_KI, CAR_PID_KP, CAR_PID_KI)) > 0.0
    assert abs(angle_pi_gain(CAR_PID_KD, 0.0) - 0.4) < 1e-12

  def test_versions(self):
    assert (ANGLE_PI_PI_BASE_VERSION, ANGLE_PI_PID_BASE_VERSION) == (105, 106)
    assert ANGLE_PI_PI_BASE_VERSION not in (ANGLE_PI_G6_LOG_VERSION, ANGLE_PI_P6_BASE_VERSION, 100, 101, 102)
    assert ANGLE_PI_PID_BASE_VERSION not in (ANGLE_PI_G6_LOG_VERSION, ANGLE_PI_P6_BASE_VERSION, 100, 101, 102)
    expected = {(0, False): 320105, (1, False): 336106, (0, True): 328105, (1, True): 344106}
    for (ctrl, soft), version in expected.items():
      assert angle_pi_log_version(ctrl, soft) == version

  def test_car_runs_the_new_sets(self):
    for ctrl, kp, ki, kd in ((0, CAR_PI_KP, CAR_PI_KI, None), (1, CAR_PID_KP, CAR_PID_KI, CAR_PID_KD), (None, CAR_PI_KP, CAR_PI_KI, None)):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, ctrl=ctrl)
      assert (c.pid.kp_coeffs, c.pid.ki_coeffs, c.pid.kd_coeffs) == (kp, ki, kd), ctrl


def _theta_meas(VM, v, angle, offset, roll):
  """P6's D input: the measured wheel angle (rad, + right) as the controller computes it, R(v) x v0's measured curvature"""
  return VM.get_steer_from_curvature(1.0, v, 0.0) * -VM.calc_curvature(math.radians(angle - offset), v, roll)


def _lp2_rates(thetas, tau=0.03):
  """the kit's LP2 derivative (candidates_pid.CarPID._rate, d_mode 'meas', without the sign): states start at the first value"""
  a = 1.0 - math.exp(-DT_CTRL / tau)
  x1 = x2 = thetas[0]
  out = []
  for th in thetas:
    prev = x2
    x1 += a * (th - x1)
    x2 += a * (x1 - x2)
    out.append((x2 - prev) / DT_CTRL)
  return out


def _p6_drive(c, VM, n, pressed=lambda k: False, active=lambda k: True, limited=lambda k: False):
  """a wavy wheel at changing speed; returns per step (CS, params, active, pid.p, pid.i, pid.d, pid.f, pid.control)"""
  out = []
  for k in range(n):
    t = k * DT_CTRL
    v = max(0.0, 3.0 + 25.0 * (1.0 + math.sin(0.05 * t)) / 2.0)
    angle = 30.0 * math.sin(1.3 * t) + 4.0 * math.sin(7.0 * t)
    CS = _cs(v, angle, pressed=pressed(k), rate=abs(30.0 * 1.3 * math.cos(1.3 * t)))
    vp = _vp(0.02 * math.sin(0.03 * t), 0.4)
    dc = 0.01 * math.sin(0.9 * t)
    c.update(active(k), CS, VM, vp, limited(k), dc, None, False, 0.2)
    out.append((CS, vp, active(k), c.pid.p, c.pid.i, c.pid.d, c.pid.f, c.pid.control))
  return out


class TestInsterLateralController(OpenpilotTestCase):
  """selector InsterLateralController (0 = PI = G6, 1 = PID = P6) of InsterAngleErrorPI; PID = PI + D on the measured wheel-angle
  rate. One option per controller type (owner, 29 Oct 2026); G2 is not selectable"""

  def test_selector_rules(self):
    params = Params()
    assert (LATERAL_PI, LATERAL_PID, LATERAL_CONTROLLER_PARAM, ANGLE_PI_GAINS_PARAM, ANGLE_PI_GAINS_LEGACY_P6) == (0, 1, CTRL_KEY, GAINS_KEY, 2)
    # (legacy G6 bool, legacy gain-set selector, new selector) -> controller type; the new selector wins, unset falls back to the
    # legacy selector (P6 -> PID, else PI; before the migration ran), unknown values -> PI
    cases = ((False, None, None, 0), (True, None, None, 0), (False, 0, None, 0), (False, 1, None, 0), (True, 2, None, 1),
             (False, 2, None, 1), (False, 7, None, 0), (False, None, 0, 0), (False, None, 1, 1), (False, 2, 0, 0), (False, 0, 1, 1),
             (False, None, 2, 0), (False, None, -1, 0), (True, 2, 5, 0))
    for legacy, old, new, expected in cases:
      params.put_bool(G6_KEY, legacy, block=True)
      for key, value in ((GAINS_KEY, old), (CTRL_KEY, new)):
        if value is None:
          params.remove(key)
        else:
          params.put(key, value, block=True)
      assert angle_lateral_controller(params) == expected, (legacy, old, new)
    params.remove(CTRL_KEY)
    params.remove(GAINS_KEY)
    assert params.get(CTRL_KEY, return_default=True) == LATERAL_PI       # the UI's default
    for ctrl, legacy_gains, kp, ki, kd, version in ((0, None, CAR_PI_KP, CAR_PI_KI, None, 320105), (1, None, CAR_PID_KP, CAR_PID_KI, CAR_PID_KD, 336106),
                                                    (None, None, CAR_PI_KP, CAR_PI_KI, None, 320105), (None, 0, CAR_PI_KP, CAR_PI_KI, None, 320105),
                                                    (None, 2, CAR_PID_KP, CAR_PID_KI, CAR_PID_KD, 336106)):
      for on in ((ANGLE_PI_KEY,), INSTER_KEYS):
        c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on, ctrl=ctrl, legacy_gains=legacy_gains)
        assert c.inster_angle_pi and (c.pid.kp_coeffs, c.pid.ki_coeffs, c.pid.kd_coeffs) == (kp, ki, kd)
        assert c.inster_angle_pi_d == (kd is not None) and c.inster_angle_pi_version == version
    # without F, with NNLC, or on another car the selector does nothing (the run equals the one without it)
    for car_name, on in ((HYUNDAI.HYUNDAI_INSTER, ()), (HYUNDAI.HYUNDAI_INSTER, STEP2_KEYS),
                         (HYUNDAI.HYUNDAI_INSTER, INSTER_KEYS + ("NeuralNetworkLateralControl",)), (HYUNDAI.HYUNDAI_KONA_EV, INSTER_KEYS)):
      for ctrl in (0, 1):
        c, VM = _make_v0(car_name, on=on, ctrl=ctrl)
        assert not c.inster_angle_pi_d and type(c.pid) is PIDController
        assert _run(c, VM) == _run(*_make_v0(car_name, on=on)), (car_name, on, ctrl)
    # PI = the default = the legacy G6 bool = the legacy selector at G2 or G6; PID = the legacy selector at P6; PID differs from PI
    pi = _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, ctrl=0))
    for kw in ({}, {"legacy_gains": 0}, {"legacy_gains": 1}):
      assert pi == _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, **kw)), kw
    assert pi == _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS + (G6_KEY,)))
    pid = _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, ctrl=1))
    assert pid != pi and pid == _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, legacy_gains=2))
    # not an Inster switch of its own: alone it leaves the Lateral Jerk Torque Controller on
    _make_v0(HYUNDAI.HYUNDAI_INSTER, on=("LateralJerkTorqueController",), ctrl=1)
    assert Params().get_bool("LateralJerkTorqueController")

  def test_p6_gains(self):
    assert (ANGLE_PI_P6_KP, ANGLE_PI_P6_KI, ANGLE_PI_P6_KD, ANGLE_PI_D_TAU) == (P6_KP, P6_KI, P6_KD, 0.03)
    for full, coded in ((P6_KP_SEARCH, P6_KP), (P6_KI_SEARCH, P6_KI), (P6_KD_SEARCH, P6_KD)):
      assert coded == tuple(float(f"{c:.6g}") for c in full)
      for v in np.linspace(0.0, 160 / 3.6, 1601):                      # rounding: within 1e-5 relative up to 160 km/h
        assert abs(angle_pi_gain(coded, v) / angle_pi_gain(full, v) - 1.0) < 1e-5, v
    for kmh, kp, ki, kd in zip(G2_ANCHOR_KMH, P6_KP_ANCHOR, P6_KI_ANCHOR, P6_KD_ANCHOR, strict=True):
      v = kmh / 3.6
      assert abs(angle_pi_gain(P6_KP, v) - kp) < 1e-5 and abs(angle_pi_gain(P6_KI, v) - ki) < 1e-5, kmh
      assert abs(angle_pi_gain(P6_KD, v) - kd) < 1e-6, kmh
    # never negative from 0 to 200 km/h: kp and kd > 0 throughout (roots 214 / 933 km/h), ki > 0 below 178.6 km/h, clipped at 0 above
    for v in np.linspace(0.0, 200 / 3.6, 2001):
      assert angle_pi_gain(P6_KP, v) > 0.0 and angle_pi_gain(P6_KD, v) > 0.0 and angle_pi_gain(P6_KI, v) >= 0.0
      if v < 178 / 3.6:
        assert angle_pi_gain(P6_KI, v) > 0.0
    assert angle_pi_gain(P6_KI, 179 / 3.6) == 0.0
    # the PI keeps kd 0 (common/pid.py's own k_d)
    for pi in (InsterAnglePI(rate=1 / DT_CTRL, kp=G6_KP, ki=G6_KI),):
      for v in (0.0, 15.3, 40.0):
        pi.speed = v
        assert pi.kd_coeffs is None and pi.k_d == 0.0
    pi = InsterAnglePI(rate=1 / DT_CTRL, kp=P6_KP, ki=P6_KI, kd=P6_KD)
    for v in (0.0, 4.2, 15.3, 31.9):
      pi.speed = v
      assert (pi.k_p, pi.k_i, pi.k_d) == (angle_pi_gain(P6_KP, v), angle_pi_gain(P6_KI, v), angle_pi_gain(P6_KD, v))

  def test_versions(self):
    # base versions (before the freeze fix the logged values); logged now with the freeze-rule bit 64 and the fixed-factor bit 256:
    # PI 320105, PID 336106 (gain sets PI-30Oct / PID-30Oct)
    assert (ANGLE_PI_G6_LOG_VERSION, ANGLE_PI_P6_LOG_VERSION) == (103, 16104)
    assert ANGLE_PI_P6_LOG_VERSION == ANGLE_PI_P6_BASE_VERSION + 1000 * ANGLE_PI_VERSION_D_BIT and ANGLE_PI_VERSION_D_BIT == 16
    for on, ctrl, expected in ((INSTER_KEYS, 1, 336106), ((ANGLE_PI_KEY,), 1, 336106), (INSTER_KEYS, 0, 320105), (INSTER_KEYS, None, 320105),
                               (STEP2_KEYS, 1, 0), ((), 1, 0), (INSTER_KEYS + ("NeuralNetworkLateralControl",), 1, 0)):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on, ctrl=ctrl)
      for active in (True, False):
        _, _, lac_log = c.update(active, _cs(10.0, 1.0), VM, _vp(), False, 0.0, None, False, 0.2)
        assert lac_log.version == expected, (on, ctrl, active)

  def test_p6_rule(self):
    # every step through the real update(): p = kp e, d = -kd d/dt LP2(theta_meas), i with common/pid.py's anti-windup incl. d,
    # control = clip(p + i + d + f); D from the first step (settled filter: d = 0 on the first step). Steps whose last request was
    # held back (scripted: every 7th towards +, every 7th + 4 towards -) freeze I if the update pushes that way, and keep D
    def direction(k):
      return 1 if k % 7 == 3 else (-1 if k % 7 == 0 else 0)
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, ctrl=1)
    c.inster_limit_observer = _ScriptedLimit(direction)
    VM.update_params(1.0, 14.0)
    steps = _p6_drive(c, VM, 1500, limited=lambda k: k % 5 == 1)   # controlsd's flag: no longer used by the PID's integrator
    def limited(k):
      e = steps[k][3] / angle_pi_gain(CAR_PID_KP, steps[k][0].vEgo)
      return freeze_angle_integrator(False, direction(k), e)
    rates = _lp2_rates([_theta_meas(VM, CS.vEgo, CS.steeringAngleDeg, vp.angleOffsetDeg, vp.roll) for CS, vp, *_ in steps])
    i_prev = 0.0
    lim = c.pid.pos_limit
    assert steps[0][5] == 0.0
    big = big_limited = 0
    for k, ((CS, _, _, p, i, d, f, control), rate) in enumerate(zip(steps, rates, strict=True)):
      v = CS.vEgo
      assert abs(d - angle_pi_gain(CAR_PID_KD, v) * -rate) <= 1e-12 * max(1.0, abs(d)), (k, v)
      if limited(k):
        expected_i = i_prev                             # frozen: the last request held back in the direction of this update
      else:
        i_new = i_prev + angle_pi_gain(CAR_PID_KI, v) * DT_CTRL * (p / angle_pi_gain(CAR_PID_KP, v))
        test = p + i_new + d + f
        expected_i = min(max(i_new, i_prev if test < -lim else -lim), i_prev if test > lim else lim)
      assert abs(i - expected_i) <= 1e-12, (k, v)
      assert control == min(max(p + i + d + f, -lim), lim)
      big += abs(d) > 0.05
      big_limited += limited(k) and abs(d) > 0.05
      i_prev = i
    assert big > 300 and big_limited > 30               # D is not negligible on this drive (m/s^2), also on steer-limited steps

  def test_p6_d_zero_while_pressed(self):
    # D = 0 while steeringPressed (not simulated; it would resist the driver's hand); the filter keeps running, so D is the same as
    # without the press on every other step
    def pressed(k):
      return 400 <= k < 520
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, ctrl=1)
    VM.update_params(1.0, 14.0)
    steps = _p6_drive(c, VM, 900, pressed=pressed)
    rates = _lp2_rates([_theta_meas(VM, CS.vEgo, CS.steeringAngleDeg, vp.angleOffsetDeg, vp.roll) for CS, vp, *_ in steps])
    for k, ((CS, _, _, _, _, d, _, _), rate) in enumerate(zip(steps, rates, strict=True)):
      if pressed(k):
        assert d == 0.0, k
      else:
        assert abs(d - angle_pi_gain(CAR_PID_KD, CS.vEgo) * -rate) <= 1e-12 * max(1.0, abs(d)), k
    assert max(abs(angle_pi_gain(CAR_PID_KD, CS.vEgo) * rate) for (CS, *_), rate in list(zip(steps, rates, strict=True))[400:520]) > 0.1

  def test_p6_filter_restart_after_inactive(self):
    # the D filter restarts settled on the first active step after lateral was inactive: d = 0 there (no kick from the wheel moved
    # meanwhile), then the kit's LP2 from that value
    def active(k):
      return not (300 <= k < 450)
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, ctrl=1)
    VM.update_params(1.0, 14.0)
    steps = _p6_drive(c, VM, 700, active=active)
    assert all(s[5] == 0.0 and s[7] == 0 for s in steps[300:450])
    assert steps[450][5] == 0.0 and abs(steps[449 - 150][5]) > 0.0
    thetas = [_theta_meas(VM, CS.vEgo, CS.steeringAngleDeg, vp.angleOffsetDeg, vp.roll) for CS, vp, *_ in steps[450:]]
    for (CS, _, _, _, _, d, _, _), rate in zip(steps[450:], _lp2_rates(thetas), strict=True):
      assert abs(d - angle_pi_gain(CAR_PID_KD, CS.vEgo) * -rate) <= 1e-12 * max(1.0, abs(d))


# Option 5 "InsterSoftSmallErrorP" (28 Oct 2026): mild parameters fitted with P6; the design's sigma written out independently
O5_KEY = "InsterSoftSmallErrorP"


def _sigma_design(e, v, r_rad, k_now, k_past, rho=0.5, e0=0.11, a0=0.3, gamma=1.2):
  """options design v2.1 section 5: v_c = max(v, 7 m/s); e_a = |e| v_c^2 / R_rad; a_p = max(|k_des|, |k_(ld ago)|) v_c^2; phi(x) = x^2 / (1 + x^2)"""
  def phi(x):
    return x * x / (1 + x * x)
  vc = max(v, 7.0)
  e_a = abs(e) * vc ** 2 / max(r_rad, math.radians(1.0))
  a_p = max(abs(k_now), abs(k_past)) * vc ** 2
  return (1 - (1 - rho) * (1 - phi(e_a / e0)) * (1 - phi(a_p / a0))) * (1 + (gamma - 1) * phi(a_p / a0))


def _o5_drive(c, VM, n, limited=lambda k: False):
  """small wheel-angle errors around a gently curving plan (no output clipping); returns per step (CS, dc, lat_delay, p, i, d, f)"""
  out = []
  for k in range(n):
    t = k * DT_CTRL
    v = 2.0 + 30.0 * (1.0 + math.sin(0.04 * t)) / 2.0
    dc = (0.004 * math.sin(0.35 * t) + 0.0015 * math.sin(1.7 * t)) * min(1.0, 1.5 / (0.0055 * v * v))   # |a| <= 1.5 m/s^2
    plan = math.degrees(VM.get_steer_from_curvature(-dc, v, 0.0))
    CS = _cs(v, plan + 1.5 * math.sin(2.3 * t) + 0.4 * math.sin(9.0 * t))
    lat_delay = 0.2 + 0.05 * math.sin(0.1 * t)
    c.update(True, CS, VM, _vp(), limited(k), dc, None, False, lat_delay)
    out.append((CS, dc, lat_delay, c.pid.p, c.pid.i, c.pid.d, c.pid.f))
  return out


class TestInsterSoftSmallErrorP(OpenpilotTestCase):
  """option 5: with InsterAngleErrorPI on, p = kp(v) sigma e for any gain set; I and D unchanged"""

  def test_sigma_formula(self):
    assert (SOFT_P_RHO, SOFT_P_E0, SOFT_P_A0, SOFT_P_GAMMA, SOFT_P_V_MIN, SOFT_P_PARAM) == (0.5, 0.11, 0.3, 1.2, 7.0, O5_KEY)
    rng = random.Random(3)
    for _ in range(20000):
      e = rng.choice([0.0, rng.gauss(0, 0.002), rng.gauss(0, 0.02), rng.gauss(0, 0.2)])
      v = rng.choice([0.0, rng.uniform(0, 7), rng.uniform(7, 45)])
      r = rng.choice([rng.uniform(0.2, 0.6), rng.uniform(0.0, 0.02)])      # rad per 1/m (the second: the 1 deg floor)
      k1, k2 = rng.gauss(0, 0.01), rng.gauss(0, 0.01)
      s = soft_p_sigma(e, v, r, k1, k2)
      assert abs(s - _sigma_design(e, v, r, k1, k2)) <= 1e-12, (e, v, r, k1, k2)
      assert SOFT_P_RHO - 1e-12 <= s <= SOFT_P_GAMMA + 1e-12                 # rho <= sigma <= gamma
    assert soft_p_sigma(0.0, 20.0, 0.4, 0.0, 0.0) == SOFT_P_RHO                # straight, no error: rho
    assert abs(soft_p_sigma(0.0, 20.0, 0.4, 1.0, 0.0) - SOFT_P_GAMMA) < 1e-4     # tight curve: -> gamma
    assert soft_p_sigma(0.01, 20.0, 0.4, 0.003, 0.001) == soft_p_sigma(-0.01, 20.0, 0.4, -0.001, -0.003)   # |e|, max of |k|

  def test_p_strictly_increasing(self):
    # p(e) = sigma(e) e strictly increasing in e for every plan; slope rho at e = 0 on a straight (sigma(0) with a plan)
    VM = VehicleModel(interfaces[HYUNDAI.HYUNDAI_INSTER].get_non_essential_params(HYUNDAI.HYUNDAI_INSTER))
    for v in (0.0, 5.0, 15.0, 30.0, 45.0):
      r = VM.get_steer_from_curvature(1.0, v, 0.0)
      es = np.linspace(-0.5, 0.5, 20001)
      for k_plan in (0.0, 0.0005, 0.002, 0.01, 0.05):
        p = np.array([soft_p_sigma(e, v, r, k_plan, 0.0) * e for e in es])
        assert np.all(np.diff(p) > 0.0), (v, k_plan)
        h = 1e-9
        slope0 = (soft_p_sigma(h, v, r, k_plan, 0.0) * h - soft_p_sigma(-h, v, r, k_plan, 0.0) * -h) / (2 * h)
        assert abs(slope0 - soft_p_sigma(0.0, v, r, k_plan, 0.0)) < 1e-6
      assert soft_p_sigma(0.0, v, r, 0.0, 0.0) == SOFT_P_RHO                   # slope rho kp at 0 on a straight

  def test_switch_rules(self):
    for ctrl, cls_d in ((0, False), (1, True)):
      for on in ((ANGLE_PI_KEY, O5_KEY), INSTER_KEYS + (O5_KEY,)):
        c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on, ctrl=ctrl)
        assert c.inster_angle_pi and c.inster_soft_p and c.inster_plan_history and c.inster_angle_pi_d == cls_d
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, ctrl=ctrl)
      assert not c.inster_soft_p and c.pid.p_sigma is None
    # without F, with NNLC, or on another car: no effect (the run equals the one without the param)
    for car_name, on in ((HYUNDAI.HYUNDAI_INSTER, ()), (HYUNDAI.HYUNDAI_INSTER, STEP2_KEYS), (HYUNDAI.HYUNDAI_INSTER, ("InsterSpeedFeedforward",)),
                         (HYUNDAI.HYUNDAI_INSTER, INSTER_KEYS + ("NeuralNetworkLateralControl",)), (HYUNDAI.HYUNDAI_KONA_EV, INSTER_KEYS)):
      c, VM = _make_v0(car_name, on=on + (O5_KEY,), ctrl=1)
      assert not c.inster_soft_p
      assert _run(c, VM) == _run(*_make_v0(car_name, on=on, ctrl=1)), (car_name, on)
    # changes the output with F on; F without the stiction term keeps its own plan history for option 5
    for on in (INSTER_KEYS, (ANGLE_PI_KEY,)):
      assert _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=on + (O5_KEY,), ctrl=1)) != _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=on, ctrl=1))
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=(ANGLE_PI_KEY, O5_KEY), ctrl=1)
    assert not c.inster_friction and c.inster_plan_history and not hasattr(c, "inster_stiction")
    # not an Inster switch of its own: alone it leaves the Lateral Jerk Torque Controller on
    assert O5_KEY not in inster_lateral.INSTER_SWITCHES
    _make_v0(HYUNDAI.HYUNDAI_INSTER, on=(O5_KEY, "LateralJerkTorqueController"), ctrl=1)
    assert Params().get_bool("LateralJerkTorqueController")

  def test_versions(self):
    assert ANGLE_PI_VERSION_SOFT_P_BIT == 8 and ANGLE_PI_VERSION_D_BIT == 16
    # incl. the freeze-rule bit 64 and the fixed-factor bit 256 (128 stays free)
    expected = {(0, False): 320105, (1, False): 336106, (0, True): 328105, (1, True): 344106}
    for (ctrl, soft), version in expected.items():
      assert angle_pi_log_version(ctrl, soft) == version, (ctrl, soft)
    always = ANGLE_PI_VERSION_FREEZE_DIR_BIT | ANGLE_PI_VERSION_FIXED_FACTOR_BIT
    assert angle_pi_log_version(LATERAL_PID, True) == ANGLE_PI_PID_BASE_VERSION + 1000 * (ANGLE_PI_VERSION_D_BIT | ANGLE_PI_VERSION_SOFT_P_BIT
                                                                                         | always)
    assert angle_pi_log_version(LATERAL_PID, False) == ANGLE_PI_PID_BASE_VERSION + 1000 * (ANGLE_PI_VERSION_D_BIT | always)
    assert angle_pi_log_version(LATERAL_PI, False) == ANGLE_PI_PI_BASE_VERSION + 1000 * always
    assert not any(angle_pi_log_version(ctrl, soft) // 1000 & 128 for ctrl, soft in expected)
    for (ctrl, soft), version in expected.items():
      for on in (INSTER_KEYS, (ANGLE_PI_KEY,)):
        c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on + ((O5_KEY,) if soft else ()), ctrl=ctrl)
        for active in (True, False):
          _, _, lac_log = c.update(active, _cs(10.0, 1.0), VM, _vp(), False, 0.0, None, False, 0.2)
          assert lac_log.version == version, (ctrl, soft, on, active)
    for on in (STEP2_KEYS + (O5_KEY,), (O5_KEY,), INSTER_KEYS + (O5_KEY, "NeuralNetworkLateralControl")):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on, ctrl=1)
      assert c.update(True, _cs(10.0, 1.0), VM, _vp(), False, 0.0, None, False, 0.2)[2].version == 0

  def test_rule_through_update(self):
    # every step: p = kp(v) sigma e with sigma from the raw e and the plan now / lat_delay ago (option 5's own 1 s history: the
    # stiction term is off here, so f does not depend on the output); I and D are the same as without option 5 (same inputs, no
    # clipping): no sigma on I, D untouched
    on = ("InsterSpeedFeedforward", ANGLE_PI_KEY)
    for ctrl, kp_c in ((0, CAR_PI_KP), (1, CAR_PID_KP)):
      def limited(k):
        return k % 11 == 5
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on + (O5_KEY,), ctrl=ctrl)
      ref, VM_ref = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on, ctrl=ctrl)
      # the same limit directions for both: the car-side ramp sees different outputs (P differs), so the real observer would freeze
      # them on different steps (at the fixed factor F0 this drive touches the 3 raw / step ramp)
      for x in (c, ref):
        x.inster_limit_observer = _ScriptedLimit(lambda n: (0, 0, 1, 0, 0, -1, 0)[n % 7])
      VM.update_params(1.0, 14.0)
      VM_ref.update_params(1.0, 14.0)
      a = _o5_drive(c, VM, 2500, limited)
      b = _o5_drive(ref, VM_ref, 2500, limited)
      hist = [0.0] * 100
      lim = c.pid.pos_limit
      sigmas = []
      for (CS, dc, ld, p, i, d, f), (_, _, _, p_ref, i_ref, d_ref, f_ref) in zip(a, b, strict=True):
        past = hist[-int(np.clip(ld / DT_CTRL, 1, 100))]
        hist = hist[1:] + [dc]
        v = CS.vEgo
        e = angle_error(VM, v, dc, -VM.calc_curvature(math.radians(CS.steeringAngleDeg), v, 0.0))
        s = _sigma_design(e, v, VM.get_steer_from_curvature(1.0, v, 0.0), dc, past)
        sigmas.append(s)
        assert abs(p - angle_pi_gain(kp_c, v) * s * e) <= 1e-12 * max(1.0, abs(p)), (ctrl, v)
        assert abs(p_ref - angle_pi_gain(kp_c, v) * e) <= 1e-12 * max(1.0, abs(p_ref)), (ctrl, v)
        assert i == i_ref and d == d_ref and f == f_ref, ctrl            # I (incl. the steer-limited freezes) and D unchanged
        assert abs(p + i + d + f) < lim                                  # no clipping on this drive
      assert min(sigmas) < 0.6 and max(sigmas) > 1.05, (ctrl, min(sigmas), max(sigmas))


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
    assert ANGLE_PI_G6_LOG_VERSION == 103                   # F drives with the crawl plan are identifiable in the logs (103 PI, 16104 PID; 101 / 102 older)

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


def _car_params(car_name):
  return interfaces[car_name].get_non_essential_params(car_name).as_reader()


class TestInsterIntegratorFreeze(OpenpilotTestCase):
  """the steering-angle controller's integrator freeze (29 Oct 2026, owner: a fix for every PI / PID variant): the car-side limiter
  re-run on the controller's own requests (same frame, no carOutput lag), frozen only in the limited direction or while pressed"""

  def test_observer_is_the_car_side_limiter(self):
    # same command as the Hyundai carcontroller for random request / driver-torque sequences (cap 409, ramp 3 / 5, driver rule)
    from opendbc.car.hyundai.values import CarControllerParams
    from opendbc.car.lateral import apply_driver_steer_torque_limits
    CP = _car_params(HYUNDAI.HYUNDAI_INSTER)
    ccp = CarControllerParams(CP)
    assert (ccp.STEER_MAX, ccp.STEER_FULL_SCALE, ccp.STEER_DELTA_UP, ccp.STEER_DELTA_DOWN) == (INSTER_STEER_MAX, INSTER_STEER_FULL_SCALE, 3, 5)
    obs = InsterSteerLimitObserver.for_car(CP)
    rng = random.Random(11)
    last = 0
    n_limited = {1: 0, -1: 0, 0: 0}
    for k in range(20000):
      torque = rng.choice((rng.uniform(-1.2, 1.2), 0.3 * math.sin(k / 40.0), rng.gauss(0.0, 0.02)))
      driver = rng.choice((0.0, rng.uniform(-200.0, 200.0)))
      new = int(round(torque * INSTER_STEER_FULL_SCALE))
      apply = apply_driver_steer_torque_limits(new, last, driver, ccp)
      last = apply
      d = obs.update(torque, driver)
      assert obs.apply_torque_last == apply, k
      assert d == (1 if new - apply > LIMIT_TOL_RAW else (-1 if new - apply < -LIMIT_TOL_RAW else 0)), k
      n_limited[d] += 1
    assert min(n_limited.values()) > 1000
    obs.reset()
    assert obs.apply_torque_last == 0 and obs.direction == 0

  def test_no_lag_artefact(self):
    # a request moving at <= 3 raw per step (inside the ramp) is never limited, however long it moves; controlsd's flag (request vs a
    # carOutput 2 frames older) read such a request as limited once it moved by more than ~2 raw per step
    obs = InsterSteerLimitObserver.for_car(_car_params(HYUNDAI.HYUNDAI_INSTER))
    for k in range(120):                                            # 0 -> 360 raw at 3 raw / step, then back at 5 raw / step
      assert obs.update(3.0 * k / INSTER_STEER_FULL_SCALE, 0.0) == 0
    for k in range(72):
      assert obs.update((357.0 - 5.0 * k) / INSTER_STEER_FULL_SCALE, 0.0) == 0
    assert obs.update(400.0 / INSTER_STEER_FULL_SCALE, 0.0) == 1    # a jump the ramp holds back: limited towards +
    assert obs.update(500.0 / INSTER_STEER_FULL_SCALE, 0.0) == 1    # beyond the 409 cap: limited towards +
    obs.reset()
    assert obs.update(-0.5, 0.0) == -1

  def test_freeze_rule(self):
    for pressed in (False, True):
      for d in (-1, 0, 1):
        for e in (-0.01, 0.0, 0.01):
          # e > 0: output to the right = actuators.torque negative, so the update pushes towards -1
          expected = pressed or (d != 0 and e != 0.0 and d == (1 if e < 0.0 else -1))
          assert freeze_angle_integrator(pressed, d, e) == expected, (pressed, d, e)

  def test_ramp_through_update(self):
    # PI and PID through the real update(): a 6 deg wheel error at 30 km/h engaged from 0. The request jumps far beyond the 3 raw / step
    # ramp: the integrator stays frozen while the car-side command ramps up (the request held back in the direction of the update),
    # and integrates again once the command has caught up; controlsd's flag (passed True throughout) does not freeze it
    for ctrl in (0, 1):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, ctrl=ctrl)
      VM.update_params(1.0, 14.0)
      frozen = []
      for _ in range(600):
        i_prev = c.pid.i
        c.update(True, _cs(30 / 3.6, angle=-6.0), VM, _vp(), True, 0.0, None, False, 0.2)
        frozen.append(bool(c.pid.i == i_prev))
      assert not frozen[0]                                         # first step: nothing held back yet
      assert all(frozen[1:20])                                     # the ramp holds the request back in the push direction
      caught_up = frozen.index(False, 1)
      assert caught_up < 100 and not any(frozen[caught_up:500])    # integrates again once the command is no longer held back
      # (at the fixed factor F0 the integrator then drives the command to the 409 cap after ~550 steps: frozen there, by the cap)
      assert c.inster_limit_observer.apply_torque_last > 0         # wheel right of a straight plan: steering left (+)
    # v0 (steering-angle controller off) is unchanged: controlsd's flag still freezes its integrator
    off, VM_off = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=STEP2_KEYS)
    for _ in range(50):
      off.update(True, _cs(30 / 3.6, angle=-6.0), VM_off, _vp(), True, 0.0, None, False, 0.2)
    assert off.pid.i == 0.0

  def test_non_finite_request(self):
    # the observer runs before controlsd's non-finite guard: a NaN / inf request must not raise and counts as the 0 the car then gets
    obs = InsterSteerLimitObserver.for_car(_car_params(HYUNDAI.HYUNDAI_INSTER))
    for _ in range(10):
      obs.update(100.0 / INSTER_STEER_FULL_SCALE, 0.0)
    last = obs.apply_torque_last
    for bad in (float('nan'), float('inf'), -float('inf')):
      obs.update(bad, 0.0)
      assert obs.apply_torque_last == last - 5 and obs.direction == -1     # towards 0 at the 5 raw / step release ramp
      last = obs.apply_torque_last
    obs.update(0.0, float('nan'))                                          # a non-finite driver torque reading: treated as 0
    assert obs.apply_torque_last == last - 5
    # and through update(): a non-finite desired curvature reaches the controller output without raising
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, ctrl=1)
    c.update(True, _cs(20.0, 1.0), VM, _vp(), False, float('nan'), None, False, 0.2)


F0_STORED = np.float32(F0)   # torque parameters are float32 (cereal)


class TestInsterFixedFactor(OpenpilotTestCase):
  """30 Oct 2026: with any Inster switch on, v0's lateral-acceleration factor is F0 for the whole path (P / I / D conversion, limits,
  feed-forward, stiction): the output does not depend on the learned or the override factor; the learned offset still applies."""

  def _drive(self, c, VM, factor, offset=0.04, override=None, n=1500):
    if override is not None:
      def _override(torque_params):
        torque_params.latAccelFactor = override
        torque_params.friction = 0.09
        return True
      c.extension.update_override_torque_params = _override
    out = []
    angle = 0.0
    rng = random.Random(7)
    for k in range(n):
      t = k * DT_CTRL
      v = max(0.0, 12.0 + 13.0 * math.sin(0.11 * t))
      dc = 0.02 * math.sin(0.7 * t) + 0.01 * math.sin(2.9 * t)
      target = math.degrees(VM.get_steer_from_curvature(-dc, v, 0.0))
      rate = (target - angle) * 3.0 + rng.gauss(0, 5.0)
      angle += rate * DT_CTRL
      if k % 100 == 0:
        c.update_torque_parameters(factor(t), offset, 0.09)
        c.extension.update_limits()
      tq, _, lac_log = c.update(not (900 <= k < 960), _cs(v, angle, pressed=(k % 700) < 25, rate=abs(rate)), VM,
                                _vp(0.02 * math.sin(0.05 * t), 0.3), k % 41 == 0, dc, None, False, 0.2)
      out.append((tq, lac_log.p, lac_log.i, lac_log.f, lac_log.output, lac_log.saturated))
    return out

  def test_output_invariant_to_learned_factor(self):
    schedules = (lambda t: 2.6, lambda t: 3.7, lambda t: 2.8 + 0.6 * math.sin(0.02 * t))
    for on, ctrl in ((("InsterStictionFriction",), None), (("InsterSpeedFeedforward",), None), (STEP2_KEYS, None),
                     ((ANGLE_PI_KEY,), None), (INSTER_KEYS, None), (INSTER_KEYS, 1), (INSTER_KEYS + (O5_KEY,), 1)):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on, ctrl=ctrl)
      assert c.inster_fixed_factor and c.torque_params.latAccelFactor == F0_STORED
      ref = self._drive(c, VM, lambda t: F0)
      for factor in schedules:
        c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on, ctrl=ctrl)
        assert self._drive(c, VM, factor) == ref, on
        assert c.torque_params.latAccelFactor == F0_STORED and c.torque_params.latAccelOffset == np.float32(0.04)
        assert abs(c.pid.pos_limit / F0 - CAP) < 1e-6                    # limits at 409 / 384 through F0
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on, ctrl=ctrl)
      assert self._drive(c, VM, lambda t: F0, override=3.7) == ref, on   # the override factor is ignored too
      assert c.torque_params.latAccelFactor == F0_STORED and c.torque_params.friction == np.float32(0.09)

  def test_learned_offset_kept(self):
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, ctrl=1)
    ref = self._drive(c, VM, lambda t: F0, n=400)
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, ctrl=1)
    assert self._drive(c, VM, lambda t: F0, offset=0.3, n=400) != ref
    assert c.torque_params.latAccelOffset == np.float32(0.3)

  def test_learner_factor_elsewhere(self):
    # other cars, the Inster with every switch off or with NNLC: v0 keeps the learned factor
    for car_name, on in ((HYUNDAI.HYUNDAI_KONA_EV, INSTER_KEYS), (HYUNDAI.HYUNDAI_INSTER, ()),
                         (HYUNDAI.HYUNDAI_INSTER, INSTER_KEYS + ("NeuralNetworkLateralControl",))):
      outs = []
      for f in (2.6, 3.7):
        c, VM = _make_v0(car_name, on=on)
        assert not c.inster_fixed_factor
        outs.append(self._drive(c, VM, lambda t, f=f: f, n=400))
        assert c.torque_params.latAccelFactor == np.float32(f)
      assert outs[0] != outs[1], (car_name, on)


# ---------------------------------------------------------------------------------------------------------------------------------
MPC_KEY = "InsterMPC"


class _MVM:
  """modelV2 stand-in for the MPC: frameId, orientationRate.z / velocity.x on T_IDXS for a constant plan curvature k, laneChangeState"""
  def __init__(self, frame_id, k, v, lane_change=0):
    self.frameId = frame_id
    self.orientationRate = type('X', (), {})()
    self.orientationRate.z = [k * v] * len(ModelConstants.T_IDXS)
    self.velocity = type('X', (), {})()
    self.velocity.x = [v] * len(ModelConstants.T_IDXS)
    self.meta = type('X', (), {})()
    self.meta.laneChangeState = lane_change
    # for the jerk / NNLC extension's update_model_v2 (unused by the v0 path)
    self.acceleration = type('X', (), {})()
    self.acceleration.y = [0.0] * 33
    self.orientation = type('X', (), {})()
    self.orientation.x = [0.0] * 33
    self.orientation.y = [0.0] * 33


def _mpc(params=None):
  return InsterMPC(DT_CTRL, params, model_t_idxs=ModelConstants.T_IDXS)


def _capnp_model(frame_id, k_of_t, v, lane_change=log.LaneChangeState.off):
  """a REAL modelV2 as SubMaster hands it to controlsd (messaging round trip -> capnp reader): plan curvature k_of_t(T) on T_IDXS"""
  ev = messaging.new_message('modelV2')
  m = ev.modelV2
  m.frameId = frame_id
  m.orientationRate.z = [k_of_t(t) * v for t in ModelConstants.T_IDXS]
  m.velocity.x = [v] * len(ModelConstants.T_IDXS)
  m.meta.laneChangeState = lane_change
  return messaging.log_from_bytes(ev.to_bytes()).modelV2


class TestInsterMPC(OpenpilotTestCase):
  """model predictive control (InsterMPC) on top of the PID: gain rows, nominal response, switch rules, limits, fallbacks, preview"""

  def test_gain_rows_steady_state(self):
    rows = mpc_gain_rows(MPC_PARAMS)
    for iv, v in enumerate(range(1, 46)):
      for k in (0.0, 0.001, -0.004, 0.02):
        u0 = rows[iv, 0] * k + rows[iv, 1] * v * v * k + rows[iv, 2:].sum() * k
        assert abs(u0 - v * v * k) <= 1e-9 * max(1.0, v * v * abs(k)), (v, k, u0)

  def test_nominal_step_response(self):
    # plan step of 1 m/s^2 at 0.6 s ahead, re-solved every block on the nominal model: the command leads, the nominal curvature follows
    # the plan on time with a small overshoot (as the kit's step_response)
    p = MPC_PARAMS
    rows = mpc_gain_rows(p)
    for v in (8.0, 15.0, 25.0, 33.0):
      K = np.array([np.interp(v, np.arange(1.0, 46.0), rows[:, i]) for i in range(rows.shape[1])])
      al = math.exp(-0.05 / mpc_tau(p, v))
      kn = ap = 0.0
      cmd, nom = [], []
      for b in range(60):
        t = b * 0.05
        P = np.array([(1.0 / v ** 2 if t + j * 0.05 >= 0.6 else 0.0) for j in range(25)])
        u = float(K[0] * kn + K[1] * ap + K[2:] @ P)
        cmd.append(u)
        nom.append(kn * v * v)
        kn = al * kn + (1 - al) * ap / v ** 2
        ap = u
      t50_cmd = 0.05 * next(i for i, x in enumerate(cmd) if x >= 0.5)
      t50_nom = 0.05 * next(i for i, x in enumerate(nom) if x >= 0.5)
      assert 0.35 <= t50_cmd <= 0.45 and 0.55 <= t50_nom <= 0.70, (v, t50_cmd, t50_nom)
      assert max(nom) < 1.10 and abs(nom[-1] - 1.0) < 1e-3

  def test_switch_rules(self):
    # only with the steering-angle controller, the PID and the speed feed-forward (its table slope converts the shift's ramp); off,
    # with the PI or without the feed-forward: the run equals the one without the param
    for on in ((ANGLE_PI_KEY, "InsterSpeedFeedforward", MPC_KEY), INSTER_KEYS + (MPC_KEY,)):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on, ctrl=1)
      assert c.inster_mpc is not None
      assert c.inster_angle_pi_version == ANGLE_PI_PID_BASE_VERSION + 1000 * (16 | 64 | ANGLE_PI_VERSION_MPC_BIT | 256)
    for car_name, on, ctrl in ((HYUNDAI.HYUNDAI_INSTER, INSTER_KEYS, 0), (HYUNDAI.HYUNDAI_INSTER, STEP2_KEYS, 1),
                               (HYUNDAI.HYUNDAI_INSTER, (ANGLE_PI_KEY,), 1),
                               (HYUNDAI.HYUNDAI_INSTER, ("InsterStictionFriction", ANGLE_PI_KEY), 1),
                               (HYUNDAI.HYUNDAI_INSTER, INSTER_KEYS + ("NeuralNetworkLateralControl",), 1),
                               (HYUNDAI.HYUNDAI_KONA_EV, INSTER_KEYS, 1)):
      c, VM = _make_v0(car_name, on=on + (MPC_KEY,), ctrl=ctrl)
      assert c.inster_mpc is None
      assert _run(c, VM) == _run(*_make_v0(car_name, on=on, ctrl=ctrl)), (car_name, on, ctrl)
    assert angle_pi_log_version(LATERAL_PID, True, True) == ANGLE_PI_PID_BASE_VERSION + 1000 * (16 | 8 | 64 | 128 | 256)
    assert MPC_PARAM == MPC_KEY
    # on: runs, changes the output, stays finite and within the output limit
    on, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS + (MPC_KEY,), ctrl=1)
    run_on = _run(on, VM, extension_model=lambda ay: _MVM(int(ay * 1000), 0.0, 15.0))
    run_off = _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, ctrl=1), extension_model=lambda ay: _MVM(int(ay * 1000), 0.0, 15.0))
    assert run_on != run_off
    assert all(math.isfinite(r[0]) and abs(r[0]) <= CAP + 1e-9 for r in run_on)

  def _drive(self, m, n=600, v=lambda k: 15.0, kd=lambda k: 0.01 * math.sin(0.01 * k), pressed=lambda k: False, lc=lambda k: 0,
             clock=None, model=True):
    out = []
    for k in range(n):
      vv, kk = v(k), kd(k)
      if model:
        m.observe_model(_MVM(k // 5, kk, vv, lc(k)))
      else:
        m.observe_model(None)
      la0 = kk * vv * vv
      kref, delta = m.update(vv, kk, 0.9 * kk, 0.2, pressed(k), la0, ff_table_slope_raw(la0, vv), clock=clock)
      out.append((kref, delta, m.w, m.h, m.u0))
    return out

  def test_limits(self):
    m = _mpc()
    out = self._drive(m, n=2000, kd=lambda k: 0.02 * math.sin(0.004 * k) + 0.01 * (k % 400 > 200))
    d = np.array([o[1] for o in out])
    assert np.all(np.abs(np.diff(d)) <= MPC_JERK_MAX * DT_CTRL + 1e-12)                    # jerk limit of the feed-forward part
    assert all(abs(o[4]) <= max(MPC_A_LIM, abs(0.03) * 15.0 ** 2) + 1e-12 for o in out)   # |u0| clip
    assert all(math.isfinite(x) for o in out for x in o)
    assert max(abs(x) for x in d) > 0.0 and max(o[2] for o in out) > 0.4                   # it acts

  def test_fallbacks(self):
    # steeringPressed / lane change (state >= 2): weight h < 0.05 within 0.45 s; below 8 km/h: w = 0
    for kw in ({'pressed': lambda k: k >= 300}, {'lc': lambda k: 2 if k >= 300 else 0}):
      out = self._drive(_mpc(), **kw)
      assert out[299][3] > 0.9 and all(o[3] < 0.05 for o in out[345:])
    out = self._drive(_mpc(), v=lambda k: 7.0 / 3.6)
    assert all(o[2] == 0.0 for o in out)
    # three solves in a row over the budget: falls back; back within it: returns
    t = [0.0]
    def slow_clock():
      t[0] += 0.0025                     # each solve reads the clock twice: 2.5 ms per solve (> the 2 ms budget)
      return t[0]
    out = self._drive(_mpc(), clock=slow_clock)
    assert out[50][3] < 0.2 and all(o[3] < 0.05 for o in out[100:])
    # non-finite plan: no exception, finite outputs
    m = _mpc()
    out = self._drive(m, n=200, kd=lambda k: float('nan') if k == 100 else 0.005)
    assert all(math.isfinite(o[1]) for o in out)

  def test_preview_and_frames(self):
    m = _mpc()
    for k in range(12):
      m.observe_model(_MVM(k // 5, 0.004, 20.0))
    assert m.age == 1 and m.frame_id == 2
    m.observe_model(None)                                     # a step without a usable frame ages it
    assert m.age == 2
    m.update(20.0, 0.004, 0.004, 0.2, False, 1.6, 100.0)     # fresh start
    P = m.preview(0.004, 20.0, 0.2)
    assert np.allclose(P, 0.004)                              # anchored frame of the same curvature: the plan everywhere
    m.observe_model(_MVM(99, 0.010, 20.0))                    # frame 0.01 vs plan 0.004: |0.006| x 400 = 2.4 m/s^2 > 0.15: not anchored
    P = m.preview(0.004, 20.0, 0.2)
    assert np.allclose(P, 0.004)                              # held plan
    m.observe_model(_MVM(100, 0.0041, 20.0))                  # anchored (0.04 m/s^2): shifted onto the plan
    assert np.allclose(m.preview(0.004, 20.0, 0.2), 0.004)
    m.observe_model(_MVM(101, 0.004, 20.0, lane_change=1))    # lane change state != off: held plan
    assert m.lane_change == 1 and np.allclose(m.preview(0.004, 20.0, 0.2), 0.004)
    for _ in range(20):
      m.observe_model(_MVM(101, 0.004, 20.0))
    assert m.age > 15                                         # too old: held

  def test_reset_restarts_from_the_wheel(self):
    m = _mpc()
    self._drive(m, n=300)
    m.reset()
    kref, delta = m.update(15.0, 0.01, 0.002, 0.2, False, 2.25, 100.0)
    # first step after the reset: the nominal state restarts at the measured wheel (0.002) with h one step up from 0, so the reference
    # is the plan moved slightly towards the wheel and the feed-forward shift starts from 0 within one jerk-limited step
    assert m.h < 0.07 and m.w < 0.05 and 0.002 <= kref <= 0.01 and abs(kref - 0.01) <= m.w * 0.008 + 1e-12
    assert abs(delta) <= MPC_JERK_MAX * DT_CTRL + 1e-12 and abs(m.k_nom - 0.002) < 0.001

  def test_real_capnp_model(self):
    # car review 1, change 1: controlsd hands the extension a capnp READER (laneChangeState is a capnp enum, int() of it raises).
    # observe_model on readers, with and without a lane change
    m = _mpc()
    for lcs in (log.LaneChangeState.off, log.LaneChangeState.preLaneChange, log.LaneChangeState.laneChangeStarting,
                log.LaneChangeState.laneChangeFinishing):
      rd = _capnp_model(int(lcs) + 10, lambda t: 0.004, 20.0, lcs)
      assert not isinstance(rd.meta.laneChangeState, int)
      m.observe_model(rd)
      assert m.lane_change == int(lcs) and m.age == 0 and m.row is not None
    m.observe_model(messaging.new_message('modelV2').modelV2)    # never-received default message: empty lists -> ages, plan held
    assert m.age == 1
    # end to end, as controlsd: extension.update_model_v2(reader) -> LatControlTorque.update, active and inactive steps, a lane change
    c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS + (MPC_KEY,), ctrl=1)
    assert c.inster_mpc is not None
    n = [0]

    def model(ay):
      n[0] += 1
      k = n[0]
      lcs = log.LaneChangeState.laneChangeStarting if 200 <= k < 260 else (log.LaneChangeState.preLaneChange if 180 <= k < 200
                                                                            else log.LaneChangeState.off)
      return _capnp_model(k // 5, lambda t, a=ay: a / 16.0 ** 2, 16.0, lcs)
    out = _run(c, VM, extension_model=model)
    assert len(out) == 1200 and all(math.isfinite(r[0]) and abs(r[0]) <= CAP + 1e-9 for r in out)
    assert c.inster_mpc.n_solve > 100

  def test_gain_set(self):
    # car review 1, change 2: the MPC variant's own PID gains and D filter, selected only when the MPC is in use. The placeholders equal
    # PID-30Oct, so distinct values are patched in to see which set is taken.
    assert (ANGLE_PI_MPC_KP, ANGLE_PI_MPC_KI, ANGLE_PI_MPC_KD, ANGLE_PI_MPC_D_TAU) == (ANGLE_PI_PID_KP, ANGLE_PI_PID_KI, ANGLE_PI_PID_KD,
                                                                                      ANGLE_PI_D_TAU)   # placeholders until the joint fit
    mkp, mki, mkd, mtau = (3.0, 0.4, -0.01), (0.8, 0.3, -0.008), (0.3, 0.001, 0.0), 0.05
    with mock.patch.multiple(latcontrol_torque_v0, ANGLE_PI_MPC_KP=mkp, ANGLE_PI_MPC_KI=mki, ANGLE_PI_MPC_KD=mkd, ANGLE_PI_MPC_D_TAU=mtau):
      c, _ = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS + (MPC_KEY,), ctrl=1)
      assert c.inster_mpc is not None
      assert (c.pid.kp_coeffs, c.pid.ki_coeffs, c.pid.kd_coeffs) == (mkp, mki, mkd)
      assert abs(c.inster_wheel_rate.a - (1.0 - math.exp(-DT_CTRL / mtau))) < 1e-15
      # MPC param on but not in use (PI; no feed-forward), or off: PI-30Oct / PID-30Oct and their filter unchanged
      for on, ctrl, gains in ((INSTER_KEYS + (MPC_KEY,), 0, (ANGLE_PI_PI_KP, ANGLE_PI_PI_KI, None)),
                              ((ANGLE_PI_KEY, MPC_KEY), 1, (ANGLE_PI_PID_KP, ANGLE_PI_PID_KI, ANGLE_PI_PID_KD)),
                              (INSTER_KEYS, 1, (ANGLE_PI_PID_KP, ANGLE_PI_PID_KI, ANGLE_PI_PID_KD))):
        c, _ = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on, ctrl=ctrl)
        assert c.inster_mpc is None
        assert (c.pid.kp_coeffs, c.pid.ki_coeffs, c.pid.kd_coeffs) == gains, (on, ctrl)
        if ctrl == 1:
          assert abs(c.inster_wheel_rate.a - (1.0 - math.exp(-DT_CTRL / ANGLE_PI_D_TAU))) < 1e-15

  def test_setup_failure_falls_back_to_the_pid(self):
    # car review 1, change 3: any exception while building the MPC -> plain PID-30Oct, version without bit 128, output as without the param
    with mock.patch.object(latcontrol_torque_v0, 'InsterMPC', side_effect=RuntimeError('gain rows')):
      c, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS + (MPC_KEY,), ctrl=1)
    assert c.inster_mpc is None
    assert c.inster_angle_pi_version == ANGLE_PI_PID_BASE_VERSION + 1000 * (16 | 64 | 256)
    assert (c.pid.kp_coeffs, c.pid.ki_coeffs, c.pid.kd_coeffs) == (ANGLE_PI_PID_KP, ANGLE_PI_PID_KI, ANGLE_PI_PID_KD)
    assert _run(c, VM) == _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS, ctrl=1))

  def test_non_finite_latch(self):
    # car review 1, change 4: a persistently non-finite solve holds the fallback (h -> 0) until a finite solve; it then returns
    m = _mpc()
    self._drive(m, n=300)
    assert m.h > 0.9
    good = m.rows
    m.rows = np.full_like(good, np.nan)
    out = self._drive(m, n=100)
    assert all(o[3] < 0.05 for o in out[45:]) and all(math.isfinite(o[0]) and math.isfinite(o[1]) for o in out)
    m.rows = good
    out = self._drive(m, n=200)
    assert out[-1][3] > 0.9
    # a non-finite nominal curvature is resynchronised to the wheel on the same step
    m.k_nom = float('nan')
    kref, delta = m.update(15.0, 0.01, 0.009, 0.2, False, 2.25, 100.0)
    assert m.k_nom == 0.009 and math.isfinite(kref) and math.isfinite(delta)

  def test_preview_linear_profile(self):
    # car review 1, change 6: a plan curvature linear in T checks the frame time offset (0.075 s + age steps) beyond lat_delay numerically
    k0, s, v, ld, age = 0.002, 0.002, 20.0, 0.2, 3
    m = _mpc()
    m.observe_model(_capnp_model(1, lambda t: k0 + s * t, v))
    for _ in range(age):
      m.observe_model(_capnp_model(1, lambda t: k0 + s * t, v))
    assert m.age == age
    a0 = 0.075 + DT_CTRL * age
    kd = k0 + s * (ld + a0)                                   # the plan now = the frame at lat_delay: anchored exactly, no shift
    m.update(v, kd, kd, ld, False, kd * v * v, 100.0)
    P = m.preview(kd, v, ld)
    taus = np.arange(25) * 0.05
    fut = taus > ld
    # tolerance: modelV2 stores Float32 (~1e-10 here); one step of age or a wrong anchor shift would be >= s x 0.01 s = 2e-5
    assert np.allclose(P[fut], k0 + s * (taus[fut] + a0), rtol=0, atol=1e-9)
    assert np.allclose(P[~fut], kd, rtol=0, atol=1e-12)         # own buffer (constant plan so far)
    # offset plan (anchored within 0.15 m/s^2): the frame shifted by kd - k_now
    kd2 = kd + 0.1 / v ** 2
    P2 = m.preview(kd2, v, ld)
    assert np.allclose(P2[fut], k0 + s * (taus[fut] + a0) + (kd2 - kd), rtol=0, atol=1e-9)
