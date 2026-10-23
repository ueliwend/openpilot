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
from openpilot.common.realtime import DT_CTRL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.sunnypilot.selfdrive.car import interfaces as sunnypilot_interfaces
from openpilot.sunnypilot.selfdrive.controls.lib import inster_lateral, latcontrol_torque_v0
from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import A_NODES, CMD_RAW, F0, SCALE_CLIP, STEER_MAX_RAW, V_NODES, \
                                                                       InsterStictionFriction, ff_lat_accel_equivalent, ff_torque, \
                                                                       table_torque
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorque as LatControlTorqueV0

INSTER_KEYS = ("InsterStictionFriction", "InsterSpeedFeedforward")
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
  for key in INSTER_KEYS + ("LateralJerkTorqueController", "NeuralNetworkLateralControl"):
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
    assert not controller.inster_friction and not controller.inster_ff
    assert not hasattr(controller, "inster_stiction")
    entered = AssertionError("Inster code path entered with both switches off")
    with mock.patch.object(latcontrol_torque_v0, "ff_lat_accel_equivalent", side_effect=entered), \
         mock.patch.object(LatControlTorqueV0, "_inster_friction_raw", side_effect=entered), \
         mock.patch.object(InsterStictionFriction, "update", side_effect=entered):
      _run(controller, VM, 300)

  def test_other_car_ignores_switches(self):
    off, VM_off = _make_v0(HYUNDAI.HYUNDAI_KONA_EV)
    on, VM_on = _make_v0(HYUNDAI.HYUNDAI_KONA_EV, on=INSTER_KEYS)
    assert not on.inster_friction and not on.inster_ff
    assert _run(off, VM_off) == _run(on, VM_on)

  def test_nnlc_on_ignores_switches(self):
    off, VM_off = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=("NeuralNetworkLateralControl",))
    on, VM_on = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=INSTER_KEYS + ("NeuralNetworkLateralControl",))
    assert not on.inster_friction and not on.inster_ff
    assert _run(off, VM_off) == _run(on, VM_on)

  def test_switches_change_the_output(self):
    base = _run(*_make_v0(HYUNDAI.HYUNDAI_INSTER))
    for on in (("InsterStictionFriction",), ("InsterSpeedFeedforward",), INSTER_KEYS):
      controller, VM = _make_v0(HYUNDAI.HYUNDAI_INSTER, on=on)
      assert controller.inster_friction == ("InsterStictionFriction" in on)
      assert controller.inster_ff == ("InsterSpeedFeedforward" in on)
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
    for on in ((), ("InsterStictionFriction",), ("InsterSpeedFeedforward",), INSTER_KEYS):
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
      for key in INSTER_KEYS + ("LateralJerkTorqueController", "NeuralNetworkLateralControl"):
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
    # the step-3 params are gone: unknown to Params (a stale file on a device is never read), only the two step-2 switches remain
    params = Params()
    keys = {k.decode() if isinstance(k, bytes) else k for k in params.all_keys()}
    for key in ("InsterExitTuning", "InsterLowSpeedP", "InsterHighTorqueLimits", "InsterTorqueCap"):
      assert key not in keys
    assert inster_lateral.INSTER_SWITCHES == INSTER_KEYS
    assert {"InsterStictionFriction", "InsterSpeedFeedforward"} <= keys

