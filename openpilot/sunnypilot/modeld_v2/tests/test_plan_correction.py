"""
Tests of the Blendful plan shift (plan_correction.py) and of its delivery in ModelState.get_action_from_model (modes Off / Regular).
Design: inster-dev results/2026-11-12_a8_right_line_blendful/DESIGN.md rev 5.1, section C and E3 (implementation round 2).
"""
import importlib.abc
import importlib.machinery
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan, smooth_value, get_accel_from_plan
from openpilot.sunnypilot.modeld_v2.blendful import Blendful
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.plan_correction import (T_IDXS, X_IDXS, approach_length, basis, fit_amplitude, instant_gain, profile, profile_length,
                                                            shift_plan)

LAT_T = 0.275


def make_plan(v=15.0, k=0.0):
  plan = np.zeros((33, 15))
  plan[:, 0] = X_IDXS
  plan[:, 1] = k * X_IDXS ** 2 / 2
  plan[:, 3] = v
  plan[:, Plan.T_FROM_CURRENT_EULER.start + 2] = v * k * T_IDXS / 2  # psi = v k t / 2 for a constant-curvature path (small angle)
  plan[:, Plan.ORIENTATION_RATE.start + 2] = v * k
  return plan


def curvature(plan, v, t_a=LAT_T):
  return get_curvature_from_plan(plan[:, Plan.T_FROM_CURRENT_EULER][:, 2], plan[:, Plan.ORIENTATION_RATE][:, 2], T_IDXS, v, t_a)


class TestPlanShift:
  @pytest.mark.parametrize("v", [5.0, 15.0, 30.0])
  @pytest.mark.parametrize("t_a", [0.275, 0.35])
  def test_curvature_gain_within_one_percent(self, v, t_a):
    k0 = 0.3 / v ** 2
    shifted = shift_plan(make_plan(v), k0, v)
    got = curvature(shifted, v, t_a) - curvature(make_plan(v), v, t_a)
    _, _, dk = profile(k0, v)
    want = np.interp(max(t_a, 0.3), T_IDXS, dk)
    assert abs(got - want) <= 0.01 * abs(k0)
    assert abs(got - k0 * instant_gain(v, t_a)) < 1e-12  # F is the same quantity, through get_curvature_from_plan

  def test_mpc_row_gains_exactly_the_curvature_change(self):
    v = 15.0
    k0 = 0.01
    base, shifted = make_plan(v, 0.002), shift_plan(make_plan(v, 0.002), k0, v)
    _, _, dk = profile(k0, v)
    row = lambda p: p[:, Plan.ORIENTATION_RATE][:, 2] / p[:, Plan.VELOCITY][:, 0]  # noqa: E731
    np.testing.assert_allclose(row(shifted) - row(base), dk, atol=1e-12)
    ay = shifted[:, Plan.ACCELERATION][:, 1] - base[:, Plan.ACCELERATION][:, 1]
    np.testing.assert_allclose(ay, v ** 2 * dk, atol=1e-12)

  @pytest.mark.parametrize("age", range(16))
  def test_anchor_residual_over_frame_ages(self, age):
    # InsterMPC anchors kd (profile at 0.3 s) against the curvature at 0.275 + 0.01 age: the residual of the correction stays small
    for v in (8.0, 15.0, 25.0):
      _, _, dk = profile(1.0, v)
      resid = abs(np.interp(0.3, T_IDXS, dk) - np.interp(0.275 + 0.01 * age, T_IDXS, dk))
      assert resid <= (0.019 if age <= 4 else 0.096) + 1e-9

  def test_position_heading_curvature_are_consistent(self):
    v, k0 = 12.0, 0.004
    dy, dpsi, dk = profile(k0, v)
    # d(dy)/dx = dpsi and d(dpsi)/dx = dk on the profile's own grid (x = v t)
    x = v * T_IDXS
    inside = x < profile_length(v) * 0.95
    assert np.allclose(np.gradient(dy, x)[inside][2:], dpsi[inside][2:], rtol=3e-2, atol=2e-4)  # finite differences on the model's grid
    assert np.allclose(np.gradient(dpsi, x)[inside][2:-1], dk[inside][2:-1], rtol=3e-2, atol=2e-4 * abs(k0) * 10)

  def test_untouched_fields_and_no_nan_at_zero_speed(self):
    plan = make_plan(0.0)
    for v in (0.0, 0.1, 1.0):
      out = shift_plan(plan, 0.01, max(v, 1.0))
      assert np.all(np.isfinite(out))
      np.testing.assert_array_equal(out[:, Plan.POSITION][:, 0], plan[:, Plan.POSITION][:, 0])
      np.testing.assert_array_equal(out[:, Plan.VELOCITY], plan[:, Plan.VELOCITY])
    assert np.isfinite(instant_gain(1.0, LAT_T))
    plan = make_plan(15.0)
    out = shift_plan(plan, 0.01, 15.0)
    np.testing.assert_array_equal(out[:, Plan.VELOCITY], plan[:, Plan.VELOCITY])
    np.testing.assert_array_equal(out[:, 0], plan[:, 0])
    assert out is not plan and np.all(plan[:, 1] == 0.0)  # a copy

  def test_zero_amplitude_is_the_plan(self):
    plan = make_plan(15.0, 0.001)
    np.testing.assert_array_equal(shift_plan(plan, 0.0, 15.0), plan)

  def test_planplus_order_is_linear(self):
    plan, planplus = make_plan(15.0, 0.001), make_plan(15.0, 0.0002)
    planplus[:, Plan.VELOCITY] = 0.0  # planplus is a correction of the plan: no velocity of its own
    a = shift_plan(plan, 0.005, 15.0) + 0.5 * planplus
    b = shift_plan(plan + 0.5 * planplus, 0.005, 15.0)
    np.testing.assert_allclose(a, b, atol=1e-12)

  def test_fit_recovers_the_profile(self):
    v, k0 = 15.0, 0.003
    la, lc = approach_length(v), profile_length(v)
    win = (X_IDXS >= 0.5 * la) & (X_IDXS <= 1.5 * la)
    err = k0 * lc ** 2 / 6 * basis(X_IDXS[win] / lc)  # the profile's own shape as a function of distance
    d, k = fit_amplitude(X_IDXS[win], err, 1.0, lc)
    assert abs(k - k0) < 1e-9 and abs(d - k0 * lc ** 2 / 6) < 1e-9

  def test_gain_value_at_two_second_approach(self):
    assert abs(instant_gain(15.0, LAT_T) - 0.821) < 0.005  # DESIGN check 13: F = 0.821 at L_a = 2.0 s, above the 10 m floor


# ---------------------------------------------------------------------------------------------------------------------------------
# C11 (rev 5): F per frame with the plan's own v_x(0); a pure numpy path (no modeld, no msgq), so it is never skipped
def vx_plan(v, scale, wobble=0.0):
  """The plan's own speed profile: v_x(0) = scale x vEgo, optionally varying along t."""
  return v * scale * (1.0 + wobble * np.sin(T_IDXS / 2.0))


def plan_with_speed(plan, v, scale, wobble):
  out = np.array(plan, copy=True)
  out[:, Plan.VELOCITY.start] = vx_plan(v, scale, wobble)
  out[:, Plan.ORIENTATION_RATE.start + 2] = out[:, Plan.VELOCITY.start] * plan[:, Plan.ORIENTATION_RATE.start + 2] / v
  return out


class TestFramewiseGain:
  @pytest.mark.parametrize("scale", [1.0, 1.08, 0.967])
  def test_f_formula(self, scale):
    v = 15.0
    vx = vx_plan(v, scale)
    _, dpsi, dk = profile(1.0, v)
    psi_ta = (LAT_T / 0.3) * np.interp(0.3, T_IDXS, dpsi)  # t_a < 0.3 s: get_curvature_from_plan's own branch
    want = 2.0 * psi_ta / (v * LAT_T) - vx[0] * dk[0] / v
    assert abs(instant_gain(v, LAT_T, vx) - want) < 1e-12
    assert abs(instant_gain(v, LAT_T) - instant_gain(v, LAT_T, np.full(33, v))) < 1e-15  # without a plan: v_x = vEgo

  def test_vego_scaling_would_be_wrong_by_the_speed_ratio(self):
    v = 15.0
    assert abs(instant_gain(v, LAT_T, vx_plan(v, 1.08)) / instant_gain(v, LAT_T) - 1.0) > 0.03  # the first implementation's 0.954 x error

  @pytest.mark.parametrize("scale", [1.08, 0.967, 1.0])
  @pytest.mark.parametrize("wobble", [0.0, 0.04])
  @pytest.mark.parametrize("v", [8.0, 15.0, 25.0])
  def test_regular_delivery_equals_upstream_and_mpc_row_gains_exactly_dk(self, scale, wobble, v):
    from openpilot.sunnypilot.modeld_v2.blendful_right import MODE_REGULAR, BlendfulCorrection
    from openpilot.sunnypilot.modeld_v2.tests.test_blendful_right import Scene
    ctl, ref = BlendfulCorrection(), Blendful()
    worst_gain = worst_row = 0.0
    nonzero = 0
    for i in range(200):
      sc = Scene(v=v, center=0.5 if i < 120 else -0.3, k=0.0 if i < 60 else 1 / 150, edges=False)
      mo = sc.model_output()
      mo['plan'][0] = plan_with_speed(mo['plan'][0], v, scale, wobble)
      plan_before = mo['plan'][0].copy()
      k_m = curvature(plan_before, v)  # the model's own curvature, as get_action_from_model passes it on
      k0 = ctl.update(mo, v, True, k_m, LAT_T, MODE_REGULAR)
      up = ref.update(sc.model_output(), v, True, k_m, enabled=True)
      shifted = shift_plan(plan_before, k0, v)
      gain = curvature(shifted, v) - curvature(plan_before, v)  # what desiredCurvature gains from the shifted plan
      worst_gain = max(worst_gain, abs(gain - up))
      nonzero += abs(up) > 1e-6
      row = lambda p: p[:, Plan.ORIENTATION_RATE][:, 2] / p[:, Plan.VELOCITY][:, 0]  # noqa: E731
      _, _, dk = profile(k0, v)
      worst_row = max(worst_row, float(np.max(np.abs(row(shifted) - row(plan_before) - dk))))
    assert nonzero > 50  # the fixture really exercises a correction
    assert worst_gain <= 1e-9  # delivered = today's add-on (E1-12), whatever v_x(0) / vEgo
    assert worst_row <= 1e-12  # the MPC row gains exactly the curvature change


# ---------------------------------------------------------------------------------------------------------------------------------
# delivery inside ModelState.get_action_from_model (modeld.py needs tinygrad, which the sandbox does not have: stubbed)
class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
  def find_spec(self, name, path, target=None):
    if name == 'tinygrad' or name.startswith('tinygrad.'):
      return importlib.machinery.ModuleSpec(name, self, is_package=True)
    return None

  def create_module(self, spec):
    m = MagicMock()
    m.__name__, m.__path__, m.__spec__ = spec.name, [], spec
    return m

  def exec_module(self, module):
    pass


def load_modeld():
  try:
    import openpilot.sunnypilot.modeld_v2.modeld as m
    return m
  except ImportError:
    pass
  sys.meta_path.insert(0, _Stub())
  try:
    import msgq.visionipc as v
    for n in ('VisionBuf', 'VisionIpcClient'):
      if not hasattr(v, n):
        setattr(v, n, MagicMock())
    import openpilot.sunnypilot.modeld_v2.modeld as m
    return m
  except ImportError as e:
    pytest.skip(f"modeld.py cannot be imported here: {e}")


def legacy_action(state, model_output, prev_action, lat_action_t, long_action_t, v_ego, blendful):
  """Verbatim copy of ModelState.get_action_from_model at 910ca62 (mode Off must equal it bit for bit)."""
  from openpilot.sunnypilot.modeld_v2.fill_model_msg import get_curvature_from_output
  plan = model_output['plan'][0]
  desired_accel = get_accel_from_plan(plan[:, Plan.VELOCITY][:, 0], plan[:, Plan.ACCELERATION][:, 0], state.constants.T_IDXS, action_t=long_action_t)
  curvature_plan = (plan + (state.PLANPLUS_CONTROL - 1.0) * model_output['planplus'][0]
                    if 'planplus' in model_output and state.PLANPLUS_CONTROL != 1.0 else plan)
  desired_curvature = get_curvature_from_output(model_output, curvature_plan, v_ego, lat_action_t, state.mlsim)
  if blendful is not None:
    desired_curvature += blendful(desired_curvature)
  stop = v_ego < 0.3 and desired_accel < 0.1
  desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, state.LONG_SMOOTH_SECONDS)
  if state.generation is not None and state.generation >= 10:
    if v_ego > state.MIN_LAT_CONTROL_SPEED:
      desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, state.LAT_SMOOTH_SECONDS)
    else:
      desired_curvature = prev_action.desiredCurvature
  return float(desired_curvature), float(desired_accel), bool(stop)


def fake_state(generation=None, planplus_control=1.0, mlsim=False):
  return types.SimpleNamespace(constants=ModelConstants, PLANPLUS_CONTROL=planplus_control, mlsim=mlsim, generation=generation,
                               MIN_LAT_CONTROL_SPEED=0.3, LAT_SMOOTH_SECONDS=0.1, LONG_SMOOTH_SECONDS=0.3)


def model_frames(n=60, v=15.0, with_planplus=True, seed=1):
  rng = np.random.default_rng(seed)
  for i in range(n):
    plan = make_plan(v, 0.001 * np.sin(i / 7.0))[None]
    plan[0, :, 1] += 0.2 * rng.standard_normal(33) * 0.01
    out = {'plan': plan}
    if with_planplus:
      out['planplus'] = 0.1 * make_plan(v, 0.0005)[None]
      out['planplus'][0, :, Plan.VELOCITY] = 0.0
    yield out


class TestDelivery:
  @pytest.fixture(autouse=True)
  def _module(self):
    self.m = load_modeld()
    self.ModelState = self.m.ModelState

  def _prev(self):
    from openpilot.cereal import log
    return log.ModelDataV2.Action()

  @pytest.mark.parametrize("generation", [None, 10])
  @pytest.mark.parametrize("planplus_control", [1.0, 0.5])
  def test_off_is_identical_to_today(self, generation, planplus_control):
    state = fake_state(generation, planplus_control)
    bf_new, bf_old = Blendful(), Blendful()
    prev_new = prev_old = self._prev()
    for mo in model_frames():
      mo_new = {k: v.copy() for k, v in mo.items()}
      new = self.ModelState.get_action_from_model(state, mo_new, prev_new, LAT_T, 0.3, 15.0,
                                                  lambda c, mo=mo_new: bf_new.update(mo, 15.0, True, c, enabled=False))
      old = legacy_action(state, mo, prev_old, LAT_T, 0.3, 15.0,
                          lambda c, mo=mo: bf_old.update(mo, 15.0, True, c, enabled=False))
      from openpilot.cereal import log
      old_action = log.ModelDataV2.Action(desiredCurvature=old[0], desiredAcceleration=old[1], shouldStop=old[2])  # as modeld stores it (float32)
      assert (new.desiredCurvature, new.desiredAcceleration, new.shouldStop) == (old_action.desiredCurvature, old_action.desiredAcceleration,
                                                                                old_action.shouldStop)
      np.testing.assert_array_equal(mo_new['plan'], mo['plan'])  # the published plan is untouched
      prev_new, prev_old = new, old_action

  def test_regular_immediate_correction_equals_upstream(self):
    from openpilot.sunnypilot.modeld_v2.blendful_right import BlendfulCorrection, MODE_REGULAR
    from openpilot.sunnypilot.modeld_v2.tests.test_blendful_right import Scene
    state = fake_state()
    ctl, ref = BlendfulCorrection(), Blendful()
    worst = 0.0
    for i in range(200):
      sc = Scene(center=0.5 if i < 120 else -0.3, k=0.0 if i < 60 else 1 / 150, edges=False, v=15.0)
      mo = sc.model_output()
      plan_before = mo['plan'].copy()
      k_m = curvature(plan_before[0], 15.0)  # the model's own curvature, as get_action_from_model passes it on
      holder = {}

      def correction(kappa, mo=mo, holder=holder):
        k0 = ctl.update(mo, 15.0, True, kappa, LAT_T, MODE_REGULAR)
        holder['k0'] = k0
        return k0, ctl.gain
      act = self.ModelState.get_action_from_model(state, mo, self._prev(), LAT_T, 0.3, 15.0, correction=correction)
      up = ref.update(sc.model_output(), 15.0, True, k_m, enabled=True)
      base = curvature(plan_before[0], 15.0)
      # desired curvature from the shifted plan = model curvature + upstream's correction
      assert abs(act.desiredCurvature - (base + up)) <= 1e-9
      worst = max(worst, abs(act.desiredCurvature - (base + up)))
      # the published plan is the raw plan shifted by exactly that profile
      np.testing.assert_allclose(mo['plan'][0], shift_plan(plan_before[0], holder['k0'], 15.0), atol=1e-12)
    assert worst <= 1e-9

  def test_regular_with_planplus_shifts_plan_before_planplus(self):
    from openpilot.sunnypilot.modeld_v2.blendful_right import BlendfulCorrection, MODE_REGULAR
    from openpilot.sunnypilot.modeld_v2.tests.test_blendful_right import Scene
    state = fake_state(planplus_control=0.5)
    ctl = BlendfulCorrection()
    for _ in range(100):
      sc = Scene(center=0.5, v=15.0, edges=False)
      mo = sc.model_output()
      mo['planplus'] = 0.1 * make_plan(15.0, 0.0005)[None]
      mo['planplus'][0, :, Plan.VELOCITY] = 0.0
      raw = mo['plan'].copy()
      holder = {}

      def correction(kappa, mo=mo, holder=holder):
        holder['k0'] = ctl.update(mo, 15.0, True, kappa, LAT_T, MODE_REGULAR)
        return holder['k0'], ctl.gain
      act = self.ModelState.get_action_from_model(state, mo, self._prev(), LAT_T, 0.3, 15.0, correction=correction)
      expected_plan = shift_plan(raw[0], holder['k0'], 15.0)
      cp = expected_plan + (0.5 - 1.0) * mo['planplus'][0]
      assert abs(act.desiredCurvature - curvature(cp, 15.0)) < 1e-9  # Action stores float32
    assert abs(holder['k0']) > 0.0

  def test_action_head_models_add_the_instantaneous_gain(self):
    state = fake_state()
    mo = {'plan': make_plan(15.0)[None], 'action': np.array([[0.2, 0.1]])}
    raw = mo['plan'].copy()
    act = self.ModelState.get_action_from_model(state, mo, self._prev(), LAT_T, 0.3, 15.0, correction=lambda c: (0.001, 0.8))
    assert abs(act.desiredCurvature - (0.2 / 15.0 ** 2 + 0.0008)) < 1e-9
    np.testing.assert_allclose(mo['plan'][0], shift_plan(raw[0], 0.001, 15.0), atol=1e-12)
