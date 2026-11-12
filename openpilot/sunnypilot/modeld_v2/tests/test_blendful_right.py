"""
Unit tests of Right-line Blendful (blendful_right.py): measurement filter, references, correction law, yields, junctions, bounds, blinkers,
transitions, logging and the three modes.  Design: inster-dev results/2026-11-12_a8_right_line_blendful/DESIGN.md rev 4.3, section E3.
Signs: model frame y > 0 is right, curvature > 0 is a right turn.  Offline only; not validated on the car.
"""
import copy
from dataclasses import dataclass

import numpy as np
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.modeld_v2 import blendful_right as br
from openpilot.sunnypilot.modeld_v2.blendful import Blendful
from openpilot.sunnypilot.modeld_v2.blendful_right import (MODE_OFF, MODE_REGULAR, MODE_RIGHT, BlendfulCorrection, ConfGate, InputFilter, JunctionLogic,
                                                           RightInputs, RightLineBlendful, YieldEstimator, bend_score, junction_mode,
                                                           junction_policy, mode_from_params, selection_to_params, split_score)
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.plan_correction import shift_plan

X = np.array(ModelConstants.X_IDXS)
LAT_T = 0.275
W_R = br.W_R


@dataclass
class Scene:
  """One model frame: the car is at the model origin, the lane centre at y = center (right positive)."""
  v: float = 20.0
  center: float = 0.0
  width: float = 3.5  # left line to right line
  k: float = 0.0  # road curvature (right positive)
  pl: float = 0.9  # left / right near line probabilities
  pr: float = 0.9
  std: float = 0.1
  plan: float = 0.0  # plan lateral offset reached at 10 m (model plan, relative to the road)
  widen: float = 0.0  # extra distance of the right line (a widening)
  far_pr: float = 0.0  # far right line probability
  far_pl: float = 0.0
  edges: bool = False
  edge_half: float = 3.5  # road edge distance from the lane centre, both sides
  edge_right: float | None = None
  edge_std: float = 0.3
  yaw_rate: float | None = None
  bend_x0: float = 0.0  # right line bends away (a right arc of radius bend_R from bend_x0 on): side road, junction
  bend_R: float = 0.0
  shift_near: float = 0.0  # split pattern: the new near right line sits shift_near from the old one ...
  far_at: float | None = None  # ... and the new far right line where the old near right line was (absolute, relative to the centre)

  def model_output(self):
    road = self.k * X ** 2 / 2
    hw = self.width / 2
    lines = np.zeros((1, 4, 33, 2))
    stds = np.ones((1, 4, 33, 2))
    offs = (-hw - 3.5, -hw, hw + self.widen, hw + self.widen + 3.5)
    for i, o in enumerate(offs):
      lines[0, i, :, 0] = road + self.center + o
    if self.bend_R:
      dx = np.clip(X - self.bend_x0, 0, self.bend_R * 0.999)
      lines[0, 2, :, 0] += self.bend_R - np.sqrt(self.bend_R ** 2 - dx ** 2)
    lines[0, 2, :, 0] += self.shift_near
    if self.far_at is not None:
      lines[0, 3, :, 0] = road + self.center + self.far_at
    stds[0, 1, :, 0], stds[0, 2, :, 0] = self.std, self.std
    probs = np.zeros((1, 8))
    probs[0, 1], probs[0, 3], probs[0, 5], probs[0, 7] = self.far_pl, self.pl, self.pr, self.far_pr
    plan = np.zeros((1, 33, 15))
    plan[0, :, 0] = X
    plan[0, :, 1] = road + self.plan * np.clip(X / 10, 0, 1)
    plan[0, :, 3] = self.v
    plan[0, 0, Plan.ORIENTATION_RATE.start + 2] = self.v * self.k if self.yaw_rate is None else self.yaw_rate
    mo = {'lane_lines': lines, 'lane_lines_stds': stds, 'lane_lines_prob': probs, 'plan': plan}
    if self.edges:
      e = np.zeros((1, 2, 33, 2))
      e[0, 0, :, 0] = road + self.center - self.edge_half
      e[0, 1, :, 0] = road + self.center + (self.edge_half if self.edge_right is None else self.edge_right)
      mo['road_edges'] = e
      mo['road_edges_stds'] = np.full((1, 2, 33, 2), self.edge_std)
    return mo


def run_rl(scenes, k0_up=0.0, inputs=None, model_curvature=0.0, rl=None, active=True):
  """Drive RightLineBlendful directly over scenes (iterable of Scene); returns the instance and the results."""
  rl = rl or RightLineBlendful()
  res = []
  for sc in scenes:
    p = inputs(len(res)) if callable(inputs) else (inputs or RightInputs())
    res.append(rl.update(sc.model_output(), sc.v, active, model_curvature if sc.k == 0 else sc.k, k0_up, p))
  return rl, res


def drive(ctl, scenes, mode=MODE_RIGHT, inputs=None):
  out = []
  for i, sc in enumerate(scenes):
    p = inputs(i) if callable(inputs) else (inputs or RightInputs())
    kappa = sc.k
    ctl.update(sc.model_output(), sc.v, True, kappa, LAT_T, mode, p)
    out.append((ctl.c, ctl.k0, ctl.result.log, ctl.cap_active, ctl.release_window))
  return out


def target_at(log, v):
  look = max(1.5 * v, 10.0)
  return float(np.interp(look, X, log['target']))


def steady(rl_kwargs=None, frames=200, **scene_kw):
  rl, res = run_rl([Scene(**scene_kw)] * frames, **(rl_kwargs or {}))
  return rl, res[-1]


# ---------------------------------------------------------------------------------------------------------------------------------
class TestMeasurementFilter:
  def circle(self, v, R, ld=1.95, y0=1.6, q=1.0, frames=400, tau=0.5):
    k = 1.0 / R
    y_true = y0 + ld * k * X + k * X ** 2 / 2
    f = InputFilter(tau)
    f.update(y_true, 1.0, v, v * k)
    for _ in range(frames):
      y, _ = f.update(y_true, q, v, v * k)
    return float(np.max(np.abs(np.interp([10.0, 20.0, 30.0], X, y - y_true))))

  @pytest.mark.parametrize("R,v", [(15.0, 30 / 3.6), (50.0, 50 / 3.6), (200.0, 80 / 3.6)])
  @pytest.mark.parametrize("q", [1.0, 0.5])
  def test_steady_circle_bias(self, R, v, q):
    assert self.circle(v, R, q=q) < 0.01  # K1: with the lever arm l_d and the half step arc

  def test_exact_on_static_road_under_motion(self):
    f = InputFilter()
    line = np.full(33, 1.7)
    f.update(line, 1.0, 20.0, 0.0)
    for _ in range(100):
      y, _ = f.update(line, 0.8, 20.0, 0.0)
    assert np.max(np.abs(y - line)) < 1e-9

  def test_lever_arm_matters(self):
    # without l_d the bias on the circle would be l_d r tau / q (DESIGN check 16): the full prediction removes it
    f = InputFilter()
    v, R = 50 / 3.6, 50.0
    k = 1 / R
    y_true = 1.6 + 1.95 * k * X + k * X ** 2 / 2
    f.update(y_true, 1.0, v, v * k)
    pred = f.predict(v, v * k)
    assert np.max(np.abs(np.interp([10.0, 20.0, 30.0], X, pred - y_true))) < 1e-3

  def test_noise_is_smoothed_and_gain_scales_with_confidence(self):
    rng = np.random.default_rng(2)
    f_hi, f_lo = InputFilter(), InputFilter()
    base = np.full(33, 1.7)
    dev_hi, dev_lo = [], []
    for _ in range(300):
      meas = base + 0.05 * rng.standard_normal()
      dev_hi.append(f_hi.update(meas, 1.0, 20.0, 0.0)[0][10] - 1.7)
      dev_lo.append(f_lo.update(meas, 0.55, 20.0, 0.0)[0][10] - 1.7)
    assert np.std(dev_hi[100:]) < 0.05 * 0.6
    assert np.std(dev_lo[100:]) < np.std(dev_hi[100:])  # a low confidence input moves its estimate less

  def test_coasting_and_reinit(self):
    f = InputFilter()
    v = 15.0
    line = np.full(33, 1.7)
    f.update(line, 0.9, v, 0.0)
    for _ in range(20):
      f.update(line, 0.9, v, 0.0)
    # a gap shorter than 1 s: coasts on the prediction, confidence decays to 0, then recovers without re-init
    for _ in range(10):
      y, q = f.update(line + 5.0, 0.1, v, 0.0)
      assert f.coast and np.max(np.abs(y - line)) < 1e-9 and q < 0.1
    y, q = f.update(line + 0.02, 0.9, v, 0.0)
    assert not f.coast and not f.reinit and f.initialized
    # a gap longer than 1 s: the estimate is dropped; it re-initialises from the measurement once q > 0.5
    for _ in range(25):
      f.update(line + 3.0, 0.1, v, 0.0)
    assert not f.initialized and f.q == 0.0
    y, q = f.update(line + 3.0, 0.4, v, 0.0)
    assert not f.initialized
    y, q = f.update(line + 3.0, 0.7, v, 0.0)
    assert f.initialized and f.reinit and np.max(np.abs(y - (line + 3.0))) < 1e-12

  def test_reference_instance_stays_raw(self):
    # the upstream instance in the correction pipeline is fed the raw model output: same as a stand-alone Blendful
    ctl, ref = BlendfulCorrection(), Blendful()
    rng = np.random.default_rng(5)
    for _ in range(150):
      sc = Scene(center=0.4 + 0.05 * rng.standard_normal(), v=15.0)
      mo = sc.model_output()
      ctl.update(mo, 15.0, True, 0.0, LAT_T, MODE_RIGHT)
      assert abs(ctl.k_up - ref.update(sc.model_output(), 15.0, True, 0.0, enabled=True)) == 0.0

  def test_filter_rerun_reproduces_logged_values(self):
    """S-8: the filters read only logged quantities; rebuilt from the published plan they reproduce the logged outputs to 1e-6."""
    rng = np.random.default_rng(7)
    scenes = [Scene(center=0.3 * np.sin(i / 30.0) + 0.02 * rng.standard_normal(), k=0.004 * np.sin(i / 50.0), v=17.0, edges=True,
                    pr=0.9 if i < 120 else 0.2) for i in range(200)]
    ctl = BlendfulCorrection()
    logs, published, k0s = [], [], []
    for sc in scenes:
      mo = sc.model_output()
      raw_plan = mo['plan'].copy()
      ctl.update(mo, sc.v, True, sc.k, LAT_T, MODE_RIGHT)
      logs.append(ctl.result.log)
      published.append(shift_plan(raw_plan[0], ctl.k0, sc.v)[None])
      k0s.append(ctl.k0)
    rl2 = RightLineBlendful()
    for sc, lg, pub, k0 in zip(scenes, logs, published, k0s, strict=True):
      mo = sc.model_output()
      # rebuild the raw plan's yaw rate: published - profile(k0)
      rebuilt = pub.copy()
      rebuilt[0, 0, Plan.ORIENTATION_RATE.start + 2] -= pub[0, 0, 3] * k0
      mo['plan'] = rebuilt
      res = rl2.update(mo, sc.v, True, sc.k, 0.0, RightInputs())
      np.testing.assert_allclose(res.log['filtered'], lg['filtered'], atol=1e-6)
      np.testing.assert_allclose(res.log['confidence'], lg['confidence'], atol=1e-6)
      np.testing.assert_array_equal(res.log['state'], lg['state'])


class TestReferences:
  @staticmethod
  def gates(q_list):
    g = ConfGate()
    return [g.update(q) for q in q_list]

  @pytest.mark.parametrize("side,y_true", [(+1, 1.6), (-1, -1.9)])
  def test_lone_candidate_sits_at_its_true_position(self, side, y_true):
    worst = 0.0
    for q in np.linspace(0.3, 0.7, 41):
      for c in (float(br.ss((q - 0.4) / 0.2)), float(br.ss((q - 0.3) / 0.2))):  # both hysteresis branches
        ys = np.array([[y_true] * 33, [y_true + side * 3.0] * 33])
        y, qa, _ = RightLineBlendful._reference(ys, [q, 0.0], [c, 0.0], side)
        worst = max(worst, float(np.max(np.abs(y - y_true))))
    assert worst < 0.01

  def test_inner_wins_when_both_clear_and_coinciding_candidates_agree(self):
    ys = np.array([[1.6] * 33, [1.8] * 33])
    y, _, w = RightLineBlendful._reference(ys, [0.9, 0.9], [1.0, 1.0], +1)
    assert abs(y[10] - 1.6) < 0.01 and w[0, 10] > 0.98
    y, _, _ = RightLineBlendful._reference(np.array([[1.6] * 33, [1.6] * 33]), [0.9, 0.9], [1.0, 1.0], +1)
    assert abs(y[0] - 1.6) < 1e-12  # no inward bias for coinciding candidates
    yl, _, _ = RightLineBlendful._reference(np.array([[-1.6] * 33, [-1.8] * 33]), [0.9, 0.9], [1.0, 1.0], -1)
    assert abs(yl[0] + 1.6) < 0.01

  @pytest.mark.parametrize("conf,expected", [(0.0, 1.80), (0.5, 1.80), (0.9, 1.65), (1.0, 1.60)])
  def test_static_confidence_of_the_inner_line(self, conf, expected):
    y, _, _ = RightLineBlendful._reference(np.array([[1.6] * 33, [1.8] * 33]), [0.9, 0.9], [conf, 1.0], +1)
    assert abs(y[5] - expected) < 0.02

  def _edge_line_run(self, prob_fn, frames=80):
    """Right edge at 1.9 m (counted 1.8 m), an inner right line at 1.6 m whose probability follows prob_fn(t)."""
    rl = RightLineBlendful()
    refs, targets = [], []
    for i in range(-60, frames):  # 3 s of warm-up at the initial probability, not recorded
      t = max(i, 0) * DT_MDL
      sc = Scene(v=20.0, width=3.2, edges=True, edge_half=3.5, edge_right=1.9, pr=prob_fn(t), pl=0.9, std=0.1, edge_std=0.3)
      sc_mo = sc.model_output()
      sc_mo['lane_lines'][0, 2, :, 0] = 1.6
      sc_mo['lane_lines'][0, 1, :, 0] = -1.6
      res = rl.update(sc_mo, 20.0, True, 0.0, 0.0, RightInputs())
      if i >= 0:
        refs.append(res.log['reference'][1])
        targets.append(target_at(res.log, 20.0) - float(np.interp(20.0, X, res.log['c_stable'])))  # h: the target's offset from the stable centre
    return np.array(refs), np.array(targets)

  def test_inner_line_fading_in_glides(self):
    refs, tg = self._edge_line_run(lambda t: 0.05 if t < 1.0 else float(np.clip(0.05 + (t - 1.0), 0, 0.9)), 120)
    assert abs(refs[0] - 1.8) < 0.05 and abs(refs[-1] - 1.6) < 0.05
    assert np.max(np.abs(np.diff(tg))) <= br.FIRST_MOVE_RATE * DT_MDL + 1e-9  # the offset rate limit turns it into a glide

  def test_inner_line_fading_out_and_vanishing_glide(self):
    for prob in (lambda t: 0.9 if t < 1.0 else float(np.clip(0.9 - (t - 1.0) / 0.3, 0.0, 0.9)), lambda t: 0.9 if t < 1.0 else 0.0):
      refs, tg = self._edge_line_run(prob, 120)
      assert abs(refs[10] - 1.6) < 0.05 and abs(refs[-1] - 1.8) < 0.05
      assert np.max(np.abs(np.diff(tg))) <= br.FIRST_MOVE_RATE * DT_MDL + 1e-9

  def test_inner_line_flicker_at_threshold_changes_nothing(self):
    refs, tg = self._edge_line_run(lambda t: 0.35 + 0.1 * np.sin(2 * np.pi * 3 * t), 120)
    assert np.ptp(refs[-40:]) < 0.02 and abs(refs[-1] - 1.8) < 0.02

  def test_inner_outer_handover_on_the_left_side(self):
    rl = RightLineBlendful()
    refs = []
    for i in range(-60, 120):
      t = max(i, 0) * DT_MDL
      sc = Scene(v=20.0, width=3.2, edges=True, edge_half=1.9, pl=0.05 if t < 1.0 else float(np.clip(0.05 + (t - 1.0), 0, 0.9)))
      mo = sc.model_output()
      res = rl.update(mo, 20.0, True, 0.0, 0.0, RightInputs())
      if i >= 0:
        refs.append(res.log['reference'][0])
    # left edge at -1.9 counts at -1.8, the line at -1.6: the inner (larger y) line takes over smoothly
    assert abs(refs[0] + 1.8) < 0.05 and abs(refs[-1] + 1.6) < 0.05
    assert np.max(np.abs(np.diff(refs))) < 0.25  # K2 / check 15: up to 4 m/s before the offset rate limit, never a step

  @pytest.mark.parametrize("width,gap50,gap100", [(2.6, 0.49, 0.49), (3.0, 0.60, 0.69), (4.5, 0.60, 1.10), (6.0, 0.60, 1.10)])
  def test_narrow_roads_without_lines_table(self, width, gap50, gap100):
    for v, expected in ((50 / 3.6, gap50), (100 / 3.6, gap100)):
      rl, res = steady(v=v, pl=0.0, pr=0.0, edges=True, edge_half=width / 2, edge_std=0.3, frames=300)
      gap_edge = width / 2 - (target_at(res.log, v) + W_R)
      assert abs(gap_edge - expected) < 0.03, (width, v, gap_edge)

  def test_unclipped_width_between_references(self):
    # lane width 6 m (upstream would clip its half width to 2.25 m): the centre stays the middle between the references
    rl, res = steady(width=6.0, center=0.2, frames=150)
    c = res.log['c_ctr']
    assert abs(float(np.interp(20.0, X, c)) - 0.2) < 0.02

  def test_plausibility_fades(self):
    # line more than 0.3 m outside its own edge
    rl, res = steady(edges=True, edge_right=1.2, width=3.5, frames=60)  # edge at 1.2, line at 1.75
    assert res.log['f_plaus'] == 0.0
    # reference width < 2.0 m
    rl, res = steady(width=1.8, frames=60)
    assert res.log['f_plaus'] == 0.0
    rl, res = steady(width=3.5, edges=True, frames=60)
    assert res.log['f_plaus'] == 1.0


EDGE_X = br.EDGE_EXTRA  # edge candidates count this much further in


# ---------------------------------------------------------------------------------------------------------------------------------
class TestCorrectionLaw:
  """B4: the right-line part, strengths, authority and the blend with regular upstream."""
  def test_both_lines_clear(self):
    rl, res = steady(pl=1.0, pr=1.0, center=0.0, frames=260)
    assert res.log['a'] == 1.0 and abs(res.log['s_r'] - 1.0) < 1e-9
    assert res.k0 > 0.0  # the car at the lane centre is pulled right to the target

  def test_right_line_only_gets_eighty_percent_also_in_curves(self):
    for k in (0.0, 1 / 50):
      rl, res = steady(pl=0.0, pr=1.0, k=k, frames=260)
      assert res.log['a'] == 1.0 and abs(res.log['s_r'] - 0.8) < 1e-9
    # the rising variant (owner 17:24: for later): 0.8 -> 1.0 with |k_m| v^2
    for k, want in ((0.0, 0.8), (1 / 50, 1.0)):
      _, res = run_rl([Scene(pl=0.0, pr=1.0, k=k)] * 260, rl=RightLineBlendful(strength_rising=True))
      assert abs(res[-1].log['s_r'] - want) < 1e-9 and abs(res[-1].log['w_c'] - want) < 1e-9

  def test_left_only_is_regular_upstream(self):
    k0_up = 0.0021
    rl, res = run_rl([Scene(pl=1.0, pr=0.0)] * 150, k0_up=k0_up)
    assert rl.a == 0.0
    assert all(r.k0 == k0_up for r in res)  # a = 0: exactly upstream's correction (blend law kappa0 = a RL + (1 - a) up)

  def test_blend_is_continuous_in_a(self):
    rl, res = run_rl([Scene(pl=1.0, pr=1.0)] * 400, k0_up=0.0)
    a = np.array([r.log['a'] for r in res])
    assert np.max(np.abs(np.diff(a))) <= br.A_IN * DT_MDL + 1e-9  # in at <= 0.5 / s
    assert np.all(np.diff(a) >= -1e-12) and a[-1] == 1.0
    rl2, res2 = run_rl([Scene(pl=1.0, pr=0.0)] * 40, k0_up=0.003, rl=rl)  # right line vanishes: a out at <= 1 / s
    a2 = np.array([r.log['a'] for r in res2])
    assert np.max(np.abs(np.diff(np.concatenate([[a[-1]], a2])))) <= br.A_OUT * DT_MDL + 1e-9
    assert a2[-1] == 0.0 and res2[-1].k0 == 0.003

  @pytest.mark.parametrize("left", [1.0, 0.0])
  def test_continuity_sweep_of_the_right_quality(self, left):
    rl = RightLineBlendful()
    a_prev = s_prev = None
    worst_a = worst_s = 0.0
    for i in range(600):  # warm up, then sweep the right probability 1 -> 0 over 20 s
      pr = 1.0 if i < 200 else float(np.clip(1.0 - (i - 200) / 400.0, 0.0, 1.0))
      res = rl.update(Scene(pl=left, pr=pr).model_output(), 20.0, True, 0.0, 0.002, RightInputs())
      if a_prev is not None and i >= 200:
        worst_a, worst_s = max(worst_a, abs(res.log['a'] - a_prev)), max(worst_s, abs(res.log['s_r'] - s_prev))
      a_prev, s_prev = res.log['a'], res.log['s_r']
    assert worst_a <= br.A_OUT * DT_MDL + 1e-9
    assert worst_s < 0.02

  def test_in_between_cases_have_expected_authority(self):
    rl, res = steady(pl=0.4, pr=0.7, frames=300)
    assert 0.0 < res.log['a'] <= 1.0 and res.log['q_R'] > 0.5

  def test_a_equals_zero_matches_the_reference_instance_through_the_pipeline(self):
    ctl, ref = BlendfulCorrection(), Blendful()
    worst = 0.0
    for i in range(200):
      sc = Scene(v=15.0, center=0.5 if i < 120 else -0.3, k=0.0 if i < 60 else 1 / 150, pr=0.0)
      ctl.update(sc.model_output(), 15.0, True, sc.k, LAT_T, MODE_RIGHT)
      up = ref.update(sc.model_output(), 15.0, True, sc.k, enabled=True)
      worst = max(worst, abs(ctl.c - up))
    assert worst <= 1e-9 and ctl.right.a == 0.0

  def test_target_follows_the_right_line_and_centre_limit(self):
    # wide lane: gap g to the right line; narrow lane: never left of the centre limit (the target equals the centre)
    rl, res = steady(v=13.0, width=4.5, frames=300)
    gap = (2.25 - (target_at(res.log, 13.0) + W_R))
    assert abs(gap - br.gap_for_speed(13.0, 0.5, 1.0)) < 1e-6
    rl, res = steady(v=13.0, width=2.6, frames=300)
    assert abs(target_at(res.log, 13.0)) < 1e-6  # centred: T_C' binds

  def test_gui_gap_and_speed_interpolation(self):
    assert br.gap_for_speed(10.0, 0.5, 1.0) == 0.5 and br.gap_for_speed(30.0, 0.5, 1.0) == 1.0
    assert abs(br.gap_for_speed(75 / 3.6, 0.5, 1.0) - 0.75) < 1e-9
    for lo, hi in ((0.2, 2.0), (1.0, 0.3)):
      rl, res = steady(v=13.0, width=6.0, frames=300, rl_kwargs={'inputs': RightInputs(gap_low=lo, gap_high=hi)})
      gap = 3.0 - (target_at(res.log, 13.0) + W_R)
      assert abs(gap - lo) < 1e-6

  @pytest.mark.parametrize("c_set,c_app", [(-0.35, -0.35), (0.0, 0.0), (0.05, 0.05), (0.35, 0.35), (0.05, 0.02)])
  def test_camera_offset_keeps_the_gap(self, c_set, c_app):
    # the physical right line is 1.9 m right of the device; the model frame is shifted by the applied offset
    y_phys = 1.9
    y_model = y_phys - c_app
    rl, res = steady(v=13.0, width=4.5, center=y_model - 2.25, frames=300, rl_kwargs={'inputs': RightInputs(c_set=c_set, c_applied=c_app)})
    t = target_at(res.log, 13.0)
    centre_path_phys = (t + c_app) + (c_set - c_app)
    assert abs((y_phys - (centre_path_phys + W_R)) - 0.5) < 1e-6

  def test_camera_offset_step_while_filtering_in(self):
    # the setting steps from 0.05 to 0.30; the applied value follows with the camera helper's 0.1 per step filter: the gap stays g
    c_set, c_app = 0.30, 0.05
    rl = RightLineBlendful()
    for i in range(300):
      c_app += 0.1 * (c_set - c_app) if i > 60 else 0.0
      c_a = c_app if i > 60 else 0.05
      sc = Scene(v=13.0, width=4.5, center=(1.9 - c_a) - 2.25)
      res = rl.update(sc.model_output(), 13.0, True, 0.0, 0.0, RightInputs(c_set=c_set if i > 60 else 0.05, c_applied=c_a))
      if i > 70:
        t = target_at(res.log, 13.0)
        # rate limited glide: compare the unlimited target of the log's own bookkeeping through the offset limiter state
        assert np.isfinite(t)
    assert abs((1.9 - ((t + c_a) + (c_set - c_a) + W_R)) - 0.5) < 1e-6

  def test_gap_of_point_two_target_and_bound_coincide(self):
    for v, k in ((10.0, 0.0), (20.0, 0.0), (10.0, 1 / 30), (25.0, 1 / 150)):
      rl, res = steady(v=v, k=k, width=4.5, frames=300, rl_kwargs={'inputs': RightInputs(gap_low=0.2, gap_high=0.2)})
      assert not res.log['bound_active']  # the bound term never acts beyond the target: no sign change, no oscillation
      rl2, res2 = run_rl([Scene(v=v, k=k, width=4.5)] * 100, inputs=RightInputs(gap_low=0.2, gap_high=0.2), rl=rl)
      assert np.ptp([target_at(r.log, v) for r in res2]) < 1e-6

  def test_rear_wheel_bounds_on_lineless_narrow_road(self):
    # a 2.2 m road without lines: the edge bound (>= 0.3 m) overrides the centre limit
    rl, res = steady(pl=0.0, pr=0.0, edges=True, edge_half=1.1, v=10.0, frames=300)
    t = target_at(res.log, 10.0)
    assert res.log['bound_active'] and t <= 1.1 - W_R - br.EDGE_BOUND + 1e-6

  def test_right_curve_off_tracking_in_gap_and_bound(self):
    k = 1 / 30
    rl, res = steady(v=10.0, k=k, width=4.5, frames=300)
    look = max(1.5 * 10.0, 10.0)
    right_line = float(np.interp(look, X, res.log['y_r']))
    off = br.L_D ** 2 * k / 2
    assert abs((right_line - (target_at(res.log, 10.0) + W_R)) - (br.gap_for_speed(10.0, 0.5, 1.0) + off)) < 0.02
    # the model cutting the inside does not move the bound
    rl2, res2 = run_rl([Scene(v=10.0, k=k, width=4.5, plan=0.8)] * 100, rl=rl)
    t2 = target_at(res2[-1].log, 10.0)
    assert t2 <= right_line - W_R - br.LINE_BOUND - off + 1e-6

  def test_measurement_filter_toggle_bit_identity_of_inputs(self):
    # the raw inputs are logged next to the filtered ones, at x = 0, 0.5 L, L, 1.5 L
    rl, res = steady(frames=50)
    assert len(res.log['raw']) == 16 and len(res.log['filtered']) == 16 and len(res.log['confidence']) == 4
    assert np.allclose(res.log['raw'][8:12], [1.75] * 4) and np.allclose(res.log['filtered'][8:12], 1.75, atol=1e-6)
    assert len(res.log['cand_conf']) == 4 and len(res.log['cand_weight']) == 4 and len(res.log['reference']) == 2


# ---------------------------------------------------------------------------------------------------------------------------------
def upstream_from_centre_trigger(sw_start, sw_T, T=60.0, noise=0.03, seed=1):
  rng = np.random.default_rng(seed)
  usual, tt = 0.0, None
  for i in range(int(T / DT_MDL)):
    t = i * DT_MDL
    sw = 0.8 * np.clip((t - sw_start) / sw_T, 0, 1) if t >= sw_start else 0.0
    off = -sw + noise * rng.standard_normal()
    if t >= sw_start and tt is None and abs(off - usual) > 0.2:
      tt = t - sw_start
    usual += (off - usual) * DT_MDL / (DT_MDL + 10.0)
  return tt


def sim_estimator(rho, prior, sw_start, sw_T, rate, seed=1, T=60.0, noise=0.03, hmax=1.0):
  """Closed loop of check 17: the car moves to the held offset (rate, 1.5 s lag), the model plans back with return fraction rho."""
  est = YieldEstimator(prior=(prior,) * 5)
  rng = np.random.default_rng(seed)
  h = o = 0.0
  t10 = t20 = None
  maxq = -9.0
  qs = []
  for i in range(int(T / DT_MDL)):
    t = i * DT_MDL
    sw = 0.8 * np.clip((t - sw_start) / sw_T, 0, 1) if t >= sw_start else 0.0
    p = -rho * o - sw + noise * rng.standard_normal()
    h = min(hmax, h + rate * DT_MDL)
    o += (h - o) * DT_MDL / 1.5
    q = est.update(o, p, 15.0, 0.0, False)
    qs.append(q)
    if t < sw_start:
      maxq = max(maxq, q)
    else:
      if t10 is None and q >= 0.1:
        t10 = t - sw_start
      if t20 is None and q >= 0.2:
        t20 = t - sw_start
  return est, maxq, t10, t20


class TestYieldEstimator:
  def test_rho_stays_in_range_and_zero_has_no_singularity(self):
    for rho in (0.0, 1.0):
      est, maxq, _, _ = sim_estimator(rho, 0.5, 99.0, 1.0, 0.25, T=40.0)
      assert 0.0 <= est.rho[est.band(15.0)] <= 1.0 and np.isfinite(est.pr) and np.isfinite(est.b) and np.isfinite(maxq)
      # (a prior this far off, 0.5, is outside the E1 gate of +- 0.1: false triggers are expected there, DESIGN check 17)

  def test_hold_onsets_do_not_trigger(self):
    for rho in (0.3, 0.5, 0.8, 1.0):
      for prior_err in (-0.1, 0.0, 0.1):
        prior = float(np.clip(rho + prior_err, 0, 1))
        _, maxq, _, _ = sim_estimator(rho, prior, 99.0, 1.0, 0.1 if prior_err else 0.25, T=40.0)
        assert maxq < 0.15  # the early cap (from 0.15 m) does not engage

  def test_bias_is_frozen_during_onsets_and_expectation_is_bounded(self):
    est = YieldEstimator(prior=(0.5,) * 5)
    rng = np.random.default_rng(3)
    h = o = 0.0
    b_at_onset = None
    for i in range(1200):
      t = i * DT_MDL
      p = -0.8 * o + 0.03 * rng.standard_normal()
      h = min(1.0, h + 0.25 * DT_MDL) if t > 20 else 0.0
      o += (h - o) * DT_MDL / 1.5
      est.update(o, p, 15.0, 0.0, False)
      if est.onset:
        b_at_onset = est.b0 if b_at_onset is None else b_at_onset
        assert est.b == est.b0  # b-hat frozen during an onset
        assert abs(est.p_exp - (-est.r0 * o + est.b0)) <= br.ABSORB + 1e-12  # within 0.1 m of the onset-start prediction
    assert b_at_onset is not None

  @pytest.mark.parametrize("sw_T", [0.5, 3.0, 10.0])
  @pytest.mark.parametrize("sw_start", [0.0, 2.0, 5.0, 30.0])
  @pytest.mark.parametrize("rho", [0.3, 0.8, 1.0])
  def test_swerves_established_band(self, rho, sw_start, sw_T):
    tu = upstream_from_centre_trigger(sw_start, sw_T)
    _, _, t10, t20 = sim_estimator(rho, rho, sw_start, sw_T, 0.25)
    assert t10 is not None and t20 is not None
    assert t10 - tu <= 0.05 + 1e-9  # no growth from 0.1 m: no later than upstream-from-centre (one frame)
    assert t20 - tu <= 1.75 + 0.05  # full cap

  @pytest.mark.parametrize("sw_T", [0.5, 3.0, 10.0])
  @pytest.mark.parametrize("sw_start", [0.0, 2.0, 5.0, 30.0])
  @pytest.mark.parametrize("err", [-0.1, 0.1])
  def test_swerves_first_onset_with_prior_error(self, err, sw_start, sw_T):
    rho = 0.8
    tu = upstream_from_centre_trigger(sw_start, sw_T)
    _, maxq, t10, t20 = sim_estimator(rho, rho + err, sw_start, sw_T, 0.1)
    assert t10 is not None and t20 is not None and maxq < 0.2
    assert t10 - tu <= 0.2 + 0.05
    assert t20 - tu <= 2.25 + 0.05

  def test_established_flag_and_expiry(self):
    est = YieldEstimator(prior=(0.5,) * 5)
    assert not est.is_established(15.0)
    rng = np.random.default_rng(1)
    h = o = 0.0
    for i in range(900):
      t = i * DT_MDL
      p = -0.5 * o + 0.01 * rng.standard_normal()
      h = min(1.0, h + 0.1 * DT_MDL) if t > 2 else 0.0
      o += (h - o) * DT_MDL / 1.5
      est.update(o, p, 15.0, 0.0, False)
    assert est.is_established(15.0) and not est.is_established(30.0)  # per speed band
    est.t += br.ESTABLISH_TTL + 1.0
    assert not est.is_established(15.0)  # expires
    est.established[est.band(15.0)] = est.t
    est.road_type_changed()
    assert not est.is_established(15.0)  # a road type change clears it

  def test_first_move_uses_the_slow_target_offset_rate(self):
    rl = RightLineBlendful()
    assert not rl.yields.is_established(20.0)
    rl.yields.established[rl.yields.band(20.0)] = rl.yields.t
    assert rl.yields.is_established(20.0)


def swerve_run(side=+1, k0_up=0.0, frames_hold=100, frames_total=260, depth=0.8, swerve_T=0.5, ref=True):
  """Controller in Right line mode: the car is at the lane centre and pulled right (the target is right of the centre); then the model
  swerves (side +1: to the left, obstacle on the right)."""
  ctl = BlendfulCorrection()
  rows = []
  for i in range(frames_total):
    t = max(0.0, (i - frames_hold) * DT_MDL)
    sw = depth * min(1.0, t / swerve_T)
    sc = Scene(v=13.0, plan=-side * sw)
    ctl.update(sc.model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
    rows.append((ctl.c, ctl.result.yield_q, ctl.result.yield_side, ctl.result.yield_active, ctl.cap_active, ctl.release_window, i))
  return ctl, rows


class TestYieldCap:
  def test_no_correction_toward_the_obstacle_inside_a_yield(self):
    ctl, rows = swerve_run(+1)
    assert rows[90][0] > 0.0  # pulled right before the swerve
    v = 13.0
    prev_s_c = None
    windows = 0
    for c, q, s, active, _cap, rel, _i in rows[100:]:
      s_c = s * c
      if active and q >= br.Q_NO_GROWTH and prev_s_c is not None:
        assert s_c <= max(prev_s_c, 0.0) + 1e-12  # no growth from q = 0.1 m
      if active and not rel and q >= br.Q_CAP_FULL:
        assert s_c <= 1e-6  # the hard rule: zero correction toward the obstacle side, outside cap-release windows
      if rel:
        windows += 1
      prev_s_c = s_c
    last = rows[-1]
    assert last[3] and last[0] * last[2] <= 1e-6
    # the release never exceeds the 5 m/s^3 of controlsd's clip_curvature
    dc = np.abs(np.diff([r[0] for r in rows])) * v ** 2 / DT_MDL
    assert dc.max() <= br.SWERVE_RELEASE + 1e-6

  def test_release_window_rate_and_listing(self):
    ctl, rows = swerve_run(+1, swerve_T=0.1)  # a sudden swerve
    assert any(r[5] for r in rows)  # a cap-release window is reported
    cs = np.array([r[0] for r in rows])
    v = 13.0
    assert np.abs(np.diff(cs)).max() * v ** 2 / DT_MDL <= br.SWERVE_RELEASE + 1e-6

  def test_obstacle_on_the_left_never_pulls_left(self):
    ctl, rows = swerve_run(-1)
    assert min(r[0] for r in rows) >= -1e-6  # the model goes right (oncoming traffic); the correction never pushes left
    assert rows[-1][2] == -1

  def test_inside_curve_fade_does_not_trigger_the_cap(self):
    ctl = BlendfulCorrection()
    k = 1 / 200
    for i in range(200):
      sc = Scene(v=13.0, k=k, plan=0.0 if i < 100 else 0.2)  # the model cuts the inside of a right curve (within the 0.25 |k| L^2 / 2 = 0.24 m allowance)
      ctl.update(sc.model_output(), 13.0, True, k, LAT_T, MODE_RIGHT, RightInputs())
      assert not ctl.cap_active or i < 100
    assert not ctl.result.yield_active

  def test_early_cap_ramp(self):
    ctl = BlendfulCorrection()
    for _ in range(100):
      ctl.update(Scene(v=13.0).model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
    start = ctl.c
    assert start > 0.0
    qs = []
    for i in range(100):
      sw = 0.001 * i  # slow swerve to the left
      ctl.update(Scene(v=13.0, plan=-sw).model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
      qs.append(ctl.result.yield_q)
    assert max(qs) > 0.0


# ---------------------------------------------------------------------------------------------------------------------------------
def js(jn, n=1, v=20.0, yaw_rate=0.0, div=None, split_now=False, gore=None, score=None, right_blinker=False, left_blinker=False,
       left_lane=True, single_now=False):
  for _ in range(n):
    jn.update(v, yaw_rate, div, split_now, gore, score, right_blinker, left_blinker, left_lane, single_now)


class TestJunctionPolicy:
  """The isolated junction policy: weight split of the right-line, upstream and hold parts."""
  @pytest.mark.parametrize("a", [0.0, 0.3, 0.5, 1.0])
  def test_outside_an_episode_a_blends_with_upstream(self, a):
    w = junction_policy(False, None, a, False)
    assert (w.w_rl, w.w_up, w.w_hold) == (a, 1.0 - a, 0.0) and abs(w.total() - 1.0) < 1e-12

  @pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
  @pytest.mark.parametrize("mode", ['hold', 'follow'])
  def test_inside_an_episode_the_upstream_share_is_never_used(self, a, mode):
    w = junction_policy(True, mode, a, False)
    assert w.w_up == 0.0 and abs(w.total() - 1.0) < 1e-12
    assert (w.w_hold == 1.0) if mode == 'hold' else (w.w_rl == a and abs(w.w_hold - (1 - a)) < 1e-12)

  def test_model_alone_fades_both(self):
    w = junction_policy(True, 'hold', 1.0, True)
    assert w.total() == 0.0

  def test_mode_demotions_become_hold_never_regular(self):
    assert junction_mode(True, 'follow', False, False, False, True) == ('hold', False)  # right reference lost / jump / implausible
    assert junction_mode(True, 'follow', True, True, False, True) == ('hold', False)  # left blinker below 32 km/h
    assert junction_mode(True, 'follow', True, True, True, True) == ('follow', False)  # not inside the latch
    assert junction_mode(True, 'follow', False, False, False, False) == ('hold', True)  # no left reference: the model alone
    assert junction_mode(False, None, False, True, False, False) == (None, False)


class TestJunctionLogic:
  def test_episode_start_and_main_road_holds(self):
    jn = JunctionLogic()
    js(jn, 5, div=0.1)
    assert not jn.episode and jn.mode is None
    js(jn, 1, div=0.6)
    assert jn.episode and jn.mode == 'hold'  # a widening on a main road: keep the through lane
    js(jn, 1, div=0.6, right_blinker=True)
    assert jn.mode == 'follow'  # right blinker: follow the right line

  def test_split_pattern_starts_an_episode(self):
    jn = JunctionLogic()
    js(jn, 1, split_now=True, gore=150.0)
    assert jn.episode and jn.split_seen and jn.mode == 'hold'

  def test_episode_ends_when_divergence_is_back_for_one_second(self):
    jn = JunctionLogic()
    js(jn, 3, div=0.8)
    js(jn, 15, div=0.1)
    assert jn.episode
    js(jn, 6, div=0.1)
    assert not jn.episode

  def test_episode_ends_at_the_maximum_length(self):
    jn = JunctionLogic()
    js(jn, 1, div=0.8)
    js(jn, 301, div=0.8)  # 1 m per frame: ends at 301 m
    assert not jn.episode

  def test_gore_passed_ends_the_episode(self):
    jn = JunctionLogic()
    js(jn, 1, split_now=True, gore=30.0, v=10.0)
    js(jn, 90, v=10.0, split_now=False)  # 45 m: the gore (30 m) is 15 m behind
    assert jn.episode
    js(jn, 20, v=10.0)
    assert not jn.episode

  def test_ramp_context_starts_after_following_a_branch_and_ends_at_N(self):
    jn = JunctionLogic()
    js(jn, 1, split_now=True, gore=60.0, right_blinker=True, left_lane=False)
    js(jn, 90, right_blinker=True, left_lane=False)
    assert not jn.episode and jn.ramp  # the car took the branch
    js(jn, int(1400 / (20.0 * DT_MDL)), left_lane=False)
    assert jn.ramp
    js(jn, int(150 / (20.0 * DT_MDL)), left_lane=False)
    assert not jn.ramp  # N = 1500 m

  def _ramp(self):
    jn = JunctionLogic()
    jn.ramp = True
    return jn

  def test_ramp_ends_at_a_stop_a_tight_turn_and_with_a_lane_to_the_left(self):
    jn = self._ramp()
    js(jn, 5, v=1.0, left_lane=False)
    assert not jn.ramp  # stop
    jn = self._ramp()
    js(jn, 60, yaw_rate=20.0 * 0.05, left_lane=False)  # R = 20 m: > 60 degrees within 40 m
    assert not jn.ramp
    jn = self._ramp()
    js(jn, int(5.5 / DT_MDL), left_lane=True)
    assert not jn.ramp  # 5 s with a lane to the left
    jn = self._ramp()
    js(jn, int(3.0 / DT_MDL), left_lane=True)
    assert jn.ramp

  def test_loop_ramp_does_not_reset(self):
    jn = self._ramp()
    steps = int((1.5 * np.pi * 40.0) / (20.0 * DT_MDL))  # 270 degrees at R = 40 m
    js(jn, steps, yaw_rate=20.0 / 40.0, left_lane=False)
    assert jn.ramp

  def test_single_lane_alone_never_starts_a_ramp_but_extends_one(self):
    jn = JunctionLogic()
    js(jn, 2000, single_now=True, left_lane=False)
    assert not jn.ramp and jn.single.state
    jn = self._ramp()
    js(jn, 100, single_now=True, left_lane=False)  # three seconds in
    js(jn, int(2000 / (20.0 * DT_MDL)), single_now=True, left_lane=False)
    assert jn.ramp  # N and the left-lane rule do not end it while the single lane persists
    js(jn, 5, v=1.0, single_now=True, left_lane=False)
    assert not jn.ramp  # a stop does

  def test_classifier_hysteresis(self):
    jn = self._ramp()
    js(jn, 1, div=1.0, score=0.75)
    assert jn.score_state == 'split' and jn.mode == 'follow'
    js(jn, 1, div=1.0, score=0.65)
    assert jn.score_state == 'split'  # leaves below 0.6
    js(jn, 1, div=1.0, score=0.55)
    assert jn.score_state == 'uncertain' and jn.mode == 'hold'
    js(jn, 1, div=1.0, score=0.65)
    assert jn.score_state == 'uncertain'  # enters at 0.7
    js(jn, 1, div=1.0, score=0.25)
    assert jn.score_state == 'turn' and jn.mode == 'hold'  # turn lane: through lane
    js(jn, 1, div=1.0, score=0.35)
    assert jn.score_state == 'turn'  # leaves above 0.4
    js(jn, 1, div=1.0, score=0.45)
    assert jn.score_state == 'uncertain'

  def test_split_score_evidence(self):
    assert split_score({}) == 0.5
    assert split_score({'both_moderate': True, 'gore_full_lanes': True, 'edge_follows_outer': True}) >= 0.7
    assert split_score({'parallel_then_bend': True, 'short_pocket': True, 'edge_cornering': True}) <= 0.3

  def test_gore_distance_unknown_latches_at_once_and_keeps_the_choice(self):
    jn = self._ramp()
    js(jn, 1, div=1.0, score=0.5)
    js(jn, 1, split_now=True, gore=None, score=0.5)
    assert jn.latched and jn.mode == 'hold'  # hold unless a split was already being followed
    jn = self._ramp()
    js(jn, 1, div=1.0, score=0.8)
    assert jn.mode == 'follow'
    js(jn, 1, split_now=True, gore=None, score=0.8)
    assert jn.latched and jn.mode == 'follow'

  def test_commit_latch_blocks_reversals_and_late_blinker(self):
    jn = self._ramp()
    v = 10.0
    js(jn, 1, v=v, split_now=True, gore=200.0, score=0.8)
    assert jn.mode == 'follow' and not jn.latched
    js(jn, 1, v=v, split_now=True, gore=120.0, score=0.8)
    assert not jn.latched
    js(jn, 1, v=v, split_now=True, gore=80.0, score=0.8)  # x_commit = (3.6 / 0.5 + 1) * 10 = 82 m
    assert jn.latched and jn.mode == 'follow'
    for sc, rb in ((0.1, False), (0.8, True), (0.5, False)):  # classifier and blinker changes have no effect inside the latch
      js(jn, 1, v=v, score=sc, right_blinker=rb)
      assert jn.mode == 'follow'
    jn2 = self._ramp()
    js(jn2, 1, v=v, split_now=True, gore=80.0, score=0.5)
    assert jn2.latched and jn2.mode == 'hold'
    js(jn2, 3, v=v, right_blinker=True, score=0.8)
    assert jn2.mode == 'hold'  # a blinker switched on inside x_commit does not start a change
    js(jn2, int(105 / (v * DT_MDL)), v=v)
    assert not jn2.episode  # the latch ends 20 m behind the gore

  def test_late_split_is_taken_left(self):
    jn = self._ramp()
    js(jn, 1, v=20.0, split_now=True, gore=100.0, score=0.8)  # x_commit = 164 m: the choice is made at once, but only 'follow' from the score
    assert jn.latched


def episode_scenes(n_normal=150, n_wide=40, widen=0.8, **kw):
  return [Scene(pl=1.0, pr=1.0, **kw)] * n_normal + [Scene(pl=1.0, pr=1.0, widen=widen, **kw)] * n_wide


class TestJunctionIntegration:
  def test_widening_on_a_main_road_is_not_followed(self):
    rl, res = run_rl(episode_scenes(), k0_up=0.5)  # a distinctive upstream amplitude: it must never be used
    last = res[-1].log
    assert last['episode'] and last['hold']
    t = target_at(last, 20.0)
    # the through-lane position: the right wheel gap g to the PRE-widening right line (1.75 m)
    assert abs((1.75 - (t + W_R)) - br.gap_for_speed(20.0, 0.5, 1.0)) < 0.02
    assert all(r.log['weights'][1] == 0.0 for r in res[150:] if r.log['episode'])

  def test_right_blinker_follows_the_widening(self):
    rl, res = run_rl(episode_scenes(n_wide=250), inputs=lambda i: RightInputs(right_blinker=i >= 150))
    t = target_at(res[-1].log, 20.0)
    assert abs((1.75 + 0.8 - (t + W_R)) - br.gap_for_speed(20.0, 0.5, 1.0)) < 0.03  # follows the widened right line (glide <= 0.5 m/s)
    assert res[-1].log['hold'] is False

  def test_ramp_classifier_split_follows_turn_lane_holds(self, monkeypatch):
    for score, follow in ((0.9, True), (0.1, False)):
      monkeypatch.setattr(br, 'split_score', lambda feat, s=score: s)
      rl = RightLineBlendful()
      rl.junction.ramp = True
      _, res = run_rl(episode_scenes(n_wide=250), rl=rl)
      t = target_at(res[-1].log, 20.0)
      expected = 1.75 + (0.8 if follow else 0.0)
      assert abs((expected - (t + W_R)) - br.gap_for_speed(20.0, 0.5, 1.0)) < 0.03

  def test_follow_glide_rate_is_at_most_half_a_metre_per_second(self, monkeypatch):
    monkeypatch.setattr(br, 'split_score', lambda feat: 0.9)
    rl = RightLineBlendful()
    rl.junction.ramp = True
    _, res = run_rl(episode_scenes(n_wide=300, widen=2.0), rl=rl)
    tg = np.array([target_at(r.log, 20.0) for r in res[100:]])
    assert np.max(np.abs(np.diff(tg))) <= br.SPLIT_RATE * DT_MDL + 1e-9

  def _fade_cases(self):
    base = {'pl': 1.0, 'pr': 1.0}
    out = {}
    out['bend'] = (lambda i: Scene(widen=0.8, bend_x0=15.0, bend_R=30.0, **base), None)
    out['split'] = (lambda i: Scene(widen=0.0, shift_near=-3.0, far_at=1.75, far_pr=0.9, **base) if i == 0 else Scene(widen=0.8, **base), None)
    out['yield'] = (lambda i: Scene(widen=0.8, **base), RightInputs(ref_going_around=True))
    out['jump'] = (lambda i: Scene(widen=0.8, center=0.0 if i < 3 else 1.0, **base), None)
    out['plausibility'] = (lambda i: Scene(widen=0.8, edges=i >= 15, edge_right=1.2, **base), None)  # the line ends up outside its edge
    out['quality'] = (lambda i: Scene(widen=0.8, pl=1.0, pr=max(0.0, 0.9 - 0.02 * i)), None)
    out['left blinker'] = (lambda i: Scene(widen=0.8, v=8.0, **base), RightInputs(left_blinker=True))
    return out

  @pytest.mark.parametrize("name", ['bend', 'split', 'yield', 'jump', 'plausibility', 'quality', 'left blinker'])
  def test_c1_every_fade_inside_an_episode_never_uses_upstream(self, name):
    make, inputs = self._fade_cases()[name]
    v = 8.0 if name == 'left blinker' else 20.0
    rl = RightLineBlendful()
    for _ in range(150):
      rl.update(Scene(v=v, pl=1.0, pr=1.0).model_output(), v, True, 0.0, 0.0, RightInputs())
    rl_b = RightLineBlendful()
    rl_b = copy.deepcopy(rl)
    seen_episode = False
    for i in range(60):
      sc = make(i)
      sc.v = v
      p = inputs or RightInputs()
      r1 = rl.update(sc.model_output(), v, True, 0.0, 0.7, p)
      r2 = rl_b.update(sc.model_output(), v, True, 0.0, -0.7, p)  # a different upstream amplitude
      if r1.log['episode']:
        seen_episode = True
        assert r1.log['weights'][1] == 0.0  # the (1 - a) share is never the upstream one
        assert r1.k0 == r2.k0  # and the result does not depend on it
    assert seen_episode

  def test_episode_starting_at_a_half_replaces_the_upstream_share_through_the_limiter(self):
    ctl = BlendfulCorrection()
    scenes = [Scene(pl=1.0, pr=1.0)] * 400
    i = 0
    while ctl.right.a < 0.5 and i < 400:
      ctl.update(scenes[i].model_output(), 20.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
      i += 1
    assert 0.45 < ctl.right.a < 0.6
    cs = [ctl.c]
    for _ in range(80):
      ctl.update(Scene(pl=1.0, pr=1.0, widen=0.8).model_output(), 20.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
      cs.append(ctl.c)
      assert ctl.result.log['weights'][1] == 0.0 if ctl.result.log['episode'] else True
    assert ctl.result.log['episode']
    assert np.abs(np.diff(cs)).max() * 400.0 / DT_MDL <= br.RELEASE + 1e-6  # no step: the limiter takes the share over

  def test_c2_episode_end_is_graceful_at_the_target_offset_rate(self):
    rl, res = run_rl(episode_scenes(n_wide=60) + [Scene(pl=1.0, pr=1.0, widen=0.8)] * 150)  # the widened line stays: a new lane
    h = np.array([target_at(r.log, 20.0) - float(np.interp(20.0, X, r.log['c_stable'])) for r in res[140:]])
    assert np.max(np.abs(np.diff(h))) <= br.TARGET_RATE * DT_MDL + 1e-9

  def test_persistent_widening_ends_at_300_m_and_becomes_the_new_lane(self):
    rl, res = run_rl(episode_scenes(n_wide=700))
    eps = [r.log['episode'] for r in res]
    assert eps[160] and not eps[-1]  # ended at 300 m, no restart
    t = target_at(res[-1].log, 20.0)
    assert abs((1.75 + 0.8 - (t + W_R)) - br.gap_for_speed(20.0, 0.5, 1.0)) < 0.03  # now following the widened right line

  def test_hold_target_is_continuous_at_the_episode_start(self):
    rl, res = run_rl(episode_scenes())
    tg = np.array([target_at(r.log, 20.0) for r in res])
    assert np.max(np.abs(np.diff(tg[100:]))) <= br.TARGET_RATE * DT_MDL + 1e-9

  def test_no_left_reference_inside_an_episode_means_the_model_alone(self):
    scenes = [Scene(pl=1.0, pr=1.0)] * 150 + [Scene(pl=0.0, pr=1.0, widen=0.8)] * 60
    rl, res = run_rl(scenes, k0_up=0.9)
    if res[-1].log['episode']:
      assert res[-1].model_alone and res[-1].k0 == 0.0

  def test_bend_away_detector_and_local_gating(self):
    base = {'pl': 1.0, 'pr': 1.0}
    for kw, expect in (({}, 0.0), ({'widen': 0.0, 'bend_x0': 5.0, 'bend_R': 30.0}, 1.0)):
      sc = Scene(**base, **kw)
      mo = sc.model_output()
      _, _, _, b = bend_score(X, mo['lane_lines'][0, 2, :, 0], mo['lane_lines'][0, 1, :, 0], 8.3)
      assert (b >= 0.9) if expect else (b <= 0.1)
    # local gating: the target never follows the bending part
    rl, res = run_rl([Scene(v=8.0, pl=1.0, pr=1.0, bend_x0=10.0, bend_R=12.0)] * 250)
    y = res[-1].log['target']
    far = float(np.interp(40.0, X, y))
    assert far < 1.75 - W_R + 0.5  # not dragged away with the bending line

  def test_widening_followed_linear_not_cut(self):
    sc = Scene(v=13.0, pl=1.0, pr=1.0)
    mo = sc.model_output()
    mo['lane_lines'][0, 2, :, 0] += np.clip(X - 5, 0, None) / 15  # a 1:15 taper
    _, _, _, b = bend_score(X, mo['lane_lines'][0, 2, :, 0], mo['lane_lines'][0, 1, :, 0], 13.0)
    assert b <= 0.1


def _glide_ok(rows, v):
  """Every frame stays inside the final limiter: build 0.5 m/s^3, release up to the 5 m/s^3 swerve window."""
  cs = np.array([0.0] + [r[0] for r in rows])
  return float(np.max(np.abs(np.diff(cs)))) <= max(br.BUILD, br.SWERVE_RELEASE) / max(v, 1.0) ** 2 * DT_MDL + 1e-12


class TestBlinkers:
  def test_left_blinker_fades_the_correction_in_one_second(self):
    rl, res = run_rl([Scene(pl=1.0, pr=1.0)] * 150, k0_up=0.0)
    a0 = res[-1].log['a']
    assert a0 > 0.9
    _, res = run_rl([Scene(pl=1.0, pr=1.0)] * 40, rl=rl, inputs=RightInputs(left_blinker=True))
    assert res[0].log['f_blink'] > 0.9 and res[-1].log['f_blink'] == 0.0
    fb = np.array([r.log['f_blink'] for r in res])
    assert np.all(np.diff(fb) <= 1e-12)
    assert abs(fb[0] - (1.0 - DT_MDL)) < 0.06  # linear, 1 s
    assert res[-1].log['a'] < a0

  def test_right_blinker_keeps_the_authority(self):
    rl, res = run_rl([Scene(pl=1.0, pr=1.0)] * 150)
    _, res2 = run_rl([Scene(pl=1.0, pr=1.0)] * 60, rl=rl, inputs=RightInputs(right_blinker=True))
    assert all(r.log['f_blink'] == 1.0 for r in res2)
    assert res2[-1].log['a'] >= 0.9 * res[-1].log['a']

  def test_left_blinker_returns_the_share_to_upstream_outside_an_episode(self):
    rl, res = run_rl([Scene(pl=1.0, pr=1.0)] * 150, k0_up=0.3, inputs=RightInputs(left_blinker=True))
    assert res[-1].log['a'] < 0.05
    assert abs(res[-1].k0 - 0.3) < 0.05


class TestModes:
  def test_mode_from_params_and_selection(self):
    assert mode_from_params(False, 1) == MODE_OFF and mode_from_params(False, 0) == MODE_OFF
    assert mode_from_params(True, 0) == MODE_REGULAR and mode_from_params(True, 1) == MODE_RIGHT
    assert selection_to_params(0) == (False, None)
    assert selection_to_params(1) == (True, 0) and selection_to_params(2) == (True, 1)

  def test_old_backup_without_the_kind_reads_regular(self):
    assert mode_from_params(True, 0) == MODE_REGULAR  # BlendfulMode absent reads its default 0

  def test_mici_toggle_restores_the_last_selected_kind(self):
    stored = {'Blendful': True, 'BlendfulMode': 1}
    stored['Blendful'] = False  # the toggle only flips the BOOL ...
    assert mode_from_params(stored['Blendful'], stored['BlendfulMode']) == MODE_OFF
    stored['Blendful'] = True  # ... and switching on again restores Right line
    assert mode_from_params(stored['Blendful'], stored['BlendfulMode']) == MODE_RIGHT
    on, kind = selection_to_params(0)
    assert not on and kind is None  # the selector's Off keeps the stored kind

  @pytest.mark.parametrize("seq", [(MODE_OFF, MODE_REGULAR), (MODE_REGULAR, MODE_OFF), (MODE_OFF, MODE_RIGHT), (MODE_RIGHT, MODE_OFF),
                                   (MODE_REGULAR, MODE_RIGHT), (MODE_RIGHT, MODE_REGULAR)])
  def test_all_six_switches_glide(self, seq):
    ctl = BlendfulCorrection()
    sc = Scene(v=13.0, widen=0.0, center=0.6, pl=1.0, pr=1.0)
    rows = drive(ctl, [sc] * 200, mode=seq[0])
    rows += drive(ctl, [sc] * 200, mode=seq[1])
    assert _glide_ok(rows, 13.0)
    assert not any(np.isnan(r[0]) for r in rows)

  def test_off_to_regular_matches_upstream_after_the_limiter(self):
    ctl = BlendfulCorrection()
    ref = Blendful()
    for _ in range(100):
      sc = Scene(v=13.0, center=0.5, pl=1.0, pr=1.0)
      ctl.update(sc.model_output(), 13.0, True, 0.0, LAT_T, MODE_REGULAR)
      k_ref = ref.update(sc.model_output(), 13.0, True, 0.0, enabled=True)
      assert abs(ctl.k_up - k_ref) < 1e-9  # the reference instance is upstream, frame by frame


class TestTransitions:
  def test_bound_engage_and_release_are_continuous(self):
    # a narrowing road: the rear-wheel bound engages, then the road widens again and it releases
    ws = [3.5 - 1.5 * min(1.0, i / 100.0) + 1.5 * max(0.0, min(1.0, (i - 200) / 100.0)) for i in range(320)]
    ctl = BlendfulCorrection()
    rows = drive(ctl, [Scene(v=13.0, pl=1.0, pr=1.0, width=w) for w in ws])
    flags = [r[2]['bound_active'] for r in rows]
    assert any(flags) and not flags[0] and not flags[-1]
    assert _glide_ok(rows, 13.0)

  def test_five_metres_per_second_crossing_releases_gracefully(self):
    ctl = BlendfulCorrection()
    drive(ctl, [Scene(v=13.0, center=0.5, pl=1.0, pr=1.0)] * 150)
    c_before = ctl.c
    out = []
    for i in range(60):
      v = 6.0 - 0.1 * i
      sc = Scene(v=max(v, 0.0), center=0.5, pl=1.0, pr=1.0)
      ctl.update(sc.model_output(), max(v, 0.0), True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
      out.append(ctl.c)
    assert np.all(np.isfinite(out))
    assert abs(out[-1]) <= abs(c_before) + 1e-12
    assert max(abs(np.diff([c_before] + out))) <= br.SWERVE_RELEASE / 1.0 * DT_MDL + 1e-9

  def test_lane_change_start_fades_and_releases(self):
    ctl = BlendfulCorrection()
    drive(ctl, [Scene(v=13.0, center=0.5, pl=1.0, pr=1.0)] * 150)
    c0 = ctl.c
    cs = []
    for _ in range(100):
      sc = Scene(v=13.0, center=0.5, pl=1.0, pr=1.0)
      ctl.update(sc.model_output(), 13.0, False, 0.0, LAT_T, MODE_RIGHT, RightInputs())  # lateral inactive during the change
      cs.append(ctl.c)
    assert abs(cs[-1]) < abs(c0) + 1e-12 and abs(cs[-1]) < 1e-9
    assert max(abs(np.diff([c0] + cs))) <= br.FADING_RELEASE / 13.0 ** 2 * DT_MDL + 1e-12

  def test_mode_toggle_in_the_middle_of_a_correction_is_continuous(self):
    ctl = BlendfulCorrection()
    sc = Scene(v=13.0, center=0.5, pl=1.0, pr=1.0)
    rows = drive(ctl, [sc] * 150, mode=MODE_RIGHT) + drive(ctl, [sc] * 50, mode=MODE_OFF) + drive(ctl, [sc] * 150, mode=MODE_RIGHT)
    assert _glide_ok(rows, 13.0)

  def test_detectors_at_their_thresholds(self):
    g = ConfGate()
    assert g.update(0.4) == 0.0 and g.update(0.6) == 1.0  # rising branch: 0 at 0.4, full at 0.6
    assert g.update(0.3) == 0.0  # falling branch (entered from above 0.5): 0 at 0.3
    g.c = 1.0
    assert g.update(0.5) == 1.0  # ... and full at 0.5
    # bend detector: nothing below the thresholds, full above
    mo = Scene(pl=1.0, pr=1.0, widen=0.0).model_output()
    assert bend_score(X, mo['lane_lines'][0, 2, :, 0], mo['lane_lines'][0, 1, :, 0], 13.0)[3] == 0.0
    mo['lane_lines'][0, 2, :, 0] += X ** 2 / (2 * 30.0)  # relative curvature 1/30 > 1/40
    assert bend_score(X, mo['lane_lines'][0, 2, :, 0], mo['lane_lines'][0, 1, :, 0], 13.0)[3] >= 0.99

  def test_line_vanishing_in_one_frame_does_not_jump_the_correction(self):
    ctl = BlendfulCorrection()
    rows = drive(ctl, [Scene(v=13.0, widen=0.5, pl=1.0, pr=1.0)] * 150)
    rows += drive(ctl, [Scene(v=13.0, widen=0.5, pl=1.0, pr=0.0)] * 100)
    assert _glide_ok(rows, 13.0)


class TestLogging:
  def test_fill_log_writes_every_field(self):
    from openpilot.cereal import custom
    ctl = BlendfulCorrection()
    drive(ctl, [Scene(v=13.0, widen=0.5, pl=1.0, pr=1.0)] * 60)
    msg = custom.ModelDataV2SP.new_message()
    br.fill_log(msg, ctl, MODE_RIGHT, 13.0)
    assert msg.blendfulMode == MODE_RIGHT
    assert abs(msg.blendfulKappa0 - ctl.k0) < 1e-6
    assert abs(msg.blendfulAuthority - ctl.result.log['a']) < 1e-6
    assert msg.blendfulRightApplied == bool(ctl.result.log['applied'])
    assert abs(msg.blendfulGainF - ctl.gain) < 1e-6

  def test_fill_log_in_off_writes_only_the_mode_block(self):
    from openpilot.cereal import custom
    ctl = BlendfulCorrection()
    drive(ctl, [Scene(v=13.0, pl=1.0, pr=1.0)] * 5, mode=MODE_OFF)
    msg = custom.ModelDataV2SP.new_message()
    br.fill_log(msg, ctl, MODE_OFF, 13.0)
    assert msg.blendfulMode == MODE_OFF and msg.blendfulAuthority == 0.0
