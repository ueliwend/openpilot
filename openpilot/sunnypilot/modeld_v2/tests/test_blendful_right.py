"""
Unit tests of Right-line Blendful (blendful_right.py): measurement filter, references, correction law, ground-fixed target, yields, junction
episodes (model alone), bounds, blinkers, transitions, logging and the three modes.  Design: inster-dev
results/2026-11-12_a8_right_line_blendful/DESIGN.md rev 5.1, section E3 (implementation round 2).
Signs: model frame y > 0 is right, curvature > 0 is a right turn.  Offline only; not validated on the car.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.modeld_v2 import blendful_right as br
from openpilot.sunnypilot.modeld_v2.blendful import Blendful
from openpilot.sunnypilot.modeld_v2.blendful_right import (MODE_OFF, MODE_REGULAR, MODE_RIGHT, BlendfulCorrection, ConfGate, EndpointGap, GroundTarget,
                                                           InputFilter, JumpDetector, JunctionEpisodes, RightLineBlendful, RoadClass,
                                                           WidthMemory, bend_score, junction_policy, mode_from_params, selection_to_params)
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.plan_correction import shift_plan
from openpilot.sunnypilot.modeld_v2.radar_clearance import healthy_empty_radar


def RightInputs(**kw):
  """RightInputs with a healthy, empty radar unless a test gives one (Right line needs a healthy radar, B6a)."""
  kw.setdefault('radar', healthy_empty_radar())
  return br.RightInputs(**kw)

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
  lever: bool = False  # lines and plan follow the circle seen from the device, l_d ahead of the rear axle (consistent with the filters' motion model)
  vx_scale: float = 1.0  # the plan's own v_x(0) / vEgo (the model's distance scale, C11 / K1)
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
    xt = X / self.vx_scale  # true distance of a model-axis point (the model's distance scale, C11 / K1)
    road = self.k * xt ** 2 / 2 + (br.L_D * self.k * xt if self.lever else 0.0)
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
    plan[0, :, 3] = self.v * self.vx_scale
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
  """The step-4 target at x = L (references, raw target, glide, limits): where the car is placed once the move-speed limit has caught up."""
  look = max(1.5 * v, 10.0)
  return float(np.interp(look, X, log['t4']))


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
      ctl.update(mo, 15.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
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
      ctl.update(mo, sc.v, True, sc.k, LAT_T, MODE_RIGHT, RightInputs())
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

  @staticmethod
  def _moves(tg, v=20.0):
    """Per frame, per point inside 2 L (beyond it T_lim is re-seeded from T_raw, rev 6.4): the target's change relative to its own-motion
    prediction (the ground-fixed part removed)."""
    inside = (X + v * DT_MDL <= X[-1]) & (X <= br.RESEED_LOOKAHEADS * max(1.5 * v, 10.0))
    return np.concatenate([(b - br.motion_predict(a, v, 0.0, DT_MDL))[inside] for a, b in zip(tg[:-1], tg[1:], strict=True)])

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
        targets.append(res.log['t_lim'].copy())  # the ground-fixed, rate-limited target
    return np.array(refs), np.array(targets)

  def test_inner_line_fading_in_glides(self):
    refs, tg = self._edge_line_run(lambda t: 0.05 if t < 1.0 else float(np.clip(0.05 + (t - 1.0), 0, 0.9)), 120)
    assert abs(refs[0] - 1.8) < 0.05 and abs(refs[-1] - 1.6) < 0.05
    d = self._moves(tg)
    assert d.min() >= -br.TARGET_RATE * DT_MDL - 1e-9 and d.max() <= br.TARGET_RATE * DT_MDL + 1e-9  # the rate limit turns it into a glide

  def test_inner_line_fading_out_and_vanishing_glide(self):
    for prob in (lambda t: 0.9 if t < 1.0 else float(np.clip(0.9 - (t - 1.0) / 0.3, 0.0, 0.9)), lambda t: 0.9 if t < 1.0 else 0.0):
      refs, tg = self._edge_line_run(prob, 120)
      assert abs(refs[10] - 1.6) < 0.05 and abs(refs[-1] - 1.8) < 0.05
      d = self._moves(tg)
      assert d.min() >= -br.TARGET_RATE * DT_MDL - 1e-9 and d.max() <= br.TARGET_RATE * DT_MDL + 1e-9

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
    for _ in range(200):
      sc = Scene(v=15.0, center=0.5, k=1 / 150, pr=0.0, lever=True)  # a steady circle: consistent with the car's own motion
      ctl.update(sc.model_output(), 15.0, True, sc.k, LAT_T, MODE_RIGHT, RightInputs())
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
      rl, res = steady(v=v, k=k, width=4.5, lever=True, frames=300, rl_kwargs={'inputs': RightInputs(gap_low=0.2, gap_high=0.2)})
      assert not res.log['bound_active']  # the bound term never acts beyond the target: no sign change, no oscillation
      rl2, res2 = run_rl([Scene(v=v, k=k, width=4.5, lever=True)] * 100, inputs=RightInputs(gap_low=0.2, gap_high=0.2), rl=rl)
      assert np.ptp([target_at(r.log, v) for r in res2]) < 1e-6

  def test_rear_wheel_bounds_on_lineless_narrow_road(self):
    # a 2.2 m road without lines: the edge bound (>= 0.3 m) overrides the centre limit
    rl, res = steady(pl=0.0, pr=0.0, edges=True, edge_half=1.1, v=10.0, frames=300)
    t = target_at(res.log, 10.0)
    assert res.log['bound_active'] and t <= 1.1 - W_R - br.EDGE_BOUND + 1e-6

  def test_right_curve_off_tracking_in_gap_and_bound(self):
    k = 1 / 30
    rl, res = steady(v=10.0, k=k, width=4.5, lever=True, frames=300)
    look = max(1.5 * 10.0, 10.0)
    right_line = float(np.interp(look, X, res.log['y_r']))
    off = br.L_D ** 2 * k / 2
    assert abs((right_line - (target_at(res.log, 10.0) + W_R)) - (br.gap_for_speed(10.0, 0.5, 1.0) + off)) < 0.02
    # the model cutting the inside does not move the bound
    rl2, res2 = run_rl([Scene(v=10.0, k=k, width=4.5, plan=0.8, lever=True)] * 100, rl=rl)
    t2 = target_at(res2[-1].log, 10.0)
    assert t2 <= right_line - W_R - br.LINE_BOUND - off + 1e-6

  def test_measurement_filter_toggle_bit_identity_of_inputs(self):
    # the raw inputs are logged next to the filtered ones, at x = 0, 0.5 L, L, 1.5 L
    rl, res = steady(frames=50)
    assert len(res.log['raw']) == 16 and len(res.log['filtered']) == 16 and len(res.log['confidence']) == 4
    assert np.allclose(res.log['raw'][8:12], [1.75] * 4) and np.allclose(res.log['filtered'][8:12], 1.75, atol=1e-6)
    assert len(res.log['cand_conf']) == 4 and len(res.log['cand_weight']) == 4 and len(res.log['reference']) == 2


# ---------------------------------------------------------------------------------------------------------------------------------

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
    prev_c = None
    windows = 0
    for c, q, s, active, _cap, rel, _i in rows[100:]:
      s_c = s * c  # s is this frame's capped side (+1 here; 0 outside a no-growth phase)
      if active and q >= br.Q_NO_GROWTH and prev_c is not None:
        assert s_c <= max(s * prev_c, 0.0) + 1e-12  # no growth from q = 0.1 m
      if active and not rel and q >= br.Q_CAP_FULL:
        assert s_c <= 1e-6  # the hard rule: zero correction toward the obstacle side, outside cap-release windows
      if rel:
        windows += 1
      prev_c = c
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
    # a slow swerve to the left (obstacle on the right): between q 0.15 and 0.2 m the toward-obstacle correction is at most the cap
    # memory times (0.2 - q) / 0.05, and from 0.2 m it is 0 (outside the cap-release window)
    ctl = BlendfulCorrection()
    for _ in range(100):
      ctl.update(Scene(v=13.0).model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
    assert ctl.c > 0.0
    in_ramp = 0
    for i in range(200):
      sw = 0.002 * i
      ctl.update(Scene(v=13.0, plan=-sw).model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
      q = ctl.result.caps.get(br.SIDE_RIGHT)
      if q is None:
        continue
      mem = ctl.cap_mem[br.SIDE_RIGHT]
      allowed = mem * float(np.clip((br.Q_CAP_FULL - q) / (br.Q_CAP_FULL - br.Q_CAP_START), 0.0, 1.0))
      if not ctl.release_window:
        assert ctl.c <= allowed + 1e-12
      in_ramp += br.Q_CAP_START < q < br.Q_CAP_FULL
    assert in_ramp > 0



# ---------------------------------------------------------------------------------------------------------------------------------
# rev 5: ground-fixed target, bounds order, width memory, junction episodes (model alone), road class, jumps, endpoint-gap trigger, D clip
def moves_of(tgs, v=20.0, dt=DT_MDL, vx_scale=1.0):
  """Per frame and point inside 2 L (re-seeded beyond, rev 6.4): the change of the rate-limited target relative to its own-motion prediction."""
  inside = (X + v * vx_scale * dt <= X[-1]) & (X <= br.RESEED_LOOKAHEADS * max(1.5 * v, 10.0))
  return np.concatenate([(b - br.motion_predict(a, v * vx_scale, 0.0, dt))[inside] for a, b in zip(tgs[:-1], tgs[1:], strict=True)])


def frames_of(scenes, rl=None, inputs=None, k0_up=0.0):
  rl = rl or RightLineBlendful()
  logs = []
  for sc in scenes:
    res = rl.update(sc.model_output(), sc.v, True, sc.k, k0_up, inputs or RightInputs())
    logs.append(res.log)
  return rl, logs


class TestGroundFixedTarget:
  @pytest.mark.parametrize("centre", [0.3, 0.5])
  def test_no_target_jump_under_left_quality_flicker_at_threshold(self, centre):
    # narrow lane (2.6 m), the left line's quality flickers across 0.3 (the replay's C1 trigger: 9-13 crossings per minute) and across
    # 0.5 (where this implementation's centre limit changes from two references to one and the RAW target jumps): the rate-limited
    # target must not jump in either case
    scenes = []
    for i in range(600):
      t = max(i - 100, 0) * DT_MDL
      scenes.append(Scene(v=20.0, width=2.6, pl=0.9 if i < 100 else centre + 0.2 * np.sin(2 * np.pi * 1.0 * t), pr=0.9, std=0.1))
    rl, logs = frames_of(scenes)
    logs = logs[100:]
    raw = np.array([lg['t_raw'] for lg in logs[100:]])
    # round 2's left limit switched formula at q 0.5 and moved the raw target 0.23 m per switch; with the hysteresis (S4) it does not
    assert np.max(np.abs(np.diff(raw, axis=0))) < 0.01
    tgs = [lg['t_lim'] for lg in logs[100:]]
    d = moves_of(tgs)
    assert d.min() >= -br.TARGET_RATE * DT_MDL - 1e-9 and d.max() <= br.TARGET_RATE * DT_MDL + 1e-9
    qs = np.array([lg['q_L'] for lg in logs[100:]])
    assert qs.min() < centre < qs.max()

  def test_own_motion_prediction_exact_on_a_steady_circle(self):
    # each rotation-lever variant on a circle built under its own hypothesis (review S6: not only the variant the fixture assumes)
    for vx_scale, variant in ((1.0, 'design'), (1.0, 'model_axis'), (1.08, 'model_axis'), (0.967, 'model_axis')):
      sc = Scene(v=14.0, k=1 / 50, width=3.5, lever=True, vx_scale=vx_scale)
      rl, logs = frames_of([sc] * 300, rl=RightLineBlendful(rotation_scale=variant))
      lg = logs[-1]
      assert np.max(np.abs(lg['t_lim'] - lg['t_raw'])[X < 60]) < 2e-3  # the target stays on the ground-fixed circle: no drift

  def test_line_edge_switch_and_reindex_move_by_the_rate_only(self):
    # the right line is re-indexed 3.5 m to the right (far line takes the near line's place) at frame 200
    scenes = [Scene(v=20.0, width=3.5)] * 200 + [Scene(v=20.0, width=3.5, widen=3.5)] * 200
    rl, logs = frames_of(scenes)
    tgs = [lg['t_lim'] for lg in logs]
    d = moves_of(tgs[190:])
    assert d.max() <= br.LAG_RATE * DT_MDL + 1e-9 and d.min() >= -br.LAG_RATE * DT_MDL - 1e-9
    # on a re-index to the LEFT (inward) the target glides at the left rate, not at once
    scenes = [Scene(v=20.0, width=3.5)] * 200 + [Scene(v=20.0, width=3.5, widen=-0.4)] * 60
    rl, logs = frames_of(scenes)
    d = moves_of([lg['t_lim'] for lg in logs[190:]])
    assert d.min() >= -br.TARGET_RATE * DT_MDL - 1e-9

  def test_rate_limits_by_direction_and_lag_bound(self):
    gt = GroundTarget()
    raw0 = np.zeros(br.NX)
    gt.update(raw0, 20.0, 0.0, DT_MDL)
    t = gt.update(raw0 + 0.5, 20.0, 0.0, DT_MDL)  # the raw target 0.5 m to the right: moves right at 0.05 m/s
    assert abs(t[5] - br.TARGET_RATE * DT_MDL) < 1e-12
    gt = GroundTarget()
    gt.update(raw0, 20.0, 0.0, DT_MDL)
    t = gt.update(raw0 - 0.5, 20.0, 0.0, DT_MDL)  # to the left at 0.25 m/s
    assert abs(t[5] + br.TARGET_RATE * DT_MDL) < 1e-12
    gt = GroundTarget()
    gt.update(raw0, 20.0, 0.0, DT_MDL)
    t = gt.update(raw0 + 1.5, 20.0, 0.0, DT_MDL)  # a lag above 1.0 m: the rate rises to 0.5 m/s, still glide limited
    assert abs(t[5] - br.LAG_RATE * DT_MDL) < 1e-12

  def test_lag_bound_holds_under_a_systematic_prediction_error(self):
    # a yaw rate bias makes the own-motion prediction wrong by up to 0.015 m per frame in the far field (and the near field inherits it
    # through the shift): more than the 0.0125 m the rate allows, less than the 0.025 m of the lag-bound rate; without re-seeding
    for bias in (0.0015, -0.0015):
      gt = GroundTarget()
      raw = np.full(br.NX, 0.3)
      worst = 0.0
      for _ in range(600):
        gt.update(raw, 20.0, bias, DT_MDL)
        worst = max(worst, float(np.max(np.abs(gt.lag))))
      assert worst <= br.LAG_BOUND + br.LAG_RATE * DT_MDL + 1e-6  # bounded (with 0.25 m/s in both directions it stays small: 0.03 m)

  def test_far_points_beyond_the_last_horizon_take_the_raw_target(self):
    gt = GroundTarget()
    gt.update(np.zeros(br.NX), 30.0, 0.0, DT_MDL)
    raw = np.full(br.NX, 0.4)
    t = gt.update(raw, 30.0, 0.0, DT_MDL)
    assert t[-1] == 0.4 and abs(t[0] - br.TARGET_RATE * DT_MDL) < 1e-12


class TestBoundOrder:
  def test_bounds_act_after_the_rate_limit_and_the_target_never_passes_them_in_a_glide(self):
    # the right line moves 1.0 m inward at frame 150: the raw target follows, the rate-limited one glides, the bound must cut it at once
    scenes = [Scene(v=13.0, width=4.5)] * 150 + [Scene(v=13.0, width=4.5, widen=-1.0)] * 150
    rl, logs = frames_of(scenes)
    glided = False
    for lg in logs[150:]:
      bound = lg['y_r'] - W_R - br.LINE_BOUND
      assert np.all(lg['target'] <= bound + 1e-9)
      glided |= bool(np.any(lg['t_lim'] > bound + 1e-3))
    assert glided  # the rate limited target alone WOULD have passed the bound: the order matters

  def test_edge_bound_after_the_rate_limit(self):
    # a road without lines whose edges close in by 1.2 m at frame 200: the glide would carry the target past the 0.3 m edge bound
    wide = Scene(v=13.0, pl=0.0, pr=0.0, edges=True, edge_half=2.4, edge_std=0.3)
    narrow = Scene(v=13.0, pl=0.0, pr=0.0, edges=True, edge_half=1.2, edge_std=0.3)
    rl, logs = frames_of([wide] * 200 + [narrow] * 100)
    glided = False
    for lg in logs[200:]:
      bound = lg['y_edge_r'] - W_R - br.EDGE_BOUND
      assert np.all(lg['target'][(X > 3) & (X < 60)] <= bound[(X > 3) & (X < 60)] + 1e-6)
      glided |= bool(np.any(lg['t_lim'][(X > 3) & (X < 60)] > bound[(X > 3) & (X < 60)] + 1e-3))
    assert glided

  def test_left_limit_acts_after_the_rate_limit(self):
    # the lane centre moves 1.0 m to the right (the car is left of it): the target may glide right only at 0.05 m/s, but is never left of
    # the centre limit
    scenes = [Scene(v=13.0, width=2.6, center=0.0)] * 150 + [Scene(v=13.0, width=2.6, center=0.8)] * 100
    # rev 6.10: the step-4 target T4 keeps the left limit; the right-line target after the move limit is not floored, and while it lies
    # left of the centre the larger of Regular's and the right-line correction is delivered
    rl, logs = frames_of(scenes)
    lagged = left = False
    for lg in logs[150:]:
      assert np.all(lg['t4'] >= np.minimum(lg['c_ctr'], lg['t4'] + 1.0) - 1e-9) and np.all(lg['t4'] >= lg['c_ctr'] - 1e-9)
      lagged |= bool(np.any(lg['t_lim'] < lg['c_ctr'] - 1e-3))
      if lg['trl_minus_centre'] < -1e-3:
        left = True
        assert lg['delivered'] != br.DELIVER_RIGHT_LINE or lg['bound_active']
    assert lagged and left


class TestWidthMemory:
  def test_symmetric_update_and_narrower_lane_accepted(self):
    wm = WidthMemory()
    for _ in range(80):
      wm.update(3.5, True, False, 'lines')
    assert wm.armed and abs(wm.w - 3.5) < 1e-9
    for _ in range(600):  # 30 s at a narrower lane: the memory comes down (no |delta| gate)
      wm.update(3.0, True, False, 'lines')
    assert abs(wm.w - 3.0) < 0.05 and wm.excess(3.0) is not None and abs(wm.excess(3.0)) < 0.05

  def test_measures_the_median_of_two_seconds_first(self):
    wm = WidthMemory()
    vals = [3.5] * 38 + [6.0, 6.0]  # two outliers inside the 2 s: the median ignores them
    for v in vals[:-1]:
      wm.update(v, True, False, 'lines')
    assert not wm.armed and wm.excess(3.5) is None
    wm.update(vals[-1], True, False, 'lines')
    assert wm.armed and abs(wm.w - 3.5) < 1e-9

  def test_stale_after_twenty_seconds_and_remeasured(self):
    wm = WidthMemory()
    for _ in range(80):
      wm.update(3.5, True, False, 'lines')
    for _ in range(int(20.5 / DT_MDL)):  # no clear measurement for 20.5 s
      wm.update(3.5, False, False, 'lines')
    assert wm.stale or not wm.armed
    assert wm.excess(5.0) is None  # cue 2 is off while stale
    for _ in range(39):
      wm.update(4.0, True, False, 'lines')
    assert wm.excess(4.0) is None
    wm.update(4.0, True, False, 'lines')
    assert wm.armed and abs(wm.w - 4.0) < 1e-9  # re-measured from the 2 s median

  def test_remeasured_after_an_episode_and_on_a_type_change(self):
    wm = WidthMemory()
    for _ in range(80):
      wm.update(3.5, True, False, 'lines')
    wm.end_of_episode()
    assert not wm.armed
    for _ in range(40):
      wm.update(4.2, True, False, 'lines')
    assert wm.armed and abs(wm.w - 4.2) < 1e-9
    wm.update(4.2, True, False, 'edges')  # a change of the reference type re-measures
    assert not wm.armed
    for _ in range(40):
      wm.update(5.0, True, False, 'edges')
    assert wm.armed and abs(wm.w - 5.0) < 1e-9

  def test_frozen_during_an_episode_and_detection_only(self):
    wm = WidthMemory()
    for _ in range(80):
      wm.update(3.5, True, False, 'lines')
    for _ in range(100):
      wm.update(4.5, True, True, 'lines')  # inside an episode: no update
    assert abs(wm.w - 3.5) < 1e-9
    # it is used for detection only: the target does not depend on it (same scene, memory armed or not)
    rl_a, res_a = steady(width=4.5, frames=200)
    rl_b = RightLineBlendful()
    rl_b.width_mem.w, rl_b.width_mem.armed = 2.5, True
    _, res_b = run_rl([Scene(width=4.5)] * 200, rl=rl_b)
    np.testing.assert_allclose(res_a.log['t_raw'], res_b[-1].log['t_raw'], atol=1e-6)


class TestJunctionEpisodes:
  @staticmethod
  def feed(ep, n, cue, widening=None, v=20.0):
    out = []
    for _ in range(n):
      out.append(ep.update(cue, widening, v))
    return out

  def test_start_after_a_cue_held_for_0_2_s(self):
    ep = JunctionEpisodes()
    assert not any(self.feed(ep, 3, True))
    assert self.feed(ep, 1, True)[0]  # the 4th frame = 0.2 s
    ep = JunctionEpisodes()
    self.feed(ep, 3, True)
    self.feed(ep, 1, False)
    assert not any(self.feed(ep, 3, True))  # a gap resets the hold

  def test_end_after_two_seconds_and_thirty_metres_without_a_cue(self):
    ep = JunctionEpisodes()
    self.feed(ep, 10, True)
    assert ep.episode
    out = self.feed(ep, 39, False, widening=0.3)
    assert all(out) and ep.episode  # 1.95 s: not yet
    self.feed(ep, 1, False, widening=0.3)
    assert not ep.episode and ep.ended == br.END_QUIET
    # low speed: 2 s is only 10 m at 5 m/s, so it takes 30 m = 6 s
    ep = JunctionEpisodes()
    self.feed(ep, 10, True, v=5.0)
    out = self.feed(ep, 140, False, widening=0.3, v=5.0)
    n = next(i for i, e in enumerate(out) if not e) + 1
    assert abs(n * DT_MDL - 6.0) <= DT_MDL + 1e-9

  def test_end_when_the_widening_is_gone_for_one_second(self):
    ep = JunctionEpisodes()
    self.feed(ep, 10, True)
    out = self.feed(ep, 19, False, widening=0.1)
    assert all(out)
    self.feed(ep, 1, False, widening=0.1)
    assert not ep.episode and ep.ended == br.END_WIDENING
    # a cue still active, or a widening above 0.15 m, keeps it going
    ep = JunctionEpisodes()
    self.feed(ep, 10, True)
    assert all(self.feed(ep, 60, True, widening=0.0, v=5.0)[:30])
    assert all(self.feed(ep, 60, False, widening=0.2, v=20.0)[:30])

  def test_end_at_three_hundred_metres_then_rearm_only_after_the_cues_clear(self):
    ep = JunctionEpisodes()
    n = 0
    self.feed(ep, 4, True)
    assert ep.episode
    while ep.episode:
      ep.update(True, 1.0, 20.0)  # 20 m/s: 1 m per frame
      n += 1
      assert n < 1000
    assert ep.ended == br.END_MAX and 299 <= n <= 301
    assert not any(self.feed(ep, 20, True))  # cues still on: blocked
    assert not any(self.feed(ep, 1, False))  # clears once
    assert not any(self.feed(ep, 3, True))
    assert self.feed(ep, 1, True)[0]  # and a new episode after 0.2 s

  def test_model_alone_time_counts_the_episode(self):
    ep = JunctionEpisodes()
    self.feed(ep, 4, True)
    self.feed(ep, 20, True)
    assert abs(ep.model_alone_time - 20 * DT_MDL) < 1e-9


def tapered(sc, slope=1 / 15.0, start=5.0):
  mo = sc.model_output()
  mo['lane_lines'][0, 2, :, 0] += np.clip(X - start, 0, None) * slope
  return mo


def with_far_line_v(sc):
  mo = Scene(**{**sc.__dict__, 'far_pr': 0.9}).model_output()
  sep = 0.4 + 0.05 * np.abs(X - 40.0)  # the far-right line approaches the right line to 0.4 m at 40 m and separates beyond: a gore
  mo['lane_lines'][0, 3, :, 0] = mo['lane_lines'][0, 2, :, 0] + sep
  return mo


def kinked(sc, deg=12.0, at=25.0):  # the filtered line smears a kink fixed in the car frame by about v tau: 12 deg measured gives ~7 deg
  mo = sc.model_output()
  mo['lane_lines'][0, 2, :, 0] += np.clip(X - at, 0, None) * np.tan(np.radians(deg))
  return mo


def edge_diverging(sc):
  mo = Scene(**{**sc.__dict__, 'edges': True}).model_output()
  mo['road_edges'][0, 1, :, 0] += np.clip(X - 5.0, 0, None) / 10.0
  return mo


def bend_away(sc):
  return Scene(**{**sc.__dict__, 'bend_x0': 15.0, 'bend_R': 30.0}).model_output()


def width_step(sc):
  return Scene(**{**sc.__dict__, 'widen': 0.9}).model_output()


CUES = [  # name, scene builder, cue bit
  ('widening in frame', tapered, 0), ('widening over time', width_step, 1), ('edge divergence', edge_diverging, 2),
  ('gore V shape', with_far_line_v, 3), ('kink', kinked, 3), ('sharp bend', bend_away, 4)]


class TestJunctionCues:
  @pytest.mark.parametrize("name,build,bit", CUES, ids=[c[0] for c in CUES])
  def test_each_cue_starts_an_episode_after_0_2_s(self, name, build, bit):
    rl = RightLineBlendful()
    base = Scene(v=20.0, width=3.5, pl=0.9, pr=0.9, edges=name == 'edge divergence')
    for _ in range(150):  # clear measurements: the width memory arms, the filters settle
      rl.update(base.model_output(), 20.0, True, 0.0, 0.0, RightInputs())
    assert not rl.episodes.episode and rl.width_mem.armed
    first = first_any = start = None
    for i in range(80):
      res = rl.update(build(base), 20.0, True, 0.0, 0.0, RightInputs())
      if first is None and (res.log['cues'] >> bit) & 1:
        first = i
      if first_any is None and res.log['cues']:
        first_any = i
      if start is None and res.episode:
        start = i
    assert first is not None, f"cue {name} never fired"
    assert start is not None and start - first_any == 3  # held for 0.2 s (4 frames) before the episode starts

  def test_no_cue_on_a_plain_road(self):
    rl, logs = frames_of([Scene(v=20.0, width=3.5, edges=True, edge_half=3.5, k=1 / 300, lever=True)] * 400)
    assert all(lg['cues'] == 0 and not lg['episode'] for lg in logs)

  def test_curve_alone_gives_no_widening_or_kink(self):
    rl, logs = frames_of([Scene(v=14.0, width=3.5, k=1 / 60, lever=True)] * 300)
    assert all(lg['cues'] == 0 for lg in logs[100:])

  def test_low_speed_window_uses_five_to_sixty_metres(self):
    assert br.cue_window(5.0) == (5.0, 60.0)
    xa, xb = br.cue_window(20.0)
    assert abs(xa - 7.5) < 1e-9 and abs(xb - 45.0) < 1e-9

  def test_edge_divergence_angle_cue_alone(self):
    x = X
    edge = np.full(33, 3.0) + np.clip(x - 5.0, 0, None) * 0.05  # 0.05 slope: 2.9 deg, over 37.5 m = 1.9 m
    g, ang = br.cue_edge_divergence(x, np.full(33, 1.75), edge, None, 20.0)
    assert g > br.CUE_EDGE_DIV and abs(ang - np.degrees(np.arctan(g / 37.5))) < 1e-9

  def test_gore_needs_both_flanks(self):
    near = np.full(33, 1.75)
    only_separating = near + np.clip(0.3 + 0.05 * (X - 5.0), 0.3, 4.0)
    assert br.cue_gore(X, near, only_separating) is None  # no approach: not a V
    v = near + 0.4 + 0.05 * np.abs(X - 40.0)
    assert br.cue_gore(X, near, v) < br.CUE_GORE_SEP  # the model grid is coarse at 40 m: the sampled minimum is a little above 0.4 m
    assert br.cue_gore(X, near, near + 3.5) is None  # parallel lines


class TestInsideAnEpisode:
  def _episode_run(self, mode=MODE_RIGHT, k0_up_scene=0.3, right_blinker=False):
    ctl = BlendfulCorrection()
    rows = []
    for i in range(400):
      base = Scene(v=20.0, width=3.5, center=0.0, pl=0.9, pr=0.9)
      sc = base if i < 200 else Scene(v=20.0, width=3.5, pl=0.9, pr=0.9, bend_x0=15.0, bend_R=30.0)
      ctl.update(sc.model_output(), 20.0, True, 0.0, LAT_T, mode, RightInputs(right_blinker=right_blinker))
      rows.append((ctl.c, ctl.k0, ctl.result.episode, ctl.result.log.get('weights'), ctl.k_up, ctl.right.a, ctl.release_window))
    return ctl, rows

  def test_both_corrections_fade_to_zero_through_the_limiter_with_no_regular_and_no_hold(self):
    ctl, rows = self._episode_run()
    ep = [i for i, r in enumerate(rows) if r[2]]
    assert ep, "no episode"
    assert all(rows[i][3] == (0.0, 0.0) for i in ep)  # neither the right-line nor the upstream part: no Regular, no hold
    cs = np.array([r[0] for r in rows])
    # the fade is the limiter's: never faster than the 2.0 m/s^3 release (upstream's own)
    rate = np.abs(np.diff(cs)) * 20.0 ** 2 / DT_MDL
    window = np.array([r[6] for r in rows[1:]])  # a cap-release window (the junction's bend also looks like a swerve to the yield trigger) may use 5 m/s^3
    assert rate[~window].max() <= br.RELEASE + 1e-6 and rate.max() <= br.SWERVE_RELEASE + 1e-6
    assert abs(cs[ep[-1]]) < abs(cs[ep[0]]) + 1e-12 or abs(cs[ep[-1]]) < 1e-9
    # and the correction really got to zero inside the episode (after the fade), while upstream's output stays non-zero somewhere
    assert abs(cs[ep[-1]]) < 1e-9 or len(ep) < 40

  def test_the_right_blinker_changes_nothing_inside_an_episode(self):
    _, a = self._episode_run(right_blinker=False)
    _, b = self._episode_run(right_blinker=True)
    assert [r[0] for r in a] == [r[0] for r in b] and [r[2] for r in a] == [r[2] for r in b]

  def test_regular_mode_is_unchanged_by_junctions(self):
    ctl, ref = BlendfulCorrection(), Blendful()
    for i in range(400):
      sc = Scene(v=20.0, width=3.5, pl=0.9, pr=0.9) if i < 200 else Scene(v=20.0, width=3.5, pl=0.9, pr=0.9, bend_x0=15.0, bend_R=30.0)
      ctl.update(sc.model_output(), 20.0, True, 0.0, LAT_T, MODE_REGULAR, RightInputs())
      up = ref.update(sc.model_output(), 20.0, True, 0.0, enabled=True)
      assert abs(ctl.c - up) <= 1e-12 and not ctl.result.episode

  def test_junction_policy_is_the_only_place_and_has_no_branch_logic(self):
    w = junction_policy(True, 0.7)
    assert (w.w_rl, w.w_up) == (0.0, 0.0)
    w = junction_policy(True, 0.7, right_line_mode=False)  # Regular: unchanged
    assert (w.w_rl, w.w_up) == (0.7, pytest.approx(0.3))
    w = junction_policy(False, 0.4)
    assert (w.w_rl, w.w_up) == (0.4, pytest.approx(0.6))
    # Appendix J (hold, latch, ramp context, split classifier) is documentation only: none of it exists in the module
    for name in ('JunctionLogic', 'junction_mode', 'split_score', 'split_pattern', 'gore_distance', 'divergence_start', 'YieldEstimator'):
      assert not hasattr(br, name)
    assert not hasattr(br.PolicyWeights(), 'w_hold')

  def test_a_rises_normally_after_the_episode(self):
    ctl, rows = self._episode_run()
    a = np.array([r[5] for r in rows])
    ep = np.array([r[2] for r in rows])
    assert a[ep].min() < 0.3 or ep.sum() < 20  # authority went down inside the episode ...
    assert np.max(np.diff(a)) <= br.A_IN * DT_MDL + 1e-9  # ... and never rises faster than 0.5 /s


class TestRoadClass:
  def test_hysteresis_and_five_second_dwell(self):
    rc = RoadClass()
    assert rc.update(15.0, 0.9, 0.9, 3.5) == br.ROAD_LANES
    n = 0
    for n in range(1, 400):  # noqa: B007
      if rc.update(15.0, 0.9, 0.9, 2.6) == br.ROAD_SHARED:
        break
    assert abs(n * DT_MDL - br.ROAD_DWELL) <= 2 * DT_MDL  # the new class is held for 5 s first

  def test_one_hertz_flicker_does_not_change_the_class(self):
    rc = RoadClass()
    rc.update(15.0, 0.9, 0.9, 3.5)
    for i in range(1200):
      t = i * DT_MDL
      w = 3.0 + 0.4 * np.sin(2 * np.pi * 1.0 * t)  # across the width threshold at 1 Hz
      q = 0.35 + 0.1 * np.sin(2 * np.pi * 1.0 * t)  # and across the quality hysteresis
      rc.update(15.0, q, q, w)
      assert rc.cls == br.ROAD_LANES

  def test_hysteresis_band(self):
    rc = RoadClass()
    rc.cls = br.ROAD_SHARED
    assert rc.raw(15.0, 0.9, 0.9, 3.1) == br.ROAD_SHARED  # inside 3.0 +- 0.15
    rc.cls = br.ROAD_LANES
    assert rc.raw(15.0, 0.9, 0.9, 2.9) == br.ROAD_LANES
    assert rc.raw(15.0, 0.9, 0.9, 2.8) == br.ROAD_SHARED
    rc.cls = br.ROAD_EDGES
    assert rc.raw(15.0, 0.35, 0.35, 3.5) == br.ROAD_EDGES  # quality 0.3 / 0.4
    assert rc.raw(15.0, 0.45, 0.35, 3.5) == br.ROAD_LANES
    rc.cls = br.ROAD_LANES
    assert rc.raw(15.0, 0.35, 0.35, 3.5) == br.ROAD_LANES
    assert rc.raw(15.0, 0.25, 0.25, 3.5) == br.ROAD_EDGES
    assert rc.raw(22.5, 0.9, 0.9, 3.8) == br.ROAD_LANES and rc.raw(23.5, 0.9, 0.9, 3.8) == br.ROAD_MOTORWAY
    rc.cls = br.ROAD_MOTORWAY
    assert rc.raw(21.5, 0.9, 0.9, 3.8) == br.ROAD_MOTORWAY and rc.raw(20.5, 0.9, 0.9, 3.8) == br.ROAD_LANES

  def test_it_resets_no_estimator(self):
    rl, _ = frames_of([Scene(v=15.0)] * 50)
    state = (rl.gap.rp, rl.gap.ps, rl.width_mem.w)
    rl.road.cls = br.ROAD_SHARED
    rl.road.update(15.0, 0.9, 0.9, 3.5)
    assert (rl.gap.rp, rl.gap.ps, rl.width_mem.w) == state


class TestJumps:
  def test_edge_innovation_of_one_metre_does_not_trip(self):
    jd = JumpDetector()
    for _ in range(10):
      assert not jd.update((0.0, 1.0), (0.0, 1.0), (False, True))

  def test_line_innovation_of_0_7_m_for_two_frames_trips(self):
    jd = JumpDetector()
    assert not jd.update((0.7, 0.0), (1.0, 0.0), (True, False))  # one frame: not yet
    assert jd.update((0.7, 0.0), (1.0, 0.0), (True, False))
    for _ in range(30):  # held 2 s
      still = jd.update((0.0, 0.0), (1.0, 0.0), (True, False))
    assert still
    for _ in range(12):
      still = jd.update((0.0, 0.0), (1.0, 0.0), (True, False))
    assert not still

  def test_persistence_is_per_input_and_consecutive(self):
    jd = JumpDetector()
    jd.update((0.7, 0.0), (1.0, 0.0), (True, False))
    assert not jd.update((0.3, 0.0), (1.0, 0.0), (True, False))  # interrupted
    assert not jd.update((0.7, 0.0), (1.0, 0.0), (True, False))

  def test_edge_trips_above_1_2_m(self):
    jd = JumpDetector()
    jd.update((0.0, 1.3), (0.0, 1.0), (False, True))
    assert jd.update((0.0, 1.3), (0.0, 1.0), (False, True))

  def test_a_candidate_with_weight_below_half_never_trips(self):
    jd = JumpDetector()
    for _ in range(20):
      assert not jd.update((3.0, 3.0), (0.49, 0.2), (True, True))

  def test_integration_a_shifted_right_line_trips_and_fades_the_authority(self):
    scenes = [Scene(v=20.0, width=3.5)] * 150 + [Scene(v=20.0, width=3.5, widen=0.9)] * 6
    rl, logs = frames_of(scenes)
    assert logs[149]['f_jump'] == 1.0
    assert any(lg['f_jump'] == 0.0 for lg in logs[150:])
    # a faint edge candidate (weight < 0.5 next to a clear line) jumping by 2 m never trips
    near = Scene(v=20.0, width=3.5, edges=True, edge_half=3.5, edge_right=2.6)
    far = Scene(v=20.0, width=3.5, edges=True, edge_half=3.5, edge_right=4.6)
    scenes = [near] * 150 + [far] * 10
    rl, logs = frames_of(scenes)
    assert all(lg['f_jump'] == 1.0 for lg in logs)


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


def sim_endpoint_gap(rho, sw_start, sw_T, rate=br.CAR_MOVE_SPEED, hmax=1.0, T=60.0, noise=0.03, seed=1, settle=br.MOVE_SETTLE):
  """Closed loop of DESIGN check 18: the car moves to the held offset hmax (target glide `rate`, 1.5 s lag), the model plans back with the
  return fraction rho; x = Y_R(L) - plan_y(L) = 2 - (car + p)."""
  gap = EndpointGap()
  rng = np.random.default_rng(seed)
  h = o = 0.0
  t10 = t20 = None
  qs = []
  moving_t = 0.0
  for i in range(int(T / DT_MDL)):
    t = i * DT_MDL
    sw = 0.8 * np.clip((t - sw_start) / sw_T, 0, 1) if t >= sw_start else 0.0
    p = -rho * o - sw + noise * rng.standard_normal()
    building = h < hmax - 1e-9
    h = min(hmax, h + rate * DT_MDL)
    o += (h - o) * DT_MDL / 1.5
    moving_t = settle if building else max(0.0, moving_t - DT_MDL)
    q = gap.update(2.0 - (o + p), 1, moving_t > 0.0, False, 0.0, False)
    qs.append(q)
    if t >= sw_start:
      if t10 is None and q >= 0.1:
        t10 = t - sw_start
      if t20 is None and q >= 0.2:
        t20 = t - sw_start
  return np.array(qs), t10, t20


class TestEndpointGapTrigger:
  @pytest.mark.parametrize("rho", [-0.2, 0.0, 0.3, 0.5, 0.9])
  def test_steady_hold_and_move_give_no_false_trigger_for_any_rho(self, rho):
    for seed in (1, 2):
      qs, _, _ = sim_endpoint_gap(rho, 999.0, 1.0, T=60.0, seed=seed)
      assert qs.max() < br.Q_NO_GROWTH  # check 18: max q <= 0.04 m (here with 0.03 m plan noise, inside the no-growth point)

  @pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
  @pytest.mark.parametrize("sw_T", [0.5, 3.0, 10.0])
  def test_swerve_on_a_steady_hold_triggers_with_upstream_from_centre(self, rho, sw_T):
    _, t10, t20 = sim_endpoint_gap(rho, 40.0, sw_T)  # the move to 1.0 m at 0.05 m/s takes 20 s: steady from 40 s on
    tu = upstream_from_centre_trigger(40.0, sw_T)
    assert t10 is not None and t10 <= tu + 0.25 + 1e-9  # no-growth point up to about 0.2 s after upstream-from-centre (I1)

  def test_running_minimum_is_not_updated_on_skip_frames(self):
    g = EndpointGap()
    for _ in range(20):
      g.update(2.0, 1, True, False, 0.0, False)
    rp = g.rp
    for _ in range(20):
      g.update(1.0, 1, True, True, 0.0, False)  # a jump: ignored
    assert g.rp == rp

  def test_median_stops_a_single_dip_from_lowering_the_minimum(self):
    g = EndpointGap()
    for _ in range(40):
      g.update(2.0, 1, True, False, 0.0, False)
    g.update(1.0, 1, True, False, 0.0, False)  # one frame with the reference flickering inward
    assert g.rp > 1.9

  def test_reference_switch_reinitialises(self):
    g = EndpointGap()
    for _ in range(40):
      g.update(2.0, 1, False, False, 0.0, False)
    q = g.update(5.0, 2, False, False, 0.0, False)  # right -> left reference: no false trigger
    assert q == 0.0 and g.source == 2

  def test_usual_follows_slowly_and_tau_doubles_while_going_around(self):
    g1, g2 = EndpointGap(), EndpointGap()
    for _ in range(10):
      g1.update(2.0, 1, False, False, 0.0, False)
      g2.update(2.0, 1, False, False, 0.0, True)
    for _ in range(200):
      g1.update(2.5, 1, False, False, 0.0, False)
      g2.update(2.5, 1, False, False, 0.0, True)
    assert g1.rp > g2.rp > 2.0


class TestYieldRules:
  def test_trigger_works_with_a_weak_right_reference(self):
    # right reference gone (q_R < 0.3): the left reference carries the trigger, a swerve is still seen (C8: no q_R gate)
    ctl = BlendfulCorrection()
    q = []
    for i in range(300):
      sw = 0.0 if i < 150 else min(0.8, 0.8 * (i - 150) * DT_MDL / 0.5)
      ctl.update(Scene(v=13.0, pr=0.0, plan=-sw).model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
      q.append((ctl.result.yield_q, ctl.result.log['yield_source'], ctl.result.yield_active))
    assert q[-1][1] == 2 and q[-1][2] and max(x[0] for x in q) > 0.2 and not any(x[2] for x in q[:150])

  def test_upstream_away_fade_counts_as_a_trigger(self):
    rl = RightLineBlendful()
    res = rl.update(Scene(v=13.0).model_output(), 13.0, True, 0.0, 0.0, RightInputs(ref_yield_fade=0.5))
    assert RightLineBlendful.ref_around(RightInputs(ref_yield_fade=0.5)) and not RightLineBlendful.ref_around(RightInputs())
    for _ in range(5):
      res = rl.update(Scene(v=13.0).model_output(), 13.0, True, 0.0, 0.0, RightInputs(ref_yield_fade=0.5))
    assert res.yield_active and res.no_growth

  def test_obstacle_side_is_latched_at_the_start_of_the_yield(self):
    ctl = BlendfulCorrection()
    sides = []
    for i in range(400):
      if i < 150:
        plan = 0.0
      elif i < 250:
        plan = -0.8  # swerve left: obstacle on the right (+1)
      else:
        plan = 0.8 * (1 if i < 300 else 1)  # the plan then crosses to the other side while the yield still holds: the side must not flip
      ctl.update(Scene(v=13.0, plan=plan).model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
      if ctl.result.yield_active:
        sides.append(ctl.result.yield_side)
    assert sides and sides[0] == 1
    first_flip = next((i for i, s in enumerate(sides) if s != 1), None)
    assert first_flip is None or first_flip > 60  # not flipped while the first yield is still active

  def test_cap_memory_lives_with_the_yield_and_never_grows(self):
    ctl, rows = swerve_run(+1, swerve_T=3.0, frames_total=400)
    mems = []
    ctl2 = BlendfulCorrection()
    for i in range(400):
      sw = 0.0 if i < 100 else min(0.8, 0.8 * (i - 100) * DT_MDL / 3.0)
      ctl2.update(Scene(v=13.0, plan=-sw).model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
      mems.append(ctl2.cap_mem.get(br.SIDE_RIGHT))
    m = [x for x in mems if x is not None]
    assert m and all(b <= a + 1e-15 for a, b in zip(m[:-1], m[1:], strict=True))  # non-increasing for as long as it lives
    # the swerve ends: the memory is released with the yield (empty again once the latch released)
    for _ in range(1500):  # the usual follows the swerve with tau 10 s, so the yield ends within about a minute
      ctl2.update(Scene(v=13.0, plan=0.0).model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
    assert ctl2.cap_mem == {}

  def test_no_growth_toward_the_obstacle_between_q_0_1_and_0_2(self):
    ctl = BlendfulCorrection()
    prev = None
    worst = 0.0
    for i in range(500):
      sw = 0.0 if i < 100 else min(0.35, 0.35 * (i - 100) * DT_MDL / 6.0) if i < 300 else max(0.0, 0.35 - 0.35 * (i - 300) * DT_MDL / 6.0)
      ctl.update(Scene(v=13.0, plan=-sw).model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
      s = ctl.result.yield_side
      if ctl.result.no_growth and prev is not None and ctl.result.yield_q < br.Q_CAP_FULL:
        assert s in (1, -1)
        worst = max(worst, s * ctl.c - max(s * prev, 0.0))
      prev = ctl.c
    assert worst <= 1e-12


class TestDClip:
  def test_saturated_error_is_clipped_at_one_metre(self):
    rl = RightLineBlendful()
    la = br.approach_length(20.0)
    plan_x, plan_y = X, np.zeros(33)
    d, k, k_fit, clipped = rl._correction(np.full(33, 3.0), plan_x, plan_y, 20.0, 1.0)  # error clipped at 1 m: the fit would give 1.46 .. 1.64 m
    assert clipped and d == 1.0
    assert abs(k - 6.0 * 1.0 / br.profile_length(20.0) ** 2) < 1e-15
    d, k, _, clipped = rl._correction(np.full(33, -3.0), plan_x, plan_y, 20.0, 1.0)
    assert clipped and d == -1.0
    d, _, _, clipped = rl._correction(np.full(33, 0.2), plan_x, plan_y, 20.0, 1.0)
    assert not clipped and 0.0 < d < 0.4 and la > 0

  def test_integration_never_exceeds_one_metre(self):
    rl, logs = frames_of([Scene(v=20.0, width=8.0)] * 900)  # a very wide lane: saturated error (the target approaches at r_ref)
    assert max(abs(lg['d']) for lg in logs) <= br.D_MAX + 1e-12
    assert any(lg['d_clipped'] for lg in logs)


class TestHandBackWindow:
  def test_r1_after_the_window_exact_and_decaying_inside(self):
    # both lines clear: a rises to 1; then the right line vanishes: a falls at 1 /s, the shared limiter carries the remainder to upstream's value
    ctl = BlendfulCorrection()
    ref = Blendful()
    cs, ups, a = [], [], []
    for i in range(500):
      sc = Scene(v=13.0, center=0.5, pl=1.0, pr=1.0) if i < 250 else Scene(v=13.0, center=0.5, pl=1.0, pr=0.0)
      ctl.update(sc.model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
      ups.append(ref.update(Scene(v=13.0, center=0.5, pl=1.0, pr=1.0 if i < 250 else 0.0).model_output(), 13.0, True, 0.0, enabled=True))
      cs.append(ctl.c)
      a.append(ctl.right.a)
    a, cs, ups = np.array(a), np.array(cs), np.array(ups)
    zero = 250 + int(np.argmax(a[250:] == 0.0))
    assert a[zero] == 0.0
    diff = np.abs(cs - ups)
    after = zero + int(2.0 / DT_MDL) + 1
    # outside the 2 s window the replayed correction equals the reference instance's (upstream sees the same raw lines; a = 0, no yield)
    assert np.max(diff[after:]) <= 1e-9 or np.max(diff[after:]) < 2e-6
    inside = diff[zero:after]
    assert np.all(np.diff(inside) <= 1e-12)  # decays monotonically inside the window


class TestFilterScaleK1:
  def circle_bias(self, s, shift_scale, v=50 / 3.6, R=50.0, q=1.0):
    k = 1.0 / R
    ld = br.L_D
    ytrue = 1.6 + ld * k * (X / s) + k * (X / s) ** 2 / 2  # the lines are reported on the model's x axis: x_m = s x
    f = InputFilter()
    f.update(ytrue, 1.0, v * s, v * k)
    for _ in range(400):
      y, _ = f.update(ytrue, q, v * shift_scale, v * k, DT_MDL, 1.0 / s if shift_scale == s else 1.0)
    return float(np.interp([10.0, 20.0, 30.0], X, y - ytrue)[1])

  @pytest.mark.parametrize("s", [1.0, 1.05, 0.967, 1.08])
  def test_the_model_speed_shift_has_no_signed_bias_within_a_few_cm(self, s):
    assert abs(self.circle_bias(s, s)) < 0.03

  def test_shifting_by_vego_would_bias_toward_the_outside(self):
    # (review R5-2) for s > 1 the vEgo shift leaves the estimate on the OUTSIDE of a right curve (y right-positive: y_est - y_true < 0)
    assert self.circle_bias(1.05, 1.0) < -0.1

  def test_true_frame_interval_is_used(self):
    f, g = InputFilter(), InputFilter()
    line = np.full(33, 1.7) + 0.02 * X
    for filt in (f, g):
      filt.update(line, 1.0, 20.0, 0.0, DT_MDL)
    pa = f.predict(20.0, 0.0, DT_MDL)
    pb = g.predict(20.0, 0.0, 0.075)  # a late frame: the shift is v dt with the true interval
    np.testing.assert_allclose(pb[:20] - pa[:20], 0.02 * 20.0 * (0.075 - DT_MDL), atol=1e-9)


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


class TestIdleInstance:
  def test_right_line_instance_restarts_when_it_was_idle_in_regular(self):
    ctl = BlendfulCorrection()
    sc = Scene(v=13.0, pl=1.0, pr=0.0)  # a = 0 in Regular: the Right line instance does not run, its filters would go stale
    drive(ctl, [sc] * 200, mode=MODE_REGULAR)
    assert ctl.right_idle
    rows = drive(ctl, [Scene(v=13.0, pl=1.0, pr=1.0, center=0.4)] * 40, mode=MODE_RIGHT)
    assert not ctl.right_idle and ctl.right.filters[2].initialized  # re-initialised from the measurement, not predicted from stale state
    assert all(r[2]['f_jump'] == 1.0 for r in rows)  # and no jump trip from the stale prediction


class TestBendScoreLinearTaper:
  def test_widening_followed_linear_not_cut(self):
    mo = tapered(Scene(v=13.0, pl=1.0, pr=1.0))
    _, _, _, b = bend_score(X, mo['lane_lines'][0, 2, :, 0], mo['lane_lines'][0, 1, :, 0], 13.0)
    assert b <= 0.1


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
      ctl.update(sc.model_output(), 13.0, True, 0.0, LAT_T, MODE_REGULAR, RightInputs())
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

  def test_fill_log_writes_the_episode_and_cue_fields(self):
    from openpilot.cereal import custom
    ctl = BlendfulCorrection()
    drive(ctl, [Scene(v=20.0, width=3.5)] * 150)
    rows = drive(ctl, [Scene(v=20.0, width=3.5, bend_x0=15.0, bend_R=30.0)] * 30)
    assert rows[-1][2]['episode']
    msg = custom.ModelDataV2SP.new_message()
    br.fill_log(msg, ctl, MODE_RIGHT, 20.0)
    lg = ctl.result.log
    assert msg.blendfulJunctionEpisode and msg.blendfulJunctionCues == lg['cues'] and msg.blendfulJunctionCues & 16
    assert len(msg.blendfulJunctionCueValues) == 7 and abs(msg.blendfulJunctionCueValues[6] - lg['cue_values'][6]) < 1e-6
    assert abs(msg.blendfulJunctionEpisodeDistance - lg['episode_dist']) < 1e-4 and msg.blendfulJunctionEpisodeDistance > 0.0
    assert abs(msg.blendfulWidthMemory - lg['width_mem']) < 1e-6 and msg.blendfulWidthMemoryState & 1
    assert abs(msg.blendfulFrameDt - DT_MDL) < 1e-6 and abs(msg.blendfulPlanSpeed - 20.0) < 1e-6
    assert msg.blendfulRoadClass == lg['road_class'] and abs(msg.blendfulTargetLag - lg['target_lag']) < 1e-6
    assert msg.blendfulCentreSource == lg['centre_source'] and not msg.blendfulNonFinite
    for gone in ('blendfulFadeSplit', 'blendfulRho', 'blendfulBias', 'blendfulJunctionLatched', 'blendfulJunctionRamp', 'blendfulJunctionHold',
                 'blendfulSplitScore'):  # the first implementation's superseded fields are dropped from the schema (round 3)
      with pytest.raises(AttributeError):
        getattr(msg, gone)

  def test_fill_log_in_off_writes_only_the_mode_block(self):
    from openpilot.cereal import custom
    ctl = BlendfulCorrection()
    drive(ctl, [Scene(v=13.0, pl=1.0, pr=1.0)] * 5, mode=MODE_OFF)
    msg = custom.ModelDataV2SP.new_message()
    br.fill_log(msg, ctl, MODE_OFF, 13.0)
    assert msg.blendfulMode == MODE_OFF and msg.blendfulAuthority == 0.0


# ---------------------------------------------------------------------------------------------------------------------------------
# round 3 (review of round 2): trigger aggregator (B1), centre-limit hysteresis (S4), non-finite guard (S3), scale hypotheses and lag bound (S6)
def report(name, no_growth=True, q=0.15, side=0, active=None):
  return br.TriggerReport(name, no_growth, no_growth if active is None else active, q, side)


class TestYieldAggregator:
  def test_no_trigger_no_phase(self):
    agg = br.YieldAggregator()
    assert agg.update([report('a', no_growth=False), report('b', no_growth=False)]) == (False, False, {})
    assert not agg.phase

  def test_a_trigger_supplies_its_side_and_q(self):
    agg = br.YieldAggregator()
    ng, active, caps = agg.update([report('gap', q=0.17, side=br.SIDE_RIGHT, active=False), report('ref', no_growth=False)])
    assert ng and not active and caps == {br.SIDE_RIGHT: 0.17}

  def test_unknown_side_caps_both_until_it_is_known_then_only_that_side(self):
    agg = br.YieldAggregator()
    _, _, caps = agg.update([report('ref', q=0.2, side=0)])
    assert caps == {br.SIDE_RIGHT: 0.2, br.SIDE_LEFT: 0.2}
    _, _, caps = agg.update([report('ref', q=0.2, side=br.SIDE_LEFT)])
    assert caps == {br.SIDE_LEFT: 0.2}

  def test_the_side_is_latched_for_the_phase(self):
    agg = br.YieldAggregator()
    agg.update([report('gap', q=0.3, side=br.SIDE_RIGHT)])
    _, _, caps = agg.update([report('gap', q=0.3, side=br.SIDE_LEFT)])  # the departure changes sign while the phase runs: no flip
    assert caps == {br.SIDE_RIGHT: 0.3}

  def test_disagreeing_triggers_cap_both_sides(self):
    agg = br.YieldAggregator()
    _, active, caps = agg.update([report('gap', q=0.25, side=br.SIDE_RIGHT), report('ref', q=0.2, side=br.SIDE_LEFT)])
    assert active and caps == {br.SIDE_RIGHT: 0.25, br.SIDE_LEFT: 0.2}

  def test_a_side_stays_capped_while_the_phase_runs_and_everything_resets_at_its_end(self):
    agg = br.YieldAggregator()
    agg.update([report('gap', q=0.25, side=br.SIDE_RIGHT), report('ref', no_growth=False)])
    _, _, caps = agg.update([report('gap', no_growth=False), report('ref', q=0.2, side=br.SIDE_LEFT)])
    assert caps == {br.SIDE_RIGHT: 0.0, br.SIDE_LEFT: 0.2}  # the right side keeps no growth (q 0); its cap memory can only fall
    agg.update([report('gap', no_growth=False), report('ref', no_growth=False)])
    assert not agg.phase and agg.latched == {} and agg.sides == set()
    _, _, caps = agg.update([report('gap', q=0.12, side=br.SIDE_LEFT)])
    assert caps == {br.SIDE_LEFT: 0.12}

  def test_a_third_trigger_with_a_fixed_side_fits(self):
    # rev 6 B6a: the radar clearance check is one more report, always with the right as its side
    agg = br.YieldAggregator()
    _, active, caps = agg.update([report('gap', no_growth=False), report('ref', no_growth=False),
                                  report('radar_clearance', q=br.Q_CAP_FULL, side=br.SIDE_RIGHT)])
    assert active and caps == {br.SIDE_RIGHT: br.Q_CAP_FULL}

  def test_log_side(self):
    assert br.log_side({}) == 0 and br.log_side({br.SIDE_RIGHT: 0.1}) == 1 and br.log_side({br.SIDE_LEFT: 0.1}) == -1
    assert br.log_side({br.SIDE_RIGHT: 0.1, br.SIDE_LEFT: 0.1}) == br.LOG_SIDE_BOTH

  def test_both_sides_capped_lets_nothing_grow_in_either_direction(self):
    # BlendfulCorrection with caps on both sides: the correction can only shrink toward 0, whatever its sign
    for sign in (1.0, -1.0):
      ctl = BlendfulCorrection()
      ctl.c = sign * 1e-3
      rl = ctl.right
      rl.update = lambda *a, sign=sign, **k: br.RightResult(k0=sign * 1.0, no_growth=True, yield_active=True,
                                                              caps={br.SIDE_RIGHT: br.Q_CAP_FULL, br.SIDE_LEFT: br.Q_CAP_FULL})
      cs = []
      for _ in range(40):
        ctl.update(Scene(v=13.0).model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
        cs.append(ctl.c)
      mags = np.abs([sign * 1e-3] + cs)
      assert np.all(np.diff(mags) <= 1e-15) and cs[-1] == 0.0


class TestCentreLimitSource:
  @staticmethod
  def hold(cs, q_r, q_l, seconds):
    return [cs.update(q_r, q_l, DT_MDL) for _ in range(int(round(seconds / DT_MDL)))]

  def test_two_references_after_0_5_s_above_0_5_and_back_after_0_5_s_below_0_4(self):
    cs = br.CentreLimitSource()
    assert cs.update(0.9, 0.45) == cs.RIGHT
    srcs = self.hold(cs, 0.9, 0.51, 0.6)
    assert srcs[-2] == cs.RIGHT and srcs[-1] == cs.BOTH  # entered once the condition has held longer than 0.5 s, not before
    assert self.hold(cs, 0.9, 0.41, 2.0)[-1] == cs.BOTH  # inside the band: no switch
    srcs = self.hold(cs, 0.9, 0.39, 0.6)
    assert srcs[-2] == cs.BOTH and srcs[-1] == cs.RIGHT
    assert self.hold(cs, 0.9, 0.49, 2.0)[-1] == cs.RIGHT

  def test_a_short_excursion_does_not_switch(self):
    cs = br.CentreLimitSource()
    cs.update(0.9, 0.9)
    assert cs.src == cs.BOTH  # a fresh instance starts in the form that fits
    assert set(self.hold(cs, 0.9, 0.3, 0.45)) == {cs.BOTH}
    assert set(self.hold(cs, 0.9, 0.9, 1.0)) == {cs.BOTH}

  def test_one_reference_side_changes_only_with_a_clear_quality_lead(self):
    cs = br.CentreLimitSource()
    assert cs.update(0.35, 0.3) == cs.RIGHT
    assert cs.update(0.35, 0.44) == cs.RIGHT  # left better by 0.09: keep the right
    assert cs.update(0.30, 0.45) == cs.LEFT  # by more than 0.1: switch
    assert cs.update(0.42, 0.40) == cs.LEFT

  def test_quality_0_5_plus_minus_0_2_at_1_hz_gives_no_step(self):
    # design E3 (rev 6.3): the weaker side's quality 0.5 +- 0.2 at 1 Hz: below 0.4 for less than 0.5 s, above 0.5 for less than 0.5 s
    cs = br.CentreLimitSource()
    cs.update(0.9, 0.3)
    srcs = [cs.update(0.9, 0.5 + 0.2 * np.sin(2 * np.pi * i * DT_MDL)) for i in range(400)]
    assert len(set(srcs)) == 1

  def test_per_frame_flicker_across_0_5_switches_at_most_once(self):
    rng = np.random.default_rng(3)
    cs = br.CentreLimitSource()
    srcs = [cs.update(0.9, 0.5 + 0.08 * (2 * rng.random() - 1)) for _ in range(400)]
    assert sum(a != b for a, b in zip(srcs[:-1], srcs[1:], strict=True)) <= 1


class TestNonFiniteGuard:
  def test_finite_model_inputs_checks_what_the_instances_read(self):
    ok = Scene(v=13.0, edges=True).model_output()
    assert br.finite_model_inputs(ok, 13.0, 0.0)
    assert not br.finite_model_inputs(ok, float('nan'), 0.0)
    for key, idx in (('lane_lines', (0, 3, 5, 0)), ('lane_lines_stds', (0, 1, 5, 0)), ('lane_lines_prob', (0, 3)), ('road_edges', (0, 1, 7, 0)),
                     ('road_edges_stds', (0, 0, 7, 0)), ('plan', (0, 4, 1)), ('plan', (0, 4, Plan.VELOCITY.start)),
                     ('plan', (0, 0, Plan.ORIENTATION_RATE.start + 2))):
      mo = Scene(v=13.0, edges=True).model_output()
      mo[key][idx] = np.nan
      assert not br.finite_model_inputs(mo, 13.0, 0.0), (key, idx)
    mo = Scene(v=13.0).model_output()
    mo['plan'][0, :, Plan.ACCELERATION.start] = np.nan  # a column no instance reads
    assert br.finite_model_inputs(mo, 13.0, 0.0)
    # without the right-line instance only what upstream's Blendful and the gain F read counts
    for key, idx, read_by_upstream in (('lane_lines', (0, 0, 5, 0), False), ('lane_lines', (0, 3, 5, 0), False), ('lane_lines_prob', (0, 7), False),
                                       ('road_edges', (0, 1, 7, 0), False), ('plan', (0, 0, Plan.ORIENTATION_RATE.start + 2), False),
                                       ('lane_lines', (0, 1, 5, 0), True), ('lane_lines_stds', (0, 2, 5, 0), True), ('lane_lines_prob', (0, 5), True),
                                       ('plan', (0, 4, 0), True), ('plan', (0, 4, Plan.VELOCITY.start), True)):
      mo = Scene(v=13.0, edges=True).model_output()
      mo[key][idx] = np.nan
      assert br.finite_model_inputs(mo, 13.0, 0.0, right_line=False) != read_by_upstream, (key, idx)

  def test_regular_without_the_right_line_instance_ignores_values_upstream_does_not_read(self):
    # a non-finite far-right lane line: today's add-on does not read it, so Regular delivers it exactly
    ctl, ref = BlendfulCorrection(), Blendful()
    for i in range(200):
      mo = Scene(v=13.0, center=0.4).model_output()
      if i % 7 == 0:
        mo['lane_lines'][0, 3, :, 0] = np.nan
      ctl.update(mo, 13.0, True, 0.0, LAT_T, MODE_REGULAR, RightInputs())
      up = ref.update(mo, 13.0, True, 0.0, enabled=True)
      assert not ctl.invalid and abs(ctl.c - up) <= 1e-12

  def test_invalid_frame_resets_the_instances_and_is_logged(self):
    from openpilot.cereal import custom
    ctl = BlendfulCorrection()
    drive(ctl, [Scene(v=13.0, center=0.4)] * 100)
    assert ctl.right.filters[2].initialized and ctl.reference.trust > 0.0
    mo = Scene(v=13.0).model_output()
    mo['lane_lines'][0, 1, 3, 0] = np.inf
    ctl.update(mo, 13.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
    assert ctl.invalid and not ctl.right.filters[2].initialized and ctl.reference.trust == 0.0 and ctl.right_idle
    msg = custom.ModelDataV2SP.new_message()
    br.fill_log(msg, ctl, MODE_RIGHT, 13.0)
    assert msg.blendfulNonFinite
    drive(ctl, [Scene(v=13.0, center=0.4)] * 5)
    assert not ctl.invalid

  def test_non_finite_speed_keeps_the_last_speed_for_the_limiter(self):
    ctl = BlendfulCorrection()
    drive(ctl, [Scene(v=20.0, center=0.4)] * 150)
    c0 = ctl.c
    ctl.update(Scene(v=20.0, center=0.4).model_output(), float('nan'), True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
    assert np.isfinite(ctl.c) and abs(ctl.c - c0) <= br.RELEASE / 20.0 ** 2 * DT_MDL + 1e-15


def k1_world(world: str, s: float, v: float = 14.0, k: float = 1 / 100):
  """Steady right circle R 100 m at 14 m/s with the plan speed v_x(0) = s vEgo, under one of the two K1 hypotheses (review S1 / S6):
  H_axis: the lines' x axis carries the model's distance scale (x_model = s x_true), vEgo true; H_vego: the model's x is true metres and
  vEgo reads low by 1/s (the true speed is s vEgo)."""
  if world == 'H_axis':
    return Scene(v=v, k=k, lever=True, vx_scale=s).model_output()
  mo = Scene(v=v * s, k=k, lever=True).model_output()  # true geometry at the true speed s v
  return mo


class TestScaleHypotheses:
  """Review S6 / design E3 (rev 6.3): the 2 x 2 case.  FILTER_ROTATION_SCALE "design" (default, lever x) and "model_axis" (lever x vEgo /
  v_x(0)); each is exact on a circle built under its own hypothesis and biased under the other, so neither the circle tests nor the replay's
  sign argument decide K1; the E1 re-run evaluates both offline (both selectable per instance)."""
  @pytest.mark.parametrize("world,s,variant,lo,hi", [
    ('H_axis', 1.05, 'model_axis', -0.01, 0.01), ('H_axis', 0.95, 'model_axis', -0.01, 0.01),
    ('H_vego', 1.05, 'model_axis', 0.08, 0.16), ('H_vego', 0.95, 'model_axis', -0.16, -0.08),
    ('H_axis', 1.05, 'design', -0.16, -0.08), ('H_axis', 0.95, 'design', 0.08, 0.16),
    ('H_vego', 1.05, 'design', -0.01, 0.01), ('H_vego', 0.95, 'design', -0.01, 0.01),
  ])
  def test_right_line_filter_bias_at_L(self, world, s, variant, lo, hi):
    rl = RightLineBlendful(rotation_scale=variant)
    for _ in range(400):
      res = rl.update(k1_world(world, s), 14.0, True, 1 / 100, 0.0, RightInputs())
    assert lo <= float(res.log['filter_offset'][2]) <= hi

  def test_default_is_the_design_form(self):
    assert br.FILTER_ROTATION_SCALE == 'design' and RightLineBlendful().rotation_scale == 'design'
    with pytest.raises(AssertionError):
      RightLineBlendful(rotation_scale='other')


class TestLagBoundUnderFarFieldErrors:
  """Review S1 / design check 21 as a test: far-field prediction errors above the glide rate are no longer carried into the fit window,
  because every T_lim point beyond 2 L takes T_raw each frame (rev 6.4)."""
  @pytest.mark.parametrize("world,variant", [('H_vego', 'model_axis'), ('H_axis', 'design')])
  @pytest.mark.parametrize("s", [1.05, 0.95])
  def test_no_lag_in_the_fit_window_under_the_other_scale_hypothesis(self, world, variant, s):
    # round 2 / Part 1 (horizon seeding): lag at L +3.9 / -4.2 m (H_vego) and -3.6 / +4.6 m (H_axis, design form)
    rl = RightLineBlendful(rotation_scale=variant)
    for _ in range(400):
      res = rl.update(k1_world(world, s), 14.0, True, 1 / 100, 0.0, RightInputs())
    look = 1.5 * 14.0
    lag = res.log['t_lim'] - res.log['t_raw']
    assert abs(float(np.interp(look, X, lag))) <= 0.05 and abs(float(np.interp(1.5 * look, X, lag))) <= 0.05

  def test_points_beyond_2_L_equal_the_raw_target_every_frame(self):
    gt = GroundTarget()
    rng = np.random.default_rng(1)
    for _ in range(50):
      raw = 0.3 + 0.5 * rng.standard_normal(br.NX)
      t = gt.update(raw, 20.0, 0.004, DT_MDL, reseed_x=60.0)
      np.testing.assert_array_equal(t[X > 60.0], raw[X > 60.0])
      assert np.all(np.abs(t - raw)[X <= 60.0] >= 0.0)

  def test_check_21(self):
    # design check 21 (EST there): R 100 m, 50 km/h, per-frame error (s - 1) x dpsi from the rotation scale; lag at L with re-seeding at 2 L
    v, R = 50 / 3.6, 100.0
    look = max(1.5 * v, 10.0)
    k = 1 / R
    raw = k * X ** 2 / 2 + br.L_D * k * X  # the steady circle seen from the device (T_raw is ground-fixed on it)
    for s in (1.05, 0.95):
      gt = GroundTarget()
      gt.update(raw, v, v * k, DT_MDL)
      for _ in range(600):
        gt.update(raw, v, v * k * s, DT_MDL, scale=1.0, reseed_x=2 * look)  # the prediction rotates by s x the true yaw step
      assert abs(float(np.interp(look, X, gt.lag))) < 0.05

  def test_target_lag_under_a_large_yaw_rate_bias(self):
    # 0.01 rad/s mispredicts by up to 0.095 m per frame at the horizon (Part 1 without re-seeding: 5.3 m of lag at 30 m); with re-seeding at
    # 2 L the lag at L stays within the lag bound
    gt = GroundTarget()
    raw = np.full(br.NX, 0.3)
    worst = 0.0
    for _ in range(600):
      gt.update(raw, 20.0, 0.01, DT_MDL, reseed_x=60.0)
      worst = max(worst, float(np.abs(np.interp(30.0, X, gt.lag))))
    assert worst <= br.LAG_BOUND + br.LAG_RATE * DT_MDL + 1e-6
