"""
Tests of design rev 6.5 (inster-dev results/2026-11-12_a8_right_line_blendful/DESIGN.md at 5f80b26, E3) added in implementation round 3
Part 2: the final move-speed limit with its re-entry resets (B5 step 4b), the closed-loop step and the check-20b cases, the radar right-side
clearance check (B6a) and the three-trigger side rule.  Closed loops here are crude (0.2 s dead time + 0.3 s lag, no model reaction, no MPC),
like the design's check 20.  Signs: model frame y > 0 is right; c > 0 steers right.  Offline only; not validated on the car.
"""
import numpy as np
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.modeld_v2 import blendful_right as br
from openpilot.sunnypilot.modeld_v2 import radar_clearance as rc
from openpilot.sunnypilot.modeld_v2.blendful import Blendful
from openpilot.sunnypilot.modeld_v2.blendful_right import (MODE_OFF, MODE_REGULAR, MODE_RIGHT, BlendfulCorrection, MoveSpeedLimit,
                                                           RightLineBlendful)
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.radar_clearance import RadarClearance, RadarFrame, healthy_empty_radar
from openpilot.sunnypilot.modeld_v2.tests.test_blendful_right import Scene

X = np.array(ModelConstants.X_IDXS)
LAT_T = 0.275
RATE = br.REFERENCE_RATE * DT_MDL


def inputs(radar=None, **kw):
  return br.RightInputs(radar=healthy_empty_radar() if radar is None else radar, **kw)


# ---------------------------------------------------------------------------------------------------------------------------------
# the limit itself (pure)
class TestMoveSpeedLimitUnit:
  def setup_method(self):
    self.y_r = np.full(br.NX, 1.75)
    self.c_ctr = np.zeros(br.NX)

  def run(self, ml, t4, frames, w=1.0, sw=None, reset_first=True, y_r=None, c_ctr=None):
    out = []
    for i in range(frames):
      out.append(ml.update(self.y_r if y_r is None else y_r, t4, self.c_ctr if c_ctr is None else c_ctr, w,
                           np.zeros(br.NX) if sw is None else sw, 0.0, DT_MDL, reset_first and i == 0))
    return out

  def test_after_a_reset_the_approach_grows_at_0_05_m_s_and_a_decrease_passes_at_once(self):
    ml = MoveSpeedLimit()
    t4 = np.full(br.NX, 0.44)
    ts = self.run(ml, t4, 300)
    assert np.all(ts[0] <= RATE + 1e-12)  # starts where the car is (gap 1.75 at the reset), plus the first step
    steps = np.diff([t[10] for t in ts])
    assert steps.max() <= RATE + 1e-12 and abs(ts[-1][10] - 0.44) < 1e-9  # r_ref, then on T4
    t_left = ml.update(self.y_r, np.full(br.NX, 0.1), self.c_ctr, 1.0, np.zeros(br.NX), 0.0, DT_MDL, False)
    assert np.allclose(t_left, 0.1)  # leftward at once

  def test_weighted_approach_and_w_zero(self):
    ml = MoveSpeedLimit()
    t4 = np.full(br.NX, 0.44)
    ts = self.run(ml, t4, 20, w=0.5)
    assert abs(ts[-1][5] - min(0.44, 20 * RATE / 0.5)) < 1e-9  # A = w (G0 - G) grows at r_ref: the target moves 1 / w as fast
    ml2 = MoveSpeedLimit()
    assert np.allclose(self.run(ml2, t4, 5, w=0.0)[-1], 0.0)  # G = G0 at w = 0

  def test_reset_at_the_centre_limit_when_the_car_is_left_of_it_and_held_until_the_car_is_there(self):
    ml = MoveSpeedLimit()
    c_ctr = np.full(br.NX, 0.3)  # the car is 0.3 m left of the centre limit
    t4 = np.full(br.NX, 0.6)
    ts = self.run(ml, t4, 40, c_ctr=c_ctr)
    assert all(np.allclose(t, 0.3) for t in ts) and not ml.released  # the floor; A held at 0 while the car is not at p0
    y_r = self.y_r - 0.27  # the car came to within 0.03 m of p0 (its gap fell), slowly
    for _ in range(3):
      t = ml.update(y_r, t4, c_ctr - 0.27, 1.0, np.zeros(br.NX), 0.0, DT_MDL, False)
    assert ml.released and t[5] > 0.03 + 1e-9

  def test_an_outward_switch_of_the_reference_does_not_move_the_target(self):
    ml = MoveSpeedLimit()
    t4 = np.full(br.NX, 0.44)
    ts = self.run(ml, t4, 100)
    before = ts[-1][5]
    sw = np.full(br.NX, 0.3)  # the reference moves 0.3 m outward through the weights (line -> edge), not with the road
    t = ml.update(self.y_r + 0.3, t4 + 0.3, self.c_ctr, 1.0, sw, 0.0, DT_MDL, False)
    assert abs(t[5] - (before + RATE)) < 1e-9  # the switch itself moves nothing; the approach continues at r_ref

  def test_an_inward_switch_reaches_the_target_at_once(self):
    ml = MoveSpeedLimit()
    ts = self.run(ml, np.full(br.NX, 0.44), 400)
    t = ml.update(self.y_r - 0.3, np.full(br.NX, 0.14), self.c_ctr, 1.0, np.full(br.NX, -0.3), 0.0, DT_MDL, False)
    assert np.allclose(t, 0.14) and ts[-1][5] == pytest.approx(0.44)

  def test_a_geometric_widening_carries_the_target_with_its_gap(self):
    ml = MoveSpeedLimit()
    ts = self.run(ml, np.full(br.NX, 1.75 - 0.81 - 0.5), 400)
    widened = self.y_r + 0.5 * (X > 30)
    t = ml.update(widened, widened - 0.81 - 0.5, self.c_ctr, 1.0, np.zeros(br.NX), 0.0, DT_MDL, False)
    assert np.allclose(t - ts[-1], 0.5 * (X > 30), atol=RATE + 1e-9)

  def test_the_centre_limit_is_a_floor(self):
    ml = MoveSpeedLimit()
    t = self.run(ml, np.full(br.NX, 0.44), 50, c_ctr=np.full(br.NX, 0.2))[-1]
    assert np.all(t >= 0.2 - 1e-12)

  def test_a_lateral_prediction_error_leaves_the_approach_unchanged(self, monkeypatch):
    # B2's motion model with a constant lateral error of 5 mm per frame: T_lim (step 3) sees it, the approach A (step 4b, same-frame
    # gaps, no lateral prediction) does not
    def run():
      rl = RightLineBlendful()
      return [rl.update(Scene(v=13.0).model_output(), 13.0, True, 0.0, 0.0, inputs()).log['approach'] for _ in range(150)]
    clean = run()
    orig = br.motion_predict
    monkeypatch.setattr(br, 'motion_predict', lambda *a, **k: orig(*a, **k) + 0.005)
    drifted = run()
    assert max(abs(c - d) for c, d in zip(clean, drifted, strict=True)) < 1e-12 and clean[-1] > 0.1

  def test_g0_is_carried_along_the_lane(self):
    ml = MoveSpeedLimit()
    ml.update(self.y_r, np.full(br.NX, 0.44), self.c_ctr, 1.0, np.zeros(br.NX), 0.0, DT_MDL, True)
    ml.g0 = 1.75 - 0.01 * X
    ml.update(self.y_r, np.full(br.NX, 0.44), self.c_ctr, 1.0, np.zeros(br.NX), 20.0, DT_MDL, False)
    inside = X + 20.0 * DT_MDL <= X[-1]
    np.testing.assert_allclose(ml.g0[inside], (1.75 - 0.01 * (X + 1.0))[inside], atol=1e-9)


# ---------------------------------------------------------------------------------------------------------------------------------
# re-entries (full instance, open loop: the car does not move, so A grows from the car's gap at r_ref)
def growth_ok(rows, start, v):
  """From frame `start` on: A carried along the lane grows by at most r_ref x dt per frame at every window point."""
  for (a0, _), (a1, _) in zip(rows[start:-1], rows[start + 1:], strict=True):
    carried = np.interp(X + v * DT_MDL, X, a0)
    if np.max(a1 - carried) > RATE + 1e-12:
      return False
  return True


def run_instance(scenes, v, inputs_fn=None, rl=None, k0_up=0.0, force_regular_fn=None):
  rl = rl or RightLineBlendful()
  rows, logs = [], []
  for i, sc in enumerate(scenes):
    p = inputs_fn(i) if inputs_fn else inputs()
    res = rl.update(sc.model_output(), v, True, 0.0, k0_up, p, force_regular=force_regular_fn(i) if force_regular_fn else False)
    rows.append((rl.move.a_lim.copy(), res.log['move_reset']))
    logs.append(res.log)
  return rl, rows, logs


def reset_frames(rows, bit):
  return [i for i, (_, r) in enumerate(rows) if r & (1 << bit)]


class TestReEntries:
  V = 13.0

  def check_reset_then_growth(self, rows, frame):
    assert np.all(rows[frame][0] <= RATE + 1e-12)  # A reset to 0 (and the first r_ref step of the same frame, as in check 20b)
    assert growth_ok(rows, frame, self.V)

  def test_mode_switched_on_from_regular(self):
    sc = [Scene(v=self.V)] * 300
    rl, rows, logs = run_instance(sc, self.V, force_regular_fn=lambda i: i < 100)
    on = [i for i in reset_frames(rows, 4) if i >= 100]
    assert on and on[0] == 100
    self.check_reset_then_growth(rows, 100)

  def test_mode_switched_on_from_off(self):
    v = self.V
    ctl = BlendfulCorrection()
    for _ in range(20):
      ctl.update(Scene(v=v).model_output(), v, True, 0.0, LAT_T, MODE_OFF, inputs())
    ctl.update(Scene(v=v).model_output(), v, True, 0.0, LAT_T, MODE_RIGHT, inputs())
    assert ctl.result.log['move_reset'] & (1 << 4) and np.all(ctl.right.move.a_lim <= RATE + 1e-12)

  @pytest.mark.parametrize("cause", ['line_loss', 'left_blinker', 'radar_health'])
  def test_a_reaching_zero(self, cause):
    scenes = [Scene(v=self.V, pr=0.0 if (cause == 'line_loss' and 150 <= i < 250) else 0.9) for i in range(500)]

    def inp(i):
      if cause == 'left_blinker':
        return inputs(left_blinker=150 <= i < 250)
      if cause == 'radar_health' and 150 <= i < 250:
        return inputs(radar=RadarFrame(tracks=[], age=1.0, fault=False, new=False, config_ok=True))  # stale
      return inputs()
    rl, rows, logs = run_instance(scenes, self.V, inp)
    zero = [i for i in reset_frames(rows, 0) if i > 150]
    assert zero, 'a never reached 0'
    last = zero[-1]
    assert logs[last]['a'] == 0.0
    self.check_reset_then_growth(rows, last)
    if cause == 'radar_health':
      assert logs[200]['f_radar'] == 0.0 and logs[-1]['f_radar'] == 1.0

  def test_junction_episode_end_with_the_car_left_of_the_centre(self):
    # the lane centre 0.3 m right of the car when the episode ends: the target starts at the centre limit and A is held there
    scenes = [Scene(v=20.0, center=0.3)] * 150 + [Scene(v=20.0, center=0.3, bend_x0=15.0, bend_R=30.0)] * 30 + [Scene(v=20.0, center=0.3)] * 200
    rl, rows, logs = run_instance(scenes, 20.0)
    ends = reset_frames(rows, 1)
    assert ends, 'no junction episode ended'
    e = ends[0]
    assert logs[e]['approach_held']
    lg = logs[e + 5]
    np.testing.assert_allclose(lg['target'], np.minimum(lg['c_ctr'], lg['t4']), atol=1e-6)  # at the centre limit (the floor)

  @pytest.mark.parametrize("kind", ['right', 'left', 'going_around'])
  def test_yield_end(self, kind):
    v = self.V

    class Ref(Blendful):
      def update(self, *a, **k):
        out = super().update(*a, **k)
        self.n = getattr(self, 'n', 0) + 1
        if kind != 'right' and 120 <= self.n < 200:
          self.going_around = True
          self.yield_departure = 0.3 if kind == 'left' else -0.3
        return out
    ctl = BlendfulCorrection(reference=Ref())
    rows = []
    for i in range(600):
      plan = -0.8 if (kind == 'right' and 120 <= i < 200) else 0.0
      ctl.update(Scene(v=v, plan=plan).model_output(), v, True, 0.0, LAT_T, MODE_RIGHT, inputs())
      rows.append((ctl.right.move.a_lim.copy(), ctl.result.log['move_reset']))
    ends = [i for i in reset_frames(rows, 2) if i > 120]
    assert ends, 'no yield ended'
    self.check_reset_then_growth(rows, ends[-1])

  def test_clearance_yield_end(self):
    v = self.V

    def inp(i):
      tracks = [(7, 6.0, -1.0, -v)] if 100 <= i < 160 else []  # an object 1.0 m right of the car centre: deficit > 0
      return inputs(radar=RadarFrame(tracks=tracks, age=0.0, fault=False, new=True, config_ok=True))
    rl, rows, logs = run_instance([Scene(v=v)] * 400, v, inp)
    assert any(lg['radar'].yield_active for lg in logs[100:170])
    ends = [i for i in reset_frames(rows, 2) if i > 100]
    assert ends
    self.check_reset_then_growth(rows, ends[-1])

  def test_right_reference_regained(self):
    scenes = [Scene(v=self.V)] * 150 + [Scene(v=self.V, pr=0.0)] * 15 + [Scene(v=self.V)] * 300  # a short loss: a does not reach 0
    rl, rows, logs = run_instance(scenes, self.V)
    regained = [i for i in reset_frames(rows, 3) if i > 150]
    assert regained
    self.check_reset_then_growth(rows, regained[0])

  def test_partial_fade_keeps_a_and_holds_the_target_back_as_a_returns(self):
    # a falls part-way (the bend fade) without a reset: A is kept, and as a returns the weighted approach still grows <= 0.05 m/s
    rl, rows, logs = run_instance([Scene(v=self.V)] * 600, self.V)
    assert growth_ok(rows, 1, self.V)

  def test_releasing_object_bound_returns_at_0_05_m_s(self):
    # an object on the right line for 3 s (the object bound binds: 0.2 m given up at <= 50 km/h), then gone: the target returns at
    # <= 0.05 m/s relative to the right reference
    v = self.V

    def inp(i):
      return inputs(radar=RadarFrame(tracks=[(3, 12.0, -1.75, -v)] if 300 <= i < 360 else [], age=0.0, fault=False, new=True, config_ok=True))
    rl, rows, logs = run_instance([Scene(v=v)] * 800, v, inp)
    gap = np.array([float(np.interp(1.5 * v, X, lg['y_r'] - lg['target'])) for lg in logs])
    assert gap[330] > gap[290] + 0.1  # the bound took shift away
    assert np.diff(gap[360:]).min() >= -RATE - 1e-9  # and it comes back at <= 0.05 m/s


# ---------------------------------------------------------------------------------------------------------------------------------
# closed loop (design check 20 / 20b as tests)
def model_output(v, y, psi, kc, left=-1.75, right=1.75, right_edge=None, right_line_prob=0.9):
  lines = np.zeros((1, 4, 33, 2))
  stds = np.ones((1, 4, 33, 2))
  for i, o in enumerate((left - 3.5, left, right, right + 3.5)):
    lines[0, i, :, 0] = o - y - psi * X
  stds[0, 1, :, 0] = stds[0, 2, :, 0] = 0.1
  probs = np.zeros((1, 8))
  probs[0, 3], probs[0, 5] = 0.9, right_line_prob
  plan = np.zeros((1, 33, 15))
  plan[0, :, 0] = X
  plan[0, :, 3] = v
  plan[0, 0, Plan.ORIENTATION_RATE.start + 2] = v * kc
  mo = {'lane_lines': lines, 'lane_lines_stds': stds, 'lane_lines_prob': probs, 'plan': plan}
  if right_edge is not None:
    e = np.zeros((1, 2, 33, 2))
    e[0, 0, :, 0] = left - 1.5 - y - psi * X
    e[0, 1, :, 0] = right_edge - y - psi * X
    mo['road_edges'] = e
    mo['road_edges_stds'] = np.full((1, 2, 33, 2), 0.3)
  return mo


class ReferenceWithoutOwnYield(Blendful):
  """Today's Blendful whose going-around / away yield is not passed on as a trigger.  Only for check 20's loop, where the plan runs along
  the car's own heading (no model reaction), so a fast sideways return looks to upstream like a swerve of its plan."""
  def update(self, *a, **k):
    out = super().update(*a, **k)
    self.going_around, self.yield_fade = False, 1.0
    return out


def closed_loop(v, scene, seconds=45.0, gap=0.44, regular_s=5.0, y0=0.0, reference=None):
  """Regular for regular_s (the car settles at the lane centre), then Right line (mode on).  scene(t) -> model_output kwargs (lane geometry
  in the ground frame); the car follows the delivered correction through 0.2 s dead time and a 0.3 s lag, with no model reaction (check
  20's loop: a correction capped to 0 leaves the car's heading as it is).  Returns time, car position, sideways speed and per frame
  (target at L, centre limit at L, authority a, junction episode) in the ground frame."""
  ctl = BlendfulCorrection(reference=reference)
  y, psi, kc = y0, 0.0, 0.0
  buf = [0.0] * 4
  out_t, ys, vs, tg, tc, aa, ep = [], [], [], [], [], [], []
  for i in range(int(seconds / DT_MDL)):
    t = i * DT_MDL
    mode = MODE_REGULAR if t < regular_s else MODE_RIGHT
    ctl.update(model_output(v, y, psi, kc, **scene(t - regular_s)), v, True, 0.0, LAT_T, mode, inputs(gap_low=gap, gap_high=gap))
    buf.append(ctl.c)
    cmd = buf.pop(0)
    for _ in range(5):
      kc += (cmd - kc) * 0.01 / 0.3
      psi += v * kc * 0.01
      y += v * psi * 0.01
    lg = ctl.result.log
    out_t.append(t - regular_s)
    ys.append(y)
    vs.append(v * psi)
    look = max(1.5 * v, 10.0)
    tg.append(float(np.interp(look, X, lg['target'])) + y + psi * look if lg and 'target' in lg else np.nan)
    tc.append(float(np.interp(look, X, lg['c_ctr'])) + y + psi * look if lg and 'c_ctr' in lg else np.nan)
    aa.append(lg.get('a', 0.0) if lg else 0.0)
    ep.append(bool(lg.get('episode', False)) if lg else False)
  closed_loop.last = (np.array(aa), np.array(ep))
  return np.array(out_t), np.array(ys), np.array(vs), np.array(tg), np.array(tc)


SPEEDS = (30, 50, 80)


class TestClosedLoop:
  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_step_of_half_a_metre(self, kmh):
    t, y, vs, _, _ = closed_loop(kmh / 3.6, lambda t: {})
    final = 1.75 - br.W_R - 0.44  # 0.5 m right of the centre
    after = t >= 0
    assert abs(y[-1] - final) < 0.01
    assert vs[after].max() <= 0.052  # design rev 6.7 E3 (check 20's loop follows r_ref 0.042 at about 0.042 m/s)
    arr = int(np.argmax(y >= 0.9 * final))
    err = np.diff((final - y)[after][: arr - int(np.sum(~after)) + 1])
    err = err[np.abs(err) > 1e-7]
    assert int(np.sum(np.sign(err[1:]) != np.sign(err[:-1]))) == 0  # no reversal before arrival
    assert y.max() - final < 0.01 * final  # overshoot < 1 %
    assert np.abs(y[arr + int(5 / DT_MDL):] - final).max() < 0.01

  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_right_reference_switch_line_to_edge_through_the_weights(self, kmh):
    # 5 s into the move the inner right line (1.75) fades over 1 s and the edge (2.15, counted 2.05) takes over.  Design rev 6.7 E3:
    # <= 0.053 m/s where w changes during the move (s_R follows q_R).  Without the switch part in G0 it would be 0.18 m/s (check 20b).
    def scene(t):
      return {'right_edge': 2.15, 'right_line_prob': float(np.clip(0.9 - 0.9 * (t - 5.0), 0.0, 0.9)) if t > 5.0 else 0.9}
    t, y, vs, _, _ = closed_loop(kmh / 3.6, scene, seconds=60.0)
    assert vs[t >= 0].max() <= 0.053
    assert abs(y[-1] - (2.15 - br.EDGE_EXTRA - br.W_R - 0.44)) < 0.01  # ends at the edge-referenced position, no false yield on the way

  @pytest.mark.parametrize("kmh", SPEEDS)
  @pytest.mark.parametrize("step,start", [(-0.3, 5.0), (-1.5, 5.0), (-0.3, 30.0), (-1.5, 30.0)])
  def test_left_reference_steps_outward_never_move_the_target_left_or_left_of_the_centre(self, kmh, step, start):
    # design E3 / check 20b.  In the whole instance a 1.5 m outward step of the left reference is also a lane widening by 1.5 m, i.e. a
    # junction cue (B7, widening over time): the episode fades both corrections (model alone), a reaches 0 and the move limit resets to
    # the car's position.  So for -1.5 m the "never moves left" part is asserted up to the episode (listed in IMPL.md); after it the
    # delivered correction is 0 (episode) and the target starts again where the car is
    def scene(t):
      return {'left': -1.75 + (step if t > start else 0.0)}
    t, y, vs, tg, tc = closed_loop(kmh / 3.6, scene, seconds=45.0)
    a, episode = closed_loop.last
    on = t >= 0.5
    assert np.all(tg[on] >= tc[on] - 1e-6)  # never left of the centre limit
    i0 = int(np.argmax(t > start))
    stop = int(np.argmax(episode[i0:])) + i0 if episode[i0:].any() else len(t)
    assert abs(step) > 1.0 or stop == len(t)  # -0.3 m is no junction cue
    assert stop > i0 + 2 and np.diff(tg[i0 - 2:stop]).min() >= -0.003  # never moves left (the car's own heading noise only, mm)

  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_binding_centre_limit_step_is_followed_at_the_loop_pace_to_the_centre(self, kmh):
    # a large GUI gap puts the right-line target left of the centre: the centre limit binds; it then steps 0.2 m right (left line 0.4 m in)
    def scene(t):
      return {'left': -1.75 + (0.4 if t > 5.0 else 0.0)}
    t, y, vs, tg, tc = closed_loop(kmh / 3.6, scene, seconds=40.0, gap=2.0)
    on = t >= 0.5
    assert np.all(tg[on] >= tc[on] - 1e-6)
    assert abs(y[-1] - 0.2) < 0.01  # ends at the new centre
    assert vs[t > 5.0].max() > 0.06  # at the loop's (Regular's) pace, not limited to 0.05 m/s

  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_start_left_of_the_centre_returns_at_the_loop_pace_then_0_05_m_s_beyond(self, kmh):
    # mode on with the car 0.4 m left of the lane centre, as after a model-alone junction episode (reset at p0 = T_C'(0), held until the
    # car is there): the return to the centre runs at the loop's pace, the move beyond the centre at <= 0.055 m/s (design E3).  In check
    # 20's loop the plan follows the car's heading, so the 0.17 m/s return makes upstream's own going-around latch (its plan end moves
    # right): with it as a trigger the cap (obstacle on the left) leaves the crude loop no way to turn the heading back.  That trigger is
    # therefore not passed on here (ReferenceWithoutOwnYield); with a real model the plan follows the road.
    t, y, vs, tg, tc = closed_loop(kmh / 3.6, lambda t: {}, seconds=50.0, regular_s=0.0, y0=-0.4, reference=ReferenceWithoutOwnYield())
    assert vs[y < -0.05].max() > 0.06
    assert vs[y > 0.05].max() <= 0.055
    assert abs(y[-1] - (1.75 - br.W_R - 0.44)) < 0.01


# ---------------------------------------------------------------------------------------------------------------------------------
# radar right-side clearance check (B6a)
def frame(tracks, **kw):
  return RadarFrame(tracks=tracks, age=kw.pop('age', 0.0), fault=kw.pop('fault', False), new=kw.pop('new', True),
                    config_ok=kw.pop('config_ok', True), **kw)


class TestRadarClearanceUnit:
  C = np.zeros(br.NX)  # centre limit
  E = np.full(br.NX, 1e3)  # no edge
  W0 = np.zeros(br.NX)  # edge weight

  def step(self, rcl, tracks, v=13.0, gap=0.5, c_ctr=None, edge=None, ew=None, **kw):
    return rcl.update(frame(tracks, **kw), v, v, 0.0, gap, 0.0, br.W_R, X, self.C if c_ctr is None else c_ctr, self.E if edge is None else edge,
                      self.W0 if ew is None else ew)

  def confirmed(self, rcl, tracks, n=6, **kw):
    for _ in range(n):
      res = self.step(rcl, tracks, **kw)
    return res

  def test_conversion_and_sign(self):
    rcl = RadarClearance()
    self.confirmed(rcl, [(1, 10.0, -2.0, 0.0)])
    tr = rcl.tracks[1]
    assert tr.x == pytest.approx(11.52) and tr.y == pytest.approx(2.0)  # yRel > 0 = left; model y > 0 = right
    rcl2 = RadarClearance()
    rcl2.update(frame([(1, 10.0, -2.0, 0.0)]), 13.0, 13.0, 0.0, 0.5, 0.1, br.W_R, X, self.C, self.E, self.W0)
    assert rcl2.tracks[1].y == pytest.approx(2.1)  # + dc

  def test_confirmation_after_six_consistent_messages_and_restart_on_a_jump(self):
    rcl = RadarClearance()
    for n in range(1, 7):
      self.step(rcl, [(1, 10.0, -1.5, 0.0)])
      assert rcl.tracks[1].confirmed == (n >= 6)
    self.step(rcl, [(1, 13.0, -1.5, 0.0)])  # the slot switched to another object 3 m further
    assert not rcl.tracks[1].confirmed and rcl.tracks[1].count == 1

  def test_memory_beside_the_car_until_one_metre_behind_the_bumper(self):
    rcl = RadarClearance()
    self.confirmed(rcl, [(1, 5.0, -1.8, -13.0)])
    n = 0
    while 1 in rcl.tracks:
      res = self.step(rcl, [])
      n += 1
      assert res.objects == (1 if 1 in rcl.tracks else 0)
      assert n < 200
    assert 6.52 - 13.0 * DT_MDL * n < rc.FORGET_X + 13.0 * DT_MDL

  def test_corridor_edge_rule(self):
    rcl = RadarClearance()
    res = self.confirmed(rcl, [(1, 10.0, -4.0, 0.0)], edge=np.full(br.NX, 2.5), ew=np.ones(br.NX))
    assert res.objects == 0  # 1.5 m beyond a clear edge: roadside
    rcl = RadarClearance()
    res = self.confirmed(rcl, [(1, 10.0, -4.0, 0.0)], edge=np.full(br.NX, 2.5), ew=np.zeros(br.NX))
    assert res.objects == 1  # no clear edge: nothing is ignored
    rcl = RadarClearance()
    res = self.confirmed(rcl, [(1, 10.0, 1.0, 0.0)])
    assert res.objects == 0  # left of the centre limit: not in the right corridor

  def test_margin_is_the_gui_gap_plus_0_2_to_0_5_m(self):
    for kmh, gap, m in ((30, 0.5, 0.7), (50, 0.5, 0.7), (75, 0.75, 1.1), (100, 1.0, 1.5), (120, 1.0, 1.5), (40, 0.8, 1.0)):
      rcl = RadarClearance()
      res = self.confirmed(rcl, [(1, 10.0, -2.0, 0.0)], v=kmh / 3.6, gap=gap)
      assert res.margin == pytest.approx(m)
      assert res.t_obj == pytest.approx(2.0 - br.W_R - m)

  def test_bound_falls_at_once_and_rises_after_one_second_at_0_05_m_s(self):
    rcl = RadarClearance()
    res = self.confirmed(rcl, [(1, 10.0, -2.0, 0.0)])
    low = res.t_obj
    res = self.step(rcl, [(1, 10.0, -2.5, 0.0)])  # the object is 0.5 m further right
    ts = [res.t_obj] + [self.step(rcl, [(1, 10.0, -2.5, 0.0)]).t_obj for _ in range(60)]
    assert ts[0] == low and np.all(np.diff(ts[:19]) == 0.0)  # held for 1 s
    assert np.diff(ts[20:]).max() <= br.REFERENCE_RATE * DT_MDL + 1e-12 and ts[-1] > low

  def test_popup_confirmed_one_second_ahead_binds_in_its_confirmation_frame(self):
    rcl = RadarClearance()
    for n in range(6):
      res = self.step(rcl, [(9, 13.0 - 13.0 * DT_MDL * n, -1.6, -13.0)])
    assert res.popups == 1 and res.t_obj == pytest.approx(1.6 - br.W_R - 0.7)  # full margin in the confirmation frame, no hold, no class wait

  def test_clearance_yield_start_ramp_and_release(self):
    rcl = RadarClearance()
    res = self.confirmed(rcl, [(1, 8.0, -1.4, 0.0)])  # clearance at the centre limit 1.4 - 0.81 = 0.59 < 0.7: d = 0.11
    assert res.deficit == pytest.approx(0.11) and res.yield_active
    assert rc.clearance_q(0.0, 0.1, 0.15, 0.2) == 0.1 and rc.clearance_q(1e-9, 0.1, 0.15, 0.2) == pytest.approx(0.15)
    assert rc.clearance_q(0.05, 0.1, 0.15, 0.2) == pytest.approx(0.2)
    for _ in range(19):
      res = self.step(rcl, [(1, 8.0, -1.65, 0.0)])  # d = -0.14 < -0.1
    assert res.yield_active
    res = self.step(rcl, [(1, 8.0, -1.65, 0.0)])
    assert not res.yield_active  # released after 1 s

  @pytest.mark.parametrize("bad", [{"age": 0.5}, {"fault": True}, None])
  def test_stale_or_faulty_radar_is_unhealthy_and_returns_after_one_second(self, bad):
    rcl = RadarClearance()
    for _ in range(25):
      res = self.step(rcl, [])
    assert res.healthy
    res = rcl.update(None, 13.0, 13.0, 0.0, 0.5, 0.0, br.W_R, X, self.C, self.E, self.W0) if bad is None else self.step(rcl, [], **bad)
    assert not res.healthy
    hs = [self.step(rcl, []).healthy for _ in range(25)]
    assert not any(hs[:19]) and all(hs[20:])

  def test_stationary_and_moving_classification_is_logged_only(self):
    v = 13.0
    rcl = RadarClearance()
    for n in range(12):  # a stationary post: vRel = -v, failing the radial test for two frames
      vr = -v + (3.0 if n in (7, 8) else 0.0)
      res = self.step(rcl, [(1, 30.0 - v * DT_MDL * n, -2.0, vr)])
    assert res.moving == 0
    rcl = RadarClearance()
    for n in range(12):  # a cyclist at 20 km/h ahead: vRel = 20/3.6 - v
      res = self.step(rcl, [(2, 20.0 - (v - 20 / 3.6) * DT_MDL * n, -2.0, 20 / 3.6 - v)])
    assert res.moving == 1


class TestRadarInTheInstance:
  V = 40 / 3.6  # <= 50 km/h: m = g + 0.2 = 0.7 m

  def t4_at_L(self, tracks, frames=60, **kw):
    rl = RightLineBlendful()
    for _ in range(frames):
      res = rl.update(Scene(v=self.V).model_output(), self.V, True, 0.0, 0.0, inputs(radar=frame(tracks, **kw)))
    return rl, res

  def test_object_on_the_line_gives_up_at_most_0_2_m_and_one_e_beyond_gives_up_nothing(self):
    _, base = self.t4_at_L([])
    look = 1.5 * self.V
    t0 = float(np.interp(look, X, base.log['t4']))
    _, on_line = self.t4_at_L([(1, 12.0, -1.75, -self.V)])
    _, beyond = self.t4_at_L([(1, 12.0, -1.95, -self.V)])
    given_up = t0 - float(np.interp(look, X, on_line.log['t4']))
    assert 0.19 <= given_up <= 0.2 + 1e-9
    assert abs(t0 - float(np.interp(look, X, beyond.log['t4']))) < 1e-9

  def test_the_bound_never_goes_left_of_the_centre_limit(self):
    _, res = self.t4_at_L([(1, 12.0, -0.9, -self.V)])  # very close: T_obj far left of the centre
    assert np.all(res.log['t4'] >= np.minimum(res.log['c_ctr'], res.log['t_lim']) - 1e-9)
    assert res.log['radar'].yield_active and res.caps == {br.SIDE_RIGHT: pytest.approx(res.caps[br.SIDE_RIGHT])}

  def test_right_blindspot_removes_the_right_line_shift(self):
    _, res = self.t4_at_L([], right_blindspot=True)
    np.testing.assert_allclose(res.log['t4'], np.minimum(res.log['c_ctr'], res.log['t_lim']), atol=1e-9)

  def test_radar_fault_hands_back_to_regular(self):
    rl, res = self.t4_at_L([])
    assert res.log['a'] > 0.9
    for _ in range(40):
      res = rl.update(Scene(v=self.V).model_output(), self.V, True, 0.0, 0.0, inputs(radar=frame([], fault=True)))
    assert res.log['f_radar'] == 0.0 and res.log['a'] == 0.0  # a = 0: the correction is regular upstream's (R1)
    res = rl.update(Scene(v=self.V).model_output(), self.V, True, 0.0, 0.0, br.RightInputs())  # no radar data at all
    assert res.log['f_radar'] == 0.0

  def test_the_check_does_not_read_the_plan(self):
    outs = []
    for plan in (0.0, -0.6, 0.6):
      rl = RightLineBlendful()
      for _ in range(40):
        res = rl.update(Scene(v=self.V, plan=plan).model_output(), self.V, True, 0.0, 0.0, inputs(radar=frame([(1, 12.0, -1.7, -self.V)])))
      r = res.log['radar']
      outs.append((r.t_obj, r.deficit, r.objects, r.yield_active, r.margin))
    assert outs[0] == outs[1] == outs[2]


class TestThreeTriggerSides:
  def test_clearance_trigger_sets_the_right_and_disagreeing_triggers_leave_no_correction(self):
    v = 13.0

    class Ref(Blendful):
      def update(self, *a, **k):
        out = super().update(*a, **k)
        self.n = getattr(self, 'n', 0) + 1
        if self.n > 150:
          self.going_around, self.yield_departure = True, 0.3  # the reference instance yields for an obstacle on the LEFT
        return out
    ctl = BlendfulCorrection(reference=Ref())
    cs = []
    for i in range(320):
      tracks = [(1, 6.0, -1.0, -v)] if i >= 100 else []  # an object close on the right from frame 100: the clearance yield
      ctl.update(Scene(v=v).model_output(), v, True, 0.0, LAT_T, MODE_RIGHT, inputs(radar=frame(tracks)))
      cs.append(ctl.c)
      if 110 <= i < 150:
        assert br.SIDE_RIGHT in ctl.result.caps and br.SIDE_LEFT not in ctl.result.caps
      if i >= 160:
        assert set(ctl.result.caps) == {br.SIDE_RIGHT, br.SIDE_LEFT}
    assert abs(cs[-1]) < 1e-9  # both caps: kappa0 -> 0 through the limiter, the model alone


# ---------------------------------------------------------------------------------------------------------------------------------
# the car's own speed with a heading-keeping model (design rev 6.7 B5 / E3: r_ref so that the car moves at <= 0.05 m/s; bound 0.052 m/s)
def model_loop(seconds, v_fn, scene_fn, mode_fn=lambda t: MODE_RIGHT, y0=0.0, regular_s=5.0, gap=(0.5, 1.0)):
  """The review's loop (results/2026-11-12_a8_impl_r3/review/rv3_move_model.py, model=True): check 20's car (0.2 s dead time, 0.3 s lag) with a
  model that keeps the road's direction (plan along the road from the car; its curvature turns the heading back in about 1 s), healthy
  empty radar.  Returns the car's peak sideways speed toward the right while it is right of the lane centre (+0.02 m), from t = 0."""
  ctl = BlendfulCorrection()
  y, psi, kc = y0, 0.0, 0.0
  buf = [0.0] * 4
  peak = 0.0
  for i in range(int(seconds / DT_MDL)):
    t = i * DT_MDL - regular_s
    v = v_fn(t)
    mo = model_output(v, y, psi, kc, **scene_fn(t))
    mo['plan'][0, :, 1] = -psi * X
    k_model = -psi / max(v, 1.0)
    ctl.update(mo, v, True, k_model, LAT_T, MODE_REGULAR if t < 0 else mode_fn(t), inputs(gap_low=gap[0], gap_high=gap[1]))
    buf.append(ctl.c + k_model)
    cmd = buf.pop(0)
    for _ in range(5):
      kc += (cmd - kc) * 0.01 / 0.3
      psi += v * kc * 0.01
      y += v * psi * 0.01
    if t >= 0 and y > 0.02:
      peak = max(peak, v * psi)
  return peak


def const(val):
  return lambda t: val


V50 = 50 / 3.6
FLICKER = 'right-line probability 0.45 +- 0.3 at 1 Hz'


class TestCarSpeedWithAHeadingKeepingModel:
  """Review R3-S1 numbers at r_ref 0.042 m/s (review/rref.txt): 0.0497 / 0.0509 / 0.0500, switch 0.0511, 100 -> 50 km/h 0.0491."""
  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_step(self, kmh):
    assert model_loop(60, const(kmh / 3.6), lambda t: {}) <= 0.052

  def test_line_to_edge_switch_mid_move(self):
    peak = model_loop(60, const(V50), lambda t: {'right_edge': 2.15, 'right_line_prob': float(np.clip(0.9 - 0.9 * (t - 5), 0, 0.9)) if t > 5 else 0.9})
    assert peak <= 0.053  # w changes during the move (design rev 6.7 E3)

  def test_speed_drop_100_to_50_km_h(self):
    assert model_loop(80, lambda t: float(np.interp(t, [40, 50], [100 / 3.6, 50 / 3.6])), lambda t: {}) <= 0.052

  # Cases the review shows r_ref cannot bound (R3-S1); expected to fail until design rev 6.8's statement is implemented (round 4 Part B)
  @pytest.mark.xfail(strict=True, reason="R3-S1: return from 0.4 m left of the centre crosses it at the loop's pace (review 0.0873 m/s at any r_ref)")
  def test_return_from_left_of_the_centre(self):
    assert model_loop(60, const(V50), lambda t: {}, y0=-0.4, regular_s=0.0) <= 0.055

  @pytest.mark.xfail(strict=True, reason="R3-S1: Right line <-> Regular every 2 s (review 0.0659 m/s at r_ref 0.042)")
  def test_mode_toggling_every_2_s(self):
    assert model_loop(70, const(V50), lambda t: {}, mode_fn=lambda t: MODE_RIGHT if (t < 20 or t > 50 or int(t / 2) % 2 == 0) else MODE_REGULAR) <= 0.052

  @pytest.mark.xfail(strict=True, reason="R3-S1: right-line quality flicker across the gate (review 0.0636 m/s at r_ref 0.042; needs <= 0.034)")
  def test_quality_flicker_across_the_gate(self):
    assert model_loop(70, const(V50), lambda t: {'right_line_prob': float(np.clip(0.45 + 0.3 * np.sin(2 * np.pi * t), 0, 1)) if 20 <= t < 50 else 0.9}) <= 0.052


# ---------------------------------------------------------------------------------------------------------------------------------
# review R3-B1: the radar configuration is part of the radar health
class TestRadarConfiguration:
  @staticmethod
  def params_for(brand, radar_unavailable, flags):
    """CarParams / CarParamsSP bytes as card.py writes them (selfdrive/car/card.py:184-194: CP.to_bytes(), convert_to_capnp(CP_SP))."""
    from opendbc.car import structs
    from opendbc.car.structs import car
    from openpilot.selfdrive.car.helpers import convert_to_capnp
    store = {"CarParams": car.CarParams.new_message(brand=brand, radarUnavailable=radar_unavailable).to_bytes(),
             "CarParamsSP": convert_to_capnp(structs.CarParamsSP(flags=flags)).to_bytes()}

    class P:
      def get(self, key):
        return store.get(key)
    return P()

  @pytest.mark.parametrize("brand,unavailable,flags,ok", [
    ("hyundai", False, 'FULL', True),  # the Inster with fingerprinted RADAR_3A5_3C4 tracks
    ("hyundai", False, 'LEAD', False),  # lead-only: one SCC lead point with yRel NaN
    ("hyundai", False, 0, False),  # no radar flag
    ("hyundai", True, 'FULL', False),
    ("toyota", False, 0, True),
    ("toyota", True, 0, False),  # RadarInterfaceBase / radarUnavailable: empty messages
  ])
  def test_radar_tracks_ok_from_the_params_card_py_writes(self, brand, unavailable, flags, ok):
    from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP
    f = {'FULL': int(HyundaiFlagsSP.RADAR_FULL_RADAR), 'LEAD': int(HyundaiFlagsSP.RADAR_LEAD_ONLY)}.get(flags, flags)
    assert rc.read_radar_tracks_ok(self.params_for(brand, unavailable, f)) is ok

  def test_missing_or_unreadable_params_are_not_ok(self):
    class P:
      def __init__(self, store):
        self.store = store

      def get(self, key):
        return self.store.get(key)
    assert not rc.read_radar_tracks_ok(P({}))
    assert not rc.read_radar_tracks_ok(P({"CarParams": b"junk", "CarParamsSP": b"junk"}))

  V = 13.0

  def run(self, frames_fn, n=300):
    rl = RightLineBlendful()
    for i in range(n):
      res = rl.update(Scene(v=self.V).model_output(), self.V, True, 0.0, 0.0, br.RightInputs(radar=frames_fn(i)))
    return res.log

  def test_lead_only_radar_hands_back_to_regular(self):
    lg = self.run(lambda i: RadarFrame(tracks=[(0, 25.0, float('nan'), 0.0)], age=0.02, fault=False, new=True, config_ok=False))
    assert lg['f_radar'] == 0.0 and lg['a'] == 0.0
    lg = self.run(lambda i: RadarFrame(tracks=[(0, 25.0, float('nan'), 0.0)], age=0.02, fault=False, new=True, config_ok=True))
    assert lg['f_radar'] == 0.0 and lg['a'] == 0.0  # the second guard: a non-finite yRel alone makes the frame a fault frame

  def test_radar_unavailable_with_empty_messages_at_20_hz_hands_back_to_regular(self):
    lg = self.run(lambda i: RadarFrame(tracks=[], age=0.02, fault=False, new=True, config_ok=False))
    assert lg['f_radar'] == 0.0 and lg['a'] == 0.0

  @pytest.mark.parametrize("bad", [(float('inf'), -1.5), (12.0, float('nan')), (float('nan'), 0.0)])
  def test_non_finite_values_are_a_fault(self, bad):
    lg = self.run(lambda i: frame([(4, bad[0], bad[1], 0.0)] if i % 7 == 0 else []), n=40)
    assert lg['f_radar'] == 0.0  # every 7th frame a fault: never 1 s of healthy frames

  def test_healthy_configuration(self):
    lg = self.run(lambda i: frame([(7, 25.0 - 1.52, -(1.75 + 0.3), 0.0)]))
    assert lg['f_radar'] == 1.0 and lg['a'] == pytest.approx(1.0) and lg['radar'].objects == 1


class TestClearanceYieldHold:
  def test_one_unconfirmed_message_does_not_release_the_clearance_yield(self):
    # review R3-S3: a track that loses its confirmation for one inconsistent message needs 0.3 s to come back; the yield is held 1 s
    rcl = RadarClearance()
    C, E, W0 = np.zeros(br.NX), np.full(br.NX, 1e3), np.zeros(br.NX)

    def step(tracks):
      return rcl.update(frame(tracks), 13.0, 0.0, 0.0, 0.5, 0.0, br.W_R, X, C, E, W0)
    for _ in range(6):
      res = step([(1, 8.0, -1.4, 0.0)])
    assert res.yield_active
    res = step([(1, 10.5, -1.4, 0.0)])  # a 2.5 m jump: confirmation restarts
    actives = [res.yield_active] + [step([(1, 10.5, -1.4, 0.0)]).yield_active for _ in range(10)]
    assert all(actives)  # held through the 0.3 s re-confirmation (and d > 0 again afterwards)

  def test_released_when_the_object_has_passed_the_rear_bumper(self):
    rcl = RadarClearance()
    C, E, W0 = np.zeros(br.NX), np.full(br.NX, 1e3), np.zeros(br.NX)
    v = 13.0
    for n in range(6):
      res = rcl.update(frame([(1, 3.0 - v * DT_MDL * n, -1.4, -v)]), v, v, 0.0, 0.5, 0.0, br.W_R, X, C, E, W0)
    assert res.yield_active
    for _ in range(30):  # leaves the field of view, remembered, passes x < -3.3 m within about 0.4 s
      res = rcl.update(frame([]), v, v, 0.0, 0.5, 0.0, br.W_R, X, C, E, W0)
    assert not res.yield_active and not rcl.tracks
