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
  """B5 step 4b with "Start and floor" (rev 6.10): the start is the car's own gap at every reset, no floor, no hold; the start follows a car
  still moving left."""
  def setup_method(self):
    self.y_r = np.full(br.NX, 1.75)

  def run(self, ml, t4, frames, w=1.0, sw=None, reset_first=True, y_r=None):
    out = []
    for i in range(frames):
      out.append(ml.update(self.y_r if y_r is None else y_r, t4, w, np.zeros(br.NX) if sw is None else sw, 0.0, DT_MDL, reset_first and i == 0))
    return out

  def test_after_a_reset_the_approach_grows_at_r_ref_and_a_decrease_passes_at_once(self):
    ml = MoveSpeedLimit()
    t4 = np.full(br.NX, 0.44)
    ts = self.run(ml, t4, 300)
    assert np.all(ts[0] <= RATE + 1e-12)  # starts where the car is (gap 1.75 at the reset), plus the first step
    steps = np.diff([t[10] for t in ts])
    assert steps.max() <= RATE + 1e-12 and abs(ts[-1][10] - 0.44) < 1e-9  # r_ref, then on T4
    t_left = ml.update(self.y_r, np.full(br.NX, 0.1), 1.0, np.zeros(br.NX), 0.0, DT_MDL, False)
    assert np.allclose(t_left, 0.1)  # leftward at once

  def test_weighted_approach_and_w_zero(self):
    ml = MoveSpeedLimit()
    t4 = np.full(br.NX, 0.44)
    ts = self.run(ml, t4, 20, w=0.5)
    assert abs(ts[-1][5] - min(0.44, 20 * RATE / 0.5)) < 1e-9  # A = w (G0 - G) grows at r_ref: the target moves 1 / w as fast
    assert np.allclose(self.run(MoveSpeedLimit(), t4, 5, w=0.0)[-1], 0.0)  # G = G0 at w = 0

  def test_reset_starts_at_the_car_wherever_it_is_with_no_floor_and_no_hold(self):
    # the car 0.3 m left of the lane centre (rev 6.10: no p0 = max(car, centre), no hold): the target starts at the car, left of the centre
    ml = MoveSpeedLimit()
    ts = self.run(ml, np.full(br.NX, 0.6), 3)
    assert np.allclose(ts[0], RATE) and np.allclose(ts[2], 3 * RATE)
    assert not hasattr(ml, 'released')

  def test_the_start_does_not_follow_a_car_moving_left(self):
    # review of rev 6.15 item 2: the follow rule is removed; a car moving left never moves the start
    ml = MoveSpeedLimit()
    t4 = np.full(br.NX, 0.44)
    self.run(ml, t4, 40)
    gap, followed = 1.75, []
    for _ in range(40):  # the car moves left at 0.1 m/s for 2 s: its gap grows 5 mm per frame
      gap += 0.1 * DT_MDL
      ml.update(np.full(br.NX, gap), t4 + (gap - 1.75), 1.0, np.zeros(br.NX), 0.0, DT_MDL, False)
      followed.append(ml.followed)
    assert not any(followed)

  def test_the_target_stays_bounded_when_the_weight_falls(self):
    # E1 on round 5 / rev 6.15: with the old weighted state the target diverged as w fell (5.8e4 m at w = 1e-6); now G stays in [G4, G0]
    ml = MoveSpeedLimit()
    y_r = np.full(br.NX, 1.75)
    ml.update(y_r, np.full(br.NX, 0.44), 1.0, np.zeros(br.NX), 0.0, DT_MDL, True)
    for _ in range(20):
      ml.update(y_r, np.full(br.NX, 0.44), 1.0, np.zeros(br.NX), 0.0, DT_MDL, False)
    t4_left = np.full(br.NX, 0.30)  # a step-4 target left of the start
    for w in (0.5, 0.1, 1e-3, 1e-6, 1e-6, 0.3, 1.0):
      t = ml.update(y_r, t4_left, w, np.zeros(br.NX), 0.0, DT_MDL, False)
      g = y_r - t
      assert np.all(np.isfinite(t))
      assert np.all(g >= (y_r - t4_left) - 1e-9) and np.all(g <= ml.g0 + 1e-9)

  def test_a_slow_leftward_drift_below_the_threshold_does_not_follow(self):
    ml = MoveSpeedLimit()
    t4 = np.full(br.NX, 0.44)
    self.run(ml, t4, 40)
    t = ml.update(np.full(br.NX, 1.75 + 0.01 * DT_MDL), t4, 1.0, np.zeros(br.NX), 0.0, DT_MDL, False)
    assert not ml.followed and t[5] > 0.0

  def test_an_outward_switch_of_the_reference_does_not_move_the_target(self):
    ml = MoveSpeedLimit()
    t4 = np.full(br.NX, 0.44)
    ts = self.run(ml, t4, 100)
    before = ts[-1][5]
    sw = np.full(br.NX, 0.3)  # the reference moves 0.3 m outward through the weights (line -> edge), not with the road
    t = ml.update(self.y_r + 0.3, t4 + 0.3, 1.0, sw, 0.0, DT_MDL, False)
    assert abs(t[5] - (before + RATE)) < 1e-9 and not ml.followed  # the switch moves nothing; the approach continues at r_ref

  def test_an_inward_switch_reaches_the_target_at_once(self):
    ml = MoveSpeedLimit()
    ts = self.run(ml, np.full(br.NX, 0.44), 400)
    t = ml.update(self.y_r - 0.3, np.full(br.NX, 0.14), 1.0, np.full(br.NX, -0.3), 0.0, DT_MDL, False)
    assert np.allclose(t, 0.14) and ts[-1][5] == pytest.approx(0.44)

  def test_a_geometric_widening_carries_the_target_with_its_gap(self):
    ml = MoveSpeedLimit()
    ts = self.run(ml, np.full(br.NX, 1.75 - 0.81 - 0.5), 400)
    widened = self.y_r + 0.5 * (X > 30)
    t = ml.update(widened, widened - 0.81 - 0.5, 1.0, np.zeros(br.NX), 0.0, DT_MDL, False)
    assert np.allclose(t - ts[-1], 0.5 * (X > 30), atol=RATE + 1e-9)

  def test_a_lateral_prediction_error_leaves_the_approach_unchanged(self, monkeypatch):
    # B2's motion model with a constant lateral error of 5 mm per frame: T_lim (step 3) sees it, the approach A (step 4b, same-frame
    # gaps, no lateral prediction) does not
    def run():
      rl = RightLineBlendful()
      return [rl.update(Scene(v=13.0).model_output(), 13.0, True, 0.0, 0.0, inputs()).log['approach'] for _ in range(150)]
    clean = run()
    orig = br.GroundTarget.update

    def drifting(self, *a, **k):  # T_lim's own lateral prediction is off by 5 mm per frame
      if self.t is not None:
        self.t = self.t + 0.005
      return orig(self, *a, **k)
    monkeypatch.setattr(br.GroundTarget, 'update', drifting)
    drifted = run()
    assert max(abs(c - d) for c, d in zip(clean, drifted, strict=True)) < 1e-12 and clean[-1] > 0.1

  def test_g0_is_carried_along_the_lane(self):
    ml = MoveSpeedLimit()
    ml.update(self.y_r, np.full(br.NX, 0.44), 1.0, np.zeros(br.NX), 0.0, DT_MDL, True)
    ml.g0 = 1.75 - 0.01 * X
    ml.update(self.y_r, np.full(br.NX, 0.44), 1.0, np.zeros(br.NX), 20.0, DT_MDL, False)
    inside = X + 20.0 * DT_MDL <= X[-1]
    # rev 6.15: a step-4 target left of the start moves the start with it (G0 >= G4 = 1.75 - 0.44)
    np.testing.assert_allclose(ml.g0[inside], np.maximum(1.75 - 0.01 * (X + 1.0), 1.75 - 0.44)[inside], atol=1e-9)


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
    # the lane centre 0.3 m right of the car when the episode ends: the target starts at the car (rev 6.10), left of the centre, and while
    # it is left of the centre the larger of Regular's and the right-line correction is delivered
    scenes = [Scene(v=20.0, center=0.3)] * 150 + [Scene(v=20.0, center=0.3, bend_x0=15.0, bend_R=30.0)] * 30 + [Scene(v=20.0, center=0.3)] * 200
    rl, rows, logs = run_instance(scenes, 20.0, k0_up=1e-4)
    ends = reset_frames(rows, 1)
    assert ends, 'no junction episode ended'
    e = ends[0]
    self.check_reset_then_growth(rows, e)
    lg = logs[e + 20]  # (at the episode end the bent right line still lets the line bound bind somewhere in the window: right-line share)
    assert lg['trl_minus_centre'] < -0.2 and lg['delivered'] in (br.DELIVER_REGULAR, br.DELIVER_LARGER_RIGHT_LINE)
    assert float(np.interp(30.0, X, lg['target'])) < float(np.interp(30.0, X, lg['c_ctr'])) - 0.2  # not floored at the centre

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
# "Start and floor" (design rev 6.10 E3, check 23b as a test): the car's own speed with a model that keeps the road's direction
def scene_loop(kmh, scene, seconds=45.0, regular=False, gap=0.44):
  """The heading-keeping loop of the round-3 review (rv3_move_model.py, model=True): check 20's car (0.2 s dead time, 0.3 s lag) with a model
  that keeps the road's direction (plan along the road from the car; its curvature turns the heading back in about 1 s) and, in 'resting',
  pulls toward 0.75 m left of the centre at line probabilities 0.55 (Regular's trust about 0.8: Regular rests about 0.15 m left of it).
  regular: the same scene in Regular only (the comparison).  Ground frame, lane centre 0, right line 1.75 m; gap 0.44 m: the right-hand
  position is +0.5 m.  Returns time, car position, sideways speed, the right-line target and the centre limit at L (ground), delivered
  source and follow events per frame."""
  v = kmh / 3.6
  ctl = BlendfulCorrection()
  sc = {'y0': 0.0, 'regular_s': 5.0, 'pull': None, 'probs': 0.9, 'mode': lambda t: MODE_RIGHT, 'prob': lambda t: None, 'left': lambda t: -1.75,
        'gap': gap, 'patch': None}
  sc.update(SCENES[scene])
  y, psi, kc = sc['y0'], 0.0, 0.0
  buf = [0.0] * 4
  rows = []
  extra = {'reseeded': [], 'a': [], 'authority_limited': [], 'car_left': []}
  look = max(1.5 * v, 10.0)
  for i in range(int(seconds / DT_MDL)):
    t = i * DT_MDL - sc['regular_s']
    if sc['patch'] is not None:
      sc['patch'](ctl, t)
    pr = sc['prob'](t)
    mo = model_output(v, y, psi, kc, left=sc['left'](t), right_line_prob=sc['probs'] if pr is None else pr)
    mo['lane_lines_prob'][0, 3] = sc['probs']
    k_model = -psi / max(v, 1.0)
    mo['plan'][0, :, 1] = -psi * X
    if sc['pull'] is not None:  # the model wants to be at the pull position (relative to the centre): plan and curvature toward it
      e = sc['pull'][kmh] - y
      mo['plan'][0, :, 1] += e * np.clip(X / look, 0.0, 1.0)
      k_model += 2.0 * e / look ** 2
    mode = MODE_REGULAR if (regular or t < 0) else sc['mode'](t)
    ctl.update(mo, v, True, k_model, LAT_T, mode, inputs(gap_low=sc['gap'], gap_high=sc['gap']))
    buf.append(ctl.c + k_model)
    cmd = buf.pop(0)
    for _ in range(5):
      kc += (cmd - kc) * 0.01 / 0.3
      psi += v * kc * 0.01
      y += v * psi * 0.01
    lg = ctl.result.log
    shift = y + psi * look
    rows.append((t, y, v * psi, float(np.interp(look, X, lg['target'])) + shift if lg else np.nan,
                 float(np.interp(look, X, lg['c_ctr'])) + shift if lg else np.nan, lg.get('delivered', -1) if lg else -1,
                 bool(lg.get('start_follows_car', False)) if lg else False))
    for k in extra:
      extra[k].append(float(lg.get(k, 0.0)) if lg else 0.0)
  scene_loop.last_log = ctl.result.log
  scene_loop.extra = {k: np.array(x) for k, x in extra.items()}
  r = np.array(rows, dtype=float)
  return r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4], r[:, 5], r[:, 6]


def toggling(t):
  return MODE_RIGHT if (t < 0 or t > 30 or int(t / 2) % 2 == 0) else MODE_REGULAR


def flicker(t):
  """The quality-flicker fixture q(t) = 0.45 + 0.3 sin(2 pi t) from t = 0 to 30 s.  Applied as the raw right-line probability (round 4; upstream's
  0.5 s probability filter delivers about 0.45 +- 0.09 to the gate: partial fades), the second, reported variant (design rev 6.12 E3, R4-S3)."""
  return float(0.45 + 0.3 * np.sin(2 * np.pi * t)) if 0 <= t < 30 else 0.9


def patch_each_frame(attr, make):
  """Wrap ctl.right.<attr>.update (the object is rebuilt on every instance reset, so the wrap is checked every frame)."""
  def patch(ctl, t):
    obj = getattr(ctl.right, attr)
    if getattr(obj, '_fixture_t', None) is None:
      obj.update = make(obj.update, obj)
    obj._fixture_t = t
  return patch


def gate_flicker_patch():
  """The shared flicker fixture defined AT THE GATE (design rev 6.12 E3, review R4-S3): the quality the right-line confidence gate reads is
  q_R(t) = 0.45 + 0.3 sin(2 pi t) from t = 0 to 30 s; each dip reaches the gate's zero and is a reset."""
  def make(orig, obj):
    return lambda q: orig(float(0.45 + 0.3 * np.sin(2 * np.pi * obj._fixture_t)) if 0 <= obj._fixture_t < 30 else q)
  return patch_each_frame('quality_gate', make)


def authority_cap_patch(cap):
  """Direct authority fixture (design rev 6.12 E3, C612-1): a* is capped at `cap`, so a settles below 1."""
  def make(orig, obj):
    return lambda a_star: orig(min(a_star, cap))
  return patch_each_frame('authority', make)


SCENES = {
  'plain': {},
  'return': {'y0': -0.4, 'regular_s': 0.0},  # mode on with the car 0.4 m left of the centre (as after a model-alone junction episode)
  # Regular rests about 0.15 m left of the centre: line probabilities 0.55 (Regular's trust about 0.8) and a model pull calibrated per speed
  'resting': {'y0': -0.4, 'regular_s': 0.0, 'pull': {30: -0.396, 50: -0.55, 80: -0.462}, 'probs': 0.55},
  'toggle': {'y0': 0.5, 'regular_s': 0.0, 'mode': toggling},  # at the right-hand position, then Right line <-> Regular every 2 s
  'flicker': {'y0': 0.5, 'regular_s': 0.0, 'prob': flicker},  # the raw-probability variant (reported)
  'flicker_gate': {'y0': 0.5, 'regular_s': 0.0, 'patch': gate_flicker_patch()},  # the shared fixture at the gate input
  # mode on 0.15 m right of the centre with a capped at a* (the authority fixture of rev 6.12; a* set per test)
  **{f'astar{int(c * 10):02d}': {'y0': 0.15, 'regular_s': 0.0, 'patch': authority_cap_patch(c)} for c in (0.6, 0.8, 0.9, 1.0)},
  # a large GUI gap puts the right-line target left of the centre; the centre then steps 0.2 m right (the left line 0.4 m in)
  'centre_step': {'gap': 2.0, 'left': lambda t: -1.75 + (0.4 if t > 5.0 else 0.0)},
}


def right_of_centre_peak(t, y, vs):
  m = (t >= 0) & (y > 0.0)
  return float(vs[m].max()) if m.any() else 0.0


def own_move_peak(t, y, vs, tg, tc):
  """Keep Right's own move: the right-line target and the car beyond the centre (as design check 23b counts it)."""
  m = (t >= 0) & (tg >= tc + 0.1) & (y > 0.05)
  return float(vs[m].max()) if m.any() else 0.0


class TestStartAndFloor:
  """Design rev 6.13 reference numbers (TEST, the designer's harness on the real module, 70 s): plain own 0.0485-0.0496; toggling 0.0483-0.0494;
  flicker (raw probability) 0.048-0.0517; a* 0.6 0.0526 / 0.0510 / 0.0524, 0.8 <= 0.0501, 0.9 <= 0.0490, 1.0 0.0477-0.0483; peak right of the
  centre <= Regular + 0.0017.  Criterion asserted: Keep Right's own move <= 0.052 m/s (a* fixtures <= 0.053, design E3 at 357cb88)."""
  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_plain_move(self, kmh):
    t, y, vs, tg, tc, src, fol = scene_loop(kmh, 'plain', seconds=50.0)
    assert own_move_peak(t, y, vs, tg, tc) <= 0.052 and right_of_centre_peak(t, y, vs) <= 0.052
    assert abs(y[-1] - 0.5) < 0.01

  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_return_from_left_of_the_centre(self, kmh):
    t, y, vs, tg, tc, src, fol = scene_loop(kmh, 'return', seconds=50.0)
    tr, yr, vr, *_ = scene_loop(kmh, 'return', seconds=50.0, regular=True)
    assert right_of_centre_peak(t, y, vs) <= right_of_centre_peak(tr, yr, vr) + 0.002
    assert own_move_peak(t, y, vs, tg, tc) <= 0.052
    assert np.any(src == br.DELIVER_REGULAR) and abs(y[-1] - 0.5) < 0.02

  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_regular_resting_left_of_the_centre(self, kmh):
    # Rev 6.12 E3: never left of Regular and the authority-limited flag (C611-2), not +0.5 m.  A model pull that holds Regular 0.15 m left
    # of the centre at trust 0.8 is a plan 0.40-0.55 m left of the centre; the right-line fit is clipped at D_MAX 1.0 m (C9) and scaled by
    # s_R, so Regular (the larger) is delivered while the right-line target is right of the centre: Keep Right ends at -0.15 .. -0.05 m
    t, y, vs, tg, tc, src, fol = scene_loop(kmh, 'resting', seconds=60.0)
    ex, lg = scene_loop.extra, scene_loop.last_log
    tr, yr, vr, *_ = scene_loop(kmh, 'resting', seconds=60.0, regular=True)
    assert -0.2 < yr[-1] < -0.1  # the fixture: Regular rests about 0.15 m left of the centre
    g = (t >= 0) & (tr >= 0)
    assert np.all(y[g] >= yr[g] - 0.005)  # never left of Regular (5 mm: the loop's integration phase)
    assert right_of_centre_peak(t, y, vs) <= 0.052
    assert vs[t >= 0].max() <= vr[tr >= 0].max() + 0.002
    # authority-limited (C611-2) exactly where Regular is delivered with the right-line target right of the centre for > 2 s (50 km/h here);
    # at 30 / 80 km/h the larger right-line correction is delivered and the car rests right of Regular (no flag)
    if src[-1] == br.DELIVER_REGULAR:
      assert ex['authority_limited'][-1] == 1.0 and lg['trl_minus_centre'] > 0.0
    else:
      assert src[-1] == br.DELIVER_LARGER_RIGHT_LINE and ex['authority_limited'][-1] == 0.0 and y[-1] > yr[-1] + 0.02
    if kmh == 50:
      assert ex['authority_limited'][-1] == 1.0

  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_reset_right_of_the_centre_with_mode_toggling_does_not_pull_back(self, kmh):
    # Regular's phases move the car toward the centre; a reset never pulls it further back than Regular alone does
    t, y, vs, tg, tc, src, fol = scene_loop(kmh, 'toggle', seconds=70.0)
    tr, yr, vr, *_ = scene_loop(kmh, 'toggle', seconds=70.0, regular=True)
    assert y[t >= 0].min() >= yr[tr >= 0].min() - 0.01 and abs(y[-1] - 0.5) < 0.01  # (8 mm more in this loop: phase of the toggles)

  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_mode_toggling_own_move(self, kmh):
    # was a strict xfail in round 4 (0.0546-0.0563); the once-per-reset re-seed (rev 6.13) fixes it.  Exactly one re-seed: the toggles end at
    # t = 30 s; each reset re-arms, and the re-seed fires (once) only after a has settled at a_cmd for 1 s
    t, y, vs, tg, tc, src, fol = scene_loop(kmh, 'toggle', seconds=70.0)
    tr, yr, vr, *_ = scene_loop(kmh, 'toggle', seconds=70.0, regular=True)
    assert own_move_peak(t, y, vs, tg, tc) <= 0.052
    assert right_of_centre_peak(t, y, vs) <= max(right_of_centre_peak(tr, yr, vr) + 0.002, 0.052)
    assert abs(y[-1] - 0.5) < 0.01

  @pytest.mark.parametrize("kmh", SPEEDS)
  @pytest.mark.parametrize("scene", ['flicker_gate', 'flicker'])
  def test_quality_flicker_across_the_gate(self, kmh, scene):
    # flicker_gate: the shared fixture at the gate input (rev 6.12 E3, R4-S3); flicker: the raw-probability variant (reported)
    # At the gate input each early dip is a reset at +0.5 m: a = 0 and Regular pulls the car back at up to 0.21 m/s; the swerve trigger reads
    # that move as a swerve (obstacle on the right) and caps the rightward correction at 0 for about 20 s, so the car waits up to 9 mm left
    # of Regular's path and Regular's own return afterwards (a = 0) is the peak right of the centre (0.066 at 50 km/h vs 0.063 in the
    # Regular-only run).  Pre-existing trigger behaviour (df4d2d5), reported in round 5; asserted: within Regular + 0.002, or below the
    # speed that upstream's own correction (a = 0) reaches in the same run
    t, y, vs, tg, tc, src, fol = scene_loop(kmh, scene, seconds=70.0)
    a = scene_loop.extra['a']
    tr, yr, vr, *_ = scene_loop(kmh, scene, seconds=70.0, regular=True)
    assert own_move_peak(t, y, vs, tg, tc) <= 0.052
    roc = right_of_centre_peak(t, y, vs)
    upstream_alone = (t >= 0) & (a == 0.0)
    assert roc <= max(right_of_centre_peak(tr, yr, vr) + 0.002, 0.052) or (scene == 'flicker_gate' and roc <= vs[upstream_alone].max())
    assert abs(y[-1] - 0.5) < 0.01 and y[t >= 0].min() >= yr[tr >= 0].min() - 0.01

  def test_the_gate_fixture_reaches_the_gate_zero(self):
    # the gate-input fixture's dips are resets (a goes to 0), unlike the raw-probability variant (partial fades)
    scene_loop(50, 'flicker_gate', seconds=35.0)
    a = scene_loop.extra['a']
    tt = np.arange(len(a)) * DT_MDL
    assert np.any((tt > 2) & (tt < 30) & (a == 0.0)) and np.any((tt > 2) & (tt < 30) & (a > 0.0))

  @pytest.mark.parametrize("kmh", SPEEDS)
  @pytest.mark.parametrize("cap", [0.6, 0.8, 0.9, 1.0])
  def test_authority_settling_below_one_after_a_reset_right_of_the_centre(self, kmh, cap):
    # C612-1: mode on 0.15 m right of the centre, a capped at a*: the start re-seeds once at the car after a settles; own move <= 0.053 m/s
    # (design harness 0.0477-0.0526); end position reported (design +0.28-0.31 / +0.39-0.41 / +0.44-0.45 / +0.50 m)
    t, y, vs, tg, tc, src, fol = scene_loop(kmh, f'astar{int(cap * 10):02d}', seconds=70.0)
    ex = scene_loop.extra
    own = own_move_peak(t, y, vs, tg, tc)
    TestStartAndFloor.astar_rows = getattr(TestStartAndFloor, 'astar_rows', {})
    TestStartAndFloor.astar_rows[(cap, kmh)] = (own, float(y[-1]))
    print(f'a*={cap} {kmh} km/h: own {own:.4f} end {y[-1]:+.3f} re-seeds {int(ex["reseeded"].sum())}')
    assert own <= 0.053
    assert int(ex['reseeded'].sum()) == 1
    assert abs(ex['a'][-1] - cap) < 1e-6
    assert {0.6: (0.26, 0.33), 0.8: (0.37, 0.43), 0.9: (0.43, 0.47), 1.0: (0.49, 0.51)}[cap][0] < y[-1] < \
      {0.6: (0.26, 0.33), 0.8: (0.37, 0.43), 0.9: (0.43, 0.47), 1.0: (0.49, 0.51)}[cap][1]

  def test_the_re_seed_needs_a_above_zero(self, monkeypatch):
    # rev 6.13: taken literally ("a has reached a_cmd and a_cmd unchanged >= 1 s") the condition holds in the reset frame itself (a = a_cmd = 0,
    # a_cmd held since the last change) and disarms the re-seed before a has risen: no re-seed ever, and the a* = 0.8 fixture fails
    assert not br.reseed_due(0.0, 0.0, 5.0)
    assert br.reseed_due(0.8, 0.8, 1.0) and not br.reseed_due(0.8, 0.8, 0.9) and not br.reseed_due(0.7, 0.8, 5.0)
    monkeypatch.setattr(br, 'reseed_due', lambda a, a_cmd, age: abs(a - a_cmd) < 1e-9 and age >= br.RESEED_STABLE - 1e-9)
    t, y, vs, tg, tc, src, fol = scene_loop(50, 'astar08', seconds=70.0)
    assert scene_loop.extra['reseeded'].sum() == 0
    assert own_move_peak(t, y, vs, tg, tc) > 0.053

  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_a_bound_left_of_the_centre_is_reached(self, kmh, monkeypatch):
    # a safety bound putting the target 0.2 m left of the centre (the line bound made 1.14 m: 1.75 - 0.81 - 1.14 = -0.2): the car goes
    # there; the right-line correction is delivered (no larger-of while a bound binds, no cap at Regular's pull)
    monkeypatch.setattr(br, 'LINE_BOUND', 1.14)
    t, y, vs, tg, tc, src, fol = scene_loop(kmh, 'plain', seconds=50.0)
    assert abs(y[-1] + 0.2) < 0.02
    assert np.all(src[t > 1.0] == br.DELIVER_RIGHT_LINE)

  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_centre_step_with_the_right_line_target_left_of_the_centre_follows_at_regular_pace(self, kmh):
    t, y, vs, tg, tc, src, fol = scene_loop(kmh, 'centre_step', seconds=40.0)
    tr, yr, vr, *_ = scene_loop(kmh, 'centre_step', seconds=40.0, regular=True)
    assert abs(y[-1] - yr[-1]) < 0.01 and abs(y[-1] - 0.2) < 0.03  # where Regular goes: the new centre
    assert np.any(src[t > 5.0] == br.DELIVER_REGULAR)

  def test_follow_threshold_against_line_jitter(self):
    # a steady hold at the right-hand position with 3 cm of independent jitter on every line, 120 s: no leftward ratchet of the target
    v = 50 / 3.6
    rng = np.random.default_rng(7)
    ctl = BlendfulCorrection()
    y, psi, kc = 0.5, 0.0, 0.0
    buf = [0.0] * 4
    tgs, follows = [], 0
    look = 1.5 * v
    for i in range(int(150 / DT_MDL)):
      mo = model_output(v, y, psi, kc)
      mo['lane_lines'][0, :, :, 0] += 0.03 * rng.standard_normal((4, 1))
      mo['plan'][0, :, 1] = -psi * X
      k_model = -psi / v
      ctl.update(mo, v, True, k_model, LAT_T, MODE_RIGHT, inputs(gap_low=0.44, gap_high=0.44))
      buf.append(ctl.c + k_model)
      cmd = buf.pop(0)
      for _ in range(5):
        kc += (cmd - kc) * 0.01 / 0.3
        psi += v * kc * 0.01
        y += v * psi * 0.01
      lg = ctl.result.log
      if i * DT_MDL >= 30:  # after the start-up (mode on at a = 0) has settled
        tgs.append(float(np.interp(look, X, lg['target'])) + y + psi * look)
        follows += int(lg['start_follows_car'])
    tgs = np.array(tgs)
    assert np.mean(tgs[-200:]) - np.mean(tgs[:200]) > -0.02  # no net leftward drift beyond the jitter
    assert abs(y - 0.5) < 0.03
    TestStartAndFloor.follow_events = follows


# ---------------------------------------------------------------------------------------------------------------------------------
# round 5 (design rev 6.12 / 6.13): the larger-of rule's window (R4-S1) and hysteresis (R4-S2), the re-seed and the swerve trigger, jitter
def static_run(v, ys, mode=MODE_RIGHT, shape=None, seconds_each=3.0, ctl=None, **kw):
  """Open loop: the car held at ground position y (lane centre 0) for seconds_each per entry of ys; shape(mo, t) edits the model output."""
  ctl = ctl or BlendfulCorrection()
  logs = []
  for j, y in enumerate(ys):
    for i in range(int(seconds_each / DT_MDL)):
      mo = model_output(v, y, 0.0, 0.0)
      if shape is not None:
        shape(mo, (j * seconds_each) + i * DT_MDL)
      ctl.update(mo, v, True, 0.0, LAT_T, mode, inputs(gap_low=0.44, gap_high=0.44, **kw))
    logs.append(dict(ctl.result.log))
  return ctl, logs


class TestLargerOfWindowAndHysteresis:
  @pytest.mark.parametrize("far, binds", [(2.0, False), (1.2, True)])
  def test_a_far_field_bound_does_not_switch_the_larger_of_rule_off(self, far, binds):
    # R4-S1: the car 0.3 m left of the centre (larger-of on); from t = 8 s the right line ahead of far * L_a moves in by 1.7 m (a lane that
    # narrows beyond a junction): the line bound puts the target left of the centre there.  Beyond the fit window [0.5, 1.5] L_a it does not
    # switch the larger-of rule off (Regular stays delivered); inside the window it does (the right-line correction is delivered)
    v = 50 / 3.6
    la = br.approach_length(v)

    def narrowing(mo, t):
      if t > 8.0:
        m = X > far * la
        mo['lane_lines'][0, 2:4, m, 0] -= 1.7
    _, logs = static_run(v, [-0.3] * 4, shape=narrowing)
    lg = logs[-1]
    assert lg['bound_active'] and np.any(lg['t4'] < lg['c_ctr'] - 0.05) and lg['car_left']
    assert lg['bound_left_in_window'] == binds
    assert (lg['delivered'] == br.DELIVER_RIGHT_LINE) == binds
    if not binds:
      assert lg['delivered'] == br.DELIVER_REGULAR

  def test_car_left_of_the_centre_has_2_cm_hysteresis(self):
    # R4-S2: "car left of the centre" switches on above +0.02 m and off below -0.02 m (offset of the centre limit from the car at x = 0)
    v = 50 / 3.6
    ys = [-0.1, 0.01, -0.01, 0.03, 0.01, -0.01, -0.03, 0.01]
    want = [True, True, True, False, False, False, True, True]
    _, logs = static_run(v, ys, seconds_each=4.0)
    offs = [float(lg['c_ctr'][0]) for lg in logs]
    for y, o in zip(ys, offs, strict=True):
      assert abs(o + y) < 0.004, (y, o)  # the fixture: the centre-limit offset at the car is the car's distance left of the centre
    assert [bool(lg['car_left']) for lg in logs] == want


class TestReseedAndSwerveTrigger:
  @staticmethod
  def first_yield(frames_hold, reseed=True, monkeypatch=None):
    if not reseed:
      monkeypatch.setattr(br, 'reseed_due', lambda *a: False)
    ctl = BlendfulCorrection()
    first, reseed_frame, q = None, None, []
    for i in range(frames_hold + 150):
      t = max(0.0, (i - frames_hold) * DT_MDL)
      ctl.update(Scene(v=13.0, plan=-0.8 * min(1.0, t / 0.5)).model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, inputs())
      lg = ctl.result.log
      if lg['reseeded']:
        reseed_frame = i
      if first is None and ctl.result.yield_active:
        first = i
      q.append(ctl.result.yield_q)
    if not reseed:
      monkeypatch.undo()
    return first, reseed_frame, np.array(q, dtype=float)

  def test_the_re_seed_leaves_the_swerve_trigger_untouched(self, monkeypatch):
    # C612-1: the re-seed moves the start of the right-line target only; the swerve trigger's running minimum and usual are not reset.  A swerve
    # (obstacle on the right) starting during or right after the re-seed is caught in the same frame, with the same endpoint gaps, as without it
    _, rs, _ = self.first_yield(400)
    assert rs is not None
    for start in (rs - 2, rs, rs + 1, rs + 10):
      with_reseed = self.first_yield(start)
      without = self.first_yield(start, reseed=False, monkeypatch=monkeypatch)
      assert with_reseed[0] is not None and with_reseed[0] == without[0], (start, with_reseed[0], without[0])
      np.testing.assert_array_equal(with_reseed[2], without[2])

  def test_no_ratchet_under_line_jitter_with_repeated_resets(self):
    # 3 cm independent jitter on every line and a 0.4 s quality dip to zero at the gate input (a reset) every 40 s, 6 times: after each reset
    # the start is the car and the re-seed fires at most once; the car returns to the right-hand position each time and neither the position
    # before a reset nor the excursion after it moves left from one reset to the next (no ratchet).  Note (pre-existing, identical at
    # df4d2d5): in this loop the blend toward Regular during the dip and the re-entry hold move the car 0.25 m left within 2 s, the swerve
    # trigger reads that as a swerve (obstacle on the right) and caps for about 20 s at -0.06 m; hence the 40 s period
    v = 50 / 3.6
    rng = np.random.default_rng(11)
    ctl = BlendfulCorrection()
    y, psi, kc = 0.5, 0.0, 0.0
    buf = [0.0] * 4
    reseeds, ys, episodes, prev_reset = 0, [], 0, 0
    n, period, t0 = 6, 40.0, 30.0
    dip = patch_each_frame('quality_gate', lambda orig, obj: (lambda q: orig(0.0 if t0 <= obj._fixture_t < t0 + n * period
                                                                               and (obj._fixture_t - t0) % period < 0.4 else q)))
    for i in range(int((t0 + n * period + 10) / DT_MDL)):
      t = i * DT_MDL
      mo = model_output(v, y, psi, kc)
      mo['lane_lines'][0, :, :, 0] += 0.03 * rng.standard_normal((4, 1))
      mo['plan'][0, :, 1] = -psi * X
      k_model = -psi / v
      dip(ctl, t)
      ctl.update(mo, v, True, k_model, LAT_T, MODE_RIGHT, inputs(gap_low=0.44, gap_high=0.44))
      reseeds += int(ctl.result.log['reseeded'])
      mr = int(ctl.result.log['move_reset'])
      episodes += int(mr != 0 and prev_reset == 0)  # a reset episode: a rising edge of the move-limit reset reasons
      prev_reset = mr
      ys.append(y)
      buf.append(ctl.c + k_model)
      cmd = buf.pop(0)
      for _ in range(5):
        kc += (cmd - kc) * 0.01 / 0.3
        psi += v * kc * 0.01
        y += v * psi * 0.01
    ys = np.array(ys)
    tt = np.arange(len(ys)) * DT_MDL
    before = [float(ys[np.argmin(abs(tt - (t0 + k * period - 0.1)))]) for k in range(n)]
    lows = [float(ys[(tt >= t0 + k * period) & (tt < t0 + (k + 1) * period)].min()) for k in range(n)]
    TestReseedAndSwerveTrigger.jitter_rows = (before, lows, reseeds, float(y), episodes)
    assert min(before) > 0.47 and before[-1] - before[0] > -0.02
    assert max(lows) - min(lows) < 0.03
    assert abs(y - 0.5) < 0.03
    # at most one re-seed per reset episode (rising edge of move_reset).  Here bit 6 (w = 0) never occurs alone (always with bit 0, a = 0);
    # the extra episodes beyond the dips come from bit 2 (yield end) and bit 3 (right reference regained): each dip's false swerve cap
    # (note above) ends in a yield end, which re-arms and re-seeds once the follow rule is gone (rev 6.15)
    assert reseeds <= episodes


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
    results = [res] + [step([(1, 10.5, -1.4, 0.0)]) for _ in range(10)]
    assert all(r.yield_active for r in results)  # held through the 0.3 s re-confirmation (and d > 0 again afterwards)
    # rev 6.10 B6a: the track that lost its confirmation stays the remembered object, the deficit is computed from it meanwhile
    assert all(r.deficit == pytest.approx(0.11) for r in results) and all(r.memory >= 1 for r in results[:5])

  def test_a_lost_track_is_remembered_for_one_second_only(self):
    rcl = RadarClearance()
    C, E, W0 = np.zeros(br.NX), np.full(br.NX, 1e3), np.zeros(br.NX)
    for _ in range(6):
      res = rcl.update(frame([(1, 30.0, -1.4, 0.0)]), 13.0, 0.0, 0.0, 0.5, 0.0, br.W_R, X, C, E, W0)  # 30 m ahead: beyond the memory zone
    assert res.objects == 1
    objs = [rcl.update(frame([]), 13.0, 0.0, 0.0, 0.5, 0.0, br.W_R, X, C, E, W0).objects for _ in range(25)]
    assert objs[:19] == [1] * 19 and objs[-1] == 0

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


# ---------------------------------------------------------------------------------------------------------------------------------
# owner 13 Nov 09:27: once Keep Right's own swerve trigger has latched, hand the whole correction to the model's path
class TestHandOverToTheModelPath:
  """Owner 13 Nov 09:27: from the latch of Keep Right's own swerve trigger the WHOLE correction goes to 0 through the final limiter.  The
  fixtures are chosen so that no cap rule can bring c to 0: the capped side is the other side than the correction's direction (a cap only
  limits a correction toward the capped side), so only the hand-over does it."""
  @staticmethod
  def latched_run(side, no_hand_over=False):
    v = 50 / 3.6
    ctl = BlendfulCorrection()
    # side +1: a rightward correction (gap 0.44 m: the right-hand position); side -1: a leftward one (gap 2.0 m puts the target left of
    # the centre, car held 0.3 m right of the centre: Regular and the right-line correction pull left)
    y0, gap = (0.0, 0.44) if side > 0 else (0.3, 2.0)
    for _ in range(int(30 / DT_MDL)):  # open loop: the car held at y0
      ctl.update(model_output(v, y0, 0.0, 0.0), v, True, 0.0, LAT_T, MODE_RIGHT, inputs(gap_low=gap, gap_high=gap))
    c_built = ctl.c
    assert side * c_built > 1e-5
    orig = ctl.right.update
    cap_side = br.SIDE_LEFT if side > 0 else br.SIDE_RIGHT  # the other side: no cap rule limits this correction

    def latched(*a, **kw):
      res = orig(*a, **kw)
      res.own_yield, res.yield_active, res.no_growth = not no_hand_over, True, True  # no_hand_over: the same yield, not handed over
      res.caps = {cap_side: 0.3}
      return res
    ctl.right.update = latched
    cs, flags = [], []
    for _ in range(int(5 / DT_MDL)):
      ctl.update(model_output(v, y0, 0.0, 0.0), v, True, 0.0, LAT_T, MODE_RIGHT, inputs(gap_low=gap, gap_high=gap))
      cs.append(ctl.c)
      flags.append(ctl.hand_over)
    return c_built, np.array(cs), flags, ctl

  @pytest.mark.parametrize("side", [+1, -1])
  def test_own_yield_releases_the_whole_correction_through_the_limiter(self, side):
    c_built, cs, flags, ctl = self.latched_run(side)
    assert all(flags)
    assert np.all(side * np.diff(cs) <= 1e-15)  # only falls toward 0, through the limiter (no step beyond its rates)
    assert abs(cs[-1]) < 1e-12  # the whole correction (Keep Right's extra and Regular's part) reaches 0: the model's own path
    # log: the correction at the latch and the time to reach 0
    assert ctl.hand_over_c == pytest.approx(c_built)
    t0 = (np.argmax(cs == 0.0) + 1) * DT_MDL
    assert ctl.hand_over_time == pytest.approx(t0, abs=1e-9) and 0.0 < ctl.hand_over_time < 5.0

  @pytest.mark.parametrize("side", [+1, -1])
  def test_without_the_hand_over_the_correction_stays(self, side):
    # the same yield and cap without the hand-over: the other-side cap does not bring c to 0, so the test above discriminates (with the
    # hand-over line removed from the code it fails: results/2026-11-13_a8_first_push/FOLLOWUP.md)
    _, cs, flags, _ = self.latched_run(side, no_hand_over=True)
    assert not any(flags)
    assert side * cs[-1] > 1e-5
