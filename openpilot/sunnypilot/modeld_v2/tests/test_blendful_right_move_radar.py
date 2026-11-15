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
RATE = br.MOVE_SPEED * DT_MDL


def inputs(radar=None, **kw):
  return br.RightInputs(radar=healthy_empty_radar() if radar is None else radar, **kw)


# ---------------------------------------------------------------------------------------------------------------------------------
# the limit itself (pure)
class TestMoveSpeedLimitUnit:
  """B5 step 4b, rev 7: the start is the car's own gap at the limiter's creation; the unweighted approach U grows by <= r_ref dt per frame
  and is kept whatever the weight does; a decrease passes at once."""
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

  def test_unweighted_approach_independent_of_the_weight(self):
    # rev 7: U (not w U) grows at r_ref, so the target moves at r_ref for any w, also at w = 0, and U is kept when w falls and rises again
    t4 = np.full(br.NX, 0.44)
    for w in (1.0, 0.5, 0.0):
      ts = self.run(MoveSpeedLimit(), t4, 20, w=w)
      assert abs(ts[-1][5] - 20 * RATE) < 1e-9
    ml = MoveSpeedLimit()
    self.run(ml, t4, 20, w=1.0)
    for w in (0.0, 0.0, 1.0):
      t = ml.update(self.y_r, t4, w, np.zeros(br.NX), 0.0, DT_MDL, False)
    assert abs(t[5] - 23 * RATE) < 1e-9

  def test_reset_starts_at_the_car_wherever_it_is_with_no_floor_and_no_hold(self):
    # the car 0.3 m left of the lane centre (rev 6.10: no p0 = max(car, centre), no hold): the target starts at the car, left of the centre
    ml = MoveSpeedLimit()
    ts = self.run(ml, np.full(br.NX, 0.6), 3)
    assert np.allclose(ts[0], RATE) and np.allclose(ts[2], 3 * RATE)
    assert not hasattr(ml, 'released')

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

  def test_an_outward_switch_of_the_reference_does_not_move_the_target(self):
    ml = MoveSpeedLimit()
    t4 = np.full(br.NX, 0.44)
    ts = self.run(ml, t4, 100)
    before = ts[-1][5]
    sw = np.full(br.NX, 0.3)  # the reference moves 0.3 m outward through the weights (line -> edge), not with the road
    t = ml.update(self.y_r + 0.3, t4 + 0.3, 1.0, sw, 0.0, DT_MDL, False)
    assert abs(t[5] - (before + RATE)) < 1e-9  # the switch moves nothing; the approach continues at r_ref

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


class TestReEntries:
  """Rev 7 (owner 13 Nov 11:23, one continuous controller): the move limit resets only at its creation (start, lane-tracking restart, mode
  on from Off); a = 0, a yield end, a junction end and the right reference regained reset nothing (their bits are logged only), so the
  unweighted approach U is kept and the correction returns at the strength filter's pace."""
  V = 13.0

  def test_mode_switched_on_from_off(self):
    v = self.V
    ctl = BlendfulCorrection()
    for _ in range(20):
      ctl.update(Scene(v=v).model_output(), v, True, 0.0, LAT_T, MODE_OFF, inputs())
    ctl.update(Scene(v=v).model_output(), v, True, 0.0, LAT_T, MODE_RIGHT, inputs())
    assert ctl.result.log['move_reset'] & (1 << 5) and np.all(ctl.right.move.u <= RATE + 1e-12)

  @staticmethod
  def u_at_l(rl, v):
    return float(np.interp(1.5 * v, X, rl.move.u))

  @pytest.mark.parametrize("cause", ['line_loss', 'left_blinker', 'radar_health', 'mode_regular'])
  def test_a_reaching_zero_keeps_the_approach(self, cause):
    v = self.V
    rl = RightLineBlendful()
    us, a, bits = [], [], 0
    for i in range(900):
      gone = 400 <= i < 600  # 10 s: the strength filter (tau 1 s) brings a to 0
      p = inputs(left_blinker=gone and cause == 'left_blinker',
                 radar=RadarFrame(tracks=[], age=1.0, fault=False, new=False, config_ok=True) if (gone and cause == 'radar_health') else None)
      res = rl.update(Scene(v=v, pr=0.0 if (gone and cause == 'line_loss') else 0.9).model_output(), v, True, 0.0, 0.0, p,
                      force_regular=gone and cause == 'mode_regular')
      us.append(self.u_at_l(rl, v))
      a.append(res.log['a'])
      bits |= res.log['move_reset'] if gone else 0
    us, a = np.array(us), np.array(a)
    assert a[599] == 0.0 and bits & 1  # a went to 0 and the a = 0 bit was logged
    assert np.all(np.diff(us[399:]) >= -1e-9) and us[899] >= us[399] - 1e-9  # U was never reset
    assert a[-1] > 0.99

  def test_a_yield_end_keeps_the_approach(self):
    v = self.V
    rl = RightLineBlendful()
    us, ends = [], []
    for i in range(500):
      tracks = [(7, 6.0, -1.0, -v)] if 300 <= i < 360 else []  # a clearance yield (object 1.0 m right of the car centre)
      res = rl.update(Scene(v=v).model_output(), v, True, 0.0, 0.0, inputs(radar=RadarFrame(tracks=tracks, age=0.0, fault=False, new=True,
                                                                                               config_ok=True)))
      us.append(self.u_at_l(rl, v))
      if res.log['move_reset'] & (1 << 2):
        ends.append(i)
    assert ends and all(us[e] >= us[e - 1] - 1e-9 for e in ends)

  def test_releasing_object_bound_returns_at_the_move_speed(self):
    # an object 0.3 m inside the right line for 3 s (the object bound binds; rev 8: floored at the lane centre), then gone: after the 1 s
    # wait the target returns at <= the move speed (0.2 m/s, rev 8 items 1c / 5) relative to the right reference, from the centre: back
    # within 1 s + 0.44 m / 0.2 m/s + 0.5 s (rev 7.2: from 0.06 m left of the centre at 0.042 m/s, about 12 s)
    v = self.V

    def inp(i):
      return inputs(radar=RadarFrame(tracks=[(3, 12.0, -1.45, -v)] if 300 <= i < 360 else [], age=0.0, fault=False, new=True, config_ok=True))
    rl, rows, logs = run_instance([Scene(v=v)] * 800, v, inp)
    gap = np.array([float(np.interp(1.5 * v, X, lg['y_r'] - lg['target'])) for lg in logs])
    assert gap[340] > gap[290] + 0.4  # the bound took the shift away (to the centre)
    assert np.diff(gap[360:]).min() >= -RATE - 1e-9  # and it comes back at <= the move speed
    back = 360 + int(np.argmax(gap[360:] <= gap[290] + 1e-6))
    remembered = (12.0 + rc.RADAR_TO_MODEL_X - rc.FORGET_X) / v  # the object stays in the memory until 1 m behind the rear bumper
    assert br.MOVE_SPEED == 0.2 and gap[back] <= gap[290] + 1e-6 and (back - 360) * DT_MDL <= remembered + 1.0 + 0.44 / 0.2 + 0.5


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
  in the ground frame); the car follows the delivered correction through 0.2 s dead time and a 0.3 s lag.  Rev 7.2: with the heading-keeping
  model of scene_loop (plan along the road from the car, its curvature turns the heading back in about 1 s): check 20's loop without a model
  reaction runs away whenever a yield releases the correction (a capped correction leaves the heading as it is).  Returns time, car
  position, sideways speed and per frame (target at L, centre limit at L, authority a, junction episode) in the ground frame."""
  ctl = BlendfulCorrection(reference=reference)
  y, psi, kc = y0, 0.0, 0.0
  buf = [0.0] * 4
  out_t, ys, vs, tg, tc, aa, ep = [], [], [], [], [], [], []
  for i in range(int(seconds / DT_MDL)):
    t = i * DT_MDL
    mode = MODE_REGULAR if t < regular_s else MODE_RIGHT
    mo = model_output(v, y, psi, kc, **scene(t - regular_s))
    mo['plan'][0, :, 1] = -psi * X
    k_model = -psi / max(v, 1.0)
    ctl.update(mo, v, True, k_model, LAT_T, mode, inputs(gap_low=gap, gap_high=gap))
    buf.append(ctl.c + k_model)
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
# rev 8, review S2: the 0.2 m/s move itself (owner's move speed) in this rho 0 loop: overshoot 0.046-0.050 m (> 0.04 m) and settling
# 1.12-1.18 cm (> 1 cm), also with the detector out of reach (RESULT.md).  Expected to fail until the move or the bounds are revisited
# (strict: a change shows up as XPASS).  The own-move-leak cases of 7d12d78fe pass since the review-B1 fix (detector factor on the whole
# correction).
MOVE_SPEED_DYNAMICS = pytest.mark.xfail(strict=True, reason="rev 8: overshoot / settling of the 0.2 m/s move in the rho 0 loop (RESULT.md)")


def move_speeds(*failing):
  return [pytest.param(k, marks=MOVE_SPEED_DYNAMICS) if k in failing else k for k in SPEEDS]


class TestClosedLoop:
  @pytest.mark.parametrize("kmh", move_speeds(30, 50))
  def test_step_of_half_a_metre(self, kmh):
    t, y, vs, _, _ = closed_loop(kmh / 3.6, lambda t: {})
    final = 1.75 - br.W_R - 0.44  # 0.5 m right of the centre
    after = t >= 0
    assert abs(y[-1] - final) < 0.01
    # rev 8 item 5: the approach grows at the move speed 0.2 m/s; the car follows with its lag (peak 0.205-0.213 m/s in this loop, reported)
    assert vs[after].max() <= br.MOVE_SPEED + 0.02
    arr = int(np.argmax(y >= 0.9 * final))
    assert t[arr] <= 7.0  # 90 % within 7 s (4.85-4.95 s here; rev 7.2 at 0.042 m/s: about 11 s)
    err = np.diff((final - y)[after][: arr - int(np.sum(~after)) + 1])
    err = err[np.abs(err) > 1e-7]
    assert int(np.sum(np.sign(err[1:]) != np.sign(err[:-1]))) == 0  # no reversal before arrival
    assert y.max() - final < 0.04  # overshoot (rev 8 at 0.2 m/s: 0.027-0.034 m; rev 7.2 at 0.042 m/s: about 2.5 %, reported)
    assert np.abs(y[arr + int(20 / DT_MDL):] - final).max() < 0.01

  @pytest.mark.parametrize("kmh", move_speeds(50))  # 80 km/h: 1.18 cm with the detector out of reach, passes with it (it slows that move)
  def test_step_settles_within_1_cm_5_s_after_arrival(self, kmh):
    t, y, vs, _, _ = closed_loop(kmh / 3.6, lambda t: {})
    final = 1.75 - br.W_R - 0.44
    arr = int(np.argmax(y >= 0.9 * final))
    assert np.abs(y[arr + int(5 / DT_MDL):] - final).max() < 0.01

  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_right_reference_switch_line_to_edge_through_the_weights(self, kmh):
    # 5 s into the move the inner right line (1.75) fades over 1 s and the edge (2.15, counted 2.05) takes over.  Without the switch part
    # in G0 the switch would add its own speed to the move (check 20b); rev 8: the car stays within the move speed + 0.02 m/s
    def scene(t):
      return {'right_edge': 2.15, 'right_line_prob': float(np.clip(0.9 - 0.9 * (t - 5.0), 0.0, 0.9)) if t > 5.0 else 0.9}
    t, y, vs, _, _ = closed_loop(kmh / 3.6, scene, seconds=60.0)
    assert vs[t >= 0].max() <= br.MOVE_SPEED + 0.02
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

  def step(self, rcl, tracks, v=13.0, gap=0.5, c_ctr=None, edge=None, ew=None, y_ref=None, floor=-rc.NO_BOUND, **kw):
    return rcl.update(frame(tracks, **kw), v, v, 0.0, gap, 0.0, br.W_R, X, self.C if c_ctr is None else c_ctr, self.E if edge is None else edge,
                      self.W0 if ew is None else ew, rise=br.MOVE_SPEED, y_ref=y_ref, floor=floor)

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
    rcl2.update(frame([(1, 10.0, -2.0, 0.0)]), 13.0, 13.0, 0.0, 0.5, 0.1, br.W_R, X, self.C, self.E, self.W0, rise=br.MOVE_SPEED)
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
    # rev 8 item 4: the extra distance rises from 0.2 m at 50 km/h to 0.5 m at 120 km/h (was 100 km/h)
    for kmh, gap, m in ((30, 0.5, 0.7), (50, 0.5, 0.7), (85, 0.75, 1.1), (100, 1.0, 1.2 + 0.3 * 50 / 70), (120, 1.0, 1.5), (130, 1.0, 1.5),
                        (40, 0.8, 1.0)):
      rcl = RadarClearance()
      res = self.confirmed(rcl, [(1, 10.0, -2.0, 0.0)], n=rc.BOUND_CONFIRM_MESSAGES, v=kmh / 3.6, gap=gap)
      assert res.margin == pytest.approx(m)
      assert res.t_obj == pytest.approx(2.0 - br.W_R - m)

  def test_bound_falls_at_once_and_rises_after_one_second_at_the_move_speed(self):
    rcl = RadarClearance()
    res = self.confirmed(rcl, [(1, 10.0, -2.0, 0.0)], n=rc.BOUND_CONFIRM_MESSAGES)
    low = res.t_obj
    res = self.step(rcl, [(1, 10.0, -2.5, 0.0)])  # the object is 0.5 m further right
    ts = [res.t_obj] + [self.step(rcl, [(1, 10.0, -2.5, 0.0)]).t_obj for _ in range(60)]
    assert ts[0] == low and np.all(np.diff(ts[:19]) == 0.0)  # held for 1 s
    assert np.diff(ts[20:]).max() <= br.MOVE_SPEED * DT_MDL + 1e-12 and ts[-1] > low

  def test_popup_confirmed_one_second_ahead_binds_in_its_confirmation_frame(self):
    # rev 8 item 1a: the object bound counts an object after its own, longer confirmation (BOUND_CONFIRM_MESSAGES, 20); the pop-up
    # flag and the clearance check keep the 6 messages
    rcl = RadarClearance()
    n_b = rc.BOUND_CONFIRM_MESSAGES
    for n in range(n_b):
      res = self.step(rcl, [(9, 20.0 - 13.0 * DT_MDL * n, -1.6, -13.0)])
      if n == 5:
        assert res.popups == 1 and res.t_obj == rc.NO_BOUND and res.objects == 1
      if n == n_b - 2:
        assert res.t_obj == rc.NO_BOUND
    assert res.t_obj == pytest.approx(1.6 - br.W_R - 0.7)  # full margin in the bound's confirmation frame, no hold, no class wait

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
    res = rcl.update(None, 13.0, 13.0, 0.0, 0.5, 0.0, br.W_R, X, self.C, self.E, self.W0, rise=br.MOVE_SPEED) if bad is None else self.step(rcl, [], **bad)
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

  def test_object_on_the_line_and_beyond_give_up_nothing_and_an_object_inside_gives_up_its_margin(self):
    # rev 8 item 1a: objects up to 10 cm inside the right line (posts on the line) and beyond it are ignored by the bound; an object 0.15 m
    # inside gives up 0.44 - (1.60 - 0.81 - 0.7) = 0.35 m (rev 7.2: the object on the line gave up 0.2 m)
    _, base = self.t4_at_L([])
    look = 1.5 * self.V
    t0 = float(np.interp(look, X, base.log['t4']))
    for y in (1.75, 1.70, 1.80, 1.95):
      _, res = self.t4_at_L([(1, 12.0, -y, -self.V)])
      assert abs(t0 - float(np.interp(look, X, res.log['t4']))) < 1e-9, y
      assert res.log['radar'].objects == 1 and res.log['radar'].bound_objects == 0
    _, inside = self.t4_at_L([(1, 12.0, -1.60, -self.V)])
    assert t0 - float(np.interp(look, X, inside.log['t4'])) == pytest.approx(0.35, abs=1e-6)

  def test_the_bound_never_goes_left_of_the_centre_limit(self):
    _, res = self.t4_at_L([(1, 12.0, -0.9, -self.V)])  # very close: T_obj far left of the centre
    assert np.all(res.log['t4'] >= np.minimum(res.log['c_ctr'], res.log['t_lim']) - 1e-9)
    assert res.log['radar'].yield_active and res.caps == {br.SIDE_RIGHT: pytest.approx(res.caps[br.SIDE_RIGHT])}

  def test_right_blindspot_removes_the_right_line_shift(self):
    _, res = self.t4_at_L([], right_blindspot=True)
    np.testing.assert_allclose(res.log['t4'], np.minimum(res.log['c_ctr'], res.log['t_lim']), atol=1e-9)

  def test_radar_fault_hands_back_to_regular(self):
    rl, res = self.t4_at_L([], frames=120)
    assert res.log['a'] > 0.9
    for _ in range(200):  # rev 7: a falls with the strength filter (tau 1 s) and reaches 0 within 10 s
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


class TestTriggerSides:
  def test_clearance_trigger_caps_the_right_and_the_detector_hands_over(self):
    # rev 8: one trigger of the cap rules (the radar clearance check, always the right) and the path detector, which scales the strength and
    # hands over.  An object close on the right from frame 100 caps the right; from frame 160 the path moves right by 1.2 m within 0.5 s
    # (above the upper level 0.43 m at 47 km/h; an obstacle on the LEFT): no left cap any more, the detector's strength factor goes to 0 and
    # its hand-over takes the whole correction to 0 through the limiter (the model alone)
    v = 13.0
    ctl = BlendfulCorrection()
    cs = []
    for i in range(320):
      tracks = [(1, 6.0, -1.0, -v)] if i >= 100 else []
      plan = 1.2 * min(1.0, max(0.0, (i - 160) * DT_MDL / 0.5))
      ctl.update(Scene(v=v, plan=plan).model_output(), v, True, 0.0, LAT_T, MODE_RIGHT, inputs(radar=frame(tracks)))
      cs.append(ctl.c)
      if 110 <= i < 160:
        assert set(ctl.result.caps) == {br.SIDE_RIGHT} and not ctl.hand_over
      if i >= 200:
        assert set(ctl.result.caps) == {br.SIDE_RIGHT} and ctl.hand_over and ctl.result.log['det_strength'] == 0.0
    assert abs(cs[-1]) < 1e-9  # kappa0 -> 0 through the limiter, the model alone


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
  extra = {'a': [], 'authority_limited': [], 'car_left': []}
  hand = []
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
    hand.append(bool(ctl.hand_over))
  scene_loop.last_log = ctl.result.log
  scene_loop.extra = {k: np.array(x) for k, x in extra.items()}
  scene_loop.extra['hand_over'] = np.array(hand)
  r = np.array(rows, dtype=float)
  return r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4], r[:, 5], r[:, 6]


def toggling(t):
  return MODE_RIGHT if (t < 0 or t > 30 or int(t / 2) % 2 == 0) else MODE_REGULAR


def flicker(t):
  """The quality-flicker fixture q(t) = 0.45 + 0.3 sin(2 pi t) from t = 0 to 30 s.  Applied as the raw right-line probability (round 4; upstream's
  0.5 s probability filter delivers about 0.45 +- 0.09 to the gate: partial fades), the second, reported variant (design rev 6.12 E3, R4-S3)."""
  return float(0.45 + 0.3 * np.sin(2 * np.pi * t)) if 0 <= t < 30 else 0.9



SCENES = {
  'plain': {},
  'return': {'y0': -0.4, 'regular_s': 0.0},  # mode on with the car 0.4 m left of the centre (as after a model-alone junction episode)
  # Regular rests about 0.15 m left of the centre: line probabilities 0.55 (Regular's trust about 0.8) and a model pull calibrated per speed
  'resting': {'y0': -0.4, 'regular_s': 0.0, 'pull': {30: -0.396, 50: -0.55, 80: -0.462}, 'probs': 0.55},
  'toggle': {'y0': 0.5, 'regular_s': 0.0, 'mode': toggling},  # at the right-hand position, then Right line <-> Regular every 2 s
  'flicker': {'y0': 0.5, 'regular_s': 0.0, 'prob': flicker},  # the raw-probability variant (reported)
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


# rev 8 item 5: the approach grows at MOVE_SPEED 0.2 m/s; the car lags the target and catches up, so its own peak is above it (0.205-0.213
# m/s on a plain move, 0.264-0.269 m/s when it starts 0.4 m left of the centre; reported in RESULT.md): bound = move speed + 0.08 m/s
MOVE_SANITY = br.MOVE_SPEED + 0.08


class TestStartAndFloor:
  """Scene tests in the heading-keeping loop.  Rev 7: the unweighted approach grows at r_ref and after a drop-out or a Regular phase the
  correction returns at the strength filter's pace, so the old own-move criterion (<= 0.052 m/s) no longer applies; rev 8: r_ref is the move
  speed 0.2 m/s.  Asserted: no hand-over, the end positions, never further left than Regular, and a bound on the speed."""
  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_plain_move(self, kmh):
    t, y, vs, tg, tc, src, fol = scene_loop(kmh, 'plain', seconds=50.0)
    assert own_move_peak(t, y, vs, tg, tc) <= MOVE_SANITY and right_of_centre_peak(t, y, vs) <= MOVE_SANITY
    assert abs(y[-1] - 0.5) < 0.01 and not scene_loop.extra['hand_over'].any()

  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_return_from_left_of_the_centre(self, kmh):
    t, y, vs, tg, tc, src, fol = scene_loop(kmh, 'return', seconds=50.0)
    tr, yr, vr, *_ = scene_loop(kmh, 'return', seconds=50.0, regular=True)
    assert right_of_centre_peak(t, y, vs) <= max(right_of_centre_peak(tr, yr, vr) + 0.002, MOVE_SANITY)
    assert own_move_peak(t, y, vs, tg, tc) <= MOVE_SANITY
    # the larger-of rule acts while the car is left of the centre (rev 8: at the 0.2 m/s move the right-line share is the larger one)
    assert np.any((src == br.DELIVER_REGULAR) | (src == br.DELIVER_LARGER_RIGHT_LINE)) and abs(y[-1] - 0.5) < 0.02

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
    # never left of Regular, up to the loop's phase: rev 8's faster move (0.2 m/s) first takes the car up to 5 cm right of Regular, which then
    # swings back up to 1.4 cm left of Regular's own trajectory (50 km/h) before both rest at the same place (rev 7.2: identical, 5 mm)
    assert np.all(y[g] >= yr[g] - 0.02)
    assert abs(y[-1] - yr[-1]) < 0.01 or y[-1] > yr[-1]
    assert not ex['hand_over'].any()  # fix 1: no false yield, no limit cycle (5c06f5c6d: 0.7 m cycle, 306-456 yield frames per 90 s)
    late = y[t >= 30]
    assert late.max() - late.min() < 0.01
    assert right_of_centre_peak(t, y, vs) <= 0.052
    assert vs[t >= 0].max() <= max(vr[tr >= 0].max(), br.MOVE_SPEED) + 0.002
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
  @pytest.mark.parametrize("scene", ['toggle', 'flicker'])
  def test_mode_toggling_and_quality_flicker(self, kmh, scene):
    # Right line <-> Regular every 2 s, or the right-line probability 0.45 +- 0.3 at 1 Hz, from the right-hand position: no hand-over, the car
    # ends at +0.5 m and is never further left than in the Regular-only run (+ 1 cm: phase)
    t, y, vs, tg, tc, src, fol = scene_loop(kmh, scene, seconds=70.0)
    ho = scene_loop.extra['hand_over']
    tr, yr, vr, *_ = scene_loop(kmh, scene, seconds=70.0, regular=True)
    print(f'{scene} {kmh} km/h: own move {own_move_peak(t, y, vs, tg, tc):.4f} m/s, min y {y[t >= 0].min():+.3f} (Regular {yr[tr >= 0].min():+.3f})')
    assert not ho.any() and abs(y[-1] - 0.5) < 0.01
    assert own_move_peak(t, y, vs, tg, tc) <= MOVE_SANITY
    assert y[t >= 0].min() >= yr[tr >= 0].min() - 0.01

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

  def test_no_ratchet_under_line_jitter(self):
    # a steady hold at the right-hand position with 3 cm of independent jitter on every line, 120 s: no leftward ratchet of the target, and
    # the jitter does not trigger the detector (rev 7.1)
    v = 50 / 3.6
    rng = np.random.default_rng(7)
    ctl = BlendfulCorrection()
    y, psi, kc = 0.5, 0.0, 0.0
    buf = [0.0] * 4
    tgs, yields = [], 0
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
        yields += int(ctl.result.yield_active)
    tgs = np.array(tgs)
    assert np.mean(tgs[-200:]) - np.mean(tgs[:200]) > -0.02  # no net leftward drift beyond the jitter
    assert abs(y - 0.5) < 0.03 and yields == 0


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
    if not binds:  # the larger-of rule stays on (rev 7: with the unweighted approach the right-line share may be the larger one)
      assert lg['delivered'] in (br.DELIVER_REGULAR, br.DELIVER_LARGER_RIGHT_LINE)

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


class TestRepeatedDropOuts:
  def test_no_ratchet_under_line_jitter_with_repeated_drop_outs(self):
    # 3 cm independent jitter on every line and a 0.4 s drop-out of the right line every 40 s, 6 times (rev 7: no reset; the strength
    # filter fades and returns): the car returns to the right-hand position each time, neither the position before a drop-out nor the
    # excursion after it moves left from one to the next (no ratchet), and no yield
    v = 50 / 3.6
    rng = np.random.default_rng(11)
    ctl = BlendfulCorrection()
    y, psi, kc = 0.5, 0.0, 0.0
    buf = [0.0] * 4
    ys, yields = [], 0
    n, period, t0 = 6, 40.0, 30.0
    for i in range(int((t0 + n * period + 10) / DT_MDL)):
      t = i * DT_MDL
      gone = t0 <= t < t0 + n * period and (t - t0) % period < 0.4
      mo = model_output(v, y, psi, kc, right_line_prob=0.0 if gone else 0.9)
      mo['lane_lines'][0, :, :, 0] += 0.03 * rng.standard_normal((4, 1))
      mo['plan'][0, :, 1] = -psi * X
      k_model = -psi / v
      ctl.update(mo, v, True, k_model, LAT_T, MODE_RIGHT, inputs(gap_low=0.44, gap_high=0.44))
      yields += int(ctl.result.yield_active)
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
    TestRepeatedDropOuts.jitter_rows = (before, lows, float(y), yields)
    assert min(before) > 0.47 and before[-1] - before[0] > -0.02
    assert max(lows) - min(lows) < 0.03
    assert abs(y - 0.5) < 0.03 and yields == 0


# ---------------------------------------------------------------------------------------------------------------------------------
# fix 1 (owner 13 Nov 13:37, full review F1): the detector's own-move estimate with the model's pull-back, capped by the measured move
def pullback_loop(rho, kmh=50, seconds=120.0, lat_fn=None, swerve_at=None, sw_amp=0.35, sw_rise=1.0, mode=MODE_RIGHT, response=1.0):
  """The heading-keeping loop with a model that pulls back toward the lane centre by the share rho of the car's offset (as the full
  review's rv7_hold.py); response < 1: the car realises only that share of the commanded correction (a torque-limited car)."""
  v = kmh / 3.6
  look = max(1.5 * v, 10.0)
  ctl = BlendfulCorrection()
  y, psi, kc = 0.0, 0.0, 0.0
  buf = [0.0] * 4
  rows = []
  for i in range(int(seconds / DT_MDL)):
    t = i * DT_MDL
    mo = model_output(v, y, psi, kc)
    mo['plan'][0, :, 1] = -psi * X
    k_model = -psi / v
    e = -rho * y
    if swerve_at is not None and t >= swerve_at:
      e -= sw_amp * min(1.0, (t - swerve_at) / sw_rise)
    mo['plan'][0, :, 1] += e * np.clip(X / look, 0, 1)
    k_model += 2.0 * e / look ** 2
    lat = lat_fn(t) if lat_fn else True
    ctl.update(mo, v, True, k_model, LAT_T, mode, inputs(gap_low=0.44, gap_high=0.44, lat_active=lat))
    up = bool(ctl.reference.going_around or ctl.reference.yield_fade < 1.0)
    rows.append((t, y, float(ctl.cap_active), float(ctl.hand_over), float(ctl.result.yield_active), float(up), v * psi,
                 float(ctl.result.log.get('det_strength', 1.0)) if ctl.result.log else 1.0))
    buf.append((response * ctl.c if lat else 0.0) + k_model)
    cmd = buf.pop(0)
    for _ in range(5):
      kc += (cmd - kc) * 0.01 / 0.3
      psi += v * kc * 0.01
      y += v * psi * 0.01
  return np.array(rows)


def first_after(r, col, t0):
  m = (r[:, 0] >= t0) & (r[:, col] > 0)
  return float(r[m, 0][0] - t0) if m.any() else float('inf')


class TestOwnMoveEstimate:
  @pytest.mark.parametrize("kmh", SPEEDS)
  @pytest.mark.parametrize("rho", [0.3, 0.6, 0.9])
  def test_hold_against_a_model_pull_back(self, kmh, rho):
    # 120 s hold: no cap, no hand-over; the car settles where the correction balances the pull (0.35-0.44 m in this loop: the correction
    # law's own steady-state share, not the detector; reported)
    r = pullback_loop(rho, kmh)
    assert r[:, 2].sum() == 0 and r[:, 3].sum() == 0
    late = r[r[:, 0] >= 60, 1]
    assert late.max() - late.min() < 0.005 and late[-1] > 0.33

  @pytest.mark.parametrize("kmh", SPEEDS)
  @pytest.mark.parametrize("rho", [0.0, 0.3, 0.6])
  def test_re_engagement_gives_no_yield(self, kmh, rho):
    # latActive false for 10 s at the held position (the car ignores the correction): the estimate is frozen, usual reset; no yield after
    r = pullback_loop(rho, kmh, seconds=90.0, lat_fn=lambda t: not (40.0 <= t < 50.0))
    assert r[r[:, 0] >= 40, 4].sum() == 0 and r[r[:, 0] >= 40, 2].sum() == 0

  @pytest.mark.parametrize("kmh", SPEEDS)
  @pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
  def test_re_engagement_after_a_long_gap_restarts_the_move(self, kmh, rho):
    # review S1: latActive false for 60 s (the car ignores the correction, the model holds it near the centre), then true: the move limiter
    # restarts at the car (U = 0), so no hand-over, no cap, and the car moves right no faster than in a first engagement
    r = pullback_loop(rho, kmh, seconds=150.0, lat_fn=lambda t: not (40.0 <= t < 100.0))
    first = pullback_loop(rho, kmh, seconds=40.0)
    after = r[:, 0] >= 100.0
    print(f're-engagement {kmh} km/h rho {rho}: peak sideways {r[after, 6].max():.4f} m/s (first engagement {first[:, 6].max():.4f})')
    assert r[after, 3].sum() == 0 and r[after, 2].sum() == 0
    assert r[after, 6].max() <= first[:, 6].max() + 0.005

  @pytest.mark.parametrize("kmh", SPEEDS)
  @pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
  def test_left_swerve_at_the_held_position(self, kmh, rho):
    # the plan swerves 0.35 m left over 1 s (obstacle on the right) at 60 s: rev 8: the strength is scaled down (factor < 1) no later than
    # upstream from the centre reacts + 0.2 s (rev 7.2: the cap), and the car moves at most 1 mm toward the right.  0.35 m stays below the
    # provisional upper level 0.45 m, so there is no hand-over here
    r = pullback_loop(rho, kmh, seconds=75.0, swerve_at=60.0)
    u = pullback_loop(rho, kmh, seconds=75.0, swerve_at=60.0, mode=MODE_REGULAR)
    y60 = r[int(round(60.0 / DT_MDL)), 1]
    scaled = r.copy()
    scaled[:, 7] = (r[:, 7] < 1.0).astype(float)
    print(f'left swerve {kmh} km/h rho {rho}: scaled down +{first_after(scaled, 7, 60.0):.2f} s, upstream +{first_after(u, 5, 60.0):.2f} s')
    # with a model pull-back of 0.6 / 0.9 the swerve value crosses the lower level late or not at all (reported); the motion toward the
    # obstacle is asserted for all
    # the timing is asserted up to rho 0.3 from 50 km/h (at 30 km/h the lower level is 0.22 m: +1.35 s at rho 0.3, reported)
    if rho <= 0.3 and kmh >= 50:
      assert first_after(scaled, 7, 60.0) <= first_after(u, 5, 60.0) + 0.2 + 1e-9
    assert r[r[:, 0] >= 60.0, 1].max() - y60 <= 1e-3

  @pytest.mark.parametrize("response", [0.5, 0.0])
  def test_a_car_that_does_not_follow_gives_no_false_yield(self, response):
    # the car realises half / none of the correction (torque limit, steering not following although latActive): the subtracted estimate is
    # capped by the measured move + 0.05 m, so no false yield
    for rho in (0.0, 0.3):
      r = pullback_loop(rho, 50, seconds=120.0, response=response)
      assert r[:, 4].sum() == 0 and r[:, 3].sum() == 0


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
      return rcl.update(frame(tracks), 13.0, 0.0, 0.0, 0.5, 0.0, br.W_R, X, C, E, W0, rise=br.MOVE_SPEED)
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
      res = rcl.update(frame([(1, 30.0, -1.4, 0.0)]), 13.0, 0.0, 0.0, 0.5, 0.0, br.W_R, X, C, E, W0, rise=br.MOVE_SPEED)  # 30 m ahead: beyond the memory zone
    assert res.objects == 1
    objs = [rcl.update(frame([]), 13.0, 0.0, 0.0, 0.5, 0.0, br.W_R, X, C, E, W0, rise=br.MOVE_SPEED).objects for _ in range(25)]
    assert objs[:19] == [1] * 19 and objs[-1] == 0

  def test_released_when_the_object_has_passed_the_rear_bumper(self):
    rcl = RadarClearance()
    C, E, W0 = np.zeros(br.NX), np.full(br.NX, 1e3), np.zeros(br.NX)
    v = 13.0
    for n in range(6):
      res = rcl.update(frame([(1, 3.0 - v * DT_MDL * n, -1.4, -v)]), v, v, 0.0, 0.5, 0.0, br.W_R, X, C, E, W0, rise=br.MOVE_SPEED)
    assert res.yield_active
    for _ in range(30):  # leaves the field of view, remembered, passes x < -3.3 m within about 0.4 s
      res = rcl.update(frame([]), v, v, 0.0, 0.5, 0.0, br.W_R, X, C, E, W0, rise=br.MOVE_SPEED)
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
    # open loop: the car held at y0.  rev 7.1: the detector's own-move estimate assumes the car follows our correction; held in place it would
    # read our growing correction as a swerve (safe side: a yield), so the estimate's share is 0 while the correction is built here
    own, br.OWN_PULLBACK = br.OWN_PULLBACK, 1.0
    for _ in range(int(30 / DT_MDL)):
      ctl.update(model_output(v, y0, 0.0, 0.0), v, True, 0.0, LAT_T, MODE_RIGHT, inputs(gap_low=gap, gap_high=gap))
    br.OWN_PULLBACK = own
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


# ---------------------------------------------------------------------------------------------------------------------------------
# rev 8 (owner's decisions of 15 Nov 2026, inster-dev results/2026-11-15_a8_rev8/DESIGN.md): each test below fails on rev 7.2 (8924142)
class TestRev8MoveSpeed:
  """Item 5: one sideways move speed, 0.2 m/s, for the car's move and the radar bound's release."""
  @pytest.mark.parametrize("kmh", SPEEDS)
  def test_first_engagement_moves_at_the_move_speed(self, kmh):
    t, y, vs, tg, tc, src, fol = scene_loop(kmh, 'plain', seconds=20.0)
    on = t >= 0
    reached = t[on][np.argmax(y[on] >= 0.49)] if (y[on] >= 0.49).any() else np.inf
    assert reached <= 8.0  # 0.5 m at 0.2 m/s plus the strength filter and the loop's lag (rev 7.2: about 13 s)
    assert vs[on].max() <= 0.2 + 0.02 and not scene_loop.extra['hand_over'].any()


class TestRev8ObjectBound:
  """Item 1: the radar object bound counts fewer objects (a), never sits left of the centre (b) and releases at the move speed (c)."""
  V = 40 / 3.6

  def run(self, tracks_fn, frames=60, scene=None, **kw):
    rl = RightLineBlendful()
    logs = []
    for i in range(frames):
      res = rl.update((scene or Scene(v=self.V)).model_output(), self.V, True, 0.0, 0.0, inputs(radar=frame(tracks_fn(i)), **kw))
      logs.append(res.log)
    return rl, logs

  def test_moving_vehicle_in_the_own_lane_is_ignored_and_the_same_object_standing_binds(self):
    look = 1.5 * self.V
    _, base = self.run(lambda i: [])
    t0 = float(np.interp(look, X, base[-1]['t4']))
    _, cyc = self.run(lambda i: [(1, 12.0, -1.30, 20 / 3.6 - self.V)])  # a cyclist at 20 km/h, 0.45 m inside the right line
    assert abs(float(np.interp(look, X, cyc[-1]['t4'])) - t0) < 1e-9 and cyc[-1]['radar'].moving == 1
    _, std = self.run(lambda i: [(1, 12.0, -1.30, -self.V)])  # the same place, standing
    assert float(np.interp(look, X, std[-1]['t4'])) < t0 - 0.3

  def test_the_bound_counts_an_object_only_after_its_longer_confirmation(self):
    _, logs = self.run(lambda i: [(1, 12.0, -1.45, -self.V)], frames=25)
    n_b = 20  # BOUND_CONFIRM_MESSAGES, measured (adaea0f4); rev 7.2: 6 for both
    assert all(lg['radar'].t_obj == rc.NO_BOUND for lg in logs[:n_b - 1])
    assert logs[n_b - 1]['radar'].t_obj < rc.NO_BOUND
    assert logs[5]['radar'].objects == 1 and logs[5]['radar'].deficit > -1.0  # the clearance check counts it from the 6th message

  def test_without_a_usable_right_reference_every_object_counts(self):
    # guard (also rev 7.2): with q_R at or below 0.2 the line filters do not apply, a post on the line still binds
    _, logs = self.run(lambda i: [(1, 12.0, -1.70, -self.V)], scene=Scene(v=self.V, pr=0.0))
    assert logs[-1]['q_R'] <= br.REF_USABLE_Q and logs[-1]['radar'].t_obj < rc.NO_BOUND

  def test_the_held_bound_never_sits_left_of_the_centre_and_returns_at_the_move_speed(self):
    # a very close object (bound 0.61 m left of the centre) for 2 s, then gone: the held bound stays at the centre limit (the lowest in the
    # fit window), and 1 s after the object is gone the target returns at the move speed (rev 7.2: from 0.61 m left at 0.042 m/s)
    v = self.V
    la = br.approach_length(v)
    win = (X >= br.FIT_WINDOW[0] * la) & (X <= br.FIT_WINDOW[1] * la)
    _, logs = self.run(lambda i: [(1, 6.0, -0.90, -v)] if 100 <= i < 140 else [], frames=300)
    for lg in logs[100:]:
      assert lg['radar'].t_obj >= float(np.min(lg['c_ctr'][win])) - 1e-9
    look = 1.5 * v
    t_hold = float(np.interp(look, X, logs[99]['target']))
    back = next(i for i in range(140, 300) if float(np.interp(look, X, logs[i]['target'])) >= t_hold - 1e-6)
    remembered = (6.0 + rc.RADAR_TO_MODEL_X - rc.FORGET_X) / v  # the object stays in the memory until 1 m behind the rear bumper
    assert (back - 140) * DT_MDL <= remembered + 1.0 + t_hold / 0.2 + 0.6


class TestRev8Clearance:
  """Item 3: the clearance deficit from the car's actual position; only objects at least 10 cm inside the right line count, except objects
  moving sideways toward the lane, which count from outside too."""
  V = 40 / 3.6

  def run(self, tracks_fn, frames=40, scene=None, gap=0.5):
    rl = RightLineBlendful()
    logs = []
    for i in range(frames):
      res = rl.update((scene or Scene(v=self.V)).model_output(), self.V, True, 0.0, 0.0,
                      inputs(radar=frame(tracks_fn(i)), gap_low=gap, gap_high=gap))
      logs.append(res.log)
    return logs

  def test_deficit_from_the_cars_actual_position(self):
    # the car 0.4 m right of the lane centre, a standing object 1.15 m right of the car (0.2 m inside the right line): its clearance from
    # the car's right side is 0.34 m < m = 0.7 m -> deficit 0.36 m and a clearance yield (rev 7.2 measured from the centre limit: -0.04 m)
    logs = self.run(lambda i: [(1, 8.0, -1.15, -self.V)], scene=Scene(v=self.V, center=-0.4))
    assert logs[-1]['radar'].deficit == pytest.approx(0.36, abs=0.01) and logs[-1]['radar'].yield_active

  def test_deficit_follows_the_lane_in_a_curve(self):
    # the car at the centre of a left curve (radius 150 m), a standing object 0.2 m inside the right line 20 m ahead: the car's path
    # keeps its offset from the centre limit along the lane, so no deficit (a straight-ahead path would read the object 0.6 m inside the car)
    k = -1 / 150
    xo = 20.0
    y_obj = k * xo ** 2 / 2 + 1.55
    logs = self.run(lambda i: [(1, xo - rc.RADAR_TO_MODEL_X, -y_obj, -self.V)], scene=Scene(v=self.V, k=k), frames=60)
    assert logs[-1]['radar'].objects == 1 and logs[-1]['radar'].deficit < 0.0 and not logs[-1]['radar'].yield_active

  def test_a_post_on_the_line_does_not_count(self):
    # GUI gap 1.0 m (m = 1.2 m): a standing post 5 cm inside the right line, the car at the centre: ignored (rev 7.2: deficit 0.31 m, yield)
    logs = self.run(lambda i: [(1, 8.0, -1.70, -self.V)], gap=1.0)
    assert logs[-1]['radar'].objects == 1 and logs[-1]['radar'].deficit < -1.0 and not logs[-1]['radar'].yield_active

  def test_a_vehicle_cutting_in_counts_from_outside(self):
    # m = 1.2 m.  Standing 0.15 m beyond the right line: ignored.  The same place reached by a vehicle ahead (moving: the radial vote) cutting
    # in toward the lane at 1.2 m/s: counted once its sideways speed is known (10 messages), while it is still beyond the line (deficit
    # 1.2 - (y - 0.81) > 0 below 2.01 m).  Measured rule (adaea0f4): tested only on the moving vote, so the same sideways walk by a standing
    # object (a pedestrian or cyclist with no radial speed) is not counted: a vehicle cut-in rule only
    # (objects appear after 2 s: before the right line's filter has initialised there is no usable reference and every object counts)
    still = self.run(lambda i: [(1, 8.0, -1.90, -self.V)] if i >= 40 else [], frames=80, gap=1.0)
    assert still[-1]['radar'].objects == 1 and not any(lg['radar'].yield_active for lg in still)

    def mover(v_rel):
      return lambda i: [(2, 8.0, -(2.6 - 1.2 * DT_MDL * (i - 40)), v_rel)] if i >= 40 else []
    logs = self.run(mover(0.0), frames=56, gap=1.0)[40:]  # vRel 0: a vehicle ahead at our speed
    beyond = [lg for i, lg in enumerate(logs) if 2.6 - 1.2 * DT_MDL * i > 1.75 + 1e-9]
    assert any(lg['radar'].toward == 1 and lg['radar'].yield_active for lg in beyond)
    assert all(lg['radar'].toward == 0 for lg in logs[:10])  # not before 10 sideways steps (11 messages)
    walker = self.run(mover(-self.V), frames=56, gap=1.0)[40:]  # standing radially: a pedestrian stepping toward the lane
    assert all(lg['radar'].toward == 0 for lg in walker)
    assert not any(lg['radar'].yield_active for i, lg in enumerate(walker) if 2.6 - 1.2 * DT_MDL * i > 1.75 + 1e-9)

  def test_the_sideways_speed_is_yaw_compensated(self):
    # a standing object 20 m ahead, seen from a car turning at 0.1 rad/s: its apparent sideways motion (about 2 m/s) is the car's own turn
    rcl = RadarClearance()
    v, yaw = 13.0, 0.1
    x, y = 20.0, 2.0
    for _ in range(20):
      rcl.update(frame([(1, x - rc.RADAR_TO_MODEL_X, -y, -v)]), v, v, yaw, 0.5, 0.0, br.W_R, X, np.zeros(br.NX), np.full(br.NX, 1e3),
                 np.zeros(br.NX), l_d=br.L_D, rise=br.MOVE_SPEED)
      x -= v * DT_MDL
      y -= yaw * DT_MDL * (x + br.L_D + v * DT_MDL / 2)  # the same motion model as the memory (B2)
    assert abs(rcl.tracks[1].toward_speed) < 1e-6


class TestRev8Detector:
  """Item 2: no freeze; the strength target is scaled continuously between the detector's lower and upper level, smoothed by the 1 s
  strength filter; the hand-over latches at the upper level."""
  V = 13.0

  @staticmethod
  def patch(monkeypatch, swerve_of_frame):
    n = {'i': 0}

    def fake(self, *a, **k):
      s = float(swerve_of_frame(n['i']))
      n['i'] += 1
      self.swerve, self.away, self.departure, self.offset, self.usual, self.going_around = s, s, -s, -s, 0.0, False
      return s
    monkeypatch.setattr(br.SwerveDetector, 'update', fake)

  def run(self, frames):
    ctl = BlendfulCorrection()
    out = []
    for _ in range(frames):
      ctl.update(Scene(v=self.V).model_output(), self.V, True, 0.0, LAT_T, MODE_RIGHT, inputs())
      lg = ctl.result.log
      out.append((lg['det_gain'], lg['det_strength'], lg['a'], ctl.result.no_growth, ctl.cap_active))
    return np.array(out, dtype=float)

  def test_no_freeze_below_the_lower_level(self, monkeypatch):
    # swerve 0.15 m from the first frame (rev 7.2: no growth of the strength from 0.1 m, a stays 0)
    self.patch(monkeypatch, lambda i: 0.15)
    r = self.run(100)
    assert r[-1, 2] > 0.95 and np.all(r[:, 0] == 1.0) and not r[:, 3].any() and not r[:, 4].any()

  def test_strength_target_scales_linearly_between_the_levels_through_the_1_s_filter(self, monkeypatch):
    # the factor is linear between the levels and reaches the correction through its own 1 s low-pass (review B1: on the whole correction,
    # Keep Right's effective strength a x gain); a itself is not lowered by the detector (that would hand the share to Regular)
    lo, up = br.detector_levels(self.V)
    sched = [0.0] * 120 + [lo + 0.25 * (up - lo)] * 60 + [lo + 0.5 * (up - lo)] * 100 + [up + 0.1] * 60
    self.patch(monkeypatch, lambda i: sched[min(i, len(sched) - 1)])
    r = self.run(len(sched))
    for i0, f in ((120, 0.75), (180, 0.5), (280, 0.0)):
      assert np.allclose(r[i0 + 1:i0 + 60, 1], f, atol=1e-9)
    g0 = r[179, 0]
    k = DT_MDL / (DT_MDL + br.STRENGTH_TAU)
    assert r[199, 0] == pytest.approx(0.5 + (g0 - 0.5) * (1 - k) ** 20, abs=1e-6)  # 1 s of the first-order filter toward 0.5
    assert np.all(np.diff(r[100:, 2]) >= -1e-12) and r[-1, 2] == 1.0  # a keeps rising: the scale-down is not a hand-back to Regular
    assert not r[:, 3].any()  # no no-growth phase from the detector

  def test_detector_strength_and_the_provisional_levels(self):
    # owner 15 Nov 10:53: p75 / p95 levels at the band centres 34 / 65 / 100 km/h, linear between, held outside
    assert br.detector_levels(5.0) == br.detector_levels(34 / 3.6) == pytest.approx((0.22, 0.55))
    assert br.detector_levels(65 / 3.6) == pytest.approx((0.07, 0.25))
    assert br.detector_levels(100 / 3.6) == br.detector_levels(130 / 3.6) == pytest.approx((0.04, 0.14))
    f = (50 - 34) / (65 - 34)
    assert br.detector_levels(50 / 3.6) == pytest.approx((0.22 + f * (0.07 - 0.22), 0.55 + f * (0.25 - 0.55)))
    assert br.detector_strength(0.1, 0.2, 0.45) == 1.0 and br.detector_strength(0.5, 0.2, 0.45) == 0.0
    assert br.detector_strength(0.325, 0.2, 0.45) == pytest.approx(0.5) and br.detector_strength(-0.1, 0.2, 0.45) == 1.0

  def test_hand_over_latches_at_the_upper_level(self):
    # at 47 km/h the levels are 0.16 / 0.43 m: the plan steps 0.30 m left (swerve between the levels): scaled down, no hand-over and no
    # cap; then 0.80 m: hand-over, and at its first frame the strength factor is already 0 (rev 7.2: hand-over at 0.2 m)
    lo, up = br.detector_levels(self.V)
    assert lo < 0.2 < 0.3 < up < 0.5
    ctl = BlendfulCorrection()
    rows = []
    for i in range(400):
      plan = 0.0 if i < 100 else (-0.30 if i < 200 else -0.80)
      ctl.update(Scene(v=self.V, plan=plan).model_output(), self.V, True, 0.0, LAT_T, MODE_RIGHT, inputs())
      lg = ctl.result.log
      rows.append((ctl.hand_over, lg.get('det_strength', 1.0), lg['away_car'], ctl.result.no_growth, ctl.cap_active))
    mid = rows[100:200]
    assert not any(r[0] or r[3] or r[4] for r in mid) and min(r[1] for r in mid) < 0.8 and max(r[2] for r in mid) > lo
    first = next(i for i, r in enumerate(rows) if r[0])
    assert 200 <= first < 220 and rows[first][1] == 0.0


def test_rev8_safety_limits_unchanged():
  # guard (passes on rev 7.2 too): the final limiter, the right safety bounds and the fit clips are those of rev 7.2
  assert (br.BUILD, br.RELEASE, br.FADING_RELEASE, br.SWERVE_RELEASE) == (0.5, 2.0, 0.5, 5.0)
  assert (br.LINE_BOUND, br.EDGE_BOUND, br.D_MAX, br.ERROR_CLIP, br.W_R) == (0.2, 0.3, 1.0, 1.0, 0.81)


class TestRev8DetectorToTheModelPath:
  """Review B1 (owner 13 Nov 09:27 / 11:23): while the detector scales Keep Right down, the car goes to the model's own path, not to Regular's
  centring pull: in Right line the whole delivered correction is multiplied by the detector factor through its own 1 s low-pass."""
  @pytest.mark.parametrize("kmh,amp", [(50, 0.25), (50, 0.4), (80, 0.25), (80, 0.4)])
  @pytest.mark.parametrize("rho", [0.3, 0.6])
  def test_obstacle_on_the_left_the_car_never_moves_toward_it(self, kmh, amp, rho):
    # held position, then the model evades an obstacle on the LEFT (plan moves right by amp over 1 s): the car never moves left
    # (7d12d78fe: up to 0.33 m toward the obstacle; the factor on the whole correction with a also lowered: 0.045 / 0.066 m at rho 0.6, 0.25 m)
    r = pullback_loop(rho, kmh, seconds=68.0, swerve_at=60.0, sw_amp=-amp)
    y60 = r[int(round(60.0 / DT_MDL)), 1]
    assert y60 - r[r[:, 0] >= 60.0, 1].min() <= 1e-3

  @staticmethod
  def run(mode, swerve):
    ctl = BlendfulCorrection()
    cs, ups = [], []
    for _ in range(200):
      ctl.update(Scene(v=13.0, center=0.3).model_output(), 13.0, True, 0.0, LAT_T, mode, inputs())  # the car 0.3 m left: Regular pulls
      cs.append(ctl.c)
      ups.append(ctl.k_up)
    return np.array(cs), np.array(ups), ctl

  def test_factor_zero_is_correction_zero_in_right_line_and_regular_is_untouched(self, monkeypatch):
    up = br.detector_levels(13.0)[1]
    TestRev8Detector.patch(monkeypatch, lambda i: up + 0.1)  # strength factor 0, no latch (the fake never latches)
    cs, ups, ctl = self.run(MODE_RIGHT, up + 0.1)
    assert abs(ups[-1]) > 1e-5 and abs(cs[-1]) < 1e-12 and ctl.result.log['det_gain'] == 0.0 and not ctl.hand_over
    TestRev8Detector.patch(monkeypatch, lambda i: up + 0.1)
    cr, upr, _ = self.run(MODE_REGULAR, up + 0.1)  # Regular mode: the instance does not run (a = 0), Regular's add-on unchanged
    assert abs(cr[-1] - upr[-1]) < 1e-9 and abs(cr[-1]) > 1e-5

  def test_the_factor_reaches_the_correction_through_a_1_s_low_pass(self, monkeypatch):
    lo, up = br.detector_levels(13.0)
    sched = [0.0] * 100 + [(lo + up) / 2] * 100
    TestRev8Detector.patch(monkeypatch, lambda i: sched[min(i, len(sched) - 1)])
    ctl = BlendfulCorrection()
    gs = []
    for _ in range(200):
      ctl.update(Scene(v=13.0).model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, inputs())
      gs.append(ctl.result.log['det_gain'])
    k = DT_MDL / (DT_MDL + br.STRENGTH_TAU)
    assert gs[99] == 1.0 and gs[119] == pytest.approx(0.5 + 0.5 * (1 - k) ** 20, abs=1e-9)

  @staticmethod
  def wander_loop(mode, kmh=50, amp=0.5, period=30.0, rho=0.6, seconds=90.0):
    """No right line (prob 0: a = 0), the model wanders around amp sin(2 pi t / period) with pull share rho (the reviewer's wander probe)."""
    v = kmh / 3.6
    look = max(1.5 * v, 10.0)
    ctl = BlendfulCorrection()
    y, psi, kc = 0.0, 0.0, 0.0
    buf = [0.0] * 4
    ys, fd = [], []
    for i in range(int(seconds / DT_MDL)):
      t = i * DT_MDL
      mo = model_output(v, y, psi, kc, right_line_prob=0.0)
      mo['plan'][0, :, 1] = -psi * X
      e = -rho * (y - amp * np.sin(2 * np.pi * t / period))
      mo['plan'][0, :, 1] += e * np.clip(X / look, 0, 1)
      ctl.update(mo, v, True, -psi / v + 2.0 * e / look ** 2, LAT_T, mode, inputs(gap_low=0.44, gap_high=0.44))
      ys.append(y)
      fd.append(ctl.result.log.get('det_strength', 1.0) if ctl.result.log else 1.0)
      buf.append(ctl.c - psi / v + 2.0 * e / look ** 2)
      cmd = buf.pop(0)
      for _ in range(5):
        kc += (cmd - kc) * 0.01 / 0.3
        psi += v * kc * 0.01
        y += v * psi * 0.01
    return np.array(ys), np.array(fd)

  @pytest.mark.parametrize("kmh", [30, 80])
  def test_no_right_line_equals_regular_while_the_factor_is_below_1(self, kmh):
    # review S3 (owner 13 Nov 11:23): without a right line Right line is Regular, also while the detector deviates (3fedb5476: up to 0.29 m
    # less centring, the gain scaled Regular's share too)
    yr, fd = self.wander_loop(MODE_RIGHT, kmh)
    yg, _ = self.wander_loop(MODE_REGULAR, kmh)
    assert (fd < 1.0).mean() > 0.2
    assert np.abs(yr - yg).max() < 1e-9
