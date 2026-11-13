"""
Regression tests for the review of A8 implementation round 2 (inster-dev results/2026-11-12_a8_impl_r2/REVIEW.md: B1, B2, S2-S5, S6) and for
replay cause C1.  Behaviour-level only: they use the API that rounds 1 and 2 already had (BlendfulCorrection, RightLineBlendful,
RightInputs, the modes and the test Scene), so the same file runs on the earlier commits and fails there:
- C1 fails on round 1 (54350d615), passes on round 2 and later;
- B1, B2, S2, S3, S4 and S5 fail on round 2 (8b1f4e805) and pass from round 3 on.
Signs: model frame y > 0 is right, curvature > 0 is a right turn; c > 0 steers right.  Offline only; not validated on the car.
"""
import numpy as np
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.modeld_v2.blendful import Blendful
from openpilot.sunnypilot.modeld_v2 import blendful_right as br
from openpilot.sunnypilot.modeld_v2.blendful_right import MODE_OFF, MODE_REGULAR, MODE_RIGHT, BlendfulCorrection, RightLineBlendful
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.radar_clearance import healthy_empty_radar


def RightInputs(**kw):
  """RightInputs with a healthy, empty radar unless a test gives one (Right line needs a healthy radar, B6a)."""
  kw.setdefault('radar', healthy_empty_radar())
  return br.RightInputs(**kw)
from openpilot.sunnypilot.modeld_v2.tests.test_blendful_right import Scene

X = np.array(ModelConstants.X_IDXS)
LAT_T = 0.275
SWERVE_RELEASE = 5.0  # m/s^3: the cap-release rate (controlsd's limit)


def narrowed_lane_with_left_quality_flicker(q_centre: float, frames: int = 900, v: float = 20.0):
  """3.5 m lane for 15 s (a width memory forms), then the left line steps 0.9 m right (a 2.6 m lane, the right line stays) and from 20 s on
  the left line's probability oscillates around q_centre (+-0.25 at 1 Hz; after upstream's 0.5 s probability filter q_L swings about
  +-0.07 around it).  Returns the final target at x = L per frame from 22.5 s on, and the delivered correction of a BlendfulCorrection
  driven with the same scenes."""
  rl, ctl = RightLineBlendful(), BlendfulCorrection()
  targets, cs, q_ls = [], [], []
  for i in range(frames):
    t = max(i - 400, 0) * DT_MDL
    narrowed = i >= 300
    sc = Scene(v=v, width=3.5 - (0.9 if narrowed else 0.0), center=0.45 if narrowed else 0.0, pr=0.9, std=0.1,
               pl=0.9 if i < 400 else q_centre + 0.25 * np.sin(2 * np.pi * t))
    lg = rl.update(sc.model_output(), v, True, 0.0, 0.0, RightInputs()).log
    ctl.update(sc.model_output(), v, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
    targets.append(float(np.interp(1.5 * v, X, lg['target'])))
    q_ls.append(lg['q_L'])
    cs.append(ctl.c)
  return np.array(targets[450:]), np.array(cs[450:]), np.array(q_ls[450:])


class TestTargetUnderQualityFlicker:
  def test_c1_remembered_width_and_left_quality_flicker_at_0_3_do_not_step_the_final_target(self):
    # replay cause C1: round 1's anchor (left reference + remembered width) switched with q_L at 0.3 and the final target jumped 0.89 m
    targets, _, q_ls = narrowed_lane_with_left_quality_flicker(0.3)
    assert q_ls.min() < 0.3 < q_ls.max()  # the flicker really crosses the threshold
    assert np.abs(np.diff(targets)).max() <= 0.02

  def test_s4_centre_limit_formula_does_not_switch_with_quality_flicker_at_0_5(self):
    # review S4: the left limit changed formula at q_L 0.5 without hysteresis: 0.23 m steps of the final target, command ripple
    targets, cs, q_ls = narrowed_lane_with_left_quality_flicker(0.5)
    assert q_ls.min() < 0.5 < q_ls.max() and q_ls.min() > 0.4  # crosses 0.5, stays inside the hysteresis band
    assert np.abs(np.diff(targets)).max() <= 0.02
    accel = cs[100:] * 20.0 ** 2  # m/s^2 at 20 m/s, after the correction settled
    # what remains is the strength's own 1 Hz dependence on the left quality (s_R, B4): 0.012 m/s^2 peak to peak and 0.04 m/s^3, the same as
    # with the flicker around 0.9; round 2 here: 0.136 m/s^2 and 2.0 m/s^3 (the final limiter's rate) from the formula switching
    assert accel.max() - accel.min() <= 0.02
    assert np.abs(np.diff(accel)).max() / DT_MDL <= 0.1


# ---------------------------------------------------------------------------------------------------------------------------------
def reference_with_forced_yield(start: int, departure: float):
  """Today's Blendful as the reference instance, forced into its going-around yield from frame `start` on with the given own departure
  offset - usual (m; < 0: its plan end moved left, i.e. the obstacle is on the right).  Stands for the cases the design keeps the reference
  trigger for: upstream sees the obstacle while our endpoint-gap trigger stays below 0.1 m."""
  class ForcedYield(Blendful):
    frame = 0

    def update(self, *args, **kwargs):
      out = super().update(*args, **kwargs)
      ForcedYield.frame += 1
      if ForcedYield.frame > start:
        self.going_around = True
        self.yield_departure = departure
      return out
  return ForcedYield()


def run_forced_reference_yield(ref_departure: float, own_plan_wobble: float, v: float = 13.0, frames: int = 320):
  """Right line with the car pulled right; from frame 195 the plan wobbles by own_plan_wobble (m at 10 m; > 0 = right), so the endpoint
  gap has a small departure (< 0.1 m) of that sign when the reference instance starts the yield at frame 200."""
  ctl = BlendfulCorrection(reference=reference_with_forced_yield(200, ref_departure))
  rows = []
  for i in range(frames):
    sc = Scene(v=v, plan=0.0 if i < 195 else own_plan_wobble)
    ctl.update(sc.model_output(), v, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
    rows.append((ctl.c, ctl.result.yield_active, ctl.release_window, ctl.result.log['away_car']))
  return rows


class TestObstacleSideFromTheTriggeringInstance:
  @pytest.mark.parametrize("own_plan_wobble", [0.03, -0.03])
  def test_b1_reference_trigger_supplies_the_side_right_obstacle(self, own_plan_wobble):
    # the reference instance sees an obstacle on the RIGHT (its plan end moved 0.3 m left); our own departure is small, of either sign.
    # Round 2 took the side from our departure's sign and kept 0.245 m/s^2 toward the obstacle for the whole yield.
    rows = run_forced_reference_yield(-0.3, own_plan_wobble)
    assert rows[199][0] > 1e-4  # pulled right before
    first = next(i for i, r in enumerate(rows) if r[1])
    assert max(abs(r[3]) for r in rows[first - 5:first + 1]) < 0.1  # our own trigger stays quiet: only the reference instance yields
    for i in range(first, len(rows)):
      assert rows[i][0] <= max(rows[i - 1][0], 0.0) + 1e-12  # no growth toward the right obstacle
      if not rows[i][2] and i > first + 5:
        assert rows[i][0] <= 1e-6  # the hard rule: no correction toward the obstacle outside the cap-release window
    c = np.array([r[0] for r in rows])
    assert np.abs(np.diff(c)).max() * 13.0 ** 2 / DT_MDL <= SWERVE_RELEASE + 1e-6

  def test_b1_reference_trigger_supplies_the_side_left_obstacle(self):
    # the reference instance sees an obstacle on the LEFT (its plan end moved right): the right-line part may stay, nothing pushes left
    rows = run_forced_reference_yield(+0.3, -0.03)
    first = next(i for i, r in enumerate(rows) if r[1])
    assert all(r[0] >= -1e-9 for r in rows[first:])
    assert rows[-1][0] > 1e-4  # the correction to the right, away from the left obstacle, is kept

  def test_b1_side_unknown_caps_both_directions(self):
    # the reference instance yields with a departure below 0.1 m: until a side is known nothing may grow in either direction
    rows = run_forced_reference_yield(-0.05, +0.03)  # our own small departure has the sign that round 2 took for an obstacle on the left
    first = next(i for i, r in enumerate(rows) if r[1])
    for i in range(first, len(rows)):
      assert abs(rows[i][0]) <= abs(rows[i - 1][0]) + 1e-12
    assert abs(rows[-1][0]) <= 1e-6


# ---------------------------------------------------------------------------------------------------------------------------------
class TestHandBackWindowCap:
  @pytest.mark.parametrize("switch_frame", [206, 210, 215, 220])
  def test_b2_cap_holds_after_a_switch_to_regular_inside_a_yield(self, switch_frame):
    # obstacle on the right (the model swerves 0.8 m left from frame 200, 0.5 s); the cap brought c to 0, then the driver switches to
    # Regular while the right-line share a is still being handed back.  Round 2: c came back toward the obstacle up to 0.18 m/s^2.
    v = 13.0
    ctl = BlendfulCorrection()
    rows = []
    for i in range(switch_frame + 120):
      sw = 0.0 if i < 200 else min(0.8, 0.8 * (i - 200) * DT_MDL / 0.5)
      mode = MODE_RIGHT if i < switch_frame else MODE_REGULAR
      ctl.update(Scene(v=v, plan=-sw).model_output(), v, True, 0.0, LAT_T, mode, RightInputs())
      rows.append((ctl.c, ctl.release_window, ctl.right.a))
    assert rows[switch_frame - 1][2] > 0.0 or switch_frame >= 220  # inside the hand-back window for the early switches
    c_at_switch = rows[switch_frame - 1][0]
    after = [r[0] for r in rows[switch_frame:]]
    assert max(after) <= max(c_at_switch, 0.0) + 1e-12  # no growth toward the obstacle after the switch
    assert max(after) * v * v <= 1e-3

  def test_b2_regular_is_exactly_upstream_once_the_window_and_the_yield_are_over(self):
    v = 13.0
    ctl, ref = BlendfulCorrection(), Blendful()
    for i in range(1400):
      sw = 0.0 if i < 200 else (min(0.8, 0.8 * (i - 200) * DT_MDL / 0.5) if i < 300 else 0.0)
      mode = MODE_RIGHT if i < 210 else MODE_REGULAR
      sc = Scene(v=v, plan=-sw, center=0.3)
      ctl.update(sc.model_output(), v, True, 0.0, LAT_T, mode, RightInputs())
      up = ref.update(sc.model_output(), v, True, 0.0, enabled=True)
    assert ctl.right_idle  # the right-line instance stopped once a = 0 and its yield ended
    assert abs(ctl.c - up) <= 1e-9  # Regular delivers today's add-on again


# ---------------------------------------------------------------------------------------------------------------------------------
class TestLaneChange:
  def test_s2_no_false_yield_after_a_lane_change(self):
    # held 0.4 m right of the centre, lane change to the left (not active for 3 s, the lines re-index when the car crosses the line), then
    # centred in the new lane.  Round 2 kept the frozen endpoint-gap state and yielded falsely for 17.6 s after the lane change.
    v = 20.0
    ctl = BlendfulCorrection()
    after = []
    for i in range(900):
      if i < 300:
        centre, active = -0.4, True
      elif i < 360:
        c = -0.4 + 3.9 * (i - 300) / 60.0
        centre, active = (c if c < 1.75 else c - 3.5), False
      else:
        centre, active = 0.0, True
      ctl.update(Scene(v=v, center=centre).model_output(), v, active, 0.0, LAT_T, MODE_RIGHT, RightInputs())
      if i >= 360:
        after.append((ctl.result.yield_active, ctl.result.no_growth, ctl.result.log.get('away_car', 0.0)))
    assert not any(r[0] for r in after)
    assert not any(r[1] for r in after)

  def test_s2_lane_change_into_a_wider_lane_is_not_a_junction(self):
    # same root cause as S2: the width memory of the old 3.0 m lane compared with the new 3.8 m lane looked like a widening by 0.8 m
    v = 20.0
    ctl = BlendfulCorrection()
    episodes = []
    for i in range(1000):
      if i < 400:
        sc, active = Scene(v=v, width=3.0), True
      elif i < 460:
        c = 3.4 * (i - 400) / 60.0  # the car moves left by one lane
        sc, active = (Scene(v=v, width=3.0, center=c) if c < 1.5 else Scene(v=v, width=3.8, center=c - 3.4)), False
      else:
        sc, active = Scene(v=v, width=3.8), True
      ctl.update(sc.model_output(), v, active, 0.0, LAT_T, MODE_RIGHT, RightInputs())
      if i >= 460:
        episodes.append(ctl.result.episode)
    assert not any(episodes)


# ---------------------------------------------------------------------------------------------------------------------------------
def nan_frame(kind: str, v: float = 13.0):
  mo = Scene(v=v).model_output()
  if kind == 'yaw_rate':
    mo['plan'][0, 0, Plan.ORIENTATION_RATE.start + 2] = np.nan
  elif kind == 'lane_line':
    mo['lane_lines'][0, 2, 10, 0] = np.nan
  elif kind == 'plan_speed':
    mo['plan'][0, :, Plan.VELOCITY.start] = np.inf
  return mo


class TestNonFiniteInputs:
  @pytest.mark.parametrize("kind", ['yaw_rate', 'lane_line', 'plan_speed'])
  @pytest.mark.parametrize("mode", [MODE_RIGHT, MODE_REGULAR])
  def test_s3_one_non_finite_frame_is_released_and_off_resumes(self, kind, mode):
    v = 13.0
    ctl = BlendfulCorrection()
    cs = []
    for i in range(200):
      mo = nan_frame(kind) if i == 100 else Scene(v=v, center=0.4).model_output()
      ctl.update(mo, v, True, 0.0, LAT_T, mode, RightInputs())
      cs.append(ctl.c)
      assert np.isfinite(ctl.c) and np.isfinite(ctl.k0)
    cs = np.array(cs)
    assert abs(cs[99]) > 1e-4  # a correction was running
    assert np.abs(np.diff(cs)).max() * v * v / DT_MDL <= 2.0 + 1e-6  # released through the limiter, no jump
    assert abs(cs[-1]) > 1e-4  # and it comes back once the frames are finite again
    for _ in range(40):
      ctl.update(Scene(v=v).model_output(), v, True, 0.0, LAT_T, MODE_OFF, RightInputs())
    assert ctl.c == 0.0 and not ctl.releasing  # Off resumes: modeld takes today's path again, no modeld restart needed

  def test_s3_non_finite_frames_in_off_while_releasing(self):
    v = 13.0
    ctl = BlendfulCorrection()
    for _ in range(100):
      ctl.update(Scene(v=v, center=0.4).model_output(), v, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
    for i in range(40):
      ctl.update(nan_frame('plan_speed') if i % 2 else Scene(v=v).model_output(), v, True, 0.0, LAT_T, MODE_OFF, RightInputs())
      assert np.isfinite(ctl.c) and np.isfinite(ctl.k0)
    assert not ctl.releasing


# ---------------------------------------------------------------------------------------------------------------------------------
class TestUpdateTime:
  def test_s5_the_whole_update_time_is_logged(self):
    from openpilot.cereal import custom
    from openpilot.sunnypilot.modeld_v2.blendful_right import fill_log
    ctl = BlendfulCorrection()
    for _ in range(30):
      ctl.update(Scene(v=13.0, edges=True).model_output(), 13.0, True, 0.0, LAT_T, MODE_RIGHT, RightInputs())
    ms = ctl.update_ms
    assert ms > 0.0 and ms >= ctl.result.log['filter_ms']  # the whole update, not only the four filters
    msg = custom.BlendfulInster.new_message()
    fill_log(msg, ctl, MODE_RIGHT, 13.0)
    assert abs(msg.blendfulUpdateTime - ms) < 1e-4
    msg = custom.BlendfulInster.new_message()
    fill_log(msg, ctl, MODE_OFF, 13.0)  # a frame on which the class did not run: no stale value
    assert msg.blendfulUpdateTime == 0.0
