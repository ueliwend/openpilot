"""
Blind-spot hold (owner, 21 Nov 2026): a side's blind spot counts as occupied until about 1 s after its lamp went off, also
when the lamp went off before the blinker. While held it blocks the start of a lane change toward that side like a lit lamp,
in every lane-change mode, with "Delay with Blind Spot" on or off. The tests drive DesireHelper frame by frame (DT_MDL).
"""
from openpilot.cereal import custom
from openpilot.common.params import Params
from openpilot.common.parameterized import parameterized
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper, LaneChangeState, LaneChangeDirection
from openpilot.sunnypilot.selfdrive.controls.lib.auto_lane_change import AutoLaneChangeMode

HOLD_S = 1.0  # owner's value; the code's constant is checked in test_hold_constant
EPS = 1e-6
V_EGO = 31.7  # m/s, 114 km/h as at 18:23:09
TIMED_MODES = [AutoLaneChangeMode.NUDGELESS, AutoLaneChangeMode.HALF_SECOND, AutoLaneChangeMode.ONE_SECOND,
               AutoLaneChangeMode.TWO_SECONDS, AutoLaneChangeMode.THREE_SECONDS]
ALL_MODES = [AutoLaneChangeMode.NUDGE] + TIMED_MODES
MODE_X_DELAY = [(m, d) for m in ALL_MODES for d in (False, True)]


class CS:
  def __init__(self, vEgo=V_EGO, leftBlinker=False, rightBlinker=False, leftBlindspot=False, rightBlindspot=False,
               steeringPressed=False, steeringTorque=0.0, brakePressed=False):
    self.vEgo = vEgo
    self.leftBlinker = leftBlinker
    self.rightBlinker = rightBlinker
    self.leftBlindspot = leftBlindspot
    self.rightBlindspot = rightBlindspot
    self.steeringPressed = steeringPressed
    self.steeringTorque = steeringTorque
    self.brakePressed = brakePressed


def frames(t0, t1):
  """Model frame times from t0 to t1 (inclusive), on the DT_MDL grid."""
  return [round(k * DT_MDL, 6) for k in range(round(t0 / DT_MDL), round(t1 / DT_MDL) + 1)]


def in_any(t, intervals):
  """t inside one of the closed intervals [(a, b), ...] (frame times)."""
  return any(a - EPS <= t <= b + EPS for a, b in intervals)


class Sim:
  def __init__(self, mode, bsm_delay):
    params = Params()
    params.put("AutoLaneChangeTimer", mode)
    params.put_bool("AutoLaneChangeBsmDelay", bsm_delay)
    self.dh = DesireHelper()
    self.dh.alc.read_params()
    self.rows = []  # (t, state, direction, blocked, left_block, right_block)

  def run(self, t0, t1, cs_fn, lateral_fn=lambda t: True, prob_fn=None):
    """prob_fn(t, sim) gives the model's lane-change probability; default 1.0 for 3 s of a change, then 0 (change done)."""
    for t in frames(t0, t1):
      prob = prob_fn(t, self) if prob_fn else (0.0 if self.dh.lane_change_timer >= 3.0 else 1.0)
      self.dh.update(cs_fn(t), lateral_fn(t), prob)
      hold = getattr(self.dh.alc, 'blindspot_hold', None)
      self.rows.append((t, self.dh.lane_change_state, self.dh.lane_change_direction, self.dh.alc.lane_change_blocked,
                        hold.left if hold else None, hold.right if hold else None))
    return self

  def starts(self):
    """Frame times at which laneChangeStarting begins."""
    out, prev = [], LaneChangeState.off
    for t, st, *_ in self.rows:
      if st == LaneChangeState.laneChangeStarting and prev != LaneChangeState.laneChangeStarting:
        out.append(t)
      prev = st
    return out

  def first_start(self):
    s = self.starts()
    return s[0] if s else None

  def state_at(self, t):
    return next(r for r in self.rows if abs(r[0] - t) < EPS)


def scenario(mode, bsm_delay, lamp, blinker_on, side='left', t_end=8.0, nudge=None):
  """lamp: closed intervals of frame times with the signalled side's lamp on; blinker on from blinker_on; nudge: time from which
  the driver steers toward the blinker side above the nudge threshold (Nudge mode), default with the blinker."""
  left = side == 'left'
  if nudge is None and mode == AutoLaneChangeMode.NUDGE:
    nudge = blinker_on

  def cs(t):
    bl = t >= blinker_on - EPS
    lamp_on = in_any(t, lamp)
    pressed = nudge is not None and t >= nudge - EPS
    return CS(leftBlinker=bl and left, rightBlinker=bl and not left,
              leftBlindspot=lamp_on and left, rightBlindspot=lamp_on and not left,
              steeringPressed=pressed, steeringTorque=(200.0 if left else -200.0) if pressed else 0.0)
  return Sim(mode, bsm_delay).run(0.0, t_end, cs)


def clean_start_delay(mode, bsm_delay):
  """Start time after the blinker frame with no blind-spot signal at all (unchanged by the hold)."""
  return scenario(mode, bsm_delay, [], 1.0).first_start() - 1.0


class TestBlindspotHold(OpenpilotTestCase):
  def test_hold_constant(self):
    from openpilot.sunnypilot.selfdrive.controls.lib.auto_lane_change import BSM_HOLD_TIME, BSM_HOLD_FRAMES
    assert BSM_HOLD_TIME == HOLD_S
    assert BSM_HOLD_FRAMES == round(HOLD_S / DT_MDL) == 20

  def test_schema_fields(self):
    msg = custom.ModelDataV2SP.new_message(leftLaneChangeBlindspotBlock=True, rightLaneChangeBlindspotBlock=False)
    assert msg.leftLaneChangeBlindspotBlock and not msg.rightLaneChangeBlindspotBlock

  @parameterized.expand(MODE_X_DELAY)
  def test_clean_timing_unchanged(self, mode, bsm_delay):
    """No blind-spot signal: start delays as on 1b1f3bfab (Nudge: with the nudge 0.05 s; FIX.md table)."""
    expected = {AutoLaneChangeMode.NUDGE: 0.05, AutoLaneChangeMode.NUDGELESS: 0.10, AutoLaneChangeMode.HALF_SECOND: 0.55,
                AutoLaneChangeMode.ONE_SECOND: 1.00, AutoLaneChangeMode.TWO_SECONDS: 2.00, AutoLaneChangeMode.THREE_SECONDS: 3.05}
    assert abs(clean_start_delay(mode, bsm_delay) - expected[mode]) < EPS

  @parameterized.expand([(m, d, gap) for m, d in MODE_X_DELAY for gap in (-0.5, 0.0, 0.05, 0.2, 0.5, 0.9, 1.0, 1.2)])
  def test_hold_every_mode(self, mode, bsm_delay, gap):
    """Lamp on 2.0-3.0 s (last frame on at 3.0, first off 3.05); blinker at 3.0 + gap (negative: while the lamp is on).
    The start is never earlier than 1.0 s after the lamp went off, never earlier than the clean (no signal) delay after the
    blinker when the lamp was off before preLaneChange (the hold never shortens a timed wait), and never later than needed."""
    lamp_last_on, lamp_off = 3.0, 3.05
    blinker = round(lamp_last_on + gap, 6)
    sim = scenario(mode, bsm_delay, [(2.0, lamp_last_on)], blinker)
    start = sim.first_start()
    assert start is not None
    assert start >= lamp_off + HOLD_S - DT_MDL - EPS            # 1.0 s after the last frame with the lamp on ...
    assert start >= lamp_last_on + HOLD_S + DT_MDL - EPS        # ... i.e. from the 21st frame: >= 1.0 s after it went off
    clean = blinker + clean_start_delay(mode, bsm_delay)
    hold_end = lamp_last_on + HOLD_S + DT_MDL
    if blinker > lamp_last_on + EPS:
      # lamp never seen in preLaneChange: the timers run exactly as without a lamp; the hold only adds the wait
      assert abs(start - max(clean, hold_end)) < EPS
    else:
      # lamp seen in preLaneChange: start when the hold ends or the (unchanged) timer allows, whichever is later
      assert start >= hold_end - EPS
      assert start <= max(clean, hold_end + DT_MDL) + EPS
    # blocked frames before the start show the blind-spot alert flag for that side
    for t, st, _d, blocked, lb, _rb in sim.rows:
      if blinker - EPS <= t < start - EPS and st == LaneChangeState.preLaneChange and t <= hold_end - DT_MDL + EPS:
        assert lb and (blocked or t < blinker + EPS)  # the first preLaneChange frame is not evaluated yet

  @parameterized.expand(MODE_X_DELAY)
  def test_incident_182309_replay(self, mode, bsm_delay):
    """18:23:09 from the qlog (10 Hz): left lamp on 02.70-08.70, off from 08.80; blinker edge about 08.95 (preLaneChange
    event 08.97); Nudgeless started at 09.08. Times here in s after 18:23:00; lamp modelled on to 08.75 (last frame before
    the first off sample). The start must not come before 08.80 + 1.0 = 09.80."""
    sim = scenario(mode, bsm_delay, [(2.70, 8.75)], 8.95, t_end=14.0)
    start = sim.first_start()
    assert start is not None
    assert start >= 8.80 + HOLD_S - EPS
    assert sim.state_at(8.95)[1] == LaneChangeState.preLaneChange and sim.state_at(8.95)[4]  # held: alert flag set
    assert sim.state_at(9.0)[3]  # blocked from the first frame the state machine evaluates preLaneChange
    if mode == AutoLaneChangeMode.NUDGELESS:
      assert abs(start - 9.80) < EPS  # 21 frames after the last lamp frame (was 09.05 on 1b1f3bfab)

  @parameterized.expand(MODE_X_DELAY)
  def test_lamp_blip_restarts_hold(self, mode, bsm_delay):
    """Lamp on 2.0-3.0, off, one-frame blip at 3.6, off again; blinker at 3.3. The blip restarts the 1 s hold."""
    sim = scenario(mode, bsm_delay, [(2.0, 3.0), (3.6, 3.6)], 3.3)
    start = sim.first_start()
    assert start is not None
    assert start >= 3.65 + HOLD_S - EPS
    assert sim.state_at(4.2)[1] == LaneChangeState.preLaneChange

  @parameterized.expand(MODE_X_DELAY)
  def test_right_side(self, mode, bsm_delay):
    """Mirror of the incident to the right."""
    sim = scenario(mode, bsm_delay, [(2.70, 8.75)], 8.95, side='right', t_end=14.0)
    start = sim.first_start()
    assert start is not None and start >= 8.80 + HOLD_S - EPS
    assert sim.rows[-1][2] in (LaneChangeDirection.right, LaneChangeDirection.none)
    assert sim.state_at(9.5)[5] and not sim.state_at(9.5)[4]

  @parameterized.expand(MODE_X_DELAY)
  def test_other_side_not_held(self, mode, bsm_delay):
    """A held LEFT blind spot does not delay a RIGHT lane change."""
    def cs(t):
      return CS(rightBlinker=t >= 3.1 - EPS, leftBlindspot=2.0 - EPS <= t <= 3.0 + EPS,
                steeringPressed=mode == AutoLaneChangeMode.NUDGE and t >= 3.1 - EPS,
                steeringTorque=-200.0 if mode == AutoLaneChangeMode.NUDGE and t >= 3.1 - EPS else 0.0)
    start = Sim(mode, bsm_delay).run(0.0, 8.0, cs).first_start()
    assert abs(start - (3.1 + clean_start_delay(mode, bsm_delay))) < EPS

  @parameterized.expand([(d,) for d in (False, True)])
  def test_nudge_blocked_during_hold(self, bsm_delay):
    """Nudge mode: a nudge held from the blinker on does not start the change during the hold; it starts in the first free
    frame. A nudge that ends inside the hold starts nothing."""
    sim = scenario(AutoLaneChangeMode.NUDGE, bsm_delay, [(2.0, 3.0)], 3.2, nudge=3.2)
    assert abs(sim.first_start() - (3.0 + HOLD_S + DT_MDL)) < EPS

    def cs(t):
      pressed = 3.2 - EPS <= t <= 3.8 + EPS
      return CS(leftBlinker=t >= 3.2 - EPS, leftBlindspot=2.0 - EPS <= t <= 3.0 + EPS,
                steeringPressed=pressed, steeringTorque=200.0 if pressed else 0.0)
    sim2 = Sim(AutoLaneChangeMode.NUDGE, bsm_delay).run(0.0, 8.0, cs)
    assert sim2.first_start() is None
    assert sim2.rows[-1][1] == LaneChangeState.preLaneChange

  @parameterized.expand(MODE_X_DELAY)
  def test_no_rearm_after_blinker_off(self, mode, bsm_delay):
    """Blinker set during the hold, released during the hold: state off; the hold then ends: nothing starts until a new
    blinker edge, which then starts normally."""
    def cs(t):
      bl = (3.2 - EPS <= t <= 3.6 + EPS) or t >= 6.0 - EPS
      pressed = mode == AutoLaneChangeMode.NUDGE and bl
      return CS(leftBlinker=bl, leftBlindspot=2.0 - EPS <= t <= 3.0 + EPS, steeringPressed=pressed,
                steeringTorque=200.0 if pressed else 0.0)
    sim = Sim(mode, bsm_delay).run(0.0, 11.0, cs)
    assert sim.state_at(3.7)[1] == LaneChangeState.off
    assert all(st == LaneChangeState.off for t, st, *_ in sim.rows if 3.7 - EPS <= t < 6.0 - EPS)
    assert abs(sim.first_start() - (6.0 + clean_start_delay(mode, bsm_delay))) < EPS

  @parameterized.expand([(m, d) for m in TIMED_MODES for d in (False, True)])
  def test_no_rearm_after_completed_change(self, mode, bsm_delay):
    """Automatic change done (model probability < 0.02) with the blinker still on: back to preLaneChange; a lamp that lights
    and clears then (hold ending) must not start a second automatic change on the same blinker."""
    def cs(t):
      return CS(leftBlinker=t >= 1.0 - EPS, leftBlindspot=8.0 - EPS <= t <= 8.5 + EPS)

    def prob(t, sim):
      return 1.0 if t < 6.0 else 0.0
    sim = Sim(mode, bsm_delay).run(0.0, 14.0, cs, prob_fn=prob)
    assert len(sim.starts()) == 1
    assert sim.rows[-1][1] == LaneChangeState.preLaneChange

  @parameterized.expand(MODE_X_DELAY)
  def test_no_rearm_below_speed(self, mode, bsm_delay):
    """Blinker set below 32 km/h during the lamp; the car then speeds up while the hold ends: no start (no new edge)."""
    def cs(t):
      pressed = mode == AutoLaneChangeMode.NUDGE and t >= 2.5 - EPS
      return CS(vEgo=5.0 if t < 3.5 else V_EGO, leftBlinker=t >= 2.5 - EPS, leftBlindspot=2.0 - EPS <= t <= 3.0 + EPS,
                steeringPressed=pressed, steeringTorque=200.0 if pressed else 0.0)
    sim = Sim(mode, bsm_delay).run(0.0, 9.0, cs)
    assert sim.first_start() is None
    assert all(st == LaneChangeState.off for _t, st, *_ in sim.rows)

  @parameterized.expand(MODE_X_DELAY)
  def test_hold_ends_with_blinker_on_starts_pending(self, mode, bsm_delay):
    """The pending preLaneChange (same blinker edge) starts when the hold ends, as it does today when a lit lamp clears."""
    sim = scenario(mode, bsm_delay, [(2.0, 4.0)], 3.0)  # blinker while the lamp is on
    start = sim.first_start()
    assert start >= 4.0 + HOLD_S + DT_MDL - EPS
    assert start <= max(3.0 + clean_start_delay(mode, bsm_delay), 4.0 + HOLD_S + 2 * DT_MDL) + EPS
    assert len(sim.starts()) == 1

  @parameterized.expand(MODE_X_DELAY)
  def test_hazards(self, mode, bsm_delay):
    """Both blinkers (hazard lights) never start a lane change, with or without a lamp or hold."""
    def cs(t):
      haz = t >= 2.5 - EPS
      return CS(leftBlinker=haz, rightBlinker=haz, leftBlindspot=2.0 - EPS <= t <= 3.0 + EPS,
                rightBlindspot=2.0 - EPS <= t <= 3.0 + EPS, steeringPressed=haz, steeringTorque=200.0 if haz else 0.0)
    sim = Sim(mode, bsm_delay).run(0.0, 8.0, cs)
    assert all(st == LaneChangeState.off for _t, st, *_ in sim.rows)

  def test_hold_tracked_without_lateral(self):
    """Lamp history counts while lateral control is off; lateral comes on with the blinker on (re-arm path, unchanged):
    the start still waits for the hold."""
    def cs(t):
      return CS(leftBlinker=t >= 2.8 - EPS, leftBlindspot=2.0 - EPS <= t <= 3.0 + EPS)
    sim = Sim(AutoLaneChangeMode.NUDGELESS, False).run(0.0, 8.0, cs, lateral_fn=lambda t: t >= 3.3 - EPS)
    assert abs(sim.first_start() - (3.0 + HOLD_S + DT_MDL)) < EPS

  def test_published_flags(self):
    """The flags modeld publishes (modelDataV2SP.left/rightLaneChangeBlindspotBlock): on with the lamp and for 20 frames
    after it, then off; independent of the blinker."""
    sim = Sim(AutoLaneChangeMode.NUDGELESS, True).run(0.0, 5.0, lambda t: CS(leftBlindspot=1.0 - EPS <= t <= 2.0 + EPS))
    left = {round(t, 2): lb for t, _s, _d, _b, lb, _rb in sim.rows}
    assert not left[0.95] and left[1.0] and left[2.0] and left[3.0] and not left[3.05]
    assert not any(rb for *_x, rb in sim.rows)
