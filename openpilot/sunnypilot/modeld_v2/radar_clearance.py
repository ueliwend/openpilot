"""
Right-side clearance check from the radar (A8 right-line Blendful, design rev 6.5 section B6a; inster-dev
results/2026-11-12_a8_right_line_blendful/DESIGN.md at 5f80b26).  A second obstacle check next to the plan-based endpoint-gap trigger: it reads
only the radar tracks (`radarTracks`, car.RadarData, published by card.py from the Inster's RADAR_3A5_3C4 tracks), the filtered references,
the centre limit and the speed, never the model's plan.

- Objects: model frame x = dRel + 1.52 m, y = -yRel + dc (radar yRel > 0 = left; radar on the car's centreline, EST).
- Confirmed: the same trackId in >= 6 consecutive messages with a consistent position (within 1.0 m of dRel + vRel dt and 1.0 m laterally).
- Memory: a confirmed object last seen at x < 20 m is kept, moved with the B2 motion model, until x < -3.3 m (1 m behind the rear bumper).
- Corridor: x <= max(15 m, 3 s v), right of the centre limit; objects more than 1.0 m beyond a clear right edge are roadside and ignored.
- Margin m(v) = g(v) + e(v): g the GUI gap in use, e 0.2 m at <= 50 km/h rising linearly to 0.5 m at >= 100 km/h (owner 22:24).
- Object bound T_obj = min (y - w_R - m - dc) over the corridor; falls at once, rises after 1 s without a fall at 0.05 m/s.
- Clearance deficit d = m - min (y - w_R - dc - T_C'(x)); d > 0 starts the clearance yield (obstacle side right), released when d < -0.1 m
  for 1 s (or the object is gone); the cap ramps over d 0 .. 0.05 m.
- Stationary / moving (radial test over 8 of 10 messages) and pop-ups are logged only.
- Health: no message, a message older than 0.3 s, an invalid message, any RadarData.errors flag, a non-finite dRel / yRel in the message,
  or a car radar configuration that publishes no side tracks (review R3-B1: radarUnavailable, or a Hyundai without RADAR_FULL_RADAR, whose
  messages are valid but empty or lead-only) -> unhealthy; healthy again 1 s after healthy messages return.  The caller turns unhealthy into
  f_radar = 0 (Right line fades to Regular).
Signs: model frame y > 0 = right.  Offline only; not validated on the car.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from openpilot.common.realtime import DT_MDL

RADAR_TO_MODEL_X = 1.52  # m, radar ahead of the model origin (radard.py:30; EST for the Inster)
CONFIRM_MESSAGES = 6
CONSISTENT_DIST, CONSISTENT_LAT = 1.0, 1.0  # m
MEMORY_X, FORGET_X = 20.0, -3.3  # m: remembered when last seen nearer than this; forgotten 1 m behind the rear bumper
CORRIDOR_TIME, CORRIDOR_MIN = 3.0, 15.0  # s, m
ROADSIDE_BEYOND_EDGE, EDGE_CLEAR_WEIGHT = 1.0, 0.5  # m, candidate weight of the right edge
MARGIN_EXTRA = (0.2, 0.5)  # m at <= 50 km/h and >= 100 km/h
MARGIN_SPEEDS = (50 / 3.6, 100 / 3.6)
BOUND_HOLD = 1.0  # s; the bound then rises at r_ref (blendful_right.REFERENCE_RATE, passed in)
DEFICIT_RAMP, DEFICIT_RELEASE, DEFICIT_RELEASE_TIME = 0.05, -0.1, 1.0  # m, m, s
STALE_AGE, HEALTHY_RETURN = 0.3, 1.0  # s
MOVING_SPEED, MOVING_WINDOW, MOVING_MIN = 1.0, 10, 8  # m/s, messages
POPUP_TIME = 2.0  # s ahead
NO_BOUND = 1e3  # m: the object bound when nothing binds


@dataclass
class RadarFrame:
  """The newest radarTracks message as seen by modeld in this model frame."""
  tracks: list = field(default_factory=list)  # (trackId, dRel, yRel, vRel)
  age: float = float('inf')  # s since the message was received
  fault: bool = True  # message invalid or any RadarData.errors flag (or never received)
  new: bool = False  # the message arrived since the last model frame
  right_blindspot: bool = False  # carState.rightBlindspot
  config_ok: bool = False  # the car's radar configuration publishes side tracks (radar_tracks_ok); unknown counts as not


def radar_tracks_ok(CP, CP_SP) -> bool:
  """Does this car's radar configuration publish side tracks in radarTracks (review R3-B1)?  Not with CarParams.radarUnavailable (card.py then
  publishes empty RadarData), and on a Hyundai only with HyundaiFlagsSP.RADAR_FULL_RADAR (opendbc sunnypilot/car/hyundai/values.py:41): the
  default without a fingerprinted track range is RADAR_LEAD_ONLY (sunnypilot/car/interfaces.py:145-152), one SCC lead point with yRel NaN
  (hyundai/radar_interface_ext.py), and radar_off_can gives empty messages (hyundai/radar_interface.py:181)."""
  from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP
  return (not CP.radarUnavailable) and (CP.brand != "hyundai" or bool(CP_SP.flags & HyundaiFlagsSP.RADAR_FULL_RADAR))


def read_radar_tracks_ok(params) -> bool:
  """radar_tracks_ok from the CarParams / CarParamsSP that card.py writes (selfdrive/car/card.py:184-194); missing or unreadable -> False."""
  try:
    from openpilot.cereal import custom, messaging
    from opendbc.car.structs import car
    cp_bytes, cp_sp_bytes = params.get("CarParams"), params.get("CarParamsSP")
    if not cp_bytes or not cp_sp_bytes:
      return False
    return radar_tracks_ok(messaging.log_from_bytes(cp_bytes, car.CarParams), messaging.log_from_bytes(cp_sp_bytes, custom.CarParamsSP))
  except Exception:
    return False


def healthy_empty_radar() -> RadarFrame:
  """A healthy radar that sees nothing (offline tests and replays without objects), on a car whose configuration publishes side tracks."""
  return RadarFrame(tracks=[], age=0.0, fault=False, new=True, config_ok=True)


def frame_finite(radar: RadarFrame) -> bool:
  """Every track's dRel and yRel is finite (a lead-only message carries yRel NaN, R3-B1 second guard)."""
  return all(np.isfinite(d) and np.isfinite(y) for _, d, y, _ in radar.tracks)


def margin_extra(v: float) -> float:
  return float(np.interp(v, MARGIN_SPEEDS, MARGIN_EXTRA))


@dataclass
class Track:
  x: float
  y: float
  d_rel: float
  y_rel: float
  v_rel: float
  count: int = 1
  confirmed: bool = False
  popup: bool = False
  seen: bool = True  # in the newest message (False: remembered beside the car)
  moving_votes: deque = field(default_factory=lambda: deque(maxlen=MOVING_WINDOW))

  @property
  def moving(self) -> bool:
    return sum(self.moving_votes) >= MOVING_MIN


@dataclass
class ClearanceResult:
  healthy: bool = False
  t_obj: float = NO_BOUND  # held object bound (target units, applied to the whole window), NO_BOUND when nothing binds
  t_obj_raw: float = NO_BOUND
  bound_active: bool = False  # set by the caller when the bound binds the target
  deficit: float = -NO_BOUND  # m
  yield_active: bool = False
  yield_q: float = 0.0  # in the units of the B6 cap ramp (0.15 at d = 0 .. 0.2 at d = 0.05 m), Q_NO_GROWTH while only latched
  margin: float = 0.0
  objects: int = 0  # confirmed objects in the corridor
  memory: int = 0  # of them remembered beside the car
  moving: int = 0  # of them classified moving (log only)
  popups: int = 0  # confirmed this frame less than 2 s ahead or beside the car (log only)
  nearest: tuple = (0.0, 0.0, 0.0)  # x, y and clearance (car's right side to the object now, m) of the corridor object with the largest deficit
  hold: float = 0.0  # s since the bound last fell
  right_blindspot: bool = False


class RadarClearance:
  def __init__(self):
    self.tracks: dict[int, Track] = {}
    self.t_obj = NO_BOUND
    self.hold = 0.0
    self.healthy_t = 0.0
    self.healthy = False
    self.yield_latch = False
    self.release_t = 0.0
    self.yield_obj_x: float | None = None  # where the object that set the deficit is now (moved with the car's travel while unseen)

  def reset(self):
    self.__init__()

  # --- health
  def _health(self, radar: RadarFrame | None, dt: float) -> bool:
    ok = radar is not None and radar.config_ok and not radar.fault and radar.age <= STALE_AGE and frame_finite(radar)
    self.healthy_t = self.healthy_t + dt if ok else 0.0
    if not ok:
      self.healthy = False
    elif self.healthy_t >= HEALTHY_RETURN - 1e-9 or self.healthy:
      self.healthy = True
    return ok

  # --- tracks
  def _ingest(self, radar: RadarFrame, v: float, msg_dt: float, dc: float):
    seen_ids = set()
    for tid, d_rel, y_rel, v_rel in radar.tracks:
      tid = int(tid)
      seen_ids.add(tid)
      x, y = float(d_rel) + RADAR_TO_MODEL_X, -float(y_rel) + dc
      tr = self.tracks.get(tid)
      if tr is not None and tr.seen:
        consistent = abs(d_rel - (tr.d_rel + tr.v_rel * msg_dt)) <= CONSISTENT_DIST and abs(y_rel - tr.y_rel) <= CONSISTENT_LAT
        count = tr.count + 1 if consistent else 1
        votes = tr.moving_votes if consistent else deque(maxlen=MOVING_WINDOW)
        confirmed = tr.confirmed and consistent
      else:
        count, votes, confirmed = 1, deque(maxlen=MOVING_WINDOW), False
      az = np.arctan2(y_rel, max(d_rel, 1e-3))
      votes.append(abs(v_rel + v * np.cos(az)) >= MOVING_SPEED)
      new = Track(x=x, y=y, d_rel=float(d_rel), y_rel=float(y_rel), v_rel=float(v_rel), count=count, confirmed=confirmed, moving_votes=votes)
      if not new.confirmed and new.count >= CONFIRM_MESSAGES:
        new.confirmed = True
        new.popup = x < POPUP_TIME * max(v, 1.0)
      self.tracks[tid] = new
    for tid, tr in list(self.tracks.items()):
      if tid in seen_ids:
        continue
      if tr.confirmed and tr.x < MEMORY_X:
        tr.seen, tr.popup = False, False  # left the field of view: remembered beside the car
      else:
        del self.tracks[tid]

  def _move_memory(self, vx0: float, yaw_rate: float, dt: float, l_d: float):
    """Remembered objects are ground-fixed: moved with the B2 motion model into the new car frame (and forgotten behind the car)."""
    for tid, tr in list(self.tracks.items()):
      if tr.seen:
        continue
      tr.x -= vx0 * dt
      tr.y -= yaw_rate * dt * (tr.x + l_d + vx0 * dt / 2)
      if tr.x < FORGET_X:
        del self.tracks[tid]

  def update(self, radar: RadarFrame | None, v: float, vx0: float, yaw_rate: float, gap: float, dc: float, w_r: float,
             x_idxs: np.ndarray, c_ctr: np.ndarray, y_edge: np.ndarray, edge_weight: np.ndarray, dt: float = DT_MDL, l_d: float = 1.95,
             rise: float = 0.042) -> ClearanceResult:
    """gap: g(v), the GUI gap in use; w_r: the car's half width to its right side; c_ctr: the centre limit T_C'(x); y_edge, edge_weight: the
    filtered right edge and its weight in the right reference (B3); l_d: the device ahead of the no-slip point (B2 motion model); rise: the
    object bound's rise rate after its hold, r_ref."""
    res = ClearanceResult()
    fresh = self._health(radar, dt)
    if fresh and radar.new:
      self._ingest(radar, v, max(dt, 1e-3), dc)
    self._move_memory(vx0, yaw_rate, dt, l_d)
    if not fresh:  # a stale or faulty radar gives no objects; the caller fades Right line to Regular
      self.tracks = {k: t for k, t in self.tracks.items() if not t.seen}
    res.healthy = self.healthy
    res.right_blindspot = bool(radar.right_blindspot) if radar is not None else False
    m = gap + margin_extra(v)
    res.margin = m
    x_h = max(CORRIDOR_MIN, CORRIDOR_TIME * v)
    t_raw, d, nearest = NO_BOUND, -NO_BOUND, None
    for tr in self.tracks.values():
      if not tr.confirmed or tr.x > x_h or tr.x < FORGET_X:
        continue
      xq = min(max(tr.x, x_idxs[0]), x_idxs[-1])
      ctr = float(np.interp(xq, x_idxs, c_ctr))
      if tr.y <= ctr:
        continue  # not right of the centre limit: not in the right corridor
      if float(np.interp(xq, x_idxs, edge_weight)) >= EDGE_CLEAR_WEIGHT and tr.y > float(np.interp(xq, x_idxs, y_edge)) + ROADSIDE_BEYOND_EDGE:
        continue  # roadside beyond a clear edge
      res.objects += 1
      res.memory += int(not tr.seen)
      res.moving += int(tr.moving)
      res.popups += int(tr.popup)
      t_raw = min(t_raw, tr.y - w_r - m - dc)
      d_i = m - (tr.y - w_r - dc - ctr)
      if d_i > d:
        d, nearest = d_i, tr
    if nearest is not None:
      res.nearest = (nearest.x, nearest.y, nearest.y - w_r - dc)  # clearance from the car's right side where the car is now
    # held object bound: falls at once, rises after 1 s without a fall at 0.05 m/s
    if t_raw < self.t_obj - 1e-9:
      self.t_obj, self.hold = t_raw, 0.0
    else:
      self.hold += dt
      if self.hold >= BOUND_HOLD - 1e-9:
        self.t_obj = min(t_raw, self.t_obj + rise * dt) if t_raw < NO_BOUND else min(NO_BOUND, self.t_obj + rise * dt)
    res.t_obj_raw, res.t_obj, res.hold = t_raw, self.t_obj, self.hold
    # clearance yield (obstacle side right): starts at d > 0; released when d < -0.1 m (or no confirmed object is left, e.g. a track that
    # lost its confirmation for a message, R3-S3) for 1 s, or at once when the object that set it is 1 m behind the rear bumper
    res.deficit = d
    if self.yield_obj_x is not None:
      self.yield_obj_x -= vx0 * dt
    if d > 0.0:
      self.yield_latch, self.release_t, self.yield_obj_x = True, 0.0, nearest.x
    elif self.yield_latch:
      if nearest is not None and nearest.x > (self.yield_obj_x if self.yield_obj_x is not None else -NO_BOUND):
        self.yield_obj_x = nearest.x
      self.release_t = self.release_t + dt if (d < DEFICIT_RELEASE or nearest is None) else 0.0
      passed = self.yield_obj_x is not None and self.yield_obj_x < FORGET_X
      if self.release_t >= DEFICIT_RELEASE_TIME - 1e-9 or passed:
        self.yield_latch, self.release_t, self.yield_obj_x = False, 0.0, None
    res.yield_active = self.yield_latch
    return res


def clearance_q(deficit: float, q_no_growth: float, q_cap_start: float, q_cap_full: float) -> float:
  """The clearance deficit in the units of the B6 cap ramp: unlimited at d = 0 (q_cap_start), 0 at d = 0.05 m (q_cap_full) and beyond;
  q_no_growth while the yield is only latched (d <= 0)."""
  if deficit <= 0.0:
    return q_no_growth
  return float(q_cap_start + (q_cap_full - q_cap_start) * deficit / DEFICIT_RAMP)
