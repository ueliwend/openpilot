"""
Right-line Blendful ("keep right", task A8): follows the right lane line (or road edge) at a speed dependent gap, never left of the lane
centre, and hands back to regular upstream Blendful where the right reference is not usable.  Design: inster-dev
results/2026-11-12_a8_right_line_blendful/DESIGN.md rev 5.1 (section letters B.. below refer to it); this is implementation round 2
(results/2026-11-12_a8_impl_r2), which answers the E1 replay causes C1-C11 / K1 (results/2026-11-12_a8_replay).

Everything is expressed as a lateral plan shift (plan_correction.py): the profile amplitude k0 is the quantity that is blended, capped
and rate limited; the instantaneous desired-curvature correction is c = F k0 (F per frame with the plan's own v_x(0), C11).

Junctions (owner 20:55, Q10): in mode Right line a junction episode lets both corrections fade through the final limiter and the model
steers alone; Regular is unchanged.  Appendix J of the design (hold / latch / ramp context / split classifier) is documentation only.

Signs: model frame [forward, right, down], y > 0 is right; curvature > 0 is a right turn.
"""
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.modeld_v2.blendful import Blendful, LANE_WIDTH_DEFAULT, LINE_STD, LOOKAHEAD_TIME, MIN_LOOKAHEAD, MIN_SPEED
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.plan_correction import approach_length, fit_amplitude, instant_gain, profile_length, shift_plan, smoothstep

X_IDXS = np.array(ModelConstants.X_IDXS)
NX = len(X_IDXS)

# modes (BlendfulMode param: 0 Regular, 1 Right line; with Blendful off the mode is OFF)
MODE_OFF, MODE_REGULAR, MODE_RIGHT = 0, 1, 2

# geometry (B1, B8)
W_R = 0.81  # m, rear outer wheel edge from the car centreline
L_D = 1.95  # m, device ahead of the no-slip point
EDGE_EXTRA = 0.1  # m, a road edge counts this much further in than a line (owner 16:55)
GAP_SPEEDS = (50 / 3.6, 100 / 3.6)  # m/s, the gap is gap_low at and below 50 km/h, gap_high from 100 km/h
LINE_BOUND = 0.2  # m, rear wheel stays this far from the right line (plus off-tracking)
EDGE_BOUND = 0.3  # m, same for a clear right edge
# measurement filter (B2)
MEAS_TAU = 0.5  # s
COAST_Q, COAST_MAX, REINIT_Q = 0.3, 1.0, 0.5
# reference formation (B3)
PUSH = 1.5  # m, outward push of an unconfident candidate (weights only)
SOFTMIN_TAU = 0.05  # m
PLAUS_TOL, WIDTH_RANGE = 0.3, (2.0, 8.0)
# correction (B4, B11)
STRENGTH_RIGHT_ONLY = 0.8
RISING_STRENGTH = False  # owner 17:24: flat 0.8; True = 0.8 -> 1.0 with |k_m| v^2 over 0.18 .. 0.53 m/s^2 (one-line option)
FIT_WINDOW = (0.5, 1.5)  # of L_a
ERROR_CLIP = 1.0  # m
D_MAX = 1.0  # m, the fitted profile amplitude is clipped here (C9)
BUILD, RELEASE, FADING_RELEASE, SWERVE_RELEASE = 0.5, 2.0, 0.5, 5.0  # m/s^3, final limiter (blendful.py:57-60; 5 = controlsd's limit)
# authority (B5)
A_DEAD, A_DWELL, A_OUT, A_IN, A_HOLD = 0.05, 0.5, 1.0, 0.5, 1.0
# ground-fixed target (B5, C1; owner 21:44 question 9)
TARGET_RATE_LEFT = 0.25  # m/s, the target moving left (toward the lane centre, away from the right line)
MOVE_RATE_RIGHT = 0.05  # m/s, the move to the right-hand position (owner 21:44, question 9 option b)
LAG_BOUND, LAG_RATE = 1.0, 0.5  # m, m/s: where the target lags its raw value by more, that point moves at LAG_RATE (still glide-limited)
# yields (B6, C7, C8)
Q_NO_GROWTH, Q_CAP_START, Q_CAP_FULL, Q_RELEASE = 0.1, 0.15, 0.2, 0.1
GAP_SMOOTH_TAU, GAP_MEDIAN_S = 0.25, 0.5  # s, endpoint gap smoothing and the median window before the running minimum
USUAL_TAU, USUAL_TAU_AROUND = 10.0, 20.0  # s
MOVE_LAG_EPS, MOVE_SETTLE = 0.03, 3.0  # m, s: a move to the right is on while the target still glides or the correction builds, plus settle time (EST)
# jumps (B5 table, C6)
JUMP_LINE, JUMP_EDGE, JUMP_FRAMES, JUMP_HOLD, JUMP_WEIGHT = 0.6, 1.2, 2, 2.0, 0.5  # m, m, frames, s, candidate weight
CAND_CHANGE = 0.1  # change of a candidate's weight per frame that freezes the running minimum
# junctions (B7); all thresholds are EST start values, set on E1
CUE_WIDEN_FRAME, CUE_WIDEN_TIME, CUE_EDGE_DIV, CUE_EDGE_ANGLE = 0.5, 0.5, 0.6, 4.0  # m, m, m, deg
CUE_GORE_SEP, CUE_GORE_SPREAD, CUE_KINK, CUE_BEND = 0.6, 0.5, 4.0, 0.5  # m, m, deg, b
EPISODE_ON_TIME, EPISODE_QUIET_TIME, EPISODE_QUIET_DIST = 0.2, 2.0, 30.0  # s, s, m
EPISODE_WIDENING, EPISODE_WIDENING_TIME, EPISODE_MAX = 0.15, 1.0, 300.0  # m, s, m
WIDTH_TAU, WIDTH_STALE, WIDTH_REMEASURE = 10.0, 20.0, 2.0  # s
# road class (B5 table, C4)
ROAD_DWELL = 5.0  # s
BLINKER_MAX_SPEED = 32 / 3.6
# episode end reasons and road classes (log)
END_NONE, END_QUIET, END_WIDENING, END_MAX = 0, 1, 2, 3
ROAD_EDGES, ROAD_SHARED, ROAD_LANES, ROAD_MOTORWAY = 0, 1, 2, 3


def ss(z):
  return smoothstep(z)


def quad_interp(xq, xs, ys):
  """Three point Lagrange interpolation on the (non uniform) model grid, exact for parabolas; beyond the last point it extrapolates the
  last parabola (a frame moves the grid by v dt only), below the first point xq is clamped."""
  xq = np.maximum(np.asarray(xq, dtype=float), xs[0])
  j = np.clip(np.searchsorted(xs, xq) - 1, 0, len(xs) - 2)
  j = np.where(j > len(xs) - 3, len(xs) - 3, j)
  x0, x1, x2 = xs[j], xs[j + 1], xs[j + 2]
  y0, y1, y2 = ys[j], ys[j + 1], ys[j + 2]
  return (y0 * (xq - x1) * (xq - x2) / ((x0 - x1) * (x0 - x2)) + y1 * (xq - x0) * (xq - x2) / ((x1 - x0) * (x1 - x2)) +
          y2 * (xq - x0) * (xq - x1) / ((x2 - x0) * (x2 - x1)))


def gap_for_speed(v_ego: float, low: float, high: float) -> float:
  return float(np.interp(v_ego, GAP_SPEEDS, (low, high)))


def motion_predict(y_prev: np.ndarray, vx0: float, yaw_rate: float, dt: float, scale: float = 1.0) -> np.ndarray:
  """Own-motion prediction of a lateral position profile into the next car frame (B2): shifted by the model's v_x(0) dt, rotated by the
  yaw step, with the device's lever arm and the rear axle's own arc.  Shared by the input filters and the ground-fixed target.
  scale = vEgo / v_x(0): the lines' x axis carries the model's distance scale (x_model = x_true v_x(0) / vEgo), so the rotation acts at the
  true distance x_model * scale + l_d + vEgo dt / 2 (DESIGN check 19; for scale = 1 this is the formula of B2 verbatim)."""
  return quad_interp(X_IDXS + vx0 * dt, X_IDXS, y_prev) - yaw_rate * dt * (X_IDXS * scale + L_D + vx0 * scale * dt / 2)


class ConfGate:
  """Confidence gate with hysteresis: rising branch ss((q - 0.4) / 0.2), falling branch ss((q - 0.3) / 0.2) (B3)."""
  def __init__(self):
    self.c = 0.0

  def update(self, q: float) -> float:
    self.c = float(ss((q - 0.4) / 0.2) if self.c < 0.5 else ss((q - 0.3) / 0.2))
    return self.c


class InputFilter:
  """Motion compensated first order filter of one 33 point lateral position (B2).  Rev 5 (K1): the shift is the model's own speed
  v_x(0) of the raw plan times the true frame interval dt (the lane lines' x axis carries the model's distance scale)."""
  def __init__(self, tau: float = MEAS_TAU):
    self.tau = tau
    self.y = np.zeros(NX)
    self.y_pred = np.zeros(NX)
    self.innovation = np.zeros(NX)
    self.q = 0.0
    self.initialized = False
    self.gap = 0.0
    self.coast = False
    self.reinit = False

  def predict(self, vx0: float, yaw_rate: float, dt: float = DT_MDL, scale: float = 1.0) -> np.ndarray:
    return motion_predict(self.y, vx0, yaw_rate, dt, scale)

  def update(self, y_meas: np.ndarray, q_meas: float, vx0: float, yaw_rate: float, dt: float = DT_MDL,
             scale: float = 1.0) -> tuple[np.ndarray, float]:
    q_meas = float(np.clip(q_meas, 0.0, 1.0))
    self.reinit = False
    alpha = dt / (dt + self.tau)
    if self.initialized:
      pred = self.predict(vx0, yaw_rate, dt, scale)
      self.y_pred = pred
      self.innovation = y_meas - pred
      if q_meas >= COAST_Q:
        self.gap, self.coast = 0.0, False
        self.y = pred + alpha * q_meas * self.innovation
        self.q = q_meas
      else:
        self.gap += dt
        if self.gap <= COAST_MAX:
          self.coast = True
          self.y = pred
          self.q = q_meas * (1.0 - self.gap / COAST_MAX)
        else:
          self.initialized, self.coast = False, False
          self.y, self.q = np.array(y_meas, dtype=float), 0.0
    else:
      self.y = np.array(y_meas, dtype=float)
      self.y_pred = self.y.copy()
      self.innovation = np.zeros(NX)
      self.coast = False
      if q_meas > REINIT_Q:
        self.initialized, self.reinit, self.gap = True, True, 0.0
        self.q = q_meas
      else:
        self.q = 0.0
    return self.y, self.q


class GroundTarget:
  """Ground-fixed, rate-limited target (B5, C1, R5-3).  Last frame's limited target is predicted into the new car frame with the filters'
  motion model and then moves toward the raw target by at most rate x dt per point: to the right (the move to the right-hand position)
  MOVE_RATE_RIGHT, to the left TARGET_RATE_LEFT.  Where it lags the raw target by more than LAG_BOUND the rate rises to LAG_RATE (still
  glide-limited), so a systematic prediction error cannot make it drift away.  Points beyond the last frame's horizon take the raw
  target.  Its own state cannot jump: a flickering or re-indexed reference moves it by at most rate x dt per point and frame."""
  def __init__(self):
    self.t: np.ndarray | None = None
    self.lag = np.zeros(NX)

  def reset(self):
    self.t = None
    self.lag = np.zeros(NX)

  def update(self, t_raw: np.ndarray, vx0: float, yaw_rate: float, dt: float, scale: float = 1.0) -> np.ndarray:
    if self.t is None:
      self.t = np.array(t_raw, dtype=float)
      self.lag = np.zeros(NX)
      return self.t.copy()
    pred = motion_predict(self.t, vx0, yaw_rate, dt, scale)
    pred = np.where(X_IDXS + vx0 * dt > X_IDXS[-1], t_raw, pred)
    diff = t_raw - pred  # > 0: the raw target is further right
    big = np.abs(diff) > LAG_BOUND
    up = np.where(big, LAG_RATE, MOVE_RATE_RIGHT) * dt
    down = np.where(big, LAG_RATE, TARGET_RATE_LEFT) * dt
    self.t = pred + np.clip(diff, -down, up)
    self.lag = t_raw - self.t
    return self.t.copy()


class Authority:
  """a follows a* with a dead band, a dwell, rate limits and a re-entry hold (B5, check 12)."""
  def __init__(self):
    self.a = 0.0
    self.a_cmd = 0.0
    self.hold = 0.0
    self.dwell = A_DWELL

  def update(self, a_star: float) -> float:
    dt = DT_MDL
    self.dwell += dt
    snap = a_star <= 1e-9 or a_star >= 1.0 - 1e-9  # the ends are reached exactly (a = 0 is regular upstream, a = 1 is full)
    if (abs(a_star - self.a_cmd) > (1e-9 if snap else A_DEAD)) and self.dwell >= A_DWELL:
      self.a_cmd, self.dwell = a_star, 0.0
    if self.a_cmd < self.a - 1e-9:
      self.hold = A_HOLD
      self.a = max(self.a_cmd, self.a - A_OUT * dt)
    else:
      self.hold = max(0.0, self.hold - dt)
      if self.hold <= 0.0:
        self.a = min(self.a_cmd, self.a + A_IN * dt)
    return self.a


class Timed:
  """Boolean that turns on after `on_time` of the condition being true and off only after `off_time` of it being false."""
  def __init__(self, off_time: float = 0.0, on_time: float = 0.0):
    self.off_time, self.on_time = off_time, on_time
    self.state, self.t_on, self.t_off = False, 0.0, 0.0

  def update(self, cond: bool) -> bool:
    if cond:
      self.t_on, self.t_off = self.t_on + DT_MDL, 0.0
      if self.t_on >= self.on_time:
        self.state = True
    else:
      self.t_on, self.t_off = 0.0, self.t_off + DT_MDL
      if self.t_off >= self.off_time:
        self.state = False
    return self.state


class JumpDetector:
  """Per-input jump detection (B5 table, C6): the innovation at x = L of the right line (threshold 0.6 m) or right edge (1.2 m) above its
  threshold for JUMP_FRAMES frames in a row, only on a candidate whose weight in the reference is >= 0.5 (and a confident input); the
  trip is held for JUMP_HOLD."""
  def __init__(self):
    self.count = [0, 0]
    self.hold = 0.0

  def update(self, innovations: tuple[float, float], weights: tuple[float, float], confident: tuple[bool, bool], dt: float = DT_MDL) -> bool:
    tripped = False
    for i, thr in enumerate((JUMP_LINE, JUMP_EDGE)):
      over = abs(innovations[i]) > thr and weights[i] >= JUMP_WEIGHT and confident[i]
      self.count[i] = self.count[i] + 1 if over else 0
      tripped |= self.count[i] >= JUMP_FRAMES
    self.hold = JUMP_HOLD if tripped else max(0.0, self.hold - dt)
    return self.hold > 0.0


class RoadClass:
  """Road class with hysteresis and a 5 s dwell (B5 table, C4); it resets no estimator.  Classes: edges only (no line), shared (narrow),
  lanes, motorway.  Thresholds: width 3.0 +- 0.15 m (motorway 3.3 +- 0.15 m), speed 22 +- 1 m/s, line quality 0.3 (enter) / 0.4 (leave)."""
  def __init__(self):
    self.cls: int | None = None
    self.cand: int | None = None
    self.t = 0.0

  def raw(self, v: float, q_l: float, q_r: float, width: float) -> int:
    cur = self.cls
    thr_q = 0.4 if cur == ROAD_EDGES else 0.3
    if q_l < thr_q and q_r < thr_q:
      return ROAD_EDGES
    if width < (3.15 if cur == ROAD_SHARED else 2.85):
      return ROAD_SHARED
    fast = v > (21.0 if cur == ROAD_MOTORWAY else 23.0)
    wide = width > (3.15 if cur == ROAD_MOTORWAY else 3.45)
    return ROAD_MOTORWAY if fast and wide else ROAD_LANES

  def update(self, v: float, q_l: float, q_r: float, width: float, dt: float = DT_MDL) -> int:
    new = self.raw(v, q_l, q_r, width)
    if self.cls is None:
      self.cls = new
    elif new == self.cls:
      self.cand, self.t = None, 0.0
    else:
      if new != self.cand:
        self.cand, self.t = new, 0.0
      self.t += dt
      if self.t >= ROAD_DWELL:
        self.cls, self.cand, self.t = new, None, 0.0
    return int(self.cls)


# ---------------------------------------------------------------------------------------------------------------------------------
# shape detectors (B7)
def bend_score(x: np.ndarray, y_right: np.ndarray, y_ref: np.ndarray, v_ego: float) -> tuple[float, float, float, float]:
  """Relative shape of the right line against the reference over [2 m, max(1.5 L, 25 m)]: d(x) = alpha x + beta x^2 / 2;
  b = max(ss((beta - 1/100) / (1/40 - 1/100)), ss((psi_end - th) / th)), th = 10 deg at <= 50 km/h falling to 3 deg at >= 100 km/h."""
  look = max(1.5 * v_ego, MIN_LOOKAHEAD)
  m = (x >= 2.0) & (x <= max(1.5 * look, 25.0))
  xm = x[m] - x[m][0]
  d = (y_right[m] - y_right[m][0]) - (y_ref[m] - y_ref[m][0])
  alpha, beta = np.linalg.lstsq(np.stack([xm, xm ** 2 / 2], 1), d, rcond=None)[0]
  psi_end = np.degrees(np.arctan(alpha + beta * xm[-1]))
  th = float(np.interp(v_ego * 3.6, (50.0, 100.0), (10.0, 3.0)))
  b = max(float(ss((beta - 1 / 100) / (1 / 40 - 1 / 100))), float(ss((psi_end - th) / th)))
  return float(alpha), float(beta), float(psi_end), float(b)


def cue_window(v_ego: float) -> tuple[float, float]:
  """The window of the in-frame widening and edge divergence cues: x = 0.25 L .. 1.5 L, or [5, 60] m while L sits at its floor (low speed)."""
  look = max(v_ego * LOOKAHEAD_TIME, MIN_LOOKAHEAD)
  if v_ego * LOOKAHEAD_TIME <= MIN_LOOKAHEAD:
    return 5.0, 60.0
  return 0.25 * look, 1.5 * look


def growth(x: np.ndarray, d: np.ndarray, xa: float, xb: float) -> float:
  """d(xb) - d(xa) of a profile d(x)."""
  return float(np.interp(xb, x, d) - np.interp(xa, x, d))


def cue_widening_frame(x: np.ndarray, y_l: np.ndarray, y_r: np.ndarray, v_ego: float) -> float:
  """Cue 1: growth of the lane width between the references along x, within the frame (no memory needed)."""
  xa, xb = cue_window(v_ego)
  return growth(x, y_r - y_l, xa, xb)


def cue_edge_divergence(x: np.ndarray, line_r: np.ndarray | None, edge_r: np.ndarray, y_l: np.ndarray | None, v_ego: float) -> tuple[float, float]:
  """Cue 3: how far the right road edge diverges (to the right) from the right line or from the left reference over the window; returns
  (growth in m, relative angle in degrees)."""
  xa, xb = cue_window(v_ego)
  g = [growth(x, edge_r - ref, xa, xb) for ref in (line_r, y_l) if ref is not None]
  best = max(g) if g else 0.0
  return best, float(np.degrees(np.arctan(best / (xb - xa))))


def cue_gore(x: np.ndarray, near: np.ndarray, far: np.ndarray) -> float | None:
  """Cue 4a: V shape.  The separation of the right line and the far-right line (or the right edge) falls to its minimum and rises again; the
  minimum (m) is returned when the separation has fallen by >= CUE_GORE_SPREAD to it and risen by >= CUE_GORE_SPREAD beyond it, else None."""
  m = (x >= 5.0) & (x <= 100.0)
  sep = (far - near)[m]
  i = int(np.argmin(sep))
  if i == 0 or i == len(sep) - 1:
    return None
  smin = float(sep[i])
  if np.max(sep[:i]) - smin >= CUE_GORE_SPREAD and np.max(sep[i + 1:]) - smin >= CUE_GORE_SPREAD:
    return smin
  return None


def cue_kink(x: np.ndarray, y_r: np.ndarray, y_ref: np.ndarray, v_ego: float) -> float:
  """Cue 4b: kink (deg) of the right line relative to the reference between two neighbouring windows [5 m, xm] and [xm, x_end]."""
  look = max(v_ego * LOOKAHEAD_TIME, MIN_LOOKAHEAD)
  x_end = max(1.5 * look, 25.0)
  xm = 0.5 * (5.0 + x_end)
  d = y_r - y_ref
  slopes = []
  for lo, hi in ((5.0, xm), (xm, x_end)):
    k = (x >= lo) & (x <= hi)
    slopes.append(float(np.polyfit(x[k], d[k], 1)[0]))
  return float(np.degrees(np.arctan(slopes[1]) - np.arctan(slopes[0])))


class WidthMemory:
  """Short width memory for the 'widening over time' cue only (B7, C2).  Low-passes the measured width (tau 10 s) whenever both references
  are clear and no episode runs; decreases are always accepted.  Stale after 20 s without an update (cue off while stale).  After an episode,
  when stale, or on a change of the reference type (lines / edges) it is re-measured: the median of the next 2 s of clear measurements."""
  def __init__(self):
    self.w = 0.0
    self.age = 0.0
    self.armed = False
    self.key = None
    self.buf: list[float] = []

  def remeasure(self):
    self.armed, self.buf = False, []

  def end_of_episode(self):
    self.remeasure()

  @property
  def stale(self) -> bool:
    return self.armed and self.age > WIDTH_STALE

  def update(self, width: float, clear: bool, episode: bool, key, dt: float = DT_MDL) -> None:
    if key != self.key:
      self.key = key
      self.remeasure()
    self.age += dt
    if self.stale:
      self.remeasure()
    if not clear or episode:
      return
    width = float(np.clip(width, *WIDTH_RANGE))
    if not self.armed:
      self.buf.append(width)
      if len(self.buf) * dt >= WIDTH_REMEASURE - 1e-9:
        self.w, self.armed, self.age, self.buf = float(np.median(self.buf)), True, 0.0, []
    else:
      self.w += (width - self.w) * dt / (dt + WIDTH_TAU)
      self.age = 0.0

  def excess(self, width: float) -> float | None:
    """Width minus the memory (cue 2), or None while the memory is not armed or stale."""
    return None if (not self.armed or self.stale) else float(width - self.w)


class JunctionEpisodes:
  """Junction episode state (B7): starts when any cue has held for 0.2 s; ends after no cue for 2 s and 30 m, or when no cue is active and
  the widening has been below 0.15 m for 1 s, or after 300 m (then the cues must clear once before a new episode can start)."""
  def __init__(self):
    self.reset()

  def reset(self):
    self.episode = False
    self.cue_t = 0.0
    self.quiet_t = self.quiet_d = self.low_t = 0.0
    self.dist = 0.0
    self.blocked = False
    self.ended = END_NONE
    self.model_alone_time = 0.0

  def update(self, cue: bool, widening: float | None, v: float, dt: float = DT_MDL) -> bool:
    ds = max(v, 0.0) * dt
    self.ended = END_NONE
    if not self.episode:
      if self.blocked:
        if not cue:
          self.blocked = False
        self.cue_t = 0.0
        return False
      self.cue_t = self.cue_t + dt if cue else 0.0
      if self.cue_t >= EPISODE_ON_TIME - 1e-9:
        self.episode, self.dist, self.quiet_t, self.quiet_d, self.low_t = True, 0.0, 0.0, 0.0, 0.0
      return self.episode
    self.dist += ds
    self.model_alone_time += dt
    if cue:
      self.quiet_t = self.quiet_d = 0.0
    else:
      self.quiet_t, self.quiet_d = self.quiet_t + dt, self.quiet_d + ds
    self.low_t = self.low_t + dt if (not cue and widening is not None and widening < EPISODE_WIDENING) else 0.0
    if self.dist >= EPISODE_MAX:
      self.ended, self.blocked = END_MAX, True
    elif self.quiet_t >= EPISODE_QUIET_TIME and self.quiet_d >= EPISODE_QUIET_DIST:
      self.ended = END_QUIET
    elif self.low_t >= EPISODE_WIDENING_TIME - 1e-9:
      self.ended = END_WIDENING
    if self.ended != END_NONE:
      self.episode, self.cue_t = False, 0.0
    return self.episode


# ---------------------------------------------------------------------------------------------------------------------------------
class EndpointGap:
  """Model-independent yield trigger (B6, C7, R5-4).  x = Y_ref(L) - plan_y(L): the plan endpoint's gap to a reference (right reference, else
  left reference, else the plan alone: x = -plan_y); x grows when the plan moves left, shrinks when the car moves right.  Smoothed 0.25 s.
  Reference R_P: during a move to the right the running minimum of the 0.5 s median of x since the move started (not updated on jump frames
  or frames with a candidate weight change > 0.1); otherwise x low-passed 10 s (20 s while going around), like upstream's usual.
  q = |x - R_P| - 0.25 |k_m| L^2 / 2; departure = x - R_P (> 0: the model moves left of its usual place: obstacle on the right)."""
  def __init__(self):
    self.reset()

  def reset(self):
    self.source: int | None = None
    self.ps: float | None = None
    self.hist: deque = deque()
    self.rp = 0.0
    self.moving = False
    self.q = 0.0
    self.departure = 0.0

  def update(self, x: float, source: int, moving: bool, skip: bool, allowance: float, going_around: bool, dt: float = DT_MDL) -> float:
    n_med = max(1, int(round(GAP_MEDIAN_S / dt)))
    if source != self.source or self.ps is None:
      self.source, self.ps, self.rp, self.moving = source, x, x, False
      self.hist = deque([x], maxlen=n_med)
    else:
      self.ps += (x - self.ps) * dt / (dt + GAP_SMOOTH_TAU)
      self.hist.append(self.ps)
    med = float(np.median(self.hist))
    if moving:
      if not self.moving:
        self.rp = med
      elif not skip:
        self.rp = min(self.rp, med)
    elif not skip:
      tau = USUAL_TAU_AROUND if going_around else USUAL_TAU
      self.rp += (self.ps - self.rp) * dt / (dt + tau)
    self.moving = moving
    self.departure = float(self.ps - self.rp)
    self.q = float(abs(self.departure) - allowance)
    return self.q


# ---------------------------------------------------------------------------------------------------------------------------------
# Junction policy (B7, owner 20:55 and Q10): the ONLY place that decides what a junction episode does to the correction.
@dataclass
class PolicyWeights:
  w_rl: float = 0.0  # right-line part
  w_up: float = 0.0  # upstream part (reference instance)

  def total(self) -> float:
    return self.w_rl + self.w_up


def junction_policy(episode: bool, a: float, right_line_mode: bool = True) -> PolicyWeights:
  """Per-frame split of the correction: k0 = w_rl k0_rl + w_up k0_up.  Outside an episode the right-line authority a blends with regular
  upstream.  Inside an episode, in mode Right line only (Q10: Regular unchanged), both fade out through the final limiter and the model
  steers alone: no Regular, no hold."""
  if episode and right_line_mode:
    return PolicyWeights()
  return PolicyWeights(a, 1.0 - a)


# ---------------------------------------------------------------------------------------------------------------------------------
@dataclass
class RightInputs:
  """Per frame inputs of RightLineBlendful besides the model output."""
  gap_low: float = 0.5
  gap_high: float = 1.0
  c_set: float = 0.0  # CameraOffset setting = the device's mounting offset
  c_applied: float = 0.0  # the offset actually applied to the model frame
  left_blinker: bool = False
  right_blinker: bool = False
  frame_dt: float = DT_MDL  # true interval between model frames (timestampEof), s
  ref_going_around: bool = False  # the reference instance's going-around yield latch
  ref_yield_fade: float = 1.0  # the reference instance's going-around / away fade (< 1: upstream sees an obstacle)
  ref_lane_width: float = LANE_WIDTH_DEFAULT  # the reference instance's filtered lane width (left limit with one reference)
  building: bool = False  # last frame's final limiter was still building the correction toward the right


@dataclass
class RightResult:
  k0: float = 0.0  # profile amplitude before the yield cap and the final limiter
  fading: bool = False
  yield_active: bool = False
  no_growth: bool = False  # q >= 0.1 or a yield is active: the cap logic of BlendfulCorrection acts
  yield_q: float = 0.0
  yield_side: int = 0  # +1: obstacle on the right (model moves left), -1: obstacle on the left; latched at the start of a yield
  episode: bool = False
  log: dict = field(default_factory=dict)


class RightLineBlendful:
  def __init__(self, strength_rising: bool | None = None):
    self.strength_rising = RISING_STRENGTH if strength_rising is None else strength_rising
    self.filters = [InputFilter() for _ in range(4)]  # left line, left edge, right line, right edge
    self.gates = [ConfGate() for _ in range(4)]
    self.reset()

  def reset(self):
    for f in self.filters:
      f.__init__()
    for g in self.gates:
      g.c = 0.0
    self.line_probs = FirstOrderFilter(np.zeros(2), 0.5, DT_MDL)
    self.edge_q = FirstOrderFilter(np.zeros(2), 0.5, DT_MDL)
    self.quality_gate = ConfGate()
    self.authority = Authority()
    self.episodes = JunctionEpisodes()
    self.width_mem = WidthMemory()
    self.ground = GroundTarget()
    self.gap = EndpointGap()
    self.jumps = JumpDetector()
    self.road = RoadClass()
    self.bend_active = Timed(off_time=1.0)
    self.blink_on = Timed(off_time=1.0, on_time=0.3)
    self.f_blink = 1.0
    self.yield_latch = Timed(off_time=1.0)
    self.ref_yield_latch = Timed(off_time=1.0)
    self.yield_phase = False
    self.yield_side = 1
    self.moving_t = 0.0
    self.w_prev = None
    self.no_growth_prev = False

  @property
  def a(self) -> float:
    return self.authority.a

  # ----------------------------------------------------------------------------------------------------------------------------
  @staticmethod
  def _reference(ys, qs, cs, side: int):
    """B3: soft minimum of the (confidence pushed) candidates gives the weights, the reference is the weighted mean of the TRUE positions."""
    pushed = side * (ys + side * (1.0 - np.asarray(cs))[:, None] * PUSH)
    w = np.exp(-(pushed - pushed.min(axis=0)) / SOFTMIN_TAU)
    w /= w.sum(axis=0)
    return (w * ys).sum(axis=0), (w * np.asarray(qs)[:, None]).sum(axis=0), w

  def _correction(self, target: np.ndarray, plan_x, plan_y, v: float, strength: float):
    """Profile-basis fit (B4); the fitted amplitude D is clipped at D_MAX (C9).  Returns (D, strength k0, k0, D was clipped)."""
    la = approach_length(max(v, 1.0))
    l_c = profile_length(max(v, 1.0))
    win = (X_IDXS >= FIT_WINDOW[0] * la) & (X_IDXS <= FIT_WINDOW[1] * la)
    x = X_IDXS[win]
    err = np.clip(target[win] - np.interp(x, plan_x, plan_y), -ERROR_CLIP, ERROR_CLIP)
    d, _ = fit_amplitude(x, err, 1.0, l_c)
    clipped = abs(d) > D_MAX
    d = float(np.clip(d, -D_MAX, D_MAX))
    k = 6.0 * d / l_c ** 2
    return d, strength * k, k, clipped

  def update(self, mo: dict, v_ego: float, active: bool, model_curvature: float, k0_up: float, p: RightInputs,
             force_regular: bool = False) -> RightResult:
    """force_regular: mode Regular (hand-back window): a* = 0 and the junction policy does not act (Q10)."""
    t0 = time.perf_counter()
    dt = float(p.frame_dt) if 0.2 * DT_MDL <= p.frame_dt <= 5.0 * DT_MDL else DT_MDL
    v = max(float(v_ego), 0.0)
    fading = (not active) or v < MIN_SPEED
    look = max(v * LOOKAHEAD_TIME, MIN_LOOKAHEAD)
    iL = look  # look-ahead distance for scalar read outs
    lines = mo['lane_lines'][0, :, :, 0]
    lstds = mo['lane_lines_stds'][0, :, :, 0]
    plan = mo['plan'][0]
    plan_x, plan_y = plan[:, 0], plan[:, 1]
    yaw_rate = float(plan[0, Plan.ORIENTATION_RATE.start + 2])
    vx0 = float(plan[0, Plan.VELOCITY.start])  # the model's own speed: the x axis of the lines carries its distance scale (K1, C11)
    if not np.isfinite(vx0) or vx0 < 1.0:
      vx0 = max(v, 0.0)
    scale = float(np.clip(v / vx0, 0.8, 1.25)) if (v >= 1.0 and vx0 >= 1.0) else 1.0  # vEgo / v_x(0): the true distance per model-axis metre
    probs_raw = mo['lane_lines_prob'][0, 1::2]
    probs = self.line_probs.update(probs_raw[1:3])
    l_q = float(probs[0] * np.interp(np.interp(0.5 * look, X_IDXS, lstds[1]), LINE_STD, [1.0, 0.0]))
    r_q = float(probs[1] * np.interp(np.interp(0.5 * look, X_IDXS, lstds[2]), LINE_STD, [1.0, 0.0]))
    if 'road_edges' in mo and 'road_edges_stds' in mo:
      edges = mo['road_edges'][0, :, :, 0]
      estds = mo['road_edges_stds'][0, :, :, 0]
      eq_raw = np.array([np.interp(np.interp(0.5 * look, X_IDXS, estds[i]), (0.3, 1.0), (0.8, 0.0)) for i in range(2)])
    else:
      edges = np.stack([np.full(NX, -1e3), np.full(NX, 1e3)])  # absent: far outside whatever the road does
      eq_raw = np.zeros(2)
    e_q = self.edge_q.update(eq_raw)
    # --- B2: the four filters (left line, left edge, right line, right edge)
    raw = [lines[1], edges[0], lines[2], edges[1]]
    qraw = [l_q, float(e_q[0]), r_q, float(e_q[1])]
    filt, qf = [], []
    for f, y, q in zip(self.filters, raw, qraw, strict=True):
      yy, qq = f.update(y, q, vx0, yaw_rate, dt, scale)
      filt.append(yy.copy())
      qf.append(qq)
    t_filter = (time.perf_counter() - t0) * 1e3
    # --- B3: references; the edge counts EDGE_EXTRA further in
    cands_r = np.stack([filt[2], filt[3] - EDGE_EXTRA])
    cands_l = np.stack([filt[0], filt[1] + EDGE_EXTRA])
    c_r = [self.gates[2].update(qf[2]), self.gates[3].update(qf[3])]
    c_l = [self.gates[0].update(qf[0]), self.gates[1].update(qf[1])]
    y_r, qa_r, w_r = self._reference(cands_r, [qf[2], qf[3]], c_r, +1)
    y_l, qa_l, w_l = self._reference(cands_l, [qf[0], qf[1]], c_l, -1)
    q_R, q_L = float(np.interp(iL, X_IDXS, qa_r)), float(np.interp(iL, X_IDXS, qa_l))
    # --- plausibility (B3)
    bad = False
    for i_line, i_edge, side in ((2, 3, +1), (0, 1, -1)):
      if qf[i_line] > 0.5 and qf[i_edge] > 0.5:
        line_l, edge_l = float(np.interp(iL, X_IDXS, filt[i_line])), float(np.interp(iL, X_IDXS, filt[i_edge]))
        bad |= side * (line_l - edge_l) > PLAUS_TOL
    width_now = float(np.interp(iL, X_IDXS, y_r - y_l))
    both = q_R > 0.5 and q_L > 0.5
    if both and not (WIDTH_RANGE[0] <= width_now <= WIDTH_RANGE[1]):
      bad = True
    f_plaus_t = 0.0 if bad else 1.0
    # --- jumps (C6): per-input thresholds, 2-frame persistence, only on a candidate with weight >= 0.5
    w_at = (float(np.interp(iL, X_IDXS, w_r[0])), float(np.interp(iL, X_IDXS, w_r[1])))
    inno = tuple(float(np.interp(iL, X_IDXS, self.filters[i].innovation)) for i in (2, 3))
    jump_on = self.jumps.update(inno, w_at, (qf[2] > 0.5, qf[3] > 0.5), dt)
    f_jump_t = 0.0 if jump_on else 1.0
    cand_changed = self.w_prev is not None and max(abs(w_at[0] - self.w_prev[0]), abs(w_at[1] - self.w_prev[1])) > CAND_CHANGE
    self.w_prev = w_at
    # --- left limit T_C': the centre between the references (unclipped width with two references, upstream's filtered width with one)
    if both:
      hw = (y_r - y_l) / 2
      c_ctr, _ = Blendful.centre_and_weight(y_l, y_r, lstds[1], lstds[2], max(q_L, 1e-4), max(q_R, 1e-4), hw)
    elif q_R >= q_L:
      c_ctr = y_r - p.ref_lane_width / 2
    else:
      c_ctr = y_l + p.ref_lane_width / 2
    # --- B7 shape detectors and cues
    use_left = q_L > 0.5
    y_ref = y_l if use_left else np.interp(X_IDXS, plan_x, plan_y)
    _, beta, _, b = bend_score(X_IDXS, y_r, y_ref, max(v, 1.0))
    bend_on = self.bend_active.update(b > 0.5 if not self.bend_active.state else b >= 0.2)
    f_bend_t = 0.0 if bend_on else 1.0
    vv = max(v, 1.0)
    clear_r = q_R > 0.5
    key = ('lines' if w_at[0] >= 0.5 else 'edge', 'lines' if float(np.interp(iL, X_IDXS, w_l[0])) >= 0.5 else 'edge')
    width_mid = float(np.interp(0.5 * look, X_IDXS, y_r - y_l))
    self.width_mem.update(width_mid, both and not bad, self.episodes.episode, key, dt)
    c1 = cue_widening_frame(X_IDXS, y_l, y_r, vv) if both else None
    c2 = self.width_mem.excess(width_mid) if both else None
    ed_g, ed_a = (0.0, 0.0)
    ed_ok = qf[3] > 0.5 and (qf[2] > 0.5 or q_L > 0.5)
    if ed_ok:
      ed_g, ed_a = cue_edge_divergence(X_IDXS, filt[2] if qf[2] > 0.5 else None, filt[3], y_l if q_L > 0.5 else None, vv)
    gore_min = None
    if float(probs_raw[2]) > 0.5 and float(probs_raw[3]) > 0.5:
      gore_min = cue_gore(X_IDXS, lines[2], lines[3])
    if gore_min is None and qf[2] > 0.5 and qf[3] > 0.5:
      gore_min = cue_gore(X_IDXS, lines[2], edges[1])
    kink = cue_kink(X_IDXS, filt[2], y_ref, vv) if clear_r else 0.0
    cue_flags = (c1 is not None and c1 > CUE_WIDEN_FRAME, c2 is not None and c2 > CUE_WIDEN_TIME,
                 ed_ok and (ed_g > CUE_EDGE_DIV or ed_a > CUE_EDGE_ANGLE),
                 (gore_min is not None and gore_min < CUE_GORE_SEP) or abs(kink) > CUE_KINK, clear_r and b >= CUE_BEND)
    cue_any = any(cue_flags)
    widening_vals = [c for c in (c1, c2, ed_g if ed_ok else None) if c is not None]
    widening = max(widening_vals) if widening_vals else None
    was_episode = self.episodes.episode
    in_episode = self.episodes.update(cue_any, widening, v, dt)
    if was_episode and not in_episode:  # the episode ended: re-measure the width, restart the target from the raw target
      self.width_mem.end_of_episode()
      self.ground.reset()
    jn_active = in_episode and not force_regular
    # --- blinker fade (left blinker hands back to upstream; B9)
    lb = self.blink_on.update(p.left_blinker)
    self.f_blink = max(0.0, self.f_blink - dt) if lb else min(1.0, self.f_blink + dt)
    # --- road class (log / reports only; resets no estimator)
    road = self.road.update(v, q_L, q_R, width_now, dt)
    # --- targets (B1, B8): references -> raw target -> rate limit -> bounds
    g = gap_for_speed(v, p.gap_low, p.gap_high)
    win_ref = (X_IDXS >= 0.5 * look) & (X_IDXS <= 1.5 * look)
    k_ref = 2.0 * float(np.polyfit(X_IDXS[win_ref], y_r[win_ref], 2)[0])
    off_track = L_D ** 2 * max(k_ref, 0.0) / 2
    dc = p.c_set - p.c_applied
    t_r_line = y_r - W_R - g - off_track - dc
    t_raw = np.maximum(t_r_line, c_ctr)
    t_lim = self.ground.update(t_raw, vx0, yaw_rate, dt, scale)
    t_left = np.maximum(t_lim, c_ctr)  # the left limit acts after the rate limit
    t_bound_line = y_r - W_R - LINE_BOUND - off_track - dc
    w_e = float(ss((qf[3] - 0.4) / 0.2))
    t_bound_edge = filt[3] - W_R - EDGE_BOUND - off_track - dc
    t_bounded = np.minimum(t_left, t_bound_line)
    target = t_bounded - w_e * np.maximum(0.0, t_bounded - t_bound_edge)
    bound_active = bool(np.any(target < t_left - 1e-9))
    lag_L = float(np.interp(iL, X_IDXS, self.ground.lag))
    # --- correction (B4)
    w_c = float(np.interp(abs(model_curvature) * v ** 2, [0.18, 0.53], [STRENGTH_RIGHT_ONLY, 1.0])) if self.strength_rising else STRENGTH_RIGHT_ONLY
    s_r = 1.0 - (1.0 - l_q) * (1.0 - w_c * q_R)
    d_rl, k0_rl, k_fit, d_clipped = self._correction(target, plan_x, plan_y, v, s_r)
    # --- yields (B6, C7, C8): the endpoint-gap trigger on the right reference, else the left one, else the plan alone
    if q_R >= 0.3:
      src, ref_L = 1, float(np.interp(look, X_IDXS, y_r))
    elif q_L >= 0.3:
      src, ref_L = 2, float(np.interp(look, X_IDXS, y_l))
    else:
      src, ref_L = 0, 0.0
    x_gap = ref_L - float(np.interp(look, plan_x, plan_y))
    allow = 0.25 * abs(model_curvature) * look ** 2 / 2
    move_now = (not self.no_growth_prev) and (lag_L > MOVE_LAG_EPS or self.authority.a < self.authority.a_cmd - 1e-6 or p.building)
    self.moving_t = MOVE_SETTLE if move_now else max(0.0, self.moving_t - dt)
    skip = (not f_jump_t) or cand_changed
    q_y = self.gap.update(x_gap, src, self.moving_t > 0.0, bool(skip), allow, self.ref_around(p), dt) if not fading else 0.0
    dep = self.gap.departure
    car_trig = self.yield_latch.update(q_y > Q_CAP_FULL if not self.yield_latch.state else q_y >= Q_RELEASE)
    ref_trig = self.ref_yield_latch.update(self.ref_around(p))
    yield_active = bool(car_trig or ref_trig) and not fading
    no_growth = bool(q_y >= Q_NO_GROWTH or yield_active) and not fading
    if no_growth and not self.yield_phase:  # the obstacle side is latched at the start of a yield and kept until it ends
      self.yield_side = 1 if dep > 0 else -1
    self.yield_phase = no_growth
    self.no_growth_prev = no_growth
    side = self.yield_side
    q_eff = max(q_y, Q_CAP_FULL) if ref_trig else q_y
    f_yl = 0.0 if (yield_active and side > 0) else 1.0
    f_yr = float(np.interp(q_eff, (0.2, 0.45), (1.0, 0.0))) if (yield_active and side < 0) else 1.0
    k0_rl *= f_yr
    # --- authority (B5)
    qa = self.quality_gate.update(q_R)
    a_star = float(qa)
    f_ep = 0.0 if jn_active else 1.0
    a_star *= f_bend_t * f_jump_t * f_plaus_t * self.f_blink * f_yl * f_ep
    if force_regular or fading:
      a_star = 0.0
    if q_y >= Q_NO_GROWTH:
      a_star = min(a_star, self.authority.a_cmd)
    a = self.authority.update(a_star)
    wts = junction_policy(in_episode, a, right_line_mode=not force_regular)
    k0 = wts.w_rl * k0_rl + wts.w_up * k0_up
    if fading:
      k0 = 0.0
    right_only = bool(q_R > 0.5 and l_q < 0.3)
    gap_now = float(np.interp(0.0, X_IDXS, y_r) - W_R)
    em = self.episodes
    cue_bits = sum(int(f) << i for i, f in enumerate(cue_flags))
    wm = self.width_mem
    log = {
      'applied': a > 0.0, 'gap': g, 'gap_now': gap_now, 'q_R': q_R, 'q_L': q_L, 'a': a, 'a_star': a_star,
      'f_bend': f_bend_t, 'f_yl': f_yl, 'f_yr': f_yr, 'f_jump': f_jump_t, 'f_plaus': f_plaus_t, 'f_blink': self.f_blink,
      'away_car': q_y, 's_r': s_r, 'w_c': w_c, 'right_only': right_only,
      'curve_accel': abs(model_curvature) * v ** 2, 'yield_active': yield_active, 'bound_active': bound_active,
      'k_rl': k0_rl, 'weights': (wts.w_rl, wts.w_up), 'l_c': profile_length(max(v, 1.0)), 'd': d_rl, 'd_clipped': d_clipped, 'filter_ms': t_filter,
      'filtered': np.concatenate([[np.interp(xq, X_IDXS, f) for xq in (0.0, 0.5 * look, look, 1.5 * look)] for f in filt]),
      'raw': np.concatenate([[np.interp(xq, X_IDXS, r_) for xq in (0.0, 0.5 * look, look, 1.5 * look)] for r_ in raw]),
      'filter_offset': np.array([float(np.interp(iL, X_IDXS, f - r_)) for f, r_ in zip(filt, raw, strict=True)]),
      'confidence': np.array(qf), 'state': np.array([int(f.coast) | (int(f.reinit) << 1) for f in self.filters]),
      'cand_conf': np.array([c_l[0], c_l[1], c_r[0], c_r[1]]),
      'cand_weight': np.array([float(np.interp(iL, X_IDXS, w_l[0])), float(np.interp(iL, X_IDXS, w_l[1])), w_at[0], w_at[1]]),
      'reference': np.array([np.interp(iL, X_IDXS, y_l), np.interp(iL, X_IDXS, y_r)]), 'target': target,
      'episode': in_episode, 'episode_ended': em.ended, 'episode_dist': em.dist if in_episode else 0.0, 'model_alone_time': em.model_alone_time,
      'cues': cue_bits, 'cue_values': np.array([c1 if c1 is not None else 0.0, c2 if c2 is not None else 0.0, ed_g if ed_ok else 0.0, ed_a if ed_ok else 0.0,
                                                 gore_min if gore_min is not None else 0.0, kink, b]),
      'widening': widening if widening is not None else 0.0,
      'width_mem': wm.w if wm.armed else 0.0, 'width_age': wm.age, 'width_state': int(wm.armed) | (int(wm.stale) << 1) | (int(not wm.armed) << 2),
      'road_class': road, 'frame_dt': dt, 'plan_speed': vx0, 'target_lag': lag_L,
      'yield_side': side if no_growth else 0, 'yield_source': src, 'yield_moving': self.moving_t > 0.0,
      't_raw': t_raw, 't_lim': t_lim, 'y_edge_r': filt[3], 'c_ctr': c_ctr, 'y_r': y_r, 'y_l': y_l, 'beta': beta, 'jump_inno': np.array(inno),
    }
    return RightResult(k0=float(k0), fading=fading, yield_active=yield_active, no_growth=no_growth, yield_q=float(q_eff), yield_side=side,
                       episode=in_episode, log=log)

  @staticmethod
  def ref_around(p: RightInputs) -> bool:
    """The reference instance sees an obstacle: its going-around latch, or its going-around / away fade below 1 (C8)."""
    return bool(p.ref_going_around or p.ref_yield_fade < 1.0)


# ---------------------------------------------------------------------------------------------------------------------------------
class BlendfulCorrection:
  """Selector, reference instance, the yield cap and the final limiter: turns the per-frame model output into the profile amplitude k0 of
  the plan shift.  mode MODE_OFF (idle) is today's car: the caller does not use this class then."""
  def __init__(self, reference: Blendful | None = None, right: RightLineBlendful | None = None):
    self.reference = reference if reference is not None else Blendful()
    self.right = right if right is not None else RightLineBlendful()
    self.mode = MODE_OFF
    self.c = 0.0  # delivered instantaneous correction after the limiter (1/m)
    self.k0 = 0.0
    self.gain = 1.0
    self.k_up = 0.0
    self.result = RightResult()
    self.cap_mem: float | None = None  # the toward-obstacle correction at the no-growth point; lives as long as the yield (C8)
    self.cap_active = False
    self.release_window = False
    self.building = False
    self.right_idle = True

  @property
  def releasing(self) -> bool:
    return self.c != 0.0

  def _limit(self, target: float, v: float, release: float) -> float:
    vv = max(v, 1.0)
    build = BUILD / vv ** 2 * DT_MDL
    rel = release / vv ** 2 * DT_MDL
    k = self.c
    return float(np.clip(target, max(k - rel, -build) if k > 0 else k - build, min(k + rel, build) if k < 0 else k + build))

  def update(self, mo: dict, v_ego: float, active: bool, model_curvature: float, lat_action_t: float, mode: int,
             p: RightInputs | None = None) -> float:
    """Returns k0.  mode MODE_OFF only releases a remaining correction (2.0 m/s^3) and then resets the instances."""
    p = p or RightInputs()
    v_eff = max(v_ego, 1.0)
    self.gain = instant_gain(v_eff, lat_action_t, mo['plan'][0][:, Plan.VELOCITY.start])  # F per frame with the plan's own v_x (C11)
    f = self.gain
    if mode == MODE_OFF:
      self.reference.update(mo, v_ego, active, model_curvature, enabled=False)
      self.cap_active = False
      c = self._limit(0.0, v_ego, RELEASE)
      self.c, self.k0, self.k_up = c, c / f, 0.0
      self.result = RightResult()
      if self.c == 0.0:
        self.right.reset()
        self.right_idle = True
      self.mode = mode
      return self.k0
    if self.mode == MODE_OFF:
      self.right.reset()
    if mode == MODE_REGULAR and self.mode == MODE_RIGHT:
      self.right.episodes.reset()
    self.mode = mode
    self.k_up = self.reference.update(mo, v_ego, active, model_curvature, enabled=True)
    k0_up = self.k_up / f
    p.ref_going_around = self.reference.going_around
    p.ref_yield_fade = self.reference.yield_fade
    p.ref_lane_width = float(self.reference.lane_width.x)
    p.building = self.building
    res = RightResult()
    if mode == MODE_RIGHT or (mode == MODE_REGULAR and self.right.a > 0.0):
      if self.right_idle:  # the instance was not running (Regular with a = 0): its filters are stale
        self.right.reset()
      self.right_idle = False
      res = self.right.update(mo, v_ego, active, model_curvature, k0_up, p, force_regular=(mode == MODE_REGULAR))
      k0_target = res.k0
    else:
      self.right_idle = True
      k0_target = k0_up
      res.fading = (not active) or v_ego < MIN_SPEED
    self.result = res
    c_target = f * k0_target
    release = FADING_RELEASE if res.fading else RELEASE
    self.cap_active = False
    self.release_window = False
    if mode == MODE_RIGHT and res.no_growth:
      s, q = res.yield_side, res.yield_q
      toward, prev = s * c_target, max(s * self.c, 0.0)
      if self.cap_mem is None:
        self.cap_mem = prev
      self.cap_mem = min(self.cap_mem, prev)  # non-increasing: no latched remainder can come back
      if toward > prev:  # no growth toward the obstacle from q = 0.1 m
        toward, self.cap_active = prev, True
      x_cap = self.cap_mem * float(np.clip((Q_CAP_FULL - q) / (Q_CAP_FULL - Q_CAP_START), 0.0, 1.0))
      if toward > x_cap:
        toward, self.cap_active = x_cap, True
      if q >= Q_CAP_FULL and s * self.c > x_cap + 1e-12:
        release, self.release_window = SWERVE_RELEASE, True
      if self.cap_active:
        c_target = s * toward if s * c_target > toward else c_target
    else:
      self.cap_mem = None
    c_new = self._limit(c_target, v_ego, release)
    self.building = (c_target - c_new) > 1e-9  # still building toward the right
    self.c = c_new
    self.k0 = self.c / f
    return self.k0


def mode_from_params(blendful_on: bool, kind: int) -> int:
  """Blendful (BOOL, upstream's key) and BlendfulMode (INT, 0 Regular / 1 Right line) -> MODE_*. Other toggles (second UI, sunnylink) only
  set Blendful, so they restore the last selected kind; an old backup without BlendfulMode reads the default 0 = Regular."""
  if not blendful_on:
    return MODE_OFF
  return MODE_RIGHT if int(kind) == 1 else MODE_REGULAR


def selection_to_params(index: int) -> tuple[bool, int | None]:
  """Settings selector index (0 Off, 1 Regular, 2 Right line) -> (Blendful, BlendfulMode or None = keep the stored kind)."""
  return (index != 0), (None if index == 0 else index - 1)


def apply_correction(mo: dict, k0: float, v_ego: float) -> dict:
  """Copy of the model output with the plan shifted by the profile of amplitude k0 (before planplus; x and velocities untouched)."""
  out = dict(mo)
  out['plan'] = shift_plan(mo['plan'][0], k0, max(v_ego, 1.0))[None]
  return out


def fill_log(builder, ctl: BlendfulCorrection, mode: int, v_ego: float) -> None:
  """Additive modelDataV2SP fields (DESIGN D logging).  Fields of the first implementation that no longer have a meaning (rho, bias, split
  score, latch / ramp / hold, split fade) are kept in the schema (additive only) and written as constants."""
  builder.blendfulMode = int(mode)
  builder.blendfulGainF = float(ctl.gain)
  builder.blendfulKappaUp = float(ctl.k_up)
  builder.blendfulKappa0 = float(ctl.k0)
  builder.blendfulCapActive = bool(ctl.cap_active)
  lg = ctl.result.log
  if mode == MODE_OFF or not lg:
    return
  builder.blendfulRightApplied = bool(lg['applied'])
  for name, key in (('RightGap', 'gap'), ('RightGapNow', 'gap_now'), ('RightQuality', 'q_R'), ('LeftRefQuality', 'q_L'), ('Authority', 'a'),
                    ('AuthorityTarget', 'a_star'), ('FadeBend', 'f_bend'), ('FadeYieldLeft', 'f_yl'),
                    ('FadeYieldRight', 'f_yr'), ('FadeJump', 'f_jump'), ('FadePlausibility', 'f_plaus'), ('FadeBlinker', 'f_blink'),
                    ('AwayCar', 'away_car'), ('Strength', 's_r'), ('StrengthWeight', 'w_c'),
                    ('CurveAccel', 'curve_accel'), ('KappaRL', 'k_rl'), ('ProfileLength', 'l_c'), ('ProfileShift', 'd'),
                    ('FilterTime', 'filter_ms'), ('JunctionEpisodeDistance', 'episode_dist'), ('JunctionWidening', 'widening'),
                    ('WidthMemory', 'width_mem'), ('WidthMemoryAge', 'width_age'), ('ModelAloneTime', 'model_alone_time'),
                    ('FrameDt', 'frame_dt'), ('PlanSpeed', 'plan_speed'), ('TargetLag', 'target_lag')):
    setattr(builder, 'blendful' + name, float(lg.get(key, 0.0)))
  builder.blendfulCapMemory = float(ctl.cap_mem or 0.0)
  builder.blendfulFadeSplit = 1.0  # superseded (junction rev 5): no split fade
  builder.blendfulRightOnly = bool(lg['right_only'])
  builder.blendfulYieldActive = bool(lg['yield_active'])
  builder.blendfulBoundActive = bool(lg['bound_active'])
  builder.blendfulJunctionEpisode = bool(lg['episode'])
  builder.blendfulDClipped = bool(lg['d_clipped'])
  builder.blendfulYieldMoving = bool(lg['yield_moving'])
  builder.blendfulJunctionCues = int(lg['cues'])
  builder.blendfulJunctionEnded = int(lg['episode_ended'])
  builder.blendfulWidthMemoryState = int(lg['width_state'])
  builder.blendfulRoadClass = int(lg['road_class'])
  builder.blendfulYieldSide = int(lg['yield_side'])
  builder.blendfulYieldSource = int(lg['yield_source'])
  builder.blendfulJunctionCueValues = [float(x) for x in lg['cue_values']]
  builder.blendfulInputFiltered = [float(x) for x in lg['filtered']]
  builder.blendfulInputRaw = [float(x) for x in lg['raw']]
  builder.blendfulFilterOffset = [float(x) for x in lg['filter_offset']]
  builder.blendfulInputConfidence = [float(x) for x in lg['confidence']]
  builder.blendfulInputState = [int(x) for x in lg['state']]
  builder.blendfulCandidateConfidence = [float(x) for x in lg['cand_conf']]
  builder.blendfulCandidateWeight = [float(x) for x in lg['cand_weight']]
  builder.blendfulReference = [float(x) for x in lg['reference']]
  if mode == MODE_RIGHT:
    builder.blendfulTarget = [float(x) for x in lg['target']]
