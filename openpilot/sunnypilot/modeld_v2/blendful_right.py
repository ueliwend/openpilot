"""
Right-line Blendful ("keep right", task A8): follows the right lane line (or road edge) at a speed dependent gap, never left of the lane
centre, and hands back to regular upstream Blendful where the right reference is not usable.  Design: inster-dev
results/2026-11-12_a8_right_line_blendful/DESIGN.md rev 5.1 / 6.2 (section letters B.. below refer to it); implementation round 2
(results/2026-11-12_a8_impl_r2) answers the E1 replay causes C1-C11 / K1 (results/2026-11-12_a8_replay), round 3 Part 1
(results/2026-11-12_a8_impl_r3) the review of round 2: trigger aggregator with per-trigger obstacle sides (B1), the yield cap whenever this
instance runs (B2), yield state reset while fading (S2), non-finite guard (S3), centre-limit hysteresis (S4), whole update time (S5).

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
from openpilot.sunnypilot.modeld_v2.radar_clearance import RadarClearance, RadarFrame, clearance_q

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
# ground-fixed target T_lim (B5 step 3, C1): glide rate in both directions, lag-bound rate, far points re-seeded from T_raw
TARGET_RATE = 0.25  # m/s, both directions (rev 6.3: the speed of moves to the right is set by the final move-speed limit)
LAG_BOUND, LAG_RATE = 1.0, 0.5  # m, m/s: where the target lags its raw value by more, that point moves at LAG_RATE (still glide-limited)
RESEED_LOOKAHEADS = 2.0  # points beyond 2 L take T_raw every frame (rev 6.4)
# final move-speed limit (B5 step 4b, rev 6.5 / 6.7; owner 21:44 question 9): the car moves at <= 0.05 m/s; the means is the reference
# rate r_ref, the rate at which the approach to the right reference may grow (start value 0.042 m/s, rev 6.7; final value from the kit)
REFERENCE_RATE = 0.042  # m/s, r_ref
CAR_MOVE_SPEED = 0.05  # m/s, the decided move speed of the car (release condition after a reset)
RELEASE_NEAR = 0.05  # m: after a reset the approach is held until the car is this close to p0 (and its own approach is <= CAR_MOVE_SPEED)
# measurement-filter rotation lever (B2, rev 6.3): "design" = lever x (scale 1, default); "model_axis" = lever x vEgo / v_x(0)
ROTATION_SCALES = ("design", "model_axis")
FILTER_ROTATION_SCALE = "design"
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
# yield trigger aggregation (B6 / C8; review B1, N10)
SIDE_KNOWN = 0.1  # m: a trigger knows the obstacle side once its own departure exceeds this ("the side of the first departure > 0.1 m")
SIDE_RIGHT, SIDE_LEFT = 1, -1
BOTH_SIDES = (SIDE_RIGHT, SIDE_LEFT)
LOG_SIDE_BOTH = 2  # log value of the yield side while both directions are capped
# left-limit formula hysteresis (review S4): two references from q > 0.5 on both sides, back to one when either falls below 0.4
CENTRE_BOTH_ON, CENTRE_BOTH_OFF, CENTRE_SIDE_SWITCH, CENTRE_DWELL = 0.5, 0.4, 0.1, 0.5  # quality, quality, quality, s (rev 6.3)
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
    self.step = np.zeros(NX)  # on a re-initialisation (or coast expiry) frame: the jump of the estimate against its own prediction

  def predict(self, vx0: float, yaw_rate: float, dt: float = DT_MDL, scale: float = 1.0) -> np.ndarray:
    return motion_predict(self.y, vx0, yaw_rate, dt, scale)

  def update(self, y_meas: np.ndarray, q_meas: float, vx0: float, yaw_rate: float, dt: float = DT_MDL,
             scale: float = 1.0) -> tuple[np.ndarray, float]:
    q_meas = float(np.clip(q_meas, 0.0, 1.0))
    self.reinit = False
    self.step = np.zeros(NX)
    alpha = dt / (dt + self.tau)
    was_initialized = self.initialized
    pred_prev = self.predict(vx0, yaw_rate, dt, scale)
    if self.initialized:
      pred = pred_prev
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
    if was_initialized != self.initialized:  # re-initialised from the measurement, or the coast expired: a step that is not road
      self.step = self.y - pred_prev
    return self.y, self.q


class GroundTarget:
  """Ground-fixed, rate-limited target T_lim (B5 step 3, C1, R5-3, rev 6.3 / 6.4).  Last frame's T_lim is predicted into the new car frame
  with the filters' motion model and then moves toward the raw target by at most TARGET_RATE x dt per point in either direction (the speed
  of moves to the right is set by the final move-speed limit, step 4b).  Where it lags the raw target by more than LAG_BOUND the rate
  rises to LAG_RATE (still glide-limited).  Points beyond the last frame's horizon, and every point beyond reseed_x (2 L), take the raw
  target every frame, so far-field prediction errors are not carried into the fit window.  Inside reseed_x its own state cannot jump."""
  def __init__(self):
    self.t: np.ndarray | None = None
    self.lag = np.zeros(NX)

  def reset(self):
    self.t = None
    self.lag = np.zeros(NX)

  def update(self, t_raw: np.ndarray, vx0: float, yaw_rate: float, dt: float, scale: float = 1.0, reseed_x: float | None = None) -> np.ndarray:
    if self.t is None:
      self.t = np.array(t_raw, dtype=float)
      self.lag = np.zeros(NX)
      return self.t.copy()
    pred = motion_predict(self.t, vx0, yaw_rate, dt, scale)
    pred = np.where(X_IDXS + vx0 * dt > X_IDXS[-1], t_raw, pred)
    diff = t_raw - pred  # > 0: the raw target is further right
    rate = np.where(np.abs(diff) > LAG_BOUND, LAG_RATE, TARGET_RATE) * dt
    self.t = pred + np.clip(diff, -rate, rate)
    if reseed_x is not None:
      self.t = np.where(X_IDXS > reseed_x, t_raw, self.t)
    self.lag = t_raw - self.t
    return self.t.copy()


class MoveSpeedLimit:
  """Final move-speed limit (B5 step 4b, rev 6.5; question 9; reviews K62-1, C63-1; impl review B3).  Same frame, lane coordinates, no
  lateral prediction.  Per window point the gap of the step-4 target to the right reference is G4 = Y_R - T4; G0 is the gap at the last
  reset, carried along the lane by the model's own travel and rebased by the switch part of the change of Y_R (amendment 1).  The weighted
  approach A = w (G0 - G), w = a s_R f_yieldR, grows by at most REFERENCE_RATE dt per frame, any decrease passes at once:
  A_lim = min(w (G0 - G4), A_prev + REFERENCE_RATE dt), G = max(G4, G0 - A_lim / w) (G = max(G4, G0) at w = 0), and the target is
  max(T_C', Y_R - G) (the centre limit as a floor, amendment 2); the caller applies the right safety bounds after it.
  Reset (A = 0): p0 = max(car position, T_C'(0)), G0 := Y_R(0) - p0 along the lane, A held at 0 until the car's gap is within 0.05 m of
  p0's and its own approach to the right reference (from Y_R(0), switch part removed) is <= CAR_MOVE_SPEED."""
  def __init__(self):
    self.g0: np.ndarray | None = None
    self.a_lim = np.zeros(NX)
    self.released = False
    self.limiting = False
    self.yr0_prev: float | None = None
    self.approach_speed = 0.0

  def update(self, y_r: np.ndarray, t4: np.ndarray, c_ctr: np.ndarray, w: float, switch_part: np.ndarray, vx0: float, dt: float,
             reset: bool) -> np.ndarray:
    yr0 = float(y_r[0])
    if self.yr0_prev is not None:
      self.approach_speed = (self.yr0_prev + float(switch_part[0]) - yr0) / dt
    self.yr0_prev = yr0
    if self.g0 is not None:
      xs = X_IDXS + vx0 * dt
      self.g0 = np.interp(xs, X_IDXS, self.g0) + switch_part
      self.a_lim = np.interp(xs, X_IDXS, self.a_lim)
    if reset or self.g0 is None:
      p0 = max(0.0, float(c_ctr[0]))
      self.g0 = np.full(NX, yr0 - p0)
      self.a_lim = np.zeros(NX)
      self.released = False
    if not self.released and yr0 <= float(self.g0[0]) + RELEASE_NEAR and self.approach_speed <= CAR_MOVE_SPEED:
      self.released = True
    g4 = y_r - t4
    if w > 1e-9:
      want = w * (self.g0 - g4)
      self.a_lim = np.minimum(want, self.a_lim + (REFERENCE_RATE * dt if self.released else 0.0))
      self.limiting = bool(np.any(self.a_lim < want - 1e-9))
      g = np.maximum(g4, self.g0 - self.a_lim / w)
    else:
      self.limiting = False
      g = np.maximum(g4, self.g0)
    return np.maximum(c_ctr, y_r - g)


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
  The plan-alone source is kept because B6 / C8 calls for it ("the left reference or the plan alone when the right reference is weak").  It
  is not own-motion compensated: a move of the car to the right raises x by rho times the move, so it can only look like a left swerve
  (review N1).  With that source q_R and q_L are < 0.3, a* is 0 and a false trigger can only fade or cap, never push toward an obstacle.
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
@dataclass
class TriggerReport:
  """One yield trigger's view of the current frame (B6, C8).  Every trigger supplies its own obstacle side: the endpoint gap from its
  departure, the reference instance from its own offset - usual, and (rev 6, B6a) the radar clearance check always the right."""
  name: str
  no_growth: bool  # the trigger asks for the no-growth rule and the cap logic this frame
  active: bool  # the trigger's yield is active (its own latch)
  q: float  # in the units of the B6 cap ramp: >= 0.1 no growth, 0.15 .. 0.2 early cap ramp, >= 0.2 full cap
  side: int = 0  # SIDE_RIGHT (+1, obstacle on the right), SIDE_LEFT (-1), 0 = not known this frame


class YieldAggregator:
  """Combines the yield triggers into one no-growth phase and the capped sides (review B1, N10).
  - The phase runs while any trigger asks for no growth; it ends (and everything below resets) when none does.
  - Each trigger latches its own side at the first frame of the phase in which it asks for no growth and knows its side; the latch holds
    until the phase ends.  The phase's sides are the union of the latched sides and are not removed before the phase ends.
  - While a trigger asks for no growth and has not latched a side yet, BOTH sides are capped.
  - Two triggers that latched different sides cap both directions (no correction toward either side; as B6a item 4 states for a
    clearance yield together with a left yield).
  - The cap quantity q of a side is the largest q of the triggers that currently ask for no growth with that side (or with no side yet);
    a latched side whose triggers no longer ask keeps q = 0, i.e. no growth only (its cap memory, which can only fall, still holds)."""
  def __init__(self):
    self.reset()

  def reset(self):
    self.phase = False
    self.latched: dict[str, int] = {}
    self.sides: set[int] = set()

  def update(self, reports: list[TriggerReport]) -> tuple[bool, bool, dict[int, float]]:
    """Returns (no_growth, yield_active, caps); caps maps each capped side to its q."""
    asking = [r for r in reports if r.no_growth]
    if not asking:
      self.reset()
      return False, False, {}
    self.phase = True
    for r in asking:
      if r.name not in self.latched and r.side in BOTH_SIDES:
        self.latched[r.name] = r.side
        self.sides.add(r.side)
    caps = dict.fromkeys(self.sides, 0.0)
    for r in asking:
      side = self.latched.get(r.name)
      for s in ((side,) if side is not None else BOTH_SIDES):
        caps[s] = max(caps.get(s, 0.0), float(r.q))
    return True, any(r.active for r in reports), caps


def log_side(caps: dict[int, float]) -> int:
  """Capped sides as one log value: 0 none, +1 right, -1 left, LOG_SIDE_BOTH both."""
  if len(caps) == 2:
    return LOG_SIDE_BOTH
  return next(iter(caps), 0)


class CentreLimitSource:
  """Which form the left limit T_C' uses (B3, rev 6.3; review S4).  Two references: their midpoint (the unclipped width); one reference:
  that reference minus / plus upstream's filtered half width.  The two-reference form is entered when the weaker side's quality has been
  above 0.5 for longer than 0.5 s and left when it has been below 0.4 for longer than 0.5 s; between the one-reference forms the side
  changes only when the other side's quality is higher by more than 0.1 (implementation choice; the design does not say).  Quality
  flicker around a threshold therefore cannot switch the form from frame to frame."""
  BOTH, RIGHT, LEFT = 0, 1, 2

  def __init__(self):
    self.src: int | None = None
    self.t = 0.0  # how long the switching condition has held, from its first frame
    self.pending = False

  def _single(self, q_r: float, q_l: float) -> int:
    if self.src == self.RIGHT:
      return self.LEFT if q_l > q_r + CENTRE_SIDE_SWITCH else self.RIGHT
    if self.src == self.LEFT:
      return self.RIGHT if q_r > q_l + CENTRE_SIDE_SWITCH else self.LEFT
    return self.RIGHT if q_r >= q_l else self.LEFT

  def update(self, q_r: float, q_l: float, dt: float = DT_MDL) -> int:
    weak = min(q_r, q_l)
    if self.src is None:
      self.src = self.BOTH if weak > CENTRE_BOTH_ON else self._single(q_r, q_l)
      return self.src
    cond = weak < CENTRE_BOTH_OFF if self.src == self.BOTH else weak > CENTRE_BOTH_ON
    self.t = (self.t + dt if self.pending else 0.0) if cond else 0.0
    self.pending = cond
    if cond and self.t > CENTRE_DWELL + 1e-9:  # held for longer than 0.5 s (a quality 0.5 +- 0.2 at 1 Hz is above 0.5 for exactly 0.5 s)
      self.src = self._single(q_r, q_l) if self.src == self.BOTH else self.BOTH
      self.t, self.pending = 0.0, False
    elif self.src != self.BOTH:
      self.src = self._single(q_r, q_l)
    return self.src


def finite_model_inputs(mo: dict, *scalars: float, right_line: bool = True) -> bool:
  """The model outputs this frame's instances read, and the given scalars, are finite (review S3).  Always: what the reference instance
  (upstream's Blendful) reads, the near lane lines with their stds and probabilities and the plan's x and y, and the plan speed of the gain F.
  With the right-line instance running also all four lane lines, their probabilities, the plan's yaw rate and the road edges when present.
  So Regular without the right-line instance reacts only to values that today's add-on or the plan-shift delivery reads."""
  plan = mo['plan'][0]
  arrays = [mo['lane_lines'][0, 1:3, :, 0], mo['lane_lines_stds'][0, 1:3, :, 0], mo['lane_lines_prob'][0, 3:6:2], plan[:, :2],
            plan[:, Plan.VELOCITY.start]]
  if right_line:
    arrays += [mo['lane_lines'][0, :, :, 0], mo['lane_lines_stds'][0, :, :, 0], mo['lane_lines_prob'][0], plan[0, Plan.ORIENTATION_RATE.start + 2]]
    if 'road_edges' in mo and 'road_edges_stds' in mo:
      arrays += [mo['road_edges'][0, :, :, 0], mo['road_edges_stds'][0, :, :, 0]]
  return bool(np.isfinite(np.concatenate([np.ravel(a) for a in arrays] + [np.asarray(scalars, dtype=float)])).all())


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
  ref_departure: float = 0.0  # the reference instance's offset - usual, m (> 0: its plan end moved right, i.e. an obstacle on the left)
  ref_lane_width: float = LANE_WIDTH_DEFAULT  # the reference instance's filtered lane width (left limit with one reference)
  building: bool = False  # last frame's final limiter was still building the correction toward the right
  radar: RadarFrame | None = None  # the newest radarTracks message (B6a); None = no radar data: f_radar = 0, Right line = Regular


@dataclass
class RightResult:
  k0: float = 0.0  # profile amplitude before the yield cap and the final limiter
  fading: bool = False
  yield_active: bool = False
  no_growth: bool = False  # a trigger asks for no growth (endpoint gap q >= 0.1, or any yield active): the cap logic of BlendfulCorrection acts
  caps: dict = field(default_factory=dict)  # capped side (SIDE_RIGHT / SIDE_LEFT) -> its cap quantity q (YieldAggregator)
  yield_q: float = 0.0  # largest q of the capped sides (log)
  yield_side: int = 0  # log_side(caps): 0 none, +1 right, -1 left, 2 both
  episode: bool = False
  log: dict = field(default_factory=dict)


class RightLineBlendful:
  def __init__(self, strength_rising: bool | None = None, rotation_scale: str | None = None):
    self.strength_rising = RISING_STRENGTH if strength_rising is None else strength_rising
    self.rotation_scale = FILTER_ROTATION_SCALE if rotation_scale is None else rotation_scale
    assert self.rotation_scale in ROTATION_SCALES, self.rotation_scale
    self.filters = [InputFilter() for _ in range(4)]  # left line, left edge, right line, right edge
    self.gates = [ConfGate() for _ in range(4)]
    self.reset()

  def reset(self):
    self.quality_gate = ConfGate()
    self.authority = Authority()
    self.episodes = JunctionEpisodes()
    self.road = RoadClass()
    self.bend_active = Timed(off_time=1.0)
    self.blink_on = Timed(off_time=1.0, on_time=0.3)
    self.f_blink = 1.0
    self.width_mem = WidthMemory()
    self.radar = RadarClearance()
    self.restart_lane_tracking()
    self.inactive_prev = False
    self.force_regular_prev = True
    self.mode_on = True  # the first frame after a (re)start is a re-entry of the move-speed limit

  def restart_lane_tracking(self):
    """Everything that describes the lane the car is in: the input filters and their confidence gates, the probability filters, the
    ground-fixed target, the jump counters, the candidate weights, the left-limit formula, the width memory (re-measured) and the yield
    state.  Restarted on the first active frame after a lane change (review S2): the lines re-index during the change, and filters or
    references carried across it would compare the old lane with the new one (a re-indexed right line converging over about a second looks
    like a 0.37 m left swerve, a different lane width like a widening)."""
    for f in self.filters:
      f.__init__()
    for g in self.gates:
      g.c = 0.0
    self.line_probs = FirstOrderFilter(np.zeros(2), 0.5, DT_MDL)
    self.edge_q = FirstOrderFilter(np.zeros(2), 0.5, DT_MDL)
    self.ground = GroundTarget()
    self.move = MoveSpeedLimit()
    self.ref_present_prev = False
    self.jumps = JumpDetector()
    self.centre_src = CentreLimitSource()
    self.w_prev = None
    self.w_r_prev = None
    self.width_mem.remeasure()
    self.reset_yields()

  def reset_yields(self):
    """Yield state: endpoint-gap trigger, both latches, the aggregator and the move flag (also while the instance fades, review S2)."""
    self.gap = EndpointGap()
    self.gap_rebase = 0.0  # cumulative switch part of the right reference at L, left out of the endpoint gap
    self.yield_latch = Timed(off_time=1.0)
    self.ref_yield_latch = Timed(off_time=1.0)
    self.yields = YieldAggregator()
    self.moving_t = 0.0
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
    if not active:  # lane change (or another desire): the width memory describes no lane until the car is in the new one (S2)
      self.width_mem.remeasure()
    elif self.inactive_prev:  # the first frame after it: restart from the new lane (S2)
      self.restart_lane_tracking()
    self.inactive_prev = not active
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
    # rotation lever (B2, FILTER_ROTATION_SCALE): "model_axis" scales it by vEgo / v_x(0), "design" (default) uses x as it is
    scale = float(np.clip(v / vx0, 0.8, 1.25)) if (self.rotation_scale == "model_axis" and v >= 1.0 and vx0 >= 1.0) else 1.0
    if self.force_regular_prev and not force_regular:  # mode switched on (Regular -> Right line): a re-entry of the move-speed limit
      self.mode_on = True
    self.force_regular_prev = force_regular
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
    # --- left limit T_C': the centre between the references (unclipped width with two references, upstream's filtered width with one);
    # the formula changes with confidence hysteresis (S4)
    centre_src = self.centre_src.update(q_R, q_L, dt)
    if centre_src == CentreLimitSource.BOTH:
      c_ctr = (y_l + y_r) / 2
    elif centre_src == CentreLimitSource.RIGHT:
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
    if was_episode and not in_episode:  # the episode ended: re-measure the width (T_lim keeps its state; the move limit resets, B5)
      self.width_mem.end_of_episode()
    jn_active = in_episode and not force_regular
    # --- blinker fade (left blinker hands back to upstream; B9)
    lb = self.blink_on.update(p.left_blinker)
    self.f_blink = max(0.0, self.f_blink - dt) if lb else min(1.0, self.f_blink + dt)
    # --- road class (log / reports only; resets no estimator)
    road = self.road.update(v, q_L, q_R, width_now, dt)
    # --- step 2 / 3: references -> raw target -> ground-fixed rate limit (B1, B5, B8)
    g = gap_for_speed(v, p.gap_low, p.gap_high)
    win_ref = (X_IDXS >= 0.5 * look) & (X_IDXS <= 1.5 * look)
    k_ref = 2.0 * float(np.polyfit(X_IDXS[win_ref], y_r[win_ref], 2)[0])
    off_track = L_D ** 2 * max(k_ref, 0.0) / 2
    dc = p.c_set - p.c_applied
    t_r_line = y_r - W_R - g - off_track - dc
    t_raw = np.maximum(t_r_line, c_ctr)
    t_lim = self.ground.update(t_raw, vx0, yaw_rate, dt, scale, reseed_x=RESEED_LOOKAHEADS * look)
    lag_L = float(np.interp(iL, X_IDXS, self.ground.lag))
    # --- radar right-side clearance check (B6a): reads the radar, the references and the centre limit, never the plan
    rc = self.radar.update(p.radar, v, vx0, yaw_rate, g, dc, W_R, X_IDXS, c_ctr, filt[3] - EDGE_EXTRA, w_r[1], dt, l_d=L_D,
                           rise=REFERENCE_RATE)
    f_radar = 1.0 if rc.healthy else 0.0
    # --- yields (B6, C7, C8): the endpoint-gap trigger on the right reference, else the left one, else the plan alone, the reference
    # instance, and the radar clearance trigger, each with its own side
    if q_R >= 0.3:
      src, ref_L = 1, float(np.interp(look, X_IDXS, y_r))
    elif q_L >= 0.3:
      src, ref_L = 2, float(np.interp(look, X_IDXS, y_l))
    else:
      src, ref_L = 0, 0.0
    # the switch part of the change of Y_R (B5 amendment 1: candidate weights, filter re-initialisation steps): not road
    sw = (w_r - self.w_r_prev) * cands_r if self.w_r_prev is not None else np.zeros_like(cands_r)
    switch_part = sw.sum(axis=0) + w_r[0] * self.filters[2].step + w_r[1] * self.filters[3].step
    self.w_r_prev = w_r.copy()
    if src == 1:  # the endpoint gap on the right reference leaves the switch part out as well, so a line -> edge switch is no departure
      self.gap_rebase += float(np.interp(look, X_IDXS, switch_part))
    x_gap = ref_L - float(np.interp(look, plan_x, plan_y)) - (self.gap_rebase if src == 1 else 0.0)
    allow = 0.25 * abs(model_curvature) * look ** 2 / 2
    if fading:  # lane change or v < 5 m/s: the gap state and the latches would compare the old lane with the new one (S2)
      self.reset_yields()
      q_y, no_growth, yield_active, caps = 0.0, False, False, {}
    else:
      move_now = (not self.no_growth_prev) and (lag_L > MOVE_LAG_EPS or self.authority.a < self.authority.a_cmd - 1e-6 or p.building
                                                or self.move.limiting)
      self.moving_t = MOVE_SETTLE if move_now else max(0.0, self.moving_t - dt)
      skip = (not f_jump_t) or cand_changed
      q_y = self.gap.update(x_gap, src, self.moving_t > 0.0, bool(skip), allow, self.ref_around(p), dt)
      dep = self.gap.departure
      car_trig = self.yield_latch.update(q_y > Q_CAP_FULL if not self.yield_latch.state else q_y >= Q_RELEASE)
      ref_trig = self.ref_yield_latch.update(self.ref_around(p))
      # each trigger supplies its own side (B1): the endpoint gap from its departure (> 0: the plan moved left, obstacle on the right), the
      # reference instance from its own offset - usual (> 0: its plan end moved right, obstacle on the left), the radar check the right
      side_gap = (SIDE_RIGHT if dep > 0 else SIDE_LEFT) if abs(dep) > SIDE_KNOWN else 0
      side_ref = (SIDE_LEFT if p.ref_departure > 0 else SIDE_RIGHT) if abs(p.ref_departure) > SIDE_KNOWN else 0
      q_rc = clearance_q(rc.deficit, Q_NO_GROWTH, Q_CAP_START, Q_CAP_FULL)
      no_growth, yield_active, caps = self.yields.update([
        TriggerReport('endpoint_gap', bool(q_y >= Q_NO_GROWTH or car_trig), bool(car_trig), q_y, side_gap),
        TriggerReport('reference', bool(ref_trig), bool(ref_trig), Q_CAP_FULL, side_ref),
        TriggerReport('radar_clearance', bool(rc.yield_active), bool(rc.yield_active), q_rc, SIDE_RIGHT),
      ])
    yield_ended = self.no_growth_prev and not no_growth
    self.no_growth_prev = no_growth
    q_eff = max(caps.values()) if caps else 0.0
    f_yl = 0.0 if (yield_active and SIDE_RIGHT in caps) else 1.0
    f_yr = float(np.interp(caps[SIDE_LEFT], (0.2, 0.45), (1.0, 0.0))) if (yield_active and SIDE_LEFT in caps) else 1.0
    # --- authority (B5)
    qa = self.quality_gate.update(q_R)
    a_star = float(qa)
    f_ep = 0.0 if jn_active else 1.0
    a_star *= f_bend_t * f_jump_t * f_plaus_t * self.f_blink * f_yl * f_ep * f_radar
    if force_regular or fading:
      a_star = 0.0
    if q_y >= Q_NO_GROWTH:
      a_star = min(a_star, self.authority.a_cmd)
    a = self.authority.update(a_star)
    # --- step 4: limits from the current references, not rate-limited: the left limit, the object bound (B6a), the right safety bounds
    t_left = np.maximum(t_lim, c_ctr)
    t_obj = c_ctr if rc.right_blindspot else np.full(NX, rc.t_obj)  # rightBlindspot: no right-line shift
    t_objb = np.minimum(t_left, np.maximum(t_obj, c_ctr))
    rc.bound_active = bool(np.any(t_objb < t_left - 1e-9))
    t_bound_line = y_r - W_R - LINE_BOUND - off_track - dc
    w_e = float(ss((qf[3] - 0.4) / 0.2))
    t_bound_edge = filt[3] - W_R - EDGE_BOUND - off_track - dc
    t4 = self.safety_bounds(t_objb, t_bound_line, t_bound_edge, w_e)
    bound_active = bool(np.any(t4 < t_objb - 1e-9))
    # --- step 4b: final move-speed limit on the approach to the right reference (rev 6.5), the centre limit as a floor, then the bounds
    w_c = float(np.interp(abs(model_curvature) * v ** 2, [0.18, 0.53], [STRENGTH_RIGHT_ONLY, 1.0])) if self.strength_rising else STRENGTH_RIGHT_ONLY
    s_r = 1.0 - (1.0 - l_q) * (1.0 - w_c * q_R)
    present = qa > 0.0
    reasons = (a <= 0.0, was_episode and not in_episode, yield_ended, present and not self.ref_present_prev, self.mode_on)
    self.ref_present_prev, self.mode_on = present, False
    t_move = self.move.update(y_r, t4, c_ctr, a * s_r * f_yr, switch_part, vx0, dt, any(reasons))
    target = self.safety_bounds(t_move, t_bound_line, t_bound_edge, w_e)
    # --- correction (B4)
    d_rl, k0_rl, k_fit, d_clipped = self._correction(target, plan_x, plan_y, v, s_r)
    k0_rl *= f_yr
    wts = junction_policy(in_episode, a, right_line_mode=not force_regular)
    k0 = wts.w_rl * k0_rl + wts.w_up * k0_up
    if fading:
      k0 = 0.0
    right_only = bool(q_R > 0.5 and l_q < 0.3)
    gap_now = float(np.interp(0.0, X_IDXS, y_r) - W_R)
    em = self.episodes
    cue_bits = sum(int(f) << i for i, f in enumerate(cue_flags))
    wm = self.width_mem
    mv = self.move
    log = {
      'applied': a > 0.0, 'gap': g, 'gap_now': gap_now, 'q_R': q_R, 'q_L': q_L, 'a': a, 'a_star': a_star,
      'f_bend': f_bend_t, 'f_yl': f_yl, 'f_yr': f_yr, 'f_jump': f_jump_t, 'f_plaus': f_plaus_t, 'f_blink': self.f_blink, 'f_radar': f_radar,
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
      'yield_side': log_side(caps), 'yield_source': src, 'centre_source': centre_src, 'yield_moving': self.moving_t > 0.0,
      't_raw': t_raw, 't_lim': t_lim, 't4': t4, 'y_edge_r': filt[3], 'c_ctr': c_ctr, 'y_r': y_r, 'y_l': y_l, 'beta': beta, 'jump_inno': np.array(inno),
      'approach': float(np.interp(iL, X_IDXS, mv.a_lim)), 'gap_reset': float(np.interp(iL, X_IDXS, mv.g0)), 'approach_held': not mv.released,
      'move_reset': sum(int(r_) << i for i, r_ in enumerate(reasons)), 'switch_part': float(np.interp(iL, X_IDXS, switch_part)),
      'move_limiting': mv.limiting, 'rotation_scale': ROTATION_SCALES.index(self.rotation_scale), 'radar': rc,
    }
    return RightResult(k0=float(k0), fading=fading, yield_active=yield_active, no_growth=no_growth, caps=caps, yield_q=float(q_eff),
                       yield_side=log_side(caps), episode=in_episode, log=log)

  @staticmethod
  def safety_bounds(t: np.ndarray, t_bound_line: np.ndarray, t_bound_edge: np.ndarray, w_e: float) -> np.ndarray:
    """The right safety bounds (B8): the line bound (0.2 m), then the edge bound (0.3 m) by the edge's clarity; they override the left limit."""
    tb = np.minimum(t, t_bound_line)
    return tb - w_e * np.maximum(0.0, tb - t_bound_edge)

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
    self.c = 0.0  # delivered instantaneous correction after the limiter (1/m); always finite (S3)
    self.k0 = 0.0
    self.gain = 1.0  # F of the last frame with a finite plan speed
    self.v_limit = 1.0  # the last finite speed, for the limiter on a frame without one (S3)
    self.k_up = 0.0
    self.result = RightResult()
    self.cap_mem: dict[int, float] = {}  # per capped side: the toward-obstacle correction at its no-growth point; lives with the phase (C8)
    self.cap_active = False
    self.release_window = False
    self.building = False
    self.right_idle = True
    self.invalid = False  # this frame's inputs or computed correction were non-finite: released toward 0, instances reset (S3)
    self.update_ms = 0.0  # time of the whole last update (S5); fill_log consumes it

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
    """Returns k0.  mode MODE_OFF only releases a remaining correction (2.0 m/s^3) and then resets the instances.  The whole update is
    timed (S5)."""
    t0 = time.perf_counter()
    k0 = self._update(mo, v_ego, active, model_curvature, lat_action_t, mode, p or RightInputs())
    self.update_ms = (time.perf_counter() - t0) * 1e3
    return k0

  def _release_invalid(self, v: float) -> float:
    """S3: a non-finite input or result.  Both instances are reset (the reference instance is today's upstream Blendful, which a
    non-finite lane line would also leave non-finite), the cap memory is dropped, and the delivered correction is released toward 0
    through the limiter (2.0 m/s^3).  Nothing non-finite is kept, so the next finite frame (and Off) start clean."""
    self.invalid = True
    self.reference.__init__()
    self.right.reset()
    self.right_idle = True
    self.result = RightResult(fading=True)
    self.cap_mem = {}
    self.cap_active = self.release_window = self.building = False
    self.k_up = 0.0
    self.c = self._limit(0.0, v, RELEASE)
    self.k0 = self.c / self.gain
    return self.k0

  def _gain(self, mo: dict, lat_action_t: float, valid: bool) -> bool:
    """F per frame with the plan's own v_x (C11), kept from the last frame where it is finite; returns whether this frame's F is."""
    gain = instant_gain(self.v_limit, lat_action_t, mo['plan'][0][:, Plan.VELOCITY.start]) if valid else float('nan')
    if np.isfinite(gain) and abs(gain) > 1e-9:
      self.gain = gain
      return True
    return False

  def _update(self, mo: dict, v_ego: float, active: bool, model_curvature: float, lat_action_t: float, mode: int, p: RightInputs) -> float:
    self.invalid = False
    if not np.isfinite(self.c):
      self.c = 0.0
    if np.isfinite(v_ego):
      self.v_limit = max(float(v_ego), 1.0)
    v_lim = self.v_limit
    if mode == MODE_OFF:
      self._gain(mo, lat_action_t, finite_model_inputs(mo, lat_action_t, right_line=False))
      f = self.gain
      self.reference.update(mo, v_ego, active, model_curvature, enabled=False)
      self.cap_active = self.release_window = False
      self.cap_mem = {}
      c = self._limit(0.0, v_lim, RELEASE)
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
    # the right-line instance runs in Right line, and in Regular while its share a is still handed back or its yield phase still runs, so
    # the yield cap acts in the hand-back window for as long as the yield lasts (B2)
    right_runs = mode == MODE_RIGHT or (mode == MODE_REGULAR and (self.right.a > 0.0 or self.right.yields.phase))
    if right_runs:
      valid = finite_model_inputs(mo, v_ego, model_curvature, lat_action_t, p.gap_low, p.gap_high, p.c_set, p.c_applied, p.frame_dt)
    else:
      valid = finite_model_inputs(mo, v_ego, model_curvature, lat_action_t, right_line=False)
    valid = self._gain(mo, lat_action_t, valid)
    f = self.gain
    if not valid:
      return self._release_invalid(v_lim)
    self.k_up = self.reference.update(mo, v_ego, active, model_curvature, enabled=True)
    k0_up = self.k_up / f
    p.ref_going_around = self.reference.going_around
    p.ref_yield_fade = self.reference.yield_fade
    p.ref_departure = self.reference.yield_departure
    p.ref_lane_width = float(self.reference.lane_width.x)
    p.building = self.building
    res = RightResult()
    if right_runs:
      if self.right_idle:  # the instance was not running (Regular with a = 0): its filters are stale
        self.right.reset()
      self.right_idle = False
      res = self.right.update(mo, v_ego, active, model_curvature, k0_up, p, force_regular=(mode == MODE_REGULAR))
      k0_target = res.k0
    else:
      self.right_idle = True
      k0_target = k0_up
      res.fading = (not active) or v_ego < MIN_SPEED
    if not (np.isfinite(k0_target) and np.isfinite(self.k_up)):
      return self._release_invalid(v_lim)
    self.result = res
    c_target = f * k0_target
    release = FADING_RELEASE if res.fading else RELEASE
    self.cap_active = False
    self.release_window = False
    if right_runs and res.no_growth:
      for s in BOTH_SIDES:  # B6 cap per capped side; a side that is not capped (any more) drops its memory
        if s not in res.caps:
          self.cap_mem.pop(s, None)
          continue
        q = res.caps[s]
        toward, prev = s * c_target, max(s * self.c, 0.0)
        self.cap_mem[s] = min(self.cap_mem.get(s, prev), prev)  # set at the no-growth point, non-increasing: no latched remainder comes back
        capped = False
        if toward > prev:  # no growth toward the obstacle from q = 0.1 m
          toward, capped = prev, True
        x_cap = self.cap_mem[s] * float(np.clip((Q_CAP_FULL - q) / (Q_CAP_FULL - Q_CAP_START), 0.0, 1.0))
        if toward > x_cap:
          toward, capped = x_cap, True
        if q >= Q_CAP_FULL and s * self.c > x_cap + 1e-12:
          release, self.release_window = SWERVE_RELEASE, True
        if capped:
          c_target = s * toward
          self.cap_active = True
    else:
      self.cap_mem = {}
    c_new = self._limit(c_target, v_lim, release)
    self.building = (c_target - c_new) > 1e-9  # still building toward the right
    self.c = c_new if np.isfinite(c_new) else 0.0
    self.k0 = self.c / f
    return self.k0

  def take_update_ms(self) -> float:
    """The last update's time (ms), once: 0 on frames where the class did not run (S5)."""
    ms, self.update_ms = self.update_ms, 0.0
    return ms


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
  """Additive modelDataV2SP fields (DESIGN D logging).  The first implementation's fields without a meaning since rev 5 are gone from the
  schema (round 3; that implementation was never pushed or driven, so no log or reader depends on them)."""
  builder.blendfulMode = int(mode)
  builder.blendfulGainF = float(ctl.gain)
  builder.blendfulKappaUp = float(ctl.k_up)
  builder.blendfulKappa0 = float(ctl.k0)
  builder.blendfulCapActive = bool(ctl.cap_active)
  builder.blendfulUpdateTime = float(ctl.take_update_ms())  # the whole update of this frame, ms; 0 when the class did not run (S5)
  builder.blendfulNonFinite = bool(ctl.invalid)
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
  builder.blendfulCapMemory = float(max(ctl.cap_mem.values(), default=0.0))
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
  builder.blendfulCentreSource = int(lg['centre_source'])
  # rev 6.5 final move-speed limit (B5 step 4b)
  builder.blendfulApproach = float(lg['approach'])
  builder.blendfulGapReset = float(lg['gap_reset'])
  builder.blendfulApproachHeld = bool(lg['approach_held'])
  builder.blendfulMoveLimiting = bool(lg['move_limiting'])
  builder.blendfulMoveReset = int(lg['move_reset'])
  builder.blendfulSwitchPart = float(lg['switch_part'])
  builder.blendfulRotationScale = int(lg['rotation_scale'])
  # radar right-side clearance check (B6a item 8)
  rc = lg['radar']
  builder.blendfulRadarHealthy = bool(rc.healthy)
  builder.blendfulRadarObjects = int(min(rc.objects, 255))
  builder.blendfulRadarMemory = int(min(rc.memory, 255))
  builder.blendfulRadarMoving = int(min(rc.moving, 255))
  builder.blendfulRadarPopups = int(min(rc.popups, 255))
  builder.blendfulRadarNearest = [float(x) for x in rc.nearest]
  builder.blendfulObjectBound = float(rc.t_obj)
  builder.blendfulObjectBoundHold = float(rc.hold)
  builder.blendfulObjectBoundActive = bool(rc.bound_active)
  builder.blendfulClearanceDeficit = float(rc.deficit)
  builder.blendfulClearanceYield = bool(rc.yield_active)
  builder.blendfulClearanceMargin = float(rc.margin)
  builder.blendfulRightBlindspot = bool(rc.right_blindspot)
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
