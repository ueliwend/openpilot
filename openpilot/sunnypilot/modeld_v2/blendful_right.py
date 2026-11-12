"""
Right-line Blendful ("keep right", task A8): follows the right lane line (or road edge) at a speed dependent gap, never left of the lane
centre, and hands back to regular upstream Blendful where the right reference is not usable.  Design: inster-dev
results/2026-11-12_a8_right_line_blendful/DESIGN.md rev 4.3 (section letters B.. below refer to it).

Everything is expressed as a lateral plan shift (plan_correction.py): the profile amplitude k0 is the quantity that is blended, capped
and rate limited; the instantaneous desired-curvature correction is c = F k0.

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
# correction (B4)
STRENGTH_RIGHT_ONLY = 0.8
RISING_STRENGTH = False  # owner 17:24: flat 0.8; True = 0.8 -> 1.0 with |k_m| v^2 over 0.18 .. 0.53 m/s^2 (one-line option)
FIT_WINDOW = (0.5, 1.5)  # of L_a
ERROR_CLIP = 1.0  # m
BUILD, RELEASE, FADING_RELEASE, SWERVE_RELEASE = 0.5, 2.0, 0.5, 5.0  # m/s^3, final limiter (blendful.py:57-60; 5 = controlsd's limit)
# authority (B5)
A_DEAD, A_DWELL, A_OUT, A_IN, A_HOLD = 0.05, 0.5, 1.0, 0.5, 1.0
TARGET_RATE, FIRST_MOVE_RATE, SPLIT_RATE = 0.25, 0.1, 0.5  # m/s, offset of the target from the lane centre
# yields (B6)
RHO_BANDS_KMH = (30.0, 50.0, 80.0, 100.0)  # band edges, five bands
RHO_PRIOR = (0.5, 0.5, 0.5, 0.5, 0.5)  # placeholder until E1 fits the per-band prior
ABSORB = 0.1  # m, bound on the expected return's change per onset
Q_NO_GROWTH, Q_CAP_START, Q_CAP_FULL, Q_RELEASE = 0.1, 0.15, 0.2, 0.1
ESTABLISH_RESIDUAL, ESTABLISH_TTL = 0.05, 600.0
# junctions (B7)
EPISODE_START, EPISODE_END, EPISODE_MAX = 0.4, 0.15, 300.0  # m (start threshold is EST, tuned on E1)
PAST_GORE = 20.0  # m
RAMP_N = 1500.0  # m
HEADING_WINDOW = 40.0  # m
HEADING_LIMIT = np.radians(60.0)
BLINKER_MAX_SPEED = 32 / 3.6


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


class ConfGate:
  """Confidence gate with hysteresis: rising branch ss((q - 0.4) / 0.2), falling branch ss((q - 0.3) / 0.2) (B3)."""
  def __init__(self):
    self.c = 0.0

  def update(self, q: float) -> float:
    self.c = float(ss((q - 0.4) / 0.2) if self.c < 0.5 else ss((q - 0.3) / 0.2))
    return self.c


class InputFilter:
  """Motion compensated first order filter of one 33 point lateral position (B2)."""
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

  def predict(self, v_ego: float, yaw_rate: float) -> np.ndarray:
    dt = DT_MDL
    return quad_interp(X_IDXS + v_ego * dt, X_IDXS, self.y) - yaw_rate * dt * (X_IDXS + L_D + v_ego * dt / 2)

  def update(self, y_meas: np.ndarray, q_meas: float, v_ego: float, yaw_rate: float) -> tuple[np.ndarray, float]:
    dt = DT_MDL
    q_meas = float(np.clip(q_meas, 0.0, 1.0))
    self.reinit = False
    alpha = dt / (dt + self.tau)
    if self.initialized:
      pred = self.predict(v_ego, yaw_rate)
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
  """Boolean that turns on at once and off only after `off_time` of the condition being false."""
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


def divergence_start(x: np.ndarray, y_right: np.ndarray, y_ref: np.ndarray, v_ego: float) -> float | None:
  """x_d: first point where the right line leaves the reference: local relative curvature > 1/60 over >= 3 points, or angle > theta."""
  look = max(1.5 * v_ego, MIN_LOOKAHEAD)
  m = (x >= 2.0) & (x <= max(1.5 * look, 25.0))
  xm, d = x[m], (y_right - y_ref)[m]
  slope = np.gradient(d, xm)
  curv = np.gradient(slope, xm)
  th = np.tan(np.radians(float(np.interp(v_ego * 3.6, (50.0, 100.0), (10.0, 3.0)))))
  hit = (curv > 1 / 60) | (np.abs(slope - slope[0]) > th)
  run = np.convolve(hit.astype(int), np.ones(3, dtype=int), mode='valid') >= 3
  if np.any(run):
    return float(xm[int(np.argmax(run))])
  return float(xm[int(np.argmax(hit))]) if np.any(hit & (np.abs(slope - slope[0]) > th)) else None


def split_pattern(near_now: float, far_now: float, near_old: float, p_near: float, p_far: float) -> bool:
  """Index shift of the four lines: the new near right line sits 2 .. 4.5 m left of the old one and the new far right line where the old
  near right line was (+- 0.5 m)."""
  return bool(p_near > 0.5 and p_far > 0.5 and -4.5 <= near_now - near_old <= -2.0 and abs(far_now - near_old) < 0.5)


def gore_distance(x: np.ndarray, near: np.ndarray, far: np.ndarray) -> float | None:
  """Distance to where the dividing (near right) line and the far right line meet, from a straight fit of their separation."""
  sep = far - near
  m = (x >= 5.0) & (x <= 100.0) & (sep > 0.3) & (sep < 4.0)
  if np.sum(m) < 4:
    return None
  slope, icpt = np.polyfit(x[m], sep[m], 1)
  if slope < 0.01:
    return None
  return float(-icpt / slope)


def split_score(feat: dict) -> float:
  """Classifier on a ramp: turn-lane evidence (parallel then bends / ends 0.3, short pocket 0.2, edge cornering 0.2) against split evidence
  (both branches at moderate curvature 0.3, gore with full-width lanes 0.3, edge following the outer branch 0.2).  Map weight 0."""
  turn = 0.3 * bool(feat.get('parallel_then_bend')) + 0.2 * bool(feat.get('short_pocket')) + 0.2 * bool(feat.get('edge_cornering'))
  split = 0.3 * bool(feat.get('both_moderate')) + 0.3 * bool(feat.get('gore_full_lanes')) + 0.2 * bool(feat.get('edge_follows_outer'))
  return float(np.clip(0.5 + split - turn, 0.0, 1.0))


class JunctionLogic:
  """Junction episodes, road context (main road / exit ramp), the split classifier's hysteresis, the commit latch (B7).
  mode: None outside an episode, 'hold' (keep the through lane) or 'follow' (follow the right line)."""
  def __init__(self):
    self.width = FirstOrderFilter(LANE_WIDTH_DEFAULT, 10.0, DT_MDL, initialized=False)
    self.single = Timed(off_time=5.0, on_time=3.0)
    self.reset()

  def reset(self):
    self.episode = False
    self.mode: str | None = None
    self.latched = False
    self.ep_dist = 0.0
    self.quiet = 0.0
    self.split_seen = False
    self.followed = False
    self.rebase_width = False
    self.gore_x: float | None = None
    self.score_state = 'uncertain'
    self.score = 0.5
    self.blink_latched = False
    self.choice_follow = False
    self.ramp = False
    self.ramp_dist = 0.0
    self.left_lane_t = 0.0
    self.turn_buf: deque = deque()
    self.single.state = False

  @property
  def width_m(self) -> float:
    return float(self.width.x)

  def update_width(self, width: float, ok: bool):
    if self.rebase_width and ok and not self.episode:  # the widened lane stays: it is the new through lane
      self.width.x, self.rebase_width = float(np.clip(width, *WIDTH_RANGE)), False
    elif not self.width.initialized:
      if ok:  # remembered through-lane width starts from the first confident, plausible measurement
        self.width.update(float(np.clip(width, *WIDTH_RANGE)))
    elif ok and not self.episode and abs(width - self.width.x) < EPISODE_START:
      self.width.update(float(np.clip(width, *WIDTH_RANGE)))

  def _context(self, v: float, yaw_rate: float, ds: float, left_lane: bool, single_now: bool):
    single = self.single.update(single_now)
    self.turn_buf.append((ds, yaw_rate * DT_MDL))
    while self.turn_buf and sum(d for d, _ in self.turn_buf) - self.turn_buf[0][0] >= HEADING_WINDOW:
      self.turn_buf.popleft()
    tight = abs(sum(p for _, p in self.turn_buf)) > HEADING_LIMIT
    if not self.ramp:
      return
    self.ramp_dist += ds
    self.left_lane_t = self.left_lane_t + DT_MDL if left_lane else 0.0
    if v < 2.0 or tight:
      self.ramp = False
    elif not single and (self.ramp_dist > RAMP_N or self.left_lane_t >= 5.0):
      self.ramp = False

  def update(self, v: float, yaw_rate: float, div: float | None, split_now: bool, gore: float | None, score: float | None,
             right_blinker: bool, left_blinker: bool, left_lane: bool, single_now: bool) -> None:
    ds = v * DT_MDL
    self._context(v, yaw_rate, ds, left_lane, single_now)
    if not self.episode:
      if split_now or (div is not None and div > EPISODE_START):
        self.reset_episode_state()
        self.episode, self.split_seen = True, split_now
      else:
        self.mode = None
        return
    else:
      self.ep_dist += ds
      self.split_seen |= split_now
    # gore bookkeeping
    if gore is not None:
      self.gore_x = gore
    elif self.gore_x is not None:
      self.gore_x -= ds
    # classifier hysteresis (ramp only): >= 0.7 split (leave < 0.6), <= 0.3 turn lane (leave > 0.4)
    if score is not None:
      self.score = score
      s = self.score_state
      if s == 'split':
        s = 'split' if score >= 0.6 else 'uncertain'
      elif s == 'turn':
        s = 'turn' if score <= 0.4 else 'uncertain'
      if s == 'uncertain':
        s = 'split' if score >= 0.7 else ('turn' if score <= 0.3 else 'uncertain')
      self.score_state = s
    # the choice, then the commit latch: inside x_commit it is frozen until the gore is 20 m behind
    if not self.latched:
      self.choice_follow = right_blinker or (self.ramp and self.score_state == 'split')
    if self.split_seen and not self.latched:
      x_commit = (self.width_m / SPLIT_RATE + 1.0) * v
      if self.gore_x is None or self.gore_x <= x_commit:
        self.latched = True
        self.blink_latched = right_blinker
    self.mode = 'follow' if self.choice_follow else 'hold'
    if self.mode == 'follow' and self.split_seen:
      self.followed = True
    # end of episode
    if div is not None and div < EPISODE_END and not self.split_seen:
      self.quiet += DT_MDL
    else:
      self.quiet = 0.0
    gore_passed = self.gore_x is not None and self.gore_x <= -PAST_GORE
    if self.quiet >= 1.0 or self.ep_dist > EPISODE_MAX or gore_passed:
      if self.followed and self.split_seen:  # the car took the branch: it is on a ramp now
        self.ramp, self.ramp_dist, self.left_lane_t = True, 0.0, 0.0
      self.episode, self.mode, self.latched, self.rebase_width = False, None, False, True
      self.split_seen, self.followed, self.gore_x = False, False, None

  def reset_episode_state(self):
    self.ep_dist, self.quiet, self.latched = 0.0, 0.0, False
    self.followed, self.gore_x, self.choice_follow = False, None, False
    self.score_state, self.score, self.blink_latched = 'uncertain', 0.5, False


# ---------------------------------------------------------------------------------------------------------------------------------
class YieldEstimator:
  """Car-referenced yield trigger with the model's expected return (B6): p_exp = -rho_hat car + b_hat, rho_hat in [0, 1] per speed band
  (RLS, 2 s forgetting, |d rho_hat / dt| <= 0.2 /s), b_hat learned only on a steady car, p_exp within 0.1 m of the onset-start prediction
  during an onset.  p = plan endpoint relative to the car in lane coordinates, car = the car's offset right of the lane centre."""
  def __init__(self, prior=RHO_PRIOR):
    self.prior = tuple(prior)
    self.reset()

  def reset(self):
    self.rho = list(self.prior)
    self.established = [None] * len(self.rho)
    self.b = 0.0
    self.pr = 1.0
    self.car_prev: float | None = None
    self.ps: float | None = None
    self.onset = False
    self.r0 = self.b0 = 0.0
    self.onset_car = 0.0
    self.t = 0.0
    self.q = 0.0
    self.p_exp = 0.0
    self.departure = 0.0

  @staticmethod
  def band(v: float) -> int:
    return int(np.searchsorted(np.array(RHO_BANDS_KMH) / 3.6, v))

  def is_established(self, v: float) -> bool:
    t = self.established[self.band(v)]
    return t is not None and self.t - t < ESTABLISH_TTL

  def road_type_changed(self):
    self.established = [None] * len(self.rho)

  def rho_hat(self, v: float) -> float:
    return self.rho[self.band(v)]

  def update(self, car: float, p: float, v: float, allowance: float, going_around: bool) -> float:
    dt = DT_MDL
    self.t += dt
    k = self.band(v)
    rh = self.rho[k]
    ps_prev = self.ps
    pl = car + p
    self.ps = pl if self.ps is None else self.ps + (pl - self.ps) * dt / (dt + 1.0)
    left = ps_prev is not None and (self.ps - ps_prev) / dt < -0.05
    do = 0.0 if self.car_prev is None else (car - self.car_prev) / dt
    self.car_prev = car
    moving = abs(do) > 0.05
    if moving and not self.onset:
      self.onset, self.r0, self.b0, self.onset_car = True, rh, self.b, car
    ended = self.onset and not moving
    praw = -rh * car + self.b
    base = -self.r0 * car + self.b0
    p_exp = base + float(np.clip(praw - base, -ABSORB, ABSORB)) if self.onset else praw
    q = abs(p - p_exp) - allowance
    if ended:
      self.onset = False
      if abs(car - self.onset_car) > 0.2 and abs(p - p_exp) < ESTABLISH_RESIDUAL:
        self.established[k] = self.t
    if q < Q_NO_GROWTH:
      lam = float(np.exp(-dt / 2.0))
      if moving and not left:
        kk = self.pr * car / (lam + car * self.pr * car)
        new = rh + kk * (-(p - self.b) - rh * car)
        self.rho[k] = float(np.clip(rh + np.clip(new - rh, -0.2 * dt, 0.2 * dt), 0.0, 1.0))
        self.pr = (self.pr - kk * car * self.pr) / lam
      if not moving:
        self.b += ((p + self.rho[k] * car) - self.b) * dt / (dt + (20.0 if going_around else 10.0))
    self.q, self.p_exp = float(q), float(p_exp)
    self.departure = float(p_exp - p)  # > 0: the model moves left of what it was expected to do (obstacle on the right)
    return self.q


# ---------------------------------------------------------------------------------------------------------------------------------
# Junction policy (B7): the ONLY place that decides what a junction episode does to the correction.  Episode detection, the road context
# and logging live in JunctionLogic; a policy change (e.g. "inside an episode both corrections fade and the model drives alone") touches
# junction_mode() / junction_policy() and their tests only.
@dataclass
class PolicyWeights:
  w_rl: float = 0.0  # right-line part
  w_up: float = 0.0  # upstream part (reference instance)
  w_hold: float = 0.0  # hold part (through lane measured from the left reference)

  def total(self) -> float:
    return self.w_rl + self.w_up + self.w_hold


def junction_mode(episode: bool, mode: str | None, right_ok: bool, left_blinker_low_speed: bool, latched: bool, left_ref_ok: bool) -> tuple[str | None, bool]:
  """Effective mode inside an episode and the model-alone flag.  Every request that would hand over to Regular becomes hold: a lost or
  implausible right reference, a jump, and the left blinker below 32 km/h (not inside the latch).  No usable left reference: model alone."""
  if not episode:
    return None, False
  if mode == 'follow' and not right_ok:
    mode = 'hold'
  if left_blinker_low_speed and not latched:
    mode = 'hold'
  return mode, (mode == 'hold' and not left_ref_ok)


def junction_policy(episode: bool, mode: str | None, a: float, model_alone: bool) -> PolicyWeights:
  """Per-frame split of the correction: k0 = w_rl k0_rl + w_up k0_up + w_hold k0_hold.  Outside an episode the right-line authority a
  blends with regular upstream; inside, the (1 - a) share is the hold correction and never the upstream one (review C1)."""
  if not episode:
    return PolicyWeights(a, 1.0 - a, 0.0)
  if model_alone:
    return PolicyWeights()
  if mode == 'hold':
    return PolicyWeights(0.0, 0.0, 1.0)
  return PolicyWeights(a, 0.0, 1.0 - a)


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
  ref_going_around: bool = False  # the reference instance's going-around yield latch


@dataclass
class RightResult:
  k0: float = 0.0  # profile amplitude before the yield cap and the final limiter
  fading: bool = False
  yield_active: bool = False
  yield_q: float = 0.0
  yield_side: int = 0  # +1: obstacle on the right (model moves left), -1: obstacle on the left
  model_alone: bool = False
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
    self.junction = JunctionLogic()
    self.yields = YieldEstimator()
    self.h_lim: float | None = None
    self.bend_active = Timed(off_time=1.0)
    self.jump_hold = 0.0
    self.blink_on = Timed(off_time=1.0, on_time=0.3)
    self.f_blink = 1.0
    self.yield_latch = Timed(off_time=1.0)
    self.ref_yield_latch = Timed(off_time=1.0)
    self.road_type: str | None = None
    self.far_prev = None
    self.q_prev = {}
    self.a_star = 0.0
    self.last_q_right = 0.0
    self.prev_a_cmd = 0.0
    self.pred_old = None

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

  def _classify_road(self, v: float, q_l: float, q_r: float, width: float) -> str:
    if q_l < 0.3 and q_r < 0.3:
      return 'edges'
    if width < 3.0:
      return 'shared'
    return 'motorway' if v > 22.0 and width > 3.3 else 'lanes'

  def _correction(self, target: np.ndarray, plan_x, plan_y, v: float, strength: float):
    la = approach_length(max(v, 1.0))
    l_c = profile_length(max(v, 1.0))
    win = (X_IDXS >= FIT_WINDOW[0] * la) & (X_IDXS <= FIT_WINDOW[1] * la)
    x = X_IDXS[win]
    err = np.clip(target[win] - np.interp(x, plan_x, plan_y), -ERROR_CLIP, ERROR_CLIP)
    d, k = fit_amplitude(x, err, 1.0, l_c)
    return d, strength * k, k

  def update(self, mo: dict, v_ego: float, active: bool, model_curvature: float, k0_up: float, p: RightInputs,
             force_regular: bool = False) -> RightResult:
    t0 = time.perf_counter()
    dt = DT_MDL
    v = max(float(v_ego), 0.0)
    fading = (not active) or v < MIN_SPEED
    look = max(v * LOOKAHEAD_TIME, MIN_LOOKAHEAD)
    iL = look  # look-ahead distance for scalar read outs
    lines = mo['lane_lines'][0, :, :, 0]
    lstds = mo['lane_lines_stds'][0, :, :, 0]
    plan = mo['plan'][0]
    plan_x, plan_y = plan[:, 0], plan[:, 1]
    yaw_rate = float(plan[0, Plan.ORIENTATION_RATE.start + 2])
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
      estds = np.ones((2, NX))
      eq_raw = np.zeros(2)
    e_q = self.edge_q.update(eq_raw)
    # --- B2: the four filters (left line, left edge, right line, right edge)
    raw = [lines[1], edges[0], lines[2], edges[1]]
    qraw = [l_q, float(e_q[0]), r_q, float(e_q[1])]
    filt, qf = [], []
    for f, y, q in zip(self.filters, raw, qraw, strict=True):
      yy, qq = f.update(y, q, v, yaw_rate)
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
    # --- jump (innovation of the right line / edge) and split pattern
    inno = max([abs(float(np.interp(iL, X_IDXS, self.filters[i].innovation))) for i in (2, 3) if qf[i] > 0.5] or [0.0])
    near_now, far_now = float(np.interp(iL, X_IDXS, lines[2])), float(np.interp(iL, X_IDXS, lines[3]))
    near_old = float(np.interp(iL, X_IDXS, self.filters[2].y_pred)) if self.filters[2].initialized else near_now
    split_now = split_pattern(near_now, far_now, near_old, float(probs_raw[2]), float(probs_raw[3]))
    if inno > 0.5 and not split_now:
      self.jump_hold = 2.0
    else:
      self.jump_hold = max(0.0, self.jump_hold - dt)
    f_jump_t = 0.0 if self.jump_hold > 0.0 else 1.0
    # --- lane width and centre (left limit)
    ok_w = both and not bad
    self.junction.update_width(width_now, ok_w)
    hw_known = self.junction.width_m / 2
    if both:
      hw = (y_r - y_l) / 2
      c_ctr, _ = Blendful.centre_and_weight(y_l, y_r, lstds[1], lstds[2], max(q_L, 1e-4), max(q_R, 1e-4), hw)
    elif q_R >= q_L:
      c_ctr = y_r - hw_known
    else:
      c_ctr = y_l + hw_known
    # --- B7 shape detectors
    use_left = q_L > 0.5
    y_ref = y_l if use_left else np.interp(X_IDXS, plan_x, plan_y)
    _, beta, _, b = bend_score(X_IDXS, y_r, y_ref, max(v, 1.0))
    bend_on = self.bend_active.update(b > 0.5 if not self.bend_active.state else b >= 0.2)
    f_bend_t = 0.0 if bend_on else 1.0
    x_d = divergence_start(X_IDXS, y_r, y_ref, max(v, 1.0)) if b > 0.0 or bend_on else None
    y_r_g = y_r.copy()
    if x_d is not None:
      k = X_IDXS >= x_d
      y_r_g[k] = np.interp(x_d, X_IDXS, y_r) + (y_ref[k] - np.interp(x_d, X_IDXS, y_ref))
    # --- junction logic
    div = (float(np.interp(iL, X_IDXS, y_r - y_l)) - self.junction.width_m) if (both and self.junction.width.initialized) else None
    gore = gore_distance(X_IDXS, lines[2], lines[3]) if split_now or self.junction.split_seen else None
    feat = self._features(X_IDXS, lines, edges, probs_raw, lstds, y_l, max(v, 1.0))
    score = split_score(feat) if self.junction.ramp else None
    left_lane = float(probs_raw[0]) > 0.5
    single_now = (float(probs_raw[0]) < 0.3 and qf[1] > 0.5 and 0.0 <= float(np.interp(iL, X_IDXS, filt[0] - filt[1])) <= 1.0)
    self.junction.update(v, yaw_rate, div, split_now, gore, score, p.right_blinker, p.left_blinker, left_lane, single_now)
    jn = self.junction
    # --- blinker fade (left blinker hands back to upstream; B9)
    lb = self.blink_on.update(p.left_blinker)
    self.f_blink = max(0.0, self.f_blink - dt) if lb else min(1.0, self.f_blink + dt)
    # --- road type (resets the established flags of the yield estimator)
    rt = self._classify_road(v, q_L, q_R, width_now)
    if self.road_type is not None and rt != self.road_type:
      self.yields.road_type_changed()
    self.road_type = rt
    # --- targets (B1, B8)
    g = gap_for_speed(v, p.gap_low, p.gap_high)
    win_ref = (X_IDXS >= 0.5 * look) & (X_IDXS <= 1.5 * look)
    k_ref = 2.0 * float(np.polyfit(X_IDXS[win_ref], y_r_g[win_ref], 2)[0])
    off_track = L_D ** 2 * max(k_ref, 0.0) / 2
    dc = p.c_set - p.c_applied
    hold_line = (y_l + jn.width_m) if use_left else None
    right_ok = q_R >= 0.3 and f_jump_t >= 1.0 and f_plaus_t >= 1.0
    mode, model_alone = junction_mode(jn.episode, jn.mode, right_ok, p.left_blinker and v < BLINKER_MAX_SPEED, jn.latched, hold_line is not None)
    y_eff = hold_line if (jn.episode and mode == 'hold' and hold_line is not None) else y_r_g
    # C_stable: the centre from the left reference and the remembered width; unlike c_ctr it does not move when the right reference
    # jumps or the right line widens, so the rate limit below acts on the target's own motion (implementation choice, IMPL.md)
    c_stable = (y_l + jn.width_m / 2) if q_L >= 0.3 else c_ctr
    c_use = c_stable if (y_eff is hold_line and hold_line is not None) else c_ctr
    t_r_line = y_eff - W_R - g - off_track - dc
    t_raw = np.maximum(t_r_line, c_use)
    t_bound_line = y_eff - W_R - LINE_BOUND - off_track - dc
    w_e = float(ss((qf[3] - 0.4) / 0.2))
    t_bound_edge = filt[3] - W_R - EDGE_BOUND - off_track - dc
    t_bounded = np.minimum(t_raw, t_bound_line)
    t_bounded = t_bounded - w_e * np.maximum(0.0, t_bounded - t_bound_edge)
    bound_active = bool(np.any(t_bounded < t_raw - 1e-9))
    # --- target offset rate (glides at splits, line <-> edge)
    h_raw = float(np.interp(iL, X_IDXS, t_bounded - c_stable))
    first_move = not self.yields.is_established(max(v, 1.0))
    rate = SPLIT_RATE if (jn.episode and mode == 'follow') else (FIRST_MOVE_RATE if first_move else TARGET_RATE)
    if self.h_lim is None:
      self.h_lim = h_raw
    self.h_lim = float(np.clip(h_raw, self.h_lim - rate * dt, self.h_lim + rate * dt))
    target = t_bounded - (h_raw - self.h_lim)
    # --- correction (B4)
    w_c = float(np.interp(abs(model_curvature) * v ** 2, [0.18, 0.53], [STRENGTH_RIGHT_ONLY, 1.0])) if self.strength_rising else STRENGTH_RIGHT_ONLY
    s_r = 1.0 - (1.0 - l_q) * (1.0 - w_c * q_R)
    d_rl, k0_rl, k_fit = self._correction(target, plan_x, plan_y, v, s_r)
    # hold correction: the through lane position measured from the left reference
    if hold_line is not None:
      th = np.minimum(np.maximum(hold_line - W_R - g - off_track - dc, c_stable), hold_line - W_R - LINE_BOUND - off_track - dc)
      _, k0_hold, _ = self._correction(th, plan_x, plan_y, v, s_r)
    else:
      k0_hold = 0.0
    # --- yields (B6)
    c0 = -float(np.interp(0.0, X_IDXS, c_ctr))  # the car's offset right of the lane centre
    p_lane = float(np.interp(look, plan_x, plan_y) - np.interp(look, X_IDXS, c_ctr))
    allow = 0.25 * abs(model_curvature) * look ** 2 / 2
    valid_y = q_R >= 0.3 and not fading
    q_y = self.yields.update(c0, p_lane - c0, max(v, 1.0), allow, p.ref_going_around) if valid_y else 0.0
    side = 1 if self.yields.departure > 0 else -1
    car_trig = self.yield_latch.update(q_y > Q_CAP_FULL if not self.yield_latch.state else q_y >= Q_RELEASE)
    ref_trig = self.ref_yield_latch.update(p.ref_going_around)
    yield_active = bool(car_trig or ref_trig) and valid_y
    q_eff = max(q_y, Q_CAP_FULL) if ref_trig else q_y
    f_yl = 0.0 if (yield_active and side > 0) else 1.0
    f_yr = float(np.interp(q_eff, (0.2, 0.45), (1.0, 0.0))) if (yield_active and side < 0) else 1.0
    k0_rl *= f_yr
    # --- authority (B5)
    qa = self.quality_gate.update(q_R)
    a_star = float(qa)
    f_split = 0.0 if (jn.episode and jn.split_seen and mode == 'hold') else 1.0
    a_star *= f_bend_t * f_split * f_jump_t * f_plaus_t * self.f_blink * f_yl
    if force_regular or fading:
      a_star = 0.0
    if q_y >= Q_NO_GROWTH:
      a_star = min(a_star, self.authority.a_cmd)
    a = self.authority.update(a_star)
    wts = junction_policy(jn.episode, mode, a, model_alone)
    k0 = wts.w_rl * k0_rl + wts.w_up * k0_up + wts.w_hold * k0_hold
    if fading:
      k0 = 0.0
    right_only = bool(q_R > 0.5 and l_q < 0.3)
    gap_now = float(np.interp(0.0, X_IDXS, y_r) - W_R)
    log = {
      'applied': a > 0.0, 'gap': g, 'gap_now': gap_now, 'q_R': q_R, 'q_L': q_L, 'a': a, 'a_star': a_star,
      'f_bend': f_bend_t, 'f_split': f_split, 'f_yl': f_yl, 'f_yr': f_yr, 'f_jump': f_jump_t, 'f_plaus': f_plaus_t, 'f_blink': self.f_blink,
      'away_car': q_y, 'rho': self.yields.rho_hat(max(v, 1.0)), 'bias': self.yields.b, 's_r': s_r, 'w_c': w_c, 'right_only': right_only,
      'curve_accel': abs(model_curvature) * v ** 2, 'yield_active': yield_active, 'bound_active': bound_active,
      'k_rl': k0_rl, 'weights': (wts.w_rl, wts.w_up, wts.w_hold), 'l_c': profile_length(max(v, 1.0)), 'd': d_rl, 'filter_ms': t_filter,
      'filtered': np.concatenate([[np.interp(xq, X_IDXS, f) for xq in (0.0, 0.5 * look, look, 1.5 * look)] for f in filt]),
      'raw': np.concatenate([[np.interp(xq, X_IDXS, r_) for xq in (0.0, 0.5 * look, look, 1.5 * look)] for r_ in raw]),
      'filter_offset': np.array([float(np.interp(iL, X_IDXS, f - r_)) for f, r_ in zip(filt, raw, strict=True)]),
      'confidence': np.array(qf), 'state': np.array([int(f.coast) | (int(f.reinit) << 1) for f in self.filters]),
      'cand_conf': np.array([c_l[0], c_l[1], c_r[0], c_r[1]]),
      'cand_weight': np.array([w_l[0][int(np.searchsorted(X_IDXS, iL))], w_l[1][int(np.searchsorted(X_IDXS, iL))],
                               w_r[0][int(np.searchsorted(X_IDXS, iL))], w_r[1][int(np.searchsorted(X_IDXS, iL))]]),
      'reference': np.array([np.interp(iL, X_IDXS, y_l), np.interp(iL, X_IDXS, y_r)]), 'target': target,
      'episode': jn.episode, 'latched': jn.latched, 'ramp': jn.ramp, 'hold': bool(jn.episode and mode == 'hold'), 'score': jn.score,
      'c_ctr': c_ctr, 'c_stable': c_stable, 'y_r': y_r, 'y_l': y_l, 'hold_line': hold_line, 'k0_hold': k0_hold, 'split_now': split_now, 'beta': beta,
    }
    return RightResult(k0=float(k0), fading=fading, yield_active=yield_active, yield_q=float(q_eff), yield_side=side,
                       model_alone=model_alone, log=log)

  @staticmethod
  def _features(x, lines, edges, probs_raw, lstds, y_l, v) -> dict:
    """Geometric evidence for the split classifier from the raw lines (B7); the short pocket uses the far right line's std growth."""
    near, far = lines[2], lines[3]
    feat: dict = {}
    try:
      _, b_near, _, bn = bend_score(x, near, y_l, v)
      _, b_far, _, bf = bend_score(x, far, y_l, v)
      feat['both_moderate'] = 1 / 1000 < b_near < 1 / 60 and 1 / 1000 < b_far < 1 / 60
      feat['parallel_then_bend'] = bn < 0.2 and bf >= 0.5
    except np.linalg.LinAlgError:
      pass
    sep = far - near
    feat['gore_full_lanes'] = bool(np.interp(60.0, x, sep) >= 2.6 and np.interp(60.0, x, near - y_l) >= 2.6)
    feat['short_pocket'] = bool(np.interp(120.0, x, lstds[3]) > 0.6)
    ed = edges[1]
    feat['edge_follows_outer'] = bool(np.std((ed - far)[(x > 5) & (x < 80)]) < 0.5)
    feat['edge_cornering'] = bool(np.interp(80.0, x, ed) - np.interp(10.0, x, ed) - (np.interp(80.0, x, far) - np.interp(10.0, x, far)) > 2.0)
    return feat


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
    self.cap_latch = 0.0
    self.cap_active = False
    self.release_window = False
    self.last_s_c = 0.0

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
    self.gain = instant_gain(v_eff, lat_action_t)
    f = self.gain
    if mode == MODE_OFF:
      self.reference.update(mo, v_ego, active, model_curvature, enabled=False)
      self.cap_active = False
      c = self._limit(0.0, v_ego, RELEASE)
      self.c, self.k0, self.k_up = c, c / f, 0.0
      self.result = RightResult()
      if self.c == 0.0:
        self.right.reset()
      self.mode = mode
      return self.k0
    if self.mode == MODE_OFF:
      self.right.reset()
    if mode == MODE_REGULAR and self.mode == MODE_RIGHT:
      self.right.junction.reset()
    self.mode = mode
    self.k_up = self.reference.update(mo, v_ego, active, model_curvature, enabled=True)
    k0_up = self.k_up / f
    p.ref_going_around = self.reference.going_around
    res = RightResult()
    if mode == MODE_RIGHT or (mode == MODE_REGULAR and self.right.a > 0.0):
      res = self.right.update(mo, v_ego, active, model_curvature, k0_up, p, force_regular=(mode == MODE_REGULAR))
      k0_target = res.k0
    else:
      k0_target = k0_up
      res.fading = (not active) or v_ego < MIN_SPEED
    self.result = res
    c_target = f * k0_target
    release = FADING_RELEASE if res.fading else RELEASE
    self.cap_active = False
    self.release_window = False
    if mode == MODE_RIGHT and res.yield_active:
      s, q = res.yield_side, res.yield_q
      toward, prev = s * c_target, max(s * self.c, 0.0)
      if q < Q_NO_GROWTH:
        self.cap_latch = 0.0
      if q >= Q_NO_GROWTH and toward > prev:
        toward, self.cap_active = prev, True
      if q >= Q_CAP_START:
        if self.cap_latch == 0.0:
          self.cap_latch = prev
        x_cap = self.cap_latch * float(np.clip((Q_CAP_FULL - q) / (Q_CAP_FULL - Q_CAP_START), 0.0, 1.0))
        if toward > x_cap:
          toward, self.cap_active = x_cap, True
        if q >= Q_CAP_FULL and s * self.c > x_cap + 1e-12:
          release, self.release_window = SWERVE_RELEASE, True
      if self.cap_active:
        c_target = s * toward if s * c_target > toward else c_target
    elif not res.yield_active:
      self.cap_latch = 0.0
    self.c = self._limit(c_target, v_ego, release)
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
  """Additive modelDataV2SP fields (DESIGN D logging)."""
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
                    ('AuthorityTarget', 'a_star'), ('FadeBend', 'f_bend'), ('FadeSplit', 'f_split'), ('FadeYieldLeft', 'f_yl'),
                    ('FadeYieldRight', 'f_yr'), ('FadeJump', 'f_jump'), ('FadePlausibility', 'f_plaus'), ('FadeBlinker', 'f_blink'),
                    ('AwayCar', 'away_car'), ('Rho', 'rho'), ('Bias', 'bias'), ('Strength', 's_r'), ('StrengthWeight', 'w_c'),
                    ('CurveAccel', 'curve_accel'), ('KappaRL', 'k_rl'), ('ProfileLength', 'l_c'), ('ProfileShift', 'd'),
                    ('FilterTime', 'filter_ms'), ('SplitScore', 'score')):
    setattr(builder, 'blendful' + name, float(lg[key]))
  builder.blendfulRightOnly = bool(lg['right_only'])
  builder.blendfulYieldActive = bool(lg['yield_active'])
  builder.blendfulBoundActive = bool(lg['bound_active'])
  builder.blendfulJunctionEpisode = bool(lg['episode'])
  builder.blendfulJunctionLatched = bool(lg['latched'])
  builder.blendfulJunctionRamp = bool(lg['ramp'])
  builder.blendfulJunctionHold = bool(lg['hold'])
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
