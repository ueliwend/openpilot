"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Hyundai Inster steering candidates for sunnypilot's torque controller v0 (latcontrol_torque_v0.py). Experimental, Inster only,
each behind its own switch that is off by default. The switches are read by v0, never here: this module only holds pure functions
and a small stateful filter class.

  * Speed feed-forward table (switch InsterSpeedFeedforward): ff_torque() gives the steering command the Inster needs to hold a
    lateral acceleration at a given speed, from a measured table, instead of v0's single linear factor. v0 uses it only for its
    feed-forward, expressed back in lateral-acceleration units (ff_lat_accel_equivalent), so P, I, the limits and the output
    conversion stay linear and unchanged.
  * Stiction friction term (switch InsterStictionFriction): InsterStictionFriction replaces v0's get_friction(). It pushes the wheel
    in the direction the plan wants it to move, smoothly (no dead zone, no hysteresis), with an amplitude matched to the Inster's
    measured steering friction, and lets go of the held torque early when the plan starts to unwind.
  * Exit tuning (switch InsterExitTuning, step 3, simulator candidate "r66_all_c"; only together with both switches above): a narrower
    smooth sign, a faster intent filter, damping of the wheel against the plan's wheel rate while the term releases, and a lower,
    flatter low-speed row of the feed-forward table above 1 m/s^2.
  * Low-speed P (switch InsterLowSpeedP, step 3, simulator lever "r6c_kplo"): v0's proportional gain x 0.75 below 20 km/h, blending
    to x 1 at 30 km/h (low_speed_kp_scale). Works with plain v0 and with the switches above.
  * Speed-dependent torque limit (switch InsterHighTorqueLimits, step 3): passed to the car side as HyundaiFlagsSP.INSTER_HIGH_TORQUE
    (opendbc); v0 may then output up to inster_steer_max(v) / 384 (> 1.0, 1.0 stays 384 raw) and openpilot plans lateral
    accelerations up to 3.3 m/s^2 instead of 3.0 (high_torque_steer_max, max_lateral_accel_no_roll).

Units and signs: commands in raw LKAS11 units (full scale STEER_MAX_RAW = 384) or as a fraction of full torque; v0's internal sign
convention (+ = right turn), the convention of v0's setpoint, measurement and output before its final negation.
Derived offline from the owner's logs (step 1, 21 Oct 2026; open-loop replays and the closed-loop steering simulator).
Nothing here is validated on the car.
"""
import math

from opendbc.car.hyundai.values import CAR, INSTER_HIGH_TORQUE_MAX_LATERAL_ACCEL, inster_steer_max
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP

# Full-scale command the table and the friction term are expressed in (opendbc INSTER_STEER_MAX at the time they were derived).
# If the Inster's cap changes, F0 and the raw amplitudes below need a review: test_inster_lateral.py checks that both agree.
STEER_MAX_RAW = 384.0


# The Inster's switches (params, all off by default, changeable only offroad, ignored while NNLC is enabled). With any of them on, the
# Lateral Jerk Torque Controller is switched off.
INSTER_SWITCHES = ("InsterStictionFriction", "InsterSpeedFeedforward", "InsterExitTuning", "InsterLowSpeedP", "InsterHighTorqueLimits")


def is_inster(CP) -> bool:
  """True for the Hyundai Inster (Casper Electric) fingerprint."""
  return CP is not None and CP.carFingerprint == CAR.HYUNDAI_INSTER


def any_inster_switch(params) -> bool:
  return any(params.get_bool(k) for k in INSTER_SWITCHES)


# ---------------------------------------------------------------------------------------------------------------------------------
# Speed-dependent torque limit (switch InsterHighTorqueLimits). The car side reads the switch at start as the CarParamsSP flag
# HyundaiFlagsSP.INSTER_HIGH_TORQUE (set only for the Inster and not while NNLC is enabled); openpilot uses the same flag, so both
# sides always agree. Torque stays normalised to STEER_MAX_RAW: 1.0 = 384 raw.
def is_inster_high_torque(CP, CP_SP) -> bool:
  return is_inster(CP) and CP_SP is not None and bool(CP_SP.flags & HyundaiFlagsSP.INSTER_HIGH_TORQUE)


def high_torque_steer_max(v_ego_raw: float) -> float:
  """v0's output limit (fraction of STEER_MAX_RAW, >= 1) with the switch on: the car side's limit at the same speed (vEgoRaw)."""
  return inster_steer_max(v_ego_raw) / STEER_MAX_RAW


def max_lateral_accel_no_roll(lat_control, default: float) -> float:
  """Highest planned lateral acceleration without roll (m/s^2): 3.3 only when the running lateral controller is torque controller v0
  with the speed-dependent limit active (lat_control.inster_high_torque: the Inster with the flag, NNLC off); otherwise the default
  (3.0), also for the other torque tune versions, which keep their output within 1.0."""
  return INSTER_HIGH_TORQUE_MAX_LATERAL_ACCEL if getattr(lat_control, "inster_high_torque", False) is True else default


# ---------------------------------------------------------------------------------------------------------------------------------
# Low-speed P (switch InsterLowSpeedP; simulator lever r6c_kplo, 22 Oct 2026): at low speed v0's P term scales with the learned factor
# (P is in lateral-acceleration units, the output is divided by the factor), so at 3.7 it was 0.75 x what it is at 2.8.
LOW_SPEED_KP_SCALE = 0.75          # P gain factor at and below LOW_SPEED_KP_BP[0]
LOW_SPEED_KP_BP = (5.56, 8.33)     # m/s: 20 and 30 km/h; linear in between, 1 from 30 km/h


def low_speed_kp_scale(v_ego: float) -> float:
  if v_ego <= LOW_SPEED_KP_BP[0]:
    return LOW_SPEED_KP_SCALE
  if v_ego >= LOW_SPEED_KP_BP[1]:
    return 1.0
  return LOW_SPEED_KP_SCALE + (1.0 - LOW_SPEED_KP_SCALE) * (v_ego - LOW_SPEED_KP_BP[0]) / (LOW_SPEED_KP_BP[1] - LOW_SPEED_KP_BP[0])


# ---------------------------------------------------------------------------------------------------------------------------------
# Speed feed-forward table (step 1 curve study, candidate "b-ff"; LOG: ten drives 14-21 Oct 2026).
# Steering command (raw) needed for a gravity-adjusted lateral acceleration |a| (m/s^2) at speed v. Odd in a, bilinear (linear between
# the nodes in v, then in a), clamped at the outer speed rows (22 km/h below, 111 km/h above) and at 3.5 m/s^2.
A_NODES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5)       # m/s^2
V_NODES = tuple(x / 3.6 for x in (22.0, 32.0, 48.0, 76.0, 111.0))          # m/s
CMD_RAW = (                                                                # rows: V_NODES, columns: A_NODES
  (0.0, 23.0, 57.1, 142.4, 250.6, 303.8, 352.2, 384.0, 384.0, 384.0, 384.0),   # 22 km/h  (measured to 1.25 m/s^2)
  (0.0, 23.0, 48.0, 108.6, 146.2, 190.6, 242.5, 314.7, 384.0, 384.0, 384.0),   # 32 km/h  (measured to 2.5)
  (0.0, 21.1, 48.0, 95.4, 116.1, 141.7, 168.6, 267.9, 328.0, 384.0, 384.0),    # 48 km/h  (measured to 3.0)
  (0.0, 19.8, 39.0, 67.9, 97.9, 130.3, 168.6, 267.9, 328.0, 384.0, 384.0),     # 76 km/h  (measured to 1.5)
  (0.0, 19.8, 38.8, 57.7, 85.2, 115.8, 154.0, 251.0, 318.6, 384.0, 384.0),     # 111 km/h (measured to 1.0)
)
F0 = 2.8                          # latAccelFactor at which the table is used as measured (the learner's value when it was built)
SCALE_CLIP = (1.0 / 1.15, 1.15)   # bound on how far the learned factor can scale the table (the table is read at |a| * F0 / factor)
_TAB = tuple(tuple(c / STEER_MAX_RAW for c in row) for row in CMD_RAW)

# Exit tuning (switch InsterExitTuning): the 22 km/h row pulled to the 32 km/h row at |a| >= EXIT_TABLE_A0 (where it is higher), i.e.
# a lower, flatter low-speed feed-forward at high lateral acceleration (simulator: tab_lo_blend 1, tab_lo_a0 1.0)
EXIT_TABLE_A0 = 1.0
EXIT_TABLE_BLEND = 1.0
CMD_RAW_EXIT = (tuple(c - EXIT_TABLE_BLEND * max(0.0, c - c32) if a >= EXIT_TABLE_A0 else c
                      for c, c32, a in zip(CMD_RAW[0], CMD_RAW[1], A_NODES, strict=True)),) + CMD_RAW[1:]
_TAB_EXIT = tuple(tuple(c / STEER_MAX_RAW for c in row) for row in CMD_RAW_EXIT)
FF_TABLE = _TAB             # the table of InsterSpeedFeedforward
FF_TABLE_EXIT = _TAB_EXIT   # the table with InsterExitTuning


def table_torque(a_abs: float, v_ego: float, tab=_TAB) -> float:
  """|a| (m/s^2, >= 0) and speed (m/s) -> torque fraction 0..1. Bilinear: linear in v between the rows, then linear in a."""
  vn = V_NODES
  if v_ego <= vn[0]:
    j, w = 0, 0.0
  elif v_ego >= vn[-1]:
    j, w = len(vn) - 2, 1.0
  else:
    j = 0
    while v_ego >= vn[j + 1]:
      j += 1
    w = (v_ego - vn[j]) / (vn[j + 1] - vn[j])
  t0, t1 = tab[j], tab[j + 1]
  an = A_NODES
  if a_abs >= an[-1]:
    return (1.0 - w) * t0[-1] + w * t1[-1]
  if a_abs <= 0.0:
    return 0.0
  i = 0
  while a_abs >= an[i + 1]:
    i += 1
  c0 = (1.0 - w) * t0[i] + w * t1[i]
  c1 = (1.0 - w) * t0[i + 1] + w * t1[i + 1]
  return c0 + (c1 - c0) * (a_abs - an[i]) / (an[i + 1] - an[i])


def ff_torque(lateral_accel: float, v_ego: float, lat_accel_factor: float, tab=_TAB) -> float:
  """Feed-forward torque (fraction of full torque, signed like lateral_accel) for a gravity-adjusted lateral acceleration (m/s^2).
  The learned factor scales the table only within SCALE_CLIP around F0. tab: _TAB, or _TAB_EXIT with the exit tuning."""
  scale = F0 / (lat_accel_factor if lat_accel_factor > 0.1 else 0.1)
  scale = SCALE_CLIP[0] if scale < SCALE_CLIP[0] else (SCALE_CLIP[1] if scale > SCALE_CLIP[1] else scale)
  if lateral_accel > 0.0:
    return table_torque(lateral_accel * scale, v_ego, tab)
  if lateral_accel < 0.0:
    return -table_torque(-lateral_accel * scale, v_ego, tab)
  return 0.0


def ff_lat_accel_equivalent(lateral_accel: float, v_ego: float, lat_accel_factor: float, tab=_TAB) -> float:
  """The table's feed-forward in v0's lateral-acceleration units (factor x torque): v0's linear torque_from_lateral_accel (/ factor)
  then returns exactly the table torque for the feed-forward part."""
  return lat_accel_factor * ff_torque(lateral_accel, v_ego, lat_accel_factor, tab)


# ---------------------------------------------------------------------------------------------------------------------------------
# Stiction friction term (step 1 closed-loop set F_a15_b65_w4p5_lo9, "lo9"; amplitudes from 1,124 logged stand-then-move events).
FRICTION_A = 15.0                  # raw: amplitude at zero held torque
FRICTION_B_IN = 0.1                # raw per raw of held torque, when pushing further in the direction already held
FRICTION_B_OUT = 0.65              # raw per raw of held torque, when releasing or unwinding (plan-led exits need ~8 + 0.67 x held, LOG)
FRICTION_MAX = 200.0               # raw: amplitude cap
HELD_H0 = 10.0                     # raw: smooth |held torque|: hs = sqrt(h^2 + h0^2) - h0
HELD_TAU = 0.3                     # s: low-pass of the controller's own output torque (the held torque h)
SHAPE_W = 4.5                      # deg/s: width of the smooth sign at >= 50 km/h ...
SHAPE_W_LOW = 9.0                  # deg/s: ... widening to this at <= 30 km/h (linear in between)
SHAPE_W_BP = (8.33, 13.89)         # m/s: 30 and 50 km/h
ERROR_TAU = 1.0                    # s: the steering-wheel angle error counts as a wheel rate of error / ERROR_TAU
INTENT_TAU = 0.05                  # s: low-pass of the intended wheel rate x
FADE_BETA = 0.5                    # share of the amplitude that fades once the wheel moves
FADE_RATE = 10.0                   # deg/s: |steering rate| at which the fading share is halved
FADE_TAU = 0.1                     # s: low-pass of |steering rate|

# Exit tuning (switch InsterExitTuning; simulator candidate r66_all_c, 22 Oct 2026): replaces these values of the term above
EXIT_SHAPE_W = 3.0                 # deg/s: smooth-sign width at >= 40 km/h ...
EXIT_SHAPE_W_LOW = 6.0             # deg/s: ... widening to this at <= 20 km/h
EXIT_SHAPE_W_BP = (5.56, 11.11)    # m/s: 20 and 40 km/h
EXIT_INTENT_TAU = 0.02             # s: low-pass of the intended wheel rate
EXIT_RELEASE_KD = 2.0              # raw per deg/s: damping against (wheel rate - plan wheel rate), only while the term releases
WHEEL_RATE_FRAMES = 4              # signed wheel rate for the damping: steering-angle change over 4 steps (40 ms), as the simulator;
                                   # carState.steeringRateDeg is unsigned on the Inster


class InsterStictionFriction:
  """Stiction friction term for v0, one update() per controlsd step (dt). Returns raw command units, + = right (v0-internal); v0 adds
  it to its feed-forward as term / STEER_MAX_RAW x latAccelFactor, so the torque it adds does not depend on the learned factor.

    x  = LP_INTENT_TAU( (theta_future - theta_now) / lat_delay + (theta_now - theta_measured) / ERROR_TAU )   intended wheel rate, deg/s
    s  = sign(x) (1 - exp(-(x / w)^2))                  smooth sign: value and slope 0 at x = 0, no dead zone, |s| < 1
    h  = LP_HELD_TAU(own last output, raw);  hs = sqrt(h^2 + h0^2) - h0
    F  = min(FRICTION_MAX, FRICTION_A + (FRICTION_B_IN if x h > 0 else FRICTION_B_OUT) hs)
    g  = 1 / (1 + (LP_FADE_TAU(|steering rate|) / FADE_RATE)^2)
    term = F s ((1 - FADE_BETA) + FADE_BETA g)
  theta_*: steering-wheel angles (deg, + right) of the curvature planned for t + lat_delay (the desired curvature now), of the
  curvature requested lat_delay ago (the plan for now) and of the measured curvature (roll-compensated, like v0's measurement).
  No latches or switching states: only first-order low-pass filters; F switches between its in / out slopes only where s and its
  slope are 0 (x = 0) or where both slopes give the same value with zero slope (h = 0). reset() clears the filters; the first update
  after it starts each filter at its input (settled), as the closed-loop simulator does at the start of each active stretch.

  exit_tuning=True (switch InsterExitTuning): the EXIT_* values instead of SHAPE_W* and INTENT_TAU, and while the term releases
  (x h <= 0) it adds  - EXIT_RELEASE_KD (wheel_rate - (theta_future - theta_now) / lat_delay),  wheel_rate = the signed wheel rate
  (deg/s, + right) passed to update(). Same order of operations as the simulator's candidate (CLExt / r66_all_c)."""

  def __init__(self, dt: float, exit_tuning: bool = False):
    self.dt = dt
    self.exit_tuning = exit_tuning
    if exit_tuning:
      self.w_hi, self.w_lo, self.w_bp, intent_tau, self.kd = EXIT_SHAPE_W, EXIT_SHAPE_W_LOW, EXIT_SHAPE_W_BP, EXIT_INTENT_TAU, EXIT_RELEASE_KD
    else:
      self.w_hi, self.w_lo, self.w_bp, intent_tau, self.kd = SHAPE_W, SHAPE_W_LOW, SHAPE_W_BP, INTENT_TAU, 0.0
    self.alpha_x = dt / (intent_tau + dt)
    self.alpha_h = dt / (HELD_TAU + dt)
    self.alpha_om = dt / (FADE_TAU + dt)
    self.inv_tau_e = 1.0 / ERROR_TAU
    self.h0_sq = HELD_H0 * HELD_H0
    self.reset()

  def reset(self) -> None:
    self.initialized = False
    self.x = 0.0     # low-passed intended wheel rate (deg/s)
    self.h = 0.0     # low-passed held torque (raw)
    self.om = 0.0    # low-passed |steering rate| (deg/s)

  def update(self, v_ego: float, theta_future: float, theta_now: float, theta_measured: float, lat_delay: float,
             steer_rate_abs: float, last_output_raw: float, wheel_rate: float = 0.0) -> float:
    plan_rate = (theta_future - theta_now) / (lat_delay if lat_delay > self.dt else self.dt)
    x_raw = plan_rate + (theta_now - theta_measured) * self.inv_tau_e
    rate = abs(steer_rate_abs)
    if not self.initialized:
      self.x, self.h, self.om = x_raw, last_output_raw, rate
      self.initialized = True

    self.x += self.alpha_x * (x_raw - self.x)
    x = self.x
    if v_ego >= self.w_bp[1]:
      w = self.w_hi
    elif v_ego <= self.w_bp[0]:
      w = self.w_lo
    else:
      w = self.w_lo + (self.w_hi - self.w_lo) * (v_ego - self.w_bp[0]) / (self.w_bp[1] - self.w_bp[0])
    if x == 0.0:
      s = 0.0
    else:
      q = x / w
      s = 1.0 - math.exp(-q * q)
      if x < 0.0:
        s = -s

    self.h += self.alpha_h * (last_output_raw - self.h)
    h = self.h
    hs = math.sqrt(h * h + self.h0_sq) - HELD_H0
    amplitude = FRICTION_A + (FRICTION_B_IN if x * h > 0.0 else FRICTION_B_OUT) * hs
    if amplitude > FRICTION_MAX:
      amplitude = FRICTION_MAX

    self.om += self.alpha_om * (rate - self.om)
    g = self.om / FADE_RATE
    g = 1.0 / (1.0 + g * g)
    term = amplitude * s * ((1.0 - FADE_BETA) + FADE_BETA * g)
    if self.kd != 0.0 and not (x * h > 0.0):
      # exit tuning: release-only damping of the wheel motion relative to the plan's wheel rate
      term += -self.kd * (wheel_rate - plan_rate)
    return term
