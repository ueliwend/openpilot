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
  * Steering-angle PI "F" (switch InsterAngleErrorPI): InsterAnglePI replaces only v0's P and I. They act on the steering-wheel
    angle error (rad) with gains that are quadratic in speed, and the integrator runs from standstill. Feed-forward, limits and output
    conversion stay v0's. Tuned and checked in the closed-loop simulator only with both switches above on.

  * Torque cap (no switch): the Inster's steering command is capped at 409 raw (opendbc values.INSTER_STEER_MAX, the highest the
    owner's steering unit took without a fault, 23 Oct 2026) with 1.0 = 384 raw (INSTER_STEER_FULL_SCALE): every lateral
    controller's output limit is 409 / 384 for the Inster (lateral_steer_max, used by LatControl), the car side caps at 409.
  * Request ceiling (no switch): max_lateral_accel_no_roll, 3.3 m/s^2 for the Inster instead of openpilot's 3.0.

Units and signs: commands in raw LKAS11 units (full scale STEER_MAX_RAW = 384 = actuators.torque 1.0) or as a fraction of full torque;
v0's internal sign convention (+ = right turn), the convention of v0's setpoint, measurement and output before its final negation.
Derived offline from the owner's logs (step 1, 21 Oct 2026; open-loop replays and the closed-loop steering simulator).
Nothing here is validated on the car.
"""
import math

from opendbc.car.hyundai.values import CAR, INSTER_STEER_FULL_SCALE, INSTER_STEER_MAX
from openpilot.common.pid import PIDController

# Full-scale command the table and the friction term are expressed in (actuators.torque 1.0; opendbc INSTER_STEER_FULL_SCALE).
# If it changes, F0 and the raw amplitudes below need a review: test_inster_lateral.py checks that both agree.
STEER_MAX_RAW = 384.0

# The Inster's switches (params, off by default, changeable only offroad, ignored while NNLC is enabled). With any of them on, the
# Lateral Jerk Torque Controller is switched off.
INSTER_SWITCHES = ("InsterStictionFriction", "InsterSpeedFeedforward", "InsterAngleErrorPI")

# Steering torque cap as a fraction of full scale: 409 / 384 = 1.065 (opendbc values.py: INSTER_STEER_MAX / INSTER_STEER_FULL_SCALE)
INSTER_STEER_MAX_FRACTION = INSTER_STEER_MAX / INSTER_STEER_FULL_SCALE
# Highest planned lateral acceleration without roll (m/s^2) for the Inster, instead of openpilot's 3.0 (drive_helpers
# MAX_LATERAL_ACCEL_NO_ROLL); used by controlsd's clip_curvature. Owner's choice for the higher torque; route 0000007f at 409: held
# 3.1-3.45 at 40-60 km/h without the command limiting (LOG), estimated 3.13-3.26 above 60 km/h (EST, unverified).
INSTER_MAX_LATERAL_ACCEL_NO_ROLL = 3.3


def is_inster(CP) -> bool:
  """True for the Hyundai Inster (Casper Electric) fingerprint."""
  return CP is not None and CP.carFingerprint == CAR.HYUNDAI_INSTER


def any_inster_switch(params) -> bool:
  return any(params.get_bool(k) for k in INSTER_SWITCHES)


def lateral_steer_max(CP) -> float:
  """Output limit of every lateral controller (fraction of full torque; LatControl.steer_max): 409 / 384 for the Inster, 1.0 else."""
  return INSTER_STEER_MAX_FRACTION if is_inster(CP) else 1.0


def max_lateral_accel_no_roll(CP, default: float) -> float:
  """Highest planned lateral acceleration without roll (m/s^2): INSTER_MAX_LATERAL_ACCEL_NO_ROLL for the Inster, else the default."""
  return INSTER_MAX_LATERAL_ACCEL_NO_ROLL if is_inster(CP) else default


# ---------------------------------------------------------------------------------------------------------------------------------
# Speed feed-forward table (step 1 curve study, candidate "b-ff"; LOG: ten drives 14-21 Oct 2026).
# Steering command (raw) needed for a gravity-adjusted lateral acceleration |a| (m/s^2) at speed v. Odd in a, bilinear (linear between
# the nodes in v, then in a), clamped at the outer speed rows (22 km/h below, 111 km/h above) and at 3.5 m/s^2.
A_NODES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5)       # m/s^2
V_NODES = tuple(x / 3.6 for x in (22.0, 32.0, 48.0, 76.0, 111.0))          # m/s
CMD_RAW = (                                                                # rows: V_NODES, columns: A_NODES
  (0.0, 23.0, 57.1, 142.4, 250.6, 303.8, 352.2, 409.0, 409.0, 409.0, 409.0),   # 22 km/h  (measured to 1.25 m/s^2)
  (0.0, 23.0, 48.0, 108.6, 146.2, 190.6, 242.5, 314.7, 387.2, 409.0, 409.0),   # 32 km/h  (measured to 2.5)
  (0.0, 21.1, 48.0, 95.4, 116.1, 141.7, 168.6, 267.9, 328.0, 396.8, 409.0),    # 48 km/h  (measured to 3.0)
  (0.0, 19.8, 39.0, 67.9, 97.9, 130.3, 168.6, 267.9, 328.0, 396.8, 409.0),     # 76 km/h  (measured to 1.5)
  (0.0, 19.8, 38.8, 57.7, 85.2, 115.8, 154.0, 251.0, 318.6, 389.9, 409.0),     # 111 km/h (measured to 1.0)
)
# 23 Oct 2026: the 13 cells that were pinned at the old cap of 384 now run up to the new cap of 409 (torque-map extension of the owner's
# route 0000007f at the 409 cap; agents/r7f_tmap/ff_table_409.json); every cell <= INSTER_STEER_MAX (tested). In raw units of the
# full scale 384, so the 409 cells are 409 / 384 = 1.065 of full torque.
F0 = 2.8                          # latAccelFactor at which the table is used as measured (the learner's value when it was built)
SCALE_CLIP = (1.0 / 1.15, 1.15)   # bound on how far the learned factor can scale the table (the table is read at |a| * F0 / factor)
_TAB = tuple(tuple(c / STEER_MAX_RAW for c in row) for row in CMD_RAW)


def table_torque(a_abs: float, v_ego: float) -> float:
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
  t0, t1 = _TAB[j], _TAB[j + 1]
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


def ff_torque(lateral_accel: float, v_ego: float, lat_accel_factor: float) -> float:
  """Feed-forward torque (fraction of full torque, signed like lateral_accel) for a gravity-adjusted lateral acceleration (m/s^2).
  The learned factor scales the table only within SCALE_CLIP around F0."""
  scale = F0 / (lat_accel_factor if lat_accel_factor > 0.1 else 0.1)
  scale = SCALE_CLIP[0] if scale < SCALE_CLIP[0] else (SCALE_CLIP[1] if scale > SCALE_CLIP[1] else scale)
  if lateral_accel > 0.0:
    return table_torque(lateral_accel * scale, v_ego)
  if lateral_accel < 0.0:
    return -table_torque(-lateral_accel * scale, v_ego)
  return 0.0


def ff_lat_accel_equivalent(lateral_accel: float, v_ego: float, lat_accel_factor: float) -> float:
  """The table's feed-forward in v0's lateral-acceleration units (factor x torque): v0's linear torque_from_lateral_accel (/ factor)
  then returns exactly the table torque for the feed-forward part."""
  return lat_accel_factor * ff_torque(lateral_accel, v_ego, lat_accel_factor)


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
  after it starts each filter at its input (settled), as the closed-loop simulator does at the start of each active stretch."""

  def __init__(self, dt: float):
    self.dt = dt
    self.alpha_x = dt / (INTENT_TAU + dt)
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
             steer_rate_abs: float, last_output_raw: float) -> float:
    x_raw = (theta_future - theta_now) / (lat_delay if lat_delay > self.dt else self.dt) + (theta_now - theta_measured) * self.inv_tau_e
    rate = abs(steer_rate_abs)
    if not self.initialized:
      self.x, self.h, self.om = x_raw, last_output_raw, rate
      self.initialized = True

    self.x += self.alpha_x * (x_raw - self.x)
    x = self.x
    if v_ego >= SHAPE_W_BP[1]:
      w = SHAPE_W
    elif v_ego <= SHAPE_W_BP[0]:
      w = SHAPE_W_LOW
    else:
      w = SHAPE_W_LOW + (SHAPE_W - SHAPE_W_LOW) * (v_ego - SHAPE_W_BP[0]) / (SHAPE_W_BP[1] - SHAPE_W_BP[0])
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
    return amplitude * s * ((1.0 - FADE_BETA) + FADE_BETA * g)


# ---------------------------------------------------------------------------------------------------------------------------------
# Steering-angle PI "F" (switch InsterAngleErrorPI; the owner's 6-parameter controller, set "recommended_F" of the 23 Oct 2026 search in
# the closed-loop steering simulator, SIM only: 9 drives x 8 plant variants x 2 seeds, on top of v0 with both switches above at 409).
# It replaces only v0's P and I:
#   e  = delta_plan - delta_meas   steering-wheel angle error, rad, + right (v0-internal sign)
#        delta_plan = wheel angle the plan's curvature needs (inverse vehicle model: steer ratio, slip factor, roll),
#        delta_meas = steeringAngleDeg - learned angleOffsetDeg.
#        The vehicle model is linear in the angle, so e = R(v) x (desired curvature - measured curvature) with
#        R(v) = VM.get_steer_from_curvature(1.0, v, 0.0) (rad per 1/m) and v0's roll-compensated measured curvature (angle_error()).
#   p  = kp(v) e,   i += ki(v) e dt   (v = vEgo in m/s; p and i in v0's lateral-acceleration units, m/s^2)
#   kp(v) = 4.3447 + 0.40366 v - 0.0072954 v^2   (m/s^2 per rad)       5.9 / 8.81 / 9.80 at 15 / 55 / 115 km/h
#   ki(v) = 0.93525 + 0.27948 v - 0.0057456 v^2  (m/s^2 per rad s)     2.0 / 3.86 / 4.00 at 15 / 55 / 115 km/h
# Both clipped at >= 0 (they reach 0 only above 186 km/h). No speed hold above 130 km/h (owner's choice): kp 7.9 / ki 2.0 at 160 km/h.
# The integrator runs from standstill; it freezes only while steeringPressed or steer-limited by safety (v0's flag, which also covers
# the 3 / 5 raw ramp), and it is reset whenever lateral control is inactive (v0 keeps its own integrator, but F's can be wound at low
# speed, which v0's cannot: without the reset it would come back as a command towards the old curve on re-engagement).
# Anti-windup and output clip are common/pid.py's (InsterAnglePI is a PIDController with these gains).
# Logs (controlsState.lateralControlState.torqueState) with F on: version = ANGLE_PI_LOG_VERSION (v0 logs 0, the upstream controller
# 1); error stays v0's lateral-acceleration error (setpoint - measurement, m/s^2), p / i / f are in m/s^2 as in v0. F's own error is
# e = p / kp(vEgo) (rad, + right; angle_pi_gain(ANGLE_PI_KP, vEgo) > 0 at every speed the car drives), or recomputed from carState
# steeringAngleDeg, liveParameters angleOffsetDeg / roll / steerRatio / stiffnessFactor and controlsState desiredCurvature with
# angle_error().
ANGLE_PI_KP = (4.3447, 0.40366, -0.0072954)
ANGLE_PI_KI = (0.93525, 0.27948, -0.0057456)
ANGLE_PI_LOG_VERSION = 100


def angle_pi_gain(coeffs: tuple[float, float, float], v_ego: float) -> float:
  """quadratic speed gain c0 + c1 v + c2 v^2, clipped at >= 0"""
  g = coeffs[0] + coeffs[1] * v_ego + coeffs[2] * v_ego * v_ego
  return g if g > 0.0 else 0.0


def angle_error(VM, v_ego: float, desired_curvature: float, measured_curvature: float) -> float:
  """steering-wheel angle error (rad, + right as v0's curvatures): the plan's wheel angle minus the measured one (minus the learned
  offset). R(v) x (desired - measured curvature); exact for the linear vehicle model when measured_curvature is v0's roll-compensated
  -VM.calc_curvature(radians(steeringAngleDeg - angleOffsetDeg), v, roll)."""
  return VM.get_steer_from_curvature(1.0, v_ego, 0.0) * (desired_curvature - measured_curvature)


class InsterAnglePI(PIDController):
  """common/pid.py with the speed gains of "F" (kp / ki above; kd 0). update(e, ...) takes the wheel-angle error in rad; p, i, f and
  the limits are in v0's lateral-acceleration units."""

  def __init__(self, rate: float):
    super().__init__(0.0, 0.0, 0.0, rate=rate)

  @property
  def k_p(self):
    return angle_pi_gain(ANGLE_PI_KP, self.speed)

  @property
  def k_i(self):
    return angle_pi_gain(ANGLE_PI_KI, self.speed)
