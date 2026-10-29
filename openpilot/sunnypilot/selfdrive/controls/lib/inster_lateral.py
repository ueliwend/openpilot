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
    Controller type (param InsterLateralController, effective only with InsterAngleErrorPI on): PI (default; gains G6) or PID
    (gains P6 = PI + D on the measured wheel-angle rate). Option 5 (InsterSoftSmallErrorP, with InsterAngleErrorPI on, either type;
    fitted for the PID): P scaled by sigma, softer for small errors on straights, stronger in curves (soft_p_sigma).

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
# Controller-type selector of InsterAngleErrorPI (same rules; alone it changes nothing, so it is not one of INSTER_SWITCHES):
# 0 = PI (default; gains G6), 1 = PID (gains P6). One option per controller type (owner, 29 Oct 2026); gain set G2 is no longer
# selectable. It replaces the gain-set selector InsterAngleErrorPIGains (28 Oct 2026: 0 = G2, 1 = G6, 2 = P6), which in turn replaced
# the bool InsterAngleErrorPIG6 (27 Oct 2026, G6 only). params_migration turns both into this one (G2 / G6 -> PI, P6 -> PID) and
# removes them; until then angle_lateral_controller() reads the gain-set selector as a fallback (the G6 bool means PI either way).
LATERAL_CONTROLLER_PARAM = "InsterLateralController"
LATERAL_PI, LATERAL_PID = 0, 1
ANGLE_PI_GAINS_PARAM = "InsterAngleErrorPIGains"   # legacy selector, read only by the migration and the fallback
ANGLE_PI_GAINS_LEGACY_P6 = 2                       # its value for P6 (0 = G2 and 1 = G6 both mean PI now)
ANGLE_PI_G6_SWITCH = "InsterAngleErrorPIG6"        # legacy bool, read only by the migration
# Option 5 of InsterAngleErrorPI (same rules; alone it changes nothing, so it is not one of INSTER_SWITCHES): soft small-error P
SOFT_P_PARAM = "InsterSoftSmallErrorP"

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
  (0.0, 23.0, 48.0, 108.6, 136.8, 190.6, 242.5, 314.7, 387.2, 409.0, 409.0),   # 32 km/h  (measured to 2.5)
  (0.0, 21.1, 48.0, 95.4, 121.8, 150.6, 183.0, 267.9, 328.0, 396.8, 409.0),    # 48 km/h  (measured to 3.0)
  (0.0, 19.8, 39.0, 67.9, 97.9, 130.3, 168.6, 267.9, 328.0, 396.8, 409.0),     # 76 km/h  (measured to 1.5)
  (0.0, 19.8, 38.8, 57.7, 85.2, 115.8, 154.0, 251.0, 318.6, 389.9, 409.0),     # 111 km/h (measured to 1.0)
)
# 26 Oct 2026, twenty-drive refit after route 000000a1: 32 km/h @ 1.0 m/s^2 146.2 -> 136.8; 48 km/h @ 1.0 / 1.25 / 1.5 m/s^2
# 116.1 / 141.7 / 168.6 -> 121.8 / 150.6 / 183.0. Change 1.3-1.8 sigma by the refit's block bootstrap, about 1 sigma by leave-one-drive-out;
# the 48 km/h move comes from that route alone (the refit without it equals the table). F0 and the clip unchanged: the filtered learned
# factor that controlsd passes ranged 2.68-3.06 since 22 Oct (scale F0 / factor 0.914-1.045), inside the clip.
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
# Since 29 Oct 2026 it runs one of two controller types in place of F's gains: PI (gains G6) or PID (gains P6), both below. It
# replaces only v0's P and I (and adds D with the PID):
#   e  = delta_plan - delta_meas   steering-wheel angle error, rad, + right (v0-internal sign)
#        delta_plan = wheel angle the plan's curvature needs (inverse vehicle model: steer ratio, slip factor, roll),
#        delta_meas = steeringAngleDeg - learned angleOffsetDeg.
#        The vehicle model is linear in the angle, so e = R(v) x (desired curvature - measured curvature) with
#        R(v) = VM.get_steer_from_curvature(1.0, v, 0.0) (rad per 1/m) and v0's roll-compensated measured curvature (angle_error()).
#   p  = kp(v) e,   i += ki(v) e dt   (v = vEgo in m/s; p and i in v0's lateral-acceleration units, m/s^2; kp, ki quadratic in v,
#   clipped at >= 0, no speed hold above 130 km/h (owner's choice))
# The integrator runs from standstill; it freezes only while steeringPressed or steer-limited by safety (v0's flag, which also covers
# the 3 / 5 raw ramp), and it is reset whenever lateral control is inactive (v0 keeps its own integrator, but F's can be wound at low
# speed, which v0's cannot: without the reset it would come back as a command towards the old curve on re-engagement).
# Anti-windup and output clip are common/pid.py's (InsterAnglePI is a PIDController with these gains).
# Logs (controlsState.lateralControlState.torqueState) with F on: version = ANGLE_PI_G6_LOG_VERSION (PI) or ANGLE_PI_P6_LOG_VERSION
# (PID) (v0 logs 0, the upstream controller 1; older drives: 100 / 101 F, 102 G2); error stays v0's lateral-acceleration error (setpoint -
# measurement, m/s^2), p / i / d / f are in m/s^2 as in v0 (d is 0 except with the PID). F's own error is e = p / kp(vEgo) (rad, + right;
# kp of the gain set in force, > 0 at every speed the car drives), or recomputed from carState steeringAngleDeg, liveParameters
# angleOffsetDeg / roll / steerRatio / stiffnessFactor and controlsState desiredCurvature with angle_error() (with option 5, version
# bit 8: p / kp = sigma e; use angle_error()).
# History, for reading older logs (no longer selectable since 29 Oct 2026): version 100 = F (kp 4.3447 + 0.40366 v - 0.0072954 v^2,
# ki 0.93525 + 0.27948 v - 0.0057456 v^2) until 26 Oct 2026, 101 = F with controlsd's crawl plan (crawl_plan_curvature), 102 = gains G2
# (kp 3.17477 + 0.606402 v - 0.0119531 v^2, ki 0.93525 + 0.329782 v - 0.00677043 v^2; Nelder-Mead group 2, objective v5), 8102 = G2
# with option 5.
# Gains G6 = the PI (27 Oct 2026, SIM only; controller type PI): Nelder-Mead group 2, objective v6 (owner 27 Oct: no
# 'getting worse' penalties), project doc claude/inster-steering-simulator.md. Less P and more I than G2 at 10-130 km/h. The search's
# coefficients rounded to 6 significant digits (gains within 6e-6 relative up to 160 km/h). Clipped at >= 0 as G2: ki reaches 0
# above 176.7 km/h, kp only above 233 km/h. kp 3.42 / 4.50 / 8.02 / 9.24 / 7.54, ki 0.94 / 2.10 / 5.54 / 5.52 / 2.04 at
# 0 / 10 / 55 / 115 / 160 km/h.
ANGLE_PI_G6_KP = (3.4164, 0.410763, -0.00715263)
ANGLE_PI_G6_KI = (0.93525, 0.44623, -0.00948069)
ANGLE_PI_G6_LOG_VERSION = 103
# Gains P6 = the PID (27 Oct 2026, SIM only; controller type PID): PID candidate of the Nelder-Mead group-2 search, objective v6 (project doc
# claude/inster-steering-simulator.md). P and I as G2 / G6, plus D on the measured wheel angle (rad, + right; v0's roll-compensated
# measured curvature x R(v), the quantity e is built from; = -radians(steeringAngleDeg - angleOffsetDeg) minus a roll-compensation
# term that does not depend on speed): d = -kd(v) x d/dt LP2(theta_meas), LP2 = two equal first-order stages with a = 1 - exp(-dt / tau), tau = 0.03 s, states
# set to the current value on the first active step after lateral was inactive (no kick). D is in common/pid.py's anti-windup test
# and output clip, as P and I. As simulated (CarPID, d_mode 'meas'; its plan filter tau_p 0.1 s only acts in d_mode 'split'), with
# one deviation: D = 0 while steeringPressed (not simulated; it would otherwise resist the driver's hand, about 90 raw at 100 deg/s).
# Coefficients rounded to 6 significant digits (gains within 2e-6 relative up to 160 km/h). Clipped at >= 0: ki reaches 0 above
# 178.6 km/h, kp above 214 km/h, kd stays > 0. kp 3.33 / 4.66 / 8.86 / 9.86 / 7.18, ki 0.94 / 2.08 / 5.46 / 5.51 / 2.20,
# kd 0.362 / 0.364 / 0.373 / 0.380 / 0.383 (m/s^2 per rad/s) at 0 / 10 / 55 / 115 / 160 km/h.
ANGLE_PI_P6_KP = (3.33059, 0.505828, -0.00943131)
ANGLE_PI_P6_KI = (0.93525, 0.436114, -0.00917132)
ANGLE_PI_P6_KD = (0.361727, 0.000854274, -8.68703e-06)
ANGLE_PI_D_TAU = 0.03
# torqueState.version: gain-set base + 1000 x mask, D = bit 16 (the options design's scheme): P6 = 104 + 16000 = 16104
ANGLE_PI_P6_BASE_VERSION = 104
ANGLE_PI_VERSION_D_BIT = 16
ANGLE_PI_P6_LOG_VERSION = ANGLE_PI_P6_BASE_VERSION + 1000 * ANGLE_PI_VERSION_D_BIT
# Option 5 "InsterSoftSmallErrorP" (28 Oct 2026, SIM only; options design v2.1 section 5, project doc
# claude/inster-steering-simulator.md): p = kp(v) sigma e, I and D unchanged (no sigma on I), with
#   v_c = max(v, 7 m/s), e_a = |e| v_c^2 / R(v) (lateral-acceleration size of the raw wheel-angle error, R in rad per 1/m, >= 1 deg),
#   a_p = max(|desired curvature|, |curvature requested lat_delay ago|) v_c^2 (the stiction term's two plan inputs), phi(x) = x^2 / (1 + x^2),
#   sigma = [1 - (1 - rho)(1 - phi(e_a / e0))(1 - phi(a_p / a0))] (1 + (gamma - 1) phi(a_p / a0)),   rho <= sigma <= gamma.
# p(e) stays strictly increasing (slope rho kp at e = 0 on a straight). Mild parameters fitted with P6 (Nelder-Mead option fit,
# objective v6; the bound-limited full fit was not taken): rho 0.5, e0 0.11 m/s^2, a0 0.3 m/s^2, gamma 1.2. As the kit's
# candidates_opts.OptsPID option 'o5'. torqueState.version = gain-set base + 1000 x (D bit 16 | option-5 bit 8): PI 8103, PID 24104
# (angle_pi_log_version). Logs: nothing new; sigma = p / (kp(vEgo) e), or recomputed from the logged inputs.
SOFT_P_RHO = 0.5
SOFT_P_E0 = 0.11
SOFT_P_A0 = 0.3
SOFT_P_GAMMA = 1.2
SOFT_P_V_MIN = 7.0
SOFT_P_R_MIN = math.radians(1.0)
ANGLE_PI_VERSION_SOFT_P_BIT = 8


def angle_pi_log_version(controller: int, soft_p: bool) -> int:
  """torqueState.version with F on: the controller type's base + 1000 x mask (D bit with the PID, option-5 bit)"""
  if controller == LATERAL_PID:
    base, mask = ANGLE_PI_P6_BASE_VERSION, ANGLE_PI_VERSION_D_BIT
  else:
    base, mask = ANGLE_PI_G6_LOG_VERSION, 0
  if soft_p:
    mask |= ANGLE_PI_VERSION_SOFT_P_BIT
  return base + 1000 * mask


def soft_p_sigma(e: float, v_ego: float, r_rad: float, curvature_now: float, curvature_past: float) -> float:
  """option 5's P multiplier sigma (see above) for the raw wheel-angle error e (rad) and R(v) = r_rad (rad per 1/m)"""
  vc = v_ego if v_ego > SOFT_P_V_MIN else SOFT_P_V_MIN
  rr = r_rad if r_rad > SOFT_P_R_MIN else SOFT_P_R_MIN
  xe = abs(e) * vc * vc / rr / SOFT_P_E0
  xa = max(abs(curvature_now), abs(curvature_past)) * vc * vc / SOFT_P_A0
  pe = xe * xe / (1.0 + xe * xe)
  pa = xa * xa / (1.0 + xa * xa)
  return (1.0 - (1.0 - SOFT_P_RHO) * (1.0 - pe) * (1.0 - pa)) * (1.0 + (SOFT_P_GAMMA - 1.0) * pa)


def angle_lateral_controller(params) -> int:
  """the selected controller type (LATERAL_PI / LATERAL_PID); unset: the legacy gain-set selector (before the migration ran;
  P6 -> PID, anything else -> PI); unknown values: PI"""
  sel = params.get(LATERAL_CONTROLLER_PARAM)
  if sel is None:
    return LATERAL_PID if params.get(ANGLE_PI_GAINS_PARAM) == ANGLE_PI_GAINS_LEGACY_P6 else LATERAL_PI
  return sel if sel in (LATERAL_PI, LATERAL_PID) else LATERAL_PI


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
  """common/pid.py with the speed gains of "F" (PI: G6, PID: P6, kp / ki / kd above; kd 0 unless given). update(e, rate, ...)
  takes the wheel-angle error in rad (and, with kd, the D input in rad/s); p, i, d, f and the limits are in v0's lateral-acceleration
  units."""

  def __init__(self, rate: float, kp: tuple[float, float, float], ki: tuple[float, float, float],
               kd: tuple[float, float, float] | None = None):
    super().__init__(0.0, 0.0, 0.0, rate=rate)
    self.kp_coeffs = kp
    self.ki_coeffs = ki
    self.kd_coeffs = kd
    self.p_sigma: float | None = None     # option 5: P multiplier of the current step (None: off)

  @property
  def k_p(self):
    kp = angle_pi_gain(self.kp_coeffs, self.speed)
    return kp if self.p_sigma is None else kp * self.p_sigma

  @property
  def k_i(self):
    return angle_pi_gain(self.ki_coeffs, self.speed)

  @property
  def k_d(self):
    return super().k_d if self.kd_coeffs is None else angle_pi_gain(self.kd_coeffs, self.speed)


class InsterWheelRateFilter:
  """P6's D input: d/dt of the measured wheel angle through LP2 (two equal first-order stages, a = 1 - exp(-dt / tau)), rad/s. The
  states start at the first value after reset(), so the first rate is 0."""

  def __init__(self, dt: float, tau: float = ANGLE_PI_D_TAU):
    self.dt = dt
    self.a = 1.0 - math.exp(-dt / tau)
    self.reset()

  def reset(self) -> None:
    self.initialized = False
    self.x1 = 0.0
    self.x2 = 0.0

  def update(self, theta: float) -> float:
    if not self.initialized:
      self.x1 = self.x2 = theta
      self.initialized = True
    prev = self.x2
    self.x1 += self.a * (theta - self.x1)
    self.x2 += self.a * (self.x1 - self.x2)
    return (self.x2 - prev) / self.dt


# ---------------------------------------------------------------------------------------------------------------------------------
# Plan at crawl speed (with "F" only; controlsd, crawl_plan_curvature). Below ~1 m/s the model's desired curvature collapses towards
# straight although the turn goes on (LOG, owner's drives 0000007[26f]/8[78]/9[7b], 26 Oct 2026):
#   * modeld converts the planned heading to curvature with v floored at 1 m/s (drive_helpers.curv_from_psis, CODE): the plan shrinks
#     by v / 1 m/s below 1 m/s (logged action.desiredCurvature reproduced from the logged plan to 1e-10);
#   * the plan's own heading change also straightens on the approach to a stop: without the floor it still gives only 0.2-0.6 of the
#     curvature the car was in at 1 m/s, while the model's current yaw rate / v still matches it (model lookahead unreliable there).
# At standstill lateral is inactive and the plan is the measured curvature (stock). In between, "F" pulled the wheel towards straight.
# Rule: a road's curvature changes with distance travelled, not with time. Below CRAWL_PLAN_SPEED the plan moves from the previous plan
# towards the model's by at most c x vEgo x dt (c in 1/m per metre travelled):
#   * towards straight (|curvature| down, up to zero): c = CRAWL_STRAIGHTEN_PER_M + |previous plan| / CRAWL_STRAIGHTEN_LENGTH
#     (a constant rate plus a proportional relaxation over CRAWL_STRAIGHTEN_LENGTH metres, which bounds a hold of a tight curve);
#   * away from straight (further into the turn, from zero, or past zero to the other side): c = CRAWL_TURN_IN_PER_M;
#   both open linearly from CRAWL_PLAN_RELEASE_SPEED to at least CRAWL_PER_M_RELEASE at CRAWL_PLAN_SPEED; from CRAWL_PLAN_SPEED up the
#   plan is the model's (a gap left by a hard acceleration through 1.4 m/s is closed by clip_curvature's jerk limit, 0.0255 1/m per step).
# Worst cases (100 Hz with clip_curvature, extra heading / lateral offset vs following the model): creep at 0.5 m/s out of a curve
# whose plan goes straight at once: 0.1 1/m (204 deg wheel) 4.0 deg / 0.08 m, 0.2 1/m (full lock) 11.4 deg / 0.35 m; S-bend at creep:
# no plan step above 7 deg steering wheel per 10 ms; turn-in from a residual opposite curvature of -0.01 / -0.03 1/m: crosses zero
# after 0.24 / 0.65 m of travel.
# Stateless here: the state is controlsd's previous desired curvature, which is the measured curvature whenever lateral is inactive
# (standstill, disengaged); while the driver steers (steeringPressed) controlsd passes the measured curvature instead of the previous
# plan, so the plan starts from the driver's wheel and F does not pull against it.
CRAWL_PLAN_SPEED = 1.4                 # m/s (5 km/h): unchanged from here up
CRAWL_PLAN_RELEASE_SPEED = 1.0         # m/s: the limits start to open here (the floor of curv_from_psis)
CRAWL_STRAIGHTEN_PER_M = 0.04          # 1/m per metre: constant part of the rate towards straight
CRAWL_STRAIGHTEN_LENGTH = 2.0          # m: proportional part, |previous plan| / this (0.2 1/m -> 0.1 1/m per metre more)
CRAWL_TURN_IN_PER_M = 0.4              # 1/m per metre away from straight (0 -> 0.1 1/m in 0.25 m)
CRAWL_PER_M_RELEASE = 1.0              # 1/m per metre: both rates are at least this at CRAWL_PLAN_SPEED (0.014 1/m per 10 ms step)


def _crawl_rate(c: float, v_ego: float) -> float:
  if v_ego <= CRAWL_PLAN_RELEASE_SPEED:
    return c
  w = (v_ego - CRAWL_PLAN_RELEASE_SPEED) / (CRAWL_PLAN_SPEED - CRAWL_PLAN_RELEASE_SPEED)
  return c + w * (max(c, CRAWL_PER_M_RELEASE) - c)


def crawl_plan_curvature(v_ego: float, prev_curvature: float, new_curvature: float, dt: float) -> float:
  """the plan's desired curvature at crawl speed (rule above): new_curvature from CRAWL_PLAN_SPEED up; below, prev_curvature moved
  towards new_curvature by the distance travelled in this step (v_ego * dt), first towards straight, then away from it"""
  if v_ego >= CRAWL_PLAN_SPEED:
    return new_curvature
  ds = max(v_ego, 0.0) * dt
  k = prev_curvature
  if k != 0.0 and (new_curvature - k) * k < 0.0:
    # towards straight: to the model's plan if it is on this side, else to zero
    target = new_curvature if new_curvature * k > 0.0 else 0.0
    c = _crawl_rate(CRAWL_STRAIGHTEN_PER_M + abs(k) / CRAWL_STRAIGHTEN_LENGTH, v_ego)
    if abs(target - k) > c * ds:
      return k - math.copysign(c * ds, k)
    ds -= abs(target - k) / c
    k = target
    if k == new_curvature:
      return k
  # away from straight (further into the turn, from zero, or past zero) with the distance left in this step
  step = _crawl_rate(CRAWL_TURN_IN_PER_M, v_ego) * ds
  delta = new_curvature - k
  return k + (delta if abs(delta) <= step else math.copysign(step, delta))
