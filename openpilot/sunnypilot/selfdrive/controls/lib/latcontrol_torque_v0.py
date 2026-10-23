import math
import numpy as np
from collections import deque

from openpilot.cereal import log
from opendbc.car.lateral import get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.pid import PIDController

from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import InsterStictionFriction, STEER_MAX_RAW, ff_lat_accel_equivalent, \
                                                                       is_inster
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext import LatControlTorqueExt

# At higher speeds (25+mph) we can assume:
# Lateral acceleration achieved by a specific car correlates to
# torque applied to the steering rack. It does not correlate to
# wheel slip, or to speed.

# This controller applies torque to achieve desired lateral
# accelerations. To compensate for the low speed effects the
# proportional gain is increased at low speeds by the PID controller.
# Additionally, there is friction in the steering wheel that needs
# to be overcome to move it at all, this is compensated for too.

KP = 1.0
KI = 0.3
KD = 0.0
INTERP_SPEEDS = [1, 1.5, 2.0, 3.0, 5, 7.5, 10, 15, 30]
KP_INTERP = [250, 120, 65, 30, 11.5, 5.5, 3.5, 2.0, KP]

LP_FILTER_CUTOFF_HZ = 1.2
LAT_ACCEL_REQUEST_BUFFER_SECONDS = 1.0
FRICTION_THRESHOLD = 0.3
VERSION = 0


class LatControlTorque(LatControl):
  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    self.pid = PIDController([INTERP_SPEEDS, KP_INTERP], KI, KD, rate=1/self.dt)
    self.update_limits()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    self.lat_accel_request_buffer_len = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / self.dt)
    self.lat_accel_request_buffer = deque([0.] * self.lat_accel_request_buffer_len , maxlen=self.lat_accel_request_buffer_len)
    self.previous_measurement = 0.0
    self.measurement_rate_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), self.dt)

    self.extension = LatControlTorqueExt(self, CP, CP_SP, CI)

    # Hyundai Inster steering candidates (experimental, off by default, see inster_lateral.py). The switches are read once and only
    # for the Inster: every other car, and the Inster with both switches off, runs the unchanged code below.
    self.inster_friction = False
    self.inster_ff = False
    if is_inster(CP):
      self._init_inster(Params())

  def _init_inster(self, params):
    friction = params.get_bool("InsterStictionFriction")
    feedforward = params.get_bool("InsterSpeedFeedforward")
    if not (friction or feedforward):
      return
    if params.get_bool("NeuralNetworkLateralControl"):
      cloudlog.warning("Inster steering switches are ignored: they apply to the v0 torque controller without NNLC")
      return
    self.inster_friction = friction
    self.inster_ff = feedforward
    # v0 keeps its own output: the Lateral Jerk Torque Controller extension must not replace it (its torque-params override stays)
    if self.extension._jerk_aware_enabled:
      cloudlog.warning("Inster steering switch enabled: the Lateral Jerk Torque Controller is not used")
      self.extension._jerk_aware_enabled = False
    if friction:
      self.inster_stiction = InsterStictionFriction(self.dt)
      # desired curvature of the last second, every step (the request made lat_delay ago is the plan for now)
      self.inster_curvature_buffer = deque([0.] * self.lat_accel_request_buffer_len, maxlen=self.lat_accel_request_buffer_len)
      self.inster_past_curvature = 0.0
      self.inster_last_output_torque = 0.0

  def update_torque_parameters(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction
    self.update_limits()

  def update_limits(self):
    self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, self.torque_params),
                        self.lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    # Override torque params from extension
    if self.extension.update_override_torque_params(self.torque_params):
      self.update_limits()

    if self.inster_friction:
      # the curvature requested lat_delay ago (the plan for now); the history is kept every step, active or not
      inster_delay_frames = int(np.clip(lat_delay / self.dt, 1, self.lat_accel_request_buffer_len))
      self.inster_past_curvature = self.inster_curvature_buffer[-inster_delay_frames]
      self.inster_curvature_buffer.append(desired_curvature)

    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION
    if not active:
      output_torque = 0.0
      pid_log.active = False
      if self.inster_friction:
        # inactive: v0 outputs 0; the term restarts settled on the next active step
        self.inster_stiction.reset()
        self.inster_last_output_torque = 0.0
    else:
      measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
      roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
      curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
      lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

      delay_frames = int(np.clip(lat_delay / self.dt, 1, self.lat_accel_request_buffer_len))
      expected_lateral_accel = self.lat_accel_request_buffer[-delay_frames]
      # TODO factor out lateral jerk from error to later replace it with delay independent alternative
      future_desired_lateral_accel = desired_curvature * CS.vEgo ** 2
      self.lat_accel_request_buffer.append(future_desired_lateral_accel)
      gravity_adjusted_future_lateral_accel = future_desired_lateral_accel - roll_compensation
      desired_lateral_jerk = (future_desired_lateral_accel - expected_lateral_accel) / max(lat_delay, self.dt)

      measurement = measured_curvature * CS.vEgo ** 2
      measurement_rate = self.measurement_rate_filter.update((measurement - self.previous_measurement) / self.dt)
      self.previous_measurement = measurement

      setpoint = lat_delay * desired_lateral_jerk + expected_lateral_accel
      error = setpoint - measurement

      # do error correction in lateral acceleration space, convert at end to handle non-linear torque responses correctly
      pid_log.error = float(error)
      ff = gravity_adjusted_future_lateral_accel
      # latAccelOffset corrects roll compensation bias from device roll misalignment relative to car roll
      ff -= self.torque_params.latAccelOffset
      if self.inster_ff:
        # Inster: feed-forward from the measured speed x lateral-acceleration table, in linear-equivalent lateral acceleration
        ff = ff_lat_accel_equivalent(ff, CS.vEgo, self.torque_params.latAccelFactor)
      if self.inster_friction:
        # Inster: stiction friction term (raw command units) instead of get_friction
        ff += self._inster_friction_raw(CS, VM, desired_curvature, self.inster_past_curvature, measured_curvature,
                                        lat_delay) / STEER_MAX_RAW * self.torque_params.latAccelFactor
      else:
        # TODO jerk is weighted by lat_delay for legacy reasons, but should be made independent of it
        ff += get_friction(error, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)

      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
      output_lataccel = self.pid.update(pid_log.error,
                                       -measurement_rate,
                                        feedforward=ff,
                                        speed=CS.vEgo,
                                        freeze_integrator=freeze_integrator)
      output_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)

      # Lateral acceleration torque controller extension updates
      # Overrides pid_log.error and output_torque
      pid_log, output_torque = self.extension.update(CS, VM, self.pid, params, ff, pid_log, setpoint, measurement, calibrated_pose, roll_compensation,
                                                     future_desired_lateral_accel, measurement, lateral_accel_deadzone, gravity_adjusted_future_lateral_accel,
                                                     desired_curvature, measured_curvature, steer_limited_by_safety, output_torque)
      if self.inster_friction:
        self.inster_last_output_torque = output_torque

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.d = float(self.pid.d)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(-output_torque) # TODO: log lat accel?
      pid_log.actualLateralAccel = float(measurement)
      pid_log.desiredLateralAccel = float(setpoint)
      pid_log.desiredLateralJerk = float(desired_lateral_jerk)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log

  def _inster_friction_raw(self, CS, VM, desired_curvature, past_curvature, measured_curvature, lat_delay):
    # steering-wheel degrees per unit curvature at this speed (vehicle model without the roll term; the measured curvature keeps its
    # roll compensation, as v0's measurement does). Curvatures and the term are + right, like v0's internal convention.
    steer_deg_per_curvature = math.degrees(VM.get_steer_from_curvature(1.0, CS.vEgo, 0.0))
    return self.inster_stiction.update(CS.vEgo,
                                       steer_deg_per_curvature * desired_curvature,
                                       steer_deg_per_curvature * past_curvature,
                                       steer_deg_per_curvature * measured_curvature,
                                       lat_delay, abs(CS.steeringRateDeg), self.inster_last_output_torque * STEER_MAX_RAW)
