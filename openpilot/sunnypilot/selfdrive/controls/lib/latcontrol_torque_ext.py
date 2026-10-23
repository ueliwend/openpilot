"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.nnlc import NeuralNetworkLateralControl
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_override import LatControlTorqueExtOverride
from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import is_inster


class LatControlTorqueExt(NeuralNetworkLateralControl, LatControlTorqueExtOverride):
  def __init__(self, lac_torque, CP, CP_SP, CI):
    NeuralNetworkLateralControl.__init__(self, lac_torque, CP, CP_SP, CI)
    LatControlTorqueExtOverride.__init__(self, CP)
    # Hyundai Inster: its steering cap (409 raw, lac_torque.steer_max = 409 / 384) is a property of the car. The jerk-aware / NNLC
    # extensions run the controller's PID in torque space; controlsd sets its limits to +-steer_max (extension.update_limits()) only
    # when it applies torque parameters, so for the Inster the extension does it itself on every step (also when useParams is off),
    # and holds its output to the cap.
    self._inster = is_inster(CP)
    if self._inster:
      self._pid = lac_torque.pid
      self.update_limits()

  def update(self, CS, VM, pid, params, ff, pid_log, setpoint, measurement, calibrated_pose, roll_compensation,
             desired_lateral_accel, actual_lateral_accel, lateral_accel_deadzone, gravity_adjusted_lateral_accel,
             desired_curvature, actual_curvature, steer_limited_by_safety, output_torque):
    self._ff = ff
    self._pid = pid
    self._pid_log = pid_log
    self._setpoint = setpoint
    self._measurement = measurement
    self._roll_compensation = roll_compensation
    self._lateral_accel_deadzone = lateral_accel_deadzone
    self._desired_lateral_accel = desired_lateral_accel
    self._actual_lateral_accel = actual_lateral_accel
    self._desired_curvature = desired_curvature
    self._actual_curvature = actual_curvature
    self._gravity_adjusted_lateral_accel = gravity_adjusted_lateral_accel
    self._steer_limited_by_safety = steer_limited_by_safety
    self._output_torque = output_torque
    if self._inster:
      self.update_limits()   # torque-space limits (+-steer_max) for the extensions' PID from the first step (no-op when neither runs)

    self.update_calculations(CS, VM, desired_lateral_accel)
    self.update_jerk_aware_torque_control(CS, roll_compensation, gravity_adjusted_lateral_accel)
    self.update_neural_network_feedforward(CS, params, calibrated_pose)

    if self._inster:
      limit = self.lac_torque.steer_max
      out = self._output_torque
      self._output_torque = limit if out > limit else (-limit if out < -limit else out)

    return self._pid_log, self._output_torque
