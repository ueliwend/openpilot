"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Hyundai Inster MPC: the learned values (lagd's lateral delay, paramsd's steer ratio and stiffness, torqued's lateral-acceleration offset)
are held at the parameter set's values (inster_lateral.MPC_FIXED) while InsterMPC runs, through controlsd's state_control; every other
configuration keeps the learners' values.
"""
import itertools
import math
import numpy as np
from types import MappingProxyType, SimpleNamespace

from openpilot.cereal import messaging
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.controls import controlsd
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.car import interfaces as sunnypilot_interfaces
from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt
from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import ANGLE_PI_VERSION_MPC_FIXED_BIT, MPC_FIXED, MPC_PARAM, \
                                                                       MPC_RAMP_PARAM, angle_pi_log_version, fixed_or_learned, \
                                                                       mpc_fixed_values, LATERAL_PID, LATERAL_PI

ANGLE_PI, FF, FRICTION = "InsterAngleErrorPI", "InsterSpeedFeedforward", "InsterStictionFriction"
MPC_ON = (ANGLE_PI, FF, FRICTION, MPC_PARAM, MPC_RAMP_PARAM)   # the configuration the parameter set was fitted in
SWITCHES = (ANGLE_PI, FF, FRICTION, MPC_PARAM, MPC_RAMP_PARAM, "InsterSoftSmallErrorP", "LateralJerkTorqueController",
            "NeuralNetworkLateralControl")


def _setup(car_name, on=(), ctrl=LATERAL_PID, enforce=False, tune=0.0):
  """params and CarParams as controlsd sees them (after the sunnypilot interface set-up); tune None: TorqueControlTune unset"""
  params = Params()
  for key in SWITCHES:
    params.put_bool(key, key in on, block=True)
  params.put("InsterLateralController", ctrl, block=True)
  params.put_bool("EnforceTorqueControl", enforce, block=True)
  if tune is None:
    params.remove("TorqueControlTune")
  else:
    params.put("TorqueControlTune", tune, block=True)
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
  CI = CarInterface(CP, CP_SP)
  sunnypilot_interfaces.setup_interfaces(CI, params)
  return params, CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI


def _lateral_control(params, CP, CP_SP, CI):
  """the lateral controller controlsd builds for a torque car: upstream LatControlTorque, then ControlsExt's choice"""
  ext = SimpleNamespace(params=params, CP=CP, CP_SP=CP_SP)
  return ControlsExt.initialize_lateral_control(ext, LatControlTorque(CP, CP_SP, CI, DT_CTRL), CI, DT_CTRL)


def _controls(car_name, on=(), ctrl=LATERAL_PID):
  """controlsd's Controls with everything state_control reads, without its messaging (the learners' messages are set per step)"""
  params, CP, CP_SP, CI = _setup(car_name, on, ctrl)
  c = controlsd.Controls.__new__(controlsd.Controls)
  c.params, c.CP, c.CP_SP, c.CI = params, CP, CP_SP, CI
  c.steer_limited_by_safety = False
  c.curvature = c.desired_curvature = 0.0
  c.calibrated_pose = None
  c.LoC = LongControl(CP, CP_SP)
  c.VM = VehicleModel(CP)
  c.LaC = _lateral_control(params, CP, CP_SP, CI)
  c.max_lat_accel_no_roll = controlsd.max_lateral_accel_no_roll(CP, controlsd.MAX_LATERAL_ACCEL_NO_ROLL)
  c.inster_crawl_plan = bool(getattr(c.LaC, 'inster_angle_pi', False))
  c.inster_mpc_fixed = getattr(c.LaC, 'inster_fixed', None)
  c.get_lat_active = lambda sm: True
  return c


class _SM(dict):
  def __init__(self):
    super().__init__()
    self.updated = {}
    self.valid = {'lateralManeuverPlan': False, 'driverAssistance': False}
    self.logMonoTime = {}

  def all_checks(self, services=None):
    return True


def _learners(k, variant):
  """lagd / paramsd / torqued values at step k: variant 0 near the fit's values, variant 1 far from them (winter-like) and moving"""
  if variant == 0:
    return 0.2, 15.55, 1.0, 0.0
  t = k * DT_CTRL
  return 0.31 + 0.05 * math.sin(0.7 * t), 13.8 + 2.5 * (1.0 - math.exp(-t)), 0.978 + 0.01 * math.sin(0.3 * t), 0.14 * min(1.0, t)


def _drive(c, variant, n=700):
  """state_control on a synthetic drive (curves at changing speed, a driver press, an inactive stretch, model frames every 5 steps;
  steering angle from a lag on the plan, independent of the controller); returns every controller output and the delay controlsd
  passes, per step"""
  out = []
  angle = 0.0
  VM_plant = VehicleModel(c.CP)
  for k in range(n):
    t = k * DT_CTRL
    lat_delay, sr, sf, offset = _learners(k, variant)
    sm = _SM()
    cs = messaging.new_message('carState').carState
    cs.vEgo = 6.0 + 10.0 * (1.0 + math.sin(0.2 * t))
    cs.vCruise = 100.0
    kd = 0.01 * math.sin(0.9 * t) + 0.003 * math.sin(3.3 * t)
    target = math.degrees(VM_plant.get_steer_from_curvature(-kd, cs.vEgo, 0.0)) + 1.2
    rate = (target - angle) * 5.0
    angle += rate * DT_CTRL
    cs.steeringAngleDeg = angle
    cs.steeringRateDeg = rate
    cs.steeringPressed = 300 <= k < 320
    cs.canValid = True
    sm['carState'] = cs
    vp = messaging.new_message('vehicleParameters').vehicleParameters
    vp.steerRatio, vp.stiffnessFactor, vp.angleOffsetDeg, vp.roll = sr, sf, 1.2, 0.01 * math.sin(0.05 * t)
    sm['vehicleParameters'] = vp
    tq = messaging.new_message('lateralTorqueParameters').lateralTorqueParameters
    tq.useParams = True
    tq.latAccelFactorFiltered, tq.latAccelOffsetFiltered, tq.frictionCoefficientFiltered = 3.4, offset, 0.09
    sm['lateralTorqueParameters'] = tq
    ld = messaging.new_message('lateralDelay').lateralDelay
    ld.lateralDelay = lat_delay
    sm['lateralDelay'] = ld
    c.lat_delay = lat_delay                         # ControlsExt.get_params_sp: LagdToggle's value (the learned one here)
    mdl = messaging.new_message('modelV2').modelV2
    mdl.frameId = k // 5
    mdl.orientationRate.z = [(kd + 0.002 * T) * cs.vEgo for T in ModelConstants.T_IDXS]
    mdl.velocity.x = [cs.vEgo] * len(ModelConstants.T_IDXS)
    mdl.acceleration.y = [0.0] * len(ModelConstants.T_IDXS)
    mdl.orientation.x = [0.0] * len(ModelConstants.T_IDXS)
    mdl.orientation.y = [0.0] * len(ModelConstants.T_IDXS)
    mdl.action.desiredCurvature = kd
    sm['modelV2'] = mdl
    sm['longitudinalPlan'] = messaging.new_message('longitudinalPlan').longitudinalPlan
    ss = messaging.new_message('selfdriveState').selfdriveState
    ss.enabled = ss.active = not (450 <= k < 500)
    sm['selfdriveState'] = ss
    sm['onroadEvents'] = messaging.new_message('onroadEvents', 0).onroadEvents
    c.get_lat_active = lambda sm, a=ss.active: a
    c.sm = sm
    passed = {}
    update = c.LaC.update

    def spy(*args, passed=passed, update=update):
      passed['lat_delay'] = args[-1]
      return update(*args)
    c.LaC.update = spy
    CC, lac = c.state_control()
    del c.LaC.update
    lac_t = tuple(getattr(lac, f) for f in ('active', 'p', 'i', 'd', 'f', 'output', 'error', 'errorRate', 'actualLateralAccel',
                                            'desiredLateralAccel', 'desiredLateralJerk', 'saturated')) if hasattr(lac, 'p') else ()
    out.append((CC.actuators.torque, CC.actuators.curvature, c.curvature, c.desired_curvature, passed['lat_delay'], lac_t))
  return out


class TestInsterMPCFixedValues(OpenpilotTestCase):
  def test_values(self):
    # the values the MPC 7 Nov fit drives carried (kit steps of the 14 fit drives, active time): delay 0.194-0.220 s, steer ratio
    # 15.34-15.94, stiffness 0.998-1.011, offset -0.039-0 m/s^2; read-only
    assert dict(MPC_FIXED) == {'lat_delay': 0.2, 'steer_ratio': 15.55, 'stiffness_factor': 1.0, 'lat_accel_offset': 0.0}
    assert isinstance(MPC_FIXED, MappingProxyType)
    assert ANGLE_PI_VERSION_MPC_FIXED_BIT == 1024
    assert angle_pi_log_version(LATERAL_PID, False, True, True, True) == 2000106
    assert angle_pi_log_version(LATERAL_PID, False, True, False, True) == 1488106
    assert fixed_or_learned(None, 'lat_delay', 0.31) == 0.31 and fixed_or_learned(MPC_FIXED, 'lat_delay', 0.31) == 0.2

  def test_selection_matches_the_controller(self):
    # modeld cannot see controlsd's controller: mpc_fixed_values (CarParams + params at start) must give MPC_FIXED exactly when the
    # lateral controller controlsd builds holds the values, for every switch, controller type, NNLC and torque-control combination
    seen = set()
    for car_name, (angle_pi, ff, mpc), ctrl, nnlc, (enforce, tune) in itertools.product(
        (HYUNDAI.HYUNDAI_INSTER, HYUNDAI.HYUNDAI_KONA_EV), itertools.product((False, True), repeat=3), (LATERAL_PI, LATERAL_PID),
        (False, True), ((False, 0.0), (True, 0.0), (True, 1.0), (True, None))):
      on = tuple(k for k, b in ((ANGLE_PI, angle_pi), (FF, ff), (MPC_PARAM, mpc), ("NeuralNetworkLateralControl", nnlc)) if b)
      params, CP, CP_SP, CI = _setup(car_name, on + (FRICTION,), ctrl, enforce, tune)
      expected = mpc_fixed_values(CP, params)
      lac = _lateral_control(params, CP, CP_SP, CI)
      assert getattr(lac, 'inster_fixed', None) is expected, (car_name, on, ctrl, enforce, tune)
      assert expected is None or getattr(lac, 'inster_mpc', None) is not None
      seen.add(expected is not None)
    assert seen == {False, True}

  def test_learners_do_not_change_the_mpc(self):
    # InsterMPC on: the same drive with the learners near the fit's values and far from them (delay 0.26-0.36 s, steer ratio from
    # 13.8, stiffness 0.97-0.99, offset to 0.14 m/s^2) gives the same controller output, log, curvatures and delay, bit for bit
    a = _drive(_controls(HYUNDAI.HYUNDAI_INSTER, MPC_ON), 0)
    b = _drive(_controls(HYUNDAI.HYUNDAI_INSTER, MPC_ON), 1)
    assert a == b
    assert {r[4] for r in a} == {0.2 + controlsd.LAT_SMOOTH_SECONDS}
    assert any(r[0] != 0.0 for r in a) and any(not r[5][0] for r in a if r[5])   # it steers, and it went inactive once
    # the drive is sensitive to the learners: with the holding switched off the two variants differ
    c = _controls(HYUNDAI.HYUNDAI_INSTER, MPC_ON)
    assert c.LaC.inster_mpc is not None and c.LaC.inster_mpc.ra is not None
    c.inster_mpc_fixed = c.LaC.inster_fixed = None
    assert _drive(c, 1) != a

  def test_other_configurations_use_the_learners(self):
    # the MPC switch on but the MPC not in use (PI), the PID without the MPC, every switch off, another car: the learners' values reach
    # the controller as before (the outputs move with them; the delay controlsd passes is lagd's)
    for car_name, on, ctrl in ((HYUNDAI.HYUNDAI_INSTER, MPC_ON, LATERAL_PI), (HYUNDAI.HYUNDAI_INSTER, (ANGLE_PI, FF, FRICTION), LATERAL_PID),
                               (HYUNDAI.HYUNDAI_INSTER, (), LATERAL_PID), (HYUNDAI.HYUNDAI_KONA_EV, MPC_ON, LATERAL_PID)):
      c = _controls(car_name, on, ctrl)
      assert c.inster_mpc_fixed is None
      a, b = _drive(c, 0, 400), _drive(_controls(car_name, on, ctrl), 1, 400)
      assert a != b, (car_name, on, ctrl)
      assert [r[4] for r in b] == [float(np.float32(_learners(k, 1)[0])) + controlsd.LAT_SMOOTH_SECONDS for k in range(400)]   # Float32

  def test_learned_values_equal_to_the_fixed_ones(self):
    # the held values only replace the learners' values: with held values equal to the learners' (as the messages carry them, Float32),
    # InsterMPC gives the run it gave before (the code without the holding, here: the same controller with the holding switched off)
    held = MappingProxyType({k: float(np.float32(v)) for k, v in MPC_FIXED.items()})
    c = _controls(HYUNDAI.HYUNDAI_INSTER, MPC_ON)
    c.inster_mpc_fixed = c.LaC.inster_fixed = held
    a = _drive(c, 0)
    c = _controls(HYUNDAI.HYUNDAI_INSTER, MPC_ON)
    c.inster_mpc_fixed = c.LaC.inster_fixed = None
    assert a == _drive(c, 0)
