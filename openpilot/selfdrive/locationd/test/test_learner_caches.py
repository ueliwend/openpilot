"""Learner caches (lagd, paramsd, torqued) across routes: a route whose car was not recognised (mock) must neither
destroy the caches nor become the previous route the next route compares against; a real car change still resets."""
from unittest import mock

import numpy as np

from openpilot.cereal import messaging
from opendbc.car.car_helpers import interfaces
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.car.card import write_prev_route_car_params
from openpilot.selfdrive.locationd import lagd, paramsd, torqued
from openpilot.selfdrive.locationd.lagd import retrieve_initial_lag
from openpilot.selfdrive.locationd.paramsd import retrieve_initial_vehicle_params
from openpilot.selfdrive.locationd.torqued import TorqueEstimator

INSTER, OTHER, MOCK = "HYUNDAI_INSTER", "HYUNDAI_KONA_EV_2ND_GEN", "MOCK"
CACHE_KEYS = ("LiveDelay", "LiveParametersV2", "LiveTorqueParameters", "LagdValueCache")


def get_CP(fingerprint):
  return interfaces[fingerprint].get_non_essential_params(fingerprint)


def start_route(params, CP):
  # what card writes at every start
  write_prev_route_car_params(params)
  cp_bytes = CP.to_bytes()
  params.put("CarParams", cp_bytes, block=True)
  params.put("CarParamsPersistent", cp_bytes, block=True)


def save_learned_caches(params):
  # caches as a recognised route's learners save them (logMonoTime 0, so the bytes are reproducible)
  lag = messaging.new_message('lateralDelay', logMonoTime=0)
  lag.lateralDelay.lateralDelayEstimate = 0.31
  lag.lateralDelay.validBlocks = 7
  lag.lateralDelay.status = 'estimated'
  lag.lateralDelay.version = lagd.VERSION
  params.put("LiveDelay", lag.to_bytes(), block=True)

  vp = messaging.new_message('vehicleParameters', logMonoTime=0)
  vp.vehicleParameters.steerRatio = 15.6
  vp.vehicleParameters.stiffnessFactor = 1.0
  vp.vehicleParameters.angleOffsetAverageDeg = 0.4
  params.put("LiveParametersV2", vp.to_bytes(), block=True)

  tp = messaging.new_message('lateralTorqueParameters', logMonoTime=0)
  ltp = tp.lateralTorqueParameters
  ltp.version = torqued.VERSION
  ltp.valid = True
  ltp.latAccelFactorFiltered = 2.1
  ltp.latAccelOffsetFiltered = 0.05
  ltp.frictionCoefficientFiltered = 0.12
  ltp.decay = 25.0
  ltp.points = [[x, 2.1 * x] for x in np.linspace(-0.25, 0.25, 21).tolist()]
  params.put("LiveTorqueParameters", tp.to_bytes(), block=True)

  params.put("LagdValueCache", 0.31, block=True)


def get_caches(params):
  return {k: params.get(k) for k in CACHE_KEYS}


def restored_state(params, CP):
  # what each learner starts from on this route
  est = TorqueEstimator(CP)
  torque = {k: f.x for k, f in est.filtered_params.items()}
  torque['decay'] = est.decay
  torque['points'] = len(est.filtered_points)
  return (retrieve_initial_lag(params, CP), retrieve_initial_vehicle_params(params, CP, False, False)[:3], torque)


# A7 (full scale 1.0 = 409): torque_data values per 409 and the pre-A7 values per 384
A7_FACTOR, A7_FRICTION = 3.9408854, 0.0798044
PRE_A7_FACTOR, PRE_A7_FRICTION = 3.7, 0.085
USER_FLOATS = {"TorqueParamsOverrideLatAccelFactor": 2.5, "TorqueParamsOverrideFriction": 0.1, "TorqueControlTune": 0.0,
               "LagdToggleDelay": 0.2}
USER_BOOLS = {"TorqueParamsOverrideEnabled": False, "CustomTorqueParams": False, "LiveTorqueParamsToggle": True,
              "LiveTorqueParamsRelaxedToggle": False, "EnforceTorqueControl": True, "LagdToggle": True}   # the owner's settings
ALL_KEYS = (*CACHE_KEYS, *USER_FLOATS, *USER_BOOLS)


def get_pre_a7_CP():
  # synthetic pre-A7 CarParams: the A7 Inster CarParams with the per-384 torque_data values
  CP = get_CP(INSTER)
  CP.lateralTuning.torque.latAccelFactor = PRE_A7_FACTOR
  CP.lateralTuning.torque.friction = PRE_A7_FRICTION
  return CP


def put_user_keys(params):
  for k, v in USER_FLOATS.items():
    params.put(k, v, block=True)
  for k, v in USER_BOOLS.items():
    params.put_bool(k, v, block=True)


def snapshot(params):
  return {k: params.get(k) for k in ALL_KEYS}


def assert_torque_reset(torque, factor, friction):
  np.testing.assert_allclose([torque['latAccelFactor'], torque['latAccelOffset'], torque['frictionCoefficient'], torque['decay']],
                             [factor, 0.0, friction, 50.0], rtol=1e-6)
  assert torque['points'] == 0


def assert_lag_and_vehicle_restored(lag, vehicle):
  assert lag == (np.float32(0.31), 7)
  np.testing.assert_allclose(vehicle, (15.6, 1.0, 0.4), rtol=1e-6)


class _Stop(Exception):
  pass


def run_main(module, n_frames=1200):
  """Run a learner's main loop for n_frames with no inputs (IPC faked), so its periodic cache writes happen."""
  class FakeSubMaster:
    def __init__(self, services, poll=None):
      self.frame = 0
      self.updated = {s: s == 'deviceMotion' for s in services}

    def update(self, timeout=None):
      if self.frame >= n_frames:
        raise _Stop
      self.frame += 1

    def all_checks(self, service_list=None):
      return False

  class FakeVehicleParamsLearner:
    def __init__(self, *args, **kwargs):
      pass

    def get_msg(self, valid, debug=False):
      return messaging.new_message('vehicleParameters')

  with mock.patch.object(module.messaging, 'SubMaster', FakeSubMaster), mock.patch.object(module.messaging, 'PubMaster'), \
       mock.patch.object(module, 'config_realtime_process'), mock.patch.object(paramsd, 'VehicleParamsLearner', FakeVehicleParamsLearner):
    try:
      module.main()
    except _Stop:
      pass


class TestLearnerCaches(OpenpilotTestCase):
  def test_mock_route_keeps_caches(self):
    # reference: two recognised routes in a row
    with OpenpilotPrefix():
      ref = Params()
      start_route(ref, get_CP(INSTER))
      save_learned_caches(ref)
      start_route(ref, get_CP(INSTER))
      direct = restored_state(ref, get_CP(INSTER))
      direct_caches = get_caches(ref)

    params = Params()
    start_route(params, get_CP(INSTER))
    save_learned_caches(params)
    saved = get_caches(params)
    inster_cp = params.get("CarParamsPersistent")

    # route with a car that was not recognised: learners start from defaults, nothing is removed or saved
    mock_CP = get_CP(MOCK)
    start_route(params, mock_CP)
    assert params.get("CarParamsPrevRoute") == inster_cp
    lag, vehicle, torque = restored_state(params, mock_CP)
    assert lag is None
    assert vehicle == (mock_CP.steerRatio, 1.0, 0.0)
    assert torque['points'] == 0
    for module in (lagd, paramsd, torqued):
      run_main(module)
    assert get_caches(params) == saved

    # next recognised route compares against the last recognised car and restores exactly as without the mock route
    start_route(params, get_CP(INSTER))
    assert params.get("CarParamsPrevRoute") == inster_cp
    after_mock = restored_state(params, get_CP(INSTER))
    assert get_caches(params) == saved == direct_caches
    assert after_mock == direct
    lag, vehicle, torque = after_mock
    assert lag == (np.float32(0.31), 7)
    np.testing.assert_allclose(vehicle, (15.6, 1.0, 0.4), rtol=1e-6)
    np.testing.assert_allclose([torque['latAccelFactor'], torque['latAccelOffset'], torque['frictionCoefficient'], torque['decay']],
                               [2.1, 0.05, 0.12, 25.0], rtol=1e-6)
    assert torque['points'] == 21

  def test_several_mock_routes(self):
    params = Params()
    start_route(params, get_CP(INSTER))
    save_learned_caches(params)
    saved = get_caches(params)
    inster_cp = params.get("CarParamsPersistent")
    for _ in range(3):
      start_route(params, get_CP(MOCK))
      run_main(lagd)
    start_route(params, get_CP(INSTER))
    assert params.get("CarParamsPrevRoute") == inster_cp
    assert get_caches(params) == saved
    assert restored_state(params, get_CP(INSTER))[0] == (np.float32(0.31), 7)

  def test_recognised_route_saves(self):
    params = Params()
    start_route(params, get_CP(INSTER))
    save_learned_caches(params)
    start_route(params, get_CP(INSTER))
    saved = get_caches(params)
    for module in (lagd, paramsd, torqued):
      run_main(module)
    now = get_caches(params)
    for k in CACHE_KEYS:
      assert now[k] != saved[k], k

  def test_real_car_change_resets(self):
    params = Params()
    start_route(params, get_CP(INSTER))
    save_learned_caches(params)
    inster_cp = params.get("CarParamsPersistent")

    other_CP = get_CP(OTHER)
    start_route(params, other_CP)
    assert params.get("CarParamsPrevRoute") == inster_cp
    lag, vehicle, torque = restored_state(params, other_CP)
    assert lag is None
    assert vehicle == (other_CP.steerRatio, 1.0, 0.0)
    assert torque['points'] == 0
    assert torque['latAccelFactor'] == other_CP.lateralTuning.torque.latAccelFactor
    assert params.get("LiveDelay") is None
    assert params.get("LiveParametersV2") is None

  def test_mock_first_boot(self):
    params = Params()
    mock_CP = get_CP(MOCK)
    start_route(params, mock_CP)
    assert params.get("CarParamsPrevRoute") is None
    lag, vehicle, torque = restored_state(params, mock_CP)
    assert lag is None and torque['points'] == 0
    for module in (lagd, paramsd, torqued):
      run_main(module)
    assert all(params.get(k) is None for k in CACHE_KEYS)

    inster_CP = get_CP(INSTER)
    start_route(params, inster_CP)
    assert params.get("CarParamsPrevRoute") is None
    lag, vehicle, torque = restored_state(params, inster_CP)
    assert lag is None
    assert vehicle == (inster_CP.steerRatio, 1.0, 0.0)
    assert torque['points'] == 0

  def test_unreadable_persistent_is_copied(self):
    params = Params()
    params.put("CarParamsPersistent", b"not carparams", block=True)
    write_prev_route_car_params(params)
    assert params.get("CarParamsPrevRoute") == b"not carparams"

  # A7 (Device & Platform, results/2026-11-12_a7_learned_torque section 6.2): the full-scale change resets torqued's cache through its
  # restore key and nothing else
  def test_a7_torque_data_is_per_409(self):
    tune = get_CP(INSTER).lateralTuning.torque
    np.testing.assert_allclose([tune.latAccelFactor, tune.friction], [A7_FACTOR, A7_FRICTION], rtol=1e-6)

  def test_full_scale_change_resets_only_torqued(self):
    params = Params()
    put_user_keys(params)
    start_route(params, get_pre_a7_CP())
    save_learned_caches(params)
    before = snapshot(params)

    start_route(params, get_CP(INSTER))               # a fresh CarParams builder per call (pycapnp writes a builder only once)
    lag, vehicle, torque = restored_state(params, get_CP(INSTER))
    assert_lag_and_vehicle_restored(lag, vehicle)
    assert_torque_reset(torque, A7_FACTOR, A7_FRICTION)
    assert snapshot(params) == before                 # starting the learners removes or changes nothing

    run_main(torqued)
    assert {k for k in ALL_KEYS if params.get(k) != before[k]} == {"LiveTorqueParameters"}

    # the next A7 route restores a cache written under A7 (no repeated reset)
    save_learned_caches(params)
    start_route(params, get_CP(INSTER))
    torque = restored_state(params, get_CP(INSTER))[2]
    np.testing.assert_allclose([torque['latAccelFactor'], torque['latAccelOffset'], torque['frictionCoefficient'], torque['decay']],
                               [2.1, 0.05, 0.12, 25.0], rtol=1e-6)
    assert torque['points'] == 21

  def test_full_scale_rollback_resets_torqued(self):
    params = Params()
    start_route(params, get_CP(INSTER))
    save_learned_caches(params)
    pre_CP = get_pre_a7_CP()
    start_route(params, pre_CP)
    lag, vehicle, torque = restored_state(params, pre_CP)
    assert_lag_and_vehicle_restored(lag, vehicle)
    assert_torque_reset(torque, PRE_A7_FACTOR, PRE_A7_FRICTION)

  def test_mock_route_after_full_scale_change(self):
    params = Params()
    start_route(params, get_pre_a7_CP())
    save_learned_caches(params)
    saved = get_caches(params)
    pre_cp = params.get("CarParamsPersistent")

    start_route(params, get_CP(MOCK))                 # first route on A7 software, car not recognised
    assert params.get("CarParamsPrevRoute") == pre_cp
    for module in (lagd, paramsd, torqued):
      run_main(module)
    assert get_caches(params) == saved                 # nothing restored, saved or removed

    start_route(params, get_CP(INSTER))
    assert params.get("CarParamsPrevRoute") == pre_cp
    lag, vehicle, torque = restored_state(params, get_CP(INSTER))
    assert_lag_and_vehicle_restored(lag, vehicle)
    assert_torque_reset(torque, A7_FACTOR, A7_FRICTION)

  def test_restore_key_holds_torque_units(self):
    # no-reset case 3: an upstream change that drops friction or latAccelFactor from torqued's restore key fails here
    key = TorqueEstimator.get_restore_key(get_CP(INSTER), torqued.VERSION)
    assert TorqueEstimator.get_restore_key(get_CP(INSTER), torqued.VERSION) == key
    for field, scale in (("latAccelFactor", 409 / 384), ("friction", 384 / 409)):
      CP = get_CP(INSTER)
      setattr(CP.lateralTuning.torque, field, getattr(CP.lateralTuning.torque, field) * scale)
      assert TorqueEstimator.get_restore_key(CP, torqued.VERSION) != key, field
