"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.system.params_migration import run_migration

# The owner's device settings (route 00000113 initData, 10 Nov 2026) as the system defaults, so a fresh install comes up configured.
# Includes the defaults that already matched (InsterMPCRampAware, InsterSoftSmallErrorP, MADS, lagd, NNLC, uploads).
DEVICE_DEFAULTS = {
  "InsterMPC": True, "InsterMPCRampAware": True, "InsterLateralController": 1, "InsterAngleErrorPI": True, "InsterSpeedFeedforward": True,
  "InsterStictionFriction": True, "InsterSoftSmallErrorP": False,
  "EnforceTorqueControl": True, "LiveTorqueParamsToggle": True, "CameraOffset": 0.05,
  "HyundaiLongitudinalTuning": 2, "LongitudinalPersonality": 0,
  "SpeedLimitMode": 2, "SpeedLimitPolicy": 4, "SpeedLimitOffsetType": 2, "SpeedLimitValueOffset": 3,
  "GreenLightAlert": True, "LeadDepartAlert": True, "BlindSpot": True,
  "LaneTurnDesire": True, "LaneTurnValue": 19.62, "RoadEdgeLaneChangeEnabled": True,
  "ShowTurnSignals": True, "TorqueBar": True, "TrueVEgoUI": True, "RoadNameToggle": True, "RocketFuel": True, "ShowAdvancedControls": True,
  "ChevronInfo": 0, "OnroadScreenOffBrightness": 1,
  "IsMetric": True, "IsLdwEnabled": True,
  "RecordFront": False,  # owner, 12 Nov 2026: driver camera not recorded by default (was 1 since 10 Nov); the training guide still asks
  "Mads": True, "MadsUnifiedEngagementMode": True, "MadsMainCruiseAllowed": True, "MadsSteeringMode": 0,
  "LagdToggle": True, "LagdToggleDelay": 0.2, "NeuralNetworkLateralControl": False, "DisengageOnAccelerator": False, "OnroadUploads": True,
  # owner, 11 Nov 2026 (the three settings changed on the device after the reinstall; initData of the 11 Nov 09:27 drive): once the blind-spot
  # monitor has seen a vehicle, a timed lane change starts 1 s after it clears (later than the set time in Nudgeless / 0.5 s / 1 s, possibly
  # earlier in 2 s / 3 s); mobile data is not treated as metered and may roam (both without effect without a SIM)
  "AutoLaneChangeBsmDelay": True, "GsmMetered": False, "GsmRoaming": True,
  # owner, 12 Nov 2026 (E10): only the road camera preview qcamera.ts is stored by default, not fcamera.hevc / ecamera.hevc
  "RecordRoadFull": False,
}

# a value different from each default (the owner's store must survive a start unchanged)
OTHER = {k: (not v if isinstance(v, bool) else (v + 1 if isinstance(v, int) else v + 0.01)) for k, v in DEVICE_DEFAULTS.items()}
OTHER["OnroadScreenOffBrightness"] = 0          # values >= 2 are bumped by the one-off brightness migration of old stores


def manager_start(params):
  """the parameter part of system/manager/manager.py manager_init on a device: migration, then every unset key gets its default"""
  run_migration(params)
  for k in params.all_keys():
    default_value = params.get_default_value(k)
    if default_value is not None and params.get(k) is None:
      params.put(k, default_value, block=True)


class TestInsterDeviceDefaults(OpenpilotTestCase):
  def test_defaults(self):
    params = Params()
    for k, v in DEVICE_DEFAULTS.items():
      assert params.get_default_value(k) == v, (k, params.get_default_value(k), v)

  def test_fresh_store_comes_up_configured(self):
    params = Params()
    for k in DEVICE_DEFAULTS:
      assert params.get(k) is None, k
    manager_start(params)
    for k, v in DEVICE_DEFAULTS.items():
      assert params.get(k) == v, (k, params.get(k), v)
    manager_start(params)                                    # a second start changes nothing
    for k, v in DEVICE_DEFAULTS.items():
      assert params.get(k) == v, (k, params.get(k), v)

  def test_existing_store_unchanged(self):
    params = Params()
    params.put("OnroadScreenOffBrightnessMigrated", "1.0", block=True)
    for k, v in OTHER.items():
      params.put(k, v, block=True)
    manager_start(params)
    for k, v in OTHER.items():
      assert params.get(k) == v, (k, params.get(k), v)

  def test_store_without_the_new_default_keys(self):
    # the keys that had no default before (5 on 10 Nov 2026, GsmRoaming on 11 Nov 2026) get their default on a store that never set
    # them (1; RecordFront 0 since 12 Nov 2026; RecordRoadFull, new on 12 Nov 2026, 0); every value the store holds stays
    params = Params()
    params.put("OnroadScreenOffBrightnessMigrated", "1.0", block=True)
    new_keys = ("EnforceTorqueControl", "LiveTorqueParamsToggle", "IsMetric", "IsLdwEnabled", "RecordFront", "GsmRoaming", "RecordRoadFull")
    for k, v in OTHER.items():
      if k not in new_keys:
        params.put(k, v, block=True)
    manager_start(params)
    for k in new_keys:
      assert params.get(k) is DEVICE_DEFAULTS[k], (k, params.get(k))
    for k, v in OTHER.items():
      if k not in new_keys:
        assert params.get(k) == v, (k, params.get(k), v)

  def test_legacy_inster_selector_keeps_the_owners_choice(self):
    # an old store with the gain-set selector (0 = G2) migrates to PI (0), not to the new default PID (1)
    params = Params()
    params.put("InsterAngleErrorPIGains", 0, block=True)
    manager_start(params)
    assert params.get("InsterLateralController") == 0 and params.get("InsterAngleErrorPIGains") is None
