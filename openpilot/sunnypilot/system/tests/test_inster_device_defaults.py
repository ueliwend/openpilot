"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.modeld_v2.blendful_right import MODE_RIGHT, mode_from_params
from openpilot.sunnypilot.system.params_migration import run_migration

# The owner's device settings (route 00000113 initData, 10 Nov 2026) as the system defaults, so a fresh install comes up configured.
# Includes the defaults that already matched (InsterMPCRampAware, InsterSoftSmallErrorP, MADS, lagd, NNLC, uploads).
DEVICE_DEFAULTS = {
  "InsterMPC": True, "InsterMPCRampAware": True, "InsterLateralController": 1, "InsterAngleErrorPI": True, "InsterSpeedFeedforward": True,
  "InsterStictionFriction": True, "InsterSoftSmallErrorP": False,
  "EnforceTorqueControl": True, "LiveTorqueParamsToggle": True, "CameraOffset": 0.05,
  "HyundaiLongitudinalTuning": 2,
  "SpeedLimitPolicy": 4, "SpeedLimitOffsetType": 2,
  "GreenLightAlert": True, "LeadDepartAlert": True, "BlindSpot": True,
  "LaneTurnDesire": True, "LaneTurnValue": 19.62, "RoadEdgeLaneChangeEnabled": True,
  "ShowTurnSignals": True, "TorqueBar": True, "TrueVEgoUI": True, "RoadNameToggle": True, "RocketFuel": True, "ShowAdvancedControls": True,
  "OnroadScreenOffBrightness": 1,
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
  # owner, 13 Nov 2026 (E11, Q3 a): every map tile of the selected countries is checked once a week
  "OsmAutoUpdate": True,
  # owner, 16 Nov 2026 (initial set speed): Toggles -> Initial Max Speed (fallback) ranges from 20 km/h, default 20 (was 40); a stored
  # value is kept, also through a start with an unrecognised car (test_existing_store_unchanged)
  "InitialCruiseSpeedKph": 20,
}
# owner, 21 Nov 2026 (E13, standing rule: his newest settings are the branch defaults): values of route 0000006b's initData (22 Nov 2026)
OWNER_DEFAULTS_22NOV = {
  "AutoLaneChangeTimer": 2,                  # Auto Lane Change by Blinker: 0.5 s (was 0 = Nudge)
  "Blendful": True, "BlendfulMode": 1,       # Blendful (Alpha): Right line (was Off; kind was 0 = Regular)
  "BlendfulRightGapLow": 0.4,                # Right line gap up to 50 km/h, m (was 0.5)
  "AlphaLongitudinalEnabled": True,          # sunnypilot Longitudinal Control (Alpha) (no default before)
  "LongitudinalPersonality": 1,              # Following Distance: Standard (was 0 = Aggressive)
  "SmartCruiseControlVision": True, "SmartCruiseControlMap": True,  # (were off)
  "SpeedLimitMode": 3,                       # Speed Limit: Assist (was 2 = Warning)
  "SpeedLimitValueOffset": 2,                # Speed Limit Offset value, % with SpeedLimitOffsetType 2 (was 3)
  "ChevronInfo": 4,                          # Display Metrics Below Chevron: All (was 0 = Off)
  "DevUIInfo": 1,                            # Developer UI: Bottom (was 0 = Off)
}
DEVICE_DEFAULTS.update(OWNER_DEFAULTS_22NOV)

# every other choice the settings offer for each key of OWNER_DEFAULTS_22NOV (a stored choice must survive every start)
OTHER_CHOICES_22NOV = {
  "AutoLaneChangeTimer": [-1, 0, 1, 3, 4, 5], "Blendful": [False], "BlendfulMode": [0], "BlendfulRightGapLow": [0.2, 0.5, 2.0],
  "AlphaLongitudinalEnabled": [False], "LongitudinalPersonality": [0, 2], "SmartCruiseControlVision": [False],
  "SmartCruiseControlMap": [False], "SpeedLimitMode": [0, 1, 2], "SpeedLimitValueOffset": [-30, 0, 3, 30],
  "ChevronInfo": [0, 1, 2, 3], "DevUIInfo": [0, 2, 3],
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
    # the keys that had no default before (5 on 10 Nov 2026, GsmRoaming on 11 Nov 2026, AlphaLongitudinalEnabled on 22 Nov 2026) get their
    # default on a store that never set them (1; RecordFront 0 since 12 Nov 2026; RecordRoadFull, new on 12 Nov 2026, 0); every value the
    # store holds stays
    params = Params()
    params.put("OnroadScreenOffBrightnessMigrated", "1.0", block=True)
    new_keys = ("EnforceTorqueControl", "LiveTorqueParamsToggle", "IsMetric", "IsLdwEnabled", "RecordFront", "GsmRoaming", "RecordRoadFull",
                "AlphaLongitudinalEnabled")
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


class TestOwnerDefaults22Nov(OpenpilotTestCase):
  """E13 (owner, 21 Nov 2026): the owner's newest settings are the defaults; a default only fills an unset key."""

  def test_params_keys_returns_the_owners_values(self):
    params = Params()
    for k, v in OWNER_DEFAULTS_22NOV.items():
      assert params.get_default_value(k) == v, (k, params.get_default_value(k), v)
      assert params.get(k) is None and params.get(k, return_default=True) == v, k   # readers with return_default see the same value
    assert params.get_default_value("LongitudinalPersonality") == log.LongitudinalPersonality.standard
    assert mode_from_params(params.get_default_value("Blendful"), params.get_default_value("BlendfulMode")) == MODE_RIGHT
    assert params.get_default_value("SpeedLimitOffsetType") == 2                    # offset type % unchanged
    assert params.get_default_value("AutoLaneChangeBsmDelay") is True               # Delay with Blind Spot: branch default kept (on)

  def test_every_stored_choice_survives_every_start(self):
    params = Params()
    params.put("OnroadScreenOffBrightnessMigrated", "1.0", block=True)
    rounds = max(len(v) for v in OTHER_CHOICES_22NOV.values())
    for i in range(rounds):
      stored = {k: v[i % len(v)] for k, v in OTHER_CHOICES_22NOV.items()}
      for k, v in stored.items():
        params.put(k, v, block=True)
      for _ in range(2):
        manager_start(params)
        for k, v in stored.items():
          assert params.get(k) == v, (k, params.get(k), v)

  def test_stored_value_equal_to_the_old_default_stays(self):
    # a device that stores the old default (it started an earlier build, so every key with a default was filled) keeps it
    params = Params()
    params.put("OnroadScreenOffBrightnessMigrated", "1.0", block=True)
    old = {"AutoLaneChangeTimer": 0, "Blendful": False, "BlendfulMode": 0, "BlendfulRightGapLow": 0.5, "LongitudinalPersonality": 0,
           "SmartCruiseControlVision": False, "SmartCruiseControlMap": False, "SpeedLimitMode": 2, "SpeedLimitValueOffset": 3,
           "ChevronInfo": 0, "DevUIInfo": 0}
    for k, v in old.items():
      params.put(k, v, block=True)
    manager_start(params)
    for k, v in old.items():
      assert params.get(k) == v, (k, params.get(k), v)
    assert params.get("AlphaLongitudinalEnabled") is True                            # had no default: an unset key is filled

  def test_removed_key_is_filled_again_at_the_next_start(self):
    # a key removed by software (e.g. a capability check) is unset, so the next manager start fills the default again
    params = Params()
    manager_start(params)
    for k in ("AlphaLongitudinalEnabled", "SmartCruiseControlVision", "SmartCruiseControlMap"):
      params.remove(k)
    manager_start(params)
    for k in ("AlphaLongitudinalEnabled", "SmartCruiseControlVision", "SmartCruiseControlMap"):
      assert params.get(k) is True, k
