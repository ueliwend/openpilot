"""Settings keys of the multi-country maps (E9) and the map updates (E11) in common/params_keys.h."""
from openpilot.common.params import Params, ParamKeyFlag, ParamKeyType


def keys(flag):
  return {k.decode() for k in Params().all_keys(flag)}


def test_flags_and_types():
  p = Params()
  assert "OsmSelectedLocations" in keys(ParamKeyFlag.BACKUP) & keys(ParamKeyFlag.PERSISTENT)
  assert "OsmLocationStatus" in keys(ParamKeyFlag.PERSISTENT) and "OsmLocationStatus" not in keys(ParamKeyFlag.BACKUP)
  assert {"OsmDownloadRequest", "OsmDeleteLocation"} <= keys(ParamKeyFlag.CLEAR_ON_MANAGER_START)
  assert p.get_type("OsmSelectedLocations") == ParamKeyType.JSON and p.get_type("OsmLocationStatus") == ParamKeyType.JSON
  assert p.get_type("OsmDownloadRequest") == ParamKeyType.STRING and p.get_type("OsmDeleteLocation") == ParamKeyType.STRING
  for old in ("OsmLocationName", "OsmLocationTitle", "OsmStateName", "OsmStateTitle", "OsmLocal", "OsmDbUpdatesCheck"):
    assert old in keys(ParamKeyFlag.ALL)  # kept for upstream code (mirrored by mapd_manager)


def test_e11_keys():
  p = Params()
  assert "OsmAutoUpdate" in keys(ParamKeyFlag.BACKUP) & keys(ParamKeyFlag.PERSISTENT)
  assert p.get_type("OsmAutoUpdate") == ParamKeyType.BOOL and p.get_default_value("OsmAutoUpdate") is True  # owner Q3 a
  assert "OsmDownloaderLive" in keys(ParamKeyFlag.CLEAR_ON_MANAGER_START) and p.get_type("OsmDownloaderLive") == ParamKeyType.JSON
  assert "OsmDownloaderLive" not in keys(ParamKeyFlag.BACKUP)
  assert "MapdRestartRequest" not in keys(ParamKeyFlag.ALL)  # removed with mapd_priority.py; deleted at boot (params.cc)


def test_sunnylink_blocked_params():
  from openpilot.sunnypilot.sunnylink.utils import BLOCKED_PARAMS
  assert "MapdRestartRequest" not in BLOCKED_PARAMS
  assert "OsmAutoUpdate" not in BLOCKED_PARAMS  # a user setting: sunnylink may write it (read with get_bool: validated as a bool)
