"""Settings keys of the multi-country maps (E9) in common/params_keys.h."""
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
