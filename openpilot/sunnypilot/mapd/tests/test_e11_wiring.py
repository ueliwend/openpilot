"""E11 wiring in upstream files (DESIGN 9.3, 9.11): process_config (swap before mapd, map_downloader, mapd_ready back to
upstream), mapd_manager (never asks the binary to download, clears a foreign request), and the removed E9 modules."""
import importlib.util
import os
import sys

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase

MAPD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestProcessConfig(OpenpilotTestCase):
  def test_mapd_command_runs_the_swap_then_execs_mapd(self):
    from openpilot.common.hardware.hw import Paths
    from openpilot.sunnypilot.mapd import offline_swap
    from openpilot.sunnypilot.mapd.mapd_manager import MAPD_PATH
    from openpilot.system.manager.process_config import managed_processes, mapd_ready, always_run
    mapd = managed_processes["mapd"]
    assert mapd.cmdline == offline_swap.launch_command(sys.executable, offline_swap.__file__, MAPD_PATH, mapd.cwd)  # root as at import
    words = mapd.cmdline[2].split()
    i = words.index(sys.executable)
    assert words[:i] == ["timeout", "-k", "5", "60", "nice", "-n", "19"] and os.path.isabs(words[i + 1]) and words[i + 1].endswith("offline_swap.py")
    downloader = managed_processes["map_downloader"]
    assert downloader.module == "openpilot.sunnypilot.mapd.map_downloader" and downloader.should_run is always_run
    params = Params()
    os.makedirs(Paths.mapd_root(), exist_ok=True)
    assert mapd_ready(False, params, None) is True  # upstream: only the folder (no restart request any more)


class TestMapdManagerGuard(OpenpilotTestCase):
  def test_foreign_download_request_is_cleared_and_ours_never_sent(self):
    from openpilot.sunnypilot.mapd import mapd_manager as mm
    mm.mem_params.put("OSMDownloadLocations", {"nations": ["CH"], "states": []}, block=True)
    mm.update_osm_db()
    assert mm.mem_params.get("OSMDownloadLocations") is None
    mm.params.put_bool("OsmDbUpdatesCheck", True, block=True)
    mm.params.put("OsmDownloadRequest", "CH", block=True)
    mm.update_osm_db()
    assert mm.mem_params.get("OSMDownloadLocations") is None  # requests are map_downloader's
    mm.mem_params.put("OSMDownloadLocations", {"nations": [], "states": []}, block=True)
    mm.update_osm_db()
    assert mm.mem_params.get("OSMDownloadLocations") == {"nations": [], "states": []}  # an empty one is left alone

  def test_main_loop_no_longer_handles_clear_cache(self):
    src = open(os.path.join(MAPD_DIR, "mapd_manager.py")).read()
    main = src[src.index("def main_thread"):]
    assert "clear_downloaded_maps()" not in main and "request_refresh_osm_location_data(" not in src[src.index("def update_osm_db"):]


def test_e9_workarounds_are_gone():
  for name in ("offline_update", "mapd_priority", "offline_maps"):
    assert importlib.util.find_spec(f"openpilot.sunnypilot.mapd.{name}") is None, name
  src = open(os.path.join(MAPD_DIR, "..", "..", "system", "manager", "process_config.py")).read()
  assert "MapdRestartRequest" not in src


def test_swap_module_uses_the_standard_library_only():
  import ast
  tree = ast.parse(open(os.path.join(MAPD_DIR, "offline_swap.py")).read())
  names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
  names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
  assert names <= set(sys.stdlib_module_names), names - set(sys.stdlib_module_names)
