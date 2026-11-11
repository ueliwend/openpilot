"""The single-country "Database Update" through mapd_manager, against the fake mapd (fake_mapd.py).

mapd v1.12.0 opens map files without truncation (download.go:240), so an update that brings shorter files leaves the
old files' tails behind. These tests fail on openpilot inster d0ba8c5 (before the offline update protection).
"""
import os
import random

import openpilot.sunnypilot.mapd.mapd_manager as mm
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.mapd.tests.fake_mapd import FakeMapd, bounding_boxes, make_tile, tiles_for_box

CH = tiles_for_box(bounding_boxes("nation")["CH"]["bounding_box"])


def area_files(tile, version):
  rnd = random.Random(f"{version}{tile}")
  length = 700 if version == "old" else 350
  return {f"{tile[0]}.{i:02d}_{tile[1]}.area": rnd.randbytes(length) for i in range(3)}


class Clock:
  t = 1000.0


class TestMapdManagerRedownload(OpenpilotTestCase):
  def _setup(self, monkeypatch):
    self.params, self.mem = Params(), Params("/dev/shm/params")
    monkeypatch.setattr(mm, "params", self.params)
    monkeypatch.setattr(mm, "mem_params", self.mem)
    self.root = Paths.mapd_root()
    self.old = {t: area_files(t, "old") for t in CH}
    self.new = {t: area_files(t, "new") for t in CH}
    for t, files in self.old.items():  # the maps of an earlier download
      d = os.path.join(self.root, "offline", str(t[0]), str(t[1]))
      os.makedirs(d)
      for name, data in files.items():
        with open(os.path.join(d, name), "wb") as f:
          f.write(data)
    self.clock = Clock()
    self.request_path = self.mem.get_param_path("OSMDownloadLocations")
    server = {t: make_tile(t, files) for t, files in self.new.items()}
    self.mapd = FakeMapd(self.root, self.request_path, self.params.get_param_path("OSMDownloadProgress"), server, self.clock, chunk=256)
    self.gen = self.mapd.run()
    next(self.gen)
    self.params.put_bool("IsOffroad", True)
    self.params.put("OsmLocationName", "CH")
    if hasattr(mm, "offline_updater"):
      from openpilot.sunnypilot.mapd.offline_update import OfflineMapUpdater, ParamsLink
      link = ParamsLink(self.params, self.mem, mm.request_refresh_osm_location_data)
      monkeypatch.setattr(mm, "_offline_updater", OfflineMapUpdater(self.root, link, monotonic=lambda: self.clock.t, sleep=self._sleep))

  def _sleep(self, seconds):
    for _ in range(4):
      if next(self.gen) == ("heartbeat",):
        return

  def _idle(self):
    updater = getattr(mm, "_offline_updater", None)
    pending = os.path.exists(self.request_path) and os.path.getsize(self.request_path) > 0
    return (self.mapd.gets and not pending and not self.params.get_bool("OsmDbUpdatesCheck")
            and (updater is None or updater.state == "idle"))

  def _run(self, polls=600, before_poll=None):
    for _ in range(polls):
      if before_poll:
        before_poll()
      mm.update_osm_db()  # one mapd_manager loop
      if self._idle():
        return
      for _ in range(3):
        next(self.gen)
    raise AssertionError("the update did not finish")

  def _disk(self, t):
    d = os.path.join(self.root, "offline", str(t[0]), str(t[1]))
    if not os.path.isdir(d):
      return None
    out = {}
    for name in os.listdir(d):
      with open(os.path.join(d, name), "rb") as f:
        out[name] = f.read()
    return out

  def test_database_update_with_shorter_tiles_leaves_readable_maps(self, monkeypatch):
    self._setup(monkeypatch)
    self.params.put_bool("OsmDbUpdatesCheck", True)
    self._run()
    for t in CH:
      assert self._disk(t) == self.new[t], f"tile {t}: map files keep the old tails (corrupt)"
    for sub in ("offline_prev", "tmp"):
      assert not os.path.exists(os.path.join(self.root, sub))

  def test_delete_all_during_an_update_leaves_no_maps(self, monkeypatch):
    self._setup(monkeypatch)
    self.params.put_bool("OsmDbUpdatesCheck", True)
    polls = []

    def main_loop_clear():  # the Mapd_ClearCache step of mapd_manager.main_thread, as on the branch under test
      polls.append(1)
      if len(polls) == 6:
        self.params.put_bool("Mapd_ClearCache", True)
      if self.params.get("Mapd_ClearCache") and (not hasattr(mm, "offline_updater") or mm.offline_updater().ready_for_clear()):
        mm.clear_downloaded_maps()
        self.params.remove("Mapd_ClearCache")
    self._run(before_poll=main_loop_clear)
    for _ in range(3):
      main_loop_clear()
    assert not self.params.get("Mapd_ClearCache")
    assert all(self._disk(t) is None for t in CH), "maps came back after DELETE ALL"
    for sub in ("offline_prev", "tmp"):
      assert not os.path.exists(os.path.join(self.root, sub))

  def test_cleanup_of_old_mapd_data_leaves_the_update_files_alone(self, monkeypatch):
    self._setup(monkeypatch)
    for sub in ("offline_prev/journal.json", "tmp/offline/44/6.tar.gz"):
      os.makedirs(os.path.dirname(os.path.join(self.root, sub)), exist_ok=True)
      open(os.path.join(self.root, sub), "w").close()
    removed = mm.get_files_for_cleanup()
    assert not [f for f in removed if "offline" in f or "/tmp" in f]
    mm.clear_downloaded_maps()  # DELETE ALL takes the update's files too
    for sub in ("offline", "offline_prev", "tmp"):
      assert not os.path.exists(os.path.join(self.root, sub))
