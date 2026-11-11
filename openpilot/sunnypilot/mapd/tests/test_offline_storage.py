"""Storage guard and deleter replay (DESIGN.md rev 4 section 3; owner: the deleter may always act, 3 h drive buffer)."""
import os
from pathlib import Path

from openpilot.common.hardware.hw import Paths
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.mapd import offline_storage as st
from openpilot.sunnypilot.mapd.offline_server import SizeCache
from openpilot.system.loggerd import deleter

GB = 1_000_000_000
DAY = 86400.0
T0 = 1_762_000_000.0  # 1 Nov 2025


def test_buffer_is_three_hours_at_the_measured_rate():
  assert st.MIN_DRIVE_BUFFER_BYTES == int(3 * 60 * 239.18e6)


def test_budget_on_a_full_disk_passes_and_computes_the_shortfall():
  cap = 95 * GB
  floor = st.floor_bytes(cap)
  assert floor == int(9.5 * GB)
  disk = st.Disk(cap, floor + int(0.1 * GB))  # drives keep the disk just above the deleter's floor
  b = st.budget(disk, drives=75 * GB, maps_now=int(0.5 * GB), maps_after=int(6 * GB), peak=1 * GB)
  assert b.ok
  assert b.shortfall == int(6 * GB) + GB - int(0.5 * GB) - int(0.1 * GB)
  assert b.other == cap - disk.available - 75 * GB - int(0.5 * GB)
  assert b.max_maps == cap - floor - b.other - st.MIN_DRIVE_BUFFER_BYTES


def test_budget_fails_when_maps_and_buffer_exceed_capacity():
  cap = 95 * GB
  disk = st.Disk(cap, st.floor_bytes(cap) + GB)
  other = 7 * GB
  drives = cap - disk.available - other
  allowed = cap - st.floor_bytes(cap) - other - st.MIN_DRIVE_BUFFER_BYTES
  assert st.budget(disk, drives, 0, allowed, 0).ok
  assert not st.budget(disk, drives, 0, allowed + 1, 0).ok
  assert not st.budget(disk, drives, 0, allowed - GB, GB + 1).ok  # the peak counts


def test_estimate_counts_unknown_and_no_data_tiles():
  est = st.estimate_tiles({(44, 6): 400}, {(44, 8): 100, (44, 10): None, (46, 6): 0}, ratio=4.0)
  assert est.on_disk == 400 and est.no_data == 1 and est.unknown == 1
  assert est.new == 400 + st.UNKNOWN_TILE_DOWNLOAD_BYTES * 4
  assert est.largest == st.UNKNOWN_TILE_DOWNLOAD_BYTES * 4 and est.largest_download == st.UNKNOWN_TILE_DOWNLOAD_BYTES
  assert st.peak_bytes(est) == 2 * est.largest + est.largest_download


def test_size_cache_ratio_and_stale_404(tmp_path):
  cache = SizeCache(str(tmp_path))
  assert cache.ratio() == st.DEFAULT_UNPACK_RATIO
  cache.set_download((44, 6), 100)
  cache.set_disk((44, 6), 300)
  cache.set_download((44, 8), 0)
  assert cache.ratio() == 3.0 and cache.no_data((44, 8))
  cache.set_disk((44, 8), 50)  # a tile on disk has data after all
  assert not cache.no_data((44, 8))
  cache.save()
  assert SizeCache(str(tmp_path)).tiles == cache.tiles
  with open(cache.path, "w") as f:
    f.write("{bad")
  assert SizeCache(str(tmp_path)).tiles == {}


class TestDeleterReplay(OpenpilotTestCase):
  def setup_method(self):
    self.root = Paths.log_root()
    os.makedirs(self.root, exist_ok=True)

  def make(self, name, size, written, lock=False, root=None):
    d = os.path.join(root or self.root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "rlog.zst"), "wb") as f:
      f.write(b"x" * size)
    if lock:
      open(os.path.join(d, "rlog.lock"), "w").close()
    for n in os.listdir(d):
      os.utime(os.path.join(d, n), (written, written))

  def test_order_sizes_and_dates_match_the_deleter(self, monkeypatch):
    # route counter order: 00000001 is the oldest drive
    for seg in range(4):
      self.make(f"00000001--aaaaaaaaaa--{seg}", 100, T0 + seg * 60)
    self.make("00000002--bbbbbbbbbb--0", 100, T0 + DAY, lock=True)  # being uploaded: skipped
    self.make("00000002--bbbbbbbbbb--1", 300, T0 + DAY + 60)
    self.make("boot", 50, T0)
    self.make("crash", 50, T0)
    bookmarked = {"00000001--aaaaaaaaaa--3"}  # preserved with the two before it
    monkeypatch.setattr(deleter, "has_preserve_xattr", lambda d: d in bookmarked)

    r = st.replay_deleter(150)
    assert r.folder_names == ["00000001--aaaaaaaaaa--0", "00000002--bbbbbbbbbb--1"] and r.enough and r.freed == 400
    r = st.replay_deleter(10_000)  # preserved segments after the others, boot / crash logs last, the locked one never
    assert r.folder_names == ["00000001--aaaaaaaaaa--0", "00000002--bbbbbbbbbb--1", "00000001--aaaaaaaaaa--1",
                              "00000001--aaaaaaaaaa--2", "00000001--aaaaaaaaaa--3", "boot", "crash"]
    assert not r.enough and r.minutes == 5 and r.folders == 7 and r.freed == 100 + 300 + 3 * 100 + 2 * 50
    assert r.first_written == T0 and r.last_written == T0 + DAY + 60
    r = st.replay_deleter(0)
    assert r.folders == 0 and r.enough

  def test_dates_come_from_file_times(self):
    self.make("00000005--cccccccccc--0", 100, T0 + 3 * DAY)
    self.make("00000006--dddddddddd--0", 100, T0)
    r = st.replay_deleter(150)
    assert r.minutes == 2 and r.first_written == T0 and r.last_written == T0 + 3 * DAY

  def test_external_drive_moves_and_loses_only_what_it_deletes(self, monkeypatch):
    ext = os.path.join(os.path.dirname(self.root), "external")
    os.makedirs(ext)
    self.make("00000001--aaaaaaaaaa--0", 100, T0)
    self.make("00000001--aaaaaaaaaa--1", 100, T0 + 60)
    self.make("00000000--eeeeeeeeee--0", 70, T0 - DAY, root=ext)
    real_is_mount = Path.is_mount
    monkeypatch.setattr(Path, "is_mount", lambda p: str(p).rstrip("/") == ext or real_is_mount(p))
    cap = 100 * GB
    disk = {ext: st.Disk(cap, st.floor_bytes(cap) + 50)}  # half a folder above its floor: short before the second move
    real_usage = st.disk_usage
    monkeypatch.setattr(st, "disk_usage", lambda p: disk.get(p) or real_usage(p))
    r = st.replay_deleter(150, external_root=ext)
    assert r.moved and r.minutes == 2 and r.freed == 200
    assert r.lost_minutes == 1 and r.lost_first_written == T0 - DAY  # the external drive's oldest segment goes
    disk[ext] = st.Disk(cap, cap // 2)
    r = st.replay_deleter(150, external_root=ext)
    assert r.moved and r.lost_minutes == 0

  def test_plan_runs_the_replay_only_when_the_budget_holds(self):
    self.make("00000001--aaaaaaaaaa--0", 100, T0)
    root = os.path.join(os.path.dirname(self.root), "osm")
    os.makedirs(root)
    est = st.estimate_tiles({}, {(44, 6): 1000}, 4.0)
    cap = 100 * GB
    plan = st.plan_download(root, est, disk=st.Disk(cap, st.floor_bytes(cap) + 10), drives=50 * GB)
    assert plan.budget.ok and plan.budget.shortfall > 0 and plan.replay.minutes == 1
    plan = st.plan_download(root, est, disk=st.Disk(50 * GB, st.floor_bytes(50 * GB) + 10), drives=0)
    assert not plan.budget.ok and plan.replay.folders == 0
