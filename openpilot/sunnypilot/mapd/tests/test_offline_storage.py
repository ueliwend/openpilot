"""Storage guard and deleter replay (DESIGN.md rev 4 section 3; owner: the deleter may always act, 3 h drive buffer).
E11 (DESIGN rev 2 section 4.6, owner Q5 a): the buffer is sized at the current recording rate, but at least full road recording."""
import os
from pathlib import Path

from openpilot.common.hardware.hw import Paths
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.mapd import offline_storage as st
from openpilot.sunnypilot.mapd.offline_server import KnownSizes
from openpilot.sunnypilot.mapd import offline_swap as sw
from openpilot.system.loggerd import deleter

GB = 1_000_000_000
DAY = 86400.0
T0 = 1_762_000_000.0  # 1 Nov 2025


def test_buffer_is_three_hours_of_full_recording():
  assert st.MIN_DRIVE_BUFFER_BYTES == int(3 * 60 * 164.28e6)  # 29.6 GB kept free for drives (owner Q5 a, 13 Nov 2026)


class Rec:
  def __init__(self, road_full, front):
    self.v = {"RecordRoadFull": road_full, "RecordFront": front}

  def get_bool(self, k):
    return self.v[k]


def test_recording_rate_from_the_settings():
  assert abs(st.recording_mb_per_min(Rec(False, False)) - 14.28) < 1e-9
  assert abs(st.recording_mb_per_min(Rec(True, False)) - 164.28) < 1e-9
  assert abs(st.recording_mb_per_min(Rec(True, True)) - 239.18) < 1e-9
  assert st.guard_mb_per_min(14.28) == 164.28 and abs(st.guard_mb_per_min(239.18) - 239.18) < 1e-9


def test_guard_sized_for_full_recording_while_preview_only():
  cap = 95 * GB
  disk = st.Disk(cap, 50 * GB)
  preview = st.budget(disk, 30 * GB, 0, 0, 0, 14.28)
  full = st.budget(disk, 30 * GB, 0, 0, 0, 164.28)
  assert preview.max_maps == full.max_maps  # Q5 a: switching full recording on later still leaves 3 h
  both = st.budget(disk, 30 * GB, 0, 0, 0, 239.18)
  assert both.max_maps < full.max_maps


def test_screen_figure_for_the_photo():
  """The photo's "about 5.0 h" used 239.18 MB/min; at the 14.28 MB/min the device records it is ~84 h, 7 h with full recording."""
  room = 5.0 * 60 * 239.18e6
  cap = 100 * GB
  floor = st.floor_bytes(cap)
  b = st.budget(st.Disk(cap, cap), 0, 0, 0, 0, 14.28)
  b = st.budget(st.Disk(cap, cap), 0, int(cap - floor - room), int(cap - floor - room), 0, 14.28)
  assert round(b.buffer_hours_now) == 84 and round(b.buffer_hours_full, 1) == 7.3


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


def test_known_sizes_from_the_store_and_e9_cache(tmp_path):
  root = str(tmp_path)
  sizes = KnownSizes(root)
  assert sizes.ratio() == st.DEFAULT_UNPACK_RATIO and sizes.download((44, 6)) is None
  m = sw.empty_manifest()
  m["tiles"]["44/6"] = {"download_bytes": 100, "disk_bytes": 300}
  m["no_data"]["44/8"] = {"checked_at": 1}
  sw.write_json_atomic(sw.live_manifest_path(root), m)
  sw.write_json_atomic(os.path.join(root, sw.SIZES_NAME), {"version": 1, "tiles": {"46/6": {"download": 50}}})
  sizes = KnownSizes(root)
  assert sizes.ratio() == 3.0 and sizes.no_data((44, 8)) and sizes.download((46, 6)) == 50 and sizes.download((44, 6)) == 100


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
