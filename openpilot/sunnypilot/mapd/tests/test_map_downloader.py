"""E11 map_downloader (DESIGN 9.3 boot order, 9.5-9.9, 9.11): queue, checks, staging, pauses, guard, crash survival, status.
Real Params (test prefix), the fake tile server, a fake clock and fake device inputs; no device, no network."""
import errno
import os
import subprocess
import sys
import time

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.mapd import map_downloader as md
from openpilot.sunnypilot.mapd import offline_swap as sw
from openpilot.sunnypilot.mapd import tile_fetch as tf
from openpilot.sunnypilot.mapd.offline_regions import entry_tiles
from openpilot.sunnypilot.mapd.offline_storage import Disk
from openpilot.sunnypilot.mapd.tests.swap_world import hashes, live, read_tile
from openpilot.sunnypilot.mapd.tests.tile_server import FakeTileServer, Tile, area_files, make_archive, session

CH = entry_tiles("CH")
AT = entry_tiles("AT")
DAY = 86400
NOW = 1.8e9  # a valid wall clock (2027)
GB = 1_000_000_000


class Inputs:
  def __init__(self):
    self.dev = {"started": False, "network": ("wifi", False), "hot": False}

  def update(self):
    return dict(self.dev)


class Mem(dict):
  def __init__(self):
    super().__init__()
    self.puts = []

  def put(self, key, value, *a, **k):
    self[key] = value
    self.puts.append((key, value))

  def get(self, key, *a, **k):
    return super().get(key)


class Clock:
  def __init__(self):
    self.wall, self.mono = NOW, 1000.0

  def now(self):
    return self.wall

  def monotonic(self):
    return self.mono

  def sleep(self, s):
    self.mono += s


def serve(server, tiles, seed="", n=4, **kw):
  bodies = {}
  for t in tiles:
    files = area_files(t, n=n, seed=seed)
    body = make_archive(t, files)
    server.put(t, Tile(body=body, etag=f'"{t[0]}_{t[1]}_{seed}"', **kw))
    bodies[t] = files
  return bodies


class Base(OpenpilotTestCase):
  def setup_method(self):
    self.server = FakeTileServer()
    self.addCleanup(self.server.close)
    self.root = os.path.join(Params().get_param_path(), "..", "mapd_e11")
    self.root = os.path.abspath(self.root)
    os.makedirs(self.root, exist_ok=True)
    self.params = Params()
    self.params.put_bool("OsmAutoUpdate", True, block=True)
    self.inputs, self.clock, self.mem = Inputs(), Clock(), Mem()
    self.cleared = []
    self.capacity, self.available = 100 * GB, 50 * GB

  def make(self, marker=True, **kw):
    if marker:
      sw.write_swap_done(self.root, sw.NOTHING)
    d = md.MapDownloader(self.root, self.params, self.mem, self.inputs, tf.Fetcher(session=session(), base_url=self.server.url, timeout=(2, 3)),
                         disk=lambda r: Disk(self.capacity, self.available), drives=lambda: 10 * GB, meminfo=lambda: 2 * 1024 ** 3,
                         now=self.clock.now, monotonic=self.clock.monotonic, sleep=self.clock.sleep, boot_id=sw.boot_id,
                         clear_hook=lambda: self.cleared.append(1), **kw)
    return d

  def ready(self, d):
    """past the start pause"""
    d.step()
    self.clock.mono += md.START_PAUSE_S + 1
    return d

  def select(self, *ids):
    from openpilot.sunnypilot.mapd.offline_selection import Selection
    Selection(self.params).write([{"kind": "nation", "code": i} for i in ids])

  def steps(self, d, n):
    for _ in range(n):
      d.step()

  def results(self, d):
    return [f["result"] for name, f in d.events if name == "map_downloader_tile"]


class TestBootOrder(Base):
  def test_no_lock_work_before_the_swap_marker(self):
    calls = []
    real = sw.Lock.__enter__
    sw.Lock.__enter__ = lambda s: (calls.append(1), real(s))[1]
    self.addCleanup(setattr, sw.Lock, "__enter__", real)
    d = self.make(marker=False)
    self.select("CH")
    d.step()
    assert not d.boot_ready and calls == [] and d.waiting == md.W_START
    sw.write_swap_done(self.root, sw.APPLIED, 2)  # the swap at start finished
    d.step()
    assert d.boot_ready and calls and d.swap_result["result"] == sw.APPLIED

  def test_after_60_s_without_a_marker_it_records_the_missing_swap(self):
    sw.write_json_atomic(sw.next_path(self.root, sw.SWAP_DONE), {"boot_id": "an-earlier-boot", "result": "applied"})
    d = self.make(marker=False)
    d.step()
    assert not d.boot_ready
    self.clock.mono += md.SWAP_WAIT_S
    d.step()
    assert d.boot_ready and d.swap_result["result"] == sw.MISSING
    assert sw.read_json(sw.next_path(self.root, sw.SWAP_DONE))["boot_id"] == sw.boot_id()

  def test_manager_restart_within_the_same_boot(self):
    sw.write_swap_done(self.root, sw.APPLIED, 1)
    d = self.make(marker=False)
    d.step()
    assert d.boot_ready  # marker already carries this boot's id


class TestQueue(Base):
  def test_first_download_120s_pause_then_one_tile_at_a_time_staged_for_the_next_start(self):
    bodies = serve(self.server, CH)
    self.select("CH")
    d = self.make()
    self.steps(d, 3)
    assert not self.server.gets() and d.waiting == md.W_START
    self.clock.mono += md.START_PAUSE_S
    d.step()
    assert len(self.server.gets()) == 1  # one tile per step
    self.steps(d, 10)
    assert len(self.server.gets()) == 6 and sorted(sw.staged_tiles(self.root)) == sorted(CH)
    assert not sw.tiles_in(os.path.join(self.root, sw.OFFLINE))  # nothing live before the next start (owner Q1 a)
    st = self.params.get("OsmLocationStatus")
    assert st["version"] == 2 and st["entries"]["CH"]["state"] == "staged" and st["entries"]["CH"]["tiles_staged"] == 6
    sw.start_swap(self.root)
    for t in CH:
      assert read_tile(sw.tile_path(self.root, sw.OFFLINE, t)) == bodies[t]
    d2 = self.make(marker=False)
    d2.step()
    assert d2.status()["entries"]["CH"]["state"] == "ok" and d2.due() == []

  def test_shared_tiles_fetched_once_and_95_for_six_countries(self):
    self.select("CH", "AT")
    d = self.make()
    d.step()
    assert len(d.due()) == len(set(CH) | set(AT)) == 14
    self.select("CH", "AT", "DE", "IT", "FR", "SI")
    assert len(d.due()) == 95

  def test_order_new_then_requested_then_aged(self):
    self.select("CH")
    for t in CH[:4]:
      live(self.root, t, area_files(t), {"fetched_at": NOW - 9 * DAY, "checked_at": NOW - (9 + CH.index(t)) * DAY, "etag": '"x"'})
    d = self.make()
    d.step()
    d.store.add_requests([CH[1]], NOW, "request")
    q = d.due()
    assert q[:2] == CH[4:6]  # not on disk
    assert q[2] == CH[1]  # requested
    assert q[3:] == [CH[3], CH[2], CH[0]]  # aged, oldest check first
    self.params.put_bool("OsmAutoUpdate", False, block=True)
    assert d.due() == CH[4:6] + [CH[1]]

  def test_onroad_request_survives_a_power_cut(self):
    """The photo case: DOWNLOAD tapped with the car on; E9 kept it in memory and lost it at switch-off."""
    self.select("CH")
    for t in CH:
      live(self.root, t, area_files(t), {"fetched_at": NOW, "checked_at": NOW, "etag": '"x"'})
    self.inputs.dev["started"] = True
    d = self.make()
    d.step()
    self.params.put("OsmDownloadRequest", "CH", block=True)
    d.step()
    assert set(sw.read_json(sw.next_path(self.root, sw.REQUESTS))) == {sw.key_of(t) for t in CH}
    d2 = self.make(marker=False)  # power cut, next start
    d2.step()
    assert d2.due() == CH and d2.status()["entries"]["CH"]["state"] == "queued"

  def test_update_all_and_db_updates_check(self):
    self.select("CH", "AT")
    for t in set(CH) | set(AT):
      live(self.root, t, area_files(t), {"fetched_at": NOW, "checked_at": NOW, "etag": '"x"'})
    d = self.make()
    d.step()
    assert d.due() == []
    self.params.put_bool("OsmDbUpdatesCheck", True, block=True)
    d.step()
    assert len(d.due()) == 14 and not self.params.get_bool("OsmDbUpdatesCheck")

  def test_capped_while_driving(self):
    d = self.make()
    self.inputs.dev["started"] = True
    d.step()
    assert d.rate() == md.CAP_BYTES_PER_S == 1_000_000
    self.inputs.dev["started"] = False
    d.step()
    assert d.rate() is None


class TestChecks(Base):
  def setup_tile(self, entry, files=None):
    self.select("CH")
    t = CH[0]
    for other in CH[1:]:
      live(self.root, other, area_files(other), {"fetched_at": NOW, "checked_at": NOW, "etag": '"x"'})
    files = files or area_files(t)
    live(self.root, t, files, entry)
    return t, files

  def test_304_updates_checked_at_only(self):
    t, files = self.setup_tile({"fetched_at": NOW - 9 * DAY, "checked_at": NOW - 9 * DAY, "etag": '"same"'})
    self.server.put(t, Tile(body=make_archive(t, files), etag='"same"'))
    d = self.ready(self.make())
    d.step()
    assert self.results(d) == ["not_modified"] and d.store.live["tiles"]["44/6"]["checked_at"] == NOW
    assert not sw.staged_tiles(self.root) and d.due() == []

  def test_server_ignoring_304_then_head_path(self):
    t, files = self.setup_tile({"fetched_at": NOW - 9 * DAY, "checked_at": NOW - 9 * DAY, "etag": '"same"'})
    body = make_archive(t, files)
    self.server.put(t, Tile(body=body, etag='"same"', honour_304=False))
    d = self.ready(self.make())
    d.step()
    assert self.results(d) == ["ignored_304"] and d.store.live["hosts"]["127.0.0.1"]["no_304"]
    self.clock.wall += 8 * DAY
    d.step()
    assert self.results(d)[-1] == "same_head" and len(self.server.gets(t)) == 1 and len(self.server.heads(t)) == 1

  def test_pre_e11_tile_without_validators_is_fetched_and_identical_content_not_staged(self):
    t, files = self.setup_tile({"fetched_at": NOW - 30 * DAY, "checked_at": None})
    self.server.put(t, Tile(body=make_archive(t, files), etag='"e"'))
    d = self.ready(self.make())
    d.step()
    assert self.results(d) == ["identical"] and not sw.staged_tiles(self.root)
    e = d.store.live["tiles"]["44/6"]
    assert e["checked_at"] == NOW and e["etag"] == '"e"' and sw.read_json(sw.live_hashes_path(self.root, t))["files"] == hashes(files)

  def test_changed_content_is_staged(self):
    t, _ = self.setup_tile({"fetched_at": NOW - 9 * DAY, "checked_at": NOW - 9 * DAY, "etag": '"old"'})
    new = area_files(t, seed="new")
    self.server.put(t, Tile(body=make_archive(t, new), etag='"new"'))
    d = self.ready(self.make())
    d.step()
    assert self.results(d) == ["staged"] and read_tile(sw.tile_path(self.root, sw.READY_SUB, t)) == new

  def test_404_on_a_live_tile_keeps_it(self):
    t, files = self.setup_tile({"fetched_at": NOW - 9 * DAY, "checked_at": NOW - 9 * DAY, "etag": '"old"'})
    self.server.put(t, Tile(status=404))
    d = self.ready(self.make())
    d.step()
    assert self.results(d) == ["no_data"] and read_tile(sw.tile_path(self.root, sw.OFFLINE, t)) == files
    assert d.store.live["tiles"]["44/6"]["server_404_at"] == NOW and d.due() == []

  def test_404_on_a_new_tile_is_no_data_and_not_asked_again_until_aged(self):
    self.select("CH")
    serve(self.server, CH[1:])
    d = self.ready(self.make())
    self.steps(d, 8)
    assert "44/6" in d.store.live["no_data"] and d.due() == []
    self.clock.wall += 8 * DAY
    assert CH[0] in d.due()

  def test_server_error_backs_off_and_a_request_asks_at_once(self):
    self.select("CH")
    serve(self.server, CH, status=403)
    d = self.ready(self.make())
    d.step()
    assert self.results(d) == ["server_error"] and d.server_error == 403
    d.step()
    assert len(self.server.gets()) == 1 and d.waiting == md.W_SERVER_ERROR
    st = self.params.get("OsmLocationStatus")
    assert st["run"]["server_error"] == 403 and st["run"]["retry_at"] > 0
    self.params.put("OsmDownloadRequest", "CH", block=True)
    d.step()
    assert len(self.server.gets()) == 2

  def test_bad_tile_backs_off_alone(self):
    self.select("CH")
    serve(self.server, CH[1:])
    self.server.put(CH[0], Tile(body=b"<html>"))
    d = self.ready(self.make())
    self.steps(d, 8)
    assert self.results(d).count("bad") == 1 and len(sw.staged_tiles(self.root)) == 5
    assert not os.path.exists(d.store.part(CH[0])) and not os.path.exists(d.store.work_dir(CH[0]))


class TestShrinkGuard(Base):
  def setup_shrink(self, live_n, new_n, etag='"small"'):
    self.select("CH")
    t = CH[0]
    for other in CH[1:]:
      live(self.root, other, area_files(other), {"fetched_at": NOW, "checked_at": NOW, "etag": '"x"'})
    live(self.root, t, area_files(t, n=live_n), {"fetched_at": NOW - 9 * DAY, "checked_at": NOW - 9 * DAY, "etag": '"big"'})
    new = area_files(t, n=new_n, seed="s") if new_n else {}
    self.server.put(t, Tile(body=make_archive(t, new), etag=etag))
    return t

  def test_25_percent_fewer_files_kept_with_record_and_status(self):
    t = self.setup_shrink(8, 6)
    d = self.ready(self.make())
    d.step()
    assert self.results(d) == ["kept_smaller"] and not sw.staged_tiles(self.root)
    rec = d.store.live["shrink"]["44/6"]
    assert rec["files"] == 6 and rec["first_seen"] == NOW
    assert self.params.get("OsmLocationStatus")["entries"]["CH"]["kept_smaller"] == [{"tile": "44/6", "pct": 25}]
    assert len(read_tile(sw.tile_path(self.root, sw.OFFLINE, t))) == 8

  def test_10_percent_fewer_is_staged(self):
    self.setup_shrink(10, 9)
    d = self.ready(self.make())
    d.step()
    assert self.results(d) == ["staged"]

  def test_zero_files_kept(self):
    self.setup_shrink(8, 0)
    d = self.ready(self.make())
    d.step()
    assert self.results(d) == ["kept_smaller"]

  def test_r22_same_archive_after_7_days_is_applied(self):
    self.setup_shrink(8, 6)
    d = self.ready(self.make())
    d.step()
    self.clock.wall += 7 * DAY + 60
    d.step()
    assert self.results(d) == ["kept_smaller", "staged"] and "44/6" not in d.store.live["shrink"]
    assert any(n == "map_downloader_shrink_applied" for n, _ in d.events)

  def test_r22_before_7_days_still_kept(self):
    self.setup_shrink(8, 6)
    d = self.ready(self.make())
    d.step()
    self.params.put("OsmDownloadRequest", "CH", block=True)  # UPDATE re-checks at once
    self.clock.wall += 2 * DAY
    self.steps(d, 7)
    mine = [f["result"] for n, f in d.events if n == "map_downloader_tile" and f["tile"] == "44/6"]
    assert mine == ["kept_smaller", "kept_smaller"] and d.store.live["shrink"]["44/6"]["first_seen"] == NOW

  def test_r22_size_change_beyond_2_percent_restarts_the_clock(self):
    t = self.setup_shrink(8, 6)
    d = self.ready(self.make())
    d.step()
    self.clock.wall += 7 * DAY + 60
    self.server.put(t, Tile(body=make_archive(t, area_files(t, n=5, seed="t")), etag='"smaller-still"'))
    d.step()
    rec = d.store.live["shrink"]["44/6"]
    assert self.results(d)[-1] == "kept_smaller" and rec["files"] == 5 and rec["first_seen"] == NOW + 7 * DAY + 60


class TestPauses(Base):
  def test_network_hot_memory(self):
    self.select("CH")
    serve(self.server, CH)
    d = self.ready(self.make())
    self.inputs.dev["network"] = ("cell4G", True)
    d.step()
    assert d.waiting == md.W_NETWORK and not self.server.gets()
    self.inputs.dev["network"] = ("cell4G", False)  # Wingo SIM with GsmMetered 0, roaming included (owner Q4 a)
    self.inputs.dev["hot"] = True
    d.step()
    assert d.waiting == md.W_HOT
    self.inputs.dev["hot"] = False
    d._meminfo = lambda: 100 * 1024 ** 2
    d.step()
    assert d.waiting == md.W_MEMORY and not self.server.gets()
    d._meminfo = lambda: 2 * 1024 ** 3
    d.step()
    assert len(self.server.gets()) == 1

  def test_onroad_edge_pauses_120s(self):
    self.select("CH")
    serve(self.server, CH)
    d = self.ready(self.make())
    d.step()
    self.inputs.dev["started"] = True
    d.step()
    n = len(self.server.gets())
    self.steps(d, 3)
    assert len(self.server.gets()) == n and d.waiting == md.W_START
    self.clock.mono += md.START_PAUSE_S
    d.step()
    assert len(self.server.gets()) == n + 1

  def test_network_lost_mid_transfer_keeps_the_part_and_resumes(self):
    self.select("CH")
    t = CH[0]
    files = {k: os.urandom(len(v)) for k, v in area_files(t, n=8, size=40000).items()}
    body = make_archive(t, files)
    self.server.put(t, Tile(body=body, etag='"v"', trickle=0.02))
    for other in CH[1:]:
      live(self.root, other, area_files(other), {"fetched_at": NOW, "checked_at": NOW, "etag": '"x"'})
    d = self.ready(self.make())
    real = d.tick
    t0 = time.monotonic()

    def tick():
      if time.monotonic() - t0 > 0.5:
        self.inputs.dev["network"] = ("none", False)
      real()
    d.tick = tick
    d.step()
    assert self.results(d) == ["aborted"] and 0 < os.path.getsize(d.store.part(t)) < len(body)
    d.tick = real
    self.inputs.dev["network"] = ("wifi", False)
    self.server.put(t, Tile(body=body, etag='"v"'))
    d.step()
    assert self.results(d)[-1] == "staged" and self.server.gets(t)[-1][2]["range"].startswith("bytes=")

  def test_delete_during_a_transfer_cancels_it(self):
    self.select("CH", "AT")
    t = CH[0]  # 44/6: Switzerland only
    body = make_archive(t, {k: os.urandom(len(v)) for k, v in area_files(t, n=8, size=40000).items()})
    self.server.put(t, Tile(body=body, etag='"v"', trickle=0.02))
    d = self.ready(self.make())
    real, t0 = d.tick, time.monotonic()

    def tick():
      if time.monotonic() - t0 > 0.4:
        self.params.put("OsmDeleteLocation", "CH", block=True)
      real()
    d.tick = tick
    d.step()
    assert self.results(d) == ["aborted"]
    d.tick = real
    d.step()
    assert not os.path.exists(d.store.part(t)) and t not in d.due()
    assert [e["code"] for e in d.selection.read()] == ["AT"]

  def test_storage_guard_counts_staged_and_sizes_for_full_recording(self):
    self.select("CH")
    serve(self.server, CH)
    self.capacity, self.available = 50 * GB, 5 * GB  # other = 50 - 5 - 10 (drives) = 35 GB; floor 5.4 GB; buffer 29.6 GB
    d = self.ready(self.make())
    d.step()
    assert d.waiting == md.W_STORAGE and "full recording" in d.message and not self.server.gets()

  def test_enospc_removes_the_work_and_pauses(self):
    self.select("CH")
    serve(self.server, CH)
    d = self.ready(self.make())

    def full(*a, **k):
      raise OSError(errno.ENOSPC, "No space left on device")
    d.fetcher.check = full
    d.step()
    assert d.message == "not enough storage" and d.waiting == "" or d.pause_until > self.clock.mono
    assert not os.path.exists(d.store.work_dir(CH[0]))


class TestDeleteClearSwap(Base):
  def test_delete_keeps_shared_tiles_and_removes_staged_copies(self):
    self.select("CH", "AT")
    for t in set(CH) | set(AT):
      live(self.root, t, area_files(t), {"fetched_at": NOW, "checked_at": NOW, "etag": '"x"'})
    from openpilot.sunnypilot.mapd.tests.swap_world import stage
    stage(self.root, CH[0], area_files(CH[0], seed="n"))
    d = self.ready(self.make())
    self.params.put("OsmDeleteLocation", "CH", block=True)
    d.step()
    on_disk = set(sw.tiles_in(os.path.join(self.root, sw.OFFLINE)))
    assert on_disk == set(AT) and not sw.staged_tiles(self.root)
    assert not os.listdir(sw.next_path(self.root, sw.OLD))

  def test_clear_all(self):
    self.select("CH")
    for t in CH:
      live(self.root, t, area_files(t), {"fetched_at": NOW, "checked_at": NOW, "etag": '"x"'})
    d = self.ready(self.make())
    self.params.put_bool("Mapd_ClearCache", True, block=True)
    d.step()
    assert self.cleared == [1] and not sw.tiles_in(os.path.join(self.root, sw.OFFLINE))
    assert d.selection.read() == [] and not self.params.get_bool("Mapd_ClearCache")
    assert not os.path.exists(sw.live_manifest_path(self.root))

  def test_offroad_swap_on_the_edge(self):
    self.select("CH")
    for t in CH:
      live(self.root, t, area_files(t), {"fetched_at": NOW, "checked_at": NOW, "etag": '"x"'})
    from openpilot.sunnypilot.mapd.tests.swap_world import stage
    stage(self.root, CH[0], area_files(CH[0], seed="n"))
    self.inputs.dev["started"] = True
    d = self.ready(self.make())
    d.step()
    assert sw.staged_tiles(self.root)  # never while driving
    self.inputs.dev["started"] = False
    d.step()
    assert not sw.staged_tiles(self.root) and read_tile(sw.tile_path(self.root, sw.OFFLINE, CH[0])) == area_files(CH[0], seed="n")


class TestRobustness(Base):
  def test_a_failing_step_is_logged_and_backs_off(self):
    d = self.make()
    d.step()
    boom = [True]

    def next_tile():
      if boom[0]:
        raise RuntimeError("boom")
      return None
    d.next_tile = next_tile
    d.step_safe()
    assert d.events[-1][0] == "map_downloader_step_failed" and d.events[-1][1]["delay"] == md.CRASH_BACKOFF_S
    d.step_safe()  # within the backoff: only the heartbeat
    assert sum(1 for n, _ in d.events if n == "map_downloader_step_failed") == 1
    self.clock.mono += md.CRASH_BACKOFF_S
    d.step_safe()
    assert d.events[-1][1]["delay"] == 2 * md.CRASH_BACKOFF_S
    boom[0] = False
    self.clock.mono += 2 * md.CRASH_BACKOFF_S
    d.step_safe()
    assert d._crashes == 0

  def test_heartbeat_in_memory_params_status_written_only_on_change(self):
    d = self.make()
    writes = []
    real = self.params.put
    self.params.put = lambda k, v, *a, **kw: (writes.append(k), real(k, v, *a, **kw))
    d.step()
    beat = self.mem[md.LIVE_KEY]["heartbeat"]
    self.clock.mono += 5
    self.steps(d, 3)
    assert self.mem[md.LIVE_KEY]["heartbeat"] > beat
    assert writes.count("OsmLocationStatus") == 1  # the persistent status only when it changed (R2-3)
    assert "heartbeat" not in str(self.params.get("OsmLocationStatus"))

  def test_apply_failed_after_two_starts_with_staged_tiles(self):
    from openpilot.sunnypilot.mapd.tests.swap_world import stage
    stage(self.root, CH[0], area_files(CH[0]))
    sw.write_json_atomic(sw.next_path(self.root, sw.SWAP_DONE), {"boot_id": sw.boot_id(), "result": "skipped", "failed_starts": 2})
    d = self.make(marker=False)
    d.step()
    assert self.params.get("OsmLocationStatus")["run"]["apply_failed"]


def test_priority_reads_back_in_a_child():
  code = "; ".join(["import os, ctypes", "from openpilot.sunnypilot.mapd.map_downloader import Priority", "r = Priority().start()",
                     "nr = {'x86_64': 252, 'aarch64': 31}[os.uname().machine]", "io = ctypes.CDLL(None).syscall(nr, 1, 0)",
                     "print(os.sched_getscheduler(0) == os.SCHED_IDLE, io >> 13, io & 7)"])
  out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=os.environ).stdout.split()
  assert out == ["True", "2", "7"], out


def test_affinity_core_6_onroad_else_0_to_3(monkeypatch):
  set_to = []
  monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0})
  monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: set_to.append(set(cpus)))
  p = md.Priority()
  monkeypatch.setattr(md.Priority, "online_cpus", staticmethod(lambda path="": set(range(8))))
  assert p.affinity() == {6}
  monkeypatch.setattr(md.Priority, "online_cpus", staticmethod(lambda path="": {0, 1, 2, 3}))
  assert p.affinity() == {0, 1, 2, 3}
  assert set_to == [{6}, {0, 1, 2, 3}]


def test_online_cpus_parsing(tmp_path):
  f = tmp_path / "online"
  f.write_text("0-3,6-7\n")
  assert md.Priority.online_cpus(str(f)) == {0, 1, 2, 3, 6, 7}


def test_network_rule():
  assert md.network_allows(("wifi", False)) and md.network_allows(("cell4G", False))
  assert not md.network_allows(("cell4G", True)) and not md.network_allows(("none", False)) and not md.network_allows(None)


class TestStoreOrder(Base):
  """Review finding 1 (reviewer's probe review_probe_test_stale_store.py): the process is built before the swap at start ends;
  the store must be read again under the lock, or the stale copy is saved later."""

  def test_migration_requests_survive_when_downloader_starts_first(self):
    from openpilot.sunnypilot.mapd.tests.swap_world import content, write_tile
    self.select("CH")
    for t in CH[:3]:
      write_tile(sw.tile_path(self.root, sw.OFFLINE, t), content(t, "old"))
    d = self.make(marker=False)  # process started before the swap
    sw.start_swap(self.root)  # first E11 start: migration requests + manifest
    assert len(sw.read_json(sw.next_path(self.root, sw.REQUESTS))) == 3
    d.step()
    assert d.boot_ready
    d.store.add_requests([CH[5]], NOW, "request")  # any later save of the requests
    after = sw.read_json(sw.next_path(self.root, sw.REQUESTS))
    assert all(sw.key_of(t) in after for t in CH[:3]), after
    assert "migrated_at" in sw.read_json(sw.live_manifest_path(self.root))

  def test_applied_tile_keeps_its_entry_after_the_next_save(self):
    from openpilot.sunnypilot.mapd.tests.swap_world import content, stage
    self.select("CH")
    t, u = CH[0], CH[1]
    live(self.root, t, content(t, "old"))
    live(self.root, u, content(u, "old"))
    stage(self.root, t, content(t, "new"))
    d = self.make(marker=False)
    sw.start_swap(self.root)
    d.step()
    assert sw.read_json(sw.live_manifest_path(self.root))["tiles"][sw.key_of(t)]["etag"] == f'"{t}"'
    d.store.mark_checked(u, NOW + 5)  # any later save of the live manifest
    assert sw.read_json(sw.live_manifest_path(self.root))["tiles"][sw.key_of(t)]["etag"] == f'"{t}"'


class TestLowItems(Base):
  def test_storage_figures_go_to_memory_params_not_the_persistent_status(self):
    """Review 9a: available space changes with every recorded minute; the persistent status must not follow it."""
    self.select("CH")
    for t in CH:
      live(self.root, t, area_files(t), {"fetched_at": NOW, "checked_at": NOW, "etag": '"x"'})
    d = self.ready(self.make())
    writes = []
    real = self.params.put
    self.params.put = lambda k, v, *a, **kw: (writes.append(k), real(k, v, *a, **kw))
    for _ in range(5):
      self.available -= GB  # drives being recorded
      self.clock.mono += md.STORAGE_REFRESH_S + 1
      d.step()
    assert writes.count("OsmLocationStatus") <= 1 and "storage" not in self.params.get("OsmLocationStatus")
    assert self.mem[md.LIVE_KEY]["storage"]["available"] == self.available

  def test_heartbeat_continues_during_a_long_delete_all(self):
    """Review 9b: a DELETE ALL of many tiles must not look like a stopped downloader."""
    self.select("CH", "AT")
    for t in set(CH) | set(AT):
      live(self.root, t, area_files(t), {"fetched_at": NOW, "checked_at": NOW, "etag": '"x"'})
    d = self.ready(self.make())
    real_rmtree = sw._rmtree

    def slow_rmtree(path):
      self.clock.mono += 20.0  # each folder takes 20 s
      real_rmtree(path)
    sw._rmtree = slow_rmtree
    self.addCleanup(setattr, sw, "_rmtree", real_rmtree)
    self.params.put_bool("Mapd_ClearCache", True, block=True)
    self.mem.puts.clear()
    d.step()
    beats = [v["heartbeat"] for k, v in self.mem.puts if k == md.LIVE_KEY]
    assert len(beats) >= 3 and max(b - a for a, b in zip(beats, beats[1:], strict=False)) <= 45.0, beats

  def test_first_hash_of_a_pre_e11_tile_runs_outside_the_lock(self):
    """Review 9c: the full-tile hash of a tile without a hash list is not taken while the store lock is held."""
    import fcntl
    from openpilot.sunnypilot.mapd import map_store
    self.select("CH")
    t = CH[0]
    files = area_files(t)
    for other in CH[1:]:
      live(self.root, other, area_files(other), {"fetched_at": NOW, "checked_at": NOW, "etag": '"x"'})
    live(self.root, t, files, {"fetched_at": NOW - 30 * DAY, "checked_at": None})
    os.remove(sw.live_hashes_path(self.root, t))  # pre-E11: no hash list
    self.server.put(t, Tile(body=make_archive(t, files), etag='"e"'))
    held = []
    real = map_store.hash_tree

    def watched(folder):
      fd = os.open(sw.next_path(self.root, sw.LOCK), os.O_RDWR | os.O_CREAT)
      try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held.append(False)
      except BlockingIOError:
        held.append(True)
      finally:
        os.close(fd)
      return real(folder)
    map_store.hash_tree = watched
    self.addCleanup(setattr, map_store, "hash_tree", real)
    d = self.ready(self.make())
    d.step()
    assert self.results(d) == ["identical"] and held == [False]
