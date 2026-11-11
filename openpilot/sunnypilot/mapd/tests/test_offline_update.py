"""Offline map update protection against mapd v1.12.0's re-download corruption, against the fake mapd (fake_mapd.py)."""
import json
import os
import random

import pytest

import openpilot.sunnypilot.mapd.offline_update as ou
from openpilot.sunnypilot.mapd.offline_regions import tile_archive, tile_dir
from openpilot.sunnypilot.mapd.tests.fake_mapd import NET_ERROR, NOT_FOUND, Cut, gzip_bytes, make_tar, make_tile, write_param
from openpilot.sunnypilot.mapd.tests.offline_world import SimulatedCrash, World, area_files, ch_tiles

T = ch_tiles()  # Switzerland: 6 tiles


@pytest.fixture
def world(monkeypatch):
  w = World(monkeypatch)
  yield w
  w.close()


def setup_update(w, old_idx=(0, 1, 2, 3, 4), new_idx=(0, 1, 2, 3, 4, 5)):
  old = {T[i]: area_files(T[i], "old") for i in old_idx}
  new = {T[i]: area_files(T[i], "new") for i in new_idx}
  w.seed(old)
  w.serve(new)
  return old, new


def assert_consistent(w, old, new):
  """No leftovers; a tile that had data is exactly its old or exactly its new copy (never mixed, never a stale tail);
  a first download holds only prefixes of the new files (a cut-short first download reads as no data, not as wrong data)."""
  assert w.leftovers() == []
  for t in T:
    disk = w.disk(t)
    if t in old:
      assert disk == old[t] or (t in new and disk == new[t]), t
    elif disk is not None:
      for name, data in disk.items():
        assert name in new[t] and new[t][name].startswith(data), (t, name)


def test_fake_mapd_reproduces_the_corruption(world):
  """Sanity check of the fake: without the manager, a shorter new tile keeps the old file's tail (download.go:240)."""
  old, new = setup_update(world)
  write_param(world.request_path, json.dumps({"nations": ["CH"], "states": []}).encode())
  while world.mapd_step() != ("heartbeat",):
    pass
  for i in range(5):
    disk = world.disk(T[i])
    for name, data in new[T[i]].items():
      assert disk[name] != data and disk[name].startswith(data) and len(disk[name]) == len(old[T[i]][name])


def test_redownload_of_shorter_tiles_stays_readable(world):
  old, new = setup_update(world)
  world.run()  # manager start-up, then idle
  world.update()
  assert world.run()
  for t in T:
    assert world.disk(t) == new[t]
  assert world.leftovers() == [] and world.sent == [(["CH"], [])]


def test_no_network_keeps_every_old_tile(world):
  old, _ = setup_update(world)
  world.serve(dict.fromkeys(T, NET_ERROR))
  world.run()
  world.update()
  assert world.run()
  for t in T:
    assert world.disk(t) == old.get(t)
  assert world.leftovers() == []


def test_partial_download(world):
  """Some tiles arrive, others fail: no network for one, a cut connection, a 404, a broken archive, fewer files."""
  old, new = setup_update(world)
  broken = gzip_bytes(make_tar(T[2], new[T[2]])[:3 * 512])  # a valid gzip stream of a tar that ends after the first file
  fewer = dict(list(new[T[4]].items())[:2])
  world.serve({T[1]: NET_ERROR, T[2]: broken, T[3]: Cut(make_tile(T[3], new[T[3]]), 100),
               T[4]: fewer, T[5]: NOT_FOUND})
  world.run()
  world.update()
  assert world.run()
  assert world.disk(T[0]) == new[T[0]]
  for i in (1, 2, 3, 4):
    assert world.disk(T[i]) == old[T[i]], i  # 4: the new copy lost a file (superset rule)
  assert world.disk(T[5]) is None
  assert world.leftovers() == []


@pytest.mark.parametrize("failed", [{1, 2}, {1}], ids=["disk_full_rest_of_tile", "single_empty_file"])
def test_empty_files_in_a_new_copy_keep_the_old_copy(world, failed):
  """Disk full while extracting: mapd creates the remaining files, cannot write them and counts the tile as done, so the
  new copy has every old file name. An empty area file is never real data (the server writes none), so the old copy stays."""
  old, new = setup_update(world)
  world.mapd.write_fails = {T[2]: failed}
  world.run()
  world.update()
  assert world.run()
  assert [e for e in world.events if e[0] == "file_write_failed"], "the fake must have failed the writes"
  assert world.disk(T[2]) == old[T[2]]
  for i in (0, 1, 3, 4, 5):
    assert world.disk(T[i]) == new[T[i]]
  assert world.leftovers() == []


def test_first_download_cut_short_is_kept_as_incomplete(world):
  _, new = setup_update(world, old_idx=())
  tar = make_tar(T[5], new[T[5]])
  world.serve({T[5]: gzip_bytes(tar[:5 * 512 + 100])})  # the tar breaks inside the second file
  world.run()
  world.update()
  assert world.run()
  disk = world.disk(T[5])
  assert disk and set(disk) <= set(new[T[5]]) and disk != new[T[5]]
  assert_consistent(world, {}, new)


def test_progress_is_reset_before_the_send(world):
  setup_update(world)
  world.serve(dict.fromkeys(T, NET_ERROR))
  world.run()
  write_param(world.progress_path, json.dumps({"total_files": 6, "downloaded_files": 6}).encode())
  world.update()
  assert world.run()
  assert world.link.downloaded_files() is None  # no tile succeeded, so no stale "6/6" either


def test_only_offroad(world):
  old, new = setup_update(world)
  world.run()
  world.offroad = False
  world.update()
  world.run(max_polls=20)
  assert world.sent == [] and world.leftovers() == [] and world.disk(T[0]) == old[T[0]]
  world.offroad = True
  assert world.run(until=lambda w: w.sent)
  world.offroad = False  # a drive that starts mid-download: the location already sent runs to its end
  assert world.run()
  assert all(world.disk(t) == new[t] for t in T)


def test_nothing_moves_while_mapd_does_not_answer(world):
  old, _ = setup_update(world)
  world.run()
  world.mapd_alive = False
  world.update()
  world.run(max_polls=10)
  assert world.sent == [] and world.leftovers() == [] and all(world.disk(t) == old.get(t) for t in T)
  world.mapd_alive = True
  assert world.run()
  assert world.sent == [(["CH"], [])]


def test_request_is_sent_right_after_a_heartbeat(world):
  """The send waits for mapd to re-create its request file, so mapd's next read (after its 1 s sleep) gets it."""
  setup_update(world)
  world.run()
  world.update()
  assert world.run(until=lambda w: w.sent)
  assert world.events[-1] == ("heartbeat",)
  world.mapd_steps(2)
  assert world.events[-1] == ("archive", T[0])  # read and started: not lost to the read/clear race


def test_unknown_location_is_sent_and_nothing_moves(world):
  old, _ = setup_update(world)
  world.run()
  world.update(code="XX")
  assert world.run()
  assert world.sent == [(["XX"], [])] and all(world.disk(t) == old.get(t) for t in T) and world.leftovers() == []


def test_panic_loop_is_stopped(world):
  """A 200 reply that is not gzip at tile 3: mapd panics and restarts the location from tile 1 for ever."""
  old, new = setup_update(world)
  world.serve({T[3]: b"<html>error</html>" * 20})
  world.run()
  world.update()
  assert world.run(events_per_poll=2)
  assert [e for e in world.events if e[0] == "panic"], "the fake must have panicked"
  for i in (0, 1, 2):
    assert world.disk(T[i]) == new[T[i]]
  for i in (3, 4):
    assert world.disk(T[i]) == old[T[i]]
  assert world.leftovers() == []
  gets = len(world.mapd.gets)
  world.mapd_steps(30)
  assert len(world.mapd.gets) == gets, "mapd must stop downloading once the request is cleared"


def test_a_tile_that_failed_before_a_panic_restart_is_not_written_into_its_old_copy(world):
  """Tile 2 fails (no network), tile 4 panics; in the restarted pass tile 2 arrives. Its old copy must not be back in
  offline/ by then, or the shorter new files would be written into it (old copies return only once mapd is idle)."""
  old, new = setup_update(world)
  good = dict(world.server)
  world.serve({T[1]: NET_ERROR, T[3]: b"<html>error</html>" * 20})
  world.run()
  world.update()

  def after_first_panic(w):
    if any(e[0] == "panic" for e in w.events) and w.server[T[1]] == NET_ERROR:
      w.server[T[1]] = good[T[1]]
    return False
  world.run(events_per_poll=2, until=after_first_panic, max_polls=400)
  assert world.run(events_per_poll=2)
  assert world.disk(T[1]) in (old[T[1]], new[T[1]])
  assert_consistent(world, old, new)


@pytest.mark.parametrize("body", [b"<html>error</html>" * 40, b"x"], ids=["error_page", "tiny_body"])
def test_panic_at_the_first_tile_is_stopped(world, body):
  """No progress and no earlier download file: seen as the first tile's download file rewritten without growing."""
  old, _ = setup_update(world)
  world.serve({T[0]: body})
  world.run()
  world.update()
  assert world.run(events_per_poll=2, max_polls=200)
  assert all(world.disk(t) == old.get(t) for t in T) and world.leftovers() == []
  gets = len(world.mapd.gets)
  world.mapd_steps(20)
  assert len(world.mapd.gets) == gets


def test_stall_clears_the_request_and_the_entry_ends_when_mapd_does(world):
  old, new = setup_update(world)
  world.run()
  world.update()
  assert world.run(until=lambda w: os.path.exists(tile_archive(w.root, T[2])) and w.updater.state == ou.RUNNING)
  world.clock.t += ou.STALL_SECONDS / 2
  world.updater.step()
  assert world.updater.state == ou.RUNNING
  world.clock.t += ou.STALL_SECONDS / 2 + 1  # no byte for 10 min
  world.updater.step()
  assert world.updater.state == ou.DRAINING and world.request_state() == "missing"
  assert world.run()  # mapd finishes the location it had started (no cancel in the binary), then counts as idle
  assert all(world.disk(t) == new[t] for t in T) and world.leftovers() == []


def test_a_slowly_growing_download_is_not_a_stall(world):
  setup_update(world)
  rnd = random.Random(7)
  big = {f"area{i}": rnd.randbytes(4000) for i in range(3)}  # incompressible: about 50 chunks
  world.serve({T[0]: make_tile(T[0], big)})
  world.run()
  world.update()
  assert world.run(until=lambda w: w.updater.state == ou.RUNNING and w.events[-1][0] == "chunk")
  for _ in range(8):
    world.clock.t += ou.STALL_SECONDS * 0.9  # one chunk every 9 min
    assert world.mapd_step()[0] == "chunk"
    world.updater.step()
    assert world.updater.state == ou.RUNNING


def test_peak_extra_space_is_one_tile_and_its_download_file(world):
  """Old copies go as soon as the new tile is complete: no doubling of the whole map during an update."""
  old, new = setup_update(world)
  world.run()
  world.update()
  old_total = sum(len(d) for files in old.values() for d in files.values())
  biggest_new = max(sum(len(d) for d in files.values()) for files in new.values())
  biggest_archive = max(len(world.server[t]) for t in T)

  def size():
    total = 0
    for d, _, files in os.walk(world.root):
      total += sum(os.path.getsize(os.path.join(d, f)) for f in files if not f.endswith(".json"))
    return total
  peak = 0

  def watch(w):
    nonlocal peak
    peak = max(peak, size())
    return False
  for _ in range(400):
    world.updater.step()
    watch(world)
    if world.updater.state == ou.IDLE and world.request_state() != "pending":
      break
    world.mapd_step()
    watch(world)
  assert all(world.disk(t) == new[t] for t in T)
  assert peak <= old_total + 2 * biggest_new + biggest_archive


def test_clear_all_waits_for_the_running_request(world):
  setup_update(world)
  world.run()
  assert world.updater.ready_for_clear()
  world.update()
  world.updater.queue.append(("nation", "AT"))
  assert world.run(until=lambda w: w.updater.state == ou.RUNNING)
  assert not world.updater.ready_for_clear() and world.updater.queue == []  # the rest is dropped at once
  assert world.run()
  assert world.updater.ready_for_clear()


def test_unreadable_journal_restores_every_old_copy(world):
  old, _ = setup_update(world)
  world.run()
  prev = os.path.join(world.root, ou.PREV)
  os.makedirs(os.path.join(prev, "44"))
  os.rename(tile_dir(world.root, "offline", T[0]), tile_dir(world.root, ou.PREV, T[0]))
  os.makedirs(tile_dir(world.root, "offline", T[0]))
  with open(os.path.join(prev, ou.JOURNAL_NAME), "w") as f:
    f.write('{"version": 1, "tiles": [[44, 6]]')  # cut short
  world.manager_restart()
  assert world.run()
  assert all(world.disk(t) == old.get(t) for t in T) and world.leftovers() == []


def test_old_copies_left_without_a_journal_go_back_before_the_next_update(world):
  old, new = setup_update(world)
  world.serve(dict.fromkeys(T, NET_ERROR))
  world.run()
  os.makedirs(os.path.join(world.root, ou.PREV, "44"))
  os.rename(tile_dir(world.root, "offline", T[1]), tile_dir(world.root, ou.PREV, T[1]))
  world.update()
  assert world.run()
  assert all(world.disk(t) == old.get(t) for t in T) and world.leftovers() == []


# ---- crash recovery at every step

def crash_scenario(monkeypatch, crash_at, kind, recovery_crash_at=None, panic=False):
  w = World(monkeypatch)
  try:
    old, new = setup_update(w)
    if panic:
      w.serve({T[3]: b"<html>error</html>" * 20})
    w.run()
    start = w.count
    w.crash_at = start + crash_at
    w.update()
    crashed = False
    try:
      w.run(events_per_poll=2)
    except SimulatedCrash:
      crashed = True
    if crashed:
      w.power_cut() if kind == "power_cut" else w.manager_restart()
      if recovery_crash_at is not None:
        w.crash_at = w.count + recovery_crash_at
        try:
          w.run(events_per_poll=2)
        except SimulatedCrash:
          w.power_cut() if kind == "power_cut" else w.manager_restart()
        w.crash_at = None  # recovery may have had fewer steps than r
      assert w.run(events_per_poll=2), "recovery did not finish"
    assert_consistent(w, old, new)
    total = w.count - start
    if panic:
      w.serve(new)  # the server answers properly again
    w.update()  # the owner taps the update again: it completes
    assert w.run(events_per_poll=2)
    for t in T:
      assert w.disk(t) == new[t], t
    assert w.leftovers() == []
    return crashed, total
  finally:
    w.close()


@pytest.mark.parametrize("kind", ["power_cut", "manager_restart"])
@pytest.mark.parametrize("panic", [False, True], ids=["clean", "panic_loop"])
def test_crash_at_every_step(monkeypatch, kind, panic):
  crashed, total = crash_scenario(monkeypatch, 10 ** 9, kind, panic=panic)
  assert not crashed and total > 40
  for k in range(1, total + 1):
    with monkeypatch.context() as m:
      crash_scenario(m, k, kind, panic=panic)


@pytest.mark.parametrize("kind", ["power_cut", "manager_restart"])
def test_crash_during_recovery(monkeypatch, kind):
  _, total = crash_scenario(monkeypatch, 10 ** 9, kind)
  for k in range(1, total + 1, 3):
    for r in range(1, 16):
      with monkeypatch.context() as m:
        crash_scenario(m, k, kind, recovery_crash_at=r)


def test_gzip_of_fake_helper():
  assert gzip_bytes(b"abc")[:3] == b"\x1f\x8b\x08"


# ---- watchdog signals, one at a time

class ProgressLink:
  request_path = "/nonexistent"

  def __init__(self):
    self.downloaded = None

  def downloaded_files(self):
    return self.downloaded


def watch_setup(downloaded=None):
  link = ProgressLink()
  link.downloaded = downloaded
  clock = [0.0]
  u = ou.OfflineMapUpdater("/nonexistent", link, monotonic=lambda: clock[0], sleep=lambda s: None)
  u._watch = ou.Watch(last_advance=0.0)
  return u, link, clock


def test_watchdog_progress_falling():
  u, link, _ = watch_setup(3)
  assert u._watchdog(3, {}) == ""
  link.downloaded = 1
  assert "downloaded_files fell" in u._watchdog(3, {})


def test_watchdog_download_file_created_again_before_the_frontier():
  u, _, _ = watch_setup()
  assert u._watchdog(2, {2: (100, 5)}) == ""
  assert u._watchdog(3, {3: (10, 6)}) == ""  # the next tile: an advance
  assert "created again" in u._watchdog(3, {0: (10, 7), 3: (20, 6)})


def test_watchdog_download_file_rewritten_without_growing():
  u, _, _ = watch_setup()
  assert u._watchdog(0, {0: (100, 5)}) == ""
  assert u._watchdog(0, {0: (100, 5)}) == ""  # unchanged: no signal
  assert u._watchdog(0, {0: (150, 6)}) == ""  # growing: no signal
  assert "rewritten" in u._watchdog(0, {0: (150, 7)})  # same size, new time: truncated and refilled


def test_watchdog_stall_and_advance():
  u, link, clock = watch_setup(0)
  assert u._watchdog(0, {0: (100, 5)}) == ""
  clock[0] = ou.STALL_SECONDS - 1
  assert u._watchdog(0, {0: (100, 5)}) == ""
  clock[0] = ou.STALL_SECONDS + 0.5
  assert u._watchdog(0, {0: (101, 6)}) == ""  # one more byte: an advance
  clock[0] = 2 * ou.STALL_SECONDS
  link.downloaded = 1
  assert u._watchdog(1, {}) == ""  # a tile done: an advance
  clock[0] = 3 * ou.STALL_SECONDS + 1
  assert "stalled" in u._watchdog(1, {})
