"""E11 swap before mapd (DESIGN 9.3, 9.4; review 2 R2-1): launch command, lock, exchange and fallback, migration, marker."""
import errno
import fcntl
import os
import subprocess
import sys
import threading
import time

import pytest

from openpilot.sunnypilot.mapd import offline_swap as sw
from openpilot.sunnypilot.mapd.tests.swap_world import content, e9_leftovers, hashes, live, read_tile, stage

A, B, C = (46, 6), (46, 8), (-34, 18)


def exchange_supported(path) -> bool:
  a, b = os.path.join(path, "xa"), os.path.join(path, "xb")
  os.makedirs(a)
  os.makedirs(b)
  try:
    sw._renameat2(a, b, sw.RENAME_EXCHANGE)
    return True
  except OSError:
    return False


@pytest.fixture
def no_exchange(monkeypatch):
  def refuse(*a):
    raise OSError(errno.EINVAL, "not supported")
  monkeypatch.setattr(sw, "_renameat2", refuse)


def setup_three(root):
  live(root, A, content(A, "old"))
  live(root, B, content(B, "old"))
  stage(root, A, content(A, "new"))
  stage(root, C, content(C, "new"))  # new tile (negative coordinates), no live copy


def check_three(root):
  assert read_tile(sw.tile_path(root, sw.OFFLINE, A)) == content(A, "new")
  assert read_tile(sw.tile_path(root, sw.OFFLINE, B)) == content(B, "old")
  assert read_tile(sw.tile_path(root, sw.OFFLINE, C)) == content(C, "new")
  assert sw.staged_tiles(root) == [] and not os.path.exists(sw.next_path(root, sw.SWAP_JOURNAL))
  m = sw.load_manifest(sw.live_manifest_path(root))
  assert m["tiles"]["46/6"]["etag"] == '"(46, 6)"' and m["tiles"]["-34/18"]["etag"] == '"(-34, 18)"' and m["tiles"]["46/8"]["etag"] == '"old"'
  assert "state" not in m["tiles"]["46/6"] and sw.load_manifest(sw.next_manifest_path(root))["tiles"] == {}
  assert sw.read_json(sw.live_hashes_path(root, A))["files"] == hashes(content(A, "new"))
  olds = os.listdir(sw.next_path(root, sw.OLD))
  assert len(olds) == 1 and read_tile(os.path.join(sw.next_path(root, sw.OLD), olds[0])) == content(A, "old")


def test_swap_with_exchange(tmp_path):
  root = str(tmp_path)
  if not exchange_supported(root):
    pytest.skip("RENAME_EXCHANGE not supported on this filesystem")
  setup_three(root)
  done = sw.start_swap(root)
  assert done["result"] == sw.APPLIED and done["tiles"] == 2 and done["failed_starts"] == 0
  check_three(root)
  assert sw.read_json(sw.next_path(root, sw.SWAP_DONE))["boot_id"] == sw.boot_id()


def test_swap_with_the_einval_fallback(tmp_path, no_exchange):
  root = str(tmp_path)
  setup_three(root)
  assert sw.start_swap(root)["result"] == sw.APPLIED
  check_three(root)


def test_nothing_to_do_still_writes_the_marker(tmp_path):
  done = sw.start_swap(str(tmp_path))
  assert done["result"] == sw.NOTHING and done["boot_id"] == sw.boot_id()


def test_unique_old_names_after_repeated_swaps(tmp_path, no_exchange):
  root = str(tmp_path)
  live(root, A, content(A, "old"))
  for v in ("v1", "v2", "v3"):
    stage(root, A, content(A, v))
    with sw.Lock(root, 1):
      sw.swap(root)
  assert read_tile(sw.tile_path(root, sw.OFFLINE, A)) == content(A, "v3")
  assert len(os.listdir(sw.next_path(root, sw.OLD))) == 3


def test_locked_out_swap_skips_and_keeps_the_staged_tiles(tmp_path):
  root = str(tmp_path)
  setup_three(root)
  fd = os.open(sw.next_path(root, sw.LOCK), os.O_RDWR | os.O_CREAT)
  fcntl.flock(fd, fcntl.LOCK_EX)
  try:
    t0 = time.monotonic()
    done = sw.start_swap(root, lock_wait=0.5)
    assert 0.4 < time.monotonic() - t0 < 3
  finally:
    os.close(fd)
  assert done["result"] == sw.SKIPPED and done["staged_left"] == 2 and done["failed_starts"] == 1
  assert read_tile(sw.tile_path(root, sw.OFFLINE, A)) == content(A, "old") and len(sw.staged_tiles(root)) == 2
  assert sw.BOOT_LOCK_WAIT == 20.0


def test_swap_waits_for_a_short_lock_holder(tmp_path):
  root = str(tmp_path)
  setup_three(root)
  fd = os.open(sw.next_path(root, sw.LOCK), os.O_RDWR | os.O_CREAT)
  fcntl.flock(fd, fcntl.LOCK_EX)
  threading.Timer(0.5, lambda: os.close(fd)).start()
  assert sw.start_swap(root, lock_wait=5)["result"] == sw.APPLIED


def test_failed_starts_count_until_applied(tmp_path):
  root = str(tmp_path)
  setup_three(root)
  fd = os.open(sw.next_path(root, sw.LOCK), os.O_RDWR | os.O_CREAT)
  fcntl.flock(fd, fcntl.LOCK_EX)
  assert sw.start_swap(root, lock_wait=0.05)["failed_starts"] == 1
  assert sw.start_swap(root, lock_wait=0.05)["failed_starts"] == 2  # the screen warns from 2 (R2-1)
  os.close(fd)
  assert sw.start_swap(root)["failed_starts"] == 0


def fake_mapd(tmp_path, root):
  """A stand-in mapd binary: records which copy of tile A is live when it starts."""
  out = tmp_path / "mapd_saw.txt"
  script = tmp_path / "mapd"
  name = sorted(content(A, "new"))[-1]
  script.write_text(f"#!/bin/bash\nif [ -e '{sw.tile_path(root, sw.OFFLINE, A)}/{name}' ]; then echo new > '{out}'; else echo old > '{out}'; fi\n")
  script.chmod(0o755)
  return str(script), out


def test_launch_command_swaps_then_execs_mapd(tmp_path):
  root = str(tmp_path / "root")
  setup_three(root)
  mapd, saw = fake_mapd(tmp_path, root)
  cmd = sw.launch_command(sys.executable, sw.__file__, mapd, root)
  assert cmd[:2] == ["bash", "-c"] and "timeout -k 5 60 nice -n 19" in cmd[2] and cmd[2].endswith("> /dev/null 2>&1") and "; exec " in cmd[2]
  subprocess.run(cmd, check=True, timeout=60)
  assert saw.read_text().strip() == "new"


@pytest.mark.parametrize("python", ["false", "missing-interpreter"])
def test_failing_swap_still_starts_mapd(tmp_path, python):
  root = str(tmp_path / "root")
  setup_three(root)
  mapd, saw = fake_mapd(tmp_path, root)
  subprocess.run(sw.launch_command(python, sw.__file__, mapd, root), timeout=60)
  assert saw.read_text().strip() == "old"


def test_hanging_swap_is_stopped_and_mapd_starts(tmp_path):
  root = str(tmp_path / "root")
  setup_three(root)
  mapd, saw = fake_mapd(tmp_path, root)
  hang = tmp_path / "hang.sh"
  hang.write_text("#!/bin/bash\ntrap '' TERM\nwhile true; do sleep 0.2; done\n")  # ignores SIGTERM: only the -k kill ends it
  hang.chmod(0o755)
  cmd = sw.launch_command(str(hang), sw.__file__, mapd, root)
  cmd[2] = cmd[2].replace("timeout -k 5 60", "timeout -k 1 2")  # the same command with a short bound
  t0 = time.monotonic()
  subprocess.run(cmd, timeout=30)
  assert time.monotonic() - t0 < 10 and saw.read_text().strip() == "old"


def test_reader_never_sees_a_missing_or_short_file_during_exchanges(tmp_path):
  root = str(tmp_path)
  if not exchange_supported(root):
    pytest.skip("RENAME_EXCHANGE not supported on this filesystem")
  versions = [content(A, "v1", n=3), content(A, "v2", n=3)]
  names = sorted(set(versions[0]) & set(versions[1]))
  live(root, A, versions[0])
  stop, bad = threading.Event(), []

  def reader():
    while not stop.is_set():
      for name in names:
        try:
          data = open(os.path.join(sw.tile_path(root, sw.OFFLINE, A), name), "rb").read()
        except FileNotFoundError:
          bad.append(("missing", name))
          continue
        if data not in (versions[0][name], versions[1][name]):
          bad.append(("short", name, len(data)))
  t = threading.Thread(target=reader)
  t.start()
  for i in range(200):
    stage(root, A, versions[(i + 1) % 2])
    with sw.Lock(root, 1):
      sw.swap(root)
  stop.set()
  t.join()
  assert not bad, bad[:5]


def test_swap_time_for_95_tiles(tmp_path):
  root = str(tmp_path)
  tiles = [(lat, lon) for lat in range(36, 56, 2) for lon in range(-6, 14, 2)][:95]
  for t in tiles:
    live(root, t, content(t, "old", n=2))
    stage(root, t, content(t, "new", n=2))
  t0 = time.monotonic()
  done = sw.start_swap(root)
  elapsed = time.monotonic() - t0
  assert done["tiles"] == 95 and elapsed < 30, elapsed
  print(f"swap of 95 tiles: {elapsed:.2f} s")


# ---- E9 migration (DESIGN 3.4, 9.4)
def test_migration_restores_the_moved_aside_copy_before_mapd(tmp_path):
  root = str(tmp_path / "root")
  e9_leftovers(root, {A: content(A, "old"), B: content(B, "old")}, {A: {"46.000000_6.000000_46.250000_6.250000": b"half"}})
  live_c = content(C, "old")
  from openpilot.sunnypilot.mapd.tests.swap_world import write_tile
  write_tile(sw.tile_path(root, sw.OFFLINE, C), live_c)  # a pre-E11 tile with no old copy
  mapd, saw = fake_mapd(tmp_path, root)
  subprocess.run(sw.launch_command(sys.executable, sw.__file__, mapd, root), check=True, timeout=60)
  assert read_tile(sw.tile_path(root, sw.OFFLINE, A)) == content(A, "old")
  assert read_tile(sw.tile_path(root, sw.OFFLINE, B)) == content(B, "old")
  assert read_tile(sw.tile_path(root, sw.OFFLINE, C)) == live_c
  assert not os.path.exists(os.path.join(root, sw.PREV)) and not os.path.exists(os.path.join(root, sw.TMP))
  req = sw.read_json(sw.next_path(root, sw.REQUESTS))
  assert {k: v[1] for k, v in req.items()} == {"46/6": "migration", "46/8": "migration", "-34/18": "migration"}
  assert sw.read_json(sw.live_manifest_path(root))["migrated_at"]
  olds = os.listdir(sw.next_path(root, sw.OLD))
  assert len(olds) == 1 and read_tile(os.path.join(sw.next_path(root, sw.OLD), olds[0])) == {"46.000000_6.000000_46.250000_6.250000": b"half"}
  # a second start does not queue again nor move anything
  sw.write_json_atomic(sw.next_path(root, sw.REQUESTS), {})
  assert sw.start_swap(root)["result"] == sw.NOTHING and sw.read_json(sw.next_path(root, sw.REQUESTS)) == {}


def test_migration_with_the_einval_fallback(tmp_path, no_exchange):
  root = str(tmp_path)
  e9_leftovers(root, {A: content(A, "old")}, {A: {"x": b"half"}})
  sw.start_swap(root)
  assert read_tile(sw.tile_path(root, sw.OFFLINE, A)) == content(A, "old")
