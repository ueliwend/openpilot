"""E11 power cut at every step (DESIGN 9.2): a child process runs one operation and SIGKILLs itself right after
filesystem step k (offline_swap.after_step), for every k until the operation finishes. The parent then does what the next
start does (the swap at start, then map_downloader's start-up) and checks: every live tile is the complete old or new
fixture, nothing partial is live, the run converges, the manifests match the disk, and leftovers are at most one tile.
A recording shim checks the fsync-before-rename and folder-fsync-after order."""
import errno
import os
import signal

import pytest

from openpilot.sunnypilot.mapd import offline_swap as sw
from openpilot.sunnypilot.mapd.map_store import Store
from openpilot.sunnypilot.mapd.tests.swap_world import content, e9_leftovers, hashes, live, read_tile, stage, write_tile

A, B, C = (46, 6), (46, 8), (-34, 18)


def killed_after(k: int, op) -> bool:
  """Runs op in a child that dies right after step k; True if it was killed (False: op finished first)."""
  pid = os.fork()
  if pid == 0:
    count = [0]

    def hook(name):
      count[0] += 1
      if count[0] == k:
        os.kill(os.getpid(), signal.SIGKILL)
    sw.after_step = hook
    try:
      op()
    finally:
      os._exit(0)
  _, status = os.waitpid(pid, 0)
  return os.WIFSIGNALED(status)


def every_cut(build, op, check, modes=("exchange", "einval"), monkeypatch=None, tmp_path=None, max_steps=400):
  """check(root, k, mode) after a cut at step k; in mode "einval" the filesystem refuses RENAME_EXCHANGE (two renames)."""
  cuts = 0
  for mode in modes:
    if mode == "einval":
      monkeypatch.setattr(sw, "_renameat2", _refuse)
    for k in range(1, max_steps):
      root = str(tmp_path / f"{mode}_{k}")
      build(root)
      was_killed = killed_after(k, lambda root=root: op(root))
      check(root, k, mode)
      if not was_killed:
        break
      cuts += 1
    else:
      raise AssertionError("operation never finished")
    monkeypatch.undo()
  return cuts


def _refuse(*a):
  raise OSError(errno.EINVAL, "not supported")


def next_start(root):
  """What the next start does: the swap before mapd, then map_downloader's start-up under the lock."""
  sw.start_swap(root)
  store = Store(root)
  with store.lock():
    store.startup(1.8e9)
  return store


def assert_whole(root, tile, allowed):
  got = read_tile(sw.tile_path(root, sw.OFFLINE, tile))
  assert got in allowed, (tile, None if got is None else sorted(got))


def manifests_match_disk(root):
  store = Store(root)
  live_keys = {sw.key_of(t) for t in store.live_tiles()}
  assert live_keys <= set(store.live["tiles"]), (live_keys, set(store.live["tiles"]))
  assert set(store.next["tiles"]) == {sw.key_of(t) for t in store.staged_tiles()}
  assert not os.path.exists(sw.next_path(root, sw.SWAP_JOURNAL))


def test_swap_cut_at_every_step(tmp_path, monkeypatch):
  def build(root):
    live(root, A, content(A, "old"))
    live(root, B, content(B, "old"))
    stage(root, A, content(A, "new"))
    stage(root, C, content(C, "new"))

  def op(root):
    with sw.Lock(root, 1):
      sw.swap(root)

  def check(root, k, mode):
    # at the cut itself: each tile old or new, never mixed (two renames: absent between them, finished by the next holder)
    assert_whole(root, A, [content(A, "old"), content(A, "new")] + ([None] if mode == "einval" else []))
    assert_whole(root, C, [None, content(C, "new")])
    next_start(root)
    assert_whole(root, A, [content(A, "new")])
    assert_whole(root, B, [content(B, "old")])
    assert_whole(root, C, [content(C, "new")])
    manifests_match_disk(root)
    m = sw.load_manifest(sw.live_manifest_path(root))
    assert m["tiles"]["46/6"]["etag"] == '"(46, 6)"' and m["tiles"]["-34/18"]["etag"] == '"(-34, 18)"'
    assert sw.read_json(sw.live_hashes_path(root, A))["files"] == hashes(content(A, "new"))
    store = Store(root)
    store.empty_old(budget_s=10)
    assert not os.listdir(sw.next_path(root, sw.OLD))
  assert every_cut(build, op, check, monkeypatch=monkeypatch, tmp_path=tmp_path) >= 15  # 8 cut points with the exchange, 10 with two renames


def test_r21_killed_swap_then_downloader_60s_path_then_next_start(tmp_path, monkeypatch):
  """Review 2 R2-1: a swap killed mid-batch; the downloader (no marker, 60 s) recovers before anything else."""
  def build(root):
    live(root, A, content(A, "old"))
    stage(root, A, content(A, "new"))
    stage(root, C, content(C, "new"))

  def op(root):
    sw.start_swap(root)

  def check(root, k, mode):
    store = Store(root)
    with store.lock():  # map_downloader's lock: recovery first
      store.startup(1.8e9)
    assert not os.path.exists(sw.next_path(root, sw.SWAP_JOURNAL))
    for t in (A, C):
      got = read_tile(sw.tile_path(root, sw.OFFLINE, t))
      staged = read_tile(sw.tile_path(root, sw.READY_SUB, t))
      assert got == content(t, "new") or (staged == content(t, "new") and got in (None, content(t, "old")))
    manifests_match_disk(root)
    next_start(root)
    assert_whole(root, A, [content(A, "new")])
    assert_whole(root, C, [content(C, "new")])
  every_cut(build, op, check, modes=("exchange",), monkeypatch=monkeypatch, tmp_path=tmp_path)


def test_migration_cut_at_every_step(tmp_path, monkeypatch):
  def build(root):
    e9_leftovers(root, {A: content(A, "old"), B: content(B, "old")}, {A: {"46.000000_6.000000_46.250000_6.250000": b"half"}})

  def op(root):
    sw.start_swap(root)

  def check(root, k, mode):
    assert_whole(root, A, [{"46.000000_6.000000_46.250000_6.250000": b"half"}, content(A, "old")] + ([None] if mode == "einval" else []))
    next_start(root)
    assert_whole(root, A, [content(A, "old")])
    assert_whole(root, B, [content(B, "old")])
    assert not os.path.exists(os.path.join(root, sw.PREV)) and not os.path.exists(os.path.join(root, sw.TMP))
    req = sw.read_json(sw.next_path(root, sw.REQUESTS))
    assert set(req) == {"46/6", "46/8"}
  every_cut(build, op, check, monkeypatch=monkeypatch, tmp_path=tmp_path)


def test_commit_cut_at_every_step(tmp_path, monkeypatch):
  files = content(A, "new")

  def build(root):
    live(root, A, content(A, "old"))
    store = Store(root)
    write_tile(store.work_dir(A), files)
    os.makedirs(os.path.dirname(store.part(A)), exist_ok=True)
    open(store.part(A), "wb").write(b"part")

  def op(root):
    store = Store(root)
    with store.lock():
      store.commit(A, {"fetched_at": 1.8e9, "checked_at": 1.8e9, "etag": '"n"'}, hashes(files))

  def check(root, k, mode):
    assert_whole(root, A, [content(A, "old")])  # nothing goes live before the next start
    staged = read_tile(sw.tile_path(root, sw.READY_SUB, A))
    assert staged in (None, files)
    store = next_start(root)
    assert_whole(root, A, [content(A, "old"), files])
    if read_tile(sw.tile_path(root, sw.OFFLINE, A)) == files:
      assert store.live["tiles"]["46/6"].get("etag") in ('"n"', None)
    manifests_match_disk(root)
    assert not sw.tiles_in(sw.next_path(root, sw.WORK))  # work folders cleared (the .part may stay for a resume)
  every_cut(build, op, check, modes=("exchange",), monkeypatch=monkeypatch, tmp_path=tmp_path)


def test_delete_cut_at_every_step(tmp_path, monkeypatch):
  def build(root):
    live(root, A, content(A, "old"))
    live(root, B, content(B, "old"))
    stage(root, A, content(A, "new"))

  def op(root):
    store = Store(root)
    with store.lock():
      store.delete_tiles([A])

  def check(root, k, mode):
    assert_whole(root, A, [None, content(A, "old")])
    assert_whole(root, B, [content(B, "old")])
    store = next_start(root)
    assert_whole(root, B, [content(B, "old")])
    assert_whole(root, A, [None, content(A, "old"), content(A, "new")])  # a cut before the delete finished: redone by its request
    with store.lock():
      store.delete_tiles([A])
    assert read_tile(sw.tile_path(root, sw.OFFLINE, A)) is None and "46/6" not in Store(root).live["tiles"]
  every_cut(build, op, check, modes=("exchange",), monkeypatch=monkeypatch, tmp_path=tmp_path)


def test_fsync_order(tmp_path, monkeypatch):
  """fsync of the files and folder before the rename that commits, the parent folder after; swap.json before the swap."""
  log = []
  real_fsync_dir, real_rename, real_exchange, real_write = sw._fsync_dir, sw._rename, sw._exchange, sw.write_json_atomic
  monkeypatch.setattr(sw, "_fsync_dir", lambda p: (log.append(("fsync_dir", p)), real_fsync_dir(p)))
  monkeypatch.setattr(sw, "_rename", lambda a, b: (log.append(("rename", a, b)), real_rename(a, b)))
  monkeypatch.setattr(sw, "_exchange", lambda a, b: (log.append(("exchange", a, b)), real_exchange(a, b)))
  monkeypatch.setattr(sw, "write_json_atomic", lambda p, d: (log.append(("json", os.path.basename(p))), real_write(p, d)))
  root = str(tmp_path)
  live(root, A, content(A, "old"))
  store = Store(root)
  write_tile(store.work_dir(A), content(A, "new"))
  log.clear()
  with store.lock():
    store.commit(A, {"fetched_at": 1.8e9}, hashes(content(A, "new")))
  ready = sw.tile_path(root, sw.READY_SUB, A)
  i_commit = log.index(("rename", store.work_dir(A), ready))
  assert ("fsync_dir", store.work_dir(A)) in log[:i_commit] and ("json", "manifest.json") in log[:i_commit]
  assert ("json", f"{A[0]}_{A[1]}.json") in log[:i_commit]  # the hash list before the commit
  assert ("fsync_dir", os.path.dirname(ready)) in log[i_commit:]
  log.clear()
  with sw.Lock(root, 1):
    sw.swap(root)
  i_journal = log.index(("json", "swap.json"))
  i_swap = next(i for i, e in enumerate(log) if e[0] == "exchange")
  i_parent = log.index(("fsync_dir", os.path.join(root, sw.OFFLINE, str(A[0]))))
  i_manifest = log.index(("json", "offline_manifest.json"))
  assert i_journal < i_swap < i_parent < i_manifest


def test_download_cut_mid_transfer_resumes(tmp_path):
  """A power cut while a tile streams: the child dies inside the transfer; the next run resumes from the .part."""
  from openpilot.sunnypilot.mapd import tile_fetch as tf
  from openpilot.sunnypilot.mapd.tests.tile_server import FakeTileServer, Tile, area_files, make_archive, session
  server = FakeTileServer()
  try:
    files = {k: os.urandom(len(v)) for k, v in area_files(A, n=8, size=60000).items()}
    body = make_archive(A, files)
    server.put(A, Tile(body=body, etag='"v1"'))
    root = str(tmp_path)
    store = Store(root)
    part, work = store.part(A), store.work_dir(A)
    pid = os.fork()
    if pid == 0:
      def tick():
        if os.path.exists(part) and os.path.getsize(part) > len(body) // 2:
          os.kill(os.getpid(), signal.SIGKILL)
      try:
        tf.Fetcher(session=session(), base_url=server.url).check(A, part, work, None, False, tick, lambda: 100_000)
      finally:
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.WIFSIGNALED(status) and 0 < os.path.getsize(part) < len(body)
    next_start(root)  # work folder cleared, .part kept
    assert os.path.exists(part) and not os.path.exists(work)
    out = tf.Fetcher(session=session(), base_url=server.url).check(A, part, work, None, False, lambda: None, lambda: None)
    assert out.kind == tf.OK and out.files == hashes(files)
    assert server.gets(A)[-1][2].get("range", "").startswith("bytes=")
  finally:
    server.close()


@pytest.fixture(autouse=True)
def _reset_hook():
  yield
  sw.after_step = lambda name: None
