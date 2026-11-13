#!/usr/bin/env python3
"""Staged offline maps go live here, before the mapd binary starts (Inster port, E11). Standard library only.

Layout under the mapd root (names must not start with "v" or be "db" / "tmp": mapd_manager.get_files_for_cleanup):
  offline/<lat>/<lon>/...              live tiles (mapd reads them with whole-file reads)
  offline_manifest.json                live tiles: fetched_at, checked_at, validators, sizes; "no data" tiles; shrink records
  offline_hashes/<lat>_<lon>.json      per area file of a live tile: size and SHA-256 (content comparison)
  offline_next/.lock                   flock; every holder keeps it <= 1 s (the swap: a few s), never across network I/O
  offline_next/swap_done               result of the swap at the last start: boot id, result, tiles, failed starts
  offline_next/requests.json           requested tiles {"lat/lon": [requested_at, reason]} (UPDATE, Update all, migration)
  offline_next/work/<lat>/<lon>.tar.gz.part, work/<lat>/<lon>/   a download in progress (never live)
  offline_next/ready/<lat>/<lon>/      complete, checked, fsynced tile waiting for the swap (the rename into ready/ commits)
  offline_next/hashes/<lat>_<lon>.json the hash list of a ready tile
  offline_next/manifest.json           ready tiles: the live manifest's fields, state pending | ready
  offline_next/swap.json               the swap batch in progress: [[tile, inode of the staged folder], ...]
  offline_next/old/<lat>_<lon>_<ns>/   replaced or deleted tiles (unique names); emptied by map_downloader only

Launch (process_config.py): `timeout -k 5 60 nice -n 19 <python> offline_swap.py <mapd root>; exec <mapd>`: the swap ends before
mapd exists, and mapd starts even if the swap fails, hangs or is killed. Per tile it is atomic (renameat2 RENAME_EXCHANGE,
or two renames where the filesystem refuses it); swap.json records the inode of each staged folder, so a power cut at any
step leaves each tile old or new, never mixed, and the next lock holder finishes the batch (recover()).
Design: inster-dev results/2026-11-13_map_update_redesign/DESIGN.md rev 2 section 3, review 2 (R2-1).
"""
import ctypes
import errno
import fcntl
import json
import os
import shutil
import sys
import time

OFFLINE = "offline"
LIVE_MANIFEST = "offline_manifest.json"
HASHES = "offline_hashes"
NEXT = "offline_next"
LOCK = ".lock"
SWAP_DONE = "swap_done"
REQUESTS = "requests.json"
WORK = "work"
READY = "ready"
NEXT_HASHES = "hashes"
NEXT_MANIFEST = "manifest.json"
SWAP_JOURNAL = "swap.json"
OLD = "old"
READY_SUB = os.path.join(NEXT, READY)  # relative to the mapd root
PREV, TMP = "offline_prev", "tmp"  # E9 leftovers (offline_update.py, removed): migrated once, before mapd
SIZES_NAME = "offline_sizes.json"  # E9 size cache: merged into the live manifest by map_downloader

BATCH = 16
BOOT_LOCK_WAIT = 20.0  # the swap at start waits this long for the lock, then skips (staged tiles stay)
MANIFEST_VERSION = 1
RENAME_EXCHANGE = 2
AT_FDCWD = -100
SYS_RENAMEAT2 = {"aarch64": 276, "x86_64": 316}

APPLIED, NOTHING, SKIPPED, FAILED, MISSING = "applied", "nothing", "skipped", "failed", "missing"


# ---- filesystem steps that change state; module level so the tests can stop the process after any of them
def after_step(name: str) -> None:
  """Hook for the power-cut tests (a child process kills itself after step k); no-op in production."""


def _fsync_dir(path: str) -> None:
  try:
    fd = os.open(path, os.O_RDONLY)
  except FileNotFoundError:
    return
  try:
    os.fsync(fd)
  finally:
    os.close(fd)


def _rename(src: str, dst: str) -> None:
  os.makedirs(os.path.dirname(dst), exist_ok=True)
  os.rename(src, dst)
  after_step("rename")


_libc = None


def _renameat2(src: str, dst: str, flags: int) -> None:
  global _libc
  if _libc is None:
    _libc = ctypes.CDLL(None, use_errno=True)
  fn = getattr(_libc, "renameat2", None)
  if fn is not None:
    ret = fn(AT_FDCWD, os.fsencode(src), AT_FDCWD, os.fsencode(dst), flags)
  else:  # glibc older than 2.28: the raw system call
    nr = SYS_RENAMEAT2.get(os.uname().machine)
    if nr is None:
      raise OSError(errno.ENOSYS, "renameat2 not available")
    ret = _libc.syscall(nr, AT_FDCWD, os.fsencode(src), AT_FDCWD, os.fsencode(dst), flags)
  if ret != 0:
    e = ctypes.get_errno()
    raise OSError(e, os.strerror(e), src)


def _exchange(a: str, b: str) -> None:
  """Atomically swaps two existing paths (ext4 since 3.15). OSError EINVAL / ENOSYS where not supported."""
  _renameat2(a, b, RENAME_EXCHANGE)
  after_step("exchange")


def _rmtree(path: str) -> None:
  if os.path.isdir(path) and not os.path.islink(path):
    shutil.rmtree(path, ignore_errors=True)
  elif os.path.lexists(path):
    os.remove(path)
  after_step("rmtree")


def _remove(path: str) -> None:
  try:
    os.remove(path)
  except FileNotFoundError:
    return
  after_step("remove")


def write_json_atomic(path: str, data) -> None:
  """temp + fsync + rename + folder fsync: the old or the new content survives a power cut."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(data, f, separators=(",", ":"))
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp, path)
  _fsync_dir(os.path.dirname(path))
  after_step("write_json")


def read_json(path: str, default=None):
  try:
    with open(path) as f:
      return json.load(f)
  except FileNotFoundError:
    return default
  except (ValueError, OSError):
    return default


# ---- paths
def key_of(tile) -> str:
  return f"{tile[0]}/{tile[1]}"


def parse_key(key) -> tuple[int, int] | None:
  try:
    lat, lon = str(key).split("/")
    return int(lat), int(lon)
  except (ValueError, AttributeError):
    return None


def flat_of(tile) -> str:
  return f"{tile[0]}_{tile[1]}"


def tile_path(root: str, sub: str, tile) -> str:
  return os.path.join(root, sub, str(tile[0]), str(tile[1]))


def next_path(root: str, *parts) -> str:
  return os.path.join(root, NEXT, *parts)


def live_hashes_path(root: str, tile) -> str:
  return os.path.join(root, HASHES, f"{flat_of(tile)}.json")


def staged_hashes_path(root: str, tile) -> str:
  return next_path(root, NEXT_HASHES, f"{flat_of(tile)}.json")


def old_path(root: str, tile, tag: str = "") -> str:
  return next_path(root, OLD, f"{flat_of(tile)}{tag}_{time.time_ns()}")


def tiles_in(base: str) -> list[tuple[int, int]]:
  """Tile folders `<base>/<lat>/<lon>` (other names are ignored)."""
  out = []
  try:
    lats = os.listdir(base)
  except (FileNotFoundError, NotADirectoryError):
    return out
  for lat in lats:
    lat_dir = os.path.join(base, lat)
    try:
      lons = os.listdir(lat_dir)
    except (FileNotFoundError, NotADirectoryError):
      continue
    for lon in lons:
      tile = parse_key(f"{lat}/{lon}")
      if tile is not None and os.path.isdir(os.path.join(lat_dir, lon)) and not os.path.islink(os.path.join(lat_dir, lon)):
        out.append(tile)
  return sorted(out)


def inode(path: str) -> int | None:
  try:
    return os.lstat(path).st_ino
  except FileNotFoundError:
    return None


# ---- manifests
def empty_manifest() -> dict:
  return {"version": MANIFEST_VERSION, "tiles": {}, "no_data": {}, "shrink": {}, "hosts": {}, "sizes": {}}


def load_manifest(path: str) -> dict:
  data = read_json(path)
  out = empty_manifest()
  if isinstance(data, dict):
    for k in out:
      if k != "version" and isinstance(data.get(k), dict):
        out[k] = {kk: vv for kk, vv in data[k].items() if (k == "hosts" or parse_key(kk) is not None)}
    for k in ("migrated_at",):
      if k in data:
        out[k] = data[k]
  return out


def live_manifest_path(root: str) -> str:
  return os.path.join(root, LIVE_MANIFEST)


def next_manifest_path(root: str) -> str:
  return next_path(root, NEXT_MANIFEST)


# ---- lock
class LockTimeout(Exception):
  pass


class Lock:
  """flock on offline_next/.lock with a bounded wait (polled, so it works from any thread)."""

  def __init__(self, root: str, wait: float, poll: float = 0.02):
    self.path = next_path(root, LOCK)
    self.wait, self.poll = wait, poll
    self.fd = None

  def __enter__(self):
    os.makedirs(os.path.dirname(self.path), exist_ok=True)
    fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + self.wait
    while True:
      try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
      except BlockingIOError:
        if time.monotonic() >= deadline:
          os.close(fd)
          raise LockTimeout(self.path) from None
        time.sleep(self.poll)
    self.fd = fd
    return self

  def __exit__(self, *exc):
    if self.fd is not None:
      fcntl.flock(self.fd, fcntl.LOCK_UN)
      os.close(self.fd)
      self.fd = None
    return False


# ---- the swap (callers hold the lock)
def _move_aside(root: str, src: str, tile, tag: str) -> None:
  if os.path.lexists(src):
    _rename(src, old_path(root, tile, tag))


def _place(root: str, src_sub: str, tile) -> None:
  """src_sub/T goes live: exchanged with offline/T (the old copy is then in src_sub/T), or renamed if there is no live copy.
  EINVAL / ENOSYS: offline/T to old/ first, then src_sub/T to offline/T (recover() finishes the second rename)."""
  src, dst = tile_path(root, src_sub, tile), tile_path(root, OFFLINE, tile)
  if os.path.lexists(dst):
    try:
      _exchange(src, dst)
      return
    except OSError as e:
      if e.errno not in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP):
        raise
    _move_aside(root, dst, tile, "")
  _rename(src, dst)


def _apply_entries(root: str, src_sub: str, tiles: list) -> None:
  """Step 3 of a batch: the staged hash lists and manifest entries become the live ones (idempotent)."""
  if src_sub != READY_SUB:  # E9 migration: no manifest entry, so each tile is downloaded again (requests.json)
    for t in tiles:
      _remove(live_hashes_path(root, t))
    return
  for t in tiles:
    staged = staged_hashes_path(root, t)
    if os.path.exists(staged):
      _rename(staged, live_hashes_path(root, t))
    elif not hashes_describe(live_hashes_path(root, t), tile_path(root, OFFLINE, t)):
      _remove(live_hashes_path(root, t))  # unknown content: rebuilt from disk at the next comparison
  nxt = load_manifest(next_manifest_path(root))
  live = load_manifest(live_manifest_path(root))
  changed = False
  for t in tiles:
    k = key_of(t)
    entry = nxt["tiles"].get(k)
    if entry is not None:
      entry = {f: v for f, v in entry.items() if f != "state"}
      live["tiles"][k] = entry
      changed = True
    elif k not in live["tiles"]:
      live["tiles"][k] = {"fetched_at": None, "checked_at": None}
      changed = True
    if live["no_data"].pop(k, None) is not None:
      changed = True
    if live["shrink"].pop(k, None) is not None:
      changed = True
  if changed:
    write_json_atomic(live_manifest_path(root), live)
  if any(key_of(t) in nxt["tiles"] for t in tiles):
    for t in tiles:
      nxt["tiles"].pop(key_of(t), None)
    write_json_atomic(next_manifest_path(root), nxt)


def hashes_describe(hashes_path: str, folder: str) -> bool:
  """The hash list names exactly the files of the folder, with their sizes (a cut after its rename left it in place)."""
  data = read_json(hashes_path)
  if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
    return False
  try:
    on_disk = {e.name: e.stat().st_size for e in os.scandir(folder) if e.is_file(follow_symlinks=False)}
  except OSError:
    return False
  return on_disk == {k: v[0] for k, v in data["files"].items() if isinstance(v, list) and v}


def _finish_batch(root: str, src_sub: str, tiles: list) -> None:
  _apply_entries(root, src_sub, tiles)
  for t in tiles:  # the displaced old copies (exchange) leave the staging folder under unique names
    _move_aside(root, tile_path(root, src_sub, t), t, "")
  _remove(next_path(root, SWAP_JOURNAL))


def _sync_parents(root: str, src_sub: str, tiles: list) -> None:
  for lat in sorted({t[0] for t in tiles}):
    _fsync_dir(os.path.join(root, OFFLINE, str(lat)))
    _fsync_dir(os.path.join(root, src_sub, str(lat)))
  _fsync_dir(next_path(root, OLD))


def recover(root: str) -> int:
  """Finishes a swap batch a power cut or a kill interrupted (swap.json present). Run first by every lock holder (R2-1).
  Returns the number of tiles of the batch."""
  journal = read_json(next_path(root, SWAP_JOURNAL))
  if journal is None:
    if os.path.exists(next_path(root, SWAP_JOURNAL)):
      _remove(next_path(root, SWAP_JOURNAL))  # unreadable: written atomically, so only a foreign file; nothing was moved
    return 0
  src_sub = READY_SUB if journal.get("src", READY) == READY else PREV
  tiles = []
  for item in journal.get("tiles", []):
    t, ino = (int(item[0][0]), int(item[0][1])), item[1]
    tiles.append(t)
    live, staged = tile_path(root, OFFLINE, t), tile_path(root, src_sub, t)
    if inode(live) == ino:
      continue  # placed
    if inode(staged) == ino:
      _place(root, src_sub, t)
  _sync_parents(root, src_sub, tiles)
  _finish_batch(root, src_sub, tiles)
  return len(tiles)


def _swap_from(root: str, src_sub: str, tiles: list) -> int:
  done = 0
  for i in range(0, len(tiles), BATCH):
    batch = tiles[i:i + BATCH]
    write_json_atomic(next_path(root, SWAP_JOURNAL),
                      {"src": READY if src_sub == READY_SUB else PREV,
                       "tiles": [[list(t), inode(tile_path(root, src_sub, t))] for t in batch]})
    for t in batch:
      _place(root, src_sub, t)
    _sync_parents(root, src_sub, batch)
    _finish_batch(root, src_sub, batch)
    done += len(batch)
  return done


def staged_tiles(root: str) -> list:
  return tiles_in(next_path(root, READY))


def swap(root: str) -> int:
  """Every ready tile goes live (caller holds the lock). Returns the number of tiles placed."""
  n = recover(root)
  return n + _swap_from(root, READY_SUB, staged_tiles(root))


def migrate_e9(root: str) -> int:
  """Once, at the first E11 start (before mapd): E9's moved-aside copies (offline_prev/, each a complete live tile) go back
  over offline/ by the same exchange (the possibly half-written copy goes to old/); journal.json is ignored; tmp/ and
  offline_prev/ are removed. Pre-E11 tiles (no live manifest yet) are requested again ("migration"), whatever the
  auto-update setting: that covers a half-written tile without an old copy and tiles hit by the old truncation bug."""
  n = 0
  prev = os.path.join(root, PREV)
  if os.path.isdir(prev):
    n = _swap_from(root, PREV, tiles_in(prev))
    _rmtree(prev)
  if os.path.lexists(os.path.join(root, TMP)):
    _rmtree(os.path.join(root, TMP))
  if not os.path.exists(live_manifest_path(root)):
    now = time.time()  # noqa: TID251 - request time, a wall clock like checked_at
    req = read_json(next_path(root, REQUESTS), {})
    req = req if isinstance(req, dict) else {}
    for t in tiles_in(os.path.join(root, OFFLINE)):
      req.setdefault(key_of(t), [now, "migration"])
    write_json_atomic(next_path(root, REQUESTS), req)
    manifest = empty_manifest()
    manifest["migrated_at"] = now
    write_json_atomic(live_manifest_path(root), manifest)
  return n


# ---- boot marker
def boot_id() -> str:
  try:
    with open("/proc/sys/kernel/random/boot_id") as f:
      return f.read().strip()
  except OSError:
    return ""


def uptime() -> float:
  try:
    with open("/proc/uptime") as f:
      return float(f.read().split()[0])
  except (OSError, ValueError, IndexError):
    return 0.0


def write_swap_done(root: str, result: str, tiles: int = 0, error: str = "") -> dict:
  """swap_done at the end of every start's swap, also when there was nothing to do. failed_starts counts consecutive
  starts after which staged tiles were still waiting (the screen warns from 2 on, R2-1)."""
  previous = read_json(next_path(root, SWAP_DONE), {})
  previous = previous if isinstance(previous, dict) else {}
  left = len(staged_tiles(root))
  failed = int(previous.get("failed_starts", 0) or 0) + 1 if left else 0
  wall = time.time()  # noqa: TID251 - shown to the owner as a date
  done = {"boot_id": boot_id(), "uptime": uptime(), "time": wall, "result": result, "tiles": tiles, "error": error[:200],
          "staged_left": left, "failed_starts": failed}
  write_json_atomic(next_path(root, SWAP_DONE), done)
  return done


def start_swap(root: str, lock_wait: float = BOOT_LOCK_WAIT) -> dict:
  """The launch-path step: lock (bounded wait), recovery, E9 migration, swap, marker. Never raises."""
  try:
    os.makedirs(next_path(root), exist_ok=True)
  except OSError as e:
    return {"result": FAILED, "error": str(e)}
  try:
    with Lock(root, lock_wait):
      try:
        n = recover(root)  # R2-1: an interrupted batch first, before anything else touches the staging folder
        n += migrate_e9(root)
        n += swap(root)
      except Exception as e:  # mapd must start; the next lock holder finishes the batch (recover)
        return write_swap_done(root, FAILED, 0, f"{type(e).__name__}: {e}")
      return write_swap_done(root, APPLIED if n else NOTHING, n)
  except LockTimeout:
    try:
      return write_swap_done(root, SKIPPED, 0, "lock held")
    except OSError as e:
      return {"result": SKIPPED, "error": str(e)}
  except OSError as e:
    return {"result": FAILED, "error": str(e)}


def launch_command(python: str, script: str, mapd: str, root: str) -> list[str]:
  """process_config's mapd command: the swap (SIGTERM after 60 s, SIGKILL 5 s later; lowest CPU priority), then mapd in its place."""
  import shlex
  q = shlex.quote
  script = os.path.abspath(script)
  return ["bash", "-c", f"timeout -k 5 60 nice -n 19 {q(python)} {q(script)} {q(root)} > /dev/null 2>&1; exec {q(mapd)} > /dev/null 2>&1"]


def main(argv: list[str]) -> int:
  if len(argv) != 2:
    return 2
  start_swap(argv[1])
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
