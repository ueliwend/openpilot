"""The map store seen by map_downloader (Inster port, E11): manifests, requests, staging, comparison, deletes.

Only map_downloader writes here while mapd runs; the swap at start (offline_swap.py) writes under the same lock. Every
change happens under `offline_next/.lock`, held for one short step (never across network I/O or a folder deletion:
deleted tiles are renamed into `offline_next/old/` under the lock and removed outside it). Before anything else, every
lock holder finishes an interrupted swap batch (offline_swap.recover, review 2 R2-1).

Staging (DESIGN 3.2 step 5): the next manifest gets the tile as `pending`, the hash list is written, the work folder is
fsynced, renamed into `ready/` (the commit) and its parent fsynced, then the entry becomes `ready` and the `.part` goes.
A tile whose files and hashes equal the live tile's is not staged (only checked_at and validators change). A tile with
> 20 % fewer area files or bytes than the live one, or none, is not staged either (shrink guard, DESIGN 5.5): it is
applied only if a check >= 7 days after the first such answer returns the same or a new archive within +-2 % of that first
smaller one (a stable server state, review 2 R2-2); a size beyond that starts the clock again.
"""
import os
import time

from openpilot.sunnypilot.mapd import offline_swap as sw
from openpilot.sunnypilot.mapd.tile_fetch import hash_tree, remove_part

SHRINK_SHARE = 0.20
SHRINK_SAME = 0.02
SHRINK_WAIT_SECONDS = 7 * 86400
LOCK_WAIT = 2.0  # map_downloader's own wait; the swap at start waits longer (offline_swap.BOOT_LOCK_WAIT)

KEEP, APPLY, IDENTICAL = "keep", "apply", "identical"


def within(a: float, b: float, share: float) -> bool:
  return abs(a - b) <= share * b if b else a == b


def shrink_decision(live_files: int, live_bytes: int, new_files: int, new_bytes: int, record: dict | None, now: float | None,
                    etag) -> tuple[str, dict | None]:
  """(APPLY | KEEP, the shrink record to keep). now None = no valid clock: a smaller tile is kept, no clock started."""
  smaller = (live_files > 0 and new_files == 0) or new_files < (1 - SHRINK_SHARE) * live_files or \
    new_bytes < (1 - SHRINK_SHARE) * live_bytes
  if not smaller:
    return APPLY, None
  if now is None:
    return KEEP, record
  if record and within(new_files, record["files"], SHRINK_SAME) and within(new_bytes, record["bytes"], SHRINK_SAME):
    if now - record["first_seen"] >= SHRINK_WAIT_SECONDS:
      return APPLY, None
    return KEEP, dict(record, etag=etag, last_seen=now)
  return KEEP, {"etag": etag, "files": new_files, "bytes": new_bytes, "first_seen": now, "last_seen": now,
                "live_files": live_files, "live_bytes": live_bytes}


class Store:
  def __init__(self, root: str):
    self.root = root
    self.live = sw.empty_manifest()
    self.next = sw.empty_manifest()
    self.requests: dict[str, list] = {}
    self.reload()

  # ---- paths
  def work_dir(self, tile) -> str:
    return sw.tile_path(self.root, os.path.join(sw.NEXT, sw.WORK), tile)

  def part(self, tile) -> str:
    return sw.next_path(self.root, sw.WORK, str(tile[0]), f"{tile[1]}.tar.gz.part")

  def lock(self, wait: float = LOCK_WAIT) -> sw.Lock:
    return _RecoveringLock(self, wait)

  # ---- state
  def reload(self) -> None:
    self.live = sw.load_manifest(sw.live_manifest_path(self.root))
    self.next = sw.load_manifest(sw.next_manifest_path(self.root))
    req = sw.read_json(sw.next_path(self.root, sw.REQUESTS), {})
    self.requests = {k: v for k, v in req.items() if sw.parse_key(k) and isinstance(v, list) and len(v) == 2} \
      if isinstance(req, dict) else {}

  def live_tiles(self) -> list:
    return sw.tiles_in(os.path.join(self.root, sw.OFFLINE))

  def staged_tiles(self) -> list:
    return sw.staged_tiles(self.root)

  def save_live(self) -> None:
    sw.write_json_atomic(sw.live_manifest_path(self.root), self.live)

  def save_next(self) -> None:
    sw.write_json_atomic(sw.next_manifest_path(self.root), self.next)

  def save_requests(self) -> None:
    sw.write_json_atomic(sw.next_path(self.root, sw.REQUESTS), self.requests)

  # ---- start-up (after the swap at start, or 60 s): under the lock
  def startup(self, now: float | None) -> dict:
    """Pending entries without a ready folder go, work folders are emptied (the .part files stay for a resume), E9's size
    cache is merged, and every live tile without an entry gets one (pre-E11: fetched_at = oldest file mtime)."""
    report = {"pending_dropped": 0, "registered": 0}
    staged = {sw.key_of(t) for t in self.staged_tiles()}
    for k in [k for k in self.next["tiles"] if k not in staged]:
      del self.next["tiles"][k]
      report["pending_dropped"] += 1
    if report["pending_dropped"]:
      self.save_next()
    for k in staged - set(self.next["tiles"]):
      self.next["tiles"][k] = {"fetched_at": None, "checked_at": None, "state": "ready"}
    work = sw.next_path(self.root, sw.WORK)
    for t in sw.tiles_in(work):
      sw._rename(sw.tile_path(self.root, os.path.join(sw.NEXT, sw.WORK), t), sw.old_path(self.root, t, "_work"))
    changed = False
    sizes = sw.read_json(os.path.join(self.root, sw.SIZES_NAME))
    if isinstance(sizes, dict) and isinstance(sizes.get("tiles"), dict):
      for k, v in sizes["tiles"].items():
        if sw.parse_key(k) and isinstance(v, dict) and isinstance(v.get("download"), int):
          self.live["sizes"].setdefault(k, v["download"])
      changed = True
    for t in self.live_tiles():
      k = sw.key_of(t)
      if k not in self.live["tiles"]:
        mtimes = [e.stat().st_mtime for e in os.scandir(sw.tile_path(self.root, sw.OFFLINE, t)) if e.is_file(follow_symlinks=False)]
        self.live["tiles"][k] = {"fetched_at": min(mtimes) if mtimes else None, "checked_at": None, "pre_e11": True}
        report["registered"] += 1
        changed = True
    if changed:
      self.save_live()
    if os.path.exists(os.path.join(self.root, sw.SIZES_NAME)):
      sw._remove(os.path.join(self.root, sw.SIZES_NAME))
    return report

  # ---- requests
  def add_requests(self, tiles, now: float, reason: str) -> int:
    added = 0
    for t in tiles:
      k = sw.key_of(t)
      if k not in self.requests:
        self.requests[k] = [now, reason]
        added += 1
    if added:
      self.save_requests()
    return added

  def drop_request(self, tile) -> None:
    if self.requests.pop(sw.key_of(tile), None) is not None:
      self.save_requests()

  # ---- results of a check (under the lock)
  def mark_checked(self, tile, now: float | None, headers: dict | None = None, no_data: bool = False) -> None:
    k = sw.key_of(tile)
    if no_data and not os.path.isdir(sw.tile_path(self.root, sw.OFFLINE, tile)):
      self.live["no_data"][k] = {"checked_at": now}
      self.live["tiles"].pop(k, None)
    else:
      entry = self.live["tiles"].setdefault(k, {"fetched_at": None})
      entry["checked_at"] = now
      if no_data:
        entry["server_404_at"] = now  # a live tile is kept when the server has no file any more (logged by the caller)
      for f in ("etag", "last_modified"):
        if headers and headers.get(f):
          entry[f] = headers[f]
    self.save_live()
    self.drop_request(tile)

  def set_no_304(self, host: str) -> None:
    self.live["hosts"].setdefault(host, {})["no_304"] = True

  def prepare_hashes(self, tile) -> tuple[int, dict] | None:
    """Outside the lock (read only): the full hash of a live tile that has no hash list yet (pre-E11), with the folder's
    inode, so the lock is held for a short step only. None when a list exists or there is no live tile."""
    data = sw.read_json(sw.live_hashes_path(self.root, tile))
    folder = sw.tile_path(self.root, sw.OFFLINE, tile)
    if (isinstance(data, dict) and isinstance(data.get("files"), dict)) or not os.path.isdir(folder):
      return None
    ino = sw.inode(folder)
    return (ino, hash_tree(folder)) if ino is not None else None

  def live_hashes(self, tile, prepared: tuple[int, dict] | None = None) -> dict[str, list]:
    """The live tile's hash list; built once from disk if it has none (pre-E11 tiles): from `prepared` when the live folder
    is still the one that was hashed, else (never expected: only this process changes it) here."""
    data = sw.read_json(sw.live_hashes_path(self.root, tile))
    if isinstance(data, dict) and isinstance(data.get("files"), dict):
      return data["files"]
    folder = sw.tile_path(self.root, sw.OFFLINE, tile)
    if not os.path.isdir(folder):
      return {}
    files = prepared[1] if prepared is not None and prepared[0] == sw.inode(folder) else hash_tree(folder)
    sw.write_json_atomic(sw.live_hashes_path(self.root, tile), {"files": files})
    return files

  def judge(self, tile, files: dict, etag, now: float | None, prepared: tuple[int, dict] | None = None) -> tuple[str, dict | None]:
    """IDENTICAL (nothing to stage), KEEP (shrink guard: live kept) or APPLY (stage it)."""
    if not os.path.isdir(sw.tile_path(self.root, sw.OFFLINE, tile)):
      return APPLY, None
    live = self.live_hashes(tile, prepared)
    if live == files:
      return IDENTICAL, None
    k = sw.key_of(tile)
    decision, record = shrink_decision(len(live), sum(v[0] for v in live.values()), len(files), sum(v[0] for v in files.values()),
                                       self.live["shrink"].get(k), now, etag)
    return decision, record

  def record_shrink(self, tile, record: dict | None) -> None:
    k = sw.key_of(tile)
    if record is None:
      if self.live["shrink"].pop(k, None) is not None:
        self.save_live()
      return
    self.live["shrink"][k] = record
    self.save_live()

  def commit(self, tile, entry: dict, files: dict) -> None:
    """work/T -> ready/T, as in the module docstring. Caller holds the lock."""
    k = sw.key_of(tile)
    work, ready = self.work_dir(tile), sw.tile_path(self.root, sw.READY_SUB, tile)
    self.next["tiles"][k] = dict(entry, state="pending")
    self.save_next()
    sw.write_json_atomic(sw.staged_hashes_path(self.root, tile), {"files": files})
    sw._fsync_dir(work)
    if os.path.lexists(ready):  # never expected (staged tiles are not checked again); keeps the commit atomic anyway
      sw._rename(ready, sw.old_path(self.root, tile, "_ready"))
    sw._rename(work, ready)
    sw._fsync_dir(os.path.dirname(ready))
    self.next["tiles"][k]["state"] = "ready"
    self.save_next()
    remove_part(self.part(tile))
    self.drop_request(tile)

  def discard_work(self, tile, keep_part: bool = False) -> None:
    if os.path.lexists(self.work_dir(tile)):
      sw._rename(self.work_dir(tile), sw.old_path(self.root, tile, "_work"))
    if not keep_part:
      remove_part(self.part(tile))

  # ---- deletes (under the lock): folders go to old/, removed later outside the lock
  def delete_tiles(self, tiles) -> dict:
    report = {"live": 0, "staged": 0}
    for t in tiles:
      k = sw.key_of(t)
      live, ready = sw.tile_path(self.root, sw.OFFLINE, t), sw.tile_path(self.root, sw.READY_SUB, t)
      if os.path.lexists(live):
        sw._rename(live, sw.old_path(self.root, t, "_live"))
        report["live"] += 1
      if os.path.lexists(ready):
        sw._rename(ready, sw.old_path(self.root, t, "_ready"))
        report["staged"] += 1
      self.discard_work(t)
      for path in (sw.live_hashes_path(self.root, t), sw.staged_hashes_path(self.root, t)):
        sw._remove(path)
      for section in ("tiles", "no_data", "shrink"):
        self.live[section].pop(k, None)
      self.next["tiles"].pop(k, None)
      self.requests.pop(k, None)
    self.save_live()
    self.save_next()
    self.save_requests()
    for lat in {t[0] for t in tiles}:
      for sub in (sw.OFFLINE, sw.READY_SUB):
        try:
          os.rmdir(os.path.join(self.root, sub, str(lat)))
        except OSError:
          pass
    return report

  def clear_all(self) -> None:
    """DELETE ALL: every map folder and the store files go to old/ (removed outside the lock by the caller)."""
    tag = f"all_{time.time_ns()}"
    old = sw.next_path(self.root, sw.OLD)
    for name, path in (("offline", os.path.join(self.root, sw.OFFLINE)), ("ready", sw.next_path(self.root, sw.READY)),
                       ("work", sw.next_path(self.root, sw.WORK)), ("hashes", os.path.join(self.root, sw.HASHES)),
                       ("next_hashes", sw.next_path(self.root, sw.NEXT_HASHES))):
      if os.path.lexists(path):
        sw._rename(path, os.path.join(old, f"{name}_{tag}"))
    for path in (sw.live_manifest_path(self.root), sw.next_manifest_path(self.root), sw.next_path(self.root, sw.REQUESTS),
                 os.path.join(self.root, sw.SIZES_NAME)):
      sw._remove(path)
    self.live, self.next, self.requests = sw.empty_manifest(), sw.empty_manifest(), {}

  def empty_old(self, budget_s: float = 0.5, monotonic=time.monotonic, tick=None) -> bool:
    """Removes replaced and deleted copies (outside the lock), calling tick() between folders (heartbeat); True when empty."""
    old = sw.next_path(self.root, sw.OLD)
    try:
      names = sorted(os.listdir(old))
    except FileNotFoundError:
      return True
    start = monotonic()
    for name in names:
      sw._rmtree(os.path.join(old, name))
      if tick is not None:
        tick()
      if monotonic() - start > budget_s:
        return False
    return True


class _RecoveringLock(sw.Lock):
  """The store's lock: takes it, then finishes an interrupted swap batch first (R2-1) and reloads the manifests."""

  def __init__(self, store: Store, wait: float):
    super().__init__(store.root, wait)
    self.store = store

  def __enter__(self):
    super().__enter__()
    try:
      if os.path.exists(sw.next_path(self.store.root, sw.SWAP_JOURNAL)):
        sw.recover(self.store.root)
        self.store.reload()
    except BaseException:
      super().__exit__(None, None, None)
      raise
    return self
