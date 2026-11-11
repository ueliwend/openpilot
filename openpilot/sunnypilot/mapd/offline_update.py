"""Keeps offline maps readable while pfeiferj's mapd v1.12.0 downloads them again (Inster port).

The binary opens each map file without truncation (download.go:240 at v1.12.0, fixed upstream only in aa04ba4 / v2.4.0),
so a new file that is shorter than the old one keeps the old file's tail, and that map area becomes unreadable. The
binary is not changed. Instead, for one location (nation or US state) per request:

- prepare: every tile folder of the location that already exists is moved aside to `offline_prev/` (a rename, no extra
  space), after a journal (`offline_prev/journal.json`) has recorded the tiles, their order and the old file names;
  the binary then always writes into empty folders;
- while it runs (1 Hz): a tile the binary has finished whose new folder holds at least the old file names, none of them
  empty, is kept, and its old copy is deleted. Every other tile waits until the binary is idle;
- at the end (binary idle): the remaining tiles are decided from disk with the "frontier" rule (below); a tile without a
  complete new copy gets its old copy back;
- after a crash or power cut the same rule runs from the journal before anything else.

Frontier = the last tile, in the binary's order, that has a folder in `offline/` (only the binary can have made it, since
all old folders were moved aside) or a download file in `tmp/offline/`. The binary works on one tile at a time, in
order, so every tile before the frontier is finished. The frontier tile is finished only if its folder exists and its
download file is gone (the binary removes it after any extraction that does not panic). If `tmp/` is gone, the binary
reached the end of the location (download.go:273) and every tile is finished.

The binary is idle when its request (`OSMDownloadLocations` in the memory params) exists and is empty after the
manager wrote or removed it: the binary empties it only at the end of a download loop (download.go:156) and when it
starts (params.go:86-88). Watchdog: the binary restarts a location from its first tile after a panic (a non-gzip reply,
download.go:211-218, recovered in mapd.go:211-224) and keeps the request, so it would loop. A restart is seen as
downloaded_files falling, a download file created again for a tile at or before the frontier seen so far, or a download
file rewritten without growing past its largest size. A stall is no tile advance and no download file growing past its
largest size for 10 min. Either way the request is cleared; the entry ends as failed once the binary is idle.

Downloads start only offroad. Design: inster-dev results/2026-11-11_offline_maps_multi_country/DESIGN.md (rev 4),
sections 4.3-4.5; this is the subset for the single-country "Database Update" (results/2026-11-11_map_redownload_fix).
"""
import json
import os
import shutil
import time
from dataclasses import dataclass, field

from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.mapd.offline_regions import Tile, tile_archive, tile_dir, tile_key, tile_order

OFFLINE = "offline"
PREV = "offline_prev"  # names must not start with "v" or be "db" / "tmp": mapd_manager.get_files_for_cleanup, download.go:273
TMP = "tmp"
JOURNAL_NAME = "journal.json"
JOURNAL_VERSION = 1
STALL_SECONDS = 600.0
TICK_POLL_SECONDS = 0.05
TICK_WAIT_SECONDS = 2.0

START, IDLE, QUEUED, RUNNING, DRAINING, RECOVERING = "start", "idle", "queued", "running", "draining", "recovering"


# Filesystem steps that change state. Module level, so the tests can stop the process between any two of them.
def _rename(src: str, dst: str) -> None:
  os.makedirs(os.path.dirname(dst), exist_ok=True)
  os.rename(src, dst)


def _rmtree(path: str) -> None:
  if os.path.isdir(path):
    shutil.rmtree(path)


def _remove(path: str) -> None:
  try:
    os.remove(path)
  except FileNotFoundError:
    pass


def _fsync_dir(path: str) -> None:
  try:
    fd = os.open(path, os.O_RDONLY)
  except FileNotFoundError:
    return
  try:
    os.fsync(fd)
  finally:
    os.close(fd)


def _write_json_atomic(path: str, data: dict) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(data, f)
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp, path)
  _fsync_dir(os.path.dirname(path))


def listing(path: str) -> dict[str, int] | None:
  """File names and sizes directly inside a tile folder; None if the folder does not exist."""
  try:
    return {e.name: e.stat().st_size for e in os.scandir(path) if e.is_file(follow_symlinks=False)}
  except (FileNotFoundError, NotADirectoryError):
    return None


class ParamsLink:
  """The binary's interface through openpilot params (production)."""

  def __init__(self, params, mem_params, send):
    self._params = params
    self._mem_params = mem_params
    self._send = send
    self.request_path = mem_params.get_param_path("OSMDownloadLocations")

  def send(self, nations: list[str], states: list[str]) -> None:
    self._send(nations, states)

  def clear_request(self) -> None:
    self._mem_params.remove("OSMDownloadLocations")

  def remove_progress(self) -> None:
    self._params.remove("OSMDownloadProgress")

  def downloaded_files(self) -> int | None:
    progress = self._params.get("OSMDownloadProgress")
    try:
      return int(progress["downloaded_files"])
    except (TypeError, KeyError, ValueError):
      return None

  def is_offroad(self) -> bool:
    return self._params.get_bool("IsOffroad")


@dataclass
class Watch:
  """Watchdog state for the request that is running."""
  last_advance: float
  max_frontier: int = -1
  last_downloaded: int = -1
  archives: dict[int, tuple[int, int]] = field(default_factory=dict)  # tile index -> (size, mtime_ns) at the last poll
  max_size: dict[int, int] = field(default_factory=dict)


class OfflineMapUpdater:
  def __init__(self, root: str, link, monotonic=time.monotonic, sleep=time.sleep):
    self.root = root
    self.link = link
    self._monotonic = monotonic
    self._sleep = sleep
    self.state = START
    self.queue: list[tuple[str, str]] = []  # (kind, code), kind "nation" or "state"
    self._watch: Watch | None = None
    self._journal: dict | None = None  # the running request's journal (also on disk)
    self._failure = ""
    self._released: set[int] = set()

  # ---- paths and journal
  @property
  def journal_path(self) -> str:
    return os.path.join(self.root, PREV, JOURNAL_NAME)

  def load_journal(self) -> dict | None:
    """The journal; None if there is none; {"invalid": True} if it cannot be read (then every old copy goes back)."""
    try:
      with open(self.journal_path) as f:
        journal = json.load(f)
      if journal.get("version") != JOURNAL_VERSION:
        raise ValueError(f"journal version {journal.get('version')}")
      journal["tiles"] = [(int(t[0]), int(t[1])) for t in journal["tiles"]]
      if not isinstance(journal["old"], dict) or journal["phase"] not in ("prepared", "sent"):
        raise ValueError("journal fields")
      return journal
    except FileNotFoundError:
      return None
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
      cloudlog.exception("mapd offline update: journal unreadable, restoring every old copy")
      return {"invalid": True}

  def _request_state(self) -> str:
    try:
      return "empty" if os.path.getsize(self.link.request_path) == 0 else "pending"
    except FileNotFoundError:
      return "missing"

  # ---- interface for mapd_manager
  def accepts_request(self) -> bool:
    return self.state == IDLE and not self.queue

  def request(self, nations: list[str], states: list[str]) -> None:
    """Queue one entry per location, nations first (the binary's order). Sent one at a time."""
    self.queue = [("nation", n) for n in nations] + [("state", s) for s in states]
    if self.queue:
      if self.state == IDLE:
        self.state = QUEUED  # otherwise recovery runs first, then the queue
    else:
      self.link.send([], [])  # nothing to download: as before, an empty request (mapd ignores it)

  def ready_for_clear(self) -> bool:
    """Mapd_ClearCache: allowed once no request of ours runs. Entries not yet sent are dropped at once."""
    self.queue.clear()
    if self.state in (IDLE, QUEUED):
      self.state = IDLE
      return True
    return False

  def step(self) -> None:
    try:
      self._step()
    except Exception:
      cloudlog.exception("mapd offline update: step failed, retrying next second")

  # ---- state machine
  def _step(self) -> None:
    if self.state == START:
      # A request left in /dev/shm survives a manager restart without a reboot; it is not ours to run again.
      self.link.clear_request()
      self.state = RECOVERING
    if self.state == RECOVERING:
      if self._request_state() != "empty":
        return
      self._reconcile_leftovers("recovered after a restart")
      self.state = QUEUED if self.queue else IDLE
    if self.state == QUEUED:
      self._start_next()
    elif self.state == RUNNING:
      self._poll_running()
    elif self.state == DRAINING:
      if self._request_state() == "empty":
        self._end_request()

  def _start_next(self) -> None:
    if not self.queue:
      self.state = IDLE
      return
    if not self.link.is_offroad() or self._request_state() != "empty":
      return  # onroad, or mapd busy / not seen alive since the request was last removed
    kind, code = self.queue[0]
    tiles = tile_order(kind, code) or []
    self._reconcile_leftovers("recovered before a new request")
    self._prepare(kind, code, tiles)
    if not self._wait_for_tick():
      self._finish(self.load_journal(), "not sent (no mapd heartbeat)", log=False)
      return
    journal = self.load_journal()
    journal["phase"] = "sent"
    _write_json_atomic(self.journal_path, journal)
    self.link.remove_progress()
    self.link.send([code] if kind == "nation" else [], [code] if kind == "state" else [])
    self.queue.pop(0)
    self._journal = journal
    self._watch = Watch(last_advance=self._monotonic())
    self._failure = ""
    self._released = set()
    self.state = RUNNING

  def _wait_for_tick(self) -> bool:
    """Send right after the binary emptied its request: it then sleeps about 1 s before reading it again
    (mapd.go loop: download check, then 1 s sleep), so a request written now is not lost to the read/clear race."""
    self.link.clear_request()
    waited = 0.0
    while waited < TICK_WAIT_SECONDS:
      if self._request_state() != "missing":
        return True
      self._sleep(TICK_POLL_SECONDS)
      waited += TICK_POLL_SECONDS
    return False

  def _prepare(self, kind: str, code: str, tiles: list[Tile]) -> None:
    _rmtree(os.path.join(self.root, TMP))  # a stale download file would move the frontier
    old = {}
    for t in tiles:
      files = listing(tile_dir(self.root, OFFLINE, t))
      if files is not None:
        old[tile_key(t)] = files
    journal = {"version": JOURNAL_VERSION, "phase": "prepared", "kind": kind, "code": code,
               "tiles": [list(t) for t in tiles], "old": old}
    _write_json_atomic(self.journal_path, journal)
    touched = set()
    for t in tiles:
      if tile_key(t) in old:
        src, dst = tile_dir(self.root, OFFLINE, t), tile_dir(self.root, PREV, t)
        _rename(src, dst)
        touched.update((os.path.dirname(src), os.path.dirname(dst)))
    for d in touched:
      _fsync_dir(d)

  def _traces(self, tiles: list[Tile]) -> tuple[int, dict[int, tuple[int, int]]]:
    frontier, archives = -1, {}
    for i, t in enumerate(tiles):
      # Folder first, then download file: mapd creates the file before the folder and removes it only after the
      # extraction, so "folder seen, then no file" proves the extraction of that tile had ended.
      if os.path.isdir(tile_dir(self.root, OFFLINE, t)):
        frontier = i
      try:
        st = os.stat(tile_archive(self.root, t))
        archives[i] = (st.st_size, st.st_mtime_ns)
        frontier = i
      except FileNotFoundError:
        pass
    return frontier, archives

  def _finished(self, i: int, t: Tile, frontier: int, archives: dict) -> bool:
    return i < frontier or (i == frontier and i not in archives and os.path.isdir(tile_dir(self.root, OFFLINE, t)))

  def _new_copy_complete(self, journal: dict, t: Tile) -> bool:
    """At least the old file names, and no empty file: the server writes no file for an area without ways
    (generate_offline.go:208), so an empty one is a failed write (e.g. disk full: every later file of the tile ends empty)."""
    new = listing(tile_dir(self.root, OFFLINE, t))
    return new is not None and set(journal["old"][tile_key(t)]) <= set(new) and all(size > 0 for size in new.values())

  def _poll_running(self) -> None:
    if self._request_state() == "empty":
      self._end_request()
      return
    journal = self._journal
    tiles = journal["tiles"]
    frontier, archives = self._traces(tiles)
    reason = self._watchdog(frontier, archives)
    if reason:
      self._failure = reason
      cloudlog.warning(f"mapd offline update: {reason}; request cleared ({journal['code']})")
      self.link.clear_request()
      self.state = DRAINING
      return
    # Keep finished tiles whose new copy is complete. Old copies of the others go back only once mapd is idle:
    # after a panic restart mapd writes the earlier tiles again, and must not write into a restored old copy.
    for i, t in enumerate(tiles[:frontier + 1]):
      if i in self._released or tile_key(t) not in journal["old"] or not self._finished(i, t, frontier, archives):
        continue
      if os.path.isdir(tile_dir(self.root, PREV, t)) and self._new_copy_complete(journal, t):
        _rmtree(tile_dir(self.root, PREV, t))
        self._released.add(i)

  def _watchdog(self, frontier: int, archives: dict[int, tuple[int, int]]) -> str:
    w = self._watch
    downloaded = self.link.downloaded_files()
    reason = ""
    if downloaded is not None and downloaded < w.last_downloaded:
      reason = "mapd restarted the location (downloaded_files fell)"
    advance = frontier > w.max_frontier or (downloaded is not None and downloaded > w.last_downloaded)
    for i, (size, mtime) in archives.items():
      seen = w.archives.get(i)
      if seen is None and i <= w.max_frontier:
        reason = reason or "mapd restarted the location (download file created again before the frontier)"
      elif seen is not None and mtime != seen[1] and size <= w.max_size.get(i, -1):
        reason = reason or "mapd restarted the location (download file rewritten without growing past its largest size)"
      if size > w.max_size.get(i, -1):
        advance = True
        w.max_size[i] = size
    now = self._monotonic()
    if advance:
      w.last_advance = now
    elif not reason and now - w.last_advance > STALL_SECONDS:
      reason = f"mapd download stalled for {STALL_SECONDS:.0f} s"
    w.archives = dict(archives)
    w.max_frontier = max(w.max_frontier, frontier)
    if downloaded is not None:
      w.last_downloaded = max(w.last_downloaded, downloaded)
    return reason

  def _end_request(self) -> None:
    journal = self.load_journal()
    if journal is not None:
      self._finish(journal, self._failure or "done")
    self._watch = self._journal = None
    self.state = QUEUED if self.queue else IDLE

  # ---- decisions from disk (end of a request, crash recovery, undo of an unsent prepare)
  def _finish(self, journal: dict, reason: str, log: bool = True) -> dict:
    if journal.get("invalid"):
      counts = self._restore_every_old_copy()
      self._clean_up()
      cloudlog.event("mapd_offline_update", result=f"{reason}; journal unreadable", **counts)
      return counts
    tiles, old = journal["tiles"], journal["old"]
    sent = journal["phase"] == "sent"
    if sent and os.path.isdir(os.path.join(self.root, TMP)):
      frontier, archives = self._traces(tiles)
      finished = [self._finished(i, t, frontier, archives) for i, t in enumerate(tiles)]
    else:
      finished = [sent] * len(tiles)  # tmp/ gone: mapd reached the end of the location; not sent: it never started
    counts = {"refreshed": 0, "kept_old": 0, "incomplete": 0, "missing": 0, "decided_before": 0}
    touched = set()
    for i, t in enumerate(tiles):
      new_dir, prev_dir = tile_dir(self.root, OFFLINE, t), tile_dir(self.root, PREV, t)
      had_old = tile_key(t) in old
      if os.path.isdir(prev_dir):
        if not sent and os.path.isdir(new_dir):
          _rmtree(prev_dir)  # not sent: a folder still in offline/ is the original, never moved
          counts["kept_old"] += 1
        elif finished[i] and self._new_copy_complete(journal, t):
          _rmtree(prev_dir)
          counts["refreshed"] += 1
        else:
          _rmtree(new_dir)
          _rename(prev_dir, new_dir)
          counts["kept_old"] += 1
        touched.update((os.path.dirname(new_dir), os.path.dirname(prev_dir)))
      elif not os.path.isdir(new_dir):
        counts["missing"] += 1  # no data: none before and none now (failed first download, or nothing on the server)
      elif not sent:
        counts["kept_old" if had_old else "missing"] += 1
      elif had_old:
        counts["refreshed" if i in self._released else "decided_before"] += 1  # old copy deleted or put back earlier
      elif finished[i]:
        counts["refreshed"] += 1
      else:
        counts["incomplete"] += 1  # first download of this tile, cut short: kept, a later update completes it
    for d in touched:
      _fsync_dir(d)
    self._clean_up()
    if log:
      cloudlog.event("mapd_offline_update", kind=journal["kind"], code=journal["code"], result=reason,
                     tiles=len(tiles), had_old=len(old), **counts)
    return counts

  def _clean_up(self) -> None:
    _rmtree(os.path.join(self.root, TMP))
    _remove(self.journal_path)
    _fsync_dir(os.path.join(self.root, PREV))
    _rmtree(os.path.join(self.root, PREV))

  def _restore_every_old_copy(self) -> dict:
    """Without a usable journal: every old copy in offline_prev/ replaces whatever is in offline/ (the state before)."""
    counts = {"kept_old": 0}
    prev_root = os.path.join(self.root, PREV)
    for lat in sorted(os.listdir(prev_root)) if os.path.isdir(prev_root) else []:
      lat_dir = os.path.join(prev_root, lat)
      for lon in sorted(os.listdir(lat_dir)) if os.path.isdir(lat_dir) else []:
        new_dir = os.path.join(self.root, OFFLINE, lat, lon)
        _rmtree(new_dir)
        _rename(os.path.join(lat_dir, lon), new_dir)
        counts["kept_old"] += 1
      _fsync_dir(lat_dir)
    return counts

  def _reconcile_leftovers(self, reason: str) -> None:
    """At start-up and before a new prepare: finish a journal still present; old copies left without one go back."""
    journal = self.load_journal()
    if journal is not None:
      self._finish(journal, reason)
    elif os.path.isdir(os.path.join(self.root, PREV)):
      self._restore_every_old_copy()
      self._clean_up()
