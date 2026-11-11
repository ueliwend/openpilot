"""Offline maps for several locations at once (Inster port, E9): selection, requests, deletes, gates and status.

Settings keys (common/params_keys.h):
- `OsmSelectedLocations` (PERSISTENT | BACKUP, JSON): the selected locations, the only source of truth (offline_selection.py).
- `OsmDownloadRequest` (CLEAR_ON_MANAGER_START, STRING): "all", or location ids ("AT", "US:CA", comma separated). An id not
  selected yet is added to the selection first. The old `OsmDbUpdatesCheck` counts as "all".
- `OsmDeleteLocation` (CLEAR_ON_MANAGER_START, STRING): location ids, or "OTHER" (map areas no selected location uses).
  The location leaves the selection and the queue at once; its tiles are deleted once no request of ours runs and mapd
  is idle, keeping every tile another selected location uses.
- `OsmLocationStatus` (PERSISTENT, JSON): written only here, derived from the tiles on disk (see status()).
Every value sunnylink may write is validated here; the policies below are constants, not settings.

Gates before each location is sent (offline_update.py runs the rest), in this order:
- offroad (offline_update.py) and mapd idle;
- network: any unmetered network (owner, 11 Nov 2026): Wi-Fi or Ethernet the device reports unmetered, and cellular
  while the device reports it unmetered (never comma's SIM: hardware.py get_network_metered); otherwise wait;
- server check, in a worker thread so the manager loop (and map speed data) never waits on the network: a HEAD request
  for each tile not on disk (download size, 404 = no data on the server), then the probe: the first tile with data must
  start with the gzip magic (offline_server.py). No answer, or a server error (e.g. HTTP 403, shown with its code), makes
  the run wait 1, 2, 4 ... min (at most 1 h); a request by hand (DOWNLOAD, Update all, ADD) asks again at once;
- storage (offline_storage.py): the budget against the capacity with the 3 h drive buffer; refused = the run stops;
- a reply that is not gzip skips the location (failed); otherwise it is sent.
"""
import os
import threading
import time
from datetime import datetime

from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.mapd.offline_regions import (Tile, entry_id, entry_tiles, location_title, parse_entry_id, tile_dir,
                                                        tile_stats, tiles_on_disk, union_tiles)
from openpilot.sunnypilot.mapd.offline_selection import Selection, id_of
from openpilot.sunnypilot.mapd.offline_server import GZIP, NO_ANSWER, SERVER_ERROR, SIZES_NAME, SizeCache, check
from openpilot.sunnypilot.mapd.offline_storage import (MIN_DRIVE_BUFFER_HOURS, budget, disk_usage, drives_bytes, estimate_tiles,
                                                        maps_bytes, plan_download)
from openpilot.sunnypilot.mapd.offline_update import GO, OFFLINE, PREV, SKIP, STOP, WAIT, OfflineMapUpdater, _rmtree

REQUEST_KEY = "OsmDownloadRequest"
DELETE_KEY = "OsmDeleteLocation"
STATUS_KEY = "OsmLocationStatus"
OTHER = "OTHER"
ALL = "all"
STATUS_VERSION = 1

UNMETERED_TYPES = ("wifi", "ethernet", "cell2G", "cell3G", "cell4G", "cell5G")  # owner: any unmetered network
SERVER_RETRY_SECONDS = 60.0
SERVER_RETRY_MAX_SECONDS = 3600.0
CHECK_MAX_AGE_SECONDS = 300.0
DISK_REFRESH_SECONDS = 10.0
STORAGE_REFRESH_SECONDS = 60.0

WAIT_NETWORK = "waiting for an unmetered network"
WAIT_SERVER = "map server not reachable"
WAIT_SERVER_ERROR = "map server error"
WAIT_CHECKING = "checking the map server"
SKIP_NOT_GZIP = "map server error (reply not gzip)"
STOP_STORAGE = "not enough storage"


def network_allows(network: tuple[str, bool] | None) -> bool:
  """Owner's answer 2: any network the device reports unmetered (Wi-Fi, Ethernet, a non-comma SIM with GsmMetered off)."""
  return network is not None and network[0] in UNMETERED_TYPES and not network[1]


def thread_runner(work) -> None:
  threading.Thread(target=work, daemon=True, name="mapd_server_check").start()


def parse_ids(value) -> list[str]:
  if not isinstance(value, str):
    return []
  return [t.strip() for t in value.split(",") if t.strip()]


class OfflineMaps:
  def __init__(self, params, root: str, updater: OfflineMapUpdater, server, disk=disk_usage, drives=drives_bytes,
               now=lambda: datetime.now().timestamp(), monotonic=time.monotonic, runner=thread_runner):
    self.params = params
    self.root = root
    self.updater = updater
    self.server = server
    self._disk = disk
    self._drives = drives
    self._now = now
    self._monotonic = monotonic
    self.selection = Selection(params)
    self.cache = SizeCache(root)
    self.pending_deletes: list[str] = []
    self.records: dict[str, dict] = self._load_records()
    self.message = ""
    self.run_ids: list[str] = []  # locations of the run in progress, in order
    self.run_done: list[str] = []
    self._retry_at = 0.0
    self._failures = 0
    self._retry_wall = 0
    self._server_wait = ""
    self.server_error: int | None = None  # HTTP status of the last server error (shown in the status)
    self._runner = runner
    self._check: dict | None = None  # the server check of the location at the head of the queue
    self._check_lock = threading.Lock()
    self._disk_info: dict[Tile, tuple[int, float]] = {}
    self._disk_at = -1e9
    self._storage: dict = {}
    self._storage_at = -1e9
    self._last_status = None
    updater.gate = self._gate
    updater.on_entry_end = self._entry_end

  # ---- interface for mapd_manager
  def step(self) -> None:
    try:
      selection = self.selection.read()
      self._take_deletes()
      self._take_requests(selection)
    except Exception:
      cloudlog.exception("mapd offline maps: reading the settings failed")
    self.updater.step()
    try:
      self._run_deletes()
      self._write_status()
    except Exception:
      cloudlog.exception("mapd offline maps: delete or status failed, retrying next second")

  def ready_for_clear(self) -> bool:
    """DELETE ALL (Mapd_ClearCache): queued deletes and entries not yet sent are dropped; waits for a running entry."""
    self.pending_deletes.clear()
    self.run_ids, self.run_done = [], []
    return self.updater.ready_for_clear()

  def cleared(self) -> None:
    """After DELETE ALL (mapd_manager.clear_downloaded_maps): no selection, no records, no size cache, no pending keys."""
    try:
      os.remove(os.path.join(self.root, SIZES_NAME))
    except FileNotFoundError:
      pass
    for key in (REQUEST_KEY, DELETE_KEY):
      self.params.remove(key)
    self.selection.write([])
    self.records = {}
    self.cache = SizeCache(self.root)
    self.message = ""
    self._check = None
    self._retry_at, self._failures, self._server_wait, self.server_error = 0.0, 0, "", None
    self._disk_at = self._storage_at = -1e9
    self.params.remove(STATUS_KEY)
    self._last_status = None

  # ---- requests and deletes
  def _take_requests(self, selection: list[dict]) -> None:
    ids = parse_ids(self.params.get(REQUEST_KEY))
    if ids:
      self.params.remove(REQUEST_KEY)
    if self.params.get_bool("OsmDbUpdatesCheck"):
      self.params.put_bool("OsmDbUpdatesCheck", False, block=True)
      ids.append(ALL)
    if not ids:
      return
    wanted: list[str] = []
    for i in ids:
      if i.lower() == ALL:
        wanted += [id_of(e) for e in selection]
      elif self.selection.add(i) is not None:
        wanted.append(entry_id(*parse_entry_id(i)))
      else:
        cloudlog.warning(f"mapd offline maps: unknown location {i!r} in {REQUEST_KEY}")
    entries = [parse_entry_id(i) for i in dict.fromkeys(wanted)]
    if not entries:
      return
    self._retry_at, self._failures, self._server_wait = 0.0, 0, ""  # a request by hand asks the server at once
    if not self.updater.busy:
      self.run_ids, self.run_done = [], []
      self.message = ""
    for kind, code in entries:
      if entry_id(kind, code) not in self.run_ids:
        self.run_ids.append(entry_id(kind, code))
    self.updater.enqueue(entries)
    cloudlog.event("mapd_offline_request", locations=[entry_id(*e) for e in entries])

  def _take_deletes(self) -> None:
    ids = parse_ids(self.params.get(DELETE_KEY))
    if not ids:
      return
    self.params.remove(DELETE_KEY)
    for i in ids:
      if i.upper() == OTHER:
        if OTHER not in self.pending_deletes:
          self.pending_deletes.append(OTHER)
        continue
      parsed = parse_entry_id(i)
      if parsed is None:
        cloudlog.warning(f"mapd offline maps: unknown location {i!r} in {DELETE_KEY}")
        continue
      location_id = entry_id(*parsed)
      self.selection.remove(location_id)
      self.updater.dequeue(parsed)
      self.records.pop(location_id, None)
      if location_id in self.run_ids and location_id not in self.run_done:
        self.run_ids.remove(location_id)
      if location_id not in self.pending_deletes:
        self.pending_deletes.append(location_id)
    self._disk_at = -1e9

  def _run_deletes(self) -> None:
    if not self.pending_deletes or not self.updater.tiles_idle():
      return
    keep = union_tiles(id_of(e) for e in self.selection.read())
    for location_id in self.pending_deletes:
      candidates = tiles_on_disk(self.root) if location_id == OTHER else entry_tiles(location_id)
      gone, freed = [], 0
      for t in candidates:
        if t in keep:
          continue
        stats = tile_stats(self.root, t)
        if stats is None:
          continue
        _rmtree(tile_dir(self.root, OFFLINE, t))
        try:
          os.rmdir(os.path.dirname(tile_dir(self.root, OFFLINE, t)))
        except OSError:
          pass  # other tiles of that latitude remain
        gone.append(t)
        freed += stats[0]
      cloudlog.event("mapd_offline_delete", location=location_id, tiles=len(gone), freed=freed)
    self.pending_deletes = []
    self._disk_at = self._storage_at = -1e9

  # ---- gates (called by the updater before each send, offroad and with mapd idle)
  def _gate(self, kind: str, code: str, tiles: list[Tile]) -> tuple[str, str]:
    """Never waits on the network: the server check runs in a worker (self._runner); the gate starts it, answers WAIT
    while it runs, and decides from its result at a later step."""
    if not network_allows(self.updater.link.network()):
      return WAIT, WAIT_NETWORK
    if self._monotonic() < self._retry_at:
      return WAIT, self._server_wait or WAIT_SERVER
    location_id = entry_id(kind, code)
    job = self._check
    if job is None or job["id"] != location_id or job["tiles"] != tuple(tiles):
      self._start_check(location_id, tiles)
      job = self._check
    with self._check_lock:
      result = job["result"]
    if result is None:
      return WAIT, WAIT_CHECKING
    self._check = None
    if self._monotonic() - job["started"] > CHECK_MAX_AGE_SECONDS:
      return WAIT, WAIT_CHECKING  # an old answer (e.g. the car was driven meanwhile): ask again at the next step
    return self._decide(kind, code, tiles, result)

  def _start_check(self, location_id: str, tiles: list[Tile]) -> None:
    missing = []
    for t in tiles:
      stats = tile_stats(self.root, t)
      if stats is None:
        missing.append(t)
      else:
        self.cache.set_disk(t, stats[0])
    no_data = {t for t in tiles if self.cache.no_data(t)}
    job = {"id": location_id, "tiles": tuple(tiles), "started": self._monotonic(), "result": None}
    self._check = job
    server = self.server

    def work():
      try:
        result = check(server, list(tiles), missing, no_data)
      except Exception:
        cloudlog.exception("mapd offline maps: server check failed")
        result = {"sizes": {}, "answer": NO_ANSWER, "code": None, "no_data": []}
      with self._check_lock:
        job["result"] = result
    self._runner(work)

  def _backoff(self, reason: str) -> tuple[str, str]:
    """Wait before asking the server again: 1, 2, 4 ... min, at most an hour; a request by hand asks at once."""
    self._failures += 1
    delay = min(SERVER_RETRY_SECONDS * 2 ** (self._failures - 1), SERVER_RETRY_MAX_SECONDS)
    self._retry_at, self._retry_wall = self._monotonic() + delay, round(self._now() + delay)
    self._server_wait = reason
    return WAIT, reason

  def _decide(self, kind: str, code: str, tiles: list[Tile], result: dict) -> tuple[str, str]:
    for t, size in result["sizes"].items():
      self.cache.set_download(t, size)
    for t in result["no_data"]:
      self.cache.set_download(t, 0)
    self.cache.save()
    if result["answer"] == NO_ANSWER:
      return self._backoff(WAIT_SERVER)
    if result["answer"] == SERVER_ERROR:
      reason = f"{WAIT_SERVER_ERROR} (HTTP {result['code']})"
      self.message = reason
      self.server_error = result["code"]
      return self._backoff(reason)
    self.server_error = None
    on_disk, server = {}, {}
    for t in tiles:
      stats = tile_stats(self.root, t)
      if stats is not None:
        on_disk[t] = stats[0]
      else:
        server[t] = result["sizes"].get(t, self.cache.download(t))
    est = estimate_tiles(on_disk, server, self.cache.ratio())
    plan = plan_download(self.root, est, disk=self._disk(self.root), drives=self._drives())
    b, replay = plan.budget, plan.replay
    cloudlog.event("mapd_offline_storage", location=entry_id(kind, code), ok=b.ok, maps_now=b.maps_now, maps_after=b.maps_after,
                   peak=b.peak, shortfall=b.shortfall, max_maps=b.max_maps, unknown_tiles=est.unknown,
                   deleter_minutes=replay.minutes, deleter_moved=replay.moved, deleter_lost_minutes=replay.lost_minutes,
                   deleter_first_written=replay.first_written, deleter_last_written=replay.last_written)
    if not b.ok:
      need, allowed = (b.maps_after + b.peak - b.maps_now) / 1e9, max(0, b.max_maps) / 1e9
      self.message = f"{STOP_STORAGE}: {location_title(kind, code)} needs about {need:.1f} GB, " + \
                     f"maps may use {allowed:.1f} GB in all to keep {MIN_DRIVE_BUFFER_HOURS:.0f} h of drive recording"
      return STOP, STOP_STORAGE
    self._failures, self._server_wait = 0, ""
    if result["answer"] != GZIP:
      return SKIP, SKIP_NOT_GZIP
    if self.message.startswith(WAIT_SERVER_ERROR):
      self.message = ""
    return GO, ""

  # ---- results and status
  def _entry_end(self, kind: str, code: str, reason: str, refreshed: list[Tile], tiles: list[Tile]) -> None:
    location_id = entry_id(kind, code)
    if location_id not in self.run_done:
      self.run_done.append(location_id)
    for t in refreshed:
      stats = tile_stats(self.root, t)
      if stats is not None:
        self.cache.set_disk(t, stats[0])
    self.cache.save()
    self._disk_at = self._storage_at = -1e9
    if location_id not in [id_of(e) for e in self.selection.read()]:
      return
    tiles = tiles or entry_tiles(location_id)
    expected = [t for t in tiles if not self.cache.no_data(t)]
    n = len(set(refreshed) & set(expected))
    result = "ok" if n >= len(expected) else ("partial" if n > 0 else "failed")
    self.records[location_id] = {"result": result, "refreshed": n, "expected": len(expected), "reason": reason,
                                 "time": round(self._now())}

  def _load_records(self) -> dict[str, dict]:
    status = self.params.get(STATUS_KEY)
    out = {}
    try:
      for location_id, entry in status["entries"].items():
        last = entry.get("last")
        if parse_entry_id(location_id) and isinstance(last, dict) and last.get("result") in ("ok", "partial", "failed"):
          out[location_id] = {"result": last["result"], "refreshed": int(last.get("refreshed", 0)),
                              "expected": int(last.get("expected", 0)), "reason": str(last.get("reason", "")),
                              "time": int(last.get("time", 0))}
    except (TypeError, KeyError, AttributeError, ValueError):
      pass
    return out

  def _refresh_disk(self) -> None:
    now = self._monotonic()
    if now - self._disk_at < DISK_REFRESH_SECONDS:
      return
    self._disk_at = now
    self._disk_info = {}
    for sub in (OFFLINE, PREV):  # a tile moved aside while its location downloads again still counts as present
      for t in tiles_on_disk(self.root, sub):
        stats = tile_stats(self.root, t, sub)
        if stats is not None and t not in self._disk_info:
          self._disk_info[t] = stats

  def _refresh_storage(self) -> None:
    now = self._monotonic()
    if now - self._storage_at < STORAGE_REFRESH_SECONDS:
      return
    self._storage_at = now
    try:
      disk = self._disk(self.root)
      maps = maps_bytes(self.root)
      b = budget(disk, self._drives(), maps, maps, 0)
      self._storage = {"capacity": disk.capacity, "available": disk.available, "maps": maps, "drives": b.drives,
                       "floor": b.floor, "other": b.other, "max_maps": b.max_maps, "buffer_hours": round(b.buffer_hours_after, 2)}
    except OSError:
      self._storage = {}

  def status(self) -> dict:
    """OsmLocationStatus: per selected location its state, tiles and bytes from disk; the run; other map areas; storage."""
    self._refresh_disk()
    self._refresh_storage()
    selection = self.selection.read()
    ids = [id_of(e) for e in selection]
    current = self.updater.current
    current_id = entry_id(*current) if current else None
    queued = {entry_id(*e) for e in self.updater.queue} | {entry_id(*e) for e in self.updater._skipped}
    entries = {}
    for e in selection:
      location_id = id_of(e)
      tiles = entry_tiles(location_id)
      present = [t for t in tiles if t in self._disk_info]
      expected = [t for t in tiles if t in self._disk_info or not self.cache.no_data(t)]
      others = [i for i in ids if i != location_id]
      other_tiles = union_tiles(others)
      exclusive = [t for t in present if t not in other_tiles]
      shared_with = [x["title"] for x in selection if id_of(x) in others and set(entry_tiles(id_of(x))) & set(present)]
      record = self.records.get(location_id)
      if location_id == current_id:
        state = "downloading"
      elif location_id in queued:
        state = "queued"
      elif expected and not present:
        state = "not_downloaded"
      elif len(present) < len(expected):
        state = "partial"
      elif record is not None:
        state = record["result"]
      else:
        state = "ok"
      entries[location_id] = {
        "kind": e["kind"], "code": e["code"], "title": e["title"], "state": state,
        "tiles_total": len(tiles), "tiles_expected": len(expected), "tiles_present": len(present),
        "tiles_refreshed": record["refreshed"] if record else 0,
        "bytes": sum(self._disk_info[t][0] for t in present),
        "exclusive_bytes": sum(self._disk_info[t][0] for t in exclusive),
        "shared_tiles": len(present) - len(exclusive), "shared_with": shared_with,
        "date": round(min(self._disk_info[t][1] for t in present)) if present else 0,
        "last": record,
      }
    keep = union_tiles(ids)
    orphans = [t for t in self._disk_info if t not in keep]
    progress_done = self.updater.link.downloaded_files() if current else None
    run = {
      "active": self.updater.busy, "waiting": self.updater.waiting if self.updater.busy else "",
      "current": current_id, "index": min(len(self.run_done) + 1, len(self.run_ids)) if self.updater.busy else 0,
      "count": len(self.run_ids) if self.updater.busy else 0,
      "tiles_done": progress_done or 0, "tiles_total": len(entry_tiles(current_id)) if current_id else 0,
      "deletes_pending": list(self.pending_deletes),
      "server_error": self.server_error if self.updater.busy else None,
      "retry_at": self._retry_wall if self.updater.busy and self._server_wait else 0,  # wall clock: constant while waiting
    }
    return {"version": STATUS_VERSION, "entries": entries, "order": ids, "run": run,
            "other": {"tiles": len(orphans), "bytes": sum(self._disk_info[t][0] for t in orphans)},
            "storage": self._storage, "message": self.message}

  def _write_status(self) -> None:
    status = self.status()
    if status != self._last_status:
      self.params.put(STATUS_KEY, status, block=True)
      self._last_status = status

