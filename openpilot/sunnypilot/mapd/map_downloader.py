"""map_downloader: offline map downloads in a process of their own, separate from the serving mapd (Inster port, E11).

The mapd binary never gets a download request again (mapd_manager clears any), so it never stops serving. This process:
- takes the screen's requests: `OsmDownloadRequest` ("all" or location ids, also the per-country UPDATE; `OsmDbUpdatesCheck`
  counts as "all"), `OsmDeleteLocation` (ids or "OTHER"), `Mapd_ClearCache` (DELETE ALL); requests are copied into
  `offline_next/requests.json` as soon as they are read, so they survive a power cut (the onroad request E9 dropped);
- derives one tile queue each second (DESIGN 5.3): (1) tiles of selected countries not on disk and not staged, (2) requested
  tiles, oldest request first, (3) with `OsmAutoUpdate` on: tiles last checked more than 7 days ago; staged tiles and
  tiles in backoff are skipped, a tile shared by several countries is fetched once;
- fetches one tile at a time (tile_fetch.py), stages it into `offline_next/ready/` after its checks, the content comparison
  and the shrink guard (map_store.py); staged tiles go live at the next start (offline_swap.py, before mapd) and on the
  onroad -> offroad edge while the device stays on (owner 13 Nov, Q1 a, Q6 a);
- runs at the lowest priority, also while driving: SCHED_IDLE, core 6 (cores 0-3 while 4-7 are offline), best-effort I/O 7,
  1 MB/s while driving (owner Q2 a), no lock work until the swap at start is done (or 60 s), no download for 120 s after the
  start and after the offroad -> onroad edge, pauses while the device is hot, MemAvailable < 500 MB or the network is not
  allowed (any unmetered network, roaming included: owner Q4 a);
- guards storage (offline_storage.py): sized for full recording (owner Q5 a); maps are never deleted to make room;
- writes `OsmLocationStatus` (persistent) only when it changes, and the heartbeat, byte progress and storage figures (which
  change with every recorded minute) to the memory param `OsmDownloaderLive` (review 2 R2-3).
Each 1 s step catches and logs its own errors (backoff 30 s -> 10 min); the manager does not restart a dead process, and
the screen says so when the heartbeat stops. Design: inster-dev results/2026-11-13_map_update_redesign/DESIGN.md rev 2.
"""
import ctypes
import os
import time
from urllib.parse import urlparse

from openpilot.sunnypilot.mapd import offline_swap as sw
from openpilot.sunnypilot.mapd.tile_fetch import Abort

REQUEST_KEY = "OsmDownloadRequest"
DELETE_KEY = "OsmDeleteLocation"
STATUS_KEY = "OsmLocationStatus"
LIVE_KEY = "OsmDownloaderLive"  # memory param (/dev/shm): heartbeat and byte progress
AUTO_KEY = "OsmAutoUpdate"
CLEAR_KEY = "Mapd_ClearCache"
OTHER = "OTHER"
ALL = "all"
STATUS_VERSION = 2

CAP_BYTES_PER_S = 1_000_000  # while driving (owner Q2 a)
START_PAUSE_S = 120.0  # after the manager start and after the offroad -> onroad edge
SWAP_WAIT_S = 60.0  # no lock work before the swap at start wrote its marker, or this long
AGE_LIMIT_S = 7 * 86400
MIN_MEM_AVAILABLE = 500 * 1024 * 1024
SERVER_RETRY_S, SERVER_RETRY_MAX_S = 60.0, 3600.0
CRASH_BACKOFF_S, CRASH_BACKOFF_MAX_S = 30.0, 600.0
STORAGE_REFRESH_S = 60.0
LIVE_WRITE_S = 1.0
VALID_CLOCK_AFTER = 1762992000  # 13 Nov 2026 00:00 UTC (build date): earlier wall clocks are not trusted
UNMETERED_TYPES = ("wifi", "ethernet", "cell2G", "cell3G", "cell4G", "cell5G")
CORE = 6
IOPRIO_BE_7 = (2 << 13) | 7
SYS_IOPRIO_SET = {"aarch64": 30, "x86_64": 251}

# waiting reasons (the screen maps them to texts)
W_START, W_NETWORK, W_HOT, W_MEMORY, W_SERVER, W_SERVER_ERROR, W_STORAGE = (
  "start", "network", "hot", "memory", "server", "server_error", "storage")


def network_allows(network) -> bool:
  """Owner Q4 a: any network the device reports unmetered (Wi-Fi, Ethernet, a non-comma SIM with GsmMetered 0)."""
  return network is not None and network[0] in UNMETERED_TYPES and not network[1]


def mem_available(path: str = "/proc/meminfo") -> int | None:
  try:
    with open(path) as f:
      for line in f:
        if line.startswith("MemAvailable:"):
          return int(line.split()[1]) * 1024
  except (OSError, ValueError, IndexError):
    pass
  return None


def parse_ids(value) -> list[str]:
  return [t.strip() for t in value.split(",") if t.strip()] if isinstance(value, str) else []


class Priority:
  """Lowest priority for this process (production; tests use a stand-in)."""

  def start(self) -> dict:
    out = {}
    try:
      os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
      out["sched"] = "idle"
    except (OSError, AttributeError) as e:
      out["sched"] = f"failed: {e}"
    nr = SYS_IOPRIO_SET.get(os.uname().machine)
    try:
      libc = ctypes.CDLL(None, use_errno=True)
      out["ioprio"] = "be7" if nr is not None and libc.syscall(nr, 1, 0, IOPRIO_BE_7) == 0 else "failed"
    except OSError:
      out["ioprio"] = "failed"
    self.affinity()
    return out

  @staticmethod
  def online_cpus(path: str = "/sys/devices/system/cpu/online") -> set[int]:
    cpus: set[int] = set()
    try:
      with open(path) as f:
        for part in f.read().strip().split(","):
          a, _, b = part.partition("-")
          cpus.update(range(int(a), int(b or a) + 1))
    except (OSError, ValueError):
      pass
    return cpus

  def affinity(self) -> set[int]:
    """{6} while core 6 is online (onroad), else {0-3} (offroad cores 4-7 are offline: hardware.py)."""
    want = {CORE} if CORE in self.online_cpus() else {0, 1, 2, 3}
    try:
      if os.sched_getaffinity(0) != want:
        os.sched_setaffinity(0, want)
    except OSError:
      pass
    return want


class DeviceInputs:
  """deviceState through cereal (production)."""

  def __init__(self):
    from openpilot.cereal import messaging
    self.sm = messaging.SubMaster(["deviceState"])

  def update(self) -> dict:
    self.sm.update(0)
    if not self.sm.seen["deviceState"]:
      return {"started": None, "network": None, "hot": False}
    ds = self.sm["deviceState"]
    return {"started": bool(ds.started), "network": (str(ds.networkType), bool(ds.networkMetered)),
            "hot": str(ds.thermalStatus) != "ok"}


class Cancel(Abort):
  """A running transfer stops (pause, delete, DELETE ALL); its .part is kept for a resume."""


class MapDownloader:
  def __init__(self, root: str, params, mem_params, inputs, fetcher, priority=None, disk=None, drives=None, meminfo=mem_available,
               now=time.time,  # noqa: TID251 - wall clock for checked_at and the 7-day age
               monotonic=time.monotonic, sleep=time.sleep, boot_id=sw.boot_id, clear_hook=None, legacy_cleanup=None):
    from openpilot.common.swaglog import cloudlog
    from openpilot.sunnypilot.mapd.map_store import Store
    from openpilot.sunnypilot.mapd.offline_selection import Selection
    from openpilot.sunnypilot.mapd.offline_storage import disk_usage, drives_bytes
    self.log = cloudlog
    self.root = root
    self.params, self.mem_params = params, mem_params
    self.inputs, self.fetcher = inputs, fetcher
    self.priority = priority
    self._disk, self._drives = disk or disk_usage, drives or drives_bytes
    self._meminfo = meminfo
    self._now, self._monotonic, self._sleep = now, monotonic, sleep
    self._boot_id = boot_id
    self._clear_hook, self._legacy_cleanup = clear_hook, legacy_cleanup
    os.makedirs(sw.next_path(root), exist_ok=True)
    self.store = Store(root)
    self.selection = Selection(params)
    self.started_at = monotonic()
    self.boot_ready = False
    self.pause_until = self.started_at + START_PAUSE_S
    self.was_started: bool | None = None
    self.dev = {"started": None, "network": None, "hot": False}
    self.waiting = W_START
    self.current = None  # tile in flight
    self.progress = {"done": 0, "total": 0}
    self.retry_at, self.retry_wall, self.failures = 0.0, 0, 0
    self.server_error: int | None = None
    self.tile_retry: dict[str, tuple[float, int]] = {}  # key -> (monotonic retry time, failures)
    self.pending_deletes: list[str] = []
    self.clear_pending = False
    self.storage: dict = {}
    self.storage_at = -1e9
    self.storage_block = ""
    self.swap_result: dict = {}
    self.message = ""
    self._last_status = None
    self._live_at = -1e9
    self._crashes, self._crash_until = 0, 0.0
    self._offroad_swap = False
    self.events: list[tuple] = []  # (name, fields) also sent to cloudlog; kept for the tests

  # ---- helpers
  def event(self, name: str, **fields) -> None:
    self.events.append((name, fields))
    self.log.event(name, **fields)

  def wall(self) -> float | None:
    t = self._now()
    return t if t >= VALID_CLOCK_AFTER else None

  def rate(self):
    return CAP_BYTES_PER_S if self.dev.get("started") else None

  def selected_ids(self) -> list[str]:
    from openpilot.sunnypilot.mapd.offline_selection import id_of
    return [id_of(e) for e in self.selection.read()]

  @staticmethod
  def tiles_of(location_id: str) -> list:
    from openpilot.sunnypilot.mapd.offline_regions import entry_tiles
    return entry_tiles(location_id)

  # ---- main loop
  def run(self) -> None:
    while True:
      t0 = self._monotonic()
      self.step_safe()
      self._sleep(max(0.0, 1.0 - (self._monotonic() - t0)))

  def step_safe(self) -> None:
    if self._monotonic() < self._crash_until:
      self.write_live(force=True)
      return
    try:
      self.step()
      self._crashes = 0
    except Exception:
      self._crashes += 1
      delay = min(CRASH_BACKOFF_S * 2 ** (self._crashes - 1), CRASH_BACKOFF_MAX_S)
      self._crash_until = self._monotonic() + delay
      self.log.exception(f"map_downloader: step failed, next try in {delay:.0f} s")
      self.events.append(("map_downloader_step_failed", {"delay": delay}))

  def step(self) -> None:
    self.write_live(force=True)
    if self.priority is not None:
      self.priority.affinity()
    self.read_device()
    if not self.boot_ready:
      if not self.boot_gate():
        self.waiting = W_START
        self.write_status()
        return
    self.take_inputs()
    if self.clear_pending:
      self.do_clear()
    if self.pending_deletes:
      self.do_deletes()
    if self._offroad_swap and self.dev.get("started") is False:
      self.do_offroad_swap()
    self.store.empty_old(monotonic=self._monotonic, tick=self.write_live)
    tile = self.next_tile()
    self.write_status()
    if tile is not None:
      self.download(tile)
      self.write_status()

  def read_device(self) -> None:
    dev = self.inputs.update()
    self.dev = dev
    started = dev.get("started")
    if started is not None and self.was_started is not None and started != self.was_started:
      if started:
        self.pause_until = max(self.pause_until, self._monotonic() + START_PAUSE_S)
        self._offroad_swap = False
      else:
        self._offroad_swap = True
    if started is not None:
      self.was_started = started

  # ---- start-up order (DESIGN 3.3)
  def boot_gate(self) -> bool:
    done = sw.read_json(sw.next_path(self.root, sw.SWAP_DONE), {})
    done = done if isinstance(done, dict) else {}
    marker_ok = bool(done.get("boot_id")) and done.get("boot_id") == self._boot_id()
    if not marker_ok and self._monotonic() - self.started_at < SWAP_WAIT_S:
      return False
    try:
      with self.store.lock():
        self.store.reload()  # read at construction, before the swap at start may have changed it (review finding 1)
        if not marker_ok:  # the swap at start did not finish (killed, or never ran): its result is recorded here
          done = sw.write_swap_done(self.root, sw.MISSING, 0, "no marker from the swap at start")
        report = self.store.startup(self.wall())
    except sw.LockTimeout:
      return False
    self.swap_result = {k: done.get(k) for k in ("result", "tiles", "error", "staged_left", "failed_starts")}
    self.event("map_downloader_start", swap=self.swap_result, **report)
    self.boot_ready = True
    return True

  # ---- inputs from the screen and sunnylink
  def take_inputs(self) -> None:
    ids = parse_ids(self.params.get(REQUEST_KEY))
    if ids:
      self.params.remove(REQUEST_KEY)
    if self.params.get_bool("OsmDbUpdatesCheck"):
      self.params.put_bool("OsmDbUpdatesCheck", False, block=True)
      ids.append(ALL)
    if ids:
      self.take_requests(ids)
    dels = parse_ids(self.params.get(DELETE_KEY))
    if dels:
      self.params.remove(DELETE_KEY)
      self.take_deletes(dels)
    if self.params.get_bool(CLEAR_KEY):
      self.clear_pending = True

  def take_requests(self, ids: list[str]) -> None:
    from openpilot.sunnypilot.mapd.offline_regions import entry_id, parse_entry_id
    selected = self.selected_ids()
    wanted: list[str] = []
    for i in ids:
      if i.lower() == ALL:
        wanted += selected
      elif self.selection.add(i) is not None:
        wanted.append(entry_id(*parse_entry_id(i)))
      else:
        self.log.warning(f"map_downloader: unknown location {i!r} in {REQUEST_KEY}")
    wanted = list(dict.fromkeys(wanted))
    if not wanted:
      return
    staged = set(self.store.staged_tiles())
    tiles = [t for i in wanted for t in self.tiles_of(i) if t not in staged]
    try:
      with self.store.lock():
        added = self.store.add_requests(tiles, self._now(), "request")
    except sw.LockTimeout:
      self.params.put(REQUEST_KEY, ",".join(wanted), block=True)  # not lost: read again next second
      return
    self.retry_at, self.failures, self.server_error = 0.0, 0, None  # a request by hand asks the server at once
    for t in tiles:
      self.tile_retry.pop(sw.key_of(t), None)
    self.storage_at = -1e9
    if self._legacy_cleanup is not None:
      self._legacy_cleanup()
    self.event("map_downloader_request", locations=wanted, tiles=added)

  def take_deletes(self, ids: list[str]) -> None:
    from openpilot.sunnypilot.mapd.offline_regions import entry_id, parse_entry_id
    for i in ids:
      if i.upper() == OTHER:
        location_id = OTHER
      else:
        parsed = parse_entry_id(i)
        if parsed is None:
          self.log.warning(f"map_downloader: unknown location {i!r} in {DELETE_KEY}")
          continue
        location_id = entry_id(*parsed)
        self.selection.remove(location_id)
      if location_id not in self.pending_deletes:
        self.pending_deletes.append(location_id)

  def still_needed(self, tile) -> bool:
    keep = {t for i in self.selected_ids() for t in self.tiles_of(i)}
    return tile in keep

  # ---- deletes and DELETE ALL
  def do_deletes(self) -> None:
    keep = {t for i in self.selected_ids() for t in self.tiles_of(i)}
    gone = []
    for location_id in self.pending_deletes:
      candidates = self.store.live_tiles() + self.store.staged_tiles() if location_id == OTHER else self.tiles_of(location_id)
      gone += [t for t in candidates if t not in keep and t not in gone]
    try:
      with self.store.lock():
        report = self.store.delete_tiles(gone)
    except sw.LockTimeout:
      return
    self.event("map_downloader_delete", locations=list(self.pending_deletes), **report)
    self.pending_deletes = []
    self.storage_at = -1e9
    self.store.empty_old(budget_s=30.0, monotonic=self._monotonic, tick=self.write_live)

  def do_clear(self) -> None:
    try:
      with self.store.lock():
        self.store.clear_all()
    except sw.LockTimeout:
      return
    self.store.empty_old(budget_s=600.0, monotonic=self._monotonic, tick=self.write_live)  # heartbeat between folders
    if self._clear_hook is not None:
      self._clear_hook()  # mapd_manager.clear_downloaded_maps: upstream's params and any leftover folder
    for key in (REQUEST_KEY, DELETE_KEY, STATUS_KEY, CLEAR_KEY):
      self.params.remove(key)
    self.selection.write([])
    self.pending_deletes, self.clear_pending = [], False
    self.tile_retry, self.retry_at, self.failures, self.server_error, self.message = {}, 0.0, 0, None, ""
    self.storage_at, self._last_status = -1e9, None
    self.event("map_downloader_clear")

  def do_offroad_swap(self) -> None:
    self._offroad_swap = False
    try:
      with self.store.lock(wait=10.0):
        n = sw.swap(self.root)
        self.store.reload()
    except sw.LockTimeout:
      return
    self.swap_result = dict(self.swap_result, offroad_tiles=n, offroad_at=self.wall())
    if n:
      self.event("map_downloader_offroad_swap", tiles=n)
    self.storage_at = -1e9

  # ---- the queue (DESIGN 5.3)
  def due(self) -> list:
    """Tiles to check, in order (backoff ignored), each once."""
    selected = self.selected_ids()
    union: list = []
    for i in selected:
      union += [t for t in self.tiles_of(i) if t not in union]
    live = set(self.store.live_tiles())
    staged = set(self.store.staged_tiles())
    no_data = self.store.live["no_data"]
    out: dict = {}
    for t in union:  # 1: not on disk, not staged, not known as "no data"
      if t not in live and t not in staged and sw.key_of(t) not in no_data:
        out[t] = None
    union_set = set(union)
    for k, (_, _reason) in sorted(self.store.requests.items(), key=lambda kv: kv[1][0]):  # 2: requested, oldest first
      t = sw.parse_key(k)
      if t in union_set and t not in staged:
        out[t] = None
    now = self.wall()
    if self.params.get_bool(AUTO_KEY) and now is not None:  # 3: older than 7 days, oldest first
      aged = []
      for t in union:
        if t in staged or t in out:
          continue
        k = sw.key_of(t)
        entry = self.store.live["tiles"].get(k) if t in live else no_data.get(k)
        if entry is None:
          continue
        checked = entry.get("checked_at")
        if checked is None or now - checked > AGE_LIMIT_S:
          aged.append((checked or 0.0, t))
      for _, t in sorted(aged):
        out[t] = None
    return list(out)

  def next_tile(self):
    queue = self.due()
    if not queue:
      self.waiting = ""
      return None
    mono = self._monotonic()
    if mono < self.pause_until:
      self.waiting = W_START
      return None
    if not network_allows(self.dev.get("network")):
      self.waiting = W_NETWORK
      return None
    if self.dev.get("hot"):
      self.waiting = W_HOT
      return None
    mem = self._meminfo()
    if mem is not None and mem < MIN_MEM_AVAILABLE:
      self.waiting = W_MEMORY
      return None
    if mono < self.retry_at:
      self.waiting = W_SERVER_ERROR if self.server_error else W_SERVER
      return None
    for t in queue:
      until = self.tile_retry.get(sw.key_of(t), (0.0, 0))[0]
      if mono >= until:
        if not self.storage_ok(t):
          self.waiting = W_STORAGE
          return None
        self.waiting = ""
        return t
    self.waiting = W_SERVER
    return None

  # ---- storage guard (offline_storage.py; owner Q5 a)
  def refresh_storage(self) -> None:
    from openpilot.sunnypilot.mapd.offline_storage import budget, folder_bytes, maps_bytes, recording_mb_per_min
    if self._monotonic() - self.storage_at < STORAGE_REFRESH_S:
      return
    self.storage_at = self._monotonic()
    try:
      disk = self._disk(self.root)
      maps = maps_bytes(self.root)
      staged = folder_bytes(sw.next_path(self.root, sw.READY))[0]
      drives = self._drives()
      rate = recording_mb_per_min(self.params)
      b = budget(disk, drives, maps, maps, 0, rate)
      self.storage = {"capacity": disk.capacity, "available": disk.available, "maps": maps - staged, "staged": staged,
                      "drives": drives, "floor": b.floor, "other": b.other, "max_maps": b.max_maps, "rate_mb_min": rate,
                      "buffer_hours": round(b.buffer_hours_now, 2), "buffer_hours_full": round(b.buffer_hours_full, 2)}
    except OSError:
      self.storage = {}

  def storage_ok(self, tile) -> bool:
    from openpilot.sunnypilot.mapd.offline_storage import (DEFAULT_UNPACK_RATIO, UNKNOWN_TILE_DOWNLOAD_BYTES, budget,
                                                            recording_mb_per_min)
    self.refresh_storage()
    if not self.storage:
      return True
    k = sw.key_of(tile)
    entry = self.store.live["tiles"].get(k, {})
    download = entry.get("download_bytes") or self.store.live["sizes"].get(k) or UNKNOWN_TILE_DOWNLOAD_BYTES
    disk_size = entry.get("disk_bytes") or int(download * self.unpack_ratio(DEFAULT_UNPACK_RATIO))
    s = self.storage
    from openpilot.sunnypilot.mapd.offline_storage import Disk
    maps_now = s["maps"] + s["staged"]
    b = budget(Disk(s["capacity"], s["available"]), s["drives"], maps_now, maps_now + disk_size, download,
               recording_mb_per_min(self.params))
    if not b.ok:
      if self.storage_block != k:
        self.event("map_downloader_storage", tile=k, ok=False, maps_after=b.maps_after, max_maps=b.max_maps)
      self.storage_block = k
      self.message = f"not enough storage: maps may use {max(0, b.max_maps) / 1e9:.1f} GB in all to keep 3 h of full recording"
      return False
    if self.storage_block:
      self.storage_block, self.message = "", ""
    return True

  def unpack_ratio(self, default: float) -> float:
    disk = download = 0
    for e in self.store.live["tiles"].values():
      if e.get("disk_bytes") and e.get("download_bytes"):
        disk += e["disk_bytes"]
        download += e["download_bytes"]
    return disk / download if download else default

  # ---- one tile
  def tick(self) -> None:
    """During a transfer (twice a second): heartbeat, device state, the requests; raises Cancel to stop it."""
    self.write_live()
    self.read_device()
    if self.dev.get("started") and self._monotonic() < self.pause_until:
      raise Cancel("onroad: start pause")
    if not network_allows(self.dev.get("network")):
      raise Cancel("network")
    if self.dev.get("hot"):
      raise Cancel("hot")
    mem = self._meminfo()
    if mem is not None and mem < MIN_MEM_AVAILABLE:
      raise Cancel("memory")
    ids = parse_ids(self.params.get(REQUEST_KEY))
    if ids or self.params.get_bool("OsmDbUpdatesCheck"):
      self.take_inputs()
    dels = parse_ids(self.params.get(DELETE_KEY))
    if dels:
      self.params.remove(DELETE_KEY)
      self.take_deletes(dels)
    if self.params.get_bool(CLEAR_KEY):
      self.clear_pending = True
      raise Cancel("clear")
    if self.current is not None and self.pending_deletes and not self.still_needed(self.current):
      raise Cancel("deleted")

  def download(self, tile) -> None:
    k = sw.key_of(tile)
    live_entry = self.store.live["tiles"].get(k) if os.path.isdir(sw.tile_path(self.root, sw.OFFLINE, tile)) else None
    host = urlparse(self.fetcher.base_url).hostname or ""
    no_304 = bool(self.store.live["hosts"].get(host, {}).get("no_304"))
    self.current = tile
    self.progress = {"done": 0, "total": (live_entry or {}).get("download_bytes") or self.store.live["sizes"].get(k) or 0}
    work, part = self.store.work_dir(tile), self.store.part(tile)
    os.makedirs(os.path.dirname(part), exist_ok=True)

    def tick():
      self.progress["done"] = os.path.getsize(part) if os.path.exists(part) else 0
      self.tick()
    try:
      out = self.fetcher.check(tile, part, work, live_entry, no_304, tick, self.rate)
    except Cancel as e:
      self.current = None
      self.event("map_downloader_tile", tile=k, result="aborted", reason=str(e))
      return
    except OSError as e:
      self.current = None
      self.after_oserror(tile, e)
      return
    self.current = None
    try:
      self.apply_outcome(tile, out, live_entry, host)
    except OSError as e:
      self.after_oserror(tile, e)

  def after_oserror(self, tile, e: OSError) -> None:
    import errno
    try:
      with self.store.lock():
        self.store.discard_work(tile)
    except sw.LockTimeout:
      pass
    if e.errno == errno.ENOSPC:
      self.storage_at = -1e9
      self.message = "not enough storage"
      self.pause_until = max(self.pause_until, self._monotonic() + STORAGE_REFRESH_S)
    self.event("map_downloader_tile", tile=sw.key_of(tile), result="os_error", errno=e.errno)
    self.tile_backoff(tile)

  def server_backoff(self, code: int | None) -> None:
    self.failures += 1
    delay = min(SERVER_RETRY_S * 2 ** (self.failures - 1), SERVER_RETRY_MAX_S)
    self.retry_at, self.retry_wall = self._monotonic() + delay, round(self._now() + delay)
    self.server_error = code

  def tile_backoff(self, tile) -> None:
    k = sw.key_of(tile)
    _, n = self.tile_retry.get(k, (0.0, 0))
    self.tile_retry[k] = (self._monotonic() + min(SERVER_RETRY_S * 2 ** n, SERVER_RETRY_MAX_S), n + 1)

  def apply_outcome(self, tile, out, live_entry, host: str) -> None:
    from openpilot.sunnypilot.mapd import tile_fetch as tf
    from openpilot.sunnypilot.mapd.map_store import APPLY, IDENTICAL
    k = sw.key_of(tile)
    now = self.wall()
    result = out.kind
    if out.kind in (tf.NO_ANSWER, tf.SERVER_ERROR):
      self.server_backoff(out.code if out.kind == tf.SERVER_ERROR else None)
      self.event("map_downloader_tile", tile=k, result=result, code=out.code, reason=out.reason)
      return
    self.failures, self.server_error, self.retry_at = 0, None, 0.0
    if out.kind == tf.BAD:
      with self.store.lock():
        self.store.discard_work(tile)
      self.tile_backoff(tile)
      self.event("map_downloader_tile", tile=k, result=result, reason=out.reason)
      return
    self.tile_retry.pop(k, None)
    prepared = self.store.prepare_hashes(tile) if out.kind == tf.OK else None  # a full-tile hash, outside the lock
    with self.store.lock():
      if out.kind == tf.IGNORED_304:
        self.store.set_no_304(host)
        self.store.mark_checked(tile, now)
      elif out.kind in (tf.NOT_MODIFIED, tf.SAME_HEAD):
        self.store.mark_checked(tile, now)
      elif out.kind == tf.NO_DATA:
        self.store.mark_checked(tile, now, no_data=True)
      elif out.kind == tf.OK:
        entry = {"fetched_at": now, "checked_at": now, "etag": out.etag, "last_modified": out.last_modified,
                 "download_bytes": out.length, "disk_bytes": out.disk, "files": len(out.files)}
        decision, record = self.store.judge(tile, out.files, out.etag, now, prepared)
        if decision == IDENTICAL:
          self.store.discard_work(tile)
          self.store.live["tiles"].setdefault(k, {}).update({f: entry[f] for f in ("checked_at", "download_bytes", "disk_bytes", "files")} |
                                                            {f: entry[f] for f in ("etag", "last_modified") if entry[f]})
          self.store.record_shrink(tile, None)
          self.store.mark_checked(tile, now, out.headers)
          result = "identical"
        elif decision == APPLY:
          if self.store.live["shrink"].get(k):
            self.event("map_downloader_shrink_applied", tile=k, record=self.store.live["shrink"][k], files=len(out.files), bytes=out.disk)
          self.store.record_shrink(tile, None)
          self.store.commit(tile, entry, out.files)
          result = "staged"
        else:  # shrink guard: the live tile stays, and so do its validators (the next check fetches the archive again)
          self.store.discard_work(tile)
          self.store.record_shrink(tile, record)
          self.store.mark_checked(tile, now)
          result = "kept_smaller"
    if out.kind == tf.NO_DATA and live_entry is not None:
      self.log.warning(f"map_downloader: server has no file for live tile {k}; kept")
    self.storage_at = -1e9
    self.event("map_downloader_tile", tile=k, result=result, code=out.code, bytes=out.length)

  # ---- status
  def write_live(self, force: bool = False) -> None:
    mono = self._monotonic()
    if not force and mono - self._live_at < LIVE_WRITE_S:
      return
    self._live_at = mono
    live = {"heartbeat": mono, "tile": sw.key_of(self.current) if self.current else None, "done": self.progress["done"],
            "total": self.progress["total"], "cap": self.rate(), "storage": self.storage}
    try:
      self.mem_params.put(LIVE_KEY, live)
    except Exception:
      pass

  def status(self) -> dict:
    from openpilot.sunnypilot.mapd.offline_selection import id_of
    self.refresh_storage()
    selection = self.selection.read()
    ids = [id_of(e) for e in selection]
    live = set(self.store.live_tiles())
    staged = set(self.store.staged_tiles())
    due = set(self.due()) if self.boot_ready else set()
    no_data = self.store.live["no_data"]
    tiles_meta = self.store.live["tiles"]
    next_meta = self.store.next["tiles"]
    sizes = {}
    entries = {}
    union_others = {}
    for i in ids:
      union_others[i] = {t for j in ids if j != i for t in self.tiles_of(j)}
    for e in selection:
      i = id_of(e)
      tiles = self.tiles_of(i)
      present = [t for t in tiles if t in live]
      st = [t for t in tiles if t in staged]
      expected = [t for t in tiles if sw.key_of(t) not in no_data or t in live]
      d = [t for t in tiles if t in due]
      for t in present:
        sizes.setdefault(t, (tiles_meta.get(sw.key_of(t)) or {}).get("disk_bytes") or self.tile_bytes(t))
      dates = [self.tile_date(t) for t in present]
      dates = [x for x in dates if x]
      exclusive = [t for t in present if t not in union_others[i]]
      kept = []
      for t in tiles:
        rec = self.store.live["shrink"].get(sw.key_of(t))
        if rec and rec.get("live_bytes"):
          kept.append({"tile": sw.key_of(t), "pct": round(100 * (1 - rec["bytes"] / rec["live_bytes"]))})
      if self.current is not None and self.current in tiles:
        state = "downloading"
      elif d:
        state = "queued"
      elif not present and not st and expected:
        state = "not_downloaded"
      elif st:
        state = "staged"
      else:
        state = "ok"
      entries[i] = {
        "kind": e["kind"], "code": e["code"], "title": e["title"], "state": state,
        "tiles_total": len(tiles), "tiles_expected": len(expected), "tiles_present": len(present), "tiles_staged": len(st),
        "tiles_due": len(d), "tiles_done": max(0, len(expected) - len(d)),
        "bytes": sum(sizes[t] for t in present), "exclusive_bytes": sum(sizes[t] for t in exclusive),
        "shared_tiles": len(present) - len(exclusive),
        "shared_with": [x["title"] for x in selection if id_of(x) != i and set(self.tiles_of(id_of(x))) & set(present)],
        "date_oldest": round(min(dates)) if dates else 0, "date_newest": round(max(dates)) if dates else 0,
        "staged": len(st), "due": len(d), "kept_smaller": kept,
        "staged_dates": sorted({round((next_meta.get(sw.key_of(t)) or {}).get("fetched_at") or 0) for t in st}),
      }
    union = {t for i in ids for t in self.tiles_of(i)}
    orphans = [t for t in live if t not in union]
    current_location = None
    if self.current is not None:
      current_location = next((i for i in ids if self.current in self.tiles_of(i)), None)
    run = {
      "active": bool(due), "current": sw.key_of(self.current) if self.current else None, "current_location": current_location,
      "waiting": self.waiting if due else "", "server_error": self.server_error if due else None,
      "retry_at": self.retry_wall if due and self.waiting in (W_SERVER, W_SERVER_ERROR) and self.retry_at else 0,
      "queue_tiles": len(due), "staged": len(staged), "cap": CAP_BYTES_PER_S, "deletes_pending": list(self.pending_deletes),
      "swap": self.swap_result, "apply_failed": int(self.swap_result.get("failed_starts") or 0) >= 2 and bool(staged),
      "auto": self.params.get_bool(AUTO_KEY),
    }
    return {"version": STATUS_VERSION, "entries": entries, "order": ids, "run": run,
            "other": {"tiles": len(orphans), "bytes": sum(self.tile_bytes(t) for t in orphans)},
            "message": self.message}  # storage figures change while driving: memory param only (write_live)

  def tile_bytes(self, tile) -> int:
    meta = self.store.live["tiles"].get(sw.key_of(tile)) or {}
    if meta.get("disk_bytes"):
      return meta["disk_bytes"]
    total = 0
    try:
      for e in os.scandir(sw.tile_path(self.root, sw.OFFLINE, tile)):
        if e.is_file(follow_symlinks=False):
          total += e.stat().st_size
    except OSError:
      pass
    return total

  def tile_date(self, tile) -> float:
    meta = self.store.live["tiles"].get(sw.key_of(tile)) or {}
    lm = meta.get("last_modified")
    if lm:
      from email.utils import parsedate_to_datetime
      try:
        return parsedate_to_datetime(lm).timestamp()
      except (TypeError, ValueError):
        pass
    return meta.get("fetched_at") or 0.0

  def write_status(self) -> None:
    status = self.status()
    self.write_live(force=True)  # the storage figures just measured (memory param)
    if status != self._last_status:
      self.params.put(STATUS_KEY, status, block=True)
      self._last_status = status


def main() -> None:
  priority = Priority()
  applied = priority.start()  # before the heavy imports
  from openpilot.common.hardware.hw import Paths
  from openpilot.common.params import Params
  from openpilot.common.swaglog import cloudlog
  from openpilot.sunnypilot.mapd import tile_fetch

  def clear_hook():
    from openpilot.sunnypilot.mapd.mapd_manager import clear_downloaded_maps
    clear_downloaded_maps()

  def legacy_cleanup():
    from openpilot.sunnypilot.mapd.mapd_manager import cleanup_old_osm_data, get_files_for_cleanup
    cleanup_old_osm_data(get_files_for_cleanup())

  cloudlog.event("map_downloader_priority", **applied)
  root = Paths.mapd_root()
  os.makedirs(root, exist_ok=True)
  downloader = MapDownloader(root, Params(), Params("/dev/shm/params"), DeviceInputs(), tile_fetch.Fetcher(), priority=priority,
                             clear_hook=clear_hook, legacy_cleanup=legacy_cleanup)
  downloader.run()


if __name__ == "__main__":
  main()
