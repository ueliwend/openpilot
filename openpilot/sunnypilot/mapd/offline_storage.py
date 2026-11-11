"""Storage guard for offline maps (Inster port): maps may never take the space of the minimum drive buffer.

Maps live on the same filesystem as the drive recordings (Paths.mapd_root, Paths.log_root). Once drives have filled the
disk, free space sits just above the deleter's floor, so the guard does not compare with today's free space. It checks
the budget against the total capacity, in bytes measured on the device:

    maps_after + peak + floor + other + minimum drive buffer <= capacity

- floor: the deleter's own limit, max(10 % of capacity, 5 GiB) (system/loggerd/deleter.py);
- other: capacity - available - drives - maps (system, reserved blocks, anything else);
- maps_after: maps now + the estimated size of the tiles not on disk yet (a tile on disk is replaced at its own size);
- peak: extra space while a tile is replaced: 2 x the largest tile + its download file;
- minimum drive buffer: 3 h of driving (owner, 11 Nov 2026) at 239.18 MB per driving minute (H3, EST).

When the download needs more than the free space above the floor (the shortfall D), the deleter frees it by deleting the
oldest drive segments while the download runs (owner: always allowed). replay_deleter() replays the deleter's order on
the measured folder sizes, so the confirmation can say how many minutes go and when they were last written (file
modification times; segment folder names carry no date). With an external drive mounted, the deleter moves segments
there instead, and only what the external drive deletes is lost.
Design: inster-dev results/2026-11-11_offline_maps_multi_country/DESIGN.md (rev 4) section 3; owner's answers 3 and 4.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from openpilot.common.hardware.hw import Paths
from openpilot.system.loggerd import deleter
from openpilot.system.loggerd.uploader import listdir_by_creation

BYTES_PER_DRIVE_MINUTE = 239.18e6  # all cameras, H3 storage budget (EST)
MIN_DRIVE_BUFFER_HOURS = 3.0  # owner, 11 Nov 2026 (DESIGN question 4)
MIN_DRIVE_BUFFER_BYTES = int(MIN_DRIVE_BUFFER_HOURS * 60 * BYTES_PER_DRIVE_MINUTE)
DEFAULT_UNPACK_RATIO = 4.0  # on-disk bytes per downloaded byte until measured (DESIGN section 3)
UNKNOWN_TILE_DOWNLOAD_BYTES = 150_000_000  # a tile whose size the server did not tell (EST, high side)


@dataclass
class Disk:
  capacity: int
  available: int


def disk_usage(path: str) -> Disk:
  st = os.statvfs(path)
  return Disk(capacity=st.f_blocks * st.f_frsize, available=st.f_bavail * st.f_frsize)


def floor_bytes(capacity: int) -> int:
  """The free space below which the deleter starts deleting (deleter.py: MIN_PERCENT, MIN_BYTES)."""
  return int(max(capacity * deleter.MIN_PERCENT / 100, deleter.MIN_BYTES))


def folder_bytes(path: str) -> tuple[int, float]:
  """(bytes, newest file mtime) of everything below a folder."""
  size, newest = 0, 0.0
  stack = [path]
  while stack:
    try:
      entries = list(os.scandir(stack.pop()))
    except (FileNotFoundError, NotADirectoryError, PermissionError):
      continue
    for e in entries:
      try:
        if e.is_dir(follow_symlinks=False):
          stack.append(e.path)
        elif e.is_file(follow_symlinks=False):
          st = e.stat(follow_symlinks=False)
          size += st.st_size
          newest = max(newest, st.st_mtime)
      except FileNotFoundError:
        continue
  return size, newest


def maps_bytes(root: str) -> int:
  """Everything mapd and the offline update keep under the mapd root (tiles, old copies, download files, caches)."""
  return folder_bytes(root)[0]


@dataclass
class Budget:
  ok: bool
  capacity: int
  available: int
  floor: int
  other: int
  drives: int
  maps_now: int
  maps_after: int
  peak: int
  shortfall: int  # D: bytes the deleter must free while the download runs (0 = none)
  max_maps: int  # most map data the buffer allows
  buffer_hours_after: float  # drive recording the disk can hold once the maps are in


def budget(disk: Disk, drives: int, maps_now: int, maps_after: int, peak: int) -> Budget:
  floor = floor_bytes(disk.capacity)
  other = max(0, disk.capacity - disk.available - drives - maps_now)
  ok = maps_after + peak + floor + other + MIN_DRIVE_BUFFER_BYTES <= disk.capacity
  shortfall = max(0, (maps_after + peak - maps_now) - (disk.available - floor))
  max_maps = disk.capacity - floor - other - MIN_DRIVE_BUFFER_BYTES
  buffer_hours = max(0.0, (disk.capacity - floor - other - maps_after) / BYTES_PER_DRIVE_MINUTE / 60)
  return Budget(ok, disk.capacity, disk.available, floor, other, drives, maps_now, maps_after, peak, shortfall, max_maps,
                buffer_hours)


def is_segment(name: str) -> bool:
  prefix, _, number = name.rpartition("--")
  return bool(prefix) and number.isdigit()


@dataclass
class DeleterReplay:
  needed: int
  freed: int = 0
  minutes: int = 0  # drive segments (1 min each) that leave the internal storage
  folders: int = 0  # all folders, incl. boot / crash logs
  first_written: float = 0.0  # newest file mtime of the oldest / newest folder that goes
  last_written: float = 0.0
  moved: bool = False  # external drive mounted: segments are moved there, not deleted
  lost_minutes: int = 0  # segments deleted for good (on the external drive when moved)
  lost_first_written: float = 0.0
  lost_last_written: float = 0.0
  enough: bool = True  # False: all deletable drive data would not cover the shortfall
  folder_names: list[str] = field(default_factory=list)


def _dates(replay_first: float, replay_last: float, newest: float) -> tuple[float, float]:
  if newest <= 0:
    return replay_first, replay_last
  return (min(replay_first, newest) if replay_first else newest), max(replay_last, newest)


def replay_deleter(needed: int, log_root: str | None = None, external_root: str | None = None) -> DeleterReplay:
  """Which drive folders the deleter removes to free `needed` bytes, in its own order (deleter.deleter_thread):
  folders by creation (route counter), boot / crash folders last, the preserved segments last (the 5 most recent
  bookmarked segments and the two before each), folders holding a `.lock` file skipped. With the external drive mounted,
  each folder is moved there; when the external drive is short of space it first deletes its oldest folder (by name)."""
  log_root = log_root or Paths.log_root()
  external_root = external_root if external_root is not None else Paths.log_root_external()
  replay = DeleterReplay(needed=max(0, needed))
  if replay.needed <= 0:
    return replay
  dirs = listdir_by_creation(log_root)
  preserved = deleter.get_preserved_segments(dirs)
  external = bool(external_root) and Path(external_root).is_mount()
  if external:
    replay.moved = True
    ext_disk = disk_usage(external_root)
    ext_floor = floor_bytes(ext_disk.capacity)
    ext_available = ext_disk.available
    ext_dirs = sorted(listdir_by_creation(external_root))
  for name in sorted(dirs, key=lambda d: (d in deleter.DELETE_LAST, d in preserved)):
    if replay.freed >= replay.needed:
      break
    path = os.path.join(log_root, name)
    try:
      if any(n.endswith(".lock") for n in os.listdir(path)):
        continue
    except OSError:
      continue
    size, newest = folder_bytes(path)
    if external:
      if ext_available < ext_floor and ext_dirs:
        gone = ext_dirs.pop(0)
        gone_size, gone_newest = folder_bytes(os.path.join(external_root, gone))
        ext_available += gone_size
        if is_segment(gone):
          replay.lost_minutes += 1
          replay.lost_first_written, replay.lost_last_written = _dates(replay.lost_first_written, replay.lost_last_written,
                                                                       gone_newest)
      ext_available -= size
      ext_dirs.append(name)
    elif is_segment(name):
      replay.lost_minutes += 1
      replay.lost_first_written, replay.lost_last_written = _dates(replay.lost_first_written, replay.lost_last_written, newest)
    replay.freed += size
    replay.folders += 1
    replay.folder_names.append(name)
    if is_segment(name):
      replay.minutes += 1
      replay.first_written, replay.last_written = _dates(replay.first_written, replay.last_written, newest)
  replay.enough = replay.freed >= replay.needed
  return replay


@dataclass
class TileEstimate:
  on_disk: int = 0  # bytes of the tiles already on disk (replaced at their own size)
  new: int = 0  # estimated on-disk bytes of the tiles not on disk yet
  largest: int = 0  # largest tile (on disk or estimated)
  largest_download: int = 0  # its download file
  download: int = 0  # bytes to download
  unknown: int = 0  # tiles whose size the server did not tell (counted at UNKNOWN_TILE_DOWNLOAD_BYTES)
  no_data: int = 0  # tiles the server has no file for (404)


def estimate_tiles(on_disk: dict, server: dict, ratio: float) -> TileEstimate:
  """on_disk: {tile: bytes} for the tiles present; server: {tile: download bytes | None (unknown) | 0 (no data)} for the
  others. The download of a tile on disk is its size / ratio."""
  est = TileEstimate()
  for size in on_disk.values():
    est.on_disk += size
    download = int(size / ratio)
    est.download += download
    if size > est.largest:
      est.largest, est.largest_download = size, download
  for download in server.values():
    if download == 0:
      est.no_data += 1
      continue
    if download is None:
      est.unknown += 1
      download = UNKNOWN_TILE_DOWNLOAD_BYTES
    size = int(download * ratio)
    est.new += size
    est.download += download
    if size > est.largest:
      est.largest, est.largest_download = size, download
  return est


def peak_bytes(est: TileEstimate) -> int:
  return 2 * est.largest + est.largest_download


def drives_bytes(log_root: str | None = None) -> int:
  return folder_bytes(log_root or Paths.log_root())[0]


@dataclass
class Plan:
  budget: Budget
  estimate: TileEstimate
  replay: DeleterReplay


def plan_download(root: str, est: TileEstimate, disk: Disk | None = None, drives: int | None = None,
                  log_root: str | None = None) -> Plan:
  """The guard and the deleter replay for downloading the tiles of `est` now (disk and drives measured if not given)."""
  disk = disk or disk_usage(root if os.path.isdir(root) else os.path.dirname(root))
  drives = drives_bytes(log_root) if drives is None else drives
  maps_now = maps_bytes(root)
  b = budget(disk, drives, maps_now, maps_now + est.new, peak_bytes(est))
  return Plan(b, est, replay_deleter(b.shortfall, log_root) if b.ok else DeleterReplay(needed=b.shortfall))
