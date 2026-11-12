"""The map tile server, seen from the offline update (Inster port): sizes, "no data" tiles and the gzip probe.

- head(tile): the download size from a HEAD request (Content-Length), 0 for a 404 (the server makes no file for an area
  without roads: generate_offline.go:208, mapd.go:459), NO_ANSWER_SIZE when the server cannot be reached, None when the
  size is unknown (other status, no length).
- probe(tile): (answer, HTTP status) for the first two bytes of the tile (`Range: bytes=0-1`; if the server ignores the
  range, 2 bytes are read and the connection closed). mapd v1.12.0 panics and loops on a reply that is not gzip
  (download.go:211-218), so a location is sent only if its first tile with data starts with the gzip magic 1f 8b. The
  Content-Type is not used. A status other than 200 / 206 / 404 (e.g. 403) is a server error, kept with its code.
- check(): HEAD sizes and the probe for one location, run off the manager loop (offline_maps.py).
- SizeCache: `<mapd root>/offline_sizes.json`, download and on-disk size per tile, and 0 for "no data". It feeds the
  status ("expected" tiles leave out the no-data ones), the size estimates and the unpack ratio. A tile found on disk
  is never "no data", and every run asks the server again for the tiles it does not have.
"""
import json
import os

from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.mapd.offline_regions import Tile, parse_tile_key, tile_key, tile_url
from openpilot.sunnypilot.mapd.offline_storage import DEFAULT_UNPACK_RATIO

SIZES_NAME = "offline_sizes.json"  # must not start with "v" or be "db" / "tmp" (mapd_manager.get_files_for_cleanup)
GZIP_MAGIC = b"\x1f\x8b"
TIMEOUT = (5, 10)  # connect, read: every request ends within seconds (offline_maps.py also caps the whole check)

GZIP, NOT_GZIP, NO_DATA, NO_ANSWER, SERVER_ERROR = "gzip", "not_gzip", "no_data", "no_answer", "server_error"
NO_ANSWER_SIZE = -1


class TileServer:
  """HTTP access to the tile host (production)."""

  def head(self, tile: Tile) -> int | None:
    import requests
    try:
      r = requests.head(tile_url(tile), timeout=TIMEOUT, allow_redirects=True)
    except Exception:
      return NO_ANSWER_SIZE
    if r.status_code == 404:
      return 0
    if r.status_code != 200:
      return None
    try:
      return int(r.headers.get("Content-Length", "")) or None
    except ValueError:
      return None

  def probe(self, tile: Tile) -> tuple[str, int | None]:
    import requests
    try:
      with requests.get(tile_url(tile), headers={"Range": "bytes=0-1"}, timeout=TIMEOUT, stream=True) as r:
        if r.status_code == 404:
          return NO_DATA, 404
        if r.status_code not in (200, 206):
          return SERVER_ERROR, r.status_code
        head = r.raw.read(2)
    except Exception:
      return NO_ANSWER, None
    return (GZIP if head == GZIP_MAGIC else NOT_GZIP), r.status_code


def check(server, tiles: list[Tile], missing: list[Tile], no_data: set[Tile], progress=None, on_size=None) -> dict:
  """HEAD sizes of `missing` (tiles not on disk; stops at the first one without an answer), then the probe of the first
  tile of `tiles` in the binary's order not known as "no data" (a 404 moves on to the next tile). Touches no file: the
  caller applies the result. progress(done, total) is called before each HEAD request, on_size(tile, size) after each one
  that got an answer (so a caller can keep the sizes of a check that does not finish)."""
  sizes: dict = {}
  for i, t in enumerate(missing):
    if progress is not None:
      progress(i, len(missing))
    size = server.head(t)
    if size == NO_ANSWER_SIZE:
      return {"sizes": sizes, "answer": NO_ANSWER, "code": None, "no_data": []}
    sizes[t] = size
    if on_size is not None:
      on_size(t, size)
  known = set(no_data) | {t for t, s in sizes.items() if s == 0}
  probe_no_data = []
  for t in tiles:
    if t in known:
      continue
    answer, code = server.probe(t)
    if answer == NO_DATA:
      probe_no_data.append(t)
      continue
    return {"sizes": sizes, "answer": answer, "code": code, "no_data": probe_no_data}
  return {"sizes": sizes, "answer": GZIP, "code": None, "no_data": probe_no_data}  # no tile with data: nothing to download


class SizeCache:
  def __init__(self, root: str):
    self.path = os.path.join(root, SIZES_NAME)
    self.tiles: dict[str, dict] = {}
    self.load()

  def load(self) -> None:
    try:
      with open(self.path) as f:
        data = json.load(f)
      tiles = data["tiles"]
      self.tiles = {k: {f: int(v[f]) for f in ("download", "disk") if isinstance(v.get(f), int)}
                    for k, v in tiles.items() if parse_tile_key(k) is not None and isinstance(v, dict)}
    except FileNotFoundError:
      self.tiles = {}
    except (ValueError, KeyError, TypeError, AttributeError):
      cloudlog.warning("mapd offline update: size cache unreadable, starting a new one")
      self.tiles = {}

  def save(self) -> None:
    os.makedirs(os.path.dirname(self.path), exist_ok=True)
    tmp = self.path + ".tmp"
    with open(tmp, "w") as f:
      json.dump({"version": 1, "tiles": self.tiles}, f)
    os.replace(tmp, self.path)

  def download(self, tile: Tile) -> int | None:
    """Download bytes; 0 = no data on the server; None = not known."""
    return self.tiles.get(tile_key(tile), {}).get("download")

  def no_data(self, tile: Tile) -> bool:
    return self.download(tile) == 0

  def set_download(self, tile: Tile, size: int | None) -> None:
    if size is None or size < 0:
      return
    entry = self.tiles.setdefault(tile_key(tile), {})
    entry["download"] = size
    if size == 0:
      entry.pop("disk", None)

  def set_disk(self, tile: Tile, size: int) -> None:
    entry = self.tiles.setdefault(tile_key(tile), {})
    entry["disk"] = size
    if entry.get("download") == 0:  # a tile on disk has data: the earlier 404 is out of date
      entry.pop("download")

  def ratio(self) -> float:
    disk = download = 0
    for v in self.tiles.values():
      if v.get("download") and v.get("disk"):
        disk += v["disk"]
        download += v["download"]
    return disk / download if download else DEFAULT_UNPACK_RATIO

