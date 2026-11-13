"""The map tile server for the screen's size estimates (Inster port): HEAD sizes and what the store already knows.

- head(tile): the download size from a HEAD request (Content-Length), 0 for a 404 (the server makes no file for an area
  without roads: generate_offline.go:208, mapd.go:459), NO_ANSWER_SIZE when the server cannot be reached, None when the
  size is unknown (other status, no length).
- known_sizes(root): download sizes and on-disk sizes the store keeps (offline_manifest.json: live tiles, "no data" tiles,
  E9's merged size cache), and the unpack ratio from them.
Downloads and their checks are map_downloader's (tile_fetch.py); E9's probe() and check() are gone with E11.
"""
import os

from openpilot.sunnypilot.mapd import offline_swap as sw
from openpilot.sunnypilot.mapd.offline_regions import Tile, tile_url
from openpilot.sunnypilot.mapd.offline_storage import DEFAULT_UNPACK_RATIO

TIMEOUT = (5, 10)  # connect, read: every request ends within seconds
NO_ANSWER_SIZE = -1


class TileServer:
  """HTTP access to the tile host (production)."""

  def head(self, tile: Tile) -> int | None:
    import requests
    try:
      r = requests.head(tile_url(tile), timeout=TIMEOUT, allow_redirects=True, headers={"Accept-Encoding": "identity"})
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


class KnownSizes:
  """What the store knows about tile sizes (read only; map_downloader is the writer)."""

  def __init__(self, root: str):
    m = sw.load_manifest(sw.live_manifest_path(root))
    legacy = sw.read_json(os.path.join(root, sw.SIZES_NAME), {})  # before map_downloader merged it
    legacy = legacy.get("tiles", {}) if isinstance(legacy, dict) and isinstance(legacy.get("tiles"), dict) else {}
    self.tiles = m["tiles"]
    self.downloads = {k: v.get("download") for k, v in legacy.items() if isinstance(v, dict) and isinstance(v.get("download"), int)}
    self.downloads.update({k: v for k, v in m["sizes"].items() if isinstance(v, int)})
    self.downloads.update(dict.fromkeys(m["no_data"], 0))
    self.downloads.update({k: v["download_bytes"] for k, v in m["tiles"].items() if isinstance(v.get("download_bytes"), int)})

  def download(self, tile: Tile) -> int | None:
    """Download bytes; 0 = no data on the server; None = not known."""
    return self.downloads.get(sw.key_of(tile))

  def no_data(self, tile: Tile) -> bool:
    return self.download(tile) == 0

  def ratio(self) -> float:
    disk = download = 0
    for v in self.tiles.values():
      if v.get("download_bytes") and v.get("disk_bytes"):
        disk += v["disk_bytes"]
        download += v["download_bytes"]
    return disk / download if download else DEFAULT_UNPACK_RATIO
