"""Tile geometry of pfeiferj's mapd v1.12.0 offline maps (Inster port).

The binary stores offline map data in 2-degree tiles, one folder per tile: `<mapd root>/offline/<lat>/<lon>/`.
A download of a location (nation or US state) fetches every tile of the location's bounding box, row by row:
latitude rising, then longitude rising (download.go:199-200 at v1.12.0). The bounding boxes are vendored from the
same version (they are embedded in the binary), so this module knows which tiles a request touches and in which order.
"""
import json
import math
import os
from functools import cache

GRID_DEGREES = 2  # GROUP_AREA_BOX_DEGREES, generate_offline.go:48
BOXES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bounding_boxes")
BOX_FILES = {"nation": "nation_bounding_boxes.json", "state": "us_states_bounding_boxes.json"}
BOXES_VERSION = "v1.12.0"  # the mapd version the vendored boxes come from

Tile = tuple[int, int]


@cache
def bounding_boxes(kind: str) -> dict:
  with open(os.path.join(BOXES_DIR, BOX_FILES[kind])) as f:
    return json.load(f)


def adjusted_bounds(box: dict) -> tuple[int, int, int, int]:
  """Exact copy of adjustedBounds (download.go:177-190): the box clipped outwards to the tile grid."""
  min_lat = int(math.floor(box["min_lat"] / GRID_DEGREES)) * GRID_DEGREES
  min_lon = int(math.floor(box["min_lon"] / GRID_DEGREES)) * GRID_DEGREES
  max_lat = int(math.floor(box["max_lat"] / GRID_DEGREES)) * GRID_DEGREES
  max_lon = int(math.floor(box["max_lon"] / GRID_DEGREES)) * GRID_DEGREES
  if box["max_lat"] > max_lat:
    max_lat += GRID_DEGREES
  if box["max_lon"] > max_lon:
    max_lon += GRID_DEGREES
  return min_lat, min_lon, max_lat, max_lon


def tiles_for_box(box: dict) -> list[Tile]:
  """The tiles of a bounding box in the binary's download order (DownloadBounds, download.go:199-200)."""
  min_lat, min_lon, max_lat, max_lon = adjusted_bounds(box)
  return [(lat, lon) for lat in range(min_lat, max_lat, GRID_DEGREES) for lon in range(min_lon, max_lon, GRID_DEGREES)]


def tile_order(kind: str, code: str) -> list[Tile] | None:
  """Tiles the binary downloads for one location, in order; None if the binary does not know the code."""
  location = bounding_boxes(kind).get(code)
  if location is None:
    return None
  return tiles_for_box(location["bounding_box"])


def tile_key(tile: Tile) -> str:
  return f"{tile[0]}/{tile[1]}"


def tile_dir(root: str, sub: str, tile: Tile) -> str:
  """`<root>/<sub>/<lat>/<lon>`: the tile folder (sub = "offline") or its moved-aside copy."""
  return os.path.join(root, sub, str(tile[0]), str(tile[1]))


def tile_archive(root: str, tile: Tile) -> str:
  """The binary's download file for a tile: `<root>/tmp/offline/<lat>/<lon>.tar.gz` (download.go:201-203)."""
  return os.path.join(root, "tmp", "offline", str(tile[0]), f"{tile[1]}.tar.gz")
