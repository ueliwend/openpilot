"""Fixtures for the swap and power-cut tests: a mapd root with live tiles, staged tiles and E9 leftovers built on disk."""
import hashlib
import json
import os

from openpilot.sunnypilot.mapd import offline_swap as sw


def content(tile, version: str, n: int = 3) -> dict[str, bytes]:
  lat, lon = tile
  return {f"{lat + i * 0.25:f}_{lon:f}_{lat + (i + 1) * 0.25:f}_{lon + 0.25:f}": hashlib.sha256(f"{tile}{version}{i}".encode()).digest() * (40 + i)
          for i in range(n + (1 if version == "new" else 0))}


def hashes(files: dict[str, bytes]) -> dict[str, list]:
  return {k: [len(v), hashlib.sha256(v).hexdigest()] for k, v in files.items()}


def write_tile(folder: str, files: dict[str, bytes]) -> None:
  os.makedirs(folder, exist_ok=True)
  for name, data in files.items():
    with open(os.path.join(folder, name), "wb") as f:
      f.write(data)


def read_tile(folder: str) -> dict[str, bytes] | None:
  if not os.path.isdir(folder):
    return None
  return {e.name: open(e.path, "rb").read() for e in os.scandir(folder) if e.is_file()}


def stage(root: str, tile, files: dict[str, bytes], fetched_at: float = 1.8e9) -> None:
  """A ready tile as map_store.commit leaves it."""
  write_tile(sw.tile_path(root, sw.READY_SUB, tile), files)
  nxt = sw.load_manifest(sw.next_manifest_path(root))
  nxt["tiles"][sw.key_of(tile)] = {"fetched_at": fetched_at, "checked_at": fetched_at, "etag": f'"{tile}"', "state": "ready",
                                   "disk_bytes": sum(len(v) for v in files.values()), "files": len(files)}
  sw.write_json_atomic(sw.next_manifest_path(root), nxt)
  sw.write_json_atomic(sw.staged_hashes_path(root, tile), {"files": hashes(files)})


def live(root: str, tile, files: dict[str, bytes], entry: dict | None = None) -> None:
  write_tile(sw.tile_path(root, sw.OFFLINE, tile), files)
  m = sw.load_manifest(sw.live_manifest_path(root))
  m["tiles"][sw.key_of(tile)] = entry or {"fetched_at": 1.7e9, "checked_at": 1.7e9, "etag": '"old"'}
  sw.write_json_atomic(sw.live_manifest_path(root), m)
  sw.write_json_atomic(sw.live_hashes_path(root, tile), {"files": hashes(files)})


def e9_leftovers(root: str, tiles_prev: dict, half_written: dict) -> None:
  for t, files in tiles_prev.items():
    write_tile(sw.tile_path(root, sw.PREV, t), files)
  with open(os.path.join(root, sw.PREV, "journal.json"), "w") as f:
    json.dump({"version": 1, "garbage": True}, f)
  for t, files in half_written.items():
    write_tile(sw.tile_path(root, sw.OFFLINE, t), files)
  os.makedirs(os.path.join(root, sw.TMP, "offline", "46"), exist_ok=True)
  with open(os.path.join(root, sw.TMP, "offline", "46", "8.tar.gz"), "wb") as f:
    f.write(b"\x1f\x8bpartial")
