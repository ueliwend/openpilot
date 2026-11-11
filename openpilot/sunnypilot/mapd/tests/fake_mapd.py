"""Test-only fake of pfeiferj's mapd v1.12.0 download behaviour (source tag 46cd71a; DESIGN.md section 7).

Reproduces what the offline update relies on, as a generator that yields after every observable step, so a test can
poll the manager or stop the process (a crash or power cut) between any two steps:
- start-up: the request and the progress are emptied (ResetParams, mapd.go:466, params.go:86-88);
- loop: 1 s sleep, then the download check (mapd.go loop); the request is emptied at the end of every check, by temp
  file + rename (download.go:156, params.go:147-208): the heartbeat;
- per location, tiles row by row, latitude then longitude (download.go:199-200);
- per tile: the download file is created (truncated) before the GET (download.go:43, 206); after a failed GET (no network,
  404, connection cut) it stays and the next tile follows (207-209); a body that is not gzip panics (211-218): the loop
  recovers (mapd.go:211-224) with the request and tmp/ still in place, so the next loop starts the location again
  from its first tile with progress reset (111-114);
- extraction: files opened WITHOUT truncation (download.go:240: the flaw), a broken tar stops the extraction and the tile
  still counts as done (216-259); the download file is removed after the extraction (256); progress is written only
  after that (259-270); tmp/ is removed at the end of the location (273).
"""
import io
import json
import math
import os
import shutil
import tarfile
import zlib

# The fake keeps its own copy of the binary's geometry, independent of the code under test (offline_regions.py).
BOXES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bounding_boxes")  # v1.12.0's JSON

def bounding_boxes(kind: str) -> dict:
  name = "nation_bounding_boxes.json" if kind == "nation" else "us_states_bounding_boxes.json"
  with open(os.path.join(BOXES_DIR, name)) as f:
    return json.load(f)


def tiles_for_box(b: dict) -> list:
  """adjustedBounds (download.go:177-190) and the tile loop of DownloadBounds (199-200)."""
  lat0, lon0 = math.floor(b["min_lat"] / 2) * 2, math.floor(b["min_lon"] / 2) * 2
  lat1, lon1 = math.floor(b["max_lat"] / 2) * 2, math.floor(b["max_lon"] / 2) * 2
  lat1 += 2 if b["max_lat"] > lat1 else 0
  lon1 += 2 if b["max_lon"] > lon1 else 0
  return [(lat, lon) for lat in range(lat0, lat1, 2) for lon in range(lon0, lon1, 2)]


def tile_archive(root: str, tile) -> str:
  return os.path.join(root, "tmp", "offline", str(tile[0]), f"{tile[1]}.tar.gz")  # download.go:201-203


NET_ERROR = "net_error"  # http.Get fails: no network (download.go:50-52)
NOT_FOUND = "not_found"  # 404: no data on the server for this tile (download.go:57-58)


class Cut:
  """The connection breaks after n bytes of the body (io.Copy error, download.go:62-64)."""
  def __init__(self, body: bytes, n: int):
    self.body, self.n = body, n


class MapdPanic(Exception):
  pass


def make_tile(tile, files: dict[str, bytes]) -> bytes:
  """A server tile like scripts/compress_offline.sh makes it: tar -czf of offline/<lat>/<lon>/."""
  return gzip_bytes(make_tar(tile, files))


def make_tar(tile, files: dict[str, bytes]) -> bytes:
  buf = io.BytesIO()
  with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tar:
    folder = f"offline/{tile[0]}/{tile[1]}"
    info = tarfile.TarInfo(folder)
    info.type, info.mode, info.mtime = tarfile.DIRTYPE, 0o755, 0
    tar.addfile(info)
    for name, data in files.items():
      info = tarfile.TarInfo(f"{folder}/{name}")
      info.size, info.mode, info.mtime = len(data), 0o644, 0
      tar.addfile(info, io.BytesIO(data))
  return buf.getvalue()


def gzip_bytes(data: bytes) -> bytes:
  c = zlib.compressobj(6, zlib.DEFLATED, 31)
  return c.compress(data) + c.flush()


def write_param(path: str, data: bytes) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = f"{path}.tmp_fake"
  with open(tmp, "wb") as f:
    f.write(data)
  os.rename(tmp, path)


class FakeMapd:
  def __init__(self, root: str, request_path: str, progress_path: str, server: dict, clock, chunk: int = 512):
    self.root, self.request_path, self.progress_path = root, request_path, progress_path
    self.server, self.clock, self.chunk = server, clock, chunk
    self.gets: list = []  # tiles requested from the server, in order
    self.write_fails: dict = {}  # tile -> indexes of its files whose writes fail (disk full): created, left empty

  def run(self):
    write_param(self.request_path, b"")
    write_param(self.progress_path, b"")
    yield ("start",)
    while True:
      self.clock.t += 1.0
      yield ("sleep",)
      try:
        yield from self._download_if_triggered()
      except MapdPanic as e:
        yield ("panic", e.args[0])
        continue
      write_param(self.request_path, b"")
      yield ("heartbeat",)

  def _download_if_triggered(self):
    self.progress = {"total_files": 0, "downloaded_files": 0, "locations_to_download": [], "location_details": {}}
    try:
      with open(self.request_path, "rb") as f:
        raw = f.read()
    except FileNotFoundError:
      return
    if not raw:
      return
    try:
      locations = json.loads(raw)
    except ValueError:
      return
    nations, states = locations.get("nations") or [], locations.get("states") or []
    for codes in (nations, states):
      for code in codes:
        self.progress["location_details"].setdefault(code, {"location_total_files": 0, "location_downloaded_files": 0})
    for kind, codes in (("nation", nations), ("state", states)):
      for code in codes:
        location = bounding_boxes(kind).get(code)
        if location is not None:
          yield from self._download_bounds(tiles_for_box(location["bounding_box"]), code)

  def _touch(self, path: str) -> None:
    self.clock.t += 0.001
    ns = int(self.clock.t * 1e9)
    os.utime(path, ns=(ns, ns))

  def _download_bounds(self, tiles, code):
    self.progress["location_details"][code]["location_total_files"] = len(tiles)
    for tile in tiles:
      archive = tile_archive(self.root, tile)
      os.makedirs(os.path.dirname(archive), exist_ok=True)
      with open(archive, "wb"):
        pass
      self._touch(archive)
      yield ("archive", tile)
      reply = self.server.get(tile, NOT_FOUND)
      self.gets.append(tile)
      if reply in (NET_ERROR, NOT_FOUND):
        yield ("download_failed", tile)
        continue
      body, limit = (reply.body, min(reply.n, len(reply.body))) if isinstance(reply, Cut) else (reply, len(reply))
      written = 0
      with open(archive, "ab") as f:
        while written < limit:
          part = body[written:min(limit, written + self.chunk)]
          f.write(part)
          f.flush()
          written += len(part)
          self._touch(archive)
          yield ("chunk", tile, written)
      if isinstance(reply, Cut):
        yield ("download_failed", tile)
        continue
      if len(body) < 10 or body[:3] != b"\x1f\x8b\x08":
        raise MapdPanic(tile)  # gzip.NewReader fails, then tar.Next on the nil reader panics
      yield from self._extract(tile, body)
      os.remove(archive)
      self.progress["downloaded_files"] += 1
      self.progress["location_details"][code]["location_downloaded_files"] += 1
      write_param(self.progress_path, json.dumps(self.progress).encode())
      yield ("tile_done", tile)
    shutil.rmtree(os.path.join(self.root, "tmp"), ignore_errors=True)
    yield ("tmp_removed",)

  def _extract(self, tile, body):
    d = zlib.decompressobj(31)
    data = b""
    for i in range(0, len(body), 64):  # a truncated or broken stream yields what came before the break
      try:
        data += d.decompress(body[i:i + 64])
      except zlib.error:
        break
    pos, index = 0, -1
    while pos + 512 <= len(data):
      header = data[pos:pos + 512]
      if header == b"\0" * 512:
        break
      name = header[:100].rstrip(b"\0").decode()
      size = int(header[124:136].strip(b"\0 ") or b"0", 8)
      kind = header[156:157]
      mode = int(header[100:108].strip(b"\0 ") or b"644", 8)
      pos += 512
      content = data[pos:pos + size]
      pos += (size + 511) // 512 * 512
      target = os.path.join(self.root, name)
      if kind == b"5":
        if not os.path.exists(target):
          os.makedirs(target, 0o755)
      elif kind in (b"0", b"\0"):
        index += 1
        if index in self.write_fails.get(tile, ()):
          # OpenFile succeeds, io.Copy fails (no space): only logged (logde), the file stays empty, extraction goes on
          os.close(os.open(target, os.O_CREAT | os.O_RDWR, mode))
          yield ("file_write_failed", tile, os.path.basename(name))
          continue
        try:
          fd = os.open(target, os.O_CREAT | os.O_RDWR, mode)  # no O_TRUNC: download.go:240
        except OSError:
          continue
        try:  # io.Copy writes in pieces: a crash can leave a file half written
          half = len(content) // 2
          os.write(fd, content[:half])
          yield ("file_part", tile, os.path.basename(name))
          os.write(fd, content[half:])
        finally:
          os.close(fd)
        yield ("file", tile, os.path.basename(name), len(content) < size)
        if len(content) < size:
          break  # io.Copy fails mid-file, the next tr.Next fails: extraction ends, the tile still counts as done
