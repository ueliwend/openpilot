"""A fake map tile server for the E11 tests (http.server in a thread, no network) and archives built like the real ones.

Archives: `tar -czvf offline/<lat>/<lon>.tar.gz offline/<lat>/<lon>` (mapd v1.12.0 scripts/compress_offline.sh): GNU tar
format, gzip, members `offline/<lat>/<lon>/` and `offline/<lat>/<lon>/<minLat>_<minLon>_<maxLat>_<maxLon>` (%f).
Each path has a Tile behaviour: status, ETag / Last-Modified, whether 304 / Range / If-Range are honoured, a drop after N
bytes, a trickle, a wrong Content-Length. Every request is recorded (method, path, headers).
"""
import gzip
import hashlib
import io
import tarfile
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def area_name(lat: float, lon: float, size: float = 0.25) -> str:
  return f"{lat:f}_{lon:f}_{lat + size:f}_{lon + size:f}"


def area_files(tile, n: int = 4, size: int = 3000, seed: str = "") -> dict[str, bytes]:
  lat, lon = tile
  out = {}
  for i in range(n):
    a, b = divmod(i, 8)
    name = area_name(lat + a * 0.25, lon + b * 0.25)
    out[name] = hashlib.sha256(f"{tile}{i}{seed}".encode()).digest() * (size // 32) + bytes([i % 251]) * (size % 32)
  return out


def make_tar(tile, files: dict[str, bytes], extra: list | None = None, folder: bool = True) -> bytes:
  lat, lon = tile
  buf = io.BytesIO()
  with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tar:
    if folder:
      d = tarfile.TarInfo(f"offline/{lat}/{lon}/")
      d.type, d.mode, d.mtime = tarfile.DIRTYPE, 0o755, 1_700_000_000
      tar.addfile(d)
    for name, data in files.items():
      info = tarfile.TarInfo(f"offline/{lat}/{lon}/{name}")
      info.size, info.mode, info.mtime = len(data), 0o644, 1_700_000_000
      tar.addfile(info, io.BytesIO(data))
    for info, data in extra or []:
      tar.addfile(info, io.BytesIO(data) if data is not None else None)
  return buf.getvalue()


def make_archive(tile, files: dict[str, bytes], extra: list | None = None) -> bytes:
  return gzip.compress(make_tar(tile, files, extra), mtime=0)


def md5_etag(body: bytes) -> str:
  return f'"{hashlib.md5(body).hexdigest()}"'


@dataclass
class Tile:
  body: bytes = b""
  status: int = 200
  etag: str | None = None
  last_modified: str | None = "Wed, 11 Nov 2026 10:00:00 GMT"
  honour_304: bool = True
  honour_range: bool = True
  honour_if_range: bool = True
  drop_at: int | None = None  # close the connection after this many body bytes (once)
  trickle: float = 0.0  # seconds between 1 KiB pieces
  length: int | None = None  # Content-Length to send (default: the body's)
  head_status: int | None = None
  drops: int = field(default=0)


class Handler(BaseHTTPRequestHandler):
  protocol_version = "HTTP/1.1"

  def log_message(self, *a):
    pass

  def _tile(self):
    srv = self.server
    srv.requests.append((self.command, self.path, {k.lower(): v for k, v in self.headers.items()}))
    return srv.tiles.get(self.path)

  def _headers(self, t: Tile, code: int, length: int, extra: dict | None = None):
    self.send_response(code)
    if t.etag:
      self.send_header("ETag", t.etag)
    if t.last_modified:
      self.send_header("Last-Modified", t.last_modified)
    self.send_header("Content-Type", "application/gzip")
    self.send_header("Content-Length", str(length))
    for k, v in (extra or {}).items():
      self.send_header(k, v)
    self.end_headers()

  def do_HEAD(self):
    t = self._tile()
    if t is None or (t.head_status or t.status) == 404:
      self.send_response(404)
      self.send_header("Content-Length", "0")
      self.end_headers()
      return
    code = t.head_status or t.status
    if code != 200:
      self.send_response(code)
      self.send_header("Content-Length", "0")
      self.end_headers()
      return
    self._headers(t, 200, len(t.body) if t.length is None else t.length)

  def do_GET(self):
    t = self._tile()
    if t is None or t.status != 200:
      code = 404 if t is None else t.status
      body = b"<html>error</html>"
      self.send_response(code)
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
      return
    h = {k.lower(): v for k, v in self.headers.items()}
    if t.honour_304 and ((t.etag and h.get("if-none-match") == t.etag) or
                         (not h.get("if-none-match") and t.last_modified and h.get("if-modified-since") == t.last_modified)):
      self.send_response(304)
      if t.etag:
        self.send_header("ETag", t.etag)
      self.send_header("Content-Length", "0")
      self.end_headers()
      return
    body, code, extra = t.body, 200, {}
    rng = h.get("range")
    if rng and t.honour_range:
      if_range = h.get("if-range")
      ok = not if_range or not t.honour_if_range or if_range in (t.etag, t.last_modified)
      if ok:
        start = int(rng.split("=")[1].split("-")[0])
        body, code = t.body[start:], 206
        extra = {"Content-Range": f"bytes {start}-{len(t.body) - 1}/{len(t.body)}"}
    length = len(body) if t.length is None or code == 206 else t.length
    self._headers(t, code, length, extra)
    sent = 0
    try:
      while sent < len(body):
        piece = body[sent:sent + (1024 if t.trickle else 65536)]
        if t.drop_at is not None and t.drops == 0 and code == 200 and sent + len(piece) > t.drop_at:
          piece = piece[:max(0, t.drop_at - sent)]
          self.wfile.write(piece)
          self.wfile.flush()
          t.drops += 1
          self.close_connection = True
          return
        self.wfile.write(piece)
        sent += len(piece)
        if t.trickle:
          self.wfile.flush()
          time.sleep(t.trickle)
    except (BrokenPipeError, ConnectionResetError):
      self.close_connection = True


class FakeTileServer:
  def __init__(self):
    self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    self.httpd.daemon_threads = True
    self.httpd.tiles = {}
    self.httpd.requests = []
    self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
    self.thread.start()

  @property
  def url(self) -> str:
    return f"http://127.0.0.1:{self.httpd.server_address[1]}"

  @property
  def requests(self) -> list:
    return self.httpd.requests

  def put(self, tile, t: Tile) -> Tile:
    self.httpd.tiles[f"/offline/{tile[0]}/{tile[1]}.tar.gz"] = t
    return t

  def gets(self, tile=None) -> list:
    path = None if tile is None else f"/offline/{tile[0]}/{tile[1]}.tar.gz"
    return [r for r in self.requests if r[0] == "GET" and (path is None or r[1] == path)]

  def heads(self, tile=None) -> list:
    path = None if tile is None else f"/offline/{tile[0]}/{tile[1]}.tar.gz"
    return [r for r in self.requests if r[0] == "HEAD" and (path is None or r[1] == path)]

  def close(self):
    self.httpd.shutdown()
    self.httpd.server_close()


def session():
  """A requests session that ignores the sandbox's proxy variables (the device has none)."""
  import requests
  s = requests.Session()
  s.trust_env = False
  return s
