"""One map tile from the tile server into a work folder, checked as it streams (Inster port, E11).

The tile host serves `offline/<lat>/<lon>.tar.gz`, made by `tar -czvf` (mapd v1.12.0 scripts/compress_offline.sh). mapd
v1.12.0 extracts it with no checks (download.go:201-245). Here a tile counts as complete only if:
- every member is the tile folder `offline/<lat>/<lon>/` (or a parent folder) or an area file
  `offline/<lat>/<lon>/<minLat>_<minLon>_<maxLat>_<maxLon>` (%f, negative values allowed) inside the requested tile:
  directories and regular files only, no `..`, no absolute path, no links, no repeats, no empty file;
- the tar end was read (zero blocks, nothing but zeros after it);
- the gzip stream reached its end (CRC32 and length checked by zlib) with exactly one member and no trailing bytes;
- the byte count equals Content-Length (when the server sends one);
- a strong ETag that looks like a plain MD5 (`"<32 hex>"`, never `W/"..."`) equals the MD5 of the bytes.
Files are written with O_TRUNC and mode 0644, hashed (SHA-256) while written, then fdatasync + POSIX_FADV_DONTNEED.

Requests: `Accept-Encoding: identity`, 10 s connect / 30 s read timeouts, 64 KiB reads under a token bucket (the cap while
driving). Which request (DESIGN 5.1): a resume of `<tile>.tar.gz.part` (Range + If-Range with the stored strong ETag or
Last-Modified; 206 continues after replaying the part, 200 starts again); else a conditional GET for a live tile with
validators (304 = unchanged; 200 with the stored ETag, or the same Last-Modified and length = the host ignores conditional
requests: the body is closed, the caller marks the host `no_304`); for a `no_304` host a HEAD first (equal = unchanged);
else a plain GET. 404 = no data for this tile. tick() is called at least twice a second and may raise Abort.
"""
import hashlib
import json
import os
import re
import time
import zlib

from openpilot.sunnypilot.mapd import offline_swap as sw

CHUNK = 64 * 1024
TIMEOUT = (10, 30)  # connect, read
MAX_TILE_DISK_BYTES = 2_000_000_000  # an archive that unpacks to more than this is refused (EST: the largest tiles ~0.4 GB)
TICK_SECONDS = 0.5
MD5_ETAG = re.compile(r'^"([0-9a-fA-F]{32})"$')
NUM = r"-?\d+\.\d{6}"
AREA = re.compile(rf"^({NUM})_({NUM})_({NUM})_({NUM})$")

# outcomes
OK, NOT_MODIFIED, IGNORED_304, SAME_HEAD, NO_DATA, SERVER_ERROR, NO_ANSWER, BAD, ABORTED = (
  "ok", "not_modified", "ignored_304", "same_head", "no_data", "server_error", "no_answer", "bad", "aborted")


class Abort(Exception):
  """Raised by tick() to stop a transfer (pause, cancel); the .part is kept."""


class BadTile(Exception):
  pass


class Restart(Exception):
  pass


class Outcome:
  def __init__(self, kind: str, code: int | None = None, reason: str = "", headers: dict | None = None, files: dict | None = None,
               length: int = 0, disk: int = 0):
    self.kind, self.code, self.reason = kind, code, reason
    self.headers = headers or {}
    self.files = files or {}  # area file name -> [size, sha256]
    self.length, self.disk = length, disk

  @property
  def etag(self):
    return self.headers.get("etag")

  @property
  def last_modified(self):
    return self.headers.get("last_modified")

  def __repr__(self):
    return f"Outcome({self.kind}, {self.code}, {self.reason!r})"


def strong_etag(etag) -> str | None:
  return etag if isinstance(etag, str) and etag.startswith('"') and not etag.startswith("W/") else None


def validators(resp_headers) -> dict:
  h = {k.lower(): v for k, v in resp_headers.items()}
  try:
    length = int(h["content-length"]) if "content-length" in h else None
  except ValueError:
    length = None
  return {"etag": h.get("etag"), "last_modified": h.get("last-modified"), "length": length}


def same_archive(a: dict, b: dict) -> bool:
  """Validators describe the same archive: equal ETag, else equal Last-Modified and length."""
  if a.get("etag") and b.get("etag"):
    return a["etag"] == b["etag"]
  return bool(a.get("last_modified")) and a.get("last_modified") == b.get("last_modified") and \
    a.get("length") is not None and a.get("length") == b.get("length")


class TokenBucket:
  """At most rate() bytes/s (None = no cap); the burst is one read."""

  def __init__(self, rate, monotonic=time.monotonic, sleep=time.sleep):
    self.rate, self._monotonic, self._sleep = rate, monotonic, sleep
    self.tokens = float(CHUNK)
    self.at = monotonic()

  def take(self, n: int, tick) -> None:
    """Accounts n bytes just read; sleeps (calling tick) until the average is back under the cap."""
    self.tokens -= n
    while True:
      rate = self.rate()
      now = self._monotonic()
      if rate is None:
        self.tokens, self.at = float(CHUNK), now
        return
      self.tokens = min(float(CHUNK), self.tokens + (now - self.at) * rate)
      self.at = now
      if self.tokens >= -1.0:  # within a byte: float rounding never loops
        return
      self._sleep(min(TICK_SECONDS, max(0.001, -self.tokens / rate)))
      tick()


class TarWriter:
  """Streaming tar reader (push mode) that writes the area files of one tile into `dest`."""

  def __init__(self, dest: str, tile):
    self.dest, self.tile = dest, tile
    self.buf = b""
    self.file = None  # (fd, name, remaining, padding, sha, size)
    self.skip = 0
    self.ended = False
    self.files: dict[str, list] = {}
    self.disk = 0
    os.makedirs(dest, exist_ok=True)

  def _member(self, name: str, typeflag: bytes) -> str | None:
    """The area file name for a file member, None for an allowed folder; BadTile otherwise."""
    lat, lon = self.tile
    if name.startswith("/") or "\\" in name or any(p in ("..", ".") for p in name.rstrip("/").split("/")):
      raise BadTile(f"unsafe member {name!r}")
    parts = name.rstrip("/").split("/")
    if typeflag == b"5":
      if parts in (["offline"], ["offline", str(lat)], ["offline", str(lat), str(lon)]):
        return None
      raise BadTile(f"unexpected folder {name!r}")
    if typeflag not in (b"0", b"\0"):
      raise BadTile(f"member type {typeflag!r} not allowed: {name!r}")
    if len(parts) != 4 or parts[:3] != ["offline", str(lat), str(lon)] or name.endswith("/"):
      raise BadTile(f"member outside the tile: {name!r}")
    m = AREA.match(parts[3])
    if m is None:
      raise BadTile(f"bad area name {name!r}")
    min_lat, min_lon, max_lat, max_lon = (float(x) for x in m.groups())
    if not (lat <= min_lat < max_lat <= lat + 2 and lon <= min_lon < max_lon <= lon + 2):
      raise BadTile(f"area outside the tile: {name!r}")
    if parts[3] in self.files:
      raise BadTile(f"repeated member {name!r}")
    return parts[3]

  def _header(self, block: bytes) -> None:
    if block == b"\0" * 512:
      self.ended = True
      return
    stored = block[148:156].split(b"\0")[0].strip()
    try:
      chk = int(stored, 8)
    except ValueError:
      raise BadTile("bad tar header checksum field") from None
    unsigned = sum(block[:148]) + 8 * 32 + sum(block[156:])
    if chk != unsigned:
      raise BadTile("tar header checksum mismatch")
    name = block[0:100].split(b"\0")[0].decode("utf-8", "strict")
    if block[257:263] == b"ustar\x00":  # POSIX ustar: optional prefix
      prefix = block[345:500].split(b"\0")[0].decode("utf-8", "strict")
      if prefix:
        name = f"{prefix}/{name}"
    raw = block[124:136]
    if raw[0] & 0x80:
      raise BadTile("tar size field in base-256 (file too large)")
    try:
      size = int(raw.split(b"\0")[0].strip() or b"0", 8)
    except ValueError:
      raise BadTile("bad tar size field") from None
    typeflag = block[156:157]
    area = self._member(name, typeflag)
    padding = (512 - size % 512) % 512
    if area is None:
      if size:
        raise BadTile(f"folder with data: {name!r}")
      return
    if size == 0:
      raise BadTile(f"empty area file {name!r}")
    self.disk += size
    if self.disk > MAX_TILE_DISK_BYTES:
      raise BadTile("tile unpacks to more than the limit")
    fd = os.open(os.path.join(self.dest, area), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.fchmod(fd, 0o644)
    self.file = [fd, area, size, padding, hashlib.sha256(), size]

  def _close_file(self) -> None:
    fd, area, _, _, sha, size = self.file
    try:
      os.fdatasync(fd)
      if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
      os.close(fd)
    self.files[area] = [size, sha.hexdigest()]
    self.skip = self.file[3]
    self.file = None

  def feed(self, data: bytes) -> None:
    view = memoryview(data)
    while view:
      if self.ended:
        if bytes(view).strip(b"\0"):
          raise BadTile("data after the tar end")
        return
      if self.file is not None:
        n = min(self.file[2], len(view))
        piece = view[:n]
        os.write(self.file[0], piece)  # regular files on ext4: short writes do not happen for these sizes
        self.file[4].update(piece)
        self.file[2] -= n
        view = view[n:]
        if self.file[2] == 0:
          self._close_file()
        continue
      if self.skip:
        n = min(self.skip, len(view))
        self.skip -= n
        view = view[n:]
        continue
      need = 512 - len(self.buf)
      self.buf += bytes(view[:need])
      view = view[need:]
      if len(self.buf) == 512:
        block, self.buf = self.buf, b""
        self._header(block)

  def finish(self) -> None:
    if self.file is not None or self.skip or self.buf.strip(b"\0") or not self.ended:
      raise BadTile("tar ended early")

  def abandon(self) -> None:
    if self.file is not None:
      os.close(self.file[0])
      self.file = None


class Pipeline:
  """gzip (one member, nothing after it) -> tar -> files; MD5 and count of the compressed bytes."""

  def __init__(self, dest: str, tile):
    self.z = zlib.decompressobj(16 + zlib.MAX_WBITS)
    self.tar = TarWriter(dest, tile)
    self.md5 = hashlib.md5()
    self.count = 0

  def feed(self, data: bytes) -> None:
    if not data:
      return
    self.md5.update(data)
    self.count += len(data)
    if self.z.eof:
      raise BadTile("bytes after the gzip member (second member or trailing data)")
    try:
      out = self.z.decompress(data, 1 << 20)
      self.tar.feed(out)
      while self.z.unconsumed_tail and not self.z.eof:
        out = self.z.decompress(self.z.unconsumed_tail, 1 << 20)
        self.tar.feed(out)
    except zlib.error as e:
      raise BadTile(f"gzip: {e}") from None
    if self.z.eof and self.z.unused_data:
      raise BadTile("bytes after the gzip member (second member or trailing data)")

  def finish(self, expected_length: int | None, etag) -> None:
    if not self.z.eof:
      raise BadTile("gzip stream ended early")
    self.tar.finish()
    if expected_length is not None and self.count != expected_length:
      raise BadTile(f"length {self.count} != Content-Length {expected_length}")
    m = MD5_ETAG.match(etag or "")
    if m and m.group(1).lower() != self.md5.hexdigest():
      raise BadTile("MD5 does not match the ETag")

  def abandon(self) -> None:
    self.tar.abandon()


def _part_meta_path(part: str) -> str:
  return part + ".json"


def remove_part(part: str) -> None:
  for p in (part, _part_meta_path(part)):
    try:
      os.remove(p)
    except FileNotFoundError:
      pass


class Fetcher:
  def __init__(self, session=None, base_url: str = "https://map-data.pfeifer.dev", timeout=TIMEOUT, monotonic=time.monotonic,
               sleep=time.sleep):
    self.base_url = base_url.rstrip("/")
    self.timeout = timeout
    self._session = session
    self._monotonic, self._sleep = monotonic, sleep

  @property
  def session(self):
    if self._session is None:
      import requests
      self._session = requests.Session()
    return self._session

  def url(self, tile) -> str:
    return f"{self.base_url}/offline/{tile[0]}/{tile[1]}.tar.gz"

  def head(self, tile) -> dict | None:
    """Validators from a HEAD: {"status", "etag", "last_modified", "length"}; None when the server does not answer."""
    try:
      r = self.session.head(self.url(tile), headers={"Accept-Encoding": "identity"}, timeout=self.timeout, allow_redirects=True)
    except Exception:
      return None
    out = validators(r.headers)
    out["status"] = r.status_code
    return out

  def check(self, tile, part: str, work: str, live: dict | None, no_304: bool, tick, rate) -> Outcome:
    """Checks one tile and, if it changed (or is new), streams it into `work`. live: the live tile's manifest entry
    (validators "etag", "last_modified", "download_bytes"), None for a tile not on disk."""
    tick = _throttled(tick, self._monotonic)
    meta = sw.read_json(_part_meta_path(part))
    if os.path.exists(part) and isinstance(meta, dict) and (strong_etag(meta.get("etag")) or meta.get("last_modified")):
      try:
        return self._get(tile, part, work, None, resume=meta, tick=tick, rate=rate)
      except Restart:
        pass
    remove_part(part)
    stored = {"etag": (live or {}).get("etag"), "last_modified": (live or {}).get("last_modified"),
              "length": (live or {}).get("download_bytes")}
    has_validators = bool(stored["etag"] or stored["last_modified"])
    if live is not None and has_validators and no_304:
      h = self.head(tile)
      if h is None:
        return Outcome(NO_ANSWER, reason="HEAD: no answer")
      if h["status"] == 200 and same_archive(h, stored):
        return Outcome(SAME_HEAD, 200, headers=h)
      if h["status"] == 404:
        return Outcome(NO_DATA, 404)
      if h["status"] not in (200, 405, 501):  # a refused HEAD falls through to a GET
        return Outcome(SERVER_ERROR, h["status"])
      return self._get(tile, part, work, None, resume=None, tick=tick, rate=rate)
    conditional = stored if (live is not None and has_validators) else None
    try:
      return self._get(tile, part, work, conditional, resume=None, tick=tick, rate=rate)
    except Restart:
      remove_part(part)
      return self._get(tile, part, work, None, resume=None, tick=tick, rate=rate)

  def _get(self, tile, part, work, conditional, resume, tick, rate) -> Outcome:
    headers = {"Accept-Encoding": "identity"}
    if conditional:
      if conditional.get("etag"):
        headers["If-None-Match"] = conditional["etag"]
      if conditional.get("last_modified"):
        headers["If-Modified-Since"] = conditional["last_modified"]
    offset = 0
    if resume:
      offset = os.path.getsize(part)
      headers["Range"] = f"bytes={offset}-"
      headers["If-Range"] = strong_etag(resume.get("etag")) or resume["last_modified"]
    tick()
    try:
      r = self.session.get(self.url(tile), headers=headers, timeout=self.timeout, stream=True, allow_redirects=True)
    except Abort:
      raise
    except Exception as e:
      return Outcome(NO_ANSWER, reason=f"{type(e).__name__}")
    with r:
      return self._body(r, tile, part, work, conditional, resume, offset, tick, rate)

  def _body(self, r, tile, part, work, conditional, resume, offset, tick, rate) -> Outcome:
    v = validators(r.headers)
    if r.status_code == 304:
      return Outcome(NOT_MODIFIED, 304, headers=v)
    if r.status_code == 404:
      return Outcome(NO_DATA, 404)
    if r.status_code not in (200, 206) or (r.status_code == 206 and not resume):
      return Outcome(SERVER_ERROR, r.status_code)
    if conditional and r.status_code == 200 and same_archive(v, conditional):
      return Outcome(IGNORED_304, 200, headers=v)  # the server ignores conditional requests; body not read
    total = v["length"]
    if r.status_code == 206:
      m = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)", r.headers.get("Content-Range", ""))
      if m is None or int(m.group(1)) != offset or (resume.get("etag") and v["etag"] and v["etag"] != resume["etag"]):
        raise Restart  # not the continuation of our part: start again
      total = int(m.group(3)) if m.group(3) != "*" else None
      v = {"etag": resume.get("etag"), "last_modified": resume.get("last_modified"), "length": total}
    else:
      offset = 0
      remove_part(part)
      if strong_etag(v["etag"]) or v["last_modified"]:
        sw.write_json_atomic(_part_meta_path(part), v)
    if os.path.lexists(work):
      sw._rmtree(work)
    pipe = Pipeline(work, tile)
    os.makedirs(os.path.dirname(part), exist_ok=True)
    try:
      with open(part, "ab" if offset else "wb") as out:
        if offset:  # replay the bytes kept from before
          with open(part, "rb") as old:
            while chunk := old.read(CHUNK):
              pipe.feed(chunk)
              tick()
        bucket = TokenBucket(rate, self._monotonic, self._sleep)
        while True:
          tick()
          try:
            chunk = r.raw.read1(CHUNK) if hasattr(r.raw, "read1") else r.raw.read(CHUNK, decode_content=False)
          except Abort:
            raise
          except Exception as e:
            out.flush()
            return Outcome(NO_ANSWER, reason=f"transfer: {type(e).__name__}", headers=v)
          if not chunk:
            break
          out.write(chunk)
          pipe.feed(chunk)
          bucket.take(len(chunk), tick)
        out.flush()
        os.fdatasync(out.fileno())
      pipe.finish(total, v["etag"])
    except BadTile as e:
      pipe.abandon()
      return Outcome(BAD, r.status_code, reason=str(e), headers=v)
    except Abort:
      pipe.abandon()
      raise
    except OSError:
      pipe.abandon()
      raise
    return Outcome(OK, r.status_code, headers=v, files=pipe.tar.files, length=pipe.count, disk=pipe.tar.disk)


def _throttled(tick, monotonic):
  last = [-1e18]

  def call():
    now = monotonic()
    if now - last[0] >= TICK_SECONDS / 2:
      last[0] = now
      tick()
  return call


def hash_tree(folder: str) -> dict[str, list]:
  """{area file name: [size, sha256]} of a tile folder on disk (pre-E11 tiles, or a live tile without a hash list)."""
  out = {}
  for e in sorted(os.scandir(folder), key=lambda x: x.name):
    if not e.is_file(follow_symlinks=False):
      continue
    sha = hashlib.sha256()
    size = 0
    with open(e.path, "rb") as f:
      while chunk := f.read(CHUNK):
        sha.update(chunk)
        size += len(chunk)
    out[e.name] = [size, sha.hexdigest()]
  return out


def dump(obj) -> str:
  return json.dumps(obj, separators=(",", ":"))
