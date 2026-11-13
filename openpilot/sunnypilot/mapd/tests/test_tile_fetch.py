"""E11 tile fetch against the fake tile server (DESIGN 9.1): archive checks, request kinds, resume, cap, abort."""
import gzip
import hashlib
import io
import os
import stat
import tarfile
import time

import pytest

from openpilot.sunnypilot.mapd import tile_fetch as tf
from openpilot.sunnypilot.mapd.tests.tile_server import FakeTileServer, Tile, area_files, make_archive, md5_etag, session

T = (46, 8)
NEG = (42, -6)


@pytest.fixture
def server():
  s = FakeTileServer()
  yield s
  s.close()


def fetcher(server):
  return tf.Fetcher(session=session(), base_url=server.url, timeout=(2, 3))


def paths(tmp_path, tile=T):
  return str(tmp_path / "work" / f"{tile[0]}" / f"{tile[1]}.tar.gz.part"), str(tmp_path / "work" / f"{tile[0]}" / f"{tile[1]}")


def run(server, tmp_path, tile=T, live=None, no_304=False, tick=lambda: None, rate=lambda: None):
  part, work = paths(tmp_path, tile)
  return fetcher(server).check(tile, part, work, live, no_304, tick, rate), part, work


def hashes(files):
  return {k: [len(v), hashlib.sha256(v).hexdigest()] for k, v in files.items()}


def test_plain_get_writes_checked_files_with_mode_0644(server, tmp_path):
  files = area_files(NEG, n=6)
  body = make_archive(NEG, files)
  server.put(NEG, Tile(body=body, etag=md5_etag(body)))
  out, part, work = run(server, tmp_path, NEG)
  assert out.kind == tf.OK and out.files == hashes(files) and out.length == len(body)
  for name, data in files.items():
    p = os.path.join(work, name)
    assert open(p, "rb").read() == data and stat.S_IMODE(os.stat(p).st_mode) == 0o644
  assert server.gets(NEG)[0][2]["accept-encoding"] == "identity"
  assert os.path.getsize(part) == len(body)  # kept until the commit removes it


@pytest.mark.parametrize("status,kind", [(404, tf.NO_DATA), (403, tf.SERVER_ERROR), (500, tf.SERVER_ERROR)])
def test_status_codes(server, tmp_path, status, kind):
  server.put(T, Tile(body=b"x", status=status))
  out, _, _ = run(server, tmp_path)
  assert out.kind == kind and out.code == status


def test_no_answer(tmp_path):
  f = tf.Fetcher(session=session(), base_url="http://127.0.0.1:9", timeout=(1, 1))
  part, work = paths(tmp_path)
  assert f.check(T, part, work, None, False, lambda: None, lambda: None).kind == tf.NO_ANSWER


def test_html_body_with_200_is_bad(server, tmp_path):
  server.put(T, Tile(body=b"<html>maintenance</html>"))
  assert run(server, tmp_path)[0].kind == tf.BAD


def test_content_length_shorter_than_the_archive_is_bad(server, tmp_path):
  body = make_archive(T, area_files(T))
  server.put(T, Tile(body=body, length=len(body) - 10))
  assert run(server, tmp_path)[0].kind == tf.BAD


def test_drop_then_resume_with_range(server, tmp_path):
  files = area_files(T, n=8, size=40000)
  body = make_archive(T, files)
  server.put(T, Tile(body=body, etag='"v1"', drop_at=len(body) // 2))
  out, part, _ = run(server, tmp_path)
  assert out.kind == tf.NO_ANSWER and 0 < os.path.getsize(part) < len(body)
  out, _, _ = run(server, tmp_path)
  assert out.kind == tf.OK and out.files == hashes(files) and out.length == len(body)
  last = server.gets(T)[-1][2]
  assert last["range"] == f"bytes={len(body) // 2}-" and last["if-range"] == '"v1"'


def test_resume_when_the_server_ignores_range(server, tmp_path):
  files = area_files(T, n=8, size=40000)
  body = make_archive(T, files)
  server.put(T, Tile(body=body, etag='"v1"', drop_at=len(body) // 3, honour_range=False))
  assert run(server, tmp_path)[0].kind == tf.NO_ANSWER
  out, _, _ = run(server, tmp_path)
  assert out.kind == tf.OK and out.files == hashes(files)


def test_etag_changes_between_the_part_and_the_resume(server, tmp_path):
  body1 = make_archive(T, area_files(T, n=8, size=40000))
  server.put(T, Tile(body=body1, etag='"v1"', drop_at=len(body1) // 2))
  assert run(server, tmp_path)[0].kind == tf.NO_ANSWER
  files2 = area_files(T, n=8, size=40000, seed="new")
  body2 = make_archive(T, files2)
  server.put(T, Tile(body=body2, etag='"v2"'))  # If-Range does not match: a full 200
  out, _, _ = run(server, tmp_path)
  assert out.kind == tf.OK and out.files == hashes(files2) and out.etag == '"v2"'


def test_server_that_ignores_if_range_and_sends_another_archive(server, tmp_path):
  body1 = make_archive(T, area_files(T, n=8, size=40000))
  server.put(T, Tile(body=body1, etag='"v1"', drop_at=len(body1) // 2))
  assert run(server, tmp_path)[0].kind == tf.NO_ANSWER
  files2 = area_files(T, n=8, size=40000, seed="new")
  server.put(T, Tile(body=make_archive(T, files2), etag='"v2"', honour_if_range=False))  # 206 of the new archive
  out, _, _ = run(server, tmp_path)
  assert out.kind == tf.OK and out.files == hashes(files2)


def live_entry(body, etag='"v1"'):
  return {"etag": etag, "last_modified": "Wed, 11 Nov 2026 10:00:00 GMT", "download_bytes": len(body)}


def test_conditional_get_304(server, tmp_path):
  body = make_archive(T, area_files(T))
  server.put(T, Tile(body=body, etag='"v1"'))
  out, _, _ = run(server, tmp_path, live=live_entry(body))
  assert out.kind == tf.NOT_MODIFIED and server.gets(T)[0][2]["if-none-match"] == '"v1"'


def test_server_ignoring_304_is_seen_and_the_body_not_read(server, tmp_path):
  body = make_archive(T, area_files(T, n=8, size=200000))
  server.put(T, Tile(body=body, etag='"v1"', honour_304=False))
  out, _, work = run(server, tmp_path, live=live_entry(body))
  assert out.kind == tf.IGNORED_304 and not os.path.exists(work)


def test_no_304_host_asks_head_first(server, tmp_path):
  body = make_archive(T, area_files(T))
  server.put(T, Tile(body=body, etag='"v1"', honour_304=False))
  out, _, _ = run(server, tmp_path, live=live_entry(body), no_304=True)
  assert out.kind == tf.SAME_HEAD and len(server.heads(T)) == 1 and not server.gets(T)
  server.put(T, Tile(body=make_archive(T, area_files(T, seed="x")), etag='"v2"', honour_304=False))
  out, _, _ = run(server, tmp_path, live=live_entry(body), no_304=True)
  assert out.kind == tf.OK and len(server.gets(T)) == 1 and "if-none-match" not in server.gets(T)[0][2]


def test_md5_etag_checked_weak_etag_never(server, tmp_path):
  body = make_archive(T, area_files(T))
  server.put(T, Tile(body=body, etag='"' + "0" * 32 + '"'))
  assert run(server, tmp_path)[0].kind == tf.BAD
  server.put(T, Tile(body=body, etag='W/"' + "0" * 32 + '"'))
  assert run(server, tmp_path)[0].kind == tf.OK
  server.put(T, Tile(body=body, etag=md5_etag(body)))
  assert run(server, tmp_path)[0].kind == tf.OK


def info(name, kind=tarfile.REGTYPE, size=0, link=""):
  i = tarfile.TarInfo(name)
  i.type, i.size, i.linkname, i.mode = kind, size, link, 0o644
  return i


UNSAFE = {
  "dotdot": [(info("offline/46/8/../../x", size=1), b"x")],
  "absolute": [(info("/etc/passwd", size=1), b"x")],
  "symlink": [(info("offline/46/8/link", tarfile.SYMTYPE, link="/etc/passwd"), None)],
  "hardlink": [(info("offline/46/8/hl", tarfile.LNKTYPE, link="offline/46/8/x"), None)],
  "other_tile": [(info("offline/48/8/48.000000_8.000000_48.250000_8.250000", size=1), b"x")],
  "bad_name": [(info("offline/46/8/readme.txt", size=1), b"x")],
  "outside_bounds": [(info("offline/46/8/50.000000_8.000000_50.250000_8.250000", size=1), b"x")],
  "empty_file": [(info("offline/46/8/46.500000_8.500000_46.750000_8.750000", size=0), b"")],
  "repeat": [(info("offline/46/8/46.000000_8.000000_46.250000_8.250000", size=1), b"x")],
}


REASON = {"dotdot": "unsafe member", "absolute": "unsafe member", "symlink": "member type", "hardlink": "member type",
          "other_tile": "member outside", "bad_name": "bad area name", "outside_bounds": "area outside", "empty_file": "empty area",
          "repeat": "repeated member"}


@pytest.mark.parametrize("case", sorted(UNSAFE))
def test_unsafe_members_fail_the_tile(server, tmp_path, case):
  server.put(T, Tile(body=make_archive(T, area_files(T, n=1), extra=UNSAFE[case])))
  out, _, _ = run(server, tmp_path)
  assert out.kind == tf.BAD and out.reason.startswith(REASON[case]), out  # each guard by itself, not a later one


def test_link_named_like_an_area_file_is_refused(tmp_path):
  w = tf.TarWriter(str(tmp_path / "w"), T)
  for flag in (b"1", b"2"):
    with pytest.raises(tf.BadTile, match="member type"):
      w._member("offline/46/8/46.000000_8.000000_46.250000_8.250000", flag)


def test_pipeline_rejects_bytes_after_the_member_in_the_same_read(tmp_path):
  good = make_archive(T, area_files(T))
  for extra in (b"garbage", gzip.compress(b"second", mtime=0)):
    p = tf.Pipeline(str(tmp_path / f"p{len(extra)}"), T)
    with pytest.raises(tf.BadTile, match="after the gzip member"):
      p.feed(good + extra)  # one read: only the unused_data check sees it


def test_pipeline_checks_the_length(tmp_path):
  good = make_archive(T, area_files(T))
  p = tf.Pipeline(str(tmp_path / "p"), T)
  p.feed(good)
  with pytest.raises(tf.BadTile, match="Content-Length"):
    p.finish(len(good) + 1, None)


def test_tar_without_end_blocks_is_refused(server, tmp_path):
  from openpilot.sunnypilot.mapd.tests.tile_server import make_tar
  tar = make_tar(T, area_files(T)).rstrip(b"\0")
  tar += b"\0" * ((512 - len(tar) % 512) % 512)  # complete members, no end blocks
  server.put(T, Tile(body=gzip.compress(tar, mtime=0)))
  out, _, _ = run(server, tmp_path)
  assert out.kind == tf.BAD and "tar ended early" in out.reason


def test_negative_area_names_are_accepted():
  name = "offline/42/-6/42.750000_-5.000000_43.000000_-4.750000"
  w = tf.TarWriter.__new__(tf.TarWriter)
  w.tile, w.files = NEG, {}
  assert w._member(name, b"0") == "42.750000_-5.000000_43.000000_-4.750000"


def test_bad_crc_trailing_garbage_and_two_members(server, tmp_path):
  good = make_archive(T, area_files(T))
  bad_crc = good[:-8] + bytes([good[-8] ^ 0xFF]) + good[-7:]
  for body in (bad_crc, good + b"garbage", good + gzip.compress(b"second", mtime=0), good[:-20]):
    server.put(T, Tile(body=body))
    assert run(server, tmp_path)[0].kind == tf.BAD


def test_tar_without_its_end_is_bad(server, tmp_path):
  raw = io.BytesIO()
  from openpilot.sunnypilot.mapd.tests.tile_server import make_tar
  tar = make_tar(T, area_files(T))
  raw.write(tar[:-1024 * 10])  # cut inside the last member / before the end blocks
  server.put(T, Tile(body=gzip.compress(raw.getvalue(), mtime=0)))
  assert run(server, tmp_path)[0].kind == tf.BAD


def test_abort_from_tick_keeps_the_part_even_on_a_trickle(server, tmp_path):
  body = make_archive(T, {k: os.urandom(len(v)) for k, v in area_files(T, n=8, size=40000).items()})
  server.put(T, Tile(body=body, etag='"v1"', trickle=0.05))
  t0 = time.monotonic()

  def tick():
    if time.monotonic() - t0 > 0.6:
      raise tf.Abort("pause")
  with pytest.raises(tf.Abort):
    run(server, tmp_path, tick=tick)
  assert time.monotonic() - t0 < 2.0  # reads return as data arrives (read1), so a pause stops a trickle at once
  part, _ = paths(tmp_path)
  assert 0 < os.path.getsize(part) < len(body)


def test_token_bucket_holds_the_cap():
  clock = [0.0]
  slept = []

  def sleep(s):
    slept.append(s)
    clock[0] += s
  b = tf.TokenBucket(lambda: 1_000_000, monotonic=lambda: clock[0], sleep=sleep)
  for _ in range(160):  # 10 MiB in 64 KiB reads, read instantly
    b.take(tf.CHUNK, lambda: None)
  assert 160 * tf.CHUNK / 1e6 - 0.1 <= clock[0] <= 160 * tf.CHUNK / 1e6 + 0.1
  assert max(slept) <= tf.TICK_SECONDS
  b2 = tf.TokenBucket(lambda: None, monotonic=lambda: clock[0], sleep=sleep)
  n = len(slept)
  b2.take(10 * tf.CHUNK, lambda: None)
  assert len(slept) == n  # no cap offroad


def test_capped_transfer_takes_its_time(server, tmp_path):
  body = make_archive(T, {k: os.urandom(len(v)) for k, v in area_files(T, n=4, size=60000).items()})
  server.put(T, Tile(body=body))
  t0 = time.monotonic()
  out, _, _ = run(server, tmp_path, rate=lambda: 200_000)
  assert out.kind == tf.OK and time.monotonic() - t0 >= len(body) / 200_000 - 0.5
