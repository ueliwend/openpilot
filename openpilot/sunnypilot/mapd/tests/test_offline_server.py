"""TileServer HEAD sizes for the screen's estimates, with a stand-in for requests (no network). E9's probe() / check() are
gone with E11 (map_downloader checks each tile while it streams: test_tile_fetch.py)."""
import sys
import types

import pytest

from openpilot.sunnypilot.mapd import offline_server as srv


class Reply:
  def __init__(self, code, body=b"", length=None):
    self.status_code, self.headers = code, {} if length is None else {"Content-Length": str(length)}
    self.raw = types.SimpleNamespace(read=lambda n: body[:n])

  def __enter__(self):
    return self

  def __exit__(self, *a):
    return False


@pytest.fixture
def requests_stub(monkeypatch):
  replies = {}

  def answer(method):
    def call(url, **kw):
      reply = replies[method]
      if isinstance(reply, Exception):
        raise reply
      return reply
    return call
  monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(head=answer("head"), get=answer("get")))
  return replies


def test_head(requests_stub):
  s = srv.TileServer()
  for reply, expected in ((Reply(200, length=1234), 1234), (Reply(404), 0), (Reply(200), None), (Reply(405), None),
                          (Reply(403, length=10), None), (ConnectionError("down"), srv.NO_ANSWER_SIZE)):
    requests_stub["head"] = reply
    assert s.head((44, 6)) == expected, reply


def test_probe_and_check_are_gone():
  assert not hasattr(srv.TileServer, "probe") and not hasattr(srv, "check")
