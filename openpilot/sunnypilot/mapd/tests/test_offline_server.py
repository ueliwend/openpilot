"""TileServer (HEAD sizes, probe with its HTTP status) and check(), with a stand-in for requests (no network)."""
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


def test_probe_keeps_the_http_status(requests_stub):
  s = srv.TileServer()
  for reply, expected in ((Reply(206, b"\x1f\x8b\x08"), (srv.GZIP, 206)), (Reply(200, b"\x1f\x8b"), (srv.GZIP, 200)),
                          (Reply(200, b"<html>"), (srv.NOT_GZIP, 200)), (Reply(200, b""), (srv.NOT_GZIP, 200)),
                          (Reply(404), (srv.NO_DATA, 404)), (Reply(403), (srv.SERVER_ERROR, 403)), (Reply(500), (srv.SERVER_ERROR, 500)),
                          (TimeoutError("slow"), (srv.NO_ANSWER, None))):
    requests_stub["get"] = reply
    assert s.probe((44, 6)) == expected, reply


def test_check_stops_at_the_first_head_without_answer_and_probes_past_404s():
  asked = []

  class Server:
    def head(self, t):
      asked.append(("head", t))
      return {(1, 1): 0, (2, 2): srv.NO_ANSWER_SIZE}.get(t, 100)

    def probe(self, t):
      asked.append(("probe", t))
      return (srv.NO_DATA, 404) if t == (3, 3) else (srv.GZIP, 200)
  r = srv.check(Server(), [(0, 0), (1, 1), (3, 3), (4, 4)], [(1, 1), (2, 2), (5, 5)], set())
  assert r["answer"] == srv.NO_ANSWER and asked == [("head", (1, 1)), ("head", (2, 2))]
  asked.clear()
  r = srv.check(Server(), [(0, 0), (1, 1), (3, 3), (4, 4)], [(1, 1)], {(0, 0)})
  assert asked == [("head", (1, 1)), ("probe", (3, 3)), ("probe", (4, 4))]
  assert r == {"sizes": {(1, 1): 0}, "answer": srv.GZIP, "code": 200, "no_data": [(3, 3)]}
