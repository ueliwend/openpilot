"""Test harness for several map locations: offline_maps.OfflineMaps + the updater + the fake mapd, with real Params,
a fake tile server (same replies as the fake mapd gets), a fake disk size and fake drive bytes."""
from openpilot.sunnypilot.mapd.offline_maps import OfflineMaps
from openpilot.sunnypilot.mapd.offline_regions import entry_tiles, union_tiles
from openpilot.sunnypilot.mapd.offline_server import GZIP, GZIP_MAGIC, NO_ANSWER, NO_ANSWER_SIZE, NO_DATA, NOT_GZIP, SERVER_ERROR
from openpilot.sunnypilot.mapd.offline_storage import Disk, floor_bytes
from openpilot.sunnypilot.mapd.offline_selection import entry_from_id
from openpilot.sunnypilot.mapd.tests.fake_mapd import NET_ERROR, NOT_FOUND, Cut, HttpStatus
from openpilot.sunnypilot.mapd.tests.offline_world import World, area_files
import openpilot.sunnypilot.mapd.offline_update as ou

GB = 1_000_000_000


class FakeServer:
  def __init__(self, world):
    self.world = world
    self.heads: list = []
    self.probes: list = []

  def _body(self, tile):
    reply = self.world.server.get(tile, NOT_FOUND)
    if reply in (NOT_FOUND, NET_ERROR) or isinstance(reply, HttpStatus):
      return reply
    return reply.body if isinstance(reply, Cut) else reply

  def head(self, tile):
    self.heads.append(tile)
    body = self._body(tile)
    if isinstance(body, HttpStatus):
      return None  # a status other than 200 / 404: size unknown
    return 0 if body == NOT_FOUND else NO_ANSWER_SIZE if body == NET_ERROR else (len(body) or None)  # 200 without length: unknown

  def probe(self, tile):
    self.probes.append(tile)
    body = self._body(tile)
    if isinstance(body, HttpStatus):
      return SERVER_ERROR, body.code
    if body == NOT_FOUND:
      return NO_DATA, 404
    if body == NET_ERROR:
      return NO_ANSWER, None
    return (GZIP if body[:2] == GZIP_MAGIC else NOT_GZIP), 200


class FakeDisk:
  """capacity fixed; available = floor + slack (a full disk: drives keep it just above the deleter's floor)."""
  def __init__(self, capacity=100 * GB, slack=50 * GB):
    self.capacity, self.slack = capacity, slack

  def __call__(self, _root):
    return Disk(self.capacity, floor_bytes(self.capacity) + self.slack)


class MapsWorld(World):
  def __init__(self, monkeypatch, params, disk=None, drives=40 * GB, runner="inline"):
    self.params = params
    # "inline": the server check at once (deterministic); "default": OfflineMaps' own runner (production: a thread)
    self.runner = (lambda work: work()) if runner == "inline" else runner
    self.disk_size = disk or FakeDisk()
    self.drives = drives
    super().__init__(monkeypatch)

  def start_manager(self):
    super().start_manager()
    self.tile_server = FakeServer(self)
    extra = {} if self.runner == "default" else {"runner": self.runner}
    self.maps = OfflineMaps(self.params, self.root, self.updater, self.tile_server, disk=self.disk_size, drives=lambda: self.drives,
                            now=lambda: 1_762_800_000.0 + self.clock.t, monotonic=lambda: self.clock.t, **extra)

  # ---- settings
  def select(self, *ids):
    self.params.put("OsmSelectedLocations", [entry_from_id(i) for i in ids], block=True)

  def selected(self):
    return [e["code"] if e["kind"] == "nation" else f"US:{e['code']}" for e in self.params.get("OsmSelectedLocations") or []]

  def request(self, value):
    self.params.put("OsmDownloadRequest", value, block=True)

  def delete(self, value):
    self.params.put("OsmDeleteLocation", value, block=True)

  def status(self):
    return self.params.get("OsmLocationStatus")

  def entry(self, location_id):
    return self.status()["entries"][location_id]

  # ---- disk and server content
  def seed_locations(self, *ids, version="old"):
    tiles = union_tiles(ids)
    self.seed({t: area_files(t, version) for t in tiles})
    return tiles

  def serve_locations(self, *ids, version="new"):
    tiles = union_tiles(ids)
    self.serve({t: area_files(t, version) for t in tiles})
    return tiles

  def sent_codes(self):
    return [(n or s)[0] for n, s in self.sent]

  # ---- driving
  def idle(self):
    u = self.updater
    return (u.state == ou.IDLE and not u.busy and self.request_state() != "pending" and not self.maps.pending_deletes
            and not self.params.get("OsmDownloadRequest") and not self.params.get("OsmDeleteLocation"))

  def run(self, events_per_poll=3, max_polls=20000, until=None, before_poll=None):
    for _ in range(max_polls):
      if before_poll is not None:
        before_poll(self)
      self.maps.step()
      if until is not None and until(self):
        return True
      if until is None and self.idle():
        return True
      self.mapd_steps(events_per_poll)
    return False


def tiles_of(location_id):
  return entry_tiles(location_id)
