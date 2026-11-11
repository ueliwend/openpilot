"""Test harness: the offline update manager and the fake mapd on one disk, with crashes between any two steps."""
import json
import os
import random
import shutil
import tempfile

import openpilot.sunnypilot.mapd.offline_update as ou
from openpilot.sunnypilot.mapd.offline_regions import tile_dir, tile_order
from openpilot.sunnypilot.mapd.tests.fake_mapd import FakeMapd, make_tile, write_param


class SimulatedCrash(BaseException):
  """The process stops here (BaseException, so the manager's own error handling does not catch it)."""


class Clock:
  def __init__(self):
    self.t = 1000.0


class FileLink:
  """ParamsLink's interface on plain files (the manager-level test covers ParamsLink itself)."""

  def __init__(self, world):
    self.world = world
    self.request_path = world.request_path

  def send(self, nations, states):
    self.world.action("send")
    write_param(self.request_path, json.dumps({"nations": nations, "states": states}).encode())
    self.world.sent.append((nations, states))

  def clear_request(self):
    self.world.action("clear_request")
    if os.path.exists(self.request_path):
      os.remove(self.request_path)

  def remove_progress(self):
    self.world.action("remove_progress")
    if os.path.exists(self.world.progress_path):
      os.remove(self.world.progress_path)

  def downloaded_files(self):
    try:
      with open(self.world.progress_path) as f:
        return int(json.load(f)["downloaded_files"])
    except (OSError, ValueError, KeyError, TypeError):
      return None

  def is_offroad(self):
    return self.world.offroad


class World:
  def __init__(self, monkeypatch, root=None, request_path=None, progress_path=None, link=None):
    self.tmp = tempfile.mkdtemp(prefix="osm_world_")
    self.shm = tempfile.mkdtemp(prefix="osm_world_", dir="/dev/shm")
    self.root = root or os.path.join(self.tmp, "osm")
    self.request_path = request_path or os.path.join(self.shm, "OSMDownloadLocations")
    self.progress_path = progress_path or os.path.join(self.tmp, "params", "OSMDownloadProgress")
    os.makedirs(self.root, exist_ok=True)
    self.clock = Clock()
    self.server: dict = {}
    self.offroad = True
    self.mapd_alive = True
    self.sent: list = []
    self.events: list = []
    self.count = 0
    self.crash_at: int | None = None
    self.link = link or FileLink(self)
    self._patch(monkeypatch)
    self.start_mapd()
    self.start_manager()

  def close(self):
    shutil.rmtree(self.tmp, ignore_errors=True)
    shutil.rmtree(self.shm, ignore_errors=True)

  # ---- crash points
  def action(self, label):
    self.count += 1
    if self.crash_at is not None and self.count == self.crash_at:
      self.crash_at = None
      raise SimulatedCrash(label)

  def _patch(self, monkeypatch):
    world = self
    rename, remove, write_json = ou._rename, ou._remove, ou._write_json_atomic

    def crash_rename(src, dst):
      world.action(("rename", src))
      rename(src, dst)

    def crash_rmtree(path):
      if not os.path.isdir(path):
        return
      try:
        world.action(("rmtree", path))
      except SimulatedCrash:
        for d, _, files in os.walk(path):  # a crash in the middle of a delete: some files are gone already
          if files:
            os.remove(os.path.join(d, files[0]))
            break
        raise
      shutil.rmtree(path)

    def crash_remove(path):
      world.action(("remove", path))
      remove(path)

    def crash_write_json(path, data):
      world.action(("write_json", path))
      write_json(path, data)

    monkeypatch.setattr(ou, "_rename", crash_rename)
    monkeypatch.setattr(ou, "_rmtree", crash_rmtree)
    monkeypatch.setattr(ou, "_remove", crash_remove)
    monkeypatch.setattr(ou, "_write_json_atomic", crash_write_json)

  # ---- processes
  def start_mapd(self):
    self.mapd = FakeMapd(self.root, self.request_path, self.progress_path, self.server, self.clock, chunk=256)
    self.gen = self.mapd.run()
    self.mapd_step()  # start-up reset

  def start_manager(self):
    self.updater = ou.OfflineMapUpdater(self.root, self.link, monotonic=lambda: self.clock.t, sleep=self._sleep)

  def power_cut(self):
    """Reboot: /dev/shm is lost, both processes start again."""
    for p in (self.request_path,):
      if os.path.exists(p):
        os.remove(p)
    self.start_mapd()
    self.start_manager()

  def manager_restart(self):
    self.start_manager()

  def mapd_step(self):
    if not self.mapd_alive:
      return None
    event = next(self.gen)
    self.events.append(event)
    self.action(("mapd", event))
    return event

  def _sleep(self, seconds):
    self.clock.t += seconds
    for _ in range(4):  # an idle mapd: sleep, check, heartbeat
      if self.mapd_step() in (("heartbeat",), None):
        return

  def mapd_steps(self, n):
    for _ in range(n):
      self.mapd_step()

  # ---- disk content
  def seed(self, tiles: dict):
    """Old map data on disk: {tile: {name: bytes}}."""
    for tile, files in tiles.items():
      d = tile_dir(self.root, "offline", tile)
      os.makedirs(d, exist_ok=True)
      for name, data in files.items():
        with open(os.path.join(d, name), "wb") as f:
          f.write(data)

  def serve(self, tiles: dict):
    for tile, files in tiles.items():
      self.server[tile] = make_tile(tile, files) if isinstance(files, dict) else files

  def disk(self, tile) -> dict | None:
    d = tile_dir(self.root, "offline", tile)
    if not os.path.isdir(d):
      return None
    out = {}
    for name in os.listdir(d):
      with open(os.path.join(d, name), "rb") as f:
        out[name] = f.read()
    return out

  def leftovers(self) -> list[str]:
    return [p for p in (ou.PREV, ou.TMP) if os.path.exists(os.path.join(self.root, p))]

  def request_state(self) -> str:
    return self.updater._request_state()

  # ---- driving
  def update(self, code="CH", kind="nation"):
    self.updater.request([code] if kind == "nation" else [], [code] if kind == "state" else [])

  def run(self, events_per_poll=3, max_polls=3000, until=None):
    """Alternate manager polls (1 Hz) and mapd steps until the manager is idle with no request (or `until`)."""
    for _ in range(max_polls):
      self.updater.step()
      if until is not None and until(self):
        return True
      if until is None and self.updater.state == ou.IDLE and not self.updater.queue and self.request_state() != "pending":
        return True
      self.mapd_steps(events_per_poll)
    return False


def ch_tiles():
  return tile_order("nation", "CH")


def area_files(tile, version: str, n: int = 3, size: int = 700) -> dict[str, bytes]:
  """Map area files of a tile; version "old" files are longer than version "new" ones (the case that corrupts)."""
  length = size if version == "old" else size // 2
  rnd = random.Random(f"{version}{tile}")  # incompressible, so a tile takes several download chunks
  return {f"{tile[0]}.{i:02d}_{tile[1]}.area": version.encode() + rnd.randbytes(length - len(version)) for i in range(n)}
