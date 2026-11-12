"""Several map locations at once (E9): selection, requests, deletes, gates and status, against the fake mapd.

DESIGN.md rev 4 section 7 with the owner's answers (11 Nov 2026): any unmetered network, the deleter may always act, a
3 h drive buffer, the selection in the sunnylink backup, other map areas shown with DELETE."""
import os
import threading
import time

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.mapd import offline_maps as om
from openpilot.sunnypilot.mapd.offline_regions import tile_dir, union_tiles
from openpilot.sunnypilot.mapd.tests.fake_mapd import NET_ERROR, NOT_FOUND, Cut, HttpStatus
from openpilot.sunnypilot.mapd.tests.maps_world import GB, FakeDisk, MapsWorld, tiles_of
from openpilot.sunnypilot.mapd.tests.offline_world import area_files

CH, AT, DE, IT = tiles_of("CH"), tiles_of("AT"), tiles_of("DE"), tiles_of("IT")
SHARED_CH_AT = [(46, 8), (46, 10)]
FAR = (10, 10)  # a tile none of the test locations uses


def on_disk(w, tile):
  return os.path.isdir(tile_dir(w.root, "offline", tile))


class MapsCase(OpenpilotTestCase):
  def world(self, monkeypatch, **kw):
    w = MapsWorld(monkeypatch, Params(), **kw)
    self.addCleanup(w.close)
    return w


class TestRequests(MapsCase):
  def test_adding_a_country_downloads_only_that_one(self, monkeypatch):
    w = self.world(monkeypatch)
    w.seed_locations("CH")
    w.select("CH")
    w.serve_locations("CH", "AT")
    assert w.run()
    w.request("AT")
    assert w.run()
    assert w.sent_codes() == ["AT"] and w.selected() == ["CH", "AT"]
    assert sorted(set(w.mapd.gets)) == sorted(AT)
    assert w.entry("AT")["state"] == "ok" and w.entry("CH")["state"] == "ok"
    assert w.entry("AT")["shared_with"] == ["Switzerland"] and w.entry("AT")["shared_tiles"] == 2

  def test_any_country_of_the_binary_list_can_be_added(self, monkeypatch):
    w = self.world(monkeypatch)
    w.serve_locations("SI", "ES")
    w.request("SI,ES,XX")  # XX: not in mapd v1.12.0's list, ignored
    assert w.run()
    assert w.sent_codes() == ["SI", "ES"] and w.selected() == ["SI", "ES"]
    assert w.status()["entries"]["ES"]["title"] == "Spain"

  def test_update_all_five_skips_the_covered_country(self, monkeypatch):
    w = self.world(monkeypatch)
    w.seed_locations("CH")
    w.select("CH", "AT", "DE", "IT", "FR")
    w.serve_locations("CH", "AT", "DE", "IT", "FR")
    w.request("all")
    assert w.run()
    assert w.sent_codes() == ["AT", "DE", "IT", "FR"]  # Switzerland lies inside Italy's tiles
    assert len(w.mapd.gets) == 122 and len(set(w.mapd.gets)) == 95
    ch = w.entry("CH")
    assert ch["state"] == "ok" and ch["tiles_refreshed"] == 6 and ch["last"]["reason"] == "covered by other locations"
    assert all(w.entry(i)["state"] == "ok" for i in ("AT", "DE", "IT", "FR"))
    for t in CH:
      assert w.disk(t) == area_files(t, "new")

  def test_legacy_database_update_counts_as_update_all(self, monkeypatch):
    w = self.world(monkeypatch)
    w.select("CH", "AT")
    w.serve_locations("CH", "AT")
    w.params.put_bool("OsmDbUpdatesCheck", True, block=True)
    assert w.run()
    assert w.sent_codes() == ["CH", "AT"] and not w.params.get_bool("OsmDbUpdatesCheck")

  def test_a_covered_country_is_sent_when_its_cover_leaves_the_queue(self, monkeypatch):
    w = self.world(monkeypatch)
    w.select("CH", "IT")
    w.serve_locations("CH", "IT")
    w.offroad = False
    w.request("all")
    w.run(until=lambda w: w.updater.waiting == "onroad")
    w.delete("IT")
    w.run(until=lambda w: not w.params.get("OsmDeleteLocation"))
    w.offroad = True
    assert w.run()
    assert w.sent_codes() == ["CH"] and w.entry("CH")["state"] == "ok"

  def test_progress_names_the_country_and_its_place_in_the_run(self, monkeypatch):
    w = self.world(monkeypatch)
    w.select("CH", "AT")
    w.serve_locations("CH", "AT")
    w.request("all")
    w.run(until=lambda w: w.updater.current == ("nation", "AT"))
    w.maps.step()
    run = w.status()["run"]
    assert run["active"] and run["current"] == "AT" and run["index"] == 2 and run["count"] == 2 and run["tiles_total"] == 10
    assert w.entry("AT")["state"] == "downloading"
    assert w.run()
    assert not w.status()["run"]["active"]


class TestDeletes(MapsCase):
  def test_delete_keeps_tiles_another_country_uses(self, monkeypatch):
    w = self.world(monkeypatch)
    w.seed_locations("CH", "AT")
    w.select("CH", "AT")
    assert w.run()
    assert w.entry("AT")["exclusive_bytes"] == sum(len(b) for t in AT if t not in CH for b in area_files(t, "old").values())
    w.delete("AT")
    assert w.run()
    assert w.selected() == ["CH"]
    assert all(on_disk(w, t) for t in CH)
    assert not any(on_disk(w, t) for t in AT if t not in CH)
    assert w.entry("CH")["state"] == "ok" and w.status()["other"]["tiles"] == 0

  def test_delete_of_the_running_country_waits_for_its_end(self, monkeypatch):
    w = self.world(monkeypatch)
    w.seed_locations("CH")
    w.select("CH")
    w.serve_locations("CH", "AT")
    w.request("AT")
    w.run(until=lambda w: w.updater.current == ("nation", "AT") and w.mapd.gets)
    w.delete("AT")
    w.maps.step()
    assert w.selected() == ["CH"] and w.maps.pending_deletes == ["AT"]
    assert w.updater.current == ("nation", "AT")  # the binary has no cancel: the location runs to its end
    assert w.run()
    assert not any(on_disk(w, t) for t in AT if t not in CH)
    assert all(w.disk(t) in (area_files(t, "old"), area_files(t, "new")) for t in CH)
    assert w.leftovers() == []

  def test_delete_of_a_queued_country_is_never_sent(self, monkeypatch):
    w = self.world(monkeypatch)
    w.select("CH", "DE")
    w.serve_locations("CH", "DE")
    w.request("all")
    w.run(until=lambda w: w.updater.current == ("nation", "CH"))
    w.delete("DE")
    assert w.run()
    assert w.sent_codes() == ["CH"] and w.selected() == ["CH"]
    assert not any(on_disk(w, t) for t in DE if t not in CH)

  def test_delete_of_an_idle_country_is_done_at_once(self, monkeypatch):
    w = self.world(monkeypatch)
    w.seed_locations("CH", "AT")
    w.select("CH", "AT")
    assert w.run()
    w.delete("CH")
    w.maps.step()
    assert not any(on_disk(w, t) for t in CH if t not in AT) and all(on_disk(w, t) for t in SHARED_CH_AT)

  def test_other_map_areas_are_shown_and_deleted_alone(self, monkeypatch):
    w = self.world(monkeypatch)
    w.seed_locations("CH")
    w.seed({FAR: area_files(FAR, "old")})
    w.select("CH")
    assert w.run()
    other = w.status()["other"]
    assert other["tiles"] == 1 and other["bytes"] == sum(len(b) for b in area_files(FAR, "old").values())
    w.delete("OTHER")
    assert w.run()
    assert not on_disk(w, FAR) and all(on_disk(w, t) for t in CH)
    assert w.status()["other"] == {"tiles": 0, "bytes": 0}

  def test_migration_with_tiles_of_an_earlier_choice(self, monkeypatch):
    w = self.world(monkeypatch)
    w.seed_locations("CH", "AT")  # Austria chosen once, then Switzerland: Austria's own tiles are left over
    w.params.put("OsmLocationName", "CH", block=True)
    assert w.run()
    assert w.selected() == ["CH"]
    assert w.status()["other"]["tiles"] == len([t for t in AT if t not in CH])

  def test_delete_all_during_a_run_waits_and_empties_the_selection(self, monkeypatch):
    w = self.world(monkeypatch)
    w.select("CH", "DE")
    w.serve_locations("CH", "DE")
    w.request("all")
    w.run(until=lambda w: w.updater.current == ("nation", "CH"))
    cleared = []

    def clear_step(w):  # mapd_manager.main_thread's Mapd_ClearCache step
      if not cleared and w.maps.ready_for_clear():
        for sub in ("offline", "offline_prev", "tmp"):
          om._rmtree(os.path.join(w.root, sub))
        w.maps.cleared()
        cleared.append(1)
    assert w.run(before_poll=clear_step)
    clear_step(w)
    assert cleared and w.sent_codes() == ["CH"] and w.selected() == []
    assert not w.params.get("OsmLocationName") and not w.params.get_bool("OsmLocal")


class TestGates(MapsCase):
  def test_network_policy_any_unmetered(self):
    allowed = {("wifi", False), ("ethernet", False), ("cell2G", False), ("cell3G", False), ("cell4G", False), ("cell5G", False)}
    for kind in ("none", "wifi", "ethernet", "cell2G", "cell3G", "cell4G", "cell5G"):
      for metered in (False, True):
        assert om.network_allows((kind, metered)) == ((kind, metered) in allowed), (kind, metered)
    assert not om.network_allows(None)

  def test_metered_cellular_waits_and_unmetered_cellular_downloads(self, monkeypatch):
    w = self.world(monkeypatch)
    w.serve_locations("CH")
    w.network = ("cell4G", True)  # comma's SIM, or GsmMetered on
    w.request("CH")
    w.run(max_polls=30, until=lambda w: False)
    assert w.sent == [] and w.status()["run"]["waiting"] == om.WAIT_NETWORK and w.tile_server.heads == []
    w.network = ("cell4G", False)  # the owner's SIM with GsmMetered off, also when roaming
    assert w.run()
    assert w.sent_codes() == ["CH"]

  def test_network_change_pauses_at_the_country_boundary(self, monkeypatch):
    w = self.world(monkeypatch)
    w.select("CH", "DE")
    w.serve_locations("CH", "DE")
    w.request("all")
    w.run(until=lambda w: w.updater.current == ("nation", "CH"))
    w.network = ("wifi", True)  # a Wi-Fi marked metered
    w.run(until=lambda w: w.updater.waiting == om.WAIT_NETWORK)
    assert w.sent_codes() == ["CH"] and w.entry("CH")["state"] == "ok" and w.entry("DE")["state"] == "queued"
    w.network = ("wifi", False)
    assert w.run()
    assert w.sent_codes() == ["CH", "DE"]

  def test_a_drive_mid_run_lets_the_current_country_finish(self, monkeypatch):
    w = self.world(monkeypatch)
    w.select("CH", "DE")
    w.serve_locations("CH", "DE")
    w.request("all")
    w.run(until=lambda w: w.updater.current == ("nation", "CH"))
    w.offroad = False
    w.run(until=lambda w: w.updater.waiting == "onroad")
    assert w.sent_codes() == ["CH"] and w.entry("CH")["state"] == "ok"
    w.offroad = True
    assert w.run()
    assert w.sent_codes() == ["CH", "DE"]

  def test_storage_guard_refuses_and_says_why(self, monkeypatch):
    w = self.world(monkeypatch, disk=FakeDisk(capacity=50 * GB, slack=1 * GB), drives=0)
    w.serve_locations("CH")
    w.request("CH")
    assert w.run()
    assert w.sent == [] and w.status()["message"].startswith(om.STOP_STORAGE)
    assert w.entry("CH")["state"] == "not_downloaded" and not w.status()["run"]["active"]

  def test_remote_writes_cannot_bypass_the_guard(self, monkeypatch):
    w = self.world(monkeypatch, disk=FakeDisk(capacity=50 * GB, slack=1 * GB), drives=0)
    w.serve_locations("CH", "AT")
    w.select("CH", "AT")
    w.params.put("OsmLocationStatus", {"entries": {"CH": {"last": {"result": "ok", "refreshed": 6, "expected": 6}}}}, block=True)
    w.params.put_bool("OsmDbUpdatesCheck", True, block=True)  # every key sunnylink may write
    w.request("all")
    assert w.run()
    w.params.put("OsmLocationName", "DE", block=True)
    assert w.run()
    assert w.sent == [] and w.mapd.gets == []
    assert w.entry("CH")["state"] == "not_downloaded"  # a written status never claims tiles that are not there
    assert w.params.get("OsmLocationName") == "CH"  # the old key mirrors the list

  def test_full_disk_passes_when_the_buffer_holds(self, monkeypatch):
    w = self.world(monkeypatch, disk=FakeDisk(capacity=100 * GB, slack=100), drives=80 * GB)
    w.serve_locations("CH")
    w.request("CH")
    assert w.run()
    assert w.sent_codes() == ["CH"] and w.entry("CH")["state"] == "ok"

  def test_probe_skips_tiles_without_data(self, monkeypatch):
    w = self.world(monkeypatch)
    w.serve_locations("CH")
    w.serve({CH[0]: NOT_FOUND, CH[1]: NOT_FOUND})
    w.request("CH")
    assert w.run()
    assert w.tile_server.probes == [CH[2]] and w.sent_codes() == ["CH"]  # the HEAD requests found the two 404s
    e = w.entry("CH")
    assert e["state"] == "ok" and e["tiles_expected"] == 4 and e["tiles_present"] == 4

  def test_probe_moves_on_after_a_404_the_head_request_did_not_see(self, monkeypatch):
    w = self.world(monkeypatch)
    w.serve_locations("CH")
    w.serve({CH[0]: NOT_FOUND, CH[1]: NOT_FOUND})
    head = w.tile_server.head
    monkeypatch.setattr(w.tile_server, "head", lambda t: None if t in CH[:2] else head(t))
    w.request("CH")
    assert w.run()
    assert w.tile_server.probes == [CH[0], CH[1], CH[2]] and w.sent_codes() == ["CH"]
    assert w.maps.cache.no_data(CH[0]) and w.entry("CH")["tiles_expected"] == 4

  def test_a_location_without_any_data_passes_with_nothing_downloaded(self, monkeypatch):
    w = self.world(monkeypatch)
    w.serve(dict.fromkeys(CH, NOT_FOUND))
    w.request("CH")
    assert w.run()
    assert w.sent_codes() == ["CH"] and w.entry("CH")["tiles_expected"] == 0 and w.entry("CH")["state"] == "ok"

  def test_an_error_page_is_never_sent(self, monkeypatch):
    self._not_gzip(monkeypatch, b"<html>error</html>")

  def test_an_empty_reply_is_never_sent(self, monkeypatch):
    self._not_gzip(monkeypatch, b"")

  def _not_gzip(self, monkeypatch, body):
    w = self.world(monkeypatch)
    w.serve_locations("CH", "AT")
    w.serve({CH[0]: body})
    w.select("CH", "AT")
    w.request("all")
    assert w.run()
    assert w.sent_codes() == ["AT"] and not on_disk(w, CH[0])
    ch = w.entry("CH")  # only the 2 tiles Austria shares are on disk
    assert ch["state"] == "partial" and ch["tiles_present"] == 2 and ch["last"]["reason"] == om.SKIP_NOT_GZIP

  def test_no_answer_from_the_server_waits_and_retries(self, monkeypatch):
    w = self.world(monkeypatch)
    w.serve(dict.fromkeys(CH, NET_ERROR))
    w.request("CH")
    w.run(max_polls=40, until=lambda w: False)
    assert w.sent == [] and w.status()["run"]["waiting"] == om.WAIT_SERVER
    assert len(w.tile_server.heads) <= 2 and w.tile_server.probes == []  # one try per minute, not per second
    w.serve_locations("CH")
    assert w.run()
    assert w.sent_codes() == ["CH"]


  def test_an_update_waits_while_the_server_does_not_answer_the_probe(self, monkeypatch):
    w = self.world(monkeypatch)
    w.seed_locations("CH")  # an update: every tile on disk, so no HEAD request, only the probe
    w.select("CH")
    w.serve(dict.fromkeys(CH, NET_ERROR))
    w.request("all")
    w.run(max_polls=40, until=lambda w: False)
    assert w.tile_server.heads == [] and len(w.tile_server.probes) <= 2
    assert w.sent == [] and w.status()["run"]["waiting"] == om.WAIT_SERVER and w.status()["run"]["active"]
    w.serve_locations("CH")
    w.clock.t += om.SERVER_RETRY_SECONDS
    assert w.run()
    assert w.sent_codes() == ["CH"] and w.entry("CH")["state"] == "ok"


class TestServerCheckOffTheLoop(MapsCase):
  """S1: the manager loop never waits on the tile server (map speed data keeps flowing)."""
  def test_the_loop_step_returns_while_the_server_hangs(self, monkeypatch):
    w = self.world(monkeypatch, runner="default")  # production wiring: OfflineMaps' default runner
    w.serve_locations("CH")
    hang, entered = threading.Event(), threading.Event()
    head = w.tile_server.head

    def hanging_head(t):
      entered.set()
      hang.wait(30)
      return head(t)
    monkeypatch.setattr(w.tile_server, "head", hanging_head)
    w.request("CH")
    slowest = 0.0
    for i in range(20):
      if i == 10:
        w.offroad = False  # the car starts during the check
      t0 = time.monotonic()
      w.maps.step()
      slowest = max(slowest, time.monotonic() - t0)
      w.mapd_steps(2)
    assert entered.is_set() and slowest < 0.5, slowest
    assert w.sent == [] and w.status()["run"]["waiting"] == "onroad"
    w.offroad = True
    w.maps.step()
    assert w.status()["run"]["waiting"] == om.WAIT_CHECKING
    hang.set()
    deadline = time.monotonic() + 10
    while w.maps._check is not None and w.maps._check["result"] is None and time.monotonic() < deadline:
      time.sleep(0.01)
    assert w.run()
    assert w.sent_codes() == ["CH"] and w.entry("CH")["state"] == "ok"

  def test_an_answer_older_than_five_minutes_is_asked_again(self, monkeypatch):
    jobs = []
    w = self.world(monkeypatch, runner=jobs.append)  # the worker runs only when the test says so
    w.serve_locations("CH")
    assert w.run()
    w.request("CH")
    w.maps.step()
    assert len(jobs) == 1 and w.status()["run"]["waiting"] == om.WAIT_CHECKING
    w.offroad = False  # a drive starts right after the answer arrives
    jobs.pop()()
    w.clock.t += om.CHECK_MAX_AGE_SECONDS + 1
    w.offroad = True
    w.maps.step()  # the answer, now older than five minutes, is dropped
    assert w.sent == [] and len(jobs) == 0
    w.maps.step()  # and a new check starts
    assert len(jobs) == 1
    jobs.pop()()
    assert w.run()
    assert w.sent_codes() == ["CH"]


  def test_a_slow_check_is_not_dropped_as_stale(self, monkeypatch):
    """The age counts from the answer: a check that took longer than five minutes (many tiles, slow server) is used."""
    jobs = []
    w = self.world(monkeypatch, runner=jobs.append)
    w.serve_locations("CH")
    assert w.run()
    w.request("CH")
    w.maps.step()
    w.clock.t += om.CHECK_MAX_AGE_SECONDS + 60
    jobs.pop()()
    assert w.run()
    assert w.sent_codes() == ["CH"] and jobs == []

  def test_a_check_that_never_answers_ends_after_the_timeout(self, monkeypatch):
    jobs = []
    w = self.world(monkeypatch, runner=jobs.append)  # the worker never runs: a hung server
    w.serve_locations("CH")
    assert w.run()
    w.request("CH")
    w.maps.step()
    assert w.status()["run"]["waiting"] == om.WAIT_CHECKING
    w.clock.t += om.CHECK_TIMEOUT_SECONDS + 1
    w.maps.step()
    run = w.status()["run"]
    assert run["waiting"] == om.WAIT_SERVER_SLOW and run["retry_at"] > 0 and w.status()["message"] == om.WAIT_SERVER_SLOW
    w.request("all")  # a retry by hand starts a new check at once
    w.maps.step()
    assert len(jobs) == 2
    jobs.pop()()
    assert w.run()
    assert w.sent_codes() == ["CH"]

  def test_check_progress_is_in_the_status(self, monkeypatch):
    jobs = []
    w = self.world(monkeypatch, runner=jobs.append)
    w.serve_locations("AT")
    assert w.run()
    w.request("AT")
    w.maps.step()
    check = w.status()["run"]["check"]
    assert check == {"location": "AT", "done": 0, "total": 10}
    jobs.pop()()
    assert w.run()
    assert w.status()["run"]["check"] is None


class TestSlowServer(MapsCase):
  """Review S1: a slow server that keeps answering is waited for; sizes of a check that stops are kept for the retry."""
  def _blocking_head(self, monkeypatch, w):
    gate, answered = threading.Semaphore(0), []
    head = w.tile_server.head

    def slow_head(t):
      gate.acquire()
      answered.append(t)
      return head(t)
    monkeypatch.setattr(w.tile_server, "head", slow_head)
    return gate, answered

  def _answer(self, w, gate, answered, seconds):
    """The server answers one more HEAD after `seconds`; returns once the worker has it."""
    n = len(answered)
    w.clock.t += seconds
    gate.release()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
      job = w.maps._check
      if job is None or job.get("cancelled") or len(job["sizes"]) > n or job["result"] is not None:
        break  # answered, or the check has ended (then the test's assertions decide)
      time.sleep(0.002)
    w.maps.step()

  def test_a_slow_server_answering_every_30_s_completes_a_five_country_batch(self, monkeypatch):
    w = self.world(monkeypatch, runner="default")
    five = ["CH", "AT", "DE", "IT", "FR"]
    w.serve_locations(*five)
    gate, answered = self._blocking_head(monkeypatch, w)
    assert w.run()
    w.request(",".join(five))
    w.maps.step()
    total = w.maps._check["total"]
    assert total == len(union_tiles(five))  # nothing on disk: every tile is asked once
    for _ in range(total):
      self._answer(w, gate, answered, 30)
      assert w.status()["run"]["waiting"] in (om.WAIT_CHECKING, ""), len(answered)  # never "too slow" while answers come ("": sent)
    assert 30 * total > om.CHECK_TIMEOUT_SECONDS  # far longer than the limit in all, never 10 min without an answer
    assert w.status()["run"]["waiting"] != om.WAIT_SERVER_SLOW
    gate.release(1000)
    assert w.run()
    assert w.sent_codes() == ["AT", "DE", "IT", "FR"] and len(answered) == total  # CH covered by Italy

  def test_a_check_that_stops_keeps_its_sizes_and_the_retry_continues(self, monkeypatch):
    w = self.world(monkeypatch, runner="default")
    w.serve_locations("AT")
    gate, answered = self._blocking_head(monkeypatch, w)
    assert w.run()
    w.request("AT")
    w.maps.step()
    for _ in range(6):
      self._answer(w, gate, answered, 30)
    w.clock.t += om.CHECK_TIMEOUT_SECONDS + 1  # then 10 min without an answer
    w.maps.step()
    assert w.status()["run"]["waiting"] == om.WAIT_SERVER_SLOW
    assert all(w.maps.cache.download(t) for t in answered)  # the 6 sizes are kept
    w.request("all")  # retry by hand
    w.maps.step()
    assert w.maps._check["total"] == len(AT) - 6  # only the rest is asked
    gate.release(1000)
    assert w.run()
    assert w.sent_codes() == ["AT"] and len(answered) == len(AT) + 1  # the hanging 7th answer arrives late, unused


class TestBatches(MapsCase):
  """Several countries picked at once, added during a download, and checked together."""
  def test_countries_added_during_a_download_extend_the_queue(self, monkeypatch):
    w = self.world(monkeypatch)
    w.serve_locations("CH", "AT", "DE")
    w.request("CH")
    w.run(until=lambda w: w.updater.current == ("nation", "CH"))
    w.request("AT,DE")  # ADD with two ticks while Switzerland downloads
    w.maps.step()
    run = w.status()["run"]
    assert run["count"] == 3 and run["current"] == "CH"
    assert run["queue_tiles_total"] == len(CH) + len(AT) + len(DE)
    assert w.run()
    assert w.sent_codes() == ["CH", "AT", "DE"] and w.selected() == ["CH", "AT", "DE"]

  def test_queue_progress_counts_every_country(self, monkeypatch):
    w = self.world(monkeypatch)
    w.serve_locations("CH", "AT")
    w.request("CH,AT")
    w.run(until=lambda w: w.updater.current == ("nation", "AT"))
    w.maps.step()
    run = w.status()["run"]
    assert run["queue_tiles_total"] == 16 and len(CH) <= run["queue_tiles_done"] < 16

  def test_the_guard_checks_the_whole_queue(self, monkeypatch):
    # room for one country's tiles but not for three: refused before the first is sent, with all names
    w = self.world(monkeypatch)
    w.serve_locations("CH", "AT", "DE")
    sizes = dict.fromkeys(union_tiles(["CH", "AT", "DE"]), GB // 2)  # what the server says: 2 GB per tile on disk (ratio 4)
    monkeypatch.setattr(w.tile_server, "head", lambda t: sizes.get(t, 0))
    w.disk_size.capacity, w.disk_size.slack = 100 * GB, 85 * GB  # maps may use about 42 GB: CH 12 GB fits, all three 86 GB not
    w.drives = 0
    w.request("CH,AT,DE")
    assert w.run()
    assert w.sent == [] and "Switzerland, Austria, Germany" in w.status()["message"]
    w.request("CH")  # alone it fits
    assert w.run()
    assert w.sent_codes() == ["CH"]

  def test_a_deleted_country_being_downloaded_is_shown_as_a_queued_delete(self, monkeypatch):
    w = self.world(monkeypatch)
    w.serve_locations("CH", "AT")
    w.request("CH,AT")
    w.run(until=lambda w: w.updater.current == ("nation", "CH") and w.mapd.gets)
    w.delete("CH")
    w.maps.step()
    run = w.status()["run"]
    assert run["deletes_pending"] == ["CH"] and run["current"] == "CH" and run["count"] == 2
    assert "CH" not in w.status()["entries"]
    assert w.run()
    assert w.sent_codes() == ["CH", "AT"] and w.status()["run"]["deletes_pending"] == []
    assert not any(on_disk(w, t) for t in CH if t not in AT)


class TestServerErrors(MapsCase):
  """S2: a status other than 200 / 206 / 404 is a server error with its code, retried with backoff and by hand."""
  def probes_in(self, w, seconds):
    """One manager step per second of the fake clock (an idle mapd: sleep 1 s, heartbeat)."""
    end = w.clock.t + seconds
    while w.clock.t < end:
      w.maps.step()
      w.mapd_steps(2)
    return len(w.tile_server.probes)

  def test_http_403_is_shown_with_its_code_and_retried_with_backoff(self, monkeypatch):
    w = self.world(monkeypatch)
    w.serve(dict.fromkeys(CH, HttpStatus(403)))
    assert w.run()
    w.request("CH")
    assert self.probes_in(w, 500) == 4  # at 0, 60, 180 and 420 s: 1, 2, 4 min apart
    run = w.status()["run"]
    assert run["active"] and run["waiting"] == "map server error (HTTP 403)" and run["server_error"] == 403
    assert run["retry_at"] > 0 and w.status()["message"] == "map server error (HTTP 403)"
    assert w.sent == [] and w.mapd.gets == []

  def test_backoff_doubles_up_to_an_hour(self, monkeypatch):
    w = self.world(monkeypatch)
    delays = []
    for _ in range(9):
      w.maps._backoff("x")
      delays.append(round(w.maps._retry_at - w.clock.t))
    assert delays == [60, 120, 240, 480, 960, 1920, 3600, 3600, 3600]

  def test_a_request_by_hand_asks_again_at_once(self, monkeypatch):
    w = self.world(monkeypatch)
    w.select("CH")
    w.serve(dict.fromkeys(CH, HttpStatus(403)))
    assert w.run()
    w.request("CH")
    self.probes_in(w, 5)
    assert len(w.tile_server.probes) == 1
    w.serve_locations("CH")  # the server works again
    w.request("all")  # Update all / DOWNLOAD while it waits
    self.probes_in(w, 5)  # well inside the 60 s backoff
    assert len(w.tile_server.probes) == 2 and w.sent_codes() == ["CH"]
    assert w.run()
    assert w.status()["run"]["server_error"] is None and w.status()["message"] == ""

  def test_no_answer_backs_off_too(self, monkeypatch):
    w = self.world(monkeypatch)
    w.seed_locations("CH")
    w.select("CH")
    w.serve(dict.fromkeys(CH, NET_ERROR))
    assert w.run()
    w.request("all")
    assert self.probes_in(w, 500) == 4
    assert w.status()["run"]["waiting"] == om.WAIT_SERVER and w.status()["run"]["server_error"] is None


class TestStatus(MapsCase):
  def test_every_tile_failing_is_failed_never_ok(self, monkeypatch):
    w = self.world(monkeypatch)
    old = w.seed_locations("CH")
    w.select("CH")
    new = {t: area_files(t, "new") for t in CH}
    w.serve(new)
    w.request("CH")
    assert w.run()
    assert w.entry("CH")["state"] == "ok"
    from openpilot.sunnypilot.mapd.tests.fake_mapd import make_tile
    w.serve({t: Cut(make_tile(t, new[t]), 40) for t in CH})  # starts as gzip, then the connection breaks
    w.request("CH")
    assert w.run()
    e = w.entry("CH")
    assert e["state"] == "failed" and e["tiles_present"] == 6 and e["last"]["refreshed"] == 0
    assert all(w.disk(t) == area_files(t, "new") for t in old)

  def test_status_is_judged_from_disk_after_a_restart(self, monkeypatch):
    w = self.world(monkeypatch)
    w.select("CH")
    w.serve_locations("CH")
    w.request("CH")
    assert w.run()
    om._rmtree(tile_dir(w.root, "offline", CH[0]))  # a tile lost
    w.manager_restart()
    assert w.run()
    e = w.entry("CH")
    assert e["state"] == "partial" and e["tiles_present"] == 5 and e["last"]["result"] == "ok"

  def test_a_restored_selection_shows_not_downloaded_and_downloads_nothing(self, monkeypatch):
    w = self.world(monkeypatch)
    w.serve_locations("CH", "AT")
    w.select("CH", "AT")  # as a sunnylink restore writes it
    assert w.run()
    assert w.sent == [] and w.tile_server.heads == []
    assert [w.entry(i)["state"] for i in ("CH", "AT")] == ["not_downloaded", "not_downloaded"]

  def test_tiles_moved_aside_during_an_update_still_count(self, monkeypatch):
    w = self.world(monkeypatch)
    w.seed_locations("CH", "AT")
    w.select("CH", "AT")
    w.serve_locations("AT")
    w.request("AT")
    w.run(until=lambda w: w.updater.current == ("nation", "AT"))
    w.maps._disk_at = -1e9
    w.maps.step()
    assert w.entry("CH")["tiles_present"] == 6 and w.status()["other"]["tiles"] == 0


class TestSelection(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()
    from openpilot.sunnypilot.mapd.offline_selection import Selection
    self.sel = Selection(self.params)

  def test_migration_keeps_the_country_chosen_before(self):
    self.params.put("OsmLocationName", "CH", block=True)
    self.params.put("OsmLocationTitle", "Switzerland", block=True)
    assert self.sel.read() == [{"kind": "nation", "code": "CH", "title": "Switzerland"}]
    assert self.params.get("OsmSelectedLocations") == [{"kind": "nation", "code": "CH", "title": "Switzerland"}]
    assert self.params.get_bool("OsmLocal")
    self.params.put("OsmLocationName", "AT", block=True)  # migration runs once: the list now wins
    assert [e["code"] for e in self.sel.read()] == ["CH"] and self.params.get("OsmLocationName") == "CH"

  def test_migration_of_the_us_choices(self):
    from openpilot.sunnypilot.mapd.offline_selection import migrate
    self.params.put("OsmLocationName", "US", block=True)
    self.params.put("OsmStateName", "CA", block=True)
    assert migrate(self.params) == [{"kind": "state", "code": "CA", "title": "California"}]
    self.params.put("OsmStateName", "All", block=True)
    assert migrate(self.params) == [{"kind": "nation", "code": "US", "title": "United States"}]
    self.params.remove("OsmStateName")
    assert migrate(self.params)[0]["code"] == "US"
    self.params.put("OsmLocationName", "XX", block=True)
    assert migrate(self.params) == []

  def test_state_mirror(self):
    self.sel.write([{"kind": "state", "code": "CA"}, {"kind": "nation", "code": "CH"}])
    assert self.params.get("OsmLocationName") == "US" and self.params.get("OsmStateName") == "CA"
    assert self.params.get("OsmStateTitle") == "California"
    self.sel.write([])
    assert self.params.get("OsmLocationName") is None and self.params.get("OsmStateName", return_default=True) == "All"
    assert not self.params.get_bool("OsmLocal")

  def test_written_values_are_validated(self):
    self.params.put("OsmSelectedLocations", [{"kind": "nation", "code": "CH", "title": "fake"}, {"kind": "nation", "code": "XX"},
                                             {"kind": "state", "code": "CH"}, "AT", {"kind": "nation", "code": "CH"},
                                             {"kind": "nation", "code": "AT"}], block=True)
    clean = [{"kind": "nation", "code": "CH", "title": "Switzerland"}, {"kind": "nation", "code": "AT", "title": "Austria"}]
    assert self.sel.read() == clean and self.params.get("OsmSelectedLocations") == clean
    with open(self.params.get_param_path("OsmSelectedLocations"), "w") as f:
      f.write("{not json")
    assert self.sel.read() == clean and self.params.get("OsmSelectedLocations") == clean  # the last valid list comes back
    self.params.put("OsmSelectedLocations", {"kind": "nation"}, block=True)
    assert self.sel.read() == clean

  def test_migration_runs_only_before_the_list_was_ever_written(self):
    """S3: the marker is the key's own file; an unreadable list after a restart is not rebuilt from the mirror."""
    from openpilot.sunnypilot.mapd.offline_selection import Selection
    self.params.put("OsmLocationName", "CH", block=True)
    assert not os.path.exists(self.params.get_param_path("OsmSelectedLocations"))
    assert [e["code"] for e in self.sel.read()] == ["CH"]  # first start: migrated
    assert os.path.exists(self.params.get_param_path("OsmSelectedLocations"))
    self.sel.write([])  # e.g. every country deleted: stays empty, no second migration
    assert Selection(self.params).read() == [] and self.params.get("OsmLocationName") is None

  def test_unreadable_list_after_a_restart_comes_from_the_last_status(self):
    from openpilot.sunnypilot.mapd.offline_selection import Selection
    self.sel.write([{"kind": "nation", "code": "CH"}, {"kind": "nation", "code": "AT"}, {"kind": "state", "code": "CA"}])
    self.params.put("OsmLocationStatus", {"order": ["CH", "AT", "US:CA", "XX", 7]}, block=True)
    with open(self.params.get_param_path("OsmSelectedLocations"), "w") as f:
      f.write("garbage")
    assert self.params.get("OsmLocationName") == "CH"  # the mirror holds one entry only
    restarted = Selection(self.params)
    assert [e["code"] for e in restarted.read()] == ["CH", "AT", "CA"]
    assert [e["code"] for e in self.params.get("OsmSelectedLocations")] == ["CH", "AT", "CA"]

  def test_unreadable_list_without_any_status_falls_back_to_the_mirror(self):
    from openpilot.sunnypilot.mapd.offline_selection import Selection
    self.sel.write([{"kind": "nation", "code": "AT"}, {"kind": "nation", "code": "CH"}])
    with open(self.params.get_param_path("OsmSelectedLocations"), "w") as f:
      f.write("garbage")
    assert [e["code"] for e in Selection(self.params).read()] == ["AT"]

def test_union_of_all_five_has_95_tiles():
  assert len(union_tiles(["CH", "AT", "DE", "IT", "FR"])) == 95
