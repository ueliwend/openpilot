import math
import threading
import time
from unittest import mock

import pytest

import openpilot.system.hardware.stats_reader as stats_reader
from openpilot.system.hardware.stats_reader import CallTimer, NetworkContext, PeriodicReader, Sample, wait_first_samples


class FakeClock:
  def __init__(self, t=1000.):
    self.t = t

  def monotonic(self):
    return self.t

  def sleep(self, dt):
    self.t += dt


@pytest.fixture
def clock(monkeypatch):
  c = FakeClock()
  monkeypatch.setattr(stats_reader, "time", c)
  return c


@pytest.fixture
def log(monkeypatch):
  fake = mock.MagicMock()
  monkeypatch.setattr(stats_reader, "cloudlog", fake)
  return fake


def events(log, name):
  return [c.kwargs for c in log.event.call_args_list if c.args[0] == name]


class TestNetworkContext:
  def test_tracks_previous_type_and_change_time(self, clock):
    net = NetworkContext()
    assert net.as_dict() == {'type': None, 'prev_type': None, 'changed_s_ago': None}
    net.update("none")
    clock.sleep(30.)
    net.update("none")  # unchanged: keeps the change time
    net.update("wifi")
    clock.sleep(1.5)
    assert net.as_dict() == {'type': 'wifi', 'prev_type': 'none', 'changed_s_ago': 1.5}


class TestCallTimer:
  def test_fast_call_not_logged(self, clock, log):
    timer = CallTimer("stats")
    assert timer.call("fast", lambda x: x + 1, 41) == 42
    assert timer.call("at_threshold", clock.sleep, stats_reader.SLOW_CALL_THRESHOLD) is None  # strict threshold
    log.event.assert_not_called()

  def test_slow_call_logged_with_context(self, clock, log):
    net = NetworkContext()
    net.update("none")
    timer = CallTimer("hw_state", net)

    def slow_get_network_type():
      clock.sleep(4.25)
      net.update("wifi")
      return 1

    assert timer.call("get_network_type", slow_get_network_type) == 1
    (evt,) = events(log, "hardwared slow call")
    assert evt['thread'] == "hw_state"
    assert evt['call'] == "get_network_type"
    assert evt['duration_ms'] == 4250
    assert evt['network'] == {'type': 'wifi', 'prev_type': 'none', 'changed_s_ago': 0.0}
    assert evt['suppressed'] == {}
    assert evt['error']

  def test_exception_passes_through_and_is_timed(self, clock, log):
    timer = CallTimer("stats")

    def fail():
      clock.sleep(1.)
      raise OSError("EIO")

    with pytest.raises(OSError):
      timer.call("statvfs", fail)
    assert [e['call'] for e in events(log, "hardwared slow call")] == ["statvfs"]
    assert timer.running() is None

  def test_rate_limit_summarises_suppressed_calls(self, clock, log):
    timer = CallTimer("stats")
    timer.call("a", clock.sleep, 1.)
    timer.call("b", clock.sleep, 2.)
    timer.call("b", clock.sleep, 0.75)
    timer.call("a", clock.sleep, 0.6)
    assert len(events(log, "hardwared slow call")) == 1
    clock.sleep(10.)
    timer.call("c", clock.sleep, 0.7)
    first, second = events(log, "hardwared slow call")
    assert first['call'] == "a" and first['suppressed'] == {}
    assert second['call'] == "c" and second['suppressed'] == {'b': [2, 2000], 'a': [1, 600]}

  def test_logging_failure_never_raises(self, clock, log):
    log.event.side_effect = RuntimeError("log broken")
    timer = CallTimer("stats")
    assert timer.call("slow", lambda: clock.sleep(1.) or "ok") == "ok"

  def test_running_names_the_call_in_progress(self, clock, log):
    timer = CallTimer("thermal")
    seen = []

    def stuck():
      clock.sleep(3.)
      seen.append(timer.running())

    timer.call("thermal:cpu0-gold-usr", stuck)
    assert seen == [{'call': "thermal:cpu0-gold-usr", 'running_ms': 3000}]
    assert timer.running() is None


class TestPeriodicReaderFreshness:
  def test_stall_tolerance_equals_the_limit(self, log):
    # a sample started at t=100 (interval 0.5): the next one is due at 100.5, so a read stuck from then on may stall
    # for exactly the limit (3 s here) and the reader turns stale right after 103.5
    reader = PeriodicReader("thermal", lambda timer: None, 0.5, 3.)
    assert reader.age(100.) == math.inf and reader.overdue(100.) == math.inf
    assert not reader.fresh(100.)  # no sample yet
    reader.latest = Sample(100., {})
    assert reader.overdue(100.5) == 0.
    assert reader.fresh(103.5)
    assert reader.overdue(103.5) == 3. and reader.age(103.5) == 3.5  # data age = limit + interval
    assert not reader.fresh(103.5 + 1. / 1024)

  def test_stale_logged_with_stuck_call_and_recovery(self, log):
    net = NetworkContext()
    reader = PeriodicReader("thermal", lambda timer: None, 0.5, 5., net)
    reader.latest = Sample(100., {})
    reader.timer.in_progress = ("thermal:ddr-usr", time.monotonic() - 2.)
    t0 = 100. + 0.5 + 5. + 0.125
    # stale for 25 s: logged when it turns stale, then once per LOG_INTERVAL
    for i in range(51):
      assert not reader.fresh(t0 + i * 0.5)
    stale = events(log, "hardwared stale readings")
    assert len(stale) == 3
    assert stale[0]['reader'] == "thermal"
    assert stale[0]['stalled_ms'] == 5125 and stale[0]['age_ms'] == 5625 and stale[0]['limit_ms'] == 5000
    assert stale[0]['running']['call'] == "thermal:ddr-usr"
    assert stale[0]['running']['running_ms'] >= 2000
    assert stale[0]['thread_alive'] is False  # never started
    assert stale[0]['network'] == net.as_dict()
    reader.latest = Sample(t0 + 25., {})
    assert reader.fresh(t0 + 25.5)
    (recovered,) = events(log, "hardwared readings fresh")
    assert recovered['reader'] == "thermal" and recovered['stale_ms'] == 25500

  def test_no_sample_logged_without_age(self, log):
    reader = PeriodicReader("stats", lambda timer: None, 0.5, 30.)
    assert not reader.fresh(10.)
    (evt,) = events(log, "hardwared stale readings")
    assert evt['age_ms'] is None and evt['stalled_ms'] is None and evt['running'] is None

  def test_logging_failure_never_raises(self, log):
    log.event.side_effect = RuntimeError("log broken")
    reader = PeriodicReader("stats", lambda timer: None, 0.5, 30.)
    assert not reader.fresh(10.)
    reader.latest = Sample(10., {})
    assert reader.fresh(10.)


class TestWaitFirstSamples:
  def test_returns_when_all_have_a_sample(self, log):
    readers = [PeriodicReader(n, lambda timer: None, 0.5, 5.) for n in ("a", "b")]
    for r in readers:
      r.first_sample.set()
    t = time.monotonic()
    wait_first_samples(readers, 5., threading.Event())
    assert time.monotonic() - t < 0.1

  def test_times_out(self, log):
    readers = [PeriodicReader("a", lambda timer: None, 0.5, 5.)]
    t = time.monotonic()
    wait_first_samples(readers, 0.3, threading.Event())
    assert 0.25 < time.monotonic() - t < 1.

  def test_honours_shutdown(self, log):
    readers = [PeriodicReader("a", lambda timer: None, 0.5, 5.)]
    end_event = threading.Event()
    threading.Timer(0.2, end_event.set).start()
    t = time.monotonic()
    wait_first_samples(readers, 5., end_event)
    assert time.monotonic() - t < 0.6


class TestPeriodicReaderThread:
  def run_reader(self, read, interval=0.05):
    end_event = threading.Event()
    reader = PeriodicReader("test", read, interval, 5.)
    reader.start(end_event)
    return reader, end_event

  def test_samples_and_stops(self, log):
    n = iter(range(1000))
    reader, end_event = self.run_reader(lambda timer: timer.call("next", next, n))
    assert reader.first_sample.wait(2.)
    time.sleep(0.3)
    assert reader.latest.data >= 2
    assert reader.age(time.monotonic()) < 0.5
    end_event.set()
    reader.thread.join(2.)
    assert not reader.thread.is_alive()

  def test_blocked_read_keeps_last_sample(self, log):
    gate = threading.Event()
    gate.set()
    n = iter(range(1000))

    def read(timer):
      gate.wait()
      return next(n)

    reader, end_event = self.run_reader(read)
    assert reader.first_sample.wait(2.)
    gate.clear()
    time.sleep(0.2)
    stuck = reader.latest
    time.sleep(0.5)
    assert reader.latest is stuck
    assert reader.age(time.monotonic()) >= 0.5
    gate.set()
    time.sleep(0.3)
    assert reader.latest.data > stuck.data
    end_event.set()
    reader.thread.join(2.)

  def test_failing_read_logged_rate_limited_and_thread_survives(self, log):
    calls = []

    def read(timer):
      calls.append(1)
      if len(calls) > 1:
        raise OSError("EIO")
      return "first"

    reader, end_event = self.run_reader(read, interval=0.02)
    time.sleep(0.5)
    assert reader.thread.is_alive()
    assert len(calls) > 5
    assert reader.latest.data == "first"  # failed reads keep the previous sample, which ages
    assert log.exception.call_count == 1
    log.exception.side_effect = RuntimeError("log broken")
    reader.last_error_log = None
    time.sleep(0.2)
    assert reader.thread.is_alive()
    end_event.set()
    reader.thread.join(2.)
