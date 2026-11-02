"""hardware_thread() with fake messaging, params and hardware, in real time: a stalled read must not stop deviceState,
and a read stalled for longer than its reader's limit must surface (valid=False, startup blocked; for thermal also fan
at 100 % and thermal status held)."""
import copy
import queue
import threading
import time
from unittest import mock

import pytest

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from openpilot.common.hardware.base import ThermalConfig, ThermalZone
import openpilot.system.hardware.hardwared as hardwared
import openpilot.system.hardware.stats_reader as stats_reader

ThermalStatus = log.DeviceState.ThermalStatus


class Gate:
  """Reads pass while open; a closed gate blocks them (a stalled sysfs/statvfs read) until it opens again."""
  def __init__(self):
    self.open_event = threading.Event()
    self.open_event.set()

  def close(self):
    self.open_event.clear()

  def open(self):
    self.open_event.set()

  def wait(self):
    assert self.open_event.wait(30.)


class FakeZone(ThermalZone):
  def __init__(self, name, gate, temps):
    super().__init__(name)
    self.gate = gate
    self.temps = temps

  def read(self):
    self.gate.wait()
    return self.temps['t']


class FakeHardware:
  def __init__(self, thermal_gate, temps):
    zone = lambda name: FakeZone(name, thermal_gate, temps)  # noqa: E731
    self.thermal_config = ThermalConfig(cpu=[zone("cpu0"), zone("cpu1")], gpu=[zone("gpu0")], memory=zone("ddr"), pmic=[zone("pmic")])
    self.power_save = []

  def get_device_type(self):
    return "pc"

  def initialize_hardware(self):
    pass

  def get_thermal_config(self):
    return self.thermal_config

  def booted(self):
    return True

  def set_power_save(self, enabled):
    self.power_save.append(enabled)

  def get_current_power_draw(self):
    return 0.

  def get_som_power_draw(self):
    return 0.

  def get_gpu_usage_percent(self):
    return 12.

  def get_screen_brightness(self):
    return 50


class FakeParams:
  def __init__(self):
    self.d = {
      "HasAcceptedTerms": hardwared.terms_version,
      "HasAcceptedTermsSP": hardwared.terms_version_sp,
      "CompletedTrainingVersion": hardwared.training_version,
    }

  def get(self, key, block=False, return_default=False):
    return self.d.get(key, 0. if return_default else None)

  def get_bool(self, key, block=False):
    return bool(self.d.get(key, False))

  def put(self, key, val, block=False):
    self.d[key] = val

  def put_bool(self, key, val, block=False):
    self.d[key] = bool(val)


class FakeSubMaster:
  """pandaStates at 10 Hz, so hardware_thread publishes every 0.5 s, like with a panda."""
  ignition = True

  def __init__(self, services, poll=None):
    self.services = services
    self.frame = 0
    self.updated = dict.fromkeys(services, False)
    self.alive = dict.fromkeys(services, False)
    self.valid = dict.fromkeys(services, False)
    self.recv_time = dict.fromkeys(services, 0.)
    self.data = {s: getattr(messaging.new_message(s), s) for s in services if s != 'pandaStates'}
    self.data['pandaStates'] = messaging.new_message('pandaStates', 1).pandaStates
    ps = self.data['pandaStates'][0]
    ps.pandaType = log.PandaState.PandaType.tres
    ps.ignitionLine = self.ignition
    ps.harnessStatus = log.PandaState.HarnessStatus.normal

  def __getitem__(self, s):
    return self.data[s]

  def update(self, timeout):
    time.sleep(0.1)
    self.frame += 1
    self.updated = {s: s == 'pandaStates' for s in self.services}
    self.recv_time['pandaStates'] = time.monotonic()


class FakePubMaster:
  sent: list = []

  def __init__(self, services):
    pass

  def send(self, service, msg):
    FakePubMaster.sent.append((time.monotonic(), msg.valid, msg.deviceState.to_dict()))


class Harness:
  def __init__(self, monkeypatch):
    self.thermal_gate = Gate()
    self.stats_gate = Gate()
    self.memory_gate = Gate()
    self.memory = {'pct': 40.}
    self.temps = {'t': 50.}
    self.hw = FakeHardware(self.thermal_gate, self.temps)
    self.log = mock.MagicMock()
    # record copies: hardware_thread logs dicts it keeps changing (startup_conditions), cloudlog serialises them at once
    self.logged = []
    self.log.event.side_effect = lambda *args, **kwargs: self.logged.append((args, copy.deepcopy(kwargs)))
    FakePubMaster.sent = []

    def get_available_percent(default):
      self.stats_gate.wait()
      return 60.

    def memory_usage_percent():
      self.memory_gate.wait()
      return self.memory['pct']

    def read_memory_usage(system_stats, timer):
      return timer.call("memory_usage_percent", memory_usage_percent)

    power_monitor = mock.MagicMock()
    power_monitor.get_power_used.return_value = 0
    power_monitor.get_car_battery_capacity.return_value = 0
    power_monitor.should_shutdown.return_value = False
    patches = {
      "HARDWARE": self.hw, "Params": FakeParams, "PowerMonitoring": lambda: power_monitor, "statlog": mock.MagicMock(),
      "set_offroad_alert": mock.MagicMock(), "get_short_branch": lambda: "test", "get_usb_state": list,
      "get_build_metadata": lambda: mock.MagicMock(channel_type="release", channel="test"), "ChestnutStatus": mock.MagicMock,
      "get_available_percent": get_available_percent, "read_memory_usage": read_memory_usage, "cloudlog": self.log,
    }
    for name, value in patches.items():
      monkeypatch.setattr(hardwared, name, value)
    monkeypatch.setattr(hardwared, "prev_offroad_states", {})
    monkeypatch.setattr(stats_reader, "cloudlog", self.log)
    monkeypatch.setattr(messaging, "SubMaster", FakeSubMaster)
    monkeypatch.setattr(messaging, "PubMaster", FakePubMaster)

    self.end_event = threading.Event()
    self.thread = threading.Thread(target=hardwared.hardware_thread, args=(self.end_event, queue.Queue(maxsize=1), stats_reader.NetworkContext()))

  def start(self):
    self.t0 = time.monotonic()
    self.thread.start()

  def run_for(self, seconds):
    time.sleep(seconds)
    assert self.thread.is_alive()
    return time.monotonic()

  def stop(self):
    self.thermal_gate.open()
    self.stats_gate.open()
    self.memory_gate.open()
    self.end_event.set()
    self.thread.join(10.)
    assert not self.thread.is_alive()

  @property
  def sent(self):
    return FakePubMaster.sent

  def between(self, t1, t2):
    return [(t, valid, ds) for t, valid, ds in self.sent if t1 <= t < t2]

  def max_gap(self):
    times = [t for t, _, _ in self.sent]
    return max(b - a for a, b in zip(times, times[1:], strict=False))

  def events(self, name):
    return [kwargs for args, kwargs in self.logged if args and args[0] == name]


@pytest.fixture
def harness(monkeypatch):
  h = Harness(monkeypatch)
  yield h
  h.stop()


# the loop publishes every 0.5 s; before, a stalled read held deviceState back for the whole stall
MAX_GAP = 1.25


def test_transient_thermal_stall_keeps_publishing(harness):
  harness.start()
  t_onroad = harness.run_for(2.0)
  assert harness.sent[-1][2]['started']

  harness.temps['t'] = 60.
  harness.thermal_gate.close()
  t_stall = harness.run_for(hardwared.THERMAL_STALL_LIMIT - 0.5)  # the read stalls for at most this long, < the limit
  harness.thermal_gate.open()
  harness.run_for(1.5)

  assert harness.max_gap() < MAX_GAP
  assert all(valid for _, valid, _ in harness.sent)
  during = harness.between(t_onroad + 0.6, t_stall)
  assert len(during) >= 4
  assert all(ds['cpuTempC'] == [50., 50.] and ds['started'] and ds['fanSpeedPercentDesired'] < 100 for _, _, ds in during)
  assert harness.sent[-1][2]['cpuTempC'] == [60., 60.]
  slow = [e for e in harness.events("hardwared slow call") if e['thread'] == "thermal"]
  assert len(slow) == 1 and slow[0]['call'].startswith("thermal:") and slow[0]['duration_ms'] >= 3500
  assert harness.events("hardwared stale readings") == []


def test_stale_thermal_surfaces(harness, monkeypatch):
  monkeypatch.setattr(hardwared, "THERMAL_STALL_LIMIT", 1.0)
  harness.start()
  harness.run_for(2.0)
  assert harness.sent[-1][2]['started'] and harness.sent[-1][1]
  max_temp = harness.sent[-1][2]['maxTempC']

  harness.temps['t'] = 99.  # would be overheated, if it were read
  harness.thermal_gate.close()
  t_stall = time.monotonic()
  t_release = harness.run_for(3.5)
  harness.thermal_gate.open()
  harness.run_for(1.5)

  assert harness.max_gap() < MAX_GAP
  invalid = [(t, ds) for t, valid, ds in harness.sent if not valid]
  # the stuck read is the first one due after the gate closed (within one interval), stale once it has stalled for the
  # limit, published at the next 0.5 s iteration
  assert t_stall + 1.0 < invalid[0][0] < t_stall + 0.5 + 1.0 + 0.5 + 0.2
  assert invalid[-1][0] < t_release + 0.6
  assert len(invalid) >= 3
  for _, ds in invalid:
    assert ds['fanSpeedPercentDesired'] == 100
    assert ds['thermalStatus'] == 'ok'  # held, not moved on frozen (or missing) readings
    assert ds['maxTempC'] == max_temp
    assert ds['started']  # stays onroad, selfdrived raises commIssue on valid=False

  (stale,) = harness.events("hardwared stale readings")
  assert stale['reader'] == "thermal" and stale['running']['call'].startswith("thermal:")
  (fresh,) = harness.events("hardwared readings fresh")
  assert fresh['reader'] == "thermal"
  assert harness.sent[-1][1] and harness.sent[-1][2]['cpuTempC'] == [99., 99.]


def test_stale_readings_block_startup(harness, monkeypatch):
  monkeypatch.setattr(hardwared, "THERMAL_STALL_LIMIT", 1.0)
  harness.thermal_gate.close()  # the very first thermal read is stuck
  harness.start()
  t_release = harness.run_for(3.0)
  harness.thermal_gate.open()
  harness.run_for(2.0)

  before = harness.between(0., t_release)
  assert before[0][0] - harness.t0 < 1.0 + 0.75  # first deviceState after at most the first-sample wait
  assert all(not valid and not ds['started'] and ds['fanSpeedPercentDesired'] == 100 for _, valid, ds in before)
  blocked = harness.events("Startup blocked")
  assert blocked and not blocked[0]['startup_conditions']['device_readings_fresh']
  assert harness.sent[-1][1] and harness.sent[-1][2]['started']
  assert harness.max_gap() < MAX_GAP


def test_stale_stats_surfaces_but_thermal_keeps_working(harness, monkeypatch):
  monkeypatch.setattr(hardwared, "STATS_STALL_LIMIT", 1.0)
  harness.start()
  harness.run_for(2.0)

  harness.stats_gate.close()
  harness.temps['t'] = 70.
  t_release = harness.run_for(3.0)
  harness.stats_gate.open()
  harness.run_for(1.5)

  assert harness.max_gap() < MAX_GAP
  invalid = [ds for _, valid, ds in harness.sent if not valid]
  assert len(invalid) >= 3
  assert all(ds['fanSpeedPercentDesired'] < 100 and ds['started'] and ds['freeSpacePercent'] == 60. for ds in invalid)
  assert invalid[-1]['cpuTempC'] == [70., 70.]  # temperatures still update
  assert invalid[-1]['maxTempC'] > invalid[0]['maxTempC']
  (stale,) = harness.events("hardwared stale readings")
  assert stale['reader'] == "stats" and stale['running']['call'] == "get_available_percent"
  slow = [e for e in harness.events("hardwared slow call") if e['thread'] == "stats"]
  assert slow[0]['call'] == "get_available_percent" and slow[0]['duration_ms'] >= 2500
  assert all(valid for t, valid, _ in harness.sent if t > t_release + 0.6)
  assert not any(harness.hw.power_save)  # ignition on: power save stays off, also with stale brightness


def test_failing_thermal_read_surfaces(harness, monkeypatch):
  monkeypatch.setattr(hardwared, "THERMAL_STALL_LIMIT", 1.0)
  harness.start()
  harness.run_for(1.5)
  fail = mock.MagicMock(side_effect=OSError("EIO"))
  monkeypatch.setattr(harness.hw.thermal_config.cpu[0], "read", fail)
  harness.run_for(2.5)
  assert fail.call_count > 1
  assert not harness.sent[-1][1] and harness.sent[-1][2]['fanSpeedPercentDesired'] == 100
  assert harness.log.exception.call_count >= 1
  assert harness.max_gap() < MAX_GAP


def test_memory_feeding_low_memory_has_the_thermal_bound():
  # memoryUsagePercent feeds selfdrived's lowMemory (soft disable): it must not be held longer than the temperatures
  assert hardwared.MEMORY_STALL_LIMIT <= hardwared.THERMAL_STALL_LIMIT


def test_stale_memory_surfaces(harness, monkeypatch):
  monkeypatch.setattr(hardwared, "MEMORY_STALL_LIMIT", 1.0)
  harness.start()
  harness.run_for(2.0)
  assert harness.sent[-1][2]['memoryUsagePercent'] == 40

  harness.memory['pct'] = 95.
  harness.memory_gate.close()
  t_stall = time.monotonic()
  harness.temps['t'] = 60.
  harness.run_for(3.0)
  harness.memory_gate.open()
  harness.run_for(1.5)

  assert harness.max_gap() < MAX_GAP
  invalid = [(t, ds) for t, valid, ds in harness.sent if not valid]
  assert t_stall + 1.0 < invalid[0][0] < t_stall + 0.5 + 1.0 + 0.5 + 0.2
  for _, ds in invalid:
    assert ds['memoryUsagePercent'] == 40  # the held value, but marked invalid
    assert ds['fanSpeedPercentDesired'] < 100 and ds['started']  # temperatures are fresh: thermal logic keeps running
  assert invalid[-1][1]['cpuTempC'] == [60., 60.]
  (stale,) = harness.events("hardwared stale readings")
  assert stale['reader'] == "memory" and stale['running']['call'] == "memory_usage_percent"
  assert harness.sent[-1][1] and harness.sent[-1][2]['memoryUsagePercent'] == 95


def test_storage_stall_within_stats_limit_keeps_valid(harness):
  # an October-style storage stall (statvfs on a slow /data, a few seconds) stays below STATS_STALL_LIMIT
  harness.start()
  harness.run_for(1.5)
  harness.stats_gate.close()
  harness.run_for(4.0)
  harness.stats_gate.open()
  harness.run_for(1.0)
  assert harness.max_gap() < MAX_GAP
  assert all(valid for _, valid, _ in harness.sent)
  assert harness.events("hardwared stale readings") == []
  slow = [e for e in harness.events("hardwared slow call") if e['thread'] == "stats"]
  assert slow[0]['call'] == "get_available_percent" and slow[0]['duration_ms'] >= 3000


def test_shutdown_during_first_sample_wait(harness):
  harness.thermal_gate.close()  # first read stuck: hardware_thread waits up to THERMAL_STALL_LIMIT for it
  harness.start()
  time.sleep(0.3)
  harness.end_event.set()
  harness.thread.join(1.5)
  assert not harness.thread.is_alive()
  assert harness.sent == []


def test_hw_state_thread_logs_slow_network_call(monkeypatch):
  fake_log = mock.MagicMock()
  monkeypatch.setattr(stats_reader, "cloudlog", fake_log)
  NetworkType = log.DeviceState.NetworkType

  hw = mock.MagicMock()

  def slow_network_type():
    time.sleep(0.6)
    return NetworkType.wifi

  hw.get_network_type.side_effect = slow_network_type
  hw.get_modem_temperatures.return_value = []
  hw.get_modem_data_usage.return_value = (-1, -1)
  monkeypatch.setattr(hardwared, "HARDWARE", hw)
  monkeypatch.setattr(hardwared, "get_usb_topology", set)
  monkeypatch.setattr(hardwared, "get_usb_state", list)

  network = stats_reader.NetworkContext()
  network.update("none")
  end_event = threading.Event()
  hw_queue = queue.Queue(maxsize=1)
  t = threading.Thread(target=hardwared.hw_state_thread, args=(end_event, hw_queue, network))
  t.start()
  try:
    hw_state = hw_queue.get(timeout=5.)
  finally:
    end_event.set()
    t.join(5.)
  assert hw_state.network_type == NetworkType.wifi
  (evt,) = [c.kwargs for c in fake_log.event.call_args_list if c.args[0] == "hardwared slow call"]
  assert evt['thread'] == "hw_state" and evt['call'] == "get_network_type" and evt['duration_ms'] >= 600
  assert evt['network']['type'] == "wifi" and evt['network']['prev_type'] == "none"
