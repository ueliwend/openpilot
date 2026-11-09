import threading
import time
from types import SimpleNamespace
from unittest import mock

from openpilot.cereal import custom, log
from opendbc.car.structs import car
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.selfdrived import selfdrived

EventName = log.OnroadEvent.EventName
EventNameSP = custom.OnroadEventSP.EventName
CHESTNUT_KEYS = ("ChestnutLoading", "ChestnutActive")


class TestBigModelParams(OpenpilotTestCase):
  def setup_method(self):
    build = SimpleNamespace(openpilot=SimpleNamespace(comma_remote=False, sunnypilot_remote=False), tested_channel=False)
    with mock.patch.object(selfdrived, "get_build_metadata", return_value=build):
      CP = car.CarParams.new_message(brand="mock")
      self.sd = selfdrived.SelfdriveD(CP.as_reader(), custom.CarParamsSP.new_message().as_reader())
    self.CS = car.CarState.new_message().as_reader()

  def params_thread_step(self):
    self.sd.chestnut_params = self.sd.read_chestnut_params()

  def events(self):
    self.sd.update_events(self.CS)
    return set(self.sd.events.names), set(self.sd.events_sp.names)

  def test_not_read_per_cycle(self):
    with mock.patch.object(self.sd.params, "get", wraps=self.sd.params.get) as get, \
         mock.patch.object(self.sd.params, "get_bool", wraps=self.sd.params.get_bool) as get_bool:
      for _ in range(10):
        self.sd.update_events(self.CS)
    keys = [c.args[0] for c in get.call_args_list + get_bool.call_args_list]
    assert not any(k in CHESTNUT_KEYS for k in keys), keys

  def test_events_follow_params_thread(self):
    params = self.sd.params
    params.put_bool("ChestnutLoading", True, block=True)
    events, _ = self.events()
    assert EventName.bigModelLoading not in events   # not seen before the params thread has read it
    self.params_thread_step()
    events, _ = self.events()
    assert EventName.bigModelLoading in events

    params.put_bool("ChestnutLoading", False, block=True)
    params.put_bool("ChestnutActive", True, block=True)
    self.params_thread_step()
    events, events_sp = self.events()
    assert EventNameSP.bigModelReady in events_sp
    assert EventName.bigModelLoading not in events and EventName.bigModelFailed not in events

    params.put_bool("ChestnutActive", False, block=True)
    events, _ = self.events()
    assert EventName.bigModelFailed not in events
    self.params_thread_step()
    events, _ = self.events()
    assert EventName.bigModelFailed in events

  def test_params_thread_refreshes(self):
    assert self.sd.chestnut_params == (False, None)
    self.sd.params.put_bool("ChestnutLoading", True, block=True)
    self.sd.params.put_bool("ChestnutActive", False, block=True)
    evt = threading.Event()
    t = threading.Thread(target=self.sd.params_thread, args=(evt,))
    t.start()
    try:
      deadline = time.monotonic() + 2.0
      while self.sd.chestnut_params != (True, False) and time.monotonic() < deadline:
        time.sleep(0.01)
    finally:
      evt.set()
      t.join()
    assert self.sd.chestnut_params == (True, False)
