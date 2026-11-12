"""The mapd binary at low priority while one of our downloads runs (mapd_priority.py; owner 12 Nov 2026 11:36)."""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import pytest

import openpilot.sunnypilot.mapd.mapd_priority as mp
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase


class FakeOps:
  """Threads of mapd binaries: {pid: {tid: [nice, ioprio]}}; raising priority fails unless `may_raise`."""

  def __init__(self, procs, may_raise=True):
    self.procs, self.may_raise = procs, may_raise
    self.calls: list = []
    self.signals: list = []
    self.scans = self.pid_checks = 0

  def mapd_pids(self):
    self.scans += 1  # a full /proc scan in production
    return list(self.procs)

  def is_mapd(self, pid):
    self.pid_checks += 1
    return pid in self.procs

  def threads(self, pid):
    return list(self.procs.get(pid, {}))

  def _thread(self, tid):
    for threads in self.procs.values():
      if tid in threads:
        return threads[tid]
    raise ProcessLookupError(tid)

  def get_nice(self, tid):
    return self._thread(tid)[0]

  def set_nice(self, tid, nice):
    t = self._thread(tid)
    if nice < t[0] and not self.may_raise:
      raise PermissionError(13, "Permission denied")
    self.calls.append(("nice", tid, nice))
    t[0] = nice

  def get_ioprio(self, tid):
    return self._thread(tid)[1]

  def set_ioprio(self, tid, value):
    self.calls.append(("ioprio", tid, value))
    self._thread(tid)[1] = value

  def kill(self, pid, sig):
    self.signals.append((pid, sig))


class FakeManager:
  def __init__(self):
    self.stopped = False

  def mapd_stopped(self):
    return self.stopped


class Clock:
  t = 100.0


def normal(*tids):
  return {t: [mp.NORMAL_NICE, mp.NORMAL_IOPRIO] for t in tids}


class TestMapdPriority(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()
    self.clock = Clock()

  def make(self, procs, may_raise=True):
    self.ops, self.manager = FakeOps(procs, may_raise), FakeManager()
    return mp.MapdPriority(self.params, ops=self.ops, manager=self.manager, monotonic=lambda: self.clock.t)

  def test_no_renice_while_idle(self):
    p = self.make({500: normal(500, 501, 502)})
    for _ in range(3):
      p.step(False)
    assert self.ops.calls == [] and p.state == mp.NORMAL and not self.params.get_bool(mp.RESTART_KEY)

  def test_low_priority_while_a_download_runs_also_for_new_threads(self):
    p = self.make({500: normal(500, 501)})
    p.step(True)
    assert self.ops.procs[500] == {500: [19, mp.LOW_IOPRIO], 501: [19, mp.LOW_IOPRIO]} and p.state == mp.LOW
    assert mp.LOW_IOPRIO == (2 << 13) | 7  # best-effort class, level 7
    self.ops.procs[500][502] = [0, 0]  # the Go runtime starts another thread
    calls = len(self.ops.calls)
    p.step(True)
    assert self.ops.procs[500][502] == [19, mp.LOW_IOPRIO] and len(self.ops.calls) == calls + 2  # only the new one

  def test_normal_priority_when_the_download_ends(self):
    p = self.make({500: normal(500, 501)})
    p.step(True)
    p.step(False)
    assert self.ops.procs[500] == normal(500, 501) and p.state == mp.NORMAL
    assert not self.params.get_bool(mp.RESTART_KEY) and self.ops.signals == []

  def test_restored_at_start_after_a_crash(self):
    p = self.make({500: {500: [19, mp.LOW_IOPRIO], 501: [19, 0]}})  # mapd_manager restarted while the binary was low
    p.step(False)
    assert self.ops.procs[500] == normal(500, 501)

  def test_restart_through_the_manager_when_restoring_fails(self):
    p = self.make({500: normal(500, 501)}, may_raise=False)
    p.step(True)
    p.step(False)  # nice 19 -> 0 refused (no CAP_SYS_NICE / RLIMIT_NICE)
    assert p.state == mp.RESTARTING and self.params.get_bool(mp.RESTART_KEY) and self.ops.signals == [(500, signal.SIGTERM)]
    p.step(False)  # still alive, within the grace time: nothing more
    assert self.ops.signals == [(500, signal.SIGTERM)] and self.params.get_bool(mp.RESTART_KEY)
    self.clock.t += mp.KILL_GRACE_SECONDS + 1
    p.step(False)
    assert self.ops.signals[-1] == (500, signal.SIGKILL)
    del self.ops.procs[500]  # gone, but the manager has not noticed yet: the request stays (or mapd would never restart)
    p.step(False)
    assert self.params.get_bool(mp.RESTART_KEY)
    self.manager.stopped = True
    p.step(False)
    assert not self.params.get_bool(mp.RESTART_KEY) and p.state == mp.NORMAL  # the manager starts a new binary
    self.ops.procs[600] = normal(600)  # the new binary, at normal priority
    p.step(False)
    assert ("nice", 600, 0) not in self.ops.calls

  def test_a_restart_request_that_is_not_ours_is_removed(self):
    p = self.make({500: normal(500)})
    self.params.put_bool(mp.RESTART_KEY, True, block=True)  # left by a crash, or written remotely
    p.step(False)
    assert not self.params.get_bool(mp.RESTART_KEY)

  def test_mapd_ready_is_false_while_a_restart_is_requested(self):
    from openpilot.system.manager.process_config import mapd_ready
    os.makedirs(mp_root(), exist_ok=True)
    assert mapd_ready(False, self.params, None) is True
    self.params.put_bool(mp.RESTART_KEY, True, block=True)
    assert mapd_ready(False, self.params, None) is False

  def test_sunnylink_may_not_write_the_restart_request(self):
    from openpilot.sunnypilot.sunnylink.utils import BLOCKED_PARAMS
    assert mp.RESTART_KEY in BLOCKED_PARAMS


class TestReviewSuggestions(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()
    self.clock = Clock()

  def make(self, procs, may_raise=True):
    self.ops, self.manager = FakeOps(procs, may_raise), FakeManager()
    return mp.MapdPriority(self.params, ops=self.ops, manager=self.manager, monotonic=lambda: self.clock.t)

  def test_no_proc_work_while_idle_after_the_start_check(self):
    p = self.make({500: normal(500, 501)})
    p.step(False)  # the one check at start
    assert self.ops.scans == 1
    for _ in range(10):
      p.step(False)
    assert self.ops.scans == 1 and self.ops.pid_checks == 0

  def test_the_pid_is_cached_while_a_download_runs(self):
    p = self.make({500: normal(500)})
    p.step(False)
    for _ in range(5):
      p.step(True)
    assert self.ops.scans == 1 and self.ops.pid_checks == 5  # one cheap cmdline check per step, no full scan
    self.ops.procs = {700: normal(700)}  # the binary was restarted (e.g. crashed) during the download
    p.step(True)
    assert self.ops.scans == 2 and self.ops.procs[700][700] == [19, mp.LOW_IOPRIO]
    p.step(False)  # the download ends: restored, then idle again without scans
    p.step(False)
    p.step(False)
    assert self.ops.procs[700][700] == [0, mp.NORMAL_IOPRIO] and self.ops.scans == 2

  def test_agnos_default_io_priority_counts_as_normal(self):
    be4, none4 = (2 << 13) | 4, (0 << 13) | 4  # best-effort 4 (app group) and "none, level 4" (no I/O context yet)
    p = self.make({500: {500: [0, be4], 501: [0, none4], 502: [0, 0]}})
    p.step(False)
    assert self.ops.calls == [] and mp.NORMAL_IOPRIO == be4

  def test_restore_sets_best_effort_4(self):
    p = self.make({500: normal(500)})
    p.step(True)
    p.step(False)
    assert ("ioprio", 500, (2 << 13) | 4) in self.ops.calls

  def test_restart_request_cleared_after_30_s_without_confirmation(self):
    p = self.make({500: normal(500)}, may_raise=False)
    p.step(True)
    p.step(False)
    assert self.params.get_bool(mp.RESTART_KEY)
    del self.ops.procs[500]
    self.clock.t += 20
    p.step(False)  # gone, but managerState never confirms
    assert self.params.get_bool(mp.RESTART_KEY) and p.state == mp.RESTARTING
    self.clock.t += mp.RESTART_MAX_WAIT_SECONDS - 20 + 1
    p.step(False)
    assert not self.params.get_bool(mp.RESTART_KEY) and p.state == mp.NORMAL
    self.ops.procs[800] = normal(800)  # the manager started a new binary: checked once, left alone
    p.step(False)
    assert self.ops.calls[-1][1] != 800


class TestManagerHook(OpenpilotTestCase):
  def test_update_osm_db_lowers_only_while_our_download_runs(self, monkeypatch):
    import types
    import openpilot.sunnypilot.mapd.mapd_manager as mm
    import openpilot.sunnypilot.mapd.offline_update as ou
    seen = []
    updater = types.SimpleNamespace(state=ou.IDLE)
    monkeypatch.setattr(mm, "_offline_updater", updater)
    monkeypatch.setattr(mm, "_offline_maps", types.SimpleNamespace(step=lambda: None))
    monkeypatch.setattr(mm, "_mapd_priority", types.SimpleNamespace(step=seen.append))
    for state in (ou.IDLE, ou.QUEUED, ou.RUNNING, ou.DRAINING, ou.RECOVERING, ou.IDLE):
      updater.state = state
      mm.update_osm_db()
    assert seen == [False, False, True, True, False, False]


def mp_root():
  from openpilot.common.hardware.hw import Paths
  return Paths.mapd_root()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc and syscalls")
def test_system_ops_on_a_real_process():
  """SystemOps against a real process started from a copy of `sleep` named like the binary: found by argv[0], every
  thread lowered and restored (nice and ioprio syscalls)."""
  d = tempfile.mkdtemp()
  binary = os.path.join(d, "mapd")
  shutil.copy(shutil.which("sleep"), binary)
  proc = subprocess.Popen([binary, "30"])
  try:
    time.sleep(0.2)
    ops = mp.SystemOps(binary=binary)
    assert ops.mapd_pids() == [proc.pid]
    tids = ops.threads(proc.pid)
    assert tids == [proc.pid]
    p = mp.MapdPriority(FakeParams(), ops=ops, manager=FakeManager())
    p.step(True)
    assert ops.get_nice(proc.pid) == 19 and ops.get_ioprio(proc.pid) in (mp.LOW_IOPRIO, 0)  # 0: machine without ioprio syscall
    p.step(False)
    if p.state == mp.RESTARTING:  # no right to raise priority here: the restart path took over
      assert proc.wait(5) is not None
    else:
      assert ops.get_nice(proc.pid) == 0 and ops.get_ioprio(proc.pid) in (mp.NORMAL_IOPRIO, (2 << 13) | 4)
  finally:
    proc.kill()
    proc.wait()
    shutil.rmtree(d)


class FakeParams(dict):
  def get_bool(self, key):
    return bool(self.get(key))

  def put_bool(self, key, value, block=False):
    self[key] = value

  def remove(self, key):
    self.pop(key, None)
