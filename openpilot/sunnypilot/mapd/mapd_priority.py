"""The mapd binary at low CPU and I/O priority while one of our map downloads runs (Inster port; owner 12 Nov 2026 11:36).

Why: a download (extracting tiles, writing area files) competes with openpilot's own start-up, which on the car was
delayed by 1.4-1.7 s when a download ran (inster-dev results/2026-11-12_startup_slowdown, fix2.md section 3, review 2).
The binary is one process for downloads and for onroad lookups, so it is lowered only while a download of ours runs:

- running = the offline updater has sent a location and it has not ended (state RUNNING or DRAINING); our request
  `OSMDownloadLocations` is non-empty in exactly that time (offline_update.py);
- then every thread of the binary gets nice 19 and best-effort I/O level 7 (`setpriority` / `ioprio_set` per thread id;
  re-applied each step, so threads the Go runtime creates later are caught too);
- afterwards, and once at mapd_manager start (a crash may have left it low), nice 0 and best-effort I/O 4 come back
  (AGNOS starts openpilot's app group at best-effort 4, run_in_app_group.sh; "none" with level 4 means the same and also
  counts as normal, so nothing is touched then);
- no /proc work while no download runs: one check at start, then the binary's pid is cached and its threads are listed
  only while a download runs or the binary is still low (a full /proc scan only if the cached pid is gone);
- raising priority again is allowed for openpilot's user `comma`: AGNOS's comma.service sets LimitNICE=-10, and every
  process inherits it, so nice 19 -> 0 needs no capability (review: agnos-builder comma.service). Fallback if it is still
  refused: the binary is restarted through the manager. `MapdRestartRequest` makes process_config's mapd_ready false, the
  manager stops the binary and, once the request is cleared, starts a new one at normal priority. The request is cleared
  when managerState shows mapd stopped, and after 30 s at the latest (logged), so mapd never stays stopped.

A download that continues onroad keeps the low priority until it ends (owner accepted, 12 Nov 11:37: it is not stopped).
Map lookups then lose CPU only when all cores are busy (best-effort 7 is time-sliced, not starved: INFERRED).
"""
import ctypes
import os
import platform
import signal
import time

from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.mapd import MAPD_PATH

RESTART_KEY = "MapdRestartRequest"
LOW_NICE, NORMAL_NICE = 19, 0
IOPRIO_CLASS_SHIFT = 13
IOPRIO_CLASS_NONE, IOPRIO_CLASS_BE = 0, 2
IOPRIO_WHO_PROCESS = 1  # a thread id (Linux ioprio_set)
LOW_IOPRIO = (IOPRIO_CLASS_BE << IOPRIO_CLASS_SHIFT) | 7
NORMAL_IOPRIO = (IOPRIO_CLASS_BE << IOPRIO_CLASS_SHIFT) | 4  # AGNOS's app-group default (run_in_app_group.sh)
SYS_IOPRIO = {"aarch64": (30, 31), "x86_64": (251, 252)}  # (ioprio_set, ioprio_get) syscall numbers
KILL_GRACE_SECONDS = 5.0
RESTART_MAX_WAIT_SECONDS = 30.0  # the request is cleared at the latest after this, confirmed by managerState or not


def is_normal_io(value: int) -> bool:
  """Best-effort 4, or class none (derived from nice 0: best-effort 4 as well)."""
  return value == NORMAL_IOPRIO or (value >> IOPRIO_CLASS_SHIFT) == IOPRIO_CLASS_NONE

NORMAL, LOW, RESTARTING = "normal", "low", "restarting"


class SystemOps:
  """Process access on Linux (production); tests use a fake."""

  def __init__(self, binary: str = MAPD_PATH, proc: str = "/proc"):
    self.binary, self.proc = binary, proc
    self._libc = None

  def mapd_pids(self) -> list[int]:
    """Processes whose program is the mapd binary (argv[0]); the bash wrapper of the manager is not one of them."""
    out = []
    for name in os.listdir(self.proc):
      if not name.isdigit():
        continue
      try:
        with open(os.path.join(self.proc, name, "cmdline"), "rb") as f:
          argv0 = f.read().split(b"\0", 1)[0].decode(errors="replace")
      except OSError:
        continue
      if argv0 == self.binary:
        out.append(int(name))
    return out

  def is_mapd(self, pid: int) -> bool:
    """The cached pid still is the binary (cheaper than a full scan)."""
    try:
      with open(os.path.join(self.proc, str(pid), "cmdline"), "rb") as f:
        return f.read().split(b"\0", 1)[0].decode(errors="replace") == self.binary
    except OSError:
      return False

  def threads(self, pid: int) -> list[int]:
    try:
      return [int(t) for t in os.listdir(os.path.join(self.proc, str(pid), "task")) if t.isdigit()]
    except OSError:
      return []

  def get_nice(self, tid: int) -> int:
    return os.getpriority(os.PRIO_PROCESS, tid)

  def set_nice(self, tid: int, nice: int) -> None:
    os.setpriority(os.PRIO_PROCESS, tid, nice)

  def _syscall(self, index: int, *args) -> int:
    numbers = SYS_IOPRIO.get(platform.machine())
    if numbers is None:
      return 0  # no ioprio on this machine: nice only
    if self._libc is None:
      self._libc = ctypes.CDLL(None, use_errno=True)
    ret = self._libc.syscall(numbers[index], *args)
    if ret < 0:
      err = ctypes.get_errno()
      raise OSError(err, os.strerror(err))
    return ret

  def get_ioprio(self, tid: int) -> int:
    return self._syscall(1, IOPRIO_WHO_PROCESS, tid)

  def set_ioprio(self, tid: int, value: int) -> None:
    self._syscall(0, IOPRIO_WHO_PROCESS, tid, value)

  def kill(self, pid: int, sig: int) -> None:
    os.kill(pid, sig)


class ManagerState:
  """Whether the manager has stopped the mapd process (managerState: proc gone = pid 0); production reads cereal."""

  def __init__(self):
    self._sm = None

  def mapd_stopped(self) -> bool | None:
    if self._sm is None:
      from openpilot.cereal import messaging
      self._sm = messaging.SubMaster(["managerState"])
    self._sm.update(0)
    if not self._sm.seen["managerState"]:
      return None
    for p in self._sm["managerState"].processes:
      if p.name == "mapd":
        return not p.running and p.pid == 0
    return None


class MapdPriority:
  def __init__(self, params, ops=None, manager=None, monotonic=time.monotonic):
    self.params = params
    self.ops = ops or SystemOps()
    self.manager = manager or ManagerState()
    self._monotonic = monotonic
    self.state = NORMAL
    self._restart_started = 0.0
    self._killed: set[int] = set()
    self._checked_at_start = False
    self._pids: list[int] = []

  def step(self, downloading: bool) -> None:
    try:
      self._step(downloading)
    except Exception:
      cloudlog.exception("mapd priority: step failed, retrying next second")

  def _step(self, downloading: bool) -> None:
    if self.state == RESTARTING:
      self._continue_restart()
      return
    if self.params.get_bool(RESTART_KEY):  # not ours (left by a crash, or written remotely): never keep mapd stopped
      self.params.remove(RESTART_KEY)
    if not downloading and self.state == NORMAL and self._checked_at_start:
      return  # idle and normal: no /proc work
    self._checked_at_start = True
    tids = [(pid, tid) for pid in self._mapd_pids() for tid in self.ops.threads(pid)]
    if downloading:
      self._lower(tids)
    elif self._any_low(tids):  # after a download, and at mapd_manager start (a crash may have left it low)
      self._restore(tids)
    else:
      self.state = NORMAL

  def _lower(self, tids) -> None:
    changed = 0
    for _, tid in tids:
      try:
        if self.ops.get_nice(tid) != LOW_NICE:
          self.ops.set_nice(tid, LOW_NICE)
          changed += 1
        if self.ops.get_ioprio(tid) != LOW_IOPRIO:
          self.ops.set_ioprio(tid, LOW_IOPRIO)
      except ProcessLookupError:
        continue  # the thread ended
    if self.state != LOW and tids:
      cloudlog.event("mapd_priority", action="low", threads=len(tids), changed=changed)
    if tids:
      self.state = LOW

  def _mapd_pids(self) -> list[int]:
    """The binary's pid from the cache while it is still the binary; a full /proc scan otherwise."""
    if self._pids and all(self.ops.is_mapd(pid) for pid in self._pids):
      return self._pids
    self._pids = self.ops.mapd_pids()
    return self._pids

  def _any_low(self, tids) -> bool:
    for _, tid in tids:
      try:
        if self.ops.get_nice(tid) != NORMAL_NICE or not is_normal_io(self.ops.get_ioprio(tid)):
          return True
      except ProcessLookupError:
        continue
    return False

  def _restore(self, tids) -> None:
    try:
      for _, tid in tids:
        try:
          self.ops.set_nice(tid, NORMAL_NICE)
          self.ops.set_ioprio(tid, NORMAL_IOPRIO)
        except ProcessLookupError:
          continue
    except PermissionError:
      cloudlog.event("mapd_priority", action="restore_failed_restarting", threads=len(tids))
      self._begin_restart({pid for pid, _ in tids})
      return
    cloudlog.event("mapd_priority", action="normal", threads=len(tids))
    self.state = NORMAL

  # ---- restart through the manager (process_config.mapd_ready is false while the request is set)
  def _begin_restart(self, pids: set[int]) -> None:
    self.params.put_bool(RESTART_KEY, True, block=True)
    self.state = RESTARTING
    self._restart_started = self._monotonic()
    self._killed = set()
    self._signal(pids, signal.SIGTERM)

  def _signal(self, pids, sig) -> None:
    for pid in pids:
      try:
        self.ops.kill(pid, sig)
      except ProcessLookupError:
        pass
      self._killed.add(pid)

  def _continue_restart(self) -> None:
    alive = set(self.ops.mapd_pids())
    waited = self._monotonic() - self._restart_started
    if alive and waited > KILL_GRACE_SECONDS:
      self._signal(alive, signal.SIGKILL)
    if not alive and self.manager.mapd_stopped():
      self.params.remove(RESTART_KEY)  # the manager starts a new binary at normal priority at its next loop
      cloudlog.event("mapd_priority", action="restarted")
      self._end_restart()
    elif waited > RESTART_MAX_WAIT_SECONDS:  # never confirmed (no managerState, or the binary would not die)
      self.params.remove(RESTART_KEY)
      cloudlog.event("mapd_priority", action="restart_request_cleared_unconfirmed", alive=len(alive), waited=round(waited))
      self._end_restart()

  def _end_restart(self) -> None:
    self.state = NORMAL
    self._pids = []
    self._checked_at_start = False  # check the new binary once


def restart_requested(params) -> bool:
  """For process_config.mapd_ready: the binary is stopped while mapd_manager asks for a restart."""
  return params.get_bool(RESTART_KEY)
