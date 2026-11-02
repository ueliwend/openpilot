"""Hardware reads that can block (sysfs, procfs, statvfs, modem/network queries), moved off hardwared's deviceState
publishing path, and the diagnostics that name the read when one stalls.

- CallTimer times single calls of one thread and logs the slow ones ("hardwared slow call"); it never raises.
- PeriodicReader runs a read every interval in its own thread and keeps the latest complete Sample. The publishing
  loop only takes that sample, so a read that blocks holds up the reader thread, not deviceState.
- Freshness (PeriodicReader.fresh): each reader has a stall limit. The reader is stale once the next sample has been
  due for longer than that limit; hardware_thread() then publishes deviceState with valid=False (see the limits there).
"""
import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from openpilot.common.swaglog import cloudlog

SLOW_CALL_THRESHOLD = 0.5  # s, a single call slower than this is logged
LOG_INTERVAL = 10.  # s, at most one log of a kind per thread (or reader) per interval


class NetworkContext:
  """The network type hw_state_thread saw last, and the one before it. Attached to the slow-call and stale-reading logs:
  the 2 Nov stall coincided with Wi-Fi coming up. Written by hw_state_thread only; readers may see a mix of two
  updates, which is fine for diagnostics."""

  def __init__(self) -> None:
    self.network_type: str | None = None
    self.prev_network_type: str | None = None
    self.changed_t: float | None = None

  def update(self, network_type: str) -> None:
    if network_type != self.network_type:
      self.prev_network_type, self.network_type, self.changed_t = self.network_type, network_type, time.monotonic()

  def as_dict(self) -> dict[str, Any]:
    changed_t = self.changed_t
    return {
      'type': self.network_type,
      'prev_type': self.prev_network_type,
      'changed_s_ago': None if changed_t is None else round(time.monotonic() - changed_t, 1),
    }


class CallTimer:
  """Diagnostics only: times single hardware calls of one thread and logs each call slower than SLOW_CALL_THRESHOLD
  with the thread, the call, its duration and the network context. Logs at most once per LOG_INTERVAL; slow calls in
  between are counted per call (count, max_ms) and reported with the next log. The logging never raises, the call's
  own result and exceptions pass through unchanged."""

  def __init__(self, thread: str, network: NetworkContext | None = None) -> None:
    self.thread = thread
    self.network = network
    self.in_progress: tuple[str, float] | None = None  # (call, start), read by other threads, see running()
    self.last_log: float | None = None
    self.suppressed: dict[str, list[int]] = {}

  def call(self, name: str, fn: Callable[..., Any], *args, **kwargs) -> Any:
    start = time.monotonic()
    self.in_progress = (name, start)
    try:
      return fn(*args, **kwargs)
    finally:
      self.in_progress = None
      self._done(name, start)

  def running(self) -> dict[str, Any] | None:
    """The call this thread is in right now and for how long, or None. Names a call that hasn't returned (yet)."""
    try:
      cur = self.in_progress
      if cur is None:
        return None
      return {'call': cur[0], 'running_ms': round((time.monotonic() - cur[1]) * 1e3)}
    except Exception:
      return None

  def _done(self, name: str, start: float) -> None:
    try:
      now = time.monotonic()
      duration_ms = round((now - start) * 1e3)
      if now - start <= SLOW_CALL_THRESHOLD:
        return
      if self.last_log is not None and now - self.last_log < LOG_INTERVAL:
        count_max = self.suppressed.setdefault(name, [0, 0])
        count_max[0] += 1
        count_max[1] = max(count_max[1], duration_ms)
        return
      self.last_log, suppressed, self.suppressed = now, self.suppressed, {}
      cloudlog.event("hardwared slow call", error=True, thread=self.thread, call=name, duration_ms=duration_ms,
                     network=self.network.as_dict() if self.network is not None else None, suppressed=suppressed)
    except Exception:
      pass


@dataclass(frozen=True)
class Sample:
  t: float  # monotonic time the read started: the data is at most now - t old
  data: Any


class PeriodicReader:
  """Calls read(timer) every `interval` seconds (counted from the start of a read; a read slower than that is followed
  by the next one at once) in its own daemon thread and keeps the latest complete result in `latest`. A read that
  blocks only holds up this thread. A read that raises is logged (rate-limited) and keeps the previous sample.

  Stall and staleness, exactly: after a sample started at t the next one is due at t + interval. overdue(now) =
  now - (t + interval) is how long that next sample has been missing, i.e. how long the current read has been stuck
  (or the reads have been failing). The reader is stale when overdue > stall_limit, so a read may stall for exactly
  stall_limit before the reader turns stale; the data is then at most stall_limit + interval old. Before the first
  sample the reader is stale. fresh() is meant for a single consumer thread and also logs the transitions."""

  def __init__(self, name: str, read: Callable[[CallTimer], Any], interval: float, stall_limit: float,
               network: NetworkContext | None = None) -> None:
    self.name = name
    self.read = read
    self.interval = interval
    self.stall_limit = stall_limit
    self.network = network
    self.timer = CallTimer(name, network)
    self.latest: Sample | None = None
    self.first_sample = threading.Event()
    self.thread: threading.Thread | None = None
    self.last_error_log: float | None = None
    # consumer side, see fresh()
    self.stale_since: float | None = None
    self.last_stale_log: float | None = None

  def start(self, end_event: threading.Event) -> None:
    self.thread = threading.Thread(target=self.run, args=(end_event,), name=f"hardwared_{self.name}", daemon=True)
    self.thread.start()

  def run(self, end_event: threading.Event) -> None:
    while not end_event.is_set():
      start = time.monotonic()
      try:
        self.latest = Sample(start, self.read(self.timer))
        self.first_sample.set()
      except Exception:
        self._log_error()
      end_event.wait(max(0., self.interval - (time.monotonic() - start)))

  def _log_error(self) -> None:
    try:
      now = time.monotonic()
      if self.last_error_log is None or now - self.last_error_log >= LOG_INTERVAL:
        self.last_error_log = now
        cloudlog.exception(f"hardwared {self.name} read failed")
    except Exception:
      pass

  def age(self, now: float) -> float:
    """How old the latest sample's data is (from the start of its read), inf before the first sample."""
    latest = self.latest
    return math.inf if latest is None else now - latest.t

  def overdue(self, now: float) -> float:
    """How long the next sample has been due, see the class docstring; inf before the first sample."""
    return self.age(now) - self.interval

  def fresh(self, now: float) -> bool:
    """True unless the reader is stale (overdue > stall_limit). Logs "hardwared stale readings" when it turns stale
    and every LOG_INTERVAL while it stays stale (with the call the reader is stuck in, if any, and whether its thread
    is alive), and "hardwared readings fresh" when it recovers. The logging never raises."""
    overdue = self.overdue(now)
    is_fresh = overdue <= self.stall_limit
    try:
      if is_fresh:
        if self.stale_since is not None:
          cloudlog.event("hardwared readings fresh", error=True, reader=self.name, stale_ms=round((now - self.stale_since) * 1e3))
          self.stale_since = self.last_stale_log = None
      else:
        if self.stale_since is None:
          self.stale_since = now
        if self.last_stale_log is None or now - self.last_stale_log >= LOG_INTERVAL:
          self.last_stale_log = now
          age = self.age(now)
          cloudlog.event("hardwared stale readings", error=True, reader=self.name,
                         stalled_ms=round(overdue * 1e3) if math.isfinite(overdue) else None,
                         age_ms=round(age * 1e3) if math.isfinite(age) else None, limit_ms=round(self.stall_limit * 1e3),
                         running=self.timer.running(), thread_alive=self.thread is not None and self.thread.is_alive(),
                         network=self.network.as_dict() if self.network is not None else None)
    except Exception:
      pass
    return is_fresh


def wait_first_samples(readers: Iterable[PeriodicReader], timeout: float, end_event: threading.Event) -> None:
  """Waits until every reader has its first sample, at most `timeout` seconds, and returns early on end_event."""
  deadline = time.monotonic() + timeout
  for reader in readers:
    while not reader.first_sample.is_set() and not end_event.is_set():
      remaining = deadline - time.monotonic()
      if remaining <= 0.:
        return
      reader.first_sample.wait(min(0.1, remaining))
