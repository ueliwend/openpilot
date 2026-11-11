"""Texts and decisions of the offline maps screen (Inster port, E9), without any widget, so they are testable offline.

The screen reads `OsmLocationStatus` (written by mapd_manager from the tiles on disk) and writes only requests:
`OsmDownloadRequest` (ids or "all"; a new id is added to the selection by mapd_manager) and `OsmDeleteLocation`
(ids or "OTHER"). The download confirmation measures what it can on the device (tiles on disk, drive folders, free
space) and asks the tile server for the sizes it does not know yet.
"""
import datetime
import os

from openpilot.system.ui.lib.multilang import tr
from openpilot.sunnypilot.mapd.offline_regions import locations, tile_stats, union_tiles
from openpilot.sunnypilot.mapd.offline_server import NO_ANSWER_SIZE, SizeCache
from openpilot.sunnypilot.mapd.offline_storage import MIN_DRIVE_BUFFER_HOURS, Plan, estimate_tiles, plan_download

REQUEST_KEY = "OsmDownloadRequest"
DELETE_KEY = "OsmDeleteLocation"
STATUS_KEY = "OsmLocationStatus"
OTHER = "OTHER"
DATE_UNKNOWN_BEFORE = datetime.datetime(2020, 1, 1).timestamp()  # an unset clock writes 1970 times
NEEDS_DOWNLOAD = ("not_downloaded", "partial", "failed")


def size_text(n: int | float) -> str:
  return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def day_text(ts: float) -> str:
  if not ts or ts < DATE_UNKNOWN_BEFORE:
    return tr("date unknown")
  d = datetime.datetime.fromtimestamp(ts)
  return f"{d.day} {d.strftime('%b')}"


def day_range_text(first: float, last: float) -> str:
  if not first or first < DATE_UNKNOWN_BEFORE or not last or last < DATE_UNKNOWN_BEFORE:
    return tr("date unknown")
  a, b = datetime.datetime.fromtimestamp(first), datetime.datetime.fromtimestamp(last)
  if a.date() == b.date():
    return day_text(first)
  if (a.year, a.month) == (b.year, b.month):
    return f"{a.day}–{b.day} {b.strftime('%b')}"
  return f"{day_text(first)} – {day_text(last)}"


def entries(status: dict | None) -> list[dict]:
  """The selected locations in selection order, each with its id."""
  if not isinstance(status, dict) or not isinstance(status.get("entries"), dict):
    return []
  out = []
  for location_id in status.get("order") or list(status["entries"]):
    entry = status["entries"].get(location_id)
    if isinstance(entry, dict):
      out.append(dict(entry, id=location_id))
  return out


def run_of(status: dict | None) -> dict:
  run = status.get("run") if isinstance(status, dict) else None
  return run if isinstance(run, dict) else {}


def busy(status: dict | None) -> bool:
  return bool(run_of(status).get("active"))


def entry_value(entry: dict, run: dict) -> str:
  state = entry.get("state")
  size, date = size_text(entry.get("bytes", 0)), day_text(entry.get("date", 0))
  if state == "downloading":
    return tr("{}/{} downloading").format(run.get("tiles_done", 0), run.get("tiles_total", entry.get("tiles_total", 0)))
  if state == "queued":
    return tr("queued")
  if state == "not_downloaded":
    return tr("not downloaded")
  if state == "partial":
    if entry.get("tiles_present", 0) < entry.get("tiles_expected", 0):
      return tr("partial {}/{} · {}").format(entry.get("tiles_present", 0), entry.get("tiles_expected", 0), size)
    last = entry.get("last") or {}
    return tr("partly updated {}/{} · {} · {}").format(last.get("refreshed", 0), last.get("expected", 0), size, date)
  if state == "failed":
    return tr("failed – data from {} kept").format(date) if entry.get("tiles_present") else tr("failed")
  return f"{size} · {date}"


def needs_download(status: dict | None) -> list[dict]:
  return [e for e in entries(status) if e.get("state") in NEEDS_DOWNLOAD]


WAIT_TEXTS = {
  "onroad": "waiting until the car is off",
  "mapd": "starting",
  "waiting for an unmetered network": "waiting for Wi-Fi or another unmetered network",
  "checking the map server": "checking the map server",
  "map server not reachable": "map server not reachable",
}
SERVER_WAITS = ("map server not reachable", "map server error")


def time_text(ts: float) -> str:
  return datetime.datetime.fromtimestamp(ts).strftime("%H:%M") if ts and ts >= DATE_UNKNOWN_BEFORE else ""


def can_retry(status: dict | None) -> bool:
  """A run waits for the map server (no answer or an error): a request by hand asks again at once."""
  run = run_of(status)
  return bool(run.get("active")) and not run.get("current") and str(run.get("waiting", "")).startswith(SERVER_WAITS)


def progress_text(status: dict | None) -> str:
  run = run_of(status)
  if not run.get("active"):
    return ""
  if run.get("current"):
    entry = (status.get("entries") or {}).get(run["current"], {})
    return tr("Country {} of {} · {} {}/{}").format(run.get("index", 1), run.get("count", 1), entry.get("title", run["current"]),
                                                   run.get("tiles_done", 0), run.get("tiles_total", 0))
  waiting = run.get("waiting") or "mapd"
  if run.get("server_error"):
    text = tr("map server error (HTTP {})").format(run["server_error"])
  else:
    text = tr(WAIT_TEXTS.get(waiting, waiting))
  if can_retry(status) and time_text(run.get("retry_at", 0)):
    text += " · " + tr("next try {}").format(time_text(run["retry_at"]))
  return text


def progress_fraction(status: dict | None) -> float:
  run = run_of(status)
  total = run.get("tiles_total", 0)
  return max(0.0, min(1.0, run.get("tiles_done", 0) / total)) if total else 0.0


def storage_text(status: dict | None) -> str:
  storage = status.get("storage") if isinstance(status, dict) else None
  if not storage:
    return ""
  return tr("Maps {} · free {} · drive buffer ≈ {:.1f} h").format(size_text(storage.get("maps", 0)), size_text(storage.get("available", 0)),
                                                                  storage.get("buffer_hours", 0.0))


def other_text(status: dict | None) -> str:
  other = status.get("other") if isinstance(status, dict) else None
  return size_text(other["bytes"]) if isinstance(other, dict) and other.get("bytes") else ""


def delete_text(entry: dict) -> str:
  text = tr("Delete the maps of {}? Frees {}.").format(entry.get("title", entry.get("id")), size_text(entry.get("exclusive_bytes", 0)))
  if entry.get("shared_tiles"):
    text += " " + tr("{} map tiles shared with {} stay.").format(entry["shared_tiles"], ", ".join(entry.get("shared_with") or []))
  return text


def other_delete_text(status: dict | None) -> str:
  return tr("Delete the map areas no selected country uses? Frees {}.").format(other_text(status) or size_text(0))


def deleter_text(plan: Plan) -> str:
  """What the deleter removes while the download runs (owner: always allowed; the confirmation says how much and when)."""
  r = plan.replay
  if plan.budget.shortfall <= 0 or not r.folders:
    return ""
  if r.moved:
    text = tr("Moves the oldest {} min of drive recordings to the external drive (last written {}).").format(
      r.minutes, day_range_text(r.first_written, r.last_written))
    if r.lost_minutes:
      text += " " + tr("The external drive deletes its oldest {} min (last written {}).").format(
        r.lost_minutes, day_range_text(r.lost_first_written, r.lost_last_written))
    return text
  return tr("Frees space by deleting the oldest {} min of drive recordings (last written {}).").format(
    r.minutes, day_range_text(r.first_written, r.last_written))


def download_text(titles: list[str], plan: Plan | None) -> str:
  names = ", ".join(titles)
  if plan is None:
    return tr("Download {}? The size is not known (no answer from the map server).").format(names)
  est = plan.estimate
  if not plan.budget.ok:
    return tr("Not enough storage for {}: maps may use {} in all to keep {:.0f} h of drive recording.").format(
      names, size_text(max(0, plan.budget.max_maps)), MIN_DRIVE_BUFFER_HOURS)
  approx = "≥ " if est.unknown else "≈ "
  text = tr("Download {}? {}{} on the device, {}{} to download.").format(names, approx, size_text(est.on_disk + est.new), approx,
                                                                       size_text(est.download))
  extra = deleter_text(plan)
  return f"{text} {extra}" if extra else text


def estimate(root: str, location_ids: list[str], server, log_root: str | None = None) -> Plan | None:
  """Guard and deleter replay for downloading these locations now (runs in a thread: it may ask the server for sizes)."""
  cache = SizeCache(root)
  on_disk, unknown = {}, {}
  answered, reachable = False, True
  for t in sorted(union_tiles(location_ids)):
    stats = tile_stats(root, t)
    if stats is not None:
      on_disk[t] = stats[0]
      continue
    size = cache.download(t)
    if size is None and reachable:
      size = server.head(t)
      answered = answered or (size is not None and size != NO_ANSWER_SIZE)
      if size == NO_ANSWER_SIZE:  # do not wait for every tile when the server does not answer
        size, reachable = None, False
    elif size is not None:
      answered = True
    unknown[t] = size
  if not reachable or (unknown and not answered and not on_disk):
    return None
  return plan_download(root, estimate_tiles(on_disk, unknown, cache.ratio()), log_root=log_root)


def country_options() -> list[tuple[str, str]]:
  """Every country of mapd v1.12.0's list (code, name), sorted by name."""
  return locations("nation")


def state_options() -> list[tuple[str, str]]:
  return locations("state")


def map_size(root: str) -> int:
  """Bytes of the downloaded maps, incl. the old copies moved aside and the download files of a running update."""
  total = 0
  for sub in ("offline", "offline_prev", "tmp"):
    stack = [os.path.join(root, sub)]
    while stack:
      try:
        for e in os.scandir(stack.pop()):
          if e.is_dir(follow_symlinks=False):
            stack.append(e.path)
          elif e.is_file(follow_symlinks=False):
            total += e.stat().st_size
      except OSError:
        pass
  return total
