"""Offline maps screen, several countries at once (Inster port, E9; E11 map updates): the rows below "Downloaded Maps".

- Storage: maps (+ staged), free space and the drive buffer at the current recording settings and with full recording;
- progress: the current country and its tiles, what is staged for the next start and the cap while driving, or why it
  waits; "map downloads stopped - restart the device" when map_downloader's heartbeat stops; "map update could not be
  applied" when staged tiles survived two starts;
- Update maps automatically: the weekly check (`OsmAutoUpdate`, on by default; owner 13 Nov 2026);
- Update all: checks every tile of every selected country now;
- Not downloaded: countries with no map tile and none due, with DOWNLOAD (e.g. after a sunnylink restore);
- Other map areas: map tiles no selected country uses, with DELETE;
- Countries: ADD, always available: tick any number of countries (United States: then states or all states); one
  confirmation with per-country and total sizes and, on a full disk, which old drives go;
- one row per selected country: its value (size and tile dates, "update queued", "downloading 3/6", "4 of 6 updated |
  applies at next start", the shrink notice) with UPDATE (QUEUED while the country has tiles due) and DELETE (tiles shared
  with other countries stay).
Downloads also run while driving (at most 1 MB/s); new maps go live at the next start. Texts and decisions:
osm_locations_model.py; widgets: osm_widgets.py. The screen only writes requests; map_downloader runs them.
"""
import threading
import time

from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.selfdrive.ui.sunnypilot.layouts.settings import osm_locations_model as model
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.osm_widgets import CountryPickDialog, QueueProgressAction, ValueTwoButtonsAction
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.mapd.offline_server import TileServer
from openpilot.sunnypilot.mapd.offline_storage import recording_mb_per_min
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.lib.utils import NoElideButtonAction
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, toggle_item_sp
from openpilot.system.ui.sunnypilot.widgets.tree_dialog import TreeFolder, TreeNode
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.list_view import ListItem, text_item

US = "US"
ALL_STATES = "All"
MEM_PARAMS_PATH = "/dev/shm/params"


def append_request(key: str, ids: list[str]) -> None:
  """Adds ids to a request map_downloader has not read yet (it reads and clears it once a second), so no tap is lost."""
  pending = [i for i in (ui_state.params.get(key) or "").split(",") if i]
  ui_state.params.put(key, ",".join(dict.fromkeys(pending + ids)))


class OsmLocationsPanel:
  def __init__(self, server=None, mem_params=None, monotonic=time.monotonic):
    self._root = Paths.mapd_root()
    self._server = server or TileServer()
    self._mem_params = mem_params
    self._monotonic = monotonic
    self._status: dict | None = None
    self._live: dict | None = None
    self._row_ids: list[str] = []
    self._row_queued: list[bool] = []
    self._checking = False
    self._storage = text_item(tr("Storage"), "")
    self._progress = ListItem(title=tr("Map Updates"), action_item=QueueProgressAction())
    self._auto = toggle_item_sp(tr(model.AUTO_TITLE), tr(model.AUTO_DESCRIPTION), initial_state=ui_state.params.get_bool(model.AUTO_KEY),
                                param=model.AUTO_KEY)
    self._update_all_btn = ListItemSP(tr("Update all"), action_item=NoElideButtonAction(tr("UPDATE"), enabled=True), callback=self._update_all)
    self._missing_btn = ListItemSP(tr("Not downloaded"), action_item=NoElideButtonAction(tr("DOWNLOAD"), enabled=True),
                                   callback=self._download_missing)
    self._other_btn = ListItemSP(tr("Other map areas"), action_item=NoElideButtonAction(tr("DELETE"), enabled=True), callback=self._delete_other)
    self._add_btn = ListItemSP(tr("Countries"), action_item=NoElideButtonAction(tr("ADD"), enabled=True), callback=self._add)
    self.rows: list[ListItemSP] = []
    self.items = [self._storage, self._progress, self._auto, self._update_all_btn, self._missing_btn, self._other_btn, self._add_btn]
    self._progress.set_visible(False)

  def _read_live(self) -> dict | None:
    if self._mem_params is None:
      try:
        self._mem_params = Params(MEM_PARAMS_PATH)
      except Exception:
        return None
    try:
      return self._mem_params.get(model.LIVE_KEY)
    except Exception:
      return None

  # ---- refresh (once a second, from OSMLayout)
  def update(self, scroller) -> bool:
    """Reads OsmLocationStatus and the heartbeat and updates the rows; returns whether a download runs."""
    self._status = ui_state.params.get(model.STATUS_KEY)
    self._live = self._read_live()
    status = self._status
    stopped = model.downloader_stopped(status, self._live, self._monotonic())
    running = model.busy(status)
    free = not self._checking

    self._storage.action_item.set_text(model.storage_text(status, self._live))
    self._progress.set_visible(model.show_progress(status, stopped))
    current = bool(model.run_of(status).get("current")) and not stopped
    self._progress.action_item.update(model.progress_fraction(status, self._live) * 100, model.progress_text(status, self._live, stopped),
                                      show_progress=current)

    entries = model.entries(status)
    self._update_all_btn.set_visible(bool(entries))
    self._update_all_btn.action_item.set_enabled(free)
    self._update_all_btn.action_item.set_value(status.get("message", "") if isinstance(status, dict) else "")

    missing = model.needs_download(status)
    self._missing_btn.set_visible(bool(missing))
    self._missing_btn.action_item.set_enabled(free)
    self._missing_btn.action_item.set_value(", ".join(e.get("title", e["id"]) for e in missing))

    other = model.other_text(status)
    self._other_btn.set_visible(bool(other))
    self._other_btn.action_item.set_value(other)
    self._other_btn.action_item.set_enabled(free)
    self._add_btn.action_item.set_enabled(free)

    run = model.run_of(status)
    pending = [i for i in model.pending_deletes(status) if i not in {e["id"] for e in entries}]
    shown = [(e["id"], e.get("title", e["id"]), model.entry_value(e, run), False, model.update_enabled(e)) for e in entries]
    shown += [(i, model.title_of(i), tr("delete queued"), True, False) for i in pending]
    while len(self.rows) < len(shown):
      i = len(self.rows)
      action = ValueTwoButtonsAction(tr("UPDATE"), tr("DELETE"), left_callback=lambda i=i: self._update_row(i),
                                     right_callback=lambda i=i: self._delete_row(i))
      row = ListItemSP("", action_item=action)
      self.rows.append(row)
      scroller.add_widget(row)
    self._row_ids = [x[0] for x in shown]
    self._row_queued = [x[3] for x in shown]
    for i, row in enumerate(self.rows):
      row.set_visible(i < len(shown))
      if i < len(shown):
        _, title, value, delete_queued, can_update = shown[i]
        row.set_title(title)
        row.action_item.set_value(value)
        row.action_item.set_text(tr("UPDATE") if can_update or delete_queued else tr("QUEUED"))
        row.action_item.set_left_enabled(free and can_update and not delete_queued)
        row.action_item.set_right_text(tr("QUEUED") if delete_queued else tr("DELETE"))
        row.action_item.set_right_enabled(free and not delete_queued)
    return running

  # ---- dialogs
  def _confirm(self, text: str, confirm_text: str, on_confirm) -> None:
    gui_app.push_widget(ConfirmDialog(text, confirm_text, callback=lambda res: on_confirm() if res == DialogResult.CONFIRM else None))

  def _rate(self) -> float:
    return recording_mb_per_min(ui_state.params)

  def _ask_download(self, location_ids: list[str], titles: list[str], request: str) -> None:
    """Update all / DOWNLOAD: measure in a thread (tile sizes may come from the server), then ask."""
    self._checking = True

    def work():
      try:
        plan = model.estimate(self._root, location_ids, self._server, rate_mb_min=self._rate())
      except Exception:
        plan = None
      self._checking = False
      text = model.download_text(titles, plan)
      if plan is not None and not plan.budget.ok:
        gui_app.push_widget(ConfirmDialog(text, tr("OK")))
        return
      self._confirm(text, tr("Download"), lambda: append_request(model.REQUEST_KEY, request.split(",")))
    threading.Thread(target=work, daemon=True).start()

  def _ask_batch(self, location_ids: list[str]) -> None:
    """ADD: one confirmation for all ticked countries, checked together with those already queued."""
    self._checking = True
    queued = model.queued_ids(self._status)

    def work():
      try:
        batch = model.estimate_batch(self._root, location_ids, queued, self._server, rate_mb_min=self._rate())
      except Exception:
        batch = None
      self._checking = False
      text = model.batch_text([model.title_of(i) for i in location_ids], batch)
      if batch is not None and not batch.plan.budget.ok:
        gui_app.push_widget(ConfirmDialog(text, tr("OK")))
        return
      self._confirm(text, tr("Download"), lambda: append_request(model.REQUEST_KEY, location_ids))
    threading.Thread(target=work, daemon=True).start()

  def _update_all(self) -> None:
    entries = model.entries(self._status)
    if entries:
      self._ask_download([e["id"] for e in entries], [e.get("title", e["id"]) for e in entries], "all")

  def _download_missing(self) -> None:
    missing = model.needs_download(self._status)
    if missing:
      ids = [e["id"] for e in missing]
      self._ask_download(ids, [e.get("title", e["id"]) for e in missing], ",".join(ids))

  def _update_row(self, i: int) -> None:
    """Per-country UPDATE: every tile of the country is checked now, whatever its age (map_downloader)."""
    if i >= len(self._row_ids) or self._row_queued[i]:
      return
    append_request(model.REQUEST_KEY, [self._row_ids[i]])

  def _delete_row(self, i: int) -> None:
    if i >= len(self._row_ids) or self._row_queued[i]:
      return
    location_id = self._row_ids[i]
    entry = next((e for e in model.entries(self._status) if e["id"] == location_id), {"id": location_id, "title": location_id})
    self._confirm(model.delete_text(entry), tr("Delete"), lambda: append_request(model.DELETE_KEY, [location_id]))

  def _delete_other(self) -> None:
    self._confirm(model.other_delete_text(self._status), tr("Delete"), lambda: append_request(model.DELETE_KEY, [model.OTHER]))

  def _add(self) -> None:
    selected = set(model.selected_ids(self._status))
    options = [(c, n) for c, n in model.country_options() if c not in selected]
    self._pick(tr("Select Countries"), options, self._on_countries)

  def _pick(self, title: str, options: list[tuple[str, str]], on_pick, first: TreeNode | None = None) -> None:
    nodes = [TreeNode(ref=code, data={"display_name": name}) for code, name in options]
    if first is not None:
      nodes.insert(0, first)
    dialog = CountryPickDialog(title, [TreeFolder(folder="", nodes=nodes)], search_prompt=tr("Perform a search"))
    dialog.on_exit = lambda res: on_pick(list(dialog.selected_refs)) if res == DialogResult.CONFIRM and dialog.selected_refs else None
    gui_app.push_widget(dialog)

  def _on_countries(self, codes: list[str]) -> None:
    if US in codes:  # the US path: states, or all states (= the whole US)
      others = [c for c in codes if c != US]
      selected = set(model.selected_ids(self._status))
      states = [(c, n) for c, n in model.state_options() if f"US:{c}" not in selected]
      self._pick(tr("Select States"), states, lambda picked: self._on_states(others, picked),
                 first=TreeNode(ref=ALL_STATES, data={"display_name": tr("All states (~6.0 GB)")}))
      return
    self._ask_batch(codes)

  def _on_states(self, countries: list[str], states: list[str]) -> None:
    ids = list(countries) + ([US] if ALL_STATES in states else [f"US:{s}" for s in states])
    self._ask_batch(ids)
