"""Offline maps screen, several countries at once (Inster port, E9): the rows below "Downloaded Maps" in Settings > OSM.

- Storage: free space and the drive buffer left (the maps' size is on "Downloaded Maps", same units);
- progress: a bar over the whole queue with its percentage, the country and its tiles ("23% | Germany 7/30 | country 1
  of 2"), or why it waits (with the server check's progress);
- Update all: downloads every selected country again;
- Not downloaded: the countries without (complete) maps, with DOWNLOAD (e.g. after a sunnylink restore);
- Other map areas: map tiles no selected country uses, with DELETE;
- Countries: ADD, always available: tick any number of countries (United States: then states or all states); one
  confirmation with per-country and total sizes and, on a full disk, which old drives go; added to the running queue;
- one row per selected country: size and date, or its state, and DELETE (tiles shared with other countries stay); a
  DELETE during a download is queued and shown as "delete queued" until the download ends.
Texts and decisions: osm_locations_model.py; widgets: osm_widgets.py. The screen only writes requests; mapd_manager runs
them (offline_maps.py).
"""
import threading

from openpilot.common.hardware.hw import Paths
from openpilot.selfdrive.ui.sunnypilot.layouts.settings import osm_locations_model as model
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.osm_widgets import CountryPickDialog, QueueProgressAction
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.mapd.offline_server import TileServer
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.lib.utils import NoElideButtonAction
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP
from openpilot.system.ui.sunnypilot.widgets.tree_dialog import TreeFolder, TreeNode
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.list_view import ListItem, text_item

US = "US"
ALL_STATES = "All"


def append_request(key: str, ids: list[str]) -> None:
  """Adds ids to a request mapd_manager has not read yet (it reads and clears it once a second), so no tap is lost."""
  pending = [i for i in (ui_state.params.get(key) or "").split(",") if i]
  ui_state.params.put(key, ",".join(dict.fromkeys(pending + ids)))


class OsmLocationsPanel:
  def __init__(self, server=None):
    self._root = Paths.mapd_root()
    self._server = server or TileServer()
    self._status: dict | None = None
    self._row_ids: list[str] = []
    self._row_queued: list[bool] = []
    self._checking = False
    self._storage = text_item(tr("Storage"), "")
    self._progress = ListItem(title=tr("Downloading Maps"), action_item=QueueProgressAction())
    self._update_all_btn = ListItemSP(tr("Update all"), action_item=NoElideButtonAction(tr("UPDATE"), enabled=True), callback=self._update_all)
    self._missing_btn = ListItemSP(tr("Not downloaded"), action_item=NoElideButtonAction(tr("DOWNLOAD"), enabled=True),
                                   callback=self._download_missing)
    self._other_btn = ListItemSP(tr("Other map areas"), action_item=NoElideButtonAction(tr("DELETE"), enabled=True), callback=self._delete_other)
    self._add_btn = ListItemSP(tr("Countries"), action_item=NoElideButtonAction(tr("ADD"), enabled=True), callback=self._add)
    self.rows: list[ListItemSP] = []
    self.items = [self._storage, self._progress, self._update_all_btn, self._missing_btn, self._other_btn, self._add_btn]
    self._progress.set_visible(False)

  # ---- refresh (once a second, from OSMLayout)
  def update(self, scroller) -> bool:
    """Reads OsmLocationStatus and updates the rows; returns whether a download runs."""
    self._status = ui_state.params.get(model.STATUS_KEY)
    status = self._status
    running = model.busy(status)
    retry = model.can_retry(status)  # waiting for the map server: DOWNLOAD / Update all ask again at once
    free = not self._checking  # ADD and DELETE work during a download (added to the queue / queued)
    idle = (not running or retry) and free  # Update all would download countries of this run a second time

    self._storage.action_item.set_text(model.storage_text(status))
    self._progress.set_visible(running)
    if running:
      self._progress.action_item.update(model.queue_fraction(status) * 100, model.progress_text(status), show_progress=True)

    entries = model.entries(status)
    self._update_all_btn.set_visible(bool(entries))
    self._update_all_btn.action_item.set_enabled(idle)
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

    pending = [i for i in model.pending_deletes(status) if i not in {e["id"] for e in entries}]
    shown = [(e["id"], e.get("title", e["id"]), model.entry_value(e, model.run_of(status)), False) for e in entries]
    shown += [(i, model.title_of(i), tr("delete queued"), True) for i in pending]
    while len(self.rows) < len(shown):
      i = len(self.rows)
      row = ListItemSP("", action_item=NoElideButtonAction(tr("DELETE"), enabled=True), callback=lambda i=i: self._delete_row(i))
      self.rows.append(row)
      scroller.add_widget(row)
    self._row_ids = [x[0] for x in shown]
    self._row_queued = [x[3] for x in shown]
    for i, row in enumerate(self.rows):
      row.set_visible(i < len(shown))
      if i < len(shown):
        _, title, value, queued = shown[i]
        row.set_title(title)
        row.action_item.set_value(value)
        row.action_item.set_text(tr("QUEUED") if queued else tr("DELETE"))
        row.action_item.set_enabled(free and not queued)
    return running

  # ---- dialogs
  def _confirm(self, text: str, confirm_text: str, on_confirm) -> None:
    gui_app.push_widget(ConfirmDialog(text, confirm_text, callback=lambda res: on_confirm() if res == DialogResult.CONFIRM else None))

  def _ask_download(self, location_ids: list[str], titles: list[str], request: str) -> None:
    """Update all / DOWNLOAD: measure in a thread (tile sizes may come from the server), then ask."""
    self._checking = True

    def work():
      try:
        plan = model.estimate(self._root, location_ids, self._server)
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
        batch = model.estimate_batch(self._root, location_ids, queued, self._server)
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

  def _delete_row(self, i: int) -> None:
    if i >= len(self._row_ids) or self._row_queued[i]:
      return
    location_id = self._row_ids[i]
    entry = next((e for e in model.entries(self._status) if e["id"] == location_id), {"id": location_id, "title": location_id})
    self._confirm(model.delete_text(entry, running=model.busy(self._status)), tr("Delete"),
                  lambda: append_request(model.DELETE_KEY, [location_id]))

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
