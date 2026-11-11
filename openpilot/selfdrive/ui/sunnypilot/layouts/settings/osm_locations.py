"""Offline maps screen, several countries at once (Inster port, E9): the rows below "Downloaded Maps" in Settings > OSM.

- Storage: maps size, free space and the drive buffer left;
- progress of a running download ("Country 2 of 4 · Germany 12/30", or why it waits);
- Update all: downloads every selected country again;
- Not downloaded: the countries without (complete) maps, with DOWNLOAD (e.g. after a sunnylink restore);
- Other map areas: map tiles no selected country uses, with DELETE;
- Countries: ADD opens the country list (any country of mapd v1.12.0's list; United States, then a state or all states);
- one row per selected country: size and date, or its state, and DELETE (tiles shared with other countries stay).
Every download asks first, with the size and, when the disk is full, which old drive recordings the deleter removes.
Texts and decisions: osm_locations_model.py. The screen only writes requests; mapd_manager runs them (offline_maps.py).
"""
import threading

from openpilot.common.hardware.hw import Paths
from openpilot.selfdrive.ui.sunnypilot.layouts.settings import osm_locations_model as model
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.mapd.offline_server import TileServer
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.lib.utils import NoElideButtonAction
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP
from openpilot.system.ui.sunnypilot.widgets.progress_bar import progress_item
from openpilot.system.ui.sunnypilot.widgets.tree_dialog import TreeFolder, TreeNode, TreeOptionDialog
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.list_view import text_item

US = "US"
ALL_STATES = "All"


class OsmLocationsPanel:
  def __init__(self, server=None):
    self._root = Paths.mapd_root()
    self._server = server or TileServer()
    self._status: dict | None = None
    self._row_ids: list[str] = []
    self._checking = False
    self._storage = text_item(tr("Storage"), "")
    self._progress = progress_item(tr("Downloading Maps"))
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
    idle = (not running or retry) and not self._checking

    self._storage.action_item.set_text(model.storage_text(status))
    self._progress.set_visible(running)
    if running:
      self._progress.action_item.update(model.progress_fraction(status) * 100, model.progress_text(status),
                                        show_progress=bool(model.run_of(status).get("current")))

    entries = model.entries(status)
    self._update_all_btn.set_visible(bool(entries))
    self._update_all_btn.action_item.set_enabled(idle)
    self._update_all_btn.action_item.set_value(status.get("message", "") if isinstance(status, dict) else "")

    missing = model.needs_download(status)
    self._missing_btn.set_visible(bool(missing) and (not running or retry))
    self._missing_btn.action_item.set_enabled(idle)
    self._missing_btn.action_item.set_value(", ".join(e.get("title", e["id"]) for e in missing))

    other = model.other_text(status)
    self._other_btn.set_visible(bool(other))
    self._other_btn.action_item.set_value(other)
    self._other_btn.action_item.set_enabled(idle)
    self._add_btn.action_item.set_enabled(idle)

    while len(self.rows) < len(entries):
      i = len(self.rows)
      row = ListItemSP("", action_item=NoElideButtonAction(tr("DELETE"), enabled=True), callback=lambda i=i: self._delete_row(i))
      self.rows.append(row)
      scroller.add_widget(row)
    self._row_ids = [e["id"] for e in entries]
    run = model.run_of(status)
    for i, row in enumerate(self.rows):
      row.set_visible(i < len(entries))
      if i < len(entries):
        row.set_title(entries[i].get("title", entries[i]["id"]))
        row.action_item.set_value(model.entry_value(entries[i], run))
        row.action_item.set_enabled(idle)
    return running

  # ---- dialogs
  def _confirm(self, text: str, confirm_text: str, on_confirm) -> None:
    gui_app.push_widget(ConfirmDialog(text, confirm_text, callback=lambda res: on_confirm() if res == DialogResult.CONFIRM else None))

  def _ask_download(self, location_ids: list[str], titles: list[str], request: str) -> None:
    """Measure in a thread (tile sizes may come from the server), then ask."""
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
      self._confirm(text, tr("Download"), lambda: ui_state.params.put(model.REQUEST_KEY, request))
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
    if i >= len(self._row_ids):
      return
    location_id = self._row_ids[i]
    entry = next((e for e in model.entries(self._status) if e["id"] == location_id), {"id": location_id, "title": location_id})
    self._confirm(model.delete_text(entry), tr("Delete"), lambda: ui_state.params.put(model.DELETE_KEY, location_id))

  def _delete_other(self) -> None:
    self._confirm(model.other_delete_text(self._status), tr("Delete"), lambda: ui_state.params.put(model.DELETE_KEY, model.OTHER))

  def _add(self) -> None:
    self._pick(tr("Select Country"), model.country_options(), self._on_country)

  def _pick(self, title: str, options: list[tuple[str, str]], on_pick, first: TreeNode | None = None) -> None:
    nodes = [TreeNode(ref=code, data={"display_name": name}) for code, name in options]
    if first is not None:
      nodes.insert(0, first)
    dialog = TreeOptionDialog(title, [TreeFolder(folder="", nodes=nodes)], current_ref="", search_prompt=tr("Perform a search"))
    dialog.on_exit = lambda res: on_pick(dialog.selection_ref) if res == DialogResult.CONFIRM and dialog.selection_ref else None
    gui_app.push_widget(dialog)

  def _on_country(self, code: str) -> None:
    if code == US:  # the US path: one state, or all states (= the whole US)
      self._pick(tr("Select State"), model.state_options(), self._on_state,
                 first=TreeNode(ref=ALL_STATES, data={"display_name": tr("All states (~6.0 GB)")}))
      return
    self._add_location(code)

  def _on_state(self, code: str) -> None:
    self._add_location(US if code == ALL_STATES else f"US:{code}")

  def _add_location(self, location_id: str) -> None:
    title = dict(model.country_options()).get(location_id) or dict(model.state_options()).get(location_id.removeprefix("US:"), location_id)
    self._ask_download([location_id], [title], location_id)
