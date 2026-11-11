"""Settings > OSM with several countries at once (Inster port, E9).

Upstream's OSMLayout (osm.py) stays unchanged, so its edits merge cleanly; this subclass only swaps the rows below
"Downloaded Maps" for the multi-country rows (osm_locations.py) and refreshes them. settings.py imports it in place of
OSMLayout (one line). Upstream's single-country picker and "Database Update" rows are still created by
OSMLayout._initialize_items but no longer shown; mapd_manager writes the old keys only as a mirror (offline_selection.py).
"""
from openpilot.common.hardware.hw import Paths
from openpilot.selfdrive.ui.sunnypilot.layouts.settings import osm_locations_model as model
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.osm import OSMLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.osm_locations import OsmLocationsPanel
from openpilot.selfdrive.ui.ui_state import device
from openpilot.system.ui.lib.multilang import tr


class OSMLayoutMulti(OSMLayout):
  def _initialize_items(self):
    super()._initialize_items()
    self._locations = OsmLocationsPanel()
    self.items = [self._mapd_version, self._delete_maps_btn, *self._locations.items]

  def calculate_size(self):
    """As upstream, but also counts the old copies moved aside and the download files of a running update."""
    total_size = model.map_size(Paths.mapd_root())
    self._delete_maps_btn.action_item.set_value(f"{total_size / 1024 ** 2:.2f} MB" if total_size < 1024 ** 3 else f"{total_size / 1024 ** 3:.2f} GB")

  def _delete_maps(self):
    self._show_confirm(tr("This will delete ALL downloaded maps and the list of countries\n\nAre you sure you want to delete all maps?"),
                       tr("Yes, delete all maps"), self._on_confirm_delete_maps)

  def _update_labels(self):
    downloading = self._locations.update(self._scroller)
    if downloading:
      device._reset_interactive_timeout()
      self._update_map_size()
    self._delete_maps_btn.action_item.set_enabled(not downloading)
