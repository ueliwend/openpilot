"""Offline maps screen for several countries (E9): texts (osm_locations_model.py) and the rows' wiring
(osm_locations.py, loaded with small stand-ins for the widgets, so neither a window nor raylib is needed)."""
import datetime
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.sunnypilot.layouts.settings import osm_locations_model as model
from openpilot.sunnypilot.mapd.offline_storage import Budget, DeleterReplay, Plan, TileEstimate

LAYOUT_PY = Path(__file__).resolve().parents[1] / "sunnypilot/layouts/settings/osm_locations.py"
MULTI_PY = Path(__file__).resolve().parents[1] / "sunnypilot/layouts/settings/osm_multi.py"
SETTINGS_PY = Path(__file__).resolve().parents[1] / "sunnypilot/layouts/settings/settings.py"
OSM_PY = Path(__file__).resolve().parents[1] / "sunnypilot/layouts/settings/osm.py"
NOV2 = datetime.datetime(2026, 11, 2, 12).timestamp()
NOV3 = datetime.datetime(2026, 11, 3, 9).timestamp()
GB = 1_000_000_000


def status(**entries):
  run = entries.pop("run", {"active": False})
  return {"version": 1, "order": list(entries), "entries": entries, "run": run, "other": {"tiles": 0, "bytes": 0},
          "storage": {"maps": int(2.1 * GB), "available": 81 * GB, "buffer_hours": 5.3}, "message": ""}


def entry(title, state, **kw):
  return {"title": title, "state": state, "bytes": int(1.4 * GB), "date": NOV2, "tiles_total": 10, "tiles_expected": 10,
          "tiles_present": 10, "exclusive_bytes": int(0.1 * GB), "shared_tiles": 0, "shared_with": [], "last": None} | kw


def plan(ok=True, shortfall=0, minutes=0, moved=False, lost=0, unknown=0):
  b = Budget(ok, 95 * GB, 10 * GB, int(9.5 * GB), 7 * GB, 60 * GB, GB, 2 * GB, GB, shortfall, 30 * GB, 5.0)
  r = DeleterReplay(needed=shortfall, minutes=minutes, folders=minutes, first_written=NOV2, last_written=NOV3, moved=moved,
                    lost_minutes=lost, lost_first_written=NOV2, lost_last_written=NOV2)
  return Plan(b, TileEstimate(on_disk=0, new=int(0.9 * GB), download=int(0.45 * GB), unknown=unknown), r)


class TestTexts(OpenpilotTestCase):
  def test_row_values(self):
    run = {"active": True, "tiles_done": 4, "tiles_total": 10}
    assert model.entry_value(entry("Austria", "ok"), {}) == "1.4 GB · 2 Nov"
    assert model.entry_value(entry("Austria", "downloading"), run) == "4/10 downloading"
    assert model.entry_value(entry("Austria", "queued"), run) == "queued"
    assert model.entry_value(entry("Austria", "not_downloaded", tiles_present=0, bytes=0), {}) == "not downloaded"
    assert model.entry_value(entry("Austria", "partial", tiles_present=8), {}) == "partial 8/10 · 1.4 GB"
    assert model.entry_value(entry("Austria", "partial", last={"refreshed": 7, "expected": 10}), {}) == "partly updated 7/10 · 1.4 GB · 2 Nov"
    assert model.entry_value(entry("Austria", "failed"), {}) == "failed – data from 2 Nov kept"
    assert model.entry_value(entry("Austria", "ok", bytes=int(0.35 * GB)), {}) == "350 MB · 2 Nov"
    assert model.entry_value(entry("Austria", "ok", date=3600), {}) == "1.4 GB · date unknown"  # unset clock

  def test_days(self):
    assert model.day_range_text(NOV2, NOV3) == "2–3 Nov"
    assert model.day_range_text(NOV2, NOV2 + 3600) == "2 Nov"
    assert model.day_range_text(datetime.datetime(2026, 10, 30).timestamp(), NOV2) == "30 Oct – 2 Nov"
    assert model.day_range_text(0, NOV2) == "date unknown"

  def test_download_confirmation_says_minutes_and_days(self):
    assert model.download_text(["Austria"], plan()) == "Download Austria? ≈ 900 MB on the device, ≈ 450 MB to download."
    assert model.download_text(["Austria"], plan(shortfall=GB, minutes=7)).endswith(
      "Frees space by deleting the oldest 7 min of drive recordings (last written 2–3 Nov).")
    text = model.download_text(["Austria"], plan(shortfall=GB, minutes=7, moved=True, lost=2))
    assert "Moves the oldest 7 min of drive recordings to the external drive (last written 2–3 Nov)." in text
    assert text.endswith("The external drive deletes its oldest 2 min (last written 2 Nov).")
    assert model.download_text(["Austria"], plan(unknown=1)).startswith("Download Austria? ≥ ")
    assert model.download_text(["Austria"], plan(ok=False)).startswith("Not enough storage for Austria: maps may use 30.0 GB in all to keep 3 h")
    assert "size is not known" in model.download_text(["Austria"], None)

  def test_delete_and_storage_texts(self):
    e = entry("Austria", "ok", shared_tiles=2, shared_with=["Switzerland", "Germany"])
    assert model.delete_text(e) == "Delete the maps of Austria? Frees 100 MB. 2 map tiles shared with Switzerland, Germany stay."
    assert model.delete_text(entry("Austria", "ok")) == "Delete the maps of Austria? Frees 100 MB."
    assert model.storage_text(status()) == "Maps 2.1 GB · free 81.0 GB · drive buffer ≈ 5.3 h"
    s = status()
    s["other"] = {"tiles": 3, "bytes": int(0.4 * GB)}
    assert model.other_text(s) == "400 MB" and model.other_text(status()) == ""

  def test_progress(self):
    s = status(CH=entry("Switzerland", "ok"), DE=entry("Germany", "downloading"),
               run={"active": True, "current": "DE", "index": 2, "count": 4, "tiles_done": 12, "tiles_total": 30, "waiting": ""})
    assert model.progress_text(s) == "Country 2 of 4 · Germany 12/30" and model.progress_fraction(s) == 0.4
    s["run"] = {"active": True, "current": None, "waiting": "onroad"}
    assert model.progress_text(s) == "waiting until the car is off"
    s["run"]["waiting"] = "waiting for an unmetered network"
    assert model.progress_text(s) == "waiting for Wi-Fi or another unmetered network"
    assert model.progress_text(status()) == ""

  def test_server_error_shows_its_code_and_allows_a_retry(self):
    retry_at = datetime.datetime(2026, 11, 12, 7, 41).timestamp()
    s = status(CH=entry("Switzerland", "queued"),
               run={"active": True, "current": None, "waiting": "map server error (HTTP 403)", "server_error": 403, "retry_at": retry_at})
    assert model.progress_text(s) == "map server error (HTTP 403) · next try 07:41" and model.can_retry(s)
    s["run"] = {"active": True, "current": None, "waiting": "map server not reachable", "server_error": None, "retry_at": retry_at}
    assert model.progress_text(s) == "map server not reachable · next try 07:41" and model.can_retry(s)
    s["run"] = {"active": True, "current": None, "waiting": "onroad"}
    assert not model.can_retry(s)
    s["run"] = {"active": True, "current": "CH", "waiting": ""}
    assert not model.can_retry(s)

  def test_needs_download_and_country_list(self):
    s = status(CH=entry("Switzerland", "ok"), AT=entry("Austria", "not_downloaded"), DE=entry("Germany", "failed"))
    assert [e["id"] for e in model.needs_download(s)] == ["AT", "DE"]
    codes = dict(model.country_options())
    assert {"CH", "AT", "DE", "IT", "FR", "SI", "ES", "US"} <= set(codes) and codes["SI"] == "Slovenia"
    assert model.entries(None) == [] and model.entries({"entries": "bad"}) == []


class TestEstimate(OpenpilotTestCase):
  def test_estimate_asks_the_server_for_unknown_tiles_only(self):
    import os
    import tempfile
    from openpilot.sunnypilot.mapd.offline_server import NO_ANSWER_SIZE
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "offline", "44", "6"))
    with open(os.path.join(root, "offline", "44", "6", "a.area"), "wb") as f:
      f.write(b"x" * 1000)
    asked = []
    p = model.estimate(root, ["CH"], types.SimpleNamespace(head=lambda t: asked.append(t) or 250), log_root=root)
    assert (44, 6) not in asked and len(asked) == 5
    assert p.estimate.on_disk == 1000 and p.estimate.download == 1000 // 4 + 5 * 250 and p.estimate.new == 5 * 1000
    asked.clear()
    assert model.estimate(root, ["CH"], types.SimpleNamespace(head=lambda t: asked.append(t) or NO_ANSWER_SIZE), log_root=root) is None
    assert len(asked) == 1  # stops asking once the server does not answer


# ---- the rows, with stand-ins
class Action:
  def __init__(self, text="", enabled=True):
    self.text, self.value, self.enabled = text, "", enabled

  def set_text(self, text):
    self.text = text

  def set_value(self, value):
    self.value = value

  def set_enabled(self, enabled):
    self.enabled = enabled

  def update(self, progress, text, show_progress=False):
    self.value = text


class Item:
  def __init__(self, title="", action_item=None, callback=None, **_):
    self.title, self.action_item, self.callback, self.visible = title, action_item or Action(), callback, True

  def set_visible(self, visible):
    self.visible = visible

  def set_title(self, title):
    self.title = title


class FakeParams(dict):
  def get(self, key, *a, **k):
    return super().get(key)

  def put(self, key, value, *a, **k):
    self[key] = value


class Node(types.SimpleNamespace):
  def __init__(self, ref, data):
    super().__init__(ref=ref, data=data)


class Dialog:
  def __init__(self, *args, callback=None, **kw):
    self.args, self.callback, self.kw = args, callback, kw
    self.on_exit, self.selection_ref = None, ""


def load_panel():
  ui_state = types.SimpleNamespace(params=FakeParams())
  pushed = []
  stand_ins = {
    "openpilot.selfdrive.ui.ui_state": {"ui_state": ui_state},
    "openpilot.system.ui.lib.application": {"gui_app": types.SimpleNamespace(push_widget=pushed.append)},
    "openpilot.system.ui.lib.multilang": {"tr": lambda s: s},
    "openpilot.system.ui.widgets": {"DialogResult": types.SimpleNamespace(CONFIRM=1, CANCEL=0)},
    "openpilot.system.ui.widgets.confirm_dialog": {"ConfirmDialog": Dialog},
    "openpilot.system.ui.widgets.list_view": {"text_item": lambda title, value: Item(title, Action(value))},
    "openpilot.system.ui.sunnypilot.lib.utils": {"NoElideButtonAction": Action},
    "openpilot.system.ui.sunnypilot.widgets.list_view": {"ListItemSP": Item},
    "openpilot.system.ui.sunnypilot.widgets.progress_bar": {"progress_item": lambda title: Item(title)},
    "openpilot.system.ui.sunnypilot.widgets.tree_dialog": {"TreeFolder": lambda folder, nodes: nodes,
                                                            "TreeNode": Node,
                                                            "TreeOptionDialog": Dialog},
  }
  modules = {}
  for name, attributes in stand_ins.items():
    modules[name] = types.ModuleType(name)
    vars(modules[name]).update(attributes)
  spec = importlib.util.spec_from_file_location("osm_locations_under_test", LAYOUT_PY)
  module = importlib.util.module_from_spec(spec)
  with mock.patch.dict(sys.modules, modules):
    spec.loader.exec_module(module)
  return module, ui_state.params, pushed


class Scroller:
  def __init__(self):
    self.added = []

  def add_widget(self, w):
    self.added.append(w)


class TestRows(OpenpilotTestCase):
  def setup_method(self):
    self.module, self.params, self.pushed = load_panel()
    self.panel = self.module.OsmLocationsPanel(server=types.SimpleNamespace(head=lambda t: None))
    self.scroller = Scroller()
    self.asked = []
    # the estimate runs in a thread in the app; here at once, with a fixed plan
    monkey = mock.patch.object(self.module.model, "estimate", lambda root, ids, server: self.asked.append(ids) or plan())
    monkey.start()
    self.addCleanup(monkey.stop)
    thread = mock.patch.object(self.module.threading, "Thread", lambda target, daemon: types.SimpleNamespace(start=target))
    thread.start()
    self.addCleanup(thread.stop)

  def confirm_last(self):
    dialog = self.pushed[-1]
    dialog.callback(1)

  def pick(self, ref):
    dialog = self.pushed[-1]
    dialog.selection_ref = ref
    dialog.on_exit(1)

  def test_one_row_per_country_with_delete(self):
    self.params[model.STATUS_KEY] = status(CH=entry("Switzerland", "ok"), AT=entry("Austria", "not_downloaded", tiles_present=0))
    assert self.panel.update(self.scroller) is False
    assert [r.title for r in self.scroller.added] == ["Switzerland", "Austria"]
    assert self.scroller.added[0].action_item.value == "1.4 GB · 2 Nov" and self.scroller.added[0].action_item.text == "DELETE"
    assert self.panel._missing_btn.visible and self.panel._missing_btn.action_item.value == "Austria"
    self.scroller.added[1].callback()
    assert "Delete the maps of Austria?" in self.pushed[-1].args[0]
    self.confirm_last()
    assert self.params[model.DELETE_KEY] == "AT"
    self.params[model.STATUS_KEY] = status(CH=entry("Switzerland", "ok"))  # fewer rows: the spare one is hidden, not removed
    self.panel.update(self.scroller)
    assert [r.visible for r in self.scroller.added] == [True, False]

  def test_add_any_country_asks_with_the_size_then_requests_it(self):
    self.params[model.STATUS_KEY] = status()
    self.panel.update(self.scroller)
    self.panel._add_btn.callback()
    self.pick("SI")
    assert self.asked == [["SI"]] and self.pushed[-1].args[0].startswith("Download Slovenia?")
    self.confirm_last()
    assert self.params[model.REQUEST_KEY] == "SI"

  def test_us_state_path(self):
    self.panel._add()
    self.pick("US")
    assert self.pushed[-1].args[0] == "Select State"
    self.pick("CA")
    self.confirm_last()
    assert self.params[model.REQUEST_KEY] == "US:CA"
    self.panel._add()
    self.pick("US")
    self.pick("All")
    self.confirm_last()
    assert self.params[model.REQUEST_KEY] == "US"

  def test_update_all_and_other_areas(self):
    s = status(CH=entry("Switzerland", "ok"), AT=entry("Austria", "ok"))
    s["other"] = {"tiles": 2, "bytes": int(0.2 * GB)}
    self.params[model.STATUS_KEY] = s
    self.panel.update(self.scroller)
    assert self.panel._other_btn.visible and self.panel._other_btn.action_item.value == "200 MB"
    self.panel._update_all_btn.callback()
    assert self.asked == [["CH", "AT"]]
    self.confirm_last()
    assert self.params[model.REQUEST_KEY] == "all"
    self.panel._other_btn.callback()
    self.confirm_last()
    assert self.params[model.DELETE_KEY] == "OTHER"

  def test_storage_refusal_offers_no_download(self):
    with mock.patch.object(self.module.model, "estimate", lambda root, ids, server: plan(ok=False)):
      self.panel._add()
      self.pick("FR")
    assert self.pushed[-1].args[0].startswith("Not enough storage") and self.pushed[-1].args[1] == "OK"
    assert model.REQUEST_KEY not in self.params

  def test_buttons_work_while_the_run_waits_for_the_map_server(self):
    self.params[model.STATUS_KEY] = status(CH=entry("Switzerland", "queued"),
                                           run={"active": True, "current": None, "waiting": "map server error (HTTP 403)", "server_error": 403})
    assert self.panel.update(self.scroller) is True
    assert self.panel._update_all_btn.action_item.enabled  # a retry by hand
    self.panel._update_all_btn.callback()
    self.confirm_last()
    assert self.params[model.REQUEST_KEY] == "all"

  def test_buttons_wait_while_a_download_runs(self):
    self.params[model.STATUS_KEY] = status(CH=entry("Switzerland", "downloading"),
                                           run={"active": True, "current": "CH", "index": 1, "count": 1, "tiles_done": 2, "tiles_total": 6})
    assert self.panel.update(self.scroller) is True
    assert not self.panel._add_btn.action_item.enabled and not self.panel._update_all_btn.action_item.enabled
    assert not self.scroller.added[0].action_item.enabled
    assert self.panel._progress.visible and self.panel._progress.action_item.value == "Country 1 of 1 · Switzerland 2/6"


# ---- S4: upstream's osm.py unchanged; the screen is a subclass, imported by settings.py in its place
class UpstreamLayout:
  """The parts of upstream's OSMLayout the subclass relies on (osm.py: __init__ calls _initialize_items, then hides
  _progress and _state_btn, then builds the scroller from self.items)."""
  def __init__(self):
    self._initialize_items()
    self._update_map_size()
    self._progress.set_visible(False)
    self._state_btn.set_visible(False)
    self._scroller = Scroller()
    self.confirms = []

  def _initialize_items(self):
    self._mapd_version, self._delete_maps_btn, self._progress = Item("Mapd Version"), Item("Downloaded Maps"), Item("Downloading Map")
    self._update_btn, self._country_btn, self._state_btn = Item("Database Update"), Item("Country"), Item("State")
    self.items = [self._mapd_version, self._delete_maps_btn, self._progress, self._update_btn, self._country_btn, self._state_btn]

  def _update_map_size(self):
    self.calculate_size()

  def _show_confirm(self, msg, confirm_text, func):
    self.confirms.append(msg)

  def _on_confirm_delete_maps(self):
    pass


class FakePanel:
  def __init__(self):
    self.items, self.running, self.updates = [Item("Storage"), Item("Countries")], False, 0

  def update(self, scroller):
    self.updates += 1
    return self.running


def load_multi():
  device = types.SimpleNamespace(resets=0)
  device._reset_interactive_timeout = lambda: setattr(device, "resets", device.resets + 1)
  stand_ins = {
    "openpilot.selfdrive.ui.sunnypilot.layouts.settings.osm": {"OSMLayout": UpstreamLayout},
    "openpilot.selfdrive.ui.sunnypilot.layouts.settings.osm_locations": {"OsmLocationsPanel": FakePanel},
    "openpilot.selfdrive.ui.ui_state": {"device": device},
    "openpilot.system.ui.lib.multilang": {"tr": lambda s: s},
  }
  modules = {}
  for name, attributes in stand_ins.items():
    modules[name] = types.ModuleType(name)
    vars(modules[name]).update(attributes)
  spec = importlib.util.spec_from_file_location("osm_multi_under_test", MULTI_PY)
  module = importlib.util.module_from_spec(spec)
  with mock.patch.dict(sys.modules, modules):
    spec.loader.exec_module(module)
  return module, device


class TestUpstreamHook(OpenpilotTestCase):
  def test_settings_uses_the_subclass_in_place_of_upstreams_layout(self):
    src = SETTINGS_PY.read_text()
    assert "from openpilot.selfdrive.ui.sunnypilot.layouts.settings.osm_multi import OSMLayoutMulti as OSMLayout" in src
    assert "settings.osm import OSMLayout" not in src

  def test_subclass_swaps_the_rows_and_refreshes_them(self):
    module, device = load_multi()
    layout = module.OSMLayoutMulti()
    assert [i.title for i in layout.items] == ["Mapd Version", "Downloaded Maps", "Storage", "Countries"]
    assert layout._delete_maps_btn.action_item.value.endswith("MB")
    layout._update_labels()
    assert layout._locations.updates == 1 and layout._delete_maps_btn.action_item.enabled and device.resets == 0
    layout._locations.running = True
    layout._update_labels()
    assert not layout._delete_maps_btn.action_item.enabled and device.resets == 1
    layout._delete_maps()
    assert "the list of countries" in layout.confirms[-1]


# ---- S5: the upstream methods our subclass overrides or relies on, pinned by the SHA-256 of inspect.getsource
# (upstream osm.py as at openpilot inster 579baf8). A change means the override in osm_multi.py may be out of date.
PINNED_UPSTREAM_METHODS = {
  # overridden by OSMLayoutMulti
  "_initialize_items": "cfb0646e738c9e3767d44ddef227897df6aac4b6c21ac999aa439d3c61b28763",
  "calculate_size": "7712e736a8e20bade84b6746688e06e6a33038e5326307e8d6213d85778080d9",
  "_delete_maps": "3e88dcd67b694ff309bebfa71bf310b7aeec2bee2a3e87c14eb5781511de469b",
  "_update_labels": "e0ac255b7706f58987abea4e9e2f97f3e2aa431969886386548207992fa90de3",
  # relied on: call order in __init__, the confirm and delete-all path, the size thread, the 1 s refresh
  "__init__": "03ac852fdfce46b3ddf61ccecb76688491135bec5597752ed40ef25a9ba9397e",
  "_show_confirm": "cc2f0726bd71264a9bbaa586cdfd46b3bfc158293dc34fc407ea4c407e125b4d",
  "_on_confirm_delete_maps": "bfd3493d3374b7edbff82910ba212f13edb9c8553a2fee319c9287eada102d2f",
  "_update_map_size": "9779b00902b42b4bcc2515a4e689e85b41a4b5003051f00a7f312ec3281950e8",
  "_update_state": "fd3b9deea1e99ab34c341608fc2e3e1655f46232809118aa357ae94f8ffc327c",
}


def load_upstream_osm():
  """upstream's osm.py as a module of its own, with stand-ins for the UI modules it imports (no raylib needed)."""
  names = ["openpilot.selfdrive.ui.ui_state", "openpilot.selfdrive.ui.layouts.settings.software", "openpilot.system.ui.lib.application",
           "openpilot.system.ui.lib.multilang", "openpilot.system.ui.widgets", "openpilot.system.ui.widgets.confirm_dialog",
           "openpilot.system.ui.widgets.list_view", "openpilot.system.ui.widgets.scroller_tici", "openpilot.system.ui.sunnypilot.lib.utils",
           "openpilot.system.ui.sunnypilot.widgets.list_view", "openpilot.system.ui.sunnypilot.widgets.tree_dialog",
           "openpilot.system.ui.sunnypilot.widgets.progress_bar"]
  attrs = {"device": None, "ui_state": None, "time_ago": None, "gui_app": None, "tr": lambda s: s, "DialogResult": None,
           "Widget": type("Widget", (), {}), "ConfirmDialog": None, "text_item": None, "Scroller": None, "NoElideButtonAction": None,
           "ListItemSP": None, "TreeFolder": None, "TreeNode": None, "TreeOptionDialog": None, "progress_item": None}
  modules = {}
  for name in names:
    modules[name] = types.ModuleType(name)
    vars(modules[name]).update(attrs)
  spec = importlib.util.spec_from_file_location("upstream_osm_under_test", OSM_PY)
  module = importlib.util.module_from_spec(spec)
  with mock.patch.dict(sys.modules, modules):
    spec.loader.exec_module(module)
  return module


class TestUpstreamPins(OpenpilotTestCase):
  def test_upstream_methods_the_subclass_depends_on_are_unchanged(self):
    import hashlib
    import inspect
    layout = load_upstream_osm().OSMLayout
    changed = [name for name, digest in PINNED_UPSTREAM_METHODS.items()
               if hashlib.sha256(inspect.getsource(getattr(layout, name)).encode()).hexdigest() != digest]
    where = "selfdrive/ui/sunnypilot/layouts/settings/osm.py"
    action = "Upstream Sync & Watch must review the override in osm_multi.py (OSMLayoutMulti) against the new upstream code, " + \
             "then update PINNED_UPSTREAM_METHODS"
    assert not changed, f"upstream OSMLayout changed in {', '.join(changed)} ({where}): {action}"

  def test_every_override_is_pinned(self):
    import ast
    tree = ast.parse(MULTI_PY.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OSMLayoutMulti")
    overridden = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert overridden <= set(PINNED_UPSTREAM_METHODS), overridden - set(PINNED_UPSTREAM_METHODS)
