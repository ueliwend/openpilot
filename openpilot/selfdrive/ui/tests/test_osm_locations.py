"""Offline maps screen for several countries (E9): texts (osm_locations_model.py), the rows' wiring (osm_locations.py),
the widgets (osm_widgets.py) and the subclass of upstream's layout (osm_multi.py), loaded with small stand-ins for the
widgets and raylib, so neither a window nor raylib is needed. Every text is checked against the glyphs the UI loads."""
import ast
import datetime
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest import mock

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.sunnypilot.layouts.settings import osm_locations_model as model
from openpilot.sunnypilot.mapd.offline_storage import Budget, DeleterReplay, Plan, TileEstimate

UI = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
LAYOUT_PY = UI / "sunnypilot/layouts/settings/osm_locations.py"
WIDGETS_PY = UI / "sunnypilot/layouts/settings/osm_widgets.py"
MULTI_PY = UI / "sunnypilot/layouts/settings/osm_multi.py"
SETTINGS_PY = UI / "sunnypilot/layouts/settings/settings.py"
OSM_PY = UI / "sunnypilot/layouts/settings/osm.py"
MODEL_PY = UI / "sunnypilot/layouts/settings/osm_locations_model.py"
APPLICATION_PY = REPO / "system/ui/lib/application.py"
MAPS_PY = REPO / "sunnypilot/mapd/map_downloader.py"  # its message reaches the screen
NOV2 = datetime.datetime(2026, 11, 2, 12).timestamp()
NOV3 = datetime.datetime(2026, 11, 3, 9).timestamp()
GB = 1_000_000_000
ASCII = {chr(c) for c in range(32, 127)}


def status(**entries):
  run = entries.pop("run", {"active": False})
  return {"version": 1, "order": list(entries), "entries": entries, "run": run, "other": {"tiles": 0, "bytes": 0},
          "storage": {"maps": int(2.1 * GB), "available": 81 * GB, "buffer_hours": 5.3}, "message": ""}


def entry(title, state, **kw):
  return {"title": title, "state": state, "bytes": int(1.4 * GB), "date": NOV2, "tiles_total": 10, "tiles_expected": 10,
          "tiles_present": 10, "exclusive_bytes": int(0.1 * GB), "shared_tiles": 0, "shared_with": [], "last": None} | kw


def tiles(on_disk=0, new=int(0.9 * GB), download=int(0.45 * GB), unknown=0):
  return TileEstimate(on_disk=on_disk, new=new, download=download, unknown=unknown)


def plan(ok=True, shortfall=0, minutes=0, moved=False, lost=0, unknown=0, est=None):
  b = Budget(ok, 95 * GB, 10 * GB, int(9.5 * GB), 7 * GB, 60 * GB, GB, 2 * GB, GB, shortfall, 30 * GB, 5.0)
  r = DeleterReplay(needed=shortfall, minutes=minutes, folders=minutes, first_written=NOV2, last_written=NOV3, moved=moved,
                    lost_minutes=lost, lost_first_written=NOV2, lost_last_written=NOV2)
  return Plan(b, est or tiles(unknown=unknown), r)


def batch(per, ok=True, shortfall=0, minutes=0, queued=()):
  total = TileEstimate(on_disk=sum(e.on_disk for _, _, e in per), new=sum(e.new for _, _, e in per),
                       download=sum(e.download for _, _, e in per), unknown=sum(e.unknown for _, _, e in per))
  return model.BatchPlan(per=per, total=total, plan=plan(ok=ok, shortfall=shortfall, minutes=minutes, est=total), queued=list(queued))


class TestTexts(OpenpilotTestCase):
  def test_row_values(self):
    run = {"active": True, "tiles_done": 4, "tiles_total": 10}
    assert model.entry_value(entry("Austria", "ok"), {}) == "1.4 GB | 2 Nov"
    assert model.entry_value(entry("Austria", "downloading"), run) == "4/10 downloading"
    assert model.entry_value(entry("Austria", "queued"), run) == "queued"
    assert model.entry_value(entry("Austria", "not_downloaded", tiles_present=0, bytes=0), {}) == "not downloaded"
    assert model.entry_value(entry("Austria", "partial", tiles_present=8), {}) == "partial 8/10 | 1.4 GB"
    assert model.entry_value(entry("Austria", "partial", last={"refreshed": 7, "expected": 10}), {}) == "partly updated 7/10 | 1.4 GB | 2 Nov"
    assert model.entry_value(entry("Austria", "failed"), {}) == "failed - data from 2 Nov kept"
    assert model.entry_value(entry("Austria", "ok", bytes=int(0.35 * GB)), {}) == "350 MB | 2 Nov"
    assert model.entry_value(entry("Austria", "ok", date=3600), {}) == "1.4 GB | date unknown"  # unset clock

  def test_days(self):
    assert model.day_range_text(NOV2, NOV3) == "2-3 Nov"
    assert model.day_range_text(NOV2, NOV2 + 3600) == "2 Nov"
    assert model.day_range_text(datetime.datetime(2026, 10, 30).timestamp(), NOV2) == "30 Oct - 2 Nov"
    assert model.day_range_text(0, NOV2) == "date unknown"

  def test_single_country_confirmation(self):
    one = batch([("AT", "Austria", tiles())])
    assert model.batch_text(["Austria"], one) == "Download Austria? About 900 MB on the device, 450 MB to download."
    one = batch([("AT", "Austria", tiles())], shortfall=GB, minutes=7)
    assert model.batch_text(["Austria"], one).endswith("\nFrees space by deleting the oldest 7 min of drive recordings (last written 2-3 Nov).")
    assert model.batch_text(["Austria"], batch([("AT", "Austria", tiles(unknown=1))])).startswith("Download Austria? At least ")
    assert "size is not known" in model.batch_text(["Austria"], None)

  def test_batch_confirmation_lists_each_country_and_the_total(self):
    b = batch([("AT", "Austria", tiles(new=int(1.5 * GB), download=int(0.386 * GB))),
               ("DE", "Germany", tiles(on_disk=int(0.2 * GB), new=int(2.3 * GB), download=int(0.6 * GB)))], shortfall=GB, minutes=27)
    assert model.batch_text(["Austria", "Germany"], b).split("\n") == [
      "Download 2 countries?",
      "Austria: about 1.5 GB on the device, 386 MB to download",
      "Germany: about 2.5 GB on the device, 600 MB to download",
      "Total: about 4.0 GB on the device, 986 MB to download.",
      "Frees space by deleting the oldest 27 min of drive recordings (last written 2-3 Nov).",
    ]
    refused = batch([("AT", "Austria", tiles()), ("DE", "Germany", tiles())], ok=False, queued=["France"])
    assert model.batch_text(["Austria", "Germany"], refused) == \
      "Not enough storage for Austria, Germany (with France already queued): maps may use 30.0 GB in all to keep 3 h of full drive recording."

  def test_update_all_confirmation(self):
    assert model.download_text(["Austria"], plan()) == "Download Austria? About 900 MB on the device, 450 MB to download."
    text = model.download_text(["Austria"], plan(shortfall=GB, minutes=7, moved=True, lost=2))
    assert "Moves the oldest 7 min of drive recordings to the external drive (last written 2-3 Nov)." in text
    assert text.endswith("The external drive deletes its oldest 2 min (last written 2 Nov).")
    assert model.download_text(["Austria"], plan(ok=False)).startswith("Not enough storage for Austria: maps may use 30.0 GB in all to keep 3 h")

  def test_delete_and_storage_texts(self):
    e = entry("Austria", "ok", shared_tiles=2, shared_with=["Switzerland", "Germany"])
    assert model.delete_text(e) == "Delete the maps of Austria? Frees 100 MB. 2 map tiles shared with Switzerland, Germany stay."
    assert model.delete_text(entry("Austria", "ok"), running=True) == \
      "Delete the maps of Austria? Frees 100 MB. A download is running: the files go when it ends."
    assert model.storage_text(status()) == "free 81.0 GB | drive buffer about 5.3 h"  # the maps' size is on Downloaded Maps
    s = status()
    s["other"] = {"tiles": 3, "bytes": int(0.4 * GB)}
    assert model.other_text(s) == "400 MB" and model.other_text(status()) == ""

  def test_progress_shows_the_whole_queue(self):
    s = status(CH=entry("Switzerland", "ok"), DE=entry("Germany", "downloading"),
               run={"active": True, "current": "DE", "index": 2, "count": 2, "tiles_done": 7, "tiles_total": 30, "waiting": "",
                    "queue_tiles_done": 13, "queue_tiles_total": 36})
    assert model.progress_text(s) == "36% | Germany 7/30 | country 2 of 2" and abs(model.queue_fraction(s) - 13 / 36) < 1e-9
    s["run"] = {"active": True, "current": "AT", "tiles_done": 1, "tiles_total": 10, "queue_tiles_done": 1, "queue_tiles_total": 10}
    assert model.progress_text(s) == "10% | Austria 1/10 | country 1 of 1"  # a deleted country still downloading: title from the list
    s["run"] = {"active": True, "current": None, "waiting": "onroad"}
    assert model.progress_text(s) == "waiting until the car is off" and model.queue_fraction(s) == 0.0
    s["run"] = {"active": True, "current": None, "waiting": "waiting for an unmetered network"}
    assert model.progress_text(s) == "waiting for Wi-Fi or another unmetered network"
    s["run"] = {"active": True, "current": None, "waiting": "checking the map server", "check": {"location": "AT", "done": 3, "total": 8}}
    assert model.progress_text(s) == "checking the map server (3/8)"
    assert model.progress_text(status()) == ""

  def test_server_error_shows_its_code_and_allows_a_retry(self):
    retry_at = datetime.datetime(2026, 11, 12, 7, 41).timestamp()
    s = status(CH=entry("Switzerland", "queued"),
               run={"active": True, "current": None, "waiting": "map server error (HTTP 403)", "server_error": 403, "retry_at": retry_at})
    assert model.progress_text(s) == "map server error (HTTP 403) | next try 07:41" and model.can_retry(s)
    s["run"] = {"active": True, "current": None, "waiting": "map server too slow", "retry_at": retry_at}
    assert model.progress_text(s) == "map server too slow | next try 07:41" and model.can_retry(s)
    s["run"] = {"active": True, "current": None, "waiting": "onroad"}
    assert not model.can_retry(s)
    s["run"] = {"active": True, "current": "CH", "waiting": ""}
    assert not model.can_retry(s)

  def test_needs_download_queued_and_pending(self):
    s = status(CH=entry("Switzerland", "ok"), AT=entry("Austria", "not_downloaded"), DE=entry("Germany", "failed"),
               FR=entry("France", "queued"), IT=entry("Italy", "downloading"), run={"active": True, "deletes_pending": ["SI", "OTHER"]})
    assert [e["id"] for e in model.needs_download(s)] == ["AT", "DE"]
    assert model.queued_ids(s) == ["FR", "IT"] and model.pending_deletes(s) == ["SI"] and model.title_of("SI") == "Slovenia"
    codes = dict(model.country_options())
    assert {"CH", "AT", "DE", "IT", "FR", "SI", "ES", "US"} <= set(codes) and codes["SI"] == "Slovenia"
    assert model.entries(None) == [] and model.entries({"entries": "bad"}) == []


class TestEstimate(OpenpilotTestCase):
  def _root(self):
    import os
    import tempfile
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "offline", "44", "6"))
    with open(os.path.join(root, "offline", "44", "6", "a.area"), "wb") as f:
      f.write(b"x" * 1000)
    return root

  def test_estimate_asks_the_server_for_unknown_tiles_only(self):
    from openpilot.sunnypilot.mapd.offline_server import NO_ANSWER_SIZE
    root, asked = self._root(), []
    p = model.estimate(root, ["CH"], types.SimpleNamespace(head=lambda t: asked.append(t) or 250), log_root=root)
    assert (44, 6) not in asked and len(asked) == 5
    assert p.estimate.on_disk == 1000 and p.estimate.download == 1000 // 4 + 5 * 250
    assert p.estimate.new == 5 * 1000 + 1000  # E11: a tile on disk is staged next to its live copy until the next start
    asked.clear()
    assert model.estimate(root, ["CH"], types.SimpleNamespace(head=lambda t: asked.append(t) or NO_ANSWER_SIZE), log_root=root) is None
    assert len(asked) == 1  # stops asking once the server does not answer

  def test_batch_estimate_per_country_total_and_queued(self):
    from openpilot.sunnypilot.mapd.offline_regions import entry_tiles, union_tiles
    root = self._root()
    b = model.estimate_batch(root, ["CH", "AT"], ["DE"], types.SimpleNamespace(head=lambda t: 250), log_root=root)
    (ch_id, ch_title, ch), (at_id, at_title, at) = b.per
    assert (ch_id, ch_title, at_title) == ("CH", "Switzerland", "Austria")
    assert ch.on_disk == 1000 and ch.new == 5 * 1000 and at.new == len(entry_tiles("AT")) * 1000
    shared = len(set(entry_tiles("CH")) & set(entry_tiles("AT")))
    assert b.total.new == (5 + len(entry_tiles("AT")) - shared) * 1000  # shared tiles counted once in the total
    assert b.plan.estimate.new == (len(union_tiles(["CH", "AT", "DE"])) - 1) * 1000  # the guard includes the queued country
    assert b.queued == ["Germany"]


# ---- every text against the glyphs the UI loads (system/ui/lib/application.py _load_fonts: ASCII 32-126, EXTRA_FONT_CHARS,
# the characters of the translation files; anything else is drawn as "?", as on the car with 7166f13)
def loaded_glyphs() -> set[str]:
  tree = ast.parse(APPLICATION_PY.read_text())
  extra = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
               and any(getattr(t, "id", "") == "EXTRA_FONT_CHARS" for t in node.targets))
  src = APPLICATION_PY.read_text()
  assert "set(map(chr, range(32, 127))) | set(EXTRA_FONT_CHARS)" in src, "the UI's glyph set changed: re-check this test"
  return ASCII | set(extra)


def screen_strings(path: Path) -> list[str]:
  """String constants of a module that can reach the screen (docstrings left out)."""
  tree = ast.parse(path.read_text())
  docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
          and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
  return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def not_drawable(texts) -> list[tuple[str, list[str]]]:
  allowed = ASCII | {"\n"}  # our rule: printable ASCII (inside every glyph set the UI loads); "\n" is a line break, as upstream's dialogs use
  return [(t, sorted(set(t) - allowed)) for t in texts if set(t) - allowed]


class TestGlyphs(OpenpilotTestCase):
  def test_our_rule_is_inside_the_loaded_glyphs(self):
    assert ASCII <= loaded_glyphs()
    assert "\u2248" not in loaded_glyphs() and "\u00b7" not in loaded_glyphs()  # the "?" on the car: approx sign, middle dot

  def test_every_literal_of_the_screen_modules(self):
    texts = []
    for path in (LAYOUT_PY, MODEL_PY, WIDGETS_PY, MULTI_PY, MAPS_PY):
      texts += screen_strings(path)
    assert not not_drawable(texts), not_drawable(texts)

  def test_every_country_and_state_name(self):
    names = []
    for kind in ("nation", "us_states"):
      data = json.loads((REPO / f"sunnypilot/mapd/bounding_boxes/{kind}_bounding_boxes.json").read_text())
      names += [v["full_name"] for v in data.values()]
    assert not not_drawable(names), not_drawable(names)

  def test_every_text_the_model_produces(self):
    retry = datetime.datetime(2026, 11, 12, 7, 41).timestamp()
    out = []
    states = ["ok", "queued", "downloading", "not_downloaded", "partial", "failed"]
    for st in states:
      for e in (entry("Austria", st), entry("Austria", st, tiles_present=3, last={"refreshed": 2, "expected": 10}), entry("Austria", st, date=0)):
        out.append(model.entry_value(e, {"tiles_done": 3, "tiles_total": 10}))
        out += [model.delete_text(e), model.delete_text(dict(e, shared_tiles=2, shared_with=["Switzerland"]), running=True)]
    runs = [{"active": True, "current": "DE", "index": 2, "count": 3, "tiles_done": 7, "tiles_total": 30, "queue_tiles_done": 9, "queue_tiles_total": 40}]
    runs += [{"active": True, "current": None, "waiting": w, "retry_at": retry} for w in
             list(model.WAIT_TEXTS) + ["map server error (HTTP 403)", "map server not reachable", "map server too slow", "not enough storage"]]
    runs += [{"active": True, "current": None, "waiting": "map server error (HTTP 403)", "server_error": 403, "retry_at": retry},
             {"active": True, "current": None, "waiting": "checking the map server", "check": {"done": 2, "total": 9}}]
    for run in runs:
      s = status(DE=entry("Germany", "downloading"), run=run)
      out.append(model.progress_text(s))
    s = status()
    s["other"] = {"bytes": 5 * GB}
    out += [model.storage_text(s), model.other_text(s), model.other_delete_text(s), model.size_text(1), model.size_text(3 * GB)]
    for ok in (True, False):
      for p in (plan(ok=ok), plan(ok=ok, shortfall=GB, minutes=3), plan(ok=ok, shortfall=GB, minutes=3, moved=True, lost=1), plan(ok=ok, unknown=2)):
        out.append(model.download_text(["Austria", "Germany"], p))
      out.append(model.batch_text(["Austria", "Germany"], batch([("AT", "Austria", tiles()), ("DE", "Germany", tiles(unknown=1))],
                                                                 ok=ok, shortfall=GB, minutes=5, queued=["France"])))
    out.append(model.batch_text(["Austria"], None))
    assert not not_drawable(out), not_drawable(out)


# ---- the rows and widgets, with stand-ins
class Action:
  def __init__(self, text="", enabled=True, width=None):
    self.text, self.value, self.enabled = text, "", enabled
    self.progress, self.show_progress = 0.0, False

  def set_text(self, text):
    self.text = text

  def set_value(self, value):
    self.value = value

  def set_enabled(self, enabled):
    self.enabled = enabled

  def update(self, progress, text, show_progress=False):
    self.progress, self.value, self.show_progress = progress, text, show_progress


class Item:
  def __init__(self, title="", action_item=None, callback=None, **kw):
    self.title, self.action_item, self.callback, self.visible = title, action_item or Action(), callback, True
    self.kw = kw

  def set_visible(self, visible):
    self.visible = visible

  def set_title(self, title):
    self.title = title


class FakeParams(dict):
  def get(self, key, *a, **k):
    return super().get(key)

  def get_bool(self, key, *a, **k):
    return bool(super().get(key))

  def put(self, key, value, *a, **k):
    self[key] = value


class Node(types.SimpleNamespace):
  def __init__(self, ref, data):
    super().__init__(ref=ref, data=data)


class TwoAction(Action):
  """ValueTwoButtonsAction as far as the panel uses it: UPDATE (left, the action's own text) and DELETE (right)."""
  def __init__(self, left_text, right_text, left_callback=None, right_callback=None, enabled=True):
    super().__init__(left_text, enabled)
    self.right_text, self.right_enabled = right_text, True
    self.left_callback, self.right_callback = left_callback, right_callback

  def set_left_enabled(self, enabled):
    self.enabled = enabled

  def set_right_enabled(self, enabled):
    self.right_enabled = enabled

  def set_right_text(self, text):
    self.right_text = text


class Dialog:
  def __init__(self, *args, callback=None, **kw):
    self.args, self.callback, self.kw = args, callback, kw
    self.on_exit, self.selected_refs = None, []


def load_panel():
  ui_state = types.SimpleNamespace(params=FakeParams())
  pushed = []
  stand_ins = {
    "openpilot.selfdrive.ui.ui_state": {"ui_state": ui_state},
    "openpilot.selfdrive.ui.sunnypilot.layouts.settings.osm_widgets": {"CountryPickDialog": Dialog, "QueueProgressAction": Action,
                                                                       "ValueTwoButtonsAction": TwoAction},
    "openpilot.system.ui.lib.application": {"gui_app": types.SimpleNamespace(push_widget=pushed.append)},
    "openpilot.system.ui.lib.multilang": {"tr": lambda s: s},
    "openpilot.system.ui.widgets": {"DialogResult": types.SimpleNamespace(CONFIRM=1, CANCEL=0)},
    "openpilot.system.ui.widgets.confirm_dialog": {"ConfirmDialog": Dialog},
    "openpilot.system.ui.widgets.list_view": {"text_item": lambda title, value: Item(title, Action(value)), "ListItem": Item},
    "openpilot.system.ui.sunnypilot.lib.utils": {"NoElideButtonAction": Action},
    "openpilot.system.ui.sunnypilot.widgets.list_view": {"ListItemSP": Item, "toggle_item_sp": lambda title, description, initial_state=False,
                                                         param=None: Item(title, Action(), param=param, description=description, state=initial_state)},
    "openpilot.system.ui.sunnypilot.widgets.tree_dialog": {"TreeFolder": lambda folder, nodes: nodes, "TreeNode": Node},
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


RUNNING = {"active": True, "current": "DE", "index": 1, "count": 1, "tiles_done": 7, "tiles_total": 30, "queue_tiles_done": 7,
           "queue_tiles_total": 30, "deletes_pending": []}


class TestRows(OpenpilotTestCase):
  def setup_method(self):
    self.module, self.params, self.pushed = load_panel()
    self.mem = FakeParams()
    self.mono = [100.0]
    self.panel = self.module.OsmLocationsPanel(server=types.SimpleNamespace(head=lambda t: None), mem_params=self.mem,
                                               monotonic=lambda: self.mono[0])
    self.scroller = Scroller()
    self.asked = []
    # the estimates run in a thread in the app; here at once, with fixed results
    patches = [
      mock.patch.object(self.module.model, "estimate", lambda root, ids, server, **kw: self.asked.append(ids) or plan()),
      mock.patch.object(self.module.model, "estimate_batch", lambda root, ids, queued, server, **kw: self.asked.append((ids, queued)) or
                        batch([(i, model.title_of(i), tiles()) for i in ids], queued=[model.title_of(q) for q in queued])),
      mock.patch.object(self.module.threading, "Thread", lambda target, daemon: types.SimpleNamespace(start=target)),
    ]
    for p in patches:
      p.start()
      self.addCleanup(p.stop)

  def confirm_last(self):
    self.pushed[-1].callback(1)

  def pick(self, *refs):
    dialog = self.pushed[-1]
    dialog.selected_refs = list(refs)
    dialog.on_exit(1)

  def texts(self):
    """Every text the rows show now (titles, values, button texts) and every dialog text so far."""
    out = []
    for item in self.panel.items + self.panel.rows:
      out += [str(item.title), str(item.action_item.text), str(item.action_item.value), str(getattr(item.action_item, "right_text", ""))]
    out += [a for d in self.pushed for a in d.args if isinstance(a, str)]
    return out

  def test_one_row_per_country_with_delete(self):
    self.params[model.STATUS_KEY] = status(CH=entry("Switzerland", "ok"), AT=entry("Austria", "not_downloaded", tiles_present=0))
    assert self.panel.update(self.scroller) is False
    assert [r.title for r in self.scroller.added] == ["Switzerland", "Austria"]
    row = self.scroller.added[0]
    assert row.action_item.value == "1.4 GB | 2 Nov" and (row.action_item.text, row.action_item.right_text) == ("UPDATE", "DELETE")
    assert self.panel._missing_btn.visible and self.panel._missing_btn.action_item.value == "Austria"
    self.scroller.added[1].action_item.right_callback()
    assert "Delete the maps of Austria?" in self.pushed[-1].args[0]
    self.confirm_last()
    assert self.params[model.DELETE_KEY] == "AT"
    self.params[model.STATUS_KEY] = status(CH=entry("Switzerland", "ok"))  # fewer rows: the spare one is hidden, not removed
    self.panel.update(self.scroller)
    assert [r.visible for r in self.scroller.added] == [True, False]
    assert not not_drawable(self.texts())

  def test_add_several_countries_with_one_confirmation(self):
    self.params[model.STATUS_KEY] = status(CH=entry("Switzerland", "ok"))
    self.panel.update(self.scroller)
    self.panel._add_btn.callback()
    dialog = self.pushed[-1]
    assert dialog.args[0] == "Select Countries" and "CH" not in [n.ref for n in dialog.args[1][0]]  # already selected: not offered
    self.pick("SI", "ES", "AT")
    assert self.asked == [(["SI", "ES", "AT"], [])]
    lines = self.pushed[-1].args[0].split("\n")
    assert lines[0] == "Download 3 countries?" and lines[1].startswith("Slovenia: about") and lines[-1].startswith("Total: about")
    self.confirm_last()
    assert self.params[model.REQUEST_KEY] == "SI,ES,AT"
    assert not not_drawable(self.texts())

  def test_add_during_a_download_extends_the_queue_and_is_checked_with_it(self):
    self.params[model.STATUS_KEY] = status(CH=entry("Switzerland", "ok"), DE=entry("Germany", "downloading"), FR=entry("France", "queued"),
                                           run=RUNNING)
    assert self.panel.update(self.scroller) is True
    assert self.panel._add_btn.action_item.enabled  # ADD stays available
    assert self.panel._progress.visible and self.panel._progress.action_item.show_progress
    assert self.panel._progress.action_item.value == "23% | Germany 7/30 | country 1 of 1"
    assert abs(self.panel._progress.action_item.progress - 100 * 7 / 30) < 1e-6  # the bar: the whole queue
    self.panel._add_btn.callback()
    self.pick("AT")
    assert self.asked == [(["AT"], ["DE", "FR"])]
    self.params[model.REQUEST_KEY] = "IT"  # a request mapd_manager has not read yet is kept
    self.confirm_last()
    assert self.params[model.REQUEST_KEY] == "IT,AT"

  def test_delete_during_a_download_is_queued_and_shown(self):
    self.params[model.STATUS_KEY] = status(CH=entry("Switzerland", "ok"), AT=entry("Austria", "ok"), DE=entry("Germany", "downloading"),
                                           run=RUNNING)
    self.panel.update(self.scroller)
    assert all(r.action_item.right_enabled for r in self.scroller.added)  # DELETE works during a download
    assert self.panel._update_all_btn.action_item.enabled  # E11: requests are queued (requests.json), never refused
    self.scroller.added[1].action_item.right_callback()
    assert self.pushed[-1].args[0].startswith("Delete the maps of Austria?")
    self.confirm_last()
    assert self.params[model.DELETE_KEY] == "AT"
    # mapd_manager took it: Austria left the list, its tiles wait for the download's end
    self.params[model.STATUS_KEY] = status(CH=entry("Switzerland", "ok"), DE=entry("Germany", "downloading"), run=RUNNING | {"deletes_pending": ["AT"]})
    self.panel.update(self.scroller)
    row = self.scroller.added[2]
    assert (row.title, row.action_item.value, row.action_item.right_text, row.action_item.right_enabled) == ("Austria", "delete queued", "QUEUED", False)
    row.action_item.right_callback()
    assert len(self.pushed) == 1  # a queued row asks nothing
    assert not not_drawable(self.texts())

  def test_us_state_path_with_other_countries(self):
    self.panel._add()
    self.pick("AT", "US")
    assert self.pushed[-1].args[0] == "Select States"
    self.pick("CA", "NV")
    self.confirm_last()
    assert self.params[model.REQUEST_KEY] == "AT,US:CA,US:NV"
    self.params.clear()
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
    with mock.patch.object(self.module.model, "estimate_batch", lambda root, ids, queued, server, **kw: batch([("FR", "France", tiles())], ok=False)):
      self.panel._add()
      self.pick("FR")
    assert self.pushed[-1].args[0].startswith("Not enough storage") and self.pushed[-1].args[1] == "OK"
    assert model.REQUEST_KEY not in self.params

  def test_update_all_refused_by_storage_offers_no_download(self):
    self.params[model.STATUS_KEY] = status(CH=entry("Switzerland", "ok"))
    self.panel.update(self.scroller)
    with mock.patch.object(self.module.model, "estimate", lambda root, ids, server, **kw: plan(ok=False)):
      self.panel._update_all_btn.callback()
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


# ---- E11: status version 2 (map_downloader), per-country UPDATE, the toggle, the heartbeat
NOV6, NOV13 = datetime.datetime(2026, 11, 6, 9).timestamp(), datetime.datetime(2026, 11, 13, 9).timestamp()


def status2(**entries):
  run = entries.pop("run", {"active": False, "staged": 0})
  storage = entries.pop("storage", {"maps": int(1.2 * GB), "staged": int(0.6 * GB), "available": int(53.9 * GB), "buffer_hours": 83.7,
                                    "buffer_hours_full": 7.3})
  return {"version": 2, "order": list(entries), "entries": entries, "run": run, "other": {"tiles": 0, "bytes": 0}, "storage": storage,
          "message": ""}


def entry2(title, state, **kw):
  return {"title": title, "state": state, "bytes": 512_000_000, "tiles_total": 6, "tiles_expected": 6, "tiles_present": 6, "tiles_staged": 0,
          "tiles_due": 0, "tiles_done": 6, "date_oldest": NOV6, "date_newest": NOV13, "exclusive_bytes": 0, "shared_tiles": 0,
          "shared_with": [], "kept_smaller": [], "staged": 0, "due": 0} | kw


class TestE11Texts(OpenpilotTestCase):
  def test_country_values(self):
    run = {"active": True, "waiting": ""}
    assert model.entry_value(entry2("Switzerland", "ok"), run) == "512 MB | 6-13 Nov"
    assert model.entry_value(entry2("Switzerland", "queued", tiles_due=6, tiles_done=0), run) == "update queued"
    assert model.entry_value(entry2("Germany", "queued", tiles_present=0, tiles_due=30), run) == "queued"
    assert model.entry_value(entry2("Switzerland", "downloading", tiles_due=3, tiles_done=3), run) == "downloading 3/6"
    assert model.entry_value(entry2("Switzerland", "staged", tiles_staged=4), run) == "4 of 6 updated | applies at next start"
    assert model.entry_value(entry2("Austria", "not_downloaded", tiles_present=0), run) == "not downloaded"
    kept = entry2("Germany", "ok", kept_smaller=[{"tile": "48/8", "pct": 35}])
    assert model.entry_value(kept, run) == "512 MB | 6-13 Nov | 1 tile kept - new map 35% smaller"
    waits = {"network": "waiting for Wi-Fi or SIM", "hot": "paused: device hot", "storage": "not enough storage",
             "memory": "paused: low memory"}
    for code, text in waits.items():
      assert model.entry_value(entry2("Switzerland", "queued", tiles_due=1), {"active": True, "waiting": code}) == "update queued | " + text
    retry = datetime.datetime(2026, 11, 13, 14, 5).timestamp()
    assert model.entry_value(entry2("Switzerland", "queued", tiles_due=1), {"active": True, "waiting": "server", "retry_at": retry}) == \
      "update queued | map server not reachable | next try 14:05"

  def test_version_1_still_shown(self):
    assert model.entry_value(entry("Austria", "ok"), {}) == "1.4 GB | 2 Nov"
    assert model.storage_text(status()) == "free 81.0 GB | drive buffer about 5.3 h"
    assert not model.downloader_stopped(status(), None, 1e6)  # E9 software has no heartbeat

  def test_storage_line_with_staged_and_both_buffers(self):
    """Version 2: the figures are in the memory param (they change while driving), not in the persistent status."""
    s = status2()
    live = {"heartbeat": 1.0, "storage": s.pop("storage")}
    assert model.storage_text(s, live) == "Maps 1.2 GB (+600 MB staged) | free 53.9 GB | drive buffer about 84 h (7.3 h with full recording)"
    live["storage"] = {"maps": GB, "staged": 0, "available": GB, "buffer_hours": 7.3, "buffer_hours_full": 7.3}
    assert model.storage_text(s, live) == "Maps 1.0 GB | free 1.0 GB | drive buffer about 7.3 h"
    assert model.storage_text(status2()) == ""  # version 2 without the memory param: nothing (never stale figures)

  def test_progress_row(self):
    s = status2(DE=entry2("Germany", "downloading", tiles_expected=30, tiles_done=12, tiles_due=18),
                run={"active": True, "current": "48/8", "current_location": "DE", "staged": 5, "waiting": ""})
    live = {"heartbeat": 99.0, "tile": "48/8", "done": 25, "total": 100, "cap": 1_000_000}
    assert model.progress_text(s, live) == "Germany 12/30 | 5 staged for next start | 1.0 MB/s while driving"
    assert model.progress_fraction(s, live) == 0.25
    assert model.progress_text(s, dict(live, cap=None)) == "Germany 12/30 | 5 staged for next start"
    s["run"] = {"active": True, "current": None, "waiting": "network", "staged": 0}
    assert model.progress_text(s, live) == "waiting for Wi-Fi or SIM"
    s["run"] = {"active": False, "staged": 3, "apply_failed": True}
    assert model.progress_text(s, live) == "map update could not be applied | 3 staged for next start" and model.show_progress(s, False)
    assert model.progress_text(s, live, stopped=True) == "map downloads stopped - restart the device"

  def test_heartbeat(self):
    s = status2()
    assert not model.downloader_stopped(s, {"heartbeat": 100.0}, 140.0)
    assert model.downloader_stopped(s, {"heartbeat": 100.0}, 100.0 + model.STALE_SECONDS + 1)
    assert not model.downloader_stopped(s, None, 60.0) and model.downloader_stopped(s, None, model.NO_HEARTBEAT_SECONDS + 1)

  def test_not_downloaded_only_without_tiles_and_nothing_due(self):
    s = status2(CH=entry2("Switzerland", "ok"), AT=entry2("Austria", "not_downloaded", tiles_present=0),
                DE=entry2("Germany", "queued", tiles_present=0, tiles_due=30))
    assert [e["id"] for e in model.needs_download(s)] == ["AT"]

  def test_every_e11_text_is_drawable(self):
    out = [model.AUTO_TITLE, model.AUTO_DESCRIPTION]
    for st in ("ok", "queued", "downloading", "staged", "not_downloaded"):
      for e in (entry2("Germany", st), entry2("Germany", st, tiles_present=0, kept_smaller=[{"tile": "1/2", "pct": 30}] * 2)):
        for w in list(model.WAIT_TEXTS) + ["server_error", ""]:
          out.append(model.entry_value(e, {"active": True, "waiting": w, "server_error": 403 if w == "server_error" else None}))
    s = status2(DE=entry2("Germany", "downloading"), run={"active": True, "current": "48/8", "current_location": "DE", "staged": 2,
                                                          "apply_failed": True})
    out += [model.progress_text(s, {"cap": 1e6}), model.progress_text(s, None, True), model.storage_text(s, {"storage": status2()["storage"]})]
    assert not not_drawable(out), not_drawable(out)


class TestE11Rows(TestRows):
  def test_toggle_is_the_auto_update_setting(self):
    assert self.panel._auto in self.panel.items and self.panel._auto.kw["param"] == "OsmAutoUpdate"
    assert self.panel._auto.title == "Update maps automatically" and "next start" in self.panel._auto.kw["description"]

  def test_per_country_update_writes_the_id_and_reads_queued_while_due(self):
    self.params[model.STATUS_KEY] = status2(CH=entry2("Switzerland", "ok"), AT=entry2("Austria", "ok"))
    self.panel.update(self.scroller)
    row = self.scroller.added[1]
    assert (row.action_item.text, row.action_item.enabled) == ("UPDATE", True)
    row.action_item.left_callback()
    assert self.params[model.REQUEST_KEY] == "AT" and not self.pushed  # no dialog: the request is queued at once
    self.params[model.STATUS_KEY] = status2(CH=entry2("Switzerland", "ok"), AT=entry2("Austria", "queued", tiles_due=10),
                                            run={"active": True, "waiting": "network", "staged": 0})
    self.panel.update(self.scroller)
    assert (row.action_item.text, row.action_item.enabled, row.action_item.right_enabled) == ("QUEUED", False, True)
    assert row.action_item.value == "update queued | waiting for Wi-Fi or SIM"
    assert self.panel._progress.visible and self.panel._progress.action_item.value == "waiting for Wi-Fi or SIM"
    assert not not_drawable(self.texts())

  def test_stopped_downloader_is_shown(self):
    self.params[model.STATUS_KEY] = status2(CH=entry2("Switzerland", "ok"))
    self.mem[model.LIVE_KEY] = {"heartbeat": 10.0}
    self.mono[0] = 200.0
    self.panel.update(self.scroller)
    assert self.panel._progress.visible and self.panel._progress.action_item.value == "map downloads stopped - restart the device"
    self.mem[model.LIVE_KEY] = {"heartbeat": 199.0}
    self.panel.update(self.scroller)
    assert not self.panel._progress.visible


# ---- the widgets (osm_widgets.py) on stand-ins for raylib and the upstream widgets
class Rect:
  def __init__(self, x, y, w, h):
    self.x, self.y, self.width, self.height = x, y, w, h


class StubTreeItem:
  def __init__(self, text, ref):
    self.text, self.ref, self.selected, self.style = text, ref, False, None

  def set_button_style(self, style):
    self.style = style


class StubTreeDialog:
  """TreeOptionDialog as far as the subclass relies on it: __init__ builds the rows, a tap calls _select_node, and each
  frame upstream sets every row's `selected` and style from its single `selection`."""
  def __init__(self, title, folders, current_ref="", **kw):
    self.folders, self.selection, self.selection_ref, self.current = folders, current_ref, current_ref, current_ref
    self._build_visible_items()

  def _build_visible_items(self, reset_scroll=True):
    self.visible_items = [StubTreeItem(n.data["display_name"], n.ref) for f in self.folders for n in f.nodes]

  def frame(self):
    for item in self.visible_items:
      item.selected = item.text == self.selection
      item.set_button_style("PRIMARY" if item.text == self.selection else "NORMAL")


class StubProgress:
  def __init__(self, width=600):
    self.width, self.progress, self.text, self.show_progress, self.text_color, self._font = width, 0.0, "", False, "GRAY", None


class StubButton:
  def __init__(self, text, click_callback=None, button_style=None, **kw):
    self.text, self.click_callback, self.style, self.enabled, self.rects = text, click_callback, button_style, True, []

  def set_text(self, text):
    self.text = text

  def set_enabled(self, enabled):
    self.enabled = enabled

  def set_touch_valid_callback(self, cb):
    pass

  def render(self, rect):
    self.rects.append((rect.x, rect.width))

  def tap(self):
    if self.enabled:
      self.click_callback()


class StubButtonActionSP:
  """sunnypilot's ButtonActionSP as far as ValueTwoButtonsAction relies on it (its button, pressed flag, text, value)."""
  def __init__(self, text, width=250, enabled=True):
    self._text, self._enabled, self._value, self._font, self._value_color = text, enabled, "", None, "GRAY"
    self._pressed = False

    def pressed():
      self._pressed = True
    self._button = StubButton(text, click_callback=pressed)

  text = property(lambda self: self._text)
  value = property(lambda self: self._value)
  enabled = property(lambda self: self._enabled)

  def set_text(self, text):
    self._text = text

  def set_value(self, value):
    self._value = value

  def set_enabled(self, enabled):
    self._enabled = enabled

  def set_touch_valid_callback(self, cb):
    pass


def load_widgets():
  drawn = []
  rl = types.SimpleNamespace(Rectangle=Rect, Vector2=lambda x, y: (x, y), Color=lambda *c: c, WHITE="WHITE",
                             draw_rectangle_rounded=lambda r, *a: drawn.append(("rect", r.x, r.width, a[-1])),
                             draw_text_ex=lambda font, text, pos, *a: drawn.append(("text", text, pos[0])))
  stand_ins = {
    "pyray": vars(rl),
    "openpilot.system.ui.lib.text_measure": {"measure_text_cached": lambda font, text, size: types.SimpleNamespace(x=10 * len(text), y=40)},
    "openpilot.system.ui.sunnypilot.widgets.progress_bar": {"ProgressBarAction": StubProgress},
    "openpilot.system.ui.sunnypilot.widgets.tree_dialog": {"TreeItemWidget": StubTreeItem, "TreeOptionDialog": StubTreeDialog},
    "openpilot.system.ui.widgets.button": {"ButtonStyle": types.SimpleNamespace(PRIMARY="PRIMARY", NORMAL="NORMAL", DANGER="DANGER"),
                                           "Button": StubButton},
    "openpilot.system.ui.lib.application": {"FontWeight": types.SimpleNamespace(NORMAL=0), "TextAlignment": types.SimpleNamespace(LEFT=0),
                                            "TextAlignmentVertical": types.SimpleNamespace(MIDDLE=0)},
    "openpilot.system.ui.sunnypilot.lib.styles": {"style": types.SimpleNamespace(ITEM_TEXT_FONT_SIZE=50)},
    "openpilot.system.ui.sunnypilot.widgets.list_view": {"ButtonActionSP": StubButtonActionSP},
    "openpilot.system.ui.widgets.label": {"gui_label": lambda rect, text, **kw: drawn.append(("label", text, rect.x, rect.width))},
    "openpilot.system.ui.widgets.list_view": {"BUTTON_BORDER_RADIUS": 50, "BUTTON_FONT_SIZE": 35, "BUTTON_FONT_WEIGHT": 1, "BUTTON_HEIGHT": 100,
                                              "BUTTON_WIDTH": 250, "TEXT_PADDING": 20, "_resolve_value": lambda v, d=None: v() if callable(v) else v},
  }
  modules = {}
  for name, attributes in stand_ins.items():
    modules[name] = types.ModuleType(name)
    vars(modules[name]).update(attributes)
  spec = importlib.util.spec_from_file_location("osm_widgets_under_test", WIDGETS_PY)
  module = importlib.util.module_from_spec(spec)
  with mock.patch.dict(sys.modules, modules):
    spec.loader.exec_module(module)
  return module, drawn


class TestWidgets(OpenpilotTestCase):
  def test_country_dialog_ticks_several(self):
    widgets, _ = load_widgets()
    nodes = [Node(c, {"display_name": n}) for c, n in (("AT", "Austria"), ("DE", "Germany"), ("SI", "Slovenia"))]
    d = widgets.CountryPickDialog("Select Countries", [types.SimpleNamespace(folder="", nodes=nodes)])
    assert d.selected_refs == [] and d.selection == d.current  # "Select" disabled
    rows = {i.ref: i for i in d.visible_items}
    d._select_node(nodes[2])
    d._select_node(nodes[0])
    d._select_node(nodes[1])
    d._select_node(nodes[1])  # a second tap unticks
    assert d.selected_refs == ["SI", "AT"] and d.selection != d.current  # "Select" enabled
    d.frame()  # upstream's per-frame single-choice pass does not undo the ticks
    assert [rows[c].selected for c in ("AT", "DE", "SI")] == [True, False, True]
    assert [rows[c].style for c in ("AT", "DE", "SI")] == ["PRIMARY", "NORMAL", "PRIMARY"]

  def test_queue_bar_spans_the_whole_action(self):
    widgets, drawn = load_widgets()
    bar = widgets.QueueProgressAction()
    assert bar.width == 1000
    bar.progress, bar.text, bar.show_progress = 25.0, "25% | Germany 7/30 | country 1 of 2", True
    bar._render(Rect(500, 0, 1000, 100))
    track, fill, text = drawn
    assert track[:3] == ("rect", 500, 1000) and fill[:2] == ("rect", 504) and abs(fill[2] - 0.25 * 992) < 1e-6
    assert text[1] == bar.text and abs(text[2] - (500 + (1000 - 10 * len(bar.text)) / 2)) < 1e-6  # centred on the bar
    drawn.clear()
    bar.show_progress, bar.text = False, "checking the map server (2/9)"
    bar._render(Rect(500, 0, 1000, 100))
    assert [d[0] for d in drawn] == ["text"]  # waiting: text only, right-aligned like a value


class TestValueTwoButtons(OpenpilotTestCase):
  def test_update_and_delete_each_call_their_own_callback(self):
    widgets, drawn = load_widgets()
    calls = []
    a = widgets.ValueTwoButtonsAction("UPDATE", "DELETE", left_callback=lambda: calls.append("update"), right_callback=lambda: calls.append("delete"))
    a.set_value("512 MB | 6-13 Nov")
    assert a._right_button.style == "DANGER" and a.get_width_hint() == 10 * len("512 MB | 6-13 Nov") + 2 * 250 + 20 + 20 + 1
    a._button.tap()
    assert a._render(Rect(0, 0, 1200, 100)) is False and calls == ["update"]  # the row's own callback is never used
    a._right_button.tap()
    a._render(Rect(0, 0, 1200, 100))
    assert calls == ["update", "delete"]
    assert a._right_button.rects[-1] == (950, 250) and a._button.rects[-1] == (680, 250)  # DELETE at the right edge, UPDATE left of it
    assert drawn[-1] == ("label", "512 MB | 6-13 Nov", 0, 660)  # the value left of both
    a.set_left_enabled(False)
    a.set_text("QUEUED")
    a._button.enabled = True
    a._pressed = True
    a._render(Rect(0, 0, 1200, 100))
    assert calls == ["update", "delete"] and a._button.text == "QUEUED" and not a._button.enabled


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
    with mock.patch.object(module.model, "map_size", lambda root: 512_000_000):
      layout = module.OSMLayoutMulti()
    assert layout._delete_maps_btn.action_item.value == "512 MB"  # the screen's units (upstream: 488.28 MB)
    assert [i.title for i in layout.items] == ["Mapd Version", "Downloaded Maps", "Storage", "Countries"]
    assert layout._delete_maps_btn.action_item.value.endswith("MB")
    layout._update_labels()
    assert layout._locations.updates == 1 and layout._delete_maps_btn.action_item.enabled and device.resets == 0
    layout._locations.running = True
    layout._update_labels()
    assert layout._delete_maps_btn.action_item.enabled and device.resets == 1  # E11: DELETE ALL cancels a running download
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


# ---- upstream widgets our osm_widgets.py subclasses rely on, pinned like S5 (sunnypilot tree_dialog.py, progress_bar.py
# as at openpilot inster 060895e). A change: Upstream Sync & Watch must review CountryPickDialog / QueueProgressAction.
TREE_PY = REPO / "system/ui/sunnypilot/widgets/tree_dialog.py"
BAR_PY = REPO / "system/ui/sunnypilot/widgets/progress_bar.py"
PINNED_UPSTREAM_WIDGETS = {
  ("TreeOptionDialog", "__init__"): "84180dd3f60dcf50529f8b2f2b3dc1130050411cf21aa4764160391d37dc6719",
  ("TreeOptionDialog", "_select_node"): "05f1fae2a9d74b141e84547650795cd3c7cd1528e46a973de38e2200e5e8d0f8",
  ("TreeOptionDialog", "_build_visible_items"): "516e1db8a54fd92c820bebc8c2655613d75f19b208d7d860affe072504ef1a46",
  ("TreeOptionDialog", "_render"): "2f104274e9d2db98a231399ea77e6fe373ce10c7dec4d09cf4f99c53882c1846",
  ("TreeItemWidget", "__init__"): "9a420bc82b91175ba69f569f741f04b9ed1103ab1a9b9537138e20974fc41aa9",
  ("TreeItemWidget", "_render"): "f2c34a9e24f0a6f795afc1ee6281c1f9a42d9bb8d68226bb052975c255d87beb",
  ("ProgressBarAction", "__init__"): "6b73a62a4c447305b4ddbdf0494f7637f76fff8bf168cfa099c2003aa3999cd8",
  ("ProgressBarAction", "update"): "89065d45f83b8935c6829c1b659352fca96dd82f25e93e589414bd9575bc8b8e",
}


def load_upstream(path: Path, module_names: list[str], attrs: dict):
  modules = {}
  for name in module_names:
    modules[name] = types.ModuleType(name)
    vars(modules[name]).update(attrs)
  spec = importlib.util.spec_from_file_location(f"upstream_{path.stem}_under_test", path)
  module = importlib.util.module_from_spec(spec)
  with mock.patch.dict(sys.modules, modules):
    spec.loader.exec_module(module)
  return module


def upstream_widget_classes() -> dict:
  base = type("Base", (), {})
  attrs = {"Params": None, "TextAlignment": None, "FontWeight": types.SimpleNamespace(MEDIUM=None, NORMAL=None), "tr": lambda s: s,
           "DialogResult": None, "Button": base, "ButtonStyle": None, "BUTTON_PRESSED_BACKGROUND_COLORS": None, "gui_label": None,
           "MultiOptionDialog": base, "style": None, "search_from_list": None, "draw_star": None, "InputDialogSP": None, "gui_app": None,
           "measure_text_cached": None, "ListItem": base, "ItemAction": base, "__getattr__": lambda name: None}  # other names: None
  tree = load_upstream(TREE_PY, ["pyray", "openpilot.common.params", "openpilot.system.ui.lib.application", "openpilot.system.ui.lib.multilang",
                                 "openpilot.system.ui.widgets", "openpilot.system.ui.widgets.button", "openpilot.system.ui.widgets.label",
                                 "openpilot.system.ui.widgets.option_dialog", "openpilot.system.ui.sunnypilot.lib.styles",
                                 "openpilot.system.ui.sunnypilot.widgets.helpers.fuzzy_search",
                                 "openpilot.system.ui.sunnypilot.widgets.helpers.star_icon", "openpilot.system.ui.sunnypilot.widgets.input_dialog"], attrs)
  bar = load_upstream(BAR_PY, ["pyray", "openpilot.system.ui.lib.application", "openpilot.system.ui.lib.text_measure",
                               "openpilot.system.ui.widgets.list_view"], attrs)
  return {"TreeOptionDialog": tree.TreeOptionDialog, "TreeItemWidget": tree.TreeItemWidget, "ProgressBarAction": bar.ProgressBarAction}


class TestUpstreamWidgetPins(OpenpilotTestCase):
  def test_upstream_widgets_the_subclasses_depend_on_are_unchanged(self):
    import hashlib
    import inspect
    classes = upstream_widget_classes()
    changed = [f"{c}.{m}" for (c, m), digest in PINNED_UPSTREAM_WIDGETS.items()
               if hashlib.sha256(inspect.getsource(getattr(classes[c], m)).encode()).hexdigest() != digest]
    action = "Upstream Sync & Watch must review CountryPickDialog / QueueProgressAction (osm_widgets.py), then update PINNED_UPSTREAM_WIDGETS"
    assert not changed, f"upstream widgets changed in {', '.join(changed)}: {action}"


# ---- E11: upstream widgets ValueTwoButtonsAction and the toggle row rely on (sunnypilot list_view.py, upstream list_view.py as
# at openpilot inster 5c06f5c). A change: Upstream Sync & Watch must review ValueTwoButtonsAction (osm_widgets.py) and the
# toggle row (osm_locations.py), then update PINNED_UPSTREAM_ROW_WIDGETS.
SP_LIST_PY = REPO / "system/ui/sunnypilot/widgets/list_view.py"
LIST_PY = REPO / "system/ui/widgets/list_view.py"
PINNED_UPSTREAM_ROW_WIDGETS = {
  'ButtonActionSP.__init__': 'da10ef2a9c4b4d9bfb573aa177f5c6b445c4c7730c57b99f7206e61f70ea39d2',
  'ButtonActionSP.set_value': '6a282bd3d318ab81a7f461807bab702b23c4c62edffb64ac44acc36ee6247b95',
  'ButtonActionSP._render': 'c70f48eb88294d9bc4bc34008ca28995fad921aa96d63a82197f9773441bdc26',
  'toggle_item_sp': 'be354b03394ce7efa5d13ef04846e9df2041e998cbfc1e019c1bd01296719e90',
  'ButtonAction.__init__': '8f12f464f976e7c7a5599936dc8d8272f892641b9e26238186f10590c777fff7',
  'ButtonAction.text': 'db572b77f6c612e14efb7218f723ca4899ae83a2bee045e73f03e4bc7b6fe31f',
  'ButtonAction.value': '83ebd33e7c830baa63825ccc6f26e8219ba82cf805ce9adb3ba3f3a68282daa6',
  'ButtonAction.set_text': '8f3f057b2119ead6c77c8e362126f7e949469a67df921cb3913e8563a1e8fc77',
}


def upstream_row_widgets() -> dict:
  import abc

  class Anything(abc.ABCMeta):  # every imported name: a class whose attributes are itself (bases, styles, constants alike)
    def __getattr__(cls, name):
      return cls

    def __call__(cls, *a, **k):
      return cls
  base = Anything("Base", (), {})
  attrs = {"__getattr__": lambda name: base}
  sp = load_upstream(SP_LIST_PY, ["pyray", "openpilot.common.params", "openpilot.system.ui.lib.application", "openpilot.system.ui.lib.text_measure",
                                  "openpilot.system.ui.sunnypilot.widgets.toggle", "openpilot.system.ui.widgets", "openpilot.system.ui.widgets.button",
                                  "openpilot.system.ui.widgets.label", "openpilot.system.ui.widgets.list_view",
                                  "openpilot.system.ui.widgets.scroller_tici", "openpilot.system.ui.sunnypilot.lib.styles",
                                  "openpilot.system.ui.sunnypilot.widgets.option_control"], attrs)
  up = load_upstream(LIST_PY, ["pyray", "openpilot.system.ui.lib.application", "openpilot.system.ui.lib.text_measure",
                               "openpilot.system.ui.lib.multilang", "openpilot.system.ui.widgets", "openpilot.system.ui.widgets.button",
                               "openpilot.system.ui.widgets.toggle", "openpilot.system.ui.widgets.label", "openpilot.system.ui.widgets.html_render",
                               "openpilot.system.ui.lib.wrap_text", "openpilot.system.ui.lib.scroll_panel"], attrs)
  return {"ButtonActionSP.__init__": sp.ButtonActionSP.__init__, "ButtonActionSP.set_value": sp.ButtonActionSP.set_value,
          "ButtonActionSP._render": sp.ButtonActionSP._render, "toggle_item_sp": sp.toggle_item_sp,
          "ButtonAction.__init__": up.ButtonAction.__init__, "ButtonAction.text": up.ButtonAction.text.fget,
          "ButtonAction.value": up.ButtonAction.value.fget, "ButtonAction.set_text": up.ButtonAction.set_text}


class TestUpstreamRowWidgetPins(OpenpilotTestCase):
  def test_upstream_row_widgets_unchanged(self):
    import hashlib
    import inspect
    got = {k: hashlib.sha256(inspect.getsource(v).encode()).hexdigest() for k, v in upstream_row_widgets().items()}
    changed = [k for k, d in PINNED_UPSTREAM_ROW_WIDGETS.items() if got.get(k) != d]
    assert set(got) == set(PINNED_UPSTREAM_ROW_WIDGETS), sorted(set(got) ^ set(PINNED_UPSTREAM_ROW_WIDGETS))
    assert not changed, f"upstream row widgets changed in {', '.join(changed)}: review ValueTwoButtonsAction / the toggle row, then re-pin"
