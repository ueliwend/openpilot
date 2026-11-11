"""The selected map locations (Inster port): `OsmSelectedLocations` is the only source of truth.

Value: a JSON list `[{"kind": "nation", "code": "CH", "title": "Switzerland"}, {"kind": "state", "code": "CA", ...}]`,
in the order the locations were added. Any location of mapd v1.12.0's lists can be selected (vendored in
bounding_boxes/). The key has the BACKUP flag (owner, 11 Nov 2026), and sunnylink may write it, so every read is
validated: unknown codes, malformed and repeated entries are dropped and the clean list is written back; the title
always comes from the vendored list.

Migration: once, when the key has never been written (its file does not exist: the first start with this software). It
is seeded from the single-location keys of the old screen (`OsmLocationName` / `OsmStateName`), so the country chosen
before stays selected. The marker is the key's own file, not a separate key: every read writes the list back, so the
file exists from the first start on, nothing else has to be kept in step, and DELETE ALL writes an empty list.

An unreadable value (e.g. a bad remote write) is never migrated again from the one-entry mirror. It is replaced by the
last valid list in memory, or after a restart by the list of the last status (`OsmLocationStatus` "order", written by
mapd_manager whenever the list changes); only if neither exists, by the mirror (better one entry than none).

Mirror: the old keys (`OsmLocationName/Title`, `OsmStateName/Title`, `OsmLocal`) stay for upstream code and are written
only here, from the first entry of the list (`OsmLocal` = the list is not empty). A remote write to them is overwritten.
"""
import os

from openpilot.sunnypilot.mapd.offline_regions import entry_id, location_title, parse_entry_id

KEY = "OsmSelectedLocations"
STATUS_KEY = "OsmLocationStatus"
MIRROR_KEYS = ("OsmLocationName", "OsmLocationTitle", "OsmStateName", "OsmStateTitle", "OsmLocal")


def make_entry(kind: str, code: str) -> dict:
  return {"kind": kind, "code": code, "title": location_title(kind, code)}


def entry_from_id(location_id: str) -> dict | None:
  parsed = parse_entry_id(location_id)
  return make_entry(*parsed) if parsed else None


def id_of(entry: dict) -> str:
  return entry_id(entry["kind"], entry["code"])


def normalize(value) -> list[dict] | None:
  """The clean list, or None if the value is not a list at all."""
  if not isinstance(value, list):
    return None
  out, seen = [], set()
  for item in value:
    if not isinstance(item, dict) or item.get("kind") not in ("nation", "state") or not isinstance(item.get("code"), str):
      continue
    entry = entry_from_id(entry_id(item["kind"], item["code"]))
    if entry is None or id_of(entry) in seen:
      continue
    seen.add(id_of(entry))
    out.append(entry)
  return out


def migrate(params) -> list[dict]:
  """The list the old single-location keys describe (US with a state = that state; US with "All" = all of the US)."""
  country = params.get("OsmLocationName")
  if not country:
    return []
  if country == "US":
    state = params.get("OsmStateName") or "All"
    if state.lower() != "all":
      entry = entry_from_id(f"US:{state}")
      return [entry] if entry else []
  entry = entry_from_id(country) if not country.startswith("US:") else None
  return [entry] if entry else []


def mirror_values(selection: list[dict]) -> dict:
  if not selection:
    return {"OsmLocationName": None, "OsmLocationTitle": None, "OsmStateName": None, "OsmStateTitle": None, "OsmLocal": False}
  first = selection[0]
  if first["kind"] == "state":
    return {"OsmLocationName": "US", "OsmLocationTitle": location_title("nation", "US"), "OsmStateName": first["code"],
            "OsmStateTitle": first["title"], "OsmLocal": True}
  return {"OsmLocationName": first["code"], "OsmLocationTitle": first["title"], "OsmStateName": None, "OsmStateTitle": None,
          "OsmLocal": True}


class Selection:
  def __init__(self, params):
    self.params = params
    self._last_valid: list[dict] | None = None

  def read(self) -> list[dict]:
    """The validated list (written back if it had to be cleaned), migrating once; mirrors the old keys."""
    raw = self.params.get(KEY)
    selection = normalize(raw)
    if selection is None:
      if self._last_valid is not None:
        selection = self._last_valid
      elif not os.path.exists(self.params.get_param_path(KEY)):
        selection = migrate(self.params)  # never written: the first start with this software
      else:
        selection = self._recover()
      self.params.put(KEY, selection, block=True)
    elif selection != raw:
      self.params.put(KEY, selection, block=True)
    self._last_valid = selection
    self._mirror(selection)
    return list(selection)

  def _recover(self) -> list[dict]:
    """The list of the last status (written by mapd_manager after every change), else the mirror."""
    status = self.params.get(STATUS_KEY)
    order = status.get("order") if isinstance(status, dict) else None
    if isinstance(order, list):
      return normalize([entry_from_id(i) for i in order if entry_from_id(i)]) or []
    return migrate(self.params)

  def write(self, selection: list[dict]) -> None:
    selection = normalize(selection) or []
    self.params.put(KEY, selection, block=True)
    self._last_valid = selection
    self._mirror(selection)

  def add(self, location_id: str) -> dict | None:
    entry = entry_from_id(location_id)
    if entry is None:
      return None
    selection = self.read()
    if id_of(entry) not in [id_of(e) for e in selection]:
      self.write(selection + [entry])
    return entry

  def remove(self, location_id: str) -> bool:
    selection = self.read()
    kept = [e for e in selection if id_of(e) != location_id]
    if len(kept) != len(selection):
      self.write(kept)
      return True
    return False

  def _mirror(self, selection: list[dict]) -> None:
    for key, value in mirror_values(selection).items():
      if key == "OsmLocal":
        if self.params.get_bool(key) != value:
          self.params.put_bool(key, value, block=True)
      elif value is None:
        if self.params.get(key) is not None:
          self.params.remove(key)
      elif self.params.get(key) != value:
        self.params.put(key, value, block=True)
