import hashlib
import os

from openpilot.sunnypilot.mapd.mapd_installer import VERSION
from openpilot.sunnypilot.mapd.offline_regions import BOX_FILES, BOXES_DIR, BOXES_VERSION, adjusted_bounds, tile_order, tiles_for_box

# sha256 of nation_bounding_boxes.json / us_states_bounding_boxes.json at pfeiferj/openpilot-mapd v1.12.0 (46cd71a),
# the files the binary embeds (download.go:24-38)
VENDORED_SHA256 = {
  "nation": "e4847e0eadb93fad77e2f036dd452b23a7922f1248f34016116b1b52583333d0",
  "state": "ce25ca289f1ddebcd9473a0a42163a66bed4b78538c35cf208628880d8a6bac9",
}


def test_vendored_boxes_match_the_binary_version():
  assert VERSION == BOXES_VERSION == "v1.12.0", "mapd changed: re-vendor the bounding boxes and re-check offline_update.py"
  for kind, name in BOX_FILES.items():
    with open(os.path.join(BOXES_DIR, name), "rb") as f:
      assert hashlib.sha256(f.read()).hexdigest() == VENDORED_SHA256[kind]


def test_switzerland_tiles_in_binary_order():
  assert tile_order("nation", "CH") == [(44, 6), (44, 8), (44, 10), (46, 6), (46, 8), (46, 10)]


def test_unknown_code():
  assert tile_order("nation", "XX") is None
  assert tile_order("state", "CH") is None


def test_negative_coordinates_and_grid_boundaries():
  # floor towards minus infinity; a max on a grid line does not add a tile; a max above it does
  assert adjusted_bounds({"min_lat": -3.5, "min_lon": -0.5, "max_lat": -2.0, "max_lon": 2.0}) == (-4, -2, -2, 2)
  assert adjusted_bounds({"min_lat": 4.0, "min_lon": 4.0, "max_lat": 6.01, "max_lon": 5.0}) == (4, 4, 8, 6)
  assert tiles_for_box({"min_lat": -3.5, "min_lon": -0.5, "max_lat": -2.0, "max_lon": 2.0}) == [(-4, -2), (-4, 0)]


def test_us_and_a_state():
  us = tile_order("nation", "US")
  assert len(us) == len(set(us)) == 13 * 30  # 24..50 N x 126..66 W
  assert us[0] == (24, -126) and us[1] == (24, -124)  # longitude first within a row
  assert tile_order("state", "CA")
