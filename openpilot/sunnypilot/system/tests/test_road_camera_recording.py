"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Inster (E10, owner 12 Nov 2026): RecordRoadFull decides whether the full-resolution road and wide road cameras (fcamera.hevc,
ecamera.hevc) are written; the road camera preview (qcamera.ts), the thumbnails and the encode indexes stay. The behaviour test is
system/loggerd/tests/test_loggerd.py test_record_road_full (needs the built loggerd / encoderd); this file checks, without a build,
the source facts that behaviour rests on, so a sync that moves them fails here.
"""
import re
from pathlib import Path

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params, ParamKeyFlag
from openpilot.common.test import OpenpilotTestCase

LOGGERD = Path(BASEDIR) / "openpilot/system/loggerd"
TOGGLES = Path(BASEDIR) / "openpilot/selfdrive/ui/layouts/settings/toggles.py"


def encoder_infos(src: str) -> dict[str, dict[str, str]]:
  """{file name: {field: expression}} of every `const EncoderInfo <name> = {...};` with a file name (designated initialisers)"""
  out = {}
  for body in re.findall(r"const EncoderInfo \w+ = \{(.*?)\n\};", src, re.S):
    fields = dict(re.findall(r"^\s*\.(\w+) = (.*?),?\s*$", body, re.M))
    if "filename" in fields:
      out[fields["filename"].strip('"')] = fields
  return out


class TestRoadCameraRecording(OpenpilotTestCase):
  def test_key(self):
    params = Params()
    assert params.get_default_value("RecordRoadFull") is False
    assert b"RecordRoadFull" in params.all_keys(ParamKeyFlag.BACKUP)
    assert b"RecordRoadFull" in params.all_keys(ParamKeyFlag.PERSISTENT)

  def test_record_flags(self):
    infos = encoder_infos((LOGGERD / "loggerd.h").read_text())
    assert set(infos) == {"fcamera.hevc", "ecamera.hevc", "dcamera.hevc", "qcamera.ts"}, set(infos)
    for fn in ("fcamera.hevc", "ecamera.hevc"):
      assert infos[fn].get("record") == 'Params().getBool("RecordRoadFull")', (fn, infos[fn])
    assert infos["dcamera.hevc"].get("record") == 'Params().getBool("RecordFront")'
    assert "record" not in infos["qcamera.ts"], infos["qcamera.ts"]           # the preview keeps the class default
    assert re.search(r"^\s*bool record = true;", (LOGGERD / "loggerd.h").read_text(), re.M)

  def test_preview_shares_the_road_camera(self):
    src = (LOGGERD / "loggerd.h").read_text()
    m = re.search(r"const LogCameraInfo narrow_road_camera_info\{(.*?)\};", src, re.S)
    assert m and re.search(r"\.encoder_infos = \{main_road_encoder_info, qcam_encoder_info\}", m.group(1)), m

  def test_writer_only_when_recorded(self):
    # loggerd opens a VideoWriter only for a recorded stream; the encode index goes to the log either way
    src = (LOGGERD / "loggerd.cc").read_text()
    assert re.search(r"if \(encoder_info\.record\) \{\s*assert\(encoder_info\.filename != NULL\);\s*re\.writer\.reset\(new VideoWriter", src)
    assert re.search(r"if \(re\.writer\) \{\s*auto data = edata\.getData\(\);", src)
    assert "(evt.*(encoder_info.set_encode_idx_func))(idx);" in src

  def test_thumbnail_does_not_depend_on_recording(self):
    # encoderd makes the thumbnail from the camera frame of the stream's first encoder, whatever `record` says
    src = (LOGGERD / "encoderd.cc").read_text()
    assert "if (auto thumbnail_name = cam_info.encoder_infos[0].thumbnail_name) {" in src
    assert ".record" not in src
    assert encoder_infos((LOGGERD / "loggerd.h").read_text())["fcamera.hevc"].get("thumbnail_name") == '"thumbnail"'

  def test_toggle(self):
    src = TOGGLES.read_text()
    assert re.search(r'"RecordRoadFull": tr_noop\(', src)
    # (title, description, icon, needs_restart): True = blocked while engaged, and a change while the car is on restarts sunnypilot
    assert re.search(r'"RecordRoadFull": \(\s*lambda: tr\("Record Full-Resolution Road Cameras"\),\s*DESCRIPTIONS\["RecordRoadFull"\],' +
                     r'\s*"road.png",\s*True,\s*\),', src)
