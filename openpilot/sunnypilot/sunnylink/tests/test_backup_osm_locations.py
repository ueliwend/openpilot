"""sunnylink backup of the selected map countries (E9, owner 11 Nov 2026: the list is in the backup, the maps are not)."""
import base64
from unittest import mock

from openpilot.common.params import Params, ParamKeyFlag
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.sunnylink.backups import manager
from openpilot.sunnypilot.sunnylink.utils import BLOCKED_PARAMS

SELECTION = [{"kind": "nation", "code": "CH", "title": "Switzerland"}, {"kind": "state", "code": "CA", "title": "California"}]


class TestBackupOsmLocations(OpenpilotTestCase):
  def setup_method(self):
    patcher = mock.patch.object(manager.messaging, "PubMaster")
    patcher.start()
    self.addCleanup(patcher.stop)
    self.params = Params()
    self.backup_manager = manager.BackupManagerSP()

  def test_selection_is_backed_up_and_restored(self):
    assert "OsmSelectedLocations" in self.backup_manager._backup_keys()
    assert "OsmSelectedLocations" not in BLOCKED_PARAMS
    self.params.put("OsmSelectedLocations", SELECTION, block=True)
    config = self.backup_manager._collect_config_data()
    assert "OsmSelectedLocations" in config
    self.params.remove("OsmSelectedLocations")
    self.backup_manager._apply_config({"OsmSelectedLocations": config["OsmSelectedLocations"]})
    assert self.params.get("OsmSelectedLocations") == SELECTION

  def test_derived_and_request_keys_stay_out_of_the_backup(self):
    backup = {k.decode() for k in self.params.all_keys(ParamKeyFlag.BACKUP)}
    assert not backup & {"OsmLocationStatus", "OsmDownloadRequest", "OsmDeleteLocation", "OsmLocationName", "OsmLocal"}
    self.params.put("OsmDownloadRequest", "all", block=True)
    assert "OsmDownloadRequest" not in self.backup_manager._collect_config_data()

  def test_restore_still_skips_blocked_keys_next_to_the_selection(self):
    self.params.put("GithubUsername", "device-user", block=True)
    config = {"OsmSelectedLocations": base64.b64encode(b'[{"kind": "nation", "code": "AT"}]').decode(),
              "GithubUsername": base64.b64encode(b"planted").decode(),
              "OsmDownloadRequest": base64.b64encode(b"all").decode()}  # not a backup key: never written by a restore
    self.backup_manager._apply_config(config)
    assert self.params.get("OsmSelectedLocations") == [{"kind": "nation", "code": "AT"}]  # mapd_manager validates it on read
    assert self.params.get("GithubUsername") == "device-user"
    assert self.params.get("OsmDownloadRequest") is None
