"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import base64
from unittest import mock

from openpilot.common.params import Params, ParamKeyFlag, ParamKeyType
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.sunnylink.backups import manager
from openpilot.sunnypilot.sunnylink.utils import BLOCKED_PARAMS

DEVICE_VALUES = {ParamKeyType.BOOL: False, ParamKeyType.INT: 3, ParamKeyType.STRING: "device value"}
BACKUP_VALUES = {ParamKeyType.BOOL: b"1", ParamKeyType.INT: b"7", ParamKeyType.STRING: b"ssh-ed25519 AAAA planted"}
RESTORABLE = {"IsMetric": b"0", "GsmRoaming": b"0", "AutoLaneChangeTimer": b"2"}


class TestBackupManager(OpenpilotTestCase):
  def setup_method(self):
    patcher = mock.patch.object(manager.messaging, "PubMaster")  # no message queue needed: status messages are not tested
    patcher.start()
    self.addCleanup(patcher.stop)
    self.params = Params()
    self.backup_manager = manager.BackupManagerSP()

  def _restore_blocked(self):
    """Restores a backup holding every blocked key as spelled, in lower and in upper case, plus three restorable keys.
    Returns the names the restore logged as blocked."""
    blocked = sorted(BLOCKED_PARAMS)
    for key in blocked:
      self.params.put(key, DEVICE_VALUES[self.params.get_type(key)], block=True)
    # restore matches names case-insensitively
    config = {}
    for key in blocked:
      for name in (key, key.lower(), key.upper()):
        config[name] = base64.b64encode(BACKUP_VALUES[self.params.get_type(key)]).decode()
    config |= {key: base64.b64encode(value).decode() for key, value in RESTORABLE.items()}

    with mock.patch.object(manager, "cloudlog") as log:
      self.backup_manager._apply_config(config)

    for key in blocked:
      assert self.params.get(key) == DEVICE_VALUES[self.params.get_type(key)], key
    assert self.params.get("IsMetric") is False
    assert self.params.get("GsmRoaming") is False
    assert self.params.get("AutoLaneChangeTimer") == 2
    messages = [c.args[0] for c in log.warning.call_args_list if "blocked for remote changes" in c.args[0]]
    assert len(messages) == len(config) - len(RESTORABLE)
    return {name for name in config if any(f" {name}: " in m for m in messages)}, set(config) - set(RESTORABLE)

  def test_restore_skips_blocked_params(self):
    logged, blocked_names = self._restore_blocked()
    assert logged == blocked_names  # one "blocked" warning per blocked name and spelling (27)

  def test_restore_blocks_even_if_marked_for_backup(self):
    # the restore's own check alone: blocked keys offered as backup keys are still never written
    all_backup = [k.decode() for k in self.params.all_keys(ParamKeyFlag.BACKUP)]
    with mock.patch.object(self.backup_manager, "_backup_keys", return_value=all_backup):
      logged, blocked_names = self._restore_blocked()
    assert logged == blocked_names

  def test_backup_leaves_out_blocked_params(self):
    self.params.put("GithubUsername", "device-user", block=True)
    self.params.put("GithubSshKeys", "ssh-ed25519 AAAA device", block=True)
    self.params.put_bool("AdbEnabled", True, block=True)
    self.params.put_bool("IsMetric", True, block=True)

    config = self.backup_manager._collect_config_data()

    assert "IsMetric" in config
    assert not set(config) & BLOCKED_PARAMS
    backup_keys = {k.decode() for k in self.params.all_keys(ParamKeyFlag.BACKUP)}
    assert set(self.backup_manager._backup_keys()) == backup_keys - BLOCKED_PARAMS
    assert backup_keys & BLOCKED_PARAMS  # the blocked keys this guards: AdbEnabled, GithubUsername, GithubSshKeys
