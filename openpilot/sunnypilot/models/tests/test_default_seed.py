"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.cereal import custom
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.models import helpers
from openpilot.sunnypilot.models.manager import DEFAULT_SEED_BUNDLE_REF, DEFAULT_SEED_PARAM, ModelManagerSP

SEED = DEFAULT_SEED_BUNDLE_REF


def bundle(ref: str) -> custom.ModelManagerSP.ModelBundle:
  b = custom.ModelManagerSP.ModelBundle.new_message()
  b.ref = ref
  b.minimumSelectorVersion = helpers.REQUIRED_JSON_VERSION
  return custom.ModelManagerSP.ModelBundle(**b.to_dict())


def boot(params: Params) -> ModelManagerSP:
  """the seed part of ModelManagerSP.__init__ (no fetcher / messaging)"""
  m = ModelManagerSP.__new__(ModelManagerSP)
  m.params = params
  m.source_models = {}
  m._decide_default_seed()
  return m


def tick(m: ModelManagerSP, catalog: dict | None) -> None:
  m.source_models = catalog or {"qcom": [], "chestnut": []}
  m._seed_default_bundle()


CATALOG = {"qcom": [bundle("cd210"), bundle(SEED)], "chestnut": [bundle("big")]}


class TestDefaultModelSeed(OpenpilotTestCase):
  def test_constants(self):
    assert SEED == "62bf6fb072880905a4c490f0f4f4a6b3c23346ec" and DEFAULT_SEED_PARAM == "ModelManager_DefaultSeed"
    assert Params().get_default_value(DEFAULT_SEED_PARAM) is None

  def test_fresh_store_requests_once_and_installs(self):
    params = Params()
    m = boot(params)
    assert params.get(DEFAULT_SEED_PARAM) == "pending"
    tick(m, None)                                          # offline: wait, no request
    assert params.get(DEFAULT_SEED_PARAM) == "pending" and params.get("ModelManager_DownloadRef") is None
    tick(m, CATALOG)
    assert params.get(DEFAULT_SEED_PARAM) == "requested" and params.get("ModelManager_DownloadRef") == SEED
    tick(m, CATALOG)                                       # in flight: nothing changes
    assert params.get(DEFAULT_SEED_PARAM) == "requested"
    params.put(helpers.ACTIVE_BUNDLE_KEYS["qcom"], bundle(SEED).to_dict(), block=True)   # the download finished
    params.remove("ModelManager_DownloadRef")
    tick(m, CATALOG)
    assert params.get(DEFAULT_SEED_PARAM) == "done:installed"
    m2 = boot(params)                                      # later boots: nothing
    tick(m2, CATALOG)
    assert params.get(DEFAULT_SEED_PARAM) == "done:installed" and params.get("ModelManager_DownloadRef") is None

  def test_failed_download_keeps_the_default_and_never_retries(self):
    params = Params()
    m = boot(params)
    tick(m, CATALOG)
    params.remove("ModelManager_DownloadRef")              # failed or cancelled: no active bundle
    tick(m, CATALOG)
    assert params.get(DEFAULT_SEED_PARAM).startswith("done:download failed")
    assert params.get(helpers.ACTIVE_BUNDLE_KEYS["qcom"]) is None
    for _ in range(3):
      tick(boot(params), CATALOG)
      assert params.get("ModelManager_DownloadRef") is None

  def test_restart_during_download_does_not_request_again(self):
    params = Params()
    tick(boot(params), CATALOG)
    params.remove("ModelManager_DownloadRef")              # manager start clears the request (CLEAR_ON_MANAGER_START)
    m = boot(params)
    tick(m, CATALOG)
    assert params.get(DEFAULT_SEED_PARAM).startswith("done:download failed, cancelled or interrupted")
    tick(m, CATALOG)
    assert params.get("ModelManager_DownloadRef") is None

  def test_existing_store_is_never_seeded(self):
    for key, value in (("ModelManager_ModelsCache", {"bundles": []}), ("ModelManager_ModelsCache_Chestnut", {"bundles": []}),
                       (helpers.ACTIVE_BUNDLE_KEYS["qcom"], bundle("cd210").to_dict()),
                       (helpers.ACTIVE_BUNDLE_KEYS["chestnut"], bundle("big").to_dict())):
      params = Params()
      params.put(key, value, block=True)
      m = boot(params)
      assert params.get(DEFAULT_SEED_PARAM) == "done:existing store", key
      tick(m, CATALOG)
      assert params.get("ModelManager_DownloadRef") is None, key
      params.remove(key)
      params.remove(DEFAULT_SEED_PARAM)

  def test_a_users_choice_wins(self):
    params = Params()
    m = boot(params)
    params.put("ModelManager_DownloadRef", "cd210", block=True)  # the user asks for a model before the catalog was reachable
    tick(m, CATALOG)
    assert params.get(DEFAULT_SEED_PARAM) == "done:model requested" and params.get("ModelManager_DownloadRef") == "cd210"
    params.remove("ModelManager_DownloadRef")              # even if that download fails, the seed stays off
    tick(m, CATALOG)
    assert params.get("ModelManager_DownloadRef") is None

    params = Params()
    params.remove(DEFAULT_SEED_PARAM)
    m = boot(params)
    params.put(helpers.ACTIVE_BUNDLE_KEYS["qcom"], bundle("cd210").to_dict(), block=True)
    tick(m, CATALOG)
    assert params.get(DEFAULT_SEED_PARAM) == "done:model selected" and params.get("ModelManager_DownloadRef") is None

  def test_unknown_bundle_keeps_the_shipped_default(self):
    params = Params()
    m = boot(params)
    tick(m, {"qcom": [bundle("cd210")], "chestnut": [bundle("big")]})
    assert params.get(DEFAULT_SEED_PARAM).startswith("done:bundle not in the catalog")
    assert params.get("ModelManager_DownloadRef") is None
    tick(m, {"qcom": [], "chestnut": [bundle(SEED)]})      # the qcom catalog alone decides; nothing requested later
    assert params.get("ModelManager_DownloadRef") is None

  def test_waits_for_the_qcom_catalog(self):
    params = Params()
    m = boot(params)
    tick(m, {"qcom": [], "chestnut": [bundle("big")]})     # chestnut catalog only: not decided yet
    assert params.get(DEFAULT_SEED_PARAM) == "pending" and params.get("ModelManager_DownloadRef") is None
