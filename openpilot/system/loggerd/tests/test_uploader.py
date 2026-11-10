import os
import threading
import logging
import json
from pathlib import Path
from openpilot.common.hardware.hw import Paths

from openpilot.common.swaglog import cloudlog
from openpilot.common.params import UnknownKeyName
import openpilot.system.loggerd.uploader as uploader_module
from openpilot.system.loggerd.uploader import clear_locks, main, Uploader, UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE

from openpilot.system.loggerd.tests.loggerd_tests_common import UploaderTestCase


class FakeLogHandler(logging.Handler):
  def __init__(self):
    logging.Handler.__init__(self)
    self.condition = threading.Condition()
    self.reset()

  def reset(self):
    with self.condition:
      self.upload_order = []
      self.upload_ignored = []

  def emit(self, record):
    try:
      j = json.loads(record.getMessage())
      with self.condition:
        if j["event"] == "upload_success":
          self.upload_order.append(j["key"])
        if j["event"] == "upload_ignored":
          self.upload_ignored.append(j["key"])
        self.condition.notify_all()
    except Exception:
      pass

  def wait_for_uploads(self, count: int, ignored: bool = False):
    uploads = self.upload_ignored if ignored else self.upload_order
    with self.condition:
      assert self.condition.wait_for(lambda: len(uploads) >= count, timeout=1), "Uploader did not process all files"

log_handler = FakeLogHandler()
cloudlog.addHandler(log_handler)


class TestUploader(UploaderTestCase):
  def setup_method(self):
    super().openpilot_setup_method()
    log_handler.reset()

  def start_thread(self):
    self.end_event = threading.Event()
    self.up_thread = threading.Thread(target=main, args=[self.end_event])
    self.up_thread.daemon = True
    self.up_thread.start()

  def join_thread(self):
    self.end_event.set()
    self.up_thread.join()

  def gen_files(self, lock=False, xattr: bytes | None = None, boot=True) -> list[Path]:
    f_paths = []
    for t in ["qlog", "rlog", "dcamera.hevc", "fcamera.hevc"]:
      f_paths.append(self.make_file_with_data(self.seg_dir, t, 1, lock=lock, upload_xattr=xattr))

    if boot:
      f_paths.append(self.make_file_with_data("boot", f"{self.seg_dir}", 1, lock=lock, upload_xattr=xattr))
    return f_paths

  def gen_order(self, seg1: list[int], seg2: list[int], boot=True) -> list[str]:
    keys = []
    if boot:
      keys += [f"boot/{self.seg_format.format(i)}.zst" for i in seg1]
      keys += [f"boot/{self.seg_format2.format(i)}.zst" for i in seg2]
    keys += [f"{self.seg_format.format(i)}/qlog.zst" for i in seg1]
    keys += [f"{self.seg_format2.format(i)}/qlog.zst" for i in seg2]
    return keys

  def test_upload(self):
    self.gen_files(lock=False)
    exp_order = self.gen_order([self.seg_num], [])

    self.start_thread()
    log_handler.wait_for_uploads(len(exp_order))
    self.join_thread()

    assert len(log_handler.upload_ignored) == 0, "Some files were ignored"
    assert not len(log_handler.upload_order) < len(exp_order), "Some files failed to upload"
    assert not len(log_handler.upload_order) > len(exp_order), "Some files were uploaded twice"
    for f_path in exp_order:
      assert os.getxattr((Path(Paths.log_root()) / f_path).with_suffix(""), UPLOAD_ATTR_NAME) == UPLOAD_ATTR_VALUE, "All files not uploaded"

    assert log_handler.upload_order == exp_order, "Files uploaded in wrong order"

  def test_upload_with_wrong_xattr(self):
    self.gen_files(lock=False, xattr=b'0')
    exp_order = self.gen_order([self.seg_num], [])

    self.start_thread()
    log_handler.wait_for_uploads(len(exp_order))
    self.join_thread()

    assert len(log_handler.upload_ignored) == 0, "Some files were ignored"
    assert not len(log_handler.upload_order) < len(exp_order), "Some files failed to upload"
    assert not len(log_handler.upload_order) > len(exp_order), "Some files were uploaded twice"
    for f_path in exp_order:
      assert os.getxattr((Path(Paths.log_root()) / f_path).with_suffix(""), UPLOAD_ATTR_NAME) == UPLOAD_ATTR_VALUE, "All files not uploaded"

    assert log_handler.upload_order == exp_order, "Files uploaded in wrong order"

  def test_upload_ignored(self):
    self.set_ignore()
    self.gen_files(lock=False)
    exp_order = self.gen_order([self.seg_num], [])

    self.start_thread()
    log_handler.wait_for_uploads(len(exp_order), ignored=True)
    self.join_thread()

    assert len(log_handler.upload_order) == 0, "Some files were not ignored"
    assert not len(log_handler.upload_ignored) < len(exp_order), "Some files failed to ignore"
    assert not len(log_handler.upload_ignored) > len(exp_order), "Some files were ignored twice"
    for f_path in exp_order:
      assert os.getxattr((Path(Paths.log_root()) / f_path).with_suffix(""), UPLOAD_ATTR_NAME) == UPLOAD_ATTR_VALUE, "All files not ignored"

    assert log_handler.upload_ignored == exp_order, "Files ignored in wrong order"

  def test_upload_files_in_create_order(self):
    seg1_nums = [0, 1, 2, 10, 20]
    for i in seg1_nums:
      self.seg_dir = self.seg_format.format(i)
      self.gen_files(boot=False)
    seg2_nums = [5, 50, 51]
    for i in seg2_nums:
      self.seg_dir = self.seg_format2.format(i)
      self.gen_files(boot=False)

    exp_order = self.gen_order(seg1_nums, seg2_nums, boot=False)

    self.start_thread()
    log_handler.wait_for_uploads(len(exp_order))
    self.join_thread()

    assert len(log_handler.upload_ignored) == 0, "Some files were ignored"
    assert not len(log_handler.upload_order) < len(exp_order), "Some files failed to upload"
    assert not len(log_handler.upload_order) > len(exp_order), "Some files were uploaded twice"
    for f_path in exp_order:
      assert os.getxattr((Path(Paths.log_root()) / f_path).with_suffix(""), UPLOAD_ATTR_NAME) == UPLOAD_ATTR_VALUE, "All files not uploaded"

    assert log_handler.upload_order == exp_order, "Files uploaded in wrong order"

  def test_no_upload_with_lock_file(self):
    f_paths = self.gen_files(lock=True, boot=False)
    uploader = Uploader("0000000000000000", Paths.log_root())

    for f_path in f_paths:
      fn = f_path.with_suffix(f_path.suffix.replace(".zst", ""))
      assert all(candidate[2] != str(fn) for candidate in uploader.list_upload_files(metered=False)), "Locked file selected for upload"

  def test_no_upload_with_xattr(self):
    f_paths = self.gen_files(lock=False, xattr=UPLOAD_ATTR_VALUE)
    uploader = Uploader("0000000000000000", Paths.log_root())
    upload_candidates = {candidate[2] for candidate in uploader.list_upload_files(metered=False)}
    assert upload_candidates.isdisjoint(map(str, f_paths)), "Uploaded file selected again"

  def test_clear_locks_on_startup(self, mocker):
    f_paths = self.gen_files(lock=True, boot=False)
    locks_cleared = threading.Event()

    def clear_locks_and_signal(root):
      clear_locks(root)
      locks_cleared.set()

    mocker.patch("openpilot.system.loggerd.uploader.clear_locks", side_effect=clear_locks_and_signal)
    self.start_thread()
    assert locks_cleared.wait(timeout=1), "Uploader did not clear locks on startup"
    self.join_thread()

    for f_path in f_paths:
      lock_path = f_path.with_suffix(f_path.suffix + ".lock")
      assert not lock_path.is_file(), "File lock not cleared on startup"

  def test_rlogs_not_uploaded_by_default(self):
    self.make_file_with_data(self.seg_dir, "qlog", upload_xattr=UPLOAD_ATTR_VALUE)
    self.make_file_with_data(self.seg_dir, "rlog")
    uploader = Uploader("0000000000000000", Paths.log_root())

    assert uploader.next_file_to_upload(metered=False) is None, "rlog selected while UploadRlogs is off"
    assert self.params.get("UploadRlogsFromRoute") is None

  def test_upload_rlogs_from_route_in_progress_on(self):
    # route 4 is older; route 5 is the newest (RouteCount 6) when the toggle is switched on
    for i in [0, 1]:
      self.seg_dir = self.seg_format.format(i)
      self.gen_files(boot=False)
      self.seg_dir = self.seg_format2.format(i)
      self.gen_files(boot=False)
    self.params.put("RouteCount", 6)
    self.params.put_bool("UploadRlogs", True)

    exp_order = self.gen_order([0, 1], [0, 1], boot=False)
    exp_order += [f"{self.seg_format2.format(i)}/rlog.zst" for i in [0, 1]]

    self.start_thread()
    log_handler.wait_for_uploads(len(exp_order))
    self.join_thread()

    assert log_handler.upload_order == exp_order, "qlogs first, then only the rlogs of route 5"
    assert self.params.get("UploadRlogsFromRoute") == 5

  def test_rlogs_wait_for_unmetered_network(self):
    self.make_file_with_data(self.seg_dir, "qlog", upload_xattr=UPLOAD_ATTR_VALUE)
    rlog = self.make_file_with_data(self.seg_dir, "rlog")
    self.params.put("RouteCount", 5)
    self.params.put_bool("UploadRlogs", True)
    uploader = Uploader("0000000000000000", Paths.log_root())

    assert uploader.next_file_to_upload(metered=True) is None, "rlog selected on a metered network"
    assert uploader.next_file_to_upload(metered=False) == ("rlog", f"{self.seg_dir}/rlog", str(rlog))

  def test_switching_off_forgets_start_route(self):
    self.make_file_with_data(self.seg_format.format(0), "rlog")
    self.params.put("RouteCount", 5)
    self.params.put_bool("UploadRlogs", True)
    uploader = Uploader("0000000000000000", Paths.log_root())
    assert uploader.rlog_route_range() == (4, 5)

    self.params.put_bool("UploadRlogs", False)
    assert uploader.next_file_to_upload(metered=False) is None
    assert self.params.get("UploadRlogsFromRoute") is None

    # a newer drive happens while the toggle is off; switching it on again starts from that drive
    self.make_file_with_data(self.seg_format2.format(0), "rlog")
    self.params.put("RouteCount", 6)
    self.params.put_bool("UploadRlogs", True)
    assert uploader.rlog_route_range() == (5, 6)
    assert uploader.next_file_to_upload(metered=False)[1] == f"{self.seg_format2.format(0)}/rlog"

  def test_route_counter_reset_restarts_from_zero(self):
    self.make_file_with_data(self.seg_format2.format(0), "rlog", upload_xattr=UPLOAD_ATTR_VALUE)
    self.params.put("RouteCount", 6)
    self.params.put_bool("UploadRlogs", True)
    uploader = Uploader("0000000000000000", Paths.log_root())
    assert uploader.rlog_route_range() == (5, 6)

    # the counter restarts: new route 0 is in progress, the old routes 4 and 5 (higher counters) stay out
    self.make_file_with_data(self.seg_format.format(0), "rlog")
    new_route = "00000000--1a2b3c4d5e--0"
    self.make_file_with_data(new_route, "rlog")
    self.params.put("RouteCount", 1)
    assert uploader.next_file_to_upload(metered=False)[1] == f"{new_route}/rlog"
    assert self.params.get("UploadRlogsFromRoute") == 0

  def test_failed_rlog_never_holds_up_qlogs(self):
    self.make_file_with_data(self.seg_dir, "rlog")
    self.params.put("RouteCount", 5)
    self.params.put_bool("UploadRlogs", True)
    uploader = Uploader("0000000000000000", Paths.log_root())
    key = f"{self.seg_dir}/rlog"

    uploader.rlog_upload_done(key, success=False)
    assert uploader.next_file_to_upload(metered=False) is None, "failed rlog tried again during its pause"
    qlog = self.make_file_with_data(self.seg_format2.format(0), "qlog")
    assert uploader.next_file_to_upload(metered=False)[2] == str(qlog), "qlog must not wait behind the paused rlog"

    # after RLOG_MAX_TRIES failures it waits for the next start, while a newer rlog still goes up
    for _ in range(uploader_module.RLOG_MAX_TRIES - 1):
      uploader.rlog_upload_done(key, success=False)
    os.remove(qlog)
    newer = self.make_file_with_data(self.seg_format2.format(1), "rlog")
    self.params.put("RouteCount", 6)
    uploader.rlog_pause_until = 0
    assert uploader.next_file_to_upload(metered=False)[2] == str(newer)

  def test_unknown_params_treated_as_off(self, mocker):
    # update installed without the rebuild: the params library does not know the new keys
    qlog = self.make_file_with_data(self.seg_dir, "qlog")
    self.make_file_with_data(self.seg_dir, "rlog")
    uploader = Uploader("0000000000000000", Paths.log_root())
    mocker.patch.object(uploader.params, "get_bool", side_effect=UnknownKeyName("UploadRlogs"))

    assert uploader.next_file_to_upload(metered=False)[2] == str(qlog), "qlogs must keep uploading"
    assert uploader.rlog_route_range() is None
