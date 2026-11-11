import subprocess
import time
import datetime
from openpilot.common.time_helpers import system_time_valid
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr, trn
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.list_view import button_item, text_item, ListItem
from openpilot.system.ui.widgets.option_dialog import MultiOptionDialog
from openpilot.system.ui.widgets.scroller_tici import Scroller

if gui_app.sunnypilot_ui():
  from openpilot.system.ui.sunnypilot.widgets.list_view import button_item_sp as button_item

# TODO: remove this. updater fails to respond on startup if time is not correct
UPDATED_TIMEOUT = 10  # seconds to wait for updated to respond

# Mapping updater internal states to translated display strings
STATE_TO_DISPLAY_TEXT = {
  "checking...": tr("checking..."),
  "downloading...": tr("downloading..."),
  "finalizing update...": tr("finalizing update..."),
}


def update_failure_text(failure) -> tuple[str, str]:
  """Download row after a failed attempt, from the updater's stored record (param UpdaterLastFailure):
  the row's value ("check failed" or "download failed" by the stage, with the short reason) and the full record
  as the text that opens when the row is tapped."""
  if not isinstance(failure, dict) or not failure.get("stage"):
    return tr("failed to check for update"), ""

  what = tr("download failed") if failure["stage"] in ("download", "finalize") else tr("check failed")
  reason = str(failure.get("reason") or "")
  lines = [f"{what} ({failure['stage']}), {str(failure.get('time') or '').replace('T', ' ').replace('Z', ' UTC')}"]
  if failure.get("cmd"):
    lines.append(f"{failure['cmd']} (exit code {failure.get('rc')})")
  lines += str(failure.get("output") or reason).splitlines()
  if failure.get("retry") == "ok":
    lines.append("immediate retry: ok")
  elif failure.get("retry"):
    lines.append(f"immediate retry: failed ({failure.get('retry_stage')}): {failure.get('retry_cmd') or ''}")
    lines.append(str(failure.get("retry_reason") or ""))
  # the description is rendered as HTML: keep the text free of tags, one line each
  details = "<br>".join(line.replace("<", "(").replace(">", ")") for line in lines if line.strip())
  return (f"{what}: {reason}" if reason else what), details


def time_ago(date: datetime.datetime | None) -> str:
  if not date:
    return tr("never")

  if not system_time_valid():
    return date.strftime("%a %b %d %Y")

  now = datetime.datetime.now(datetime.UTC)
  if date.tzinfo is None:
    date = date.replace(tzinfo=datetime.UTC)

  diff_seconds = int((now - date).total_seconds())
  if diff_seconds < 60:
    return tr("now")
  if diff_seconds < 3600:
    m = diff_seconds // 60
    return trn("{} minute ago", "{} minutes ago", m).format(m)
  if diff_seconds < 86400:
    h = diff_seconds // 3600
    return trn("{} hour ago", "{} hours ago", h).format(h)
  if diff_seconds < 604800:
    d = diff_seconds // 86400
    return trn("{} day ago", "{} days ago", d).format(d)
  return date.strftime("%a %b %d %Y")


class SoftwareLayout(Widget):
  def __init__(self):
    super().__init__()

    self._onroad_label = ListItem(lambda: tr("Updates are only downloaded while the car is off."))
    self._version_item = text_item(lambda: tr("Current Version"), ui_state.params.get("UpdaterCurrentDescription") or "")
    self._download_btn = button_item(lambda: tr("Download"), lambda: tr("CHECK"), callback=self._on_download_update)

    # Install button is initially hidden
    self._install_btn = button_item(lambda: tr("Install Update"), lambda: tr("INSTALL"), callback=self._on_install_update)
    self._install_btn.set_visible(False)

    # Track waiting-for-updater transition to avoid brief re-enable while still idle
    self._waiting_for_updater = False
    self._waiting_start_ts: float = 0.0

    # Branch switcher
    self._branch_btn = button_item(lambda: tr("Target Branch"), lambda: tr("SELECT"), callback=self._on_select_branch)
    self._branch_btn.action_item.set_value(ui_state.params.get("UpdaterTargetBranch") or "")
    self._branch_dialog: MultiOptionDialog | None = None

    self._scroller = Scroller([
      self._onroad_label,
      self._version_item,
      self._download_btn,
      self._install_btn,
      self._branch_btn,
      button_item(lambda: tr("Uninstall"), lambda: tr("UNINSTALL"), callback=self._on_uninstall),
    ], line_separator=True, spacing=0)

  def show_event(self):
    super().show_event()
    self._scroller.show_event()

  def _render(self, rect):
    self._scroller.render(rect)

  def _update_state(self):
    # Show/hide onroad warning
    self._onroad_label.set_visible(ui_state.is_onroad())

    # Update current version and release notes
    current_desc = ui_state.params.get("UpdaterCurrentDescription") or ""
    current_release_notes = (ui_state.params.get("UpdaterCurrentReleaseNotes") or b"").decode("utf-8", "replace")
    self._version_item.action_item.set_text(current_desc)
    self._version_item.set_description(current_release_notes)

    # Update download button visibility and state
    self._download_btn.set_visible(ui_state.is_offroad())

    updater_state = ui_state.params.get("UpdaterState") or "idle"
    failed_count = ui_state.params.get("UpdateFailedCount") or 0
    fetch_available = ui_state.params.get_bool("UpdaterFetchAvailable")
    update_available = ui_state.params.get_bool("UpdateAvailable")

    failure_details = ""
    if updater_state != "idle":
      # Updater responded
      self._waiting_for_updater = False
      self._download_btn.action_item.set_enabled(False)
      # Use the mapping, with a fallback to the original state string
      display_text = STATE_TO_DISPLAY_TEXT.get(updater_state, updater_state)
      self._download_btn.action_item.set_value(display_text)
    else:
      if failed_count > 0:
        failure_value, failure_details = update_failure_text(ui_state.params.get("UpdaterLastFailure"))
        self._download_btn.action_item.set_value(failure_value)
        self._download_btn.action_item.set_text(tr("CHECK"))
      elif fetch_available:
        self._download_btn.action_item.set_value(tr("update available"))
        self._download_btn.action_item.set_text(tr("DOWNLOAD"))
      else:
        last_update = ui_state.params.get("LastUpdateTime")
        if last_update:
          formatted = time_ago(last_update)
          self._download_btn.action_item.set_value(tr("up to date, last checked {}").format(formatted))
        else:
          self._download_btn.action_item.set_value(tr("up to date, last checked never"))
        self._download_btn.action_item.set_text(tr("CHECK"))

      # If we've been waiting too long without a state change, reset state
      if self._waiting_for_updater and (time.monotonic() - self._waiting_start_ts > UPDATED_TIMEOUT):
        self._waiting_for_updater = False

      # Only enable if we're not waiting for updater to flip out of idle
      self._download_btn.action_item.set_enabled(not self._waiting_for_updater)

    self._set_download_details(failure_details)

    # Update target branch button value
    current_branch = ui_state.params.get("UpdaterTargetBranch") or ""
    self._branch_btn.action_item.set_value(current_branch)

    # Update install button
    self._install_btn.set_visible(ui_state.is_offroad() and update_available)
    if update_available:
      new_desc = ui_state.params.get("UpdaterNewDescription") or ""
      new_release_notes = (ui_state.params.get("UpdaterNewReleaseNotes") or b"").decode("utf-8", "replace")
      self._install_btn.action_item.set_text(tr("INSTALL"))
      self._install_btn.action_item.set_value(new_desc)
      self._install_btn.set_description(new_release_notes)
      # Enable install button for testing (like Qt showEvent)
      self._install_btn.action_item.set_enabled(True)
    else:
      self._install_btn.set_visible(False)

  def _set_download_details(self, details: str):
    # The text that opens when the Download row is tapped: what failed, and only while the row says that it failed.
    if not details and self._download_btn.description_visible:
      self._download_btn._set_description_visible(False)  # close it while there still is a text to close
    self._download_btn.set_description(details)

  def _on_download_update(self):
    # Check if we should start checking or start downloading
    self._download_btn.action_item.set_enabled(False)
    if self._download_btn.action_item.text == tr("CHECK"):
      # Start checking for updates
      self._waiting_for_updater = True
      self._waiting_start_ts = time.monotonic()
      subprocess.run("pkill -SIGUSR1 -f openpilot.system.updated.updated", shell=True)
    else:
      # Start downloading
      self._waiting_for_updater = True
      self._waiting_start_ts = time.monotonic()
      subprocess.run("pkill -SIGHUP -f openpilot.system.updated.updated", shell=True)

  def _on_uninstall(self):
    def handle_uninstall_confirmation(result: DialogResult):
      if result == DialogResult.CONFIRM:
        ui_state.params.put_bool("DoUninstall", True, block=True)

    dialog = ConfirmDialog(tr("Are you sure you want to uninstall?"), tr("Uninstall"), callback=handle_uninstall_confirmation)
    gui_app.push_widget(dialog)

  def _on_install_update(self):
    # Trigger reboot to install update
    self._install_btn.action_item.set_enabled(False)
    ui_state.params.put_bool("DoReboot", True, block=True)

  def _on_select_branch(self):
    # Get available branches and order
    current_git_branch = ui_state.params.get("GitBranch") or ""
    branches_str = ui_state.params.get("UpdaterAvailableBranches") or ""
    branches = [b for b in branches_str.split(",") if b]

    for b in [current_git_branch, "devel-staging", "devel", "nightly", "nightly-dev", "master"]:
      if b in branches:
        branches.remove(b)
        branches.insert(0, b)

    current_target = ui_state.params.get("UpdaterTargetBranch") or ""

    def handle_selection(result: DialogResult):
      # Confirmed selection
      if result == DialogResult.CONFIRM and self._branch_dialog is not None and self._branch_dialog.selection:
        selection = self._branch_dialog.selection
        ui_state.params.put("UpdaterTargetBranch", selection, block=True)
        self._branch_btn.action_item.set_value(selection)
        self._download_btn.action_item.set_enabled(False)
        self._waiting_for_updater = True
        self._waiting_start_ts = time.monotonic()
        subprocess.run("pkill -SIGUSR1 -f openpilot.system.updated.updated", shell=True)
      self._branch_dialog = None

    self._branch_dialog = MultiOptionDialog(tr("Select a branch"), branches, current_target, callback=handle_selection)
    gui_app.push_widget(self._branch_dialog)
