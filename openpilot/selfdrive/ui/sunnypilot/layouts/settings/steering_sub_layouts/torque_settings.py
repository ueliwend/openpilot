"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json
import math
import os
from collections.abc import Callable
import pyray as rl

from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.selfdrive.controls.lib.inster_lateral import ANGLE_PI_GAINS_PARAM, SOFT_P_PARAM, angle_pi_gain_set, is_inster
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.lib.utils import NoElideButtonAction
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, multiple_button_item_sp, toggle_item_sp, option_item_sp
from openpilot.system.ui.sunnypilot.widgets.tree_dialog import TreeOptionDialog, TreeFolder, TreeNode
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller

TORQUE_VERSIONS_PATH = os.path.join(BASEDIR, "openpilot", "sunnypilot", "selfdrive", "controls", "lib", "latcontrol_torque_versions.json")


class TorqueSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()
    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)
    self._torque_version_dialog: TreeOptionDialog | None = None
    self.cached_torque_versions = {}
    self._load_versions()
    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _load_versions(self):
    with open(TORQUE_VERSIONS_PATH) as f:
      self.cached_torque_versions = json.load(f)

  def _initialize_items(self):
    self._jerk_aware_toggle = toggle_item_sp(
      param="LateralJerkTorqueController",
      title=lambda: tr("Lateral Jerk Torque Controller"),
      description=lambda: tr("Looks ahead at planned steering to reduce sudden corrections, so the wheel moves " +
                             "more smoothly through turns. Works with Self-Tune and custom tuning. " +
                             "Thanks to @twilsonco for the implementation."),
    )
    self._torque_control_versions = ListItemSP(
      title=tr("Torque Control Tune Version"),
      description="Select the version of Torque Control Tune to use.",
      action_item=NoElideButtonAction(tr("SELECT")),
      callback=self._show_torque_version_dialog,
    )
    self._self_tune_toggle = toggle_item_sp(
      param="LiveTorqueParamsToggle",
      title=lambda: tr("Self-Tune"),
      description=lambda: tr("Enables self-tune for Torque lateral control for platforms that do not use " +
                             "Torque lateral control by default."),
    )
    self._relaxed_tune_toggle = toggle_item_sp(
      param="LiveTorqueParamsRelaxedToggle",
      title=lambda: tr("Less Restrict Settings for Self-Tune (Beta)"),
      description=lambda: tr("Less strict settings when using Self-Tune. This allows torqued to be more " +
                             "forgiving when learning values."),
    )
    self._custom_tune_toggle = toggle_item_sp(
      param="CustomTorqueParams",
      title=lambda: tr("Enable Custom Tuning"),
      description=lambda: tr("Enables custom tuning for Torque lateral control. " +
                             "Modifying Lateral Acceleration Factor and Friction below will override the offline values " +
                             "indicated in the YAML files within \"opendbc/car/torque_data\". " +
                             "The values will also be used live when \"Manual Real-Time Tuning\" toggle is enabled."),
    )
    self._torque_prams_override_toggle = toggle_item_sp(
      param="TorqueParamsOverrideEnabled",
      title=lambda: tr("Manual Real-Time Tuning"),
      description=lambda: tr("Enforces the torque lateral controller to use the fixed values instead of the learned " +
                             "values from Self-Tune. Enabling this toggle overrides Self-Tune values."),
    )
    self._torque_lat_accel_factor = option_item_sp(
      title=lambda: tr("Lateral Acceleration Factor"),
      param="TorqueParamsOverrideLatAccelFactor",
      description="",
      min_value=1,
      max_value=500,
      value_change_step=1,
      label_callback=(lambda x: f"{x/100} m/s^2"),
      use_float_scaling=True
    )

    self._torque_friction = option_item_sp(
      title=lambda: tr("Friction"),
      param="TorqueParamsOverrideFriction",
      description="",
      min_value=1,
      max_value=100,
      value_change_step=1,
      label_callback=(lambda x: f"{x/100}"),
      use_float_scaling=True
    )

    # Hyundai Inster only (torque controller v0 without NNLC; see sunnypilot/selfdrive/controls/lib/inster_lateral.py)
    self._inster_friction_toggle = toggle_item_sp(
      param="InsterStictionFriction",
      title=lambda: tr("Inster: Stiction Friction (experimental)"),
      description=lambda: tr("Hyundai Inster only. Replaces the friction part of the torque controller with a smooth term matched to " +
                             "the Inster's steering friction, so the wheel starts moving sooner when steering into and out of curves. " +
                             "Off by default."),
    )
    self._inster_ff_toggle = toggle_item_sp(
      param="InsterSpeedFeedforward",
      title=lambda: tr("Inster: Speed Feed-Forward Table (experimental)"),
      description=lambda: tr("Hyundai Inster only. Uses a measured table of steering torque per speed and sideways acceleration for " +
                             "the feed-forward instead of one linear factor. Off by default."),
    )
    self._inster_angle_pi_toggle = toggle_item_sp(
      param="InsterAngleErrorPI",
      title=lambda: tr("Inster: Steering-Angle PI \"F\" (experimental)"),
      description=lambda: tr("Hyundai Inster only. Replaces the torque controller's P and I with ones that act on the steering-wheel " +
                             "angle error, with speed-dependent gains and a stronger integrator that also works at low speed. " +
                             "Tuned in simulation with both Inster switches above on. Turning it on switches the Lateral Jerk " +
                             "Torque Controller off. Off by default."),
    )
    self._inster_angle_pi_gains = multiple_button_item_sp(
      param=ANGLE_PI_GAINS_PARAM,
      title=lambda: tr("Inster: Angle-PI Gain Set (experimental)"),
      description=lambda: tr("Hyundai Inster only, with the steering-angle PI above on. G2: the default. G6: less P, more I; calmer " +
                             "on straights and drifts wide less, but runs curves a little further inside and releases curve exits " +
                             "slightly later. P6: PID; smaller, smoother corrections, less exit overshoot and closer to the line " +
                             "in mid-speed curves; wider in slow tight turns, a little further inside on the motorway, and slightly " +
                             "later exit release. Simulation only, not validated on the car."),
      buttons=["G2", "G6", "P6"],
      inline=False,
      button_width=250,
    )
    # the gain set in force (also before the G6 bool's migration ran)
    self._inster_angle_pi_gains.action_item.set_selected_button(angle_pi_gain_set(ui_state.params))
    self._inster_soft_p_toggle = toggle_item_sp(
      param=SOFT_P_PARAM,
      title=lambda: tr("Inster: Soft Small-Error P, for P6 (experimental)"),
      description=lambda: tr("Softer P for small errors on straights, stronger P in curves. In simulation (with P6) much calmer on " +
                             "straights; at 30-80 km/h slightly closer inside but wider outside; slightly more lateral offset on " +
                             "straights and more exit overshoot. Fitted for P6. Not validated on the car. Off by default."),
    )

    items = [
      self._jerk_aware_toggle,
      self._torque_control_versions,
      self._self_tune_toggle,
      self._relaxed_tune_toggle,
      self._custom_tune_toggle,
      self._torque_prams_override_toggle,
      self._torque_lat_accel_factor,
      self._torque_friction,
      self._inster_friction_toggle,
      self._inster_ff_toggle,
      self._inster_angle_pi_toggle,
      self._inster_angle_pi_gains,
      self._inster_soft_p_toggle,
    ]
    return items

  def _update_state(self):
    super()._update_state()
    nnlc_enabled = ui_state.params.get_bool("NeuralNetworkLateralControl")
    inster = is_inster(ui_state.CP)
    inster_toggles = (self._inster_friction_toggle, self._inster_ff_toggle, self._inster_angle_pi_toggle)
    inster_switch_on = inster and any(t.action_item.get_state() for t in inster_toggles)
    if inster_switch_on and self._jerk_aware_toggle.action_item.get_state():
      self._jerk_aware_toggle.action_item.set_state(False)  # the param itself is switched off by ui_state._enforce_constraints
    self._jerk_aware_toggle.action_item.set_enabled(ui_state.is_offroad() and not nnlc_enabled and not inster_switch_on)
    for inster_toggle in inster_toggles:
      inster_toggle.set_visible(inster)
      inster_toggle.action_item.set_enabled(ui_state.is_offroad() and not nnlc_enabled)
    # gain set G2 / G6 / P6: a selector of the steering-angle PI, shown only while that is on (same change rules)
    self._inster_angle_pi_gains.set_visible(inster and self._inster_angle_pi_toggle.action_item.get_state())
    self._inster_angle_pi_gains.action_item.set_enabled(ui_state.is_offroad() and not nnlc_enabled)
    # option 5 (soft small-error P): under the selector, shown only while the steering-angle PI is on (same change rules)
    self._inster_soft_p_toggle.set_visible(inster and self._inster_angle_pi_toggle.action_item.get_state())
    self._inster_soft_p_toggle.action_item.set_enabled(ui_state.is_offroad() and not nnlc_enabled)
    if not ui_state.params.get_bool("LiveTorqueParamsToggle"):
      ui_state.params.remove("LiveTorqueParamsRelaxedToggle")
      self._relaxed_tune_toggle.action_item.set_state(False)
    self._self_tune_toggle.action_item.set_enabled(ui_state.is_offroad())
    self._relaxed_tune_toggle.action_item.set_enabled(ui_state.is_offroad() and self._self_tune_toggle.action_item.get_state())
    self._custom_tune_toggle.action_item.set_enabled(ui_state.is_offroad())
    custom_tune_enabled = self._custom_tune_toggle.action_item.get_state()
    self._torque_prams_override_toggle.set_visible(custom_tune_enabled)
    self._torque_lat_accel_factor.set_visible(custom_tune_enabled)
    self._torque_friction.set_visible(custom_tune_enabled)

    self._torque_prams_override_toggle.action_item.set_enabled(ui_state.is_offroad())
    sliders_enabled = self._torque_prams_override_toggle.action_item.get_state() or ui_state.is_offroad()
    self._torque_lat_accel_factor.action_item.set_enabled(sliders_enabled)
    self._torque_friction.action_item.set_enabled(sliders_enabled)

    title_text = tr("Real-Time & Offline") if ui_state.params.get("TorqueParamsOverrideEnabled") else tr("Offline Only")
    self._torque_lat_accel_factor.set_title(lambda: tr("Lateral Acceleration Factor") + " (" + title_text + ")")
    self._torque_friction.set_title(lambda: tr("Friction") + " (" + title_text + ")")
    self._torque_control_versions.action_item.set_value(self._get_current_torque_version_label())

  def _render(self, rect):
    self._back_button.set_position(self._rect.x, self._rect.y + 20)
    self._back_button.render()
    # subtract button
    content_rect = rl.Rectangle(rect.x, rect.y + self._back_button.rect.height + 40, rect.width, rect.height - self._back_button.rect.height - 40)
    self._scroller.render(content_rect)

  def show_event(self):
    self._scroller.show_event()

  def _get_current_torque_version_label(self):
    current_val_bytes = ui_state.params.get("TorqueControlTune")
    if current_val_bytes is None:
      return tr("Default")

    try:
      current_val = float(current_val_bytes)
      for label, info in self.cached_torque_versions.items():
        if math.isclose(float(info["version"]), current_val, rel_tol=1e-5):
          return label
    except (ValueError, KeyError):
      pass

    return tr("Default")

  def _show_torque_version_dialog(self):
    options_map = {}
    for label, info in self.cached_torque_versions.items():
      try:
        options_map[label] = float(info["version"])
      except (ValueError, KeyError):
        pass

    # Sort options by label in descending order
    sorted_labels = sorted(options_map.keys(), key=lambda k: options_map[k], reverse=True)

    nodes = [TreeNode(tr("Default"))]
    for label in sorted_labels:
      nodes.append(TreeNode(label))

    folders = [TreeFolder("", nodes)]

    current_label = self._get_current_torque_version_label()

    def handle_selection(result: int):
      if result == DialogResult.CONFIRM and self._torque_version_dialog:
        selected_ref = self._torque_version_dialog.selection_ref
        if selected_ref == tr("Default"):
          ui_state.params.remove("TorqueControlTune")
        elif selected_ref in options_map:
          ui_state.params.put("TorqueControlTune", options_map[selected_ref])
      self._torque_version_dialog = None

    self._torque_version_dialog = TreeOptionDialog(
      tr("Select Torque Control Tune Version"),
      folders,
      current_ref=current_label,
      option_font_weight=FontWeight.UNIFONT,
      on_exit=handle_selection,
    )
    gui_app.push_widget(self._torque_version_dialog)
