"""Widgets of the offline maps screen (Inster port, E9), as small subclasses so the upstream files stay unchanged.

- CountryPickDialog: sunnypilot's TreeOptionDialog (search, list) with several ticks instead of one choice. A tap ticks or
  unticks a country (ticked rows use the selected colour); "Select" is enabled once one is ticked; `selected_refs` keeps
  the order of the taps.
- QueueProgressAction: upstream's ProgressBarAction sizes its bar to the text and fills it with the fraction it is given,
  which on a short text looked like a highlight on the first letters. This one draws a track over the whole action width
  and fills it with the fraction of the whole queue (all countries of the run), the text (with the percentage) on top.
- ValueTwoButtonsAction (E11): a country row's value with UPDATE (normal) and DELETE (danger) on its right, laid out like
  sunnypilot's ButtonActionSP (value left of the buttons). Upstream's DualButtonAction takes the full row width with no
  title, so it does not fit a country row. Each button calls its own callback; the row's callback is not used.
"""
import math

import pyray as rl

from openpilot.system.ui.lib.application import FontWeight, TextAlignment, TextAlignmentVertical
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.sunnypilot.lib.styles import style
from openpilot.system.ui.sunnypilot.widgets.list_view import ButtonActionSP
from openpilot.system.ui.sunnypilot.widgets.progress_bar import ProgressBarAction
from openpilot.system.ui.sunnypilot.widgets.tree_dialog import TreeItemWidget, TreeOptionDialog
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.list_view import BUTTON_BORDER_RADIUS, BUTTON_FONT_SIZE, BUTTON_FONT_WEIGHT, BUTTON_HEIGHT, \
                                                  BUTTON_WIDTH, TEXT_PADDING, _resolve_value

TRACK_COLOR = rl.Color(60, 60, 60, 255)
FILL_COLOR = rl.Color(30, 121, 232, 255)


class _TickedItem(TreeItemWidget):
  """A tree row whose selected state comes from the dialog's ticks (upstream sets it from a single choice each frame)."""
  dialog: "CountryPickDialog"

  @property
  def selected(self):
    return self.ref in self.dialog.selected_refs

  @selected.setter
  def selected(self, _value):
    pass

  def set_button_style(self, _style):
    super().set_button_style(ButtonStyle.PRIMARY if self.selected else ButtonStyle.NORMAL)


class CountryPickDialog(TreeOptionDialog):
  def __init__(self, title, folders, **kwargs):
    self.selected_refs: list[str] = []
    super().__init__(title, folders, current_ref="", **kwargs)
    self.current = ""  # upstream enables "Select" while selection != current

  def _select_node(self, node):
    if node.ref in self.selected_refs:
      self.selected_refs.remove(node.ref)
    else:
      self.selected_refs.append(node.ref)
    self.selection = ",".join(self.selected_refs)
    self.selection_ref = self.selection

  def _build_visible_items(self, reset_scroll=True):
    super()._build_visible_items(reset_scroll)
    for item in self.visible_items:
      item.__class__ = _TickedItem  # same object, ticks from this dialog
      item.dialog = self


class QueueProgressAction(ProgressBarAction):
  def __init__(self, width=1000):
    super().__init__(width=width)

  def _render(self, rect: rl.Rectangle):
    font_size = 40
    bar_height = 60
    bar = rl.Rectangle(rect.x, rect.y + (rect.height - bar_height) / 2, rect.width, bar_height)
    if self.show_progress:
      rl.draw_rectangle_rounded(bar, 0.2, 10, TRACK_COLOR)
      fill = max(0.0, min(1.0, self.progress / 100.0)) * (bar.width - 8)
      if fill > 0:
        rl.draw_rectangle_rounded(rl.Rectangle(bar.x + 4, bar.y + 4, fill, bar.height - 8), 0.2, 10, FILL_COLOR)
    size = measure_text_cached(self._font, self.text, font_size)
    x = bar.x + (bar.width - size.x) / 2 if self.show_progress else bar.x + bar.width - size.x
    color = rl.WHITE if self.show_progress else self.text_color
    rl.draw_text_ex(self._font, self.text, rl.Vector2(x, bar.y + (bar_height - size.y) / 2), font_size, 0, color)


BUTTON_GAP = 20


class ValueTwoButtonsAction(ButtonActionSP):
  """Value, then the left button (ButtonActionSP's own, e.g. UPDATE), then a second button (DELETE, danger style)."""

  def __init__(self, left_text, right_text, left_callback=None, right_callback=None, enabled=True):
    super().__init__(text=left_text, width=2 * BUTTON_WIDTH + BUTTON_GAP, enabled=enabled)
    self.left_callback, self.right_callback = left_callback, right_callback
    self._right_text_source = right_text
    self._right_enabled = True
    self._right_pressed = False

    def right_pressed():
      self._right_pressed = True
    self._right_button = Button(right_text, font_size=BUTTON_FONT_SIZE, font_weight=BUTTON_FONT_WEIGHT, button_style=ButtonStyle.DANGER,
                                border_radius=BUTTON_BORDER_RADIUS, click_callback=right_pressed, text_padding=0)

  def set_right_text(self, text):
    self._right_text_source = text

  def set_left_enabled(self, enabled: bool):
    self.set_enabled(enabled)

  def set_right_enabled(self, enabled: bool):
    self._right_enabled = enabled

  def set_touch_valid_callback(self, touch_callback) -> None:
    super().set_touch_valid_callback(touch_callback)
    self._right_button.set_touch_valid_callback(touch_callback)

  def get_width_hint(self) -> float:
    buttons = 2 * BUTTON_WIDTH + BUTTON_GAP
    value = self.value
    if value:
      return math.ceil(measure_text_cached(self._font, value, style.ITEM_TEXT_FONT_SIZE).x) + buttons + TEXT_PADDING + 1
    return buttons

  def _render(self, rect: rl.Rectangle) -> bool:
    y = rect.y + (rect.height - BUTTON_HEIGHT) / 2
    right = rl.Rectangle(rect.x + rect.width - BUTTON_WIDTH, y, BUTTON_WIDTH, BUTTON_HEIGHT)
    left = rl.Rectangle(right.x - BUTTON_GAP - BUTTON_WIDTH, y, BUTTON_WIDTH, BUTTON_HEIGHT)
    self._button.set_text(self.text)
    self._button.set_enabled(_resolve_value(self.enabled))
    self._button.render(left)
    self._right_button.set_text(_resolve_value(self._right_text_source, ""))
    self._right_button.set_enabled(self._right_enabled)
    self._right_button.render(right)
    value_text = self.value
    if value_text:
      value_rect = rl.Rectangle(rect.x, rect.y, left.x - rect.x - TEXT_PADDING, rect.height)
      gui_label(value_rect, value_text, font_size=style.ITEM_TEXT_FONT_SIZE, color=self._value_color, font_weight=FontWeight.NORMAL,
                alignment=TextAlignment.LEFT, alignment_vertical=TextAlignmentVertical.MIDDLE)
    if self._pressed and self.left_callback and _resolve_value(self.enabled):
      self.left_callback()
    if self._right_pressed and self.right_callback and self._right_enabled:
      self.right_callback()
    self._pressed = self._right_pressed = False
    return False
