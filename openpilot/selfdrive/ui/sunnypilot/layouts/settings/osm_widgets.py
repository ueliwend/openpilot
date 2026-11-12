"""Widgets of the offline maps screen (Inster port, E9), as small subclasses so the upstream files stay unchanged.

- CountryPickDialog: sunnypilot's TreeOptionDialog (search, list) with several ticks instead of one choice. A tap ticks or
  unticks a country (ticked rows use the selected colour); "Select" is enabled once one is ticked; `selected_refs` keeps
  the order of the taps.
- QueueProgressAction: upstream's ProgressBarAction sizes its bar to the text and fills it with the fraction it is given,
  which on a short text looked like a highlight on the first letters. This one draws a track over the whole action width
  and fills it with the fraction of the whole queue (all countries of the run), the text (with the percentage) on top.
"""
import pyray as rl

from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.sunnypilot.widgets.progress_bar import ProgressBarAction
from openpilot.system.ui.sunnypilot.widgets.tree_dialog import TreeItemWidget, TreeOptionDialog
from openpilot.system.ui.widgets.button import ButtonStyle

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
