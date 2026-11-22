"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.cereal import custom
from opendbc.car.structs import car
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.selfdrive.car.helpers import is_car_recognised
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.common import Mode as SpeedLimitMode, OffsetType


def get_speed_limit_offset(offset_type: int, offset_value: float, speed_limit: float, is_metric: bool) -> float:
  """Speed Limit Offset setting applied to a speed limit (m/s): None 0, Fixed value in km/h or mph, % of the limit"""
  if offset_type == OffsetType.off:
    return 0.
  elif offset_type == OffsetType.fixed:
    return float(offset_value * (CV.KPH_TO_MS if is_metric else CV.MPH_TO_MS))
  elif offset_type == OffsetType.percentage:
    return float(offset_value * 0.01 * speed_limit)
  else:
    raise NotImplementedError("Offset not supported")


def compare_cluster_target(v_cruise_cluster: float, target_set_speed: float, is_metric: bool) -> tuple[bool, bool]:
  speed_conv = CV.MS_TO_KPH if is_metric else CV.MS_TO_MPH
  v_cruise_cluster_conv = round(v_cruise_cluster * speed_conv)
  target_set_speed_conv = round(target_set_speed * speed_conv)

  req_plus = v_cruise_cluster_conv < target_set_speed_conv
  req_minus = v_cruise_cluster_conv > target_set_speed_conv

  return req_plus, req_minus


def set_speed_limit_assist_availability(CP: car.CarParams, CP_SP: custom.CarParamsSP, params: Params | None = None) -> bool:
  if params is None:
    params = Params()

  # Unrecognised car (mock platform after a failed fingerprint): Speed Limit Assist is not available, but the
  # user's Speed Limit mode is kept as it is for the next recognised start
  if not is_car_recognised(CP):
    return False

  is_release = params.get_bool("IsReleaseSpBranch")
  disallow_in_release = CP.brand == "tesla" and is_release
  always_disallow = CP.brand == "rivian"
  allowed = True

  if disallow_in_release or always_disallow:
    allowed = False

  if not CP.openpilotLongitudinalControl and CP_SP.pcmCruiseSpeed:
    allowed = False

  if not allowed:
    if params.get("SpeedLimitMode", return_default=True) == SpeedLimitMode.assist:
      params.put("SpeedLimitMode", int(SpeedLimitMode.warning), block=True)

  return allowed
