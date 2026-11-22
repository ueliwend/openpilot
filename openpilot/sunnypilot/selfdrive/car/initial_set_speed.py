"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Initial set speed when the driver engages and sunnypilot sets the cruise speed itself (openpilot longitudinal, not pcmCruise).
Owner's decisions 16 and 21 Nov 2026 (Inster), in this order:
  1. Cruise -> Speed Limit -> Speed Limit set to Assist, and the car's displayed speed limit and the map's speed limit are both known
     and agree: that limit plus the Speed Limit Offset setting, whether the car is slower or faster than it;
  2. otherwise the current speed, if it is at or above Cruise -> Initial Max Speed (fallback);
  3. otherwise Cruise -> Initial Max Speed (fallback).
Rules 2 and 3 are upstream's max(current speed, setting) (selfdrive/car/cruise.py); this module decides rule 1.
Resume (accelerate / resume button) keeps the last set speed, as before (cruise.py).
The car's limit here is the speed limit the car shows in its cluster: opendbc's CarState.speed_limit_cluster (m/s, 0 = blank; Hyundai
CAN: LKAS12 CF_Lkas_TsrSpeed_Display_Clu, the camera module's displayed value, which fuses sign recognition and navigation and can carry
over from the previous drive; taken as is, the map's agreement is the safety net, owner 22 Nov 2026; every other car 0 = no rule 1).
carStateSP.speedLimit additionally falls back to the head unit's Navi_HU value (Speed Limit Assist keeps using that one). No cereal or
struct field.
"""
from openpilot.common.constants import CV
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.common import Mode as SpeedLimitMode, OffsetType
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.helpers import get_speed_limit_offset

OFFSET_TYPES = tuple(int(t) for t in OffsetType)
MAP_LIMIT_MAX_AGE = 2.  # s, liveMapDataSP is sent at 1 Hz


def current_map_speed_limit(map_data, msg_valid: bool, alive: bool, age: float) -> float:
  """The map's limit for the current road (m/s; not the look-ahead), or 0 unless mapd marks it valid, the message is valid (mapd sets
  valid = the localizer's gpsOK, so a limit kept after the position was lost does not count), alive and at most MAP_LIMIT_MAX_AGE old"""
  if not (msg_valid and alive and 0. <= age <= MAP_LIMIT_MAX_AGE and map_data.speedLimitValid):
    return 0.
  return float(map_data.speedLimit)


def engagement_speed_limits(CI, sm, now: float) -> tuple[float, float]:
  """card, at engagement (after CI.update of this step): the car's displayed speed limit (opendbc CarState.speed_limit_cluster) and
  the map's usable current-road limit (m/s)"""
  return float(CI.CS.speed_limit_cluster), current_map_speed_limit(sm['liveMapDataSP'], sm.valid['liveMapDataSP'], sm.alive['liveMapDataSP'],
                                                               now - sm.recv_time['liveMapDataSP'])


def agreed_speed_limit(car_limit: float, map_limit: float, is_metric: bool) -> float:
  """The road's speed limit (m/s) when the car and the map both know one and it reads the same in the display unit
  (whole km/h or mph, as the car's displayed limit and the map are shown), else 0."""
  if not (car_limit > 0. and map_limit > 0.):
    return 0.
  conv = CV.MS_TO_KPH if is_metric else CV.MS_TO_MPH
  if round(car_limit * conv) != round(map_limit * conv):
    return 0.
  return min(car_limit, map_limit)  # the value Cruise -> Speed Limit -> Customize Source "Combined" would take


def speed_limit_initial_set_speed(car_limit: float, map_limit: float, speed_limit_mode: int, offset_type: int, offset_value: float,
                                  is_metric: bool) -> float | None:
  """Rule 1: set speed (km/h) = agreed limit + Speed Limit Offset, rounded to a whole display unit (km/h, or mph converted back);
  None when rule 1 does not apply (Speed Limit not Assist, a limit missing or disputed, or an offset type this code does not know)."""
  if speed_limit_mode != SpeedLimitMode.assist or offset_type not in OFFSET_TYPES:
    return None
  limit = agreed_speed_limit(car_limit, map_limit, is_metric)
  if limit <= 0.:
    return None
  conv = CV.MS_TO_KPH if is_metric else CV.MS_TO_MPH
  # same sum and rounding as the speed limit resolver and Speed Limit Assist (speed limit + offset, round in the display unit)
  target = round((limit + get_speed_limit_offset(offset_type, offset_value, limit, is_metric)) * conv)
  return float(target) if is_metric else round(target * CV.MPH_TO_KPH, 1)
