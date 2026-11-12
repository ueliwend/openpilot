"""
Plan shift for Blendful (modes Regular and Right line): the lateral correction is delivered as a shift of the model plan, so that
desiredCurvature, the published plan (and with it the InsterMPC preview and the UI) all see the same correction.

Profile (smoothstep): for u = v t / L_c <= 1 the curvature change is dk = k0 (1 - 2u), the heading change k0 L_c (u - u^2) and the
lateral shift D s(u) with s = 3u^2 - 2u^3 and D = k0 L_c^2 / 6.  L_c = sqrt(3) L_a makes the three mutually consistent.
Signs as in the model frame: y > 0 is right, curvature > 0 is a right turn.
"""
import numpy as np

from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan

T_IDXS = np.array(ModelConstants.T_IDXS)
X_IDXS = np.array(ModelConstants.X_IDXS)
APPROACH_TIME = 2.0  # s, L_a = max(APPROACH_TIME v, MIN_APPROACH)
MIN_APPROACH = 10.0  # m
LC_FACTOR = float(np.sqrt(3.0))


def approach_length(v_ego: float, approach_time: float = APPROACH_TIME) -> float:
  return max(approach_time * v_ego, MIN_APPROACH)


def profile_length(v_ego: float, approach_time: float = APPROACH_TIME) -> float:
  return LC_FACTOR * approach_length(v_ego, approach_time)


def smoothstep(z):
  z = np.clip(z, 0.0, 1.0)
  return z * z * (3.0 - 2.0 * z)


def basis(u):
  """Profile basis s(u) = 3u^2 - 2u^3 (u clipped to [0, 1])."""
  u = np.clip(u, 0.0, 1.0)
  return 3.0 * u ** 2 - 2.0 * u ** 3


def fit_amplitude(x: np.ndarray, error: np.ndarray, weight: np.ndarray | float, l_c: float) -> tuple[float, float]:
  """Profile-basis least squares fit of D s(x / L_c) to the error: returns (D, k0 = 6 D / L_c^2)."""
  s = basis(x / l_c)
  w = np.broadcast_to(np.asarray(weight, dtype=float), s.shape)
  den = float(np.sum(w * s * s))
  d = float(np.sum(w * error * s) / den) if den > 0.0 else 0.0
  return d, 6.0 * d / l_c ** 2


def profile(k0: float, v_ego: float, approach_time: float = APPROACH_TIME):
  """(dy, dpsi, drate_per_vx, dk) over T_IDXS for amplitude k0; drate = v_x dk and daccel = v_x^2 dk follow with the plan's own v_x."""
  l_c = profile_length(v_ego, approach_time)
  vt = v_ego * T_IDXS
  u = np.clip(vt / l_c, 0.0, 1.0)
  dk = k0 * (1.0 - 2.0 * u) * (vt <= l_c)
  dpsi = k0 * l_c * (u - u ** 2)
  dy = k0 * l_c ** 2 / 6.0 * basis(u)
  return dy, dpsi, dk


def shift_plan(plan: np.ndarray, k0: float, v_ego: float, approach_time: float = APPROACH_TIME) -> np.ndarray:
  """Return a copy of plan (33 x 15) shifted by the profile.  x, z and the velocities are untouched; the plan's own v_x(t) turns the
  curvature change into orientation rate and lateral acceleration (so the MPC row, rate / v_x, gains exactly dk)."""
  out = np.array(plan, copy=True)
  if k0 == 0.0:
    return out
  dy, dpsi, dk = profile(k0, v_ego, approach_time)
  vx = plan[:, Plan.VELOCITY][:, 0]
  out[:, Plan.POSITION.start + 1] += dy
  out[:, Plan.T_FROM_CURRENT_EULER.start + 2] += dpsi
  out[:, Plan.ORIENTATION_RATE.start + 2] += vx * dk
  out[:, Plan.ACCELERATION.start + 1] += vx ** 2 * dk
  return out


def instant_gain(v_ego: float, lat_action_t: float, approach_time: float = APPROACH_TIME) -> float:
  """F: the instantaneous desired-curvature correction per unit profile amplitude, through get_curvature_from_plan itself."""
  _, dpsi, dk = profile(1.0, v_ego, approach_time)
  return float(get_curvature_from_plan(dpsi, v_ego * dk, T_IDXS, v_ego, lat_action_t))
