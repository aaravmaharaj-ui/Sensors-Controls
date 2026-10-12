"""LQR trajectory tracker for a unicycle, running on the ESTIMATED pose.

Tracking error in the robot frame (Kanayama et al., 1990):
    e_x  = cos(th)(x_r - x) + sin(th)(y_r - y)      along-track
    e_y  = -sin(th)(x_r - x) + cos(th)(y_r - y)     lateral
    e_th = th_r - th
Linearised about the reference (v_r, w_r) with inputs
u = [v_r cos(e_th) - v, w_r - w]:
    e_dot = A e + B u,  A = [[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0]],
                        B = [[1, 0], [0, 0], [0, 1]]
(A, B) is controllable only if v_r or w_r is non-zero, so we linearise at
v_r >= V_FLOOR. The gain is recomputed every step (gain scheduling).
"""
import numpy as np
from scipy.linalg import solve_continuous_are

from . import config as C
from .sensors import wrap


def error_model(vr, wr):
    A = np.array([[0.0, wr, 0.0], [-wr, 0.0, vr], [0.0, 0.0, 0.0]])
    B = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    return A, B


class LQRTracker:
    def __init__(self, Q=C.LQR_Q, R=C.LQR_R):
        self.Q, self.R = Q, R

    def gain(self, vr, wr):
        A, B = error_model(max(vr, C.V_FLOOR), wr)
        P = solve_continuous_are(A, B, self.Q, self.R)
        return np.linalg.solve(self.R, B.T @ P)

    def __call__(self, x_est, ref):
        """x_est: EKF pose estimate. ref: (x_r, y_r, th_r, v_r, w_r).
        Returns (v, w, tracking error in robot frame)."""
        x, y, th = x_est
        xr, yr, thr, vr, wr = ref
        c, s = np.cos(th), np.sin(th)
        e = np.array([c * (xr - x) + s * (yr - y),
                      -s * (xr - x) + c * (yr - y),
                      wrap(thr - th)])
        du = self.gain(vr, wr) @ e            # u = -K e  =>  v = v_r cos e_th + (K e)_0
        v = vr * np.cos(e[2]) + du[0]
        w = wr + du[1]
        v = np.clip(v, -0.05, C.V_MAX)        # allow a little reverse for docking corrections
        w = np.clip(w, -C.W_MAX, C.W_MAX)
        return v, w, e


def wheel_speeds(v, w):
    """Unicycle (v, w) -> wheel angular velocities [rad/s]."""
    b, r = C.ODOM_TRACK_EFF, C.ODOM_R_EFF
    return (v - w * b / 2) / r, (v + w * b / 2) / r
