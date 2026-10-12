"""Extended Kalman filter for the TurtleBot3 pose x = [x, y, theta] (world frame).

Predict: wheel-encoder odometry (ds, dth) drives the unicycle motion model.
Update:  ArUco marker measurement z = [range, bearing, marker_yaw] from the
         camera, which sits CAM_X ahead of the wheel axle.
"""
import numpy as np

from . import config as C
from .sensors import wrap


class EKF:
    def __init__(self, x0, P0, meas_model=C.MEAS_MODEL):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.rows = [0, 1, 2] if meas_model == "pose" else [0, 1]
        self.n_reject = 0      # consecutive gated measurements

    # ---------------------------------------------------------- predict ---
    def predict(self, ds, dth):
        th_mid = self.x[2] + dth / 2.0          # midpoint heading (2nd-order accurate)
        c, s = np.cos(th_mid), np.sin(th_mid)
        self.x += [ds * c, ds * s, dth]
        self.x[2] = wrap(self.x[2])

        F = np.array([[1, 0, -ds * s],
                      [0, 1, ds * c],
                      [0, 0, 1]])
        # Jacobian w.r.t. the odometry inputs (ds, dth)
        V = np.array([[c, -0.5 * ds * s],
                      [s, 0.5 * ds * c],
                      [0, 1]])
        a1, a2, a3 = C.ODOM_ALPHA
        sig_s = a1 * abs(ds) + C.ODOM_FLOOR[0]
        sig_th = a2 * abs(dth) + a3 * abs(ds) + C.ODOM_FLOOR[1]
        M = np.diag([sig_s ** 2, sig_th ** 2])
        self.P = F @ self.P @ F.T + V @ M @ V.T

    # ------------------------------------------------------ measurement ---
    @staticmethod
    def h(x):
        """Expected [range, bearing, marker_yaw] and its Jacobian H (3x3)."""
        th = x[2]
        dcx, dcy = -C.CAM_X * np.sin(th), C.CAM_X * np.cos(th)   # d(camera pos)/d(theta)
        cx, cy = x[0] + C.CAM_X * np.cos(th), x[1] + C.CAM_X * np.sin(th)
        dx, dy = C.MARKER_POS[0] - cx, C.MARKER_POS[1] - cy
        q = dx * dx + dy * dy
        r = np.sqrt(q)
        z = np.array([r, wrap(np.arctan2(dy, dx) - th), wrap(C.MARKER_YAW - th)])
        H = np.array([
            [-dx / r, -dy / r, -(dx * dcx + dy * dcy) / r],
            [dy / q, -dx / q, (dy * dcx - dx * dcy) / q - 1.0],
            [0.0, 0.0, -1.0],
        ])
        return z, H

    @staticmethod
    def R_of(z):
        std = C.R_STD0 + C.R_STD_PER_M * z[0]
        return np.diag(std ** 2)

    def update(self, z):
        """Returns (accepted, NIS)."""
        # Marker yaw from PnP is unreliable far away (calibrate.py: ~4% flips, sigma grows
        # with range), so it is only used inside YAW_MAX_RANGE.
        idx = self.rows if z[0] < C.YAW_MAX_RANGE else [0, 1]
        zhat, H = self.h(self.x)
        y = z[idx] - zhat[idx]
        y[1:] = wrap(y[1:])                      # angle residuals
        H = H[idx]
        R = self.R_of(z)[np.ix_(idx, idx)]
        S = H @ self.P @ H.T + R
        nis = float(y @ np.linalg.solve(S, y))
        if nis > C.GATE_CHI2[len(idx)]:          # outlier (e.g. PnP flip) -> reject
            self.n_reject += z[0] < C.YAW_MAX_RANGE      # only trust close, reliable views
            if self.n_reject >= C.RECOVER_AFTER:
                # Several rejections in a row means the FILTER is wrong, not the camera
                # (e.g. the robot was pushed). Inflate P so the marker is accepted again.
                self.P += np.diag(C.RECOVER_STD ** 2)
                self.n_reject = 0
            return False, nis
        self.n_reject = 0
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T   # Joseph form keeps P symmetric PD
        return True, nis
