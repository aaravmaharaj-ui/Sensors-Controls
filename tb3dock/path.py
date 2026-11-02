"""Smooth reference trajectory through waypoints with a trapezoidal speed profile."""
import numpy as np
from scipy.interpolate import CubicSpline


class RefPath:
    def __init__(self, waypoints, v_ref, a_ref, start_heading=0.0, end_heading=0.0):
        w = np.asarray(waypoints, float)
        u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(w, axis=0), axis=1))]
        bc = lambda f: ((1, f(start_heading)), (1, f(end_heading)))
        sx = CubicSpline(u, w[:, 0], bc_type=bc(np.cos))
        sy = CubicSpline(u, w[:, 1], bc_type=bc(np.sin))
        uu = np.linspace(0, u[-1], 4000)
        self.x, self.y = sx(uu), sy(uu)
        dx, dy, ddx, ddy = sx(uu, 1), sy(uu, 1), sx(uu, 2), sy(uu, 2)
        self.th = np.unwrap(np.arctan2(dy, dx))
        self.kappa = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
        self.s = np.r_[0, np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y)))]
        self.L = self.s[-1]

        self.v, self.a = v_ref, a_ref
        self.ta = v_ref / a_ref
        self.da = 0.5 * a_ref * self.ta ** 2
        assert 2 * self.da < self.L, "path too short for the speed profile"
        self.T = 2 * self.ta + (self.L - 2 * self.da) / v_ref

    def _s(self, t):
        """Arc length and speed along the path at time t."""
        if t <= 0:
            return 0.0, 0.0
        if t < self.ta:
            return 0.5 * self.a * t * t, self.a * t
        if t < self.T - self.ta:
            return self.da + self.v * (t - self.ta), self.v
        if t < self.T:
            tr = self.T - t
            return self.L - 0.5 * self.a * tr * tr, self.a * tr
        return self.L, 0.0

    def at(self, t):
        """Reference (x, y, theta, v, omega) at time t."""
        s, v = self._s(t)
        x = np.interp(s, self.s, self.x)
        y = np.interp(s, self.s, self.y)
        th = np.interp(s, self.s, self.th)
        k = np.interp(s, self.s, self.kappa)
        return np.array([x, y, th, v, v * k])
