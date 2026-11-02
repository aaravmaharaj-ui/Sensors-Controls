
import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import config as C
from .path import RefPath


class OccupancyGrid:
    def __init__(self):
        self.res = C.NAV_RES
        self.x0, self.y0 = C.NAV_BOUNDS[0], C.NAV_BOUNDS[2]
        self.nx = int(round((C.NAV_BOUNDS[1] - C.NAV_BOUNDS[0]) / self.res))
        self.ny = int(round((C.NAV_BOUNDS[3] - C.NAV_BOUNDS[2]) / self.res))
        self.logodds = np.zeros((self.ny, self.nx))
        self.angles = np.linspace(-np.pi, np.pi, C.LIDAR_N, endpoint=False)

    def cell(self, x, y):
        return (np.floor((np.asarray(y) - self.y0) / self.res).astype(int),
                np.floor((np.asarray(x) - self.x0) / self.res).astype(int))

    def centre(self, r, c):
        return self.x0 + (c + 0.5) * self.res, self.y0 + (r + 0.5) * self.res

    def inside(self, r, c):
        return (r >= 0) & (r < self.ny) & (c >= 0) & (c < self.nx)

    def update(self, pose, scan):
        """Add one scan taken at the ESTIMATED pose."""
        x, y, th = pose
        a = th + self.angles
        hit = scan < C.LIDAR_MAX - 0.05      # max-range readings are 'nothing there'
        # free space: points along every beam, stopping one cell short of the hit
        steps = np.arange(self.res, C.LIDAR_MAX, self.res)
        reach = np.where(hit, scan - self.res, C.LIDAR_MAX)
        d = steps[None, :] * np.ones((len(a), 1))
        mask = d < reach[:, None]
        fx = (x + d * np.cos(a)[:, None])[mask]
        fy = (y + d * np.sin(a)[:, None])[mask]
        r, c = self.cell(fx, fy)
        ok = self.inside(r, c)
        free = np.zeros_like(self.logodds, dtype=bool)
        free[r[ok], c[ok]] = True
        self.logodds[free] += C.NAV_L_FREE
        # occupied: the beam end points
        hx, hy = x + scan[hit] * np.cos(a[hit]), y + scan[hit] * np.sin(a[hit])
        r, c = self.cell(hx, hy)
        ok = self.inside(r, c)
        occ = np.zeros_like(free)
        occ[r[ok], c[ok]] = True
        self.logodds[occ] += C.NAV_L_OCC
        np.clip(self.logodds, -C.NAV_L_MAX, C.NAV_L_MAX, out=self.logodds)

    def occupied(self):
        return self.logodds > C.NAV_L_THRESH

    def clearance(self):
        """Distance [m] from every cell to the nearest occupied cell."""
        occ = self.occupied()
        if not occ.any():
            return np.full(occ.shape, 10.0)
        return distance_transform_edt(~occ) * self.res


class Navigator:
    """Plans and re-plans the reference path from the LiDAR map."""

    def __init__(self):
        self.grid = OccupancyGrid()
        self.path = None
        self.t0 = 0.0          # time the current path started
        self.n_plans = 0
        self.history = []      # (time, reason) of every plan

    # ------------------------------------------------------------- planning
    def _astar(self, start, goal, clear):
        g = self.grid
        s, e = tuple(int(v) for v in g.cell(*start)), tuple(int(v) for v in g.cell(*goal))
        blocked = clear < C.NAV_INFLATE
        blocked[s] = False                            # never block where we stand
        cost = 1.0 + C.NAV_COMFORT_W * np.clip((C.NAV_COMFORT - clear) / C.NAV_COMFORT, 0, 1)
        nbrs = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
                (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        h = lambda a: np.hypot(a[0] - e[0], a[1] - e[1])
        best = {s: 0.0}
        came = {}
        openq = [(h(s), 0.0, s)]
        while openq:
            _, gcost, cur = heapq.heappop(openq)
            if cur == e:
                cells = [cur]
                while cur in came:
                    cur = came[cur]
                    cells.append(cur)
                return cells[::-1]
            if gcost > best.get(cur, np.inf):
                continue
            for dr, dc, step in nbrs:
                n = (cur[0] + dr, cur[1] + dc)
                if not (0 <= n[0] < g.ny and 0 <= n[1] < g.nx) or blocked[n]:
                    continue
                ng = gcost + step * cost[n]
                if ng < best.get(n, np.inf):
                    best[n], came[n] = ng, cur
                    heapq.heappush(openq, (ng + h(n), ng, n))
        return None

    def _segment_clear(self, p, q, clear, need):
        n = max(2, int(np.hypot(*(q - p)) / (self.grid.res / 2)))
        pts = p + np.linspace(0, 1, n)[:, None] * (q - p)
        r, c = self.grid.cell(pts[:, 0], pts[:, 1])
        ok = self.grid.inside(r, c)
        return bool(np.all(clear[r[ok], c[ok]] >= need))

    def _shortcut(self, pts, clear):
        """Fewest straight segments through the A* cells that keep NAV_COMFORT clearance."""
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._segment_clear(pts[i], pts[j], clear, C.NAV_COMFORT):
                j -= 1
            out.append(pts[j])
            i = j
        return np.array(out)

    def plan(self, pose, t, reason):
        clear = self.grid.clearance()
        start, goal = np.array(pose[:2]), np.array(C.NAV_PREDOCK)
        cells = self._astar(start, goal, clear)
        if cells is None:                       # no route: keep the old one (or none)
            self.history.append((t, reason + " (no route found)"))
            return False
        pts = np.array([self.grid.centre(r, c) for r, c in cells])
        pts[0], pts[-1] = start, goal
        way = self._shortcut(pts, clear)
        # straight final approach in front of the marker, facing it
        way = np.vstack([way, [C.DOCK_TARGET[0] - 0.12, 0.0], C.DOCK_TARGET[:2]])
        # drop waypoints closer than 10 cm (keeps the spline well behaved)
        keep = [0] + [i for i in range(1, len(way)) if np.hypot(*(way[i] - way[i - 1])) > 0.10]
        way = way[keep]
        path = RefPath(way, C.V_REF, C.A_REF, start_heading=pose[2],
                       end_heading=C.DOCK_TARGET[2])
        if self._path_clearance(path, clear) < C.NAV_BLOCKED:
            # the spline cut a corner: use the A* cells more densely instead
            way = np.vstack([pts[::5], [C.DOCK_TARGET[0] - 0.12, 0.0], C.DOCK_TARGET[:2]])
            keep = [0] + [i for i in range(1, len(way)) if np.hypot(*(way[i] - way[i - 1])) > 0.10]
            way = way[keep]
            path = RefPath(way, C.V_REF, C.A_REF, start_heading=pose[2],
                           end_heading=C.DOCK_TARGET[2])
        self.path = path
        self.path.waypoints = way
        self.t0 = t
        self.n_plans += 1
        self.history.append((t, reason))
        return True

    def _path_clearance(self, path, clear):
        r, c = self.grid.cell(path.x, path.y)
        ok = self.grid.inside(r, c)
        return float(clear[r[ok], c[ok]].min()) if ok.any() else 10.0

    # ------------------------------------------------------------ main hooks
    def blocked_ahead(self, pose, t):
        """True if a newly mapped obstacle sits on the next NAV_LOOKAHEAD m of the route."""
        if self.path is None:
            return True
        clear = self.grid.clearance()
        s_now = self.path._s(t - self.t0)[0]
        sel = (self.path.s >= s_now) & (self.path.s <= s_now + C.NAV_LOOKAHEAD)
        if not sel.any():
            return False
        r, c = self.grid.cell(self.path.x[sel], self.path.y[sel])
        ok = self.grid.inside(r, c)
        return bool(np.any(clear[r[ok], c[ok]] < C.NAV_BLOCKED))

    def ref(self, t):
        return self.path.at(t - self.t0)

    def finished(self, t):
        return self.path is not None and t - self.t0 >= self.path.T