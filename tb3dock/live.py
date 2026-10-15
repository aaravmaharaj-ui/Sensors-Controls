
import numpy as np
import cv2
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .plots import draw_room, cov_ellipse
from . import config as C

SPLIT = 0.5             # share of the window width given to the 3D view (left)
BG = (30, 30, 34)       # background colour of the right half (RGB)


class LiveOverlay:
    def __init__(self, view, cam, ref_path):
        self.view, self.cam = view, cam
        self.ok = all(hasattr(view, a) for a in ("set_images", "set_texts", "viewport", "cam", "m"))
        if not self.ok:
            print("Note: this MuJoCo version has no viewer overlays; showing the 3D view only.")
            return
        self.fig, self.ax = plt.subplots(figsize=(5.2, 3.9), dpi=90)   # 4:3, same shape as the camera
        self.canvas = FigureCanvasAgg(self.fig)
        self.ref_xy = np.c_[ref_path.x, ref_path.y]
        self.n, self.map_img, self.framed = 0, None, False
        self.angles = np.linspace(-np.pi, np.pi, C.LIDAR_N, endpoint=False)

    def _frame_scene(self, vp):
        """Point the viewer camera so the room sits in the LEFT part of the window."""
        m, cam = self.view.m, self.view.cam
        aspect = vp.width / vp.height
        t = np.tan(np.deg2rad(m.vis.global_.fovy) / 2)
        d = (5.4 / SPLIT) / 2 / (t * aspect)                 # fit the 4.2 m room plus margin
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [1.5 + 0.85 * (1 - SPLIT) * d * t * aspect, 0.0, 0.0]
        cam.distance, cam.azimuth, cam.elevation = d, 90.0, -60.0
        self.framed = True

    @staticmethod
    def _fit(img, w, h):
        """Resize keeping the aspect ratio so it fits inside w x h."""
        s = min(w / img.shape[1], h / img.shape[0])
        return cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))

    def _map_image(self, L, scan=None):
        ax = self.ax
        ax.clear()
        draw_room(ax)
        xe, xt = np.array(L["x_est"]), np.array(L["x_true"])
        ax.plot(*self.ref_xy.T, "k--", lw=.8)
        ax.plot(xt[:, 0], xt[:, 1], color="#2a9d55", lw=2, label="true (eval only)")
        ax.plot(xe[:, 0], xe[:, 1], color="#1f5fbf", lw=1.2, label="EKF estimate")
        if scan is not None:      # LiDAR hits drawn where the robot THINKS it is
            hit = scan < C.LIDAR_MAX - 1e-3
            x, y, th = xe[-1]
            ax.scatter(x + scan[hit] * np.cos(th + self.angles[hit]),
                       y + scan[hit] * np.sin(th + self.angles[hit]),
                       s=4, c="#d62728", label="LiDAR scan", zorder=3)
        ax.add_patch(cov_ellipse(L["P"][-1], xe[-1, :2], 3, fill=True, fc="#1f5fbf",
                                 alpha=.25, ec="#1f5fbf", label="3σ"))
        seen = bool(L["accepted"][-1])
        ax.set_title("marker seen" if seen else "odometry only", fontsize=10,
                     color="#e08a00" if seen else "0.4")
        ax.set_aspect("equal"); ax.set_xlim(-0.7, 3.7); ax.set_ylim(-1.7, 1.7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="upper left", fontsize=7)
        self.fig.tight_layout()
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3]

    def update(self, L, scan=None):
        if not self.ok:
            return
        try:
            vp = self.view.viewport
            if vp is None or vp.width < 300 or vp.height < 300:
                return
            if not self.framed:
                self._frame_scene(vp)
            pad = 12
            rw = int(vp.width * (1 - SPLIT))                 # right half
            panel = np.empty((vp.height, rw, 3), np.uint8)
            panel[:] = BG
            tile_w, tile_h = rw - 2 * pad, (vp.height - 3 * pad) // 2
            if self.map_img is None or self.n % 3 == 0:      # map is slower to draw: ~3 Hz
                self.map_img = self._map_image(L, scan)
            self.n += 1
            tiles = [self.cam.last_image, self.map_img]      # top: camera, bottom: map
            for i, img in enumerate(tiles):
                if img is None:
                    continue
                im = self._fit(img, tile_w, tile_h)
                y = pad + i * (tile_h + pad) + (tile_h - im.shape[0]) // 2
                x = pad + (tile_w - im.shape[1]) // 2
                panel[y:y + im.shape[0], x:x + im.shape[1]] = im
            rect = mujoco.MjrRect(vp.left + vp.width - rw, vp.bottom, rw, vp.height)
            images = [(rect, panel)]
            self.view.set_images(images)

            err = np.hypot(*(np.array(L["x_true"][-1][:2]) - L["x_est"][-1][:2]))
            status = "marker seen" if L["accepted"][-1] else "odometry only"
            self.view.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                                 mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                 "time\nestimate\nest. error",
                                 f"{L['t'][-1]:5.1f} s\n{status}\n{100 * err:4.1f} cm"))
        except Exception as e:      # overlays must never stop the run
            print(f"Overlay disabled ({type(e).__name__}: {e})")
            self.ok = False

    def close(self):
        if self.ok:
            plt.close(self.fig)