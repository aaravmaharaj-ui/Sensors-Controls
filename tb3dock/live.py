
import numpy as np
import cv2
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .plots import draw_room, cov_ellipse


class LiveOverlay:
    def __init__(self, view, cam, ref_path):
        self.view, self.cam = view, cam
        self.ok = all(hasattr(view, a) for a in ("set_images", "set_texts", "viewport"))
        if not self.ok:
            print("Note: this MuJoCo version has no viewer overlays; showing the 3D view only.")
            return
        self.fig, self.ax = plt.subplots(figsize=(4, 4), dpi=80)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ref_xy = np.c_[ref_path.x, ref_path.y]
        self.n, self.map_img = 0, None

    def _map_image(self, L):
        ax = self.ax
        ax.clear()
        draw_room(ax)
        xe, xt = np.array(L["x_est"]), np.array(L["x_true"])
        ax.plot(*self.ref_xy.T, "k--", lw=.8)
        ax.plot(xt[:, 0], xt[:, 1], color="#2a9d55", lw=2, label="true (eval only)")
        ax.plot(xe[:, 0], xe[:, 1], color="#1f5fbf", lw=1.2, label="EKF estimate")
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

    def update(self, L):
        if not self.ok:
            return
        try:
            vp = self.view.viewport
            if vp is None or vp.width < 300 or vp.height < 300:
                return
            pad = 10
            pw = int(min(vp.width * 0.3, (vp.height - 3 * pad) / 1.75))   # panel width
            ch = int(pw * 3 / 4)                                            # camera height (4:3)
            left = vp.left + vp.width - pw - pad
            cam_rect = mujoco.MjrRect(left, vp.bottom + vp.height - ch - pad, pw, ch)
            map_rect = mujoco.MjrRect(left, cam_rect.bottom - pw - pad, pw, pw)
            images = []
            if self.cam.last_image is not None:
                images.append((cam_rect, cv2.resize(self.cam.last_image, (pw, ch))))
            if self.map_img is None or self.n % 3 == 0:      # map is slower to draw: ~3 Hz
                self.map_img = self._map_image(L)
            self.n += 1
            images.append((map_rect, cv2.resize(self.map_img, (pw, pw))))
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