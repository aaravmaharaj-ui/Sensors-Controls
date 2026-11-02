
import numpy as np
import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from . import config as C
from .plots import draw_room, cov_ellipse


class Recorder:
    def __init__(self, path, cam, ref_path, fps=10):
        self.w = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=8)
        self.cam = cam                       # reuse the camera's renderer (one GL context)
        self.fig, self.ax = plt.subplots(figsize=(4.8, 4.8), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ref_xy = np.c_[ref_path.x, ref_path.y]

    def frame(self, sim, L):
        r = self.cam.renderer
        r.update_scene(sim.d, camera="overview")
        scene = r.render().copy()
        if self.cam.last_image is not None:
            inset = cv2.resize(self.cam.last_image, (213, 160))
            scene[8:168, -221:-8] = inset
            cv2.rectangle(scene, (640 - 221, 8), (640 - 8, 168), (255, 255, 255), 2)
        t = L["t"][-1]
        cv2.putText(scene, f"t = {t:5.1f} s", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)

        ax = self.ax
        ax.clear()
        draw_room(ax, getattr(self, "obstacles", None))
        g = getattr(self, "grid", None)
        if g is not None:      # the robot's own LiDAR map: occupied cells in dark grey
            ax.imshow(np.where(g.occupied(), 1.0, np.nan), origin="lower", cmap="Greys",
                      vmin=0, vmax=1.3, extent=[g.x0, g.x0 + g.nx * g.res, g.y0, g.y0 + g.ny * g.res],
                      interpolation="nearest", zorder=2)
        xe, xt = np.array(L["x_est"]), np.array(L["x_true"])
        ax.plot(*self.ref_xy.T, "k--", lw=.8)
        ax.plot(xt[:, 0], xt[:, 1], color="#2a9d55", lw=2, label="true")
        ax.plot(xe[:, 0], xe[:, 1], color="#1f5fbf", lw=1.2, label="estimate")
        ax.add_patch(cov_ellipse(L["P"][-1], xe[-1, :2], 3, fill=True, fc="#1f5fbf", alpha=.25,
                                 ec="#1f5fbf", label="3σ"))
        seen = L["accepted"][-1]
        ax.set_title("marker seen" if seen else "odometry only", fontsize=10,
                     color="#e08a00" if seen else "0.4")
        ax.set_aspect("equal"); ax.set_xlim(-0.7, 3.7); ax.set_ylim(-1.7, 1.7)
        ax.legend(loc="upper left", fontsize=7)
        self.fig.tight_layout()
        self.canvas.draw()
        plot = np.asarray(self.canvas.buffer_rgba())[:, :, :3]
        plot = cv2.resize(plot, (480, 480))
        self.w.append_data(np.hstack([scene, plot]))

    def close(self):
        self.w.close()
        plt.close(self.fig)