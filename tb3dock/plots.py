
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse

from . import config as C
from .sensors import wrap


DEFAULT_OBSTACLES = [("box", 0.9, -0.15, 0.2), ("box", 1.6, 0.55, 0.15),
                     ("cyl", 2.1, -0.35, 0.2), ("box", 2.35, 0.6, 0.15)]


def draw_room(ax, obstacles=None):
    """obstacles: list of (kind, x, y, half-size) from Sim.obstacles; None = default layout."""
    ax.add_patch(Rectangle((-0.6, -1.6), 4.2, 3.2, fill=False, lw=2, ec="0.3"))
    for kind, x, y, h in (DEFAULT_OBSTACLES if obstacles is None else obstacles):
        if kind == "cyl":
            ax.add_patch(Circle((x, y), h, color="#8059bf", alpha=.8))
        else:
            ax.add_patch(Rectangle((x - h, y - h), 2 * h, 2 * h, color="#8059bf", alpha=.8))
    ax.add_patch(Rectangle((3.31, -0.15), 0.04, 0.3, color="#3366cc"))
    ax.plot(*C.MARKER_POS, "ks", ms=6)
    ax.annotate("ArUco dock", C.MARKER_POS, (2.75, -0.45), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=.8))


def cov_ellipse(P, xy, nsig=3, **kw):
    vals, vecs = np.linalg.eigh(P[:2, :2])
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * nsig * np.sqrt(np.maximum(vals[::-1], 0))
    return Ellipse(xy, w, h, angle=ang, **kw)


def summary_figure(log, m, path):
    t, xe, xt, P, ref = log["t"], log["x_est"], log["x_true"], log["P"], log["ref"]
    seen = np.isfinite(log["meas"][:, 0]) & log["accepted"]
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1])

    ax = fig.add_subplot(gs[:, 0])
    draw_room(ax, log.get("obstacles"))
    if "occ" in log:       # the robot's LiDAR map at the end of the run
        ax.imshow(np.where(log["occ"], 1.0, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=1.3,
                  extent=log["occ_extent"], interpolation="nearest", zorder=2)
        ax.plot([], [], "s", color="0.3", ms=6, label="LiDAR map (occupied)")
    ax.plot(*log["path_xy"].T, "k--", lw=1, label="reference")
    ax.plot(xt[:, 0], xt[:, 1], color="#2a9d55", lw=2, label="true (eval only)")
    ax.plot(xe[:, 0], xe[:, 1], color="#1f5fbf", lw=1.3, label="EKF estimate")
    ax.plot(xe[seen, 0], xe[seen, 1], ".", color="#e08a00", ms=4, label="marker update")
    for i in range(0, len(t), int(1.5 / C.DT)):
        ax.add_patch(cov_ellipse(P[i], xe[i, :2], fill=False, ec="#1f5fbf", lw=.8, alpha=.7))
    ax.set_aspect("equal"); ax.set_xlim(-0.7, 3.7); ax.set_ylim(-1.7, 1.7)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ok = "DOCKED" if m["dock_success"] else "MISSED"
    ax.set_title(f"{ok}: final error {100*m['dock_pos_err_m']:.1f} cm, "
                 f"{m['dock_yaw_err_deg']:.1f} deg  (ellipses = 3σ)")
    ax.legend(loc="upper left", fontsize=8)

    err = xt - xe; err[:, 2] = wrap(err[:, 2])
    sig = np.sqrt(np.einsum("nii->ni", P))
    labels = ["x error [m]", "y error [m]", "θ error [deg]"]
    for j in range(3):
        a = fig.add_subplot(gs[j, 1])
        k = np.rad2deg if j == 2 else (lambda v: v)
        a.fill_between(t, -3 * k(sig[:, j]), 3 * k(sig[:, j]), color="#1f5fbf", alpha=.15, label="±3σ")
        a.plot(t, k(err[:, j]), color="#1f5fbf", lw=1.2, label="true − estimate")
        for ts in t[seen]:
            a.axvline(ts, color="#e08a00", alpha=.05)
        a.axhline(0, color="k", lw=.5)
        a.set_ylabel(labels[j])
        if j == 0:
            a.legend(fontsize=8, loc="upper left")
            a.set_title(f"estimation error  (pos RMSE {100*m['est_pos_rmse_m']:.1f} cm, "
                        f"mean NEES {m['mean_nees']:.1f}; orange = marker seen)", fontsize=10)
    a.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)