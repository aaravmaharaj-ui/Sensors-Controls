
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Arc

from tb3dock import config as C
from tb3dock.plots import draw_room

ROOT = os.path.dirname(os.path.abspath(__file__))
RED, GREEN, BLUE, GREY = "#d62728", "#2a9d55", "#1f5fbf", "0.35"


def axis(ax, p, ang, length, color, label, lw=2.2, fs=10, off=1.15):
    d = np.array([np.cos(ang), np.sin(ang)])
    ax.add_patch(FancyArrowPatch(p, p + length * d, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, zorder=6))
    ax.text(*(p + off * length * d), label, color=color, fontsize=fs, ha="center", va="center",
            fontweight="bold", zorder=7)


fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.45, 1]})

# ------------------------------------------------------------- top-down view
a.add_patch(Rectangle((-0.6, -1.6), 4.2, 3.2, fill=False, lw=2, ec="0.6"))
for kind, ox, oy, h in [("box", 0.9, -0.15, 0.2), ("box", 1.6, 0.55, 0.15), ("cyl", 2.1, -0.35, 0.2), ("box", 2.35, 0.6, 0.15)]:
    a.add_patch(Circle((ox, oy), h, color="#8059bf", alpha=.2) if kind == "cyl"
                else Rectangle((ox - h, oy - h), 2 * h, 2 * h, color="#8059bf", alpha=.2))
a.add_patch(Rectangle((3.31, -0.15), 0.04, 0.3, color="#3366cc"))
L = 0.38
# room / world frame at the start point
axis(a, np.zeros(2), 0, L, RED, "$x_{room}$", off=1.35)
axis(a, np.zeros(2), np.pi / 2, L, GREEN, "$y_{room}$", off=1.2)
a.plot(0, 0, "ko", ms=5, zorder=8)
a.text(-0.08, -0.12, "room frame\n(robot start)", fontsize=9, ha="right", va="top")

# robot at an example pose
x, y, th = 1.55, -1.05, np.deg2rad(8)
P = np.array([x, y])
a.add_patch(Circle(P, 0.10, fc="0.2", ec="k", alpha=.85, zorder=4))
axis(a, P, th, L, RED, "$x_{robot}$", off=1.32)
axis(a, P, th + np.pi / 2, L, GREEN, "$y_{robot}$", off=1.25)
a.text(x - 0.35, y - 0.17, "robot frame (wheel axle)\npose $(x, y, \\theta)$ from the EKF", fontsize=9, va="top")
# theta arc
a.plot([x, x + 0.3], [y, y], color=GREY, lw=.8, ls="--")
a.add_patch(Arc(P, 0.42, 0.42, theta1=0, theta2=np.rad2deg(th), color=GREY, lw=1.2))
a.text(x + 0.25, y + 0.06, "$\\theta$", fontsize=12, color=GREY)

# camera point and the measurement to the marker
cam = P + C.CAM_X * np.array([np.cos(th), np.sin(th)])
M = np.array(C.MARKER_POS)
a.plot(*cam, "s", color=BLUE, ms=6, zorder=8)
a.plot([cam[0], M[0]], [cam[1], M[1]], ls="--", color=BLUE, lw=1.3, zorder=5)
mid = (cam + M) / 2
a.text(mid[0] + 0.12, mid[1] - 0.16, "range $\\rho$", color=BLUE, fontsize=11)
bear = np.arctan2(M[1] - cam[1], M[0] - cam[0])
a.add_patch(Arc(cam, 1.1, 1.1, theta1=np.rad2deg(min(bear, th)), theta2=np.rad2deg(max(bear, th)), color=BLUE, lw=1.3))
a.plot([cam[0], cam[0] + 0.75 * np.cos(th)], [cam[1], cam[1] + 0.75 * np.sin(th)], color=BLUE, lw=.8, ls=":")
bm = (bear + th) / 2
a.text(*(cam + 0.6 * np.array([np.cos(bm), np.sin(bm)])), "  bearing $\\beta$", color=BLUE, fontsize=10, va="center")

# marker frame (normal points back into the room, along -x)
axis(a, M, C.MARKER_YAW, 0.32, BLUE, "$z_{marker}$", off=1.45)
a.text(M[0] - 0.02, M[1] + 0.32, "marker frame\n(known pose)", fontsize=9, ha="right")
a.text(M[0] - 0.45, M[1] + 0.75, "marker angle $\\psi$\n= marker yaw $-\\ \\theta$", fontsize=9, color=BLUE, ha="center")

a.set_aspect("equal"); a.set_xlim(-0.7, 3.7); a.set_ylim(-1.7, 1.7)
a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
a.set_title("Top view: room, robot and marker frames", fontsize=12)

# ---------------------------------------------------------------- side view
s = 100  # draw in cm
axle_h = C.WHEEL_R * s
b.axhline(0, color="k", lw=1.5)
b.add_patch(Circle((0, axle_h), C.WHEEL_R * s, fc="0.55", ec="k", zorder=2))                      # wheel
b.add_patch(Rectangle((-9.5, axle_h + 0.5), 14, 7, fc="0.25", ec="k", alpha=.85, zorder=1))     # chassis
b.add_patch(Rectangle((-3, axle_h + 7.5), 6, 3, fc="0.1", ec="k", zorder=3))                    # LiDAR head
b.add_patch(Rectangle((C.CAM_X * s - 1, axle_h + 9), 2, 2, fc=BLUE, ec="k", zorder=4))          # camera

O = np.array([0.0, axle_h])
axis(b, O, 0, 6, RED, "$x_{robot}$", fs=10, off=1.35)
axis(b, O, np.pi / 2, 5, "w", "", fs=10, off=1.3)
b.plot(*O, "ko", ms=5, zorder=8)
b.text(-5, -2.2, "robot frame: wheel axle centre", fontsize=9, ha="right", va="center")

lid = O + [0, C.LIDAR_HEIGHT * s]
b.plot(*lid, "o", color="#d62728", ms=6, zorder=8)
b.plot([lid[0] - 13, lid[0] + 13], [lid[1], lid[1]], color="#d62728", lw=1, ls=":", zorder=0)
b.text(lid[0] - 16, lid[1] + 2.5, "LiDAR scan plane\n(same x, y as robot)", fontsize=9, color="#d62728", ha="right", va="center")

camp = O + [C.CAM_X * s, 0.10 * s]
axis(b, camp, 0, 6, BLUE, "$z_{cam}$", fs=10, off=1.3)
axis(b, camp, -np.pi / 2, 4, BLUE, "$y_{cam}$", fs=10, off=1.35)
b.text(camp[0] + 0.6, camp[1] + 1.6, "camera (OpenCV axes:\nz forward, y down, x right)", fontsize=9, color=BLUE)

# dimensions
def dim(p0, p1, text, offset, horiz):
    b.annotate("", p1, p0, arrowprops=dict(arrowstyle="<->", color=GREY, lw=1))
    m = (np.array(p0) + p1) / 2
    b.text(m[0] + (0 if horiz else offset), m[1] + (offset if horiz else 0), text, fontsize=9,
           color=GREY, ha="center" if horiz else "left", va="center")

dim((0, -1.2), (C.CAM_X * s, -1.2), f"{C.CAM_X*s:.0f} cm", -1.1, True)
dim((14.5, axle_h), (14.5, axle_h + 10), "10 cm", 0.5, False)
dim((-15, axle_h), (-15, axle_h + C.LIDAR_HEIGHT * s), f"{C.LIDAR_HEIGHT*s:.0f} cm", 0.5, False)
dim((4.5, 0), (4.5, axle_h), f"{C.WHEEL_R*s:.1f} cm", 0.5, False)

b.set_aspect("equal"); b.set_xlim(-30, 22); b.set_ylim(-4, 22)
b.set_xlabel("forward [cm]"); b.set_ylabel("height [cm]")
b.set_title("Side view: sensor mounts on the robot", fontsize=12)

fig.tight_layout()
out = os.path.join(ROOT, "results", "frames.png")
fig.savefig(out, dpi=140)
print(f"saved {out}")