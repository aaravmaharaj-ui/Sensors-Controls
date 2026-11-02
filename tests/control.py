
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tb3dock import config as C
from tb3dock.controller import LQRTracker
from tb3dock.run import run_trial

ROOT = os.path.dirname(os.path.abspath(__file__))

SETTINGS = [
    ("soft  Q = diag(1, 2, 0.2)", np.diag([1.0, 2.0, 0.2]), "#e08a00"),
    ("ours  Q = diag(10, 20, 2)", C.LQR_Q, "#1f5fbf"),
    ("aggressive  Q = diag(100, 200, 20)", np.diag([100.0, 200.0, 20.0]), "#d62728"),
]

def jerk(log):
    """RMS rate of change of the turn-rate command: how hard the steering works."""
    return float(np.sqrt(np.mean((np.diff(log["u"][:, 1]) / C.DT) ** 2)))


runs = []
for name, Q, col in SETTINGS:
    print(f"running {name} ...")
    LQRTracker.__init__.__defaults__ = (Q, C.LQR_R)
    log, m = run_trial(seed=1, verbose=False)
    runs.append((name, col, log, m))
LQRTracker.__init__.__defaults__ = (C.LQR_Q, C.LQR_R)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# left: our controller in the normal run
name, col, log, m = runs[1]
t, e = log["t"], log["e_ctrl"]
a = ax[0]
a.plot(t, 100 * e[:, 1], color="#1f5fbf", lw=2, label="lateral error [cm]")
a.plot(t, np.rad2deg(e[:, 2]), color="#2a9d55", lw=1.5, label="heading error [deg]")
a.axhline(0, color="k", lw=.5)
a.set_xlabel("time [s]", fontsize=12)
a.set_title(f"Our LQR tracking the planned route (estimate vs route)\n"
            f"lateral RMSE {100*np.sqrt(np.mean(e[:, 1]**2)):.1f} cm, "
            f"heading RMSE {np.rad2deg(np.sqrt(np.mean(e[:, 2]**2))):.1f}°, "
            f"docked {100*m['dock_pos_err_m']:.1f} cm / {m['dock_yaw_err_deg']:.1f}°", fontsize=11)
a.legend(fontsize=10)

# right: tuning comparison
a = ax[1]
for name, col, log, m in runs:
    e = log["e_ctrl"]
    a.plot(log["t"], 100 * e[:, 1], color=col, lw=1.6,
           label=f"{name}: lateral RMSE {100*np.sqrt(np.mean(e[:, 1]**2)):.1f} cm, "
                 f"steering jerk {jerk(log):.1f} rad/s², "
                 f"dock {100*m['dock_pos_err_m']:.1f} cm" + ("" if m["dock_success"] else " (MISSED)"))
a.axhline(0, color="k", lw=.5)
a.set_xlabel("time [s]", fontsize=12); a.set_ylabel("lateral tracking error [cm]", fontsize=12)
a.set_title("LQR tuning: same run with three Q weightings", fontsize=11)
a.legend(fontsize=8.5, loc="lower left")

fig.tight_layout()
out = os.path.join(ROOT, "results", "control.png")
fig.savefig(out, dpi=140)
for name, col, log, m in runs:
    e = log["e_ctrl"]
    print(f"{name:38s} lateral RMSE {100*np.sqrt(np.mean(e[:, 1]**2)):.2f} cm, "
          f"heading RMSE {np.rad2deg(np.sqrt(np.mean(e[:, 2]**2))):.2f} deg, "
          f"steering jerk {jerk(log):.2f}, dock {100*m['dock_pos_err_m']:.1f} cm / "
          f"{m['dock_yaw_err_deg']:.1f} deg, success {m['dock_success']}")
print(f"saved {out}")