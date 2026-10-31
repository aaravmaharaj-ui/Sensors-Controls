
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))

log = np.load(os.path.join(ROOT, "results", "nominal", "log.npz"), allow_pickle=True)
t, xe, xt, P = log["t"], log["x_est"], log["x_true"], log["P"]
seen = np.isfinite(log["meas"][:, 0]) & log["accepted"]
# camera runs every 2nd step: fill the gaps so the shading is continuous
shade = seen | np.r_[seen[1:], False] | np.r_[False, seen[:-1]]

err = 100 * np.hypot(*(xt[:, :2] - xe[:, :2]).T)                       # cm
sig = 100 * 3 * np.sqrt(np.maximum(np.linalg.eigvalsh(P[:, :2, :2])[:, 1], 0))  # 3-sigma, worst direction

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.fill_between(t, 0, sig, zorder=2, color="#1f5fbf", alpha=.15, label="filter's uncertainty (3σ)")
ax.plot(t, err, color="#1f5fbf", lw=2.2, zorder=3, label="position error (true − estimate)")
ymax = ax.get_ylim()[1]
ax.fill_between(t, 0, ymax, where=shade, color="#e08a00", alpha=.25, lw=0,
                label="marker seen")
ax.set_ylim(0, ymax)
ax.set_xlim(t[0], t[-1])
ax.set_xlabel("time [s]", fontsize=12); ax.set_ylabel("error [cm]", fontsize=12)
first = t[seen][0] if seen.any() else None
before = err[t < first].max() if first else err.max()
after = np.median(err[t > first + 1]) if first else np.nan
ax.set_title(f"EKF position error: up to {before:.0f} cm on encoders alone, "
             f"~{after:.1f} cm once the marker is seen", fontsize=13)
ax.legend(loc="upper left", fontsize=11)
fig.tight_layout()
out = os.path.join(ROOT, "results", "ekf_error.png")
fig.savefig(out, dpi=150)
print(f"max error before marker {before:.1f} cm, median after {after:.1f} cm\nsaved {out}")