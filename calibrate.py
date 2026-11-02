
import json
import os
import numpy as np
import mujoco

from tb3dock import config as C
from tb3dock.sim import Sim
from tb3dock.sensors import Encoders, ArucoCamera, odometry, wrap
from tb3dock.controller import wheel_speeds
from tb3dock.groundtruth import true_pose
from tb3dock.ekf import EKF

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def drive(sim, enc, v, w, seconds):
    t0, p0 = enc.read(), true_pose(sim)
    for _ in range(int(seconds / C.DT)):
        sim.step(*wheel_speeds(v, w))
    return enc.read() - t0, p0, true_pose(sim)


def calibrate_odometry():
    """Effective wheel radius from straight drives, effective track width from ARCS.

    The track is calibrated on arcs (v = 0.15 m/s, |w| = 0.3 and 0.6 rad/s) rather than
    spinning on the spot, because the wheels slip differently in the two cases and the
    robot drives arcs when it navigates. A spin-in-place calibration left odometry
    over-counting turns by ~5% on arcs, which drifted the heading ~25 deg on long
    LiDAR-planned routes before the marker came into view.
    """
    r_ratio, b_ratio = [], []
    # straight lanes along the north and south walls, clear of every obstacle layout
    for start in [(-0.45, 1.3, 0.0), (-0.45, -1.3, 0.0)]:
        sim = Sim(start); enc = Encoders(sim)
        dt, p0, p1 = drive(sim, enc, 0.15, 0.0, 6.0)
        ds, _ = odometry(dt)
        r_ratio.append(np.hypot(*(p1[:2] - p0[:2])) / ds)
    # left and right arcs in the clear corners
    for start, v, w, secs in [((-0.3, -1.1, 0.0), 0.15, 0.3, 6.0), ((-0.3, 1.1, 0.0), 0.15, -0.3, 6.0),
                              ((-0.3, -1.2, 0.0), 0.15, 0.6, 4.0), ((-0.3, 1.2, 0.0), 0.15, -0.6, 4.0)]:
        sim = Sim(start); enc = Encoders(sim)
        th, ticks, prev = [true_pose(sim)[2]], enc.read(), None
        dth_odom = 0.0
        for _ in range(int(secs / C.DT)):
            sim.step(*wheel_speeds(v, w))
            t2 = enc.read()
            dth_odom += odometry(t2 - ticks)[1]
            ticks = t2
            th.append(true_pose(sim)[2])
        true_dth = np.unwrap(th)[-1] - th[0]
        b_ratio.append(dth_odom / true_dth)
    r_eff = C.ODOM_R_EFF * np.mean(r_ratio)
    # dth_odom = r*(dr-dl)/b: correct the radius first, then the track
    b_eff = C.ODOM_TRACK_EFF * np.mean(b_ratio) * np.mean(r_ratio)
    return float(r_eff), float(b_eff)


def odometry_noise(seed=0):
    rng = np.random.default_rng(seed)
    sim = Sim((0.0, 0.0, 0.0)); enc = Encoders(sim)
    rows = []
    ticks, p = enc.read(), true_pose(sim)
    for seg in range(40):
        v, w = rng.uniform(0.0, C.V_MAX), rng.uniform(-1.2, 1.2)
        for _ in range(20):
            sim.step(*wheel_speeds(v, w))
            t2, p2 = enc.read(), true_pose(sim)
            ds, dth = odometry(t2 - ticks)
            # true increment expressed in the robot frame at the start of the step
            dx, dy = p2[:2] - p[:2]
            c, s = np.cos(p[2]), np.sin(p[2])
            ds_true = c * dx + s * dy
            rows.append([ds, dth, ds - ds_true, wrap(dth - wrap(p2[2] - p[2]))])
            ticks, p = t2, p2
        # stay inside the room: drive back towards the centre occasionally
        if np.hypot(*p[:2]) > 1.0:
            sim = Sim((0.0, 0.0, rng.uniform(-np.pi, np.pi))); enc = Encoders(sim)
            ticks, p = enc.read(), true_pose(sim)
    a = np.array(rows)
    # least-squares fit of |error| ~ alpha*|input| (scale by sqrt(pi/2) for mean-abs -> std)
    k = np.sqrt(np.pi / 2)
    a1 = k * np.sum(np.abs(a[:, 0]) * np.abs(a[:, 2])) / np.sum(a[:, 0] ** 2)
    X = np.c_[np.abs(a[:, 1]), np.abs(a[:, 0])]
    a2, a3 = k * np.linalg.lstsq(X, np.abs(a[:, 3]), rcond=None)[0]
    return [float(a1), float(max(a2, 0.0)), float(max(a3, 0.0))]


def camera_noise(n=400, seed=0):
    rng = np.random.default_rng(seed)
    sim = Sim((2.5, 0.0, 0.0)); cam = ArucoCamera(sim, rng)
    data = []
    for _ in range(n):
        r = rng.uniform(0.25, 2.2)
        phi = rng.uniform(-0.6, 0.6)                 # angle around the marker
        x = C.MARKER_POS[0] - r * np.cos(phi)
        y = C.MARKER_POS[1] + r * np.sin(phi)
        th = np.arctan2(C.MARKER_POS[1] - y, C.MARKER_POS[0] - x) + rng.uniform(-0.3, 0.3)
        sim.d.qpos[0:2] = [x, y]
        sim.d.qpos[3:7] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
        mujoco.mj_forward(sim.m, sim.d)
        z = cam.measure()
        zt, _ = EKF.h(true_pose(sim))
        if z is None:
            data.append([zt[0], np.nan, np.nan, np.nan]); continue
        e = z - zt
        e[1:] = wrap(e[1:])
        data.append([zt[0], *e])
    cam.renderer.close()
    d = np.array(data)
    det = np.isfinite(d[:, 1])
    std0, per_m, table = [], [], []
    bins = np.linspace(0.25, 2.2, 6)
    for j in range(1, 4):
        mids, sig = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = det & (d[:, 0] >= lo) & (d[:, 0] < hi)
            if sel.sum() > 5:
                mids.append((lo + hi) / 2)
                sig.append(1.4826 * np.median(np.abs(d[sel, j])))   # robust std (ignores flips)
        b, a = np.polyfit(mids, sig, 1)
        std0.append(max(a, 1e-3)); per_m.append(max(b, 0.0))
        table.append(sig)
    flips = np.mean(np.abs(d[det, 3]) > 0.5)
    rate = [float(np.mean(det[(d[:, 0] >= lo) & (d[:, 0] < hi)])) for lo, hi in zip(bins[:-1], bins[1:])]
    return std0, per_m, float(flips), rate, bins.tolist()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("1/3 odometry calibration ...")
    r_eff, b_eff = calibrate_odometry()
    print(f"    ODOM_R_EFF = {r_eff:.5f}   (datasheet {C.WHEEL_R})")
    print(f"    ODOM_TRACK_EFF = {b_eff:.5f}   (datasheet {C.TRACK})")
    C.ODOM_R_EFF, C.ODOM_TRACK_EFF = r_eff, b_eff   # use them for the noise fit
    print("2/3 odometry noise ...")
    alpha = odometry_noise()
    print(f"    ODOM_ALPHA = ({alpha[0]:.4f}, {alpha[1]:.4f}, {alpha[2]:.4f})")
    print("3/3 camera noise (400 random views) ...")
    std0, per_m, flips, rate, bins = camera_noise()
    print(f"    R_STD0      = {np.round(std0, 4).tolist()}   [range m, bearing rad, yaw rad]")
    print(f"    R_STD_PER_M = {np.round(per_m, 4).tolist()}")
    print(f"    marker-yaw flips (|err|>0.5 rad): {100 * flips:.1f}% of detections")
    print("    detection rate by range:", {f"{lo:.2f}-{hi:.2f} m": round(r, 2)
                                          for lo, hi, r in zip(bins[:-1], bins[1:], rate)})
    out = dict(ODOM_R_EFF=r_eff, ODOM_TRACK_EFF=b_eff, ODOM_ALPHA=alpha, R_STD0=std0,
               R_STD_PER_M=per_m, yaw_flip_rate=flips, detection_rate=rate, range_bins=bins)
    with open(os.path.join(OUT, "calibration.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/calibration.json -> copy the values into tb3dock/config.py")