
import time
import numpy as np

from . import config as C
from .sim import Sim
from .sensors import Encoders, Lidar, ArucoCamera, odometry, wrap
from .ekf import EKF
from .controller import LQRTracker, wheel_speeds
from .path import RefPath
from . import groundtruth   # EVALUATION ONLY


def run_trial(seed=0, start_noise=True, push=None, wheel_radius_err=0.0,
              meas_model=C.MEAS_MODEL, video=None, viewer=False, settle_time=5.0, verbose=True):
    """
    seed             random seed (start pose, sensor noise)
    start_noise      randomise the TRUE start pose around the nominal one (P0_STD)
    push             (time_s, force_x, force_y, duration_s) external shove, or None
    wheel_radius_err real right wheel radius error, e.g. 0.02 = 2% (odometry is not told)
    video            path to an .mp4 to record, or None
    """
    rng = np.random.default_rng(seed)
    true_start = C.START_POSE + (rng.normal(0, C.P0_STD) if start_noise else 0.0)
    sim = Sim(true_start, wheel_radius_err)
    enc, lidar, cam = Encoders(sim), Lidar(sim, rng), ArucoCamera(sim, rng)
    ekf = EKF(C.START_POSE, np.diag(C.P0_STD ** 2), meas_model)   # filter only knows the nominal start
    ctrl = LQRTracker()
    path = RefPath(C.WAYPOINTS, C.V_REF, C.A_REF)

    rec = None
    if video:
        from .video import Recorder
        rec = Recorder(video, cam, path)
    view = None
    if viewer:
        import mujoco.viewer
        view = mujoco.viewer.launch_passive(sim.m, sim.d)
        print("MuJoCo viewer opened: the robot drives in real time (~30 s). Close the window to finish.")

    n_steps = int((path.T + settle_time) / C.DT)
    ticks_prev = enc.read()
    L = {k: [] for k in ["t", "x_est", "P", "x_true", "ref", "u", "e_ctrl",
                         "meas", "accepted", "nis", "safety", "front"]}

    t_wall0 = time.perf_counter()
    for k in range(n_steps):
        t = k * C.DT

        # ---- 1. Proprioception: encoders -> odometry -> EKF predict
        ticks = enc.read()
        ds, dth = odometry(ticks - ticks_prev)
        ticks_prev = ticks
        ekf.predict(ds, dth)

        # ---- 2. Exteroception: ArUco camera -> EKF update (10 Hz)
        z, accepted, nis = None, False, np.nan
        if k % C.CAM_EVERY == 0:
            z = cam.measure()
            if z is not None:
                accepted, nis = ekf.update(z)

        # ---- 3. LiDAR safety layer
        scan = lidar.scan()
        front = lidar.front_min(scan)

        # ---- 4. Control on the ESTIMATED state
        ref = path.at(t)
        v, w, e = ctrl(ekf.x, ref)
        safety = front < C.SAFETY_DIST and v > 0
        if safety:
            v = 0.0
        sim.step(*wheel_speeds(v, w))

        # ---- EVALUATION ONLY: ground truth for logging, never fed back
        x_true = groundtruth.true_pose(sim)

        for key, val in zip(L.keys(), [t, ekf.x.copy(), ekf.P.copy(), x_true, ref, [v, w], e,
                                       z if z is not None else [np.nan] * 3, accepted, nis,
                                       safety, front]):
            L[key].append(val)

        if push is not None and abs(t - push[0]) < C.DT / 2:
            sim.apply_push(push[1:3], push[3])
        if rec is not None and k % 2 == 0:
            rec.frame(sim, L)
        if view is not None:
            if not view.is_running():
                break
            view.sync()
            # run at real-time speed so you can watch it
            lag = (k + 1) * C.DT - (time.perf_counter() - t_wall0)
            if lag > 0:
                time.sleep(lag)

    if view is not None:
        print("Run finished. Close the viewer window to see the results.")
        while view.is_running():
            view.sync()
            time.sleep(0.05)
        view.close()

    if rec is not None:
        rec.close()
    cam.renderer.close()
    log = {k: np.array(v) for k, v in L.items()}
    log["T_path"] = path.T
    log["path_xy"] = np.c_[path.x, path.y]
    m = metrics(log)
    if verbose:
        print_metrics(m)
    return log, m


def metrics(log):
    xe, xt, P = log["x_est"], log["x_true"], log["P"]
    err = xt - xe
    err[:, 2] = wrap(err[:, 2])
    nees = np.einsum("ni,nij,nj->n", err, np.linalg.inv(P), err)
    on_path = log["t"] <= log["T_path"]
    trk = np.hypot(*(log["ref"][on_path, :2] - xt[on_path, :2]).T)
    final = xt[-1]
    dock_pos = float(np.hypot(*(final[:2] - C.DOCK_TARGET[:2])))
    dock_yaw = float(abs(wrap(final[2] - C.DOCK_TARGET[2])))
    nis = log["nis"][np.isfinite(log["nis"])]
    acc = log["accepted"]
    return {
        "est_pos_rmse_m": float(np.sqrt(np.mean(np.sum(err[:, :2] ** 2, axis=1)))),
        "est_yaw_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(err[:, 2] ** 2)))),
        "track_pos_rmse_m": float(np.sqrt(np.mean(trk ** 2))),
        "track_pos_max_m": float(trk.max()),
        "dock_pos_err_m": dock_pos,
        "dock_yaw_err_deg": float(np.rad2deg(dock_yaw)),
        "dock_success": bool(dock_pos < C.DOCK_TOL_POS and dock_yaw < C.DOCK_TOL_YAW),
        "mean_nees": float(nees.mean()),           # ~3 if consistent (3 states)
        "n_detections": int(len(nis)),
        "n_rejected": int(len(nis) - acc.sum()),
        "safety_stops": int(log["safety"].sum()),
        "min_front_range_m": float(log["front"].min()),
    }


def print_metrics(m):
    for k, v in m.items():
        print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")