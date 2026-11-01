
import time
import numpy as np

from . import config as C
from .sim import Sim
from .sensors import Encoders, Lidar, ArucoCamera, odometry, wrap
from .ekf import EKF
from .controller import LQRTracker, wheel_speeds
from .path import RefPath
from .navigation import Navigator
from . import groundtruth   # EVALUATION ONLY


def _wait_for_enter(view, refresh):

    import threading
    done = threading.Event()

    def _wait():
        try:
            input()
        except EOFError:
            pass
        done.set()

    threading.Thread(target=_wait, daemon=True).start()
    while view.is_running() and not done.is_set():
        refresh()
        view.sync()
        time.sleep(0.1)


def run_trial(seed=0, start_noise=True, push=None, wheel_radius_err=0.0,
              meas_model=C.MEAS_MODEL, video=None, viewer=False, settle_time=5.0, verbose=True,
              nav_mode=C.NAV_MODE, random_obstacles=False, speed=1.0, wait_start=True):

    rng = np.random.default_rng(seed)
    true_start = C.START_POSE + (rng.normal(0, C.P0_STD) if start_noise else 0.0)
    sim = Sim(true_start, wheel_radius_err,
              obstacle_rng=np.random.default_rng(seed + 7919) if random_obstacles else None)
    enc, lidar, cam = Encoders(sim), Lidar(sim, rng), ArucoCamera(sim, rng)
    ekf = EKF(C.START_POSE, np.diag(C.P0_STD ** 2), meas_model)   # filter only knows the nominal start
    ctrl = LQRTracker()
    nav = None
    if nav_mode == "lidar":
        # look around before moving: map from 3 scans, then plan the first route
        nav = Navigator()
        for _ in range(3):
            nav.grid.update(ekf.x, lidar.scan())
        if not nav.plan(ekf.x, 0.0, "initial plan from first scans"):
            raise RuntimeError("no route to the dock found in the first LiDAR map")
        path = nav.path
    else:
        path = RefPath(C.WAYPOINTS, C.V_REF, C.A_REF)

    rec = None
    if video:
        from .video import Recorder
        rec = Recorder(video, cam, path)
        rec.grid = nav.grid if nav else None
        rec.obstacles = sim.obstacles
    view = None
    if viewer:
        import mujoco.viewer
        view = mujoco.viewer.launch_passive(sim.m, sim.d)
        speed_txt = "real time" if speed == 1.0 else f"{speed:g}x speed"
        print(f"MuJoCo viewer opened ({speed_txt}).")
        from .live import LiveOverlay
        overlay = LiveOverlay(view, cam, path)
        overlay.grid = nav.grid if nav else None
        overlay.obstacles = sim.obstacles
        if wait_start:
            # show the start state (first LiDAR map, planned route, camera view) and wait.
            # The camera image is rendered directly, NOT measured, so the random numbers
            # (and therefore the run) are identical to a run without the viewer.
            cam.renderer.update_scene(sim.d, camera="cam")
            cam.last_image = cam.renderer.render().copy()
            L0 = {"t": [0.0], "x_est": [ekf.x.copy()], "P": [ekf.P.copy()],
                  "x_true": [ekf.x.copy()], "accepted": [False]}   # no truth needed before the start
            print("\n>>> Click Enter to start <<<\n")

            _wait_for_enter(view, lambda: overlay.update(L0))
            if not view.is_running():
                view.close(); cam.renderer.close()
                raise SystemExit("viewer closed before the run started")

    n_steps = int(C.NAV_MAX_TIME / C.DT)
    last_plan = 0.0
    ticks_prev = enc.read()
    L = {k: [] for k in ["t", "x_est", "P", "x_true", "ref", "u", "e_ctrl",
                         "meas", "accepted", "nis", "safety", "front"]}

    t_wall0 = time.perf_counter()      # starts after the Enter press
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

        # ---- 3. LiDAR: map the room, replan if the route is blocked, safety stop
        scan = lidar.scan()
        front = lidar.front_min(scan)
        if nav is not None:
            if k % C.NAV_MAP_EVERY == 0:
                nav.grid.update(ekf.x, scan)        # placed with the ESTIMATED pose
            check = k % int(round(C.NAV_CHECK_EVERY / C.DT)) == 0
            approaching_dock = ekf.x[0] > C.NAV_PREDOCK[0] - 0.1
            if (check and not approaching_dock and t - last_plan >= C.NAV_REPLAN_COOLDOWN
                    and nav.blocked_ahead(ekf.x, t)):
                if nav.plan(ekf.x, t, "route blocked by a newly mapped obstacle"):
                    path, last_plan = nav.path, t
                    if verbose:
                        print(f"  t = {t:5.1f} s: replanned (plan {nav.n_plans})")
                    for obj in (rec, overlay if view is not None else None):
                        if obj is not None:
                            obj.ref_xy = np.c_[path.x, path.y]

        # ---- 4. Control on the ESTIMATED state
        ref = nav.ref(t) if nav is not None else path.at(t)
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
        t_end = (nav.t0 if nav is not None else 0.0) + path.T + settle_time
        if t >= t_end:
            break
        if view is not None:
            if not view.is_running():
                break
            if k % 2 == 0:
                overlay.update(L, scan)
            view.sync()
            # run at real-time speed (or slower / faster with speed) so you can watch it
            lag = (k + 1) * C.DT / speed - (time.perf_counter() - t_wall0)
            if lag > 0:
                time.sleep(lag)

    if view is not None:
        # keep the window open until you are done looking (the viewer draws on its own thread)
        print("Run finished. Press Enter in this terminal to close the viewer and see the results.")
        # keep the panels updating (so they follow window or side-panel resizes) until Enter
        _wait_for_enter(view, lambda: overlay.update(L, scan))
        overlay.close()
        view.close()

    if rec is not None:
        rec.close()
    cam.renderer.close()
    log = {k: np.array(v) for k, v in L.items()}
    log["T_path"] = (nav.t0 if nav is not None else 0.0) + path.T
    log["path_xy"] = np.c_[path.x, path.y]
    log["obstacles"] = sim.obstacles
    if nav is not None:
        g = nav.grid
        log["occ"] = g.occupied()
        log["occ_extent"] = np.array([g.x0, g.x0 + g.nx * g.res, g.y0, g.y0 + g.ny * g.res])
    m = metrics(log)
    m["n_plans"] = nav.n_plans if nav is not None else 0
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