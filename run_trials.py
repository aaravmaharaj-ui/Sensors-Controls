
import argparse
import json
import os
import numpy as np

from tb3dock.run import run_trial

ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--wheel-err", type=float, default=0.0)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--meas", choices=["pose", "range_bearing"], default="pose")
    ap.add_argument("--random-obstacles", action="store_true",
                    help="move the obstacles to random positions (the robot is not told)")
    ap.add_argument("--nav", choices=["lidar", "fixed"], default="lidar",
                    help="lidar: plan the route from the LiDAR map; fixed: hand-placed waypoints")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or (f"trials_{a.nav}_we{a.wheel_err}_push{int(a.push)}_{a.meas}"
                      + ("_randobs" if a.random_obstacles else ""))

    rows = []
    for seed in range(100, 100 + a.n):
        rng = np.random.default_rng(seed)
        push = None
        if a.push:   # random shove of 2-4 N for 0.3 s somewhere along the path
            ang = rng.uniform(0, 2 * np.pi); f = rng.uniform(2, 4)
            push = (rng.uniform(3, 18), f * np.cos(ang), f * np.sin(ang), 0.3)
        _, m = run_trial(seed=seed, push=push, wheel_radius_err=a.wheel_err,
                         meas_model=a.meas, nav_mode=a.nav, random_obstacles=a.random_obstacles, verbose=False)
        m["seed"] = seed
        rows.append(m)
        print(f"seed {seed}: {'OK  ' if m['dock_success'] else 'FAIL'} "
              f"dock {100*m['dock_pos_err_m']:.1f} cm / {m['dock_yaw_err_deg']:.1f} deg, "
              f"est RMSE {100*m['est_pos_rmse_m']:.1f} cm, NEES {m['mean_nees']:.1f}")

    keys = ["dock_pos_err_m", "dock_yaw_err_deg", "est_pos_rmse_m", "track_pos_rmse_m", "mean_nees"]
    summary = {"n": a.n, "success_rate": float(np.mean([r["dock_success"] for r in rows]))}
    for k in keys:
        v = np.array([r[k] for r in rows])
        summary[k] = {"mean": float(v.mean()), "std": float(v.std()), "max": float(v.max())}
    print(json.dumps(summary, indent=2))
    out = os.path.join(ROOT, "results", name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "trials.json"), "w") as f:
        json.dump({"args": vars(a), "summary": summary, "trials": rows}, f, indent=2)
    print(f"saved {out}/trials.json")