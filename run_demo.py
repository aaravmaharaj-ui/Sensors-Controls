
import argparse
import json
import os
import time
import numpy as np

from tb3dock.run import run_trial
from tb3dock.plots import summary_figure

ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="nominal")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-start-noise", action="store_true")
    ap.add_argument("--push", type=float, nargs=4, metavar=("T", "FX", "FY", "DUR"))
    ap.add_argument("--wheel-err", type=float, default=0.0)
    ap.add_argument("--meas", choices=["pose", "range_bearing"], default="pose")
    ap.add_argument("--random-obstacles", action="store_true",
                    help="move the obstacles to random positions (the robot is not told)")
    ap.add_argument("--nav", choices=["lidar", "fixed"], default="lidar",
                    help="lidar: plan the route from the LiDAR map; fixed: hand-placed waypoints")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="viewer playback speed: 1 = real time (default), 0.5 = half speed")
    ap.add_argument("--no-wait", action="store_true",
                    help="viewer: start driving straight away instead of waiting for Enter")
    ap.add_argument("--random", action="store_true",
                    help="randomise start pose, sensor noise, obstacle positions, wheel size error and a kick")
    a = ap.parse_args()

    if a.random:
        a.random_obstacles = True
        a.seed = int(time.time() * 1000) % 100000
        r = np.random.default_rng(a.seed)
        a.wheel_err = float(np.round(r.uniform(-0.005, 0.005), 4))
        a.push = [float(np.round(r.uniform(18, 21), 1)), 0.0,
                  float(np.round(r.choice([-1, 1]) * r.uniform(3, 5), 1)), 0.2]
        if a.name == "nominal":
            a.name = f"random_{a.seed}"
        print(f"Random trial: seed {a.seed} (start pose, sensor noise, obstacle positions), "
              f"right wheel {100 * a.wheel_err:+.2f}% size error, "
              f"kick {a.push[2]:+.1f} N sideways at t = {a.push[0]} s")

    out = os.path.join(ROOT, "results", a.name)
    os.makedirs(out, exist_ok=True)
    log, m = run_trial(seed=a.seed, start_noise=not a.no_start_noise, push=a.push,
                       wheel_radius_err=a.wheel_err, meas_model=a.meas, nav_mode=a.nav, random_obstacles=a.random_obstacles,
                       video=os.path.join(out, "demo.mp4") if a.video else None,
                       viewer=a.viewer, speed=a.speed, wait_start=not a.no_wait)
    summary_figure(log, m, os.path.join(out, "summary.png"))
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"args": vars(a), **m}, f, indent=2)
    np.savez_compressed(os.path.join(out, "log.npz"), **log)
    print(f"saved {out}/summary.png, metrics.json, log.npz" + (", demo.mp4" if a.video else ""))