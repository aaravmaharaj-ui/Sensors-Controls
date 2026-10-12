"""Run one end-to-end trial and save a summary figure, metrics and (optionally) a video.

Examples
    python run_demo.py                          # nominal run -> results/nominal/
    python run_demo.py --viewer                 # watch it live in the MuJoCo viewer
    python run_demo.py --video                  # also record results/<name>/demo.mp4
    python run_demo.py --name pushed --push 9 0 3 0.3       # shove sideways at t = 9 s
    python run_demo.py --name bad_wheel --wheel-err 0.03    # right wheel 3% larger than odometry thinks
    python run_demo.py --meas range_bearing     # EKF without the marker-yaw measurement
"""
import argparse
import json
import os
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
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    a = ap.parse_args()

    out = os.path.join(ROOT, "results", a.name)
    os.makedirs(out, exist_ok=True)
    log, m = run_trial(seed=a.seed, start_noise=not a.no_start_noise, push=a.push,
                       wheel_radius_err=a.wheel_err, meas_model=a.meas,
                       video=os.path.join(out, "demo.mp4") if a.video else None,
                       viewer=a.viewer)
    summary_figure(log, m, os.path.join(out, "summary.png"))
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"args": vars(a), **m}, f, indent=2)
    np.savez_compressed(os.path.join(out, "log.npz"), **log)
    print(f"saved {out}/summary.png, metrics.json, log.npz" + (", demo.mp4" if a.video else ""))
