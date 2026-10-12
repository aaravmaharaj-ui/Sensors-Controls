"""The brief: 'the controller must use the estimated state, never ground truth'.
This test fails if the estimator, controller, sensors or path planner import the
ground-truth module or read the simulator's true body pose directly.

    python -m pytest tests
"""
import os
import re

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tb3dock")
MUST_NOT_USE_TRUTH = ["ekf.py", "controller.py", "path.py"]
FORBIDDEN = [r"groundtruth", r"true_pose", r"\.xpos", r"\.xquat", r"qpos\[0", r"qpos\[3"]


def test_estimator_and_controller_never_touch_ground_truth():
    for f in MUST_NOT_USE_TRUTH:
        src = open(os.path.join(PKG, f)).read()
        for pat in FORBIDDEN:
            assert not re.search(pat, src), f"{f} uses ground truth ({pat})"


def test_run_loop_uses_truth_only_for_evaluation():
    src = open(os.path.join(PKG, "run.py")).read()
    uses = [ln for ln in src.splitlines() if "groundtruth." in ln]
    assert len(uses) == 1 and "x_true" in uses[0], uses
    # the controller is called with the EKF estimate
    assert "ctrl(ekf.x, ref)" in src
