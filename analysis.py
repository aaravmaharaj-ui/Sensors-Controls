"""Part A: controllability of the tracking-error model and observability of the
EKF model for each sensor combination. Prints results and writes
results/analysis.txt.

    python analysis.py
"""
import os
import numpy as np

from tb3dock import config as C
from tb3dock.controller import error_model, LQRTracker
from tb3dock.ekf import EKF
from tb3dock.path import RefPath

ROOT = os.path.dirname(os.path.abspath(__file__))
lines = []


def out(s=""):
    print(s); lines.append(s)


def rank(M, tol=1e-6):
    sv = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(sv > tol * sv[0])) if sv[0] > 0 else 0, sv


# ------------------------------------------------------------ controllability
out("=== Controllability of the linearised tracking-error model ===")
for vr, wr in [(0.15, 0.0), (0.15, 0.5), (0.0, 0.5), (0.0, 0.0)]:
    A, B = error_model(vr, wr)
    Ctrb = np.hstack([B, A @ B, A @ A @ B])
    r, _ = rank(Ctrb)
    out(f"  v_r={vr:4.2f}, w_r={wr:3.1f}:  rank = {r}/3  -> {'controllable' if r == 3 else 'NOT controllable'}")
out("  => at standstill the lateral error cannot be corrected (non-holonomic);")
out(f"     this is why the LQR is linearised at v_r >= V_FLOOR = {C.V_FLOOR} m/s.")
K = LQRTracker().gain(C.V_REF, 0.0)
out(f"  LQR gain at v_r={C.V_REF}, w_r=0:\n{np.array2string(K, precision=3)}")
A, B = error_model(C.V_REF, 0.0)
out(f"  closed-loop poles: {np.round(np.linalg.eigvals(A - B @ K), 3)}")

# --------------------------------------------------------------- observability
out("\n=== Local observability of the EKF model along the docking approach ===")
path = RefPath(C.WAYPOINTS, C.V_REF, C.A_REF)
ts = np.arange(path.T - 8.0, path.T, C.DT)          # last 8 s: marker in view


def obs_matrix(rows, moving=True):
    """O = [H_0; H_1 F_0; H_2 F_1 F_0; ...] stacked along the reference trajectory."""
    blocks, Phi = [], np.eye(3)
    for i, t in enumerate(ts):
        x = path.at(t)[:3] if moving else path.at(ts[0])[:3]
        if rows:
            _, H = EKF.h(x)
            blocks.append(H[rows] @ Phi)
        nx = path.at(ts[min(i + 1, len(ts) - 1)])[:3] if moving else x
        ds = np.hypot(*(nx[:2] - x[:2]))
        thm = x[2] + (nx[2] - x[2]) / 2
        F = np.array([[1, 0, -ds * np.sin(thm)], [0, 1, ds * np.cos(thm)], [0, 0, 1]])
        Phi = F @ Phi
    return np.vstack(blocks) if blocks else np.zeros((1, 3))


cases = [("odometry only (no exteroceptive measurement)", [], True),
         ("range + bearing, robot stationary", [0, 1], False),
         ("range + bearing, robot moving", [0, 1], True),
         ("range + bearing + marker yaw (pose), stationary", [0, 1, 2], False)]
for name, rows, moving in cases:
    r, sv = rank(obs_matrix(rows, moving))
    out(f"  {name:48s} rank {r}/3   singular values {np.array2string(sv, precision=3)}")

# Null space of the range-bearing case: rotation about the marker
x = path.at(ts[0])[:3]
_, H = EKF.h(x)
null = np.linalg.svd(H[:2])[2][-1]
rot = np.array([-(x[1] - C.MARKER_POS[1]), x[0] - C.MARKER_POS[0], 1.0])
rot /= np.linalg.norm(rot)
out(f"\n  range+bearing unobservable direction  {np.round(null * np.sign(null[2]), 3)}")
out(f"  rotation of the robot about the marker {np.round(rot, 3)}")
out("  => range and bearing to ONE point landmark cannot tell the robot apart from the same")
out("     pose rotated about that landmark. The symmetry holds for any trajectory, so")
out("     moving does not fix it (rank stays 2). An EKF linearised at its own changing")
out("     estimate can wrongly 'see' this direction as observable, which is a known cause")
out("     of overconfidence. Measuring the marker's orientation too (pose measurement)")
out("     removes the symmetry, which is why MEAS_MODEL = 'pose'.")

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "analysis.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
