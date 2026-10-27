
import os
import numpy as np
import mujoco
import cv2

from . import config as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "room.xml")
MARKER_PNG = os.path.join(ROOT, "assets", "aruco_0.png")


def make_marker_texture(path=MARKER_PNG, px=200, border=40):
 
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = cv2.aruco.generateImageMarker(d, C.MARKER_ID, px)
    img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(path, img)


def yaw_to_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class Sim:
    def __init__(self, start_pose=C.START_POSE, wheel_radius_err=0.0, obstacle_rng=None):
        make_marker_texture()
        self.m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.d = mujoco.MjData(self.m)
        self.n_sub = int(round(C.DT / self.m.opt.timestep))
        self.base_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")

        # Disturbance: make the real right wheel bigger than odometry assumes
        if wheel_radius_err:
            gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "wheel_r")
            self.m.geom_size[gid][0] *= 1.0 + wheel_radius_err

        # Randomise where the obstacles are (the robot is never told)
        if obstacle_rng is not None:
            for name, (xr, yr) in C.OBSTACLE_RANGES.items():
                gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid < 0:          # this room has no obstacle by that name: skip it
                    continue
                self.m.geom_pos[gid][:2] = [obstacle_rng.uniform(*xr), obstacle_rng.uniform(*yr)]
        self.obstacles = self._obstacle_list()

        x, y, yaw = start_pose
        self.d.qpos[0:3] = [x, y, 0.034]
        self.d.qpos[3:7] = yaw_to_quat(yaw)
        mujoco.mj_forward(self.m, self.d)
        # let the robot settle on its wheels
        for _ in range(100):
            mujoco.mj_step(self.m, self.d)
        self.push = None   # (force_xy, t_end)

    def _obstacle_list(self):
        """(kind, x, y, half-size or radius) of each obstacle, for drawing the room."""
        out = []
        for name in C.OBSTACLE_RANGES:
            gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                continue
            kind = "cyl" if self.m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER else "box"
            out.append((kind, *self.m.geom_pos[gid][:2], self.m.geom_size[gid][0]))
        return out

    @property
    def time(self):
        return self.d.time

    def apply_push(self, force_xy, duration):
        """External shove in the world frame (e.g. someone kicks the robot)."""
        self.push = (np.asarray(force_xy, float), self.d.time + duration)

    def step(self, w_left, w_right):
        """Command wheel angular velocities [rad/s] and advance one control period."""
        self.d.ctrl[:] = [w_left, w_right]
        for _ in range(self.n_sub):
            if self.push is not None and self.d.time < self.push[1]:
                self.d.xfrc_applied[self.base_id, 0:2] = self.push[0]
            else:
                self.d.xfrc_applied[self.base_id, :] = 0.0
            mujoco.mj_step(self.m, self.d)