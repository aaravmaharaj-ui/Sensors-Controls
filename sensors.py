"""Sensor models: wheel encoders, 2D LiDAR and the ArUco camera.

Each sensor produces what the real TurtleBot3 would: encoder ticks, a range
scan, and a camera image that is processed with OpenCV. None of them return
the true pose.
"""
import numpy as np
import mujoco
import cv2

from . import config as C


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- encoders ---
class Encoders:
    def __init__(self, sim):
        self.sim = sim
        m = sim.m
        self.adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                    for n in ("wheel_l", "wheel_r")]

    def read(self):
        """Integer tick counts (quantised like the XL430 encoder)."""
        ang = self.sim.d.qpos[self.adr]
        return np.round(ang / (2 * np.pi) * C.TICKS_PER_REV).astype(np.int64)


def odometry(dticks):
    """Tick increments -> (distance travelled ds, heading change dth)."""
    dphi = dticks * 2 * np.pi / C.TICKS_PER_REV
    dl, dr = C.ODOM_R_EFF * dphi
    return (dl + dr) / 2.0, (dr - dl) / C.ODOM_TRACK_EFF


# ------------------------------------------------------------------- LiDAR ---
class Lidar:
    def __init__(self, sim, rng):
        self.sim, self.rng = sim, rng
        self.angles = np.linspace(-np.pi, np.pi, C.LIDAR_N, endpoint=False)
        self.group = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)  # ignore visual-only geoms
        self._gid = np.zeros(1, dtype=np.int32)

    def scan(self):
        m, d, b = self.sim.m, self.sim.d, self.sim.base_id
        R = d.xmat[b].reshape(3, 3)
        origin = d.xpos[b] + R @ np.array([0.0, 0.0, C.LIDAR_HEIGHT])
        ranges = np.empty(C.LIDAR_N)
        for i, a in enumerate(self.angles):
            vec = R @ np.array([np.cos(a), np.sin(a), 0.0])
            dist = mujoco.mj_ray(m, d, origin, vec, self.group, 1, b, self._gid)
            ranges[i] = dist if 0 <= dist < C.LIDAR_MAX else C.LIDAR_MAX
        ranges += self.rng.normal(0, C.LIDAR_SIGMA, C.LIDAR_N)
        return np.clip(ranges, 0.0, C.LIDAR_MAX)

    def front_min(self, ranges):
        return ranges[np.abs(self.angles) < C.SAFETY_CONE].min()


# ---------------------------------------------------------- ArUco camera ---
def camera_matrix():
    """Pinhole intrinsics. In sim these follow from the field of view; on the
    real robot replace them with a checkerboard calibration (cv2.calibrateCamera)."""
    f = (C.CAM_H / 2) / np.tan(np.deg2rad(C.CAM_FOVY) / 2)
    return np.array([[f, 0, C.CAM_W / 2], [0, f, C.CAM_H / 2], [0, 0, 1.0]])


class ArucoCamera:
    def __init__(self, sim, rng):
        self.sim, self.rng = sim, rng
        self.renderer = mujoco.Renderer(sim.m, C.CAM_H, C.CAM_W)
        self.K = camera_matrix()
        self.dist = np.zeros(5)
        dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
        s = C.MARKER_SIZE / 2
        # Corner order returned by ArUco: TL, TR, BR, BL (marker frame, z out of the marker)
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
        self.last_image = None

    def measure(self):
        """Render, detect the marker and return z = [range, bearing, marker_yaw]
        in the robot frame (camera at CAM_X ahead of the axle), or None."""
        self.renderer.update_scene(self.sim.d, camera="cam")
        rgb = self.renderer.render()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        self.last_image = rgb.copy()
        if ids is None or C.MARKER_ID not in ids.flatten():
            return None
        i = list(ids.flatten()).index(C.MARKER_ID)
        img_pts = corners[i].reshape(4, 2).astype(np.float64)
        cv2.aruco.drawDetectedMarkers(self.last_image, [corners[i]], ids[i:i + 1])
        img_pts += self.rng.normal(0, C.CORNER_PIX_SIGMA, img_pts.shape)
        ok, rvec, tvec = cv2.solvePnP(self.obj, img_pts, self.K, self.dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        Rm, _ = cv2.Rodrigues(rvec)
        tx, _, tz = tvec.flatten()          # OpenCV camera frame: x right, y down, z forward
        n = Rm[:, 2]                         # marker normal expressed in the camera frame
        rng = np.hypot(tz, tx)               # horizontal range camera -> marker
        bearing = np.arctan2(-tx, tz)        # robot frame: forward = +z_cam, left = -x_cam
        yaw = np.arctan2(-n[0], n[2])        # direction of marker normal in robot frame
        return np.array([rng, bearing, yaw])
