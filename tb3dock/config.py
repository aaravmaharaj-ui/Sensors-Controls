
import numpy as np


DT = 0.05            # estimator + controller period [s] (20 Hz)
CAM_EVERY = 2        # camera processed every 2nd step -> 10 Hz
PHYS_DT = 0.002      # MuJoCo timestep (must match models/room.xml)


WHEEL_R = 0.033      # wheel radius [m]
TRACK = 0.160        # distance between wheels [m]
TICKS_PER_REV = 4096 # Dynamixel XL430 encoder resolution

ODOM_R_EFF = 0.03076
ODOM_TRACK_EFF = 0.16486
V_MAX = 0.22        # max linear speed [m/s] (datasheet)
W_MAX = 2.0          # max angular speed used [rad/s] (datasheet max 2.84)


CAM_X = 0.06                       # camera position ahead of wheel axle, robot frame [m]
CAM_W, CAM_H, CAM_FOVY = 640, 480, 49.0
CORNER_PIX_SIGMA = 0.5             # added corner noise [px] to mimic a real sensor

# ArUco dock marker (world frame)
MARKER_ID = 0
MARKER_SIZE = 0.10                 # black square side [m]
MARKER_POS = np.array([3.3055, 0.0])
MARKER_YAW = np.pi                 # marker outward normal points along -x

# Docked pose: wheel axle 0.22 m in front of the marker, facing it
DOCK_TARGET = np.array([MARKER_POS[0] - 0.22, 0.0, 0.0])
DOCK_TOL_POS = 0.03                # success if within 3 cm ...
DOCK_TOL_YAW = np.deg2rad(5.0)     # ... and 5 degrees

# LiDAR 
LIDAR_N = 180
LIDAR_MAX = 3.5
LIDAR_SIGMA = 0.01
LIDAR_HEIGHT = 0.09                # above the axle
SAFETY_DIST = 0.17                 # stop if anything this close in the front cone [m]
SAFETY_CONE = np.deg2rad(25.0)

#EKF 

MEAS_MODEL = "pose"

ODOM_ALPHA = (0.12, 0.24, 0.0)     # fitted by calibrate.py
ODOM_FLOOR = (1e-4, 2e-4)

R_STD0 = np.array([0.003, 0.003, 0.01])        # calibrate.py: [0.001, 0.0014, 0.0062], floored
R_STD_PER_M = np.array([0.032, 0.002, 0.065])  # calibrate.py: [0.032, 0.0, 0.065]
GATE_CHI2 = {2: 9.21, 3: 11.34}    # 99% chi-square gate for outlier rejection
YAW_MAX_RANGE = 1.2                # use marker yaw only closer than this [m]
RECOVER_AFTER = 3                  # consecutive rejections before assuming the filter is lost
RECOVER_STD = np.array([0.3, 0.3, np.deg2rad(30.0)])   # covariance inflation when lost
P0_STD = np.array([0.02, 0.02, np.deg2rad(2.0)])   # initial pose uncertainty

# LQR trajectory tracker
LQR_Q = np.diag([10.0, 20.0, 2.0])   # weights on [e_along, e_lateral, e_heading]
LQR_R = np.diag([1.0, 0.5])          # weights on [v, w] corrections
V_FLOOR = 0.03                        # linearise at >= this speed (model uncontrollable at v=0)

#Reference path
V_REF = 0.15
A_REF = 0.15
START_POSE = np.array([0.0, 0.0, 0.0])
WAYPOINTS = np.array([
    # path keeps >= 0.27 m from the obstacle edges, so a few cm of estimate error
    # cannot drive the robot into a corner before it first sees the marker
    [0.00, 0.00],
    [0.35, 0.08],
    [0.70, 0.50],
    [1.00, 0.60],
    [1.40, 0.55],
    [1.80, 0.25],
    [2.20, 0.03],
    [2.60, 0.00],
    [2.85, 0.00],
    [DOCK_TARGET[0], 0.00],
])