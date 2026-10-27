
import numpy as np

# ---- Timing -----------------------------------------------------------------
DT = 0.05            # estimator + controller period [s] (20 Hz)
CAM_EVERY = 2        # camera processed every 2nd step -> 10 Hz
PHYS_DT = 0.002      # MuJoCo timestep (must match models/room.xml)

# ---- Platform (TurtleBot3 Burger datasheet values) ---------------------------
WHEEL_R = 0.033      # wheel radius [m]
TRACK = 0.160        # distance between wheels [m]
TICKS_PER_REV = 4096 # Dynamixel XL430 encoder resolution
# Effective values from calibrate.py (straight drives and driving arcs).
# Wheels compress and skid a little, so the kinematic values differ from the datasheet.
ODOM_R_EFF = 0.03078
ODOM_TRACK_EFF = 0.16788
V_MAX = 0.22        # max linear speed [m/s] (datasheet)
W_MAX = 2.0          # max angular speed used [rad/s] (datasheet max 2.84)

# ---- Camera (Raspberry Pi Camera v2-like) -----------------------------------
CAM_X = 0.06                       # camera position ahead of wheel axle, robot frame [m]
CAM_W, CAM_H, CAM_FOVY = 640, 480, 49.0
CORNER_PIX_SIGMA = 0.5             # added corner noise [px] to mimic a real sensor

# ---- ArUco dock marker (world frame) ----------------------------------------
MARKER_ID = 0
MARKER_SIZE = 0.10                 # black square side [m]
MARKER_POS = np.array([3.3055, 0.0])
MARKER_YAW = np.pi                 # marker outward normal points along -x

# Docked pose: wheel axle 0.22 m in front of the marker, facing it
DOCK_TARGET = np.array([MARKER_POS[0] - 0.22, 0.0, 0.0])
DOCK_TOL_POS = 0.03                # success if within 3 cm ...
DOCK_TOL_YAW = np.deg2rad(5.0)     # ... and 5 degrees

# ---- LiDAR (LDS-01-like) ----------------------------------------------------
LIDAR_N = 180
LIDAR_MAX = 3.5
LIDAR_SIGMA = 0.01
LIDAR_HEIGHT = 0.09                # above the axle
SAFETY_DIST = 0.17                 # stop if anything this close in the front cone [m]
SAFETY_CONE = np.deg2rad(25.0)

# ---- EKF ----------------------------------------------------------------------
# Measurement model: "pose" = range, bearing and marker yaw (fully observable)
#                    "range_bearing" = range and bearing only
MEAS_MODEL = "pose"
# Odometry noise: sigma_s = a1*|ds|, sigma_th = a2*|dth| + a3*|ds|  (+ small floor)
ODOM_ALPHA = (0.12, 0.24, 0.0)     # fitted by calibrate.py
ODOM_FLOOR = (1e-4, 2e-4)
# Measurement noise std devs for [range m, bearing rad, marker yaw rad], growing
# with range: sigma = R_STD0 + R_STD_PER_M * range. Fitted by calibrate.py.
# Marker yaw from a small planar marker gets much worse with distance (PnP ambiguity).
R_STD0 = np.array([0.003, 0.003, 0.01])        # calibrate.py: [0.001, 0.0014, 0.0062], floored
R_STD_PER_M = np.array([0.032, 0.002, 0.065])  # calibrate.py: [0.032, 0.0, 0.065]
GATE_CHI2 = {2: 9.21, 3: 11.34}    # 99% chi-square gate for outlier rejection
YAW_MAX_RANGE = 1.2                # use marker yaw only closer than this [m]
RECOVER_AFTER = 3                  # consecutive rejections before assuming the filter is lost
RECOVER_STD = np.array([0.3, 0.3, np.deg2rad(30.0)])   # covariance inflation when lost
P0_STD = np.array([0.02, 0.02, np.deg2rad(2.0)])   # initial pose uncertainty

# ---- LQR trajectory tracker ------------------------------------------------
LQR_Q = np.diag([10.0, 20.0, 2.0])   # weights on [e_along, e_lateral, e_heading]
LQR_R = np.diag([1.0, 0.5])          # weights on [v, w] corrections
V_FLOOR = 0.03                        # linearise at >= this speed (model uncontrollable at v=0)

# ---- Obstacle randomisation (--random-obstacles): centre x and y ranges [m] ----
# Chosen so a route always exists and the dock approach (x > 2.6 m) stays clear.
OBSTACLE_RANGES = {
    "obs_box":  ((0.70, 1.20), (-0.50, 0.40)),
    "obs_box2": ((1.40, 1.90), (0.30, 1.00)),
    "obs_cyl":  ((1.75, 2.30), (-1.00, -0.35)),
    "obs_box3": ((2.10, 2.40), (0.45, 1.00)),
}

# ---- LiDAR navigation (navigation.py) -----------------------------------------
# "lidar": map the room from LiDAR scans and plan the route with A* (default)
# "fixed": follow the hand-placed WAYPOINTS below (kept for comparison)
NAV_MODE = "lidar"
NAV_RES = 0.05                          # occupancy grid cell size [m]
NAV_BOUNDS = (-0.8, 3.8, -1.8, 1.8)     # grid extent x_min, x_max, y_min, y_max [m]
NAV_L_FREE, NAV_L_OCC = -0.4, 0.9       # log-odds added per beam pass / per hit
NAV_L_MAX, NAV_L_THRESH = 4.0, 0.6      # log-odds clamp; above THRESH = occupied
NAV_INFLATE = 0.20      # cells closer than this to an obstacle are blocked:
                        # robot radius 0.10 m + 0.10 m for estimate error
NAV_COMFORT = 0.40      # cells closer than this cost extra, so routes keep their distance
NAV_COMFORT_W = 4.0     # how much extra (cost of a cell right at NAV_INFLATE = 1 + ~2)
NAV_BLOCKED = 0.17      # replan if the route ahead passes closer than this to an obstacle
NAV_LOOKAHEAD = 1.2     # how far ahead [m] along the route to check for new obstacles
NAV_MAP_EVERY = 2       # add a scan to the map every 2 steps (10 Hz)
NAV_CHECK_EVERY = 0.5   # check the route ahead every 0.5 s
NAV_REPLAN_COOLDOWN = 1.0   # at most one replan per second
NAV_PREDOCK = (2.75, 0.0)   # A* goal, 0.56 m in front of the marker; a straight
                            # approach to DOCK_TARGET is appended
NAV_MAX_TIME = 90.0     # give up after this long [s]

# ---- Reference path ---------------------------------------------------------
V_REF = 0.15
A_REF = 0.15
START_POSE = np.array([0.0, 0.0, 0.0])
WAYPOINTS = np.array([
    # hand-placed route for the default layout (used only with --nav fixed, for comparison).
    # It keeps >= 0.22 m from every obstacle edge in models/room.xml.
    [0.00, 0.00],
    [0.40, 0.20],
    [0.85, 0.35],
    [1.25, 0.25],
    [1.65, 0.10],
    [2.20, 0.12],
    [2.60, 0.02],
    [2.85, 0.00],
    [DOCK_TARGET[0], 0.00],
])