# Sensors-Controls

## Overview

**Visual Waypoint Navigation** is a simulated mobile robotics project for 41014/42043 Sensors and Control in Mechatronics Systems (Track A). The project integrates sensing, state estimation and feedback control to navigate a TurtleBot3 Burger through an indoor room containing four obstacles: three boxes and a cylinder.

The robot operates in MuJoCo using a 2D LiDAR sensor, a camera and wheel-encoder odometry. It plans a route around the obstacles and visually docks at a charging station identified by an ArUco marker.

The project develops and integrates the following components:

1. **Sensing:** the LiDAR maps the obstacles and walls, and the camera detects the ArUco marker with OpenCV to measure its distance, direction and angle. The encoders were calibrated and the camera’s noise was measured.
2. **State estimation:** an Extended Kalman Filter (EKF) estimates the robot's pose by fusing wheel-encoder odometry with the marker's range, bearing and angle from the camera.
3. **Path following:** a state-space model of the robot and a Linear Quadratic Regulator (LQR) controller track the planned route.
4. **Visual docking:** the robot approaches and aligns with the charging station using the camera's view of the ArUco marker, docking within 3 cm and 5° tolerance (1.8 cm and 3 degrees in the normal run).
5. **Robustness testing:** performance was tested under sensor noise, randomised obstacle positions, sideways pushes, wheel-size errors and periods when the marker is out of view.

A key requirement is that the controller uses the **estimated robot state rather than ground-truth simulator data**. Ground-truth data is only used for evaluation and comparison, never as an input to the controller. This is checked automatically by `tests/test_no_ground_truth.py`.

## Project Requirements met

The simulation demonstrates an integrated sensing, estimation and control pipeline.

| Component | How we met it |
| --- | --- |
| Environment | MuJoCo room (4.2 × 3.2 m) with three boxes and a cylinder; obstacle positions can be randomised. |
| Robot platform | TurtleBot3 Burger with a 180-beam 2D LiDAR, a 640×480 camera at 10 Hz and 4096-tick wheel encoders. |
| State-space model | Unicycle model of position and heading, plus a tracking-error model, with controllability and observability analysis (`analysis.py`). |
| State estimation | EKF fusing encoder odometry with the ArUco marker's range, bearing and angle (`tb3dock/ekf.py`). |
| Trajectory tracking | Gain-scheduled LQR controller using only the EKF estimate (`tb3dock/controller.py`). |
| Visual docking | Docks at the ArUco-marked station within the 3 cm / 5° tolerance (1.8 cm and 3° in the normal run). |
| Evaluation | Estimate error, NEES, tracking error and docking error, measured over repeated randomised trials (`run_trials.py`). |

## Repository structure

| Path | Purpose |
| --- | --- |
| `assets/` | ArUco marker texture (DICT_4X4_50, id 0) used on the dock. |
| `models/room.xml` | MuJoCo model: the room, obstacles, dock and TurtleBot3 (wheels, camera, LiDAR mount). |
| `results/` | Generated graphs, videos and trial results. |
| `tests/test_no_ground_truth.py` | Checks that the EKF, controller, path and navigation code never read the simulator's true position. |
| **`tb3dock/`** | **Main package** |
| `tb3dock/config.py` | All parameters in one place (robot, sensors, noise values, LQR weights, mapping). |
| `tb3dock/sim.py` | Runs the MuJoCo simulation; randomises obstacles; applies pushes. |
| `tb3dock/sensors.py` | Wheel encoders and odometry, LiDAR, ArUco camera (OpenCV detection + solvePnP). |
| `tb3dock/ekf.py` | Extended Kalman Filter with outlier rejection and recovery. |
| `tb3dock/navigation.py` | LiDAR occupancy map, A* route planning and replanning. |
| `tb3dock/path.py` | Smooth reference path with a speed profile. |
| `tb3dock/controller.py` | LQR trajectory tracker. |
| `tb3dock/run.py` | Main loop: sense → map → estimate → plan → control. |
| `tb3dock/live.py` | Live camera and map panels in the MuJoCo viewer. |
| `tb3dock/groundtruth.py` | True position, used **only** for evaluation. |
| `run_demo.py` | Runs one demonstration (`--viewer`, `--random`, `--video`). |
| `run_trials.py` | Runs repeated trials and summarises the results. |
| `calibrate.py` | Calibrates the encoders and measures odometry and camera noise. |
| `analysis.py` | Controllability and observability analysis. |
| `requirements.txt` | Python dependencies. |

## System architecture

The pipeline runs every 0.05 s (20 Hz):

1. **Sensor measurements:** read the wheel encoders (20 Hz), LiDAR scan (10 Hz) and camera image (10 Hz).
2. **State prediction:** the EKF propagates the robot's pose using the motion model and encoder odometry.
3. **State correction:** when the camera detects the ArUco marker, the EKF corrects the estimate using its range, bearing and angle. Readings that don't fit are rejected.
4. **Mapping and planning:** LiDAR scans are added to an occupancy grid using the estimated pose. A* plans a route around the mapped obstacles and replans if a new obstacle blocks it.
5. **Trajectory tracking:** the LQR controller calculates speed and turn-rate commands from the difference between the planned route and the estimated pose.
6. **Robot motion:** the commands are sent to the wheels of the simulated TurtleBot3. The LiDAR stops the robot if anything comes within 17 cm in front.

When the marker is out of view, the EKF keeps predicting from odometry, and its uncertainty grows until the marker is seen again.

## Experimental evaluation

All errors are measured against the simulator's true pose, which is used for evaluation only.

### Normal run (`python run_demo.py`)

| Measure | Result |
| --- | --- |
| Docking accuracy | **1.8 cm, 3.0°** (tolerance 3 cm / 5°) |
| Position estimation error (RMSE) | 5.2 cm overall; up to 10 cm on odometry alone, under 1 cm once the marker is seen |
| Heading estimation error (RMSE) | 3.1° |
| Trajectory tracking error (true path vs planned route) | 5.4 cm RMSE, 9.7 cm max |
| Marker detections | 57 used, 4 rejected as outliers |
| Mean NEES | 13.9 (expected ≈ 3, so the filter is overconfident; see Limitations) |
| Closest obstacle in front | 19.6 cm, with no safety stops needed |

### Repeated trials (`python run_trials.py`)

| Condition | Docked | Mean dock error | Mean estimate RMSE |
| --- | --- | --- | --- |
| Normal, random start and noise (n = 20) | TBD | TBD | TBD |
| Random pushes (n = 20) | TBD | TBD | TBD |
| 1% wheel-size error (n = 20) | TBD | TBD | TBD |
| Random obstacles, LiDAR planning (n = 8) | TBD | TBD | TBD |
| Random obstacles, fixed route (n = 8) | TBD | TBD | TBD |

## Setup

Requires **Python 3.10 or newer**. Tested on Windows 11 and Linux.

Clone the repository:

```bash
git clone https://github.com/aaravmaharaj-ui/Sensors-Controls.git
cd Sensors-Controls
```

Create and activate a virtual environment:

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the dependencies (MuJoCo, OpenCV with ArUco, NumPy, SciPy, Matplotlib):

```bash
python -m pip install -r requirements.txt
```

No other setup is needed: MuJoCo installs through pip, and the room model and marker texture are included in the repository.

## Running the project

Run all commands from the repository root with the virtual environment active.

```bash
python run_demo.py --viewer
python run_demo.py --random --viewer
python run_demo.py
python run_trials.py --n 20
python run_trials.py --n 20 --push
python run_trials.py --n 20 --wheel-err 0.01
python run_trials.py --n 8 --random-obstacles
python run_trials.py --n 8 --random-obstacles --nav fixed
python calibrate.py
python analysis.py
python -m pytest tests
```

## Testing

```bash
python -m pytest tests
```

`tests/test_no_ground_truth.py` checks that the EKF, controller, path and navigation code never read the simulator's true position, so the controller only ever uses the estimated state.

## Team

| Member | Responsibilities |
| --- | --- |
| To be confirmed | Sensing and sensor calibration |
| To be confirmed | State estimation and EKF |
| To be confirmed | State-space modelling and trajectory tracking |
| To be confirmed | Simulation integration, testing and evaluation |

Responsibilities will be updated to reflect each team member's actual contributions.

## Limitations and improvements

- **Overconfident filter:** the mean NEES is 13.9, where about 3 is expected, so the EKF thinks it is more certain than it is. Increasing the odometry noise (process noise) would make its uncertainty more realistic.
- **Late marker sighting:** the marker is only seen in the last third of the route, so the estimate drifts up to 10 cm on odometry alone. A second marker partway along, or matching LiDAR scans to the map, would correct it earlier.
- **Cautious routes:** the planner keeps about 40 cm from obstacles, so the robot goes around the obstacle group instead of weaving between them. Lowering the clearance would give shorter routes but less safety margin.
- **Close-range dropouts:** marker detection occasionally misses a frame within about 17 cm of the marker, so the last few centimetres rely mostly on odometry.
- **Simulation only:** the system has not been tested on the real TurtleBot3. Real wheel slip, lighting and LiDAR reflections would need retuning.
