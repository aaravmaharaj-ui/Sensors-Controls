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

The repository is organised into the following directories and scripts:

| Path | Purpose |
| --- | --- |
| `assets/` | Supporting assets used by the simulation. |
| `models/` | Robot, environment or sensor model files. |
| `results/` | Generated outputs, plots and experimental results. |
| `tb3dock/` | Main project package for the TurtleBot3 docking and navigation system. |
| `tests/` | Tests used to check project functionality. |
| `analysis.py` | Analysis of simulation results and controller or estimator performance. |
| `calibrate.py` | Sensor calibration utilities. |
| `run_demo.py` | Entry point for running the demonstration. |
| `run_trials.py` | Script for running repeated trials and evaluating performance. |
| `requirements.txt` | Python dependencies required by the project. |

The exact responsibilities of each module may be updated as the implementation develops.

## System architecture

The intended control pipeline is:

1. **Sensor measurements:** acquire wheel-encoder odometry, LiDAR scans and camera observations.
2. **State prediction:** propagate the robot's estimated pose using the motion model and odometry.
3. **State correction:** update the EKF estimate using available camera observations and their associated measurement model.
4. **Trajectory tracking:** calculate control commands from the difference between the desired trajectory and the estimated robot state.
5. **Robot motion:** apply the control commands to the simulated TurtleBot3.
6. **Performance evaluation:** compare estimated and ground-truth states, analyse tracking errors and assess docking accuracy.

When camera observations are unavailable, the estimator should continue predicting the robot's state using the motion model and available odometry, with uncertainty increasing as appropriate until measurements become available again.

## Experimental evaluation

The simulation will be evaluated using quantitative measures where applicable, including:

- **Position estimation error:** difference between the estimated and ground-truth robot position.
- **Heading estimation error:** difference between the estimated and ground-truth orientation.
- **Trajectory tracking error:** lateral and heading deviation from the desired path.
- **Docking accuracy:** final position and orientation relative to the charging station.
- **Robustness:** changes in performance under sensor noise, missing observations and wheel-induced heading disturbances.
- **Repeatability:** consistency of results across multiple simulation trials.

Ground-truth simulator measurements are reserved for evaluation and are not fed into the controller.

## Setup

Clone the repository and navigate to its root directory.

Create and activate a Python virtual environment:

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Use a Python version compatible with the dependencies specified in `requirements.txt`. Additional simulator or model setup may be required depending on the installed MuJoCo and project packages.

## Running the project

Run the main demonstration from the repository root:

```bash
python run_demo.py
```

Run the experimental trials:

```bash
python run_trials.py
```

Analyse the generated results:

```bash
python analysis.py
```

The scripts may require additional arguments or configuration depending on the current implementation. Refer to each script and update these instructions as the project develops.

## Testing

Run the available tests from the repository root:

```bash
python -m pytest
```

Tests should be used to verify individual components and the integrated pipeline, including sensor processing, state estimation, controller behaviour and docking performance where corresponding tests are implemented.

## Team

| Member | Responsibilities |
| --- | --- |
| To be confirmed | Sensing and sensor calibration |
| To be confirmed | State estimation and EKF |
| To be confirmed | State-space modelling and trajectory tracking |
| To be confirmed | Simulation integration, testing and evaluation |

Responsibilities will be updated to reflect each team member's actual contributions.

## Project status

This repository contains the initial project structure for the Sensors and Controls assignment. The README will be updated as the navigation pipeline is implemented, tested and evaluated.

The final submission will document the system architecture, implementation, experimental results and limitations, with evidence demonstrating that the controller operates using the estimated state.
