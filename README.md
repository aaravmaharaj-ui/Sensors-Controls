# Sensors-Controls

## Group and contributions

**Group Number:** 49

**Group Members:**
- Aarav Maharaj (25880232)
- Amani Dutt (26075364)
- Anotida Kamungazi (25653320)

**Demo video:** 

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

## A. Modelling

### Motion model
We modelled the TurtleBot3 as a unicycle. The state is its pose in the room, $\mathbf{x} = [x,\ y,\ \theta]^T$, and the inputs are forward speed $v$ and turn rate $\omega$:

$$\dot x = v\cos\theta,\qquad \dot y = v\sin\theta,\qquad \dot\theta = \omega$$

In the EKF we use the discrete version, driven by the change in wheel angle from the encoders each step. We use the heading at the middle of the step, which is more accurate when the robot is turning.

$$\Delta s = \tfrac{r}{2}(\Delta\phi_R + \Delta\phi_L),\qquad \Delta\theta = \tfrac{r}{b}(\Delta\phi_R - \Delta\phi_L)$$

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \begin{bmatrix}\Delta s\cos(\theta_k + \Delta\theta/2)\\ \Delta s\sin(\theta_k + \Delta\theta/2)\\ \Delta\theta\end{bmatrix}$$

$r$ = 30.78 mm and $b$ = 167.9 mm are our calibrated wheel radius and wheel spacing (see B).

### Measurement model
The camera is 6 cm in front of the wheel axle. From each image we get the range $\rho$ and bearing $\beta$ to the marker, and the marker's angle $\psi$ relative to the robot:

$$\rho = \sqrt{(m_x - c_x)^2 + (m_y - c_y)^2},\quad \beta = \mathrm{atan2}(m_y - c_y,\ m_x - c_x) - \theta,\quad \psi = \theta_m - \theta$$

Here $(c_x, c_y) = (x + 0.06\cos\theta,\ y + 0.06\sin\theta)$ is where the camera is, and $(m_x, m_y, \theta_m)$ is the marker's known position and direction.

### Tracking-error model
For the controller we wrote the error between the planned route $(x_r, y_r, \theta_r)$ and the estimated pose in the robot's own frame:

$$\mathbf{e} = \begin{bmatrix} \cos\theta & \sin\theta & 0\\ -\sin\theta & \cos\theta & 0\\ 0&0&1\end{bmatrix}\begin{bmatrix}x_r - x\\ y_r - y\\ \theta_r - \theta\end{bmatrix}$$

Linearising around the route's speed $v_r$ and turn rate $\omega_r$ gives

$$\dot{\mathbf{e}} = A\mathbf{e} + B\mathbf{u},\qquad A = \begin{bmatrix}0 & \omega_r & 0\\ -\omega_r & 0 & v_r\\ 0&0&0\end{bmatrix},\quad B = \begin{bmatrix}1 & 0 \\ 0 & 0 \\ 0 & 1\end{bmatrix}$$

### Controllability
We checked the rank of $[B\ AB\ A^2B]$ in `analysis.py`:

| Route | Rank | Controllable |
| --- | --- | --- |
| Straight, $v_r$ = 0.15 | 3 | yes |
| Turning, $v_r$ = 0.15, $\omega_r$ = 0.5 | 3 | yes |
| Spinning on the spot, $\omega_r$ = 0.5 | 3 | yes |
| Stopped | 2 | no |

When the robot is stopped it can't fix a sideways error, since the wheels can't push it sideways. To avoid this, the controller always linearises at 0.03 m/s or faster.

### Observability
| What the EKF gets | Rank | Observable |
| --- | --- | --- |
| Odometry only | 0 | no |
| Marker range and bearing | 2 | no |
| Marker range, bearing and angle | 3 | yes |

With just range and bearing to one marker, the robot could be anywhere on a circle around the marker and the readings would look the same. `analysis.py` confirms the missing direction is exactly that rotation. Adding the marker's angle fixes it, so we use all three.

---

## B. Sensing

<img width="2210" height="676" alt="sensor_proof (1)" src="https://github.com/user-attachments/assets/6bde4998-e697-450f-8c9a-d437c7a2e5b0" />

### Calibration
`calibrate.py` compares the sensors against the simulator's true motion, like checking a real robot with a tape measure.

The encoders were the main one. We drove the robot straight to get the effective wheel radius, and along arcs to get the wheel spacing. We got 30.78 mm and 167.9 mm, compared with 33 mm and 160 mm on the datasheet, because the wheels squash and slip a little. We first calibrated the spacing by spinning on the spot, but that over-counted turns by about 5% on arcs, so we switched to arcs since that's how the robot actually drives. On a short S-shaped drive, the drift went from 5.6 cm with the datasheet values to 2.0 cm with ours (left graph).

We also fitted the odometry noise ($\sigma_s = 0.12|\Delta s|$, $\sigma_\theta = 0.24|\Delta\theta|$) and measured the camera noise from 300 random views of the marker. The camera's range error is about 0.6 cm under 1 m but about 5.7 cm past 1.8 m (middle graph), and it detects the marker less often further away (100% under 1 m, about 46% past 1.8 m).

### Noise handling
- Camera noise in the EKF grows with distance, so close readings count more.
- A 99% chi-square gate throws out readings that don't match the prediction. This mostly catches the ArUco angle occasionally flipping.
- We only use the marker's angle within 1.2 m, since it's unreliable further out.
- LiDAR beams that hit nothing return exactly max range with no noise added, so they can't show up as fake obstacles in the map.

### Frames
| Frame | Where it is | How we convert it |
| --- | --- | --- |
| Room | Origin at the robot's start, x towards the dock | The EKF state and the map are in this frame |
| Robot | Middle of the wheel axle, x forward, y left | To the room frame using the estimated pose |
| Camera | 6 cm forward and 10 cm up from the axle; OpenCV axes (x right, y down, z forward) | Robot forward = camera z, robot left = −camera x |
| Marker | Centre of the marker | `solvePnP` gives it in the camera frame; we turn that into range, bearing and angle |
| LiDAR | 9 cm above the axle, same axes as the robot | Each hit goes to the room frame using the estimated pose, then into the map |

Because the map uses the estimated pose, estimate errors show up as small shifts in the map. Compared with the real obstacles, the median error of the mapped cells is 2.5 cm (right graph).

<img width="2100" height="868" alt="frames" src="https://github.com/user-attachments/assets/6283a9ce-9f02-4250-83af-867e3863e5e4" />


---

## C. Estimation

<img width="1500" height="675" alt="ekf_error (2)" src="https://github.com/user-attachments/assets/745fa3b8-2191-404b-b112-e895a59d987a" />

We used an EKF to estimate $[x, y, \theta]$ at 20 Hz.

The prediction step uses the encoder odometry and the motion model from A. The odometry noise is mapped into the state with the input Jacobian, $P \leftarrow FPF^T + VMV^T$. When the camera sees the marker (up to 10 Hz), the update uses range, bearing and angle. We use the Joseph form for the covariance update so $P$ stays symmetric.

Two extra things made it a lot more reliable:
- Outlier rejection with the 99% chi-square gate described in B.
- Recovery after a push. If three close-up readings in a row get rejected, it's more likely the filter is wrong than the camera. We then inflate $P$ so the next readings get accepted and the estimate jumps back.

We didn't hand-tune the noise values. They come straight from the calibration in B. The starting uncertainty is 2 cm and 2°.

Results from the normal run:

| | |
| --- | --- |
| Position error on odometry alone | up to 10 cm |
| Position error once the marker is seen | under 1 cm (median 0.9 cm) |
| Position RMSE, whole run | 5.2 cm |
| Heading RMSE | 3.1° |
| Marker readings | 57 used, 4 rejected |
| Mean NEES | 13.9 (should be about 3) |

In the graph, the uncertainty (light blue) keeps growing while the robot only has odometry, then drops almost to zero once the marker is seen (orange). The NEES is too high, though, which means the filter thinks it's more accurate than it is. We think this is wheel slip on turns that our odometry noise model doesn't fully capture. Raising the process noise should fix it, but we ran out of time to retune it.

---

## D. Control

<img width="2100" height="700" alt="control" src="https://github.com/user-attachments/assets/6b55dae0-0b7d-4bc9-81a1-53166d02ce89" />


We used an LQR on the tracking-error model from A. It minimises

$$J = \int (\mathbf{e}^T Q \mathbf{e} + \mathbf{u}^T R \mathbf{u}) dt,\qquad \mathbf{u} = -K\mathbf{e}$$

The route's speed and turn rate change along the path, so we recompute $K$ every step by solving the Riccati equation (gain scheduling). The commands sent to the robot are

$$v = v_r\cos e_\theta + (K\mathbf{e})_1,\qquad \omega = \omega_r + (K\mathbf{e})_2$$

capped at 0.22 m/s and 2 rad/s. The controller only ever gets the EKF estimate, not the true pose.

### Tuning
We used $Q = \mathrm{diag}(10, 20, 2)$ on [along-track, sideways, heading] error and $R = \mathrm{diag}(1, 0.5)$ on [speed, turn rate]. Sideways error gets the most weight because that's what decides whether the robot clears the obstacles and lines up with the dock. Driving straight at 0.15 m/s, the closed-loop poles are −3.16, −1.94 and −0.49, so it's stable.

To check the choice, we ran the same run with a softer and a more aggressive $Q$ (right graph):

| Q | Sideways RMSE | Steering effort | Docking |
| --- | --- | --- | --- |
| diag(1, 2, 0.2) | 2.4 cm | 0.71 rad/s² | 5.1 cm / 5.8°, missed |
| diag(10, 20, 2) (ours) | 1.0 cm | 0.76 rad/s² | 1.8 cm / 3.0° |
| diag(100, 200, 20) | 1.2 cm | 1.22 rad/s² | 1.2 cm / 1.6° |

Steering effort is the RMS rate of change of the turn-rate command.

The soft setting drifts too far off the route and misses the dock. The aggressive one actually docked a bit closer, but it steers about 60% harder, and harder steering means more wheel slip and worse odometry. Ours tracked about as well as the aggressive one while steering almost as smoothly as the soft one, so we kept it.

In the normal run the sideways tracking error RMSE was 1.0 cm and the heading RMSE was 6.7°. The big heading spike at around 30 s is most likely the sharp turn onto the dock approach, where the route turns faster than the robot's 2 rad/s limit. It still docked at 1.8 cm and 3.0°.

---

## E. Integration and robustness

We tested the full system with:
- a random start pose and sensor noise in every trial
- random pushes of 2–4 N for 0.3 s, which the encoders don't notice because the wheels slip
- the real right wheel 1% bigger than the odometry assumes
- obstacles moved to random positions, which the robot has to find with the LiDAR
- the marker being out of view for most of the route

Results from `run_trials.py`:

| Condition | Docked | Mean dock error* | Mean estimate RMSE |
| --- | --- | --- | --- |
| Normal (n = 20) | 19 / 20 | 1.6 cm | 6.4 cm |
| Random pushes (n = 20) | 20 / 20 | 1.6 cm | 6.2 cm |
| 1% wheel-size error (n = 20) | 18 / 20 | 1.9 cm | 7.3 cm |
| Random obstacles, LiDAR planning (n = 8) | 7 / 8 | 0.8 cm | 5.4 cm |
| Random obstacles, fixed route (n = 8) | 5 / 8 | 1.0 cm | 4.7 cm |

\*Mean over the runs that docked. Failed runs are described below.

The last two rows compare our LiDAR planning with a fixed hand-placed route when the obstacles move. The fixed route failed 3 of 8 times because it drove into obstacles that had moved onto it, and the robot stopped about 2.5 m short of the dock. With LiDAR planning, only 1 of 8 failed, and that one reached the dock but was 6.6 cm out.

The two failures with wheel-size error were near misses (3.8 cm and 6.2 cm, against the 3 cm tolerance). The one normal-run failure (seed 114) was different. Its random start sent the planner on a long route south of the obstacles, and the marker wasn't seen until late. Over that time, the heading drifted about 17° and the position estimate drifted almost 50 cm. The robot ended up against an obstacle it thought it was clear of, and the safety stop held it there. This is the main weakness of having only one marker (see Limitations).

<img width="1690" height="1040" alt="summary" src="https://github.com/user-attachments/assets/ced791a3-0a35-4d5c-a517-5e4641ec1816" />


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
```

## Testing

```bash
python -m pytest tests
```

`tests/test_no_ground_truth.py` checks that the EKF, controller, path and navigation code never read the simulator's true position, so the controller only ever uses the estimated state.

## Team

| Member | Responsibilities |
| --- | --- |
| Amani | Sensing and sensor calibration |
| All 3 members | State estimation and EKF |
| Dev (Aarav) | State-space modelling and trajectory tracking |
| Anotida | Simulation integration, testing and evaluation |
| Dev (Aarav) + Amani | project page |
| Amani + Anotida | video editing |

## Limitations and improvements

- **Overconfident filter:** the mean NEES is 13.9, where about 3 is expected, so the EKF thinks it is more certain than it is. Increasing the odometry noise (process noise) would make its uncertainty more realistic.
- **Late marker sighting:** the marker is only seen in the last third of the route, so the estimate drifts up to 10 cm on odometry alone. A second marker partway along, or matching LiDAR scans to the map, would correct it earlier.
- **Cautious routes:** the planner keeps about 40 cm from obstacles, so the robot goes around the obstacle group instead of weaving between them. Lowering the clearance would give shorter routes but less safety margin.
- **Close-range dropouts:** marker detection occasionally misses a frame within about 17 cm of the marker, so the last few centimetres rely mostly on odometry.


## AI Decleration


