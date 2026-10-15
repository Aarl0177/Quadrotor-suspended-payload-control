import time
import numpy as np
import casadi as ca
import mujoco
import mujoco.viewer
import imageio.v2 as imageio
import matplotlib.pyplot as plt


# ============================================================
# NMPC and simulation settings
# ============================================================
N = 25                  # NMPC prediction horizon
Nt = 120                # closed-loop simulation steps
dt = 0.10               # sample time [s]

# Quadrotor physical parameters
mass = 1.0              # kg
gravity = 9.81          # m/s^2
Ix = 0.020              # kg m^2
Iy = 0.020
Iz = 0.040

# State:
# x = [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
x_dim = 12

# Control:
# u = [total_thrust, tau_roll, tau_pitch, tau_yaw]
u_dim = 4

hover_control = np.array([mass * gravity, 0.0, 0.0, 0.0])

# Input limits
thrust_min = 0.0
thrust_max = 2.2 * mass * gravity
tau_roll_max = 0.80
tau_pitch_max = 0.80
tau_yaw_max = 0.40

# State limits
z_min, z_max = 0.25, 6.0
max_roll = np.deg2rad(45.0)
max_pitch = np.deg2rad(45.0)

# Tracking, input, input-rate, and terminal weights
Q = np.diag([
    20.0, 20.0, 30.0,       # position
     3.0,  3.0,  5.0,       # linear velocity
     8.0,  8.0,  2.0,       # roll, pitch, yaw
     0.4,  0.4,  0.2        # body angular rates
])

R = np.diag([
    0.04,                    # thrust deviation from hover
    0.30, 0.30, 0.15        # body torques
])

Rd = np.diag([
    0.08,                    # thrust-rate smoothing
    0.40, 0.40, 0.20        # torque-rate smoothing
])

Qf = np.diag([
    80.0, 80.0, 100.0,
    10.0, 10.0,  15.0,
    15.0, 15.0,   5.0,
     1.0,  1.0,   0.5
])

# Obstacle and safety settings
obs_radius = 0.35
quad_radius = 0.30
margin = 0.25
safe_radius = obs_radius + quad_radius + margin
slack_weight = 1.0e6

# Reference motion
desired_speed = 1.0

# Video settings
SAVE_VIDEO = True
VIDEO_PATH = "quadrotor_nmpc_obstacle_avoidance.mp4"
VIDEO_FPS = int(round(1.0 / dt))
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720


# ============================================================
# Quadrotor dynamics
# ============================================================
def rotation_matrix_zyx_symbolic(roll, pitch, yaw):
    """Body-to-world rotation matrix for ZYX Euler angles."""
    cr, sr = ca.cos(roll), ca.sin(roll)
    cp, sp = ca.cos(pitch), ca.sin(pitch)
    cy, sy = ca.cos(yaw), ca.sin(yaw)

    return ca.vertcat(
        ca.horzcat(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        ca.horzcat(sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        ca.horzcat(-sp, cp * sr, cp * cr)
    )


def rotation_matrix_zyx_numeric(roll, pitch, yaw):
    """Numeric body-to-world rotation matrix for ZYX Euler angles."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]
    ])


def quadrotor_continuous_symbolic(state, control):
    """Continuous-time nonlinear rigid-body quadrotor dynamics."""
    px, py, pz = state[0], state[1], state[2]
    vx, vy, vz = state[3], state[4], state[5]
    roll, pitch, yaw = state[6], state[7], state[8]
    p, q, r = state[9], state[10], state[11]

    thrust = control[0]
    tau_roll = control[1]
    tau_pitch = control[2]
    tau_yaw = control[3]

    rotation = rotation_matrix_zyx_symbolic(roll, pitch, yaw)
    thrust_world = rotation @ ca.vertcat(0.0, 0.0, thrust)
    acceleration = thrust_world / mass - ca.vertcat(0.0, 0.0, gravity)

    # ZYX Euler-angle kinematics
    cos_pitch_safe = ca.cos(pitch) + 1.0e-6
    tan_pitch = ca.sin(pitch) / cos_pitch_safe

    roll_dot = p + q * ca.sin(roll) * tan_pitch + r * ca.cos(roll) * tan_pitch
    pitch_dot = q * ca.cos(roll) - r * ca.sin(roll)
    yaw_dot = (
        q * ca.sin(roll) / cos_pitch_safe
        + r * ca.cos(roll) / cos_pitch_safe
    )

    # Rigid-body angular dynamics for diagonal inertia
    p_dot = (tau_roll + (Iy - Iz) * q * r) / Ix
    q_dot = (tau_pitch + (Iz - Ix) * p * r) / Iy
    r_dot = (tau_yaw + (Ix - Iy) * p * q) / Iz

    return ca.vertcat(
        vx, vy, vz,
        acceleration[0], acceleration[1], acceleration[2],
        roll_dot, pitch_dot, yaw_dot,
        p_dot, q_dot, r_dot
    )


def quadrotor_continuous_numeric(state, control):
    """Numeric version of the nonlinear rigid-body dynamics."""
    vx, vy, vz = state[3], state[4], state[5]
    roll, pitch, yaw = state[6], state[7], state[8]
    p, q, r = state[9], state[10], state[11]

    thrust, tau_roll, tau_pitch, tau_yaw = control

    rotation = rotation_matrix_zyx_numeric(roll, pitch, yaw)
    thrust_world = rotation @ np.array([0.0, 0.0, thrust])
    acceleration = thrust_world / mass - np.array([0.0, 0.0, gravity])

    cos_pitch_safe = np.cos(pitch)
    if abs(cos_pitch_safe) < 1.0e-6:
        cos_pitch_safe = np.sign(cos_pitch_safe) * 1.0e-6 if cos_pitch_safe != 0 else 1.0e-6

    tan_pitch = np.sin(pitch) / cos_pitch_safe

    roll_dot = p + q * np.sin(roll) * tan_pitch + r * np.cos(roll) * tan_pitch
    pitch_dot = q * np.cos(roll) - r * np.sin(roll)
    yaw_dot = (
        q * np.sin(roll) / cos_pitch_safe
        + r * np.cos(roll) / cos_pitch_safe
    )

    p_dot = (tau_roll + (Iy - Iz) * q * r) / Ix
    q_dot = (tau_pitch + (Iz - Ix) * p * r) / Iy
    r_dot = (tau_yaw + (Ix - Iy) * p * q) / Iz

    return np.array([
        vx, vy, vz,
        acceleration[0], acceleration[1], acceleration[2],
        roll_dot, pitch_dot, yaw_dot,
        p_dot, q_dot, r_dot
    ])


def rk4_step_symbolic(state, control):
    """One symbolic RK4 integration step."""
    k1 = quadrotor_continuous_symbolic(state, control)
    k2 = quadrotor_continuous_symbolic(state + 0.5 * dt * k1, control)
    k3 = quadrotor_continuous_symbolic(state + 0.5 * dt * k2, control)
    k4 = quadrotor_continuous_symbolic(state + dt * k3, control)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_step_numeric(state, control):
    """One numeric RK4 integration step."""
    k1 = quadrotor_continuous_numeric(state, control)
    k2 = quadrotor_continuous_numeric(state + 0.5 * dt * k1, control)
    k3 = quadrotor_continuous_numeric(state + 0.5 * dt * k2, control)
    k4 = quadrotor_continuous_numeric(state + dt * k3, control)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ============================================================
# NMPC controller
# ============================================================
class QuadrotorNMPC:
    def __init__(self, num_obstacles):
        self.num_obstacles = num_obstacles
        self.opti = ca.Opti()

        self.X = self.opti.variable(x_dim, N + 1)
        self.U = self.opti.variable(u_dim, N)
        self.S = self.opti.variable(num_obstacles, N + 1)

        self.X0_param = self.opti.parameter(x_dim)
        self.Xref_param = self.opti.parameter(x_dim, N + 1)
        self.Uprev_param = self.opti.parameter(u_dim)

        # Obstacle prediction is packed as:
        # rows 3*j : 3*j+3 -> [x, y, z] of obstacle j
        self.Obs_param = self.opti.parameter(3 * num_obstacles, N + 1)

        self._build_problem()

        self.last_X = None
        self.last_U = None
        self.last_S = None

    def _build_problem(self):
        Q_ca = ca.DM(Q)
        R_ca = ca.DM(R)
        Rd_ca = ca.DM(Rd)
        Qf_ca = ca.DM(Qf)
        hover_ca = ca.DM(hover_control)

        self.opti.subject_to(self.X[:, 0] == self.X0_param)
        # CasADi treats an inequality on a 2-D matrix as a
        # positive-semidefinite matrix inequality. Vectorize S so that
        # the lower bound is applied element by element.
        self.opti.subject_to(
            self.opti.bounded(0.0, ca.vec(self.S), ca.inf)
        )

        cost = 0.0

        for k in range(N):
            x_k = self.X[:, k]
            u_k = self.U[:, k]

            self.opti.subject_to(self.X[:, k + 1] == rk4_step_symbolic(x_k, u_k))

            # Input constraints
            self.opti.subject_to(
                self.opti.bounded(thrust_min, u_k[0], thrust_max)
            )
            self.opti.subject_to(
                self.opti.bounded(-tau_roll_max, u_k[1], tau_roll_max)
            )
            self.opti.subject_to(
                self.opti.bounded(-tau_pitch_max, u_k[2], tau_pitch_max)
            )
            self.opti.subject_to(
                self.opti.bounded(-tau_yaw_max, u_k[3], tau_yaw_max)
            )

            # Flight-envelope constraints
            self.opti.subject_to(
                self.opti.bounded(z_min, x_k[2], z_max)
            )
            self.opti.subject_to(
                self.opti.bounded(-max_roll, x_k[6], max_roll)
            )
            self.opti.subject_to(
                self.opti.bounded(-max_pitch, x_k[7], max_pitch)
            )

            # 3D spherical obstacle-avoidance constraints
            for j in range(self.num_obstacles):
                obstacle_position = self.Obs_param[3 * j:3 * j + 3, k]
                position_error = x_k[0:3] - obstacle_position
                distance_squared = ca.dot(position_error, position_error)

                self.opti.subject_to(
                    distance_squared >= safe_radius**2 - self.S[j, k]
                )
                cost += slack_weight * self.S[j, k] ** 2

            state_error = x_k - self.Xref_param[:, k]
            input_error = u_k - hover_ca

            if k == 0:
                delta_u = u_k - self.Uprev_param
            else:
                delta_u = u_k - self.U[:, k - 1]

            cost += ca.mtimes([state_error.T, Q_ca, state_error])
            cost += ca.mtimes([input_error.T, R_ca, input_error])
            cost += ca.mtimes([delta_u.T, Rd_ca, delta_u])

        # Terminal flight envelope
        self.opti.subject_to(
            self.opti.bounded(z_min, self.X[2, N], z_max)
        )
        self.opti.subject_to(
            self.opti.bounded(-max_roll, self.X[6, N], max_roll)
        )
        self.opti.subject_to(
            self.opti.bounded(-max_pitch, self.X[7, N], max_pitch)
        )

        # Terminal obstacle constraints
        for j in range(self.num_obstacles):
            obstacle_position = self.Obs_param[3 * j:3 * j + 3, N]
            position_error = self.X[0:3, N] - obstacle_position
            distance_squared = ca.dot(position_error, position_error)

            self.opti.subject_to(
                distance_squared >= safe_radius**2 - self.S[j, N]
            )
            cost += slack_weight * self.S[j, N] ** 2

        terminal_error = self.X[:, N] - self.Xref_param[:, N]
        cost += ca.mtimes([terminal_error.T, Qf_ca, terminal_error])

        self.opti.minimize(cost)

        solver_options = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 500,
            "ipopt.tol": 1.0e-4,
            "ipopt.acceptable_tol": 1.0e-3,
            "ipopt.warm_start_init_point": "yes"
        }

        self.opti.solver("ipopt", solver_options)

    @staticmethod
    def reference_trajectory(current_state, goal_position):
        """
        Build a dynamically reasonable reference:
        move from the current position toward the goal at desired_speed.
        """
        reference = np.zeros((x_dim, N + 1))

        current_position = current_state[0:3]
        displacement = goal_position - current_position
        distance = np.linalg.norm(displacement)

        if distance > 1.0e-8:
            direction = displacement / distance
            reference_yaw = np.arctan2(direction[1], direction[0])
        else:
            direction = np.zeros(3)
            reference_yaw = current_state[8]

        for k in range(N + 1):
            travel = min(desired_speed * dt * k, distance)
            position_ref = current_position + travel * direction

            if travel < distance:
                velocity_ref = desired_speed * direction
            else:
                velocity_ref = np.zeros(3)

            reference[0:3, k] = position_ref
            reference[3:6, k] = velocity_ref
            reference[6:9, k] = np.array([0.0, 0.0, reference_yaw])
            reference[9:12, k] = 0.0

        return reference

    def predict_obstacles(self, obstacle_states):
        """
        Constant-velocity obstacle prediction.

        obstacle_states[j] =
        [ox, oy, oz, ovx, ovy, ovz]
        """
        prediction = np.zeros((3 * self.num_obstacles, N + 1))

        for j in range(self.num_obstacles):
            position = obstacle_states[j, 0:3]
            velocity = obstacle_states[j, 3:6]

            for k in range(N + 1):
                prediction[3 * j:3 * j + 3, k] = (
                    position + velocity * dt * k
                )

        return prediction

    def solve(self, current_state, goal_position, obstacle_states, previous_control):
        reference = self.reference_trajectory(current_state, goal_position)
        obstacle_prediction = self.predict_obstacles(obstacle_states)

        self.opti.set_value(self.X0_param, current_state)
        self.opti.set_value(self.Xref_param, reference)
        self.opti.set_value(self.Obs_param, obstacle_prediction)
        self.opti.set_value(self.Uprev_param, previous_control)

        if self.last_X is None:
            self.opti.set_initial(self.X, reference)
            self.opti.set_initial(
                self.U,
                np.tile(hover_control.reshape(-1, 1), (1, N))
            )
            self.opti.set_initial(self.S, 0.01)
        else:
            shifted_X = np.hstack((self.last_X[:, 1:], self.last_X[:, -1:]))
            shifted_U = np.hstack((self.last_U[:, 1:], self.last_U[:, -1:]))
            shifted_S = np.hstack((self.last_S[:, 1:], self.last_S[:, -1:]))

            self.opti.set_initial(self.X, shifted_X)
            self.opti.set_initial(self.U, shifted_U)
            self.opti.set_initial(self.S, shifted_S)

        try:
            solution = self.opti.solve()

            self.last_X = solution.value(self.X)
            self.last_U = solution.value(self.U)
            self.last_S = solution.value(self.S)

            return self.last_U[:, 0].copy()

        except RuntimeError:
            print("NMPC solver failed:", self.opti.debug.return_status())
            print("Applying hover control for this step.")

            # Clear the failed warm start before the next call.
            self.last_X = None
            self.last_U = None
            self.last_S = None

            return hover_control.copy()


# ============================================================
# Closed-loop simulation
# ============================================================
def obstacle_dynamics_all(obstacle_states):
    """Update all moving obstacles with constant velocity."""
    next_states = obstacle_states.copy()
    next_states[:, 0:3] += dt * obstacle_states[:, 3:6]
    return next_states


def run_nmpc_simulation():
    # Initial state:
    # [px, py, pz, vx, vy, vz, roll, pitch, yaw, p, q, r]
    state = np.zeros(x_dim)
    state[0:3] = np.array([0.0, 0.0, 1.0])

    goal_position = np.array([10.0, 2.5, 3.0])

    # Moving obstacles:
    # [x, y, z, vx, vy, vz]
    obstacle_states = np.array([
        [3.2, 0.45, 1.55,  0.00,  0.12,  0.00],
        [5.8, 2.10, 2.15,  0.00, -0.30,  0.00],
        [8.1, 1.60, 2.70, -0.04,  0.00, -0.02]
    ])

    controller = QuadrotorNMPC(num_obstacles=obstacle_states.shape[0])

    state_history = [state.copy()]
    obstacle_history = [obstacle_states.copy()]
    control_history = []

    previous_control = hover_control.copy()

    for step in range(Nt):
        distance_to_goal = np.linalg.norm(state[0:3] - goal_position)
        print(
            f"NMPC step {step + 1:03d}/{Nt} | "
            f"distance to goal = {distance_to_goal:.3f} m"
        )

        control = controller.solve(
            state,
            goal_position,
            obstacle_states,
            previous_control
        )

        state = rk4_step_numeric(state, control)
        obstacle_states = obstacle_dynamics_all(obstacle_states)

        control_history.append(control.copy())
        state_history.append(state.copy())
        obstacle_history.append(obstacle_states.copy())

        previous_control = control.copy()

        # Stop once the quadrotor is close to the goal and nearly stationary.
        if (
            np.linalg.norm(state[0:3] - goal_position) < 0.15
            and np.linalg.norm(state[3:6]) < 0.15
        ):
            print("Goal reached.")
            break

    return (
        np.asarray(state_history),
        np.asarray(obstacle_history),
        np.asarray(control_history),
        goal_position
    )


# ============================================================
# MuJoCo visualization
# ============================================================
def euler_zyx_to_quaternion(roll, pitch, yaw):
    """
    Convert ZYX Euler angles to a MuJoCo quaternion [w, x, y, z].
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ])


def build_mujoco_xml(num_obstacles, start_position, goal_position):
    obstacle_bodies = "\n".join(
        f"""
        <body name="obs_{j}" pos="0 0 1">
            <geom name="obs_geom_{j}"
                  type="sphere"
                  size="{obs_radius}"
                  rgba="0.90 0.10 0.10 1"/>
        </body>

        <body name="safe_{j}" pos="0 0 1">
            <geom name="safe_geom_{j}"
                  type="sphere"
                  size="{safe_radius}"
                  rgba="1.0 0.0 0.0 0.10"
                  contype="0"
                  conaffinity="0"/>
        </body>
        """
        for j in range(num_obstacles)
    )

    path_fromto = (
        f"{start_position[0]} {start_position[1]} {start_position[2]} "
        f"{goal_position[0]} {goal_position[1]} {goal_position[2]}"
    )

    return f"""
<mujoco model="quadrotor_nmpc_obstacle_avoidance">

    <option timestep="{dt}" gravity="0 0 -9.81"/>

    <visual>
        <quality shadowsize="2048"/>
        <global offwidth="{VIDEO_WIDTH}" offheight="{VIDEO_HEIGHT}"/>
        <map znear="0.01" zfar="100"/>
    </visual>

    <asset>
        <texture name="grid"
                 type="2d"
                 builtin="checker"
                 width="512"
                 height="512"
                 rgb1="0.20 0.20 0.20"
                 rgb2="0.30 0.30 0.30"/>

        <material name="grid_mat"
                  texture="grid"
                  texrepeat="20 20"
                  reflectance="0.10"/>

        <material name="quad_blue" rgba="0.10 0.25 0.90 1"/>
        <material name="arm_dark" rgba="0.05 0.06 0.08 1"/>
        <material name="rotor_dark" rgba="0.02 0.02 0.02 1"/>
    </asset>

    <worldbody>
        <light pos="5 2 10"
               dir="0 0 -1"
               diffuse="1 1 1"/>

        <geom name="ground"
              type="plane"
              size="20 10 0.1"
              material="grid_mat"/>

        <!-- Straight reference line from start to goal -->
        <geom name="reference_path"
              type="capsule"
              fromto="{path_fromto}"
              size="0.025"
              rgba="0.10 0.85 0.20 0.65"
              contype="0"
              conaffinity="0"/>

        <!-- Goal marker -->
        <body name="goal" pos="{goal_position[0]} {goal_position[1]} {goal_position[2]}">
            <geom type="sphere"
                  size="0.18"
                  rgba="0.10 0.90 0.20 0.75"
                  contype="0"
                  conaffinity="0"/>
        </body>

        <!-- Kinematically replayed quadrotor -->
        <body name="quadrotor" pos="{start_position[0]} {start_position[1]} {start_position[2]}">

            <!-- Main body -->
            <geom name="quad_body"
                  type="box"
                  size="0.14 0.10 0.055"
                  material="quad_blue"/>

            <!-- Arms -->
            <geom type="capsule"
                  fromto="-0.30 0 0 0.30 0 0"
                  size="0.025"
                  material="arm_dark"/>

            <geom type="capsule"
                  fromto="0 -0.30 0 0 0.30 0"
                  size="0.025"
                  material="arm_dark"/>

            <!-- Rotors -->
            <geom name="rotor_front"
                  type="cylinder"
                  pos="0.30 0 0"
                  size="0.13 0.012"
                  material="rotor_dark"/>

            <geom name="rotor_rear"
                  type="cylinder"
                  pos="-0.30 0 0"
                  size="0.13 0.012"
                  material="rotor_dark"/>

            <geom name="rotor_left"
                  type="cylinder"
                  pos="0 0.30 0"
                  size="0.13 0.012"
                  material="rotor_dark"/>

            <geom name="rotor_right"
                  type="cylinder"
                  pos="0 -0.30 0"
                  size="0.13 0.012"
                  material="rotor_dark"/>

            <!-- Front indicator -->
            <geom type="sphere"
                  pos="0.18 0 0.02"
                  size="0.035"
                  rgba="0.95 0.80 0.10 1"/>
        </body>

        {obstacle_bodies}
    </worldbody>
</mujoco>
"""


def get_body_ids(model, num_obstacles):
    return {
        "quadrotor": model.body("quadrotor").id,
        "goal": model.body("goal").id,
        "obstacles": [
            model.body(f"obs_{j}").id for j in range(num_obstacles)
        ],
        "safe_zones": [
            model.body(f"safe_{j}").id for j in range(num_obstacles)
        ]
    }


def set_replay_frame(model, data, body_ids, state_history, obstacle_history, frame):
    state = state_history[frame]
    position = state[0:3]
    roll, pitch, yaw = state[6:9]

    model.body_pos[body_ids["quadrotor"]] = position
    model.body_quat[body_ids["quadrotor"]] = euler_zyx_to_quaternion(
        roll, pitch, yaw
    )

    for j in range(obstacle_history.shape[1]):
        obstacle_position = obstacle_history[frame, j, 0:3]
        model.body_pos[body_ids["obstacles"][j]] = obstacle_position
        model.body_pos[body_ids["safe_zones"][j]] = obstacle_position

    mujoco.mj_forward(model, data)


def save_mujoco_video(
    state_history,
    obstacle_history,
    control_history,
    goal_position,
    output_path=VIDEO_PATH
):
    del control_history  # replay only needs state and obstacle histories

    num_obstacles = obstacle_history.shape[1]
    start_position = state_history[0, 0:3]

    xml = build_mujoco_xml(num_obstacles, start_position, goal_position)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    body_ids = get_body_ids(model, num_obstacles)

    renderer = mujoco.Renderer(
        model,
        height=VIDEO_HEIGHT,
        width=VIDEO_WIDTH
    )
    writer = imageio.get_writer(
        output_path,
        fps=VIDEO_FPS,
        macro_block_size=1
    )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 14.0
    camera.azimuth = 135.0
    camera.elevation = -25.0
    camera.lookat[:] = 0.5 * (start_position + goal_position)

    print("Saving video to:", output_path)

    for frame in range(len(state_history)):
        set_replay_frame(
            model,
            data,
            body_ids,
            state_history,
            obstacle_history,
            frame
        )
        renderer.update_scene(data, camera=camera)
        writer.append_data(renderer.render())

    writer.close()
    renderer.close()
    print("Video saved:", output_path)


def visualize_in_mujoco(
    state_history,
    obstacle_history,
    control_history,
    goal_position
):
    del control_history

    num_obstacles = obstacle_history.shape[1]
    start_position = state_history[0, 0:3]

    xml = build_mujoco_xml(num_obstacles, start_position, goal_position)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    body_ids = get_body_ids(model, num_obstacles)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 14.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
        viewer.cam.lookat[:] = 0.5 * (start_position + goal_position)

        frame = 0

        while viewer.is_running():
            set_replay_frame(
                model,
                data,
                body_ids,
                state_history,
                obstacle_history,
                frame
            )
            viewer.sync()

            frame = (frame + 1) % len(state_history)
            time.sleep(dt)


# ============================================================
# Result plots
# ============================================================
def plot_results(
    state_history,
    obstacle_history,
    control_history,
    goal_position
):
    state_time = np.arange(len(state_history)) * dt
    control_time = np.arange(len(control_history)) * dt

    positions = state_history[:, 0:3]
    velocities = state_history[:, 3:6]
    euler_angles = state_history[:, 6:9]
    angular_rates = state_history[:, 9:12]

    # Controls
    control_labels = [
        "Total thrust [N]",
        "Roll torque [N m]",
        "Pitch torque [N m]",
        "Yaw torque [N m]"
    ]

    plt.figure(figsize=(10, 9))
    for i in range(u_dim):
        plt.subplot(4, 1, i + 1)
        plt.plot(control_time, control_history[:, i], linewidth=2)
        plt.ylabel(control_labels[i])
        plt.grid(True)
        if i == 0:
            plt.title("NMPC control inputs")
        if i == u_dim - 1:
            plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.savefig("quadrotor_controls.png", dpi=300)
    plt.show()

    # Position and velocity
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(state_time, positions[:, 0], label="x", linewidth=2)
    plt.plot(state_time, positions[:, 1], label="y", linewidth=2)
    plt.plot(state_time, positions[:, 2], label="z", linewidth=2)
    plt.ylabel("Position [m]")
    plt.title("Quadrotor translational states")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(state_time, velocities[:, 0], label="vx", linewidth=2)
    plt.plot(state_time, velocities[:, 1], label="vy", linewidth=2)
    plt.plot(state_time, velocities[:, 2], label="vz", linewidth=2)
    plt.ylabel("Velocity [m/s]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("quadrotor_translation_states.png", dpi=300)
    plt.show()

    # Attitude and angular rates
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(state_time, np.rad2deg(euler_angles[:, 0]), label="roll", linewidth=2)
    plt.plot(state_time, np.rad2deg(euler_angles[:, 1]), label="pitch", linewidth=2)
    plt.plot(state_time, np.rad2deg(euler_angles[:, 2]), label="yaw", linewidth=2)
    plt.ylabel("Euler angles [deg]")
    plt.title("Quadrotor rotational states")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(state_time, angular_rates[:, 0], label="p", linewidth=2)
    plt.plot(state_time, angular_rates[:, 1], label="q", linewidth=2)
    plt.plot(state_time, angular_rates[:, 2], label="r", linewidth=2)
    plt.ylabel("Angular rate [rad/s]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("quadrotor_rotation_states.png", dpi=300)
    plt.show()

    # 3D trajectory
    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")

    axis.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        linewidth=3,
        label="Quadrotor trajectory"
    )
    axis.scatter(
        positions[0, 0],
        positions[0, 1],
        positions[0, 2],
        s=80,
        marker="o",
        label="Start"
    )
    axis.scatter(
        goal_position[0],
        goal_position[1],
        goal_position[2],
        s=100,
        marker="*",
        label="Goal"
    )

    for j in range(obstacle_history.shape[1]):
        obstacle_path = obstacle_history[:, j, 0:3]
        axis.plot(
            obstacle_path[:, 0],
            obstacle_path[:, 1],
            obstacle_path[:, 2],
            linestyle="--",
            linewidth=2,
            label=f"Obstacle {j + 1} path"
        )
        axis.scatter(
            obstacle_path[0, 0],
            obstacle_path[0, 1],
            obstacle_path[0, 2],
            s=60,
            marker="o"
        )

    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title("3D quadrotor trajectory and moving obstacles")
    axis.legend()
    figure.tight_layout()
    figure.savefig("quadrotor_3d_trajectory.png", dpi=300)
    plt.show()

    # Safety-distance history
    plt.figure(figsize=(10, 6))

    for j in range(obstacle_history.shape[1]):
        relative_position = (
            positions - obstacle_history[:, j, 0:3]
        )
        distances = np.linalg.norm(relative_position, axis=1)
        plt.plot(
            state_time,
            distances,
            linewidth=2,
            label=f"Distance to obstacle {j + 1}"
        )

    plt.axhline(
        safe_radius,
        linestyle="--",
        linewidth=2,
        label="Required safety radius"
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Center-to-center distance [m]")
    plt.title("Obstacle-clearance history")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("quadrotor_obstacle_clearance.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    (
        state_history,
        obstacle_history,
        control_history,
        goal_position
    ) = run_nmpc_simulation()

    plot_results(
        state_history,
        obstacle_history,
        control_history,
        goal_position
    )

    if SAVE_VIDEO:
        save_mujoco_video(
            state_history,
            obstacle_history,
            control_history,
            goal_position,
            VIDEO_PATH
        )

    visualize_in_mujoco(
        state_history,
        obstacle_history,
        control_history,
        goal_position
    )