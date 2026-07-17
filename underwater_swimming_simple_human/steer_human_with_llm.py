import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import mujoco
import mujoco.viewer
import numpy as np
import requests
import json
import time

import queue
import threading
import copy
from scipy.spatial.transform import Rotation as R  # <--- Add this import


# --- Configuration ---
MODEL_PATH = 'underwater_swimming_simple_human/humanoid_laying.xml'
WATER_SURFACE_HEIGHT = 3.0
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
class LLMHumanoidController:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.step_count = 0
        self.current_targets = {}
        self.limb_bases = {
            'hand_left': 'torso',
            'hand_right': 'torso',
            'foot_left': 'torso',
            'foot_right': 'torso',
        }
        self.integral_error = {limb: np.zeros(3) for limb in self.limb_bases.keys()}
        self.prev_error = {}

    def get_euler_angles(self):
        quat_mujoco = self.data.qpos[3:7] 
        quat_scipy = np.array([quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]])
        return R.from_quat(quat_scipy).as_euler('xyz', degrees=False) 
    
    def get_local_pos(self, body_name):
        base_name = self.limb_bases.get(body_name)
        base_pos = self.data.xpos[self.model.body(base_name).id]
        target_pos = self.data.xpos[self.model.body(body_name).id]
        relative_vec_global = target_pos - base_pos
        base_mat = self.data.xmat[self.model.body(base_name).id].reshape(3, 3)
        return (base_mat.T @ relative_vec_global).tolist()

    def get_obs(self):
        head_height = self.data.xpos[self.model.body('head').id][2]
        torso_pos = self.data.xpos[self.model.body('torso').id]
        return {
            "head_height": head_height, "body_height": torso_pos[2],
            "full_state": {
                "left_arm_pos": self.get_local_pos('hand_left'),
                "right_arm_pos": self.get_local_pos('hand_right'),
                "left_leg_pos": self.get_local_pos('foot_left'),
                "right_leg_pos": self.get_local_pos('foot_right'),
            }
        }

    def set_targets(
        self,
        action_json,
        reset_integral=False,
    ):
        """
        Update the current Cartesian targets.

        Parameters
        ----------
        action_json : dict
            {
                "left_arm": [...],
                "right_arm": [...],
                "left_leg": [...],
                "right_leg": [...],
            }

        reset_integral : bool
            If True, reset the integral error for the updated limbs.
            Leave False during continuous swimming.
        """

        mapping = {
            "left_arm": "hand_left",
            "right_arm": "hand_right",
            "left_leg": "foot_left",
            "right_leg": "foot_right",
        }

        self.current_targets = {
            body_name: np.asarray(
                action_json[action_name],
                dtype=np.float64,
            )
            for action_name, body_name in mapping.items()
            if (
                action_name in action_json
                and action_json[action_name] is not None
            )
        }

        if reset_integral:
            for body_name in self.current_targets:
                self.integral_error[body_name][:] = 0.0

        self.prev_error = {
            body_name: float("inf")
            for body_name in self.current_targets
        }
    
    def physics_step(self, debug=False) -> bool:
        """
        Move active end-effectors toward torso-relative Cartesian targets
        using Jacobian-transpose PD control.

        Returns True when every active target is within the position
        threshold and below the velocity threshold.
        """

        KP = 100.0
        KD = 12.0
        KI =  0.0 #12.0

        

        POSITION_THRESHOLD = 0.01
        VELOCITY_THRESHOLD = 0.05
        MAX_CARTESIAN_FORCE = 100.0
        INTEGRAL_LIMIT = 0.15

        PHYSICS_STEPS_PER_CONTROL = 1

        self.model.opt.timestep = 0.002

        # Clear controls once before processing all active targets.
        self.data.ctrl[:] = 0.0

        reached_count = 0
        active_target_count = 0

        for body_name, local_target in self.current_targets.items():
            if local_target is None:
                continue

            active_target_count += 1

            target_local = np.asarray(
                local_target,
                dtype=np.float64,
            )

            body_id = self.model.body(body_name).id

            base_name = self.limb_bases[body_name]
            base_id = self.model.body(base_name).id

            # ---------------------------------
            # 1. Current torso-relative position
            # ---------------------------------
            body_pos_world = self.data.xpos[body_id].copy()
            base_pos_world = self.data.xpos[base_id].copy()

            base_rot_world = (
                self.data.xmat[base_id]
                .reshape(3, 3)
                .copy()
            )

            relative_position_world = (
                body_pos_world - base_pos_world
            )

            current_local = (
                base_rot_world.T
                @ relative_position_world
            )

            error_local = target_local - current_local
            distance = float(np.linalg.norm(error_local))

            # ---------------------------------
            # 2. Hand and torso Jacobians
            # ---------------------------------
            jac_hand_pos = np.zeros(
                (3, self.model.nv),
                dtype=np.float64,
            )
            jac_hand_rot = np.zeros(
                (3, self.model.nv),
                dtype=np.float64,
            )

            jac_base_pos = np.zeros(
                (3, self.model.nv),
                dtype=np.float64,
            )
            jac_base_rot = np.zeros(
                (3, self.model.nv),
                dtype=np.float64,
            )

            mujoco.mj_jacBody(
                self.model,
                self.data,
                jac_hand_pos,
                jac_hand_rot,
                body_id,
            )

            mujoco.mj_jacBody(
                self.model,
                self.data,
                jac_base_pos,
                jac_base_rot,
                base_id,
            )

            # ---------------------------------
            # 3. Exact torso-local Jacobian
            # ---------------------------------
            def skew(vector):
                x, y, z = vector

                return np.array([
                    [0.0, -z, y],
                    [z, 0.0, -x],
                    [-y, x, 0.0],
                ])

            # Local position:
            # p_local = R_base.T @ (p_hand - p_base)
            #
            # Its velocity includes:
            # - hand translation
            # - torso translation
            # - torso rotation
            jac_relative_world = (
                jac_hand_pos
                - jac_base_pos
                + skew(relative_position_world)
                @ jac_base_rot
            )

            jac_local = (
                base_rot_world.T
                @ jac_relative_world
            )

            velocity_local = (
                jac_local @ self.data.qvel
            )

            speed = float(
                np.linalg.norm(velocity_local)
            )

            # ---------------------------------
            # 4. Check whether target is reached
            # ---------------------------------
            target_reached = (
                distance <= POSITION_THRESHOLD
                and speed <= VELOCITY_THRESHOLD
            )

            if target_reached:
                reached_count += 1

                if debug:
                    print(f"\nBody: {body_name}")
                    print("Target reached")
                    print(
                        "Position error:",
                        round(distance, 5),
                    )
                    print(
                        "Relative speed:",
                        round(speed, 5),
                    )

                continue

            # ---------------------------------
            # 5. Cartesian PD in torso frame
            # ---------------------------------
            dt = self.model.opt.timestep

            if distance < 0.005:
                self.integral_error[body_name] *= 0.9

            # Accumulate Cartesian error.
            self.integral_error[body_name] += (
                error_local * dt
            )

            # Anti-windup.
            self.integral_error[body_name] = np.clip(
                self.integral_error[body_name],
                -INTEGRAL_LIMIT,
                INTEGRAL_LIMIT,
            )

            p_force = KP * error_local

            i_force = (
                KI
                * self.integral_error[body_name]
            )

            d_force = -KD * velocity_local

            cartesian_force = (
                p_force
                + i_force
                + d_force
            )

            force_norm = float(
                np.linalg.norm(cartesian_force)
            )

            if force_norm > MAX_CARTESIAN_FORCE:
                cartesian_force *= (
                    MAX_CARTESIAN_FORCE
                    / force_norm
                )

            # The Jacobian and force are both torso-local.
            desired_torque = (
                jac_local.T @ cartesian_force
            )

            # ---------------------------------
            # 5. Map generalized torque
            #    to actuator controls
            # ---------------------------------
            for actuator_id in range(self.model.nu):
                transmission_type = (
                    self.model.actuator_trntype[
                        actuator_id
                    ]
                )

                if (
                    transmission_type
                    != mujoco.mjtTrn.mjTRN_JOINT
                ):
                    continue

                joint_id = int(
                    self.model.actuator_trnid[
                        actuator_id,
                        0,
                    ]
                )

                if joint_id < 0:
                    continue

                dof_id = int(
                    self.model.jnt_dofadr[joint_id]
                )

                gear = float(
                    self.model.actuator_gear[
                        actuator_id,
                        0,
                    ]
                )

                if abs(gear) < 1e-8:
                    continue

                joint_torque = desired_torque[dof_id]

                # Use += so multiple active targets can
                # contribute to the same actuator.
                self.data.ctrl[actuator_id] += (
                    joint_torque / gear
                )

            # ---------------------------------
            # 6. Debug output
            # ---------------------------------


            if debug:
                print(f"\nBody: {body_name}")

                print(
                    "Target local:",
                    np.round(target_local, 4),
                )
                print(
                    "Current local:",
                    np.round(current_local, 4),
                )
                print(
                    "Error local:",
                    np.round(error_local, 5),
                )
                print(
                    "Distance:",
                    round(distance, 5),
                )
                print(
                    "Velocity local:",
                    np.round(velocity_local, 5),
                )
                print(
                    "Speed:",
                    round(speed, 5),
                )
                print(
                    "P-force norm:",
                    round(
                        np.linalg.norm(p_force),
                        5,
                    ),
                )
                print(
                    "D-force norm:",
                    round(
                        np.linalg.norm(d_force),
                        5,
                    ),
                )

                print(
                    "I-force norm:",
                    round(
                        np.linalg.norm(i_force),
                        5,
                    ),
                )
                print(
                    "Cartesian force norm:",
                    round(
                        np.linalg.norm(cartesian_force),
                        5,
                    ),
                )
                print(
                    "Max |desired torque|:",
                    round(
                        np.max(np.abs(desired_torque)),
                        5,
                    ),
                )

            

        # ---------------------------------
        # 7. Apply actuator control limits
        # ---------------------------------
        for actuator_id in range(self.model.nu):
            if self.model.actuator_ctrllimited[
                actuator_id
            ]:
                ctrl_min, ctrl_max = (
                    self.model.actuator_ctrlrange[
                        actuator_id
                    ]
                )

                self.data.ctrl[actuator_id] = (
                    np.clip(
                        self.data.ctrl[actuator_id],
                        ctrl_min,
                        ctrl_max,
                    )
                )


        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.step_count += 1

        if active_target_count == 0:
            return True

        return reached_count == active_target_count
# ---------------------------------------------------------------------
# LLM SWIMMING CONTROL
# ---------------------------------------------------------------------

REQUIRED_ACTION_KEYS = {
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
}

ACTION_TO_OBSERVATION_KEY = {
    "left_arm": "left_arm_pos",
    "right_arm": "right_arm_pos",
    "left_leg": "left_leg_pos",
    "right_leg": "right_leg_pos",
}

# Maximum change for each Cartesian coordinate between two actions.
MAX_ACTION_DELTA = 0.10

# Broad physical workspaces.
# These are safety limits, not desired poses.
ACTION_LIMITS = {
    "left_arm": {
        "min": np.array(
            [-0.60, -0.65, -0.60],
            dtype=np.float64,
        ),
        "max": np.array(
            [0.60, 0.65, 0.60],
            dtype=np.float64,
        ),
    },
    "right_arm": {
        "min": np.array(
            [-0.60, -0.65, -0.60],
            dtype=np.float64,
        ),
        "max": np.array(
            [0.60, 0.65, 0.60],
            dtype=np.float64,
        ),
    },
    "left_leg": {
        "min": np.array(
            [-0.30, -0.30, -1.20],
            dtype=np.float64,
        ),
        "max": np.array(
            [0.30, 0.30, 0.10],
            dtype=np.float64,
        ),
    },
    "right_leg": {
        "min": np.array(
            [-0.30, -0.30, -1.20],
            dtype=np.float64,
        ),
        "max": np.array(
            [0.30, 0.30, 0.10],
            dtype=np.float64,
        ),
    },
}


def get_llm_action(prompt):
    """
    Request one Cartesian swimming action from Ollama.

    Returns:
        dict | None
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.4,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        response_data = response.json()
        raw_response = response_data.get(
            "response",
            "",
        )

        if isinstance(raw_response, dict):
            return raw_response

        return json.loads(raw_response)

    except requests.RequestException as exc:
        print(f"LLM request error: {exc}")
        return None

    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"LLM response error: {exc}")
        return None


def validate_action(action):
    """
    Verify that the LLM returned all four limbs as finite 3-D vectors.
    """
    if not isinstance(action, dict):
        return False

    if not REQUIRED_ACTION_KEYS.issubset(
        action.keys()
    ):
        return False

    for action_name in REQUIRED_ACTION_KEYS:
        target = action[action_name]

        if not isinstance(
            target,
            (list, tuple, np.ndarray),
        ):
            return False

        try:
            target_array = np.asarray(
                target,
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            return False

        if target_array.shape != (3,):
            return False

        if not np.all(
            np.isfinite(target_array)
        ):
            return False

    return True


def sanitize_action(
    requested_action,
    current_positions,
):
    """
    Apply only general safety constraints.

    This function:
    - converts targets to floats
    - limits movement to MAX_ACTION_DELTA per coordinate
    - clips targets to broad limb workspaces

    It does not impose a preferred leg posture.
    """
    sanitized_action = {}

    for action_name in REQUIRED_ACTION_KEYS:
        observation_name = (
            ACTION_TO_OBSERVATION_KEY[
                action_name
            ]
        )

        requested_target = np.asarray(
            requested_action[action_name],
            dtype=np.float64,
        )

        current_position = np.asarray(
            current_positions[
                observation_name
            ],
            dtype=np.float64,
        )

        requested_delta = (
            requested_target
            - current_position
        )

        safe_delta = np.clip(
            requested_delta,
            -MAX_ACTION_DELTA,
            MAX_ACTION_DELTA,
        )

        safe_target = (
            current_position
            + safe_delta
        )

        lower_limit = ACTION_LIMITS[
            action_name
        ]["min"]

        upper_limit = ACTION_LIMITS[
            action_name
        ]["max"]

        safe_target = np.clip(
            safe_target,
            lower_limit,
            upper_limit,
        )

        sanitized_action[action_name] = (
            safe_target.tolist()
        )

    return sanitized_action


def format_for_print(value):
    """
    Produce shorter readable output for logs.
    """
    if isinstance(value, dict):
        return {
            key: format_for_print(item)
            for key, item in value.items()
        }

    if isinstance(value, np.ndarray):
        return format_for_print(
            value.tolist()
        )

    if isinstance(value, (list, tuple)):
        return [
            format_for_print(item)
            for item in value
        ]

    if isinstance(
        value,
        (float, np.floating),
    ):
        return round(float(value), 3)

    return value


def build_llm_prompt(
    observation,
    action_history,
):
    """
    Construct the swimming prompt from the latest state.
    """
    full_state = observation["full_state"]

    recent_actions = action_history[-10:]

    return f"""
You control a humanoid robot learning to swim backstroke in MuJoCo.

All limb coordinates are expressed in the LOCAL TORSO FRAME.

CURRENT STATE

Head height relative to the water surface:
{observation["head_height"] - WATER_SURFACE_HEIGHT:.3f}

Torso height relative to the water surface:
{observation["body_height"] - WATER_SURFACE_HEIGHT:.3f}

Current torso-relative limb positions:

left_arm:
{full_state["left_arm_pos"]}

right_arm:
{full_state["right_arm_pos"]}

left_leg:
{full_state["left_leg_pos"]}

right_leg:
{full_state["right_leg_pos"]}

COORDINATE SYSTEM

x:
- positive moves toward the sky
- negative moves toward the floor

y:
- positive moves toward the humanoid's left
- negative moves toward the humanoid's right

z:
- positive moves toward the head
- negative moves toward the feet and away from the torso

SWIMMING GOAL

Perform a continuous backstroke motion.

Arms:
- Alternate the arms.
- One arm should recover toward and above the head.
- The other arm should pull through the water toward the hip.
- Advance the stroke gradually.
- Avoid moving both arms through exactly the same phase.

Legs:
- Produce a smooth alternating kicking motion.
- Move one foot toward the sky while the other moves toward the floor.
- Reverse their directions gradually.
- Choose the leg extension based on the current pose and swimming motion.
- Avoid sudden, extreme or physically impossible changes.

GENERAL MOVEMENT RULES

- Return targets for all four limbs.
- Each target must contain exactly three numbers.
- Change each coordinate by at most {MAX_ACTION_DELTA:.2f} metres.
- Make smooth and incremental movements.
- Continue from the current pose.
- Do not jump directly to a distant stroke pose.
- Avoid repeatedly returning exactly the same action.
- Avoid large sideways movements unless needed for the stroke.
- Remain within the specified workspaces.

WORKSPACE LIMITS

left_arm:
- x: -0.60 to 0.60
- y: -0.65 to 0.65
- z: -0.60 to 0.60

right_arm:
- x: -0.60 to 0.60
- y: -0.65 to 0.65
- z: -0.60 to 0.60

left_leg:
- x: -0.30 to 0.30
- y: -0.30 to 0.30
- z: -1.20 to 0.10

right_leg:
- x: -0.30 to 0.30
- y: -0.30 to 0.30
- z: -1.20 to 0.10

RECENT ACTIONS

{json.dumps(recent_actions)}

OUTPUT FORMAT

Return only one valid JSON object.

Do not include explanations, Markdown or nested objects.

Use exactly this structure:

{{
  "left_arm": [x, y, z],
  "right_arm": [x, y, z],
  "left_leg": [x, y, z],
  "right_leg": [x, y, z]
}}
""".strip()


def llm_worker(
    action_queue,
    controller,
    controller_lock,
    action_history,
    history_lock,
    stop_event,
):
    """
    Generate actions without blocking the simulation thread.
    """
    while not stop_event.is_set():
        # Do not request another action when one is
        # already waiting to be executed.
        if action_queue.full():
            time.sleep(0.05)
            continue

        # MuJoCo data is shared between threads, so read it
        # while holding the controller lock.
        with controller_lock:
            observation = copy.deepcopy(
                controller.get_obs()
            )

        with history_lock:
            history_snapshot = copy.deepcopy(
                action_history
            )

        prompt = build_llm_prompt(
            observation,
            history_snapshot,
        )

        raw_action = get_llm_action(
            prompt
        )

        if raw_action is None:
            time.sleep(0.5)
            continue

        print(
            "Raw LLM action:",
            format_for_print(raw_action),
        )

        if not validate_action(raw_action):
            print(
                "Rejected invalid LLM action."
            )
            time.sleep(0.2)
            continue

        safe_action = sanitize_action(
            raw_action,
            observation["full_state"],
        )

        print(
            "Sanitized action:",
            format_for_print(safe_action),
        )

        try:
            action_queue.put(
                safe_action,
                timeout=0.5,
            )

        except queue.Full:
            pass


# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------

print(
    "Initializing LLM Humanoid Environment..."
)

controller = LLMHumanoidController(
    MODEL_PATH
)

# Ensure body positions and orientations are initialized
# before obtaining the first observation.
mujoco.mj_forward(
    controller.model,
    controller.data,
)

action_queue = queue.Queue(
    maxsize=1
)

controller_lock = threading.Lock()
history_lock = threading.Lock()
stop_event = threading.Event()

action_history = []

with controller_lock:
    initial_observation = copy.deepcopy(
        controller.get_obs()
    )

initial_action = {
    "left_arm": initial_observation[
        "full_state"
    ]["left_arm_pos"],

    "right_arm": initial_observation[
        "full_state"
    ]["right_arm_pos"],

    "left_leg": initial_observation[
        "full_state"
    ]["left_leg_pos"],

    "right_leg": initial_observation[
        "full_state"
    ]["right_leg_pos"],
}

action_history.append(
    copy.deepcopy(initial_action)
)

worker_thread = threading.Thread(
    target=llm_worker,
    args=(
        action_queue,
        controller,
        controller_lock,
        action_history,
        history_lock,
        stop_event,
    ),
    daemon=True,
)

worker_thread.start()

MAX_ACTIONS = 25

# Fixed-duration poses are generally more suitable for
# continuous swimming than waiting for exact convergence.
STEPS_PER_ACTION = 150

VIEWER_SYNC_INTERVAL = 10
RECORD_INTERVAL = 5

executed_actions = 0

try:
    with mujoco.viewer.launch_passive(
        controller.model,
        controller.data,
    ) as viewer:

        while (
            viewer.is_running()
            and executed_actions < MAX_ACTIONS
        ):
            try:
                queued_action = (
                    action_queue.get(
                        timeout=0.1
                    )
                )

            except queue.Empty:
                viewer.sync()
                continue

            if not validate_action(
                queued_action
            ):
                print(
                    "Rejected invalid queued action."
                )
                continue

            # The humanoid may have moved while Ollama was
            # producing its response. Sanitize once more using
            # the newest state.
            with controller_lock:
                current_observation = (
                    copy.deepcopy(
                        controller.get_obs()
                    )
                )

            action = sanitize_action(
                queued_action,
                current_observation[
                    "full_state"
                ],
            )

            executed_actions += 1

            with history_lock:
                action_history.append(
                    copy.deepcopy(action)
                )

            print()
            print("=" * 70)
            print(
                f"Executing LLM action "
                f"{executed_actions}/{MAX_ACTIONS}"
            )
            print(
                format_for_print(action)
            )

            with controller_lock:
                controller.set_targets(
                    action
                )

            target_reached_at_least_once = False

            for _ in range(
                STEPS_PER_ACTION
            ):
                if not viewer.is_running():
                    break

                with controller_lock:
                    targets_reached = (
                        controller.physics_step()
                    )

                    current_step = (
                        controller.step_count
                    )

                if targets_reached:
                    target_reached_at_least_once = True

                # Record only when a recorder exists.
                if (
                    "recorder" in globals()
                    and current_step
                    % RECORD_INTERVAL
                    == 0
                ):
                    with controller_lock:
                        recorder.record_frame()

                if (
                    current_step
                    % VIEWER_SYNC_INTERVAL
                    == 0
                ):
                    viewer.sync()

            with controller_lock:
                observation_after = (
                    copy.deepcopy(
                        controller.get_obs()
                    )
                )

            print(
                "Positions after action:",
                format_for_print(
                    observation_after[
                        "full_state"
                    ]
                ),
            )

            print(
                "Reached strict controller "
                "threshold:",
                target_reached_at_least_once,
            )

            print("=" * 70)

finally:
    stop_event.set()

    with controller_lock:
        controller.data.ctrl[:] = 0.0

    worker_thread.join(
        timeout=2.0
    )


if "recorder" in globals():
    recorder.save_video(
        VIDEO_FILENAME
    )

print(
    "LLM swimming simulation complete."
)