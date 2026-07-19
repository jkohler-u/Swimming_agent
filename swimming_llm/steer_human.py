import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import mujoco
import mujoco.viewer
import numpy as np
import requests
import json
import time
import math
from scipy.spatial.transform import Rotation as R  # <--- Add this import


MODEL_PATH = 'humanoid_laying.xml'
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

    def set_targets(self, action_json):
        mapping = {
            "left_arm": "hand_left",
            "right_arm": "hand_right",
            "left_leg": "foot_left",
            "right_leg": "foot_right",
        }

        self.current_targets = {
            body_name: np.asarray(action_json[action_name], dtype=float)
            for action_name, body_name in mapping.items()
            if action_name in action_json
            and action_json[action_name] is not None
        }


        self.prev_error = {
            body_name: float("inf")
            for body_name in self.current_targets
        }
    
    def physics_step(self, debug=False) -> bool:

        KP = 70.0
        KD = 20.0

        POSITION_THRESHOLD = 0.03
        VELOCITY_THRESHOLD = 0.15
        MAX_CARTESIAN_FORCE = 70.0

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


            # ---------------------------------
            # 5. Cartesian PD in torso frame
            # ---------------------------------
            #dt = self.model.opt.timestep

            p_force = KP * error_local

            d_force = -KD * velocity_local

            cartesian_force = (
                p_force
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



        mujoco.mj_step(
            self.model,
            self.data,
        )

        self.step_count += 1

        if active_target_count == 0:
            return True

        return reached_count == active_target_count

def get_llm_action(prompt):
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json"}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return json.loads(response.json()['response'])
    except Exception as e:
        print(f"LLM Error: {e}")
        return None
    

def run_single_limb_test(
    controller,
    body_name,
    action_name,
    axis,
    displacement,
    max_control_steps=3000,
    debug_interval=50,
):
    """
    Tests one torso-local displacement for one limb.

    axis:
        0 = local x
        1 = local y
        2 = local z

    displacement:
        movement in metres, for example +0.05 or -0.05
    """

    mujoco.mj_resetData(
        controller.model,
        controller.data,
    )

    if body_name == "hand_left":
        bend_joint_name = "elbow_left"
        bend_angle = np.deg2rad(-25.0)

    elif body_name == "hand_right":
        bend_joint_name = "elbow_right"
        bend_angle = np.deg2rad(-25.0)

    elif body_name == "foot_left":
        bend_joint_name = "knee_left"
        bend_angle = np.deg2rad(-20.0)

    elif body_name == "foot_right":
        bend_joint_name = "knee_right"
        bend_angle = np.deg2rad(-20.0)

    else:
        bend_joint_name = None
        bend_angle = 0.0

    if bend_joint_name is not None:
        bend_joint_id = controller.model.joint(
            bend_joint_name
        ).id

        bend_qpos_id = int(
            controller.model.jnt_qposadr[
                bend_joint_id
            ]
        )

        controller.data.qpos[bend_qpos_id] = bend_angle

    mujoco.mj_forward(
        controller.model,
        controller.data,
    )

    controller.step_count = 0
    controller.current_targets = {}

    initial_position = np.asarray(
        controller.get_local_pos(body_name),
        dtype=np.float64,
    )

    target_position = initial_position.copy()
    target_position[axis] += displacement

    axis_names = ["x", "y", "z"]
    direction = "+" if displacement > 0 else "-"

    test_name = (
        f"{body_name}: "
        f"{direction}{axis_names[axis]} "
        f"{abs(displacement):.3f} m"
    )

    print("\n" + "=" * 70)
    print("TEST:", test_name)
    print(
        "Initial:",
        np.round(initial_position, 4),
    )
    print(
        "Target:",
        np.round(target_position, 4),
    )

    controller.set_targets({
        action_name: target_position.tolist()
    })

    reached = False
    steps_used = 0

    for control_step in range(max_control_steps):
        debug = (
            debug_interval > 0
            and control_step % debug_interval == 0
        )

        reached = controller.physics_step(
            debug=debug
        )

        steps_used = control_step + 1

        if reached:
            break

    final_position = np.asarray(
        controller.get_local_pos(body_name),
        dtype=np.float64,
    )

    final_error = float(
        np.linalg.norm(
            target_position - final_position
        )
    )

    result = {
        "test": test_name,
        "reached": reached,
        "steps": steps_used,
        "initial": initial_position,
        "target": target_position,
        "final": final_position,
        "error": final_error,
    }

    print("\nResult")
    print("Reached:", reached)
    print("Control steps:", steps_used)
    print(
        "Final:",
        np.round(final_position, 4),
    )
    print(
        "Final error:",
        round(final_error, 5),
    )

    return result



RUN_CONTROLLER_TEST = False

if RUN_CONTROLLER_TEST:


    print("Initializing controller...")

    controller = LLMHumanoidController(
        MODEL_PATH
    )

    mujoco.mj_forward(
        controller.model,
        controller.data,
    )

    # Each tuple contains:
    # (axis index, displacement in metres)
    test_movements = [
        (0, +0.05),
        (0, -0.05),
        (1, +0.05),
        (1, -0.05),
        (2, +0.05),
    ]

    test_results = []

    with mujoco.viewer.launch_passive(
        controller.model,
        controller.data,
    ) as viewer:

        for axis, displacement in test_movements:
            if not viewer.is_running():
                break

            result = run_single_limb_test(
                controller=controller,
                body_name="foot_left",
                action_name="left_leg",
                axis=axis,
                displacement=displacement,
                max_control_steps=3000,
                debug_interval=50,
            )

            test_results.append(result)

            viewer.sync()

            # Pause briefly between tests so the reset is visible.
            time.sleep(0.5)

        print("\n" + "=" * 70)
        print("LEFT-FOOT TEST SUMMARY")
        print("=" * 70)

        for result in test_results:
            status = (
                "PASS"
                if result["reached"]
                else "FAIL"
            )

            print(
                f"{status:4} | "
                f"{result['test']:28} | "
                f"steps={result['steps']:4d} | "
                f"error={result['error']:.5f} m"
            )

        # Keep the viewer open after all tests.
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)



SWIMMING_SIMULATION = True


if SWIMMING_SIMULATION:

    print("Initializing LLM Humanoid Environment...")

    controller = LLMHumanoidController(MODEL_PATH)

    # ------------------------------------------------------------------
    # Backstroke configuration
    # ------------------------------------------------------------------
    #
    # Coordinates are torso-local Cartesian positions:
    #   x: toward sky / floor
    #   y: left / right
    #   z: toward head / feet
    #
    # The arms follow a closed elliptical path. The right arm is exactly
    # 180 degrees out of phase with the left arm, producing an alternating
    # backstroke cycle.
    #
    # Positive local x is used for the recovery over the body.
    # Lower local x is used for the underwater pull.
    # ------------------------------------------------------------------

    LEG_Z = -1.20

    # Number of Cartesian targets in one complete arm cycle.
    # Increase this value for smoother motion.
    STROKE_STEPS = 32

    # Number of complete arm cycles to simulate.
    STROKE_CYCLES = 10

    # Number of low-level controller steps spent moving toward each pose.
    MAX_STEPS_PER_POSE = 100

    # Viewer refresh interval in physics steps.
    VIEWER_SYNC_INTERVAL = 10

    # Arm ellipse in the torso-local x-z plane.

    ARM_X_CENTER = 0.04
    ARM_X_RADIUS = 0.3

    ARM_Z_CENTER = -0.03
    ARM_Z_RADIUS = 0.25


    ARM_RECOVERY_Y = 0.18
    ARM_PULL_Y = 0.18

    LEFT_LEG_Y = 0.09
    RIGHT_LEG_Y = -0.09
    KICK_AMPLITUDE = 0.1

    KICK_CYCLES_PER_ARM_CYCLE = 3

    required_action_keys = {
        "left_arm",
        "right_arm",
        "left_leg",
        "right_leg",
    }

    def format_for_print(value):

        if isinstance(value, dict):
            return {
                key: format_for_print(item)
                for key, item in value.items()
            }

        if isinstance(value, np.ndarray):
            return [
                format_for_print(item)
                for item in value.tolist()
            ]

        if isinstance(value, (list, tuple)):
            return [
                format_for_print(item)
                for item in value
            ]

        if isinstance(value, (float, np.floating)):
            return round(float(value), 3)

        return value


    def smoothstep01(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)


    def make_arm_target(phase, side):
        s = math.sin(phase)
        c = math.cos(phase)

        # Smoothly blend between pull and recovery.
        # blend=1 during recovery, blend=0 during pull.
        blend = smoothstep01(0.5 * (s + 1.0))

        x = ARM_X_CENTER + ARM_X_RADIUS * s
        z = ARM_Z_CENTER + ARM_Z_RADIUS * c


        y_distance = (
            (1.0 - blend) * ARM_PULL_Y
            + blend * ARM_RECOVERY_Y
        )

        return [x, side * y_distance, z]

    def make_leg_target(kick_phase, side):
        """Return one alternating flutter-kick foot target."""

        if side > 0.0:
            y = LEFT_LEG_Y
        else:
            y = RIGHT_LEG_Y

        x = KICK_AMPLITUDE * math.sin(kick_phase)

        return [
            x,
            y,
            LEG_Z,
        ]

    def make_backstroke_action(phase):


        left_arm_phase = phase
        right_arm_phase = phase + math.pi

        kick_phase = (
            KICK_CYCLES_PER_ARM_CYCLE
            * phase
        )

        action = {
            "left_arm": make_arm_target(
                left_arm_phase,
                side=1.0,
            ),
            "right_arm": make_arm_target(
                right_arm_phase,
                side=-1.0,
            ),
            "left_leg": make_leg_target(
                kick_phase,
                side=1.0,
            ),
            "right_leg": make_leg_target(
                kick_phase + math.pi,
                side=-1.0,
            ),
        }

        return action

    # Generate one complete closed cycle.

    stroke_actions = [
        make_backstroke_action(
            2.0 * math.pi * step / STROKE_STEPS
        )
        for step in range(STROKE_STEPS)
    ]

    print(f"Generated {len(stroke_actions)} poses for one complete backstroke cycle.")
    torso_body_id = controller.model.body("torso").id

    mujoco.mj_forward(
        controller.model,
        controller.data,
    )

    initial_torso_position = (
        controller.data.xpos[
            torso_body_id
        ].copy()
    )

    # Clear any residual actuator commands.
    controller.data.ctrl[:] = 0.0

    total_poses = (
        STROKE_CYCLES
        * len(stroke_actions)
    )

    pose_number = 0

    with mujoco.viewer.launch_passive(
        controller.model,
        controller.data,
    ) as viewer:

        for cycle_index in range(STROKE_CYCLES):

            if not viewer.is_running():
                break

            print()
            print("=" * 70)
            print(
                f"BACKSTROKE CYCLE "
                f"{cycle_index + 1}/{STROKE_CYCLES}"
            )
            print("=" * 70)

            for action_index, action in enumerate(stroke_actions):

                if not viewer.is_running():
                    break

                pose_number += 1

                torso_position_before = (
                    controller.data.xpos[
                        torso_body_id
                    ].copy()
                )

                controller.set_targets(action)

                target_reached_at_least_once = False
                completed_steps = 0

                targets_reached = False

                for control_step in range(MAX_STEPS_PER_POSE):
                    if not viewer.is_running():
                        break

                    targets_reached = controller.physics_step()
                    completed_steps = control_step + 1

                    if controller.step_count % VIEWER_SYNC_INTERVAL == 0:
                        viewer.sync()

                    if targets_reached:
                        target_reached_at_least_once = True
                        break

                torso_position_after = (
                    controller.data.xpos[
                        torso_body_id
                    ].copy()
                )

                pose_displacement = (
                    torso_position_after
                    - torso_position_before
                )

                total_displacement = (
                    torso_position_after
                    - initial_torso_position
                )

                print(
                    f"Pose {pose_number}/{total_poses} | "
                    f"cycle pose "
                    f"{action_index + 1}/{len(stroke_actions)}"
                )

                print("  Target:",format_for_print(action),)

                print("  Steps:",completed_steps,"| reached:",target_reached_at_least_once,)

                print("  Pose torso displacement:",format_for_print(pose_displacement),)

                print("  Total torso displacement:",format_for_print(total_displacement),
                )

        # Stop all actuator commands before closing.
        controller.data.ctrl[:] = 0.0

    final_torso_position = (
        controller.data.xpos[
            torso_body_id
        ].copy()
    )

    final_displacement = (
        final_torso_position
        - initial_torso_position
    )

    print()
    print("=" * 70)
    print("BACKSTROKE SWIMMING TEST COMPLETE")
    print(
        "Final torso displacement:",
        format_for_print(
            final_displacement
        ),
    )
    print("=" * 70)