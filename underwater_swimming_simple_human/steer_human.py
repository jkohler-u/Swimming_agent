import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import mujoco
import mujoco.viewer
import numpy as np
import requests
import json
import time
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

        for body_name in self.current_targets:
            self.integral_error[body_name] = np.zeros(3)

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



FIRST_TEST_BLOCK = False

if FIRST_TEST_BLOCK:


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

    LEG_Z = -1.20

    print("Initializing LLM Humanoid Environment...")

    
    controller = LLMHumanoidController(MODEL_PATH)



    # Every action must contain exactly these four limb targets.
    required_action_keys = {
        "left_arm",
        "right_arm",
        "left_leg",
        "right_leg",
    }

    # Conservative initial backstroke-like test sequence.
    #
    # Coordinates are torso-local Cartesian positions:
    #   x: toward sky / floor
    #   y: left / right
    #   z: toward head / feet
    #
    # Consecutive targets are intentionally closer together than in the
    # original sequence to avoid large Cartesian jumps.
    stroke_actions = [
        {
            "left_arm":  [0.10,  0.10, -0.55],
            "right_arm": [0.10, -0.10,  0.45],
            "left_leg":  [0.08,  0.09, LEG_Z],
            "right_leg": [-0.03, -0.09, LEG_Z],
        },
        {
            "left_arm":  [0.20,  0.16, -0.40],
            "right_arm": [0.16, -0.16,  0.35],
            "left_leg":  [0.03,  0.09, LEG_Z],
            "right_leg": [0.03, -0.09, LEG_Z],
        },
        {
            "left_arm":  [0.32,  0.22, -0.20],
            "right_arm": [0.25, -0.22,  0.18],
            "left_leg":  [-0.03,  0.09, LEG_Z],
            "right_leg": [0.08, -0.09, LEG_Z],
        },
        {
            "left_arm":  [0.38,  0.25,  0.00],
            "right_arm": [0.32, -0.25,  0.00],
            "left_leg":  [0.03,  0.09, LEG_Z],
            "right_leg": [0.03, -0.09, LEG_Z],
        },
        {
            "left_arm":  [0.28,  0.20,  0.20],
            "right_arm": [0.38, -0.20, -0.20],
            "left_leg":  [0.08,  0.09, LEG_Z],
            "right_leg": [-0.03, -0.09, LEG_Z],
        },
        {
            "left_arm":  [0.16,  0.14,  0.38],
            "right_arm": [0.28, -0.14, -0.38],
            "left_leg":  [0.03,  0.09, LEG_Z],
            "right_leg": [0.03, -0.09, LEG_Z],
        },
        {
            "left_arm":  [0.10,  0.10,  0.48],
            "right_arm": [0.16, -0.10, -0.52],
            "left_leg":  [-0.03,  0.09, LEG_Z],
            "right_leg": [0.08, -0.09, LEG_Z],
        },
        {
            "left_arm":  [0.10,  0.10,  0.30],
            "right_arm": [0.10, -0.10, -0.55],
            "left_leg":  [0.03,  0.09, LEG_Z],
            "right_leg": [0.03, -0.09, LEG_Z],
        },
        {
            "left_arm":  [0.10,  0.10,  0.05],
            "right_arm": [0.20, -0.16, -0.40],
            "left_leg":  [0.08,  0.09, LEG_Z],
            "right_leg": [-0.03, -0.09, LEG_Z],
        },
        {
            "left_arm":  [0.10,  0.10, -0.25],
            "right_arm": [0.32, -0.22, -0.20],
            "left_leg":  [0.03,  0.09, LEG_Z],
            "right_leg": [0.03, -0.09, LEG_Z],
        },
    ]

    def format_for_print(value):
        """Round nested floating-point values for compact debug output."""
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

    def validate_action(action):
        """Validate the structure and numerical contents of an action."""
        if not isinstance(action, dict):
            return False

        if not required_action_keys.issubset(action.keys()):
            return False

        for key in required_action_keys:
            target = action[key]

            if not isinstance(
                target,
                (list, tuple, np.ndarray),
            ):
                return False

            if len(target) != 3:
                return False

            try:
                target_array = np.asarray(
                    target,
                    dtype=float,
                )
            except (TypeError, ValueError):
                return False

            if target_array.shape != (3,):
                return False

            if not np.all(np.isfinite(target_array)):
                return False

        return True

    # Number of discrete Cartesian poses to execute.
    max_runs = 40

    # Fixed simulation duration for each pose.
    #
    # Do not wait indefinitely for every target to be reached. Swimming
    # should remain continuous, even if some poses retain a small error.
    steps_per_pose = 150

    # Update the viewer every few physics steps rather than every step.
    viewer_sync_interval = 10

    # Record the initial torso position for a rough propulsion measurement.
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

    # Clear residual actuator commands before starting.
    controller.data.ctrl[:] = 0.0

    # Reset integral state once at the beginning, when supported by the
    # controller implementation.
    if hasattr(controller, "reset_integral"):
        controller.reset_integral()

    runs = 0

    with mujoco.viewer.launch_passive(
        controller.model,
        controller.data,
    ) as viewer:

        while viewer.is_running():
            runs += 1

            if runs > max_runs:
                break

            action_index = (
                runs - 1
            ) % len(stroke_actions)

            action = stroke_actions[
                action_index
            ]

            if not validate_action(action):
                print(
                    f"Invalid action at index "
                    f"{action_index}. Skipping."
                )

                controller.data.ctrl[:] = 0.0

                for _ in range(50):
                    controller.physics_step()

                    if (
                        controller.step_count
                        % viewer_sync_interval
                        == 0
                    ):
                        viewer.sync()

                    if not viewer.is_running():
                        break

                continue

            obs_before = controller.get_obs()

            torso_position_before = (
                controller.data.xpos[
                    torso_body_id
                ].copy()
            )

            print()
            print("=" * 70)
            print(
                f"SWIMMING POSE "
                f"{runs}/{max_runs}"
            )
            print(
                f"Stroke action index: "
                f"{action_index}"
            )
            print(
                "Target action:",
                format_for_print(action),
            )
            print(
                "State before:"
            )
            print(
                "  L-ARM:",
                format_for_print(
                    obs_before[
                        "full_state"
                    ][
                        "left_arm_pos"
                    ]
                ),
            )
            print(
                "  R-ARM:",
                format_for_print(
                    obs_before[
                        "full_state"
                    ][
                        "right_arm_pos"
                    ]
                ),
            )
            print(
                "  L-LEG:",
                format_for_print(
                    obs_before[
                        "full_state"
                    ][
                        "left_leg_pos"
                    ]
                ),
            )
            print(
                "  R-LEG:",
                format_for_print(
                    obs_before[
                        "full_state"
                    ][
                        "right_leg_pos"
                    ]
                ),
            )

            # Do not reset integral error for every new trajectory point.
            #
            # Use the first version when set_targets supports the argument:
            try:
                controller.set_targets(
                    action,
                    reset_integral=False,
                )
            except TypeError:
                # Compatibility fallback if the current set_targets method
                # does not yet expose reset_integral.
                controller.set_targets(action)

            completed_steps = 0
            target_reached_at_least_once = False

            for _ in range(steps_per_pose):
                if not viewer.is_running():
                    break

                targets_reached = (
                    controller.physics_step()
                )

                completed_steps += 1

                if targets_reached:
                    target_reached_at_least_once = True

                if (
                    controller.step_count
                    % viewer_sync_interval
                    == 0
                ):
                    viewer.sync()

            if not viewer.is_running():
                break

            # Read a fresh observation after executing the action.
            obs_after = controller.get_obs()

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
                "State after:"
            )
            print(
                "  L-ARM:",
                format_for_print(
                    obs_after[
                        "full_state"
                    ][
                        "left_arm_pos"
                    ]
                ),
            )
            print(
                "  R-ARM:",
                format_for_print(
                    obs_after[
                        "full_state"
                    ][
                        "right_arm_pos"
                    ]
                ),
            )
            print(
                "  L-LEG:",
                format_for_print(
                    obs_after[
                        "full_state"
                    ][
                        "left_leg_pos"
                    ]
                ),
            )
            print(
                "  R-LEG:",
                format_for_print(
                    obs_after[
                        "full_state"
                    ][
                        "right_leg_pos"
                    ]
                ),
            )
            print(
                "Steps executed:",
                completed_steps,
            )
            print(
                "Target reached at least once:",
                target_reached_at_least_once,
            )
            print(
                "Torso displacement this pose:",
                format_for_print(
                    pose_displacement
                ),
            )
            print(
                "Total torso displacement:",
                format_for_print(
                    total_displacement
                ),
            )
            print("=" * 70)

            # Small visualization delay only. Avoid the original one-second
            # freeze between poses.
            time.sleep(0.02)

        # Explicitly clear the controls before leaving the viewer.
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
    print("SWIMMING TEST COMPLETE")
    print(
        "Final torso displacement:",
        format_for_print(
            final_displacement
        ),
    )
    print("=" * 70)
