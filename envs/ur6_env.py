# Update the patched env with improvements to address plateauing and speed up convergence:
# - Larger, symmetric workspace (supports negative x) and higher z
# - Multi-iteration DLS IK per step (ik_iters)
# - Nullspace/posture bias to avoid kinematic dead-ends
# - Optional auto goal tracking (alpha) so it keeps moving toward x_goal even with small/no actions
# - Slightly higher default gains
# - Keep API compatible; defaults chosen for stability

# envs/ur6_env.py
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco as mj
from mujoco import viewer

from utils.mj_utils import get_site_pos_vel, get_pos_jacobian, actuator_tau_limits

class UR6TorqueEEEnv(gym.Env):
    """
    Option B: IK + Joint-Space Computed Torque

    Action: delta-x (EEF position increment in meters), shape (3,)
            -> internally converts to torque via DLS-IK + computed torque
    Observation: concat([q, qdot, x_ee, x_goal])
    Reward: -||x_ee - x_goal||
    Termination: distance < success_radius (held for success_hold_steps) OR step >= max_steps
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        model_path: str,
        eef_site: str = "ee_site",
        frame_skip: int = 5,
        # gains (computed torque)
        kp: float = 400.0,
        kd: float = 80.0,
        kq: float = 1.0,         # optional viscous joint damping
        # task-space action limit
        dx_limit: float = 0.004,
        # goal / workspace box (expanded & symmetric to allow negative X and higher Z)
        goal_box=(( -1.2, -0.8, 0.05), ( 1.2, 0.8, 1.20)),
        # success
        success_radius: float = 0.01,
        success_hold_steps: int = 5,
        hold_at_goal: bool = True,
        # episode
        max_steps: int = 300,
        # rendering
        render_mode: str | None = None,
        # IK parameters
        ik_lambda: float = 2e-3,   # DLS damping
        ik_step_scale: float = 1.0,
        ik_iters: int = 5,         # multi-iteration IK per step
        # posture regularization (nullspace)
        posture_weight: float = 1e-2,
        # auto tracking toward goal (helps when policy actions are small)
        auto_track_alpha: float = 0.2
    ):
        super().__init__()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo XML not found: {model_path}")

        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        mj.mj_forward(self.model, self.data)

        self.eef_site = eef_site
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        # counts
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu

        # workspace / goal
        self.goal_box = goal_box
        self.dx_limit = float(dx_limit)

        # action space: delta-x in R^3
        act_high = np.array([self.dx_limit, self.dx_limit, self.dx_limit], dtype=np.float32)
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)

        # observation space: q(nv) + qdot(nv) + x(3) + goal(3)
        obs_high = np.inf*np.ones(self.nv + self.nv + 3 + 3, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # torque limits from actuators
        self.tau_low, self.tau_high = actuator_tau_limits(self.model)

        # success / episode
        self.success_radius = float(success_radius)
        self.success_hold_steps = int(success_hold_steps)
        self.hold_at_goal = bool(hold_at_goal)
        self.max_steps = int(max_steps)

        # controller gains
        self.kp = float(kp)
        self.kd = float(kd)
        self.kq = float(kq)

        # IK params
        self.ik_lambda = float(ik_lambda)
        self.ik_step_scale = float(ik_step_scale)
        self.ik_iters = int(ik_iters)

        # posture (nullspace) & auto tracking
        self.posture_weight = float(posture_weight)
        self.auto_track_alpha = float(auto_track_alpha)

        # internal state
        self._succ_k = 0
        self._t = 0
        self.x_des = None
        self.x_goal = None
        self.q_home = self.data.qpos.copy()  # posture bias target

        # renderers
        self._viewer = None
        self._renderer = None
        self._render_width = 960
        self._render_height = 720

    # --------------------------- helpers ---------------------------

    def _sample_goal(self):
        low, high = np.array(self.goal_box[0]), np.array(self.goal_box[1])
        return np.random.uniform(low=low, high=high).astype(np.float32)

    def set_goal(self, x_goal):
        x_goal = np.asarray(x_goal, dtype=np.float32)
        low, high = np.array(self.goal_box[0]), np.array(self.goal_box[1])
        self.x_goal = np.minimum(np.maximum(x_goal, low), high)
        return self.x_goal

    def _clip_to_workspace(self, x):
        low, high = np.array(self.goal_box[0]), np.array(self.goal_box[1])
        return np.minimum(np.maximum(x, low), high)

    def _get_obs(self):
        q = self.data.qpos.copy()
        qd = self.data.qvel.copy()
        x, _ = get_site_pos_vel(self.model, self.data, self.eef_site)
        return np.concatenate([q, qd, x, self.x_goal]).astype(np.float32)

    # --------------------------- gym API ---------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # reset state with mj_resetData
        self.data = mj.MjData(self.model)
        mj.mj_forward(self.model, self.data)

        # initialize goal and desired ee position
        if options is not None and "x_goal" in options and options["x_goal"] is not None:
            self.x_goal = self.set_goal(options["x_goal"])
        else:
            self.x_goal = self._sample_goal()

        x, _ = get_site_pos_vel(self.model, self.data, self.eef_site)
        self.x_des = x.copy()

        self._succ_k = 0
        self._t = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # ------------- read state -------------
        x, xd = get_site_pos_vel(self.model, self.data, self.eef_site)
        dist = float(np.linalg.norm(x - self.x_goal))

        # ------------- action integration -------------
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, -self.dx_limit, self.dx_limit)

        # optional auto-tracking toward goal to keep moving
        self.x_des = self._clip_to_workspace(
            (1.0 - self.auto_track_alpha) * self.x_des +
            self.auto_track_alpha * self.x_goal
        )

        # deadband hold: if close, pin x_des to goal
        if self.hold_at_goal and dist < self.success_radius:
            self.x_des = self.x_goal.copy()
            dx_cmd = np.zeros(3, dtype=np.float32)
        else:
            dx_cmd = action

        # x_des update & workspace clip
        self.x_des = self._clip_to_workspace(self.x_des + dx_cmd)

        # ------------- Multi-iter DLS IK + posture nullspace -------------
        q = self.data.qpos.copy()
        qd = self.data.qvel.copy()
        nv = self.model.nv
        I = np.eye(nv)

        q_d = q.copy()
        for _ in range(self.ik_iters):
            # forward (fresh pose)
            x_now, _ = get_site_pos_vel(self.model, self.data, self.eef_site)
            dx_task = (self.x_des - x_now) * self.ik_step_scale

            Jx = get_pos_jacobian(self.model, self.data, self.eef_site)  # (3, nv)
            JJt = Jx @ Jx.T + (self.ik_lambda**2) * np.eye(3)
            invJJt = np.linalg.inv(JJt)
            Jplus = Jx.T @ invJJt                                       # (nv,3)

            # posture/nullspace bias
            N = I - Jplus @ Jx
            dq = Jplus @ dx_task + self.posture_weight * (N @ (self.q_home - q_d))

            q_d[:len(dq)] = q_d[:len(dq)] + dq
            # temporarily write q_d to data for better linearization in next IK iter
            self.data.qpos[:len(q_d)] = q_d
            mj.mj_forward(self.model, self.data)

        # restore data with updated q_d applied by dynamics step later
        self.data.qpos[:] = q
        mj.mj_forward(self.model, self.data)

        # ------------- Joint-space Computed Torque -------------
        # desired acceleration
        qdd_ref = self.kd * (0.0 - qd) + self.kp * (q_d - q)

        # inverse dynamics: tau = M qdd_ref + h - kq*qd
        M = np.zeros((nv, nv))
        mj.mj_fullM(self.model, M, self.data.qM)
        h = self.data.qfrc_bias.copy()
        tau = M @ qdd_ref + h - self.kq * qd

        # torque limits
        tau = np.clip(tau, self.tau_low, self.tau_high)

        # ------------- apply and step -------------
        self.data.ctrl[:len(tau)] = tau
        for _ in range(self.frame_skip):
            mj.mj_step(self.model, self.data)

        # ------------- outputs -------------
        x_next, xd_next = get_site_pos_vel(self.model, self.data, self.eef_site)
        dist = float(np.linalg.norm(x_next - self.x_goal))
        reward = -dist
        self._t += 1

        # success counting
        if dist < self.success_radius:
            self._succ_k += 1
        else:
            self._succ_k = 0

        terminated = self._succ_k >= self.success_hold_steps
        truncated = self._t >= self.max_steps

        obs = np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            x_next,
            self.x_goal
        ]).astype(np.float32)

        info = {
            "dist": dist,
            "tau_norm": float(np.linalg.norm(tau)),
            "xd_norm": float(np.linalg.norm(xd_next)),
        }

        return obs, reward, terminated, truncated, info

    # --------------------------- rendering ---------------------------

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        elif self.render_mode == "rgb_array":
            from mujoco import mjrender
            if self._renderer is None:
                self._renderer = mjrender.Renderer(self.model, self._render_width, self._render_height)
            self._renderer.update_scene(self.data)
            rgb = self._renderer.render()
            return rgb
        else:
            return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None