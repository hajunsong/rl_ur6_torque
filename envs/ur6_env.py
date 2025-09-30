# envs/ur6_env.py (6D extension with quaternion orientation tracking)
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco as mj
from mujoco import viewer

from utils.mj_utils import (
    get_site_pos_vel, get_pos_jacobian, actuator_tau_limits,
    quat_err_vec
)

class UR6TorqueEEEnv(gym.Env):
    """
    Option B (+orientation): DLS-IK (6D) + Joint-Space Computed Torque

    Action: delta-x (EEF position increment in meters), shape (3,)
            -> internally updates desired pose x_des and uses orientation goal q_goal (quaternion)
    Observation: concat([q, qdot, x_ee(3), q_ee(4), x_goal(3), q_goal(4)])
    Reward: -||x_ee - x_goal|| - ori_w*||e_o||
    Termination: distance < success_radius & ori_error < success_ori (held for success_hold_steps) OR step >= max_steps
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        model_path: str,
        eef_site: str = "ee_site",
        frame_skip: int = 5,
        # gains (computed torque)
        kp: float = 500.0,
        kd: float = 100.0,
        kq: float = 1.5,         # optional viscous joint damping
        # task-space action limit
        dx_limit: float = 0.004,
        # goal / workspace box
        goal_box=(( -1.2, -0.8, 0.05), ( 1.2, 0.8, 1.20)),
        # success
        success_radius: float = 0.01,
        success_ori: float = 0.03,   # ||orientation error vec|| threshold (rad)
        success_hold_steps: int = 5,
        hold_at_goal: bool = True,
        # episode
        max_steps: int = 300,
        # rendering
        render_mode: str | None = None,
        # IK parameters
        ik_lambda: float = 2e-3,   # base DLS damping
        ik_step_scale: float = 1.0,
        ik_iters: int = 6,         # multi-iteration IK per step
        # posture regularization (nullspace)
        posture_weight: float = 1e-2,
        # auto tracking toward goal
        auto_track_alpha: float = 0.2,
        # limits & smoothers
        task_vmax: float = 0.6,        # [m/s]
        joint_qdd_max: float = 400.0,  # [rad/s^2]
        joint_qd_soft: float = 2.5,    # [rad/s]
        joint_qd_soft_k: float = 10.0, # soft-limit gain
        tau_slew_rate: float = 800.0,  # [Nm/s]
        # orientation weights in IK (scale for rotational task)
        ori_gain_task: float = 1.0,
        ori_weight_cost: float = 1.0,  # cost weighting vs translation in DLS
        # reward weights
        ori_reward_weight: float = 0.2
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

        # observation space: q(nv) + qdot(nv) + x(3) + q(4) + goal(3) + goal_q(4)
        obs_high = np.inf*np.ones(self.nv + self.nv + 3 + 4 + 3 + 4, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # torque limits
        self.tau_low, self.tau_high = actuator_tau_limits(self.model)

        # success / episode
        self.success_radius = float(success_radius)
        self.success_ori = float(success_ori)
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

        # posture & tracking
        self.posture_weight = float(posture_weight)
        self.auto_track_alpha = float(auto_track_alpha)

        # spike mitigators
        self.task_vmax = float(task_vmax)
        self.joint_qdd_max = float(joint_qdd_max)
        self.joint_qd_soft = float(joint_qd_soft)
        self.joint_qd_soft_k = float(joint_qd_soft_k)
        self.tau_slew_rate = float(tau_slew_rate)

        # orientation scaling
        self.ori_gain_task = float(ori_gain_task)
        self.ori_weight_cost = float(ori_weight_cost)

        # reward
        self.ori_reward_weight = float(ori_reward_weight)

        # internal state
        self._succ_k = 0
        self._t = 0
        self.x_des = None
        self.q_goal = None   # quaternion (w,x,y,z)
        self.x_goal = None
        self.q_home = self.data.qpos.copy()  # posture bias target

        # smoothing
        self._prev_tau = np.zeros(self.nu)

        # renderers
        self._viewer = None
        self._renderer = None
        self._render_width = 960
        self._render_height = 720

    # --------------------------- helpers ---------------------------

    def _sample_goal(self):
        low, high = np.array(self.goal_box[0]), np.array(self.goal_box[1])
        xg = np.random.uniform(low=low, high=high).astype(np.float32)
        # orientation: default keep current orientation
        qg = self._get_site_quat(self.eef_site)
        return xg, qg

    def set_goal_pose(self, x_goal, q_goal):
        x_goal = np.asarray(x_goal, dtype=np.float32)
        q_goal = np.asarray(q_goal, dtype=np.float64)
        low, high = np.array(self.goal_box[0]), np.array(self.goal_box[1])
        self.x_goal = np.minimum(np.maximum(x_goal, low), high)
        # normalize quaternion
        q_goal = q_goal / np.linalg.norm(q_goal)
        self.q_goal = q_goal.astype(np.float64)
        return self.x_goal, self.q_goal

    def set_goal(self, x_goal):
        # backward-compat: position-only setter keeps current quat as goal
        return self.set_goal_pose(x_goal, self._get_site_quat(self.eef_site))

    def _clip_to_workspace(self, x):
        low, high = np.array(self.goal_box[0]), np.array(self.goal_box[1])
        return np.minimum(np.maximum(x, low), high)

    def _get_site_quat(self, site_name):
        sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xquat[sid].copy()

    def _get_angvel(self, site_name):
        # omega = Jr * qdot
        sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, site_name)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mj.mj_jacSite(self.model, self.data, jacp, jacr, sid)
        qdot = self.data.qvel.copy()
        return jacr @ qdot

    def _dt(self):
        return float(self.model.opt.timestep * self.frame_skip)

    @staticmethod
    def _saturate(vec, limit):
        return np.clip(vec, -limit, limit)

    @staticmethod
    def _slew_limit(curr, prev, rate_limit, dt):
        max_delta = rate_limit * dt
        return prev + np.clip(curr - prev, -max_delta, max_delta)

    # --------------------------- gym API ---------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.data = mj.MjData(self.model)
        mj.mj_forward(self.model, self.data)

        if options is not None and "x_goal" in options:
            if "q_goal" in options and options["q_goal"] is not None:
                self.x_goal, self.q_goal = self.set_goal_pose(options["x_goal"], options["q_goal"])
            else:
                self.x_goal, self.q_goal = self.set_goal_pose(options["x_goal"], self._get_site_quat(self.eef_site))
        else:
            self.x_goal, self.q_goal = self._sample_goal()

        x, _ = get_site_pos_vel(self.model, self.data, self.eef_site)
        self.x_des = x.copy()

        self._succ_k = 0
        self._t = 0
        self._prev_tau = np.zeros(self.nu)

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        q = self.data.qpos.copy()
        qd = self.data.qvel.copy()
        x, _ = get_site_pos_vel(self.model, self.data, self.eef_site)
        q_cur = self._get_site_quat(self.eef_site)
        return np.concatenate([q, qd, x, q_cur, self.x_goal, self.q_goal]).astype(np.float32)

    def step(self, action):
        # ------------- read state -------------
        x, xd = get_site_pos_vel(self.model, self.data, self.eef_site)
        q_cur = self._get_site_quat(self.eef_site)
        omega = self._get_angvel(self.eef_site)

        e_p = self.x_goal - x
        e_o = quat_err_vec(self.q_goal, q_cur)   # small-angle vector (rad)
        dist = float(np.linalg.norm(e_p))
        ori_norm = float(np.linalg.norm(e_o))

        # ------------- action integration -------------
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, -self.dx_limit, self.dx_limit)

        # auto-track toward goal
        self.x_des = self._clip_to_workspace(
            (1.0 - self.auto_track_alpha) * self.x_des +
            self.auto_track_alpha * self.x_goal
        )

        # deadband hold
        if self.hold_at_goal and (dist < self.success_radius and ori_norm < self.success_ori):
            self.x_des = self.x_goal.copy()
            dx_cmd = np.zeros(3, dtype=np.float32)
        else:
            dx_cmd = action

        # x_des update with task-space velocity cap
        dt = self._dt()
        max_step = self.task_vmax * dt
        step_vec = dx_cmd
        step_norm = float(np.linalg.norm(step_vec))
        if step_norm > max_step and step_norm > 1e-9:
            step_vec = step_vec * (max_step / step_norm)
        self.x_des = self._clip_to_workspace(self.x_des + step_vec)

        # ------------- Multi-iter 6D DLS IK + posture nullspace -------------
        q = self.data.qpos.copy()
        qd = self.data.qvel.copy()
        nv = self.model.nv
        I = np.eye(nv)

        q_d = q.copy()
        for _ in range(self.ik_iters):
            # measure at current linearization point
            x_now, _ = get_site_pos_vel(self.model, self.data, self.eef_site)
            q_now = self._get_site_quat(self.eef_site)
            e_o_now = quat_err_vec(self.q_goal, q_now)

            # 6D task error (translation + scaled orientation)
            dx_task = np.concatenate([(self.x_des - x_now) * self.ik_step_scale,
                                      self.ori_gain_task * e_o_now])

            # 6D Jacobian
            sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, self.eef_site)
            Jp = np.zeros((3, nv)); Jr = np.zeros((3, nv))
            mj.mj_jacSite(self.model, self.data, Jp, Jr, sid)
            J6 = np.vstack([Jp, Jr])  # (6, nv)

            # Weighted DLS (orientation vs translation)
            W6 = np.diag([1,1,1, self.ori_weight_cost, self.ori_weight_cost, self.ori_weight_cost])
            J6w = W6 @ J6
            dxw  = W6 @ dx_task

            # Adaptive damping via sigma_min
            U, S, Vt = np.linalg.svd(J6w, full_matrices=False)
            sigma_min = S[-1] if S.size > 0 else 0.0
            lam_eff = max(self.ik_lambda, 0.5 * sigma_min)
            JJt = J6w @ J6w.T + (lam_eff**2) * np.eye(6)
            Jplus = J6w.T @ np.linalg.inv(JJt)   # (nv,6)

            # posture nullspace
            N = I - Jplus @ (W6 @ J6)           # use unweighted J in N-projection
            dq = Jplus @ dxw + self.posture_weight * (N @ (self.q_home - q_d))

            q_d[:len(dq)] = q_d[:len(dq)] + dq
            # write q_d for next linearization
            self.data.qpos[:len(q_d)] = q_d
            mj.mj_forward(self.model, self.data)

        # restore real state
        self.data.qpos[:] = q
        mj.mj_forward(self.model, self.data)

        # ------------- Joint-space Computed Torque -------------
        qdd_ref = self.kd * (0.0 - qd) + self.kp * (q_d - q)

        # soft velocity limiting
        if self.joint_qd_soft > 0.0 and self.joint_qd_soft_k > 0.0:
            qd_soft = self.joint_qd_soft
            k_soft = self.joint_qd_soft_k
            damp_extra = k_soft * np.tanh(qd / qd_soft)
            qdd_ref += -damp_extra

        # per-joint acceleration clamp
        qdd_ref = np.clip(qdd_ref, -self.joint_qdd_max, self.joint_qdd_max)

        # inverse dynamics
        M = np.zeros((nv, nv))
        mj.mj_fullM(self.model, M, self.data.qM)
        h = self.data.qfrc_bias.copy()
        tau = M @ qdd_ref + h - self.kq * qd

        # torque limits + slew rate
        tau = np.clip(tau, self.tau_low, self.tau_high)
        tau = self._slew_limit(tau, self._prev_tau, self.tau_slew_rate, dt)
        tau = np.clip(tau, self.tau_low, self.tau_high)
        self._prev_tau = tau.copy()

        # apply and step
        self.data.ctrl[:len(tau)] = tau
        for _ in range(self.frame_skip):
            mj.mj_step(self.model, self.data)

        # outputs
        x_next, xd_next = get_site_pos_vel(self.model, self.data, self.eef_site)
        q_next = self._get_site_quat(self.eef_site)
        e_p = self.x_goal - x_next
        e_o = quat_err_vec(self.q_goal, q_next)
        dist = float(np.linalg.norm(e_p))
        ori_norm = float(np.linalg.norm(e_o))

        reward = -dist - self.ori_reward_weight * ori_norm
        self._t += 1

        # success counting
        if dist < self.success_radius and ori_norm < self.success_ori:
            self._succ_k += 1
        else:
            self._succ_k = 0

        terminated = self._succ_k >= self.success_hold_steps
        truncated = self._t >= self.max_steps

        obs = np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            x_next,
            q_next,
            self.x_goal,
            self.q_goal
        ]).astype(np.float32)

        info = {
            "dist": dist,
            "ori_err": ori_norm,
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
