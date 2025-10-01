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

from utils.mj_utils import get_site_pos_vel, get_pos_jacobian, actuator_tau_limits
from utils.mj_utils import quat_err_vec, get_site_pose_vel

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
        auto_track_alpha: float = 0.2,
        # 회전 오차를 작업오차에 섞을 때의 스케일
        ori_gain_task: float = 1.0,
        # 성공 판정용 각오차 임계 (rad : ~1.7도)
        success_ori: float = 0.03,
        # 자세 오차 가중치
        ori_weight: float = 0.2
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
        # obs_high = np.inf*np.ones(self.nv + self.nv + 3 + 3, dtype=np.float32)
        # self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        obs_high = np.inf * np.ones(self.nv + self.nv + 3 + 4 + 3, dtype=np.float32)
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
        self._render_width = 480    
        self._render_height = 640

        self.ori_gain_task = float(ori_gain_task)
        self.success_ori = float(success_ori)

        # 목표 쿼터니언 상태 저장용
        self.q_goal = None

        self.ori_weight = float(ori_weight)

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
        q_cur = self._get_site_quat(self.eef_site)
        return np.concatenate([q, qd, x, q_cur, self.x_goal, self.q_goal]).astype(np.float32)
    
    def _get_site_quat(self, site_name: str):
        """Return site orientation as quaternion [w, x, y, z]."""
        sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, site_name)
        # site_xmat는 3x3 회전행렬이 9개 원소로 평탄화되어 저장됨
        mat9 = np.array(self.data.site_xmat[sid], dtype=np.float64)
        quat = np.empty(4, dtype=np.float64)
        mj.mju_mat2Quat(quat, mat9)   # MuJoCo util: R(9) -> quat(4)
        return quat

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

        _, _, quat, _, _, _ = get_site_pose_vel(self.model, self.data, self.eef_site)
        if options is not None and "q_goal" in options and options["q_goal"] is not None:
            self.q_goal = options.get("q_goal")
        else:
            self.q_goal = self._get_site_quat(self.eef_site).copy()

        self._succ_k = 0
        self._t = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # ------------- read state -------------
        x, xd, q_cur, omega, Jp, Jr = get_site_pose_vel(self.model, self.data, self.eef_site)
        e_p = self.x_goal - x
        e_o = quat_err_vec(self.q_goal, q_cur)
        dist = float(np.linalg.norm(e_p))
        ori_norm = float(np.linalg.norm(e_o))

        # ------------- action integration -------------
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, -self.dx_limit, self.dx_limit)

        # optional auto-tracking toward goal to keep moving
        self.x_des = self._clip_to_workspace(
            (1.0 - self.auto_track_alpha) * self.x_des +
            self.auto_track_alpha * self.x_goal
        )

        # deadband hold: if close, pin x_des to goal
        if self.hold_at_goal and (dist < self.success_radius and ori_norm < self.success_ori):
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
            # 최신 포즈/자코비안으로 선형화
            x_now, xd_now, q_now, omega_now, Jp, Jr = get_site_pose_vel(self.model, self.data, self.eef_site)
            e_o_now = quat_err_vec(self.q_goal, q_now)

            # 6D 작업오차: [위치; 회전]  (회전은 스케일 ori_gain_task)
            dx_lin = (self.x_des - x_now) * self.ik_step_scale   # (3,)
            dx_ang = self.ori_gain_task * e_o_now                # (3,)
            dx_task = np.concatenate([dx_lin, dx_ang])           # (6,)

            # 6×nv 자코비안
            J6 = np.vstack([Jp, Jr])                             # (6, nv)

            # DLS (필요시 작은 정규화)
            JJt = J6 @ J6.T + (self.ik_lambda**2) * np.eye(6)
            Jplus = J6.T @ np.linalg.inv(JJt)

            # nullspace posture bias
            N = I - Jplus @ J6
            dq = Jplus @ dx_task + self.posture_weight * (N @ (self.q_home - q_d))

            q_d[:len(dq)] += dq
            # 선형화 갱신을 위해 일시 적용
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
        # x_next, xd_next = get_site_pos_vel(self.model, self.data, self.eef_site)
        # dist = float(np.linalg.norm(x_next - self.x_goal))
        # reward = -dist
        x_next, xd_next, q_next, omega_next, _, _ = get_site_pose_vel(self.model, self.data, self.eef_site)
        e_p_next = self.x_goal - x_next
        e_o_next = quat_err_vec(self.q_goal, q_next)
        dist = float(np.linalg.norm(e_p_next))
        ori_norm = float(np.linalg.norm(e_o_next))

        # 보상에 각오차를 살짝 페널티로 줄 수도 있음
        reward = -dist - self.ori_weight * ori_norm
        self._t += 1

        # 성공 판정: 위치 + 자세
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
            self.x_goal
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
            try:
                from mujoco import viewer
            except Exception as e:
                raise RuntimeError(
                    "On-screen viewer unavailable. Use render_mode='rgb_array' and MUJOCO_GL=egl/osmesa."
                ) from e
            if self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

        elif self.render_mode == "rgb_array":
            from mujoco import Renderer
            # 모델의 offscreen framebuffer 한계(기본 640x480)를 읽음
            try:
                offw  = int(getattr(self.model.vis.global_, "offwidth",  640))
                offh  = int(getattr(self.model.vis.global_, "offheight", 480))
            except Exception:
                offw, offh = 640, 480

            # 절대적으로 프레임버퍼를 넘지 않도록 하드 캡
            req_w = min(int(self._render_width),  offw)
            req_h = min(int(self._render_height), offh)

            # 이미 만들어둔 렌더러가 있더라도, 크기가 바뀌면 재생성
            if (self._renderer is None or 
                getattr(self._renderer, "width", None)  != req_w or
                getattr(self._renderer, "height", None) != req_h):
                # 이전 컨텍스트 정리
                if self._renderer is not None:
                    try: self._renderer.close()
                    except: pass
                self._renderer = Renderer(self.model, req_w, req_h)

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