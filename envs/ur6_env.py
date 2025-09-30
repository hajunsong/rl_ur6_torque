# envs/ur6_env.py
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco as mj
from mujoco import viewer
from utils.mj_utils import get_site_pos_vel, get_pos_jacobian, actuator_tau_limits
from utils.mj_utils import get_site_pose_vel, quat_err_vec
from controllers.osc import OperationalSpaceController, OSC6D

class UR6TorqueEEEnv(gym.Env):
    """
    액션: EEF 목표 '증분' dx (m)  -> 내부에서 τ로 변환하여 적용
    관측: [q, qdot, x_ee, x_goal]
    보상: -||x_ee - x_goal||,   done: 거리 < thresh 또는 타임아웃
    """
    # metadata = {"render_modes": ["human", "none"]}
    metadata = {"render_modes" : ["human", "rgb_array", "none"]}

    def __init__(self,
                 model_path="assets/ur10e/scene.xml",
                 eef_site="ee_site",
                 frame_skip=10,
                 render_mode=None,
                 dx_limit=0.03,                 # 한 스텝 목표변위 한계 (m)
                 goal_box=((-0.5, -0.5, 0.1), (0.5, 0.5, 0.8)),
                 kp=500.0, kd=40.0, kq=1.0,
                  # ★ 추가: 성공/유지/길이 제한
                 success_radius=0.001,           # 1 cm
                 success_hold_steps=5,          # 연속 유지 스텝
                 hold_at_goal=True,             # 근접 시 x_des 고정
                 max_steps=500,                # 에피소드 길이 제한
                 ori_control=True,
                 kp_ori=80.0,
                 kd_ori=8.0):
        
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        mj.mj_forward(self.model, self.data)

        self.eef_site = eef_site
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        # 조인트/액추에이터 수
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu

        self.goal_box = goal_box
        self.dx_limit = dx_limit

        # # 작업공간 컨트롤러
        # self.osc = OperationalSpaceController(
        #     kp=kp, kd=kd, kq=kq,
        #     ee_workspace=goal_box
        # )

        # 액션: dx in R^3
        self.dx_limit = dx_limit
        act_high = np.array([dx_limit, dx_limit, dx_limit], dtype=np.float32)
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)

        # 관측: q(nv) + qdot(nv) + x(3) + goal(3)
        obs_high = np.inf*np.ones(self.nv + self.nv + 3 + 3, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # 토크 한계
        self.tau_low, self.tau_high = actuator_tau_limits(self.model)

        self.success_radius = success_radius
        self.success_hold_steps = success_hold_steps
        self.hold_at_goal = hold_at_goal
        self.max_steps = max_steps
        self._succ_k = 0
        self._t = 0

        self._viewer = None
        self._renderer = None
        self._render_width = 960
        self._render_height = 720

        self.ori_control = ori_control
        if self.ori_control:
            self.osc6 = OSC6D(kp_pos=kp, kd_pos=kd, kp_ori=kp_ori, kd_ori=kd_ori,
                              kq=kq, ee_workspace=goal_box, f_max=200.0, m_max=80.0)
        else:
            self.osc = OperationalSpaceController(kp=kp, kd=kd, kq=kq, ee_workspace=goal_box)

        # 목표 & 내부 상태
        self.reset(seed=None)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def _sample_goal(self):
        low, high = np.array(self.goal_box[0]), np.array(self.goal_box[1])
        return np.random.uniform(low=low, high=high).astype(np.float32)
    
    def set_goal(self, x_goal):
        # 외부에서 임의 목표를 지정할 때 사용
        x_goal = np.asarray(x_goal, dtype=np.float32)
        # 워크스페이스로 안전 클리핑
        low, high = np.array(self.goal_box[0]), np.array(self.goal_box[1])
        self.x_goal = np.minimum(np.maximum(x_goal, low), high)
        return self.x_goal

    def reset(self, seed=None, options=None):
        self.seed(seed)
        # 초기자세 (Menagerie 기본값 사용; 필요시 data.qpos 세팅)
        home = np.deg2rad([0, -90, 90, -90, -90, 0])   # UR 시리즈 전형적인 ready pose
        noise = np.deg2rad([2,2,2,2,2,2])
        self.data.qpos[:6] = home + np.random.uniform(-noise, noise)
        self.data.qvel[:] = 0.0
        mj.mj_forward(self.model, self.data)

        # 현재 EEF 위치/목표 설정
        _, _, q_cur, _, _, _ = get_site_pose_vel(self.model, self.data, self.eef_site)
        self.q_des = q_cur.copy()
        self.q_goal = q_cur.copy()  # 필요하면 별도 목표 자세 설정
    
        x, _ = get_site_pos_vel(self.model, self.data, self.eef_site)
        self.x_des = x.copy()

        # ★ 하드코딩 삭제하고, 옵션 우선 → 없으면 샘플링
        if options is not None and "x_goal" in options:
            self.set_goal(options["x_goal"])
        else:
            self.x_goal = self._sample_goal()

        obs = self._get_obs()
        info = {}

        self._succ_k = 0
        self._t = 0

        return obs, info

    def _get_obs(self):
        q = self.data.qpos.copy()
        qd = self.data.qvel.copy()
        x, _ = get_site_pos_vel(self.model, self.data, self.eef_site)
        return np.concatenate([q, qd, x, self.x_goal]).astype(np.float32)

    def step(self, action):
        # 현재 상태
        if self.ori_control:
            x, xd, q_cur, omega, Jp, Jr = get_site_pose_vel(self.model, self.data, self.eef_site)
            dist = np.linalg.norm(x - self.x_goal)
        else:
            x, xd = get_site_pos_vel(self.model, self.data, self.eef_site)
            dist = np.linalg.norm(x - self.x_goal)

        # 근접 데드밴드: 목표 반경 안이면 x_des = goal로 고정
        if self.hold_at_goal and dist < self.success_radius:
            self.x_des = self.x_goal.copy()
            dx = np.zeros(3, dtype=np.float32)
        else:
            # 가변 스텝: 멀 땐 더 크게, 가까울 땐 더 작게 (최소 0.3*dx_limit)
            # local_dx = max(0.3*self.dx_limit, min(self.dx_limit, 0.2*dist))
            dx = np.clip(action, -self.dx_limit, self.dx_limit).astype(np.float32)
            self.x_des = self.x_des + dx

        q_des = self.q_des

        if self.ori_control:
            e_o = quat_err_vec(q_des, q_cur)
            tau, e = self.osc6(
                x=x, xd=xd, q_cur=q_cur, omega=omega, Jp=Jp, Jr=Jr, qdot=self.data.qvel.copy(),
                x_des=self.x_des, q_des=q_des, xd_des=np.zeros(3), omega_des=np.zeros(3),
                tau_limit=(self.tau_low, self.tau_high), qfrc_bias=self.data.qfrc_bias.copy(), e_o=e_o
            )
        else:
            # 자코비안/토크
            Jx = get_pos_jacobian(self.model, self.data, self.eef_site)
            tau, e = self.osc(
                x=x, xd=xd, Jx=Jx, qdot=self.data.qvel.copy(),
                x_des=self.x_des, xd_des=np.zeros(3),
                tau_limit=(self.tau_low, self.tau_high),
                qfrc_bias=self.data.qfrc_bias.copy()   # 중력/바이어스 보상
            )

        # ★ 액추에이터 경로로 토크 적용 (ctrl 사용)
        self.data.ctrl[:len(tau)] = tau
        for _ in range(self.frame_skip):
            mj.mj_step(self.model, self.data)

        # 보상/종료
        x_new, xd_new = get_site_pos_vel(self.model, self.data, self.eef_site)
        dist = np.linalg.norm(x_new - self.x_goal)
        reward = -dist

        # ← 여기서부터 success_radius로 판정
        if dist < self.success_radius:
            self._succ_k += 1
        else:
            self._succ_k = 0
        terminated = (self._succ_k >= self.success_hold_steps)

        self._t += 1
        truncated = (self.max_steps is not None) and (self._t >= self.max_steps)

        obs = self._get_obs()
        info = {"dist": dist,
                "tau_norm": float(np.linalg.norm(tau)),
                "xd_norm": float(np.linalg.norm(xd_new))}
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info


    def render(self):
        if self.render_mode == "human":
            # 인터랙티브 뷰어 창
            if self._viewer is None:
                # 패시브 모드: 우리가 mj_step을 돌리고, 뷰어는 동기화만
                self._viewer = viewer.launch_passive(self.model, self.data)
            # 현재 data 상태를 창에 반영
            self._viewer.sync()
            return None

        elif self.render_mode == "rgb_array":
            # 프레임 버퍼 렌더 → numpy 이미지 반환
            if self._renderer is None:
                self._renderer = mj.Renderer(self.model, self._render_width, self._render_height)
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
