# envs/ur6_env.py
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco as mj
from utils.mj_utils import get_site_pos_vel, get_pos_jacobian, actuator_tau_limits
from controllers.osc import OperationalSpaceController

class UR6TorqueEEEnv(gym.Env):
    """
    액션: EEF 목표 '증분' dx (m)  -> 내부에서 τ로 변환하여 적용
    관측: [q, qdot, x_ee, x_goal]
    보상: -||x_ee - x_goal||,   done: 거리 < thresh 또는 타임아웃
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self,
                 model_path="assets/ur10e/scene.xml",
                 eef_site="ee_site",
                 frame_skip=10,
                 render_mode=None,
                 dx_limit=0.01,                 # 한 스텝 목표변위 한계 (m)
                 goal_box=((-0.5, -0.5, 0.1), (0.5, 0.5, 0.8)),
                 kp=500.0, kd=40.0, kq=1.0):
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

        # 작업공간 컨트롤러
        self.osc = OperationalSpaceController(
            kp=kp, kd=kd, kq=kq,
            ee_workspace=goal_box
        )

        # 액션: dx in R^3
        self.dx_limit = dx_limit
        act_high = np.array([dx_limit, dx_limit, dx_limit], dtype=np.float32)
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)

        # 관측: q(nv) + qdot(nv) + x(3) + goal(3)
        obs_high = np.inf*np.ones(self.nv + self.nv + 3 + 3, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # 토크 한계
        self.tau_low, self.tau_high = actuator_tau_limits(self.model)

        # 목표 & 내부 상태
        self.reset(seed=None)

        self.render_mode = render_mode
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mj.Renderer(self.model, 640, 480)  # 해상도 자유

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def _sample_goal(self):
        low, high = np.array(self.goal_box[0]), np.array(self.goal_box[1])
        return np.random.uniform(low=low, high=high).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.seed(seed)
        # 초기자세 (Menagerie 기본값 사용; 필요시 data.qpos 세팅)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        mj.mj_forward(self.model, self.data)

        # 현재 EEF 위치/목표 설정
        self.goal_box = ((-0.4, -0.4, 0.2), (0.4, 0.4, 0.8))
        x, _ = get_site_pos_vel(self.model, self.data, self.eef_site)
        self.x_des = x.copy()                         # 시작은 현재 위치
        self.x_goal = self._sample_goal()

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        q = self.data.qpos.copy()
        qd = self.data.qvel.copy()
        x, _ = get_site_pos_vel(self.model, self.data, self.eef_site)
        return np.concatenate([q, qd, x, self.x_goal]).astype(np.float32)

    def step(self, action):
        # 1) 목표 업데이트 (작은 변위)
        dx = np.clip(action, -self.dx_limit, self.dx_limit)
        self.x_des = self.x_des + dx

        # 2) 현재 상태/자코비안
        x, xd = get_site_pos_vel(self.model, self.data, self.eef_site)
        Jx = get_pos_jacobian(self.model, self.data, self.eef_site)  # (3, nv)

        # 3) 토크 계산 (OSC)
        tau, e = self.osc(
            x=x, xd=xd, Jx=Jx, qdot=self.data.qvel.copy(),
            x_des=self.x_des, xd_des=np.zeros(3),
            tau_limit=(self.tau_low, self.tau_high)
        )

        # 4) 시뮬레이션 진행
        self.data.ctrl[:] = tau
        for _ in range(self.frame_skip):
            mj.mj_step(self.model, self.data)

        # 5) 보상/종료
        x_new, _ = get_site_pos_vel(self.model, self.data, self.eef_site)
        dist = np.linalg.norm(x_new - self.x_goal)
        reward = -dist
        terminated = dist < 0.02
        truncated = False

        obs = self._get_obs()
        info = {
            "dist": dist,
            "x_ee": x_new.copy(),        # ★ EEF 좌표를 info로 함께 로깅
            "x_goal": self.x_goal.copy()
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array" or self.renderer is None:
            return
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self):
        self.renderer = None
