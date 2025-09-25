# main.py
import gymnasium as gym
import numpy as np
from envs.ur6_env import UR6TorqueEEEnv

if __name__ == "__main__":
    env = UR6TorqueEEEnv(
        # model_path="assets/ur10e/scene.xml",
        model_path="/home/keti/rl_ur6_torque/assets/ur10e/scene.xml",
        eef_site="ee_site",
        frame_skip=10,
        kp=600.0, kd=50.0, kq=1.5
    )

    obs, info = env.reset(seed=0)

    for step in range(2000):
        # 간단한 heuristic: 목표점 쪽으로 EEF 이동
        qposdim = env.model.nq
        x = obs[2*qposdim:2*qposdim+3]
        x_goal = obs[2*qposdim+3:2*qposdim+6]
        dx = np.clip(x_goal - x, -0.01, 0.01).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(dx)
        if (step+1) % 100 == 0:
            print(f"step {step+1}, dist={info['dist']:.3f}, reward={reward:.3f}")
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
