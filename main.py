# main.py
import gymnasium as gym
import numpy as np
from envs.ur6_env import UR6TorqueEEEnv

if __name__ == "__main__":
    env = UR6TorqueEEEnv(
        # model_path="assets/ur10e/scene.xml",
        model_path="/home/keti/Project/12_FieldSensor/rl_ur6_torque/assets/ur10e/scene.xml",
        eef_site="ee_site",
        frame_skip=5,
        kp=600.0, kd=60.0, kq=2.0,
        render_mode="human",
        dx_limit=0.001,
        goal_box=((-1.1, -1.3, -0.3), (1.6, 1.3, 2.8))  # 내 작업공간 설정
    )

    fixed_goal = np.array([-0.85, -0.2, 0.85], dtype=np.float32)
    obs, info = env.reset(seed=0, options={"x_goal": fixed_goal})

    episode = 1
    for step in range(2000):
        nv = env.model.nv
        x = obs[2*nv:2*nv+3]
        x_goal = obs[2*nv+3:2*nv+6]

        dx = np.clip(x_goal - x, -0.001, 0.001).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(dx)
        env.render()

        if (step+1) % 100 == 0:
            print(f"step {step+1}, dist={info['dist']:.3f}, reward={reward:.3f}")
            print("EEF pos =", x, " | goal =", x_goal)

        if terminated or truncated:
            print(f"==> episode {episode} end: dist={info['dist']:.3f}")
            episode += 1
            obs, info = env.reset(options={"x_goal": fixed_goal})

    env.close()
