# main.py
import os
os.environ["MUJOCO_GL"] = "egl"     # GPU (NVIDIA) headless
# os.environ["MUJOCO_GL"] = "osmesa"  # CPU-only fallback if you have OSMesa installed

import gymnasium as gym
import numpy as np
from envs.ur6_env import UR6TorqueEEEnv
import imageio
import mujoco as mj

if __name__ == "__main__":
    env = UR6TorqueEEEnv(
        # model_path="assets/ur10e/scene.xml",
        model_path="/home/keti/rl_ur6_torque/assets/ur10e/scene.xml",
        eef_site="ee_site",
        frame_skip=5,
        kp=300.0, kd=60.0, kq=1.0,
        render_mode="rgb_array",
        dx_limit=0.004,
        goal_box=((-1.2, -0.8, 0.05), (1.2, 0.8, 1.2)),  # 내 작업공간 설정
        success_radius=0.002,
        success_hold_steps=1,       # 5스텝 연속 유지 후 성공
        hold_at_goal=False,
        max_steps=300,
        ik_lambda = 5e-2,
        ik_iters = 2,
        auto_track_alpha = 0.1,
        posture_weight = 5e-2
    )

    fixed_goal = np.array([-0.65, -0.2, 0.85], dtype=np.float32)
    obs, info = env.reset(seed=0, options={"x_goal": fixed_goal})
    q_init = np.deg2rad([0, -90, 90, -90, -90, 0])  # 예: UR10e home 자세
    env.data.qpos[:6] = q_init
    env.data.qvel[:6] =  np.zeros(6)
    mj.mj_forward(env.model, env.data)

    writer = imageio.get_writer("ur10e_run.mp4", fps=30)

    for step in range(2000):
        nv = env.model.nv
        x = obs[2*nv:2*nv+3]
        x_goal = obs[2*nv+3:2*nv+6]
        dist = np.linalg.norm(x_goal - x)

        dx = np.clip(x_goal - x, -env.dx_limit, env.dx_limit).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(dx)
        # env.render()
        frame = env.render()
        if frame is not None:
            writer.append_data(frame)

        # if (step+1) % 100 == 0:
        print(f"step {step+1}, dist={info['dist']:.5f}, |tau|={info['tau_norm']:.2f}, |xd|={info['xd_norm']:.3f}")
        print("EEF pos =", x, " | goal =", x_goal)

        if terminated or truncated:
            print(f"==> episode {episode} end: dist={info['dist']:.3f}")
            episode += 1
            obs, info = env.reset(options={"x_goal": fixed_goal})
            break

    writer.close()
    env.close()
