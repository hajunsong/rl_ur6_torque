# eval_and_video.py
import os
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from envs.ur6_env import UR6TorqueEEEnv

if __name__ == "__main__":
    env = UR6TorqueEEEnv(model_path="assets/ur10e/scene.xml",
                         eef_site="ee_site",
                         render_mode="rgb_array",
                         frame_skip=10)
    env = RecordVideo(env, video_folder="videos",
                      episode_trigger=lambda ep: True,   # 매 에피소드 저장
                      name_prefix="eval")
    model = PPO.load("ur6_ppo")
    for ep in range(5):
        obs, info = env.reset(seed=ep)
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            done = terminated or truncated
        print(f"[Eval] ep={ep}, return={ep_ret:.3f}, dist={info['dist']:.3f}")
    env.close()
