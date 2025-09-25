# train.py
import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from envs.ur6_env import UR6TorqueEEEnv
from callbacks.tb_callback import InfoLoggerCallback

def make_env(record_video=False):
    render_mode = "rgb_array" if record_video else None
    env = UR6TorqueEEEnv(model_path="assets/ur10e/scene.xml",
                         eef_site="ee_site",
                         frame_skip=10,
                         render_mode=render_mode,
                         kp=600.0, kd=50.0, kq=1.5)
    # SB3 모니터: monitor.csv 저장
    os.makedirs("logs", exist_ok=True)
    env = Monitor(env, filename=os.path.join("logs", "monitor.csv"))
    # Gymnasium 에피소드 통계
    env = RecordEpisodeStatistics(env)
    # 주기적으로 영상 저장 (예: 50에피소드마다)
    if record_video:
        os.makedirs("videos", exist_ok=True)
        env = RecordVideo(env, video_folder="videos",
                          episode_trigger=lambda ep: ep % 50 == 0,
                          name_prefix="train")
    return env

if __name__ == "__main__":
    env = make_env(record_video=True)
    # TensorBoard 로그 디렉토리 지정
    model = PPO("MlpPolicy", env,
                n_steps=2048, batch_size=256, gamma=0.99,
                learning_rate=3e-4, clip_range=0.2,
                tensorboard_log="runs", verbose=1)

    callback = InfoLoggerCallback(log_every=100)
    # 학습
    model.learn(total_timesteps=500_000, callback=callback)
    model.save("ur6_ppo")

    env.close()
