# main.py
import os
os.environ["MUJOCO_GL"] = "egl"     # GPU (NVIDIA) headless
# os.environ["MUJOCO_GL"] = "osmesa"  # CPU-only fallback if you have OSMesa installed

import gymnasium as gym
import numpy as np
from envs.ur6_env import UR6TorqueEEEnv
import imageio
import mujoco as mj
from utils.mj_utils import get_site_pose_vel

import matplotlib.pyplot as plt
from utils.mj_utils import get_site_pose_vel, quat_err_vec

def quat_to_euler_zyx(q):
    # q = [w, x, y, z]
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    yaw   = np.arctan2(R[1,0], R[0,0])    # Z
    pitch = np.arcsin(-R[2,0])            # Y
    roll  = np.arctan2(R[2,1], R[2,2])    # X
    return roll, pitch, yaw

if __name__ == "__main__":
    env = UR6TorqueEEEnv(
        model_path="/home/keti/rl_ur6_torque/assets/ur10e/scene.xml",
        eef_site="ee_site",
        frame_skip=5,
        kp=300.0, kd=60.0, kq=1.0,
        render_mode="rgb_array",
        dx_limit=0.004,
        goal_box=((-1.2, -0.8, 0.05), (1.2, 0.8, 1.2)),  # 내 작업공간 설정
        success_radius=0.002,
        success_hold_steps=10,       # 5스텝 연속 유지 후 성공
        hold_at_goal=False,
        max_steps=300,
        ik_lambda = 5e-2,
        ik_iters = 5,
        auto_track_alpha = 0.1,
        posture_weight = 5e-2,
        success_ori = 0.03,
        ori_weight = 0.5,
    )

    x_goal = np.array([-0.85, -0.2, 0.65], dtype=np.float32)
    qg = np.array([-0.5, 0.5, 0.5, -0.5], dtype=np.float32)
    q_goal = qg / (np.linalg.norm(qg) + 1e-12)
    obs, info = env.reset(seed=0, options={"x_goal": x_goal, "q_goal": q_goal})
    q_init = np.deg2rad([0, -90, 90, -90, -90, 0])  # 예: UR10e home 자세
    env.data.qpos[:6] = q_init
    env.data.qvel[:6] =  np.zeros(6)
    mj.mj_forward(env.model, env.data)

    x, xd, quat, omega, Jp, Jr = get_site_pose_vel(env.model, env.data, "ee_site")
    print("EEF position =", x)
    print("EEF orientation (quat) =", quat)

    writer = imageio.get_writer("ur10e_run.mp4", fps=30)

    # time/step
    t_log = []
    # position (current/goal)
    x_log = [] # (N,3)
    xg_log = [] # (N,3)
    # orientation (current/goal) [rad]
    rpy_log = [] # (N,3) roll, pitch, yaw
    rpyg_log = [] # (N,3)
    # scalar index
    dist_log     = []  # ||x - x_goal||
    ori_err_log  = []  # ||quat_err_vec(q_goal, q_cur)||
    tau_log      = []  # 선택: 토크 노름 기록 (info["tau_norm"]) 있으면

    episode = 1
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

        # 현재 포즈/자코비안 등
        x, xd, q_cur, omega, Jp, Jr = get_site_pose_vel(env.model, env.data, "ee_site")

        # 목표
        x_goal = env.x_goal.astype(np.float64)
        q_goal = env.q_goal.astype(np.float64)

        # 오차
        e_p = x_goal - x
        e_o = quat_err_vec(q_goal, q_cur)           # (3,) 소각벡터
        dist = float(np.linalg.norm(e_p))
        ori  = float(np.linalg.norm(e_o))

        # RPY(라디안)
        r_cur = quat_to_euler_zyx(q_cur)            # tuple of 3
        r_goal = quat_to_euler_zyx(q_goal)

        # 로그 적재
        step_idx = len(t_log)
        t_log.append(step_idx)
        x_log.append(x.copy())
        xg_log.append(x_goal.copy())
        rpy_log.append(np.array(r_cur, dtype=np.float64))
        rpyg_log.append(np.array(r_goal, dtype=np.float64))
        dist_log.append(dist)
        ori_err_log.append(ori)

        # if (step+1) % 100 == 0:
        print(f"step {step+1}, dist={info['dist']:.5f}, |tau|={info['tau_norm']:.2f}, |xd|={info['xd_norm']:.3f}")
        print("EEF pos =", x, " | goal =", x_goal)

        if terminated or truncated:
            print(f"==> episode {episode} end: dist={info['dist']:.3f}")
            episode += 1
            obs, info = env.reset(options={"x_goal": x_goal, "q_goal": q_goal})
            break

    # 배열화
    t = np.asarray(t_log)
    x_arr   = np.asarray(x_log)     # (N,3)
    xg_arr  = np.asarray(xg_log)    # (N,3)
    rpy_arr = np.asarray(rpy_log)   # (N,3) [rad]
    rpyg_arr= np.asarray(rpyg_log)  # (N,3) [rad]
    dist_arr = np.asarray(dist_log)
    ori_arr  = np.asarray(ori_err_log)
    # deg로 보려면:
    rpy_deg   = np.degrees(rpy_arr)
    rpyg_deg  = np.degrees(rpyg_arr)

    # === Figure 1: 위치 추적 (x,y,z) ===
    plt.figure()
    plt.title("EEF Position Tracking")
    plt.plot(t, x_arr[:,0], label="x")
    plt.plot(t, x_arr[:,1], label="y")
    plt.plot(t, x_arr[:,2], label="z")
    plt.plot(t, xg_arr[:,0], "--", label="x_goal")
    plt.plot(t, xg_arr[:,1], "--", label="y_goal")
    plt.plot(t, xg_arr[:,2], "--", label="z_goal")
    plt.xlabel("step"); plt.ylabel("position [m]"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("pos_tracking.png", dpi=150)

    # === Figure 2: 자세 추적 (RPY in deg) ===
    plt.figure()
    plt.title("EEF Orientation Tracking (RPY deg)")
    plt.plot(t, rpy_deg[:,0], label="roll")
    plt.plot(t, rpy_deg[:,1], label="pitch")
    plt.plot(t, rpy_deg[:,2], label="yaw")
    plt.plot(t, rpyg_deg[:,0], "--", label="roll_goal")
    plt.plot(t, rpyg_deg[:,1], "--", label="pitch_goal")
    plt.plot(t, rpyg_deg[:,2], "--", label="yaw_goal")
    plt.xlabel("step"); plt.ylabel("angle [deg]"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("ori_tracking.png", dpi=150)

    # === Figure 3: 스칼라 지표 (거리/각오차) ===
    plt.figure()
    plt.title("Errors")
    plt.plot(t, dist_arr, label="‖x - x_goal‖ [m]")
    plt.plot(t, ori_arr, label="‖e_o‖ [rad]")
    plt.xlabel("step"); plt.ylabel("error"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("errors.png", dpi=150)

    # (선택) 바로 화면에 보고 싶으면:
    plt.show()

    writer.close()
    env.close()
