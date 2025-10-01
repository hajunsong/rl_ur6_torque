# main.py
import os
os.environ["MUJOCO_GL"] = "egl"  # WSL/헤드리스면 egl 권장

import numpy as np
import imageio
import mujoco as mj
import matplotlib.pyplot as plt

from envs.ur6_env import UR6TorqueEEEnv
from utils.mj_utils import get_site_pose_vel, quat_err_vec

def policy(obs, nu):
    # 데모용: 제로 토크. RL(PPO/SAC) 붙일 때 교체.
    return np.zeros(nu, dtype=np.float32)

def quat_to_euler_zyx(q):
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x+y*y)]
    ], dtype=np.float64)
    yaw   = np.arctan2(R[1,0], R[0,0])  # Z
    pitch = np.arcsin(-R[2,0])          # Y
    roll  = np.arctan2(R[2,1], R[2,2])  # X
    return roll, pitch, yaw

if __name__ == "__main__":
    # 출력 디렉토리 보장
    os.makedirs("video", exist_ok=True)
    os.makedirs("plot", exist_ok=True)

    env = UR6TorqueEEEnv(
        model_path=os.getcwd() + "/assets/ur10e/scene.xml",
        eef_site="ee_site",
        frame_skip=5,
        render_mode="rgb_array",
        # ===== rl_tau 모드 =====
        control_mode="rl_tau",
        gravity_comp=False,   # True로 두면 수렴 쉬움(완전 순수 RL 원하면 False)
        kq=1.0,               # 물리 점성(가상 스프링 아님)
        # ===== 목표/성공 조건 =====
        goal_box=((-1.2, -0.8, 0.05), (1.2, 0.8, 1.2)),
        success_radius=0.005,
        success_hold_steps=10,
        max_steps=300,
        success_ori=0.03,
        ori_weight=0.5,
    )

    # 목표 (원하면 랜덤 샘플도 가능)
    x_goal = np.array([-0.85, -0.20, 0.65], dtype=np.float32)
    qg = np.array([0.7071, 0.0, 0.7071, 0.0], dtype=np.float32)  # yaw ~ 90deg
    q_goal = qg / (np.linalg.norm(qg) + 1e-12)

    obs, info = env.reset(seed=0, options={"x_goal": x_goal, "q_goal": q_goal})

    # 초기 관절자세 (home)
    q_init = np.deg2rad([0, -90, 90, -90, -90, 0])
    env.data.qpos[:6] = q_init
    env.data.qvel[:6] = 0.0
    mj.mj_forward(env.model, env.data)

    # 로그 / 영상
    writer = imageio.get_writer("video/ur10e_run.mp4", fps=30)

    t_log, x_log, xg_log = [], [], []
    rpy_log, rpyg_log = [], []
    dist_log, ori_err_log = [], []

    nu = env.model.nu
    episode = 1
    frame = None

    for step in range(2000):
        # 현 상태
        x, xd, q_cur, omega, Jp, Jr = get_site_pose_vel(env.model, env.data, "ee_site")
        xg = env.x_goal.astype(np.float64)
        qg = env.q_goal.astype(np.float64)

        # RL 액션 (토크 직접)
        a = policy(obs, nu)               # [-1,1]^nu
        obs, reward, terminated, truncated, info = env.step(a)

        # 프레임 기록
        frame = env.render()
        if frame is not None:
            writer.append_data(frame)

        # 로깅
        e_p = xg - x
        e_o = quat_err_vec(qg, q_cur)
        dist = float(np.linalg.norm(e_p))
        ori  = float(np.linalg.norm(e_o))

        r_cur = quat_to_euler_zyx(q_cur)
        r_goal = quat_to_euler_zyx(qg)

        t_log.append(step)
        x_log.append(x.copy())
        xg_log.append(xg.copy())
        rpy_log.append(np.array(r_cur, dtype=np.float64))
        rpyg_log.append(np.array(r_goal, dtype=np.float64))
        dist_log.append(dist)
        ori_err_log.append(ori)

        # 진행 출력
        print(f"step {step+1:4d}, dist={info['dist']:.5f}, ori={info['ori_err']:.5f}, "
              f"|tau|={info['tau_norm']:.2f}, |xd|={info['xd_norm']:.3f}")

        if terminated or truncated:
            print(f"==> episode {episode} end: dist={info['dist']:.3f}, ori={info['ori_err']:.3f}")
            episode += 1
            break

    # === 플롯 ===
    import numpy as np
    t = np.asarray(t_log)
    x_arr   = np.asarray(x_log)
    xg_arr  = np.asarray(xg_log)
    rpy_arr = np.asarray(rpy_log)
    rpyg_arr= np.asarray(rpyg_log)
    dist_arr= np.asarray(dist_log)
    ori_arr = np.asarray(ori_err_log)
    rpy_deg   = np.degrees(rpy_arr)
    rpyg_deg  = np.degrees(rpyg_arr)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.title("EEF Position Tracking")
    plt.plot(t, x_arr[:,0], label="x");    plt.plot(t, x_arr[:,1], label="y");    plt.plot(t, x_arr[:,2], label="z")
    plt.plot(t, xg_arr[:,0], "--", label="x_goal"); plt.plot(t, xg_arr[:,1], "--", label="y_goal"); plt.plot(t, xg_arr[:,2], "--", label="z_goal")
    plt.xlabel("step"); plt.ylabel("position [m]"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig("plot/pos_tracking.png", dpi=150)

    plt.figure()
    plt.title("EEF Orientation Tracking (RPY deg)")
    plt.plot(t, rpy_deg[:,0], label="roll");  plt.plot(t, rpy_deg[:,1], label="pitch");  plt.plot(t, rpy_deg[:,2], label="yaw")
    plt.plot(t, rpyg_deg[:,0], "--", label="roll_goal"); plt.plot(t, rpyg_deg[:,1], "--", label="pitch_goal"); plt.plot(t, rpyg_deg[:,2], "--", label="yaw_goal")
    plt.xlabel("step"); plt.ylabel("angle [deg]"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig("plot/ori_tracking.png", dpi=150)

    plt.figure()
    plt.title("Errors")
    plt.plot(t, dist_arr, label="‖x - x_goal‖ [m]"); plt.plot(t, ori_arr, label="‖e_o‖ [rad]")
    plt.xlabel("step"); plt.ylabel("error"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig("plot/errors.png", dpi=150)

    plt.show()

    writer.close()
    env.close()
