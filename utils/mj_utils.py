# utils/mj_utils.py
import numpy as np
import mujoco as mj

def get_site_pos_vel(model, data, site_name):
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    x = data.site_xpos[sid].copy()               # (3,)
    # 속도: Jx * qdot 사용 (직접 site_xvelp도 가능하지만 Jx*qdot가 일반적)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, jacp, jacr, sid)
    qdot = data.qvel.copy()
    xd = jacp @ qdot                              # (3,)
    return x, xd

def get_pos_jacobian(model, data, site_name):
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, jacp, jacr, sid)   # position/orientation Jacobians
    return jacp

def actuator_tau_limits(model):
    # MuJoCo actuator ctrlrange -> 토크 한계로 사용(토크 액추에이터라면 동일)
    low = model.actuator_ctrlrange[:, 0].copy()
    high = model.actuator_ctrlrange[:, 1].copy()
    return low, high

def get_site_pose_vel(model, data, site_name):
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    x = data.site_xpos[sid].copy()       # (3,)
    # 회전 행렬 -> 쿼터니언
    R = data.site_xmat[sid].reshape(3,3)
    # MuJoCo는 행-주파수 xmat; 필요시 엄밀한 변환 사용
    # 여기선 간단히 mj_mat2Quat 사용
    quat = np.zeros(4)
    mj.mju_mat2Quat(quat, R.flatten())
    # Jacobian
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, jacp, jacr, sid)
    qdot = data.qvel.copy()
    xd = jacp @ qdot                     # 선속도
    omega = jacr @ qdot                  # 각속도
    return x, xd, quat, omega, jacp, jacr

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def quat_conj(q):
    w,x,y,z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)

def quat_err_vec(q_des, q_cur):
    # q_err = q_des * conj(q_cur)
    qe = quat_mul(q_des, quat_conj(q_cur))
    if qe[0] < 0:  # 최단 회전 선택
        qe = -qe
    # 소각도 근사에서 2*vector가 회전 오차
    return 2.0 * qe[1:4]  # (3,)