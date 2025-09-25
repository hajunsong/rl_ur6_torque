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
