# controllers/osc.py
import numpy as np

class OperationalSpaceController:
    """
    x_des(3,) -> tau(n,)
    f = Kp*(x_des - x) + Kd*(xd_des - xd)
    tau = J^T f - Kq*qdot (+ qfrc_bias)
    """
    def __init__(self, kp=500.0, kd=40.0, kq=1.0, pos_limit=None, ee_workspace=None, f_max=None):
        self.kp = kp
        self.kd = kd
        self.kq = kq
        self.pos_limit = pos_limit
        self.ee_workspace = ee_workspace
        self.f_max = f_max  # 작업공간 힘 클리핑(선택)

    def clip3(self, v, bounds):
        if bounds is None:
            return v
        low, high = bounds
        return np.minimum(np.maximum(v, low), high)

    def __call__(self, x, xd, Jx, qdot, x_des, xd_des=None, tau_limit=None, qfrc_bias=None):
        x_des = self.clip3(x_des, self.ee_workspace)
        if xd_des is None:
            xd_des = np.zeros_like(x)

        e  = x_des - x
        ed = xd_des - xd
        f  = self.kp * e + self.kd * ed
        if self.f_max is not None:
            f = np.clip(f, -self.f_max, self.f_max)

        tau = Jx.T @ f - self.kq * qdot
        if qfrc_bias is not None:
            tau = tau + qfrc_bias  # 중력/바이어스 보상

        if tau_limit is not None:
            low, high = tau_limit
            tau = np.minimum(np.maximum(tau, low), high)
        return tau.astype(np.float64), e

class OSC6D:
    """
    6D 작업공간 PD:
      f_lin  = Kp_pos*(x_des - x) + Kd_pos*(xd_des - xd)
      f_ang  = Kp_ori*e_o         + Kd_ori*(omega_des - omega)
      tau    = Jp^T f_lin + Jr^T f_ang - Kq*qdot (+ qfrc_bias)
    """
    def __init__(self, kp_pos=600.0, kd_pos=60.0,
                       kp_ori=80.0,  kd_ori=8.0,
                       kq=2.0, ee_workspace=None,
                       f_max=200.0, m_max=80.0):
        self.kp_pos, self.kd_pos = kp_pos, kd_pos
        self.kp_ori, self.kd_ori = kp_ori, kd_ori
        self.kq = kq
        self.ee_workspace = ee_workspace
        self.f_max = f_max  # 선형 힘 [N]
        self.m_max = m_max  # 회전 모멘트 [N·m]

    def _clip3(self, v, box):
        if box is None: return v
        low, high = box
        return np.minimum(np.maximum(v, low), high)

    def __call__(self, x, xd, q_cur, omega, Jp, Jr, qdot,
                 x_des, q_des, xd_des=None, omega_des=None,
                 tau_limit=None, qfrc_bias=None, e_o=None):
        x_des = self._clip3(x_des, self.ee_workspace)
        if xd_des is None:    xd_des = np.zeros(3)
        if omega_des is None: omega_des = np.zeros(3)

        # 위치/자세 오차
        e_p  = x_des - x
        e_v  = xd_des - xd
        if e_o is None:
            raise ValueError("e_o (orientation error) required")
        e_w  = omega_des - omega

        f_lin = self.kp_pos*e_p + self.kd_pos*e_v
        m_ang = self.kp_ori*e_o + self.kd_ori*e_w
        if self.f_max is not None:
            f_lin = np.clip(f_lin, -self.f_max, self.f_max)
        if self.m_max is not None:
            m_ang = np.clip(m_ang, -self.m_max, self.m_max)

        tau = Jp.T @ f_lin + Jr.T @ m_ang - self.kq*qdot
        if qfrc_bias is not None:
            tau = tau + qfrc_bias

        if tau_limit is not None:
            low, high = tau_limit
            tau = np.minimum(np.maximum(tau, low), high)
        return tau.astype(np.float64), e_p