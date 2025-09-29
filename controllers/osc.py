# controllers/osc.py
import numpy as np

class OperationalSpaceController:
    """
    x_des(3,)  ->  tau(n,)
    f = Kp*(x_des - x) + Kd*(xd_des - xd)
    tau = J^T * f  -  Kq * qdot
    """
    def __init__(self, kp=500.0, kd=40.0, kq=1.0, pos_limit=None, ee_workspace=None, gravity_comp=True):
        self.kp = kp
        self.kd = kd
        self.kq = kq
        self.pos_limit = pos_limit           # (low(3,), high(3,))
        self.ee_workspace = ee_workspace     # (low(3,), high(3,)) for x_des safety
        self.gravity_comp = gravity_comp

    def clip3(self, v, bounds):
        if bounds is None:
            return v
        low, high = bounds
        return np.minimum(np.maximum(v, low), high)

    def __call__(self, x, xd, Jx, qdot, x_des, xd_des=None, tau_limit=None, qfrc_bias=None):
        x_des = self.clip3(x_des, self.ee_workspace)
        if xd_des is None:
            xd_des = np.zeros_like(x)
        # 작업공간 PD
        e = x_des - x
        ed = xd_des - xd
        f = self.kp * e + self.kd * ed                    # (3,)
        tau = Jx.T @ f - self.kq * qdot                   # (n,)

        if self.gravity_comp and (qfrc_bias is not None):
            tau = tau + qfrc_bias[:tau.shape[0]]     # 관절 공간 중력/바이어스 보상

        if tau_limit is not None:
            low, high = tau_limit
            tau = np.minimum(np.maximum(tau, low), high)
        return tau, e
