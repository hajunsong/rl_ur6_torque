# callbacks/tb_callback.py
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class InfoLoggerCallback(BaseCallback):
    """
    env.info의 'dist', 'x_ee', 'x_goal'을 TensorBoard에 기록.
    """
    def __init__(self, log_every=100, verbose=0):
        super().__init__(verbose)
        self.log_every = log_every
        self.step_count = 0
        self.success_win = []

    def _on_training_start(self) -> None:
        # 벡터/단일 환경 케이스 모두 대응
        try:
            self.env0 = self.training_env.envs[0]
        except Exception:
            self.env0 = self.training_env

    def _on_step(self) -> bool:
        self.step_count += 1
        infos = self.locals.get("infos", None)
        if infos is None:
            return True
        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
        if "dist" in info0:
            d = float(info0["dist"])
            self.logger.record("train/dist", d)
            # 성공 윈도우(예: 2cm 이내 도달)
            self.success_win.append(1.0 if d < 0.02 else 0.0)
            if len(self.success_win) > 1000:
                self.success_win.pop(0)
            self.logger.record("train/success_rate_recent", float(np.mean(self.success_win)))
        if "x_ee" in info0 and "x_goal" in info0:
            x = np.asarray(info0["x_ee"], dtype=float)
            g = np.asarray(info0["x_goal"], dtype=float)
            for i, ax in enumerate("xyz"):
                self.logger.record(f"train/x_ee_{ax}", float(x[i]))
                self.logger.record(f"train/x_goal_{ax}", float(g[i]))
        # 주기적으로만 flush (SB3 로거가 알아서 처리하지만 가독성 위해)
        return True
