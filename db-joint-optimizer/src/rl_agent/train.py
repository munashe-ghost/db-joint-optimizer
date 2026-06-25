from stable_baselines3 import PPO
import numpy as np


class RLConfigTuner:
    def __init__(self):
        self.model = PPO.load("ppo_postgres")

    def select_action(self, metrics, failure_risk):
        obs = np.array([
            metrics['cpu'],
            metrics['mem'],
            metrics['p95_latency_ms'],
            metrics['tps'],
            metrics['errors'],
            failure_risk
        ], dtype=np.float32)

        action, _ = self.model.predict(obs)

        return {
            "shared_buffers": f"{int(action[0])}MB",
            "work_mem": f"{int(action[1])}MB",
            "max_connections": str(int(action[2]))
        }

    def update(self, *args):
        pass
