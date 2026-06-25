import gym
from gym import spaces
import numpy as np
import psycopg2
import time


class PostgresEnv(gym.Env):
    def __init__(self, telemetry):
        super(PostgresEnv, self).__init__()

        self.telemetry = telemetry

        #  DEFINE spaces FIRST (critical for SB3)
        self.action_space = spaces.Box(
            low=np.array([64, 1, 50], dtype=np.float32),
            high=np.array([1024, 32, 300], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(6,),
            dtype=np.float32
        )

        #  DB connection
        self.conn = psycopg2.connect(
            dbname="tpch",
            user="postgres",
            password="postgres",
            host="127.0.0.1"
        )
        self.conn.autocommit = True

    def reset(self):
        return self.get_obs()

    def step(self, action):
        config = self.action_to_config(action)

        self.apply_config(config)

        time.sleep(2)

        metrics = self.telemetry.get_metrics()

        obs = self.metrics_to_obs(metrics)
        reward = self.calculate_reward(metrics)

        done = False

        return obs, reward, done, {}

    def apply_config(self, config):
        cur = self.conn.cursor()

        for k, v in config.items():
            cur.execute(f"ALTER SYSTEM SET {k} = '{v}'")

        cur.execute("SELECT pg_reload_conf()")

    def action_to_config(self, action):
        return {
            "shared_buffers": f"{int(action[0])}MB",
            "work_mem": f"{int(action[1])}MB",
            "max_connections": f"{int(action[2])}"
        }

    def metrics_to_obs(self, metrics):
        return np.array([
            metrics['cpu'],
            metrics['mem'],
            metrics['latency'],
            metrics['tps'],
            metrics['errors'],
            0
        ], dtype=np.float32)

    def calculate_reward(self, metrics):
        return metrics['tps'] - 0.5 * metrics['latency']

    def get_obs(self):
        metrics = self.telemetry.get_metrics()
        return self.metrics_to_obs(metrics)
