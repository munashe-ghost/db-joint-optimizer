import time
import psycopg2
import random

from telemetry_collector import TelemetryCollector
from models.failure_predictor import FailurePredictor
from rl_agent_train import RLConfigTuner
from failure_injector import FailureInjector


class JointOptimizer:
    def __init__(self):
        #  Failure injection
        self.injector = FailureInjector()

        #  Core components
        self.telemetry = TelemetryCollector()
        self.failure_model = FailurePredictor()
        self.rl_agent = RLConfigTuner()

        #  Config search space (IMPORTANT FIX)
        self.shared_buffers_options = ['64MB', '128MB', '256MB']
        self.work_mem_options = ['1MB', '4MB', '8MB']
        self.max_connections_options = ['50', '100', '200']

        #  Exploration rate (CRITICAL FIX)
        self.epsilon = 0.6  # higher = more exploration initially

        #  Reward weights
        self.alpha = 1.0
        self.beta = 0.5
        self.gamma = 2.0
        self.delta = 3.0

        #  Availability tracking
        self.total_steps = 0
        self.downtime_events = 0

    #  Reward function
    def calculate_reward(self, metrics, failure_risk, downtime):
        return (
            self.alpha * metrics['tps']
            - self.beta * metrics['p95_latency_ms']
            - self.gamma * (failure_risk * 100)
            - self.delta * (downtime * 100)
        )

    #  Downtime detection
    def detect_downtime(self, metrics):
        if metrics['p95_latency_ms'] > 180:
            return 1
        if metrics.get('errors', 0) > 0:
            return 1
        return 0

    #  Random config generator
    def get_random_config(self):
        return {
            "shared_buffers": random.choice(self.shared_buffers_options),
            "work_mem": random.choice(self.work_mem_options),
            "max_connections": random.choice(self.max_connections_options)
        }

    #  Apply configuration to DB
    def apply_config(self, conn, config):
        cur = conn.cursor()

        try:
            for key, value in config.items():
                query = f"ALTER SYSTEM SET {key} = '{value}';"
                cur.execute(query)

            cur.execute("SELECT pg_reload_conf();")

            print(f" Applying configuration: {config}")

        except Exception as e:
            print(f" Config error: {e}")

        finally:
            cur.close()

    #  MAIN LOOP
    def run(self):
        print(" Starting JOINT optimizer...")

        conn = psycopg2.connect(
            dbname="tpch",
            user="postgres",
            password="postgres",
            host="localhost"
        )
        conn.autocommit = True

        rewards = []

        while True:
            self.total_steps += 1
            print("\n Running optimization step...")

            #  1. Collect metrics FIRST
            metrics = self.telemetry.get_metrics()

            #  2. Inject faults based on metrics
            self.injector.inject(metrics)

            #  3. Choose config (epsilon-greedy FIX)
            if random.random() < self.epsilon:
                print(" Random config selected")
                config = self.get_random_config()
            else:
                print(" RL config selected")
                config = self.rl_agent.select_action()

            #  4. Apply config
            self.apply_config(conn, config)

            #  5. Collect metrics again (post-change)
            metrics = self.telemetry.get_metrics()

            #  6. Predict failure risk
            failure_risk = self.failure_model.predict(metrics)

            #  7. Detect downtime
            downtime = self.detect_downtime(metrics)
            self.downtime_events += downtime

            #  8. Compute reward
            reward = self.calculate_reward(metrics, failure_risk, downtime)
            rewards.append(reward)

            # 9. Availability
            availability = 1 - (self.downtime_events / self.total_steps)

            #  10. Train RL agent
            self.rl_agent.update(metrics, config, reward)

            #  Logging
            print(
                f"TPS {metrics['tps']:.2f} | "
                f"Latency {metrics['p95_latency_ms']:.1f} | "
                f"Risk {failure_risk:.3f} | "
                f"Downtime {downtime} | "
                f"Reward {reward:.2f} | "
                f"Availability {availability:.3f}"
            )

            if len(rewards) >= 10:
                avg_reward = sum(rewards[-10:]) / 10
                print(f" Avg Reward (last 10): {avg_reward:.2f}")

            #  OPTIONAL: decay exploration over time
            if self.epsilon > 0.1:
                self.epsilon *= 0.995

            time.sleep(5)


if __name__ == "__main__":
    optimizer = JointOptimizer()
    optimizer.run()
