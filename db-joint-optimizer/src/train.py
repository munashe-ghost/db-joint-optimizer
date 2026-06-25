from stable_baselines3 import PPO
from rl_agent.env import PostgresEnv
from telemetry_collector import TelemetryCollector


def train_rl():
    print("Initializing training...")

    telemetry = TelemetryCollector()
    env = PostgresEnv(telemetry)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4
    )

    model.learn(total_timesteps=3000)

    model.save("ppo_postgres")
    print("Training complete.")


if __name__ == "__main__":
    train_rl()
