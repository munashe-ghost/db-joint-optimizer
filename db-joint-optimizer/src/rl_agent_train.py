import random
import json
import os

class RLConfigTuner:
    def __init__(self):
        #  Same action space as framework
        self.shared_buffers_options = ['64MB', '128MB', '256MB']
        self.work_mem_options = ['1MB', '4MB', '8MB']
        self.max_connections_options = ['50', '100', '200']

        #  Q-table
        self.q_table = {}

        #  Learning parameters
        self.alpha = 0.1     # learning rate
        self.gamma = 0.9     # discount factor
        self.epsilon = 0.2   # exploration rate

        #  Save file
        self.q_file = "q_table.json"

        self.load_q_table()

    #  Convert config to key
    def _config_to_key(self, config):
        return (
            config["shared_buffers"],
            config["work_mem"],
            config["max_connections"]
        )

    #  Generate random config
    def _random_config(self):
        return {
            "shared_buffers": random.choice(self.shared_buffers_options),
            "work_mem": random.choice(self.work_mem_options),
            "max_connections": random.choice(self.max_connections_options)
        }

    #  Choose best action (or explore)
    def select_action(self):
        if random.random() < self.epsilon:
            return self._random_config()

        # If no knowledge yet → random
        if not self.q_table:
            return self._random_config()

        # Pick best Q-value
        best_key = max(self.q_table, key=self.q_table.get)

        return {
            "shared_buffers": best_key[0],
            "work_mem": best_key[1],
            "max_connections": best_key[2]
        }

    #  Update Q-table
    def update(self, metrics, config, reward):
        key = self._config_to_key(config)

        current_q = self.q_table.get(key, 0)

        # Simple Q-learning update (no next state)
        new_q = current_q + self.alpha * (reward - current_q)

        self.q_table[key] = new_q

        self.save_q_table()

    #  Save Q-table to disk
    def save_q_table(self):
        try:
            with open(self.q_file, "w") as f:
                json.dump(
                    {str(k): v for k, v in self.q_table.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            print(f" Failed to save Q-table: {e}")

    #  Load Q-table
    def load_q_table(self):
        if not os.path.exists(self.q_file):
            return

        try:
            with open(self.q_file, "r") as f:
                data = json.load(f)
                self.q_table = {
                    tuple(k.strip("()").replace("'", "").split(", ")): v
                    for k, v in data.items()
                }
        except Exception as e:
            print(f" Failed to load Q-table: {e}")
