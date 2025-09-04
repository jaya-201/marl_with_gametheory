import os
from sb3_contrib import RecurrentPPO
from src.config import MODEL_DIR, LOG_DIR

class AirlinePPOAgent:
    """
    PPO Agent wrapper for AirlineMarketEnv
    """

    def __init__(self, env, policy="MlpLstmPolicy", name="ppo_airline", load_path=None, verbose=1):
        self.env = env
        self.name = name
        self.verbose = verbose

        if load_path:
            self.model = RecurrentPPO.load(load_path, env=self.env)
            if self.verbose:
                print(f"Loaded model from {load_path}")
        else:
            self.model = RecurrentPPO(
                policy,
                self.env,
                verbose=self.verbose,
                tensorboard_log=LOG_DIR
            )

    def train(self, total_timesteps, checkpoint_freq=50000):
        from stable_baselines3.common.callbacks import CheckpointCallback
        os.makedirs(MODEL_DIR, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=MODEL_DIR,
            name_prefix=self.name
        )

        print(f"Starting training for {self.name}...")
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        print("Training completed.")

    def save(self, filename=None):
        if not filename:
            filename = os.path.join(MODEL_DIR, f"{self.name}_final")
        self.model.save(filename)
        if self.verbose:
            print(f"Model saved to {filename}")

    def predict(self, observation, deterministic=True):
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action