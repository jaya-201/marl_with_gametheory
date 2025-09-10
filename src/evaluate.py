import numpy as np
import os
from src.airline_market import AirlineMarketEnv
from src.ppo_agent import AirlinePPOAgent
from src.config import AGENTS, MODEL_DIR

# Supersuit + PettingZoo
from pettingzoo.utils.conversions import parallel_to_aec, aec_to_parallel
import supersuit as ss


def make_eval_env():
    env = AirlineMarketEnv(airlines=AGENTS)
    env = parallel_to_aec(env)
    env = aec_to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    return env


def evaluate(model_path, episodes=5):
    env = make_eval_env()
    agent = AirlinePPOAgent(env, load_path=model_path, verbose=0)

    all_rewards = []
    for ep in range(episodes):
        obs = env.reset()
        done = [False]
        total_reward = 0.0

        while not done[0]:
            action, _ = agent.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]

        all_rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward}")

    avg_reward = np.mean(all_rewards)
    print(f"\nAverage reward over {episodes} episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    model_file = os.path.join(MODEL_DIR, "ppo_airline_final.zip")
    evaluate(model_file, episodes=5)