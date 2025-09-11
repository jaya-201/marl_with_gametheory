import os
import numpy as np
from src.airline_market import AirlineMarketEnv
from src.ppo_agent import AirlinePPOAgent
from src.config import AGENTS, MODEL_DIR, TOTAL_TIMESTEPS

from pettingzoo.utils.conversions import parallel_to_aec, aec_to_parallel
import supersuit as ss

def make_env():
    env = AirlineMarketEnv(airlines=AGENTS)
    env = parallel_to_aec(env)
    env = aec_to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    return env

#runs a test tournament with your trained PPO pricing agents.
def evaluate(model_path, n_episodes=5):
    env = make_env()
    agent = AirlinePPOAgent(env, load_path=model_path, verbose=0)

    all_rewards = {airline: [] for airline in AGENTS}

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_rewards = {airline: 0.0 for airline in AGENTS}

        while not done:
            action, _ = agent.model.predict(obs, deterministic=True) #PPO chooses pricing action
            obs, rewards, dones, infos = env.step(action) #next day of ticket sales

            # rewards is a numpy array here because of vectorization we need to map it back to airlines
            if isinstance(rewards, np.ndarray):
                for i, airline in enumerate(AGENTS):
                    ep_rewards[airline] += rewards[i]
            else:
                for airline, r in rewards.items():
                    ep_rewards[airline] += r

            done = np.all(dones)

        # Save episode results
        for airline in AGENTS:
            all_rewards[airline].append(ep_rewards[airline])

        print(f"Episode {ep+1}: " +
              ", ".join([f"{airline} = {ep_rewards[airline]:.2f}" for airline in AGENTS]))

    print("\nAverage Rewards over episodes")
    for airline in AGENTS:
        avg = np.mean(all_rewards[airline])
        print(f"{airline}: {avg:.2f}")


if __name__ == "__main__":
    model_path = os.path.join(MODEL_DIR, "ppo_airline_final.zip")
    evaluate(model_path)
#which airline’s strategy won on average.