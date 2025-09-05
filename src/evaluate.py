from src.airline_market import AirlineMarketEnv
from src.ppo_agent import AirlinePPOAgent
from src.config import AGENTS
from pettingzoo.utils.conversions import parallel_to_aec, aec_to_parallel
import supersuit as ss
import numpy as np

def make_vec_env():
    # 1. Original environment
    env = AirlineMarketEnv(airlines=AGENTS, render_mode="human")

    # 2. Convert to AEC (required by PettingZoo wrappers)
    env = parallel_to_aec(env)

    # 3. Convert back to Parallel, then vectorized for SB3
    env = aec_to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    return env

def evaluate(model_path, episodes=3):
    env = make_vec_env()

    # Load trained PPO agent
    agent = AirlinePPOAgent(env, load_path=model_path)

    for ep in range(episodes):
        print(f"\n=== Episode {ep+1} ===")
        obs = env.reset()
        done = np.array([False])
        total_rewards = {a: 0.0 for a in AGENTS}
        day = 0

        while not all(done):
            day += 1
            # Predict actions for all agents
            actions, _ = agent.model.predict(obs, deterministic=True)

            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)

            done = np.array(list(terminations.values()))

            # Accumulate rewards
            for i, a in enumerate(AGENTS):
                total_rewards[a] += rewards[i]

            # Print daily prices
            current_prices = env.envs[0].env.current_prices  # access actual env
            print(f"Day {day}: ", end="")
            for i, a in enumerate(AGENTS):
                print(f"{a}=${current_prices[i]:.2f} ", end="")
            print()

        # Episode summary
        print("Total revenue per airline:")
        for a in AGENTS:
            print(f"  {a}: {total_rewards[a]:.2f}")

if __name__ == "__main__":
    evaluate("models/ppo_airline_final.zip", episodes=3)
