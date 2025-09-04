from sb3_contrib.common.wrappers import DummyVecEnv
from src.airline_market import AirlineMarketEnv
from src.ppo_agent import AirlinePPOAgent
from src.config import AGENTS

def evaluate(model_path, episodes=5):
    env = DummyVecEnv([lambda: AirlineMarketEnv(airlines=AGENTS)])
    agent = AirlinePPOAgent(env, load_path=model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_rewards = {agent_name: 0 for agent_name in AGENTS}

        while not done:
            actions = agent.predict(obs)
            obs, rewards, terminations, truncs, infos = env.step(actions)
            for a in AGENTS:
                total_rewards[a] += rewards[a]

            done = all(terminations.values())

        print(f"Episode {ep+1} rewards: {total_rewards}")

if __name__ == "__main__":
    # Replace with your trained model path
    evaluate("models/ppo_airline_final.zip")