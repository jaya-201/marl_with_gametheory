from stable_baselines3.common.vec_env import DummyVecEnv
from src.airline_market import AirlineMarketEnv
from src.ppo_agent import AirlinePPOAgent
from src.config import AGENTS, TOTAL_TIMESTEPS

def make_env():
    return AirlineMarketEnv(airlines=AGENTS)

def train():
    env = DummyVecEnv([make_env])
    agent = AirlinePPOAgent(env)
    agent.train(total_timesteps=TOTAL_TIMESTEPS)
    agent.save()

if __name__ == "__main__":
    train()
