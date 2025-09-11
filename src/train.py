import os
from src.airline_market import AirlineMarketEnv
from src.ppo_agent import AirlinePPOAgent
from src.config import AGENTS, TOTAL_TIMESTEPS

# PettingZoo + Supersuit imports
from pettingzoo.utils.conversions import parallel_to_aec, aec_to_parallel
import supersuit as ss
# ParallelEnv: all agents act simultaneously
# AECEnv: agents act one after the other

def make_env():
    env = AirlineMarketEnv(airlines=AGENTS)  # ParallelEnv
    env = parallel_to_aec(env)               # convert to AEC
    env = aec_to_parallel(env)               # convert back to ParallelEnv 
    #compatibility with SuperSuit wrappers
    env = ss.pettingzoo_env_to_vec_env_v1(env)  # vectorize for SB3
    #PPO for multiagent but SB3 is single agent -> need for vectoried enviroment
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3') #batching -> merges multiple copies of environment
    return env # Gym-compatible environment

def train():
    env = make_env()
    agent = AirlinePPOAgent(env)
    agent.train(total_timesteps=TOTAL_TIMESTEPS)
    agent.save()

if __name__ == "__main__":
    train()