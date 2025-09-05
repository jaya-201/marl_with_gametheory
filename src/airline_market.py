import gymnasium
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
import numpy as np
import joblib
import os
from src.config import AGENTS, MAX_DAYS, PRICE_POINTS

class AirlineMarketEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "airline_market_v0"}

    def __init__(self, airlines=AGENTS, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.agents = airlines
        self.possible_agents = airlines
        self.agent_to_idx = {agent: i for i, agent in enumerate(self.agents)}

        self.demand_models = {}
        for airline in self.agents:
            model_path = os.path.join('models', f'demand_model_{airline}.pkl')
            self.demand_models[airline] = joblib.load(model_path)

        self.price_points = np.array(PRICE_POINTS, dtype=np.float32)
        self.obs_size = len(self.agents) + 1
        self.max_days = MAX_DAYS

    def observation_space(self, agent):
        return Box(low=0.0, high=1.0, shape=(self.obs_size,), dtype=np.float32)

    def action_space(self, agent):
        return Discrete(len(self.price_points))

    def reset(self, seed=None, options=None):
        self.days_left = self.max_days
        self.current_prices = np.full(len(self.agents), 400.0, dtype=np.float32)
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_obs(self, agent):
        normalized_prices = self.current_prices / 1000.0
        normalized_days = self.days_left / self.max_days
        obs = np.append(normalized_prices, normalized_days).astype(np.float32)
        return obs

    def step(self, actions):
        prices = np.array([self.price_points[actions[agent]] for agent in self.agents])
        self.current_prices = prices.astype(np.float32)

        # 2. Compute rewards
        rewards = {}
        for agent in self.agents:
            predicted_pax = self.demand_models[agent].predict(prices.reshape(1, -1))[0]
            predicted_pax = max(0, predicted_pax)
            # Normalize reward
            rewards[agent] = (prices[self.agent_to_idx[agent]] * predicted_pax) / 1000.0

        # 3. Update state
        self.days_left -= 1

        # 4. Check for termination
        terminations = {agent: self.days_left <= 0 for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        # 5. Observations and infos
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos


    def render(self):
        if self.render_mode != "human":
            return  # do nothing if render_mode is not human
        print(f"Days Left: {self.days_left}")
        for agent in self.agents:
            price = self.current_prices[self.agent_to_idx[agent]]
            print(f"  - {agent}: Price = ${price:.2f}")