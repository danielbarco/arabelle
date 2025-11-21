import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn
from minlp_smr_battery_storage import objective, x_from_var, electric_demand, reactor_models, module_capacity_mwh, module_power_mw, storage_leakage_per_hour

# ------------------------- Problem data -------------------------------------
horizon = 24
n_reactor_options = [1,2,3]
n_storage_options = [0,5,10,15]

interest_rate = 0.04
reactor_cap_a = 2.0e7
reactor_cap_b = 0.8
reactor_years = 60
reactor_annuity_factor = interest_rate / (1 - (1 + interest_rate) ** (-reactor_years))
reactor_fixed_om_frac = 0.03

module_cost = 1.0e7
storage_years = 20
storage_annuity_factor = interest_rate / (1 - (1 + interest_rate) ** (-storage_years))
storage_fixed_om_frac = 0.02

fuel_price = 5.0  # EUR/MWh
price_base = 70.0
price_sensitivity = 60.0
price_surplus = 10.0

# ------------------------- Storage efficiencies -----------------------------
def charge_efficiency(charge_mw: np.ndarray, max_charge_mw: float) -> np.ndarray:
    if max_charge_mw < 1e-6:
        return np.ones(horizon)
    r = 0.95 - 0.15 * (charge_mw / max_charge_mw) ** 2
    return np.maximum(r, 0.6)

def discharge_efficiency(dis_mw: np.ndarray, max_dis_mw: float) -> np.ndarray:
    if max_dis_mw < 1e-6:
        return np.ones(horizon)
    r = 0.96 - 0.2 * (dis_mw / max_dis_mw) ** 2
    return np.maximum(r, 0.55)

# ------------------------- Single-step Environment --------------------------
class SMRStorageEnv(gym.Env):
    def __init__(self,seed=None):
        super().__init__()
        self.n_time = horizon
        self.n_demand = len(electric_demand)
        self.seed(seed)
        
        # Actions: [reactor_model_idx, n_reactor_idx, n_storage_idx, reactor_smoothness]
        self.action_space = spaces.Box(
            low=np.array([0, 1, 0, 0.0], dtype=np.float32),
            high=np.array([len(reactor_models)-1, 3, 15, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observations: normalized demand
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_demand,), dtype=np.float32
        )

        # Reward normalization for PPO
        self.EPISODE_NORM = np.sum(electric_demand) * price_base * 365
        self.reset()
        
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed) 
        return [seed]
    
    def reset(self):
        obs = electric_demand / np.max(electric_demand)
        return obs.astype(np.float32)

    def step(self, action):
        # --- Map action to design ---
        reactor_model = int(np.clip(np.round(action[0]), 0, len(reactor_models)-1))
        n_reactor = int(np.clip(np.round(action[1]), 1, 3))
        n_storage = int(np.clip(np.round(action[2]), 0, 15))
        smoothness = np.clip(action[3], 0.0, 1.0)

        # --- Reactor profile ---
        avg_demand = np.mean(electric_demand)
        reactor_production = smoothness * np.full(self.n_time, avg_demand) + (1 - smoothness) * electric_demand
        reactor_capacity = reactor_models[reactor_model] * n_reactor
        reactor_production = np.clip(reactor_production, 0, reactor_capacity)

        # --- Storage SOC ---
        soc = np.zeros(self.n_time)
        max_power = n_storage * module_power_mw
        max_energy = n_storage * module_capacity_mwh
        soc[0] = 0.3 * max_energy

        for t in range(self.n_time):
            mismatch = reactor_production[t] - electric_demand[t]
            if mismatch > 0:
                ch = min(mismatch, max_power, max_energy - soc[t])
                dis = 0.0
            else:
                dis = min(-mismatch, max_power, soc[t])
                ch = 0.0
            if t < self.n_time - 1:
                soc[t+1] = soc[t] * (1 - storage_leakage_per_hour) + ch - dis

        # --- Evaluate annual profit ---
        x = x_from_var(reactor_model, n_reactor, n_storage, reactor_production, soc)
        annual_profit = objective(x)
        reward = annual_profit / self.EPISODE_NORM

        obs = electric_demand / np.max(electric_demand)
        done = True
        info = {
            "design": (reactor_model, n_reactor, n_storage),
            "annual_profit_eur": annual_profit,
            "reactor_production": reactor_production,
            "soc": soc,
            'variability_smoothness': smoothness
        }

        return obs.astype(np.float32), reward, done, info

# ------------------------- Reward Logging Callback --------------------------
class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_rewards_eur = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "annual_profit_eur" in info:
                self.episode_rewards.append(info.get("reward", None))
                self.episode_rewards_eur.append(info["annual_profit_eur"])
        return True

# ------------------------- Main training -------------------------------------
if __name__ == "__main__":
    from stable_baselines3.common.utils import set_random_seed
    seed = 42
    set_random_seed(seed)
    env = SMRStorageEnv(seed=seed)
    
    callback_reward = RewardCallback()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=get_schedule_fn(1e-4),
        n_steps=2048,
        batch_size=1024,
        n_epochs=20,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.2,
        ent_coef=0.5,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )

    model.learn(total_timesteps=100_000, callback=callback_reward)
    # Plot rewards (EUR)
    window = 20
    smoothed = np.convolve([r for r in callback_reward.episode_rewards_eur if r is not None],
                           np.ones(window)/window, mode="valid")
    plt.figure()
    plt.plot(smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Total profit (EUR)")
    plt.title("PPO Profit with Curriculum")
    plt.grid(True)
    plt.show()
    
    # --- Evaluate final policy ---
    obs = env.reset()
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)

    print(f"Total annual profit (EUR): {info['annual_profit_eur']:,}")
    print(f"Design -> Reactor model: {info['design'][0]}, n_reactor: {info['design'][1]}, n_storage: {info['design'][2]}, smoothness: {info['variability_smoothness']:.2f}")

    print(f"Reactor production (MW): {info['reactor_production']}")
    print(f"SOC (MWh): {info['soc']}")
    # --- Plot operation ---
    hours = np.arange(horizon)
    plt.figure()
    plt.plot(hours, electric_demand, "k--", label="Demand")
    plt.plot(hours, info["reactor_production"], label="Reactor output")
    plt.plot(hours, info["soc"], label="SOC")
    plt.xlabel("Hour")
    plt.ylabel("MW / MWh")
    plt.legend()
    plt.grid(True)
    plt.show()
