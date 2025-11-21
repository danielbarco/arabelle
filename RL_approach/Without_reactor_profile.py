import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn
from minlp_smr_battery_storage import objective, x_from_var, electric_demand, reactor_models, module_capacity_mwh, module_power_mw, storage_leakage_per_hour,constraints_residuals

# ------------------------- Problem data -------------------------------------
horizon = 24

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

# ------------------------- Environment --------------------------------------
class SMRStorageEnv(gym.Env):
    def __init__(self,seed=None):
        super().__init__()
        self.n_time = horizon
        self.n_demand = len(electric_demand)
        self.seed(seed)

        # Actions: [reactor_model_idx, n_reactor_idx, n_storage_idx] + 24 reactor outputs
        self.action_space = spaces.Box(
            low=np.array([0, 1, 0] + [0.0]*horizon, dtype=np.float32),
            high=np.array([len(reactor_models)-1, 3, 15] + [1]*horizon, dtype=np.float32),
            dtype=np.float32
        )

        # Observations: normalized demand
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_demand,), dtype=np.float32
        )
        self.electric_demand = electric_demand
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

        # Reactor outputs chosen by agent
        faction_reactor = action[3:]
        reactor_capacity = reactor_models[reactor_model] * n_reactor
        reactor_production = np.clip(faction_reactor * reactor_capacity, 0, reactor_capacity)

        # --- Storage SOC ---
        soc = np.zeros(self.n_time)
        max_power = n_storage * module_power_mw
        max_energy = n_storage * module_capacity_mwh
        soc[0] = 0.3 * max_energy  # start at 30%
        ch_list = []
        dis_list = []

        for t in range(self.n_time):
            mismatch = reactor_production[t] - electric_demand[t]
            if mismatch > 0:
                ch = min(mismatch, max_power, max_energy - soc[t])
                dis = 0.0
            else:
                dis = min(-mismatch, max_power, soc[t])
                ch = 0.0
            ch_list.append(ch)
            dis_list.append(dis)

            # Update SOC
            if t < self.n_time - 1:
                soc[t + 1] = np.clip(
                    soc[t] * (1 - storage_leakage_per_hour) + ch - dis, 
                    0, max_energy
                )

        # --- Enforce SOC to return to initial 30% at t=24 ---
        final_soc = soc[0]
        soc_diff = soc[-1] - final_soc
        # Adjust the last timestep slightly
        soc[-1] = final_soc
        # Optionally, adjust net supply to account for small correction
        if soc_diff > 0:
            dis_list[-1] += soc_diff
        else:
            ch_list[-1] += -soc_diff
        soc[-1] = np.clip(soc[-1], 0, max_energy)
        ch_list[-1] = np.clip(ch_list[-1], 0, max_power)
        dis_list[-1] = np.clip(dis_list[-1], 0, max_power)

        # --- Evaluate annual profit ---
        x = x_from_var(reactor_model, n_reactor, n_storage, reactor_production, soc)
        annual_profit = objective(x)

        # --- Compute net supply & mismatch ---
        net_supply = reactor_production + np.array(dis_list) - np.array(ch_list)
        reactor_real_frac = np.clip((electric_demand - np.array(dis_list) + np.array(ch_list)) / reactor_capacity, 0, 1)
        error = (reactor_real_frac - faction_reactor) ** 2
        match_reward = np.sum(error)

        # --- Final reward ---
        reward = annual_profit / self.EPISODE_NORM - match_reward

        obs = electric_demand / np.max(electric_demand)
        done = True
        info = {
            "design": (reactor_model, n_reactor, n_storage),
            "annual_profit_eur": annual_profit,
            "reactor_production": reactor_production,
            "soc": soc,
            "charge": ch_list,
            "discharge": dis_list,
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


def plot_result_old_style(
        reactor_production, 
        ch, 
        dis, 
        soc, 
        net_supply, 
        electric_demand, 
        horizon,
        algorithm_name="N/A",
        duration=0.0,
        settings_desc="",
        profit=0.0
    ):

    # --- Ensure numpy arrays ---
    reactor_production = np.asarray(reactor_production)
    ch = np.asarray(ch)
    dis = np.asarray(dis)
    soc = np.asarray(soc)
    net_supply = np.asarray(net_supply)
    electric_demand = np.asarray(electric_demand)

    hours = np.arange(horizon)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Main plots
    ax1.plot(hours, electric_demand, 'k--', label='Demand', lw=2, zorder=5)
    ax1.plot(hours, reactor_production, 'b-', label='Reactor', lw=2, alpha=0.85)
    ax1.plot(hours, net_supply, 'm-', label='Net Supply', lw=1.6)

    # Charge/discharge shading
    ax1.fill_between(hours, 0, dis, color='red', alpha=0.3, label='Discharge')
    ax1.fill_between(hours, 0, -ch, color='green', alpha=0.3, label='Charge')

    ax1.set_ylabel("Power (MW)")
    ax1.set_xlabel("Hour")

    ax1.set_title(
        f"Algo: {algorithm_name} ({settings_desc})\n"
        f"Profit: EUR {profit:,.0f}/yr"
    )

    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # SOC on second axis
    ax2 = ax1.twinx()
    ax2.plot(hours, soc, 'c.-', lw=1.5, label='SOC')
    ax2.set_ylabel("SOC (MWh)", color='c')

    plt.tight_layout()
    plt.show()


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
        verbose=1,seed=seed
    )

    model.learn(total_timesteps=200_000, callback=callback_reward)
    
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
    cand = x_from_var(
        reactor_model=info['design'][0],
        n_reactor=info['design'][1],
        n_storage=info['design'][2],
        reactor_production=info["reactor_production"],
        soc=info["soc"],
    )
    obj = objective(cand)
    ch = info["charge"]
    dis = info["discharge"]
    res = constraints_residuals(cand,ch,dis)
    print(f"{info['annual_profit_eur']=:,} EUR annual profit from env info")
    print(f"Objective (annual profit EUR): {obj:,}")
    print("Min inequality residual (>=0 for feasibility):", np.min(res))
    print(f"Design -> Reactor model: {info['design'][0]}, n_reactor: {info['design'][1]}, n_storage: {info['design'][2]}")

    print(f"Reactor production (MW): {info['reactor_production']}")
    print(f"SOC (MWh): {info['soc']}")
    reactor_production = info["reactor_production"]
    soc = info["soc"]
    
    net_supply = np.zeros(horizon)
    for t in range(horizon):
        net_supply[t] = reactor_production[t] + dis[t] - ch[t]

    
    # --- Plot all flows ---
    plot_result_old_style(
        reactor_production, 
        ch, 
        dis, 
        soc, 
        net_supply, 
        electric_demand, 
        horizon,
        algorithm_name="Reinforcement Learning PPO",
        duration=0.0,
        settings_desc="30% SOC fixed",
        profit=obj
    )
    