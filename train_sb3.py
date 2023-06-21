from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from envs.fog_env import FogOrchestrationEnv

SEED = 2

env_kwargs = {"n_nodes": 10, "arrival_rate_r": 100, "call_duration_r": 1, "episode_length": 100}
MONITOR_PATH = f"./results/ppo_mask_sb3_{SEED}_plot.monitor.csv"
# MONITOR_PATH = None

if __name__ == "__main__":
    env = FogOrchestrationEnv(n_nodes=10, arrival_rate_r=100, call_duration_r=1, episode_length=100)
    _, _, _, info = env.step(0)
    info_keywords = tuple(info.keys())
    envs = SubprocVecEnv([lambda: FogOrchestrationEnv(**env_kwargs, seed=2) for i in range(8)])
    envs = VecMonitor(envs, MONITOR_PATH, info_keywords=info_keywords)
    agent = MaskablePPO("MlpPolicy", envs, gamma=0.95, n_steps=100, batch_size=32, verbose=1, seed=SEED)
    agent.learn(1500000)
    # agent.save(
    #    f"./agents/ppo_mask_sb3_{SEED}_n{env_kwargs['n_nodes']}_lam{env_kwargs['arrival_rate_r']}_mu{env_kwargs['call_duration_r']}"
    # )
