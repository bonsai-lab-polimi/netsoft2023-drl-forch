import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from tqdm import tqdm

from envs.fog_env import FogOrchestrationEnv

AGENT_SEED = 2
SEED = 2
env_kwargs = {"n_nodes": 10, "arrival_rate_r": 1000, "call_duration_r": 1, "episode_length": 100}
MONITOR_PATH = f"./results/test/ppo_sb3_{SEED}_n{env_kwargs['n_nodes']}_lam{env_kwargs['arrival_rate_r']}_mu{env_kwargs['call_duration_r']}.monitor.csv"

if __name__ == "__main__":
    env = FogOrchestrationEnv(10, 100, 1)
    env.reset()
    _, _, _, info = env.step(0)
    info_keywords = tuple(info.keys())

    envs = DummyVecEnv([lambda: FogOrchestrationEnv(**env_kwargs, seed=45)])
    envs = VecMonitor(envs, MONITOR_PATH, info_keywords=info_keywords)
    agent = MaskablePPO("MlpPolicy", envs) 
    agent.load(f"./agents/ppo_mask_sb3_{AGENT_SEED}_n{env_kwargs['n_nodes']}_lam{1000}_mu{env_kwargs['call_duration_r']}")
    for _ in tqdm(range(1000)):
        obs = envs.reset()
        action_mask = np.array(envs.env_method("action_masks"))
        done = False
        while not done:
            action = agent.predict(obs, action_masks=action_mask)
            obs, reward, dones, info = envs.step(action)
            action_mask = np.array(envs.env_method("action_masks"))
            done = dones[0]
