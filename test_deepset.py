import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from tqdm import tqdm

from drl_algos.ppo.ppo import PPO
from envs.fog_env import FogOrchestrationEnv

SEED = 1
env_kwargs = {"n_nodes": 10, "arrival_rate_r": 100, "call_duration_r": 1, "episode_length": 100}
MONITOR_PATH = f"./results/test/ppo_deepset_{SEED}_n{env_kwargs['n_nodes']}_lam{env_kwargs['arrival_rate_r']}_mu{env_kwargs['call_duration_r']}.monitor.csv"

if __name__ == "__main__":
    env = FogOrchestrationEnv(10, 100, 1)
    env.reset()
    _, _, _, info = env.step(0)
    info_keywords = tuple(info.keys())

    envs = DummyVecEnv([lambda: FogOrchestrationEnv(**env_kwargs, seed=1)])
    envs = VecMonitor(envs, MONITOR_PATH, info_keywords=info_keywords)
    # envs = VecNormalize(envs, norm_obs=True, norm_reward=False)
    agent = PPO(envs, num_steps=100, n_minibatches=8, ent_coef=0.001, tensorboard_log=None, seed=SEED)
    agent.load(f"./agents/ppo_deepset_{SEED}")
    for _ in tqdm(range(1000)):
        obs = envs.reset()
        action_mask = np.array(envs.env_method("action_masks"))
        done = False
        while not done:
            action = agent.predict(obs, action_mask)
            obs, reward, dones, info = envs.step(action)
            action_mask = np.array(envs.env_method("action_masks"))
            done = dones[0]
