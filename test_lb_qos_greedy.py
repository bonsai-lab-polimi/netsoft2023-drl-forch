import numpy as np
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

from envs.fog_env import FogOrchestrationEnv
from envs.utils import greedy_lb_qos_policy

MONITOR_PATH = "./results/lb_qos_greedy.monitor.csv"

if __name__ == "__main__":
    env = FogOrchestrationEnv(n_nodes=10, arrival_rate_r=100, call_duration_r=1, episode_length=100)
    env.reset()
    _, _, _, info = env.step(0)
    info_keywords = tuple(info.keys())
    env = FogOrchestrationEnv(n_nodes=10, arrival_rate_r=100, call_duration_r=1, episode_length=100)
    env = Monitor(env, filename=MONITOR_PATH, info_keywords=info_keywords)
    returns = []
    for _ in tqdm(range(10000)):
        obs = env.reset()
        action_mask = env.action_masks()
        return_ = 0.0
        done = False
        while not done:
            action = greedy_lb_qos_policy(obs, action_mask, env.lat_val, env.request.latency)
            obs, reward, done, info = env.step(action)
            action_mask = env.action_masks()
            return_ += reward
        returns.append(return_)
    print(info["block_prob"])
    print(f"{np.mean(returns)} +/- {1.96 * np.std(returns) / np.sqrt(len(returns))}")
