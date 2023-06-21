from typing import Optional, Type

import gym


def make_env(env_class: Type[gym.Env], seed: int, env_kwargs: Optional[dict] = None):
    def thunk():
        env = env_class(**env_kwargs)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
