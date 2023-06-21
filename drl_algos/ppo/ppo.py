import random
import time
from typing import Optional, Union

import gym
import gym.spaces
import numpy as np
import numpy.typing as npt
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from drl_algos.common.algorithm import Algorithm
from drl_algos.ppo.agents import DeepSetAgent


# borrowed from CleanRL single-file PPO implementation
class PPO(Algorithm):
    def __init__(
        self,
        env: Union[SubprocVecEnv, DummyVecEnv],
        learning_rate: float = 2.5e-4,
        anneal_lr: bool = False,
        num_steps: int = 128,
        gae: bool = True,
        gae_lambda: float = 0.97,
        gamma: float = 0.95,
        n_minibatches: int = 4,
        update_epochs: int = 4,
        norm_adv: bool = True,
        clip_coef: float = 0.2,
        clip_vloss: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        seed: int = 1,
        device: str = "cpu",
        tensorboard_log: str = "./run",
    ):
        super().__init__(env, learning_rate, seed, device, tensorboard_log)
        self.num_envs = env.num_envs
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.num_steps = num_steps
        self.gae = gae
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_minibatches = n_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.hyperparams = {
            "num_envs": self.num_envs,
            "learning_rate": self.learning_rate,
            "anneal_lr": anneal_lr,
            "num_steps": num_steps,
            "use_gae": self.gae,
            "gae_lambda": self.gae_lambda,
            "gamma": self.gamma,
            "n_minibatches": self.n_minibatches,
            "update_epochs": self.update_epochs,
            "normalize_advantages": self.norm_adv,
            "clip_coefficient": self.clip_coef,
            "clip_value_loss": self.clip_vloss,
            "entropy_coefficient": self.ent_coef,
            "value_function_coefficient": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
        }

        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.n_minibatches)

        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in self.hyperparams.items()])),
        )

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # TODO: modify it to multibinary
        assert isinstance(self.env.action_space, gym.spaces.Discrete), "only discrete action space is supported"

        self.agent = DeepSetAgent(self.env).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((self.num_steps, self.num_envs) + self.env.observation_space.shape).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + self.env.action_space.shape).to(self.device)
        self.masks = torch.zeros((self.num_steps, self.num_envs, self.env.action_space.n), dtype=torch.bool).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

    def learn(self, total_timesteps: int = 500000):
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(self.env.reset()).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        next_masks = torch.tensor(np.array(self.env.env_method("action_masks")), dtype=torch.bool).to(self.device)
        num_updates = total_timesteps // self.batch_size

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += 1 * self.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done
                self.masks[step] = next_masks

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs, masks=next_masks)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the enviroment and log data.
                next_obs, reward, done, info = self.env.step(action.cpu().numpy())
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_masks = torch.tensor(np.array(self.env.env_method("action_masks")), dtype=torch.bool).to(self.device)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

                for item in info:
                    if "episode" in item.keys():
                        print(f"global_step: {global_step}, episodic_return={item['episode']['r']}")
                        self.writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        self.writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        break

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values

            # flatten the batch
            b_obs = self.obs.reshape((-1,) + self.env.observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.env.action_space.shape)
            b_masks = self.masks.reshape((-1, +self.env.action_space.n))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds], b_masks[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("FPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar("charts/FPS", int(global_step / (time.time() - start_time)), global_step)

    def predict(self, obs: npt.NDArray, masks: Optional[npt.NDArray] = None) -> npt.NDArray:
        with torch.no_grad():
            action = self.agent.get_action(
                torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(masks, dtype=torch.bool), deterministic=True
            ).numpy()
        return action

    def save(self, path: str) -> None:
        torch.save(self.agent.state_dict(), path)

    def load(self, path: str) -> None:
        self.agent.load_state_dict(torch.load(path))
