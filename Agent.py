from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import Categorical
from torchvision import transforms as T
import torch.nn.functional as F

import os
from abc import ABC, abstractmethod

from Network import ActorCritic
from ReplayBuffer import Prioritized_Replay_Buffer

from stable_baselines3.common.buffers import RolloutBuffer

from arguments import Algorithm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(ABC):
    def __init__(self, args, env_name, algo):
        self.DIR = "TrainingData/" + env_name.capitalize() + "/"
        if not os.path.exists(self.DIR):
            os.makedirs(self.DIR)

        self.env_name = env_name
        self.algo = algo
        self.file_name = args.file_name
        self.cnn = args.atari
        self.device = device
        self.training = args.train
        print(self.device)

        if args.train:
            self.gamma = args.gamma
            self.GRAPH_FILE = self.DIR + self.file_name + ".png"
            self.n_steps = args.n_steps
        
        self.lr = args.lr
        self.LOG_FILE = self.DIR + self.file_name + ".log" #why is this here

    def save_graph(self, rewards_per_env : list, episode_lengths):

        if not rewards_per_env:
            return

        fig = plt.figure(1)
        fig.suptitle(str(self.file_name).capitalize() + ' Training Performance', fontsize=14)

        #first subplot: average rewards
        plt.subplot(121) 
        plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')

        collected_rewards = 0
        for i in range(len(rewards_per_env)):
            if not rewards_per_env[i]:
                continue
            collected_rewards += 1
            window = min(50, len(rewards_per_env[i]) + 1)
            mean_rewards = np.convolve(rewards_per_env[i], np.ones(window)/window, mode="valid")
            plt.plot(mean_rewards, label=f"Env#{i + 1}")

        if collected_rewards:
            plt.legend()
        else:
            return


        #second subplot: epsiode lengths
        plt.subplot(122) 
        plt.xlabel('Episodes')
        plt.ylabel('Mean Lengths')

        for i in range(len(episode_lengths)):
            if not episode_lengths[i]:
                continue
            window = min(50, len(episode_lengths[i]) + 1)
            mean_lengths = np.convolve(episode_lengths[i], np.ones(window)/window, mode="valid")
            plt.plot(mean_lengths, label=f"Env#{i + 1}")

        plt.legend()
        

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def save(self, network, file_name: str = ""):
        if file_name == "":
            file_name = self.file_name
        network.save(file_name)

    def np_to_tensor(self, x, dtype = torch.float32):
        if torch.is_tensor(x):
            return x
        if isinstance(x, (list, tuple)):
            x = np.stack(list(x))  # stack along new batch dimension
        x = np.ascontiguousarray(x)
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    @abstractmethod
    def initialize_networks(self, obs_dims, action_dims, load_model: bool = False, file_name: str = ""):
        pass

    @abstractmethod
    def start_eval(self, eval=True):
        pass

class ActorCritic_Agent(Agent):
    def __init__(self, args, env_name, algo):
        super().__init__(args, env_name, algo)

        if self.training:
            self.gae_lambda = args.gae_lambda
            self.entropy_weight = args.entropy_weight
            self.value_weight = args.value_weight
            self.max_norm = args.max_norm
            self.n_envs = args.num_envs
            self.clip_grad = not args.no_clip_grad
            print(f'Clip Grad = {self.clip_grad}')
            self.buffer = None
            self.update_epochs = args.update_epochs
            self.mini_batch_size = (self.n_envs * self.n_steps) // args.num_mini_batch
            self.clip_coef = args.clip_coef
            self.target_kl = args.target_kl

    def initialize_buffer(self, envs):
        self.buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=envs.single_observation_space,
            action_space=envs.single_action_space,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs = self.n_envs
        )

    def initialize_networks(self, obs_dims, action_dims, load_model=False, file_name=""):
        self.nets = ActorCritic(self.device, obs_dims, action_dims, self.cnn, model_dir=self.DIR).to(self.device)
        self.optimizer = torch.optim.Adam(self.nets.parameters(), lr=self.lr, eps=1e-5)
        if load_model:
            self.nets.load(file_name)

    def get_actor_critic_values(self, state, sample_action = False, no_grad=True):
        state = self.np_to_tensor(state)
        if no_grad:
            with torch.no_grad():
                actor_logits, critic_values = self.nets(state)
        else:
            actor_logits, critic_values = self.nets(state)

        distribution = Categorical(logits=actor_logits)
        actions = None
        if sample_action:
            actions = distribution.sample()
        return actor_logits, critic_values, actions
        
    def get_log_probs_and_entropy(self, logits, actions):
        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probs, entropy.mean()
    
    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.as_tensor(returns, dtype=torch.float32, device=self.device)
    
    def update_with_buffer(self, next_obs, dones):
        _, next_values, _ = self.get_actor_critic_values(next_obs)
        self.buffer.compute_returns_and_advantage(next_values.squeeze(-1), dones)

        for batch in self.buffer.get(batch_size=None):
            obs = batch.observations
            actions = batch.actions.flatten()
            returns = batch.returns
            advantages = batch.advantages
            log_probs = batch.log_probs
            old_values = batch.values

            if self.algo == Algorithm.A2C:
                self.optimize_A2C(obs, returns, advantages, actions)
            elif self.algo == Algorithm.PPO:
                self.optimize_PPO(obs, old_values, returns, advantages, actions, log_probs)
            self.buffer.reset()

    def update_no_buffer(self, next_obs, actions, values, action_logits, rewards, dones):
         _, next_values, _ = self.get_actor_critic_values(next_obs)

         rewards = self.np_to_tensor(rewards.flatten())
         dones = self.np_to_tensor(dones.flatten())
         actions = self.np_to_tensor(actions.flatten())
         values = values.squeeze(-1)
         target = rewards + (1- dones) * self.gamma * next_values.flatten()

         advantages = target - values
         self.optimize_A2C(values, target, advantages.detach(), action_logits, actions)

    def optimize_PPO(self, obs, old_values, returns, advantages, actions, old_log_probs):
        indxs = np.arange(len(advantages))
        for _ in range(self.update_epochs):
            np.random.shuffle(indxs)
            for start in range(0, len(advantages), self.mini_batch_size):
                end = start + self.mini_batch_size
                m_batch_idx = indxs[start:end]

                mb_obs = obs[m_batch_idx]
                action_logits, new_values, _ = self.get_actor_critic_values(mb_obs, no_grad=False)

                new_log_probs, entropy = self.get_log_probs_and_entropy(action_logits, actions[m_batch_idx])
                logratio = new_log_probs - old_log_probs[m_batch_idx]
                ratio = logratio.exp()

                m_batch_advantages = advantages[m_batch_idx]
                m_batch_advantages = (m_batch_advantages - m_batch_advantages.mean()) / (m_batch_advantages.std() + 1e-8)

                pg_loss1 = -m_batch_advantages * ratio
                pg_loss2 = -m_batch_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # clip value loss as well 
                new_values = new_values.view(-1)
                v_loss_unclipped = (new_values - returns[m_batch_idx]) ** 2
                v_clipped = old_values[m_batch_idx] + torch.clamp(new_values - old_values[m_batch_idx], -self.clip_coef, self.clip_coef)
                v_loss_clipped = (v_clipped - returns[m_batch_idx]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                loss = policy_loss + v_loss * self.value_weight - self.entropy_weight * entropy 

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.nets.parameters(), self.max_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                if approx_kl > self.target_kl:
                    break

    def optimize_A2C(self, obs, target, advantages, actions):
        action_logits, pred, _ = self.get_actor_critic_values(obs, no_grad=False)
        new_log_probs, entropy = self.get_log_probs_and_entropy(action_logits, actions)

        #calculate value loss 
        value_loss = F.mse_loss(pred.flatten(), target)

        #calculate policy loss
        policy_loss = -(new_log_probs * advantages).mean()

        #calculate total loss (Maximize -> policy return and entropy loss ..... Minimize -> value loss (the error between the expected and actual return))
        loss = policy_loss + value_loss * self.value_weight - entropy * self.entropy_weight

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.nets.parameters(), self.max_norm)
        self.optimizer.step()

    def start_eval(self, eval=True):
        if eval:
            self.nets.eval()
        else:
            self.nets.train()

    def save(self, file_name: str = ""):
        super().save(self.nets, file_name)