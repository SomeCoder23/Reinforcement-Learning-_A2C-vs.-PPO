from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from tqdm import tqdm
import gymnasium as gym


plt.rcParams["figure.figsize"] = (10, 5)

class Policy_Network(nn.Module):

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()

        hidden_space1 = 32 
        hidden_space2 = 64  

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        self.value_net = nn.Sequential(
            nn.Linear(hidden_space2, 1)
        )

    def forward(self, input: torch.tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        shared_data = self.shared_net(input.float())

        mean = self.policy_mean_net(shared_data)

        #Softplus Funciton: this isto ensure standard deviation is always positive
        stddev = torch.log( 1 + torch.exp(self.policy_stddev_net(shared_data)))
        state_value = self.value_net(shared_data)

        return mean, stddev, state_value
    

class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int, file_name : str, load : bool = False):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 3e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards
        self.state_values = []

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        if load:
            self.load(file_name)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed 
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs, state_value = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action).sum()

        self.state_values.append(state_value)
        self.probs.append(prob)

        return action.numpy()
    
    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)
        self.state_values = torch.stack(self.state_values).squeeze()
        log_probs = torch.stack(self.probs)

        advantage = deltas - self.state_values.detach().squeeze()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)

        loss = -(log_probs * advantage).sum()
        

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []
        self.state_values = []

    def save(self, file_name):
        torch.save(self.net.state_dict(), (file_name + ".pth"))

    def load(self, file_name):
        print("Loading Previous Data...")
        self.net.load_state_dict(torch.load(file_name + ".pth"))
        self.net.eval()

env = gym.make("LunarLander-v3", continuous=True)
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]

def train(total_num_episodes : int = 5e3, load : bool = False, save_file_name :str = "LunerLander"):

    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, total_num_episodes)  
    seed = 16
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    win_rate = 0
    # Reinitialize agent every seed
    agent = REINFORCE(obs_space_dims, action_space_dims, save_file_name, load)
    reward_over_episodes = []
    timesteps_over_episodes = []

    for episode in tqdm(range(total_num_episodes)):
        obs, info = wrapped_env.reset(seed=seed)
        timesteps = 0
        done = False
        while not done:
            action = agent.sample_action(obs.flatten())
            timesteps += 1

            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            done = terminated or truncated

        if wrapped_env.return_queue[-1] >= 200:
            win_rate += 1
        reward_over_episodes.append(wrapped_env.return_queue[-1])
        timesteps_over_episodes.append(timesteps)
        agent.update()

        if episode % 100 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    agent.save(save_file_name)

    win_rate = (win_rate / total_num_episodes) * 100
    print("WIN RATE: " + str(win_rate) + "%")

    df = pd.DataFrame({
        "episode": range(len(reward_over_episodes)),
        "reward": reward_over_episodes,
        "timesteps": timesteps_over_episodes,
    })
    df["reward_rolling"] = df["reward"].rolling(50).mean()
    df["timesteps_rolling"] = df["timesteps"].rolling(50).mean()


    sns.set(style="darkgrid", context="talk", palette="muted")
    fig, axes = plt.subplots(2, 1, figsize=(12,10), sharex=True)

    # Rewards
    sns.lineplot(ax=axes[0], x="episode", y="reward", data=df, label="Reward per episode")
    sns.lineplot(ax=axes[0], x="episode", y="reward_rolling", data=df, label=f"Rolling Avg ({total_num_episodes} episodes)")
    axes[0].axhline(200, color="red", linestyle="--", label="Solve Threshold (200)")
    axes[0].set_title("Rewards per Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    # Timesteps
    sns.lineplot(ax=axes[1], x="episode", y="timesteps", data=df, label="Timesteps per episode")
    sns.lineplot(ax=axes[1], x="episode", y="timesteps_rolling", data=df, label=f"Rolling Avg ({total_num_episodes} episodes)")
    axes[1].axhline(1600, color="orange", linestyle="--", label="Max Steps (1600)")
    axes[1].set_title("Timesteps per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Timesteps")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def test(episodes, load_file : str = "LunerLander01"):

    print("Starting tests...")
    env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env, episodes)
    total_rewards = []
    agent = REINFORCE(obs_space_dims, action_space_dims, load_file, True)
    for episode in tqdm(range(episodes)):

        obs, info = env.reset()
        done = False
        ep_return = 0

        while not done:

            action = agent.sample_action(obs.flatten())
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

        total_rewards.append(ep_return)

    average_reward = np.mean(total_rewards)

    print(f"Test Results over {episodes} episodes:")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

train(1000, True, "LunerLander02")
test(10, "LunerLander02")