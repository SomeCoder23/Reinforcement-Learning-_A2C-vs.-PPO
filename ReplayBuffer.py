import torch
from collections import deque
import random
import numpy as np


class Prioritized_Replay_Buffer:

    def __init__(self, max_size: int, device, total_steps, alpha, beta=0.4, offset=1e-5, n_step=3, n_envs=1, gamma=0.99):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)

        self.device = device
        self.alpha = alpha #controls how much prioritization to use
        self.beta = beta #controls the amount of importance-sampling correction
        self.offset = offset
        self.beta_increment = (1 - beta) / total_steps # to linearly increase beta to 1 at the end of training
        self.max_priority = 1.0

        self.n_step = n_step
        if self.n_step > 1:
            self.n_step_buffer = [deque(maxlen=n_step) for _ in range(n_envs)]
            self.gamma = gamma

    def add(self, experience: tuple, env:int = 0):
        
        # if using n_step add to n_step buffer
        # then check if reached the desired number of steps to then calculate the accumulated reward 
        # after that add the experience to the PER buffer
        if self.n_step > 1:
            self.n_step_buffer[env].append(experience)

            #checks if done is true to then empty ou the n-step buffer
            if experience[3]:
                self.empty_out(env)
                return
            
            #if the n-step buffer is not full yet we return
            if len(self.n_step_buffer[env]) < self.n_step:
                return
            
            experience = self.calculate_return(self.n_step_buffer[env])

        self.buffer.append(experience)
        self.priorities.append(self.max_priority)

    def empty_out(self, env=0):
        while(len(self.n_step_buffer[env]) > 0):
            experience = self.calculate_return(self.n_step_buffer[env], terminated=True) #TODO: maybe later, sum up returns with reversed buffer faster
            self.buffer.append(experience)
            self.priorities.append(max(self.priorities, default=1.0))

    def calculate_return(self, n_buffer:deque, terminated=False):
        n_step_return = 0.0
        _, s, _, _, a = n_buffer[0]
        _, _, ns, d, _ = n_buffer[-1]
        for i in range(len(n_buffer)):
            n_step_return += n_buffer[i][0] * (self.gamma ** i)

        n_buffer.popleft()
        return (n_step_return, s, ns, d or terminated, a)

    def _get_probabilities(self):
        scaled_priorities = np.array(self.priorities, dtype=np.float32) ** self.alpha # controls how much priorities affect sampling probability
        probabilities = scaled_priorities / scaled_priorities.sum()
        return probabilities

    def sample(self, batch_size: int):
        probs = self._get_probabilities()
        indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)
        batch = tuple(self.buffer[i] for i in indices)

        #calculate importance sampling weights to avoid overfitting to high-priority samples
        importance_sampling_weights = (1 / (len(self.buffer) *  probs[np.array(indices)])) ** self.beta
        importance_sampling_weights /= importance_sampling_weights.max() #normalization

        rewards, states, next_states, dones, actions = zip(*batch)
        
        return (
            rewards,
            states,
            next_states,
            dones,
            actions
        ), importance_sampling_weights, indices

    def update_priorities(self, indices, td_errors):
        for i, error in zip(indices, td_errors):
            priority = abs(error) + self.offset #offset to avoid zero priorities
            self.priorities[i] = priority
            if priority > self.max_priority:
                self.max_priority = priority
        self._increase_beta()

    def _increase_beta(self):
        #we increase beta to change how much weight each sampled experience gets when computing loss
        #at first high priority samples have more influence on the loss, gradually that influence decreases as beta increases to 1
        self.beta = min(1.0, self.beta + self.beta_increment)
    
    def __len__(self):
        return len(self.buffer)




    