from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

import os


class Network(nn.Module):
    def __init__(self, device, obs_dims, action_dims, with_cnn=False, model_dir: str = ""):
        super().__init__()
        self.device = device

        self.cnn = with_cnn
        self.DIR = model_dir + "Models/"
        os.makedirs(self.DIR, exist_ok=True)

        if with_cnn:
            c, _, _ = obs_dims
            self.shared_net = nn.Sequential(
                self.layer_init(nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)),
                nn.ReLU(),
                self.layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
                nn.ReLU(),
                self.layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                self.layer_init(nn.Linear(3136, 512)),
                nn.ReLU(),
            )

        else:

            self.shared_net = nn.Sequential(
                self.layer_init(nn.Linear(obs_dims, 128)),
                nn.ReLU()
            )

    def layer_init(self, layer, std=np.sqrt(2)):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, 0.0)
        return layer
    
    def forward(self, x):
        #if input is an image -> normalize
        if self.cnn:
            x = x / 255.0 
        values = self.shared_net(x)
        return values
    
    def copy_weights(self, other_network: Network, tau:float = 1):
         with torch.no_grad():
            target_state = self.state_dict()
            source_state = other_network.state_dict()
            
            for key in source_state:
                if 'epsilon' not in key:
                    target_state[key].copy_(
                            tau * source_state[key] + (1.0 - tau) * target_state[key]
                        )
                    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def save(self, file_name:str):
        print(f'Saving model..file: {file_name}')
        file_path = self.DIR + file_name + "model.pt"
        torch.save(self.state_dict(), file_path)

    def load(self, file_name:str):
        print("Loading Previous Data...File: " + file_name)
        file_path = self.DIR + file_name + "model.pt"
        self.load_state_dict(torch.load(file_path, map_location=self.device))
        self.eval()


class ActorCritic(Network):
    def __init__(self, device, obs_dims, action_dims, with_cnn=False, model_dir: str = ""):
        super().__init__(device, obs_dims, action_dims, with_cnn, model_dir)

        n_hidden = 512 if with_cnn else 128
        self.actor = self.layer_init(nn.Linear(n_hidden, action_dims), std=0.01)   #policy network
        self.critic = self.layer_init(nn.Linear(n_hidden, 1), std=1.0)            #value network

    def forward(self, x):
        features = super().forward(x)
        return self.actor(features), self.critic(features)
