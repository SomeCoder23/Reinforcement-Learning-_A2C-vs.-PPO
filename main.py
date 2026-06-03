from __future__ import annotations

import argparse
import yaml
import sys

from Environment import ActorCritic_Environment
from arguments import Arguments, Algorithm

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    #get the name of the environment the user wants to train
    initial_parser = argparse.ArgumentParser(description='Program to train/test Reinforcement Learning Agent.')
    initial_parser.add_argument('env_name', help='', default="cartpole")
    initial_parser.add_argument('--algo', help='Type of algorithm to train with. (REINFORCE, A2C, PPO)', default=Algorithm.A2C)
    initial_args, remaining_args = initial_parser.parse_known_args()

    #With the env name get the hyperparameter set for that environment
    env_name = initial_args.env_name.lower()
    with open('hyperparameters.yml', 'r') as file:
        hyperparameter_set = yaml.safe_load(file)[env_name]

    #parse the rest of the arguments
    parser = Arguments(hyperparameter_set)
    remaining_args, algo = parser.build_parser(remaining_args, initial_args.algo)
    remaining_args.file_name += "_" + algo.name.upper()
    print(remaining_args.file_name)

    #if training, setup writer for TensorBoard
    if remaining_args.train:
        print("Initializing writer...")
        writer = SummaryWriter(f"runs/{remaining_args.file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(remaining_args).items()])),
        )
    else:
        writer = None

    #initialize Agent and Environment
    print("Starting... with algorithim: " + str(algo))
    print('Arguments: ')
    print(remaining_args)
    env = None
    env = ActorCritic_Environment(env_name, remaining_args, writer, algo)

    #Start training or testing, depending on what the user prompted
    if env != None and remaining_args.train:
        env.train()
    else:
        env.test(remaining_args.test_episodes, plot_results=True)

    print("DONE")
