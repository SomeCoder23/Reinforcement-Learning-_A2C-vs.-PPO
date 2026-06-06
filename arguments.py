import argparse
from enum import Enum

class Algorithm(Enum):
    A2C = 1
    PPO = 2

class Arguments:
    """
    Parse training arguments for a Dueling-DDQN setup. Defaults are loaded from
    hyperparameters.yml if present 
    """

    def __init__(self, hyperparameters_set) -> None:
        self.hyperparameters = hyperparameters_set

   
    def build_parser(self, args, algo:Algorithm) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Reinforcement Learning training arguments")

        p.add_argument("--env-id", default=self.hyperparameters["env_id"])
        p.add_argument("--seed", type=int, default=None)
        p.add_argument('--train', help='Training mode', action='store_true')
        p.add_argument('--test-episodes', help='Number of episodes to test for', default=100, type=int)
        p.add_argument("--atari", type=self._str2bool, nargs="?", const=True, default=self.hyperparameters.get("atari", False))
        
        algo = self._parse_algorithm(algo)
        config = {}

        if algo == Algorithm.A2C or algo == Algorithm.PPO:
            print("Getting actor-critic hyperparameters...")
            config = self.hyperparameters['a2c']
            p.add_argument('--no-clip-grad', help='If true, does not clips gradient during training', action='store_true')
            p.add_argument('--no-anneal-lr', help='If true, does not anneal learning rate during training', action='store_true')
            p.add_argument("--gae-lambda",type=float, default=float(config.get("g_lambda", 0.95)))
            p.add_argument("--max-norm",type=float, default=float(config.get("max-norm", 0.5)))

            if algo == Algorithm.PPO:
                config = self.hyperparameters['ppo']
                p.add_argument("--update-epochs", type=int, default=config.get("update_epochs", 3))
                p.add_argument("--num-mini-batch", type=int, default=config.get("num_mini_batch", 4))
                p.add_argument("--clip-coef", type=float, default=0.1, help="the clipping coefficient")
                p.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
         
            p.add_argument("--entropy-weight",type=float, help="entropy coefficient for loss", default=float(config.get("entropy-weight", 0.01)))
            p.add_argument("--value-weight",type=float, help="value function coefficient for loss", default=float(config.get("value-weight", 0.25)))

        p.add_argument("--total-steps", help='Number of steps to train for', type=int, default=config.get("steps", 10_000))
        p.add_argument("--num-envs", help='Number of envs to run for training', type=int, default=config.get("num_envs", 1))
        p.add_argument("--n-steps", help='Number of steps to apply DQN multi-step feature for (discount reward for how many steps?) or in the context of actor critic, number of steps per rollout', type=int, default=config.get("n_steps", 3))
        p.add_argument("--max-steps", help='The max steps at which to cap the episode', type=int, default=self.hyperparameters.get("max_steps", 0))
        p.add_argument("--max-return", help='The max return at which to stop early if achieved consecutively', type=int, default=self.hyperparameters.get("max_return", 500))

        p.add_argument("--gamma", help='Discount Factor',type=float, default=float(self.hyperparameters.get("gamma", 0.99)))
        p.add_argument("--eval-freq", help='Every how many steps until evaluating the model', type=int, default=self.hyperparameters.get("eval_freq", 200000))
        p.add_argument("--lr", help='Learning Rate',type=float, default=float(config.get("lr", 0.0002)))
        p.add_argument("--log-freq", help='Every how many steps to log training progress', type=int, default=self.hyperparameters.get("log_freq", 5000))
        p.add_argument("--file-name", default=self.hyperparameters["file_name"])

        return p.parse_args(args), algo

    @staticmethod
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        if isinstance(v, str):
            return v.lower() in ("yes", "true", "t", "1")
        return bool(v)
    

    def _parse_algorithm(self, algo):
        try:
            if isinstance(algo, str):
                if algo.isdigit():
                    return Algorithm(int(algo))
                else:
                    return Algorithm[algo.upper()]
            elif isinstance(algo, Algorithm):
                return algo
        except (KeyError, ValueError):
            raise ValueError(f'Invalid Algorithim. Valid Algorithims:\n {list(Algorithm)}')
    
