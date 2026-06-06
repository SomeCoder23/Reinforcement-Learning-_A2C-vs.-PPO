import ale_py
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

from Agent import ActorCritic_Agent

import random
from datetime import datetime, timedelta
from tqdm import tqdm


# from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.atari_wrappers import (  
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

DATE_FORMAT = "%m-%d %H:%M:%S"

class Environment():
    def __init__(self, env_name, args, writer, algo):
        self.args = args
        self.env_name = env_name
        self.writer = writer
        self.env_id = args.env_id
        self.file_name = args.file_name
        self.agent = ActorCritic_Agent(args, env_name, algo)

    def setup_envs(self, render_mode, num_envs, testing = False):
        print("Setting up " + str(num_envs) + " envs...")
        if self.args.atari:
            if testing:
                envs = self.make_env_atari(self.args.seed, 0, test=True)
            else:
                envs = gym.vector.SyncVectorEnv(
                    [lambda: self.make_env_atari(self.args.seed, i) for i in range(num_envs)],
                    autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
                )
            self.obs_space_dims = (4, 84, 84)
            self.action_space_dims = envs.action_space.n if testing else envs.single_action_space.n

        else:
            if testing:
                envs = self.make_env(self.args.seed, 0, render_mode, record=True)
            else:
                envs = gym.vector.SyncVectorEnv(
                    [lambda: self.make_env(self.args.seed, i, render_mode) for i in range(num_envs)],
                    autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
                )
            self.obs_space_dims = np.array(envs.observation_space.shape if testing else envs.single_observation_space.shape).prod()

            action_space = envs.action_space if testing else envs.single_action_space
            if isinstance(action_space, gym.spaces.Discrete):
                self.action_space_dims = action_space.n
            else:
                self.action_space_dims = action_space.shape
        return envs
   
    def record_agent(self, training: bool, env: gym.Env):
        video_dir = "TrainingData/" + self.env_name.capitalize() + "/Videos"
        if training:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder= video_dir,
                name_prefix= self.args.file_name +"_Training_",
                episode_trigger=lambda x: x % 1000 == 0 
            )

        else:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder= video_dir,
                name_prefix= "__" + self.args.file_name +"_Evaluation_",
                episode_trigger=lambda x: True
            )
        return env

    def make_env(self, seed, index, render_mode, record = False):
        env = gym.make(self.env_id, render_mode= render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)  

        if seed is not None:
            env.action_space.seed(seed + index)
            env.observation_space.seed(seed + index)

        if record:
            env = self.record_agent(training=False, env=env)
        return env
    
    def make_env_atari(self, seed, index, test = False):

        if test:
            env = gym.make(self.env_id, render_mode= "rgb_array")
            env = self.record_agent(training=False, env=env)
        else:
            env = gym.make(self.env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env) 

        if index == 0:
            env = self.record_agent(training=True, env=env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4) #repeats action 4 times
        if not test:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if not test:
            env = ClipRewardEnv(env)

        if self.args.max_steps > 0:
            env = gym.wrappers.TimeLimit(env, self.args.max_steps)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)

        if seed is not None:
            env.action_space.seed(seed + index)
            env.observation_space.seed(seed + index)

        return env

    def train(self, load_model = False, file_name: str = ""):
        if file_name != "":
            self.file_name = file_name

        #setup envs for training and single env for evaluation during training
        envs = self.setup_envs("rgb_array", self.args.num_envs)
        eval_env = self.setup_envs("rgb_array", 1, testing=True)

        #for actor critic methods: initialize rollout buffer if agent uses one
        if hasattr(self.agent, "initialize_buffer") and self.args.n_steps > 1:
            self.agent.initialize_buffer(envs)

        #initialize neural networks
        self.agent.initialize_networks(self.obs_space_dims, self.action_space_dims, load_model=load_model, file_name=self.file_name)
        print(f"Training on device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

        #set seed
        SEED = self.args.seed
        if SEED is not None:
            torch.manual_seed(SEED)
            random.seed(SEED)
            np.random.seed(SEED)
        
        #this makes gpu give more consistent and repeatable results across runs (at the cost of being slightly slower)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Training starting...Total steps: {self.args.total_steps}"
        print(log_message)
        with open(self.agent.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')

        num_envs = self.args.num_envs

        # #initialize some variables for showcasing results
        rewards_per_episode = [[] for _ in range(num_envs)]
        episode_lengths = [[] for _ in range(num_envs)]
        episode_count = [1 for _ in range(num_envs)]
        start_time = datetime.now()
        best_return = -float('inf')
        
        last_graph_update_time = start_time
        last_log_time = start_time
        last_save_time = start_time

        obs, _ = envs.reset(seed=SEED)
        envs_starting = np.ones(num_envs)
        envs_resets = np.zeros(num_envs)
        self.no_grad = self.args.n_steps > 1
        step = 0

        #training loop, trains for a certain amount of steps not episodes since we are using vectorized environments
        while step < self.args.total_steps:
            # next_obs, terminations, truncations, infos, step_increment = self.run_step(step, obs, envs_resets, envs)
            if not self.args.no_anneal_lr:
                frac = 1.0 - (step - 1.0) / self.args.total_steps
                lrnow = frac * self.args.lr
                self.agent.optimizer.param_groups[0]["lr"] = lrnow

            for _ in range(self.args.n_steps):
                action_logits, values, actions = self.agent.get_actor_critic_values(obs, sample_action=True, no_grad=self.no_grad)
                log_probs, _ = self.agent.get_log_probs_and_entropy(action_logits, actions)

                actions_np = actions.cpu().numpy()
                next_obs, rewards, terminations, truncations, infos = envs.step(actions_np)

                #add to rollout buffer if using one (n-steps more than one)
                if self.args.n_steps > 1:
                    self.agent.buffer.add(
                        obs,
                        actions_np,
                        rewards,
                        envs_starting,
                        values.squeeze(-1),
                        log_probs
                    )
                envs_starting = np.logical_or(terminations, truncations)
                obs = next_obs
                step += self.args.num_envs

                #if episode finished save episode return and length + check if best return was beaten
                current_time = datetime.now()
                for i, (terminated, truncated) in enumerate(zip(terminations, truncations)):
                    if terminated or truncated:
                        if 'episode' in infos:
                            episode_count[i] += 1
                            eps_return = infos['episode']['r'][i]
                            ep_length = infos['episode']['l'][i]
                            rewards_per_episode[i].append(eps_return)
                            episode_lengths[i].append(ep_length)
                            self.writer.add_scalar("charts/episodic_return", eps_return, step)
                            self.writer.add_scalar("charts/episodic_length", ep_length, step)

                            #check if best return was surpassed
                            if (eps_return.max() > best_return).any():
                                best_return = eps_return.max()
                                with open(self.agent.LOG_FILE, 'a') as file:
                                    file.write(f'{current_time.strftime(DATE_FORMAT)}: Step: {step} Env #{i}: New best return at episode #{episode_count[i] - 1} -> {best_return}!!\n')
                

            #update actor and critic networks every n-step
            if self.agent.buffer != None:
                self.agent.update_with_buffer(next_obs, envs_starting)

            elif self.args.n_steps <= 1:
                mask = np.logical_not(envs_resets)
                if mask.any():
                    self.agent.update_no_buffer(
                        next_obs[mask], 
                        actions[mask], 
                        values[mask], 
                        action_logits[mask], 
                        rewards[mask], 
                        envs_starting[mask]
                    )

            #update results graph every 10 seconds
            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=10):
                self.agent.save_graph(rewards_per_episode, episode_lengths)
                last_graph_update_time = current_time

            #log results so far
            if step % self.args.log_freq == 0 and step != 0:
                window = min(100, len(rewards_per_episode[0]) + 5)
                mean_rewards_per_env = [np.convolve(env_rewards, np.ones(window)/window, mode="valid") if env_rewards else 0 for env_rewards in rewards_per_episode]
                avg_reward = np.mean(np.array([np.mean(env_rewards) for env_rewards in mean_rewards_per_env]))
                elapsed_time = current_time - last_log_time
                log_message = f'{current_time.strftime(DATE_FORMAT)}: Progress: {round((step/self.args.total_steps) * 100, 2)}% Step: {step}, Average Reward: {round(avg_reward, 3)}, Time Since Last Log: {elapsed_time}'
                last_log_time = current_time
                print(log_message)
                with open(self.agent.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')

            #evaluate model over ten episodes 
            if step % self.args.eval_freq == 0 and step != 0:
                self.test(10, env = eval_env)

            #save model
            if current_time - last_save_time > timedelta(seconds=1800):
                self.agent.save(self.file_name)
                last_save_time = current_time

            #check if achieved max score consecutively for early stopping
            achieved_max = np.all(np.array(
                [np.all(np.equal(env_rewards[-10:], self.args.max_return)) 
                 if len(env_rewards) >= 10 else False 
                 for env_rewards in rewards_per_episode]
                ))
            if achieved_max:
                print("YAYYYY")
                with open(self.agent.LOG_FILE, 'a') as file:
                    file.write(f'Step: {step} -> Achieved Max Reward!! \n')
                break
            envs_resets = np.logical_or(terminations, truncations)

        envs.close()
        self.agent.save(self.file_name)

        avg, std, max_reward = self.test(100, env = eval_env)
        print(f'avg: {avg}\tstd: {std}\tmax:{max_reward}')
        self.writer.add_hparams(
            vars(self.args),
            {
                "avg_eval_reward" : avg,
                "std_eval_reward" : std,
                "max_eval_reward": max_reward,
                "best_training_reward": best_return
            }
        )
        self.writer.close()

        self._plot_final_rewards(rewards_per_episode)

    def test(self, episodes, env = None, plot_results = False):

        print("Starting "+ str(episodes) +" tests...with file " + self.args.file_name)
        if env == None:
            env = self.setup_envs("rgb_array", 1, testing=True)
            self.agent.initialize_networks(self.obs_space_dims, self.action_space_dims, load_model=True, file_name=self.args.file_name)
        else:
            self.agent.start_eval()

        total_rewards = []   
        for i in tqdm(range(episodes)):
            obs, info = env.reset()
            done = False
            ep_rewards = 0
            while not done:
                _, _, action = self.agent.get_actor_critic_values([obs], sample_action=True)
                obs, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
                ep_rewards += reward

            total_rewards.append(ep_rewards)

        average_reward = np.mean(total_rewards)
        rewards_std = np.std(total_rewards)
        log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Test Results over {episodes} episodes: \nAverage Reward: {average_reward:.3f} \nMax Reward: {np.max(total_rewards)} \nMin Reward: {np.min(total_rewards)} \nStandard Deviation: {rewards_std:.3f}"
        with open(self.agent.LOG_FILE, 'a') as file:
            file.write(log_message + '\n')
        print(log_message)

        env.close()
        self.agent.start_eval(False)

        if plot_results:
            self._plot_final_rewards(total_rewards, testing=True)
        for i, reward in enumerate(total_rewards):
            print(f'Env #{i}: Reward: {reward}')

        return average_reward, rewards_std, np.max(total_rewards)
    
    def _plot_final_rewards(self, rewards, testing = False):

        if not testing:
            for i in range(len(rewards)):
                plt.plot(rewards[i], label=f"Env#{i + 1}")
            plt.legend()
        else:
            plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Rewards per Episode')
        plt.show()

        #plot histogram of rewards
        if not testing:
            rewards = [r for env in rewards for r in env]
        plt.hist(rewards, bins=20)
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('Histogram of Rewards')
        plt.show()
        
    
