# Actor-Critic Reinforcement Learning From Scratch (A2C & PPO)

A clean and modular implementation of **Advantage Actor-Critic (A2C)** and **Proximal Policy Optimization (PPO)** written from scratch in Python using **PyTorch** and **Gymnasium**.

---

## 🛠️ Architecture Overview

The codebase is split into modular components to mimic professional production frameworks while maintaining visibility into the tensor math:

* `main.py`: Main entry point configuring environment hyper-parameters and orchestrating the total training run.
* `Environment.py`: Handles vectorized environment stepping, rollout tracking, and state collection loops.
* `Agent.py`: Contains step execution and the optimization functions for the different algorithms (A2C & PPO).
* `Network.py`: Dynamically constructs the neural network architectures (CNN or MLP backends linked to separate Actor and Critic heads) based on the environment's observation space, and handles model serialization (saving and loading).
* `TrainingData/`: Serves as the central repository for all artifact logging. Organizes outputs into environment-specific subdirectories containing saved models, evaluation gameplay videos, raw execution logs, and performance graphs.

---

## 📊 Hyperparameters 

This repository includes structural configurations similar to the standard hyperparameters found in the [paper](https://arxiv.org/pdf/1707.06347).

* **Breakout** - Trained for **10M** steps with the following hyperparameters:
  
| Hyperparameter | A2C | PPO |
| :--- | :--- | :--- |
| **Base Learning Rate** | $2 \times 10^{-4}$ | $2.5 \times 10^{-4}$ |
| **Horizon / Rollout Steps ($T$)** | 5 | 128 |
| **# of Environments** | 16 | 8 |
| **Optimization Epochs** | 1 | 3 |
| **GAE Parameter ($\lambda$)** | 0.95 | 0.95 |
| **Clip Coefficient ($\epsilon$)** | N/A | 0.1 |
| **Value Loss Weight** | 0.25 | 1.0 |
| **Entropy Coefficient** | 0.01 | 0.01 |


## 📈 Results
Both algorithms outperformed the average reward (across 100 episodes) reported in the paper.
| Source | A2C | PPO |
| :--- | :--- | :--- |
| **Paper** | 303.0 | 274.8 |
| **This Implementation** | 406.5 | 426.35 |
