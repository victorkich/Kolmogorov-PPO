import time
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym


import os
import random
import time
from dataclasses import dataclass

import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import sys
import os

# Adiciona o caminho para a pasta cleanrl_utils ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cleanrl_utils.evals.ppo_eval import evaluate


# Função para inicializar o ambiente
def make_env(env_id, gamma):
    env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

# Definir a classe do Agente conforme o modelo salvo
class Agent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# Função para inicializar as camadas do modelo
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Caminho do modelo salvo
model_path = "ppo_continuous_action.cleanrl_model"
env_id = "HalfCheetah-v4"
gamma = 0.99

# Carregar o ambiente
env = make_env(env_id, gamma)
observation_space = env.observation_space
action_space = env.action_space

# Carregar o modelo
agent = Agent(observation_space, action_space)
agent.load_state_dict(torch.load(model_path))
agent.eval()

# Função para medir o tempo de execução
def evaluate_agent(agent, env, num_steps=1000):
    obs, _ = env.reset(seed=42)
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    done = False
    total_reward = 0
    start_time = time.time()

    for step in range(num_steps):
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)
        action = action.squeeze(0)  # Ajustar a forma da ação
        obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        total_reward += reward
        done = terminated or truncated
        if done:
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    return total_reward, elapsed_time

# Executar a avaliação
total_reward, elapsed_time = evaluate_agent(agent, env, num_steps=1000)

print(f"Total Reward: {total_reward}")
print(f"Elapsed Time: {elapsed_time:.2f} seconds")

# Fechar o ambiente
env.close()