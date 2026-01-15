from traffic_env import TrafficSignalEnv
from ppo_agent import PPO
from config import Config
import torch
import numpy as np

def evaluate_model(model_path, n_episodes=10, render=True):
    env = TrafficSignalEnv(render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPO(state_dim, action_dim, Config)
    agent.load_model(model_path)
    
    total_rewards = []