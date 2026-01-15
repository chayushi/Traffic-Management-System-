import numpy as np
from traffic_env import TrafficSignalEnv
from ppo_agent import PPO
from config import Config
import torch
import time
import argparse
import traci

def run_simulation(model_path=None, episodes=1, step_delay=100):
    """
    Run simulation with or without a trained model
    
    Args:
        model_path: Path to the trained model (None for manual control)
        episodes: Number of episodes to run
        step_delay: Delay between steps in milliseconds (for visualization)
    """
    
    env = TrafficSignalEnv(render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = None
    if model_path:
        agent = PPO(state_dim, action_dim, Config)
        agent.load_model(model_path)
        print(f"Loaded model from {model_path}")
    
    print("Starting simulation...")
    print("Press 'Ctrl+C' to stop the simulation")
    
    for episode in range(episodes):
        print(f"\n=== Episode {episode + 1}/{episodes} ===")
        
        state, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            if agent:
                # Use trained model
                action, _, _ = agent.act(state)
                action_name = "NS Green" if action == 0 else "EW Green"
            else:
                # Manual control - switch phases every 20 steps
                action = 0 if (step // 20) % 2 == 0 else 1
                action_name = "NS Green" if action == 0 else "EW Green"
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Display current status
            print(f"Step {step:3d} | Action: {action_name:8s} | Reward: {reward:6.2f} | "
                  f"Vehicles: {get_vehicle_counts()} | Waiting: {get_waiting_vehicles()}")
            
            state = next_state
            episode_reward += reward
            step += 1
            
            # Add delay for better visualization
            time.sleep(step_delay / 1000)
        
        print(f"Episode {episode + 1} completed with total reward: {episode_reward:.2f}")
        print(f"Total simulation steps: {step}")
        print(f"Total vehicles departed: {get_total_departed()}")
        print(f"Average waiting time: {get_average_waiting_time():.2f}")
    
    env.close()

def get_vehicle_counts():
    """Get number of vehicles on each approach"""
    try:
        approaches = {
            'North': ['E0_0', 'E0_1', 'E0_2'],
            'South': ['-E0_0', '-E0_1', '-E0_2'],
            'East': ['E1_0', 'E1_1', 'E1_2'],
            'West': ['-E1_0', '-E1_1', '-E1_2']
        }
        
        counts = {}
        for direction, lanes in approaches.items():
            total = 0
            for lane in lanes:
                try:
                    total += traci.lane.getLastStepVehicleNumber(lane)
                except:
                    pass
            counts[direction] = total
        
        return counts
    except:
        return {"North": 0, "South": 0, "East": 0, "West": 0}

def get_waiting_vehicles():
    """Get number of waiting vehicles on each approach"""
    try:
        approaches = {
            'North': ['E0_0', 'E0_1', 'E0_2'],
            'South': ['-E0_0', '-E0_1', '-E0_2'],
            'East': ['E1_0', 'E1_1', 'E1_2'],
            'West': ['-E1_0', '-E1_1', '-E1_2']
        }
        counts = {}
        waiting = {}
        for direction, lanes in approaches.items():
            total = 0
            for lane in lanes:
                try:
                    total += traci.lane.getLastStepHaltingNumber(lane)
                except:
                    pass
            waiting[direction] = total
        
        return waiting
    except:
        return {"North": 0, "South": 0, "East": 0, "West": 0}

def get_total_departed():
    """Get total number of vehicles that departed"""
    try:
        return traci.simulation.getDepartedNumber()
    except:
        return 0

def get_average_waiting_time():
    """Get average waiting time of vehicles"""
    try:
        waiting_times = []
        for vehicle_id in traci.vehicle.getIDList():
            try:
                waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                if waiting_time > 0:
                    waiting_times.append(waiting_time)
            except:
                pass
        
        return np.mean(waiting_times) if waiting_times else 0
    except:
        return 0

def compare_strategies():
    """Compare RL agent vs fixed-time strategy"""
    print("=== Comparing Strategies ===")
    
    # Test fixed-time strategy
    print("\n1. Fixed-time strategy (30s each phase):")
    fixed_time_rewards = test_fixed_time_strategy()
    print(f"Average reward: {np.mean(fixed_time_rewards):.2f}")
    
    # Test RL strategy if model exists
    model_path = "./models/ppo_traffic_final.zip"
    try:
        print("\n2. RL strategy:")
        env = TrafficSignalEnv(render_mode=None)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = PPO(state_dim, action_dim, Config)
        agent.load_model(model_path)
        
        rl_rewards = []
        for _ in range(3):
            state, info = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _, _ = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
            
            rl_rewards.append(episode_reward)
        
        env.close()
        print(f"Average reward: {np.mean(rl_rewards):.2f}")
        
        # Comparison
        print(f"\nComparison:")
        print(f"Fixed-time: {np.mean(fixed_time_rewards):.2f}")
        print(f"RL agent: {np.mean(rl_rewards):.2f}")
        improvement = ((np.mean(rl_rewards) - np.mean(fixed_time_rewards)) / abs(np.mean(fixed_time_rewards))) * 100
        print(f"Improvement: {improvement:+.2f}%")
        
    except FileNotFoundError:
        print("No trained model found. Train the model first using train.py")

def test_fixed_time_strategy():
    """Test fixed-time traffic light strategy"""
    env = TrafficSignalEnv(render_mode=None)
    rewards = []
    
    for _ in range(3):
        state, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # Fixed-time strategy: switch every 30 steps (30 seconds)
            action = 0 if (step // 30) % 2 == 0 else 1
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            step += 1
        
        rewards.append(episode_reward)
    
    env.close()
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run traffic simulation with RL agent')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--delay', type=int, default=100, help='Delay between steps in milliseconds')
    parser.add_argument('--compare', action='store_true', help='Compare RL vs fixed-time strategy')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_strategies()
    else:
        run_simulation(
            model_path=args.model,
            episodes=args.episodes,
            step_delay=args.delay
        )