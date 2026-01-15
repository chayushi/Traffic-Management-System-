import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sumolib
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import matplotlib.pyplot as plt

class TrafficSignalEnv(gym.Env):
    """Custom Environment for Traffic Signal Control using SUMO"""
    metadata = {'render_modes': ['human']}
    
    def __init__(self, sumo_cfg_file, max_steps=1000, render_mode=None):
        super(TrafficSignalEnv, self).__init__()
        
        # Initialize SUMO
        self.sumo_cfg_file = "simu.sumocfg"
        self.max_steps = max_steps
        self.step_count = 0
        self.render_mode = render_mode
        
        # Define action and observation space
        # Actions: 0 = keep current phase, 1 = switch to next phase
        self.action_space = spaces.Discrete(2)
        
        # Observation space: queue lengths from each approach
        self.observation_space = spaces.Box(
            low=0, 
            high=100, 
            shape=(8,),  # 8 approaches to the intersection
            dtype=np.float32
        )
        
        # Traffic light ID
        self.tl_id = "clusterJ5_J7_J8_J9"
        
        # Current phase index
        self.current_phase = 0
        self.phases = [
            "rrrrGGGGrrrrGGGG",  # Phase 0: East-West green
            "rrrryyyyrrrryyyy",  # Phase 1: Yellow transition
            "GGGGrrrrGGGGrrrr",  # Phase 2: North-South green
            "yyyyrrrryyyyrrrr"   # Phase 3: Yellow transition
        ]
        
        # Start SUMO
        self.start_simulation()
        
    def start_simulation(self):
        """Start SUMO simulation"""
        sumo_binary = sumolib.checkBinary('sumo-gui' if self.render_mode == 'human' else 'sumo')
        traci.start([sumo_binary, "-c", self.sumo_cfg_file, "--tripinfo-output", "tripinfo.xml"])
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.step_count = 0
        traci.close()
        self.start_simulation()
        
        # Set initial phase
        traci.trafficlight.setRedYellowGreenState(self.tl_id, self.phases[0])
        self.current_phase = 0
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Execute one time step within the environment"""
        self.step_count += 1
        
        # Apply action
        self._apply_action(action)
        
        # Advance simulation
        traci.simulationStep()
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._get_reward()
        
        # Check if done
        terminated = self.step_count >= self.max_steps
        truncated = False  # We don't use truncation in this environment
        
        # Info dictionary
        info = {}
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action):
        """Apply the selected action"""
        if action == 1:  # Switch to next phase
            self.current_phase = (self.current_phase + 1) % len(self.phases)
            traci.trafficlight.setRedYellowGreenState(self.tl_id, self.phases[self.current_phase])
        # Else, keep current phase
    
    def _get_observation(self):
        """Get the current observation from the environment"""
        # Get queue lengths from each approach
        observation = np.zeros(8, dtype=np.float32)
        
        # Define the edges for each approach
        approaches = {
            'E1': ['E1_0', 'E1_1', 'E1_2'],          # From J4 to intersection
            '-E1': ['-E1_0', '-E1_1', '-E1_2'],      # From J5 to intersection
            'E0': ['E0_0', 'E0_1', 'E0_2'],          # From J2 to intersection
            '-E0': ['-E0_0', '-E0_1', '-E0_2']       # From J3 to intersection
        }
        
        # Calculate queue lengths for each approach
        for i, (approach, lanes) in enumerate(approaches.items()):
            total_vehicles = 0
            halting_vehicles = 0
            for lane in lanes:
                total_vehicles += traci.lane.getLastStepVehicleNumber(lane)
                halting_vehicles += traci.lane.getLastStepHaltingNumber(lane)
            observation[i] = total_vehicles
            observation[i+4] = halting_vehicles
        
        return observation
    
    def _get_reward(self):
        """Calculate the reward based on current state"""
        # Reward is negative of total waiting time
        total_waiting_time = 0
        
        # Get all lanes in the network
        lanes = traci.lane.getIDList()
        
        for lane in lanes:
            # Skip internal lanes
            if lane.startswith(':'):
                continue
                
            # Get waiting time for this lane
            waiting_time = traci.lane.getWaitingTime(lane)
            total_waiting_time += waiting_time
        
        # Negative reward to minimize waiting time
        reward = -total_waiting_time
        
        return reward
    
    def render(self):
        """Render the environment"""
        # SUMO GUI handles rendering
        pass
    
    def close(self):
        """Close the environment"""
        traci.close()

def train_ppo_model(env, total_timesteps=10000):
    """Train a PPO model on the environment"""
    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./ppo_traffic_tensorboard/"
    )
    
    # Create callback for evaluation
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=500,
        deterministic=True,
        render=False
    )
    
    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    
    # Save the model
    model.save("ppo_traffic_signal")
    
    return model

def evaluate_model(model, env, num_episodes=10):
    """Evaluate the trained model"""
    rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print(f"Mean Reward: {mean_reward} ± {std_reward}")
    
    return mean_reward, std_reward

def run_simulation_with_model(model, env, render=True):
    """Run the simulation with the trained model"""
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        if render:
            env.render()
    
    print(f"Total reward: {total_reward}")
    return total_reward

if __name__ == "__main__":
    # Create environment
    sumo_cfg_file = "simu.sumocfg"  # Make sure this file exists and points to your network
    
    # First, let's create a simple test to make sure SUMO works
    try:
        # Test SUMO connection
        sumo_binary = sumolib.checkBinary('sumo')
        traci.start([sumo_binary, "-c", sumo_cfg_file, "--tripinfo-output", "tripinfo.xml"])
        print("SUMO connection successful!")
        traci.close()
        
        # Now create the environment
        env = TrafficSignalEnv(sumo_cfg_file)
        
        # Wrap environment for Stable Baselines3
        env = DummyVecEnv([lambda: env])
        
        # Train the model
        print("Training PPO model...")
        model = train_ppo_model(env, total_timesteps=5000)  # Reduced for testing
        
        # Evaluate the model
        print("Evaluating model...")
        mean_reward, std_reward = evaluate_model(model, env, num_episodes=3)
        
        # Run simulation with trained model
        print("Running simulation with trained model...")
        run_simulation_with_model(model, env, render=False)
        
        # Close environment
        env.close()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Created a simu.sumocfg file that points to your network and route files")
        print("2. Installed all required packages: pip install stable-baselines3 gymnasium sumo pygame numpy matplotlib")
        print("3. Set the SUMO_HOME environment variable if needed")