import os
import numpy as np
class Config:
    # SUMO Configuration
    SUMO_BINARY = "sumo-gui"  # or "sumo" for no GUI
    NET_FILE = "simu.net.xml"
    ROUTE_FILE = "simu.rou.xml"
    ADDITIONAL_FILE = "TLS.xml"
    RANDOM_SEED = np.random.randint(0, 200000)
    SUMO_CMD = [
        "sumo",
        "-c", "simu.sumocfg",
        "--seed", str(RANDOM_SEED),
        "--collision.action", "warn",
        "--time-to-teleport", "-1",
        # ... other args ...
    ]
    
    # Simulation Configuration
    MAX_STEPS = 1000
    YELLOW_TIME = 3
    MIN_GREEN_TIME = 5
    MAX_GREEN_TIME = 60
    
    # PPO Configuration
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    CLIP_EPSILON = 0.2
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    PPO_EPOCHS = 10   # Increase epochs for better training
    BATCH_SIZE = 64
    HIDDEN_SIZE = 256  # Increase hidden size
    
    # Training Configuration
    TOTAL_TIMESTEPS = 200000   # Increased total training timesteps
    SAVE_FREQ = 10000
    LOG_DIR = "./logs/"
    MODEL_DIR = "./models/"
    
    # Evaluation Configuration
    EVAL_EPISODES = 5
    EVAL_FREQ = 5000
    
    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)


Config.setup_directories()
