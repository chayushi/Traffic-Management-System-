import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traci
from config import Config
import time

class TrafficSignalEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(TrafficSignalEnv, self).__init__()

        self.render_mode = render_mode
        self.sumo_cmd = Config.SUMO_CMD.copy()
        if render_mode == "human":
            self.sumo_cmd[0] = "sumo-gui"
        else:
            self.sumo_cmd[0] = "sumo"
        self.tls_id = "clusterJ5_J7_J8_J9"
        self.max_steps = Config.MAX_STEPS
        self.yellow_time = Config.YELLOW_TIME
        self.min_green_time = Config.MIN_GREEN_TIME
        self.max_green_time = Config.MAX_GREEN_TIME

        self.action_space = spaces.Discrete(2)  # NS green or EW green

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(10,), dtype=np.float32
        )

        self.current_step = 0
        self.last_action = 0
        self.green_time = 0
        self.yellow_phase = False
        self.yellow_time_remaining = 0

        self.prev_total_wait = None

        self._start_simulation()

    def _start_simulation(self):
        try:
            traci.close()
            time.sleep(0.1)
        except Exception:
            pass
        try:
            traci.start(self.sumo_cmd)
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            raise

    def _get_observation(self):
        observation = []
        approaches = {
            'north': ['E0_0', 'E0_1', 'E0_2'],
            'south': ['-E0_0', '-E0_1', '-E0_2'],
            'east': ['E1_0', 'E1_1', 'E1_2'],
            'west': ['-E1_0', '-E1_1', '-E1_2']
        }
        for direction, lanes in approaches.items():
            total_vehicles = 0
            total_waiting = 0
            for lane in lanes:
                try:
                    vehicles = traci.lane.getLastStepVehicleNumber(lane)
                    waiting = traci.lane.getLastStepHaltingNumber(lane)
                    total_vehicles += vehicles
                    total_waiting += waiting
                except Exception:
                    pass
            observation.extend([total_vehicles, total_waiting])
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        observation.extend([current_phase, self.green_time])
        if len(observation) != 10:
            observation = observation[:10] + [0] * (10 - len(observation))
        return np.array(observation, dtype=np.float32)

    def _get_reward(self):
        total_waiting = 0
        for lane in traci.lane.getIDList():
            if ':' not in lane:
                total_waiting += traci.lane.getLastStepHaltingNumber(lane)
        departed = traci.simulation.getDepartedNumber()

        if self.prev_total_wait is None:
            self.prev_total_wait = total_waiting

        # Reward: reduction in waiting time + departed bonus
        reward = (self.prev_total_wait - total_waiting) + departed * 0.2
        self.prev_total_wait = total_waiting

        reward = max(reward, 0)  # always keep reward positive

        return reward

    def _set_phase(self, action):
        if action == 0:
            traci.trafficlight.setPhase(self.tls_id, 0)
        else:
            traci.trafficlight.setPhase(self.tls_id, 2)

    def step(self, action):
        self.current_step += 1

        if self.yellow_phase:
            self.yellow_time_remaining -= 1
            if self.yellow_time_remaining <= 0:
                self.yellow_phase = False
                self._set_phase(action)
                self.last_action = action
                self.green_time = 0
        else:
            if action != self.last_action and self.green_time >= self.min_green_time:
                if self.last_action == 0:
                    traci.trafficlight.setPhase(self.tls_id, 1)
                else:
                    traci.trafficlight.setPhase(self.tls_id, 3)
                self.yellow_phase = True
                self.yellow_time_remaining = self.yellow_time
            else:
                if self.green_time >= self.max_green_time:
                    self._set_phase(action)
                    self.last_action = action
                    self.green_time = 0
                else:
                    self.green_time += 1

        traci.simulationStep()
        observation = self._get_observation()
        reward = self._get_reward()
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_action = 0
        self.green_time = 0
        self.yellow_phase = False
        self.yellow_time_remaining = 0
        self.prev_total_wait = None
        self._start_simulation()
        self._set_phase(0)
        traci.simulationStep()
        observation = self._get_observation()
        info = {}
        return observation, info

    def render(self):
        pass

    def close(self):
        try:
            if traci.isLoaded():
                traci.close()
        except Exception as e:
            print(f"Error when closing TraCI connection: {e}")
