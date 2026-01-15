import numpy as np
from traffic_env import TrafficSignalEnv
from ppo_agent import PPO
from config import Config
import time
import os
import traci


def safe_close_env(env):
    try:
        if traci.isLoaded():
            traci.close()
    except Exception as e:
        print(f"Error when closing TraCI connection: {e}")
    try:
        env.close()
    except Exception as e:
        print(f"Error when closing environment: {e}")


def train():
    env = TrafficSignalEnv(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    agent = PPO(state_dim, action_dim, Config)

    episode_rewards = []
    total_timesteps = 0
    episode = 0

    print("Starting training...")

    while total_timesteps < Config.TOTAL_TIMESTEPS:
        try:
            state, info = env.reset()
            print(f"Initial state shape: {state.shape}, values: {state}")
            episode_reward = 0
            done = False

            while not done:
                action, log_prob, value = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.store_transition(state, action, log_prob, value, reward, done)

                state = next_state
                episode_reward += reward
                total_timesteps += 1

                # Update if we have enough samples
                if len(agent.memory['states']) >= Config.BATCH_SIZE:
                    print(f"Updating policy with {len(agent.memory['states'])} samples")
                    agent.update()

            episode_rewards.append(episode_reward)
            episode += 1

            # Logging
            if episode % 5 == 0:
                avg_reward = np.mean(episode_rewards[-5:])
                print(f"Episode: {episode}, Timesteps: {total_timesteps}, Avg Reward: {avg_reward:.2f}")

            # Save model
            if total_timesteps % Config.SAVE_FREQ == 0 and total_timesteps > 0:
                model_path = os.path.join(Config.MODEL_DIR, f"ppo_traffic_{total_timesteps}.zip")
                agent.save_model(model_path)
                print(f"Model saved at {model_path}")

            # Evaluate
            if total_timesteps % Config.EVAL_FREQ == 0 and total_timesteps > 0:
                avg_eval_reward = evaluate(agent, Config.EVAL_EPISODES)
                print(f"Evaluation after {total_timesteps} timesteps: Avg Reward: {avg_eval_reward:.2f}")

        except traci.exceptions.FatalTraCIError:
            print("SUMO connection lost. Restarting simulation...")
            safe_close_env(env)
            env = TrafficSignalEnv(render_mode=None)
            continue
        except Exception as e:
            print(f"Error during training: {e}")
            safe_close_env(env)
            env = TrafficSignalEnv(render_mode=None)
            continue

    safe_close_env(env)

    # Save final model
    final_model_path = os.path.join(Config.MODEL_DIR, "ppo_traffic_final.zip")
    agent.save_model(final_model_path)
    print(f"Final model saved at {final_model_path}")


def evaluate(agent, n_episodes=5):
    env = TrafficSignalEnv(render_mode=None)
    total_rewards = []

    for _ in range(n_episodes):
        try:
            state, info = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _, _ = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)
        except traci.exceptions.FatalTraCIError:
            print("SUMO connection lost during evaluation. Skipping episode...")
            continue
        except Exception as e:
            print(f"Error during evaluation: {e}")
            continue

    env.close()
    return np.mean(total_rewards) if total_rewards else 0


if __name__ == "__main__":
    train()
