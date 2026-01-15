import streamlit as st
import numpy as np
import pandas as pd
from traffic_env import TrafficSignalEnv
from ppo_agent import PPO
from config import Config
import matplotlib.pyplot as plt

st.title("Traffic RL Model Dashboard")

# Model upload and loading
model_file = st.file_uploader("Upload RL Model (.zip)", type=["zip"])

if model_file is not None:
    # Save uploaded zip temporarily
    model_path = "uploaded_model.zip"
    with open(model_path, "wb") as f:
        f.write(model_file.getbuffer())

    # Load environment and agent
    env = TrafficSignalEnv(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, action_dim, Config)
    agent.load_model(model_path)
    st.success("Model loaded successfully!")

    # Evaluation
    n_episodes = st.number_input('Number of episodes to evaluate', min_value=1, max_value=50, value=5)
    rewards = []
    for ep in range(n_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _, _ = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        rewards.append(episode_reward)
    avg_rl = np.mean(rewards)
    st.write(f"Average RL Agent Reward ({n_episodes} runs): {avg_rl:.2f}")

    # (Optional) Add fixed-time evaluation below this section as you did in your compare script, then plot both.

    # Plot the rewards
    fig, ax = plt.subplots()
    ax.plot(range(1, n_episodes+1), rewards, marker='o', label='RL Agent')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('RL Agent Rewards per Episode')
    ax.legend()
    st.pyplot(fig)

    # Show table
    df = pd.DataFrame({"Episode": range(1, n_episodes+1), "RL Reward": rewards})
    st.dataframe(df)

    # Allow user to download results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "rl_rewards.csv")

else:
    st.info("Upload a .zip RL model above to begin evaluation!")