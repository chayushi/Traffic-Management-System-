import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ppo_agent import PPO
from traffic_env import TrafficSignalEnv
from config import Config
import tempfile
import time

# Title and description
st.title("🚦 Traffic Signal Control RL Dashboard")
st.markdown("""
This dashboard allows you to:
- Upload and evaluate a trained RL model
- Run fixed-time baseline simulation
- Compare rewards and improvements
- Visualize results interactively
""")


@st.cache_resource(show_spinner=False)
def load_agent(model_path):
    env = TrafficSignalEnv(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, action_dim, Config)
    agent.load_model(model_path)
    return agent

def run_episode(env, agent=None, action_policy=None):
    """Run one episode with either RL agent or fixed-time policy."""
    state, info = env.reset()
    done = False
    total_reward = 0
    step = 0
    rewards = []

    while not done and step < 1000:
        if agent is not None:
            action, _, _ = agent.act(state)
        elif action_policy is not None:
            action = action_policy(step)
        else:
            action = 0  # Default to fixed 0 if no policy
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        
        total_reward += reward
        rewards.append(total_reward)
        step += 1

    return total_reward, rewards

def fixed_time_policy(step):
    # Switch phase every 30 steps
    return 0 if (step // 30) % 2 == 0 else 1

# Sidebar: model upload and settings
st.sidebar.header("Model and Evaluation Settings")
uploaded_file = st.sidebar.file_uploader("Upload your trained RL model (.zip)", type=["zip"])
num_episodes = st.sidebar.slider("Number of Episodes", 1, 30, 5)
run_eval = st.sidebar.button("Run Evaluation")

if uploaded_file:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp.write(uploaded_file.read())
        model_path = tmp.name

    agent = load_agent(model_path)
    st.sidebar.success("Model loaded successfully.")
else:
    agent = None
    model_path = None

if run_eval:
    env_rl = TrafficSignalEnv(render_mode=None)
    env_fixed = TrafficSignalEnv(render_mode=None)
    
    # Run RL agent episodes
    rl_rewards = []
    rl_cumulative = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(num_episodes):
        status_text.text(f"Running RL episode {i+1}/{num_episodes}...")
        total_reward, rewards = run_episode(env_rl, agent=agent)
        rl_rewards.append(total_reward)
        rl_cumulative.append(rewards)
        progress_bar.progress((i+1)/num_episodes)
    env_rl.close()
    
    # Run fixed-time episodes
    fixed_rewards = []
    fixed_cumulative = []
    for i in range(num_episodes):
        status_text.text(f"Running Fixed-time episode {i+1}/{num_episodes}...")
        total_reward, rewards = run_episode(env_fixed, action_policy=fixed_time_policy)
        fixed_rewards.append(total_reward)
        fixed_cumulative.append(rewards)
    env_fixed.close()
    progress_bar.empty()
    status_text.text("Evaluation Complete!")

    # Show metrics
    avg_rl = np.mean(rl_rewards)
    avg_fixed = np.mean(fixed_rewards)
    improvement = ((avg_rl - avg_fixed) / abs(avg_fixed)) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Average RL Reward", f"{avg_rl:.2f}")
    col2.metric("Average Fixed-time Reward", f"{avg_fixed:.2f}")
    col3.metric("Improvement %", f"{improvement:.2f}%")

    # Prepare DataFrame for episode-wise rewards
    df = pd.DataFrame({
        "Episode": list(range(1, num_episodes+1)),
        "RL Agent Reward": rl_rewards,
        "Fixed-time Reward": fixed_rewards,
        "Improvement (%)": [((r - f) / abs(f)) * 100 for r, f in zip(rl_rewards, fixed_rewards)]
    })

    # Line plot of rewards
    st.subheader("Episode Reward Comparison")
    plt.figure(figsize=(10, 5))
    plt.plot(df["Episode"], df["RL Agent Reward"], marker='o', label="RL Agent", color='tab:blue')
    plt.plot(df["Episode"], df["Fixed-time Reward"], marker='s', label="Fixed-time", color='tab:orange')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards per Episode")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())

    # Bar plot for improvement %
    st.subheader("Improvement Percentage per Episode")
    plt.figure(figsize=(10, 5))
    plt.bar(df["Episode"], df["Improvement (%)"], color='tab:green', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Improvement (%)")
    plt.title("RL Agent Performance Improvement Over Fixed-time")
    plt.grid(True)
    st.pyplot(plt.gcf())

    # Show detailed table
    st.subheader("Detailed Episode-wise Results")
    st.dataframe(df)

    # Download results
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results CSV", csv_data, "traffic_rl_comparison_results.csv")

else:
    st.info("Upload a RL model and set the number of episodes, then click 'Run Evaluation' from the sidebar.")