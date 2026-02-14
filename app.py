import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import time

# -------------------- Generate 100+ Realistic Indian Trains --------------------
def generate_train_data():
    """
    Generate a list of 100+ realistic Indian trains with numbers and names.
    """
    cities = [
        "Mumbai", "Delhi", "Kolkata", "Chennai", "Bengaluru", "Hyderabad", "Ahmedabad",
        "Pune", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Visakhapatnam",
        "Patna", "Vadodara", "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut", "Rajkot",
        "Varanasi", "Srinagar", "Aurangabad", "Dhanbad", "Amritsar", "Allahabad", "Ranchi",
        "Howrah", "Jabalpur", "Gwalior", "Vijayawada", "Jodhpur", "Madurai", "Raipur",
        "Kota", "Chandigarh", "Guwahati", "Solapur", "Hubli", "Mysore", "Tiruchirappalli",
        "Bareilly", "Aligarh", "Moradabad", "Bhubaneswar", "Coimbatore", "Kozhikode"
    ]
    
    train_types = [
        ("Rajdhani Express", 85, 95),
        ("Shatabdi Express", 90, 100),
        ("Duronto Express", 85, 95),
        ("Garib Rath", 70, 80),
        ("Humsafar Express", 75, 85),
        ("Superfast Express", 70, 85),
        ("Express", 60, 75),
        ("Mail", 55, 70),
        ("Passenger", 40, 55)
    ]
    
    trains = []
    for i in range(120):
        city1 = np.random.choice(cities)
        city2 = np.random.choice(cities)
        while city2 == city1:
            city2 = np.random.choice(cities)
        
        type_name, speed_min, speed_max = train_types[np.random.randint(len(train_types))]
        
        if np.random.random() > 0.5:
            name = f"{city1} {city2} {type_name}"
        else:
            name = f"{city1} {type_name}"
        
        base = np.random.choice([1, 2]) * 10000
        number = base + np.random.randint(1001, 9999)
        
        while any(t['number'] == str(number) for t in trains):
            number += 1
        
        base_speed = np.random.uniform(speed_min, speed_max)
        
        trains.append({
            "number": str(number),
            "name": name,
            "base_speed": round(base_speed, 1)
        })
        
        if len(trains) >= 110:
            break
    
    np.random.seed(42)
    positions = np.random.uniform(0, 20, size=len(trains))
    speeds = [t["base_speed"] + np.random.randint(-5, 6) for t in trains]
    for i, t in enumerate(trains):
        t["position_km"] = positions[i]
        t["speed_kmh"] = speeds[i]
    
    trains.sort(key=lambda x: x["position_km"])
    return trains

# -------------------- AI Environment for Training --------------------
class SingleTrainControlEnv(gym.Env):
    """
    Environment for controlling one train (ego) with a lead train ahead and a follow train behind.
    State: [ego_speed, front_distance, back_distance] normalized.
    Action: 0 = decelerate, 1 = maintain, 2 = accelerate.
    Reward: encourage safe distance (1.5-3 km) and high speed.
    """
    def __init__(self, max_speed=100, min_speed=30, time_step=1/3600):  # 1 sec in hours
        super().__init__()
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.time_step = time_step

        self.action_space = spaces.Discrete(3)

        high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=high, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.lead_pos = 15.0
        self.ego_pos = 10.0
        self.follow_pos = 5.0
        self.ego_speed = 60.0
        self.lead_speed = 65.0
        self.follow_speed = 55.0
        return self._get_obs(), {}

    def _get_obs(self):
        speed_norm = (self.ego_speed - self.min_speed) / (self.max_speed - self.min_speed)
        front_dist = max(0, self.lead_pos - self.ego_pos)
        back_dist = max(0, self.ego_pos - self.follow_pos)
        front_norm = min(front_dist / 10.0, 1.0)
        back_norm = min(back_dist / 10.0, 1.0)
        return np.array([speed_norm, front_norm, back_norm], dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.ego_speed -= 5
        elif action == 2:
            self.ego_speed += 5
        self.ego_speed = np.clip(self.ego_speed, self.min_speed, self.max_speed)

        self.lead_speed += np.random.randint(-2, 3)
        self.lead_speed = np.clip(self.lead_speed, self.min_speed, self.max_speed)
        self.follow_speed += np.random.randint(-2, 3)
        self.follow_speed = np.clip(self.follow_speed, self.min_speed, self.max_speed)

        self.lead_pos += self.lead_speed * self.time_step
        self.ego_pos += self.ego_speed * self.time_step
        self.follow_pos += self.follow_speed * self.time_step

        self.lead_pos = min(self.lead_pos, 20.0)
        self.ego_pos = min(self.ego_pos, 20.0)
        self.follow_pos = min(self.follow_pos, 20.0)

        front_dist = self.lead_pos - self.ego_pos
        back_dist = self.ego_pos - self.follow_pos
        reward = 0

        if front_dist < 0.5:
            reward -= 10
        elif front_dist < 1.5:
            reward -= 2
        elif 1.5 <= front_dist <= 3.0:
            reward += 5
        elif front_dist > 5.0:
            reward -= 1

        if back_dist < 0.5:
            reward -= 5

        reward += 0.05 * self.ego_speed

        if action == 0 or action == 2:
            reward -= 0.2

        done = False
        truncated = False
        return self._get_obs(), reward, done, truncated, {}

# -------------------- Train or Load DQN Model --------------------
@st.cache_resource
def load_or_train_model():
    model_path = "dqn_train_control.zip"
    if os.path.exists(model_path):
        model = DQN.load(model_path)
    else:
        env = DummyVecEnv([lambda: SingleTrainControlEnv()])
        model = DQN("MlpPolicy", env, verbose=0, learning_rate=0.001, buffer_size=10000,
                    learning_starts=100, batch_size=32, tau=0.1, gamma=0.99,
                    train_freq=4, gradient_steps=1)
        model.learn(total_timesteps=5000)
        model.save(model_path)
    return model

# -------------------- Helper: Find Front/Back Trains --------------------
def get_front_back(trains, selected_index):
    front_dist = None
    back_dist = None
    if selected_index < len(trains) - 1:
        front_dist = trains[selected_index + 1]["position_km"] - trains[selected_index]["position_km"]
    if selected_index > 0:
        back_dist = trains[selected_index]["position_km"] - trains[selected_index - 1]["position_km"]
    return front_dist, back_dist

# -------------------- Throughput Simulation Functions --------------------
def get_observation_for_train(train, trains_sorted, idx, min_speed=30, max_speed=100):
    """Return normalized observation [speed_norm, front_norm, back_norm] for a given train."""
    speed_norm = (train['speed_kmh'] - min_speed) / (max_speed - min_speed)
    # front distance
    if idx < len(trains_sorted) - 1:
        front_dist = trains_sorted[idx+1]['position_km'] - train['position_km']
    else:
        front_dist = 10.0  # no train ahead → large distance
    # back distance
    if idx > 0:
        back_dist = train['position_km'] - trains_sorted[idx-1]['position_km']
    else:
        back_dist = 10.0
    front_norm = min(front_dist / 10.0, 1.0)
    back_norm = min(back_dist / 10.0, 1.0)
    return np.array([speed_norm, front_norm, back_norm], dtype=np.float32)

def run_throughput_simulation(model, num_trains=20, duration_hours=1.0, use_ai=True, dt_seconds=1):
    """
    Simulate multiple trains for a given duration.
    Trains are recycled when they reach 20 km.
    Returns:
        completed_trains: total number of trains that finished the section
        time_points: list of simulation times (hours)
        completed_counts: cumulative completed trains at each time point
    """
    # Initialise trains randomly on 0-20 km
    np.random.seed(42)
    positions = np.random.uniform(0, 20, size=num_trains)
    speeds = np.random.uniform(40, 90, size=num_trains)  # random initial speeds
    trains = []
    for i in range(num_trains):
        trains.append({
            'position_km': positions[i],
            'speed_kmh': speeds[i],
            'number': f"SIM{i:03d}"   # dummy number for tracking
        })
    trains.sort(key=lambda x: x['position_km'])

    min_speed, max_speed = 30, 100
    time_step_hours = dt_seconds / 3600.0
    total_steps = int(duration_hours * 3600 / dt_seconds)
    
    completed = 0
    time_points = [0.0]
    completed_counts = [0]
    
    # For AI, we apply action every 10 seconds to reduce computation
    action_interval = 10  # seconds
    steps_since_action = 0
    
    for step in range(total_steps):
        # 1. Possibly apply AI actions to all trains
        if use_ai and steps_since_action >= action_interval:
            # Get observations for each train
            obs_list = []
            for idx, t in enumerate(trains):
                obs = get_observation_for_train(t, trains, idx, min_speed, max_speed)
                obs_list.append(obs)
            # Predict actions (vectorised prediction would be better, but we loop)
            actions = []
            for obs in obs_list:
                action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
                actions.append(action.item())
            # Apply actions to speeds
            for idx, t in enumerate(trains):
                if actions[idx] == 0:
                    t['speed_kmh'] -= 5
                elif actions[idx] == 2:
                    t['speed_kmh'] += 5
                t['speed_kmh'] = np.clip(t['speed_kmh'], min_speed, max_speed)
            steps_since_action = 0
        else:
            # If not using AI, random speed changes (baseline)
            if not use_ai:
                for t in trains:
                    # random walk
                    t['speed_kmh'] += np.random.randint(-2, 3)
                    t['speed_kmh'] = np.clip(t['speed_kmh'], min_speed, max_speed)
        
        # 2. Update positions
        for t in trains:
            t['position_km'] += t['speed_kmh'] * time_step_hours
        
        # 3. Check for completed trains (position >= 20 km)
        new_trains = []
        for t in trains:
            if t['position_km'] >= 20.0:
                completed += 1
                # Insert a new train at position 0 with random speed
                new_t = {
                    'position_km': 0.0,
                    'speed_kmh': np.random.uniform(min_speed, max_speed),
                    'number': f"SIM{completed:03d}"
                }
                new_trains.append(new_t)
            else:
                new_trains.append(t)
        # Keep only trains inside section + newly inserted ones, but maintain total count
        # Actually we want to keep exactly num_trains trains. Some may have been removed and replaced.
        # The new_trains list now contains all trains that haven't finished plus new ones inserted.
        # But if multiple finished, we inserted multiple new ones, so length might exceed num_trains.
        # We need to ensure we always have num_trains trains in the section.
        # Simpler: when a train finishes, we remove it and add a new one at 0. So total remains constant.
        # In the loop above we added one new for each finished, so total remains constant.
        trains = new_trains
        trains.sort(key=lambda x: x['position_km'])
        
        # 4. Record throughput every minute (for plotting)
        if step % int(60 / dt_seconds) == 0:
            time_points.append((step+1) * time_step_hours)
            completed_counts.append(completed)
        
        steps_since_action += dt_seconds
    
    return completed, time_points, completed_counts

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="AI Train Control - Indian Railways", layout="wide")
st.title("🚉 Maximizing Section Throughput with AI (Indian Railways)")
st.markdown("---")

# Initialize session state for train data
if "trains" not in st.session_state:
    st.session_state.trains = generate_train_data()
    st.session_state.selected_idx = 0

# Load AI model
model = load_or_train_model()

# Create tabs
tab1, tab2 = st.tabs(["🚦 Driver Advisory", "📊 Throughput Analysis"])

with tab1:
    # Sidebar: Train selection
    st.sidebar.header("Select Your Train")
    train_options = [f"{t['number']} - {t['name']}" for t in st.session_state.trains]
    selected_train = st.sidebar.selectbox("Train Number/Name", train_options)
    selected_idx = train_options.index(selected_train)
    st.session_state.selected_idx = selected_idx

    # Display current train info
    train = st.session_state.trains[selected_idx]
    front_dist, back_dist = get_front_back(st.session_state.trains, selected_idx)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Speed", f"{train['speed_kmh']:.1f} km/h")
    with col2:
        st.metric("Train Ahead Distance", f"{front_dist:.2f} km" if front_dist else "No train")
    with col3:
        st.metric("Train Behind Distance", f"{back_dist:.2f} km" if back_dist else "No train")

    # Track Visualization
    st.subheader("Track View (0 to 20 km)")
    fig, ax = plt.subplots(figsize=(12, 2))

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=2)

    positions = [t['position_km'] for t in st.session_state.trains]
    ax.scatter(positions, [0]*len(positions), c='lightblue', s=30, alpha=0.6, zorder=1)

    ax.scatter(train['position_km'], 0, c='red', s=100, zorder=2, edgecolors='darkred', linewidth=2)
    ax.text(train['position_km'], 0.05, train['number'], ha='center', fontsize=9, fontweight='bold')

    if front_dist is not None and selected_idx < len(st.session_state.trains)-1:
        front_train = st.session_state.trains[selected_idx+1]
        ax.scatter(front_train['position_km'], 0, c='orange', s=70, zorder=2)
        ax.text(front_train['position_km'], -0.08, front_train['number'], ha='center', fontsize=8, color='orange')

    if back_dist is not None and selected_idx > 0:
        back_train = st.session_state.trains[selected_idx-1]
        ax.scatter(back_train['position_km'], 0, c='orange', s=70, zorder=2)
        ax.text(back_train['position_km'], -0.08, back_train['number'], ha='center', fontsize=8, color='orange')

    ax.set_xlim(0, 20)
    ax.set_ylim(-0.2, 0.2)
    ax.set_yticks([])
    ax.set_xlabel("Position (km)")
    ax.set_title("Train Positions (Selected in Red, Neighbors in Orange)")

    st.pyplot(fig)

    # AI Recommendation Button
    if st.button("🚦 Get AI Speed Recommendation"):
        min_speed, max_speed = 30, 100
        speed_norm = (train['speed_kmh'] - min_speed) / (max_speed - min_speed)
        front_norm = min((front_dist if front_dist else 10) / 10.0, 1.0)
        back_norm = min((back_dist if back_dist else 10) / 10.0, 1.0)
        obs = np.array([[speed_norm, front_norm, back_norm]], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)
        action = action.item()

        speed_change = {0: -5, 1: 0, 2: 5}[action]
        new_speed = train['speed_kmh'] + speed_change
        new_speed = np.clip(new_speed, min_speed, max_speed)

        # Simulate one time step (1 second) to update positions
        time_step = 1/3600
        for t in st.session_state.trains:
            t['position_km'] += t['speed_kmh'] * time_step
            t['position_km'] = min(t['position_km'], 20.0)

        st.session_state.trains[selected_idx]['speed_kmh'] = new_speed
        st.session_state.trains.sort(key=lambda x: x['position_km'])
        for i, t in enumerate(st.session_state.trains):
            if t['number'] == train['number']:
                st.session_state.selected_idx = i
                break

        new_front, new_back = get_front_back(st.session_state.trains, st.session_state.selected_idx)
        front_text = f"{new_front:.2f} km" if new_front is not None else "No train"
        back_text = f"{new_back:.2f} km" if new_back is not None else "No train"

        st.success("### 📢 Driver Advisory")
        st.info(
            f"**Train {train['number']} - {train['name']}**\n\n"
            f"🚄 **Recommended Speed:** {new_speed:.0f} km/h\n"
            f"🔹 Train ahead at {front_text}\n"
            f"🔸 Train behind at {back_text}"
        )

with tab2:
    st.header("Throughput Analysis")
    st.markdown("""
    Compare how many trains can pass through the 20 km section **with and without AI assistance**.
    The simulation recycles trains – whenever a train reaches 20 km, it is counted and a new train enters at 0 km.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        num_trains = st.number_input("Number of trains in section", min_value=5, max_value=50, value=20, step=1)
    with col2:
        duration = st.number_input("Simulation duration (hours)", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    with col3:
        sim_button = st.button("🚀 Run Simulation")

    if sim_button:
        with st.spinner("Running baseline (random) simulation..."):
            base_completed, base_times, base_counts = run_throughput_simulation(
                model, num_trains=num_trains, duration_hours=duration, use_ai=False
            )
        with st.spinner("Running AI simulation..."):
            ai_completed, ai_times, ai_counts = run_throughput_simulation(
                model, num_trains=num_trains, duration_hours=duration, use_ai=True
            )

        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Baseline (random)", f"{base_completed} trains", delta=None)
        with col2:
            st.metric("AI Controlled", f"{ai_completed} trains", 
                      delta=f"{ai_completed - base_completed} more")

        # Plot cumulative completed trains
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(base_times, base_counts, label='Baseline (random)', marker='o', linestyle='--')
        ax.plot(ai_times, ai_counts, label='AI Controlled', marker='s', linestyle='-')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Cumulative completed trains')
        ax.set_title('Throughput Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Bar chart of final counts
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(['Baseline', 'AI'], [base_completed, ai_completed], color=['skyblue', 'coral'])
        ax2.set_ylabel('Trains completed in {} hr'.format(duration))
        ax2.set_title('Total Throughput')
        for i, v in enumerate([base_completed, ai_completed]):
            ax2.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        st.pyplot(fig2)

        # Throughput per hour
        st.info(f"**Throughput (trains/hour):** Baseline = {base_completed/duration:.1f}  |  AI = {ai_completed/duration:.1f}")

st.markdown("---")
st.caption("AI model trained to maintain 1.5–3 km headway while maximizing speed. The throughput simulation shows how AI control can increase the number of trains passing through the section per hour.")
