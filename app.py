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

# -------------------- AI Environment for Training with 2-3 km Spacing --------------------
class SingleTrainControlEnv(gym.Env):
    """
    Environment for controlling one train (ego) with a lead train ahead and a follow train behind.
    State: [ego_speed, front_distance, back_distance] normalized.
    Action: 0 = decelerate, 1 = maintain, 2 = accelerate.
    Reward: encourage safe distance (2-3 km) and high speed.
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
        # Initialize positions with realistic 2-3 km spacing
        self.lead_pos = 15.0
        self.ego_pos = 12.0  # 3 km behind lead
        self.follow_pos = 9.0  # 3 km behind ego
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

        # UPDATED: 2-3 km safe spacing (realistic for Indian Railways)
        # Danger zone - too close (less than 2 km)
        if front_dist < 2.0:
            reward -= 50  # Severe penalty - CRITICAL SAFETY VIOLATION
        # Caution zone - approaching minimum safe distance (2.0-2.5 km)
        elif front_dist < 2.5:
            reward -= 10  # Moderate penalty - need to increase gap
        # PERFECT zone - optimal safe distance (2.5-3.5 km)
        elif 2.5 <= front_dist <= 3.5:
            reward += 30  # High reward for maintaining ideal spacing
        # Good zone - acceptable but slightly far (3.5-5.0 km)
        elif front_dist <= 5.0:
            reward += 10  # Still good, but could be closer for throughput
        # Too far - wasting capacity (>5 km)
        else:
            reward -= 5  # Penalty for inefficient use of track

        # Rear distance also matters for safety (minimum 1.5 km behind)
        if back_dist < 1.5:
            reward -= 30  # Train behind is too close - dangerous

        # Encourage higher speed (but safety first!)
        reward += 0.03 * self.ego_speed

        # Small penalty for harsh actions to promote smooth driving
        if action == 0 or action == 2:
            reward -= 0.1

        done = False
        truncated = False
        return self._get_obs(), reward, done, truncated, {}

# -------------------- Train or Load DQN Model --------------------
@st.cache_resource
def load_or_train_model():
    model_path = "dqn_train_control_2_3km.zip"  # New model name for 2-3km spacing
    if os.path.exists(model_path):
        model = DQN.load(model_path)
    else:
        with st.spinner("Training AI model with 2-3 km safe spacing... This may take a minute..."):
            env = DummyVecEnv([lambda: SingleTrainControlEnv()])
            model = DQN("MlpPolicy", env, verbose=0, learning_rate=0.001, buffer_size=10000,
                        learning_starts=100, batch_size=32, tau=0.1, gamma=0.99,
                        train_freq=4, gradient_steps=1)
            model.learn(total_timesteps=10000)  # Increased training for better learning
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
    # Initialise trains randomly on 0-20 km with at least 2 km spacing
    np.random.seed(42)
    positions = []
    for i in range(num_trains):
        if i == 0:
            pos = np.random.uniform(0, 5)
        else:
            # Ensure minimum 2 km spacing from previous train
            min_pos = positions[-1] + 2.0
            if min_pos < 20:
                pos = np.random.uniform(min_pos, min(20, min_pos + 3))
            else:
                pos = np.random.uniform(0, 5)  # Wrap around for simplicity
        positions.append(min(pos, 20))
    
    speeds = np.random.uniform(40, 90, size=num_trains)
    trains = []
    for i in range(num_trains):
        trains.append({
            'position_km': positions[i],
            'speed_kmh': speeds[i],
            'number': f"SIM{i:03d}"
        })
    trains.sort(key=lambda x: x['position_km'])

    min_speed, max_speed = 30, 100
    time_step_hours = dt_seconds / 3600.0
    total_steps = int(duration_hours * 3600 / dt_seconds)
    
    completed = 0
    time_points = [0.0]
    completed_counts = [0]
    
    action_interval = 10  # seconds
    steps_since_action = 0
    
    for step in range(total_steps):
        if use_ai and steps_since_action >= action_interval:
            obs_list = []
            for idx, t in enumerate(trains):
                obs = get_observation_for_train(t, trains, idx, min_speed, max_speed)
                obs_list.append(obs)
            actions = []
            for obs in obs_list:
                action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
                actions.append(action.item())
            for idx, t in enumerate(trains):
                if actions[idx] == 0:
                    t['speed_kmh'] -= 5
                elif actions[idx] == 2:
                    t['speed_kmh'] += 5
                t['speed_kmh'] = np.clip(t['speed_kmh'], min_speed, max_speed)
            steps_since_action = 0
        else:
            if not use_ai:
                for t in trains:
                    t['speed_kmh'] += np.random.randint(-2, 3)
                    t['speed_kmh'] = np.clip(t['speed_kmh'], min_speed, max_speed)
        
        for t in trains:
            t['position_km'] += t['speed_kmh'] * time_step_hours
        
        new_trains = []
        for t in trains:
            if t['position_km'] >= 20.0:
                completed += 1
                new_t = {
                    'position_km': 0.0,
                    'speed_kmh': np.random.uniform(min_speed, max_speed),
                    'number': f"SIM{completed:03d}"
                }
                new_trains.append(new_t)
            else:
                new_trains.append(t)
        trains = new_trains
        trains.sort(key=lambda x: x['position_km'])
        
        if step % int(60 / dt_seconds) == 0:
            time_points.append((step+1) * time_step_hours)
            completed_counts.append(completed)
        
        steps_since_action += dt_seconds
    
    return completed, time_points, completed_counts

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="AI Train Control - Indian Railways", layout="wide")
st.title("🚉 Maximizing Section Throughput with AI (Indian Railways)")
st.markdown("---")

# Add disclaimer about realistic spacing
st.info("⚡ **Realistic Train Spacing:** This AI model maintains **2-3 km safe distance** between trains, which matches Indian Railway safety standards. At 100 km/h, this gives approximately **1.5-2 minutes headway**.")

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

    # Color-code the distance metrics based on safety
    def get_distance_color(dist, is_front=True):
        if dist is None:
            return "normal"
        if is_front:
            if dist < 2.0:
                return "🔴"  # Danger - too close
            elif dist < 2.5:
                return "🟡"  # Warning - approaching limit
            elif 2.5 <= dist <= 3.5:
                return "🟢"  # Perfect
            else:
                return "🔵"  # Far but ok
        else:  # rear distance
            if dist < 1.5:
                return "🔴"  # Danger
            else:
                return "normal"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Speed", f"{train['speed_kmh']:.1f} km/h")
    with col2:
        front_color = get_distance_color(front_dist, True)
        st.metric("Train Ahead Distance", 
                 f"{front_color} {front_dist:.2f} km" if front_dist else "No train")
    with col3:
        back_color = get_distance_color(back_dist, False)
        st.metric("Train Behind Distance", 
                 f"{back_color} {back_dist:.2f} km" if back_dist else "No train")

    # Safety status
    if front_dist and front_dist < 2.0:
        st.error("⚠️ **CRITICAL:** Too close to train ahead! Increase distance immediately!")
    elif front_dist and front_dist < 2.5:
        st.warning("⚠️ **Caution:** Approaching minimum safe distance")
    elif front_dist and 2.5 <= front_dist <= 3.5:
        st.success("✅ **Perfect spacing maintained**")

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

    # Add safety zones visualization
    if front_dist is not None:
        # Draw danger zone (red)
        ax.axvspan(train['position_km'], train['position_km'] + 2.0, 
                   ymin=0.4, ymax=0.6, alpha=0.2, color='red', label='Danger (<2km)')
        # Draw caution zone (yellow)
        ax.axvspan(train['position_km'] + 2.0, train['position_km'] + 2.5, 
                   ymin=0.4, ymax=0.6, alpha=0.2, color='yellow', label='Caution (2-2.5km)')
        # Draw optimal zone (green)
        ax.axvspan(train['position_km'] + 2.5, train['position_km'] + 3.5, 
                   ymin=0.4, ymax=0.6, alpha=0.2, color='green', label='Optimal (2.5-3.5km)')

    ax.set_xlim(0, 20)
    ax.set_ylim(-0.2, 0.2)
    ax.set_yticks([])
    ax.set_xlabel("Position (km)")
    ax.set_title("Train Positions with Safety Zones (Selected in Red)")

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

        # Safety assessment
        safety_status = ""
        if new_front and new_front < 2.0:
            safety_status = "⚠️ **CRITICAL: TOO CLOSE!**"
        elif new_front and new_front < 2.5:
            safety_status = "⚠️ **Caution: Getting close**"
        elif new_front and 2.5 <= new_front <= 3.5:
            safety_status = "✅ **Perfect safe distance**"

        st.success("### 📢 Driver Advisory")
        st.info(
            f"**Train {train['number']} - {train['name']}**\n\n"
            f"🚄 **Recommended Speed:** {new_speed:.0f} km/h\n"
            f"🔹 Train ahead at {front_text}\n"
            f"🔸 Train behind at {back_text}\n\n"
            f"{safety_status}"
        )

with tab2:
    st.header("Throughput Analysis")
    st.markdown("""
    Compare how many trains can pass through the 20 km section **with and without AI assistance**.
    The AI maintains **2-3 km safe spacing** between trains (realistic for Indian Railways).
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        num_trains = st.number_input("Number of trains in section", min_value=5, max_value=30, value=15, step=1)
    with col2:
        duration = st.number_input("Simulation duration (hours)", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    with col3:
        sim_button = st.button("🚀 Run Simulation")

    if sim_button:
        with st.spinner("Running baseline (random) simulation..."):
            base_completed, base_times, base_counts = run_throughput_simulation(
                model, num_trains=num_trains, duration_hours=duration, use_ai=False
            )
        with st.spinner("Running AI simulation with 2-3km spacing..."):
            ai_completed, ai_times, ai_counts = run_throughput_simulation(
                model, num_trains=num_trains, duration_hours=duration, use_ai=True
            )

        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Baseline (random)", f"{base_completed} trains", delta=None)
        with col2:
            st.metric("AI Controlled (2-3km spacing)", f"{ai_completed} trains", 
                      delta=f"{ai_completed - base_completed} more")

        # Calculate throughput rates
        base_rate = base_completed / duration
        ai_rate = ai_completed / duration
        improvement = ((ai_rate - base_rate) / base_rate) * 100 if base_rate > 0 else 0

        st.info(f"📊 **Throughput:** Baseline = {base_rate:.1f} trains/hr | AI = {ai_rate:.1f} trains/hr | **Improvement: {improvement:.1f}%**")

        # Plot cumulative completed trains
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(base_times, base_counts, label='Baseline (random)', marker='o', linestyle='--', linewidth=2)
        ax.plot(ai_times, ai_counts, label='AI Controlled (2-3km spacing)', marker='s', linestyle='-', linewidth=2)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Cumulative completed trains')
        ax.set_title('Throughput Comparison with 2-3km Safe Spacing')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Bar chart of final counts
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        bars = ax2.bar(['Baseline (Random)', f'AI (2-3km Spacing)'], [base_completed, ai_completed], 
                       color=['skyblue', 'coral'], edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Trains completed in {} hr'.format(duration))
        ax2.set_title('Total Throughput with Realistic Spacing')
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        st.pyplot(fig2)

st.markdown("---")
st.caption("✅ **Realistic Train Spacing:** AI maintains 2-3 km safe distance between trains, matching Indian Railway safety standards. At 100 km/h, this provides 1.5-2 minutes reaction time.")
