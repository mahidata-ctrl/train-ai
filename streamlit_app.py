# streamlit_app.py
import streamlit as st
import time
import pandas as pd
from datetime import datetime
from train_api_client import IndianRailwaySimulator
from optimizer.genetic_optimizer import Train, SectionOptimizer

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="AI Train Section Optimizer",
    page_icon="🚆",
    layout="wide"
)

# ============ INITIALISE SIMULATOR ============
@st.cache_resource
def get_simulator():
    return IndianRailwaySimulator()

sim = get_simulator()

# ============ SESSION STATE ============
if 'active_trains' not in st.session_state:
    st.session_state.active_trains = []
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# ============ HELPER FUNCTIONS ============
def update_train_positions():
    """Move trains forward (called every 5 seconds)"""
    if st.session_state.active_trains:
        for train in st.session_state.active_trains:
            # Simulate movement: increase position
            train.position += train.speed * 0.01  # small step for demo
            # Keep within section length
            if train.position > 200:
                train.position = 200

def generate_notifications():
    """Create driver advice messages from current train positions"""
    trains = st.session_state.active_trains
    if not trains:
        return []
    sorted_trains = sorted(trains, key=lambda t: t.position)
    notifications = []
    for i, train in enumerate(sorted_trains):
        dist_ahead = sorted_trains[i+1].position - train.position if i < len(sorted_trains)-1 else None
        dist_behind = train.position - sorted_trains[i-1].position if i > 0 else None
        
        message = f"MAINTAIN {round(train.speed,1)} km/h."
        if dist_ahead is not None:
            message += f" TRAIN {sorted_trains[i+1].id} {round(dist_ahead,1)}km AHEAD."
        if dist_behind is not None:
            message += f" TRAIN {sorted_trains[i-1].id} {round(dist_behind,1)}km BEHIND."
        
        notifications.append({
            "train_id": train.id,
            "advised_speed": round(train.speed, 1),
            "train_ahead": sorted_trains[i+1].id if i < len(sorted_trains)-1 else "None",
            "distance_ahead": round(dist_ahead, 1) if dist_ahead is not None else 0,
            "train_behind": sorted_trains[i-1].id if i > 0 else "None",
            "distance_behind": round(dist_behind, 1) if dist_behind is not None else 0,
            "message": message,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    return notifications

# ============ UI – HEADER ============
st.title("🚆 Indian Railways – AI Section Throughput Optimizer")
st.markdown("---")

# ============ SIDEBAR – TRAIN INPUT ============
with st.sidebar:
    st.header("🔍 1. Fetch Live Train Data")
    train_number = st.text_input("Enter Train Number (e.g., 12051, 12301)", key="train_input")
    if st.button("Get Live Status"):
        if train_number:
            info = sim.get_train_info(train_number)
            if "error" in info:
                st.error(info["error"])
            else:
                live = sim.get_live_status(train_number)
                st.success(f"**{live['train_name']}** ({live['train_number']})")
                col1, col2 = st.columns(2)
                col1.metric("Speed", f"{live['current_speed']} km/h")
                col2.metric("Delay", f"{live['delay_minutes']} min")
                st.metric("Position", f"{live['distance_covered']} / {live['total_distance']} km")
                st.caption(f"Last updated: {live['last_updated']}")
        else:
            st.warning("Enter a train number")

    st.markdown("---")
    st.header("🧠 2. Run AI Optimization")
    if st.button("🚀 START OPTIMIZATION", type="primary"):
        # Get 5 trains from simulator
        trains_data = sim.get_all_trains_in_section()
        if len(trains_data) < 2:
            # Fallback dummy trains
            trains_data = [
                {"train_number": "12301", "current_speed": 110, "priority": 1, "distance_covered": 10},
                {"train_number": "12627", "current_speed": 90, "priority": 2, "distance_covered": 25},
                {"train_number": "12951", "current_speed": 130, "priority": 1, "distance_covered": 40},
                {"train_number": "11013", "current_speed": 80, "priority": 3, "distance_covered": 55},
                {"train_number": "12245", "current_speed": 85, "priority": 2, "distance_covered": 70},
            ]
        
        # Create Train objects
        trains = []
        for i, t in enumerate(trains_data[:5]):
            train = Train(
                id=t.get('train_number', f'DUMMY{i}'),
                base_speed=t.get('current_speed', 80),
                priority=t.get('priority', 3),
                position=t.get('distance_covered', i*15)
            )
            trains.append(train)
        
        # Run Genetic Algorithm
        optimizer = SectionOptimizer(trains, section_length=200, safety_distance=8)
        optimal_speeds = optimizer.optimize()
        for i, train in enumerate(trains):
            train.speed = optimal_speeds[i]
        
        st.session_state.active_trains = trains
        st.session_state.simulation_running = True
        st.session_state.last_update = time.time()
        st.success(f"✅ Optimisation complete! {len(trains)} trains active.")
        st.rerun()

# ============ MAIN PANEL – LIVE NOTIFICATIONS ============
st.header("📡 3. LIVE DRIVER ADVISORY NOTIFICATIONS")

# Auto-refresh every 5 seconds
if st.session_state.simulation_running:
    # Update positions every 5 seconds
    if time.time() - st.session_state.last_update > 5:
        update_train_positions()
        st.session_state.last_update = time.time()
        st.rerun()

# Display notifications
if st.session_state.active_trains:
    notifications = generate_notifications()
    
    # Show as cards
    for notif in notifications:
        with st.container():
            cols = st.columns([1, 2, 2, 1])
            with cols[0]:
                st.markdown(f"### 🚂 {notif['train_id']}")
            with cols[1]:
                st.markdown(f"**Advised Speed:**  \n⚡ {notif['advised_speed']} km/h")
            with cols[2]:
                st.markdown(f"**Train Ahead:** {notif['train_ahead']}  \n📏 {notif['distance_ahead']} km")
            with cols[3]:
                st.markdown(f"**Train Behind:** {notif['train_behind']}  \n📏 {notif['distance_behind']} km")
            st.info(f"💬 {notif['message']}")
            st.caption(f"🕒 {notif['timestamp']}")
            st.divider()
    
    # Option to stop simulation
    if st.button("⏹️ Stop Simulation"):
        st.session_state.simulation_running = False
        st.session_state.active_trains = []
        st.rerun()
else:
    st.info("👆 Start the AI optimization from the sidebar to see live driver notifications.")

# ============ FOOTER ============
st.markdown("---")
st.caption("AI-Powered Section Throughput Maximizer – Indian Railways Simulation (No API Key Required)")
