# app.py
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import time

from train_api_client import IndianRailwaySimulator
from optimizer.genetic_optimizer import Train, SectionOptimizer

app = Flask(__name__, template_folder='optimizer/templates')
app.config['SECRET_KEY'] = 'railways-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# ============ GLOBAL SIMULATION STATE ============
rail_client = IndianRailwaySimulator()
active_trains = []
optimizer_thread = None
notification_running = False

# ============ ROUTES ============
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/get_train_data', methods=['POST'])
def get_train_data():
    """Requirement 1: Input train number -> fetch live data"""
    train_identifier = request.form.get('train_identifier', '').strip()
    if not train_identifier:
        return jsonify({'success': False, 'message': 'Enter train number'})
    
    # Get static info
    info = rail_client.get_train_info(train_identifier)
    if 'error' in info:
        return jsonify({'success': False, 'message': info['error']})
    
    # Get live status
    live = rail_client.get_live_status(train_identifier)
    
    return jsonify({
        'success': True,
        'train_details': info,
        'live_status': live
    })

@app.route('/start_simulation')
def start_simulation():
    """Requirement 2 & 3: Run optimizer & start driver notifications"""
    global active_trains, notification_running, optimizer_thread
    
    # Get 5 running trains from simulator
    trains_data = rail_client.get_all_trains_in_section()
    if len(trains_data) < 2:
        # fallback: create dummy trains
        trains_data = [
            {"train_number": "12301", "current_speed": 110, "priority": 1, "distance_covered": 10},
            {"train_number": "12627", "current_speed": 90, "priority": 2, "distance_covered": 25},
            {"train_number": "12951", "current_speed": 130, "priority": 1, "distance_covered": 40},
            {"train_number": "11013", "current_speed": 80, "priority": 3, "distance_covered": 55},
            {"train_number": "12245", "current_speed": 85, "priority": 2, "distance_covered": 70},
        ]
    
    # Convert to Train objects
    active_trains = []
    for i, t in enumerate(trains_data[:5]):  # max 5 trains
        train = Train(
            id=t.get('train_number', f'DUMMY{i}'),
            base_speed=t.get('current_speed', 80),
            priority=t.get('priority', 3),
            position=t.get('distance_covered', i*15)
        )
        active_trains.append(train)
    
    # Run Genetic Algorithm optimization
    optimizer = SectionOptimizer(active_trains, section_length=200, safety_distance=8)
    optimal_speeds = optimizer.optimize()
    for i, train in enumerate(active_trains):
        train.speed = optimal_speeds[i]
    
    # Start notification thread (if not already running)
    if not notification_running:
        notification_running = True
        def notify_loop():
            while notification_running:
                socketio.emit('driver_advice', generate_notifications())
                time.sleep(4)  # send every 4 seconds
        optimizer_thread = threading.Thread(target=notify_loop)
        optimizer_thread.daemon = True
        optimizer_thread.start()
    
    return jsonify({'status': 'simulation_started', 'trains': len(active_trains)})

def generate_notifications():
    """Requirement 3: Build message with speed, train ahead/behind, distances"""
    global active_trains
    # sort by position (ascending)
    sorted_trains = sorted(active_trains, key=lambda t: t.position)
    notifications = []
    for i, train in enumerate(sorted_trains):
        # calculate distances
        dist_ahead = sorted_trains[i+1].position - train.position if i < len(sorted_trains)-1 else None
        dist_behind = train.position - sorted_trains[i-1].position if i > 0 else None
        
        message = f"MAINTAIN {round(train.speed,1)} km/h."
        if dist_ahead is not None:
            message += f" TRAIN {sorted_trains[i+1].id} {round(dist_ahead,1)}km AHEAD."
        if dist_behind is not None:
            message += f" TRAIN {sorted_trains[i-1].id} {round(dist_behind,1)}km BEHIND."
        
        notifications.append({
            'train_id': train.id,
            'advised_speed': round(train.speed, 1),
            'train_ahead_id': sorted_trains[i+1].id if i < len(sorted_trains)-1 else None,
            'distance_ahead': round(dist_ahead, 1) if dist_ahead is not None else 0,
            'train_behind_id': sorted_trains[i-1].id if i > 0 else None,
            'distance_behind': round(dist_behind, 1) if dist_behind is not None else 0,
            'message': message
        })
    return {'timestamp': time.time(), 'notifications': notifications}

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_response', {'data': 'Connected to Section Controller'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)