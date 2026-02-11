# train_api_client.py
# SIMULATED LIVE INDIAN RAILWAYS DATA – NO API KEY, FULLY SELF-CONTAINED
import random
from datetime import datetime

class IndianRailwaySimulator:
    """
    Simulates live train data using a built-in train database.
    No external files, no API keys. Ready to run.
    """
    
    def __init__(self):
        # ============= INDIAN RAILWAYS TRAIN DATASET (EMBEDDED) =============
        self.train_db = {
            "12051": {
                "train_name": "DRR HWH JANSHATABDI",
                "from_stn": "DRR",
                "to_stn": "HWH",
                "distance": 100,
                "base_speed": 110,
                "priority": 1
            },
            "12301": {
                "train_name": "HOWRAH RAJDHANI",
                "from_stn": "NDLS",
                "to_stn": "HWH",
                "distance": 1450,
                "base_speed": 130,
                "priority": 1
            },
            "12627": {
                "train_name": "KARNATAKA EXPRESS",
                "from_stn": "SBC",
                "to_stn": "NDLS",
                "distance": 2400,
                "base_speed": 90,
                "priority": 2
            },
            "12951": {
                "train_name": "MUMBAI RAJDHANI",
                "from_stn": "MMCT",
                "to_stn": "NDLS",
                "distance": 1384,
                "base_speed": 130,
                "priority": 1
            },
            "11013": {
                "train_name": "COIMBATORE EXP",
                "from_stn": "LTT",
                "to_stn": "CBE",
                "distance": 1250,
                "base_speed": 80,
                "priority": 3
            },
            "12245": {
                "train_name": "YESVANTPUR EXP",
                "from_stn": "YPR",
                "to_stn": "NDLS",
                "distance": 2360,
                "base_speed": 85,
                "priority": 2
            },
            "16336": {
                "train_name": "GANDHIDHAM EXP",
                "from_stn": "NCJ",
                "to_stn": "GIMB",
                "distance": 2100,
                "base_speed": 70,
                "priority": 3
            },
            "12801": {
                "train_name": "PURUSHOTTAM EXP",
                "from_stn": "NDLS",
                "to_stn": "PURI",
                "distance": 2000,
                "base_speed": 85,
                "priority": 2
            }
        }
        # ====================================================================
        
        # Simulated "live" positions and delays
        self.live_state = {}
        self.last_update = datetime.now()
        self._init_live_state()
    
    def _init_live_state(self):
        """Initialise random positions/delays for all trains"""
        import random
        for tn, info in self.train_db.items():
            self.live_state[tn] = {
                "current_distance": random.uniform(0, info["distance"] * 0.3),
                "delay": random.randint(-5, 15),
                "speed": info["base_speed"],
                "last_update": datetime.now()
            }
    
    def _update_positions(self):
        """Move trains forward slightly – gives 'live' feel"""
        now = datetime.now()
        elapsed = (now - self.last_update).total_seconds() / 3600.0  # hours
        for tn, state in self.live_state.items():
            if state["current_distance"] < self.train_db[tn]["distance"]:
                # Move at current speed (km/h) * time (hours) * small factor for smoothness
                state["current_distance"] += state["speed"] * elapsed * 0.5
                if state["current_distance"] > self.train_db[tn]["distance"]:
                    state["current_distance"] = self.train_db[tn]["distance"]
        self.last_update = now
    
    def get_train_info(self, train_number):
        """Return static schedule info for a train"""
        tn = str(train_number)
        if tn in self.train_db:
            info = self.train_db[tn].copy()
            info["train_number"] = tn
            return info
        else:
            return {"error": f"Train {tn} not found in database"}
    
    def get_live_status(self, train_number):
        """Return simulated live running status"""
        tn = str(train_number)
        if tn not in self.train_db:
            return {"error": f"Train {tn} not found"}
        
        self._update_positions()
        info = self.train_db[tn]
        state = self.live_state[tn]
        
        return {
            "train_number": tn,
            "train_name": info["train_name"],
            "current_speed": round(state["speed"], 1),
            "distance_covered": round(state["current_distance"], 2),
            "total_distance": info["distance"],
            "delay_minutes": state["delay"],
            "status": "Running" if state["current_distance"] < info["distance"] else "Arrived",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_all_trains_in_section(self):
        """Return a list of trains currently active (for simulation)"""
        self._update_positions()
        trains = []
        for tn in self.train_db:
            status = self.get_live_status(tn)
            if status["status"] == "Running":
                trains.append(status)
        return trains[:5]  # limit for demo

# For standalone testing
if __name__ == "__main__":
    client = IndianRailwaySimulator()
    print(client.get_train_info(12051))
    print(client.get_live_status(12301))
