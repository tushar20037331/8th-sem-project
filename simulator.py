# simulator.py
import time
import numpy as np
import cv2
import threading
from traffic_signal_controller import SignalState

class TrafficSimulator:
    """A simple simulator for testing the traffic management system."""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.junctions = {}
        self.emergency_vehicles = []
        self.incidents = []
        self.running = False
        self.sim_thread = None
        
    def add_junction(self, junction_id, position, controller):
        """Add a junction to the simulation."""
        self.junctions[junction_id] = {
            "position": position,
            "controller": controller.junctions.get(junction_id)
        }
        
    def add_emergency_vehicle(self, vehicle_type, start_position, route):
        """Add an emergency vehicle to the simulation."""
        self.emergency_vehicles.append({
            "type": vehicle_type,
            "position": start_position,
            "route": route,
            "speed": 5,
            "active": True
        })
        
    def add_incident(self, incident_type, position, severity):
        """Add an incident to the simulation."""
        self.incidents.append({
            "type": incident_type,
            "position": position,
            "severity": severity,
            "time": time.time()
        })
        
    def start(self):
        """Start the simulation."""
        if self.running:
            return
            
        self.running = True
        self.sim_thread = threading.Thread(target=self._simulation_loop)
        self.sim_thread.daemon = True
        self.sim_thread.start()
        print("Traffic Simulator started")
        
    def stop(self):
        """Stop the simulation."""
        self.running = False
        if self.sim_thread:
            self.sim_thread.join(timeout=1.0)
        print("Traffic Simulator stopped")
        
    def _simulation_loop(self):
        """Main simulation loop."""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Update emergency vehicles
            self._update_emergency_vehicles(dt)
            
            # Update the display
            self._render_frame()
            
            # Show the frame
            cv2.imshow("Traffic Simulation", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.05)  # 20 FPS approx
            
        cv2.destroyAllWindows()
        
    def _update_emergency_vehicles(self, dt):
        """Update the positions of emergency vehicles."""
        for vehicle in self.emergency_vehicles:
            if not vehicle["active"]:
                continue
                
            # Simple movement along route
            if vehicle["route"]:
                target = vehicle["route"][0]
                
                # Calculate direction and move
                dx = target[0] - vehicle["position"][0]
                dy = target[1] - vehicle["position"][1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < vehicle["speed"] * dt:
                    # Reached waypoint
                    vehicle["position"] = target
                    vehicle["route"].pop(0)
                    
                    # If at junction, request emergency priority
                    for j_id, junction in self.junctions.items():
                        if np.linalg.norm(np.array(vehicle["position"]) - np.array(junction["position"])) < 50:
                            # Simplified direction calculation
                            approach_dir = 0  # Would be calculated based on approach angle
                            target_dir = 2    # Would be calculated based on next waypoint
                            
                            # Request priority
                            self._request_emergency_priority(j_id, approach_dir, target_dir)
                else:
                    # Move toward target
                    vehicle["position"] = (
                        vehicle["position"][0] + dx/distance * vehicle["speed"] * dt,
                        vehicle["position"][1] + dy/distance * vehicle["speed"] * dt
                    )
            else:
                # Reached end of route
                vehicle["active"] = False
                
    def _request_emergency_priority(self, junction_id, approach_dir, target_dir):
        """Request emergency priority at a junction."""
        if junction_id in self.junctions and self.junctions[junction_id]["controller"]:
            print(f"Requesting emergency priority at junction {junction_id}")
            # This would communicate with the actual controller
            # For simulation, we directly handle it via the junction object
            self.junctions[junction_id]["controller"].set_emergency_route(approach_dir, target_dir)
            
    def _render_frame(self):
        """Render the simulation frame."""
        # Clear the frame
        self.frame.fill(0)
        
        # Draw junctions
        for j_id, junction in self.junctions.items():
            pos = junction["position"]
            cv2.circle(self.frame, (int(pos[0]), int(pos[1])), 30, (100, 100, 100), -1)
            
            # Draw signals if controller is available
            if junction["controller"]:
                for i, signal in enumerate(junction["controller"].signals):
                    # Calculate signal position around junction
                    angle = i * np.pi/2  # 4 signals at 90 degree intervals
                    sig_x = int(pos[0] + np.cos(angle) * 30)
                    sig_y = int(pos[1] + np.sin(angle) * 30)
                    
                    # Draw signal with appropriate color
                    color = {
                        SignalState.RED: (0, 0, 255),
                        SignalState.YELLOW: (0, 255, 255),
                        SignalState.GREEN: (0, 255, 0)
                    }[signal.state]
                    
                    cv2.circle(self.frame, (sig_x, sig_y), 8, color, -1)
                    
        # Draw emergency vehicles
        for vehicle in self.emergency_vehicles:
            if not vehicle["active"]:
                continue
                
            pos = vehicle["position"]
            
            # Different colors for different vehicle types
            color = {
                "ambulance": (255, 255, 255),
                "police_car": (0, 0, 255),
                "fire_truck": (0, 0, 255)
            }.get(vehicle["type"], (0, 255, 255))
            
            cv2.rectangle(self.frame,
                         (int(pos[0])-10, int(pos[1])-5),
                         (int(pos[0])+10, int(pos[1])+5),
                         color, -1)
                         
        # Draw incidents
        for incident in self.incidents:
            pos = incident["position"]
            
            # Different colors for different severity
            color = {
                "HIGH": (0, 0, 255),
                "MEDIUM": (0, 165, 255),
                "LOW": (0, 255, 255)
            }.get(incident["severity"], (0, 255, 0))
            
            cv2.circle(self.frame, (int(pos[0]), int(pos[1])), 15, color, -1)
            
            # Add text to show incident type
            cv2.putText(self.frame, incident["type"],
                       (int(pos[0])-30, int(pos[1])-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)