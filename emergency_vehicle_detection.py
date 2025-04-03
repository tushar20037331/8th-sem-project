# emergency_vehicle_detection.py
import cv2
import numpy as np
import time

class EmergencyVehicleDetector:
    """Module for detecting emergency vehicles in video streams using computer vision."""
    
    def __init__(self, model_path="models/emergency_vehicle_model.h5"):
        self.detection_threshold = 0.7
        # In a real implementation, load a pre-trained model
        print(f"Initializing Emergency Vehicle Detector with model: {model_path}")
        # This is a placeholder for model initialization
        self.classes = ["car", "bus", "truck", "ambulance", "police_car", "fire_truck"]
        
    def preprocess_frame(self, frame):
        """Preprocess frame for model input."""
        # Resize the frame
        resized = cv2.resize(frame, (416, 416))
        # Normalize
        normalized = resized / 255.0
        return normalized
    
    def detect_emergency_vehicles(self, frame):
        """
        Detect emergency vehicles in the given frame.
        
        Args:
            frame: Image frame from video stream
            
        Returns:
            List of dictionaries containing detected emergency vehicles with bounding boxes and confidence
        """
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame)
        
        # In a real implementation, this would use the loaded model for inference
        # This is a simulated detection for demonstration purposes
        emergency_vehicles = []
        
        # Simulate detection (random for demonstration)
        if np.random.random() < 0.2:  # 20% chance to detect an emergency vehicle
            vehicle_type = np.random.choice(["ambulance", "police_car", "fire_truck"])
            confidence = np.random.uniform(0.7, 0.99)
            
            # Create a bounding box
            h, w = frame.shape[:2]
            x1 = int(np.random.uniform(0, w/2))
            y1 = int(np.random.uniform(0, h/2))
            x2 = int(x1 + np.random.uniform(100, w/2))
            y2 = int(y1 + np.random.uniform(100, h/2))
            
            emergency_vehicles.append({
                "type": vehicle_type,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            })
            
        return emergency_vehicles
    
    def process_video_stream(self, video_source):
        """Process video stream and detect emergency vehicles."""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return []
            
        detections = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            vehicles = self.detect_emergency_vehicles(frame)
            if vehicles:
                detections.extend(vehicles)
                print(f"Detected emergency vehicle: {vehicles[0]['type']} with {vehicles[0]['confidence']:.2f} confidence")
                
            # Break after a few frames for demonstration
            if len(detections) > 5:
                break
                
            time.sleep(0.1)  # Simulate processing time
            
        cap.release()
        return detections

