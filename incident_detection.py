# incident_detection.py
import cv2
import numpy as np
import time

class IncidentDetector:
    """Module for detecting incidents like accidents or road blockages using computer vision."""
    
    def __init__(self, model_path="models/incident_detection_model.h5"):
        self.detection_threshold = 0.6
        # In a real implementation, load a pre-trained model
        print(f"Initializing Incident Detector with model: {model_path}")
        
        # Incident types that can be detected
        self.incident_types = ["accident", "roadblock", "construction", "flooding", "fallen_tree"]
        
    def preprocess_frame(self, frame):
        """Preprocess frame for model input."""
        # Resize the frame
        resized = cv2.resize(frame, (416, 416))
        # Normalize
        normalized = resized / 255.0
        return normalized
    
    def detect_incidents(self, frame):
        """
        Detect incidents in the given frame.
        
        Args:
            frame: Image frame from video stream
            
        Returns:
            List of dictionaries containing detected incidents with locations and confidence
        """
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame)
        
        # In a real implementation, this would use the loaded model for inference
        # This is a simulated detection for demonstration purposes
        incidents = []
        
        # Simulate detection (random for demonstration)
        if np.random.random() < 0.15:  # 15% chance to detect an incident
            incident_type = np.random.choice(self.incident_types)
            confidence = np.random.uniform(0.6, 0.95)
            
            # Create a location
            h, w = frame.shape[:2]
            x = int(np.random.uniform(0, w))
            y = int(np.random.uniform(0, h))
            
            incidents.append({
                "type": incident_type,
                "confidence": confidence,
                "location": (x, y),
                "timestamp": time.time()
            })
            
        return incidents
    
    def process_video_stream(self, video_source):
        """Process video stream and detect incidents."""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return []
            
        detections = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process every 5th frame to simulate real-time processing
            if frame_count % 5 == 0:
                incidents = self.detect_incidents(frame)
                if incidents:
                    detections.extend(incidents)
                    print(f"Detected incident: {incidents[0]['type']} with {incidents[0]['confidence']:.2f} confidence")
            
            frame_count += 1
                
            # Break after processing enough frames for demonstration
            if frame_count > 50:
                break
                
            time.sleep(0.05)  # Simulate processing time
            
        cap.release()
        return detections
    
    def analyze_incident_severity(self, incident):
        """Analyze the severity of a detected incident."""
        severity_levels = {
            "accident": 0.9,
            "roadblock": 0.7,
            "construction": 0.5,
            "flooding": 0.8,
            "fallen_tree": 0.6
        }
        
        base_severity = severity_levels.get(incident["type"], 0.5)
        # Adjust severity based on confidence
        adjusted_severity = base_severity * incident["confidence"]
        
        if adjusted_severity > 0.8:
            return "HIGH"
        elif adjusted_severity > 0.5:
            return "MEDIUM"
        else:
            return "LOW"