"""
Cricket Shot Classifier using LSTM on Pose Sequences
Classifies 10 cricket batting shots from pose landmarks
"""

import numpy as np
from collections import deque
import pickle
import os


class CricketShotClassifier:
    """
    Cricket shot classification using LSTM on pose landmark sequences.
    Uses a simple rule-based approach for demo (can be replaced with trained LSTM).
    """
    
    def __init__(self, sequence_length=30):
        """
        Initialize the cricket shot classifier.
        
        Args:
            sequence_length: Number of frames to analyze for shot classification
        """
        self.sequence_length = sequence_length
        self.pose_sequence = deque(maxlen=sequence_length)
        
        # 10 Cricket Shot Classes (from CricShot10 dataset)
        self.shot_classes = [
            'Cover Drive',
            'Defense', 
            'Flick',
            'Hook',
            'Late Cut',
            'Lofted',
            'Pull',
            'Square Cut',
            'Straight Drive',
            'Sweep'
        ]
        
        # Pose-based shot detection rules (simplified heuristics)
        # These will be replaced with actual LSTM model predictions
        self.initialize_model()
        
    def initialize_model(self):
        """
        Initialize the classification model.
        For quick demo, uses rule-based heuristics.
        Can be replaced with trained LSTM/Transformer model.
        """
        # Check if pre-trained model exists
        model_path = os.path.join(os.path.dirname(__file__), 'cricket_lstm_model.pkl')
        
        if os.path.exists(model_path):
            # Load pre-trained model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.use_ml_model = True
            print("✅ Loaded pre-trained cricket shot classifier")
        else:
            # Use rule-based heuristics for demo
            self.model = None
            self.use_ml_model = False
            print("⚠️ Using rule-based shot classification (demo mode)")
    
    def extract_shot_features(self, landmarks):
        """
        Extract relevant features from pose landmarks for shot classification.
        
        Args:
            landmarks: MediaPipe pose landmarks (33 x 4)
            
        Returns:
            Feature vector for classification
        """
        if landmarks is None:
            return None
        
        # Key landmarks for cricket shots
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # Calculate key features
        features = []
        
        # 1. Wrist positions (relative to hips)
        hip_center = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
        features.extend([
            right_wrist[0] - hip_center[0],  # Horizontal wrist position
            right_wrist[1] - hip_center[1],  # Vertical wrist position
            left_wrist[0] - hip_center[0],
            left_wrist[1] - hip_center[1]
        ])
        
        # 2. Shoulder-hip angle (body rotation)
        shoulder_center = [(left_shoulder[0] + right_shoulder[0])/2, 
                          (left_shoulder[1] + right_shoulder[1])/2]
        body_angle = np.arctan2(shoulder_center[1] - hip_center[1],
                               shoulder_center[0] - hip_center[0])
        features.append(body_angle)
        
        # 3. Bat angle (right wrist to right shoulder)
        bat_angle = np.arctan2(right_wrist[1] - right_shoulder[1],
                              right_wrist[0] - right_shoulder[0])
        features.append(bat_angle)
        
        # 4. Knee bend (average of both knees)
        left_knee_y = left_knee[1]
        right_knee_y = right_knee[1]
        avg_knee_bend = (left_knee_y + right_knee_y) / 2
        features.append(avg_knee_bend)
        
        # 5. Foot placement
        foot_width = abs(left_ankle[0] - right_ankle[0])
        features.append(foot_width)
        
        return np.array(features)
    
    def add_frame(self, landmarks):
        """
        Add a frame to the pose sequence buffer.
        
        Args:
            landmarks: MediaPipe pose landmarks (33 x 4)
        """
        features = self.extract_shot_features(landmarks)
        if features is not None:
            self.pose_sequence.append(features)
    
    def classify_shot_rule_based(self):
        """
        Rule-based shot classification (demo mode).
        Uses simple heuristics based on pose features.
        
        Returns:
            (shot_name, confidence)
        """
        if len(self.pose_sequence) < 10:
            return "Stance", 0.5
        
        # Get recent features
        recent_features = list(self.pose_sequence)[-10:]
        avg_features = np.mean(recent_features, axis=0)
        
        # Extract key indicators
        wrist_x = avg_features[0]  # Right wrist horizontal
        wrist_y = avg_features[1]  # Right wrist vertical
        bat_angle = avg_features[5]
        foot_width = avg_features[7]
        
        # Simple heuristic rules
        if wrist_y < -0.2:  # Wrist high - lofted shots
            if wrist_x > 0.1:
                return "Lofted", 0.75
            else:
                return "Straight Drive", 0.72
        
        elif wrist_x > 0.3:  # Wrist far to right - cut shots
            if bat_angle > 0.5:
                return "Square Cut", 0.70
            else:
                return "Late Cut", 0.68
        
        elif wrist_x < -0.2:  # Wrist to left - leg side shots
            if wrist_y > 0.1:
                return "Sweep", 0.65
            elif foot_width > 0.25:
                return "Pull", 0.73
            else:
                return "Flick", 0.71
        
        elif abs(wrist_x) < 0.15:  # Wrist central - straight shots
            if wrist_y > 0.05:
                return "Defense", 0.80
            elif wrist_y < -0.1:
                return "Cover Drive", 0.78
            else:
                return "Straight Drive", 0.76
        
        else:
            # Default to most common shot
            return "Defense", 0.60
    
    def classify_shot(self):
        """
        Classify the current cricket shot from pose sequence.
        
        Returns:
            dict: {
                'shot_name': str,
                'confidence': float,
                'all_predictions': dict of {shot: probability}
            }
        """
        if len(self.pose_sequence) < self.sequence_length // 3:
            return {
                'shot_name': 'Analyzing...',
                'confidence': 0.0,
                'all_predictions': {}
            }
        
        if self.use_ml_model and self.model is not None:
            # Use trained ML model
            sequence = np.array(list(self.pose_sequence))
            # Pad if needed
            if len(sequence) < self.sequence_length:
                padding = np.zeros((self.sequence_length - len(sequence), sequence.shape[1]))
                sequence = np.vstack([padding, sequence])
            
            # Predict (shape: 1, sequence_length, features)
            prediction = self.model.predict(sequence.reshape(1, self.sequence_length, -1))
            shot_idx = np.argmax(prediction)
            confidence = float(prediction[0][shot_idx])
            
            # Get all predictions
            all_preds = {self.shot_classes[i]: float(prediction[0][i]) 
                        for i in range(len(self.shot_classes))}
            
            return {
                'shot_name': self.shot_classes[shot_idx],
                'confidence': confidence,
                'all_predictions': all_preds
            }
        else:
            # Use rule-based classification
            shot_name, confidence = self.classify_shot_rule_based()
            
            # Create fake probability distribution for demo
            all_preds = {shot: 0.05 for shot in self.shot_classes}
            all_preds[shot_name] = confidence
            
            # Add some noise to other shots
            remaining = 1.0 - confidence
            for shot in self.shot_classes:
                if shot != shot_name:
                    all_preds[shot] = remaining / (len(self.shot_classes) - 1)
            
            return {
                'shot_name': shot_name,
                'confidence': confidence,
                'all_predictions': all_preds
            }
    
    def reset(self):
        """Reset the pose sequence buffer."""
        self.pose_sequence.clear()
    
    def get_top_predictions(self, n=3):
        """
        Get top N shot predictions.
        
        Args:
            n: Number of top predictions to return
            
        Returns:
            List of (shot_name, confidence) tuples
        """
        result = self.classify_shot()
        all_preds = result['all_predictions']
        
        if not all_preds:
            return []
        
        sorted_preds = sorted(all_preds.items(), key=lambda x: x[1], reverse=True)
        return sorted_preds[:n]


if __name__ == "__main__":
    # Test the classifier
    classifier = CricketShotClassifier()
    print("\n🏏 Cricket Shot Classifier Initialized")
    print(f"Shot classes: {', '.join(classifier.shot_classes)}")
    print(f"Mode: {'ML Model' if classifier.use_ml_model else 'Rule-Based (Demo)'}")
    
    # Simulate some pose data
    dummy_landmarks = np.random.rand(33, 4)
    for i in range(40):
        classifier.add_frame(dummy_landmarks)
    
    result = classifier.classify_shot()
    print(f"\nTest Classification:")
    print(f"  Shot: {result['shot_name']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"\nTop 3 predictions:")
    for shot, conf in classifier.get_top_predictions(3):
        print(f"  {shot}: {conf:.2%}")
