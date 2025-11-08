"""
Updated Cricket Shot Classifier with Trained LSTM Model Support

This version can:
1. Load and use a trained LSTM model if available
2. Fall back to rule-based classification if model not found
"""

import numpy as np
import pickle
from pathlib import Path

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - using rule-based classification")


class CricketShotClassifierV2:
    """
    Enhanced cricket shot classifier with trained model support
    """
    
    def __init__(self, model_path=None, metadata_path=None):
        """
        Initialize classifier with optional trained model
        Auto-detects improved model first, then falls back to original
        
        Args:
            model_path: Path to trained Keras model (auto-detects if None)
            metadata_path: Path to model metadata (auto-detects if None)
        """
        self.model = None
        self.metadata = None
        self.use_trained_model = False
        self.pose_buffer = []
        self.max_buffer_size = 60
        
        # Auto-detect model (prefer advanced → improved → original)
        if model_path is None:
            if Path('models/cricket_lstm_advanced.h5').exists():
                model_path = 'models/cricket_lstm_advanced.h5'
                metadata_path = 'models/cricket_lstm_advanced.pkl'
            elif Path('models/cricket_lstm_improved.h5').exists():
                model_path = 'models/cricket_lstm_improved.h5'
                metadata_path = 'models/cricket_lstm_improved.pkl'
            elif Path('models/cricket_lstm_full.h5').exists():
                model_path = 'models/cricket_lstm_full.h5'
                metadata_path = 'models/cricket_lstm_model.pkl'
        
        # Shot classes (fallback for rule-based)
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
        
        # Try to load trained model
        if TENSORFLOW_AVAILABLE:
            self._load_trained_model(model_path, metadata_path)
        else:
            print("⚠️ Using rule-based shot classification (demo mode)")
    
    def _load_trained_model(self, model_path, metadata_path):
        """Load trained LSTM model if available"""
        try:
            model_file = Path(model_path)
            metadata_file = Path(metadata_path)
            
            if model_file.exists() and metadata_file.exists():
                # Load model
                self.model = load_model(str(model_file))
                
                # Load metadata
                with open(metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                self.shot_classes = self.metadata['classes']
                self.max_buffer_size = self.metadata['sequence_length']
                
                self.use_trained_model = True
                
                print(f"✅ Loaded trained LSTM model from: {model_path}")
                print(f"   Classes: {len(self.shot_classes)}")
                print(f"   Sequence length: {self.max_buffer_size}")
            else:
                print(f"⚠️ Trained model not found at: {model_path}")
                print(f"⚠️ Using rule-based shot classification (demo mode)")
        except Exception as e:
            print(f"⚠️ Error loading trained model: {e}")
            print(f"⚠️ Using rule-based shot classification (demo mode)")
            self.use_trained_model = False
    
    def extract_pose_features(self, pose_landmarks):
        """
        Extract 8 key features from pose landmarks
        
        Features:
        1. avg_wrist_x - Average wrist X position
        2. avg_wrist_y - Average wrist Y position
        3. left_knee_angle - Left knee bend angle (normalized)
        4. right_knee_angle - Right knee bend angle (normalized)
        5. hip_shoulder_alignment - Vertical alignment
        6. foot_width - Horizontal distance between feet
        7. bat_angle - Estimated bat angle from wrists
        8. body_rotation - Shoulder vs hip rotation
        """
        if not pose_landmarks:
            return np.zeros(8)
        
        try:
            # Extract key landmarks
            left_wrist = np.array([pose_landmarks[15].x, pose_landmarks[15].y, pose_landmarks[15].z])
            right_wrist = np.array([pose_landmarks[16].x, pose_landmarks[16].y, pose_landmarks[16].z])
            
            left_shoulder = np.array([pose_landmarks[11].x, pose_landmarks[11].y, pose_landmarks[11].z])
            right_shoulder = np.array([pose_landmarks[12].x, pose_landmarks[12].y, pose_landmarks[12].z])
            
            left_hip = np.array([pose_landmarks[23].x, pose_landmarks[23].y, pose_landmarks[23].z])
            right_hip = np.array([pose_landmarks[24].x, pose_landmarks[24].y, pose_landmarks[24].z])
            
            left_knee = np.array([pose_landmarks[25].x, pose_landmarks[25].y, pose_landmarks[25].z])
            right_knee = np.array([pose_landmarks[26].x, pose_landmarks[26].y, pose_landmarks[26].z])
            
            left_ankle = np.array([pose_landmarks[27].x, pose_landmarks[27].y, pose_landmarks[27].z])
            right_ankle = np.array([pose_landmarks[28].x, pose_landmarks[28].y, pose_landmarks[28].z])
            
            # Calculate features
            avg_wrist_x = (left_wrist[0] + right_wrist[0]) / 2
            avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
            
            # Knee angles
            left_knee_angle = self._calculate_angle(left_hip[:2], left_knee[:2], left_ankle[:2])
            right_knee_angle = self._calculate_angle(right_hip[:2], right_knee[:2], right_ankle[:2])
            
            # Hip-shoulder alignment
            hip_center = (left_hip + right_hip) / 2
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_shoulder_alignment = np.linalg.norm(shoulder_center[:2] - hip_center[:2])
            
            # Foot width
            foot_width = abs(left_ankle[0] - right_ankle[0])
            
            # Bat angle
            bat_angle = np.arctan2(right_wrist[1] - left_wrist[1], 
                                  right_wrist[0] - left_wrist[0])
            
            # Body rotation
            shoulder_angle = np.arctan2(right_shoulder[1] - left_shoulder[1],
                                       right_shoulder[0] - left_shoulder[0])
            hip_angle = np.arctan2(right_hip[1] - left_hip[1],
                                  right_hip[0] - left_hip[0])
            body_rotation = abs(shoulder_angle - hip_angle)
            
            return np.array([
                avg_wrist_x,
                avg_wrist_y,
                left_knee_angle / 180.0,
                right_knee_angle / 180.0,
                hip_shoulder_alignment,
                foot_width,
                bat_angle / np.pi,
                body_rotation / np.pi
            ])
            
        except Exception as e:
            return np.zeros(8)
    
    def _calculate_angle(self, a, b, c):
        """Calculate angle at point b"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def add_pose(self, pose_landmarks):
        """Add pose to buffer"""
        features = self.extract_pose_features(pose_landmarks)
        self.pose_buffer.append(features)
        
        # Keep buffer at max size
        if len(self.pose_buffer) > self.max_buffer_size:
            self.pose_buffer.pop(0)
    
    def classify_shot(self):
        """
        Classify the current shot based on pose buffer
        
        Returns:
            tuple: (shot_name, confidence, top_predictions)
        """
        if len(self.pose_buffer) < 10:
            return "Analyzing...", 0.0, []
        
        if self.use_trained_model and self.model is not None:
            return self._classify_with_model()
        else:
            return self._classify_rule_based()
    
    def _classify_with_model(self):
        """Use trained LSTM model for classification"""
        try:
            # Prepare sequence
            sequence = np.array(self.pose_buffer)
            
            # Pad if needed
            if len(sequence) < self.max_buffer_size:
                padding = np.zeros((self.max_buffer_size - len(sequence), 8))
                sequence = np.vstack([sequence, padding])
            
            # Reshape for model input
            sequence = sequence[:self.max_buffer_size]  # Truncate if too long
            sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
            
            # Predict
            predictions = self.model.predict(sequence, verbose=0)[0]
            
            # Get top predictions
            top_indices = np.argsort(predictions)[::-1][:3]
            top_predictions = [
                (self.shot_classes[i], float(predictions[i]))
                for i in top_indices
            ]
            
            # Get best prediction
            best_idx = top_indices[0]
            shot_name = self.shot_classes[best_idx]
            confidence = float(predictions[best_idx])
            
            return shot_name, confidence, top_predictions
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return self._classify_rule_based()
    
    def _classify_rule_based(self):
        """
        Fallback rule-based classification
        (Original heuristic approach)
        """
        if len(self.pose_buffer) < 10:
            return "Analyzing...", 0.0, []
        
        # Get recent features (last 30 frames)
        recent_poses = self.pose_buffer[-30:]
        avg_features = np.mean(recent_poses, axis=0)
        
        avg_wrist_x = avg_features[0]
        avg_wrist_y = avg_features[1]
        foot_width = avg_features[5]
        bat_angle = avg_features[6]
        
        # Rule-based classification
        shot_name = "Defense"
        confidence = 0.6
        
        if avg_wrist_y < 0.4:  # Wrist high
            if abs(avg_wrist_x - 0.5) < 0.2:  # Wrist center
                shot_name = "Lofted"
                confidence = 0.65
            else:
                shot_name = "Square Cut" if avg_wrist_x > 0.5 else "Pull"
                confidence = 0.6
        elif avg_wrist_x > 0.6:  # Wrist to the right
            if bat_angle > 0.3:
                shot_name = "Square Cut"
                confidence = 0.62
            else:
                shot_name = "Late Cut"
                confidence = 0.58
        elif avg_wrist_x < 0.4:  # Wrist to the left
            if foot_width > 0.3:
                shot_name = "Pull"
                confidence = 0.63
            else:
                shot_name = "Hook"
                confidence = 0.57
        elif 0.45 <= avg_wrist_x <= 0.55:  # Wrist straight
            if avg_wrist_y > 0.6:  # Low wrist
                shot_name = "Sweep"
                confidence = 0.61
            elif bat_angle > 0:
                shot_name = "Cover Drive"
                confidence = 0.64
            else:
                shot_name = "Straight Drive"
                confidence = 0.62
        else:
            shot_name = "Flick"
            confidence = 0.55
        
        # Generate top predictions (simulated)
        all_shots = [s for s in self.shot_classes if s != shot_name]
        top_predictions = [
            (shot_name, confidence),
            (all_shots[0], confidence * 0.5),
            (all_shots[1], confidence * 0.3)
        ]
        
        return shot_name, confidence, top_predictions
    
    def reset(self):
        """Reset pose buffer"""
        self.pose_buffer = []
