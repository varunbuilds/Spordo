"""
ML-Enhanced Cricket Batting Coach
Combines shot classification (ML) with biomechanics analysis (rules)
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from cricket_shot_classifier import CricketShotClassifier


class MLCricketCoach:
    """
    Cricket batting coach with ML shot classification + biomechanics analysis.
    """
    
    def __init__(self, buffer_size=15):
        """Initialize the ML cricket coach."""
        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Shot classifier
        self.shot_classifier = CricketShotClassifier(sequence_length=30)
        
        # Motion tracking for phase detection
        self.pose_buffer = deque(maxlen=buffer_size)
        self.wrist_velocities = deque(maxlen=10)
        
        # Shot phase tracking
        self.current_phase = 'stance'
        
    def extract_landmarks(self, frame):
        """Extract pose landmarks from frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ])
            return np.array(landmarks), results.pose_landmarks
        
        return None, None
    
    def calculate_angle(self, a, b, c):
        """Calculate angle at point b given three points."""
        ba = a - b
        bc = c - b
        
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def is_batting_stance(self, landmarks):
        """
        Validate if the person is actually in a batting stance.
        Returns (is_valid, reason) tuple.
        """
        # Get key landmarks
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        nose = landmarks[0]
        
        # Check 1: Person should be STANDING
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        knee_center_y = (left_knee[1] + right_knee[1]) / 2
        ankle_center_y = (left_ankle[1] + right_ankle[1]) / 2
        
        if hip_center_y >= knee_center_y - 0.05:
            return False, "⚠️ Please STAND UP to analyze batting stance - currently sitting/crouching"
        
        # Check 2: Legs should be relatively straight
        left_knee_angle = self.calculate_angle(
            landmarks[23, :3],  # left hip
            landmarks[25, :3],  # left knee
            landmarks[27, :3]   # left ankle
        )
        right_knee_angle = self.calculate_angle(
            landmarks[24, :3],  # right hip
            landmarks[26, :3],  # right knee
            landmarks[28, :3]   # right ankle
        )
        
        if left_knee_angle < 100 and right_knee_angle < 100:
            return False, "⚠️ Please STAND UP to analyze batting stance - knees are too bent (sitting position)"
        
        # Check 3: Minimum height requirement
        body_height = abs(ankle_center_y - nose[1])
        if body_height < 0.3:
            return False, "⚠️ Please STAND UP fully - detected pose is too low"
        
        # Check 4: Feet should be visible
        if left_ankle[3] < 0.3 or right_ankle[3] < 0.3:
            return False, "⚠️ Make sure your full body is visible, including feet"
        
        return True, "✅ Valid batting stance detected"
    
    def analyze_head_position(self, landmarks):
        """Analyze head position."""
        nose = landmarks[0]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        
        feedback = {}
        
        # Vertical position
        if nose[1] > hip_center_y + 0.15:
            feedback['head_vertical'] = {
                'status': 'error',
                'message': '🧍 Head too low - keep your head up and eyes on the ball',
                'severity': 'high',
                'metric': f'Head-to-hip ratio: {(nose[1] - hip_center_y):.3f}'
            }
        elif nose[1] < hip_center_y - 0.3:
            feedback['head_vertical'] = {
                'status': 'warning',
                'message': '🧍 Head position slightly high - maintain natural stance',
                'severity': 'low'
            }
        else:
            feedback['head_vertical'] = {
                'status': 'success',
                'message': '✅ Good head position',
                'severity': 'low'
            }
        
        # Horizontal alignment
        head_offset = abs(nose[0] - hip_center_x)
        if head_offset > 0.15:
            feedback['head_alignment'] = {
                'status': 'warning',
                'message': '⚖️ Keep your head aligned with your body center',
                'severity': 'medium',
                'metric': f'Offset: {head_offset:.3f}'
            }
        else:
            feedback['head_alignment'] = {
                'status': 'success',
                'message': '✅ Good head alignment',
                'severity': 'low'
            }
        
        return feedback
    
    def analyze_foot_placement(self, landmarks):
        """Analyze foot placement and balance."""
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        feedback = {}
        
        # Foot width
        foot_distance = abs(left_ankle[0] - right_ankle[0])
        
        if foot_distance < 0.1:
            feedback['foot_width'] = {
                'status': 'error',
                'message': '👣 Feet too close together - widen your stance for better balance',
                'severity': 'high',
                'metric': f'Foot width: {foot_distance:.3f}'
            }
        elif foot_distance > 0.35:
            feedback['foot_width'] = {
                'status': 'warning',
                'message': '👣 Stance too wide - bring feet slightly closer',
                'severity': 'medium',
                'metric': f'Foot width: {foot_distance:.3f}'
            }
        else:
            feedback['foot_width'] = {
                'status': 'success',
                'message': '✅ Good foot placement - balanced stance',
                'severity': 'low'
            }
        
        # Foot alignment
        foot_y_diff = abs(left_ankle[1] - right_ankle[1])
        if foot_y_diff > 0.1:
            feedback['foot_alignment'] = {
                'status': 'warning',
                'message': '👣 Align your feet more evenly - one foot is too far forward',
                'severity': 'medium'
            }
        
        return feedback
    
    def analyze_body_balance(self, landmarks):
        """Analyze overall body balance."""
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        feedback = {}
        
        # Hip-to-ankle distance
        left_balance = np.linalg.norm(left_hip[:2] - left_ankle[:2])
        right_balance = np.linalg.norm(right_hip[:2] - right_ankle[:2])
        total_balance = left_balance + right_balance
        
        balance_ratio = abs(left_balance - right_balance) / (total_balance + 1e-8)
        
        if balance_ratio > 0.3:
            feedback['weight_distribution'] = {
                'status': 'warning',
                'message': '⚖️ Weight distribution uneven - balance your weight between both feet',
                'severity': 'medium',
                'metric': f'Balance ratio: {balance_ratio:.3f}'
            }
        else:
            feedback['weight_distribution'] = {
                'status': 'success',
                'message': '✅ Good weight distribution',
                'severity': 'low'
            }
        
        # Body posture
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        ankle_center_y = (left_ankle[1] + right_ankle[1]) / 2
        
        if hip_center_y > ankle_center_y + 0.05:
            feedback['body_posture'] = {
                'status': 'warning',
                'message': '🧍 Stand more upright - you\'re leaning back too much',
                'severity': 'medium'
            }
        
        return feedback
    
    def analyze_batting_stance(self, landmarks):
        """Comprehensive batting stance analysis."""
        # Knee angles
        left_knee_angle = self.calculate_angle(
            landmarks[23, :3],  # left hip
            landmarks[25, :3],  # left knee
            landmarks[27, :3]   # left ankle
        )
        
        right_knee_angle = self.calculate_angle(
            landmarks[24, :3],  # right hip
            landmarks[26, :3],  # right knee
            landmarks[28, :3]   # right ankle
        )
        
        feedback = {}
        
        # Knee bend analysis
        if left_knee_angle > 175 or right_knee_angle > 175:
            feedback['knee_bend'] = {
                'status': 'error',
                'message': '🦵 Bend your knees more - locked knees reduce power and balance',
                'severity': 'high',
                'metric': f'L: {left_knee_angle:.1f}° R: {right_knee_angle:.1f}°'
            }
        elif left_knee_angle < 130 or right_knee_angle < 130:
            feedback['knee_bend'] = {
                'status': 'warning',
                'message': '🦵 Knees too bent - straighten slightly for better balance',
                'severity': 'medium',
                'metric': f'L: {left_knee_angle:.1f}° R: {right_knee_angle:.1f}°'
            }
        else:
            feedback['knee_bend'] = {
                'status': 'success',
                'message': '✅ Excellent knee position',
                'severity': 'low'
            }
        
        return feedback
    
    def detect_shot_phase(self, landmarks):
        """Detect which phase of the shot the player is in."""
        right_wrist = landmarks[16]
        
        # Track wrist movement
        if len(self.pose_buffer) > 0:
            prev_wrist = self.pose_buffer[-1][16]
            velocity = np.linalg.norm(right_wrist[:2] - prev_wrist[:2])
            self.wrist_velocities.append(velocity)
        
        # Phase detection based on wrist velocity
        if len(self.wrist_velocities) > 5:
            avg_velocity = np.mean(list(self.wrist_velocities)[-5:])
            
            if avg_velocity < 0.01:
                phase = 'stance'
            elif avg_velocity < 0.05:
                phase = 'backswing'
            elif avg_velocity >= 0.05:
                phase = 'swing'
            else:
                phase = 'follow_through'
        else:
            phase = 'stance'
        
        self.current_phase = phase
        return phase
    
    def calculate_form_score(self, all_feedback):
        """Calculate overall form score based on feedback."""
        score = 100
        
        for category, feedback_items in all_feedback.items():
            if isinstance(feedback_items, dict):
                for key, item in feedback_items.items():
                    if isinstance(item, dict) and 'status' in item:
                        if item['status'] == 'error':
                            score -= 15
                        elif item['status'] == 'warning':
                            score -= 8
        
        return max(0, min(100, score))
    
    def analyze_frame(self, frame):
        """
        Main analysis function combining ML shot classification + biomechanics.
        """
        # Extract landmarks
        landmarks, pose_landmarks = self.extract_landmarks(frame)
        
        if landmarks is None:
            return {
                'pose_detected': False,
                'message': '❌ No pose detected - step into frame and face the camera'
            }
        
        # VALIDATE BATTING STANCE FIRST
        is_valid, validation_message = self.is_batting_stance(landmarks)
        
        if not is_valid:
            # Reset shot classifier when not in stance
            self.shot_classifier.reset()
            return {
                'pose_detected': True,
                'pose_landmarks': pose_landmarks,
                'is_batting_stance': False,
                'phase': 'invalid',
                'shot_name': 'N/A',
                'shot_confidence': 0.0,
                'top_shots': [],
                'feedback': [{
                    'status': 'error',
                    'message': validation_message,
                    'severity': 'high'
                }],
                'form_score': 0,
                'message': validation_message
            }
        
        # Add to buffers
        self.pose_buffer.append(landmarks)
        self.shot_classifier.add_frame(landmarks)
        
        # Detect shot phase
        phase = self.detect_shot_phase(landmarks)
        
        # CLASSIFY SHOT (ML)
        shot_result = self.shot_classifier.classify_shot()
        shot_name = shot_result['shot_name']
        shot_confidence = shot_result['confidence']
        top_shots = self.shot_classifier.get_top_predictions(3)
        
        # Run biomechanics analyses
        all_feedback = {}
        all_feedback['head'] = self.analyze_head_position(landmarks)
        all_feedback['feet'] = self.analyze_foot_placement(landmarks)
        all_feedback['balance'] = self.analyze_body_balance(landmarks)
        all_feedback['stance'] = self.analyze_batting_stance(landmarks)
        
        # Flatten feedback for display
        flattened_feedback = []
        for category, items in all_feedback.items():
            if isinstance(items, dict):
                for key, item in items.items():
                    if isinstance(item, dict) and 'message' in item:
                        flattened_feedback.append(item)
        
        # Sort by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        flattened_feedback.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 2))
        
        # Calculate form score
        form_score = self.calculate_form_score(all_feedback)
        
        return {
            'pose_detected': True,
            'pose_landmarks': pose_landmarks,
            'is_batting_stance': True,
            'phase': phase,
            'shot_name': shot_name,
            'shot_confidence': shot_confidence,
            'top_shots': top_shots,
            'feedback': flattened_feedback,
            'form_score': form_score,
            'detailed_feedback': all_feedback,
            'message': f'{shot_name} ({shot_confidence:.0%}) | Phase: {phase.upper()} | Score: {form_score}/100'
        }
    
    def close(self):
        """Release resources."""
        self.pose.close()


if __name__ == "__main__":
    print("🏏 ML Cricket Coach - Test Mode")
    print("=" * 60)
    
    coach = MLCricketCoach()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        result = coach.analyze_frame(frame)
        
        if result['pose_detected'] and result.get('is_batting_stance', False):
            print(f"\r{result['message']}", end='')
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    coach.close()
    print("\n✅ Test completed!")
