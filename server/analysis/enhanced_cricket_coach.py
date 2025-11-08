"""
Enhanced Cricket Batting Coach
Combines Kaggle notebook's biomechanics analysis with live feed capability.
Provides detailed feedback based on cricket batting fundamentals.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque


class EnhancedCricketCoach:
    """
    Cricket batting coach with comprehensive biomechanics analysis.
    Works on both live feed and recorded videos.
    """
    
    def __init__(self, buffer_size=15):
        """Initialize the enhanced cricket coach."""
        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Motion tracking for shot phase detection
        self.pose_buffer = deque(maxlen=buffer_size)
        self.wrist_velocities = deque(maxlen=10)
        
        # Shot phase tracking
        self.current_phase = 'stance'  # stance, backswing, contact, follow_through
        
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
    
    def analyze_head_position(self, landmarks):
        """
        Analyze head position (from Kaggle notebook logic).
        Head should be over the ball and stable.
        """
        nose = landmarks[0]  # NOSE
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Head should be centered and not too high/low
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        
        feedback = {}
        
        # Vertical position (head should be above hips)
        if nose[1] > hip_center_y + 0.15:  # Too low
            feedback['head_vertical'] = {
                'status': 'error',
                'message': '🧍 Head too low - keep your head up and eyes on the ball',
                'severity': 'high',
                'metric': f'Head-to-hip ratio: {(nose[1] - hip_center_y):.3f}'
            }
        elif nose[1] < hip_center_y - 0.3:  # Too high
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
        
        # Horizontal alignment (head should be centered)
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
        """
        Analyze foot placement and balance (from Kaggle notebook).
        """
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_heel = landmarks[29] if len(landmarks) > 29 else left_ankle
        right_heel = landmarks[30] if len(landmarks) > 30 else right_ankle
        
        feedback = {}
        
        # Foot width (should be shoulder-width apart)
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
        
        # Foot alignment (should be roughly parallel)
        foot_y_diff = abs(left_ankle[1] - right_ankle[1])
        if foot_y_diff > 0.1:
            feedback['foot_alignment'] = {
                'status': 'warning',
                'message': '👣 Align your feet more evenly - one foot is too far forward',
                'severity': 'medium'
            }
        
        return feedback
    
    def analyze_body_balance(self, landmarks):
        """
        Analyze overall body balance (from Kaggle notebook).
        Weight distribution and center of gravity.
        """
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        feedback = {}
        
        # Hip-to-ankle distance (indicates weight distribution)
        left_balance = np.linalg.norm(left_hip[:2] - left_ankle[:2])
        right_balance = np.linalg.norm(right_hip[:2] - right_ankle[:2])
        total_balance = left_balance + right_balance
        
        # Check if weight is evenly distributed
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
        
        # Check if body is upright (hip center should be above ankle center)
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
        """
        Comprehensive batting stance analysis.
        """
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
        
        # Elbow angles
        right_elbow_angle = self.calculate_angle(
            landmarks[12, :3],  # right shoulder
            landmarks[14, :3],  # right elbow
            landmarks[16, :3]   # right wrist
        )
        
        left_elbow_angle = self.calculate_angle(
            landmarks[11, :3],  # left shoulder
            landmarks[13, :3],  # left elbow
            landmarks[15, :3]   # left wrist
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
        
        # Elbow position (bat lift)
        if right_elbow_angle < 90:
            feedback['bat_lift'] = {
                'status': 'warning',
                'message': '🏏 Lift your bat higher - elbow should be at comfortable angle',
                'severity': 'medium',
                'metric': f'Elbow angle: {right_elbow_angle:.1f}°'
            }
        
        return feedback
    
    def detect_shot_phase(self, landmarks):
        """
        Detect which phase of the shot the player is in.
        """
        right_wrist = landmarks[16]
        
        # Track wrist movement
        if len(self.pose_buffer) > 0:
            prev_wrist = self.pose_buffer[-1][16]
            velocity = np.linalg.norm(right_wrist[:2] - prev_wrist[:2])
            self.wrist_velocities.append(velocity)
        
        # Simple phase detection based on wrist velocity
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
    
    def is_batting_stance(self, landmarks):
        """
        Validate if the person is actually in a batting stance.
        Returns (is_valid, reason) tuple.
        RELAXED VERSION - just check if person is visible and upright
        """
        # Get key landmarks
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        nose = landmarks[0]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # Check 1: Ankles should be visible (person is in frame)
        if left_ankle[3] < 0.2 or right_ankle[3] < 0.2:  # Very low visibility
            return False, "⚠️ Make sure your full body is visible in the camera"
        
        # Check 2: Basic upright posture (hips above ankles)
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        ankle_center_y = (left_ankle[1] + right_ankle[1]) / 2
        
        # Relaxed check - just needs to be somewhat upright
        if hip_center_y >= ankle_center_y + 0.1:  # Very relaxed threshold
            return False, "⚠️ Please stand up - currently in sitting/crouching position"
        
        # If we pass basic checks, allow analysis
        return True, "✅ Valid pose detected"
    
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
        Main analysis function combining all checks.
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
            return {
                'pose_detected': True,
                'pose_landmarks': pose_landmarks,
                'is_batting_stance': False,
                'phase': 'invalid',
                'feedback': [{
                    'status': 'error',
                    'message': validation_message,
                    'severity': 'high'
                }],
                'form_score': 0,
                'message': validation_message
            }
        
        # Add to buffer
        self.pose_buffer.append(landmarks)
        
        # Detect shot phase
        phase = self.detect_shot_phase(landmarks)
        
        # Run all analyses
        all_feedback = {}
        
        # Core biomechanics (from Kaggle notebook)
        all_feedback['head'] = self.analyze_head_position(landmarks)
        all_feedback['feet'] = self.analyze_foot_placement(landmarks)
        all_feedback['balance'] = self.analyze_body_balance(landmarks)
        
        # Batting technique
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
            'feedback': flattened_feedback,
            'form_score': form_score,
            'detailed_feedback': all_feedback,
            'message': f'Phase: {phase.upper()} | Score: {form_score}/100'
        }
    
    def close(self):
        """Release resources."""
        self.pose.close()


# Example usage - Live feed
if __name__ == "__main__":
    coach = EnhancedCricketCoach()
    cap = cv2.VideoCapture(0)
    
    print("🏏 Enhanced Cricket Batting Coach")
    print("=" * 60)
    print("Based on cricket biomechanics principles")
    print("Press 'q' to quit, 'r' to reset")
    print("=" * 60)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze frame
        result = coach.analyze_frame(frame)
        
        # Display results
        if result['pose_detected']:
            # Draw pose landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                result['pose_landmarks'],
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Display phase
            cv2.putText(frame, f"Phase: {result['phase'].upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Display form score
            score = result['form_score']
            color = (0, 255, 0) if score > 80 else (0, 255, 255) if score > 60 else (0, 0, 255)
            cv2.putText(frame, f"Form: {score:.0f}/100", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Display top feedback (errors and warnings only)
            y_offset = 110
            feedback_shown = 0
            for fb in result['feedback']:
                if fb['status'] != 'success' and feedback_shown < 3:
                    fb_color = (0, 0, 255) if fb['status'] == 'error' else (0, 165, 255)
                    # Show message
                    cv2.putText(frame, fb['message'][:50], (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, fb_color, 1)
                    y_offset += 25
                    feedback_shown += 1
                    
                    # Show metric if available
                    if 'metric' in fb and y_offset < 250:
                        cv2.putText(frame, f"  {fb['metric']}", (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                        y_offset += 20
        else:
            cv2.putText(frame, result['message'], (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Enhanced Cricket Coach', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            coach.pose_buffer.clear()
            coach.wrist_velocities.clear()
    
    cap.release()
    cv2.destroyAllWindows()
    coach.close()
    
    print("\n✅ Coaching session ended!")
