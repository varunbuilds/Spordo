# server/app.py
from fastapi import FastAPI, WebSocket
import uvicorn
import base64
import numpy as np
import cv2
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Use enhanced cricket coach (lighter, no TensorFlow)
from analysis.enhanced_cricket_coach import EnhancedCricketCoach

# ML model imports - kept for future use but not active by default
# from models.cricket_shot_classifier_v2 import CricketShotClassifierV2

load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please add it to the server/.env file.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Enhanced Cricket Coach
cricket_coach = EnhancedCricketCoach()

# Simple rule-based shot classifier (active by default)
class SimpleShotClassifier:
    """Simple rule-based shot classifier without TensorFlow dependencies."""
    def __init__(self):
        self.shots = ['Cover Drive', 'Defense', 'Flick', 'Hook', 'Late Cut', 
                     'Lofted', 'Pull', 'Square Cut', 'Straight Drive', 'Sweep']
        self.pose_buffer = []
        
    def add_pose(self, landmarks):
        if landmarks is not None:
            self.pose_buffer.append(landmarks)
            if len(self.pose_buffer) > 30:
                self.pose_buffer.pop(0)
    
    def classify_shot(self):
        if len(self.pose_buffer) < 10:
            return 'Analyzing...', 0.0, []
        
        # Simple heuristic based on wrist position
        recent = self.pose_buffer[-10:]
        avg_wrist_x = np.mean([p[16][0] for p in recent])  # Right wrist X
        avg_wrist_y = np.mean([p[16][1] for p in recent])  # Right wrist Y
        
        if avg_wrist_y < 0.4:  # High wrist
            shot = 'Lofted' if abs(avg_wrist_x - 0.5) < 0.2 else 'Square Cut'
            conf = 0.75
        elif avg_wrist_x > 0.6:  # Right side
            shot = 'Cover Drive'
            conf = 0.72
        elif avg_wrist_x < 0.4:  # Left side
            shot = 'Flick'
            conf = 0.70
        else:  # Center
            shot = 'Straight Drive' if avg_wrist_y < 0.5 else 'Defense'
            conf = 0.68
        
        # Generate top predictions
        other_shots = [s for s in self.shots if s != shot]
        top_predictions = [(shot, conf)]
        top_predictions.extend([(other_shots[i], conf - 0.1 * (i+1)) for i in range(min(2, len(other_shots)))])
        
        return shot, conf, top_predictions
    
    def reset(self):
        self.pose_buffer = []

# Initialize rule-based shot classifier
shot_classifier = SimpleShotClassifier()

# Initialize rule-based shot classifier
shot_classifier = SimpleShotClassifier()

app = FastAPI()

def process_image(data_url: str):
    """Decodes a base64 data URL and returns an OpenCV image."""
    try:
        # Split the metadata from the actual data
        head, data = data_url.split(',', 1)
        # Decode the base64 data
        image_bytes = base64.b64decode(data)
        # Convert bytes to a numpy array
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        # Decode the numpy array into an image
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None
        return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected!")
    frame_count = 0
    try:
        while True:
            data_url = await websocket.receive_text()
            frame = process_image(data_url)
            frame_count += 1

            if frame is not None:
                # Analyze frame with enhanced cricket coach
                result = cricket_coach.analyze_frame(frame)

                if result['pose_detected']:
                    # Add shot classification
                    landmarks, _ = cricket_coach.extract_landmarks(frame)
                    if landmarks is not None:
                        # Add pose to classifier buffer
                        shot_classifier.add_pose(landmarks)
                        
                        # Classify shot
                        shot_name, shot_confidence, top_predictions = shot_classifier.classify_shot()
                        
                        result['shot_name'] = shot_name
                        result['shot_confidence'] = shot_confidence
                        result['top_shots'] = top_predictions
                    else:
                        result['shot_name'] = 'N/A'
                        result['shot_confidence'] = 0.0
                        result['top_shots'] = []
                    
                    # Check if it's a valid batting stance
                    if not result.get('is_batting_stance', True):
                        # Not in batting stance - send validation message
                        shot_classifier.reset()  # Reset shot buffer
                        await websocket.send_json({
                            "type": "coaching_feedback",
                            "message": result.get('message', ''),
                            "form_score": 0,
                            "phase": "invalid",
                            "shot_name": "N/A",
                            "shot_confidence": 0.0,
                            "top_shots": [],
                            "all_feedback": result.get('feedback', [])
                        })
                        continue
                else:
                    # No pose detected - send appropriate message
                    await websocket.send_json({
                        "type": "coaching_feedback",
                        "message": result.get('message', 'No pose detected'),
                        "form_score": 0,
                        "phase": "no_pose",
                        "shot_name": "N/A",
                        "shot_confidence": 0.0,
                        "top_shots": [],
                        "all_feedback": []
                    })
                    continue
                
                # Get top priority feedback (errors and warnings only)
                priority_feedback = [
                    fb for fb in result.get('feedback', [])
                    if fb.get('status') in ['error', 'warning']
                ]
                
                # Send real-time data for visual overlays (every frame)
                # Only generate Gemini coaching tip every 30 frames to avoid overwhelming
                coaching_tip = None
                if priority_feedback and frame_count % 30 == 0:
                    # Get the most important feedback
                    top_feedback = priority_feedback[0]
                    
                    # Generate Coaching Tip with Gemini (include shot name)
                    prompt = f"""You are a professional cricket coach. 
                    The batsman is playing a {result.get('shot_name', 'shot')} shot.
                    Form issue: '{top_feedback['message']}'.
                    Phase: {result['phase']}.
                    Form score: {result['form_score']}/100.
                    
                    Give ONE short, actionable coaching tip (max 15 words)."""
                    
                    try:
                        response = model.generate_content(prompt)
                        coaching_tip = response.text
                    except Exception as e:
                        print(f"Gemini error: {e}")
                        coaching_tip = top_feedback['message']
                    
                    print(f"Shot: {result.get('shot_name')} ({result.get('shot_confidence', 0):.0%}) | Phase: {result['phase']} | Score: {result['form_score']} | Tip: {coaching_tip}")

                # Send feedback (with or without Gemini tip)
                feedback_data = {
                    "type": "coaching_feedback",
                    "message": coaching_tip if coaching_tip else "",
                    "form_score": result.get('form_score', 0),
                    "phase": result.get('phase', ''),
                    "shot_name": result.get('shot_name', 'Analyzing...'),
                    "shot_confidence": result.get('shot_confidence', 0.0),
                    "top_shots": result.get('top_shots', []),
                    "all_feedback": priority_feedback[:3]  # Top 3 issues
                }
                
                # Debug logging every 30 frames
                if frame_count % 30 == 0:
                    print(f"📊 Frame {frame_count}: Score={feedback_data['form_score']}, Phase={feedback_data['phase']}, Feedback={len(feedback_data['all_feedback'])}")
                
                await websocket.send_json(feedback_data)
                    
    except Exception as e:
        print(f"Client disconnected or error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)