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

load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please add it to the server/.env file.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Enhanced Cricket Coach
cricket_coach = EnhancedCricketCoach()

# Initialize lightweight shot classifier (no TensorFlow)
class SimpleShotClassifier:
    """Simple rule-based shot classifier without TensorFlow dependencies."""
    def __init__(self):
        self.shots = ['Cover Drive', 'Defense', 'Flick', 'Hook', 'Late Cut', 
                     'Lofted', 'Pull', 'Square Cut', 'Straight Drive', 'Sweep']
        self.pose_buffer = []
        
    def add_frame(self, landmarks):
        if landmarks is not None:
            self.pose_buffer.append(landmarks)
            if len(self.pose_buffer) > 30:
                self.pose_buffer.pop(0)
    
    def classify_shot(self):
        if len(self.pose_buffer) < 10:
            return {'shot_name': 'Analyzing...', 'confidence': 0.0}
        
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
            
        return {'shot_name': shot, 'confidence': conf}
    
    def get_top_predictions(self, n=3):
        result = self.classify_shot()
        # Generate fake top predictions for demo
        shot = result['shot_name']
        conf = result['confidence']
        other_shots = [s for s in self.shots if s != shot]
        top = [(shot, conf)]
        top.extend([(other_shots[i], conf - 0.1 * (i+1)) for i in range(min(n-1, len(other_shots)))])
        return top[:n]
    
    def reset(self):
        self.pose_buffer = []

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
                        shot_classifier.add_frame(landmarks)
                        shot_result = shot_classifier.classify_shot()
                        result['shot_name'] = shot_result['shot_name']
                        result['shot_confidence'] = shot_result['confidence']
                        result['top_shots'] = shot_classifier.get_top_predictions(3)
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
                            "message": "",
                            "form_score": 0,
                            "phase": "invalid",
                            "shot_name": "N/A",
                            "shot_confidence": 0.0,
                            "top_shots": [],
                            "all_feedback": result.get('feedback', [])
                        })
                        continue
                    
                    # Get top priority feedback (errors and warnings only)
                    priority_feedback = [
                        fb for fb in result['feedback'] 
                        if fb['status'] in ['error', 'warning']
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
                    await websocket.send_json({
                        "type": "coaching_feedback",
                        "message": coaching_tip if coaching_tip else "",
                        "form_score": result['form_score'],
                        "phase": result['phase'],
                        "shot_name": result.get('shot_name', 'Analyzing...'),
                        "shot_confidence": result.get('shot_confidence', 0.0),
                        "top_shots": result.get('top_shots', []),
                        "all_feedback": priority_feedback[:3]  # Top 3 issues
                    })
                    
    except Exception as e:
        print(f"Client disconnected or error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)