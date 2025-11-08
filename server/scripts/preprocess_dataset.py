"""
Preprocess Cricket Shot Videos - Extract Pose Sequences

This script processes the cricket shot videos from the 3D CNN dataset and extracts
MediaPipe pose landmarks to create training data for LSTM model.

Usage:
    python preprocess_dataset.py --dataset_path /path/to/cropped_videos --output pose_sequences.pkl

Dataset Structure Expected:
    cropped_videos/
        cut/
            video1.avi
            video2.avi
        defence/
            video1.avi
        ... (other classes)
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Class mapping from dataset to your system
CLASS_MAPPING = {
    'cut': 'Square Cut',      # or 'Late Cut' based on analysis
    'defence': 'Defense',
    'glance': 'Flick',
    'offdrive': 'Cover Drive',
    'ondrive': 'Straight Drive',
    'pull': 'Pull',
    'sweep': 'Sweep',
    'unorthodox': 'Lofted'
}


def extract_pose_features(landmarks):
    """
    Extract relevant pose features from MediaPipe landmarks
    
    Returns 8 features matching your existing classifier:
    - avg_wrist_x, avg_wrist_y
    - left_knee_angle, right_knee_angle
    - hip_shoulder_alignment
    - foot_width
    - bat_angle (estimated from wrists)
    - body_rotation
    """
    if not landmarks:
        return np.zeros(8)
    
    # Extract key landmark positions
    left_wrist = np.array([landmarks[15].x, landmarks[15].y, landmarks[15].z])
    right_wrist = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
    
    left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
    right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
    
    left_hip = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])
    right_hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
    
    left_knee = np.array([landmarks[25].x, landmarks[25].y, landmarks[25].z])
    right_knee = np.array([landmarks[26].x, landmarks[26].y, landmarks[26].z])
    
    left_ankle = np.array([landmarks[27].x, landmarks[27].y, landmarks[27].z])
    right_ankle = np.array([landmarks[28].x, landmarks[28].y, landmarks[28].z])
    
    # Calculate features
    avg_wrist_x = (left_wrist[0] + right_wrist[0]) / 2
    avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
    
    # Knee angles (simplified)
    left_knee_angle = calculate_angle(left_hip[:2], left_knee[:2], left_ankle[:2])
    right_knee_angle = calculate_angle(right_hip[:2], right_knee[:2], right_ankle[:2])
    
    # Hip-shoulder alignment
    hip_center = (left_hip + right_hip) / 2
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_shoulder_alignment = np.linalg.norm(shoulder_center[:2] - hip_center[:2])
    
    # Foot width
    foot_width = abs(left_ankle[0] - right_ankle[0])
    
    # Bat angle (estimated from wrist positions)
    bat_angle = np.arctan2(right_wrist[1] - left_wrist[1], 
                          right_wrist[0] - left_wrist[0])
    
    # Body rotation (shoulder angle relative to hips)
    shoulder_angle = np.arctan2(right_shoulder[1] - left_shoulder[1],
                               right_shoulder[0] - left_shoulder[0])
    hip_angle = np.arctan2(right_hip[1] - left_hip[1],
                          right_hip[0] - left_hip[0])
    body_rotation = abs(shoulder_angle - hip_angle)
    
    return np.array([
        avg_wrist_x,
        avg_wrist_y,
        left_knee_angle / 180.0,  # Normalize to [0, 1]
        right_knee_angle / 180.0,
        hip_shoulder_alignment,
        foot_width,
        bat_angle / np.pi,  # Normalize to [-1, 1]
        body_rotation / np.pi
    ])


def calculate_angle(a, b, c):
    """Calculate angle at point b given three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)


def extract_poses_from_video(video_path, max_frames=60):
    """
    Extract pose sequences from a video file
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        numpy array of shape (num_frames, 8) containing pose features
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return None
    
    poses = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame skip to get approximately max_frames
    skip = max(1, total_frames // max_frames)
    
    while cap.isOpened() and len(poses) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames if needed
        if frame_count % skip != 0:
            frame_count += 1
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Extract features
            features = extract_pose_features(results.pose_landmarks.landmark)
            poses.append(features)
        
        frame_count += 1
    
    cap.release()
    
    if len(poses) == 0:
        return None
    
    return np.array(poses)


def process_dataset(dataset_path, output_path, max_frames=60):
    """
    Process entire dataset and save pose sequences
    
    Args:
        dataset_path: Path to cropped_videos folder
        output_path: Path to save pickle file
        max_frames: Maximum frames per video
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    data = {
        'sequences': [],
        'labels': [],
        'original_labels': [],
        'video_paths': []
    }
    
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'per_class': {}
    }
    
    # Get all class folders
    class_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    
    print(f"\n{'='*60}")
    print(f"Processing Cricket Shot Dataset")
    print(f"{'='*60}")
    print(f"Dataset path: {dataset_path}")
    print(f"Classes found: {len(class_folders)}")
    print(f"Max frames per video: {max_frames}")
    print(f"{'='*60}\n")
    
    # Process each class
    for class_folder in sorted(class_folders):
        class_name = class_folder.name
        
        if class_name not in CLASS_MAPPING:
            print(f"Skipping unknown class: {class_name}")
            continue
        
        mapped_label = CLASS_MAPPING[class_name]
        stats['per_class'][class_name] = {'total': 0, 'success': 0}
        
        # Get all video files
        video_files = list(class_folder.glob('*.avi')) + \
                     list(class_folder.glob('*.mp4')) + \
                     list(class_folder.glob('*.mov'))
        
        print(f"\n📁 Processing class: {class_name} → {mapped_label}")
        print(f"   Videos found: {len(video_files)}")
        
        # Process each video with progress bar
        for video_path in tqdm(video_files, desc=f"   {class_name}", ncols=80):
            stats['total'] += 1
            stats['per_class'][class_name]['total'] += 1
            
            # Extract poses
            pose_sequence = extract_poses_from_video(video_path, max_frames)
            
            if pose_sequence is not None and len(pose_sequence) > 0:
                data['sequences'].append(pose_sequence)
                data['labels'].append(mapped_label)
                data['original_labels'].append(class_name)
                data['video_paths'].append(str(video_path))
                
                stats['success'] += 1
                stats['per_class'][class_name]['success'] += 1
            else:
                stats['failed'] += 1
    
    # Save processed data
    print(f"\n{'='*60}")
    print(f"Saving processed data to: {output_path}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Total videos processed: {stats['total']}")
    print(f"Successfully extracted: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"Failed: {stats['failed']}")
    print(f"\nPer-class statistics:")
    
    for class_name in sorted(stats['per_class'].keys()):
        s = stats['per_class'][class_name]
        mapped = CLASS_MAPPING[class_name]
        print(f"  {class_name:12s} → {mapped:15s}: {s['success']:3d}/{s['total']:3d} videos")
    
    print(f"\n{'='*60}")
    print(f"✅ Dataset ready for training!")
    print(f"{'='*60}\n")
    
    return data, stats


def main():
    parser = argparse.ArgumentParser(description='Preprocess cricket shot videos for LSTM training')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to cropped_videos folder')
    parser.add_argument('--output', type=str, default='pose_sequences.pkl',
                       help='Output pickle file path')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Maximum frames per video (default: 60)')
    
    args = parser.parse_args()
    
    # Process dataset
    data, stats = process_dataset(args.dataset_path, args.output, args.max_frames)
    
    print(f"\n📊 Data shape summary:")
    print(f"   Number of sequences: {len(data['sequences'])}")
    if len(data['sequences']) > 0:
        print(f"   Sequence shape example: {data['sequences'][0].shape}")
        print(f"   Features per frame: {data['sequences'][0].shape[1]}")
    print(f"\n✅ Ready to train! Run: python train_lstm.py --data {args.output}")


if __name__ == "__main__":
    main()
