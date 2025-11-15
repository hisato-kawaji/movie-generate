import cv2
import os
from PIL import Image
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image

def extract_frames(video_path, output_folder):
    """Extracts frames from a video file."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, f'frame_{frame_count:04d}.png'), frame)
        frame_count += 1
    cap.release()
    return frame_count

def estimate_pose(image_path, openpose):
    """Estimates the pose from an image."""
    image = load_image(image_path)
    pose_image = openpose(image)
    return pose_image
