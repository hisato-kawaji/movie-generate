import cv2
import torch
from diffusers.utils import load_image
from PIL import Image
import os
from controlnet_aux import OpenposeDetector

def extract_frames(video_path, output_folder):
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
    image = load_image(image_path)
    pose_image = openpose(image)
    return pose_image

if __name__ == '__main__':
    video_path = 'input_video.mp4'
    frames_folder = 'frames'
    pose_folder = 'poses'

    if not os.path.exists(pose_folder):
        os.makedirs(pose_folder)

    print("Extracting frames...")
    num_frames = extract_frames(video_path, frames_folder)
    print(f"Extracted {num_frames} frames.")

    print("Loading OpenPose model...")
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

    print("Estimating poses...")
    for i in range(num_frames):
        frame_path = os.path.join(frames_folder, f'frame_{i:04d}.png')
        pose_image = estimate_pose(frame_path, openpose)
        pose_image.save(os.path.join(pose_folder, f'pose_{i:04d}.png'))
    print("Pose estimation complete.")
