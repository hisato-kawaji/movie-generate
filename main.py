import argparse
import os
import shutil
import ffmpeg
from download_video import download_video
from extract_pose import extract_frames, estimate_pose
from generate_video import generate_video

def main(youtube_url, prompt, output_path):
    # Define temporary folders and files
    video_path = 'input_video.mp4'
    frames_folder = 'frames'
    pose_folder = 'poses'
    temp_output_path = 'temp_output.mp4'

    # Clean up previous runs
    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)
    if os.path.exists(pose_folder):
        shutil.rmtree(pose_folder)
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
    if os.path.exists(output_path):
        os.remove(output_path)

    # 1. Download video
    print("Downloading video...")
    download_video(youtube_url, video_path)

    # 2. Extract frames
    print("Extracting frames...")
    num_frames = extract_frames(video_path, frames_folder)
    print(f"Extracted {num_frames} frames.")

    # 3. Estimate poses
    print("Loading OpenPose model...")
    from controlnet_aux import OpenposeDetector
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

    print("Estimating poses...")
    if not os.path.exists(pose_folder):
        os.makedirs(pose_folder)
    for i in range(num_frames):
        frame_path = os.path.join(frames_folder, f'frame_{i:04d}.png')
        pose_image = estimate_pose(frame_path, openpose)
        pose_image.save(os.path.join(pose_folder, f'pose_{i:04d}.png'))
    print("Pose estimation complete.")

    # 4. Generate video
    print("Generating video...")
    generate_video(pose_folder, prompt, temp_output_path)
    print(f"Video generated at {temp_output_path}")

    # 5. Add audio to the generated video
    print("Adding audio...")
    input_video_stream = ffmpeg.input(temp_output_path)
    input_audio_stream = ffmpeg.input(video_path).audio
    ffmpeg.output(input_video_stream, input_audio_stream, output_path, vcodec='copy', acodec='aac').run()
    print(f"Final video with audio saved at {output_path}")

    # 6. Clean up temporary files
    print("Cleaning up temporary files...")
    os.remove(video_path)
    os.remove(temp_output_path)
    shutil.rmtree(frames_folder)
    shutil.rmtree(pose_folder)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a dancing video from a YouTube URL and a prompt.')
    parser.add_argument('--youtube_url', type=str, required=True, help='The URL of the YouTube video.')
    parser.add_argument('--prompt', type=str, required=True, help='A prompt describing the subject.')
    parser.add_argument('--output_path', type=str, default='output.mp4', help='The path to save the generated MP4 video.')
    args = parser.parse_args()

    main(args.youtube_url, args.prompt, args.output_path)
