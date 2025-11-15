import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, ControlNetModel
from diffusers.utils import export_to_video
from PIL import Image
import os

def generate_video(pose_folder, prompt, output_path):
    # Load the motion adapter
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

    # Load the base model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Enable xformers for memory savings
    pipe.enable_xformers_memory_efficient_attention()

    # Load controlnet
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
    pipe.controlnet = controlnet

    # Load pose images
    pose_images = []
    for filename in sorted(os.listdir(pose_folder)):
        if filename.endswith(".png"):
            pose_images.append(Image.open(os.path.join(pose_folder, filename)))

    # Generate video frames
    output = pipe(
        prompt=prompt,
        negative_prompt="bad quality, worse quality",
        num_frames=len(pose_images),
        guidance_scale=7.5,
        num_inference_steps=20,
        control_image=pose_images,
        controlnet_conditioning_scale=0.8,
    )
    frames = output.frames[0]

    # Export to video
    export_to_video(frames, output_path)


if __name__ == '__main__':
    pose_folder = 'poses'
    # The prompt should describe the subject of the input image
    prompt = 'a photo of a person dancing'
    output_video_path = 'output.mp4'

    generate_video(pose_folder, prompt, output_video_path)
    print(f'Video generated at {output_video_path}')
