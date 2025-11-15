import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, ControlNetModel
from diffusers.utils import export_to_video
from PIL import Image
import os

def generate_video(pose_folder, prompt, output_path):
    """Generates a video from a folder of pose images and a prompt."""
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
    pipe.controlnet = controlnet

    pose_images = []
    for filename in sorted(os.listdir(pose_folder)):
        if filename.endswith(".png"):
            pose_images.append(Image.open(os.path.join(pose_folder, filename)))

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

    export_to_video(frames, output_path)
