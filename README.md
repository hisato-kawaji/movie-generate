# AI Animate Anyone MVP

This project is a Minimum Viable Product (MVP) for a Python script that generates a video of a person dancing, based on a YouTube video and a text prompt. It uses a combination of open-source AI models to achieve this, including AnimateDiff, ControlNet, and OpenPose.

## How it Works

The process is broken down into the following steps:

1.  **Video Download**: The script downloads a video from a given YouTube URL.
2.  **Frame Extraction**: The video is split into individual frames.
3.  **Pose Estimation**: For each frame, the script uses a pre-trained OpenPose model to extract the human pose, which is represented as a skeleton image.
4.  **Video Generation**: The script uses AnimateDiff and ControlNet to generate a new sequence of frames. It takes the estimated poses and a text prompt (describing the desired subject) as input, and generates a video of the subject dancing according to the poses.
5.  **Audio Integration**: The audio from the original YouTube video is extracted and merged with the newly generated video.

## Prerequisites

- Python 3.8 or higher
- `ffmpeg` installed and available in your system's PATH.

## Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/ai-animate-anyone-mvp.git
    cd ai-animate-anyone-mvp
    ```

2.  Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the main script from the command line, providing a YouTube URL and a text prompt.

```bash
python main.py --youtube_url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --prompt "a photo of a stormtrooper dancing"
```

### Command-line arguments

-   `--youtube_url`: (Required) The URL of the YouTube video to use as the motion source.
-   `--prompt`: (Required) A text prompt describing the subject you want to animate.
-   `--output_path`: (Optional) The path to save the generated video file. Defaults to `output.mp4`.

## Example

```bash
python main.py \
    --youtube_url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
    --prompt "a beautiful shot of a ninja dancing in the snow" \
    --output_path "ninja_dancing.mp4"
```

This will generate a video named `ninja_dancing.mp4` in the project's root directory.

## Contributing

This is an MVP and there is much room for improvement. Contributions are welcome! Please feel free to open an issue or submit a pull request.
