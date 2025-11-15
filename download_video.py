import yt_dlp

def download_video(url, output_path):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == '__main__':
    video_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'  # Example URL
    output_video_path = 'input_video.mp4'
    download_video(video_url, output_video_path)
    print(f'Video downloaded to {output_video_path}')
