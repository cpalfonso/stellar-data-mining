import subprocess

from moviepy.config import FFMPEG_BINARY
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def create_animation(
    image_filenames,
    output_filename,
    fps=5,
    codec="auto",
    bitrate="5000k",
    output_fps=30,
    ffmpeg_params=None,
    **kwargs
):
    if codec == "hevc":
        if hwaccel_available():
            codec = "hevc_videotoolbox"
        else:
            codec = "hevc"
    elif codec == "auto":
        codec = "libx264"

    if ffmpeg_params is None:
        ffmpeg_params = [
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
        ]

    logger = kwargs.pop("logger", None)
    audio = kwargs.pop("audio", False)

    with ImageSequenceClip(image_filenames, fps=fps) as clip:
        clip.write_videofile(
            output_filename,
            fps=output_fps,
            codec=codec,
            bitrate=bitrate,
            audio=audio,
            logger=logger,
            ffmpeg_params=ffmpeg_params,
            **kwargs,
        )


def codec_available(codec):
    result = _test_codec(codec)
    return result.returncode == 0


def hwaccel_available(codec="hevc_videotoolbox"):
    return codec_available(codec)


def _test_codec(codec):
    cmd = [
        FFMPEG_BINARY,
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", "color=color=black:size=1080x1080",
        "-vframes", "1",
        "-pix_fmt", "yuv420p10le",
        "-an",
        "-c:v", codec,
        "-f", "null",
        "-",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
    )
    return result
