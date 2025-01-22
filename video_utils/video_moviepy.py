import os
import cv2
import tempfile
import numpy as np
import librosa
import shutil
import logging

from moviepy.editor import VideoFileClip, ImageSequenceClip

ENCODING = "libx264"


def read_movie(input_file, resize=None, max_size=None):
    try:
        clip = VideoFileClip(input_file)

        frames = list()
        for frame in clip.iter_frames():
            if max_size and len(frames) >= max_size:
                break

            if not resize is None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)

            frames.append(frame)

        frames = np.array(frames)

        logging.debug("Video with was read successfully: {}.".format(input_file))
        clip.close()

    except Exception as e:
        frames = None
        logging.debug("Error occurred while reading {}: {}.".format(input_file, e))

    return frames


def extract_movie_info(input_file):
    try:
        clip = VideoFileClip(input_file)

        fps = clip.fps
        duration = clip.duration
        resolution = clip.size

        info = {"duration": duration, "resolution": resolution, "fps": fps}
        clip.close()

    except Exception as e:
        logging.debug(
            "Error occurred while extracting movie information: {}: {}.".format(input_file, e)
        )
        info = {"duration": None, "resolution": [None, None], "fps": None}

    return info


def read_audio_from_mp4(mp4_path, temp_audio_path="audio.wav"):
    try:
        clip = VideoFileClip(mp4_path)
        with tempfile.TemporaryDirectory() as tmpdirname:
            audio_path = os.path.join(tmpdirname, temp_audio_path)
            clip.audio.write_audiofile(audio_path)
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            clip.close()
            shutil.rmtree(tmpdirname)

        logging.debug("Audio with was read successfully: {}.".format(mp4_path))

    except Exception as e:
        audio, sr = None, None

        logging.debug("Error occurred while reading {}: {}.".format(mp4_path, e))

    return audio, sr


def save_frames_as_movie(frames, mp4_path, fps=25):
    if not frames:
        raise ValueError("The list of frames is empty.")

    try:
        frames = [np.clip(frame, 0, 255).astype(np.uint8) for frame in frames]
        clip = ImageSequenceClip(list(frames), fps=fps)

        clip.write_videofile(
            mp4_path,
            codec=ENCODING,
            audio=False,
            bitrate="4000k",
            preset="slow",
            ffmpeg_params=["-crf", "18"],
        )

        clip.close()
        logging.debug("Saved video to: {}.".format(mp4_path))
    except Exception as e:
        logging.debug("Error occurred while saving: {}: {}.".format(mp4_path, e))
