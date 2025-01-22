import os
import cv2
import tempfile
import numpy as np
import librosa
import shutil
import logging

from moviepy.video.io.VideoFileClip import VideoFileClip

ENCODING = "mp4v"
WIDTH_IDX = 0
HEIGHT_IDX = 1


def rmdir(dir_path):
    try:
        shutil.rmtree(dir_path)
    except Exception as e:
        print("An error occurred: {}".format(e))


def downrate_fps(input_file, output_file, target_fps, resize=None, max_size=None):
    try:
        video_capture = cv2.VideoCapture(input_file)

        if resize is None:
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            frame_width = resize[WIDTH_IDX]
            frame_height = resize[HEIGHT_IDX]

        original_fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(original_fps / target_fps))

        output_codec = cv2.VideoWriter_fourcc(*ENCODING)
        output_video = cv2.VideoWriter(
            output_file,
            output_codec,
            original_fps / float(frame_interval),
            (frame_width, frame_height),
        )

        frame_number, frame_written = 0, 0
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if (not ret) or (max_size and frame_written >= max_size):
                break

            if frame_number % frame_interval == 0:
                if not resize is None:
                    frame = cv2.resize(
                        frame,
                        (frame_width, frame_height),
                        interpolation=cv2.INTER_CUBIC,
                    )
                output_video.write(frame)
                frame_written += 1

            frame_number += 1

        video_capture.release()
        output_video.release()
        logging.debug(
            "Video with downrated fps saved successfully: {}.".format(input_file)
        )

    except Exception as e:
        logging.debug("Error occurred while downrate fps {}: {}.".format(input_file, e))


def extract_movie_segment(input_file, output_file, start_time, end_time):
    try:
        video_capture = cv2.VideoCapture(input_file)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        output_codec = cv2.VideoWriter_fourcc(*ENCODING)
        output_video = cv2.VideoWriter(
            output_file, output_codec, fps, (frame_width, frame_height)
        )

        frame_number = 0
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            if frame_number >= start_frame and frame_number <= end_frame:
                output_video.write(frame)

            frame_number += 1

            if frame_number > end_frame:
                break

        video_capture.release()
        output_video.release()
        logging.debug("Movie segment extracted successfully: {}.".format(input_file))

    except Exception as e:
        logging.debug(
            "Error occurred while extracted segment {}: {}.".format(input_file, e)
        )


def resize_movie(input_file, output_file, target_width, target_height):
    try:
        video_capture = cv2.VideoCapture(input_file, apiPreference=cv2.CAP_FFMPEG)
        original_fps = video_capture.get(cv2.CAP_PROP_FPS)

        output_codec = cv2.VideoWriter_fourcc(*ENCODING)
        output_video = cv2.VideoWriter(
            output_file, output_codec, original_fps, (target_width, target_height)
        )

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            resized_frame = cv2.resize(
                frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC
            )
            output_video.write(resized_frame)

        video_capture.release()
        output_video.release()
        logging.debug("Movie resized successfully: {}.".format(input_file))

    except Exception as e:
        logging.debug("Error occurred while resizing {}: {}.".format(input_file, e))


def extract_movie_info(input_file):
    try:
        video_capture = cv2.VideoCapture(input_file)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_capture.release()
        info = {
            "duration": duration,
            "resolution": [frame_width, frame_height],
            "fps": fps,
        }

    except Exception as e:
        logging.debug(
            "Error occurred while extracting movie information {}: {}.".format(
                input_file, e
            )
        )
        info = {"duration": None, "resolution": [None, None], "fps": None}

    return info


def read_movie(input_file, resize=None, max_size=None):
    try:
        frames = list()
        video_capture = cv2.VideoCapture(input_file)

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if (not ret) or max_size and len(frames) >= max_size:
                break

            if not resize is None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        video_capture.release()
        frames = np.array(frames)
        logging.debug("Video with was read successfully: {}.".format(input_file))

    except Exception as e:
        logging.debug("Error occurred while reading {}: {}.".format(input_file, e))
        frames = []

    return frames


def read_audio_from_mp4(mp4_path):
    audio, sr = [], None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as temp_audio_file:
            temp_audio_path = temp_audio_file.name

        video_clip = VideoFileClip(mp4_path)
        video_clip.audio.write_audiofile(temp_audio_path)

        audio, sr = librosa.load(temp_audio_path, sr=None, mono=True)
        video_clip.close()

        logging.debug(f"Audio was read successfully: {mp4_path}")

    except Exception as e:
        logging.error(f"Error occurred while reading audio from {mp4_path}: {e}")
        audio, sr = [], None

    finally:
        if "temp_audio_path" in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return audio, sr


def read_and_skip_movie(input_file, skip_frame=2, resize=None, max_size=None):
    try:
        frames = list()
        video_capture = cv2.VideoCapture(input_file)

        frame_number = 0
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if (not ret) or max_size and len(frames) >= max_size:
                break

            frame_number += 1
            if frame_number % skip_frame == 0:
                if not resize is None:
                    frame = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        video_capture.release()
        frames = np.array(frames)
        logging.debug("Video was read successfully: {}.".format(input_file))

    except Exception as e:
        logging.debug("Error occurred while reading {}: {}.".format(input_file, e))
        frames = None

    return frames


def save_frames_as_movie(frames, mp4_path, fps=25):
    if len(frames) == 0:
        raise ValueError("The list of frames is empty.")

    try:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*ENCODING)
        out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        logging.debug("Saved video to: {}.".format(mp4_path))

    except Exception as e:
        logging.debug("Error occurred while saving{}: {}.".format(mp4_path, e))


def resize_frames(frames, resize):
    resized_frames = list()

    try:
        for frame in frames:
            resized_frames.append(
                cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)
            )
        logging.debug("Resized successfully to: {}.".format("x".join(resize)))

    except Exception as e:
        logging.debug(
            "Error occurred while resizing {}: {}.".format("x".join(resize, e))
        )

    return resized_frames
