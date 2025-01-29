import os
import argparse
import datetime
import subprocess
import pandas as pd
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip

import video_utils.misc as video_misc


YOUTUBE_FORMAT = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"


def split_video_to_clips(input_video_path, clip_duration_minutes, output_dir):
    clip_duration_seconds = clip_duration_minutes * 60
    video = VideoFileClip(input_video_path)
    video_duration = int(video.duration)
    num_clips = (video_duration + clip_duration_seconds - 1) // clip_duration_seconds

    for i in range(num_clips):
        start_time = i * clip_duration_seconds
        end_time = min((i + 1) * clip_duration_seconds, video_duration)
        output_filename = os.path.join(
            output_dir,
            os.path.basename(input_video_path).replace(".mp4", "") + f"_part_{i+1}.mp4",
        )

        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_video_path,
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-c",
            "copy",
            output_filename,
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(
                f"Error in splitting video: {output_filename}\n{result.stderr.decode()}"
            )

    return num_clips


def download_video_from_id(row, args):
    video_id = row["video_id"]
    video_folder = os.path.join(args.session_folder, video_id)
    result = {
        "video_id": video_id,
        "downloaded": False,
        "failed": False,
        "path": None,
        "num_clips": 0,
    }

    if os.path.exists(video_folder):
        return result

    os.makedirs(video_folder, exist_ok=True)
    video_path = os.path.join(video_folder, f"{video_id}.mp4")

    try:
        if video_misc.ytdownload(video_id, video_path, YOUTUBE_FORMAT):
            num_clips = split_video_to_clips(
                video_path, args.clip_duration, video_folder
            )
            result.update(
                {
                    "downloaded": True,
                    "path": os.path.relpath(video_path, args.records_dir),
                    "num_clips": num_clips,
                }
            )
        else:
            raise RuntimeError(f"Failed to download video {video_id}")
    except Exception as e:
        result["failed"] = True
        print(f"Error processing video {video_id}: {e}")

    return result


def get_arguments():
    example_text = """
    Example:
        python yt-download.py --records-dir ./downloads --urls urls/faces/yt-@Oscars.csv --num-videos 5
    """

    parser = argparse.ArgumentParser(
        description="Download and process YouTube videos",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--urls", required=True, help="Path to CSV containing video IDs"
    )
    parser.add_argument(
        "--records-dir", required=True, help="Root directory for downloaded videos"
    )
    parser.add_argument(
        "--clip-duration", type=int, default=1, help="Clip duration in minutes"
    )
    parser.add_argument(
        "--num-videos", type=int, default=0, help="Number of videos to download"
    )
    parser.add_argument(
        "--num-processes", type=int, default=8, help="Number of parallel processes"
    )
    return parser.parse_args()


def main():
    args = get_arguments()

    os.makedirs(args.records_dir, exist_ok=True)
    args.session_folder = os.path.join(
        args.records_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(args.session_folder, exist_ok=True)

    df = pd.read_csv(args.urls)
    metadata_path = os.path.join(args.records_dir, "metadata.csv")

    if os.path.exists(metadata_path):
        df_metadata = pd.read_csv(metadata_path, index_col="video_id")
    else:
        df_metadata = pd.DataFrame(
            columns=["video_id", "downloaded", "failed", "path", "num_clips"]
        ).set_index("video_id")

    df_metadata["downloaded"] = df_metadata.get("downloaded", False)
    df_metadata["failed"] = df_metadata.get("failed", False)

    new_videos = df[
        ~df["video_id"].isin(
            df_metadata.index[df_metadata["downloaded"] | df_metadata["failed"]]
        )
    ]

    if args.num_videos > 0:
        new_videos = new_videos.head(args.num_videos)

    data = [row for _, row in new_videos.iterrows()]

    with mp.Pool(processes=args.num_processes) as pool:
        func_partial = partial(download_video_from_id, args=args)
        results = list(
            tqdm(
                pool.imap(func_partial, data),
                total=len(data),
                desc="Downloading videos",
                unit="video",
            )
        )

    for result in results:
        video_id = result["video_id"]
        df_metadata.loc[video_id] = result

    df_metadata.to_csv(metadata_path)


if __name__ == "__main__":
    main()
