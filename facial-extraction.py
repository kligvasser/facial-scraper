import gc
import os
import re
import torch
import argparse
import pandas as pd
import multiprocessing as mp
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from face.extractor import FaceExtractor
from face.clipper import CLIPImageEmbedder
from face.iqa import IQAQualityAssessor
from video_utils import video_cv2
from video_utils import transcoding_utils


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
LMDB_SKIP_FRAME = 2
MAX_PROCESS_PER_GPU = 1


def list_all_recursive(directory):
    return [str(p) for p in Path(directory).rglob("*") if p.is_file()]


def parse_existing_extracted_files(output_folder: str, file_path: str, output_dir: str):
    """
    Reads existing .mp4 blocks from `output_folder`, parses their filename
    to build metadata rows. This allows skipping expensive extraction if
    blocks already exist, including reusing IQA ("hyperiqa", "clipiqa+")
    scores from the filenames.

    The expected filename pattern is something like:
      {video_name}_block_{i+1}
      _start_{start_time:.2f}_end_{end_time:.2f}
      _frames_{len(face_frames)}_res_{width}x{height}
      _origdur_{duration:.2f}_origres_{resolution[0]}x{resolution[1]}
      _origfps_{fps:.2f}_clipscore_{clip_score:.4f}
      _iqa_hyperiqa_{hyper_val:.4f}_iqa_clipiqa+_{clip_val:.4f}.mp4
    """
    rows = []
    video_name = Path(file_path).stem

    filename_pattern = re.compile(
        r"_block_(\d+)"  # (1) block_idx
        r"_start_([\d.]+)_end_([\d.]+)"  # (2,3) start_time, end_time
        r"_frames_(\d+)_res_(\d+)x(\d+)"  # (4,5,6) num_frames, width, height
        r"_origdur_([\d.]+)_origres_(\d+)x(\d+)"  # (7,8,9) original_duration, orig_w, orig_h
        r"_origfps_([\d.]+)_clipscore_([\d.]+)"  # (10,11) original_fps, clip_score
        r"_iqa_hyperiqa_([\d.]+)"  # (12) hyperiqa
        r"_iqa_clipiqa\+_([\d.]+)\.mp4$"  # (13) clipiqa+ (note the escaped '+')
    )

    if not os.path.isdir(output_folder):
        return rows

    all_files = os.listdir(output_folder)
    for fname in all_files:
        if fname.endswith(".mp4") and fname.startswith(video_name + "_block_"):
            match = filename_pattern.search(fname)
            if not match:
                continue

            block_idx = int(match.group(1))
            start_time = float(match.group(2))
            end_time = float(match.group(3))
            num_frames = int(match.group(4))
            width = int(match.group(5))
            height = int(match.group(6))
            original_duration = float(match.group(7))
            orig_w = int(match.group(8))
            orig_h = int(match.group(9))
            original_fps = float(match.group(10))
            clip_score = float(match.group(11))
            hyper_val = float(match.group(12))
            clip_plus_val = float(match.group(13))

            output_path = os.path.join(output_folder, fname)

            audio_path = output_path.replace(".mp4", ".wav")
            audio_relative = None
            if os.path.exists(audio_path):
                audio_relative = os.path.relpath(audio_path, output_dir)

            lmdb_path = output_path.replace(".mp4", ".lmdb")
            extracted_lmdb = None
            lmdb_num_frames = None
            if os.path.exists(lmdb_path):
                extracted_lmdb = os.path.relpath(lmdb_path, output_dir)

            row = {
                "file_original": file_path,
                "file_relative": os.path.relpath(output_path, output_dir),
                "lmdb_file": lmdb_path if os.path.exists(lmdb_path) else None,
                "lmdb_relative": extracted_lmdb,
                "lmdb_num_frames": lmdb_num_frames,
                "audio_file": audio_path if os.path.exists(audio_path) else None,
                "audio_relative": audio_relative,
                "start_frame": None,
                "end_frame": None,
                "start_time": start_time,
                "end_time": end_time,
                "num_frames": num_frames,
                "width": width,
                "height": height,
                "original_duration": original_duration,
                "original_resolution": [orig_w, orig_h],
                "original_fps": original_fps,
                "clip_score": clip_score,
                "hyperiqa": hyper_val,
                "clipiqa+": clip_plus_val,
            }
            rows.append(row)

    return rows


def process_movie(
    file_path, output_dir, device, make_lmdb, skip_face_detection, include_audio
):
    rows = []

    face_extractor = None
    clip_scorer = None
    iqa_scorer = None

    video_name = Path(file_path).stem
    output_folder = os.path.join(output_dir, video_name)

    existing_rows = parse_existing_extracted_files(
        output_folder=output_folder, file_path=file_path, output_dir=output_dir
    )
    if len(existing_rows) > 0:
        return existing_rows

    try:
        torch.cuda.set_device(device)

        clip_info = video_cv2.extract_movie_info(file_path)
        duration = clip_info.get("duration")
        resolution = clip_info.get("resolution", [None, None])
        fps = clip_info.get("fps")

        frames = video_cv2.read_movie(file_path)

        if include_audio:
            audio, sr = video_cv2.read_audio_from_mp4(file_path)
        else:
            audio, sr = [], None

        if len(frames) == 0:
            return []

        if skip_face_detection:
            faces = [frames]
            indexes = [(0, len(frames) - 1)]
        else:
            face_extractor = FaceExtractor(device=device)
            faces, indexes = face_extractor.extract(frames)

        clip_scorer = CLIPImageEmbedder(device=device)
        iqa_scorer = IQAQualityAssessor(device=device)

        os.makedirs(output_folder, exist_ok=True)

        for i, (face_frames, (start_idx, end_idx)) in enumerate(zip(faces, indexes)):
            if len(face_frames) == 0:
                continue

            height, width, _ = face_frames[0].shape
            start_time = float(start_idx) / fps
            end_time = float(end_idx) / fps

            clip_score = clip_scorer.compute_clip_score(face_frames, k=5)

            iqa_scores = iqa_scorer(
                face_frames, k=2
            )  # => {"hyperiqa": X, "clipiqa+": Y}
            hyper_val = iqa_scores["hyperiqa"]
            clip_plus_val = iqa_scores["clipiqa+"]

            output_filename = (
                f"{video_name}_block_{i+1}"
                f"_start_{start_time:.2f}_end_{end_time:.2f}"
                f"_frames_{len(face_frames)}_res_{width}x{height}"
                f"_origdur_{duration:.2f}_origres_{resolution[0]}x{resolution[1]}"
                f"_origfps_{fps:.2f}_clipscore_{clip_score:.4f}"
                f"_iqa_hyperiqa_{hyper_val:.4f}_iqa_clipiqa+_{clip_plus_val:.4f}.mp4"
            )
            output_path = os.path.join(output_folder, output_filename)

            video_cv2.save_frames_as_movie(face_frames, output_path, fps)

            if sr is not None and len(audio) > 0:
                audio_path = output_path.replace(".mp4", ".wav")
                audio_relative = os.path.relpath(audio_path, output_dir)
                audio_segment = audio[int(start_time * sr) : int(end_time * sr)]
                sf.write(audio_path, audio_segment, sr)
            else:
                audio_path = None
                audio_relative = None

            if make_lmdb:
                face_frames_ = face_frames[::LMDB_SKIP_FRAME]
                lmdb_path = output_path.replace(".mp4", ".lmdb")
                extracted_lmdb = os.path.relpath(lmdb_path, output_dir)
                lmdb_num_frames = len(face_frames_)
                transcoding_utils.save_frames_as_lmdb(face_frames_, lmdb_path)
            else:
                lmdb_path = None
                extracted_lmdb = None
                lmdb_num_frames = None

            row = {
                "file_original": file_path,
                "file_relative": os.path.relpath(output_path, output_dir),
                "lmdb_file": lmdb_path,
                "lmdb_relative": extracted_lmdb,
                "lmdb_num_frames": lmdb_num_frames,
                "audio_file": audio_path,
                "audio_relative": audio_relative,
                "start_frame": start_idx,
                "end_frame": end_idx,
                "start_time": start_time,
                "end_time": end_time,
                "num_frames": len(face_frames),
                "width": width,
                "height": height,
                "original_duration": duration,
                "original_resolution": resolution,
                "original_fps": fps,
                "clip_score": clip_score,
                "hyperiqa": hyper_val,
                "clipiqa+": clip_plus_val,
            }
            rows.append(row)

    except Exception as e:
        print(f"[Failed] Movie {file_path} with error: {e}")

    finally:
        del face_extractor
        del clip_scorer
        del iqa_scorer
        gc.collect()

    return rows


def validate_cuda_devices(cuda_devices):
    """
    Validates the provided CUDA devices against those that are actually available.
    """
    num_devices = torch.cuda.device_count()
    valid_devices = [f"cuda:{i}" for i in range(num_devices)]

    for device in cuda_devices:
        if device not in valid_devices:
            raise ValueError(
                f"Invalid CUDA device: {device}. Available devices are: {valid_devices}"
            )


def process_wrapper(args):
    """
    Wrapper for multiprocessing: unpacks arguments and calls `process_movie`.
    """
    file_path, output_dir, device, make_lmdb, skip_face_detection, include_audio = args
    return process_movie(
        file_path, output_dir, device, make_lmdb, skip_face_detection, include_audio
    )


def parallel_face_extraction(
    input_dir,
    output_dir,
    cuda_devices,
    make_lmdb,
    num_processes=4,
    skip_face_detection=False,
    include_audio=True,
):
    """
    Main pipeline for parallel face extraction across multiple videos,
    with optional LMDB creation and audio extraction.
    """
    mp.set_start_method("spawn", force=True)

    validate_cuda_devices(cuda_devices)
    total_gpus = len(cuda_devices)

    num_processes = min(num_processes, total_gpus * MAX_PROCESS_PER_GPU)

    video_files = [
        f for f in list_all_recursive(input_dir) if f.lower().endswith(VIDEO_EXTENSIONS)
    ]

    os.makedirs(output_dir, exist_ok=True)

    device_mapping = [
        int(cuda_devices[i % total_gpus].split(":")[1]) for i in range(num_processes)
    ]

    task_args = [
        (
            file_path,
            output_dir,
            device_mapping[i % num_processes],
            make_lmdb,
            skip_face_detection,
            include_audio,
        )
        for i, file_path in enumerate(video_files)
    ]

    with mp.Pool(processes=num_processes) as pool:
        all_rows = list(
            tqdm(
                pool.imap_unordered(process_wrapper, task_args),
                total=len(video_files),
                desc="Processing videos",
            )
        )

    metadata = [row for rows in all_rows for row in rows]
    metadata_df = pd.DataFrame(metadata)

    metadata_csv = os.path.join(output_dir, "ex-metadata.csv")
    metadata_df.to_csv(metadata_csv, index=False)

    metadata_pickle = os.path.join(output_dir, "ex-metadata.pkl")
    metadata_df.to_pickle(metadata_pickle)

    print("Extraction complete.")
    print("Metadata saved to:", metadata_csv)
    print("Metadata DataFrame saved to:", metadata_pickle)


def get_arguments():
    """
    Parses CLI arguments when running this script directly.
    """
    example_text = """
    Example:
       python facial_extraction_updated.py --num-processes 8 --include-audio 
           --input-dir downloads/yt-search/2025-01-30_11-07-27 
           --output-dir downloads/yt-search/2025-01-30_11-07-27-ex
    """

    parser = argparse.ArgumentParser(
        description="Extract stable facial frames from videos, with IQA and CLIP scoring.",
        epilog=example_text,
    )
    parser.add_argument(
        "--input-dir", required=True, help="Path to folder containing input videos"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to folder to save extracted videos and metadata",
    )
    parser.add_argument(
        "--num-processes", type=int, default=4, help="Number of parallel processes"
    )
    parser.add_argument(
        "--cuda-devices",
        nargs="+",
        default=[f"cuda:{i}" for i in range(0, 8)],
        help="List of CUDA devices to use (e.g., cuda:0 cuda:1)",
    )
    parser.add_argument(
        "--make-lmdb",
        action="store_true",
        help="Store extracted frames in LMDB format as well.",
    )
    parser.add_argument(
        "--skip-face-detection",
        action="store_true",
        help="Skip face detection if clips already contain extracted faces",
    )
    parser.add_argument(
        "--include-audio",
        action="store_true",
        help="Extract and save audio segments as .wav alongside .mp4 blocks.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    parallel_face_extraction(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cuda_devices=args.cuda_devices,
        make_lmdb=args.make_lmdb,
        num_processes=args.num_processes,
        skip_face_detection=args.skip_face_detection,
        include_audio=args.include_audio,
    )
