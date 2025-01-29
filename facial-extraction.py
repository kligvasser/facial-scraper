import torch
import os
import argparse
import pandas as pd
import multiprocessing as mp
import soundfile as sf
from tqdm import tqdm
from face.extractor import FaceExtractor
from face.clipper import CLIPImageEmbedder
from face.iqa import IQAQualityAssessor
from video_utils import video_cv2
from video_utils import transcoding_utils


LMDB_SKIP_FRAME = 2


def list_all_recursive(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            all_files.append(os.path.join(root, name))
        for name in dirs:
            all_files.append(os.path.join(root, name))
    return all_files


def process_movie(file_path, output_dir, device, make_lmdb, skip_face_detection):
    rows = []

    try:
        torch.cuda.set_device(device)

        clip_info = video_cv2.extract_movie_info(file_path)
        duration = clip_info.get("duration")
        resolution = clip_info.get("resolution", [None, None])
        fps = clip_info.get("fps")

        clip_scorer = CLIPImageEmbedder(device=device)
        iqa_scorer = IQAQualityAssessor(device=device)

        frames = video_cv2.read_movie(file_path)
        audio, sr = video_cv2.read_audio_from_mp4(file_path)

        if len(frames) == 0:
            return []

        if skip_face_detection:
            faces = [frames]
            indexes = [(0, len(frames) - 1)]
        else:
            face_extractor = FaceExtractor(device=device)
            faces, indexes = face_extractor.extract(frames)

        video_name = os.path.basename(file_path).split(".")[0]
        output_folder = os.path.join(output_dir, video_name)
        os.makedirs(output_folder, exist_ok=True)

        for i, (face_frames, (start_idx, end_idx)) in enumerate(zip(faces, indexes)):
            if len(face_frames) != 0:
                height, width, _ = face_frames[0].shape
                start_time = float(start_idx) / fps
                end_time = float(end_idx) / fps
                clip_score = clip_scorer.compute_clip_score(face_frames)
                iqa_scores = iqa_scorer(face_frames)

                output_filename = (
                    f"{video_name}_block_{i+1}"
                    f"_start_{start_time:.2f}_end_{end_time:.2f}"
                    f"_frames_{len(face_frames)}_res_{width}x{height}"
                    f"_origdur_{duration:.2f}_origres_{resolution[0]}x{resolution[1]}"
                    f"_origfps_{fps:.2f}_clipscore_{clip_score:.4f}.mp4"
                )
                output_path = os.path.join(output_folder, output_filename)
                video_cv2.save_frames_as_movie(face_frames, output_path, fps)

                if not sr is None:
                    audio_path = output_path.replace(".mp4", ".wav")
                    audio_relative = os.path.relpath(audio_path, output_dir)
                    audio_ = audio[int(start_time * sr) : int(end_time * sr)]
                    sf.write(audio_path, audio_, sr)
                else:
                    audio_path = None
                    audio_relative = None

                if make_lmdb:
                    face_frames_ = face_frames[::LMDB_SKIP_FRAME]
                    lmdb_path = output_path.replace(".mp4", ".lmdb")
                    extracted_lmdb = os.path.relpath(lmdb_path, output_dir)
                    lmdb_num_frames = len(face_frames_)
                    transcoding_utils.save_frames_as_lmdb(
                        face_frames_,
                        lmdb_path,
                    )
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
                }
                row.update(iqa_scores)
                rows.append(row)

    except Exception as e:
        print(f" [Failed] Movie {file_path} with the error: {e}")

    return rows


def validate_cuda_devices(cuda_devices):
    num_devices = torch.cuda.device_count()
    valid_devices = [f"cuda:{i}" for i in range(num_devices)]

    for device in cuda_devices:
        if device not in valid_devices:
            raise ValueError(
                f"Invalid CUDA device: {device}. Available devices are: {valid_devices}"
            )


def parallel_face_extraction(
    input_dir,
    output_dir,
    cuda_devices,
    make_lmdb,
    num_processes=4,
    skip_face_detection=False,
):
    mp.set_start_method("spawn", force=True)

    video_files = [
        os.path.join(input_dir, f)
        for f in list_all_recursive(input_dir)
        if (skip_face_detection or "_part_" in f)
        and f.endswith((".mp4", ".avi", ".mov"))
    ]

    os.makedirs(output_dir, exist_ok=True)

    device_mapping = [
        int(device.split(":")[1])
        for device in (
            cuda_devices[i % len(cuda_devices)] for i in range(len(video_files))
        )
    ]

    with mp.Pool(processes=num_processes) as pool:
        all_rows = list(
            tqdm(
                pool.starmap(
                    process_movie,
                    [
                        (file, output_dir, device, make_lmdb, skip_face_detection)
                        for file, device in zip(video_files, device_mapping)
                    ],
                ),
                total=len(video_files),
                desc="Processing videos",
            )
        )

    metadata = [row for rows in all_rows for row in rows]
    metadata_df = pd.DataFrame(metadata)

    metadata_csv = os.path.join(output_dir, "metadata.csv")
    metadata_df.to_csv(metadata_csv, index=False)

    metadata_pickle = os.path.join(output_dir, "metadata.pkl")
    metadata_df.to_pickle(metadata_pickle)

    print("Extraction complete.")
    print("Metadata saved to:", metadata_csv)
    print("Metadata DataFrame saved to:", metadata_pickle)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Extract stable facial frames from videos"
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
        default=[f"cuda:{i}" for i in range(3, 8)],
        help="List of CUDA devices to use (e.g., cuda:0 cuda:1)",
    )
    parser.add_argument("--make-lmdb", action="store_true")
    parser.add_argument(
        "--skip-face-detection",
        action="store_true",
        help="Skip face detection if clips already contain extracted faces",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    parallel_face_extraction(
        args.input_dir,
        args.output_dir,
        args.cuda_devices,
        args.make_lmdb,
        args.num_processes,
        args.skip_face_detection,
    )
