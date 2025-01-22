import logging
import h5py
import lmdb
import pickle

import numpy as np
import cv2

from video_utils.video_cv2 import read_movie


LMDB_MAP_SIZE = 1 << 40


def video_to_lmdb(src, dst, size=None):
    logging.debug("Building frames lmdb for video {} at {}...".format(src, dst))
    frames = read_movie(src)

    env = lmdb.open(dst, map_size=LMDB_MAP_SIZE)
    with env.begin(write=True, buffers=True) as txn:
        for ind, frame in enumerate(frames):
            key = str(ind).zfill(10)
            if size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
            value = frame.astype(np.uint8)
            frame_bytes = cv2.imencode(".png", value)[1].tobytes()
            txn.put(key.encode("ascii"), frame_bytes)
    env.close()
    return ind


def save_frames_as_lmdb(frames, lmdb_path, size=None):
    logging.debug("Building frames lmdb for video at {}...".format(lmdb_path))

    env = lmdb.open(lmdb_path, map_size=LMDB_MAP_SIZE)
    with env.begin(write=True, buffers=True) as txn:
        for ind, frame in enumerate(frames):
            key = str(ind).zfill(10)
            if size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
            value = frame.astype(np.uint8)
            frame_bytes = cv2.imencode(".png", value)[1].tobytes()
            txn.put(key.encode("ascii"), frame_bytes)
    env.close()
    return ind


def extract_frames_from_lmdb(lmdb_file, frame_numbers):
    frames = list()

    env = lmdb.open(lmdb_file, readonly=True)
    for frame_number in frame_numbers:
        with env.begin() as txn:
            key = str(frame_number).zfill(10)
            value = txn.get(key.encode())
        frame = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
        frames.append(frame)
    env.close()
    return frames


def extract_all_frames_from_lmdb(lmdb_file):
    frames = list()

    env = lmdb.open(lmdb_file, readonly=True, max_dbs=0)
    with env.begin() as txn:
        with txn.cursor() as cursor:
            for _, value in cursor:
                frame = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
                frames.append(frame)
    env.close()
    return frames


def extract_frame_from_lmdb(lmdb_file, frame_number):
    env = lmdb.open(lmdb_file, readonly=True)
    with env.begin() as txn:
        key = str(frame_number).zfill(10)
        value = txn.get(key.encode())
    frame = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    env.close()

    return frame


def video2h5py(src, dst, size=None):
    logging.debug("Building frames sstable for video {} at {}...".format(src, dst))
    frames = read_movie(src)

    h5py_db = h5py.File(dst, "a")
    for ind, frame in enumerate(frames):
        if size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
        value = frame.astype(np.uint8)
        key = str(ind).zfill(10)
        h5py_db[key] = np.void(value)
    h5py_db.close()
