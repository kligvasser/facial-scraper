import time
import socket
import os
import IPython.display as display
from PIL import Image

import yt_dlp
import logging


YOUTUBE_URL = "https://www.youtube.com/watch?v="
YOUTUBE_FORMAT = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"


def create_cookies():
    try:
        logging.info("Trying to use browser cookies for authentication...")
        with yt_dlp.YoutubeDL({"cookies-from-browser": "chrome"}) as ydl:
            ydl.download(["https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
        logging.info("Using Chrome cookies for YouTube authentication.")
    except Exception:
        logging.info("Failed to get cookies from browser, trying manual cookies.txt...")


def is_tor_running():
    try:
        with socket.create_connection(("127.0.0.1", 9050), timeout=2):
            return True
    except (socket.error, socket.timeout):
        return False


def is_cookies_file_available():
    return os.path.exists("cookies.txt")


def ytdownload(video_id, target_mp4, format=YOUTUBE_FORMAT):
    url = YOUTUBE_URL + video_id
    ydl_opts = {
        "format": format,
        "outtmpl": target_mp4,
        "ratelimit": 5000000,
        "sleep_interval": 2,
        "max_sleep_interval": 5,
        "retries": 5,
        "fragment_retries": 5,
        "nocheckcertificate": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36"
        },
    }

    if is_tor_running():
        logging.info("Tor detected! Using Tor proxy for download.")
        ydl_opts["proxy"] = "socks5://127.0.0.1:9050"
    else:
        logging.info("Tor not detected. Downloading normally.")

    if is_cookies_file_available():
        ydl_opts["cookies"] = "cookies.txt"
        logging.info("Using manually exported cookies from cookies.txt.")
    else:
        logging.warning("No cookies available. Some videos may be blocked.")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logging.info("Skipping {}, Error {}, {}".format(url, e, type(e)))
        return False

    logging.info("Downloaded {}".format(url))
    return True


def animate_frames(frames, sleep=0):
    for frame in frames:
        img = Image.fromarray(frame)
        display.display(img)
        time.sleep(sleep)
        display.clear_output(wait=True)


def slice_to_chunks(lst, k):
    for i in range(0, len(lst), k):
        yield lst[i : i + k]
