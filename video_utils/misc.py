import time
import IPython.display as display
from PIL import Image

import yt_dlp
import logging


YOUTUBE_URL = "https://www.youtube.com/watch?v="
YOUTUBE_FORMAT = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"


def ytdownload(video_id, target_mp4, format=YOUTUBE_FORMAT):
    url = YOUTUBE_URL + video_id
    ydl_opts = {
        "format": format,
        "outtmpl": target_mp4,
    }
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
