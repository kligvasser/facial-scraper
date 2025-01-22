import requests

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from urllib.parse import urlparse, parse_qs


def is_url(path):
    parsed = urlparse(path)
    return bool(parsed.scheme) and bool(parsed.netloc)


def scrapt_channel_id_from_handle(handle):
    url = "https://www.youtube.com/" + handle
    resp = requests.get(url)

    if resp.status_code == 200:
        return resp.text.split('"channelId":"')[1].split('",')[0]
    else:
        return None


class YouTubeAPI:
    def __init__(self, api_key):
        self.youtube = build("youtube", "v3", developerKey=api_key)

    def get_recently_uploaded_videos_by_category(self, category_id, max_results=50):
        try:
            video_response = (
                self.youtube.videos()
                .list(
                    part="snippet",
                    chart="mostPopular",
                    videoCategoryId=category_id,
                    maxResults=max_results,
                )
                .execute()
            )

            videos = list()
            for video in video_response.get("items", []):
                videos.append(f'https://www.youtube.com/watch?v={video["id"]}')
            return videos
        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
            return None

    def get_video_information(self, video_url):
        video_id = self.extract_video_id(video_url)
        try:
            video_response = (
                self.youtube.videos()
                .list(part="snippet,statistics,contentDetails", id=video_id)
                .execute()
            )

            if video_response["items"]:
                video = video_response['items'][0]
                is_hd = video["contentDetails"]["definition"] == "hd"
                return {
                    "url": video_url,
                    "video_id": video_id,
                    'title': video['snippet']['title'],
                    'views': video['statistics']['viewCount'],
                    'likes': video['statistics'].get('likeCount'),
                    'description': video['snippet']['description'],
                    'is_hd': is_hd,
                    'duration': video['contentDetails']['duration'],
                    'dimension': video['contentDetails']['dimension'],
                    'definition': video['contentDetails']['definition'],  # 'hd' or 'sd'
                }
            else:
                return None
        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
            return None

    def get_video_captions(self, video_url):
        video_id = self.extract_video_id(video_url)
        try:
            captions_list = self.youtube.captions().list(part="snippet", videoId=video_id).execute()

            captions = list()
            for caption in captions_list.get("items", []):
                captions.append(caption["snippet"])
            return captions
        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
            return None

    @staticmethod
    def extract_video_id(url):
        """Extract the video ID from a YouTube URL."""
        parsed_url = urlparse(url)
        video_id = parse_qs(parsed_url.query).get("v")
        if video_id:
            return video_id[0]
        return None

    def get_all_videos_in_playlist(self, playlist_id):
        videos = list()
        next_page_token = None

        try:
            while True:
                playlist_items_response = (
                    self.youtube.playlistItems()
                    .list(
                        part="snippet",
                        playlistId=playlist_id,
                        maxResults=50,
                        pageToken=next_page_token,
                    )
                    .execute()
                )

                for item in playlist_items_response.get("items", []):
                    video_id = item["snippet"]["resourceId"]["videoId"]
                    videos.append(f'https://www.youtube.com/watch?v={video_id}')

                next_page_token = playlist_items_response.get("nextPageToken")
                if not next_page_token:
                    break

            return videos
        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
            return None

    def get_all_playlists_in_channel(self, channel_id):
        if is_url(channel_id):
            channel_id = urlparse(channel_id).path.rsplit("/", 1)[-1]

        playlists = list()
        next_page_token = None

        try:
            while True:
                playlists_response = (
                    self.youtube.playlists()
                    .list(
                        part="snippet",
                        channelId=channel_id,
                        maxResults=50,
                        pageToken=next_page_token,
                    )
                    .execute()
                )

                for playlist in playlists_response.get("items", []):
                    playlists.append(
                        {
                            "kind": playlist["kind"],
                            "playlist_id": playlist["id"],
                            "url": f'https://www.youtube.com/playlist?list={playlist["id"]}',
                        }
                    )

                next_page_token = playlists_response.get("nextPageToken")
                if not next_page_token:
                    break

            return playlists
        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
            return None

    def get_channel_id_by_handle(self, handle, num_ids=5):
        try:
            response = (
                self.youtube.search().list(part="snippet", type="channel", q=handle).execute()
            )

            items = response.get("items", [])
            return [item["snippet"]["channelId"] for item in items[:num_ids]]

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def search_youtube(self, search_term, max_results=20):
        try:
            search_response = (
                self.youtube.search()
                .list(q=search_term, part="id", maxResults=max_results, type="video")
                .execute()
            )

            video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
            return video_ids

        except HttpError as e:
            print(f"An HTTP error occurred: {e.resp.status} {e.content}")
            return []


if __name__ == "__main__":
    api_key = ""
    youtube_api = YouTubeAPI(api_key)

    videos = youtube_api.get_recently_uploaded_videos_by_category("10")
    print(videos)

    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    video_info = youtube_api.get_video_information(video_url)
    print(video_info)

    captions = youtube_api.get_video_captions(video_url)
    print(captions)

    videos = youtube_api.get_all_videos_in_playlist("PL8E54R76rowDjPwvbpgQUkf7TgHJUPS1J")
    print(videos)

    playlists = youtube_api.get_all_playlists_in_channel(
        "UCKJSif-_lZBCzPYRQU0n86g"
    )  # "DisneyMusicVEVO"
    print(playlists)

    channel_id = youtube_api.get_channel_id_by_handle("now14")
    print(channel_id)

    channel_id = scrapt_channel_id_from_handle("@now14")
    print(channel_id)
