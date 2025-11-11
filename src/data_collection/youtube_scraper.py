import time
import pandas as pd
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from src.config import YT_API_KEY, CHANNEL_ID, VIDEOS_CSV, COMMENTS_CSV
from src.data_collection.utils import read_csv_safe, write_csv_safe, merge_unique_comments


# -------------------------- LOGGER --------------------------
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


# ----------------------- CLIENT SETUP ------------------------
def youtube_client(api_key):
    if not api_key:
        raise RuntimeError("Missing YouTube API Key. Please set YT_API_KEY in .env.")
    return build("youtube", "v3", developerKey=api_key)


# ------------------ FETCH VIDEO METADATA ---------------------
def fetch_videos(youtube, channel_id):
    """Fetch full list of videos and metadata for the given channel."""
    all_videos = []
    next_page_token = None

    while True:
        request = youtube.search().list(
            part="id",
            channelId=channel_id,
            maxResults=50,
            order="date",
            type="video",
            pageToken=next_page_token
        )
        response = request.execute()

        video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
        if not video_ids:
            break

        # Fetch video details for each batch
        stats_request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids)
        )
        stats_response = stats_request.execute()

        for item in stats_response.get("items", []):
            snippet = item["snippet"]
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})

            # Convert numeric fields safely
            view_count = int(stats.get("viewCount", 0))
            like_count = int(stats.get("likeCount", 0))
            comment_count = int(stats.get("commentCount", 0))

            # Calculate total engagement
            engagement_total = like_count + comment_count

            all_videos.append({
                "video_id": item["id"],
                "title": snippet.get("title"),
                "published_at": snippet.get("publishedAt"),
                "duration": content.get("duration"),
                "viewCount": view_count,
                "likeCount": like_count,
                "commentCount": comment_count,
                "engagement_total": engagement_total,
                "tags": ",".join(snippet.get("tags", [])),
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.2)  # Prevent quota exhaustion

    # Convert published_at column to normal datetime (no timezone)
    df = pd.DataFrame(all_videos)
    if not df.empty and "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df["published_at"] = df["published_at"].dt.tz_localize(None)  # Remove UTC info

    return df


# ------------------ FETCH COMMENTS & REPLIES -----------------
def fetch_comments_for_video(youtube, video_id):
    """Fetch all comments and replies for a given video."""
    comments = []
    next_page_token = None

    while True:
        try:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText",
                pageToken=next_page_token
            )
            response = request.execute()
        except HttpError as e:
            if e.resp.status == 403 and "commentsDisabled" in str(e):
                log(f"Comments disabled for video {video_id}. Skipping.")
                return pd.DataFrame()  # Skip video
            else:
                log(f"Error fetching comments for {video_id}: {e}")
                return pd.DataFrame()

        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "comment_id": item["id"],
                "author": top.get("authorDisplayName"),
                "text": top.get("textOriginal"),
                "like_count": top.get("likeCount"),
                "published_at": top.get("publishedAt"),
                "reply_to": None  # top-level comment
            })

            # Extract replies (if any)
            replies = item.get("replies", {}).get("comments", [])
            for reply in replies:
                r_sn = reply["snippet"]
                comments.append({
                    "video_id": video_id,
                    "comment_id": reply["id"],
                    "author": r_sn.get("authorDisplayName"),
                    "text": r_sn.get("textOriginal"),
                    "like_count": r_sn.get("likeCount"),
                    "published_at": r_sn.get("publishedAt"),
                    "reply_to": item["id"]
                })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.05)

    # Convert published_at column to normal datetime (no timezone)
    df = pd.DataFrame(comments)
    if not df.empty and "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df["published_at"] = df["published_at"].dt.tz_localize(None)  # Remove UTC info

    return df


# ------------------- MAIN SCRAPING PIPELINE ------------------
def scrape_channel(channel_id=CHANNEL_ID, api_key=YT_API_KEY, replace_comments=True):
    start_time = datetime.now()
    log("🚀 Starting YouTube scraping pipeline...")

    youtube = youtube_client(api_key)

    # Phase 1: Fetch videos
    log("Fetching video metadata...")
    videos_df = fetch_videos(youtube, channel_id)
    write_csv_safe(videos_df, VIDEOS_CSV)
    log(f"Saved {len(videos_df)} videos to {VIDEOS_CSV}.")

    # Phase 2: Fetch comments
    all_comments = []
    for idx, row in videos_df.iterrows():
        vid = row["video_id"]
        title = row["title"]
        log(f"Fetching comments for [{title}] ({idx + 1}/{len(videos_df)}) ...")
        comments_df = fetch_comments_for_video(youtube, vid)
        if not comments_df.empty:
            all_comments.append(comments_df)
        time.sleep(0.1)

    if all_comments:
        new_comments = pd.concat(all_comments, ignore_index=True)
    else:
        new_comments = pd.DataFrame(columns=[
            "video_id", "comment_id", "author", "text", "like_count", "published_at", "reply_to"
        ])

    if replace_comments:
        write_csv_safe(new_comments, COMMENTS_CSV)
        log(f"✅ Replaced {COMMENTS_CSV} with {len(new_comments)} comments.")
    else:
        existing = read_csv_safe(COMMENTS_CSV)
        merged = merge_unique_comments(new_comments, existing)
        write_csv_safe(merged, COMMENTS_CSV)
        log(f"Merged comments saved to {COMMENTS_CSV} (total {len(merged)})")

    # Phase 3: Timing summary
    end_time = datetime.now()
    duration = end_time - start_time
    log(f"Scraping started at: {start_time}")
    log(f"Scraping ended at: {end_time}")
    log(f"Total duration: {duration}")

    log("YouTube scraping completed successfully!")


if __name__ == "__main__":
    scrape_channel()
