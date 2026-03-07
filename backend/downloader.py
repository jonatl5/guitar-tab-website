from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

def download_video(url: str, output_dir: str = "data/videos/") -> str:
    """
    Download a video URL to MP4 using yt-dlp and return an absolute path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import yt_dlp
    except Exception as exc:
        raise RuntimeError(
            "yt-dlp is not installed. Install dependencies from requirements.txt first."
        ) from exc

    normalized_url = _normalize_video_url(url)

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        # Helps with recent YouTube web-client/SABR 403 issues.
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        "retries": 5,
        "fragment_retries": 5,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(normalized_url, download=True)
            downloaded_path = ydl.prepare_filename(info)

        # If merged by ffmpeg, final path is mp4.
        final_path = Path(downloaded_path).with_suffix(".mp4")
        if not final_path.exists():
            # Fallback when extractor already gave final extension.
            candidate = Path(downloaded_path)
            if candidate.exists():
                final_path = candidate
            else:
                raise FileNotFoundError("Download completed but output file was not found.")

        return str(final_path.resolve())
    except Exception as exc:
        raise RuntimeError(f"Failed to download video from URL: {normalized_url}. Error: {exc}") from exc


def _normalize_video_url(url: str) -> str:
    """
    Normalize YouTube URLs to a direct watch link when possible.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if "youtube.com" in host or "youtu.be" in host:
        if "youtu.be" in host:
            video_id = parsed.path.strip("/")
            if video_id:
                return f"https://www.youtube.com/watch?v={video_id}"
        else:
            query = parse_qs(parsed.query)
            video_id = (query.get("v") or [None])[0]
            if video_id:
                return f"https://www.youtube.com/watch?v={video_id}"

    return url
