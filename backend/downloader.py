from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

ProgressCallback = Optional[Callable[[float, str], None]]


def download_video(url: str, output_dir: str = "data/videos/", progress_callback: ProgressCallback = None) -> str:
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

    def _emit(progress: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0.0, min(100.0, progress)), message)

    def _progress_hook(progress_dict) -> None:
        status = progress_dict.get("status")
        if status == "downloading":
            percent = _parse_percent(progress_dict.get("_percent_str", ""))
            if percent is not None:
                speed = progress_dict.get("_speed_str", "n/a")
                _emit(percent, f"Downloading video ({percent:.1f}%) @ {speed}")
        elif status == "finished":
            _emit(100.0, "Download complete, finalizing file...")

    def _ydl_opts() -> dict:
        return {
            # The pipeline only needs frames, so we prefer video-only streams.
            # This avoids ffmpeg merge requirements on platforms like Bilibili.
            "format": "bestvideo[ext=mp4]/bestvideo/best[ext=mp4]/best",
            "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "progress_hooks": [_progress_hook],
            # Helps with recent YouTube web-client/SABR 403 issues.
            "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
            "retries": 5,
            "fragment_retries": 5,
        }

    def _download_with_opts(ydl_opts: dict) -> str:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(normalized_url, download=True)
            return ydl.prepare_filename(info)

    try:
        downloaded_path = _download_with_opts(_ydl_opts())
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download video from URL: {normalized_url}. Error: {exc}"
        ) from exc

    try:
        # Most sources yield .mp4, but keep a fallback to the original extension.
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


def _parse_percent(percent_str: str) -> Optional[float]:
    cleaned = re.sub(r"\x1b\[[0-9;]*m", "", str(percent_str)).strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


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
