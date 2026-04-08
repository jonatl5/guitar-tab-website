from __future__ import annotations

import base64
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

ProgressCallback = Optional[Callable[[float, str], None]]


def download_video(
    url: str,
    output_dir: str = "data/videos/",
    progress_callback: ProgressCallback = None,
    cookies_text: Optional[str] = None,
) -> str:
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
    cookies_file = _resolve_cookie_file(cookies_text)

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
        extractor_args = {}
        if _is_youtube_url(normalized_url):
            # Prefer client profiles that are more resilient to YouTube rate limits.
            extractor_args["youtube"] = {"player_client": ["android", "mweb", "web"]}

        opts = {
            # The pipeline only needs frames, so we prefer video-only streams.
            # This avoids ffmpeg merge requirements on platforms like Bilibili.
            "format": "bestvideo[ext=mp4]/bestvideo/best[ext=mp4]/bestvideo/best",
            "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "progress_hooks": [_progress_hook],
            "retries": 5,
            "fragment_retries": 5,
            "extractor_args": extractor_args,
        }

        if cookies_file:
            opts["cookiefile"] = cookies_file

        return opts

    def _download_with_opts(ydl_opts: dict) -> str:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(normalized_url, download=True)
            return ydl.prepare_filename(info)

    try:
        downloaded_path = _download_with_opts(_ydl_opts())
    except Exception as exc:
        raise RuntimeError(_build_download_error_message(normalized_url, exc, cookies_file)) from exc

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


def _resolve_cookie_file(cookies_text: Optional[str] = None) -> Optional[str]:
    inline_cookies = (cookies_text or "").strip()
    if inline_cookies:
        temp_dir = Path(tempfile.gettempdir()) / "guitartab"
        temp_dir.mkdir(parents=True, exist_ok=True)
        cookie_file = temp_dir / "yt-dlp-inline-cookies.txt"
        cookie_file.write_text(inline_cookies, encoding="utf-8")
        return str(cookie_file)

    configured_path = os.getenv("YTDLP_COOKIE_FILE", "").strip()
    if configured_path:
        cookie_path = Path(configured_path)
        if cookie_path.exists():
            return str(cookie_path)

    encoded_cookies = os.getenv("YTDLP_COOKIES_B64", "").strip()
    if not encoded_cookies:
        return None

    try:
        decoded = base64.b64decode(encoded_cookies).decode("utf-8")
    except Exception as exc:
        raise RuntimeError(
            "Invalid YTDLP_COOKIES_B64 value. Expected base64-encoded Netscape cookies content."
        ) from exc

    temp_dir = Path(tempfile.gettempdir()) / "guitartab"
    temp_dir.mkdir(parents=True, exist_ok=True)
    cookie_file = temp_dir / "yt-dlp-cookies.txt"
    cookie_file.write_text(decoded, encoding="utf-8")
    return str(cookie_file)


def _build_download_error_message(url: str, exc: Exception, has_cookies: bool) -> str:
    message = str(exc)
    lowered = message.lower()

    if _is_youtube_url(url) and (
        "sign in to confirm you're not a bot" in lowered
        or "use --cookies-from-browser or --cookies" in lowered
        or "login required" in lowered
    ):
        if has_cookies:
            return (
                "YouTube blocked this server-side download even with configured cookies. "
                "Try a local file upload for this video, refresh the exported cookies, or use a different source URL. "
                f"Original yt-dlp error: {message}"
            )

        return (
            "YouTube blocked anonymous server-side download for this video. "
            "Try uploading the video file directly, or configure backend env YTDLP_COOKIES_B64 "
            "(base64-encoded Netscape cookies.txt) or YTDLP_COOKIE_FILE for yt-dlp. "
            f"Original yt-dlp error: {message}"
        )

    return f"Failed to download video from URL: {url}. Error: {message}"


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


def _is_youtube_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "youtube.com" in host or "youtu.be" in host
