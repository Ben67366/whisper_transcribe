#!/usr/bin/env python3
"""
whisper_transcribe.py

Command-line tool to transcribe audio/video files using Faster-Whisper large-v3-turbo.
Generates .txt (clean transcript), .srt, and .vtt subtitle files next to the input.

Features:
- Automatic GPU (CUDA) or CPU selection with appropriate compute type.
- Japanese language default with optional auto-detection.
- Silero VAD filtering enabled.
- Word-level timestamps for accurate segment timing.
- Japanese-specific post-processing: filler word removal and short segment merging.
- Subtitle line wrapping: max 42 chars/line, max 2 lines per cue.
- Clean .txt output (text only, no timestamps).
- Progress display using tqdm.
- Full error handling and clear messages.
- Type hints and production-quality structure.

Usage:
    python whisper_transcribe.py audio.mp3
    python whisper_transcribe.py video.mp4 --language auto
    python whisper_transcribe.py interview.wav --language en
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any


SUPPORTED_EXTENSIONS: set[str] = {
    ".mp3", ".wav", ".m4a", ".flac", ".mp4", ".mkv"
}

# Common Japanese filler words / hesitation markers to remove for cleaner subtitles.
# Only clear hesitation / disfluency markers (avoid common particles that would corrupt normal words like "です").
JAPANESE_FILLERS: List[str] = [
    "えー", "えっと", "えーっと", "あの", "あのー", "あのう", "あのですね",
    "まあ", "うーん", "あー", "んー", "んーん", "まー",
    "そうですね",
]


def get_device_and_compute_type() -> Tuple[str, str]:
    """Detect available hardware and return optimal (device, compute_type)."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            # NVIDIA GPU with CUDA
            return "cuda", "float16"
        else:
            return "cpu", "int8"
    except Exception:
        # Fallback if torch import or detection fails
        return "cpu", "int8"


def clean_text(text: str, lang: Optional[str] = None) -> str:
    """Clean segment text. Applies Japanese filler removal when appropriate."""
    if not text:
        return ""

    cleaned = text.strip()

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)

    is_japanese = False
    if lang == "ja":
        is_japanese = True
    elif lang is None:
        # Heuristic: presence of Hiragana, Katakana or common Kanji
        if any("\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff" for c in cleaned):
            is_japanese = True

    if is_japanese:
        # Use word-boundary style matching (including CJK ranges) so we do not
        # accidentally strip fillers that are prefixes of normal words (e.g. "で" inside "です").
        punct_class = r"[、。！？,.!?\s、。]*"
        # Negative lookbehind / lookahead for "word" characters (alnum + Japanese)
        boundary_before = r"(?<![\w\u3040-\u30ff\u4e00-\u9fff])"
        boundary_after = r"(?![\w\u3040-\u30ff\u4e00-\u9fff])"
        for filler in JAPANESE_FILLERS:
            pattern = boundary_before + re.escape(filler) + boundary_after + punct_class
            cleaned = re.sub(pattern, " ", cleaned)

        # Collapse multiple spaces again
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Deduplicate consecutive punctuation
        cleaned = re.sub(r"([、。！？])\1+", r"\1", cleaned)

        # Remove leading/trailing punctuation and spaces that may remain
        cleaned = re.sub(r"^[、。！？\s,.!?]+", "", cleaned)
        cleaned = re.sub(r"[、。！？\s,.!?]+$", "", cleaned)

    # Final strip
    return cleaned.strip()


def merge_short_segments(
    segments: List[Dict[str, Any]],
    lang: Optional[str] = None,
    min_duration: float = 0.45,
    min_chars: int = 7,
    max_gap: float = 0.6,
) -> List[Dict[str, Any]]:
    """Merge extremely short segments for better subtitle readability.

    Especially useful for Japanese speech which often contains many short
    hesitation segments.
    """
    if not segments:
        return []

    merged: List[Dict[str, Any]] = []
    current: Dict[str, Any] = segments[0].copy()

    for next_seg in segments[1:]:
        gap = next_seg["start"] - current["end"]
        duration = current["end"] - current["start"]
        text_len = len(current["text"].strip())

        should_merge = (
            (duration < min_duration or text_len < min_chars)
            and gap <= max_gap
            and next_seg["text"].strip()
        )

        if should_merge:
            # Concatenate text (no extra space for Japanese)
            sep = " " if (" " in current["text"] or " " in next_seg["text"]) else ""
            current["text"] = (current["text"].rstrip() + sep + next_seg["text"].lstrip()).strip()
            current["end"] = next_seg["end"]
            # Re-clean after merge
            current["text"] = clean_text(current["text"], lang)
        else:
            if current["text"].strip():
                merged.append(current)
            current = next_seg.copy()

    if current["text"].strip():
        merged.append(current)

    # Final filter for any empty segments after cleaning
    return [s for s in merged if s["text"].strip()]


def post_process_segments(
    segments: List[Dict[str, Any]], lang: Optional[str]
) -> List[Dict[str, Any]]:
    """Apply cleaning and merging to improve subtitle quality."""
    if not segments:
        return []

    # Clean every segment first
    for seg in segments:
        seg["text"] = clean_text(seg["text"], lang)

    # Merge short segments (primarily beneficial for Japanese)
    processed = merge_short_segments(segments, lang=lang)

    return processed


def split_text_for_subtitle(
    text: str, max_chars: int = 42, max_lines: int = 5
) -> List[str]:
    """Split text into at most max_lines, each <= max_chars.

    Prefers breaks at Japanese/English punctuation and spaces.
    Designed to work for both Japanese (no spaces) and other languages.
    """
    text = text.strip()
    if not text:
        return [""]

    if len(text) <= max_chars:
        return [text]

    lines: List[str] = []
    remaining = text

    for _ in range(max_lines):
        if not remaining:
            break
        if len(remaining) <= max_chars:
            lines.append(remaining)
            remaining = ""
            break

        # Look for a good split point near the max_chars boundary
        window_end = min(len(remaining), max_chars)
        window = remaining[:window_end]

        split_point = max_chars
        # Search backwards from end of window for punctuation or space
        for j in range(len(window) - 1, max(0, len(window) - 18), -1):
            ch = window[j]
            if ch in "、。．！？,.!? ":
                split_point = j + 1
                break

        # Avoid creating tiny first line
        if split_point < max_chars * 0.55:
            split_point = max_chars

        line = remaining[:split_point].strip()
        if not line:
            line = remaining[:max_chars].strip()
            split_point = max_chars

        lines.append(line)
        remaining = remaining[split_point:].strip()

    if remaining and lines:
        # Attach remainder to last line if possible, otherwise truncate to limit
        last = lines[-1]
        combined = (last + remaining).strip()
        if len(combined) <= max_chars:
            lines[-1] = combined
        else:
            # Force split remainder onto last line (respecting max 2 lines total)
            lines[-1] = (last + remaining)[:max_chars].strip()
    elif remaining:
        lines.append(remaining[:max_chars].strip())

    # Ensure we never exceed max_lines and all lines are non-empty
    final_lines = [line for line in lines if line][:max_lines]
    if not final_lines:
        final_lines = [text[:max_chars]]
    return final_lines


def seconds_to_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def seconds_to_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to VTT timestamp format: HH:MM:SS.mmm"""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def generate_txt(segments: List[Dict[str, Any]], output_path: Path) -> None:
    """Write clean plain-text transcript (no timestamps)."""
    content = "\n".join(seg["text"].strip() for seg in segments if seg["text"].strip())
    if content:
        content += "\n"
    output_path.write_text(content, encoding="utf-8")


def generate_srt(segments: List[Dict[str, Any]], output_path: Path) -> None:
    """Write valid SRT subtitle file with proper line wrapping."""
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for index, seg in enumerate(segments, start=1):
            start_ts = seconds_to_srt_timestamp(seg["start"])
            end_ts = seconds_to_srt_timestamp(seg["end"])
            lines = split_text_for_subtitle(seg["text"])

            f.write(f"{index}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            for line in lines:
                f.write(f"{line}\n")
            f.write("\n")


def generate_vtt(segments: List[Dict[str, Any]], output_path: Path) -> None:
    """Write valid VTT subtitle file."""
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("WEBVTT\n\n")
        for index, seg in enumerate(segments, start=1):
            start_ts = seconds_to_vtt_timestamp(seg["start"])
            end_ts = seconds_to_vtt_timestamp(seg["end"])
            lines = split_text_for_subtitle(seg["text"])

            f.write(f"{index}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            for line in lines:
                f.write(f"{line}\n")
            f.write("\n")


def transcribe(
    input_path: Path,
    language: str,
    device: str,
    compute_type: str,
) -> None:
    """Core transcription + output generation logic."""
    # Lazy imports so that --help and argument parsing work without runtime deps installed.
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Error: faster-whisper is not installed.")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)

    try:
        from tqdm import tqdm
    except ImportError:
        print("Error: tqdm is not installed.")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)

    print(f"Loading model 'large-v3-turbo' (device={device}, compute_type={compute_type})...")
    try:
        model = WhisperModel(
            "large-v3-turbo",
            device=device,
            compute_type=compute_type,
        )
    except Exception as exc:
        print(f"Error: Failed to load model 'large-v3-turbo'.")
        print(f"Details: {exc}")
        print("Possible causes: network issue during first-time download, or insufficient disk space.")
        sys.exit(1)

    print("Model loaded successfully.")

    # Resolve language argument
    whisper_language: Optional[str] = None if language.lower() == "auto" else language

    print(f"Starting transcription of: {input_path.name}")
    if whisper_language:
        print(f"Language forced: {whisper_language}")
    else:
        print("Language: auto-detect")

    try:
        segments_iter, info = model.transcribe(
            str(input_path),
            language=whisper_language,
            word_timestamps=True,
            vad_filter=True,
            # Additional sensible defaults for quality
            beam_size=5,
            best_of=5,
            temperature=0.0,
        )
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    except Exception as exc:
        print(f"Error: Transcription failed for {input_path.name}")
        print(f"Details: {exc}")
        if "ffmpeg" in str(exc).lower() or "format" in str(exc).lower():
            print("Hint: For .mp4/.mkv/.m4a files, ensure ffmpeg is installed and in PATH.")
            print("  Windows: choco install ffmpeg   or download from https://ffmpeg.org")
        sys.exit(1)

    print(f"Detected language: {info.language} (probability: {info.language_probability:.1%})")
    print(f"Duration: {info.duration:.1f} seconds")

    # Collect segments with live progress bar (by audio time covered)
    raw_segments: List[Dict[str, Any]] = []
    pbar = tqdm(
        total=info.duration,
        unit="s",
        desc="Transcribing",
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}, {rate_fmt}]",
        leave=True,
    )
    last_reported = 0.0

    try:
        for segment in segments_iter:
            raw_segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text or "",
                }
            )
            # Advance progress bar to current segment end time
            delta = max(0.0, segment.end - last_reported)
            if delta > 0:
                pbar.update(delta)
            last_reported = max(last_reported, segment.end)
            # Show a snippet of current text in postfix (truncated)
            short_text = (segment.text or "").strip().replace("\n", " ")[:28]
            pbar.set_postfix_str(short_text, refresh=False)
    finally:
        pbar.close()

    print(f"Raw transcription produced {len(raw_segments)} segments.")

    # Post-processing
    effective_lang = whisper_language or info.language
    processed_segments = post_process_segments(raw_segments, effective_lang)
    print(f"After cleaning & merging: {len(processed_segments)} segments.")

    if not processed_segments:
        print("Warning: No speech segments detected after VAD and post-processing.")
        # Still create empty outputs so user has the files
        processed_segments = []

    # Determine output paths (same directory as input)
    stem = input_path.stem
    output_dir = input_path.parent
    txt_path = output_dir / f"{stem}.txt"
    srt_path = output_dir / f"{stem}.srt"
    vtt_path = output_dir / f"{stem}.vtt"

    print("Generating output files...")

    try:
        generate_txt(processed_segments, txt_path)
        print(f"  ✓ {txt_path.name}")

        generate_srt(processed_segments, srt_path)
        print(f"  ✓ {srt_path.name}")

        generate_vtt(processed_segments, vtt_path)
        print(f"  ✓ {vtt_path.name}")
    except Exception as exc:
        print(f"Error: Failed to write one or more output files.")
        print(f"Details: {exc}")
        sys.exit(1)

    print("\nTranscription complete.")
    print(f"Output files saved next to: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video using Faster-Whisper large-v3-turbo and generate subtitles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python whisper_transcribe.py audio.mp3
  python whisper_transcribe.py meeting.m4a --language ja
  python whisper_transcribe.py podcast.mp4 --language auto
  python whisper_transcribe.py lecture_en.wav --language en
        """,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input audio or video file (mp3, wav, m4a, flac, mp4, mkv)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ja",
        help="Language code (e.g. 'ja', 'en', 'zh') or 'auto' for detection. Default: ja",
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()

    # Validation
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    if not input_path.is_file():
        print(f"Error: Not a file: {input_path}")
        sys.exit(1)

    ext = input_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"Error: Unsupported file extension '{ext}'.")
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    # Device selection
    device, compute_type = get_device_and_compute_type()
    if device == "cuda":
        print("CUDA detected. Using GPU acceleration (float16).")
    else:
        print("No CUDA. Using CPU (int8).")

    # Run transcription
    try:
        transcribe(
            input_path=input_path,
            language=args.language,
            device=device,
            compute_type=compute_type,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
