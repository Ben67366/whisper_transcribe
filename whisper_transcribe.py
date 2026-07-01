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
    python whisper_transcribe.py                 # batch: process un-transcribed files from folderPath in .env (defaults to ./audios)
"""

from __future__ import annotations

import argparse
import os
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


def check_cuda_status() -> str:
    """Return 'ok', 'broken', or 'no_gpu'.

    'ok'     — GPU visible and CUDA runtime is working.
    'broken' — GPU hardware detected but CUDA runtime DLLs are missing.
    'no_gpu' — No NVIDIA GPU found.
    """
    gpu_visible = False
    try:
        import ctranslate2  # type: ignore
        if ctranslate2.get_cuda_device_count() > 0:
            gpu_visible = True
    except Exception:
        pass

    if not gpu_visible:
        return "no_gpu"

    # GPU is visible — now verify the runtime DLLs are actually loadable.
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "ok"
    except Exception:
        pass

    return "broken"


def prompt_device_choice() -> Tuple[str, str]:
    """Interactive prompt shown when GPU is detected but CUDA runtime is broken.

    Prints the situation, explains the fix, and asks the user to choose.
    Returns (device, compute_type) ready for use.
    """
    print()
    print("=" * 60)
    print("  GPU detected, but CUDA runtime is not ready.")
    print("=" * 60)
    print()
    print("  The CUDA runtime library (cublas64_12.dll) could not be")
    print("  loaded. This usually means torch is not installed in the")
    print("  current Python environment.")
    print()
    print("  To fix (run once, then re-run this script):")
    print()
    print("    pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("  Or if you are using a virtual environment (.venv):")
    print()
    print("    .venv\\Scripts\\pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("-" * 60)
    print("  [G] Fix GPU later - exit now and follow the steps above")
    print("  [C] Continue with CPU - slower, but runs immediately")
    print("-" * 60)
    print()

    while True:
        try:
            choice = input("  Your choice (G/C): ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted.")
            sys.exit(130)

        if choice == "G":
            print()
            print("Exiting. Install torch with CUDA support and re-run.")
            sys.exit(0)
        elif choice == "C":
            print()
            print("Continuing with CPU (int8).")
            return "cpu", "int8"
        else:
            print("  Please enter G or C.")


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


def parse_output_types(raw: Optional[str]) -> set:
    """Parse TranscribeType value from .env into a normalised set of extensions.

    Accepts comma-separated values: srt, txt, vtt (also accepts 'src' as alias for 'srt').
    Returns {'srt', 'txt', 'vtt'} if raw is None or empty.
    """
    defaults = {"srt", "txt", "vtt"}
    if not raw:
        return defaults
    aliases = {"src": "srt"}
    result = set()
    for token in raw.split(","):
        t = token.strip().lower()
        t = aliases.get(t, t)
        if t in defaults:
            result.add(t)
    return result if result else defaults


def compute_output_destination(audio_path: Path, output_base: Optional[Path]) -> Tuple[Path, str]:
    """Return (output_dir, output_stem) for a given audio file.

    If output_base is None: outputs go next to the source file, stem unchanged.
    If output_base is set:  outputs go into output_base, stem prefixed with the
                            file's creation datetime as yyyymmddHHMMSS_.
    """
    if output_base is None:
        return audio_path.parent, audio_path.stem

    import datetime
    ctime = audio_path.stat().st_ctime
    dt = datetime.datetime.fromtimestamp(ctime)
    prefix = dt.strftime("%Y%m%d%H%M%S")
    return output_base, f"{prefix}_{audio_path.stem}"


def is_already_done(audio_path: Path, output_dir: Path, output_stem: str, output_types: set) -> bool:
    """Return True if outputs already exist in EITHER location:
    1. Next to the audio file (original stem, no datetime prefix)
    2. In the output folder (datetime-prefixed stem)
    """
    in_source = all((audio_path.parent / f"{audio_path.stem}.{ext}").exists() for ext in output_types)
    if in_source:
        return True
    in_output = all((output_dir / f"{output_stem}.{ext}").exists() for ext in output_types)
    return in_output


def write_report(
    report_path: Path,
    audios_dir: Path,
    device: str,
    compute_type: str,
    output_types: set,
    results: List[Dict[str, Any]],
) -> None:
    """Append a run summary to report.log."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    succeeded = [r for r in results if r["status"] == "success"]
    skipped   = [r for r in results if r["status"] == "skipped"]
    failed    = [r for r in results if r["status"] == "error"]

    lines = [
        "",
        f"=== Transcription Report  {timestamp} ===",
        f"Folder : {audios_dir}",
        f"Device : {device} ({compute_type})",
        f"Types  : {', '.join(sorted(output_types))}",
        "",
    ]
    for r in results:
        source_line = f"  Source : {r['source']}"
        if r["status"] == "success":
            lines.append(f"[SUCCESS] {r['rel']}")
            lines.append(source_line)
            lines.append(f"  Output : {r['outputs']}")
        elif r["status"] == "skipped":
            lines.append(f"[SKIPPED] {r['rel']}  (all outputs exist)")
            lines.append(source_line)
        else:
            lines.append(f"[ERROR]   {r['rel']}")
            lines.append(source_line)
            lines.append(f"  Detail : {r['detail']}")
    lines += [
        "",
        f"Succeeded: {len(succeeded)}  Skipped: {len(skipped)}  Failed: {len(failed)}",
        "=" * 50,
    ]

    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def load_model(device: str, compute_type: str, num_workers: int = 1) -> Tuple[Any, str, str]:
    """Load WhisperModel with CUDA fallback. Returns (model, device, compute_type)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Error: faster-whisper is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    print(f"Loading model 'large-v3-turbo' (device={device}, compute_type={compute_type}, workers={num_workers})...")
    try:
        model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type, num_workers=num_workers)
    except Exception as exc:
        exc_str = str(exc).lower()
        if device == "cuda" and any(kw in exc_str for kw in ("dll", "cublas", "cuda", "library")):
            print(f"[WARN] CUDA runtime library not found ({exc}).")
            print("[WARN] Falling back to CPU. To fix: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            device, compute_type = "cpu", "int8"
            print(f"Loading model 'large-v3-turbo' (device={device}, compute_type={compute_type}, workers={num_workers})...")
            try:
                model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type, num_workers=num_workers)
            except Exception as exc2:
                print(f"Error: Failed to load model on CPU fallback. Details: {exc2}")
                sys.exit(1)
        else:
            print(f"Error: Failed to load model 'large-v3-turbo'. Details: {exc}")
            print("Possible causes: network issue during first-time download, or insufficient disk space.")
            sys.exit(1)

    print("Model loaded successfully.")
    return model, device, compute_type


def _collect_segments_chunked(
    input_path: Path,
    model: Any,
    whisper_language: Optional[str],
    show_progress: bool,
    chunk_minutes: int = 10,
) -> Tuple[List[Dict[str, Any]], str, float]:
    """Fallback for files too large to transcribe at once.

    Splits audio into fixed-length WAV chunks via ffmpeg, transcribes each,
    and merges the resulting segments with corrected time offsets.
    Returns (raw_segments, detected_language, total_duration_seconds).
    """
    import subprocess
    import tempfile

    chunk_sec = chunk_minutes * 60

    # Determine total duration with ffprobe
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(input_path),
            ],
            capture_output=True, text=True, check=True,
        )
        total_duration = float(probe.stdout.strip())
    except Exception as e:
        raise RuntimeError(f"ffprobe failed to read audio duration: {e}")

    n_chunks = int(total_duration / chunk_sec) + 1
    print(f"[{input_path.name}] Splitting into {n_chunks} x {chunk_minutes}-min chunk(s) for processing...")

    raw_segments: List[Dict[str, Any]] = []
    detected_lang: str = whisper_language or "ja"

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx in range(n_chunks):
            offset = idx * chunk_sec
            if offset >= total_duration:
                break

            chunk_end = min(offset + chunk_sec, total_duration)
            chunk_path = Path(tmpdir) / f"chunk_{idx:04d}.wav"

            # Extract chunk: 16 kHz mono WAV (same format Whisper expects internally)
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(input_path),
                        "-ss", str(offset), "-t", str(chunk_sec),
                        "-ar", "16000", "-ac", "1",
                        str(chunk_path),
                    ],
                    capture_output=True, check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"  [chunk {idx+1}/{n_chunks}] ffmpeg error: {e.stderr.decode(errors='replace')[-200:]}")
                continue

            if not chunk_path.exists() or chunk_path.stat().st_size < 1000:
                continue

            print(f"  [chunk {idx+1}/{n_chunks}] {offset/60:.1f} - {chunk_end/60:.1f} min ...")

            seg_iter, chunk_info = model.transcribe(
                str(chunk_path),
                language=whisper_language,
                word_timestamps=True,
                vad_filter=True,
                beam_size=5,
                best_of=5,
                temperature=0.0,
            )
            if idx == 0:
                detected_lang = chunk_info.language

            for seg in seg_iter:
                raw_segments.append({
                    "start": float(seg.start) + offset,
                    "end": float(seg.end) + offset,
                    "text": seg.text or "",
                })

    return raw_segments, detected_lang, total_duration


def transcribe(
    input_path: Path,
    language: str,
    device: str,
    compute_type: str,
    output_types: Optional[set] = None,
    output_dir: Optional[Path] = None,
    output_stem: Optional[str] = None,
    model: Any = None,
    show_progress: bool = True,
) -> List[str]:
    """Core transcription + output generation logic. Returns list of generated file paths."""
    if output_types is None:
        output_types = {"srt", "txt", "vtt"}
    if output_dir is None:
        output_dir = input_path.parent
    if output_stem is None:
        output_stem = input_path.stem

    # Load model here only in single-file mode (batch pre-loads and shares it)
    if model is None:
        try:
            from tqdm import tqdm  # noqa: F401 — verify tqdm is available early
        except ImportError:
            print("Error: tqdm is not installed. Run: pip install -r requirements.txt")
            sys.exit(1)
        model, device, compute_type = load_model(device, compute_type)

    try:
        from tqdm import tqdm
    except ImportError:
        print("Error: tqdm is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    # Resolve language argument
    whisper_language: Optional[str] = None if language.lower() == "auto" else language

    print(f"[{input_path.name}] Starting transcription...")
    if whisper_language:
        print(f"[{input_path.name}] Language forced: {whisper_language}")

    raw_segments: List[Dict[str, Any]] = []
    detected_language: str = whisper_language or "ja"
    total_duration: float = 0.0

    try:
        # MemoryError may be raised here (eager mel spectrogram computation) or
        # during iteration below (lazy computation).  Both are caught by the outer
        # except MemoryError block so we can fall back to chunked processing.
        try:
            segments_iter, info = model.transcribe(
                str(input_path),
                language=whisper_language,
                word_timestamps=True,
                vad_filter=True,
                beam_size=5,
                best_of=5,
                temperature=0.0,
            )
        except FileNotFoundError:
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
        except MemoryError:
            raise  # handled by outer block
        except Exception as exc:
            print(f"Error: Transcription failed for {input_path.name}. Details: {exc}")
            if "ffmpeg" in str(exc).lower() or "format" in str(exc).lower():
                print("Hint: For .mp4/.mkv/.m4a files, ensure ffmpeg is installed and in PATH.")
            sys.exit(1)

        detected_language = info.language
        total_duration = info.duration
        print(f"[{input_path.name}] Language: {info.language} ({info.language_probability:.1%}), duration: {info.duration:.1f}s")

        # Collect segments — tqdm bar in single-worker mode, plain iteration in parallel mode
        # (parallel tqdm with position= causes terminal control code deadlocks).
        if show_progress:
            pbar = tqdm(
                total=info.duration,
                unit="s",
                desc=input_path.name[:30],
                bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}, {rate_fmt}]",
                leave=True,
            )
            last_reported = 0.0
            try:
                for segment in segments_iter:
                    raw_segments.append({"start": float(segment.start), "end": float(segment.end), "text": segment.text or ""})
                    delta = max(0.0, segment.end - last_reported)
                    if delta > 0:
                        pbar.update(delta)
                    last_reported = max(last_reported, segment.end)
                    pbar.set_postfix_str((segment.text or "").strip().replace("\n", " ")[:28], refresh=False)
            finally:
                pbar.close()
        else:
            for segment in segments_iter:
                raw_segments.append({"start": float(segment.start), "end": float(segment.end), "text": segment.text or ""})

    except MemoryError as exc:
        print(f"[{input_path.name}] Out of memory: {exc}")
        print(f"[{input_path.name}] Retrying with chunked processing (10-min ffmpeg splits)...")
        try:
            raw_segments, detected_language, total_duration = _collect_segments_chunked(
                input_path, model, whisper_language, show_progress
            )
        except Exception as chunk_exc:
            print(f"Error: Chunked transcription also failed for {input_path.name}. Details: {chunk_exc}")
            sys.exit(1)

    print(f"[{input_path.name}] {len(raw_segments)} raw segments collected.")

    # Post-processing
    effective_lang = whisper_language or detected_language
    processed_segments = post_process_segments(raw_segments, effective_lang)
    print(f"[{input_path.name}] {len(processed_segments)} segments after cleaning.")

    if not processed_segments:
        print(f"[{input_path.name}] Warning: No speech segments detected after VAD and post-processing.")
        processed_segments = []

    # Ensure output directory exists (relevant when outputPath is set)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Error: Could not create output directory '{output_dir}': {exc}")
        sys.exit(1)

    generators = {
        "txt": (output_dir / f"{output_stem}.txt", generate_txt),
        "srt": (output_dir / f"{output_stem}.srt", generate_srt),
        "vtt": (output_dir / f"{output_stem}.vtt", generate_vtt),
    }

    print("Generating output files...")
    generated: List[str] = []

    try:
        for ext in ("txt", "srt", "vtt"):
            if ext not in output_types:
                continue
            out_path, generator = generators[ext]
            generator(processed_segments, out_path)
            print(f"  ✓ {out_path}")
            generated.append(str(out_path))
    except Exception as exc:
        print(f"Error: Failed to write one or more output files.")
        print(f"Details: {exc}")
        sys.exit(1)

    print("\nTranscription complete.")
    return generated


def resolve_device() -> Tuple[str, str]:
    """Detect GPU status, prompt if needed, and return (device, compute_type)."""
    status = check_cuda_status()

    if status == "ok":
        print("CUDA detected. Using GPU acceleration (float16).")
        return "cuda", "float16"
    elif status == "no_gpu":
        print("No NVIDIA GPU detected. Using CPU (int8).")
        return "cpu", "int8"
    else:  # broken
        return prompt_device_choice()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video using Faster-Whisper large-v3-turbo and generate subtitles. Run with no arguments to batch-process the 'audios' folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python whisper_transcribe.py audio.mp3
  python whisper_transcribe.py meeting.m4a --language ja
  python whisper_transcribe.py podcast.mp4 --language auto
  python whisper_transcribe.py lecture_en.wav --language en
  python whisper_transcribe.py                 # auto: process files from folderPath in .env (skips if .srt exists)
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        type=str,
        help="Path to a single audio/video file. If omitted, batch process all supported files in the 'audios' folder.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ja",
        help="Language code (e.g. 'ja', 'en', 'zh') or 'auto' for detection. Default: ja",
    )

    args = parser.parse_args()

    # Run transcription
    try:
        if args.input is None:
            # Batch / auto mode (triggered by run_auto.bat or running with no args):
            # - Load folderPath from .env (create .env + default if missing or variable undefined)
            # - Only transcribe files that do not already have a matching <name>.srt
            try:
                from dotenv import load_dotenv
            except ImportError:
                load_dotenv = None  # fallback to manual parse

            env_file = Path.cwd() / ".env"
            folder_str = None
            transcribe_type_raw = None
            output_path_str = None
            max_workers_raw = None

            if load_dotenv is not None:
                load_dotenv(dotenv_path=str(env_file))
                folder_str = os.getenv("folderPath") or os.getenv("FOLDERPATH")
                transcribe_type_raw = os.getenv("TranscribeType") or os.getenv("TRANSCRIBETYPE")
                output_path_str = os.getenv("outputPath") or os.getenv("OUTPUTPATH")
                max_workers_raw = os.getenv("maxWorkers") or os.getenv("MAXWORKERS")
            else:
                # Manual .env parse (simple key=value) if python-dotenv not available
                if env_file.exists():
                    try:
                        for line in env_file.read_text(encoding="utf-8").splitlines():
                            stripped = line.strip()
                            if not stripped or stripped.startswith("#"):
                                continue
                            if "=" not in stripped:
                                continue
                            key, val = stripped.split("=", 1)
                            key = key.strip().lower()
                            val = val.strip()
                            # Strip matched outer quotes only (e.g. "path with spaces")
                            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                                val = val[1:-1]
                            if key == "folderpath":
                                folder_str = val
                            elif key == "transcribetype":
                                transcribe_type_raw = val
                            elif key == "outputpath":
                                output_path_str = val
                            elif key == "maxworkers":
                                max_workers_raw = val
                    except Exception:
                        pass

            output_types = parse_output_types(transcribe_type_raw)

            # Resolve optional output path
            output_base: Optional[Path] = None
            if output_path_str:
                op = Path(output_path_str).expanduser()
                # For absolute paths (incl. UNC: \\server\share) keep as-is to avoid
                # resolve() changing the path format (e.g. \\?\UNC\...) which would
                # break relative_to() comparisons later.
                output_base = op if op.is_absolute() else (Path.cwd() / op).resolve()

            if not folder_str:
                # .env not found OR folderPath variable not defined -> create/append default
                default_folder = "./audios"
                line_to_add = "folderPath=./audios\n"
                if not env_file.exists():
                    env_file.write_text(line_to_add, encoding="utf-8")
                    print(f"[INFO] Created {env_file.name} with default: folderPath={default_folder}")
                else:
                    try:
                        content = env_file.read_text(encoding="utf-8")
                        with env_file.open("a", encoding="utf-8") as f:
                            if content and not content.endswith("\n"):
                                f.write("\n")
                            f.write(line_to_add)
                        print(f"[INFO] Added folderPath={default_folder} to existing {env_file.name}")
                    except Exception:
                        print(f"[WARN] Could not update {env_file.name}; using default in memory only.")
                folder_str = default_folder

            # Resolve folder path (respect relative paths from CWD)
            p = Path(folder_str).expanduser()
            # For absolute paths (incl. UNC: \\server\share) keep as-is to avoid
            # resolve() changing the path format (e.g. \\?\UNC\...) which would
            # break rglob() / relative_to() comparisons for paths with spaces.
            audios_dir = p if p.is_absolute() else (Path.cwd() / p).resolve()

            # Create the folder if it does not exist (first-run convenience)
            if not audios_dir.exists():
                try:
                    audios_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[INFO] Created audio folder: {audios_dir}")
                except Exception as e:
                    print(f"Error: Could not create folder '{audios_dir}': {e}")
                    sys.exit(1)

            if not audios_dir.is_dir():
                print(f"Error: '{audios_dir}' is not a directory.")
                sys.exit(1)

            # Recursively collect all supported files
            all_candidates = sorted(
                p for p in audios_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )

            if not all_candidates:
                print(f"\nNo supported audio/video files found under '{audios_dir}'.")
                print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
                sys.exit(0)

            # Determine output destination for each file and check what needs processing
            destinations = {
                p: compute_output_destination(p, output_base)
                for p in all_candidates
            }
            to_process = [
                p for p in all_candidates
                if not is_already_done(p, destinations[p][0], destinations[p][1], output_types)
            ]
            pre_skipped = len(all_candidates) - len(to_process)

            if not to_process:
                print(f"\nAll {len(all_candidates)} file(s) already have all requested outputs. Nothing to do.")
                sys.exit(0)

            device, compute_type = resolve_device()

            num_workers = max(1, int(max_workers_raw)) if max_workers_raw else 1
            parallel = num_workers > 1

            print(f"\nFound {len(to_process)} file(s) to transcribe (skipped {pre_skipped} already done).")
            print(f"Output types : {', '.join(sorted(output_types))}")
            if output_base:
                print(f"Output folder: {output_base}")
            print(f"Workers      : {num_workers}")
            print(f"Starting batch transcription...\n")

            shared_model, device, compute_type = load_model(device, compute_type, num_workers=num_workers)

            report_path = Path.cwd() / "report.log"
            run_results: List[Dict[str, Any]] = []

            # Record pre-skipped files
            for p in all_candidates:
                od, os_ = destinations[p]
                if is_already_done(p, od, os_, output_types):
                    run_results.append({
                        "status": "skipped",
                        "rel": str(p.relative_to(audios_dir)),
                        "source": str(p),
                        "outputs": "",
                        "detail": "",
                    })

            import threading
            from concurrent.futures import ThreadPoolExecutor, as_completed

            results_lock = threading.Lock()
            success = 0
            failed = 0

            def _process(audio_path: Path) -> Dict[str, Any]:
                out_dir, out_stem = destinations[audio_path]
                # Re-check immediately before transcribing — another worker or external
                # process may have created the output since the initial scan.
                if is_already_done(audio_path, out_dir, out_stem, output_types):
                    print(f"[SKIP] {audio_path.name} — already done (detected at worker start)")
                    return {
                        "status": "skipped",
                        "rel": str(audio_path.relative_to(audios_dir)),
                        "source": str(audio_path),
                        "outputs": "",
                        "detail": "",
                    }
                generated = transcribe(
                    input_path=audio_path,
                    language=args.language,
                    device=device,
                    compute_type=compute_type,
                    output_types=output_types,
                    output_dir=out_dir,
                    output_stem=out_stem,
                    model=shared_model,
                    show_progress=not parallel,
                )
                return {
                    "status": "success",
                    "rel": str(audio_path.relative_to(audios_dir)),
                    "source": str(audio_path),
                    "outputs": ", ".join(generated),
                    "detail": "",
                }

            total = len(to_process)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_path = {
                    executor.submit(_process, audio_path): audio_path
                    for audio_path in to_process
                }
                done_count = 0
                for future in as_completed(future_to_path):
                    audio_path = future_to_path[future]
                    done_count += 1
                    try:
                        result = future.result()
                        with results_lock:
                            success += 1
                            run_results.append(result)
                        print(f"[{done_count}/{total}] Done: {audio_path.relative_to(audios_dir)}")
                    except SystemExit as exc:
                        if exc.code == 130:
                            raise
                        with results_lock:
                            failed += 1
                            run_results.append({
                                "status": "error",
                                "rel": str(audio_path.relative_to(audios_dir)),
                                "source": str(audio_path),
                                "outputs": "",
                                "detail": "see console output above",
                            })
                        print(f"[{done_count}/{total}] Error: {audio_path.name}")
                    except Exception as exc:
                        with results_lock:
                            failed += 1
                            run_results.append({
                                "status": "error",
                                "rel": str(audio_path.relative_to(audios_dir)),
                                "source": str(audio_path),
                                "outputs": "",
                                "detail": str(exc),
                            })
                        print(f"[{done_count}/{total}] Error: {audio_path.name} — {exc}")

            write_report(report_path, audios_dir, device, compute_type, output_types, run_results)
            print("\n" + "=" * 60)
            print(f"Batch complete: {success} succeeded, {pre_skipped} skipped, {failed} failed.")
            print(f"Full report written to: {report_path}")
            if failed > 0:
                sys.exit(1)
        else:
            # Single file mode (original behavior)
            input_path = Path(args.input).expanduser().resolve()

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

            device, compute_type = resolve_device()
            transcribe(
                input_path=input_path,
                language=args.language,
                device=device,
                compute_type=compute_type,
            )  # output_types defaults to all three in single-file mode
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
