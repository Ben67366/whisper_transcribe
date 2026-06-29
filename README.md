# Whisper Transcription Tool (Faster-Whisper large-v3-turbo)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-quality command-line tool for Windows that transcribes audio and video files using Faster-Whisper `large-v3-turbo` and generates clean subtitle files.

## Features

- **Supported input formats**: `.mp3`, `.wav`, `.m4a`, `.flac`, `.mp4`, `.mkv`
- **Output files** (created in the same directory as the input):
  - `filename.txt` — Clean plain-text transcript (no timestamps)
  - `filename.srt` — Standard SRT subtitle file
  - `filename.vtt` — WebVTT subtitle file
- **Engine**: `faster-whisper` + `large-v3-turbo` model (auto-downloaded on first run, ~800MB–1.5GB cached)
- **Language**: Japanese (`ja`) by default. Supports `--language auto` for detection or any Whisper language code (`en`, `zh`, etc.)
- **GPU/CPU**: Automatically selects CUDA (float16) on NVIDIA GPUs. If CUDA runtime is missing it prompts you to fix it or continue on CPU.
- **Batch launcher**: `run_auto.bat` — double-click to process all new files in the `audios/` folder
- **VAD**: Silero VAD filtering enabled to strip silence
- **Japanese subtitle improvements**: filler word removal, short segment merging, 42-char line wrapping

## Project Structure

```
whisper_transcribe/
├── audios/                 # Drop audio/video files here for batch processing
├── .env                    # Config: folderPath=./audios (auto-created on first run)
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── run_auto.bat            # Double-click launcher
└── whisper_transcribe.py
```

## Quick Start (Double-click)

1. Put audio/video files in the `audios/` folder
2. Double-click `run_auto.bat`
3. Find `.txt`, `.srt`, `.vtt` files next to each audio file

Files that already have a matching `.srt` are automatically skipped.

## Installation

### 1. Install Python 3.11+

Download from [python.org](https://www.python.org/downloads/windows/).

During installation, check **"Add python.exe to PATH"**.

Verify:

```powershell
python --version
# Python 3.11.x or higher
```

### 2. Create a Virtual Environment

Open PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you get an execution policy error:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Dependencies

**GPU users (NVIDIA, CUDA 12.1):**

```powershell
pip install -r requirements.txt
```

`requirements.txt` already includes the CUDA-enabled torch wheel. This is the recommended path.

**CPU-only users:**

```powershell
pip install faster-whisper tqdm python-dotenv
```

### 4. (Optional) Install ffmpeg

Required for `.mp4`, `.m4a`, `.mkv` files:

```powershell
# Chocolatey
choco install ffmpeg
```

Or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` folder to PATH.

## GPU / CUDA Setup

The tool detects your GPU status automatically at startup:

| Situation | What happens |
|---|---|
| NVIDIA GPU + CUDA working | Runs immediately on GPU |
| No NVIDIA GPU | Runs on CPU, no prompt |
| GPU detected, CUDA runtime missing | Interactive prompt (see below) |

**Interactive prompt when CUDA runtime is missing:**

```
============================================================
  GPU detected, but CUDA runtime is not ready.
============================================================

  The CUDA runtime library (cublas64_12.dll) could not be
  loaded. This usually means torch is not installed in the
  current Python environment.

  To fix (run once, then re-run this script):

    pip install torch --index-url https://download.pytorch.org/whl/cu121

  Or if you are using a virtual environment (.venv):

    .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu121

------------------------------------------------------------
  [G] Fix GPU later - exit now and follow the steps above
  [C] Continue with CPU - slower, but runs immediately
------------------------------------------------------------

  Your choice (G/C):
```

Choose **G** to exit and follow the fix steps, or **C** to run on CPU right away.

## Running the Tool

### Batch mode (recommended — via launcher)

Drop files in `audios/`, double-click `run_auto.bat`. Already-transcribed files are skipped.

### Command line — batch

```powershell
python whisper_transcribe.py --language auto
```

### Command line — single file

```powershell
# Japanese (default)
python whisper_transcribe.py audio.mp3

# Auto language detection
python whisper_transcribe.py podcast.mp4 --language auto

# English
python whisper_transcribe.py lecture.wav --language en
```

### Example output

```
CUDA detected. Using GPU acceleration (float16).

Found 2 file(s) to transcribe in '...\audios' (skipped 1 already done).
Starting batch transcription (one by one)...

============================================================
[1/2] meeting.m4a
============================================================
Loading model 'large-v3-turbo' (device=cuda, compute_type=float16)...
Model loaded successfully.
Starting transcription of: meeting.m4a
Language: auto-detect
Detected language: ja (probability: 97.2%)
Duration: 843.5 seconds
Transcribing: 100%|████████████| 843.5/843.5s [...]
Raw transcription produced 134 segments.
After cleaning & merging: 108 segments.
Generating output files...
  ✓ meeting.txt
  ✓ meeting.srt
  ✓ meeting.vtt

Transcription complete.
```

## Troubleshooting

### "Library cublas64_12.dll is not found"
CUDA runtime DLLs are missing. Run:
```powershell
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### "Error: File not found"
Check the path. Wrap paths with spaces in quotes:
```powershell
python whisper_transcribe.py "C:\recordings\my meeting.mp4"
```

### ffmpeg errors on .mp4 / .m4a / .mkv
Install ffmpeg and add its `bin` folder to PATH (see Installation step 4).

### Model download fails
The model downloads from Hugging Face on first run (~800MB–1.5GB). Check internet connection and disk space at `%USERPROFILE%\.cache\huggingface`.

### Very slow on CPU
The `large-v3-turbo` model is large. For long files on CPU, transcription takes time. GPU is strongly recommended. Using `--language ja` (explicit) is slightly faster than `--language auto`.

### Output files are empty or subtitles look wrong
- VAD may have removed everything if audio is extremely noisy/silent
- Japanese post-processing aggressively cleans short segments — check the `.txt` first

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Credits

- Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- Model: `large-v3-turbo` (OpenAI Whisper family, distilled for speed)

Happy transcribing!
