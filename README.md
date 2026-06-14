# Whisper Transcription Tool (Faster-Whisper large-v3-turbo)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-quality command-line tool for Windows that transcribes audio and video files using Faster-Whisper `large-v3-turbo` and generates clean subtitle files.

## Features

- **Supported input formats**: `.mp3`, `.wav`, `.m4a`, `.flac`, `.mp4`, `.mkv`
- **Output files** (created in the same directory as the input):
  - `filename.txt` — Clean plain-text transcript (no timestamps)
  - `filename.srt` — Standard SRT subtitle file
  - `filename.vtt` — WebVTT subtitle file
- **Engine**: `faster-whisper` + `large-v3-turbo` model (automatically downloaded on first run)
- **Language**: Japanese (`ja`) by default. Supports `--language auto` for detection or any Whisper language code (e.g. `en`, `zh`).
- **GPU/CPU**: Automatically selects CUDA (float16) when an NVIDIA GPU is available, otherwise falls back to CPU (int8).
- **VAD**: Silero VAD filtering enabled (`vad_filter=True`) to remove silence.
- **Timestamps**: Accurate segment timing using `word_timestamps=True`.
- **Japanese subtitle improvements**:
  - Repeated filler words removed (えー, あの, うーん, etc.)
  - Extremely short segments merged for readability
- **Subtitle formatting** (SRT/VTT):
  - Maximum 42 characters per line
  - Maximum 2 lines per subtitle cue
  - Intelligent splitting at punctuation and natural boundaries
- Clean progress display with `tqdm`.
- Robust error handling and clear messages.

## Project Structure

```
whisper_transcribe/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── whisper_transcribe.py
```

## Requirements

- Windows 11
- Python 3.11 or newer
- (Optional but recommended for video files) ffmpeg in PATH

## Installation

### 1. Install Python

Download and install Python 3.11+ from the official site:

https://www.python.org/downloads/windows/

**Important during installation**:
- Check the box **"Add python.exe to PATH"**
- Choose "Install Now" or customize as needed.

Verify installation:

```powershell
python --version
# Should print Python 3.11.x or higher
```

### 2. Create a Virtual Environment

Open PowerShell or Command Prompt in the `whisper_transcribe` folder (or wherever you placed the files).

```powershell
# In the project directory
python -m venv .venv
```

Activate the virtual environment:

```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1

# If you get an execution policy error, run this first (once):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Command Prompt (cmd)
.\.venv\Scripts\activate.bat
```

You should see `(.venv)` at the start of your prompt.

### 3. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

This installs `faster-whisper` and `tqdm`.

**First run** will automatically download the `large-v3-turbo` model (~800MB–1.5GB depending on cache format). It is cached for future runs.

## Running the Tool

Basic usage (Japanese by default):

```powershell
python whisper_transcribe.py audio.mp3
```

With explicit language or auto-detection:

```powershell
# Force Japanese
python whisper_transcribe.py meeting.m4a --language ja

# Auto language detection
python whisper_transcribe.py podcast.mp4 --language auto

# English
python whisper_transcribe.py lecture_en.wav --language en
```

The three output files will be created next to the input file:

- `audio.txt`
- `audio.srt`
- `audio.vtt`

### Example Console Output

```
No CUDA. Using CPU (int8).
Loading model 'large-v3-turbo' (device=cpu, compute_type=int8)...
Model loaded successfully.
Starting transcription of: audio.mp3
Language forced: ja
Detected language: ja (probability: 98.4%)
Duration: 1243.7 seconds
Transcribing: 100%|████████████████| 1243.7/1243.7s [...]
Raw transcription produced 187 segments.
After cleaning & merging: 142 segments.
Generating output files...
  ✓ audio.txt
  ✓ audio.srt
  ✓ audio.vtt

Transcription complete.
Output files saved next to: C:\path\to\your\audio.mp3
```

## GPU (CUDA) Setup Instructions

For significantly faster transcription on NVIDIA GPUs:

1. Make sure you have recent NVIDIA drivers installed.

2. Install a CUDA-enabled version of PyTorch **before or instead of** the CPU version pulled by faster-whisper.

   Recommended (CUDA 12.1):

   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   For CUDA 11.8:

   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. Then install the project dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Verify GPU is detected by running the tool. You should see:

   ```
   CUDA detected. Using GPU acceleration (float16).
   ```

If you already installed the package with CPU torch, reinstall torch with the CUDA index URL above (it will replace the CPU wheel).

**Note**: You must have a compatible NVIDIA GPU and CUDA toolkit/drivers. The program will safely fall back to CPU if CUDA is not available.

## CPU-Only Setup Instructions

No special action needed:

```powershell
pip install -r requirements.txt
```

The script will automatically choose:

```
device=cpu, compute_type=int8
```

This provides good quality on modern CPUs and works without any NVIDIA hardware.

## Troubleshooting

### "Error: File not found"
- Double-check the path. Use quotes around paths with spaces:
  ```powershell
  python whisper_transcribe.py "C:\path\to\your\audio\file.mp3"
  ```
- Use absolute or relative paths correctly from the current directory.

### "Unsupported file extension"
- Only the listed formats are supported. Convert other formats to one of the supported ones (e.g. using ffmpeg).

### Model download fails or is very slow
- Ensure you have a stable internet connection on first run.
- The model is downloaded from Hugging Face. You can also pre-download it manually if needed, but automatic download is the intended flow.
- Check disk space in your user cache (`%USERPROFILE%\.cache\huggingface`).

### ffmpeg-related errors on .mp4 / .m4a / .mkv
- faster-whisper uses ffmpeg under the hood for many container formats.
- Install ffmpeg on Windows:
  - Using Chocolatey: `choco install ffmpeg`
  - Or download the full build from https://ffmpeg.org/download.html, extract, and add the `bin` folder to your PATH.
- After installing, restart your terminal/PowerShell.

### CUDA not detected even though I have an NVIDIA GPU
- Install the CUDA-enabled torch wheel (see GPU section above).
- Verify with Python:
  ```powershell
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- Ensure you are using the activated `.venv`.
- Update NVIDIA drivers.

### Very slow on CPU / high memory usage
- The `large-v3-turbo` model is still large. Transcription of long files can take time on CPU.
- Consider using `--language ja` explicitly instead of auto (slightly faster).
- Close other applications. The int8 quantization already helps keep memory reasonable.

### Output files are empty or subtitles look broken
- The VAD filter may have removed everything if the audio is extremely noisy or silent.
- Japanese post-processing can remove very short segments. This is intentional for readability.
- Check the `.txt` file — if it has content, the subtitles may simply have been aggressively cleaned.

### "torch is not installed" or import errors after venv activation
- Make sure you activated the virtual environment before running pip or the script.
- Re-run `pip install -r requirements.txt` inside the activated environment.

### KeyboardInterrupt / stopping the script
- Press Ctrl+C. The script handles it gracefully.

## Advanced Notes

- The script always overwrites output files with the same base name.
- All output files are written with UTF-8 encoding (correct for Japanese and other languages).
- Timestamps in SRT/VTT come directly from the model segments (word-level timestamps contribute to segment boundaries).
- For best Japanese subtitle quality, the combination of VAD + filler removal + short-segment merging + 42-char wrapping is applied.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- Model: `large-v3-turbo` (OpenAI Whisper family, distilled for speed)

Happy transcribing!
