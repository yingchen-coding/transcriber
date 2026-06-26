# Transcriber

[![CI](https://github.com/yingchen-coding/transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/yingchen-coding/transcriber/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Fast local transcription for anything you need to remember.

Transcriber records an explicitly started room-microphone session, shows a live transcript, saves
the raw audio locally, and produces a higher-context final transcript after recording stops. It is
designed for meetings, lectures, voice notes, debugging sessions, calls on speaker, and personal
review workflows.

Raw audio stays on your machine by default. Recording never starts automatically. Get consent from
everyone being recorded.

## Star This If

- You want a local meeting/interview/lecture recorder that keeps raw audio and transcripts on disk.
- You need both low-latency live notes and a cleaner final transcript after the session.
- You want analysis prompts grounded in transcript evidence without sending data anywhere by default.

## What It Does

- Records a session to a durable 16 kHz WAV file.
- Opens a visible Terminal monitor with a recording heartbeat and live transcript.
- Uses MLX Whisper large-v3 on Apple Silicon Macs.
- Re-transcribes 30-second windows after stop for a cleaner final transcript.
- Preserves live and final TXT/JSONL artifacts plus per-chunk latency metrics.
- Generates an evidence-constrained analysis prompt without sending data anywhere by default.

## Install

```bash
python3 --version  # must be 3.10+
python3 -m pip install -e '.[dev]'
```

The first run downloads the MLX Whisper large-v3 model. Transcriber is intended for Apple Silicon
macOS. The macOS Command Line Tools Python may be 3.9 and is not supported; use Homebrew, Conda, or
another Python 3.10+ environment.

## Use

Chinese-dominant speech with embedded English terms:

```bash
transcriber start --name team-sync --language zh
```

English-dominant speech:

```bash
transcriber start --name lecture-notes --language en
```

Control and analyze:

```bash
transcriber status
transcriber stop
transcriber refine ~/Documents/transcripts/SESSION
transcriber analyze ~/Documents/transcripts/SESSION
```

`analyze` defaults to generating a local `analysis-prompt.txt` without invoking a model. Running
`analyze SESSION --engine claude` explicitly sends the prompt and transcript through the configured
Claude CLI; use it only when that data-handling path is acceptable.

Sessions are stored under `~/Documents/transcripts/`.

## Artifacts

- `audio.wav`: complete source recording
- `transcript-live.txt` / `.jsonl`: live pass
- `transcript.txt` / `.jsonl`: final pass
- `transcription-metrics.jsonl`: inference latency and filtering outcomes
- `metadata.json`: models, duration, language mode, and dropped-block count
- `analysis-prompt.txt`: timestamp-grounded analysis input
- `analysis.md`: optional generated analysis

## Boundaries

- Select the primary language explicitly. Use `zh` for Chinese with embedded English terms and `en`
  for English-dominant sessions. Short-chunk automatic language detection is not reliable enough.
- The current backend captures a microphone, not macOS system audio. Use speakers rather than
  headphones so remote participants are audible.
- Mac mini has no built-in microphone. Accuracy depends heavily on the connected input device,
  distance, room echo, and overlapping speakers.
- The live transcript prioritizes visibility; use the final transcript for review.

## Local Review

```bash
scripts/pr_review_check.sh
```

This runs Ruff, unit tests, compile checks, package install, CLI smoke, and a public-surface scan.

## Test

```bash
python3 -m unittest discover -s tests -v
python3 -m ruff check transcriber.py tests
```
