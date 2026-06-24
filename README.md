# Interview Transcriber v1

Local, session-based interview recording and transcription for Apple Silicon Macs.

## What v1 Does

- Records an explicitly started room-microphone session to durable 16 kHz WAV.
- Opens a visible Terminal monitor with a recording heartbeat and live transcript.
- Uses full MLX Whisper large-v3 with 15-second live chunks.
- Re-transcribes 30-second windows after stop for a higher-context final transcript.
- Preserves live and final TXT/JSONL artifacts plus per-chunk latency metrics.
- Generates an evidence-constrained interview analysis prompt and can invoke a configured Claude CLI.

Recording never starts automatically. Obtain consent from everyone being recorded.

## Install

```bash
python3 --version  # must be 3.10+
python3 -m pip install -e '.[dev]'
```

The first run downloads the MLX Whisper large-v3 model. v1 is intended for Apple Silicon macOS.
The macOS Command Line Tools Python may be 3.9 and is not supported; use Homebrew, Conda, or another
Python 3.10+ environment.

## Use

Chinese-dominant speech with embedded English terminology:

```bash
interview-transcriber start --name mixed-test --language zh
```

English-dominant interview:

```bash
interview-transcriber start --name google-interview --language en
```

Control and analyze:

```bash
interview-transcriber status
interview-transcriber stop
interview-transcriber refine ~/Documents/interview-transcripts/SESSION
interview-transcriber analyze ~/Documents/interview-transcripts/SESSION
```

`analyze` defaults to generating a local `analysis-prompt.txt` without invoking a model. Running
`analyze SESSION --engine claude` explicitly sends the prompt and transcript through the configured
Claude CLI; use it only when that data-handling path is acceptable.

Sessions are stored under `~/Documents/interview-transcripts/`.

## Artifacts

- `audio.wav`: complete source recording
- `transcript-live.txt` / `.jsonl`: live pass
- `transcript.txt` / `.jsonl`: final pass
- `transcription-metrics.jsonl`: inference latency and filtering outcomes
- `metadata.json`: models, duration, language mode, and dropped-block count
- `analysis-prompt.txt`: timestamp-grounded analysis input
- `interview-analysis.md`: optional generated analysis

## v1 Boundaries

- Select the primary language explicitly. Use `zh` for Chinese with embedded English terms and `en`
  for English interviews. Short-chunk automatic language detection is not reliable enough.
- The current backend captures a microphone, not macOS system audio. Use speakers rather than
  headphones so the remote interviewer is audible.
- Mac mini has no built-in microphone. Accuracy depends heavily on the connected input device,
  distance, room echo, and overlapping speakers.
- The live transcript prioritizes visibility; use the final transcript for analysis.

## Test

```bash
python3 -m unittest discover -s tests -v
ruff check interview_transcriber.py tests
```
