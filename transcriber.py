#!/usr/bin/env python3
"""Fast, durable room recording and transcription for macOS.

Commands:
    transcriber start --name lecture-notes
    transcriber status
    transcriber stop
    transcriber analyze SESSION_DIR [--engine claude]

The recorder is intentionally session-based. It never starts at login and always
prints where audio is being stored. Raw audio is written before transcription so
model failures cannot destroy the source recording.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import wave
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import mlx_whisper
import numpy as np
import sounddevice as sd


def _version() -> str:
    try:
        return version("local-transcriber")
    except PackageNotFoundError:
        return Path(__file__).with_name("VERSION").read_text(encoding="utf-8").strip()


__version__ = _version()


SAMPLE_RATE = 16_000
LIVE_MODEL = "mlx-community/whisper-large-v3-mlx"
FINAL_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_ROOT = Path.home() / "Documents" / "transcripts"
ACTIVE_FILE = DEFAULT_ROOT / ".active-session.json"
DEFAULT_CHUNK_SECONDS = 15.0
DEFAULT_OVERLAP_SECONDS = 2.0
FINAL_CHUNK_SECONDS = 30.0
FINAL_OVERLAP_SECONDS = 2.0
BRACKETED_TRANSCRIPT_LINE = re.compile(r"^\[(?P<time>[^\]]+)\]\s*(?P<text>.*)$")
SRT_TIMESTAMP_LINE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})(?:\s+.*)?$"
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"Could not read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _is_expected_recorder(pid: int, session_dir: str) -> bool:
    if not _process_alive(pid):
        return False
    completed = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        text=True,
        capture_output=True,
        timeout=5,
    )
    if completed.returncode != 0:
        return False
    command = completed.stdout
    return "transcriber" in command and "_record" in command and session_dir in command


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return value[:60] or "session"


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _clean_transcript(text: str) -> str:
    patterns = (
        r"(?:Thank you\.?\s*){2,}",
        r"\b(?:you\s*){4,}",
        r"请不吝点赞[^。\n]*",
        r"中文字幕志愿者[^。\n]*",
        r"订阅\s*转发\s*打赏[^。\n]*",
    )
    cleaned = text.strip()
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    compact = re.sub(r"[\s\W_]+", "", cleaned)
    if len(compact) >= 20:
        _character, count = Counter(compact).most_common(1)[0]
        if count / len(compact) >= 0.45:
            return ""
    words = re.findall(r"\w+", cleaned.lower())
    if len(words) >= 4 and len(set(words)) / len(words) <= 0.25:
        return ""
    return cleaned


def _contains_speech(audio: np.ndarray) -> bool:
    """Conservative energy gate that adapts to the current room noise floor."""
    frame_samples = int(SAMPLE_RATE * 0.02)
    usable = len(audio) - (len(audio) % frame_samples)
    if usable < frame_samples:
        return False
    frames = audio[:usable].reshape(-1, frame_samples)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    noise_floor = float(np.percentile(frame_rms, 20))
    speech_threshold = max(0.0025, noise_floor * 2.2)
    voiced_frames = int(np.count_nonzero(frame_rms >= speech_threshold))
    return voiced_frames >= max(3, int(len(frame_rms) * 0.025))


def _result_text(result: dict[str, Any]) -> str:
    """Reject decode fallbacks that are characteristic of Whisper hallucinations."""
    text = result.get("text", "")
    segments = [
        segment for segment in result.get("segments", []) if segment.get("text", "").strip()
    ]
    if not segments:
        return text
    if max(float(segment.get("compression_ratio", 0.0)) for segment in segments) >= 3.0:
        return ""
    average_logprob = sum(float(segment.get("avg_logprob", -10.0)) for segment in segments) / len(
        segments
    )
    maximum_temperature = max(float(segment.get("temperature", 0.0)) for segment in segments)
    if average_logprob < -0.65 and maximum_temperature >= 0.6:
        return ""
    return text


def transcribe_audio(audio: np.ndarray, language: str | None = None) -> str:
    """Transcribe one in-memory audio buffer (float32, 16 kHz mono) to clean text.

    The one-shot counterpart to the streaming TranscriptionWorker: it applies the same speech
    gate, tuned decode settings, hallucination rejection, and cleanup, so callers (e.g. the F13
    voice hotkey) get the transcriber's quality without reimplementing it. Returns "" for silence
    or audio that decodes to a hallucination.
    """
    if not _contains_speech(audio):
        return ""
    kwargs: dict[str, Any] = {
        "path_or_hf_repo": FINAL_MODEL,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.8,
        "compression_ratio_threshold": 1.8,
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language
    result = mlx_whisper.transcribe(audio, **kwargs)
    return _clean_transcript(_result_text(result))


def _remove_overlap(previous: str, current: str, minimum: int = 6) -> str:
    """Remove a repeated character suffix caused by overlapped audio chunks."""
    left = re.sub(r"\s+", "", previous)
    right = re.sub(r"\s+", "", current)
    limit = min(160, len(left), len(right))
    overlap = 0
    for size in range(limit, minimum - 1, -1):
        if left[-size:] == right[:size]:
            overlap = size
            break
    if not overlap:
        return current.strip()

    consumed = 0
    index = 0
    while index < len(current) and consumed < overlap:
        if not current[index].isspace():
            consumed += 1
        index += 1
    return current[index:].lstrip(" ,.;:\uff0c\u3002\uff1b\uff1a")


@dataclass
class AudioChunk:
    index: int
    start_seconds: float
    end_seconds: float
    audio: np.ndarray


class TranscriptWriter:
    def __init__(self, session_dir: Path, pass_name: str = "live") -> None:
        self.text_path = session_dir / "transcript.txt"
        self.jsonl_path = session_dir / "transcript.jsonl"
        self.errors_path = session_dir / "transcription-errors.log"
        self.metrics_path = session_dir / "transcription-metrics.jsonl"
        self.pass_name = pass_name
        self._lock = threading.Lock()
        self._last_text = ""

    def write_segment(
        self, chunk: AudioChunk, text: str, transcription_seconds: float | None = None
    ) -> None:
        cleaned = _clean_transcript(text)
        with self._lock:
            cleaned = _remove_overlap(self._last_text, cleaned)
            if not cleaned:
                return
            record = {
                "chunk": chunk.index,
                "start_seconds": round(chunk.start_seconds, 3),
                "end_seconds": round(chunk.end_seconds, 3),
                "text": cleaned,
                "pass": self.pass_name,
            }
            if transcription_seconds is not None:
                record["transcription_seconds"] = round(transcription_seconds, 3)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            with self.text_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"[{_clock(chunk.start_seconds)}-{_clock(chunk.end_seconds)}] {cleaned}\n"
                )
            self._last_text = (self._last_text + " " + cleaned)[-600:]

    def write_metric(
        self,
        chunk: AudioChunk,
        transcription_seconds: float,
        outcome: str,
    ) -> None:
        record = {
            "pass": self.pass_name,
            "chunk": chunk.index,
            "audio_seconds": round(chunk.end_seconds - chunk.start_seconds, 3),
            "transcription_seconds": round(transcription_seconds, 3),
            "outcome": outcome,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def write_error(self, chunk: AudioChunk, error: Exception) -> None:
        with self.errors_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{datetime.now().isoformat()} chunk={chunk.index} "
                f"range={chunk.start_seconds:.2f}-{chunk.end_seconds:.2f} "
                f"error={type(error).__name__}: {error}\n"
            )


class TranscriptionWorker(threading.Thread):
    def __init__(
        self,
        jobs: queue.Queue[AudioChunk | None],
        writer: TranscriptWriter,
        language: str | None,
    ) -> None:
        super().__init__(name="whisper-transcriber", daemon=False)
        self.jobs = jobs
        self.writer = writer
        self.language = language
        self.ready = threading.Event()
        self.startup_error: Exception | None = None

    def run(self) -> None:
        try:
            mlx_whisper.transcribe(
                np.zeros(SAMPLE_RATE, dtype=np.float32),
                path_or_hf_repo=LIVE_MODEL,
                language=self.language,
                condition_on_previous_text=False,
                no_speech_threshold=0.8,
            )
        except Exception as error:
            self.startup_error = error
            self.ready.set()
            return
        self.ready.set()

        while True:
            chunk = self.jobs.get()
            try:
                if chunk is None:
                    return
                if not _contains_speech(chunk.audio):
                    self.writer.write_metric(chunk, 0.0, "silence_skipped")
                    continue
                kwargs: dict[str, Any] = {
                    "path_or_hf_repo": LIVE_MODEL,
                    "condition_on_previous_text": False,
                    "no_speech_threshold": 0.8,
                    "compression_ratio_threshold": 1.8,
                    "temperature": 0.0,
                }
                if self.language:
                    kwargs["language"] = self.language
                started = time.monotonic()
                result = mlx_whisper.transcribe(chunk.audio, **kwargs)
                elapsed = time.monotonic() - started
                text = _result_text(result)
                self.writer.write_segment(chunk, text, elapsed)
                outcome = "written" if _clean_transcript(text) else "hallucination_filtered"
                self.writer.write_metric(chunk, elapsed, outcome)
            except Exception as error:
                if chunk is not None:
                    self.writer.write_error(chunk, error)
            finally:
                self.jobs.task_done()


def _analysis_prompt(transcript: str, metadata: dict[str, Any]) -> str:
    return f"""You are a rigorous transcript analyst reviewing a recorded session.

Session metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Important limits:
- The recording is mono and has no reliable speaker diarization. Infer speakers only when clear.
- Do not invent missing questions, intent, decisions, feedback, or facts.
- Every important finding must cite one or more transcript timestamps.
- Distinguish observed problems from uncertain inferences.
- Focus on useful, evidence-grounded takeaways instead of cosmetic speaking style.

Produce the report in Simplified Chinese with these sections:
1. Executive summary: the three highest-impact takeaways.
2. Session map: topic, speaker flow, decisions, open questions, and outcomes.
3. Useful details worth preserving, with timestamp evidence.
4. Problems or weak moments ordered by practical impact.
5. For each major issue: quote or paraphrase the relevant moment, explain why it matters,
   and provide a stronger alternative if applicable.
6. Missed signals or moments where the conversation should have changed course.
7. A focused follow-up plan: at most five actions, each tied to evidence.
8. Uncertainties caused by audio quality or missing context.

Transcript:
{transcript}
"""


def _write_analysis_packet(session_dir: Path) -> Path:
    transcript_path = session_dir / "transcript.txt"
    metadata = _read_json(session_dir / "metadata.json") or {}
    if not transcript_path.exists() or not transcript_path.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"No transcript found in {session_dir}")
    prompt_path = session_dir / "analysis-prompt.txt"
    prompt_path.write_text(
        _analysis_prompt(transcript_path.read_text(encoding="utf-8"), metadata), encoding="utf-8"
    )
    return prompt_path


def _read_translation_segments(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl_translation_segments(path)
    if suffix in {".srt", ".vtt"}:
        return _read_subtitle_translation_segments(path)
    return _read_text_translation_segments(path)


def _read_jsonl_translation_segments(path: Path) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            message = f"Invalid transcript JSONL at {path}:{line_number}: {error}"
            raise RuntimeError(message) from error
        if not isinstance(record, dict):
            raise RuntimeError(f"Expected transcript object at {path}:{line_number}")
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        start = record.get("start_seconds", "")
        end = record.get("end_seconds", "")
        if isinstance(start, int | float) and isinstance(end, int | float):
            timestamp = f"{_clock(float(start))}-{_clock(float(end))}"
        else:
            timestamp = f"segment-{len(segments) + 1}"
        segments.append({"time": timestamp, "text": text})
    return segments


def _read_text_translation_segments(path: Path) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        match = BRACKETED_TRANSCRIPT_LINE.match(line)
        if match:
            timestamp = match.group("time").strip()
            text = match.group("text").strip()
        else:
            timestamp = f"line-{index}"
            text = line
        if text:
            segments.append({"time": timestamp, "text": text})
    return segments


def _read_subtitle_translation_segments(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    segments: list[dict[str, str]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or lines == ["WEBVTT"]:
            continue
        timestamp_index = next(
            (index for index, line in enumerate(lines) if SRT_TIMESTAMP_LINE.match(line)),
            -1,
        )
        if timestamp_index < 0:
            continue
        match = SRT_TIMESTAMP_LINE.match(lines[timestamp_index])
        if not match:
            continue
        payload = " ".join(lines[timestamp_index + 1 :]).strip()
        if payload:
            start = match.group("start").replace(",", ".")
            end = match.group("end").replace(",", ".")
            segments.append({"time": f"{start}-{end}", "text": payload})
    return segments


def _translation_glossary(segments: list[dict[str, str]], limit: int = 40) -> list[str]:
    text = "\n".join(segment["text"] for segment in segments)
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9+.#/-]{1,}\b|\b[A-Za-z]+(?:-[A-Za-z]+)+\b", text)
    ignored = {"I", "The", "And", "A"}
    terms: list[str] = []
    for candidate in candidates:
        if candidate in ignored or candidate in terms:
            continue
        terms.append(candidate)
        if len(terms) >= limit:
            break
    return terms


def _translation_prompt(source_language: str, target_language: str, terms: list[str]) -> str:
    term_text = ", ".join(terms) if terms else "(none detected)"
    return (
        f"Translate the transcript from {source_language} to {target_language}.\n"
        "Rules:\n"
        "- Preserve every timestamp exactly.\n"
        "- Preserve speaker labels if present.\n"
        "- Keep technical terms consistent; do not over-translate product names.\n"
        "- If the source mixes Chinese and English, translate meaning rather than word order.\n"
        "- Mark unclear audio as [unclear] instead of inventing content.\n"
        f"Detected terms to preserve or translate consistently: {term_text}\n"
    )


def _translation_packet(
    transcript: Path,
    *,
    source_language: str,
    target_language: str,
) -> dict[str, Any]:
    segments = _read_translation_segments(transcript)
    if not segments:
        raise RuntimeError(f"No transcript segments found in {transcript}")
    terms = _translation_glossary(segments)
    return {
        "version": 1,
        "source_name": transcript.name,
        "created_at": datetime.now().isoformat(),
        "source_language": source_language,
        "target_language": target_language,
        "segments": segments,
        "glossary": terms,
        "translation_prompt": _translation_prompt(source_language, target_language, terms),
    }


def _write_translation_packet(
    transcript: Path,
    output_root: Path | None,
    source_language: str,
    target_language: str,
) -> Path:
    packet = _translation_packet(
        transcript,
        source_language=source_language,
        target_language=target_language,
    )
    root = output_root or transcript.parent / "translation-packets"
    out = root / _slug(transcript.stem)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "packet.json", packet)
    (out / "prompt.txt").write_text(packet["translation_prompt"], encoding="utf-8")
    (out / "segments.txt").write_text(
        "\n".join(f"[{item['time']}] {item['text']}" for item in packet["segments"]) + "\n",
        encoding="utf-8",
    )
    (out / "bilingual-template.md").write_text(_render_bilingual_template(packet), encoding="utf-8")
    return out


def _render_bilingual_template(packet: dict[str, Any]) -> str:
    lines = [
        "# Translation Packet",
        "",
        f"Source: {packet['source_name']}",
        f"Target language: {packet['target_language']}",
        "",
        "## Glossary",
        "",
    ]
    terms = packet.get("glossary") or []
    if terms:
        lines.extend(f"- {term}: " for term in terms)
    else:
        lines.append("- (none detected)")
    lines.extend(["", "## Segments", ""])
    for segment in packet["segments"]:
        lines.append(f"### [{segment['time']}]")
        lines.append(f"Source: {segment['text']}")
        lines.append("Translation: ")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _extract_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix != ".pdf":
        raise RuntimeError("document-packet input must be PDF, TXT, or Markdown")
    executable = shutil.which("pdftotext")
    if not executable:
        raise RuntimeError("pdftotext is required for PDF document-packet extraction")
    completed = subprocess.run(
        [executable, "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed.stdout


def _document_segments(path: Path, text: str, max_chars: int = 1800) -> list[dict[str, str]]:
    if path.suffix.lower() == ".pdf":
        pages = [page.strip() for page in text.split("\f")]
        return [
            {"loc": f"page-{index}", "text": page}
            for index, page in enumerate(pages, start=1)
            if page
        ]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    segments: list[dict[str, str]] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            segments.append({"loc": f"section-{len(segments) + 1}", "text": paragraph})
            continue
        for start in range(0, len(paragraph), max_chars):
            chunk = paragraph[start : start + max_chars].strip()
            if chunk:
                segments.append({"loc": f"section-{len(segments) + 1}", "text": chunk})
    return segments


def _document_glossary(segments: list[dict[str, str]], limit: int = 50) -> list[str]:
    text = "\n".join(segment["text"] for segment in segments)
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9+.#/-]{1,}\b|\b[A-Za-z]+(?:-[A-Za-z]+)+\b", text)
    ignored = {"I", "The", "And", "This", "That"}
    terms: list[str] = []
    for candidate in candidates:
        if candidate in ignored or candidate in terms:
            continue
        terms.append(candidate)
        if len(terms) >= limit:
            break
    return terms


def _document_prompt(packet: dict[str, Any], mode: str) -> str:
    source = packet["source_name"]
    locations = ", ".join(segment["loc"] for segment in packet["segments"][:12])
    if len(packet["segments"]) > 12:
        locations += ", ..."
    common = (
        f"Source document: {source}\n"
        f"Available locations: {locations or '(none)'}\n\n"
        "Rules:\n"
        "- Cite page/section locations for every important point.\n"
        "- Do not invent facts that are not in the provided segments.\n"
        "- Separate direct evidence from inference.\n"
        "- Keep private/local paths out of the output.\n\n"
    )
    if mode == "actions":
        return common + (
            "Turn this document into implementation-ready action items. For each item include: "
            "title, why it matters, source location, acceptance criteria, and first local test.\n"
        )
    if mode == "claims":
        return common + (
            "Extract claims that should be verified before use. For each claim include: subject, "
            "claim type, exact evidence location, missing evidence, and recommendation "
            "reject/verify-first/track/validated.\n"
        )
    if mode == "tasks":
        return common + (
            "Turn this document into a local workboard queue. For each task include: title, "
            "source location, expected artifact, dependency, and done condition.\n"
        )
    raise RuntimeError(f"Unknown document prompt mode: {mode}")


def _document_packet(document: Path) -> dict[str, Any]:
    text = _extract_document_text(document)
    segments = _document_segments(document, text)
    if not segments:
        raise RuntimeError(f"No extractable text found in {document}")
    packet = {
        "version": 1,
        "source_name": document.name,
        "created_at": datetime.now().isoformat(),
        "characters": len(text),
        "segments": segments,
        "glossary": _document_glossary(segments),
    }
    packet["action_prompt"] = _document_prompt(packet, "actions")
    packet["claim_prompt"] = _document_prompt(packet, "claims")
    packet["task_prompt"] = _document_prompt(packet, "tasks")
    return packet


def _write_document_packet(document: Path, output_root: Path | None) -> Path:
    packet = _document_packet(document)
    root = output_root or document.parent / "document-packets"
    out = root / _slug(document.stem)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "packet.json", packet)
    (out / "extracted.txt").write_text(
        "\n\n".join(f"[{item['loc']}]\n{item['text']}" for item in packet["segments"]) + "\n",
        encoding="utf-8",
    )
    (out / "action-prompt.txt").write_text(packet["action_prompt"], encoding="utf-8")
    (out / "claim-prompt.txt").write_text(packet["claim_prompt"], encoding="utf-8")
    (out / "task-prompt.txt").write_text(packet["task_prompt"], encoding="utf-8")
    return out


def _refine_transcript(session_dir: Path, language: str | None) -> None:
    """Create a higher-context final transcript while preserving the live transcript."""
    text_path = session_dir / "transcript.txt"
    jsonl_path = session_dir / "transcript.jsonl"
    live_text_path = session_dir / "transcript-live.txt"
    live_jsonl_path = session_dir / "transcript-live.jsonl"
    if text_path.exists():
        if live_text_path.exists():
            text_path.unlink()
        else:
            text_path.replace(live_text_path)
    if jsonl_path.exists():
        if live_jsonl_path.exists():
            jsonl_path.unlink()
        else:
            jsonl_path.replace(live_jsonl_path)

    metrics_path = session_dir / "transcription-metrics.jsonl"
    if metrics_path.exists():
        retained_metrics = []
        for line_number, line in enumerate(
            metrics_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid metric JSON at {metrics_path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise RuntimeError(f"Expected metric object at {metrics_path}:{line_number}")
            if record.get("pass") != "final":
                retained_metrics.append(json.dumps(record))
        content = "\n".join(retained_metrics)
        metrics_path.write_text(content + ("\n" if content else ""), encoding="utf-8")

    writer = TranscriptWriter(session_dir, pass_name="final")
    chunk_samples = int(FINAL_CHUNK_SECONDS * SAMPLE_RATE)
    overlap_samples = int(FINAL_OVERLAP_SECONDS * SAMPLE_RATE)
    offset = 0
    index = 0
    try:
        with wave.open(str(session_dir / "audio.wav"), "rb") as wav_file:
            if (
                wav_file.getnchannels() != 1
                or wav_file.getframerate() != SAMPLE_RATE
                or wav_file.getsampwidth() != 2
            ):
                raise RuntimeError("Expected 16 kHz mono 16-bit WAV for final transcription")
            total_samples = wav_file.getnframes()
            while offset < total_samples:
                wav_file.setpos(offset)
                frame_count = min(chunk_samples, total_samples - offset)
                raw_audio = wav_file.readframes(frame_count)
                chunk_audio = np.frombuffer(raw_audio, dtype="<i2").astype(np.float32) / 32767.0
                if len(chunk_audio) < SAMPLE_RATE:
                    break
                end = offset + len(chunk_audio)
                chunk = AudioChunk(index, offset / SAMPLE_RATE, end / SAMPLE_RATE, chunk_audio)
                if not _contains_speech(chunk_audio):
                    writer.write_metric(chunk, 0.0, "silence_skipped")
                else:
                    kwargs: dict[str, Any] = {
                        "path_or_hf_repo": FINAL_MODEL,
                        "condition_on_previous_text": False,
                        "no_speech_threshold": 0.8,
                        "compression_ratio_threshold": 1.8,
                    }
                    if language:
                        kwargs["language"] = language
                    started = time.monotonic()
                    result = mlx_whisper.transcribe(chunk_audio, **kwargs)
                    elapsed = time.monotonic() - started
                    text = _result_text(result)
                    writer.write_segment(chunk, text, elapsed)
                    outcome = "written" if _clean_transcript(text) else "hallucination_filtered"
                    writer.write_metric(chunk, elapsed, outcome)
                if end == total_samples:
                    break
                offset = end - overlap_samples
                index += 1
    except Exception:
        text_path.unlink(missing_ok=True)
        jsonl_path.unlink(missing_ok=True)
        if live_text_path.exists():
            shutil.copy2(live_text_path, text_path)
        if live_jsonl_path.exists():
            shutil.copy2(live_jsonl_path, jsonl_path)
        raise


def _run_analysis(session_dir: Path, engine: str) -> Path:
    prompt_path = _write_analysis_packet(session_dir)
    output_path = session_dir / "analysis.md"
    if engine == "none":
        return prompt_path
    executable = shutil.which(engine)
    if not executable:
        raise RuntimeError(f"Analysis engine is not installed: {engine}")
    if engine == "claude":
        command = [executable, "-p", "--output-format", "text"]
    else:
        raise RuntimeError(f"Unsupported analysis engine: {engine}")
    completed = subprocess.run(
        command,
        input=prompt_path.read_text(encoding="utf-8"),
        text=True,
        capture_output=True,
        timeout=1800,
    )
    if completed.returncode != 0:
        (session_dir / "analysis-error.log").write_text(completed.stderr, encoding="utf-8")
        raise RuntimeError(f"{engine} analysis failed; prompt was preserved at {prompt_path}")
    output_path.write_text(completed.stdout.strip() + "\n", encoding="utf-8")
    return output_path


def _record(args: argparse.Namespace) -> int:
    session_dir = Path(args.session_dir).resolve()
    session_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now()
    stop_event = threading.Event()
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=400)
    audio_status_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
    transcript_queue: queue.Queue[AudioChunk | None] = queue.Queue()
    writer = TranscriptWriter(session_dir)
    worker = TranscriptionWorker(transcript_queue, writer, args.language)
    dropped_blocks = 0

    metadata: dict[str, Any] = {
        "name": args.name,
        "session_dir": str(session_dir),
        "started_at": started.isoformat(),
        "status": "starting",
        "pid": os.getpid(),
        "sample_rate": SAMPLE_RATE,
        "live_model": LIVE_MODEL,
        "final_model": FINAL_MODEL,
        "language": args.language or "auto",
        "chunk_seconds": args.chunk_seconds,
        "overlap_seconds": args.overlap_seconds,
        "input_device": args.device,
        "capture_mode": "room microphone",
        "warning": "Use speakers, not headphones; no system-audio loopback device is installed.",
    }
    _atomic_json(session_dir / "metadata.json", metadata)
    _atomic_json(ACTIVE_FILE, metadata)

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def callback(indata: np.ndarray, _frames: int, _timing: Any, status: Any) -> None:
        nonlocal dropped_blocks
        if stop_event.is_set():
            return
        if status:
            audio_status_queue.put(f"{datetime.now().isoformat()} {status}")
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            dropped_blocks += 1

    chunk_samples = int(args.chunk_seconds * SAMPLE_RATE)
    overlap_samples = int(args.overlap_seconds * SAMPLE_RATE)
    pending = np.empty(0, dtype=np.float32)
    total_samples = 0
    chunk_index = 0

    worker.start()
    if not worker.ready.wait(timeout=60):
        metadata["status"] = "failed"
        metadata["error"] = "Timed out while warming the live transcription model"
        _atomic_json(session_dir / "metadata.json", metadata)
        ACTIVE_FILE.unlink(missing_ok=True)
        raise RuntimeError("Timed out while warming the live transcription model")
    if worker.startup_error is not None:
        metadata["status"] = "failed"
        metadata["error"] = str(worker.startup_error)
        _atomic_json(session_dir / "metadata.json", metadata)
        ACTIVE_FILE.unlink(missing_ok=True)
        raise RuntimeError(f"Could not warm live model: {worker.startup_error}")
    audio_path = session_dir / "audio.wav"
    try:
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=args.device,
                callback=callback,
                blocksize=1600,
            ):
                metadata["status"] = "recording"
                _atomic_json(session_dir / "metadata.json", metadata)
                _atomic_json(ACTIVE_FILE, metadata)
                print(f"RECORDING {session_dir}", flush=True)
                print("Use room speakers, not headphones. Stop with the stop command.", flush=True)
                while not stop_event.is_set() or not audio_queue.empty():
                    try:
                        block = audio_queue.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    pcm = np.clip(block, -1.0, 1.0)
                    wav_file.writeframes((pcm * 32767).astype("<i2").tobytes())
                    total_samples += len(block)
                    pending = np.concatenate((pending, block))
                    audio_queue.task_done()
                    statuses = []
                    while not audio_status_queue.empty():
                        statuses.append(audio_status_queue.get_nowait())
                    if statuses:
                        with (session_dir / "audio-status.log").open(
                            "a", encoding="utf-8"
                        ) as handle:
                            handle.write("\n".join(statuses) + "\n")
                    while len(pending) >= chunk_samples:
                        start = max(0, total_samples - len(pending)) / SAMPLE_RATE
                        audio = pending[:chunk_samples].copy()
                        transcript_queue.put(
                            AudioChunk(chunk_index, start, start + args.chunk_seconds, audio)
                        )
                        chunk_index += 1
                        pending = pending[chunk_samples - overlap_samples :]
    finally:
        if len(pending) >= SAMPLE_RATE:
            start = max(0, total_samples - len(pending)) / SAMPLE_RATE
            transcript_queue.put(
                AudioChunk(chunk_index, start, total_samples / SAMPLE_RATE, pending.copy())
            )
        transcript_queue.put(None)
        transcript_queue.join()
        worker.join(timeout=5)
        if worker.is_alive():
            raise RuntimeError("Transcription worker did not stop after draining its queue")
        metadata["status"] = "finalizing"
        _atomic_json(session_dir / "metadata.json", metadata)
        _atomic_json(ACTIVE_FILE, metadata)
        refinement_error = None
        try:
            _refine_transcript(session_dir, args.language)
        except Exception as error:
            refinement_error = f"{type(error).__name__}: {error}"
            with (session_dir / "final-transcription-error.log").open(
                "w", encoding="utf-8"
            ) as handle:
                handle.write(refinement_error + "\n")
        metadata.update(
            {
                "status": "complete",
                "ended_at": datetime.now().isoformat(),
                "duration_seconds": round(total_samples / SAMPLE_RATE, 3),
                "dropped_audio_blocks": dropped_blocks,
                "chunks_submitted": chunk_index + (1 if len(pending) >= SAMPLE_RATE else 0),
                "live_chunk_seconds": args.chunk_seconds,
                "final_chunk_seconds": FINAL_CHUNK_SECONDS,
                "final_transcription": "complete" if refinement_error is None else "live_fallback",
            }
        )
        try:
            _write_analysis_packet(session_dir)
            metadata["analysis_packet"] = "complete"
        except RuntimeError as error:
            metadata["analysis_packet"] = "not_generated"
            metadata["analysis_packet_error"] = str(error)
        _atomic_json(session_dir / "metadata.json", metadata)
        active = _read_json(ACTIVE_FILE)
        if active and active.get("pid") == os.getpid():
            ACTIVE_FILE.unlink(missing_ok=True)
        print(f"COMPLETE {session_dir}", flush=True)
    return 0


def _start(args: argparse.Namespace) -> int:
    active = _read_json(ACTIVE_FILE)
    if active and _is_expected_recorder(
        int(active.get("pid", -1)), str(active.get("session_dir", ""))
    ):
        raise RuntimeError(f"A recording is already active: {active.get('session_dir')}")
    ACTIVE_FILE.unlink(missing_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = DEFAULT_ROOT / f"{stamp}-{_slug(args.name)}"
    log_path = session_dir.with_suffix(".startup.log")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_record",
        "--name",
        args.name,
        "--session-dir",
        str(session_dir),
        "--chunk-seconds",
        str(args.chunk_seconds),
        "--overlap-seconds",
        str(args.overlap_seconds),
    ]
    if args.language:
        command.extend(("--language", args.language))
    if args.device is not None:
        command.extend(("--device", str(args.device)))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.time() + 75
    while time.time() < deadline:
        active = _read_json(ACTIVE_FILE)
        if active and active.get("status") == "recording":
            _show_live_window(Path(active["session_dir"]))
            print(f"Recording started: {active['session_dir']}")
            print("IMPORTANT: use room speakers, not headphones.")
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.25)
    details = log_path.read_text(encoding="utf-8") if log_path.exists() else "no startup log"
    raise RuntimeError(f"Recorder did not start. Log:\n{details}")


def _stop() -> int:
    active = _read_json(ACTIVE_FILE)
    if not active:
        print("No active recording.")
        return 0
    pid = int(active.get("pid", -1))
    session_dir = str(active.get("session_dir", ""))
    if not _is_expected_recorder(pid, session_dir):
        ACTIVE_FILE.unlink(missing_ok=True)
        raise RuntimeError("Active state did not identify the recorder; stale state was removed.")
    os.kill(pid, signal.SIGINT)
    deadline = time.time() + 120
    while time.time() < deadline and _process_alive(pid):
        time.sleep(0.25)
    if _process_alive(pid):
        latest = _read_json(Path(session_dir) / "metadata.json") or active
        print(
            f"Recording stopped; {latest.get('status', 'finalizing')} continues in the background: "
            f"{active.get('session_dir')}"
        )
        return 0
    print(f"Recording stopped: {active.get('session_dir')}")
    return 0


def _status() -> int:
    active = _read_json(ACTIVE_FILE)
    if not active:
        print("No active recording.")
        return 0
    pid = int(active.get("pid", -1))
    session_dir = str(active.get("session_dir", ""))
    state = active.get("status", "running") if _is_expected_recorder(pid, session_dir) else "stale"
    print(f"Status: {state}")
    print(f"Session: {session_dir}")
    print(f"PID: {pid}")
    for line in _session_status_lines(Path(session_dir)):
        print(line)
    return 0


def _session_status_lines(session_dir: Path) -> list[str]:
    metadata = _read_json(session_dir / "metadata.json") or {}
    audio_path = session_dir / "audio.wav"
    transcript_path = session_dir / "transcript.txt"
    live_transcript_path = session_dir / "transcript-live.txt"
    metrics_path = session_dir / "transcription-metrics.jsonl"
    lines = [
        f"Metadata status: {metadata.get('status', 'unknown')}",
        f"Duration seconds: {metadata.get('duration_seconds', 'recording')}",
        f"Audio bytes: {audio_path.stat().st_size if audio_path.exists() else 0}",
        f"Final transcript lines: {_line_count(transcript_path)}",
        f"Live transcript lines: {_line_count(live_transcript_path)}",
        f"Metric records: {_line_count(metrics_path)}",
        f"Dropped audio blocks: {metadata.get('dropped_audio_blocks', 0)}",
    ]
    return lines


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return sum(1 for line in lines if line)


def _doctor() -> int:
    checks = _doctor_checks()
    for check in checks:
        mark = "ok" if check["ok"] else "FAIL"
        detail = f" - {check['detail']}" if check.get("detail") else ""
        print(f"{mark:<4} {check['name']}{detail}")
    return 0 if all(check["ok"] for check in checks) else 1


def _doctor_checks() -> list[dict[str, object]]:
    checks: list[dict[str, object]] = [
        {
            "name": "python>=3.10",
            "ok": sys.version_info >= (3, 10),
            "detail": ".".join(str(part) for part in sys.version_info[:3]),
        },
        {
            "name": "macos",
            "ok": sys.platform == "darwin",
            "detail": sys.platform,
        },
        {
            "name": "mlx-whisper",
            "ok": hasattr(mlx_whisper, "transcribe"),
            "detail": LIVE_MODEL,
        },
    ]
    try:
        devices = sd.query_devices()
    except Exception as error:
        checks.append(
            {
                "name": "microphone",
                "ok": False,
                "detail": f"could not query audio devices: {type(error).__name__}: {error}",
            }
        )
        return checks
    input_devices = [
        device
        for device in devices
        if isinstance(device, dict) and int(device.get("max_input_channels", 0) or 0) > 0
    ]
    checks.append(
        {
            "name": "microphone",
            "ok": bool(input_devices),
            "detail": f"{len(input_devices)} input device(s)",
        }
    )
    return checks


def _monitor(session_dir: Path) -> int:
    transcript_path = session_dir / "transcript.txt"
    metadata_path = session_dir / "metadata.json"
    position = 0
    started = time.monotonic()
    last_heartbeat = 0.0
    print("LIVE TRANSCRIPT")
    print(f"Session: {session_dir}")
    print("Recording is active. New text appears after each configured chunk.\n", flush=True)
    while True:
        if transcript_path.exists():
            if transcript_path.stat().st_size < position:
                position = 0
                print("\nFINAL TRANSCRIPT\n", flush=True)
            with transcript_path.open(encoding="utf-8") as handle:
                handle.seek(position)
                content = handle.read()
                position = handle.tell()
            if content:
                print("\r" + (" " * 78) + "\r", end="")
                print(content, end="", flush=True)
        metadata = _read_json(metadata_path) or {}
        if metadata.get("status") == "complete":
            print("\nRECORDING COMPLETE", flush=True)
            return 0
        now = time.monotonic()
        if now - last_heartbeat >= 2.0:
            elapsed = _clock(now - started)
            print(
                f"\r[RECORDING {elapsed}] waiting for next transcript chunk...", end="", flush=True
            )
            last_heartbeat = now
        time.sleep(0.5)


def _show_live_window(session_dir: Path) -> None:
    command = " ".join(
        shlex.quote(part)
        for part in (
            sys.executable,
            str(Path(__file__).resolve()),
            "_monitor",
            "--session-dir",
            str(session_dir),
        )
    )
    script = f'tell application "Terminal" to do script {json.dumps(command)}'
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                script,
                "-e",
                'tell application "Terminal" to activate',
                "-e",
                (
                    'display notification "Live transcript window opened" '
                    'with title "Transcriber recording active"'
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Warning: could not open live transcript window: {error}", file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start a background recording")
    start.add_argument("--name", default="session")
    start.add_argument(
        "--language",
        choices=("en", "zh"),
        default="zh",
        help="Primary spoken language. zh preserves embedded English terms; use en for English.",
    )
    start.add_argument("--device", type=int, default=None)
    start.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    start.add_argument("--overlap-seconds", type=float, default=DEFAULT_OVERLAP_SECONDS)

    record = subparsers.add_parser("_record", help=argparse.SUPPRESS)
    record.add_argument("--name", required=True)
    record.add_argument("--session-dir", required=True)
    record.add_argument("--language", choices=("en", "zh"), default=None)
    record.add_argument("--device", type=int, default=None)
    record.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    record.add_argument("--overlap-seconds", type=float, default=DEFAULT_OVERLAP_SECONDS)

    subparsers.add_parser("stop", help="Stop the active recording and finish transcription")
    subparsers.add_parser("status", help="Show the active recording")
    subparsers.add_parser("doctor", help="Check local recording/transcription prerequisites")

    analyze = subparsers.add_parser("analyze", help="Prepare or run transcript analysis")
    analyze.add_argument("session_dir", type=Path)
    analyze.add_argument("--engine", choices=("none", "claude"), default="none")

    refine = subparsers.add_parser("refine", help="Re-transcribe saved audio with the final model")
    refine.add_argument("session_dir", type=Path)
    refine.add_argument("--language", choices=("en", "zh"), default=None)

    packet = subparsers.add_parser(
        "translation-packet",
        help="Build a timestamp-preserving translation packet from a transcript",
    )
    packet.add_argument("transcript", type=Path)
    packet.add_argument("--output-root", type=Path)
    packet.add_argument("--source-language", default="auto")
    packet.add_argument("--target-language", default="zh-CN")

    document = subparsers.add_parser(
        "document-packet",
        help="Build local action, claim, and task prompts from a PDF/TXT/Markdown document",
    )
    document.add_argument("document", type=Path)
    document.add_argument("--output-root", type=Path)

    monitor = subparsers.add_parser("_monitor", help=argparse.SUPPRESS)
    monitor.add_argument("--session-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command in {"start", "_record"}:
        if args.chunk_seconds < 5:
            raise RuntimeError("chunk-seconds must be at least 5")
        if not 0 <= args.overlap_seconds < args.chunk_seconds / 2:
            raise RuntimeError("overlap-seconds must be non-negative and less than half a chunk")
    if args.command == "start":
        return _start(args)
    if args.command == "_record":
        return _record(args)
    if args.command == "stop":
        return _stop()
    if args.command == "status":
        return _status()
    if args.command == "doctor":
        return _doctor()
    if args.command == "_monitor":
        return _monitor(args.session_dir.resolve())
    if args.command == "analyze":
        output = _run_analysis(args.session_dir.resolve(), args.engine)
        print(f"Analysis artifact: {output}")
        return 0
    if args.command == "refine":
        session_dir = args.session_dir.resolve()
        _refine_transcript(session_dir, args.language)
        _write_analysis_packet(session_dir)
        print(f"Final transcript: {session_dir / 'transcript.txt'}")
        return 0
    if args.command == "translation-packet":
        output = _write_translation_packet(
            args.transcript.resolve(),
            args.output_root.resolve() if args.output_root else None,
            args.source_language,
            args.target_language,
        )
        print(f"Translation packet: {output}")
        return 0
    if args.command == "document-packet":
        output = _write_document_packet(
            args.document.resolve(),
            args.output_root.resolve() if args.output_root else None,
        )
        print(f"Document packet: {output}")
        return 0
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
