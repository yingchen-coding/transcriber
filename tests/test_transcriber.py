import importlib.util
import io
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "transcriber.py"
SPEC = importlib.util.spec_from_file_location("transcriber", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TranscriberTest(unittest.TestCase):
    def test_slug_is_safe(self):
        self.assertEqual(MODULE._slug(" Google L6 / round 2 "), "Google-L6-round-2")

    def test_clock(self):
        self.assertEqual(MODULE._clock(3661.9), "01:01:01")

    def test_overlap_removal_works_for_chinese(self):
        previous = "我们先讨论系统设计的容量估算"
        current = "系统设计的容量估算然后进入存储方案"
        self.assertEqual(MODULE._remove_overlap(previous, current), "然后进入存储方案")

    def test_non_overlap_is_preserved(self):
        self.assertEqual(MODULE._remove_overlap("first answer", "next question"), "next question")

    def test_analysis_prompt_requires_evidence(self):
        prompt = MODULE._analysis_prompt("[00:00:00] hello", {"name": "test"})
        self.assertIn("timestamp", prompt)
        self.assertIn("Do not invent", prompt)
        self.assertIn("Simplified Chinese", prompt)
        self.assertIn("recorded session", prompt)

    def test_repetition_hallucination_is_removed(self):
        self.assertEqual(MODULE._clean_transcript("回" * 80), "")
        self.assertEqual(MODULE._clean_transcript("Kindle Kindle Kindle Kindle"), "")

    def test_silence_gate(self):
        silence = np.zeros(MODULE.SAMPLE_RATE * 8, dtype=np.float32)
        speech_like = silence.copy()
        speech_like[10_000:14_000] = 0.05
        self.assertFalse(MODULE._contains_speech(silence))
        self.assertTrue(MODULE._contains_speech(speech_like))

    def test_high_compression_decode_is_rejected(self):
        result = {
            "text": "Bring the language. " * 8,
            "segments": [
                {
                    "text": "Bring the language.",
                    "compression_ratio": 15.3,
                    "avg_logprob": -0.7,
                    "temperature": 1.0,
                }
            ],
        }
        self.assertEqual(MODULE._result_text(result), "")

    def test_confident_decode_is_kept(self):
        result = {
            "text": "Clarify the requirements.",
            "segments": [
                {
                    "text": "Clarify the requirements.",
                    "compression_ratio": 0.94,
                    "avg_logprob": -0.28,
                    "temperature": 0.0,
                }
            ],
        }
        self.assertEqual(MODULE._result_text(result), result["text"])

    def test_invalid_json_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "valid JSON"):
                MODULE._read_json(path)

    def test_final_pass_streams_bounded_wave_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory)
            audio_path = session_dir / "audio.wav"
            seconds = 65
            samples = np.zeros(MODULE.SAMPLE_RATE * seconds, dtype=np.float32)
            for start in range(0, len(samples), MODULE.SAMPLE_RATE):
                samples[start : start + MODULE.SAMPLE_RATE // 2] = 0.05
            pcm = (samples * 32767).astype("<i2")
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(MODULE.SAMPLE_RATE)
                wav_file.writeframes(pcm.tobytes())
            (session_dir / "transcript.txt").write_text("live\n", encoding="utf-8")
            (session_dir / "transcript.jsonl").write_text("{}\n", encoding="utf-8")

            lengths = []

            def fake_transcribe(audio, **_kwargs):
                lengths.append(len(audio))
                index = len(lengths)
                return {
                    "text": f"segment {index}",
                    "segments": [
                        {
                            "text": f"segment {index}",
                            "compression_ratio": 1.0,
                            "avg_logprob": -0.1,
                            "temperature": 0.0,
                        }
                    ],
                }

            with mock.patch.object(MODULE.mlx_whisper, "transcribe", side_effect=fake_transcribe):
                MODULE._refine_transcript(session_dir, "en")
                MODULE._refine_transcript(session_dir, "en")

            self.assertGreater(len(lengths), 1)
            self.assertLessEqual(max(lengths), MODULE.FINAL_CHUNK_SECONDS * MODULE.SAMPLE_RATE)
            self.assertTrue((session_dir / "transcript-live.txt").exists())
            self.assertEqual(
                (session_dir / "transcript-live.txt").read_text(encoding="utf-8"), "live\n"
            )
            records = [
                json.loads(line)
                for line in (session_dir / "transcript.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(all(record["pass"] == "final" for record in records))

    def test_version_has_one_source(self):
        expected = MODULE_PATH.with_name("VERSION").read_text(encoding="utf-8").strip()
        self.assertEqual(MODULE.__version__, expected)

    def test_analysis_is_local_by_default(self):
        with mock.patch.object(sys, "argv", ["transcriber", "analyze", "/tmp/session"]):
            args = MODULE._parser().parse_args()
        self.assertEqual(args.engine, "none")

    def test_recorder_pid_requires_matching_command(self):
        completed = mock.Mock(returncode=0, stdout="python transcriber.py _record /tmp/a")
        with (
            mock.patch.object(MODULE, "_process_alive", return_value=True),
            mock.patch.object(MODULE.subprocess, "run", return_value=completed),
        ):
            self.assertTrue(MODULE._is_expected_recorder(123, "/tmp/a"))
            self.assertFalse(MODULE._is_expected_recorder(123, "/tmp/other"))

    def test_status_marks_live_wrong_pid_as_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            active_file = Path(directory) / ".active-session.json"
            MODULE._atomic_json(
                active_file,
                {
                    "pid": 123,
                    "session_dir": "/tmp/session",
                    "status": "recording",
                },
            )
            completed = mock.Mock(returncode=0, stdout="python other.py")
            output = io.StringIO()
            with (
                mock.patch.object(MODULE, "ACTIVE_FILE", active_file),
                mock.patch.object(MODULE, "_process_alive", return_value=True),
                mock.patch.object(MODULE.subprocess, "run", return_value=completed),
                mock.patch("sys.stdout", output),
            ):
                self.assertEqual(MODULE._status(), 0)
            self.assertIn("Status: stale", output.getvalue())


if __name__ == "__main__":
    unittest.main()
