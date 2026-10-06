"""PCM timeline and normalized draft mix; no metadata is sent to TTS."""

import hashlib
import json
import math
import random
import re
import subprocess
import sys
import wave
from array import array

from crosstalk.artifacts import digest, write_json
from crosstalk.editor.artifacts import active_spoken, stage_paths
from crosstalk.editor.quality import discussion_approved
from crosstalk.editor.speech import speech_text
from crosstalk.models import Turn

RATE = 24000


def timeline(turns, durations, seed=7):
    rng = random.Random(seed)
    rows, by_idx, cursor = [], {}, 0
    for turn, duration in zip(turns, durations, strict=True):
        if turn.kind == "backchannel":
            previous = next((r for r in reversed(rows) if r["kind"] != "backchannel"), None)
            start = (previous["start_ms"] + previous["end_ms"]) / 2 if previous else cursor
        elif turn.interrupts_idx is not None:
            if turn.interrupts_idx not in by_idx:
                raise ValueError("Interruption must point to an earlier clip")
            start = max(0, by_idx[turn.interrupts_idx]["end_ms"] - (turn.overlap_ms or 300))
        else:
            start = cursor + (rng.randint(150, 500) if rows else 0)
        row = {
            **turn.model_dump(),
            "start_ms": round(start),
            "end_ms": round(start + duration),
            "gain_db": -6 if turn.kind == "backchannel" else 0,
        }
        rows.append(row)
        by_idx[turn.idx] = row
        if turn.kind != "backchannel":
            cursor = max(cursor, start + duration)
    return rows


def pcm_samples(path, interrupted=False):
    samples = array("h")
    samples.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        samples.byteswap()
    if interrupted:
        # Preserve the last audible sample plus 75 ms, then fade the last 20 ms.
        last = next(
            (i for i in range(len(samples) - 1, -1, -1) if abs(samples[i]) > 250), len(samples) - 1
        )
        samples = samples[: min(len(samples), last + int(RATE * 0.075) + 1)]
        fade = min(len(samples), int(RATE * 0.02))
        for i in range(fade):
            samples[len(samples) - fade + i] = int(
                samples[len(samples) - fade + i] * (1 - i / max(fade - 1, 1))
            )
    return samples


def render_audio(store, tts, research_dir, voices, preview=True):
    factcheck_dir, polish_dir = stage_paths(research_dir)
    # Legacy fixture/draft fallback; CLI always creates the current checks before audio.
    if not (factcheck_dir / "factcheck.json").exists():
        factcheck_dir = research_dir / "factcheck"
    originals = json.loads((research_dir / "discussion_full/transcript.json").read_text())
    turns = active_spoken(research_dir)
    expected = [{**t.model_dump(), "text": speech_text(t.text)} for t in turns]
    report = json.loads((factcheck_dir / "factcheck.json").read_text())
    speech = json.loads((factcheck_dir / "speech_ready.json").read_text())
    if report["transcript_hash"] != digest([t.model_dump() for t in turns]) or speech != expected:
        raise ValueError("Speech artifact does not match the checked transcript")
    base_hash = report.get("base_transcript_hash", report["transcript_hash"])
    if base_hash != digest(originals):
        raise ValueError("Evidence report is not linked to the approved discussion")
    turns = [Turn.model_validate(t) for t in speech]
    if not discussion_approved(research_dir):
        raise ValueError("M3 approval is required for audio")
    if not preview and not report["publication_gate_open"]:
        raise ValueError("Evidence review must pass before a publication render")
    if not {t.speaker for t in turns} <= set(voices):
        raise ValueError("Missing persona voice mapping")
    mix_hash = digest({"speech": speech, "voices": voices, "preview": preview, "version": 3})
    exported_files = [
        store.path / name
        for name in ("episode_preview.mp3", "episode_preview.wav", "timed_transcript.json")
    ]
    saved_report = store.path / "audio_acceptance.json"
    if saved_report.exists():
        saved = json.loads(saved_report.read_text())
        if (
            saved.get("mix_hash") == mix_hash
            and all(p.exists() for p in exported_files)
            and saved.get("export_hashes")
            == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in exported_files}
        ):
            saved["publication_gate_open"] = report["publication_gate_open"]
            write_json(saved_report, saved)
            return saved
    interrupted = {t.interrupts_idx for t in turns if t.interrupts_idx is not None}
    clips = [pcm_samples(tts.render(t, voices[t.speaker]), t.idx in interrupted) for t in turns]
    rows = timeline(turns, [len(c) * 1000 / RATE for c in clips])
    length = math.ceil(max(r["end_ms"] for r in rows) * RATE / 1000)
    mixed = array("f", [0]) * length
    for samples, row in zip(clips, rows, strict=True):
        start = round(row["start_ms"] * RATE / 1000)
        gain = 10 ** (row["gain_db"] / 20)
        for i, sample in enumerate(samples):
            if start + i < length:
                mixed[start + i] += sample * gain
    peak = max(abs(s) for s in mixed)
    scale = min(1, 30000 / max(peak, 1))
    output = array("h", (int(s * scale) for s in mixed))
    if sys.byteorder != "little":
        output.byteswap()
    store.path.mkdir(parents=True, exist_ok=True)
    raw = store.path / "mix_raw.wav"
    with wave.open(str(raw), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(output.tobytes())
    wav = store.path / "episode_preview.wav"
    mp3 = store.path / "episode_preview.mp3"
    analysis = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(raw),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    measured = json.loads(re.search(r"\{[^{}]*\}", analysis.stderr).group())
    normalization = (
        "loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:"
        f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(raw),
            "-af",
            normalization,
            "-ar",
            "48000",
            str(wav),
        ],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-b:a", "192k", str(mp3)], check=True
    )
    # Meter the encoded output: normalization can undershoot the requested LUFS.
    metering = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(mp3),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    final_levels = json.loads(re.search(r"\{[^{}]*\}", metering.stderr).group())
    gain = -16 - float(final_levels["input_i"])
    if abs(gain) > 0.3:
        adjusted = store.path / "mix_adjusted.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(wav),
                "-af",
                f"volume={gain}dB,alimiter=limit=0.82:level=false:latency=true",
                str(adjusted),
            ],
            check=True,
        )
        adjusted.replace(wav)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-b:a", "192k", str(mp3)],
            check=True,
        )
    write_json(store.path / "timed_transcript.json", rows)
    result = {
        "mix_hash": mix_hash,
        "export_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in exported_files
        },
        "original_transcript_hash": digest(originals),
        "spoken_transcript_hash": digest(speech),
        "polish_applied": (polish_dir / "polished.json").exists(),
        "preview": preview,
        "publication_gate_open": report["publication_gate_open"],
        "duration_seconds": length / RATE,
        "clipping_prevented": True,
        "human_listening_review_passed": False,
        "mp3": str(mp3),
        "wav": str(wav),
        "voices": voices,
        "limitations": [
            "Prototype tts-1 does not accept emotion instructions.",
            "Word-level alignment is not available; cut intelligibility and overlap require listening review.",
        ],
    }
    listening_review = store.path / "human_review.json"
    if listening_review.exists():
        approved = json.loads(listening_review.read_text())
        result["human_listening_review_passed"] = (
            approved.get("accepted") is True
            and approved.get("mp3_sha256") == hashlib.sha256(mp3.read_bytes()).hexdigest()
        )
    write_json(store.path / "audio_acceptance.json", result)
    return result
