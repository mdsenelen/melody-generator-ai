"""Chunked full-audio transcription — see docs/PLAN-full-transcription-split.md.

The heavy risk is `_merge_chunk_notes`: reconciling per-chunk Basic Pitch
output at overlapping chunk boundaries without splitting sustained notes or
merging genuine re-articulations. Those cases are covered here as pure-function
tests; the orchestration (`_transcribe_and_mood_chunked`) is covered with
Basic Pitch stubbed out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import inference  # noqa: E402

# Chunk geometry used in these tests: CHUNK=30, OVERLAP=4 -> HOP=26.
# chunk 0 = [0, 30], chunk 1 = [26, 56] (prev_chunk_end=30), ...


def _n(start, end, pitch, velocity=80):
    return {"start": float(start), "end": float(end), "pitch": int(pitch), "velocity": int(velocity)}


def test_first_chunk_passes_through_untouched():
    notes = [_n(1, 2, 60), _n(3, 4, 62)]
    out = inference._merge_chunk_notes([], notes, chunk_start=0.0, prev_chunk_end=None)
    assert out == notes
    assert out is not notes  # copied, not aliased


def test_weld_reconnects_a_note_truncated_at_the_boundary():
    merged = [_n(20, 30, 60)]  # ends exactly at chunk 0's right edge -> truncated
    chunk1 = [_n(26.1, 40, 60, velocity=90)]  # born at chunk 1's left edge, same pitch
    out = inference._merge_chunk_notes(merged, chunk1, chunk_start=26.0, prev_chunk_end=30.0)
    assert len(out) == 1
    assert out[0]["start"] == 20.0 and out[0]["end"] == 40.0 and out[0]["pitch"] == 60


def test_no_weld_when_the_previous_note_ended_cleanly():
    # merged note ends at 24, well before chunk 0's edge (30) -> a same-pitch
    # note at chunk 1's start is a real re-articulation, must stay separate.
    merged = [_n(20, 24, 60)]
    chunk1 = [_n(26.1, 40, 60)]
    out = inference._merge_chunk_notes(merged, chunk1, chunk_start=26.0, prev_chunk_end=30.0)
    assert len(out) == 2
    assert {round(o["start"], 1) for o in out} == {20.0, 26.1}


def test_duplicate_note_fully_inside_the_overlap_is_deduped():
    merged = [_n(27.0, 28.5, 64)]        # chunk 0's detection
    chunk1 = [_n(27.05, 28.4, 64)]       # chunk 1's detection of the same note
    out = inference._merge_chunk_notes(merged, chunk1, chunk_start=26.0, prev_chunk_end=30.0)
    assert len(out) == 1


def test_fresh_note_near_the_edge_with_no_match_is_kept():
    merged = [_n(10, 12, 55)]
    chunk1 = [_n(26.05, 29, 67)]  # near the edge but nothing to weld to
    out = inference._merge_chunk_notes(merged, chunk1, chunk_start=26.0, prev_chunk_end=30.0)
    assert len(out) == 2
    assert any(o["pitch"] == 67 for o in out)


def test_weld_chains_across_three_chunks():
    merged = inference._merge_chunk_notes([], [_n(20, 30, 60)], chunk_start=0.0, prev_chunk_end=None)
    merged = inference._merge_chunk_notes(
        merged, [_n(26.1, 56, 60)], chunk_start=26.0, prev_chunk_end=30.0
    )
    merged = inference._merge_chunk_notes(
        merged, [_n(52.1, 75, 60)], chunk_start=52.0, prev_chunk_end=56.0
    )
    assert len(merged) == 1
    assert merged[0]["start"] == 20.0 and merged[0]["end"] == 75.0


def test_transcribe_and_mood_chunked_assembles_full_midi_and_reports_progress(monkeypatch):
    sr = inference.NOTEBOOK_VARIANT_AUDIO_DEFAULTS["sample_rate"]

    monkeypatch.setattr(inference, "_probe_source_duration_sec", lambda raw: 90.0)
    monkeypatch.setattr(
        inference, "_decode_audio_window",
        lambda raw, target_sr, offset, dur: np.zeros(int(target_sr * dur), dtype=np.float32),
    )
    # Each chunk "detects" one note near its own start (relative time).
    monkeypatch.setattr(
        inference, "_run_basic_pitch_predict",
        lambda path: (b"", [_n(0.5, 2.0, 60)]),
    )
    monkeypatch.setattr(inference, "_detect_chords_from_audio", lambda audio, sr: ["C", "G"])

    progress_calls: list[int] = []
    result = inference._transcribe_and_mood_chunked(
        b"fake-audio", on_progress=lambda p: progress_calls.append(p) or True
    )

    assert result["source_duration_sec"] == 90.0
    assert result["truncated"] is False
    assert result["n_chunks"] >= 3
    assert result["n_notes"] >= 3           # one per chunk, at absolute offsets
    assert len(result["midi_bytes"]) > 0
    assert progress_calls == sorted(progress_calls)  # monotonic
    assert progress_calls[-1] <= 90
    assert result["detected_chords"] == ["C", "G"]


def test_transcribe_and_mood_chunked_rejects_audio_over_the_cap(monkeypatch):
    monkeypatch.setattr(inference, "_probe_source_duration_sec", lambda raw: inference.MAX_UPLOAD_DURATION_SEC + 60)
    with pytest.raises(HTTPException) as exc:
        inference._transcribe_and_mood_chunked(b"way-too-long")
    assert exc.value.status_code == 400
