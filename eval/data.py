"""MAESTRO excerpt selection, ground-truth extraction and FluidSynth rendering."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pretty_midi
import soundfile as sf

from . import config

ArchiveName = "maestro-v3.0.0"
MIN_NOTE_SEC = 0.02
PEAK_TARGET = 0.9
MIDI_RESOLUTION = 960  # ticks per beat at 120 bpm is about 0.5 ms per tick


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- dataset access

def download_archive() -> Path:
    """Fetch the MIDI-only archive (about 58 MB) if it is not already on disk."""
    import urllib.request

    archive = config.DATA_DIR / config.MAESTRO_ARCHIVE
    if not archive.exists():
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        part = archive.with_suffix(".part")
        urllib.request.urlretrieve(config.MAESTRO_URL, part)
        part.rename(archive)
    return archive


def extracted_root() -> Path:
    return config.DATA_DIR / "maestro" / ArchiveName


def ensure_extracted() -> Path:
    """Unzip the (already downloaded) MIDI-only archive once. Never executes anything from it."""
    root = extracted_root()
    if (root / f"{ArchiveName}.json").exists():
        return root
    archive = download_archive()
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(config.DATA_DIR / "maestro")
    return root


def load_catalog(root: Path, split: str) -> list[dict]:
    """Rows of the MAESTRO metadata for one split, sorted by file name so selection never depends on file order."""
    table = json.loads((root / f"{ArchiveName}.json").read_text())
    rows = []
    for key in table["midi_filename"]:
        if table["split"][key] != split:
            continue
        rows.append(
            {
                "midi_filename": table["midi_filename"][key],
                "composer": table["canonical_composer"][key],
                "title": table["canonical_title"][key],
                "year": int(table["year"][key]),
                "duration_sec": float(table["duration"][key]),
            }
        )
    return sorted(rows, key=lambda r: r["midi_filename"])


# ---------------------------------------------------------------- excerpt selection (pure)

def select_excerpts(
    catalog: Sequence[dict],
    *,
    n: int,
    length_sec: float,
    seed: int,
    count_notes: Callable[[dict, float, float], int],
    min_notes_per_sec: float,
    exclude: Iterable[str] = (),
    id_prefix: str = "c",
    max_tries_per_piece: int = 20,
) -> list[dict]:
    """Pick n excerpts from n distinct pieces. Deterministic in (catalog contents, seed).

    Pieces are shuffled with a seeded RNG; for each, random start times are drawn
    until the window holds enough notes (skips silent or near-empty passages)."""
    skip = set(exclude)
    pool = sorted(
        (r for r in catalog if r["duration_sec"] >= length_sec and r["midi_filename"] not in skip),
        key=lambda r: r["midi_filename"],
    )
    rng = random.Random(seed)
    order = rng.sample(pool, len(pool))
    picked: list[dict] = []
    for piece in order:
        if len(picked) == n:
            break
        for _ in range(max_tries_per_piece):
            start = round(rng.uniform(0.0, piece["duration_sec"] - length_sec), 3)
            notes = count_notes(piece, start, length_sec)
            if notes >= min_notes_per_sec * length_sec:
                picked.append(
                    {
                        "id": f"{id_prefix}{len(picked):02d}",
                        **piece,
                        "source_start_sec": start,
                        "length_sec": length_sec,
                        "n_notes": int(notes),
                    }
                )
                break
    if len(picked) < n:
        raise RuntimeError(f"only {len(picked)} of {n} excerpts met the density rule")
    return picked


def build_manifest(catalog: Sequence[dict], count_notes: Callable[[dict, float, float], int], *, archive_sha256: str, seed: int = config.SEED) -> dict:
    clips = select_excerpts(
        catalog, n=config.N_CLIPS, length_sec=config.CLIP_SEC, seed=seed,
        count_notes=count_notes, min_notes_per_sec=config.MIN_NOTES_PER_SEC,
    )
    long_clips = select_excerpts(
        catalog, n=config.N_LONG_CLIPS, length_sec=config.LONG_CLIP_SEC, seed=seed + 1,
        count_notes=count_notes, min_notes_per_sec=config.MIN_NOTES_PER_SEC,
        exclude=[c["midi_filename"] for c in clips], id_prefix="L",
    )
    return {
        "dataset": {
            "name": "MAESTRO v3.0.0 (MIDI only)",
            "url": config.MAESTRO_URL,
            "archive_sha256": archive_sha256,
            "split": config.MAESTRO_SPLIT,
        },
        "seed": seed,
        "selection": {
            "clip_sec": config.CLIP_SEC,
            "long_clip_sec": config.LONG_CLIP_SEC,
            "min_notes_per_sec": config.MIN_NOTES_PER_SEC,
        },
        "clips": clips,
        "long_clips": long_clips,
    }


def dump_manifest(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------- ground truth and audio

_midi_cache: dict[Path, pretty_midi.PrettyMIDI] = {}


def _parse(path: Path) -> pretty_midi.PrettyMIDI:
    if path not in _midi_cache:
        _midi_cache[path] = pretty_midi.PrettyMIDI(str(path))
    return _midi_cache[path]


def window_notes(path: Path, start_sec: float, length_sec: float) -> list[dict]:
    """Notes of a MIDI file inside [start, start+length], clipped to the window and shifted to 0.

    Sustain pedal (CC64) is deliberately not part of the result: the excerpt MIDI
    written from these notes carries no pedal, so the rendered audio and the
    note-off times agree exactly."""
    end_sec = start_sec + length_sec
    out = []
    for inst in _parse(path).instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if note.end <= start_sec or note.start >= end_sec:
                continue
            s, e = max(note.start, start_sec) - start_sec, min(note.end, end_sec) - start_sec
            if e - s >= MIN_NOTE_SEC:
                out.append({"start": s, "end": e, "pitch": int(note.pitch), "velocity": int(note.velocity)})
    return sorted(out, key=lambda n: (n["start"], n["pitch"]))


def counter_for(root: Path) -> Callable[[dict, float, float], int]:
    return lambda piece, start, length: len(window_notes(root / piece["midi_filename"], start, length))


def write_excerpt_midi(notes: Sequence[dict], path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=MIDI_RESOLUTION)
    piano = pretty_midi.Instrument(program=0)
    piano.notes = [pretty_midi.Note(velocity=n["velocity"], pitch=n["pitch"], start=n["start"], end=n["end"]) for n in notes]
    midi.instruments.append(piano)
    midi.write(str(path))


def read_ground_truth(midi_path: Path) -> list[dict]:
    """The exact reference notes: read back from the very MIDI file that gets rendered."""
    notes = [n for inst in pretty_midi.PrettyMIDI(str(midi_path)).instruments for n in inst.notes]
    return sorted(
        ({"start": float(n.start), "end": float(n.end), "pitch": int(n.pitch)} for n in notes),
        key=lambda n: (n["start"], n["pitch"]),
    )


def render_wav(midi_path: Path, wav_path: Path, length_sec: float) -> None:
    """FluidSynth CLI render to 16 kHz mono 16-bit WAV. Reverb and chorus are off so the
    'clean' condition is dry and the reverb condition is the only reverb in the study."""
    exe = shutil.which("fluidsynth")
    if exe is None:
        raise RuntimeError("fluidsynth CLI not found (brew install fluidsynth / apt install fluidsynth)")
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.wav"
        subprocess.run(
            [exe, "-ni", "-R", "0", "-C", "0", "-g", "0.6", "-r", str(config.RENDER_SR),
             "-T", "wav", "-O", "s16", "-F", str(raw), str(config.SOUNDFONT_PATH), str(midi_path)],
            check=True, capture_output=True,
        )
        audio, sr = sf.read(raw, dtype="float64", always_2d=True)
    assert sr == config.RENDER_SR, sr
    mono = audio.mean(axis=1)
    n = int(round(length_sec * sr))
    mono = np.pad(mono, (0, max(0, n - len(mono))))[:n]
    peak = float(np.max(np.abs(mono)))
    if peak <= 0.0:
        raise RuntimeError(f"FluidSynth rendered silence for {midi_path.name}")
    sf.write(str(wav_path), mono * (PEAK_TARGET / peak), sr, subtype="PCM_16")


def fluidsynth_version() -> str:
    exe = shutil.which("fluidsynth")
    if exe is None:
        return "not installed"
    return subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.splitlines()[0].strip()


def excerpt_paths(clip_id: str) -> tuple[Path, Path]:
    d = config.DATA_DIR / "excerpts"
    return d / f"{clip_id}.mid", d / f"{clip_id}.wav"


def prepare(*, rebuild_manifest: bool = False) -> dict:
    """Make the manifest (if absent) and materialise every excerpt's MIDI and WAV. Idempotent."""
    root = ensure_extracted()
    if rebuild_manifest or not config.MANIFEST_PATH.exists():
        archive_sha = sha256_file(config.DATA_DIR / config.MAESTRO_ARCHIVE)
        catalog = load_catalog(root, config.MAESTRO_SPLIT)
        manifest = build_manifest(catalog, counter_for(root), archive_sha256=archive_sha)
        config.MANIFEST_PATH.write_text(dump_manifest(manifest))
    manifest = json.loads(config.MANIFEST_PATH.read_text())
    (config.DATA_DIR / "excerpts").mkdir(parents=True, exist_ok=True)
    for clip in manifest["clips"] + manifest["long_clips"]:
        midi_path, wav_path = excerpt_paths(clip["id"])
        if midi_path.exists() and wav_path.exists():
            continue
        notes = window_notes(root / clip["midi_filename"], clip["source_start_sec"], clip["length_sec"])
        write_excerpt_midi(notes, midi_path)
        render_wav(midi_path, wav_path, clip["length_sec"])
    return manifest
