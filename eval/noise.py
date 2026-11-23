"""Degradations applied to a mono float signal. All pure and seeded."""
from __future__ import annotations

import zlib

import numpy as np
from scipy.signal import fftconvolve


def stable_rng(*parts: object, seed: int) -> np.random.Generator:
    """RNG derived from stable (cross-process) hashes, never Python's salted hash()."""
    words = [seed] + [zlib.crc32(str(p).encode()) for p in parts]
    return np.random.default_rng(words)


def power(x: np.ndarray) -> float:
    return float(np.mean(np.square(x, dtype=np.float64)))


def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f power spectrum (-3 dB/octave) by shaping white noise in the FFT domain."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    scale[0] = 0.0  # no DC
    return np.fft.irfft(spectrum * scale, n)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Return signal + noise scaled so the *realised* SNR equals snr_db.

    SNR is clip-level average power, so quiet passages see a worse local SNR
    than the headline number."""
    p_signal = power(signal)
    if p_signal <= 0.0:
        raise ValueError("cannot set an SNR against a silent signal")
    p_noise = power(noise)
    gain = np.sqrt(p_signal / (p_noise * 10.0 ** (snr_db / 10.0)))
    return signal.astype(np.float64) + gain * noise


def measured_snr_db(signal: np.ndarray, mixed: np.ndarray) -> float:
    return 10.0 * np.log10(power(signal) / power(mixed.astype(np.float64) - signal))


def exp_decay_ir(rt60_sec: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Noise impulse response whose amplitude falls 60 dB in rt60_sec."""
    n = int(round(rt60_sec * sr))
    t = np.arange(n) / sr
    return rng.standard_normal(n) * 10.0 ** (-3.0 * t / rt60_sec)


def apply_reverb(signal: np.ndarray, sr: int, rt60_sec: float, rng: np.random.Generator) -> np.ndarray:
    """100 % wet convolution, trimmed to the input length and RMS-matched to the
    dry signal so level changes do not confound the comparison."""
    wet = fftconvolve(signal.astype(np.float64), exp_decay_ir(rt60_sec, sr, rng))[: len(signal)]
    return wet * np.sqrt(power(signal) / power(wet))


def hard_clip(signal: np.ndarray, peak_ratio: float) -> tuple[np.ndarray, float]:
    """Clip at peak_ratio * the signal's own peak. Returns (clipped, fraction_of_samples_clipped)."""
    limit = peak_ratio * float(np.max(np.abs(signal)))
    clipped = np.clip(signal.astype(np.float64), -limit, limit)
    return clipped, float(np.mean(np.abs(signal) > limit))


def condition_list(snrs_db: list[int], rt60s: list[float], clip_ratio: float) -> list[dict]:
    conditions: list[dict] = [{"id": "clean", "kind": "clean"}]
    for kind in ("white", "pink"):
        for snr in snrs_db:
            conditions.append({"id": f"{kind}_snr{snr}", "kind": kind, "snr_db": snr})
    for rt60 in rt60s:
        conditions.append({"id": f"reverb_rt60_{rt60}", "kind": "reverb", "rt60_sec": rt60})
    conditions.append({"id": f"clip_{clip_ratio}", "kind": "clip", "peak_ratio": clip_ratio})
    return conditions


def apply_condition(
    signal: np.ndarray, sr: int, condition: dict, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Apply one condition. Returns (degraded float64 signal, measured facts about it)."""
    kind = condition["kind"]
    if kind == "clean":
        return signal.astype(np.float64), {}
    if kind in ("white", "pink"):
        noise = (white_noise if kind == "white" else pink_noise)(len(signal), rng)
        mixed = mix_at_snr(signal, noise, condition["snr_db"])
        return mixed, {"measured_snr_db": measured_snr_db(signal, mixed)}
    if kind == "reverb":
        return apply_reverb(signal, sr, condition["rt60_sec"], rng), {}
    if kind == "clip":
        clipped, fraction = hard_clip(signal, condition["peak_ratio"])
        return clipped, {"clipped_fraction": fraction}
    raise ValueError(f"unknown condition kind {kind!r}")
