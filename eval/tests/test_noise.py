import numpy as np
import pytest

from eval import noise

SR = 16_000


@pytest.fixture()
def signal():
    t = np.arange(SR * 5) / SR
    env = np.exp(-3.0 * (t % 1.0))  # decaying, note-like envelope
    return (0.4 * np.sin(2 * np.pi * 261.63 * t) + 0.2 * np.sin(2 * np.pi * 659.25 * t)) * env


@pytest.mark.parametrize("kind", ["white", "pink"])
@pytest.mark.parametrize("snr", [30, 20, 10, 5, 0])
def test_measured_snr_within_tenth_of_a_db(signal, kind, snr):
    rng = noise.stable_rng("t", kind, snr, seed=1)
    mixed, facts = noise.apply_condition(signal, SR, {"id": "x", "kind": kind, "snr_db": snr}, rng)
    assert abs(noise.measured_snr_db(signal, mixed) - snr) < 0.1
    assert abs(facts["measured_snr_db"] - snr) < 0.1


def test_pink_noise_falls_about_three_db_per_octave():
    x = noise.pink_noise(SR * 20, np.random.default_rng(0))
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)

    def band(lo, hi):  # mean power spectral density; pink has equal power per octave, not per Hz
        return spec[(freqs >= lo) & (freqs < hi)].mean()

    drop = 10 * np.log10(band(200, 400) / band(1600, 3200))  # three octaves apart
    assert 7.0 < drop < 11.0  # ideal: 9 dB


def test_noise_is_reproducible_for_the_same_keys(signal):
    a = noise.apply_condition(signal, SR, {"kind": "white", "snr_db": 10}, noise.stable_rng("c", seed=7))[0]
    b = noise.apply_condition(signal, SR, {"kind": "white", "snr_db": 10}, noise.stable_rng("c", seed=7))[0]
    c = noise.apply_condition(signal, SR, {"kind": "white", "snr_db": 10}, noise.stable_rng("d", seed=7))[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_silent_signal_has_no_defined_snr():
    with pytest.raises(ValueError):
        noise.mix_at_snr(np.zeros(100), np.ones(100), 10)


def test_reverb_decay_rate_matches_rt60():
    rt60 = 0.5
    ir = noise.exp_decay_ir(rt60, SR, np.random.default_rng(3))
    env = np.sqrt(np.convolve(ir**2, np.ones(400) / 400, mode="same"))
    early, late = env[SR // 20], env[int(0.3 * SR)]
    measured_db_per_sec = 20 * np.log10(early / late) / (0.3 - 0.05)
    assert 55 < measured_db_per_sec * rt60 < 65  # about 60 dB over RT60


def test_reverb_keeps_length_and_rms(signal):
    out = noise.apply_reverb(signal, SR, 0.8, np.random.default_rng(0))
    assert len(out) == len(signal)
    assert noise.power(out) == pytest.approx(noise.power(signal), rel=1e-6)


def test_hard_clip_limits_peak_and_reports_fraction(signal):
    clipped, fraction = noise.hard_clip(signal, 0.25)
    assert np.max(np.abs(clipped)) == pytest.approx(0.25 * np.max(np.abs(signal)))
    assert fraction == pytest.approx(np.mean(np.abs(signal) > 0.25 * np.max(np.abs(signal))))
    assert fraction > 0
