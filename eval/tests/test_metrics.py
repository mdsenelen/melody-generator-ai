import pytest

from eval.metrics import transcription_scores


def note(start, end, pitch):
    return {"start": start, "end": end, "pitch": pitch}


REF = [note(0.0, 1.0, 60), note(1.0, 2.0, 64), note(2.0, 3.0, 67)]


def test_perfect_estimate_scores_one():
    s = transcription_scores(REF, [dict(n) for n in REF])
    for key in ("onset", "onset_offset"):
        assert s[key] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_hand_checked_pair():
    est = [
        note(0.02, 1.0, 60),   # correct (20 ms late onset)
        note(1.0, 2.0, 64),    # correct
        note(2.10, 3.0, 67),   # onset 100 ms late: outside 50 ms
        note(2.0, 3.0, 66),    # wrong pitch
    ]
    s = transcription_scores(REF, est)
    # 2 of 4 estimates and 2 of 3 references match: P = 1/2, R = 2/3, F1 = 4/7
    assert s["onset"]["precision"] == pytest.approx(0.5)
    assert s["onset"]["recall"] == pytest.approx(2 / 3)
    assert s["onset"]["f1"] == pytest.approx(4 / 7)
    assert (s["n_ref"], s["n_est"]) == (3, 4)


def test_offsets_are_penalised_only_in_the_stricter_metric():
    est = [note(0.0, 0.3, 60), note(1.0, 2.0, 64), note(2.0, 3.0, 67)]  # first note cut short
    s = transcription_scores(REF, est)
    assert s["onset"]["f1"] == pytest.approx(1.0)
    assert s["onset_offset"]["recall"] == pytest.approx(2 / 3)


def test_empty_estimate_scores_zero_without_raising():
    s = transcription_scores(REF, [])
    assert s["onset"]["f1"] == 0.0 and s["onset_offset"]["f1"] == 0.0
