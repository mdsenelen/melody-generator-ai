import json
import random

import pytest

from eval import data


def make_catalog(n=60):
    return [
        {"midi_filename": f"2018/piece_{i:03d}.midi", "composer": "X", "title": f"T{i}", "year": 2018,
         "duration_sec": 100.0 + 7 * i}
        for i in range(n)
    ]


def dense_after_five(piece, start, length):
    return 10_000 if start > 5.0 else 0  # forces the retry path for early windows


def pick(catalog, seed, **kw):
    args = dict(n=20, length_sec=30.0, seed=seed, count_notes=dense_after_five, min_notes_per_sec=2.0)
    args.update(kw)
    return data.select_excerpts(catalog, **args)


def test_same_seed_gives_identical_selection():
    assert pick(make_catalog(), 42) == pick(make_catalog(), 42)


def test_different_seed_gives_different_selection():
    assert pick(make_catalog(), 42) != pick(make_catalog(), 43)


def test_selection_does_not_depend_on_catalog_order():
    shuffled = make_catalog()
    random.Random(0).shuffle(shuffled)
    assert pick(shuffled, 42) == pick(make_catalog(), 42)


def test_pieces_are_distinct_and_windows_fit_inside_the_piece():
    clips = pick(make_catalog(), 42)
    assert len({c["midi_filename"] for c in clips}) == 20
    for c in clips:
        assert 5.0 < c["source_start_sec"] <= c["duration_sec"] - c["length_sec"]


def test_density_rule_and_exclusions_are_honoured():
    catalog = make_catalog()
    first = pick(catalog, 42)
    excluded = [c["midi_filename"] for c in first]
    second = pick(catalog, 42, n=3, length_sec=90.0, exclude=excluded, id_prefix="L")
    assert not {c["midi_filename"] for c in second} & set(excluded)
    assert [c["id"] for c in second] == ["L00", "L01", "L02"]
    with pytest.raises(RuntimeError):
        pick(catalog, 42, count_notes=lambda *_: 0)


def test_manifest_text_is_byte_identical_across_builds():
    kwargs = dict(archive_sha256="abc", seed=42)
    a = data.dump_manifest(data.build_manifest(make_catalog(), dense_after_five, **kwargs))
    b = data.dump_manifest(data.build_manifest(make_catalog(), dense_after_five, **kwargs))
    assert a == b
    parsed = json.loads(a)
    assert len(parsed["clips"]) == 20 and len(parsed["long_clips"]) == 3
    assert not {c["midi_filename"] for c in parsed["clips"]} & {c["midi_filename"] for c in parsed["long_clips"]}
