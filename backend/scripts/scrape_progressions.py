"""Chord progression scraper.

Combines two sources:
  1. Musicca hardcoded progressions (always available, no credentials needed)
  2. Hooktheory Trends API (optional, requires --username / --password)

Usage:
  python scripts/scrape_progressions.py
  python scripts/scrape_progressions.py --username EMAIL --password PW
  python scripts/scrape_progressions.py --username EMAIL --password PW --depth 4 --min-prob 0.08

Output: backend/app/data/progressions.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_PATH = Path(__file__).resolve().parent.parent / \
    "app" / "data" / "progressions.json"

HOOKTHEORY_BASE = "https://api.hooktheory.com/v1"

# ---------------------------------------------------------------------------
# Roman numeral → C-major chord conversion
# ---------------------------------------------------------------------------

_DEGREE_TO_ROOT: dict[str, str] = {
    "I":    "C",  "bII":  "Db", "II":   "D",  "bIII": "Eb",
    "III":  "E",  "IV":   "F",  "#IV":  "F#", "bV":   "Gb",
    "V":    "G",  "bVI":  "Ab", "VI":   "A",  "bVII": "Bb",
    "VII":  "B",  "#I":   "C#", "#V":   "G#",
}

# Default chord quality per diatonic degree
_UPPER_DEFAULT: dict[str, str] = {
    "I": "", "II": "", "III": "", "IV": "", "V": "",
    "VI": "", "VII": "",
    "bII": "", "bIII": "", "bVI": "", "bVII": "", "#IV": "",
}
_LOWER_DEFAULT: dict[str, str] = {
    "I": "m", "II": "m", "III": "m", "IV": "m",
    "V": "m", "VI": "m", "VII": "dim",
}

_ROMAN_RE = re.compile(
    r'^([b#]?)'
    r'(VII|III|bVII|bIII|bVI|bII|#IV|#I|#V|IV|VI|II|VII|I|V|'
    r'vii|iii|bvii|biii|bvi|bii|iv|vi|ii|v|i)'
    r'(.*?)$'
)


def roman_to_chord(chord_html: str) -> Optional[str]:
    """Convert a Hooktheory chord_HTML string to a C-major chord name.

    Returns None if the string cannot be parsed (node will be skipped).
    """
    s = (chord_html.strip()
         .replace("♭", "b")    # ♭
         .replace("♯", "#")    # ♯
         .replace("Δ", "maj7")  # Δ
         .replace("△", "maj7")  # △
         .replace("°", "dim")  # °
         .replace("ø", "m7b5")  # ø  half-diminished
         .replace("^", "maj7"))     # ^ shorthand for maj7

    m = _ROMAN_RE.match(s)
    if not m or not m.group(2):
        return None

    acc, roman, suffix = m.group(1), m.group(2), m.group(3)
    is_upper = roman[0].isupper()
    degree_key = acc + roman.upper()

    root = _DEGREE_TO_ROOT.get(degree_key)
    if root is None:
        return None

    default_q = (
        _UPPER_DEFAULT.get(roman.upper(), "")
        if is_upper
        else _LOWER_DEFAULT.get(roman.upper(), "m")
    )

    sl = suffix.lower()
    if "maj7" in sl:
        q = "maj7"
    elif "m7b5" in sl or "m7♭5" in sl:
        q = "m7"       # half-diminished → simplify to m7
    elif "dim7" in sl:
        q = "dim"
    elif "m7" in sl:
        q = "m7"
    elif "7" in sl:
        q = "m7" if default_q == "m" else "7"
    elif "dim" in sl:
        q = "dim"
    elif "aug" in sl or "+" in sl:
        q = "aug"
    elif "sus4" in sl:
        q = "sus4"
    elif "sus2" in sl:
        q = "sus2"
    elif "add9" in sl:
        q = "add9"
    elif "9" in sl:
        q = "9"
    elif sl.startswith("m") and default_q != "m":
        q = "m"
    else:
        q = default_q

    return root + q

# ---------------------------------------------------------------------------
# Genre inference (used for Hooktheory progressions without genre metadata)
# ---------------------------------------------------------------------------


def infer_genre(chords: list[str]) -> str:
    has_maj7 = any("maj7" in c for c in chords)
    has_dom7 = any(c.endswith("7") and "maj" not in c for c in chords)
    has_minor = any(c.endswith("m") or "m7" in c or c.endswith("m9")
                    for c in chords)
    dom7_count = sum(1 for c in chords if c.endswith("7") and "maj" not in c)

    if has_maj7 or (has_dom7 and has_minor and dom7_count == 1):
        return "Jazz"
    if dom7_count >= 2:
        return "Blues"
    if has_minor:
        return "Rock"
    return "Pop"

# ---------------------------------------------------------------------------
# Hooktheory API helpers
# ---------------------------------------------------------------------------


def _get_token(username: str, password: str) -> str:
    import requests
    resp = requests.post(
        f"{HOOKTHEORY_BASE}/users/auth",
        json={"username": username, "password": password},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("activkey") or data.get("token")
    if not token:
        raise ValueError("Hooktheory auth did not return a token")
    return token


def _api_get_nodes(session, cp: str = "") -> list[dict]:
    url = f"{HOOKTHEORY_BASE}/trends/nodes"
    if cp:
        url += f"?cp={cp}"
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  [hooktheory] 429 rate-limited, waiting {wait}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json() or []
        except Exception as exc:
            print(f"  [hooktheory] API error cp={cp!r}: {exc}")
            return []
    print(f"  [hooktheory] Giving up on cp={cp!r} after 3 attempts")
    return []


def _traverse(
    session,
    cp: str = "",
    current_chain: Optional[list[str]] = None,
    depth: int = 4,
    min_prob: float = 0.08,
    max_children: int = 5,
) -> list[list[str]]:
    """DFS traverse Hooktheory Trends to build complete chord chains."""
    if current_chain is None:
        current_chain = []

    nodes = _api_get_nodes(session, cp)
    time.sleep(1.2)  # respect rate limit (~1 req/s)

    # Take only the most probable nodes at each level
    nodes_filtered = sorted(
        (n for n in nodes if n.get("probability", 0) >= min_prob),
        key=lambda n: n["probability"],
        reverse=True,
    )[:max_children]

    results: list[list[str]] = []
    for node in nodes_filtered:
        chord = roman_to_chord(node.get("chord_HTML", ""))
        if chord is None:
            continue

        new_chain = current_chain + [chord]
        node_cp = node.get("child_path", "")

        if len(new_chain) >= depth:
            results.append(new_chain)
        else:
            sub = _traverse(session, node_cp, new_chain,
                            depth, min_prob, max_children)
            results.extend(sub)

    return results


def fetch_hooktheory(
    username: str,
    password: str,
    depth: int = 4,
    min_prob: float = 0.08,
    max_children: int = 5,
) -> list[dict]:
    try:
        import requests
    except ImportError:
        print("[hooktheory] 'requests' not installed — skipping")
        return []

    print("[hooktheory] Authenticating…")
    try:
        token = _get_token(username, password)
    except Exception as exc:
        print(f"[hooktheory] Auth failed: {exc}")
        return []

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    print("[hooktheory] Traversing trends (this may take a minute)…")
    chains = _traverse(session, depth=depth,
                       min_prob=min_prob, max_children=max_children)

    seen: set[tuple[str, ...]] = set()
    entries: list[dict] = []
    for i, chords in enumerate(chains):
        key = tuple(chords)
        if key in seen:
            continue
        seen.add(key)
        genre = infer_genre(chords)
        entries.append({
            "id": f"ht-{i + 1:04d}",
            "name": "–".join(chords),
            "genre": genre,
            "source": "hooktheory",
            "chords": chords,
            "song_title": None,
            "artist": None,
            "year": None,
        })

    print(f"[hooktheory] Fetched {len(entries)} unique progressions")
    return entries

# ---------------------------------------------------------------------------
# Musicca hardcoded data
# ---------------------------------------------------------------------------


_MUSICCA_GENRE: list[dict] = [
    # Pop
    {"id": "pop-1",  "name": "I–V–vi–IV",
        "genre": "Pop",  "chords": ["C", "G", "Am", "F"]},
    {"id": "pop-2",  "name": "I–IV–V–I",
        "genre": "Pop",  "chords": ["C", "F", "G", "C"]},
    {"id": "pop-3",  "name": "I–IV–Vsus4–V",
        "genre": "Pop",  "chords": ["C", "F", "Gsus4", "G"]},
    {"id": "pop-4",  "name": "I–vi–ii–V",
        "genre": "Pop",  "chords": ["C", "Am", "Dm", "G"]},
    {"id": "pop-5",  "name": "I–ii–IV–V",
        "genre": "Pop",  "chords": ["C", "Dm", "F", "G"]},
    {"id": "pop-6",  "name": "I–V–vi–V",
        "genre": "Pop",  "chords": ["C", "G", "Am", "G"]},
    # Rock
    {"id": "rock-1", "name": "I–vi–IV–V",
        "genre": "Rock", "chords": ["C", "Am", "F", "G"]},
    {"id": "rock-2", "name": "I–IV–I–V",
        "genre": "Rock", "chords": ["C", "F", "C", "G"]},
    {"id": "rock-3", "name": "I–♭III–IV–I",
        "genre": "Rock", "chords": ["C", "Eb", "F", "C"]},
    {"id": "rock-4", "name": "III–IV–IVm",
        "genre": "Rock", "chords": ["E", "F", "F", "Fm"]},
    {"id": "rock-5", "name": "ii–I–V–♭VII",
        "genre": "Rock", "chords": ["Dm", "C", "G", "Bb"]},
    {"id": "rock-6", "name": "I–♭VII–IV–vi",
        "genre": "Rock", "chords": ["G", "F", "C", "Am"]},
    # Jazz
    {"id": "jazz-1", "name": "ii7–V7–IΔ",           "genre": "Jazz",
        "chords": ["Dm7", "G7", "Cmaj7", "Cmaj7"]},
    {"id": "jazz-2", "name": "ii7–V7–IΔ–VI7",
        "genre": "Jazz", "chords": ["Dm7", "G7", "Cmaj7", "A7"]},
    {"id": "jazz-3", "name": "IΔ–vi7–ii7–V7",        "genre": "Jazz",
        "chords": ["Cmaj7", "Am7", "Dm7", "G7"]},
    {"id": "jazz-4", "name": "IΔ–#I°–ii7–V7",        "genre": "Jazz",
        "chords": ["Cmaj7", "C#dim", "Dm7", "G7"]},
    {"id": "jazz-5", "name": "IΔ–I7–IVΔ–IVm7",       "genre": "Jazz",
        "chords": ["Cmaj7", "C7", "Fmaj7", "Fm7"]},
    {"id": "jazz-6", "name": "iiø–ii7–V7–IΔ",        "genre": "Jazz",
        "chords": ["Am7", "Dm7", "G7", "Cmaj7"]},
    {"id": "jazz-7", "name": "iiø–V7–im7",
        "genre": "Jazz", "chords": ["Dm7", "G7", "Cm7", "Cm7"]},
    {"id": "jazz-8", "name": "im7–V7 (Minor vamp)",
     "genre": "Jazz", "chords": ["Cm7", "G7", "Cm7", "G7"]},
    # Blues
    {"id": "blues-1", "name": "12-Bar Blues", "genre": "Blues",
     "chords": ["C7", "C7", "C7", "C7", "F7", "F7", "C7", "C7", "G7", "F7", "C7", "C7"]},
]

_MUSICCA_SONGS: list[dict] = [
    {"id": "song-parcels-2018",       "name": "With or Without You",    "genre": "Pop",
        "chords": ["F", "Dm7", "Gsus4", "G"],           "song_title": "With or Without You",    "artist": "Parcels",                       "year": 2018},
    {"id": "song-shawnmendes-2016",   "name": "Mercy",                   "genre": "Pop",
        "chords": ["Em", "G", "Bm", "A"],               "song_title": "Mercy",                  "artist": "Shawn Mendes",                  "year": 2016},
    {"id": "song-johnnewman-2013",    "name": "Love Me Again",           "genre": "Pop",
        "chords": ["Gm", "Bb", "Dm", "C"],              "song_title": "Love Me Again",          "artist": "John Newman",                   "year": 2013},
    {"id": "song-capitalcities-2011", "name": "Safe And Sound",          "genre": "Pop",
        "chords": ["F", "C", "Am", "G"],                "song_title": "Safe And Sound",         "artist": "Capital Cities",                "year": 2011},
    {"id": "song-jessiej-2010",       "name": "Price Tag",               "genre": "Pop",
        "chords": ["F", "Am", "Dm", "Bb"],              "song_title": "Price Tag",              "artist": "Jessie J",                      "year": 2010},
    {"id": "song-florence-2009",      "name": "You Got the Love",        "genre": "Pop",
        "chords": ["Abm", "Gb", "Db", "Db"],            "song_title": "You Got the Love",       "artist": "Florence and the Machine",      "year": 2009},
    {"id": "song-bep-2003",           "name": "Where Is the Love?",      "genre": "Pop",
        "chords": ["F", "C", "Dm", "Bb"],               "song_title": "Where Is the Love?",     "artist": "The Black Eyed Peas",           "year": 2003},
    {"id": "song-gorillaz-2001",      "name": "19-2000 Soulchild Remix", "genre": "Pop",
        "chords": ["G", "F", "C", "G"],                 "song_title": "19-2000 Soulchild Remix", "artist": "Gorillaz",                      "year": 2001},
    {"id": "song-marymay-1999",       "name": "Shackles",                "genre": "Pop",
        "chords": ["Gm7", "Cm7", "D7", "Gm7"],          "song_title": "Shackles",               "artist": "Mary May",                      "year": 1999},
    {"id": "song-eagleeye-1997",      "name": "Save Tonight",            "genre": "Pop",
        "chords": ["Dm", "Bb", "F", "C"],               "song_title": "Save Tonight",           "artist": "Eagle Eye Cherry",              "year": 1997},
    {"id": "song-verve-1997",         "name": "Bitter Sweet Symphony",   "genre": "Rock",
        "chords": ["E", "Bm7", "Asus4", "A"],           "song_title": "Bitter Sweet Symphony",  "artist": "The Verve",                     "year": 1997},
    {"id": "song-cranberries-1994",   "name": "Zombie",                  "genre": "Rock",
        "chords": ["Em", "C", "G", "D"],                "song_title": "Zombie",                 "artist": "The Cranberries",               "year": 1994},
    {"id": "song-tlc-1994",           "name": "Waterfalls",              "genre": "Pop",
        "chords": ["Ebmaj7", "Bbadd9", "Dbmaj7", "Ab"], "song_title": "Waterfalls",             "artist": "TLC",                           "year": 1994},
    {"id": "song-haddaway-1993",      "name": "What Is Love",            "genre": "Pop",
        "chords": ["Gm", "Bb", "Dm", "F"],              "song_title": "What Is Love",           "artist": "Haddaway",                      "year": 1993},
    {"id": "song-isaak-1989",         "name": "Wicked Game",             "genre": "Rock",
        "chords": ["Bm", "A", "E", "E"],                "song_title": "Wicked Game",            "artist": "Chris Isaak",                   "year": 1989},
    {"id": "song-mcferrin-1988",      "name": "Don't Worry Be Happy",    "genre": "Pop",
        "chords": ["B", "C#m", "E", "B"],               "song_title": "Don't Worry Be Happy",   "artist": "Bobby McFerrin",                "year": 1988},
    {"id": "song-u2-1986",            "name": "With or Without You",     "genre": "Rock",
        "chords": ["D", "A", "Bm", "G"],                "song_title": "With or Without You",    "artist": "U2",                            "year": 1986},
    {"id": "song-laidback-1983",      "name": "Sunshine Reggae",         "genre": "Pop",
        "chords": ["Ab", "Cm7", "Fm7", "Bbm7"],         "song_title": "Sunshine Reggae",        "artist": "Laid Back",                     "year": 1983},
    {"id": "song-stevemiller-1982",   "name": "Abracadabra",             "genre": "Rock",
        "chords": ["Am", "Dm", "E", "Am"],              "song_title": "Abracadabra",            "artist": "Steve Miller Band",             "year": 1982},
    {"id": "song-grover-1981",        "name": "Just the Two of Us",      "genre": "Jazz",
        "chords": ["Db", "C7", "Fm7", "Ab"],            "song_title": "Just the Two of Us",     "artist": "Grover Washington Jr.",         "year": 1981},
    {"id": "song-bto-1974",           "name": "Takin' Care of Business", "genre": "Rock",
        "chords": ["C", "Bb", "F", "C"],                "song_title": "Takin' Care of Business", "artist": "Bachman Turner Overdrive",       "year": 1974},
    {"id": "song-eagles-1974",        "name": "Already Gone",            "genre": "Rock",
        "chords": ["G", "D", "C", "C"],                 "song_title": "Already Gone",           "artist": "The Eagles",                    "year": 1974},
    {"id": "song-hendrix-1967",       "name": "All Along the Watchtower", "genre": "Rock",
        "chords": ["Cm", "Bb", "Ab", "Bb"],             "song_title": "All Along the Watchtower", "artist": "Jimi Hendrix",                 "year": 1967},
]


def build_musicca_entries() -> list[dict]:
    entries: list[dict] = []
    for item in _MUSICCA_GENRE:
        entries.append({
            "id":         item["id"],
            "name":       item["name"],
            "genre":      item["genre"],
            "source":     "musicca",
            "chords":     item["chords"],
            "song_title": None,
            "artist":     None,
            "year":       None,
        })
    for item in _MUSICCA_SONGS:
        entries.append({
            "id":         item["id"],
            "name":       item["name"],
            "genre":      item["genre"],
            "source":     "musicca",
            "chords":     item["chords"],
            "song_title": item["song_title"],
            "artist":     item["artist"],
            "year":       item["year"],
        })
    return entries

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build progressions.json")
    parser.add_argument("--username", default="",
                        help="Hooktheory username/email")
    parser.add_argument("--password", default="", help="Hooktheory password")
    parser.add_argument("--depth",    type=int,   default=4,
                        help="Traversal depth (default 4)")
    parser.add_argument("--min-prob", type=float, default=0.08,
                        help="Min probability threshold (default 0.08)")
    parser.add_argument("--max-children", type=int, default=5,
                        help="Max children per node (default 5)")
    args = parser.parse_args()

    print("Building musicca entries…")
    all_entries = build_musicca_entries()
    print(f"  {len(all_entries)} entries from musicca")

    if args.username and args.password:
        ht_entries = fetch_hooktheory(
            args.username,
            args.password,
            depth=args.depth,
            min_prob=args.min_prob,
            max_children=args.max_children,
        )
        # Deduplicate by chord tuple against already existing entries
        existing_keys = {tuple(e["chords"]) for e in all_entries}
        new_ht = [e for e in ht_entries if tuple(
            e["chords"]) not in existing_keys]
        all_entries.extend(new_ht)
        print(f"  {len(new_ht)} new entries from Hooktheory (after dedup)")
    else:
        print("  No Hooktheory credentials — skipping API fetch")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(
        all_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_entries)} progressions -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
