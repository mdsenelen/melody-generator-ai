# backend/app/progressions.py
from typing import List

# Kurulu popüler progresyonlar (Roman numerals)
CURATED_PROGRESSIONS: List[List[str]] = [
    ["I", "V", "vi", "IV"],  # Popüler 4-chord progresyon
    ["ii", "V", "I"],        # Jazz ii-V-I
    ["I", "vi", "IV", "V"],  # Pop ballad
    ["I", "IV", "V", "IV"],  # Blues
    ["vi", "IV", "I", "V"],  # Popüler minor başlangıç
    ["I", "V", "vi", "iii"], # Popüler varyasyon
    ["I", "IV", "vi", "V"],  # Popüler 4-chord
]

NOTE_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_DEGREES = {
    "I": 0, "ii": 2, "iii": 4, "IV": 5, "V": 7, "vi": 9, "vii": 11
}

def _note_plus_semitones(root: str, semitones: int) -> str:
    """Bir notadan belirli yarım ton ilerideki notayı bulur"""
    i = NOTE_ORDER.index(root)
    return NOTE_ORDER[(i + semitones) % 12]

def roman_to_chords(roman: List[str], key: str = "C") -> List[str]:
    """Roman numeral progresyonunu gerçek akorlara çevirir"""
    chords: List[str] = []
    for deg in roman:
        base = deg.replace("°", "").replace("7", "").replace("maj7", "")
        if base not in MAJOR_DEGREES:
            continue
        semitones = MAJOR_DEGREES[base]
        root_note = _note_plus_semitones(key, semitones)
        
        # Maj/min basit kural: büyük harf maj, küçük harf min
        if base.islower() or base in {"ii", "iii", "vi"}:
            suffix = "m"
        elif base.startswith("vii"):
            suffix = "dim"
        else:
            suffix = ""
        chords.append(f"{root_note}{suffix}")
    return chords

def fetch_popular_progressions() -> List[List[str]]:
    """Kurulu popüler progresyonları döndürür"""
    return CURATED_PROGRESSIONS

def get_progression_names() -> List[str]:
    """Progresyon isimlerini döndürür"""
    return [
        "Popüler 4-Chord",
        "Jazz ii-V-I", 
        "Pop Ballad",
        "Blues",
        "Minor Pop",
        "Pop Varyasyon",
        "Popüler 4-Chord 2"
    ]