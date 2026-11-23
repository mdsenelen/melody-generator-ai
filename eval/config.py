"""Every tunable of the evaluation lives here so a result file can echo it back."""
from __future__ import annotations

from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
BACKEND_DIR = REPO_ROOT / "backend"
DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"
PLOTS_DIR = EVAL_DIR / "plots"
MANIFEST_PATH = EVAL_DIR / "manifest.json"
SOUNDFONT_PATH = BACKEND_DIR / "app" / "soundfonts" / "GeneralUser-GS.sf2"

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
MAESTRO_ARCHIVE = "maestro-v3.0.0-midi.zip"
MAESTRO_SPLIT = "test"

SEED = 42
RENDER_SR = 16_000  # matches NOTEBOOK_VARIANT_AUDIO_DEFAULTS["sample_rate"] in production

N_CLIPS = 20
CLIP_SEC = 30.0
N_LONG_CLIPS = 3
LONG_CLIP_SEC = 150.0
MIN_NOTES_PER_SEC = 2.0  # skip near-empty windows (slow openings, cadenza gaps)

SNRS_DB = [30, 20, 10, 5, 0]
RT60S_SEC = [0.3, 0.8]
CLIP_PEAK_RATIO = 0.25  # hard clip at this fraction of the clip's own peak

# Onset tolerance is mir_eval's conventional 50 ms; offsets use its default
# 20 % of note length (with a 50 ms floor), the standard MAESTRO/Onsets-and-Frames setting.
ONSET_TOLERANCE_SEC = 0.05
OFFSET_RATIO = 0.2

# Weld sweep: production's current default is read from the code, not typed here.
WELD_EPS_ALTERNATIVES = [0.05, 0.30, 0.50]

QUICK_N_CLIPS = 3
