# 📁 backend/app/inference.py
import io
import torch
import os
import numpy as np
from app.model.vae import WebVAE  # Updated model class
from app.model.utils import AudioProcessor
from librosa.feature.inverse import mel_to_audio
import soundfile as sf
import json
# from chord_model import predict_chords
from .chord_utils import get_chord_label, get_all_chord_labels
from .model.utils import AudioProcessor  # if needed
from .model.vae import WebVAE           # if needed

# Always use os.path for cross-platform compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(current_dir, 'model', 'weights')

# 1. Load model weights
model = WebVAE()
model_path = os.path.join(MODEL_DIR, 'web_model.pt')
state = torch.load(model_path, map_location=device)
model.load_state_dict(state['state_dict'])
model.eval()

# 2. Load audio parameters
config_path = os.path.join(MODEL_DIR, 'audio_params.json')
with open(config_path, 'r') as f:
    audio_config = json.load(f)

# Initialize processor with loaded config
processor = AudioProcessor(**audio_config['audio'])

# 3. Load latent vectors (if needed)
latent_path = os.path.join(MODEL_DIR, 'latent_vectors.npz')
latent_data = np.load(latent_path)
latent_vectors = latent_data['vectors']
chord_labels = latent_data['chords']  # If chord info is saved

# 🔧 Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔌 Model structure - Updated to match new config
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "web_model.pt")
MODEL_PATH = os.path.abspath(MODEL_PATH)
CONFIG = {
    "audio": {
        "sample_rate": 22050,
        "n_fft": 2048,
        "hop_length": 512,
        "win_length": 1024,
        "n_mels": 128,
        "fmin": 30,
        "fmax": 8000,
        "max_frames": 256
    }
}

# 🎛️ Audio processor - Updated for 128 mel bands
processor = AudioProcessor(
    sample_rate=CONFIG['audio']['sample_rate'],
    n_fft=CONFIG['audio']['n_fft'],
    hop_length=CONFIG['audio']['hop_length'],
    n_mels=CONFIG['audio']['n_mels'],
    fmin=CONFIG['audio']['fmin'],
    fmax=CONFIG['audio']['fmax']
)

# 🧠 Load VAE Model - Updated to new architecture
model = WebVAE()  # Using the fixed architecture
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state['state_dict'])
model.to(device)
model.eval()

# 🎼 Main inference function (called by API)


def generate_from_audio(audio_bytes: bytes, creativity: float = 0.7, chord: str = None):
    """
    Enhanced audio generation with chord conditioning
    Args:
        audio_bytes: Input audio as bytes
        creativity: Control variation (0-1)
        chord: Optional chord label (e.g., "C", "Am")
    Returns:
        tuple: (reconstructed_mel, generated_audio_bytes)
    """
    # 1. Preprocess input audio
    audio_stream = io.BytesIO(audio_bytes)
    temp_path = "temp_input.wav"
    with open(temp_path, "wb") as f:
        f.write(audio_stream.read())

    # Process to 128x256 mel spectrogram
    mel_tensor = processor.preprocess(temp_path).unsqueeze(0).to(device)

    # 2. Model inference with chord conditioning
    with torch.no_grad():
        if chord:  # Chord-conditioned generation
            z = torch.randn(1, model.latent_dim).to(device)
            if 'm' in chord.lower():  # Minor chord
                z[:, 0] = -1.0
            else:  # Major chord
                z[:, 0] = 1.0
            recon = model.decode(z)
        elif creativity > 0.5:  # Creative variation
            mu, logvar = model.encode(mel_tensor)
            z = model.reparameterize(mu, logvar * creativity)
            recon = model.decode(z)
        else:  # Direct reconstruction
            recon, _, _ = model(mel_tensor)

    # 3. Post-processing
    mel_output = recon.squeeze().cpu().numpy()

    # Convert mel to audio
    mel_db = (mel_output * 40) - 40  # Denormalize
    mel_power = librosa.db_to_amplitude(mel_db)
    audio = mel_to_audio(
        mel_power,
        sr=CONFIG['audio']['sample_rate'],
        n_fft=CONFIG['audio']['n_fft'],
        hop_length=CONFIG['audio']['hop_length'],
        win_length=CONFIG['audio']['win_length'],
        n_iter=32
    )

    # Convert audio to bytes
    audio_bytes_out = io.BytesIO()
    sf.write(audio_bytes_out, audio,
             CONFIG['audio']['sample_rate'], format='WAV')
    audio_bytes_out.seek(0)

    detected_chords = predict_chords(audio_bytes)

    # tam audio input’u kullanarak akorları çıkar
    return mel_output.tolist(), audio_bytes_out.read(), detected_chords


# Chord-based generation only


def generate_from_chord(chord: str = "C", duration: float = 5.0):
    """Generate audio from chord only"""
    with torch.no_grad():
        z = torch.randn(1, model.latent_dim).to(device)
        if 'm' in chord.lower():  # Minor chord
            z[:, 0] = -1.0
        else:  # Major chord
            z[:, 0] = 1.0

        recon = model.decode(z).squeeze().cpu().numpy()

        # Convert to audio
        mel_db = (recon * 40) - 40
        mel_power = librosa.db_to_amplitude(mel_db)
        audio = mel_to_audio(
            mel_power,
            sr=CONFIG['audio']['sample_rate'],
            n_fft=CONFIG['audio']['n_fft'],
            hop_length=CONFIG['audio']['hop_length'],
            win_length=CONFIG['audio']['win_length'],
            n_iter=32
        )

        # Trim to duration
        audio = audio[:int(duration * CONFIG['audio']['sample_rate'])]

        # Convert to bytes
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio,
                 CONFIG['audio']['sample_rate'], format='WAV')
        audio_bytes.seek(0)

        return recon.tolist(), audio_bytes.read(), [chord]
