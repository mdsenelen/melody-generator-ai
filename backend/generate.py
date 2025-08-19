# backend/generate.py

import torch
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from vae import VAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/vae.pt"
N_MELS = 64
SR = 22050


def generate_sample(model, z_dim=128):
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, z_dim).to(DEVICE)
        recon = model.decode(z)
        recon = recon.squeeze().cpu().numpy()
        recon = (recon * 80) - 80  # unnormalize
        mel = librosa.db_to_power(recon)
        audio = librosa.feature.inverse.mel_to_audio(mel, sr=SR)
        return audio


if __name__ == "__main__":
    model = VAE(input_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    audio = generate_sample(model)

    output_path = "generated_sample.wav"
    sf.write(output_path, audio, SR)
    print(f"✅ Generated sample saved as {output_path}")
