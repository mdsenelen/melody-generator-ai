# 📁 backend/test_model.py
import torch
from model_utils import load_model


def run_tests():
    # 1. Model yükleme testi
    print("⏳ Model yükleniyor...")
    model, metadata = load_model('data/checkpoints/melody_vae_enhanced.pt')
    print(f"✅ Model yüklendi | Metadata: {metadata}")

    # 2. Boyut testi
    dummy_input = torch.randn(1, 1, 64, 431)
    with torch.no_grad():
        output, _, _ = model(dummy_input)
    print(
        f"🎛️ Input shape: {dummy_input.shape} -> Output shape: {output.shape}")

    # 3. Donanım testi
    print(f"⚡ Çalışma cihazı: {next(model.parameters()).device}")


if __name__ == "__main__":
    run_tests()
