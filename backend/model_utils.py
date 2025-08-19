# 📁 backend/model_utils.py
import torch
from model.vae import EnhancedVAE


def save_model(model, path, metadata=None):
    """Colab'de kullanılacak kayıt fonksiyonu"""
    state = {
        'state_dict': model.state_dict(),
        'config': {
            'input_shape': (1, 64, 431),
            'latent_dim': 128
        },
        'metadata': metadata or {}
    }
    torch.save(state, path)


def load_model(path, device='auto'):
    """Hem main.py hem test için ortak yükleme fonksiyonu"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    state = torch.load(path, map_location=device)
    model = EnhancedVAE(
        input_shape=state['config']['input_shape'],
        latent_dim=state['config']['latent_dim']
    )
    model.load_state_dict(state['state_dict'])
    model.to(device)
    model.eval()
    return model, state.get('metadata', {})
