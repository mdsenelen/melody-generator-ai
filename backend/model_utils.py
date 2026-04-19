import torch

try:
    from app.model.vae import WebVAE as EnhancedVAE
except Exception:  # pragma: no cover - legacy fallback
    from model.vae import WebVAE as EnhancedVAE


def save_model(model, path, metadata=None):
    """Shared model save helper."""
    state = {
        "state_dict": model.state_dict(),
        "config": {
            "input_shape": getattr(model, "input_shape", (1, 128, 256)),
            "latent_dim": getattr(model, "latent_dim", 258),
        },
        "metadata": metadata or {},
    }
    torch.save(state, path)


def load_model(path, device="auto"):
    """Shared model load helper for legacy scripts and tests."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(path, map_location=device)
    config = state.get("config", {})
    model = EnhancedVAE(
        input_shape=config.get("input_shape", (1, 128, 256)),
        latent_dim=config.get("latent_dim", 258),
    )
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()
    return model, state.get("metadata", {})
