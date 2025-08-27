# backend/main.py
import os
import sys
import json
import shutil
from pathlib import Path
import argparse

# Notebook/kitaplıktaki modüllere erişim (proje kökünü path'e ekle)
ROOT = Path(__file__).resolve().parents[1]  # repo root varsayımı
sys.path.append(str(ROOT))

# ---- Kullanıcı tarafı eğitim modülü (notebook kodu paketlenmiş sayalım)
# Aşağıdaki import, senin geliştirdiğin notebook kodunun Python modülü olarak erişilebilir
# olduğunu varsayıyor. Eğer dosya adı farklıysa burayı değiştir:
try:
    import singsong_improved as ss
except Exception:
    # Canvas’ta oluşturduğum dosya adını kullandıysan:
    import importlib.util
    nb_path = ROOT / "singsong_improved.py"
    if not nb_path.exists():
        raise RuntimeError("singsong_improved.py bulunamadı. Notebook kodunu bir .py dosyası olarak kaydet.")
    spec = importlib.util.spec_from_file_location("ss", str(nb_path))
    ss = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(ss)  # type: ignore

DEFAULT_WEIGHTS_DIR = ROOT / "backend" / "app" / "model" / "weights"
DEFAULT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

def _pick_best_checkpoint(ckpt_dir: Path) -> Path | None:
    """Öncelik: ft_best.pth > pre_best.pth > en son ft_ep*.pth > en son pre_ep*.pth"""
    ft_best = ckpt_dir / "ft_best.pth"
    pre_best = ckpt_dir / "pre_best.pth"
    if ft_best.exists():
        return ft_best
    if pre_best.exists():
        return pre_best
    ft_eps = sorted(ckpt_dir.glob("ft_ep*.pth"))
    if ft_eps:
        return ft_eps[-1]
    pre_eps = sorted(ckpt_dir.glob("pre_ep*.pth"))
    if pre_eps:
        return pre_eps[-1]
    return None

def _write_audio_params(out_dir: Path, audio_cfg: dict):
    out = out_dir / "audio_params.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump({"audio": audio_cfg}, f, indent=2)
    return out

def main():
    p = argparse.ArgumentParser(description="Train SingSong VAE and export final_vocal2accomp.pth for backend inference.")
    p.add_argument("--big-data", required=True, help="Ön-eğitim için büyük veri klasörü (wav).")
    p.add_argument("--my-data", required=True, help="Kişisel/ince ayar veri klasörü (wav).")
    p.add_argument("--epochs-pretrain", type=int, default=None)
    p.add_argument("--epochs-finetune", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--impl", choices=["transformer", "cnn"], default=os.getenv("WEBVAE_IMPL", "transformer"),
                   help="Model implementasyonu (backend/app/inference.py ile uyumlu).")
    p.add_argument("--weights-dir", type=str, default=str(DEFAULT_WEIGHTS_DIR),
                   help="final_vocal2accomp.pth ve audio_params.json çıkış klasörü.")
    p.add_argument("--final-name", type=str, default="final_vocal2accomp.pth")
    args = p.parse_args()

    # Implementasyon seçimi (CNN/Transformer)
    os.environ["WEBVAE_IMPL"] = args.impl

    # Notebook config ile arg’ları eşitle
    if args.epochs_pretrain is not None:
        ss.CONFIG["train"]["epochs_pretrain"] = args.epochs_pretrain
    if args.epochs_finetune is not None:
        ss.CONFIG["train"]["epochs_finetune"] = args.epochs_finetune
    if args.batch_size is not None:
        ss.CONFIG["model"]["batch_size"] = args.batch_size

    # Eğitim
    vae, chord_vocab, session_dir = ss.run_singsong_pipeline(
        big_data_dir=args.big_data,
        my_data_dir=args.my_data,
        epochs_pretrain=ss.CONFIG["train"]["epochs_pretrain"],
        epochs_finetune=ss.CONFIG["train"]["epochs_finetune"],
        batch_size=ss.CONFIG["model"]["batch_size"],
    )

    ckpt_dir = session_dir / "checkpoints"
    best = _pick_best_checkpoint(ckpt_dir)
    if best is None:
        raise SystemExit("Hiç checkpoint bulunamadı. Eğitim başarısız ya da veri yok.")

    # final_vocal2accomp.pth olarak kopyala
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    final_path = weights_dir / args.final_name
    shutil.copy(best, final_path)

    # chord_vocab zaten checkpoint içine gömülü; audio_params.json da yazalım
    audio_cfg = {
        "sample_rate": ss.CONFIG["audio"]["sample_rate"],
        "n_fft": ss.CONFIG["audio"]["n_fft"],
        "hop_length": ss.CONFIG["audio"]["hop_length"],
        "n_mels": ss.CONFIG["audio"]["n_mels"],
        "fmin": ss.CONFIG["audio"]["fmin"],
        "fmax": ss.CONFIG["audio"]["fmax"],
    }
    audio_json = _write_audio_params(weights_dir, audio_cfg)

    print("✅ Export tamam:")
    print("  Final weights :", final_path)
    print("  Audio params  :", audio_json)
    print("  Session       :", session_dir)

if __name__ == "__main__":
    main()
