from __future__ import annotations

import bisect
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency
    import pretty_midi
except Exception:  # pragma: no cover - optional dependency
    pretty_midi = None


PITCH_OFFSET = 0
DUR_OFFSET = 128
TEMPO_OFFSET = 160
PAD_TOKEN = 176
VOCAB_SIZE = 177
DUR_BINS = [0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
TEMPO_BINS = list(range(40, 205, 10))
MOOD_LABELS = ["happy", "sad", "neutral"]


class MelodyEncoder(nn.Module):
    def __init__(self, vocab: int, emb_dim: int, hidden: int, latent: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=vocab - 1)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_var = nn.Linear(hidden, latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.gru(self.emb(x))
        hidden = hidden.squeeze(0)
        return self.fc_mu(hidden), self.fc_var(hidden)


class MelodyDecoder(nn.Module):
    def __init__(self, vocab: int, emb_dim: int, hidden: int, latent: int, n_cond: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=vocab - 1)
        self.fc_z = nn.Linear(latent + n_cond, hidden)
        self.gru = nn.GRU(emb_dim + hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, z_cond: torch.Tensor, tokens: torch.Tensor, tf: float = 0.5) -> torch.Tensor:
        batch_size, seq_len = tokens.size()
        hidden = torch.tanh(self.fc_z(z_cond)).unsqueeze(0)
        current = tokens[:, 0]
        logits: list[torch.Tensor] = []
        for step in range(1, seq_len):
            embedded = self.emb(current).unsqueeze(1)
            context = hidden.squeeze(0).unsqueeze(1).expand(-1, 1, -1)
            output, hidden = self.gru(torch.cat([embedded, context], dim=-1), hidden)
            step_logits = self.out(output.squeeze(1))
            logits.append(step_logits)
            if torch.rand(1).item() < tf:
                current = tokens[:, step]
            else:
                current = step_logits.argmax(-1)
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate(self, z_cond: torch.Tensor, seq_len: int = 128, temperature: float = 1.0) -> torch.Tensor:
        batch_size = z_cond.size(0)
        hidden = torch.tanh(self.fc_z(z_cond)).unsqueeze(0)
        current = torch.zeros(batch_size, dtype=torch.long, device=z_cond.device)
        outputs = [current]
        for _ in range(seq_len - 1):
            embedded = self.emb(current).unsqueeze(1)
            context = hidden.squeeze(0).unsqueeze(1).expand(-1, 1, -1)
            output, hidden = self.gru(torch.cat([embedded, context], dim=-1), hidden)
            logits = self.out(output.squeeze(1)) / max(temperature, 1e-4)
            current = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
            outputs.append(current)
        return torch.stack(outputs, dim=1)


class MelodyCVAE(nn.Module):
    def __init__(self, vocab: int = 177, emb_dim: int = 32, hidden: int = 64, latent: int = 16, n_moods: int = 3):
        super().__init__()
        self.latent = latent
        self.n_moods = n_moods
        self.encoder = MelodyEncoder(vocab, emb_dim, hidden, latent)
        self.decoder = MelodyDecoder(vocab, emb_dim, hidden, latent, n_moods)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def forward(self, tokens: torch.Tensor, mood_oh: torch.Tensor, tf: float = 0.5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(tokens)
        z = self.reparameterise(mu, logvar)
        return self.decoder(torch.cat([z, mood_oh], dim=-1), tokens, tf), mu, logvar


class MelStateEncoder(nn.Module):
    def __init__(self, mel_bins: int = 80, T_win: int = 16, enc_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(mel_bins, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, enc_dim),
            nn.LayerNorm(enc_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransitionDiscriminator(nn.Module):
    def __init__(self, enc_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(enc_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, s_enc: torch.Tensor, sp_enc: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([s_enc, sp_enc], dim=-1))


class MINENetwork(nn.Module):
    def __init__(self, sa_dim: int, sp_dim: int, hidden: int = 64):
        super().__init__()
        self.T = nn.Sequential(
            nn.Linear(sa_dim + sp_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("ema_et", torch.tensor(1.0))
        self.ema_decay = 0.99

    def forward(self, sa: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        batch_size = sa.size(0)
        t_joint = self.T(torch.cat([sa, sp], dim=-1))
        sp_shuf = sp[torch.randperm(batch_size, device=sa.device)]
        t_marginal = self.T(torch.cat([sa, sp_shuf], dim=-1))
        et = torch.exp(t_marginal)
        with torch.no_grad():
            self.ema_et = self.ema_decay * self.ema_et + (1 - self.ema_decay) * et.mean()
        return t_joint.mean() - (et / (self.ema_et + 1e-8)).mean()


class MelodyPPOActorCritic(nn.Module):
    def __init__(self, enc_dim: int = 64, latent_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.actor_mu = nn.Sequential(
            nn.Linear(enc_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(latent_dim))
        self.critic = nn.Sequential(
            nn.Linear(enc_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.actor_mu(s_enc)
        std = self.log_std.exp().expand_as(mu)
        value = self.critic(s_enc).squeeze(-1)
        return mu, std, value

    def get_dist(self, s_enc: torch.Tensor) -> torch.distributions.Normal:
        mu, std, _ = self(s_enc)
        return torch.distributions.Normal(mu, std)


def midi_to_tokens(midi_path: str | Path, max_notes: int = 64) -> list[int]:
    if pretty_midi is None:
        raise RuntimeError("pretty_midi is required for midi_to_tokens")
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        tempo = midi.estimate_tempo()
    except Exception:
        return []

    tokens = [TEMPO_OFFSET + min(bisect.bisect_left(TEMPO_BINS, tempo), 15)]
    notes = sorted(
        [note for instrument in midi.instruments if not instrument.is_drum for note in instrument.notes],
        key=lambda note: note.start,
    )
    for note in notes[:max_notes]:
        duration = note.end - note.start
        tokens.append(max(0, min(127, note.pitch)))
        tokens.append(DUR_OFFSET + min(bisect.bisect_left(DUR_BINS, duration), 31))
    return tokens


def tokens_to_midi(tokens: list[int], bpm: float = 120.0, out_path: str | Path = "/tmp/gen.mid") -> str:
    if pretty_midi is None:
        raise RuntimeError("pretty_midi is required for tokens_to_midi")
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0)
    current_time = 0.0
    pending_pitch: Optional[int] = None
    for token in tokens:
        if PITCH_OFFSET <= token < DUR_OFFSET:
            pending_pitch = token - PITCH_OFFSET
        elif DUR_OFFSET <= token < TEMPO_OFFSET and pending_pitch is not None:
            duration = float(DUR_BINS[min(token - DUR_OFFSET, len(DUR_BINS) - 1)])
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=80,
                    pitch=pending_pitch,
                    start=current_time,
                    end=current_time + duration,
                )
            )
            current_time += duration
            pending_pitch = None
    midi.instruments.append(instrument)
    midi.write(str(out_path))
    return str(out_path)


def heuristic_mood_from_metrics(tempo_bpm: float, average_pitch: float, key_label: str = "") -> tuple[int, str]:
    # key_label may be "" when key detection failed; both flags are False then,
    # which gracefully falls back to the original pitch-only heuristic.
    is_minor = "minor" in key_label.lower()
    is_major = "major" in key_label.lower()

    # Fast + bright key (or high register) → happy
    if tempo_bpm > 110 and (is_major or average_pitch > 65):
        return 0, MOOD_LABELS[0]
    # Slow + dark key (or low register) → sad
    if tempo_bpm < 80 and (is_minor or average_pitch < 60):
        return 1, MOOD_LABELS[1]
    # Minor key at moderate tempo (80–100 BPM) still reads as melancholic
    if is_minor and tempo_bpm < 100:
        return 1, MOOD_LABELS[1]
    return 2, MOOD_LABELS[2]
