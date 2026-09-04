from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# The token/MIDI helpers and mood heuristic now live in the torch-free
# ``tokens`` module so the transcription path doesn't import torch. Re-exported
# here for backward compatibility -- existing
# ``from .model.colab_parity import <name>`` sites keep working.
from .tokens import (  # noqa: F401
    DUR_BINS,
    DUR_OFFSET,
    MOOD_LABELS,
    PAD_TOKEN,
    PITCH_OFFSET,
    TEMPO_BINS,
    TEMPO_OFFSET,
    VOCAB_SIZE,
    heuristic_mood_from_metrics,
    midi_to_tokens,
    tokens_to_bar_idx,
    tokens_to_chord_idx,
    tokens_to_midi,
)


class ChordEmbedding(nn.Module):
    def __init__(self, n_classes: int = 25, emb_dim: int = 12):
        super().__init__()
        self.emb = nn.Embedding(n_classes, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class BarPosEmbedding(nn.Module):
    def __init__(self, n_bins: int = 8, emb_dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(n_bins, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class MelodyEncoder(nn.Module):
    def __init__(self, vocab: int, emb_dim: int, hidden: int, latent: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=vocab - 1)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_lv = nn.Linear(hidden, latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.gru(self.emb(x))
        hidden = hidden.squeeze(0)
        return self.fc_mu(hidden), self.fc_lv(hidden)


class MelodyDecoder(nn.Module):
    def __init__(self, vocab: int, emb_dim: int, hidden: int, latent: int, n_cond: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=vocab - 1)
        self.fc_z = nn.Linear(latent + n_cond, hidden)
        self.gru = nn.GRU(emb_dim + hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward_teacher(self, z_cond: torch.Tensor, tokens: torch.Tensor, tf: float = 0.5) -> torch.Tensor:
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

    def forward(self, z_cond: torch.Tensor, tokens: torch.Tensor, tf: float = 0.5) -> torch.Tensor:
        return self.forward_teacher(z_cond, tokens, tf)

    def sample(
        self,
        z_cond: torch.Tensor,
        seq_len: int = 129,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = z_cond.size(0)
        device = z_cond.device
        hidden = torch.tanh(self.fc_z(z_cond)).unsqueeze(0)
        current = torch.zeros(batch_size, dtype=torch.long, device=device)
        outputs = [current]
        log_probs = torch.zeros(batch_size, device=device)
        entropies = torch.zeros(batch_size, device=device)
        for _ in range(seq_len - 1):
            embedded = self.emb(current).unsqueeze(1)
            context = hidden.squeeze(0).unsqueeze(1).expand(-1, 1, -1)
            output, hidden = self.gru(torch.cat([embedded, context], dim=-1), hidden)
            logits = self.out(output.squeeze(1)) / max(temperature, 1e-4)
            probs = F.softmax(logits, dim=-1)
            current = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            lp = torch.log(probs.gather(1, current.unsqueeze(1)).squeeze(1) + 1e-8)
            ent = -(probs * torch.log(probs + 1e-8)).sum(-1)
            log_probs = log_probs + lp
            entropies = entropies + ent
            outputs.append(current)
        return torch.stack(outputs, dim=1), log_probs, entropies / max(seq_len - 1, 1)

    @torch.no_grad()
    def generate(self, z_cond: torch.Tensor, seq_len: int = 128, temperature: float = 1.0) -> torch.Tensor:
        tokens, _, _ = self.sample(z_cond, seq_len=seq_len, temperature=temperature)
        return tokens

    def sequence_log_prob(
        self, z_cond: torch.Tensor, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward_teacher(z_cond, tokens, tf=1.0)
        log_p = F.log_softmax(logits, dim=-1)
        target = tokens[:, 1:]
        lp = log_p.gather(2, target.unsqueeze(2)).squeeze(2)
        mask = (target != PAD_TOKEN).float()
        log_probs = (lp * mask).sum(-1)
        entropy = -(F.softmax(logits, dim=-1) * log_p).sum(-1)
        mean_entropy = (entropy * mask).sum(-1) / (mask.sum(-1) + 1e-8)
        return log_probs, mean_entropy


class MelodyCVAE(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        vocab = int(cfg.get("vocab", 177))
        emb_dim = int(cfg.get("emb_dim", 48))
        hidden = int(cfg.get("hidden", 128))
        latent = int(cfg.get("latent_dim", 32))
        n_moods = int(cfg.get("n_moods", 3))
        use_chord = bool(cfg.get("use_chord_cond", True))
        use_bar = bool(cfg.get("use_bar_cond", True))
        chord_emb_dim = int(cfg.get("chord_emb_dim", 12)) if use_chord else 0
        bar_emb_dim = int(cfg.get("bar_emb_dim", 8)) if use_bar else 0
        n_chord_classes = int(cfg.get("n_chord_classes", 25))
        n_bar_bins = int(cfg.get("n_bar_bins", 8))

        self.latent = latent
        self.n_moods = n_moods
        self.use_chord = use_chord
        self.use_bar = use_bar
        self.chord_emb_dim = chord_emb_dim
        self.bar_emb_dim = bar_emb_dim

        self.chord_emb = ChordEmbedding(n_chord_classes, chord_emb_dim) if use_chord else None
        self.bar_emb = BarPosEmbedding(n_bar_bins, bar_emb_dim) if use_bar else None

        n_cond = n_moods + chord_emb_dim + bar_emb_dim
        self.encoder = MelodyEncoder(vocab, emb_dim, hidden, latent)
        self.decoder = MelodyDecoder(vocab, emb_dim, hidden, latent, n_cond)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def build_cond(
        self,
        mood_oh: torch.Tensor,
        chord_idx: Optional[torch.Tensor] = None,
        bar_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [mood_oh]
        if self.use_chord and chord_idx is not None and self.chord_emb is not None:
            parts.append(self.chord_emb(chord_idx))
        if self.use_bar and bar_idx is not None and self.bar_emb is not None:
            parts.append(self.bar_emb(bar_idx))
        return torch.cat(parts, dim=-1)

    def encode_mu(self, tokens: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(tokens)
        return mu

    def forward(
        self,
        tokens: torch.Tensor,
        mood_oh: torch.Tensor,
        chord_idx: Optional[torch.Tensor] = None,
        bar_idx: Optional[torch.Tensor] = None,
        tf: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(tokens)
        z = self.reparameterise(mu, logvar)
        cond = self.build_cond(mood_oh, chord_idx, bar_idx)
        return self.decoder.forward_teacher(torch.cat([z, cond], dim=-1), tokens, tf), mu, logvar


class MelStateEncoder(nn.Module):
    def __init__(self, mel_bins: int = 80, T_win: int = 32, enc_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(mel_bins, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, enc_dim),
            nn.LayerNorm(enc_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransitionDiscriminator(nn.Module):
    def __init__(self, enc_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, s_enc: torch.Tensor, sp_enc: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s_enc, sp_enc], dim=-1))

    @torch.no_grad()
    def reward(self, s_enc: torch.Tensor, sp_enc: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(self.net(torch.cat([s_enc, sp_enc], dim=-1)))
        return -torch.log(1.0 - prob + 1e-8).squeeze(-1)


class MINENetwork(nn.Module):
    def __init__(self, sa_dim: int, sp_dim: int, hidden: int = 96):
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


class ActorCritic(nn.Module):
    def __init__(self, enc_dim: int = 96, latent_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
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

    def dist_and_value(
        self, s_enc: torch.Tensor
    ) -> tuple[torch.distributions.Normal, torch.Tensor]:
        mu = self.actor(s_enc)
        std = self.log_std.exp().expand_as(mu)
        value = self.critic(s_enc).squeeze(-1)
        return torch.distributions.Normal(mu, std), value

    def get_dist(self, s_enc: torch.Tensor) -> torch.distributions.Normal:
        dist, _ = self.dist_and_value(s_enc)
        return dist

    def forward(self, s_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.actor(s_enc)
        std = self.log_std.exp().expand_as(mu)
        value = self.critic(s_enc).squeeze(-1)
        return mu, std, value


MelodyPPOActorCritic = ActorCritic


def build_mood_onehot(
    mood_idx_list: list[int],
    n_moods: int = 3,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    B = len(mood_idx_list)
    oh = torch.zeros(B, n_moods, device=device)
    for i, idx in enumerate(mood_idx_list):
        oh[i, max(0, min(int(idx), n_moods - 1))] = 1.0
    return oh
