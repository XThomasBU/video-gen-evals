import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import *


class OrthogonalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        seq_len = 35
        # Generate random orthogonal matrix [seq_len, d_model]
        rand = torch.randn(seq_len, d_model)
        q, _ = torch.linalg.qr(rand)
        self.pe = nn.Parameter(q.unsqueeze(0), requires_grad=False)  # Not learnable, fixed

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 35):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]


class TemporalTransformerV1(nn.Module):
    def __init__(self, input_dim, latent_dim, d_model=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, d_model),
        )
        self.positional = SinusoidalPositionalEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # [1, 1, d_model]
        self.proj = nn.Linear(d_model, latent_dim)
        self.frame_proj = nn.Linear(d_model, latent_dim)

    def forward(self, x, lengths=None):  # lengths unused
        """
        Args:
            x: [B, T, input_dim]
        Returns:
            x_out: [B, latent_dim]          # sequence-level embedding
            frame_embeddings: [B, T+1, d_model]   # frame-level (incl. CLS)
        """
        x = self.input_proj(x)  # [B, T, d_model]
        B, T, D = x.shape

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, 1, D)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)        # [B, T+1, D]

        x = self.positional(x)                       # [B, T+1, D]
        x = self.transformer(x)                      # [B, T+1, D]
        frame_embeddings = x

        # CLS token output at position 0
        cls_emb = x[:, 0, :]                         # [B, D]
        x_out = self.proj(cls_emb)                   # [B, latent_dim]
        x_out = nn.functional.normalize(x_out, p=2, dim=-1)

        frame_embeddings = self.frame_proj(frame_embeddings)  # [B, T+1, latent_dim]
        frame_embeddings = nn.functional.normalize(frame_embeddings, p=2, dim=-1)  # Normalize frame embeddings
        return x_out, frame_embeddings, 0


class TemporalTransformerV2(nn.Module):
    def __init__(self, input_dim, latent_dim, d_model=256, n_heads=4, n_layers=2, dropout=0.1, conv_kernel=5):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # ---- modality dims (RAW half) ----
        self.d_vit   = 1024
        self.d_glob  = 9
        self.d_pose  = 207
        self.d_beta  = 10
        self.d_kp2d  = 120
        self.one_pass_dim = self.d_vit + self.d_glob + self.d_pose + self.d_beta + self.d_kp2d  # 1370

        self.has_diff = (input_dim >= 2 * self.one_pass_dim)
        self.total_dim = 2 * self.one_pass_dim if self.has_diff else self.one_pass_dim

        # register modality names in fixed order (for weights)
        self.modalities = ["vit", "global", "pose", "beta", "kp2d"]
        self.M = len(self.modalities)

        # ---- per-modality projectors -> d_model (raw + optional diff) ----
        def proj(d_in):
            return nn.Sequential(
                nn.Linear(d_in, d_in),
                nn.GELU(),
                nn.Linear(d_in, d_model),
            )

        self.proj_raw = nn.ModuleDict({
            "vit":   proj(self.d_vit),
            "global":proj(self.d_glob),
            "pose":  proj(self.d_pose),
            "beta":  proj(self.d_beta),
            "kp2d":  proj(self.d_kp2d),
        })
        if self.has_diff:
            self.proj_diff = nn.ModuleDict({
                "vit":   proj(self.d_vit),
                "global":proj(self.d_glob),
                "pose":  proj(self.d_pose),
                "beta":  proj(self.d_beta),
                "kp2d":  proj(self.d_kp2d),
            })
        else:
            self.proj_diff = None

        # small norm per modality to align scales
        self.mod_norm = nn.ModuleDict({m: nn.LayerNorm(d_model) for m in self.modalities})

        # ---- gating: per-sample scalar weight for each modality (softmax) ----
        # compute from time-mean of each modality's projected features
        self.gate = nn.ModuleDict({m: nn.Linear(d_model, 1, bias=False) for m in self.modalities})
        self.register_buffer("tau", torch.tensor(1.0))  # softmax temperature (keep 1.0)

        # ---- transformer stack ----
        self.positional = SinusoidalPositionalEmbedding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # [1, 1, d_model]
        self.proj = nn.Linear(d_model, latent_dim)
        self.frame_proj = nn.Linear(d_model, latent_dim)

        # will hold [B, M] weights from last forward (vit, global, pose, beta, kp2d)
        self.last_mod_weights = None

        # --- attention pooling over frame tokens ---
        self.read_queries = nn.Parameter(torch.randn(4, d_model))  # 4 queries
        self.q_ln = nn.LayerNorm(d_model)
        self.kv_ln = nn.LayerNorm(d_model)
        self.read_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

    def _split_raw_diff(self, x):
        # x: [B,T,total_dim] -> (raw, diff?) each [B,T,1370]
        if self.has_diff:
            return x[:, :, :self.one_pass_dim], x[:, :, self.one_pass_dim:]
        else:
            return x, None

    def _split_modalities(self, x_part):
        # x_part: [B,T,1370] -> dict of [B,T,dim] in fixed order
        if x_part is None:
            return {m: None for m in self.modalities}
        d = [self.d_vit, self.d_glob, self.d_pose, self.d_beta, self.d_kp2d]
        chunks = torch.split(x_part, d, dim=-1)
        return dict(zip(self.modalities, chunks))

    def forward(self, x, lengths=None):  # lengths unused (fixed T)
        """
        Args:
            x: [B, T, input_dim] where input_dim is 1370 (RAW) or 2740 (RAW|DIFF)
        Returns:
            x_out: [B, latent_dim]
            frame_embeddings: [B, T+1, latent_dim]  # frame-level (incl. CLS), L2-normalized
        """
        x = x[:, :, :self.total_dim]  # in case extra dims sneak in
        B, T, _ = x.shape

        # ---- split into RAW / DIFF halves and then into modalities ----
        raw, diff = self._split_raw_diff(x)
        raw_parts = self._split_modalities(raw)
        diff_parts = self._split_modalities(diff)

        # ---- per-modality projected sequences (possibly raw+diff fused by addition) ----
        mod_seq = {}  # m -> [B,T,d_model]
        mod_mean = {} # m -> [B,d_model] (time-mean for gating)
        for m in self.modalities:
            seq = self.proj_raw[m](raw_parts[m])
            if self.has_diff:
                seq = seq + self.proj_diff[m](diff_parts[m])
            seq = self.mod_norm[m](seq)
            mod_seq[m] = seq
            mod_mean[m] = seq.mean(dim=1)  # [B,D]

        # ---- per-sample modality weights (softmax over M) ----
        logits = torch.stack([self.gate[m](mod_mean[m]).squeeze(-1) for m in self.modalities], dim=-1)  # [B,M]
        weights = torch.softmax(logits / self.tau, dim=-1)  # [B,M]
        self.last_mod_weights = weights           # for inspection

        # ---- fuse into a single token per frame by weighted sum ----
        # token_t = sum_m w_m * seq_m[:, t, :]
        fused = 0
        for i, m in enumerate(self.modalities):
            w = weights[:, i].unsqueeze(-1).unsqueeze(-1)   # [B,1,1]
            fused = fused + w * mod_seq[m]                  # broadcast over T and D
        x = fused                                           # [B, T, d_model]

        # ---- the rest is your original model ----
        D = x.size(-1)
        cls_tokens = self.cls_token.expand(B, 1, D)         # [B,1,D]
        x = torch.cat([cls_tokens, x], dim=1)               # [B, T+1, D]

        x = self.positional(x)                              # [B, T+1, D]
        x = self.transformer(x)                             # [B, T+1, D]
        frame_embeddings = x

        cls_emb = x[:, 0, :]                                # [B, D]
        x_out = self.proj(cls_emb)                          # [B, latent_dim]
        x_out = F.normalize(x_out, p=2, dim=-1)

        frame_embeddings = self.frame_proj(frame_embeddings)  # [B, T+1, latent_dim]
        frame_embeddings = F.normalize(frame_embeddings, p=2, dim=-1)
        # # return x_out, frame_embeddings

        # frame_embeddings = x

        # # ---- CLS + attention pooling over frames ----
        # cls_emb = x[:, 0, :]        # [B, D]
        # tokens  = x[:, 1:, :]       # [B, T, D]  (exclude CLS for attention)

        # Q = self.q_ln(self.read_queries.unsqueeze(0).expand(B, -1, -1))  # [B,Q,D]
        # K = self.kv_ln(tokens)                                           # [B,T,D]
        # V = K
        # readouts, _ = self.read_attn(Q, K, V)                            # [B,Q,D]
        # pooled = readouts.mean(dim=1)                                    # [B,D]  (or max)
        # # mixed  = 0.5 * cls_emb + 0.5 * pooled
        # concatenated = torch.cat([cls_emb, pooled], dim=-1)

        # # project to embedding
        # x_out = self.proj(concatenated)                           # [B, latent_dim]
        # x_out = F.normalize(x_out, p=2, dim=-1)

        # # keep returning frame-level embeddings as you already do
        # frame_embeddings = self.frame_proj(frame_embeddings)   # [B, T+1, latent_dim]
        # frame_embeddings = F.normalize(frame_embeddings, p=2, dim=-1)

        return x_out, frame_embeddings, _

def init_weight(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, a=0.2)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Fixed sinusoidal positions
# -----------------------------
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 35):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] or [B, T+1, D]
        return x + self.pe[:, :x.size(1), :]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Residual TCN block (no resampling)
# -----------------------------
class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, p: float = 0.1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride=1,
                               padding=pad, dilation=dilation, bias=False)
        self.act1  = nn.GELU()
        self.drop1 = nn.Dropout(p)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, stride=1,
                               padding=pad, dilation=dilation, bias=False)
        self.act2  = nn.GELU()
        self.norm  = nn.GroupNorm(1, channels)

    def forward(self, x):  # x: [B, C, T]
        res = x
        y = self.act1(self.conv1(x))
        y = self.drop1(y)
        y = self.conv2(y)
        y = self.act2(y + res)
        return self.norm(y)


# -----------------------------
# Length-preserving per-modality encoder (TCN)
# -----------------------------
class MovementConvEncoder(nn.Module):
    """
    Input:  [B, T, F]
    Output: [B, T, D]   (no downsampling)
    """
    def __init__(self, d_in: int, d_out: int, p: float = 0.1, k: int = 5, dilations=(1, 2, 4)):
        super().__init__()
        self.stem = nn.Conv1d(d_in, d_out, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList([
            TemporalConvBlock(d_out, kernel_size=k, dilation=d, p=p) for d in dilations
        ])
        self.proj = nn.Linear(d_out, d_out, bias=False)

    def forward(self, x_btf: torch.Tensor) -> torch.Tensor:
        # x_btf: [B, T, F]
        x = x_btf.transpose(1, 2)                 # [B, F, T]
        y = self.stem(x)                           # [B, D, T]
        for blk in self.blocks:
            y = blk(y)                             # [B, D, T]
        y = y.transpose(1, 2)                      # [B, T, D]
        return self.proj(y)                        # [B, T, D]


# -----------------------------
# Modality energy equalizer (RMS to 1 per modality)
# -----------------------------
class ModalityEqualizer(nn.Module):
    """
    Normalizes each modality sequence to target RMS=1 (per-batch, per-modality).
    Keeps gradient flow (no detach).
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, m_tokens: torch.Tensor) -> torch.Tensor:
        # m_tokens: [B, T, M, D]
        rms = torch.sqrt(m_tokens.pow(2).mean(dim=(1, 3), keepdim=True).clamp_min(self.eps))  # [B,1,M,1]
        return m_tokens / rms  # full grad path


# -----------------------------
# Modality dropout (training only)
# -----------------------------
class ModalityDropout(nn.Module):
    """
    Drops whole modalities per-frame with prob p (independently per modality).
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, m_tokens: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return m_tokens
        B, T, M, D = m_tokens.shape
        mask = torch.bernoulli(m_tokens.new_full((B, T, M, 1), 1 - self.p))  # 1 keep / 0 drop
        keep_counts = mask.sum(dim=2, keepdim=True).clamp_min(1.0)           # [B,T,1,1]
        return m_tokens * mask / keep_counts


# -----------------------------
# Per-frame fusion with per-modality temp & bias (+ attention log)
# -----------------------------
class MinimalPerFrameFusion(nn.Module):
    def __init__(self, d_model: int, n_modalities: int, p: float = 0.1):
        super().__init__()
        self.d = d_model
        self.m = n_modalities
        self.latent = nn.Parameter(torch.randn(1, 1, d_model))

        self.q_ln = nn.LayerNorm(d_model)
        self.kv_ln = nn.LayerNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        # per-modality temperature and bias (learned)
        self.logit_temp = nn.Parameter(torch.zeros(n_modalities))  # softplus -> >0
        self.logit_bias = nn.Parameter(torch.zeros(n_modalities))  # added to logits

        self.dropout = nn.Dropout(p)
        self.last_attn = None  # [BT, M] (with grad)

    def forward(self, M_tokens: torch.Tensor) -> torch.Tensor:
        B, T, M, D = M_tokens.shape
        kv = self.kv_ln(M_tokens).view(B * T, M, D)
        q  = self.q_ln(self.latent.expand(B * T, 1, D))            # [BT,1,D]

        Q = self.Wq(q)                      # [BT,1,D]
        K = self.Wk(kv)                     # [BT,M,D]
        V = self.Wv(kv)                     # [BT,M,D]

        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # [BT,1,M]
        tau = F.softplus(self.logit_temp) + 1e-3                      # [M]
        logits = logits / tau.view(1, 1, M)
        logits = logits + self.logit_bias.view(1, 1, M)

        A = logits.softmax(dim=-1)                                     # [BT,1,M]
        self.last_attn = A.squeeze(1)                                   # [BT,M], grad-carrying

        fused = torch.matmul(self.dropout(A), V).squeeze(1)            # [BT,D]
        fused = self.Wo(fused)
        return fused.view(B, T, D)


# ---------------------------------------------
# TemporalTransformerV2Plus (no resampling; no detach; no hard-coded init)
# ---------------------------------------------
class TemporalTransformerV2Plus(nn.Module):
    def __init__(self,
                 input_dim: int,                 # 1370 (RAW) or 2740 (RAW|MOTION)
                 d_model: int = 256,
                 latent_dim: int = 128,
                 time_layers: int = 4,
                 time_heads: int = 8,
                 dropout: float = 0.1,
                 modality_dropout_p: float = 0.15,
                 dims_map = {"vit":1024, "global":9, "pose":207, "beta":10, "kp2d":120}):
        super().__init__()
        self.modalities = ["vit", "global", "pose", "beta", "kp2d"]
        self.dim_map = dims_map
        self.one_pass = sum(self.dim_map[m] for m in self.modalities)
        self.has_diff = (input_dim >= 2 * self.one_pass)
        self.total_in = 2 * self.one_pass if self.has_diff else self.one_pass
        self.M = len(self.modalities)

        # per-modality encoders (state & motion), length-preserving
        def enc(d_in): return MovementConvEncoder(d_in, d_model, p=dropout)
        self.state_enc  = nn.ModuleDict({m: enc(self.dim_map[m]) for m in self.modalities})
        self.motion_enc = nn.ModuleDict({m: enc(self.dim_map[m]) for m in self.modalities}) if self.has_diff else None
        self.fuse_w = nn.ParameterDict({m: nn.Parameter(torch.tensor(0.0)) for m in self.modalities}) if self.has_diff else None

        # equalizer & modality dropout
        self.equalizer = ModalityEqualizer()

        # per-frame fusion
        self.fusion = MinimalPerFrameFusion(d_model, self.M, p=dropout)

        # temporal backbone (CLS adds one token)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc = SinusoidalPositionalEmbedding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, time_heads, 4 * d_model, dropout, batch_first=True)
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=time_layers)

        # heads
        self.seq_proj   = nn.Linear(d_model, latent_dim)
        self.frame_proj = nn.Linear(d_model, latent_dim)

        # logs
        self.last_attn = None
        self.last_losses = {}

    def _split_raw_diff(self, x):
        if self.has_diff:
            return x[:, :, :self.one_pass], x[:, :, self.one_pass:]
        return x, None

    def _split_modalities(self, part):
        if part is None:
            return {m: None for m in self.modalities}
        sizes = [self.dim_map[m] for m in self.modalities]
        chunks = torch.split(part, sizes, dim=-1)
        return dict(zip(self.modalities, chunks))

    def forward(self, x: torch.Tensor, lengths=None):
        """
        x: [B, T, total_in]  (RAW | MOTION features already geometry-aware)
        returns:
          seq_embed:    [B, latent_dim]
          frame_embeds: [B, T+1, latent_dim]  (CLS at index 0)
        """
        x = x[:, :, :self.total_in]
        B, T, _ = x.shape

        raw, diff = self._split_raw_diff(x)
        rawp  = self._split_modalities(raw)
        diffp = self._split_modalities(diff) if self.has_diff else {m: None for m in self.modalities}

        # encode each modality at full rate (length-preserving)
        per_mod = []
        for m in self.modalities:
            s = self.state_enc[m](rawp[m])                 # [B, T, D]
            if self.has_diff and diffp[m] is not None:
                u = self.motion_enc[m](diffp[m])           # [B, T, D]
                g = torch.sigmoid(self.fuse_w[m])          # scalar gate
                s = (1 - g) * s + g * u
            s = F.layer_norm(s, (s.size(-1),))             # per-modality LN
            per_mod.append(s.unsqueeze(2))                 # [B, T, 1, D]

        M_tokens = torch.cat(per_mod, dim=2)               # [B, T, M, D]

        # # if training
        # if self.training:   
        #     # equalize energy across modalities; dropout modalities (training only)
        #     M_tokens = self.equalizer(M_tokens)                # [B, T, M, D]

        # per-frame fusion & attention log
        frame_tok = self.fusion(M_tokens)                  # [B, T, D]
        self.last_attn = self.fusion.last_attn             # [BT, M] (with grad)

        # coverage regularizer (encourage using all modalities on average)
        if self.last_attn is not None:
            p = self.last_attn                                          # [BT, M]
            p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
            u = torch.full_like(p, 1.0 / p.size(-1))
            L_cov = (p.clamp_min(1e-8) * (p.clamp_min(1e-8).log() - u.log())).sum(dim=-1).mean()
        else:
            L_cov = torch.tensor(0.0, device=frame_tok.device)

        self.last_losses = {"coverage": L_cov}

        # temporal Transformer with CLS
        tokens = torch.cat([self.cls.expand(B, 1, self.cls.size(-1)), frame_tok], dim=1)  # [B, T+1, D]
        tokens = self.pos_enc(tokens)
        tokens = self.temporal(tokens)

        # heads (L2-normalized)
        cls_out = tokens[:, 0, :]
        seq_embed    = F.normalize(self.seq_proj(cls_out), dim=-1)   # [B, latent_dim]
        frame_embeds = F.normalize(self.frame_proj(tokens),  dim=-1) # [B, T+1, latent_dim]
        return seq_embed, frame_embeds, None
