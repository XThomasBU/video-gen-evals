import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        seq_len = 33
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
        return x_out, frame_embeddings


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
                nn.ReLU(),
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
        return x_out, frame_embeddings