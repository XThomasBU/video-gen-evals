import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import typing as T


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

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

    def forward(self, x):
        res = x
        y = self.act1(self.conv1(x))
        y = self.drop1(y)
        y = self.conv2(y)
        y = self.act2(y + res)
        return self.norm(y)


class MovementConvEncoder(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.1, k: int = 5, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.stem = nn.Conv1d(d_in, d_out, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList([
            TemporalConvBlock(d_out, kernel_size=k, dilation=d, p=p) for d in dilations
        ])
        self.proj = nn.Linear(d_out, d_out, bias=False)

    def forward(self, x_btf: torch.Tensor) -> torch.Tensor:
        x = x_btf.transpose(1, 2)
        y = self.stem(x)
        for blk in self.blocks:
            y = blk(y)
        y = y.transpose(1, 2)
        return self.proj(y)


class MinimalPerFrameFusion(nn.Module):
    def __init__(self, d_model: int, n_modalities: int, p: float = 0.1):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(1, 1, d_model))

        self.q_ln = nn.LayerNorm(d_model)
        self.kv_ln = nn.LayerNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        self.logit_temp = nn.Parameter(torch.zeros(n_modalities))
        self.logit_bias = nn.Parameter(torch.zeros(n_modalities))

        self.dropout = nn.Dropout(p)
        self.last_attn = None

    def forward(self, M_tokens: torch.Tensor, mask=None) -> torch.Tensor:
        B, T, M, D = M_tokens.shape
        kv = self.kv_ln(M_tokens).view(B * T, M, D)
        q  = self.q_ln(self.latent.expand(B * T, 1, D))

        Q = self.Wq(q)
        K = self.Wk(kv)
        V = self.Wv(kv)

        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        tau = F.softplus(self.logit_temp) + 1e-3
        logits = logits / tau.view(1, 1, M)
        logits = logits + self.logit_bias.view(1, 1, M)

        A = logits.softmax(dim=-1)
        self.last_attn = A.squeeze(1)

        fused = torch.matmul(self.dropout(A), V).squeeze(1)
        fused = self.Wo(fused)
        return fused.view(B, T, D)



class HumanActionScorer(nn.Module):
    def __init__(self,
                 dims_map_raw: T.Dict[str, int],
                 dims_map_diff: T.Dict[str, int],
                 d_model: int = 256,
                 latent_dim: int = 128,
                 time_layers: int = 4,
                 time_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        if not isinstance(dims_map_raw, dict) or not isinstance(dims_map_diff, dict):
            raise ValueError("dims_map_raw and dims_map_diff must be dicts of {modality_name: dim}.")

        if set(dims_map_raw.keys()) != set(dims_map_diff.keys()):
            raise ValueError("dims_map_raw and dims_map_diff must have the same modality keys.")
        self.modalities = list(dims_map_raw.keys())

        self.dim_map_raw  = {m: int(dims_map_raw[m])  for m in self.modalities}
        self.dim_map_diff = {m: int(dims_map_diff[m]) for m in self.modalities}

        self.one_pass_raw  = sum(self.dim_map_raw[m]  for m in self.modalities)
        self.one_pass_diff = sum(self.dim_map_diff[m] for m in self.modalities)
        self.has_diff = any(self.dim_map_diff[m] > 0 for m in self.modalities)

        def enc(d_in):
            return MovementConvEncoder(d_in, d_model, p=dropout)

        self.state_enc = nn.ModuleDict({m: enc(self.dim_map_raw[m]) for m in self.modalities})

        if self.has_diff:
            self.motion_enc = nn.ModuleDict({
                m: enc(self.dim_map_diff[m]) for m in self.modalities if self.dim_map_diff[m] > 0
            })
        else:
            self.motion_enc = None

        self.M = len(self.modalities)

        self.fusion = MinimalPerFrameFusion(d_model, self.M, p=dropout)

        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc = SinusoidalPositionalEmbedding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, time_heads, 4 * d_model, dropout, batch_first=True)
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=time_layers)

        self.last_attn = None

    def _split_raw_diff(self, x: torch.Tensor) -> T.Tuple[torch.Tensor, T.Optional[torch.Tensor]]:
        if self.has_diff:
            raw = x[:, :, :self.one_pass_raw]
            diff = x[:, :, self.one_pass_raw:self.one_pass_raw + self.one_pass_diff]
            return raw, diff
        return x, None

    def _split_modalities(self, part: torch.Tensor, dim_map: T.Dict[str, int]) -> T.Dict[str, torch.Tensor]:
        sizes = [dim_map[m] for m in self.modalities]
        chunks = torch.split(part, sizes, dim=-1)
        return dict(zip(self.modalities, chunks))

    def forward(self, x: torch.Tensor, modality_mask = None):
        B, T, D = x.shape

        raw, diff = self._split_raw_diff(x)
        rawp  = self._split_modalities(raw,  self.dim_map_raw)
        diffp = self._split_modalities(diff, self.dim_map_diff) if self.has_diff else {m: None for m in self.modalities}

        per_mod = []
        for m in self.modalities:
            s = self.state_enc[m](rawp[m])
            if self.has_diff and self.dim_map_diff[m] > 0:
                u = self.motion_enc[m](diffp[m])
                s = s + u
            s = F.layer_norm(s, (s.size(-1),))
            per_mod.append(s.unsqueeze(2))

        M_tokens = torch.cat(per_mod, dim=2)

        fusion_mask = None
        if modality_mask is not None:
            fusion_mask = modality_mask.view(1, 1, self.M).expand(B, T, self.M).to(M_tokens.device)

        frame_tok = self.fusion(M_tokens, mask=fusion_mask)
        self.last_attn = self.fusion.last_attn

        tokens = torch.cat([self.cls.expand(B, 1, self.cls.size(-1)), frame_tok], dim=1)
        tokens = self.pos_enc(tokens)
        tokens = self.temporal(tokens)
        cls_out = tokens[:, 0, :]
        seq_embed    = F.normalize(cls_out)
        frame_embeds = F.normalize(tokens,  dim=-1)
        return seq_embed, frame_embeds, tokens
