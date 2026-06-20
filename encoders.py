"""Swappable building blocks for the binding model, selected by name from config.

Three registries let you drop in alternative components without touching the
training code:

  DNA_ENCODERS       one-hot DNA [B,4,L] -> [B, out_dim]
  PROTEIN_ENCODERS   cached protein vector [B,D] -> [B, out_dim]
  INTERACTIONS       (prot_emb, dna_emb) -> affinity scalar [B]

Add a new encoder by subclassing the relevant base and decorating it with
@register_*("name"); then set that name in config.yaml. `build_*` reads a spec
dict {"name": ..., <kwargs>} and constructs the module.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# registries
# ---------------------------------------------------------------------------
DNA_ENCODERS: dict[str, type] = {}
PROTEIN_ENCODERS: dict[str, type] = {}
INTERACTIONS: dict[str, type] = {}


def _register(reg):
    def deco(name):
        def inner(cls):
            reg[name] = cls
            return cls
        return inner
    return deco


register_dna_encoder = _register(DNA_ENCODERS)
register_protein_encoder = _register(PROTEIN_ENCODERS)
register_interaction = _register(INTERACTIONS)


def _build(reg, spec: dict, **injected):
    """Instantiate reg[spec['name']] with injected kwargs + the spec keys that the
    target constructor actually accepts.

    Spec keys the chosen component doesn't take are dropped, so a single config
    block (e.g. the `interaction` block) can carry the union of options' params
    and you can swap components by changing `name` alone. The flip side: a typo'd
    config key is silently ignored rather than raising.
    """
    spec = dict(spec)
    name = spec.pop("name")
    if name not in reg:
        raise KeyError(f"unknown component {name!r}; available: {sorted(reg)}")
    cls = reg[name]
    accepted = inspect.signature(cls).parameters
    kwargs = {k: v for k, v in spec.items() if k in accepted}
    return cls(**injected, **kwargs)


def build_dna_encoder(spec: dict, seq_len: int) -> "BaseDNAEncoder":
    return _build(DNA_ENCODERS, spec, seq_len=seq_len)


def build_protein_encoder(spec: dict, in_dim: int) -> "BaseProteinEncoder":
    return _build(PROTEIN_ENCODERS, spec, in_dim=in_dim)


def build_interaction(spec: dict, prot_dim: int, dna_dim: int) -> nn.Module:
    return _build(INTERACTIONS, spec, prot_dim=prot_dim, dna_dim=dna_dim)


# ---------------------------------------------------------------------------
# DNA encoders: one-hot [B, 4, L] -> [B, out_dim]
# ---------------------------------------------------------------------------
class BaseDNAEncoder(nn.Module):
    out_dim: int


@register_dna_encoder("cnn")
class CNNEncoder(BaseDNAEncoder):
    """DeepBind/DeepSEA-style conv stack with global max+avg pooling."""

    def __init__(self, seq_len: int, channels: int = 128, out_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.conv = nn.Sequential(
            nn.Conv1d(4, channels, kernel_size=8, padding="same"),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding="same"),
            nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding="same"),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.project = nn.Sequential(nn.Linear(2 * channels, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        pooled = torch.cat([h.amax(dim=-1), h.mean(dim=-1)], dim=-1)
        return self.project(pooled)


@register_dna_encoder("cnn_attn")
class CNNAttnEncoder(BaseDNAEncoder):
    """Same conv stack as `cnn`, but the uniform average-pool branch is replaced
    by a learned multi-head *attentive pool* over sequence positions.

    A 1x1 conv scores every position per head; softmax over the length axis turns
    those scores into attention weights, and each head returns its weighted sum of
    conv features. The global max-pool branch is kept identical to `cnn`, so the
    only thing that changes versus `cnn` is avg-pool -> attentive-pool. That makes
    the A/B comparison a clean test of whether *learned* position weighting beats
    uniform averaging -- i.e. whether a full transformer (self-attention across
    positions) is likely worth the extra engineering.

    pooled = concat[ global_max(h) , head_1, ..., head_H ]  -> Linear -> LayerNorm
    """

    def __init__(self, seq_len: int, channels: int = 128, out_dim: int = 256,
                 dropout: float = 0.2, attn_heads: int = 4):
        super().__init__()
        self.out_dim = out_dim
        self.attn_heads = attn_heads
        self.conv = nn.Sequential(
            nn.Conv1d(4, channels, kernel_size=8, padding="same"),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding="same"),
            nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding="same"),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        # per-position, per-head attention logits over the length axis
        self.attn = nn.Conv1d(channels, attn_heads, kernel_size=1)
        # max branch (1*channels) + attentive branch (attn_heads*channels)
        self.project = nn.Sequential(
            nn.Linear((1 + attn_heads) * channels, out_dim), nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)                                   # [B, C, L]
        w = torch.softmax(self.attn(h), dim=-1)            # [B, H, L] (over positions)
        attn_pooled = torch.bmm(w, h.transpose(1, 2))      # [B, H, C]
        attn_pooled = attn_pooled.reshape(h.shape[0], -1)  # [B, H*C]
        pooled = torch.cat([h.amax(dim=-1), attn_pooled], dim=-1)
        return self.project(pooled)


@register_dna_encoder("cnn_deep")
class CNNDeepEncoder(BaseDNAEncoder):
    """Deeper residual conv stack + multi-head attentive pool + global max pool.

    Strengthens the DNA tower over `cnn`/`cnn_attn`:
      - 4 conv blocks (kernel 11->7->5->3) with BatchNorm/GELU and inter-block
        residual connections (cheap to add since all blocks share `channels`).
      - pooling concatenates global max-pool with an H-head attentive pool, so
        the protein-independent probe embedding captures both the single strongest
        motif hit (max) and a learned weighted summary over positions (attention).
    The DNA side has lots of signal (2000 probes x many proteins), so extra
    capacity here is low overfitting risk -- unlike the protein side (320 TFs).
    """

    def __init__(self, seq_len: int, channels: int = 256, out_dim: int = 256,
                 dropout: float = 0.2, attn_heads: int = 4):
        super().__init__()
        self.out_dim = out_dim
        self.attn_heads = attn_heads
        self.stem = nn.Sequential(
            nn.Conv1d(4, channels, kernel_size=11, padding="same"),
            nn.BatchNorm1d(channels), nn.GELU(),
        )
        # residual blocks (same width -> can add input back)
        self.block1 = self._block(channels, 7, dropout)
        self.block2 = self._block(channels, 5, dropout)
        self.block3 = self._block(channels, 3, dropout)
        self.attn = nn.Conv1d(channels, attn_heads, kernel_size=1)
        self.project = nn.Sequential(
            nn.Linear((1 + attn_heads) * channels, out_dim), nn.LayerNorm(out_dim)
        )

    @staticmethod
    def _block(channels: int, kernel: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel, padding="same"),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        h = h + self.block3(h)
        w = torch.softmax(self.attn(h), dim=-1)            # [B, H, L]
        attn_pooled = torch.bmm(w, h.transpose(1, 2)).reshape(h.shape[0], -1)
        pooled = torch.cat([h.amax(dim=-1), attn_pooled], dim=-1)
        return self.project(pooled)


@register_dna_encoder("cnn_deep_pos")
class CNNDeepPosEncoder(BaseDNAEncoder):
    """Same residual conv trunk as `cnn_deep`, but returns the *per-position*
    feature map `[B, L, out_dim]` instead of a pooled vector. The cross-attention
    interaction head needs every position (candidate motif site) as keys/values,
    so we keep the length axis and only project channels -> out_dim per position.
    """

    def __init__(self, seq_len: int, channels: int = 256, out_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.stem = nn.Sequential(
            nn.Conv1d(4, channels, kernel_size=11, padding="same"),
            nn.BatchNorm1d(channels), nn.GELU(),
        )
        self.block1 = CNNDeepEncoder._block(channels, 7, dropout)
        self.block2 = CNNDeepEncoder._block(channels, 5, dropout)
        self.block3 = CNNDeepEncoder._block(channels, 3, dropout)
        self.proj = nn.Conv1d(channels, out_dim, kernel_size=1)  # per-position
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        h = h + self.block3(h)
        h = self.proj(h).transpose(1, 2)   # [B, L, out_dim]
        return self.norm(h)

@register_dna_encoder("cnn_deep_pos_dnafeat")
class CNNDeepPosDNAFeatEncoder(BaseDNAEncoder):
    """CNNDeepPosEncoder with simple DNA biochemical feature channels.

    Input is the regular one-hot DNA tensor [B, 4, L] ordered as A,C,G,T.
    We append cheap biochemical channels:
      GC, AT, purine, pyrimidine, amino, keto.
    """

    def __init__(self, seq_len: int, out_dim: int = 128, channels: int = 256, dropout: float = 0.2):
        super().__init__()
        self.seq_len = seq_len
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Conv1d(10, channels, kernel_size=11, padding="same"),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(channels, channels, kernel_size=7, padding="same"),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(channels, out_dim, kernel_size=5, padding="same"),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, out_dim))
        nn.init.normal_(self.pos_emb, std=0.02)

    @staticmethod
    def _dna_features(x: torch.Tensor) -> torch.Tensor:
        # x: [B, 4, L], order A,C,G,T
        a = x[:, 0:1, :]
        c = x[:, 1:2, :]
        g = x[:, 2:3, :]
        t = x[:, 3:4, :]

        gc = g + c
        at = a + t

        # Purines are larger two-ring bases: A/G.
        purine = a + g

        # Pyrimidines are smaller one-ring bases: C/T.
        pyrimidine = c + t

        # Amino bases: A/C. Keto bases: G/T.
        amino = a + c
        keto = g + t

        return torch.cat([x, gc, at, purine, pyrimidine, amino, keto], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._dna_features(x)
        h = self.net(x).transpose(1, 2)  # [B, L, out_dim]
        return h + self.pos_emb[:, : h.size(1), :]

@register_dna_encoder("rnn")
class BiLSTMEncoder(BaseDNAEncoder):
    """Bidirectional LSTM over the DNA sequence; mean-pools hidden states."""

    def __init__(self, seq_len: int, hidden: int = 128, num_layers: int = 1,
                 out_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.lstm = nn.LSTM(
            input_size=4, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.project = nn.Sequential(nn.Linear(2 * hidden, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x.transpose(1, 2))  # [B, L, 2*hidden]
        return self.project(h.mean(dim=1))


# ---------------------------------------------------------------------------
# Protein encoders: cached vector [B, D] -> [B, out_dim]
# ---------------------------------------------------------------------------
class BaseProteinEncoder(nn.Module):
    out_dim: int


@register_protein_encoder("mlp")
class MLPProteinEncoder(BaseProteinEncoder):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_protein_encoder("mlp_res")
class ResMLPProteinEncoder(BaseProteinEncoder):
    """Project to `hidden`, then `n_blocks` pre-norm residual MLP blocks, then to
    `out_dim`. Deeper than `mlp` but residual + dropout keep it trainable on the
    small (320-protein) train set; this is the swappable protein head."""

    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 768,
                 dropout: float = 0.3, n_blocks: int = 2):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, hidden),
            )
            for _ in range(n_blocks)
        )
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


@register_protein_encoder("attn_pool_tokens")
class AttnPoolTokensEncoder(BaseProteinEncoder):
    """Per-residue ESM reps [B, R, in_dim] (+ valid mask [B, R]) -> K "specificity"
    query tokens [B, K, out_dim] via learned multi-query attention pooling.

    Instead of mean-pooling the whole protein (which dilutes the DNA-binding
    domain), K learnable queries each attend over the residues and pull out a
    distinct pooled summary. Those K tokens are what the cross-attention head
    later scans the DNA with -- i.e. a small, learned, per-TF set of motif probes.
    Masking ignores padded residues, so variable-length proteins share one tensor.
    """

    def __init__(self, in_dim: int, out_dim: int = 128, n_tokens: int = 8,
                 hidden: int = 512, attn_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.out_dim = out_dim
        self.n_tokens = n_tokens
        self.attn_dim = attn_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.key = nn.Linear(hidden, attn_dim)
        self.val = nn.Linear(hidden, out_dim)
        self.queries = nn.Parameter(torch.randn(n_tokens, attn_dim) * (attn_dim ** -0.5))
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, R, in_dim]; mask: [B, R] bool (True = real residue)
        h = self.proj(x)                                  # [B, R, hidden]
        k = self.key(h)                                   # [B, R, attn_dim]
        v = self.val(h)                                   # [B, R, out_dim]
        logits = torch.einsum("ka,bra->bkr", self.queries, k) * (self.attn_dim ** -0.5)
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))  # [B,K,R]
        attn = torch.softmax(logits, dim=-1)
        tokens = torch.einsum("bkr,brv->bkv", attn, v)    # [B, K, out_dim]
        return self.out_norm(tokens)

@register_protein_encoder("attn_pool_tokens_physbias")
class AttnPoolTokensPhysBiasEncoder(AttnPoolTokensEncoder):
    """AttnPoolTokensEncoder with a small biochemical bias toward DNA-binding residues.

    Assumes that when physchem v1 is used, the last 6 protein features are:
      charge, hydrophobicity, polarity, aromatic, positive, negative.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 128,
        n_tokens: int = 8,
        hidden: int = 512,
        attn_dim: int = 128,
        dropout: float = 0.3,
        phys_bias_init: float = 0.10,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            n_tokens=n_tokens,
            hidden=hidden,
            attn_dim=attn_dim,
            dropout=dropout,
        )
        self.phys_bias_scale = nn.Parameter(torch.tensor(float(phys_bias_init)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, R, in_dim]; mask: [B, R] bool
        h = self.proj(x)
        k = self.key(h)
        v = self.val(h)

        logits = torch.einsum("ka,bra->bkr", self.queries, k) * (self.attn_dim ** -0.5)

        # Last 6 dims in the physchem v1 cache:
        # charge, hydrophobicity, polarity, aromatic, positive, negative.
        if x.size(-1) >= 1286:
            phys = x[..., -6:].to(dtype=logits.dtype)
            charge = phys[..., 0]
            positive = phys[..., 4]
            negative = phys[..., 5]

            # Higher for K/R/H-like positive residues; lower for D/E-like negative residues.
            basic_score = charge 

            logits = logits + self.phys_bias_scale * basic_score.unsqueeze(1)

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        tokens = torch.einsum("bkr,brv->bkv", attn, v)
        return self.out_norm(tokens)
    
@register_protein_encoder("linear")
class LinearProteinEncoder(BaseProteinEncoder):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_protein_encoder("identity")
class IdentityProteinEncoder(BaseProteinEncoder):
    """Passthrough; lets the interaction head consume raw embeddings directly."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# Interactions: (prot_emb, dna_emb) -> affinity scalar [B]
# ---------------------------------------------------------------------------
@register_interaction("concat_hadamard")
class ConcatHadamardHead(nn.Module):
    """Project both towers to a shared space; fuse concat[p, d, p*d] -> MLP -> 1."""

    def __init__(self, prot_dim: int, dna_dim: int, emb_dim: int = 256,
                 hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.p_proj = nn.Linear(prot_dim, emb_dim)
        self.d_proj = nn.Linear(dna_dim, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(3 * emb_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        p, d = self.p_proj(p), self.d_proj(d)
        return self.head(torch.cat([p, d, p * d], dim=-1)).squeeze(-1)


@register_interaction("cross_attn")
class CrossAttnHead(nn.Module):
    """Motif-match cross-attention: a TF's K query tokens scan the DNA positions.

    Protein tokens p [.., K, Dp] and DNA per-position features d [.., L, Dd] are
    projected to a shared `attn_dim`; the match score of token k at position l is
    their scaled dot product. Per token we summarise the scan over positions by
    [max, mean, std] -- max captures "this motif occurs *somewhere* in the probe",
    mean/std its overall presence/sharpness. A small per-token MLP embeds those
    stats; we pool over tokens (mean+max, so the head is agnostic to K) and read
    out a scalar affinity. This models binding as a learned, per-TF motif scan
    rather than a dot product of two globally pooled vectors.

    `score_grid` computes the full probe x protein grid in one batched einsum
    (the trainer's outer-product step), avoiding any [Bn*P, ...] materialisation.
    """

    def __init__(self, prot_dim: int, dna_dim: int, attn_dim: int = 64,
                 tok_embed: int = 32, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.scale = attn_dim ** -0.5
        self.p_proj = nn.Linear(prot_dim, attn_dim)
        self.d_proj = nn.Linear(dna_dim, attn_dim)
        self.tok_mlp = nn.Sequential(
            nn.Linear(3, tok_embed), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(tok_embed, tok_embed),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * tok_embed), nn.Linear(2 * tok_embed, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    @staticmethod
    def _pos_stats(s: torch.Tensor) -> torch.Tensor:
        """Summarise scan scores over the position axis (last dim) -> [..., 3]."""
        return torch.stack([s.amax(-1), s.mean(-1), s.std(-1)], dim=-1)

    def _readout(self, stats: torch.Tensor) -> torch.Tensor:
        """stats [..., K, 3] -> scalar [...]. Pool over tokens with mean+max."""
        e = self.tok_mlp(stats)                                   # [..., K, tok_embed]
        agg = torch.cat([e.mean(dim=-2), e.amax(dim=-2)], dim=-1)  # [..., 2*tok_embed]
        return self.head(agg).squeeze(-1)

    def score_grid(self, p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """p [P, K, Dp], d [Bn, L, Dd] -> affinity grid [Bn, P]."""
        pa = self.p_proj(p)                       # [P, K, A]
        da = self.d_proj(d)                       # [Bn, L, A]
        s = torch.einsum("pka,bla->bpkl", pa, da) * self.scale   # [Bn, P, K, L]
        return self._readout(self._pos_stats(s))                 # [Bn, P]

    def forward(self, p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # Paired path (also lets the generic model.py smoke test feed 2-D inputs).
        if p.dim() == 2:
            p = p.unsqueeze(1)   # [N, 1, Dp]
        if d.dim() == 2:
            d = d.unsqueeze(1)   # [N, 1, Dd]
        pa, da = self.p_proj(p), self.d_proj(d)                  # [N,K,A],[N,L,A]
        s = torch.einsum("nka,nla->nkl", pa, da) * self.scale     # [N, K, L]
        return self._readout(self._pos_stats(s))                  # [N]


@register_interaction("concat")
class ConcatHead(nn.Module):
    """Plain concatenation of the two towers -> MLP -> 1 (no explicit product)."""

    def __init__(self, prot_dim: int, dna_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(prot_dim + dna_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([p, d], dim=-1)).squeeze(-1)


@register_interaction("bilinear")
class BilinearHead(nn.Module):
    """Low-rank bilinear interaction score: p^T W d via shared projection."""

    def __init__(self, prot_dim: int, dna_dim: int, rank: int = 256, dropout: float = 0.3):
        super().__init__()
        self.p_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(prot_dim, rank))
        self.d_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(dna_dim, rank))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return (self.p_proj(p) * self.d_proj(d)).sum(dim=-1) + self.bias
