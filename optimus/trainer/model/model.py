from __future__ import annotations

import abc
import typing as tp

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from optimus.trainer.script.cache import Cache

################################
### Import Optimized Modules ###
################################

try:
    import flash_attn

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

try:
    from liger_kernel.nn import (
        LigerCrossEntropyLoss,
        liger_rotary_pos_emb,
    )
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction

    LIGER_KERNEL_AVAILABLE = True
except ImportError:
    LIGER_KERNEL_AVAILABLE = False


#################################
### Transformer Encoder Model ###
#################################


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        blocks: tp.Iterable[Block],
        final_layernorm: nn.Module,
        lm_head: nn.Linear,
        fused_cross_entropy: bool = False,
    ):
        super().__init__()
        self.embedding = embedding
        self.blocks = nn.ModuleList(blocks)
        self.final_layernorm = final_layernorm
        self.lm_head = lm_head
        self.fused_cross_entropy = fused_cross_entropy

        if self.fused_cross_entropy:
            assert LIGER_KERNEL_AVAILABLE, "Liger kernel is not available."
            self.ligerCrossEntropy = LigerCrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: tp.Optional[torch.Tensor] = None,
        max_seqlen: tp.Optional[int] = None,
        labels: tp.Optional[torch.Tensor] = None,
        cache: tp.Optional[Cache] = None,
    ) -> tuple[tp.Optional[torch.Tensor], tp.Optional[torch.Tensor]]:
        # Model forward pass
        residuals = self.embedding(x)
        for block in self.blocks:
            residuals = block(
                residuals,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cache=cache,
            )
        h = self.final_layernorm(residuals)
        h = self.lm_head(h)

        # Model loss calculation
        if labels is None:
            return h, None
        elif self.fused_cross_entropy:
            with torch.autocast(device_type="cuda", enabled=False):
                return h, self.ligerCrossEntropy(
                    h.view(-1, h.size(-1)), labels.view(-1)
                )
        else:
            labels = labels.to(h.device)
            loss = F.cross_entropy(h.view(-1, h.size(-1)), labels.view(-1))
            return h, loss


####################################
### Transformer Embedding Module ###
####################################


class CustomEmbedding(torch.nn.Module):
    """This custom embedding class replaces the torch.nn.Embedding class, which has
    compilation issues."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(vocab_size, embed_dim))
        self.reset_parameters()

    def forward(self, input_ids):
        return self.weight[input_ids, :]

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)


##################################
### Transformer Encoder Module ###
##################################


class Block(nn.Module):
    def __init__(
        self,
        attention: SelfAttention,
        mlp: nn.Module,
        attn_norm: nn.Module,
        mlp_norm: nn.Module,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn_norm = attn_norm
        self.attn = attention
        self.attn_dropout = nn.Dropout(dropout)
        self.mlp = mlp
        self.mlp_norm = mlp_norm
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(
        self,
        residuals: torch.Tensor,
        *,
        cu_seqlens: tp.Optional[torch.Tensor] = None,
        max_seqlen: tp.Optional[int] = None,
        cache: tp.Optional[Cache] = None,
    ) -> torch.Tensor:
        h = self.attn_norm(residuals)
        h = self.attn(
            h,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cache=cache,
        )
        h = self.attn_dropout(h)
        residuals = residuals + h
        h = self.mlp_norm(residuals)
        h = self.mlp(h)
        h = self.mlp_dropout(h)
        residuals = residuals + h
        return residuals


###################################
### Attention Module Interfaces ###
###################################


class SelfAttention(nn.Module, abc.ABC):
    def __init__(
        self,
        *,
        embed_dim: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        block_size: int,
        rope: tp.Optional[RoPE],
        dropout: float,
        bias: bool,
        flash: str = "torch",
    ):
        super().__init__()
        self._validate_gqa_args(num_heads, num_kv_heads, dropout)
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.dropout = dropout
        self.rope = rope
        self.flash = flash
        if self.flash:
            assert FLASH_ATTN_AVAILABLE, "Flash attention is not installed"

        num_proj = num_heads + 2 * num_kv_heads
        self.qkv_proj = nn.Linear(embed_dim, num_proj * head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=bias)

    def _validate_gqa_args(self, num_heads: int, num_kv_heads: int, dropout: float = 0):
        assert (
            0 <= num_kv_heads <= num_heads
        ), f"num_kv_heads ({num_kv_heads}) must be between 0 and num_heads ({num_heads})"
        assert (
            num_heads % num_kv_heads == 0
        ), f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        assert 0 <= dropout < 1, "dropout must be in [0, 1)"

    def forward(
        self,
        h: torch.Tensor,
        *,
        cu_seqlens: tp.Optional[torch.Tensor] = None,
        max_seqlen: tp.Optional[int] = None,
        cache: tp.Optional[Cache] = None,
    ) -> torch.Tensor:
        """
        Computes the self-attention output.

        Args:
            h (torch.Tensor): Input tensor of shape
                [batch, seq_len, num_heads, head_dim] or [total, num_heads, head_dim].
            cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths tensor of shape [batch + 1].
            cache (Cache, optional): Dictionary containing precomputed cosine and sine values for RoPE.

        Returns:
            torch.Tensor: Output tensor of shape [batch, seq_len, embed_dim] or equivalent.
        """
        # Project input to Q, K, V
        qkv = self.qkv_proj(h)
        splits = [
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        ]
        q, k, v = torch.split(qkv, splits, dim=-1)

        # Rearrange tensors based on packed or non-packed input
        if cu_seqlens is not None:
            shape = "thd"
            q = einops.rearrange(q, "... (h d) -> ... h d", h=self.num_heads)
            k = einops.rearrange(k, "... (h d) -> ... h d", h=self.num_kv_heads)
            v = einops.rearrange(v, "... (h d) -> ... h d", h=self.num_kv_heads)
        elif self.flash:
            shape = "blhd"
            q = einops.rearrange(q, "b l (h d) -> b l h d", h=self.num_heads)
            k = einops.rearrange(k, "b l (h d) -> b l h d", h=self.num_kv_heads)
            v = einops.rearrange(v, "b l (h d) -> b l h d", h=self.num_kv_heads)
        else:
            shape = "bldh"
            q = einops.rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
            k = einops.rearrange(k, "b l (h d) -> b h l d", h=self.num_kv_heads)
            v = einops.rearrange(v, "b l (h d) -> b h l d", h=self.num_kv_heads)

        # Apply RoPE if enabled
        if self.rope:
            if cache is None:
                raise ValueError("Cache must be provided for RoPE")
            q = self.rope(q, cu_seqlens=cu_seqlens, shape=shape, cache=cache)
            k = self.rope(k, cu_seqlens=cu_seqlens, shape=shape, cache=cache)

        # Compute attention
        if cu_seqlens is not None:
            if self.flash:
                attn = flash_attn.flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    dropout_p=self.dropout,
                    causal=False,
                )
            else:
                attn = self._matmul_packed_sdpa(q, k, v, cu_seqlens)
        else:
            if self.flash:
                attn = flash_attn.flash_attn_func(
                    q, k, v, causal=False, dropout_p=self.dropout
                )
            else:
                k, v = self._maybe_repeat_keys_values(
                    k, v, self.num_heads, self.num_kv_heads
                )
                attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
                attn = einops.rearrange(attn, "b h l d -> b l h d")

        # Final projection
        attn = einops.rearrange(attn, "... h d -> ... (h d)")
        return self.out_proj(attn)

    def _matmul_packed_sdpa(self, q, k, v, cu_seqlens):
        """Compute scaled dot-product attention for packed sequences."""
        _, num_heads, head_dim = q.size()
        _, num_kv_heads, _ = k.size()

        # Expand key and value heads to match query heads
        k = einops.repeat(
            k, "n h d -> n (h expand) d", expand=num_heads // num_kv_heads
        )
        v = einops.repeat(
            v, "n h d -> n (h expand) d", expand=num_heads // num_kv_heads
        )

        # Compute scaled attention scores
        q = q / head_dim**0.5
        attn = torch.einsum("q h d, k h d -> h q k", q, k)

        # Apply non-causal mask
        mask = self._make_packed_seqs_non_causal_mask(cu_seqlens, device=q.device)
        attn += mask

        # Normalize attention scores
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).type_as(q)

        # Apply dropout if specified
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)

        # Compute weighted values
        return torch.einsum("h q k, k h d -> q h d", attn, v)

    def _make_packed_seqs_non_causal_mask(
        self, cu_seqlens: torch.Tensor, device: torch.device = torch.device("cpu")
    ):
        """
        Create a non-causal mask for packed sequences.

        Args:
            cu_seqlens (torch.Tensor): Cumulative sequence lengths tensor of shape [batch + 1].
            device (torch.device): Device for the mask tensor.

        Returns:
            torch.Tensor: Non-causal mask of shape [total_q_len, total_k_len].
        """
        total_len = cu_seqlens[-1]
        mask = torch.full((total_len, total_len), float("-inf"), device=device)

        for i in range(1, len(cu_seqlens)):
            q_start, q_end = cu_seqlens[i - 1], cu_seqlens[i]
            k_start, k_end = q_start, q_end
            mask[q_start:q_end, k_start:k_end] = 0

        return mask

    def _maybe_repeat_keys_values(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_heads == num_kv_heads:
            return k, v

        queries_per_kv_head = num_heads // num_kv_heads
        repeat_pattern = "b kv_head l head_dim -> b (kv_head repeat) l head_dim"
        k = einops.repeat(k, repeat_pattern, repeat=queries_per_kv_head)
        v = einops.repeat(v, repeat_pattern, repeat=queries_per_kv_head)
        return k, v


###################################
### Positional Encoding Modules ###
###################################


class RoPE(nn.Module):
    """RotaryPositionEmbeddings module according to https://arxiv.org/abs/2104.09864

    The RoPE paper proposes the fast implementation:
             [  x_1  ] [  cos(m Theta_1)  ]   [ -x_2  ] [  sin(m Theta_1)  ]
             [  x_2  ] [  cos(m Theta_1)  ]   [  x_1  ] [  sin(m Theta_1)  ]
             [  x_3  ] [  cos(m Theta_2)  ]   [ -x_4  ] [  sin(m Theta_2)  ]
    R_{m}x = [  x_4  ] [  cos(m Theta_2)  ] + [  x_3  ] [  sin(m Theta_2)  ]
             [  ...  ] [        ...       ]   [  ...  ] [        ...       ]
             [ x_d-1 ] [ cos(m Theta_d/2) ]   [ -x_d  ] [ sin(m Theta_d/2) ]
             [  x_d  ] [ cos(m Theta_d/2) ]   [ x_d-1 ] [ sin(m Theta_d/2) ]

    However, in order to take advantage of contiguous memory, we implement as:
             [   x_1   ] [  cos(m Theta_1)  ]   [ -x_d//2+1 ] [  sin(m Theta_1)  ]
             [   x_2   ] [  cos(m Theta_2)  ]   [ -x_d//2+2 ] [  sin(m Theta_2)  ]
             [   ...   ] [        ...       ]   [    ...    ] [        ...       ]
    R_{m}x = [  x_d/2  ] [ cos(m Theta_d/2) ] + [   -x_d    ] [ sin(m Theta_d/2) ]
             [ x_d/2+1 ] [  cos(m Theta_1)  ]   [    x_1    ] [  sin(m Theta_1)  ]
             [ x_d/2+2 ] [  cos(m Theta_2)  ]   [    x_2    ] [  sin(m Theta_2)  ]
             [   ...   ] [        ...       ]   [    ...    ] [        ...       ]
             [   x_d   ] [ cos(m Theta_d/2) ]   [   x_d//2  ] [ sin(m Theta_d/2) ]

    which can be reordered as
             [   x_1   ] [  cos(m Theta_1)  ]   [ -x_d//2+1 ] [  sin(m Theta_1)  ]
             [ x_d/2+1 ] [  cos(m Theta_1)  ]   [    x_1    ] [  sin(m Theta_1)  ]
             [   x_2   ] [  cos(m Theta_2)  ]   [ -x_d//2+2 ] [  sin(m Theta_2)  ]
    R_{m}x = [ x_d/2+2 ] [  cos(m Theta_2)  ] + [    x_2    ] [  sin(m Theta_2)  ]
             [   ...   ] [        ...       ]   [    ...    ] [        ...       ]
             [  x_d/2  ] [ cos(m Theta_d/2) ]   [   -x_d    ] [ sin(m Theta_d/2) ]
             [   x_d   ] [ cos(m Theta_d/2) ]   [   x_d//2  ] [ sin(m Theta_d/2) ]

    Since we will dot product the rotated queries and keys, the row order is irrelevant,
    meaning the second and third formulation are equivalent.
    Regarding the first formulation, RoPE supports any pairing so we just chose one that
    is more efficient to implement and get a similar effect.
    """

    def __init__(
        self,
        dim: int,
        base: float,
        block_size: int,
        fused_rope: bool = None,
    ):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.base = base
        self.block_size = block_size
        self.fused_rope = fused_rope

    # RoPE initialization
    def init_cache(
        self,
        cache: Cache,
        *,
        device: tp.Optional[torch.device] = None,
        dtype: tp.Optional[torch.dtype] = None,
    ):
        cache.maybe_init("rope", self._init_cos_sin, device=device)

    def _init_cos_sin(
        self,
        *,
        device: tp.Optional[torch.device] = None,
    ):
        idxs = torch.arange(0, self.dim, 2, device=device).float() / self.dim
        thetas = (1.0 / (self.base**idxs)).float()
        ms = torch.arange(self.block_size, dtype=torch.int64, device=device).float()

        ms_thetas = einops.einsum(ms, thetas, "m, d -> m d")
        ms_thetas = einops.repeat(ms_thetas, "m d -> m (2 d)")

        cos = ms_thetas.cos().to(device=device)
        sin = ms_thetas.sin().to(device=device)
        return cos, sin

    # RoPE forward pass
    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: tp.Optional[torch.Tensor] = None,
        shape: tp.Optional[str] = None,
        cache: tp.Mapping[str, tp.Any],
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape [batch, seq_len, num_heads, head_dim] or
            [total, num_heads, head_dim] if cu_seqlens is not None
            cu_seqlens: cumulative sequence lengths tensor of shape [batch + 1]
            shape: shape of the input tensor. str['thd', 'blhd', 'bldh']
            cache: dictionary containing the cosine and sine values for the RoPE
        """
        input_type = x.dtype
        x = x.float()

        cos, sin = cache.get("rope")
        if shape == "thd":
            x = self._apply_rope_thd(x, cu_seqlens, cos, sin)
        elif shape == "blhd":
            x = self._apply_rope_blhd(x, cos, sin)
        else:
            if self.fused_rope:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
                x, _ = liger_rotary_pos_emb(x, x, cos, sin)
            else:
                x = self._apply_rope_bhld(x, cos, sin)

        return x.to(dtype=input_type)

    def _apply_rope_thd(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rope to x where the shape is [batch x num_seqs, num_heads, head_dim]
        """
        dim = cos.size(-1)

        x, x_pass = x[..., :dim], x[..., dim:]
        high, low = einops.rearrange(x, "... (split d) -> split ... d", split=2)
        rot = einops.rearrange([-low, high], "merge ... d -> ... (merge d)", merge=2)

        pos_ids = self._cu_seqlens_to_pos_ids(cu_seqlens)

        cos = einops.rearrange(cos, "m d -> m 1 d")
        sin = einops.rearrange(sin, "m d -> m 1 d")

        x_rope = (x * cos[pos_ids] + rot * sin[pos_ids]).to(dtype=x.dtype)
        return torch.cat((x_rope, x_pass), dim=-1)

    def _apply_rope_bhld(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rope to x where the shape is [batch, num_heads, seq_len, head_dim]
        """
        dim = cos.size(-1)

        x, x_pass = x[..., :dim], x[..., dim:]
        high, low = einops.rearrange(x, "b h l (split d) -> split b h l d", split=2)
        rot = einops.rearrange(
            [-low, high], "merge b h l d -> b h l (merge d)", merge=2
        )

        seq_len = x.size(2)
        x_rope = (x * cos[:seq_len] + rot * sin[:seq_len]).to(dtype=x.dtype)
        return torch.cat((x_rope, x_pass), dim=-1)

    def _apply_rope_blhd(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rope to x where the shape is [batch, seq_len, num_heads, head_dim]
        """
        dim = cos.size(-1)

        x, x_pass = x[..., :dim], x[..., dim:]
        high, low = einops.rearrange(x, "b l h (split d) -> split b l h d", split=2)
        rot = einops.rearrange(
            [-low, high], "merge b l h d -> b l h (merge d)", merge=2
        )

        cos = einops.rearrange(cos, "m d -> m 1 d")
        sin = einops.rearrange(sin, "m d -> m 1 d")

        seq_len = x.size(1)
        x_rope = (x * cos[:seq_len] + rot * sin[:seq_len]).to(dtype=x.dtype)
        return torch.cat((x_rope, x_pass), dim=-1)

    @torch.compiler.disable
    def _cu_seqlens_to_pos_ids(self, cu_seqlens: torch.Tensor) -> torch.Tensor:
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        return torch.cat([torch.arange(seq_len) for seq_len in seq_lens])


#####################################
### Feedforward Module Interfaces ###
#####################################


class GeluMLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        bias: bool,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.proj = nn.Linear(hidden_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.proj(x)


class SwigluMLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
        bias: bool,
        fused_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.fc_2 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.fused_swiglu = fused_swiglu
        if fused_swiglu:
            assert not bias, "Fused SwiGLU does not support bias"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        if self.fused_swiglu:
            assert LIGER_KERNEL_AVAILABLE, "Liger kernel is not available"
            x = LigerSiLUMulFunction.apply(x_fc_1, x_fc_2)
        else:
            x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        x = self.dropout(x)
        return self.proj(x)


######################################
### Layer Normalization Interfaces ###
######################################


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, embed_dim: int, eps: float, dim: int = -1) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embed_dim))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed.to(x_dtype)
