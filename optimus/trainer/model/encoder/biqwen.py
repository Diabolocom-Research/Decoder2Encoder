from typing import Optional

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_layers import (
    GradientCheckpointingLayer,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

try:
    from liger_kernel.transformers import LigerCrossEntropyLoss
    LIGER_KERNEL_AVAILABLE = True
except ImportError:
    LIGER_KERNEL_AVAILABLE = False


class CustomEmbedding(torch.nn.Module):
    """This custom embedding class replaces the torch.nn.Embedding class, which has
    compilation issues."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(vocab_size, embed_dim))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        return self.weight[input_ids, :]


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, None, :, :].expand(num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(num_key_value_heads * n_rep, slen, head_dim)

def create_packed_seqs_mask(
    cu_seqlens: torch.Tensor,
    causal: bool = True,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create a causal or non-causal attention mask for packed sequences.

    Args:
        cu_seqlens (torch.Tensor): Cumulative sequence lengths of shape [batch + 1].
        is_causal (bool): If True, create a causal (lower triangular) mask within
            each sequence. If False, a full attention mask is created within each sequence.
        device (torch.device): Target device for the mask.

    Returns:
        torch.Tensor: Attention mask of shape [total_len, total_len] with 0.0 (allowed)
            and -inf (masked).
    """
    total_len = cu_seqlens[-1].item()
    seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)

    seq_indices = torch.repeat_interleave(
        torch.arange(len(seq_lengths), device=device),
        seq_lengths
    )

    seq_mask = seq_indices.unsqueeze(0) == seq_indices.unsqueeze(1)

    if causal:
        causal_mask = torch.tril(torch.ones(total_len, total_len, device=device, dtype=torch.bool))
        combined_mask = seq_mask & causal_mask
    else:
        combined_mask = seq_mask

    attention_mask = torch.full((total_len, total_len), float('-inf'), device=device)
    attention_mask.masked_fill_(combined_mask, 0.0)

    return attention_mask

def sdpa_attention_forward(
        q, k, v, 
        cu_seqlens, 
        scaling, 
        dropout: float = 0.0, 
        causal: bool = True
    ):
    """Compute scaled dot-product attention for packed sequences."""
    attn_weights = torch.matmul(q, k.transpose(1, 2)) * scaling

    mask = create_packed_seqs_mask(cu_seqlens, causal, q.device)
    attn_weights = attn_weights + mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous()

    return attn_output, attn_weights

class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: Optional[int],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(0, 1)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(0, 1)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(0, 1)

        query_states, key_states = query_states.unsqueeze(0), key_states.unsqueeze(0),
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states, key_states = query_states.squeeze(0), key_states.squeeze(0),
            
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.config._attn_implementation == "flash_attention_2":
            attn_output = flash_attn.flash_attn_varlen_func(
                query_states.transpose(0, 1),
                key_states.transpose(0, 1),
                value_states.transpose(0, 1),
                cu_seqlens,
                cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=self.scaling,
                causal=False,
            ).contiguous()
        else:
            attn_output, _ = sdpa_attention_forward(
                query_states,
                key_states,
                value_states,
                cu_seqlens=cu_seqlens,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling,
                causal=False,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class Qwen3EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,

    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_record_outputs = {}


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = CustomEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.mask_converter = AttentionMaskConverter(True)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(len(input_ids), device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
            )

        return self.norm(hidden_states)


class Qwen3ForMaskedLM(Qwen3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, fused_cross_entropy=False):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.fused_cross_entropy = fused_cross_entropy
        if self.fused_cross_entropy:
            assert LIGER_KERNEL_AVAILABLE, "Liger kernel is not available."
            self.ligerCrossEntropy = LigerCrossEntropyLoss()

        self.post_init()

    def forward(
        self,
        x: torch.LongTensor,
        *,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        hidden_states = self.model(
            input_ids=x,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if self.fused_cross_entropy:
                loss = self.ligerCrossEntropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
            else:
                loss = self.loss_function(
                    logits, labels, vocab_size=self.config.vocab_size
                )

        return logits, loss


__all__ = [
    "Qwen3PreTrainedModel",
    "Qwen3Model",
    "Qwen3ForMaskedLM",
]