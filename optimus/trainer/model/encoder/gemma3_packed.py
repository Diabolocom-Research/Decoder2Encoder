# coding=utf-8
# Packed sequence version of Gemma3 for efficient training with variable-length sequences
# Based on gemma3.py with modifications to support cu_seqlens packed format

from typing import Optional, Callable
import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

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


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Gemma3TextConfig, device=None, layer_type=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.layer_types = list(set(config.layer_types))
        self.rope_type = {}
        for layer_type in self.layer_types:
            # Compatibility fix: Support both config formats
            # - GitHub transformers main branch uses: config.rope_parameters[layer_type]
            # - PyPI transformers 4.57.1 and pretrained models use: config.rope_theta directly
            # This ensures the model can load pretrained weights from HuggingFace Hub
            if hasattr(self.config, 'rope_parameters') and self.config.rope_parameters is not None:
                rope_params = self.config.rope_parameters.get(layer_type)
                if rope_params is None:
                    continue
                self.rope_type[layer_type] = rope_params.get("rope_type", "default")
            else:
                # Pretrained models use rope_theta directly (no rope_parameters dict)
                self.rope_type[layer_type] = "default"

            rope_init_fn: Callable = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, device, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
            setattr(self, f"{layer_type}_original_inv_freq", curr_inv_freq)
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Gemma3TextConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
        layer_type: Optional[str] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PretrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        # Compatibility fix: Support both config formats
        # - GitHub transformers main branch uses: config.rope_parameters[layer_type]["rope_theta"]
        # - PyPI transformers 4.57.1 and pretrained models use: config.rope_theta directly
        # This ensures the model can load pretrained weights from HuggingFace Hub
        if hasattr(config, 'rope_parameters') and config.rope_parameters is not None:
            # GitHub main branch format: rope_parameters dict per layer type
            base = config.rope_parameters[layer_type]["rope_theta"]
        else:
            # Pretrained model format: direct rope_theta attribute
            base = getattr(config, "rope_theta", 10000.0)

        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    def forward(self, x, position_ids, layer_type):
        with torch.no_grad():
            inv_freq = getattr(self, f"{layer_type}_inv_freq")
            attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

            inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
            position_ids_expanded = position_ids[:, None, :].float()

            device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):  # Force float32
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * attention_scaling
                sin = emb.sin() * attention_scaling

            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary position embeddings to packed sequences.

    Note: cos/sin are already correctly indexed for packed sequences because
    position_ids are created to restart at 0 for each sequence. We just need
    to broadcast them to match the query/key dimensions.

    Args:
        q: [num_heads, total_len, head_dim]
        k: [num_heads, total_len, head_dim]
        cos: [total_len, head_dim]
        sin: [total_len, head_dim]
        cu_seqlens: Not used, kept for API compatibility
    """
    # cos/sin already have correct values at each absolute position
    # Just add batch dimension for broadcasting: [1, total_len, head_dim]
    
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
    window_size: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Create an attention mask for packed sequences with optional sliding window.

    Args:
        cu_seqlens: Cumulative sequence lengths [batch + 1]
        causal: If True, apply causal masking
        device: Target device
        window_size: (left_window, right_window) or None for full attention
            - left_window: positions to the left (e.g., 2048)
            - right_window: positions to the right (e.g., 2048 for bidirectional, 0 for causal)
            - Use (-1, -1) or None for no window restriction

    Returns:
        Attention mask [total_len, total_len] with 0.0 (attend) and -inf (masked)
    """
    total_len = cu_seqlens[-1].item()
    seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)

    # 1 - Sequence Boundary Mask 
    # Prevent attention across different sequences in the batch
    seq_indices = torch.repeat_interleave(
        torch.arange(len(seq_lengths), device=device),
        seq_lengths
    )
    seq_mask = seq_indices.unsqueeze(0) == seq_indices.unsqueeze(1)

    # 2 - Causal Mask (if needed) 
    if causal:
        causal_mask = torch.tril(
            torch.ones(total_len, total_len, device=device, dtype=torch.bool)
        )
        combined_mask = seq_mask & causal_mask
    else:
        combined_mask = seq_mask

    # 3 - Sliding Window Mask (if specified) 
    if window_size is not None:
        left_window, right_window = window_size
        
        # Only apply if not (-1, -1) which means full attention
        if left_window >= 0 or right_window >= 0:
            # Create position-based distance matrix
            positions = torch.arange(total_len, device=device)
            # distance[i, j] = j - i (how far j is from i)
            distance = positions.unsqueeze(0) - positions.unsqueeze(1)
            
            # Window constraints:
            # - Can attend if distance is in range [-left_window, right_window]
            # - distance < 0 means looking backward (left)
            # - distance > 0 means looking forward (right)
            if left_window >= 0:
                within_left = distance >= -left_window
            else:
                within_left = torch.ones_like(distance, dtype=torch.bool)
            
            if right_window >= 0:
                within_right = distance <= right_window
            else:
                within_right = torch.ones_like(distance, dtype=torch.bool)
            
            window_mask = within_left & within_right
            combined_mask = combined_mask & window_mask

    # 4 - Convert to attention mask format 
    attention_mask = torch.full((total_len, total_len), float('-inf'), device=device)
    attention_mask.masked_fill_(combined_mask, 0.0)

    return attention_mask


def sdpa_attention_forward(
        q, k, v, 
        cu_seqlens, 
        scaling, 
        dropout: float = 0.0, 
        causal: bool = True,
        window_size: Optional[tuple[int, int]] = None,
    ):
    """Compute scaled dot-product attention for packed sequences."""
    attn_weights = torch.matmul(q, k.transpose(1, 2)) * scaling

    mask = create_packed_seqs_mask(cu_seqlens, causal, q.device, window_size)
    attn_weights = attn_weights + mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous()

    return attn_output, attn_weights

class Gemma3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper with packed sequence support"""

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = not self.config.use_bidirectional_attention

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
        self.attn_logit_softcapping = self.config.attn_logit_softcapping

        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (total_len, hidden_size) for packed sequences
            position_embeddings: (cos, sin) tuple
            cu_seqlens: Cumulative sequence lengths [batch + 1]
            max_seqlen: Maximum sequence length in the batch
        """
        input_shape = hidden_states.shape[:-1] #seq_len
        hidden_shape = (*input_shape, -1, self.head_dim) #(seq_len, -1, head_dim)
        
        #hidden_states: [seq_len, 256]
        #after q_proj (256, 1024): [seq_len, 1024] / #after k/v_proj : [seq_len, 512]
        #after view : [seq_len, num_heads, self.head_dim=256] -> q: [seq_len, 2, self.head_dim=256] / kv: [seq_len, 4, self.head_dim=256]
        #after transpose: [num_heads, seq_len, self.head_dim=256]
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(0, 1) 
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(0, 1)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(0, 1)

        # Apply Q/K normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # In your attention layer:
        if self.layer_type == "sliding_attention" and not self.is_causal:
            # Bidirectional sliding window (can look left AND right)
            window = (self.config.sliding_window - 1, self.config.sliding_window - 1)
        elif self.layer_type == "sliding_attention" and self.is_causal:
            # Causal sliding window (only look left)
            window = (self.config.sliding_window - 1, 0)
        else:
            # Full attention (no window restriction)
            window = None  # or (-1, -1)


        # Use flash attention if available and configured
        if self.config._attn_implementation == "flash_attention_2" and FLASH_ATTN_AVAILABLE and cu_seqlens is not None:

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
                causal=self.is_causal,
                window_size=window
            )
        else:
            attn_output, _ = sdpa_attention_forward(
                query_states,
                key_states,
                value_states,
                cu_seqlens=cu_seqlens,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling,
                causal=self.is_causal,
                window_size=window
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class Gemma3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3PreTrainedModel(PreTrainedModel):
    config: Gemma3TextConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["Gemma3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module):
        super()._init_weights(module)
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        if "RMSNorm" in module.__class__.__name__:
            module.weight.data.zero_()


class Gemma3TextModel(Gemma3PreTrainedModel):
    """
    Gemma3 Text Model with packed sequence support for efficient training.
    Input is expected as packed sequences with cu_seqlens.
    """

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (total_len,) packed input token IDs
            cu_seqlens: (batch + 1,) cumulative sequence lengths
            max_seqlen: Maximum sequence length in the batch
            
        Returns:
            hidden_states: (total_len, hidden_size) packed output
        """
        # Embed tokens: (total_len,) -> (total_len, hidden_size)
        hidden_states = self.embed_tokens(input_ids)
        
        # Create position IDs for the packed sequence
        # For packed sequences, position IDs should restart at 0 for each sequence
        total_len = input_ids.shape[0]
        if cu_seqlens is not None:
            # Generate position IDs that restart at 0 for each sequence
            position_ids_list = []
            for i in range(len(cu_seqlens) - 1):
                seq_len = cu_seqlens[i + 1] - cu_seqlens[i]
                position_ids_list.append(torch.arange(seq_len, device=hidden_states.device))
            position_ids = torch.cat(position_ids_list).unsqueeze(0)
        else:
            position_ids = torch.arange(total_len, device=hidden_states.device).unsqueeze(0)
        
        # Get position embeddings for each layer type
        position_embeddings = {}
        for layer_type in self.config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # Process through decoder layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[decoder_layer.attention_type],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class Gemma3ForCausalLM(Gemma3PreTrainedModel):
    """
    Gemma3 for Masked Language Modeling with packed sequences.
    """
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Gemma3TextConfig, fused_cross_entropy: bool = False):
        super().__init__(config)
        self.model = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.fused_cross_entropy = fused_cross_entropy
        if self.fused_cross_entropy:
            assert LIGER_KERNEL_AVAILABLE, "Liger kernel is not available."
            self.ligerCrossEntropy = LigerCrossEntropyLoss()

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (total_len,) packed input token IDs
            cu_seqlens: (batch + 1,) cumulative sequence lengths
            max_seqlen: Maximum sequence length in the batch
            labels: (total_len,) labels for MLM, -100 for non-masked tokens
            
        Returns:
            logits: (total_len, vocab_size) prediction scores
            loss: scalar loss if labels provided, else None
        """
        hidden_states = self.model(
            input_ids=input_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        
        logits = self.lm_head(hidden_states)

        # Apply final logit softcapping if configured
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

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
    "Gemma3PreTrainedModel",
    "Gemma3TextModel",
    "Gemma3ForCausalLM",
]
