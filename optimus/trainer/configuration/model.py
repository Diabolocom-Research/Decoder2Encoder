from dataclasses import dataclass


@dataclass
class ModelConfig:
    huggingface_id: str = (
        None  # Huggingface model id (if provided, other model parameters are ignored)
    )
    tokenizer_path_or_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    mask_token_id: int = 128002
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    # Model parameters
    model_name: str = "eurobert"
    model_size: str | None = "210m"

    # If parameters are not provided, default values of the model size are used.
    vocab_size: int | None = None  # Vocabulary size
    embedding_size: int | None = None  # Embedding size
    num_head: int | None = None  # Number of heads
    num_kv_head: int | None = None  # Number of key-value heads
    num_layer: int | None = None  # Number of layers
    block_size: int | None = None  # Block size
    dropout: int | None = None  # Dropout
    mlp_hidden_dim: int | None = None  # MLP hidden dimension
    bias: bool | None = None  # Bias
    attn_impl: str | None = None  # Attention implementation
    rope_base: int | None = None  # Rope base

    gpu: bool = True
    fused_rms_norm: bool = False  # Fused RMS norm
    fused_rope: bool = False  # Fused rope
    fused_swiglu: bool = False  # Fused swiglu
    fused_cross_entropy: bool = False  # Fused cross entropy
    fused_linear_cross_entropy: bool = False  # Fused linear cross entropy
