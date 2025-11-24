"""Test that gemma3_packed produces the same outputs as gemma3"""

import torch
import torch.nn as nn
from typing import List, Tuple

from gemma3 import Gemma3ForCausalLM as Gemma3TextModel
from gemma3_packed import Gemma3ForCausalLM as Gemma3PackedTextModel
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig


def create_test_config(**kwargs):
    """Create a test config with sensible defaults. Pass kwargs to override."""
    defaults = dict(
        _sliding_window_pattern=6,
        architectures=["Gemma3ForCausalLM"],
        attention_bias=False,
        attention_dropout=0.0,
        attn_logit_softcapping=None,
        bos_token_id=2,
        dtype=torch.float32,
        eos_token_id=1,
        final_logit_softcapping=None,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        hidden_size=640,
        initializer_range=0.02,
        intermediate_size=2048,
        sliding_window=512,
        max_position_embeddings=32768,
        model_type="gemma3_text",
        num_attention_heads=4,
        num_hidden_layers=18,
        num_key_value_heads=1,
        pad_token_id=0,
        query_pre_attn_scalar=256,
        rms_norm_eps=1e-06,
        rope_local_base_freq=10000.0,
        rope_scaling=None,
        rope_theta=1000000.0,
        layer_types=["sliding_attention"]*5 + ["full_attention"] + ["sliding_attention"]*5 + ["full_attention"] + ["sliding_attention"]*5 + ["full_attention"],
        transformers_version="4.56.2",
        use_bidirectional_attention=False,
        use_cache=True,
        vocab_size=262144,
    )
    
    # Update with any kwargs that match
    for key in defaults:
        if key in kwargs:
            defaults[key] = kwargs[key]
    
    return Gemma3TextConfig(**defaults)


def pack_sequences(seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Pack multiple sequences into one tensor, return cumulative lengths."""
    lengths = [len(s) for s in seqs]
    cu_seqlens = torch.tensor([0] + [sum(lengths[:i+1]) for i in range(len(seqs))], dtype=torch.int32)
    packed = torch.cat(seqs, dim=0)
    return packed, cu_seqlens, max(lengths)


def unpack_sequences(packed: torch.Tensor, cu_seqlens: torch.Tensor) -> List[torch.Tensor]:
    """Split packed tensor back into individual sequences."""
    return [packed[cu_seqlens[i]:cu_seqlens[i+1]] for i in range(len(cu_seqlens) - 1)]


def pad_sequences(seqs: List[torch.Tensor], pad_val=0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences to the same length and return attention mask."""
    batch_size = len(seqs)
    max_len = max(len(s) for s in seqs)
    device = seqs[0].device
    dtype = seqs[0].dtype
    
    padded = torch.full((batch_size, max_len), pad_val, dtype=dtype, device=device)
    mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
        mask[i, :len(seq)] = 1
    
    return padded, mask


def copy_weights(src: nn.Module, tgt: nn.Module):
    """Copy weights from source model to target model."""
    src_dict = src.state_dict()
    tgt_dict = tgt.state_dict()
    
    for (src_name, src_param), (tgt_name, tgt_param) in zip(src_dict.items(), tgt_dict.items()):
        if src_param.shape == tgt_param.shape:
            tgt_dict[tgt_name] = src_param.clone()
        else:
            print(f"Shape mismatch: {src_name} {src_param.shape} vs {tgt_name} {tgt_param.shape}")
    
    tgt.load_state_dict(tgt_dict)


def test_equivalence(config, seqs, tol=1e-5, pretrained=False, device="cpu", attn_implementation="eager"):
    """Test that packed and unpacked models give the same outputs."""
    print(f"\nSequence lengths: {[len(s) for s in seqs]}")

    # Load or create models
    if pretrained:
        print('Loading pretrained models from HuggingFace')
        standard = Gemma3TextModel.from_pretrained("google/gemma-3-270m").to(device).to(config.dtype).eval()
        packed = Gemma3PackedTextModel.from_pretrained("google/gemma-3-270m").to(device).to(config.dtype).eval()
        standard.config._attn_implementation = attn_implementation
        packed.config._attn_implementation = attn_implementation

        # Apply config settings to both models
        for key in config.to_dict().keys():
            if hasattr(config, key):
                setattr(standard.config, key, getattr(config, key))
                setattr(packed.config, key, getattr(config, key))
            else:
                print(f"Warning: config missing attribute {key}")

        # Verify configs match
        assert standard.config.to_dict() == packed.config.to_dict(), "Configs don't match"

        # Update attention layer settings (they don't auto-update when config changes)
        for layer in standard.model.layers:
            layer.self_attn.is_causal = not config.use_bidirectional_attention
            if layer.self_attn.layer_type == "sliding_attention":
                layer.self_attn.sliding_window = config.sliding_window
        
        for layer in packed.model.layers:
            layer.self_attn.is_causal = not config.use_bidirectional_attention
            if layer.self_attn.layer_type == "sliding_attention":
                layer.self_attn.sliding_window = config.sliding_window
    
    else:
        # Create models from scratch
        standard = Gemma3TextModel(config).to(device).to(config.dtype).eval()
        packed = Gemma3PackedTextModel(config).to(device).to(config.dtype).eval()
        standard.config._attn_implementation = attn_implementation
        packed.config._attn_implementation = attn_implementation

        assert standard.config.to_dict() == packed.config.to_dict(), "Configs don't match"
        copy_weights(standard, packed)
    
    if attn_implementation == 'flash_attention_2':
        # WHY MLP NEEDS FLOAT64 FOR FLASH_ATTENTION_2:
        # -------------------------------------------------
        # Flash attention internally uses packed sequences (flash_attn_varlen_func) for both
        # standard and packed models, so attention outputs match perfectly.
        #
        # However, MLP layers operate on different tensor shapes:
        #   - Standard model: [batch, padded_seq_len, hidden] → uses batched matmul (bmm)
        #   - Packed model:   [total_seq_len, hidden]         → uses regular matmul (mm)
        #
        # Using float64 for MLP computation eliminates these kernel precision differences
        # while keeping attention in bfloat16 (required by flash attention).
        # -------------------------------------------------
        def make_float64_mlp_forward(mlp):
            orig_forward = mlp.forward
            def float64_forward(x):
                orig_dtype = x.dtype
                x = x.to(torch.float64)
                with torch.autocast(device_type='cuda', enabled=False):
                    result = orig_forward(x)
                return result.to(orig_dtype)
            return float64_forward

        for layer in standard.model.layers:
            layer.mlp.to(torch.float64)
            layer.mlp.forward = make_float64_mlp_forward(layer.mlp)
        
        for layer in packed.model.layers:
            layer.mlp.to(torch.float64)
            layer.mlp.forward = make_float64_mlp_forward(layer.mlp)

    # Prepare inputs
    padded_ids, attn_mask = pad_sequences(seqs, config.pad_token_id)
    padded_ids, attn_mask = padded_ids.to(device), attn_mask.to(device)
    
    packed_ids, cu_seqlens, max_seqlen = pack_sequences(seqs)
    packed_ids, cu_seqlens = packed_ids.to(device), cu_seqlens.to(device)
    
    # Run both models
    with torch.no_grad():
        print("\n--- Running standard model ---")
        standard_out = standard(input_ids=padded_ids, attention_mask=attn_mask, output_hidden_states=True).hidden_states[-1]
        
        print("\n--- Running packed model ---")
        packed_out = packed.model(input_ids=packed_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        
        print("\n--- Comparing outputs ---")
    
    # Compare results
    unpacked = unpack_sequences(packed_out, cu_seqlens)
    max_diff = 0.0
    all_match = True
    
    for i, (seq, unpacked_seq) in enumerate(zip(seqs, unpacked)):
        seq_len = len(seq)
        std_seq = standard_out[i, :seq_len, :]
        diff = torch.abs(std_seq - unpacked_seq).max().item()
        max_diff = max(max_diff, diff)
        ok = diff < tol
        all_match = all_match and ok
        print(f"  Seq {i}: len={seq_len:3d}, diff={diff:.2e} {'✓' if ok else '✗'}")
    
    print(f"Max diff: {max_diff:.2e} - {'PASS' if all_match else 'FAIL'}")
    return all_match


if __name__ == "__main__":
    # Test different configurations
    test_configs = [
        {
            "name": "eager + float64 + causal",
            "pretrained": True,
            "device": "cuda",
            "attn_implementation": "eager",
            "dtype": torch.float64,
            "use_bidirectional_attention": False,
            "seed": 42,
        },
        {
            "name": "eager + float64 + bidirectional",
            "pretrained": True,
            "device": "cuda",
            "attn_implementation": "eager",
            "dtype": torch.float64,
            "use_bidirectional_attention": True,
            "seed": 42,
        },
        {
            "name": "flash_attention_2 + bfloat16 + causal (MLPs in float64)",
            "pretrained": True,
            "device": "cuda",
            "attn_implementation": "flash_attention_2",
            "dtype": torch.bfloat16,
            "use_bidirectional_attention": False,
            "seed": 42,
        },
        {
            "name": "flash_attention_2 + bfloat16 + bidirectional (MLPs in float64)",
            "pretrained": True,
            "device": "cuda",
            "attn_implementation": "flash_attention_2",
            "dtype": torch.bfloat16,
            "use_bidirectional_attention": True,
            "seed": 42,
        },
    ]

    results = []

    for i, test_config in enumerate(test_configs):
        # Set random seed for reproducibility
        torch.manual_seed(test_config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(test_config["seed"])

        # Print what we're testing
        print("\n" + "#"*70)
        print(f"# TEST {i + 1}: {test_config['name']}")
        print("#"*70)
        print(f"Attention:      {test_config['attn_implementation']}")
        print(f"Dtype:          {test_config['dtype']}")
        print(f"Bidirectional:  {test_config['use_bidirectional_attention']}")

        config = create_test_config(**test_config)

        # Test 1: Different length sequences
        print("\n" + "-"*50)
        print("Test: Variable-length sequences [30, 300]")
        print("-"*50)
        seqs = [torch.randint(0, 1000, (30,)), torch.randint(0, 1000, (300,))]
        result1 = test_equivalence(
            config,
            seqs,
            pretrained=test_config["pretrained"],
            device=test_config["device"],
            attn_implementation=test_config["attn_implementation"]
        )

        # Test 2: Same length sequences
        print("\n" + "-"*50)
        print("Test: Same-length sequences [30, 30]")
        print("-"*50)
        seqs = [torch.randint(0, 1000, (30,)), torch.randint(0, 1000, (30,))]
        result2 = test_equivalence(
            config,
            seqs,
            pretrained=test_config["pretrained"],
            device=test_config["device"],
            attn_implementation=test_config["attn_implementation"]
        )

        results.append({
            "name": test_config["name"],
            "variable_length": "PASS" if result1 else "FAIL",
            "same_length": "PASS" if result2 else "FAIL",
        })

    # Print summary of all tests
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for r in results:
        print(f"{r['name']:50} | var: {r['variable_length']} | same: {r['same_length']}")