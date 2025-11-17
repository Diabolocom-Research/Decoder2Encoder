"""Test that gemma3_packed produces same outputs as gemma3"""

import torch
import torch.nn as nn
from typing import List, Tuple
import sys
import os

from gemma3 import Gemma3ForCausalLM as Gemma3TextModel
from gemma3_packed import Gemma3ForCausalLM as Gemma3PackedTextModel

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

# Read attention implementation from environment variable, default to "eager"
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "eager")


def create_test_config(
    vocab_size=1000,
    hidden_size=512,
    num_layers=4,
    num_heads=8,
    num_kv_heads=4,
    bidirectional=False,
    layer_type="sliding_attention",
    sliding_window=2,
    layer_types=None 
):
    if layer_types is None:
        layer_types = [layer_type] * num_layers
    
    return Gemma3TextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=2048,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_dropout=0.0,
        attention_bias=False,
        query_pre_attn_scalar=256,
        use_bidirectional_attention=bidirectional,
        layer_types=layer_types,
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
        sliding_window=sliding_window,
        _attn_implementation="eager",
    )


def pack_sequences(seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Pack sequences into single tensor with cu_seqlens"""
    lens = [len(s) for s in seqs]
    cu_seqlens = torch.tensor([0] + [sum(lens[:i+1]) for i in range(len(seqs))], dtype=torch.long)
    packed = torch.cat(seqs, dim=0)
    return packed, cu_seqlens, max(lens)


def unpack_sequences(packed: torch.Tensor, cu_seqlens: torch.Tensor) -> List[torch.Tensor]:
    """Unpack flat tensor back to list of sequences"""
    return [packed[cu_seqlens[i]:cu_seqlens[i+1]] for i in range(len(cu_seqlens) - 1)]


def pad_sequences(seqs: List[torch.Tensor], pad_val=0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences to same length"""
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
    """Copy weights from source to target model"""
    src_dict = src.state_dict()
    tgt_dict = tgt.state_dict()
    
    for (src_name, src_param), (tgt_name, tgt_param) in zip(src_dict.items(), tgt_dict.items()):
        if src_param.shape == tgt_param.shape:
            tgt_dict[tgt_name] = src_param.clone()
        else:
            print(f"Shape mismatch: {src_name} {src_param.shape} vs {tgt_name} {tgt_param.shape}")
    
    tgt.load_state_dict(tgt_dict)


def test_equivalence(config, seqs, tol=1e-5, pretrained=False, device="cpu"):
    """Test packed vs unpacked models produce same outputs"""
    print(f"\nSequence lengths: {[len(s) for s in seqs]}")
    

    if pretrained:
        print('Using pretrained models from HuggingFace')
        standard = Gemma3TextModel.from_pretrained("google/gemma-3-270m").to(device).eval()
        packed = Gemma3PackedTextModel.from_pretrained("google/gemma-3-270m").to(device).eval()
        standard.config._attn_implementation = ATTN_IMPLEMENTATION #sdpa is not comparable to packed custom sdpa
        packed.config._attn_implementation = ATTN_IMPLEMENTATION
    else:
        # Create and prepare models from scratch
        standard = Gemma3TextModel(config).to(device).eval()
        packed = Gemma3PackedTextModel(config).to(device).eval()
        standard.config._attn_implementation = ATTN_IMPLEMENTATION
        packed.config._attn_implementation = ATTN_IMPLEMENTATION
        copy_weights(standard, packed)

        
    # Prepare inputs
    padded_ids, attn_mask = pad_sequences(seqs, config.pad_token_id)
    padded_ids, attn_mask = padded_ids.to(device), attn_mask.to(device)
    
    packed_ids, cu_seqlens, max_seqlen = pack_sequences(seqs)
    packed_ids, cu_seqlens = packed_ids.to(device), cu_seqlens.to(device)
    
    # Run inference
    with torch.no_grad():
        print("\n--- Running STANDARD (gemma3.py) ---")
        standard_out = standard(input_ids=padded_ids, attention_mask=attn_mask, output_hidden_states=True).hidden_states[-1]
        print("\n--- Running PACKED (gemma3_packed.py) ---")
        packed_out = packed.model(input_ids=packed_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        print("\n--- Comparing outputs ---")
    
    # Compare
    unpacked = unpack_sequences(packed_out, cu_seqlens)
    max_diff = 0.0
    all_ok = True
    
    for i, (seq, unpacked_seq) in enumerate(zip(seqs, unpacked)):
        seq_len = len(seq)
        std_seq = standard_out[i, :seq_len, :]
        diff = torch.abs(std_seq - unpacked_seq).max().item()
        max_diff = max(max_diff, diff)
        ok = diff < tol
        all_ok = all_ok and ok
        print(f"  Seq {i}: len={seq_len:3d}, diff={diff:.2e} {'✓' if ok else '✗'}")
    
    print(f"Max diff: {max_diff:.2e} - {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test(pretrained=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "="*60)
    print(f"Testing Gemma3 vs Gemma3Packed (pretrained={pretrained})")
    print(f"Device: {device}")
    print(f"Attention Implementation: {ATTN_IMPLEMENTATION}")
    
    # Test 1: causal
    print("\n" + "="*60)
    print("Test 1: causal")
    print("="*60)
    config = create_test_config(hidden_size=256, num_layers=2, num_heads=4, num_kv_heads=2)
    seqs = [torch.randint(0, 1000, (10,)), torch.randint(0, 1000, (15,)), torch.randint(0, 1000, (8,))]
    t1 = test_equivalence(config, seqs, pretrained=pretrained, device=device)
    
    # Test 2: causal + sliding
    print("\n" + "="*60)
    print("Test 2: causal + sliding")
    print("="*60)
    config = create_test_config(hidden_size=256, num_layers=2, num_heads=4, num_kv_heads=2, layer_type="sliding_attention", sliding_window=2)
    seqs = [torch.randint(0, 1000, (10,)), torch.randint(0, 1000, (15,)), torch.randint(0, 1000, (8,))]
    t2 = test_equivalence(config, seqs, pretrained=pretrained, device=device)
    
    # Test 3: bidirectional
    print("\n" + "="*60)
    print("Test 3: bidirectional")
    print("="*60)
    config = create_test_config(hidden_size=256, num_layers=2, num_heads=4, num_kv_heads=2, bidirectional=True)
    seqs = [torch.randint(0, 1000, (32,)), torch.randint(0, 1000, (28,)), 
            torch.randint(0, 1000, (35,)), torch.randint(0, 1000, (30,))]
    t3 = test_equivalence(config, seqs, pretrained=pretrained, device=device)

    # Test 4: bidirectional + sliding
    print("\n" + "="*60)
    print("Test 4: bidirectional + sliding")
    print("="*60)
    config = create_test_config(hidden_size=256, num_layers=2, num_heads=4, num_kv_heads=2, bidirectional=True, layer_type="sliding_attention", sliding_window=2)
    seqs = [torch.randint(0, 1000, (32,)), torch.randint(0, 1000, (28,)), 
            torch.randint(0, 1000, (35,)), torch.randint(0, 1000, (30,))]
    t4 = test_equivalence(config, seqs, pretrained=pretrained, device=device)
    
    # Test 5: causal with mixed layers [full_attention, sliding_attention]
    print("\n" + "="*60)
    print("Test 5: Causal with mixed layers [full_attention, sliding_attention]")
    print("="*60)
    config = create_test_config(
        hidden_size=256, 
        num_layers=2, 
        num_heads=4, 
        num_kv_heads=2, 
        bidirectional=False,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=2
    )
    seqs = [torch.randint(0, 1000, (10,)), torch.randint(0, 1000, (15,)), torch.randint(0, 1000, (8,))]
    t5 = test_equivalence(config, seqs, pretrained=pretrained, device=device)
    
    # Test 6: bidirectional with mixed layers [full_attention, sliding_attention]
    print("\n" + "="*60)
    print("Test 6: Bidirectional with mixed layers [full_attention, sliding_attention]")
    print("="*60)
    config = create_test_config(
        hidden_size=256, 
        num_layers=2, 
        num_heads=4, 
        num_kv_heads=2, 
        bidirectional=True,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=2
    )
    seqs = [torch.randint(0, 1000, (32,)), torch.randint(0, 1000, (28,)), 
            torch.randint(0, 1000, (35,)), torch.randint(0, 1000, (30,))]
    t6 = test_equivalence(config, seqs, pretrained=pretrained, device=device)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test 1: {'PASS' if t1 else 'FAIL'}")
    print(f"Test 2: {'PASS' if t2 else 'FAIL'}")
    print(f"Test 3: {'PASS' if t3 else 'FAIL'}")
    print(f"Test 4: {'PASS' if t4 else 'FAIL'}")
    print(f"Test 5: {'PASS' if t5 else 'FAIL'}")
    print(f"Test 6: {'PASS' if t6 else 'FAIL'}")
    print("="*60)
    
    return all([t1, t2, t3, t4, t5, t6])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    test(pretrained=False)

    test(pretrained=True)