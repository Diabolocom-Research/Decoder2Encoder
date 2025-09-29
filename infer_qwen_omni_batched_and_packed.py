import torch
import gc
import time
import sys
from collections import Counter
from transformers import Qwen2_5OmniProcessor

sys.path.insert(0, '/root/theo_db/projets/Decoder2Encoder')
from optimus.trainer.model.encoder.biqwen_omni_ref import Qwen2_5OmniThinkerForConditionalGeneration
from optimus.trainer.model.encoder.biqwen_omni import Qwen2_5OmniThinkerForMaskedLM as ModifiedModel
from qwen_omni_utils import process_mm_info

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
USE_AUDIO_IN_VIDEO = False

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")

# Special token IDs
AUDIO_TOKEN_ID = processor.tokenizer.audio_token_id
IMAGE_TOKEN_ID = processor.tokenizer.image_token_id
VIDEO_TOKEN_ID = processor.tokenizer.video_token_id
VISION_BOS_TOKEN_ID = processor.tokenizer.vision_bos_token_id
VISION_EOS_TOKEN_ID = processor.tokenizer.vision_eos_token_id
AUDIO_BOS_TOKEN_ID = processor.tokenizer.audio_bos_token_id
AUDIO_EOS_TOKEN_ID = processor.tokenizer.audio_eos_token_id

conversations = [
    [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen..."}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Describe:"},
            {"type": "audio", "audio": "data_test/1-second-of-silence.mp3"}
        ]}
    ],
    [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen..."}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Describe:"},
            {"type": "audio", "audio": "data_test/1-second-of-silence.mp3"},
            {"type": "image", "image": "data_test/diabolocom_logo.png"}
        ]}
    ]
]

# Batched inputs
text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)
batched_inputs = processor(text=text, audio=audios, images=images, videos=videos,
                           return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)

# Analyze token types
print("\nToken Analysis:")
for batch_idx in range(batched_inputs["input_ids"].shape[0]):
    input_ids_seq = batched_inputs["input_ids"][batch_idx]
    attention_mask_seq = batched_inputs["attention_mask"][batch_idx]
    token_types = []
    
    for i, token_id in enumerate(input_ids_seq):
        if attention_mask_seq[i] == 0:
            continue
        
        if token_id == AUDIO_TOKEN_ID: token_types.append("AUDIO")
        elif token_id == IMAGE_TOKEN_ID: token_types.append("IMAGE")
        elif token_id == VIDEO_TOKEN_ID: token_types.append("VIDEO")
        elif token_id == AUDIO_BOS_TOKEN_ID: token_types.append("AUDIO_BOS")
        elif token_id == AUDIO_EOS_TOKEN_ID: token_types.append("AUDIO_EOS")
        elif token_id == VISION_BOS_TOKEN_ID: token_types.append("VISION_BOS")
        elif token_id == VISION_EOS_TOKEN_ID: token_types.append("VISION_EOS")
        else: token_types.append("TEXT")
    
    type_counts = Counter(token_types)
    print(f"\nConv {batch_idx + 1}: {dict(sorted(type_counts.items()))}")
    
    type_indices = {}
    for i, tt in enumerate(token_types):
        type_indices.setdefault(tt, []).append(i)
    
    for tt in ['AUDIO_BOS', 'AUDIO', 'AUDIO_EOS', 'VISION_BOS', 'IMAGE', 'VISION_EOS', 'TEXT']:
        if tt in type_indices:
            idx = type_indices[tt]
            print(f"  {tt}: {idx[:5]}{'...' if len(idx) > 5 else ''}")


# Packed inputs
all_ids, cu_lens = [], [0]
for conv in conversations:
    single = processor(
        text=processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False),
        audio=process_mm_info([conv], use_audio_in_video=USE_AUDIO_IN_VIDEO)[0],
        images=process_mm_info([conv], use_audio_in_video=USE_AUDIO_IN_VIDEO)[1],
        return_tensors="pt", padding=False, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    all_ids.append(single["input_ids"].squeeze(0))
    cu_lens.append(cu_lens[-1] + single["input_ids"].shape[1])

packed_inputs = {
    "input_ids": torch.cat(all_ids),
    "cu_seq_lens": torch.tensor(cu_lens, dtype=torch.int32),
    "max_seqlen": max(len(s) for s in all_ids),
    **{k: batched_inputs[k] for k in ["input_features", "pixel_values", "pixel_values_videos",
                                       "feature_attention_mask", "audio_feature_lengths", 
                                       "image_grid_thw", "video_grid_thw"] if k in batched_inputs}
}


def run_model(model, inputs):
    inputs = {k: v.to(model.device).to(model.dtype if k in ["pixel_values", "input_features"] 
              else v.dtype) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        out = model(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return out, (time.perf_counter() - start) * 1000

def batch_tensor_to_packed_tensor(tensor, attention_mask=None):
    """Transform tensor shape by removing batch dimensions and handling different formats."""
    if tensor is None:
        return None

    # Handle different tensor shapes between batched and packed formats
    if len(tensor.shape) >= 3:
        # For batched inputs, we need to handle multiple conversations
        if tensor.shape[0] > 1:  # Multiple conversations
            # Flatten batch and sequence dimensions: (batch, seq_len, dim) -> (batch*seq_len, dim)
            flattened_tensor = tensor.view(-1, tensor.shape[-1])
            
            # If attention mask is provided, use it to remove padding
            if attention_mask is not None:
                # Flatten attention mask to match tensor
                attention_mask_flat = attention_mask.view(-1)  # [batch*seq_len]
                    
                # Remove padding tokens using attention mask
                if len(attention_mask_flat) == flattened_tensor.shape[0]:
                    valid_indices = attention_mask_flat.bool()
                    flattened_tensor = flattened_tensor[valid_indices]
            
            return flattened_tensor

    return tensor


def compare(n1, n2, name):
    torch.equal(n1, n2)
    diff = torch.abs(n1 - n2).float()
    close = torch.allclose(n1, n2, rtol=1e-6, atol=1e-6)
    print(f"{name}: max_diff={diff.max():.2e}, {'✓' if close else '✗'}")
    return close


# Reference model
ref_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B", attn_implementation="flash_attention_2", torch_dtype=dtype
).to(device).eval()
ref_model.disable_talker() if hasattr(ref_model, "disable_talker") else None

ref_out, ref_time = run_model(ref_model, batched_inputs)
del ref_model
torch.cuda.empty_cache(); gc.collect()

# Modified model
mod_model = ModifiedModel.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B", attn_implementation="flash_attention_2", torch_dtype=dtype
).to(device).eval()
mod_model.disable_talker() if hasattr(mod_model, "disable_talker") else None

mod_out, mod_time = run_model(mod_model, packed_inputs)
del mod_model
torch.cuda.empty_cache(); gc.collect()

# Compare
print("")
mask = batched_inputs.get("attention_mask")
for i, name in enumerate(["Embeddings", "LAST Hidden States", "LAST Logits"]):
    ref_out_packed = batch_tensor_to_packed_tensor(ref_out[i], mask)
    print(name, ref_out[i].shape, "->", ref_out_packed.shape, "vs", mod_out[i].shape)
    compare(ref_out_packed, mod_out[i], name)

print(f"\nRef: {ref_time:.1f}ms | Mod: {mod_time:.1f}ms | Speedup: {ref_time/mod_time:.2f}x")