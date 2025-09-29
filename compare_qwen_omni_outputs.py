import torch
import torch.nn as nn
import torch.nn.functional as F

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

def show_difference_positions(tensor1, tensor2, tensor_name, max_display=20):
    """Show positions where two tensors differ."""
    print(f"\n  Finding difference positions for {tensor_name}...")
    
    # Check if tensors are not equal element-wise
    diff_mask = ~torch.isclose(tensor1, tensor2, rtol=1e-5, atol=1e-8)
    
    # Get indices where tensors differ
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    
    num_diffs = diff_indices.shape[0]
    print(f"  Total number of differing elements: {num_diffs} out of {tensor1.numel()} ({100*num_diffs/tensor1.numel():.4f}%)")
    
    if num_diffs > 0:
        # Show first few differences
        num_to_show = min(max_display, num_diffs)
        print(f"  Showing first {num_to_show} differences:")
        print(f"  {'Position':<30} {'Tensor1 Value':<20} {'Tensor2 Value':<20} {'Abs Diff':<15}")
        print(f"  {'-'*30} {'-'*20} {'-'*20} {'-'*15}")
        
        for i in range(num_to_show):
            idx = tuple(diff_indices[i].tolist())
            val1 = tensor1[idx].item()
            val2 = tensor2[idx].item()
            abs_diff = abs(val1 - val2)
            
            # Format position based on number of dimensions
            if len(idx) == 1:
                pos_str = f"[{idx[0]}]"
            else:
                pos_str = str(idx)
            
            print(f"  {pos_str:<30} {val1:<20.6f} {val2:<20.6f} {abs_diff:<15.6e}")
        
        if num_diffs > max_display:
            print(f"  ... and {num_diffs - max_display} more differences")

# Function to verify tensor equality
def verify_tensors(tensor_name, mod_tensor, ref_tensor, attention_mask):
    print(f"=== {tensor_name} ===")
    print(f"mod.shape: {mod_tensor.shape}")
    print(f"ref.shape: {ref_tensor.shape}")
    
    # Apply normalization to ref tensor
    packed_ref = batch_tensor_to_packed_tensor(ref_tensor, attention_mask)
    print(f"normalize_tensor_shape(ref, attention_mask).shape: {packed_ref.shape}")
    
    # Find the dimension to squeeze in mod_tensor
    mod_squeezed = mod_tensor
    squeeze_dim = None
    
    # Look for dimensions of size 1 that need to be squeezed
    for i, size in enumerate(mod_tensor.shape):
        if size == 1:
            # Check if squeezing this dimension would make shapes compatible
            test_shape = list(mod_tensor.shape)
            test_shape.pop(i)
            if tuple(test_shape) == packed_ref.shape:
                squeeze_dim = i
                mod_squeezed = mod_tensor.squeeze(i)
                break
    
    if squeeze_dim is not None:
        print(f"mod.squeeze({squeeze_dim}).shape: {mod_squeezed.shape}")
        
        # Check equality
        is_equal = torch.equal(packed_ref, mod_squeezed)
        print(f"torch.equal(normalize_tensor_shape(ref, attention_mask), mod.squeeze({squeeze_dim})): {is_equal}")
        
        if not is_equal:
            print(f"Max absolute difference: {torch.max(torch.abs(packed_ref - mod_squeezed)).item()}")
            print(f"Mean absolute difference: {torch.mean(torch.abs(packed_ref - mod_squeezed)).item()}")
            show_difference_positions(packed_ref, mod_squeezed, tensor_name)
    else:
        # Check equality without squeezing
        is_equal = torch.equal(packed_ref, mod_tensor)
        print(f"torch.equal(normalize_tensor_shape(ref, attention_mask), mod): {is_equal}")
        
        if not is_equal:
            if packed_ref.shape == mod_tensor.shape:
                print(f"Max absolute difference: {torch.max(torch.abs(packed_ref - mod_tensor)).item()}")
                print(f"Mean absolute difference: {torch.mean(torch.abs(packed_ref - mod_tensor)).item()}")
                show_difference_positions(packed_ref, mod_tensor, tensor_name)
            else:
                print(f"Shape mismatch: cannot compare tensors directly")

    print()

# Function to verify proj_weights tensors without batch_tensor_to_packed_tensor
def verify_proj_weights(tensor_name, mod_tensor, ref_tensor):
    print(f"=== {tensor_name} (Direct Comparison) ===")
    print(f"mod.shape: {mod_tensor.shape}")
    print(f"ref.shape: {ref_tensor.shape}")
    
    # Direct comparison without any transformation
    if mod_tensor.shape == ref_tensor.shape:
        is_equal = torch.equal(mod_tensor, ref_tensor)
        print(f"torch.equal(mod, ref): {is_equal}")
        
        if not is_equal:
            print(f"Max absolute difference: {torch.max(torch.abs(mod_tensor - ref_tensor)).item()}")
            print(f"Mean absolute difference: {torch.mean(torch.abs(mod_tensor - ref_tensor)).item()}")
            show_difference_positions(mod_tensor, ref_tensor, tensor_name)

        # Also check dtypes to be sure
        print(f"mod.dtype: {mod_tensor.dtype}, ref.dtype: {ref_tensor.dtype}")
    else:
        print("Shape mismatch: cannot compare tensors directly")
        
        # Try squeezing dimensions of size 1 from both tensors
        mod_squeezed = mod_tensor.squeeze()
        ref_squeezed = ref_tensor.squeeze()
        
        print(f"mod.squeeze().shape: {mod_squeezed.shape}")
        print(f"ref.squeeze().shape: {ref_squeezed.shape}")
        
        if mod_squeezed.shape == ref_squeezed.shape:
            is_equal = torch.equal(mod_squeezed, ref_squeezed)
            print(f"torch.equal(mod.squeeze(), ref.squeeze()): {is_equal}")
            
            if not is_equal:
                print(f"Max absolute difference: {torch.max(torch.abs(mod_squeezed - ref_squeezed)).item()}")
                print(f"Mean absolute difference: {torch.mean(torch.abs(mod_squeezed - ref_squeezed)).item()}")
                show_difference_positions(mod_squeezed, ref_squeezed, tensor_name)
        else:
            print("Still shape mismatch after squeezing")
    
    print()

# Load the saved tensors
print("Loading tensors...")
mod_data = torch.load('/root/theo_db/projets/Decoder2Encoder/data_test/mod_attention_layer0.pt')
ref_data = torch.load('/root/theo_db/projets/Decoder2Encoder/data_test/ref_attention_layer0.pt')
attention_mask = torch.load('/root/theo_db/projets/Decoder2Encoder/data_test/ref_attention_mask_layer0.pt')

print(f"Attention mask shape: {attention_mask.shape}")
print(f"Attention mask valid tokens per batch: {attention_mask.sum(dim=1)}")
print()

# Find common keys between both dictionaries
mod_keys = set(mod_data.keys())
ref_keys = set(ref_data.keys())
common_keys = mod_keys & ref_keys

# Categorize keys
proj_weights_keys = [key for key in common_keys if 'proj_weights' in key or 'proj_bias' in key]
saved_projection_keys = [key for key in common_keys if any(x in key for x in ['query_states', 'key_states', 'value_states']) 
                         and 'proj' not in key]
other_keys = [key for key in common_keys if key not in proj_weights_keys and key not in saved_projection_keys]

# ============================================================================
# SECTION 1: COMPARISON OF WEIGHTS AND BIASES
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: COMPARISON OF PROJECTION WEIGHTS AND BIASES")
print("="*80 + "\n")

for tensor_name in sorted(proj_weights_keys):
    verify_proj_weights(tensor_name, mod_data[tensor_name], ref_data[tensor_name])

# ============================================================================
# SECTION 2: COMPARISON OF SAVED PROJECTION OUTPUTS
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: COMPARISON OF SAVED PROJECTION OUTPUTS")
print("="*80 + "\n")

if saved_projection_keys:
    for tensor_name in sorted(saved_projection_keys):
        verify_tensors(tensor_name, mod_data[tensor_name], ref_data[tensor_name], attention_mask)
else:
    print("No saved projection outputs found in the data files.\n")

# ============================================================================
# SECTION 3: COMPARISON OF COMPUTED PROJECTION OUTPUTS
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: COMPARISON OF COMPUTED PROJECTION OUTPUTS")
print("="*80 + "\n")

num_heads = 8
num_key_value_heads = 2
head_dim = 128
hidden_size = 2048

# Get device and dtype from loaded data
device = ref_data['hidden_states_attention_layer0'].device
dtype = ref_data['hidden_states_attention_layer0'].dtype

print(f"Setting up projection layers with device={device}, dtype={dtype}\n")

# Create linear layers with consistent device/dtype
ref_q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True).to(device).to(dtype)
ref_k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True).to(device).to(dtype)
ref_v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True).to(device).to(dtype)

mod_q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True).to(device).to(dtype)
mod_k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True).to(device).to(dtype)
mod_v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True).to(device).to(dtype)

# Load weights and biases
ref_q_proj.weight.data = ref_data['q_proj_weights']
ref_k_proj.weight.data = ref_data['k_proj_weights']
ref_v_proj.weight.data = ref_data['v_proj_weights']
ref_q_proj.bias.data = ref_data['q_proj_bias']
ref_k_proj.bias.data = ref_data['k_proj_bias']
ref_v_proj.bias.data = ref_data['v_proj_bias']

mod_q_proj.weight.data = mod_data['q_proj_weights']
mod_k_proj.weight.data = mod_data['k_proj_weights']
mod_v_proj.weight.data = mod_data['v_proj_weights']
mod_q_proj.bias.data = mod_data['q_proj_bias']
mod_k_proj.bias.data = mod_data['k_proj_bias']
mod_v_proj.bias.data = mod_data['v_proj_bias']

# Apply projections
print("Computing projections from hidden states...\n")
ref_query_states = ref_q_proj(ref_data["hidden_states_attention_layer0"])
ref_key_states = ref_k_proj(ref_data["hidden_states_attention_layer0"])
ref_value_states = ref_v_proj(ref_data["hidden_states_attention_layer0"])

mod_query_states = mod_q_proj(mod_data["hidden_states_attention_layer0"])
mod_key_states = mod_k_proj(mod_data["hidden_states_attention_layer0"])
mod_value_states = mod_v_proj(mod_data["hidden_states_attention_layer0"])

# Verify the projected states
verify_tensors("computed_query_states", mod_query_states, ref_query_states, attention_mask)
verify_tensors("computed_key_states", mod_key_states, ref_key_states, attention_mask)
verify_tensors("computed_value_states", mod_value_states, ref_value_states, attention_mask)