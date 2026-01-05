import torch.distributed.checkpoint as dcp
import sys

def list_dcp_keys(checkpoint_dir):
    try:
        # 1. Create a reader for the specific checkpoint directory
        reader = dcp.FileSystemReader(checkpoint_dir)
        
        # 2. Read only the metadata (lightweight, does not load tensors)
        metadata = reader.read_metadata()
        
        # 3. Extract keys
        # The metadata structure contains a 'state_dict_metadata' dictionary
        keys = list(metadata.state_dict_metadata.keys())
        
        print(f"\n--- Found {len(keys)} keys in checkpoint: {checkpoint_dir} ---")
        
        # Sort for easier reading
        keys.sort()
        
        for key in keys:
            print(key)
            
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

if __name__ == "__main__":
        list_dcp_keys("/Users/nboizard/Downloads/lora_weight")