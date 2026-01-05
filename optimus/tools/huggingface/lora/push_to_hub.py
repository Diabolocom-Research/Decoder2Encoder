import argparse
import torch
import torch.distributed.checkpoint as dcp
from peft import PeftModel, PeftConfig
from transformers import AutoModel

def reassemble_fsdp_lora(
    base_model_id: str,
    checkpoint_dir: str,
    output_dir: str,
    device: str = "cpu"
):
    peft_config = PeftConfig.from_pretrained(checkpoint_dir)
    
    try:
        model = AutoModel.from_pretrained(
            base_model_id,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    model = PeftModel(model, peft_config)
    
    local_state_dict = model.state_dict()
    keys_to_load = {k: v for k, v in local_state_dict.items() if "lora" in k}
    # keys_to_load = {k: v for k, v in local_state_dict.items()}
    print(f"Keys to load: {list(keys_to_load.keys())}")

    
    dcp_container = {"model": keys_to_load}

    try:
        dcp.load(
            state_dict=dcp_container,
            checkpoint_id=checkpoint_dir,
        )
    except RuntimeError:
        fsdp_keys_to_load = {}
        for k, v in keys_to_load.items():
            fsdp_keys_to_load[f"_fsdp_wrapped_module.{k}"] = v
            fsdp_keys_to_load[f"module.{k}"] = v
            
        dcp_container = {"model": fsdp_keys_to_load}
        
        dcp.load(
            state_dict=dcp_container,
            checkpoint_id=checkpoint_dir,
        )

    model.save_pretrained(output_dir)
    print(f"Saved LoRA model to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reassemble FSDP LoRA checkpoints")
    parser.add_argument("--base_model_id", type=str, required=True, help="HF Base model ID")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to FSDP checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output path for merged LoRA")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load model on")
    
    args = parser.parse_args()

    reassemble_fsdp_lora(
        base_model_id=args.base_model_id,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        device=args.device
    )