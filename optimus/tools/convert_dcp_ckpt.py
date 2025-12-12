import argparse
import os
import torch
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.default_planner as dcp_default_planner
import torch.distributed.checkpoint.metadata as dcp_metadata
import torch.distributed.checkpoint.state_dict_loader as dcp_sdl
import torch.distributed.checkpoint._traverse as dcp_traverse

from typing import Optional


def main():
    print("Converting DCP checkpoint to Torch save file")
    parser = argparse.ArgumentParser(description="Convert DCP checkpoint to Torch save file")
    parser.add_argument("--dcp_dir", type=str, required=True, help="Directory containing the DCP checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the converted checkpoint")
    parser.add_argument("--weights_only", action="store_true", default=True, help="Only convert model weights (default: True)")
    
    args = parser.parse_args()
    
    dcp_dir = args.dcp_dir
    output_dir = args.output_dir
    weights_only = args.weights_only
    
    # This code is losly based on the function dcp_to_torch_save in:
    # torch/distributed/checkpoint/format_utils.py
    sd: dcp_metadata.STATE_DICT_TYPE = {}
    print(f"Loading DCP checkpoint from {dcp_dir}")
    dcp_sdl._load_state_dict(
        state_dict=sd,
        storage_reader=dcp.FileSystemReader(dcp_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(f"Saving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model_sd = sd["model"]
    torch.save(model_sd, f"{output_dir}/model.pt")
    
    if not weights_only:
        print(f"Saving optimizer to {output_dir}")
        optimizer_sd = sd["optimizer"]
        torch.save(optimizer_sd, f"{output_dir}/optimizer.pt")
    print("Conversion complete")


class _EmptyStateDictLoadPlanner(dcp_default_planner.DefaultLoadPlanner):
    """
    Copied from torch/distributed/checkpoint/default_planner.py

    Extension of DefaultLoadPlanner, which rebuilds state_dict from the saved metadata.
    Useful for loading in state_dict without first initializing a model, such as
    when converting a DCP checkpoint into a Torch save file.

    . N.B. `state_dict` must be an empty dictionary when used with this LoadPlanner

    .. warning::
        Because the entire state dict is initialized, It's recommended to only utilize
        this LoadPlanner on a single rank or process to avoid OOM.

    """

    def __init__(self, keys=None, *args, **kwargs):
        self.keys = keys
        super().__init__(*args, **kwargs)

    def _should_include_key(self, key: str, metadata: dcp_metadata.Metadata) -> bool:
        if self.keys is None:
            return True

        if key in self.keys:
            return True

        unflattened_keys: list[str] = []
        planner_data = metadata.planner_data.get(key)
        for unflattened_key in planner_data:
            if unflattened_keys:
                unflattened_keys.append(
                    ".".join([unflattened_keys[-1], str(unflattened_key)])
                )

            else:
                unflattened_keys.append(unflattened_key)

        if any(unflattened_key in self.keys for unflattened_key in unflattened_keys):
            return True

        return False

    def set_up_planner(
        self,
        state_dict: dcp_metadata.STATE_DICT_TYPE,
        metadata: Optional[dcp_metadata.Metadata] = None,
        is_coordinator: bool = False,
    ) -> None:
        assert not state_dict
        assert metadata is not None

        # rebuild the state dict from the metadata
        for k, v in metadata.state_dict_metadata.items():
            if not self._should_include_key(k, metadata):
                continue

            if isinstance(v, dcp_metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            if k in metadata.planner_data:
                dcp_traverse.set_element(state_dict, metadata.planner_data[k], v)
            else:
                state_dict[k] = v

        super().set_up_planner(state_dict, metadata, is_coordinator)


if __name__ == "__main__":
    main()
