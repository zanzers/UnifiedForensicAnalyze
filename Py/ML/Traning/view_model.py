import torch
import sys
import os

def inspect_pth(pth_path):
    if not os.path.exists(pth_path):
        print(f"File not found: {pth_path}")
        return

    print(f"Inspecting: {pth_path}\n")

    try:
        checkpoint = torch.load(pth_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # Case 1: full model object
    if hasattr(checkpoint, 'state_dict'):
        print("This file contains a full model object.\n")
        try:
            print("Model architecture:\n", checkpoint)
            print("\nState dict summary:")
            for name, param in checkpoint.state_dict().items():
                print(f" - {name}: {tuple(param.shape)}")
        except Exception as e:
            print("⚠️ Could not fully print model:", e)

    # Case 2: dictionary checkpoint (common case)
    elif isinstance(checkpoint, dict):
        print("This file contains a checkpoint dictionary.\n")
        print("Top-level keys:", list(checkpoint.keys()), "\n")

        # If it has model weights
        if "model_state_dict" in checkpoint:
            print("Model state dict keys:")
            for name, param in checkpoint["model_state_dict"].items():
                print(f" - {name}: {tuple(param.shape)}")
        else:
            # Might just be a plain state dict
            first_item = next(iter(checkpoint.items()))
            if torch.is_tensor(first_item[1]):
                print("Detected a pure state_dict format.\n")
                for name, param in checkpoint.items():
                    print(f" - {name}: {tuple(param.shape)}")
            else:
                print("Other checkpoint content (not tensors):")
                for key, val in checkpoint.items():
                    print(f" - {key}: {type(val)}")

    else:
        print(f"Unknown file structure: {type(checkpoint)}")


if __name__ == "__main__":

    pth_location ="cnn_model.pth"
    inspect_pth(pth_location)
