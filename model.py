from ModelArch import CharacterTransformer
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"running on device: {device}")

MODEL = CharacterTransformer().to(device)
MODEL.load_state_dict(
    torch.load("character_transformer.pth", map_location=device)
)