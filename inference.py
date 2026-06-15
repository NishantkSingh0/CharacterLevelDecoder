import re
import torch 
from utils import EncoderForInference, Decoding

device = "mps" if torch.backends.mps.is_available() else "cpu"

@torch.no_grad()
def generate(model, startChar, max_new_tokens=50, temperature=0.8):

    model.eval()

    tokens = torch.tensor(
        [EncoderForInference(re.sub(r'[^A-Za-z\s]', '', startChar.lower()))],
        dtype=torch.long
    ).to(device)
    startChar="< "+ startChar
    print(startChar, end="", flush=True)
    for _ in range(max_new_tokens):
        tokens_input = tokens[:, -168:]
        logits = model(tokens_input)
        next_token_logits = logits[:, -1]
        probs = torch.softmax(next_token_logits, dim=-1)
        logits_scaled = next_token_logits / temperature
        probs = torch.softmax(logits_scaled, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        tokens = torch.cat(
            [tokens, next_token.unsqueeze(1)],
            dim=1
        )
        print(Decoding[next_token.item()], end="", flush=True)

        if tokens[0, -1].item() == 28:
            print()
            break