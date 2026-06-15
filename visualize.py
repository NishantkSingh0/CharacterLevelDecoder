from model import MODEL
from utils import Decoding
import matplotlib.pyplot as plt

# fig, axes = plt.subplots(4, 4, figsize=(16, 12))
# axes = axes.flatten()

# for i, (name, param) in enumerate(MODEL.named_parameters()):
#     if i >= len(axes): break
#     axes[i].hist(param.detach().cpu().numpy().flatten(), bins=50)
#     axes[i].set_title(name, fontsize=7)

# plt.tight_layout()
# plt.show()


emb = MODEL.embedding.token_embedding.weight.detach().cpu()

plt.figure(figsize=(12, 6))
plt.imshow(emb, aspect="auto", cmap="RdBu")
plt.colorbar()
plt.yticks(range(31), list(Decoding.values()))
plt.title("Token Embedding Weights")
plt.xlabel("d_model dimensions")
plt.show()