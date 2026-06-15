

import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size=31, d_model=64, max_seq_len=302):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,d_model)
        self.position_embedding = nn.Embedding(max_seq_len,d_model)

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len,device=x.device).unsqueeze(0)

        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)

        return token_emb + pos_emb


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model=64,num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        B, T, C = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        scores = scores.masked_fill(mask == 0,float('-inf'))

        attention = torch.softmax(scores, dim=-1)

        out = attention @ V
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, C)

        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self,d_model=64,hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=8, hidden_dim=256):
        super().__init__()
        self.attn1 = MultiHeadAttention(d_model,num_heads)
        self.attn2 = MultiHeadAttention(d_model,num_heads)

        self.ffn = FeedForward(d_model,hidden_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.attn1(self.norm1(x),mask)
        x = x + self.attn2(self.norm2(x),mask)
        x = x + self.ffn(self.norm3(x))

        return x


class CharacterTransformer(nn.Module):
    def __init__(self, vocab_size=31, d_model=256, max_seq_len=302, num_layers=6, num_heads=16, hidden_dim=512):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size,d_model,max_seq_len)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model,num_heads,hidden_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model,vocab_size)

    def causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len,device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, T = x.shape
        mask = self.causal_mask(T,x.device)

        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        logits = self.classifier(x)

        return logits