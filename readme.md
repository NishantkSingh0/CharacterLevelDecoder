# Character-Level Decoder-Only Transformer

A lightweight GPT-style decoder-only transformer implemented from scratch in PyTorch and trained at the character level.

The model learns autoregressive next-character prediction using causal self-attention, enabling generation of coherent text without relying on pretrained weights or transformer libraries.

## Architecture

* Character-level tokenizer
* Token Embeddings (64-dim)
* Learned Positional Embeddings
* 4 Decoder Blocks
* 4-Head Masked Self-Attention
* Pre-LayerNorm Architecture
* Residual Connections
* Feed Forward Network (64 → 256 → 64)
* Linear Language Modeling Head

**Parameters:** ~282K

## Training

* Objective: Causal Language Modeling
* Loss: Cross Entropy
* Optimizer: AdamW
* Context Length: 168 Tokens
* Batch Size: 128

Training samples are converted into character sequences and optimized using teacher-forced next-token prediction.

## Features

* Decoder-only Transformer implementation from first principles
* Custom multi-head attention implementation
* Causal masking for autoregressive generation
* Character-level language modeling
* End-to-end training and inference pipeline
* Text generation with temperature-based sampling

## Example

```text
Input:
cryvutiuyt90

Output:
cryvutiuyt90 no suckin pled priming...
```

## Motivation

This project was built to understand the internal mechanics of modern autoregressive language models, including attention, masking, residual pathways, normalization, positional representations, and token generation without abstraction layers.
