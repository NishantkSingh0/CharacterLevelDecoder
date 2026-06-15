# Character-Level Decoder-Only Transformer (GPT Style)

A **GPT-style Character-Level Decoder-Only Transformer** built **completely from scratch** using **PyTorch**.

Unlike traditional token-based language models, this model processes **individual characters** as tokens and learns word formation, spelling, grammar, and sentence generation entirely from character sequences.

No pretrained models, Hugging Face architectures, or external Transformer implementations were used. Every component was implemented manually to gain a deeper understanding of modern Large Language Models.

---

# Model Architecture

| Component              | Value                       |
| ---------------------- | --------------------------- |
| Architecture           | Decoder-Only Transformer    |
| Total Parameters       | **4.8 Million**             |
| Decoder Layers         | 6                           |
| Attention Heads        | 16                          |
| Attention Dimension    | 256                         |
| Feed Forward Dimension | 512                         |
| Context Length         | 300 Characters (300 Tokens) |
| Vocabulary Size        | 31 Characters               |

### Vocabulary

```
a-z
'
space
PAD
<
>
```

Since this is a **Character-Level Language Model**, every character corresponds to exactly **one token**.

---

# Dataset
Cleaned book corpus: https://drive.google.com/file/d/10huFLWk3dwL4hmWB6u5o26EPah_teygP/view?usp=sharing
The model was trained on the **BookCorpus** dataset.

### Dataset Statistics
* Approximately **1.5 Million Paragraphs**
* Paragraph Length: **50–300 Characters**
* Character-Level Training

The objective is next-character prediction using causal language modeling.

---

# Features

* Character Embeddings
* Positional Encoding
* Multi-Head Masked Self Attention
* Causal Attention Mask
* Feed Forward Networks
* Residual Connections
* Layer Normalization
* Decoder Blocks
* Autoregressive Text Generation
* Custom Training Pipeline
* Custom Inference Pipeline

Every component has been implemented from scratch.

---

# Current Progress

The model has successfully learned to

* Generate correctly spelled English words
* Learn word boundaries
* Produce meaningful English sentences
* Understand sentence structures
* Generate coherent text from only a few input characters

Although grammar and long-range reasoning are still improving, the model already demonstrates that character-level language modeling can learn language structure without predefined token vocabularies.

---

# Sample Outputs

### Input

```text
hello
```

### Output

```text
hellor remember to you the tenant martin says these days have hoped that i should be everything with you
```

---

### Input

```text
Who Am I
```

### Output

```text
Who Am In the opposite country emergency the desire to extend a late over a received that destroys as we didnt really have the more but its the one that we do when the dragon had exited hard and for the press
```

---

### Input

```text
yo
```

### Output

```text
yon one real person to strike the scope of him whenever he finds his divider he must be spared and out of his accommodation stain are
```

---

### Input

```text
Nishant
```

### Output

```text
Nishanters up was religional how and she took it moving her for the two younger warning the look of the royan and would be nice to be the most skin and my machethell would have come nothing else in it
```

---

# training
[▶️ Watch Demo](./Training.mp4)

# Infererence
[▶️ Watch Demo](./Inferencing.mp4)


# Training

Training was performed on an **NVIDIA A100 GPU**, utilizing nearly **100% GPU usage for approximately 4 hours**.

Special thanks to **Lightning AI** for providing free access to A100 GPUs, enabling independent developers and students to experiment with large-scale deep learning models.

---

# Learning Objectives

This project was built to understand the internal workings of modern decoder-only language models by implementing everything manually.

Topics explored include:

* Character-Level Language Modeling
* Self-Attention
* Multi-Head Attention
* Causal Masking
* Transformer Decoder Architecture
* Positional Encoding
* Feed Forward Networks
* Residual Learning
* Layer Normalization
* Autoregressive Generation
* Cross Entropy Training
* Next Character Prediction

---

# Future Improvements

* Larger model sizes
* Longer context windows
* Better grammatical consistency
* Improved long-range dependencies
* Larger training datasets
* Optimized inference speed
* Mixed Precision Training
* Flash Attention
* KV Cache for faster inference

---

# Repository Structure

```
├── model.py                       - > feeding model Weights inside Model Architecture
├── ModelArch.py                   - > Definition of Original Model Architecture Used in training
├── utils.py                       - > definition of some important functions used in Model Inference
├── inference.py                   - > File That Contain Inference Logics
├── main.py                        - > the main File to Infer Model with Input Text
├── character_transformer.pth      - > Saved Actual Model Weights
├── visualize.py                   - > Some model weights Visualization Logics in matplotlib
├── CharacterLevelDecoder.ipynb    - > The Complete Jupyter File Used While Training
├── attentionweights.png           - > Visualization of Models Attention Weights
├── Figure_1.png                   - > Visualization of Attention embeddings
├── Training.mp4
├── Inferencing.mp4
├── .gitignore
└── README.md
```

---

# Technologies Used

* Python
* PyTorch
* CUDA
* Lightning AI
* NVIDIA A100 GPU

---

# Acknowledgements

Special thanks to **Lightning AI** for providing free A100 GPU resources that made this project possible.