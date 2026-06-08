# SAiDL Summer Assignment 2026

**Society for Artificial Intelligence and Deep Learning**  
Summer 2026 Induction Assignment

---

## Repository Structure

```
SAiDL-Summer-Assignment-2026/
├── README.md
├── LICENSE
└── BITS/
    └── SAiDL_2026/
        ├── core-ml.ipynb
        ├── sparsity-optimization.ipynb
        ├── SAiDL_CoreML(2).pdf
        └── SAiDL_Sparsity_Optimization(1).pdf
```

---

## Overview

This repository contains my submission for the SAiDL Summer 2026 Induction Assignment. I have attempted the following tracks:

- **Core ML** — Long-context sequence modeling on WikiText-2
- **Sparsity & Optimization** — Comparing LoRA, AdaLoRA, and SoRA on CoLA (GLUE) with DeBERTa-v3-base

---

## Core ML

**Task:** Long-context sequence modeling on WikiText-2

**What is implemented:**
- Standard Transformer baseline (Attention Is All You Need)
- Attention variants: Sliding Window, Linear Attention, Multi-Query Attention
- Positional encodings: RoPE, ALiBi, Relative Positional Encoding
- Convolution + Attention hybrids
- Extrapolation experiments (train on 512, test on 512/1024/2048)

**How to run:**  
Open `BITS/SAiDL_2026/core-ml.ipynb` on Kaggle or Google Colab. All dependencies are installed within the notebook.

---

## Sparsity & Optimization

**Task:** Parameter-efficient fine-tuning comparison on CoLA (GLUE) using DeBERTa-v3-base

**What is implemented:**
- **LoRA** — Fixed rank low-rank adaptation
- **AdaLoRA** — Adaptive rank allocation via importance-based SVD pruning
- **SoRA** — Sparse gating with L1 regularization and proximal gradient updates
- SGD subgradient vs proximal soft-thresholding comparison (NumPy + PyTorch)

**How to run:**  
Open `BITS/SAiDL_2026/sparsity-optimization.ipynb` on Kaggle or Google Colab. All dependencies are installed within the notebook.

**Results Summary:**

| Method  | Best MCC | Trainable Params | Effective Rank | Time/Epoch |
|---------|----------|------------------|----------------|------------|
| LoRA    | -        | -                | Fixed          | -          |
| AdaLoRA | -        | -                | Adaptive       | -          |
| SoRA    | -        | -                | Sparse-gated   | -          |

*(Fill in final numbers before submitting)*

---

## Setup

All experiments were run on Kaggle (GPU T4 x2). No local setup required — open the notebooks directly on Kaggle or Colab.

**Key dependencies:**

```
torch
transformers
datasets
numpy
scipy
```

---

## Reports

Detailed analysis, derivations, plots, and results are in the PDF reports inside `BITS/SAiDL_2026/`, written in LaTeX.

---

## Contact

**Aayush Kushwaha**  
SAiDL Summer 2026 Induction Candidate