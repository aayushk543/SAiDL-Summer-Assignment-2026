# SAiDL Summer Assignment 2026 — Sparsity & Optimization

## Comparing PEFT Methods: LoRA, AdaLoRA, and SoRA on DeBERTaV3

This repository contains the implementation and report for the **SAiDL (Society for Artificial Intelligence and Deep Learning)** Summer Assignment 2026, focused on **Parameter-Efficient Fine-Tuning (PEFT)** methods with an emphasis on sparsity-aware low-rank adaptation.

---

## 📋 Assignment Overview

| Part | Topic | Status |
|------|-------|--------|
| **Part 1** | Comparing LoRA, AdaLoRA, and SoRA on CoLA (DeBERTaV3-base) | ✅ Complete |
| **Part 2** | SGD Subgradient vs Proximal Gradient Descent for SoRA | ✅ Complete |
| **Part 3** | Extending SoRA to Recurrent Architectures (xLSTM, Mamba/SSM) | ✅ Complete |

---

## 🏗️ Repository Structure

```
.
├── README.md                          # This file
├── sparsity-optimization.ipynb        # Main Kaggle notebook (all experiments)
├── report/
│   ├── main.tex                       # LaTeX report source
│   └── main.pdf                       # Compiled report
├── sparsity/
│   ├── config.py                      # Hyperparameter configuration
│   ├── sora.py                        # SoRA, LoRA, AdaLoRA implementations
│   ├── train.py                       # Training loops and evaluation
│   └── data_loader.py                 # CoLA dataset loading
└── SAiDL_Sparsity_Optimization (1).pdf  # Assignment specification
```

---

## 🔬 Methods Implemented

### Part 1: PEFT Method Comparison

| Method | Description | Key Innovation |
|--------|-------------|----------------|
| **LoRA** | Low-Rank Adaptation with fixed rank decomposition ΔW = BA | Frozen base model, trainable low-rank adapters |
| **AdaLoRA** | Adaptive LoRA with SVD triplet parameterization ΔW = PΛQ | Importance-based rank allocation across layers |
| **SoRA** | Sparse Low-Rank Adaptation with learnable gates ΔW = B·diag(g)·A | Proximal gradient descent induces exact sparsity in g |

### Part 2: Optimization Methods for SoRA

- **Proximal Gradient Descent**: Soft-thresholding operator produces **exact zeros** in gate vectors
- **SGD Subgradient**: Standard gradient descent with ℓ₁ subgradient — gates oscillate near zero but **never reach exact zero**
- Analysis of subgradient choice at g=0: sign(0)∈{0, +1, random}

### Part 3: Recurrent Architectures

- **xLSTM (sLSTM)**: Exponential gating variant with SoRA on all gate projections
- **Mamba/SSM**: Diagonal State Space Model with SoRA on input/output and state projections

---

## 📊 Key Results

### Part 1: DeBERTaV3-base on CoLA

| Method | Best MCC | Trainable % | Effective Rank |
|--------|----------|-------------|----------------|
| LoRA (r=8) | 0.3574 | 0.56% | 8/8 |
| AdaLoRA (r=12) | 0.3919 | 0.68% | 6.2/12 |
| SoRA (r=8) | 0.3711 | 0.24% | 1.0/8 |

> SoRA achieves 97% of AdaLoRA's performance with **3× fewer trainable parameters** and automatically discovers that only 1 of 8 rank dimensions is needed.

### Part 2: Proximal vs SGD

| Method | MCC | Exact Zeros in g |
|--------|-----|-----------------|
| Proximal (soft-thresholding) | 0.3711 | 87.50% |
| SGD Subgradient | 0.3384 | 0.00% |

---

## 🛠️ Setup & Reproduction

### Environment
- **Platform**: Kaggle (GPU P100, CUDA 12.x)
- **Base Model**: `microsoft/deberta-v3-base`
- **Dataset**: CoLA (Corpus of Linguistic Acceptability) from GLUE
- **Framework**: PyTorch + HuggingFace Transformers

### Running the Notebook

1. Upload `sparsity-optimization.ipynb` to [Kaggle](https://www.kaggle.com/)
2. Enable **GPU accelerator** (P100 or T4)
3. Run all cells sequentially

### Key Hyperparameters

```python
# Shared
learning_rate = 2e-4
weight_decay = 0.01
max_seq_length = 128
batch_size = 32

# LoRA
lora_rank = 8
lora_alpha = 16

# AdaLoRA
adalora_init_rank = 12
adalora_alpha = 24

# SoRA
sora_rank = 8
sora_lambda = 0.05  # L1 regularization strength
proximal_lr = 1e-2
```

---

## 📝 Report

The full report is available at [`report/main.pdf`](report/main.pdf). It covers:

1. Mathematical derivations of LoRA, AdaLoRA (SVD triplet), and SoRA (gated + proximal)
2. Theoretical comparison of proximal vs SGD subgradient optimization
3. Analysis of subgradient choice and its effect on sparsity
4. Extension to recurrent architectures (xLSTM, Mamba)
5. Comprehensive experimental results and ablation studies

---

## 📚 References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- Zhang et al., *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*, ICML 2023
- Ding et al., *SoRA: Sparse Low-Rank Adaptation of Pre-trained Language Models*, EMNLP 2023
- Beck et al., *xLSTM: Extended Long Short-Term Memory*, NeurIPS 2024
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, ICLR 2024
- He et al., *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*, ICLR 2021

---

## 👤 Author

**Aayush Kushwaha**
SAiDL Summer Assignment 2026
