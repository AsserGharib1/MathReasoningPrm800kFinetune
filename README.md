# Process-Supervised Math Reasoning — LLM Fine-Tuning on PRM800K

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AsserGharib1/MathReasoningPrm800kFinetune/blob/main/prm800k_math_reasoning_finetune.ipynb)
[![View on nbviewer](https://img.shields.io/badge/view%20full%20notebook-nbviewer-F37626?logo=jupyter&logoColor=white)](https://nbviewer.org/github/AsserGharib1/MathReasoningPrm800kFinetune/blob/main/prm800k_math_reasoning_finetune.ipynb)


Fine-tuning a causal LLM on **PRM800K** (OpenAI's process-supervision dataset of step-level ratings for mathematical solutions) so the model learns to judge solution steps as correct (+), neutral (0), or incorrect (−).

## Highlights

- **Step-level process supervision**: PRM800K solution steps mapped to +/0/− rating tokens — the same paradigm behind process reward models (PRMs) used to improve LLM reasoning.
- **Parameter-efficient fine-tuning** with LoRA (`peft`) and `trl`, using Hugging Face `transformers` + `datasets`.
- Custom data loading/formatting for PRM800K's phase-1/phase-2 JSONL structure.
- Token-level evaluation of rating predictions with accuracy breakdowns and training curves (outputs preserved).

## Training curves

![Training curves](figures/training_curves.png)

## Repository contents

- `prm800k_math_reasoning_finetune.ipynb` — data preparation, tokenization, LoRA fine-tuning, and evaluation.

## Running

```bash
pip install -r requirements.txt
```

The notebook clones the PRM800K dataset (Lightman et al., *Let's Verify Step by Step*, 2023) and runs on a single GPU (Colab-compatible).
