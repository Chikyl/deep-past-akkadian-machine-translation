# Deep Past Initiative: Akkadian → English Machine Translation

A baseline neural machine translation pipeline for the **Deep Past Initiative Machine Translation** Kaggle competition. It fine-tunes **ByT5-small** to translate noisy Akkadian transliterations into English, then generates a competition-ready `submission.csv` with beam search.

> Based on the public starter notebooks by Takamichi Toda ([train](https://www.kaggle.com/code/takamichitoda/dpc-starter-train) · [infer](https://www.kaggle.com/code/takamichitoda/dpc-starter-infer)), adapted and run with my own trained weights.

---

## Table of Contents

- [Overview](#overview)
- [Why ByT5?](#why-byt5)
- [Repository Structure](#repository-structure)
- [Pipeline](#pipeline)
- [Configuration](#configuration)
- [Results](#results)

---

## Overview

| Item | Detail |
|---|---|
| **Task** | Translate Akkadian transliteration → English |
| **Competition** | Deep Past Initiative: Machine Translation (Kaggle) |
| **Model** | `google/byt5-small` (byte-level seq2seq Transformer) |
| **Task prefix** | `translate Akkadian to English: ` |
| **Training data** | 1,561 document-level pairs (`train.csv`) |
| **Validation metric** | chrF (via 🤗 `evaluate`), part of the competition metric |
| **Framework** | PyTorch + Hugging Face Transformers (`Seq2SeqTrainer`) |

---

## Why ByT5?

Akkadian transliterations are full of noise, rare spellings, and out-of-vocabulary words. Subword tokenizers (SentencePiece/BPE) fragment these inconsistently. **ByT5 operates directly on UTF-8 bytes**, so:

- There are **no unknown tokens**, and every string is representable.
- It is robust to spelling variation and transliteration conventions.
- It needs no language-specific tokenizer training.

The trade-off is longer sequences, which is why `MAX_LENGTH = 512` is used.

---

## Repository Structure

```
.
├── dpc-starter-train.ipynb    # Data alignment, tokenization, fine-tuning, model saving
├── dpc-starter-infer.ipynb    # Loads the fine-tuned model, runs beam search, writes submission.csv
└── README.md
```

---

## Pipeline

```
train.csv ──► sentence aligner ──► 90/10 split ──► ByT5 tokenizer ──► Seq2SeqTrainer ──► saved model
                                                                                              │
test.csv ──► prefix + tokenize ──► beam search (4 beams) ──► decode ──► submission.csv ◄───────┘
```

### 1. Training (`dpc-starter-train.ipynb`)

1. **Load data** from `train.csv` / `test.csv`.
2. **Simple sentence alignment.** Training rows are whole documents, but test rows are single sentences. The aligner splits a document into sentence pairs *only when* the number of English sentences (split on `.`, `!`, `?`) exactly equals the number of Akkadian lines (split on `\n`). If the counts differ, the original document pair is kept as a safe fallback. Pairs where either side is 3 characters or fewer are dropped as junk.
3. **Train/validation split**: 90% / 10%, seed 42.
4. **Preprocess**: prepend `translate Akkadian to English: ` to each source and tokenize inputs and targets (max 512 tokens, truncation on).
5. **Fine-tune** `google/byt5-small` with `Seq2SeqTrainer`, evaluating chrF at the end of every epoch.
6. **Save** the model and tokenizer to `./byt5-akkadian-model`, which the inference notebook loads.

### 2. Inference (`dpc-starter-infer.ipynb`)

1. Load the fine-tuned model and tokenizer.
2. Prepend the same task prefix to every test transliteration.
3. Generate with **beam search** (`num_beams=4`, `early_stopping=True`, `max_length=512`), batch size 16.
4. Replace any empty output with `"broken text"` so no submission row is blank.
5. Write `submission.csv` with columns `id`, `translation`.

---

## Configuration

### Training

| Hyperparameter | Value |
|---|---|
| Base model | `google/byt5-small` |
| Max sequence length | 512 |
| Epochs | 20 |
| Learning rate | 2e-4 |
| Per-device batch size | 4 |
| Gradient accumulation | 2 (effective batch size 8) |
| Weight decay | 0.01 |
| Precision | **FP32** (`fp16=False`) |
| Eval / save strategy | Every epoch (keep last 2 checkpoints) |
| Seed | 42 |

> **Why FP32?** Mixed-precision (FP16) training of ByT5 is prone to producing `NaN` losses. Training in FP32 avoids this, and the batch size is halved with gradient accumulation to compensate for the extra memory.

### Inference

| Setting | Value |
|---|---|
| Batch size | 16 |
| Beams | 4 |
| Max length | 512 |
| Device | CUDA if available, else CPU |

---

## Results

| Metric | Value |
|---|---|
| Optimizer steps | 3,520 |
| Epochs | 20 |
| Final training loss (mean) | 0.577 |
| Training time | ~84 min (5,065 s) |
| Throughput | 5.54 samples/s |
| Validation chrF | _add from your final epoch eval_ |
| Public leaderboard score | _add after submission_ |

### Sample predictions (test set)

| id | Model output (truncated) |
|---|---|
| 0 | From the Kanesh colony to the messenger of our… |
| 1 | On the tablet from the City when the City was … |
| 2 | Once or twice you hear our messages have settl… |
| 3 | I shall enter the colony to the colony and the… |

The outputs are fluent and capture the Old Assyrian merchant-letter register, but some show repetition (e.g. "the colony to the colony"), typical of a small baseline model.
