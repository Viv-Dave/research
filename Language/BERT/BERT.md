# Implementation of BERT from scratch and testing the model against standard GLUE Benchmarks
  
**Date:** 7th October 2025  

---

## 1. Project Overview

This repository contains a PyTorch implementation of a Bidirectional Encoder Rpresentations from Transformers. The primary task as per the paper is to identify the correct token masked, masking 15\% randomly.

This project serves as my first step towards thr Transformer architecture and its usecases.
<!-- 
The core research questions explored are:

- -->
---

## 2. Model Architecture

The model `BERT` is composed of a standard Transformer Encoder based architecture with self attention.

### Encoder

- **Attention Head:** The attention head is a key component of any Transformer based architecture. It has three weight matrices calculating vectors for Query, Key and Value. The output is a weighted sum of all values.
- **MultiHeadAttention:** Multiple attention heads calclate multiple tokens in a sequence.
- **Positional Encoding:** Learnable positional embeddings are added to the patch embeddings to retain spatial information.
- **Transformer Blocks:** The sequence of embeddings is processed by a stack of `4` standard Transformer encoder layers.

## 3. Results & Analysis

After training for `4500` epochs, the model achieved a Cross Entropy Loss of `2.93`. The model was then tested against three standard GLUE benchmarks. It was finetuned for `5` EPOCHS and then evaluated on the validation set.

|Benchmark | MRPC | CoLA | SST-2 |
|:-----------------:|:------:|:------:|:-------:|
|Training Loss| 0.19 | 0.43 | 0.20 |
|Validation Accuracy | 0.69 | 0.65 | 0.75 |

### Key Observations

- **Successes:** .
- **Limitations:** .
- **Impact of Architectural Changes:** .

---

## 4. Setup and Usage

This project uses Python 3.x and PyTorch.
