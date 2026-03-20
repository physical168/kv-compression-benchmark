# KV Compression Benchmark

A comprehensive benchmark for evaluating KV cache compression methods on large language models, specifically testing Expected Attention and KVZip compression techniques on Llama 3.1 8B Instruct.

## Overview

This notebook evaluates different KV cache compression algorithms on movie review sentiment analysis and information extraction tasks. It compares the performance and efficiency of various compression ratios.

## Quick Start with Google Colab

### Option 1: Direct Upload
1. Visit [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Upload notebook`
3. Select `main_eval.ipynb`
4. Set Runtime to GPU: `Runtime` → `Change runtime type` → `T4 GPU` or higher

### Option 2: Open from GitHub (Recommended)
Click the badge below to open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/physical168/kv-compression-benchmark/blob/main/main_eval.ipynb)

Or manually:
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Open notebook` → `GitHub` tab
3. Enter: `https://github.com/physical168/kv-compression-benchmark`
4. Select `main_eval.ipynb`

## Prerequisites

### Hugging Face Token
You need a Hugging Face account and token to access the Llama 3.1 model:

1. Create an account at [Hugging Face](https://huggingface.co/)
2. Request access to [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
3. Generate a token at [Settings → Tokens](https://huggingface.co/settings/tokens)
4. In Colab, run Cell 3 and paste your token when prompted

### GPU Requirements
- Recommended: T4 GPU or higher (available in Colab)
- Minimum VRAM: 8GB (for 4-bit quantized model)

## Notebook Structure

1. **Environment Setup**: Clone kvpress repository and install dependencies
2. **Authentication**: Login to Hugging Face Hub
3. **Library Optimization**: Update bitsandbytes for quantization support
4. **Model Loading**: Load Llama 3.1 8B with 4-bit quantization
5. **Query Definition**: Define filter and extraction queries for movie reviews
6. **Compression Testing**: Test Expected Attention and KVZip compression methods
7. **Benchmark Evaluation**: Comprehensive evaluation across multiple compression ratios

## Features

- **Multiple Compression Methods**: Expected Attention, KVZip
- **Flexible Compression Ratios**: Test from 10% to 90% compression
- **Dual Task Types**: 
  - Filter queries (yes/no questions)
  - Extraction queries (information extraction)
- **Performance Metrics**: Accuracy, latency, memory usage comparison

## Usage

Run cells sequentially from top to bottom. The notebook is designed to be self-contained and will:
1. Set up the environment automatically
2. Download and configure the model
3. Run benchmarks with various compression settings
4. Generate comparison results and visualizations

## Dependencies

All dependencies are automatically installed in the notebook:
- `kvpress`: KV cache compression library
- `transformers`: Hugging Face transformers
- `accelerate`: Model acceleration utilities
- `bitsandbytes`: Quantization support
- `torch`: PyTorch deep learning framework

## Results

The notebook generates detailed comparison metrics including:
- Compression ratio vs accuracy trade-offs
- Inference speed improvements
- Memory usage reduction
- Per-query performance analysis

## License

This project is for educational and research purposes.

## Acknowledgments

- [kvpress](https://github.com/GabrieleSanmartino/kvpress) - KV cache compression library
- Meta AI - Llama 3.1 model
- Hugging Face - Model hosting and transformers library
