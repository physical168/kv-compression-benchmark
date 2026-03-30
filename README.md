# KV Compression Benchmark

A comprehensive benchmark for evaluating KV cache compression methods on large language models, specifically testing Expected Attention and KVzip (`KVzipPress`) compression techniques on Llama 3.1 8B Instruct.

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

1. **Environment Setup**: Clone [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) and install in editable mode (works with Colab **transformers 5.x** / `DynamicCache` without patching). If the course links [GabrieleSanmartino/kvpress](https://github.com/GabrieleSanmartino/kvpress), it is the same upstream line; NVIDIA `main` is typically newer.
2. **Step 1b (Colab)**: Run the small monkeypatch cell after install if you see `DynamicCache` / `key_cache` errors with **ExpectedAttention** or **KVzip** under `model.generate`—some stacks mis-detect `QuantizedCache` on the container; the patch branches on per-layer `QuantizedLayer` instead.
3. **Authentication**: Login to Hugging Face Hub
4. **Library Optimization**: Update bitsandbytes for quantization support
5. **Model Loading**: Load Llama 3.1 8B with 4-bit quantization
6. **Query Definition**: Define filter and extraction queries for movie reviews
7. **Compression Testing**: Smoke test Expected Attention vs KVzip (`KVzipPress`)
8. **Benchmark protocol**: Instructor settings — ratios in `[0.2, 0.9]`, optional CSV of ~1000 reviews (sample 5–10%), optional query subsampling; full grid is commented out (slow, especially KVzip)
9. **`analyze_benchmark.ipynb`**: Load `benchmark_runs.csv`, English narrative, and plots (also see `analyze_benchmark.py` for CLI)
10. **`eval_extract.ipynb`**: Colab‑first notebook for **extract tasks only** (`query_010`–`019`). Default data path: **`/content/movie_result/`** (place the CSV files there, or change `MOVIE_RESULTS_DIR`). By default it mounts **Google Drive** and appends each finished row to **`MyDrive/kv-compression-benchmark/extract_eval/extract_predictions_checkpoint.csv`** (with fsync), so disconnects are recoverable with **`RESUME_FROM_CHECKPOINT = True`**. Default **`MAX_ROWS_PER_QUERY = 20`** per CSV; set **`0`** to use **`SAMPLE_FRAC`** only. **`SMOKE_MAX_ROWS=3`** caps further for a quick test. Open in Colab: [`eval_extract.ipynb`](https://colab.research.google.com/github/physical168/kv-compression-benchmark/blob/main/eval_extract.ipynb). Regenerate: `python scripts/generate_eval_extract_notebook.py`.

## Features

- **Multiple Compression Methods**: Expected Attention, KVzip (`KVzipPress` — note the lowercase `z` in `zip`)
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

- [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) - KV cache compression library (upstream; course may reference [GabrieleSanmartino/kvpress](https://github.com/GabrieleSanmartino/kvpress))
- Meta AI - Llama 3.1 model
- Hugging Face - Model hosting and transformers library
