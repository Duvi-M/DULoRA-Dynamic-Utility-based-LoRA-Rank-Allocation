# DULoRA: Dynamic Utility-based LoRA Rank Allocation

DULoRA is a master's thesis research pipeline for studying **dynamic rank allocation in LoRA-based fine-tuning**. The project compares a fixed-rank LoRA baseline against an adaptive LoRA configuration where ranks are assigned per layer using gradient-based utility scores.

The current implementation focuses on binary sentiment classification with transformer models, using the IMDb dataset as the main benchmark and Hugging Face tooling for model training, evaluation, and dataset handling.

## Research Motivation

Low-Rank Adaptation (LoRA) reduces the number of trainable parameters required to fine-tune large transformer models. However, standard LoRA usually applies the same rank across all target layers, even though not all layers contribute equally to the task.

DULoRA explores the idea that LoRA rank should be allocated dynamically based on layer utility. Instead of assigning a uniform rank everywhere, the pipeline estimates layer importance from gradient signals and distributes a fixed rank budget toward the layers that appear more useful for the downstream task.

## Objectives

- Implement a reproducible LoRA fine-tuning pipeline for transformer-based text classification.
- Compare fixed-rank Baseline LoRA with Adaptive Gradient-Aware LoRA.
- Estimate layer utility using gradient norms from LoRA parameters.
- Allocate per-layer LoRA ranks under a configurable rank budget.
- Save metrics, layer scores, rank patterns, allocation history, and plots for later analysis.
- Provide a clean experiment structure suitable for thesis reporting and future extensions.

## Method Overview

The experiment pipeline follows three main stages:

1. **Baseline LoRA**
   A standard LoRA model is trained with a fixed rank defined by `baseline_lora.r`.

2. **Layer Utility Scoring**
   A temporary scoring model is built with the minimum adaptive rank. Gradients are collected for LoRA parameters over a configurable number of warmup batches. These gradient norms are normalized and used as layer utility scores.

3. **Adaptive LoRA**
   A final LoRA model is built with a `rank_pattern` generated from the layer scores. The rank allocator starts from `min_rank` and increases selected layers by `rank_step` until the configured `total_budget` is reached or no further allocation is possible.

The core allocation logic lives in `src/rank_allocator.py`.

## Project Structure

```text
DULoRA-Dynamic-Utility-based-LoRA-Rank-Allocation/
│
├── configs/
│   ├── default.yaml
│   ├── colab_budget_144.yaml
│   └── colab_budget_192.yaml
│
├── outputs/
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── rank_allocator.py
│   ├── experiment.py
│   ├── plots.py
│   ├── utils.py
│   └── evaluate.py
│
├── run_experiment.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Main Components

| File | Purpose |
| --- | --- |
| `configs/default.yaml` | Default experiment configuration. |
| `configs/colab_budget_144.yaml` | Colab-oriented configuration with adaptive rank budget 144. |
| `configs/colab_budget_192.yaml` | Colab-oriented configuration with adaptive rank budget 192. |
| `src/data.py` | Loads and tokenizes the Hugging Face dataset. |
| `src/model.py` | Builds the base transformer model and applies LoRA through PEFT. |
| `src/rank_allocator.py` | Collects gradient-based layer scores and allocates adaptive ranks. |
| `src/experiment.py` | Orchestrates baseline training, scoring, adaptive training, and output saving. |
| `src/evaluate.py` | Defines evaluation metrics for Hugging Face `Trainer`. |
| `src/plots.py` | Saves CSV files and plots for analysis. |
| `src/utils.py` | Handles config loading, seed control, device selection, and parameter counting. |
| `run_experiment.py` | Command-line entrypoint for running experiments. |

## Installation

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project depends on:

- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- Accelerate
- Evaluate
- Scikit-learn
- Matplotlib
- Pandas
- PyYAML

## Running an Experiment

Run the default experiment:

```bash
python run_experiment.py --config configs/default.yaml
```

Run a specific budget configuration:

```bash
python run_experiment.py --config configs/colab_budget_144.yaml
```

```bash
python run_experiment.py --config configs/colab_budget_192.yaml
```

## Configuration

The default configuration uses this structure:

```yaml
seed: 42
experiment_name: run_001
output_dir: outputs

dataset:
  name: imdb
  train_size: 2000
  test_size: 1000
  max_length: 256

model:
  name: bert-base-uncased
  num_labels: 2

training:
  epochs: 2
  batch_size: 8
  learning_rate: 0.00005
  logging_steps: 50

lora:
  alpha: 16
  dropout: 0.1
  target_modules:
    - query
    - value

baseline_lora:
  r: 8

adaptive_lora:
  algorithm: gradient_aware
  min_rank: 2
  max_rank: 8
  rank_step: 2
  total_budget: 96
  warmup_steps: 20
```

### Output Directory Behavior

Experiments are saved in a dedicated run directory.

If `experiment_name` is provided:

```yaml
experiment_name: budget_144
output_dir: outputs
```

the run is saved to:

```text
outputs/budget_144/
```

If `experiment_name` is omitted, the pipeline creates a timestamped folder:

```text
outputs/run_2026-05-08_15-30-20/
```

## Generated Outputs

Each experiment run saves the following artifacts inside its final output directory:

```text
outputs/<experiment_name>/
│
├── metrics.csv
├── layer_scores.csv
├── rank_pattern.csv
├── allocation_history.csv
│
├── layer_scores.png
├── rank_pattern.png
├── budget_history.png
│
├── baseline_lora/
└── adaptive_lora/
```

### Output Files

| File | Description |
| --- | --- |
| `metrics.csv` | Baseline and adaptive accuracy, loss, and trainable parameter counts. |
| `layer_scores.csv` | Normalized gradient-based utility score for each LoRA layer. |
| `rank_pattern.csv` | Final adaptive rank assigned to each layer. |
| `allocation_history.csv` | Iteration-by-iteration record of budget allocation. |
| `layer_scores.png` | Bar plot of layer utility scores. |
| `rank_pattern.png` | Bar plot of assigned adaptive ranks. |
| `budget_history.png` | Plot of rank budget usage over allocation iterations. |
| `baseline_lora/` | Hugging Face `Trainer` output directory for Baseline LoRA. |
| `adaptive_lora/` | Hugging Face `Trainer` output directory for Adaptive LoRA. |

## Reproducibility

The pipeline uses the configured `seed` to improve reproducibility:

- Dataset shuffling uses the configured seed.
- Baseline, scoring, and adaptive model initialization reset the seed before model construction.
- Hugging Face `TrainingArguments` receives both `seed` and `data_seed`.
- Output folders are isolated by `experiment_name` or timestamp to avoid mixing results across runs.

Exact reproducibility can still depend on hardware, backend behavior, CUDA/cuDNN settings, and library versions.

## Current Evaluation

The current evaluation reports:

- Accuracy
- Evaluation loss
- Training time
- Trainable parameter count
- Total parameter count

These values are returned by the experiment pipeline and saved to `metrics.csv`.

## Figures and Results

This section is reserved for thesis figures and experiment summaries.

Suggested figures to include later:

- Layer utility scores
- Adaptive rank distribution
- Rank budget allocation history
- Accuracy comparison across budgets
- Trainable parameter comparison
- Loss comparison between Baseline LoRA and Adaptive LoRA

Example Markdown placeholders:

```md
![Layer scores](outputs/run_001/layer_scores.png)
![Rank pattern](outputs/run_001/rank_pattern.png)
![Budget history](outputs/run_001/budget_history.png)
```

## Example Research Questions

- Can gradient-based utility scores identify layers that benefit from higher LoRA rank?
- Does adaptive rank allocation improve performance under the same or similar trainable parameter budget?
- How does the total rank budget affect accuracy, loss, and parameter efficiency?
- Are rank allocations stable across seeds, dataset sizes, or model architectures?

## Future Work

Possible extensions include:

- Testing additional datasets beyond IMDb.
- Evaluating other transformer backbones such as RoBERTa or DistilBERT.
- Adding multiple seeds and confidence intervals.
- Comparing different utility functions for rank allocation.
- Logging experiments with Weights & Biases or TensorBoard.
- Extending evaluation metrics beyond accuracy.
- Adding automated experiment tables for thesis reporting.

## Author

Duvan Mendoza  
MSc Software Engineering and Big Data  
MEPhI - Moscow Engineering Physics Institute

Research focus:

- Machine Learning
- Natural Language Processing
- Transformer Models
- Parameter-Efficient Fine-Tuning
- LoRA Rank Allocation
