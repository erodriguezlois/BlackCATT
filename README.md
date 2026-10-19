![License](https://img.shields.io/badge/License-Research-green)

# BlackCATT: Black-box Collusion Aware Traitor Tracing in Federated Learning

A Flower/PyTorch implementation of **BlackCATT**, an aggregator-side collusion-resistant watermarking scheme for black box traitor tracing in Federated Learning. BlackCATT enables detection and tracing of malicious clients in federated learning systems through unique watermark embedding and verification through Tardos Codes.

Preprint available in arXiv: https://arxiv.org/abs/2602.12138

## Citation

If you use this code in your research, please cite the corresponding paper:
```bash
@misc{rodríguezlois2026blackcattblackboxcollusionaware,
      title={BlackCATT: Black-box Collusion Aware Traitor Tracing in Federated Learning}, 
      author={Elena Rodríguez-Lois and Fabio Brau and Maura Pintor and Battista Biggio and Fernando Pérez-González},
      year={2026},
      eprint={2602.12138},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2602.12138}, 
}
```

## Overview

This repository contains the code for reproducing the experimental results from the BlackCATT paper. The implementation includes:
- A federated learning framework based on Flower with PyTorch
- Watermark embedding and verification mechanisms
- Support for multiple model architectures (ResNet18, VGG16)
- Evaluation and visualization tools

## Requirements

- **Python**: 3.10.13 (or compatible 3.10.x version)
- **Key Dependencies**: 
  - Flower (Federated Learning Framework) >= 1.15.1
  - PyTorch >= 2.1.1
  - torchvision >= 0.16.1

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Run with the Simulation Engine

In the root directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

This will start a simulation with 20 clients (configurable via `pyproject.toml`) running 1500 rounds (configurable) of federated training.

### Configuration

Key parameters can be adjusted in:
- **`pyproject.toml`**: Flower framework configuration (number of rounds, fraction of clients, local epochs, GPU allocation)
- **`wm_config.py`**: Watermarking scheme parameters and training hyperparameters

## Project Structure

### Core Training Components

- **`server_app.py`**: Server-side federated learning configuration using Flower framework
- **`client_app.py`**: Virtual client configuration for federated training
- **`models.py`**: ResNet18 and VGG16 model implementations
- **`task.py`**: Functions for local main task training
- **`wm_task.py`**: Watermark embedding, verification, traitor tracing, and evaluation metrics
- **`wm_config.py`**: Configuration parameters for watermarking scheme and training
- **`checkpoint.py`**: Checkpoint management for experiment resumption and recovery

### Traitor Tracing Vectors

- **`wm_constants/`**: Directory containing pre-computed Tardos code vectors (`clients_tardos_q_*.csv` and p-bias `p_secret_*.csv`). These vectors can be generated using the generation scripts in `metrics.ipynb`.

### Analysis and Visualization

- **`pfp_randomtest.py`**: Script for training independent non-watermarked models for experimental FPR estimation
- **`metrics.ipynb`**: Jupyter notebook for computing and analyzing performance metrics after training
- **`Graphs.ipynb`**: Jupyter notebook for generating publication-quality graphs from computed metrics

## Usage

### Running Experiments

1. Configure parameters in `wm_config.py` and `pyproject.toml` as needed
2. Run the federated learning simulation:
   ```bash
   flwr run .
   ```
3. Results will be saved to the configured output directory specified in `wm_config.py`

### Analyzing Results

1. Compute performance metrics using the notebook:
   ```bash
   jupyter notebook metrics.ipynb
   ```
2. Generate visualizations:
   ```bash
   jupyter notebook Graphs.ipynb
   ```

## Disclaimer

This is research code developed for the BlackCATT paper. The code may contain bugs or inefficiencies and is provided as-is for reproducibility purposes. We welcome bug reports and contributions.
