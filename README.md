# GTN: Meta-Path Based Graph Transformer Network for Recommendation

## Overview

This repository contains an implementation of a meta-path based graph neural network architecture designed for recommendation systems. The project explores how graph neural networks with transformer mechanisms can effectively model relationships between users and items through heterogeneous graph structures and meta-paths.

## Project Motivation

Recommendation systems benefit from understanding complex relationships in heterogeneous information networks (HINs). This project reproduces and extends Graph Transformer Network approaches to leverage meta-path information for improved recommendation performance.

## Architecture & Components

### Core Models
- **GTN (Graph Transformer Network)**: Main implementation utilizing meta-path based graph aggregation with transformer attention mechanisms
- **Baseline Models**: 
  - **DIN**: Deep Interest Network for capturing temporal dynamics in user behavior
  - **NRMS**: Neural Recommendation with Multi-head Self-attention
  - **Wide & Deep**: Classical wide & deep learning architecture for recommendations

### Key Files

| File | Purpose |
|------|---------|
| `train.py` | Training pipeline and model optimization |
| `main.py` | Entry point for running experiments |
| `common_utils.py` | Utility functions for data handling and model evaluation |
| `GTN.ipynb` | Exploratory notebook with GTN implementation details |
| `grapnAggRecommendation.ipynb` | Graph aggregation and recommendation experiments |

### Project Structure

```
GTN/
├── DIN/                          # Deep Interest Network implementation
├── NRMS/                         # Multi-head Self-attention model
├── wide_deep/                    # Wide & Deep model variant
├── preprocess/                   # Data preprocessing utilities
├── data/                         # Dataset storage
├── GTN.ipynb                     # Core GTN notebook
├── grapnAggRecommendation.ipynb  # Graph aggregation experiments
├── train.py                      # Training script
├── main.py                       # Main execution entry point
├── common_utils.py               # Common utilities
├── requirement.txt               # Python dependencies
└── debug.sh                      # Debugging utilities
```

## Key Features

- **Meta-Path Based Aggregation**: Leverages meta-paths in heterogeneous graphs to capture meaningful semantic information
- **Transformer Attention**: Incorporates multi-head self-attention mechanisms for flexible relationship modeling
- **Multiple Baselines**: Includes comparisons with DIN, NRMS, and Wide-Deep architectures
- **Flexible Framework**: Supports experimentation with different graph aggregation strategies and attention patterns
- **Comprehensive Utilities**: Includes data preprocessing, evaluation metrics, and debugging tools

## Dependencies

Key requirements (see `requirement.txt`):
- PyTorch or TensorFlow for neural network implementation
- Graph neural network libraries (DGL, PyTorch Geometric, or similar)
- NumPy, Pandas for data manipulation
- Scikit-learn for evaluation metrics

## Usage

### Installation
```bash
pip install -r requirement.txt
```

### Training
```bash
python train.py
```

### Running Experiments
```bash
python main.py
```

### Interactive Exploration
- For detailed GTN implementation walkthrough: `GTN.ipynb`
- For graph aggregation strategies: `grapnAggRecommendation.ipynb`

## Experiments

The project includes implementations of multiple recommendation architectures:
1. **Graph Transformer Network (GTN)**: Meta-path based attention-enhanced graph networks
2. **Deep Interest Network (DIN)**: Attention mechanism for user interest evolution
3. **Neural Recommendation with Multi-head Self-attention (NRMS)**: Multi-perspective learning
4. **Wide & Deep Learning**: Combined memorization and generalization

## Results & Evaluation

The project evaluates models on standard recommendation metrics:
- Hit Rate (HR@K)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Mean Reciprocal Rank (MRR)

## Future Work

- Scale to larger heterogeneous information networks
- Explore additional meta-paths and their combinations
- Integrate temporal dynamics in graph structure
- Compare with more recent graph neural network architectures

## License

This project is a research reproduction and comparison study.

## Author

[ybxfatfat](https://github.com/ybxfatfat)

---

**Note**: This is a reproduction and extension of Graph Transformer Network concepts for recommendation systems. For the original GTN paper and methodology, refer to the academic literature on graph neural networks and recommendation systems.
