# QKernels4Molecules

Exploring quantum-inspired feature maps and graph kernels for molecular machine learning. This project implements various quantum-inspired techniques to analyze molecular structures using graph theory and quantum mechanics concepts.

## Project Structure

```
QKernels4Molecules/
├── data/                  # Original molecular datasets
├── features/             # Extracted hybrid graph features
├── results/              # Trained models and performance summaries
├── plots/               # Visualizations of accuracy and performance
├── feature_extraction.py # Feature engineering pipeline
├── main.py              # Main training and evaluation script
├── visualize_results.py # Plot generation and analysis
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Overview

This project implements several quantum-inspired kernel methods for analyzing molecular structures:

1. **Quantum Walk Features (QW)**
   - Time-averaged probability distributions
   - Quantum walk embeddings with configurable time steps
   - Permutation-invariant feature extraction

2. **Spectral Features (Spec)**
   - Normalized Laplacian eigenvalues
   - Algebraic connectivity
   - Spectral gap analysis

3. **Local Structural Features (Local)**
   - Degree distributions
   - Clustering coefficients
   - Triangle counts
   - Node label histograms

4. **Effective Resistance Features (ER) - Optional**
   - Average effective resistance
   - Resistance-based graph statistics

## Installation

```bash
# Clone the repository
git clone https://github.com/Alain-Abraham-hub/QKernels4Molecules.git
cd QKernels4Molecules

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Feature Extraction

The main feature extraction pipeline is implemented in `quantum_kernels.py`. To process a molecular dataset:

```python
from quantum_kernels import process_tudataset

# Configure parameters
qw_times = np.linspace(0.1, 10.0, 20)  # Time points for quantum walk
params = {
    'qw_topk': 20,          # Number of top probabilities to keep
    'spec_m': 10,           # Number of spectral features
    'deg_bins': 10,         # Degree histogram bins
    'clustering_bins': 10,   # Clustering histogram bins
    'target_dim': 256,      # Target dimension after PCA
    'include_er': False     # Whether to include effective resistance features
}

# Process dataset
process_tudataset(
    name='PROTEINS',        # Dataset name
    qw_times=qw_times,
    **params
)
```

### Supported Features

1. **Quantum Walk Features**
   - Time-evolved probability distributions
   - Top-k probability selection
   - Statistical moments (mean, variance, skewness)
   - Entropy-based features

2. **Spectral Analysis**
   - First m eigenvalues of normalized Laplacian
   - Algebraic connectivity (λ₂)
   - Spectral gap
   - Laplacian trace

3. **Local Structure Analysis**
   - Degree distributions
   - Clustering patterns
   - Triangle motifs
   - Node label statistics

## Dataset Support

Currently supports various molecular datasets from TUDataset:
- PROTEINS (protein structures)
- MUTAG (mutagenic compounds)
- NCI1 (cancer research compounds)
- PTC-MR (toxicology compounds)
- AIDS (antiviral compounds)

## Technical Details

### Feature Dimensions

The feature vector φ(G) for each graph consists of:
- QW features: top-k probabilities + 4 statistical measures
- Spectral features: m eigenvalues + 3 graph properties
- Local features: degree and clustering histograms + motif counts
- (Optional) ER features: 2-dimensional resistance statistics

### Computational Complexity

- QW computation: O(n³) for n nodes (eigendecomposition)
- Spectral features: O(n³) for full spectrum
- Local features: O(m) for m edges
- ER features: O(n³) for resistance computations

## Results

The extracted features are saved in NPZ format containing:
- X: Processed feature matrix
- y: Graph labels
- raw_X: Unprocessed features before normalization/PCA

## Future Work

- Implementation of additional quantum kernel methods
- Support for larger molecular datasets
- GPU acceleration for feature computation
- Integration with deep learning frameworks

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- New features
- Documentation improvements
- Performance optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{QKernels4Molecules,
  title = {QKernels4Molecules: Quantum-Inspired Feature Maps for Molecular Machine Learning},
  author = {Alain Abraham},
  year = {2025},
  url = {https://github.com/Alain-Abraham-hub/QKernels4Molecules}
}
```
