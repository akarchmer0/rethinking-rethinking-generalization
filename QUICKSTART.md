# Quick Start Guide

Get started with the rethinking generalization experiments in 5 minutes!

## Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/rethinking-generalization-rebuttal.git
cd rethinking-generalization-rebuttal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Test (5-10 minutes)

Run a quick test to ensure everything works:

```bash
python run_all_experiments.py --quick-test
```

This will:
- Train ResNet-18 on true and random labels (50 epochs)
- Analyze smoothness
- Run two-stage learning experiment
- Complete in ~10 minutes on GPU

## Run Individual Experiments

### 1. Baseline Replication (Zhang et al.)

```bash
python src/experiments/baseline_replication.py
```

**What it does**: Trains networks on CIFAR-10 with true vs random labels

**Expected output**: Both achieve ~0 training error, but random labels get ~10% test accuracy

**Time**: ~2 hours per architecture on GPU

### 2. Smoothness Analysis

```bash
python src/experiments/smoothness_analysis.py
```

**What it does**: Measures how smooth the learned functions are

**Expected output**: Random-label models are less smooth but still structured

**Time**: ~30 minutes

### 3. Two-Stage Learning (KEY EXPERIMENT!)

```bash
python src/experiments/two_stage_learning.py
```

**What it does**: 
- Stage 1: Train Network_1 on random noise with random labels
- Stage 2: Train Network_2 to mimic Network_1's function

**Expected output**: Network_2 achieves >85% agreement with Network_1

**Why it matters**: Proves that Network_1 learned a generalizable smooth function, not arbitrary memorization

**Time**: ~4 hours

### 4. Frequency Analysis

```bash
python src/experiments/frequency_analysis.py
```

**What it does**: Analyzes learned functions in frequency domain

**Time**: ~1 hour

### 5. Progressive Corruption

```bash
python src/experiments/complexity_measures.py
```

**What it does**: Trains with varying levels of label corruption

**Time**: ~2 hours per architecture

## Run All Experiments

```bash
# Full run (several hours)
python run_all_experiments.py

# Quick test (10 minutes)
python run_all_experiments.py --quick-test
```

## Analyze Results

### Option 1: Jupyter Notebooks (Recommended)

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

Interactive exploration of all results.

### Option 2: Generate Figures

```bash
jupyter notebook notebooks/figure_generation.ipynb
```

Creates publication-quality figures.

### Option 3: Python Scripts

```python
import pickle
from src.utils.config import ExperimentConfig

# Load results
with open(ExperimentConfig.get_results_save_path('two_stage_learning'), 'rb') as f:
    results = pickle.load(f)

print(f"Agreement: {results['final_agreement']:.2f}%")
```

## Project Structure

```
├── src/
│   ├── experiments/     # All experiment scripts
│   ├── models/          # Network architectures
│   ├── analysis/        # Metrics and visualization
│   └── utils/           # Configuration and data
├── notebooks/           # Jupyter notebooks
├── results/             # Saved results (created automatically)
└── tests/               # Unit tests
```

## Common Issues

### CUDA Out of Memory

Edit `src/utils/config.py`:
```python
BATCH_SIZE = 64  # Default: 128
```

### Slow Training

- Make sure you're using a GPU: Check `torch.cuda.is_available()`
- Reduce epochs for testing
- Use `--quick-test` mode

### Missing Dependencies

```bash
pip install -r requirements.txt --upgrade
```

## Next Steps

1. **Read the Paper**: See detailed explanations in the paper
2. **Explore Results**: Open `notebooks/exploratory_analysis.ipynb`
3. **Modify Experiments**: Edit configs in `src/utils/config.py`
4. **Add New Experiments**: Follow structure in `src/experiments/`

## Key Results to Expect

| Experiment | Key Finding | Implication |
|------------|-------------|-------------|
| Baseline | Networks fit random labels with 0 train error | Confirms Zhang et al. |
| Smoothness | Random-label models are less smooth but structured | Not arbitrary memorization |
| Two-Stage | Network_2 achieves >85% agreement | Network_1 learned generalizable function |
| Frequency | Random-label models have more high-freq content | But still prefer smooth functions |
| Corruption | Smooth degradation with corruption | Realizability matters but smoothness persists |

## Getting Help

- **Issues**: Check [GitHub Issues](https://github.com/yourusername/rethinking-generalization-rebuttal/issues)
- **Documentation**: See [README.md](README.md) for full documentation
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## Citation

If you use this code:

```bibtex
@software{rethinking_generalization_2024,
  title = {Rethinking Deep Learning Generalization: A Challenge to Zhang et al.},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/rethinking-generalization-rebuttal}
}
```

---

**Happy Experimenting! 🚀**

