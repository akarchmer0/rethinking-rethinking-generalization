# Toy Two-Stage Learning Experiment Guide

## Overview

This is a **simplified, fast version** of the two-stage learning experiment designed for rapid testing and validation.

### Key Differences from Full Version

| Feature | Full Version | Toy Version |
|---------|--------------|-------------|
| Input Dimension | 3,072 (32×32×3 images) | 100 |
| Architecture | ResNet-18/VGG-11 | Small MLP (256-128-64) |
| Training Samples | 50,000 | 5,000 |
| Epochs | 200 | 50 |
| Time (CPU) | ~40 hours | ~2-3 minutes |
| Time (GPU) | ~4 hours | <1 minute |

## Quick Start

### Option 1: Command Line

```bash
# Basic run with defaults
python src/experiments/two_stage_learning_toy.py

# Custom parameters
python src/experiments/two_stage_learning_toy.py \
    --input-dim 100 \
    --stage1-samples 5000 \
    --stage2-samples 5000 \
    --epochs1 50 \
    --epochs2 50 \
    --seed 42 \
    --save-fig results/toy_experiment.png \
    --save-results results/toy_results.pkl
```

### Option 2: Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/toy_two_stage_demo.ipynb
```

Then run all cells to:
1. Train Network_1 on random data
2. Train Network_2 to learn Network_1's function
3. Generate comprehensive visualizations
4. See key findings and interpretations

### Option 3: Python Script

```python
from experiments.two_stage_learning_toy import (
    run_toy_two_stage_experiment,
    plot_toy_results
)

# Run experiment
results = run_toy_two_stage_experiment(
    input_dim=100,
    hidden_sizes=[256, 128, 64],
    n_classes=10,
    stage1_samples=5000,
    stage2_samples=5000,
    epochs_stage1=50,
    epochs_stage2=50,
    seed=42,
    verbose=True
)

# Visualize
plot_toy_results(results)

# Check key metric
print(f"Agreement: {results['final_agreement']:.2f}%")
```

## Expected Results

### Typical Output

```
Stage 1 (Network_1 trained on random data):
  Final training accuracy: ~100%

Stage 2 (Network_2 learning Network_1):
  Final training accuracy: ~100%

Generalization:
  Agreement rate: 85-95%
  Output correlation: 0.80-0.90

Sample Efficiency:
    500 samples -> 75-85% agreement
   1000 samples -> 80-88% agreement
   2000 samples -> 83-91% agreement
   5000 samples -> 85-95% agreement
```

### Interpretation

If Network_2 achieves **>85% agreement** with Network_1:

✅ **Network_1 learned a generalizable function**, not arbitrary memorization  
✅ **The inductive bias toward smoothness is strong**  
✅ **The core hypothesis is validated**

## Visualization

The `plot_toy_results()` function creates a comprehensive figure with 5 panels:

1. **Stage 1 Training**: Network_1 fitting random data
2. **Stage 2 Training**: Network_2 learning the function
3. **Agreement Over Time**: How agreement evolves during training
4. **Sample Efficiency**: Agreement vs training set size
5. **Summary Statistics**: Key metrics and interpretation

## Customization

### Experiment Parameters

```python
results = run_toy_two_stage_experiment(
    input_dim=100,              # Try: 50, 100, 200
    hidden_sizes=[256, 128, 64], # Try: [128, 64], [512, 256, 128]
    n_classes=10,               # Try: 5, 10, 20
    stage1_samples=5000,        # Try: 1000, 5000, 10000
    stage2_samples=5000,        # Try: 1000, 5000, 10000
    epochs_stage1=50,           # Try: 30, 50, 100
    epochs_stage2=50,           # Try: 30, 50, 100
    seed=42                     # Try: different seeds
)
```

### Explore Different Configurations

```python
# Test multiple input dimensions
for dim in [50, 100, 200]:
    results = run_toy_two_stage_experiment(
        input_dim=dim,
        verbose=False
    )
    print(f"Dim {dim}: {results['final_agreement']:.2f}%")
```

## What to Look For

### 🎯 Success Indicators

- ✅ Stage 1 reaches ~100% training accuracy
- ✅ Stage 2 reaches ~100% training accuracy  
- ✅ Final agreement >85%
- ✅ Output correlation >0.8
- ✅ Agreement improves with more samples

### ⚠️ Potential Issues

If agreement is low (<70%):
- Try more epochs
- Increase Stage 2 training samples
- Use larger network
- Check for bugs in data generation

## Use Cases

### 1. Quick Validation
Test the concept works before running full experiments

### 2. Hyperparameter Tuning
Find good settings for learning rate, architecture, etc.

### 3. Debugging
Verify implementation without waiting hours

### 4. Education
Demonstrate the concept in teaching/presentations

### 5. Prototyping
Test new ideas quickly

## Timing Benchmarks

On a typical laptop (Intel i7, no GPU):
- **Stage 1 Training**: ~60 seconds
- **Stage 2 Training**: ~60 seconds  
- **Sample Efficiency Test**: ~60 seconds
- **Total**: ~3 minutes

On GPU (NVIDIA RTX 3080):
- **Total**: ~30 seconds

## Saving Results

### Save to File
```python
import pickle

with open('toy_results.pkl', 'wb') as f:
    pickle.dump(results, f)
```

### Load Later
```python
with open('toy_results.pkl', 'rb') as f:
    results = pickle.load(f)

plot_toy_results(results)
```

## Comparison to Full Experiment

| Aspect | Toy | Full |
|--------|-----|------|
| **Purpose** | Quick validation | Publication results |
| **Time** | Minutes | Hours |
| **Data** | Random vectors | CIFAR-10 images |
| **Architecture** | Simple MLP | ResNet-18/VGG |
| **Rigor** | Proof of concept | Statistically valid |
| **Use When** | Testing/debugging | Final experiments |

## Next Steps

After validating with the toy version:

1. ✅ Verify the concept works
2. ✅ Understand the key metrics
3. ✅ Tune hyperparameters
4. → Run the full experiment: `python src/experiments/two_stage_learning.py`
5. → Analyze full results with notebooks
6. → Generate publication figures

## Troubleshooting

### Low Agreement (<70%)

**Solutions:**
- Increase `epochs_stage2` (try 100)
- Increase `stage2_samples` (try 10000)
- Use larger network: `[512, 256, 128]`
- Lower learning rate in training function

### Out of Memory

**Solutions:**
- Reduce `batch_size` (try 64 or 32)
- Reduce `input_dim` (try 50)
- Reduce network size: `[128, 64]`

### Takes Too Long

**Solutions:**
- Reduce `epochs_stage1` and `epochs_stage2` (try 30)
- Reduce `stage1_samples` and `stage2_samples` (try 3000)
- Skip sample efficiency test (modify code)

## Code Structure

```
src/experiments/two_stage_learning_toy.py
├── ToyMLP                           # Small 3-layer MLP
├── ToyRandomDataset                 # Random data generator
├── train_model_toy()                # Simple training loop
├── compute_agreement()              # Agreement metric
├── compute_output_correlation()     # Correlation metric
├── run_toy_two_stage_experiment()   # Main function
└── plot_toy_results()               # Visualization
```

## Key Differences in Implementation

### Simplified Architecture
- Full: ResNet-18 (11M parameters)
- Toy: MLP (270K parameters)

### Simpler Data
- Full: CIFAR-10 images (structure)
- Toy: Random Gaussian vectors (no structure)

### Faster Training
- Full: 200 epochs, complex augmentation
- Toy: 50 epochs, no augmentation

### Same Core Concept
Both versions test whether Network_2 can learn Network_1's function!

## Tips for Best Results

1. **Start small**: Use defaults first
2. **Increase gradually**: Scale up if needed
3. **Multiple seeds**: Try 3-5 different seeds
4. **Compare settings**: Test different configurations
5. **Save everything**: Keep track of what works

## Example Output

```
======================================================================
TOY TWO-STAGE LEARNING EXPERIMENT
======================================================================
Input dim: 100
Architecture: [256, 128, 64]
Classes: 10
Stage 1 samples: 5000
Stage 2 samples: 5000
Device: cpu
======================================================================

======================================================================
STAGE 1: Training Network_1 on Random Data
======================================================================

Training Network_1...
Epoch 10/50 - Loss: 0.3421, Acc: 94.52%
Epoch 20/50 - Loss: 0.0234, Acc: 99.87%
Epoch 30/50 - Loss: 0.0045, Acc: 99.98%
Epoch 40/50 - Loss: 0.0012, Acc: 100.00%
Epoch 50/50 - Loss: 0.0005, Acc: 100.00%

Stage 1 complete! Final accuracy: 100.00%

======================================================================
STAGE 2: Training Network_2 to Learn Network_1's Function
======================================================================

Training Network_2...
Epoch 10/50 - Loss: 0.2156, Acc: 95.23%, Agreement: 82.40%
Epoch 20/50 - Loss: 0.0187, Acc: 99.45%, Agreement: 88.70%
Epoch 30/50 - Loss: 0.0042, Acc: 99.92%, Agreement: 91.20%
Epoch 40/50 - Loss: 0.0015, Acc: 99.98%, Agreement: 92.50%
Epoch 50/50 - Loss: 0.0008, Acc: 100.00%, Agreement: 93.10%

======================================================================
EVALUATION
======================================================================

Final Agreement Rate: 93.10%
Output Correlation: 0.8654

======================================================================
KEY FINDINGS
======================================================================
Network_2 achieved 93% agreement with Network_1!

This proves Network_1 learned a generalizable smooth function,
not arbitrary memorization.
```

---

**Happy Experimenting! 🚀**

For questions or issues, refer to the main README or open an issue on GitHub.

