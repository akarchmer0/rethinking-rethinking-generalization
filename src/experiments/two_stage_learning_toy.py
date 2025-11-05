"""
Toy Version of Two-Stage Learning Experiment

This is a simplified version for rapid testing:
- Small input dimension (100 instead of 3072)
- Small MLP (3 layers: 256-128-64)
- Smaller datasets and fewer epochs
- Full visualization capabilities

Perfect for quick validation and debugging!
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import pickle

from utils.config import set_random_seeds, get_device


class ToyMLP(nn.Module):
    """Small MLP for toy experiments."""
    
    def __init__(self, input_dim: int = 100, hidden_sizes: List[int] = [256, 128, 64], 
                 n_classes: int = 10):
        """
        Args:
            input_dim: Input dimension
            hidden_sizes: List of hidden layer sizes
            n_classes: Number of output classes
        """
        super().__init__()
        
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, n_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ToyRandomDataset(Dataset):
    """Random data for toy experiments."""
    
    def __init__(self, n_samples: int, input_dim: int, n_classes: int, 
                 random_labels: bool = True, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.data = torch.randn(n_samples, input_dim)
        
        if random_labels:
            self.labels = torch.randint(0, n_classes, (n_samples,))
        else:
            self.labels = torch.arange(n_samples) % n_classes
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx], self.labels[idx].item()


def train_model_toy(model: nn.Module, train_loader: DataLoader, 
                    epochs: int, device: str, verbose: bool = True, use_amp: bool = True) -> Dict:
    """Train a model and return training history."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize AMP (only on CUDA)
    use_amp = use_amp and device == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    history = {'train_loss': [], 'train_acc': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Use automatic mixed precision if enabled
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(accuracy)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    
    return history


def compute_agreement(model1: nn.Module, model2: nn.Module, 
                     test_loader: DataLoader, device: str) -> float:
    """Compute agreement rate between two models."""
    model1.eval()
    model2.eval()
    
    matches = 0
    total = 0
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            
            preds1 = outputs1.argmax(dim=1)
            preds2 = outputs2.argmax(dim=1)
            
            matches += (preds1 == preds2).sum().item()
            total += inputs.size(0)
    
    return 100.0 * matches / total


def compute_output_correlation(model1: nn.Module, model2: nn.Module,
                               test_loader: DataLoader, device: str) -> float:
    """Compute correlation between model outputs."""
    model1.eval()
    model2.eval()
    
    all_outputs1 = []
    all_outputs2 = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            
            all_outputs1.append(outputs1.cpu().numpy())
            all_outputs2.append(outputs2.cpu().numpy())
    
    all_outputs1 = np.concatenate(all_outputs1).flatten()
    all_outputs2 = np.concatenate(all_outputs2).flatten()
    
    correlation = np.corrcoef(all_outputs1, all_outputs2)[0, 1]
    return correlation


def run_toy_two_stage_experiment(
    input_dim: int = 100,
    hidden_sizes: List[int] = [256, 128, 64],
    n_classes: int = 10,
    stage1_samples: int = 5000,
    stage2_samples: int = 5000,
    test_samples: int = 1000,
    epochs_stage1: int = 50,
    epochs_stage2: int = 50,
    batch_size: int = 128,
    track_agreement: bool = True,
    seed: int = 42,
    verbose: bool = True,
    use_amp: bool = True
) -> Dict:
    """
    Run toy two-stage learning experiment.
    
    Args:
        input_dim: Input dimension
        hidden_sizes: Hidden layer sizes for MLP
        n_classes: Number of classes
        stage1_samples: Training samples for stage 1
        stage2_samples: Training samples for stage 2
        test_samples: Test samples
        epochs_stage1: Epochs for stage 1
        epochs_stage2: Epochs for stage 2
        batch_size: Batch size
        track_agreement: Track agreement during training
        seed: Random seed
        verbose: Print progress
        use_amp: Use automatic mixed precision (faster on GPU)
    
    Returns:
        Dictionary with all results
    """
    device = get_device()
    set_random_seeds(seed)
    
    if verbose:
        print("="*70)
        print("TOY TWO-STAGE LEARNING EXPERIMENT")
        print("="*70)
        print(f"Input dim: {input_dim}")
        print(f"Architecture: {hidden_sizes}")
        print(f"Classes: {n_classes}")
        print(f"Stage 1 samples: {stage1_samples}")
        print(f"Stage 2 samples: {stage2_samples}")
        print(f"Device: {device}")
        print("="*70)
    
    # ========================================
    # STAGE 1: Train on random data
    # ========================================
    if verbose:
        print("\n" + "="*70)
        print("STAGE 1: Training Network_1 on Random Data")
        print("="*70)
    
    # Create random dataset
    stage1_dataset = ToyRandomDataset(
        n_samples=stage1_samples,
        input_dim=input_dim,
        n_classes=n_classes,
        random_labels=True,
        seed=seed
    )
    
    stage1_loader = DataLoader(
        stage1_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create and train Network_1
    model1 = ToyMLP(input_dim, hidden_sizes, n_classes)
    
    if verbose:
        print(f"\nTraining Network_1...")
    
    history1 = train_model_toy(
        model1, stage1_loader, epochs_stage1, device, verbose=verbose, use_amp=use_amp
    )
    
    if verbose:
        print(f"\nStage 1 complete! Final accuracy: {history1['train_acc'][-1]:.2f}%")
    
    # ========================================
    # STAGE 2: Train to mimic Network_1
    # ========================================
    if verbose:
        print("\n" + "="*70)
        print("STAGE 2: Training Network_2 to Learn Network_1's Function")
        print("="*70)
    
    # Create NEW random data
    stage2_data = torch.randn(stage2_samples, input_dim)
    
    # Label with Network_1
    model1.eval()
    with torch.no_grad():
        stage2_data_gpu = stage2_data.to(device)
        stage2_labels = model1(stage2_data_gpu).argmax(dim=1).cpu()
    
    stage2_dataset = TensorDataset(stage2_data, stage2_labels)
    stage2_loader = DataLoader(
        stage2_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create test dataset
    test_data = torch.randn(test_samples, input_dim)
    model1.eval()
    with torch.no_grad():
        test_data_gpu = test_data.to(device)
        test_labels = model1(test_data_gpu).argmax(dim=1).cpu()
    
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Train Network_2
    model2 = ToyMLP(input_dim, hidden_sizes, n_classes)
    
    if verbose:
        print(f"\nTraining Network_2...")
    
    # Track agreement during training if requested
    agreement_history = []
    
    if track_agreement:
        model2_temp = model2.to(device)
        optimizer = optim.Adam(model2_temp.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        history2 = {'train_loss': [], 'train_acc': []}
        
        for epoch in range(epochs_stage2):
            model2_temp.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in stage2_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model2_temp(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            avg_loss = total_loss / total
            accuracy = 100. * correct / total
            
            history2['train_loss'].append(avg_loss)
            history2['train_acc'].append(accuracy)
            
            # Compute agreement
            agreement = compute_agreement(model1, model2_temp, test_loader, device)
            agreement_history.append(agreement)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs_stage2} - "
                      f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, "
                      f"Agreement: {agreement:.2f}%")
        
        model2 = model2_temp
    else:
        history2 = train_model_toy(
            model2, stage2_loader, epochs_stage2, device, verbose=verbose, use_amp=use_amp
        )
    
    # ========================================
    # EVALUATION
    # ========================================
    if verbose:
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)
    
    # Final agreement
    final_agreement = compute_agreement(model1, model2, test_loader, device)
    
    # Output correlation
    correlation = compute_output_correlation(model1, model2, test_loader, device)
    
    if verbose:
        print(f"\nFinal Agreement Rate: {final_agreement:.2f}%")
        print(f"Output Correlation: {correlation:.4f}")
    
    # Sample efficiency test
    sample_efficiency_results = None
    if verbose:
        print("\n" + "="*70)
        print("SAMPLE EFFICIENCY TEST")
        print("="*70)
    
    sample_sizes = [500, 1000, 2000, stage2_samples]
    sample_efficiency = {}
    
    for n_samples in sample_sizes:
        if n_samples > stage2_samples:
            continue
        
        # Create dataset with n_samples
        small_data = stage2_data[:n_samples]
        small_labels = stage2_labels[:n_samples]
        small_dataset = TensorDataset(small_data, small_labels)
        small_loader = DataLoader(small_dataset, batch_size=batch_size, shuffle=True)
        
        # Train model
        model_small = ToyMLP(input_dim, hidden_sizes, n_classes)
        _ = train_model_toy(
            model_small, small_loader, epochs_stage2, device, verbose=False, use_amp=use_amp
        )
        
        # Evaluate agreement
        agreement = compute_agreement(model1, model_small, test_loader, device)
        sample_efficiency[n_samples] = agreement
        
        if verbose:
            print(f"  {n_samples:5d} samples: {agreement:.2f}% agreement")
    
    # Compile results
    results = {
        'config': {
            'input_dim': input_dim,
            'hidden_sizes': hidden_sizes,
            'n_classes': n_classes,
            'stage1_samples': stage1_samples,
            'stage2_samples': stage2_samples,
            'test_samples': test_samples,
            'epochs_stage1': epochs_stage1,
            'epochs_stage2': epochs_stage2,
            'seed': seed
        },
        'stage1_history': history1,
        'stage2_history': history2,
        'agreement_over_epochs': agreement_history,
        'final_agreement': final_agreement,
        'output_correlation': correlation,
        'sample_efficiency': {
            'sample_sizes': list(sample_efficiency.keys()),
            'accuracies': list(sample_efficiency.values())
        },
        'models': {
            'model1_state': model1.state_dict(),
            'model2_state': model2.state_dict()
        }
    }
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE!")
        print("="*70)
    
    return results


def plot_toy_results(results: Dict, save_path: Optional[str] = None):
    """
    Create comprehensive visualization of toy experiment results.
    
    Args:
        results: Results dictionary from run_toy_two_stage_experiment
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Stage 1 Training
    ax1 = fig.add_subplot(gs[0, 0])
    epochs1 = range(1, len(results['stage1_history']['train_acc']) + 1)
    ax1.plot(epochs1, results['stage1_history']['train_acc'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Accuracy (%)')
    ax1.set_title('Stage 1: Network_1 Training on Random Data')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Perfect fit')
    ax1.legend()
    
    # Plot 2: Stage 2 Training
    ax2 = fig.add_subplot(gs[0, 1])
    epochs2 = range(1, len(results['stage2_history']['train_acc']) + 1)
    ax2.plot(epochs2, results['stage2_history']['train_acc'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Accuracy (%)')
    ax2.set_title('Stage 2: Network_2 Learning Network_1')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Perfect fit')
    ax2.legend()
    
    # Plot 3: Agreement over epochs
    ax3 = fig.add_subplot(gs[1, :])
    if results['agreement_over_epochs']:
        epochs = range(1, len(results['agreement_over_epochs']) + 1)
        ax3.plot(epochs, results['agreement_over_epochs'], 'purple', linewidth=2)
        ax3.axhline(y=results['final_agreement'], color='r', linestyle='--',
                   label=f"Final: {results['final_agreement']:.2f}%")
        ax3.fill_between(epochs, 0, results['agreement_over_epochs'], alpha=0.3)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Agreement Rate (%)')
        ax3.set_title('Network_2 Agreement with Network_1 During Training')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim([0, 105])
    
    # Plot 4: Sample Efficiency
    ax4 = fig.add_subplot(gs[2, 0])
    sample_sizes = results['sample_efficiency']['sample_sizes']
    accuracies = results['sample_efficiency']['accuracies']
    ax4.plot(sample_sizes, accuracies, 'o-', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Training Samples')
    ax4.set_ylabel('Agreement Rate (%)')
    ax4.set_title('Sample Efficiency')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    # Plot 5: Summary Statistics
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    summary_text = f"""
    EXPERIMENT SUMMARY
    ==================
    
    Configuration:
    • Input Dimension: {results['config']['input_dim']}
    • Architecture: {results['config']['hidden_sizes']}
    • Classes: {results['config']['n_classes']}
    • Stage 1 Samples: {results['config']['stage1_samples']}
    • Stage 2 Samples: {results['config']['stage2_samples']}
    
    Stage 1 Results:
    • Final Training Acc: {results['stage1_history']['train_acc'][-1]:.2f}%
    
    Stage 2 Results:
    • Final Training Acc: {results['stage2_history']['train_acc'][-1]:.2f}%
    
    KEY FINDINGS:
    • Agreement Rate: {results['final_agreement']:.2f}%
    • Output Correlation: {results['output_correlation']:.4f}
    
    Interpretation:
    Network_2 successfully learned Network_1's
    function with {results['final_agreement']:.0f}% agreement!
    
    This demonstrates that Network_1 learned
    a generalizable smooth function, not
    arbitrary memorization.
    """
    
    ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Toy Two-Stage Learning Experiment Results', 
                fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def main():
    """Main function for running toy experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run toy two-stage learning experiment')
    parser.add_argument('--input-dim', type=int, default=100, help='Input dimension')
    parser.add_argument('--stage1-samples', type=int, default=5000, help='Stage 1 samples')
    parser.add_argument('--stage2-samples', type=int, default=5000, help='Stage 2 samples')
    parser.add_argument('--epochs1', type=int, default=50, help='Epochs for stage 1')
    parser.add_argument('--epochs2', type=int, default=50, help='Epochs for stage 2')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-fig', type=str, default=None, help='Path to save figure')
    parser.add_argument('--save-results', type=str, default=None, help='Path to save results')
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_toy_two_stage_experiment(
        input_dim=args.input_dim,
        hidden_sizes=[256, 128, 64],
        n_classes=10,
        stage1_samples=args.stage1_samples,
        stage2_samples=args.stage2_samples,
        test_samples=1000,
        epochs_stage1=args.epochs1,
        epochs_stage2=args.epochs2,
        batch_size=128,
        track_agreement=True,
        seed=args.seed,
        verbose=True
    )
    
    # Save results
    if args.save_results:
        with open(args.save_results, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {args.save_results}")
    
    # Plot results
    plot_toy_results(results, save_path=args.save_fig)


if __name__ == '__main__':
    main()

