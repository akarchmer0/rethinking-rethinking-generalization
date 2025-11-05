"""Training utilities for neural networks."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Callable
from tqdm import tqdm
import time


class Trainer:
    """Trainer class for neural networks."""
    
    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 0.01, momentum: float = 0.9,
                 weight_decay: float = 5e-4, optimizer_type: str = 'SGD',
                 scheduler_type: Optional[str] = 'cosine', use_amp: bool = True):
        """
        Args:
            model: Neural network model
            device: Device to train on
            learning_rate: Initial learning rate
            momentum: Momentum for SGD
            weight_decay: Weight decay (L2 regularization)
            optimizer_type: Type of optimizer ('SGD' or 'Adam')
            scheduler_type: Type of learning rate scheduler ('cosine', 'step', or None)
            use_amp: Use automatic mixed precision (faster on modern GPUs)
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize optimizer
        if optimizer_type == 'SGD':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                                      momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                                       weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Initialize scheduler
        self.scheduler = None
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=200)
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=0.1)
        
        # Initialize AMP (only on CUDA)
        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epoch_times': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Use automatic mixed precision if enabled
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, test_loader: Optional[DataLoader] = None,
              epochs: int = 200, verbose: bool = True,
              checkpoint_path: Optional[str] = None,
              checkpoint_interval: int = 50) -> Dict:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader (optional)
            epochs: Number of epochs to train
            verbose: Whether to print progress
            checkpoint_path: Path to save checkpoints (optional)
            checkpoint_interval: Save checkpoint every N epochs
        
        Returns:
            Training history dictionary
        """
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Evaluate
            if test_loader is not None:
                test_loss, test_acc = self.evaluate(test_loader)
                self.history['test_loss'].append(test_loss)
                self.history['test_acc'].append(test_acc)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            epoch_time = time.time() - start_time
            self.history['epoch_times'].append(epoch_time)
            
            # Print progress
            if verbose:
                msg = f"Epoch {epoch}/{epochs} - "
                msg += f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
                if test_loader is not None:
                    msg += f", Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
                msg += f" ({epoch_time:.2f}s)"
                print(msg)
            
            # Save checkpoint
            if checkpoint_path is not None and epoch % checkpoint_interval == 0:
                self.save_checkpoint(f"{checkpoint_path}_epoch{epoch}.pt")
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history
        }
        if self.use_amp and self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.use_amp and self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.history = checkpoint.get('history', self.history)


def train_model(model: nn.Module, train_loader: DataLoader, 
                test_loader: Optional[DataLoader] = None,
                epochs: int = 200, learning_rate: float = 0.01,
                device: str = 'cuda', verbose: bool = True,
                checkpoint_path: Optional[str] = None) -> Tuple[nn.Module, Dict]:
    """
    Convenience function to train a model.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader (optional)
        epochs: Number of epochs to train
        learning_rate: Learning rate
        device: Device to train on
        verbose: Whether to print progress
        checkpoint_path: Path to save checkpoints (optional)
    
    Returns:
        Trained model and training history
    """
    device = torch.device(device)
    trainer = Trainer(model, device, learning_rate=learning_rate)
    history = trainer.train(train_loader, test_loader, epochs=epochs,
                          verbose=verbose, checkpoint_path=checkpoint_path)
    return trainer.model, history

