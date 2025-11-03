"""Metrics for analyzing neural network properties."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
from torch.utils.data import DataLoader


class SmoothnessMetrics:
    """Metrics for measuring function smoothness."""
    
    @staticmethod
    def gradient_norm(model: nn.Module, x: torch.Tensor, 
                     device: str = 'cuda') -> float:
        """
        Compute average gradient norm at point x.
        
        Args:
            model: Neural network model
            x: Input tensor (batch of images)
            device: Device to run computation on
        
        Returns:
            Average gradient norm
        """
        model.eval()
        x = x.to(device).requires_grad_(True)
        
        outputs = model(x)
        # Sum of outputs for gradient computation
        outputs.sum().backward()
        
        grad_norms = torch.norm(x.grad.view(x.size(0), -1), dim=1)
        return grad_norms.mean().item()
    
    @staticmethod
    def average_gradient_norm(model: nn.Module, dataloader: DataLoader,
                             n_samples: Optional[int] = None,
                             device: str = 'cuda') -> float:
        """
        Compute average gradient norm over a dataset.
        
        Args:
            model: Neural network model
            dataloader: Data loader
            n_samples: Number of samples to use (None for all)
            device: Device to run computation on
        
        Returns:
            Average gradient norm
        """
        model.eval()
        total_norm = 0.0
        count = 0
        
        for inputs, _ in dataloader:
            inputs = inputs.to(device).requires_grad_(True)
            
            outputs = model(inputs)
            outputs.sum().backward()
            
            grad_norms = torch.norm(inputs.grad.view(inputs.size(0), -1), dim=1)
            total_norm += grad_norms.sum().item()
            count += inputs.size(0)
            
            if n_samples and count >= n_samples:
                break
        
        return total_norm / count
    
    @staticmethod
    def local_lipschitz(model: nn.Module, x: torch.Tensor,
                       radius: float = 0.1, n_samples: int = 100,
                       device: str = 'cuda') -> float:
        """
        Estimate local Lipschitz constant around point x.
        
        Args:
            model: Neural network model
            x: Input tensor (single image or batch)
            radius: Radius of neighborhood to sample
            n_samples: Number of samples to use for estimation
            device: Device to run computation on
        
        Returns:
            Estimated local Lipschitz constant
        """
        model.eval()
        x = x.to(device)
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            f_x = model(x)
        
        max_lipschitz = 0.0
        
        for _ in range(n_samples):
            # Sample perturbation
            perturbation = torch.randn_like(x) * radius
            x_perturbed = x + perturbation
            
            with torch.no_grad():
                f_x_perturbed = model(x_perturbed)
            
            # Compute Lipschitz ratio
            output_diff = torch.norm(f_x_perturbed - f_x, dim=1)
            input_diff = torch.norm(perturbation.view(perturbation.size(0), -1), dim=1)
            
            lipschitz = (output_diff / (input_diff + 1e-8)).max().item()
            max_lipschitz = max(max_lipschitz, lipschitz)
        
        return max_lipschitz
    
    @staticmethod
    def spectral_norm(model: nn.Module) -> float:
        """
        Compute product of spectral norms of weight matrices.
        
        Args:
            model: Neural network model
        
        Returns:
            Product of spectral norms
        """
        product = 1.0
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                if isinstance(module, nn.Conv2d):
                    # Reshape conv weights to 2D matrix
                    weight = weight.view(weight.size(0), -1)
                
                # Compute spectral norm (largest singular value)
                _, s, _ = torch.svd(weight)
                spectral_norm = s[0].item()
                product *= spectral_norm
        
        return product
    
    @staticmethod
    def path_norm(model: nn.Module) -> float:
        """
        Compute product of Frobenius norms along paths.
        
        Args:
            model: Neural network model
        
        Returns:
            Product of norms
        """
        product = 1.0
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                norm = torch.norm(weight).item()
                product *= norm
        
        return product
    
    @staticmethod
    def local_variation(model: nn.Module, x: torch.Tensor,
                       epsilon: float = 0.01, n_samples: int = 100,
                       device: str = 'cuda') -> float:
        """
        Compute local variation: Var[f(x + ε)] for small ε.
        
        Args:
            model: Neural network model
            x: Input tensor
            epsilon: Size of perturbation
            n_samples: Number of samples
            device: Device to run computation on
        
        Returns:
            Local variation (variance)
        """
        model.eval()
        x = x.to(device)
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        outputs = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                perturbation = torch.randn_like(x) * epsilon
                x_perturbed = x + perturbation
                output = model(x_perturbed)
                outputs.append(output)
        
        outputs = torch.stack(outputs)
        variance = torch.var(outputs, dim=0).mean().item()
        
        return variance


class GeneralizationMetrics:
    """Metrics for measuring generalization."""
    
    @staticmethod
    def agreement_rate(model1: nn.Module, model2: nn.Module,
                      dataloader: DataLoader, device: str = 'cuda') -> float:
        """
        Compute percentage of matching predictions between two models.
        
        Args:
            model1: First model
            model2: Second model
            dataloader: Data loader
            device: Device to run computation on
        
        Returns:
            Agreement rate (0 to 100)
        """
        model1.eval()
        model2.eval()
        
        total = 0
        matches = 0
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                
                outputs1 = model1(inputs)
                outputs2 = model2(inputs)
                
                preds1 = outputs1.argmax(dim=1)
                preds2 = outputs2.argmax(dim=1)
                
                matches += (preds1 == preds2).sum().item()
                total += inputs.size(0)
        
        return 100.0 * matches / total
    
    @staticmethod
    def function_distance(model1: nn.Module, model2: nn.Module,
                         dataloader: DataLoader, device: str = 'cuda') -> float:
        """
        Compute MSE between function outputs.
        
        Args:
            model1: First model
            model2: Second model
            dataloader: Data loader
            device: Device to run computation on
        
        Returns:
            Mean squared error
        """
        model1.eval()
        model2.eval()
        
        total_mse = 0.0
        count = 0
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                
                outputs1 = model1(inputs)
                outputs2 = model2(inputs)
                
                mse = torch.mean((outputs1 - outputs2) ** 2)
                total_mse += mse.item() * inputs.size(0)
                count += inputs.size(0)
        
        return total_mse / count
    
    @staticmethod
    def output_correlation(model1: nn.Module, model2: nn.Module,
                          dataloader: DataLoader, device: str = 'cuda') -> float:
        """
        Compute correlation between model outputs.
        
        Args:
            model1: First model
            model2: Second model
            dataloader: Data loader
            device: Device to run computation on
        
        Returns:
            Correlation coefficient
        """
        model1.eval()
        model2.eval()
        
        all_outputs1 = []
        all_outputs2 = []
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                
                outputs1 = model1(inputs)
                outputs2 = model2(inputs)
                
                all_outputs1.append(outputs1.cpu())
                all_outputs2.append(outputs2.cpu())
        
        all_outputs1 = torch.cat(all_outputs1).numpy().flatten()
        all_outputs2 = torch.cat(all_outputs2).numpy().flatten()
        
        correlation = np.corrcoef(all_outputs1, all_outputs2)[0, 1]
        return correlation
    
    @staticmethod
    def test_accuracy(model: nn.Module, dataloader: DataLoader,
                     device: str = 'cuda') -> float:
        """
        Compute test accuracy.
        
        Args:
            model: Neural network model
            dataloader: Data loader
            device: Device to run computation on
        
        Returns:
            Accuracy (0 to 100)
        """
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100.0 * correct / total

