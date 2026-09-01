# ------------------------------------------------------------------------------------------
# Copyright (c) 2026 Chen Zhang, Tao Li
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import tqdm
import copy
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.set_default_dtype(torch.float64)

from model_gg import Model


class ClfNN(nn.Module):
    """Enhanced Neural Network for classification - combines proven architecture with optimized initialization and gradient monitoring"""
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64, 32, 16], dropout_rate=0.3):
        super(ClfNN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Track gradient statistics
        self.gradient_stats = {}
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers with layer normalization and dropout
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim, dtype=torch.float32))
            self.layer_norms.append(nn.LayerNorm(hidden_dim, dtype=torch.float32))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim, dtype=torch.float32)
        
        # Apply optimized initialization for ReLU + improved output scaling
        self.apply_kaiming_initialization()
        self.scale_output_layer()
        
        # Register hooks for gradient monitoring
        self.register_gradient_hooks()
    
    def apply_kaiming_initialization(self):
        """Apply Kaiming/He initialization optimized for ReLU activations"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)
        
        # Initialize LayerNorm weights and biases
        for ln in self.layer_norms:
            init.ones_(ln.weight)
            init.zeros_(ln.bias)
        
        # Initialize output layer with Xavier (better for final classification layer)
        init.xavier_uniform_(self.output_layer.weight, gain=1.0)
        if self.output_layer.bias is not None:
            init.zeros_(self.output_layer.bias)
    
    def scale_output_layer(self):
        """Scale output layer weights to prevent vanishing gradients"""
        with torch.no_grad():
            # Use Xavier/Glorot initialization for output layer instead of aggressive scaling
            # This maintains gradient flow while preventing saturation
            n_in = self.output_layer.in_features
            n_out = self.output_layer.out_features
            limit = np.sqrt(6.0 / (n_in + n_out))
            self.output_layer.weight.data.uniform_(-limit, limit)
            # Small random bias to break symmetry
            self.output_layer.bias.data.uniform_(-0.01, 0.01)
    
    def register_gradient_hooks(self):
        """Register hooks to monitor gradients during backpropagation"""
        def create_hook_fn(layer_name):
            def hook_fn(module, grad_input, grad_output):
                if self.training and grad_output[0] is not None:
                    grad_norm = grad_output[0].norm().item()
                    grad_mean = grad_output[0].mean().item()
                    grad_std = grad_output[0].std().item()
                    
                    if layer_name not in self.gradient_stats:
                        self.gradient_stats[layer_name] = []
                    
                    self.gradient_stats[layer_name].append({
                        'norm': grad_norm,
                        'mean': grad_mean,
                        'std': grad_std
                    })
            return hook_fn
        
        # Register hooks for hidden layers with unique names
        for i, layer in enumerate(self.layers):
            layer_name = f"hidden_{i+1}"
            layer.register_backward_hook(create_hook_fn(layer_name))
        
        # Register hook for output layer
        self.output_layer.register_backward_hook(create_hook_fn("output"))
    
    def get_gradient_summary(self):
        """Get summary of gradient statistics"""
        summary = {}
        for module_name, stats in self.gradient_stats.items():
            if stats:
                recent_stats = stats[-100:]  # Last 100 updates
                summary[module_name] = {
                    'avg_norm': np.mean([s['norm'] for s in recent_stats]),
                    'avg_mean': np.mean([s['mean'] for s in recent_stats]),
                    'avg_std': np.mean([s['std'] for s in recent_stats]),
                    'max_norm': np.max([s['norm'] for s in recent_stats]),
                    'min_norm': np.min([s['norm'] for s in recent_stats])
                }
        return summary
    
    def clear_gradient_stats(self):
        """Clear gradient statistics"""
        self.gradient_stats = {}
        
    def forward(self, x):
        for i, (layer, ln, dropout) in enumerate(zip(self.layers, self.layer_norms, self.dropouts)):
            x = layer(x)
            x = ln(x)
            x = F.relu(x)
            x = dropout(x)
        
        x = self.output_layer(x)
        return x


class CLF_NN(Model):
    def __init__(self,
                 model_name=None,
                 model_version=None,
                 model_type=None,
                 model_target_vars=None,
                 model_cont_vars=None,
                 random_seed=42,
                 dropout_rate=0.3,
                 l2_reg=0.001,
                 learning_rate=0.001):
        """ Classification NN Model Class - Optimized for Unlimited Synthetic Training

        Args:
            model_name: String denoting model name
            model_version: String denoting model version
            model_type: String denoting model type
            model_cont_vars: List of continuous variables
            model_target_vars: List of target variables
            random_seed: random seed
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization weight
            learning_rate: Learning rate for optimizer
        """
        super().__init__(model_name=model_name,
                         model_version=model_version,
                         model_type=model_type,
                         model_cont_vars=model_cont_vars,
                         model_target_vars=model_target_vars,
                         random_seed=random_seed)

        self.best_value = None
        self.best_params = None
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        
        # No scalers needed for unlimited synthetic training
        # Synthetic data is already well-behaved and normalized

        # Define enhanced model
        self.estimator = ClfNN(
            input_dim=len(self.config['model_cont_vars']),
            output_dim=2,  # Binary classification
            hidden_dims=[128, 64, 32, 16],
            dropout_rate=self.dropout_rate
        )

        # -----Set device-----
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.estimator.to(self.device)
        logger.info(f"Using device: {self.device}")

    def get_gradient_info(self):
        """Get current gradient statistics"""
        return self.estimator.get_gradient_summary()
    
    def clear_gradient_info(self):
        """Clear gradient statistics"""
        self.estimator.clear_gradient_stats()

    def fit(self, X, y, hyper_tuning=True, debug=False):
        """
        Placeholder method for compatibility.
        Note: Actual training is handled by the phase training scripts (train_clf.py / train_reg_x.py / train_reg_y.py).
        No scalers needed since synthetic data is already well-behaved.
        """
        logger.info("Classification model ready - no preprocessing needed for synthetic data")
        return self

    def predict(self, data):
        """
        Run model inference (no scaling needed for synthetic data)
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch Tensor")

        # Use data directly - no scaling needed for synthetic data
        X_tensor = data.to(self.device)

        # Set model to evaluation mode
        self.estimator.eval()
        self.training = False
        
        # Make predictions
        with torch.no_grad():
            y_pred = self.estimator(X_tensor)
            _, predicted = torch.max(y_pred, 1)
        
        return predicted

    def predict_proba(self, data):
        """
        Get prediction probabilities (no scaling needed for synthetic data)
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch Tensor")

        # Use data directly - no scaling needed for synthetic data
        X_tensor = data.to(self.device)

        # Set model to evaluation mode
        self.estimator.eval()
        self.training = False
        
        # Make predictions
        with torch.no_grad():
            y_pred = self.estimator(X_tensor)
            probabilities = F.softmax(y_pred, dim=1)
        
        return probabilities

    def save(self, model_output_path):
        """
        Save the model (no scalers needed)
        """
        import os
        
        flag_exist = os.path.exists(model_output_path)
        if not flag_exist:
            os.makedirs(model_output_path)

        # Save config
        with open(f"{model_output_path}/model_config.json", "w", encoding='utf_8') as outfile:
            json.dump(self.config, outfile, ensure_ascii=False, indent=4)

        # Save model state dict and parameters
        save_dict = {
            'model_state_dict': self.estimator.state_dict(),
            'config': self.config,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'learning_rate': self.learning_rate,
        }
        torch.save(save_dict, os.path.join(model_output_path, 'model_complete.pth'))
        
        # Save gradient statistics if available
        grad_summary = self.estimator.get_gradient_summary()
        if grad_summary:
            with open(f"{model_output_path}/gradient_stats.json", "w") as f:
                json.dump(grad_summary, f, indent=4, default=str)
        
        logger.info(f"Enhanced model with diagnostics saved to {model_output_path}")

    def load(self, model_input_path):
        """
        Load the model (no scalers needed)
        """
        import os
        
        checkpoint = torch.load(f"{model_input_path}/model_complete.pth", map_location=self.device)
        
        # Update config and parameters
        self.config = checkpoint['config']
        self.dropout_rate = checkpoint.get('dropout_rate', 0.3)
        self.l2_reg = checkpoint.get('l2_reg', 0.001)
        self.learning_rate = checkpoint.get('learning_rate', 0.001)
        
        # Recreate model with correct architecture
        self.estimator = ClfNN(
            input_dim=len(self.config['model_cont_vars']),
            output_dim=2,  # Binary classification
            hidden_dims=[128, 64, 32, 16],
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Load weights
        self.estimator.load_state_dict(checkpoint['model_state_dict'])
        self.estimator.eval()
        
        logger.info(f"Enhanced model with diagnostics loaded from {model_input_path}")
        
        return True