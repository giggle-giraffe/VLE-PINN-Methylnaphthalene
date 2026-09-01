# ------------------------------------------------------------------------------------------
# Copyright (c) 2026 Chen Zhang, Tao Li
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

import numpy as np
from loguru import logger
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from model_gg import Model


def init_weights_xavier(m):
    """Xavier/Glorot initialization for better gradient flow with LayerNorm"""
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        init.ones_(m.weight)
        init.zeros_(m.bias)


def init_weights_kaiming(m):
    """Kaiming/He initialization for ReLU activations"""
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        init.ones_(m.weight)
        init.zeros_(m.bias)


class RegNN(nn.Module):
    """Enhanced Neural Network with gradient monitoring, residual connections for regression
    
    Supports automatic normalization of mole fraction outputs to ensure chemical validity:
    - Vapor mole fractions (y*) sum to 1.0
    - Liquid mole fractions (x*) sum to 1.0
    - Vapor fraction (VF) constrained to [0, 1]
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.15, 
                 n_mole_fractions=None, has_vf=False, init_method='kaiming',
                 n_y_components=None, n_x_components=None):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            n_mole_fractions: Legacy parameter (deprecated, use n_y_components and n_x_components)
            has_vf: Whether output includes VF (vapor fraction) as first output
            init_method: Weight initialization method ('xavier', 'kaiming', or 'default')
            n_y_components: Number of vapor mole fraction components (y*)
            n_x_components: Number of liquid mole fraction components (x*), None for reg_y model
        """
        super(RegNN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.projection_layers = nn.ModuleList()  # For residual connections
        self.init_method = init_method
        
        # Store composition information for normalization
        self.n_mole_fractions = n_mole_fractions  # Legacy (for bias initialization only)
        self.has_vf = has_vf  # Whether output includes VF as first element
        self.n_y_components = n_y_components  # Number of y* components
        self.n_x_components = n_x_components  # Number of x* components (None for reg_y)
        
        # Calculate output structure indices for normalization
        # Output structure for reg_x: [VF, y1, y2, ..., yn, x1, x2, ..., xm]
        # Output structure for reg_y: [y1, y2, ..., yn]
        if has_vf and n_y_components is not None:
            # reg_x model: VF + y* + x*
            self.vf_idx = 0
            self.y_start_idx = 1
            self.y_end_idx = 1 + n_y_components
            if n_x_components is not None:
                self.x_start_idx = self.y_end_idx
                self.x_end_idx = self.x_start_idx + n_x_components
            else:
                self.x_start_idx = None
                self.x_end_idx = None
        elif n_y_components is not None:
            # reg_y model: only y*
            self.vf_idx = None
            self.y_start_idx = 0
            self.y_end_idx = n_y_components
            self.x_start_idx = None
            self.x_end_idx = None
        else:
            # Fallback: no normalization
            self.vf_idx = None
            self.y_start_idx = None
            self.y_end_idx = None
            self.x_start_idx = None
            self.x_end_idx = None
        
        # Track gradient statistics
        self.gradient_stats = {}
        
        # Input projection to match first hidden dimension for residual connection
        self.input_projection = nn.Linear(input_dim, hidden_dims[0], dtype=torch.float32)
        prev_dim = hidden_dims[0]
        
        # Hidden layers with layer normalization, dropout, and residual connections
        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.Linear(prev_dim, hidden_dim, dtype=torch.float32)
            self.layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(hidden_dim, dtype=torch.float32))
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            # Add projection layer for residual connection if dimensions don't match
            if prev_dim != hidden_dim:
                projection = nn.Linear(prev_dim, hidden_dim, dtype=torch.float32)
                self.projection_layers.append(projection)
            else:
                self.projection_layers.append(nn.Identity())
            
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim, dtype=torch.float32)
        
        # Apply weight initialization
        self.apply_initialization()
        
        # Improve output layer initialization for better convergence
        with torch.no_grad():
            # Scale output layer weights smaller for more stable initial predictions
            self.output_layer.weight.data *= 0.01
            
            # Initialize output layer bias to target means for unconstrained learning
            if self.n_mole_fractions is not None:
                expected_means = torch.tensor([0.58, 0.42, 0.0007, 0.0006, 0.0007, 0.0007, 0.0007], dtype=torch.float32)
                if len(expected_means) == output_dim:
                    # Initialize bias to expected target means
                    self.output_layer.bias.copy_(expected_means)
        
        # Register hooks for gradient monitoring
        self.register_gradient_hooks()
    
    def apply_initialization(self):
        """Apply selected weight initialization method"""
        if self.init_method == 'xavier':
            self.apply(init_weights_xavier)
        elif self.init_method == 'kaiming':
            self.apply(init_weights_kaiming)
        else:
            # Default PyTorch initialization
            pass
        
        logger.info(f"Applied {self.init_method} initialization")
    
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
        
        # Register hook for input projection
        self.input_projection.register_backward_hook(create_hook_fn("input_proj"))
        
        # Register hooks for hidden layers with unique names
        for i, layer in enumerate(self.layers):
            layer_name = f"hidden_{i+1}"
            layer.register_backward_hook(create_hook_fn(layer_name))
        
        # Register hooks for projection layers (if not Identity)
        for i, proj_layer in enumerate(self.projection_layers):
            if not isinstance(proj_layer, nn.Identity):
                proj_layer.register_backward_hook(create_hook_fn(f"proj_{i+1}"))
        
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
        # Input projection
        x = self.input_projection(x)
        x = F.gelu(x)  # GELU activation
        
        # Process through hidden layers with residual connections
        for i, (layer, ln, dropout, proj_layer) in enumerate(zip(self.layers, self.layer_norms, self.dropouts, self.projection_layers)):
            # Store residual connection
            residual = proj_layer(x)
            
            # Forward through layer
            x = layer(x)
            x = ln(x)
            x = F.gelu(x)  # GELU activation instead of ReLU
            x = dropout(x)
            
            # Add residual connection
            x = x + residual
        
        # Output layer (no residual connection here)
        x = self.output_layer(x)
        
        # Apply normalization constraints to ensure chemical validity
        # Mole fractions must sum to 1.0 within each phase
        x = self._apply_mole_fraction_normalization(x)
        
        return x
    
    def _apply_mole_fraction_normalization(self, x):
        """
        Apply normalization constraints to ensure mole fractions sum to 1.0
        
        Output structure depends on model type:
        - reg_x: [VF, y1, y2, ..., yn, x1, x2, ..., xm]
          - VF: sigmoid to constrain [0, 1]
          - y*: softmax to ensure sum = 1.0
          - x*: softmax to ensure sum = 1.0
        - reg_y: [y1, y2, ..., yn]
          - y*: softmax to ensure sum = 1.0
        """
        # If no normalization indices are set, return raw output
        if self.y_start_idx is None:
            return x
        
        # Build normalized output
        output_parts = []
        
        # Handle VF (vapor fraction) - sigmoid to [0, 1]
        if self.vf_idx is not None:
            vf = torch.sigmoid(x[:, self.vf_idx:self.vf_idx+1])
            output_parts.append(vf)
        
        # Handle y* (vapor mole fractions) - softmax to sum to 1.0
        if self.y_start_idx is not None and self.y_end_idx is not None:
            y_raw = x[:, self.y_start_idx:self.y_end_idx]
            y_normalized = F.softmax(y_raw, dim=1)
            output_parts.append(y_normalized)
        
        # Handle x* (liquid mole fractions) - softmax to sum to 1.0
        if self.x_start_idx is not None and self.x_end_idx is not None:
            x_raw = x[:, self.x_start_idx:self.x_end_idx]
            x_normalized = F.softmax(x_raw, dim=1)
            output_parts.append(x_normalized)
        
        # Concatenate all parts
        if output_parts:
            return torch.cat(output_parts, dim=1)
        else:
            return x


class REG_NN(Model):
    def __init__(self,
                 model_name=None,
                 model_version=None,
                 model_type=None,
                 model_target_vars=None,
                 model_cont_vars=None,
                 random_seed=42,
                 dropout_rate=0.2,
                 l2_reg=0.001,
                 init_method='xavier',
                 learning_rate=None,
                 use_gradient_clipping=False,
                 gradient_clip_value=None):
        """ Enhanced Regression NN Model with gradient monitoring
        
        Args:
            model_name: String denoting model name
            model_version: String denoting model version
            model_type: String denoting model type
            model_cont_vars: List of continuous variables
            model_target_vars: List of target variables
            random_seed: random seed
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization weight
            init_method: Weight initialization method ('xavier', 'kaiming', or 'default')
            learning_rate: Initial learning rate
            use_gradient_clipping: Whether to use gradient clipping
            gradient_clip_value: Maximum gradient norm for clipping
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
        self.init_method = init_method
        self.learning_rate = learning_rate
        self.use_gradient_clipping = use_gradient_clipping
        self.gradient_clip_value = gradient_clip_value
        
        # Determine mole fraction groups from target variables
        n_mole_fractions = []
        has_vf = False
        
        # Count x variables (liquid mole fractions)
        x_vars = [v for v in self.config['model_target_vars'] if v.startswith('x')]
        n_x_components = len(x_vars) if x_vars else None
        if x_vars:
            n_mole_fractions.append(len(x_vars))
        
        # Count y variables (gas mole fractions)
        y_vars = [v for v in self.config['model_target_vars'] if v.startswith('y')]
        n_y_components = len(y_vars) if y_vars else None
        if y_vars:
            n_mole_fractions.append(len(y_vars))
        
        # Check for VF (vapor fraction)
        if 'VF' in self.config['model_target_vars']:
            has_vf = True
        
        # Store component counts for save/load
        self.n_y_components = n_y_components
        self.n_x_components = n_x_components
        
        # Define enhanced model with mole fraction normalization
        self.estimator = RegNN(
            input_dim=len(self.config['model_cont_vars']),
            output_dim=len(self.config['model_target_vars']),
            hidden_dims=[256, 128, 64, 32],
            dropout_rate=self.dropout_rate,
            n_mole_fractions=n_mole_fractions if n_mole_fractions else None,
            has_vf=has_vf,
            init_method=self.init_method,
            n_y_components=n_y_components,
            n_x_components=n_x_components
        )
        
        logger.info(f"Enhanced regression model configured with mole fraction normalization:")
        logger.info(f"  - n_y_components (vapor): {n_y_components}")
        logger.info(f"  - n_x_components (liquid): {n_x_components}")
        logger.info(f"  - has_vf: {has_vf}")

        # -----Set device-----
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.estimator.to(self.device)
        logger.info(f"Using device: {self.device}")
        logger.info(f"Enhanced regression model with {init_method} initialization, LR={learning_rate}")
        if use_gradient_clipping and gradient_clip_value:
            logger.info(f"Gradient clipping enabled with max norm={gradient_clip_value}")
        else:
            logger.info("Gradient clipping disabled - natural gradient flow")

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
        """
        logger.info("Enhanced regression model ready - gradient monitoring enabled")
        return self

    def predict(self, data):
        """
        Run model inference
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch Tensor")

        X_tensor = data.to(self.device)

        # Set model to evaluation mode
        self.estimator.eval()
        self.training = False
        
        # Make predictions
        with torch.no_grad():
            y_pred = self.estimator(X_tensor)
        
        return y_pred

    def save(self, model_output_path):
        """
        Save the enhanced model
        """
        
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
            'init_method': self.init_method,
            'learning_rate': self.learning_rate,
            'use_gradient_clipping': self.use_gradient_clipping,
            'gradient_clip_value': self.gradient_clip_value,
            'n_y_components': self.n_y_components,
            'n_x_components': self.n_x_components,
        }
        torch.save(save_dict, os.path.join(model_output_path, 'model_complete.pth'))
        
        # Save gradient statistics if available
        grad_summary = self.estimator.get_gradient_summary()
        if grad_summary:
            with open(f"{model_output_path}/gradient_stats.json", "w") as f:
                json.dump(grad_summary, f, indent=4, default=str)
        
        logger.info(f"Enhanced model saved to {model_output_path}")

    def load(self, model_input_path):
        """
        Load the enhanced model
        """
        
        checkpoint = torch.load(f"{model_input_path}/model_complete.pth", map_location=self.device)
        
        # Update config and parameters
        self.config = checkpoint['config']
        self.dropout_rate = checkpoint.get('dropout_rate', 0.2)
        self.l2_reg = checkpoint.get('l2_reg', 0.001)
        self.init_method = checkpoint.get('init_method', 'kaiming')
        self.learning_rate = checkpoint.get('learning_rate', 0.0005)
        self.use_gradient_clipping = checkpoint.get('use_gradient_clipping', True)
        self.gradient_clip_value = checkpoint.get('gradient_clip_value', 1.0)
        
        # Determine mole fraction groups from target variables
        n_mole_fractions = []
        has_vf = False
        
        # Count x variables (liquid mole fractions)
        x_vars = [v for v in self.config['model_target_vars'] if v.startswith('x')]
        if x_vars:
            n_mole_fractions.append(len(x_vars))
        
        # Count y variables (gas mole fractions)
        y_vars = [v for v in self.config['model_target_vars'] if v.startswith('y')]
        if y_vars:
            n_mole_fractions.append(len(y_vars))
        
        # Check for VF (vapor fraction)
        if 'VF' in self.config['model_target_vars']:
            has_vf = True
        
        # Get component counts - use saved values if available, otherwise derive from target vars
        # This ensures backward compatibility with models saved before this update
        n_y_components = checkpoint.get('n_y_components', len(y_vars) if y_vars else None)
        n_x_components = checkpoint.get('n_x_components', len(x_vars) if x_vars else None)
        self.n_y_components = n_y_components
        self.n_x_components = n_x_components
        
        # Recreate model with correct architecture and normalization
        self.estimator = RegNN(
            input_dim=len(self.config['model_cont_vars']),
            output_dim=len(self.config['model_target_vars']),
            hidden_dims=[256, 128, 64, 32],
            dropout_rate=self.dropout_rate,
            n_mole_fractions=n_mole_fractions if n_mole_fractions else None,
            has_vf=has_vf,
            init_method=self.init_method,
            n_y_components=n_y_components,
            n_x_components=n_x_components
        ).to(self.device)
        
        # Load weights
        self.estimator.load_state_dict(checkpoint['model_state_dict'])
        self.estimator.eval()
        
        logger.info(f"Enhanced model loaded from {model_input_path}")
        logger.info(f"  - n_y_components: {n_y_components}, n_x_components: {n_x_components}, has_vf: {has_vf}")
        
        return True