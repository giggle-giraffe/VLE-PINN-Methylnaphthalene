#!/usr/bin/env python
# ------------------------------------------------------------------------------------------
# Copyright (c) 2026 Chen Zhang, Tao Li
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------
import setproctitle
import os
import argparse
import yaml
import json
import time
from datetime import datetime

import numpy as np
import torch
# Set matplotlib backend for headless servers (must be before pyplot import)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster/server environments
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger
import platform

from reg_nn import REG_NN
from plot_util import plot_loss_history, plot_calibration, plot_individual_gradient_flow
from data_util_hdf5 import (HDF5ThermoRegressionDataset, InMemoryThermoRegressionDataset,
                             load_pretrained_scalers)
from util import (EarlyStopping, get_optimizer, get_scheduler, set_random_seeds,
                  load_checkpoint, save_checkpoint)

torch.set_default_dtype(torch.float32)


class MoleFractionLoss(nn.Module):
    """
    Custom loss function for mole fraction regression with:
    - Standard MSE for VF (vapor fraction) - range [0, 1]
    - Log-space MSE for mole fractions - handles multi-scale (1e-6 to 1.0)
    
    Output structure for reg_x: [VF, y1, y2, ..., yn, x1, x2, ..., xm]
    - VF at index 0: sigmoid output, standard MSE loss
    - y* at indices 1 to n_y: softmax output, log-MSE loss
    - x* at indices n_y+1 to end: softmax output, log-MSE loss
    """
    def __init__(self, n_y_components, n_x_components=None, has_vf=True, epsilon=1e-8, 
                 vf_weight=1.0, mf_weight=1.0):
        """
        Args:
            n_y_components: Number of vapor mole fraction components
            n_x_components: Number of liquid mole fraction components (None for reg_y)
            has_vf: Whether VF is included as first output (True for reg_x, False for reg_y)
            epsilon: Small constant for log stability
            vf_weight: Weight for VF loss term
            mf_weight: Weight for mole fraction loss terms
        """
        super().__init__()
        self.n_y_components = n_y_components
        self.n_x_components = n_x_components
        self.has_vf = has_vf
        self.epsilon = epsilon
        self.vf_weight = vf_weight
        self.mf_weight = mf_weight
        
        # Calculate indices
        if has_vf:
            self.vf_idx = 0
            self.y_start = 1
            self.y_end = 1 + n_y_components
        else:
            self.vf_idx = None
            self.y_start = 0
            self.y_end = n_y_components
        
        if n_x_components is not None:
            self.x_start = self.y_end
            self.x_end = self.x_start + n_x_components
        else:
            self.x_start = None
            self.x_end = None
    
    def forward(self, pred, target):
        """
        Compute combined loss:
        - VF: standard MSE
        - Mole fractions: MSE in log space
        """
        total_loss = 0.0
        
        # VF loss (standard MSE)
        if self.has_vf and self.vf_idx is not None:
            vf_pred = pred[:, self.vf_idx]
            vf_target = target[:, self.vf_idx]
            vf_loss = torch.mean((vf_pred - vf_target) ** 2)
            total_loss = total_loss + self.vf_weight * vf_loss
        
        # Vapor mole fractions (y*) - log-space MSE
        y_pred = pred[:, self.y_start:self.y_end]
        y_target = target[:, self.y_start:self.y_end]
        # Clamp to avoid log(0)
        y_pred_safe = torch.clamp(y_pred, min=self.epsilon, max=1.0)
        y_target_safe = torch.clamp(y_target, min=self.epsilon, max=1.0)
        y_loss = torch.mean((torch.log(y_pred_safe) - torch.log(y_target_safe)) ** 2)
        total_loss = total_loss + self.mf_weight * y_loss
        
        # Liquid mole fractions (x*) - log-space MSE (if present)
        if self.x_start is not None and self.x_end is not None:
            x_pred = pred[:, self.x_start:self.x_end]
            x_target = target[:, self.x_start:self.x_end]
            # Clamp to avoid log(0)
            x_pred_safe = torch.clamp(x_pred, min=self.epsilon, max=1.0)
            x_target_safe = torch.clamp(x_target, min=self.epsilon, max=1.0)
            x_loss = torch.mean((torch.log(x_pred_safe) - torch.log(x_target_safe)) ** 2)
            total_loss = total_loss + self.mf_weight * x_loss
        
        return total_loss


def train_epoch_efficient(model, dataloader, criterion, optimizer, device, dataset, progress_log_interval=10):
    """Memory-efficient training for one epoch - computes metrics incrementally"""
    model.train()
    # Use GPU tensor for loss accumulator to avoid sync every batch
    running_loss = torch.tensor(0.0, device=device)
    running_samples = 0
    
    # For detailed metrics, store limited samples ON GPU (transfer once at end)
    max_samples_for_metrics = 10000
    collected_preds_gpu = []
    collected_targets_gpu = []
    samples_collected = 0
    
    # Configurable progress tracking for cluster logging
    total_batches = len(dataloader)
    log_interval = max(1, total_batches * progress_log_interval // 100)
    
    logger.info(f"Starting training epoch with {total_batches} batches...")
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Incremental statistics
        batch_size = targets.size(0)
        running_loss += loss.detach() * batch_size  # Keep on GPU, no sync!
        running_samples += batch_size
        
        # Collect limited samples on GPU (no R2 computation per batch - wasteful!)
        if samples_collected < max_samples_for_metrics:
            remaining_slots = max_samples_for_metrics - samples_collected
            samples_to_take = min(batch_size, remaining_slots)
            
            collected_preds_gpu.append(outputs[:samples_to_take].detach())
            collected_targets_gpu.append(targets[:samples_to_take])
            samples_collected += samples_to_take
        
        # Progress logging
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            progress_pct = (batch_idx + 1) / total_batches * 100
            avg_loss = (running_loss / running_samples).item() if running_samples > 0 else 0
            logger.info(f"Training: {progress_pct:.1f}% ({batch_idx+1}/{total_batches}) - "
                       f"Loss: {loss.item():.6f}, Avg Loss: {avg_loss:.6f}")
    
    # Final metrics - convert GPU tensor to Python float (single sync at end!)
    epoch_loss = (running_loss / running_samples).item() if running_samples > 0 else 0.0
    
    # Transfer collected samples to CPU once and compute metrics
    if collected_preds_gpu:
        sample_preds_scaled = torch.cat(collected_preds_gpu).cpu().numpy()
        sample_targets_scaled = torch.cat(collected_targets_gpu).cpu().numpy()
        
        # Convert to original scale for final metrics
        sample_preds_original = dataset.inverse_transform_targets(sample_preds_scaled)
        sample_targets_original = dataset.inverse_transform_targets(sample_targets_scaled)
        
        # Calculate final R2 on original scale (computed once instead of 24,999 times!)
        final_r2 = r2_score(sample_targets_original, sample_preds_original, multioutput='uniform_average')
    else:
        sample_preds_original = np.array([])
        sample_targets_original = np.array([])
        final_r2 = 0.0
    
    return epoch_loss, final_r2, sample_preds_original, sample_targets_original


def validate_epoch_efficient(model, dataloader, criterion, device, dataset, validation_log_interval=20):
    """Memory-efficient validation for one epoch - computes metrics incrementally"""
    model.eval()
    # Use GPU tensor for loss accumulator to avoid sync every batch
    running_loss = torch.tensor(0.0, device=device)
    running_samples = 0
    
    # For detailed metrics, store limited samples ON GPU (transfer once at end)
    max_samples_for_metrics = 20000  # Slightly more for validation
    collected_preds_gpu = []
    collected_targets_gpu = []
    samples_collected = 0
    
    with torch.no_grad():
        total_batches = len(dataloader)
        log_interval = max(1, total_batches * validation_log_interval // 100)
        
        logger.info(f"Starting validation with {total_batches} batches...")
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Incremental statistics
            batch_size = targets.size(0)
            running_loss += loss.detach() * batch_size  # Keep on GPU, no sync!
            running_samples += batch_size
            
            # Collect limited samples on GPU (no transfer until end)
            if samples_collected < max_samples_for_metrics:
                remaining_slots = max_samples_for_metrics - samples_collected
                samples_to_take = min(batch_size, remaining_slots)
                
                collected_preds_gpu.append(outputs[:samples_to_take])
                collected_targets_gpu.append(targets[:samples_to_take])
                samples_collected += samples_to_take
            
            # Progress logging
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                progress_pct = (batch_idx + 1) / total_batches * 100
                avg_loss = (running_loss / running_samples).item() if running_samples > 0 else 0
                logger.info(f"Validation: {progress_pct:.1f}% ({batch_idx+1}/{total_batches}) - "
                           f"Loss: {loss.item():.6f}, Avg Loss: {avg_loss:.6f}")
    
    # Final metrics on collected samples - convert GPU tensor to Python float (single sync at end!)
    epoch_loss = (running_loss / running_samples).item() if running_samples > 0 else 0.0
    
    # Transfer to CPU once and compute metrics
    if collected_preds_gpu:
        all_preds_scaled = torch.cat(collected_preds_gpu).cpu().numpy()
        all_targets_scaled = torch.cat(collected_targets_gpu).cpu().numpy()
        
        # Convert to original scale for metrics
        all_preds_original = dataset.inverse_transform_targets(all_preds_scaled)
        all_targets_original = dataset.inverse_transform_targets(all_targets_scaled)
        
        # Calculate metrics on original scale
        r2 = r2_score(all_targets_original, all_preds_original, multioutput='uniform_average')
        mse = mean_squared_error(all_targets_original, all_preds_original)
        mae = mean_absolute_error(all_targets_original, all_preds_original)
    else:
        all_preds_original = np.array([])
        all_targets_original = np.array([])
        r2 = 0.0
        mse = float('inf')
        mae = float('inf')
    
    return epoch_loss, r2, mse, mae, all_preds_original, all_targets_original


def train_model(config, model_save_path):
    """Main training function"""
    
    # Set random seeds for reproducibility (using centralized function)
    set_random_seeds(config['random_seed'])
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Construct HDF5 data file path
    data_file = config['data']['train_file']
    if not data_file.endswith('.h5'):
        raise ValueError(f"Only HDF5 format (.h5) is supported. Got: {data_file}")
    
    if not os.path.isabs(data_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            data_file,
            os.path.join('data', data_file),
            os.path.join(script_dir, '..', 'data', data_file),
            os.path.join(script_dir, '..', '..', 'data', data_file),
        ]
        
        data_file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_file_path = path
                break
        
        if data_file_path is None:
            raise FileNotFoundError(
                f"Could not find HDF5 data file '{data_file}' in any of these locations:\n" +
                "\n".join(f"  - {os.path.abspath(p)}" for p in possible_paths)
            )
    else:
        data_file_path = data_file
    
    logger.info(f"Using HDF5 data file: {os.path.abspath(data_file_path)}")
    
    # Check if pre-fitted scalers exist
    # Scaler path structure: phase/scalers/{system}/{version}/
    system = config['data']['system']
    model_version = config['model']['version']
    scaler_dir = os.path.join('phase/scalers', system, model_version)
    use_pretrained_scalers = os.path.exists(os.path.join(scaler_dir, 'reg_x_feature_scaler.pkl'))
    
    # Get data loading mode
    data_loading_mode = config['data']['data_loading_mode']
    logger.info(f"Data loading mode: {data_loading_mode}")
    
    # Check for v3.0 skip_target_scaling mode
    skip_target_scaling = config['data'].get('skip_target_scaling', False)
    if skip_target_scaling:
        logger.info("v3.0 mode: skip_target_scaling=True (model outputs normalized via softmax)")
    
    if data_loading_mode == 'classic':
        # Classic mode: Load entire dataset into memory/GPU
        classic_device = config['data']['classic_mode_device']
        logger.info(f"Classic mode: Loading entire dataset into {classic_device}")
        
        if use_pretrained_scalers and not skip_target_scaling:
            logger.info("Loading pre-fitted scalers...")
            feature_scaler, target_scaler = load_pretrained_scalers(scaler_dir, 'reg_x')
            
            full_dataset = InMemoryThermoRegressionDataset(
                hdf5_file=data_file_path,
                target_cols=config['data']['target_cols'],
                phase_filter='two_phase',  # For liquid phase (x*, y*, VF)
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                fit_scalers=False,
                device=classic_device,
                pin_memory=True,
                skip_target_scaling=skip_target_scaling
            )
        elif skip_target_scaling:
            # v3.0: Only load feature scaler, skip target scaler
            logger.info("v3.0 mode: Loading feature scaler only (target scaling skipped)...")
            feature_scaler_path = os.path.join(scaler_dir, 'reg_x_feature_scaler.pkl')
            if os.path.exists(feature_scaler_path):
                import pickle
                with open(feature_scaler_path, 'rb') as f:
                    feature_scaler = pickle.load(f)
                fit_scalers = False
            else:
                feature_scaler = None
                fit_scalers = True
                logger.info("Feature scaler not found, will fit new scaler")
            
            full_dataset = InMemoryThermoRegressionDataset(
                hdf5_file=data_file_path,
                target_cols=config['data']['target_cols'],
                phase_filter='two_phase',  # For liquid phase (x*, y*, VF)
                feature_scaler=feature_scaler,
                target_scaler=None,  # Not used in v3.0
                fit_scalers=fit_scalers,
                device=classic_device,
                pin_memory=True,
                skip_target_scaling=True
            )
        else:
            logger.info("Pre-fitted scalers not found. Fitting new scalers...")
            logger.info("TIP: Run 'python phase/src/fit_scalers.py' first to pre-fit scalers")
            
            full_dataset = InMemoryThermoRegressionDataset(
                hdf5_file=data_file_path,
                target_cols=config['data']['target_cols'],
                phase_filter='two_phase',  # For liquid phase (x*, y*, VF)
                fit_scalers=True,
                device=classic_device,
                pin_memory=True,
                skip_target_scaling=skip_target_scaling
            )
    else:
        # HDF5 mode: Lazy loading (default)
        if use_pretrained_scalers:
            logger.info("Loading pre-fitted scalers...")
            feature_scaler, target_scaler = load_pretrained_scalers(scaler_dir, 'reg_x')
            
            # Load HDF5 dataset with pre-fitted scalers
            full_dataset = HDF5ThermoRegressionDataset(
                hdf5_file=data_file_path,
                target_cols=config['data']['target_cols'],
                phase_filter='two_phase',  # For liquid phase (x*, y*, VF)
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                fit_scalers=False  # Use pre-fitted scalers
            )
        else:
            logger.info("Pre-fitted scalers not found. Fitting new scalers...")
            logger.info("TIP: Run 'python phase/src/fit_scalers.py' first to pre-fit scalers")
            
            # Load HDF5 dataset and fit scalers
            full_dataset = HDF5ThermoRegressionDataset(
                hdf5_file=data_file_path,
                target_cols=config['data']['target_cols'],
                phase_filter='two_phase',  # For liquid phase (x*, y*, VF)
                fit_scalers=True  # Fit scalers on the dataset
            )
        
        # Ensure HDF5 file handle is open for training
        full_dataset.open()
    
    logger.info(f"Dataset loaded successfully:")
    logger.info(f"  Total samples: {len(full_dataset):,}")
    logger.info(f"  Input columns: {full_dataset.input_cols}")
    logger.info(f"  Target columns: {full_dataset.target_cols}")
    logger.info(f"  Phase filter: {full_dataset.phase_filter}")
    
    # Split indices manually
    dataset_size = len(full_dataset)
    train_size = int(config['data']['train_val_split'] * dataset_size)
    
    # Create indices and shuffle
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets using indices
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    logger.info(f"Training samples: {len(train_subset):,}, Validation samples: {len(val_subset):,}")

    # Create data loaders optimized for HDF5 random access
    # Determine number of workers based on platform and device
    system = platform.system()

    # Default number of workers
    default_num_workers = config['data']['num_workers']

    # IMPORTANT: CUDA tensors cannot be shared across processes
    # When using classic mode with GPU, we MUST use num_workers=0
    if data_loading_mode == 'classic' and config['data']['classic_mode_device'] == 'cuda':
        num_workers = 0
        logger.warning("Classic mode with GPU: setting num_workers=0 (CUDA tensors cannot be shared across processes)")
    # Check if we're on macOS or Windows where multiprocessing can be problematic
    elif system in ['Darwin', 'Windows'] and device.type == 'cpu':
        # Use 0 workers for CPU on macOS/Windows to avoid pickling issues
        num_workers = 0
        logger.warning(f"Running on {system} with CPU - setting num_workers=0 to avoid multiprocessing issues")
    else:
        num_workers = default_num_workers

    # Determine pin_memory setting
    # If data is already on GPU (classic mode with cuda), don't use pin_memory
    # Pin memory is only for CPU tensors being transferred to GPU
    if data_loading_mode == 'classic' and config['data']['classic_mode_device'] == 'cuda':
        use_pin_memory = False  # Data already on GPU
    else:
        use_pin_memory = config['data']['pin_memory'] if device.type == 'cuda' else False
    
    logger.info("Creating DataLoaders with HDF5 backend...")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Num workers: {num_workers}")
    logger.info(f"  Pin memory: {use_pin_memory}")

    # Create DataLoader arguments
    loader_args = {
        'batch_size': config['training']['batch_size'],
        'num_workers': num_workers,
        'pin_memory': use_pin_memory,
    }

    # Only add prefetch_factor and persistent_workers if using multiple workers
    if num_workers > 0:
        loader_args.update({
            'prefetch_factor': config['data']['prefetch_factor'],
            'persistent_workers': config['data']['persistent_workers']
        })

    train_loader = DataLoader(train_subset, shuffle=True, **loader_args)
    logger.info(f"Training DataLoader created with {len(train_loader)} batches")

    val_loader = DataLoader(val_subset, shuffle=False, **loader_args)
    logger.info(f"Validation DataLoader created with {len(val_loader)} batches")
    
    # Initialize model
    reg_model = REG_NN(
        model_name=config['model']['name'],
        model_version=config['model']['version'],
        model_type='NN',
        model_cont_vars=config['data']['input_cols'],
        model_target_vars=config['data']['target_cols'],
        random_seed=config['random_seed'],
        dropout_rate=config['model']['dropout_rate'],
        l2_reg=config['model']['l2_reg'],
        init_method=config['model']['init_method'],
        learning_rate=config['training']['learning_rate']
    )
    
    model = reg_model.estimator
    model.to(device)
    
    # Determine component counts for loss function
    target_cols = config['data']['target_cols']
    has_vf = 'VF' in target_cols
    y_vars = [v for v in target_cols if v.startswith('y')]
    x_vars = [v for v in target_cols if v.startswith('x')]
    n_y_components = len(y_vars) if y_vars else 0
    n_x_components = len(x_vars) if x_vars else None
    
    # Loss function (MoleFractionLoss for normalized mole fractions with log-space MSE)
    criterion = MoleFractionLoss(
        n_y_components=n_y_components,
        n_x_components=n_x_components,
        has_vf=has_vf,
        epsilon=1e-8,
        vf_weight=1.0,
        mf_weight=1.0
    )
    logger.info(f"Using MoleFractionLoss: has_vf={has_vf}, n_y={n_y_components}, n_x={n_x_components}")
    
    # Optimizer (using centralized function)
    optimizer = get_optimizer(
        optimizer_name=config['training']['optimizer'],
        model_parameters=model.parameters(),
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['model']['l2_reg'],
        momentum=config['training']['momentum']
    )
    
    # Learning rate scheduler (using centralized function)
    scheduler = get_scheduler(
        scheduler_name=config['training']['scheduler'],
        optimizer=optimizer,
        config=config['training']
    )
    
    # Early stopping
    checkpoint_path = os.path.join(model_save_path, 'checkpoint_best.pth')
    early_stopping = EarlyStopping(
        patience=int(config['training']['early_stopping_patience']),
        min_delta=float(config['training']['early_stopping_delta']),
        verbose=True
    )
    
    # Checkpoint settings for long-running jobs
    checkpoint_frequency = config['logging']['checkpoint_frequency']
    save_checkpoints = config['logging']['save_checkpoints']

    # Load checkpoint if exists
    start_epoch, best_val_r2, history = load_checkpoint(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=checkpoint_path,
        model_save_path=model_save_path,
        device=device,
        model_type='regression'
    )

    # Training loop
    logger.info("="*80)
    logger.info("Starting training...")

    # Redefine epochs as number of epochs to train (not final epoch number)
    epochs_to_train = config['training']['epochs']
    target_epoch = start_epoch + epochs_to_train

    if start_epoch > 0:
        logger.info(f"Resuming from epoch {start_epoch}")
    logger.info(f"Number of epochs to train: {epochs_to_train}")
    logger.info(f"Training from epoch {start_epoch + 1} to epoch {target_epoch}")
    logger.info(f"Checkpoint frequency: every {checkpoint_frequency} epochs")
    logger.info(f"Early stopping patience: {config['training']['early_stopping_patience']} epochs")
    logger.info("="*80)

    # Handle validation-only mode when epochs_to_train=0
    if epochs_to_train == 0:
        logger.info("Training epochs set to 0 - running validation only")

    for epoch in range(start_epoch, target_epoch):
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Epoch {epoch+1}/{target_epoch}")
        logger.info(f"{'='*60}")
        start_time = time.time()
        
        # Train with memory-efficient function
        logger.info("Starting training phase...")
        train_loss, train_r2, train_preds, train_targets = train_epoch_efficient(
            model, train_loader, criterion, optimizer, device, full_dataset,
            progress_log_interval=config['logging']['progress_log_interval']
        )
        
        # Validate with memory-efficient function
        val_loss, val_r2, val_mse, val_mae, val_preds, val_targets = validate_epoch_efficient(
            model, val_loader, criterion, device, full_dataset,
            validation_log_interval=config['logging']['validation_log_interval']
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_r2'].append(train_r2)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Update scheduler
        if scheduler:
            if config['training']['scheduler'].lower() == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Log progress with comprehensive metrics
        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch [{epoch+1}/{config['training']['epochs']}] "
            f"Train Loss: {train_loss:.4f}, Train R2: {train_r2:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}, "
            f"MSE: {val_mse:.6f}, MAE: {val_mae:.6f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )
        
        # Save best model
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            reg_model.save(model_save_path)
            
            # Save best model checkpoint with full training state
            best_checkpoint_metrics = {
                'train_loss': train_loss,
                'train_r2': train_r2,
                'val_loss': val_loss,
                'val_r2': val_r2,
                'best_val_r2': best_val_r2
            }
            save_checkpoint(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=best_checkpoint_metrics,
                history=history,
                checkpoint_path=checkpoint_path,  # This is checkpoint_best.pth
                model_type='regression'
            )
            
            logger.info(f"New best model saved with validation R2: {best_val_r2:.6f}")
        
        # Periodic checkpointing for long-running jobs
        if save_checkpoints and (epoch + 1) % checkpoint_frequency == 0:
            checkpoint_name = f'checkpoint_epoch_{epoch+1}.pth'
            checkpoint_full_path = os.path.join(model_save_path, checkpoint_name)
            metrics = {
                'train_loss': train_loss,
                'train_r2': train_r2,
                'val_loss': val_loss,
                'val_r2': val_r2,
                'best_val_r2': best_val_r2
            }
            save_checkpoint(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
                history=history,
                checkpoint_path=checkpoint_full_path,
                model_type='regression'
            )

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

        # Periodic detailed evaluation
        if (epoch + 1) % config['training']['eval_frequency'] == 0:
            logger.info(f"\n--- Detailed Evaluation (Epoch {epoch+1}) ---")
            logger.info(f"Validation Metrics:")
            logger.info(f"  R2 Score: {val_r2:.6f}")
            logger.info(f"  MSE: {val_mse:.6f}")
            logger.info(f"  MAE: {val_mae:.6f}")
            logger.info(f"  Loss: {val_loss:.6f}")
            logger.info(f"Training Progress: {((epoch+1)/config['training']['epochs'])*100:.1f}% complete")
            logger.info(f"{'='*50}")

    
    # Reload best model before final evaluation and plotting
    best_model_path = os.path.join(model_save_path, 'model_complete.pth')
    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        logger.info(f"Reloaded best model from {best_model_path} for final evaluation")
    else:
        logger.warning("No model_complete.pth found - using last epoch model for final evaluation")

    # Final evaluation
    logger.info("Running final evaluation...")
    val_loss, val_r2, val_mse, val_mae, val_preds, val_targets = validate_epoch_efficient(
        model, val_loader, criterion, device, full_dataset
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL TRAINING RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Final Validation Metrics:")
    logger.info(f"  Loss: {val_loss:.6f}")
    logger.info(f"  R2 Score: {val_r2:.6f}")
    logger.info(f"  MSE: {val_mse:.6f}")
    logger.info(f"  MAE: {val_mae:.6f}")
    logger.info(f"  Best R2: {best_val_r2:.6f}")
    
    # Per-component metrics (only if we have validation predictions)
    target_names = config['data']['target_cols']
    if len(val_preds) > 0 and len(val_targets) > 0:
        logger.info(f"\nPer-Component Performance:")
        for i, target_name in enumerate(target_names):
            component_r2 = r2_score(val_targets[:, i], val_preds[:, i])
            component_mae = mean_absolute_error(val_targets[:, i], val_preds[:, i])
            logger.info(f"  {target_name}: R2={component_r2:.6f}, MAE={component_mae:.6f}")
    
    # Save final metrics
    metrics = {
        'final_val_loss': float(val_loss),
        'final_val_r2': float(val_r2),
        'final_val_mse': float(val_mse),
        'final_val_mae': float(val_mae),
        'best_val_r2': float(best_val_r2),
        'training_history': history,
        'config': config
    }
    
    with open(os.path.join(model_save_path, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate plots
    logger.info("Generating training plots...")
    
    # Training history plot
    plot_loss_history(
        history,
        model_name=config['model']['name'],
        save_path=os.path.join(model_save_path, 'training_history.png')
    )
    
    # Calibration plot (only if we have predictions)
    if len(val_preds) > 0 and len(val_targets) > 0:
        plot_calibration(
            real_np=val_targets,
            predicted_np=val_preds,
            list_var=target_names
        )
        plt.savefig(os.path.join(model_save_path, 'calibration_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Gradient flow diagnostics
    grad_summary = reg_model.get_gradient_info()
    if grad_summary:
        # Save gradient statistics
        with open(os.path.join(model_save_path, 'gradient_stats.json'), 'w') as f:
            json.dump(grad_summary, f, indent=4, default=str)
        
        # Generate gradient flow plot
        plot_individual_gradient_flow(
            model_name=f"{config['model']['name']}_reg_x",
            gradient_stats=grad_summary,
            save_path=os.path.join(model_save_path, 'gradient_flow.png')
        )
        logger.info("Generated gradient flow diagnostics")
    
    # Cleanup
    if hasattr(full_dataset, 'close'):
        full_dataset.close()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETED SUCCESSFULLY")
    logger.info(f"Model saved to: {model_save_path}")
    logger.info(f"Final validation R2: {val_r2:.6f}")
    logger.info(f"{'='*60}")
    
    return reg_model, history


def main():
    parser = argparse.ArgumentParser(description='Train Regression Neural Network (X - Liquid Phase)')
    parser.add_argument('--config', type=str, default='phase/config/training_reg_x_mn.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='phase/model',
                        help='Directory to save trained model')
    parser.add_argument('--system', type=str, required=True,
                        help='System name for process title (e.g., mn)')
    args = parser.parse_args()

    # Set process title for easy identification
    setproctitle.setproctitle(f"phase_{args.system}_train_reg_x")

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_version = config['model']['version']
    model_save_path = os.path.join(
        args.output_dir,
        f"{config['model']['name']}_{model_version}_reg_x"
    )
    os.makedirs(model_save_path, exist_ok=True)
    
    # Setup logging
    logger.add(
        os.path.join(model_save_path, 'training.log'),
        rotation="10 MB",
        retention="10 days",
        level="INFO"
    )
    
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Model will be saved to {model_save_path}")
    logger.info(f"Configuration:\n{yaml.dump(config, default_flow_style=False)}")
    
    # Train model
    try:
        model, history = train_model(config, model_save_path)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()