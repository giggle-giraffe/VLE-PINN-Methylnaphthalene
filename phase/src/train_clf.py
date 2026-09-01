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
import joblib
import numpy as np
import torch
# Set matplotlib backend for headless servers (must be before pyplot import)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster/server environments
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report, precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
from loguru import logger
import platform

from clf_nn import CLF_NN
from plot_util import (plot_loss_history, plot_classification_metrics,
                       plot_individual_gradient_flow, plot_confusion_matrix)
from data_util_hdf5 import (HDF5ThermoDataset, InMemoryThermoDataset, 
                             load_pretrained_scalers)
from util import (EarlyStopping, get_optimizer, get_scheduler, set_random_seeds,
                  load_checkpoint, save_checkpoint)

torch.set_default_dtype(torch.float32)


def train_epoch_efficient(model, dataloader, criterion, optimizer, device, label_smoothing=0.0, progress_log_interval=10):
    """
    Memory-efficient training for one epoch.
    Computes metrics incrementally without storing all predictions.
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    # For detailed metrics (collect on GPU first, transfer once at end)
    sample_preds_gpu = []
    sample_labels_gpu = []
    max_samples_for_metrics = 10000  # Only store limited samples for detailed metrics
    samples_collected = 0
    
    # Configurable progress tracking
    total_batches = len(dataloader)
    log_interval = max(1, total_batches * progress_log_interval // 100)
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):

        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Apply label smoothing if specified
        if label_smoothing > 0:
            num_classes = outputs.size(1)
            smooth_labels = torch.zeros_like(outputs)
            smooth_labels.fill_(label_smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - label_smoothing)
            loss = -(smooth_labels * torch.log_softmax(outputs, dim=1)).sum(dim=1).mean()
        else:
            loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Compute running statistics (memory efficient)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size  # Weight by batch size for exact average
        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == labels).sum().item()
        running_total += batch_size
        
        # Store limited samples on GPU (no transfer until end)
        if samples_collected < max_samples_for_metrics:
            batch_size = min(labels.size(0), max_samples_for_metrics - samples_collected)
            sample_preds_gpu.append(predicted[:batch_size])
            sample_labels_gpu.append(labels[:batch_size])
            samples_collected += batch_size
        
        # Progress logging
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            progress_pct = (batch_idx + 1) / total_batches * 100
            current_acc = running_correct / running_total if running_total > 0 else 0
            logger.info(f"Training: {progress_pct:.1f}% ({batch_idx+1}/{total_batches}) - "
                       f"Loss: {loss.item():.6f}, Acc: {current_acc:.4f}")
    
    # Transfer samples to CPU only once at the end
    if len(sample_preds_gpu) > 0:
        sample_preds = torch.cat(sample_preds_gpu).cpu().numpy().tolist()
        sample_labels = torch.cat(sample_labels_gpu).cpu().numpy().tolist()
    else:
        sample_preds = []
        sample_labels = []
    
    epoch_loss = running_loss / running_total if running_total > 0 else 0
    epoch_acc = running_correct / running_total if running_total > 0 else 0
    
    return epoch_loss, epoch_acc, sample_preds, sample_labels


def validate_epoch_efficient(model, dataloader, criterion, device, validation_log_interval=20):
    """
    Memory-efficient validation for one epoch.
    Computes metrics incrementally without storing all predictions.
    """
    model.eval()
    # Use GPU tensors for accumulators to avoid sync every batch!
    running_loss = torch.tensor(0.0, device=device)
    running_correct = torch.tensor(0, device=device, dtype=torch.int64)
    running_total = 0
    
    # Sample for detailed metrics (collect on GPU first)
    sample_preds_gpu = []
    sample_labels_gpu = []
    sample_probs_gpu = []
    max_samples = 10000
    samples_collected = 0
    
    with torch.no_grad():
        total_batches = len(dataloader)
        log_interval = max(1, total_batches * validation_log_interval // 100)
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Skip .to(device) if data is already on correct device
            if inputs.device != device:
                inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Running statistics (minimal operations only!)
            # Keep loss on GPU, accumulate there to avoid sync
            batch_size = labels.size(0)
            running_loss += loss.detach() * batch_size  # Weight by batch size for exact average
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum()
            running_total += batch_size
            
            # Store limited samples (keep on GPU until the end)
            if samples_collected < max_samples:
                batch_size = min(labels.size(0), max_samples - samples_collected)
                sample_preds_gpu.append(predicted[:batch_size])
                sample_labels_gpu.append(labels[:batch_size])
                # Only compute probs when needed!
                probs = torch.softmax(outputs[:batch_size], dim=1)
                sample_probs_gpu.append(probs)
                samples_collected += batch_size
            
            # Progress logging
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                progress_pct = (batch_idx + 1) / total_batches * 100
                current_acc = (running_correct.item() / running_total) if running_total > 0 else 0
                logger.info(f"Validation: {progress_pct:.1f}% ({batch_idx+1}/{total_batches}) - "
                          f"Loss: {loss.item():.6f}, Acc: {current_acc:.4f}")
    
    # Convert samples to numpy (single transfer at end)
    if len(sample_preds_gpu) > 0:
        sample_preds = torch.cat(sample_preds_gpu).cpu().numpy().tolist()
        sample_labels = torch.cat(sample_labels_gpu).cpu().numpy().tolist()
        sample_probs = torch.cat(sample_probs_gpu).cpu().numpy().tolist()
        
        # Compute confusion matrix from samples only (much faster!)
        conf_matrix = confusion_matrix(sample_labels, sample_preds)
    else:
        sample_preds = []
        sample_labels = []
        sample_probs = []
        conf_matrix = np.zeros((2, 2), dtype=np.int64)
    
    # Convert accumulated GPU tensors to Python floats (single sync at end!)
    epoch_loss = (running_loss / running_total).item() if running_total > 0 else 0.0
    epoch_acc = (running_correct / running_total).item() if running_total > 0 else 0.0
    
    return epoch_loss, epoch_acc, sample_preds, sample_labels, sample_probs, conf_matrix


def train_model(config, model_save_path):
    """Main training function with memory-efficient implementation"""

    # Set random seeds for reproducibility
    set_random_seeds(config['random_seed'])
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Memory optimization settings
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        # Enable memory efficient attention if available
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Construct HDF5 data file path
    data_file = config['data']['train_file']
    
    # Ensure it's an HDF5 file
    if not data_file.endswith('.h5'):
        logger.error(f"Invalid data file format: {data_file}")
        logger.error("Only HDF5 (.h5) files are supported for training")
        logger.error("To convert CSV to HDF5, run:")
        logger.error(f"  python phase/src/convert_csv_to_hdf5.py --input your_file.csv")
        raise ValueError("HDF5 file required. Only .h5 format is supported.")
    
    # Find the HDF5 file
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
            logger.error(f"HDF5 file not found: {data_file}")
            logger.error("Searched in:")
            for path in possible_paths:
                logger.error(f"  - {os.path.abspath(path)}")
            raise FileNotFoundError(f"HDF5 data file not found: {data_file}")
    else:
        data_file_path = data_file
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"HDF5 data file not found: {data_file_path}")
    
    logger.info(f"Using HDF5 data file: {os.path.abspath(data_file_path)}")
    
    # Get data loading mode
    data_loading_mode = config['data']['data_loading_mode']
    logger.info(f"Data loading mode: {data_loading_mode}")
    
    # Check for pre-fitted scalers
    # Scaler path structure: phase/scalers/{system}/{version}/
    system = config['data']['system']
    model_version = config['model']['version']
    scaler_dir = os.path.join('phase/scalers', system, model_version)
    use_pretrained_scalers = os.path.exists(os.path.join(scaler_dir, 'clf_feature_scaler.pkl'))
    
    if data_loading_mode == 'classic':
        # Classic mode: Load entire dataset into memory/GPU
        classic_device = config['data']['classic_mode_device']
        logger.info(f"Classic mode: Loading entire dataset into {classic_device}")
        
        if use_pretrained_scalers:
            logger.info("Loading pre-fitted scalers...")
            feature_scaler, class_weights = load_pretrained_scalers(scaler_dir, 'clf')
            logger.info("Pre-fitted scalers loaded successfully")
            
            full_dataset = InMemoryThermoDataset(
                hdf5_file=data_file_path,
                scaler=feature_scaler,
                fit_scaler=False,
                device=classic_device,
                pin_memory=True
            )
            
            if class_weights is not None:
                full_dataset.class_weights = class_weights
                logger.info(f"Using pre-fitted class weights: {class_weights.numpy()}")
        else:
            logger.info("Pre-fitted scalers not found. Fitting new scalers...")
            logger.info("TIP: Run 'python phase/src/fit_scalers.py' first to pre-fit scalers")
            
            full_dataset = InMemoryThermoDataset(
                hdf5_file=data_file_path,
                scaler=None,
                fit_scaler=True,
                device=classic_device,
                pin_memory=True
            )
    else:
        # HDF5 mode: Lazy loading (default)
        if use_pretrained_scalers:
            logger.info("Loading pre-fitted scalers...")
            feature_scaler, class_weights = load_pretrained_scalers(scaler_dir, 'clf')
            logger.info("Pre-fitted scalers loaded successfully")
            
            # Create HDF5 dataset with pre-fitted scaler
            logger.info("Creating HDF5 dataset with pre-fitted scaler...")
            logger.info(f"HDF5 file: {data_file_path}")
            
            full_dataset = HDF5ThermoDataset(
                hdf5_file=data_file_path,
                scaler=feature_scaler,
                fit_scaler=False
            )
            logger.info("HDF5 dataset created successfully")
            
            if class_weights is not None:
                full_dataset.class_weights = class_weights
                logger.info(f"Using pre-fitted class weights: {class_weights.numpy()}")
        else:
            logger.info("Pre-fitted scalers not found. Fitting new scalers...")
            logger.info("TIP: Run 'python phase/src/fit_scalers.py' first to pre-fit scalers")
            logger.info(f"HDF5 file: {data_file_path}")
            logger.info("Creating HDF5 dataset and fitting scaler...")
            
            # Create HDF5 dataset and fit scaler
            full_dataset = HDF5ThermoDataset(
                hdf5_file=data_file_path,
                scaler=None,
                fit_scaler=True
            )
            logger.info("HDF5 dataset created and scaler fitted successfully")
    
    # Split indices for train/val
    logger.info("Preparing train/validation split...")
    dataset_size = len(full_dataset)
    logger.info(f"Total dataset size: {dataset_size:,} samples")
    
    train_size = int(config['data']['train_val_split'] * dataset_size)
    logger.info(f"Train size: {train_size:,}, Val size: {dataset_size - train_size:,}")
    
    # Create indices and shuffle
    logger.info("Creating and shuffling indices...")
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    logger.info("Creating dataset subsets...")
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    logger.info(f"Dataset split complete - Training: {len(train_subset):,}, Validation: {len(val_subset):,}")
    
    # Create data loaders - HDF5 supports efficient parallel access
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
    logger.info(f"  Prefetch factor: {config['data']['prefetch_factor'] if num_workers > 0 else None}")
    logger.info(f"  Pin memory: {use_pin_memory}")

    # Create DataLoader arguments
    train_loader_args = {
        'batch_size': config['training']['batch_size'],
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': use_pin_memory,
    }

    val_loader_args = {
        'batch_size': config['training']['batch_size'],
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': use_pin_memory,
    }

    # Only add prefetch_factor and persistent_workers if using multiple workers
    if num_workers > 0:
        train_loader_args.update({
            'prefetch_factor': config['data']['prefetch_factor'],
            'persistent_workers': config['data']['persistent_workers']
        })
        val_loader_args.update({
            'prefetch_factor': config['data']['prefetch_factor'],
            'persistent_workers': config['data']['persistent_workers']
        })

    train_loader = DataLoader(train_subset, **train_loader_args)
    logger.info(f"Training DataLoader created with {len(train_loader)} batches")

    val_loader = DataLoader(val_subset, **val_loader_args)
    logger.info(f"Validation DataLoader created with {len(val_loader)} batches")
    
    # Initialize model
    clf_model = CLF_NN(
        model_name=config['model']['name'],
        model_version=config['model']['version'],
        model_type='NN',
        model_cont_vars=config['data']['input_cols'],
        model_target_vars=[config['data']['target_col']],
        random_seed=config['random_seed'],
        dropout_rate=config['model']['dropout_rate'],
        l2_reg=config['model']['l2_reg'],
        learning_rate=config['training']['learning_rate']
    )
    
    model = clf_model.estimator
    model.to(device)
    
    # Loss function with class weights
    use_class_weights = config['training']['use_class_weights']
    if use_class_weights:
        class_weights = full_dataset.get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Using class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = get_optimizer(
        optimizer_name=config['training']['optimizer'],
        model_parameters=model.parameters(),
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['model']['l2_reg'],
        momentum=config['training']['momentum']
    )
    
    # Learning rate scheduler
    scheduler_type = config['training']['scheduler']
    scheduler = get_scheduler(scheduler_type, optimizer, config['training'])
    
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
    start_epoch, best_val_acc, history = load_checkpoint(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=checkpoint_path,
        model_save_path=model_save_path,
        device=device,
        model_type='classification'
    )
    
    # Test dataset access before training
    logger.info("Testing dataset access...")
    try:
        logger.info("Accessing first sample from dataset...")
        test_sample = full_dataset[0]
        logger.info(f"✓ First sample loaded: features shape {test_sample[0].shape}, label {test_sample[1]}")
        
        logger.info("Testing DataLoader with one batch...")
        test_batch = next(iter(train_loader))
        logger.info(f"✓ First batch loaded: batch size {test_batch[0].shape[0]}")
        del test_batch  # Free memory
        
    except Exception as e:
        logger.error(f"✗ Dataset access failed: {e}")
        raise

    # Training loop
    logger.info("="*80)
    logger.info("Starting memory-efficient training loop...")

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

    label_smoothing = float(config['training']['label_smoothing'])

    # Handle the case where epochs=0 (validation-only mode)
    if epochs_to_train == 0:
        logger.info("Training epochs set to 0 - running validation only")
        val_preds = []
        val_labels = []
        val_probs = []
        val_acc = 0.0
        final_conf_matrix = np.zeros((2, 2), dtype=np.int64)

    for epoch in range(start_epoch, target_epoch):
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Epoch {epoch+1}/{target_epoch}")
        logger.info(f"{'='*60}")
        start_time = time.time()
        
        # Train with memory-efficient function
        logger.info("Starting training phase...")
        train_loss, train_acc, train_sample_preds, train_sample_labels = train_epoch_efficient(
            model, train_loader, criterion, optimizer, device, label_smoothing,
            progress_log_interval=config['logging']['progress_log_interval']
        )
        
        # Validate with memory-efficient function
        val_loss, val_acc, val_sample_preds, val_sample_labels, val_sample_probs, val_conf_matrix = validate_epoch_efficient(
            model, val_loader, criterion, device,
            validation_log_interval=config['logging']['validation_log_interval']
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Update scheduler
        if scheduler:
            if scheduler_type == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Log progress
        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch [{epoch+1}/{config['training']['epochs']}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            clf_model.save(model_save_path)
            
            # Save scaler
            scaler_path = os.path.join(model_save_path, 'feature_scaler.pkl')
            joblib.dump(full_dataset.get_scaler(), scaler_path)
            
            # Save best model checkpoint with full training state
            best_checkpoint_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            }
            save_checkpoint(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=best_checkpoint_metrics,
                history=history,
                checkpoint_path=checkpoint_path,  # This is checkpoint_best.pth
                model_type='classification'
            )
            
            logger.info(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
        
        # Periodic checkpointing for long-running jobs
        if save_checkpoints and (epoch + 1) % checkpoint_frequency == 0:
            checkpoint_name = f'checkpoint_epoch_{epoch+1}.pth'
            checkpoint_full_path = os.path.join(model_save_path, checkpoint_name)
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            }
            save_checkpoint(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
                history=history,
                checkpoint_path=checkpoint_full_path,
                model_type='classification'
            )
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
        
        # Periodic detailed evaluation using samples
        if (epoch + 1) % config['training']['eval_frequency'] == 0:
            if len(val_sample_preds) > 0 and len(val_sample_labels) > 0:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    val_sample_labels, val_sample_preds, average='weighted'
                )
                logger.info(f"Sample metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Log confusion matrix
            logger.info(f"Confusion Matrix:\n{val_conf_matrix}")
        
        # Clear GPU cache periodically
        if device.type == 'cuda' and (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # Reload best model before final evaluation and plotting
    best_model_path = os.path.join(model_save_path, 'model_complete.pth')
    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        logger.info(f"Reloaded best model from {best_model_path} for final evaluation")
    else:
        logger.warning("No model_complete.pth found - using last epoch model for final evaluation")

    # Final evaluation - run even if epochs_to_train=0 (validation-only mode)
    logger.info("Running final evaluation...")
    if epochs_to_train == 0:
        logger.info("(Validation-only mode - no training was performed)")
    val_loss, val_acc, val_preds, val_labels, val_probs, final_conf_matrix = validate_epoch_efficient(
        model, val_loader, criterion, device
    )
    
    # Compute detailed final metrics
    if len(val_preds) > 0 and len(val_labels) > 0:
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='weighted'
        )
        
        # Classification report
        clf_report = classification_report(val_labels, val_preds, digits=4)
        logger.info(f"Final Classification Report:\n{clf_report}")
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            val_labels, val_preds, average=None
        )
        
        logger.info("Per-class metrics:")
        for i, (p, r, f, s) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class, support_per_class)):
            logger.info(f"  Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}, Support={s}")
    else:
        precision = recall = f1 = 0.0
        clf_report = "No predictions available for detailed metrics"
        precision_per_class = recall_per_class = f1_per_class = support_per_class = []
    
    # Save comprehensive final metrics
    metrics = {
        'final_val_loss': float(val_loss),
        'final_val_acc': float(val_acc),
        'best_val_acc': float(best_val_acc),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'precision_per_class': precision_per_class.tolist() if len(precision_per_class) > 0 else [],
        'recall_per_class': recall_per_class.tolist() if len(recall_per_class) > 0 else [],
        'f1_per_class': f1_per_class.tolist() if len(f1_per_class) > 0 else [],
        'support_per_class': support_per_class.tolist() if len(support_per_class) > 0 else [],
        'confusion_matrix': final_conf_matrix.tolist(),
        'classification_report': clf_report,
        'training_history': history,
        'config': config
    }
    
    with open(os.path.join(model_save_path, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate comprehensive diagnostic plots
    logger.info("Generating comprehensive training diagnostics...")
    
    # 1. Training history plot (loss curves)
    logger.info("Generating training history plot...")
    plot_loss_history(
        history,
        model_name=config['model']['name'],
        save_path=os.path.join(model_save_path, 'training_history.png')
    )
    
    # 2. Classification metrics plot (accuracy, precision, recall over epochs)
    logger.info("Generating classification metrics plot...")
    # Convert history format for compatibility with plot_classification_metrics
    plot_history = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'accuracy': history['val_acc']  # Map val_acc to accuracy for plot function
    }
    
    plot_classification_metrics(
        plot_history,
        model_name=config['model']['name'],
        save_path=os.path.join(model_save_path, 'classification_metrics.png')
    )
    
    # 3. Confusion matrix plot
    logger.info("Generating confusion matrix plot...")
    if len(val_preds) > 0:
        fig_cm = plot_confusion_matrix(
            real_np=val_labels,
            predicted_np=val_preds,
            list_var=None
        )
        fig_cm.savefig(os.path.join(model_save_path, 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
        plt.close(fig_cm)
        logger.info(f"Confusion matrix saved to {os.path.join(model_save_path, 'confusion_matrix.png')}")
    
    # 4. Gradient flow diagnostics (if available)
    logger.info("Generating gradient flow diagnostics...")
    grad_summary = clf_model.get_gradient_info()
    if grad_summary:
        logger.info("Gradient information available, creating gradient flow plot...")
        
        # Save gradient statistics
        with open(os.path.join(model_save_path, 'gradient_stats.json'), 'w') as f:
            json.dump(grad_summary, f, indent=4, default=str)
        
        # Plot individual gradient flow
        try:
            plot_individual_gradient_flow(
                model_name=config['model']['name'],
                gradient_stats=grad_summary,
                save_path=os.path.join(model_save_path, 'gradient_flow.png')
            )
            logger.info(f"Gradient flow plot saved to {os.path.join(model_save_path, 'gradient_flow.png')}")
        except Exception as e:
            logger.warning(f"Could not generate gradient flow plot: {e}")
        
        logger.info(f"Gradient diagnostics saved to {model_save_path}")
    else:
        logger.warning("No gradient information available for plotting")
    
    # 5. Precision-Recall and ROC curves (if we have probability predictions)
    if len(val_probs) > 0 and len(val_labels) > 0:
        logger.info("Generating precision-recall and ROC curves...")
        try:
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(val_labels, [p[1] for p in val_probs])
            pr_auc = auc(recall, precision)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(val_labels, [p[1] for p in val_probs])
            roc_auc = auc(fpr, tpr)
            
            # Create subplot figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot Precision-Recall curve
            ax1.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
            ax1.set_xlabel('Recall')
            ax1.set_ylabel('Precision')
            ax1.set_title('Precision-Recall Curve')
            ax1.legend()
            ax1.grid(True)
            
            # Plot ROC curve
            ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            ax2.plot([0, 1], [0, 1], 'k--', label='Random')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curve')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(model_save_path, 'pr_roc_curves.png'), dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Precision-Recall and ROC curves saved (PR AUC: {pr_auc:.3f}, ROC AUC: {roc_auc:.3f})")
            
        except Exception as e:
            logger.warning(f"Could not generate PR/ROC curves: {e}")
    
    # 6. Model calibration plot
    if len(val_probs) > 0 and len(val_labels) > 0:
        logger.info("Generating calibration plot...")
        try:
            # Calculate calibration
            prob_true, prob_pred = calibration_curve(val_labels, [p[1] for p in val_probs], n_bins=10)
            
            # Create calibration plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(prob_pred, prob_true, 's-', label='Model')
            ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Calibration Plot')
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(model_save_path, 'calibration_plot.png'), dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            logger.info("Calibration plot saved")
            
        except Exception as e:
            logger.warning(f"Could not generate calibration plot: {e}")
    
    logger.info("All diagnostic plots generated successfully!")
    
    logger.info(f"Training completed. Model saved to {model_save_path}")
    logger.info(f"Final validation accuracy: {val_acc:.4f}")
    
    return clf_model, history


def main():
    parser = argparse.ArgumentParser(description='Classification Neural Network Training (HDF5-based)')
    parser.add_argument('--config', type=str, default='phase/config/training_clf_mn.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='phase/model',
                        help='Directory to save trained model')
    parser.add_argument('--system', type=str, required=True,
                        help='System name for process title (e.g., mn)')
    args = parser.parse_args()

    # Set process title for easy identification
    setproctitle.setproctitle(f"phase_{args.system}_train_clf")

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_version = config['model']['version']
    model_save_path = os.path.join(
        args.output_dir,
        f"{config['model']['name']}_{model_version}_clf"
    )
    os.makedirs(model_save_path, exist_ok=True)
    
    # Setup logging
    logger.add(
        os.path.join(model_save_path, 'training.log'),
        rotation="10 MB",
        retention="10 days",
        level="INFO"
    )
    
    logger.info(f"HDF5-based training mode")
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