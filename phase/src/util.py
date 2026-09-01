# ------------------------------------------------------------------------------------------
# Copyright (c) 2026 Chen Zhang, Tao Li
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

import numpy as np
import torch
import polars as pl
from loguru import logger

def df_to_tensor(df, dtype=torch.float64):
    """
    Convert a Polars DataFrame to a PyTorch tensor.

    Args:
        df (pl.DataFrame): The input Polars DataFrame.
        dtype (torch.dtype): The desired data type of the tensor.
    """
    return torch.tensor(df.to_numpy(), dtype=dtype)


def stratified_split(df, class_column, train_ratio=0.9, seed=42):
    """
    Perform a stratified train-validation split using Polars
    
    Args:
        df: Polars DataFrame
        class_column: Name of the column containing class labels
        train_ratio: Proportion of data to use for training
        seed: Random seed for reproducibility
        
    Returns:
        train_df, val_df: Stratified train and validation DataFrames
    """
    # Get unique class labels
    unique_classes = df.select(pl.col(class_column).unique()).to_series().to_list()
    
    # Initialize empty DataFrames for train and validation
    train_dfs = []
    val_dfs = []
    
    # For each class, sample proportionally
    for class_label in unique_classes:
        # Filter data for this class
        class_df = df.filter(pl.col(class_column) == class_label)
        
        # Calculate split index for this class
        class_split_index = int(len(class_df) * train_ratio)
        
        # Shuffle this class data
        class_df_shuffled = class_df.sample(fraction=1.0, seed=seed)
        
        # Split into train and validation
        class_train = class_df_shuffled.slice(0, class_split_index)
        class_val = class_df_shuffled.slice(class_split_index, len(class_df_shuffled) - class_split_index)
        
        # Add to lists
        train_dfs.append(class_train)
        val_dfs.append(class_val)
    
    # Combine all class-specific splits
    train_df = pl.concat(train_dfs)
    val_df = pl.concat(val_dfs)
    
    # Shuffle the final datasets
    train_df = train_df.sample(fraction=1.0, seed=seed)
    val_df = val_df.sample(fraction=1.0, seed=seed)
    
    return train_df, val_df


class EarlyStopping:
    """Early stopping to prevent overfitting during training"""
    
    def __init__(self, patience=10, min_delta=0.0001, mode='min', verbose=True):
        """
        Args:
            patience: Number of epochs with no improvement to wait before stopping
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            mode: One of 'min' or 'max'. In 'min' mode, training stops when monitored quantity stops decreasing
            verbose: If True, prints messages about early stopping progress
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        
    def __call__(self, val_loss):
        """
        Call the early stopping monitor
        
        Args:
            val_loss: Current validation loss
        """
        if self.mode == 'min':
            score = -val_loss
        else:
            score = val_loss
            
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0


def set_random_seeds(seed):
    """Set random seeds for reproducibility across different libraries"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # For reproducible results on CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seeds to {seed} for reproducibility")


def get_optimizer(optimizer_name, model_parameters, learning_rate, weight_decay=0.0, momentum=0.9):
    """
    Create optimizer based on configuration
    
    Args:
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
        model_parameters: Model parameters to optimize
        learning_rate: Initial learning rate
        weight_decay: Weight decay (L2 regularization)
        momentum: Momentum factor (only for SGD)
        
    Returns:
        Configured optimizer
    """
    import torch.optim as optim
    
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model_parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    logger.info(f"Created {optimizer_name} optimizer with lr={learning_rate}, weight_decay={weight_decay}")
    return optimizer


def get_scheduler(scheduler_name, optimizer, config):
    """
    Create learning rate scheduler based on configuration

    Args:
        scheduler_name: Name of scheduler ('reduce_on_plateau', 'cosine_annealing', or None)
        optimizer: Optimizer to schedule
        config: Training configuration dictionary

    Returns:
        Configured scheduler or None
    """
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

    if not scheduler_name or scheduler_name.lower() == 'none':
        return None

    scheduler_name = scheduler_name.lower()

    if scheduler_name == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(config.get('scheduler_factor', 0.5)),
            patience=int(config.get('scheduler_patience', 5)),
            min_lr=float(config.get('min_lr', 1e-6))
        )
        logger.info(f"Created ReduceLROnPlateau scheduler")
    elif scheduler_name == 'cosine_annealing':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(config.get('T_0', 10)),
            T_mult=int(config.get('T_mult', 2)),
            eta_min=float(config.get('min_lr', 1e-6))
        )
        logger.info(f"Created CosineAnnealingWarmRestarts scheduler")
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return scheduler


def load_checkpoint(config, model, optimizer, scheduler, checkpoint_path, model_save_path, device, model_type='classification'):
    """
    Load checkpoint for resuming training

    Args:
        config: Training configuration dictionary
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler to load state into (can be None)
        checkpoint_path: Path to checkpoint_best.pth
        model_save_path: Directory containing model checkpoints
        device: Device to load tensors to
        model_type: Type of model ('classification' or 'regression')

    Returns:
        tuple: (start_epoch, best_metric, history)
            - start_epoch: Epoch to resume from
            - best_metric: Best validation metric achieved
            - history: Training history dictionary
    """
    import os

    # Initialize default values
    start_epoch = 0
    best_metric = 0.0 if model_type == 'classification' else -np.inf
    history = {
        'train_loss': [],
        'train_acc' if model_type == 'classification' else 'train_r2': [],
        'val_loss': [],
        'val_acc' if model_type == 'classification' else 'val_r2': [],
        'learning_rates': []
    }

    if model_type == 'regression':
        history['val_mse'] = []
        history['val_mae'] = []

    # Check for explicit resume checkpoint path
    resume_checkpoint = config['training'].get('resume_checkpoint', None)

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        logger.info(f"Loading checkpoint from: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_metric = checkpoint.get('best_val_acc' if model_type == 'classification' else 'best_val_r2', best_metric)
        history = checkpoint.get('history', history)
        metric_name = 'val_acc' if model_type == 'classification' else 'val_r2'
        logger.info(f"Resumed from epoch {start_epoch} with best {metric_name}: {best_metric:.6f}")

    # Check for best checkpoint in model directory
    elif os.path.exists(checkpoint_path):
        logger.info(f"Found best checkpoint, loading from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # If it's a full checkpoint with optimizer state
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                best_metric = checkpoint.get('best_val_acc' if model_type == 'classification' else 'best_val_r2', best_metric)
                history = checkpoint.get('history', history)
                logger.info(f"Resumed from best checkpoint at epoch {start_epoch}")
            else:
                # If it's just model weights
                model.load_state_dict(checkpoint)
                logger.info("Loaded model weights from best checkpoint")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting fresh training.")

    # Check for the latest epoch checkpoint
    elif os.path.exists(model_save_path):
        checkpoint_files = [f for f in os.listdir(model_save_path)
                          if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        if checkpoint_files:
            # Get the latest checkpoint by epoch number
            latest_checkpoint = max(checkpoint_files,
                                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_checkpoint_path = os.path.join(model_save_path, latest_checkpoint)
            logger.info(f"Found checkpoint, loading from: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_metric = checkpoint.get('best_val_acc' if model_type == 'classification' else 'best_val_r2', best_metric)
            history = checkpoint.get('history', history)
            metric_name = 'val_acc' if model_type == 'classification' else 'val_r2'
            logger.info(f"Resumed from epoch {start_epoch} with best {metric_name}: {best_metric:.6f}")
        else:
            logger.info("No checkpoint found, starting fresh training.")
    else:
        logger.info("Starting fresh training.")

    return start_epoch, best_metric, history


def save_checkpoint(epoch, model, optimizer, scheduler, metrics, history, checkpoint_path,
                   model_type='classification'):
    """
    Save training checkpoint

    Args:
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state to save (can be None)
        metrics: Dictionary containing current metrics (train_loss, val_loss, etc.)
        history: Training history dictionary
        checkpoint_path: Path to save checkpoint
        model_type: Type of model ('classification' or 'regression')
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'history': history
    }

    # Add metrics based on model type
    if model_type == 'classification':
        checkpoint.update({
            'train_loss': metrics.get('train_loss'),
            'val_loss': metrics.get('val_loss'),
            'val_acc': metrics.get('val_acc'),
            'best_val_acc': metrics.get('best_val_acc')
        })
    else:  # regression
        checkpoint.update({
            'train_loss': metrics.get('train_loss'),
            'train_r2': metrics.get('train_r2'),
            'val_loss': metrics.get('val_loss'),
            'val_r2': metrics.get('val_r2'),
            'best_val_r2': metrics.get('best_val_r2')
        })

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")