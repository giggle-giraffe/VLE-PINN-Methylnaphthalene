# ------------------------------------------------------------------------------------------
# Checkpoint utilities for PINN training
# Copyright (c) 2026 Chen Zhang, Tao Li, Xiaoqian Dang
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

import torch
import os
import time
from loguru import logger


def save_checkpoint(
    checkpoint_dir,
    epoch,
    model,
    optimizer,
    loss,
    best_loss,
    parameters_history,
    loss_history,
    loss_history_gradnorm,
    curr_weights=None,
    is_best=False,
    save_periodic=False,
    save_final=False,
    pde_normalization_factor=None,
    loss_calibration_factors=None,
    gradnorm_calibrated=False,
):
    """
    Save model checkpoint with training state.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        epoch: Current epoch number
        model: PINN model instance
        optimizer: Optimizer instance
        loss: Current loss value
        best_loss: Best loss value so far
        parameters_history: History of model parameters
        loss_history: History of loss values
        loss_history_gradnorm: GradNorm loss history
        curr_weights: Current adaptive weights dictionary
        is_best: Whether this is the best model so far
        save_periodic: Whether to save a periodic checkpoint
        save_final: Whether to save as final checkpoint
        pde_normalization_factor: PDE log-scale normalization factor (computed at epoch 0)
        loss_calibration_factors: GradNorm auto-calibration weights dict
        gradnorm_calibrated: Whether GradNorm auto-calibration has been performed
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_loss': best_loss,
        'parameters_history': parameters_history,
        'loss_history': loss_history,
        'loss_history_gradnorm': loss_history_gradnorm,
        'device': str(model.device),
        'adaptive_weights': curr_weights,
        'pde_normalization_factor': pde_normalization_factor,
        'loss_calibration_factors': loss_calibration_factors if loss_calibration_factors else {},
        'gradnorm_calibrated': gradnorm_calibrated,
        # Architectural scalars (delta_t, output_scale, bounds) for deterministic model
        # reconstruction in extend_finetune without depending on pinn_model.pt or the YAML.
        'model_config': getattr(model, 'config', None),
    }
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save periodic checkpoint
    if save_periodic:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pt')
        torch.save(checkpoint, checkpoint_path)
    
    # Save final checkpoint
    if save_final:
        final_path = os.path.join(checkpoint_dir, f'checkpoint_final_{epoch}_{timestamp}.pt')
        torch.save(checkpoint, final_path)
    
    # Always save as latest checkpoint (overwrite) for any save operation
    if save_periodic or save_final or is_best:
        latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)


def load_checkpoint(
    resume_from,
    model,
    optimizer,
    weights,
    loss_history_gradnorm,
    grad_history,
    weight_history,
    lr_schedule,
    task='train'
):
    """
    Load checkpoint and restore training state.
    
    Args:
        resume_from: Path to checkpoint file
        model: PINN model instance
        optimizer: Optimizer instance
        weights: Current weights dictionary (will be updated)
        loss_history_gradnorm: GradNorm loss history dict (will be updated)
        grad_history: Gradient history dict (will be updated)
        weight_history: Weight history dict (will be updated)
        lr_schedule: Learning rate schedule dictionary
        task: 'train' or 'predict'
    
    Returns:
        Dictionary with:
            - start_epoch: Epoch to resume from
            - best_loss: Best loss from checkpoint
            - parameters_history: Restored parameters history
            - loss_history: Restored loss history
            - weights: Updated weights
    """
    if not os.path.exists(resume_from):
        raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
    
    logger.info(f"Loading checkpoint from: {resume_from}")
    
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(resume_from, map_location=map_location, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
    # Load optimizer state for continuity (momentum, etc.)
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✅ Loaded optimizer state (momentum, learning rate history)")
        except Exception as e:
            logger.warning(f"⚠️ Could not load optimizer state: {e}")
            logger.warning("Optimizer state will be reset (momentum, etc.)")
    
    best_loss = checkpoint['best_loss']
    
    # Set learning rate based on mode
    if task == 'predict':
        # For predict mode, use a more aggressive learning rate for continuous training
        predict_lr = max(1e-5, min(5e-5, lr_schedule[max([e for e in lr_schedule.keys() if e <= start_epoch])]))
        logger.info(f"🔄 Predict mode: Using learning rate {predict_lr:.2e}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = predict_lr
    else:
        # For train mode, continue with original schedule
        current_lr = lr_schedule[max([e for e in lr_schedule.keys() if e <= start_epoch])]
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    
    # Load histories if available
    parameters_history = checkpoint.get('parameters_history', [])
    loss_history = checkpoint.get('loss_history', [])
    
    # Load GradNorm loss history if available
    if 'loss_history_gradnorm' in checkpoint:
        checkpoint_gradnorm_history = checkpoint['loss_history_gradnorm']
        for k in weights.keys():
            if k in checkpoint_gradnorm_history:
                loss_history_gradnorm[k] = checkpoint_gradnorm_history[k].copy()
                logger.info(f"Loaded GradNorm history for '{k}': {len(loss_history_gradnorm[k])} entries")
        logger.info("GradNorm loss history loaded for continuous adaptive weighting")
    else:
        logger.warning("No GradNorm loss history found in checkpoint - starting with empty history")
    
    # Load adaptive weights if available
    if 'adaptive_weights' in checkpoint:
        loaded_weights = checkpoint['adaptive_weights']
        weights.update(loaded_weights)
        logger.info(f"Loaded adaptive weights from checkpoint: {weights}")
        
        # Update history structures if weights changed
        for k in weights.keys():
            if k not in loss_history_gradnorm:
                loss_history_gradnorm[k] = []
                logger.info(f"Initialized empty GradNorm history for new weight key: {k}")
            if k not in grad_history:
                grad_history[k] = []
            if k not in weight_history:
                weight_history[k] = []
        
        # Remove entries for weight keys no longer present
        for history_dict, name in [(loss_history_gradnorm, "GradNorm"), 
                                    (grad_history, "gradient"), 
                                    (weight_history, "weight")]:
            keys_to_remove = [k for k in history_dict.keys() if k not in weights]
            for k in keys_to_remove:
                del history_dict[k]
                logger.info(f"Removed {name} history for unused weight key: {k}")
    
    # Load calibration state if available
    pde_normalization_factor = checkpoint.get('pde_normalization_factor', None)
    loss_calibration_factors = checkpoint.get('loss_calibration_factors', {})
    gradnorm_calibrated = checkpoint.get('gradnorm_calibrated', False)
    if pde_normalization_factor is not None:
        logger.info(f"Loaded PDE normalization factor: {pde_normalization_factor:.4e}")
    if gradnorm_calibrated:
        logger.info(f"GradNorm auto-calibration already performed: {loss_calibration_factors}")

    logger.info(f"Resuming training from epoch {start_epoch} with best loss: {best_loss:.4e}")

    return {
        'start_epoch': start_epoch,
        'best_loss': best_loss,
        'parameters_history': parameters_history,
        'loss_history': loss_history,
        'weights': weights,
        'pde_normalization_factor': pde_normalization_factor,
        'loss_calibration_factors': loss_calibration_factors,
        'gradnorm_calibrated': gradnorm_calibrated,
    }
