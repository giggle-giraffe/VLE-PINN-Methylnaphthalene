# ------------------------------------------------------------------------------------------
# Loss computation utilities for PINN training epochs
# Copyright (c) 2026 Chen Zhang, Tao Li, Xiaoqian Dang
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

import torch
from loguru import logger

from ..loss import (
    compute_initial_condition_loss,
    compute_data_loss_with_coverage,
    compute_phase_loss,
    compute_flowrate_conservation_loss,
    compute_monotonicity_loss_autograd,
)


def compute_all_losses(
    model,
    f_initial_epoch,
    f_final,
    compute_pde_loss_fn,
    mse_loss,
    device,
    task,
    run_mode,
    include_initial_loss,
    enable_data_shuffling,
    shuffle_indices,
    pde_collocation_points=None,
    compute_phase=True,
    compute_carbon_balance=True,
    min_rate_threshold=1e-5,
    extend_target_time=None,
):
    """
    Compute all loss components for a training epoch.

    This consolidates the loss computation logic that's used in multiple places
    (curriculum transition, GradNorm updates, regular training).

    Args:
        model: PINN model instance
        f_initial_epoch: Initial features for current epoch (possibly shuffled).
                         Column 0 holds the original experimental target time per sample.
        f_final: Final/target features
        compute_pde_loss_fn: PDE loss function
        mse_loss: MSE loss function instance
        device: Torch device
        task: 'train', 'predict', or 'extend_finetune'
        run_mode: 'pinn' or 'pinn+phase'
        include_initial_loss: Whether to compute initial condition loss
        enable_data_shuffling: Whether data shuffling is enabled
        shuffle_indices: Shuffled indices for data
        pde_collocation_points: Optional tensor of continuous time values for PDE evaluation.
                                If None, uses the forward pass time_points grid.
        compute_phase: Whether to compute phase loss
        extend_target_time: If set (extend_finetune mode), the forward pass extends the
                            time-axis iteration to this value (via model.forward's
                            max_target_time_override) while leaving f_initial[:, 0] at the
                            original per-sample t_x. Data loss / get_predictions still use
                            the original column 0, so data is compared at the experimental
                            t_x, not at the extended time; PDE / monotonicity / IC /
                            flowrate are evaluated over [0, extend_target_time].

    Returns:
        Dictionary containing:
            - f_pred: Forward predictions
            - time_points: Time points from forward pass
            - x_transformed: Phase-transformed concentrations
            - f_recalculated: Recalculated flow rates
            - f_pred_at_target: Predictions at target time
            - x_pred_at_target: X predictions at target time
            - loss_data: Data loss (normalized)
            - loss_data_raw: Raw data loss
            - loss_initial_condition: Initial condition loss (normalized)
            - loss_initial_condition_raw: Raw initial condition loss
            - loss_pde_col: PDE collocation loss
            - loss_phase_transformed: Phase loss (normalized)
            - loss_phase_raw: Raw phase loss
            - flowrate_conservation: Flowrate conservation loss (normalized)
            - flowrate_conservation_raw: Raw flowrate conservation loss
            - pde_residuals: Tuple of (r1, r2, r3, r4) PDE residuals
            - monotonicity_loss, rate_decrease_loss, min_dynamics_loss (+ *_raw):
              Physics constraint losses (normalized and raw) on the trajectory.
    """
    f_initial_dyn = f_initial_epoch
    forward_kwargs = {}
    if extend_target_time is not None:
        forward_kwargs['max_target_time_override'] = extend_target_time

    # Forward pass (always full trajectory)
    f_pred, time_points = model(f_initial=f_initial_dyn, **forward_kwargs)

    # Get phase transformed concentrations
    x_transformed, f_recalculated, _ = model.get_phase_transformed_x(
        f_pred=f_pred, time_points=time_points
    )

    # Get final predictions at the ORIGINAL per-sample target time
    f_pred_at_target, x_pred_at_target = model.get_predictions(
        f_pred=f_pred,
        x_transformed=x_transformed,
        f_initial=f_initial_epoch,
        time_points=time_points
    )

    # Data loss (always uses all samples - no coverage weighting)
    if task == 'predict':
        loss_data = torch.tensor(0.0, device=device, dtype=torch.float64)
        loss_data_raw = torch.tensor(0.0, device=device, dtype=torch.float64)
    elif run_mode == 'pinn+phase':
        target_data = f_final[shuffle_indices][:, 5:9] if enable_data_shuffling else f_final[:, 5:9]
        loss_data, loss_data_raw = compute_data_loss_with_coverage(
            f_pred_at_target[:, :4], target_data,
            output_scale=model.output_scale, coverage_weights=None
        )
    elif run_mode == 'pinn':
        target_data = f_final[shuffle_indices][:, 3:9] if enable_data_shuffling else f_final[:, 3:9]
        loss_data, loss_data_raw = compute_data_loss_with_coverage(
            x_pred_at_target[:, 2:8], target_data,
            output_scale=None, coverage_weights=None
        )
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}")

    # Initial condition loss
    if include_initial_loss:
        loss_initial_condition, loss_initial_condition_raw = compute_initial_condition_loss(
            f_pred, f_initial_epoch, model.output_scale, mse_loss
        )
    else:
        loss_initial_condition = torch.tensor(0.0, device=device, dtype=torch.float64)
        loss_initial_condition_raw = torch.tensor(0.0, device=device, dtype=torch.float64)

    pde_time_points = pde_collocation_points if pde_collocation_points is not None else time_points
    pde_returns = compute_pde_loss_fn(
        model, f_initial=f_initial_dyn, time_points=pde_time_points,
        return_carbon_balance=compute_carbon_balance,
    )
    if compute_carbon_balance:
        r1_col, r2_col, r3_col, r4_col, dFsum_dt_all = pde_returns
    else:
        r1_col, r2_col, r3_col, r4_col = pde_returns
        dFsum_dt_all = None
    loss_pde_col = sum(torch.mean(r ** 2) for r in [r1_col, r2_col, r3_col, r4_col])

    # Phase loss
    if compute_phase and run_mode == 'pinn+phase':
        loss_phase_transformed, loss_phase_raw = compute_phase_loss(
            f_pred, f_recalculated, model.output_scale, mse_loss
        )
    else:
        loss_phase_transformed = torch.tensor(0.0, device=device, dtype=torch.float64)
        loss_phase_raw = torch.tensor(0.0, device=device, dtype=torch.float64)

    # Flowrate conservation loss (column 0 not used by this loss)
    flowrate_conservation, flowrate_conservation_raw = compute_flowrate_conservation_loss(
        f_pred, f_initial_epoch, model.output_scale, mse_loss
    )

    # Monotonicity loss
    min_dynamics_max_time = (
        f_initial_epoch[:, 0].max().item() if extend_target_time is not None else None
    )
    (monotonicity_loss, rate_decrease_loss, min_dynamics_loss,
     monotonicity_raw, rate_decrease_raw, min_dynamics_raw) = compute_monotonicity_loss_autograd(
        model, f_initial_dyn, time_points, model.output_scale,
        min_rate_threshold=min_rate_threshold,
        min_dynamics_max_time=min_dynamics_max_time,
    )

    # Carbon balance — pointwise d(FA+FB+FC+FD)/dt = 0.
    if compute_carbon_balance and dFsum_dt_all is not None:
        carbon_balance_loss = ((dFsum_dt_all / model.output_scale) ** 2).mean()
        carbon_balance_raw = (dFsum_dt_all ** 2).mean()
    else:
        carbon_balance_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        carbon_balance_raw = torch.tensor(0.0, device=device, dtype=torch.float64)

    return {
        'f_pred': f_pred,
        'time_points': time_points,
        'x_transformed': x_transformed,
        'f_recalculated': f_recalculated,
        'f_pred_at_target': f_pred_at_target,
        'x_pred_at_target': x_pred_at_target,
        'loss_data': loss_data,
        'loss_data_raw': loss_data_raw,
        'loss_initial_condition': loss_initial_condition,
        'loss_initial_condition_raw': loss_initial_condition_raw,
        'loss_pde_col': loss_pde_col,
        'loss_phase_transformed': loss_phase_transformed,
        'loss_phase_raw': loss_phase_raw,
        'flowrate_conservation': flowrate_conservation,
        'flowrate_conservation_raw': flowrate_conservation_raw,
        'pde_residuals': (r1_col, r2_col, r3_col, r4_col),
        'monotonicity_loss': monotonicity_loss,
        'monotonicity_raw': monotonicity_raw,
        'rate_decrease_loss': rate_decrease_loss,
        'rate_decrease_raw': rate_decrease_raw,
        'min_dynamics_loss': min_dynamics_loss,
        'min_dynamics_raw': min_dynamics_raw,
        'carbon_balance_loss': carbon_balance_loss,
        'carbon_balance_raw': carbon_balance_raw,
    }


def compute_total_loss(
    loss_results,
    weights,
    task,
    run_mode,
    include_initial_loss,
    include_phase_loss,
    device,
    include_monotonicity_loss=True,
):
    """
    Compute the weighted total loss from individual loss components.

    Args:
        loss_results: Dictionary from compute_all_losses()
        weights: Weight dictionary for each loss term
        task: 'train' or 'predict'
        run_mode: 'pinn' or 'pinn+phase'
        include_initial_loss: Whether to include initial condition loss
        include_phase_loss: Whether to include phase loss (depends on epoch)
        device: Torch device
        include_monotonicity_loss: Whether to include monotonicity constraints

    Returns:
        Tuple of (total_loss, loss_components_dict)
    """
    loss_data = loss_results['loss_data']
    loss_pde_col = loss_results['loss_pde_col']
    loss_initial_condition = loss_results['loss_initial_condition']
    loss_phase_transformed = loss_results['loss_phase_transformed']
    flowrate_conservation = loss_results['flowrate_conservation']
    monotonicity_loss = loss_results['monotonicity_loss']
    rate_decrease_loss = loss_results['rate_decrease_loss']
    carbon_balance_loss = loss_results['carbon_balance_loss']

    loss_components = [
        weights['pde'] * loss_pde_col,
        weights['flowrate'] * flowrate_conservation,
    ]

    if 'carbon_balance' in weights:
        loss_components.append(weights['carbon_balance'] * carbon_balance_loss)

    if run_mode == 'pinn+phase' and include_phase_loss:
        loss_components.append(weights['phase'] * loss_phase_transformed)

    if task in ('train', 'extend_finetune'):
        loss_components.append(weights['data'] * loss_data)

    if include_initial_loss:
        loss_components.append(weights['initial'] * loss_initial_condition)

    # Add monotonicity constraints for physically correct reaction dynamics
    if include_monotonicity_loss:
        monotonicity_weight = weights.get('monotonicity', 1.0)
        rate_decrease_weight = weights.get('rate_decrease', 1.0)
        loss_components.append(monotonicity_weight * monotonicity_loss)
        loss_components.append(rate_decrease_weight * rate_decrease_loss)

    # Add min dynamics penalty
    if 'min_dynamics' in weights:
        min_dynamics_loss = loss_results['min_dynamics_loss']
        loss_components.append(weights['min_dynamics'] * min_dynamics_loss)

    loss = sum(loss_components)

    return loss


def sanitize_losses(loss_results, device):
    """
    Check for NaN/Inf in losses and replace with safe fallback values.
    
    Args:
        loss_results: Dictionary from compute_all_losses()
        device: Torch device
    
    Returns:
        Updated loss_results dictionary with sanitized values
    """
    loss_data = loss_results['loss_data']
    loss_pde_col = loss_results['loss_pde_col']
    loss_phase_transformed = loss_results['loss_phase_transformed']
    flowrate_conservation = loss_results['flowrate_conservation']
    
    if torch.isnan(loss_data) or torch.isinf(loss_data):
        loss_results['loss_data'] = torch.tensor(1000.0, device=device, dtype=torch.float64)
        logger.warning("NaN detected in loss_data - using fallback value")
    
    if torch.isnan(loss_pde_col) or torch.isinf(loss_pde_col):
        loss_results['loss_pde_col'] = torch.tensor(1000.0, device=device, dtype=torch.float64)
        logger.warning("NaN detected in loss_pde_col - using fallback value")
    
    if torch.isnan(loss_phase_transformed) or torch.isinf(loss_phase_transformed):
        loss_results['loss_phase_transformed'] = torch.tensor(0.0, device=device, dtype=torch.float64)
        logger.warning("NaN detected in loss_phase_transformed - using fallback value")
    
    if torch.isnan(flowrate_conservation) or torch.isinf(flowrate_conservation):
        loss_results['flowrate_conservation'] = torch.tensor(1000.0, device=device, dtype=torch.float64)
        logger.warning("NaN detected in flowrate_conservation - using fallback value")

    carbon_balance_loss = loss_results.get('carbon_balance_loss', None)
    if carbon_balance_loss is not None and (torch.isnan(carbon_balance_loss) or torch.isinf(carbon_balance_loss)):
        loss_results['carbon_balance_loss'] = torch.tensor(1000.0, device=device, dtype=torch.float64)
        logger.warning("NaN detected in carbon_balance_loss - using fallback value")

    return loss_results


def build_loss_terms_dict(loss_results, run_mode, include_phase, epoch, phase_intro_epoch):
    """
    Build a dictionary of loss terms for GradNorm weight updates.
    
    Args:
        loss_results: Dictionary from compute_all_losses()
        run_mode: 'pinn' or 'pinn+phase'
        include_phase: Whether to include phase loss
        epoch: Current epoch
        phase_intro_epoch: Epoch when phase loss is introduced
    
    Returns:
        Dictionary of loss terms
    """
    loss_terms = {
        'data': loss_results['loss_data'],
        'pde': loss_results['loss_pde_col'],
        'initial': loss_results['loss_initial_condition'],
        'flowrate': loss_results['flowrate_conservation'],
        'carbon_balance': loss_results['carbon_balance_loss'],
        'monotonicity': loss_results['monotonicity_loss'],
        'rate_decrease': loss_results['rate_decrease_loss'],
        'min_dynamics': loss_results['min_dynamics_loss'],
    }

    if run_mode == 'pinn+phase' and include_phase and epoch >= phase_intro_epoch:
        loss_terms['phase'] = loss_results['loss_phase_transformed']

    return loss_terms
