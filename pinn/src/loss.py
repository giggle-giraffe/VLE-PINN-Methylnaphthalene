# ------------------------------------------------------------------------------------------
# Copyright (c) 2026 Chen Zhang, Tao Li, Xiaoqian Dang
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

import torch
from loguru import logger
import os
import sys
import numpy as np


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "model")


def compute_initial_condition_loss(f_pred, f_initial, output_scale, mse_loss_fn):
    """
    Compute the initial condition loss in normalized space for consistent gradients.

    The normalization is critical because:
    - Raw flow rates are ~1e-5 scale, so raw IC loss would be ~1e-10
    - Such tiny loss values produce negligible gradients
    - Normalizing by output_scale brings loss to ~1-100 scale with meaningful gradients

    Args:
        f_pred: Predicted flow rates tensor [batch_size, n_time_steps, 4]
                First time step (index 0) contains initial predictions for [FA, FB, FC, FD]
        f_initial: Initial conditions tensor [batch_size, n_features]
                   Columns 6:10 contain [FA_in, FB_in, FC_in, FD_in]
        output_scale: Scale factor used in model (typically 1e-6)
        mse_loss_fn: MSE loss function (torch.nn.MSELoss)

    Returns:
        Tuple of (normalized_loss, raw_loss)
        - normalized_loss: Loss in normalized space (~1-100 scale) for training
        - raw_loss: Raw MSE loss (~1e-10 scale) for logging/monitoring
    """
    # Extract predictions at t=0 and true initial values
    f_pred_ic = f_pred[:, 0, :4]

    f_target_ic = f_initial[:, 6:10]

    f_pred_ic_norm = f_pred_ic / output_scale
    f_target_ic_norm = f_target_ic / output_scale

    normalized_loss = mse_loss_fn(f_pred_ic_norm, f_target_ic_norm)
    raw_loss = mse_loss_fn(f_pred_ic, f_target_ic)

    return normalized_loss, raw_loss


def compute_data_loss(f_pred_at_target, f_target, output_scale, mse_loss_fn):
    """
    Compute the data loss in normalized space for consistent gradients.

    The normalization is critical because:
    - Raw flow rates are ~1e-5 scale, so raw MSE would be ~1e-10
    - Such tiny loss values produce negligible gradients
    - Normalizing by output_scale brings loss to ~1-100 scale with meaningful gradients

    This mirrors the normalization applied in compute_initial_condition_loss to ensure
    both losses contribute meaningfully during training.

    Args:
        f_pred_at_target: Predicted flow rates at target time [batch_size, 4]
                          Contains [FA_out, FB_out, FC_out, FD_out]
        f_target: Target flow rates tensor [batch_size, 4]
                  Contains true outlet flow rates [FA_out, FB_out, FC_out, FD_out]
        output_scale: Scale factor used in model (typically 1e-6)
        mse_loss_fn: MSE loss function (torch.nn.MSELoss)

    Returns:
        Tuple of (normalized_loss, raw_loss)
        - normalized_loss: Loss in normalized space (~1-100 scale) for training
        - raw_loss: Raw MSE loss (~1e-10 scale) for logging/monitoring
    """
    # Normalize both by output_scale for consistent gradient magnitudes
    # This transforms loss from ~1e-10 scale to ~1-100 scale
    f_pred_norm = f_pred_at_target / output_scale
    f_target_norm = f_target / output_scale

    normalized_loss = mse_loss_fn(f_pred_norm, f_target_norm)
    raw_loss = mse_loss_fn(f_pred_at_target, f_target)

    return normalized_loss, raw_loss


def compute_data_loss_with_coverage(
    predictions,
    targets,
    output_scale=None,
    coverage_weights=None,
    device=None
):
    """
    Compute the data loss with optional time curriculum coverage weighting.

    This function supports both run modes:
    - pinn+phase: Uses output_scale normalization for flow rate predictions
    - pinn: No normalization needed (mole fractions are already ~0-1 scale)

    Args:
        predictions: Predicted values [batch_size, n_features]
        targets: Target values [batch_size, n_features]
        output_scale: Scale factor for normalization (None for pinn mode)
        coverage_weights: Per-sample weights for time curriculum [batch_size] or None
        device: Torch device for creating tensors

    Returns:
        Tuple of (weighted_loss, raw_loss)
        - weighted_loss: Coverage-weighted loss for training
        - raw_loss: Unweighted MSE loss for logging/monitoring
    """
    # Normalize if output_scale provided (pinn+phase mode)
    if output_scale is not None:
        pred_norm = predictions / output_scale
        target_norm = targets / output_scale
    else:
        pred_norm = predictions
        target_norm = targets

    # Compute per-sample loss for coverage weighting
    per_sample_loss = ((pred_norm - target_norm) ** 2).mean(dim=1)  # [batch_size]

    # Apply coverage weights if provided
    if coverage_weights is not None:
        weighted_loss = (coverage_weights * per_sample_loss).mean()
    else:
        weighted_loss = per_sample_loss.mean()

    # Raw loss for logging (unweighted, unnormalized)
    raw_loss = torch.nn.functional.mse_loss(predictions, targets)

    return weighted_loss, raw_loss


def compute_phase_loss(f_pred, f_recalculated, output_scale, mse_loss_fn):
    """
    Compute the phase consistency loss in normalized space for consistent gradients.

    This loss ensures that the predicted flow rates are consistent with the
    phase equilibrium calculations (f_pred should match f_recalculated from phase model).

    The normalization is critical because:
    - Raw flow rates are ~1e-5 scale, so raw MSE would be ~1e-10
    - Such tiny loss values produce negligible gradients
    - Normalizing by output_scale brings loss to ~1-100 scale with meaningful gradients

    Args:
        f_pred: Predicted flow rates [batch_size, time_steps, 4]
                Contains [FA_out, FB_out, FC_out, FD_out]
        f_recalculated: Recalculated flow rates from phase model [batch_size, time_steps, >=6]
                        Columns 2:6 contain recalculated [FA, FB, FC, FD]
        output_scale: Scale factor used in model (typically 1e-6)
        mse_loss_fn: MSE loss function (torch.nn.MSELoss)

    Returns:
        Tuple of (normalized_loss, raw_loss)
        - normalized_loss: Loss in normalized space (~1-100 scale) for training
        - raw_loss: Raw MSE loss (~1e-10 scale) for logging/monitoring
    """
    f_pred_slice = f_pred[:, 1:, 0:4]
    f_recalc_slice = f_recalculated[:, 1:, 2:6]

    # Normalize both by output_scale for consistent gradient magnitudes
    f_pred_norm = f_pred_slice / output_scale
    f_recalc_norm = f_recalc_slice / output_scale

    normalized_loss = mse_loss_fn(f_pred_norm, f_recalc_norm)
    raw_loss = mse_loss_fn(f_pred_slice, f_recalc_slice)

    return normalized_loss, raw_loss


def compute_flowrate_conservation_loss(f_pred, f_initial, output_scale, mse_loss_fn):
    """
    Compute total molar flow rate conservation penalty in normalized space.

    This loss enforces that total outlet flow rate (FA+FB+FC+FD) equals total inlet flow rate,
    ensuring flow rate conservation in the reaction system (A→B→C→D preserves total moles).

    The normalization is critical because:
    - Raw flow rates are ~1e-5 scale, so raw MSE would be ~1e-10
    - Such tiny loss values produce negligible gradients
    - Normalizing by output_scale brings loss to ~1-100 scale with meaningful gradients

    Args:
        f_pred: Predicted flow rates [batch_size, time_steps, 4+]
                First 4 columns: [FA_out, FB_out, FC_out, FD_out]
        f_initial: Initial flow rates [batch_size, n_features]
                   Columns 6:10 contain [FA_in, FB_in, FC_in, FD_in]
        output_scale: Scale factor used in model (typically 1e-6)
        mse_loss_fn: MSE loss function (torch.nn.MSELoss)

    Returns:
        Tuple of (normalized_loss, raw_loss)
        - normalized_loss: Loss in normalized space (~1-100 scale) for training
        - raw_loss: Raw MSE loss (~1e-10 scale) for logging/monitoring
    """
    # Calculate total inlet molar flow rate from initial conditions
    true_total_flow = torch.sum(f_initial[:, 6:10], dim=1, keepdim=True)  # [batch_size, 1]

    # Calculate total outlet molar flow rate at each timestep
    FA_out = f_pred[:, :, 0]
    FB_out = f_pred[:, :, 1]
    FC_out = f_pred[:, :, 2]
    FD_out = f_pred[:, :, 3]
    predicted_total_flow = FA_out + FB_out + FC_out + FD_out  # [batch_size, time_steps]

    # Normalize by output_scale for consistent gradient magnitudes
    pred_norm = predicted_total_flow / output_scale
    true_norm = true_total_flow / output_scale

    normalized_loss = mse_loss_fn(pred_norm, true_norm)
    raw_loss = mse_loss_fn(predicted_total_flow, true_total_flow)

    return normalized_loss, raw_loss


def hybrid_concentration_loss(pred, target, epsilon=1e-6, weight_relative=0.6, return_per_sample=False):
    """
    Robust hybrid loss optimized for multi-scale chemical concentration data
    
    Args:
        pred: Predicted values
        target: Target values
        epsilon: Small value for numerical stability
        weight_relative: Weight for relative error component (0-1)
        
    Returns:
        Weighted combination of relative and log-space errors
    """
    # 1. Ensure positive values with proper clamping
    pred_safe = torch.clamp(pred, min=epsilon)
    target_safe = torch.clamp(target, min=epsilon)
    
    # Calculate relative error per sample
    rel_error = torch.abs(pred_safe - target_safe) / (target_safe + epsilon)
    
    # 3. Calculate log-space error to handle order-of-magnitude differences
    log_pred = torch.log10(pred_safe)
    log_target = torch.log10(target_safe)
    log_error = torch.abs(log_pred - log_target)
    
    # 4. Combine both errors with proper weighting
    combined_error = weight_relative * rel_error + (1 - weight_relative) * log_error

    # Get per-sample losses
    per_sample_loss = torch.mean(combined_error, dim=-1)  # Average over species dimension only
    
    if return_per_sample:
        return per_sample_loss
    
    # Average over samples
    return torch.mean(per_sample_loss)


def compute_monotonicity_loss_autograd(model, f_initial, time_points, output_scale, **kwargs):
    """
    Compute physics-based monotonicity constraints using autograd for exact derivatives.
    (More accurate than finite difference, consistent with PDE loss computation)

    For a consumption reaction A → products:
    1. dFA/dt < 0: FA should always decrease (reactant consumed)
    2. |dFA/dt| should decrease over time: rate is fastest at t=0 when [A] is highest
    3. |dFA/dt| must exceed a minimum threshold (prevents trivial 0=0 solutions)

    Args:
        model: PINN model instance
        f_initial: Initial features [batch_size, n_features]
        time_points: Time points tensor [n_time_points]
        output_scale: Scale factor used in model for normalization
        **kwargs: Optional keyword arguments:
            min_rate_threshold: Minimum |dFA/dt| threshold (default 1e-5)
            min_dynamics_max_time: If set, only time points <= this value are scored
                against min_rate_threshold. Used by extend_finetune so FA's natural
                approach to equilibrium past the experimental window doesn't show
                up as a min_dynamics violation.

    Returns:
        Tuple of (monotonicity_loss, rate_decrease_loss, min_dynamics_loss,
                  raw_monotonicity, raw_rate_decrease, raw_min_dynamics)
    """
    device = model.device
    batch_size = f_initial.shape[0]
    n_time_points = len(time_points)

    # Extract sample features (same as in PDE loss).
    # SampleEncoder handles boundary conditions only; time is owned by TimeEncoder.
    sample_features = torch.cat([
        f_initial[:, 1:2],   # T
        f_initial[:, 2:3],   # P
        f_initial[:, 3:4],   # m (catalyst mass)
        f_initial[:, 4:5],   # FH0
        f_initial[:, 5:6],   # FS
        f_initial[:, 6:7],   # FA_in
        f_initial[:, 7:8],   # FB_in
        f_initial[:, 8:9],   # FC_in
        f_initial[:, 9:10],  # FD_in
    ], dim=1)

    # Encode sample features once
    sample_encoding = model.sample_encoder(sample_features)

    # Collect dFA/dt at each time point using autograd
    all_dFA_dt = []

    for t_idx in range(n_time_points):
        # Create time tensor with gradient tracking
        t_tensor = torch.ones(batch_size, 1, dtype=torch.float64, device=device) * time_points[t_idx]
        t_tensor.requires_grad_(True)

        # Encode time
        time_encoding = model.time_encoder(t_tensor)

        # Forward pass
        if model.pinn_inputs == 'time':
            f_pred_batch = model.net(t_tensor)
        elif model.pinn_inputs == 'time+initials':
            combined_features = model.feature_fusion(time_encoding, sample_encoding)
            f_pred_batch = model.net(combined_features)

        # Apply output scale
        f_pred_batch = f_pred_batch * model.output_scale

        # Extract FA (first output)
        FA = f_pred_batch[:, 0]

        # Compute dFA/dt using autograd
        batch_ones = torch.ones_like(FA)
        dFA_dt = torch.autograd.grad(
            outputs=FA,
            inputs=t_tensor,
            grad_outputs=batch_ones,
            create_graph=True,
            retain_graph=True
        )[0].squeeze(1)

        all_dFA_dt.append(dFA_dt)

    # Stack all derivatives: [n_time_points, batch_size]
    dFA_dt_all = torch.stack(all_dFA_dt, dim=0)  # [n_time_points, batch_size]
    dFA_dt_all = dFA_dt_all.T  # [batch_size, n_time_points]

    monotonicity_violation = torch.relu(dFA_dt_all)  # [batch_size, n_time_points]

    monotonicity_violation_norm = monotonicity_violation / output_scale
    monotonicity_loss = monotonicity_violation_norm.mean()
    raw_monotonicity = monotonicity_violation.mean()

    abs_dFA_dt = torch.abs(dFA_dt_all)  # [batch_size, n_time_points]

    rate_change = abs_dFA_dt[:, 1:] - abs_dFA_dt[:, :-1]  # [batch_size, n_time_points - 1]

    rate_increase_violation = torch.relu(rate_change)

    rate_increase_violation_norm = rate_increase_violation / output_scale
    rate_decrease_loss = rate_increase_violation_norm.mean()
    raw_rate_decrease = rate_increase_violation.mean()

    min_rate_threshold = kwargs.get('min_rate_threshold', 1e-5)
    trivial_violation = torch.relu(min_rate_threshold - abs_dFA_dt)  # [batch_size, n_time_points]
    trivial_violation_norm = trivial_violation / output_scale

    min_dynamics_max_time = kwargs.get('min_dynamics_max_time', None)
    if min_dynamics_max_time is not None:
        time_mask = (time_points <= min_dynamics_max_time).to(device=device, dtype=trivial_violation_norm.dtype)
        time_mask = time_mask.unsqueeze(0)  # [1, n_time_points]
        denom = time_mask.sum().clamp(min=1.0) * batch_size
        min_dynamics_loss = (trivial_violation_norm * time_mask).sum() / denom
        raw_min_dynamics = (trivial_violation * time_mask).sum() / denom
    else:
        min_dynamics_loss = trivial_violation_norm.mean()
        raw_min_dynamics = trivial_violation.mean()

    return (monotonicity_loss, rate_decrease_loss, min_dynamics_loss,
            raw_monotonicity, raw_rate_decrease, raw_min_dynamics)


def compute_carbon_balance_loss_autograd(model, f_initial, time_points, output_scale, **kwargs):
    """
    Compute the instantaneous carbon-balance residual using autograd.

    For the MN reaction network (A→B→D, A→C→D), the four species in
    {A, B, C, D} carry all of the
    carbon-bearing mass entering the reactor as A. Mass conservation therefore
    requires d(FA + FB + FC + FD)/dt = 0 at every point along the trajectory.
    The flow-rate-conservation loss enforces a level constraint
    (sum == FA_in) but is dominated by other tasks under GradNorm; this
    pointwise rate constraint adds a complementary, derivative-level penalty.

    Args:
        model: PINN model instance. Sample features are sliced generically
               via the model's num_sample_features.
        f_initial: Initial features [batch_size, n_features]. Column 0 is the
                   per-sample target time; columns 1..1+num_sample_features
                   carry the boundary conditions.
        time_points: Time points tensor [n_time_points] (typically the PDE
                     collocation grid, possibly extending past per-sample t_x).
        output_scale: Scale factor used in the model's output_scale. dF_sum/dt
                      is divided by output_scale before the squared mean for
                      consistent gradient magnitude with flowrate_conservation.
        **kwargs: Reserved for future options.

    Returns:
        Tuple (carbon_balance_loss, raw_carbon_balance) where
            carbon_balance_loss = mean( (dF_sum/dt / output_scale)^2 )
            raw_carbon_balance  = mean( (dF_sum/dt)^2 )
    """
    device = model.device
    batch_size = f_initial.shape[0]
    n_time_points = len(time_points)

    # Slice boundary-condition features generically.
    # `num_sample_features` is stored on the SampleEncoder, not on the PINN itself.
    n_sf = model.sample_encoder.num_sample_features
    sample_features = f_initial[:, 1:1 + n_sf]

    # Encode sample features once
    sample_encoding = model.sample_encoder(sample_features)

    all_dFsum_dt = []

    for t_idx in range(n_time_points):
        t_tensor = torch.ones(batch_size, 1, dtype=torch.float64, device=device) * time_points[t_idx]
        t_tensor.requires_grad_(True)

        time_encoding = model.time_encoder(t_tensor)

        if model.pinn_inputs == 'time':
            f_pred_batch = model.net(t_tensor)
        elif model.pinn_inputs == 'time+initials':
            combined_features = model.feature_fusion(time_encoding, sample_encoding)
            f_pred_batch = model.net(combined_features)
        else:
            raise ValueError(f"Invalid PINN inputs: {model.pinn_inputs}")

        f_pred_batch = f_pred_batch * model.output_scale

        # Sum of carbon-bearing species: FA + FB + FC + FD (first 4 outputs)
        F_sum = f_pred_batch[:, 0] + f_pred_batch[:, 1] + f_pred_batch[:, 2] + f_pred_batch[:, 3]

        batch_ones = torch.ones_like(F_sum)
        dFsum_dt = torch.autograd.grad(
            outputs=F_sum,
            inputs=t_tensor,
            grad_outputs=batch_ones,
            create_graph=True,
            retain_graph=True,
        )[0].squeeze(1)

        all_dFsum_dt.append(dFsum_dt)

    dFsum_dt_all = torch.stack(all_dFsum_dt, dim=0).T  # [batch_size, n_time_points]

    # Squared deviation from zero, normalized for consistent gradient magnitude.
    dFsum_dt_norm = dFsum_dt_all / output_scale
    carbon_balance_loss = (dFsum_dt_norm ** 2).mean()
    raw_carbon_balance = (dFsum_dt_all ** 2).mean()

    return carbon_balance_loss, raw_carbon_balance


def r2_score(y_true, y_pred):
    """
    Calculate the R² score (coefficient of determination) between true and predicted values
    
    Args:
        y_true: Tensor or numpy array of true values
        y_pred: Tensor or numpy array of predicted values
        
    Returns:
        R² score as a scalar
    """
    # Check if inputs are numpy arrays
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        # NumPy implementation
        total_sum_squares = np.sum((y_true - np.mean(y_true))**2)
        residual_sum_squares = np.sum((y_true - y_pred)**2)
        r2 = 1 - (residual_sum_squares / total_sum_squares)
        return r2
    else:
        # PyTorch implementation
        total_sum_squares = torch.sum((y_true - torch.mean(y_true))**2)
        residual_sum_squares = torch.sum((y_true - y_pred)**2)
        r2 = 1 - (residual_sum_squares / total_sum_squares)
        return r2.item() if hasattr(r2, 'item') else r2


def compute_pde_loss_mn(model, f_initial=None, time_points=None, debug=False,
                        return_carbon_balance=False):
    """
    Compute the PDE residual using predicted molecular fractions and environment parameters

    Args:
        model: PINN model
        f_initial: Initial flow rates [batch_size, n_features]
        time_points: Tensor of time points [n_time_points]. These can be arbitrary
                     continuous values (not tied to the forward pass grid).
        debug: Whether to print debug information
        return_carbon_balance: If True, also compute d(FA+FB+FC+FD)/dt at each
            collocation point and return it as a 5th tuple element. Reuses the
            forward graph and t_tensor that already exist for the PDE residual,
            adding only one extra autograd.grad call per time point. Avoids the
            ~5 GiB cost of a separate forward pass loop in
            compute_carbon_balance_loss_autograd.

    Returns:
        Tuple of residuals for each species. If return_carbon_balance is True, a
        5th element dFsum_dt_all of shape [batch_size, n_time_points] is appended.
    """
    device = model.device
    batch_size = f_initial.shape[0]

    n_time_points = len(time_points)

    def debug_parameter_version(name, param):
        if hasattr(param, '_version'):
            logger.debug(f"Parameter '{name}' shape={param.shape}, version={param._version}, "
                         f"requires_grad={param.requires_grad}")
        return param

    # -----Constants-----
    A_TRANSFORM_BASE = 10.0
    E_TRANSFORM_OFFSET = 1000.0
    E_MIN = getattr(model, 'e_min', 1000.0)
    E_MAX = 80000.0
    E_RANGE = E_MAX - E_MIN
    R = 8.314
    r_inv = 1.0 / R
    RXN_A_LOG_MIN = getattr(model, 'rxn_a_log_min', -1.0)
    RXN_A_LOG_RANGE = getattr(model, 'rxn_a_log_range', 9.0)
    ADS_A_LOG_MIN = getattr(model, 'ads_a_log_min', -3.0)
    ADS_A_LOG_RANGE = getattr(model, 'ads_a_log_range', 6.0)
    ADS_DH_MAX = getattr(model, 'ads_dh_max', 30000.0)
    ADS_DH_MULTIPLIER = ADS_DH_MAX - E_TRANSFORM_OFFSET

    # -----Initialize residual collections-----
    all_residuals = [[] for _ in range(4)]
    all_dFsum_dt = [] if return_carbon_balance else None

    # -----Extract catalyst mass for dimensionless ODE-----
    m_catalyst = f_initial[:, 3]

    # -----Extract sample features for all samples at once-----
    # SampleEncoder handles boundary conditions only; time is owned by TimeEncoder.
    sample_features = torch.cat([
        f_initial[:, 1:2],   # T
        f_initial[:, 2:3],   # P
        f_initial[:, 3:4],   # m (catalyst mass)
        f_initial[:, 4:5],   # FH0
        f_initial[:, 5:6],   # FS
        f_initial[:, 6:7],   # FA_in
        f_initial[:, 7:8],   # FB_in
        f_initial[:, 8:9],   # FC_in
        f_initial[:, 9:10],  # FD_in
    ], dim=1)

    # -----Encode sample features for all samples-----
    sample_encoding = model.sample_encoder(sample_features)  # [batch_size, encoding_dim]

    # -----Pre-compute helper functions for rate and equilibrium constants-----
    def compute_rate_constant(x_A_raw, y_E_raw, T):
        A = A_TRANSFORM_BASE ** (RXN_A_LOG_MIN + RXN_A_LOG_RANGE * torch.sigmoid(x_A_raw))
        E = E_MIN + E_RANGE * torch.sigmoid(y_E_raw)
        # Rate constant: k = A * exp(-E / RT)
        return A * torch.exp(-E * r_inv / T)

    def compute_equil_constant(x_A_raw, z_Delta_H_raw, T):
        A = A_TRANSFORM_BASE ** (ADS_A_LOG_MIN + ADS_A_LOG_RANGE * torch.sigmoid(x_A_raw))
        Delta_H = E_TRANSFORM_OFFSET + ADS_DH_MULTIPLIER * torch.sigmoid(z_Delta_H_raw)
        return A * torch.exp(Delta_H * r_inv / T)

    # -----Process time points-----
    for t_idx in range(n_time_points):
        # Create time tensor for all samples at this time point
        t_tensor = torch.ones(batch_size, 1, dtype=torch.float64, device=device) * time_points[t_idx]
        t_tensor.requires_grad_(True)  # Enable gradient tracking

        # Encode time for all samples
        time_encoding = model.time_encoder(t_tensor)  # [batch_size, time_encoder_dim]

        # Process through feature extractor with per-sample processing
        if model.pinn_inputs == 'time':
            f_pred_batch = model.net(t_tensor)
        elif model.pinn_inputs == 'time+initials':
            # Each sample gets its own feature fusion
            combined_features = model.feature_fusion(time_encoding, sample_encoding)
            f_pred_batch = model.net(combined_features)
        else:
            raise ValueError(f"Invalid PINN inputs: {model.pinn_inputs}")

        f_pred_batch = f_pred_batch * model.output_scale

        # Extract predicted flow rates from model output (4 outputs)
        FA_out, FB_out, FC_out, FD_out = [f_pred_batch[:, i] for i in range(4)]

        if return_carbon_balance:
            F_sum_t = FA_out + FB_out + FC_out + FD_out
            dFsum_dt_t = torch.autograd.grad(
                outputs=F_sum_t,
                inputs=t_tensor,
                grad_outputs=torch.ones_like(F_sum_t),
                create_graph=True,
                retain_graph=True,
            )[0].squeeze(1)
            all_dFsum_dt.append(dFsum_dt_t)

        # Extract input conditions from f_initial
        T = f_initial[:, 1]
        P = f_initial[:, 2]
        FH0 = f_initial[:, 4]
        FS = f_initial[:, 5]
        FA_in = f_initial[:, 6]

        # Flash calculation (MN stoichiometry)
        FH2_out = FH0 - 2.0 * FB_out - 2.0 * FC_out - 5.0 * FD_out

        time_col = torch.full((batch_size,), time_points[t_idx].item(), dtype=torch.float64, device=model.device)
        step_f_all = torch.stack([
            time_col,
            T,
            P,
            FH2_out,
            FS,
            FA_out,
            FB_out,
            FC_out,
            FD_out
        ], dim=1)

        x_phase_mole_fractions, _ = model.calc_phase(step_f=step_f_all)

        ya_mf = x_phase_mole_fractions[:, 3]
        yb_mf = x_phase_mole_fractions[:, 4]
        yc_mf = x_phase_mole_fractions[:, 5]
        yd_mf = x_phase_mole_fractions[:, 6]
        xa_mf = x_phase_mole_fractions[:, 9]
        xb_mf = x_phase_mole_fractions[:, 10]
        xc_mf = x_phase_mole_fractions[:, 11]
        xd_mf = x_phase_mole_fractions[:, 12]

        yh2_mf = x_phase_mole_fractions[:, 1]

        ph_batch = yh2_mf * P

        VF_blend = x_phase_mole_fractions[:, 0]
        alpha = torch.sigmoid((VF_blend - 0.99) / 0.001)
        XA_out = (1 - alpha) * xa_mf + alpha * ya_mf
        XB_out = (1 - alpha) * xb_mf + alpha * yb_mf
        XC_out = (1 - alpha) * xc_mf + alpha * yc_mf
        XD_out = (1 - alpha) * xd_mf + alpha * yd_mf

        # Calculate all gradients with a single backward pass
        all_grads = []
        for species_out in [XA_out, XB_out, XC_out, XD_out]:
            # Create batch-sized vector of ones to preserve batch dimension
            batch_ones = torch.ones_like(species_out)
            grad = torch.autograd.grad(
                outputs=species_out,
                inputs=t_tensor,
                grad_outputs=batch_ones,
                create_graph=True,
                retain_graph=True
            )[0]
            all_grads.append(grad.squeeze(1))

        dXA_dt, dXB_dt, dXC_dt, dXD_dt = all_grads

        # Get environment parameters for rate calculations
        T_batch = T
        
        # Compute rate constants for all samples in batch
        k_values = []
        for idx in range(1, 5):
            A_param = getattr(model, f'x_A_{idx}')
            E_param = getattr(model, f'y_E_{idx}')
            k_values.append(compute_rate_constant(A_param, E_param, T_batch))
        
        k_1, k_2, k_3, k_4 = k_values

        # Compute equilibrium constants
        K_values = []
        for s in ['H', 'A', 'B', 'C', 'D']:
            A_param = getattr(model, f'x_A_{s}')
            H_param = getattr(model, f'z_Delta_H_{s}')
            K_values.append(compute_equil_constant(A_param, H_param, T_batch))

        K_H, K_A, K_B, K_C, K_D = K_values

        # Calculate denominator
        ADS = 1.0 + K_H * ph_batch + K_A * XA_out + K_B * XB_out + K_C * XC_out + K_D * XD_out

        # Pre-compute common terms
        term_1A = k_1 * K_H * ph_batch * K_A * XA_out / ADS
        term_3A = k_3 * K_H * ph_batch * K_A * XA_out / ADS
        term_2B = k_2 * K_H * ph_batch * K_B * XB_out / ADS
        term_4C = k_4 * K_H * ph_batch * K_C * XC_out / ADS

        # Calculate reaction terms (net rate of change for each species)
        reaction_term_A = -(term_1A + term_3A)
        reaction_term_B = term_1A - term_2B
        reaction_term_C = term_3A - term_4C
        reaction_term_D = term_2B + term_4C

        residuals = [
            dXA_dt / m_catalyst - reaction_term_A,
            dXB_dt / m_catalyst - reaction_term_B,
            dXC_dt / m_catalyst - reaction_term_C,
            dXD_dt / m_catalyst - reaction_term_D
        ]

        for i in range(len(residuals)):
            all_residuals[i].append(residuals[i])

        # Free memory every few iterations
        if (t_idx + 1) % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----Concatenate all residuals-----
    final_residuals = [torch.cat(res_list) for res_list in all_residuals]

    if return_carbon_balance:
        # Stack into [batch_size, n_time_points] so callers can take MSE directly.
        dFsum_dt_all = torch.stack(all_dFsum_dt, dim=0).T
        return tuple(final_residuals) + (dFsum_dt_all,)

    return tuple(final_residuals)