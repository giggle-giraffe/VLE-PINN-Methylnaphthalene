# ------------------------------------------------------------------------------------------
# Copyright (c) 2026 Chen Zhang, Tao Li, Xiaoqian Dang
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from loguru import logger
from pathlib import Path
import os
import time
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "model")

from .loss import (
    compute_pde_loss_mn,
    compute_initial_condition_loss,
    compute_data_loss_with_coverage,
    compute_phase_loss,
    compute_flowrate_conservation_loss,
    compute_monotonicity_loss_autograd,
)

from .time_curriculum import (
    get_pde_concentration,
    sample_pde_collocation_points,
)

from .training import (
    save_checkpoint,
    load_checkpoint,
    plot_gradnorm_analysis,
    gradnorm_update_weights,
    parse_initial_weights,
    parse_curriculum_stages,
    derive_phase_intro_epoch,
    parse_gradnorm_params,
    parse_target_rates,
    parse_weight_caps,
    parse_time_curriculum,
    parse_learning_rate_schedule,
    detect_weight_explosion,
    parse_gradnorm_excluded_tasks,
    interpolate_curriculum_weight,
    compute_all_losses,
    compute_total_loss,
    sanitize_losses,
    build_loss_terms_dict,
)

from .util import KINETIC_PARAM_PREFIXES, is_kinetic_param


def toggle_parameter_freezing(model, epoch, phase_transition_epoch=300, frozen_parameters=None):
    """Freeze/unfreeze parameters based on training phase
    
    In Phase 1 (epoch < phase_transition_epoch): Freeze dynamics parameters
    In Phase 2 (epoch >= phase_transition_epoch): Unfreeze all parameters
    """
    if frozen_parameters is None:
        frozen_params = list(KINETIC_PARAM_PREFIXES)
    else:
        frozen_params = frozen_parameters
        
    if epoch < phase_transition_epoch:
        for name, param in model.named_parameters():
            if any(x in name for x in frozen_params):
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        if epoch == 0 or epoch % 50 == 0:  # Log only occasionally
            logger.info(f"Phase 1 (epoch {epoch}): Training initial condition parameters only")
    
    elif epoch == phase_transition_epoch:
        # Transition to Phase 2: Unfreeze all parameters
        for name, param in model.named_parameters():
            param.requires_grad = True
        logger.info(f"Phase 2 (epoch {epoch}): Unfreezing all parameters for dynamics learning")


def train_pinn(model=None, n_epochs=None, input_features=None, output_targets=None,
               checkpoint_dir=None, model_folder=None, checkpoint_freq=1000, run_mode=None, pinn_inputs=None, resume_from=None,
               task=None, logging_freq=100, adaptive_start_epoch=None,
               pre_training_epochs=300, debug=False, curriculum_config=None,
               extend_target_time=None):
    """
    Train a Physics-Informed Neural Network with regular checkpointing

    Args:
        model: PINN model instance
        n_epochs: Total number of epochs
        input_features: Input features for data loss
        output_targets: Target values for data loss
        checkpoint_dir: Directory to save checkpoints (default: "checkpoints")
        checkpoint_freq: Save checkpoint every N epochs (default: 500)
        run_mode: Mode to run the model ('pinn' or 'pinn+phase')
        pinn_inputs: Inputs to the PINN (['time', 'time+initials'])
        resume_from: Path to checkpoint to resume from (optional)
        task: Task mode ('train' or 'predict')
        logging_freq: Frequency of logging (default: 100)
        adaptive_start_epoch: Epoch to start adaptive weighting (optional)
        pre_training_epochs: Number of epochs for pre-training phase (default: 300)
        debug: Whether to print debug information
        curriculum_config: Curriculum learning configuration dictionary

    Returns:
        parameters_history: History of model parameters
        loss_history: History of loss values
    """
    assert pinn_inputs in ['time', 'time+initials'], f"Invalid PINN inputs: {pinn_inputs}"
    assert run_mode in ['pinn', 'pinn+phase'], f"Invalid run mode: {run_mode}"
    assert task in ['train', 'predict', 'extend_finetune'], \
        f"Invalid task: {task}. Must be 'train', 'predict', or 'extend_finetune'."
    if task == 'extend_finetune':
        assert extend_target_time is not None and extend_target_time > 0, \
            "extend_finetune task requires a positive extend_target_time"
        logger.info(f"extend_finetune mode: trajectory will extend to t={extend_target_time}")

    logging_freq = logging_freq
    
    # -----Enable data shuffling for each epoch-----
    enable_data_shuffling = True  # Toggle for data shuffling

    # -----Enable anomaly detection-----
    if debug:
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.autograd.set_detect_anomaly(False)

    def debug_tensor_version(name, tensor):
        if hasattr(tensor, '_version'):
            logger.debug(f"Tensor '{name}' shape={tensor.shape}, version={tensor._version}, "
                         f"requires_grad={tensor.requires_grad}, is_leaf={tensor.is_leaf}")
        return tensor

    # -----Get device from model-----
    device = model.device

    # -----Move input data to device-----
    input_features = input_features.to(device, dtype=torch.float64)
    output_targets = output_targets.to(device, dtype=torch.float64)

    # -----Setup checkpoint directory-----
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(MODEL_DIR, "default_checkpoints")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # -----Set up training configuration based on input mode-----
    include_initial_loss = True

    # -----Parse all configuration from curriculum_config-----
    weights = parse_initial_weights(curriculum_config, task, run_mode)
    curriculum_stages = parse_curriculum_stages(curriculum_config, task)
    curriculum_end_epoch = adaptive_start_epoch
    gradnorm_alpha, gradnorm_lr, weight_update_freq = parse_gradnorm_params(curriculum_config)
    target_rates = parse_target_rates(curriculum_config)
    weight_caps = parse_weight_caps(curriculum_config, task, run_mode)
    gradnorm_excluded_tasks = parse_gradnorm_excluded_tasks(curriculum_config)
    if gradnorm_excluded_tasks:
        logger.info(f"GradNorm excluded tasks (will use curriculum interpolation): {gradnorm_excluded_tasks}")

    # Min dynamics penalty and data floor protection
    data_floor_mult = curriculum_config.get('gradnorm_data_floor_mult', 0.1) if curriculum_config else 0.1
    max_pde_data_ratio = curriculum_config.get('max_pde_data_ratio', None) if curriculum_config else None
    min_rate_threshold = curriculum_config.get('min_rate_threshold', 1e-5) if curriculum_config else 1e-5

    # -----Time Curriculum Setup (PDE Collocation Sampling)-----
    time_curriculum = parse_time_curriculum(curriculum_config)
    time_curriculum_enabled = time_curriculum['enabled']
    pde_concentration_schedule = time_curriculum['concentration_schedule']
    n_pde_points = time_curriculum['n_pde_points']
    
    # -----Phase loss introduction (derived from stages, single source of truth)-----
    phase_intro_epoch = derive_phase_intro_epoch(curriculum_stages)
    if phase_intro_epoch is not None:
        logger.info(f"Phase loss introduction epoch: {phase_intro_epoch} (derived from curriculum stages)")
    else:
        # No stage has phase > 0; set to inf so all `epoch >= phase_intro_epoch` checks return False
        phase_intro_epoch = float('inf')
        logger.info("Phase loss is never introduced (no stage has phase weight > 0)")
    
    # -----Learning rate schedule-----
    lr_schedule, warmup_epochs, base_lr, min_lr = parse_learning_rate_schedule(curriculum_config)

    # -----Store gradient and weight history for visualization-----
    grad_history = {k: [] for k in weights.keys()}
    weight_history = {k: [] for k in weights.keys()}

    # -----Store loss history for GradNorm (managed tasks only)-----
    loss_history_gradnorm = {k: [] for k in weights.keys() if k not in gradnorm_excluded_tasks}

    # -----Initialize tracking variables-----
    start_epoch = 0
    best_loss = float('inf')

    # -----Default calibration state (may be overwritten by checkpoint restore below)-----
    pde_normalization_factor = None  # Computed once at epoch 0 if not loaded
    gradnorm_calibrated = False  # Auto-calibration fires at GradNorm transition if not loaded
    loss_calibration_factors = {}  # Persisted across checkpoints

    # -----Initialize optimizer-----
    thermodynamic_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'z_Delta_H_' in name or 'y_E_' in name or 'x_A_' in name:
            thermodynamic_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_schedule[0])
    
    lbfgs_switch_fraction = curriculum_config.get('lbfgs_switch_fraction', 0.8) if curriculum_config else 0.8
    lbfgs_switch_epoch = start_epoch + int(lbfgs_switch_fraction * n_epochs)
    lbfgs_optimizer = None

    mse_loss = nn.MSELoss()
    parameters_history = []
    loss_history = []

    # -----For tracking losses and weight adjustments-----
    adaptive_weight_log = []

    # -----Store previous weights for stability checks-----
    previous_weights = weights.copy()
    stable_weights_backup = weights.copy()

    # -----Resume from checkpoint if provided-----
    if resume_from is not None and os.path.exists(resume_from):
        checkpoint_data = load_checkpoint(
            resume_from=resume_from,
            model=model,
            optimizer=optimizer,
            weights=weights,
            loss_history_gradnorm=loss_history_gradnorm,
            grad_history=grad_history,
            weight_history=weight_history,
            lr_schedule=lr_schedule,
            task=task
        )
        start_epoch = checkpoint_data['start_epoch']
        best_loss = checkpoint_data['best_loss']
        parameters_history = checkpoint_data['parameters_history']
        loss_history = checkpoint_data['loss_history']
        weights = checkpoint_data['weights']
        if task == 'extend_finetune':
            prev_best = best_loss
            best_loss = float('inf')
            logger.info(
                f"extend_finetune: resetting best_loss from {prev_best:.4e} to +inf — "
                f"checkpoint_best.pt will track best fine-tune epoch instead of training's."
            )
        pde_normalization_factor = checkpoint_data.get('pde_normalization_factor', None)
        loss_calibration_factors = checkpoint_data.get('loss_calibration_factors', {})
        gradnorm_calibrated = checkpoint_data.get('gradnorm_calibrated', False)

        if task == 'extend_finetune':
            gradnorm_calibrated = False
            pde_normalization_factor = None
            loss_calibration_factors = {}
            fresh_initial_weights = parse_initial_weights(curriculum_config, task, run_mode)
            weights.clear()
            weights.update(fresh_initial_weights)
            for k in loss_history_gradnorm:
                loss_history_gradnorm[k] = []
            previous_weights = weights.copy()
            stable_weights_backup = weights.copy()
            logger.info(
                "extend_finetune: cleared training-era calibration state and reset weights "
                f"to config initial_weights {weights} — calibration will re-fire on first "
                f"weight-update epoch (epoch {start_epoch})."
            )
    
    def do_save_checkpoint(epoch, loss, curr_weights=None, is_best=False, save_periodic=False, save_final=False):
        """Wrapper to call the imported save_checkpoint with closure variables"""
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            loss=loss,
            best_loss=best_loss,
            parameters_history=parameters_history,
            loss_history=loss_history,
            loss_history_gradnorm=loss_history_gradnorm,
            curr_weights=curr_weights,
            is_best=is_best,
            save_periodic=save_periodic,
            save_final=save_final,
            pde_normalization_factor=pde_normalization_factor,
            loss_calibration_factors=loss_calibration_factors,
            gradnorm_calibrated=gradnorm_calibrated,
        )
    
    # -----Preprocess initial and final inputs-----
    f_initial = input_features.clone()
    f_final = output_targets.clone()
    x_initial, f_phase_initial, env_dict = model.preprocess_initial_inputs(f=input_features)
    x_final, f_phase_final, env_dict_final = model.preprocess_initial_inputs(f=output_targets)

    # Save environment dictionaries as csv
    env_dict_cpu = {k: v.cpu().detach().numpy() for k, v in env_dict.items()}
    env_dict_df = pd.DataFrame(env_dict_cpu)
    env_dict_df.to_csv(os.path.join(model_folder, 'env_dict.csv'), index=False)

    env_dict_final_cpu = {k: v.cpu().detach().numpy() for k, v in env_dict_final.items()}
    env_dict_final_df = pd.DataFrame(env_dict_final_cpu)
    env_dict_final_df.to_csv(os.path.join(model_folder, 'env_dict_final.csv'), index=False)

    if debug:
        # Log several rows of x_initial for inspection
        num_rows_to_log = min(5, x_initial.shape[0])
        logger.debug(f"Logging first {num_rows_to_log} rows of x_initial:")
        for i in range(num_rows_to_log):
            logger.debug(f"x_initial[{i}]: {x_initial[i].cpu().detach().numpy()}")
        logger.debug(f"Logging first {num_rows_to_log} rows of x_final:")
        for i in range(num_rows_to_log):
            logger.debug(f"x_final[{i}]: {x_final[i].cpu().detach().numpy()}")
        logger.debug(f"Logging first {num_rows_to_log} rows of f_phase_initial:")
        for i in range(num_rows_to_log):
            logger.debug(f"f_phase_initial[{i}]: {f_phase_initial[i].cpu().detach().numpy()}")
        logger.debug(f"Logging first {num_rows_to_log} rows of f_phase_final:")
        for i in range(num_rows_to_log):
            logger.debug(f"f_phase_final[{i}]: {f_phase_final[i].cpu().detach().numpy()}")

    # -----Sanity check for FA flow rate decrease-----
    fa_initial = f_initial[:, 6]  # Get FA flow rate from input
    fa_final = f_final[:, 5]      # Get FA flow rate from output

    # Calculate decrease ratio and find valid samples
    fa_decrease = fa_initial > fa_final

    # Log statistics about the filtering
    total_samples = len(fa_initial)
    valid_samples = fa_decrease.sum().item()
    removed_samples = total_samples - valid_samples
    logger.info(f"FA flow rate check - Total samples: {total_samples}, Valid samples: {valid_samples}")
    if removed_samples > 0:
        logger.warning(f"Removed {removed_samples} samples where FA didn't decrease")

    # Create detailed log of removed samples
    for i in range(total_samples):
        if not fa_decrease[i]:
            logger.warning(f"Sample {i}: FA increased from {fa_initial[i]:.6f} to {fa_final[i]:.6f} "
                        f"(change: {(fa_final[i] - fa_initial[i]):.6f})")

    # Filter out invalid samples
    valid_mask = fa_decrease
    x_initial = x_initial[valid_mask]
    x_final = x_final[valid_mask]
    f_initial = f_initial[valid_mask]
    f_final = f_final[valid_mask]

    # Update environment dictionaries
    env_dict = {k: v[valid_mask] for k, v in env_dict.items()}
    env_dict_final = {k: v[valid_mask] for k, v in env_dict_final.items()}

    if len(f_initial) == 0:
        raise ValueError("No valid samples remaining after FA flow rate check!")

    # -----Make sure f_initial, f_final, env_dict, env_dict_final have the right dtype-----
    f_initial = f_initial.to(device, dtype=torch.float64)
    env_dict = {k: v.to(device, dtype=torch.float64) for k, v in env_dict.items()}
    f_final = f_final.to(device, dtype=torch.float64)
    env_dict_final = {k: v.to(device, dtype=torch.float64) for k, v in env_dict_final.items()}

    # -----Pre-training initialization-----
    if start_epoch == 0 and task == 'train':  # Only for fresh training, not when resuming or predict
        logger.info(f"Pre-training initialization: Focusing on initial condition matching for {pre_training_epochs} epochs")
        ic_params = []
        for name, param in model.named_parameters():
            if not is_kinetic_param(name):
                ic_params.append(param)

        pre_optimizer = torch.optim.Adam(ic_params, lr=5e-3)

        output_scale_for_norm = model.output_scales.to(device)

        for pre_epoch in range(pre_training_epochs):
            if enable_data_shuffling:
                pre_shuffle_indices = torch.randperm(f_initial.shape[0], device=device)
                pre_f_initial = f_initial[pre_shuffle_indices]
            else:
                pre_f_initial = f_initial

            pre_optimizer.zero_grad()

            with torch.enable_grad():
                f_pred, _ = model(f_initial=pre_f_initial)

                ic_loss, ic_loss_raw = compute_initial_condition_loss(
                    f_pred, pre_f_initial, output_scale_for_norm, mse_loss
                )

                ic_loss.backward(retain_graph=True)
                pre_optimizer.step()

            if pre_epoch % 10 == 0:
                logger.info(f"Pre-training epoch {pre_epoch}/{pre_training_epochs}: Normalized IC loss: {ic_loss.item():.4e}, Raw IC loss: {ic_loss_raw.item():.4e}")

        logger.info(f"Pre-training complete ({pre_training_epochs} epochs). Starting main training loop.")

    # -----Log scaling factors-----
    logger.info(f"=== Training Configuration ===")
    logger.info(f"Output Scales: {[f'{s:.2e}' for s in model.output_scales.tolist()]} (per-species)")
    logger.info(f"Adaptive Start Epoch: {curriculum_end_epoch}")
    logger.info(f"==============================")

    # -----PDE auto-calibration setup-----
    pde_auto_calibrate = curriculum_config.get('pde_auto_calibrate', False) if curriculum_config else False
    if pde_auto_calibrate:
        logger.info("PDE auto-calibration ENABLED: one-time normalization at epoch 0")
        logger.info("  PDE weights in stages are regular weights (not contribution targets)")
        if 'pde' not in weight_history:
            weight_history['pde'] = []

    # -----Baseline snapshot for extend_finetune-----
    baseline_data_raw = None
    baseline_pde_raw = None
    baseline_flow_raw = None
    baseline_ic_raw = None
    if task == 'extend_finetune' and extend_target_time is not None:
        logger.info("=" * 72)
        logger.info("BASELINE SNAPSHOT — pretrained model evaluated over extended horizon")
        logger.info("=" * 72)
        with torch.enable_grad():
            _bl_shuf = torch.arange(f_initial.shape[0], device=device)
            baseline = compute_all_losses(
                model=model, f_initial_epoch=f_initial, f_final=f_final,
                compute_pde_loss_fn=compute_pde_loss_mn, mse_loss=mse_loss, device=device,
                task=task, run_mode=run_mode,
                include_initial_loss=include_initial_loss,
                enable_data_shuffling=False, shuffle_indices=_bl_shuf,
                pde_collocation_points=None, compute_phase=False,
                compute_carbon_balance=('carbon_balance' in weights),
                min_rate_threshold=min_rate_threshold,
                extend_target_time=extend_target_time,
            )
        max_orig_t = f_initial[:, 0].max().item()
        baseline_data_raw = baseline['loss_data_raw'].item()
        baseline_pde_raw = baseline['loss_pde_col'].item()
        baseline_flow_raw = baseline['flowrate_conservation_raw'].item()
        baseline_ic_raw = baseline['loss_initial_condition_raw'].item()
        logger.info(f"  Max original t_x: {max_orig_t:.4f}, extend_target_time: {extend_target_time:.4f}")
        logger.info(f"  Data loss  (raw): {baseline_data_raw:.4e}  (norm): {baseline['loss_data'].item():.4e}")
        logger.info(f"  PDE        (raw): {baseline_pde_raw:.4e}  (pre log1p/calibration)")
        logger.info(f"  IC         (raw): {baseline_ic_raw:.4e}")
        logger.info(f"  Flowrate   (raw): {baseline_flow_raw:.4e}")
        f_pred_bl = baseline['f_pred']
        f_pred_at_orig_bl = baseline['f_pred_at_target']
        n_show = min(3, f_pred_bl.shape[0])
        for s in range(n_show):
            orig_t = f_initial[s, 0].item()
            at_orig = f_pred_at_orig_bl[s, :4].detach().cpu().numpy()
            at_ext = f_pred_bl[s, -1, :4].detach().cpu().numpy()
            logger.info(
                f"  Sample {s}: @ orig t={orig_t:.3f} "
                f"FA={at_orig[0]:.3e},FB={at_orig[1]:.3e},FC={at_orig[2]:.3e},FD={at_orig[3]:.3e} | "
                f"@ ext t={extend_target_time:.3f} "
                f"FA={at_ext[0]:.3e},FB={at_ext[1]:.3e},FC={at_ext[2]:.3e},FD={at_ext[3]:.3e}"
            )
        logger.info("=" * 72)
        del baseline

    # -----Main training loop-----
    pde_concentration = 0.0  # Initialize for PDE collocation tracking
    pde_collocation_points = None  # Continuous time values for PDE evaluation
    for epoch in range(start_epoch, start_epoch + n_epochs):
        # -----Shuffle data for each epoch-----
        if enable_data_shuffling:
            shuffle_indices = torch.randperm(f_initial.shape[0], device=device)
            f_initial_epoch = f_initial[shuffle_indices]
            f_final_epoch = f_final[shuffle_indices]
            env_dict_epoch = {k: v[shuffle_indices] for k, v in env_dict.items()}
            env_dict_final_epoch = {k: v[shuffle_indices] for k, v in env_dict_final.items()}
            
            if debug:
                logger.debug(f"Epoch {epoch}: Using shuffled data with {len(f_initial_epoch)} samples")
        else:
            f_initial_epoch = f_initial
            f_final_epoch = f_final
            env_dict_epoch = env_dict
            env_dict_final_epoch = env_dict_final
        
        # -----PDE Collocation: Sample continuous time points for PDE evaluation-----
        if time_curriculum_enabled:
            prev_concentration = pde_concentration
            pde_concentration = get_pde_concentration(epoch, pde_concentration_schedule)

            # Log milestone when concentration changes
            if pde_concentration != prev_concentration:
                logger.info(f"=" * 70)
                logger.info(f"PDE COLLOCATION SCHEDULE CHANGE at epoch {epoch}")
                logger.info(f"  Concentration: {prev_concentration:.2f} -> {pde_concentration:.2f}")
                logger.info(f"  n_pde_points: {n_pde_points}")
                logger.info(f"=" * 70)

            # Sample continuous collocation time points in [0, t_max].
            # For extend_finetune, PDE residuals must cover the extended horizon
            if task == 'extend_finetune' and extend_target_time is not None:
                max_target_time = extend_target_time
            else:
                max_target_time = f_initial_epoch[:, 0].max().item()
            pde_collocation_points = sample_pde_collocation_points(
                max_target_time, n_pde_points, pde_concentration, device
            )
        else:
            pde_concentration = 0.0
            pde_collocation_points = None

        # -----For 'predict' task, train all parameters including dynamics-----
        if task == 'predict':
            for name, param in model.named_parameters():
                param.requires_grad = True

            if epoch == 0 or epoch % logging_freq == 0:  # Log only occasionally
                logger.info(f"Predict mode: Training all parameters including dynamics")
        elif task == 'extend_finetune':
            if epoch == start_epoch or epoch % logging_freq == 0:
                kinetic_trainable = any(
                    p.requires_grad for n, p in model.named_parameters()
                    if is_kinetic_param(n)
                )
                if kinetic_trainable:
                    logger.info("extend_finetune: kinetic params UNFROZEN — full NN + kinetics refined")
                else:
                    logger.info("extend_finetune: kinetic params frozen, only NN refined")
        elif task == 'train':
            phase_transition_epoch = curriculum_config.get('parameter_freezing', {}).get('phase_transition_epoch', 3000) if curriculum_config else 3000
            frozen_parameters = curriculum_config.get('parameter_freezing', {}).get('frozen_parameters', None) if curriculum_config else None
            toggle_parameter_freezing(model, epoch, phase_transition_epoch=phase_transition_epoch, frozen_parameters=frozen_parameters)

        # -----Determine weight update strategy with gradual transition-----
        transition_buffer = min(200, curriculum_end_epoch // 2)  # Adaptive transition period
        transition_start = max(0, curriculum_end_epoch - transition_buffer)
        
        # Log transition milestones
        if epoch == transition_start and transition_start > 0:
            logger.warning(f"🚀 ENTERING TRANSITION PHASE at epoch {epoch} (until epoch {curriculum_end_epoch})")
            logger.warning(f"Current weights: {weights}")
        elif epoch == curriculum_end_epoch:
            logger.warning(f"🎯 STARTING FULL ADAPTIVE PHASE at epoch {epoch}")
            logger.warning(f"Loss history lengths: {[len(loss_history_gradnorm[k]) for k in loss_history_gradnorm.keys()]}")
            logger.warning(f"Starting weights: {weights}")
        
        if epoch < transition_start and curriculum_stages:
            current_stage = max([stage for stage in curriculum_stages.keys() if stage <= epoch])
            target_weights = curriculum_stages[current_stage]

            next_stages = [stage for stage in curriculum_stages.keys() if stage > current_stage]
            if next_stages:
                next_stage = min(next_stages)
                next_weights = curriculum_stages[next_stage]
                alpha = (epoch - current_stage) / (next_stage - current_stage)
                for k in weights.keys():
                    if k in target_weights and k in next_weights:
                        weights[k] = (1 - alpha) * target_weights[k] + alpha * next_weights[k]
            else:
                for k in weights.keys():
                    if k in target_weights:
                        weights[k] = target_weights[k]

        elif epoch < curriculum_end_epoch and curriculum_stages:
            current_stage = max([stage for stage in curriculum_stages.keys() if stage <= epoch])
            target_weights = curriculum_stages[current_stage]

            for k in weights.keys():
                if k in target_weights:
                    weights[k] = target_weights[k]

            if epoch % 10 == 0:
                with torch.enable_grad():
                    loss_results = compute_all_losses(
                        model=model, f_initial_epoch=f_initial_epoch, f_final=f_final,
                        compute_pde_loss_fn=compute_pde_loss_mn, mse_loss=mse_loss, device=device,
                        task=task, run_mode=run_mode, include_initial_loss=include_initial_loss,
                        enable_data_shuffling=enable_data_shuffling, shuffle_indices=shuffle_indices,
                        pde_collocation_points=pde_collocation_points,
                        compute_phase=False,
                        compute_carbon_balance=('carbon_balance' in weights),
                        min_rate_threshold=min_rate_threshold,
                        extend_target_time=extend_target_time,
                    )

                loss_terms = build_loss_terms_dict(loss_results, run_mode, False, epoch, phase_intro_epoch)
                
                for k in weights.keys():
                    if k in gradnorm_excluded_tasks:
                        continue
                    if k in loss_terms and k in loss_history_gradnorm:
                        if torch.is_tensor(loss_terms[k]):
                            loss_history_gradnorm[k].append(loss_terms[k].item())
                        else:
                            loss_history_gradnorm[k].append(loss_terms[k])

                logger.info(f"Transition Phase - Building loss history at epoch {epoch}")
                
        # -----Full Adaptive weight updates-----
        elif epoch % weight_update_freq == 0:
            if not gradnorm_calibrated:
                with torch.enable_grad():
                    calib_loss_results = compute_all_losses(
                        model=model, f_initial_epoch=f_initial_epoch, f_final=f_final,
                        compute_pde_loss_fn=compute_pde_loss_mn, mse_loss=mse_loss, device=device,
                        task=task, run_mode=run_mode, include_initial_loss=include_initial_loss,
                        enable_data_shuffling=enable_data_shuffling, shuffle_indices=shuffle_indices,
                        pde_collocation_points=pde_collocation_points,
                        compute_phase=False,
                        compute_carbon_balance=('carbon_balance' in weights),
                        min_rate_threshold=min_rate_threshold,
                        extend_target_time=extend_target_time,
                    )

                # Apply PDE log normalization for calibration.
                calib_pde_raw = calib_loss_results['loss_pde_col']
                if pde_auto_calibrate and pde_normalization_factor is None:
                    import math as _math
                    pde_raw_0 = calib_pde_raw.item()
                    data_norm_0 = calib_loss_results['loss_data'].item()
                    pde_log_0 = _math.log1p(pde_raw_0)
                    if pde_log_0 > 1e-30 and data_norm_0 > 1e-30:
                        pde_normalization_factor = data_norm_0 / pde_log_0
                    else:
                        pde_normalization_factor = 1.0
                    # Cap factor only during ab-initio training
                    if task != 'extend_finetune':
                        PDE_FACTOR_CAP = 3.5
                        if pde_normalization_factor > PDE_FACTOR_CAP:
                            logger.info(
                                f"PDE factor {pde_normalization_factor:.4e} exceeds cap "
                                f"{PDE_FACTOR_CAP}, capping."
                            )
                            pde_normalization_factor = PDE_FACTOR_CAP
                    logger.info(
                        f"PDE ONE-TIME CALIBRATION (during GradNorm calibration, log-scale): "
                        f"pde_raw_0={pde_raw_0:.4e}, pde_log_0={pde_log_0:.4e}, "
                        f"data_norm_0={data_norm_0:.4e}, factor={pde_normalization_factor:.4e}"
                    )
                if pde_auto_calibrate and pde_normalization_factor is not None:
                    calib_pde = torch.log1p(calib_pde_raw) * pde_normalization_factor
                else:
                    calib_pde = calib_pde_raw

                # Build calibration mapping: task → current loss value
                loss_scale_map = {
                    'data': calib_loss_results['loss_data'].item(),
                    'pde': calib_pde.item(),
                    'flowrate': calib_loss_results['flowrate_conservation'].item(),
                    'initial': calib_loss_results['loss_initial_condition'].item(),
                }
                if 'carbon_balance' in weights:
                    loss_scale_map['carbon_balance'] = calib_loss_results['carbon_balance_loss'].item()

                for task_name, loss_val in loss_scale_map.items():
                    if task_name in weights:
                        if loss_val > 1e-10:
                            weights[task_name] = 1.0 / loss_val
                        else:
                            weights[task_name] = 1.0  # Fallback for near-zero losses
                        loss_calibration_factors[task_name] = weights[task_name]

                wide_floor_tasks = ('data', 'initial', 'flowrate', 'carbon_balance')
                for task_name in loss_scale_map:
                    if task_name in weights:
                        floor_mult = data_floor_mult if task_name in wide_floor_tasks else 0.1
                        weight_caps[task_name] = [weights[task_name] * floor_mult, weights[task_name] * 10.0]

                excluded_scale_map = {}
                for excluded_task in gradnorm_excluded_tasks:
                    if excluded_task in weights:
                        loss_key_map = {
                            'monotonicity': 'monotonicity_loss',
                            'rate_decrease': 'rate_decrease_loss',
                            'min_dynamics': 'min_dynamics_loss',
                        }
                        loss_key = loss_key_map.get(excluded_task)
                        if loss_key and loss_key in calib_loss_results:
                            loss_val = calib_loss_results[loss_key].item()
                            excluded_scale_map[excluded_task] = loss_val
                            if loss_val > 1e-10:
                                loss_calibration_factors[excluded_task] = 1.0 / loss_val
                            else:
                                loss_calibration_factors[excluded_task] = 1.0
                            stage_value = interpolate_curriculum_weight(
                                excluded_task, epoch, curriculum_stages
                            ) if curriculum_stages else weights[excluded_task]
                            if not stage_value:
                                stage_value = weights[excluded_task]
                            weights[excluded_task] = stage_value * loss_calibration_factors[excluded_task]
                            if excluded_task in weight_caps:
                                cap_min, cap_max = weight_caps[excluded_task]
                                weight_caps[excluded_task] = [
                                    cap_min * loss_calibration_factors[excluded_task],
                                    cap_max * loss_calibration_factors[excluded_task],
                                ]

                gradnorm_calibrated = True
                logger.info(f"GRADNORM AUTO-CALIBRATION at epoch {epoch}:")
                logger.info(f"  Loss scales (GradNorm): {loss_scale_map}")
                logger.info(f"  Calibrated weights (GradNorm): { {k: weights[k] for k in loss_scale_map if k in weights} }")
                logger.info(f"  Weight caps (GradNorm): { {k: weight_caps[k] for k in loss_scale_map if k in weight_caps} }")
                if excluded_scale_map:
                    logger.info(f"  Loss scales (excluded): {excluded_scale_map}")
                    logger.info(f"  Calibration factors (excluded): { {k: loss_calibration_factors[k] for k in excluded_scale_map if k in loss_calibration_factors} }")
                    logger.info(f"  Calibrated weights (excluded): { {k: weights[k] for k in excluded_scale_map if k in weights} }")
                    logger.info(f"  Weight caps (excluded): { {k: weight_caps[k] for k in excluded_scale_map if k in weight_caps} }")

            # Forward pass to compute all losses separately for GradNorm
            with torch.enable_grad():
                loss_results = compute_all_losses(
                    model=model, f_initial_epoch=f_initial_epoch, f_final=f_final,
                    compute_pde_loss_fn=compute_pde_loss_mn, mse_loss=mse_loss, device=device,
                    task=task, run_mode=run_mode, include_initial_loss=include_initial_loss,
                    enable_data_shuffling=enable_data_shuffling, shuffle_indices=shuffle_indices,
                    pde_collocation_points=pde_collocation_points,
                    compute_phase=(run_mode == 'pinn+phase'),
                    compute_carbon_balance=('carbon_balance' in weights),
                    min_rate_threshold=min_rate_threshold,
                    extend_target_time=extend_target_time,
                )

            # Build loss terms dictionary for GradNorm (phase excluded from training)
            loss_terms = build_loss_terms_dict(loss_results, run_mode, False, epoch, phase_intro_epoch)
            
            # Store current parameter gradients to restore later
            param_grads = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_grads[name] = param.grad.detach().clone()
            
            # Compute gradient magnitudes for each loss term (following paper methodology)
            grad_magnitudes = {}
            for loss_name in weights.keys():
                if task == 'predict' and loss_name == 'data':
                    continue
                if loss_name not in loss_terms:
                    continue  # Skip phase loss if not yet introduced
                if loss_name in gradnorm_excluded_tasks:
                    continue  # Uses curriculum interpolation, not GradNorm

                # Zero gradients before computing individual gradients
                optimizer.zero_grad()
                
                # Compute gradients for this loss term only
                if torch.is_tensor(loss_terms[loss_name]) and loss_terms[loss_name].requires_grad:
                    loss_terms[loss_name].backward(retain_graph=True)
                
                # Calculate gradient magnitude across all parameters
                grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                
                grad_magnitudes[loss_name] = grad_norm ** 0.5
                grad_history[loss_name].append(grad_norm ** 0.5)
            
            if debug:
                logger.debug(f"Gradient magnitudes: {grad_magnitudes}")
                logger.debug(f"Any positive gradients: {any(g > 0 for g in grad_magnitudes.values())}")
                logger.debug(f"All positive gradients: {all(g > 0 for g in grad_magnitudes.values())}")
            
            # Restore original gradients
            optimizer.zero_grad()
            for name, param in model.named_parameters():
                if name in param_grads:
                    param.grad = param_grads[name]
            
            # -----Advanced GradNorm-based adaptive weight updates-----
            # Calculate individual loss terms for GradNorm
            for k in weights.keys():
                if k in gradnorm_excluded_tasks:
                    continue  # Excluded tasks use curriculum interpolation, not GradNorm
                if k in loss_terms and k in loss_history_gradnorm:
                    if torch.is_tensor(loss_terms[k]):
                        loss_history_gradnorm[k].append(loss_terms[k].item())
                    else:
                        loss_history_gradnorm[k].append(loss_terms[k])

            # Update weights using GradNorm (more sophisticated than basic momentum)
            if all(g > 0 for g in grad_magnitudes.values()):
                # Apply more conservative GradNorm parameters for early adaptive phase
                epochs_since_adaptive_start = epoch - curriculum_end_epoch
                if epochs_since_adaptive_start < 100:
                    conservative_alpha = gradnorm_alpha * 0.3
                    conservative_lr = gradnorm_lr * 0.5
                    logger.info(f"Early adaptive phase - using conservative GradNorm: gradnorm_alpha={conservative_alpha:.4f}, gradnorm_lr={conservative_lr:.4f}")
                else:
                    conservative_alpha = gradnorm_alpha
                    conservative_lr = gradnorm_lr

                # Apply GradNorm algorithm (excluded tasks returned unchanged)
                new_weights = gradnorm_update_weights(
                    weights=weights,
                    grad_magnitudes=grad_magnitudes,
                    loss_terms=loss_terms,
                    target_rates=target_rates,
                    alpha=conservative_alpha,
                    learning_rate=conservative_lr,
                    loss_history=loss_history_gradnorm,
                    epoch=epoch,
                    excluded_tasks=gradnorm_excluded_tasks
                )

                # Apply weight constraints to GradNorm-managed tasks
                for k in new_weights.keys():
                    if k in gradnorm_excluded_tasks:
                        continue  # Excluded tasks handled below
                    if k in weight_caps:
                        # Soft clamping to prevent abrupt changes
                        min_val, max_val = weight_caps[k]
                        min_val, max_val = float(min_val), float(max_val)
                        old_weight = weights[k]

                        # Extra conservative during early adaptive phase
                        if epochs_since_adaptive_start < 100:
                            # Limit changes to 20% during transition period
                            max_change_factor = 1.2
                            min_change_factor = 0.8
                            new_weights[k] = np.clip(new_weights[k],
                                                   old_weight * min_change_factor,
                                                   old_weight * max_change_factor)

                        # Apply exponential smoothing for constraint enforcement
                        new_weight_val = float(new_weights[k])
                        if new_weight_val < min_val:
                            new_weights[k] = min_val + 0.1 * (old_weight - min_val)
                        elif new_weight_val > max_val:
                            new_weights[k] = max_val - 0.1 * (max_val - old_weight)

                    # Special handling for phase loss
                    if k == 'phase' and epoch < phase_intro_epoch:
                        new_weights[k] = 0.0  # Keep at zero until phase_intro_epoch

                if (max_pde_data_ratio and task != 'extend_finetune'
                        and 'data' in new_weights and 'pde' in new_weights):
                    min_data_from_ratio = new_weights['pde'] / max_pde_data_ratio
                    if new_weights['data'] < min_data_from_ratio:
                        new_weights['data'] = min_data_from_ratio

                # Set excluded tasks from curriculum interpolation
                for excluded_task in gradnorm_excluded_tasks:
                    if excluded_task in new_weights and curriculum_stages:
                        interp_weight = interpolate_curriculum_weight(
                            excluded_task, epoch, curriculum_stages
                        )
                        calib_factor = loss_calibration_factors.get(excluded_task, 1.0)
                        new_weights[excluded_task] = interp_weight * calib_factor

                if task == 'extend_finetune' and curriculum_config is not None:
                    _warmup = int(curriculum_config.get('gradnorm_warmup_epochs', 0) or 0)
                    if _warmup > 0:
                        _since_calib = epoch - curriculum_end_epoch
                        if _since_calib < _warmup:
                            if epoch % logging_freq == 0:
                                logger.info(
                                    f"GradNorm warmup (epoch {epoch}): freezing weights at "
                                    f"calibration for {_warmup - _since_calib} more epochs"
                                )
                            new_weights = weights.copy()

                # Update weights
                weights.update(new_weights)

                weight_explosion_check_needed = True

                stable_weights_backup = weights.copy()
                previous_weights = weights.copy()

                # Initialize explosion tracking variables (will be updated after loss computation)
                is_exploded = False
                explosion_msg = None

                # Log the advanced weight changes
                log_entry = {
                    'epoch': epoch,
                    'method': 'GradNorm',
                    'weights': weights.copy(),
                    'grad_magnitudes': grad_magnitudes.copy(),
                    'loss_terms': {k: v.item() if torch.is_tensor(v) else v for k, v in loss_terms.items()},
                    'target_rates': target_rates.copy(),
                    'alpha': gradnorm_alpha,
                    'learning_rate': gradnorm_lr,
                    'exploded': is_exploded,
                    'explosion_msg': explosion_msg,
                    'epochs_since_adaptive_start': epochs_since_adaptive_start,
                    'conservative_alpha': conservative_alpha,
                    'conservative_lr': conservative_lr
                }
                adaptive_weight_log.append(log_entry)

                # Store current weights for history
                for k in weights.keys():
                    weight_history[k].append(weights[k])

        # -----Update learning rate based on epoch, not loss-----
        current_lr = lr_schedule[max([e for e in lr_schedule.keys() if e <= epoch])]
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # -----Switch to LBFGS for final fine-tuning to reduce oscillations-----
        if task != 'extend_finetune' and epoch == lbfgs_switch_epoch and epoch < start_epoch + n_epochs - 100:  # Leave some epochs for LBFGS
            logger.info(f"Epoch {epoch}: Switching from Adam to LBFGS for final fine-tuning")
            lbfgs_params = curriculum_config.get('lbfgs_params', {}) if curriculum_config else {}
            lbfgs_optimizer = torch.optim.LBFGS(
                model.parameters(), 
                lr=lbfgs_params.get('lr', 1e-3),
                max_iter=lbfgs_params.get('max_iter', 20),
                history_size=lbfgs_params.get('history_size', 10),
                tolerance_grad=lbfgs_params.get('tolerance_grad', 1e-12),
                tolerance_change=lbfgs_params.get('tolerance_change', 1e-15)
            )
            optimizer = lbfgs_optimizer
            
            weight_update_freq = 50

        # -----Memory diagnostic: reset peak tracker at start of each epoch-----
        if debug and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # -----Start of training step-----
        optimizer.zero_grad()

        # -----Forward pass and loss computation-----
        with torch.enable_grad():
            loss_results = compute_all_losses(
                model=model, f_initial_epoch=f_initial_epoch, f_final=f_final,
                compute_pde_loss_fn=compute_pde_loss_mn, mse_loss=mse_loss, device=device,
                task=task, run_mode=run_mode, include_initial_loss=include_initial_loss,
                enable_data_shuffling=enable_data_shuffling, shuffle_indices=shuffle_indices,
                pde_collocation_points=pde_collocation_points,
                compute_phase=(run_mode == 'pinn+phase'),
                compute_carbon_balance=('carbon_balance' in weights),
                min_rate_threshold=min_rate_threshold,
                extend_target_time=extend_target_time,
            )

            # Debug logging
            if debug:
                f_pred = loss_results['f_pred']
                debug_tensor_version("f_pred", f_pred)
                logger.debug("\n=== Prediction Diversity Check ===")
                for sample_idx in range(min(3, f_pred.shape[0])):
                    logger.debug(f"\nSample {sample_idx} predictions:")
                    time_indices = [0, f_pred.shape[1]//2, -1]
                    for t_idx in time_indices:
                        logger.debug(f"Time point {t_idx}: {f_pred[sample_idx, t_idx].detach().cpu().numpy()}")
            
            # Sanitize losses (replace NaN/Inf with fallback values)
            loss_results = sanitize_losses(loss_results, device)

            ext_region_pde_raw = None
            if (task == 'extend_finetune' and extend_target_time is not None
                    and pde_collocation_points is not None):
                _max_orig_t_best = f_initial_epoch[:, 0].max().item()
                _ext_time_mask_best = pde_collocation_points > _max_orig_t_best
                if _ext_time_mask_best.any():
                    _n_batch_best = f_initial_epoch.shape[0]
                    _ext_full_mask = _ext_time_mask_best.repeat_interleave(_n_batch_best)
                    with torch.no_grad():
                        ext_region_pde_raw = sum(
                            (r.detach()[_ext_full_mask] ** 2).mean()
                            for r in loss_results['pde_residuals']
                        ).item()

            # Extract losses for easier reference
            f_pred = loss_results['f_pred']
            time_points = loss_results['time_points']
            loss_data = loss_results['loss_data']
            loss_data_raw = loss_results['loss_data_raw']
            loss_initial_condition = loss_results['loss_initial_condition']
            loss_initial_condition_raw = loss_results['loss_initial_condition_raw']
            loss_pde_col = loss_results['loss_pde_col']
            loss_phase_transformed = loss_results['loss_phase_transformed']
            loss_phase_raw = loss_results['loss_phase_raw']
            flowrate_conservation = loss_results['flowrate_conservation']
            flowrate_conservation_raw = loss_results['flowrate_conservation_raw']
            monotonicity_loss = loss_results['monotonicity_loss']
            monotonicity_raw = loss_results['monotonicity_raw']
            rate_decrease_loss = loss_results['rate_decrease_loss']
            rate_decrease_raw = loss_results['rate_decrease_raw']
            min_dynamics_loss = loss_results['min_dynamics_loss']
            min_dynamics_raw = loss_results['min_dynamics_raw']
            carbon_balance_loss = loss_results['carbon_balance_loss']
            carbon_balance_raw = loss_results['carbon_balance_raw']

            # PDE one-time calibration (log-scale): normalize log(1+PDE) to match data loss scale
            if pde_auto_calibrate and pde_normalization_factor is None:
                import math
                pde_raw_0 = loss_pde_col.item()
                data_norm_0 = loss_data.item()  # Already normalized by output_scale
                pde_log_0 = math.log1p(pde_raw_0)
                if pde_log_0 > 1e-30 and data_norm_0 > 1e-30:
                    pde_normalization_factor = data_norm_0 / pde_log_0
                else:
                    pde_normalization_factor = 1.0
                if task != 'extend_finetune':
                    PDE_FACTOR_CAP = 3.5
                    if pde_normalization_factor > PDE_FACTOR_CAP:
                        logger.info(f"PDE factor {pde_normalization_factor:.4e} exceeds cap {PDE_FACTOR_CAP}, capping.")
                        pde_normalization_factor = PDE_FACTOR_CAP
                logger.info(f"PDE ONE-TIME CALIBRATION (log-scale): pde_raw_0={pde_raw_0:.4e}, "
                            f"pde_log_0={pde_log_0:.4e}, data_norm_0={data_norm_0:.4e}, "
                            f"factor={pde_normalization_factor:.4e}")

            loss_pde_col_raw = loss_pde_col  # Keep raw for logging/history
            if pde_auto_calibrate and pde_normalization_factor is not None:
                loss_pde_col = torch.log1p(loss_pde_col) * pde_normalization_factor
                loss_results['loss_pde_col'] = loss_pde_col

            # Compute total loss (phase loss removed from training — PDE already uses flash-corrected concentrations)
            include_phase = False
            loss = compute_total_loss(
                loss_results=loss_results,
                weights=weights,
                task=task,
                run_mode=run_mode,
                include_initial_loss=include_initial_loss,
                include_phase_loss=include_phase,
                device=device
            )
        
            # Check for NaN in loss
            if torch.isnan(loss).any():
                raise RuntimeError("NaN detected in loss")
            
            # -----Check for weight explosion after loss computation-----
            if 'weight_explosion_check_needed' in locals() and weight_explosion_check_needed:
                # Adaptive loss threshold based on training phase
                epochs_since_adaptive_start = epoch - curriculum_end_epoch
                if epochs_since_adaptive_start < 50:
                    # Very early adaptive phase - allow higher losses
                    adaptive_loss_threshold = 1e6  
                    threshold_phase = "very early"
                elif epochs_since_adaptive_start < 200:
                    # Early adaptive phase - moderate threshold
                    adaptive_loss_threshold = 1e5
                    threshold_phase = "early"
                else:
                    # Mature adaptive phase - stricter threshold
                    adaptive_loss_threshold = 1e4
                    threshold_phase = "mature"
                
                is_exploded, explosion_msg = detect_weight_explosion(
                    current_weights=weights,
                    previous_weights=previous_weights,
                    loss_value=loss.item(),
                    threshold_factor=15.0,
                    loss_threshold=adaptive_loss_threshold
                )
                
                # Update the most recent log entry with explosion status
                if adaptive_weight_log:
                    adaptive_weight_log[-1]['exploded'] = is_exploded
                    adaptive_weight_log[-1]['explosion_msg'] = explosion_msg
                
                if is_exploded:
                    logger.warning(f"WEIGHT EXPLOSION DETECTED at epoch {epoch}: {explosion_msg}")
                    logger.warning("Rolling back to stable weights and reducing learning rate")
                    
                    # Rollback to stable weights
                    weights = stable_weights_backup.copy()
                    
                    # Emergency learning rate reduction (less aggressive)
                    emergency_lr = optimizer.param_groups[0]['lr'] * 0.5  # Less drastic reduction
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = emergency_lr
                    
                    logger.warning(f"Reduced learning rate to {emergency_lr:.2e} for stability")
                    
                    gradnorm_alpha = max(gradnorm_alpha * 0.8, 0.01)
                    gradnorm_lr = max(gradnorm_lr * 0.8, 0.0005)
                    
                    logger.warning(f"Reduced GradNorm parameters: gradnorm_alpha={gradnorm_alpha:.3f}, gradnorm_lr={gradnorm_lr:.4f}")
                
                weight_explosion_check_needed = False  # Reset flag
            
            # -----Backward pass-----
            if task == 'train':
                phase_transition_epoch = curriculum_config.get('parameter_freezing', {}).get('phase_transition_epoch', 300) if curriculum_config else 300
                frozen_parameters = curriculum_config.get('parameter_freezing', {}).get('frozen_parameters', None) if curriculum_config else None
                toggle_parameter_freezing(model, epoch, phase_transition_epoch=phase_transition_epoch, frozen_parameters=frozen_parameters)

            loss.backward(retain_graph=True)

            if debug:
                logger.debug("\n=== Gradient Check ===")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Calculate gradient statistics per sample if possible
                        if len(param.grad.shape) > 1 and param.grad.shape[0] == f_initial_epoch.shape[0]:
                            for sample_idx in range(min(3, param.grad.shape[0])):
                                logger.debug(f"Sample {sample_idx} gradient for {name}:")
                                logger.debug(f"Mean: {param.grad[sample_idx].mean().item():.4e}")
                                logger.debug(f"Std: {param.grad[sample_idx].std().item():.4e}")
                        else:
                            logger.debug(f"Parameter {name} gradient statistics:")
                            logger.debug(f"Mean: {param.grad.mean().item():.4e}")
                            logger.debug(f"Std: {param.grad.std().item():.4e}")
            
            # -----Check for NaN in gradients-----
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    logger.warning(f"NaN gradient detected in {name}")
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                raise RuntimeError("NaN detected in gradients")
            
            # -----Optimizer step with different handling for LBFGS vs Adam-----
            if isinstance(optimizer, torch.optim.LBFGS):
                # LBFGS requires a closure function
                def closure():
                    optimizer.zero_grad()

                    with torch.enable_grad():
                        f_pred_closure, time_points_closure = model(f_initial=f_initial_epoch)
                        x_transformed_closure, f_recalculated_closure, _ = model.get_phase_transformed_x(
                            f_pred=f_pred_closure,
                            time_points=time_points_closure
                            )
                        f_pred_at_target_closure, x_pred_at_target_closure = model.get_predictions(
                            f_pred=f_pred_closure,
                            x_transformed=x_transformed_closure,
                            f_initial=f_initial_epoch,
                            time_points=time_points_closure
                            )

                        # Recompute losses (no coverage weighting - all samples included)
                        if task == 'predict':
                            loss_data_closure = torch.tensor(0.0, device=device, dtype=torch.float64)
                        elif run_mode == 'pinn+phase':
                            target_data_closure = f_final[shuffle_indices][:, 5:9] if enable_data_shuffling else f_final[:, 5:9]
                            loss_data_closure, _ = compute_data_loss_with_coverage(
                                f_pred_at_target_closure[:, :4], target_data_closure,
                                output_scale=model.output_scales.to(device), coverage_weights=None
                            )
                        elif run_mode == 'pinn':
                            target_data_closure = f_final[shuffle_indices][:, 3:9] if enable_data_shuffling else f_final[:, 3:9]
                            loss_data_closure, _ = compute_data_loss_with_coverage(
                                x_pred_at_target_closure[:, 2:8], target_data_closure,
                                output_scale=None, coverage_weights=None
                            )

                        if include_initial_loss:
                            loss_initial_condition_closure, _ = compute_initial_condition_loss(
                                f_pred_closure, f_initial_epoch, model.output_scales.to(device), mse_loss
                            )
                        else:
                            loss_initial_condition_closure = torch.tensor(0.0, device=device, dtype=torch.float64)

                        pde_time_points_closure = pde_collocation_points if pde_collocation_points is not None else time_points_closure
                        r1_col_closure, r2_col_closure, r3_col_closure, r4_col_closure = compute_pde_loss_mn(
                            model,
                            f_initial=f_initial_epoch,
                            time_points=pde_time_points_closure,
                            )
                        loss_pde_col_closure = sum(torch.mean(r ** 2) for r in [r1_col_closure, r2_col_closure, r3_col_closure, r4_col_closure])

                        # Apply log-scale PDE normalization factor
                        if pde_auto_calibrate and pde_normalization_factor is not None:
                            loss_pde_col_closure = torch.log1p(loss_pde_col_closure) * pde_normalization_factor

                        # Phase and mass conservation losses (normalized for consistent gradients)
                        loss_phase_transformed_closure, _ = compute_phase_loss(
                            f_pred_closure, f_recalculated_closure, model.output_scale, mse_loss
                        )
                        flowrate_conservation_closure, _ = compute_flowrate_conservation_loss(
                            f_pred_closure, f_initial_epoch, model.output_scale, mse_loss
                        )

                        # Compute total loss (phase loss removed from training)
                        loss_components_closure = [
                            weights['pde'] * loss_pde_col_closure,
                            weights['flowrate'] * flowrate_conservation_closure,
                        ]
                        if task == 'train':
                            loss_components_closure.append(weights['data'] * loss_data_closure)
                        if include_initial_loss:
                            loss_components_closure.append(weights['initial'] * loss_initial_condition_closure)
                        loss_closure = sum(loss_components_closure)

                        loss_closure.backward(retain_graph=True)
                    return loss_closure
                
                optimizer.step(closure)
                
                # -----Re-compute losses after L-BFGS for accurate logging-----
                with torch.enable_grad():
                    f_pred_post, time_points_post = model(f_initial=f_initial_epoch)
                    x_transformed_post, f_recalculated_post, _ = model.get_phase_transformed_x(
                        f_pred=f_pred_post, time_points=time_points_post)
                    f_pred_at_target_post, x_pred_at_target_post = model.get_predictions(
                        f_pred=f_pred_post, x_transformed=x_transformed_post,
                        f_initial=f_initial_epoch, time_points=time_points_post)

                    # Recompute both normalized and raw losses
                    if run_mode == 'pinn+phase':
                        target_data_post = f_final[shuffle_indices][:, 5:9] if enable_data_shuffling else f_final[:, 5:9]
                        loss_data, loss_data_raw = compute_data_loss_with_coverage(
                            f_pred_at_target_post[:, :4], target_data_post,
                            output_scale=model.output_scales.to(device), coverage_weights=None)
                    else:
                        target_data_post = f_final[shuffle_indices][:, 3:9] if enable_data_shuffling else f_final[:, 3:9]
                        loss_data, loss_data_raw = compute_data_loss_with_coverage(
                            x_pred_at_target_post[:, 2:8], target_data_post,
                            output_scale=None, coverage_weights=None)

                    loss_initial_condition, loss_initial_condition_raw = compute_initial_condition_loss(
                        f_pred_post, f_initial_epoch, model.output_scales.to(device), mse_loss)

                    pde_time_points_post = pde_collocation_points if pde_collocation_points is not None else time_points_post
                    _want_cb_post = 'carbon_balance' in weights
                    pde_returns_post = compute_pde_loss_mn(
                        model, f_initial=f_initial_epoch, time_points=pde_time_points_post,
                        return_carbon_balance=_want_cb_post,
                    )
                    if _want_cb_post:
                        r1, r2, r3, r4, dFsum_dt_all_post = pde_returns_post
                        carbon_balance_loss = ((dFsum_dt_all_post / model.output_scale) ** 2).mean()
                        carbon_balance_raw = (dFsum_dt_all_post ** 2).mean()
                    else:
                        r1, r2, r3, r4 = pde_returns_post
                        carbon_balance_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
                        carbon_balance_raw = torch.tensor(0.0, device=device, dtype=torch.float64)
                    loss_pde_col = sum(torch.mean(r ** 2) for r in [r1, r2, r3, r4])

                    # Apply log-scale PDE normalization and track raw
                    loss_pde_col_raw = loss_pde_col
                    if pde_auto_calibrate and pde_normalization_factor is not None:
                        loss_pde_col = torch.log1p(loss_pde_col) * pde_normalization_factor

                    loss_phase_transformed, loss_phase_raw = compute_phase_loss(
                        f_pred_post, f_recalculated_post, model.output_scale, mse_loss)
                    flowrate_conservation, flowrate_conservation_raw = compute_flowrate_conservation_loss(
                        f_pred_post, f_initial_epoch, model.output_scale, mse_loss)

                    # Recompute monotonicity/rate_decrease/min_dynamics losses
                    (monotonicity_loss, rate_decrease_loss, min_dynamics_loss,
                     monotonicity_raw, rate_decrease_raw, min_dynamics_raw) = compute_monotonicity_loss_autograd(
                        model, f_initial_epoch, time_points_post, model.output_scale,
                        min_rate_threshold=min_rate_threshold)

                    # Recompute total loss using normalized losses (matching Adam path)
                    loss = weights['pde'] * loss_pde_col + weights['flowrate'] * flowrate_conservation
                    loss += weights['data'] * loss_data
                    if include_initial_loss:
                        loss += weights['initial'] * loss_initial_condition
                    loss += weights.get('monotonicity', 1.0) * monotonicity_loss
                    loss += weights.get('rate_decrease', 1.0) * rate_decrease_loss
                    if 'min_dynamics' in weights:
                        loss += weights['min_dynamics'] * min_dynamics_loss
                    if 'carbon_balance' in weights:
                        loss += weights['carbon_balance'] * carbon_balance_loss

                # Update f_pred and time_points for logging
                f_pred = f_pred_post
                time_points = time_points_post
            else:
                # Standard Adam optimization
                optimizer.step()

        # -----Store history-----
        parameters_history.append(model.get_parameters())
        current_lr = optimizer.param_groups[0]['lr']
        loss_history.append([
            # Raw losses (0-8)
            loss.item(),                          # 0: total weighted loss
            loss_data_raw.item(),                 # 1: data raw
            loss_pde_col_raw.item(),              # 2: pde raw
            loss_phase_raw.item(),                # 3: phase raw
            loss_initial_condition_raw.item(),     # 4: ic raw
            flowrate_conservation_raw.item(),      # 5: flowrate raw
            monotonicity_raw.item(),               # 6: monotonicity raw
            rate_decrease_raw.item(),              # 7: rate_decrease raw
            min_dynamics_raw.item(),               # 8: min_dynamics raw
            # Normalized losses (9-16)
            loss_data.item(),                      # 9: data normalized
            loss_pde_col.item(),                   # 10: pde log-normalized
            loss_phase_transformed.item(),         # 11: phase normalized
            loss_initial_condition.item(),          # 12: ic normalized
            flowrate_conservation.item(),           # 13: flowrate normalized
            monotonicity_loss.item(),               # 14: monotonicity normalized
            rate_decrease_loss.item(),              # 15: rate_decrease normalized
            min_dynamics_loss.item(),               # 16: min_dynamics normalized
            # Current weights (17-24)
            weights.get('data', 0.0),              # 17: weight_data
            weights.get('pde', 0.0),               # 18: weight_pde
            weights.get('initial', 0.0),           # 19: weight_initial
            weights.get('flowrate', 0.0),          # 20: weight_flowrate
            weights.get('phase', 0.0),             # 21: weight_phase
            weights.get('monotonicity', 0.0),      # 22: weight_monotonicity
            weights.get('rate_decrease', 0.0),     # 23: weight_rate_decrease
            weights.get('min_dynamics', 0.0),      # 24: weight_min_dynamics
            # Learning rate (25)
            current_lr,                            # 25: learning_rate
            # Carbon balance — appended at the end so older CSV readers keep
            # the same indices for the legacy 0-25 columns.
            carbon_balance_raw.item(),             # 26: carbon_balance raw
            carbon_balance_loss.item(),            # 27: carbon_balance normalized
            weights.get('carbon_balance', 0.0),    # 28: weight_carbon_balance
        ])

        # -----Memory diagnostic: log end-of-epoch alloc + peak + reserved-----
        if debug and torch.cuda.is_available():
            _alloc = torch.cuda.memory_allocated() / 1024**3
            _peak = torch.cuda.max_memory_allocated() / 1024**3
            _reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"[mem] epoch={epoch} alloc={_alloc:.3f} peak={_peak:.3f} "
                f"reserved={_reserved:.3f} GiB"
            )

        # -----Check if this is the best model so far-----
        if task == 'extend_finetune' and ext_region_pde_raw is not None:
            current_best_metric = ext_region_pde_raw
            best_metric_name = 'ext-region PDE raw'
        else:
            current_best_metric = loss.item()
            best_metric_name = 'total loss'
        is_best = current_best_metric < best_loss
        if is_best:
            best_loss = current_best_metric

        # -----Save checkpoint at specified frequency (guaranteed)-----
        should_save_periodic = ((epoch + 1) % checkpoint_freq == 0)
        should_save_final = (epoch == (start_epoch + n_epochs - 1))
        
        if should_save_periodic or should_save_final or is_best:
            do_save_checkpoint(
                epoch=epoch, 
                loss=loss.item(), 
                curr_weights=weights, 
                is_best=is_best,
                save_periodic=should_save_periodic,
                save_final=should_save_final
            )
            
            reasons = []
            if should_save_periodic:
                reasons.append(f"periodic (every {checkpoint_freq} epochs)")
            if should_save_final:
                reasons.append("final epoch")
            if is_best:
                reasons.append(f"best model ({best_metric_name}={current_best_metric:.4e})")

            logger.info(f"Checkpoint trigger at epoch {epoch}: {', '.join(reasons)}")

            # Release temp buffers held by torch.save's state_dict copy back to the CUDA caching allocator.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # -----Log phase loss introduction-----
        if epoch == phase_intro_epoch and run_mode == 'pinn+phase':
            logger.info(f"Epoch {epoch}: Introducing phase loss with initial weight {weights['phase']}")
        
        # -----Log training progress with GradNorm details-----
        if epoch % logging_freq == 0:
            weights_str = ", ".join([f"{k}: {v:.4e}" for k, v in weights.items() if k != 'initial' or include_initial_loss])
    
            log_components = [
                f'Epoch [{epoch}/{start_epoch + n_epochs}]',
                f'Total Loss: {loss.item():.4e}\n',
                f'PDE Loss (log-normalized): {loss_pde_col.item():.4e}',
                f'PDE Loss (raw): {loss_pde_col_raw.item():.4e}',
                f'PDE Loss (log1p): {torch.log1p(loss_pde_col_raw).item():.4e}',
                f'PDE Loss (weighted): {(weights["pde"] * loss_pde_col).item():.4e}',
                f'Flowrate Cons (normalized): {flowrate_conservation.item():.4e}',
                f'Flowrate Cons (raw): {flowrate_conservation_raw.item():.4e}',
                f'Flowrate Cons (weighted): {(weights["flowrate"] * flowrate_conservation).item():.4e}',
            ]
            if task in ('train', 'extend_finetune'):
                log_components.insert(2, f'Data Loss (normalized): {loss_data.item():.4e}')
                log_components.insert(3, f'Data Loss (raw): {loss_data_raw.item():.4e}')
                log_components.insert(4, f'Data Loss (weighted): {(weights["data"] * loss_data).item():.4e}')
            if run_mode == 'pinn+phase':
                log_components.append(f'Phase Loss (tracked, not in gradient): {loss_phase_transformed.item():.4e}')
                log_components.append(f'Phase Loss (raw): {loss_phase_raw.item():.4e}')

            # Only include initial condition loss in log if used
            if include_initial_loss:
                log_components.append(f'Initial Loss (normalized): {loss_initial_condition.item():.4e}')
                log_components.append(f'Initial Loss (raw): {loss_initial_condition_raw.item():.4e}')
                log_components.append(f'Initial Loss (weighted): {(weights["initial"] * loss_initial_condition).item():.4e}')

            # Monotonicity physics constraints
            log_components.append(f'Monotonicity Loss (normalized): {monotonicity_loss.item():.4e}')
            log_components.append(f'Monotonicity Loss (raw): {monotonicity_raw.item():.4e}')
            log_components.append(f'Monotonicity Loss (weighted): {(weights.get("monotonicity", 1.0) * monotonicity_loss).item():.4e}')
            log_components.append(f'Rate Decrease Loss (normalized): {rate_decrease_loss.item():.4e}')
            log_components.append(f'Rate Decrease Loss (raw): {rate_decrease_raw.item():.4e}')
            log_components.append(f'Rate Decrease Loss (weighted): {(weights.get("rate_decrease", 1.0) * rate_decrease_loss).item():.4e}')
            log_components.append(f'Min Dynamics Loss (normalized): {min_dynamics_loss.item():.4e}')
            log_components.append(f'Min Dynamics Loss (raw): {min_dynamics_raw.item():.4e}')
            if 'min_dynamics' in weights:
                log_components.append(f'Min Dynamics Loss (weighted): {(weights["min_dynamics"] * min_dynamics_loss).item():.4e}')

            if 'carbon_balance' in weights:
                log_components.append(f'Carbon Balance Loss (normalized): {carbon_balance_loss.item():.4e}')
                log_components.append(f'Carbon Balance Loss (raw): {carbon_balance_raw.item():.4e}')
                log_components.append(f'Carbon Balance Loss (weighted): {(weights["carbon_balance"] * carbon_balance_loss).item():.4e}')

            # Total loss breakdown — shows each component's weighted contribution
            total_val = loss.item()
            if total_val > 1e-30:
                breakdown_parts = []
                contrib_data = (weights.get('data', 0.0) * loss_data).item() if task in ('train', 'extend_finetune') else 0.0
                contrib_pde = (weights['pde'] * loss_pde_col).item()
                contrib_ic = (weights.get('initial', 0.0) * loss_initial_condition).item() if include_initial_loss else 0.0
                contrib_flow = (weights['flowrate'] * flowrate_conservation).item()
                contrib_mono = (weights.get('monotonicity', 0.0) * monotonicity_loss).item()
                contrib_rate = (weights.get('rate_decrease', 0.0) * rate_decrease_loss).item()
                contrib_mind = (weights.get('min_dynamics', 0.0) * min_dynamics_loss).item()
                contrib_carb = (weights.get('carbon_balance', 0.0) * carbon_balance_loss).item()
                for name, val in [('data', contrib_data), ('pde', contrib_pde), ('ic', contrib_ic),
                                  ('flow', contrib_flow), ('carb', contrib_carb),
                                  ('mono', contrib_mono),
                                  ('rate_dec', contrib_rate), ('min_dyn', contrib_mind)]:
                    pct = val / total_val * 100.0
                    if pct > 0.01:  # Only show non-trivial contributions
                        breakdown_parts.append(f'{name}={pct:.1f}%')
                log_components.append(f'\nLoss Breakdown: {", ".join(breakdown_parts)}')

            log_components.append(f'Weights: {weights_str}')
            log_components.append(f'LR: {optimizer.param_groups[0]["lr"]:.4e}')
            
            # Add PDE collocation logging
            if time_curriculum_enabled:
                n_pde_pts = len(pde_collocation_points) if pde_collocation_points is not None else len(time_points)
                n_total_pts = len(time_points)
                log_components.append(f'PDE Collocation: concentration={pde_concentration:.2f}, n_pde_pts={n_pde_pts}/{n_total_pts}')
            
            # Add GradNorm-specific logging
            if epoch >= curriculum_end_epoch:
                log_components.append(f'Method: GradNorm (gradnorm_alpha={gradnorm_alpha}, gradnorm_lr={gradnorm_lr})')
                
                # Log target rates
                target_rates_str = ", ".join([f"{k}: {v:.2f}" for k, v in target_rates.items() if k in weights])
                log_components.append(f'Target Rates: {target_rates_str}')
            
            logger.info(', '.join(log_components))
            # -----Log gradient norms for different parameter groups-----
            reaction_grad_norm = 0
            species_grad_norm = 0
            network_grad_norm = 0
            reaction_any_trainable = False
            species_any_trainable = False

            for name, param in model.named_parameters():
                is_reaction = name.startswith('x_A_')
                is_species = name.startswith('z_Delta_H_') or name.startswith('y_E_')
                if is_reaction and param.requires_grad:
                    reaction_any_trainable = True
                if is_species and param.requires_grad:
                    species_any_trainable = True
                if param.grad is not None:
                    if is_reaction:
                        reaction_grad_norm += param.grad.norm().item() ** 2
                    elif is_species:
                        species_grad_norm += param.grad.norm().item() ** 2
                    else:
                        network_grad_norm += param.grad.norm().item() ** 2

            reaction_grad_norm = reaction_grad_norm ** 0.5
            species_grad_norm = species_grad_norm ** 0.5
            network_grad_norm = network_grad_norm ** 0.5

            reaction_str = f'{reaction_grad_norm:.4e}' if reaction_any_trainable else 'frozen'
            species_str = f'{species_grad_norm:.4e}' if species_any_trainable else 'frozen'

            logger.info(f'Gradient Norms - Reaction: {reaction_str}, '
                        f'Species: {species_str}, Network: {network_grad_norm:.4e}')

            # -----Log concentration metrics-----
            with torch.no_grad():
                f_mean = torch.mean(torch.abs(f_pred[:, 1:, :])).item()
                logger.info(f"F_pred mean abs: {f_mean:.4e}")

                # Check time dynamics by logging concentrations at different time points
                for t_idx in [1, len(time_points)//2, len(time_points)-2]:
                    species_values = []
                    for s_idx in range(min(3, f_pred.shape[2])):  # First 3 species
                        species_values.append(f"F{s_idx}={f_pred[0, t_idx, s_idx].item():.4e}")

                    time_val = time_points[t_idx].item()
                    logger.info(f"t={time_val:.4e}: " + ", ".join(species_values))

            # -----extend_finetune extra diagnostics-----
            if task == 'extend_finetune' and extend_target_time is not None:
                if baseline_data_raw is not None:
                    d_data = loss_data_raw.item() - baseline_data_raw
                    d_pde = loss_pde_col_raw.item() - baseline_pde_raw
                    d_flow = flowrate_conservation_raw.item() - baseline_flow_raw
                    d_ic = loss_initial_condition_raw.item() - baseline_ic_raw
                    logger.info(
                        f"Delta vs baseline (raw): "
                        f"data={d_data:+.2e} (base {baseline_data_raw:.2e}) | "
                        f"pde={d_pde:+.2e} (base {baseline_pde_raw:.2e}) | "
                        f"ic={d_ic:+.2e} (base {baseline_ic_raw:.2e}) | "
                        f"flow={d_flow:+.2e} (base {baseline_flow_raw:.2e})"
                    )

                if pde_collocation_points is not None:
                    max_orig_t = f_initial_epoch[:, 0].max().item()
                    pts = pde_collocation_points
                    train_mask = pts <= max_orig_t
                    ext_mask = pts > max_orig_t
                    n_tr = int(train_mask.sum().item())
                    n_ex = int(ext_mask.sum().item())
                    with torch.enable_grad():
                        pde_tr_str = "N/A"
                        pde_ex_str = "N/A"
                        if n_tr > 0:
                            r_tr = compute_pde_loss_mn(model, f_initial=f_initial_epoch, time_points=pts[train_mask])
                            pde_tr = sum(torch.mean(r ** 2) for r in r_tr).item()
                            pde_tr_str = f"{pde_tr:.4e}"
                        if n_ex > 0:
                            r_ex = compute_pde_loss_mn(model, f_initial=f_initial_epoch, time_points=pts[ext_mask])
                            pde_ex = sum(torch.mean(r ** 2) for r in r_ex).item()
                            pde_ex_str = f"{pde_ex:.4e}"
                    logger.info(
                        f"PDE split (raw): "
                        f"train-region[0, {max_orig_t:.3f}]={pde_tr_str} ({n_tr} pts) | "
                        f"ext-region({max_orig_t:.3f}, {extend_target_time:.3f}]={pde_ex_str} ({n_ex} pts)"
                    )

                # (c) Predictions at extend_target_time for a few samples — watch the endpoint converge
                n_ext_show = min(2, f_pred.shape[0])
                for s in range(n_ext_show):
                    ext_pred = f_pred[s, -1, :4].detach().cpu().numpy()
                    logger.info(
                        f"Sample {s} @ t={extend_target_time:.3f}: "
                        f"FA={ext_pred[0]:.4e}, FB={ext_pred[1]:.4e}, "
                        f"FC={ext_pred[2]:.4e}, FD={ext_pred[3]:.4e}"
                    )

        # -----Clean up CUDA cache periodically-----
        if epoch % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----Plot weight history and GradNorm metrics after training-----
    plot_gradnorm_analysis(
        weight_history=weight_history,
        grad_history=grad_history,
        adaptive_weight_log=adaptive_weight_log,
        checkpoint_dir=checkpoint_dir
    )

    return parameters_history, loss_history
