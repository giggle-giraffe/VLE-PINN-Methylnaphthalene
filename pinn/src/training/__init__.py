# ------------------------------------------------------------------------------------------
# Training utilities module for PINN
# Copyright (c) 2026 Chen Zhang, Tao Li, Xiaoqian Dang
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

from .checkpoint import save_checkpoint, load_checkpoint
from .training_plots import plot_gradnorm_analysis
from .weight_adapt import gradnorm_update_weights
from .config import (
    parse_initial_weights,
    parse_curriculum_stages,
    derive_phase_intro_epoch,
    parse_gradnorm_params,
    parse_target_rates,
    parse_weight_caps,
    parse_time_curriculum,
    parse_learning_rate_schedule,
    parse_all_training_config,
    detect_weight_explosion,
    parse_gradnorm_excluded_tasks,
    interpolate_curriculum_weight,
)
from .epoch_losses import (
    compute_all_losses,
    compute_total_loss,
    sanitize_losses,
    build_loss_terms_dict,
)

__all__ = [
    'save_checkpoint',
    'load_checkpoint', 
    'plot_gradnorm_analysis',
    'gradnorm_update_weights',
    'parse_initial_weights',
    'parse_curriculum_stages',
    'derive_phase_intro_epoch',
    'parse_gradnorm_params',
    'parse_target_rates',
    'parse_weight_caps',
    'parse_time_curriculum',
    'parse_learning_rate_schedule',
    'parse_all_training_config',
    'detect_weight_explosion',
    'parse_gradnorm_excluded_tasks',
    'interpolate_curriculum_weight',
    'compute_all_losses',
    'compute_total_loss',
    'sanitize_losses',
    'build_loss_terms_dict',
]
