# ------------------------------------------------------------------------------------------
# Configuration parsing utilities for PINN training
# Copyright (c) 2026 Chen Zhang, Tao Li, Xiaoqian Dang
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

from loguru import logger


def parse_initial_weights(curriculum_config, task, run_mode):
    """
    Parse initial weights from curriculum config.
    
    Args:
        curriculum_config: Curriculum configuration dictionary or None
        task: 'train' or 'predict'
        run_mode: 'pinn' or 'pinn+phase'
    
    Returns:
        Dictionary of initial weights
    """
    if curriculum_config is not None and 'initial_weights' in curriculum_config:
        iw = curriculum_config['initial_weights']
        if task == 'predict' and 'predict' in iw:
            weights = iw['predict'].copy()
        elif task == 'extend_finetune' and 'extend_finetune' in iw:
            weights = iw['extend_finetune'].copy()
        elif task == 'extend_finetune' and 'train' in iw:
            weights = iw['train'].copy()
        elif task == 'train' and 'train' in iw:
            weights = iw['train'].copy()
        else:
            weights = _get_default_weights(task)
    else:
        weights = _get_default_weights(task)

    if run_mode == 'pinn+phase':
        weights['phase'] = 0.0

    if task == 'predict' and 'data' in weights:
        del weights['data']

    return weights


def _get_default_weights(task):
    """Get default weights based on task."""
    if task == 'predict':
        return {
            'pde': 1e0,
            'initial': 1e0,
            'flowrate': 1e0,
            'carbon_balance': 1e0,
            'monotonicity': 1e2,
            'rate_decrease': 1e2,
            'min_dynamics': 1e2,
        }
    else:
        return {
            'data': 1e0,
            'pde': 1e0,
            'initial': 1e0,
            'flowrate': 1e0,
            'carbon_balance': 1e0,
            'monotonicity': 1e2,
            'rate_decrease': 1e2,
            'min_dynamics': 1e2,
        }


def parse_curriculum_stages(curriculum_config, task):
    """
    Parse curriculum learning stages from config.
    
    Args:
        curriculum_config: Curriculum configuration dictionary or None
        task: 'train' or 'predict'
    
    Returns:
        Dictionary mapping epoch -> weight adjustments
    """
    if curriculum_config is not None and 'stages' in curriculum_config:
        curriculum_stages = curriculum_config['stages']
        # Convert string keys to integers and ensure values are floats
        curriculum_stages = {
            int(k): {
                loss_type: float(loss_value) if isinstance(loss_value, (str, int, float)) else loss_value
                for loss_type, loss_value in v.items()
            } 
            for k, v in curriculum_stages.items()
        }
    else:
        if task == 'predict':
            curriculum_stages = {}
        else:
            curriculum_stages = {
                # Phase 1: Initial condition focus with minimal physics
                0: {'data': 1e2, 'pde': 1e-15, 'initial': 1e3, 'flowrate': 1e1},
                # Phase 2: Gradual PDE introduction
                200: {'data': 1e2, 'pde': 2e-14, 'initial': 3e2, 'flowrate': 1e1},
                # Phase 3: PDE becomes significant
                700: {'data': 1e2, 'pde': 1e-12, 'initial': 8e1, 'flowrate': 1e1},
            }
    return curriculum_stages


def derive_phase_intro_epoch(curriculum_stages):
    """
    Derive the phase loss introduction epoch from curriculum stages.

    Scans the stages to find the first epoch where phase weight > 0.
    This eliminates the need for a separate phase_loss_introduction_epoch
    config parameter, ensuring a single source of truth.

    Args:
        curriculum_stages: Dictionary mapping epoch -> weight adjustments

    Returns:
        The first epoch where phase weight becomes non-zero, or None if
        phase is never introduced (i.e., always 0 or absent from stages).
    """
    for epoch in sorted(curriculum_stages.keys()):
        stage_weights = curriculum_stages[epoch]
        phase_weight = stage_weights.get('phase', 0.0)
        if phase_weight > 0:
            return epoch
    return None


def parse_gradnorm_params(curriculum_config):
    """
    Parse GradNorm hyperparameters from config.
    
    Args:
        curriculum_config: Curriculum configuration dictionary or None
    
    Returns:
        Tuple of (alpha, learning_rate, weight_update_freq)
    """
    gradnorm_alpha = curriculum_config.get('gradnorm_alpha', 0.01) if curriculum_config else 0.01
    gradnorm_lr = curriculum_config.get('gradnorm_lr', 0.001) if curriculum_config else 0.001
    weight_update_freq = curriculum_config.get('weight_update_frequency', 200) if curriculum_config else 200
    
    return gradnorm_alpha, gradnorm_lr, weight_update_freq


def parse_target_rates(curriculum_config):
    """
    Parse target relative training rates for GradNorm.
    
    Args:
        curriculum_config: Curriculum configuration dictionary or None
    
    Returns:
        Dictionary of target rates
    """
    if curriculum_config is not None and 'target_rates' in curriculum_config:
        return curriculum_config['target_rates'].copy()
    else:
        return {
            'data': 1.0,
            'pde': 1.0,
            'initial': 1.0,
            'flowrate': 1.0,
            'carbon_balance': 1.0,
            'phase': 1.0,
        }


def parse_weight_caps(curriculum_config, task, run_mode):
    """
    Parse weight caps to prevent extreme values.
    
    Args:
        curriculum_config: Curriculum configuration dictionary or None
        task: 'train' or 'predict'
        run_mode: 'pinn' or 'pinn+phase'
    
    Returns:
        Dictionary mapping weight name to (min, max) tuple
    """
    if curriculum_config is not None and 'weight_caps' in curriculum_config:
        wc = curriculum_config['weight_caps']
        if task == 'predict' and 'predict' in wc:
            weight_caps = {k: tuple(float(val) for val in v) for k, v in wc['predict'].items()}
        elif task == 'extend_finetune' and 'extend_finetune' in wc:
            weight_caps = {k: tuple(float(val) for val in v) for k, v in wc['extend_finetune'].items()}
        elif task == 'extend_finetune' and 'train' in wc:
            weight_caps = {k: tuple(float(val) for val in v) for k, v in wc['train'].items()}
        elif task == 'train' and 'train' in wc:
            weight_caps = {k: tuple(float(val) for val in v) for k, v in wc['train'].items()}
        else:
            weight_caps = _get_default_weight_caps(task)
    else:
        weight_caps = _get_default_weight_caps(task)

    # Add phase caps if needed
    if run_mode == 'pinn+phase':
        if (curriculum_config is not None and 'weight_caps' in curriculum_config and
            task in curriculum_config['weight_caps'] and
            'phase' in curriculum_config['weight_caps'][task]):
            pass  # Already loaded from config
        else:
            weight_caps['phase'] = (1e-3, 1e3)

    return weight_caps


def _get_default_weight_caps(task):
    """Get default weight caps based on task."""
    if task == 'predict':
        return {
            'pde': (1e-3, 1e3),
            'initial': (1e-3, 1e3),
            'flowrate': (1e-3, 1e3),
            'carbon_balance': (1e-3, 1e3),
            'monotonicity': (1e1, 1e4),
            'rate_decrease': (1e1, 1e4),
            'min_dynamics': (1e0, 1e2),
        }
    else:
        return {
            'data': (1e-3, 1e3),
            'pde': (1e-3, 1e3),
            'initial': (1e-3, 1e3),
            'flowrate': (1e-3, 1e3),
            'carbon_balance': (1e-3, 1e3),
            'monotonicity': (1e1, 1e4),
            'rate_decrease': (1e1, 1e4),
            'min_dynamics': (1e0, 1e2),
        }


def parse_time_curriculum(curriculum_config):
    """
    Parse time curriculum configuration for PDE collocation sampling.

    The new approach always uses the full trajectory for forward pass and data loss,
    but samples PDE collocation points non-uniformly (dense at early times, sparse
    at late times), gradually shifting to uniform sampling.

    Args:
        curriculum_config: Curriculum configuration dictionary or None

    Returns:
        Dictionary with time curriculum settings:
            - enabled: bool
            - concentration_schedule: dict mapping epoch -> concentration value
            - n_pde_points: int or None (None = use all points)
    """
    time_curriculum_config = curriculum_config.get('time_curriculum', {}) if curriculum_config else {}
    enabled = time_curriculum_config.get('enabled', False)

    if enabled:
        pde_collocation_config = time_curriculum_config.get('pde_collocation', {})
        n_pde_points = pde_collocation_config.get('n_pde_points', None)
        concentration_schedule = pde_collocation_config.get('concentration_schedule', {0: 0.0})
        # Convert string keys to int if needed
        concentration_schedule = {int(k): float(v) for k, v in concentration_schedule.items()}

        logger.info("Time curriculum ENABLED (PDE collocation sampling):")
        logger.info(f"  n_pde_points: {n_pde_points}")
        logger.info(f"  Concentration schedule: {concentration_schedule}")
    else:
        concentration_schedule = {0: 0.0}
        n_pde_points = None
        logger.info("Time curriculum DISABLED (uniform PDE sampling, full trajectory)")

    return {
        'enabled': enabled,
        'concentration_schedule': concentration_schedule,
        'n_pde_points': n_pde_points,
    }


def parse_learning_rate_schedule(curriculum_config):
    """
    Parse learning rate schedule with warmup.
    
    Args:
        curriculum_config: Curriculum configuration dictionary or None
    
    Returns:
        Tuple of (lr_schedule dict, warmup_epochs, base_lr, min_lr)
    """
    lr_schedule = {}
    
    if curriculum_config is not None and 'learning_rate' in curriculum_config:
        lr_config = curriculum_config['learning_rate']
        warmup_epochs = lr_config.get('warmup_epochs', 20)
        base_lr = lr_config.get('base_lr', 1e-4)
        min_lr = lr_config.get('min_lr', 1e-5)
        
        # Add warmup steps from a smaller learning rate. warmup_epochs=0 means no warmup —
        # skip the loop entirely (extend_finetune uses this since the model is already
        # converged and we don't want an LR ramp-up).
        if warmup_epochs > 0:
            for i in range(warmup_epochs + 1):
                lr_schedule[i] = min_lr + (base_lr - min_lr) * (i / warmup_epochs)
        
        # Add regular learning rate schedule after warmup
        if 'schedule' in lr_config:
            for epoch_str, lr_value in lr_config['schedule'].items():
                lr_schedule[int(epoch_str)] = float(lr_value)
        else:
            lr_schedule[warmup_epochs] = 1e-4
            lr_schedule[20000] = 1e-5
    else:
        warmup_epochs = 20
        base_lr = 1e-4
        min_lr = 1e-5
        
        for i in range(warmup_epochs + 1):
            lr_schedule[i] = min_lr + (base_lr - min_lr) * (i / warmup_epochs)
        
        lr_schedule[warmup_epochs] = 1e-4
        lr_schedule[20000] = 1e-5
    
    return lr_schedule, warmup_epochs, base_lr, min_lr


def detect_weight_explosion(current_weights, previous_weights, loss_value, 
                           threshold_factor=15.0, loss_threshold=1e5):
    """
    Detect if weights have exploded based on sudden weight increases or loss explosion.
    
    Args:
        current_weights: Current weight dictionary
        previous_weights: Previous weight dictionary
        loss_value: Current loss value
        threshold_factor: Maximum allowed weight increase ratio
        loss_threshold: Maximum allowed loss value
    
    Returns:
        Tuple of (is_exploded: bool, message: str)
    """
    if not previous_weights:
        return False, "No previous weights to compare"
    
    # Check for sudden weight increases
    for key in current_weights:
        if key in previous_weights:
            ratio = current_weights[key] / max(previous_weights[key], 1e-10)
            if ratio > threshold_factor:
                return True, f"Weight '{key}' increased {ratio:.2f}x (from {previous_weights[key]:.2e} to {current_weights[key]:.2e})"
    
    # Check for loss explosion
    if loss_value > loss_threshold:
        return True, f"Loss exploded to {loss_value:.2e} (threshold: {loss_threshold:.2e})"
    
    return False, "Weights stable"


def parse_gradnorm_excluded_tasks(curriculum_config):
    """Parse tasks excluded from GradNorm (use curriculum interpolation instead).

    Args:
        curriculum_config: Curriculum configuration dictionary or None

    Returns:
        Set of task names excluded from GradNorm
    """
    if curriculum_config and 'gradnorm_excluded_tasks' in curriculum_config:
        return set(curriculum_config['gradnorm_excluded_tasks'])
    return set()


def interpolate_curriculum_weight(task, epoch, curriculum_stages):
    """Interpolate a task's weight from curriculum stages at any epoch.

    Args:
        task: Task name (e.g., 'data', 'phase')
        epoch: Current epoch
        curriculum_stages: Dictionary mapping epoch -> weight adjustments

    Returns:
        Interpolated weight value
    """
    if not curriculum_stages:
        return 1.0
    stage_epochs = sorted(curriculum_stages.keys())
    current_stage = max([s for s in stage_epochs if s <= epoch], default=stage_epochs[0])
    target_w = curriculum_stages[current_stage].get(task, 0.0)
    next_stages = [s for s in stage_epochs if s > current_stage]
    if next_stages:
        next_stage = min(next_stages)
        next_w = curriculum_stages[next_stage].get(task, target_w)
        alpha = min((epoch - current_stage) / (next_stage - current_stage), 1.0)
        return (1 - alpha) * target_w + alpha * next_w
    return target_w


def parse_all_training_config(curriculum_config, task, run_mode, adaptive_start_epoch, n_epochs, start_epoch=0):
    """
    Parse all training configuration into a structured object.
    
    This is a convenience function that calls all individual parse functions
    and returns a consolidated configuration dictionary.
    
    Args:
        curriculum_config: Curriculum configuration dictionary or None
        task: 'train' or 'predict'
        run_mode: 'pinn' or 'pinn+phase'
        adaptive_start_epoch: Epoch to start adaptive weighting
        n_epochs: Total number of epochs
        start_epoch: Starting epoch (for resume)
    
    Returns:
        Dictionary with all parsed configuration
    """
    weights = parse_initial_weights(curriculum_config, task, run_mode)
    curriculum_stages = parse_curriculum_stages(curriculum_config, task)
    gradnorm_alpha, gradnorm_lr, weight_update_freq = parse_gradnorm_params(curriculum_config)
    target_rates = parse_target_rates(curriculum_config)
    weight_caps = parse_weight_caps(curriculum_config, task, run_mode)
    time_curriculum = parse_time_curriculum(curriculum_config)
    lr_schedule, warmup_epochs, base_lr, min_lr = parse_learning_rate_schedule(curriculum_config)
    
    # Phase loss introduction epoch (derived from stages)
    phase_intro_epoch = derive_phase_intro_epoch(curriculum_stages)
    if phase_intro_epoch is None:
        phase_intro_epoch = float('inf')
    
    # LBFGS switch epoch
    lbfgs_switch_fraction = curriculum_config.get('lbfgs_switch_fraction', 0.8) if curriculum_config else 0.8
    lbfgs_switch_epoch = start_epoch + int(lbfgs_switch_fraction * n_epochs)

    # Min dynamics and data floor parameters
    gradnorm_data_floor_mult = curriculum_config.get('gradnorm_data_floor_mult', 0.1) if curriculum_config else 0.1
    max_pde_data_ratio = curriculum_config.get('max_pde_data_ratio', None) if curriculum_config else None
    min_rate_threshold = curriculum_config.get('min_rate_threshold', 1e-5) if curriculum_config else 1e-5

    return {
        'weights': weights,
        'curriculum_stages': curriculum_stages,
        'curriculum_end_epoch': adaptive_start_epoch,
        'gradnorm_alpha': gradnorm_alpha,
        'gradnorm_lr': gradnorm_lr,
        'weight_update_freq': weight_update_freq,
        'target_rates': target_rates,
        'weight_caps': weight_caps,
        'time_curriculum': time_curriculum,
        'lr_schedule': lr_schedule,
        'warmup_epochs': warmup_epochs,
        'base_lr': base_lr,
        'min_lr': min_lr,
        'phase_intro_epoch': phase_intro_epoch,
        'lbfgs_switch_fraction': lbfgs_switch_fraction,
        'lbfgs_switch_epoch': lbfgs_switch_epoch,
        'gradnorm_data_floor_mult': gradnorm_data_floor_mult,
        'max_pde_data_ratio': max_pde_data_ratio,
        'min_rate_threshold': min_rate_threshold,
    }
