# ------------------------------------------------------------------------------------------
# GradNorm adaptive weight update utilities for PINN training
# Copyright (c) 2026 Chen Zhang, Tao Li, Xiaoqian Dang
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

import numpy as np
from loguru import logger


def gradnorm_update_weights(
    weights,
    grad_magnitudes,
    loss_terms,
    target_rates,
    alpha=0.12,
    learning_rate=0.025,
    loss_history=None,
    epoch=None,
    excluded_tasks=None
):
    """
    Advanced GradNorm-based adaptive weighting strategy.

    This implements an improved version of the GradNorm algorithm that dynamically
    adjusts loss weights based on gradient magnitudes and training rates to balance
    multi-task learning objectives.

    Args:
        weights: Current loss weights dictionary {task_name: weight}
        grad_magnitudes: Gradient magnitudes for each loss term {task_name: grad_mag}
        loss_terms: Current loss values (unused but kept for API consistency)
        target_rates: Target relative training rates for each task {task_name: rate}
        alpha: GradNorm hyperparameter (restoring force strength), default 0.12
        learning_rate: Weight update learning rate, default 0.025
        loss_history: Historical loss values for computing training rates
                     {task_name: [loss_values...]}
        epoch: Current epoch number
        excluded_tasks: Set of task names excluded from GradNorm adjustment.
                       These tasks are returned unchanged (managed by curriculum).

    Returns:
        Dictionary of updated weights {task_name: new_weight}
    """
    if excluded_tasks is None:
        excluded_tasks = set()

    managed_tasks = [t for t in weights.keys() if t not in excluded_tasks]

    # Calculate relative training rates (how fast each loss is decreasing)
    r_i = {}
    min_epoch_req = 3 if any(len(v) > 10 for v in loss_history.values()) else 10
    min_history_req = 3 if any(len(v) > 10 for v in loss_history.values()) else 5

    if loss_history and epoch and epoch > min_epoch_req:
        for task in managed_tasks:
            if task in loss_history and len(loss_history[task]) >= min_history_req:
                current_loss = loss_history[task][-1]
                window_size = min(10, max(3, len(loss_history[task]) // 3))
                past_loss = np.mean(loss_history[task][-window_size-3:-3]) if len(loss_history[task]) >= window_size + 3 else loss_history[task][0]

                if past_loss > 1e-10 and current_loss > 1e-10:
                    rate = (past_loss - current_loss) / (past_loss + 1e-8)
                    r_i[task] = max(0.01, min(2.0, rate))
                else:
                    r_i[task] = target_rates.get(task, 1.0) / 5.0
            else:
                r_i[task] = target_rates.get(task, 1.0) / 5.0
    else:
        for task in managed_tasks:
            r_i[task] = target_rates.get(task, 1.0) / 5.0

    if r_i:
        avg_rate = np.mean(list(r_i.values()))
        avg_rate = max(avg_rate, 1e-6)
    else:
        return weights.copy()

    grad_norm_targets = {}
    grad_values = list(grad_magnitudes.values())
    if not grad_values:
        return weights.copy()

    median_grad = np.median(grad_values) if len(grad_values) > 1 else grad_values[0]
    mean_grad = np.mean(grad_values)
    avg_grad = 0.7 * median_grad + 0.3 * mean_grad
    avg_grad = max(avg_grad, 1e-10)

    for task in managed_tasks:
        if task in grad_magnitudes and task in r_i:
            target_rel_rate = target_rates.get(task, 1.0)

            relative_rate = r_i[task] / avg_rate
            relative_rate = np.clip(relative_rate, 0.2, 5.0)

            alpha_damped = min(alpha, 0.5)
            grad_norm_targets[task] = avg_grad * (relative_rate ** alpha_damped) * target_rel_rate

    new_weights = weights.copy()
    max_weight_change = 0.3

    for task in managed_tasks:
        if task in grad_magnitudes and task in grad_norm_targets:
            current_grad = max(grad_magnitudes[task], 1e-10)
            target_grad = max(grad_norm_targets[task], 1e-10)

            grad_ratio = current_grad / target_grad
            grad_ratio = np.clip(grad_ratio, 0.1, 10.0)

            if grad_ratio > 1.0:
                adaptive_lr = learning_rate / (1.0 + 0.5 * (grad_ratio - 1.0))
            else:
                adaptive_lr = learning_rate * (0.5 + 0.5 * grad_ratio)

            weight_update = adaptive_lr * np.tanh(1 - grad_ratio)
            weight_update = np.clip(weight_update, -max_weight_change, max_weight_change)

            log_weight = np.log(max(weights[task], 1e-10))
            new_log_weight = log_weight + weight_update
            new_weights[task] = np.exp(new_log_weight)

            change_ratio = new_weights[task] / weights[task]
            if change_ratio > 1.5:
                new_weights[task] = weights[task] * 1.5
            elif change_ratio < 0.67:
                new_weights[task] = weights[task] * 0.67

    total_old = sum(weights[t] for t in managed_tasks)
    total_new = sum(new_weights[t] for t in managed_tasks)
    if total_new > 0 and total_old > 0:
        scale_factor = total_old / total_new
        scale_factor = np.clip(scale_factor, 0.8, 1.2)

        for task in managed_tasks:
            new_weights[task] = new_weights[task] * scale_factor

    return new_weights
