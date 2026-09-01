# ------------------------------------------------------------------------------------------
# Training visualization utilities for PINN
# Copyright (c) 2026 Chen Zhang, Tao Li, Xiaoqian Dang
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from loguru import logger


def plot_gradnorm_analysis(
    weight_history,
    grad_history,
    adaptive_weight_log,
    checkpoint_dir
):
    """
    Plot comprehensive GradNorm analysis visualizations.
    
    Args:
        weight_history: Dictionary mapping weight names to their history lists
        grad_history: Dictionary mapping gradient names to their history lists
        adaptive_weight_log: List of dictionaries with adaptive weight log entries
        checkpoint_dir: Directory to save the plots
    
    Returns:
        True if plots were saved successfully, False otherwise
    """
    try:
        # Check if we have any data to plot
        valid_weights = {}
        for k, v in weight_history.items():
            if len(v) > 0:  # Only include non-empty lists
                valid_weights[k] = v
        
        if not valid_weights or not any(len(v) > 0 for v in valid_weights.values()):
            logger.warning("No weight history data available to plot")
            return False
        
        # Create subplot layout for comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Weight evolution
        ax1 = axes[0, 0]
        weight_lengths = [len(v) for v in valid_weights.values()]
        if all(length > 0 for length in weight_lengths):
            has_valid_data = False
            for k, v in valid_weights.items():
                if len(v) > 0:
                    weight_epochs = list(range(len(v)))
                    v_array = np.array(v)
                    if np.all(np.isfinite(v_array)) and np.all(v_array > 0):
                        try:
                            ax1.semilogy(weight_epochs, v, label=f'{k} weight', linewidth=2)
                            has_valid_data = True
                        except Exception as e:
                            logger.warning(f"Failed to plot weight {k}: {e}")
            
            if has_valid_data:
                ax1.set_xlabel('Update Steps')
                ax1.set_ylabel('Weight Value (log scale)')
                ax1.set_title('GradNorm Weight Evolution')
                ax1.grid(True, which='both', linestyle='--', alpha=0.6)
                ax1.legend()
            else:
                ax1.text(0.5, 0.5, 'No valid weight data for plotting', 
                        ha='center', va='center', transform=ax1.transAxes)
        else:
            ax1.text(0.5, 0.5, 'No weight evolution data available', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # Plot 2: Gradient magnitudes
        ax2 = axes[0, 1]
        valid_grads = {}
        for k, v in grad_history.items():
            if len(v) > 0:
                valid_grads[k] = v
        
        if valid_grads:
            has_valid_grad_data = False
            for k, v in valid_grads.items():
                if len(v) > 0:
                    grad_epochs = list(range(len(v)))
                    v_array = np.array(v)
                    if np.all(np.isfinite(v_array)) and np.all(v_array > 0):
                        try:
                            ax2.semilogy(grad_epochs, v, label=f'{k} gradient', linewidth=2)
                            has_valid_grad_data = True
                        except Exception as e:
                            logger.warning(f"Failed to plot gradient {k}: {e}")
            
            if has_valid_grad_data:
                ax2.set_xlabel('Update Steps')
                ax2.set_ylabel('Gradient Magnitude (log scale)')
                ax2.set_title('Gradient Magnitude Evolution')
                ax2.grid(True, which='both', linestyle='--', alpha=0.6)
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'No valid gradient data for plotting', 
                        ha='center', va='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, 'No gradient data available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # Plot 3: Target rates vs current rates (if available)
        ax3 = axes[1, 0]
        if adaptive_weight_log:
            target_rates_plot = {}
            for entry in adaptive_weight_log:
                if 'target_rates' in entry:
                    for task, rate in entry['target_rates'].items():
                        if task not in target_rates_plot:
                            target_rates_plot[task] = []
                        target_rates_plot[task].append(rate)
            
            if target_rates_plot and any(len(rates) > 0 for rates in target_rates_plot.values()):
                for task, rates in target_rates_plot.items():
                    if len(rates) > 0:
                        ax3.plot(rates, label=f'{task} target', linewidth=2)
                
                ax3.set_xlabel('Update Steps')
                ax3.set_ylabel('Target Training Rate')
                ax3.set_title('GradNorm Target Rates')
                ax3.grid(True, linestyle='--', alpha=0.6)
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No target rate data available', 
                        ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, 'No adaptive weight log data', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Weight adaptation metrics
        ax4 = axes[1, 1]
        if adaptive_weight_log and len(adaptive_weight_log) > 1:
            weight_changes = {}
            for i, entry in enumerate(adaptive_weight_log[1:], 1):
                prev_entry = adaptive_weight_log[i-1]
                if 'weights' in entry and 'weights' in prev_entry:
                    for task in entry['weights']:
                        if task in prev_entry['weights']:
                            if task not in weight_changes:
                                weight_changes[task] = []
                            change = abs(entry['weights'][task] - prev_entry['weights'][task])
                            weight_changes[task].append(change)
            
            if weight_changes and any(len(changes) > 0 for changes in weight_changes.values()):
                for task, changes in weight_changes.items():
                    if len(changes) > 0:
                        ax4.semilogy(changes, label=f'{task} change', linewidth=2)
                
                ax4.set_xlabel('Update Steps')
                ax4.set_ylabel('Weight Change Magnitude (log scale)')
                ax4.set_title('Weight Adaptation Dynamics')
                ax4.grid(True, which='both', linestyle='--', alpha=0.6)
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, 'No weight change data available', 
                        ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for weight changes', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plot_path = os.path.join(checkpoint_dir, 'gradnorm_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"GradNorm analysis plot saved to: {plot_path}")
        plt.close()
        
        # Also create the simple weight plot for backward compatibility
        _plot_simple_weights(valid_weights, checkpoint_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to plot GradNorm analysis: {str(e)}")
        logger.error(f"Weight history keys: {list(weight_history.keys())}")
        for k, v in weight_history.items():
            logger.error(f"Weight '{k}' has {len(v)} entries")
        for k, v in grad_history.items():
            logger.error(f"Gradient '{k}' has {len(v)} entries")
        return False


def _plot_simple_weights(valid_weights, checkpoint_dir):
    """
    Create a simple weight evolution plot for backward compatibility.
    
    Args:
        valid_weights: Dictionary of weight histories with non-empty lists
        checkpoint_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    has_simple_plot_data = False
    
    for k, v in valid_weights.items():
        if len(v) > 0:
            epochs = list(range(len(v)))
            v_array = np.array(v)
            if np.all(np.isfinite(v_array)) and np.all(v_array > 0):
                try:
                    plt.semilogy(epochs, v, label=f'{k} weight', linewidth=2)
                    has_simple_plot_data = True
                except Exception as e:
                    logger.warning(f"Failed to plot weight {k} in simple plot: {e}")
    
    if has_simple_plot_data:
        plt.xlabel('Update Steps')
        plt.ylabel('Weight Value (log scale)')
        plt.title('GradNorm Adaptive Weight Evolution')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        
        plot_path = os.path.join(checkpoint_dir, 'adaptive_weights.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Adaptive weight history plot saved to: {plot_path}")
        plt.close()
    else:
        plt.close()
        logger.warning("No valid data for simple weight plot")
