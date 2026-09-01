# ------------------------------------------------------------------------------------------
# Copyright (c) 2026 Chen Zhang, Tao Li
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

# Set matplotlib backend for headless servers (must be before pyplot import)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster/server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from loguru import logger


def plot_calibration(real_np=None, predicted_np=None, list_var=None):

    num_subplots = len(list_var)

    # -----Set fixed dimensions for each subplot-----
    subplot_height = 3  # height in inches for each subplot
    subplot_width = 7   # width in inches for the figure
    
    # -----Calculate total figure height based on number of subplots-----
    total_height = subplot_height * num_subplots
    
    # -----Create figure with dynamic height-----
    fig, axes = plt.subplots(nrows=num_subplots, ncols=1, 
                            figsize=(subplot_width, total_height))
    
    # -----Ensure axes is always an array (even with single subplot)-----
    if num_subplots == 1:
        axes = [axes]

    for i in range(num_subplots):
        plot_min = min(real_np[:, i].min(), predicted_np[:, i].min())
        plot_max = max(real_np[:, i].max(), predicted_np[:, i].max())

        # -----Plot scatter and diagonal line-----
        axes[i].scatter(real_np[:, i], predicted_np[:, i], alpha=0.5)
        axes[i].plot([plot_min, plot_max], [plot_min, plot_max], 'r--')
        
        # -----Set labels and title-----
        axes[i].set_title(f'Dimension {list_var[i]}')
        axes[i].set_xlabel('Real Values')
        axes[i].set_ylabel('Predicted Values')

        # -----Set axis limits-----
        axes[i].set_xlim(plot_min, plot_max)
        axes[i].set_ylim(plot_min, plot_max)

    # -----Adjust spacing between subplots-----
    fig.tight_layout(pad=2.0)  # Increase padding between subplots
    
    return fig, axes

def plot_confusion_matrix(real_np=None, predicted_np=None, list_var=None):
    # -----Generate confusion matrix-----
    cm = confusion_matrix(real_np, predicted_np)

    # -----Plot heatmap-----
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    return fig


def plot_loss_history(loss_history, model_name, save_path=None):
    """
    Plot training and validation loss history for a model
    
    Args:
        loss_history: Dictionary containing 'train_loss' and 'val_loss' lists
        model_name: Name of the model for the plot title
        save_path: Optional path to save the figure
    
    Returns:
        fig: matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Handle empty loss history
    if not loss_history.get('train_loss') or len(loss_history['train_loss']) == 0:
        ax.text(0.5, 0.5, 'No training data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(f'{model_name} - Loss History')
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Empty loss history plot saved to {save_path}")
        return fig
    
    epochs = range(1, len(loss_history['train_loss']) + 1)
    
    # Plot training and validation loss
    ax.plot(epochs, loss_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, loss_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Labels and title
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{model_name} - Training History', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    # Set y-axis to log scale if the range is large
    if max(loss_history['train_loss']) / min(loss_history['train_loss']) > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Loss (log scale)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Loss history plot saved to {save_path}")
    
    return fig


def plot_classification_metrics(loss_history, model_name, save_path=None):
    """
    Plot training history for classification model including loss and accuracy
    
    Args:
        loss_history: Dictionary containing 'train_loss', 'val_loss', and 'accuracy' lists
        model_name: Name of the model for the plot title
        save_path: Optional path to save the figure
    
    Returns:
        fig: matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Handle empty loss history
    if not loss_history.get('train_loss') or len(loss_history['train_loss']) == 0:
        ax1.text(0.5, 0.5, 'No training data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes, fontsize=14)
        ax1.set_title(f'{model_name} - Loss History')
        ax2.text(0.5, 0.5, 'No accuracy data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Accuracy History')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Empty classification metrics plot saved to {save_path}")
        return fig
    
    epochs = range(1, len(loss_history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, loss_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, loss_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Loss History', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    
    # Plot accuracy
    ax2.plot(epochs, loss_history['accuracy'], 'g-', label='Validation Accuracy', linewidth=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'{model_name} - Accuracy History', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Classification metrics plot saved to {save_path}")
    
    return fig


def plot_all_models_comparison(loss_histories, model_names, save_path=None):
    """
    Plot comparison of loss histories for multiple models
    
    Args:
        loss_histories: List of loss history dictionaries
        model_names: List of model names
        save_path: Optional path to save the figure
    
    Returns:
        fig: matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Check if all histories are empty
    all_empty = all(not history.get('train_loss') or len(history['train_loss']) == 0 
                    for history in loss_histories)
    
    if all_empty:
        ax1.text(0.5, 0.5, 'No training data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Training Loss Comparison')
        ax2.text(0.5, 0.5, 'No validation data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Validation Loss Comparison')
        plt.suptitle('Model Training Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Empty model comparison plot saved to {save_path}")
        return fig
    
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
    
    # Plot training losses
    for i, (history, name) in enumerate(zip(loss_histories, model_names)):
        if history.get('train_loss') and len(history['train_loss']) > 0:
            epochs = range(1, len(history['train_loss']) + 1)
            color = colors[i % len(colors)]
            ax1.plot(epochs, history['train_loss'], f'{color}-', label=f'{name}', linewidth=2, alpha=0.7)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    
    # Plot validation losses
    for i, (history, name) in enumerate(zip(loss_histories, model_names)):
        if history.get('val_loss') and len(history['val_loss']) > 0:
            epochs = range(1, len(history['val_loss']) + 1)
            color = colors[i % len(colors)]
            ax2.plot(epochs, history['val_loss'], f'{color}-', label=f'{name}', linewidth=2, alpha=0.7)
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.suptitle('Model Training Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    return fig


def plot_gradient_diagnostics(gradient_diagnostics, grad_summary, model_name, save_path=None):
    """
    Create comprehensive gradient flow diagnostics plot
    
    Args:
        gradient_diagnostics: Dict with model gradient statistics
        grad_summary: Dict with gradient health summary
        model_name: Name for plot title
        save_path: Path to save plot
    
    Returns:
        matplotlib figure
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import numpy as np
    
    if len(gradient_diagnostics) == 0:
        logger.warning("No gradient diagnostics data provided")
        return None
    
    # Filter out models with all-zero gradients to prevent plotting issues
    filtered_diagnostics = {}
    for model_key, stats in gradient_diagnostics.items():
        has_non_zero = False
        for layer_name, layer_stats in stats.items():
            if layer_stats['avg_norm'] > 1e-15:  # Check for meaningful gradients
                has_non_zero = True
                break
        if has_non_zero:
            filtered_diagnostics[model_key] = stats
        else:
            logger.warning(f"Model {model_key} has zero gradients everywhere - excluding from gradient plots")
    
    if len(filtered_diagnostics) == 0:
        logger.warning("All models have zero gradients - cannot create gradient plots")
        # Create a simple informational figure instead of returning None
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'All models have zero gradients\nNo gradient diagnostics available', 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=16, fontweight='bold', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{model_name} - Gradient Diagnostics: No Data Available', fontsize=14)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"No-data gradient plot saved to {save_path}")
        
        return fig
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Gradient Flow Diagnostics (Non-Zero Models Only)', fontsize=16, fontweight='bold')
    
    # Collect data for plotting from filtered diagnostics
    model_names = list(filtered_diagnostics.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot 1: Average gradient norms by layer
    ax1 = axes[0, 0]
    for i, (model_name_key, stats) in enumerate(filtered_diagnostics.items()):
        layer_names = list(stats.keys())
        avg_norms = [stats[layer]['avg_norm'] for layer in layer_names]
        x_pos = range(len(layer_names))
        ax1.bar([x + i*0.25 for x in x_pos], avg_norms, 
               width=0.25, label=model_name_key, color=colors[i], alpha=0.7)
    
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Average Gradient Norm')
    ax1.set_title('Average Gradient Norms by Layer')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Max vs Min gradient norms
    ax2 = axes[0, 1]
    for i, (model_name_key, stats) in enumerate(filtered_diagnostics.items()):
        layer_names = list(stats.keys())
        max_norms = [stats[layer]['max_norm'] for layer in layer_names]
        min_norms = [stats[layer]['min_norm'] for layer in layer_names]
        x_pos = range(len(layer_names))
        ax2.scatter([x + i*0.1 for x in x_pos], max_norms, 
                   label=f'{model_name_key} (max)', color=colors[i], marker='^', s=50)
        ax2.scatter([x + i*0.1 for x in x_pos], min_norms, 
                   label=f'{model_name_key} (min)', color=colors[i], marker='v', s=50, alpha=0.6)
    
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Max vs Min Gradient Norms by Layer')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Overall gradient health comparison
    ax3 = axes[1, 0]
    if grad_summary:
        model_names_summary = list(grad_summary.keys())
        avg_norms_summary = [grad_summary[name]['overall_avg_norm'] for name in model_names_summary]
        colors_bar = ['green' if grad_summary[name]['gradient_health'] == 'healthy' else 'red' 
                     for name in model_names_summary]
        
        bars = ax3.bar(model_names_summary, avg_norms_summary, color=colors_bar, alpha=0.7)
        ax3.set_ylabel('Overall Average Gradient Norm')
        ax3.set_title('Overall Gradient Health by Model')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Add health status text on bars
        for bar, model_name_key in zip(bars, model_names_summary):
            height = bar.get_height()
            health = grad_summary[model_name_key]['gradient_health']
            ax3.text(bar.get_x() + bar.get_width()/2., height*1.5,
                    health, ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Gradient standard deviation analysis
    ax4 = axes[1, 1]
    for i, (model_name_key, stats) in enumerate(filtered_diagnostics.items()):
        layer_names = list(stats.keys())
        avg_stds = [stats[layer]['avg_std'] for layer in layer_names]
        x_pos = range(len(layer_names))
        ax4.plot(x_pos, avg_stds, marker='o', label=model_name_key, 
                color=colors[i], linewidth=2, markersize=6)
    
    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Average Gradient Standard Deviation')
    ax4.set_title('Gradient Variability by Layer')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gradient diagnostics plot saved to {save_path}")
    
    return fig


def plot_individual_gradient_flow(model_name, gradient_stats, save_path=None):
    """
    Create detailed gradient flow plot for individual model
    
    Args:
        model_name: Name of the model
        gradient_stats: Gradient statistics for the model
        save_path: Path to save plot
    
    Returns:
        matplotlib figure
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import numpy as np
    
    if not gradient_stats:
        logger.warning(f"No gradient statistics provided for {model_name}")
        return None
    
    # Check if model has meaningful gradients
    has_non_zero = False
    for layer_name, layer_stats in gradient_stats.items():
        if layer_stats['avg_norm'] > 1e-15:
            has_non_zero = True
            break
    
    if not has_non_zero:
        logger.warning(f"Model {model_name} has zero gradients everywhere - skipping individual gradient plot")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{model_name.upper()} Model - Detailed Gradient Analysis', fontsize=14, fontweight='bold')
    
    layer_names = list(gradient_stats.keys())
    x_pos = range(len(layer_names))
    
    # Left plot: All gradient statistics
    avg_norms = [max(gradient_stats[layer]['avg_norm'], 1e-15) for layer in layer_names]  # Prevent log(0)
    max_norms = [max(gradient_stats[layer]['max_norm'], 1e-15) for layer in layer_names]
    min_norms = [max(gradient_stats[layer]['min_norm'], 1e-15) for layer in layer_names]
    
    ax1.plot(x_pos, avg_norms, 'o-', label='Average Norm', linewidth=2, markersize=6)
    ax1.plot(x_pos, max_norms, '^-', label='Max Norm', linewidth=1, markersize=4, alpha=0.7)
    ax1.plot(x_pos, min_norms, 'v-', label='Min Norm', linewidth=1, markersize=4, alpha=0.7)
    
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Gradient Norm')
    ax1.set_title('Gradient Norms by Layer')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Mean and std
    avg_means = [max(abs(gradient_stats[layer]['avg_mean']), 1e-15) for layer in layer_names]  # Prevent log(0)
    avg_stds = [max(gradient_stats[layer]['avg_std'], 1e-15) for layer in layer_names]
    
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(x_pos, avg_means, 'o-', color='blue', label='Avg |Mean|', linewidth=2, markersize=6)
    line2 = ax2_twin.plot(x_pos, avg_stds, 's-', color='red', label='Avg Std', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Average |Gradient Mean|', color='blue')
    ax2_twin.set_ylabel('Average Gradient Std', color='red')
    ax2.set_title('Gradient Mean and Variability')
    ax2.set_yscale('log')
    ax2_twin.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Individual gradient flow plot for {model_name} saved to {save_path}")
    
    return fig
