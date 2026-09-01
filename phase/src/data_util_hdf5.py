#!/usr/bin/env python3
# ------------------------------------------------------------------------------------------
# Copyright (c) 2026 Chen Zhang, Tao Li
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------
# HDF5-based dataset for ultra-fast random access

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger
import pickle


class LogStandardScaler(BaseEstimator, TransformerMixin):
    """
    A scaler that applies log-transform before StandardScaler.
    
    This is designed for targets that span multiple orders of magnitude within
    each variable. The log-transform compresses the range, making MSE loss
    treat relative errors more equally across different scales.
    
    Transform: y_scaled = StandardScaler(log(y + epsilon))
    Inverse:   y = exp(StandardScaler.inverse(y_scaled)) - epsilon
    
    Parameters:
        epsilon: Small constant to handle zeros (default: 1e-8)
        clip_min: Minimum value after inverse transform (default: 0.0)
        clip_max: Maximum value after inverse transform (default: 1.0 for mole fractions)
    """
    
    def __init__(self, epsilon=1e-8, clip_min=0.0, clip_max=1.0):
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.standard_scaler = StandardScaler()
        self._is_fitted = False
    
    def fit(self, X, y=None):
        """Fit the scaler on log-transformed data."""
        X = np.asarray(X)
        
        # Apply log transform
        X_log = np.log(X + self.epsilon)
        
        # Fit StandardScaler on log-transformed data
        self.standard_scaler.fit(X_log)
        
        # Store statistics for logging
        self.mean_ = self.standard_scaler.mean_
        self.scale_ = self.standard_scaler.scale_
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        self._is_fitted = True
        
        return self
    
    def transform(self, X):
        """Transform: log(X + epsilon) -> StandardScaler."""
        X = np.asarray(X)
        
        # Apply log transform
        X_log = np.log(X + self.epsilon)
        
        # Apply StandardScaler
        return self.standard_scaler.transform(X_log)
    
    def inverse_transform(self, X_scaled):
        """Inverse transform: StandardScaler.inverse -> exp - epsilon."""
        X_scaled = np.asarray(X_scaled)
        
        # Inverse StandardScaler
        X_log = self.standard_scaler.inverse_transform(X_scaled)
        
        # Inverse log transform
        X_original = np.exp(X_log) - self.epsilon
        
        # Clip to valid range (mole fractions should be in [0, 1])
        X_original = np.clip(X_original, self.clip_min, self.clip_max)
        
        return X_original
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'epsilon': self.epsilon,
            'clip_min': self.clip_min,
            'clip_max': self.clip_max
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def __repr__(self):
        return (f"LogStandardScaler(epsilon={self.epsilon}, "
                f"clip_min={self.clip_min}, clip_max={self.clip_max})")


class HDF5ThermoDataset(Dataset):
    """
    Ultra-fast PyTorch Dataset using HDF5 for random access.
    ~1000x faster than CSV-based lazy loading for large datasets.
    """
    
    def __init__(self, hdf5_file, scaler=None, fit_scaler=True, mode='r'):
        """
        Args:
            hdf5_file: Path to HDF5 file
            scaler: StandardScaler for features (if None, creates new one)
            fit_scaler: Whether to fit the scaler on this data
            mode: 'r' for read-only (training), 'r+' for read-write (if needed)
        """
        self.hdf5_file = hdf5_file
        self.mode = mode
        
        logger.info(f"Opening HDF5 dataset: {hdf5_file}")
        
        # Open HDF5 file and get basic info
        with h5py.File(hdf5_file, mode) as h5f:
            self.total_rows = h5f.attrs['total_rows']
            self.input_cols = h5f.attrs['input_cols'].split(',')
            self.target_col = h5f.attrs['class_col']
            
            # Get class distribution
            class_0_count = h5f.attrs.get('class_0_count', 0)
            class_1_count = h5f.attrs.get('class_1_count', 0)
            
            logger.info(f"Dataset size: {self.total_rows:,} samples")
            logger.info(f"Input columns: {self.input_cols}")
            logger.info(f"Target column: {self.target_col}")
            
            # Compute class weights
            if class_0_count > 0 and class_1_count > 0:
                total = class_0_count + class_1_count
                self.class_weights = torch.tensor(
                    [total / (2 * class_0_count), total / (2 * class_1_count)],
                    dtype=torch.float32
                )
                logger.info(f"Class distribution:")
                logger.info(f"  Class 0: {class_0_count:,} ({100*class_0_count/total:.2f}%)")
                logger.info(f"  Class 1: {class_1_count:,} ({100*class_1_count/total:.2f}%)")
                logger.info(f"Class weights: {self.class_weights.numpy()}")
            else:
                self.class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
                logger.warning("Class distribution not found in HDF5 attributes")
        
        # Initialize or use provided scaler
        self.scaler = scaler if scaler is not None else StandardScaler()
        
        # Fit scaler if needed
        if fit_scaler:
            self._fit_scaler()
        else:
            logger.info("Using pre-fitted scaler")
        
        # Keep file handle open for fast access during training
        self.h5_file = None
        self.features = None
        self.targets = None
    
    def _fit_scaler(self):
        """Fit scaler on a sample of the data"""
        logger.info("Fitting scaler on data sample...")
        
        with h5py.File(self.hdf5_file, 'r') as h5f:
            # Sample up to 1M rows for fitting
            sample_size = min(1000000, self.total_rows)
            
            # Use random sampling for better representation
            if sample_size < self.total_rows:
                indices = np.random.choice(self.total_rows, sample_size, replace=False)
                indices.sort()  # Sort for sequential access
            else:
                indices = np.arange(self.total_rows)
            
            # Load sample data in chunks for memory efficiency
            chunk_size = 100000
            sample_data = []
            
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i:i+chunk_size]
                sample_data.append(h5f['features'][chunk_indices])
            
            # Fit scaler
            X_sample = np.vstack(sample_data)
            self.scaler.fit(X_sample)
            logger.info(f"Fitted scaler on {len(X_sample):,} samples")
            
            # Log scaling statistics
            logger.info(f"Feature means: {self.scaler.mean_}")
            logger.info(f"Feature stds: {self.scaler.scale_}")
    
    def open(self):
        """Open HDF5 file for reading (call at start of epoch)"""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_file, self.mode)
            self.features = self.h5_file['features']
            self.targets = self.h5_file['class_labels']
    
    def close(self):
        """Close HDF5 file (call at end of epoch)"""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
            self.features = None
            self.targets = None
    
    def __len__(self):
        return self.total_rows
    
    def __getitem__(self, idx):
        """Get a single sample with ultra-fast HDF5 random access"""
        # Ensure file is open
        if self.h5_file is None:
            self.open()
        
        # Direct HDF5 access (microseconds instead of seconds!)
        x = self.features[idx].astype(np.float32)
        y = self.targets[idx]
        
        # Apply scaling
        x = self.scaler.transform(x.reshape(1, -1)).flatten()
        
        # Convert to tensors
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    
    def get_batch(self, indices):
        """Get multiple samples at once (more efficient for batching)"""
        if self.h5_file is None:
            self.open()
        
        # Get batch data
        X_batch = self.features[indices].astype(np.float32)
        y_batch = self.targets[indices]
        
        # Apply scaling
        X_batch = self.scaler.transform(X_batch)
        
        # Convert to tensors
        return torch.tensor(X_batch, dtype=torch.float32), torch.tensor(y_batch, dtype=torch.long)
    
    def __del__(self):
        """Ensure file is closed when dataset is deleted"""
        self.close()

    def __getstate__(self):
        """Prepare object for pickling (for multiprocessing)"""
        state = self.__dict__.copy()
        # Remove the unpicklable h5py objects
        state['h5_file'] = None
        state['features'] = None
        state['targets'] = None
        return state

    def __setstate__(self, state):
        """Restore object after unpickling (for multiprocessing)"""
        self.__dict__.update(state)
        # File will be opened on first access in worker process

    def get_class_weights(self):
        """Return class weights for balanced training"""
        return self.class_weights

    def get_scaler(self):
        """Return the fitted scaler"""
        return self.scaler


def load_pretrained_scalers(scaler_dir, model_type):
    """
    Load pre-fitted scalers for a specific model type
    
    Args:
        scaler_dir: Directory containing the saved scalers
        model_type: One of 'clf', 'reg_x', or 'reg_y'
    
    Returns:
        For clf: (feature_scaler, class_weights)
        For reg_x/reg_y: (feature_scaler, target_scaler)
    """
    if model_type == 'clf':
        feature_path = os.path.join(scaler_dir, 'clf_feature_scaler.pkl')
        weights_path = os.path.join(scaler_dir, 'clf_class_weights.npy')
        
        with open(feature_path, 'rb') as f:
            feature_scaler = pickle.load(f)
        
        if os.path.exists(weights_path):
            class_weights = np.load(weights_path)
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            class_weights = None
        
        logger.info(f"Loaded pre-fitted classification scalers from {scaler_dir}")
        return feature_scaler, class_weights
    
    elif model_type in ['reg_x', 'reg_y']:
        feature_path = os.path.join(scaler_dir, f'{model_type}_feature_scaler.pkl')
        target_path = os.path.join(scaler_dir, f'{model_type}_target_scaler.pkl')
        
        with open(feature_path, 'rb') as f:
            feature_scaler = pickle.load(f)
        
        with open(target_path, 'rb') as f:
            target_scaler = pickle.load(f)
        
        logger.info(f"Loaded pre-fitted {model_type} scalers from {scaler_dir}")
        return feature_scaler, target_scaler
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class HDF5ThermoRegressionDataset(Dataset):
    """
    Ultra-fast PyTorch Dataset for regression using HDF5.
    Supports phase filtering and target scaling.
    """
    
    def __init__(self, 
                 hdf5_file: str,
                 target_cols: list,
                 phase_filter: str = None,
                 feature_scaler: StandardScaler = None,
                 target_scaler: StandardScaler = None,
                 fit_scalers: bool = True,
                 mode: str = 'r'):
        """
        Args:
            hdf5_file: Path to HDF5 file
            target_cols: List of target column names for regression
            phase_filter: 'two_phase' (class_true==0), 'single_phase' (class_true==1), or None
            feature_scaler: StandardScaler for features
            target_scaler: StandardScaler for targets
            fit_scalers: Whether to fit the scalers on this data
            mode: HDF5 file access mode
        """
        self.hdf5_file = hdf5_file
        self.target_cols = target_cols
        self.phase_filter = phase_filter
        self.mode = mode
        
        logger.info(f"Opening HDF5 regression dataset: {hdf5_file}")
        logger.info(f"Phase filter: {phase_filter}")
        logger.info(f"Target columns: {target_cols}")
        
        # Open HDF5 file and get basic info
        with h5py.File(hdf5_file, mode) as h5f:
            self.input_cols = h5f.attrs['input_cols'].split(',')
            self.total_rows_raw = h5f.attrs['total_rows']
            
            # Find where the requested target columns are located
            self.targets_dataset_name = None
            self.target_indices = None
            
            # Try reg_x_targets first
            if 'reg_x_cols' in h5f.attrs and 'reg_x_targets' in h5f:
                reg_x_cols = h5f.attrs['reg_x_cols'].split(',')
                if all(col in reg_x_cols for col in target_cols):
                    self.targets_dataset_name = 'reg_x_targets'
                    self.target_indices = [reg_x_cols.index(col) for col in target_cols]
            
            # Try reg_y_targets if not found yet
            if self.targets_dataset_name is None and 'reg_y_cols' in h5f.attrs and 'reg_y_targets' in h5f:
                reg_y_cols = h5f.attrs['reg_y_cols'].split(',')
                if all(col in reg_y_cols for col in target_cols):
                    self.targets_dataset_name = 'reg_y_targets'
                    self.target_indices = [reg_y_cols.index(col) for col in target_cols]
            
            # If still not found, raise error
            if self.targets_dataset_name is None:
                available_cols = []
                if 'reg_x_cols' in h5f.attrs:
                    available_cols.append(f"reg_x_cols: {h5f.attrs['reg_x_cols']}")
                if 'reg_y_cols' in h5f.attrs:
                    available_cols.append(f"reg_y_cols: {h5f.attrs['reg_y_cols']}")
                
                raise ValueError(
                    f"Cannot find requested target columns in HDF5 file.\n"
                    f"Requested: {target_cols}\n"
                    f"Available:\n  " + "\n  ".join(available_cols) if available_cols else "  None"
                )
            
            logger.info(f"Will load targets from '{self.targets_dataset_name}': {target_cols}")
            
            # Apply phase filtering to get actual dataset size
            if phase_filter is not None:
                logger.info("Applying phase filter...")
                
                # Validate class_labels dataset exists
                if 'class_labels' not in h5f:
                    raise ValueError(
                        f"Required 'class_labels' dataset not found in HDF5 file.\n"
                        f"Available datasets: {list(h5f.keys())}\n"
                        f"Phase filtering requires 'class_labels' to identify two-phase vs single-phase samples."
                    )
                
                targets_all = h5f['class_labels'][:]  # Load class labels
                
                # Validate class_labels contains only 0 and 1
                unique_values = np.unique(targets_all)
                if not np.all(np.isin(unique_values, [0, 1])):
                    raise ValueError(
                        f"Invalid values in 'class_labels' dataset: {unique_values}\n"
                        f"Expected only 0 (two-phase) and 1 (single-phase)."
                    )
                
                if phase_filter == 'two_phase':
                    self.phase_mask = targets_all == 0
                elif phase_filter == 'single_phase':
                    self.phase_mask = targets_all == 1
                else:
                    raise ValueError(f"Unknown phase_filter: {phase_filter}")
                
                self.filtered_indices = np.where(self.phase_mask)[0]
                self.total_rows = len(self.filtered_indices)
                
                # Ensure we have samples after filtering
                if self.total_rows == 0:
                    class_0_count = np.sum(targets_all == 0)
                    class_1_count = np.sum(targets_all == 1)
                    raise ValueError(
                        f"Phase filter '{phase_filter}' resulted in ZERO samples!\n"
                        f"Dataset has {class_0_count:,} two-phase and {class_1_count:,} single-phase samples.\n"
                        f"Check if 'class_labels' is set up correctly in your HDF5 file."
                    )
                
                logger.info(f"Phase filtering: {self.total_rows:,} samples match '{phase_filter}' "
                           f"({100*self.total_rows/self.total_rows_raw:.1f}% of total)")
            else:
                self.phase_mask = None
                self.filtered_indices = None
                self.total_rows = self.total_rows_raw
                logger.info(f"No phase filtering: using all {self.total_rows:,} samples")
        
        # Initialize scalers
        self.feature_scaler = feature_scaler if feature_scaler is not None else StandardScaler()
        self.target_scaler = target_scaler if target_scaler is not None else StandardScaler()
        
        # Store target column indices (we'll need to extract from features)
        # For now, assume targets are stored separately in HDF5 or we extract from features
        # This will depend on your HDF5 structure
        
        # Fit scalers if needed
        if fit_scalers:
            self._fit_scalers()
        else:
            logger.info("Using pre-fitted scalers")
        
        # Keep file handle open
        self.h5_file = None
        self.features = None
        self.targets_raw = None
    
    def _fit_scalers(self):
        """Fit scalers on a sample of the filtered data"""
        logger.info("Fitting scalers on regression data sample...")
        
        with h5py.File(self.hdf5_file, 'r') as h5f:
            # Sample up to 1M rows for fitting
            if self.filtered_indices is not None:
                sample_size = min(1000000, len(self.filtered_indices))
                if sample_size < len(self.filtered_indices):
                    sample_indices = np.random.choice(len(self.filtered_indices), sample_size, replace=False)
                    sample_row_indices = self.filtered_indices[sample_indices]
                else:
                    sample_row_indices = self.filtered_indices
            else:
                sample_size = min(1000000, self.total_rows)
                sample_row_indices = np.random.choice(self.total_rows, sample_size, replace=False)
            
            sample_row_indices.sort()  # Sort for sequential access
            
            # Load sample data
            X_sample = h5f['features'][sample_row_indices]
            
            # Load regression targets using the determined dataset and column indices
            y_sample_full = h5f[self.targets_dataset_name][sample_row_indices].astype(np.float32)
            y_sample = y_sample_full[:, self.target_indices]
            
            # Fit scalers
            self.feature_scaler.fit(X_sample)
            self.target_scaler.fit(y_sample)
            
            logger.info(f"Fitted scalers on {len(X_sample):,} samples")
    
    def open(self):
        """Open HDF5 file for reading"""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_file, self.mode)
            self.features = self.h5_file['features']
            self.targets_raw = self.h5_file['class_labels']  # Class labels
    
    def close(self):
        """Close HDF5 file"""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
            self.features = None
            self.targets_raw = None
    
    def __len__(self):
        return self.total_rows
    
    def __getitem__(self, idx):
        """Get a single regression sample"""
        if self.h5_file is None:
            self.open()
        
        # Map filtered index to actual row index
        if self.filtered_indices is not None:
            actual_idx = self.filtered_indices[idx]
        else:
            actual_idx = idx
        
        # Load features
        x = self.features[actual_idx].astype(np.float32)
        
        # Load regression targets using the determined dataset and column indices
        y_full = self.h5_file[self.targets_dataset_name][actual_idx].astype(np.float32)
        y = y_full[self.target_indices]
        
        # Apply scaling
        x = self.feature_scaler.transform(x.reshape(1, -1)).flatten()
        y = self.target_scaler.transform(y.reshape(1, -1)).flatten()
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def inverse_transform_targets(self, scaled_targets):
        """Convert scaled predictions back to original scale"""
        if isinstance(scaled_targets, torch.Tensor):
            scaled_targets = scaled_targets.cpu().numpy()
        return self.target_scaler.inverse_transform(scaled_targets)
    
    def get_feature_scaler(self):
        """Return the fitted feature scaler"""
        return self.feature_scaler
    
    def get_target_scaler(self):
        """Return the fitted target scaler"""
        return self.target_scaler
    
    def __del__(self):
        """Ensure file is closed"""
        self.close()

    def __getstate__(self):
        """Prepare object for pickling (for multiprocessing)"""
        state = self.__dict__.copy()
        # Remove the unpicklable h5py objects
        state['h5_file'] = None
        state['features'] = None
        return state

    def __setstate__(self, state):
        """Restore object after unpickling (for multiprocessing)"""
        self.__dict__.update(state)
        # File will be opened on first access in worker process

# ========================================
# Classic Mode: In-Memory Datasets
# ========================================

class InMemoryThermoDataset(Dataset):
    """
    Classic mode: Load entire dataset into memory (RAM/GPU) for fastest training.
    Best for smaller datasets or when you have sufficient GPU memory.
    """
    
    def __init__(self, hdf5_file, scaler=None, fit_scaler=True, device='cpu', pin_memory=True):
        """
        Args:
            hdf5_file: Path to HDF5 file
            scaler: StandardScaler for features (if None, creates new one)
            fit_scaler: Whether to fit the scaler on this data
            device: Device to store tensors ('cpu' or 'cuda')
            pin_memory: If True and device='cpu', pin memory for faster GPU transfer
        """
        self.device = device
        self.pin_memory = pin_memory and device == 'cpu'
        
        logger.info(f"Loading dataset into memory: {hdf5_file}")
        logger.info(f"Target device: {device}")
        
        # Load all data from HDF5
        with h5py.File(hdf5_file, 'r') as h5f:
            # Load metadata
            self.total_rows = h5f.attrs['total_rows']
            self.input_cols = h5f.attrs['input_cols'].split(',')
            self.target_col = h5f.attrs['class_col']
            
            # Load all features and targets into memory
            logger.info(f"Loading {self.total_rows:,} samples into memory...")
            features = h5f['features'][:].astype(np.float32)
            targets = h5f['class_labels'][:].astype(np.int64)
            
            # Get class distribution
            class_0_count = h5f.attrs.get('class_0_count', 0)
            class_1_count = h5f.attrs.get('class_1_count', 0)
            
            logger.info(f"Dataset size: {self.total_rows:,} samples")
            logger.info(f"Input columns: {self.input_cols}")
            logger.info(f"Target column: {self.target_col}")
            logger.info(f"Memory usage: ~{features.nbytes / 1024**3:.2f} GB (features)")
            
            # Compute class weights
            if class_0_count > 0 and class_1_count > 0:
                total = class_0_count + class_1_count
                self.class_weights = torch.tensor(
                    [total / (2 * class_0_count), total / (2 * class_1_count)],
                    dtype=torch.float32
                )
                logger.info(f"Class distribution:")
                logger.info(f"  Class 0: {class_0_count:,} ({100*class_0_count/total:.2f}%)")
                logger.info(f"  Class 1: {class_1_count:,} ({100*class_1_count/total:.2f}%)")
                logger.info(f"Class weights: {self.class_weights.numpy()}")
            else:
                self.class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
        
        # Initialize or use provided scaler
        self.scaler = scaler if scaler is not None else StandardScaler()
        
        # Fit scaler if needed
        if fit_scaler:
            logger.info("Fitting scaler on dataset...")
            self.scaler.fit(features)
            logger.info(f"Feature means: {self.scaler.mean_}")
            logger.info(f"Feature stds: {self.scaler.scale_}")
        else:
            logger.info("Using pre-fitted scaler")
        
        # Apply scaling
        logger.info("Applying feature scaling...")
        features = self.scaler.transform(features).astype(np.float32)
        
        # Convert to tensors and move to device
        logger.info(f"Converting to tensors on device: {device}")
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)
        
        if device != 'cpu':
            logger.info(f"Moving data to {device}...")
            self.features = self.features.to(device)
            self.targets = self.targets.to(device)
            logger.info(f"✓ Data moved to {device}")
        elif self.pin_memory:
            logger.info("Pinning memory for faster GPU transfer...")
            self.features = self.features.pin_memory()
            self.targets = self.targets.pin_memory()
            logger.info("✓ Memory pinned")
        
        logger.info("✓ Dataset fully loaded into memory")
    
    def __len__(self):
        return self.total_rows
    
    def __getitem__(self, idx):
        """Direct tensor indexing - extremely fast!"""
        return self.features[idx], self.targets[idx]
    
    def get_class_weights(self):
        """Return class weights for balanced training"""
        return self.class_weights
    
    def get_scaler(self):
        """Return the fitted scaler"""
        return self.scaler


class InMemoryThermoRegressionDataset(Dataset):
    """
    Classic mode: Load entire regression dataset into memory (RAM/GPU).
    Supports phase filtering and target scaling.
    """
    
    def __init__(self, 
                 hdf5_file: str,
                 target_cols: list,
                 phase_filter: str = None,
                 feature_scaler: StandardScaler = None,
                 target_scaler: StandardScaler = None,
                 fit_scalers: bool = True,
                 device: str = 'cpu',
                 pin_memory: bool = True,
                 skip_target_scaling: bool = False):
        """
        Args:
            hdf5_file: Path to HDF5 file
            target_cols: List of target column names for regression
            phase_filter: 'two_phase' (class_true==0), 'single_phase' (class_true==1), or None
            feature_scaler: StandardScaler for features
            target_scaler: StandardScaler for targets (ignored if skip_target_scaling=True)
            fit_scalers: Whether to fit the scalers on this data
            device: Device to store tensors ('cpu' or 'cuda')
            pin_memory: If True and device='cpu', pin memory for faster GPU transfer
            skip_target_scaling: If True, skip target scaling entirely (for v3.0 models with softmax output)
        """
        self.target_cols = target_cols
        self.phase_filter = phase_filter
        self.device = device
        self.pin_memory = pin_memory and device == 'cpu'
        self.skip_target_scaling = skip_target_scaling
        
        logger.info(f"Loading regression dataset into memory: {hdf5_file}")
        logger.info(f"Phase filter: {phase_filter}")
        logger.info(f"Target columns: {target_cols}")
        logger.info(f"Target device: {device}")
        
        # Load all data from HDF5
        with h5py.File(hdf5_file, 'r') as h5f:
            self.input_cols = h5f.attrs['input_cols'].split(',')
            total_rows_raw = h5f.attrs['total_rows']
            
            # Load all data
            features_all = h5f['features'][:].astype(np.float32)
            class_labels = h5f['class_labels'][:]
            
            # Find the requested target columns in available datasets
            targets_all = None
            targets_dataset = None
            
            # Try reg_x_targets first
            if 'reg_x_cols' in h5f.attrs and 'reg_x_targets' in h5f:
                reg_x_cols = h5f.attrs['reg_x_cols'].split(',')
                if all(col in reg_x_cols for col in target_cols):
                    target_indices = [reg_x_cols.index(col) for col in target_cols]
                    targets_all = h5f['reg_x_targets'][:, target_indices].astype(np.float32)
                    targets_dataset = 'reg_x_targets'
            
            # Try reg_y_targets if not found yet
            if targets_all is None and 'reg_y_cols' in h5f.attrs and 'reg_y_targets' in h5f:
                reg_y_cols = h5f.attrs['reg_y_cols'].split(',')
                if all(col in reg_y_cols for col in target_cols):
                    target_indices = [reg_y_cols.index(col) for col in target_cols]
                    targets_all = h5f['reg_y_targets'][:, target_indices].astype(np.float32)
                    targets_dataset = 'reg_y_targets'
            
            # If still not found, raise error
            if targets_all is None:
                available_cols = []
                if 'reg_x_cols' in h5f.attrs:
                    available_cols.append(f"reg_x_cols: {h5f.attrs['reg_x_cols']}")
                if 'reg_y_cols' in h5f.attrs:
                    available_cols.append(f"reg_y_cols: {h5f.attrs['reg_y_cols']}")
                
                raise ValueError(
                    f"Cannot find requested target columns in HDF5 file.\n"
                    f"Requested: {target_cols}\n"
                    f"Available:\n  " + "\n  ".join(available_cols) if available_cols else "  None"
                )
            
            logger.info(f"Loaded targets from '{targets_dataset}': {target_cols}")
            
            # Apply phase filtering
            if phase_filter is not None:
                logger.info("Applying phase filter...")
                
                # Validate class_labels dataset exists
                if 'class_labels' not in h5f:
                    raise ValueError(
                        f"Required 'class_labels' dataset not found in HDF5 file.\n"
                        f"Available datasets: {list(h5f.keys())}\n"
                        f"Phase filtering requires 'class_labels' to identify two-phase vs single-phase samples."
                    )
                
                # Validate class_labels contains only 0 and 1
                unique_values = np.unique(class_labels)
                if not np.all(np.isin(unique_values, [0, 1])):
                    raise ValueError(
                        f"Invalid values in 'class_labels' dataset: {unique_values}\n"
                        f"Expected only 0 (two-phase) and 1 (single-phase)."
                    )
                
                if phase_filter == 'two_phase':
                    mask = class_labels == 0
                elif phase_filter == 'single_phase':
                    mask = class_labels == 1
                else:
                    raise ValueError(f"Unknown phase_filter: {phase_filter}")
                
                features = features_all[mask]
                targets = targets_all[mask]
                
                # Ensure we have samples after filtering
                if len(features) == 0:
                    class_0_count = np.sum(class_labels == 0)
                    class_1_count = np.sum(class_labels == 1)
                    raise ValueError(
                        f"Phase filter '{phase_filter}' resulted in ZERO samples!\n"
                        f"Dataset has {class_0_count:,} two-phase and {class_1_count:,} single-phase samples.\n"
                        f"Check if 'class_labels' is set up correctly in your HDF5 file."
                    )
                
                logger.info(f"Filtered: {len(features):,} / {total_rows_raw:,} samples")
            else:
                features = features_all
                targets = targets_all
            
            self.total_rows = len(features)
            logger.info(f"Dataset size: {self.total_rows:,} samples")
            logger.info(f"Input columns: {self.input_cols}")
            logger.info(f"Memory usage: ~{features.nbytes / 1024**3:.2f} GB (features)")
            logger.info(f"Memory usage: ~{targets.nbytes / 1024**3:.2f} GB (targets)")
        
        # Initialize or use provided scalers
        self.feature_scaler = feature_scaler if feature_scaler is not None else StandardScaler()
        self.target_scaler = target_scaler if target_scaler is not None else StandardScaler()
        
        # Fit scalers if needed
        if fit_scalers:
            logger.info("Fitting scalers on dataset...")
            self.feature_scaler.fit(features)
            if not skip_target_scaling:
                self.target_scaler.fit(targets)
                logger.info(f"Target means: {self.target_scaler.mean_}")
                logger.info(f"Target stds: {self.target_scaler.scale_}")
            else:
                logger.info("Skipping target scaler fitting (skip_target_scaling=True)")
            logger.info(f"Feature means: {self.feature_scaler.mean_}")
            logger.info(f"Feature stds: {self.feature_scaler.scale_}")
        else:
            logger.info("Using pre-fitted scalers")
        
        # Apply scaling
        if skip_target_scaling:
            logger.info("Applying feature scaling only (skip_target_scaling=True for v3.0)")
            features = self.feature_scaler.transform(features).astype(np.float32)
            # Keep targets as raw mole fractions (model outputs normalized via softmax)
            targets = targets.astype(np.float32)
        else:
            logger.info("Applying feature and target scaling...")
            features = self.feature_scaler.transform(features).astype(np.float32)
            targets = self.target_scaler.transform(targets).astype(np.float32)
        
        # Convert to tensors and move to device
        logger.info(f"Converting to tensors on device: {device}")
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)
        
        if device != 'cpu':
            logger.info(f"Moving data to {device}...")
            self.features = self.features.to(device)
            self.targets = self.targets.to(device)
            logger.info(f"✓ Data moved to {device}")
        elif self.pin_memory:
            logger.info("Pinning memory for faster GPU transfer...")
            self.features = self.features.pin_memory()
            self.targets = self.targets.pin_memory()
            logger.info("✓ Memory pinned")
        
        logger.info("✓ Dataset fully loaded into memory")
    
    def __len__(self):
        return self.total_rows
    
    def __getitem__(self, idx):
        """Direct tensor indexing - extremely fast!"""
        return self.features[idx], self.targets[idx]

    def inverse_transform_targets(self, scaled_targets):
        """Convert scaled predictions back to original scale
        
        For v3.0 (skip_target_scaling=True): returns as-is (already valid mole fractions)
        For v2.0 (skip_target_scaling=False): applies inverse transform
        """
        if isinstance(scaled_targets, torch.Tensor):
            scaled_targets = scaled_targets.cpu().numpy()
        
        if self.skip_target_scaling:
            # v3.0: model outputs are already valid mole fractions (softmax normalized)
            return scaled_targets
        else:
            # v2.0: inverse transform scaled targets
            return self.target_scaler.inverse_transform(scaled_targets)

    def get_feature_scaler(self):
        """Return the fitted feature scaler"""
        return self.feature_scaler

    def get_target_scaler(self):
        """Return the fitted target scaler (may be unused for v3.0)"""
        return self.target_scaler
