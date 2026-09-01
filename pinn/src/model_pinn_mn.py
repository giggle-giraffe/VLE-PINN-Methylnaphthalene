# ------------------------------------------------------------------------------------------
# Copyright (c) 2026 Chen Zhang, Tao Li, Xiaoqian Dang
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

import math
import os
import sys
from loguru import logger

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(ROOT_DIR)
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

from .model_phase_mn import PhaseModel
from .plot import plot_trajectory, plot_extended_overlay

import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
torch.set_default_dtype(torch.float64)
add_safe_globals([nn.Sequential, nn.Linear, nn.ReLU, nn.Tanh, nn.Softplus])


class TimeEncoder(nn.Module):
    """
    Time encoder with logarithmic scaling for balanced sensitivity across all times.

    Problem solved:
    - Raw time values (e.g., 0, 0.02, 0.04...) are too small for Tanh to differentiate
    - Linear scaling (t * scale) causes Tanh saturation at late times, blocking gradients

    Solution: Logarithmic transformation log(1 + t * scale)
    - Spreads out early time values (good differentiation at t~0)
    - Compresses late time values (prevents Tanh saturation)
    - Maintains gradient flow across entire time range

    Example with time_scale=10.0:
        t=0.02 → log(1 + 0.2) ≈ 0.18  (spread out, Tanh active)
        t=0.20 → log(1 + 2.0) ≈ 1.10  (good gradient region)
        t=1.00 → log(1 + 10)  ≈ 2.40  (still in active region)
        t=6.67 → log(1 + 66.7) ≈ 4.22 (moderate, not saturated)
    """
    def __init__(self, encoder_dim=16, time_scale=10.0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.time_scale = time_scale

        self.time_net = nn.Sequential(
            nn.Linear(1, encoder_dim),
            nn.Tanh(),
            nn.Linear(encoder_dim, encoder_dim),
        )

        # Initialize with moderate gain (log transform already provides good spread)
        for i, m in enumerate(self.time_net.modules()):
            if isinstance(m, nn.Linear):
                # First layer: moderate gain (1.5) - log transform handles scaling
                # Second layer: standard gain (1.0) for stable output
                gain = 1.5 if i == 1 else 1.0
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t):
        """
        Args:
            t: Time tensor [batch_size, 1] with raw time values (e.g., 0, 0.02, 0.04...)
        Returns:
            Time encoding [batch_size, encoder_dim]
        """
        # Logarithmic scaling: spreads early times, compresses late times
        # log1p(x) = log(1 + x), numerically stable for small x
        t_transformed = torch.log1p(t * self.time_scale)

        return self.time_net(t_transformed)


class SampleEncoder(nn.Module):
    def __init__(self, num_sample_features=None, encoder_dim=32):
        super().__init__()
        # Network for processing actual features
        self.num_sample_features = num_sample_features

        self.register_buffer('feature_mean', torch.tensor([
            567.23,   # T: mean temperature (K)
            47.95,    # P: mean pressure (bar)
            1.17,     # m: mean catalyst mass (g)
            0.011,    # FH0: mean H2 flow rate
            0.0053,   # FS: mean solvent flow rate
            3.08e-4,  # FA_in: mean MN initial flow rate
            0.0,      # FB_in: typically 0 at start
            0.0,      # FC_in: typically 0 at start
            0.0,      # FD_in: typically 0 at start
        ], dtype=torch.float64))

        self.register_buffer('feature_std', torch.tensor([
            35.8,     # T: std from data
            11.0,     # P: std from data
            0.81,     # m: std from data (catalyst mass)
            0.0051,   # FH0: std from data
            0.0027,   # FS: std from data
            2.28e-4,  # FA_in: std from data
            1e-6,     # FB_in: small std to avoid division issues (typically 0)
            1e-6,     # FC_in: small std to avoid division issues (typically 0)
            1e-6,     # FD_in: small std to avoid division issues (typically 0)
        ], dtype=torch.float64))

        # Network for processing NORMALIZED features
        self.feature_net = nn.Sequential(
            nn.Linear(self.num_sample_features, 64),
            nn.Tanh(),
            nn.Linear(64, encoder_dim),
        )

        # Add direct connection from normalized input to output (skip connection)
        self.skip_connection = nn.Linear(self.num_sample_features, encoder_dim)

        # Initialize for feature encoding with proper gain for Tanh
        for m in self.feature_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=5/3)  # Gain for Tanh
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize skip connection with smaller but consistent gain ratio
        nn.init.xavier_uniform_(self.skip_connection.weight, gain=0.3)
        nn.init.zeros_(self.skip_connection.bias)

    def normalize_features(self, x):
        """Normalize input features to have zero mean and unit variance."""
        # Handle case where we have fewer features than expected (e.g., during testing)
        n_features = min(x.shape[1], self.num_sample_features)
        mean = self.feature_mean[:n_features]
        std = self.feature_std[:n_features]

        # Z-score normalization: (x - mean) / std
        # Add small epsilon to std to avoid division by zero
        normalized = (x[:, :n_features] - mean) / (std + 1e-10)

        return normalized

    def forward(self, x):
        real_features = x[:, :self.num_sample_features]
        normalized_features = self.normalize_features(real_features)

        # Process normalized features through neural network
        feature_encoding = self.feature_net(normalized_features)

        # Direct connection from normalized features
        skip_output = self.skip_connection(normalized_features)

        # Generate a unique fingerprint for each sample based on NORMALIZED features
        # This ensures different samples get different encodings
        batch_size = x.shape[0]
        unique_encodings = torch.zeros_like(feature_encoding)

        for i in range(batch_size):
            # Create a deterministic but unique encoding based on normalized features
            sample_features = normalized_features[i]

            # Create unique hash with alternating positive and negative weights
            weights = torch.tensor(
                [1.0, -1.5, 1.2, -0.9, 0.8, -1.2, 1.0, -0.7, 1.1][:self.num_sample_features],
                device=x.device, dtype=x.dtype
            )

            feature_hash_1 = torch.sum(sample_features * weights)

            feature_hash_2 = torch.sum(sample_features * torch.sin(
                torch.linspace(0.0, 6.28, self.num_sample_features, device=x.device, dtype=x.dtype)
            ))

            for j in range(feature_encoding.shape[1]):
                frequency = 3.0 + (j % 7) * 0.8
                phase_shift = (j % 5) * 0.5
                unique_encodings[i, j] = 1.0 * torch.sin(
                    frequency * torch.tensor(j, device=x.device, dtype=x.dtype) * feature_hash_1 / 5.0 + feature_hash_2 + phase_shift
                )

        alpha = 0.6
        beta = 0.2
        gamma = 0.2

        total = alpha + beta + gamma
        alpha = alpha / total
        beta = beta / total
        gamma = gamma / total

        mixed_encoding = alpha * feature_encoding + beta * unique_encodings + gamma * skip_output

        return mixed_encoding


class FeatureFusion(nn.Module):
    def __init__(self, time_encoder_dim=32, sample_encoder_dim=16):
        super().__init__()
        
        self.time_proj = nn.Linear(time_encoder_dim, time_encoder_dim)
        
        self.sample_proj = nn.Linear(sample_encoder_dim, sample_encoder_dim)
        
        self.time_norm = nn.LayerNorm(time_encoder_dim)
        self.sample_norm = nn.LayerNorm(sample_encoder_dim)
        
        self.mixer = nn.Sequential(
            nn.Linear(time_encoder_dim + sample_encoder_dim, time_encoder_dim + sample_encoder_dim),
            nn.GELU(),
            nn.Linear(time_encoder_dim + sample_encoder_dim, time_encoder_dim + sample_encoder_dim),
        )
        
        # Initialize weights consistently using xavier_uniform_
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use consistent xavier initialization
                if m is self.sample_proj:
                    # Higher gain for sample features but with uniform distribution
                    nn.init.xavier_uniform_(m.weight, gain=1.5)
                elif m in self.mixer:
                    # For GELU, use a gain similar to ReLU
                    nn.init.xavier_uniform_(m.weight, gain=1.41)  # √2
                else:
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, time_features, sample_features):        
        # Direct projections with normalization
        time_proj = self.time_norm(self.time_proj(time_features))
        sample_proj = self.sample_norm(self.sample_proj(sample_features))
        
        # Store originals for residual connection
        time_orig = time_features
        sample_orig = sample_features
        
        # Concatenate features (important: sample features first to give them priority)
        combined = torch.cat([sample_proj, time_proj], dim=1)
        
        # Apply mixer with residual connection
        mixed = self.mixer(combined)
        
        # Add strong residual connection from original features
        # This ensures we don't lose the original sample identity
        orig_combined = torch.cat([sample_orig, time_orig], dim=1)
        output = mixed + 0.5 * orig_combined  # High weight (0.5) for original features
        
        return output


class PINN(nn.Module):
    def __init__(self, num_sample_features=None, num_outputs=None, delta_t=None, config_phase_model=None,
                 device=None, time_batch_size=None, run_mode=None, pinn_inputs=None,
                 training_flag=None, output_scale=1e-6,
                 ads_a_log_min=-3.0, ads_a_log_range=6.0, ads_dh_max=30000.0, e_min=1000.0,
                 rxn_a_log_min=-1.0, rxn_a_log_range=9.0):
        """
        Args:
            num_sample_features: number of sample features to the PINN
            num_outputs: number of outputs from the PINN
            delta_t: time step for the PINN
            config_phase_model: configuration for the phase model
            device: device to run the PINN on
            time_batch_size: number of time points to process in each batch
            run_mode: mode to run the PINN in
            pinn_inputs: inputs to the PINN (['time', 'time+initials'])
            training_flag: flag to indicate if the PINN is in training mode
            output_scale: scale factor for network output (default: 1e-6)
            rxn_a_log_min: log10 lower bound for reaction pre-exponential (default: -1.0 → A_min=0.1)
            rxn_a_log_range: log10 range for reaction pre-exponential (default: 9.0 → A_max=10^8)
            ads_a_log_min: log10 lower bound for adsorption pre-exponential (default: -3.0 → A_min=0.001)
            ads_a_log_range: log10 range for adsorption pre-exponential (default: 6.0 → A_max=1000)
            ads_dh_max: upper bound for adsorption enthalpy ΔH in J/mol (default: 30000.0)
            e_min: minimum activation energy in J/mol (default: 1000.0). E ∈ [e_min, 80000]
        """
        super().__init__()

        assert pinn_inputs in ['time', 'time+initials'], f"Invalid PINN inputs: {pinn_inputs}"
        assert run_mode in ['pinn', 'pinn+phase'], f"Invalid run mode: {run_mode}"

        # Set device (GPU if available, otherwise CPU)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.config = {
            'num_sample_features': num_sample_features,
            'num_outputs': num_outputs,
            'delta_t': delta_t,
            'config_phase_model': config_phase_model,
            'time_batch_size': time_batch_size,
            'run_mode': run_mode,
            'pinn_inputs': pinn_inputs,
            'training_flag': training_flag,
            'output_scale': output_scale,
            'ads_a_log_min': ads_a_log_min,
            'ads_a_log_range': ads_a_log_range,
            'ads_dh_max': ads_dh_max,
            'e_min': e_min,
            'rxn_a_log_min': rxn_a_log_min,
            'rxn_a_log_range': rxn_a_log_range,
        }

        if isinstance(output_scale, (list, tuple)):
            self.output_scales = torch.tensor(output_scale, dtype=torch.float64)
            self.output_scale = max(output_scale)  # scalar for flowrate/monotonicity losses
        else:
            self.output_scales = torch.tensor([output_scale] * num_outputs, dtype=torch.float64)
            self.output_scale = output_scale
        self.rxn_a_log_min = rxn_a_log_min
        self.rxn_a_log_range = rxn_a_log_range
        self.ads_a_log_min = ads_a_log_min
        self.ads_a_log_range = ads_a_log_range
        self.ads_dh_max = ads_dh_max
        self.e_min = e_min

        # -----Initialize PINN parameters-----
        self.delta_t = delta_t
        self.epsilon = 1e-10
        self.time_batch_size = time_batch_size
        self.run_mode = run_mode
        self.pinn_inputs = pinn_inputs
        self.training_flag = training_flag
        # -----Phase model-----
        object.__setattr__(self, '_phase_model', PhaseModel(config=config_phase_model))

        # -----Neural network architecture-----
        self.time_encoder_dim = 16
        self.sample_encoder_dim = 16
        self.time_scale = 10.0  # Scale factor to amplify early time differences

        if pinn_inputs == 'time':
            self.num_inputs_pinn = 1
        else:
            self.num_inputs_pinn = self.time_encoder_dim + self.sample_encoder_dim
        
        self.sample_encoder = SampleEncoder(num_sample_features=self.config['num_sample_features'], encoder_dim=self.sample_encoder_dim)
        self.time_encoder = TimeEncoder(encoder_dim=self.time_encoder_dim, time_scale=self.time_scale)
        self.feature_fusion = FeatureFusion(time_encoder_dim=self.time_encoder_dim, sample_encoder_dim=self.sample_encoder_dim)

        # Final network layer setup
        self.net = nn.Sequential(
            nn.Linear(self.num_inputs_pinn, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, num_outputs),
            nn.Softplus(),
        )
        
        # Initialize weights for main network with proper gains
        for i, module in enumerate(self.net):
            if isinstance(module, nn.Linear):
                # Check what activation follows this linear layer
                if i < len(self.net) - 2 and isinstance(self.net[i+2], nn.Tanh):
                    # For Tanh activation
                    nn.init.xavier_uniform_(module.weight, gain=5/3)
                elif i == len(self.net) - 2:
                    # For Softplus in last layer
                    nn.init.xavier_uniform_(module.weight, gain=1.2)
                else:
                    # Default
                    nn.init.xavier_uniform_(module.weight, gain=1.0)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.sample_encoder.to(self.device)
        self.time_encoder.to(self.device)
        self.feature_fusion.to(self.device)
        self.net.to(self.device)

        # -----Initialize reaction parameters-----
        reaction_indices = [1, 2, 3, 4]
        species_indices = ['A', 'B', 'C', 'D', 'H']

        # Initialize pre-exponential factors with greater spread (A parameters)
        for i in reaction_indices:
            mean = 0.3 + 0.1 * i
            std = 0.05
                
            # Randomized initialization centered on physically reasonable values
            init_value = mean + std * torch.randn(1)
            setattr(self, f'x_A_{i}', nn.Parameter(init_value))

        # Initialize activation energies with physically meaningful spread (E parameters)
        for i in reaction_indices:
            if i % 3 == 0:
                mean = -1.5
                std = 1.0
            elif i % 3 == 1:
                mean = 1.5
                std = 1.0
            else:
                mean = 0.0
                std = 1.5
            
            # Randomized initialization with physical meaning
            init_value = mean + std * torch.randn(1)
            # Clamp to reasonable range for sigmoid input
            init_value = torch.clamp(init_value, -4.0, 4.0)
            setattr(self, f'y_E_{i}', nn.Parameter(init_value))

        # Initialize adsorption parameters with physically relevant but random values
        for i, s in enumerate(species_indices):
            # Pre-exponential factors for adsorption
            mean_a = 0.01 * (i+1)
            std_a = 0.005
            init_value_a = mean_a + std_a * torch.randn(1)
            setattr(self, f'x_A_{s}', nn.Parameter(init_value_a))
            
            if i % 3 == 0:
                mean_h = -1.0
                std_h = 0.8
            elif i % 3 == 1:
                mean_h = 1.0
                std_h = 0.8
            else:
                mean_h = 0.0
                std_h = 1.2
            
            init_value_h = mean_h + std_h * torch.randn(1)
            # Clamp to reasonable range for sigmoid input
            init_value_h = torch.clamp(init_value_h, -4.0, 4.0)
            setattr(self, f'z_Delta_H_{s}', nn.Parameter(init_value_h))
        
        self.to(self.device)

    @property
    def phase_model(self):
        """
        Access the phase model.
        
        Note: The phase model is NOT registered as a submodule (using object.__setattr__)
        to ensure its weights are NOT included in PINN's state_dict. This prevents
        stale phase model weights from being saved/loaded with PINN checkpoints.
        The phase model should always load fresh from its own checkpoint files.
        """
        return self._phase_model

    def to(self, device):
        """Override to() method to properly move all components including phase model to device"""
        super().to(device)
        
        # Update the device attribute
        self.device = device
        
        # Move the phase model to the device (using _phase_model since it's not a registered submodule)
        if hasattr(self, '_phase_model'):
            object.__setattr__(self, '_phase_model', self._phase_model.to(device))
            
        return self

    def preprocess_initial_inputs(self, f):
        """
        Preprocess raw inputs to generate time, environment parameters, and initial concentrations
        
        Args:
            f: Raw input features [batch_size, 9] with order:
               [t, T, P, fh2, fs, fa, fb, fc, fd]
        
        Returns:
            x_initial: tensor with processed inputs for PINN
            f_phase: total flow rates from phase model
            env_dict: dictionary with environment parameters
        """
        # Ensure input tensor is on the same device as the model
        f = f.to(self.device)
        
        # -----Extract basic features following the standardized order-----
        t = f[:, 0].clone()
        T = f[:, 1]
        P = f[:, 2]
        FH2 = f[:, 3]
        FS = f[:, 4]
        
        # -----Run phase model on initial conditions-----
        x_phase_mole_fractions, f_phase = self.calc_phase(step_f=f[:, :9])

        # -----Generate initial conditions for PINN and environment parameters-----
        ya_mf = x_phase_mole_fractions[:, 3]
        yb_mf = x_phase_mole_fractions[:, 4]
        yc_mf = x_phase_mole_fractions[:, 5]
        yd_mf = x_phase_mole_fractions[:, 6]
        xa_mf = x_phase_mole_fractions[:, 9]
        xb_mf = x_phase_mole_fractions[:, 10]
        xc_mf = x_phase_mole_fractions[:, 11]
        xd_mf = x_phase_mole_fractions[:, 12]
        
        VF = x_phase_mole_fractions[:, 0]
        alpha = torch.sigmoid((VF - 0.99) / 0.001)
        XA = (1 - alpha) * xa_mf + alpha * ya_mf
        XB = (1 - alpha) * xb_mf + alpha * yb_mf
        XC = (1 - alpha) * xc_mf + alpha * yc_mf
        XD = (1 - alpha) * xd_mf + alpha * yd_mf

        yh2_mf = x_phase_mole_fractions[:, 1]  # mole fraction for partial pressure
        ys_mf = x_phase_mole_fractions[:, 2]
        xh2_mf = x_phase_mole_fractions[:, 7]
        xs_mf = x_phase_mole_fractions[:, 8]
        
        # Compute partial pressures using mole fractions
        ph = yh2_mf * P

        x_initial = torch.stack([t, T, P, XA, XB, XC, XD, ph], dim=1)
        x_initial.requires_grad_(True)

        env_dict = {
            't': t,
            'T': T,
            'P': P,
            'VF': VF,
            'yh2': yh2_mf,
            'ys': ys_mf,
            'ya': ya_mf,
            'yb': yb_mf,
            'yc': yc_mf,
            'yd': yd_mf,
            'xh2': xh2_mf,
            'xs': xs_mf,
            'xa': xa_mf,
            'xb': xb_mf,
            'xc': xc_mf,
            'xd': xd_mf,
        }

        return x_initial, f_phase, env_dict
    
    def calc_phase(self, step_f=None, debug=False):
        """
        Calculate phase and concentrations with adaptive gradient flow

        Args:
            step_f: tensor with step input [batch_size, 9] with order:
                    [time, T, P, fh2, fs, fa, fb, fc, fd]
            debug: whether to print debug information
        Returns:
            x_out_phase: tensor with X phase predictions (mole fractions) [batch_size, 13]
                        [VF, yh2, ys, ya, yb, yc, yd, xh2, xs, xa, xb, xc, xd]
            f_out_phase: tensor with F concentrations (total flow rates) [batch_size, 6]
                        [fh2, fs, fa, fb, fc, fd]
        """
        assert step_f.dim() == 2, f"Expected 2D input, got {step_f.dim()}D"
        assert step_f.shape[1] == 9, f"Expected 9 features, got {step_f.shape[1]}"

        # Ensure input tensor is on the same device as the model
        step_f = step_f.to(self.device)
        step_f_safe = step_f.clone()

        # -----Prepare input for Phase model-----
        T = step_f_safe[:, 1]
        P_bar = step_f_safe[:, 2]
        FH2 = step_f_safe[:, 3]
        FS = step_f_safe[:, 4]
        FA_in = step_f_safe[:, 5]
        FB_in = step_f_safe[:, 6]
        FC_in = step_f_safe[:, 7]
        FD_in = step_f_safe[:, 8]

        # -----Convert pressure from bar to Pascals-----
        P_pascals = P_bar * 1e5

        # -----Stack inputs for the phase model-----
        f_in_phase = torch.stack([T, P_pascals, FH2, FS, FA_in, FB_in, FC_in, FD_in], dim=1)

        # -----Get phase model predictions with gradient flow-----
        _, x_out_phase, f_out_phase = self.phase_model(f_in_phase)

        # -----Check for NaN values-----
        if torch.isnan(x_out_phase).any():
            raise RuntimeError("NaN detected in mole fraction computation")
        
        if torch.isnan(f_out_phase).any():
            raise RuntimeError("NaN detected in total flow rate computation")

        if debug:
            logger.debug("Using direct gradient flow through phase model")

        # Return  mole fractions and total flow rates
        return x_out_phase, f_out_phase

    def forward(self, f_initial=None, debug=False, max_target_time_override=None):
        """
        Forward pass that predicts flow rates at regular time intervals

        Args:
            f_initial: Input tensor [batch_size, 9] containing:
                        [t, T, P, FH0, FS, FA_in, FB_in, FC_in, FD_in]
            debug: Whether to print debug information
            max_target_time_override: If set, the time-axis iteration extends to this
                value instead of being determined by f_initial[:, 0].max(). Lets callers
                keep per-sample target times in column 0 while still tracing trajectories
                past those times — used by extend_finetune to extend the trajectory to a
                shared horizon without rewriting per-sample t_x.

        Returns:
            f_pred: Predicted flow rates [batch_size, n_time_points, 9] containing:
                    [FA_out, FB_out, FC_out, FD_out, FH0, FS, FA_in, T, P]
                    - FA_out, FB_out, FC_out, FD_out: Predicted output flow rates
                    - FH0, FS, FA_in: Constant input values from f_initial
                    - T, P: Constant input values from f_initial
            time_points: Tensor of time points [n_time_points]
        """
        batch_size = f_initial.shape[0]

        # -----Extract sample-specific features once-----
        sample_features = torch.cat([
            f_initial[:, 1:2],   # T
            f_initial[:, 2:3],   # P
            f_initial[:, 3:4],   # m
            f_initial[:, 4:5],   # FH0
            f_initial[:, 5:6],   # FS
            f_initial[:, 6:7],   # FA_in
            f_initial[:, 7:8],   # FB_in
            f_initial[:, 8:9],   # FC_in
            f_initial[:, 9:10],  # FD_in
        ], dim=1)

        # -----Encode sample features once-----
        sample_encoding = self.sample_encoder(sample_features)  # [batch_size, encoding_dim]
        if f_initial.shape[0] > 1:
            while True:
                idx1, idx2 = torch.randint(0, f_initial.shape[0], (2,))
                if idx1 != idx2:
                    break
            diff = torch.mean(torch.abs(sample_encoding[idx1] - sample_encoding[idx2]))
            if diff.item() < 1e-6:
                logger.warning("WARNING: Sample encodings are nearly identical!")
        
       # -----Get target times for each sample-----
        target_times = f_initial[:, 0]

        # -----Calculate number of time steps needed for each sample-----
        if max_target_time_override is not None:
            max_steps = int(math.ceil(float(max_target_time_override) / self.delta_t))
        else:
            n_steps = torch.ceil(target_times / self.delta_t).long()
            full_max_steps = n_steps.max().item()
            max_steps = full_max_steps
        
        # -----Process all time points (including time 0)-----
        time_batches_results = []
        
        # -----Process time points in batches to reduce memory usage-----
        time_batch_size = self.time_batch_size
        
        for t_start in range(0, max_steps + 1, time_batch_size):  # Start from 0 to include initial time
            t_end = min(t_start + time_batch_size, max_steps + 1)
            
            # Process each time point individually for clarity
            for t_idx in range(t_start, t_end):
                # Create independent time tensor for each sample (raw time values)
                t_relative = torch.ones(batch_size, dtype=torch.float64, device=self.device) * (t_idx * self.delta_t)
                t_batch_tensor = t_relative.unsqueeze(1)
                t_batch_tensor.requires_grad_(True)

                # -----Encode time using logarithmic scaling-----
                time_encoding = self.time_encoder(t_batch_tensor)  # [batch_size, time_encoder_dim]
                
                if self.pinn_inputs == 'time':
                    species_outputs = self.net(t_batch_tensor)
                elif self.pinn_inputs == 'time+initials':
                    combined_features = self.feature_fusion(time_encoding, sample_encoding)
                    species_outputs = self.net(combined_features)

                time_batches_results.append(species_outputs)
        
        # Stack results to create final tensor [batch_size, n_time_points, 4] for [FA_out, FB_out, FC_out, FD_out]
        f_pred_species = torch.stack(time_batches_results, dim=1)

        # Apply per-species output scaling to match flow rate magnitudes
        f_pred_species = f_pred_species * self.output_scales.to(f_pred_species.device)

        # Create time points tensor
        time_points = torch.linspace(0, max_steps * self.delta_t, max_steps + 1, dtype=torch.float64, device=self.device)
        
        # -----Add FH0, FS, FA_in, T, P to the output-----
        T = f_initial[:, 1:2]    # [batch_size, 1]
        P = f_initial[:, 2:3]    # [batch_size, 1]
        FH0 = f_initial[:, 4:5]  # [batch_size, 1]
        FS = f_initial[:, 5:6]   # [batch_size, 1]
        FA_in = f_initial[:, 6:7]  # [batch_size, 1]
        
        n_time_points = f_pred_species.shape[1]
        T_expanded = T.unsqueeze(1).expand(-1, n_time_points, -1)
        P_expanded = P.unsqueeze(1).expand(-1, n_time_points, -1)
        FH0_expanded = FH0.unsqueeze(1).expand(-1, n_time_points, -1)
        FS_expanded = FS.unsqueeze(1).expand(-1, n_time_points, -1)
        FA_in_expanded = FA_in.unsqueeze(1).expand(-1, n_time_points, -1)
        
        f_pred_final = torch.cat([f_pred_species, FH0_expanded, FS_expanded, FA_in_expanded, T_expanded, P_expanded], dim=2)
        
        return f_pred_final, time_points
    
    def get_phase_transformed_x(self, f_pred=None, time_points=None):
        """
        Get phase-transformed concentrations and recalculated flow rates using vectorized operations
        
        Args:
            f_pred: Predicted flow rates [batch_size, n_time_points, 9] containing:
                    [FA_out, FB_out, FC_out, FD_out, FH0, FS, FA_in, T, P]
            time_points: Tensor of time points [n_time_points]
        Returns:
            x_transformed: Predicted concentrations [batch_size, n_time_points, 7] containing:
                            [T, P, XA, XB, XC, XD, ph]
                            Where XA, XB, XC, XD are conditioned on liquid phase existence
                            (liquid mole fraction if non-zero, otherwise vapor mole fraction)
            f_recalculated: Recalculated flow rates [batch_size, n_time_points, 6] containing:
                            [fh2, fs, fa, fb, fc, fd] calculated from phase-transformed mole fractions
        """
        batch_size = f_pred.shape[0]
        n_time_points = len(time_points)
        
        F_values_reshaped = f_pred.reshape(batch_size * n_time_points, 9)
        
        FH2_out = F_values_reshaped[:, 4] - 2.0 * F_values_reshaped[:, 1] - 2.0 * F_values_reshaped[:, 2] - 5.0 * F_values_reshaped[:, 3]
        
        time_values_reshaped = time_points.unsqueeze(0).expand(batch_size, -1).reshape(batch_size * n_time_points)
        
        step_f_all = torch.zeros((batch_size * n_time_points, 9), dtype=torch.float64, device=self.device)

        step_f_all[:, 0] = time_values_reshaped     # time
        step_f_all[:, 1] = F_values_reshaped[:, 7]  # T (from f_pred index 7)
        step_f_all[:, 2] = F_values_reshaped[:, 8]  # P (from f_pred index 8)
        step_f_all[:, 3] = FH2_out                  # FH2_out (calculated)
        step_f_all[:, 4] = F_values_reshaped[:, 5]  # FS (from f_pred index 5)
        step_f_all[:, 5] = F_values_reshaped[:, 0]  # FA_out (from f_pred index 0)
        step_f_all[:, 6] = F_values_reshaped[:, 1]  # FB_out (from f_pred index 1)
        step_f_all[:, 7] = F_values_reshaped[:, 2]  # FC_out (from f_pred index 2)
        step_f_all[:, 8] = F_values_reshaped[:, 3]  # FD_out (from f_pred index 3)
        
        x_phase_mole_fractions_all, _ = self.calc_phase(step_f=step_f_all)
        
        # -----Apply similar logic as preprocess_initial_inputs-----
        # Extract vapor and liquid mole fractions
        ya_mf = x_phase_mole_fractions_all[:, 3]
        yb_mf = x_phase_mole_fractions_all[:, 4]
        yc_mf = x_phase_mole_fractions_all[:, 5]
        yd_mf = x_phase_mole_fractions_all[:, 6]
        xa_mf = x_phase_mole_fractions_all[:, 9]
        xb_mf = x_phase_mole_fractions_all[:, 10]
        xc_mf = x_phase_mole_fractions_all[:, 11]
        xd_mf = x_phase_mole_fractions_all[:, 12]
        
        # Smooth blending between liquid and vapor mole fractions based on VF
        VF_blend = x_phase_mole_fractions_all[:, 0]
        alpha = torch.sigmoid((VF_blend - 0.99) / 0.001)
        XA = (1 - alpha) * xa_mf + alpha * ya_mf
        XB = (1 - alpha) * xb_mf + alpha * yb_mf
        XC = (1 - alpha) * xc_mf + alpha * yc_mf
        XD = (1 - alpha) * xd_mf + alpha * yd_mf
        
        # Extract T and P from F_values_reshaped
        T = F_values_reshaped[:, 7]
        P = F_values_reshaped[:, 8]
        
        # Extract mole fractions for partial pressure calculations
        yh2_mf = x_phase_mole_fractions_all[:, 1]
        
        # Compute partial pressures using mole fractions
        ph = yh2_mf * P
        
        x_transformed_all = torch.stack([T, P, XA, XB, XC, XD, ph], dim=1)
        
        x_transformed = x_transformed_all.reshape(batch_size, n_time_points, 7)
        
        # -----Calculate flow rates from mole fractions (vectorized approach)-----
        flow_rate_sum = torch.sum(step_f_all[:, 3:9], dim=1, keepdim=True)  # Shape: [batch_size * n_time_points, 1]
        
        VF = x_phase_mole_fractions_all[:, 0:1]
        
        vapor_mole_fractions = x_phase_mole_fractions_all[:, 1:7]
        
        liquid_mole_fractions = x_phase_mole_fractions_all[:, 7:13]
        
        vapor_flowrates = vapor_mole_fractions * VF * flow_rate_sum
        liquid_flowrates = liquid_mole_fractions * (1 - VF) * flow_rate_sum
        total_flowrates_all = liquid_flowrates + vapor_flowrates  # Shape: [batch_size * n_time_points, 6]
        
        # Reshape back to [batch_size, n_time_points, 6]
        f_recalculated = total_flowrates_all.reshape(batch_size, n_time_points, 6)

        # Reshape phase mole fractions to [batch_size, n_time_points, 13]
        # Structure: [VF, yh2, ys, ya, yb, yc, yd, xh2, xs, xa, xb, xc, xd]
        x_phase_all = x_phase_mole_fractions_all.reshape(batch_size, n_time_points, 13)

        return x_transformed, f_recalculated, x_phase_all

    def get_predictions(self, f_pred=None, x_transformed=None, f_initial=None, time_points=None):
        """
        Get predictions for both f_pred and x_pred based on target time in f_initial
        
        Args:
            f_pred: Predicted flow rates [batch_size, n_time_points, 9] containing:
                    [FA_out, FB_out, FC_out, FD_out, FH0, FS, FA_in, T, P]
            x_transformed: Predicted concentrations [batch_size, n_time_points, 7] containing:
                    [T, P, XA, XB, XC, XD, ph]
            f_initial: Input initial flow rates tensor [batch_size, 9] containing:
                        [t, T, P, FH0, FS, FA_in, FB_in, FC_in, FD_in]
            time_points: Tensor of time points [n_time_points]
        Returns:
            f_pred_at_target: Predicted flow rates at target times [batch_size, 9] containing:
                                [FA_out, FB_out, FC_out, FD_out, FH0, FS, FA_in, T, P]
            x_pred_at_target: Predicted concentrations at target times [batch_size, 7] containing:
                                [T, P, XA, XB, XC, XD, ph]
        """
        batch_size = x_transformed.shape[0]

        # -----Extract target times from f_initial-----
        target_times = f_initial[:, 0]
        
        # -----Initialize output tensors-----
        f_pred_at_target = torch.zeros((batch_size, 9), dtype=torch.float64, device=self.device)
        x_pred_at_target = torch.zeros((batch_size, 7), dtype=torch.float64, device=self.device)
        
        # -----Process each sample independently-----
        for i in range(batch_size):
            # Get target time for this sample
            t_target = target_times[i]
            
            # Find the closest time point that's not beyond target time
            time_diffs = torch.abs(time_points - t_target)
            closest_idx = torch.argmin(time_diffs).item()

            # If closest time is beyond target, take the previous time point
            if time_points[closest_idx] > t_target and closest_idx > 0:
                closest_idx -= 1
            
            # Get flow rate predictions at the closest valid time point
            f_pred_at_target[i] = f_pred[i, closest_idx]
            
            # Get concentration predictions at the closest valid time point
            x_pred_at_target[i] = x_transformed[i, closest_idx]
        
        return f_pred_at_target, x_pred_at_target

    def get_parameters(self):
        """
        Return all learnable reaction parameters.

        Returns tuple of:
        - x_A_1, x_A_2, x_A_3, x_A_4: Pre-exponential factors for reactions 1, 2, 3, 4
        - x_A_A, x_A_B, x_A_C, x_A_D, x_A_H: Pre-exponential factors for adsorption
        - y_E_1, y_E_2, y_E_3, y_E_4: Activation energies for reactions 1, 2, 3, 4
        - z_Delta_H_A, z_Delta_H_B, z_Delta_H_C, z_Delta_H_D, z_Delta_H_H: Adsorption enthalpies
        """
        return (
            self.x_A_1.item(), self.x_A_2.item(), self.x_A_3.item(), self.x_A_4.item(),
            self.x_A_A.item(), self.x_A_B.item(), self.x_A_C.item(), self.x_A_D.item(), self.x_A_H.item(),
            self.y_E_1.item(), self.y_E_2.item(), self.y_E_3.item(), self.y_E_4.item(),
            self.z_Delta_H_A.item(), self.z_Delta_H_B.item(), self.z_Delta_H_C.item(), self.z_Delta_H_D.item(), self.z_Delta_H_H.item()
        )

    def save_model(self, save_path):
        """
        Save PINN model with all necessary information
        
        Args:
            save_path: path to save model file
        """
        try:
            # Get the directory path
            save_dir = os.path.dirname(save_path)
            
            # Create directory if it doesn't exist
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                logger.info(f"Created directory: {save_dir}")
            
            # Save the model
            model_save = {
                'state_dict': self.state_dict(),
                'config': self.config
            }
            torch.save(model_save, save_path)
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load_model(cls, model_path):
        """
        Load saved PINN model

        Args:
            model_path: path to saved model file
        Returns:
            loaded PINN model
        """
        # -----Determine the appropriate device-----
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -----Load the checkpoint with device mapping-----
        checkpoint = torch.load(model_path, map_location=map_location)

        # Get the device object if string was provided
        device = map_location if isinstance(map_location, torch.device) else torch.device(map_location)

        # -----Handle config key compatibility-----
        config = checkpoint['config']

        # Support both old 'num_inputs' and new 'num_sample_features' keys
        if 'num_sample_features' in config:
            num_sample_features = config['num_sample_features']
        else:
            raise KeyError("Config must contain 'num_sample_features'")

        # -----Initialize model with saved config-----
        if 'pinn_inputs' in config.keys():
            model = cls(
                num_sample_features=num_sample_features,
                num_outputs=config['num_outputs'],
                delta_t=config['delta_t'],
                config_phase_model=config['config_phase_model'],
                time_batch_size=config['time_batch_size'],
                run_mode=config['run_mode'],
                pinn_inputs=config['pinn_inputs'],
                training_flag=config['training_flag'],
                output_scale=config['output_scale'],
                device=device
            )
        else:
            model = cls(
                num_sample_features=num_sample_features,
                num_outputs=config['num_outputs'],
                delta_t=config['delta_t'],
                config_phase_model=config['config_phase_model'],
                time_batch_size=config['time_batch_size'],
                run_mode=config['run_mode'],
                pinn_inputs='time',
                training_flag=False,
                output_scale=config['output_scale'],
                device=device
            )

        # -----Load state dict-----
        model.load_state_dict(checkpoint['state_dict'])

        # -----Set to evaluation mode-----
        model.eval()

        logger.info(f"Model loaded from {model_path} to device: {device}")
        return model
    
    def inference(self, input_features=None, output_subfolder=None, debug=False, sample_ids=None):
        """
        Run model inference and create plots for flow rate and mole fraction changes over time
        
        Args:
            input_features: Tensor with input features [batch_size, 9] containing:
                            [t, T, P, fh2, fs, fa, fb, fc, fd]
                            First column contains target time for each sample
            output_subfolder: Name of subfolder to save plots (under OUTPUT_DIR)
            debug: Boolean flag to enable debug logging
            sample_ids: Optional list/array of sample IDs from input data. If provided,
                        these IDs will be used for naming trajectory plot files instead of
                        sequential batch indices.
            
        Returns:
            f_pred_at_target: Predicted flow rates at target times [batch_size, 6]
                                [fh2, fs, fa, fb, fc, fd]
            x_pred_at_target: Predicted mole fractions at target times [batch_size, 13]
                                [VF, yh2, ys, ya, yb, yc, yd, xh2, xs, xa, xb, xc, xd]
            all_predictions: Dictionary mapping sample index to predictions at all time points
        """
        self.training_flag = False
        self.eval()

        # -----Move input to device-----
        input_features = input_features.to(self.device)

        if output_subfolder is not None:
            logger.info(f"output_subfolder: {output_subfolder}")
            plot_dir = os.path.join(OUTPUT_DIR, output_subfolder)
            os.makedirs(plot_dir, exist_ok=True)

        with torch.no_grad():
            batch_size = input_features.shape[0]

            # -----Run PINN-----
            f_pred, time_points = self.forward(f_initial=input_features)
            x_transformed, _, x_phase_all = self.get_phase_transformed_x(f_pred=f_pred, time_points=time_points)
            f_pred_at_target, x_pred_at_target = self.get_predictions(f_pred=f_pred, x_transformed=x_transformed, f_initial=input_features, 
                                                                       time_points=time_points)
            
            if debug:
                logger.debug(f"f_pred shape: {f_pred.shape}")
                logger.debug(f"f_pred initial: {f_pred[:, 0, :].detach().cpu().numpy()}")
                logger.debug(f"x_transformed shape: {x_transformed.shape}")
                logger.debug(f"x_transformed initial: {x_transformed[:, 0, :].detach().cpu().numpy()}")
            
            # -----Store predictions for each sample at all time points-----
            all_predictions = {}
            
            # -----Process each sample for visualization-----
            for batch_idx in range(batch_size):
                # Create full state tensor for this sample
                # f_pred structure: [FA_out, FB_out, FC_out, FD_out, FH0, FS, FA_in, T, P]
                # We'll construct: [t, T, P, FH2_out, FS, FA_out, FB_out, FC_out, FD_out]
                full_state = torch.zeros((len(time_points), 9), dtype=torch.float64, device=self.device)

                # Set time, T, P from f_pred
                full_state[:, 0] = time_points
                full_state[:, 1] = f_pred[batch_idx, :, 7]  # T (from f_pred index 7)
                full_state[:, 2] = f_pred[batch_idx, :, 8]  # P (from f_pred index 8)
                
                # Calculate FH2_out: FH2_out = FH0 - 2*FB_out - 2*FC_out - 5*FD_out
                FH2_out = f_pred[batch_idx, :, 4] - 2.0 * f_pred[batch_idx, :, 1] - 2.0 * f_pred[batch_idx, :, 2] - 5.0 * f_pred[batch_idx, :, 3]
                full_state[:, 3] = FH2_out
                
                # Set FS, FA_out, FB_out, FC_out, FD_out
                full_state[:, 4] = f_pred[batch_idx, :, 5]  # FS
                full_state[:, 5] = f_pred[batch_idx, :, 0]  # FA_out
                full_state[:, 6] = f_pred[batch_idx, :, 1]  # FB_out
                full_state[:, 7] = f_pred[batch_idx, :, 2]  # FC_out
                full_state[:, 8] = f_pred[batch_idx, :, 3]  # FD_out

                # Store all predictions for this sample
                all_predictions[batch_idx] = {
                    'times': time_points,
                    'flow_predictions': f_pred[batch_idx],
                    'mole_fraction_predictions': x_transformed[batch_idx]
                }

                # Create visualization if output folder is provided
                if output_subfolder is not None:
                    # Use sample_id from input data if provided, otherwise use batch_idx
                    sample_id = sample_ids[batch_idx] if sample_ids is not None else batch_idx
                    plot_trajectory(
                        batch_idx=batch_idx,
                        times=time_points,
                        predictions=full_state,
                        c_predictions=x_transformed[batch_idx],
                        phase_predictions=x_phase_all[batch_idx],
                        plot_dir=plot_dir,
                        sample=input_features[batch_idx:batch_idx+1],
                        env_pred=None,
                        run_mode=self.run_mode,
                        system='mn',
                        sample_id=sample_id
                    )

                    if debug:
                        logger.debug(f"Processed batch {batch_idx} (sample_id={sample_id}):")
                        logger.debug(f"Target time: {input_features[batch_idx, 0]:.3f}")
                        logger.debug(f"Number of time points: {len(time_points)}")

        return f_pred_at_target, x_pred_at_target, all_predictions

    def extend_inference(self, input_features, extend_target_time, original_target_times=None,
                         sample_ids=None, splits=None, ground_truth_outputs=None,
                         output_dir=None, system='mn'):
        """
        Run extended inference: integrate all samples to the same extended time.
        Produces per-sample trajectory plots and an overlay plot.

        Column 0 of input_features must carry the ORIGINAL per-sample t_x; the
        time-axis extension to extend_target_time is driven by forward()'s
        max_target_time_override.

        Args:
            input_features: [batch_size, 10] sample features.
                            MN format: [t_x, T, P, m, FH0, FS, FA_in, FB_in, FC_in, FD_in].
                            Column 0 is the per-sample original t_x.
            extend_target_time: Shared extension horizon. The forward pass extends
                                the trajectory to this time via max_target_time_override.
            original_target_times: [batch_size] original experimental target times per sample
            sample_ids: list of sample IDs for file naming
            splits: list of 'train'/'test' per sample
            ground_truth_outputs: [batch_size, 9] original output_targets
                                  MN format: [t_x, T, P, FH2_out, FS, FA_out, FB_out, FC_out, FD_out]
            output_dir: directory to save all outputs
            system: 'mn'

        Returns:
            all_predictions: dict mapping sample index to trajectory data
        """
        self.training_flag = False
        self.eval()

        input_features = input_features.to(self.device)

        plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        logger.info(f"Running extend inference: {input_features.shape[0]} samples to t={extend_target_time:.4f}")
        logger.info(f"Model delta_t: {self.delta_t}, estimated steps: {int(extend_target_time / self.delta_t)}")

        with torch.no_grad():
            batch_size = input_features.shape[0]

            f_pred, time_points = self.forward(
                f_initial=input_features,
                max_target_time_override=extend_target_time,
            )
            x_transformed, _, x_phase_all = self.get_phase_transformed_x(f_pred=f_pred, time_points=time_points)

            logger.info(f"Forward pass complete: f_pred shape {f_pred.shape}, {len(time_points)} time points")

            all_predictions = {}
            sample_metadata = []

            for batch_idx in range(batch_size):
                full_state = torch.zeros((len(time_points), 9), dtype=torch.float64, device=self.device)
                full_state[:, 0] = time_points
                full_state[:, 1] = f_pred[batch_idx, :, 7]  # T
                full_state[:, 2] = f_pred[batch_idx, :, 8]  # P
                FH2_out = f_pred[batch_idx, :, 4] - 2.0 * f_pred[batch_idx, :, 1] - 2.0 * f_pred[batch_idx, :, 2] - 5.0 * f_pred[batch_idx, :, 3]
                full_state[:, 3] = FH2_out
                full_state[:, 4] = f_pred[batch_idx, :, 5]  # FS
                full_state[:, 5] = f_pred[batch_idx, :, 0]  # FA_out
                full_state[:, 6] = f_pred[batch_idx, :, 1]  # FB_out
                full_state[:, 7] = f_pred[batch_idx, :, 2]  # FC_out
                full_state[:, 8] = f_pred[batch_idx, :, 3]  # FD_out

                all_predictions[batch_idx] = {
                    'times': time_points,
                    'flow_predictions': f_pred[batch_idx],
                    'mole_fraction_predictions': x_transformed[batch_idx]
                }

                # Build ground truth dict for this sample
                orig_t = original_target_times[batch_idx].item() if original_target_times is not None else None
                gt = None
                if ground_truth_outputs is not None and orig_t is not None:
                    gt = {
                        'FA_out': ground_truth_outputs[batch_idx, 5].item(),
                        'FB_out': ground_truth_outputs[batch_idx, 6].item(),
                        'FC_out': ground_truth_outputs[batch_idx, 7].item(),
                        'FD_out': ground_truth_outputs[batch_idx, 8].item(),
                    }

                # Build metadata for overlay
                T_val = input_features[batch_idx, 1].item()
                P_val = input_features[batch_idx, 2].item()
                FA_in_val = input_features[batch_idx, 6].item()  # MN: FA_in at index 6
                sample_id = sample_ids[batch_idx] if sample_ids is not None else batch_idx
                split = splits[batch_idx] if splits is not None else 'unknown'

                sample_metadata.append({
                    'T': T_val, 'P': P_val, 'FA_in': FA_in_val,
                    'original_t_x': orig_t,
                    'split': split,
                    'sample_id': sample_id,
                    'ground_truth': gt,
                })

                plot_trajectory(
                    batch_idx=batch_idx,
                    times=time_points,
                    predictions=full_state,
                    c_predictions=x_transformed[batch_idx],
                    phase_predictions=x_phase_all[batch_idx],
                    plot_dir=plot_dir,
                    sample=input_features[batch_idx:batch_idx+1],
                    env_pred=None,
                    run_mode=self.run_mode,
                    system=system,
                    sample_id=sample_id,
                    original_target_time=orig_t,
                    ground_truth=gt,
                    extend_target_time=extend_target_time,
                )

            # Overlay plot
            plot_extended_overlay(
                all_trajectories=all_predictions,
                sample_metadata=sample_metadata,
                extend_target_time=extend_target_time,
                plot_dir=output_dir,
                system=system,
            )

        return all_predictions