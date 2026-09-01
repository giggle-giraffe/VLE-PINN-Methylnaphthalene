# ------------------------------------------------------------------------------------------
# Copyright (c) 2026 Chen Zhang, Tao Li
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

# Set process title for easy identification
try:
    import setproctitle
    setproctitle.setproctitle("thermo_gen_synthetics")
except ImportError:
    pass

import pandas as pd
import numpy as np
import yaml
import os
import sys
from typing import Optional, Tuple, List, Iterator
from datetime import datetime
import time
import itertools
import gc

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Use standard logging if loguru is not available
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

try:
    from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, PRMIX, FlashVL
    from thermo.interaction_parameters import IPDB
    THERMO_AVAILABLE = True
except ImportError:
    logger.warning("Thermo package not available. Synthetic data generation will use mock data.")
    THERMO_AVAILABLE = False


# Setup paths
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")


def get_memory_usage():
    """Get current memory usage in MB"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    else:
        return 0.0  # Return 0 if psutil is not available


class OptimizedThermoGenerator:
    """
    Memory-optimized thermodynamics data generator for synthetic training data
    """

    def __init__(self, chemicals: List[str], component_names: List[str], use_real_thermo: bool = True):
        """
        Initialize the thermodynamics generator

        Args:
            chemicals: List of CAS numbers for the chemical components
            component_names: List of component names (e.g., ['fh2', 'fs', 'fa', ...])
            use_real_thermo: Whether to use real thermodynamics calculations or mock data
        """
        self.chemicals = chemicals
        self.component_names = component_names
        self.num_components = len(component_names)
        self.num_dimensions = 2 + self.num_components  # temp, pres, + flow rates
        self.use_real_thermo = use_real_thermo and THERMO_AVAILABLE
        self.flasher = None

        logger.info(f"Initialized with {self.num_components} components: {component_names}")
        logger.info(f"Grid dimensions: {self.num_dimensions} (temp, pres, + {self.num_components} flow rates)")

        if self.use_real_thermo:
            self._initialize_thermo_engine()
        else:
            logger.info("Using mock thermodynamics data generation")
    
    def _initialize_thermo_engine(self):
        """Initialize the thermodynamics calculation engine"""
        try:
            # Load chemical constants and properties
            constants, properties = ChemicalConstantsPackage.from_IDs(self.chemicals)
            kijs = IPDB.get_ip_asymmetric_matrix('ChemSep PR', constants.CASs, 'kij')
            
            eos_kwargs = {
                'Pcs': constants.Pcs, 
                'Tcs': constants.Tcs, 
                'omegas': constants.omegas, 
                'kijs': kijs
            }
            
            gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
            liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
            
            self.flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)
            
        except Exception as e:
            logger.error(f"Failed to initialize thermodynamics engine: {e}")
            logger.info("Falling back to mock data generation")
            self.use_real_thermo = False
    
    def generate_grid_indices_iterator(self, grid_points: int,
                                      batch_size: int = 10000) -> Iterator[np.ndarray]:
        """
        Generate grid indices in batches using itertools to avoid memory issues

        Args:
            grid_points: Number of points per dimension
            batch_size: Number of samples per batch

        Yields:
            Batch of grid indices (shape: [batch_size, num_dimensions])
        """
        # Create ranges for each dimension (temp, pres, + component flow rates)
        ranges = [range(grid_points) for _ in range(self.num_dimensions)]

        # Use itertools.product to generate combinations without storing all in memory
        batch = []
        for idx, combo in enumerate(itertools.product(*ranges)):
            batch.append(combo)

            # Yield batch when it reaches the desired size
            if len(batch) >= batch_size:
                yield np.array(batch)
                batch = []

        # Yield remaining items
        if batch:
            yield np.array(batch)
    
    def _create_piecewise_grid(self, min_val: float, max_val: float, 
                               breakpoints: List[float], grid_points: int) -> np.ndarray:
        """
        Create a piecewise uniform grid that distributes points equally across segments.
        
        For example, with min=1e-6, max=1e-4, breakpoints=[1e-5], grid_points=10:
        - 5 points uniformly in [1e-6, 1e-5]
        - 5 points uniformly in [1e-5, 1e-4]
        
        Args:
            min_val: Minimum value
            max_val: Maximum value
            breakpoints: List of breakpoint values (must be between min and max)
            grid_points: Total number of grid points
            
        Returns:
            Array of grid values with piecewise uniform distribution
        """
        # Build segment boundaries: [min, bp1, bp2, ..., max]
        boundaries = [min_val] + sorted(breakpoints) + [max_val]
        n_segments = len(boundaries) - 1
        
        # Distribute points equally among segments
        # If grid_points doesn't divide evenly, extra points go to later segments
        base_points_per_segment = grid_points // n_segments
        extra_points = grid_points % n_segments
        
        all_points = []
        for i in range(n_segments):
            seg_min = boundaries[i]
            seg_max = boundaries[i + 1]
            
            # Calculate points for this segment
            # Give extra points to later segments (which typically have larger ranges)
            seg_points = base_points_per_segment + (1 if i >= n_segments - extra_points else 0)
            
            if seg_points > 0:
                # Generate points for this segment
                # For first segment, include min; for last segment, include max
                # For middle segments, exclude the start to avoid duplicates
                if i == 0:
                    segment_grid = np.linspace(seg_min, seg_max, seg_points + 1)[:-1]  # Exclude end
                elif i == n_segments - 1:
                    segment_grid = np.linspace(seg_min, seg_max, seg_points)  # Include end
                else:
                    segment_grid = np.linspace(seg_min, seg_max, seg_points + 1)[:-1]  # Exclude end
                
                all_points.extend(segment_grid)
        
        result = np.array(all_points)
        
        # Ensure we have exactly grid_points values
        if len(result) < grid_points:
            # Pad with max value if needed
            result = np.concatenate([result, np.full(grid_points - len(result), max_val)])
        elif len(result) > grid_points:
            # Truncate if too many (shouldn't happen with correct logic)
            result = result[:grid_points]
            
        return np.sort(result)

    def generate_batch_conditions(self, indices: np.ndarray,
                                 config_ranges: dict,
                                 grid_points: int,
                                 batch_number: int) -> pd.DataFrame:
        """
        Generate conditions for a batch of grid indices

        Args:
            indices: Batch of grid indices (shape: [batch_size, num_dimensions])
            config_ranges: Configuration ranges from YAML
            grid_points: Number of points per dimension
            batch_number: Current batch number for sample numbering

        Returns:
            DataFrame with conditions for this batch
        """
        batch_size = len(indices)

        # Extract ranges
        temp_min = config_ranges['temp_range']['min']
        temp_max = config_ranges['temp_range']['max']
        pres_min = config_ranges['pres_range']['min']
        pres_max = config_ranges['pres_range']['max']
        flow_ranges = config_ranges.get('flow_rate_ranges', {})

        # Create grid values for each dimension
        temp_grid = np.linspace(temp_min, temp_max, grid_points)
        pres_grid = np.linspace(pres_min, pres_max, grid_points)

        # Build flow grids dynamically based on component_names
        flow_grids = {}
        for comp in self.component_names:
            if comp in flow_ranges:
                comp_config = flow_ranges[comp]
                flow_min = comp_config['min']
                flow_max = comp_config['max']
                
                # Check sampling mode
                sampling_mode = comp_config.get('sampling_mode', 'uniform')
                
                if sampling_mode == 'piecewise':
                    # Use piecewise uniform sampling
                    breakpoints = comp_config.get('breakpoints', [])
                    if not breakpoints:
                        # If no breakpoints specified, fall back to uniform
                        logger.warning(f"Piecewise sampling for '{comp}' has no breakpoints, using uniform")
                        flow_grids[comp] = np.linspace(flow_min, flow_max, grid_points)
                    else:
                        flow_grids[comp] = self._create_piecewise_grid(
                            flow_min, flow_max, breakpoints, grid_points
                        )
                        # logger.debug(f"Created piecewise grid for '{comp}': {flow_grids[comp]}")
                else:
                    # Default: uniform sampling
                    flow_grids[comp] = np.linspace(flow_min, flow_max, grid_points)
            else:
                raise ValueError(f"Component '{comp}' not found in flow_rate_ranges configuration.")

        # Map indices to actual values
        temp = temp_grid[indices[:, 0]]
        pres = pres_grid[indices[:, 1]]

        # Extract raw flow rates for each component dynamically
        flow_raw = {}
        for i, comp in enumerate(self.component_names):
            flow_raw[comp] = flow_grids[comp][indices[:, 2 + i]]

        # Calculate total flow for normalization
        total_flow = np.sum([flow_raw[comp] for comp in self.component_names], axis=0)

        # Calculate starting sample number for this batch
        start_no = batch_number * batch_size + 1

        # Create DataFrame with dynamic columns
        data = {
            'no': range(start_no, start_no + batch_size),
            'temp': temp,
            'pres': pres,
        }
        # Add normalized flow rates for each component
        for comp in self.component_names:
            data[comp] = flow_raw[comp] / total_flow

        conditions_df = pd.DataFrame(data)

        return conditions_df
    
    def _get_phase_column_names(self, prefix: str) -> List[str]:
        """
        Generate phase column names from component names.
        E.g., for component 'fh2' with prefix 'y', returns 'yh2'.

        Args:
            prefix: 'y' for gas phase, 'x' for liquid phase

        Returns:
            List of column names for the phase
        """
        # Convert 'fXXX' to 'yXXX' or 'xXXX' by replacing the 'f' prefix
        return [prefix + comp[1:] if comp.startswith('f') else prefix + comp
                for comp in self.component_names]

    def calculate_flash_single(self, temp: float, pres: float, composition: List[float]) -> dict:
        """
        Calculate flash equilibrium for a single condition

        Args:
            temp: Temperature in K
            pres: Pressure in Pa
            composition: Normalized composition list (same order as component_names)

        Returns:
            Dictionary with flash calculation results
        """
        if not self.use_real_thermo:
            return self._mock_flash_calculation(temp, pres, composition)

        try:
            # Use pressure as-is (should already be in Pa from config)
            P_pa = pres

            # Normalize composition
            total_comp = sum(composition)
            if total_comp <= 1e-12:
                raise ValueError("Zero composition")

            zs = [comp / total_comp for comp in composition]

            # Perform flash calculation
            PT = self.flasher.flash(T=temp, P=P_pa, zs=zs)

            if PT is None:
                raise ValueError("Flash calculation returned None")

            # Extract results
            result = {
                'temp': temp,
                'pres': pres,
                'VF': getattr(PT, 'VF', 1.0),
                'error': None
            }

            # Get dynamic column names for gas (y) and liquid (x) phases
            y_cols = self._get_phase_column_names('y')
            x_cols = self._get_phase_column_names('x')

            # Gas phase results
            if hasattr(PT, 'gas') and PT.gas is not None:
                gas_zs = PT.gas.zs
                for i, col in enumerate(y_cols):
                    result[col] = gas_zs[i]
            else:
                # Fallback to normalized input composition
                for i, col in enumerate(y_cols):
                    result[col] = zs[i]
                result['error'] = "Gas phase is None"

            # Liquid phase results
            if hasattr(PT, 'liquid0') and PT.liquid0 is not None:
                liquid_zs = PT.liquid0.zs
                for i, col in enumerate(x_cols):
                    result[col] = liquid_zs[i]
            else:
                # No liquid phase
                for col in x_cols:
                    result[col] = 0.0

            return result

        except Exception as e:
            logger.warning(f"Flash calculation failed for T={temp}, P={pres}: {e}")
            return self._mock_flash_calculation(temp, pres, composition)
    
    def _mock_flash_calculation(self, temp: float, pres: float, composition: List[float]) -> dict:
        """
        Generate mock flash calculation results when real thermo is not available.
        Uses simplified thermodynamic assumptions for any number of components.
        """
        n = self.num_components

        # Normalize composition
        total_comp = sum(composition)
        if total_comp <= 1e-12:
            zs = [1.0 / n] * n  # Equal distribution if zero
        else:
            zs = [comp / total_comp for comp in composition]

        # Simple mock logic based on temperature and pressure
        vapor_fraction = min(1.0, max(0.0, (temp - 400) / 250 + (80 - pres / 100000) / 100))

        # Get column names
        y_cols = self._get_phase_column_names('y')
        x_cols = self._get_phase_column_names('x')

        # Mock gas phase: first component (typically H2) concentrates in gas
        # Lighter components have higher vapor affinity
        y_values = []
        for i, z in enumerate(zs):
            # First component (H2) has highest vapor affinity, decreases for heavier components
            affinity = 1.0 + vapor_fraction * (1.0 - i * 0.1)
            y_values.append(z * affinity)

        # Normalize gas phase
        y_total = sum(y_values)
        if y_total > 0:
            y_values = [y / y_total for y in y_values]
        else:
            y_values = [1.0 / n] * n

        # Mock liquid phase (only if vapor fraction < 1)
        if vapor_fraction < 0.95:
            x_values = []
            for i, z in enumerate(zs):
                # Heavier components (higher index) concentrate in liquid
                affinity = (2.0 - vapor_fraction) * (0.5 + i * 0.1)
                x_values.append(z * affinity)

            # Normalize liquid phase
            x_total = sum(x_values)
            if x_total > 0:
                x_values = [x / x_total for x in x_values]
            else:
                x_values = [1.0 / n] * n
        else:
            x_values = [0.0] * n

        # Build result dictionary
        result = {
            'temp': temp,
            'pres': pres,
            'VF': vapor_fraction,
            'error': 'Mock calculation'
        }

        # Add gas phase columns
        for col, val in zip(y_cols, y_values):
            result[col] = val

        # Add liquid phase columns
        for col, val in zip(x_cols, x_values):
            result[col] = val

        return result
    
    def process_batch(self, conditions_df: pd.DataFrame,
                     progress_offset: int = 0) -> pd.DataFrame:
        """
        Process a batch of conditions through flash calculations

        Args:
            conditions_df: DataFrame with conditions to process
            progress_offset: Offset for progress reporting (total samples processed so far)

        Returns:
            DataFrame with flash calculation results
        """
        results = []

        for idx, row in conditions_df.iterrows():
            composition = [row[col] for col in self.component_names]
            flash_result = self.calculate_flash_single(row['temp'], row['pres'], composition)

            # Combine condition and result
            result_row = {
                'no': row['no'],
                'temp': row['temp'],
                'pres': row['pres'],
                **{col: row[col] for col in self.component_names}
            }
            result_row.update(flash_result)
            results.append(result_row)

        batch_df = pd.DataFrame(results)

        # Filter out samples with errors (for real thermo only)
        if self.use_real_thermo:
            successful_mask = batch_df['error'].isna()
            filtered_df = batch_df[successful_mask].copy()
        else:
            filtered_df = batch_df.copy()

        # Remove error column
        if 'error' in filtered_df.columns:
            filtered_df = filtered_df.drop('error', axis=1)

        # Add phase classification
        vf_threshold_high = 0.999
        filtered_df['class_true'] = 0  # Default to two-phase
        single_gas_mask = filtered_df['VF'] >= vf_threshold_high
        filtered_df.loc[single_gas_mask, 'class_true'] = 1

        return filtered_df
    
    def generate_synthetic_dataset_optimized(self,
                                            config_ranges: dict,
                                            grid_points: int,
                                            batch_size: int = 10000,
                                            output_path: str = None,
                                            append_mode: bool = False) -> None:
        """
        Generate synthetic dataset with memory-efficient batch processing

        Args:
            config_ranges: Configuration ranges from YAML file
            grid_points: Number of points per dimension
            batch_size: Number of samples to process in each batch
            output_path: Path to save the output CSV file
            append_mode: Whether to append to existing file or overwrite
        """
        total_samples = grid_points ** self.num_dimensions
        logger.info(f"Starting optimized generation: {total_samples} total samples")
        logger.info(f"Processing in batches of {batch_size} samples")
        logger.info(f"Initial memory usage: {get_memory_usage():.1f} MB")
        
        start_time = time.time()
        
        # Initialize statistics
        total_processed = 0
        total_successful = 0
        total_two_phase = 0
        total_single_gas = 0
        batch_number = 0
        
        # Prepare output file
        first_batch = True
        
        # Process batches
        for batch_indices in self.generate_grid_indices_iterator(grid_points, batch_size):
            batch_start_time = time.time()
            current_batch_size = len(batch_indices)
            
            # Generate conditions for this batch
            conditions_df = self.generate_batch_conditions(
                batch_indices, config_ranges, grid_points, batch_number
            )
            
            # Process batch through flash calculations
            results_df = self.process_batch(conditions_df, total_processed)
            
            # Update statistics
            batch_successful = len(results_df)
            batch_two_phase = (results_df['class_true'] == 0).sum()
            batch_single_gas = (results_df['class_true'] == 1).sum()
            
            total_processed += current_batch_size
            total_successful += batch_successful
            total_two_phase += batch_two_phase
            total_single_gas += batch_single_gas
            
            # Save batch to CSV
            if output_path:
                if first_batch and not append_mode:
                    results_df.to_csv(output_path, index=False, mode='w')
                else:
                    results_df.to_csv(output_path, index=False, mode='a', header=False)
                first_batch = False
            
            # Clear memory
            del conditions_df
            del results_df
            gc.collect()
            
            # Progress report
            batch_time = time.time() - batch_start_time
            elapsed_time = time.time() - start_time
            progress_pct = (total_processed / total_samples) * 100
            success_rate = (total_successful / total_processed) * 100 if total_processed > 0 else 0
            
            # ETA calculation
            if total_processed > 0:
                time_per_sample = elapsed_time / total_processed
                remaining_samples = total_samples - total_processed
                eta_seconds = remaining_samples * time_per_sample
                eta_minutes = eta_seconds / 60
                
                if eta_minutes > 1:
                    eta_str = f"ETA: {eta_minutes:.1f} min"
                else:
                    eta_str = f"ETA: {eta_seconds:.0f} sec"
            else:
                eta_str = "ETA: calculating..."
            
            logger.info(
                f"Batch {batch_number + 1} | Progress: {total_processed}/{total_samples} ({progress_pct:.1f}%) | "
                f"Success: {success_rate:.1f}% | {eta_str} | "
                f"Memory: {get_memory_usage():.1f} MB | "
                f"Batch time: {batch_time:.1f}s"
            )
            
            batch_number += 1
        
        # Final statistics
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"Generation completed in {total_time/60:.1f} minutes")
        logger.info(f"Total samples processed: {total_processed}")
        logger.info(f"Successful calculations: {total_successful} ({(total_successful/total_processed)*100:.1f}%)")
        logger.info(f"Two-phase samples: {total_two_phase}")
        logger.info(f"Single gas phase samples: {total_single_gas}")
        logger.info(f"Average time per sample: {(total_time/total_processed)*1000:.1f} ms")
        logger.info(f"Final memory usage: {get_memory_usage():.1f} MB")
        if output_path:
            logger.info(f"Data saved to: {output_path}")


def main():
    """Main function to generate synthetic data with optimized memory usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic thermodynamic data')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to config file (required)')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load configuration
    config_file = args.config

    if not os.path.exists(config_file):
        logger.error(f"Configuration file not found: {config_file}")
        logger.info("Available config files in thermo_gen/config/:")
        for f in os.listdir(CONFIG_DIR):
            if f.endswith('.yaml'):
                logger.info(f"  - {f}")
        return

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loading configuration from {os.path.basename(config_file)}")
    
    # Extract configuration parameters
    grid_points = config.get('grid_points', None)
    batch_size = config.get('batch_size', 10000)  # Default batch size
    synthetic_data_config = config.get('synthetic_data', {})
    output_config = config.get('output', {})

    # Extract chemicals and component names from config
    chemicals_config = config.get('chemicals', None)
    if chemicals_config is None:
        logger.error("chemicals not specified in configuration")
        return

    chemicals = []
    component_names = []
    for chem in chemicals_config:
        if 'cas' not in chem:
            logger.error("Each chemical entry must have a 'cas' field")
            return
        if 'name' not in chem:
            logger.error("Each chemical entry must have a 'name' field")
            return
        chemicals.append(chem['cas'])
        component_names.append(chem['name'])

    num_components = len(chemicals)
    num_dimensions = 2 + num_components  # temp, pres, + flow rates
    logger.info(f"Loaded {num_components} chemicals from configuration: {component_names}")

    if grid_points is None:
        logger.error("grid_points not specified in configuration")
        return

    # Prepare output filename
    base_filename = output_config.get('filename', 'synthetic_thermo_data.csv')
    add_timestamp = output_config.get('add_timestamp', False)

    if add_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_parts = os.path.splitext(base_filename)
        output_filename = f"{filename_parts[0]}_{timestamp}{filename_parts[1]}"
    else:
        output_filename = base_filename

    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Log configuration
    total_samples = grid_points ** num_dimensions
    logger.info(f"Grid sampling: {grid_points} points per dimension")
    logger.info(f"Total samples to generate: {total_samples} ({grid_points}^{num_dimensions})")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of batches: {(total_samples + batch_size - 1) // batch_size}")
    logger.info(f"Output will be saved to: {output_path}")

    # Generate synthetic data
    generator = OptimizedThermoGenerator(
        chemicals=chemicals,
        component_names=component_names,
        use_real_thermo=THERMO_AVAILABLE
    )
    
    generator.generate_synthetic_dataset_optimized(
        config_ranges=synthetic_data_config,
        grid_points=grid_points,
        batch_size=batch_size,
        output_path=output_path,
        append_mode=False
    )
    
    logger.info("Generation complete!")


if __name__ == "__main__":
    main()