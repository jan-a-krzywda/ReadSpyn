# ReadSpyn Examples

This directory contains example scripts demonstrating the JAX-based ReadSpyn implementation.

## Available Examples

### 1. `white_noise_example.py`
A comprehensive example demonstrating white noise effects in quantum dot readout.

**Features:**
- Two-dot, one-sensor system
- White noise (OU noise) affecting capacitance
- 1/f noise affecting energy offset
- Signal quality analysis with SNR and separation metrics
- I, Q scatter plots and performance visualization

**Usage:**
```bash
python examples/white_noise_example.py
```

**Expected Output:**
- Simulation parameters and system setup
- Signal quality metrics (separation, noise level, SNR)
- Performance analysis plots
- Comprehensive noise effects demonstration

### 2. `simple_1f_noise_example.py`
A focused example demonstrating 1/f noise effects with asymmetric coupling.

**Features:**
- Two-dot, one-sensor system with asymmetric coupling
- 1/f noise affecting energy offset (epsilon)
- White noise (OU noise) affecting capacitance
- Coulomb peak visualization showing asymmetric coupling
- Parameter study of different 1/f noise amplitudes
- Signal degradation analysis

**Usage:**
```bash
python examples/simple_1f_noise_example.py
```

**Expected Output:**
- Coulomb peaks plot showing asymmetric coupling effects
- Clean vs noisy signal comparison
- 1/f noise parameter study results
- Signal degradation analysis plots
- Comprehensive noise effects demonstration

### 3. `geometric_system_example.py`
A geometric example demonstrating quantum dot system creation from spatial positions.

**Features:**
- Geometric quantum dot system creation
- Capacitance calculation from spatial positions
- Basic simulation setup
- System parameter demonstration

**Usage:**
```bash
python examples/geometric_system_example.py
```

**Expected Output:**
- Geometric system setup and parameters
- Capacitance matrix calculations
- Basic simulation demonstration

## Key Features Demonstrated

### JAX-based Implementation
- **Precomputed Noise Trajectories**: Noise is generated once and reused across all states
- **State Scanning**: Uses JAX scan for efficient processing of multiple charge states
- **Vectorized Operations**: All computations are vectorized for GPU acceleration
- **Post-processing Noise**: White noise is added after signal generation

### Noise Models
- **OU_noise**: Ornstein-Uhlenbeck noise with exponential autocorrelation (white noise)
- **OverFNoise**: 1/f noise using multiple fluctuators
- **Precomputed Trajectories**: Efficient noise generation and reuse

### System Configurations
- **Two-dot, one-sensor**: Asymmetric coupling demonstration
- **Geometric systems**: Spatial position-based system creation
- **Capacitive coupling**: Realistic quantum dot interactions

## Requirements

- Python 3.9+ with JAX installed
- ReadSpyn package installed
- Matplotlib for visualization

## Running Examples

1. **White Noise Example** (comprehensive noise effects):
   ```bash
   python examples/white_noise_example.py
   ```

2. **1/f Noise Example** (asymmetric coupling effects):
   ```bash
   python examples/simple_1f_noise_example.py
   ```

3. **Geometric System Example** (spatial system creation):
   ```bash
   python examples/geometric_system_example.py
   ```

## Expected Performance

The examples demonstrate significant performance improvements:

- **Efficient JAX operations**: Vectorized computations
- **GPU acceleration**: Compatible with JAX's GPU acceleration
- **Scalable**: Performance scales well with number of states and realizations

## Key Insights

### White Noise Example
- Demonstrates fundamental noise effects in quantum dot readout
- Shows SNR scaling and signal quality metrics
- Provides baseline for noise comparison

### 1/f Noise Example
- Demonstrates asymmetric coupling effects
- Shows how 1/f noise affects different charge states differently
- Provides realistic quantum dot system modeling
- Demonstrates complex noise coupling patterns

### Geometric System Example
- Shows how to create systems from spatial positions
- Demonstrates capacitance calculation methods
- Provides foundation for realistic system modeling

## Troubleshooting

If you encounter issues:

1. **JAX Installation**: Ensure JAX is properly installed for your Python version
2. **Import Errors**: Make sure the ReadSpyn package is in your Python path
3. **Memory Issues**: Reduce the number of realizations or time points for large simulations

## Customization

You can modify the examples to:

- Change quantum dot system parameters
- Adjust noise model parameters
- Modify simulation time and resolution
- Add custom analysis functions
- Integrate with your own workflows

The examples serve as templates for building your own quantum dot readout simulations using the JAX-based ReadSpyn implementation. 