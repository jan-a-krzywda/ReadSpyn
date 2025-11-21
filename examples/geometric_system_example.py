#!/usr/bin/env python3
"""
Geometric Quantum Dot System Example

This script demonstrates how to define quantum dot systems using geometric positions
of dots and sensors, allowing users to specify physical layouts and automatically
compute capacitance matrices based on geometric considerations.

Key Features:
1. Geometric positioning: Define dots and sensors by their 2D or 3D positions
2. Automatic capacitance computation: Capacitance matrices are computed based on distances
3. Visual system layout: Plot the geometric arrangement of dots and sensors
4. Configurable parameters: Adjust coupling strength and distance decay parameters
5. Integration with existing simulator: Use the geometric system with the full ReadSpyn simulator

Implementation Details:
- Dot-dot capacitances follow C_ij = C0 for i=j, -α*C0*exp(-β*d_ij) for i≠j
- Dot-sensor capacitances follow C_ij = -α*C0*exp(-β*d_ij)
- Distances are computed using Euclidean norm
- The system integrates seamlessly with existing ReadSpyn components
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from readout_simulator import (
        GeometricQuantumDotSystem,
        RLC_sensor, 
        JAXReadoutSimulator,
        OU_noise, 
        OverFNoise
    )
    print("✓ Successfully imported ReadSpyn components")
except ImportError as e:
    print(f"✗ Failed to import ReadSpyn components: {e}")
    sys.exit(1)

def main():
    print("ReadSpyn Geometric Quantum Dot System Example")
    print("=" * 50)
    
    # Set JAX random key for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Example 1: Simple 2-dot, 1-sensor system
    print("\n1. Creating a simple 2-dot, 1-sensor system...")
    
   
    # Example 2: More complex 3-dot, 2-sensor system
    print("\n\n2. Creating a more complex 3-dot, 2-sensor system...")
    
    # Define positions for a triangular arrangement
    dot_positions = np.array([
        [0.0, 0.0],      # Dot 1     # Dot 3 (equilateral triangle)
    ])
    
    sensor_positions = np.array([
        [25.0, 43.3],    # Sensor 1 (center of triangle)
        [75.0, 43.3]     # Sensor 2 (offset from center)
    ])
    
    # Create geometric system with different parameters
    geo_system = GeometricQuantumDotSystem(
        dot_positions=dot_positions,
        sensor_positions=sensor_positions,
        C0=1e-15,      # Base capacitance: 1 fF
        alpha=0.8,     # Stronger coupling
        beta=0.005     # Longer range coupling
    )
    
    print(f"Created complex geometric system: {geo_system}")
    
    # Plot the complex system layout
    print("\nPlotting complex system layout...")
    geo_system.plot_system(show_capacitances=True, capacitance_threshold=0.005)
    
    # Get coupling information for complex system
    coupling_info_2 = geo_system.get_coupling_info()
    print(f"\nComplex system coupling information:")
    print(f"  Average dot-dot coupling: {coupling_info_2['avg_dot_dot_coupling']:.2e} F")
    print(f"  Average dot-sensor coupling: {coupling_info_2['avg_dot_sensor_coupling']:.2e} F")
    print(f"  Max coupling strength: {coupling_info_2['max_coupling']:.3f}")
    print(f"  Min coupling strength: {coupling_info_2['min_coupling']:.3f}")
    
    # Example 3: Integration with full simulator
    print("\n\n3. Integrating with full ReadSpyn simulator...")
    
    # Use the first geometric system for simulation
    dot_system = geo_system.dot_system
    
    print(f"Using geometric system: {dot_system.num_dots} dots, {dot_system.num_sensors} sensors")
    
    # Configure sensor parameters
    params_resonator = {
        'Lc': 800e-9,      # Inductance (H)
        'Cp': 0.5e-12,     # Parasitic capacitance (F)
        'RL': 40,          # Load resistance (Ω)
        'Rc': 100e6,       # Coupling resistance (Ω)
        'Z0': 50           # Characteristic impedance (Ω)
    }
    
    params_coulomb_peak = {
        'g0': 1/50,        # Maximum conductance (S)
        'eps0': 0.5,       # Operating point (relative to eps_width)
        'eps_width': 1.0   # Energy width (eV)
    }
    
    # Create noise models
    eps_noise = OverFNoise(n_fluctuators=3, S1=1e-6, sigma_couplings=0.1,
                           ommax=1e6, ommin=1e3, equally_dist=True)
    c_noise = OU_noise(sigma=1e-12, gamma=1e5)
    
    # Create sensor
    sensor = RLC_sensor(params_resonator, params_coulomb_peak, c_noise, eps_noise)
    
    print(f"Sensor resonant frequency: {sensor.f0/1e9:.2f} GHz")
    
    # Create simulator
    simulator = JAXReadoutSimulator(dot_system, [sensor])
    
    # Define charge states to simulate
    charge_states = jnp.array([
        [1, 0],  # First dot occupied
        [0, 1],  # Second dot occupied
    ])
    
    print(f"\nCharge states to simulate:")
    for i, state in enumerate(charge_states):
        print(f"  State {i}: {state}")
    
    # Define simulation parameters
    t_end = 1e-6  # 1 μs absolute end time
    dt = 0.5e-9   # 0.5 ns time step
    times = jnp.arange(0, t_end, dt)
    
    print(f"\nSimulation parameters:")
    print(f"  End time: {t_end*1e6:.2f} μs (absolute)")
    print(f"  Time step: {dt*1e9:.1f} ns")
    print(f"  Number of time points: {len(times)}")
    
    # Calculate average separation between charge states
    print(f"\nCalculating average separation between charge states...")
    avg_separation = simulator.calculate_average_separation(charge_states, sensor_idx=0, t_end=t_end)
    print(f"Average separation between charge states: {avg_separation:.6f}")
    
    # Example 4: Parameter study - effect of geometric parameters
    print("\n\n4. Parameter study - effect of geometric parameters...")
    
    # Test different coupling strengths
    alpha_values = [0.5, 1.0, 1.5]
    beta_values = [0.005, 0.01, 0.02]
    
    print(f"Testing {len(alpha_values)} × {len(beta_values)} = {len(alpha_values) * len(beta_values)} parameter combinations...")
    
    results = []
    
    for alpha in alpha_values:
        for beta in beta_values:
            # Create system with these parameters
            test_system = GeometricQuantumDotSystem(
                dot_positions=dot_positions,
                sensor_positions=sensor_positions,
                C0=1e-15,
                alpha=alpha,
                beta=beta
            )
            
            # Get coupling info
            coupling_info = test_system.get_coupling_info()
            
            results.append({
                'alpha': alpha,
                'beta': beta,
                'avg_dot_dot': coupling_info['avg_dot_dot_coupling'],
                'avg_dot_sensor': coupling_info['avg_dot_sensor_coupling'],
                'max_coupling': coupling_info['max_coupling'],
                'min_coupling': coupling_info['min_coupling']
            })
    
    # Plot parameter study results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Extract data for plotting
    alphas = [r['alpha'] for r in results]
    betas = [r['beta'] for r in results]
    avg_dot_dot = [r['avg_dot_dot'] for r in results]
    avg_dot_sensor = [r['avg_dot_sensor'] for r in results]
    max_coupling = [r['max_coupling'] for r in results]
    min_coupling = [r['min_coupling'] for r in results]
    
    # Plot 1: Average dot-dot coupling vs parameters
    scatter1 = axes[0].scatter(alphas, betas, c=avg_dot_dot, s=100, cmap='viridis')
    axes[0].set_xlabel('α (coupling strength)')
    axes[0].set_ylabel('β (distance decay)')
    axes[0].set_title('Average Dot-Dot Coupling')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Plot 2: Average dot-sensor coupling vs parameters
    scatter2 = axes[1].scatter(alphas, betas, c=avg_dot_sensor, s=100, cmap='viridis')
    axes[1].set_xlabel('α (coupling strength)')
    axes[1].set_ylabel('β (distance decay)')
    axes[1].set_title('Average Dot-Sensor Coupling')
    plt.colorbar(scatter2, ax=axes[1])
    
    # Plot 3: Max coupling strength vs parameters
    scatter3 = axes[2].scatter(alphas, betas, c=max_coupling, s=100, cmap='viridis')
    axes[2].set_xlabel('α (coupling strength)')
    axes[2].set_ylabel('β (distance decay)')
    axes[2].set_title('Max Coupling Strength')
    plt.colorbar(scatter3, ax=axes[2])
    
    # Plot 4: Min coupling strength vs parameters
    scatter4 = axes[3].scatter(alphas, betas, c=min_coupling, s=100, cmap='viridis')
    axes[3].set_xlabel('α (coupling strength)')
    axes[3].set_ylabel('β (distance decay)')
    axes[3].set_title('Min Coupling Strength')
    plt.colorbar(scatter4, ax=axes[3])
    
    plt.tight_layout()
    
    # Save the parameter study plot
    param_plot_path = os.path.join(os.path.dirname(__file__), 'geometric_parameter_study.png')
    plt.savefig(param_plot_path, dpi=150, bbox_inches='tight')
    print(f"  Parameter study plot saved to: {param_plot_path}")
    plt.show()
    
    # Summary
    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("This example demonstrates the geometric quantum dot system functionality:")
    print()
    print("1. Geometric Positioning: Define dots and sensors by their 2D/3D positions")
    print("2. Automatic Capacitance Computation: Capacitance matrices are computed")
    print("   based on geometric distances using exponential decay model")
    print("3. Visual System Layout: Plot the geometric arrangement with capacitance")
    print("   connections shown as dashed lines")
    print("4. Parameter Control: Adjust coupling strength (α) and distance decay (β)")
    print("5. Integration: Seamlessly integrate with existing ReadSpyn simulator")
    print()
    print("Key Parameters:")
    print("  - C0: Base capacitance scale (typically 1 fF)")
    print("  - α: Coupling strength parameter (typically 0.5-1.5)")
    print("  - β: Distance decay parameter (smaller = longer range coupling)")
    print()
    print("Capacitance Model:")
    print("  - C_ii = C0 (self-capacitance)")
    print("  - C_ij = -α * C0 * exp(-β * d_ij) (mutual capacitance)")
    print("  - where d_ij is the Euclidean distance between elements")
    print()
    print("This approach provides an intuitive way to define quantum dot systems")
    print("based on physical layout, making it easier to model realistic devices.")
    
    print(f"\n✓ Geometric quantum dot system example completed successfully!")

if __name__ == "__main__":
    main() 