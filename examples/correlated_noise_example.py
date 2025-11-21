#!/usr/bin/env python3
"""
Correlated Noise Example for ReadSpyn

This example demonstrates how to use correlated noise between multiple sensors
in the ReadSpyn simulator. It shows how to:

1. Create a quantum dot system with multiple sensors
2. Set up correlated noise models with different correlation strengths
3. Compare correlated vs uncorrelated noise effects
4. Visualize the correlation effects

Author: AI Assistant
Date: 2024
"""

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from typing import Dict, List

# Import ReadSpyn components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from readout_simulator.quantum_dot_system import QuantumDotSystem
from readout_simulator.sensor_backend import RLC_sensor
from readout_simulator.jax_simulator import JAXReadoutSimulator
from readout_simulator.noise_models import OU_noise, OverFNoise, CorrelatedNoise


def create_quantum_dot_system() -> QuantumDotSystem:
    """Create a quantum dot system with 2 dots and 2 sensors."""
    # Dot-dot capacitance matrix (2x2)
    Cdd = np.array([
        [1.0e-15, 0.2e-15],  # Dot 1 self-capacitance and coupling to dot 2
        [0.2e-15, 1.0e-15]   # Dot 2 self-capacitance and coupling to dot 1
    ])
    
    # Dot-sensor capacitance matrix (2x2)
    Cds = np.array([
        [0.5e-15, 0.1e-15],  # Dot 1 coupling to sensors
        [0.1e-15, 0.5e-15]   # Dot 2 coupling to sensors
    ])
    
    # Create quantum dot system
    dot_system = QuantumDotSystem(Cdd, Cds)
    print(f"Created quantum dot system with {dot_system.num_dots} dots and {dot_system.num_sensors} sensors")
    
    return dot_system


def create_sensors() -> List[RLC_sensor]:
    """Create two RLC sensors with identical parameters."""
    # Resonator parameters (identical for both sensors)
    params_resonator = {
        'Lc': 800e-9,           # Inductance (H)
        'Cp': 0.6e-12,          # Parasitic capacitance (F)
        'RL': 40,               # Load resistance (Ω)
        'Rc': 100e6,            # Coupling resistance (Ω)
        'Z0': 50.0,             # Characteristic impedance (Ω)
        'self_capacitance': 0   # Additional self-capacitance (F)
    }
    
    # Coulomb peak parameters (identical for both sensors)
    params_coulomb_peak = {
        'g0': 1/50,             # Maximum conductance (S)
        'eps0': 0.5,            # Operating point (relative to eps_width)
        'eps_width': 1.0        # Energy width (eV)
    }
    
    # Create two identical sensors
    sensors = []
    for i in range(2):
        sensor = RLC_sensor(params_resonator, params_coulomb_peak)
        sensors.append(sensor)
        print(f"Sensor {i+1} resonant frequency: {sensor.f0/1e9:.2f} GHz")
    
    return sensors


def create_correlation_matrices() -> Dict[str, jnp.ndarray]:
    """Create different correlation matrices for testing."""
    correlation_matrices = {}
    
    # 1. No correlation (identity matrix)
    correlation_matrices['uncorrelated'] = jnp.eye(2)
    
    # 2. Moderate positive correlation
    correlation_matrices['moderate_positive'] = jnp.array([
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    
    # 3. Strong positive correlation
    correlation_matrices['strong_positive'] = jnp.array([
        [1.0, 0.9],
        [0.9, 1.0]
    ])
    
    # 4. Moderate negative correlation
    correlation_matrices['moderate_negative'] = jnp.array([
        [1.0, -0.3],
        [-0.3, 1.0]
    ])
    
    return correlation_matrices


def run_correlated_simulation(dot_system: QuantumDotSystem, 
                            sensors: List[RLC_sensor],
                            correlation_matrix: jnp.ndarray,
                            correlation_name: str,
                            key: jr.PRNGKey) -> Dict:
    """Run simulation with correlated noise."""
    print(f"\n=== Running simulation with {correlation_name} correlation ===")
    
    # Create base noise model (1/f noise)
    base_noise = OverFNoise(
        n_fluctuators=5, 
        S1=1e-6, 
        sigma_couplings=0.1,
        ommax=1e6, 
        ommin=1e3, 
        equally_dist=True
    )
    
    # Create correlated noise model
    correlated_noise = CorrelatedNoise(base_noise, correlation_matrix)
    
    # Create simulator
    simulator = JAXReadoutSimulator(dot_system, sensors)
    
    # Define charge states to simulate
    charge_states = jnp.array([
        [1, 0],  # First dot occupied
        [0, 1],  # Second dot occupied
    ])
    
    # Define simulation parameters
    t_end = 1e-6  # 1 μs
    dt = 0.5e-9   # 0.5 ns
    times = jnp.arange(0, t_end, dt)
    n_realizations = 10
    
    print(f"Simulation parameters:")
    print(f"  End time: {t_end*1e6:.2f} μs")
    print(f"  Time step: {dt*1e9:.1f} ns")
    print(f"  Number of time points: {len(times)}")
    print(f"  Number of realizations: {n_realizations}")
    
    # Precompute noise trajectories
    simulator.precompute_noise(key, times, n_realizations, correlated_noise)
    
    # Run simulation
    params = {'snr': 1.0, 't_end': t_end}
    results = simulator.run_simulation(charge_states, times, params, key)
    
    return results


def analyze_correlation_effects(results_dict: Dict[str, Dict]) -> None:
    """Analyze and visualize the correlation effects."""
    print("\n=== Analyzing Correlation Effects ===")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Correlated Noise Effects in ReadSpyn', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (correlation_name, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        
        # Plot noise trajectories for first realization
        ax1 = axes[0, 0] if i < 2 else axes[0, 1]
        sensor_idx = 0 if i < 2 else 1
        
        noise_traj = results['sensor_results'][sensor_idx]['noise_trajectory'][0, 0]  # First state, first realization
        times = results['times']
        
        ax1.plot(times*1e6, noise_traj*1e6, color=color, alpha=0.7, 
                label=f'{correlation_name} (Sensor {sensor_idx+1})')
        ax1.set_xlabel('Time (μs)')
        ax1.set_ylabel('Noise (μeV)')
        ax1.set_title('Noise Trajectories (First Realization)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot correlation between sensors for different correlation types
    ax2 = axes[1, 0]
    for i, (correlation_name, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        
        # Extract noise trajectories for both sensors
        sensor1_noise = results['sensor_results'][0]['noise_trajectory'][0, 0]  # First state, first realization
        sensor2_noise = results['sensor_results'][1]['noise_trajectory'][0, 0]
        
        # Calculate correlation coefficient
        correlation_coeff = np.corrcoef(sensor1_noise, sensor2_noise)[0, 1]
        
        ax2.scatter(sensor1_noise*1e6, sensor2_noise*1e6, color=color, alpha=0.6, 
                   label=f'{correlation_name} (r={correlation_coeff:.3f})', s=20)
    
    ax2.set_xlabel('Sensor 1 Noise (μeV)')
    ax2.set_ylabel('Sensor 2 Noise (μeV)')
    ax2.set_title('Noise Correlation Between Sensors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot integrated IQ signals
    ax3 = axes[1, 1]
    for i, (correlation_name, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        
        # Get integrated IQ for first sensor, first state
        sensor_results = results['sensor_results'][0]
        I_signal = sensor_results['I'][0, 0]  # First state, first realization
        Q_signal = sensor_results['Q'][0, 0]
        
        # Integrate over time
        integrated_I = jnp.sum(I_signal)
        integrated_Q = jnp.sum(Q_signal)
        
        ax3.scatter(integrated_I, integrated_Q, color=color, s=100, 
                   label=f'{correlation_name}', alpha=0.7)
    
    ax3.set_xlabel('Integrated I Signal')
    ax3.set_ylabel('Integrated Q Signal')
    ax3.set_title('Integrated IQ Signals (First State)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print correlation analysis
    print("\nCorrelation Analysis:")
    print("-" * 50)
    for correlation_name, results in results_dict.items():
        sensor1_noise = results['sensor_results'][0]['noise_trajectory'][0, 0]
        sensor2_noise = results['sensor_results'][1]['noise_trajectory'][0, 0]
        correlation_coeff = np.corrcoef(sensor1_noise, sensor2_noise)[0, 1]
        print(f"{correlation_name:20s}: r = {correlation_coeff:.4f}")


def main():
    """Main function to run the correlated noise example."""
    print("ReadSpyn Correlated Noise Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    key = jr.PRNGKey(42)
    
    # Create quantum dot system and sensors
    dot_system = create_quantum_dot_system()
    sensors = create_sensors()
    
    # Create correlation matrices
    correlation_matrices = create_correlation_matrices()
    
    # Run simulations with different correlation types
    results_dict = {}
    for correlation_name, correlation_matrix in correlation_matrices.items():
        key, subkey = jr.split(key)
        results = run_correlated_simulation(
            dot_system, sensors, correlation_matrix, correlation_name, subkey
        )
        results_dict[correlation_name] = results
    
    # Analyze and visualize results
    analyze_correlation_effects(results_dict)
    
    print("\nExample completed successfully!")
    print("\nKey findings:")
    print("- Correlated noise creates dependencies between sensor readings")
    print("- Strong positive correlation makes sensors move together")
    print("- Negative correlation makes sensors move in opposite directions")
    print("- This affects the overall readout fidelity and error rates")


if __name__ == "__main__":
    main()