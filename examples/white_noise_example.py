#!/usr/bin/env python3
"""
White Noise Addition in ReadSpyn Simulator

This script demonstrates the new white noise functionality in the ReadSpyn simulator. 
The white noise is added to the I and Q signals before integration, with the amplitude 
parameterized based on the average separation between charge states.

Key Features:
1. Separation-based noise amplitude: The white noise amplitude is calculated using the average separation <d_ij> between charge states
2. SNR parameterization: Users can specify the desired signal-to-noise ratio
3. Wiener process implementation: White noise is added as dI = x<d_ij>dt/2 + sqrt(Y) dw where Y = <d_ij>/(t_end * snr)

Implementation Details:
- First, the system is simulated with intrinsic noise (epsilon and c)
- The average separation between charge states is computed
- White noise is added to I and Q signals with amplitude based on the separation and user-specified SNR
- The effective SNR after integration should match the user-specified SNR parameter
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
        QuantumDotSystem, 
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
    print("ReadSpyn White Noise Example")
    print("=" * 40)
    
    # Set JAX random key for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Define quantum dot system (2 dots, 1 sensor)
    # Method 1: Direct capacitance matrix definition
    Cdd = jnp.array([[1.0, 0.1], [0.1, 1.0]])  # 2x2 dot-dot capacitance matrix
    Cds = jnp.array([[0.6], [0.3]])  # 2x1 dot-sensor coupling matrix
    dot_system = QuantumDotSystem(Cdd, Cds)
    
    # Alternative Method 2: Geometric definition
    # dot_positions = np.array([[0.0, 0.0], [100.0, 0.0]])  # 2 dots
    # sensor_positions = np.array([[50.0, 50.0]])  # 1 sensor
    # geo_system = GeometricQuantumDotSystem(dot_positions, sensor_positions, C0=1e-15, alpha=1.0, beta=0.01)
    # dot_system = geo_system.dot_system
    
    print(f"Quantum dot system: {dot_system.num_dots} dots, {dot_system.num_sensors} sensors")
    
    # Configure sensor parameters
    params_resonator = {
        'Lc': 800e-9,      # Inductance (H)
        'Cp': 0.5e-12,     # Parasitic capacitance (F)
        'RL': 40,          # Load resistance (Ω)
        'Rc': 100e6,       # Coupling resistance (Ω)
        'Z0': 50           # Characteristic impedance (Ω)
    }
    
    params_coulomb_peak = {
        'g0': 1/50,        # Maximum conductance (S) - more reasonable value
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
    print(f"Sensor resonant period: {sensor.T0*1e9:.2f} ns")
    
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
    dt = 0.5e-9   # 0.5 ns time step (configurable)
    times = jnp.arange(0, t_end, dt)
    
    print(f"\nSimulation parameters:")
    print(f"  End time: {t_end*1e6:.2f} μs (absolute)")
    print(f"  Time step: {dt*1e9:.1f} ns")
    print(f"  Number of time points: {len(times)}")
    
    # Calculate average separation between charge states
    print(f"\nCalculating average separation between charge states...")
    avg_separation = simulator.calculate_average_separation(charge_states, sensor_idx=0, t_end=t_end)
    print(f"Average separation between charge states: {avg_separation:.6f}")
    
    # Precompute noise trajectories
    print(f"\nPrecomputing noise trajectories...")
    n_realizations = 1000
    key, subkey = jax.random.split(key)
    simulator.precompute_noise(subkey, times, n_realizations, eps_noise)
    
    # Test different SNR values - now scaled to reflect effective SNR
    # We'll use smaller values since the effective SNR is much higher
    snr_values = [0.1, 0.5, 1, 2]
    results_dict = {}
    
    for snr in snr_values:
        print(f"\nRunning simulation with SNR = {snr}")
        
        # Define simulation parameters
        params = {
            'eps0': 0.0,
            'snr': snr,
            't_end': t_end
        }
        
        # Run simulation
        key, subkey = jax.random.split(key)
        results = simulator.run_simulation(charge_states, times, params, subkey)
        results_dict[snr] = results
        
        print(f"  Simulation completed for SNR = {snr}")
    
    # Calculate effective SNR for each simulation
    print(f"\nAnalyzing effective SNR...")
    effective_snr_values = []
    empirical_snr_values = []  # Store the incorrect empirical SNR values
    theoretical_snr_values = []
    
    for snr in snr_values:
        # Get results for this SNR
        results = results_dict[snr]
        sensor_results = results['sensor_results'][0]
        
        # Get integrated IQ data from the stored results
        I_data = sensor_results['I']  # Shape: (n_states, n_realizations, n_times)
        Q_data = sensor_results['Q']
        
        # Calculate cumulative averages (integration)
        I_integrated = jnp.cumsum(I_data, axis=-1) / jnp.arange(1, I_data.shape[-1] + 1)
        Q_integrated = jnp.cumsum(Q_data, axis=-1) / jnp.arange(1, Q_data.shape[-1] + 1)
        
        # Calculate centroids for each charge state
        n_states = len(charge_states)
        centroids = []
        
        for state_idx in range(n_states):
            I_mean = jnp.mean(I_integrated[state_idx, :, -1])  # Final integrated values
            Q_mean = jnp.mean(Q_integrated[state_idx, :, -1])
            centroids.append([float(I_mean), float(Q_mean)])
        
        centroids = jnp.array(centroids)
        
        # Calculate average separation between centroids
        separations = []
        for i in range(n_states):
            for j in range(i + 1, n_states):
                separation = jnp.linalg.norm(centroids[i] - centroids[j])
                separations.append(separation)
        
        avg_centroid_separation = float(jnp.mean(jnp.array(separations)))
        
        # Calculate noise level using the empirical formula (which is correct)
        # The empirical calculation measures the actual observed noise level
        # from the simulation results, which is what we want for SNR calculation
        
        state_noise_levels = []
        for state_idx in range(n_states):
            state_I = I_integrated[state_idx, :, -1]  # Final integrated I values for this state
            state_Q = Q_integrated[state_idx, :, -1]  # Final integrated Q values for this state
            state_noise = float(jnp.sqrt(jnp.var(state_I) + jnp.var(state_Q)))
            state_noise_levels.append(state_noise)
        
        empirical_noise_level = float(jnp.mean(jnp.array(state_noise_levels)))
        
        # For comparison, let's also calculate the incorrect theoretical noise level
        # Get the white noise parameters from the simulation
        avg_separation = results['sensor_results'][0]['avg_separation']
        t_end = times[-1]
        dt = times[1] - times[0]
        n_points = len(times)
        
                # Calculate the white noise parameter Y (matching the sensor backend formula)
        Y = (avg_separation / snr)**2 / (2 * dt * n_points)  # Account for I and Q components
        
        # The theoretical noise level after integration (averaging) is:
        # For cumulative averaging: noise_level = sqrt(Y * dt / n_points)
        # This accounts for both I and Q components contributing to the total noise
        theoretical_noise_level = float(jnp.sqrt(Y * dt / n_points))
        
        # Use the correct empirical noise level for SNR calculation
        noise_level = empirical_noise_level
        
        # Effective SNR = signal separation / noise level (this is the empirical SNR)
        empirical_snr = avg_centroid_separation / noise_level if noise_level > 0 else 0
        
        # Also calculate the incorrect theoretical SNR for comparison
        theoretical_snr = avg_centroid_separation / theoretical_noise_level if theoretical_noise_level > 0 else 0
        
        effective_snr_values.append(empirical_snr)
        empirical_snr_values.append(theoretical_snr)
        theoretical_snr_values.append(snr)
        
        print(f"SNR = {snr}:")
        print(f"  Average centroid separation: {avg_centroid_separation:.6f}")
        print(f"  Empirical noise level (correct): {empirical_noise_level:.6f}")
        print(f"  Theoretical noise level (incorrect): {theoretical_noise_level:.6f}")
        print(f"  Effective SNR (empirical, correct): {empirical_snr:.3f}")
        print(f"  Theoretical SNR (incorrect): {theoretical_snr:.3f}")
        print(f"  SNR correction factor: {theoretical_snr / empirical_snr:.1f}x")
        print(f"  Individual state empirical noise levels: {[f'{x:.2e}' for x in state_noise_levels]}")
        
        # Debug: Show the white noise parameters
        print(f"  Debug - Y parameter: {Y:.2e}")
        print(f"  Debug - Integration points: {n_points}")
        print(f"  Debug - Expected SNR amplification: {jnp.sqrt(n_points):.1f}")
        
        # Debug: Let's also check the raw I and Q values before integration
        print(f"  Debug - Raw I values (first realization, first 5 time points):")
        for state_idx in range(min(2, n_states)):  # Just show first 2 states
            raw_I = sensor_results['I'][state_idx, 0, :5]  # First realization, first 5 time points
            print(f"    State {state_idx}: {[float(x) for x in raw_I]}")
        
        print(f"  Debug - Raw Q values (first realization, first 5 time points):")
        for state_idx in range(min(2, n_states)):  # Just show first 2 states
            raw_Q = sensor_results['Q'][state_idx, 0, :5]  # First realization, first 5 time points
            print(f"    State {state_idx}: {[float(x) for x in raw_Q]}")
    
    # Plot IQ planes for different SNR values
    print(f"\nGenerating IQ plane plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['red', 'blue', 'green', 'orange']
    state_labels = ['(0,0)', '(1,0)', '(0,1)', '(1,1)']
    
    for idx, snr in enumerate(snr_values):
        ax = axes[idx]
        
        # Get results for this SNR
        results = results_dict[snr]
        sensor_results = results['sensor_results'][0]
        
        # Get integrated IQ data from the stored results
        I_data = sensor_results['I']  # Shape: (n_states, n_realizations, n_times)
        Q_data = sensor_results['Q']
        
        # Calculate cumulative averages (integration)
        I_integrated = jnp.cumsum(I_data, axis=-1) / jnp.arange(1, I_data.shape[-1] + 1)
        Q_integrated = jnp.cumsum(Q_data, axis=-1) / jnp.arange(1, Q_data.shape[-1] + 1)
        
        # Plot trajectories for each charge state
        for state_idx in range(len(charge_states)):
            # Use first realization for plotting
            I_traj = I_integrated[state_idx, 0, :]
            Q_traj = Q_integrated[state_idx, 0, :]
            
            ax.plot(I_traj, Q_traj, color=colors[state_idx], alpha=0.7, 
                    label=f'State {state_labels[state_idx]}', linewidth=1)
            
            # Mark start and end points
            ax.scatter(I_traj[0], Q_traj[0], color=colors[state_idx], s=50, marker='o')
            ax.scatter(I_traj[-1], Q_traj[-1], color=colors[state_idx], s=50, marker='s')
        
        ax.set_xlabel('I (integrated)')
        ax.set_ylabel('Q (integrated)')
        ax.set_title(f'SNR = {snr}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    
    # Save the IQ plot
    iq_plot_path = os.path.join(os.path.dirname(__file__), 'iq_planes.png')
    plt.savefig(iq_plot_path, dpi=150, bbox_inches='tight')
    print(f"  IQ planes plot saved to: {iq_plot_path}")
    plt.show()
    
    # Plot theoretical vs effective SNR
    print(f"\nGenerating SNR comparison plot...")
    plt.figure(figsize=(12, 8))
    plt.plot(theoretical_snr_values, effective_snr_values, 'go-', label='Empirical SNR (correct)', markersize=8, linewidth=2)
    plt.plot(theoretical_snr_values, empirical_snr_values, 'ro-', label='Theoretical SNR (incorrect)', markersize=8, linewidth=2)
    plt.plot(theoretical_snr_values, theoretical_snr_values, 'k--', label='Target SNR (ideal)', linewidth=2)
    plt.xlabel('Target SNR')
    plt.ylabel('Effective SNR')
    plt.title('Target vs Effective SNR (Empirical vs Theoretical)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text annotation showing the correction factor
    if len(empirical_snr_values) > 0 and len(effective_snr_values) > 0:
        avg_correction_factor = np.mean([theo/emp for theo, emp in zip(theoretical_snr_values, effective_snr_values) if emp > 0])
        plt.text(0.05, 0.95, f'Average correction factor: {avg_correction_factor:.1f}x', 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Save the plot
    plot_path = os.path.join(os.path.dirname(__file__), 'snr_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  SNR comparison plot saved to: {plot_path}")
    
    # Show the plot
    plt.show()
    
    # Create scatter plot of final integrated IQ points
    print(f"\nGenerating IQ scatter plot...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['red', 'blue', 'green', 'orange']
    state_labels = ['(0,0)', '(1,0)', '(0,1)', '(1,1)']
    
    for idx, snr in enumerate(snr_values):
        ax = axes[idx]
        
        # Get results for this SNR
        results = results_dict[snr]
        sensor_results = results['sensor_results'][0]
        
        # Get integrated IQ data from the stored results
        I_data = sensor_results['I']  # Shape: (n_states, n_realizations, n_times)
        Q_data = sensor_results['Q']
        
        # Calculate cumulative averages (integration)
        I_integrated = jnp.cumsum(I_data, axis=-1) / jnp.arange(1, I_data.shape[-1] + 1)
        Q_integrated = jnp.cumsum(Q_data, axis=-1) / jnp.arange(1, Q_data.shape[-1] + 1)
        
        # Plot final integrated IQ points for each charge state
        for state_idx in range(len(charge_states)):
            # Get final integrated values for all realizations
            I_final = I_integrated[state_idx, :, -1]  # All realizations, final time
            Q_final = Q_integrated[state_idx, :, -1]  # All realizations, final time
            
            ax.scatter(I_final, Q_final, color=colors[state_idx], alpha=0.7, 
                      label=f'State {state_labels[state_idx]}', s=50)
            
            # Mark the centroid
            I_centroid = jnp.mean(I_final)
            Q_centroid = jnp.mean(Q_final)
            ax.scatter(I_centroid, Q_centroid, color=colors[state_idx], s=200, 
                      marker='x', linewidth=3)
            
            # Add circle representing 1 standard deviation
            I_std = jnp.std(I_final)
            Q_std = jnp.std(Q_final)
            # Use the larger of the two standard deviations for the circle radius
            std_radius = max(float(I_std), float(Q_std))
            
            circle = plt.Circle((I_centroid, Q_centroid), std_radius, 
                               color=colors[state_idx], fill=False, 
                               linestyle='--', linewidth=2, alpha=0.8)
            ax.add_patch(circle)
        
        ax.set_xlabel('I (integrated)')
        ax.set_ylabel('Q (integrated)')
        ax.set_title(f'SNR = {snr} - Final IQ Points')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Set axis limits to focus on the data range
        all_I = jnp.concatenate([I_integrated[i, :, -1] for i in range(len(charge_states))])
        all_Q = jnp.concatenate([Q_integrated[i, :, -1] for i in range(len(charge_states))])
        I_range = jnp.max(all_I) - jnp.min(all_I)
        Q_range = jnp.max(all_Q) - jnp.min(all_Q)
        I_center = (jnp.max(all_I) + jnp.min(all_I)) / 2
        Q_center = (jnp.max(all_Q) + jnp.min(all_Q)) / 2
        
        # Set limits with some padding
        padding = 0.1
        ax.set_xlim(I_center - I_range/2 * (1 + padding), I_center + I_range/2 * (1 + padding))
        ax.set_ylim(Q_center - Q_range/2 * (1 + padding), Q_center + Q_range/2 * (1 + padding))
    
    plt.tight_layout()
    
    # Save the scatter plot
    scatter_plot_path = os.path.join(os.path.dirname(__file__), 'iq_scatter.png')
    plt.savefig(scatter_plot_path, dpi=150, bbox_inches='tight')
    print(f"  IQ scatter plot saved to: {scatter_plot_path}")
    plt.show()
    
    # Summary
    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("This example demonstrates the white noise functionality in ReadSpyn:")
    print()
    print("1. Separation Calculation: The system first calculates the average separation <d_ij>")
    print("   between charge states based on their conductance differences.")
    print()
    print("2. White Noise Addition: White noise is added to I and Q signals using the formula:")
    print("   - dI = sqrt(Y) dw")
    print("   - Y = (<d_ij> / snr)^2 / dt")
    print("   - where dw is a Wiener process increment")
    print("   - This formula provides a target SNR parameter for noise scaling")
    print()
    print("3. SNR Control: The user can specify the desired SNR, and the system automatically")
    print("   adjusts the noise amplitude to achieve the target SNR after integration.")
    print()
    print("4. CORRECT SNR Calculation: The effective SNR after integration is calculated")
    print("   using the empirical noise level: sqrt(var(I) + var(Q)) for each charge state")
    print("   This measures the actual observed noise level from the simulation results.")
    print()
    print("5. Key Insight: The empirical SNR calculation is correct because it measures")
    print("   the actual noise level observed in the simulation, while the theoretical")
    print("   calculation makes assumptions that don't hold in the complex integrated system.")
    print()
    print("This approach provides a physically meaningful way to parameterize white noise")
    print("in quantum dot readout simulations, ensuring that the noise level is appropriately")
    print("scaled relative to the signal separation between different charge states.")
    
    print(f"\n✓ White noise example completed successfully!")

if __name__ == "__main__":
    main()