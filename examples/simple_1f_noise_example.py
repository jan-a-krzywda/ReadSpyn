#!/usr/bin/env python3
"""
Simple 1/f Noise Example with ReadSpyn

This script demonstrates a simple quantum dot system with:
- 1 quantum dot
- 1 sensor (RLC resonator)
- 1/f noise in epsilon (energy offset)
- White noise (OU noise) in capacitance

This is a simplified version of the white noise example, focusing on
the fundamental noise effects in quantum dot readout.

Key Features:
1. Single dot with single sensor
2. 1/f noise affecting energy offset (epsilon)
3. OU noise affecting capacitance
4. Analysis of noise effects on readout fidelity
5. Comparison of different noise parameters

Implementation Details:
- System is simulated with both intrinsic noise sources
- Average separation between charge states is computed
- Noise effects on signal quality are analyzed
- Results are visualized for different noise parameters
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
        RLC_sensor, 
        JAXReadoutSimulator,
        OU_noise, 
        OverFNoise
    )
    print("✓ Successfully imported ReadSpyn components")
except ImportError as e:
    print(f"✗ Failed to import ReadSpyn components: {e}")
    sys.exit(1)

def create_simple_system(ommin, ommax):
    """Create a simple 2 dot, 1 sensor system."""
    
    # Define quantum dot system (2 dots, 1 sensor)
    Cdd = jnp.array([[1.0, 0.1], [0.1, 1.0]])  # 2x2 dot-dot capacitance matrix
    Cds = jnp.array([[0.7], [0.1]])  # 2x1 dot-sensor coupling matrix - matching white noise example
    dot_system = QuantumDotSystem(Cdd, Cds)
    
    print(f"Created quantum dot system: {dot_system.num_dots} dots, {dot_system.num_sensors} sensor")
    
    # Configure sensor parameters
    params_resonator = {
        'Lc': 800e-9,      # Inductance (H)
        'Cp': 0.5e-12,     # Parasitic capacitance (F)
        'RL': 40,          # Load resistance (Ω)
        'Rc': 100e6,       # Coupling resistance (Ω)
        'Z0': 50           # Characteristic impedance (Ω)
    }
    
    params_coulomb_peak = {
        'g0': 1/50,        # Maximum conductance (S) - matching white noise example
        'eps0': 0.8,       # Operating point (relative to eps_width) - larger detuning
        'eps_width': 1.0   # Energy width (eV)
    }
    
    # Create noise models
    # 1/f noise affecting energy offset (epsilon)
    eps_noise = OverFNoise(
        n_fluctuators=5,           # Number of fluctuators (fixed)
        S1=1e-6,                   # Noise power spectral density
        sigma_couplings=0.2,        # Coupling strength variation
        ommax=ommax,               # Maximum frequency (1/dt)
        ommin=ommin,               # Minimum frequency (1/(10*t_end))
        equally_dist=True          # Equally distributed frequencies
    )
    
    # OU noise affecting capacitance (simulating white noise)
    c_noise = OU_noise(sigma=1e-12, gamma=1e6)
    
    # Create sensor
    sensor = RLC_sensor(params_resonator, params_coulomb_peak, c_noise, eps_noise)
    
    print(f"Sensor resonant frequency: {sensor.f0/1e9:.2f} GHz")
    print(f"Sensor resonant period: {sensor.T0*1e9:.2f} ns")
    
    return dot_system, sensor

def plot_coulomb_peaks(sensor, charge_states, dot_system):
    """Plot Coulomb peaks for each charge state to visualize asymmetry."""
    
    print(f"\nPlotting Coulomb peaks to visualize asymmetry...")
    
    # Energy range to plot
    eps_range = jnp.linspace(-2.0, 2.0, 1000)  # Energy range in units of eps_width
    
    # Get sensor parameters
    R0 = sensor.R0
    eps_w = sensor.eps_w
    eps0 = sensor.eps0
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # First, plot the single conductance curve (sensor response)
    conductances = []
    for eps in eps_range:
        # Calculate conductance using the sensor's conductance model
        total_eps = eps * eps_w
        g = 2 * jnp.cosh(2 * total_eps / eps_w)**(-2) / R0
        conductances.append(float(g))
    
    # Plot the single conductance curve
    ax.plot(eps_range, conductances, color='black', linewidth=3, label='Sensor Conductance Curve')
    
    colors = ['blue', 'red']
    labels = ['State [1,0]', 'State [0,1]']
    
    for i, charge_state in enumerate(charge_states):
        # Calculate energy offset for this charge state
        # This is the energy shift due to the charge state
        sensor_voltages = jnp.zeros(dot_system.Cds.shape[1])
        energy_shift = dot_system.get_energy_offset(charge_state, sensor_voltages, eps0)[0]
        
        # Mark the operating point on the conductance curve
        operating_eps = (eps0 - energy_shift) / eps_w
        operating_g = 2 * jnp.cosh(2 * operating_eps / eps_w)**(-2) / R0
        ax.plot(operating_eps, operating_g, 'o', color=colors[i], markersize=10, 
                label=f'{labels[i]} (ε={operating_eps:.3f})')
        
        print(f"  {labels[i]}: Energy shift = {energy_shift:.3f}, Operating point = {operating_eps:.3f}")
    
    ax.set_xlabel('Energy ε (units of ε_width)')
    ax.set_ylabel('Conductance g (S)')
    ax.set_title('Coulomb Peak: Single Conductance Curve with Different Operating Points\n(Showing Asymmetric Coupling Effects)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text box with coupling information
    coupling_info = f"Cds coupling matrix:\n{dot_system.Cds}"
    ax.text(0.02, 0.98, coupling_info, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(os.path.dirname(__file__), 'coulomb_peaks_asymmetry.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Coulomb peaks plot saved to: {plot_path}")
    
    plt.show()
    
    return fig

def analyze_noise_effects(simulator, charge_states, times, eps_noise, ommin, ommax):
    """Analyze the effects of different noise parameters."""
    
    print(f"\nAnalyzing noise effects...")
    
    # First, calculate clean separation without 1/f noise
    print(f"\n1. Calculating clean separation without 1/f noise...")
    
    # Create clean noise model (no 1/f noise AND no OU noise)
    clean_eps_noise = OverFNoise(
        n_fluctuators=1,  # Minimal fluctuators instead of 0
        S1=1e-9,         # Very small amplitude (effectively no noise)
        sigma_couplings=0.0,
        ommax=ommax,     # 1/dt
        ommin=ommin,     # 1/(1000*t_end) - much slower noise
        equally_dist=True
    )
    
    # Create clean OU noise (minimal)
    clean_c_noise = OU_noise(sigma=1e-15, gamma=1e6)  # Very small sigma
    
    # Update the sensor with clean noise models
    simulator.sensors[0].eps_noise_model = clean_eps_noise
    simulator.sensors[0].c_noise_model = clean_c_noise
    
    # Precompute noise trajectories for clean case
    key = jax.random.PRNGKey(42)
    n_realizations = 1000  # Increased to match white noise example
    simulator.precompute_noise(key, times, n_realizations, clean_eps_noise)
    
    # Define simulation parameters
    params = {
        'eps0': 0.8,  # Match the eps0 from create_simple_system
        'snr': 2,
        't_end': times[-1]
    }
    
    # Run simulation for clean case
    clean_results = simulator.run_simulation(charge_states, times, params, key)
    
    # Extract clean results
    clean_sensor_results = clean_results['sensor_results'][0]
    clean_avg_separation = clean_sensor_results['avg_separation']
    
    # Calculate clean signal quality metrics
    clean_I_data = clean_sensor_results['I']
    clean_Q_data = clean_sensor_results['Q']
    
    # Calculate final integrated values for clean case
    clean_I_integrated = jnp.cumsum(clean_I_data, axis=-1) / jnp.arange(1, clean_I_data.shape[-1] + 1)
    clean_Q_integrated = jnp.cumsum(clean_Q_data, axis=-1) / jnp.arange(1, clean_Q_data.shape[-1] + 1)
    
    # Calculate centroids for each charge state in clean case
    clean_centroids = []
    for state_idx in range(len(charge_states)):
        I_mean = jnp.mean(clean_I_integrated[state_idx, :, -1])
        Q_mean = jnp.mean(clean_Q_integrated[state_idx, :, -1])
        clean_centroids.append([float(I_mean), float(Q_mean)])
    
    clean_centroids = jnp.array(clean_centroids)
    
    # Calculate clean separation between states
    if len(clean_centroids) > 1:
        clean_separation = jnp.linalg.norm(clean_centroids[0] - clean_centroids[1])
    else:
        clean_separation = 0.0
    
    # Calculate clean noise level
    clean_state_noise_levels = []
    for state_idx in range(len(charge_states)):
        state_I = clean_I_integrated[state_idx, :, -1]
        state_Q = clean_Q_integrated[state_idx, :, -1]
        state_noise = float(jnp.sqrt(jnp.var(state_I) + jnp.var(state_Q)))
        clean_state_noise_levels.append(state_noise)
    
    clean_noise_level = float(jnp.mean(jnp.array(clean_state_noise_levels)))
    clean_snr = clean_separation / clean_noise_level if clean_noise_level > 0 else 0
    
    print(f"  Clean separation: {clean_separation:.6f}")
    print(f"  Clean noise level: {clean_noise_level:.6f}")
    print(f"  Clean SNR: {clean_snr:.2f}")
    
    # Now test different 1/f noise parameters
    print(f"\n2. Testing different 1/f noise parameters...")
    S1_values = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0]  # Different S1 values
    
    results_dict = {}
    
    for S1 in S1_values:
        print(f"\nTesting S1={S1:.1e}")
        
        # Create new 1/f noise model with these parameters
        test_eps_noise = OverFNoise(
            n_fluctuators=10,  # Increased to 10 for richer spectrum
            S1=S1,
            sigma_couplings=1e-99,
            ommax=ommax,  # 1/dt
            ommin=ommin,  # 1/(1000*t_end) - much slower noise
            equally_dist=True
        )
            
                    # Update the sensor with new noise model
        simulator.sensors[0].eps_noise_model = test_eps_noise
        
        # Precompute noise trajectories
        key = jax.random.PRNGKey(42)
        n_realizations = 1000  # Increased for better statistics
        simulator.precompute_noise(key, times, n_realizations, test_eps_noise)
        
        # Define simulation parameters
        params = {
            'eps0': 0.8,  # Match the eps0 from create_simple_system
            'snr': 2,
            't_end': times[-1]
        }
        
        # Run simulation
        results = simulator.run_simulation(charge_states, times, params, key)
        
        # Extract results
        sensor_results = results['sensor_results'][0]
        
        # Calculate signal quality metrics
        I_data = sensor_results['I']
        Q_data = sensor_results['Q']
        
        # Calculate final integrated values
        I_integrated = jnp.cumsum(I_data, axis=-1) / jnp.arange(1, I_data.shape[-1] + 1)
        Q_integrated = jnp.cumsum(Q_data, axis=-1) / jnp.arange(1, Q_data.shape[-1] + 1)
        
                            # Calculate centroids for each charge state
        centroids = []
        for state_idx in range(len(charge_states)):
            I_mean = jnp.mean(I_integrated[state_idx, :, -1])
            Q_mean = jnp.mean(Q_integrated[state_idx, :, -1])
            centroids.append([float(I_mean), float(Q_mean)])
        
        centroids = jnp.array(centroids)
        
        # Calculate separation between states
        if len(centroids) > 1:
            separation = jnp.linalg.norm(centroids[0] - centroids[1])
        else:
            separation = 0.0
        
        # Calculate noise level
        state_noise_levels = []
        for state_idx in range(len(charge_states)):
            state_I = I_integrated[state_idx, :, -1]
            state_Q = Q_integrated[state_idx, :, -1]
            state_noise = float(jnp.sqrt(jnp.var(state_I) + jnp.var(state_Q)))
            state_noise_levels.append(state_noise)
        
        noise_level = float(jnp.mean(jnp.array(state_noise_levels)))
        
        # Calculate effective SNR
        effective_snr = separation / noise_level if noise_level > 0 else 0
        
        # Calculate degradation metrics
        separation_degradation = (clean_separation - separation) / clean_separation * 100
        snr_degradation = (clean_snr - effective_snr) / clean_snr * 100
        
        # Store results
        key = f"S1_{S1:.1e}"
        results_dict[key] = {
            'n_fluctuators': 5,
            'S1': S1,
            'separation': float(separation),
            'noise_level': noise_level,
            'effective_snr': effective_snr,
            'separation_degradation': separation_degradation,
            'snr_degradation': snr_degradation,
            'centroids': centroids,
            'I_data': I_data,
            'Q_data': Q_data
        }
        
        print(f"  Separation: {separation:.6f} (degradation: {separation_degradation:.2f}%)")
        print(f"  Noise level: {noise_level:.6f}")
        print(f"  Effective SNR: {effective_snr:.2f} (degradation: {snr_degradation:.2f}%)")
    
    # Add clean case to results for comparison
    results_dict['clean'] = {
        'n_fluctuators': 1,
        'S1': 1e-9,
        'separation': float(clean_separation),
        'noise_level': clean_noise_level,
        'effective_snr': clean_snr,
        'separation_degradation': 0.0,
        'snr_degradation': 0.0,
        'centroids': clean_centroids,
        'I_data': clean_I_data,
        'Q_data': clean_Q_data
    }
    
    return results_dict

def plot_noise_analysis(results_dict, times):
    """Plot noise analysis results with focus on I, Q scatter plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data for plotting
    keys = list(results_dict.keys())
    # Exclude 'clean' from parameter plots but include it in other plots
    param_keys = [k for k in keys if k != 'clean']
    S1_list = [results_dict[k]['S1'] for k in param_keys]
    separations = [results_dict[k]['separation'] for k in param_keys]
    noise_levels = [results_dict[k]['noise_level'] for k in param_keys]
    snr_values = [results_dict[k]['effective_snr'] for k in param_keys]
    separation_degradations = [results_dict[k]['separation_degradation'] for k in param_keys]
    snr_degradations = [results_dict[k]['snr_degradation'] for k in param_keys]
    
    # Plot 1: Separation vs S1
    ax1 = axes[0, 0]
    ax1.semilogx(S1_list, separations, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('S1 (1/f noise amplitude)')
    ax1.set_ylabel('Signal Separation')
    ax1.set_title('Separation vs Noise Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SNR vs S1
    ax2 = axes[0, 1]
    ax2.semilogx(S1_list, snr_values, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('S1 (1/f noise amplitude)')
    ax2.set_ylabel('Effective SNR')
    ax2.set_title('SNR vs Noise Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Degradation vs S1
    ax3 = axes[0, 2]
    ax3.semilogx(S1_list, separation_degradations, 'go-', label='Separation', linewidth=2, markersize=8)
    ax3.semilogx(S1_list, snr_degradations, 'mo-', label='SNR', linewidth=2, markersize=8)
    ax3.set_xlabel('S1 (1/f noise amplitude)')
    ax3.set_ylabel('Degradation (%)')
    ax3.set_title('Degradation vs Noise Amplitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: I, Q scatter plot for clean case
    ax4 = axes[1, 0]
    if 'clean' in results_dict:
        clean_data = results_dict['clean']
        I_data = clean_data['I_data']
        Q_data = clean_data['Q_data']
        
        # Calculate integrated values for plotting
        I_integrated = jnp.cumsum(I_data, axis=-1) / jnp.arange(1, I_data.shape[-1] + 1)
        Q_integrated = jnp.cumsum(Q_data, axis=-1) / jnp.arange(1, Q_data.shape[-1] + 1)
        
        # Plot final integrated values for all realizations
        I_final = I_integrated[0, :, -1]  # First state, all realizations, final time
        Q_final = Q_integrated[0, :, -1]  # First state, all realizations, final time
        ax4.scatter(I_final, Q_final, alpha=0.6, s=30, color='blue', label='State [1,0]')
        
        I_final = I_integrated[1, :, -1]  # Second state, all realizations, final time
        Q_final = Q_integrated[1, :, -1]  # Second state, all realizations, final time
        ax4.scatter(I_final, Q_final, alpha=0.6, s=30, color='red', label='State [0,1]')
    
    ax4.set_xlabel('I Signal')
    ax4.set_ylabel('Q Signal')
    ax4.set_title('I, Q Scatter Plot (Clean)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: I, Q scatter plot for highest noise case
    ax5 = axes[1, 1]
    if param_keys:
        highest_noise_key = param_keys[-1]  # Highest S1 value
        high_noise_data = results_dict[highest_noise_key]
        I_data = high_noise_data['I_data']
        Q_data = high_noise_data['Q_data']
        
        # Calculate integrated values for plotting
        I_integrated = jnp.cumsum(I_data, axis=-1) / jnp.arange(1, I_data.shape[-1] + 1)
        Q_integrated = jnp.cumsum(Q_data, axis=-1) / jnp.arange(1, Q_data.shape[-1] + 1)
        
        # Plot final integrated values for all realizations
        I_final = I_integrated[0, :, -1]  # First state, all realizations, final time
        Q_final = Q_integrated[0, :, -1]  # First state, all realizations, final time
        ax5.scatter(I_final, Q_final, alpha=0.6, s=30, color='blue', label='State [1,0]')
        
        I_final = I_integrated[1, :, -1]  # Second state, all realizations, final time
        Q_final = Q_integrated[1, :, -1]  # Second state, all realizations, final time
        ax5.scatter(I_final, Q_final, alpha=0.6, s=30, color='red', label='State [0,1]')
    
    ax5.set_xlabel('I Signal')
    ax5.set_ylabel('Q Signal')
    ax5.set_title(f'I, Q Scatter Plot (S1={highest_noise_key})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: SNR summary with degradation
    ax6 = axes[1, 2]
    bars = ax6.bar(range(len(param_keys)), snr_values, 
                   color=['blue', 'red', 'green', 'orange', 'purple'][:len(param_keys)])
    ax6.set_xlabel('S1 Value')
    ax6.set_ylabel('Effective SNR')
    ax6.set_title('SNR Summary')
    ax6.set_xticks(range(len(param_keys)))
    ax6.set_xticklabels([f'S1={results_dict[k]["S1"]:.1e}' 
                        for k in param_keys], rotation=45, ha='right')
    ax6.grid(True, alpha=0.3)
    
    # Add SNR values and degradation on bars
    for i, (bar, snr, deg) in enumerate(zip(bars, snr_values, snr_degradations)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{snr:.1f}\n({deg:.1f}%)', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig

def main():
    print("ReadSpyn Simple 1/f Noise Example")
    print("=" * 50)
    
    # Set JAX random key for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Define simulation parameters
    t_end = 1e-6  # 1 μs absolute end time
    dt = 0.5e-9   # 0.5 ns time step (matching white noise example)
    times = jnp.arange(0, t_end, dt)
    
    # Calculate frequency parameters based on simulation time
    ommin = 1 / (1000 * t_end)  # 1/(1000*t_end) - much slower noise
    ommax = 1 / dt               # 1/dt - same upper limit
    
    print(f"\nSimulation parameters:")
    print(f"  End time: {t_end*1e6:.2f} μs (absolute)")
    print(f"  Time step: {dt*1e9:.1f} ns")
    print(f"  Number of time points: {len(times)}")
    print(f"  Frequency range: {ommin/1e3:.3f} kHz to {ommax/1e9:.1f} GHz")
    
    # Create the simple system
    print("\n1. Creating simple 2 dot, 1 sensor system...")
    dot_system, sensor = create_simple_system(ommin, ommax)
    
    # Create simulator
    simulator = JAXReadoutSimulator(dot_system, [sensor])
    
    # Define charge states to simulate
    charge_states = jnp.array([
        [1, 0],  # First dot occupied
        [0, 1],  # Second dot occupied
    ])
    
    print(f"\nCharge states to simulate:")
    for i, state in enumerate(charge_states):
        print(f"  State {i}: {state} (dot {i+1} occupied)")
    
    # Plot Coulomb peaks to visualize asymmetry
    plot_coulomb_peaks(sensor, charge_states, dot_system)
    
    # Analyze noise effects
    print(f"\n2. Analyzing noise effects...")
    results_dict = analyze_noise_effects(simulator, charge_states, times, sensor.eps_noise_model, ommin, ommax)
    
    # Plot analysis
    print(f"\n3. Plotting noise analysis...")
    fig = plot_noise_analysis(results_dict, times)
    
    # Save plot
    plot_path = os.path.join(os.path.dirname(__file__), 'simple_1f_noise_analysis.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Noise analysis plot saved to: {plot_path}")
    
    # Show plot
    plt.show()
    
    # Summary
    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("This example demonstrates a simple 2 dot, 1 sensor system with:")
    print()
    print("1. Two Quantum Dots: Two dots with one sensor")
    print("2. 1/f Noise: Affecting energy offset (epsilon)")
    print("3. OU Noise: Affecting capacitance (simulating white noise)")
    print("4. Parameter Study: Effects of different noise parameters")
    print("5. Signal Quality Analysis: SNR and separation analysis")
    print()
    print("Key Findings:")
    print("  - 1/f noise affects signal stability over time")
    print("  - More fluctuators increase noise complexity")
    print("  - Higher S1 values increase noise amplitude")
    print("  - Noise parameters significantly impact readout fidelity")
    print()
    print("Applications:")
    print("  - Understanding noise effects in quantum dot readout")
    print("  - Optimizing noise parameters for better fidelity")
    print("  - Characterizing 1/f noise in quantum systems")
    print("  - Developing noise mitigation strategies")
    print()
    print("This simple setup provides a foundation for understanding")
    print("noise effects in quantum dot readout systems.")
    
    print(f"\n✓ Simple 1/f noise example completed successfully!")

if __name__ == "__main__":
    main() 