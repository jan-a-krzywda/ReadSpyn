"""
RLC Sensor Backend Module

This module provides the RLC_sensor class for simulating resonator-based sensors
used in quantum dot readout systems, with both NumPy and JAX implementations.
"""

import numpy as np
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Create dummy jax and jnp for when JAX is not available
    class DummyJAX:
        def __getattr__(self, name):
            raise ImportError("JAX is not available. Install JAX to use JAX features.")
    jax = DummyJAX()
    jnp = DummyJAX()
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from numba import njit
from typing import Dict, Any, Optional, Tuple
from functools import partial

from .quantum_dot_system import QuantumDotSystem
from .noise_models import OU_noise


@njit
def conductance_fun_numba(eps: float, w0: float, R0_val: float) -> float:
    """
    Calculate conductance using a Numba-optimized function.
    
    Args:
        eps: Energy offset
        w0: Energy width
        R0_val: Base resistance
        
    Returns:
        float: Conductance value
    """
    return 2 * np.cosh(2 * eps / w0)**(-2) / R0_val


@njit
def rlc_ode_system_numba(t: float, y: np.ndarray, Lc: float, Cp_t: float, 
                         RL_eff: float, Rc: float, eps_val: float, 
                         V_s_val: float, w0: float, R0: float) -> np.ndarray:
    """
    Numba-optimized RLC circuit ODE system.
    
    Args:
        t: Time
        y: State vector [v_Cp, i_L]
        Lc: Inductance
        Cp_t: Capacitance (can be time-dependent)
        RL_eff: Effective resistance
        Rc: Coupling resistance
        eps_val: Energy offset
        V_s_val: Source voltage
        w0: Energy width
        R0: Base resistance
        
    Returns:
        np.ndarray: Derivatives [dv_Cp_dt, di_L_dt]
    """
    v_Cp, i_L = y
    G_s_t = conductance_fun_numba(eps_val, w0, R0)

    if np.abs(Rc) < 1e-9:
        v_A = v_Cp 
        dv_Cp_dt = (i_L - v_Cp * G_s_t) / Cp_t
    else:
        v_A = (1 / (1 + Rc * G_s_t)) * (Rc * i_L + v_Cp)
        dv_Cp_dt = (v_A - v_Cp) / (Rc * Cp_t)
    
    di_L_dt = (V_s_val - RL_eff * i_L - v_A) / Lc
    
    return np.array([dv_Cp_dt, di_L_dt])


class RLC_sensor:
    """
    Represents a resonator sensor with an RLC circuit.
    
    This class simulates the behavior of an RLC resonator used for quantum dot
    readout, including noise effects and signal processing.
    
    Attributes:
        Lc (float): Inductance of the resonator (H)
        Cp (float): Parasitic capacitance (F)
        RL (float): Load resistance (Ω)
        Rc (float): Coupling resistance (Ω)
        Z0 (float): Characteristic impedance (Ω)
        R0 (float): Base resistance for conductance (Ω)
        eps_w (float): Energy width (eV)
        eps0 (float): Operating point energy offset (eV)
        omega0 (float): Resonant angular frequency (rad/s)
        f0 (float): Resonant frequency (Hz)
        T0 (float): Resonant period (s)
    """
    
    def __init__(self, params_resonator: Dict[str, float], 
                 params_coulomb_peak: Dict[str, float],
                 c_noise_model: Optional[OU_noise] = None, 
                 eps_noise_model: Optional[OU_noise] = None):
        """
        Initialize the RLC sensor.
        
        Args:
            params_resonator: Dictionary containing resonator parameters
                - Lc: Inductance (H)
                - Cp: Parasitic capacitance (F)
                - RL: Load resistance (Ω)
                - Rc: Coupling resistance (Ω)
                - Z0: Characteristic impedance (Ω)
                - self_capacitance: Additional self-capacitance (F)
            params_coulomb_peak: Dictionary containing Coulomb peak parameters
                - g0: Maximum conductance (S)
                - eps0: Operating point (relative to eps_width)
                - eps_width: Energy width (eV)
            c_noise_model: Capacitance noise model
            eps_noise_model: Energy offset noise model
        """
        # Extract resonator parameters with defaults
        self.Lc = params_resonator.get('Lc', 800e-9)
        self.Cp = params_resonator.get('Cp', 0.6e-12)
        self.RL = params_resonator.get('RL', 40)
        self.Rc = params_resonator.get('Rc', 100e6)
        self.Z0 = params_resonator.get('Z0', 50.0)
        self.self_capacitance = params_resonator.get('self_capacitance', 0)
        self.C_total = self.Cp + self.self_capacitance

        # Extract Coulomb peak parameters
        self.R0 = 1 / params_coulomb_peak.get('g0', 1/50*1e6)
        self.eps_w = params_coulomb_peak.get('eps_width', 500e-6)
        self.eps0 = params_coulomb_peak.get('eps0', 0.0) * self.eps_w

        # Resonant frequency calculations
        self.omega0 = 1 / np.sqrt(self.Lc * self.C_total)
        self.f0 = self.omega0 / (2 * np.pi)
        self.T0 = 1 / self.f0

        # Noise models
        self.C_noise_model = c_noise_model
        self.eps_noise_model = eps_noise_model

        # Print initialization summary
        self._print_initialization_summary()

    def _print_initialization_summary(self):
        """Print a summary of the sensor initialization."""
        print(f"[RLC_sensor] Initialized with:")
        print(f"  Lc = {self.Lc:.3e} H")
        print(f"  Cp = {self.Cp:.3e} F")
        print(f"  Self-capacitance = {self.self_capacitance:.3e} F")
        print(f"  Total capacitance = {self.C_total:.3e} F")
        print(f"  RL = {self.RL} Ω")
        print(f"  Rc = {self.Rc:.3e} Ω")
        print(f"  Z0 = {self.Z0} Ω")
        print(f"  R0 = {self.R0:.3e} Ω")
        print(f"  g0 = {1/self.R0:.3e} S")
        print(f"  eps_w = {self.eps_w:.3e} eV")
        print(f"  Resonant frequency = {self.f0:.3e} Hz")
        print(f"  Resonant period = {self.T0:.3e} s")
        
        if self.C_noise_model:
            print(f"  Capacitance noise model: {type(self.C_noise_model).__name__}")
        else:
            print("  Capacitance noise model: None")
            
        if self.eps_noise_model:
            print(f"  Energy noise model: {type(self.eps_noise_model).__name__}")
        else:
            print("  Energy noise model: None")

    def calculate_meaningful_snr(self, dot_system: QuantumDotSystem, 
                               charge_states: list, sensor_index: int = 0) -> float:
        """
        Calculate a meaningful SNR based on conductance difference between charge states.
        
        SNR = |G(state1) - G(state2)| / mean(G(state1), G(state2))
        
        Args:
            dot_system: Quantum dot system
            charge_states: List of charge state arrays to compare (typically 2 states)
            sensor_index: Index of the sensor to analyze
            
        Returns:
            float: SNR as the ratio of conductance difference to average conductance
        """
        if len(charge_states) < 2:
            raise ValueError("At least 2 charge states are required for SNR calculation")
        
        # Calculate energy offsets for each charge state
        sensor_voltages = np.zeros(dot_system.Cds.shape[1])
        energy_offsets = []
        
        for charge_state in charge_states:
            energy_offset = dot_system.get_energy_offset(charge_state, sensor_voltages, self.eps0)[sensor_index]
            energy_offsets.append(energy_offset)
        
        # Calculate conductances for each charge state
        conductances = [conductance_fun_numba(eps, self.eps_w, self.R0) for eps in energy_offsets]
        
        # Calculate the conductance difference between states
        conductance_diff = abs(conductances[1] - conductances[0])
        
        # Calculate the average conductance (for normalization)
        avg_conductance = np.mean(conductances)
        
        # Return SNR as the ratio of conductance difference to average conductance
        return conductance_diff / avg_conductance

    def get_snr_details(self, dot_system: QuantumDotSystem, 
                        charge_states: list, sensor_index: int = 0) -> dict:
        """
        Get detailed SNR information including conductances and energy offsets.
        
        Args:
            dot_system: Quantum dot system
            charge_states: List of charge state arrays to compare
            sensor_index: Index of the sensor to analyze
            
        Returns:
            dict: Detailed SNR information
        """
        if len(charge_states) < 2:
            raise ValueError("At least 2 charge states are required for SNR calculation")
        
        # Calculate energy offsets for each charge state
        sensor_voltages = np.zeros(dot_system.Cds.shape[1])
        energy_offsets = []
        
        for charge_state in charge_states:
            energy_offset = dot_system.get_energy_offset(charge_state, sensor_voltages, self.eps0)[sensor_index]
            energy_offsets.append(energy_offset)
        
        # Calculate conductances for each charge state
        conductances = [conductance_fun_numba(eps, self.eps_w, self.R0) for eps in energy_offsets]
        
        # Calculate the conductance difference between states
        conductance_diff = abs(conductances[1] - conductances[0])
        
        # Calculate the average conductance (for normalization)
        avg_conductance = np.mean(conductances)
        
        # Calculate SNR
        snr = conductance_diff / avg_conductance
        
        return {
            'snr': snr,
            'energy_offsets': energy_offsets,
            'conductances': conductances,
            'conductance_difference': conductance_diff,
            'average_conductance': avg_conductance,
            'charge_states': charge_states,
            'eps_w': self.eps_w,
            'R0': self.R0
        }

    def get_signal(self, times: np.ndarray, dot_system: QuantumDotSystem,
                   charge_state: np.ndarray, sensor_index: int, params: Dict[str, Any],
                   noise_trajectory: Optional[np.ndarray] = None,
                   use_stationary_initial_state: bool = False,
                   trim_edges: bool = True,
                   edge_padding: float = 200e-9) -> tuple:
        """
        Simulate the IQ signal for a given charge state and noise trajectory.
        
        Args:
            times: Time array for simulation
            dot_system: Quantum dot system
            charge_state: Charge state vector
            sensor_index: Index of this sensor
            params: Simulation parameters
            noise_trajectory: Optional noise trajectory
            
        Returns:
            tuple: (I, Q, V_refl_t, times)
                - I: In-phase component
                - Q: Quadrature component  
                - V_refl_t: Raw reflected voltage
                - times: Time array
        """
        eps0 = params.get('eps0', 0.0) * self.eps_w
        sensor_voltages = np.zeros(dot_system.Cds.shape[1])
        energy_offset = dot_system.get_energy_offset(charge_state, sensor_voltages, eps0)[sensor_index]
        
        # Calculate meaningful SNR if charge states are provided
        SNR_white = params.get('SNR_white', 1.0)
        if 'charge_states' in params:
            # Calculate SNR based on conductance difference between charge states
            meaningful_snr = self.calculate_meaningful_snr(dot_system, params['charge_states'], sensor_index)
            SNR_eff = meaningful_snr * SNR_white  # Scale by the provided factor
        else:
            SNR_eff = params.get('SNR_eff', SNR_white)
        
        t_end = times[-1]
        
        # ── Tail padding ──────────────────────────────────────────────────────
        # Append edge_padding to the end so the Hilbert transform has no edge ringing.
        # The extra window is stripped from all outputs before returning.
        t_tail = edge_padding if trim_edges else 0.0
        if trim_edges:
            dt = times[1] - times[0]
            tail_times = np.arange(times[-1] + dt, times[-1] + t_tail + dt * 0.5, dt)
            times_full = np.concatenate([times, tail_times])
            n_tail = len(tail_times)
            # Extend noise / eps for the tail window (hold last value)
            eps_tail = np.full(n_tail, energy_offset if noise_trajectory is None
                               else noise_trajectory[-1] + energy_offset)
        else:
            times_full = times
            n_tail = 0
        t_end_full = times_full[-1]
        eps_values_core = (noise_trajectory if noise_trajectory is not None else 0) + energy_offset

        # Build eps array over the full (possibly extended) time axis
        if trim_edges and n_tail > 0:
            eps_values = np.concatenate([
                np.atleast_1d(eps_values_core) if noise_trajectory is not None
                else np.full(len(times), energy_offset),
                eps_tail,
            ])
        else:
            eps_values = eps_values_core

        # Create interpolator for energy values
        eps_interpolator = interp1d(times_full, eps_values, kind='cubic',
                                    fill_value="extrapolate", bounds_error=False)

        # Initial conditions and effective resistance
        RL_effective = self.RL + self.Z0

        # Generate capacitance noise if model is provided
        C_noise = np.zeros_like(times_full)
        if self.C_noise_model:
            for i in range(len(times)):
                dt_i = times[i] - (times[i-1] if i > 0 else 0)
                C_noise[i] = self.C_noise_model.update(dt_i)

        def v_s_source_func(t):  #TODO: input
            """Source voltage function."""
            return 1.0 * np.sin(self.omega0 * t)

        if not use_stationary_initial_state:
            y0 = [0.0, 0.0]  # [v_Cp, i_L]
            t_start = 0.0
            t_eval = times_full
        else:
            # Extend the simulation backwards by t_warmup, starting from zero.
            # The ring-up transient occurs in this prepended window and is
            # discarded from the output, leaving only the settled signal.
            t_warmup = edge_padding
            dt = times[1] - times[0]
            warmup_times = np.arange(-t_warmup, 0.0, dt)
            t_eval = np.concatenate([warmup_times, times_full])
            # Extend the eps / noise trajectory with the noiseless initial value
            eps_warmup = np.full(len(warmup_times), energy_offset)
            eps_values = np.concatenate([eps_warmup, eps_values])
            # Rebuild interpolator over the extended time array
            eps_interpolator = interp1d(t_eval, eps_values, kind='cubic',
                                        fill_value="extrapolate", bounds_error=False)
            # Extend C_noise with zeros for the warm-up window
            C_noise = np.concatenate([np.zeros(len(warmup_times)), C_noise])
            y0 = [0.0, 0.0]
            t_start = t_eval[0]

        def ode_wrapper(t, y):
            """Wrapper for the ODE system."""
            eps_val = eps_interpolator(t)
            v_s_val = v_s_source_func(t)
            
            # Get noisy capacitance
            C_noisy = self.C_total + C_noise[np.argmin(np.abs(t_eval - t))]
            
            return rlc_ode_system_numba(t, y, self.Lc, C_noisy, RL_effective, 
                                       self.Rc, eps_val, v_s_val, self.eps_w, self.R0)

        # Solve ODE system (over full extended range including tail padding)
        sol = solve_ivp(ode_wrapper, [t_start, t_end_full], y0, t_eval=t_eval,
                       method='Radau', rtol=1e-3, atol=1e-4)

        # Compute V_refl and IQ over the FULL sol (including warm-up + tail) so
        # the Hilbert transform has no edge artifact at either boundary.
        i_L_full = sol.y[1, :]
        V_s_full = v_s_source_func(sol.t)
        V_refl_full = V_s_full - self.Z0 * i_L_full - (V_s_full / 2.0)

        # Demodulate over full array (including warmup + tail) using lock-in
        # multiplication + low-pass filter.  This avoids the FFT periodicity
        # assumption of scipy.hilbert which causes ~1/T_SIM beat-frequency
        # artefacts even with zero noise.
        t_arr = sol.t
        I_raw = V_refl_full * 2.0 * np.cos(self.omega0 * t_arr)
        Q_raw = V_refl_full * (-2.0) * np.sin(self.omega0 * t_arr)

        # Low-pass filter: cutoff at f0/2 removes the 2·f0 mixing product while
        # comfortably passing the RLC envelope bandwidth (≈ RL/2πLc).
        dt_sol = t_arr[1] - t_arr[0]
        fs_sol = 1.0 / dt_sol
        fc     = self.f0 / 2.0          # cutoff well below 2·f0
        nyq    = 0.5 * fs_sol
        if fc < nyq:
            b, a   = butter(4, fc / nyq, btype='low')
            I_filt = filtfilt(b, a, I_raw)
            Q_filt = filtfilt(b, a, Q_raw)
        else:
            I_filt = I_raw
            Q_filt = Q_raw

        if use_stationary_initial_state:
            n_warmup = len(warmup_times)
            sol_t    = sol.t[n_warmup:]
            V_refl_t = V_refl_full[n_warmup:]
            I_filt   = I_filt[n_warmup:]
            Q_filt   = Q_filt[n_warmup:]
        else:
            sol_t    = sol.t
            V_refl_t = V_refl_full

        # Remove tail padding from the end
        if trim_edges and n_tail > 0:
            sol_t    = sol_t[:-n_tail]
            V_refl_t = V_refl_t[:-n_tail]
            I_filt   = I_filt[:-n_tail]
            Q_filt   = Q_filt[:-n_tail]

        I = I_filt
        Q = Q_filt


        return I, Q, V_refl_t, times
    
    def get_signal_jax(self, times: jax.Array, dot_system: QuantumDotSystem,
                      charge_state: jax.Array, sensor_index: int, params: Dict[str, Any],
                      noise_trajectory: jax.Array, key: jax.random.PRNGKey) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        JAX-compatible signal generation for a given charge state and noise trajectory.
        
        Args:
            times: Time array for simulation
            dot_system: Quantum dot system
            charge_state: Charge state vector
            sensor_index: Index of this sensor
            params: Simulation parameters
                - avg_separation: Average separation <d_ij> between charge states
                - snr: Signal-to-noise ratio for white noise
                - t_end: End time for SNR calculation
            noise_trajectory: Precomputed noise trajectory
            
        Returns:
            Tuple[jax.Array, jax.Array, jax.Array]: (I, Q, V_refl_t)
                - I: In-phase component
                - Q: Quadrature component  
                - V_refl_t: Raw reflected voltage
        """
        eps0 = params.get('eps0', 0.0) * self.eps_w
        sensor_voltages = jnp.zeros(dot_system.Cds.shape[1])
        energy_offset = dot_system.get_energy_offset(charge_state, sensor_voltages, eps0)[sensor_index]
        
        # Apply noise trajectory
        eps_values = noise_trajectory + energy_offset
        
        # Calculate conductance values
        def conductance_fun(eps):
            return 2 * jnp.cosh(2 * eps / self.eps_w)**(-2) / self.R0
        
        conductances = jax.vmap(conductance_fun)(eps_values)
        
        # Generate source voltage
        V_s_t = jnp.sin(self.omega0 * times)
        
        # Calculate reflected voltage (simplified model for JAX compatibility)
        # This is a simplified version that avoids ODE solving for efficiency
        V_refl_t = V_s_t * (1 - conductances / (conductances + self.Z0))
        
        # Extract I and Q components using simplified demodulation
        # For JAX compatibility, we use a simpler approach than Hilbert transform
        I = V_refl_t * jnp.cos(self.omega0 * times)
        Q = V_refl_t * jnp.sin(self.omega0 * times)
        
        # Apply low-pass filtering (simplified)
        def low_pass_filter(signal):
            # Simple moving average as low-pass filter
            window_size = min(10, len(signal) // 10)
            kernel = jnp.ones(window_size) / window_size
            return jnp.convolve(signal, kernel, mode='same')
        
        I = low_pass_filter(I)
        Q = low_pass_filter(Q)
        
        # Add white noise to I and Q before integration
        # According to the specification:
        # dI = x<d_ij>dt/2 + sqrt(Y) dw
        # where Y = <d_ij>/(t_end * snr)
        avg_separation = params.get('avg_separation', 0.0)
        snr = params.get('snr', 1.0)
        t_end = params.get('t_end', times[-1])
        
        if avg_separation > 0 and snr > 0:
            # Calculate time step
            dt = times[1] - times[0]
            
            
            # Calculate noise variance to get final SNR after integration
            # We want: final_SNR = signal_separation / noise_std_after_integration = snr
            # After integration: noise_std_after_integration = sqrt(Y * dt * n_points)
            # So: snr = signal_separation / sqrt(Y * dt * n_points)
            # Therefore: Y * dt * n_points = (signal_separation / snr)^2
            # This gives us: Y = (signal_separation / snr)^2 / (dt * n_points)
            # Note: Since we have both I and Q components, the total noise variance is 2*Y*dt
            # So we need to account for the √2 factor: Y = (signal_separation / snr)^2 / (2 * dt * n_points)
            dt = times[1] - times[0]
            t_end = times[-1]
            Y = (avg_separation / snr)**2 * t_end / (dt**2)/2
            
            # Generate Wiener process increments
            # Use a more varied key to ensure different noise for different realizations, charge states, and sensors
            # Include the charge state in the key to ensure different noise for different states
            charge_state_hash = jnp.sum(charge_state).astype(jnp.int32) if hasattr(charge_state, 'shape') else 0
            # Use a more unique key that includes the charge state pattern, not just the sum
            charge_state_key = jnp.sum(charge_state * jnp.arange(len(charge_state))).astype(jnp.int32)
            noise_key = jax.random.fold_in(key, jnp.sum(noise_trajectory).astype(jnp.int32) + sensor_index * 1000 + charge_state_key * 100)
            
            # Generate Wiener process increments (dw) - these should be different for each time step
            dw_I = jax.random.normal(noise_key, shape=I.shape) * jnp.sqrt(dt)
            dw_Q = jax.random.normal(jax.random.fold_in(noise_key, 1), shape=Q.shape) * jnp.sqrt(dt)
            
            # Add white noise: dI = sqrt(Y) dw
            # The signal component should come from the underlying I and Q signals, not be added separately
            noise_component_I = jnp.sqrt(Y) * dw_I
            noise_component_Q = jnp.sqrt(Y) * dw_Q
            
            I = I + noise_component_I
            Q = Q + noise_component_Q
        
        return I, Q, V_refl_t