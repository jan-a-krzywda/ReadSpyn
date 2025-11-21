"""
JAX-based Readout Simulator for ReadSpyn

This module provides the main JAXReadoutSimulator class for simulating quantum dot
readout systems with multiple sensors using JAX for efficient vectorized operations
and state scanning.
"""

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
from typing import List, Dict, Any, Optional, Tuple, Union
from functools import partial

from .quantum_dot_system import QuantumDotSystem
from .sensor_backend import RLC_sensor
from .noise_models import OU_noise, OverFNoise, CorrelatedNoise, precompute_noise_trajectories


class JAXReadoutSimulator:
    """
    JAX-based readout simulator for quantum dot systems.
    
    This class provides efficient simulation of quantum dot readout systems
    using JAX for vectorized operations and scan for state processing.
    
    Attributes:
        dot_system (QuantumDotSystem): The quantum dot system to simulate
        sensors (List[RLC_sensor]): List of sensor objects
        noise_trajectories (jax.Array): Precomputed noise trajectories
        results (Dict): Simulation results
    """
    
    def __init__(self, dot_system: QuantumDotSystem, sensors: Optional[List[RLC_sensor]] = None):
        """
        Initialize the JAXReadoutSimulator.
        
        Args:
            dot_system: The quantum dot system to simulate
            sensors: List of sensor objects. If None, default sensors will be created
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for JAXReadoutSimulator. Install JAX to use this feature.")
            
        if sensors and len(sensors) != dot_system.num_sensors:
            raise ValueError(f"Number of provided sensors ({len(sensors)}) must match "
                           f"the Cds matrix dimension ({dot_system.num_sensors}).")
        
        self.dot_system = dot_system
        self.sensors = sensors or []
        self.noise_trajectories = None
        self.results = {}
        
        # Validate sensor configuration
        if not self.sensors:
            raise ValueError("At least one sensor must be provided.")
    
    def precompute_noise(self, 
                        key: jax.random.PRNGKey,
                        times: jax.Array,
                        n_realizations: int,
                        noise_model: Union[OU_noise, OverFNoise, CorrelatedNoise]) -> None:
        """
        Precompute noise trajectories for all realizations.
        
        This generates one continuous noise trajectory and then creates
        n_realizations segments from it, ensuring continuity across states.
        
        Args:
            key: JAX PRNG key for random number generation
            times: Time array
            n_realizations: Number of noise realizations to generate
            noise_model: Noise model to use (can be correlated for multiple sensors)
        """
        print(f"Precomputing {n_realizations} noise trajectory segments...")
        
        # Check if this is correlated noise
        if isinstance(noise_model, CorrelatedNoise):
            print(f"Using correlated noise for {noise_model.n_sensors} sensors")
            # Generate one long continuous trajectory for all sensors
            extended_times = jnp.arange(0, times[-1] * n_realizations, times[1] - times[0])
            long_trajectory = noise_model.generate_trajectory(key, extended_times)
            
            # Split into segments for different realizations
            segment_length = len(times)
            segments = []
            for i in range(n_realizations):
                start_idx = i * segment_length
                end_idx = (i + 1) * segment_length
                if end_idx <= long_trajectory.shape[1]:  # Check time dimension
                    segments.append(long_trajectory[:, start_idx:end_idx])
                else:
                    # If we don't have enough data, pad with zeros
                    segment = long_trajectory[:, start_idx:]
                    padding = jnp.zeros((noise_model.n_sensors, segment_length - segment.shape[1]))
                    segments.append(jnp.concatenate([segment, padding], axis=1))
            
            self.noise_trajectories = jnp.array(segments)
        else:
            # Original behavior for single-sensor noise
            extended_times = jnp.arange(0, times[-1] * n_realizations, times[1] - times[0])
            long_trajectory = noise_model.generate_trajectory(key, extended_times)
            
            # Split into segments for different realizations
            segment_length = len(times)
            segments = []
            for i in range(n_realizations):
                start_idx = i * segment_length
                end_idx = (i + 1) * segment_length
                if end_idx <= len(long_trajectory):
                    segments.append(long_trajectory[start_idx:end_idx])
                else:
                    # If we don't have enough data, pad with zeros
                    segment = long_trajectory[start_idx:]
                    padding = jnp.zeros(segment_length - len(segment))
                    segments.append(jnp.concatenate([segment, padding]))
            
            self.noise_trajectories = jnp.array(segments)
        
        print("Noise trajectory segments precomputed successfully.")
    
    def run_simulation(self, 
                      charge_states: jax.Array,
                      times: jax.Array,
                      params: Dict[str, Any],
                      key: jax.random.PRNGKey) -> Dict[str, Any]:
        """
        Run the simulation for all charge states using JAX scan.
        
        Args:
            charge_states: Array of charge states to simulate (n_states, n_dots)
            times: Time array
            params: Dictionary containing simulation parameters
                - snr: Signal-to-noise ratio for white noise (optional)
                - t_end: End time for SNR calculation (optional)
            key: JAX PRNG key for random number generation
            
        Returns:
            Dict: Simulation results
        """
        if self.noise_trajectories is None:
            raise ValueError("Noise trajectories must be precomputed before running simulation.")
        
        n_states = charge_states.shape[0]
        n_realizations = self.noise_trajectories.shape[0]
        
        print(f"Running simulation for {n_states} charge states with {n_realizations} realizations...")
        
        # Initialize results dictionary
        self.results = {
            'charge_states': charge_states,
            'times': times,
            'sensor_results': []
        }
        
        # Process each sensor
        for sensor_idx, sensor in enumerate(self.sensors):
            print(f"Processing sensor {sensor_idx}...")
            
            # Calculate energy offsets for all charge states
            energy_offsets = self._calculate_energy_offsets(charge_states, sensor, sensor_idx)
            
            # Calculate conductance values
            conductance_values = self._calculate_conductance(energy_offsets, sensor)
            
            # Calculate average separation for white noise amplitude
            avg_separation = self.calculate_average_separation(charge_states, sensor_idx)
            
            # Add separation and SNR parameters to params
            sensor_params = params.copy()
            sensor_params['avg_separation'] = avg_separation
            sensor_params['t_end'] = params.get('t_end', times[-1])
            sensor_params['snr'] = params.get('snr', 1.0)
            
            print(f"Sensor {sensor_idx}: Average separation = {avg_separation:.6f}")
            
            # Run simulation for this sensor using JAX scan
            sensor_results = self._simulate_sensor(
                sensor, sensor_idx, charge_states, times, sensor_params, key
            )
            
            # Add sensor-specific results
            sensor_results.update({
                'energy_offsets': energy_offsets,
                'conductance_values': conductance_values,
                'avg_separation': avg_separation
            })
            
            self.results['sensor_results'].append(sensor_results)
        
        print("Simulation completed successfully.")
        return self.results
    
    def _calculate_energy_offsets(self, 
                                 charge_states: jax.Array,
                                 sensor: RLC_sensor,
                                 sensor_idx: int) -> jax.Array:
        """
        Calculate energy offsets for all charge states.
        
        Args:
            charge_states: Array of charge states
            sensor: Sensor object
            sensor_idx: Sensor index
            
        Returns:
            jax.Array: Energy offsets for each charge state
        """
        def get_offset(charge_state):
            return self.dot_system.get_energy_offset(
                charge_state, jnp.zeros(self.dot_system.num_sensors), sensor.eps0
            )[sensor_idx]
        
        return jax.vmap(get_offset)(charge_states)
    
    def _calculate_conductance(self, 
                              energy_offsets: jax.Array,
                              sensor: RLC_sensor) -> jax.Array:
        """
        Calculate conductance values for energy offsets.
        
        Args:
            energy_offsets: Energy offset array
            sensor: Sensor object
            
        Returns:
            jax.Array: Conductance values
        """
        def conductance_fun(eps):
            return 2 * jnp.cosh(2 * eps / sensor.eps_w)**(-2) / sensor.R0
        
        return jax.vmap(conductance_fun)(energy_offsets)
    
    def _simulate_sensor(self,
                        sensor: RLC_sensor,
                        sensor_idx: int,
                        charge_states: jax.Array,
                        times: jax.Array,
                        params: Dict[str, Any],
                        key: jax.random.PRNGKey) -> Dict[str, Any]:
        """
        Simulate a single sensor for all charge states and noise realizations.
        
        Args:
            sensor: Sensor object
            sensor_idx: Sensor index
            charge_states: Array of charge states
            times: Time array
            params: Simulation parameters
            key: JAX PRNG key
            
        Returns:
            Dict: Sensor simulation results
        """
        n_states = charge_states.shape[0]
        n_realizations = self.noise_trajectories.shape[0]
        n_times = len(times)
        
        # Define the simulation step function
        def simulation_step(carry, inputs):
            state_idx, noise_idx = inputs
            
            # Get charge state and noise trajectory
            charge_state = charge_states[state_idx]
            noise_trajectory = self.noise_trajectories[noise_idx]
            
            # Handle correlated noise: extract trajectory for this specific sensor
            if noise_trajectory.ndim > 1:  # Correlated noise case
                sensor_noise_trajectory = noise_trajectory[sensor_idx]
            else:  # Single sensor case
                sensor_noise_trajectory = noise_trajectory
            
            # Get sensor signal
            # Create a unique key for this realization
            realization_key = jax.random.fold_in(key, noise_idx)
            I, Q, V_refl_t = sensor.get_signal_jax(
                times, self.dot_system, charge_state, sensor_idx, params, sensor_noise_trajectory, realization_key
            )
            
            # Store results
            result = {
                'state_idx': state_idx,
                'noise_idx': noise_idx,
                'charge_state': charge_state,
                'I': I,
                'Q': Q,
                'V_refl_t': V_refl_t,
                'noise_trajectory': sensor_noise_trajectory
            }
            
            return carry, result
        
        # Create input pairs for all state-noise combinations
        # Each noise realization should be used for all charge states
        # So we organize as: (noise_idx, state_idx) pairs
        noise_indices = jnp.arange(n_realizations)
        state_indices = jnp.arange(n_states)
        
        # Create pairs where each noise realization is used for all states
        inputs = []
        for noise_idx in range(n_realizations):
            for state_idx in range(n_states):
                inputs.append([state_idx, noise_idx])
        inputs = jnp.array(inputs)
        
        # Run simulation using scan
        init_carry = None
        _, results = jax.lax.scan(simulation_step, init_carry, inputs)
        
        # Reshape results
        # The results are organized as: [noise_idx * n_states, ...]
        # We need to reshape to: [n_states, n_realizations, ...]
        results_reshaped = {}
        for key in results.keys():
            if key in ['state_idx', 'noise_idx']:
                # Reshape to [n_realizations, n_states] then transpose to [n_states, n_realizations]
                temp = results[key].reshape(n_realizations, n_states)
                results_reshaped[key] = temp.T
            else:
                # Reshape to [n_realizations, n_states, ...] then transpose to [n_states, n_realizations, ...]
                temp = results[key].reshape(n_realizations, n_states, -1)
                results_reshaped[key] = temp.transpose(1, 0, *range(2, temp.ndim))
        
        return results_reshaped
    
    def get_integrated_IQ(self, sensor_idx: int = 0) -> Tuple[jax.Array, jax.Array]:
        """
        Get integrated IQ data for a specific sensor.
        
        Args:
            sensor_idx: Index of the sensor
            
        Returns:
            Tuple[jax.Array, jax.Array]: (integrated_I, integrated_Q)
        """
        if not self.results or 'sensor_results' not in self.results:
            raise ValueError("No simulation results available.")
        
        sensor_results = self.results['sensor_results'][sensor_idx]
        I_data = sensor_results['I']  # Shape: (n_states, n_realizations, n_times)
        Q_data = sensor_results['Q']
        
        # Calculate cumulative averages
        I_integrated = jnp.cumsum(I_data, axis=-1) / jnp.arange(1, I_data.shape[-1] + 1)
        Q_integrated = jnp.cumsum(Q_data, axis=-1) / jnp.arange(1, Q_data.shape[-1] + 1)
        
        return I_integrated, Q_integrated
    
    def calculate_fidelity(self, sensor_idx: int = 0, method: str = 'iq_separation') -> float:
        """
        Calculate readout fidelity for a specific sensor.
        
        Args:
            sensor_idx: Index of the sensor
            method: Method for fidelity calculation ('iq_separation' or 'overlap')
            
        Returns:
            float: Readout fidelity (0-1)
        """
        if not self.results or 'sensor_results' not in self.results:
            return 0.0
        
        I_integrated, Q_integrated = self.get_integrated_IQ(sensor_idx)
        
        # Use final integrated values
        I_final = I_integrated[:, :, -1]  # Shape: (n_states, n_realizations)
        Q_final = Q_integrated[:, :, -1]
        
        # Calculate centroids for each charge state
        n_states = I_final.shape[0]
        centroids = []
        
        for state_idx in range(n_states):
            I_mean = jnp.mean(I_final[state_idx])
            Q_mean = jnp.mean(Q_final[state_idx])
            centroids.append(jnp.array([I_mean, Q_mean]))
        
        centroids = jnp.array(centroids)
        
        if method == 'iq_separation':
            # Calculate separation between states
            if n_states >= 2:
                separation = jnp.linalg.norm(centroids[0] - centroids[1])
                # Simple fidelity estimate based on separation
                return jnp.minimum(1.0, separation / 10.0)
        
        elif method == 'overlap':
            # Calculate overlap between state distributions
            if n_states >= 2:
                # Calculate standard deviations
                I_std = jnp.std(I_final, axis=1)
                Q_std = jnp.std(Q_final, axis=1)
                
                # Simple overlap calculation
                overlap = jnp.exp(-0.5 * jnp.sum((centroids[0] - centroids[1])**2 / 
                                                (I_std[0]**2 + I_std[1]**2 + Q_std[0]**2 + Q_std[1]**2)))
                return 1.0 - overlap
        
        return 0.0
    
    def get_sensor_results(self, sensor_idx: int) -> Dict[str, Any]:
        """
        Get results for a specific sensor.
        
        Args:
            sensor_idx: Index of the sensor
            
        Returns:
            Dict: Results for the specified sensor
        """
        if not self.results or 'sensor_results' not in self.results:
            return {}
        
        return self.results['sensor_results'][sensor_idx]
    
    def get_charge_state_results(self, state_idx: int) -> Dict[str, Any]:
        """
        Get results for a specific charge state across all sensors.
        
        Args:
            state_idx: Index of the charge state
            
        Returns:
            Dict: Results for the specified charge state
        """
        if not self.results or 'sensor_results' not in self.results:
            return {}
        
        results = {}
        for sensor_idx, sensor_results in enumerate(self.results['sensor_results']):
            sensor_state_results = {}
            for key, value in sensor_results.items():
                if isinstance(value, jax.Array) and value.ndim >= 2:
                    sensor_state_results[key] = value[state_idx]
                else:
                    sensor_state_results[key] = value
            results[f'sensor_{sensor_idx}'] = sensor_state_results
        
        return results
    
    def calculate_average_separation(self, charge_states: jax.Array, sensor_idx: int = 0, t_end: float = None) -> float:
        """
        Calculate the average separation d_ij between charge states in IQ space.
        
        This method simulates the system with intrinsic noise (epsilon and c) to compute
        the separation between different charge states in the IQ plane after integration,
        which is then used to determine the amplitude of white noise to be added.
        
        Args:
            charge_states: Array of charge states to compare (n_states, n_dots)
            sensor_idx: Index of the sensor to analyze
            t_end: End time for the separation calculation (should match main simulation)
            
        Returns:
            float: Average separation <d_ij> between charge states in IQ space
        """
        if charge_states.shape[0] < 2:
            raise ValueError("At least 2 charge states are required for separation calculation")
        
        # We need to run a quick simulation to get the IQ separations
        # Use the same time duration as the main simulation for consistency
        sensor = self.sensors[sensor_idx]
        if t_end is None:
            t_end = 1000 * sensor.T0  # Default fallback
        dt = 0.5e-9
        times = jnp.arange(0, t_end, dt)
        
        # Create a simple noise trajectory (zeros for this calculation)
        noise_trajectory = jnp.zeros_like(times)
        
        # Calculate IQ signals for each charge state
        iq_points = []
        for i, charge_state in enumerate(charge_states):
            # Get signal without additional white noise
            params = {'eps0': 0.0, 'avg_separation': 0.0, 'snr': 1.0, 't_end': t_end}
            key = jax.random.PRNGKey(42)
            I, Q, _ = sensor.get_signal_jax(
                times, self.dot_system, charge_state, sensor_idx, params, noise_trajectory, key
            )
            
            # Integrate the signals
            I_integrated = jnp.cumsum(I) / jnp.arange(1, len(I) + 1)
            Q_integrated = jnp.cumsum(Q) / jnp.arange(1, len(Q) + 1)
            
            # Use final integrated values
            iq_points.append([I_integrated[-1], Q_integrated[-1]])
        
        iq_points = jnp.array(iq_points)
        
        # Calculate separations between all pairs of states in IQ space
        separations = []
        n_states = len(charge_states)
        for i in range(n_states):
            for j in range(i + 1, n_states):
                # Separation is the Euclidean distance in IQ space
                separation = jnp.linalg.norm(iq_points[i] - iq_points[j])
                separations.append(separation)
        
        # Return average separation
        if separations:
            return float(jnp.mean(jnp.array(separations)))
        else:
            return 0.0 