"""
Noise Models for ReadSpyn

This module provides noise models for simulating realistic quantum dot
readout systems, including Ornstein-Uhlenbeck processes and 1/f noise,
all implemented in pure NumPy (no JAX dependency).
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union


class OU_noise:
    """
    NumPy-based Ornstein-Uhlenbeck noise model.

    This class implements a continuous-time Markov process that generates
    correlated noise with exponential autocorrelation.

    Attributes:
        sigma (float): Noise amplitude
        gamma (float): Correlation rate (Hz)
        tc (float): Correlation time (1/gamma)
    """

    def __init__(self, sigma: float, gamma: float):
        """
        Initialize OU noise.

        Args:
            sigma: Noise amplitude
            gamma: Correlation rate (Hz)
        """
        self.sigma = sigma
        self.gamma = gamma
        self.tc = 1 / gamma

    def generate_trajectory(self, times: np.ndarray, seed: int = 0) -> np.ndarray:
        """
        Generate a complete noise trajectory using NumPy.

        Args:
            times: Time array
            seed: Random seed for reproducibility

        Returns:
            np.ndarray: Noise trajectory
        """
        rng = np.random.default_rng(seed)
        dt = times[1] - times[0]
        n_steps = len(times)

        # Generate Wiener process increments
        dw = rng.normal(0.0, np.sqrt(2 * self.gamma * dt), size=n_steps)

        # Initialize state
        x = rng.normal(0.0, self.sigma)

        # Generate trajectory
        trajectory = np.empty(n_steps)
        for i in range(n_steps):
            dx = -self.gamma * x * dt + self.sigma * dw[i]
            x = x + dx
            trajectory[i] = x

        return trajectory

    def get_spectrum(self, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the theoretical power spectrum.

        Args:
            times: Time array

        Returns:
            Tuple[np.ndarray, np.ndarray]: (frequencies, power_spectrum)
        """
        dt = times[1] - times[0]
        freqs = np.fft.fftfreq(len(times), dt)

        # Theoretical power spectrum for OU process
        omega = 2 * np.pi * freqs
        spectrum = (2 * self.sigma**2 * self.tc) / (1 + (omega * self.tc)**2)

        return freqs, spectrum


class SpectrumNoise:
    """
    Generate Gaussian noise with a prescribed power spectral density (PSD).

    The trajectory is built in the frequency domain:
      1. Evaluate the user-supplied PSD function on the FFT frequency grid.
      2. Draw independent complex Gaussian amplitudes scaled by sqrt(PSD/2).
      3. Enforce Hermitian symmetry so the inverse FFT is real.
      4. Normalise the result so its variance equals ``sigma**2``.

    This is numpy-only (no JAX required) so it works even without a GPU.

    Parameters
    ----------
    psd_func : callable  f(freqs_Hz) -> array_like
        Power spectral density as a function of frequency in Hz.
        Only the *shape* of the spectrum matters; overall amplitude is
        fixed by ``sigma``.
    sigma : float
        Target RMS amplitude of the trajectory in the same units as the
        energy offset (i.e. multiples of ``eps_width``).
        A value of ~0.3–1.0 produces clearly visible IQ modulation.
    """

    def __init__(self, psd_func, sigma: float):
        self.psd_func = psd_func
        self.sigma = sigma

    def generate_trajectory(self, times: np.ndarray, seed: int = 0) -> np.ndarray:
        """
        Generate one noise realisation.

        Parameters
        ----------
        times : np.ndarray
            Uniformly-spaced time array (seconds).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray  shape (len(times),)
        """
        rng = np.random.default_rng(seed)
        n   = len(times)
        dt  = times[1] - times[0]

        freqs = np.fft.rfftfreq(n, d=dt)           # positive frequencies only
        psd   = np.asarray(self.psd_func(freqs), dtype=float)
        psd   = np.where(psd < 0, 0.0, psd)        # safety: no negative power
        psd[0] = 0.0                                # zero DC component

        # Complex amplitudes:  amplitude ∝ sqrt(PSD), random phase
        amplitude = np.sqrt(psd / 2.0)
        noise_fft = amplitude * (rng.standard_normal(len(freqs))
                                 + 1j * rng.standard_normal(len(freqs)))

        traj = np.fft.irfft(noise_fft, n=n)

        # Normalise to target sigma
        std = traj.std()
        if std > 0:
            traj = traj * (self.sigma / std)

        return traj

    def generate_trajectories(self, times: np.ndarray,
                                n: int, seed: int = 0) -> np.ndarray:
        """
        Generate *n* independent realisations.

        Returns
        -------
        np.ndarray  shape (n, len(times))
        """
        return np.stack([
            self.generate_trajectory(times, seed=seed + k) for k in range(n)
        ])


class OverFNoise:
    """
    NumPy-based 1/f noise model using multiple fluctuators.

    This class implements 1/f noise by combining multiple Ornstein-Uhlenbeck
    fluctuators with different correlation times.

    Attributes:
        n_fluctuators (int): Number of fluctuators
        S1 (float): 1/f noise amplitude
        sigma_couplings (float): Coupling strength variation
        ommax (float): Maximum frequency
        ommin (float): Minimum frequency
        fluctuators (list): List of individual OU fluctuators
    """

    def __init__(self, n_fluctuators: int, S1: float, sigma_couplings: float,
                 ommax: float, ommin: float, equally_dist: bool = False):
        """
        Initialize 1/f noise.

        Args:
            n_fluctuators: Number of fluctuators
            S1: 1/f noise amplitude
            sigma_couplings: Coupling strength variation
            ommax: Maximum frequency
            ommin: Minimum frequency
            equally_dist: Whether to distribute frequencies equally
        """
        self.n_fluctuators = n_fluctuators
        self.S1 = S1
        self.sigma_couplings = sigma_couplings
        self.ommax = ommax
        self.ommin = ommin
        self.equally_dist = equally_dist

        # Create individual fluctuators
        self._create_fluctuators()

    def _create_fluctuators(self):
        """Create individual OU fluctuators with distributed parameters."""
        rng = np.random.default_rng(0)

        # Generate correlation times
        if self.equally_dist:
            # Equally distributed in log space
            log_gammas = np.linspace(np.log(self.ommin), np.log(self.ommax), self.n_fluctuators)
            gammas = np.exp(log_gammas)
        else:
            # Log-uniformly distributed
            uni = rng.uniform(0, 1, size=self.n_fluctuators)
            gammas = self.ommax * np.exp(-np.log(self.ommax / self.ommin) * uni)

        # Calculate individual noise amplitudes
        total_variance = 2 * self.S1 * np.log(self.ommax / self.ommin)
        base_sigma = np.sqrt(total_variance / self.n_fluctuators)

        # Create fluctuators
        self.fluctuators = []
        for gamma in gammas:
            sigma = base_sigma * (1 + self.sigma_couplings * rng.normal())
            # Store as dict for pure-NumPy approach (no class coupling needed)
            self.fluctuators.append({'sigma': sigma, 'gamma': gamma})

    def generate_trajectory(self, times: np.ndarray, seed: int = 0) -> np.ndarray:
        """
        Generate a complete 1/f noise trajectory.

        Args:
            times: Time array
            seed: Random seed for reproducibility

        Returns:
            np.ndarray: Combined noise trajectory
        """
        # Generate individual trajectories
        trajectories = []
        for i, fluct in enumerate(self.fluctuators):
            ou = OU_noise(sigma=fluct['sigma'], gamma=fluct['gamma'])
            traj = ou.generate_trajectory(times, seed=seed + i)
            trajectories.append(traj)

        # Sum all trajectories
        return np.sum(np.array(trajectories), axis=0)

    def get_spectrum(self, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the theoretical power spectrum.

        Args:
            times: Time array

        Returns:
            Tuple[np.ndarray, np.ndarray]: (frequencies, power_spectrum)
        """
        dt = times[1] - times[0]
        freqs = np.fft.fftfreq(len(times), dt)

        # Sum individual spectra
        total_spectrum = np.zeros_like(freqs)
        for fluct in self.fluctuators:
            ou = OU_noise(sigma=fluct['sigma'], gamma=fluct['gamma'])
            _, spectrum = ou.get_spectrum(times)
            total_spectrum += spectrum

        return freqs, total_spectrum


class CorrelatedNoise:
    """
    NumPy-based correlated noise model for multiple sensors.

    This class generates correlated noise trajectories for multiple sensors by
    applying a correlation matrix to independent noise sources.

    Attributes:
        base_noise_model: Base noise model (OU_noise or OverFNoise)
        correlation_matrix: Correlation matrix between sensors
        n_sensors: Number of sensors
    """

    def __init__(self, base_noise_model: Union[OU_noise, OverFNoise],
                 correlation_matrix: np.ndarray):
        """
        Initialize correlated noise model.

        Args:
            base_noise_model: Base noise model to use for each sensor
            correlation_matrix: Correlation matrix of shape (n_sensors, n_sensors)
                               Should be symmetric and positive semi-definite
        """
        self.base_noise_model = base_noise_model
        self.correlation_matrix = np.array(correlation_matrix)
        self.n_sensors = self.correlation_matrix.shape[0]

        # Validate correlation matrix
        if self.correlation_matrix.shape != (self.n_sensors, self.n_sensors):
            raise ValueError(
                f"Correlation matrix must be square with shape ({self.n_sensors}, {self.n_sensors})"
            )

        # Check if matrix is symmetric
        if not np.allclose(self.correlation_matrix, self.correlation_matrix.T):
            raise ValueError("Correlation matrix must be symmetric")

        # Check if matrix is positive semi-definite
        eigenvals = np.linalg.eigvals(self.correlation_matrix)
        if np.any(eigenvals < -1e-10):  # Small tolerance for numerical errors
            raise ValueError("Correlation matrix must be positive semi-definite")

        # Compute Cholesky decomposition for efficient sampling
        self.cholesky_matrix = np.linalg.cholesky(self.correlation_matrix)

    def generate_trajectory(self, times: np.ndarray, seed: int = 0) -> np.ndarray:
        """
        Generate correlated noise trajectories for all sensors.

        Args:
            times: Time array
            seed: Random seed for reproducibility

        Returns:
            np.ndarray: Correlated noise trajectories of shape (n_sensors, n_times)
        """
        # Generate independent noise trajectories for each sensor
        independent_trajectories = []
        for i in range(self.n_sensors):
            traj = self.base_noise_model.generate_trajectory(times, seed=seed + i)
            independent_trajectories.append(traj)

        # Stack trajectories: shape (n_sensors, n_times)
        independent_noise = np.stack(independent_trajectories, axis=0)

        # Apply correlation using Cholesky decomposition
        # correlated_noise = L @ independent_noise
        # where L is the Cholesky factor of the correlation matrix
        correlated_noise = self.cholesky_matrix @ independent_noise

        return correlated_noise

    def get_spectrum(self, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the theoretical power spectrum for correlated noise.

        Args:
            times: Time array

        Returns:
            Tuple[np.ndarray, np.ndarray]: (frequencies, power_spectrum)
        """
        # Get spectrum from base noise model
        freqs, base_spectrum = self.base_noise_model.get_spectrum(times)

        # The correlation doesn't change the power spectrum of individual sensors
        # but creates cross-correlations between them
        return freqs, base_spectrum


def precompute_noise_trajectories(
    noise_model: Union[OU_noise, OverFNoise, CorrelatedNoise],
    times: np.ndarray,
    n_realizations: int,
    base_seed: int = 0
) -> np.ndarray:
    """
    Precompute multiple noise trajectory realizations.

    Args:
        noise_model: Noise model to use
        times: Time array
        n_realizations: Number of realizations to generate
        base_seed: Base seed for reproducibility

    Returns:
        np.ndarray: Array of shape (n_realizations, n_times) or
                    (n_realizations, n_sensors, n_times)
                    containing noise trajectories
    """
    trajectories = []
    for k in range(n_realizations):
        traj = noise_model.generate_trajectory(times, seed=base_seed + k)
        trajectories.append(traj)

    return np.array(trajectories)