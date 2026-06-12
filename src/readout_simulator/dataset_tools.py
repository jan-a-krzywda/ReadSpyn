"""
Dataset utilities for generating and integrating IQ readout traces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np


def integrate_iq_traces(I: np.ndarray, Q: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute running-average integrated IQ traces along a time axis.

    Args:
        I: In-phase trace array.
        Q: Quadrature trace array with the same shape as ``I``.
        axis: Time axis to integrate over.

    Returns:
        Tuple ``(I_integrated, Q_integrated)`` with the same shape as the inputs.
    """
    I = np.asarray(I)
    Q = np.asarray(Q)
    if I.shape != Q.shape:
        raise ValueError(f"I and Q must have the same shape, got {I.shape} and {Q.shape}.")

    axis = axis % I.ndim
    n_time = I.shape[axis]
    normaliser = np.arange(1, n_time + 1, dtype=float)
    shape = [1] * I.ndim
    shape[axis] = n_time
    normaliser = normaliser.reshape(shape)

    I_integrated = np.cumsum(I, axis=axis) / normaliser
    Q_integrated = np.cumsum(Q, axis=axis) / normaliser
    return I_integrated, Q_integrated


def sample_integrated_iq(I_integrated: np.ndarray, Q_integrated: np.ndarray,
                         integration_indices: Sequence[int], axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """
    Select integrated IQ points at specific integration-time indices.
    """
    indices = np.asarray(integration_indices, dtype=int)
    return (
        np.take(I_integrated, indices, axis=axis),
        np.take(Q_integrated, indices, axis=axis),
    )


def generate_sensor_iq_dataset(dot_system,
                               sensors,
                               charge_states: np.ndarray,
                               times: np.ndarray,
                               simulation_params: Optional[Dict[str, Any]] = None,
                               noise_model=None,
                               noise_trajectories: Optional[np.ndarray] = None,
                               n_realisations: int = 1,
                               integration_indices: Optional[Sequence[int]] = None,
                               use_stationary_initial_state: bool = True,
                               trim_edges: bool = True,
                               edge_padding: float = 200e-9,
                               output_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Generate a multi-sensor IQ dataset for a set of charge states.

    The same noise trajectory is reused across sensors for a given realisation,
    which is convenient for comparing sensor responses to identical detuning noise.
    """
    simulation_params = dict(simulation_params or {})
    charge_states = np.asarray(charge_states, dtype=float)
    times = np.asarray(times, dtype=float)

    n_states = len(charge_states)
    n_sensors = len(sensors)
    n_times = len(times)

    if noise_trajectories is None:
        if noise_model is None:
            shared_noise = np.zeros((n_realisations, n_times), dtype=float)
        elif hasattr(noise_model, "generate_trajectories"):
            shared_noise = np.asarray(
                noise_model.generate_trajectories(times, n=n_realisations, seed=0),
                dtype=float,
            )
        else:
            shared_noise = np.stack(
                [np.asarray(noise_model.generate_trajectory(times, seed=k), dtype=float)
                 for k in range(n_realisations)],
                axis=0,
            )
    else:
        shared_noise = np.asarray(noise_trajectories, dtype=float)
        n_realisations = shared_noise.shape[0]

    I_raw = np.zeros((n_states, n_realisations, n_sensors, n_times), dtype=float)
    Q_raw = np.zeros_like(I_raw)
    V_refl = np.zeros_like(I_raw)

    for realisation_idx in range(n_realisations):
        noise_traj = shared_noise[realisation_idx]
        for state_idx, charge_state in enumerate(charge_states):
            for sensor_idx, sensor in enumerate(sensors):
                I, Q, V, _ = sensor.get_signal(
                    times=times,
                    dot_system=dot_system,
                    charge_state=charge_state,
                    sensor_index=sensor_idx,
                    params=simulation_params,
                    noise_trajectory=noise_traj,
                    use_stationary_initial_state=use_stationary_initial_state,
                    trim_edges=trim_edges,
                    edge_padding=edge_padding,
                )
                I_raw[state_idx, realisation_idx, sensor_idx] = I
                Q_raw[state_idx, realisation_idx, sensor_idx] = Q
                V_refl[state_idx, realisation_idx, sensor_idx] = V

    I_integrated, Q_integrated = integrate_iq_traces(I_raw, Q_raw, axis=-1)

    dataset: Dict[str, np.ndarray] = {
        "times_s": times,
        "charge_states": charge_states,
        "shared_noise": shared_noise,
        "I_raw": I_raw,
        "Q_raw": Q_raw,
        "V_refl": V_refl,
        "I_integrated": I_integrated,
        "Q_integrated": Q_integrated,
    }

    if integration_indices is not None:
        indices = np.asarray(integration_indices, dtype=int)
        I_points, Q_points = sample_integrated_iq(I_integrated, Q_integrated, indices, axis=-1)
        dataset["integration_indices"] = indices
        dataset["I_points"] = I_points
        dataset["Q_points"] = Q_points

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, **dataset)

    return dataset
