"""
ReadSpyn - JAX-based quantum dot readout simulator

A comprehensive simulator for quantum dot readout systems with realistic noise models,
RLC resonator sensors, and efficient JAX-based state scanning.
"""

from .quantum_dot_system import (
    QuantumDotSystem,
    GeometricQuantumDotSystem,
    capacitance_matrices_to_geometry,
)
from .sensor_backend import RLC_sensor
from .noise_models import OU_noise, OverFNoise, SpectrumNoise
from .jax_simulator import JAXReadoutSimulator
from .dataset_tools import integrate_iq_traces, sample_integrated_iq, generate_sensor_iq_dataset

__version__ = "2.0.0"
__author__ = "Jan A. Krzywda, Rouven Koch"

__all__ = [
    "QuantumDotSystem",
    "GeometricQuantumDotSystem",
    "capacitance_matrices_to_geometry",
    "RLC_sensor", 
    "OU_noise",
    "OverFNoise",
    "SpectrumNoise",
    "JAXReadoutSimulator",
    "integrate_iq_traces",
    "sample_integrated_iq",
    "generate_sensor_iq_dataset",
] 
