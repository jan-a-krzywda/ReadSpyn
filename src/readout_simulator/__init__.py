"""
ReadSpyn - RF reflectometry readout simulator for quantum dot systems

A comprehensive simulator for quantum dot readout systems with realistic noise models,
RLC resonator sensors, and efficient time-domain simulation.
"""

from .quantum_dot_system import (
    QuantumDotSystem,
    GeometricQuantumDotSystem,
    capacitance_matrices_to_geometry,
)
from .sensor_backend import RLC_sensor
from .noise_models import OU_noise, OverFNoise, SpectrumNoise
<<<<<<< HEAD
=======
from .jax_simulator import JAXReadoutSimulator
from .dataset_tools import integrate_iq_traces, sample_integrated_iq, generate_sensor_iq_dataset
>>>>>>> b4e6f446a4938f45a85fe3a6d86fe6b056ab8cf4

__version__ = "2.0.0"
__author__ = "Jan A. Krzywda, Rouven Koch"

__all__ = [
    "QuantumDotSystem",
    "GeometricQuantumDotSystem",
<<<<<<< HEAD
    "RLC_sensor",
    "OU_noise",
    "OverFNoise",
    "SpectrumNoise",
]
=======
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
>>>>>>> b4e6f446a4938f45a85fe3a6d86fe6b056ab8cf4
