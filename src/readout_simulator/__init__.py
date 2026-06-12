"""
ReadSpyn - RF reflectometry readout simulator for quantum dot systems

A comprehensive simulator for quantum dot readout systems with realistic noise models,
RLC resonator sensors, and efficient time-domain simulation.
"""

from .quantum_dot_system import QuantumDotSystem, GeometricQuantumDotSystem
from .sensor_backend import RLC_sensor
from .noise_models import OU_noise, OverFNoise, SpectrumNoise

__version__ = "2.0.0"
__author__ = "Jan A. Krzywda, Rouven Koch"

__all__ = [
    "QuantumDotSystem",
    "GeometricQuantumDotSystem",
    "RLC_sensor",
    "OU_noise",
    "OverFNoise",
    "SpectrumNoise",
]
