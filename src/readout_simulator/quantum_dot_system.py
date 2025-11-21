"""
Quantum Dot System Module

This module provides the QuantumDotSystem class for modeling systems of quantum
dots with capacitive coupling and sensor interactions.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
import matplotlib.pyplot as plt


class GeometricQuantumDotSystem:
    """
    A quantum dot system defined by geometric positions of dots and sensors.
    
    This class allows users to specify the positions of quantum dots and sensors
    in 2D or 3D space, and automatically computes capacitance matrices based on
    geometric considerations.
    
    Attributes:
        dot_positions (np.ndarray): Positions of quantum dots (Ndots × 2 or Ndots × 3)
        sensor_positions (np.ndarray): Positions of sensors (Nsensors × 2 or Nsensors × 3)
        dimensions (int): Number of spatial dimensions (2 or 3)
        num_dots (int): Number of quantum dots
        num_sensors (int): Number of sensors
        Cdd (np.ndarray): Computed dot-dot capacitance matrix
        Cds (np.ndarray): Computed dot-sensor capacitance matrix
        dot_system (QuantumDotSystem): The underlying quantum dot system
    """
    
    def __init__(self, dot_positions: np.ndarray, sensor_positions: np.ndarray,
                 C0: float = 1e-15, alpha: float = 1.0, beta: float = 0.1):
        """
        Initialize the geometric quantum dot system.
        
        Args:
            dot_positions: Array of dot positions (Ndots × 2 or Ndots × 3)
            sensor_positions: Array of sensor positions (Nsensors × 2 or Nsensors × 3)
            C0: Base capacitance scale (default: 1 fF)
            alpha: Coupling strength parameter (default: 1.0)
            beta: Distance decay parameter (default: 0.1)
            
        Raises:
            ValueError: If position arrays have inconsistent dimensions
        """
        self.dot_positions = np.array(dot_positions)
        self.sensor_positions = np.array(sensor_positions)
        
        # Check dimensions
        if self.dot_positions.shape[1] != self.sensor_positions.shape[1]:
            raise ValueError("Dot and sensor positions must have the same number of dimensions")
        
        self.dimensions = self.dot_positions.shape[1]
        if self.dimensions not in [2, 3]:
            raise ValueError("Positions must be 2D or 3D")
        
        self.num_dots = self.dot_positions.shape[0]
        self.num_sensors = self.sensor_positions.shape[0]
        self.C0 = C0
        self.alpha = alpha
        self.beta = beta
        
        # Compute capacitance matrices
        self.Cdd = self._compute_dot_dot_capacitance()
        self.Cds = self._compute_dot_sensor_capacitance()
        
        # Create the underlying quantum dot system
        from .quantum_dot_system import QuantumDotSystem
        self.dot_system = QuantumDotSystem(self.Cdd, self.Cds)
    
    def _compute_dot_dot_capacitance(self) -> np.ndarray:
        """
        Compute dot-dot capacitance matrix based on geometric distances.
        
        Returns:
            np.ndarray: Dot-dot capacitance matrix (Ndots × Ndots)
        """
        Cdd = np.zeros((self.num_dots, self.num_dots))
        
        for i in range(self.num_dots):
            for j in range(self.num_dots):
                if i == j:
                    # Self-capacitance (diagonal elements)
                    Cdd[i, j] = self.C0
                else:
                    # Mutual capacitance (off-diagonal elements)
                    distance = np.linalg.norm(self.dot_positions[i] - self.dot_positions[j])
                    # Use inverse distance law with exponential decay
                    Cdd[i, j] = -self.alpha * self.C0 * np.exp(-self.beta * distance)
        
        return Cdd
    
    def _compute_dot_sensor_capacitance(self) -> np.ndarray:
        """
        Compute dot-sensor capacitance matrix based on geometric distances.
        
        Returns:
            np.ndarray: Dot-sensor capacitance matrix (Ndots × Nsensors)
        """
        Cds = np.zeros((self.num_dots, self.num_sensors))
        
        for i in range(self.num_dots):
            for j in range(self.num_sensors):
                distance = np.linalg.norm(self.dot_positions[i] - self.sensor_positions[j])
                # Use inverse distance law with exponential decay
                Cds[i, j] = -self.alpha * self.C0 * np.exp(-self.beta * distance)
        
        return Cds
    
    def plot_system(self, figsize: Tuple[int, int] = (10, 8), 
                   show_capacitances: bool = True, 
                   capacitance_threshold: float = 0.01) -> None:
        """
        Plot the geometric layout of dots and sensors.
        
        Args:
            figsize: Figure size (width, height)
            show_capacitances: Whether to show capacitance connections
            capacitance_threshold: Minimum capacitance to show connection
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot dots
        ax.scatter(self.dot_positions[:, 0], self.dot_positions[:, 1], 
                  c='red', s=200, label='Quantum Dots', zorder=3)
        
        # Plot sensors
        ax.scatter(self.sensor_positions[:, 0], self.sensor_positions[:, 1], 
                  c='blue', s=150, marker='s', label='Sensors', zorder=3)
        
        # Add labels
        for i, pos in enumerate(self.dot_positions):
            ax.annotate(f'D{i}', (pos[0], pos[1]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=12, fontweight='bold')
        
        for i, pos in enumerate(self.sensor_positions):
            ax.annotate(f'S{i}', (pos[0], pos[1]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=12, fontweight='bold')
        
        # Show capacitance connections if requested
        if show_capacitances:
            # Dot-dot connections
            for i in range(self.num_dots):
                for j in range(i + 1, self.num_dots):
                    if abs(self.Cdd[i, j]) > capacitance_threshold * self.C0:
                        ax.plot([self.dot_positions[i, 0], self.dot_positions[j, 0]],
                               [self.dot_positions[i, 1], self.dot_positions[j, 1]], 
                               'r--', alpha=0.5, linewidth=1)
            
            # Dot-sensor connections
            for i in range(self.num_dots):
                for j in range(self.num_sensors):
                    if abs(self.Cds[i, j]) > capacitance_threshold * self.C0:
                        ax.plot([self.dot_positions[i, 0], self.sensor_positions[j, 0]],
                               [self.dot_positions[i, 1], self.sensor_positions[j, 1]], 
                               'b--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Geometric Quantum Dot System Layout')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    def get_coupling_info(self) -> dict:
        """
        Get information about the computed coupling strengths.
        
        Returns:
            dict: Dictionary with coupling information
        """
        coupling_matrix = self.dot_system.get_coupling_matrix()
        
        return {
            'dot_dot_capacitances': self.Cdd,
            'dot_sensor_capacitances': self.Cds,
            'coupling_matrix': coupling_matrix,
            'avg_dot_dot_coupling': np.mean(np.abs(self.Cdd[self.Cdd != 0])),
            'avg_dot_sensor_coupling': np.mean(np.abs(self.Cds)),
            'max_coupling': np.max(np.abs(coupling_matrix)),
            'min_coupling': np.min(np.abs(coupling_matrix))
        }
    
    def __repr__(self) -> str:
        """String representation of the geometric quantum dot system."""
        return (f"GeometricQuantumDotSystem(num_dots={self.num_dots}, "
                f"num_sensors={self.num_sensors}, dimensions={self.dimensions})")
    
    def __str__(self) -> str:
        """Detailed string representation."""
        coupling_info = self.get_coupling_info()
        return (f"GeometricQuantumDotSystem:\n"
                f"  Number of dots: {self.num_dots}\n"
                f"  Number of sensors: {self.num_sensors}\n"
                f"  Dimensions: {self.dimensions}D\n"
                f"  Base capacitance (C0): {self.C0:.2e} F\n"
                f"  Coupling strength (α): {self.alpha}\n"
                f"  Distance decay (β): {self.beta}\n"
                f"  Average dot-dot coupling: {coupling_info['avg_dot_dot_coupling']:.2e} F\n"
                f"  Average dot-sensor coupling: {coupling_info['avg_dot_sensor_coupling']:.2e} F\n"
                f"  Max coupling strength: {coupling_info['max_coupling']:.3f}\n"
                f"  Min coupling strength: {coupling_info['min_coupling']:.3f}")


class QuantumDotSystem:
    """
    Manages a system of quantum dots with a constant interaction model.
    
    This class represents a system of quantum dots characterized by their
    mutual capacitances and coupling to sensors.
    
    Attributes:
        num_dots (int): Number of quantum dots in the system
        num_sensors (int): Number of sensors coupled to the system
        Cdd_inv (np.ndarray): Inverse of dot-dot capacitance matrix
        Cds (np.ndarray): Dot-sensor capacitance matrix
    """
    
    def __init__(self, Cdd: np.ndarray, Cds: np.ndarray):
        """
        Initialize the quantum dot system.
        
        Args:
            Cdd: Dot-dot capacitance matrix (Ndots × Ndots)
            Cds: Dot-sensor capacitance matrix (Ndots × Nsensors)
            
        Raises:
            ValueError: If matrix dimensions are inconsistent
        """
        if Cdd.shape[0] != Cdd.shape[1] or Cdd.shape[0] != Cds.shape[0]:
            raise ValueError(f"Inconsistent dimensions: Cdd is {Cdd.shape}, Cds is {Cds.shape}. "
                           f"Cdd must be square and Cds must have same number of rows.")
        
        self.num_dots = Cdd.shape[0]
        self.num_sensors = Cds.shape[1]
        self.Cdd_inv = np.linalg.inv(Cdd)
        self.Cds = Cds
        
        # Print coupling information
        coupling_strength = Cds.T @ self.Cdd_inv
        print(f"Dot-sensor coupling strength (Δε/ε_w): {coupling_strength}")

    @classmethod
    def from_random(cls, num_dots: int, num_sensors: int, 
                   Css: float = 1e-15) -> 'QuantumDotSystem':
        """
        Generate a QuantumDotSystem with random, experimentally relevant parameters.
        
        Args:
            num_dots: Number of quantum dots
            num_sensors: Number of sensors
            Css: Scale factor for capacitances (default: 1 fF)
            
        Returns:
            QuantumDotSystem: New instance with random parameters
        """
        # Generate diagonal elements (self-capacitances)
        Cdd_diag = np.random.uniform(8, 12, num_dots) * Css
        
        # Generate off-diagonal elements (mutual capacitances)
        Cdd_off_diag = np.random.uniform(-0.2, -0.1, (num_dots, num_dots)) * Css
        
        # Construct full capacitance matrix
        Cdd = np.diag(Cdd_diag) + Cdd_off_diag
        
        # Generate dot-sensor coupling capacitances
        Cds = np.random.uniform(-0.1, -0.05, (num_dots, num_sensors)) * Css
        
        return cls(Cdd, Cds)

    def get_energy_offset(self, charge_state: np.ndarray, 
                         sensor_voltages: np.ndarray,   #TODO: change to detuning
                         eps0: float) -> np.ndarray:
        """
        Calculate the energy offset for each sensor.
        
        The energy offset is determined by the charge state, sensor voltages,
        and a common gate voltage offset.
        
        Args:
            charge_state: Vector of charge states for each dot
            sensor_voltages: Vector of voltages applied to sensors
            eps0: Common gate voltage offset
            
        Returns:
            np.ndarray: Energy offset for each sensor
            
        Raises:
            ValueError: If charge_state has wrong dimension
        """
        if charge_state.shape[0] != self.num_dots:
            raise ValueError(f"Charge state vector must have length {self.num_dots}, "
                           f"but got {charge_state.shape[0]}.")
            
        # Calculate energy offset including DC gate voltage
        energy_offset = (self.Cds.T @ self.Cdd_inv @ 
                        (charge_state + self.Cds @ sensor_voltages) + eps0)
        
        return energy_offset

    def get_coupling_matrix(self) -> np.ndarray:
        """
        Get the effective coupling matrix between dots and sensors.
        
        Returns:
            np.ndarray: Coupling matrix Cds^T @ Cdd^(-1)
        """
        return self.Cds.T @ self.Cdd_inv
    
    def get_dot_energies(self, charge_state: np.ndarray, 
                        sensor_voltages: np.ndarray, 
                        eps0: float) -> np.ndarray:
        """
        Calculate the energy of each quantum dot.
        
        Args:
            charge_state: Vector of charge states for each dot
            sensor_voltages: Vector of voltages applied to sensors
            eps0: Common gate voltage offset
            
        Returns:
            np.ndarray: Energy of each dot
        """
        if charge_state.shape[0] != self.num_dots:
            raise ValueError(f"Charge state vector must have length {self.num_dots}.")
            
        # Energy of each dot
        dot_energies = self.Cdd_inv @ (charge_state + self.Cds @ sensor_voltages) + eps0
        return dot_energies

    def __repr__(self) -> str:
        """String representation of the quantum dot system."""
        return (f"QuantumDotSystem(num_dots={self.num_dots}, "
                f"num_sensors={self.num_sensors})")
    
    def __str__(self) -> str:
        """Detailed string representation."""
        return (f"QuantumDotSystem:\n"
                f"  Number of dots: {self.num_dots}\n"
                f"  Number of sensors: {self.num_sensors}\n"
                f"  Dot-dot capacitance scale: {np.mean(np.abs(self.Cdd_inv)):.2e} F⁻¹\n"
                f"  Dot-sensor coupling scale: {np.mean(np.abs(self.Cds)):.2e} F")