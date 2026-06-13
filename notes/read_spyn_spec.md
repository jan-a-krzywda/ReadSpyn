# ReadSpyn v2.0 Architecture Specification
**Objective:** Finalize the ReadSpyn simulation framework by implementing four core advanced features: Distance-Correlated Sensor Noise, a Gymnasium Environment Wrapper, a JAX/Diffrax acceleration backend, and Dispersive Readout ($C_q$) capabilities.

---

## 1. Distance-Correlated Sensor Array Noise
**Goal:** Replace independent, isolated sensor noise with a spatially correlated noise bath to accurately simulate substrate-wide charge fluctuations across multiple quantum dot sensors.

### Agent Implementation Directives:
* **Module:** Extend the existing noise generation module (e.g., `noise_models.py`).
* **Class:** Create a `CorrelatedNoiseModel` that accepts a vector of sensor coordinates.
* **Mechanism:** 1. Calculate the Euclidean distance matrix $d_{ij}$ between all sensors.
    2. Construct a spatial covariance matrix $\Sigma$ using an exponential decay model:
       $$\Sigma_{ij} = \sigma_i \sigma_j \exp\left(-\frac{d_{ij}}{\lambda}\right)$$
       where $\lambda$ is the user-defined correlation length and $\sigma$ is the noise standard deviation.
    3. Compute the Cholesky decomposition: $\Sigma = LL^T$.
    4. Generate a base vector of independent noise trajectories (1/f or Ornstein-Uhlenbeck) $\mathbf{z}(t)$.
    5. Yield the final correlated noise vector $\delta\boldsymbol{\varepsilon}(t) = L\mathbf{z}(t)$.
* **Capabilities Enabled:** Realistic simulation of spatial crosstalk and common-mode noise rejection strategies in multiplexed readout arrays.

---

## 2. Active Control `Gymnasium` Environment Wrapper
**Goal:** Expose ReadSpyn as a standard reinforcement learning environment, transforming it from a static dataset generator into a dynamic, closed-loop active control testbed.

### Agent Implementation Directives:
* **Module:** Create a new file `readspyn_env.py`.
* **Dependency:** Subclass `gymnasium.Env`.
* **State & Action Space:**
    * `action_space`: `Box` representing continuous adjustments to the sensor gate voltages ($\mathbf{V}_s$).
    * `observation_space`: `Box` containing the current demodulated and integrated $I$ and $Q$ quadratures, plus the elapsed integration time.
* **Core Methods:**
    * `reset()`: Initializes a new time-domain trajectory with a random charge state and fresh noise seed. Returns initial $(I, Q)$ and `info` dict.
    * `step(action)`: 
        1. Updates the static detuning based on the new gate voltage `action`.
        2. Integrates the underlying RLC ODE forward by a fixed time-window $\Delta t$.
        3. Returns the new observation $(I, Q)$, a reward (e.g., based on state discriminability or SNR), a `terminated` flag (if max readout time is reached), and `truncated`.
* **Capabilities Enabled:** Direct interfacing with stable-baselines3 or Ray RLlib for training agents to perform real-time Coulomb peak tracking and autonomous readout optimization.

---

## 3. JAX / Diffrax Backend for Massively Parallel SDE Solving
**Goal:** Circumvent the computational bottleneck of `scipy.integrate.solve_ivp` by implementing a fully vectorized solver backend, enabling hardware-accelerated (GPU/TPU) batch generation of noisy readout traces.

### Agent Implementation Directives:
* **Module:** Create `jax_backend.py`.
* **Dependencies:** `jax`, `jax.numpy`, `diffrax`.
* **Implementation:**
    1. Re-write the core RLC equations of motion (Eq. 5 and Eq. 6) as a JAX-compatible callable.
    2. Formulate the time-domain noise (OU or 1/f) as a Diffrax `ControlTerm` to allow stochastic differential equation (SDE) integration.
    3. Use `diffrax.diffeqsolve` with an appropriate solver (e.g., `diffrax.Tsit5` for ODEs or `diffrax.EulerHeun` for SDEs).
    4. **Crucial:** Wrap the solver function in `jax.vmap` to map over an array of initial conditions and JAX `PRNGKey` arrays.
* **Capabilities Enabled:** Generating tens of thousands of readout trajectories in seconds. Essential for training deep neural networks for state classification or Bayesian filtering.

---

## 4. Dispersive Readout via Quantum Capacitance ($C_q$)
**Goal:** Extend the physical measurement model beyond dissipative (SET conductance) sensing to include dispersive (gate-based reflectometry) sensing by making the tank capacitance dynamic.

### Agent Implementation Directives:
* **Module:** Update the `RLC_sensor` and ODE definitions.
* **Mechanism:** Currently, the state vector updates via $G_s(t)$. Introduce an optional dynamic capacitance mode where $C_p$ becomes $C_p(t) = C_{p,0} + C_q(\varepsilon(t))$.
* **Models to Implement:** Allow the user to select the physical origin of the quantum capacitance:
    1.  *Dot-to-Reservoir (Thermal):*
        $$C_q(\varepsilon) = \frac{(e \alpha)^2}{4 k_B T} \cosh^{-2}\left(\frac{\varepsilon}{2 k_B T}\right)$$
    2.  *Inter-Dot (Tunnel-Coupled):*
        $$C_q(\varepsilon) = (e \alpha)^2 \frac{t_c^2}{(\varepsilon^2 + 4t_c^2)^{3/2}}$$
        *(where $\alpha$ is the lever arm, $t_c$ is tunnel coupling, and $T$ is electron temperature).*
* **ODE Modification:** Ensure the $dV_{C_p}/dt$ equation correctly handles a time-varying capacitance (recall $I = d(CV)/dt = C(dV/dt) + V(dC/dt)$).
* **Capabilities Enabled:** Simulating phase shifts in the reflected RF tone, allowing the modeling of scalable gate-based architectures without dedicated SET structures.