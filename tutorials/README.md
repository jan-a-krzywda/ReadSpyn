# Tutorials

This folder contains a **curated, publication-facing tutorial path** for ReadSpyn.
Unlike the exploratory notebooks in `examples/`, these notebooks are designed to:

- introduce one concept at a time
- explain **what is being defined and why**
- produce cleaner figures that can be adapted for talks, supplements, or papers

## Suggested Order

1. [01_first_readout_and_iq.ipynb](./01_first_readout_and_iq.ipynb)
   - First end-to-end simulation with one dot and one sensor
   - Conductance peak, noiseless readout, then `1/f` and OU-noise comparisons
   - Reflected voltage, demodulated `I/Q`, integrated traces, and the IQ-plane view

2. [02_qubit_layout_and_couplings.ipynb](./02_qubit_layout_and_couplings.ipynb)
   - Geometry-driven layouts for dots and sensors
   - Coupling matrices, layout figures, and comparison of multiple device geometries
   - Sensor operating-point examples that map geometry-driven state shifts onto Coulomb peaks

3. [03_noise_integration_and_dataset.ipynb](./03_noise_integration_and_dataset.ipynb)
   - Shared `1/f` noise, integration-time sampling, and compact dataset generation
   - Conductance-peak noise-path visualisation and publication-style integrated IQ cloud figures

   - Reconstruct one plausible geometry from chosen `Cdd`/`Cds` matrices
   - Tune `Cds`, `Cdd`, and sensor operating point to change detuning, conductance, and IQ separation

5. [04_two_sensor_multi_qubit_readout.ipynb](./04_two_sensor_multi_qubit_readout.ipynb)
   - Multi-qubit, multi-sensor example
   - Compare sensor-specific IQ traces, integrated IQ clouds, and sensor-specific conductance peaks under shared noise

## Design Notes

- These notebooks are intentionally more tutorial-style than the notebooks in `examples/`.
- Each notebook starts from a clearly stated question and ends with short takeaways.
- Parameter-definition cells are separated from plotting cells so users can see where to modify the model.
- Heavy parameter sweeps, temporary tests, and debugging workflows should still live in `examples/`.
