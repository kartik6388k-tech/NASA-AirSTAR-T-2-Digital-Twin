# NASA AirSTAR T-2 Digital Twin

## Project Purpose
The NASA AirSTAR T-2 Digital Twin is an engineering simulation and post-processing environment for the NASA Airborne Subscale Transport Aircraft Research (AirSTAR) 5.5% Generic Transport Model (GTM) T-2 aircraft. 

Its primary purpose is to simulate and analyze the longitudinal short-period dynamics of the aircraft, driven by the aircraft configuration, atmospheric model, available aerodynamic model, and defined flight scenarios, with propulsion represented through an explicit model interface. The project distinguishes explicitly between NASA-verified physical data (e.g. baseline geometry, mass properties, reference conditions) and project-derived implementations (e.g. reduced-order estimators, sensors, integrated components, and control schedules).

## Final Architecture
The architecture is structured across clearly defined domains:
- **`models/`**: The core physical models. Includes `aircraft.py` for physical baselines, `aerodynamics.py` for longitudinal aerodynamic derivatives, `atmosphere.py` for standard ISA atmosphere, and `propulsion.py` for JetCat P70 twin turbines.
- **`simulation/`**: Components responsible for scenario definitions and numerical execution. Contains `simulator.py` (which runs RK4 integration of reduced-order state) and `scenarios.py` (the scenario registry).
- **`sensors/`**: The `sensor_model.py` providing a project-defined sensor noise and bias model for the simulator's true state.
- **`digital_twin/`**: Higher-level integration logic. Contains `estimator.py` (Kalman-style state estimation), `health_monitor.py` (residuals and rule-based diagnostic evaluation), and `integration.py` (tying true simulation to measurements and estimations).
- **`dashboard/`**: The presentation layer (`app.py`), serving a read-only aerospace telemetry console to visualize saved simulation runs (`.csv`), matching scenario metadata, and integrated flight data.
- **`scripts/`**: One-off scripts and utilities, such as `validate_short_period.py` for dynamic verification.
- **`tests/`**: Comprehensive `pytest` suite validating models, simulation, health rules, mismatch protections, and the web UI.

## Run and Test Commands

**Run a simulation scenario:**
```bash
python main.py --scenario flight_15_reference_condition_run
```
Outputs are saved as CSV files with metadata in `data/processed/`.

**Run the aerospace dashboard:**
```bash
python dashboard/app.py --host 127.0.0.1 --port 8000
```
This serves the read-only post-processing interface at `http://127.0.0.1:8000`.

**Run the full test suite:**
```bash
python -m pytest tests/ -q
```

## Important Assumptions and Limitations
1. **Reduced-Order Scope:** The simulation only tracks longitudinal short-period state `[delta_w, delta_q]`. It does not propagate a full 6DOF inertial state.
2. **Propulsion Dynamics:** Propulsion is represented as the T-2 twin JetCat P70 configuration with an explicit model interface. An identified thrust model is not currently provided; MODEL_NOT_IDENTIFIED indicates this model limitation, not an engine failure.
3. **No Automatic Correction:** Estimators and health monitors are strictly read-only diagnostics. They do not inject feedback or trim corrections into the aerodynamic simulation.
4. **Dashboard Post-Processing:** The dashboard does not run or modify physics simulations. Missing values are displayed as "N/A" rather than interpolated or synthetically generated.
5. **No GUI Dependencies:** The dashboard relies entirely on Python standard library modules (`http.server`) and vanilla HTML/CSS/JS without external frontend frameworks.

## Data Provenance
A strict distinction is enforced throughout the codebase and dashboard:
- **NASA_VERIFIED**: Values drawn directly from AirSTAR documentation, flight-test references (e.g. AIAA 2015-2704), and measured references.
- **PROJECT_DEFINED** / **PROJECT_DERIVED**: Elements synthesized by this project, including specific test control schedules, simplified noise models, the reduced-order Kalman estimator, and health monitoring bounds.
