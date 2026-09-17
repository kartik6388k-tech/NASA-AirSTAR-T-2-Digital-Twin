"""
simulation/simulator.py

Core time-stepping simulation environment.

Integrates the validated physics modules (aircraft, atmosphere, aerodynamics,
conversion, flight dynamics, propulsion) over time using a Runge-Kutta 4th Order
(RK4) numerical integration scheme.

Architecture Notes:
- The actual equations of motion and state-derivative generation strictly
  remain inside flight_dynamics.py.
- Aerodynamic forces and Propulsion thrust are evaluated here strictly as
  DIAGNOSTIC-ONLY outputs. They do not currently feed back into or affect the
  2-state short-period state evolution, which relies entirely on the
  conversion -> flight_dynamics matrix formulation. In particular,
  `total_thrust_N` may legitimately be NaN when `propulsion_model_status` is
  "MODEL_NOT_IDENTIFIED" -- this is safe precisely because thrust is never
  read back into the RK4 integration, only logged.
- `mass`/`iyy` used for integration are taken from `flight_data.reference_mass_properties`
  (the NASA T-2 flight-test record), NOT from the `aircraft` argument. This is
  an explicit design choice to reproduce the exact physical condition paired
  with the identified aerodynamic derivatives; it means this simulator is a
  flight-test reference model for the selected FlightCondition, not a
  simulation of the active 2017 aircraft baseline described by aircraft.py.
- `aircraft` is retained on the instance for future use (e.g. a fully-coupled
  model driven by aircraft.py's own mass/geometry/inertia) but is NOT
  currently read by any part of the integration or diagnostics below. Do not
  assume the simulator is coupled to the aircraft baseline just because an
  Aircraft instance is passed in and stored.
"""

import csv
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional

from models.aircraft import Aircraft
from models.atmosphere import get_atmosphere_from_altitude
from models.aerodynamics import (
    FlightTestAeroData,
    compute_longitudinal_delta_coefficients,
    compute_longitudinal_delta_forces
)
from models.conversion import convert_to_dimensional_derivatives
from models.flight_dynamics import ReferenceTrim, PerturbationState, FlightDynamics
from models.propulsion import PropulsionModel

# Tolerance used only to decide whether `duration` lands on an exact multiple
# of `dt` (up to floating-point rounding). This is NOT a physics tolerance.
_STEP_COUNT_EPS = 1e-9


@dataclass
class SimulationCommand:
    """Instantaneous control inputs provided to the simulation at time t."""
    delta_elevator_rad: float
    left_throttle: float = 0.0
    right_throttle: float = 0.0
    left_rpm: float = 0.0
    right_rpm: float = 0.0
    left_lambda: float = 0.0   # PROJECT_DEFINED explicit healthy engine baseline
    right_lambda: float = 0.0  # PROJECT_DEFINED explicit healthy engine baseline

    def validate(self) -> None:
        """Ensures all command inputs are mathematically finite and physically bounded."""
        # Note: Elevator deflection limits are explicitly unbounded here until
        # verified NASA T-2 actuator limits are sourced. Do not invent limits.
        for attr in (
            "delta_elevator_rad", "left_throttle", "right_throttle",
            "left_rpm", "right_rpm", "left_lambda", "right_lambda"
        ):
            val = getattr(self, attr)
            if not math.isfinite(val):
                raise ValueError(f"SimulationCommand.{attr} must be finite, got {val}")

        if not (0.0 <= self.left_throttle <= 1.0):
            raise ValueError(f"left_throttle must be in [0, 1], got {self.left_throttle}")
        if not (0.0 <= self.right_throttle <= 1.0):
            raise ValueError(f"right_throttle must be in [0, 1], got {self.right_throttle}")
        if self.left_rpm < 0.0:
            raise ValueError(f"left_rpm must be non-negative, got {self.left_rpm}")
        if self.right_rpm < 0.0:
            raise ValueError(f"right_rpm must be non-negative, got {self.right_rpm}")
        if not (0.0 <= self.left_lambda <= 1.0):
            raise ValueError(f"left_lambda must be in [0, 1], got {self.left_lambda}")
        if not (0.0 <= self.right_lambda <= 1.0):
            raise ValueError(f"right_lambda must be in [0, 1], got {self.right_lambda}")


class Simulator:
    """
    Executes the time-stepping simulation, applying RK4 integration to the
    short-period state derivatives and logging subsystem outputs.
    """

    def __init__(
        self,
        aircraft: Aircraft,
        flight_data: FlightTestAeroData,
        propulsion: PropulsionModel
    ):
        # Stored for future use only -- see module docstring. Not read by
        # the integration or diagnostics in this class.
        self.aircraft = aircraft
        self.flight_data = flight_data
        self.propulsion = propulsion

        # 1. Initialize Atmosphere for Nominal Altitude
        self.nominal_altitude = self.flight_data.nominal_condition.altitude_m
        self.atmosphere = get_atmosphere_from_altitude(self.nominal_altitude)

        # 2. Convert Nondimensional Aero to Dimensional Stability Derivatives
        self.dimensional_model = convert_to_dimensional_derivatives(
            self.flight_data, self.atmosphere
        )
        self.short_period_derivatives = self.dimensional_model.derivatives

        # 3. Initialize Flight Dynamics Engine
        self.Ue = self.flight_data.nominal_condition.airspeed_mps
        self.trim = ReferenceTrim(
            flight_condition=self.flight_data.condition,
            u0=self.Ue
        )
        self.fd_engine = FlightDynamics(self.trim)

        # 4. Reference mass/inertia for integration.
        # EXPLICIT DESIGN CHOICE: We use the flight_data reference mass/inertia
        # rather than the active aircraft.py baseline to preserve the exact physical
        # condition that corresponds directly to the identified aerodynamic derivatives.
        # This makes the simulator a flight-test reference model for the selected
        # FlightCondition, not a simulation of the aircraft.py 2017 baseline.
        self.mass = self.flight_data.reference_mass_properties.mass_kg
        self.iyy = self.flight_data.reference_mass_properties.Iyy_kg_m2

        # Data logging
        self.history: List[Dict[str, Any]] = []

    def _get_derivatives(self, state: PerturbationState, delta_e: float) -> tuple[float, float]:
        """Helper to extract dx/dt for the RK4 step."""
        res = self.fd_engine.calculate_state_derivatives(
            mass=self.mass,
            iyy=self.iyy,
            state=state,
            derivatives=self.short_period_derivatives,
            delta_e=delta_e
        )
        return res['derivatives']['delta_w_dot'], res['derivatives']['delta_q_dot']

    def _rk4_step(
        self,
        current_state: PerturbationState,
        t: float,
        dt: float,
        control_schedule: Callable[[float], SimulationCommand]
    ) -> PerturbationState:
        """Performs one RK4 numerical integration step of size `dt` starting at time `t`."""
        # k1
        cmd1 = control_schedule(t)
        cmd1.validate()
        dw_dot1, dq_dot1 = self._get_derivatives(current_state, cmd1.delta_elevator_rad)

        # k2
        cmd2 = control_schedule(t + dt / 2.0)
        cmd2.validate()
        state2 = PerturbationState(
            delta_w=current_state.delta_w + 0.5 * dt * dw_dot1,
            delta_q=current_state.delta_q + 0.5 * dt * dq_dot1
        )
        dw_dot2, dq_dot2 = self._get_derivatives(state2, cmd2.delta_elevator_rad)

        # k3
        cmd3 = control_schedule(t + dt / 2.0)
        cmd3.validate()
        state3 = PerturbationState(
            delta_w=current_state.delta_w + 0.5 * dt * dw_dot2,
            delta_q=current_state.delta_q + 0.5 * dt * dq_dot2
        )
        dw_dot3, dq_dot3 = self._get_derivatives(state3, cmd3.delta_elevator_rad)

        # k4
        cmd4 = control_schedule(t + dt)
        cmd4.validate()
        state4 = PerturbationState(
            delta_w=current_state.delta_w + dt * dw_dot3,
            delta_q=current_state.delta_q + dt * dq_dot3
        )
        dw_dot4, dq_dot4 = self._get_derivatives(state4, cmd4.delta_elevator_rad)

        # Advance state
        next_w = current_state.delta_w + (dt / 6.0) * (dw_dot1 + 2.0 * dw_dot2 + 2.0 * dw_dot3 + dw_dot4)
        next_q = current_state.delta_q + (dt / 6.0) * (dq_dot1 + 2.0 * dq_dot2 + 2.0 * dq_dot3 + dq_dot4)

        return PerturbationState(delta_w=next_w, delta_q=next_q)

    def _check_state_finite(self, state: PerturbationState, time: float) -> None:
        """Raises if the integrated state has diverged to NaN/Inf."""
        if not math.isfinite(state.delta_w) or not math.isfinite(state.delta_q):
            raise RuntimeError(f"Numerical integration diverged at t={time:.4f}s. State contains NaN/Inf.")

    def _record_step(
        self,
        state: PerturbationState,
        time: float,
        control_schedule: Callable[[float], SimulationCommand]
    ) -> None:
        """
        Evaluates the diagnostic-only aerodynamic and propulsion subsystems at
        `time` for the given `state`, and appends one row to self.history.

        Shared by every place a state gets logged (full steps, the final
        recorded full-step state, and the optional trailing partial step) so
        the diagnostic evaluation logic exists in exactly one place.
        """
        cmd = control_schedule(time)
        cmd.validate()

        # PROJECT_DERIVED: Small angle approximation mapping vertical velocity to angle of attack
        delta_alpha = state.delta_w / self.Ue

        aero_coeffs = compute_longitudinal_delta_coefficients(
            flight_data=self.flight_data,
            delta_alpha_rad=delta_alpha,
            delta_pitch_rate_rads=state.delta_q,
            delta_elevator_rad=cmd.delta_elevator_rad
        )
        aero_forces = compute_longitudinal_delta_forces(
            flight_data=self.flight_data,
            delta_coeffs=aero_coeffs,
            atmosphere=self.atmosphere
        )

        # DIAGNOSTIC-ONLY: never fed back into the state equations. See
        # module docstring re: total_thrust_N possibly being NaN.
        prop_state = self.propulsion.calculate_thrust(
            left_rpm=cmd.left_rpm,
            left_throttle=cmd.left_throttle,
            left_lambda=cmd.left_lambda,
            right_rpm=cmd.right_rpm,
            right_throttle=cmd.right_throttle,
            right_lambda=cmd.right_lambda,
            airspeed=self.Ue
        )

        fd_res = self.fd_engine.calculate_state_derivatives(
            mass=self.mass,
            iyy=self.iyy,
            state=state,
            derivatives=self.short_period_derivatives,
            delta_e=cmd.delta_elevator_rad
        )

        self.history.append({
            "time_s": time,
            "delta_w_mps": state.delta_w,
            "delta_q_rads": state.delta_q,
            "delta_e_rad": cmd.delta_elevator_rad,
            "delta_w_dot": fd_res['derivatives']['delta_w_dot'],
            "delta_q_dot": fd_res['derivatives']['delta_q_dot'],
            "perturbation_az_mps2": fd_res['outputs']['perturbation_az'],
            "delta_alpha_rad": delta_alpha,  # PROJECT_DERIVED approximation
            "delta_CL": aero_coeffs.delta_CL,
            "delta_Cm": aero_coeffs.delta_Cm,
            "delta_Lift_N": aero_forces.delta_lift_N,
            "delta_PitchMom_Nm": aero_forces.delta_pitching_moment_Nm,
            "total_thrust_N": prop_state.total_thrust_N,
            "propulsion_model_status": prop_state.model_status
        })

    def run(
        self,
        initial_state: PerturbationState,
        duration: float,
        dt: float,
        control_schedule: Callable[[float], SimulationCommand]
    ) -> List[Dict[str, Any]]:
        """
        Executes the main simulation loop and returns the state history.

        `duration` need not be an exact multiple of `dt`: the simulator runs
        as many full-size `dt` steps as fit, then -- if a nonzero remainder
        is left over -- performs one additional, explicitly shorter RK4 step
        of that remainder so the run always ends exactly at t=duration.
        """
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"Simulation duration must be strictly positive, got {duration}")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"Simulation timestep (dt) must be strictly positive, got {dt}")
        if not callable(control_schedule):
            raise TypeError("control_schedule must be a callable.")

        # Reliable (non-floating-point-fragile) determination of how many
        # full-size dt steps fit in duration. `duration % dt` is avoided
        # entirely: float modulo of two non-exactly-representable values is
        # not a trustworthy way to detect "is duration an exact multiple of dt".
        raw_steps = duration / dt
        nearest_int_steps = round(raw_steps)
        if math.isclose(raw_steps, nearest_int_steps, rel_tol=_STEP_COUNT_EPS, abs_tol=_STEP_COUNT_EPS):
            num_full_steps = int(nearest_int_steps)
        else:
            num_full_steps = int(math.floor(raw_steps))

        full_steps_duration = num_full_steps * dt
        remainder = duration - full_steps_duration
        if remainder < 0.0:
            # Only possible from floating-point rounding right at the boundary.
            remainder = 0.0
        has_partial_step = remainder > _STEP_COUNT_EPS

        if has_partial_step:
            print(
                f"Note: duration ({duration}s) is not an exact multiple of dt ({dt}s). "
                f"Running {num_full_steps} full step(s) of {dt}s, then one final "
                f"partial step of {remainder:.6f}s to land exactly on t={duration}s."
            )

        print(f"Starting simulation: duration={duration}s, dt={dt}s")
        self.history.clear()

        current_state = initial_state

        for step in range(num_full_steps):
            time = step * dt
            self._record_step(current_state, time, control_schedule)
            current_state = self._rk4_step(current_state, time, dt, control_schedule)
            self._check_state_finite(current_state, time + dt)

        # Record the state reached after all full-size steps.
        self._record_step(current_state, full_steps_duration, control_schedule)

        if has_partial_step:
            current_state = self._rk4_step(current_state, full_steps_duration, remainder, control_schedule)
            self._check_state_finite(current_state, duration)
            self._record_step(current_state, duration, control_schedule)

        return self.history

    def save_results(self, run_name: str = "simulation_history", scenario_name: Optional[str] = None) -> None:
        """
        Exports the simulation trajectory to CSV.
        """
        if not self.history:
            print("No simulation data to save.")
            return

        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / "data" / "processed"
        data_dir.mkdir(parents=True, exist_ok=True)

        output_file = data_dir / f"{run_name}.csv"
        headers = self.history[0].keys()

        with open(output_file, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(self.history)

        print(f"Simulation saved successfully to: {output_file}")
        
        if scenario_name is not None:
            meta_file = data_dir / f"{run_name}.meta.json"
            with open(meta_file, mode='w', encoding='utf-8') as mf:
                json.dump({"scenario_name": scenario_name}, mf, indent=2)
            print(f"Scenario metadata saved to: {meta_file}")


# =====================================================================
# Standalone Execution Example
# =====================================================================
if __name__ == "__main__":
    import yaml
    from models.aerodynamics import AerodynamicsDatabase, FlightCondition

    # 1. Load config and databases
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    aircraft = Aircraft.from_config(config)
    aero_db = AerodynamicsDatabase.from_config(config)
    propulsion = PropulsionModel.from_config(config)

    # 2. Select Flight 41
    flight_data = aero_db.get_flight_test_data(FlightCondition.FLIGHT_41)

    # 3. Initialize Simulator
    sim = Simulator(
        aircraft=aircraft,
        flight_data=flight_data,
        propulsion=propulsion
    )

    # 4. Define Control Schedule (TEST_ONLY demonstration values)
    def elevator_doublet(t: float) -> SimulationCommand:
        de = 0.0
        if 1.0 <= t < 1.5:
            de = -0.05  # TEST_ONLY: -0.05 rad doublet pulse
        elif 1.5 <= t < 2.0:
            de = 0.05   # TEST_ONLY: +0.05 rad doublet pulse

        return SimulationCommand(
            delta_elevator_rad=de,
            left_rpm=50000.0,    # TEST_ONLY: arbitrary engine speed
            right_rpm=50000.0,   # TEST_ONLY: arbitrary engine speed
            left_throttle=0.5,   # TEST_ONLY: nominal throttle
            right_throttle=0.5,  # TEST_ONLY: nominal throttle
            left_lambda=0.0,     # TEST_ONLY: explicitly healthy
            right_lambda=0.0     # TEST_ONLY: explicitly healthy
        )

    # 5. Run Simulation (10 seconds, 50Hz / 0.02s timestep)
    initial_state = PerturbationState(delta_w=0.0, delta_q=0.0)
    history = sim.run(initial_state, duration=10.0, dt=0.02, control_schedule=elevator_doublet)

    # 6. Save Data
    sim.save_results("t2_short_period_doublet_response")