"""
tests/test_simulator.py

Tests for simulation/simulator.py.

Fixture design note
--------------------
Simulator wires together six modules (aircraft, atmosphere, aerodynamics,
conversion, flight_dynamics, propulsion). To test RK4 numerics in isolation
we still have to go through the real Simulator/FlightDynamics/conversion
pipeline (that's the actual code under test), but we deliberately choose
TEST_ONLY aerodynamic derivatives that make the resulting short-period
system solvable in closed form:

    CL_alpha = CL_q = Cm_alpha = Cm_q = 0   ->   Zw = Zq = Mw = Mq = 0
    CL_delta_e, Cm_delta_e nonzero          ->   Zde, Mde nonzero

With Zw = Zq = Mw = Mq = 0 and a constant elevator command, the equations of
motion reduce to:

    qdot = Mde*de / Iyy                      = c2                 (constant)
    q(t) = q0 + c2*t                                               (linear)
    wdot = Zde*de/mass + Ue*q(t)             = c1 + Ue*q0 + Ue*c2*t (linear in t)
    w(t) = w0 + (c1 + Ue*q0)*t + 0.5*Ue*c2*t^2                     (quadratic)

Both w(t) and q(t) are polynomials of degree <= 2 in t. Classical RK4 is
exact (to floating-point precision) for any ODE whose solution is a
polynomial of degree <= 4, so this gives an exact analytic reference to
check the RK4 implementation against -- not just a "smaller error at smaller
dt" argument.

`propulsion` and `aircraft` are stubbed out with lightweight doubles:
Simulator never reads `aircraft` for the integration/diagnostics path, and
`propulsion.calculate_thrust` is used only for diagnostic logging, never fed
back into the RK4 state derivatives -- so a duck-typed stand-in is faithful
to how the real code uses these objects.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Allow `import models...` / `import simulation...` when tests are run from
# the project root (adjust if your repository layout differs).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.aerodynamics import (
    FlightCondition,
    LongitudinalDerivativeSet,
    IdentifiedDerivative,
    NominalFlightCondition,
    ReferenceMassProperties,
    SourceProvenance,
    PerturbationValidityLimits,
    FlightTestAeroData,
)
from models.flight_dynamics import PerturbationState
from simulation.simulator import Simulator, SimulationCommand


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _StubPropulsion:
    """Duck-typed stand-in for PropulsionModel: diagnostic-only, never
    consulted by the RK4 integration itself."""

    def calculate_thrust(self, **kwargs):
        return SimpleNamespace(total_thrust_N=0.0, model_status="TEST_STUB")


def _make_flight_data(
    *,
    cl_de=0.5,
    cm_de=-1.0,
    mass_kg=1000.0,
    iyy_kg_m2=2000.0,
    airspeed_mps=50.0,
    altitude_m=1000.0,
    mac_m=1.0,
    wing_area_m2=10.0,
) -> FlightTestAeroData:
    """Builds a TEST_ONLY FlightTestAeroData record. CL_alpha/CL_q/Cm_alpha/Cm_q
    are pinned to zero so Zw=Zq=Mw=Mq=0 (see module docstring)."""
    prov = SourceProvenance(
        document="TEST_ONLY", authors="TEST_ONLY", year=2024,
        table="TEST_ONLY", flight="TEST_ONLY",
    )
    derivatives = LongitudinalDerivativeSet(
        CL_alpha=IdentifiedDerivative(0.0, 0.0),
        CL_q=IdentifiedDerivative(0.0, 0.0),
        CL_delta_e=IdentifiedDerivative(cl_de, 0.0),
        Cm_alpha=IdentifiedDerivative(0.0, 0.0),
        Cm_q=IdentifiedDerivative(0.0, 0.0),
        Cm_delta_e=IdentifiedDerivative(cm_de, 0.0),
    )
    nominal = NominalFlightCondition(
        flight_id="TEST_ONLY_FLIGHT",
        turbulence_context="None",
        airspeed_mps=airspeed_mps,
        angle_of_attack_rad=0.0,
        altitude_m=altitude_m,
        throttle=0.5,
        mac_m=mac_m,
        wing_area_m2=wing_area_m2,
    )
    mass_props = ReferenceMassProperties(mass_kg=mass_kg, Iyy_kg_m2=iyy_kg_m2)
    return FlightTestAeroData(
        condition=FlightCondition.FLIGHT_41,
        derivatives=derivatives,
        nominal_condition=nominal,
        reference_mass_properties=mass_props,
        derivative_provenance=prov,
        nominal_condition_provenance=prov,
        validity_limits=PerturbationValidityLimits(),
    )


def _make_simulator(**flight_data_kwargs) -> Simulator:
    # `aircraft=None` here is an explicit TEST STUB, not a claim that Aircraft
    # is optional in production. It works only because the current
    # Simulator implementation never reads `self.aircraft` anywhere in the
    # integration or diagnostics path (see simulator.py's module docstring).
    # If Simulator is later changed to consume Aircraft (e.g. a fully-coupled
    # model), these tests must be updated to pass a real Aircraft instance.
    flight_data = _make_flight_data(**flight_data_kwargs)
    return Simulator(aircraft=None, flight_data=flight_data, propulsion=_StubPropulsion())


def _constant_elevator_schedule(de_rad: float):
    def schedule(t: float) -> SimulationCommand:
        return SimulationCommand(delta_elevator_rad=de_rad)
    return schedule


def _analytic_w_q(sim: Simulator, w0: float, q0: float, de_rad: float, t: float):
    """Closed-form w(t), q(t) for the Zw=Zq=Mw=Mq=0 constant-elevator case."""
    Zde = sim.short_period_derivatives.Zde
    Mde = sim.short_period_derivatives.Mde
    mass = sim.mass
    iyy = sim.iyy
    Ue = sim.Ue

    c1 = Zde * de_rad / mass
    c2 = Mde * de_rad / iyy

    q_t = q0 + c2 * t
    w_t = w0 + (c1 + Ue * q0) * t + 0.5 * Ue * c2 * t * t
    return w_t, q_t


# ---------------------------------------------------------------------------
# RK4 correctness against the known analytic (polynomial) solution
#
# NOTE ON SCOPE: the derivatives built by _make_flight_data() (CL_alpha=
# CL_q=Cm_alpha=Cm_q=0) are synthetic, chosen purely to make the resulting
# short-period system solvable in closed form so RK4's numerics can be
# checked against an exact reference. They are NOT NASA T-2 flight-test
# values and this file does not claim to validate the physical short-period
# behavior published in AIAA 2015-2704 -- that is exactly the Level-3
# validation validate_short_period.py flags as still pending real NASA
# dynamic-reference values. These tests validate the integrator, not the
# aircraft physics.
# ---------------------------------------------------------------------------

def test_rk4_single_step_matches_analytic_polynomial_solution():
    sim = _make_simulator()
    de = 0.03
    dt = 0.1
    state0 = PerturbationState(delta_w=1.5, delta_q=0.02)

    next_state = sim._rk4_step(state0, t=0.0, dt=dt, control_schedule=_constant_elevator_schedule(de))

    expected_w, expected_q = _analytic_w_q(sim, state0.delta_w, state0.delta_q, de, dt)

    assert math.isclose(next_state.delta_w, expected_w, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(next_state.delta_q, expected_q, rel_tol=1e-9, abs_tol=1e-9)


def test_rk4_multi_step_matches_analytic_polynomial_solution():
    sim = _make_simulator()
    de = -0.02
    dt = 0.05
    n_steps = 37  # arbitrary, deliberately not a "round" number
    state = PerturbationState(delta_w=-0.4, delta_q=0.01)

    for step in range(n_steps):
        t = step * dt
        state = sim._rk4_step(state, t=t, dt=dt, control_schedule=_constant_elevator_schedule(de))

    total_t = n_steps * dt
    expected_w, expected_q = _analytic_w_q(sim, -0.4, 0.01, de, total_t)

    # Accumulated floating-point error over many steps, still tight.
    assert math.isclose(state.delta_w, expected_w, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(state.delta_q, expected_q, rel_tol=1e-6, abs_tol=1e-6)


def test_run_end_to_end_matches_analytic_polynomial_solution():
    sim = _make_simulator()
    de = 0.01
    dt = 0.02
    duration = 2.0  # exact multiple of dt
    initial_state = PerturbationState(delta_w=0.2, delta_q=-0.005)

    history = sim.run(initial_state, duration=duration, dt=dt, control_schedule=_constant_elevator_schedule(de))

    last_row = history[-1]
    expected_w, expected_q = _analytic_w_q(sim, 0.2, -0.005, de, last_row["time_s"])

    assert math.isclose(last_row["delta_w_mps"], expected_w, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(last_row["delta_q_rads"], expected_q, rel_tol=1e-6, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# duration / dt handling (the reported bug)
# ---------------------------------------------------------------------------

def test_exact_multiple_duration_produces_expected_step_count():
    sim = _make_simulator()
    dt = 0.02
    duration = 1.0  # exactly 50 steps

    history = sim.run(
        PerturbationState(0.0, 0.0), duration=duration, dt=dt,
        control_schedule=_constant_elevator_schedule(0.0),
    )

    # One row per full step (t=0..49*dt) plus the final row at t=duration.
    assert len(history) == 51
    assert math.isclose(history[-1]["time_s"], duration, rel_tol=0.0, abs_tol=1e-9)


def test_non_multiple_duration_ends_exactly_at_duration_with_partial_step():
    sim = _make_simulator()
    dt = 0.3
    duration = 1.0  # 3 full steps of 0.3s + a 0.1s partial step

    history = sim.run(
        PerturbationState(0.0, 0.0), duration=duration, dt=dt,
        control_schedule=_constant_elevator_schedule(0.0),
    )

    # 3 full-step rows (t=0, 0.3, 0.6) + 1 row after the 3 full steps (t=0.9)
    # + 1 final row for the partial step (t=1.0).
    assert len(history) == 5
    assert math.isclose(history[-1]["time_s"], duration, rel_tol=0.0, abs_tol=1e-9)
    # The run must not silently stop short of `duration`.
    assert history[-1]["time_s"] > history[-2]["time_s"]


def test_floating_point_near_multiple_duration_is_not_misclassified():
    # 0.1 + 0.1 + 0.1 != 0.3 in binary floating point; duration/dt here is
    # ~2.9999999999999996, not exactly 3 -- this must still resolve to
    # exactly 3 full steps and no partial step, not a phantom tiny extra step.
    sim = _make_simulator()
    dt = 0.1
    duration = 0.1 + 0.1 + 0.1

    history = sim.run(
        PerturbationState(0.0, 0.0), duration=duration, dt=dt,
        control_schedule=_constant_elevator_schedule(0.0),
    )

    assert len(history) == 4  # t=0, dt, 2dt, and the final row at duration
    assert math.isclose(history[-1]["time_s"], duration, rel_tol=0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_duration", [0.0, -1.0, float("nan"), float("inf")])
def test_run_rejects_invalid_duration(bad_duration):
    sim = _make_simulator()
    with pytest.raises(ValueError):
        sim.run(
            PerturbationState(0.0, 0.0), duration=bad_duration, dt=0.02,
            control_schedule=_constant_elevator_schedule(0.0),
        )


@pytest.mark.parametrize("bad_dt", [0.0, -0.02, float("nan"), float("inf")])
def test_run_rejects_invalid_dt(bad_dt):
    sim = _make_simulator()
    with pytest.raises(ValueError):
        sim.run(
            PerturbationState(0.0, 0.0), duration=1.0, dt=bad_dt,
            control_schedule=_constant_elevator_schedule(0.0),
        )


def test_run_rejects_non_callable_control_schedule():
    sim = _make_simulator()
    with pytest.raises(TypeError):
        sim.run(PerturbationState(0.0, 0.0), duration=1.0, dt=0.02, control_schedule="not_callable")


def test_command_validation_runs_at_every_rk4_substep():
    """A control_schedule that returns an invalid command only at the k2/k3
    midpoint time must still be caught -- proves cmd.validate() actually
    runs for all four RK4 stages, not just t and t+dt."""
    sim = _make_simulator()
    dt = 0.1

    def bad_at_midpoint(t: float) -> SimulationCommand:
        if math.isclose(t, dt / 2.0, rel_tol=0.0, abs_tol=1e-9):
            return SimulationCommand(delta_elevator_rad=0.0, left_throttle=5.0)  # out of [0,1]
        return SimulationCommand(delta_elevator_rad=0.0)

    with pytest.raises(ValueError):
        sim._rk4_step(PerturbationState(0.0, 0.0), t=0.0, dt=dt, control_schedule=bad_at_midpoint)


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"left_rpm": -1.0},
        {"right_rpm": -100.0},
        {"left_throttle": -0.1},
        {"left_throttle": 1.1},
        {"right_throttle": -0.5},
        {"right_throttle": 2.0},
        {"left_lambda": -0.1},
        {"left_lambda": 1.1},
        {"right_lambda": -1.0},
        {"right_lambda": 5.0},
    ],
)
def test_run_rejects_out_of_range_engine_commands(bad_kwargs):
    """Negative RPM, and throttle/lambda outside [0, 1], must be rejected by
    the simulator's own control-schedule validation at run() time -- not
    just by SimulationCommand.validate() called in isolation."""
    sim = _make_simulator()

    def bad_schedule(t: float) -> SimulationCommand:
        return SimulationCommand(delta_elevator_rad=0.0, **bad_kwargs)

    with pytest.raises(ValueError):
        sim.run(
            PerturbationState(0.0, 0.0), duration=0.5, dt=0.1,
            control_schedule=bad_schedule,
        )


# ---------------------------------------------------------------------------
# History length / finite-state checks / zero-input behavior
# ---------------------------------------------------------------------------

def test_history_cleared_between_runs():
    sim = _make_simulator()
    sim.run(PerturbationState(0.0, 0.0), duration=0.5, dt=0.1, control_schedule=_constant_elevator_schedule(0.0))
    first_len = len(sim.history)
    assert first_len > 0

    sim.run(PerturbationState(0.0, 0.0), duration=0.2, dt=0.1, control_schedule=_constant_elevator_schedule(0.0))
    assert len(sim.history) == 3  # t=0, 0.1, 0.2 -- not appended to the previous run's rows


def test_all_recorded_states_are_finite():
    sim = _make_simulator()
    history = sim.run(
        PerturbationState(0.1, 0.01), duration=1.0, dt=0.02,
        control_schedule=_constant_elevator_schedule(0.02),
    )
    for row in history:
        assert math.isfinite(row["delta_w_mps"])
        assert math.isfinite(row["delta_q_rads"])
        assert math.isfinite(row["delta_w_dot"])
        assert math.isfinite(row["delta_q_dot"])


def test_diverging_integration_raises_runtime_error(monkeypatch):
    # Every physically-derived path to a non-finite state is already
    # rejected earlier (conversion.py rejects overflowing dimensional
    # derivatives; FlightDynamics rejects non-finite dx/dt at each RK4
    # substage) -- by design, those are separate guards this test is not
    # meant to re-check. What run()'s own post-step check is uniquely
    # responsible for is catching a non-finite *result* out of the RK4
    # weighted-average combination itself, so we verify that guard directly
    # by stubbing _rk4_step to hand back a NaN state.
    sim = _make_simulator()
    bad_state = PerturbationState(delta_w=0.0, delta_q=0.0)
    bad_state.delta_w = float("nan")  # PerturbationState is a plain mutable object, not frozen
    monkeypatch.setattr(sim, "_rk4_step", lambda *args, **kwargs: bad_state)

    with pytest.raises(RuntimeError, match="diverged"):
        sim.run(
            PerturbationState(0.0, 0.0), duration=1.0, dt=0.5,
            control_schedule=_constant_elevator_schedule(0.0),
        )


def test_zero_input_zero_initial_state_stays_at_rest():
    sim = _make_simulator()
    history = sim.run(
        PerturbationState(0.0, 0.0), duration=1.0, dt=0.1,
        control_schedule=_constant_elevator_schedule(0.0),
    )
    for row in history:
        assert row["delta_w_mps"] == 0.0
        assert row["delta_q_rads"] == 0.0
        assert row["delta_w_dot"] == 0.0
        assert row["delta_q_dot"] == 0.0


# ---------------------------------------------------------------------------
# Diagnostic-only propulsion handling
# ---------------------------------------------------------------------------

def test_propulsion_not_identified_status_does_not_affect_state_integration():
    """total_thrust_N=NaN with model_status=MODEL_NOT_IDENTIFIED must be
    purely diagnostic: the RK4-integrated state must stay finite and match
    the analytic solution regardless of what propulsion reports."""

    class _NotIdentifiedPropulsion:
        def calculate_thrust(self, **kwargs):
            return SimpleNamespace(total_thrust_N=float("nan"), model_status="MODEL_NOT_IDENTIFIED")

    flight_data = _make_flight_data()
    sim = Simulator(aircraft=None, flight_data=flight_data, propulsion=_NotIdentifiedPropulsion())

    de = 0.01
    history = sim.run(
        PerturbationState(0.0, 0.0), duration=0.5, dt=0.1,
        control_schedule=_constant_elevator_schedule(de),
    )

    assert all(row["propulsion_model_status"] == "MODEL_NOT_IDENTIFIED" for row in history)
    assert all(math.isnan(row["total_thrust_N"]) for row in history)
    # State integration is unaffected by the NaN thrust value.
    expected_w, expected_q = _analytic_w_q(sim, 0.0, 0.0, de, history[-1]["time_s"])
    assert math.isclose(history[-1]["delta_w_mps"], expected_w, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(history[-1]["delta_q_rads"], expected_q, rel_tol=1e-6, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# save_results()
# ---------------------------------------------------------------------------

def test_save_results_writes_csv_with_expected_headers_and_rows(tmp_path, monkeypatch):
    import csv
    import simulation.simulator as simulator_module

    # save_results() locates its output directory as
    # Path(__file__).parent.parent / "data" / "processed", relative to
    # simulator.py's own location. Point that at an isolated tmp_path
    # instead of writing into the real project tree.
    monkeypatch.setattr(simulator_module, "__file__", str(tmp_path / "simulation" / "simulator.py"))

    sim = _make_simulator()
    sim.run(
        PerturbationState(0.0, 0.0), duration=0.2, dt=0.1,
        control_schedule=_constant_elevator_schedule(0.01),
    )

    sim.save_results("test_run_name")

    output_file = tmp_path / "data" / "processed" / "test_run_name.csv"
    assert output_file.exists()

    with open(output_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == list(sim.history[0].keys())
    assert len(rows) == len(sim.history)
    # Spot-check the first row's values round-trip through the CSV (as strings).
    assert rows[0]["time_s"] == str(sim.history[0]["time_s"])
    assert rows[0]["delta_w_mps"] == str(sim.history[0]["delta_w_mps"])
    
    # By default (without scenario_name), no meta.json is written
    assert not (tmp_path / "data" / "processed" / "test_run_name.meta.json").exists()


def test_save_results_with_scenario_metadata(tmp_path, monkeypatch):
    import json
    import simulation.simulator as simulator_module

    monkeypatch.setattr(simulator_module, "__file__", str(tmp_path / "simulation" / "simulator.py"))

    sim = _make_simulator()
    sim.run(
        PerturbationState(0.0, 0.0), duration=0.2, dt=0.1,
        control_schedule=_constant_elevator_schedule(0.01),
    )

    sim.save_results("test_meta_run", scenario_name="test_scenario_xyz")

    output_csv = tmp_path / "data" / "processed" / "test_meta_run.csv"
    output_meta = tmp_path / "data" / "processed" / "test_meta_run.meta.json"
    
    assert output_csv.exists()
    assert output_meta.exists()

    with open(output_meta, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
        
    assert meta_data == {"scenario_name": "test_scenario_xyz"}


def test_save_results_with_empty_history_writes_no_file(tmp_path, monkeypatch, capsys):
    import simulation.simulator as simulator_module

    monkeypatch.setattr(simulator_module, "__file__", str(tmp_path / "simulation" / "simulator.py"))

    sim = _make_simulator()  # .run() never called -- history stays empty
    sim.save_results("should_not_exist")

    assert not (tmp_path / "data" / "processed" / "should_not_exist.csv").exists()
    assert "No simulation data to save." in capsys.readouterr().out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))