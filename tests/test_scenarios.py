"""
tests/test_scenarios.py

Tests for simulation/scenarios.py.

These tests check the scenario-definition layer only (provenance,
metadata, control-schedule shape). They do not re-validate RK4 numerics
or aerodynamic physics -- that is tests/test_simulator.py's job.
"""

import math
import sys
from pathlib import Path

import pytest
import yaml

# Allow `import models...` / `import simulation...` when tests are run from
# the project root (matches tests/test_simulator.py's convention).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.aerodynamics import AerodynamicsDatabase, FlightCondition
from models.flight_dynamics import PerturbationState
from simulation.simulator import SimulationCommand
from simulation.scenarios import (
    Provenance,
    SimulationScenario,
    create_zero_input_stability_scenario,
    create_small_disturbance_scenario,
    create_flight_41_scenario,
    create_flight_15_scenario,
    create_multisine_excitation_scenario,
    get_all_scenarios,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@pytest.fixture(scope="module")
def aero_db() -> AerodynamicsDatabase:
    with open(_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return AerodynamicsDatabase.from_config(config)


def _mps_to_ft_s(v: float) -> float:
    return v * 3.280839895


def _rad_to_deg(r: float) -> float:
    return r * 180.0 / math.pi


def _m_to_ft(m: float) -> float:
    return m * 3.280839895


# ---------------------------------------------------------------------------
# 1 & 2: every constructor returns a valid SimulationScenario with valid
# duration/dt
# ---------------------------------------------------------------------------

def test_all_scenarios_are_valid_simulation_scenarios():
    for scenario in get_all_scenarios():
        assert isinstance(scenario, SimulationScenario)
        assert isinstance(scenario.duration_s, float) and scenario.duration_s > 0.0
        assert isinstance(scenario.dt_s, float) and scenario.dt_s > 0.0
        assert math.isfinite(scenario.duration_s)
        assert math.isfinite(scenario.dt_s)
        assert scenario.dt_s <= scenario.duration_s


# ---------------------------------------------------------------------------
# 3: every control schedule returns SimulationCommand (checked at several
# times, not just t=0)
# ---------------------------------------------------------------------------

def test_every_control_schedule_returns_simulation_command():
    for scenario in get_all_scenarios():
        for t in (0.0, scenario.dt_s, scenario.duration_s / 2.0, scenario.duration_s):
            cmd = scenario.control_schedule(t)
            assert isinstance(cmd, SimulationCommand)
            cmd.validate()  # must not raise


# ---------------------------------------------------------------------------
# 4 & 5: Flight 41 / Flight 15 use the correct NASA-verified reference
# condition (cross-checked against config.yaml via AerodynamicsDatabase)
# ---------------------------------------------------------------------------

def test_flight_41_scenario_uses_correct_verified_reference_condition(aero_db):
    scenario = create_flight_41_scenario()
    assert scenario.flight_condition is FlightCondition.FLIGHT_41

    nominal = aero_db.get_flight_test_data(FlightCondition.FLIGHT_41).nominal_condition
    assert math.isclose(_mps_to_ft_s(nominal.airspeed_mps), 139.1, abs_tol=0.05)
    assert math.isclose(_rad_to_deg(nominal.angle_of_attack_rad), 4.077, abs_tol=0.01)
    assert math.isclose(_m_to_ft(nominal.altitude_m), 1227.0, abs_tol=0.5)
    assert math.isclose(nominal.throttle, 0.290, abs_tol=1e-9)


def test_flight_15_scenario_uses_correct_verified_reference_condition(aero_db):
    scenario = create_flight_15_scenario()
    assert scenario.flight_condition is FlightCondition.FLIGHT_15

    nominal = aero_db.get_flight_test_data(FlightCondition.FLIGHT_15).nominal_condition
    assert math.isclose(_mps_to_ft_s(nominal.airspeed_mps), 136.0, abs_tol=0.05)
    assert math.isclose(_rad_to_deg(nominal.angle_of_attack_rad), 3.893, abs_tol=0.01)
    assert math.isclose(_m_to_ft(nominal.altitude_m), 1467.0, abs_tol=0.5)
    assert math.isclose(nominal.throttle, 0.315, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 6: Flight 41 and Flight 15 are never mixed
# ---------------------------------------------------------------------------

def test_flight_41_and_flight_15_scenarios_are_never_mixed():
    f41 = create_flight_41_scenario()
    f15 = create_flight_15_scenario()
    assert f41.flight_condition is FlightCondition.FLIGHT_41
    assert f15.flight_condition is FlightCondition.FLIGHT_15
    assert f41.flight_condition != f15.flight_condition


# ---------------------------------------------------------------------------
# 7 & 8: Flight 41/15 marked NASA_VERIFIED_REFERENCE_CONDITION with
# PROJECT_DEFINED control schedules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "factory", [create_flight_41_scenario, create_flight_15_scenario]
)
def test_flight_reference_scenarios_have_correct_provenance(factory):
    scenario = factory()
    assert (
        scenario.reference_condition_provenance
        == Provenance.NASA_VERIFIED_REFERENCE_CONDITION
    )
    assert scenario.control_schedule_provenance == Provenance.PROJECT_DEFINED


# ---------------------------------------------------------------------------
# 9 & 10: multisine scenario is marked NASA_VERIFIED_REFERENCE +
# PROJECT_DEFINED_WAVEFORM, and no exact phase/amplitude is claimed as NASA
# ---------------------------------------------------------------------------

def test_multisine_scenario_has_correct_composite_provenance():
    scenario = create_multisine_excitation_scenario()
    assert scenario.reference_condition_provenance == Provenance.NASA_VERIFIED_REFERENCE
    assert scenario.control_schedule_provenance == Provenance.PROJECT_DEFINED_WAVEFORM
    assert scenario.classification == "NASA_VERIFIED_REFERENCE + PROJECT_DEFINED_WAVEFORM"


def test_multisine_scenario_does_not_claim_a_verified_flight_condition():
    scenario = create_multisine_excitation_scenario()
    # No FlightCondition enum member exists for Flight 33 in this project;
    # it must not be silently mapped onto FLIGHT_41 or FLIGHT_15.
    assert scenario.flight_condition is None
    assert scenario.reference_condition_provenance != Provenance.NASA_VERIFIED_REFERENCE_CONDITION


def test_multisine_waveform_is_project_defined_not_exact_nasa_data():
    scenario = create_multisine_excitation_scenario()
    # The waveform provenance is explicitly the "not exact" category.
    assert scenario.control_schedule_provenance == Provenance.PROJECT_DEFINED_WAVEFORM
    assert scenario.control_schedule_provenance != Provenance.NASA_VERIFIED_REFERENCE
    assert scenario.control_schedule_provenance != Provenance.NASA_VERIFIED_REFERENCE_CONDITION


# ---------------------------------------------------------------------------
# 11: zero-input scenario really produces zero control commands
# ---------------------------------------------------------------------------

def test_zero_input_scenario_produces_zero_commands_everywhere():
    scenario = create_zero_input_stability_scenario()
    assert scenario.initial_state.delta_w == 0.0
    assert scenario.initial_state.delta_q == 0.0
    for t in (0.0, 0.01, scenario.duration_s / 3.0, scenario.duration_s):
        cmd = scenario.control_schedule(t)
        assert cmd.delta_elevator_rad == 0.0
        assert cmd.left_throttle == 0.0
        assert cmd.right_throttle == 0.0
        assert cmd.left_rpm == 0.0
        assert cmd.right_rpm == 0.0
        assert cmd.left_lambda == 0.0
        assert cmd.right_lambda == 0.0


# ---------------------------------------------------------------------------
# 12: small-disturbance scenario is clearly TEST_ONLY
# ---------------------------------------------------------------------------

def test_small_disturbance_scenario_is_test_only_and_actually_perturbed():
    scenario = create_small_disturbance_scenario()
    assert scenario.reference_condition_provenance == Provenance.TEST_ONLY
    assert scenario.control_schedule_provenance == Provenance.TEST_ONLY
    assert scenario.initial_state.delta_w != 0.0 or scenario.initial_state.delta_q != 0.0
    # Free response: still zero control input.
    cmd = scenario.control_schedule(0.0)
    assert cmd.delta_elevator_rad == 0.0


# ---------------------------------------------------------------------------
# 13: invalid scenario parameters are rejected
# ---------------------------------------------------------------------------

def _valid_kwargs():
    return dict(
        name="valid_scenario",
        description="A valid scenario for negative testing.",
        flight_condition=FlightCondition.FLIGHT_41,
        reference_condition_provenance=Provenance.TEST_ONLY,
        initial_state=PerturbationState(delta_w=0.0, delta_q=0.0),
        duration_s=1.0,
        dt_s=0.1,
        control_schedule=lambda t: SimulationCommand(delta_elevator_rad=0.0),
        control_schedule_provenance=Provenance.TEST_ONLY,
        source="TEST_ONLY",
        assumptions=("test assumption",),
    )


def test_baseline_valid_kwargs_actually_construct():
    SimulationScenario(**_valid_kwargs())


@pytest.mark.parametrize(
    "override,error_type",
    [
        ({"duration_s": 0.0}, ValueError),
        ({"duration_s": -1.0}, ValueError),
        ({"duration_s": float("nan")}, ValueError),
        ({"dt_s": 0.0}, ValueError),
        ({"dt_s": -0.1}, ValueError),
        ({"dt_s": 2.0}, ValueError),  # dt_s > duration_s
        ({"control_schedule": "not_callable"}, TypeError),
        ({"initial_state": (0.0, 0.0)}, TypeError),
        ({"flight_condition": "FLIGHT_41"}, TypeError),
        ({"name": ""}, ValueError),
        ({"source": "   "}, ValueError),
    ],
)
def test_invalid_scenario_parameters_are_rejected(override, error_type):
    kwargs = _valid_kwargs()
    kwargs.update(override)
    with pytest.raises(error_type):
        SimulationScenario(**kwargs)


def test_nasa_verified_reference_condition_requires_a_flight_condition():
    kwargs = _valid_kwargs()
    kwargs["flight_condition"] = None
    kwargs["reference_condition_provenance"] = Provenance.NASA_VERIFIED_REFERENCE_CONDITION
    with pytest.raises(ValueError):
        SimulationScenario(**kwargs)


def test_control_schedule_must_return_simulation_command():
    kwargs = _valid_kwargs()
    kwargs["control_schedule"] = lambda t: "not_a_command"
    with pytest.raises(TypeError):
        SimulationScenario(**kwargs)


def test_control_schedule_output_must_itself_be_valid():
    kwargs = _valid_kwargs()
    kwargs["control_schedule"] = lambda t: SimulationCommand(
        delta_elevator_rad=0.0, left_throttle=5.0  # out of [0, 1]
    )
    with pytest.raises(ValueError):
        SimulationScenario(**kwargs)


# ---------------------------------------------------------------------------
# 14: no full 6-DOF assumptions are introduced
# ---------------------------------------------------------------------------

def test_no_scenario_introduces_a_full_6dof_state():
    # PerturbationState is the project's 2-state (delta_w, delta_q) short
    # period state. Every scenario's initial_state must be exactly that
    # type -- not some richer 6-DOF state this module might otherwise be
    # tempted to invent.
    for scenario in get_all_scenarios():
        assert isinstance(scenario.initial_state, PerturbationState)
        assert set(vars(scenario.initial_state).keys()) == {"delta_w", "delta_q"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))