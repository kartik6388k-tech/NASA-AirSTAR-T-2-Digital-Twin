"""
tests/test_turbulence.py

Tests for the Project-Defined Longitudinal 1-Cosine Gust model, its integration
into the reduced-order flight dynamics equations, RK4 simulation, telemetry
logging, and preservation of baseline scenarios.
"""

import math
import sys
from pathlib import Path
import pytest
import yaml

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.aerodynamics import FlightCondition, AerodynamicsDatabase
from models.flight_dynamics import (
    FlightDynamics,
    PerturbationState,
    ReferenceTrim,
    ShortPeriodDerivatives,
)
from models.turbulence import LongitudinalCosineGust, PROVENANCE_TAG
from models.aircraft import Aircraft
from models.propulsion import PropulsionModel
from simulation.simulator import Simulator, SimulationCommand
from simulation.scenarios import (
    create_flight_41_scenario,
    create_flight_41_gust_scenario,
    get_all_scenarios,
)
from dashboard.app import TelemetryState, make_handler


# --------------------------------------------------------------------------
# 1. LongitudinalCosineGust Unit Tests
# --------------------------------------------------------------------------

class TestLongitudinalCosineGustWaveform:
    """Verifies deterministic 1-cosine gust shape and window bounds."""

    def test_gust_waveform_mathematical_profile(self):
        gust = LongitudinalCosineGust(
            amplitude_mps=2.0,
            duration_s=1.0,
            start_time_s=1.0,
            enabled=True,
        )
        # At start time: 0.0
        assert math.isclose(gust(1.0), 0.0, abs_tol=1e-12)
        # At peak (t = 1.5): exactly amplitude
        assert math.isclose(gust(1.5), 2.0, rel_tol=1e-9)
        # At quarter points (t = 1.25, 1.75): half amplitude
        assert math.isclose(gust(1.25), 1.0, rel_tol=1e-9)
        assert math.isclose(gust(1.75), 1.0, rel_tol=1e-9)
        # At end time (t = 2.0): 0.0
        assert math.isclose(gust(2.0), 0.0, abs_tol=1e-12)

    def test_zero_outside_window(self):
        gust = LongitudinalCosineGust(
            amplitude_mps=2.0,
            duration_s=1.0,
            start_time_s=1.0,
            enabled=True,
        )
        # Strictly before window
        assert gust(0.0) == 0.0
        assert gust(0.5) == 0.0
        assert gust(0.999) == 0.0
        # Strictly after window
        assert gust(2.001) == 0.0
        assert gust(3.0) == 0.0
        assert gust(10.0) == 0.0

    def test_disabled_returns_zero_everywhere(self):
        gust = LongitudinalCosineGust(
            amplitude_mps=2.0,
            duration_s=1.0,
            start_time_s=1.0,
            enabled=False,
        )
        assert gust(0.5) == 0.0
        assert gust(1.5) == 0.0
        assert gust(2.0) == 0.0

    def test_from_config_defaults(self):
        config = {
            "project_defined_parameters": {
                "longitudinal_gust": {
                    "gust_amplitude_mps": 1.5,
                    "gust_duration_s": 1.2,
                    "start_time_s": 0.8,
                    "enabled": False,
                }
            }
        }
        gust = LongitudinalCosineGust.from_config(config)
        assert gust.amplitude_mps == 1.5
        assert gust.duration_s == 1.2
        assert gust.start_time_s == 0.8
        assert gust.enabled is False
        assert gust.provenance == PROVENANCE_TAG

        # Test enable_override
        gust_active = LongitudinalCosineGust.from_config(config, enable_override=True)
        assert gust_active.enabled is True

    def test_input_validation(self):
        with pytest.raises(ValueError):
            LongitudinalCosineGust(amplitude_mps=1.0, duration_s=0.0)
        with pytest.raises(ValueError):
            LongitudinalCosineGust(amplitude_mps=1.0, duration_s=-1.0)
        with pytest.raises(ValueError):
            LongitudinalCosineGust(amplitude_mps=1.0, duration_s=1.0, start_time_s=-0.5)


# --------------------------------------------------------------------------
# 2. FlightDynamics State Derivatives with w_gust
# --------------------------------------------------------------------------

class TestFlightDynamicsGustIntegration:
    """Verifies aerodynamic relative vertical velocity delta_w_aero = delta_w - w_gust."""

    @pytest.fixture
    def fd(self):
        trim = ReferenceTrim(flight_condition=FlightCondition.FLIGHT_41, u0=50.0)
        return FlightDynamics(trim)

    @pytest.fixture
    def derivs(self):
        return ShortPeriodDerivatives(
            flight_condition=FlightCondition.FLIGHT_41,
            Zw=-50.0, Zq=-5.0, Zde=-1000.0,
            Mw=-2.0, Mq=-10.0, Mde=-500.0,
        )

    def test_zero_gust_matches_standard_derivatives(self, fd, derivs):
        state = PerturbationState(delta_w=1.5, delta_q=0.02)
        res_baseline = fd.calculate_state_derivatives(
            mass=1000.0, iyy=2000.0, state=state, derivatives=derivs, delta_e=0.01, w_gust=0.0
        )
        res_default = fd.calculate_state_derivatives(
            mass=1000.0, iyy=2000.0, state=state, derivatives=derivs, delta_e=0.01
        )
        assert res_baseline['derivatives']['delta_w_dot'] == res_default['derivatives']['delta_w_dot']
        assert res_baseline['derivatives']['delta_q_dot'] == res_default['derivatives']['delta_q_dot']

    def test_hand_calculation_gust_relative_velocity(self, fd, derivs):
        """
        x = [delta_w, delta_q] = [2.0, 0.1], delta_e = 0.05, w_gust = 0.5
        delta_w_aero = 2.0 - 0.5 = 1.5

        Z = Zw * delta_w_aero + Zq * delta_q + Zde * delta_e
          = (-50)*(1.5) + (-5)*(0.1) + (-1000)*(0.05)
          = -75.0 - 0.5 - 50.0 = -125.5 N

        M = Mw * delta_w_aero + Mq * delta_q + Mde * delta_e
          = (-2)*(1.5) + (-10)*(0.1) + (-500)*(0.05)
          = -3.0 - 1.0 - 25.0 = -29.0 N*m

        wdot = Z/m + Ue * q = -125.5 / 1000.0 + 50.0 * 0.1 = -0.1255 + 5.0 = 4.8745
        qdot = M / Iy = -29.0 / 2000.0 = -0.0145
        az   = Z/m = -0.1255
        """
        state = PerturbationState(delta_w=2.0, delta_q=0.1)
        res = fd.calculate_state_derivatives(
            mass=1000.0,
            iyy=2000.0,
            state=state,
            derivatives=derivs,
            delta_e=0.05,
            w_gust=0.5,
        )

        assert math.isclose(res['diagnostics']['delta_w_aero_mps'], 1.5, rel_tol=1e-9)
        assert math.isclose(res['diagnostics']['delta_Z_force_N'], -125.5, rel_tol=1e-9)
        assert math.isclose(res['diagnostics']['delta_M_pitch_Nm'], -29.0, rel_tol=1e-9)
        assert math.isclose(res['derivatives']['delta_w_dot'], 4.8745, rel_tol=1e-9)
        assert math.isclose(res['derivatives']['delta_q_dot'], -0.0145, rel_tol=1e-9)
        assert math.isclose(res['outputs']['perturbation_az'], -0.1255, rel_tol=1e-9)

    def test_state_vector_keys_preserved_without_6dof_states(self, fd, derivs):
        state = PerturbationState(delta_w=1.0, delta_q=0.0)
        res = fd.calculate_state_derivatives(
            mass=1000.0, iyy=2000.0, state=state, derivatives=derivs, delta_e=0.0, w_gust=1.0
        )
        assert set(res['outputs'].keys()) == {"delta_w", "delta_q", "perturbation_az"}


# --------------------------------------------------------------------------
# 3. Simulator Integration & CSV Telemetry
# --------------------------------------------------------------------------

class TestSimulatorGustResponse:
    """Verifies that gust excites short-period dynamics and is logged in telemetry."""

    @pytest.fixture
    def config(self):
        with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def sim(self, config):
        aircraft = Aircraft.from_config(config)
        aero_db = AerodynamicsDatabase.from_config(config)
        propulsion = PropulsionModel.from_config(config)
        flight_data = aero_db.get_flight_test_data(FlightCondition.FLIGHT_41)
        return Simulator(aircraft=aircraft, flight_data=flight_data, propulsion=propulsion)

    def test_zero_initial_state_excited_by_gust(self, sim):
        """Zero initial state and zero elevator: gust alone excites delta_w and delta_q."""
        gust = LongitudinalCosineGust(
            amplitude_mps=2.0,
            duration_s=1.0,
            start_time_s=0.5,
            enabled=True,
        )
        def zero_control(t: float):
            return SimulationCommand(delta_elevator_rad=0.0)

        initial = PerturbationState(delta_w=0.0, delta_q=0.0)
        history = sim.run(initial, duration=3.0, dt=0.02, control_schedule=zero_control, gust_model=gust)

        # Before gust (t < 0.5): aircraft remains at trim
        row_pre = [r for r in history if math.isclose(r["time_s"], 0.2, abs_tol=1e-5)][0]
        assert math.isclose(row_pre["delta_w_mps"], 0.0, abs_tol=1e-6)
        assert math.isclose(row_pre["delta_q_rads"], 0.0, abs_tol=1e-6)
        assert math.isclose(row_pre["w_gust_mps"], 0.0, abs_tol=1e-6)

        # During gust (t = 1.0): gust is at peak, delta_w and delta_q are excited
        row_gust = [r for r in history if math.isclose(r["time_s"], 1.0, abs_tol=1e-5)][0]
        assert math.isclose(row_gust["w_gust_mps"], 2.0, rel_tol=1e-3)
        assert abs(row_gust["delta_w_mps"]) > 0.05
        assert abs(row_gust["delta_q_rads"]) > 0.001

        # Check telemetry fields are present in every row
        for row in history:
            assert "w_gust_mps" in row
            assert "delta_w_aero_mps" in row
            assert math.isfinite(row["w_gust_mps"])
            assert math.isfinite(row["delta_w_aero_mps"])

    def test_non_gust_run_preserves_baseline_identically(self, sim):
        """When gust_model is None, simulation output is identical to existing baseline."""
        def step_control(t: float):
            return SimulationCommand(delta_elevator_rad=-0.02 if t >= 0.5 else 0.0)

        initial = PerturbationState(delta_w=0.0, delta_q=0.0)
        h1 = sim.run(initial, duration=2.0, dt=0.02, control_schedule=step_control, gust_model=None)
        h2 = sim.run(initial, duration=2.0, dt=0.02, control_schedule=step_control)

        for r1, r2 in zip(h1, h2):
            assert r1["time_s"] == r2["time_s"]
            assert r1["delta_w_mps"] == r2["delta_w_mps"]
            assert r1["delta_q_rads"] == r2["delta_q_rads"]
            assert r1["w_gust_mps"] == 0.0

    def test_gust_response_is_bounded_and_damped(self, sim):
        """1.5 m/s gust excites physically bounded perturbations that damp to near-zero."""
        gust = LongitudinalCosineGust(
            amplitude_mps=1.5,
            duration_s=1.0,
            start_time_s=1.0,
            enabled=True,
        )
        def zero_control(t: float):
            return SimulationCommand(delta_elevator_rad=0.0)

        initial = PerturbationState(delta_w=0.0, delta_q=0.0)
        history = sim.run(initial, duration=10.0, dt=0.02, control_schedule=zero_control, gust_model=gust)

        ws = [r["delta_w_mps"] for r in history]
        qs = [r["delta_q_rads"] for r in history]

        # Verify physical boundedness (1.5 m/s gust produces ~1.66 m/s peak w, ~0.11 rad/s peak q)
        assert max(ws) < 2.5
        assert min(ws) > -1.5
        assert max(qs) < 0.25
        assert min(qs) > -0.25

        # Verify short-period damping brings states back to trim by t=10s
        final_w = history[-1]["delta_w_mps"]
        final_q = history[-1]["delta_q_rads"]
        assert math.isclose(final_w, 0.0, abs_tol=1e-5)
        assert math.isclose(final_q, 0.0, abs_tol=1e-5)



# --------------------------------------------------------------------------
# 4. Scenario Registry
# --------------------------------------------------------------------------

def test_flight_41_gust_scenario_registered():
    scenarios = get_all_scenarios()
    names = [s.name for s in scenarios]
    assert "flight_41_gust_response_run" in names

    scenario = create_flight_41_gust_scenario()
    assert scenario.flight_condition == FlightCondition.FLIGHT_41
    assert set(vars(scenario.initial_state).keys()) == {"delta_w", "delta_q"}
    assert "PROJECT_DEFINED_TURBULENCE" in scenario.source


# --------------------------------------------------------------------------
# 5. Dashboard Telemetry API with Environment
# --------------------------------------------------------------------------

def test_telemetry_state_serves_environment_gust(tmp_path):
    # CSV with gust data
    csv_content = """time_s,delta_w_mps,delta_q_rads,delta_e_rad,w_gust_mps,propulsion_model_status
0.0,0.0,0.0,0.0,0.0,MODEL_NOT_IDENTIFIED
0.02,-0.05,0.002,-0.01,1.5,MODEL_NOT_IDENTIFIED
"""
    csv_file = tmp_path / "gust_run.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    state = TelemetryState(data_dir=tmp_path)
    state.load_run("gust_run")

    frame0 = state.get_telemetry()
    assert frame0["environment"]["w_gust_mps"] == 0.0
    assert frame0["environment"]["provenance"] == "PROJECT_DEFINED_TURBULENCE"

    frame1 = state.get_telemetry()
    assert frame1["environment"]["w_gust_mps"] == 1.5
    assert frame1["environment"]["provenance"] == "PROJECT_DEFINED_TURBULENCE"
