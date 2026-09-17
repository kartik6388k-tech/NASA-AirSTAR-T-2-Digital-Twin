"""
tests/test_main.py

Test suite for main.py, the NASA AirSTAR T-2 simulation entry point.

Structure (per the project's own recommended layout):
    A. Unit / orchestration tests — mock only what main.py itself owns
       (low-level YAML parsing, the run_scenario call) so real project
       logic (the scenario registry, factories, Simulator) is exercised
       wherever it's cheap and safe to do so.
    B. Real end-to-end integration tests — use the actual project
       config.yaml and the real scenario registry; only
       Simulator.save_results is mocked, to avoid writing CSVs to disk
       during the test run.

Faithfulness note: every scenario used below is fetched from the real
`get_all_scenarios()` registry by its real `.name` string — none are
synthetic Mock objects with guessed attributes. `PerturbationState`,
`SimulationCommand`, `Provenance`, and `FlightCondition` are exercised
only indirectly, as they already appear inside those real scenario
objects; this file doesn't import or construct them directly because
main.py never does either — validating their own construction/validation
rules belongs in the test file for whichever module defines them, not
here.
"""

import math
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import main
from main import (
    ConfigError,
    load_config,
    find_scenario,
    validate_overrides,
    run_scenario,
)
from simulation.scenarios import get_all_scenarios

# tests/ sits one level below the repo root, alongside the real config.yaml.
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Real scenario names, copied verbatim from simulation/scenarios.py.
# Never hardcode a fixture name that isn't actually in the registry.
ZERO_INPUT_STABILITY = "zero_input_stability"
SMALL_DISTURBANCE = "small_disturbance_free_response"
FLIGHT_41 = "flight_41_reference_condition_run"
FLIGHT_15 = "flight_15_reference_condition_run"
MULTISINE = "multisine_excitation_concept"


def _real_scenario(name: str):
    """Fetch a real scenario object from the actual registry by name."""
    scenarios = get_all_scenarios()
    match = find_scenario(name, scenarios)
    assert match is not None, f"Expected '{name}' in the real scenario registry"
    return match


# ---------------------------------------------------------------------------
# A. Unit / orchestration tests
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_missing_file_raises_config_error(self, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(ConfigError, match="not found"):
            load_config(str(missing))

    def test_malformed_yaml_raises_config_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("aircraft: [unterminated\n")
        with pytest.raises(ConfigError, match="parse"):
            load_config(str(bad))

    def test_non_mapping_top_level_raises_config_error(self, tmp_path):
        not_a_mapping = tmp_path / "list.yaml"
        not_a_mapping.write_text("- one\n- two\n")
        with pytest.raises(ConfigError, match="mapping"):
            load_config(str(not_a_mapping))

    def test_empty_file_is_a_valid_but_empty_config(self, tmp_path):
        # An empty YAML document is not a load failure -- it's a
        # legitimate (if useless) config. Downstream factories are
        # responsible for rejecting it as unusable; see
        # TestRunScenarioOrchestration.test_empty_config_fails_clearly_downstream.
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        assert load_config(str(empty)) == {}

    def test_config_parsed_exactly_once(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("aircraft:\n  name: test\n")
        with patch("main.yaml.safe_load", wraps=yaml.safe_load) as mock_load:
            load_config(str(cfg_file))
            assert mock_load.call_count == 1

    def test_config_parsed_exactly_once_through_main(self):
        # Per the recommended structure: mock the low-level parser and
        # stop execution right after config load by mocking run_scenario,
        # so this stays a fast unit test with no real model init.
        with patch("main.yaml.safe_load", wraps=yaml.safe_load) as mock_load, \
             patch("main.run_scenario", return_value=0):
            main.main(["--scenario", FLIGHT_41, "--config", str(CONFIG_PATH)])
            assert mock_load.call_count == 1


class TestScenarioRegistryReuse:
    def test_find_scenario_matches_real_object_by_name(self):
        scenarios = get_all_scenarios()
        found = find_scenario(FLIGHT_41, scenarios)
        assert found is not None
        assert found.name == FLIGHT_41

    def test_unknown_name_returns_none(self):
        scenarios = get_all_scenarios()
        assert find_scenario("not_a_real_scenario", scenarios) is None

    def test_registry_loaded_once_on_success(self):
        with patch("main.get_all_scenarios", wraps=get_all_scenarios) as mock_registry, \
             patch("main.run_scenario", return_value=0):
            main.main(["--scenario", FLIGHT_41, "--config", str(CONFIG_PATH)])
            assert mock_registry.call_count == 1

    def test_registry_loaded_once_on_unknown_scenario(self):
        # This is the exact case the fix targets: main() used to call
        # get_all_scenarios() a second time here, just to print names.
        with patch("main.get_all_scenarios", wraps=get_all_scenarios) as mock_registry:
            exit_code = main.main(
                ["--scenario", "totally_bogus", "--config", str(CONFIG_PATH)]
            )
        assert exit_code == 1
        assert mock_registry.call_count == 1


class TestValidateOverrides:
    @pytest.mark.parametrize("duration", [0.0, -1.0, math.inf, math.nan])
    def test_rejects_bad_duration(self, duration):
        assert validate_overrides(duration, None, None) is False

    @pytest.mark.parametrize("dt", [0.0, -0.01, math.inf, math.nan])
    def test_rejects_bad_dt(self, dt):
        assert validate_overrides(None, dt, None) is False

    @pytest.mark.parametrize(
        "output", ["", "   ", "run/one", "run\\one", "..", "run..name", "run name"]
    )
    def test_rejects_bad_output_names(self, output):
        assert validate_overrides(None, None, output) is False

    def test_bad_output_name_message_matches_current_wording(self, capsys):
        # Pin the actual printed message so a future rewrite of
        # validate_overrides() can't silently drift from what's asserted
        # elsewhere (docs, other tests, CI log expectations).
        assert validate_overrides(None, None, "bad/name") is False
        captured = capsys.readouterr()
        assert "must be a simple run name" in captured.out
        assert "not a path with slashes" not in captured.out

    @pytest.mark.parametrize("output", ["run_1", "Flight-41-rerun", "abc123"])
    def test_accepts_simple_run_names(self, output):
        assert validate_overrides(None, None, output) is True

    def test_accepts_no_overrides(self):
        assert validate_overrides(None, None, None) is True


class TestRunScenarioOrchestration:
    def test_multisine_scenario_without_flight_condition_fails_cleanly(self):
        scenario = _real_scenario(MULTISINE)
        assert scenario.flight_condition is None
        config = load_config(str(CONFIG_PATH))
        result = run_scenario(config, scenario, None, None, None)
        assert result == 1

    def test_dt_greater_than_duration_is_rejected(self):
        scenario = _real_scenario(FLIGHT_41)
        config = load_config(str(CONFIG_PATH))
        result = run_scenario(
            config, scenario,
            override_duration=1.0,
            override_dt=2.0,
            override_output=None,
        )
        assert result == 1

    def test_cli_dt_greater_than_duration_returns_one(self, capsys):
        # Same check as above, but plumbed through the real argparse CLI
        # entry point rather than calling run_scenario() directly. This
        # never reaches model init (it fails before step 4), so it needs
        # no Simulator.save_results mock.
        exit_code = main.main([
            "--scenario", FLIGHT_41,
            "--config", str(CONFIG_PATH),
            "--duration", "1.0",
            "--dt", "2.0",
        ])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "cannot exceed duration" in captured.out

    def test_empty_config_fails_clearly_downstream(self, capsys):
        # An empty-but-valid config ({}) is main.py's business to accept;
        # rejecting it as *unusable* is the factories' job. Confirm that
        # rejection actually happens and is reported clearly.
        scenario = _real_scenario(FLIGHT_41)
        result = run_scenario({}, scenario, None, None, None)
        assert result == 1
        captured = capsys.readouterr()
        assert "Failed to initialize models" in captured.out

    def test_unexpected_model_exception_reports_type_and_message(self, capsys):
        scenario = _real_scenario(FLIGHT_41)
        config = load_config(str(CONFIG_PATH))
        with patch("main.Aircraft.from_config", side_effect=KeyError("aircraft")):
            result = run_scenario(config, scenario, None, None, None)
        assert result == 1
        captured = capsys.readouterr()
        assert "KeyError" in captured.out

    def test_debug_flag_exposes_traceback(self, capsys):
        scenario = _real_scenario(FLIGHT_41)
        config = load_config(str(CONFIG_PATH))
        with patch("main.Aircraft.from_config", side_effect=RuntimeError("boom")):
            run_scenario(config, scenario, None, None, None, debug=True)
        captured = capsys.readouterr()
        assert "Traceback" in captured.out or "Traceback" in captured.err

    def test_without_debug_flag_traceback_is_suppressed(self, capsys):
        scenario = _real_scenario(FLIGHT_41)
        config = load_config(str(CONFIG_PATH))
        with patch("main.Aircraft.from_config", side_effect=RuntimeError("boom")):
            run_scenario(config, scenario, None, None, None, debug=False)
        captured = capsys.readouterr()
        assert "Traceback" not in captured.out and "Traceback" not in captured.err


# ---------------------------------------------------------------------------
# B. Real end-to-end integration tests
# ---------------------------------------------------------------------------

class TestRealEndToEndExecution:
    """
    Exercises the real scenario registry, the real factories (Aircraft,
    AerodynamicsDatabase, PropulsionModel), and the real Simulator against
    the actual project config.yaml. Only Simulator.save_results is mocked,
    to avoid writing CSVs during the test run.
    """

    @patch("simulation.simulator.Simulator.save_results")
    def test_valid_execution_flight_41(self, mock_save):
        config = load_config(str(CONFIG_PATH))
        scenario = _real_scenario(FLIGHT_41)
        result = run_scenario(config, scenario, None, None, None)
        assert result == 0
        mock_save.assert_called_once_with(run_name=FLIGHT_41, scenario_name=FLIGHT_41)

    @patch("simulation.simulator.Simulator.save_results")
    def test_valid_execution_flight_15(self, mock_save):
        config = load_config(str(CONFIG_PATH))
        scenario = _real_scenario(FLIGHT_15)
        result = run_scenario(config, scenario, None, None, None)
        assert result == 0
        mock_save.assert_called_once_with(run_name=FLIGHT_15, scenario_name=FLIGHT_15)

    @patch("simulation.simulator.Simulator.save_results")
    def test_valid_execution_zero_input_stability(self, mock_save):
        # Defaults to FlightCondition.FLIGHT_41 in the real registry.
        config = load_config(str(CONFIG_PATH))
        scenario = _real_scenario(ZERO_INPUT_STABILITY)
        result = run_scenario(config, scenario, None, None, None)
        assert result == 0
        mock_save.assert_called_once_with(run_name=ZERO_INPUT_STABILITY, scenario_name=ZERO_INPUT_STABILITY)

    @patch("simulation.simulator.Simulator.save_results")
    def test_valid_execution_small_disturbance(self, mock_save):
        config = load_config(str(CONFIG_PATH))
        scenario = _real_scenario(SMALL_DISTURBANCE)
        result = run_scenario(config, scenario, None, None, None)
        assert result == 0
        mock_save.assert_called_once_with(run_name=SMALL_DISTURBANCE, scenario_name=SMALL_DISTURBANCE)

    @patch("simulation.simulator.Simulator.save_results")
    def test_output_override_becomes_run_name(self, mock_save):
        config = load_config(str(CONFIG_PATH))
        scenario = _real_scenario(FLIGHT_41)
        run_scenario(config, scenario, None, None, "custom_run_name")
        mock_save.assert_called_once_with(run_name="custom_run_name", scenario_name=FLIGHT_41)

    def test_full_cli_invocation_flight_41(self, capsys):
        with patch("simulation.simulator.Simulator.save_results"):
            exit_code = main.main(
                ["--scenario", FLIGHT_41, "--config", str(CONFIG_PATH)]
            )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert FLIGHT_41 in captured.out
        assert "Simulation Summary" in captured.out