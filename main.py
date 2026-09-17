"""
main.py

Application entry point / orchestrator for the NASA AirSTAR T-2 simulation.

Its job:
    1. Parse simple CLI arguments
    2. Load config.yaml exactly once
    3. Select a scenario using the simulation/scenarios.py registry
    4. Initialize models
    5. Construct the Simulator
    6. Execute the simulation (Simulator.run)
    7. Save results and print a concise summary

It does NOT define aerospace physics, aerodynamic equations, dimensional
conversion, or scenario specifics.
"""

import argparse
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Optional, List, Sequence

import yaml

from models.aircraft import Aircraft
from models.aerodynamics import AerodynamicsDatabase
from models.propulsion import PropulsionModel
from simulation.scenarios import get_all_scenarios
from simulation.simulator import Simulator


class ConfigError(Exception):
    """Raised when the configuration file cannot be found, read, or parsed.

    Kept distinct from a successfully-loaded config so callers never have
    to guess whether a falsy return means "failed to load" or "loaded
    fine, and it happens to be empty".
    """


# A run name should be a simple token usable as a filename component: no
# path separators, no "parent directory" tricks, no whitespace-only names.
# This is a positive allow-list rather than a blocklist of "dangerous"
# substrings, which is both simpler and harder to bypass.
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file exactly once.

    Returns:
        The parsed config as a dict. An empty (but syntactically valid)
        YAML document is returned as ``{}`` — that is a legitimate,
        successfully-loaded config, not a failure.

    Raises:
        ConfigError: if the file does not exist, cannot be read, cannot
            be parsed as YAML, or does not parse to a mapping at the
            top level. Any other, unexpected exception is left to
            propagate rather than being silently swallowed here.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found at {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
    except OSError as e:
        raise ConfigError(f"Failed to read config file '{config_path}': {e}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config '{config_path}': {e}") from e

    if loaded is None:
        # Syntactically valid but empty YAML document (e.g. an empty
        # file, or just "---"). This is a valid, if useless, config.
        return {}

    if not isinstance(loaded, dict):
        raise ConfigError(
            f"Config at '{config_path}' must be a mapping at the top level, "
            f"got {type(loaded).__name__}"
        )

    return loaded


def find_scenario(scenario_name: str, scenarios: Sequence):
    """Retrieves a scenario by name from an already-loaded scenario list.

    Callers are expected to call ``get_all_scenarios()`` once and pass the
    result in here, rather than each place that needs a scenario re-querying
    the registry.
    """
    for s in scenarios:
        if s.name == scenario_name:
            return s
    return None


def validate_overrides(duration: Optional[float], dt: Optional[float], output: Optional[str]) -> bool:
    """Validates CLI overrides for finiteness and basic safety."""
    if duration is not None:
        if not math.isfinite(duration) or duration <= 0.0:
            print(f"Error: --duration must be a positive finite number, got {duration}")
            return False
    if dt is not None:
        if not math.isfinite(dt) or dt <= 0.0:
            print(f"Error: --dt must be a positive finite number, got {dt}")
            return False

    if output is not None:
        if not output.strip():
            print("Error: --output cannot be empty or just whitespace.")
            return False
        if not _RUN_NAME_RE.fullmatch(output):
            print(
                "Error: --output must be a simple run name containing only "
                f"letters, digits, underscores, and hyphens, got '{output}'."
            )
            return False

    return True


def run_scenario(
    config: dict,
    scenario,
    override_duration: Optional[float],
    override_dt: Optional[float],
    override_output: Optional[str],
    debug: bool = False,
) -> int:
    """
    Executes a valid scenario that possesses a FlightCondition.
    """
    # 1. Print Provenance
    print(f"Scenario: {scenario.name}")
    print(f"Reference provenance: {scenario.reference_condition_provenance.value}")
    print(f"Control provenance: {scenario.control_schedule_provenance.value}")
    print(f"Classification: {scenario.classification}")
    print(f"Source: {scenario.source}")
    if scenario.assumptions:
        print("Assumptions:")
        for assumption in scenario.assumptions:
            print(f"  - {assumption}")
    print()

    # 2. Check for missing FlightCondition (e.g. Flight 33 Multisine)
    if scenario.flight_condition is None:
        print(
            f"Error: Scenario '{scenario.name}' is reference/design-only and "
            "cannot be executed by the current simulator because it has no "
            "derivative-backed FlightCondition."
        )
        return 1

    # 3. Apply Overrides
    duration_s = override_duration if override_duration is not None else scenario.duration_s
    dt_s = override_dt if override_dt is not None else scenario.dt_s
    run_name = override_output if override_output is not None else scenario.name

    if dt_s > duration_s:
        print(f"Error: Timestep dt ({dt_s}s) cannot exceed duration ({duration_s}s).")
        return 1

    # 4. Initialize Models using actual APIs
    try:
        aircraft = Aircraft.from_config(config)
        aero_db = AerodynamicsDatabase.from_config(config)
        propulsion = PropulsionModel.from_config(config)
        flight_data = aero_db.get_flight_test_data(scenario.flight_condition)
    except Exception as e:
        print(
            "Error: Failed to initialize models or retrieve flight data: "
            f"{type(e).__name__}: {e}"
        )
        if debug:
            traceback.print_exc()
        return 1

    # 5. Construct Simulator
    try:
        sim = Simulator(aircraft=aircraft, flight_data=flight_data, propulsion=propulsion)
    except Exception as e:
        print(f"Error: Failed to construct Simulator: {type(e).__name__}: {e}")
        if debug:
            traceback.print_exc()
        return 1

    # 6. Execute Scenario
    try:
        history = sim.run(
            initial_state=scenario.initial_state,
            duration=duration_s,
            dt=dt_s,
            control_schedule=scenario.control_schedule,
        )
    except Exception as e:
        print(f"Error: Simulation failed during execution: {type(e).__name__}: {e}")
        if debug:
            traceback.print_exc()
        return 1

    # 7. Save and Summarize
    try:
        sim.save_results(run_name=run_name, scenario_name=scenario.name)
    except Exception as e:
        print(f"Error: Failed to save results: {type(e).__name__}: {e}")
        if debug:
            traceback.print_exc()
        return 1

    if not history:
        print("Error: Simulation returned empty history.")
        return 1

    last_row = history[-1]

    print("\n--- Simulation Summary ---")
    print(f"Scenario              : {scenario.name}")
    print(f"Final Time            : {last_row['time_s']:.4f} s")
    print(f"Samples Recorded      : {len(history)}")
    print(f"Final delta_w         : {last_row['delta_w_mps']:.6g} m/s")
    print(f"Final delta_q         : {last_row['delta_q_rads']:.6g} rad/s")
    print(f"Propulsion Status     : {last_row.get('propulsion_model_status', 'UNKNOWN')}")
    print("--------------------------")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NASA AirSTAR T-2 Simulation Entry Point")
    parser.add_argument("--scenario", required=True, help="Name of the scenario to run")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--duration", type=float, help="Override simulation duration (seconds)")
    parser.add_argument("--dt", type=float, help="Override simulation timestep (seconds)")
    parser.add_argument("--output", type=str, help="Override output run name for CSV")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full tracebacks for unexpected errors instead of a one-line summary.",
    )

    args = parser.parse_args(argv)

    if not validate_overrides(args.duration, args.dt, args.output):
        return 1

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Error: {e}")
        return 1

    # Load the scenario registry exactly once and reuse it, whether the
    # requested scenario is found or we need to list what's available.
    scenarios = get_all_scenarios()
    scenario = find_scenario(args.scenario, scenarios)
    if not scenario:
        print(f"Error: Unknown scenario '{args.scenario}'")
        print("Available scenarios:")
        for s in scenarios:
            print(f"  - {s.name}")
        return 1

    return run_scenario(config, scenario, args.duration, args.dt, args.output, debug=args.debug)


if __name__ == "__main__":
    sys.exit(main())