"""
NASA AirSTAR T-2 engineering dashboard (V1).

Purpose
-------
This module is a presentation / post-processing layer for the existing NASA
AirSTAR 5.5% GTM / T-2 Digital Twin.  It intentionally does not own or
reimplement aircraft physics, RK4 integration, sensor modelling, state
estimation, health rules, scenario definitions, dimensional conversion, or
configuration mutation.

Supported data sources
----------------------
1. Existing Simulator CSV files in ``data/processed/*.csv``.
2. Existing ``digital_twin.integration.IntegratedStepResult`` sequences passed
   programmatically to :func:`prepare_dashboard_data`.

The CLI web application is intentionally read-only and uses only the Python
standard library.  No new UI dependency is required.  It displays saved CSV
runs; sensor/estimate/health sections are displayed only when genuine
integration-layer results are supplied to the preparation API.

V1 is strictly post-processing.  There is deliberately no dashboard control
that executes simulations or mutates configuration.  The explicit path for
sensor/estimate/health content is programmatic: run the existing
``digital_twin.integration`` pipeline and pass the resulting
``IntegratedStepResult`` sequence to
``prepare_dashboard_data(..., integrated_results=...)``; when no such results
are supplied, those sections honestly display "Unavailable".

Provenance behaviour
--------------------
Provenance/status strings are displayed exactly at their owning layer.  The
module never promotes PROJECT_DEFINED data to NASA_VERIFIED, never renames
``MODEL_NOT_IDENTIFIED`` into a propulsion failure, and never labels simulator
truth as measured/real flight data.

Limitations
-----------
The current Simulator CSV schema contains simulation truth and diagnostics, not
sensor measurements, estimator outputs, or HealthMonitor results.  Therefore a
CSV-only V1 view intentionally marks those sections unavailable.  Callers that
already own real ``IntegratedStepResult`` objects may pass them to
``prepare_dashboard_data(..., integrated_results=...)``.

Launch
------
From the project root::

    python dashboard/app.py --host 127.0.0.1 --port 8000

Then open ``http://127.0.0.1:8000``.

Architecture boundary
---------------------
    dashboard = presentation / analysis layer
    dashboard != physics model
    dashboard != simulator
    dashboard != estimator
    dashboard != health monitor
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import threading
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, quote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Running ``python dashboard/app.py`` puts dashboard/ rather than the project
# root on sys.path.  Add the project root only for normal project imports.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# The minimum columns required to provide the dashboard's primary V1 true-state
# view.  Other Simulator columns are optional in the dashboard so older/future
# result files can still be inspected without inventing missing values.
REQUIRED_RESULT_COLUMNS: Tuple[str, ...] = (
    "time_s",
    "delta_w_mps",
    "delta_q_rads",
)

# Actual numeric fields emitted by Simulator._record_step in the inspected
# project.  Values are parsed as floats when present.  ``total_thrust_N`` may
# legitimately be NaN when propulsion status is MODEL_NOT_IDENTIFIED.
KNOWN_NUMERIC_COLUMNS: Tuple[str, ...] = (
    "time_s",
    "delta_w_mps",
    "delta_q_rads",
    "delta_e_rad",
    "delta_w_dot",
    "delta_q_dot",
    "perturbation_az_mps2",
    "delta_alpha_rad",
    "delta_CL",
    "delta_Cm",
    "delta_Lift_N",
    "delta_PitchMom_Nm",
    "total_thrust_N",
)

# Verified Simulator output contract: ``total_thrust_N`` may legitimately be
# NaN when the propulsion model status is MODEL_NOT_IDENTIFIED.  No other
# numeric Simulator CSV column is documented with a legitimate NaN, so NaN is
# rejected everywhere else instead of being silently accepted.
NAN_PERMITTED_NUMERIC_COLUMNS: FrozenSet[str] = frozenset({"total_thrust_N"})

FIELD_UNITS: Mapping[str, str] = MappingProxyType({
    "time_s": "s",
    "delta_w_mps": "m/s",
    "delta_q_rads": "rad/s",
    "delta_e_rad": "rad",
    "delta_w_dot": "m/s²",
    "delta_q_dot": "rad/s²",
    "perturbation_az_mps2": "m/s²",
    "delta_alpha_rad": "rad",
    "delta_CL": "dimensionless",
    "delta_Cm": "dimensionless",
    "delta_Lift_N": "N",
    "delta_PitchMom_Nm": "N·m",
    "total_thrust_N": "N",
    "delta_w_hat": "m/s",
    "delta_q_hat": "rad/s",
})


class DashboardError(Exception):
    """Base class for concise user-facing dashboard failures."""


class ResultNotFoundError(DashboardError):
    """Raised when a requested saved result does not exist."""


class EmptyResultError(DashboardError):
    """Raised when a saved result contains no data samples."""


class ResultSchemaError(DashboardError):
    """Raised when a saved result is missing required columns."""


class MalformedResultError(DashboardError):
    """Raised when a required/numeric result value cannot be interpreted."""


class ScenarioRegistryError(DashboardError):
    """Raised when the real project scenario registry cannot be loaded."""


@dataclass(frozen=True)
class RunData:
    """Read-only in-memory representation of one saved Simulator CSV result."""

    path: Path
    headers: Tuple[str, ...]
    rows: Tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.stem


@dataclass(frozen=True)
class ScenarioView:
    """Presentation-only projection of an existing SimulationScenario."""

    name: str
    description: str
    classification: str
    reference_condition_provenance: str
    control_schedule_provenance: str
    source: str
    flight_condition: Optional[str]
    runnable: bool
    assumptions: Tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


@dataclass(frozen=True)
class DashboardBundle:
    """Display-ready data with no physics/estimation logic."""

    run: RunData
    scenario: Optional[ScenarioView]
    summary: Mapping[str, Any]
    true_state: Mapping[str, Any]
    sensor_data: Mapping[str, Any]
    estimate_data: Mapping[str, Any]
    residual_data: Mapping[str, Any]
    health_data: Mapping[str, Any]
    provenance_data: Mapping[str, Any]


@dataclass
class _CacheEntry:
    mtime_ns: int
    size: int
    data: RunData


class RunRepository:
    """Read-only, mtime-aware cache for saved CSV results.

    The cache avoids reparsing an unchanged CSV on every browser redraw while
    never modifying the file.  It contains no simulator, estimator, or health
    state.
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir).resolve()
        self._cache: Dict[Path, _CacheEntry] = {}
        self._lock = threading.RLock()

    def discover(self) -> Tuple[Path, ...]:
        if not self.data_dir.is_dir():
            return tuple()
        return tuple(sorted(
            (p for p in self.data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"),
            key=lambda p: p.name.lower(),
        ))

    def load_by_name(self, run_name: str) -> RunData:
        if not isinstance(run_name, str) or not run_name.strip():
            raise ResultNotFoundError("No run was selected.")
        candidate = (self.data_dir / f"{run_name}.csv").resolve()
        try:
            candidate.relative_to(self.data_dir)
        except ValueError as exc:
            raise ResultNotFoundError("Invalid run selection.") from exc
        return self.load(candidate)

    def load(self, path: Path) -> RunData:
        path = Path(path).resolve()
        try:
            path.relative_to(self.data_dir)
        except ValueError as exc:
            raise ResultNotFoundError("Result path is outside the configured data directory.") from exc

        if not path.is_file():
            raise ResultNotFoundError(f"Result file not found: {path.name}")

        stat = path.stat()
        with self._lock:
            cached = self._cache.get(path)
            if cached and cached.mtime_ns == stat.st_mtime_ns and cached.size == stat.st_size:
                return cached.data
            data = load_run(path)
            self._cache[path] = _CacheEntry(stat.st_mtime_ns, stat.st_size, data)
            return data


def _enum_value(value: Any) -> Any:
    """Return an Enum's exact value; leave every other object unchanged."""

    return value.value if isinstance(value, Enum) else value


def _safe_text(value: Any) -> str:
    if value is None:
        return "Unavailable"
    value = _enum_value(value)
    return str(value)


def _finite_float(column: str, raw: str, row_number: int, *, allow_nan: bool = False) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise MalformedResultError(
            f"Row {row_number}: column '{column}' is not numeric: {raw!r}."
        ) from exc
    if math.isnan(value) and allow_nan:
        return value
    if not math.isfinite(value):
        raise MalformedResultError(
            f"Row {row_number}: column '{column}' must be finite; got {raw!r}."
        )
    return value


def validate_result_schema(headers: Optional[Iterable[str]]) -> Tuple[str, ...]:
    """Validate the minimum real Simulator CSV schema needed for V1.

    No missing column is fabricated.  Optional columns remain optional and the
    corresponding visualisation is hidden when they are absent.
    """

    if headers is None:
        raise ResultSchemaError("CSV has no header row.")
    normalized = tuple(str(h).strip() for h in headers if h is not None)
    if not normalized:
        raise ResultSchemaError("CSV has an empty header row.")
    missing = [c for c in REQUIRED_RESULT_COLUMNS if c not in normalized]
    if missing:
        raise ResultSchemaError(
            "CSV is missing required column(s): " + ", ".join(missing)
        )
    return normalized


def load_run(path: Path) -> RunData:
    """Load and validate one existing Simulator CSV result.

    The source file is opened read-only.  Required true-state/time values must
    be finite.  Known optional numeric fields are validated if present;
    ``total_thrust_N`` alone permits NaN because the inspected Simulator
    explicitly documents that value for an unidentified propulsion model.
    """

    path = Path(path)
    if not path.is_file():
        raise ResultNotFoundError(f"Result file not found: {path}")

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = validate_result_schema(reader.fieldnames)
            parsed_rows: List[Mapping[str, Any]] = []
            for row_number, row in enumerate(reader, start=2):
                if row is None or not any((v or "").strip() for v in row.values() if v is not None):
                    continue
                parsed: Dict[str, Any] = {}
                for column in headers:
                    raw = row.get(column)
                    if raw is None:
                        parsed[column] = None
                        continue
                    raw = raw.strip()
                    if column in KNOWN_NUMERIC_COLUMNS:
                        if raw == "":
                            # Required numeric fields cannot be empty. Optional
                            # numeric data remains unavailable, not invented.
                            if column in REQUIRED_RESULT_COLUMNS:
                                raise MalformedResultError(
                                    f"Row {row_number}: required column '{column}' is empty."
                                )
                            parsed[column] = None
                        else:
                            parsed[column] = _finite_float(
                                column,
                                raw,
                                row_number,
                                allow_nan=(column in NAN_PERMITTED_NUMERIC_COLUMNS),
                            )
                    else:
                        parsed[column] = raw
                parsed_rows.append(MappingProxyType(parsed))
    except UnicodeDecodeError as exc:
        raise MalformedResultError(f"Result file is not valid UTF-8: {path.name}") from exc
    except OSError as exc:
        raise DashboardError(f"Unable to read result file '{path.name}': {exc}") from exc

    if not parsed_rows:
        raise EmptyResultError(f"Result file '{path.name}' contains no data samples.")

    # The dashboard uses the source timestamps exactly and preserves row order.
    # Reject reversal/duplication rather than constructing a replacement clock.
    previous: Optional[float] = None
    for index, row in enumerate(parsed_rows, start=1):
        current = row["time_s"]
        if previous is not None and current <= previous:
            raise MalformedResultError(
                f"Sample {index}: time_s={current} is not strictly greater than the previous timestamp {previous}."
            )
        previous = current

    metadata: Dict[str, Any] = {}
    meta_path = path.with_suffix(".meta.json")
    if meta_path.is_file():
        try:
            with meta_path.open("r", encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
        except Exception:
            pass
            
    return RunData(path=path.resolve(), headers=headers, rows=tuple(parsed_rows), metadata=MappingProxyType(metadata))


def _scenario_to_view(scenario: Any) -> ScenarioView:
    """Project one real SimulationScenario into immutable display metadata."""

    flight_condition = getattr(scenario, "flight_condition", None)
    if flight_condition is not None:
        flight_condition_text = str(_enum_value(flight_condition))
    else:
        flight_condition_text = None

    classification_attr = getattr(scenario, "classification")
    classification = classification_attr() if callable(classification_attr) else classification_attr

    return ScenarioView(
        name=str(getattr(scenario, "name")),
        description=str(getattr(scenario, "description", "") or ""),
        classification=str(classification),
        reference_condition_provenance=str(_enum_value(getattr(scenario, "reference_condition_provenance"))),
        control_schedule_provenance=str(_enum_value(getattr(scenario, "control_schedule_provenance"))),
        source=str(getattr(scenario, "source")),
        flight_condition=flight_condition_text,
        runnable=flight_condition is not None,
        assumptions=tuple(str(v) for v in getattr(scenario, "assumptions", tuple())),
        notes=str(getattr(scenario, "notes", "")),
    )


def load_scenarios(registry_func: Optional[Any] = None) -> Tuple[ScenarioView, ...]:
    """Load scenario metadata from the project's existing registry.

    ``registry_func`` exists only for isolated testing/injection.  Production
    code imports ``simulation.scenarios.get_all_scenarios``; no scenario is
    redefined in this module.
    """

    if registry_func is None:
        try:
            from simulation.scenarios import get_all_scenarios  # type: ignore
        except Exception as exc:
            raise ScenarioRegistryError(
                f"Unable to import simulation.scenarios.get_all_scenarios: {exc}"
            ) from exc
        registry_func = get_all_scenarios

    try:
        scenarios = tuple(registry_func())
    except Exception as exc:
        raise ScenarioRegistryError(f"Unable to load scenario registry: {exc}") from exc
    return tuple(_scenario_to_view(s) for s in scenarios)


def find_scenario_view(name: Optional[str], scenarios: Sequence[ScenarioView]) -> Optional[ScenarioView]:
    if not name:
        return None
    for scenario in scenarios:
        if scenario.name == name:
            return scenario
    return None


def prepare_state_data(run: RunData) -> Mapping[str, Any]:
    """Prepare source timestamps and TRUE / SIMULATED reduced-order state."""

    return MappingProxyType({
        "time_s": tuple(row["time_s"] for row in run.rows),
        "delta_w_mps": tuple(row["delta_w_mps"] for row in run.rows),
        "delta_q_rads": tuple(row["delta_q_rads"] for row in run.rows),
        "labels": MappingProxyType({
            "delta_w_mps": "True / Simulated delta_w",
            "delta_q_rads": "True / Simulated delta_q",
        }),
        "units": MappingProxyType({
            "time_s": FIELD_UNITS["time_s"],
            "delta_w_mps": FIELD_UNITS["delta_w_mps"],
            "delta_q_rads": FIELD_UNITS["delta_q_rads"],
        }),
    })


def _get_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def _mapping_copy(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping, got {type(value).__name__}")
    return MappingProxyType(dict(value))


def prepare_sensor_data(integrated_results: Optional[Sequence[Any]]) -> Mapping[str, Any]:
    """Prepare genuine SensorModel output from IntegratedStepResult objects.

    No sensor channel is created here.  Only keys actually present in every
    sample's existing ``sensor_result['measured']`` mapping are exposed.
    """

    if not integrated_results:
        return MappingProxyType({"available": False, "reason": "No integration-layer sensor results were supplied."})

    measured_rows: List[Mapping[str, Any]] = []
    statuses: List[str] = []
    times: List[float] = []
    for item in integrated_results:
        times.append(float(_get_attr(item, "timestamp_s")))
        sensor_result = _get_attr(item, "sensor_result")
        if not isinstance(sensor_result, Mapping) or "measured" not in sensor_result:
            raise MalformedResultError("Integrated result is missing sensor_result['measured'].")
        measured = sensor_result["measured"]
        if not isinstance(measured, Mapping):
            raise MalformedResultError("sensor_result['measured'] is not a mapping.")
        measured_rows.append(measured)
        if "sensor_model_status" in sensor_result:
            statuses.append(str(sensor_result["sensor_model_status"]))

    common_keys = set(measured_rows[0].keys())
    for row in measured_rows[1:]:
        common_keys.intersection_update(row.keys())

    series: Dict[str, Tuple[float, ...]] = {}
    for key in sorted(common_keys):
        values: List[float] = []
        usable = True
        for row in measured_rows:
            try:
                val = float(row[key])
            except (TypeError, ValueError):
                usable = False
                break
            if not math.isfinite(val):
                usable = False
                break
            values.append(val)
        if usable:
            series[key] = tuple(values)

    return MappingProxyType({
        "available": bool(series),
        "time_s": tuple(times),
        "series": MappingProxyType(series),
        "sensor_model_status": tuple(dict.fromkeys(statuses)),
    })


def prepare_estimate_data(integrated_results: Optional[Sequence[Any]]) -> Mapping[str, Any]:
    """Prepare genuine EstimatedState outputs without recomputing estimates."""

    if not integrated_results:
        return MappingProxyType({"available": False, "reason": "No integration-layer estimator results were supplied."})

    times: List[float] = []
    w_hat: List[float] = []
    q_hat: List[float] = []
    statuses: List[str] = []
    provenance: Any = None
    for item in integrated_results:
        times.append(float(_get_attr(item, "timestamp_s")))
        state = _get_attr(item, "estimated_state")
        w = float(_get_attr(state, "delta_w_hat"))
        q = float(_get_attr(state, "delta_q_hat"))
        if not (math.isfinite(w) and math.isfinite(q)):
            raise MalformedResultError("EstimatedState contains a non-finite reduced-order state.")
        w_hat.append(w)
        q_hat.append(q)
        statuses.append(str(_get_attr(state, "estimator_status")))
        if provenance is None:
            provenance = _get_attr(state, "provenance")

    return MappingProxyType({
        "available": True,
        "time_s": tuple(times),
        "delta_w_hat": tuple(w_hat),
        "delta_q_hat": tuple(q_hat),
        "estimator_status": tuple(dict.fromkeys(statuses)),
        "provenance": provenance,
        "units": MappingProxyType({
            "delta_w_hat": FIELD_UNITS["delta_w_hat"],
            "delta_q_hat": FIELD_UNITS["delta_q_hat"],
        }),
    })


def prepare_residual_data(integrated_results: Optional[Sequence[Any]]) -> Mapping[str, Any]:
    """Expose residual diagnostics already produced by HealthMonitor.

    The HealthMonitor may derive a scalar residual from measurement - estimate.
    The dashboard does not reproduce that operation; it reads the monitor's
    existing diagnostics and displays its reported absolute ``observed`` value.
    """

    if not integrated_results:
        return MappingProxyType({"available": False, "reason": "No health diagnostics were supplied."})

    rule_map = {
        "delta_w": "SENSOR_RESIDUAL_CHECK.delta_w",
        "delta_q": "SENSOR_RESIDUAL_CHECK.delta_q",
    }
    data: Dict[str, List[Optional[float]]] = {key: [] for key in rule_map}
    sources: Dict[str, List[str]] = {key: [] for key in rule_map}
    units: Dict[str, str] = {}
    times: List[float] = []

    for item in integrated_results:
        times.append(float(_get_attr(item, "timestamp_s")))
        health = _get_attr(item, "health_result")
        diagnostics = _get_attr(health, "diagnostics")
        for key, rule in rule_map.items():
            diag = diagnostics.get(rule) if isinstance(diagnostics, Mapping) else None
            if not isinstance(diag, Mapping) or diag.get("observed") is None:
                data[key].append(None)
                continue
            observed = float(diag["observed"])
            data[key].append(observed if math.isfinite(observed) else None)
            if diag.get("residual_source") is not None:
                sources[key].append(str(diag["residual_source"]))
            if diag.get("unit") is not None:
                units[key] = str(diag["unit"])

    available = any(any(v is not None for v in values) for values in data.values())
    return MappingProxyType({
        "available": available,
        "time_s": tuple(times),
        "series": MappingProxyType({k: tuple(v) for k, v in data.items()}),
        "sources": MappingProxyType({k: tuple(dict.fromkeys(v)) for k, v in sources.items()}),
        "units": MappingProxyType(units),
    })


def _fault_flag_to_dict(flag: Any) -> Mapping[str, Any]:
    return MappingProxyType({
        "rule": str(_get_attr(flag, "rule")),
        "severity": str(_enum_value(_get_attr(flag, "severity"))),
        "provenance": str(_enum_value(_get_attr(flag, "provenance"))),
        "category": str(_enum_value(_get_attr(flag, "category"))),
        "observed": _get_attr(flag, "observed"),
        "limit": _get_attr(flag, "limit"),
        "unit": str(_get_attr(flag, "unit")),
        "message": str(_get_attr(flag, "message")),
    })


def prepare_health_data(integrated_results: Optional[Sequence[Any]]) -> Mapping[str, Any]:
    """Prepare HealthMonitorResult data exactly as emitted by the health layer."""

    if not integrated_results:
        return MappingProxyType({"available": False, "reason": "No integration-layer health results were supplied."})

    timeline: List[Mapping[str, Any]] = []
    for item in integrated_results:
        timestamp = float(_get_attr(item, "timestamp_s"))
        result = _get_attr(item, "health_result")
        flags = tuple(_fault_flag_to_dict(f) for f in _get_attr(result, "fault_flags"))
        diagnostics = _get_attr(result, "diagnostics")
        timeline.append(MappingProxyType({
            "time_s": timestamp,
            "overall_status": str(_enum_value(_get_attr(result, "overall_status"))),
            "severity": str(_enum_value(_get_attr(result, "severity"))),
            "coverage": str(_enum_value(_get_attr(result, "coverage"))),
            "skipped_rules": tuple(str(v) for v in _get_attr(result, "skipped_rules")),
            "fault_flags": flags,
            "diagnostics": _mapping_copy(diagnostics),
            "provenance": str(_get_attr(result, "provenance")),
        }))

    last = timeline[-1]
    return MappingProxyType({
        "available": True,
        "overall_status": last["overall_status"],
        "severity": last["severity"],
        "coverage": last["coverage"],
        "skipped_rules": last["skipped_rules"],
        "fault_flags": last["fault_flags"],
        "diagnostics": last["diagnostics"],
        "provenance": last["provenance"],
        "timeline": tuple(timeline),
    })


def _object_to_plain_dict(obj: Any) -> Mapping[str, Any]:
    if obj is None:
        return MappingProxyType({})
    if isinstance(obj, Mapping):
        return MappingProxyType({str(k): _enum_value(v) for k, v in obj.items()})
    if is_dataclass(obj):
        return MappingProxyType({f.name: _enum_value(getattr(obj, f.name)) for f in fields(obj)})
    if hasattr(obj, "__dict__"):
        return MappingProxyType({str(k): _enum_value(v) for k, v in vars(obj).items() if not str(k).startswith("_")})
    return MappingProxyType({"value": _enum_value(obj)})


def prepare_provenance_data(
    scenario: Optional[ScenarioView],
    integrated_results: Optional[Sequence[Any]],
) -> Mapping[str, Any]:
    """Prepare section-level provenance without creating a global label."""

    result: Dict[str, Any] = {}
    if scenario is not None:
        result["scenario"] = MappingProxyType({
            "reference_condition": scenario.reference_condition_provenance,
            "control_schedule": scenario.control_schedule_provenance,
            "classification": scenario.classification,
            "source": scenario.source,
        })

    if integrated_results:
        first = integrated_results[0]
        sensor_result = _get_attr(first, "sensor_result")
        if isinstance(sensor_result, Mapping) and "sensor_model_status" in sensor_result:
            result["sensor"] = MappingProxyType({"model_status": sensor_result["sensor_model_status"]})
        state = _get_attr(first, "estimated_state")
        result["estimator"] = _object_to_plain_dict(_get_attr(state, "provenance"))
        health = _get_attr(first, "health_result")
        result["health"] = MappingProxyType({"method": str(_get_attr(health, "provenance"))})
        if hasattr(first, "pipeline_provenance") or (isinstance(first, Mapping) and "pipeline_provenance" in first):
            result["pipeline"] = MappingProxyType({"provenance": str(_get_attr(first, "pipeline_provenance"))})

    return MappingProxyType(result)


def prepare_run_summary(
    run: RunData,
    integrated_results: Optional[Sequence[Any]] = None,
) -> Mapping[str, Any]:
    """Summarize only values genuinely present in the selected data source."""

    final_row = run.rows[-1]
    summary: Dict[str, Any] = {
        "final_time_s": final_row["time_s"],
        "samples": len(run.rows),
    }
    if "propulsion_model_status" in run.headers:
        status = final_row.get("propulsion_model_status")
        if status not in (None, ""):
            summary["propulsion_status"] = status
    if "simulation_status" in run.headers:
        status = final_row.get("simulation_status")
        if status not in (None, ""):
            summary["simulation_status"] = status

    if integrated_results:
        last = integrated_results[-1]
        health = _get_attr(last, "health_result")
        summary["overall_health_status"] = str(_enum_value(_get_attr(health, "overall_status")))
        summary["health_severity"] = str(_enum_value(_get_attr(health, "severity")))
        summary["health_coverage"] = str(_enum_value(_get_attr(health, "coverage")))
        propulsion = _get_attr(last, "propulsion_status")
        if propulsion is not None:
            summary["propulsion_status"] = propulsion

    return MappingProxyType(summary)


def _validate_integrated_alignment(run: RunData, integrated_results: Sequence[Any]) -> None:
    """Require exact timestamp alignment between a run and integration output.

    ``IntegratedStepResult`` sequences are produced from the same Simulator
    history as the saved CSV, so their ``timestamp_s`` values must equal the
    run timestamps exactly.  A matching sample count alone is insufficient;
    mismatched clocks are rejected instead of repaired with replacement
    timestamps.
    """

    for index, item in enumerate(integrated_results, start=1):
        ts = float(_get_attr(item, "timestamp_s"))
        run_ts = run.rows[index - 1]["time_s"]
        if ts != run_ts:
            raise MalformedResultError(
                f"Integrated result sample {index}: timestamp {ts} does not match "
                f"the selected run timestamp {run_ts}. Replacement timestamps are "
                "not constructed."
            )


def prepare_dashboard_data(
    run: RunData,
    scenario: Optional[ScenarioView] = None,
    integrated_results: Optional[Sequence[Any]] = None,
) -> DashboardBundle:
    """Prepare all display sections without mutating source data."""

    if integrated_results is not None and len(integrated_results) != len(run.rows):
        raise MalformedResultError(
            "Integrated result sample count does not match the selected Simulator run."
        )
    if integrated_results is not None:
        _validate_integrated_alignment(run, integrated_results)

    return DashboardBundle(
        run=run,
        scenario=scenario,
        summary=prepare_run_summary(run, integrated_results),
        true_state=prepare_state_data(run),
        sensor_data=prepare_sensor_data(integrated_results),
        estimate_data=prepare_estimate_data(integrated_results),
        residual_data=prepare_residual_data(integrated_results),
        health_data=prepare_health_data(integrated_results),
        provenance_data=prepare_provenance_data(scenario, integrated_results),
    )


# ---------------------------------------------------------------------------
# Rendering helpers — presentation only
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.6g}"
    return html.escape(str(_enum_value(value)))


def _badge(value: Any) -> str:
    text = _safe_text(value)
    css = "badge"
    if text in {"HEALTHY", "FULL"}:
        css += " good"
    elif text in {"WARNING", "PARTIAL", "MODEL_NOT_IDENTIFIED"}:
        css += " warn"
    elif text in {"FAULT", "CRITICAL", "INVALID"}:
        css += " bad"
    return f'<span class="{css}">{html.escape(text)}</span>'


def _card(label: str, value: Any) -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{_fmt(value)}</div>'
        '</div>'
    )


# Centralized UI palette so every SVG chart stays visually consistent.
CHART_PALETTE: Tuple[str, ...] = ("#2f6feb", "#c850c0", "#2da44e", "#bf8700")


def _polyline_points(x: Sequence[float], y: Sequence[Optional[float]], width: int, height: int) -> str:
    finite_pairs = [(float(a), float(b)) for a, b in zip(x, y) if b is not None and math.isfinite(float(a)) and math.isfinite(float(b))]
    if not finite_pairs:
        return ""
    xmin, xmax = min(a for a, _ in finite_pairs), max(a for a, _ in finite_pairs)
    ymin, ymax = min(b for _, b in finite_pairs), max(b for _, b in finite_pairs)
    if math.isclose(xmin, xmax):
        xmax = xmin + 1.0
    if math.isclose(ymin, ymax):
        pad = max(abs(ymin) * 0.05, 1e-9)
        ymin -= pad
        ymax += pad
    left, right, top, bottom = 58.0, 18.0, 18.0, 42.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    pts = []
    for a, b in finite_pairs:
        px = left + (a - xmin) / (xmax - xmin) * plot_w
        py = top + (ymax - b) / (ymax - ymin) * plot_h
        pts.append(f"{px:.2f},{py:.2f}")
    return " ".join(pts)


def _render_chart(
    title: str,
    x: Sequence[float],
    series: Sequence[Tuple[str, Sequence[Optional[float]]]],
    y_unit: str,
) -> str:
    """Render exact samples as SVG polylines; no smoothing/resampling."""

    width, height = 760, 280
    palette = CHART_PALETTE
    paths: List[str] = []
    legend: List[str] = []
    for index, (label, values) in enumerate(series):
        points = _polyline_points(x, values, width, height)
        if not points:
            continue
        color = palette[index % len(palette)]
        paths.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}" />'
        )
        legend.append(
            f'<span class="legend-item"><span class="legend-line" style="background:{color}"></span>{html.escape(label)}</span>'
        )
    if not paths:
        return '<div class="callout">No finite values are available for this plot.</div>'

    return f"""
    <div class="chart-card">
      <div class="chart-title">{html.escape(title)}</div>
      <svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        <line x1="58" y1="238" x2="742" y2="238" class="axis" />
        <line x1="58" y1="18" x2="58" y2="238" class="axis" />
        {''.join(paths)}
        <text x="400" y="270" text-anchor="middle" class="axis-label">time [s] — source timestamps</text>
        <text x="17" y="135" transform="rotate(-90 17 135)" text-anchor="middle" class="axis-label">{html.escape(y_unit)}</text>
      </svg>
      <div class="legend">{''.join(legend)}</div>
    </div>
    """


def _render_kv_table(title: str, mapping: Mapping[str, Any]) -> str:
    rows = []
    for key, value in mapping.items():
        if isinstance(value, (tuple, list)):
            display = ", ".join(_safe_text(v) for v in value) if value else "None"
        else:
            display = _safe_text(value)
        rows.append(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(display)}</td></tr>"
        )
    return f'<div class="table-card"><h3>{html.escape(title)}</h3><table>{"".join(rows)}</table></div>'


def _render_scenario(scenario: Optional[ScenarioView], *, user_supplied: bool = False) -> str:
    if scenario is None:
        return '<div class="callout">Scenario is not encoded in the saved Simulator CSV. Select a scenario only if you know the run association; the dashboard does not infer one from the filename.</div>'
    runnable_note = "Runnable by current Simulator" if scenario.runnable else "REFERENCE / DESIGN-ONLY — no derivative-backed FlightCondition; not runnable by the current Simulator"
    assumptions = "".join(f"<li>{html.escape(v)}</li>" for v in scenario.assumptions)
    association_note = ""
    if user_supplied:
        association_note = (
            '<div class="callout warn-callout"><strong>User-supplied metadata association:</strong> '
            'the selected scenario is displayed as metadata only. The dashboard cannot verify '
            'that this scenario produced the selected run, and scenario identity is never '
            'inferred from filenames.</div>'
        )
    return f"""
    {association_note}
    <div class="scenario-grid">
      {_card('Scenario', scenario.name)}
      {_card('Classification', scenario.classification)}
      {_card('Flight condition', scenario.flight_condition or 'None')}
      {_card('Execution capability', runnable_note)}
    </div>
    <div class="table-card">
      <h3>Scenario source and provenance</h3>
      <table>
        <tr><th>Reference condition provenance</th><td>{html.escape(scenario.reference_condition_provenance)}</td></tr>
        <tr><th>Control schedule provenance</th><td>{html.escape(scenario.control_schedule_provenance)}</td></tr>
        <tr><th>Source</th><td>{html.escape(scenario.source)}</td></tr>
        <tr><th>Description</th><td>{html.escape(scenario.description)}</td></tr>
        <tr><th>Notes</th><td>{html.escape(scenario.notes or 'None')}</td></tr>
      </table>
      {f'<h4>Assumptions</h4><ul>{assumptions}</ul>' if assumptions else ''}
    </div>
    """


def _render_health(health_data: Mapping[str, Any]) -> str:
    if not health_data.get("available"):
        return '<div class="callout">HealthMonitor output is unavailable in this saved Simulator CSV. No health status has been fabricated.</div>'

    flags = health_data.get("fault_flags", tuple())
    flag_rows = []
    for flag in flags:
        flag_rows.append(
            "<tr>"
            f"<td>{html.escape(str(flag['rule']))}</td>"
            f"<td>{_badge(flag['severity'])}</td>"
            f"<td>{html.escape(str(flag['category']))}</td>"
            f"<td>{_fmt(flag['observed'])}</td>"
            f"<td>{_fmt(flag['limit'])}</td>"
            f"<td>{html.escape(str(flag['unit']))}</td>"
            f"<td>{html.escape(str(flag['provenance']))}</td>"
            f"<td>{html.escape(str(flag['message']))}</td>"
            "</tr>"
        )
    flag_table = (
        '<div class="table-card"><h3>Fault flags</h3><table><thead><tr>'
        '<th>Rule</th><th>Severity</th><th>Category</th><th>Observed</th><th>Limit</th><th>Unit</th><th>Provenance</th><th>Message</th>'
        f'</tr></thead><tbody>{"".join(flag_rows)}</tbody></table></div>'
        if flag_rows else '<div class="callout good-callout">No fault flags are present in the latest HealthMonitorResult.</div>'
    )

    skipped = health_data.get("skipped_rules", tuple())
    skipped_text = ", ".join(str(v) for v in skipped) if skipped else "None"
    return f"""
    <div class="metrics">
      {_card('Overall health status', health_data.get('overall_status'))}
      {_card('Severity', health_data.get('severity'))}
      {_card('Coverage', health_data.get('coverage'))}
      {_card('Health methodology', health_data.get('provenance'))}
    </div>
    <div class="callout"><strong>Skipped rules:</strong> {html.escape(skipped_text)}</div>
    {flag_table}
    """


def _render_provenance(provenance_data: Mapping[str, Any]) -> str:
    if not provenance_data:
        return '<div class="callout">No section-level provenance beyond the saved Simulator columns is available for this view.</div>'
    blocks = []
    for section, values in provenance_data.items():
        if isinstance(values, Mapping):
            blocks.append(_render_kv_table(str(section).replace("_", " ").title(), values))
    return "".join(blocks)


def _render_telemetry_cards(bundle, summary):
    """Build aerospace-style telemetry cards from existing pipeline data only."""
    final_row = bundle.run.rows[-1]
    has_delta_e = "delta_e_rad" in bundle.run.headers and final_row.get("delta_e_rad") is not None
    cards = []

    def _tcard(tag_class, tag_text, label, value):
        return (
            f'<div class="telem-card"><div class="telem-tag {tag_class}">{tag_text}</div>'
            f'<div class="telem-label">{html.escape(label)}</div>'
            f'<div class="telem-value">{value}</div></div>'
        )

    # SIMULATED values from stored CSV
    cards.append(_tcard("sim-tag", "SIMULATED", "Simulation Time",
                        f"{_fmt(summary.get('final_time_s'))} s"))
    cards.append(_tcard("sim-tag", "SIMULATED", "delta_w (final)",
                        f"{_fmt(final_row.get('delta_w_mps'))} m/s"))
    cards.append(_tcard("sim-tag", "SIMULATED", "delta_q (final)",
                        f"{_fmt(final_row.get('delta_q_rads'))} rad/s"))
    if has_delta_e:
        cards.append(_tcard("sim-tag", "SIMULATED", "Elevator Input (final)",
                            f"{_fmt(final_row.get('delta_e_rad'))} rad"))
    else:
        cards.append(_tcard("na-tag", "N/A", "Elevator Input",
                            "NOT AVAILABLE IN SAVED RESULT"))
    cards.append(_tcard("sim-tag", "SIMULATED", "Samples", str(len(bundle.run.rows))))
    prop = summary.get("propulsion_status", "NOT AVAILABLE IN SAVED RESULT")
    cards.append(_tcard("sim-tag", "SIMULATED", "Propulsion Status",
                        html.escape(str(prop))))

    # REFERENCE values from scenario metadata
    if bundle.scenario:
        fc = bundle.scenario.flight_condition or "None"
        cards.append(_tcard("ref-tag", "REFERENCE", "Flight Condition",
                            html.escape(fc)))
        cards.append(_tcard("ref-tag", "REFERENCE", "Classification",
                            html.escape(bundle.scenario.classification)))
        cards.append(_tcard("ref-tag", "REFERENCE", "Ref. Condition Provenance",
                            html.escape(bundle.scenario.reference_condition_provenance)))
        cards.append(_tcard("ref-tag", "REFERENCE", "Control Schedule Provenance",
                            html.escape(bundle.scenario.control_schedule_provenance)))
    else:
        cards.append(_tcard("na-tag", "N/A", "Reference Condition",
                            "NOT AVAILABLE IN SAVED RESULT"))

    return "".join(cards)


def _render_time_selector_payload(bundle):
    """Serialize stored samples (and a derived pitch-rate scale) for the
    client-side time selector and the qualitative pitch-rate visualization.

    ``q_scale`` is the largest magnitude of the stored ``delta_q_rads``
    column for this run; it is used only to bound the client-side rotation
    of the aircraft graphic (see js_block in render_dashboard_page) and is
    never used to compute or integrate a pitch angle.
    """
    has_de = "delta_e_rad" in bundle.run.headers
    entries = []
    q_values = []
    for row in bundle.run.rows:
        e = {"time_s": row["time_s"],
             "delta_w_mps": row["delta_w_mps"],
             "delta_q_rads": row["delta_q_rads"]}
        if has_de:
            e["delta_e_rad"] = row.get("delta_e_rad")
        entries.append(e)
        q_values.append(abs(row["delta_q_rads"]))
    q_scale = max(q_values) if q_values else 0.0
    return json.dumps({"samples": entries, "q_scale": q_scale})


# Aircraft schematic, redrawn as a recognizable top-down airliner silhouette
# (tapered fuselage, swept wings with engine pods, swept horizontal
# stabilizers, and a small vertical-fin accent at the tail) rather than the
# abstract bowtie shapes used previously. viewBox is 0 0 640 260 so the
# aircraft can be the dashboard's main visual rather than a decorative
# header thumbnail. Nose points toward the left edge of the viewBox.
#
# The rotating <g id="q-indicator"> group is the qualitative pitch-rate
# visualization: its rotation is set client-side, directly from the
# selected stored ``delta_q_rads`` sample (see the js_block in
# render_dashboard_page). This is a bounded, non-physical indicator of the
# stored pitch *rate* only -- it does not compute or integrate a pitch
# angle and must never be labeled aircraft attitude.
_AIRCRAFT_HERO_SVG = (
    '<svg class="aircraft-hero-svg" viewBox="0 0 640 260" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" role="img" '
    'aria-label="T-2 aircraft schematic (simplified, not geometrically exact); '
    'rotation reflects the qualitative pitch-rate visualization, not aircraft attitude">'
    '<g id="q-indicator" style="transform-origin:320px 130px;">'
    # Horizontal tail stabilizers, drawn first so the fuselage/fin sit on top.
    '<path d="M540 122 L600 102 L614 108 L562 130 Z" '
    'fill="#16233c" stroke="#78a9ff" stroke-width="1.6"/>'
    '<path d="M540 138 L600 158 L614 152 L562 130 Z" '
    'fill="#16233c" stroke="#78a9ff" stroke-width="1.6"/>'
    # Main wings, swept back, drawn under the fuselage so the roots tuck
    # cleanly beneath it instead of meeting in a point at the centerline.
    '<path d="M250 114 L430 35 L462 58 L320 124 Z" '
    'fill="#1a2845" stroke="#78a9ff" stroke-width="2"/>'
    '<path d="M250 146 L430 225 L462 202 L320 136 Z" '
    'fill="#1a2845" stroke="#78a9ff" stroke-width="2"/>'
    # Engine pods, hung forward/under the wing leading edge.
    '<rect x="306" y="64" width="40" height="15" rx="6.5" '
    'fill="#22304d" stroke="#5993e6" stroke-width="1.4"/>'
    '<rect x="306" y="181" width="40" height="15" rx="6.5" '
    'fill="#22304d" stroke="#5993e6" stroke-width="1.4"/>'
    # Fuselage: tapered nose (left) to tapered tail (right), drawn on top
    # so it reads as one continuous body with the wing/tail roots hidden
    # beneath it.
    '<path d="M18 130 Q20 112 100 106 L200 114 Q320 116 430 116 L500 120 '
    'Q580 124 616 130 Q580 136 500 140 L430 144 Q320 144 200 146 '
    'L100 154 Q20 148 18 130 Z" '
    'fill="#1a2845" stroke="#78a9ff" stroke-width="2.2"/>'
    # Vertical fin, shown top-down as a small accent triangle at the tail.
    '<path d="M576 130 L590 100 L604 130 Z" '
    'fill="#2f6feb" stroke="#5993e6" stroke-width="1.2" opacity="0.9"/>'
    # Cockpit / nose highlight.
    '<circle cx="32" cy="130" r="6" fill="#78a9ff" opacity="0.7"/>'
    '</g>'
    '</svg>'
)


def render_dashboard_page(
    bundle: DashboardBundle,
    runs: Sequence[Path],
    scenarios: Sequence[ScenarioView],
    scenario_registry_error: Optional[str] = None,
    scenario_user_supplied: bool = False,
) -> str:
    """Render one complete engineering dashboard page.

    ``scenario_user_supplied`` is True when the scenario was picked through
    the UI (or CLI default) rather than encoded in the saved result; in that
    case the association is labeled as unverified user-supplied metadata.
    """

    run_options = []
    for run_path in runs:
        selected = " selected" if run_path.stem == bundle.run.name else ""
        run_options.append(
            f'<option value="{html.escape(run_path.stem)}"{selected}>{html.escape(run_path.stem)}</option>'
        )

    scenario_options = ['<option value="">Unknown / not associated in CSV</option>']
    selected_name = bundle.scenario.name if bundle.scenario else ""
    for scenario in scenarios:
        selected = " selected" if scenario.name == selected_name else ""
        scenario_options.append(
            f'<option value="{html.escape(scenario.name)}"{selected}>{html.escape(scenario.name)}</option>'
        )

    summary = bundle.summary
    summary_cards = [
        _card("Selected run", bundle.run.name),
        _card("Final time [s]", summary.get("final_time_s")),
        _card("Samples", summary.get("samples")),
    ]
    if "propulsion_status" in summary:
        summary_cards.append(_card("Propulsion status", summary["propulsion_status"]))
    if "simulation_status" in summary:
        summary_cards.append(_card("Simulation status", summary["simulation_status"]))
    if "overall_health_status" in summary:
        summary_cards.extend([
            _card("Overall health", summary["overall_health_status"]),
            _card("Health severity", summary["health_severity"]),
            _card("Health coverage", summary["health_coverage"]),
        ])

    true_state = bundle.true_state
    true_charts = (
        _render_chart(
            "TRUE / SIMULATED \u2014 delta_w vs time",
            true_state["time_s"],
            (("True / Simulated delta_w", true_state["delta_w_mps"]),),
            "delta_w [m/s]",
        )
        + _render_chart(
            "TRUE / SIMULATED \u2014 delta_q vs time",
            true_state["time_s"],
            (("True / Simulated delta_q", true_state["delta_q_rads"]),),
            "delta_q [rad/s]",
        )
    )

    sensor = bundle.sensor_data
    if sensor.get("available"):
        sensor_parts = []
        for key, values in sensor["series"].items():
            sensor_parts.append(_render_chart(
                f"MEASURED / SENSOR \u2014 {key}",
                sensor["time_s"],
                ((f"Measured / Sensor {key}", values),),
                f"{key} [{FIELD_UNITS.get(key, 'source unit')}]",
            ))
        sensor_html = "".join(sensor_parts) + _render_kv_table(
            "Sensor model status", {"sensor_model_status": sensor.get("sensor_model_status", tuple())}
        )
    else:
        sensor_html = '<div class="callout">Sensor measurements are not present in the selected Simulator CSV. No missing sensor channels are created.</div>'

    estimate = bundle.estimate_data
    if estimate.get("available"):
        estimate_html = (
            _render_chart(
                "ESTIMATED / FILTERED \u2014 delta_w_hat vs time",
                estimate["time_s"],
                (("Estimated / Filtered delta_w_hat", estimate["delta_w_hat"]),),
                "delta_w_hat [m/s]",
            )
            + _render_chart(
                "ESTIMATED / FILTERED \u2014 delta_q_hat vs time",
                estimate["time_s"],
                (("Estimated / Filtered delta_q_hat", estimate["delta_q_hat"]),),
                "delta_q_hat [rad/s]",
            )
        )
    else:
        estimate_html = '<div class="callout">EstimatedState output is not present in the selected Simulator CSV. No estimate is fabricated.</div>'

    if sensor.get("available") and estimate.get("available"):
        common_w = sensor["series"].get("delta_w_mps")
        common_q = sensor["series"].get("delta_q_rads")
        compare_parts = []
        if common_w is not None:
            compare_parts.append(_render_chart(
                "MEASURED vs ESTIMATED \u2014 delta_w",
                estimate["time_s"],
                (("Measured / Sensor delta_w", common_w), ("Estimated / Filtered delta_w_hat", estimate["delta_w_hat"])),
                "m/s",
            ))
        if common_q is not None:
            compare_parts.append(_render_chart(
                "MEASURED vs ESTIMATED \u2014 delta_q",
                estimate["time_s"],
                (("Measured / Sensor delta_q", common_q), ("Estimated / Filtered delta_q_hat", estimate["delta_q_hat"])),
                "rad/s",
            ))
        compare_html = "".join(compare_parts) or '<div class="callout">No common measured/estimated channels are available.</div>'
    else:
        compare_html = '<div class="callout">Measured-vs-estimated comparison requires genuine integration-layer sensor and estimator results.</div>'

    residual = bundle.residual_data
    if residual.get("available"):
        residual_parts = []
        for key, values in residual["series"].items():
            if not any(v is not None for v in values):
                continue
            unit = residual["units"].get(key, FIELD_UNITS.get(f"{key}_hat", "source unit"))
            residual_parts.append(_render_chart(
                f"HEALTH-MONITOR RESIDUAL DIAGNOSTIC \u2014 |{key} residual|",
                residual["time_s"],
                ((f"HealthMonitor observed |{key} residual|", values),),
                unit,
            ))
        residual_html = "".join(residual_parts)
    else:
        residual_html = '<div class="callout">Residual diagnostics are unavailable. The dashboard does not recreate Kalman innovations or residual mathematics.</div>'

    registry_message = (
        f'<div class="callout bad-callout"><strong>Scenario registry unavailable:</strong> {html.escape(scenario_registry_error)}</div>'
        if scenario_registry_error else ""
    )

    column_rows = "".join(
        f"<tr><td>{html.escape(column)}</td><td>{html.escape(FIELD_UNITS.get(column, 'status / source-defined'))}</td></tr>"
        for column in bundle.run.headers
    )

    # --- Feature 1: Aerospace header ---
    scenario_display = html.escape(bundle.scenario.name) if bundle.scenario else "Not associated"
    run_display = html.escape(bundle.run.name)

    # --- Feature 2: Telemetry cards ---
    telemetry_html = _render_telemetry_cards(bundle, summary)

    # --- Feature 3: Elevator control input chart ---
    final_row = bundle.run.rows[-1]
    has_delta_e = "delta_e_rad" in bundle.run.headers and final_row.get("delta_e_rad") is not None
    if has_delta_e:
        delta_e_values = tuple(row.get("delta_e_rad") for row in bundle.run.rows)
        elevator_chart = _render_chart(
            "CONTROL INPUT \u2014 delta_e (elevator) vs time [PROJECT_DEFINED schedule]",
            true_state["time_s"],
            (("Stored delta_e (elevator)", delta_e_values),),
            "delta_e [rad]",
        )
    else:
        elevator_chart = '<div class="callout">Elevator input (delta_e_rad) is not present in the selected Simulator CSV.</div>'

    # --- Feature 4: Time selector data (also carries q_scale for the
    # qualitative pitch-rate visualization) ---
    time_data_json = _render_time_selector_payload(bundle)
    max_index = len(bundle.run.rows) - 1

    # Build the page using string concatenation instead of a single f-string
    # to avoid brace-escaping issues.
    css_block = (
        ":root { color-scheme: light dark; --bg:#0b1020; --panel:#121a2b; --panel2:#18233a; "
        "--text:#e7edf7; --muted:#9fb0c8; --line:#2b3a58; --accent:#78a9ff; --good:#38b26b; "
        "--warn:#d5a21a; --bad:#e45d68; }\n"
        "* { box-sizing:border-box; }\n"
        'body { margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; '
        "background:var(--bg); color:var(--text); }\n"
        "header { padding:0; border-bottom:1px solid var(--line); "
        "background:linear-gradient(120deg,#0d1528,#0b1020 40%,#101e36); position:sticky; top:0; z-index:2; }\n"
        ".header-inner { display:flex; align-items:center; gap:24px; padding:16px clamp(18px,4vw,52px); }\n"
        ".header-info { flex:1; min-width:0; }\n"
        ".aircraft-hero-wrap { max-width:640px; margin:0 auto; }\n"
        ".aircraft-hero-svg { width:100%; height:auto; display:block; }\n"
        "#q-indicator { transition: transform 0.12s ease-out; }\n"
        "header h1 { margin:0 0 4px; font-size:clamp(20px,2.5vw,30px); letter-spacing:-.02em; }\n"
        "header h1 .hdr-nasa { color:var(--accent); }\n"
        ".header-meta { display:flex; flex-wrap:wrap; gap:6px 18px; margin-top:6px; }\n"
        ".header-tag { display:inline-flex; align-items:center; gap:5px; font-size:11px; font-weight:700; "
        "text-transform:uppercase; letter-spacing:.06em; color:var(--muted); padding:3px 9px; "
        "border:1px solid var(--line); border-radius:6px; background:rgba(120,169,255,.06); }\n"
        ".header-tag.ro { border-color:rgba(56,178,107,.35); color:var(--good); }\n"
        "main { max-width:1500px; margin:0 auto; padding:24px clamp(14px,3vw,38px) 70px; }\n"
        "section { background:var(--panel); border:1px solid var(--line); border-radius:16px; "
        "margin:18px 0; padding:22px; box-shadow:0 12px 35px rgba(0,0,0,.16); }\n"
        "h2 { margin:0 0 16px; font-size:21px; }\n"
        "h3 { margin:0 0 12px; font-size:16px; }\n"
        "h4 { margin:16px 0 6px; }\n"
        ".controls { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; align-items:end; }\n"
        "label { color:var(--muted); font-size:13px; font-weight:650; display:block; margin-bottom:6px; }\n"
        "select,button { width:100%; padding:11px 12px; border-radius:10px; border:1px solid var(--line); "
        "background:var(--panel2); color:var(--text); font:inherit; }\n"
        "button { background:#234b8d; cursor:pointer; font-weight:700; }\n"
        ".metrics,.scenario-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }\n"
        ".metric { background:var(--panel2); border:1px solid var(--line); border-radius:12px; padding:14px; min-height:88px; }\n"
        ".metric-label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }\n"
        ".metric-value { margin-top:9px; font-size:17px; font-weight:750; overflow-wrap:anywhere; }\n"
        ".callout { border:1px dashed var(--line); background:rgba(120,169,255,.07); color:var(--muted); "
        "padding:13px 15px; border-radius:10px; line-height:1.45; }\n"
        ".good-callout { border-color:rgba(56,178,107,.5); } .bad-callout { border-color:rgba(228,93,104,.6); } "
        ".warn-callout { border-color:rgba(213,162,26,.55); }\n"
        ".chart-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr)); gap:16px; }\n"
        ".chart-card,.table-card { margin-top:14px; background:var(--panel2); border:1px solid var(--line); "
        "border-radius:12px; padding:14px; overflow:auto; }\n"
        ".chart-title { font-weight:700; margin-bottom:8px; } .chart { width:100%; min-width:520px; height:auto; "
        "background:#0f1729; border-radius:8px; }\n"
        ".axis { stroke:#7184a5; stroke-width:1; } .axis-label { fill:#9fb0c8; font-size:12px; }\n"
        ".legend { display:flex; flex-wrap:wrap; gap:14px; margin-top:8px; color:var(--muted); font-size:12px; } "
        ".legend-item { display:flex; align-items:center; gap:6px; } "
        ".legend-line { width:20px; height:3px; border-radius:4px; display:inline-block; }\n"
        "table { width:100%; border-collapse:collapse; min-width:620px; } "
        "th,td { text-align:left; border-bottom:1px solid var(--line); padding:9px 10px; vertical-align:top; } "
        "th { color:var(--muted); font-size:12px; } td { font-size:13px; }\n"
        ".badge { display:inline-block; border:1px solid var(--line); padding:3px 7px; border-radius:999px; "
        "font-size:11px; font-weight:800; } "
        ".badge.good { border-color:var(--good); color:#7fe1a5; } "
        ".badge.warn { border-color:var(--warn); color:#f1c24b; } "
        ".badge.bad { border-color:var(--bad); color:#ff929a; }\n"
        ".small { color:var(--muted); font-size:12px; line-height:1.5; }\n"
        "code { background:#0f1729; padding:2px 5px; border-radius:5px; }\n"
        ".telem-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; }\n"
        ".telem-card { background:var(--panel2); border:1px solid var(--line); border-radius:12px; padding:12px 14px; }\n"
        ".telem-tag { display:inline-block; font-size:9px; font-weight:800; letter-spacing:.08em; "
        "text-transform:uppercase; padding:2px 7px; border-radius:4px; margin-bottom:6px; }\n"
        ".sim-tag { background:rgba(47,111,235,.2); color:#78a9ff; border:1px solid rgba(47,111,235,.4); }\n"
        ".ref-tag { background:rgba(56,178,107,.15); color:#7fe1a5; border:1px solid rgba(56,178,107,.35); }\n"
        ".na-tag { background:rgba(159,176,200,.1); color:var(--muted); border:1px solid var(--line); }\n"
        ".telem-label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }\n"
        ".telem-value { font-size:15px; font-weight:700; margin-top:4px; overflow-wrap:anywhere; }\n"
        ".time-sel { display:grid; grid-template-columns:200px 1fr; gap:16px; align-items:start; }\n"
        ".time-sel-ctrl { display:flex; flex-direction:column; gap:8px; }\n"
        ".time-sel-ctrl input[type=range] { width:100%; accent-color:var(--accent); }\n"
        ".time-sel-readout { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:8px; }\n"
        ".time-sel-val { background:var(--panel2); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }\n"
        ".time-sel-val .ts-label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }\n"
        ".time-sel-val .ts-value { font-size:16px; font-weight:700; font-variant-numeric:tabular-nums; margin-top:3px; }\n"
        "@media (max-width:650px) { header { position:static; } section { padding:15px; } "
        ".time-sel { grid-template-columns:1fr; } }\n"
    )

    js_block = (
        "(function() {\n"
        "  var PAYLOAD = " + time_data_json + ";\n"
        "  var slider = document.getElementById('time-slider');\n"
        "  var indexDisp = document.getElementById('time-index-display');\n"
        "  var tsTime = document.getElementById('ts-time');\n"
        "  var tsDw = document.getElementById('ts-dw');\n"
        "  var tsDq = document.getElementById('ts-dq');\n"
        "  var tsDe = document.getElementById('ts-de');\n"
        "  var qIndicator = document.getElementById('q-indicator');\n"
        "  var qIndicatorRaw = document.getElementById('q-indicator-raw');\n"
        "  function fmt(v) { return v == null ? 'N/A' : (typeof v === 'number' ? v.toPrecision(6) : String(v)); }\n"
        "  function update() {\n"
        "    var i = parseInt(slider.value, 10);\n"
        "    var d = PAYLOAD.samples[i];\n"
        "    indexDisp.textContent = 'Index: ' + i + ' / ' + (PAYLOAD.samples.length - 1);\n"
        "    tsTime.textContent = fmt(d.time_s) + ' s';\n"
        "    tsDw.textContent = fmt(d.delta_w_mps) + ' m/s';\n"
        "    tsDq.textContent = fmt(d.delta_q_rads) + ' rad/s';\n"
        "    tsDe.textContent = d.delta_e_rad != null ? fmt(d.delta_e_rad) + ' rad' : 'NOT AVAILABLE IN SAVED RESULT';\n"
        "    // Qualitative pitch-rate visualization: bounded rotation driven directly by\n"
        "    // the exact stored delta_q_rads sample at the selected index. Saturating\n"
        "    // (tanh) so no finite stored value can break the layout. Not aircraft\n"
        "    // attitude -- no pitch angle is computed or integrated.\n"
        "    var qScale = PAYLOAD.q_scale > 0 ? PAYLOAD.q_scale : 1e-9;\n"
        "    var qNormalized = Math.tanh(d.delta_q_rads / qScale);\n"
        "    var qAngleDeg = qNormalized * 75;\n"
        "    qIndicator.style.transform = 'rotate(' + qAngleDeg.toFixed(3) + 'deg)';\n"
        "    qIndicatorRaw.textContent = fmt(d.delta_q_rads) + ' rad/s';\n"
        "  }\n"
        "  slider.addEventListener('input', update);\n"
        "  update();\n"
        "})();\n"
    )

    scenario_label = (
        "Scenario metadata (user-supplied association)"
        if scenario_user_supplied
        else "Scenario metadata (automatic association)"
    )

    parts = [
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>NASA AirSTAR T-2 Digital Twin Dashboard</title>\n"
        "<style>\n", css_block, "</style>\n</head>\n<body>\n"
        "<header>\n  <div class=\"header-inner\">\n"
        "    <div class=\"header-info\">\n"
        "      <h1><span class=\"hdr-nasa\">NASA AirSTAR</span> T-2 Digital Twin</h1>\n"
        "      <div class=\"header-meta\">\n"
        f"        <span class=\"header-tag\" id=\"hdr-run\">RUN: {run_display}</span>\n"
        f"        <span class=\"header-tag\" id=\"hdr-scenario\">SCENARIO: {scenario_display}</span>\n"
        "        <span class=\"header-tag\">Reduced-Order Short-Period Model</span>\n"
        "        <span class=\"header-tag ro\">Read-Only</span>\n"
        "      </div>\n    </div>\n  </div>\n</header>\n<main>\n"
        # Aircraft section (main visual) -- the qualitative pitch-rate
        # visualization is the rotating aircraft graphic driven by the
        # currently selected stored delta_q_rads sample.
        "  <section id=\"aircraft-section\">\n    <h2>Aircraft</h2>\n"
        "    <div class=\"aircraft-hero-wrap\">", _AIRCRAFT_HERO_SVG, "</div>\n"
        "    <p class=\"small\">Qualitative pitch-rate visualization: the aircraft graphic's "
        "rotation is a bounded, non-physical indicator driven directly by the selected stored "
        "delta_q_rads sample. This is not aircraft attitude \u2014 no pitch angle is computed "
        "or integrated.</p>\n"
        "    <div class=\"time-sel-val\" style=\"max-width:320px\">"
        "<div class=\"ts-label\">delta_q_rads (raw, selected sample)</div>"
        "<div class=\"ts-value\" id=\"q-indicator-raw\">\u2014</div></div>\n"
        "  </section>\n\n"
        # Telemetry section
        "  <section id=\"telemetry-section\">\n    <h2>Telemetry Overview</h2>\n"
        f"    <div class=\"telem-grid\">{telemetry_html}</div>\n"
        "    <p class=\"small\" style=\"margin-top:10px\">SIMULATED values are from the stored Simulator CSV result. "
        "REFERENCE values are from the associated scenario metadata. Values not available through the existing "
        "dashboard pipeline are marked N/A.</p>\n  </section>\n\n"
        # Run / Scenario section
        "  <section>\n    <h2>A. Run / Scenario</h2>\n"
        "    <form method=\"get\" action=\"/\" class=\"controls\">\n"
        f"      <div><label for=\"run\">Saved result</label><select id=\"run\" name=\"run\">{''.join(run_options)}</select></div>\n"
        f"      <div><label for=\"scenario\">{scenario_label}</label><select id=\"scenario\" name=\"scenario\">{''.join(scenario_options)}</select></div>\n"
        "      <div><button type=\"submit\">Load view</button></div>\n    </form>\n"
        "    <p class=\"small\">Selections are session/request state only. The dashboard does not write config.yaml, "
        "scenario definitions, or saved results.</p>\n"
        f"    {registry_message}\n"
        f"    {_render_scenario(bundle.scenario, user_supplied=scenario_user_supplied)}\n"
        "  </section>\n\n"
        # Summary section
        f"  <section><h2>B. Run Summary</h2><div class=\"metrics\">{''.join(summary_cards)}</div>\n"
        "    <div class=\"callout\" style=\"margin-top:12px\"><strong>Propulsion semantics:</strong> "
        "<code>MODEL_NOT_IDENTIFIED</code> is displayed unchanged as model status; it is not relabeled as engine failure.</div>\n"
        "  </section>\n\n"
        # Control input section
        "  <section id=\"control-input-section\">\n    <h2>Control Input</h2>\n"
        f"    <div class=\"chart-grid\">{elevator_chart}</div>\n"
        "    <p class=\"small\">Control input is the stored delta_e_rad from the Simulator CSV. Control schedules are "
        "PROJECT_DEFINED unless otherwise noted in scenario provenance.</p>\n"
        "  </section>\n\n"
        # Time selector section
        "  <section id=\"time-selector-section\">\n    <h2>Time / Sample Selector</h2>\n"
        "    <div class=\"time-sel\">\n"
        "      <div class=\"time-sel-ctrl\">\n"
        "        <label for=\"time-slider\">Sample Index</label>\n"
        f"        <input type=\"range\" id=\"time-slider\" min=\"0\" max=\"{max_index}\" value=\"0\" step=\"1\">\n"
        f"        <div class=\"small\" id=\"time-index-display\">Index: 0 / {max_index}</div>\n"
        "      </div>\n"
        "      <div class=\"time-sel-readout\" id=\"time-readout\">\n"
        "        <div class=\"time-sel-val\"><div class=\"ts-label\">Time [s]</div>"
        "<div class=\"ts-value\" id=\"ts-time\">\u2014</div></div>\n"
        "        <div class=\"time-sel-val\"><div class=\"ts-label\">delta_w [m/s]</div>"
        "<div class=\"ts-value\" id=\"ts-dw\">\u2014</div></div>\n"
        "        <div class=\"time-sel-val\"><div class=\"ts-label\">delta_q [rad/s]</div>"
        "<div class=\"ts-value\" id=\"ts-dq\">\u2014</div></div>\n"
        "        <div class=\"time-sel-val\"><div class=\"ts-label\">delta_e [rad]</div>"
        "<div class=\"ts-value\" id=\"ts-de\">\u2014</div></div>\n"
        "      </div>\n    </div>\n"
        "    <p class=\"small\">Values displayed are the exact stored samples at the selected index. "
        "No interpolation or smoothing is applied.</p>\n  </section>\n\n"
        # Existing sections
        f"  <section><h2>C. True State</h2><div class=\"chart-grid\">{true_charts}</div></section>\n"
        f"  <section><h2>D. Sensor Data</h2><div class=\"chart-grid\">{sensor_html}</div></section>\n"
        f"  <section><h2>E. Estimated State</h2><div class=\"chart-grid\">{estimate_html}</div></section>\n"
        f"  <section><h2>F. Measured vs Estimated</h2><div class=\"chart-grid\">{compare_html}</div></section>\n"
        f"  <section><h2>G. Residuals</h2><div class=\"chart-grid\">{residual_html}</div></section>\n"
        f"  <section><h2>H. Health</h2>{_render_health(bundle.health_data)}</section>\n"
        f"  <section><h2>I. Provenance</h2>{_render_provenance(bundle.provenance_data)}</section>\n"
        f"  <section><h2>Source Columns / Units</h2><div class=\"table-card\"><table><thead><tr><th>Column</th>"
        f"<th>Displayed unit semantics</th></tr></thead><tbody>{column_rows}</tbody></table></div>\n"
        "    <p class=\"small\">No smoothing, interpolation, resampling, normalization, or independent time vector "
        "is used. Internal perturbation_az remains m/s\u00b2 and is not presented as g.</p>\n"
        "  </section>\n"
        "</main>\n<script>\n", js_block, "</script>\n</body>\n</html>"
    ]
    return "".join(parts)


class DashboardApplication:
    """Read-only web application facade around the data-preparation functions."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, default_run: Optional[str] = None, default_scenario: Optional[str] = None) -> None:
        self.repository = RunRepository(data_dir)
        self.default_run = default_run
        self.default_scenario = default_scenario
        try:
            self.scenarios = load_scenarios()
            self.scenario_registry_error: Optional[str] = None
        except ScenarioRegistryError as exc:
            self.scenarios = tuple()
            self.scenario_registry_error = str(exc)

    def render(self, query: Mapping[str, Sequence[str]]) -> str:
        runs = self.repository.discover()
        if not runs:
            # V1 empty state: the dashboard must remain usable (and its
            # /healthz endpoint reachable) when no saved results exist yet.
            return render_empty_state_page(
                scenarios=self.scenarios,
                scenario_registry_error=self.scenario_registry_error,
            )

        requested_run = (query.get("run") or [self.default_run or runs[0].stem])[0]
        run = self.repository.load_by_name(requested_run)

        meta_scenario = run.metadata.get("scenario_name")
        local_registry_error = self.scenario_registry_error

        if meta_scenario:
            run_name_is_known = any(s.name == run.name for s in self.scenarios)
            meta_is_known = any(s.name == meta_scenario for s in self.scenarios)
            if run_name_is_known and meta_is_known and run.name != meta_scenario:
                requested_scenario = ""
                scenario_user_supplied = False
                mismatch_msg = f"Scenario metadata mismatch: run name = {run.name}, metadata scenario = {meta_scenario}"
                local_registry_error = f"{local_registry_error} | {mismatch_msg}" if local_registry_error else mismatch_msg
            else:
                requested_scenario = meta_scenario
                scenario_user_supplied = False
        else:
            requested_scenario = (query.get("scenario") or [self.default_scenario or ""])[0]
            scenario_user_supplied = bool(requested_scenario)
            
        scenario = find_scenario_view(requested_scenario, self.scenarios)
        if requested_scenario and scenario is None:
            raise ScenarioRegistryError(f"Unknown scenario selection: {requested_scenario}")

        bundle = prepare_dashboard_data(run, scenario=scenario, integrated_results=None)
        return render_dashboard_page(
            bundle,
            runs=runs,
            scenarios=self.scenarios,
            scenario_registry_error=local_registry_error,
            scenario_user_supplied=scenario_user_supplied,
        )


def render_empty_state_page(
    scenarios: Sequence[ScenarioView] = (),
    scenario_registry_error: Optional[str] = None,
) -> str:
    """Render the normal V1 empty state when no saved results exist yet.

    This is not an error: the dashboard is a read-only post-processing view,
    and a fresh project may legitimately have no ``data/processed/*.csv``
    files.  Simulation execution remains owned by ``main.py`` / the
    Simulator, not by the dashboard.
    """

    registry_message = (
        f'<div class="callout bad-callout"><strong>Scenario registry unavailable:</strong> {html.escape(scenario_registry_error)}</div>'
        if scenario_registry_error else ""
    )
    scenario_items = "".join(f"<li>{html.escape(s.name)}</li>" for s in scenarios)
    scenario_block = (
        f'<div class="table-card"><h3>Known scenario registry entries (metadata only)</h3><ul>{scenario_items}</ul></div>'
        if scenario_items else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NASA AirSTAR T-2 Digital Twin Dashboard</title>
<style>
:root {{ color-scheme: light dark; --bg:#0b1020; --panel:#121a2b; --panel2:#18233a; --text:#e7edf7; --muted:#9fb0c8; --line:#2b3a58; --accent:#78a9ff; --good:#38b26b; --warn:#d5a21a; --bad:#e45d68; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }}
header {{ padding:28px clamp(18px,4vw,52px); border-bottom:1px solid var(--line); background:linear-gradient(120deg,#10192b,#0b1020); }}
header h1 {{ margin:0 0 6px; font-size:clamp(22px,3vw,34px); }}
header p {{ margin:0; color:var(--muted); }}
main {{ max-width:1500px; margin:0 auto; padding:24px clamp(14px,3vw,38px) 70px; }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; margin:18px 0; padding:22px; box-shadow:0 12px 35px rgba(0,0,0,.16); }}
h2 {{ margin:0 0 16px; font-size:21px; }} h3 {{ margin:0 0 12px; font-size:16px; }}
.callout {{ border:1px dashed var(--line); background:rgba(120,169,255,.07); color:var(--muted); padding:13px 15px; border-radius:10px; line-height:1.45; }}
.bad-callout {{ border-color:rgba(228,93,104,.6); }}
.table-card {{ margin-top:14px; background:var(--panel2); border:1px solid var(--line); border-radius:12px; padding:14px; }}
.small {{ color:var(--muted); font-size:12px; line-height:1.5; }}
code {{ background:#0f1729; padding:2px 5px; border-radius:5px; }}
ul {{ color:var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>NASA AirSTAR T-2 Digital Twin</h1>
  <p>Engineering post-processing dashboard · reduced-order state x = [delta_w, delta_q] · read-only</p>
</header>
<main>
  <section>
    <h2>No simulation results available</h2>
    <div class="callout">No saved Simulator CSV files were found in the configured result directory. The V1 dashboard is strictly post-processing: it does not execute simulations and never fabricates data. Produce results first with the existing project entry point, e.g. <code>python main.py --scenario &lt;scenario-name&gt;</code>, then reload this page.</div>
    {registry_message}
    {scenario_block}
    <p class="small">Launch command (unchanged): <code>python dashboard/app.py --host 127.0.0.1 --port 8000</code></p>
  </section>
</main>
</body>
</html>"""


def _error_page(message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Dashboard error</title>
<style>body{{font-family:system-ui;background:#0b1020;color:#e7edf7;padding:40px}}.box{{max-width:760px;margin:auto;background:#121a2b;border:1px solid #2b3a58;border-radius:14px;padding:24px}}a{{color:#78a9ff}}</style></head>
<body><div class="box"><h1>{status.value} {html.escape(status.phrase)}</h1><p>{html.escape(message)}</p><p><a href="/">Return to dashboard</a></p></div></body></html>"""


def make_handler(application: DashboardApplication):
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "AirSTARDashboard/1.0"

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            if parsed.path == "/healthz":
                body = b"ok\n"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path != "/":
                page = _error_page("Page not found.", HTTPStatus.NOT_FOUND)
                self._write_html(HTTPStatus.NOT_FOUND, page)
                return

            try:
                page = application.render(parse_qs(parsed.query, keep_blank_values=True))
                self._write_html(HTTPStatus.OK, page)
            except ResultNotFoundError as exc:
                self._write_html(HTTPStatus.NOT_FOUND, _error_page(str(exc), HTTPStatus.NOT_FOUND))
            except (EmptyResultError, ResultSchemaError, MalformedResultError, ScenarioRegistryError) as exc:
                self._write_html(HTTPStatus.BAD_REQUEST, _error_page(str(exc), HTTPStatus.BAD_REQUEST))
            except DashboardError as exc:
                self._write_html(HTTPStatus.INTERNAL_SERVER_ERROR, _error_page(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR))
            except Exception:
                # Normal UI operation never exposes a raw traceback.
                self._write_html(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    _error_page("Unexpected dashboard failure. Check the server log for diagnostics.", HTTPStatus.INTERNAL_SERVER_ERROR),
                )

        def _write_html(self, status: HTTPStatus, page: str) -> None:
            body = page.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep the standard concise server log; no browser-facing traceback.
            sys.stderr.write("[dashboard] " + (fmt % args) + "\n")

    return DashboardHandler


def create_server(
    application: DashboardApplication,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    """Create the HTTP server.  Separate from serve_forever for smoke tests."""

    return ThreadingHTTPServer((host, port), make_handler(application))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NASA AirSTAR T-2 engineering post-processing dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing existing Simulator CSV results")
    parser.add_argument("--run", dest="default_run", help="Initial saved run name, without .csv")
    parser.add_argument("--scenario", dest="default_scenario", help="Initial scenario name from the existing registry")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not (0 <= args.port <= 65535):
        print(f"Error: --port must be between 0 and 65535, got {args.port}", file=sys.stderr)
        return 2

    app = DashboardApplication(
        data_dir=args.data_dir,
        default_run=args.default_run,
        default_scenario=args.default_scenario,
    )
    server = create_server(app, host=args.host, port=args.port)
    host, port = server.server_address[:2]
    print(f"AirSTAR T-2 dashboard: http://{host}:{port}")
    print(f"Read-only result directory: {app.repository.data_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())