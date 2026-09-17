"""
digital_twin/integration.py

PROJECT_DEFINED_ORCHESTRATION layer for the NASA AirSTAR T-2 digital twin.

==============================================================================
WHY THIS FILE EXISTS (architecture decision: CREATE)
==============================================================================
main.py currently constructs and runs only:

    Aircraft, AerodynamicsDatabase, PropulsionModel  -> Simulator.run()
        -> sim.save_results()

It never imports or calls sensors.sensor_model.SensorModel,
digital_twin.estimator.KalmanStateEstimator, or
digital_twin.health_monitor.HealthMonitor. No other module wires those
three together either -- health_monitor.py's own docstring explicitly says
it was written without read access to the other modules and expects a
caller to build its `HealthCheckInput` "from whatever those modules
actually return", and estimator.py explicitly ships a
`sensor_output -> flat measurement vector` adapter
(`extract_measurement_from_sensor_output`) for exactly this purpose but
nothing in the project calls it.

That is a genuine, reusable, library-level orchestration gap -- not
something already provided by main.py (which is CLI/application
orchestration only) -- so CREATE is justified per the project's own
decision gate.

==============================================================================
ACTUAL DATA FLOW USED BELOW (verified against the uploaded source, not
guessed)
==============================================================================
simulation.simulator.Simulator.run() has NO public single-step method and
no simulation-status field; it runs an entire scenario internally (RK4 +
control-schedule evaluation) and returns `self.history`: a List[Dict] with
these exact keys per row (see Simulator._record_step):

    time_s, delta_w_mps, delta_q_rads, delta_e_rad, delta_w_dot,
    delta_q_dot, perturbation_az_mps2, delta_alpha_rad, delta_CL, delta_Cm,
    delta_Lift_N, delta_PitchMom_Nm, total_thrust_N, propulsion_model_status

There is no "simulation_status" field anywhere in the real Simulator --
this module therefore never invents one (see SIMULATION STATUS below).

    Simulator history row (truth)
        -> SensorModel.measure(delta_w, delta_q, perturbation_az)
        -> {"truth": {...}, "measured": {"delta_w_mps", "delta_q_rads",
            "perturbation_az_mps2"}, "sensor_model_status": ...}
        -> estimator.extract_measurement_from_sensor_output(...)   [EXISTING
           ADAPTER already shipped in digital_twin/estimator.py -- reused
           here verbatim, not reimplemented]
        -> KalmanStateEstimator.step(z, dt, u) -> EstimatedState
           (fields: delta_w_hat, delta_q_hat, covariance, estimator_status,
           step_index, provenance)
        -> HealthCheckInput(estimated_delta_w=..., estimated_delta_q=...,
           measured_delta_w=..., measured_delta_q=..., covariance=...,
           propulsion_status=..., simulation_status=None)
        -> HealthMonitor.evaluate(...) -> HealthMonitorResult

Field-name verification (per the project's own naming-collision warning):
    delta_w / delta_w_mps / delta_w_hat, and delta_q / delta_q_rads /
    delta_q_hat are three DIFFERENT names for three DIFFERENT things
    (truth, sensor-measured, estimated) and are never merged or aliased
    here -- each keeps its own field name exactly as its owning module
    defines it.

==============================================================================
STRICT OWNERSHIP (respected, not re-implemented here)
==============================================================================
This module contains NO aircraft/aerodynamic/propulsion physics, no
dimensional conversion, no RK4, no Kalman-filter equations, no health
threshold logic, no scenario definitions, no CLI parsing, and no CSV
persistence. It only coordinates the real public APIs of SensorModel,
KalmanStateEstimator, and HealthMonitor, adapts boundary data using the
adapter estimator.py already provides, validates its own boundary inputs,
and tracks the small piece of state HealthMonitor's own docstring says the
caller (not HealthMonitor) must own.

==============================================================================
CONTROL INPUT
==============================================================================
KalmanStateEstimator.predict()/step() accept an optional `u` (default
0.0) -- the actual API does not require a control input. Since Simulator's
history rows already carry the real elevator command as `delta_e_rad`,
this module passes that value through unchanged rather than either
inventing a command or ignoring one that is already available. If `B` is
the default zero matrix (as in `default_reduced_order_estimator`), this
has no numerical effect; it only matters if the caller supplied an
estimator built with `build_transition_from_flight_dynamics`'s non-zero B.

==============================================================================
PROPULSION / SIMULATION STATUS
==============================================================================
`propulsion_model_status` (e.g. "MODEL_NOT_IDENTIFIED",
"PLUGIN_MODEL_EVALUATED" -- see models/propulsion.py) is passed through to
HealthCheckInput.propulsion_status completely unchanged; this module never
reclassifies it as an engine/propulsion fault (HealthMonitor owns that
interpretation). `simulation_status` is always None today because the real
Simulator does not emit one -- see DATA FLOW above. If a future Simulator
revision adds one, extracting `truth_row.get("simulation_status")` (already
written below) will pick it up with no changes required here.

==============================================================================
RESIDUALS
==============================================================================
KalmanStateEstimator.update() computes its innovation (z - Hx) internally
but does not expose it on EstimatedState or return it publicly. Rather
than approximating it a second time here (which would duplicate estimator
math this module is not supposed to own), this module supplies only
`measured_*` and `estimated_*` to HealthCheckInput and relies on
HealthMonitor's own documented fallback: "if omitted but both a
measurement and an estimate are available for a channel, the monitor
falls back to the clearly-defined scalar difference residual = measurement
- estimate."

==============================================================================
PERSISTENCE HISTORY (HealthMonitor's own delegated responsibility)
==============================================================================
health_monitor.py's HealthMonitor.evaluate() docstring is explicit: "This
monitor never stores this history itself -- the caller decides whether/or
how to keep it, which keeps evaluate a pure function." Nothing else in the
project keeps that history, so this module does -- see
`_infer_raw_critical_rules` below for why it must be derived from the RAW
per-rule diagnostics rather than from the already-persistence-adjusted
`fault_flags` (using the adjusted result as history would let a genuinely
sustained fault get permanently stuck below the persistence threshold).

==============================================================================
RESET AND RUN ISOLATION
==============================================================================
Neither Simulator nor KalmanStateEstimator exposes a reset() method. This
module does not invent one on either class. Instead, `DigitalTwinPipeline`
is scoped to one run's worth of samples; `reset_run()` clears this
pipeline's OWN bookkeeping (timestamp-derived dt tracking, persistence
history) and requires the caller to supply a freshly-constructed
KalmanStateEstimator for true state isolation between separate runs,
since only the caller can construct a new one with the right x0/P0.

==============================================================================
NASA / PROVENANCE BOUNDARY
==============================================================================
This module introduces no new NASA claims. It preserves, unmodified,
whatever provenance/status strings SensorModel, KalmanStateEstimator, and
HealthMonitor already attach to their own outputs (e.g.
PROJECT_DEFINED_PLACEHOLDER_NOT_NASA_VERIFIED,
PROJECT_DEFINED_REDUCED_ORDER_LINEAR_KALMAN_FILTER,
PROJECT_DEFINED_RULE_BASED_HEALTH_MONITOR). This module's own contribution
is labeled PIPELINE_PROVENANCE = "PROJECT_DEFINED_ORCHESTRATION". Running
this pipeline end-to-end proves module interface compatibility only -- it
is NOT NASA validation of the estimator or health thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence

from sensors.sensor_model import SensorModel
from .estimator import (
    EstimatedState,
    KalmanStateEstimator,
    extract_measurement_from_sensor_output,
)
from .health_monitor import HealthCheckInput, HealthMonitor, HealthMonitorResult

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

PIPELINE_PROVENANCE = "PROJECT_DEFINED_ORCHESTRATION"

# Truth-row keys this module reads directly from a real
# simulation.simulator.Simulator history entry. Kept as named constants (not
# scattered string literals) so the exact dependency on Simulator's schema is
# visible in one place.
_REQUIRED_TRUTH_KEYS = ("time_s", "delta_w_mps", "delta_q_rads", "perturbation_az_mps2")


# ---------------------------------------------------------------------------
# Errors -- differentiated per the project's "do not convert every problem
# into a generic pipeline-failed message" requirement. Each wraps its real
# cause via `raise ... from exc` rather than swallowing it.
# ---------------------------------------------------------------------------


class IntegrationError(Exception):
    """Base class for all digital_twin.integration failures."""


class InvalidTruthSampleError(IntegrationError):
    """The truth sample given to the pipeline is malformed: a missing
    required field, a non-finite value, or a non-increasing timestamp
    within the current run."""


class SensorModelFailureError(IntegrationError):
    """sensors.sensor_model.SensorModel.measure() raised while processing
    an otherwise well-formed truth sample."""


class EstimatorFailureError(IntegrationError):
    """digital_twin.estimator.KalmanStateEstimator.predict()/update()
    raised while processing an otherwise well-formed measurement."""


class HealthMonitorFailureError(IntegrationError):
    """digital_twin.health_monitor.HealthMonitor.evaluate() raised while
    processing an otherwise well-formed HealthCheckInput."""


# ---------------------------------------------------------------------------
# Validation helper (mirrors the pattern already used in
# models/flight_dynamics.py, sensors/sensor_model.py, and
# digital_twin/estimator.py, rather than importing another module's
# private helper of the same shape).
# ---------------------------------------------------------------------------


def _require_finite_scalar(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidTruthSampleError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise InvalidTruthSampleError(f"{name} must be finite, got {value!r}")
    return float(value)




# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntegratedStepResult:
    """One fully-processed timestep: truth -> measurement -> estimate ->
    health assessment.

    Retains the ORIGINAL objects produced by each real module rather than
    copying every field into a new structure, so none of their own
    provenance/status metadata is lost, renamed, or re-labeled.
    """

    timestamp_s: float
    sensor_result: Dict[str, Any]
    estimated_state: EstimatedState
    health_result: HealthMonitorResult
    propulsion_status: Optional[str]
    pipeline_provenance: str = PIPELINE_PROVENANCE


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DigitalTwinPipeline:
    """
    Reusable, library-level orchestration for one simulation run:

        truth sample (a simulation.simulator.Simulator history row)
            -> SensorModel.measure()
            -> KalmanStateEstimator.step() / .update()
            -> HealthMonitor.evaluate()
            -> IntegratedStepResult

    This class coordinates three already-existing, independently-owned
    subsystems. main.py remains the CLI/application entry point (argparse,
    scenario selection, config loading, starting the simulation, printing
    a summary); this class is the argparse-free, reusable counterpart --
    something a notebook, a test, or a future dashboard can import and
    drive directly against a Simulator's returned history.

    Scoped to ONE run's worth of samples per instance -- see "RESET AND RUN
    ISOLATION" in the module docstring for why, and use `reset_run()`
    before reusing an instance for a second, independent run.
    """

    def __init__(
        self,
        sensor_model: SensorModel,
        estimator: KalmanStateEstimator,
        health_monitor: HealthMonitor,
        measurement_channels: Sequence[str] = ("delta_w_mps", "delta_q_rads"),
    ) -> None:
        if not isinstance(sensor_model, SensorModel):
            raise TypeError(f"sensor_model must be a SensorModel, got {type(sensor_model).__name__}")
        if not isinstance(estimator, KalmanStateEstimator):
            raise TypeError(f"estimator must be a KalmanStateEstimator, got {type(estimator).__name__}")
        if not isinstance(health_monitor, HealthMonitor):
            raise TypeError(f"health_monitor must be a HealthMonitor, got {type(health_monitor).__name__}")
        if not measurement_channels:
            raise ValueError("measurement_channels must not be empty.")

        self.sensor_model = sensor_model
        self.estimator = estimator
        self.health_monitor = health_monitor
        # Forwarded verbatim to the EXISTING
        # estimator.extract_measurement_from_sensor_output adapter; only
        # needs overriding if the caller constructed an estimator with a
        # measurement model wider than the H=I 2-state default (e.g. one
        # that also fuses accelerometer_az).
        self.measurement_channels = tuple(measurement_channels)

        self._last_time_s: Optional[float] = None
        self._recent_critical_rules: List[FrozenSet[str]] = []

    def reset_run(self, estimator: Optional[KalmanStateEstimator] = None) -> None:
        """Starts a fresh, isolated run.

        Clears this pipeline's own per-run bookkeeping (dt-tracking
        timestamp, persistence history). If `estimator` is given it
        REPLACES self.estimator -- required for true state isolation,
        since KalmanStateEstimator carries its (x, P) internally and has
        no reset() of its own. If omitted, the existing estimator instance
        is kept as-is; the caller is responsible for that instance not
        carrying state from a previous run.
        """
        if estimator is not None:
            if not isinstance(estimator, KalmanStateEstimator):
                raise TypeError(f"estimator must be a KalmanStateEstimator, got {type(estimator).__name__}")
            self.estimator = estimator
        self._last_time_s = None
        self._recent_critical_rules = []

    def process_truth_sample(self, truth_row: Mapping[str, Any]) -> IntegratedStepResult:
        """Processes one Simulator history row through
        SensorModel -> KalmanStateEstimator -> HealthMonitor and returns
        the combined result.

        Raises:
            InvalidTruthSampleError: missing/non-finite required field, or
                a timestamp that does not strictly increase within this run.
            SensorModelFailureError, EstimatorFailureError,
                HealthMonitorFailureError: the corresponding real module
                raised; the original exception is preserved as __cause__.
        """
        # 1. Validate + extract boundary data (integration.py's own job --
        #    it does not silently map similarly-named fields, and it does
        #    not let a malformed row reach the real modules unexamined).
        missing = [k for k in _REQUIRED_TRUTH_KEYS if k not in truth_row]
        if missing:
            raise InvalidTruthSampleError(
                f"truth_row is missing required key(s) {missing}; expected the "
                f"simulation.simulator.Simulator history-row schema "
                f"{_REQUIRED_TRUTH_KEYS}."
            )
        time_s = _require_finite_scalar("time_s", truth_row["time_s"])
        delta_w = _require_finite_scalar("delta_w_mps", truth_row["delta_w_mps"])
        delta_q = _require_finite_scalar("delta_q_rads", truth_row["delta_q_rads"])
        perturbation_az = _require_finite_scalar(
            "perturbation_az_mps2", truth_row["perturbation_az_mps2"]
        )

        raw_control = truth_row.get("delta_e_rad", 0.0)
        control_input = 0.0 if raw_control is None else _require_finite_scalar("delta_e_rad", raw_control)

        propulsion_status = truth_row.get("propulsion_model_status")
        # Not currently emitted anywhere by the real Simulator -- see
        # module docstring. Read defensively so this module needs no
        # change if a future Simulator revision adds one.
        simulation_status = truth_row.get("simulation_status")

        # 2. Timestep bookkeeping. Never invents an independent clock --
        #    dt is always derived from the real simulator timestamps
        #    already on the truth samples themselves.
        if self._last_time_s is not None and time_s <= self._last_time_s:
            raise InvalidTruthSampleError(
                f"Non-increasing timestamp: previous sample was at "
                f"t={self._last_time_s}s, this sample is at t={time_s}s. "
                f"process_truth_sample() requires strictly increasing "
                f"time_s values from a single run -- call reset_run() "
                f"before starting a new one."
            )
        dt = None if self._last_time_s is None else (time_s - self._last_time_s)
        if dt is not None and (not math.isfinite(dt) or dt <= 0.0):
            raise InvalidTruthSampleError(f"dt must be positive and finite, got {dt}")
        self._last_time_s = time_s

        # 3. SensorModel: truth -> measurement (real API call, no adapter
        #    needed at this boundary -- SensorModel.measure()'s own
        #    signature already takes these three truth scalars directly).
        try:
            sensor_result = self.sensor_model.measure(
                delta_w=delta_w, delta_q=delta_q, perturbation_az=perturbation_az
            )
        except Exception as exc:
            raise SensorModelFailureError(
                f"SensorModel.measure() failed for t={time_s}s: {exc}"
            ) from exc

        # 4. Estimator: measurement -> estimate, via the EXISTING adapter
        #    (digital_twin.estimator.extract_measurement_from_sensor_output)
        #    rather than re-deriving the sensor->estimator mapping here.
        try:
            z = extract_measurement_from_sensor_output(
                sensor_result, channels=self.measurement_channels
            )
            if dt is None:
                # First sample of this run: x0/P0 already represent the
                # state at this instant, before any measurement. No
                # predict() is performed since no time has passed yet,
                # and no update() is performed to preserve the exact
                # initial conditions.
                estimated_state = self.estimator.get_estimated_state()
            else:
                estimated_state = self.estimator.step(z, dt=dt, u=control_input)
        except Exception as exc:
            raise EstimatorFailureError(
                f"Estimator failed for t={time_s}s: {exc}"
            ) from exc

        # 5. HealthMonitor: build the actual HealthCheckInput contract.
        #    Residuals are intentionally omitted -- see "RESIDUALS" in the
        #    module docstring for why HealthMonitor's own measured/
        #    estimated fallback is used instead of approximating the
        #    estimator's internal innovation here.
        if "measured" not in sensor_result:
            raise IntegrationError("SensorModel output missing 'measured' dict.")
        try:
            measured_w = sensor_result["measured"]["delta_w_mps"]
            measured_q = sensor_result["measured"]["delta_q_rads"]
        except KeyError as exc:
            raise IntegrationError(f"SensorModel output missing required channel: {exc}") from exc

        health_input = HealthCheckInput(
            estimated_delta_w=estimated_state.delta_w_hat,
            estimated_delta_q=estimated_state.delta_q_hat,
            measured_delta_w=measured_w,
            measured_delta_q=measured_q,
            covariance=estimated_state.covariance,
            propulsion_status=propulsion_status,
            simulation_status=simulation_status,
        )
        history_arg = tuple(self._recent_critical_rules) if self._recent_critical_rules else None
        try:
            health_result = self.health_monitor.evaluate(
                health_input, recent_critical_rules=history_arg
            )
        except Exception as exc:
            raise HealthMonitorFailureError(
                f"HealthMonitor.evaluate() failed for t={time_s}s: {exc}"
            ) from exc

        # 6. Own the persistence-history bookkeeping HealthMonitor's own
        #    docstring explicitly delegates to the caller -- see
        #    "PERSISTENCE HISTORY" in the module docstring.
        self._recent_critical_rules.append(self.health_monitor.infer_raw_critical_rules(health_result.diagnostics))
        required_history = max(self.health_monitor.config.persistence_samples - 1, 0)
        if required_history == 0:
            self._recent_critical_rules.clear()
        else:
            excess = len(self._recent_critical_rules) - required_history
            if excess > 0:
                del self._recent_critical_rules[:excess]

        return IntegratedStepResult(
            timestamp_s=time_s,
            sensor_result=sensor_result,
            estimated_state=estimated_state,
            health_result=health_result,
            propulsion_status=propulsion_status,
        )

    def process_history(
        self, history: Sequence[Mapping[str, Any]]
    ) -> List[IntegratedStepResult]:
        """Thin convenience wrapper: feeds an already-completed
        simulation.simulator.Simulator.run() history through
        process_truth_sample() in order.

        Requires a fresh pipeline (or a call to reset_run() with a NEW
        estimator instance) for each history. The estimator's internal state
        is mutable and must not be blindly carried over from a previous run.

        This does NOT run a new simulation loop. Simulator remains solely
        responsible for time-stepping, RK4 integration, and control-
        schedule execution; this method only consumes its already-computed
        samples.
        """
        if self._last_time_s is not None:
            raise IntegrationError("process_history must be called on a fresh pipeline; call reset_run(estimator=...) first.")
        if not history:
            raise InvalidTruthSampleError("history is empty; nothing to process.")
        return [self.process_truth_sample(row) for row in history]