"""
digital_twin/health_monitor.py

PROJECT_DEFINED_RULE_BASED_HEALTH_MONITOR
==========================================

A simple, transparent, rule-based health/diagnostic layer for the reduced
order (2-state) Digital Twin of the NASA AirSTAR 5.5% Generic Transport
Model (GTM / T-2).

WHAT THIS MODULE IS
--------------------
    sensor measurements
          +
    estimated state (delta_w, delta_q)
          +
    estimator residuals / covariance diagnostics
          +
    propulsion / simulation diagnostics
          v
    HealthMonitor.evaluate(...)
          v
    HealthMonitorResult (status, severity, coverage, fault flags,
                          diagnostics, full threshold provenance)

This module OBSERVES the outputs of the existing pipeline. It does not:

    * control the aircraft
    * modify simulation state
    * modify estimator state
    * perform state estimation or Kalman filtering
    * perform aerodynamic identification
    * perform machine learning
    * perform full 6-DOF monitoring
    * invent NASA fault thresholds

RESEARCH / PROVENANCE NOTE
---------------------------
A literature check (NASA AirSTAR / GTM-T2 technical reports and the
published fault-detection literature built on the GTM, e.g. GTM-DesignSim
studies, DUKF-based FDD papers, and NASA's "probing" health-monitoring
work) did NOT turn up a publicly documented, authoritative numeric limit
for this project's specific reduced-order signals:

    * delta_w  (perturbation state, this project's reduced-order model)
    * delta_q  (perturbation state, this project's reduced-order model)
    * normal/body-Z acceleration
    * sensor residual magnitudes
    * sensor failure thresholds
    * actuator failure thresholds
    * propulsion anomaly thresholds

for THIS project's 2-state reduced-order representation. NASA/academic GTM
fault-detection work exists (see e.g. NASA "Probing the NASA Generic
Transport Aircraft for Health Monitoring", and GTM actuator-fault DUKF
studies), but those use different (6-DOF, actuator-fault-injection, or
statistical change-detection) formulations and do not publish limits that
map directly onto this project's delta_w / delta_q reduced state.

Consequently, EVERY numeric threshold in this module is provenance-tagged
PROJECT_DEFINED_LIMIT (or TEST_ONLY_LIMIT for unit tests). None are tagged
NASA_VERIFIED_LIMIT. If a genuinely NASA-sourced, directly-applicable
limit is later identified, update the relevant ThresholdSpec's
`provenance` field (and cite the source) -- do not just change the label.

INTERFACE NOTE
---------------
This task did not have read access to the project's actual
`sensors/sensor_model.py`, `digital_twin/estimator.py`,
`models/propulsion.py`, or `simulation/simulator.py` source, so this
module intentionally depends on nothing but plain Python values (floats,
dicts, strings) via the `HealthCheckInput` dataclass, rather than
importing those classes directly. Build a `HealthCheckInput` from
whatever those modules actually return (e.g.
`HealthCheckInput(estimated_delta_w=state.delta_w, ...)`). No sensor
noise model, Kalman filter, aerodynamics, propulsion, or simulation
integration logic is reimplemented here.

CHANGELOG (v1.1, over the original single-pass design)
--------------------------------------------------------
* Added `Coverage` (FULL / PARTIAL) to `HealthMonitorResult`, plus
  `skipped_rules`, so a caller can tell "nothing wrong" apart from
  "not enough optional input was supplied to check everything."
* Split fault rules into hard (structural / explicit-status) and soft
  (continuous threshold crossings). The optional temporal-persistence
  feature (`persistence_samples > 1`) now only ever softens SOFT rules
  (STATE_LIMIT_CHECK.*, SENSOR_RESIDUAL_CHECK.*). HARD rules
  (FINITE_VALUE_CHECK, ESTIMATOR_COVARIANCE_CHECK, PROPULSION_STATUS_CHECK,
  SIMULATION_STATUS_CHECK) always report at full severity immediately --
  persistence never delays or downgrades them. This addresses "a single
  noisy sample shouldn't read as FAULT" without also delaying reporting
  of unambiguous structural/status faults.
* Added `FaultCategory` (SENSOR / ESTIMATOR / DYNAMICS / PROPULSION /
  SIMULATION) to `FaultFlag` for basic root-cause grouping/triage.

Deliberately NOT done in this pass (see project review notes -- these
need either real project source access or a product decision, not just
more rule-writing):
* Normalized/NIS-based residual checks (residuals here remain raw
  scalar differences).
* Replacing the propulsion/simulation status string sets with the
  actual enums emitted by models/propulsion.py and
  simulation/simulator.py -- those sets remain project-defined and
  configurable (see HealthMonitorConfig) until that source is available.
* Hysteresis / trend detection across samples beyond the existing
  persistence mechanism.
* De-duplicating covariance validation against the estimator's own
  covariance contract (this module still does its own boundary
  sanity-check).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Provenance",
    "Severity",
    "HealthStatus",
    "Coverage",
    "FaultCategory",
    "ThresholdSpec",
    "FaultFlag",
    "HealthCheckInput",
    "HealthMonitorResult",
    "HealthMonitorConfig",
    "HealthMonitor",
]


# ======================================================================
# Provenance / status / severity / coverage / category vocabulary
# ======================================================================

class Provenance(str, Enum):
    """Where a threshold or rule came from. Every threshold MUST carry one."""

    #: A limit directly supported by an identifiable, authoritative NASA
    #: source for THIS quantity on THIS vehicle/model. Not used by any
    #: default threshold in this module -- see module docstring.
    NASA_VERIFIED_LIMIT = "NASA_VERIFIED_LIMIT"

    #: A limit chosen by this project (not NASA-sourced) for reduced-order
    #: monitoring purposes. This is the default provenance for all
    #: operational thresholds in this module.
    PROJECT_DEFINED_LIMIT = "PROJECT_DEFINED_LIMIT"

    #: A limit that exists only to exercise a code path in a unit test.
    #: Must never be used as an operational default.
    TEST_ONLY_LIMIT = "TEST_ONLY_LIMIT"

    #: Not a numeric limit at all -- a structural/logical rule (e.g.
    #: "reject non-finite input", "covariance must be positive
    #: semi-definite"). Still project-defined, not NASA-sourced.
    PROJECT_DEFINED_RULE = "PROJECT_DEFINED_RULE"


class Severity(str, Enum):
    """Deterministic, non-probabilistic severity levels for a single check."""

    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return {"HEALTHY": 0, "WARNING": 1, "CRITICAL": 2}[self.value]


class HealthStatus(str, Enum):
    """Overall result status. Smallest sensible set, per spec."""

    #: All evaluated checks passed.
    HEALTHY = "HEALTHY"
    #: At least one check is out of tolerance, but nothing structurally broken.
    WARNING = "WARNING"
    #: At least one check indicates a critical, well-formed-but-out-of-limits
    #: condition.
    FAULT = "FAULT"
    #: The input itself is unusable (non-finite values, malformed
    #: covariance, etc.) -- no health judgement can be made.
    INVALID = "INVALID"


class Coverage(str, Enum):
    """How much of the optional input surface was available for this evaluation.

    FULL: every rule that could have been evaluated for this sample was
    evaluated (no rule was skipped for lack of optional input).
    PARTIAL: at least one rule was skipped because its optional input
    (residual/measurement, covariance, propulsion_status,
    simulation_status) was not supplied. PARTIAL is not itself a fault --
    it just means overall_status reflects fewer checks than a complete
    input set would allow, so a HEALTHY+PARTIAL result should be read
    with that caveat.
    """

    FULL = "FULL"
    PARTIAL = "PARTIAL"


class FaultCategory(str, Enum):
    """Coarse root-cause grouping for a FaultFlag, for triage/routing.

    This is a project-defined convenience grouping, not a diagnosis of
    which subsystem actually caused the underlying condition -- e.g. a
    SENSOR-categorized residual flag could equally reflect an estimator
    problem. It exists so callers can sort/route flags without having to
    parse rule-name strings.
    """

    SENSOR = "SENSOR"
    ESTIMATOR = "ESTIMATOR"
    DYNAMICS = "DYNAMICS"
    PROPULSION = "PROPULSION"
    SIMULATION = "SIMULATION"


# ======================================================================
# Thresholds / configuration
# ======================================================================

@dataclass(frozen=True)
class ThresholdSpec:
    """A single, provenance-tagged numeric threshold.

    Thresholds are never hidden inside comparison logic -- every limit
    used by a rule is represented as one of these and carried through to
    the result's diagnostics for traceability.
    """

    value: float
    provenance: Provenance
    severity: Severity = Severity.WARNING
    unit: str = "unspecified"
    description: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError(f"ThresholdSpec.value must be finite, got {self.value!r}")
        if self.severity is Severity.HEALTHY:
            raise ValueError("A threshold cannot have severity HEALTHY (it must WARN or be CRITICAL when exceeded)")


@dataclass(frozen=True)
class HealthMonitorConfig:
    """All configurable thresholds/statuses used by :class:`HealthMonitor`.

    Every threshold is a :class:`ThresholdSpec` (value + provenance +
    severity), never a bare number. Two-tier (WARNING / CRITICAL)
    thresholds are used where useful; this is still a purely deterministic
    comparison against fixed limits, not a probabilistic score.
    """

    # --- state limits (estimated delta_w, delta_q) ---
    delta_w_warning_limit: ThresholdSpec = field(default_factory=lambda: ThresholdSpec(
        value=3.0, unit="m/s", severity=Severity.WARNING,
        provenance=Provenance.PROJECT_DEFINED_LIMIT,
        description="Soft bound on |estimated delta_w| for the reduced-order model.",
    ))
    delta_w_critical_limit: ThresholdSpec = field(default_factory=lambda: ThresholdSpec(
        value=6.0, unit="m/s", severity=Severity.CRITICAL,
        provenance=Provenance.PROJECT_DEFINED_LIMIT,
        description="Hard bound on |estimated delta_w| for the reduced-order model.",
    ))
    delta_q_warning_limit: ThresholdSpec = field(default_factory=lambda: ThresholdSpec(
        value=0.20, unit="rad/s", severity=Severity.WARNING,
        provenance=Provenance.PROJECT_DEFINED_LIMIT,
        description="Soft bound on |estimated delta_q| for the reduced-order model.",
    ))
    delta_q_critical_limit: ThresholdSpec = field(default_factory=lambda: ThresholdSpec(
        value=0.35, unit="rad/s", severity=Severity.CRITICAL,
        provenance=Provenance.PROJECT_DEFINED_LIMIT,
        description="Hard bound on |estimated delta_q| for the reduced-order model.",
    ))

    # --- residual limits (measurement - estimate) ---
    residual_delta_w_warning_limit: ThresholdSpec = field(default_factory=lambda: ThresholdSpec(
        value=0.5, unit="m/s", severity=Severity.WARNING,
        provenance=Provenance.PROJECT_DEFINED_LIMIT,
        description="Soft bound on |measured - estimated| delta_w residual.",
    ))
    residual_delta_w_critical_limit: ThresholdSpec = field(default_factory=lambda: ThresholdSpec(
        value=1.0, unit="m/s", severity=Severity.CRITICAL,
        provenance=Provenance.PROJECT_DEFINED_LIMIT,
        description="Hard bound on |measured - estimated| delta_w residual.",
    ))
    residual_delta_q_warning_limit: ThresholdSpec = field(default_factory=lambda: ThresholdSpec(
        value=0.02, unit="rad/s", severity=Severity.WARNING,
        provenance=Provenance.PROJECT_DEFINED_LIMIT,
        description="Soft bound on |measured - estimated| delta_q residual.",
    ))
    residual_delta_q_critical_limit: ThresholdSpec = field(default_factory=lambda: ThresholdSpec(
        value=0.05, unit="rad/s", severity=Severity.CRITICAL,
        provenance=Provenance.PROJECT_DEFINED_LIMIT,
        description="Hard bound on |measured - estimated| delta_q residual.",
    ))

    # --- covariance sanity ---
    covariance_finite_provenance: Provenance = Provenance.PROJECT_DEFINED_RULE
    covariance_psd_provenance: Provenance = Provenance.PROJECT_DEFINED_RULE
    # allow tiny negative diagonal/determinant from floating point noise
    covariance_psd_tolerance: float = 1e-9

    # --- propulsion status classification (see models/propulsion.py) ---
    # Statuses considered fully nominal.
    propulsion_ok_statuses: FrozenSet[str] = field(default_factory=lambda: frozenset({"NOMINAL", "OK"}))
    # Statuses that are a MODEL limitation, not an aircraft fault. Per
    # spec, MODEL_NOT_IDENTIFIED must never be classified as a fault.
    propulsion_model_status_statuses: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"MODEL_NOT_IDENTIFIED"})
    )
    # Statuses this project has explicitly and deliberately decided to
    # treat as a propulsion fault. Empty by default: this module does not
    # invent propulsion fault semantics. Extend this set only with status
    # strings the propulsion module actually emits with a known meaning.
    # NOTE: these are project-defined placeholder strings pending access
    # to models/propulsion.py's actual status contract (see CHANGELOG).
    propulsion_fault_statuses: FrozenSet[str] = field(default_factory=frozenset)

    # --- simulation status classification (see simulation/simulator.py) ---
    simulation_ok_statuses: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"OK", "RUNNING", "NOMINAL"})
    )
    # Statuses this project treats as simulation-level faults. Adjust to
    # match the actual status strings emitted by simulation/simulator.py.
    # NOTE: these are project-defined placeholder strings pending access
    # to simulation/simulator.py's actual status contract (see CHANGELOG).
    simulation_fault_statuses: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"DIVERGED", "INVALID"})
    )

    # --- optional temporal persistence (see PersistenceConfig below) ---
    # Number of consecutive samples a CRITICAL condition must hold before
    # it is reported as FAULT instead of WARNING. 1 = instantaneous
    # (default, matches "current implementation = instantaneous rule
    # evaluation" in the spec). This is PROJECT_DEFINED and optional, and
    # -- as of v1.1 -- only ever applies to SOFT rules (continuous
    # threshold crossings: STATE_LIMIT_CHECK.*, SENSOR_RESIDUAL_CHECK.*).
    # HARD rules (FINITE_VALUE_CHECK, ESTIMATOR_COVARIANCE_CHECK,
    # PROPULSION_STATUS_CHECK, SIMULATION_STATUS_CHECK) are never delayed
    # or downgraded by this setting.
    persistence_samples: int = 1

    def __post_init__(self) -> None:
        if self.persistence_samples < 1:
            raise ValueError("persistence_samples must be >= 1")


# ======================================================================
# Inputs
# ======================================================================

@dataclass(frozen=True)
class HealthCheckInput:
    """Everything the health monitor is given for one evaluation.

    Only ``estimated_delta_w`` and ``estimated_delta_q`` are required.
    Everything else is optional -- rules that need an unavailable input
    are simply skipped (and this is recorded in diagnostics, and rolled
    up into ``HealthMonitorResult.coverage``/``skipped_rules``), per the
    "do not require all of these for basic operation" requirement.
    """

    estimated_delta_w: float
    estimated_delta_q: float

    measured_delta_w: Optional[float] = None
    measured_delta_q: Optional[float] = None

    # Residuals may be supplied directly by the estimator (preferred --
    # this module never recomputes estimator equations). If omitted but
    # both a measurement and an estimate are available for a channel, the
    # monitor falls back to the clearly-defined scalar difference
    # residual = measurement - estimate.
    residual_delta_w: Optional[float] = None
    residual_delta_q: Optional[float] = None

    # 2x2 state error covariance [[P_ww, P_wq], [P_qw, P_qq]], if available.
    covariance: Optional[Sequence[Sequence[float]]] = None

    propulsion_status: Optional[str] = None
    simulation_status: Optional[str] = None


# ======================================================================
# Outputs
# ======================================================================

@dataclass(frozen=True)
class FaultFlag:
    """One violated rule, fully traceable back to its threshold."""

    rule: str
    severity: Severity
    provenance: Provenance
    category: FaultCategory
    observed: Optional[float]
    limit: Optional[float]
    unit: str
    message: str


@dataclass(frozen=True)
class HealthMonitorResult:
    """The full, structured output of one :meth:`HealthMonitor.evaluate` call."""

    overall_status: HealthStatus
    severity: Severity
    fault_flags: Tuple[FaultFlag, ...]
    #: Per-rule record (including passing rules) for full traceability:
    #: {rule_name: {"observed":..., "limit":..., "provenance":..., "passed":...}}
    diagnostics: Mapping[str, Mapping[str, object]]
    #: FULL if every applicable rule had the input it needed; PARTIAL if
    #: one or more rules were skipped for lack of optional input.
    coverage: Coverage
    #: Names of rules that were skipped for lack of optional input (a
    #: subset of ``diagnostics.keys()`` where that rule's entry has
    #: ``"passed": None``). Empty when coverage is FULL.
    skipped_rules: Tuple[str, ...]
    #: Overall provenance label for this module's methodology.
    provenance: str = "PROJECT_DEFINED_RULE_BASED_HEALTH_MONITOR"


# ======================================================================
# The monitor
# ======================================================================

class HealthMonitor:
    """PROJECT_DEFINED_RULE_BASED_HEALTH_MONITOR for the 2-state digital twin.

    ``evaluate`` is a pure function of its arguments: it does not mutate
    ``inputs``, does not hold internal mutable state, does not touch the
    estimator/simulator/aircraft, and issues no control commands. The same
    inputs always produce an identical, deep-equal result.
    """

    #: Rules that represent a structural condition or an explicit status
    #: string rather than a continuous threshold crossing. These are
    #: exempt from the optional persistence mechanism -- they are always
    #: reported at full severity on the very sample they occur, since
    #: softening them would delay reporting of an unambiguous fault.
    _PERSISTENCE_EXEMPT_RULES: FrozenSet[str] = frozenset({
        "FINITE_VALUE_CHECK",
        "ESTIMATOR_COVARIANCE_CHECK",
        "PROPULSION_STATUS_CHECK",
        "SIMULATION_STATUS_CHECK",
    })

    def __init__(self, config: Optional[HealthMonitorConfig] = None) -> None:
        self.config = config or HealthMonitorConfig()

    # -- public API -----------------------------------------------------

    def evaluate(
        self,
        inputs: HealthCheckInput,
        recent_critical_rules: Optional[Sequence[FrozenSet[str]]] = None,
    ) -> HealthMonitorResult:
        """Evaluate one sample and return a :class:`HealthMonitorResult`.

        Parameters
        ----------
        inputs:
            The values to check. Not modified.
        recent_critical_rules:
            Optional, caller-owned history for the PROJECT_DEFINED,
            optional temporal-persistence feature: a sequence of
            frozensets, each containing the SOFT rule names that were
            CRITICAL on one of the previous
            ``config.persistence_samples - 1`` samples, oldest first.
            Ignored when ``config.persistence_samples == 1`` (the
            default) or for HARD rules (see
            ``_PERSISTENCE_EXEMPT_RULES``). This monitor never stores
            this history itself -- the caller decides whether/how to
            keep it, which keeps ``evaluate`` a pure function.
        """
        cfg = self.config
        diagnostics: Dict[str, Dict[str, object]] = {}
        fault_flags: List[FaultFlag] = []

        # 1) FINITE_VALUE_CHECK -----------------------------------------
        non_finite = self._finite_value_check(inputs, diagnostics)
        if non_finite:
            coverage, skipped_rules = self._compute_coverage(diagnostics)
            return HealthMonitorResult(
                overall_status=HealthStatus.INVALID,
                severity=Severity.CRITICAL,
                fault_flags=tuple(non_finite),
                diagnostics=diagnostics,
                coverage=coverage,
                skipped_rules=skipped_rules,
            )

        # 2) ESTIMATOR_COVARIANCE_CHECK (structural -- may also be INVALID)
        cov_invalid = self._covariance_check(inputs, cfg, diagnostics, fault_flags)
        if cov_invalid:
            coverage, skipped_rules = self._compute_coverage(diagnostics)
            return HealthMonitorResult(
                overall_status=HealthStatus.INVALID,
                severity=Severity.CRITICAL,
                fault_flags=tuple(fault_flags),
                diagnostics=diagnostics,
                coverage=coverage,
                skipped_rules=skipped_rules,
            )

        # 3) STATE_LIMIT_CHECK --------------------------------------------
        self._state_limit_check(inputs, cfg, diagnostics, fault_flags)

        # 4) SENSOR_RESIDUAL_CHECK -----------------------------------------
        self._residual_check(inputs, cfg, diagnostics, fault_flags)

        # 5) PROPULSION_STATUS_CHECK ---------------------------------------
        self._propulsion_status_check(inputs, cfg, diagnostics, fault_flags)

        # 6) SIMULATION_STATUS_CHECK ---------------------------------------
        self._simulation_status_check(inputs, cfg, diagnostics, fault_flags)

        # -- optional temporal persistence (PROJECT_DEFINED, optional,
        #    SOFT rules only -- see _PERSISTENCE_EXEMPT_RULES) -----------
        fault_flags = self._apply_persistence(fault_flags, cfg, recent_critical_rules, diagnostics)

        overall_status, severity = self._classify(fault_flags)
        coverage, skipped_rules = self._compute_coverage(diagnostics)

        return HealthMonitorResult(
            overall_status=overall_status,
            severity=severity,
            fault_flags=tuple(fault_flags),
            diagnostics=diagnostics,
            coverage=coverage,
            skipped_rules=skipped_rules,
        )

    # -- coverage -----------------------------------------------------

    @staticmethod
    def infer_raw_critical_rules(diagnostics: Mapping[str, Mapping[str, Any]]) -> FrozenSet[str]:
        """Reconstructs which threshold-based rules were raw-CRITICAL on this
        sample, i.e. BEFORE HealthMonitor's own optional persistence smoothing.
        """
        raw_critical = set()
        for rule_name, entry in diagnostics.items():
            if not isinstance(entry, Mapping) or entry.get("passed") is not False:
                continue
            observed = entry.get("observed")
            critical_limit = entry.get("critical_limit")
            if (
                isinstance(observed, (int, float)) and not isinstance(observed, bool)
                and isinstance(critical_limit, (int, float)) and not isinstance(critical_limit, bool)
                and observed >= critical_limit
            ):
                raw_critical.add(rule_name)
        return frozenset(raw_critical)

    @staticmethod
    def _compute_coverage(
        diagnostics: Mapping[str, Mapping[str, object]],
    ) -> Tuple[Coverage, Tuple[str, ...]]:
        """A rule counts as skipped when its diagnostics entry has
        ``"passed": None`` -- the convention every rule below uses to mean
        "not enough optional input was supplied to evaluate this."."""
        skipped = tuple(
            rule
            for rule, entry in diagnostics.items()
            if isinstance(entry, dict) and entry.get("passed", "MISSING") is None
        )
        coverage = Coverage.PARTIAL if skipped else Coverage.FULL
        return coverage, skipped

    # -- individual rules -------------------------------------------------

    @staticmethod
    def _finite_value_check(
        inputs: HealthCheckInput,
        diagnostics: Dict[str, Dict[str, object]],
    ) -> List[FaultFlag]:
        """Rule: FINITE_VALUE_CHECK. Reject NaN/Inf in monitored inputs."""
        candidates: Dict[str, Optional[float]] = {
            "estimated_delta_w": inputs.estimated_delta_w,
            "estimated_delta_q": inputs.estimated_delta_q,
            "measured_delta_w": inputs.measured_delta_w,
            "measured_delta_q": inputs.measured_delta_q,
            "residual_delta_w": inputs.residual_delta_w,
            "residual_delta_q": inputs.residual_delta_q,
        }
        flags: List[FaultFlag] = []
        entry = {"provenance": Provenance.PROJECT_DEFINED_RULE.value, "checked_fields": []}
        for name, value in candidates.items():
            if value is None:
                continue
            entry["checked_fields"].append(name)
            if not math.isfinite(value):
                # measured_* comes from the sensor model; estimated_*/
                # residual_* comes from the estimator.
                category = FaultCategory.SENSOR if name.startswith("measured_") else FaultCategory.ESTIMATOR
                flags.append(FaultFlag(
                    rule="FINITE_VALUE_CHECK",
                    severity=Severity.CRITICAL,
                    provenance=Provenance.PROJECT_DEFINED_RULE,
                    category=category,
                    observed=value,
                    limit=None,
                    unit="n/a",
                    message=f"{name} is not finite (NaN/Inf): {value!r}",
                ))
        entry["passed"] = len(flags) == 0
        diagnostics["FINITE_VALUE_CHECK"] = entry
        return flags

    @staticmethod
    def _covariance_check(
        inputs: HealthCheckInput,
        cfg: HealthMonitorConfig,
        diagnostics: Dict[str, Dict[str, object]],
        fault_flags: List[FaultFlag],
    ) -> bool:
        """Rule: ESTIMATOR_COVARIANCE_CHECK.

        Returns True if the covariance is structurally invalid enough that
        the whole result must be INVALID (wrong shape or non-finite
        entries). A finite-but-not-positive-semi-definite covariance is
        recorded as a CRITICAL FaultFlag but does not, by itself, make the
        whole result INVALID (the state estimate may still be usable).
        """
        cov = inputs.covariance
        if cov is None:
            diagnostics["ESTIMATOR_COVARIANCE_CHECK"] = {
                "provenance": cfg.covariance_finite_provenance.value,
                "passed": None,
                "note": "no covariance supplied - rule skipped",
            }
            return False

        rows = list(cov)
        shape_ok = len(rows) == 2 and all(len(r) == 2 for r in rows)
        if not shape_ok:
            diagnostics["ESTIMATOR_COVARIANCE_CHECK"] = {
                "provenance": cfg.covariance_finite_provenance.value,
                "passed": False,
                "note": f"covariance must be 2x2, got shape-like {rows!r}",
            }
            fault_flags.append(FaultFlag(
                rule="ESTIMATOR_COVARIANCE_CHECK",
                severity=Severity.CRITICAL,
                provenance=cfg.covariance_finite_provenance,
                category=FaultCategory.ESTIMATOR,
                observed=None,
                limit=None,
                unit="n/a",
                message="Covariance is not a 2x2 matrix.",
            ))
            return True

        try:
            flat = [float(v) for row in rows for v in row]
            if not all(math.isfinite(v) for v in flat):
                raise ValueError("non-finite")
        except (TypeError, ValueError):
            diagnostics["ESTIMATOR_COVARIANCE_CHECK"] = {
                "provenance": cfg.covariance_finite_provenance.value,
                "passed": False,
                "note": "covariance contains non-finite or non-numeric entries",
            }
            fault_flags.append(FaultFlag(
                rule="ESTIMATOR_COVARIANCE_CHECK",
                severity=Severity.CRITICAL,
                provenance=cfg.covariance_finite_provenance,
                category=FaultCategory.ESTIMATOR,
                observed=None,
                limit=None,
                unit="n/a",
                message=f"Covariance has non-finite or non-numeric entries: {rows!r}",
            ))
            return True

        p_ww, p_wq = flat[0], flat[1]
        p_qw, p_qq = flat[2], flat[3]
        tol = cfg.covariance_psd_tolerance
        symmetric = abs(p_wq - p_qw) <= max(tol, 1e-6 * max(abs(p_wq), abs(p_qw), 1.0))
        det = p_ww * p_qq - p_wq * p_qw
        psd = (p_ww >= -tol) and (p_qq >= -tol) and (det >= -tol)

        passed = symmetric and psd
        diagnostics["ESTIMATOR_COVARIANCE_CHECK"] = {
            "provenance": cfg.covariance_psd_provenance.value,
            "passed": passed,
            "observed": {"P_ww": p_ww, "P_wq": p_wq, "P_qw": p_qw, "P_qq": p_qq, "det": det},
            "symmetric": symmetric,
            "positive_semi_definite": psd,
        }
        if not passed:
            fault_flags.append(FaultFlag(
                rule="ESTIMATOR_COVARIANCE_CHECK",
                severity=Severity.CRITICAL,
                provenance=cfg.covariance_psd_provenance,
                category=FaultCategory.ESTIMATOR,
                observed=det,
                limit=0.0,
                unit="n/a",
                message="Covariance is not a valid (symmetric, positive semi-definite) 2x2 matrix.",
            ))
        # Finite + correctly shaped covariance never forces INVALID, even
        # if it fails the PSD/symmetry check -- that is reported as a
        # CRITICAL fault flag instead, since the estimate itself is still
        # usable.
        return False

    @staticmethod
    def _state_limit_check(
        inputs: HealthCheckInput,
        cfg: HealthMonitorConfig,
        diagnostics: Dict[str, Dict[str, object]],
        fault_flags: List[FaultFlag],
    ) -> None:
        """Rule: STATE_LIMIT_CHECK. |estimated delta_w|, |estimated delta_q| vs configured limits."""
        checks = [
            ("STATE_LIMIT_CHECK.delta_w", inputs.estimated_delta_w,
             cfg.delta_w_warning_limit, cfg.delta_w_critical_limit),
            ("STATE_LIMIT_CHECK.delta_q", inputs.estimated_delta_q,
             cfg.delta_q_warning_limit, cfg.delta_q_critical_limit),
        ]
        for rule_name, value, warn_spec, crit_spec in checks:
            observed = abs(value)
            triggered_spec: Optional[ThresholdSpec] = None
            if observed >= crit_spec.value:
                triggered_spec = crit_spec
            elif observed >= warn_spec.value:
                triggered_spec = warn_spec

            diagnostics[rule_name] = {
                "observed": observed,
                "warning_limit": warn_spec.value,
                "critical_limit": crit_spec.value,
                "unit": warn_spec.unit,
                "provenance": warn_spec.provenance.value,
                "passed": triggered_spec is None,
            }
            if triggered_spec is not None:
                fault_flags.append(FaultFlag(
                    rule=rule_name,
                    severity=triggered_spec.severity,
                    provenance=triggered_spec.provenance,
                    category=FaultCategory.DYNAMICS,
                    observed=observed,
                    limit=triggered_spec.value,
                    unit=triggered_spec.unit,
                    message=(
                        f"{rule_name}: |{observed:.6g}| {warn_spec.unit} >= "
                        f"{triggered_spec.severity.value} limit {triggered_spec.value:.6g} {warn_spec.unit}"
                    ),
                ))

    @staticmethod
    def _residual_check(
        inputs: HealthCheckInput,
        cfg: HealthMonitorConfig,
        diagnostics: Dict[str, Dict[str, object]],
        fault_flags: List[FaultFlag],
    ) -> None:
        """Rule: SENSOR_RESIDUAL_CHECK.

        Uses a supplied residual if present. Otherwise, if both a
        measurement and an estimate are available for a channel, falls
        back to the clearly-defined scalar difference
        residual = measurement - estimate. Never recomputes estimator
        equations.
        """
        channels = [
            ("SENSOR_RESIDUAL_CHECK.delta_w", inputs.residual_delta_w,
             inputs.measured_delta_w, inputs.estimated_delta_w,
             cfg.residual_delta_w_warning_limit, cfg.residual_delta_w_critical_limit),
            ("SENSOR_RESIDUAL_CHECK.delta_q", inputs.residual_delta_q,
             inputs.measured_delta_q, inputs.estimated_delta_q,
             cfg.residual_delta_q_warning_limit, cfg.residual_delta_q_critical_limit),
        ]
        for rule_name, supplied_residual, measured, estimated, warn_spec, crit_spec in channels:
            if supplied_residual is not None:
                residual = supplied_residual
                source = "supplied"
            elif measured is not None:
                residual = measured - estimated
                source = "derived (measurement - estimate)"
            else:
                diagnostics[rule_name] = {
                    "passed": None,
                    "provenance": warn_spec.provenance.value,
                    "note": "no residual or measurement supplied - rule skipped",
                }
                continue

            observed = abs(residual)
            triggered_spec: Optional[ThresholdSpec] = None
            if observed >= crit_spec.value:
                triggered_spec = crit_spec
            elif observed >= warn_spec.value:
                triggered_spec = warn_spec

            diagnostics[rule_name] = {
                "observed": observed,
                "residual_source": source,
                "warning_limit": warn_spec.value,
                "critical_limit": crit_spec.value,
                "unit": warn_spec.unit,
                "provenance": warn_spec.provenance.value,
                "passed": triggered_spec is None,
            }
            if triggered_spec is not None:
                fault_flags.append(FaultFlag(
                    rule=rule_name,
                    severity=triggered_spec.severity,
                    provenance=triggered_spec.provenance,
                    category=FaultCategory.SENSOR,
                    observed=observed,
                    limit=triggered_spec.value,
                    unit=triggered_spec.unit,
                    message=(
                        f"{rule_name}: |{observed:.6g}| {warn_spec.unit} >= "
                        f"{triggered_spec.severity.value} limit {triggered_spec.value:.6g} {warn_spec.unit} "
                        f"({source})"
                    ),
                ))

    @staticmethod
    def _propulsion_status_check(
        inputs: HealthCheckInput,
        cfg: HealthMonitorConfig,
        diagnostics: Dict[str, Dict[str, object]],
        fault_flags: List[FaultFlag],
    ) -> None:
        """Rule: PROPULSION_STATUS_CHECK.

        MODEL_NOT_IDENTIFIED is surfaced as a model-status diagnostic, not
        classified as an aircraft/engine fault.
        """
        status = inputs.propulsion_status
        if status is None:
            diagnostics["PROPULSION_STATUS_CHECK"] = {"passed": None, "note": "no propulsion status supplied"}
            return

        if status in cfg.propulsion_ok_statuses:
            diagnostics["PROPULSION_STATUS_CHECK"] = {
                "passed": True, "observed_status": status, "classification": "NOMINAL",
            }
            return

        if status in cfg.propulsion_model_status_statuses:
            # Explicitly NOT a fault: a model-identification limitation.
            diagnostics["PROPULSION_STATUS_CHECK"] = {
                "passed": True,
                "observed_status": status,
                "classification": "MODEL_STATUS_LIMITATION",
                "note": "Propulsion numerical model is not fully identified. "
                        "This is a model-status condition, not a claimed engine failure.",
            }
            return

        if status in cfg.propulsion_fault_statuses:
            diagnostics["PROPULSION_STATUS_CHECK"] = {
                "passed": False, "observed_status": status, "classification": "FAULT",
            }
            fault_flags.append(FaultFlag(
                rule="PROPULSION_STATUS_CHECK",
                severity=Severity.CRITICAL,
                provenance=Provenance.PROJECT_DEFINED_RULE,
                category=FaultCategory.PROPULSION,
                observed=None,
                limit=None,
                unit="n/a",
                message=f"Propulsion status '{status}' is in the project-defined fault-status set.",
            ))
            return

        # Unrecognized status: flagged as a caution, not asserted as any
        # specific physical failure mode.
        diagnostics["PROPULSION_STATUS_CHECK"] = {
            "passed": False, "observed_status": status, "classification": "UNRECOGNIZED",
        }
        fault_flags.append(FaultFlag(
            rule="PROPULSION_STATUS_CHECK",
            severity=Severity.WARNING,
            provenance=Provenance.PROJECT_DEFINED_RULE,
            category=FaultCategory.PROPULSION,
            observed=None,
            limit=None,
            unit="n/a",
            message=f"Propulsion status '{status}' is not recognized (not in ok/model-status/fault sets).",
        ))

    @staticmethod
    def _simulation_status_check(
        inputs: HealthCheckInput,
        cfg: HealthMonitorConfig,
        diagnostics: Dict[str, Dict[str, object]],
        fault_flags: List[FaultFlag],
    ) -> None:
        """Rule: SIMULATION_STATUS_CHECK. Detect divergence/invalid simulation status."""
        status = inputs.simulation_status
        if status is None:
            diagnostics["SIMULATION_STATUS_CHECK"] = {"passed": None, "note": "no simulation status supplied"}
            return

        if status in cfg.simulation_ok_statuses:
            diagnostics["SIMULATION_STATUS_CHECK"] = {
                "passed": True, "observed_status": status, "classification": "NOMINAL",
            }
            return

        if status in cfg.simulation_fault_statuses:
            diagnostics["SIMULATION_STATUS_CHECK"] = {
                "passed": False, "observed_status": status, "classification": "FAULT",
            }
            fault_flags.append(FaultFlag(
                rule="SIMULATION_STATUS_CHECK",
                severity=Severity.CRITICAL,
                provenance=Provenance.PROJECT_DEFINED_RULE,
                category=FaultCategory.SIMULATION,
                observed=None,
                limit=None,
                unit="n/a",
                message=f"Simulation status '{status}' indicates divergence/invalid simulation.",
            ))
            return

        diagnostics["SIMULATION_STATUS_CHECK"] = {
            "passed": False, "observed_status": status, "classification": "UNRECOGNIZED",
        }
        fault_flags.append(FaultFlag(
            rule="SIMULATION_STATUS_CHECK",
            severity=Severity.WARNING,
            provenance=Provenance.PROJECT_DEFINED_RULE,
            category=FaultCategory.SIMULATION,
            observed=None,
            limit=None,
            unit="n/a",
            message=f"Simulation status '{status}' is not recognized (not in ok/fault sets).",
        ))

    @classmethod
    def _apply_persistence(
        cls,
        fault_flags: List[FaultFlag],
        cfg: HealthMonitorConfig,
        recent_critical_rules: Optional[Sequence[FrozenSet[str]]],
        diagnostics: Dict[str, Dict[str, object]],
    ) -> List[FaultFlag]:
        """Optional PROJECT_DEFINED consecutive-sample persistence rule,
        scoped to SOFT rules only (see ``_PERSISTENCE_EXEMPT_RULES``).

        With ``persistence_samples == 1`` (default) this is a no-op:
        every check is evaluated instantaneously, exactly as specified.
        With ``persistence_samples > 1``, a CRITICAL flag on a SOFT rule
        (a continuous threshold crossing: STATE_LIMIT_CHECK.*,
        SENSOR_RESIDUAL_CHECK.*) is only kept as CRITICAL if the same
        rule name was also CRITICAL on each of the previous
        ``persistence_samples - 1`` samples (supplied explicitly by the
        caller); otherwise it is downgraded to WARNING so a single
        outlier does not immediately read as FAULT. CRITICAL flags on
        HARD rules (structural checks or an explicit fault/unrecognized
        status string) are never downgraded or delayed by this rule --
        they are unambiguous the moment they occur.
        """
        if cfg.persistence_samples <= 1:
            return fault_flags

        history = recent_critical_rules or ()
        required_history = cfg.persistence_samples - 1
        diagnostics["PERSISTENCE_RULE"] = {
            "provenance": Provenance.PROJECT_DEFINED_RULE.value,
            "persistence_samples": cfg.persistence_samples,
            "history_samples_provided": len(history),
            "exempt_rules": sorted(cls._PERSISTENCE_EXEMPT_RULES),
        }

        downgraded: List[FaultFlag] = []
        for flag in fault_flags:
            if flag.severity is not Severity.CRITICAL:
                downgraded.append(flag)
                continue
            if flag.rule in cls._PERSISTENCE_EXEMPT_RULES:
                # Hard rule: always reported immediately, unmodified.
                downgraded.append(flag)
                continue
            if len(history) < required_history:
                persisted = False
            else:
                window = history[-required_history:] if required_history else ()
                persisted = all(flag.rule in sample for sample in window)
            if persisted:
                downgraded.append(flag)
            else:
                downgraded.append(FaultFlag(
                    rule=flag.rule,
                    severity=Severity.WARNING,
                    provenance=flag.provenance,
                    category=flag.category,
                    observed=flag.observed,
                    limit=flag.limit,
                    unit=flag.unit,
                    message=flag.message + " [transient: persistence threshold not yet met]",
                ))
        return downgraded

    @staticmethod
    def _classify(fault_flags: Sequence[FaultFlag]) -> Tuple[HealthStatus, Severity]:
        """Deterministic overall classification from the collected flags."""
        if not fault_flags:
            return HealthStatus.HEALTHY, Severity.HEALTHY
        worst = max(fault_flags, key=lambda f: f.severity.rank)
        if worst.severity is Severity.CRITICAL:
            return HealthStatus.FAULT, Severity.CRITICAL
        return HealthStatus.WARNING, Severity.WARNING