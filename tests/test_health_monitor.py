"""
tests/test_health_monitor.py

Unit and light integration tests for digital_twin/health_monitor.py.

All numeric thresholds used directly in these tests (as opposed to the
module's own PROJECT_DEFINED defaults) are constructed with
Provenance.TEST_ONLY_LIMIT and are clearly local to this file. They must
never be copied into production config.
"""

from __future__ import annotations

import copy
import math

import pytest

from digital_twin.health_monitor import (
    Coverage,
    FaultCategory,
    FaultFlag,
    HealthCheckInput,
    HealthMonitor,
    HealthMonitorConfig,
    HealthMonitorResult,
    HealthStatus,
    Provenance,
    Severity,
    ThresholdSpec,
)


# ======================================================================
# Helpers
# ======================================================================

def make_test_only_config() -> HealthMonitorConfig:
    """A config built entirely from TEST_ONLY_LIMIT thresholds.

    Using small, easy-to-hand-verify numbers keeps the analytic test
    cases simple, and keeps TEST_ONLY numbers clearly separated from the
    module's PROJECT_DEFINED production defaults.
    """
    return HealthMonitorConfig(
        delta_w_warning_limit=ThresholdSpec(
            value=1.0, unit="m/s", severity=Severity.WARNING, provenance=Provenance.TEST_ONLY_LIMIT,
        ),
        delta_w_critical_limit=ThresholdSpec(
            value=2.0, unit="m/s", severity=Severity.CRITICAL, provenance=Provenance.TEST_ONLY_LIMIT,
        ),
        delta_q_warning_limit=ThresholdSpec(
            value=0.20, unit="rad/s", severity=Severity.WARNING, provenance=Provenance.TEST_ONLY_LIMIT,
        ),
        delta_q_critical_limit=ThresholdSpec(
            value=0.35, unit="rad/s", severity=Severity.CRITICAL, provenance=Provenance.TEST_ONLY_LIMIT,
        ),
        residual_delta_w_warning_limit=ThresholdSpec(
            value=0.5, unit="m/s", severity=Severity.WARNING, provenance=Provenance.TEST_ONLY_LIMIT,
        ),
        residual_delta_w_critical_limit=ThresholdSpec(
            value=1.0, unit="m/s", severity=Severity.CRITICAL, provenance=Provenance.TEST_ONLY_LIMIT,
        ),
        residual_delta_q_warning_limit=ThresholdSpec(
            value=0.02, unit="rad/s", severity=Severity.WARNING, provenance=Provenance.TEST_ONLY_LIMIT,
        ),
        residual_delta_q_critical_limit=ThresholdSpec(
            value=0.05, unit="rad/s", severity=Severity.CRITICAL, provenance=Provenance.TEST_ONLY_LIMIT,
        ),
    )


@pytest.fixture
def monitor() -> HealthMonitor:
    return HealthMonitor(config=make_test_only_config())


def nominal_input() -> HealthCheckInput:
    return HealthCheckInput(
        estimated_delta_w=0.05,
        estimated_delta_q=0.01,
        measured_delta_w=0.06,
        measured_delta_q=0.015,
        covariance=[[0.01, 0.0], [0.0, 0.001]],
        propulsion_status="NOMINAL",
        simulation_status="OK",
    )


# ======================================================================
# 1. Healthy nominal input
# ======================================================================

def test_healthy_nominal_input(monitor: HealthMonitor) -> None:
    result = monitor.evaluate(nominal_input())
    assert result.overall_status == HealthStatus.HEALTHY
    assert result.severity == Severity.HEALTHY
    assert result.fault_flags == ()


# ======================================================================
# 2. NaN/Inf input detection
# ======================================================================

@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_estimated_delta_w_is_invalid(monitor: HealthMonitor, bad_value: float) -> None:
    inputs = HealthCheckInput(estimated_delta_w=bad_value, estimated_delta_q=0.01)
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.INVALID
    assert result.severity == Severity.CRITICAL
    assert any(f.rule == "FINITE_VALUE_CHECK" for f in result.fault_flags)


def test_non_finite_estimated_delta_q_is_invalid(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(estimated_delta_w=0.0, estimated_delta_q=float("nan"))
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.INVALID
    assert any(f.rule == "FINITE_VALUE_CHECK" for f in result.fault_flags)


def test_non_finite_measurement_is_invalid(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0, measured_delta_w=float("inf"),
    )
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.INVALID


# ======================================================================
# 3. delta_w limit violation
# ======================================================================

def test_delta_w_warning_violation(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(estimated_delta_w=1.5, estimated_delta_q=0.0)
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.WARNING
    flag = next(f for f in result.fault_flags if f.rule == "STATE_LIMIT_CHECK.delta_w")
    assert flag.severity == Severity.WARNING
    assert flag.provenance == Provenance.TEST_ONLY_LIMIT
    assert flag.limit == pytest.approx(1.0)


def test_delta_w_critical_violation(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(estimated_delta_w=2.5, estimated_delta_q=0.0)
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.FAULT
    assert result.severity == Severity.CRITICAL
    flag = next(f for f in result.fault_flags if f.rule == "STATE_LIMIT_CHECK.delta_w")
    assert flag.severity == Severity.CRITICAL


# ======================================================================
# 4. delta_q limit violation
# ======================================================================

def test_delta_q_warning_violation(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(estimated_delta_w=0.0, estimated_delta_q=0.25)
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.WARNING
    flag = next(f for f in result.fault_flags if f.rule == "STATE_LIMIT_CHECK.delta_q")
    assert flag.severity == Severity.WARNING


def test_delta_q_critical_violation(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(estimated_delta_w=0.0, estimated_delta_q=0.40)
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.FAULT
    flag = next(f for f in result.fault_flags if f.rule == "STATE_LIMIT_CHECK.delta_q")
    assert flag.severity == Severity.CRITICAL


# ======================================================================
# 5. Residual threshold violation
# ======================================================================

def test_residual_violation_supplied_directly(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0, residual_delta_q=0.03,
    )
    result = monitor.evaluate(inputs)
    flag = next(f for f in result.fault_flags if f.rule == "SENSOR_RESIDUAL_CHECK.delta_q")
    assert flag.severity == Severity.WARNING
    assert result.overall_status == HealthStatus.WARNING


def test_residual_derived_from_measurement_and_estimate(monitor: HealthMonitor) -> None:
    # residual = measured - estimated = 0.10 - 0.05 = 0.05 -> at the
    # TEST_ONLY critical limit (0.05) -> CRITICAL / FAULT.
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.05,
        measured_delta_q=0.10,
    )
    result = monitor.evaluate(inputs)
    flag = next(f for f in result.fault_flags if f.rule == "SENSOR_RESIDUAL_CHECK.delta_q")
    assert flag.observed == pytest.approx(0.05)
    assert flag.severity == Severity.CRITICAL
    assert result.overall_status == HealthStatus.FAULT
    assert result.diagnostics["SENSOR_RESIDUAL_CHECK.delta_q"]["residual_source"].startswith("derived")


def test_no_residual_and_no_measurement_skips_rule(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(estimated_delta_w=0.0, estimated_delta_q=0.0)
    result = monitor.evaluate(inputs)
    assert result.diagnostics["SENSOR_RESIDUAL_CHECK.delta_q"]["passed"] is None
    assert result.overall_status == HealthStatus.HEALTHY


# ======================================================================
# 6 & 7. Covariance valid / invalid
# ======================================================================

def test_valid_covariance_accepted(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        covariance=[[0.02, 0.001], [0.001, 0.005]],
    )
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.HEALTHY
    assert result.diagnostics["ESTIMATOR_COVARIANCE_CHECK"]["passed"] is True


def test_non_finite_covariance_is_invalid(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        covariance=[[float("nan"), 0.0], [0.0, 0.01]],
    )
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.INVALID
    assert any(f.rule == "ESTIMATOR_COVARIANCE_CHECK" for f in result.fault_flags)


def test_non_positive_semi_definite_covariance_is_fault_not_invalid(monitor: HealthMonitor) -> None:
    # Finite, correctly shaped, but not PSD (negative diagonal).
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        covariance=[[-1.0, 0.0], [0.0, 0.01]],
    )
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.FAULT
    assert result.overall_status != HealthStatus.INVALID
    flag = next(f for f in result.fault_flags if f.rule == "ESTIMATOR_COVARIANCE_CHECK")
    assert flag.severity == Severity.CRITICAL


def test_wrong_shape_covariance_is_invalid(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        covariance=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.INVALID


# ======================================================================
# 8. Propulsion MODEL_NOT_IDENTIFIED handled as model-status, not fault
# ======================================================================

def test_propulsion_model_not_identified_is_not_a_fault(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        propulsion_status="MODEL_NOT_IDENTIFIED",
    )
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.HEALTHY
    assert not any(f.rule == "PROPULSION_STATUS_CHECK" for f in result.fault_flags)
    diag = result.diagnostics["PROPULSION_STATUS_CHECK"]
    assert diag["classification"] == "MODEL_STATUS_LIMITATION"
    assert diag["passed"] is True


def test_propulsion_unrecognized_status_is_warning(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        propulsion_status="SOMETHING_UNEXPECTED",
    )
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.WARNING
    flag = next(f for f in result.fault_flags if f.rule == "PROPULSION_STATUS_CHECK")
    assert flag.severity == Severity.WARNING


def test_propulsion_explicit_fault_status(monitor: HealthMonitor) -> None:
    cfg = make_test_only_config()
    cfg = HealthMonitorConfig(
        **{**cfg.__dict__, "propulsion_fault_statuses": frozenset({"ENGINE_OUT_TEST_ONLY"})}
    )
    m = HealthMonitor(config=cfg)
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        propulsion_status="ENGINE_OUT_TEST_ONLY",
    )
    result = m.evaluate(inputs)
    assert result.overall_status == HealthStatus.FAULT
    flag = next(f for f in result.fault_flags if f.rule == "PROPULSION_STATUS_CHECK")
    assert flag.severity == Severity.CRITICAL


# ======================================================================
# 9. Simulation divergence/invalid status detected
# ======================================================================

def test_simulation_diverged_is_fault(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        simulation_status="DIVERGED",
    )
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.FAULT
    flag = next(f for f in result.fault_flags if f.rule == "SIMULATION_STATUS_CHECK")
    assert flag.severity == Severity.CRITICAL


def test_simulation_ok_status(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        simulation_status="RUNNING",
    )
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.HEALTHY


# ======================================================================
# 10. Severity classification is deterministic
# ======================================================================

def test_severity_classification_is_deterministic(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(estimated_delta_w=2.5, estimated_delta_q=0.40)
    results = [monitor.evaluate(inputs) for _ in range(25)]
    statuses = {r.overall_status for r in results}
    severities = {r.severity for r in results}
    assert statuses == {HealthStatus.FAULT}
    assert severities == {Severity.CRITICAL}


# ======================================================================
# 11. Threshold provenance is preserved
# ======================================================================

def test_threshold_provenance_is_preserved(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(estimated_delta_w=1.5, estimated_delta_q=0.0)
    result = monitor.evaluate(inputs)
    flag = next(f for f in result.fault_flags if f.rule == "STATE_LIMIT_CHECK.delta_w")
    assert flag.provenance == Provenance.TEST_ONLY_LIMIT
    assert result.diagnostics["STATE_LIMIT_CHECK.delta_w"]["provenance"] == Provenance.TEST_ONLY_LIMIT.value


def test_project_defined_default_config_has_no_nasa_verified_limits() -> None:
    """None of the module's shipped defaults may claim NASA provenance
    unless a real NASA source is later verified and cited (see module
    docstring)."""
    cfg = HealthMonitorConfig()
    thresholds = [
        cfg.delta_w_warning_limit, cfg.delta_w_critical_limit,
        cfg.delta_q_warning_limit, cfg.delta_q_critical_limit,
        cfg.residual_delta_w_warning_limit, cfg.residual_delta_w_critical_limit,
        cfg.residual_delta_q_warning_limit, cfg.residual_delta_q_critical_limit,
    ]
    for t in thresholds:
        assert t.provenance == Provenance.PROJECT_DEFINED_LIMIT
        assert t.provenance != Provenance.NASA_VERIFIED_LIMIT


# ======================================================================
# 12. TEST_ONLY thresholds are clearly labelled
# ======================================================================

def test_test_only_thresholds_are_labelled(monitor: HealthMonitor) -> None:
    cfg = monitor.config
    assert cfg.delta_w_warning_limit.provenance == Provenance.TEST_ONLY_LIMIT
    assert cfg.delta_q_critical_limit.provenance == Provenance.TEST_ONLY_LIMIT
    assert cfg.residual_delta_q_warning_limit.provenance == Provenance.TEST_ONLY_LIMIT


def test_threshold_spec_rejects_non_finite_value() -> None:
    with pytest.raises(ValueError):
        ThresholdSpec(value=float("nan"), provenance=Provenance.TEST_ONLY_LIMIT, severity=Severity.WARNING)


# ======================================================================
# 13. Repeated identical inputs produce identical outputs
# ======================================================================

def test_repeated_identical_inputs_produce_identical_outputs(monitor: HealthMonitor) -> None:
    inputs = nominal_input()
    r1 = monitor.evaluate(inputs)
    r2 = monitor.evaluate(inputs)
    assert r1 == r2


def test_repeated_faulted_inputs_produce_identical_outputs(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(estimated_delta_w=5.0, estimated_delta_q=0.40)
    r1 = monitor.evaluate(inputs)
    r2 = monitor.evaluate(inputs)
    assert r1 == r2


# ======================================================================
# 14 & 15. No state / estimator mutation occurs
# ======================================================================

def test_no_mutation_of_frozen_input(monitor: HealthMonitor) -> None:
    inputs = nominal_input()
    before = copy.deepcopy(inputs)
    monitor.evaluate(inputs)
    assert inputs == before  # HealthCheckInput is frozen; this also
    # verifies no in-place mutation of any mutable field (e.g. covariance).


def test_no_mutation_of_mutable_covariance_list(monitor: HealthMonitor) -> None:
    cov = [[0.02, 0.001], [0.001, 0.005]]
    inputs = HealthCheckInput(estimated_delta_w=0.0, estimated_delta_q=0.0, covariance=cov)
    before = copy.deepcopy(cov)
    monitor.evaluate(inputs)
    assert cov == before


def test_no_mutation_of_config(monitor: HealthMonitor) -> None:
    before = copy.deepcopy(monitor.config)
    monitor.evaluate(HealthCheckInput(estimated_delta_w=5.0, estimated_delta_q=0.5))
    assert monitor.config == before


# ======================================================================
# 16. No control command is generated
# ======================================================================

def test_result_contains_no_control_command_surface(monitor: HealthMonitor) -> None:
    result = monitor.evaluate(HealthCheckInput(estimated_delta_w=5.0, estimated_delta_q=0.5))
    forbidden_attrs = {"command", "control", "actuator_command", "elevator", "throttle"}
    result_fields = set(HealthMonitorResult.__dataclass_fields__.keys())
    assert result_fields.isdisjoint(forbidden_attrs)


def test_evaluate_has_no_side_effects_beyond_return_value(monitor: HealthMonitor) -> None:
    # A pure diagnostic call: calling it repeatedly must not accumulate
    # any internal state on the monitor itself.
    monitor_attrs_before = vars(monitor).copy()
    for _ in range(10):
        monitor.evaluate(HealthCheckInput(estimated_delta_w=5.0, estimated_delta_q=0.5))
    assert vars(monitor) == monitor_attrs_before


# ======================================================================
# Analytic (hand-calculated) case from the task specification
# ======================================================================

def test_analytic_residual_case_from_spec() -> None:
    """measured_q=0.10, estimated_q=0.05 -> residual=0.05, threshold=0.02
    (all TEST_ONLY) -> expect WARNING or FAULT, per spec."""
    cfg = HealthMonitorConfig(
        residual_delta_q_warning_limit=ThresholdSpec(
            value=0.02, unit="rad/s", severity=Severity.WARNING, provenance=Provenance.TEST_ONLY_LIMIT,
        ),
        residual_delta_q_critical_limit=ThresholdSpec(
            value=0.10, unit="rad/s", severity=Severity.CRITICAL, provenance=Provenance.TEST_ONLY_LIMIT,
        ),
    )
    m = HealthMonitor(config=cfg)
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.05, measured_delta_q=0.10,
    )
    result = m.evaluate(inputs)
    assert result.overall_status in (HealthStatus.WARNING, HealthStatus.FAULT)
    flag = next(f for f in result.fault_flags if f.rule == "SENSOR_RESIDUAL_CHECK.delta_q")
    assert flag.observed == pytest.approx(0.05)
    assert flag.severity == Severity.WARNING  # 0.05 >= 0.02 warn, < 0.10 crit
    assert result.overall_status == HealthStatus.WARNING


# ======================================================================
# Optional PROJECT_DEFINED persistence feature (SOFT rules)
# ======================================================================

def test_persistence_downgrades_single_outlier_to_warning() -> None:
    cfg = make_test_only_config()
    cfg = HealthMonitorConfig(**{**cfg.__dict__, "persistence_samples": 3})
    m = HealthMonitor(config=cfg)
    inputs = HealthCheckInput(estimated_delta_w=5.0, estimated_delta_q=0.0)  # critical delta_w

    # No history yet -> not persisted -> downgraded to WARNING.
    result = m.evaluate(inputs, recent_critical_rules=())
    assert result.overall_status == HealthStatus.WARNING


def test_persistence_confirms_after_enough_history() -> None:
    cfg = make_test_only_config()
    cfg = HealthMonitorConfig(**{**cfg.__dict__, "persistence_samples": 3})
    m = HealthMonitor(config=cfg)
    inputs = HealthCheckInput(estimated_delta_w=5.0, estimated_delta_q=0.0)

    history = (
        frozenset({"STATE_LIMIT_CHECK.delta_w"}),
        frozenset({"STATE_LIMIT_CHECK.delta_w"}),
    )
    result = m.evaluate(inputs, recent_critical_rules=history)
    assert result.overall_status == HealthStatus.FAULT


# ======================================================================
# NEW: hard faults are never delayed/downgraded by persistence
# ======================================================================

def test_persistence_never_downgrades_finite_value_check() -> None:
    # FINITE_VALUE_CHECK short-circuits to INVALID before persistence is
    # even reached, regardless of persistence_samples.
    cfg = make_test_only_config()
    cfg = HealthMonitorConfig(**{**cfg.__dict__, "persistence_samples": 5})
    m = HealthMonitor(config=cfg)
    result = m.evaluate(
        HealthCheckInput(estimated_delta_w=float("nan"), estimated_delta_q=0.0),
        recent_critical_rules=(),
    )
    assert result.overall_status == HealthStatus.INVALID
    assert result.severity == Severity.CRITICAL


def test_persistence_never_downgrades_propulsion_fault_status() -> None:
    cfg = make_test_only_config()
    cfg = HealthMonitorConfig(**{
        **cfg.__dict__,
        "persistence_samples": 5,
        "propulsion_fault_statuses": frozenset({"ENGINE_OUT_TEST_ONLY"}),
    })
    m = HealthMonitor(config=cfg)
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        propulsion_status="ENGINE_OUT_TEST_ONLY",
    )
    # No history at all -- a SOFT critical flag would be downgraded here,
    # but PROPULSION_STATUS_CHECK is a HARD rule and must report FAULT
    # immediately regardless.
    result = m.evaluate(inputs, recent_critical_rules=())
    assert result.overall_status == HealthStatus.FAULT
    flag = next(f for f in result.fault_flags if f.rule == "PROPULSION_STATUS_CHECK")
    assert flag.severity == Severity.CRITICAL
    assert "transient" not in flag.message


def test_persistence_never_downgrades_simulation_fault_status() -> None:
    cfg = make_test_only_config()
    cfg = HealthMonitorConfig(**{**cfg.__dict__, "persistence_samples": 5})
    m = HealthMonitor(config=cfg)
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        simulation_status="DIVERGED",
    )
    result = m.evaluate(inputs, recent_critical_rules=())
    assert result.overall_status == HealthStatus.FAULT
    flag = next(f for f in result.fault_flags if f.rule == "SIMULATION_STATUS_CHECK")
    assert flag.severity == Severity.CRITICAL


def test_persistence_never_downgrades_covariance_psd_fault() -> None:
    cfg = make_test_only_config()
    cfg = HealthMonitorConfig(**{**cfg.__dict__, "persistence_samples": 5})
    m = HealthMonitor(config=cfg)
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        covariance=[[-1.0, 0.0], [0.0, 0.01]],
    )
    result = m.evaluate(inputs, recent_critical_rules=())
    assert result.overall_status == HealthStatus.FAULT
    flag = next(f for f in result.fault_flags if f.rule == "ESTIMATOR_COVARIANCE_CHECK")
    assert flag.severity == Severity.CRITICAL


# ======================================================================
# NEW: coverage reporting
# ======================================================================

def test_coverage_full_when_every_optional_input_supplied(monitor: HealthMonitor) -> None:
    result = monitor.evaluate(nominal_input())
    assert result.coverage == Coverage.FULL
    assert result.skipped_rules == ()


def test_coverage_partial_when_optional_inputs_missing(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(estimated_delta_w=0.0, estimated_delta_q=0.0)
    result = monitor.evaluate(inputs)
    assert result.coverage == Coverage.PARTIAL
    assert "SENSOR_RESIDUAL_CHECK.delta_w" in result.skipped_rules
    assert "SENSOR_RESIDUAL_CHECK.delta_q" in result.skipped_rules
    assert "ESTIMATOR_COVARIANCE_CHECK" in result.skipped_rules
    assert "PROPULSION_STATUS_CHECK" in result.skipped_rules
    assert "SIMULATION_STATUS_CHECK" in result.skipped_rules
    # A PARTIAL, under-covered evaluation can still be HEALTHY -- coverage
    # and status are orthogonal.
    assert result.overall_status == HealthStatus.HEALTHY


def test_coverage_partial_result_is_still_invalid_when_input_is_bad(monitor: HealthMonitor) -> None:
    # Coverage bookkeeping must not interfere with an INVALID short-circuit.
    inputs = HealthCheckInput(estimated_delta_w=float("nan"), estimated_delta_q=0.0)
    result = monitor.evaluate(inputs)
    assert result.overall_status == HealthStatus.INVALID
    assert result.coverage in (Coverage.FULL, Coverage.PARTIAL)  # just must not raise


# ======================================================================
# NEW: fault categorization (root-cause grouping)
# ======================================================================

def test_state_limit_flags_are_categorized_dynamics(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(estimated_delta_w=5.0, estimated_delta_q=0.0)
    result = monitor.evaluate(inputs)
    flag = next(f for f in result.fault_flags if f.rule == "STATE_LIMIT_CHECK.delta_w")
    assert flag.category == FaultCategory.DYNAMICS


def test_residual_flags_are_categorized_sensor(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(estimated_delta_w=0.0, estimated_delta_q=0.0, residual_delta_q=0.03)
    result = monitor.evaluate(inputs)
    flag = next(f for f in result.fault_flags if f.rule == "SENSOR_RESIDUAL_CHECK.delta_q")
    assert flag.category == FaultCategory.SENSOR


def test_covariance_flags_are_categorized_estimator(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        covariance=[[-1.0, 0.0], [0.0, 0.01]],
    )
    result = monitor.evaluate(inputs)
    flag = next(f for f in result.fault_flags if f.rule == "ESTIMATOR_COVARIANCE_CHECK")
    assert flag.category == FaultCategory.ESTIMATOR


def test_propulsion_flags_are_categorized_propulsion(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        propulsion_status="SOMETHING_UNEXPECTED",
    )
    result = monitor.evaluate(inputs)
    flag = next(f for f in result.fault_flags if f.rule == "PROPULSION_STATUS_CHECK")
    assert flag.category == FaultCategory.PROPULSION


def test_simulation_flags_are_categorized_simulation(monitor: HealthMonitor) -> None:
    inputs = HealthCheckInput(
        estimated_delta_w=0.0, estimated_delta_q=0.0,
        simulation_status="DIVERGED",
    )
    result = monitor.evaluate(inputs)
    flag = next(f for f in result.fault_flags if f.rule == "SIMULATION_STATUS_CHECK")
    assert flag.category == FaultCategory.SIMULATION


def test_finite_value_check_categorizes_by_field_origin(monitor: HealthMonitor) -> None:
    sensor_side = monitor.evaluate(
        HealthCheckInput(estimated_delta_w=0.0, estimated_delta_q=0.0, measured_delta_w=float("inf"))
    )
    sensor_flag = next(f for f in sensor_side.fault_flags if f.rule == "FINITE_VALUE_CHECK")
    assert sensor_flag.category == FaultCategory.SENSOR

    estimator_side = monitor.evaluate(
        HealthCheckInput(estimated_delta_w=float("nan"), estimated_delta_q=0.0)
    )
    estimator_flag = next(f for f in estimator_side.fault_flags if f.rule == "FINITE_VALUE_CHECK")
    assert estimator_flag.category == FaultCategory.ESTIMATOR


# ======================================================================
# Integration ("wiring") test: SensorModel -> Estimator -> HealthMonitor
#
# The real sensors/sensor_model.py and digital_twin/estimator.py sources
# were not available in this task's context, so this test uses minimal
# local stand-ins with a plausible, typical output shape (a measurement
# dict and an estimate object exposing delta_w/delta_q). It proves that
# HealthCheckInput can be constructed from that shape and fed through
# HealthMonitor -- it does NOT validate the real project's estimator or
# sensor noise model, and it is not a NASA validation of any kind.
# ======================================================================

class _FakeSensorModel:
    """Stand-in for sensors/sensor_model.py's public shape."""

    def measure(self, true_delta_w: float, true_delta_q: float) -> dict:
        # Deterministic "measurement" for test wiring purposes only.
        return {"delta_w": true_delta_w + 0.01, "delta_q": true_delta_q - 0.005}


class _FakeEstimate:
    def __init__(self, delta_w: float, delta_q: float) -> None:
        self.delta_w = delta_w
        self.delta_q = delta_q


class _FakeEstimator:
    """Stand-in for digital_twin/estimator.py's public shape."""

    def estimate(self, measurement: dict) -> _FakeEstimate:
        # A trivial pass-through "estimate" for test wiring purposes only.
        return _FakeEstimate(delta_w=measurement["delta_w"], delta_q=measurement["delta_q"])


def test_integration_sensor_estimator_health_monitor_wiring() -> None:
    sensor_model = _FakeSensorModel()
    estimator = _FakeEstimator()
    monitor = HealthMonitor(config=make_test_only_config())

    measurement = sensor_model.measure(true_delta_w=0.02, true_delta_q=0.01)
    estimate = estimator.estimate(measurement)

    inputs = HealthCheckInput(
        estimated_delta_w=estimate.delta_w,
        estimated_delta_q=estimate.delta_q,
        measured_delta_w=measurement["delta_w"],
        measured_delta_q=measurement["delta_q"],
    )
    result = monitor.evaluate(inputs)

    assert isinstance(result, HealthMonitorResult)
    assert result.overall_status == HealthStatus.HEALTHY
    # This proves module-to-module wiring compatibility only, not NASA
    # validation of the underlying physical model.


def test_integration_with_real_modules() -> None:
    from sensors.sensor_model import SensorModel
    from digital_twin.estimator import (
        KalmanStateEstimator,
        default_process_noise,
        default_initial_covariance,
        identity_transition_matrix,
        identity_measurement_matrix,
    )

    sensor_model = SensorModel()
    estimator = KalmanStateEstimator(
        F=identity_transition_matrix(),
        H=identity_measurement_matrix(),
        Q=default_process_noise(),
        R=[[1.0e-4, 0.0], [0.0, 1.0e-4]],
        x0=[0.0, 0.0],
        P0=default_initial_covariance(),
    )
    monitor = HealthMonitor(config=make_test_only_config())

    truth_delta_w = 0.02
    truth_delta_q = 0.01

    measurement = sensor_model.measure(delta_w=truth_delta_w, delta_q=truth_delta_q, perturbation_az=0.0)
    measured_w = measurement["measured"]["delta_w_mps"]
    measured_q = measurement["measured"]["delta_q_rads"]

    # Run a predict/update cycle
    estimate = estimator.step(z=[measured_w, measured_q], dt=0.02)

    inputs = HealthCheckInput(
        estimated_delta_w=estimate.delta_w_hat,
        estimated_delta_q=estimate.delta_q_hat,
        measured_delta_w=measured_w,
        measured_delta_q=measured_q,
        covariance=estimate.covariance,
    )
    result = monitor.evaluate(inputs)

    assert isinstance(result, HealthMonitorResult)
    # Status may vary based on noise, but the plumbing should work without error.
    assert result.overall_status in (HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.FAULT)


# ======================================================================
# NEW: infer_raw_critical_rules directly tested
# ======================================================================

def test_infer_raw_critical_rules_identifies_threshold_failures() -> None:
    diagnostics = {
        "STATE_LIMIT_CHECK.delta_w": {
            "passed": False,
            "observed": 5.0,
            "critical_limit": 2.0,
        },
        "STATE_LIMIT_CHECK.delta_q": {
            "passed": True,
            "observed": 0.1,
            "critical_limit": 0.5,
        },
        "SENSOR_RESIDUAL_CHECK.delta_w": {
            "passed": False,
            "observed": 1.5,
            "critical_limit": 1.0,
        },
        # A structural check that is NOT a numeric threshold
        "ESTIMATOR_COVARIANCE_CHECK": {
            "passed": False,
            "note": "covariance not PSD"
        }
    }
    raw_critical = HealthMonitor.infer_raw_critical_rules(diagnostics)
    assert raw_critical == frozenset({
        "STATE_LIMIT_CHECK.delta_w",
        "SENSOR_RESIDUAL_CHECK.delta_w"
    })