"""
tests/test_integration.py

Tests digital_twin.integration.DigitalTwinPipeline against the REAL project
interfaces: sensors.sensor_model.SensorModel, digital_twin.estimator
(KalmanStateEstimator / default_reduced_order_estimator), and
digital_twin.health_monitor.HealthMonitor. No fakes/mocks of these three
classes are used anywhere in this file -- every test exercises the actual
Kalman math, the actual noise model, and the actual rule-based health
checks. This is deliberate: it is what the project's own testing
requirements call for ("at least one integration test MUST use the actual
... modules; do not use fake implementations for every integration test").

This proves module interface COMPATIBILITY only. It is NOT NASA validation
of the estimator or the health-monitor thresholds (all of which remain
PROJECT_DEFINED, see digital_twin/estimator.py and
digital_twin/health_monitor.py).
"""

import copy
import math

import pytest

from sensors.sensor_model import SensorModel, SensorSuiteParameters, SensorChannelNoiseModel
from digital_twin.estimator import (
    KalmanStateEstimator,
    default_reduced_order_estimator,
    identity_transition_matrix,
)
from digital_twin.health_monitor import HealthMonitor, HealthMonitorConfig, HealthCheckInput
from digital_twin.integration import (
    DigitalTwinPipeline,
    IntegratedStepResult,
    IntegrationError,
    InvalidTruthSampleError,
    EstimatorFailureError,
    PIPELINE_PROVENANCE,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _zero_noise_sensor_suite() -> SensorSuiteParameters:
    """A deterministic (zero-noise, zero-bias) real SensorSuiteParameters,
    so tests can assert on exact numeric values without statistical noise
    getting in the way. Still the real SensorChannelNoiseModel/
    SensorSuiteParameters classes, not a fake."""
    return SensorSuiteParameters(
        w_sensor=SensorChannelNoiseModel(name="w_sensor", units="m/s", bias=0.0, noise_std=0.0),
        rate_gyro_q=SensorChannelNoiseModel(name="rate_gyro_q", units="rad/s", bias=0.0, noise_std=0.0),
        accelerometer_az=SensorChannelNoiseModel(name="accelerometer_az", units="m/s^2", bias=0.0, noise_std=0.0),
    )


def make_pipeline(persistence_samples: int = 1) -> DigitalTwinPipeline:
    """Builds a DigitalTwinPipeline out of REAL SensorModel,
    KalmanStateEstimator (via the real default_reduced_order_estimator
    baseline assembler), and HealthMonitor."""
    sensor_model = SensorModel(suite=_zero_noise_sensor_suite(), seed=1234)
    estimator = default_reduced_order_estimator(x0=(0.0, 0.0), sensor_suite=_zero_noise_sensor_suite())
    health_monitor = HealthMonitor(HealthMonitorConfig(persistence_samples=persistence_samples))
    return DigitalTwinPipeline(sensor_model=sensor_model, estimator=estimator, health_monitor=health_monitor)


def truth_row(time_s, delta_w_mps, delta_q_rads, perturbation_az_mps2=0.0,
              delta_e_rad=0.0, propulsion_model_status="MODEL_NOT_IDENTIFIED"):
    """Builds a dict matching simulation.simulator.Simulator's REAL
    history-row schema exactly (see Simulator._record_step)."""
    return {
        "time_s": time_s,
        "delta_w_mps": delta_w_mps,
        "delta_q_rads": delta_q_rads,
        "delta_e_rad": delta_e_rad,
        "delta_w_dot": 0.0,
        "delta_q_dot": 0.0,
        "perturbation_az_mps2": perturbation_az_mps2,
        "delta_alpha_rad": 0.0,
        "delta_CL": 0.0,
        "delta_Cm": 0.0,
        "delta_Lift_N": 0.0,
        "delta_PitchMom_Nm": 0.0,
        "total_thrust_N": float("nan"),
        "propulsion_model_status": propulsion_model_status,
    }


class RecordingEstimator(KalmanStateEstimator):
    """A REAL KalmanStateEstimator (same math, same validation, same
    public contract -- isinstance(obj, KalmanStateEstimator) is True)
    that additionally records the (dt, u) arguments each predict() call
    receives, purely for test assertions. Used instead of a mock so the
    actual Kalman prediction/update equations still run.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorded_predict_calls = []

    def predict(self, dt, u=0.0):
        self.recorded_predict_calls.append((dt, u))
        return super().predict(dt, u=u)


def make_recording_pipeline():
    sensor_model = SensorModel(suite=_zero_noise_sensor_suite(), seed=99)
    estimator = RecordingEstimator(
        F=identity_transition_matrix(),
        H=[[1.0, 0.0], [0.0, 1.0]],
        Q=[[1.0e-4, 0.0], [0.0, 1.0e-6]],
        R=[[1.0e-6, 0.0], [0.0, 1.0e-6]],
        x0=[0.0, 0.0],
        P0=[[1.0, 0.0], [0.0, 1.0e-2]],
    )
    health_monitor = HealthMonitor()
    pipeline = DigitalTwinPipeline(sensor_model=sensor_model, estimator=estimator, health_monitor=health_monitor)
    return pipeline, estimator


# ---------------------------------------------------------------------------
# 1. Pipeline construction
# ---------------------------------------------------------------------------


def test_pipeline_construction_with_real_dependencies():
    pipeline = make_pipeline()
    assert isinstance(pipeline.sensor_model, SensorModel)
    assert isinstance(pipeline.estimator, KalmanStateEstimator)
    assert isinstance(pipeline.health_monitor, HealthMonitor)


def test_pipeline_construction_rejects_wrong_types():
    real_sensor = SensorModel()
    real_estimator = default_reduced_order_estimator()
    real_health = HealthMonitor()

    with pytest.raises(TypeError):
        DigitalTwinPipeline(sensor_model="not a sensor model", estimator=real_estimator, health_monitor=real_health)
    with pytest.raises(TypeError):
        DigitalTwinPipeline(sensor_model=real_sensor, estimator="not an estimator", health_monitor=real_health)
    with pytest.raises(TypeError):
        DigitalTwinPipeline(sensor_model=real_sensor, estimator=real_estimator, health_monitor="not a health monitor")


# ---------------------------------------------------------------------------
# 2/3. One real end-to-end cycle + correct data mapping
# ---------------------------------------------------------------------------


def test_single_real_pipeline_step_and_data_mapping():
    pipeline = make_pipeline()
    row = truth_row(time_s=0.0, delta_w_mps=1.5, delta_q_rads=0.02, perturbation_az_mps2=0.3)

    result = pipeline.process_truth_sample(row)

    assert isinstance(result, IntegratedStepResult)
    # Sensor truth passthrough must exactly match the input truth (zero
    # noise/bias configured above), each under ITS OWN field name -- never
    # aliased to the estimator's delta_w_hat/delta_q_hat names.
    assert result.sensor_result["truth"]["delta_w_mps"] == pytest.approx(1.5)
    assert result.sensor_result["truth"]["delta_q_rads"] == pytest.approx(0.02)
    assert result.sensor_result["measured"]["delta_w_mps"] == pytest.approx(1.5)
    assert result.sensor_result["measured"]["delta_q_rads"] == pytest.approx(0.02)
    # Estimated state uses its own distinct field names.
    assert hasattr(result.estimated_state, "delta_w_hat")
    assert hasattr(result.estimated_state, "delta_q_hat")
    # First sample -> neither predict() nor update() -> step_index is 0.
    assert result.estimated_state.step_index == 0
    assert result.health_result is not None


# ---------------------------------------------------------------------------
# 4/5. Timestamp preservation + dt propagation
# ---------------------------------------------------------------------------


def test_timestamp_preservation_no_invented_clock():
    pipeline = make_pipeline()
    row = truth_row(time_s=3.14159, delta_w_mps=0.0, delta_q_rads=0.0)
    result = pipeline.process_truth_sample(row)
    assert result.timestamp_s == 3.14159


def test_dt_propagation_matches_real_simulator_timestamps():
    pipeline, estimator = make_recording_pipeline()

    pipeline.process_truth_sample(truth_row(time_s=0.0, delta_w_mps=0.0, delta_q_rads=0.0))
    pipeline.process_truth_sample(truth_row(time_s=0.02, delta_w_mps=0.0, delta_q_rads=0.0))
    pipeline.process_truth_sample(truth_row(time_s=0.05, delta_w_mps=0.0, delta_q_rads=0.0))

    # First sample: update()-only, no predict() call at all.
    # Second/third samples: predict() called with the REAL delta between
    # consecutive simulator timestamps, not an invented fixed step.
    assert len(estimator.recorded_predict_calls) == 2
    dt1, _ = estimator.recorded_predict_calls[0]
    dt2, _ = estimator.recorded_predict_calls[1]
    assert dt1 == pytest.approx(0.02)
    assert dt2 == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# 6. Provenance preservation
# ---------------------------------------------------------------------------


def test_provenance_is_preserved_and_never_upgraded():
    pipeline = make_pipeline()
    result = pipeline.process_truth_sample(truth_row(time_s=0.0, delta_w_mps=0.1, delta_q_rads=0.01))

    assert result.sensor_result["sensor_model_status"] == "PROJECT_DEFINED_PLACEHOLDER_NOT_NASA_VERIFIED"
    assert result.estimated_state.provenance.estimator_method == "PROJECT_DEFINED_REDUCED_ORDER_LINEAR_KALMAN_FILTER"
    assert result.health_result.provenance == "PROJECT_DEFINED_RULE_BASED_HEALTH_MONITOR"
    assert result.pipeline_provenance == PIPELINE_PROVENANCE == "PROJECT_DEFINED_ORCHESTRATION"

    for value in (
        result.sensor_result["sensor_model_status"],
        result.estimated_state.provenance.estimator_method,
        result.health_result.provenance,
        result.pipeline_provenance,
    ):
        # Every provenance label the pipeline surfaces must start with
        # PROJECT_DEFINED (this project's actual vocabulary for
        # non-NASA-sourced values) and never claim NASA_VERIFIED status.
        assert value.startswith("PROJECT_DEFINED")
        assert value != "NASA_VERIFIED"


# ---------------------------------------------------------------------------
# 7. Status propagation (propulsion)
# ---------------------------------------------------------------------------


def test_propulsion_status_passes_through_unreclassified():
    pipeline = make_pipeline()
    row = truth_row(time_s=0.0, delta_w_mps=0.0, delta_q_rads=0.0,
                     propulsion_model_status="MODEL_NOT_IDENTIFIED")

    result = pipeline.process_truth_sample(row)

    assert result.propulsion_status == "MODEL_NOT_IDENTIFIED"
    prop_diag = result.health_result.diagnostics["PROPULSION_STATUS_CHECK"]
    # The real HealthMonitorConfig default explicitly treats this as a
    # model-status limitation, never as an engine/propulsion FAULT.
    assert prop_diag["classification"] == "MODEL_STATUS_LIMITATION"
    assert prop_diag["passed"] is True


def test_simulation_status_is_none_because_real_simulator_never_emits_one():
    pipeline = make_pipeline()
    row = truth_row(time_s=0.0, delta_w_mps=0.0, delta_q_rads=0.0)
    assert "simulation_status" not in row  # matches the real Simulator schema
    result = pipeline.process_truth_sample(row)
    assert result.health_result.diagnostics["SIMULATION_STATUS_CHECK"]["passed"] is None


# ---------------------------------------------------------------------------
# 8. Invalid truth/sensor input handling
# ---------------------------------------------------------------------------


def test_non_finite_truth_value_rejected_before_sensor_model():
    pipeline = make_pipeline()
    row = truth_row(time_s=0.0, delta_w_mps=float("nan"), delta_q_rads=0.0)
    with pytest.raises(InvalidTruthSampleError):
        pipeline.process_truth_sample(row)


def test_missing_required_truth_fields_rejected():
    pipeline = make_pipeline()
    with pytest.raises(InvalidTruthSampleError):
        pipeline.process_truth_sample({"time_s": 0.0})  # missing delta_w_mps etc.


# ---------------------------------------------------------------------------
# 9. Invalid estimator-boundary data handling (real estimator, real error)
# ---------------------------------------------------------------------------


def test_measurement_dimension_mismatch_raises_estimator_failure():
    # A REAL 3-measurement-row estimator (w, q, az) paired with a pipeline
    # still configured for the default 2-channel adapter -> real
    # dimension mismatch inside KalmanStateEstimator.update().
    sensor_model = SensorModel(suite=_zero_noise_sensor_suite(), seed=7)
    estimator = KalmanStateEstimator(
        F=identity_transition_matrix(),
        H=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        Q=[[1.0e-4, 0.0], [0.0, 1.0e-6]],
        R=[[1.0e-6, 0.0, 0.0], [0.0, 1.0e-6, 0.0], [0.0, 0.0, 1.0e-6]],
        x0=[0.0, 0.0],
        P0=[[1.0, 0.0], [0.0, 1.0e-2]],
    )
    health_monitor = HealthMonitor()
    pipeline = DigitalTwinPipeline(sensor_model=sensor_model, estimator=estimator, health_monitor=health_monitor)

    # The first sample doesn't update(), so we need to run a second sample
    # to trigger the EstimatorFailureError caused by the mismatch in update()
    pipeline.process_truth_sample(truth_row(time_s=0.0, delta_w_mps=0.0, delta_q_rads=0.0))
    with pytest.raises(EstimatorFailureError) as exc_info:
        pipeline.process_truth_sample(truth_row(time_s=0.02, delta_w_mps=0.0, delta_q_rads=0.0))
    assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# 10. Invalid health-monitor input handling (real HealthMonitor)
# ---------------------------------------------------------------------------


def test_health_monitor_reports_invalid_for_malformed_covariance():
    pipeline = make_pipeline()
    bad_input = HealthCheckInput(estimated_delta_w=0.0, estimated_delta_q=0.0, covariance=[[1.0]])
    result = pipeline.health_monitor.evaluate(bad_input)
    assert result.overall_status.value == "INVALID"


# ---------------------------------------------------------------------------
# 11. Multiple sequential steps
# ---------------------------------------------------------------------------


def test_sequential_multi_step_execution():
    pipeline = make_pipeline()
    rows = [truth_row(time_s=0.02 * i, delta_w_mps=0.01 * i, delta_q_rads=0.001 * i) for i in range(5)]

    results = pipeline.process_history(rows)

    assert [r.timestamp_s for r in results] == [0.0, 0.02, 0.04, 0.06, 0.08]
    assert [r.estimated_state.step_index for r in results] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 12. Run-to-run state isolation
# ---------------------------------------------------------------------------


def test_run_to_run_state_isolation_requires_fresh_estimator():
    pipeline = make_pipeline()
    pipeline.process_history([
        truth_row(time_s=0.0, delta_w_mps=1.0, delta_q_rads=0.1),
        truth_row(time_s=0.02, delta_w_mps=1.0, delta_q_rads=0.1),
    ])
    assert pipeline.estimator.get_estimated_state().step_index == 1

    # Re-using process_history without resetting must fail fast
    with pytest.raises(IntegrationError):
        pipeline.process_history([truth_row(time_s=0.04, delta_w_mps=0.0, delta_q_rads=0.0)])

    # Re-using time_s=0.0 without resetting must fail fast (non-increasing
    # timestamp), proving the pipeline does not silently start a new run.
    with pytest.raises(InvalidTruthSampleError):
        pipeline.process_truth_sample(truth_row(time_s=0.0, delta_w_mps=0.0, delta_q_rads=0.0))

    fresh_estimator = default_reduced_order_estimator(x0=(0.0, 0.0))
    pipeline.reset_run(estimator=fresh_estimator)

    result = pipeline.process_truth_sample(truth_row(time_s=0.0, delta_w_mps=0.0, delta_q_rads=0.0))
    assert result.estimated_state.step_index == 0  # fresh run, not continuing at 2


def test_process_history_twice_fails_without_reset():
    pipeline = make_pipeline()
    history = [
        truth_row(time_s=0.0, delta_w_mps=1.0, delta_q_rads=0.1),
        truth_row(time_s=0.02, delta_w_mps=1.0, delta_q_rads=0.1),
    ]
    pipeline.process_history(history)

    # Calling process_history again on the same pipeline instance without
    # resetting must fail, because the first run left timestamps/state behind.
    with pytest.raises(IntegrationError):
        pipeline.process_history(history)

    # But resetting with a fresh estimator makes it work again.
    fresh_estimator = default_reduced_order_estimator(x0=(0.0, 0.0))
    pipeline.reset_run(estimator=fresh_estimator)
    results = pipeline.process_history(history)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# 13. No simulator-truth mutation
# ---------------------------------------------------------------------------


def test_truth_row_is_not_mutated():
    pipeline = make_pipeline()
    row = truth_row(time_s=0.0, delta_w_mps=2.0, delta_q_rads=0.05)
    row_before = copy.deepcopy(row)

    pipeline.process_truth_sample(row)

    assert row == row_before


# ---------------------------------------------------------------------------
# 14. No control-command generation
# ---------------------------------------------------------------------------


def test_control_input_is_passed_through_never_invented():
    pipeline, estimator = make_recording_pipeline()

    pipeline.process_truth_sample(truth_row(time_s=0.0, delta_w_mps=0.0, delta_q_rads=0.0, delta_e_rad=0.0))
    pipeline.process_truth_sample(truth_row(time_s=0.02, delta_w_mps=0.0, delta_q_rads=0.0, delta_e_rad=-0.05))

    # Only one predict() call so far (the second sample); its u must be
    # EXACTLY the real elevator command already on that truth row, not a
    # fabricated value.
    assert len(estimator.recorded_predict_calls) == 1
    _, u = estimator.recorded_predict_calls[0]
    assert u == pytest.approx(-0.05)


def test_missing_control_field_defaults_to_zero_not_fabricated():
    pipeline, estimator = make_recording_pipeline()
    row_no_control = {k: v for k, v in truth_row(time_s=0.0, delta_w_mps=0.0, delta_q_rads=0.0).items()
                       if k != "delta_e_rad"}
    row2 = {k: v for k, v in truth_row(time_s=0.02, delta_w_mps=0.0, delta_q_rads=0.0).items()
            if k != "delta_e_rad"}

    pipeline.process_truth_sample(row_no_control)
    pipeline.process_truth_sample(row2)

    _, u = estimator.recorded_predict_calls[0]
    assert u == 0.0


# ---------------------------------------------------------------------------
# 15/16. No duplicated simulation/estimator physics
# ---------------------------------------------------------------------------


def test_pipeline_does_not_alter_estimator_or_sensor_model_configuration():
    pipeline = make_pipeline()
    F_before = copy.deepcopy(pipeline.estimator.F)
    H_before = copy.deepcopy(pipeline.estimator.H)
    suite_before = pipeline.sensor_model.suite

    pipeline.process_truth_sample(truth_row(time_s=0.0, delta_w_mps=1.0, delta_q_rads=0.1))

    # The pipeline must never rewrite the estimator's own transition/
    # measurement model or the sensor model's noise/bias parameters --
    # that physics belongs entirely to those modules.
    assert pipeline.estimator.F == F_before
    assert pipeline.estimator.H == H_before
    assert pipeline.sensor_model.suite is suite_before


# ---------------------------------------------------------------------------
# 17. Malformed / empty input handling
# ---------------------------------------------------------------------------


def test_empty_history_rejected():
    pipeline = make_pipeline()
    with pytest.raises(InvalidTruthSampleError):
        pipeline.process_history([])


def test_empty_truth_row_rejected():
    pipeline = make_pipeline()
    with pytest.raises(InvalidTruthSampleError):
        pipeline.process_truth_sample({})


# ---------------------------------------------------------------------------
# End-to-end smoke test (real SensorModel -> real KalmanStateEstimator ->
# real HealthMonitor, driven by a short hand-built truth trajectory shaped
# exactly like a real Simulator.run() history). Proves module
# compatibility only -- NOT NASA validation.
# ---------------------------------------------------------------------------


def test_real_module_end_to_end_smoke():
    pipeline = make_pipeline(persistence_samples=1)
    trajectory = [
        truth_row(time_s=0.00, delta_w_mps=0.0, delta_q_rads=0.0, perturbation_az_mps2=0.0),
        truth_row(time_s=0.02, delta_w_mps=0.2, delta_q_rads=0.01, perturbation_az_mps2=0.1),
        truth_row(time_s=0.04, delta_w_mps=0.35, delta_q_rads=0.015, perturbation_az_mps2=0.15),
        truth_row(time_s=0.06, delta_w_mps=0.40, delta_q_rads=0.012, perturbation_az_mps2=0.12),
    ]

    results = pipeline.process_history(trajectory)

    assert len(results) == 4
    for r in results:
        assert math.isfinite(r.estimated_state.delta_w_hat)
        assert math.isfinite(r.estimated_state.delta_q_hat)
        assert r.health_result.overall_status.value in ("HEALTHY", "WARNING", "FAULT")