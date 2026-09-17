"""
tests/test_estimator.py

Tests for digital_twin/estimator.py.

Coverage map (see project spec):
    1.  Correct initialization
    2.  Correct state dimension = 2
    3.  Measurement dimension is correct
    4.  One prediction/update cycle works
    5.  State estimates remain finite
    6.  Covariance remains finite
    7.  Covariance remains symmetric within numerical tolerance
    8.  Invalid NaN/Inf measurements are rejected
    9.  Invalid covariance matrices are rejected
    10. Invalid dt is rejected
    11. Zero-noise deterministic case behaves predictably
    12. Repeated identical measurements produce reproducible estimates
    13. PROJECT_DEFINED provenance is present
    14. NASA_REFERENCE_METHOD is not incorrectly claimed as implemented
    15. Estimated state is clearly separated from true state
    + Simple analytic test (x_k = x_{k-1}, H = I, known Q/R)
    + NASA/T-2 project-level integration test (SensorModel -> Estimator)
"""

import math

import pytest

from digital_twin.estimator import (
    STATE_DIM,
    ESTIMATOR_METHOD,
    NASA_REFERENCE_METHOD_STATUS,
    EstimatorError,
    EstimatorProvenance,
    EstimatedState,
    KalmanStateEstimator,
    identity_transition_matrix,
    identity_measurement_matrix,
    default_process_noise,
    default_initial_covariance,
    build_measurement_noise_from_sensor_suite,
    default_reduced_order_estimator,
    extract_measurement_from_sensor_output,
    extract_measurement_from_history_row,
)

from sensors.sensor_model import SensorModel, default_sensor_suite


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _simple_estimator(x0=(0.0, 0.0), q_val=1.0e-3, r_val=1.0e-2, p0_val=1.0):
    """A simple x_k = x_{k-1}, H = I estimator with round, known Q/R/P0 --
    the 'simple known linear system' called for by the project spec's
    analytic-test requirement."""
    F = identity_transition_matrix()
    H = identity_measurement_matrix()
    Q = [[q_val, 0.0], [0.0, q_val]]
    R = [[r_val, 0.0], [0.0, r_val]]
    P0 = [[p0_val, 0.0], [0.0, p0_val]]
    return KalmanStateEstimator(F=F, H=H, Q=Q, R=R, x0=list(x0), P0=P0)


# ---------------------------------------------------------------------------
# 1. Correct initialization
# ---------------------------------------------------------------------------


def test_correct_initialization():
    est = _simple_estimator(x0=(0.5, -0.1))
    # Confirm x0/P0 represent the estimate at t=0. They are intentionally
    # the known initial state and must not be overwritten by a blind update()
    # on the very first sample.
    assert est.x == [0.5, -0.1]
    assert est.P == [[1.0, 0.0], [0.0, 1.0]]
    assert est._step_index == 0


# ---------------------------------------------------------------------------
# 2. Correct state dimension = 2
# ---------------------------------------------------------------------------


def test_state_dimension_is_two():
    est = _simple_estimator()
    assert STATE_DIM == 2
    assert len(est.x) == 2
    assert len(est.F) == 2 and len(est.F[0]) == 2


def test_state_dimension_mismatch_rejected():
    F = identity_transition_matrix()
    H = identity_measurement_matrix()
    Q = default_process_noise()
    R = [[1.0e-2, 0.0], [0.0, 1.0e-2]]
    P0 = default_initial_covariance()
    with pytest.raises(EstimatorError):
        KalmanStateEstimator(F=F, H=H, Q=Q, R=R, x0=[0.0, 0.0, 0.0], P0=P0)


# ---------------------------------------------------------------------------
# 3. Measurement dimension is correct
# ---------------------------------------------------------------------------


def test_measurement_dimension_matches_h():
    # A 3-row H (e.g. w, q, and a hypothetical third channel) should
    # require a matching 3-dim R and 3-dim measurement vector.
    F = identity_transition_matrix()
    H = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    Q = default_process_noise()
    R = [[1.0e-2, 0.0, 0.0], [0.0, 1.0e-3, 0.0], [0.0, 0.0, 1.0e-2]]
    P0 = default_initial_covariance()
    est = KalmanStateEstimator(F=F, H=H, Q=Q, R=R, x0=[0.0, 0.0], P0=P0)
    assert est.measurement_dim == 3

    est.predict(dt=0.1)
    with pytest.raises(EstimatorError):
        est.update([0.1, 0.01])  # wrong length: only 2 given, 3 expected


def test_h_column_count_must_match_state_dim():
    F = identity_transition_matrix()
    H = [[1.0, 0.0, 0.0]]  # 3 columns -- invalid, state dim is 2
    Q = default_process_noise()
    R = [[1.0e-2]]
    P0 = default_initial_covariance()
    with pytest.raises(EstimatorError):
        KalmanStateEstimator(F=F, H=H, Q=Q, R=R, x0=[0.0, 0.0], P0=P0)


# ---------------------------------------------------------------------------
# 4. One prediction/update cycle works
# ---------------------------------------------------------------------------


def test_one_predict_update_cycle_works():
    est = _simple_estimator()
    est.predict(dt=0.1)
    x_after_update, p_after_update = est.update([0.2, 0.01])
    assert isinstance(x_after_update, list)
    assert isinstance(p_after_update, list)
    state = est.get_estimated_state()
    assert isinstance(state, EstimatedState)


def test_step_convenience_method():
    est = _simple_estimator()
    state = est.step(z=[0.2, 0.01], dt=0.1)
    assert isinstance(state, EstimatedState)
    assert state.step_index == 1


# ---------------------------------------------------------------------------
# 5 & 6. State estimates and covariance remain finite
# ---------------------------------------------------------------------------


def test_state_and_covariance_remain_finite_over_many_steps():
    est = _simple_estimator()
    for i in range(50):
        z = [0.1 * math.sin(i * 0.3), 0.01 * math.cos(i * 0.3)]
        est.step(z=z, dt=0.05)

    assert math.isfinite(est.x[0])
    assert math.isfinite(est.x[1])
    for row in est.P:
        for val in row:
            assert math.isfinite(val)


# ---------------------------------------------------------------------------
# 7. Covariance remains symmetric within numerical tolerance
# ---------------------------------------------------------------------------


def test_covariance_remains_symmetric():
    est = _simple_estimator()
    for i in range(25):
        est.step(z=[0.05 * i, -0.01 * i], dt=0.1)
        assert abs(est.P[0][1] - est.P[1][0]) < 1.0e-9


# ---------------------------------------------------------------------------
# 8. Invalid NaN/Inf measurements are rejected
# ---------------------------------------------------------------------------


def test_nan_measurement_rejected():
    est = _simple_estimator()
    est.predict(dt=0.1)
    with pytest.raises(EstimatorError):
        est.update([float("nan"), 0.0])


def test_inf_measurement_rejected():
    est = _simple_estimator()
    est.predict(dt=0.1)
    with pytest.raises(EstimatorError):
        est.update([float("inf"), 0.0])


# ---------------------------------------------------------------------------
# 9. Invalid covariance matrices are rejected
# ---------------------------------------------------------------------------


def test_non_symmetric_p0_rejected():
    F = identity_transition_matrix()
    H = identity_measurement_matrix()
    Q = default_process_noise()
    R = [[1.0e-2, 0.0], [0.0, 1.0e-2]]
    P0 = [[1.0, 0.5], [0.0, 1.0]]  # not symmetric
    with pytest.raises(EstimatorError):
        KalmanStateEstimator(F=F, H=H, Q=Q, R=R, x0=[0.0, 0.0], P0=P0)


def test_negative_definite_q_rejected():
    F = identity_transition_matrix()
    H = identity_measurement_matrix()
    Q = [[-1.0, 0.0], [0.0, -1.0]]  # negative variance -- invalid
    R = [[1.0e-2, 0.0], [0.0, 1.0e-2]]
    P0 = default_initial_covariance()
    with pytest.raises(EstimatorError):
        KalmanStateEstimator(F=F, H=H, Q=Q, R=R, x0=[0.0, 0.0], P0=P0)


def test_non_positive_semidefinite_r_rejected():
    F = identity_transition_matrix()
    H = identity_measurement_matrix()
    Q = default_process_noise()
    # symmetric but not PSD: det = 1*1 - 2*2 = -3 < 0
    R = [[1.0, 2.0], [2.0, 1.0]]
    P0 = default_initial_covariance()
    with pytest.raises(EstimatorError):
        KalmanStateEstimator(F=F, H=H, Q=Q, R=R, x0=[0.0, 0.0], P0=P0)


def test_zero_covariance_matrices_are_accepted():
    # PSD (not strictly PD) must be accepted -- required for the zero-noise
    # deterministic test below.
    F = identity_transition_matrix()
    H = identity_measurement_matrix()
    Q = [[0.0, 0.0], [0.0, 0.0]]
    R = [[0.0, 0.0], [0.0, 0.0]]
    P0 = [[1.0, 0.0], [0.0, 1.0]]
    est = KalmanStateEstimator(F=F, H=H, Q=Q, R=R, x0=[0.0, 0.0], P0=P0)
    assert est.Q == Q


# ---------------------------------------------------------------------------
# 10. Invalid dt is rejected
# ---------------------------------------------------------------------------


def test_zero_dt_rejected():
    est = _simple_estimator()
    with pytest.raises(EstimatorError):
        est.predict(dt=0.0)


def test_negative_dt_rejected():
    est = _simple_estimator()
    with pytest.raises(EstimatorError):
        est.predict(dt=-0.1)


def test_nan_dt_rejected():
    est = _simple_estimator()
    with pytest.raises(EstimatorError):
        est.predict(dt=float("nan"))


# ---------------------------------------------------------------------------
# 11. Zero-noise deterministic case behaves predictably
# ---------------------------------------------------------------------------


def test_zero_noise_deterministic_case():
    # F = H = I, Q = R = 0: with a positive-definite P0, the Kalman gain
    # reduces exactly to K = I, so the updated estimate must equal the
    # measurement exactly (up to floating-point round-off).
    F = identity_transition_matrix()
    H = identity_measurement_matrix()
    Q = [[0.0, 0.0], [0.0, 0.0]]
    R = [[0.0, 0.0], [0.0, 0.0]]
    P0 = [[1.0, 0.0], [0.0, 1.0]]
    est = KalmanStateEstimator(F=F, H=H, Q=Q, R=R, x0=[0.0, 0.0], P0=P0)

    z = [3.0, -2.0]
    est.predict(dt=1.0)
    est.update(z)

    assert math.isclose(est.x[0], z[0], abs_tol=1.0e-9)
    assert math.isclose(est.x[1], z[1], abs_tol=1.0e-9)


# ---------------------------------------------------------------------------
# 12. Repeated identical measurements produce reproducible estimates
# ---------------------------------------------------------------------------


def test_reproducible_across_identical_runs():
    z_sequence = [[0.1, 0.01], [0.15, 0.005], [0.05, 0.02]]

    est_a = _simple_estimator()
    est_b = _simple_estimator()

    for z in z_sequence:
        est_a.step(z=z, dt=0.1)
        est_b.step(z=z, dt=0.1)

    assert est_a.x == est_b.x
    assert est_a.P == est_b.P


def test_reproducible_with_repeated_identical_measurement():
    est = _simple_estimator()
    results = []
    for _ in range(5):
        state = est.step(z=[1.0, 0.0], dt=0.1)
        results.append((state.delta_w_hat, state.delta_q_hat))

    # Estimate should monotonically approach the repeated measurement
    # rather than jump around unpredictably.
    ws = [r[0] for r in results]
    assert all(earlier <= later + 1e-12 for earlier, later in zip(ws, ws[1:]))
    assert ws[-1] <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 13. PROJECT_DEFINED provenance is present
# ---------------------------------------------------------------------------


def test_provenance_present_and_project_defined():
    est = _simple_estimator()
    assert isinstance(est.provenance, EstimatorProvenance)
    assert est.provenance.estimator_method == ESTIMATOR_METHOD
    assert "PROJECT_DEFINED" in est.provenance.estimator_method

    state = est.step(z=[0.1, 0.0], dt=0.1)
    assert state.provenance.estimator_method == ESTIMATOR_METHOD


def test_default_reduced_order_estimator_provenance_traces_to_sensor_model():
    est = default_reduced_order_estimator()
    assert "DERIVED_FROM_SENSOR_MODEL" in est.provenance.measurement_noise_provenance
    assert "w_sensor" in est.provenance.measurement_noise_provenance
    assert "RANDOM_WALK_ASSUMPTION" in est.provenance.transition_model_provenance


# ---------------------------------------------------------------------------
# 14. NASA_REFERENCE_METHOD is not incorrectly claimed as implemented
# ---------------------------------------------------------------------------


def test_nasa_reference_method_not_claimed_as_implemented():
    est = _simple_estimator()
    assert "NOT_IMPLEMENTED" in est.provenance.nasa_reference_method_status
    assert est.provenance.nasa_reference_method_status == NASA_REFERENCE_METHOD_STATUS
    assert est.provenance.estimator_method != est.provenance.nasa_reference_method_status
    assert "filter-error" in est.provenance.nasa_reference_method_status.lower()


# ---------------------------------------------------------------------------
# 15. Estimated state is clearly separated from true state
# ---------------------------------------------------------------------------


def test_estimated_state_field_names_are_distinct_from_truth():
    est = _simple_estimator()
    state = est.step(z=[0.3, 0.02], dt=0.1)

    field_names = set(state.as_dict().keys())
    assert "delta_w_hat" in field_names
    assert "delta_q_hat" in field_names
    # The true-state field names used elsewhere in the project
    # (models/flight_dynamics.PerturbationState) must not appear here
    # unqualified, so truth and estimate can never be confused.
    assert "delta_w" not in field_names
    assert "delta_q" not in field_names


# ---------------------------------------------------------------------------
# Simple analytic test: x_k = x_{k-1}, H = I, known Q/R
# ---------------------------------------------------------------------------


def test_analytic_update_moves_toward_measurement_correctly():
    """Verifies estimator mathematics in isolation from T-2 physics, using
    the simplest possible linear system (F = H = I)."""
    est = _simple_estimator(x0=(0.0, 0.0), q_val=1.0e-3, r_val=1.0e-2, p0_val=1.0)

    z = [2.0, 1.0]
    est.predict(dt=1.0)
    x_before = list(est.x)
    est.update(z)

    # The update must move the estimate strictly toward the measurement
    # (not overshoot past it, not move away from it) in both components.
    for i in range(2):
        assert x_before[i] <= est.x[i] <= z[i] + 1e-9 if z[i] >= x_before[i] else (
            z[i] - 1e-9 <= est.x[i] <= x_before[i]
        )

    # With P0 >> R (P0=1.0 vs R=0.01), the filter should trust the
    # measurement heavily: the estimate should land close to z, not close
    # to the prior.
    assert abs(est.x[0] - z[0]) < abs(x_before[0] - z[0]) * 0.5
    assert abs(est.x[1] - z[1]) < abs(x_before[1] - z[1]) * 0.5


def test_analytic_kalman_gain_matches_hand_derivation():
    """For F=H=I with scalar-diagonal Q/R/P0, the 1D Kalman gain has a
    closed form: K = P_pred / (P_pred + R). Verify the implementation
    matches that closed form exactly (to floating-point precision)."""
    q_val, r_val, p0_val = 0.01, 0.05, 1.0
    est = _simple_estimator(x0=(0.0, 0.0), q_val=q_val, r_val=r_val, p0_val=p0_val)

    est.predict(dt=1.0)
    p_pred = p0_val + q_val  # F=I, so P_pred = P0 + Q for each diagonal entry
    expected_k = p_pred / (p_pred + r_val)

    z = [1.0, 1.0]
    est.update(z)

    expected_x = expected_k * z[0]  # x_pred was 0
    assert math.isclose(est.x[0], expected_x, rel_tol=1e-9)
    assert math.isclose(est.x[1], expected_x, rel_tol=1e-9)

    expected_p = (1.0 - expected_k) * p_pred
    assert math.isclose(est.P[0][0], expected_p, rel_tol=1e-9)
    assert math.isclose(est.P[1][1], expected_p, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# NASA/T-2 project-level integration test: SensorModel -> Estimator
#
# NOTE: this proves project-module compatibility (SensorModel's dict output
# can be adapted and consumed by the estimator), NOT NASA validation of the
# estimator's numerical results.
# ---------------------------------------------------------------------------


def test_sensor_model_to_estimator_integration():
    sensor = SensorModel(suite=default_sensor_suite(), seed=1234)
    est = default_reduced_order_estimator(x0=(0.0, 0.0), sensor_suite=default_sensor_suite())

    # Synthetic truth trajectory -- a simple decaying oscillation standing
    # in for delta_w/delta_q/perturbation_az truth. This test does not
    # claim these numbers come from models/flight_dynamics.py; it only
    # proves SensorModel -> Estimator wiring.
    dt = 0.05
    true_w = 0.0
    true_q = 0.0
    for i in range(40):
        true_w = 1.0 * math.exp(-0.1 * i) * math.cos(0.5 * i)
        true_q = 0.05 * math.exp(-0.1 * i) * math.sin(0.5 * i)
        true_az = 0.2 * true_w  # arbitrary stand-in, not a physics claim

        sensor_output = sensor.measure(true_w, true_q, true_az)
        z = extract_measurement_from_sensor_output(sensor_output)

        state = est.step(z=z, dt=dt)

    assert math.isfinite(state.delta_w_hat)
    assert math.isfinite(state.delta_q_hat)
    # Loose sanity bound only -- this is a wiring/compatibility check, not a
    # claim of estimator accuracy against any validated reference.
    assert abs(state.delta_w_hat - true_w) < 1.0
    assert abs(state.delta_q_hat - true_q) < 1.0
    assert state.provenance.estimator_method == ESTIMATOR_METHOD


def test_history_row_adapter():
    row = {"time_s": 1.0, "delta_w_mps": 0.42, "delta_q_rads": -0.01}
    z = extract_measurement_from_history_row(row)
    assert z == [0.42, -0.01]


def test_sensor_output_adapter_rejects_malformed_input():
    with pytest.raises(EstimatorError):
        extract_measurement_from_sensor_output({"not_measured": {}})


# ---------------------------------------------------------------------------
# Optional physically-informed transition matrix, built from the REAL
# models.flight_dynamics module. Skipped unless models.aerodynamics (a
# transitive dependency of models.flight_dynamics) is importable in this
# environment.
# ---------------------------------------------------------------------------


def test_build_transition_from_flight_dynamics_when_available():
    pytest.importorskip(
        "models.aerodynamics",
        reason=(
            "models/aerodynamics.py was not provided in this environment; "
            "build_transition_from_flight_dynamics is exercised only when "
            "the full project (including models/aerodynamics.py) is present."
        ),
    )
    from models.aerodynamics import FlightCondition
    from models.flight_dynamics import (
        FlightDynamics,
        ReferenceTrim,
        ShortPeriodDerivatives,
    )
    from digital_twin.estimator import build_transition_from_flight_dynamics

    condition = FlightCondition.FLIGHT_41
    trim = ReferenceTrim(flight_condition=condition, u0=45.0)
    fd = FlightDynamics(trim)
    derivs = ShortPeriodDerivatives(
        flight_condition=condition,
        Zw=-1.0, Zq=-0.1, Zde=-2.0,
        Mw=-0.5, Mq=-0.2, Mde=-3.0,
    )

    F, B = build_transition_from_flight_dynamics(
        flight_dynamics=fd, mass=24.0, iyy=6.3, derivatives=derivs, dt=0.01
    )
    assert len(F) == 2 and len(F[0]) == 2
    assert len(B) == 2 and len(B[0]) == 1
    for row in F + B:
        for val in row:
            assert math.isfinite(val)
