"""
digital_twin/estimator.py

PROJECT_DEFINED_REDUCED_ORDER_STATE_ESTIMATOR for the NASA AirSTAR T-2
digital twin.

Responsibility (and ONLY responsibility):

    sensor measurements (measured_delta_w, measured_delta_q, [optional
    measured_perturbation_az])
        |
    linear discrete Kalman filter
        |
    estimated reduced-order state x_hat = [delta_w_hat, delta_q_hat]

==============================================================================
PROVENANCE -- READ BEFORE USING ANY VALUE FROM THIS MODULE
==============================================================================

NASA reference method (NOT implemented here):

    Jared A. Grauer and Eugene A. Morelli, "A New Formulation of the
    Filter-Error Method for Aerodynamic Parameter Estimation in
    Turbulence," AIAA 2015-2704 / NASA NTRS 20160006007.

    That paper describes a continuous-discrete filter-error method with
    Gauss-Newton aerodynamic-parameter identification, operating on an
    [alpha, q] output formulation. NONE of that is implemented by this
    module: no Gauss-Newton optimization, no online aerodynamic-coefficient
    identification, no turbulence-parameter estimation, and no silent
    renaming of this project's [delta_w, delta_q] state into [alpha, q].
    See NASA_REFERENCE_METHOD_STATUS below, which every EstimatedState
    carries in its `provenance`.

This module instead implements:

    ESTIMATOR_METHOD = PROJECT_DEFINED_REDUCED_ORDER_LINEAR_KALMAN_FILTER

    A standard 2-state linear discrete Kalman filter operating directly on
    the project's existing perturbation state x = [delta_w, delta_q], as
    defined in models/flight_dynamics.py (PerturbationState). This is the
    "simple, defensible" first estimator called for in the project plan,
    not a reproduction of the NASA filter-error method.

Q (process noise), R (measurement noise), and P0 (initial covariance) are
PROJECT_DEFINED_PARAMETER values unless a caller supplies sourced ones. This
module never labels a value NASA-verified without a caller explicitly
supplying provenance metadata to that effect, and the conservative defaults
provided below are placeholders, not "realistic" values with any external
source. Every KalmanStateEstimator and EstimatedState exposes an
`EstimatorProvenance` record so a user can see exactly where Q, R, and P0
came from.

==============================================================================
SCOPE BOUNDARIES (explicit, matching the pattern used elsewhere in the
project -- e.g. models/aircraft.py's "layer boundary" docstring)
==============================================================================

This module intentionally contains NO:
    - Gauss-Newton / filter-error aerodynamic parameter identification
      (NASA_REFERENCE_METHOD; future: estimation/filter_error_estimator.py)
    - fault detection, anomaly detection, degradation estimation, or any
      other health-monitoring logic (future: digital_twin/health_monitor.py)
    - machine learning of any kind
    - a redefinition of the project state from [delta_w, delta_q] to
      [alpha, q]
    - a reimplementation of models/flight_dynamics.py's equations of
      motion, models/conversion.py's Zw/Zq/... dimensionalization, or
      sensors/sensor_model.py's noise/bias model. Where this module needs
      numbers that "belong" to those layers (a linearized transition
      matrix, a measurement-noise covariance), it either (a) requires the
      caller to supply them explicitly, or (b) *calls* the existing code
      (see `build_transition_from_flight_dynamics`) rather than
      re-deriving its physics.

==============================================================================
STATE AND MEASUREMENT MODEL
==============================================================================

State (fixed at 2, matching the current project model -- do not silently
generalize this to 3+ states without an explicit project decision):

    x = [delta_w, delta_q]^T
        delta_w: perturbation body-axis vertical velocity (m/s)
        delta_q: perturbation pitch rate (rad/s)

Minimum-required measurement model (matches sensors/sensor_model.py's
`w_sensor` and `rate_gyro_q` channels, both PROJECT_DEFINED /
NASA-documented-channel-but-project-defined-noise as described there):

    z = H x + v,   v ~ N(0, R),   H = I (2x2)

    i.e. measured_delta_w and measured_delta_q are treated as direct noisy
    observations of delta_w and delta_q. This is the "at minimum" model
    called for by the project plan.

Optional third channel (perturbation_az): sensors/sensor_model.py also
emits a noisy `accelerometer_az` measurement. Fusing it into this filter
requires a measurement row that expresses az as a linear function of the
state, e.g. az = (Zw*delta_w + Zq*delta_q)/mass (the same relationship
computed by models/flight_dynamics.py's FlightDynamics.calculate_state_
derivatives()['outputs']['perturbation_az'], MINUS the elevator term,
since delta_e is a control input rather than part of x). This module does
NOT fabricate that row internally -- doing so would duplicate
flight_dynamics.py's physics with project-derived dimensional derivatives
this module has no independent way to verify. A caller that has those
derivatives (Zw, Zq, mass) may construct a 3-row H accordingly and pass it
to `KalmanStateEstimator` directly; this module places no numerical value
between accelerometer_az (m/s^2, SI) and any NASA `g`-unit representation,
per the same warning documented in sensors/sensor_model.py.

==============================================================================
NUMERICAL IMPLEMENTATION NOTE
==============================================================================

No numpy import exists anywhere else in this project (models/, sensors/,
main.py all use plain Python + the stdlib), and no requirements file was
provided authorizing a new dependency. For a fixed 2-state filter with a
measurement dimension of at most 3, hand-written list-of-lists linear
algebra is entirely sufficient and keeps this module dependency-free,
consistent with existing project conventions. If a future revision of this
project standardizes on numpy, that is a project-level decision to be made
explicitly elsewhere, not something this module should reach for
unilaterally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# sensors/sensor_model.py has no dependency on models/aerodynamics.py or
# models/atmosphere.py, so importing it here does not create any risk of
# import failure due to those (separately-owned) modules being unavailable.
from sensors.sensor_model import SensorSuiteParameters, default_sensor_suite

# ---------------------------------------------------------------------------
# Provenance constants
# ---------------------------------------------------------------------------

NASA_REFERENCE_METHOD_CITATION = (
    "Jared A. Grauer and Eugene A. Morelli, \"A New Formulation of the "
    "Filter-Error Method for Aerodynamic Parameter Estimation in "
    "Turbulence,\" AIAA 2015-2704 / NASA NTRS 20160006007."
)

NASA_REFERENCE_METHOD_STATUS = (
    "NASA_REFERENCE_METHOD_NOT_IMPLEMENTED: the filter-error method, its "
    "Gauss-Newton aerodynamic-parameter optimization, and its [alpha, q] "
    "output formulation (see citation) are NOT implemented by this module "
    "and are not claimed to be reproduced by it. They remain a future "
    "target, e.g. estimation/filter_error_estimator.py."
)

ESTIMATOR_METHOD = "PROJECT_DEFINED_REDUCED_ORDER_LINEAR_KALMAN_FILTER"

# Fixed by the current project's reduced-order model: x = [delta_w, delta_q].
STATE_DIM = 2

_SYMMETRY_TOL = 1.0e-9
_PSD_TOL = 1.0e-9


class EstimatorError(ValueError):
    """Raised when estimator construction or inputs fail validation.

    Subclasses ValueError so existing project code that catches ValueError
    (see models/aircraft.py, models/flight_dynamics.py, models/conversion.py
    for the same pattern with their own *ConfigError types) keeps working.
    """


# ---------------------------------------------------------------------------
# Minimal linear algebra (pure Python; see module docstring for why no
# numpy). Matrices are row-major List[List[float]]; vectors are List[float].
# ---------------------------------------------------------------------------


def _identity(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _zeros(rows: int, cols: int) -> List[List[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _mat_transpose(A: Sequence[Sequence[float]]) -> List[List[float]]:
    rows = len(A)
    cols = len(A[0]) if rows else 0
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def _mat_mult(
    A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]
) -> List[List[float]]:
    n = len(A)
    k = len(A[0]) if n else 0
    k2 = len(B)
    m = len(B[0]) if k2 else 0
    if k != k2:
        raise EstimatorError(
            f"Matrix multiply shape mismatch: {n}x{k} @ {k2}x{m}"
        )
    result = _zeros(n, m)
    for i in range(n):
        for j in range(m):
            result[i][j] = sum(A[i][p] * B[p][j] for p in range(k))
    return result


def _mat_add(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _mat_sub(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _scalar_mult(k: float, A: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[k * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _mat_vec_mult(A: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    n = len(A)
    k = len(A[0]) if n else 0
    if k != len(v):
        raise EstimatorError(f"Matrix-vector shape mismatch: {n}x{k} @ {len(v)}")
    return [sum(A[i][p] * v[p] for p in range(k)) for i in range(n)]


def _vec_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [a[i] + b[i] for i in range(len(a))]


def _vec_sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [a[i] - b[i] for i in range(len(a))]


def _symmetrize(A: Sequence[Sequence[float]]) -> List[List[float]]:
    """Averages A with its transpose.

    Mathematically, P (and Q, R, S) are always exactly symmetric; this
    guards purely against floating-point round-off drift accumulated across
    repeated predict/update cycles. It never changes a matrix that is
    already symmetric beyond floating-point noise, and it never masks a
    genuinely asymmetric input -- construction-time validation
    (`_require_covariance`) rejects those before they ever reach here.
    """
    n = len(A)
    return [[0.5 * (A[i][j] + A[j][i]) for j in range(n)] for i in range(n)]


def _mat_inverse(A: Sequence[Sequence[float]]) -> List[List[float]]:
    """Gauss-Jordan matrix inverse with partial pivoting.

    Raises EstimatorError (not a bare exception) if A is singular to
    working precision, so callers get a clear estimator-domain error
    instead of an opaque ZeroDivisionError.
    """
    n = len(A)
    # Augment [A | I]
    aug = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1.0e-12:
            raise EstimatorError(
                "Matrix is singular (or numerically singular) and cannot be inverted."
            )
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        aug[col] = [val / pivot for val in aug[col]]

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor != 0.0:
                aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(2 * n)]

    return [row[n:] for row in aug]


def _is_symmetric(A: Sequence[Sequence[float]], tol: float = _SYMMETRY_TOL) -> bool:
    n = len(A)
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - A[j][i]) > tol:
                return False
    return True


def _is_positive_semidefinite(A: Sequence[Sequence[float]], tol: float = _PSD_TOL) -> bool:
    """LDL^T-based PSD check for a symmetric matrix (no numpy/eigendecomp).

    Attempts an LDL^T decomposition (A = L D L^T, L unit lower-triangular,
    D diagonal) with pivot tolerance. A is positive semi-definite iff every
    computed diagonal pivot is >= -tol. This is equivalent to attempting a
    Cholesky factorization while tolerating exact/near-zero pivots (which a
    plain Cholesky cannot), which is exactly the case this project needs to
    support (e.g. Q = 0 or R = 0 for the zero-noise deterministic test).
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    D = [0.0] * n
    for i in range(n):
        s = A[i][i] - sum(L[i][k] ** 2 * D[k] for k in range(i))
        if s < -tol:
            return False
        D[i] = max(s, 0.0)
        L[i][i] = 1.0
        for j in range(i + 1, n):
            if D[i] > tol:
                L[j][i] = (
                    A[j][i] - sum(L[j][k] * L[i][k] * D[k] for k in range(i))
                ) / D[i]
            else:
                L[j][i] = 0.0
    return True


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_finite_scalar(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EstimatorError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise EstimatorError(f"{name} must be finite, got {value!r}")
    return float(value)


def _require_finite_vector(
    name: str, v: Any, expected_len: Optional[int] = None
) -> List[float]:
    if not isinstance(v, (list, tuple)):
        raise EstimatorError(f"{name} must be a list/tuple, got {type(v).__name__}")
    if expected_len is not None and len(v) != expected_len:
        raise EstimatorError(
            f"{name} must have length {expected_len}, got {len(v)}"
        )
    return [_require_finite_scalar(f"{name}[{i}]", val) for i, val in enumerate(v)]


def _require_finite_matrix(
    name: str, M: Any, expected_shape: Optional[Tuple[int, int]] = None
) -> List[List[float]]:
    if not isinstance(M, (list, tuple)) or (M and not isinstance(M[0], (list, tuple))):
        raise EstimatorError(f"{name} must be a 2D list of lists, got {M!r}")
    rows = len(M)
    cols = len(M[0]) if rows else 0
    for row in M:
        if len(row) != cols:
            raise EstimatorError(f"{name} has inconsistent row lengths.")
    if expected_shape is not None:
        exp_rows, exp_cols = expected_shape
        if rows != exp_rows or cols != exp_cols:
            raise EstimatorError(
                f"{name} must be {exp_rows}x{exp_cols}, got {rows}x{cols}"
            )
    return [
        [_require_finite_scalar(f"{name}[{i}][{j}]", val) for j, val in enumerate(row)]
        for i, row in enumerate(M)
    ]


def _require_covariance(
    name: str, M: Any, expected_n: Optional[int] = None
) -> List[List[float]]:
    """Validates M as a square, finite, symmetric, positive-semidefinite
    covariance matrix. Zero matrices (e.g. Q=0 or R=0 for a deterministic
    test) are accepted -- PSD, not strict PD, is what a covariance matrix
    must satisfy in general."""
    rows = len(M) if isinstance(M, (list, tuple)) else 0
    shape = (expected_n, expected_n) if expected_n is not None else None
    M = _require_finite_matrix(name, M, expected_shape=shape)
    n = len(M)
    if n == 0 or len(M[0]) != n:
        raise EstimatorError(f"{name} must be square, got shape {n}x{len(M[0]) if M else 0}")
    if not _is_symmetric(M):
        raise EstimatorError(f"{name} must be symmetric (within {_SYMMETRY_TOL}).")
    if not _is_positive_semidefinite(M):
        raise EstimatorError(f"{name} must be positive semi-definite.")
    return M


# ---------------------------------------------------------------------------
# Provenance / output data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EstimatorProvenance:
    """Machine-readable provenance record attached to every estimator and
    every EstimatedState it produces (mirrors the pattern used by
    models/conversion.py's ProjectDerivedDimensionalModel)."""

    estimator_method: str = ESTIMATOR_METHOD
    nasa_reference_method_status: str = NASA_REFERENCE_METHOD_STATUS
    nasa_reference_citation: str = NASA_REFERENCE_METHOD_CITATION
    transition_model_provenance: str = "PROJECT_DEFINED_PARAMETER"
    measurement_model_provenance: str = "PROJECT_DEFINED_MEASUREMENT_MODEL"
    process_noise_provenance: str = "PROJECT_DEFINED_PARAMETER"
    measurement_noise_provenance: str = "PROJECT_DEFINED_PARAMETER"
    initial_covariance_provenance: str = "PROJECT_DEFINED_PARAMETER"
    notes: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EstimatedState:
    """An ESTIMATE of the project's reduced-order state -- NOT the true
    simulator state produced by models/flight_dynamics.py. Field names
    (`delta_w_hat`, `delta_q_hat`) are deliberately distinct from the truth
    field names (`delta_w`, `delta_q`) used elsewhere in the project so the
    two can never be confused by a shared key name."""

    delta_w_hat: float
    delta_q_hat: float
    covariance: List[List[float]]
    estimator_status: str
    step_index: int
    provenance: EstimatorProvenance

    def as_dict(self) -> Dict[str, Any]:
        return {
            "delta_w_hat": self.delta_w_hat,
            "delta_q_hat": self.delta_q_hat,
            "covariance": [row[:] for row in self.covariance],
            "estimator_status": self.estimator_status,
            "step_index": self.step_index,
            "provenance": self.provenance,
        }


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class KalmanStateEstimator:
    """
    PROJECT_DEFINED_REDUCED_ORDER_LINEAR_KALMAN_FILTER for the 2-state
    project model x = [delta_w, delta_q].

    Implements the standard linear discrete Kalman filter equations
    exactly as specified for this project:

        Prediction:
            x_k^- = F x_{k-1} + B u_k
            P_k^- = F P_{k-1} F^T + Q

        Update:
            y_k = z_k - H x_k^-
            S_k = H P_k^- H^T + R
            K_k = P_k^- H^T S_k^{-1}
            x_k = x_k^- + K_k y_k
            P_k = (I - K_k H) P_k^-

    F, H, Q, R, x0, and P0 are all explicit constructor arguments -- this
    class never silently invents a transition or noise model. See the
    module-level helpers (`identity_transition_matrix`,
    `default_process_noise`, `build_measurement_noise_from_sensor_suite`,
    `build_transition_from_flight_dynamics`) for documented, provenance-
    tagged ways to obtain each matrix; `default_reduced_order_estimator`
    assembles a complete, clearly-labeled baseline instance from them.
    """

    def __init__(
        self,
        F: Sequence[Sequence[float]],
        H: Sequence[Sequence[float]],
        Q: Sequence[Sequence[float]],
        R: Sequence[Sequence[float]],
        x0: Sequence[float],
        P0: Sequence[Sequence[float]],
        B: Optional[Sequence[Sequence[float]]] = None,
        provenance: Optional[EstimatorProvenance] = None,
    ):
        self.F = _require_finite_matrix("F", F, expected_shape=(STATE_DIM, STATE_DIM))
        self.H = _require_finite_matrix("H", H)
        if len(self.H[0]) != STATE_DIM:
            raise EstimatorError(
                f"H must have {STATE_DIM} columns (one per state), got {len(self.H[0])}"
            )
        measurement_dim = len(self.H)
        if measurement_dim < 1:
            raise EstimatorError("H must have at least one measurement row.")

        self.Q = _require_covariance("Q", Q, expected_n=STATE_DIM)
        self.R = _require_covariance("R", R, expected_n=measurement_dim)
        self.x = _require_finite_vector("x0", x0, expected_len=STATE_DIM)
        self.P = _require_covariance("P0", P0, expected_n=STATE_DIM)

        if B is None:
            B = [[0.0] for _ in range(STATE_DIM)]
        self.B = _require_finite_matrix("B", B, expected_shape=(STATE_DIM, 1))

        self.measurement_dim = measurement_dim
        self.provenance = provenance if provenance is not None else EstimatorProvenance()
        self._step_index = 0

    # -- core Kalman filter cycle -------------------------------------

    def predict(self, dt: float, u: float = 0.0) -> Tuple[List[float], List[List[float]]]:
        """Time-update: propagates (x, P) forward by dt under (F, B, Q)."""
        dt = _require_finite_scalar("dt", dt)
        if dt <= 0.0:
            raise EstimatorError(f"dt must be a positive finite number, got {dt}")
        u = _require_finite_scalar("u", u)

        x_pred = _mat_vec_mult(self.F, self.x)
        Bu = [self.B[i][0] * u for i in range(STATE_DIM)]
        x_pred = _vec_add(x_pred, Bu)

        FP = _mat_mult(self.F, self.P)
        FPFt = _mat_mult(FP, _mat_transpose(self.F))
        P_pred = _symmetrize(_mat_add(FPFt, self.Q))

        self.x = x_pred
        self.P = P_pred
        return self.x, self.P

    def update(self, z: Sequence[float]) -> Tuple[List[float], List[List[float]]]:
        """Measurement-update: corrects (x, P) using measurement z."""
        z = _require_finite_vector("z", z, expected_len=self.measurement_dim)

        Hx = _mat_vec_mult(self.H, self.x)
        y = _vec_sub(z, Hx)

        HP = _mat_mult(self.H, self.P)
        HPHt = _mat_mult(HP, _mat_transpose(self.H))
        S = _mat_add(HPHt, self.R)

        try:
            S_inv = _mat_inverse(S)
        except EstimatorError as exc:
            raise EstimatorError(
                "Innovation covariance S is singular; cannot compute the "
                "Kalman gain. This usually means H, P, and R together leave "
                "some measurement direction with zero total variance."
            ) from exc

        PHt = _mat_mult(self.P, _mat_transpose(self.H))
        K = _mat_mult(PHt, S_inv)

        Ky = _mat_vec_mult(K, y)
        self.x = _vec_add(self.x, Ky)

        KH = _mat_mult(K, self.H)
        I_KH = _mat_sub(_identity(STATE_DIM), KH)
        self.P = _symmetrize(_mat_mult(I_KH, self.P))

        self._step_index += 1
        return self.x, self.P

    def step(
        self, z: Sequence[float], dt: float, u: float = 0.0, status: str = "OK"
    ) -> EstimatedState:
        """Convenience: one predict() followed by one update(), returning
        the resulting EstimatedState."""
        self.predict(dt, u=u)
        self.update(z)
        return self.get_estimated_state(status=status)

    def get_estimated_state(self, status: str = "OK") -> EstimatedState:
        return EstimatedState(
            delta_w_hat=self.x[0],
            delta_q_hat=self.x[1],
            covariance=[row[:] for row in self.P],
            estimator_status=status,
            step_index=self._step_index,
            provenance=self.provenance,
        )


# ---------------------------------------------------------------------------
# Documented, provenance-tagged helpers for assembling F/H/Q/R/P0
# ---------------------------------------------------------------------------


def identity_transition_matrix() -> List[List[float]]:
    """F = I: a PROJECT_DEFINED_PARAMETER RANDOM_WALK_ASSUMPTION.

    Assumes delta_w and delta_q are (locally, over one dt) unchanged in the
    absence of a linearized dynamics model, i.e. all evolution between
    updates is absorbed into Q. This is NOT a claim that the aircraft's
    short-period dynamics are a random walk -- it is the simplest possible
    placeholder transition, appropriate as a baseline and for the
    project-required "simple known linear system" analytic test. Use
    `build_transition_from_flight_dynamics` for a physically-informed F
    once models/flight_dynamics.py + models/aerodynamics.py are available.
    """
    return _identity(STATE_DIM)


def identity_measurement_matrix() -> List[List[float]]:
    """H = I: measured_delta_w and measured_delta_q are treated as direct
    (noisy) observations of delta_w and delta_q -- the minimum-required
    measurement model described in the module docstring."""
    return _identity(STATE_DIM)


def default_process_noise() -> List[List[float]]:
    """PROJECT_DEFINED_PARAMETER conservative process-noise covariance Q.

    No sourced value exists for how much the linearization/model-mismatch
    error should be per step, so these are deliberately small, round
    placeholder numbers (not zero, so the filter can track slow drift the
    transition model misses) -- NOT independently validated or NASA-sourced.
    Units: (m/s)^2 for delta_w, (rad/s)^2 for delta_q.
    """
    return [[1.0e-4, 0.0], [0.0, 1.0e-6]]


def default_initial_covariance() -> List[List[float]]:
    """PROJECT_DEFINED_PARAMETER conservative initial covariance P0.

    Deliberately large relative to `default_process_noise` so the filter
    leans on early measurements rather than an assumed-precise initial
    state. Units: (m/s)^2 for delta_w, (rad/s)^2 for delta_q.
    """
    return [[1.0, 0.0], [0.0, 1.0e-2]]


_SENSOR_CHANNEL_NAMES = ("w_sensor", "rate_gyro_q", "accelerometer_az")


def build_measurement_noise_from_sensor_suite(
    suite: Optional[SensorSuiteParameters] = None,
    channels: Sequence[str] = ("w_sensor", "rate_gyro_q"),
) -> Tuple[List[List[float]], str]:
    """Builds a diagonal measurement-noise covariance R directly from the
    noise_std values already validated inside
    sensors.sensor_model.SensorSuiteParameters, rather than re-inventing
    noise numbers independently.

    This reads sensor_model.py's existing channel parameters; it does not
    reimplement its noise/bias logic. `bias` is NOT folded into R -- a
    Kalman filter's R models zero-mean measurement noise, and a nonzero
    sensor bias is a separate (unmodeled, in this first estimator) error
    source, consistent with "do not silently hard-code hidden covariance".

    Returns:
        (R, provenance_string) -- provenance_string names each channel and
        echoes its own `model_origin` (e.g. PROJECT_DEFINED_PLACEHOLDER),
        so R's provenance is traceable back to sensor_model.py's.
    """
    if suite is None:
        suite = default_sensor_suite()
    if not isinstance(suite, SensorSuiteParameters):
        raise EstimatorError(
            f"suite must be a SensorSuiteParameters, got {type(suite).__name__}"
        )

    channel_map = {
        "w_sensor": suite.w_sensor,
        "rate_gyro_q": suite.rate_gyro_q,
        "accelerometer_az": suite.accelerometer_az,
    }

    variances: List[float] = []
    origin_parts: List[str] = []
    for name in channels:
        if name not in channel_map:
            raise EstimatorError(
                f"Unknown sensor channel '{name}'; expected one of "
                f"{_SENSOR_CHANNEL_NAMES}"
            )
        channel = channel_map[name]
        variances.append(channel.noise_std ** 2)
        origin_parts.append(f"{name}<-{channel.model_origin}")

    n = len(variances)
    R = [[variances[i] if i == j else 0.0 for j in range(n)] for i in range(n)]
    provenance_str = "DERIVED_FROM_SENSOR_MODEL(" + ", ".join(origin_parts) + ")"
    return R, provenance_str


def default_reduced_order_estimator(
    x0: Sequence[float] = (0.0, 0.0),
    sensor_suite: Optional[SensorSuiteParameters] = None,
) -> KalmanStateEstimator:
    """Assembles a complete, clearly-labeled baseline
    KalmanStateEstimator:

        F  = identity_transition_matrix()   (RANDOM_WALK_ASSUMPTION)
        H  = identity_measurement_matrix()  (direct delta_w/delta_q meas.)
        Q  = default_process_noise()        (PROJECT_DEFINED_PARAMETER)
        R  = build_measurement_noise_from_sensor_suite(sensor_suite)
        P0 = default_initial_covariance()   (PROJECT_DEFINED_PARAMETER)

    This is the estimator to reach for when no linearized short-period
    transition matrix has been wired in yet (e.g. models/aerodynamics.py is
    not available in the running environment). Once a real F is available,
    construct `KalmanStateEstimator` directly with
    `build_transition_from_flight_dynamics`'s output instead.
    """
    R, r_provenance = build_measurement_noise_from_sensor_suite(sensor_suite)
    provenance = EstimatorProvenance(
        transition_model_provenance=(
            "PROJECT_DEFINED_PARAMETER: RANDOM_WALK_ASSUMPTION (identity F; "
            "no linearized short-period dynamics wired in -- see "
            "build_transition_from_flight_dynamics)"
        ),
        measurement_noise_provenance=r_provenance,
    )
    return KalmanStateEstimator(
        F=identity_transition_matrix(),
        H=identity_measurement_matrix(),
        Q=default_process_noise(),
        R=R,
        x0=list(x0),
        P0=default_initial_covariance(),
        provenance=provenance,
    )


def build_transition_from_flight_dynamics(
    flight_dynamics: Any,
    mass: float,
    iyy: float,
    derivatives: Any,
    dt: float,
    nominal_delta_e: float = 0.0,
    epsilon: float = 1.0e-4,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Builds a physically-informed discrete transition matrix F (and
    control matrix B) by numerically differentiating the REAL
    models.flight_dynamics.FlightDynamics.calculate_state_derivatives()
    around a nominal operating point, then discretizing with a first-order
    (explicit Euler) approximation:

        F ~= I + dt * df/dx |_nominal
        B ~= dt * df/d(delta_e) |_nominal

    This function does NOT reimplement or re-derive the short-period
    equations of motion -- it repeatedly *calls* the existing
    FlightDynamics.calculate_state_derivatives(), so the Zw/Zq/Zde/Mw/Mq/Mde
    physics stays owned entirely by models/flight_dynamics.py (and, one
    layer further back, models/conversion.py). Because the short-period
    model is linear in (delta_w, delta_q, delta_e), this finite-difference
    Jacobian is exact (no linearization truncation error) regardless of the
    nominal point chosen; `nominal_delta_e` only matters if a future,
    non-linear flight_dynamics model replaces the current one.

    PROVENANCE: EULER_FIRST_ORDER_DISCRETIZATION is a PROJECT_DEFINED
    choice, appropriate for small dt. It is not an exact zero-order-hold or
    matrix-exponential discretization and is not independently validated.

    This performs a LAZY import of models.flight_dynamics so that importing
    digital_twin.estimator never requires models.aerodynamics (a transitive
    dependency of models.flight_dynamics) to be present -- only calling this
    specific function does.

    Args:
        flight_dynamics: an already-constructed
            models.flight_dynamics.FlightDynamics instance (bound to a
            ReferenceTrim).
        mass, iyy: aircraft mass (kg) and pitch inertia (kg*m^2), e.g. from
            Aircraft.mass_properties.
        derivatives: a models.flight_dynamics.ShortPeriodDerivatives
            instance, bound to the same FlightCondition as `flight_dynamics`.
        dt: discretization timestep (s); must be positive and finite.
        nominal_delta_e: elevator perturbation (rad) about which to
            linearize; immaterial for the current linear model (see above).
        epsilon: finite-difference step size.

    Returns:
        (F, B): a (2x2) discrete transition matrix and a (2x1) discrete
        control matrix, suitable for `KalmanStateEstimator(F=..., ...,
        B=...)`.

    Raises:
        EstimatorError: if models.flight_dynamics (or its own
            models.aerodynamics dependency) cannot be imported, or if any
            input is invalid.
    """
    try:
        from models.flight_dynamics import PerturbationState  # lazy, see above
    except ImportError as exc:
        raise EstimatorError(
            "build_transition_from_flight_dynamics requires "
            "models.flight_dynamics (and, transitively, "
            "models.aerodynamics) to be importable in this environment."
        ) from exc

    dt = _require_finite_scalar("dt", dt)
    if dt <= 0.0:
        raise EstimatorError(f"dt must be positive and finite, got {dt}")
    mass = _require_finite_scalar("mass", mass)
    iyy = _require_finite_scalar("iyy", iyy)
    nominal_delta_e = _require_finite_scalar("nominal_delta_e", nominal_delta_e)
    eps = _require_finite_scalar("epsilon", epsilon)
    if eps <= 0.0:
        raise EstimatorError(f"epsilon must be positive and finite, got {eps}")

    def f(dw: float, dq: float, de: float) -> Tuple[float, float]:
        state = PerturbationState(delta_w=dw, delta_q=dq)
        result = flight_dynamics.calculate_state_derivatives(
            mass, iyy, state, derivatives, de
        )
        d = result["derivatives"]
        return d["delta_w_dot"], d["delta_q_dot"]

    w0, q0, de0 = 0.0, 0.0, nominal_delta_e

    fw_plus = f(w0 + eps, q0, de0)
    fw_minus = f(w0 - eps, q0, de0)
    fq_plus = f(w0, q0 + eps, de0)
    fq_minus = f(w0, q0 - eps, de0)
    fde_plus = f(w0, q0, de0 + eps)
    fde_minus = f(w0, q0, de0 - eps)

    dwdot_dw = (fw_plus[0] - fw_minus[0]) / (2.0 * eps)
    dqdot_dw = (fw_plus[1] - fw_minus[1]) / (2.0 * eps)
    dwdot_dq = (fq_plus[0] - fq_minus[0]) / (2.0 * eps)
    dqdot_dq = (fq_plus[1] - fq_minus[1]) / (2.0 * eps)
    dwdot_de = (fde_plus[0] - fde_minus[0]) / (2.0 * eps)
    dqdot_de = (fde_plus[1] - fde_minus[1]) / (2.0 * eps)

    A = [[dwdot_dw, dwdot_dq], [dqdot_dw, dqdot_dq]]
    Bc = [[dwdot_de], [dqdot_de]]

    F = _mat_add(_identity(STATE_DIM), _scalar_mult(dt, A))
    B = _scalar_mult(dt, Bc)
    return F, B


# ---------------------------------------------------------------------------
# Small adapter: SensorModel.measure() dict -> flat measurement vector
# ---------------------------------------------------------------------------


def extract_measurement_from_sensor_output(
    sensor_output: Dict[str, Any],
    channels: Sequence[str] = ("delta_w_mps", "delta_q_rads"),
) -> List[float]:
    """Small, clean adapter from sensors.sensor_model.SensorModel.measure()'s
    dict output to the flat measurement vector KalmanStateEstimator.update()
    expects. Does not touch sensor_model.py's noise/bias logic -- it only
    reads the already-computed 'measured' values out of its return value.
    """
    if not isinstance(sensor_output, dict) or "measured" not in sensor_output:
        raise EstimatorError(
            "sensor_output must be the dict returned by "
            "sensors.sensor_model.SensorModel.measure() (missing 'measured' key)."
        )
    measured = sensor_output["measured"]
    try:
        return [_require_finite_scalar(ch, measured[ch]) for ch in channels]
    except KeyError as exc:
        raise EstimatorError(
            f"sensor_output['measured'] is missing expected channel {exc}"
        ) from exc


def extract_measurement_from_history_row(
    row: Dict[str, Any],
    channels: Sequence[str] = ("delta_w_mps", "delta_q_rads"),
) -> List[float]:
    """Adapter from a simulation/simulator.py history row (see main.py's
    use of history[-1]['delta_w_mps'] / ['delta_q_rads']) to the flat
    measurement vector KalmanStateEstimator.update() expects. Intended for
    driving the estimator directly from noise-free truth (e.g. in a smoke
    test), not as a substitute for going through SensorModel when noisy
    measurements are what is actually available.
    """
    try:
        return [_require_finite_scalar(ch, row[ch]) for ch in channels]
    except KeyError as exc:
        raise EstimatorError(f"history row is missing expected key {exc}") from exc