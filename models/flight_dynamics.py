"""
models/flight_dynamics.py

Purpose: Implement the NASA T-2 short-period longitudinal equations of
motion, converting dimensional stability/control derivatives into state
derivatives (dx/dt) for the perturbation state x = [w, q]^T.

Design decision (per review): this module implements Option A directly.

    NASA T-2 short-period model
        -> x = [w, q]
        -> M' xdot = A' x + B' delta_e
        -> wdot, qdot

Scope (explicit):
  - Longitudinal SHORT-PERIOD perturbation dynamics only (2-state: w, q).
  - Gravity/trim-force balance and full 4-state longitudinal coupling
    (Ue, theta_e appearing as separate gravity terms) are NOT part of this
    2-state formulation and are intentionally excluded. They belong to a
    future 4-state longitudinal model, if/when that formulation and its
    NASA-documented derivatives are verified and implemented.
  - Propulsion is NOT part of this 2-state short-period formulation and is
    not accepted as an input here. A thrust perturbation that is computed
    but never used in the state equations is worse than no thrust input at
    all, so it has been removed rather than kept as a decorative parameter.
  - This module does NOT convert aerodynamic coefficients (delta CL, delta
    Cm) into dimensional stability derivatives (Zw, Zq, Mw, Mq, Zde, Mde).
    That mapping is currently unverified and undocumented and must be
    implemented explicitly elsewhere (e.g. in aerodynamics.py or a
    dedicated conversion layer) before this module can be driven from
    CL/Cm outputs. Until then, callers must supply project-derived dimensional
    derivatives directly via ShortPeriodDerivatives.
  - Numerical integration is intentionally excluded; this module outputs
    instantaneous state derivatives (dx/dt) for simulation/simulator.py to
    integrate. This is a state-derivative / equations-of-motion layer, not
    an integrator.

Coordinate Conventions:
NED:  x = North,   y = East,    z = Down
Body: x = forward, y = right,   z = down
"""

import math
from dataclasses import dataclass
from typing import Optional

from .aerodynamics import FlightCondition


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _require_finite(name: str, value) -> float:
    """Validate that *value* is a finite number (not bool). Returns float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a number, got {type(value).__name__}"
        )
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ReferenceTrim:
    """
    Explicitly defines the reference/trim point for the perturbation model.
    The short-period dynamics operate as perturbations (Delta x) around this
    state.

    Only u0 (trim forward airspeed, Ue) is used by the 2-state short-period
    matrix equation implemented here -- it appears in the A' coupling term
    (Zq + m*Ue). alpha0, theta0, and thrust0 are deliberately NOT stored on
    this object: they played no verified role in the 2-state formulation
    (alpha0 and thrust0 were unused, and theta0 was only used to construct
    an unrelated trim-force/gravity balance that does not belong in the
    short-period equations). Reintroduce them, with a documented role in the
    matrix equation, only if/when a 4-state longitudinal model or a
    propulsion-coupled model is implemented.
    """

    def __init__(self, flight_condition: FlightCondition, u0: float):
        if not isinstance(flight_condition, FlightCondition):
            raise TypeError(
                f"flight_condition must be a FlightCondition enum member, "
                f"got {type(flight_condition).__name__}"
            )

        u0 = _require_finite("u0", u0)
        if u0 <= 0:
            raise ValueError(
                "Trim airspeed (u0) must be strictly positive."
            )

        self.flight_condition = flight_condition
        self.u0 = u0  # Reference forward airspeed Ue (m/s)


@dataclass(frozen=True)
class ShortPeriodDerivatives:
    """
    Dimensional stability and control derivatives for the NASA short-period
    matrix equation:

        M' xdot = A' x + B' delta_e,   x = [w, q]^T

    These are DIMENSIONAL derivatives (e.g. N per (m/s) for Zw, N per (rad/s)
    for Zq, N per rad for Zde, and the corresponding N*m units for the M
    derivatives) -- NOT the non-dimensional CL/Cm coefficients produced by
    aerodynamics.py. There is currently no verified, documented mapping from
    CL/Cm to these derivatives (see design review); until one exists,
    callers must supply project-derived dimensional derivatives directly (e.g. from
    the NASA source document), rather than this module inventing one.

    Each derivative record is bound to a specific FlightCondition so that
    Flight 41 derivatives cannot accidentally be paired with a Flight 15
    trim condition.
    """
    flight_condition: FlightCondition

    Zw: float    # dZ/dw               (N per m/s)
    Zq: float    # dZ/dq               (N per rad/s)
    Zde: float   # dZ/d(delta_e)       (N per rad)
    Mw: float    # dM/dw               (N*m per m/s)
    Mq: float    # dM/dq               (N*m per rad/s)
    Mde: float   # dM/d(delta_e)       (N*m per rad)

    def __post_init__(self):
        if not isinstance(self.flight_condition, FlightCondition):
            raise TypeError(
                f"flight_condition must be a FlightCondition enum member, "
                f"got {type(self.flight_condition).__name__}"
            )
        for name in ("Zw", "Zq", "Zde", "Mw", "Mq", "Mde"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(
                    f"Stability/control derivative {name} must be a number, "
                    f"got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise ValueError(
                    f"Stability/control derivative {name} must be finite, "
                    f"got {val}"
                )


class PerturbationState:
    """
    NASA short-period perturbation state vector: x = [delta_w, delta_q]^T.

      delta_w: perturbation body-axis vertical velocity (m/s)
      delta_q: perturbation pitch rate (rad/s)

    x = x_trim + Delta x. Do not treat delta_w or delta_q as arbitrary
    absolute aircraft states.
    """

    def __init__(self, delta_w: float, delta_q: float):
        self.delta_w = _require_finite("delta_w", delta_w)
        self.delta_q = _require_finite("delta_q", delta_q)


# ---------------------------------------------------------------------------
# Flight Dynamics (equations of motion)
# ---------------------------------------------------------------------------


class FlightDynamics:
    """
    State-derivative / equations-of-motion layer for the NASA T-2
    longitudinal short-period dynamics. Numerical integration is performed
    elsewhere (simulation/simulator.py); this class only computes dx/dt.

    Implements, directly, the NASA short-period matrix equation:

        M' xdot = A' x + B' delta_e

        M' = [[m,  0],
              [0, Iy]]

        A' = [[Zw, Zq + m*Ue],
              [Mw,  Mq       ]]

        B' = [Zde, Mde]^T

        x = [w, q]^T

    Inputs expected from outer modules:
      - aircraft.py -> mass, pitch inertia (Iyy)
      - a project-derived source of dimensional derivatives (ShortPeriodDerivatives)
      - elevator perturbation delta_e (rad)

    Propulsion and gravity/trim-force terms are explicitly excluded (see
    module docstring) rather than approximated or passed through unused.
    """

    def __init__(self, trim_condition: ReferenceTrim):
        if not isinstance(trim_condition, ReferenceTrim):
            raise TypeError(
                f"trim_condition must be a ReferenceTrim, "
                f"got {type(trim_condition).__name__}"
            )
        self.trim = trim_condition

    def calculate_state_derivatives(
        self,
        mass: float,
        iyy: float,
        state: PerturbationState,
        derivatives: ShortPeriodDerivatives,
        delta_e: float,
        w_gust: float = 0.0,
    ) -> dict:
        """
        Computes dx/dt = [wdot, qdot] directly from the NASA short-period
        matrix equation M' xdot = A' x + B' delta_e, plus the perturbation
        specific-force output in body-Z.

        Args:
            mass: Aircraft mass (kg). Must be positive, finite.
            iyy: Pitch moment of inertia (kg*m^2). Must be positive, finite.
            state: Current perturbation state [delta_w, delta_q].
            derivatives: Project-derived dimensional stability/control derivatives,
                bound to the same FlightCondition as this model's trim.
            delta_e: Elevator perturbation (rad). Must be finite.
            w_gust: Optional vertical gust velocity (m/s), body-z downward.
                Defaults to 0.0 (unperturbed atmosphere).
                Modifies relative aerodynamic vertical velocity:
                    delta_w_aero = delta_w - w_gust
                Affects only aerodynamic Zw and Mw terms.

        Returns:
            dict with 'derivatives', 'outputs', and 'diagnostics' sub-dicts.

        Raises:
            ValueError: If any input is non-finite, non-positive mass/iyy,
                or flight-condition mismatch between trim and derivatives.
        """
        # 1. Validation — mass, iyy, delta_e, w_gust
        mass = _require_finite("mass", mass)
        if mass <= 0.0:
            raise ValueError("Aircraft mass must be a positive finite value.")

        iyy = _require_finite("iyy", iyy)
        if iyy <= 0.0:
            raise ValueError(
                "Aircraft Iyy (pitch inertia) must be a positive finite value."
            )

        delta_e = _require_finite("delta_e", delta_e)
        w_gust = _require_finite("w_gust", w_gust)

        # 2. Validation — flight-condition binding
        if not isinstance(derivatives, ShortPeriodDerivatives):
            raise TypeError(
                f"derivatives must be ShortPeriodDerivatives, "
                f"got {type(derivatives).__name__}"
            )
        if derivatives.flight_condition != self.trim.flight_condition:
            raise ValueError(
                f"Flight-condition mismatch: trim is "
                f"{self.trim.flight_condition.name} but derivatives are "
                f"{derivatives.flight_condition.name}. Mixing Flight 41 "
                f"trim with Flight 15 derivatives (or vice versa) is not "
                f"permitted."
            )

        Ue = self.trim.u0

        # 3. Aerodynamic relative vertical velocity perturbation:
        #    delta_w_aero = delta_w - w_gust
        #    Body-axis vertical velocity state delta_w is preserved.
        delta_w_aero = state.delta_w - w_gust

        # 4. Perturbation Z-force and M-moment (dimensional).
        #    delta_w_aero enters the aerodynamic Zw and Mw terms.
        #    Elevator, pitch rate, and kinematic terms remain unchanged.
        delta_Z_force_N = (
            derivatives.Zw * delta_w_aero
            + derivatives.Zq * state.delta_q
            + derivatives.Zde * delta_e
        )
        delta_M_pitch_Nm = (
            derivatives.Mw * delta_w_aero
            + derivatives.Mq * state.delta_q
            + derivatives.Mde * delta_e
        )

        # 5. NASA short-period equations of motion: M' xdot = A' x + B' delta_e
        #    Row 1: m * wdot = Z + m * Ue * q  -->  wdot = Z/m + Ue * q
        delta_w_dot = delta_Z_force_N / mass + Ue * state.delta_q
        #    Row 2: Iy * qdot = M              -->  qdot = M / Iy
        delta_q_dot = delta_M_pitch_Nm / iyy

        # 6. Perturbation specific force in body-Z (what an accelerometer at
        #    the CG measures due to the modeled perturbation Z-force). This
        #    deliberately excludes the Ue*q kinematic coupling term, since an
        #    accelerometer measures applied specific force, not total
        #    kinematic acceleration. This is the NASA a_z output convention
        #    for the short-period model.
        perturbation_az = delta_Z_force_N / mass

        # 7. Verify that computed outputs are finite (guards against
        #    extreme-but-valid finite inputs causing overflow).
        for name, val in (
            ("delta_w_dot", delta_w_dot),
            ("delta_q_dot", delta_q_dot),
            ("perturbation_az", perturbation_az),
        ):
            if not math.isfinite(val):
                raise ValueError(
                    f"Computed output {name} is not finite ({val!r}). "
                    f"Check inputs for extreme values."
                )

        return {
            "derivatives": {
                "delta_w_dot": delta_w_dot,
                "delta_q_dot": delta_q_dot,
            },
            "outputs": {
                "delta_w": state.delta_w,
                "delta_q": state.delta_q,
                "perturbation_az": perturbation_az,
            },
            "diagnostics": {
                "delta_Z_force_N": delta_Z_force_N,
                "delta_M_pitch_Nm": delta_M_pitch_Nm,
                "w_gust_mps": w_gust,
                "delta_w_aero_mps": delta_w_aero,
            },
        }

