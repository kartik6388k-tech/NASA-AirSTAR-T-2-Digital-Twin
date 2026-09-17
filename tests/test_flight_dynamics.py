"""
tests/test_flight_dynamics.py

Tests for models/flight_dynamics.py.

All TEST_ONLY values below (mass, iyy, u0, derivatives) are chosen to
exercise the matrix-equation implementation. They are NOT verified NASA
T-2 parameters.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.aerodynamics import FlightCondition
from models.flight_dynamics import (
    FlightDynamics,
    PerturbationState,
    ReferenceTrim,
    ShortPeriodDerivatives,
)


# --------------------------------------------------------------------------
# Shared TEST_ONLY fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def trim_f41():
    return ReferenceTrim(
        flight_condition=FlightCondition.FLIGHT_41,
        u0=50.0,  # Ue = 50 m/s, TEST_ONLY
    )


@pytest.fixture
def fd(trim_f41):
    return FlightDynamics(trim_f41)


@pytest.fixture
def derivs_f41():
    return ShortPeriodDerivatives(
        flight_condition=FlightCondition.FLIGHT_41,
        Zw=-50.0, Zq=-5.0, Zde=-1000.0,
        Mw=-2.0, Mq=-10.0, Mde=-500.0,
    )  # TEST_ONLY, not verified T-2 derivatives


MASS = 1000.0   # kg, TEST_ONLY
IYY = 2000.0    # kg*m^2, TEST_ONLY


# --------------------------------------------------------------------------
# Hand-calculation tests
# --------------------------------------------------------------------------

def test_zero_perturbation_yields_zero_derivatives(fd, derivs_f41):
    """Zero perturbation, zero control -> zero response."""
    state_zero = PerturbationState(0.0, 0.0)
    res = fd.calculate_state_derivatives(
        mass=MASS, iyy=IYY, state=state_zero, derivatives=derivs_f41, delta_e=0.0
    )
    assert res['derivatives']['delta_w_dot'] == 0.0
    assert res['derivatives']['delta_q_dot'] == 0.0
    assert res['outputs']['perturbation_az'] == 0.0


def test_hand_calculation_nonzero_state(fd, derivs_f41):
    """
    Non-zero state and control -> matches hand calculation of
    M' xdot = A' x + B' delta_e.

    x = [w, q] = [2.0, 0.1], delta_e = 0.05

    Z = Zw*w + Zq*q + Zde*de
      = (-50)(2) + (-5)(0.1) + (-1000)(0.05) = -150.5

    M = Mw*w + Mq*q + Mde*de
      = (-2)(2) + (-10)(0.1) + (-500)(0.05) = -30.0

    wdot = Z/m + Ue*q = -0.1505 + 5.0 = 4.8495
    qdot = M/Iy = -30.0/2000.0 = -0.015
    a_z  = Z/m = -0.1505
    """
    state = PerturbationState(delta_w=2.0, delta_q=0.1)
    res = fd.calculate_state_derivatives(
        mass=MASS, iyy=IYY, state=state, derivatives=derivs_f41, delta_e=0.05
    )
    assert math.isclose(res['diagnostics']['delta_Z_force_N'], -150.5, rel_tol=1e-9)
    assert math.isclose(res['diagnostics']['delta_M_pitch_Nm'], -30.0, rel_tol=1e-9)
    assert math.isclose(res['derivatives']['delta_w_dot'], 4.8495, rel_tol=1e-9)
    assert math.isclose(res['derivatives']['delta_q_dot'], -0.015, rel_tol=1e-9)
    assert math.isclose(res['outputs']['perturbation_az'], -0.1505, rel_tol=1e-9)


def test_matrix_equation_balance(fd, derivs_f41):
    """Verify M' xdot == A' x + B' delta_e independently."""
    state = PerturbationState(delta_w=2.0, delta_q=0.1)
    res = fd.calculate_state_derivatives(
        mass=MASS, iyy=IYY, state=state, derivatives=derivs_f41, delta_e=0.05
    )
    wdot = res['derivatives']['delta_w_dot']
    qdot = res['derivatives']['delta_q_dot']

    # M' xdot
    Mprime_xdot_row1 = MASS * wdot
    Mprime_xdot_row2 = IYY * qdot

    # A' x + B' delta_e
    rhs_row1 = (derivs_f41.Zw * 2.0
                + (derivs_f41.Zq + MASS * 50.0) * 0.1
                + derivs_f41.Zde * 0.05)
    rhs_row2 = (derivs_f41.Mw * 2.0
                + derivs_f41.Mq * 0.1
                + derivs_f41.Mde * 0.05)

    assert math.isclose(Mprime_xdot_row1, rhs_row1, rel_tol=1e-9)
    assert math.isclose(Mprime_xdot_row2, rhs_row2, rel_tol=1e-9)


# --------------------------------------------------------------------------
# Input-validation tests
# --------------------------------------------------------------------------

def test_negative_mass_rejected(fd, derivs_f41):
    state_zero = PerturbationState(0.0, 0.0)
    with pytest.raises(ValueError):
        fd.calculate_state_derivatives(
            mass=-10.0, iyy=IYY, state=state_zero,
            derivatives=derivs_f41, delta_e=0.0,
        )


def test_nan_trim_airspeed_rejected():
    with pytest.raises(ValueError):
        ReferenceTrim(
            flight_condition=FlightCondition.FLIGHT_41,
            u0=float('nan'),
        )


def test_non_positive_trim_airspeed_rejected():
    with pytest.raises(ValueError):
        ReferenceTrim(
            flight_condition=FlightCondition.FLIGHT_41,
            u0=0.0,
        )


def test_nan_stability_derivative_rejected():
    with pytest.raises(ValueError):
        ShortPeriodDerivatives(
            flight_condition=FlightCondition.FLIGHT_41,
            Zw=float('nan'), Zq=-5.0, Zde=-1000.0,
            Mw=-2.0, Mq=-10.0, Mde=-500.0,
        )


def test_inf_stability_derivative_rejected():
    with pytest.raises(ValueError):
        ShortPeriodDerivatives(
            flight_condition=FlightCondition.FLIGHT_41,
            Zw=-50.0, Zq=-5.0, Zde=-1000.0,
            Mw=-2.0, Mq=float('inf'), Mde=-500.0,
        )


def test_flight_condition_mismatch_rejected(fd):
    state_zero = PerturbationState(0.0, 0.0)
    derivs_f15 = ShortPeriodDerivatives(
        flight_condition=FlightCondition.FLIGHT_15,
        Zw=-50.0, Zq=-5.0, Zde=-1000.0,
        Mw=-2.0, Mq=-10.0, Mde=-500.0,
    )
    with pytest.raises(ValueError):
        fd.calculate_state_derivatives(
            mass=MASS, iyy=IYY, state=state_zero,
            derivatives=derivs_f15, delta_e=0.0,
        )


def test_nan_delta_e_rejected(fd, derivs_f41):
    state_zero = PerturbationState(0.0, 0.0)
    with pytest.raises(ValueError):
        fd.calculate_state_derivatives(
            mass=MASS, iyy=IYY, state=state_zero,
            derivatives=derivs_f41, delta_e=float('nan'),
        )


def test_non_finite_state_rejected():
    with pytest.raises(ValueError):
        PerturbationState(delta_w=float('inf'), delta_q=0.0)


def test_extreme_inputs_produce_non_finite_rejected(fd):
    """Extreme-but-valid inputs should overflow and be caught."""
    extreme_derivs = ShortPeriodDerivatives(
        flight_condition=FlightCondition.FLIGHT_41,
        Zw=-1e200, Zq=-1e200, Zde=-1e200,
        Mw=-1e200, Mq=-1e200, Mde=-1e200,
    )
    extreme_state = PerturbationState(delta_w=1e200, delta_q=1e200)
    with pytest.raises(ValueError):
        fd.calculate_state_derivatives(
            mass=MASS, iyy=IYY, state=extreme_state,
            derivatives=extreme_derivs, delta_e=1e200,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
