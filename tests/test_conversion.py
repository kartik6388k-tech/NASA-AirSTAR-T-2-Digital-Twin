"""
tests/test_conversion.py

Tests for models/conversion.py.

All TEST_ONLY values below are chosen to exercise the conversion logic.
They are NOT verified NASA T-2 parameters.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.aerodynamics import (
    FlightCondition,
    IdentifiedDerivative,
    LongitudinalDerivativeSet,
    NominalFlightCondition,
    PerturbationValidityLimits,
    ReferenceMassProperties,
    SourceProvenance,
)
from models.atmosphere import get_atmosphere_from_altitude
from models.conversion import (
    FlightTestAeroData,
    convert_to_dimensional_derivatives,
)


# --------------------------------------------------------------------------
# Shared TEST_ONLY fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def test_provenance():
    return SourceProvenance(
        document="TEST_ONLY", authors="TEST_ONLY", year=2024,
        table="TEST_ONLY", flight="TEST_ONLY",
    )


@pytest.fixture
def test_derivatives():
    return LongitudinalDerivativeSet(
        CL_alpha=IdentifiedDerivative(5.0, 0.1),
        CL_q=IdentifiedDerivative(10.0, 0.1),
        CL_delta_e=IdentifiedDerivative(0.5, 0.1),
        Cm_alpha=IdentifiedDerivative(-1.5, 0.1),
        Cm_q=IdentifiedDerivative(-15.0, 0.1),
        Cm_delta_e=IdentifiedDerivative(-1.0, 0.1),
    )


@pytest.fixture
def test_nominal():
    return NominalFlightCondition(
        flight_id="TEST_ONLY_FLIGHT",
        turbulence_context="None",
        airspeed_mps=100.0,
        angle_of_attack_rad=0.0,
        altitude_m=1000.0,
        throttle=0.5,
        mac_m=1.0,
        wing_area_m2=10.0,
    )


@pytest.fixture
def test_mass_props():
    return ReferenceMassProperties(mass_kg=1000.0, Iyy_kg_m2=2000.0)


@pytest.fixture
def test_flight_data(test_derivatives, test_nominal, test_mass_props, test_provenance):
    return FlightTestAeroData(
        condition=FlightCondition.FLIGHT_41,
        derivatives=test_derivatives,
        nominal_condition=test_nominal,
        reference_mass_properties=test_mass_props,
        derivative_provenance=test_provenance,
        nominal_condition_provenance=test_provenance,
        validity_limits=PerturbationValidityLimits(),
    )


@pytest.fixture
def valid_atmosphere():
    return get_atmosphere_from_altitude(1000.0)


# --------------------------------------------------------------------------
# Hand-calculation test
# --------------------------------------------------------------------------

def test_hand_calculation_all_six_derivatives(test_flight_data, valid_atmosphere):
    """
    Verify all 6 dimensional derivatives against hand calculation:
        Zw  = -(qbar*S/Ue) * CL_alpha
        Zq  = -(qbar*S) * (cbar/(2*Ue)) * CL_q
        Zde = -(qbar*S) * CL_delta_e
        Mw  =  (qbar*S*cbar/Ue) * Cm_alpha
        Mq  =  (qbar*S*cbar) * (cbar/(2*Ue)) * Cm_q
        Mde =  (qbar*S*cbar) * Cm_delta_e
    """
    rho = valid_atmosphere.density_kg_m3
    qbar = 0.5 * rho * (100.0 ** 2)

    exp_Zw  = -(qbar * 10.0 / 100.0) * 5.0
    exp_Zq  = -(qbar * 10.0) * (1.0 / 200.0) * 10.0
    exp_Zde = -(qbar * 10.0) * 0.5
    exp_Mw  = (qbar * 10.0 * 1.0 / 100.0) * -1.5
    exp_Mq  = (qbar * 10.0 * 1.0) * (1.0 / 200.0) * -15.0
    exp_Mde = (qbar * 10.0 * 1.0) * -1.0

    model = convert_to_dimensional_derivatives(test_flight_data, valid_atmosphere)
    result = model.derivatives

    assert math.isclose(result.Zw, exp_Zw, rel_tol=1e-9)
    assert math.isclose(result.Zq, exp_Zq, rel_tol=1e-9)
    assert math.isclose(result.Zde, exp_Zde, rel_tol=1e-9)
    assert math.isclose(result.Mw, exp_Mw, rel_tol=1e-9)
    assert math.isclose(result.Mq, exp_Mq, rel_tol=1e-9)
    assert math.isclose(result.Mde, exp_Mde, rel_tol=1e-9)


# --------------------------------------------------------------------------
# Provenance and metadata tests
# --------------------------------------------------------------------------

def test_provenance_and_density_preserved(test_flight_data, valid_atmosphere):
    """Model origin, assumptions, and source provenance must be machine-readable."""
    rho = valid_atmosphere.density_kg_m3
    qbar = 0.5 * rho * (100.0 ** 2)

    model = convert_to_dimensional_derivatives(test_flight_data, valid_atmosphere)

    assert model.model_origin == "PROJECT_DERIVED"
    assert model.assumptions["Z_APPROX_MINUS_L"] is True
    assert model.dynamic_pressure_Pa == qbar
    assert model.source_flight_data.derivative_provenance.document == "TEST_ONLY"


# --------------------------------------------------------------------------
# Input-validation tests
# --------------------------------------------------------------------------

def test_altitude_mismatch_rejected(test_flight_data):
    mismatched_atmosphere = get_atmosphere_from_altitude(5000.0)
    with pytest.raises(ValueError):
        convert_to_dimensional_derivatives(test_flight_data, mismatched_atmosphere)


def test_wrong_flight_data_type_rejected(valid_atmosphere):
    with pytest.raises(TypeError):
        convert_to_dimensional_derivatives("NotAFlightDataObj", valid_atmosphere)  # type: ignore


def test_negative_wing_area_rejected(
    test_derivatives, test_mass_props, test_provenance, valid_atmosphere
):
    """Negative wing area must be rejected (at construction or conversion)."""
    with pytest.raises(ValueError):
        bad_nominal = NominalFlightCondition(
            flight_id="TEST", turbulence_context="None", airspeed_mps=100.0,
            angle_of_attack_rad=0.0, altitude_m=1000.0, throttle=0.5,
            mac_m=1.0, wing_area_m2=-10.0,  # Invalid area
        )
        bad_flight = FlightTestAeroData(
            condition=FlightCondition.FLIGHT_41, derivatives=test_derivatives,
            nominal_condition=bad_nominal, reference_mass_properties=test_mass_props,
            derivative_provenance=test_provenance, nominal_condition_provenance=test_provenance,
            validity_limits=PerturbationValidityLimits(),
        )
        convert_to_dimensional_derivatives(bad_flight, valid_atmosphere)


def test_non_finite_coefficients_rejected(
    test_nominal, test_mass_props, test_provenance, valid_atmosphere
):
    """NaN/inf coefficients must be rejected (at construction or conversion)."""
    with pytest.raises(ValueError):
        bad_derivs = LongitudinalDerivativeSet(
            CL_alpha=IdentifiedDerivative(float('nan'), 0.0),
            CL_q=IdentifiedDerivative(10.0, 0.0),
            CL_delta_e=IdentifiedDerivative(0.5, 0.0),
            Cm_alpha=IdentifiedDerivative(-1.5, 0.0),
            Cm_q=IdentifiedDerivative(float('inf'), 0.0),
            Cm_delta_e=IdentifiedDerivative(-1.0, 0.0),
        )
        bad_flight = FlightTestAeroData(
            condition=FlightCondition.FLIGHT_41, derivatives=bad_derivs,
            nominal_condition=test_nominal, reference_mass_properties=test_mass_props,
            derivative_provenance=test_provenance, nominal_condition_provenance=test_provenance,
            validity_limits=PerturbationValidityLimits(),
        )
        convert_to_dimensional_derivatives(bad_flight, valid_atmosphere)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
