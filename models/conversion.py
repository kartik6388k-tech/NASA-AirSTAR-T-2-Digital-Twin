"""
models/conversion.py

PROJECT-DERIVED CONVERSION

This module derives dimensional stability derivatives from NASA
nondimensional flight-test coefficients[cite: 16]. 

WARNING - PROVENANCE AND ASSUMPTIONS:
The resulting dimensional derivatives are PROJECT-DERIVED approximations, 
NOT directly published NASA values. They are synthesized to bridge the 
verified aerodynamic coefficients (aerodynamics.py) with the short-period 
equations of motion (flight_dynamics.py)[cite: 16, 19].

This mapping relies on the following explicitly defined machine-readable assumptions:

1. Z ≈ -L (PROJECT-DERIVED APPROXIMATION): 
   Drag (C_D) and elevator-induced axial forces are explicitly neglected 
   because verified C_D derivatives are unavailable. This is a small-angle 
   approximation, formally omitting base drag and trim-angle coupling.

2. delta_alpha ≈ delta_w / Ue (STANDARD MATHEMATICAL ASSUMPTION): 
   Valid strictly under small angular perturbations.

3. Constant Ue (STANDARD MATHEMATICAL ASSUMPTION): 
   Required for the isolated 2-DOF longitudinal short-period mode.

4. ISA Density (PROJECT-DERIVED): 
   The dynamic pressure used for dimensionalization is synthesized from an 
   ISA atmospheric model rather than raw measured flight-test dynamic pressure[cite: 18].

Validation Requirements:
Level 1 -> Formula vs hand calculation (Covered by tests/test_conversion.py)
Level 2 -> Source definitions / units (Addressed via code review)
Level 3 -> Resulting A'/B' behavior vs NASA short-period behavior (Pending external validation)
"""

import math
from dataclasses import dataclass, field
from typing import Dict

from .aerodynamics import FlightTestAeroData
from .atmosphere import AtmosphericState
from .flight_dynamics import ShortPeriodDerivatives


@dataclass(frozen=True)
class ProjectDerivedDimensionalModel:
    """
    Machine-readable container explicitly marking the resulting dimensional 
    derivatives as project-derived approximations. Retains full source 
    provenance, coefficient uncertainties, and exact dimensionalization state.
    """
    derivatives: ShortPeriodDerivatives
    source_flight_data: FlightTestAeroData  # Retains full provenance and coefficient uncertainties[cite: 16]
    dynamic_pressure_Pa: float
    density_kg_m3: float
    
    model_origin: str = "PROJECT_DERIVED"
    density_source: str = "PROJECT_ISA"
    dynamic_pressure_source: str = "PROJECT_DERIVED"
    z_derivatives_status: str = "PROJECT_DERIVED_APPROXIMATION"
    m_derivatives_status: str = "PROJECT_DERIVED_DIMENSIONALIZATION"
    
    assumptions: Dict[str, bool] = field(default_factory=lambda: {
        "Z_APPROX_MINUS_L": True,
        "SMALL_ANGLE_ALPHA": True,
        "CONSTANT_UE": True,
        "DRAG_DERIVATIVES_UNAVAILABLE": True
    })


def convert_to_dimensional_derivatives(
    flight_data: FlightTestAeroData,
    atmosphere: AtmosphericState
) -> ProjectDerivedDimensionalModel:
    """
    Converts NASA-verified nondimensional longitudinal derivatives into 
    project-derived dimensional stability derivatives (Zw, Zq, Zde, Mw, Mq, Mde).
    
    Args:
        flight_data: One complete, inseparable flight-test aerodynamic record[cite: 16].
        atmosphere:  The standard atmospheric state used to synthesize ISA density[cite: 18].
        
    Returns:
        ProjectDerivedDimensionalModel: Dimensional derivatives and strict provenance metadata.
    """
    # 1. Strict Type Validation
    if not isinstance(flight_data, FlightTestAeroData):
        raise TypeError(f"flight_data must be FlightTestAeroData, got {type(flight_data).__name__}")
    if not isinstance(atmosphere, AtmosphericState):
        raise TypeError(f"atmosphere must be AtmosphericState, got {type(atmosphere).__name__}")

    # 2. Environmental Consistency (Altitude Match - Engineering Tolerance)
    nominal_altitude = flight_data.nominal_condition.altitude_m
    if not math.isclose(atmosphere.altitude_m, nominal_altitude, abs_tol=1.0):
        raise ValueError(
            f"Atmosphere altitude ({atmosphere.altitude_m} m) does not match "
            f"the flight data nominal altitude ({nominal_altitude} m). "
            f"Cross-configuration mixing is prohibited."
        )

    # 3. Geometric and Kinematic Validation
    rho = atmosphere.density_kg_m3
    Ue = flight_data.nominal_condition.airspeed_mps
    S = flight_data.nominal_condition.wing_area_m2
    cbar = flight_data.nominal_condition.mac_m

    for name, val in [("rho", rho), ("Ue", Ue), ("S", S), ("cbar", cbar)]:
        if not math.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be strictly positive and finite, got {val}")

    # 4. Extract and Validate NASA-verified coefficients
    cl_a  = flight_data.derivatives.CL_alpha.value
    cl_q  = flight_data.derivatives.CL_q.value
    cl_de = flight_data.derivatives.CL_delta_e.value
    cm_a  = flight_data.derivatives.Cm_alpha.value
    cm_q  = flight_data.derivatives.Cm_q.value
    cm_de = flight_data.derivatives.Cm_delta_e.value

    for name, val in [("CL_alpha", cl_a), ("CL_q", cl_q), ("CL_delta_e", cl_de),
                      ("Cm_alpha", cm_a), ("Cm_q", cm_q), ("Cm_delta_e", cm_de)]:
        if not math.isfinite(val):
            raise ValueError(f"Coefficient {name} must be finite, got {val}")

    # 5. Dynamic Pressure (Project-derived dimensionalization base)
    qbar = 0.5 * rho * (Ue ** 2)

    # 6. Project-Derived Dimensional Transformations
    #    Derived under Z ≈ -L and delta_alpha ≈ w/Ue, employing the 
    #    source-defined pitch-rate normalization q_hat = q*cbar/(2*Ue).
    Zw_derived = -(qbar * S / Ue) * cl_a
    Zq_derived = -qbar * S * (cbar / (2.0 * Ue)) * cl_q
    Zde_derived = -qbar * S * cl_de

    Mw_derived = (qbar * S * cbar / Ue) * cm_a
    Mq_derived = qbar * S * cbar * (cbar / (2.0 * Ue)) * cm_q
    Mde_derived = qbar * S * cbar * cm_de

    # Ensure no calculation overflowed to infinity
    for name, val in [("Zw", Zw_derived), ("Zq", Zq_derived), ("Zde", Zde_derived),
                      ("Mw", Mw_derived), ("Mq", Mq_derived), ("Mde", Mde_derived)]:
        if not math.isfinite(val):
            raise ValueError(f"Derived derivative {name} overflowed or is non-finite: {val}")

    # 7. Output Construction
    dimensional_derivatives = ShortPeriodDerivatives(
        flight_condition=flight_data.condition,
        Zw=Zw_derived,
        Zq=Zq_derived,
        Zde=Zde_derived,
        Mw=Mw_derived,
        Mq=Mq_derived,
        Mde=Mde_derived
    )

    return ProjectDerivedDimensionalModel(
        derivatives=dimensional_derivatives,
        source_flight_data=flight_data,
        dynamic_pressure_Pa=qbar,
        density_kg_m3=rho
    )
