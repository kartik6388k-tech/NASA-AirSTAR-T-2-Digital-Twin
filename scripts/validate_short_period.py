"""
scripts/validate_short_period.py

Level 3 Validation: NASA T-2 Short-Period Dynamics
Executes the data pipeline:
    config.yaml -> aerodynamics -> conversion -> A'/B' -> eigenvalues

Purpose: Prove whether the project-derived Z/M dimensional mapping 
mathematically reproduces the verified NASA T-2 short-period dynamic 
behavior (natural frequency, damping ratio, eigenvalues) published in 
AIAA 2015-2704.

STATUS: The pipeline is structurally sound. NASA Level-3 validation remains 
incomplete until verified NASA dynamic-reference values are obtained.
"""

import sys
import yaml
import math
from typing import Dict, Optional

# Adjust import paths depending on your exact project structure
from models.aerodynamics import AerodynamicsDatabase, FlightCondition
from models.atmosphere import get_atmosphere_from_altitude
from models.conversion import convert_to_dimensional_derivatives

def calculate_dynamic_characteristics(A11: float, A12: float, A21: float, A22: float) -> dict:
    """
    Extracts the short-period eigenvalues (lambda_1, lambda_2), 
    natural frequency (omega_n), and damping ratio (zeta) from the 
    synthesized A' matrix. Stability is determined directly from 
    the real parts of the eigenvalues.
    """
    trace = A11 + A22
    determinant = (A11 * A22) - (A12 * A21)
    
    # Calculate Eigenvalues: lambda^2 - Trace*lambda + Det = 0
    discriminant = trace**2 - 4.0 * determinant
    
    if discriminant < 0:
        real_part = trace / 2.0
        imag_part = math.sqrt(-discriminant) / 2.0
        lambda_1 = complex(real_part, imag_part)
        lambda_2 = complex(real_part, -imag_part)
        stable = real_part < 0.0
    else:
        l1 = (trace + math.sqrt(discriminant)) / 2.0
        l2 = (trace - math.sqrt(discriminant)) / 2.0
        lambda_1 = complex(l1, 0.0)
        lambda_2 = complex(l2, 0.0)
        stable = (l1 < 0.0) and (l2 < 0.0)
    
    if determinant > 0:
        omega_n = math.sqrt(determinant)
        zeta = -trace / (2.0 * omega_n)
    else:
        omega_n = float('nan')
        zeta = float('nan')
    
    return {
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "omega_n": omega_n,
        "zeta": zeta,
        "stable": stable
    }

def format_complex(c: complex) -> str:
    """Formats a complex number for readable output."""
    sign = "+" if c.imag >= 0 else "-"
    return f"{c.real:8.4f} {sign} {abs(c.imag):6.4f}j"

def validate_pipeline(config_path: str, nasa_references: Optional[Dict[str, Dict[str, float]]] = None):
    """
    Executes the end-to-end pipeline and outputs the dynamic characteristics.
    """
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. aerodynamics.py (Load verified coefficients)
    aero_db = AerodynamicsDatabase.from_config(config)

    for condition in [FlightCondition.FLIGHT_41, FlightCondition.FLIGHT_15]:
        print(f"\n=======================================================")
        print(f" LEVEL 3 VALIDATION: {condition.value.upper()}")
        print(f"=======================================================")
        
        try:
            flight_data = aero_db.get_flight_test_data(condition)
        except KeyError:
            print(f"Data for {condition.value} not found in configuration. Skipping.")
            continue

        # 2. atmosphere.py (Match exact nominal altitude)
        nominal_altitude = flight_data.nominal_condition.altitude_m
        atmosphere = get_atmosphere_from_altitude(nominal_altitude)

        # 3. conversion.py (Project-Derived Transformation)
        dimensional_model = convert_to_dimensional_derivatives(flight_data, atmosphere)
        derivs = dimensional_model.derivatives
        
        # 4. flight_dynamics.py integration basis (Extract constants cleanly isolated by record)
        mass = flight_data.reference_mass_properties.mass_kg
        Iyy = flight_data.reference_mass_properties.Iyy_kg_m2
        Ue = flight_data.nominal_condition.airspeed_mps

        # 5. Construct A' and B' Matrices
        # A' = [[Zw/m, Zq/m + Ue], [Mw/Iy, Mq/Iy]]
        A11 = derivs.Zw / mass
        A12 = (derivs.Zq / mass) + Ue
        A21 = derivs.Mw / Iyy
        A22 = derivs.Mq / Iyy

        # B' = [Zde/m, Mde/Iy]^T
        B1 = derivs.Zde / mass
        B2 = derivs.Mde / Iyy

        print("\n[ Synthesized State-Space Matrices ]")
        print(f"  A' = [[ {A11:8.4f}, {A12:8.4f} ]")
        print(f"        [ {A21:8.4f}, {A22:8.4f} ]]")
        print(f"  B' = [[ {B1:8.4f} ]")
        print(f"        [ {B2:8.4f} ]]")
        print("  (Note: B' is reported for completeness; eigenvalue validation relies strictly on A')")

        # 6. Extract Dynamic Characteristics
        chars = calculate_dynamic_characteristics(A11, A12, A21, A22)
        
        print("\n[ Dynamic Characteristics (Project-Derived) ]")
        print(f"  Eigenvalue (lambda_1)       : {format_complex(chars['lambda_1'])}")
        print(f"  Eigenvalue (lambda_2)       : {format_complex(chars['lambda_2'])}")
        print(f"  Natural Frequency (omega_n) : {chars['omega_n']:.4f} rad/s")
        print(f"  Damping Ratio (zeta)        : {chars['zeta']:.4f}")
        print(f"  Static Stability            : {'Stable' if chars['stable'] else 'Unstable'}")

        # 7. Compare to NASA References
        print("\n[ AIAA 2015-2704 Verification ]")
        if nasa_references and condition.value in nasa_references:
            ref = nasa_references[condition.value]
            ref_omega = ref.get("omega_n")
            ref_zeta = ref.get("zeta")
            
            if ref_omega is not None and ref_zeta is not None:
                err_omega = abs(chars['omega_n'] - ref_omega) / ref_omega * 100.0
                err_zeta = abs(chars['zeta'] - ref_zeta) / ref_zeta * 100.0
                
                print(f"  NASA omega_n : {ref_omega:.4f} rad/s  | Error: {err_omega:.2f}%")
                print(f"  NASA zeta    : {ref_zeta:.4f}         | Error: {err_zeta:.2f}%")
                
                if err_omega > 10.0 or err_zeta > 15.0:
                    print("\n  STATUS: WARNING - Large deviation detected.")
                    print("  The Z ≈ -L approximation may be inadequate for this flight condition.")
                else:
                    print("\n  STATUS: PASSED - Dynamics align with NASA reference.")
        else:
            print("  NASA COMPARISON NOT AVAILABLE")
            print("  (Level-3 validation remains incomplete until verified NASA dynamic-reference values are obtained.)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_short_period.py <path_to_config.yaml>")
        sys.exit(1)
        
    config_yaml_path = sys.argv[1]
    
    # NASA references remain empty/None until verified data is extracted from the source material.
    nasa_published_references = None 
    
    validate_pipeline(config_yaml_path, nasa_published_references)