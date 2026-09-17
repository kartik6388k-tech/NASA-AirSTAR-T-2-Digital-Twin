"""
models/atmosphere.py

1976 U.S. Standard Atmosphere / ISA baseline model.

Responsibility (and ONLY responsibility):

    ALTITUDE  ->  ATMOSPHERIC STATE  (T, p, rho, a)

This module implements the standard tropospheric (0 - 11,000 m) and the
first stratospheric isothermal layer (11,000 - 20,000 m) segments of the
1976 U.S. Standard Atmosphere / ISA baseline, which are the layers
relevant to subscale flight-test vehicles such as the NASA AirSTAR T-2
GTM.

ISA is the project BASELINE atmosphere; it is not T-2 flight-test
atmospheric data. No measured T-2 atmospheric conditions are used or
implied anywhere in this module.

This module contains:
    - No aircraft-specific data (T-2 geometry, mass, CG, inertia, thrust,
      aerodynamic coefficients, sensor noise).
    - No wind, turbulence, or weather models.
    - No aerodynamics, equations of motion, sensor, or estimation logic.

config.yaml note
-----------------
config.yaml (as provided) contains only propulsion / configuration-identity
data for the T-2 GTM and defines no atmosphere-related fields. This
module does NOT read config.yaml and does NOT implement any config
override mechanism — all constants below are fixed, hard-coded standard
ISA constants. This is a current limitation, not a supported-but-unused
feature: if config-driven atmosphere constants are wanted later, loading
logic must be added explicitly. A possible future `atmosphere:` section
is sketched below purely as a placeholder for that future work:

    atmosphere:
      model: "ISA"
      T0_K: 288.15
      p0_Pa: 101325.0
      rho0_kg_m3: 1.225
      valid_altitude_range_m: [0.0, 20000.0]

Altitude convention
--------------------
This module expects GEOPOTENTIAL altitude as input (the convention the
constant-g0 ISA equations below are formulated in). No geopotential <->
geometric conversion is performed.

get_atmosphere_from_ned_z(z_ned_m) computes altitude_m = -z_ned_m and
treats the result as geopotential altitude. This is only correct when
the NED origin is defined at the atmospheric reference level (mean sea
level, altitude = 0). If the NED origin is elsewhere (e.g. a local
launch point above MSL), the caller must convert first:
altitude_m = altitude_origin_m - z_ned_m. This module does not know the
origin's MSL altitude and cannot apply that correction itself.
"""

from dataclasses import dataclass
import math


# =============================================================================
# STANDARD PHYSICAL CONSTANTS (1976 U.S. Standard Atmosphere / ISA baseline)
# These are internationally standardized constants, not project inventions.
# =============================================================================

# Sea-level reference conditions (ISA standard day)
T0_K: float = 288.15          # Sea-level standard temperature [K]
P0_PA: float = 101325.0       # Sea-level standard pressure [Pa]

# RHO0_KG_M3 is the documented ISA sea-level reference density. It is NOT
# used to compute density anywhere in this module — density is always
# derived from rho = p / (R*T) (see _isa_troposphere /
# _isa_isothermal_stratosphere). RHO0_KG_M3 exists solely as the known
# reference value that the computed sea-level density is checked against
# in _self_check().
RHO0_KG_M3: float = 1.225     # Sea-level standard density [kg/m^3] (reference/validation only)

# Gas / thermodynamic constants
R_SPECIFIC_AIR: float = 287.05287  # Specific gas constant for dry air [J/(kg*K)]
GAMMA_AIR: float = 1.4             # Ratio of specific heats for air [-]

# Gravity and lapse rate (troposphere)
G0_M_S2: float = 9.80665      # Standard gravity [m/s^2]
LAPSE_RATE_TROPO_K_M: float = -0.0065  # Tropospheric temperature lapse rate [K/m]

# Layer boundaries (geopotential altitude, ISA standard atmosphere)
TROPOPAUSE_ALT_M: float = 11000.0        # Top of troposphere [m]
STRATOSPHERE_ISOTHERMAL_TOP_M: float = 20000.0  # Top of first isothermal layer [m]

# Temperature at the tropopause (constant through the first isothermal layer)
T_TROPOPAUSE_K: float = T0_K + LAPSE_RATE_TROPO_K_M * TROPOPAUSE_ALT_M  # 216.65 K

# Pressure at the tropopause (evaluated once, used as the isothermal-layer base)
P_TROPOPAUSE_PA: float = P0_PA * (
    T_TROPOPAUSE_K / T0_K
) ** (-G0_M_S2 / (LAPSE_RATE_TROPO_K_M * R_SPECIFIC_AIR))

# Supported altitude range for THIS IMPLEMENTATION.
# This is a project-supported range, not an ISA physical limitation — the
# standard atmosphere model itself is defined below sea level (e.g. for
# below-MSL terrain) and above 20,000 m via additional layers. Those
# layers/extensions are simply not implemented here. Below-sea-level
# support could be added later if a scenario requires it; it is excluded
# now to keep the validated range unambiguous.
MIN_SUPPORTED_ALTITUDE_M: float = 0.0
MAX_SUPPORTED_ALTITUDE_M: float = STRATOSPHERE_ISOTHERMAL_TOP_M


# =============================================================================
# OUTPUT DATA STRUCTURE
# =============================================================================

@dataclass(frozen=True)
class AtmosphericState:
    """
    Atmospheric properties at a given altitude.

    All fields are in SI units.
    """
    altitude_m: float
    temperature_K: float
    pressure_Pa: float
    density_kg_m3: float
    speed_of_sound_mps: float


# =============================================================================
# INTERNAL VALIDATION
# =============================================================================

def _validate_altitude_m(altitude_m: float) -> float:
    """
    Validate a raw altitude input (positive-up convention).

    Raises TypeError for non-numeric input, ValueError for NaN, infinite,
    or out-of-supported-range values. Returns the altitude as a float.
    """
    if isinstance(altitude_m, bool) or not isinstance(altitude_m, (int, float)):
        raise TypeError(
            f"altitude_m must be a real number, got {type(altitude_m).__name__}"
        )

    altitude_m = float(altitude_m)

    if math.isnan(altitude_m):
        raise ValueError("altitude_m is NaN")

    if math.isinf(altitude_m):
        raise ValueError("altitude_m is infinite")

    if not (MIN_SUPPORTED_ALTITUDE_M <= altitude_m <= MAX_SUPPORTED_ALTITUDE_M):
        raise ValueError(
            f"altitude_m={altitude_m} is outside the supported ISA range "
            f"[{MIN_SUPPORTED_ALTITUDE_M}, {MAX_SUPPORTED_ALTITUDE_M}] m. "
            "No extrapolation policy is defined; extend the model explicitly "
            "if a wider range is required."
        )

    return altitude_m


def _validate_output_state(state: "AtmosphericState") -> "AtmosphericState":
    """
    Sanity-check a computed AtmosphericState before returning it to the
    caller. This guards against implementation errors (bad formula, sign
    error, etc.) producing a physically impossible result — it is not
    input validation, which happens in _validate_altitude_m().
    """
    # Explicitly verify finiteness first; a NaN would also fail the
    # positivity checks below, but the error message would be misleading.
    for field_name, value in (
        ("temperature_K", state.temperature_K),
        ("pressure_Pa", state.pressure_Pa),
        ("density_kg_m3", state.density_kg_m3),
        ("speed_of_sound_mps", state.speed_of_sound_mps),
    ):
        if not math.isfinite(value):
            raise ValueError(
                f"Computed non-finite {field_name}={value} at "
                f"altitude_m={state.altitude_m}"
            )
    if not (state.temperature_K > 0.0):
        raise ValueError(f"Computed non-physical temperature_K={state.temperature_K}")
    if not (state.pressure_Pa > 0.0):
        raise ValueError(f"Computed non-physical pressure_Pa={state.pressure_Pa}")
    if not (state.density_kg_m3 > 0.0):
        raise ValueError(f"Computed non-physical density_kg_m3={state.density_kg_m3}")
    if not (state.speed_of_sound_mps > 0.0):
        raise ValueError(f"Computed non-physical speed_of_sound_mps={state.speed_of_sound_mps}")
    return state


# =============================================================================
# CORE ISA EQUATIONS
# =============================================================================

def _isa_troposphere(altitude_m: float) -> AtmosphericState:
    """ISA equations for 0 <= altitude_m <= 11,000 m (linear lapse-rate layer)."""
    temperature_K = T0_K + LAPSE_RATE_TROPO_K_M * altitude_m

    pressure_Pa = P0_PA * (
        temperature_K / T0_K
    ) ** (-G0_M_S2 / (LAPSE_RATE_TROPO_K_M * R_SPECIFIC_AIR))

    density_kg_m3 = pressure_Pa / (R_SPECIFIC_AIR * temperature_K)

    speed_of_sound_mps = math.sqrt(GAMMA_AIR * R_SPECIFIC_AIR * temperature_K)

    return _validate_output_state(AtmosphericState(
        altitude_m=altitude_m,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        density_kg_m3=density_kg_m3,
        speed_of_sound_mps=speed_of_sound_mps,
    ))


def _isa_isothermal_stratosphere(altitude_m: float) -> AtmosphericState:
    """ISA equations for 11,000 m < altitude_m <= 20,000 m (isothermal layer)."""
    temperature_K = T_TROPOPAUSE_K  # constant in this layer

    pressure_Pa = P_TROPOPAUSE_PA * math.exp(
        -G0_M_S2 * (altitude_m - TROPOPAUSE_ALT_M) / (R_SPECIFIC_AIR * temperature_K)
    )

    density_kg_m3 = pressure_Pa / (R_SPECIFIC_AIR * temperature_K)

    speed_of_sound_mps = math.sqrt(GAMMA_AIR * R_SPECIFIC_AIR * temperature_K)

    return _validate_output_state(AtmosphericState(
        altitude_m=altitude_m,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        density_kg_m3=density_kg_m3,
        speed_of_sound_mps=speed_of_sound_mps,
    ))


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def get_atmosphere_from_altitude(altitude_m: float) -> AtmosphericState:
    """
    Compute ISA atmospheric state from a positive-up altitude.

    Parameters
    ----------
    altitude_m : float
        Altitude above sea level, positive up [m].
        Supported range: [0, 20000] m.

    Returns
    -------
    AtmosphericState
        Temperature, pressure, density, and speed of sound at the given
        altitude.

    Raises
    ------
    TypeError
        If altitude_m is not a real number.
    ValueError
        If altitude_m is NaN, infinite, or outside the supported range.
    """
    altitude_m = _validate_altitude_m(altitude_m)

    if altitude_m <= TROPOPAUSE_ALT_M:
        return _isa_troposphere(altitude_m)
    else:
        return _isa_isothermal_stratosphere(altitude_m)


def get_atmosphere_from_ned_z(z_ned_m: float) -> AtmosphericState:
    """
    Compute ISA atmospheric state from a NED-frame z coordinate.

    NED convention used throughout this project:
        x = North, y = East, z = Down.
    Therefore: altitude_m = -z_ned_m.

    The resulting altitude_m is interpreted as GEOPOTENTIAL altitude (see
    module docstring), and this conversion is only valid when the NED
    origin is defined at the atmospheric reference level (altitude = 0,
    i.e. mean sea level). If the NED origin sits at some other MSL
    altitude (e.g. a local launch point), the caller must apply
    altitude_m = altitude_origin_m - z_ned_m before/instead of using this
    function — this module has no knowledge of the origin's MSL altitude.

    Parameters
    ----------
    z_ned_m : float
        NED z-coordinate [m]. Negative values correspond to altitude above
        the origin (positive altitude); positive z is below the origin.

    Returns
    -------
    AtmosphericState

    Raises
    ------
    TypeError
        If z_ned_m is not a real number.
    ValueError
        If the resulting altitude is NaN, infinite, or outside the
        supported range.
    """
    if isinstance(z_ned_m, bool) or not isinstance(z_ned_m, (int, float)):
        raise TypeError(
            f"z_ned_m must be a real number, got {type(z_ned_m).__name__}"
        )

    altitude_m = -float(z_ned_m)
    return get_atmosphere_from_altitude(altitude_m)


# =============================================================================
# MINIMAL SELF-CHECK (standard ISA reference values only — no T-2 data)
# =============================================================================

def _self_check() -> None:
    """
    Lightweight smoke test against standard ISA reference values, plus
    basic boundary and monotonicity checks across both implemented layers.
    This is not a substitute for tests/test_atmosphere.py, which is the
    authoritative test suite.
    """
    # Sea level
    sea_level = get_atmosphere_from_altitude(0.0)
    assert math.isclose(sea_level.temperature_K, 288.15, rel_tol=1e-6), sea_level
    assert math.isclose(sea_level.pressure_Pa, 101325.0, rel_tol=1e-6), sea_level
    assert math.isclose(sea_level.density_kg_m3, RHO0_KG_M3, rel_tol=2e-3), sea_level
    assert math.isclose(sea_level.speed_of_sound_mps, 340.3, rel_tol=2e-3), sea_level

    # Internal consistency at sea level: rho == p / (R*T), a == sqrt(gamma*R*T)
    rho_check = sea_level.pressure_Pa / (R_SPECIFIC_AIR * sea_level.temperature_K)
    assert math.isclose(sea_level.density_kg_m3, rho_check, rel_tol=1e-9)
    a_check = math.sqrt(GAMMA_AIR * R_SPECIFIC_AIR * sea_level.temperature_K)
    assert math.isclose(sea_level.speed_of_sound_mps, a_check, rel_tol=1e-9)

    # Tropopause and top of the modeled range
    tropopause = get_atmosphere_from_altitude(TROPOPAUSE_ALT_M)
    top = get_atmosphere_from_altitude(MAX_SUPPORTED_ALTITUDE_M)
    assert math.isclose(tropopause.temperature_K, 216.65, rel_tol=1e-4), tropopause

    # Continuity across the troposphere/isothermal-layer boundary
    just_below = get_atmosphere_from_altitude(TROPOPAUSE_ALT_M - 0.01)
    just_above = get_atmosphere_from_altitude(TROPOPAUSE_ALT_M + 0.01)
    assert math.isclose(just_below.pressure_Pa, just_above.pressure_Pa, rel_tol=1e-4)
    assert math.isclose(just_below.temperature_K, just_above.temperature_K, rel_tol=1e-4)
    # Density continuity at the 11 km boundary
    assert math.isclose(
        just_below.density_kg_m3,
        just_above.density_kg_m3,
        rel_tol=1e-4,
    )
    # Speed-of-sound continuity at the 11 km boundary
    assert math.isclose(
        just_below.speed_of_sound_mps,
        just_above.speed_of_sound_mps,
        rel_tol=1e-4,
    )

    # NED conversion round-trip: get_atmosphere_from_ned_z(-1000) must
    # match get_atmosphere_from_altitude(1000) to floating-point precision.
    ned_state = get_atmosphere_from_ned_z(-1000.0)
    altitude_state = get_atmosphere_from_altitude(1000.0)
    assert math.isclose(
        ned_state.temperature_K,
        altitude_state.temperature_K,
        rel_tol=1e-12,
    )
    assert math.isclose(
        ned_state.pressure_Pa,
        altitude_state.pressure_Pa,
        rel_tol=1e-12,
    )
    assert math.isclose(
        ned_state.density_kg_m3,
        altitude_state.density_kg_m3,
        rel_tol=1e-12,
    )

    # Monotonicity across both layers: pressure and density strictly decrease
    states = [sea_level, get_atmosphere_from_altitude(5000.0), tropopause,
              get_atmosphere_from_altitude(15000.0), top]
    pressures = [s.pressure_Pa for s in states]
    densities = [s.density_kg_m3 for s in states]
    assert all(a > b for a, b in zip(pressures, pressures[1:]))
    assert all(a > b for a, b in zip(densities, densities[1:]))
    # Temperature monotonicity in the troposphere (must decrease with altitude)
    temperatures = [s.temperature_K for s in states[:3]]
    assert all(a > b for a, b in zip(temperatures, temperatures[1:]))

    print("atmosphere.py self-check passed:", sea_level)


if __name__ == "__main__":
    _self_check()