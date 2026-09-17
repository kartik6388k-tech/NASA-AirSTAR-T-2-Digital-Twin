"""
models/aerodynamics.py

NASA AirSTAR T-2 flight-test identified longitudinal perturbation model.

Responsibility (and ONLY responsibility):

    flight-specific identified data + perturbation state
        -> (delta_CL, delta_Cm)
        -> (delta_lift, delta_pitching_moment)

Architecture
------------
- config.yaml remains the source of truth, but this module never opens it.
  The application layer loads configuration once and passes the already-loaded
  mapping to AerodynamicsDatabase.from_config(...).

- Model-facing configuration values are SI. This module performs no
  imperial-to-SI conversion. Original NASA/source units may remain in
  provenance-only config fields.

- Flight 41 derivatives and nominal conditions are one inseparable record.
  Flight 15 derivatives and nominal conditions are a separate inseparable
  record. Callers cannot independently substitute nominal V, MAC, or S.

- This is a local LINEAR PERTURBATION model. It computes delta_CL and delta_Cm,
  not absolute CL or Cm. No numerical validity bounds are invented. If config
  later provides source-backed perturbation bounds, they can be enforced here.
\
Perturbation variables
----------------------
    delta_alpha = alpha_absolute - alpha_trim
    delta_q     = q_absolute     - q_trim
    delta_e     = delta_e_absolute - delta_e_trim

The public coefficient function receives these perturbations directly. A helper
is provided for alpha because alpha_trim is explicitly stored in the matched
nominal flight condition.

Equations
---------
    delta_CL = CL_alpha * delta_alpha
             + CL_q * (delta_q * c_bar / (2 * V_trim))
             + CL_delta_e * delta_e

    delta_Cm = Cm_alpha * delta_alpha
             + Cm_q * (delta_q * c_bar / (2 * V_trim))
             + Cm_delta_e * delta_e

Dimensionalization uses the selected flight record's reference geometry:

    delta_L = q_bar * S * delta_CL
    delta_M = q_bar * S * c_bar * delta_Cm

where q_bar = 0.5 * rho * V_trim^2. rho must come from AtmosphericState.

Elevator sign
-------------
This module does NOT assert a physical positive-elevator surface convention.
delta_elevator_rad must use the same sign convention as the identified source
data. The source/config convention must be locked before downstream actuator
signs are treated as authoritative.

Uncertainty
-----------
Identified 1-sigma uncertainties are preserved and validated as metadata.
Deterministic calculations use the mean derivative estimates only. This module
does not claim uncertainty propagation.

Provenance
----------
This module validates configuration structure, types, flight pairing, and
reference-case consistency. It does not independently verify NASA sources.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Optional

from .atmosphere import AtmosphericState


# =============================================================================
# ERRORS / CONSTANTS
# =============================================================================

UNVERIFIED = "NOT_FOUND_DO_NOT_ASSUME"

_REQUIRED_DERIVATIVES = (
    "CL_alpha",
    "CL_q",
    "CL_delta_e",
    "Cm_alpha",
    "Cm_q",
    "Cm_delta_e",
)


class AerodynamicsConfigError(ValueError):
    """Raised for malformed or inconsistent aerodynamic configuration."""


class FlightCondition(Enum):
    """Supported NASA T-2 flight-test identified derivative sets."""

    FLIGHT_41 = "flight_41"
    FLIGHT_15 = "flight_15"


@dataclass(frozen=True)
class _FlightBinding:
    derivative_key: str
    reference_key: str
    flight_id: str
    expected_condition_reference: str


_FLIGHT_BINDINGS = MappingProxyType({
    FlightCondition.FLIGHT_41: _FlightBinding(
        derivative_key="flight_41",
        reference_key="t2_flight_41",
        flight_id="T-2 Flight 41",
        expected_condition_reference=(
            "simulation_reference_data.reference_cases.t2_flight_41"
        ),
    ),
    FlightCondition.FLIGHT_15: _FlightBinding(
        derivative_key="flight_15",
        reference_key="t2_flight_15",
        flight_id="T-2 Flight 15",
        expected_condition_reference=(
            "simulation_reference_data.reference_cases.t2_flight_15"
        ),
    ),
})


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def _is_number(value: Any) -> bool:
    return type(value) in (int, float)


def _require_mapping(root: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    value = _require_value(root, path)
    if not isinstance(value, Mapping):
        raise AerodynamicsConfigError(
            f"Configuration field '{path}' must be a mapping, "
            f"got {type(value).__name__}."
        )
    return value


def _require_value(root: Mapping[str, Any], path: str) -> Any:
    node: Any = root
    traversed: list[str] = []

    for key in path.split("."):
        traversed.append(key)
        current_path = ".".join(traversed)
        if not isinstance(node, Mapping) or key not in node:
            raise AerodynamicsConfigError(
                f"Missing required configuration field '{current_path}'."
            )
        node = node[key]

    if node == UNVERIFIED:
        raise AerodynamicsConfigError(
            f"Configuration field '{path}' is '{UNVERIFIED}' and cannot be "
            "used as an active aerodynamic model parameter."
        )

    return node


def _require_number(
    root: Mapping[str, Any],
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    value = _require_value(root, path)

    if not _is_number(value):
        raise AerodynamicsConfigError(
            f"Configuration field '{path}' must be a numeric int/float, "
            f"got {type(value).__name__}."
        )
    if not math.isfinite(value):
        raise AerodynamicsConfigError(
            f"Configuration field '{path}' must be finite, got {value!r}."
        )
    if positive and value <= 0:
        raise AerodynamicsConfigError(
            f"Configuration field '{path}' must be > 0, got {value!r}."
        )
    if nonnegative and value < 0:
        raise AerodynamicsConfigError(
            f"Configuration field '{path}' must be >= 0, got {value!r}."
        )

    return float(value)


def _require_string(root: Mapping[str, Any], path: str) -> str:
    value = _require_value(root, path)
    if not isinstance(value, str) or not value.strip():
        raise AerodynamicsConfigError(
            f"Configuration field '{path}' must be a non-empty string."
        )
    return value.strip()


def _optional_string(root: Mapping[str, Any], path: str) -> Optional[str]:
    node: Any = root
    for key in path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]

    if node in (None, UNVERIFIED):
        return None
    if not isinstance(node, str) or not node.strip():
        raise AerodynamicsConfigError(
            f"Configuration field '{path}' must be a non-empty string "
            "when provided."
        )
    return node.strip()


def _optional_number(
    root: Mapping[str, Any],
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Optional[float]:
    node: Any = root
    for key in path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]

    if node in (None, UNVERIFIED):
        return None
    if not _is_number(node):
        raise AerodynamicsConfigError(
            f"Configuration field '{path}' must be numeric when provided, "
            f"got {type(node).__name__}."
        )
    if not math.isfinite(node):
        raise AerodynamicsConfigError(
            f"Configuration field '{path}' must be finite when provided."
        )
    if positive and node <= 0:
        raise AerodynamicsConfigError(
            f"Configuration field '{path}' must be > 0 when provided."
        )
    if nonnegative and node < 0:
        raise AerodynamicsConfigError(
            f"Configuration field '{path}' must be >= 0 when provided."
        )
    return float(node)


def _validate_runtime_number(name: str, value: Any) -> float:
    if not _is_number(value):
        raise TypeError(
            f"{name} must be a numeric int/float, got {type(value).__name__}."
        )
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return float(value)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class IdentifiedDerivative:
    """One identified nondimensional derivative and its 1-sigma uncertainty."""

    value: float
    uncertainty_1sigma: float

    def __post_init__(self) -> None:
        value = _validate_runtime_number("derivative.value", self.value)
        uncertainty = _validate_runtime_number(
            "derivative.uncertainty_1sigma",
            self.uncertainty_1sigma,
        )
        if uncertainty < 0:
            raise ValueError("derivative.uncertainty_1sigma must be >= 0.")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "uncertainty_1sigma", uncertainty)


@dataclass(frozen=True)
class LongitudinalDerivativeSet:
    CL_alpha: IdentifiedDerivative
    CL_q: IdentifiedDerivative
    CL_delta_e: IdentifiedDerivative
    Cm_alpha: IdentifiedDerivative
    Cm_q: IdentifiedDerivative
    Cm_delta_e: IdentifiedDerivative


@dataclass(frozen=True)
class NominalFlightCondition:
    """Flight-specific trim/reference values used by the linearized model."""

    flight_id: str
    turbulence_context: str
    airspeed_mps: float
    angle_of_attack_rad: float
    altitude_m: float
    throttle: float
    mac_m: float
    wing_area_m2: float

    def __post_init__(self) -> None:
        if not isinstance(self.flight_id, str) or not self.flight_id.strip():
            raise ValueError("flight_id must be a non-empty string.")
        if (
            not isinstance(self.turbulence_context, str)
            or not self.turbulence_context.strip()
        ):
            raise ValueError("turbulence_context must be a non-empty string.")

        for name in (
            "airspeed_mps",
            "angle_of_attack_rad",
            "altitude_m",
            "throttle",
            "mac_m",
            "wing_area_m2",
        ):
            value = _validate_runtime_number(name, getattr(self, name))
            object.__setattr__(self, name, value)

        if self.airspeed_mps <= 0:
            raise ValueError("airspeed_mps must be > 0.")
        if self.mac_m <= 0:
            raise ValueError("mac_m must be > 0.")
        if self.wing_area_m2 <= 0:
            raise ValueError("wing_area_m2 must be > 0.")
        if self.altitude_m < 0:
            raise ValueError("altitude_m must be >= 0.")
        if not 0.0 <= self.throttle <= 1.0:
            raise ValueError("throttle must be in [0, 1].")


@dataclass(frozen=True)
class ReferenceMassProperties:
    """Paper-specific reference mass properties retained as metadata."""

    mass_kg: float
    Iyy_kg_m2: float

    def __post_init__(self) -> None:
        mass = _validate_runtime_number("mass_kg", self.mass_kg)
        iyy = _validate_runtime_number("Iyy_kg_m2", self.Iyy_kg_m2)
        if mass <= 0:
            raise ValueError("mass_kg must be > 0.")
        if iyy <= 0:
            raise ValueError("Iyy_kg_m2 must be > 0.")
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "Iyy_kg_m2", iyy)


@dataclass(frozen=True)
class SourceProvenance:
    document: str
    authors: str
    year: int
    table: str
    flight: str
    aiaa_paper: Optional[str] = None
    nasa_document_id: Optional[str] = None
    section: Optional[str] = None
    source_url: Optional[str] = None


@dataclass(frozen=True)
class PerturbationValidityLimits:
    """Optional source/config-backed local-model perturbation limits."""

    max_abs_delta_alpha_rad: Optional[float] = None
    max_abs_delta_pitch_rate_rads: Optional[float] = None
    max_abs_delta_elevator_rad: Optional[float] = None


@dataclass(frozen=True)
class FlightTestAeroData:
    """An inseparable derivative set + its matching nominal reference case."""

    condition: FlightCondition
    derivatives: LongitudinalDerivativeSet
    nominal_condition: NominalFlightCondition
    reference_mass_properties: ReferenceMassProperties
    derivative_provenance: SourceProvenance
    nominal_condition_provenance: SourceProvenance
    validity_limits: PerturbationValidityLimits


@dataclass(frozen=True)
class LongitudinalPerturbationCoefficients:
    delta_CL: float
    delta_Cm: float
    validity_status: str
    validity_note: str


@dataclass(frozen=True)
class LongitudinalPerturbationForces:
    delta_lift_N: float
    delta_pitching_moment_Nm: float
    dynamic_pressure_Pa: float


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def _load_derivative(
    estimates: Mapping[str, Any],
    derivative_name: str,
    base_path: str,
) -> IdentifiedDerivative:
    if derivative_name not in estimates:
        raise AerodynamicsConfigError(
            f"Missing required configuration field "
            f"'{base_path}.{derivative_name}'."
        )
    if not isinstance(estimates[derivative_name], Mapping):
        raise AerodynamicsConfigError(
            f"Configuration field '{base_path}.{derivative_name}' "
            "must be a mapping."
        )

    return IdentifiedDerivative(
        value=_require_number(estimates, f"{derivative_name}.value"),
        uncertainty_1sigma=_require_number(
            estimates,
            f"{derivative_name}.uncertainty_1sigma",
            nonnegative=True,
        ),
    )


def _load_source_provenance(
    source: Mapping[str, Any],
    *,
    table: str,
    flight: str,
) -> SourceProvenance:
    year_raw = _require_value(source, "year")
    if isinstance(year_raw, bool) or type(year_raw) is not int:
        raise AerodynamicsConfigError(
            "Source field 'year' must be an integer."
        )

    return SourceProvenance(
        document=_require_string(source, "document"),
        authors=_require_string(source, "authors"),
        year=year_raw,
        table=table,
        flight=flight,
        aiaa_paper=_optional_string(source, "aiaa_paper"),
        nasa_document_id=_optional_string(source, "nasa_document_id"),
        section=_optional_string(source, "section"),
        source_url=_optional_string(source, "source_url"),
    )


def _load_validity_limits(
    derivative_record: Mapping[str, Any],
) -> PerturbationValidityLimits:
    limits = derivative_record.get("validity_limits")
    if limits is None:
        return PerturbationValidityLimits()
    if not isinstance(limits, Mapping):
        raise AerodynamicsConfigError(
            "validity_limits must be a mapping when provided."
        )

    return PerturbationValidityLimits(
        max_abs_delta_alpha_rad=_optional_number(
            limits,
            "max_abs_delta_alpha_rad",
            nonnegative=True,
        ),
        max_abs_delta_pitch_rate_rads=_optional_number(
            limits,
            "max_abs_delta_pitch_rate_rads",
            nonnegative=True,
        ),
        max_abs_delta_elevator_rad=_optional_number(
            limits,
            "max_abs_delta_elevator_rad",
            nonnegative=True,
        ),
    )


def _load_one_flight(
    config: Mapping[str, Any],
    condition: FlightCondition,
) -> FlightTestAeroData:
    binding = _FLIGHT_BINDINGS[condition]

    ft_root = _require_mapping(config, "flight_test_identified_data")
    derivative_source = _require_mapping(ft_root, "source")
    derivatives_root = _require_mapping(ft_root, "aerodynamic_derivatives")
    derivative_record = _require_mapping(
        derivatives_root,
        binding.derivative_key,
    )

    configured_flight_id = _require_string(derivative_record, "flight_id")
    if configured_flight_id != binding.flight_id:
        raise AerodynamicsConfigError(
            f"{binding.derivative_key}.flight_id must be "
            f"{binding.flight_id!r}, got {configured_flight_id!r}."
        )

    condition_reference = _require_string(
        derivative_record,
        "condition_reference",
    )
    if condition_reference != binding.expected_condition_reference:
        raise AerodynamicsConfigError(
            f"{binding.derivative_key}.condition_reference must be "
            f"{binding.expected_condition_reference!r}, "
            f"got {condition_reference!r}."
        )

    estimates = _require_mapping(derivative_record, "estimates")
    base_path = (
        "flight_test_identified_data.aerodynamic_derivatives."
        f"{binding.derivative_key}.estimates"
    )

    derivative_values = {
        name: _load_derivative(estimates, name, base_path)
        for name in _REQUIRED_DERIVATIVES
    }
    derivatives = LongitudinalDerivativeSet(**derivative_values)

    sim_root = _require_mapping(config, "simulation_reference_data")
    nominal_source = _require_mapping(sim_root, "source")
    reference_cases = _require_mapping(sim_root, "reference_cases")
    reference_case = _require_mapping(reference_cases, binding.reference_key)

    reference_flight_id = _require_string(reference_case, "flight_id")
    if reference_flight_id != binding.flight_id:
        raise AerodynamicsConfigError(
            f"reference_cases.{binding.reference_key}.flight_id must be "
            f"{binding.flight_id!r}, got {reference_flight_id!r}."
        )

    status = _require_string(reference_case, "status")
    if status != "NASA_PAPER_FLIGHT_TEST_REFERENCE":
        raise AerodynamicsConfigError(
            f"reference_cases.{binding.reference_key}.status must be "
            "'NASA_PAPER_FLIGHT_TEST_REFERENCE'."
        )

    si = _require_mapping(reference_case, "reference_values_si")
    nominal = _require_mapping(si, "nominal_flight_condition")

    nominal_condition = NominalFlightCondition(
        flight_id=binding.flight_id,
        turbulence_context=_require_string(
            derivative_record,
            "turbulence_context",
        ),
        airspeed_mps=_require_number(
            nominal,
            "airspeed_m_s",
            positive=True,
        ),
        angle_of_attack_rad=_require_number(
            nominal,
            "angle_of_attack_rad",
        ),
        altitude_m=_require_number(
            nominal,
            "altitude_m",
            nonnegative=True,
        ),
        throttle=_require_number(
            nominal,
            "throttle",
            nonnegative=True,
        ),
        mac_m=_require_number(
            si,
            "mean_aerodynamic_chord_m",
            positive=True,
        ),
        wing_area_m2=_require_number(
            si,
            "wing_reference_area_m2",
            positive=True,
        ),
    )

    reference_mass_properties = ReferenceMassProperties(
        mass_kg=_require_number(si, "mass_kg", positive=True),
        Iyy_kg_m2=_require_number(
            si,
            "pitch_moment_of_inertia_kg_m2",
            positive=True,
        ),
    )

    derivative_table = _require_string(derivative_source, "table")
    tables = _require_mapping(nominal_source, "tables")
    nominal_table = _require_string(tables, "nominal_flight_conditions")

    return FlightTestAeroData(
        condition=condition,
        derivatives=derivatives,
        nominal_condition=nominal_condition,
        reference_mass_properties=reference_mass_properties,
        derivative_provenance=_load_source_provenance(
            derivative_source,
            table=derivative_table,
            flight=binding.flight_id,
        ),
        nominal_condition_provenance=_load_source_provenance(
            nominal_source,
            table=nominal_table,
            flight=binding.flight_id,
        ),
        validity_limits=_load_validity_limits(derivative_record),
    )


def _validate_reference_geometry_consistency(
    config: Mapping[str, Any],
    flights: Mapping[FlightCondition, FlightTestAeroData],
) -> None:
    """Ensure flight dimensionalization uses the configured GTM reference geometry."""
    sim_root = _require_mapping(config, "simulation_reference_data")
    reference_cases = _require_mapping(sim_root, "reference_cases")
    gtm = _require_mapping(reference_cases, "gtm_reference")
    gtm_si = _require_mapping(gtm, "reference_values_si")

    expected_mac = _require_number(
        gtm_si,
        "mean_aerodynamic_chord_m",
        positive=True,
    )
    expected_area = _require_number(
        gtm_si,
        "wing_reference_area_m2",
        positive=True,
    )

    for condition, data in flights.items():
        nominal = data.nominal_condition
        if not math.isclose(
            nominal.mac_m,
            expected_mac,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AerodynamicsConfigError(
                f"{condition.value} MAC {nominal.mac_m} does not match "
                f"configured GTM reference MAC {expected_mac}."
            )
        if not math.isclose(
            nominal.wing_area_m2,
            expected_area,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AerodynamicsConfigError(
                f"{condition.value} wing area {nominal.wing_area_m2} does "
                f"not match configured GTM reference area {expected_area}."
            )


@dataclass(frozen=True)
class AerodynamicsDatabase:
    """Validated flight-test aerodynamic data built from an already-loaded config."""

    _flights: Mapping[FlightCondition, FlightTestAeroData]

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> "AerodynamicsDatabase":
        if not isinstance(config, Mapping):
            raise TypeError(
                f"config must be a mapping, got {type(config).__name__}."
            )

        flights = {
            condition: _load_one_flight(config, condition)
            for condition in FlightCondition
        }
        _validate_reference_geometry_consistency(config, flights)
        return cls(_flights=MappingProxyType(flights))

    def get_flight_test_data(
        self,
        condition: FlightCondition,
    ) -> FlightTestAeroData:
        if not isinstance(condition, FlightCondition):
            raise TypeError(
                "condition must be a FlightCondition enum member."
            )
        return self._flights[condition]


# =============================================================================
# PERTURBATION / DYNAMIC PRESSURE CALCULATIONS
# =============================================================================

def compute_delta_alpha_rad(
    flight_data: FlightTestAeroData,
    absolute_alpha_rad: float,
) -> float:
    """Return alpha - alpha_trim for this exact flight record."""
    if not isinstance(flight_data, FlightTestAeroData):
        raise TypeError("flight_data must be FlightTestAeroData.")
    absolute_alpha = _validate_runtime_number(
        "absolute_alpha_rad",
        absolute_alpha_rad,
    )
    return absolute_alpha - flight_data.nominal_condition.angle_of_attack_rad


def _evaluate_validity(
    flight_data: FlightTestAeroData,
    *,
    delta_alpha_rad: float,
    delta_pitch_rate_rads: float,
    delta_elevator_rad: float,
) -> tuple[str, str]:
    limits = flight_data.validity_limits
    configured_limits = (
        ("delta_alpha_rad", abs(delta_alpha_rad), limits.max_abs_delta_alpha_rad),
        (
            "delta_pitch_rate_rads",
            abs(delta_pitch_rate_rads),
            limits.max_abs_delta_pitch_rate_rads,
        ),
        (
            "delta_elevator_rad",
            abs(delta_elevator_rad),
            limits.max_abs_delta_elevator_rad,
        ),
    )

    active = [(name, value, limit) for name, value, limit in configured_limits
              if limit is not None]

    if not active:
        return (
            "VALIDITY_REGION_NOT_QUANTIFIED",
            "No source-backed numerical perturbation bounds are configured. "
            "Treat this as a local linear model about the selected trim condition.",
        )

    exceeded = [
        f"{name}={value} exceeds configured |limit|={limit}"
        for name, value, limit in active
        if value > limit
    ]
    if exceeded:
        return (
            "OUTSIDE_CONFIGURED_VALIDITY_LIMITS",
            "; ".join(exceeded),
        )

    return (
        "WITHIN_CONFIGURED_VALIDITY_LIMITS",
        "All perturbations with configured source-backed limits are within bounds.",
    )


def compute_longitudinal_delta_coefficients(
    flight_data: FlightTestAeroData,
    delta_alpha_rad: float,
    delta_pitch_rate_rads: float,
    delta_elevator_rad: float,
) -> LongitudinalPerturbationCoefficients:
    """Compute deterministic mean delta_CL and delta_Cm for one flight record."""
    if not isinstance(flight_data, FlightTestAeroData):
        raise TypeError("flight_data must be FlightTestAeroData.")

    delta_alpha = _validate_runtime_number(
        "delta_alpha_rad",
        delta_alpha_rad,
    )
    delta_q = _validate_runtime_number(
        "delta_pitch_rate_rads",
        delta_pitch_rate_rads,
    )
    delta_e = _validate_runtime_number(
        "delta_elevator_rad",
        delta_elevator_rad,
    )

    nominal = flight_data.nominal_condition
    if nominal.airspeed_mps <= 0:
        raise ValueError("Selected flight nominal airspeed must be > 0.")
    if nominal.mac_m <= 0:
        raise ValueError("Selected flight nominal MAC must be > 0.")

    q_hat = (
        delta_q
        * nominal.mac_m
        / (2.0 * nominal.airspeed_mps)
    )

    d = flight_data.derivatives
    delta_CL = (
        d.CL_alpha.value * delta_alpha
        + d.CL_q.value * q_hat
        + d.CL_delta_e.value * delta_e
    )
    delta_Cm = (
        d.Cm_alpha.value * delta_alpha
        + d.Cm_q.value * q_hat
        + d.Cm_delta_e.value * delta_e
    )

    status, note = _evaluate_validity(
        flight_data,
        delta_alpha_rad=delta_alpha,
        delta_pitch_rate_rads=delta_q,
        delta_elevator_rad=delta_e,
    )

    return LongitudinalPerturbationCoefficients(
        delta_CL=delta_CL,
        delta_Cm=delta_Cm,
        validity_status=status,
        validity_note=note,
    )


def compute_nominal_dynamic_pressure(
    flight_data: FlightTestAeroData,
    atmosphere: AtmosphericState,
) -> float:
    """Compute q_bar = 0.5*rho*V_trim^2 using atmosphere.py density."""
    if not isinstance(flight_data, FlightTestAeroData):
        raise TypeError("flight_data must be FlightTestAeroData.")
    if not isinstance(atmosphere, AtmosphericState):
        raise TypeError("atmosphere must be AtmosphericState.")

    density = _validate_runtime_number(
        "atmosphere.density_kg_m3",
        atmosphere.density_kg_m3,
    )
    if density <= 0:
        raise ValueError("atmosphere.density_kg_m3 must be > 0.")

    nominal_altitude = flight_data.nominal_condition.altitude_m
    atmosphere_altitude = _validate_runtime_number(
        "atmosphere.altitude_m",
        atmosphere.altitude_m,
    )
    if not math.isclose(
        atmosphere_altitude,
        nominal_altitude,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "AtmosphericState altitude does not match the selected flight's "
            f"nominal altitude: {atmosphere_altitude} vs {nominal_altitude} m."
        )

    V_trim = flight_data.nominal_condition.airspeed_mps
    if V_trim <= 0:
        raise ValueError("Selected flight nominal airspeed must be > 0.")

    return 0.5 * density * V_trim ** 2


def compute_longitudinal_delta_forces(
    flight_data: FlightTestAeroData,
    delta_coeffs: LongitudinalPerturbationCoefficients,
    atmosphere: AtmosphericState,
) -> LongitudinalPerturbationForces:
    """Dimensionalize with the SAME flight record's nominal q_bar, S, and c_bar."""
    if not isinstance(flight_data, FlightTestAeroData):
        raise TypeError("flight_data must be FlightTestAeroData.")
    if not isinstance(
        delta_coeffs,
        LongitudinalPerturbationCoefficients,
    ):
        raise TypeError(
            "delta_coeffs must be LongitudinalPerturbationCoefficients."
        )

    delta_CL = _validate_runtime_number(
        "delta_coeffs.delta_CL",
        delta_coeffs.delta_CL,
    )
    delta_Cm = _validate_runtime_number(
        "delta_coeffs.delta_Cm",
        delta_coeffs.delta_Cm,
    )

    q_bar = compute_nominal_dynamic_pressure(
        flight_data,
        atmosphere,
    )

    nominal = flight_data.nominal_condition
    qS = q_bar * nominal.wing_area_m2

    return LongitudinalPerturbationForces(
        delta_lift_N=qS * delta_CL,
        delta_pitching_moment_Nm=qS * nominal.mac_m * delta_Cm,
        dynamic_pressure_Pa=q_bar,
    )
