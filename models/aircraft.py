"""
models/aircraft.py

Physical aircraft configuration and state representation for the
NASA AirSTAR T-2 (5.5% dynamically scaled Generic Transport Model)
digital twin.

Layer boundary (STRICT):
    This module is the DATA / MODEL-CONFIGURATION layer only. It stores the
    active aircraft identity, geometry, mass properties, flight state, control
    inputs, and optional actuator limits. It validates structure, numeric type,
    finiteness, and physically meaningful ranges.

    It intentionally contains NO:
        - aerodynamic model
        - propulsion model
        - atmosphere model
        - equations of motion / flight-dynamics integration
        - simulation loop
        - sensor noise model
        - state estimation / filtering
        - health monitoring
        - source/provenance verification
        - dashboard logic

    Provenance remains in config.yaml. This class does NOT decide whether a
    value is NASA-verified; it only refuses to use physical parameters that are
    absent or explicitly marked NOT_FOUND_DO_NOT_ASSUME.

Dependency direction:

    config.yaml -> Aircraft -> Aerodynamics / Propulsion / Flight Dynamics -> Simulation

Coordinate conventions:
    Inertial/navigation position uses NED:
        x = North
        y = East
        z = Down

    Body axes follow the standard aircraft convention:
        +x = forward
        +y = right
        +z = down

    Euler angles and body rates must use the same NED/body-axis convention
    throughout the project.

Units for active Aircraft state/configuration (internal, always SI):
    length      m
    mass        kg
    time        s
    angle       rad
    ang. rate   rad/s
    velocity    m/s
    inertia     kg*m^2

This module never converts from imperial or performs hidden unit conversions.
If source data are stored in imperial units, config.yaml must provide the
already-derived SI values used by this class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


UNVERIFIED = "NOT_FOUND_DO_NOT_ASSUME"
EXPECTED_SCALE = 0.055
SCALE_ABS_TOL = 1.0e-4


class AircraftConfigError(ValueError):
    """Raised when the active aircraft configuration is missing or invalid."""


# ---------------------------------------------------------------------------
# Config lookup / type helpers
# ---------------------------------------------------------------------------


def _get_required(cfg: Dict[str, Any], path: str) -> Any:
    """Return a required nested value, rejecting missing/unverified values."""
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise AircraftConfigError(
                f"Required aircraft parameter '{path}' is not available in config.yaml."
            )
        node = node[key]

    if node == UNVERIFIED:
        raise AircraftConfigError(
            f"Aircraft parameter '{path}' is marked '{UNVERIFIED}' in config.yaml "
            "and cannot be used as an active physical parameter."
        )
    return node


def _get_optional(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Return an optional nested value; missing/unverified values become default."""
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    if node == UNVERIFIED:
        return default
    return node


def _is_number(value: Any) -> bool:
    """True only for real int/float scalars; bool is deliberately rejected."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _get_required_number(cfg: Dict[str, Any], path: str) -> float:
    """Fetch a required finite numeric parameter, rejecting bool and strings."""
    value = _get_required(cfg, path)
    if not _is_number(value):
        raise AircraftConfigError(
            f"Aircraft parameter '{path}' must be an int or float, "
            f"got {type(value).__name__}."
        )
    if not math.isfinite(value):
        raise AircraftConfigError(
            f"Aircraft parameter '{path}' must be finite, got {value!r}."
        )
    return float(value)


def _get_optional_number(
    cfg: Dict[str, Any], path: str, default: Optional[float] = None
) -> Optional[float]:
    """Fetch an optional finite numeric parameter without guessing a value."""
    value = _get_optional(cfg, path, default=None)
    if value is None:
        return default
    if not _is_number(value):
        raise AircraftConfigError(
            f"Aircraft parameter '{path}' must be an int or float when provided, "
            f"got {type(value).__name__}."
        )
    if not math.isfinite(value):
        raise AircraftConfigError(
            f"Aircraft parameter '{path}' must be finite when provided, got {value!r}."
        )
    return float(value)


def _get_required_string(cfg: Dict[str, Any], path: str) -> str:
    value = _get_required(cfg, path)
    if not isinstance(value, str) or not value.strip():
        raise AircraftConfigError(
            f"Aircraft parameter '{path}' must be a non-empty string."
        )
    return value.strip()


def _get_optional_string(
    cfg: Dict[str, Any], path: str, default: Optional[str] = None
) -> Optional[str]:
    value = _get_optional(cfg, path, default=None)
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise AircraftConfigError(
            f"Aircraft parameter '{path}' must be a non-empty string when provided."
        )
    return value.strip()


def _get_required_bool(cfg: Dict[str, Any], path: str) -> bool:
    value = _get_required(cfg, path)
    if type(value) is not bool:
        raise AircraftConfigError(
            f"Aircraft parameter '{path}' must be true or false, got {value!r}."
        )
    return value


def _finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(value)


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AircraftIdentification:
    name: str
    configuration: str
    scale: float
    aircraft_type: str


# ---------------------------------------------------------------------------
# Geometry -- explicit supported fields only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AircraftGeometry:
    wing_span_m: float   # b
    wing_area_m2: float  # S
    mac_m: float         # mean aerodynamic chord, c_bar

    @property
    def aspect_ratio(self) -> float:
        """Derived geometric aspect ratio: AR = b^2 / S."""
        return (self.wing_span_m ** 2) / self.wing_area_m2


# ---------------------------------------------------------------------------
# Mass properties -- explicit supported inertia products only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MassProperties:
    """Aircraft mass properties in SI units.

    CG is optional until a source-locked T-2 value and reference convention
    are available. If CG coordinates are supplied, all three coordinates plus
    ``cg_reference`` and ``cg_frame`` must be supplied together.

    Inertia products:
        Ixx, Iyy, Izz are positive moments of inertia.
        Ixy, Ixz, Iyz are stored with the sign supplied by config.yaml.
        aircraft.py does NOT independently verify the NASA/source sign
        convention and never silently flips a supplied product of inertia.

        The current validation uses product magnitudes for partial-tensor
        checks. A future 6-DOF inertia-matrix implementation must use the
        source-locked matrix/sign convention recorded by the project before
        applying Ixy/Ixz/Iyz dynamically.
    """

    mass_kg: float
    Ixx_kgm2: float
    Iyy_kgm2: float
    Izz_kgm2: float
    Ixz_kgm2: float

    # Optional products. Full tensor validation is enabled once both are known.
    Ixy_kgm2: Optional[float] = None
    Iyz_kgm2: Optional[float] = None

    # Optional CG. These remain None until verified and source-referenced.
    cg_x_m: Optional[float] = None
    cg_y_m: Optional[float] = None
    cg_z_m: Optional[float] = None
    cg_reference: Optional[str] = None
    cg_frame: Optional[str] = None

    @property
    def has_verified_cg(self) -> bool:
        return all(v is not None for v in (self.cg_x_m, self.cg_y_m, self.cg_z_m))

    @property
    def cg_m(self) -> Optional[tuple[float, float, float]]:
        """Return CG coordinates only when the complete CG definition exists."""
        if not self.has_verified_cg:
            return None
        return (self.cg_x_m, self.cg_y_m, self.cg_z_m)  # type: ignore[return-value]

    @property
    def has_full_inertia_tensor(self) -> bool:
        return self.Ixy_kgm2 is not None and self.Iyz_kgm2 is not None


# ---------------------------------------------------------------------------
# Flight state -- pure data container, NO equations of motion
# ---------------------------------------------------------------------------


@dataclass
class AircraftState:
    """Aircraft state using a project-wide NED/navigation convention.

    Position (NED):
        x_m = North, y_m = East, z_m = Down.

    Body-axis velocity:
        u_mps = forward, v_mps = right, w_mps = down.

    Euler angles are roll/pitch/yaw relative to the NED frame, and p/q/r are
    body-axis roll/pitch/yaw rates. Units are SI/radians.
    """

    x_m: float
    y_m: float
    z_m: float
    u_mps: float
    v_mps: float
    w_mps: float
    phi_rad: float
    theta_rad: float
    psi_rad: float
    p_rads: float
    q_rads: float
    r_rads: float

    def as_dict(self) -> Dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def is_finite(self) -> bool:
        """Reject non-numeric values, bool, NaN, and infinities."""
        for value in self.as_dict().values():
            if not _finite_number(value):
                return False
        return True


# ---------------------------------------------------------------------------
# Controls and optional source-verified actuator limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceDeflectionLimit:
    min_rad: float
    max_rad: float


@dataclass(frozen=True)
class ControlSurfaceLimits:
    """Optional verified limits. None means no verified limit is loaded."""

    elevator: Optional[SurfaceDeflectionLimit] = None
    aileron: Optional[SurfaceDeflectionLimit] = None
    rudder: Optional[SurfaceDeflectionLimit] = None


@dataclass
class ControlInputs:
    elevator_rad: float
    aileron_rad: float
    rudder_rad: float
    throttle_left: Optional[float] = None
    throttle_right: Optional[float] = None
    throttle_common: Optional[float] = None

    @property
    def has_independent_throttle(self) -> bool:
        return self.throttle_left is not None and self.throttle_right is not None

    def is_finite(self) -> bool:
        """Reject non-numeric values, bool, NaN, and infinities."""
        for attr in (
            "elevator_rad",
            "aileron_rad",
            "rudder_rad",
            "throttle_left",
            "throttle_right",
            "throttle_common",
        ):
            value = getattr(self, attr)
            if value is not None and not _finite_number(value):
                return False
        return True


# ---------------------------------------------------------------------------
# Aircraft
# ---------------------------------------------------------------------------


class Aircraft:
    """Physical configuration + current state for the NASA AirSTAR T-2 GTM.

    ``Aircraft`` validates structure, type, finiteness, identity, and physical
    ranges only. It does not verify NASA provenance; provenance remains the
    responsibility of config.yaml and the project's source-management layer.

    Expected config shape::

        aircraft:
          identification:
            name: "NASA AirSTAR ..."
            configuration: "... T-2 ..."
            scale: 0.055
            aircraft_type: "... GTM ..."

          geometry:
            wing_span_m: ...
            wing_area_m2: ...
            mac_m: ...

          mass_properties:
            mass_kg: ...
            Ixx_kgm2: ...
            Iyy_kgm2: ...
            Izz_kgm2: ...
            Ixz_kgm2: ...
            Ixy_kgm2: ...       # optional; only when verified
            Iyz_kgm2: ...       # optional; only when verified
            cg_x_m: ...         # optional; all CG fields required together
            cg_y_m: ...
            cg_z_m: ...
            cg_reference: ...
            cg_frame: ...

          initial_conditions:    # optional
            x_m: ...             # North
            y_m: ...             # East
            z_m: ...             # Down
            u_mps: ...
            v_mps: ...
            w_mps: ...
            phi_rad: ...
            theta_rad: ...
            psi_rad: ...
            p_rads: ...
            q_rads: ...
            r_rads: ...

          controls:              # optional
            has_independent_throttle: true | false
            elevator_rad: ...
            aileron_rad: ...
            rudder_rad: ...
            throttle_left: ...   # required only for independent throttle
            throttle_right: ...  # required only for independent throttle
            throttle_common: ... # required only for common throttle

            # Optional. Add only after the T-2 actuator limits are sourced.
            surface_limits_rad:
              elevator:
                min_rad: ...
                max_rad: ...
              aileron:
                min_rad: ...
                max_rad: ...
              rudder:
                min_rad: ...
                max_rad: ...

    Physical parameters marked ``NOT_FOUND_DO_NOT_ASSUME`` are unusable here.
    Evidence/provenance metadata elsewhere in config.yaml may legitimately keep
    that sentinel to record documented unknowns; this class does not consume
    those evidence sections.
    """

    EXPECTED_SCALE = EXPECTED_SCALE
    SCALE_ABS_TOL = SCALE_ABS_TOL

    REQUIRED_CONFIG_FIELDS = [
        "identification.name",
        "identification.configuration",
        "identification.scale",
        "identification.aircraft_type",
        "geometry.wing_span_m",
        "geometry.wing_area_m2",
        "geometry.mac_m",
        "mass_properties.mass_kg",
        "mass_properties.Ixx_kgm2",
        "mass_properties.Iyy_kgm2",
        "mass_properties.Izz_kgm2",
        "mass_properties.Ixz_kgm2",
    ]

    OPTIONAL_CG_CONFIG_FIELDS = [
        "mass_properties.cg_x_m",
        "mass_properties.cg_y_m",
        "mass_properties.cg_z_m",
        "mass_properties.cg_reference",
        "mass_properties.cg_frame",
    ]

    OPTIONAL_INERTIA_CONFIG_FIELDS = [
        "mass_properties.Ixy_kgm2",
        "mass_properties.Iyz_kgm2",
    ]

    INITIAL_STATE_CONFIG_FIELDS = [
        "initial_conditions.x_m",
        "initial_conditions.y_m",
        "initial_conditions.z_m",
        "initial_conditions.u_mps",
        "initial_conditions.v_mps",
        "initial_conditions.w_mps",
        "initial_conditions.phi_rad",
        "initial_conditions.theta_rad",
        "initial_conditions.psi_rad",
        "initial_conditions.p_rads",
        "initial_conditions.q_rads",
        "initial_conditions.r_rads",
    ]

    CONTROL_CONFIG_FIELDS = [
        "controls.has_independent_throttle",
        "controls.elevator_rad",
        "controls.aileron_rad",
        "controls.rudder_rad",
        "controls.throttle_left",
        "controls.throttle_right",
        "controls.throttle_common",
    ]

    def __init__(
        self,
        identification: AircraftIdentification,
        geometry: AircraftGeometry,
        mass_properties: MassProperties,
        state: Optional[AircraftState] = None,
        controls: Optional[ControlInputs] = None,
        control_surface_limits: Optional[ControlSurfaceLimits] = None,
    ) -> None:
        self.identification = identification
        self.geometry = geometry
        self.mass_properties = mass_properties
        self.state = state
        self.controls = controls
        self.control_surface_limits = control_surface_limits

        self._validate_static_parameters()
        self._validate_control_surface_limits()
        if self.state is not None:
            self._validate_state()
        if self.controls is not None:
            self._validate_controls()

    # -- construction from config.yaml ------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Aircraft":
        """Build one validated Aircraft from an already-loaded config dict."""
        if not isinstance(config, dict):
            raise AircraftConfigError("config.yaml root must be a mapping.")
        if "aircraft" not in config or not isinstance(config["aircraft"], dict):
            raise AircraftConfigError(
                "config.yaml must contain an 'aircraft:' mapping with "
                "identification, geometry, and mass_properties data."
            )

        cfg = config["aircraft"]

        identification = AircraftIdentification(
            name=_get_required_string(cfg, "identification.name"),
            configuration=_get_required_string(cfg, "identification.configuration"),
            scale=_get_required_number(cfg, "identification.scale"),
            aircraft_type=_get_required_string(cfg, "identification.aircraft_type"),
        )

        geometry = AircraftGeometry(
            wing_span_m=_get_required_number(cfg, "geometry.wing_span_m"),
            wing_area_m2=_get_required_number(cfg, "geometry.wing_area_m2"),
            mac_m=_get_required_number(cfg, "geometry.mac_m"),
        )

        mass_properties = MassProperties(
            mass_kg=_get_required_number(cfg, "mass_properties.mass_kg"),
            Ixx_kgm2=_get_required_number(cfg, "mass_properties.Ixx_kgm2"),
            Iyy_kgm2=_get_required_number(cfg, "mass_properties.Iyy_kgm2"),
            Izz_kgm2=_get_required_number(cfg, "mass_properties.Izz_kgm2"),
            Ixz_kgm2=_get_required_number(cfg, "mass_properties.Ixz_kgm2"),
            Ixy_kgm2=_get_optional_number(cfg, "mass_properties.Ixy_kgm2"),
            Iyz_kgm2=_get_optional_number(cfg, "mass_properties.Iyz_kgm2"),
            cg_x_m=_get_optional_number(cfg, "mass_properties.cg_x_m"),
            cg_y_m=_get_optional_number(cfg, "mass_properties.cg_y_m"),
            cg_z_m=_get_optional_number(cfg, "mass_properties.cg_z_m"),
            cg_reference=_get_optional_string(cfg, "mass_properties.cg_reference"),
            cg_frame=_get_optional_string(cfg, "mass_properties.cg_frame"),
        )

        state = (
            cls._build_state_from_config(cfg)
            if "initial_conditions" in cfg
            else None
        )
        controls = (
            cls._build_controls_from_config(cfg)
            if "controls" in cfg
            else None
        )
        control_surface_limits = cls._build_control_surface_limits_from_config(cfg)

        # Construct exactly once. Validation occurs inside __init__.
        return cls(
            identification=identification,
            geometry=geometry,
            mass_properties=mass_properties,
            state=state,
            controls=controls,
            control_surface_limits=control_surface_limits,
        )

    @staticmethod
    def _build_state_from_config(cfg: Dict[str, Any]) -> AircraftState:
        return AircraftState(
            x_m=_get_required_number(cfg, "initial_conditions.x_m"),
            y_m=_get_required_number(cfg, "initial_conditions.y_m"),
            z_m=_get_required_number(cfg, "initial_conditions.z_m"),
            u_mps=_get_required_number(cfg, "initial_conditions.u_mps"),
            v_mps=_get_required_number(cfg, "initial_conditions.v_mps"),
            w_mps=_get_required_number(cfg, "initial_conditions.w_mps"),
            phi_rad=_get_required_number(cfg, "initial_conditions.phi_rad"),
            theta_rad=_get_required_number(cfg, "initial_conditions.theta_rad"),
            psi_rad=_get_required_number(cfg, "initial_conditions.psi_rad"),
            p_rads=_get_required_number(cfg, "initial_conditions.p_rads"),
            q_rads=_get_required_number(cfg, "initial_conditions.q_rads"),
            r_rads=_get_required_number(cfg, "initial_conditions.r_rads"),
        )

    @staticmethod
    def _build_controls_from_config(cfg: Dict[str, Any]) -> ControlInputs:
        has_independent = _get_required_bool(
            cfg, "controls.has_independent_throttle"
        )

        throttle_left = _get_optional_number(cfg, "controls.throttle_left")
        throttle_right = _get_optional_number(cfg, "controls.throttle_right")
        throttle_common = _get_optional_number(cfg, "controls.throttle_common")

        if has_independent:
            if throttle_left is None or throttle_right is None:
                raise AircraftConfigError(
                    "controls.has_independent_throttle is true, so both "
                    "controls.throttle_left and controls.throttle_right are required."
                )
            if throttle_common is not None:
                raise AircraftConfigError(
                    "controls.throttle_common must be absent when independent "
                    "left/right throttles are configured."
                )
        else:
            if throttle_common is None:
                raise AircraftConfigError(
                    "controls.has_independent_throttle is false, so "
                    "controls.throttle_common is required."
                )
            if throttle_left is not None or throttle_right is not None:
                raise AircraftConfigError(
                    "controls.throttle_left/right must be absent when common "
                    "throttle is configured."
                )

        return ControlInputs(
            elevator_rad=_get_required_number(cfg, "controls.elevator_rad"),
            aileron_rad=_get_required_number(cfg, "controls.aileron_rad"),
            rudder_rad=_get_required_number(cfg, "controls.rudder_rad"),
            throttle_left=throttle_left,
            throttle_right=throttle_right,
            throttle_common=throttle_common,
        )

    @staticmethod
    def _build_control_surface_limits_from_config(
        cfg: Dict[str, Any]
    ) -> Optional[ControlSurfaceLimits]:
        """Read optional sourced actuator limits without inventing defaults.

        Supported config location:
            aircraft.controls.surface_limits_rad.<surface>.min_rad/max_rad

        If the whole block is absent or marked NOT_FOUND_DO_NOT_ASSUME, no
        surface-range check is applied. Individual surfaces may also be omitted
        until their limits are verified.
        """
        raw_limits = _get_optional(cfg, "controls.surface_limits_rad")
        if raw_limits is None:
            return None
        if not isinstance(raw_limits, dict):
            raise AircraftConfigError(
                "controls.surface_limits_rad must be a mapping when provided."
            )

        def build(surface: str) -> Optional[SurfaceDeflectionLimit]:
            raw = raw_limits.get(surface)
            if raw is None or raw == UNVERIFIED:
                return None
            if not isinstance(raw, dict):
                raise AircraftConfigError(
                    f"controls.surface_limits_rad.{surface} must be a mapping."
                )
            minimum = _get_required_number(
                cfg, f"controls.surface_limits_rad.{surface}.min_rad"
            )
            maximum = _get_required_number(
                cfg, f"controls.surface_limits_rad.{surface}.max_rad"
            )
            return SurfaceDeflectionLimit(min_rad=minimum, max_rad=maximum)

        limits = ControlSurfaceLimits(
            elevator=build("elevator"),
            aileron=build("aileron"),
            rudder=build("rudder"),
        )
        if all(
            limit is None
            for limit in (limits.elevator, limits.aileron, limits.rudder)
        ):
            return None
        return limits

    # -- derived, read-only ------------------------------------------------

    @property
    def aspect_ratio(self) -> float:
        return self.geometry.aspect_ratio

    # -- validation ---------------------------------------------------------

    def _validate_static_parameters(self) -> None:
        self._validate_identification()
        self._validate_geometry()
        self._validate_mass_properties()

    def _validate_identification(self) -> None:
        idn = self.identification

        if not all(
            isinstance(value, str) and bool(value.strip())
            for value in (idn.name, idn.configuration, idn.aircraft_type)
        ):
            raise AircraftConfigError(
                "Aircraft identification strings must be non-empty strings."
            )
        if not _finite_number(idn.scale):
            raise AircraftConfigError(
                f"identification.scale must be a finite number, got {idn.scale!r}."
            )
        if not math.isclose(
            float(idn.scale),
            self.EXPECTED_SCALE,
            rel_tol=0.0,
            abs_tol=self.SCALE_ABS_TOL,
        ):
            raise AircraftConfigError(
                "This module is locked to the 5.5% NASA AirSTAR T-2 GTM. "
                f"Expected identification.scale={self.EXPECTED_SCALE}, "
                f"got {idn.scale}."
            )

        identity = " ".join(
            (idn.name, idn.configuration, idn.aircraft_type)
        ).casefold()
        compact = "".join(ch for ch in identity if ch.isalnum())

        if "nasa" not in identity or "airstar" not in identity:
            raise AircraftConfigError(
                "Aircraft identification must explicitly identify NASA AirSTAR."
            )
        if "t2" not in compact:
            raise AircraftConfigError(
                "Aircraft identification must explicitly identify the T-2 configuration."
            )
        if "gtm" not in identity and "generic transport model" not in identity:
            raise AircraftConfigError(
                "Aircraft identification must explicitly identify the GTM "
                "(Generic Transport Model)."
            )

    def _validate_geometry(self) -> None:
        g = self.geometry
        for name, value in (
            ("wing_span_m", g.wing_span_m),
            ("wing_area_m2", g.wing_area_m2),
            ("mac_m", g.mac_m),
        ):
            if not _finite_number(value):
                raise AircraftConfigError(
                    f"{name} must be a finite numeric value, got {value!r}."
                )
            if value <= 0:
                raise AircraftConfigError(f"{name} must be positive, got {value}.")

    def _validate_mass_properties(self) -> None:
        m = self.mass_properties

        if not _finite_number(m.mass_kg) or m.mass_kg <= 0:
            raise AircraftConfigError(
                f"mass_kg must be a finite positive number, got {m.mass_kg!r}."
            )

        for name in ("Ixx_kgm2", "Iyy_kgm2", "Izz_kgm2"):
            value = getattr(m, name)
            if not _finite_number(value) or value <= 0:
                raise AircraftConfigError(
                    f"{name} must be a finite positive number, got {value!r}."
                )

        for name in ("Ixz_kgm2", "Ixy_kgm2", "Iyz_kgm2"):
            value = getattr(m, name)
            if value is not None and not _finite_number(value):
                raise AircraftConfigError(
                    f"{name} must be finite when provided, got {value!r}."
                )

        self._validate_cg_definition(m)
        self._validate_inertia_tensor(m)

    @staticmethod
    def _validate_cg_definition(m: MassProperties) -> None:
        cg_values = (m.cg_x_m, m.cg_y_m, m.cg_z_m)
        any_coordinate = any(value is not None for value in cg_values)
        all_coordinates = all(value is not None for value in cg_values)
        has_reference = m.cg_reference is not None
        has_frame = m.cg_frame is not None

        if not any_coordinate and not has_reference and not has_frame:
            return

        if not all_coordinates or not has_reference or not has_frame:
            raise AircraftConfigError(
                "CG is optional until verified, but a partial CG definition is not "
                "allowed. Provide cg_x_m, cg_y_m, cg_z_m, cg_reference, and "
                "cg_frame together, or omit/mark all of them NOT_FOUND_DO_NOT_ASSUME."
            )

        for name, value in (
            ("cg_x_m", m.cg_x_m),
            ("cg_y_m", m.cg_y_m),
            ("cg_z_m", m.cg_z_m),
        ):
            if not _finite_number(value):
                raise AircraftConfigError(
                    f"{name} must be a finite number when CG is provided."
                )

        if not isinstance(m.cg_reference, str) or not m.cg_reference.strip():
            raise AircraftConfigError("cg_reference must be a non-empty string.")
        if not isinstance(m.cg_frame, str) or not m.cg_frame.strip():
            raise AircraftConfigError("cg_frame must be a non-empty string.")

    @staticmethod
    def _validate_inertia_tensor(m: MassProperties) -> None:
        """Validate available principal minors; validate full tensor when known."""
        ixx, iyy, izz = m.Ixx_kgm2, m.Iyy_kgm2, m.Izz_kgm2
        ixy, ixz, iyz = m.Ixy_kgm2, m.Ixz_kgm2, m.Iyz_kgm2

        # Partial checks are valid even while Ixy/Iyz are not yet sourced.
        if abs(ixz) > math.sqrt(ixx * izz):
            raise AircraftConfigError(
                f"|Ixz_kgm2| ({abs(ixz):.6g}) exceeds sqrt(Ixx*Izz) "
                f"({math.sqrt(ixx * izz):.6g}); inertia sub-tensor is invalid."
            )
        if ixy is not None and abs(ixy) > math.sqrt(ixx * iyy):
            raise AircraftConfigError(
                "|Ixy_kgm2| exceeds sqrt(Ixx*Iyy); inertia sub-tensor is invalid."
            )
        if iyz is not None and abs(iyz) > math.sqrt(iyy * izz):
            raise AircraftConfigError(
                "|Iyz_kgm2| exceeds sqrt(Iyy*Izz); inertia sub-tensor is invalid."
            )

        # Full symmetric tensor validation only when every product is known.
        if ixy is None or iyz is None:
            return

        matrix = (
            (ixx, -ixy, -ixz),
            (-ixy, iyy, -iyz),
            (-ixz, -iyz, izz),
        )

        minor_xy = ixx * iyy - ixy * ixy
        minor_xz = ixx * izz - ixz * ixz
        minor_yz = iyy * izz - iyz * iyz
        determinant = (
            matrix[0][0]
            * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1]
            * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2]
            * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )

        if minor_xy <= 0 or minor_xz <= 0 or minor_yz <= 0 or determinant <= 0:
            raise AircraftConfigError(
                "The complete configured inertia tensor is not positive definite "
                "under the matrix convention used by this validation check."
            )

    def _validate_control_surface_limits(self) -> None:
        limits = self.control_surface_limits
        if limits is None:
            return

        for surface in ("elevator", "aileron", "rudder"):
            limit = getattr(limits, surface)
            if limit is None:
                continue
            if not _finite_number(limit.min_rad) or not _finite_number(limit.max_rad):
                raise AircraftConfigError(
                    f"{surface} control-surface limits must be finite numeric values."
                )
            if limit.min_rad >= limit.max_rad:
                raise AircraftConfigError(
                    f"{surface} control-surface min_rad must be less than max_rad."
                )

    def _validate_state(self) -> None:
        if self.state is not None and not self.state.is_finite():
            raise AircraftConfigError(
                "Aircraft state must contain only finite int/float values; bool, "
                "strings, NaN, and infinity are rejected."
            )

    def _validate_controls(self) -> None:
        controls = self.controls
        if controls is None:
            return

        if not controls.is_finite():
            raise AircraftConfigError(
                "Aircraft controls must contain only finite int/float values; bool, "
                "strings, NaN, and infinity are rejected."
            )

        independent_pair = (
            controls.throttle_left is not None and controls.throttle_right is not None
        )
        partial_independent = (
            (controls.throttle_left is None) != (controls.throttle_right is None)
        )

        if partial_independent:
            raise AircraftConfigError(
                "Independent throttle requires both throttle_left and throttle_right."
            )
        if independent_pair and controls.throttle_common is not None:
            raise AircraftConfigError(
                "Do not provide throttle_common together with left/right throttles."
            )
        if not independent_pair and controls.throttle_common is None:
            raise AircraftConfigError(
                "Controls must provide either independent left/right throttles or "
                "one common throttle."
            )

        for name in ("throttle_left", "throttle_right", "throttle_common"):
            value = getattr(controls, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise AircraftConfigError(
                    f"{name} must satisfy 0 <= throttle <= 1, got {value}."
                )

        limits = self.control_surface_limits
        if limits is None:
            return

        for surface, command_name in (
            ("elevator", "elevator_rad"),
            ("aileron", "aileron_rad"),
            ("rudder", "rudder_rad"),
        ):
            limit = getattr(limits, surface)
            if limit is None:
                continue
            value = getattr(controls, command_name)
            if not limit.min_rad <= value <= limit.max_rad:
                raise AircraftConfigError(
                    f"{command_name}={value} rad is outside the sourced T-2 "
                    f"limit [{limit.min_rad}, {limit.max_rad}] rad."
                )

    def __repr__(self) -> str:
        return (
            f"Aircraft(name={self.identification.name!r}, "
            f"configuration={self.identification.configuration!r}, "
            f"scale={self.identification.scale})"
        )
