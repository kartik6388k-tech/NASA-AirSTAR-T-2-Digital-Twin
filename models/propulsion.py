"""
models/propulsion.py

NASA AirSTAR T-2 / GTM Propulsion Model

Responsibility (and ONLY responsibility):

    throttle / RPM / engine state
        |
    propulsion model
        |
    left thrust + right thrust + total thrust

This module models propulsion ONLY. It contains:
    - Engine state representation
    - Propulsion input validation
    - Thrust model evaluation (plugin-based)
    - Propulsion output state

It intentionally contains NO:
    - Lift or drag calculation
    - Aerodynamic coefficients
    - Aircraft mass/inertia calculations
    - ISA equations
    - Flight-dynamics integration
    - Sensor filtering or Kalman filtering
    - Machine learning
    - Dashboard logic
    - Simulation loop

Architecture:
    config.yaml -> application layer loads once
                -> PropulsionModel.from_config(config) -> initialization
    each timestep: receive inputs -> calculate_thrust() -> PropulsionState

NASA-verified model structures (kept separate, NOT combined):
    A) engine RPM -> engine model -> thrust
       (system-identification, ground-test + ram-drag, Tang et al. 2009)
    B) pilot throttle -> first-order engine dynamics -> thrust
       (GTM nonlinear simulation, Grauer & Morelli 2015)

The code does NOT assert: throttle -> RPM -> thrust
unless a source explicitly establishes that complete chain.

Units (internal, always SI):
    thrust   -> N
    RPM      -> rpm
    throttle -> normalized [0, 1]
    airspeed -> m/s
"""

import math
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LBF_TO_N: float = 4.4482216152605  # Exact definition-based lbf -> N conversion

UNVERIFIED: str = "NOT_FOUND_DO_NOT_ASSUME"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PropulsionConfigError(ValueError):
    """Raised when the propulsion configuration is missing or invalid."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EngineState:
    """State of a single engine at one instant."""

    rpm: float
    throttle: float
    pump_voltage: float
    lambda_factor: float  # PROJECT_DEFINED thrust-loss factor [0, 1]
                          # lambda=0 -> healthy; lambda=1 -> complete modeled thrust loss
                          # This is PROJECT_DEFINED, not NASA-derived.


@dataclass
class PropulsionState:
    """Output of the propulsion model at one instant."""

    left_thrust_N: float
    right_thrust_N: float
    total_thrust_N: float
    left_engine: EngineState
    right_engine: EngineState
    model_status: str


# ---------------------------------------------------------------------------
# Propulsion Model
# ---------------------------------------------------------------------------


class PropulsionModel:
    """
    Propulsion Subsystem for NASA AirSTAR T-2 (GTM)

    Adheres strictly to verified NASA facts:
    - 2 JetCat P70 engines
    - Rated headline thrust: 16 lbf per engine (not a constant flight thrust)
    - Independent left/right engines
    - NASA verified structural relationships:
        RPM -> thrust (via ground test + ram drag)
        AND throttle -> first-order dynamics -> thrust
      These are NOT combined into an invented chain.

    This model explicitly REFUSES to fabricate a thrust map, equation, or
    time constant where NASA data is not provided.
    """

    # Fields in not_found_do_not_assume that must remain UNVERIFIED
    _REQUIRED_NOT_FOUND = (
        'rpm_to_thrust_static_map',
        'thrust_coefficient_C_T',
        'engine_time_constant_or_step_response',
        'left_right_engine_data',
        'engine_efficiency',
        'fuel_flow_map',
    )

    def __init__(
        self,
        *,
        num_engines: int,
        rated_thrust_lbf: float,
        rated_thrust_N: float,
        provenance: Dict[str, Any],
    ) -> None:
        """Direct constructor -- prefer from_config() or from_config_file().

        All parameters are keyword-only to prevent positional misuse.
        The caller is responsible for supplying validated values; use
        from_config() for automatic config-driven validation.
        """
        self.num_engines = num_engines
        self.rated_thrust_lbf = rated_thrust_lbf
        self.rated_thrust_N = rated_thrust_N
        self.provenance: Dict[str, Any] = dict(provenance)  # defensive copy
        self._initialize_engines()

    # -- construction from config -------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PropulsionModel":
        """Build a validated PropulsionModel from a pre-loaded config dict.

        This is the primary construction path.  The application layer should
        load config.yaml once and pass the resulting dict to all modules.
        """
        if not isinstance(config, dict):
            raise PropulsionConfigError(
                "config must be a dict (pre-loaded YAML)."
            )

        cls._validate_identity(config)

        nasa_data = config.get('nasa_verified_physical_data')
        if not isinstance(nasa_data, dict):
            raise PropulsionConfigError(
                "config must contain 'nasa_verified_physical_data' mapping."
            )

        # --- Engine count ---------------------------------------------------
        engines_section = nasa_data.get('engines')
        if not isinstance(engines_section, dict):
            raise PropulsionConfigError(
                "nasa_verified_physical_data.engines must be a mapping."
            )
        num_engines = engines_section.get('number_of_engines')
        if num_engines != 2:
            raise PropulsionConfigError(
                f"Expected 2 engines for T-2, got {num_engines}"
            )

        # --- Rated thrust ---------------------------------------------------
        thrust_section = nasa_data.get('thrust_rating_per_engine')
        if not isinstance(thrust_section, dict):
            raise PropulsionConfigError(
                "nasa_verified_physical_data.thrust_rating_per_engine "
                "must be a mapping."
            )
        rated_thrust_lbf = thrust_section.get('value_lbf')
        if rated_thrust_lbf != 16:
            raise PropulsionConfigError(
                f"Rated thrust must be exactly 16 lbf per NASA T-2 facts, "
                f"got {rated_thrust_lbf}"
            )

        rated_thrust_N = rated_thrust_lbf * LBF_TO_N

        # --- Provenance (source, document, year, exact location, status) ----
        provenance = {
            'thrust_rating_per_engine': {
                'source': thrust_section.get('source', {}),
                'status': thrust_section.get(
                    'value_semantics', 'UNKNOWN'
                ),
                'value_lbf': rated_thrust_lbf,
            }
        }

        # --- Verify NOT_FOUND_DO_NOT_ASSUME fields --------------------------
        not_found = config.get('not_found_do_not_assume')
        if not isinstance(not_found, dict):
            raise PropulsionConfigError(
                "config must contain 'not_found_do_not_assume' mapping."
            )
        for field_name in cls._REQUIRED_NOT_FOUND:
            if not_found.get(field_name) != UNVERIFIED:
                raise PropulsionConfigError(
                    f"'{field_name}' must remain '{UNVERIFIED}' -- "
                    f"got {not_found.get(field_name)!r}"
                )

        return cls(
            num_engines=num_engines,
            rated_thrust_lbf=rated_thrust_lbf,
            rated_thrust_N=rated_thrust_N,
            provenance=provenance,
        )

    @classmethod
    def from_config_file(cls, config_path: str) -> "PropulsionModel":
        """Convenience: load YAML from a file path, then call from_config.

        Provided for standalone use and testing.  In production the
        application layer should load config.yaml once and pass the dict
        to from_config() directly.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(config)

    # -- identity validation ------------------------------------------------

    @classmethod
    def _validate_identity(cls, config: Dict[str, Any]) -> None:
        """Verify that the configuration identifies NASA AirSTAR T-2 GTM."""
        identity = config.get('configuration_identity')
        if not isinstance(identity, dict):
            raise PropulsionConfigError(
                "config must contain 'configuration_identity' mapping."
            )

        # Aircraft identity
        aircraft = identity.get('aircraft', '')
        if not isinstance(aircraft, str):
            raise PropulsionConfigError(
                "configuration_identity.aircraft must be a string."
            )
        aircraft_lower = aircraft.casefold()
        if 'nasa' not in aircraft_lower or 'airstar' not in aircraft_lower:
            raise PropulsionConfigError(
                "configuration_identity.aircraft must identify NASA AirSTAR, "
                f"got {aircraft!r}"
            )
        compact = ''.join(ch for ch in aircraft_lower if ch.isalnum())
        if 't2' not in compact:
            raise PropulsionConfigError(
                "configuration_identity.aircraft must identify T-2, "
                f"got {aircraft!r}"
            )

        # Engine manufacturer
        manufacturer = identity.get('engine_manufacturer', '')
        if manufacturer != 'JetCat':
            raise PropulsionConfigError(
                f"engine_manufacturer must be 'JetCat', got {manufacturer!r}"
            )

        # Engine model
        model = identity.get('engine_model', '')
        if model != 'P70':
            raise PropulsionConfigError(
                f"engine_model must be 'P70', got {model!r}"
            )

    # -- engine initialization ----------------------------------------------

    def _initialize_engines(self) -> None:
        """Structurally independent left and right engines."""
        self.left_engine = EngineState(
            rpm=0.0, throttle=0.0, pump_voltage=0.0, lambda_factor=0.0
        )
        self.right_engine = EngineState(
            rpm=0.0, throttle=0.0, pump_voltage=0.0, lambda_factor=0.0
        )

    # -- input validation ---------------------------------------------------

    @staticmethod
    def validate_inputs(
        rpm: float,
        throttle: float,
        airspeed: float,
        lambda_factor: float,
        pump_voltage: float,
    ) -> None:
        """Strict mathematical constraints on propulsion inputs.

        Validates type, finiteness, and physical range for every input.
        """
        for name, value in (
            ("RPM", rpm),
            ("throttle", throttle),
            ("airspeed", airspeed),
            ("lambda_factor", lambda_factor),
            ("pump_voltage", pump_voltage),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be a number, got {type(value).__name__}"
                )
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")

        if rpm < 0:
            raise ValueError(f"RPM cannot be negative, got {rpm}")
        if throttle < 0.0 or throttle > 1.0:
            raise ValueError(f"Throttle must be in [0, 1], got {throttle}")
        if airspeed < 0:
            raise ValueError(f"Airspeed cannot be negative, got {airspeed}")
        if lambda_factor < 0.0 or lambda_factor > 1.0:
            raise ValueError(
                f"Lambda (health factor) must be in [0, 1], got {lambda_factor}"
            )
        if pump_voltage < 0:
            raise ValueError(
                f"Pump voltage cannot be negative, got {pump_voltage}"
            )

    # -- thrust calculation -------------------------------------------------

    def calculate_thrust(
        self,
        left_rpm: float,
        left_throttle: float,
        left_lambda: float,
        right_rpm: float,
        right_throttle: float,
        right_lambda: float,
        airspeed: float,
        left_pump_voltage: float = 0.0,
        right_pump_voltage: float = 0.0,
        thrust_model_plugin: Optional[
            Callable[[EngineState, float], float]
        ] = None,
    ) -> PropulsionState:
        """
        Calculate thrust each timestep.

        Args:
            left_rpm, right_rpm: Engine speeds in RPM.
            left_throttle, right_throttle: Normalized pilot throttle [0, 1].
            left_lambda, right_lambda: PROJECT_DEFINED thrust loss [0, 1].
                lambda=0 -> healthy; lambda=1 -> complete modeled thrust loss.
            airspeed: Airspeed in m/s.
            left_pump_voltage, right_pump_voltage: Fuel pump voltages (V).
            thrust_model_plugin: Injectable validated thrust map/equation.
                Signature: (engine_state: EngineState, airspeed: float) -> float
                Must return thrust in Newtons, finite and non-negative.

        Returns:
            PropulsionState containing SI thrust in Newtons and engine states.
        """
        # Validate all inputs including pump voltage
        self.validate_inputs(
            left_rpm, left_throttle, airspeed, left_lambda, left_pump_voltage
        )
        self.validate_inputs(
            right_rpm, right_throttle, airspeed, right_lambda, right_pump_voltage
        )

        # Update engine states independently
        self.left_engine = EngineState(
            left_rpm, left_throttle, left_pump_voltage, left_lambda
        )
        self.right_engine = EngineState(
            right_rpm, right_throttle, right_pump_voltage, right_lambda
        )

        if thrust_model_plugin is not None:
            # Evaluate the externally-provided thrust model
            raw_left = thrust_model_plugin(self.left_engine, airspeed)
            raw_right = thrust_model_plugin(self.right_engine, airspeed)

            # Validate plugin outputs
            if not math.isfinite(raw_left) or not math.isfinite(raw_right):
                raise ValueError("Thrust plugin returned non-finite thrust.")
            if raw_left < 0 or raw_right < 0:
                raise ValueError("Thrust plugin returned negative thrust.")

            # Apply PROJECT_DEFINED loss factors: T_actual = (1 - lambda) * T_expected
            left_thrust_N = raw_left * (1.0 - left_lambda)
            right_thrust_N = raw_right * (1.0 - right_lambda)

            return PropulsionState(
                left_thrust_N=left_thrust_N,
                right_thrust_N=right_thrust_N,
                total_thrust_N=left_thrust_N + right_thrust_N,
                left_engine=self.left_engine,
                right_engine=self.right_engine,
                model_status="PLUGIN_MODEL_EVALUATED",
            )

        # No numerical thrust model is yet identified by NASA sources.
        # DO NOT return 0, 16 lbf, 16*throttle, or an invented polynomial.
        return PropulsionState(
            left_thrust_N=float('nan'),
            right_thrust_N=float('nan'),
            total_thrust_N=float('nan'),
            left_engine=self.left_engine,
            right_engine=self.right_engine,
            model_status="MODEL_NOT_IDENTIFIED",
        )