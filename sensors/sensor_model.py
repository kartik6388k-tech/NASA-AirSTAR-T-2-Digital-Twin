"""
sensors/sensor_model.py

NASA AirSTAR T-2 Onboard Sensor Model (measurement noise/bias only)

Responsibility (and ONLY responsibility):

    TRUE STATE (delta_w, delta_q, perturbation_az)
        |
    additive sensor noise/bias model
        |
    SIMULATED SENSOR MEASUREMENT
    (measured_delta_w, measured_delta_q, measured_perturbation_az)

This module corrupts the noise-free TRUTH signals produced by
models/flight_dynamics.py -- delta_w, delta_q, and the accelerometer-
equivalent perturbation_az output of
FlightDynamics.calculate_state_derivatives() -- with a constant bias plus
zero-mean additive Gaussian noise, so downstream consumers (estimators,
dashboards, health monitors) can be developed and tested against something
closer to what an instrumented T-2 GTM would report, rather than the
noise-free integrator state.

PROVENANCE WARNING (read before using any number in this file):
    Hardware specifications (noise, bias, resolution, drift) for the actual
    T-2 GTM rate gyro, accelerometer, or air-data instruments were not
    found in the sources reviewed. However, flight-data noise estimates
    are available and should be used when modeling real sensor noise.
    Every numeric noise/bias parameter in `default_sensor_suite()` below
    is therefore a PROJECT_DEFINED_PLACEHOLDER, NOT a NASA-verified sensor
    specification. Treat outputs of this module as "a sensor model with
    the right shape and units", not as a validated representation of T-2
    instrumentation. Replace the defaults the moment real instrumentation
    specs (data sheets, calibration reports, or NASA documentation) become
    available, and update `model_origin` on the affected channel(s).
    Additionally, w_sensor is a synthetic channel. Keep it explicitly 
    PROJECT_DEFINED, not NASA-verified.

This module intentionally contains NO:
    - aerodynamic model, propulsion model, or atmosphere model
    - equations of motion / flight-dynamics integration
    - state estimation, sensor fusion, or Kalman filtering
    - health monitoring / fault injection logic
    - simulation loop
    - source/provenance verification of physical flight data (that remains
      in config.yaml / aerodynamics.py, per the pattern used elsewhere)

Dependency direction:

    models.flight_dynamics (truth)
        -> sensors.sensor_model (measurement)
        -> simulation.simulator / downstream consumers

Coordinate / signal conventions (same as models/flight_dynamics.py):
    delta_w:          perturbation body-axis vertical velocity (m/s)
    delta_q:          perturbation pitch rate (rad/s)
    perturbation_az:  perturbation specific force in body-Z (m/s^2), as
        computed by FlightDynamics.calculate_state_derivatives()
        ['outputs']['perturbation_az']. 
        WARNING: This is kept as an internal SI signal. It is NOT the NASA [g] 
        signal. Convert to g only for NASA comparison.

config.yaml note
-----------------
config.yaml (as provided) contains propulsion-side `sensor_bias` entries
for engine instrumentation (rpm_bias, pump_voltage_bias, under
`health_monitoring_states`) but defines no section for the flight-dynamics
state sensors modeled here (rate gyro / accelerometer / w-sensor). This
module does NOT read config.yaml by default and does NOT invent such a
section. `SensorModel.from_config()` will use a `sensors:` section if the
caller's config happens to provide one, but falls back to the
PROJECT_DEFINED_PLACEHOLDER defaults otherwise -- it never fabricates
NASA provenance for a value it fills in itself. A possible future
`sensors:` schema is sketched below purely as a placeholder for later
work, mirroring the same documented limitation in models/atmosphere.py
for ISA constants:

    sensors:
      w_sensor:
        bias: 0.0            # m/s
        noise_std: 0.0       # m/s
        status: "NOT_FOUND_DO_NOT_ASSUME"
      rate_gyro_q:
        bias: 0.0            # rad/s
        noise_std: 0.0       # rad/s
        status: "NOT_FOUND_DO_NOT_ASSUME"
      accelerometer_az:
        bias: 0.0            # m/s^2
        noise_std: 0.0       # m/s^2
        status: "NOT_FOUND_DO_NOT_ASSUME"
"""

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Validation helpers (local to this module, mirroring the pattern used in
# models/flight_dynamics.py and models/aerodynamics.py rather than importing
# another module's private helpers).
# ---------------------------------------------------------------------------


def _require_finite(name: str, value: Any) -> float:
    """Validate that *value* is a finite number (not bool). Returns float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def _require_non_negative_finite(name: str, value: Any) -> float:
    val = _require_finite(name, value)
    if val < 0.0:
        raise ValueError(f"{name} must be non-negative, got {val}")
    return val


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorChannelNoiseModel:
    """
    PROJECT_DEFINED additive measurement-error model for a single scalar
    sensor channel:

        measured = truth + bias + N(0, noise_std)

    `bias` and `noise_std` are PROJECT_DEFINED_PLACEHOLDER values unless the
    caller explicitly overrides them with sourced instrumentation data --
    see module docstring PROVENANCE WARNING. This class only validates
    structure and finiteness; it does not know or care whether its numbers
    are NASA-verified.
    """

    name: str
    units: str
    bias: float
    noise_std: float
    model_origin: str = "PROJECT_DEFINED_PLACEHOLDER"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Sensor channel name must be a non-empty string.")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("Sensor channel units must be a non-empty string.")
        if not isinstance(self.model_origin, str) or not self.model_origin.strip():
            raise ValueError("Sensor channel model_origin must be a non-empty string.")

        bias = _require_finite(f"{self.name}.bias", self.bias)
        noise_std = _require_non_negative_finite(f"{self.name}.noise_std", self.noise_std)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "noise_std", noise_std)


@dataclass(frozen=True)
class SensorSuiteParameters:
    """
    Bundles the PROJECT_DEFINED noise/bias models for every sensor channel
    exposed by this module. All defaults produced by `default_sensor_suite()`
    are placeholders -- see module docstring PROVENANCE WARNING.
    """

    w_sensor: SensorChannelNoiseModel
    rate_gyro_q: SensorChannelNoiseModel
    accelerometer_az: SensorChannelNoiseModel
    model_origin: str = "PROJECT_DEFINED_PLACEHOLDER_NOT_NASA_VERIFIED"

    def __post_init__(self) -> None:
        for field_name in ("w_sensor", "rate_gyro_q", "accelerometer_az"):
            val = getattr(self, field_name)
            if not isinstance(val, SensorChannelNoiseModel):
                raise TypeError(
                    f"{field_name} must be a SensorChannelNoiseModel, "
                    f"got {type(val).__name__}"
                )
        if not isinstance(self.model_origin, str) or not self.model_origin.strip():
            raise ValueError("SensorSuiteParameters.model_origin must be a non-empty string.")


def default_sensor_suite() -> SensorSuiteParameters:
    """
    Returns a SensorSuiteParameters populated with illustrative
    PROJECT_DEFINED_PLACEHOLDER noise/bias values.

    These numbers are NOT sourced from any NASA T-2 instrumentation
    document -- none was found (see module docstring). They exist only so
    this module is runnable out of the box and give roughly the right
    order of magnitude for a small MEMS-class rate gyro / accelerometer,
    nothing more. Replace them before trusting any downstream conclusion
    that depends on their exact values.
    """
    return SensorSuiteParameters(
        w_sensor=SensorChannelNoiseModel(
            name="w_sensor", units="m/s", bias=0.0, noise_std=0.05
        ),
        rate_gyro_q=SensorChannelNoiseModel(
            name="rate_gyro_q", units="rad/s", bias=0.0, noise_std=0.001
        ),
        accelerometer_az=SensorChannelNoiseModel(
            name="accelerometer_az", units="m/s^2", bias=0.0, noise_std=0.05
        ),
    )


def _channel_from_config(
    channel_cfg: Optional[Dict[str, Any]],
    default_channel: SensorChannelNoiseModel,
) -> SensorChannelNoiseModel:
    """
    Builds one SensorChannelNoiseModel from an optional config sub-mapping,
    falling back to `default_channel`'s values field-by-field when the
    mapping (or an individual key) is absent. Never fabricates a
    NASA-verified `model_origin` -- a config-sourced channel is always
    labeled PROJECT_DEFINED_CONFIG_SOURCED, distinct from the hard-coded
    PROJECT_DEFINED_PLACEHOLDER default, so callers can tell the two apart.
    """
    if channel_cfg is None:
        return default_channel

    if not isinstance(channel_cfg, dict):
        raise TypeError(
            f"sensors.{default_channel.name} must be a mapping, "
            f"got {type(channel_cfg).__name__}"
        )

    bias = channel_cfg.get("bias", default_channel.bias)
    noise_std = channel_cfg.get("noise_std", default_channel.noise_std)

    return SensorChannelNoiseModel(
        name=default_channel.name,
        units=default_channel.units,
        bias=bias,
        noise_std=noise_std,
        model_origin="PROJECT_DEFINED_CONFIG_SOURCED",
    )


# ---------------------------------------------------------------------------
# Sensor Model
# ---------------------------------------------------------------------------


class SensorModel:
    """
    Applies PROJECT_DEFINED additive noise/bias to the truth signals
    produced by models/flight_dynamics.py, producing simulated sensor
    measurements. This class does not filter, estimate, or fuse anything;
    it is a one-way truth -> measurement corruption model only.
    """

    def __init__(
        self,
        suite: Optional[SensorSuiteParameters] = None,
        seed: Optional[int] = None,
    ):
        if suite is None:
            suite = default_sensor_suite()
        if not isinstance(suite, SensorSuiteParameters):
            raise TypeError(
                f"suite must be SensorSuiteParameters, got {type(suite).__name__}"
            )
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError(f"seed must be an int or None, got {type(seed).__name__}")

        self.suite = suite
        # A dedicated RNG instance so this module's randomness is fully
        # reproducible given a seed, and never leaks into or is affected by
        # any global `random` state used elsewhere in the project.
        self._rng = random.Random(seed)

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], seed: Optional[int] = None
    ) -> "SensorModel":
        """
        Builds a SensorModel from an application config dict.

        If `config` has no `sensors` mapping (true for config.yaml as
        currently provided -- see module docstring), this returns the same
        PROJECT_DEFINED_PLACEHOLDER defaults as `default_sensor_suite()`.
        If a `sensors` mapping IS present, per-channel `bias`/`noise_std`
        overrides are read where given and defaulted otherwise; such
        channels are labeled PROJECT_DEFINED_CONFIG_SOURCED rather than
        PROJECT_DEFINED_PLACEHOLDER so it's traceable which numbers came
        from the config file.
        """
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")

        sensors_cfg = config.get("sensors")
        if sensors_cfg is not None and not isinstance(sensors_cfg, dict):
            raise TypeError(
                f"config['sensors'] must be a mapping, got {type(sensors_cfg).__name__}"
            )

        defaults = default_sensor_suite()

        def _cfg_for(name: str) -> Optional[Dict[str, Any]]:
            return sensors_cfg.get(name) if sensors_cfg else None

        suite = SensorSuiteParameters(
            w_sensor=_channel_from_config(_cfg_for("w_sensor"), defaults.w_sensor),
            rate_gyro_q=_channel_from_config(_cfg_for("rate_gyro_q"), defaults.rate_gyro_q),
            accelerometer_az=_channel_from_config(
                _cfg_for("accelerometer_az"), defaults.accelerometer_az
            ),
        )
        return cls(suite=suite, seed=seed)

    def _apply_channel(self, channel: SensorChannelNoiseModel, truth: float) -> float:
        truth = _require_finite(f"{channel.name} truth input", truth)
        noise = (
            self._rng.gauss(0.0, channel.noise_std)
            if channel.noise_std > 0.0
            else 0.0
        )
        measured = truth + channel.bias + noise
        if not math.isfinite(measured):
            raise ValueError(
                f"Simulated measurement for {channel.name} is not finite "
                f"({measured!r}); check the truth input and channel "
                f"bias/noise_std for extreme values."
            )
        return measured

    def measure(
        self, delta_w: float, delta_q: float, perturbation_az: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        Produces one simulated sensor sample from one truth sample.

        Args:
            delta_w: True perturbation body-axis vertical velocity (m/s),
                e.g. history['delta_w_mps'] from simulation/simulator.py.
            delta_q: True perturbation pitch rate (rad/s), e.g.
                history['delta_q_rads'] from simulation/simulator.py.
            perturbation_az: True perturbation specific force in body-Z
                (m/s^2), e.g. history['perturbation_az_mps2'] from
                simulation/simulator.py, or
                FlightDynamics.calculate_state_derivatives()
                ['outputs']['perturbation_az'] directly.

        Returns:
            dict with 'truth', 'measured', and 'sensor_model_status'
            entries. 'truth' echoes the validated inputs unchanged so
            callers never have to guess which side of the noise model a
            given value came from -- truth and measurement are always
            kept as clearly separate keys, never merged into one blob.

        Raises:
            TypeError / ValueError: if any input is not a finite number,
                or if a simulated measurement overflows to a non-finite
                value.
        """
        measured_w = self._apply_channel(self.suite.w_sensor, delta_w)
        measured_q = self._apply_channel(self.suite.rate_gyro_q, delta_q)
        measured_az = self._apply_channel(self.suite.accelerometer_az, perturbation_az)

        return {
            "truth": {
                "delta_w_mps": _require_finite("delta_w", delta_w),
                "delta_q_rads": _require_finite("delta_q", delta_q),
                "perturbation_az_mps2": _require_finite(
                    "perturbation_az", perturbation_az
                ),
            },
            "measured": {
                "delta_w_mps": measured_w,
                "delta_q_rads": measured_q,
                "perturbation_az_mps2": measured_az,
            },
            "sensor_model_status": self.suite.model_origin,
        }

