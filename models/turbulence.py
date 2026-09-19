"""
models/turbulence.py

Project-Defined Longitudinal 1-Cosine Gust Model for the NASA AirSTAR T-2
Digital Twin reduced-order short-period dynamics.

Terminology & Scope:
- This is a deterministic 1-cosine vertical gust input, NOT stochastic
  continuous turbulence. Calling it 'turbulence' would be technically misleading;
  it is titled 'Project-Defined Longitudinal 1-Cosine Gust' while its provenance
  is classified under PROJECT_DEFINED_TURBULENCE.
- The gust velocity w_gust(t) is defined in the body-z axis (positive downward,
  consistent with standard aircraft flight dynamics body axes where z is down).
- Physical state space is strictly preserved as x = [delta_w, delta_q]^T.
  No 6-DOF turbulence states or lateral states are introduced.
- Evaluated as a perturbation to aerodynamic relative velocity:
      delta_w_aero(t) = delta_w(t) - w_gust(t)
- Parameters (amplitude, duration, start time) are PROJECT_DEFINED simulation
  inputs, NOT verified NASA aircraft data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


PROVENANCE_TAG = "PROJECT_DEFINED_TURBULENCE"
MODEL_NAME = "Project-Defined Longitudinal 1-Cosine Gust"


def _require_finite(name: str, value: Any) -> float:
    """Validate that value is a finite number (not bool). Returns float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class LongitudinalCosineGust:
    """
    Deterministic 1-cosine vertical gust input:

        w_gust(t) = (amplitude / 2) * (1 - cos(2 * pi * (t - start_time) / duration))
        for start_time <= t <= start_time + duration, and 0.0 otherwise.

    All parameters are PROJECT_DEFINED simulation inputs.
    """
    amplitude_mps: float
    duration_s: float
    start_time_s: float = 0.0
    enabled: bool = True
    provenance: str = PROVENANCE_TAG
    name: str = MODEL_NAME

    def __post_init__(self) -> None:
        object.__setattr__(self, "amplitude_mps", _require_finite("amplitude_mps", self.amplitude_mps))
        duration = _require_finite("duration_s", self.duration_s)
        if duration <= 0.0:
            raise ValueError(f"duration_s must be strictly positive, got {duration}")
        object.__setattr__(self, "duration_s", duration)

        start_time = _require_finite("start_time_s", self.start_time_s)
        if start_time < 0.0:
            raise ValueError(f"start_time_s must be non-negative, got {start_time}")
        object.__setattr__(self, "start_time_s", start_time)

        if not isinstance(self.enabled, bool):
            raise TypeError(f"enabled must be a bool, got {type(self.enabled).__name__}")

    def calculate_gust(self, t: float) -> float:
        """
        Calculates vertical gust velocity w_gust in m/s at time t.
        Returns 0.0 if disabled or outside the gust time window [start_time, start_time + duration].
        """
        t = _require_finite("t", t)
        if not self.enabled:
            return 0.0

        if t < self.start_time_s:
            return 0.0

        elapsed = t - self.start_time_s
        if elapsed > self.duration_s:
            return 0.0

        # 1-cosine discrete gust formulation
        tau = elapsed / self.duration_s
        return float(0.5 * self.amplitude_mps * (1.0 - math.cos(2.0 * math.pi * tau)))

    def __call__(self, t: float) -> float:
        """Allows the instance to be used directly as a Callable[[float], float]."""
        return self.calculate_gust(t)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        enable_override: Optional[bool] = None
    ) -> LongitudinalCosineGust:
        """
        Constructs a LongitudinalCosineGust instance from config.yaml dictionary.
        Looks under `project_defined_parameters.longitudinal_gust`.
        """
        proj_params = config.get("project_defined_parameters", {})
        gust_cfg = proj_params.get("longitudinal_gust", {})

        amplitude = float(gust_cfg.get("gust_amplitude_mps", 1.5))
        duration = float(gust_cfg.get("gust_duration_s", 1.0))
        start_time = float(gust_cfg.get("start_time_s", 1.0))
        enabled = bool(gust_cfg.get("enabled", False))

        if enable_override is not None:
            enabled = bool(enable_override)

        return cls(
            amplitude_mps=amplitude,
            duration_s=duration,
            start_time_s=start_time,
            enabled=enabled,
            provenance=str(gust_cfg.get("provenance", PROVENANCE_TAG)),
            name=str(gust_cfg.get("name", MODEL_NAME)),
        )
