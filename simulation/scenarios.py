"""
simulation/scenarios.py

Defines WHAT to simulate for the NASA AirSTAR T-2 / 5.5% GTM digital twin.

This module does NOT define HOW the physics work. It contains no
aerodynamic equations, no propulsion equations, no equations of motion, no
RK4/numerical integration, no atmospheric equations, and no plotting/CSV
logic. All of that remains in models/*.py and simulation/simulator.py. This
module only packages, per scenario:

    flight condition (which verified NASA derivative record, if any)
    initial perturbation state
    simulation duration / timestep
    control schedule (a function of time -> SimulationCommand)
    provenance / source / assumptions metadata

Reuses existing project types rather than duplicating them:
    - FlightCondition        (models.aerodynamics)
    - PerturbationState      (models.flight_dynamics)
    - SimulationCommand      (simulation.simulator)

=============================================================================
NASA SOURCE SEPARATION (do not merge)
=============================================================================
AIAA 2015-2704
    T-2 Flight 41 / Flight 15 -- nominal flight conditions and identified
    longitudinal derivatives. Loaded through config.yaml ->
    models.aerodynamics.AerodynamicsDatabase. Cross-checked in this module's
    test suite against the reference-condition figures below; config.yaml
    is the authoritative numeric source consumed by the rest of the
    pipeline (this module does not re-derive or hardcode those numbers):

        Flight 41: V=139.1 ft/s, alpha=4.077 deg, h=1227 ft, throttle=0.290
        Flight 15: V=136.0 ft/s, alpha=3.893 deg, h=1467 ft, throttle=0.315

    The actual Flight 41 / Flight 15 CONTROL time histories are NOT
    publicly verified. Any control schedule paired with these reference
    conditions is therefore PROJECT_DEFINED and must never be presented as
    a reproduction of the recorded maneuver.

AIAA 2020-0287
    T-2 Flight 33 -- multisine excitation characteristics (nominal
    V~=135 ft/s, alpha~=4.5 deg, h~=1400 ft; 10 s duration; 7 frequencies;
    stated 0.2-2.2 Hz range; stated 0.3 Hz spacing; elevator + aileron +
    rudder; orthogonal phase-optimized multisine). No FlightCondition enum
    member or config.yaml record exists for Flight 33 in this project, so
    it is never assigned FlightCondition.FLIGHT_41 or FLIGHT_15 -- doing so
    would merge Flight 33 with Flight 41/15, which is explicitly
    prohibited. The exact per-channel frequency list, phases, and
    amplitudes NASA used are not publicly available and are NOT
    reconstructed here.

=============================================================================
PROVENANCE CATEGORIES
=============================================================================
Exactly these categories are used (Provenance enum below):
    NASA_VERIFIED_REFERENCE_CONDITION  - a concrete, derivative-backed
                                          FlightCondition record (V, alpha,
                                          h, throttle, identified
                                          derivatives) verified against a
                                          NASA source.
    NASA_VERIFIED_REFERENCE            - verified NASA facts (e.g. nominal
                                          condition, duration, frequency
                                          count/range/spacing) that are NOT
                                          backed by a config-loaded
                                          FlightCondition derivative record
                                          in this project.
    PROJECT_DEFINED                    - values chosen by this project,
                                          not sourced from NASA data.
    PROJECT_DEFINED_WAVEFORM           - a project-defined control
                                          waveform standing in for an
                                          unavailable exact NASA waveform.
    TEST_ONLY                          - values that exist purely to
                                          exercise the software, not to
                                          represent any flight.

A SimulationScenario carries TWO independent provenance fields --
`reference_condition_provenance` and `control_schedule_provenance` --
because a scenario's flight condition and its control input can (and for
Flight 41/15, must) carry different provenance. This is what makes it
impossible to confuse "NASA reference condition" with "actual NASA flight
reproduction": the control-schedule provenance is always visible
separately from the flight-condition provenance.

=============================================================================
CURRENT SIMULATOR LIMITATION (see models/flight_dynamics.py, simulator.py)
=============================================================================
The current simulator integrates only delta_w / delta_q via the 2-state
short-period equations, driven by the elevator channel only.
Aerodynamic force/moment and propulsion thrust are diagnostic-only and are
NOT fed back into the integrated state. Accordingly:
    - No scenario here assumes throttle -> thrust -> acceleration coupling.
    - No scenario here attempts a full 6-DOF definition.
    - The multisine scenario is elevator-only; SimulationCommand has no
      aileron/rudder fields at all, so those axes cannot be represented
      regardless of provenance labeling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Tuple

from models.aerodynamics import FlightCondition
from models.flight_dynamics import PerturbationState
from .simulator import SimulationCommand


# =============================================================================
# PROVENANCE
# =============================================================================

class Provenance(str, Enum):
    """Exactly the categories required by the scenario provenance rules."""

    NASA_VERIFIED_REFERENCE_CONDITION = "NASA_VERIFIED_REFERENCE_CONDITION"
    NASA_VERIFIED_REFERENCE = "NASA_VERIFIED_REFERENCE"
    PROJECT_DEFINED = "PROJECT_DEFINED"
    PROJECT_DEFINED_WAVEFORM = "PROJECT_DEFINED_WAVEFORM"
    TEST_ONLY = "TEST_ONLY"


# =============================================================================
# SCENARIO DATACLASS
# =============================================================================

@dataclass(frozen=True)
class SimulationScenario:
    """
    Defines WHAT to simulate. Contains no physics; only the inputs a
    simulation.simulator.Simulator needs (flight condition selection,
    initial state, timing, control schedule) plus strict provenance
    metadata.

    flight_condition may be None when no config-loaded, derivative-backed
    FlightCondition record exists for the scenario's nominal condition
    (see the Flight 33 / multisine discussion in the module docstring).
    In that case reference_condition_provenance may NOT be
    NASA_VERIFIED_REFERENCE_CONDITION (that label specifically claims a
    concrete derivative-backed record).
    """

    name: str
    description: str
    flight_condition: Optional[FlightCondition]
    reference_condition_provenance: Provenance
    initial_state: PerturbationState
    duration_s: float
    dt_s: float
    control_schedule: Callable[[float], SimulationCommand]
    control_schedule_provenance: Provenance
    source: str
    assumptions: Tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def __post_init__(self) -> None:
        for attr in ("name", "description", "source"):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"SimulationScenario.{attr} must be a non-empty string."
                )

        if self.flight_condition is not None and not isinstance(
            self.flight_condition, FlightCondition
        ):
            raise TypeError(
                "flight_condition must be a FlightCondition enum member or "
                f"None, got {type(self.flight_condition).__name__}."
            )

        if not isinstance(self.reference_condition_provenance, Provenance):
            raise TypeError(
                "reference_condition_provenance must be a Provenance enum "
                f"member, got {type(self.reference_condition_provenance).__name__}."
            )
        if not isinstance(self.control_schedule_provenance, Provenance):
            raise TypeError(
                "control_schedule_provenance must be a Provenance enum "
                f"member, got {type(self.control_schedule_provenance).__name__}."
            )

        if (
            self.reference_condition_provenance
            == Provenance.NASA_VERIFIED_REFERENCE_CONDITION
            and self.flight_condition is None
        ):
            raise ValueError(
                "reference_condition_provenance="
                "NASA_VERIFIED_REFERENCE_CONDITION requires a concrete, "
                "derivative-backed flight_condition; it cannot be claimed "
                "with flight_condition=None."
            )

        if not isinstance(self.initial_state, PerturbationState):
            raise TypeError(
                "initial_state must be a PerturbationState, got "
                f"{type(self.initial_state).__name__}."
            )

        for attr in ("duration_s", "dt_s"):
            value = getattr(self, attr)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"SimulationScenario.{attr} must be a number, got "
                    f"{type(value).__name__}."
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"SimulationScenario.{attr} must be finite, got {value!r}."
                )
            if value <= 0.0:
                raise ValueError(
                    f"SimulationScenario.{attr} must be > 0, got {value!r}."
                )
            object.__setattr__(self, attr, float(value))

        if self.dt_s > self.duration_s:
            raise ValueError(
                f"dt_s ({self.dt_s}) cannot exceed duration_s "
                f"({self.duration_s})."
            )

        if not callable(self.control_schedule):
            raise TypeError("control_schedule must be callable.")

        # Validate the schedule at several representative times, not only
        # t=0. This catches schedules that return invalid commands at
        # mid-duration or final-time (important for multisine and other
        # time-varying waveforms).
        _validation_times = [0.0]
        if self.duration_s > 0.0:
            _validation_times.append(self.duration_s / 2.0)
            _validation_times.append(self.duration_s)

        for t_sample in _validation_times:
            sample_command = self.control_schedule(t_sample)
            if not isinstance(sample_command, SimulationCommand):
                raise TypeError(
                    f"control_schedule({t_sample}) must return a "
                    f"SimulationCommand, got "
                    f"{type(sample_command).__name__}."
                )
            sample_command.validate()

        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        for item in self.assumptions:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    "Every item in assumptions must be a non-empty string."
                )

        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string.")

    @property
    def classification(self) -> str:
        """Human-readable combined provenance label, e.g. 'NASA_VERIFIED_
        REFERENCE + PROJECT_DEFINED_WAVEFORM' when the two provenance
        fields differ, or a single category when they match."""
        if self.reference_condition_provenance == self.control_schedule_provenance:
            return self.reference_condition_provenance.value
        return (
            f"{self.reference_condition_provenance.value} + "
            f"{self.control_schedule_provenance.value}"
        )


# =============================================================================
# SHARED CONTROL-SCHEDULE BUILDERS (no physics -- these just build
# Callable[[float], SimulationCommand] closures)
# =============================================================================

def _zero_control_schedule(t: float) -> SimulationCommand:
    """PROJECT_DEFINED / TEST_ONLY: zero elevator and zero engine commands
    at every time t. Left/right throttle, rpm, and lambda are left at
    SimulationCommand's own defaults (0.0) rather than duplicated here --
    propulsion is diagnostic-only in the current simulator (see
    simulator.py) and is not part of what this module defines."""
    if not math.isfinite(t):
        raise ValueError(f"t must be finite, got {t!r}")
    return SimulationCommand(delta_elevator_rad=0.0)


# --- Multisine-concept waveform (elevator only, PROJECT_DEFINED_WAVEFORM) --

_MULTISINE_NUM_FREQUENCIES = 7      # NASA_VERIFIED_REFERENCE (AIAA 2020-0287)
_MULTISINE_FREQ_MIN_HZ = 0.2        # NASA_VERIFIED_REFERENCE (stated range)
_MULTISINE_FREQ_MAX_HZ = 2.2        # NASA_VERIFIED_REFERENCE (stated range)
_MULTISINE_FREQ_SPACING_HZ = 0.3    # NASA_VERIFIED_REFERENCE (stated spacing)
_MULTISINE_DURATION_S = 10.0        # NASA_VERIFIED_REFERENCE (stated duration)
_MULTISINE_DEFAULT_TOTAL_AMPLITUDE_RAD = 0.05  # PROJECT_DEFINED_WAVEFORM


def _project_defined_multisine_frequencies_hz(
    num_frequencies: int = _MULTISINE_NUM_FREQUENCIES,
    freq_min_hz: float = _MULTISINE_FREQ_MIN_HZ,
    freq_spacing_hz: float = _MULTISINE_FREQ_SPACING_HZ,
    freq_max_hz: float = _MULTISINE_FREQ_MAX_HZ,
) -> Tuple[float, ...]:
    """
    PROJECT_DEFINED_WAVEFORM.

    Builds a simple evenly-spaced frequency list whose count, starting
    frequency, and spacing are consistent with the NASA-STATED aggregate
    facts from AIAA 2020-0287. The resulting frequency tuple is entirely
    PROJECT_DEFINED_WAVEFORM: the true per-channel frequency assignment
    NASA used is not publicly available and is not attempted here. Any
    resemblance to the real NASA list is coincidental, not verified.

    The generated frequencies are validated to stay within the stated
    0.2–2.2 Hz range; a ValueError is raised if they exceed freq_max_hz.
    """
    if not isinstance(num_frequencies, int) or num_frequencies <= 0:
        raise ValueError(
            f"num_frequencies must be a positive int, got {num_frequencies!r}."
        )
    for name, value in (
        ("freq_min_hz", freq_min_hz),
        ("freq_spacing_hz", freq_spacing_hz),
        ("freq_max_hz", freq_max_hz),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value!r}.")

    if freq_min_hz > freq_max_hz:
        raise ValueError(
            f"freq_min_hz ({freq_min_hz}) cannot exceed freq_max_hz "
            f"({freq_max_hz})."
        )

    frequencies_hz = tuple(
        freq_min_hz + i * freq_spacing_hz for i in range(num_frequencies)
    )

    if frequencies_hz and max(frequencies_hz) > freq_max_hz:
        raise ValueError(
            f"Generated frequency {max(frequencies_hz):.4f} Hz exceeds the "
            f"stated NASA range upper bound of {freq_max_hz} Hz. Adjust "
            f"num_frequencies, freq_min_hz, or freq_spacing_hz so that all "
            f"generated frequencies remain within [{freq_min_hz}, "
            f"{freq_max_hz}] Hz."
        )

    return frequencies_hz


def _make_multisine_elevator_schedule(
    frequencies_hz: Tuple[float, ...],
    total_amplitude_rad: float,
    duration_s: float,
) -> Callable[[float], SimulationCommand]:
    """
    PROJECT_DEFINED_WAVEFORM elevator-only demonstration schedule:

        delta_e(t) = (total_amplitude_rad / N) * sum_i sin(2*pi*f_i*t),
                     for 0 <= t <= duration_s, else 0.

    All phases are zero -- explicitly NOT NASA's orthogonal
    phase-optimized multisine (the true phase set is not publicly
    available and is not fabricated here). Aileron/rudder are not
    represented: SimulationCommand has no such fields, and the current
    simulator only integrates the elevator channel (see
    models/flight_dynamics.py).
    """
    if len(frequencies_hz) == 0:
        raise ValueError("frequencies_hz must be non-empty.")
    if not math.isfinite(total_amplitude_rad):
        raise ValueError(
            f"total_amplitude_rad must be finite, got {total_amplitude_rad!r}."
        )
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(f"duration_s must be positive and finite, got {duration_s!r}.")

    per_freq_amplitude_rad = total_amplitude_rad / len(frequencies_hz)

    def schedule(t: float) -> SimulationCommand:
        if not math.isfinite(t):
            raise ValueError(f"t must be finite, got {t!r}")
        if t < 0.0 or t > duration_s:
            delta_e = 0.0
        else:
            delta_e = per_freq_amplitude_rad * sum(
                math.sin(2.0 * math.pi * f_hz * t) for f_hz in frequencies_hz
            )
        return SimulationCommand(delta_elevator_rad=delta_e)

    return schedule


# =============================================================================
# SCENARIO CONSTRUCTORS
# =============================================================================

def create_zero_input_stability_scenario(
    flight_condition: FlightCondition = FlightCondition.FLIGHT_41,
    duration_s: float = 5.0,
    dt_s: float = 0.02,
) -> SimulationScenario:
    """
    TEST_ONLY: verify equilibrium / zero-input numerical behavior.

    Zero initial perturbation state, zero elevator, zero engine commands
    at every time. This is a software test of the integration pipeline,
    NOT a NASA maneuver -- with zero state and zero input the integrated
    state must remain identically zero for all t. `flight_condition` only
    selects which verified NASA mass/inertia/derivative record (see
    models.aerodynamics.AerodynamicsDatabase) backs the numerical
    pipeline; it does not imply this scenario represents measured NASA
    flight behavior.
    """
    return SimulationScenario(
        name="zero_input_stability",
        description=(
            "Zero initial perturbation, zero elevator, zero engine "
            "commands. Verifies the integrated short-period state stays "
            "at rest. This is a software equilibrium check, NOT a NASA "
            "flight maneuver."
        ),
        flight_condition=flight_condition,
        reference_condition_provenance=Provenance.TEST_ONLY,
        initial_state=PerturbationState(delta_w=0.0, delta_q=0.0),
        duration_s=duration_s,
        dt_s=dt_s,
        control_schedule=_zero_control_schedule,
        control_schedule_provenance=Provenance.TEST_ONLY,
        source=(
            "TEST_ONLY software check. flight_condition only selects the "
            "verified mass/inertia/derivative record from "
            "models.aerodynamics.AerodynamicsDatabase used to run the "
            "pipeline; the zero-input/zero-state design of this scenario "
            "is not sourced from NASA flight-test data."
        ),
        assumptions=(
            "Initial state and control input are TEST_ONLY, chosen purely "
            "to exercise the numerical integration path at equilibrium.",
            "duration_s and dt_s are PROJECT_DEFINED numerical parameters, "
            "not NASA flight-test data.",
        ),
    )


def create_small_disturbance_scenario(
    flight_condition: FlightCondition = FlightCondition.FLIGHT_41,
    delta_w0_mps: float = 1.0,
    delta_q0_rads: float = 0.01,
    duration_s: float = 5.0,
    dt_s: float = 0.02,
) -> SimulationScenario:
    """
    TEST_ONLY: observe the free (zero-input) short-period response to an
    explicit, arbitrary TEST_ONLY perturbation. `delta_w0_mps` and
    `delta_q0_rads` are software test values chosen to exercise the
    dynamics -- they are NOT measured NASA perturbations and must not be
    presented as such.
    """
    return SimulationScenario(
        name="small_disturbance_free_response",
        description=(
            "Explicit TEST_ONLY initial perturbation "
            f"(delta_w0={delta_w0_mps} m/s, delta_q0={delta_q0_rads} rad/s) "
            "with zero control input, to observe the free short-period "
            "response. The perturbation values are software test inputs, "
            "not measured NASA flight-test perturbations."
        ),
        flight_condition=flight_condition,
        reference_condition_provenance=Provenance.TEST_ONLY,
        initial_state=PerturbationState(
            delta_w=delta_w0_mps, delta_q=delta_q0_rads
        ),
        duration_s=duration_s,
        dt_s=dt_s,
        control_schedule=_zero_control_schedule,
        control_schedule_provenance=Provenance.TEST_ONLY,
        source=(
            "TEST_ONLY software check. flight_condition only selects the "
            "verified mass/inertia/derivative record from "
            "models.aerodynamics.AerodynamicsDatabase used to run the "
            "pipeline; the perturbation values themselves are not sourced "
            "from NASA flight-test data."
        ),
        assumptions=(
            "delta_w0_mps and delta_q0_rads are explicit TEST_ONLY "
            "perturbation values, not measured NASA perturbations.",
            "Control input is zero (TEST_ONLY) so the resulting response "
            "is the unforced/free short-period response only.",
            "duration_s and dt_s are PROJECT_DEFINED numerical parameters.",
        ),
    )


def create_flight_41_scenario(
    duration_s: float = 10.0,
    dt_s: float = 0.02,
) -> SimulationScenario:
    """
    NASA_VERIFIED_REFERENCE_CONDITION + PROJECT_DEFINED control.

    Reference-condition run using T-2 Flight 41 nominal flight condition
    and identified longitudinal derivatives (AIAA 2015-2704), loaded
    through config.yaml -> models.aerodynamics.AerodynamicsDatabase
    (V=139.1 ft/s, alpha=4.077 deg, h=1227 ft, throttle=0.290).

    Control schedule is explicitly PROJECT_DEFINED (zero-input): the
    actual Flight 41 control time history is not publicly verified and is
    NOT reproduced here. This is a reference-condition run, NOT a flight
    reproduction.

    duration_s is PROJECT_DEFINED and does not represent a Flight 41
    maneuver duration.
    """
    return SimulationScenario(
        name="flight_41_reference_condition_run",
        description=(
            "Reference-condition run using T-2 Flight 41 verified nominal "
            "flight condition and identified longitudinal derivatives "
            "(AIAA 2015-2704), with a PROJECT_DEFINED zero-input control "
            "schedule. This is a reference-condition run, NOT a "
            "reproduction of the actual Flight 41 maneuver or control "
            "time history."
        ),
        flight_condition=FlightCondition.FLIGHT_41,
        reference_condition_provenance=Provenance.NASA_VERIFIED_REFERENCE_CONDITION,
        initial_state=PerturbationState(delta_w=0.0, delta_q=0.0),
        duration_s=duration_s,
        dt_s=dt_s,
        control_schedule=_zero_control_schedule,
        control_schedule_provenance=Provenance.PROJECT_DEFINED,
        source=(
            "AIAA 2015-2704 (T-2 Flight 41 nominal flight condition and "
            "identified longitudinal derivatives), as loaded via "
            "config.yaml -> models.aerodynamics.AerodynamicsDatabase. The "
            "control schedule is PROJECT_DEFINED and is not sourced from "
            "NASA flight-recorded controls."
        ),
        assumptions=(
            "Reference condition (V, alpha, h, throttle) and identified "
            "longitudinal derivatives are NASA-verified per AIAA 2015-2704 "
            "/ config.yaml.",
            "The actual Flight 41 elevator time history is not publicly "
            "verified; the zero-input control schedule used here is "
            "PROJECT_DEFINED and must not be presented as a reproduction "
            "of the real maneuver.",
            "duration_s is a PROJECT_DEFINED numerical integration "
            "parameter, not a Flight 41 maneuver duration. dt_s is also "
            "PROJECT_DEFINED.",
            "Left/right throttle, rpm, and lambda are left at "
            "SimulationCommand's defaults (0.0) rather than duplicated "
            "here; propulsion is diagnostic-only in the current simulator.",
        ),
    )


def create_flight_15_scenario(
    duration_s: float = 10.0,
    dt_s: float = 0.02,
) -> SimulationScenario:
    """
    NASA_VERIFIED_REFERENCE_CONDITION + PROJECT_DEFINED control.

    Reference-condition run using T-2 Flight 15 nominal flight condition
    and identified longitudinal derivatives (AIAA 2015-2704), loaded
    through config.yaml -> models.aerodynamics.AerodynamicsDatabase
    (V=136.0 ft/s, alpha=3.893 deg, h=1467 ft, throttle=0.315).

    Control schedule is explicitly PROJECT_DEFINED (zero-input): the
    actual Flight 15 control time history is not publicly verified and is
    NOT reproduced here. This is a reference-condition run, NOT a flight
    reproduction.

    duration_s is PROJECT_DEFINED and does not represent a Flight 15
    maneuver duration.
    """
    return SimulationScenario(
        name="flight_15_reference_condition_run",
        description=(
            "Reference-condition run using T-2 Flight 15 verified nominal "
            "flight condition and identified longitudinal derivatives "
            "(AIAA 2015-2704), with a PROJECT_DEFINED zero-input control "
            "schedule. This is a reference-condition run, NOT a "
            "reproduction of the actual Flight 15 maneuver or control "
            "time history."
        ),
        flight_condition=FlightCondition.FLIGHT_15,
        reference_condition_provenance=Provenance.NASA_VERIFIED_REFERENCE_CONDITION,
        initial_state=PerturbationState(delta_w=0.0, delta_q=0.0),
        duration_s=duration_s,
        dt_s=dt_s,
        control_schedule=_zero_control_schedule,
        control_schedule_provenance=Provenance.PROJECT_DEFINED,
        source=(
            "AIAA 2015-2704 (T-2 Flight 15 nominal flight condition and "
            "identified longitudinal derivatives), as loaded via "
            "config.yaml -> models.aerodynamics.AerodynamicsDatabase. The "
            "control schedule is PROJECT_DEFINED and is not sourced from "
            "NASA flight-recorded controls."
        ),
        assumptions=(
            "Reference condition (V, alpha, h, throttle) and identified "
            "longitudinal derivatives are NASA-verified per AIAA 2015-2704 "
            "/ config.yaml.",
            "The actual Flight 15 elevator time history is not publicly "
            "verified; the zero-input control schedule used here is "
            "PROJECT_DEFINED and must not be presented as a reproduction "
            "of the real maneuver.",
            "duration_s is a PROJECT_DEFINED numerical integration "
            "parameter, not a Flight 15 maneuver duration. dt_s is also "
            "PROJECT_DEFINED.",
            "Left/right throttle, rpm, and lambda are left at "
            "SimulationCommand's defaults (0.0) rather than duplicated "
            "here; propulsion is diagnostic-only in the current simulator.",
        ),
    )


def create_multisine_excitation_scenario(
    total_amplitude_rad: float = _MULTISINE_DEFAULT_TOTAL_AMPLITUDE_RAD,
    duration_s: float = _MULTISINE_DURATION_S,
    dt_s: float = 0.01,
    num_frequencies: int = _MULTISINE_NUM_FREQUENCIES,
    freq_min_hz: float = _MULTISINE_FREQ_MIN_HZ,
    freq_spacing_hz: float = _MULTISINE_FREQ_SPACING_HZ,
) -> SimulationScenario:
    """
    NASA_VERIFIED_REFERENCE + PROJECT_DEFINED_WAVEFORM.

    REFERENCE / DESIGN-ONLY scenario: flight_condition is None because no
    config-loaded, derivative-backed FlightCondition record exists for T-2
    Flight 33 in this project. This scenario therefore CANNOT be run
    directly through the current Simulator, which requires
    FlightTestAeroData (backed by FLIGHT_41 or FLIGHT_15). It is retained
    as a reference/design artifact documenting the multisine excitation
    concept and its provenance. To actually simulate this waveform,
    explicitly pair it with a concrete FlightCondition (e.g. FLIGHT_41)
    and document that pairing as PROJECT_DEFINED.

    NASA-verified aggregate facts (AIAA 2020-0287, T-2 Flight 33
    multisine excitation): nominal V~=135 ft/s, alpha~=4.5 deg,
    h~=1400 ft; maneuver duration = 10 s; 7 excitation frequencies;
    stated 0.2-2.2 Hz range; stated 0.3 Hz spacing; controls = elevator +
    aileron + rudder; maneuver = orthogonal phase-optimized multisine.

    NOT reproduced (unavailable / must not be fabricated): the exact
    per-channel frequency assignment, exact phases, exact amplitudes.

    The generated frequency tuple is entirely PROJECT_DEFINED_WAVEFORM:
    it is built by evenly spacing 7 project-defined sinusoids consistent
    with the reported aggregate characteristics (count, starting
    frequency, spacing). The aggregate values (7 frequencies, 0.2–2.2 Hz,
    0.3 Hz spacing) remain NASA_VERIFIED_REFERENCE data.

    total_amplitude_rad is PROJECT_DEFINED_WAVEFORM (not a NASA
    amplitude).

    This project cannot represent aileron/rudder at all (SimulationCommand
    has no such fields), and the current simulator only integrates the
    elevator channel through the 2-state short-period model. This scenario
    is therefore an ELEVATOR-ONLY, PROJECT_DEFINED_WAVEFORM demonstration
    (a simple zero-phase sum of sinusoids at the stated count/range/
    spacing) and must never be presented as the complete NASA three-axis
    excitation or an exact reconstruction of the true waveform.
    """
    frequencies_hz = _project_defined_multisine_frequencies_hz(
        num_frequencies=num_frequencies,
        freq_min_hz=freq_min_hz,
        freq_spacing_hz=freq_spacing_hz,
    )
    schedule = _make_multisine_elevator_schedule(
        frequencies_hz=frequencies_hz,
        total_amplitude_rad=total_amplitude_rad,
        duration_s=duration_s,
    )
    return SimulationScenario(
        name="multisine_excitation_concept",
        description=(
            "REFERENCE/DESIGN-ONLY. Elevator-only, PROJECT_DEFINED_WAVEFORM "
            "demonstration of the T-2 Flight 33 multisine excitation "
            "concept (AIAA 2020-0287): 7 project-defined sinusoids "
            "consistent with the reported aggregate characteristics "
            f"(from {freq_min_hz:.1f} Hz in {freq_spacing_hz:.1f} Hz "
            f"steps), zero phase, total_amplitude_rad="
            f"{total_amplitude_rad} (PROJECT_DEFINED_WAVEFORM), over "
            f"{duration_s:.1f} s. Does NOT reconstruct the exact NASA "
            "frequencies, phases, or amplitudes, and does NOT represent "
            "the aileron/rudder axes. Cannot run through the current "
            "Simulator without pairing with a concrete FlightCondition."
        ),
        flight_condition=None,
        reference_condition_provenance=Provenance.NASA_VERIFIED_REFERENCE,
        initial_state=PerturbationState(delta_w=0.0, delta_q=0.0),
        duration_s=duration_s,
        dt_s=dt_s,
        control_schedule=schedule,
        control_schedule_provenance=Provenance.PROJECT_DEFINED_WAVEFORM,
        source=(
            "AIAA 2020-0287 (T-2 Flight 33): nominal V~=135 ft/s, "
            "alpha~=4.5 deg, h~=1400 ft, 10 s duration, 7 frequencies, "
            "stated 0.2-2.2 Hz range, 0.3 Hz spacing, elevator+aileron+"
            "rudder, orthogonal phase-optimized multisine -- verified "
            "aggregate facts only. The generated frequency list is "
            "PROJECT_DEFINED_WAVEFORM (not the exact NASA frequency "
            "assignment). total_amplitude_rad is PROJECT_DEFINED_WAVEFORM."
        ),
        assumptions=(
            "Nominal flight condition, duration, frequency count/range/"
            "spacing, and control-surface set are NASA-verified aggregate "
            "facts per AIAA 2020-0287.",
            "The generated frequency tuple is entirely "
            "PROJECT_DEFINED_WAVEFORM: 7 project-defined sinusoids "
            "consistent with the reported aggregate characteristics "
            "(count, start, spacing). It is NOT the reconstructed NASA "
            "per-channel frequency assignment.",
            "Phases and total_amplitude_rad are PROJECT_DEFINED_WAVEFORM "
            "and are not claimed to match the real NASA excitation.",
            "No NASA-verified derivative record exists for Flight 33 in "
            "this project; flight_condition is None rather than "
            "substituting Flight 41 or Flight 15 data. This scenario is "
            "REFERENCE/DESIGN-ONLY and cannot run through the current "
            "Simulator without explicit pairing with a concrete "
            "FlightCondition.",
            "Only the elevator channel is represented; aileron and rudder "
            "are not modeled by SimulationCommand or the current "
            "simulator.",
            "dt_s is a PROJECT_DEFINED numerical-integration choice, not "
            "part of the NASA-documented maneuver.",
        ),
        notes=(
            "REFERENCE/DESIGN-ONLY: flight_condition=None means this "
            "scenario cannot run through the current Simulator, which "
            "requires FlightTestAeroData backed by FLIGHT_41 or "
            "FLIGHT_15. To simulate, explicitly pair with a concrete "
            "FlightCondition and document that pairing as PROJECT_DEFINED."
        ),
    )


def get_all_scenarios() -> Tuple[SimulationScenario, ...]:
    """Convenience registry of one instance of every scenario type."""
    return (
        create_zero_input_stability_scenario(),
        create_small_disturbance_scenario(),
        create_flight_41_scenario(),
        create_flight_15_scenario(),
        create_multisine_excitation_scenario(),
    )