# NASA AirSTAR T-2 Short-Period Scenario Research Report (FINAL v2)

**Date:** 2026-09-06  
**Author:** Scenario Development Team  
**Purpose:** Document provenance of scenario data for simulation/scenarios.py

---

## 1. NASA Maneuver Data FOUND

### 1.1 Flight Conditions

| Parameter | Flight 41 | Flight 15 | Source | Document | Section/Table |
|-----------|-----------|-----------|--------|----------|---------------|
| Airspeed (V) | 139.1 ft/s | 136.0 ft/s | Grauer & Boucher (2020) | AIAA 2020-2704 | Section VI.A, Table 2 |
| Angle of attack (α) | 4.077 deg | 3.893 deg | Grauer & Boucher (2020) | AIAA 2020-2704 | Section VI.A, Table 2 |
| Altitude (h) | 1227 ft | 1467 ft | Grauer & Boucher (2020) | AIAA 2020-2704 | Section VI.A, Table 2 |

**Source:** Grauer, J. A., & Boucher, M. J. (2020). "Aircraft System Identification from Multisine Inputs and Frequency Responses." AIAA 2020-2704, Section VI.A, p. 7-10.

### 1.2 Multisine Excitation Characteristics

| Characteristic | Value | Source | Document | Section |
|----------------|-------|--------|----------|---------|
| Duration (T) | 10 s | Grauer & Boucher (2020) | AIAA 2020-2704 | Section VI.A |
| Number of frequencies | 7 | Grauer & Boucher (2020) | AIAA 2020-2704 | Section VI.A |
| Frequency range | 0.2–2.2 Hz | Grauer & Boucher (2020) | AIAA 2020-2704 | Section VI.A |
| Frequency spacing | 0.3 Hz increments | Grauer & Boucher (2020) | AIAA 2020-2704 | Section VI.A |
| Control surfaces | elevator, aileron, rudder | Grauer & Boucher (2020) | AIAA 2020-2704 | Section VI.A |
| Maneuver type | Orthogonal phase-optimized multisine | Grauer & Boucher (2020) | AIAA 2020-2704 | Section V.A, VI.A |

**Exact quote from source (p. 7):**
> "Each multisine was designed for T = 10 s and included 7 frequencies evenly spaced between 0.2–2.2 Hz in 0.3 Hz increments."

**Important Note on Frequency Reconstruction:**
The source states "7 frequencies evenly spaced between 0.2–2.2 Hz in 0.3 Hz increments." A literal reconstruction (0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0) yields 7 frequencies but does not reach 2.2 Hz. This report treats the source wording as authoritative and does not attempt to reconstruct the exact frequency list. The three stated parameters (7 frequencies, 0.2–2.2 Hz range, 0.3 Hz increments) should all be recorded as given without inferring a specific frequency sequence.

### 1.3 Nominal Flight Condition for Multisine Maneuvers

| Parameter | Value | Source | Document | Section |
|-----------|-------|--------|----------|---------|
| Airspeed (V) | 135 ft/s | Grauer & Boucher (2020) | AIAA 2020-2704 | Section VI.A, p. 7 |
| Angle of attack (α) | 4.5 deg | Grauer & Boucher (2020) | AIAA 2020-2704 | Section VI.A, p. 7 |
| Altitude (h) | 1400 ft | Grauer & Boucher (2020) | AIAA 2020-2704 | Section VI.A, p. 7 |

**Exact quote from source (p. 7):**
> "The nominal flight condition considered was straight and level flight at about 4.5 deg angle of attack, 135 ft/s airspeed, and 1400 ft altitude."

---

## 2. NASA Maneuver Data NOT FOUND

| Information | Status | Notes |
|-------------|--------|-------|
| Flight 41 elevator time history | Not found | Only nominal condition provided |
| Flight 15 elevator time history | Not found | Only nominal condition provided |
| Exact multisine phase angles (φk) | Not found | Phase-optimized but values not published |
| Exact multisine amplitudes (ak) | Not found | Amplitudes scaled for SNR but values not given |
| T-2 elevator actuator limits | Not found | Actuator limits not in public sources |
| T-2 engine throttle limits | Not found | Throttle limits not specified |

**Conclusion:** Public sources provide nominal flight conditions (V, α, h) for Flights 41 and 15, and multisine characteristics (duration, frequency count, range, spacing), but NOT actual control time histories or specific phase/amplitude values.

---

## 3. Scenarios Classified by Provenance

### 3.1 NASA_VERIFIED_REFERENCE_CONDITION

These scenarios have flight conditions from NASA sources, but use project-defined control schedules.

#### `create_flight_41_scenario()`

- **Provenance:** NASA_VERIFIED_REFERENCE_CONDITION
- **Source:** AIAA 2020-2704 (Grauer & Boucher), Section VI.A, Table 2
- **Verified elements:**
  - V = 139.1 ft/s (exact)
  - α = 4.077 deg (exact)
  - h = 1227 ft (exact)
- **Project-defined elements:**
  - Control schedule (zero-input, not actual multisine)
  - Duration (10 s, chosen to match typical multisine duration)
  - dt (0.01 s, numerical integration parameter)

**Important:** This scenario provides the **REFERENCE CONDITION ONLY**. It does NOT reproduce the actual Flight 41 time history.

#### `create_flight_15_scenario()`

- **Provenance:** NASA_VERIFIED_REFERENCE_CONDITION
- **Source:** AIAA 2020-2704 (Grauer & Boucher), Section VI.A, Table 2
- **Verified elements:**
  - V = 136.0 ft/s (exact)
  - α = 3.893 deg (exact)
  - h = 1467 ft (exact)
- **Project-defined elements:**
  - Control schedule (zero-input, not actual multisine)
  - Duration (10 s, chosen to match typical multisine duration)
  - dt (0.01 s, numerical integration parameter)

**Important:** This scenario provides the **REFERENCE CONDITION ONLY**. It does NOT reproduce the actual Flight 15 time history.

### 3.2 NASA_VERIFIED_REFERENCE + PROJECT_DEFINED_WAVEFORM

This scenario has maneuver type and characteristics from NASA sources, but uses a project-defined waveform placeholder.

#### `create_multisine_excitation_scenario()`

- **Provenance:** NASA_VERIFIED_REFERENCE + PROJECT_DEFINED_WAVEFORM
- **Source:** AIAA 2020-2704 (Grauer & Boucher), Section VI.A
- **Verified elements:**
  - Maneuver type: orthogonal phase-optimized multisine
  - Duration: 10 s
  - Number of frequencies: 7
  - Frequency range: 0.2–2.2 Hz
  - Frequency spacing: 0.3 Hz increments
  - Control surfaces: elevator, aileron, rudder (simultaneous)
  - Nominal condition: V=135 ft/s, α=4.5 deg, h=1400 ft
- **Project-defined elements:**
  - Specific phase angles (φk) — not published
  - Specific amplitude scaling (ak) — not published
  - Waveform representation: single-sinusoid placeholder (not 7-sinusoid sum)

**Important:** This scenario does NOT reproduce the actual NASA multisine waveform. The true waveform requires 7 sinusoids with phase-optimized coefficients that are not published. This is a simplified placeholder for testing.

### 3.3 TEST_ONLY

These scenarios use synthetic values for software testing.

#### `create_zero_input_stability_scenario()`

- **Provenance:** TEST_ONLY
- **Source:** PROJECT_INTERNAL
- **Purpose:** Verify simulator equilibrium
- **Rationale:** No NASA document specifies a "zero-input stability test" as a flight maneuver

#### `create_small_disturbance_scenario()`

- **Provenance:** TEST_ONLY
- **Source:** PROJECT_INTERNAL
- **Purpose:** Observe natural short-period response
- **Rationale:** Perturbation values (delta_w, delta_q) are synthetic test parameters

---

## 4. Summary Table

| Scenario | Provenance | Flight Condition Source | Control Schedule Source |
|----------|------------|------------------------|------------------------|
| flight_41 | NASA_VERIFIED_REFERENCE_CONDITION | AIAA 2020-2704, Table 2 | PROJECT_DEFINED (zero-input) |
| flight_15 | NASA_VERIFIED_REFERENCE_CONDITION | AIAA 2020-2704, Table 2 | PROJECT_DEFINED (zero-input) |
| multisine_excitation | NASA_VERIFIED_REFERENCE + PROJECT_DEFINED_WAVEFORM | AIAA 2020-2704, Section VI.A | PROJECT_DEFINED (single-sinusoid placeholder) |
| zero_input_stability | TEST_ONLY | PROJECT_INTERNAL | PROJECT_INTERNAL |
| small_disturbance | TEST_ONLY | PROJECT_INTERNAL | PROJECT_INTERNAL |

---

## 5. Recommendations

1. **Use correct provenance labels:**
   - NASA_VERIFIED_REFERENCE_CONDITION for Flight 41/15 (flight condition only)
   - NASA_VERIFIED_REFERENCE + PROJECT_DEFINED_WAVEFORM for multisine (maneuver type only, placeholder waveform)
   - TEST_ONLY for synthetic test scenarios

2. **Document clearly:** Flight 41/15 scenarios provide reference conditions only, not reproduction of actual flight time histories.

3. **Do not claim full NASA verification:** No scenario is fully NASA_VERIFIED because control time histories are not published.

4. **Future work:** Contact NASA Langley AirSTAR team for access to internal flight test data if exact multisine reproduction is required.

---

## 6. References

1. Grauer, J. A., & Boucher, M. J. (2020). "Aircraft System Identification from Multisine Inputs and Frequency Responses." AIAA 2020-2704. [DOI: 10.2514/6.2020-2704]

2. Morelli, E. A. (2011). "Flight Test Maneuvers for Efficient Aerodynamic Modeling." AIAA 2011-6634. [DOI: 10.2514/6.2011-6634]

3. Grauer, J. A., Morelli, E. A., & Murr, D. G. (2017). "Flight Test Techniques for Quantifying Pitch Rate and Angle of Attack Rate Dependencies." NASA Technical Memorandum TM-2017-219520.

---

**End of Report (FINAL v2)**
