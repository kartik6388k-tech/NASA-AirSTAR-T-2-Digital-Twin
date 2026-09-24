# NASA AirSTAR T-2 Digital Twin

> A rigorous reduced-order digital twin of NASA's 5.5% Generic Transport Model (GTM) T-2 research aircraft. Simulates longitudinal short-period flight dynamics, discrete vertical gusts, synthetic sensor noise, linear Kalman state estimation, rule-based health diagnostics, and real-time browser playback.

<!-- HERO PREVIEW SECTION: See the "GitHub Video Fix" section below to hook up your own looping recording -->
<div align="center">
  <img src="records/flight_41_gust_response.gif" alt="NASA AirSTAR T-2 Digital Twin Dashboard Playback" width="100%" onerror="this.onerror=null; this.src='https://raw.githubusercontent.com/kartik6388k-tech/NASA-AirSTAR-T-2-Digital-Twin/main/records/preview_fallback.png';" />
  <p><em>Real-time Three.js attitude visualization, Chart.js strip charts, and telemetry tracking during a 1-cosine gust encounter.</em></p>
</div>

---

## What This Project Actually Is

Physics first. No hand-waving.

This codebase is an aerospace simulation and telemetry replay environment built around NASA Langley's **Airborne Subscale Transport Aircraft Research (AirSTAR)** program. AirSTAR flew a 5.5% dynamically scaled jet transport known as the **Generic Transport Model (GTM)**, specifically in the twin-turbine **T-2** airframe configuration. NASA used this subscale drone to fly radical, high-risk maneuvers—deep stalls, upset recoveries, extreme turbulence encounters—that would kill human test pilots in a full-sized airliner.

This repository doesn't pretend to be a bloated 6-DOF game simulator. It models something much more focused: an isolated **longitudinal short-period perturbation digital twin**. 

When a jet punches through a sudden updraft or takes a sharp jab of elevator trim, it bobs and pitches rapidly long before its forward speed has time to change. That snappy pitching-and-heaving motion is the short-period mode. We capture it using a two-dimensional state vector:

$$x = \begin{bmatrix} \delta w \\ \delta q \end{bmatrix}$$

* $\delta w$ is the vertical perturbation velocity along the aircraft body Z-axis (positive downwards, in meters per second).
* $\delta q$ is the pitch rate perturbation (pitching nose-up, in radians per second).

Around this core dynamic model, the codebase wraps a complete digital twin stack: atmospheric modeling, dimensional derivative conversion, Runge-Kutta 4th order (RK4) numerical integration, synthetic sensor corruption, a discrete linear Kalman filter, automated health monitoring, and a local web dashboard powered by Three.js and Chart.js.

---

## Architectural Pipeline

Here is how data flows through the system, from the configuration file down to the browser pixels:

```
[ config.yaml ] ──────────────────────────────────────────────┐
       │                                                      │
       ▼                                                      ▼
[ models/aircraft.py ] & [ models/atmosphere.py ]    [ models/propulsion.py ]
       │                                                      │
       ▼                                                      │
[ models/aerodynamics.py ]                                    │
       │ (nondimensional CL, Cm derivatives)                  │
       ▼                                                      │
[ models/conversion.py ]                                      │
       │ (synthesized dimensional Zw, Mw, Zq, Mq derivatives) │
       ▼                                                      │
[ models/flight_dynamics.py ] ◄── [ models/turbulence.py ]     │
       │ (state derivatives dx/dt)        (1-cosine gust)     │
       ▼                                                      │
[ simulation/simulator.py ] ◄─────────────────────────────────┘
       │ (RK4 integration loop + diagnostic thrust/forces)
       ▼
 [ Telemetry CSV History ] (data/processed/)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[ sensors/sensor_model.py ]              [ dashboard/app.py ]
       │ (adds bias + Gaussian noise)            │ (HTTP & JSON API)
       ▼                                         ▼
[ digital_twin/estimator.py ]            [ dashboard/dashboard.html ]
       │ (2-state Kalman filter x_hat)          (3D plane + live strip charts)
       ▼
[ digital_twin/health_monitor.py ]
       │ (residual & covariance anomaly flags)
       ▼
[ digital_twin/integration.py ]
  (end-to-end twin pipeline)
```

---

## Deep-Dive Codebase Analysis

Every Python module in this repository has a strict separation of concerns. Physics equations don't sneak into the scenario definitions, sensor noise doesn't corrupt the integrator state, and the web server doesn't calculate flight derivatives. 

Here is what each file does:

### 1. Root & Configuration Layer
* **`config.yaml`**: The single source of truth for the entire system. Holds the active physical geometry and nominal mass properties of the T-2 GTM (from NASA/TM-2017-219795 Table 1), the twin JetCat P70 turbine ratings, and the identified stability derivatives and 1-sigma parameter uncertainties from Flight 41 and Flight 15 (sourced from AIAA 2015-2704). It enforces strict provenance tagging so speculative assumptions are never mixed up with NASA-verified figures.
* **`main.py`**: The command-line orchestrator. Reads command-line arguments, parses `config.yaml` exactly once, selects the requested scenario from the scenario registry, spins up the aircraft models, initializes the RK4 integrator, executes the simulation run, and writes the telemetry output to `data/processed/`.
* **`requirements.txt`**: Declares minimal external runtime dependencies (`PyYAML` for configuration parsing and `pytest` for the automated test harness). The entire physics engine, numerical integrator, state estimator, diagnostic engine, and web server run purely on standard library Python.
* **`RESEARCH_REPORT.md`**: Detailed research provenance log tracing every single aerodynamic number, flight condition, and multisine frequency back to its original AIAA conference paper or NASA Technical Memorandum.
* **`conftest.py`**: Pytest configuration file ensuring clean module resolution across root and subpackages during test execution.

### 2. Core Physics (`models/`)
* **`models/aircraft.py`**: Pure data schema and configuration container. Defines dataclasses for aircraft identity, geometry (wingspan $b = 2.088\text{ m}$, reference area $S = 0.548\text{ m}^2$, chord $\bar{c} = 0.280\text{ m}$), nominal mass ($23.93\text{ kg}$), and inertia tensor ($I_{xx}, I_{yy}, I_{zz}, I_{xz}$). It contains zero equations of motion and zero aerodynamic math; its job is to validate physical bounds and reject missing parameters (`NOT_FOUND_DO_NOT_ASSUME`).
* **`models/atmosphere.py`**: 1976 U.S. Standard Atmosphere (ISA) implementation covering the troposphere ($0$ to $11,000\text{ m}$) and lower stratosphere ($11,000$ to $20,000\text{ m}$). Computes air temperature $T$, static pressure $p$, air density $\rho$, and speed of sound $a$ as pure mathematical functions of geopotential altitude.
* **`models/aerodynamics.py`**: Implements NASA's identified longitudinal aerodynamic perturbation derivatives ($C_{L_\alpha}, C_{L_q}, C_{L_{\delta e}}, C_{m_\alpha}, C_{m_q}, C_{m_{\delta e}}$) estimated from real flight test data (Flight 41 in light turbulence, Flight 15 in severe turbulence). Calculates perturbation lift $\Delta L$ and pitching moment $\Delta M$. Preserves 1-sigma parameter uncertainties from system identification.
* **`models/conversion.py`**: Bridges non-dimensional aerodynamic coefficients to dimensional stability and control derivatives ($Z_w, Z_q, Z_{\delta e}, M_w, M_q, M_{\delta e}$). Formulates explicit, machine-readable assumptions: small-angle approximation ($\Delta\alpha \approx \Delta w / U_e$), $Z \approx -L$ (neglecting profile drag coupling), constant forward speed $U_e$, and ISA air density. Tags every resulting matrix as `PROJECT_DERIVED`.
* **`models/flight_dynamics.py`**: Solves the short-period equations of motion:
  $$M' \dot{x} = A' x + B' \delta e$$
  Given mass, pitch inertia $I_{yy}$, current perturbation state $[\delta w, \delta q]^T$, elevator deflection $\delta e$, and vertical gust $w_{gust}$, it outputs instantaneous derivatives $[\dot{w}, \dot{q}]^T$ along with perturbation vertical specific force $a_z$.
* **`models/propulsion.py`**: Models the twin JetCat P70 micro-turbines. NASA documented two separate propulsion architectures (an RPM-driven system-identification model and a pilot-throttle first-order lag model). Because explicit numerical coefficients for the internal engine maps were not published by NASA, this module logs engine status cleanly as `MODEL_NOT_IDENTIFIED` rather than fabricating fake engine performance curves.
* **`models/turbulence.py`**: Deterministic longitudinal 1-cosine vertical gust model:
  $$w_{gust}(t) = \frac{A}{2} \left[1 - \cos\left(\frac{2\pi (t - t_0)}{\tau}\right)\right]$$
  Modifies aerodynamic relative velocity $\delta w_{aero} = \delta w - w_{gust}$. This is a discrete gust disturbance, not stochastic continuous turbulence.

### 3. Simulation Engine (`simulation/`)
* **`simulation/simulator.py`**: The time-marching simulation core. Employs a 4th-order Runge-Kutta (RK4) numerical integrator. Steps state derivatives forward, evaluates diagnostic forces and thrust, monitors for numerical divergence/NaNs, gracefully handles timestep remainders so runs terminate on exact second boundaries, and exports full telemetry tables (15 distinct channels) to CSV and metadata JSON.
* **`simulation/scenarios.py`**: Declarative scenario registry. Enforces a strict two-tier provenance architecture: flight condition provenance (e.g. NASA-verified reference conditions) is tracked independently from control schedule provenance (e.g. project-defined waveforms). Registers six standard flight cases:
  1. `zero_input_stability`: Verifies equilibrium resting state.
  2. `small_disturbance_free_response`: Unforced natural short-period damping.
  3. `flight_41_reference_condition_run`: Baseline flight condition at $V = 139.1\text{ ft/s}$.
  4. `flight_15_reference_condition_run`: Baseline flight condition at $V = 136.0\text{ ft/s}$.
  5. `flight_41_gust_response_run`: Dynamic gust penetration scenario.
  6. `multisine_excitation_concept`: Design-only 7-frequency system identification waveform.

### 4. Synthetic Sensors & Digital Twin Layer (`sensors/` & `digital_twin/`)
* **`sensors/sensor_model.py`**: Corrupts true simulation states with constant instrument biases and zero-mean Gaussian white noise. Emulates real onboard avionics (rate gyros measuring pitch rate $q$, alpha vanes sensing vertical velocity $w$, and body-axis accelerometers measuring $a_z$).
* **`digital_twin/estimator.py`**: Two-state discrete-time linear Kalman filter. Reconstructs estimated states $\hat{x} = [\delta\hat{w}, \delta\hat{q}]^T$ from noisy sensor feeds. Updates process covariance $P$, tracks innovation measurement residuals, and documents clearly that it is a linear Kalman estimator, not NASA's filter-error parameter-estimation method.
* **`digital_twin/health_monitor.py`**: Rule-based diagnostic health monitor. Checks estimated states, measurement residuals, covariance bounds, and propulsion flags against defined safety thresholds. Assigns operational severity states (`HEALTHY`, `WARNING`, `CRITICAL`) to detect anomalies without altering control loops.
* **`digital_twin/integration.py`**: Library-level orchestrator connecting truth telemetry $\to$ sensor noise injection $\to$ Kalman filter reconstruction $\to$ diagnostic health evaluation.

### 5. Cockpit Dashboard (`dashboard/`)
* **`dashboard/app.py`**: Lightweight, zero-dependency Python HTTP server built on `http.server`. Serves the front-end dashboard and provides REST endpoints (`/api/telemetry`, `/api/runs`, `/api/run/<name>`, `/api/playback`) protected by thread locks for smooth 50ms browser streaming.
* **`dashboard/dashboard.html`**: Self-contained single-page flight dashboard. Combines Three.js for real-time 3D aircraft attitude animation, Chart.js for live rolling strip charts ($\delta w, \delta q, \delta e, a_z$, gust velocity), and SVG gauge dials for system health.

### 6. Validation & Testing (`scripts/` & `tests/`)
* **`scripts/validate_short_period.py`**: Mathematical verification script. Extracts the short-period system matrix $A'$, calculates characteristic eigenvalues, and computes the natural frequency $\omega_n$ and damping ratio $\zeta$.
* **`tests/`**: Suite of 400+ unit and integration tests verifying physics invariants, ISA tables, coordinate systems, dimensional conversions, Kalman convergence, and HTTP API contracts.

---

## Non-Technical Setup Guide (Step-by-Step)

You do not need a degree in aerospace engineering or computer science to run this simulation. If you can open a terminal and copy-paste a command, you can run this digital twin on your computer.

### Step 1: Install Python
Make sure you have Python installed (version 3.9, 3.10, 3.11, or 3.12).
* Open your terminal (on Windows: press `Win + R`, type `powershell`, and hit Enter; on macOS: press `Cmd + Space`, type `terminal`, and hit Enter).
* Check your version by typing:
  ```bash
  python --version
  ```
  If Python isn't installed, download it from [python.org](https://www.python.org/downloads/) (make sure to check the box that says **"Add Python to PATH"** on Windows).

### Step 2: Download the Code
Clone this repository with Git, or download and extract the ZIP file:
```bash
git clone https://github.com/kartik6388k-tech/NASA-AirSTAR-T-2-Digital-Twin.git
cd NASA-AirSTAR-T-2-Digital-Twin
```

### Step 3: Create a Virtual Environment
A virtual environment is simply an isolated sandbox on your computer so project packages don't interfere with anything else.

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell shows a script execution error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once, then try again).*

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
Install the required packages (takes less than 10 seconds):
```bash
pip install -r requirements.txt
```

### Step 5: Run Your First Flight Simulation
Run a 10-second simulation where the aircraft encounters a vertical wind gust:
```bash
python main.py --scenario flight_41_gust_response_run
```
You will see output detailing the run parameters, the aircraft configuration, and the final state summary. The complete second-by-second flight log is automatically generated and saved as a CSV file in `data/processed/flight_41_gust_response_run.csv`.

Want to see other scenarios? Run:
```bash
# Natural settling response after an initial pitch disturbance:
python main.py --scenario small_disturbance_free_response

# Standard baseline flight condition:
python main.py --scenario flight_41_reference_condition_run
```

### Step 6: Launch the 3D Cockpit Dashboard
To watch the aircraft pitch and heave in real time:
```bash
python dashboard/app.py
```
Open your web browser (Chrome, Edge, Firefox, or Safari) and go to:
```
http://127.0.0.1:8000
```
Use the dropdown in the upper panel to select your simulation run (`flight_41_gust_response_run`), click **Play**, and watch the 3D aircraft model respond directly to the flight telemetry!

---

## Fixing the Broken Repository Preview Video

### Why the Preview Video Breaks on GitHub
If you tried viewing or embedding demo videos on GitHub, you probably noticed a broken image box or an annoying download link. This happens for three specific reasons:
1. **The Missing File Bug**: The old README referenced `records/instability_response_run.gif`, but the `records/` folder only contains raw `.mp4` recordings.
2. **Markdown Doesn't Autoplay MP4s**: If you write `![Demo](records/demo_project.mp4)` in GitHub markdown, GitHub's renderer fails. The markdown `![]()` syntax only supports static images and animated GIFs—it cannot parse or play local MP4 files.
3. **Repository File Viewer Trap**: Clicking a plain link like `[Watch Video](records/demo.mp4)` takes visitors away from your README into GitHub's file browser. That ruins your project's first impression.

---

### The Permanent Fix: Two Working Solutions

Choose whichever option fits your workflow best:

#### Option A: Convert MP4 to an Optimized, Looping GIF (Recommended)
An animated GIF displays directly on your GitHub landing page, loops forever, and requires zero user clicks.

Because raw screen recordings make massive, laggy GIFs, you need to generate a custom color palette so the file size stays small (under 10 MB) while keeping text crisp.

1. **Install FFmpeg** (if you don't already have it):
   * Windows (via Winget): `winget install Gyan.FFmpeg`
   * macOS (via Homebrew): `brew install ffmpeg`
   * Linux: `sudo apt install ffmpeg`

2. **Run this single command** inside your project directory to convert `records/demo_project.mp4` into an optimized, buttery GIF:
   ```bash
   ffmpeg -i records/demo_project.mp4 -vf "fps=15,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" -loop 0 records/demo_project.gif
   ```
   * What this does:
     * `fps=15`: Cuts unnecessary frames to slash file size by 70% while staying completely smooth.
     * `scale=800:-1`: Scales width to 800px (crisp on laptops and phones) while preserving aspect ratio.
     * `palettegen` & `paletteuse`: Generates an adaptive 128-color palette so text and charts stay razor-sharp.
     * `-loop 0`: Tells the GIF to loop infinitely.

3. **Embed it in your `README.md`**:
   ```markdown
   ![NASA AirSTAR T-2 Digital Twin Demo](records/demo_project.gif)
   ```
   Commit and push `records/demo_project.gif` and your updated `README.md`. It will render and loop right on GitHub.

---

#### Option B: GitHub Video CDN Embed (For True 1080p Video Playback)
GitHub allows native HTML5 video streaming with autoplay and looping, but only when the video is hosted on GitHub's asset CDN rather than committed as a raw repo file.

1. Open any **Issue** or **Pull Request** in your GitHub repository (or create a draft release).
2. Drag and drop `records/demo_project.mp4` directly into the issue description box.
3. GitHub will upload the file and give you a link that looks like this:
   ```text
   https://github.com/user-attachments/assets/12345678-abcd-ef01-2345-6789abcdef01
   ```
4. Copy that URL and paste it into your `README.md` using this exact HTML5 snippet:
   ```html
   <video src="https://github.com/user-attachments/assets/YOUR-COPIED-ID.mp4" autoplay loop muted playsinline width="100%">
   </video>
   ```
   > **Note:** Browsers will block autoplay unless `muted` and `playsinline` are present. Do not remove those attributes.

---

## Demonstration Video Guide

The `records/` directory contains screen captures demonstrating different aspects of the digital twin:

| File | What It Demonstrates | Why It Matters |
|---|---|---|
| `demo_project.mp4` | Complete digital twin walkthrough | Best starting point; shows scenario selection, command execution, and live dashboard playback. |
| `flight_41_gust_response.mp4` | Aircraft hitting a 1-cosine vertical gust | Demonstrates the short-period heave and pitch response when entering sudden vertical turbulence. |
| `small_disturbance_free_response.mp4` | Unforced disturbance recovery | Shows the natural aerodynamic damping restoring equilibrium with zero pilot elevator inputs. |
| `t2_doublet_shortperiod_response.mp4` | Elevator doublet excitation | Shows the aircraft's pitch reaction to a sharp forward-and-back stick input. |
| `instability_response.mp4` | Oscillatory dynamic response | Illustrates boundary behavior and how the health monitor flags abnormal oscillations. |
| `enviroment_check_movement.mp4` | Three.js cockpit telemetry binding | Validates coordinate transformations and 3D visual rotation cues against real-time data frames. |

---

## Verifying the Test Suite

Run the full automated test suite using `pytest`:
```bash
python -m pytest tests/ -v
```
The test suite validates:
* ISA atmospheric state outputs against standard atmosphere lookup tables.
* Aerodynamic coefficient derivation against NASA reference records.
* Small-angle conversion assumptions and dimensional derivative matrices.
* RK4 numerical stability, energy conservation, and step-remainder mechanics.
* Kalman filter covariance contraction and innovation residual bounds.
* REST API endpoints, JSON responses, and HTTP playback state machines.

---

## Scientific References & Literature

The aerodynamic derivatives, vehicle geometry, and experimental context in this digital twin are grounded in published NASA Langley research:

1. **Grauer, J. A. & Morelli, E. A. (2015).** *A New Formulation of the Filter-Error Method for Aerodynamic Parameter Estimation in Turbulence.* AIAA Atmospheric Flight Mechanics Conference, AIAA Paper 2015-2704. [NASA NTRS 20160006007](https://ntrs.nasa.gov/citations/20160006007).
2. **Grauer, J. A. & Boucher, M. J. (2020).** *Aircraft System Identification from Multisine Inputs and Frequency Responses.* AIAA Scitech 2020 Forum, AIAA Paper 2020-2704.
3. **Grauer, J. A. (2017).** *Position Corrections for Airspeed and Flow Angle Measurements on Fixed-Wing Aircraft.* NASA Technical Memorandum, NASA/TM-2017-219795.
4. **Cox, D. E., Cunningham, K., & Brian, G. (2010).** *AirSTAR: Simulation and Reduced-Scale Flight Test Facility.* NASA Langley Research Center / DSTO.
5. **Tang, L. et al. (2009).** *Methodologies for Adaptive Flight Envelope Estimation and Protection.* Impact Technologies & NASA Langley Research Center.

---

## License & Attribution

This project is developed for engineering research and academic exploration. All referenced flight data and geometry belong to NASA's public technical reports. Project-derived code, dimensional conversion layers, and telemetry dashboard are released under the MIT License.
