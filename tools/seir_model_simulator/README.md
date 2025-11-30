# SEIR Model Simulator Tool

**File:** `main.py`

**Description:** Runs a configurable Susceptible–Exposed–Infectious–Recovered (SEIR) epidemiological simulation and returns structured analytics for policy scenario exploration within Open WebUI.

**Main Method:** `run_seir_simulation(...) -> str`

## Features

- Classic deterministic SEIR model solved with `scipy.integrate.solve_ivp`
- Explicit exposed compartment with configurable incubation/progression rate `σ`
- Parameter validation for population counts, epidemiological rates, and optional interventions
- Time-series output capped to 1,000 points for UI responsiveness
- Key metrics extraction (peak infections, recovery totals, threshold crossing)
- Optional PNG visualization encoded in base64 for rapid inspection

## Mathematical Model

The SEIR model divides the population into four compartments:

- **S (Susceptible):** Individuals who can contract the disease
- **E (Exposed):** Individuals infected but not yet infectious (latent period)
- **I (Infectious):** Individuals who are infectious and can transmit the disease
- **R (Recovered):**Individuals who have recovered and gained immunity

### Differential Equations

The model is governed by the following system of ordinary differential equations:

```
dS/dt = -β · S · I / N
dE/dt = β · S · I / N - σ · E
dI/dt = σ · E - γ · I
dR/dt = γ · I
```

Where:
- **β** (beta): transmission rate (contacts per day × probability of transmission)
- **σ** (sigma): progression rate from exposed to infectious = 1 / (latent period in days)
- **γ** (gamma): recovery rate = 1 / (infectious period in days)
- **N**: total population (constant: S + E + I + R = N)

### Key Epidemiological Metrics

- **Basic Reproduction Number (R₀):** Approximately β / γ, represents the average number of secondary infections from one infectious individual in a fully susceptible population
- **Latent Period:** 1 / σ (default: 5.2 days, typical for many respiratory diseases)
- **Infectious Period:** 1 / γ
- **Attack Rate:** Final proportion of population that becomes infected over the entire epidemic

### Numerical Integration

The model uses SciPy's `solve_ivp` with the **RK45 (Runge-Kutta-Fehlberg)** method:
- Adaptive step sizing for computational efficiency
- 4th/5th order accuracy
- Relative tolerance: 1e-6, Absolute tolerance: 1e-9
- Numerical stability for epidemic dynamics

## Model Assumptions & Limitations

This implementation assumes:

- **Well-mixed population:** All individuals have equal contact probability (no spatial structure or network effects)
- **Permanent immunity:** Recovered individuals cannot be reinfected (R → S transition = 0)
- **Constant population:** No births, deaths, or migration (N remains constant)
- **Deterministic dynamics:** No random fluctuations (stochastic effects not modeled)
- **Homogeneous mixing:** All individuals have the same transmission and recovery rates

These assumptions make the model suitable for:
- Policy scenario exploration and comparative analysis
- Understanding general epidemic dynamics and intervention effects
- Educational purposes and conceptual modeling

For more realistic scenarios, consider extensions like age-structured models, spatial models, or stochastic simulations.


## Parameters

| Parameter | Description |
|-----------|-------------|
| `total_population` (int, required) | Total population size `N`. Must exceed `E₀ + I₀ + R₀`. |
| `initial_infected` (int, required) | Initial number of infectious individuals `I₀`. |
| `initial_exposed` (int, default `0`) | Initial exposed individuals `E₀` (infected but not yet infectious). |
| `initial_recovered` (int, default `0`) | Initial recovered/immune individuals `R₀`. |
| `transmission_rate` (float, required) | Transmission rate β controlling infection spread. |
| `incubation_rate` (float, default `1/5.2`) | Rate σ moving individuals from exposed to infectious (inverse latent period). |
| `recovery_rate` (float, required) | Recovery rate γ (inverse infectious period). |
| `simulation_days` (int, default `160`) | Number of simulated days. |
| `time_step` (float, default `1.0`) | Desired resolution (days). Automatically coarsened if it exceeds 1,000 points. |
| `intervention_day` (int, optional) | Day when β is multiplied by `intervention_effect`. `None` disables intervention. |
| `intervention_effect` (float, default `1.0`) | Multiplier applied to β after `intervention_day` (e.g., `0.6` reduces transmission by 40%). |
| `infection_threshold` (float, default `1.0`) | Threshold used to estimate when infections drop below a critical value. |
| `max_output_points` (int, default `1000`) | Maximum time points in output arrays (range: 10-10,000). Higher values increase detail but may reduce performance. |

## Output Structure

The tool returns a JSON-formatted string containing:

- `status`: `"success"` or `"error"`
- `parameters`: Echoed inputs plus effective time-step applied
- `metrics`: Peak infections, recovered totals, threshold crossing, **R₀ approximation** (β/γ), and **attack rate**
- `validation`: Population conservation check and maximum deviation from total population
- `time_series`: Complete arrays for `time`, `susceptible`, `exposed`, `infected`, `recovered` (ready for external visualization)
- `summary`: Human-readable interpretation of the scenario

Example snippet:

```json
{
  "status": "success",
  "model": "SEIR",
  "metrics": {
    "peak_infected": 12500.0,
    "peak_day": 37.0,
    "threshold_crossing_day": 118.0,
    "r0_approximation": 3.0,
    "attack_rate": 0.627
  },
  "validation": {
    "population_conservation": true,
    "max_population_deviation": 0.003
  },
  "summary": "Peak infections reach 12,500 individuals (12.5% of the population) around day 37.0..."
}
```

## Visualizing Results

**Note:** This tool runs in the browser (Pyodide) and does not include plotting capabilities. Instead, it returns complete time-series data in JSON format for external visualization.

### Creating Plots Locally

After running a simulation in Open WebUI, copy the JSON output and visualize it using matplotlib:

```python
import json
import matplotlib.pyplot as plt

# Load results from Open WebUI
data = json.loads('''<paste JSON output here>''')
ts = data['time_series']

# Create SEIR plot
plt.figure(figsize=(10, 6))
plt.plot(ts['time'], ts['susceptible'], label='Susceptible', color='#1f77b4')
plt.plot(ts['time'], ts['exposed'], label='Exposed', color='#ff7f0e')
plt.plot(ts['time'], ts['infected'], label='Infected', color='#d62728')
plt.plot(ts['time'], ts['recovered'], label='Recovered', color='#2ca02c')

# Mark peak
peak_day = data['metrics']['peak_day']
peak_infected = data['metrics']['peak_infected']
plt.axvline(peak_day, color='red', linestyle='--', alpha=0.5, 
            label=f'Peak (day {peak_day:.0f})')
plt.scatter([peak_day], [peak_infected], color='red', s=100, zorder=5)

plt.xlabel('Days')
plt.ylabel('Population')
plt.title(f"SEIR Simulation (R₀ = {data['metrics']['r0_approximation']:.2f})")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('seir_results.png', dpi=150)
plt.show()
```

### Why External Visualization?

- **Browser Compatibility:** Pyodide (browser Python) has limited support for visualization libraries
- **Flexibility:** Use any plotting library (matplotlib, plotly, seaborn, R, Excel, etc.)
- **Performance:** Lighter tool execution in the browser
- **Customization:** Full control over plot styling and multi-panel layouts

## Usage

1. Import `tools/seir_model_simulator/seir_model_sim.py` as a tool inside Open WebUI.
2. Invoke via chat, e.g.:
   ```
   Run a SEIR simulation with N=1000000, E0=25, I0=10, beta=0.3, sigma=0.2, gamma=0.1 for 200 days and show the plot.
   ```
3. The agent calls `run_seir_simulation`, returning JSON data and optionally a visualization.

## Requirements

- Python 3.11+
- `numpy`
- `scipy`
- `matplotlib` (only needed if `generate_plot` is enabled)

Install dependencies for this tool:

```bash
pip install -r tools/seir_model_simulator/requirements.txt
```

## Notes

- Input validation ensures non-negative compartments and `N > E₀ + I₀ + R₀`.
- Time grids longer than 1,000 points are compressed to maintain responsive UIs.
- All compartment arrays are clipped to `[0, N]` to limit numerical drift.

For architecture guidance, consult `docs/how_to_create_tool.md`.
