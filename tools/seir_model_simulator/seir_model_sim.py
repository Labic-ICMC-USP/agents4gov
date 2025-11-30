import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pydantic import Field
from scipy.integrate import solve_ivp


class Tools:
    """
    Agents4Gov tool implementing a deterministic SEIR (Susceptible–Exposed–Infectious–Recovered)
    epidemiological simulator suitable for Open WebUI tool execution.

    The public `run_seir_simulation` method validates inputs, integrates the SEIR system,
    computes interpretable metrics, and returns structured JSON with complete time-series
    data for external visualization.
    """

    def __init__(self, max_time_points: int = 1000) -> None:
        self.max_time_points = max_time_points

    def _validate_parameters(
        self,
        total_population: int,
        initial_exposed: int,
        initial_infected: int,
        initial_recovered: int,
        transmission_rate: float,
        incubation_rate: float,
        recovery_rate: float,
        simulation_days: int,
        time_step: float,
        intervention_day: Optional[int],
        intervention_effect: float,
        infection_threshold: float,
    ) -> Dict[str, Any]:
        errors = []

        if total_population <= 0:
            errors.append("Total population must be greater than zero.")
        if initial_exposed < 0:
            errors.append("Initial exposed count cannot be negative.")
        if initial_infected < 0:
            errors.append("Initial infected count cannot be negative.")
        if initial_recovered < 0:
            errors.append("Initial recovered count cannot be negative.")
        if initial_exposed + initial_infected + initial_recovered > total_population:
            errors.append("Sum of exposed, infected, and recovered cannot exceed total population.")
        if transmission_rate <= 0:
            errors.append("Transmission rate β must be greater than zero.")
        if incubation_rate <= 0:
            errors.append("Incubation rate σ must be greater than zero.")
        if recovery_rate <= 0:
            errors.append("Recovery rate γ must be greater than zero.")
        if simulation_days <= 0:
            errors.append("Simulation period must be greater than zero.")
        if time_step <= 0:
            errors.append("Time step must be greater than zero.")
        if intervention_day is not None:
            if intervention_day < 0:
                errors.append("Intervention day cannot be negative.")
            if intervention_day > simulation_days:
                errors.append("Intervention day cannot exceed the simulation period.")
        if intervention_effect <= 0:
            errors.append("Intervention effect multiplier must be greater than zero.")
        if infection_threshold < 0:
            errors.append("Infection threshold cannot be negative.")

        if errors:
            raise ValueError(" ".join(errors))

        susceptible0 = total_population - initial_exposed - initial_infected - initial_recovered

        return {
            "total_population": float(total_population),
            "initial_conditions": (
                float(susceptible0),
                float(initial_exposed),
                float(initial_infected),
                float(initial_recovered),
            ),
        }

    def _build_time_grid(self, simulation_days: int, requested_step: float) -> Tuple[np.ndarray, float]:
        """
        Construct time evaluation grid with automatic coarsening for performance.

        Args:
            simulation_days: Total simulation period in days
            requested_step: Desired time step in days

        Returns:
            Tuple of (time_points array, effective_step used)
        """
        approx_points = int(np.floor(simulation_days / requested_step)) + 1
        approx_points = max(approx_points, 2)
        steps = min(approx_points, self.max_time_points)

        if steps > 1:
            effective_step = simulation_days / (steps - 1)
        else:
            effective_step = simulation_days

        time_points = np.linspace(0, simulation_days, steps)
        return time_points, effective_step

    def _seir_derivatives(
        self,
        t: float,
        y: np.ndarray,
        population: float,
        beta: float,
        sigma: float,
        gamma: float,
        intervention_day: Optional[int],
        intervention_effect: float,
    ) -> Tuple[float, float, float, float]:
        """
        Compute SEIR model derivatives (right-hand side of ODEs).

        Implements the classical SEIR equations:
            dS/dt = -β·S·I/N
            dE/dt = β·S·I/N - σ·E
            dI/dt = σ·E - γ·I
            dR/dt = γ·I

        Args:
            t: Current time in days
            y: State vector [S, E, I, R]
            population: Total population N (constant)
            beta: Transmission rate (contacts per day × transmission probability)
            sigma: Progression rate from E to I (1/latent period)
            gamma: Recovery rate (1/infectious period)
            intervention_day: Day when intervention starts (None if no intervention)
            intervention_effect: Multiplier for beta after intervention

        Returns:
            Tuple of derivatives [dS/dt, dE/dt, dI/dt, dR/dt]
        """
        susceptible, exposed, infected, recovered = y
        effective_beta = beta

        # Apply intervention if we've reached the intervention day
        if intervention_day is not None and t >= intervention_day:
            effective_beta = beta * intervention_effect

        # Force of infection: λ = β·I/N (probability of contact with infectious individual)
        infection_force = effective_beta * susceptible * infected / population

        # Susceptible: losses to exposure
        d_susceptible = -infection_force
        # Exposed: gains from S, losses to progression to infectious
        d_exposed = infection_force - sigma * exposed
        # Infectious: gains from E, losses to recovery
        d_infected = sigma * exposed - gamma * infected
        # Recovered: gains from I (permanent immunity assumed)
        d_recovered = gamma * infected

        return d_susceptible, d_exposed, d_infected, d_recovered

    def run_seir_simulation(
        self,
        total_population: int = Field(
            ...,
            gt=0,
            description="Total population size N (must exceed the sum of initial exposed, infected, and recovered).",
        ),
        initial_infected: int = Field(
            ...,
            ge=0,
            description="Initial number of infected individuals I₀.",
        ),
        initial_exposed: int = Field(
            default=0,
            ge=0,
            description="Initial number of exposed but not yet infectious individuals E₀.",
        ),
        initial_recovered: int = Field(
            default=0,
            ge=0,
            description="Initial number of recovered/immune individuals R₀.",
        ),
        transmission_rate: float = Field(
            ...,
            gt=0,
            description="Transmission rate β controlling infection spread intensity.",
        ),
        incubation_rate: float = Field(
            default=1 / 5.2,
            gt=0,
            description="Incubation/progression rate σ (1 / latent period).",
        ),
        recovery_rate: float = Field(
            ...,
            gt=0,
            description="Recovery rate γ representing the inverse of the infectious period.",
        ),
        simulation_days: int = Field(
            default=160,
            gt=0,
            description="Duration of the simulation in days.",
        ),
        time_step: float = Field(
            default=1.0,
            gt=0,
            description="Resolution of the time grid in days (automatically adjusted if too fine).",
        ),
        intervention_day: Optional[int] = Field(
            default=None,
            description="Day at which β is multiplied by the intervention effect (None disables intervention).",
        ),
        intervention_effect: float = Field(
            default=1.0,
            gt=0,
            description="Multiplier applied to β after the intervention day (e.g., 0.7 reduces transmission by 30%).",
        ),
        infection_threshold: float = Field(
            default=1.0,
            ge=0,
            description="Threshold (individual count) used to estimate when infections fall below a critical level.",
        ),
        max_output_points: int = Field(
            default=1000,
            ge=10,
            le=10000,
            description="Maximum number of time points in output arrays (higher values increase detail but reduce performance).",
        ),
    ) -> str:
        """
        Run a Susceptible–Exposed–Infectious–Recovered (SEIR) simulation.
        
        Returns a JSON string containing:
        - Complete time-series data for S, E, I, R compartments
        - Key epidemiological metrics (R₀, peak infections, attack rate)
        - Population conservation validation
        - Human-readable summary
        
        Args:
            total_population: Total population size N
            initial_infected: Initial infectious individuals I₀
            initial_exposed: Initial exposed individuals E₀
            initial_recovered: Initial recovered/immune individuals R₀
            transmission_rate: Transmission rate β
            incubation_rate: Incubation rate σ (1/latent period)
            recovery_rate: Recovery rate γ (1/infectious period)
            simulation_days: Duration of simulation in days
            time_step: Desired time step resolution
            intervention_day: Day when intervention starts (None = no intervention)
            intervention_effect: Multiplier for β after intervention
            infection_threshold: Threshold for tracking infection decline
            max_output_points: Maximum time points in output (10-10,000)
            
        Returns:
            JSON string with simulation results and metrics
        """
        
        # ========================================
        # PARAMETER VALIDATION
        # ========================================
        
        try:
            parsed = self._validate_parameters(
                total_population=total_population,
                initial_exposed=initial_exposed,
                initial_infected=initial_infected,
                initial_recovered=initial_recovered,
                transmission_rate=transmission_rate,
                incubation_rate=incubation_rate,
                recovery_rate=recovery_rate,
                simulation_days=simulation_days,
                time_step=time_step,
                intervention_day=intervention_day,
                intervention_effect=intervention_effect,
                infection_threshold=infection_threshold,
            )
        except ValueError as exc:
            error_result = {
                "status": "error",
                "error_type": "validation_error",
                "message": str(exc),
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)
        
        # ========================================
        # TIME GRID CONSTRUCTION
        # ========================================

        # Override max_time_points if specified by user
        original_max = self.max_time_points
        self.max_time_points = max_output_points
        time_points, effective_step = self._build_time_grid(simulation_days, time_step)
        self.max_time_points = original_max  # Restore original
        
        y0 = parsed["initial_conditions"]
        population = parsed["total_population"]
        
        # ========================================
        # NUMERICAL INTEGRATION
        # ========================================

        try:
            solution = solve_ivp(
                fun=lambda t, y: self._seir_derivatives(
                    t,
                    y,
                    population=population,
                    beta=transmission_rate,
                    sigma=incubation_rate,
                    gamma=recovery_rate,
                    intervention_day=intervention_day,
                    intervention_effect=intervention_effect,
                ),
                t_span=(0, simulation_days),
                y0=y0,
                t_eval=time_points,
                method="RK45",
                vectorized=False,
                rtol=1e-6,
                atol=1e-9,
            )
        except Exception as exc:
            error_result = {
                "status": "error",
                "error_type": "integration_error",
                "message": f"Failed to solve SEIR equations: {exc}",
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)

        if not solution.success:
            error_result = {
                "status": "error",
                "error_type": "integration_error",
                "message": solution.message,
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)
        
        # ========================================
        # POST-PROCESSING AND METRICS EXTRACTION
        # ========================================

        susceptible, exposed, infected, recovered = solution.y
        susceptible = np.clip(susceptible, 0, population)
        exposed = np.clip(exposed, 0, population)
        infected = np.clip(infected, 0, population)
        recovered = np.clip(recovered, 0, population)

        # Validate population conservation (S + E + I + R should equal N)
        population_totals = susceptible + exposed + infected + recovered
        max_deviation = float(np.max(np.abs(population_totals - population)))
        conservation_valid = max_deviation < (population * 1e-4)  # 0.01% tolerance

        # Extract key metrics
        peak_index = int(np.argmax(infected))
        peak_infected = float(infected[peak_index])
        peak_day = float(time_points[peak_index])
        peak_share = float(peak_infected / population)
        total_recovered = float(recovered[-1])
        recovered_share = float(total_recovered / population)
        
        # Calculate R₀ approximation (basic reproduction number)
        r0_approximation = float(transmission_rate / recovery_rate)

        threshold_day = None
        for t_value, infected_value in zip(time_points, infected):
            if infected_value <= infection_threshold:
                threshold_day = float(t_value)
                break

        summary_parts = [
            f"Peak infections reach {peak_infected:,.0f} individuals ({peak_share * 100:.2f}% of the population) around day {peak_day:.1f}.",
            f"By day {simulation_days}, approximately {total_recovered:,.0f} individuals ({recovered_share * 100:.2f}%) have recovered.",
        ]
        if threshold_day is not None:
            summary_parts.append(
                f"Infections fall below {infection_threshold} individuals near day {threshold_day:.1f}."
            )
        else:
            summary_parts.append(
                f"Infections do not drop below {infection_threshold} individuals within the simulated period."
            )
        
        # ========================================
        # BUILD STRUCTURED RESPONSE
        # ========================================

        result: Dict[str, Any] = {
            "status": "success",
            "model": "SEIR",
            "parameters": {
                "total_population": total_population,
                "initial_susceptible": int(total_population - initial_exposed - initial_infected - initial_recovered),
                "initial_exposed": initial_exposed,
                "initial_infected": initial_infected,
                "initial_recovered": initial_recovered,
                "transmission_rate": transmission_rate,
                "incubation_rate": incubation_rate,
                "recovery_rate": recovery_rate,
                "simulation_days": simulation_days,
                "time_step_requested": time_step,
                "time_step_applied": effective_step,
                "intervention_day": intervention_day,
                "intervention_effect_multiplier": intervention_effect,
                "infection_threshold": infection_threshold,
            },
            "metrics": {
                "peak_infected": peak_infected,
                "peak_day": peak_day,
                "peak_population_share": peak_share,
                "total_recovered": total_recovered,
                "recovered_population_share": recovered_share,
                "threshold_crossing_day": threshold_day,
                "r0_approximation": r0_approximation,
                "attack_rate": recovered_share,
            },
            "validation": {
                "population_conservation": conservation_valid,
                "max_population_deviation": max_deviation,
            },
            "time_series": {
                "time": time_points.tolist(),
                "susceptible": susceptible.tolist(),
                "exposed": exposed.tolist(),
                "infected": infected.tolist(),
                "recovered": recovered.tolist(),
            },
            "summary": " ".join(summary_parts),
        }
        
        # Return as formatted JSON string
        return json.dumps(result, ensure_ascii=False, indent=2)
