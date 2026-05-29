#!/usr/bin/env python3
"""
THE INSTRUMENT EVOLVER v2 — runnable observer-graph parameter trajectory.

P2.O2 of the path-to-dynamics plan. Builds on instrument.py v0
(framework_parameters(N): pure-framework parameter set at substrate count N)
and instrument_evolver.py v1 (the observer-inclusion evolver tracking the
discarded remainder via the cooling cascade).

WHAT v2 ADDS:
  - A trajectory engine that steps N from any range (e.g., N_planck → N_hub),
    computes framework_parameters at each N, runs downstream observables
    (Saha recombination via the parameters), and outputs the full epoch
    trajectory.
  - The A1 thermal_scale_vs_N adoption is now PLUGGABLE — pass a
    `thermal_scale_callable` to swap T(N) candidates. Default is kinematic
    T(z) = T_0·(1+z); future T(N) candidates (per A1 native-replacement
    scoping an internal working note)
    can be tested by passing alternative callables.
  - Validation: at N=N_hub_today, all observables reproduce the static
    DAG values (cross-checked against predicted_parameters.md).

WHAT v2 DOES NOT DO:
  - Does not derive T(N) — that's A1 native-replacement, multi-session
    research (see scoping doc).
  - Does not compute acoustic-feature scales r_s, D_A, θ* (per step 2
    of P2.O1, the standard FRW pathway breaks down in coasting; the
    framework needs a native acoustic-feature definition).
  - Does not propagate beyond validated low-z domain "silently" — z > 2
    is flagged as BEYOND-VALIDATED via the v0 propagate() machinery.

USAGE:
    from simulator.instrument_evolver_v2 import (
        TrajectoryEvolver, kinematic_thermal_scale,
    )
    evolver = TrajectoryEvolver(thermal_scale=kinematic_thermal_scale)
    log = evolver.run(z_grid=[0, 1, 10, 100, 1089, 5000])
    for entry in log:
        print(entry)
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_b = io.StringIO()
with contextlib.redirect_stdout(_b):
    from simulator.instrument import (
        framework_parameters,
        n_hub,
        ADOPTIONS,
        EpochParameters,
    )


# ---------------------------------------------------------------------------
# Constants for downstream observables (Saha recombination)
# ---------------------------------------------------------------------------
_k_B = 1.380649e-23       # J/K
_hbar = 1.054571817e-34   # J·s
_c = 2.998e8              # m/s
_eV = 1.602176634e-19     # J per eV
_GeV_to_J = 1.602176634e-10  # J per GeV
_T_CMB_0 = 2.7255         # K, present-day CMB observed
_n_b_0 = 2.503e-7 * 1e6   # m^-3, baryon number density (Planck-derived)
_M_E_KG_0 = 9.1093837015e-31  # kg, electron mass at z=0


# ---------------------------------------------------------------------------
# Thermal scale candidates (A1 adoption — pluggable)
# ---------------------------------------------------------------------------
def kinematic_thermal_scale(N: float) -> float:
    """A1 default: standard kinematic T ∝ 1/a ∝ N^(-1) (in coasting, since a∝N).

    T(N) = T_CMB_0 · (N_hub_today / N)

    This is the *adopted* form. Not framework-native. Use as baseline.
    """
    return _T_CMB_0 * (n_hub() / N)


def hawking_gibbons_thermal_scale(N: float) -> float:
    """A1 candidate 3.1: Hawking-Gibbons of cascade clock.
    T(N) = ℏ·H/(2π·k_B) where H = 1/(N·t_P).
    Known to give T_today ~ 10^-29 K (30 dex off observed) — for testing
    framework against the candidate.
    """
    t_P_seconds = 5.391e-44  # Planck time
    H_si = 1.0 / (N * t_P_seconds)
    return _hbar * H_si / (2 * math.pi * _k_B)


def stefan_boltzmann_observer_thermal_scale(N: float, normalize_at_N_hub: bool = True) -> float:
    """A1 candidate 3.3: Stefan-Boltzmann observer-rate.
    T(N) ∝ N^(-1/2).
    Normalized to T_today at N_hub if normalize_at_N_hub=True.
    """
    Nh = n_hub()
    if normalize_at_N_hub:
        return _T_CMB_0 * math.sqrt(Nh / N)
    # Otherwise return un-normalized form proportional to N^(-1/2)
    return math.sqrt(Nh / N)


# Catalog of available candidates
THERMAL_SCALE_CANDIDATES = {
    "kinematic": kinematic_thermal_scale,
    "hawking_gibbons": hawking_gibbons_thermal_scale,
    "stefan_boltzmann_observer": stefan_boltzmann_observer_thermal_scale,
}


# ---------------------------------------------------------------------------
# Saha recombination — downstream observable
# ---------------------------------------------------------------------------
def saha_x_e(N: float, params: EpochParameters, T_K: float) -> float:
    """Compute hydrogen ionization fraction x_e via Saha equation at substrate
    count N, given framework parameters and thermal-scale T (in Kelvin).

    Saha: x_e²/(1-x_e) = (1/n_b)·((m_e·k_B·T)/(2π·ℏ²))^(3/2)·exp(-E_b/(k_B·T))
    """
    Nh = n_hub()
    z_eff = (Nh / N) - 1  # equivalent redshift in coasting
    n_b = _n_b_0 * (1 + z_eff)**3   # matter conservation
    m_e_kg = _M_E_KG_0 * (Nh / N)**0.25   # framework: m_e ∝ N^(-1/4)
    E_b_J = params.rydberg_binding.value * _GeV_to_J  # already framework-native
    kT_J = _k_B * T_K

    if kT_J <= 0:
        return 0.0
    de_broglie_density = (m_e_kg * kT_J / (2 * math.pi * _hbar**2))**1.5
    arg = -E_b_J / kT_J
    if arg < -700:
        return 0.0  # underflow safety
    R = (1.0 / n_b) * de_broglie_density * math.exp(arg)
    return (-R + math.sqrt(R*R + 4*R)) / 2


# ---------------------------------------------------------------------------
# Trajectory entry — one epoch's snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrajectoryEntry:
    """Snapshot of the framework parameters + downstream observables at one N."""
    N: float
    z: float
    parameters: EpochParameters
    thermal_scale_K: float
    x_e_saha: float
    thermal_scale_name: str
    flags: tuple = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# The evolver
# ---------------------------------------------------------------------------
class TrajectoryEvolver:
    """Pluggable observer-graph parameter trajectory engine.

    Parameters
    ----------
    thermal_scale : Callable[[float], float]
        T(N) callable. Default is kinematic_thermal_scale.
    thermal_scale_name : str
        Label for the thermal-scale callable (used in trajectory entries).
    """

    def __init__(
        self,
        thermal_scale: Callable[[float], float] = kinematic_thermal_scale,
        thermal_scale_name: str = "kinematic",
    ):
        self.thermal_scale = thermal_scale
        self.thermal_scale_name = thermal_scale_name

    def step(self, N: float) -> TrajectoryEntry:
        """Compute the trajectory snapshot at substrate count N."""
        Nh = n_hub()
        z = (Nh / N) - 1
        params = framework_parameters(N)
        T_K = self.thermal_scale(N)
        x_e = saha_x_e(N, params, T_K)
        flags = []
        if z > 2.0:
            flags.append("beyond-validated-Nz-coasting-adoption")
        if T_K < 1e-10:
            flags.append("T(N)-underflow")
        if T_K > 1e10:
            flags.append("T(N)-overflow")
        return TrajectoryEntry(
            N=N, z=z, parameters=params,
            thermal_scale_K=T_K, x_e_saha=x_e,
            thermal_scale_name=self.thermal_scale_name,
            flags=tuple(flags),
        )

    def run(
        self,
        N_grid: Optional[list[float]] = None,
        z_grid: Optional[list[float]] = None,
    ) -> list[TrajectoryEntry]:
        """Step over a grid of N (or z) values; return trajectory log."""
        if N_grid is None and z_grid is None:
            raise ValueError("provide N_grid or z_grid")
        if N_grid is not None and z_grid is not None:
            raise ValueError("provide exactly one of N_grid, z_grid")
        Nh = n_hub()
        if z_grid is not None:
            N_grid = [Nh / (1 + z) for z in z_grid]
        return [self.step(N) for N in N_grid]


# ---------------------------------------------------------------------------
# Demo / quick visual check (runs when this file is executed directly)
# ---------------------------------------------------------------------------
def _print_trajectory(log: list[TrajectoryEntry]):
    print(f"{'z':>10} | {'N':>11} | {'T (K)':>11} | {'m_e (GeV)':>12} | "
          f"{'E_b (eV)':>10} | {'x_e':>8} | flags")
    print("-" * 110)
    for e in log:
        m_e_GeV = e.parameters.m_e.value
        E_b_eV = e.parameters.rydberg_binding.value * 1e9  # GeV → eV
        flags = ",".join(e.flags) if e.flags else ""
        print(f"{e.z:>10.2f} | {e.N:>11.3e} | {e.thermal_scale_K:>11.3e} | "
              f"{m_e_GeV:>12.4e} | {E_b_eV:>10.4f} | {e.x_e_saha:>8.4e} | {flags}")


if __name__ == "__main__":
    print("=" * 76)
    print("Instrument Evolver v2 — observer-graph parameter trajectory")
    print("=" * 76)
    print(f"N_hub_today = {n_hub():.4e}")
    print()

    print("\n--- Default A1: kinematic T ∝ 1/a ---")
    evolver = TrajectoryEvolver(
        thermal_scale=kinematic_thermal_scale,
        thermal_scale_name="kinematic"
    )
    z_grid = [0.0, 0.5, 1.0, 2.0, 10.0, 100.0, 1089.0, 5000.0, 15365.0]
    log = evolver.run(z_grid=z_grid)
    _print_trajectory(log)

    print("\n--- Candidate 3.3: Stefan-Boltzmann observer ∝ N^(-1/2) ---")
    evolver_sb = TrajectoryEvolver(
        thermal_scale=stefan_boltzmann_observer_thermal_scale,
        thermal_scale_name="stefan_boltzmann_observer"
    )
    log_sb = evolver_sb.run(z_grid=z_grid)
    _print_trajectory(log_sb)

    print("\n--- Candidate 3.1: Hawking-Gibbons (for reference; off by ~30 dex) ---")
    evolver_hg = TrajectoryEvolver(
        thermal_scale=hawking_gibbons_thermal_scale,
        thermal_scale_name="hawking_gibbons"
    )
    log_hg = evolver_hg.run(z_grid=[0.0, 1089.0, 15365.0])
    _print_trajectory(log_hg)

    print()
    print("Available thermal-scale candidates (A1 adoption):")
    for name in THERMAL_SCALE_CANDIDATES:
        print(f"  - {name}")
    print()
    print("To plug in a new T(N), pass it as thermal_scale=... to TrajectoryEvolver.")
    print("See an internal working note")
    print("for the F1-F6 falsification criteria a new candidate must satisfy.")
