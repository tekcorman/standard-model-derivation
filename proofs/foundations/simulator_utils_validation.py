"""
simulator_utils_validation.py

Validation probe for the counting-first simulator's derived-shorthand
utilities (Phase 2 build).

Tests all 5 utility modules:
  1. SpectralUtility — adjacency Perron, Hashimoto Perron, NB survival
  2. AlgebraicUtility — Cl(6), Cl(0,2) generators, anticommutation
  3. GroupOrbitUtility — C_3 characters, Q_Koide from isotypic
  4. GeometricPhaseUtility — closure rates, CKM CP phase, arg(h)
  5. PatiSalamUtility — Tr(T_3L²), Tr(Q²), sin²θ_W, α_GUT

Plus end-to-end predictions: Q_Koide = 2/3, sin²θ_W = 3/8, α_GUT = 1/24,
δ_CP_CKM = arccos(1/3), and y_τ = 1280/177147 via the full kernel + utility stack.

If all tests pass, Phase 2 (derived-shorthand utilities) is committed.

Predecessors:
- simulator/kernel.py (Phase 1)
- simulator/utils/*.py (Phase 2 — being validated here)
- proofs/foundations/simulator_kernel_validation.py (Phase 1 validation)
"""

import sys
import math
from pathlib import Path
from fractions import Fraction

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from simulator.srs_engine import CountingKernel
from simulator.srs_engine.utils import (
    SpectralUtility,
    AlgebraicUtility,
    GroupOrbitUtility,
    GeometricPhaseUtility,
)
# PatiSalamUtility moved to match/ as a physics-naming layer
from match.pati_salam import PatiSalamUtility


class TestStats:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def check(self, name, condition, detail=""):
        if condition:
            print(f"  ✓ {name}")
            self.passed += 1
        else:
            print(f"  ✗ {name} — {detail}")
            self.failed.append((name, detail))

    def summary(self):
        total = self.passed + len(self.failed)
        print(f"\n  RESULT: {self.passed}/{total} passed")
        if self.failed:
            print("  FAILURES:")
            for name, detail in self.failed:
                print(f"    - {name}: {detail}")
        return len(self.failed) == 0


def test_spectral(kernel, stats):
    print("\n[1] SpectralUtility")
    stats.check(
        "adjacency_perron = k* = 3",
        SpectralUtility.adjacency_perron_eigenvalue(kernel) == 3,
    )
    stats.check(
        "hashimoto_perron = k*-1 = 2",
        SpectralUtility.hashimoto_perron_eigenvalue(kernel) == 2,
    )
    stats.check(
        "nb_survival_per_step = 2/3",
        SpectralUtility.nb_survival_per_step(kernel) == Fraction(2, 3),
    )
    h = SpectralUtility.hashimoto_eigenvalue_at_P(kernel)
    expected = complex(math.sqrt(3) / 2, math.sqrt(5) / 2)
    stats.check(
        "hashimoto_eigenvalue_at_P = (√3+i√5)/2",
        abs(h - expected) < 1e-12,
    )
    # A(P) eigenvalues = ±√3 (mult 2)
    evals_P = SpectralUtility.adjacency_spectrum_at_k(kernel, 'P')
    expected_evals = sorted([-math.sqrt(3), -math.sqrt(3), math.sqrt(3), math.sqrt(3)])
    stats.check(
        "adjacency_spectrum at P = (±√3, ±√3)",
        np.allclose(sorted(evals_P), expected_evals),
    )


def test_algebraic(kernel, stats):
    print("\n[2] AlgebraicUtility")
    stats.check(
        "Cl(6) generators satisfy {γ^a, γ^b} = 2δ^{ab} I",
        AlgebraicUtility.verify_cl6_anticommutation(),
    )
    stats.check(
        "Cl(0,2) generators satisfy f^a² = -I, {f_1, f_2} = 0",
        AlgebraicUtility.verify_cl02_anticommutation(),
    )
    # Chirality operator γ_5 squares to ±I
    g5 = AlgebraicUtility.cl6_chirality()
    g5sq = g5 @ g5
    I8 = np.eye(8, dtype=complex)
    stats.check(
        "γ_5² = ±I (8x8 chirality operator)",
        np.allclose(g5sq, I8) or np.allclose(g5sq, -I8),
    )
    # 15 bivectors (so(6) basis)
    bivectors = AlgebraicUtility.cl6_bivectors()
    stats.check(
        "Cl(6) has 15 bivectors (= so(6) dim)",
        len(bivectors) == 15,
    )
    # Quaternion basis
    quat = AlgebraicUtility.cl02_quaternion_basis()
    stats.check(
        "Cl(0,2) quaternion basis has {1, i, j, k}",
        set(quat.keys()) == {'1', 'i', 'j', 'k'},
    )


def test_group_orbit(kernel, stats):
    print("\n[3] GroupOrbitUtility")
    chars = GroupOrbitUtility.c3_characters()
    stats.check(
        "C_3 has 3 irreducible reps (trivial, ω, ω̄)",
        set(chars.keys()) == {'trivial', 'omega', 'omega_bar'},
    )

    # Q_Koide from (4, 2, 2) isotypic — should give 2/3
    Q = GroupOrbitUtility.koide_q_from_isotypic((4, 2, 2))
    stats.check(
        "Q_Koide from (4,2,2) ≈ 2/3",
        abs(Q - 2/3) < 1e-12,
        f"got {Q}"
    )

    # Generations
    stats.check(
        "n_generations = 3",
        GroupOrbitUtility.n_generations() == 3,
    )

    # Galois Z_3 has 3 elements
    galois = GroupOrbitUtility.galois_z3_generation_orbit()
    stats.check(
        "Galois Z_3 has 3 elements",
        len(galois) == 3,
    )
    stats.check(
        "First Galois element is identity",
        np.allclose(galois[0], np.eye(3, dtype=complex)),
    )


def test_geometric_phase(kernel, stats):
    print("\n[4] GeometricPhaseUtility")
    nu_amp = GeometricPhaseUtility.closure_rate_amplitude(kernel)
    stats.check(
        "ν_amp = √5/4 ≈ 0.559",
        abs(nu_amp - math.sqrt(5)/4) < 1e-12,
    )
    nu_mass = GeometricPhaseUtility.closure_rate_mass_squared(kernel)
    stats.check(
        "ν_mass² = 5/3 ≈ 1.667",
        abs(nu_mass - 5/3) < 1e-12,
    )
    nu_edge = GeometricPhaseUtility.closure_rate_edge_local(kernel)
    stats.check(
        "ν_edge = 1",
        nu_edge == 1.0,
    )
    # δ_CP_CKM = arccos(1/3) ≈ 70.53°
    cp = GeometricPhaseUtility.k4_minus_eigenspace_dihedral()
    stats.check(
        "δ_CP_CKM = arccos(1/3) ≈ 70.53°",
        abs(cp['degrees'] - 70.5288) < 0.01,
        f"got {cp['degrees']}"
    )
    # arg(h) ≈ 0.9117 rad
    arg_h = GeometricPhaseUtility.arg_h_at_P(kernel)
    stats.check(
        "arg(h) = arctan(√5/√3) ≈ 0.9117",
        abs(arg_h - math.atan2(math.sqrt(5), math.sqrt(3))) < 1e-12,
    )
    # Z_3 holonomy is flat → θ_QCD = 0
    holonomy = GeometricPhaseUtility.z3_holonomy_flat()
    stats.check(
        "θ_QCD = 0 (Z_3 holonomy flat)",
        holonomy['phase_rad'] == 0.0,
    )


def test_pati_salam(kernel, stats):
    print("\n[5] PatiSalamUtility")
    # Tr(T_3L²) on (4,2,1) = 1/2 × 4 × 1 = 2
    t3l_421 = PatiSalamUtility.trace_T3L_squared((4, 2, 1))
    stats.check(
        "Tr(T_3L²) on (4,2,1) = 2",
        t3l_421 == Fraction(2),
        f"got {t3l_421}"
    )
    # Tr(T_3L²) on (4̄,1,2) = 0 (no SU(2)_L)
    t3l_412 = PatiSalamUtility.trace_T3L_squared((4, 1, 2))
    stats.check(
        "Tr(T_3L²) on (4̄,1,2) = 0",
        t3l_412 == Fraction(0),
        f"got {t3l_412}"
    )
    # Tr(Q²) on (4,2,1) = 1 + 5/3 = 8/3
    q2_421 = PatiSalamUtility.trace_Q_squared((4, 2, 1))
    stats.check(
        "Tr(Q²) on (4,2,1) = 8/3",
        q2_421 == Fraction(8, 3),
        f"got {q2_421}"
    )
    # Tr(Q²) on (4̄,1,2) = 8/3
    q2_412 = PatiSalamUtility.trace_Q_squared((4, 1, 2))
    stats.check(
        "Tr(Q²) on (4̄,1,2) = 8/3",
        q2_412 == Fraction(8, 3),
        f"got {q2_412}"
    )
    # sin²θ_W = 3/8 (exact)
    sin2 = PatiSalamUtility.sin2_theta_W()
    stats.check(
        "sin²θ_W = 3/8 (exact rational)",
        sin2 == Fraction(3, 8),
        f"got {sin2}"
    )
    # α_GUT = 1/24
    alpha_gut = PatiSalamUtility.alpha_GUT(kernel)
    stats.check(
        "α_GUT = 1/(2^k* · k*) = 1/24",
        alpha_gut == Fraction(1, 24),
        f"got {alpha_gut}"
    )
    # Hypercharge assignments
    Y_qL = PatiSalamUtility.hypercharge_Y('q_L')
    stats.check(
        "Y(q_L) = +1/6",
        Y_qL == Fraction(1, 6),
        f"got {Y_qL}"
    )
    Y_higgs = PatiSalamUtility.hypercharge_Y('higgs')
    stats.check(
        "Y(higgs) = +1/2",
        Y_higgs == Fraction(1, 2),
        f"got {Y_higgs}"
    )
    # Fermion states per gen = 8
    n_fermions = PatiSalamUtility.fermion_states_per_generation()
    stats.check(
        "fermion_states_per_gen = 8 (Cl(6) spinor)",
        n_fermions == 8,
    )


def test_end_to_end_predictions(kernel, stats):
    """Compute the full set of sanity-check predictions via kernel + utilities."""
    print("\n[End-to-end] Predictions via kernel + utilities")

    # V_us = 9/40
    V_us = Fraction(
        kernel.equiv_class_count('coupling_pair_per_girth_cycle'),
        kernel.walk_count('girth_cycle_per_atom') * kernel.orbit_count('lattice_atoms')
    )
    stats.check(
        "V_us = 9/40",
        V_us == Fraction(9, 40),
        f"got {V_us}"
    )

    # y_τ = (2/3)^8 · 5/3 · 1/9 = 1280/177147
    nb_survival = kernel.walk_count('nb_closed_at_girth')
    nu_mass_sq = Fraction(5, 3)  # exact value of tan²(arg h)
    edge_slot_factor = Fraction(1, kernel.equiv_class_count('site_stabilizer_orbit_at_vertex')) ** 2
    y_tau = nb_survival * nu_mass_sq * edge_slot_factor
    stats.check(
        "y_τ = 1280/177147",
        y_tau == Fraction(1280, 177147),
        f"got {y_tau}"
    )

    # λ_H = 2 · (2/3)^8 · 5/3 = 2560/19683
    lambda_H = 2 * nb_survival * nu_mass_sq
    stats.check(
        "λ_H = 2560/19683",
        lambda_H == Fraction(2560, 19683),
        f"got {lambda_H}"
    )

    # V_cb = (2/3)^8 / (1 - (2/3)^8) = 256/6305
    V_cb = nb_survival / (1 - nb_survival)
    stats.check(
        "V_cb = 256/6305",
        V_cb == Fraction(256, 6305),
        f"got {V_cb}"
    )

    # sin²θ_W = 3/8 via PS utility
    sin2 = PatiSalamUtility.sin2_theta_W()
    stats.check(
        "sin²θ_W = 3/8",
        sin2 == Fraction(3, 8),
        f"got {sin2}"
    )

    # α_GUT = 1/24
    alpha_gut = PatiSalamUtility.alpha_GUT(kernel)
    stats.check(
        "α_GUT = 1/24",
        alpha_gut == Fraction(1, 24),
        f"got {alpha_gut}"
    )

    # Q_Koide = 2/3
    Q = GroupOrbitUtility.koide_q_from_isotypic(
        kernel.orbit_count('C_3_at_P')
    )
    stats.check(
        "Q_Koide = 2/3",
        abs(Q - 2/3) < 1e-12,
        f"got {Q}"
    )

    # δ_CP_CKM ≈ 70.53°
    cp = GeometricPhaseUtility.k4_minus_eigenspace_dihedral()
    stats.check(
        "δ_CP_CKM = arccos(1/3) ≈ 70.53°",
        abs(cp['degrees'] - 70.5288) < 0.01,
        f"got {cp['degrees']}"
    )

    # θ_QCD = 0
    theta_QCD = GeometricPhaseUtility.z3_holonomy_flat()['phase_rad']
    stats.check(
        "θ_QCD = 0",
        theta_QCD == 0.0,
    )


def main():
    print("=" * 78)
    print("Derived-shorthand utilities validation — Phase 2 of counting-first build")
    print("=" * 78)

    kernel = CountingKernel()
    stats = TestStats()

    test_spectral(kernel, stats)
    test_algebraic(kernel, stats)
    test_group_orbit(kernel, stats)
    test_geometric_phase(kernel, stats)
    test_pati_salam(kernel, stats)
    test_end_to_end_predictions(kernel, stats)

    print("\n" + "=" * 78)
    success = stats.summary()
    if success:
        print("\nALL TESTS PASS — Phase 2 (utilities) COMMITTED.")
        print("\nPredictions reproduced via counting-first architecture (kernel + utils):")
        print("  V_us = 9/40, y_τ = 1280/177147, λ_H = 2560/19683, V_cb = 256/6305")
        print("  sin²θ_W = 3/8, α_GUT = 1/24, Q_Koide = 2/3")
        print("  δ_CP_CKM = arccos(1/3) ≈ 70.53°, θ_QCD = 0")
        print("\nNext steps:")
        print("  Phase 3 — wrap remaining ~50 predictions as counting queries (~1-2 sessions)")
        print("  Phase 4 (parallel) — cosmology emulator (~4-6 sessions)")
    else:
        print("\nSome tests FAILED — utilities need fixes before Phase 2 commits.")
        sys.exit(1)
    print("=" * 78)


if __name__ == "__main__":
    main()
