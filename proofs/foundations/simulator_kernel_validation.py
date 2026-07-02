"""
simulator_kernel_validation.py

Comprehensive validation probe for the counting-first simulator kernel
(Phase 1 build).

Tests all 6 kernel primitives against existing framework apparatus:
  1. walk_count        — against analytical NB-walk formulas + explicit enumeration
  2. orbit_count       — against (4,2,2) C₃-isotypic, PS reps, etc.
  3. equiv_class_count — against k*², k*, Cl(6) Fock counts
  4. mdl_above_waterline — basic logic test
  5. branch_measure    — against (2/3)^(L-1) formula + V_cb geometric series
  6. toggle_markov     — against edge-surprise threshold values

Then runs the three sanity-check predictions (V_us, y_τ, sin²θ_W) using
ONLY kernel + minimal computation, verifying they match existing values
to exact rational precision.

If all tests pass, Phase 1 (counting kernel) is committed.

Predecessors:
- simulator/kernel.py
- simulator/srs_substrate.py
- proofs/foundations/counting_first_sanity_check.py (prototype)
"""

import sys
import math
from pathlib import Path
from fractions import Fraction

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from simulator.srs_engine import CountingKernel, SrsSubstrate


# ============================================================================
# TEST INFRASTRUCTURE
# ============================================================================

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


# ============================================================================
# PRIMITIVE 1 — walk_count
# ============================================================================

def test_walk_count(kernel, stats):
    print("\n[1] walk_count")

    # NB survival per step = (k*-1)/k* = 2/3
    survival = kernel.walk_count('nb_per_step_survival_ratio')
    stats.check(
        "nb_per_step_survival_ratio = 2/3",
        survival == Fraction(2, 3),
        f"got {survival}"
    )

    # NB closure at girth: (2/3)^8 = 256/6561
    nb_closed = kernel.walk_count('nb_closed_at_girth')
    expected = Fraction(256, 6561)
    stats.check(
        "nb_closed_at_girth = (2/3)^8 = 256/6561",
        nb_closed == expected,
        f"got {nb_closed}"
    )

    # Girth-cycle slots per atom = g = 10
    girth_slots = kernel.walk_count('girth_cycle_per_atom')
    stats.check(
        "girth_cycle_per_atom = g = 10",
        girth_slots == 10,
        f"got {girth_slots}"
    )

    # Asymptotic adjacency Perron = k* = 3
    adj_perron = kernel.walk_count('asymptotic_perron')
    stats.check(
        "asymptotic_perron = k* = 3",
        adj_perron == 3,
        f"got {adj_perron}"
    )

    # Asymptotic Hashimoto Perron = k*-1 = 2
    hash_perron = kernel.walk_count('asymptotic_hashimoto_perron')
    stats.check(
        "asymptotic_hashimoto_perron = k*-1 = 2",
        hash_perron == 2,
        f"got {hash_perron}"
    )

    # Explicit closed walks of length 2 on K_4 quotient
    # K_4 quotient adjacency for srs: each vertex has degree 3 (k*=3)
    # Trace(A^2) = sum of degrees = 4 vertices × 3 edges = 12
    closed_2 = kernel.walk_count('closed_explicit', length=2)
    stats.check(
        "closed_explicit length=2 = sum of degrees = 12",
        closed_2 == 12,
        f"got {closed_2}"
    )

    # Explicit closed walks of length 3 on K_4 quotient
    # Trace(A^3) = number of closed length-3 walks
    # For K_4 (complete graph on 4 vertices, but srs quotient has 6 edges):
    # Each vertex has 3 neighbors; closed length-3 walks return via cycle
    closed_3 = kernel.walk_count('closed_explicit', length=3)
    stats.check(
        "closed_explicit length=3 ≥ 0 (well-formed integer)",
        isinstance(closed_3, int) and closed_3 >= 0,
        f"got {closed_3}"
    )
    print(f"    closed walks of length 3 on K_4 quotient = {closed_3}")

    # Explicit NB closed walks of length 4 (smallest possible NB cycle)
    nb_closed_4 = kernel.walk_count('nb_closed_explicit', length=4)
    print(f"    NB closed walks of length 4 = {nb_closed_4}")
    # Note: girth=10 for srs lattice, but K_4 quotient has shorter cycles
    # since cell-offset NB walks can close in fewer steps on the quotient
    stats.check(
        "nb_closed_explicit length=4 is well-formed integer",
        isinstance(nb_closed_4, int) and nb_closed_4 >= 0,
        f"got {nb_closed_4}"
    )


# ============================================================================
# PRIMITIVE 2 — orbit_count
# ============================================================================

def test_orbit_count(kernel, stats):
    print("\n[2] orbit_count")

    # Lattice atoms = |V| = 4
    n_atoms = kernel.orbit_count('lattice_atoms')
    stats.check(
        "lattice_atoms = 4",
        n_atoms == 4,
        f"got {n_atoms}"
    )

    # C₃ at P-point: (4, 2, 2)
    c3_decomp = kernel.orbit_count('C_3_at_P')
    stats.check(
        "C_3_at_P = (4, 2, 2) [trivial, ω, ω̄ multiplicities]",
        c3_decomp == (4, 2, 2),
        f"got {c3_decomp}"
    )

    # PS fermion content: [(4,2,1), (4̄,1,2)]
    ps_reps = kernel.orbit_count('PS_fermion_content')
    expected = [(4, 2, 1), (4, 1, 2)]
    stats.check(
        "PS_fermion_content = [(4,2,1), (4̄,1,2)]",
        ps_reps == expected,
        f"got {ps_reps}"
    )

    # Fermion states per generation = 8 (Cl(6) spinor dim)
    n_fermions = kernel.orbit_count('fermion_content_per_gen')
    stats.check(
        "fermion_content_per_gen = 8 (Cl(6) spinor dim)",
        n_fermions == 8,
        f"got {n_fermions}"
    )

    # Gauge bosons = 12
    n_bosons = kernel.orbit_count('gauge_bosons')
    stats.check(
        "gauge_bosons = 12 (8 + 3 + 1)",
        n_bosons == 12,
        f"got {n_bosons}"
    )

    # Generations = 3
    n_gen = kernel.orbit_count('generations')
    stats.check(
        "generations = 3 (Galois Z_3 orbit)",
        n_gen == 3,
        f"got {n_gen}"
    )


# ============================================================================
# PRIMITIVE 3 — equiv_class_count
# ============================================================================

def test_equiv_class_count(kernel, stats):
    print("\n[3] equiv_class_count")

    # Coupling pairs per girth cycle = k*² = 9
    coupling_pairs = kernel.equiv_class_count('coupling_pair_per_girth_cycle')
    stats.check(
        "coupling_pair_per_girth_cycle = k*² = 9",
        coupling_pairs == 9,
        f"got {coupling_pairs}"
    )

    # Site stabilizer orbit at vertex = k* = 3
    site_orbit = kernel.equiv_class_count('site_stabilizer_orbit_at_vertex')
    stats.check(
        "site_stabilizer_orbit_at_vertex = k* = 3",
        site_orbit == 3,
        f"got {site_orbit}"
    )

    # Cl(6) Fock label slots = 2^k* × k* = 24 (= 1/α_GUT)
    fock_slots = kernel.equiv_class_count('cl6_fock_label_slots')
    stats.check(
        "cl6_fock_label_slots = 2^k* × k* = 24",
        fock_slots == 24,
        f"got {fock_slots}"
    )


# ============================================================================
# PRIMITIVE 4 — mdl_above_waterline
# ============================================================================

def test_mdl_above_waterline(kernel, stats):
    print("\n[4] mdl_above_waterline")

    # Compression saves bits: above waterline
    above = kernel.mdl_above_waterline(
        model_bits=10, data_bits_given_model=20, raw_data_bits=100
    )
    stats.check(
        "10+20 < 100 → above waterline",
        above is True,
        f"got {above}"
    )

    # Compression doesn't save bits: below waterline
    below = kernel.mdl_above_waterline(
        model_bits=50, data_bits_given_model=80, raw_data_bits=100
    )
    stats.check(
        "50+80 > 100 → below waterline",
        below is False,
        f"got {below}"
    )

    # Exact equality at waterline: below (strict inequality)
    at_line = kernel.mdl_above_waterline(
        model_bits=50, data_bits_given_model=50, raw_data_bits=100
    )
    stats.check(
        "50+50 = 100 → below waterline (strict)",
        at_line is False,
        f"got {at_line}"
    )

    # Savings calculation
    savings = kernel.waterline_savings(10, 20, 100)
    stats.check(
        "savings = 100 - (10+20) = 70",
        savings == 70,
        f"got {savings}"
    )


# ============================================================================
# PRIMITIVE 5 — branch_measure
# ============================================================================

def test_branch_measure(kernel, stats):
    print("\n[5] branch_measure")

    # Single NB walk of length 1: μ = (2/3)^0 = 1
    mu_1 = kernel.branch_measure('nb_walk', length=1)
    stats.check(
        "nb_walk length=1 → μ = 1",
        mu_1 == 1,
        f"got {mu_1}"
    )

    # Single NB walk of length 5: μ = (2/3)^4 = 16/81
    mu_5 = kernel.branch_measure('nb_walk', length=5)
    expected = Fraction(16, 81)
    stats.check(
        "nb_walk length=5 → μ = (2/3)^4 = 16/81",
        mu_5 == expected,
        f"got {mu_5}"
    )

    # Single NB walk at girth - n_fixed = 8: μ = (2/3)^7 = 128/2187
    mu_8 = kernel.branch_measure('nb_walk', length=8)
    expected = Fraction(128, 2187)
    stats.check(
        "nb_walk length=8 → μ = (2/3)^7 = 128/2187",
        mu_8 == expected,
        f"got {mu_8}"
    )

    # V_cb geometric series: alpha_1 / (1 - alpha_1) where alpha_1 = (2/3)^(L-1)
    # For L=9 (girth - n_fixed + 1 = 8 + 1, length used in branch measure):
    # alpha_1 = (2/3)^8 = 256/6561, geometric sum = 256/6305
    v_cb = kernel.branch_measure('nb_walk_geometric_sum', length=9)
    expected = Fraction(256, 6305)
    stats.check(
        "nb_walk_geometric_sum length=9 → V_cb = 256/6305",
        v_cb == expected,
        f"got {v_cb}"
    )


# ============================================================================
# PRIMITIVE 6 — toggle_markov
# ============================================================================

def test_toggle_markov(kernel, stats):
    print("\n[6] toggle_markov")

    rates = kernel.toggle_markov()

    # p_create = 1/2 from edge-surprise threshold theorem
    stats.check(
        "p_create = 1/2",
        rates['p_create'] == Fraction(1, 2),
        f"got {rates['p_create']}"
    )

    # p_destroy = 1/k* = 1/3
    stats.check(
        "p_destroy = 1/3",
        rates['p_destroy'] == Fraction(1, 3),
        f"got {rates['p_destroy']}"
    )

    # S_fresh = 1 bit
    stats.check(
        "s_fresh_bits = 1",
        abs(rates['s_fresh_bits'] - 1.0) < 1e-12,
        f"got {rates['s_fresh_bits']}"
    )

    # S_disconfirm = log₂(3) ≈ 1.585
    stats.check(
        "s_disconfirm_bits = log₂(3)",
        abs(rates['s_disconfirm_bits'] - math.log2(3.0)) < 1e-12,
        f"got {rates['s_disconfirm_bits']}"
    )

    # Asymmetry > 0 (persistence-is-disruptive engine)
    stats.check(
        "asymmetry_bits = log₂(3/2) > 0",
        abs(rates['asymmetry_bits'] - math.log2(1.5)) < 1e-12 and rates['asymmetry_bits'] > 0,
        f"got {rates['asymmetry_bits']}"
    )


# ============================================================================
# SANITY-CHECK PREDICTIONS — V_us, y_τ, sin²θ_W via the kernel
# ============================================================================

def test_predictions_via_kernel(kernel, stats):
    """Reproduce the three sanity-check predictions using ONLY the kernel."""
    print("\n[Predictions] V_us, y_τ, sin²θ_W via kernel")

    # V_us = k*²/(g·|V|) = 9/40
    coupling_pairs = kernel.equiv_class_count('coupling_pair_per_girth_cycle')  # 9
    girth_slots = kernel.walk_count('girth_cycle_per_atom')  # 10
    n_atoms = kernel.orbit_count('lattice_atoms')  # 4
    V_us = Fraction(coupling_pairs, girth_slots * n_atoms)
    stats.check(
        "V_us = k*²/(g·|V|) = 9/40",
        V_us == Fraction(9, 40),
        f"got {V_us}"
    )

    # y_τ = (2/3)^8 × tan²(arg h) × 1/k*² = 1280/177147
    nb_survival = kernel.walk_count('nb_closed_at_girth')  # (2/3)^8 = 256/6561
    nu_mass_sq = Fraction(5, 3)  # tan²(arg h) — exact value
    edge_slot_factor = Fraction(1, kernel.equiv_class_count('site_stabilizer_orbit_at_vertex')) ** 2  # 1/9
    y_tau = nb_survival * nu_mass_sq * edge_slot_factor
    stats.check(
        "y_τ = (2/3)^8 · 5/3 · 1/9 = 1280/177147",
        y_tau == Fraction(1280, 177147),
        f"got {y_tau}"
    )

    # Verify tan²(arg h) gives 5/3 numerically (continuous shorthand check)
    h = kernel.substrate.ramanujan_eigenvalue_at_P
    nu_mass_sq_numeric = math.tan(math.atan2(h.imag, h.real)) ** 2
    stats.check(
        "tan²(arg h) ≈ 5/3 [continuous shorthand]",
        abs(nu_mass_sq_numeric - 5/3) < 1e-12,
        f"got {nu_mass_sq_numeric}"
    )

    # sin²θ_W = Σ Tr(T_3L²) / Σ Tr(Q²) on PS reps = 3/8
    ps_reps = kernel.orbit_count('PS_fermion_content')

    def trace_T3L_squared(rep):
        n_color, n_L, n_R = rep
        if n_L == 2:
            return Fraction(1, 2) * n_color * n_R
        return Fraction(0)

    def trace_Q_squared(rep):
        n_color, n_L, n_R = rep
        if (n_color, n_L, n_R) == (4, 2, 1):
            lepton_Q_sq = Fraction(0)**2 + Fraction(-1)**2
            quark_Q_sq = Fraction(2, 3)**2 + Fraction(-1, 3)**2
            return lepton_Q_sq + 3 * quark_Q_sq  # 1 + 5/3 = 8/3
        elif (n_color, n_L, n_R) == (4, 1, 2):
            lepton_Q_sq = Fraction(0)**2 + Fraction(1)**2
            quark_Q_sq = Fraction(-2, 3)**2 + Fraction(1, 3)**2
            return lepton_Q_sq + 3 * quark_Q_sq  # 8/3
        else:
            raise ValueError(f"Unexpected rep {rep}")

    sum_T3L_sq = sum(trace_T3L_squared(r) for r in ps_reps)  # 2
    sum_Q_sq = sum(trace_Q_squared(r) for r in ps_reps)  # 16/3
    sin2_theta_W = sum_T3L_sq / sum_Q_sq
    stats.check(
        "sin²θ_W = 2/(16/3) = 3/8",
        sin2_theta_W == Fraction(3, 8),
        f"got {sin2_theta_W}"
    )


# ============================================================================
# CROSS-VALIDATION against existing apparatus
# ============================================================================

def test_cross_validation(kernel, stats):
    """Cross-validate kernel against existing prediction scripts."""
    print("\n[Cross-validation] kernel vs existing apparatus")

    substrate = kernel.substrate

    # Adjacency at P should diagonalize to ±√3 (each multiplicity 2)
    A_P = substrate.adjacency_at_k('P')
    evals_P = sorted(np.linalg.eigvalsh(A_P).real)
    stats.check(
        "A(P) eigenvalues = (-√3, -√3, +√3, +√3)",
        np.allclose(evals_P, [-math.sqrt(3), -math.sqrt(3), math.sqrt(3), math.sqrt(3)]),
        f"got {evals_P}"
    )

    # Hashimoto B(P) should have h = (√3+i√5)/2 as an eigenvalue (multiplicity 2)
    B_P = substrate.hashimoto_at_k('P')
    eigs_B = np.linalg.eigvals(B_P)
    h_expected = complex(math.sqrt(3) / 2, math.sqrt(5) / 2)
    # Find closest eigenvalue to h
    closest_dist = min(abs(eig - h_expected) for eig in eigs_B)
    stats.check(
        "B(P) has h = (√3+i√5)/2 as eigenvalue",
        closest_dist < 1e-10,
        f"closest distance {closest_dist}"
    )

    # The eigenvalue h appears with multiplicity 2
    n_h_eigs = sum(1 for eig in eigs_B if abs(eig - h_expected) < 1e-10)
    stats.check(
        "B(P) eigenvalue h has multiplicity 2",
        n_h_eigs == 2,
        f"found {n_h_eigs} copies"
    )

    # Adjacency at Γ: eigenvalues should be (3, -1, -1, -1) for K_4 quotient
    A_Gamma = substrate.adjacency_at_k('Gamma')
    evals_Gamma = sorted(np.linalg.eigvalsh(A_Gamma).real, reverse=True)
    stats.check(
        "A(Γ) eigenvalues = (+3, -1, -1, -1) [K_4 spectrum]",
        np.allclose(evals_Gamma, [3.0, -1.0, -1.0, -1.0]),
        f"got {evals_Gamma}"
    )

    # Closure rates match expected values
    stats.check(
        "ν_amplitude = √5/4 ≈ 0.559",
        abs(substrate.closure_rate_amplitude - math.sqrt(5)/4) < 1e-12,
        f"got {substrate.closure_rate_amplitude}"
    )
    stats.check(
        "ν_mass² = 5/3 ≈ 1.667",
        abs(substrate.closure_rate_mass_squared - 5/3) < 1e-12,
        f"got {substrate.closure_rate_mass_squared}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("Counting kernel validation — Phase 1 of counting-first simulator build")
    print("=" * 78)

    kernel = CountingKernel()
    stats = TestStats()

    # Print substrate summary
    summary = kernel.substrate_summary()
    print(f"\nSubstrate: {summary['name']} ({summary['space_group']}, {summary['wyckoff']})")
    print(f"  k* = {summary['k_star']}, |V| = {summary['n_atoms_per_cell']}, "
          f"|E| = {summary['n_edges_per_cell']}, g = {summary['girth']}")
    print(f"  Adjacency Perron λ = {summary['adjacency_perron']}")
    print(f"  Hashimoto Perron λ = {summary['hashimoto_perron']}")
    print(f"  Ramanujan eigenvalue h = {summary['ramanujan_eigenvalue_at_P']}")
    print(f"  Closure rates: {summary['closure_rates']}")

    # Run all tests
    test_walk_count(kernel, stats)
    test_orbit_count(kernel, stats)
    test_equiv_class_count(kernel, stats)
    test_mdl_above_waterline(kernel, stats)
    test_branch_measure(kernel, stats)
    test_toggle_markov(kernel, stats)
    test_predictions_via_kernel(kernel, stats)
    test_cross_validation(kernel, stats)

    print("\n" + "=" * 78)
    success = stats.summary()
    if success:
        print("\nALL TESTS PASS — Phase 1 (counting kernel) COMMITTED.")
        print("\nNext steps:")
        print("  Phase 2 — derived-shorthand utilities (~2-3 sessions)")
        print("  Phase 3 — wrap remaining ~50 predictions as counting queries (~1-2 sessions)")
        print("  Phase 4 (parallel) — cosmology emulator (~4-6 sessions)")
    else:
        print("\nSome tests FAILED — kernel needs fixes before Phase 1 commits.")
        sys.exit(1)
    print("=" * 78)


if __name__ == "__main__":
    main()
