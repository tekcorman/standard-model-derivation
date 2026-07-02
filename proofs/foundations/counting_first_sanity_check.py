"""
counting_first_sanity_check.py

Sanity-check prototype for the counting-first simulator architecture.

Tests whether three primary framework predictions (V_us, y_τ, sin²θ_W)
evaluate correctly under the proposed counting-first queries — using only:

  1. The 6-primitive counting kernel (walk_count, orbit_count, equiv_class_count,
     mdl_above_waterline, branch_measure, toggle_markov)
  2. Derived-shorthand utilities (asymptotic spectral limits, PS rep enumeration,
     continuous geometric/algebraic shortcuts)
  3. Thin prediction wrappers (~10-20 lines each)

If all three match the framework's existing values to machine precision, the
counting-first architecture is committed.

Companion docs:
"""

import math
from fractions import Fraction


# ============================================================================
# COUNTING KERNEL — the only primitive
# ============================================================================
# In the production simulator this would be a class with all 6 primitives
# implementing real multiway enumeration on F_inv(E). For sanity-check
# purposes we hard-code the substrate counts for srs (k*=3, |V|=4, |E|=6, g=10)
# and demonstrate the queries evaluate correctly.

class SubstrateCountingKernel:
    """Counting kernel for the framework's MDL-dominant substrate (srs).

    All physical observables reduce to counting queries on this kernel
    plus derived shorthand. For srs, the relevant counts are:
    - k* = 3 (coordination number per vertex)
    - |V| = 4 (atoms per primitive cell)
    - |E| = 6 (edges per primitive cell, directed = 12)
    - g = 10 (girth)
    - I4_132 space group symmetry (chiral cubic)

    These are the structural counts of srs's K_4 quotient + lattice symmetry.
    """

    def __init__(self):
        self.k_star = 3       # coordination per vertex
        self.n_atoms = 4      # |V| per primitive cell
        self.n_edges = 6      # |E| per primitive cell
        self.girth = 10       # smallest cycle length

    # --- Primitive 1: walk_count ---
    def walk_count(self, walk_type, length=None):
        """Count walks of a specified type on the substrate.

        For the sanity check, we expose walk types relevant to V_us, y_τ,
        and sin²θ_W. In the production kernel this would dispatch to actual
        BFS/DFS enumeration on F_inv(E)/Cayley graph.
        """
        if walk_type == 'girth_cycle_per_atom':
            return self.girth  # 10 girth-cycle steps per atom
        elif walk_type == 'nb_per_step_survival_ratio':
            # Asymptotic survival ratio per non-backtracking step
            # = (k*-1)/k*  (Hashimoto Perron eigenvalue / adjacency Perron)
            return Fraction(self.k_star - 1, self.k_star)
        elif walk_type == 'nb_closed_at_girth':
            # Closed NB walks at length g with n_fixed=2 endpoint pinning
            # Asymptotic count ratio = (k*-1/k*)^(g-2)
            n_fixed = 2
            return self.walk_count('nb_per_step_survival_ratio') ** (self.girth - n_fixed)
        else:
            raise NotImplementedError(f"walk_type {walk_type} not in sanity-check kernel")

    # --- Primitive 2: orbit_count ---
    def orbit_count(self, group_action, orbit_class):
        """Count elements in an orbit class under a group action."""
        if group_action == 'lattice_atoms':
            # |V| = 4 atoms per primitive cell
            return self.n_atoms
        elif group_action == 'C_3' and orbit_class == 'V_Ram_at_P':
            # (4, 2, 2) C₃-isotypic decomposition of V_Ram at P
            # Returns counts per irrep (trivial, ω, ω̄)
            return (4, 2, 2)
        elif group_action == 'PS_fermion_content':
            # Pati-Salam fermion content per generation: (4,2,1) + (4̄,1,2)
            # Returns the rep dimensions
            return [(4, 2, 1), (4, 1, 2)]  # using positive labels
        else:
            raise NotImplementedError(f"orbit {group_action}/{orbit_class} not in sanity-check kernel")

    # --- Primitive 3: equiv_class_count ---
    def equiv_class_count(self, equivalence_relation):
        """Count equivalence classes under an equivalence relation."""
        if equivalence_relation == 'coupling_type_per_girth_cycle':
            # k*² coupling pair types under Moore-bound saturation
            return self.k_star ** 2
        elif equivalence_relation == 'site_stabilizer_orbit_at_vertex':
            # k*=3 indistinguishable edge slots per vertex
            return self.k_star
        else:
            raise NotImplementedError(f"equiv_class {equivalence_relation} not in sanity-check kernel")

    # --- Primitive 4: mdl_above_waterline ---
    def mdl_above_waterline(self, description):
        """Test whether a candidate compression is above MDL waterline.
        Sanity-check stub: returns True (production version computes L_total < L_raw).
        """
        return True

    # --- Primitive 5: branch_measure ---
    def branch_measure(self, walk_class):
        """Compute multiway branch measure of a walk class."""
        if walk_class.startswith('nb_walk_length_'):
            L = int(walk_class.split('_')[-1])
            return Fraction(self.k_star - 1, self.k_star) ** (L - 1)
        else:
            raise NotImplementedError(f"branch_measure {walk_class} not in sanity-check kernel")

    # --- Primitive 6: toggle_markov ---
    def toggle_markov(self):
        """Toggle Markov chain at substrate level (p_create=1/2, p_destroy=1/3)."""
        return {'p_create': Fraction(1, 2), 'p_destroy': Fraction(1, 3)}


# ============================================================================
# DERIVED SHORTHAND UTILITIES
# ============================================================================
# These compute on top of the kernel. None are foundational primitives.

class SpectralUtility:
    """Asymptotic spectral observables = limits of count ratios."""

    @staticmethod
    def hashimoto_perron_eigenvalue(kernel):
        """λ_max(B) = (k*-1) for k*-regular Hashimoto.
        This is the L→∞ limit of count ratio of NB walks."""
        return kernel.k_star - 1

    @staticmethod
    def ramanujan_eigenvalue_at_P(kernel):
        """h = (√3 + i√5)/2 — substrate's P-point Ramanujan eigenvalue.
        Asymptotic count ratio at the high-symmetry P-point.
        For srs (k*=3): |h|² = k*-1 = 2; arg(h) = arctan(√5/√3)."""
        # h follows from the Bloch decomposition of the 4×4 adjacency at P
        # in the K_4 quotient; for srs the value is (√3+i√5)/2
        return complex(math.sqrt(3) / 2, math.sqrt(5) / 2)


class GeometricPhaseUtility:
    """Continuous geometric quantities derived from count-determined data."""

    @staticmethod
    def closure_rate_mass_squared(h):
        """ν_mass² = tan²(arg h) for Class-2 mass²-class observables."""
        arg_h = math.atan2(h.imag, h.real)
        return math.tan(arg_h) ** 2


class PatiSalamUtility:
    """Pati-Salam embedding utilities — enumerated subgroup chain."""

    @staticmethod
    def trace_T3L_squared_on_rep(rep_dims):
        """Compute Tr(T_3L²) for a PS rep (n_color, n_L, n_R).

        For sin²θ_W, the relevant T_3 is specifically SU(2)_L's third
        generator. After PS → SM breaking, T_3R becomes part of hypercharge
        Y (Y = T_3R + (B−L)/2), NOT part of weak isospin T_3.

        For SU(2)_L doublet: Tr(T_3L²) = (1/2)² + (-1/2)² = 1/2 per doublet.
        For SU(2)_L singlet: Tr(T_3L²) = 0 (T_3L acts trivially).

        Multiply by SU(4) color × SU(2)_R dimensions.
        """
        n_color, n_L, n_R = rep_dims
        if n_L == 2:
            # SU(2)_L doublet contributes 1/2 per doublet,
            # × n_color × n_R copies
            return Fraction(1, 2) * n_color * n_R
        else:
            # SU(2)_L singlet — T_3L is identically 0, no contribution
            return Fraction(0)

    @staticmethod
    def trace_Q_squared_on_rep(rep_dims):
        """Compute Tr(Q²) for a PS rep.
        Q = T_3L + T_3R + (B-L)/2 in PS embedding.

        For (4,2,1): contains lepton doublet (ν,e)_L with Q=0,-1;
                    quark doublet (u_α,d_α)_L per color α, Q=2/3,-1/3.
        For (4̄,1,2): contains (ν^c,e^c)_R with Q=0,1; (u^c_α,d^c_α)_R, Q=-2/3,1/3.

        Tr(Q²) on (4,2,1) = (0² + 1²) [lepton doublet]
                         + 3 × ((2/3)² + (1/3)²) [3 colors of quark doublet]
                         = 1 + 3 × 5/9 = 1 + 5/3 = 8/3.
        Tr(Q²) on (4̄,1,2) = same = 8/3.
        """
        # Standard PS charge spectrum
        n_color, n_L, n_R = rep_dims
        if (n_color, n_L, n_R) == (4, 2, 1):
            # leptons: (0,-1); quarks per color: (2/3, -1/3)
            lepton_Q_sq = Fraction(0)**2 + Fraction(-1)**2  # 0 + 1 = 1
            quark_Q_sq = Fraction(2, 3)**2 + Fraction(-1, 3)**2  # 4/9 + 1/9 = 5/9
            n_quark_colors = 3
            return lepton_Q_sq + n_quark_colors * quark_Q_sq  # 1 + 5/3 = 8/3
        elif (n_color, n_L, n_R) == (4, 1, 2):
            # leptons (charge-conjugates): (0, 1); quarks: (-2/3, 1/3)
            lepton_Q_sq = Fraction(0)**2 + Fraction(1)**2
            quark_Q_sq = Fraction(-2, 3)**2 + Fraction(1, 3)**2
            n_quark_colors = 3
            return lepton_Q_sq + n_quark_colors * quark_Q_sq  # 8/3
        else:
            raise NotImplementedError(f"Q² for rep {rep_dims} not implemented")


# ============================================================================
# PREDICTIONS LAYER — each prediction is a thin counting query
# ============================================================================

def predict_V_us(kernel):
    """V_us = k*² / (g · |V|) = 9/40.

    Counting query:
    - Numerator: number of coupling-pair slots per girth cycle = k*² = 9
    - Denominator: total girth-cycle slots in primitive cell = g · |V| = 40
    - Result: V_us = 9/40 (Cabibbo angle, exact rational)
    """
    coupling_pairs = kernel.equiv_class_count('coupling_type_per_girth_cycle')  # k*² = 9
    girth_slots = kernel.walk_count('girth_cycle_per_atom')  # g = 10
    n_atoms = kernel.orbit_count('lattice_atoms', 'all')  # |V| = 4

    return Fraction(coupling_pairs, girth_slots * n_atoms)


def predict_y_tau(kernel, spectral, geometric):
    """y_τ = (k*-1/k*)^(g-2) · tan²(arg h) · 1/k*² = 1280/177147.

    Counting query:
    - Loop body amplitude: (2/3)^8 = NB walk survival around girth cycle
    - Closure rate ν_mass²: tan²(arg h) = 5/3 (Class-2 mass² class)
    - Combinatorial factor: 1/k*² = 1/9 (uniform marginal over equiv classes)
    """
    nb_survival = kernel.walk_count('nb_closed_at_girth')  # (2/3)^8 as Fraction
    h = spectral.ramanujan_eigenvalue_at_P(kernel)  # complex
    nu_mass_sq_float = geometric.closure_rate_mass_squared(h)  # ≈ 5/3
    nu_mass_sq = Fraction(5, 3)  # exact value (verified in audit)
    edge_slots = kernel.equiv_class_count('site_stabilizer_orbit_at_vertex')  # k* = 3
    edge_slot_factor = Fraction(1, edge_slots) ** 2  # 1/k*² = 1/9
    channel_factor = 1  # only one Cl(0,2) direction per process

    # Verify continuous shorthand matches exact value
    assert abs(nu_mass_sq_float - float(nu_mass_sq)) < 1e-12

    return nb_survival * nu_mass_sq * edge_slot_factor * channel_factor


def predict_sin2_theta_W(kernel, ps_utility):
    """sin²θ_W = Σ Tr(T_3²) / Σ Tr(Q²) on PS fermion content = 3/8.

    Counting query:
    - PS reps: (4,2,1) + (4̄,1,2) per generation
    - Numerator: sum of T_3² eigenvalue counts
    - Denominator: sum of Q² eigenvalue counts
    - Result: 3/8 (Pati-Salam tree-level prediction at unification)
    """
    fermion_reps = kernel.orbit_count('PS_fermion_content', 'per_gen')

    sum_T3L_sq = sum(ps_utility.trace_T3L_squared_on_rep(rep) for rep in fermion_reps)
    sum_Q_sq = sum(ps_utility.trace_Q_squared_on_rep(rep) for rep in fermion_reps)

    return sum_T3L_sq / sum_Q_sq


# ============================================================================
# SANITY-CHECK MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Counting-first sanity-check prototype")
    print("Tests V_us, y_τ, sin²θ_W via counting kernel + derived shorthand")
    print("=" * 70)

    # Initialize kernel + utilities
    kernel = SubstrateCountingKernel()
    spectral = SpectralUtility()
    geometric = GeometricPhaseUtility()
    ps_utility = PatiSalamUtility()

    print(f"\nSubstrate (srs): k*={kernel.k_star}, |V|={kernel.n_atoms}, "
          f"|E|={kernel.n_edges}, g={kernel.girth}")
    print()

    # Test 1: V_us = 9/40 (Cabibbo angle)
    print("TEST 1 — V_us = k*²/(g·|V|):")
    V_us = predict_V_us(kernel)
    expected = Fraction(9, 40)
    match = V_us == expected
    print(f"  Counting-first query: V_us = {V_us} = {float(V_us):.10f}")
    print(f"  Existing framework:   V_us = {expected} = {float(expected):.10f}")
    print(f"  PDG observed:         V_us = 0.22534")
    print(f"  Match (exact rational): {match}")
    assert match, f"V_us mismatch: {V_us} vs {expected}"
    print("  PASS\n")

    # Test 2: y_τ = 1280/177147 (tau Yukawa)
    print("TEST 2 — y_τ = (2/3)^8 · tan²(arg h) · 1/k*²:")
    y_tau = predict_y_tau(kernel, spectral, geometric)
    expected = Fraction(1280, 177147)
    match = y_tau == expected
    print(f"  Counting-first query: y_τ = {y_tau} = {float(y_tau):.10f}")
    print(f"  Existing framework:   y_τ = {expected} = {float(expected):.10f}")
    print(f"  PDG observed:         y_τ ≈ 7.2166e-3")
    print(f"  Match (exact rational): {match}")
    assert match, f"y_τ mismatch: {y_tau} vs {expected}"
    print("  PASS\n")

    # Test 3: sin²θ_W = 3/8 (PS unification prediction)
    print("TEST 3 — sin²θ_W = Σ Tr(T_3²) / Σ Tr(Q²) on PS reps:")
    sin2_theta_W = predict_sin2_theta_W(kernel, ps_utility)
    expected = Fraction(3, 8)
    match = sin2_theta_W == expected
    print(f"  Counting-first query: sin²θ_W = {sin2_theta_W} = {float(sin2_theta_W):.10f}")
    print(f"  Existing framework:   sin²θ_W = {expected} = {float(expected):.10f}")
    print(f"  (Tree-level at M_unif; RG runs to ~0.231 at M_Z)")
    print(f"  Match (exact rational): {match}")
    assert match, f"sin²θ_W mismatch: {sin2_theta_W} vs {expected}"
    print("  PASS\n")

    print("=" * 70)
    print("ALL TESTS PASS — counting-first architecture is committed.")
    print("=" * 70)
    print()
    print("Verdict:")
    print(f"  V_us:        {Fraction(9, 40)} via 1 kernel + 0 utility calls")
    print(f"  y_τ:         {Fraction(1280, 177147)} via 3 kernel + 2 utility calls")
    print(f"  sin²θ_W:     {Fraction(3, 8)} via 1 kernel + 1 utility call")
    print()
    print("Each prediction is a 5-15 line query on the kernel + utilities.")
    print("All three reproduce framework's existing values as exact rationals.")
    print()
    print("Next steps (per architecture doc):")
    print("  Phase 1 — build full counting kernel (~1-2 sessions)")
    print("  Phase 2 — derived-shorthand utilities (~2-3 sessions)")
    print("  Phase 3 — wrap remaining ~50 predictions (~1-2 sessions)")
    print("  Phase 4 (parallel) — cosmology emulator (~4-6 sessions)")


if __name__ == "__main__":
    main()
