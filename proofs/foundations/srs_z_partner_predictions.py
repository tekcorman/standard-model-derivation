#!/usr/bin/env python3
"""
srs-z partner predictions: what would a Bayesian observer ON srs-z derive?

Building on the bipartite-double-cover identification (srs-z's primitive Q_3
quotient = bipartite double cover of srs's K_4 quotient), this script applies
each of the framework's prediction formulas using srs-z's substrate parameters
in place of srs's. The output is a comprehensive comparison table.

For each prediction, classify by structural dependence:
  CLASS A — depends on (k, g) only      → SAME on srs-z (k=3, g=10 both)
  CLASS B — depends on (|V|, |E|, k)    → CHANGES on srs-z (|V|, |E| double)
  CLASS C — depends on h at saddle      → SAME (K-rational h invariant)
  CLASS D — depends on multiplicity n_γ → DOUBLES on srs-z (mult 2 → 4)
  CLASS E — depends on Pati-Salam       → SAME (Cl(2k)=Cl(6) from k=3)

Substrate parameters:
                 srs     srs-z
  k              3       3              (CLASS A: same)
  g (girth)      10      10             (CLASS A: same)
  N_atoms_prim   4       8              (CLASS B: doubled)
  N_edges_prim   6       12             (CLASS B: doubled)
  α_1=(2/3)^(g-2)  (2/3)^8 same         (CLASS A: same)
  Re(h), Im(h)   √3/2, √5/2  same       (CLASS C: same)
  mult of h       2       4             (CLASS D: doubled)
  Pati-Salam     Spin(6) Spin(6)        (CLASS E: same)

NOT a derivation of SUSY masses; this is a structural comparison showing
WHICH framework predictions naturally double under the bipartite cover and
which are invariant.
"""

from fractions import Fraction
import math


# =============================================================================
# SUBSTRATE PARAMETERS
# =============================================================================

class Substrate:
    def __init__(self, name, k, g, N_atoms, N_edges, mult_h_saddle, sg):
        self.name = name
        self.k = k
        self.g = g
        self.N_atoms = N_atoms
        self.N_edges = N_edges
        self.mult_h_saddle = mult_h_saddle
        self.sg = sg
        # Derived
        self.q_NB = Fraction(k - 1, k)
        self.alpha_1 = self.q_NB ** (g - 2)
        # Re(h), Im(h) at the K-rational C₃-stabilized saddle
        # — same for any srs-related substrate at the framework's saddle:
        #   h = (√3 + i√5) / 2,  Re=√3/2,  Im=√5/2,  |h|²=2
        # Stored symbolically as strings for clarity.
        self.re_h = "√3/2"
        self.im_h = "√5/2"
        self.h_modsq = 2  # exact, both Ramanujan

    def __repr__(self):
        return (f"{self.name}: k={self.k}, g={self.g}, "
                f"|V|={self.N_atoms}, |E|={self.N_edges}, "
                f"mult(h)={self.mult_h_saddle}, sg={self.sg}")


srs   = Substrate('srs',   k=3, g=10, N_atoms=4, N_edges=6,  mult_h_saddle=2, sg='I4₁32')
srs_z = Substrate('srs-z', k=3, g=10, N_atoms=8, N_edges=12, mult_h_saddle=4, sg='P4₁32')


# =============================================================================
# FRAMEWORK PREDICTION FORMULAS (with substrate as parameter)
# =============================================================================
# Each function takes a Substrate and returns the predicted value, plus a
# 'class' tag (A/B/C/D/E) indicating structural dependence.

def predict_V_us(s):
    """V_us = k² / (g · N_atoms)  — Moore-bound counting density."""
    return {'value': Fraction(s.k**2, s.g * s.N_atoms), 'class': 'B', 'changes': 'doubled'}


def predict_V_cb(s):
    """V_cb = α_1 / (1 − α_1)  with α_1 = ((k−1)/k)^(g−2).
    Depends only on (k, g) — independent of N_atoms!"""
    a1 = s.alpha_1
    val = a1 / (1 - a1)
    return {'value': val, 'class': 'A', 'changes': 'same'}


def predict_V_ub(s):
    """V_ub = Σ_{m≥2} α_m / (1 − α_m), α_m = q_NB^L_eff(m), L_eff(m) = m·g − 2(m−1)·s − n_fixed.

    Canonical Row P14 / `proofs/flavor/vub_multicycle_sum.py` formula. For srs
    (g=10, s_seam=2, n_fixed=2): L_eff(m) = 6m+2 → V_ub = 3.767e-3 (−0.26σ
    PDG combined). Depends on (k, g) only via L_eff(m); CLASS A.

    PRIOR BUG (fixed 2026-05-02): this function returned `α_1²/(1-α_1)` =
    single-winding form (= V_us·α_1, not the M1 multi-cycle sum). That gave
    V_ub = 0.00158 (-11σ) which propagated through audit ensembles as a
    fake -11σ residue. The docstring even said "Σ (2/3)^(8m) for m ≥ 2"
    but the implementation was wrong. Now matches the canonical M1
    multi-cycle sum exactly. See parallel fix in `rcsr_full_ensemble_audit.py`
    (commit 49bebd4) and `rcsr_survivors_full_ledger_walk.py`.

    Substrate-uniform (k, g) parameters: assumes s_seam=2, n_fixed=2 across
    all candidates (consistent with framework's general girth-cycle structure;
    these are NOT explicit Substrate fields but the formula uses substrate.g).
    """
    s_seam = 2
    n_fixed = 2
    val = 0.0
    for m in range(2, 100):
        L_eff = m * s.g - 2 * (m - 1) * s_seam - n_fixed
        if L_eff <= 0:
            continue
        alpha_m = float(s.q_NB) ** L_eff
        if 0 < alpha_m < 1:
            val += alpha_m / (1 - alpha_m)
        else:
            break
    return {'value': val, 'class': 'A', 'changes': 'same'}


def predict_Q_Koide(s):
    """Q_Koide = 2/3 = q_NB — depends on k only."""
    return {'value': s.q_NB, 'class': 'A', 'changes': 'same'}


def predict_alpha_1(s):
    """α_1 = ((k−1)/k)^(g−2)."""
    return {'value': s.alpha_1, 'class': 'A', 'changes': 'same'}


def predict_y_tau(s):
    """y_τ = α_1² / (16π²) — Yukawa, depends on α_1."""
    a1 = float(s.alpha_1)
    val = a1**2 / (16 * math.pi**2)
    return {'value': val, 'class': 'A', 'changes': 'same'}


def predict_dark_c(s):
    """Dark coefficient c = (2(|E|−|V|)+1) / (2|E|) — depends on (|V|, |E|)."""
    return {'value': Fraction(2*(s.N_edges - s.N_atoms) + 1, 2*s.N_edges),
            'class': 'B', 'changes': 'shifts'}


def predict_eta_B_factor(s):
    """η_B = ε_CP · Re(h_saddle) · α_1^M  — M = N_edges = chain length.
    Returns the magnitude as a tuple of components."""
    M = s.N_edges
    a1_pow_M = float(s.alpha_1)**M
    return {'value': a1_pow_M, 'M_chain': M, 'class': 'B',
            'changes': 'severely suppressed (chain doubles)'}


def predict_epsilon_CP(s):
    """ε_CP = (k-2)/(k+2)  — Bayesian-toggle, depends on k only."""
    return {'value': Fraction(s.k - 2, s.k + 2), 'class': 'A', 'changes': 'same'}


def predict_Re_h(s):
    """Re(h) at K-rational C₃-stabilized saddle = √3/2.
    Same K-rational value on srs (at k_P) and srs-z (at k=R) — verified
    via bipartite-double-cover spectrum probe."""
    return {'value': "√3/2 ≈ 0.8660", 'class': 'C', 'changes': 'same'}


def predict_Im_h_over_h_modsq(s):
    """Im(h)/|h|² = (√5/2)/2 = √5/4 — depends on h saddle eigenvalue only."""
    return {'value': "√5/4 ≈ 0.5590", 'class': 'C', 'changes': 'same'}


def predict_sin2_theta_W(s):
    """sin²θ_W = 3/8 — depends on Pati-Salam embedding, comes from Cl(2k)=Cl(6) at k=3.
    Same on any k=3 substrate."""
    return {'value': Fraction(3, 8), 'class': 'E', 'changes': 'same'}


def predict_alpha_GUT(s):
    """α_GUT = 1/24 (per framework's R3 + Pati-Salam embedding).
    Same as long as k=3 → Cl(6) → Spin(6)."""
    return {'value': Fraction(1, 24), 'class': 'E', 'changes': 'same'}


def predict_n_gamma(s):
    """Number of photon polarizations = multiplicity of h at saddle.
    On srs: mult 2 (L = ω-irrep + R = ω²-irrep at k_P, both C₃ irreps).
    On srs-z: mult 4 (bipartite-double splitting, EACH C₃ irrep has + and − sectors)."""
    return {'value': s.mult_h_saddle, 'class': 'D', 'changes': 'doubled (bipartite splitting)'}


def predict_n_gamma_factor(s):
    """1/n_γ factor in framework formulas (e.g. Re(h_P) absorbing photon normalization).
    On srs: 1/n_γ = 1/2.
    On srs-z: 1/n_γ = 1/4 (DOUBLED → halved). This would manifest as a factor-of-2
    shift in η_B and possibly other photon-coupled observables on srs-z."""
    return {'value': Fraction(1, s.mult_h_saddle), 'class': 'D',
            'changes': 'halves (mult doubles)'}


# =============================================================================
# COMPREHENSIVE COMPARISON
# =============================================================================

PREDICTIONS = [
    ('V_us',                       predict_V_us,                  'CKM matrix element |V_us| (Cabibbo)'),
    ('V_cb',                       predict_V_cb,                  'CKM matrix element |V_cb|'),
    ('V_ub',                       predict_V_ub,                  'CKM matrix element |V_ub|'),
    ('Q_Koide',                    predict_Q_Koide,               'Koide quark mass ratio'),
    ('α_1',                        predict_alpha_1,               'Feshbach exponent base'),
    ('y_τ',                        predict_y_tau,                 'Tau Yukawa coupling'),
    ('dark c',                     predict_dark_c,                'Dark-correction coefficient'),
    ('η_B chain factor',           predict_eta_B_factor,          'Sakharov chain α_1^M (η_B suppression)'),
    ('ε_CP',                       predict_epsilon_CP,            'Per-process CP asymmetry'),
    ('Re(h_saddle)',               predict_Re_h,                  'Hashimoto saddle real part'),
    ('Im(h)/|h|²',                 predict_Im_h_over_h_modsq,     'Dark amplitude factor'),
    ('sin²θ_W',                    predict_sin2_theta_W,          'Weak mixing angle at unification'),
    ('α_GUT',                      predict_alpha_GUT,             'GUT-scale coupling'),
    ('n_γ (h mult at saddle)',     predict_n_gamma,               'Photon polarization count'),
    ('1/n_γ factor',               predict_n_gamma_factor,        'Photon normalization in formulas'),
]

CLASS_LABELS = {
    'A': "depends on (k, g) only — bipartite-cover INVARIANT",
    'B': "depends on (|V|, |E|) — CHANGES under cover",
    'C': "depends on h saddle value — INVARIANT (K-rational h preserved)",
    'D': "depends on h multiplicity — DOUBLES under cover",
    'E': "depends on Pati-Salam ⊂ Spin(2k) — INVARIANT (k=3 same)",
}


def main():
    print("=" * 90)
    print("srs-z PARTNER PREDICTIONS — what a Bayesian observer ON srs-z would derive")
    print("=" * 90)
    print()
    print("Substrate parameters:")
    print(f"  srs:   {srs}")
    print(f"  srs-z: {srs_z}")
    print()
    print("Class system for structural dependence:")
    for c, label in CLASS_LABELS.items():
        print(f"  CLASS {c}: {label}")
    print()

    # Comparison table
    print("=" * 90)
    print(f"{'Prediction':<22s} {'Class':<6s}  {'srs value':<22s}  {'srs-z value':<22s}  {'Change'}")
    print("-" * 90)

    for name, fn, desc in PREDICTIONS:
        result_srs = fn(srs)
        result_z = fn(srs_z)
        cls = result_srs['class']
        changes = result_srs['changes']
        v_srs = result_srs['value']
        v_z = result_z['value']

        # Format values nicely
        def fmt(v):
            if isinstance(v, Fraction):
                if v.denominator == 1:
                    return str(v.numerator)
                return f"{v.numerator}/{v.denominator} = {float(v):.5g}"
            elif isinstance(v, float):
                return f"{v:.5g}"
            else:
                return str(v)

        print(f"{name:<22s} {cls:<6s}  {fmt(v_srs):<22s}  {fmt(v_z):<22s}  {changes}")

    # Spotlight — what changes
    print()
    print("=" * 90)
    print("SPOTLIGHT — predictions that DIFFER on srs-z (CLASS B + CLASS D)")
    print("=" * 90)

    # V_us
    v_us_srs = predict_V_us(srs)['value']
    v_us_z = predict_V_us(srs_z)['value']
    pdg_v_us = 0.22501
    print(f"\nV_us (CKM Cabibbo angle, depends on N_atoms):")
    print(f"  srs:    {v_us_srs} = {float(v_us_srs):.6f}  (matches PDG {pdg_v_us:.5f} at -0.015σ — framework's CONFIRMED)")
    print(f"  srs-z:  {v_us_z} = {float(v_us_z):.6f}  (50% off from PDG)")
    print(f"  → If srs-z hosts independent SM-like physics, its V_us is wrong by 100σ.")
    print(f"  → If srs-z hosts SUSY partners, this isn't 'the' V_us — it's the SUSY-flavor V_us")
    print(f"    coupling, which would manifest as a sub-leading correction to SM V_us.")

    # η_B chain
    eta_srs = predict_eta_B_factor(srs)
    eta_z = predict_eta_B_factor(srs_z)
    print(f"\nη_B chain factor α_1^M (Sakharov chain length):")
    print(f"  srs:    M=6,  α_1^M = (2/3)^48 ≈ {eta_srs['value']:.3e}")
    print(f"  srs-z:  M=12, α_1^M = (2/3)^96 ≈ {eta_z['value']:.3e}")
    print(f"  ratio (srs-z / srs) ≈ {eta_z['value'] / eta_srs['value']:.3e}")
    print(f"  → srs-z's η_B contribution is SEVERELY suppressed by extra (2/3)^48 ≈ 3.5×10⁻⁹.")
    print(f"  → SUSY interpretation: SUSY contributions to baryogenesis are negligible —")
    print(f"    matches phenomenology (SUSY models contribute to η_B mostly via SM loops).")

    # dark c
    c_srs = predict_dark_c(srs)['value']
    c_z = predict_dark_c(srs_z)['value']
    print(f"\nDark coefficient c (delocalized observables):")
    print(f"  srs:    c = {c_srs} = {float(c_srs):.5f}  (drives H_0, t_0, Ω_DM predictions on srs)")
    print(f"  srs-z:  c = {c_z} = {float(c_z):.5f}")
    print(f"  → Different rational, both in K = ℚ(√2,√3,√5).")
    print(f"  → Whether srs-z's c=3/8 has phenomenological meaning depends on whether srs-z")
    print(f"    contributes to delocalized observables. If srs-z is the SUSY substrate, its")
    print(f"    dark c would govern superparticle dark-matter sector predictions.")

    # n_γ
    n_g_srs = predict_n_gamma(srs)['value']
    n_g_z = predict_n_gamma(srs_z)['value']
    print(f"\nPhoton polarization count n_γ = mult(h at saddle):")
    print(f"  srs:    n_γ = {n_g_srs}  (L = ω-irrep + R = ω²-irrep at k_P, both C₃ irreps)")
    print(f"  srs-z:  n_γ = {n_g_z}  (each C₃ irrep splits via bipartite involution: +/- sectors)")
    print(f"  → DOUBLED count is the algebraic signature of bipartite-cover Z_2 grading.")
    print(f"  → SUSY interpretation: the doubled multiplicity provides 4 'photon-like' channels")
    print(f"    on srs-z = 2 SM photon polarizations + 2 SUSY-partner 'photino' analogs.")

    # SUSY-flavored interpretation
    print()
    print("=" * 90)
    print("SUSY-FLAVORED INTERPRETATION (structural, not derivational)")
    print("=" * 90)
    print("""
  The bipartite-double-cover relationship gives EXACTLY the structure SUSY needs:

  1. **Z_2 grading**: bipartite vertex labels ±1 ↔ boson/fermion.
  2. **State doubling**: every srs eigenstate at k_P lifts to 2 srs-z eigenstates
     at k=R, related by the bipartite involution (the substrate-level Q operator).
  3. **K-rational h preserved**: h = (√3+i√5)/2 same on both — same fundamental
     mass scale. SUSY mass HIERARCHY would come from Boltzmann suppression of
     srs-z's contribution by ~2^(-3.25) ≈ 10× per the M2a structural-DL audit.
  4. **Severe η_B suppression**: srs-z's chain doubling gives ~10⁻⁹ extra
     suppression, so SUSY contributions to baryogenesis are negligible.
  5. **Yukawa sector unchanged**: y_τ, m_τ, m_e, m_μ, V_cb depend on (k, g, α_1)
     only — bipartite-cover INVARIANT. Framework's PDG match on these is unaffected
     by srs-z presence.
  6. **V_us shift**: would be 50% off if srs-z dominates. Since framework matches
     PDG at -0.015σ, srs-z's contribution to V_us is constrained to <1%.
     This bounds srs-z's effective Boltzmann weight at <0.01 → ΔDL > 6.6 bits.

  The framework's currently-noted ~0.5% systematic floor (un-derived sub-leading
  Feshbach analog) on Yukawa-derived quantities is approximately the right order
  of magnitude for SUSY threshold corrections.

  **Concrete falsifiable prediction:** if the SUSY interpretation is correct,
  framework predictions should systematically deviate from PDG by terms of order
  (Boltzmann weight of srs-z) × (loop factor) — roughly 0.1-1%. Matches observed
  systematics. NOT a derivation; a hypothesis with quantitative structure.
""")


if __name__ == '__main__':
    main()
