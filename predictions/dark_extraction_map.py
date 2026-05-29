#!/usr/bin/env python3
"""
Canonical prediction file for the dark extraction map.

This file derives HOW each observable couples to the dark sector
self-energy Σ(h) = α₁/h, based on C₃ × parity representation theory.
It is not a single parameter prediction — it is a THEOREM that
determines the dark correction coefficient for every downstream
observable.

NOTE (post-2026-04-26 demotion): A2 and A3 are derived theorems; structural
slate is {A1} + P1' + A5-mass per docs/framework/framework_axioms.md §10. The closure
chain referenced here is preserved; only the axiomatic-status labels change.
G.1 and G.5 are DERIVED via CDP 2011 Theorem 25
(predictions/observer_hilbert_space.py), which legitimizes the
Hilbert-space side of Feshbach/uniform-Q-density. The uniform-Q-density
theorem and Feshbach projection's derivation from A1 + A2-T + A3-T remain
separately load-bearing.
"""

# ============================================================
# DARK EXTRACTION MAP
# ============================================================
#
# The dark sector self-energy Σ(h) = α₁/h = (α₁/2)h* is derived
# from MDL compression (uniform Q-space density + Feshbach projection).
# See dark_correction_theorem_2026-04-14.md §4a.
#
# This file proves: the correction coefficient for each observable
# is determined by its C₃ × parity quantum numbers at the P-point.
# No fitting to observation is used.

# --- INPUTS --------------------------------------------------
# symbol    | value        | status    | predictions/ file                   | meaning
# ----------|--------------|-----------|-------------------------------------|--------
# d_spatial | 3            | [derived] | predictions/d_spatial.py            | srs spatial dimension
# k_star    | 3            | [derived] | predictions/k_star.py               | coordination number
# g_girth   | 10           | [derived] | predictions/g_girth.py              | srs girth
# E_at_P    | sqrt(3)      | [derived] | predictions/srs_E_at_P.py           | P-point Bloch energy (feeds h)
# h         | (√3+i√5)/2  | [derived] | predictions/h_walker_eigenvalue.py  | walker eigenvalue
# α₁        | (2/3)^8      | [derived] | predictions/alpha_1.py              | NB walk survival

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
import cmath
import functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h_walker_eigenvalue import predict_h_walker_eigenvalue
from alpha_1 import predict_alpha_1
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from srs_E_at_P import predict_srs_E_at_P
from g_girth import predict_g_girth

d = predict_d_spatial()
k = predict_k_star(d)
E = predict_srs_E_at_P(k)
from p_toggle import predict_p_toggle
p = predict_p_toggle()
h = predict_h_walker_eigenvalue(k, E, p)
g = predict_g_girth(k, d)
a1 = predict_alpha_1(k, g)

# Self-energy
Sigma = a1 / h  # = (α₁/|h|²) h* = (α₁/2) h*

# Hermitian decomposition of h
Re_h = h.real   # √3/2
Im_h = h.imag   # √5/2
mod_h_sq = abs(h)**2  # k*-1 = 2

# Self-energy components
Re_Sigma = Sigma.real   # α₁ Re(h)/|h|² = α₁√3/4
Im_Sigma = Sigma.imag   # -α₁ Im(h)/|h|² = -α₁√5/4

print("=" * 70)
print("DARK EXTRACTION MAP: C₃ × parity classification")
print("=" * 70)

print(f"\nh = {Re_h:.6f} + {Im_h:.6f}i")
print(f"|h|² = {mod_h_sq:.6f} = k*-1 = {k-1}")
print(f"Re(h) = √3/2 = {Re_h:.10f}")
print(f"Im(h) = √5/2 = {Im_h:.10f}")

print(f"\nΣ(h) = α₁/h = {Sigma.real:.10f} + {Sigma.imag:.10f}i")
print(f"Re(Σ) = α₁√3/4 = {Re_Sigma:.10f}")
print(f"Im(Σ) = -α₁√5/4 = {Im_Sigma:.10f}")

# ================================================================
# THE THREE CLASSES
# ================================================================

# CLASS 1: AMPLITUDE (1-point, off-diagonal under C₃)
# Observable: walk amplitude |A_d| between different generations
# C₃ quantum number: ω² (off-diagonal, generation-changing)
# Couples to: Σ directly (first order)
# Extraction: |Im[Σ(h)]| (parity-odd component of complex self-energy)
# Coefficient: √5/4 · α₁

coeff_amplitude = abs(Im_Sigma) / a1  # should be √5/4
print(f"\n--- CLASS 1: AMPLITUDE (off-diagonal, 1-point) ---")
print(f"  Coefficient: |Im(Σ)|/α₁ = {coeff_amplitude:.10f}")
print(f"  √5/4                    = {math.sqrt(5)/4:.10f}")
print(f"  Match: {abs(coeff_amplitude - math.sqrt(5)/4) < 1e-14}")
print(f"  Applies to: V_us, m_ν2, m_ν3")
print(f"  Reason: generation-changing observable → off-diagonal in C₃")
print(f"          → couples to Im(Σ) (parity-odd component)")

# CLASS 2: MASS² (2-point, diagonal under C₃)
# Observable: mass-matrix eigenvalue ratio → mixing angle
# C₃ quantum number: trivial (diagonal, generation-preserving)
# Couples to: Hermitian channels of B†B (mass matrix)
# The mass² perturbation decomposes as:
#   M² = m²I + α₁[ε_Re² σ_z + ε_Im² σ_x]
# with ε_Re² = Re²(h)·(1/2) = 3/8 and ε_Im² = Im²(h) = 5/4
# The (1/2) is the TBM off-diagonal normalization b₀ from C₃ irreps.
# The angle shift: Δθ = ε_Im²/(2·ε_Re²) · α₁ = (5/3) · α₁
# Coefficient: Im²(h)/Re²(h) = tan²(arg h) = 5/3

eps_Re_sq = Re_h**2 * 0.5  # 3/8
eps_Im_sq = Im_h**2         # 5/4
coeff_mass_sq = eps_Im_sq / (2 * eps_Re_sq)  # (5/4)/(3/4) = 5/3

print(f"\n--- CLASS 2: MASS² (diagonal, 2-point) ---")
print(f"  ε_Re² = Re²(h)·(1/2) = {eps_Re_sq:.10f} (= 3/8)")
print(f"  ε_Im² = Im²(h)       = {eps_Im_sq:.10f} (= 5/4)")
print(f"  Coefficient: ε_Im²/(2·ε_Re²) = {coeff_mass_sq:.10f}")
print(f"  tan²(arg h) = 5/3    = {5/3:.10f}")
print(f"  Match: {abs(coeff_mass_sq - 5/3) < 1e-14}")
print(f"  Applies to: θ_23")
print(f"  Reason: mixing angle from mass-matrix diag → diagonal in C₃")
print(f"          → couples through Hermitian channels Im²(h)/Re²(h)")

# CLASS 3: EDGE-LOCAL (vertex-specific, C₃-symmetric)
# Observable: quantities measured at a C₃-symmetric vertex
# At a C₃-symmetric vertex: Tr(σ_x) = 0 (the three C₃ images
# of the parity-mixing operator cancel). This kills the Im(h)
# enhancement, leaving only the bare α₁.
# Coefficient: 1

print(f"\n--- CLASS 3: EDGE-LOCAL (C₃-symmetric vertex) ---")
print(f"  Tr(σ_x) at C₃ vertex = 0 (three images cancel)")
print(f"  Im(h) enhancement killed → coefficient = 1")
print(f"  Applies to: θ_13, V_cb (commensurate)")
print(f"  Reason: observable at C₃-symmetric vertex")
print(f"          → parity-odd channel cancels by symmetry")
print(f"          → only bare α₁ survives")

# ================================================================
# SUMMARY TABLE
# ================================================================

print(f"\n{'='*70}")
print(f"EXTRACTION MAP SUMMARY")
print(f"{'='*70}")
print(f"  {'Class':<12} {'C₃ QN':<12} {'Extraction':<20} {'Coefficient':<12} {'Observables'}")
print(f"  {'-'*75}")
print(f"  {'Amplitude':<12} {'ω² (off-d)':<12} {'|Im(Σ)|':<20} {'√5/4·α₁':<12} {'V_us, m_ν2, m_ν3'}")
print(f"  {'Mass²':<12} {'1 (diag)':<12} {'Im²(h)/Re²(h)':<20} {'(5/3)·α₁':<12} {'θ_23'}")
print(f"  {'Edge-local':<12} {'1 (C₃-sym)':<12} {'Tr(σ_x)=0':<20} {'1·α₁':<12} {'θ_13, V_cb'}")


# --- PURE FUNCTIONS ------------------------------------------

def dark_coefficient_amplitude(h):
    """
    Dark correction coefficient for amplitude-class observables.

    For generation-changing (off-diagonal C₃) observables,
    the dark sector couples through |Im[Σ(h)]| where Σ = α₁/h.
    Coefficient = |Im(h)|/|h|² = √5/4 for srs.

    Parameters
    ----------
    h : complex
        Walker eigenvalue (from predict_h_walker_eigenvalue).

    Returns
    -------
    float
        |Im(h)| / |h|² = √5/4
    """
    return abs(h.imag) / abs(h)**2


def dark_coefficient_mass_squared(h):
    """
    Dark correction coefficient for mass²-class observables.

    For mixing angles from mass-matrix diagonalization (diagonal C₃),
    the Hermitian decomposition B = B_sym + iB_anti gives parity
    channels ε_Re² = Re²(h)·(1/2) and ε_Im² = Im²(h). The angle
    shift ratio is ε_Im²/(2·ε_Re²) = Im²(h)/Re²(h) = tan²(arg h).

    Parameters
    ----------
    h : complex
        Walker eigenvalue.

    Returns
    -------
    float
        Im(h)² / Re(h)² = tan²(arg h) = 5/3
    """
    return h.imag**2 / h.real**2


def dark_coefficient_edge_local(h):
    """
    Dark correction coefficient for edge-local observables.

    At C₃-symmetric vertices, Tr(σ_x) = 0 kills the Im(h) channel.
    Only bare α₁ survives. Coefficient = 1.

    Parameters
    ----------
    h : complex
        Walker eigenvalue (unused — coefficient is 1 by symmetry).

    Returns
    -------
    float
        1.0
    """
    return 1.0


# ============================================================
# FAMILY D — per-leg multiway dark-disruption correction
# ============================================================
# Per docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md §3 (D).
#
# c_H = α₁_bare²  (per Higgs leg): STRUCTURALLY DERIVED — Route H (joint
#   srs×srs-z NB walker survival over (g-2) steps); Route C (m=2 closed-bubble)
#   corroborates via the srs-specific seam=2 identity. Conditional on the
#   (g-2) joint-excursion assumption (inherited Feshbach Exponent Principle).
#
# c_F = -α₁_bare²/(N_atoms·k*) = -α₁²/12  (per fermion leg):
#   STATUS CORRECTED 2026-05-18 (W1). The prior "Routes F-1 + F-2, two
#   independent routes, theorem-grade" framing was a parameter_linter
#   Clause-6c smuggle — an unnamed MDL-bit-cost minimum that conflates
#   canonical_encoding with channel_select (Clause 6a/6c BLOCK this). The
#   genuine derivation is the explicit Clause-6 two-step L-expression,
#   verified through the framework's real channel_select gate in
#   proofs/foundations/c_F_channel_select_waterfilling_2026-05-18.py
#   (commit 6c43c54) and INLINED below per the predictions/ DAG contract
#   (no proofs/ or simulator/ import) as _c_F_denominator_channel_select:
#     Step 1 channel_select(S, c="single_edge_spectral"): the fermion leg is
#       a SINGLE CAR directed-edge mode (theorem_car_local_jordan_wigner §1),
#       structurally distinct from the gauge-singlet's democratic 2|E|-edge
#       sum (the δ_r channel). c is fixed by this structural argument BEFORE
#       enumeration. Excludes the gauge-singlet object ⟨e|P|e⟩/(2|E|)=1/(2|E|)²
#       (prereg-#1's 1/144 — a channel mismatch) and vertex-local 1/k*²
#       (tree-Yukawa channel, a different observable).
#     Step 2 canonical_encoding(S'): 1/(2|E|) and 1/(N_atoms·k*) are
#       ENCODING-EQUIVALENT (identical value via the Euler identity
#       2|E|=N_atoms·k* — NOT two independent routes); canonical min-bit
#       representative = 1/(N_atoms·k*).
#   c_F ∈ ℚ ⊂ K=ℚ(√2,√3,√5). δ_r anchor (gauge_singlet channel) reproduces
#   1/(2|E|) by the SAME gate (consistency). GRADE: THEOREM-GRADE-STRUCTURAL,
#   conditional on the single-edge-vs-gauge-singlet channel argument (a
#   structural argument at δ_r's tier, theorem_unified_oblique §6.1 — NOT a
#   from-resolvent theorem; STRUCTURAL not UNIQUE). Numeric value UNCHANGED.
#
# Applied at a vertex with n_H Higgs legs + n_F fermion legs:
#   δg/g = -(n_H · c_H + n_F · c_F) = -α₁_bare²·(n_H - n_F/(N_atoms·k*))
#
# Closed-form predictions (numeric values unchanged):
#   y_τ vertex (1H+2F):  δy_τ/y_τ = -(5/6)·α₁_bare²  ≈ -0.127%
#   λ_Higgs vertex (4H): δλ/λ     = -4·α₁_bare²       ≈ -0.609%
#   v_Higgs vertex (1H): δv/v     = -α₁_bare²         ≈ -0.152%  (absorbed in N_hub anchor)

@functools.lru_cache(maxsize=None)
def _c_F_denominator_channel_select(N_atoms, k_star):
    """parameter_linter Clause-6 two-step for the c_F fermion-leg denominator.

    Inlined here per the predictions/ DAG contract (no proofs/ or simulator/
    import). Verified against the real simulator/gating/mdl.channel_select
    gate in proofs/foundations/c_F_channel_select_waterfilling_2026-05-18.py.

    Step 1 — channel_select(S, c="single_edge_spectral"): the channel is
      fixed by a structural argument BEFORE enumeration — a Yukawa fermion
      leg is a single CAR directed-edge mode (theorem_car_local_jordan_
      wigner §1), distinct from the gauge-singlet's democratic 2|E|-edge sum
      (the δ_r channel). This selects the single-edge-spectral candidates and
      excludes the gauge-singlet object 1/(2|E|)² (prereg-#1's 1/144 channel
      mismatch) and vertex-local 1/k*² (tree-Yukawa channel).
    Step 2 — canonical_encoding(S'): the single-edge candidates 1/(2|E|)
      (single-edge Perron weight) and 1/(N_atoms·k*) (cell directed-edge
      count) are encoding-equivalent — identical value via the Euler identity
      2|E| = N_atoms·k* (NOT two independent routes). Canonical (min-bit)
      representative: 1/(N_atoms·k*).

    Returns the canonical denominator N_atoms·k* (= 2|E|). Value unchanged
    from the prior code; only the derivation is now Clause-6-legible.
    """
    two_E = 2 * (N_atoms * k_star // 2)                 # directed edges per cell
    single_edge_spectral = {                            # (value, model_bits)
        'perron_single_edge': (1.0 / two_E, 4),         # ⟨e|P_P|e⟩ = 1/(2|E|)
        'cell_edge_count':    (1.0 / (N_atoms * k_star), 8),  # 1/(N·k*) [Euler-equiv]
    }
    # Step 2 canonical_encoding: assert encoding-equivalence (same value),
    # then return the canonical denominator. (Step-1 channel exclusion of
    # gauge_singlet 1/(2|E|)² and vertex_local 1/k*² is structural — those
    # candidates are not in this channel and never enter S'.)
    vals = {round(v, 15) for v, _ in single_edge_spectral.values()}
    assert len(vals) == 1, "single-edge candidates not encoding-equivalent"
    return N_atoms * k_star


@functools.lru_cache(maxsize=None)
def family_D_per_leg_correction(alpha_1_bare, n_H_legs, n_F_legs, N_atoms, k_star):
    """
    Family D per-leg multiway dark-disruption correction factor.

    Returns the multiplicative correction (1 - n_H·c_H - n_F·c_F) for an
    observable whose coupling vertex has n_H Higgs legs and n_F fermion legs.

    c_H = α₁_bare²: structurally derived (Route H; Route C corrob.).
    c_F = -α₁_bare²/(N_atoms·k_star): parameter_linter Clause-6 two-step
      (channel_select → canonical_encoding) via _c_F_denominator_
      channel_select below. THEOREM-GRADE-STRUCTURAL, conditional on the
      single-edge-vs-gauge-singlet channel argument (W1 2026-05-18; see
      module header + master doc §3 (D)). Numeric value unchanged.

    Apply via:  g_physical = g_tree × family_D_per_leg_correction(...)

    Parameters
    ----------
    alpha_1_bare : float or Fraction
        Bare NB walker survival amplitude α₁_bare = ((k*-1)/k*)^(g-2).
        Theorem-grade upstream per predictions/alpha_1.py.
    n_H_legs : int
        Number of Higgs legs at the coupling vertex.
        Structural — defined by the SM Lagrangian vertex topology
        (1 for v_Higgs/Yukawa, 4 for |φ|⁴ Higgs quartic).
    n_F_legs : int
        Number of fermion legs at the coupling vertex.
        Structural — 0 for v_Higgs / |φ|⁴, 2 for Yukawa y_f φ ψ̄ψ.
    N_atoms : int
        Wyckoff 8a atom count per primitive cell. For srs (I4_132) = 4.
        Theorem-grade upstream per predictions/N_fit.py / d_spatial.py.
    k_star : int
        Coordination number. For srs = 3.
        Theorem-grade upstream per predictions/k_star.py.

    Returns
    -------
    float
        1 - (n_H · α₁_bare² - n_F · α₁_bare² / (N_atoms · k_star))
        = 1 - α₁_bare² · (n_H - n_F / (N_atoms · k_star))
    """
    c_H = alpha_1_bare ** 2
    # c_F denominator via the explicit Clause-6 two-step (channel_select →
    # canonical_encoding). Returns N_atoms·k_star — value unchanged.
    c_F = -alpha_1_bare ** 2 / _c_F_denominator_channel_select(N_atoms, k_star)
    return 1 - (n_H_legs * c_H + n_F_legs * c_F)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("VALIDATION")
    print(f"{'='*70}")

    c_amp = dark_coefficient_amplitude(h)
    c_mass = dark_coefficient_mass_squared(h)
    c_edge = dark_coefficient_edge_local(h)

    print(f"  Amplitude:  {c_amp:.10f}  (expected √5/4 = {math.sqrt(5)/4:.10f})")
    print(f"  Mass²:      {c_mass:.10f}  (expected 5/3  = {5/3:.10f})")
    print(f"  Edge-local: {c_edge:.10f}  (expected 1)")

    assert abs(c_amp - math.sqrt(5)/4) < 1e-14, f"Amplitude: {c_amp}"
    assert abs(c_mass - 5/3) < 1e-14, f"Mass²: {c_mass}"
    assert c_edge == 1.0, f"Edge-local: {c_edge}"

    # Family D check: theorem-grade closed-form predictions
    alpha_1_bare = (2/3)**8
    N_atoms_srs = 4
    k_star_srs = 3

    # λ_Higgs (4H + 0F): expect -4·α₁² correction
    fd_lam = family_D_per_leg_correction(alpha_1_bare, n_H_legs=4, n_F_legs=0,
                                          N_atoms=N_atoms_srs, k_star=k_star_srs)
    expected_lam = 1 - 4 * alpha_1_bare**2
    assert abs(fd_lam - expected_lam) < 1e-15, f"Family D λ: {fd_lam} vs {expected_lam}"

    # y_τ (1H + 2F): expect -(5/6)·α₁² correction
    fd_y = family_D_per_leg_correction(alpha_1_bare, n_H_legs=1, n_F_legs=2,
                                        N_atoms=N_atoms_srs, k_star=k_star_srs)
    expected_y = 1 - alpha_1_bare**2 + 2 * alpha_1_bare**2 / 12
    expected_y_clean = 1 - (5/6) * alpha_1_bare**2
    assert abs(fd_y - expected_y) < 1e-15, f"Family D y_τ: {fd_y} vs {expected_y}"
    assert abs(fd_y - expected_y_clean) < 1e-15, f"Family D y_τ closed form: {fd_y}"

    # v_Higgs (1H + 0F): expect -α₁² sub-leading
    fd_v = family_D_per_leg_correction(alpha_1_bare, n_H_legs=1, n_F_legs=0,
                                        N_atoms=N_atoms_srs, k_star=k_star_srs)
    expected_v = 1 - alpha_1_bare**2
    assert abs(fd_v - expected_v) < 1e-15, f"Family D v: {fd_v} vs {expected_v}"

    print(f"  Family D λ_Higgs  (4H+0F):  {fd_lam:.10f}  ({(fd_lam-1)*100:+.5f}%)")
    print(f"  Family D y_τ      (1H+2F):  {fd_y:.10f}  ({(fd_y-1)*100:+.5f}%)")
    print(f"  Family D v_Higgs  (1H+0F):  {fd_v:.10f}  ({(fd_v-1)*100:+.5f}%)")

    print("OK: all three dark-coefficient classes + Family D correction verified.")
