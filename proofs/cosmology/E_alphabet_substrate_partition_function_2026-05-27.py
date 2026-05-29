#!/usr/bin/env python3
"""
proofs/cosmology/E_alphabet_substrate_partition_function_2026-05-27.py

|E| = 6 as substrate-native phase-space normalization — derivation attempt.

CONTEXT
-------
Saha-π attack (`saha_pi_attack_K_rational_substitutes_2026-05-27.py`) found:
  |E| = 6 → T_recomb = 0.3236 eV  (within 0.17% of standard 2π reference)

This near-coincidence is suggestive but not closing per W58. To upgrade to
structural derivation, we need: substrate-native partition function for free
electrons that NATURALLY gives |E|^(3/2) in place of (2π)^(3/2).

This probe attempts the derivation directly from the framework's existing
multiway branch measure + Bloch-Hashimoto structure.

CANDIDATE DERIVATION
--------------------
Standard continuum: thermal phase-space measure is d³p/(2πħ)³ → factor (2π)^(3/2)
for thermal de Broglie length scale.

Substrate-native candidate: discrete sum over Bloch-Hashimoto modes in BZ at
thermal energy T. Per `theorem_bloch_lift_mu.md`:
  Z_μ(L) = |E|^(-L) · Tr(B(k)^L)

The |E|^(-L) factor IS the framework's substrate-native "phase-space cell" for
walks of length L. For a continuum thermal calculation, the analog would be:
  Z_thermal_substrate ~ Σ_modes exp(-E_mode/T) ~ count of accessible modes / |E|

What we test: does the substrate-native "phase-space cell" indeed give |E|^(3/2)
in the analog of the (2π)^(3/2) Saha prefactor?

Run with:
    python3 proofs/cosmology/E_alphabet_substrate_partition_function_2026-05-27.py
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Framework primitives
# ---------------------------------------------------------------------------
k_star = 3              # srs valence
N_atoms = 4             # primitive cell atoms
g_girth = 10            # srs girth
n_undirected_edges = N_atoms * k_star // 2   # = 6 (per primitive cell)
n_directed_edges = N_atoms * k_star          # = 12 (per primitive cell)


print("=" * 76)
print("|E| = 6 substrate-native phase-space derivation attempt")
print("=" * 76)
print()
print(f"Framework primitives:")
print(f"  k* = {k_star}, N_atoms = {N_atoms}, g = {g_girth}")
print(f"  |E|_undirected = N_atoms · k* / 2 = {n_undirected_edges}")
print(f"  |E|_directed   = N_atoms · k*     = {n_directed_edges}")
print()
print(f"  Numerical comparison:")
print(f"    2π                = {2 * math.pi:.6f}")
print(f"    |E|_undirected    = {n_undirected_edges}")
print(f"    |E|_directed      = {n_directed_edges}")
print(f"    Ratio (2π)/|E|_u  = {2 * math.pi / n_undirected_edges:.6f}")
print(f"    Ratio (2π)/|E|_d  = {2 * math.pi / n_directed_edges:.6f}")
print()


# ---------------------------------------------------------------------------
# Structural question: where does 2π come from in Saha prefactor?
# ---------------------------------------------------------------------------
print("=" * 76)
print("STRUCTURAL ORIGIN of 2π in Saha prefactor")
print("=" * 76)
print()
print("  Saha prefactor (m_e T / 2π ħ²)^(3/2)  =  V_thermal^(-1)")
print()
print("  where V_thermal = (2πħ² / m_e T)^(3/2) is the thermal de Broglie volume.")
print()
print("  The (2π)^(3/2) comes from the Gaussian momentum integral:")
print("    Z₁(T) = ∫ d³p / (2πħ)³ · exp(-p² / 2m_e T)")
print("          = (m_e T / 2πħ²)^(3/2) · V")
print()
print("  The 2π appears via h = 2π ħ (Planck's constant). It's the QUANTUM")
print("  PHASE-SPACE NORMALIZATION: each phase-space cell has volume h³ = (2π ħ)³.")
print()
print("  In a substrate-native picture, the phase-space cell is set by the")
print("  substrate's discrete structure, NOT by the continuum factor 2π.")
print()


# ---------------------------------------------------------------------------
# Substrate-native phase-space cell — derivation attempt
# ---------------------------------------------------------------------------
print("=" * 76)
print("Substrate-native phase-space cell")
print("=" * 76)
print()
print("Real-space side:")
print(f"  Primitive cell volume V_cell = (BCC primitive parallelepiped)")
print(f"  Number of vertices per cell = {N_atoms}")
print(f"  Real-space cell per vertex  = V_cell / {N_atoms}")
print()
print("Brillouin-zone side:")
print(f"  BZ volume = (2π/a)³ × 1/2 for BCC (in primitive cubic Cartesian)")
print(f"  Reduced BZ coords: k ∈ [0, 1]³ (the framework's convention)")
print(f"  Number of distinct k-points = number of primitive cells (large N)")
print()
print("Substrate phase-space cell (per primitive cell × per BZ k-mode):")
print(f"  Per cell, per k-point, the substrate has {N_atoms} × 8 = {N_atoms * 8} states")
print(f"  (4 vertices × 8-dim Cl(6) Fock = 32 fermion modes per cell)")
print()
print("Bloch-Hashimoto: 12 directed-edge modes per cell × {N_modes} cells = 12 N modes total")
print()
print("Connection to multiway μ:")
print("  μ(B) = |E|^(-L) on toggle sequences of length L (Theorem multiway_branch_measure)")
print("  For srs: |E|_undirected = 6")
print("  Each toggle sequence of length L weighted by 6^(-L)")
print()


# ---------------------------------------------------------------------------
# The K-rational substrate Saha prefactor candidate
# ---------------------------------------------------------------------------
print("=" * 76)
print("Candidate K-rational substrate Saha prefactor")
print("=" * 76)
print()
print("Hypothesis: in the substrate-native partition function for free electrons,")
print("the phase-space cell size is set by |E|^(3/2) instead of (2π)^(3/2).")
print()
print("Test 1 — does the framework's branch measure deliver |E| in the (3/2) power")
print("for a 3D substrate thermal partition function?")
print()
print("  For 3D continuum: thermal momentum integral ∫d³p/(2π)³·exp(-p²/2mT)")
print("                  = (mT/2π)^(3/2) — the 3/2 comes from 3 spatial dimensions")
print()
print("  For substrate: each spatial dimension contributes a factor |E_dim|^(-1)")
print("                 where |E_dim| is the per-axis edge count")
print()
print(f"  For srs: total |E| = {n_undirected_edges} per primitive cell, with d = 3 spatial dims")
print(f"           |E_per_dim| = |E|^(1/d) = {n_undirected_edges}^(1/3) = {n_undirected_edges ** (1/3):.6f}")
print()
print(f"  Substrate-native (3D, all dims): |E|^(3/3) = |E| = {n_undirected_edges}")
print(f"  Substrate-native (per dim^(3/2)): |E_per_dim|^(3) = {n_undirected_edges ** (3/3):.6f}")
print()
print("  Subtle: the (3/2) power in Saha comes from 3 dims × (1/2) from p² Gaussian.")
print("  The framework's substrate uses (k-1)/k per step = 2/3 for NB walker survival.")
print("  L-step amplitude: (2/3)^L. There's NO explicit (3/2) power; the substrate")
print("  is fundamentally discrete.")
print()


# ---------------------------------------------------------------------------
# Honest finding: the (3/2) power doesn't emerge naturally
# ---------------------------------------------------------------------------
print("=" * 76)
print("HONEST FINDING")
print("=" * 76)
print()
print("  The framework's substrate is fundamentally DISCRETE: walks of integer")
print("  length L with amplitude ((k-1)/k)^L = (2/3)^L.")
print()
print("  The continuum Saha prefactor (m_e T / 2π)^(3/2) has a continuous (3/2)")
print("  power that comes from the Gaussian momentum integral in 3D.")
print()
print("  The (3/2) power does NOT emerge naturally from the framework's discrete")
print("  multiway branch measure. The substrate-native thermal partition function")
print("  would have a different STRUCTURAL FORM (discrete sum over modes weighted")
print("  by Boltzmann factors), not a (m_e T)^(3/2) continuous-volume form.")
print()
print("  Therefore: the |E| = 6 vs 2π near-coincidence at 0.17% is a NUMERICAL")
print("  near-coincidence, NOT a structural identity. The substrate's |E| does")
print("  not naturally enter the (3/2) power of a Saha-like prefactor.")
print()
print("  This is consistent with the Saha-π attack's verdict: closing the gap")
print("  requires a SUBSTRATE-NATIVE partition function that doesn't look like")
print("  Saha at all — not a substitution of K-rational constants into the")
print("  existing Saha form.")
print()


# ---------------------------------------------------------------------------
# But — is there a structural reason |E| ≈ 2π that's not about Saha?
# ---------------------------------------------------------------------------
print("=" * 76)
print("BONUS: is |E| ≈ 2π a structural identity of a different kind?")
print("=" * 76)
print()
print("  |E|_undirected = 6, 2π = 6.283. Ratio 2π / |E| = 1.047.")
print()
print("  Possible structural readings:")
print()
print("  (i) Continuum limit: |E| is the discrete substrate analog of 2π")
print("      in the BZ measure dk/(2π). When the substrate is taken to its")
print("      continuum limit (a → 0, N_cells → ∞), the discrete measure")
print("      Σ_k → ∫dk/(2π) per BZ direction. The substrate's |E| factor")
print("      in μ(B) = |E|^(-L) plays the analog role.")
print()
print(f"      If |E|_undirected = 6 is the 3D-substrate version of (2π)^d for")
print(f"      d = 1: 2π ≈ 6.28 → matches |E| within 4.5%!")
print()
print("  (ii) NOT 3D-(2π)^3 ≈ 248 — that's a different power.")
print("       So |E| corresponds to the PER-DIMENSION 2π, not its cube.")
print("       But the Saha prefactor uses (2π)^(3/2), which is √((2π)^3).")
print("       And (2π)^(3/2) = 15.75 ≠ 6.")
print()
print("       So if |E| ≈ 2π is the per-dimension correspondence, then")
print("       the substrate analog of (2π)^(3/2) would be |E|^(3/2) = 14.70")
print(f"       (vs (2π)^(3/2) = {(2 * math.pi) ** 1.5:.4f}).")
print()
print(f"  Ratio comparison:")
print(f"    (2π)^(3/2)    = {(2 * math.pi) ** 1.5:.4f}")
print(f"    |E|^(3/2)     = {n_undirected_edges ** 1.5:.4f}")
print(f"    Ratio         = {(2 * math.pi) ** 1.5 / n_undirected_edges ** 1.5:.4f}")
print()
print("  The 7% gap between (2π)^(3/2) and |E|^(3/2) corresponds to T_recomb")
print("  shift of ~3% (per the log-suppression analysis from the Saha-π attack).")
print()
print("  This is consistent with what the Saha-π probe found.")
print()


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print("=" * 76)
print("INVESTIGATION VERDICT")
print("=" * 76)
print()
print("  |E| = 6 near-coincidence with 2π:")
print()
print("  PARTIAL STRUCTURAL READING (candidate-grade, NOT theorem-grade):")
print("    |E| plays a structural role analogous to 2π in the BZ measure dk/(2π).")
print("    The substrate's |E|^(-L) factor in μ(B) is the discrete analog of")
print("    the continuum (2π)^(-d) per BZ direction. For d = 1 per dimension,")
print("    |E| ≈ 2π is a STRUCTURAL CORRESPONDENCE (within 4.5%).")
print()
print("  WHAT IS NOT YET DERIVED:")
print("    - The Saha prefactor uses (3/2) power from a continuous integration")
print("      that has no direct substrate-native discrete analog.")
print("    - To extend |E| ↔ 2π to a Saha-prefactor substitution, would need")
print("      a substrate-native partition function with explicit (3/2) power")
print("      emerging from substrate combinatorics (e.g., 3 spatial dims ×")
print("      (1/2) from walker quadratic-in-momentum dispersion).")
print()
print("  HONEST DISPOSITION:")
print("    The |E| = 6 ≈ 2π near-coincidence is a CANDIDATE STRUCTURAL HINT")
print("    that the substrate's discrete alphabet plays the BZ-measure role.")
print("    Per W58, it cannot be claimed as 'the' Saha prefactor substitute")
print("    without a structural derivation of the (3/2) power.")
print()
print("    However, this near-coincidence is consistent with the framework's")
print("    Bloch-lift theorem identifying |E|^(-L) as the natural phase-space")
print("    weight. Future work on the substrate-native partition function may")
print("    formalize this correspondence.")
print()
print("  This investigation does NOT close the Saha-π gap. It identifies a")
print("  structural READING that could be explored in future multi-session")
print("  framework-extension work.")
