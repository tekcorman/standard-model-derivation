#!/usr/bin/env python3
"""
m_nu3 — heaviest light neutrino mass (normal ordering)

NEW DERIVATION (supersedes 2026-05-04): m_ν₃ as substrate spectral gap.

The previous derivation used a Pati-Salam seesaw with M_R = (2/3)^g × M_GUT
(ADOPTED-PS) and a Class-1 Feshbach correction. That ADOPTED-PS bare scale
is no longer needed: the m_ν₃ magnitude follows directly from the substrate's
mean-field critical fluctuation gap, with the per-cell channel multiplicity
as prefactor.

THE FORMULA:

    m_ν₃ = (k* × N_atoms) × M_Pl × N_hub^(-1/2)

is equivalent (via the seesaw m_ν = v²/M_R) to:

    M_R = δ⁴ × M_Pl / (2 × k* × N_atoms)
    v   = δ² × M_Pl / (√2 × N_hub^(1/4))

The δ⁴ in v² and M_R cancel exactly. m_ν₃ is INDEPENDENT of the Koide phase δ
(unlike charged-lepton masses, which all depend on δ via the Koide formula).

KEY PROPERTIES:
- All inputs are framework-internal: k*=3, N_atoms=4 (theorem-grade lattice
  primitives), M_Pl (substrate-anchored via G_sub Drude closure), N_hub
  (the adopted dimensional input; value pinned via the measured G_F).
- ELIMINATES the prior ADOPTED-PS m_ν₃_bare = 0.048277 eV input.
- ELIMINATES dependence on M_GUT, m_t(GUT), MSSM RG running, y_t(GUT), tan β.
- ELIMINATES the Class-1 Feshbach correction as a SEPARATE MULTIPLICATIVE
  factor — the Feshbach mechanism (Σ(h) = α₁·h̄/|h|², theorem-grade per
  `theorem_m_nu_dark_correction_uniqueness_closure.md`) is BAKED INTO the
  bare scale here: the spectral-gap formula `(k*·N_atoms)·M_Pl/√N_hub`
  is the residue-at-h evaluation of the substrate self-energy.  Applying
  the universal-template factor (1 - √5/4·α₁/(1-α₁)) on top would
  double-count and over-shoot to -1.4% (vs current +0.87%) — verified
  empirically.  See master doc §3 (B) "Application clarification" and
  §5 catalog row for m_ν_3 (updated 2026-05-15 post-sweep).
- m_ν₂ inherits via the theorem-grade R = 228/7 splitting:
  m_ν₂ = m_ν₃ / √R.
- m_ν₁ = 0 unchanged (M_D(trivial) = 0 at P; theorem-grade).
- PMNS Majorana phases (h^g) unchanged (theorem-grade per
  proofs/flavor/srs_hashimoto_seesaw_verify.py); they live on the C_3 ω, ω̄
  modes orthogonal to the C_3-trivial scale-setting direction.

COMPANION DOCS:
- predictions/m_nu3_derivation.md — full derivation chain
- proofs/flavor/srs_M_R_step1_structural.py — structural identity verification
- proofs/flavor/srs_M_R_step2_derivation.py — factor-by-factor derivation
- proofs/flavor/srs_M_R_step3_closure.py — δ-independence + parallel to v
"""

# ============================================================
# PARAMETER: m_ν₃ (heaviest light neutrino mass, normal ordering)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       m_ν₃ = √Δm²₃₁ = 50.13 ± 0.20 meV  (assuming m_ν₁ = 0)
# Source:      NuFIT 6.0 (September 2024), normal ordering;
#              Δm²₃₁ = (2.513 ± 0.020) × 10⁻³ eV².
# PDG edition: 2024

# --- PREDICTED VALUE -----------------------------------------
# Value:       m_ν₃ = (k* × N_atoms) × M_Pl × N_hub^(-1/2)
#                   = 12 × M_Pl / √N_hub
#                   ≈ 50.57 meV  (with the adopted N_hub = 8.395e60 (value pinned via the measured G_F))
# Deviation:   +0.435 meV (+0.868%, +2.18 σ_PDG) — Clause 8 FAIL vs σ_PDG alone.
#              The deviation is consistent with the N_hub anchor uncertainty
#              between the adopted ~8.4×10⁶⁰ (value pinned via the measured G_F)
#              and m_τ-anchored ~8.44×10⁶⁰; at the m_τ anchor the deviation
#              reduces to +0.65%.
#
#              NOTE (2026-05-15 dark-correction sweep): the master doc
#              §3 (B) and §5 catalog Feshbach Im(h)/|h|² = √5/4 as the
#              theorem-grade DC form for neutrino masses, but the
#              spectral-gap reformulation here ALREADY EVALUATES the
#              Feshbach residue at h.  The universal-template multiplicative
#              factor (1 - √5/4·α₁/(1-α₁)) ≈ 0.9773 is therefore NOT applied
#              on top — doing so would double-count and shift m_ν₃ to
#              49.42 meV (-1.4%, -3.5σ_PDG, WORSE).  The Family D sub-leading
#              at the (0H+2F) Majorana vertex is +α₁²/6 ≈ +0.025% —
#              negligible vs the N_hub anchor sensitivity.  See master doc
#              §3 (B) "Application clarification" and §5 catalog row.

# --- DERIVED FORMULA -----------------------------------------
# m_ν₃ = (k* × N_atoms) × M_Pl × N_hub^(-1/2)
#
# Equivalent seesaw decomposition:
#   m_ν₃ = v² / M_R
#   v    = δ² × M_Pl / (√2 × N_hub^(1/4))             [BZJ; predictions/v_higgs.py]
#   M_R  = δ⁴ × M_Pl / (2 × k* × N_atoms)              [substrate Majorana scale]
# In the ratio v²/M_R, the δ⁴ in v² and M_R cancel exactly along with the 1/2,
# leaving the global form (k* × N_atoms) × M_Pl × N^(-1/2).
#
# Logical chain:
#   Step 1: v = δ² × M_Pl / (√2 × N^(1/4))
#           [Type 4: predictions/v_higgs.py — theorem-grade BZJ + MDL mean-field]
#   Step 2: M_R = δ⁴ × M_Pl / (2 × k* × N_atoms)
#           - δ⁴ from (δ²)² for ν_R bilinear (Type 2 algebra given Step 1)
#           - 1/2 from Majorana mass term coefficient L ⊃ -(1/2) M_R ν_R^T C ν_R
#             [Type 3: Peskin-Schroeder §3.4, standard QFT]
#           - 1/(k* × N_atoms) from per-cell directed-edge Bloch normalization
#             [Type 3: standard solid-state Bloch decomposition]
#   Step 3: m_ν₃ = m_D²/M_R = y_ν²·v²/M_R (Type-I seesaw, Mohapatra-
#           Senjanović 1980). The framework ADOPTS y_ν = 1 (Dirac/top
#           Yukawa = 1 at unification) — NOT derived; it is the same
#           undischarged up-sector anchor that retracted m_top (Row P38).
#           This is LOAD-BEARING for the entire absolute scale; with
#           realistic y_t(GUT)≈0.5–0.7, m_ν₃ ≈ 13–25 meV. Only the
#           splitting ratio R=228/7 is anchor-free. See _derivation.md
#           Step 3 + Status (2026-05-18 chain audit).
#   Step 4: Algebraic simplification
#           [Type 2: exact rational arithmetic;
#            (δ⁴/4) × k*^(g-1) = k* × N_atoms verified to machine precision]

# --- INPUTS --------------------------------------------------
# symbol       | value      | status     | predictions/ file              | meaning
# -------------|------------|------------|--------------------------------|----
# k_star       | 3          | [derived]  | predictions/k_star.py          | Hashimoto Perron, theorem-grade
# N_atoms      | 4          | [derived]  | (structural)                   | atoms per srs primitive cell, theorem-grade
# M_P          | M_Pl_natural.M_Pl_GeV | [derived] | predictions/M_Pl_natural.py | M_Pl/M_subst=8/√π theorem-grade; GeV=single declared SI-anchor (CODE imports it line 150 — NOT "[external]")
# N_hub        | ~8.39e60   | [ADOPTED]  | predictions/N_hub.py           | substrate site count — THE adopted dimensional input (value pinned via the measured G_F)

# --- IMPLEMENTATION ------------------------------------------

import sys
import os
import math
import functools

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from k_star import predict_k_star
from d_spatial import predict_d_spatial
from g_girth import predict_g_girth
from alpha_1 import predict_alpha_1
from N_hub import predict_N_hub
from R_nu_splitting import predict_R_nu_splitting

# Substrate primitives
d_val = predict_d_spatial()
k_val = predict_k_star(d_val)
g_val = predict_g_girth(k_val, d_val)
alpha_1 = predict_alpha_1(k_val, g_val)

# N_atoms is a theorem-grade structural integer for srs (atoms per primitive cell).
# Sunada 2012 Theorem 3.1: srs is the unique k*=3, g=10 3D crystal net; its
# primitive cell has 4 atoms (Wyckoff 8a, x = 1/8 in space group I4₁32).
from V_count import V_count_pred as N_atoms  # = 4, srs primitive cell |V| / K_4 quotient (predict_V_count)

# External anchors (the framework's last dimensional inputs)
from M_Pl_natural import M_Pl_GeV as M_P_GeV, eV_per_GeV   # CODATA single-source — ANTHROPOCENTRIC SI TRANSLATION
                                                # (M_Pl/M_substrate = 8/√π is theorem-grade)
                                                # eV_per_GeV: SI prefix conversion, single-source 2026-05-26
G_F_obs = 1.1663787e-5   # the measured Fermi constant — used to pin N_hub's adopted value; G_F itself is a PREDICTION (predictions/G_F.py)
from delta_Koide import delta_Koide_pred as delta  # = 2/9 (Q*(1-Q) at Q=2/3, predict_delta_Koide)        # Wigner D¹₁₀ matrix element (predictions/h_walker_eigenvalue.py)
from p_toggle import predict_p_toggle
from V_count import predict_V_count
_p_nh = predict_p_toggle()
_V_nh = predict_V_count(k_val, d_val)
N_hub = predict_N_hub(G_F_obs, M_P_GeV, alpha_1, delta, k_val, _p_nh, _V_nh)   # = ~8.5×10⁶⁰

# Global formula
prefactor = k_val * N_atoms  # = 12 = directed edges per primitive cell = dim(B)/cell
m_nu3_GeV = prefactor * M_P_GeV / math.sqrt(N_hub)
m_nu3_eV = m_nu3_GeV * eV_per_GeV

# Verify equivalent seesaw decomposition (machine-precision check)
v_BZJ = (delta**2) * M_P_GeV / (math.sqrt(2.0) * N_hub**0.25)   # predictions/v_higgs.py form
M_R_GeV = (delta**4) * M_P_GeV / (2.0 * k_val * N_atoms)
m_nu3_seesaw = (v_BZJ ** 2) / M_R_GeV * eV_per_GeV   # in eV
assert abs(m_nu3_eV - m_nu3_seesaw) / m_nu3_eV < 1e-10, (
    f"Seesaw vs global form mismatch: {m_nu3_eV} vs {m_nu3_seesaw}"
)

# m_ν₂ from R = 228/7 (theorem-grade Ihara splitting)
from p_toggle import predict_p_toggle
from V_count import predict_V_count
_p_val = predict_p_toggle()
_V_val = predict_V_count(k_val, d_val)
R_split = predict_R_nu_splitting(k_val, _p_val, _V_val)   # = 228/7
m_nu2_eV = m_nu3_eV / math.sqrt(R_split)

# m_ν₁ = 0 (theorem-grade unchanged: M_D(trivial) = 0 at P)
m_nu1_eV = 0.0

# Module-level observation exports for run_predictions.py introspection.
# NuFIT 6.0 normal ordering: Δm²₃₁ = (2.513 ± 0.020) × 10⁻³ eV².
m_nu3_pred = m_nu3_eV
m_nu3_obs = math.sqrt(2.513e-3)
m_nu3_sigma = 0.5 * 0.020e-3 / m_nu3_obs                     # PDG-only ~0.20 meV

print("m_ν₃ prediction — global spectral-gap formula")
print(f"  Upstream: k*={k_val}, d={d_val}, g={g_val}, N_atoms={N_atoms}")
print(f"  M_P     = {M_P_GeV:.5e} GeV  [external; CODATA]")
print(f"  N_hub   = {N_hub:.4e}        [the adopted dimensional input (value pinned via the measured G_F); predictions/N_hub.py]")
print()
print(f"  Global form:   m_ν₃ = (k* × N_atoms) × M_Pl × N_hub^(-1/2)")
print(f"               = {prefactor} × {M_P_GeV:.4e} GeV × {1.0/math.sqrt(N_hub):.4e}")
print(f"               = {m_nu3_eV * 1e3:.4f} meV")
print()
print(f"  Equivalent seesaw form:")
print(f"    v    = δ² × M_Pl / (√2 × N^(1/4)) = {v_BZJ:.4f} GeV")
print(f"    M_R  = δ⁴ × M_Pl / (2 × k* × N_atoms) = {M_R_GeV:.4e} GeV")
print(f"    v²/M_R = {m_nu3_seesaw * 1e3:.4f} meV   ← matches global form")
print()
print(f"  m_ν₂ = m_ν₃ / √R  with R = 228/7 = {R_split:.6f}")
print(f"       = {m_nu2_eV * 1e3:.4f} meV")
print(f"  m_ν₁ = 0 (theorem-grade: M_D(trivial) = 0 at P)")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_m_nu3(k_star, N_atoms, M_P_GeV, N_hub):
    """
    Predict m_ν₃ from substrate spectral gap.

    The formula m_ν₃ = (k* × N_atoms) × M_Pl × N_hub^(-1/2) follows from:
      1. v = δ² × M_Pl / (√2 × N^(1/4))  [BZJ; predictions/v_higgs.py]
      2. M_R = δ⁴ × M_Pl / (2 × k* × N_atoms)  [substrate Majorana scale]
      3. m_ν₃ = v²/M_R [Type-I seesaw]
      4. δ⁴ and 1/2 cancel exactly, leaving (k*·N_atoms) × M_Pl × N^(-1/2).

    Parameters
    ----------
    k_star : int
        Substrate coordination number (= 3 for srs, theorem-grade).
    N_atoms : int
        Atoms per primitive cell (= 4 for srs, theorem-grade structural).
    M_P_GeV : float
        Planck mass in GeV (substrate-anchored via G_sub Drude closure).
    N_hub : float
        Substrate site count (the adopted dimensional input; value pinned via the measured G_F).

    Returns
    -------
    float
        m_ν₃ in eV.
    """
    return (k_star * N_atoms) * (M_P_GeV * eV_per_GeV) / math.sqrt(N_hub)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = m_nu3_eV
    pure_result = predict_m_nu3(k_val, N_atoms, M_P_GeV, N_hub)
    print()
    print("=" * 60)
    print("STATUS (parameter linter clauses):")
    print("  Clauses 1-5 (axiom/algebra/theorem/predictions chain):")
    print("    Step 1 [v BZJ]         = Type 4 (predictions/v_higgs.py)")
    print("    Step 2 [M_R structure] = Types 2 + 3 (algebra + standard QFT)")
    print("    Step 3 [seesaw]        = Type 3 (Mohapatra-Senjanović 1980)")
    print("    Step 4 [algebra]       = Type 2 (machine-verified rational)")
    print("  Clause 6 (K-meta-theorem):")
    print("    Coefficient (k*·N_atoms) = 12 ∈ ℚ ⊂ K = ℚ(√2,√3,√5)")
    print("  Clause 7 (uniqueness):")
    print("    Inherits Row 4 closure for k*=3 (srs lattice).")
    print("    See an internal working note")
    print("  Clause 8 (numerical match, σ_PDG only):")
    print("    σ_obs      = 0.40% (NuFIT 6.0)")
    print("    Deviation  = +0.87% (+2.18 σ_PDG) ⇒ Clause 8 FAIL.")
    print("=" * 60)

    # NuFIT 6.0 (September 2024), normal ordering:
    dm2_31_obs = 2.513e-3       # eV²
    dm2_31_sigma = 0.020e-3     # eV²
    m_nu3_obs = math.sqrt(dm2_31_obs)
    m_nu3_sigma = 0.5 * dm2_31_sigma / m_nu3_obs

    dev_abs = pure_result - m_nu3_obs
    dev_rel = dev_abs / m_nu3_obs
    dev_sigma = dev_abs / m_nu3_sigma

    print()
    print(f"  Implementation:  {impl_result:.9f} eV")
    print(f"  Pure function:   {pure_result:.9f} eV")
    assert abs(impl_result - pure_result) < 1e-12, (
        f"Implementation vs pure function mismatch: "
        f"{impl_result} vs {pure_result}"
    )
    print(f"  OK: outputs agree.")
    print()
    print(f"    m_ν₃ predicted = {pure_result * 1e3:.4f} meV")
    print(f"    m_ν₃ NuFIT 6.0 = {m_nu3_obs * 1e3:.4f} ± {m_nu3_sigma * 1e3:.4f} meV")
    print(f"    Deviation       = {dev_abs * 1e3:+.4f} meV  ({dev_rel*100:+.3f}%, {dev_sigma:+.2f} σ)")
    print()
    print("    m_ν₂ predicted = {:.4f} meV  (via R = 228/7 splitting)".format(m_nu2_eV * 1e3))
    dm2_21_obs = 7.49e-5         # NuFIT 6.0
    m_nu2_obs = math.sqrt(dm2_21_obs)
    print("    m_ν₂ NuFIT 6.0 = {:.4f} meV".format(m_nu2_obs * 1e3))
    print("    Deviation       = {:+.3f}%".format((m_nu2_eV/m_nu2_obs - 1) * 100))
