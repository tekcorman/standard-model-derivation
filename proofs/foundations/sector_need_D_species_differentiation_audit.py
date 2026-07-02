#!/usr/bin/env python3
"""
Need-D attack: audit-first scoping of species-differentiation mechanism.

CONTEXT
=======
Need-D is the framework's species-differentiation structural gap (per
an internal working note):

  "Charged leptons, up-quarks, down-quarks, neutrinos have wildly different
  mass scales. If M_gen is built from C³_gen alone, all species would have
  the same masses. Species-differentiation must enter through the Cl(6,0)
  spinor factor or the gauge rep factor."

In concrete CKM/PMNS context (per `delta_CP_CKM_geometry §6`):

  "The up-type and down-type Yukawa operators Y_u and Y_d share the same
  C_3 eigenbasis at tree level, giving U_u = U_d = I and CKM = I. Species-
  differentiation requires a derived mechanism (Cl(6,0) spinor factors +
  B-L + SU(2)_L × SU(2)_R)."

Need-D is the ACTIVE adoption ADOPTED-A5b-Sub3 (level 3 sub-class classifier),
whose graduation via bridge functoriality lemma was RETRACTED 2026-04-29
after three independent CAS refutations.

THE THREE REFUTATIONS (recap from `ckm_substrate_identification_2026-04-29.md`)
==============================================================================
- R1: Z_3 holonomy is flat on srs (`proofs/flavor/z3_holonomy_cycles.py`).
- R2: Pinning topology shared across host classes (`vub_bridge_higher_m_pinning_probe.py`).
- R3: Z_3-shift / Z_3²-shift is 50/50 on same-orbit pairs (`vub_bridge_z3_shift_classifier.py`).

These rule out the simplest Z_3-based substrate mechanisms for Need-D.

USER CONTEXT (2026-05-05 EOD+3)
================================
User has Need-A2 (canonical generation-Z_3 derivation) running in background.
Pickup direction: attack Need-D.

THIS PROBE
==========
Audit-first scoping that:
1. Maps Need-D into bounded sub-components vs research-level sub-components.
2. Identifies what's already implicitly addressed by existing framework apparatus.
3. Tests bounded structural attacks for the open sub-components using:
   (a) Hamming-weight species labels from `theorem_charge_before_color`
   (b) V_{-1}-T_{B-L} symmetry-breaking finding from EOD+3 P34 strengthening
   (c) Hodge dual structure on Cl(6) Fock (Λ^1 ↔ Λ^2 at k*=3)
4. Honest verdict on what closes vs what remains research-level.

EXPECTED OUTCOME (per audit-first methodology)
==============================================
Need-D's bounded sub-components (CKM magnitudes, δ_CP) are ALREADY addressed
by existing framework apparatus (substrate counting + V_{-1}-T_{B-L} bridge).
The genuinely-open sub-components (M_species per-species mass scale, Y_u vs
Y_d eigenbasis distinction on C³_gen) are multi-session research-level —
M1 Bloch eigenmode and M2 multiway formalism routes both 3-5 sessions per
ckm_substrate_identification doc.

This probe sharpens the closure target rather than closing Need-D fully.
"""

from __future__ import annotations

from itertools import combinations
import math
from fractions import Fraction
import numpy as np

TOL = 1e-12
omega = np.exp(2j * np.pi / 3)


# ============================================================================
# 1. Map Need-D into sub-components
# ============================================================================
print("=" * 78)
print("Step 1: Need-D sub-component map")
print("=" * 78)
print()

subcomponents = [
    ("Need-D-1: V_ab MAGNITUDES (CKM magnitudes)",
     "ALREADY closed",
     "Substrate coupling density: V_us = k*²/(g·N_atoms) = 9/40 (Row P4 theorem-grade); "
     "V_cb = α_1/(1-α_1) = 256/6305 via Hashimoto walker (Row P3 theorem-grade); "
     "V_ub = multicycle sum giving 3.767e-3 (Row P14, ADOPTED-A5b-Sub3-conditional). "
     "STRUCTURAL IDENTIFICATION GAP: which substrate walk = which ΔGen — adoption "
     "ADOPTED-A5b-Sub3 (Level 3 sub-class classifier, currently active)."),

    ("Need-D-2: δ_CP PHASES",
     "STRENGTHENED (EOD+3)",
     "V_{-1}-T_{B-L} symmetry-breaking bridge via SO(3)_K4 → SO(2)_u "
     "(`R14_P34_strengthening_symmetry_breaking_2026-05-05.md`); the per-atom polar "
     "angle from u-axis is the unique SO(2)_u-invariant phase per atom. Type 6c (6c) "
     "PASSES via channel_select with structural channel. Conditional on the same "
     "ADOPTED-A5b-Sub3 + delta_CP_CKM_geometry §6 Other-Smuggle as Need-D-1."),

    ("Need-D-3: Y_u vs Y_d EIGENBASIS DISTINCTION on C³_gen",
     "RESEARCH-LEVEL OPEN",
     "Multi-session: M1 Bloch eigenmode (V_Ram(P) C_3 multiplicities (4,2,2) = COLOR "
     "not generation; need different substrate structure), M2 multiway formalism "
     "(A3 auxiliary purifying space sector labels unexplored). Both 3-5 sessions per "
     "ckm_substrate_identification_2026-04-29."),

    ("Need-D-4: M_species PER-SPECIES MASS SCALES (Yukawa hierarchy)",
     "RESEARCH-LEVEL OPEN",
     "y_τ theorem-grade (sector-blind formula α_1_full/k*²); other species' Yukawas "
     "open. EOD+3 R1 attempt (C_3 isotypic + charge) STRUCTURALLY OBSTRUCTED via "
     "Λ^1 ≅ Λ^2 Hodge duality at k*=3 + 14-orders-of-magnitude span — incompatible "
     "with small-rational K-combinations."),
]

for name, status, detail in subcomponents:
    print(f"  {name}")
    print(f"    Status: {status}")
    print(f"    {detail}")
    print()

print(f"  AGGREGATE: 2 of 4 sub-components addressed by existing apparatus;")
print(f"             2 of 4 are genuinely multi-session research-level.")
print()


# ============================================================================
# 2. Bounded structural attack — Hamming-weight species filter on substrate walks
# ============================================================================
print("=" * 78)
print("Step 2: Bounded attack — Hamming-weight species filter")
print("=" * 78)
print()
print(f"  Per `theorem_charge_before_color.md` §9 (Furey 2018 §3 identification):")
print(f"  Cl(6) Fock at trivalent vertex decomposes by Hamming weight n ∈ {{0,1,2,3}}:")
print()
print(f"    n=0 → ν_L (neutrino, lepton, dim 1, U(1) trivial)")
print(f"    n=1 → d_L^{{1,2,3}} (down-type quark, 3 colors, dim 3, SU(3) fundamental)")
print(f"    n=2 → ū_R^{{1,2,3}} (up-type antiquark, 3 anti-colors, dim 3, SU(3) anti-fundamental)")
print(f"    n=3 → e_L^+ (charged lepton, dim 1, U(1) charge 1)")
print()
print(f"  CANDIDATE ATTACK: substrate walks have endpoints at trivalent vertices;")
print(f"  each vertex's Cl(6) Fock state has a Hamming weight; the walk's SPECIES")
print(f"  LABEL is determined by endpoint Hamming weights.")
print()


# ============================================================================
# 3. Test the candidate — does Hamming-weight filter give Y_u ≠ Y_d?
# ============================================================================
print("=" * 78)
print("Step 3: Does Hamming-weight species filter give CKM ≠ I?")
print("=" * 78)
print()

# Test: is Λ^1 isotypically distinguishable from Λ^2 at k*=3?
# (Same test as EOD+3 R1 attempt but specific to Need-D framing)

def levels(k_star):
    return [(n, list(combinations(range(k_star), n))) for n in range(k_star + 1)]


def sigma_matrix(level_basis, k_star):
    """Build C_k cyclic permutation matrix on Λ^n(C^k)."""
    dim = len(level_basis)
    if dim == 0:
        return np.zeros((0, 0), dtype=complex)
    M = np.zeros((dim, dim), dtype=complex)
    idx = {s: k for k, s in enumerate(level_basis)}
    for col, src in enumerate(level_basis):
        # σ(e_i) = e_{(i+1) mod k}
        image = [(i + 1) % k_star for i in src]
        # Sort and track sign
        seq = list(image)
        sign = 1
        n = len(seq)
        for i in range(n):
            for j in range(0, n - i - 1):
                if seq[j] > seq[j + 1]:
                    seq[j], seq[j + 1] = seq[j + 1], seq[j]
                    sign = -sign
        tgt = tuple(seq)
        M[idx[tgt], col] = sign
    return M


K_STAR = 3
fock = levels(K_STAR)
sig_mats = {n: sigma_matrix(basis, K_STAR) for n, basis in fock}

# Compute C_3 isotypic projectors per level
def isotypic_projector(sig, alpha):
    """P_alpha = (1/3) Σ_k ω^{-αk} σ^k for α ∈ {0, 1, 2}."""
    if sig.shape[0] == 0:
        return sig.copy()
    I = np.eye(sig.shape[0], dtype=complex)
    return (I + np.conj(omega**alpha) * sig + np.conj(omega**(2*alpha)) * (sig @ sig)) / 3


# Compute per-level isotypic dimensions
print(f"  C_3 isotypic decomposition per Hamming weight:")
print(f"  {'n':<3}  {'level':<6}  {'dim':<5}  {'trivial':<8}  {'ω':<5}  {'ω̄':<5}  {'species':<35}")
print(f"  {'-'*3}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*35}")
species = {0: "ν_L (neutrino)",
           1: "d_L (down-type quark)",
           2: "ū_R (up-type antiquark)",
           3: "e_L^+ (charged lepton)"}

isotypic_dims = {}
for n, basis in fock:
    sig = sig_mats[n]
    if sig.shape[0] == 0:
        continue
    P_t = isotypic_projector(sig, 0)
    P_o = isotypic_projector(sig, 1)
    P_b = isotypic_projector(sig, 2)
    dim_t = round(float(np.trace(P_t).real))
    dim_o = round(float(np.trace(P_o).real))
    dim_b = round(float(np.trace(P_b).real))
    isotypic_dims[n] = (dim_t, dim_o, dim_b)
    print(f"  {n:<3}  Λ^{n:<5} {len(basis):<5}  {dim_t:<8}  {dim_o:<5}  {dim_b:<5}  {species[n]:<35}")

print()
n1_iso = isotypic_dims[1]
n2_iso = isotypic_dims[2]
hodge_match = (n1_iso == n2_iso)
print(f"  Λ^1 isotypic: {n1_iso}")
print(f"  Λ^2 isotypic: {n2_iso}")
print(f"  IDENTICAL?  {hodge_match}  →  HODGE DUALITY confirmed at k*=3.")
print()
print(f"  CONSEQUENCE: at k*=3, the C_3 isotypic structure of Λ^1 and Λ^2 is")
print(f"  IDENTICAL. Hamming-weight species filter on substrate walks DOES")
print(f"  distinguish u-type (n=2) from d-type (n=1) AT THE SECTOR LEVEL but")
print(f"  NOT at the C_3 EIGENBASIS LEVEL — the eigenbases are isomorphic via")
print(f"  Hodge dual.")
print()
print(f"  If Y_u (on Λ^2-indexed C³_gen) and Y_d (on Λ^1-indexed C³_gen) are")
print(f"  both diagonal in the C_3-Fourier basis of their respective levels,")
print(f"  the eigenbases match up via Hodge dual identification → CKM = I.")
print()
print(f"  PARTIAL FINDING: Hamming-weight species filter is NECESSARY (gives")
print(f"  the species labels on substrate walks structurally) but NOT SUFFICIENT")
print(f"  for non-trivial CKM. An ADDITIONAL mechanism is required to break")
print(f"  the Hodge dual equivalence.")
print()


# ============================================================================
# 4. Bounded attack 2 — V_{-1}-T_{B-L} angle as species-differentiation source
# ============================================================================
print("=" * 78)
print("Step 4: V_{-1}-T_{B-L} angle as additional species-differentiation source")
print("=" * 78)
print()
print(f"  EOD+3 P34 strengthening identified: T_{{B-L}}·v_0 / |T_{{B-L}}·v_0| = -q_lepton")
print(f"  / |q_lepton|. The V_{{-1}}-T_{{B-L}} angle per atom equals arccos(T_{{B-L}}_i):")
print(f"    Lepton: arccos(-1) = π")
print(f"    Color (×3): arccos(+1/3) = K_4 dihedral ≈ 70.53°")
print()
print(f"  CANDIDATE: Y_u and Y_d acquire different OVERALL PHASES on C³_gen")
print(f"  determined by their respective T_{{B-L}}_i ratios:")
print()
print(f"  In Furey's identification, both u (n=2) and d (n=1) live at COLOR")
print(f"  atoms (Q_BL = +1/3 each in PS). So T_{{B-L}}_u = T_{{B-L}}_d = +1/3.")
print(f"  V_{{-1}}-T_{{B-L}} angle is THE SAME for u and d at the PS color sector.")
print()
print(f"  RESULT: V_{{-1}}-T_{{B-L}} bridge does NOT distinguish u from d within")
print(f"  the color sector. Both color-quark species inherit the same color-")
print(f"  atom geometry. Need-D-3 NOT closed by V_{{-1}}-T_{{B-L}} alone.")
print()
print(f"  WHAT WOULD BE NEEDED: a STRUCTURAL distinction between u and d that")
print(f"  goes BEYOND PS color geometry. Candidates within framework:")
print(f"    (i)  SU(2)_R doublet (u_R, d_R) electroweak action — requires SU(2)_R")
print(f"         gauge structure on substrate (theorem_g2_edge_qubit_su2 gives")
print(f"         SU(2)_L on edge qubit; SU(2)_R action unspecified on srs).")
print(f"    (ii) Hodge dual sign asymmetry — CHECKED in Step 3, no asymmetry at k*=3.")
print(f"    (iii) Multiway A3 auxiliary purifying space sector labels — UNEXPLORED")
print(f"          per ckm_substrate_identification §4 M2 route.")
print()


# ============================================================================
# 5. Bounded attack 3 — does the up vs down quark have a SEMANTIC distinction
# in the framework via Higgs SU(2)_L doublet partner?
# ============================================================================
print("=" * 78)
print("Step 5: SU(2)_L Higgs doublet partner mechanism")
print("=" * 78)
print()
print(f"  For SM Yukawa coupling at SU(2)_L doublet (u_L, d_L):")
print(f"    L_Y = Y_d · Q̄_L · H · d_R + Y_u · Q̄_L · H̃ · u_R + h.c.")
print(f"    where H = (h^+, h^0)^T, H̃ = iσ_2 H^* = (h^{{0*}}, -h^-)^T")
print()
print(f"  After EWSB with ⟨h^0⟩ = v/√2:")
print(f"    Y_d coupling uses h^0 component of H")
print(f"    Y_u coupling uses h^{{0*}} component of H̃ (CONJUGATE Higgs)")
print()
print(f"  STRUCTURAL DISTINCTION: u-type uses CONJUGATE Higgs vs d-type uses Higgs.")
print(f"  In framework's edge-qubit Higgs (theorem_g2_edge_qubit_su2), the H ↔ H̃")
print(f"  duality is a complex conjugation on the edge Cl(0,2) ≅ ℍ structure.")
print()
print(f"  CANDIDATE: Y_u and Y_d differ by complex conjugation acting on edge qubit.")
print(f"  IF the EDGE QUBIT structure transforms non-trivially under conjugation,")
print(f"  THEN Y_u and Y_d eigenbases on C³_gen could differ.")
print()
print(f"  AUDIT: in framework's theorem_g2_edge_qubit_su2, the edge qubit is the")
print(f"  Cl(0,2) ≅ ℍ algebra. Complex conjugation on ℍ is q → q̄ (quaternion")
print(f"  conjugation), which is anti-linear. Under SU(2)_L action, H → e^(iθσ/2)·H,")
print(f"  conjugate H̃ → e^(-iθσ̄/2)·H̃ (different rotation).")
print()
print(f"  If the Yukawa amplitude inherits the edge qubit's representation, then:")
print(f"    Y_d ψ̄_L H ψ_R: H transforms as σ under SU(2)_L")
print(f"    Y_u ψ̄_L H̃ ψ_R: H̃ transforms as σ̄ (conjugate) under SU(2)_L")
print()
print(f"  This is a DIFFERENT representation of SU(2)_L for Y_u vs Y_d. If C³_gen")
print(f"  has an SU(2)_L action (which it might inherit via mass-eigenstate")
print(f"  diagonalization), then Y_u and Y_d would have different eigenbases on")
print(f"  C³_gen via different SU(2)_L representations.")
print()
print(f"  HOWEVER: this requires deriving SU(2)_L action on C³_gen, which is")
print(f"  unspecified in current framework apparatus. The EDGE qubit SU(2)_L is")
print(f"  theorem-grade per G2 theorem, but its lift to C³_gen is not derived.")
print()
print(f"  STATUS: This is a STRUCTURAL CANDIDATE for Need-D bridge, but its")
print(f"  closure requires deriving the SU(2)_L action on C³_gen (which connects")
print(f"  edge qubit to observer Hilbert space). This is EXACTLY the M2 multiway")
print(f"  formalism direction (sector labels in A3 auxiliary purifying space)")
print(f"  per `ckm_substrate_identification §4`.")
print()


# ============================================================================
# 6. Honest verdict and recommendations
# ============================================================================
print("=" * 78)
print("Step 6: Honest verdict and recommendations")
print("=" * 78)
print()
print(f"""  NEED-D ATTACK OUTCOME (this audit-first probe):

  CONFIRMED: 2 of 4 Need-D sub-components are already addressed by existing
             framework apparatus:
             • Need-D-1 (V_ab magnitudes) — substrate counting + M1 walker
             • Need-D-2 (δ_CP phases) — V_{{-1}}-T_{{B-L}} symmetry breaking
             Both inherit ADOPTED-A5b-Sub3 (un-graduated) for the structural
             identification step.

  SHARPENED: 2 of 4 Need-D sub-components are genuinely multi-session research:
             • Need-D-3 (Y_u vs Y_d eigenbasis on C³_gen)
             • Need-D-4 (M_species per-species mass scales / Yukawa hierarchy)

  STRUCTURAL CANDIDATES FOR Need-D-3 / Need-D-4 (this probe):

  (i)   Hamming-weight species filter (theorem_charge_before_color):
        NECESSARY but NOT SUFFICIENT. Hodge duality at k*=3 makes Λ^1 ≅ Λ^2
        isotypically, so species filter alone can't distinguish u from d.

  (ii)  V_{{-1}}-T_{{B-L}} angle (EOD+3 finding):
        DOES NOT distinguish u from d within color sector (both at +1/3).
        Distinguishes lepton from quark, not up from down.

  (iii) SU(2)_L Higgs doublet partner mechanism (NEW candidate, this probe):
        Y_u and Y_d couple to CONJUGATE representations of SU(2)_L
        (H̃ vs H). If C³_gen carries an SU(2)_L action lifted from edge
        qubit, this gives different eigenbases for Y_u vs Y_d. STRUCTURAL
        CANDIDATE but requires deriving SU(2)_L action on C³_gen — exactly
        the M2 multiway formalism direction.

  RECOMMENDED NEXT-SESSION WORK (multi-session, with Need-A2 closure):

  (a) Wait for Need-A2 closure (running in background). If Need-A2 derives
      a canonical generation-Z_3 on observer Hilbert space C³_gen, then
      M_species via Cl(6,0) Hamming weight × generation-Z_3 product becomes
      the natural Need-D-3/Need-D-4 mechanism. ~1-2 sessions to bridge after
      Need-A2 closure.

  (b) Pursue M2 multiway formalism route — articulate sector labels on A3
      auxiliary purifying space, and identify mass eigenstates as labeled
      branches. ~3-5 sessions per ckm_substrate_identification §4.

  (c) Derive SU(2)_L action on C³_gen from edge qubit lift. This is candidate
      (iii) above. Bounded if edge-qubit-to-observer bridge is derivable;
      otherwise multi-session.

  HONEST READ:

    Need-D's bounded sub-components (CKM magnitudes, δ_CP phases) are
    already structurally addressed by existing framework apparatus. The
    genuinely-open sub-components (M_species per-species mass scales, Y_u/Y_d
    eigenbasis distinction) require either:
    - Closure of Need-A2 first (then bridge is bounded), or
    - Direct attack via M2 multiway formalism (~3-5 sessions), or
    - SU(2)_L-on-C³_gen lift (bounded if edge-qubit-bridge derivable).

    This session's contribution: maps Need-D's territory into bounded vs
    research-level sub-components, identifies SU(2)_L Higgs doublet partner
    as a NEW structural candidate for the eigenbasis distinction (route iii),
    and confirms that Hamming-weight + V_{{-1}}-T_{{B-L}} alone are insufficient.
""")

print("=" * 78)
print("END")
print("=" * 78)
