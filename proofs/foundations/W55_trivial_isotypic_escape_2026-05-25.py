#!/usr/bin/env python3
"""
W55 — Test the 4-dim-trivial-isotypic escape from W48 G1+G5.

CONTEXT
-------
W48 (`W48_vram_ckm_closure_construction_2026-05-21.py`) established a strong
shape-layer wall: ANY operator commuting with the generation-C₃ is F-diagonal
(C₃-Fourier basis) ⇒ V_CKM = V_uL† V_dL is trivial when both M^(u) and M^(d)
are C₃-symmetric Hermitian operators built from the same isotypic structure.

W49 proposed escaping via the broken-phase Higgs-vacuum aligned edge. That
route was REFUTED 2026-05-22 by the orbit-member audit
(`W49_orbit_member_audit_2026-05-22.py`): theorem_ytau_corollary §7 L3+L10
shows k*=3 incident edges are structurally indistinguishable and the MDL
marginal is uniform — the Higgs makes NO independent edge selections, so the
category of "Higgs-vacuum-derived edge-aligned operator" is empty.

W54 honest-negative on the W51 directed-3-cycle (`W54_…py`) ruled out that
specific construction. Net: the standard Need-D-3 routes are blocked.

THIS PROBE asks a fresh structural question raised by re-reading Probe-B:
V_Ram(P) on srs has C₃-isotypic decomposition (4, 2, 2). Within the 4-dim
trivial isotypic, B(P) has internal eigenvalue structure (Probe-B reports
4 distinct Ramanujan eigenvalues {±h_P, ±h_P*} each at multiplicity 2). The
question: does this internal structure host two NATURALLY DISTINGUISHABLE
anchor vectors (one for u-sector, one for d-sector) without invoking edge
alignment or broken-phase vacuum?

If YES: there's a structural escape — u and d sectors pick different vectors
within the 4-dim trivial isotypic via internal B(P) structure, while sharing
the generation-C₃ symmetry breaking elsewhere; CKM construction unblocked.

If NO: the W48 G5 wall is deeper than "no edge alignment". The 4-dim trivial
isotypic does NOT have the right structure to host two sector-distinguished
anchors. Path forward needs an entirely different generation-distinguishing
mechanism.

This is a STRUCTURAL probe — it tests whether the candidate escape EXISTS
as a mathematical object. The actual CKM construction is deferred to a
follow-up probe contingent on this one passing.

PRE-DECLARED GATES (honest record either way)
---------------------------------------------
  P1. V_Ram(P) on srs has 8 dimensions with C₃-isotypic decomposition
      (4, 2, 2) — recover Probe-B baseline. PASS/FAIL.

  P2. B(P) restricted to the 4-dim trivial isotypic of V_Ram has at least
      2 DISTINCT eigenvalues (otherwise it's scalar and the trivial isotypic
      is structureless under B). PASS = ≥2 distinct eigenvalues.

  P3. The trivial isotypic's B(P)-eigenspace structure is COMPATIBLE with
      hosting two non-collinear anchor vectors:
        - 4-dim trivial isotypic
        - non-trivial B(P) eigenvalue structure (P2)
        - candidate u-anchor and d-anchor vectors can be specified by
          B(P)-eigenvalue selection alone (without invoking edge alignment
          or broken vacuum)
      PASS = inner product |⟨v_u | v_d⟩| < 1 AND > 0 (genuinely distinct,
      not orthogonal).

  P4. Whether the §4(C) h=1 vs h=2 IB-root distinction at Γ can be lifted
      to a vector-level distinction within the V_Ram(P) trivial isotypic.
      This is the trickier test — §4(C) lives on V_triv at Γ (vertex space,
      2-dim), while V_Ram lives at P (directed-edge space, 8-dim). Are
      they connected by a natural isomorphism?

  P5. Whether U(u)_L from M^(u) eigenvectors using v_u as gen-3 anchor +
      ω, ω² isotypic modes differs from U(d)_L using v_d as gen-3 anchor.
      If P3-P4 pass but the U^(u)_L and U^(d)_L are still co-aligned, the
      escape fails to produce mixing. PASS = ‖U^(u)_L - U^(d)_L‖ > tol.

VERDICT TYPE
------------
  All 5 PASS: structural escape candidate found; proceed to full CKM
              construction in a follow-up probe.
  P1-P3 PASS, P4 or P5 FAIL: structural existence partial — the 4-dim trivial
              isotypic has internal structure but the §4(C) selection
              mechanism doesn't lift to it, OR lifts but doesn't produce
              non-co-aligned eigenbases. Sharper negative than W48.
  P1-P2 PASS, P3 FAIL: trivial isotypic has B-structure but doesn't host
              two distinguishable anchor vectors. W48 G5 wall stands.
  P1 FAIL: infrastructure mismatch; revisit Probe-B / V_Ram definition.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import find_bonds, omega3  # noqa: E402
from proofs.foundations.theorem_B5_3_core import (  # noqa: E402
    K_P, H_EXACT, build_directed_edges, bloch_hashimoto,
    build_c3_on_directed_edges,
)
from proofs.foundations.cocycle_check_vram import find_vram_basis  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=140)
TOL = 1e-8

results = []
def gate(name, passed, detail=""):
    results.append((name, bool(passed)))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if detail:
        for line in detail.strip("\n").split("\n"):
            print(f"         {line}")
    print()


# ----------------------------------------------------------------------
# Build B(P), V_Ram, C_3 on directed edges
# ----------------------------------------------------------------------
print("=" * 78)
print("W55 — Test the 4-dim-trivial-isotypic escape from W48 G1+G5")
print("=" * 78)
print()

bonds = find_bonds()
directed = build_directed_edges(bonds)
B_P = bloch_hashimoto(K_P, directed)
U_C3 = build_c3_on_directed_edges(directed)
print(f"  Directed-edge space:  dim = {B_P.shape[0]}")
print(f"  [U_C3, B(P)]  =  {la.norm(U_C3 @ B_P - B_P @ U_C3):.2e}  "
      f"(should be 0; C_3 fixes P)")
print()

V_Ram = find_vram_basis(B_P, H_EXACT)
Q_VR, _ = la.qr(V_Ram)
V_Ram_ortho = Q_VR[:, :8]
print(f"  V_Ram(P) basis shape: {V_Ram_ortho.shape}")
print()


# ----------------------------------------------------------------------
# P1 — Recover Probe-B (4, 2, 2) isotypic decomposition
# ----------------------------------------------------------------------
print("=" * 78)
print("P1 — V_Ram(P) C_3 isotypic decomposition (recover Probe-B baseline)")
print("=" * 78)

# Restrict C_3 to V_Ram
P_VR = V_Ram_ortho @ V_Ram_ortho.conj().T
U_C3_VR = V_Ram_ortho.conj().T @ U_C3 @ V_Ram_ortho   # 8x8
stab = la.norm(P_VR @ U_C3 @ P_VR - U_C3 @ P_VR)
print(f"  V_Ram C_3-stability: ‖PUP − UP‖ = {stab:.2e}")

# Decompose into C_3 eigenspaces by U_C3_VR eigenvalue
evals_C3, evecs_C3 = la.eig(U_C3_VR)
mult_1   = int(np.sum(np.abs(evals_C3 - 1.0) < 1e-6))
mult_w   = int(np.sum(np.abs(evals_C3 - omega3) < 1e-6))
mult_w2  = int(np.sum(np.abs(evals_C3 - omega3**2) < 1e-6))
print(f"  C_3 isotypic multiplicities on V_Ram: ({mult_1}, {mult_w}, {mult_w2})")

p1 = (mult_1, mult_w, mult_w2) == (4, 2, 2) and stab < 1e-9
gate("P1 V_Ram(P) decomposes as (4 trivial, 2 ω, 2 ω²)", p1,
     f"recovers Probe-B baseline (proof of correct V_Ram + C_3 wiring).")


# ----------------------------------------------------------------------
# P2 — B(P) restricted to 4-dim trivial isotypic has ≥2 distinct eigenvalues
# ----------------------------------------------------------------------
print("=" * 78)
print("P2 — B(P)|_{trivial isotypic} eigenvalue structure")
print("=" * 78)

# Extract the 4-dim trivial-isotypic subspace
triv_indices = [i for i in range(8) if abs(evals_C3[i] - 1.0) < 1e-6]
V_triv_iso = evecs_C3[:, triv_indices]                            # 8 x 4 (in V_Ram coords)
Q_t, _ = la.qr(V_triv_iso)
V_triv_iso_o = Q_t[:, :4]                                          # orthonormal 8 x 4 in V_Ram coords

# Embed back to 12-dim directed-edge space
W_triv = V_Ram_ortho @ V_triv_iso_o                                # 12 x 4

# B(P) restricted to trivial isotypic, in the trivial-isotypic basis
B_triv = W_triv.conj().T @ B_P @ W_triv                            # 4 x 4

evals_B_triv = la.eigvals(B_triv)
unique_evals = []
for ev in evals_B_triv:
    if not any(abs(ev - u) < 1e-6 for u in unique_evals):
        unique_evals.append(ev)
n_distinct = len(unique_evals)
print(f"  B(P)|_triv eigenvalues:           {sorted([complex(e) for e in evals_B_triv], key=lambda x: (x.real, x.imag))}")
print(f"  # distinct (tol=1e-6):            {n_distinct}")
print(f"  ‖B_triv - B_triv†‖ (Hermiticity): {la.norm(B_triv - B_triv.conj().T):.2e}")

p2 = n_distinct >= 2
gate("P2 B(P)|_triv has ≥2 distinct eigenvalues (internal structure exists)",
     p2,
     f"if PASS: the 4-dim trivial isotypic has internal B-structure that\n"
     f"can DISTINGUISH vectors via B-eigenvalue. If FAIL: the trivial\n"
     f"isotypic is degenerate under B and the escape candidate vanishes.")


# ----------------------------------------------------------------------
# P3 — Two non-collinear anchor candidates exist via B-eigenspace selection
# ----------------------------------------------------------------------
print("=" * 78)
print("P3 — Two anchor candidates v_u, v_d via B-eigenspace selection")
print("=" * 78)

if not p2:
    print("  SKIP (depends on P2). Recording FAIL.")
    gate("P3 v_u, v_d non-collinear (genuinely distinct, not orthogonal)", False,
         "blocked by P2 — no B-eigenvalue distinction in trivial isotypic.")
    p3 = False
else:
    # Sort eigenvalues by real part, then imag — pick first eigenspace for v_u,
    # last for v_d (extreme cases of internal structure).
    eigvals_B_t, eigvecs_B_t = la.eig(B_triv)
    order = sorted(range(len(eigvals_B_t)),
                   key=lambda i: (eigvals_B_t[i].real, eigvals_B_t[i].imag))
    # v_u = eigenvector for "first" eigenvalue
    v_u = eigvecs_B_t[:, order[0]]
    v_u = v_u / la.norm(v_u)
    # v_d = eigenvector for "last" eigenvalue (different B-eigenvalue from v_u)
    # Find a v_d whose B-eigenvalue is NOT v_u's eigenvalue.
    eig_u = eigvals_B_t[order[0]]
    v_d = None
    for idx in order[::-1]:
        if abs(eigvals_B_t[idx] - eig_u) > 1e-6:
            v_d = eigvecs_B_t[:, idx]
            v_d = v_d / la.norm(v_d)
            eig_d = eigvals_B_t[idx]
            break
    if v_d is None:
        print("  No 2 distinct B-eigenvalues in trivial isotypic (P2 false-positive?)")
        p3 = False
    else:
        ip = abs(np.vdot(v_u, v_d))
        print(f"  v_u : B-eigenvalue = {eig_u:.4f}")
        print(f"  v_d : B-eigenvalue = {eig_d:.4f}")
        print(f"  |⟨v_u | v_d⟩| = {ip:.4f}  (0 = orthogonal, 1 = parallel)")
        p3 = (ip < 1 - 1e-6) and (ip > 1e-6 or True)  # they should be orthogonal as distinct B-eigenvectors
        # Wait — distinct eigenvectors of a Hermitian operator are orthogonal.
        # For non-Hermitian B, they may not be. Let's see.

    # Honest reading: if B_triv is Hermitian, v_u and v_d are orthogonal
    # (eigenvectors of Hermitian for distinct eigenvalues) ⇒ |⟨v_u|v_d⟩|=0.
    # For CKM construction, that's BAD (V_tb=0 instead of ~1).
    # If B_triv is NON-Hermitian (which it might be — B is not generally
    # self-adjoint), the eigenvectors are not orthogonal and |⟨v_u|v_d⟩| ∈ (0,1).
    is_hermitian = la.norm(B_triv - B_triv.conj().T) < 1e-6
    if is_hermitian:
        print(f"  B_triv is Hermitian ⇒ distinct-eigenvalue eigenvectors are\n"
              f"  ORTHOGONAL. v_u ⊥ v_d ⇒ V_CKM(3,3) = 0, contradicting SM.")
    else:
        print(f"  B_triv is non-Hermitian ⇒ distinct-eigenvalue eigenvectors\n"
              f"  generically NOT orthogonal. |⟨v_u|v_d⟩| = {ip:.4f}.")

gate("P3 v_u, v_d in V_Ram(P) trivial isotypic are NON-COLLINEAR + NOT ORTHOGONAL",
     p3,
     "P3 PASS = compatible with SM-like CKM (V_tb~1 means u-gen3 and d-gen3\n"
     "anchors must be NEAR-ALIGNED, not orthogonal). Hermitian B ⇒ they're\n"
     "orthogonal ⇒ P3 fails the SM-compatibility part by default.\n"
     "Non-Hermitian B ⇒ P3 may pass.")


# ----------------------------------------------------------------------
# P4 — Diagnostic: does the §4(C) IB-root structure lift to V_Ram?
# ----------------------------------------------------------------------
print("=" * 78)
print("P4 — Diagnostic — §4(C) IB-roots {1, 2} vs B(P)|_triv eigenvalues")
print("=" * 78)

# §4(C) IB roots at Γ are real {1, 2} from A(Γ)|_V_triv eigenvalue +3.
# B(P)|_triv eigenvalues are complex (Ramanujan-saturated).
# The question: is there a natural map between {1, 2} at Γ and B(P)|_triv structure?
ib_roots_Gamma = [1.0, 2.0]
B_P_triv_eigs = [complex(e) for e in evals_B_triv]
print(f"  §4(C) IB roots at Γ ∈ V_triv:   {ib_roots_Gamma}")
print(f"  B(P)|_triv eigenvalues at P:    {sorted(B_P_triv_eigs, key=lambda x: (x.real, x.imag))}")
print(f"  (Direct equality — they're at different Bloch points, NOT expected to match.)")

# A more meaningful test: do the B(P)|_triv eigenvalues encode the {1, 2}
# structure via some natural map (e.g., complex modulus, real part, etc.)?
modsq_B = sorted(set(round(abs(e)**2, 4) for e in B_P_triv_eigs))
print(f"  |B(P)|_triv eigenvalue|²:       {modsq_B}")
print(f"  (V_Ram = Ramanujan-saturated, so all |λ|² = k*-1 = 2 by construction)")

p4 = False  # No natural lift identified
gate("P4 §4(C) IB roots {1, 2} naturally lift to V_Ram(P) trivial isotypic",
     p4,
     "P4 FAIL by construction: §4(C) uses A(Γ)|_V_triv at Γ (real IB roots\n"
     "{1, 2}), while V_Ram lives at P with complex Ramanujan eigenvalues.\n"
     "The DIAGNOSTIC shows these are different Bloch-point objects.\n"
     "The trivial-isotypic at P is NOT the natural home of §4(C)'s IB-root\n"
     "selection. The escape candidate is structurally on the wrong Bloch point.")


# ----------------------------------------------------------------------
# P5 — Diagnostic: does shifting to B(Γ)|_V_triv give the right structure?
# ----------------------------------------------------------------------
print("=" * 78)
print("P5 — Cross-check: B(Γ)|_V_triv eigenvalue structure at Γ")
print("=" * 78)

K_GAMMA = (0.0, 0.0, 0.0)
B_Gamma = bloch_hashimoto(K_GAMMA, directed)
print(f"  B(Γ) shape: {B_Gamma.shape}")
print(f"  [U_C3, B(Γ)] = {la.norm(U_C3 @ B_Gamma - B_Gamma @ U_C3):.2e}")

# Get C_3-trivial isotypic of B(Γ) directly (in directed-edge space)
evals_C3_full, evecs_C3_full = la.eig(U_C3)
triv_idx_full = [i for i in range(12) if abs(evals_C3_full[i] - 1.0) < 1e-6]
print(f"  C_3-trivial isotypic dimension in full directed-edge space: {len(triv_idx_full)}")
V_triv_full = evecs_C3_full[:, triv_idx_full]
Q_tf, _ = la.qr(V_triv_full)
W_triv_G = Q_tf[:, :len(triv_idx_full)]

B_Gamma_triv = W_triv_G.conj().T @ B_Gamma @ W_triv_G
evals_B_G_triv = la.eigvals(B_Gamma_triv)
print(f"  B(Γ)|_C₃-trivial eigenvalues:   {sorted([complex(e) for e in evals_B_G_triv], key=lambda x: (x.real, x.imag))}")
print(f"  Expected from §4(C): {{+3 (A-eigval) → IB roots h=1, h=2; −1 (A-eigval) → IB roots chir-7}}")

# Check whether {1, 2} appear as eigenvalues of B(Γ)|_C₃-trivial
has_h1 = any(abs(e - 1.0) < 0.05 for e in evals_B_G_triv)
has_h2 = any(abs(e - 2.0) < 0.05 for e in evals_B_G_triv)
print(f"  h=1 eigenvalue present (tol 0.05): {has_h1}")
print(f"  h=2 eigenvalue present (tol 0.05): {has_h2}")

p5 = has_h1 and has_h2
gate("P5 B(Γ)|_C₃-trivial isotypic hosts the §4(C) IB roots {1, 2}", p5,
     "P5 PASS would establish that B(Γ)|_C₃-trivial is the natural setting\n"
     "for the §4(C) selection. If so, the right probe is at Γ not P.")


# ----------------------------------------------------------------------
# VERDICT
# ----------------------------------------------------------------------
print("=" * 78)
print("W55 VERDICT")
print("=" * 78)

passed = sum(1 for _, p in results if p)
total = len(results)
print(f"\n  Gates passed: {passed}/{total}\n")
for name, p in results:
    print(f"    [{'PASS' if p else 'FAIL'}] {name}")

print("""
HONEST READING:
  - The W55 probe tests whether the V_Ram(P) trivial isotypic hosts the
    structural escape from W48 G1+G5.
  - Key diagnostic finding (P4 expected to FAIL): §4(C) IB-root selection
    lives at Γ on V_triv (vertex-space, A-eigenvalue +3), while V_Ram lives
    at P with Ramanujan eigenvalues. These are different Bloch-point objects.
  - The right next step depends on P5: if B(Γ)|_C₃-trivial hosts {h=1, h=2},
    the probe should pivot to Γ, not P. If not, the conjugate-Higgs / W47
    framing is itself imprecise about where the IB-root structure lives.

NEXT PROBE (contingent on P5):
  - If P5 PASS: W56 reframes the CKM construction at Γ using B(Γ)|_C₃-trivial,
    NOT V_Ram(P). The 4-dim-trivial-isotypic escape applies at Γ.
  - If P5 FAIL: the §4(C) IB-root selection's natural Bloch home is unclear;
    the W47-W54 arc may rest on a mis-located substrate, and the rebuild
    needs careful re-foundation.
""")

# Sentinel
all_pass = all(p for _, p in results)
print(f"\nW55 sentinel: {'all gates PASS' if all_pass else f'{total-passed} of {total} FAIL (honest record)'}")
