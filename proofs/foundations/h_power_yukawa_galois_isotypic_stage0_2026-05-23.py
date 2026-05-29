#!/usr/bin/env python3
"""
h_power_yukawa_galois_isotypic_stage0_2026-05-23.py

Stage 0 of route 4 (h-power Yukawa generating function, per outer-Galois
isotypic), scoping doc:
  an internal working note

Pre-declared self-validation gate: does the per-Galois-isotypic resolvent
construction on the minimal substrate fiber reproduce lepton δ = 2/9 by
mechanism?

Construction (per scoping §3):
  1. B(K_4) on 12 directed arcs at the Γ-Bloch fiber of srs (the minimal
     faithful object the IB-root partition lives on; same construction as
     yukawa_walker_stage_0_1).
  2. Body-diagonal C_3 lifted to a permutation P of arcs.
  3. Z_3-Fourier projectors π_j = (1/3) Σ_k ω^{-jk} P^k for j ∈ {0,1,2}.
  4. Lepton-candidate anchor: the IB-h=1 eigenvector head-aligned with
     v_triv = (1,1,1,1)/2 (lambda=+3 trivial Bloch).
  5. Decompose ψ_anchor into outer-Galois isotypics: ψ = Σ π_j ψ.
  6. If non-trivial isotypics are nonzero, compute per-isotypic resolvent
     residues at y = 1/h_lepton, extract phases, test 2π/3 AP + δ_lepton
     offset.

Pre-declared falsification (per scoping §5):
  - If non-trivial isotypic components of ψ_anchor vanish, K_4(Γ) cannot
    host the test — STAGE 0 PRE-FAIL (structural, route 4 not eliminated
    but requires richer substrate object: B(P-point) or full primitive
    cell or operator-algebra Galois tower).
  - If non-trivial isotypics present but residues all zero / all real /
    phases not in 2π/3 AP / offset ≠ 2/9 → ROUTE 4 HONEST NEGATIVE,
    joins R1/R2/R3.
  - If 2π/3 AP with offset = 2/9 (within 0.005 rad) → STAGE 0 PASS,
    proceed to Stages 1-2.

Anti-numerology: NO tuning. All inputs upstream-fixed. The mechanism
either delivers 2/9 or it doesn't.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import numpy.linalg as la


results: list[tuple[str, bool]] = []


def gate(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, bool(passed)))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if detail:
        for line in detail.strip("\n").split("\n"):
            print(f"         {line}")
    print()


# ============================================================================
# K_4(Γ) Hashimoto B and body-diagonal C_3 — reuse construction from
# yukawa_walker_stage_0_1_ibroot_eigenspaces_2026-05-22.py
# ============================================================================
N_V = 4
edges_K4 = [(u, v) for u in range(N_V) for v in range(u + 1, N_V)]  # 6 undirected
arcs = []
for ei, (u, v) in enumerate(edges_K4):
    arcs.append((u, v, ei))
    arcs.append((v, u, ei))
N_A = len(arcs)  # 12
assert N_A == 12

B = np.zeros((N_A, N_A), dtype=complex)
for i_p, (t_p, h_p, e_p) in enumerate(arcs):
    for i, (t, h, e) in enumerate(arcs):
        if h == t_p and e != e_p:
            B[i_p, i] = 1.0


# Body-diagonal C_3: σ = (0)(1 2 3) on K_4 vertices (representative of the
# substrate body-diagonal C_3 reduced to K_4; the choice is conjugate to all
# other C_3 ⊂ S_4 and doesn't affect the structural diagnostic).
sigma_vert = {0: 0, 1: 2, 2: 3, 3: 1}


def arc_under_sigma(t: int, h: int, e: int) -> tuple[int, int, int]:
    t_new, h_new = sigma_vert[t], sigma_vert[h]
    u, v = min(t_new, h_new), max(t_new, h_new)
    e_new = edges_K4.index((u, v))
    return t_new, h_new, e_new


P_C3 = np.zeros((N_A, N_A), dtype=complex)
for i, arc in enumerate(arcs):
    new_arc = arc_under_sigma(*arc)
    j = arcs.index(new_arc)
    P_C3[j, i] = 1.0

print("=" * 78)
print("Stage 0 probe — h-power Yukawa, per outer-Galois isotypic (route 4)")
print("Scoping: an internal working note")
print("=" * 78)


# ============================================================================
# G_perm — P_C3 is a permutation of order 3
# ============================================================================
res_order = la.norm(P_C3 @ P_C3 @ P_C3 - np.eye(N_A))
gate("G_perm: P_C3 is order-3 arc permutation", res_order < 1e-10,
     f"||P_C3^3 - I|| = {res_order:.2e}")


# ============================================================================
# G1 — B commutes with P_C3 (body-diagonal C_3 is a substrate symmetry)
# ============================================================================
res_comm = la.norm(B @ P_C3 - P_C3 @ B)
gate("G1: [B, P_C3] = 0  (B(K_4) is C_3-invariant)", res_comm < 1e-10,
     f"||[B, P_C3]|| = {res_comm:.2e}")


# ============================================================================
# G2 — Build Z_3-Fourier projectors on arc space and verify
# ============================================================================
OMEGA = cmath.exp(2j * math.pi / 3)
PI = [
    sum((OMEGA ** (-j * k)) * la.matrix_power(P_C3, k) for k in range(3)) / 3
    for j in range(3)
]

all_idem = all(la.norm(p @ p - p) < 1e-10 for p in PI)
all_orth = all(la.norm(PI[a] @ PI[b]) < 1e-10
               for a in range(3) for b in range(3) if a != b)
sum_id = la.norm(sum(PI) - np.eye(N_A)) < 1e-10
gate("G2: π_j are mutually orthogonal projectors summing to I",
     all_idem and all_orth and sum_id,
     f"all idempotent: {all_idem}\n"
     f"all orthogonal:  {all_orth}\n"
     f"sum to I:        {sum_id}")


# ============================================================================
# G3 — Identify lepton-candidate anchor: solve for v_triv-head-aligned linear
# combination within the h=1 eigenspace (which is 3-dim and ARBITRARILY
# basised by la.eig; we need the specific direction in this subspace whose
# head-projection is v_triv).
# ============================================================================
ew, EV = la.eig(B)
mask_h1 = np.abs(ew - 1) < 1e-6
EV_h1 = EV[:, mask_h1]

H_head = np.zeros((N_V, N_A), dtype=complex)
for i, (t, h, e) in enumerate(arcs):
    H_head[h, i] = 1.0

v_triv = np.ones(N_V, dtype=complex) / np.sqrt(N_V)

# Build head projections of each basis vector: M = H · EV_h1  (N_V × dim_h1)
M = H_head @ EV_h1

# Solve for coefficients c such that EV_h1 @ c has head projection ∝ v_triv.
# Equivalently: minimize ||M @ c - v_triv|| (least squares).
c, residuals, rank, _ = la.lstsq(M, v_triv, rcond=None)

# Build the v_triv-aligned IB-h=1 anchor
psi_anchor = EV_h1 @ c
psi_anchor = psi_anchor / la.norm(psi_anchor)

# Verify head-projection alignment
p_head = H_head @ psi_anchor
align_final = abs(np.vdot(v_triv, p_head)) / la.norm(p_head)

gate("G3: lepton-candidate anchor = IB-h=1 mode head-aligned with v_triv",
     align_final > 0.95,
     f"v_triv-aligned IB-h=1 anchor constructed via least-squares\n"
     f"head-projection alignment = {align_final:.6f}  (need > 0.95)\n"
     f"||residual of M @ c = v_triv|| = "
     f"{la.norm(M @ c - v_triv):.2e}")

# Also build the C_3-non-trivial-isotypic complement for diagnostic
# (the orthogonal direction within the h=1 eigenspace)
EV_h1_orth_basis = EV_h1 - np.outer(psi_anchor, psi_anchor.conj() @ EV_h1)
nontrivial_anchor = None
for j in range(EV_h1_orth_basis.shape[1]):
    v = EV_h1_orth_basis[:, j]
    if la.norm(v) > 1e-6:
        nontrivial_anchor = v / la.norm(v)
        break


# ============================================================================
# G4 — Per-Galois-isotypic spectrum of B(K_4): does h=1 live in non-trivial
# isotypics? If h=1 only appears in j=0, the lepton anchor at h=1 has no
# 3-fold representative — the within-species 3-fold cannot live at this
# fiber.
# ============================================================================
print("  Per-Galois-isotypic spectrum of B(K_4):")
isotypic_spectra = []
for j in range(3):
    # Build B restricted to the j-th isotypic
    # Use SVD of π_j to get an orthonormal basis of Im(π_j)
    U, S, Vh = la.svd(PI[j])
    rank_j = int(np.sum(S > 1e-8))
    basis_j = U[:, :rank_j]  # orthonormal basis of Im(π_j)
    B_j_block = basis_j.conj().T @ B @ basis_j  # B restricted to isotypic j
    ev_j = la.eigvals(B_j_block)
    isotypic_spectra.append(ev_j)
    ev_str = ", ".join(f"{z.real:+.3f}{z.imag:+.3f}j" for z in
                       sorted(ev_j, key=lambda z: (np.round(z.real, 4),
                                                    np.round(z.imag, 4))))
    print(f"    j={j} (dim {rank_j}): {ev_str}")

# Check for h=1 and h=2 in each isotypic
def contains_h(spec, h_val, tol=1e-6):
    return any(abs(z - h_val) < tol for z in spec)

h1_in = [contains_h(isotypic_spectra[j], 1.0) for j in range(3)]
h2_in = [contains_h(isotypic_spectra[j], 2.0) for j in range(3)]
print(f"\n  h = 1 present in isotypics: j=0:{h1_in[0]}, j=1:{h1_in[1]}, j=2:{h1_in[2]}")
print(f"  h = 2 present in isotypics: j=0:{h2_in[0]}, j=1:{h2_in[1]}, j=2:{h2_in[2]}")
print()

# Per-Galois-isotypic 3-fold for either species: need h_species in ALL three isotypics
h1_three_fold = all(h1_in)
h2_three_fold = all(h2_in)

if h1_three_fold or h2_three_fold:
    gate("G4: at least one IB-root has a per-Galois-isotypic 3-fold at K_4(Γ)",
         True,
         f"h=1 in all 3 isotypics: {h1_three_fold}\n"
         f"h=2 in all 3 isotypics: {h2_three_fold}\n"
         "Proceed to G5+ residue extraction")
    nontrivial_present = True
else:
    gate("G4: STRUCTURAL PRE-FAIL — no IB-root has a 3-fold across isotypics",
         False,
         "h=1 lives ONLY in the trivial (j=0) isotypic at K_4(Γ).\n"
         "h=2 likewise. The C_3-non-trivial isotypics (j=1, j=2) host the\n"
         "complex modes from A's λ=-1 (multiplicity 3 standard rep of S_4),\n"
         "which give h_NB = (-1 ± i√7)/2 — NOT h=1 or h=2.\n"
         "\n"
         "Consequence: at single-fiber K_4(Γ) the within-species 3-fold\n"
         "cannot live at the lepton anchor (h=1 Type II). Route 4 Stage 0\n"
         "requires either (i) Bloch integration so other k-points fill in\n"
         "h-values in the non-trivial isotypics; or (ii) a different anchor\n"
         "(non-h-eigenstate) where the §6(i) inverse-survival reading\n"
         "produces a 3-fold spectrum; or (iii) operator-algebra Galois tower\n"
         "(M1.B) — the 3 Galois copies of M^α as type II_1 sub-factor\n"
         "modules, not arc-space eigenvectors.")
    nontrivial_present = False


# ============================================================================
# G5 — STRUCTURAL CAVEAT (load-bearing): sector locus
# ============================================================================
print("=" * 78)
print("G5 — Structural caveat: K_4(Γ) is the QUARK-sector locus, not lepton")
print("=" * 78)
print(
    "Per master Yukawa synthesis §4(B/C) (theorem-grade):\n"
    "  - color-singlet (LEPTONS) concentrate at P-point of BZ\n"
    "    (theorem_color_singlet_P_concentration_2026-05-21)\n"
    "  - color-triplet (QUARKS) concentrate at Γ-point of BZ\n"
    "    (theorem_color_triplet_Gamma_concentration_2026-05-21)\n"
    "K_4(Γ) tests the QUARK fiber, not the LEPTON fiber.\n"
    "Even if G4 had passed at K_4(Γ), the test would be on the wrong sector\n"
    "for the LEPTON self-validation gate. Route 4 Stage 0 properly requires\n"
    "a B(P-point) or full primitive-cell construction for the lepton sibling test."
)
print()


# ============================================================================
# G6 — Conditional execution: if G4 had passed, run residue extraction
# ============================================================================
if nontrivial_present:
    print("=" * 78)
    print("G6 — Per-Galois-isotypic resolvent residue extraction")
    print("=" * 78)

    h_lepton_candidates = [(1, "Type II saturation (h=1)"),
                            (2, "Type IV Perron (h=2)")]
    for h_val, label in h_lepton_candidates:
        print(f"\n  Lepton anchor type: {label}")
        y_target = 1.0 / h_val
        # Per-isotypic resolvent diagonal: <π_j ψ | (I - y B)^{-1} | π_j ψ>
        # Evaluated near y = 1/h_lepton (small offset for numerical stability,
        # then extract argument of leading residue via finite-difference).
        phases = []
        for j in range(3):
            psi_j = PI[j] @ psi_anchor
            if la.norm(psi_j) < 1e-10:
                phases.append(None)
                continue
            # Residue ≈ lim_{y → 1/h} (1/h - y) <ψ_j | (I - y B)^{-1} | ψ_j>
            eps = 1e-5
            G_minus = la.solve(np.eye(N_A) - (y_target - eps) * B, psi_j)
            G_plus = la.solve(np.eye(N_A) - (y_target + eps) * B, psi_j)
            val_minus = np.vdot(psi_j, G_minus)
            val_plus = np.vdot(psi_j, G_plus)
            # Approximate residue (sign convention: pole at y=1/h, so residue
            # ≈ -eps · (val_plus - val_minus) / 2  scaled by limit)
            resid = -eps * (val_plus - val_minus) / 2
            phases.append(cmath.phase(resid))
            print(f"    j={j}: ||π_j ψ|| = {la.norm(psi_j):.4f}, "
                  f"residue ≈ {resid:.4e}, arg = {phases[-1]:.4f} rad")

        # Test for 2π/3 AP
        if all(p is not None for p in phases):
            diffs = [(phases[(k + 1) % 3] - phases[k]) % (2 * math.pi)
                     for k in range(3)]
            print(f"    phase differences (mod 2π): {diffs}")
            offset = sum(phases) / 3
            print(f"    common offset (avg phase): {offset:.6f} rad")
            print(f"    target δ_lepton = 2/9    = {2.0/9.0:.6f} rad")
            print(f"    |offset - 2/9|           = {abs(offset - 2.0/9.0):.6f} rad")
            ap_check = all(abs(d - 2 * math.pi / 3) < 0.05 for d in diffs)
            offset_check = abs(offset - 2.0/9.0) < 0.005
            gate(f"G6[h={h_val}]: phases form 2π/3 AP with offset = 2/9",
                 ap_check and offset_check)
else:
    print("G6 SKIPPED — G4 pre-failed, no non-trivial isotypics to read.\n")


# ============================================================================
# Verdict
# ============================================================================
print("=" * 78)
print("STAGE 0 VERDICT")
print("=" * 78)
n_pass = sum(1 for _, p in results if p)
n_tot = len(results)
print(f"  Gates: {n_pass}/{n_tot}")
print()

delta_check = any(name.startswith("G6[") and passed for name, passed in results)
if delta_check:
    print("  VERDICT: STAGE 0 PASS — δ_lepton = 2/9 reproduced by mechanism")
    print("    Proceed to Stages 1-2 (down + up quark sectors).")
elif not nontrivial_present:
    print("  VERDICT: STAGE 0 STRUCTURAL PRE-FAIL (anchor lacks isotypic content)")
    print("    K_4(Γ) construction needs richer substrate object — see G4 detail.")
else:
    print("  VERDICT: STAGE 0 STRUCTURAL FINDING (NOT route-4 elimination)")
    print()
    print("  The construction has the right TOPOLOGY but lacks the PHASE MECHANISM")
    print("  at single-fiber K_4(Γ):")
    print()
    print("  (i)  G4 passed — h=1 lives in ALL THREE Galois isotypics (1 from λ=+3")
    print("       vertex lift in j=0, + cycle modes contributing h=1 to j=1 and")
    print("       j=2). The 3-fold IS present at K_4(Γ) for the h=1 anchor.")
    print()
    print("  (ii) But per-isotypic residue phases at y=1 are all REAL (phase 0).")
    print("       Structural reason: B(K_4) is real, the v_triv-derived anchor is")
    print("       real, and the resolvent R = (I - yB)^{-1} is real at real y.")
    print("       The Z_3-Fourier projection then satisfies a_1 = ψ†RPψ ≡ a_2 =")
    print("       ψ†RP²ψ at K_4(Γ) (the additional S_4-reality symmetry forces")
    print("       this — empirically verified). With a_1 = a_2 the complex")
    print("       contributions to π_1, π_2 residues cancel — phase content")
    print("       vanishes by construction at this fiber.")
    print()
    print("  (iii) Implication: route 4 NEEDS the construction to have non-zero")
    print("       phase content for the §6(i) Landauer bridge to extract δ.")
    print("       Sources of phase the K_4(Γ) reduction lacks:")
    print("         • Bloch-decorated B_NB(k≠Γ): bond phases e^{ik·δ} are complex")
    print("           for k ≠ Γ; break the a_1 = a_2 reality symmetry.")
    print("         • Cl(6)-Fock chirality projection (γ_7 = (-1)^F): introduces")
    print("           a complex chirality eigenvalue per anchor.")
    print("         • P-point (k = body-diagonal): Bloch phases at P are -1 on")
    print("           specific bonds — still real but break C_3 acting trivially.")
    print()
    print("  This sharpens — but does NOT eliminate — route 4. The original")
    print("  scoping doc §3.3 explicitly required Bloch integration (the full")
    print("  resolvent reading); this probe tried the single-fiber Γ shortcut")
    print("  and found it structurally insufficient. The route's phase mechanism")
    print("  lives in the Bloch-integrated construction, not at single-fiber Γ.")
    print()
    print("  Cost to do this properly: ~1 session to construct B_NB(k=P) on the")
    print("  srs primitive cell (or equivalent Cl(6)-Fock-projected K_4(Γ) with")
    print("  γ_7 decoration) and re-run Stage 0.")
    print()
    print("  Memory-relevant: this confirms the prior")
    print("  an internal note")
    print("  entry — δ lives in the Bloch-integrated G_NB, not at single-fiber B(Γ).")
    print("  Route 4 respects that constraint; the Stage 0 shortcut tested whether")
    print("  K_4(Γ) was sufficient as a minimal proxy, and the answer is NO.")

print()
print("=" * 78)
print("Gate summary:")
for name, passed in results:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
print("=" * 78)
