#!/usr/bin/env python3
"""
proofs/foundations/b4_a4_dirac_index_probe.py

PROBE: Compute ind(D(k)) for the substrate Dirac D = Σ_e γ^e ⊗ L_e at the
       high-symmetry k-points P and Γ on srs. A non-zero index would seed
       ADOPTED-B3 parity convention (sub-question (a) of the (Z/2)^3
       labeling ambiguity).

CONTEXT
-------
Predecessors (theorem-grade or formalism-grade):
- `docs/forward_constructions/forward_construction_substrate_atiyah_singer.md` — D and McKean-
  Singer formalism defined; first-pass conjecture ind(D(P)) = ±8.
- `docs/forward_constructions/forward_construction_substrate_lichnerowicz.md` — D² = n·I + R_sub
  closed at theorem grade with τ(R_sub²) = n(n-1) = 30 for srs (n=6).
  for this probe; gate criteria and structural risks documented.

Companion to `b4_color_vram_{p,gamma}_commutant_probe.py` (which refuted
V_Ram as the SU(3)_color seed). A.4 is the recommended pivot for ADOPTED-B3
parity, a different sub-question (chirality convention, not color).

CONSTRUCTION
------------
Substrate Dirac per Bloch fiber:

  D(k) = Σ_{a=1..6} γ^{e_a} ⊗ L_{e_a}(k)   on   ℂ³² = S ⊗ ℂ⁴_atoms.

- γ^{e_a} (a = 1..6): Cl(6,0) generators on the 8-dim Brauer-Weyl spinor
  S, built from 3 fermionic creation/annihilation operators on a Fock
  space (per `proofs/gauge/cl8_verification.py`).
- L_{e_a}(k): 4×4 Hermitian unitary involution acting as
    |s_a⟩ ↦ exp(2πi k·δ_a) |t_a⟩
    |t_a⟩ ↦ exp(-2πi k·δ_a) |s_a⟩
    |v⟩   ↦ |v⟩   for v ∉ {s_a, t_a}
  where (s_a, t_a, δ_a) are the source, target, and primitive-cell offset
  of the a-th undirected edge of srs. By construction L_{e_a}² = I, L_{e_a}
  Hermitian.
- Chirality grading γ_5 = γ¹γ²γ³γ⁴γ⁵γ⁶ (Cl(6,0) chirality, the (-1)^F
  fermionic-parity operator on S). γ_5² = I. {γ_5, γ^a} = 0, so
  {γ_5 ⊗ I_4, D(k)} = 0 — D(k) is chirality-odd.

For each k, D(k) is Hermitian, anti-commutes with the chirality grading,
and (per the substrate Lichnerowicz formula, theorem-grade) satisfies
D(k)² = n · I + R_sub(k) where n = 6 = |E|.

INDEX
-----
ind(D(k)) = dim ker(D(k))|_{S_+} − dim ker(D(k))|_{S_-}
          = signed count of zero modes of D(k) graded by γ_5.

For ind(D(k)) ≠ 0, D(k) needs zero eigenvalues. Since D(k)² = n I +
R_sub(k), zero eigenvalues require R_sub(k) eigenvalues ≤ -n. The probe
also reports min(eig R_sub(k)) at each k as a structural diagnostic.

GATE STATUS
-----------
CAS verification only. Outputs:
- ind(D(P)), ind(D(Γ)), and the sub-block decomposition.
- Verification of D Hermiticity, {γ_5, D} = 0, D² = n I + R_sub at Bloch
  level (sanity).
- min(eig R_sub(k)) at each k (structural diagnostic).
- Verdict for ADOPTED-B3 parity convention.

Run with:
    PYTHONPATH=. python3 proofs/foundations/b4_a4_dirac_index_probe.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import find_bonds, N_ATOMS

TOL = 1e-9


# =====================================================================
# Step 1: Build Cl(6,0) generators γ^a (a=1..6) on the 8-dim spinor S
# =====================================================================
# Reuses the Brauer-Weyl construction of `proofs/gauge/cl8_verification.py`
# to ensure consistency with the framework's existing Cl(6) infrastructure.

def build_fock_creation_ops():
    """Three fermionic creation operators on the 8-dim Fock space ℂ²×ℂ²×ℂ².
    Returns a list [a_1†, a_2†, a_3†] of 8×8 complex matrices."""
    dim = 8
    a_dag = [np.zeros((dim, dim), dtype=complex) for _ in range(3)]
    for state in range(dim):
        bits = [(state >> j) & 1 for j in range(3)]
        for i in range(3):
            if bits[i] == 0:
                new_state = state | (1 << i)
                sign = (-1) ** sum(bits[j] for j in range(i))
                a_dag[i][new_state, state] = sign
    return a_dag


def build_cl6_generators():
    """Six Hermitian Cl(6,0) generators γ¹..γ⁶ on the 8-dim spinor.
    γ_{2i-1} = a_i + a_i†,   γ_{2i} = i(a_i† - a_i)   (Hermitian, square to +I,
    pairwise anti-commuting)."""
    a_dag = build_fock_creation_ops()
    gammas = []
    for i in range(3):
        a = a_dag[i].conj().T
        ad = a_dag[i]
        gammas.append(ad + a)              # γ_{2i-1}
        gammas.append(1j * (ad - a))       # γ_{2i}
    # Sanity
    I8 = np.eye(8, dtype=complex)
    for mu in range(6):
        assert la.norm(gammas[mu] @ gammas[mu] - I8) < TOL, f"γ_{mu+1}² ≠ I"
        assert la.norm(gammas[mu] - gammas[mu].conj().T) < TOL, f"γ_{mu+1} not Hermitian"
    for mu in range(6):
        for nu in range(mu + 1, 6):
            assert la.norm(gammas[mu] @ gammas[nu] + gammas[nu] @ gammas[mu]) < TOL, \
                f"γ_{mu+1}, γ_{nu+1} do not anti-commute"
    return gammas


def build_chirality(gammas):
    """γ_5 = -i·γ¹γ²γ³γ⁴γ⁵γ⁶ — Cl(6) chirality on S (the (-1)^F operator).
    Convention chosen so γ_5² = I and γ_5 Hermitian."""
    G = np.eye(8, dtype=complex)
    for g in gammas:
        G = G @ g
    chirality = (-1j) ** 3 * G
    I8 = np.eye(8, dtype=complex)
    assert la.norm(chirality @ chirality - I8) < TOL, "γ_5² ≠ I"
    assert la.norm(chirality - chirality.conj().T) < TOL, "γ_5 not Hermitian"
    # γ_5 anti-commutes with each γ^a
    for mu, g in enumerate(gammas):
        assert la.norm(chirality @ g + g @ chirality) < TOL, f"{{γ_5, γ_{mu+1}}} ≠ 0"
    return chirality


# =====================================================================
# Step 2: enumerate the 6 undirected edges of srs primitive cell
# =====================================================================
# `find_bonds()` returns 12 directed bonds; pair them into 6 undirected
# edges for the Cl(6) generator assignment.

def enumerate_undirected_edges():
    """Return list of 6 (s, t, delta) tuples — one per undirected edge.
    For each undirected edge {s, t} with offset δ from s to t, the
    reverse direction has offset -δ from t to s. Both appear in
    find_bonds(); we keep the canonical orientation s < t (or s = t with
    lex-min cell)."""
    directed = find_bonds()
    seen = set()
    undirected = []
    for src, tgt, cell in directed:
        # Canonical: (s, t, cell) where s < t, or s = t with lex-min cell
        if src < tgt:
            key = (src, tgt, tuple(cell))
        elif src > tgt:
            key = (tgt, src, tuple(-c for c in cell))
        else:
            # Self-loop; canonicalize cell sign
            c = tuple(cell)
            c_neg = tuple(-x for x in cell)
            key = (src, tgt, min(c, c_neg))
        if key not in seen:
            seen.add(key)
            undirected.append(key)
    assert len(undirected) == 6, f"Expected 6 undirected edges, got {len(undirected)}"
    return undirected


# =====================================================================
# Step 3: build L_e(k) at a Bloch fiber
# =====================================================================

def build_L_edge_bloch(s, t, delta, k_frac):
    """4×4 Hermitian involution L_{e_a}(k) on the atom basis:
        |s⟩ ↦ exp(2πi k·δ)|t⟩,  |t⟩ ↦ exp(-2πi k·δ)|s⟩,  |v⟩ ↦ |v⟩ otherwise.
    Action of an undirected edge with Bloch phase given by the (s → t)
    cell offset δ."""
    L = np.eye(N_ATOMS, dtype=complex)
    phase = np.exp(2j * np.pi * np.dot(np.asarray(k_frac, dtype=float), np.asarray(delta, dtype=float)))
    if s != t:
        # Off-diagonal: replace identity blocks at (s,t) with the σ_x-with-phase 2×2 block.
        L[s, s] = 0
        L[t, t] = 0
        L[t, s] = phase
        L[s, t] = np.conj(phase)
    else:
        # Self-loop: a single phase factor on the diagonal.  Hermiticity forces
        # phase = ±1, and involutivity also forces phase² = 1, so phase ∈ {±1}.
        # Pick +1 by convention (no edge in srs is a self-loop).
        L[s, s] = 1.0
    # Sanity
    assert la.norm(L - L.conj().T) < TOL, "L_e not Hermitian"
    assert la.norm(L @ L - np.eye(N_ATOMS, dtype=complex)) < TOL, "L_e² ≠ I"
    return L


def build_substrate_dirac(k_frac, gammas, edges):
    """D(k) = Σ_{a=1..6} γ^a ⊗ L_{e_a}(k)   (32 × 32, Hermitian)."""
    D = np.zeros((32, 32), dtype=complex)
    for a, (s, t, delta) in enumerate(edges):
        L_a = build_L_edge_bloch(s, t, delta, k_frac)
        D += np.kron(gammas[a], L_a)
    return D


# =====================================================================
# Step 4: per-fiber probe
# =====================================================================

def probe_at_k(k_frac, k_label, gammas, chirality, edges, n=6):
    """Run the full per-Bloch-fiber probe at k = k_frac, label = k_label."""
    print()
    print("=" * 72)
    print(f"k = {k_label}    fractional = {tuple(k_frac)}")
    print("=" * 72)

    D = build_substrate_dirac(k_frac, gammas, edges)
    G5 = np.kron(chirality, np.eye(N_ATOMS, dtype=complex))   # γ_5 ⊗ I_4

    # Sanity checks
    herm_residual = la.norm(D - D.conj().T)
    chir_anti = la.norm(D @ G5 + G5 @ D)
    print(f"  ||D - D†||                     = {herm_residual:.2e}")
    print(f"  ||{{γ_5⊗I_4, D}}||                = {chir_anti:.2e}")
    assert herm_residual < TOL, "D(k) is not Hermitian"
    assert chir_anti < TOL, "D(k) does not anti-commute with γ_5⊗I_4"

    # Lichnerowicz: D² = n·I + R_sub(k)
    D2 = D @ D
    R_sub = D2 - n * np.eye(32, dtype=complex)
    R_sub_herm = la.norm(R_sub - R_sub.conj().T)
    print(f"  ||R_sub - R_sub†||             = {R_sub_herm:.2e}")
    assert R_sub_herm < TOL, "R_sub(k) is not Hermitian"

    R_eigs = np.sort(la.eigvalsh(R_sub))
    print(f"  R_sub(k) eigenvalue range      = [{R_eigs[0]:+.4f}, {R_eigs[-1]:+.4f}]")
    print(f"  Tr(R_sub²)/dim                 = {np.real(np.trace(R_sub @ R_sub)) / 32:.4f}    (substrate τ-norm² scale: {n*(n-1)/32:.4f} on full F_inv(E))")
    if R_eigs[0] <= -n + 1e-6:
        print(f"  ✓ min(R_sub) ≤ -n = -{n}: D(k) zero modes are admissible")
    else:
        print(f"  ✗ min(R_sub) = {R_eigs[0]:+.4f} > -n = -{n}: D(k)² > 0 strictly,")
        print(f"      no zero modes possible at this k. ind(D(k)) = 0 forced.")

    # Diagonalize D(k); identify near-zero eigenvalues
    D_eigs, D_vecs = la.eigh(D)   # D is Hermitian
    print(f"  D(k) eigenvalue range          = [{D_eigs[0]:+.4f}, {D_eigs[-1]:+.4f}]")
    print(f"  D(k) |eigenvalue| spectrum     = " +
          ", ".join(f"{abs(ev):.3f}" for ev in sorted(set(round(abs(ev), 4) for ev in D_eigs))[:8]) + " ...")

    KER_TOL = 1e-7
    ker_idx = [i for i, ev in enumerate(D_eigs) if abs(ev) < KER_TOL]
    print(f"  dim ker D(k) (|λ| < {KER_TOL:g})    = {len(ker_idx)}")

    # Chirality decompose the kernel
    ind = 0
    if ker_idx:
        ker_basis = D_vecs[:, ker_idx]
        # γ_5 should commute with the kernel projector and split it
        chir_block = ker_basis.conj().T @ G5 @ ker_basis
        chir_evs = la.eigvalsh(chir_block)
        n_plus = sum(1 for ev in chir_evs if ev > 0.5)
        n_minus = sum(1 for ev in chir_evs if ev < -0.5)
        n_other = len(chir_evs) - n_plus - n_minus
        ind = n_plus - n_minus
        print(f"  kernel chirality split: +{n_plus},  -{n_minus},  other {n_other}")
        print(f"  ind(D(k)) = +{n_plus} - {n_minus} = {ind:+d}")
    else:
        print(f"  No zero modes ⇒ ind(D(k)) = 0")

    return ind, R_eigs[0]


# =====================================================================
# Main
# =====================================================================

print("=" * 72)
print(" b4_a4_dirac_index_probe — substrate Dirac index at high-symmetry k")
print("=" * 72)

gammas = build_cl6_generators()
chirality = build_chirality(gammas)
edges = enumerate_undirected_edges()

print()
print("Cl(6,0) generators γ¹..γ⁶ built (Brauer-Weyl on 8-dim Fock).")
print(f"Chirality γ_5 = -i·γ¹γ²γ³γ⁴γ⁵γ⁶ verified ({{γ_5, γ^a}} = 0).")
print(f"Six undirected edges of srs primitive cell:")
for a, (s, t, delta) in enumerate(edges):
    print(f"  e_{a+1}: atom {s} ↔ atom {t},  cell offset {delta}")

K_P = (0.25, 0.25, 0.25)
K_GAMMA = (0.0, 0.0, 0.0)

ind_P, min_R_P = probe_at_k(K_P, "P", gammas, chirality, edges)
ind_G, min_R_G = probe_at_k(K_GAMMA, "Γ", gammas, chirality, edges)


# =====================================================================
# Verdict
# =====================================================================

print()
print("=" * 72)
print(" Verdict")
print("=" * 72)
print()
print(f"  ind(D(P)) = {ind_P:+d}    min(eig R_sub(P)) = {min_R_P:+.4f}")
print(f"  ind(D(Γ)) = {ind_G:+d}    min(eig R_sub(Γ)) = {min_R_G:+.4f}")
print()

if ind_P != 0 or ind_G != 0:
    print("  ✓ POSITIVE — at least one high-symmetry k has nonzero substrate")
    print("    Dirac index. The substrate produces a chirality-odd integer,")
    print("    seeding ADOPTED-B3 parity convention (sub-question (a)) as a")
    print("    substrate-derived canonical sign rather than a free Z_2.")
    print()
    print("    ADOPTED-B3 8-fold ambiguity → 4-fold (parity convention fixed).")
    print("    Color (sub-question (b)) and up/down (sub-question (c)) remain")
    print("    open; full ADOPTED-B3 closure still requires their sub-probes.")
else:
    print("  ✗ NEGATIVE — both ind(D(P)) and ind(D(Γ)) are zero.")
    print()
    print(f"    Per Lichnerowicz: D(k)² = n·I + R_sub(k), n = 6. For zero modes")
    print(f"    we need min(eig R_sub(k)) ≤ -n. Observed:")
    print(f"      min eig R_sub(P) = {min_R_P:+.4f}  (need ≤ -6)")
    print(f"      min eig R_sub(Γ) = {min_R_G:+.4f}  (need ≤ -6)")
    print()
    if min_R_P > -6 + 1e-3 and min_R_G > -6 + 1e-3:
        print("    The substrate Lichnerowicz inequality is STRICT at both k-points")
        print("    — D(k)² is gapped above zero, so no zero modes are possible at")
        print("    these high-symmetry k. The per-fiber Dirac index cannot seed")
        print("    ADOPTED-B3 parity through this mechanism.")
    print()
    print("    A.4 per-fiber index closes NEGATIVE for ADOPTED-B3 parity at the")
    print("    P, Γ k-points. Sequel options (per scoping doc §6):")
    print("      • η-invariant (spectral asymmetry, can be nonzero without zero modes)")
    print("      • Family-index over BZ (Chern winding of the Bloch bundle)")
    print("      • Accept parity convention as external (Route-iv-like for sub-(a))")

print()
print("=" * 72)
print(" OK: b4_a4_dirac_index_probe complete.")
print("=" * 72)
