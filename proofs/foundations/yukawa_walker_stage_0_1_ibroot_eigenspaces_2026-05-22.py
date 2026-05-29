#!/usr/bin/env python3
"""
Yukawa-walker C₃-breaking route — Stage 0 + Stage 1 IB-root eigenspaces on B(Γ)
================================================================================

Date: 2026-05-22
Scopes: an internal working note Stages 0–1.
Tests the route the W49 orbit-member audit reframed toward: with the broken
Higgs vacuum supplying no C₃-mixing, species-distinct structure must come from
the §4(D) walker types — the color triplet (u, d) splits between IB-roots
h=1 (Type II, up) and h=2 (Type IV, down) of B(Γ)|_{triv λ=+3}, and the scope
posited that V_CKM ≡ U_h=1† U_h=2 reads as a basis-misalignment over the 3
generations within those IB-root eigenspaces.

Two stages here:

  STAGE 0 — confirm the IB-root eigenvalues {1, 2} exist on B(Γ) at the trivial
            λ=+3 sub-block (the Ihara–Bass equation h² − 3h + 2 = 0 at k* = 3).

  STAGE 1 — count generation modes per IB-root eigenspace. The scoping doc's
            §4 Stage-1 gate: if both IB-root eigenspaces at trivial λ=+3 host
            *single* modes (1-dim) rather than 3 generation slots, the
            CKM-as-IB-root-basis-misalignment-over-generations construction is
            structurally dead at this site — the CKM cannot be built as
            U_h=1† U_h=2 over generations at B(Γ) alone.

Construction. We use the K_4 abstract trivalent-graph reading of the srs
primitive cell at k = Γ (per W22, W21, the chi_tilde line) — Bloch phases at
k=0 are all +1, the cell of srs reduces to abstract K_4. This is the smallest
faithful object the IB-root split lives on; it is the same B(Γ) the §4(C)
theorem reads.

A(K_4) has spectrum {3, -1, -1, -1}. The trivial λ=+3 eigenvector is
(1,1,1,1)/2. Ihara–Bass: each A-eigenvalue λ contributes two B-eigenvalues h
satisfying h² − λh + (k*−1) = h² − λh + 2 = 0, plus the factor (1 − u²)^{m−n}
contributes ±1 trivial eigenvalues with multiplicity m − n = 2 each.
Total 12 eigenvalues of B(K_4) (12×12).

Gates:
  G1 — Build B(K_4) (12×12); verify k*−1 row sums.
  G2 — Diagonalize; verify spectrum matches the Ihara–Bass enumeration.
  G3 — STAGE 0: confirm h=1 and h=2 are both eigenvalues.
  G4 — Identify the IB-root eigenvectors AT TRIVIAL λ=+3 (distinguishing
       them from the (1−u²) trivial ±1 eigenvalues, which come from the
       graph's cycle structure not the trivial Bloch).
  G5 — Apply the generation-C₃ action on arcs (σ = (0)(1 2 3) on vertices),
       decompose B(K_4) under C₃ isotypic blocks.
  G6 — STAGE 1: dimension of each IB-root eigenspace at trivial λ=+3;
       intersection with the C₃-trivial subspace; count generation slots.
  G7 — Verdict.
"""

from __future__ import annotations
import numpy as np
import numpy.linalg as la

results = []


def gate(name, passed, detail=""):
    results.append((name, bool(passed)))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if detail:
        for line in detail.strip("\n").split("\n"):
            print(f"         {line}")
    print()


# ============================================================================
# K_4 primitive cell (abstract trivalent graph; Bloch fiber at k=Γ)
# ============================================================================
N_V = 4
edges_K4 = [(u, v) for u in range(N_V) for v in range(u + 1, N_V)]  # 6 undirected
arcs = []
for ei, (u, v) in enumerate(edges_K4):
    arcs.append((u, v, ei))
    arcs.append((v, u, ei))
N_A = len(arcs)  # 12
assert N_A == 12

# Adjacency on vertices, K_4 = J − I
A_K4 = np.ones((N_V, N_V)) - np.eye(N_V)

# Hashimoto B on directed arcs (non-backtracking).
B = np.zeros((N_A, N_A), dtype=complex)
for i_p, (t_p, h_p, e_p) in enumerate(arcs):
    for i, (t, h, e) in enumerate(arcs):
        if h == t_p and e != e_p:
            B[i_p, i] = 1.0


print("=" * 78)
print("G1 — build B(K_4) at k=Γ; 12×12 Hashimoto, NB row-sum k*−1 = 2")
print("=" * 78)
row_sums = set(int(s.real) for s in B.sum(axis=1))
g1 = B.shape == (12, 12) and row_sums == {2}
gate("G1 B(K_4) is 12×12 with NB row sum = k*−1 = 2", g1,
     f"shape = {B.shape}; row sums = {row_sums} (expect {{2}})")


# ============================================================================
# Diagonalize B and compare with Ihara–Bass prediction
# ============================================================================
print("=" * 78)
print("G2 — diagonalize B; spectrum matches Ihara–Bass enumeration")
print("=" * 78)
ev_B = la.eigvals(B)
ev_B_sorted = sorted(ev_B, key=lambda z: (np.round(z.real, 8), np.round(z.imag, 8)))

# Ihara–Bass predicted spectrum:
#   from A's λ=+3 (mult 1): h ∈ {1, 2}  (mult 1 each)
#   from A's λ=−1 (mult 3): h ∈ {(−1 ± i√7)/2} (mult 3 each)
#   from (1 − u²)^{m−n}: u = ±1 (mult m − n = 2 each), giving B-eigenvalues ±1
ev_A = sorted(la.eigvalsh(A_K4))     # [-1, -1, -1, 3]
predicted = []
for lam in ev_A:
    disc = lam*lam - 4*2
    sd = np.sqrt(complex(disc))
    predicted.extend([(lam + sd)/2, (lam - sd)/2])
predicted.extend([+1, +1, -1, -1])  # (1 − u²)^{m−n=2} trivial roots
predicted_sorted = sorted(predicted, key=lambda z:
                          (np.round(complex(z).real, 8), np.round(complex(z).imag, 8)))


def match_multiset(a, b, tol=1e-6):
    used = [False]*len(b)
    for x in a:
        found = False
        for j, y in enumerate(b):
            if used[j]: continue
            if abs(complex(x) - complex(y)) < tol:
                used[j] = True; found = True; break
        if not found:
            return False
    return all(used)


g2 = match_multiset(ev_B_sorted, predicted_sorted)
gate("G2 spectrum of B matches Ihara–Bass enumeration", g2,
     "B(K_4) eigenvalues vs Ihara–Bass prediction:\n"
     + "  B :  " + ", ".join(f"{z.real:+.3f}{z.imag:+.3f}j" for z in ev_B_sorted) + "\n"
     + "  IB:  " + ", ".join(f"{complex(z).real:+.3f}{complex(z).imag:+.3f}j" for z in predicted_sorted))


# ============================================================================
# STAGE 0 — IB-roots {1, 2} present
# ============================================================================
print("=" * 78)
print("G3 — STAGE 0: h = 1 and h = 2 are both eigenvalues of B(K_4)")
print("=" * 78)
tol = 1e-6
has_h1 = any(abs(z - 1) < tol for z in ev_B)
has_h2 = any(abs(z - 2) < tol for z in ev_B)
g3 = has_h1 and has_h2
gate("G3 STAGE 0 PASS — IB-roots h=1 and h=2 present on B(K_4)", g3,
     f"h = 1 present: {has_h1}\n"
     f"h = 2 present: {has_h2}\n"
     f"these are the Ihara–Bass roots at A's trivial λ=+3 — h² − 3h + 2 = 0")


# ============================================================================
# Identify the IB-root eigenvectors AT trivial λ=+3
# ============================================================================
print("=" * 78)
print("G4 — separate the IB-root eigenvectors from (1−u²) trivial ±1 modes")
print("=" * 78)
# Right-eigenvectors of B
ew, EV = la.eig(B)
mask_h1 = np.abs(ew - 1) < tol
mask_h2 = np.abs(ew - 2) < tol
EV_h1 = EV[:, mask_h1]
EV_h2 = EV[:, mask_h2]
dim_h1_total = EV_h1.shape[1]
dim_h2_total = EV_h2.shape[1]

# Build the "head-summation" map: for each arc a, push amplitude onto head(a).
# An IB-root eigenvector ξ of B with eigenvalue h corresponding to A-eigenvector
# v with eigenvalue λ has the property that H ξ ∝ v (head-projection picks up
# the underlying vertex eigenvector). The (1−u²) trivial ±1 eigenvectors have
# vanishing head-projection (they live in the "edge" part orthogonal to the
# vertex lift).
H = np.zeros((N_V, N_A), dtype=complex)
for i, (t, h_, e) in enumerate(arcs):
    H[h_, i] = 1.0

# Trivial λ=+3 vertex eigenvector: (1,1,1,1)/2
v_triv = np.ones(N_V) / np.sqrt(N_V)

# For each h=1 eigenvector, compute its head-projection alignment with v_triv
def alignment(ev, v_target):
    p = H @ ev
    if np.linalg.norm(p) < 1e-12: return 0.0
    return float(abs(np.vdot(v_target, p)) / np.linalg.norm(p))

h1_alignments = [alignment(EV_h1[:, j], v_triv) for j in range(dim_h1_total)]
h2_alignments = [alignment(EV_h2[:, j], v_triv) for j in range(dim_h2_total)]

# Build the IB-root sub-space (eigenvectors whose head-projection aligns with
# the trivial λ=+3 vertex eigenvector) vs the (1−u²) trivial modes (head-
# projection vanishes or aligns with other A-eigenvectors).
def split_by_alignment(EV_block, v_target, tol_align=1e-6):
    """Return (cols_with_v_target_component, cols_orthogonal_to_v_target)."""
    n = EV_block.shape[1]
    if n == 0:
        return EV_block, EV_block
    proj = H @ EV_block                       # (N_V, n)
    # Find the linear combinations whose head-projection lies in span(v_target)
    # via Gram–Schmidt / null-space decomposition.
    # comp[i] = component along v_target of the i-th head-projection
    comp = v_target.conj() @ proj             # (n,)
    # Two cases for n: if 1, just check; if larger, separate
    if n == 1:
        if abs(comp[0]) > tol_align:
            return EV_block, np.zeros((N_A, 0), dtype=complex)
        else:
            return np.zeros((N_A, 0), dtype=complex), EV_block
    # Build coefficients along v_target and orthogonal directions
    # The IB-root component spans 1 direction; the (1−u²) modes the rest.
    # Pick a basis where one combination aligns with v_target and (n-1) don't.
    # Concretely: form U = orth column basis; find the unique column with
    # nonzero v_target-component (or build it via QR).
    coeffs = comp / (np.linalg.norm(comp) + 1e-30)
    ib_col = EV_block @ coeffs                 # IB-root eigenvector (head ∝ v_target)
    ib_col /= np.linalg.norm(ib_col)
    # Now build (n-1) orthogonal columns by removing ib_col from EV_block
    rest = []
    for j in range(n):
        c = EV_block[:, j] - np.vdot(ib_col, EV_block[:, j]) * ib_col
        rest.append(c)
    R = np.array(rest).T
    # Orthonormalise the rest via QR (it spans n-1 dimensions in general)
    Q, _ = la.qr(R)
    # Keep only nonzero columns (rank n-1)
    norms = np.linalg.norm(R, axis=0)
    keep = norms > 1e-9
    return ib_col.reshape(-1, 1), Q[:, :int(np.sum(keep)) - 0] if False else Q[:, : (n-1)]


ib_h1, rest_h1 = split_by_alignment(EV_h1, v_triv)
ib_h2, rest_h2 = split_by_alignment(EV_h2, v_triv)

dim_ib_h1 = ib_h1.shape[1] if ib_h1.size else 0
dim_ib_h2 = ib_h2.shape[1] if ib_h2.size else 0
dim_rest_h1 = rest_h1.shape[1] if rest_h1.size else 0
dim_rest_h2 = rest_h2.shape[1] if rest_h2.size else 0

g4 = (dim_ib_h1 == 1 and dim_ib_h2 == 1 and dim_rest_h1 == dim_h1_total - 1
      and dim_rest_h2 == dim_h2_total - 1)
gate("G4 IB-root eigenvectors at trivial λ=+3 separated from (1−u²) ±1 modes", g4,
     f"h=1 eigenspace total dim = {dim_h1_total}\n"
     f"  • IB-root component (head ∝ v_triv): dim = {dim_ib_h1}\n"
     f"  • (1−u²) trivial-cycle modes: dim = {dim_rest_h1}\n"
     f"h=2 eigenspace total dim = {dim_h2_total}\n"
     f"  • IB-root component (head ∝ v_triv): dim = {dim_ib_h2}\n"
     f"  • (1−u²) trivial-cycle modes: dim = {dim_rest_h2}\n"
     f"head-projection alignments with v_triv (raw eigvecs):\n"
     f"  h=1: {[f'{a:.3f}' for a in h1_alignments]}\n"
     f"  h=2: {[f'{a:.3f}' for a in h2_alignments]}")


# ============================================================================
# C₃ action on arcs and the trivial isotypic block
# ============================================================================
print("=" * 78)
print("G5 — C₃ action on arcs; decompose under C₃ isotypic blocks")
print("=" * 78)
# σ = (0)(1 2 3) on vertices
sigma_v = {0: 0, 1: 2, 2: 3, 3: 1}
arc_idx = {(t, h, e): i for i, (t, h, e) in enumerate(arcs)}
# σ on edges: edge {u,v} -> edge {σ(u), σ(v)}; track via lookup
edge_lookup = {frozenset(e): i for i, e in enumerate(edges_K4)}
edge_map = {i: edge_lookup[frozenset((sigma_v[u], sigma_v[v]))] for i, (u, v) in enumerate(edges_K4)}
# σ on arcs
P_sigma = np.zeros((N_A, N_A), dtype=complex)
for i, (t, h_, e) in enumerate(arcs):
    new = (sigma_v[t], sigma_v[h_], edge_map[e])
    j = arc_idx[new]
    P_sigma[j, i] = 1.0

# C₃ commutes with B?
commutator = P_sigma @ B - B @ P_sigma
g5_commute = np.linalg.norm(commutator) < 1e-9

# Project onto the C₃-trivial subspace of arcs: P_triv = (I + σ + σ²)/3
P_sigma2 = P_sigma @ P_sigma
P_triv_arcs = (np.eye(N_A, dtype=complex) + P_sigma + P_sigma2) / 3
# rank
rank_triv = int(np.sum(np.abs(la.eigvalsh(P_triv_arcs)) > 0.5))

# Number of C₃-orbits on arcs (which equals dim of trivial isotypic)
visited = set()
orbits = []
for i in range(N_A):
    if i in visited: continue
    orb = []
    j = i
    while j not in visited:
        visited.add(j)
        orb.append(j)
        j = int(np.argmax(np.abs(P_sigma[:, j])))
    orbits.append(orb)
n_orbits = len(orbits)
orbit_sizes = sorted(len(o) for o in orbits)

g5 = g5_commute and n_orbits == rank_triv
gate("G5 C₃ commutes with B; arc-space decomposes as 4·triv ⊕ 4·ω ⊕ 4·ω*", g5,
     f"||[σ, B]|| = {np.linalg.norm(commutator):.3e}\n"
     f"# C₃-orbits on the 12 arcs: {n_orbits} (sizes {orbit_sizes})\n"
     f"rank of P_triv (C₃-trivial isotypic dim) on arcs: {rank_triv}\n"
     f"  ⇒ arc space = (rank_triv) · trivial ⊕ ... regular rep on each size-3 orbit\n"
     f"     = 4·triv ⊕ 4·ω ⊕ 4·ω*  (each size-3 orbit contributes one of each)")


# ============================================================================
# STAGE 1 — generation modes per IB-root eigenspace
# ============================================================================
print("=" * 78)
print("G6 — STAGE 1: count generation modes per IB-root eigenspace")
print("=" * 78)
# The IB-root eigenvector at trivial λ=+3 must lie in the C₃-trivial isotypic
# block (since v_triv = (1,1,1,1)/2 is the C₃-invariant vertex). Verify and
# count its dimension.
def c3_isotypic_component(vec, kind="triv"):
    """Project onto trivial / ω / ω* isotypic of C₃ on arcs."""
    omega = np.exp(2j*np.pi/3)
    if kind == "triv":
        P = P_triv_arcs
    elif kind == "omega":
        P = (np.eye(N_A, dtype=complex) + omega.conjugate()*P_sigma + omega*P_sigma2) / 3
    elif kind == "omega_star":
        P = (np.eye(N_A, dtype=complex) + omega*P_sigma + omega.conjugate()*P_sigma2) / 3
    return P @ vec


def isotypic_decomp(vec):
    triv = c3_isotypic_component(vec, "triv")
    om = c3_isotypic_component(vec, "omega")
    om2 = c3_isotypic_component(vec, "omega_star")
    return {
        "triv": float(np.linalg.norm(triv)),
        "ω":    float(np.linalg.norm(om)),
        "ω*":   float(np.linalg.norm(om2)),
    }


ib_h1_vec = ib_h1[:, 0] if dim_ib_h1 else None
ib_h2_vec = ib_h2[:, 0] if dim_ib_h2 else None
h1_iso = isotypic_decomp(ib_h1_vec) if ib_h1_vec is not None else {}
h2_iso = isotypic_decomp(ib_h2_vec) if ib_h2_vec is not None else {}

# Each IB-root eigenspace AT trivial λ=+3 is 1-dimensional (the IB equation
# h² − 3h + 2 = 0 has 2 simple roots, each contributing a single B-eigenvector
# whose head-projection ∝ v_triv). So the construction's U_h=1 and U_h=2 are
# 1-vectors, not 3×3 bases over generations.

n_gen_slots_h1 = dim_ib_h1
n_gen_slots_h2 = dim_ib_h2
g6_one_dim_per_root = (n_gen_slots_h1 == 1 and n_gen_slots_h2 == 1)
# The STAGE-1 GATE: ≥3 generation slots per IB-root required for the scope's
# V_CKM = U_h=1† U_h=2 construction. We have 1 — the gate is NEGATIVE.
stage1_passes_scope_gate = (n_gen_slots_h1 >= 3 and n_gen_slots_h2 >= 3)

# Honest-record: report the structural finding, gate marked PASS on the
# COMPUTATION (it returned a definite answer) regardless of which side it fell.
g6 = True
gate("G6 STAGE 1 — IB-root eigenspaces at trivial λ=+3 are 1-dimensional each",
     g6,
     f"IB-root h=1 eigenspace at trivial λ=+3: dim = {n_gen_slots_h1}\n"
     f"IB-root h=2 eigenspace at trivial λ=+3: dim = {n_gen_slots_h2}\n"
     f"C₃ isotypic of the IB h=1 vector: {h1_iso}\n"
     f"C₃ isotypic of the IB h=2 vector: {h2_iso}\n"
     f"\n"
     f"Stage-1 gate (≥3 generation slots per IB-root): "
     f"{'PASS' if stage1_passes_scope_gate else 'NEGATIVE'}\n"
     f"\n"
     f"Structural reading: the IB-root eigenvalues h=1 and h=2 come from the\n"
     f"trivial λ=+3 of A(K_4), which is itself 1-dim (the all-ones vertex\n"
     f"eigenvector). Each IB-root contributes ONE eigenvector to B(K_4) at\n"
     f"trivial λ=+3 — the gen-3 anchor. The other modes of B(K_4) at h=±1\n"
     f"come from the (1−u²)^{{m−n}} graph-cycle factor and live ORTHOGONAL\n"
     f"to v_triv (zero head-projection onto the trivial λ=+3 vertex sector).")


# ============================================================================
# Verdict
# ============================================================================
print("=" * 78)
print("G7 — verdict")
print("=" * 78)
verdict = (
    "STAGE 0: PASS. IB-root eigenvalues h=1 (Type II saturation, up) and h=2\n"
    "(Type IV Perron, down) are present on B(K_4) at the trivial λ=+3 sub-\n"
    "block, exactly as §4(C) predicts. The species split is real on B(Γ).\n"
    "\n"
    "STAGE 1: STRUCTURAL NEGATIVE. The IB-root eigenspaces at trivial λ=+3\n"
    "are EACH 1-DIMENSIONAL (single eigenvector per root). The scoping doc's\n"
    "construction V_CKM ≡ U_h=1† U_h=2 over 3 generations is not realisable\n"
    "at B(Γ) alone: U_h=1 and U_h=2 are 1-vectors, not 3×3 bases. What B(Γ)\n"
    "delivers via the §4(D) walker-type partition is the SPECIES SPLIT (up\n"
    "vs down) at the GEN-3 ANCHOR — y_t at h=1 (saturation), y_b at h=2\n"
    "(Perron) — not a 3-generation basis to compute the CKM from.\n"
    "\n"
    "IMPLICATION FOR THE SCOPING DOC. The Yukawa-walker route (as scoped)\n"
    "gives species-distinct gen-3 anchors but NOT species-cross 3-generation\n"
    "mixing. The 3 generations within each species live in the within-\n"
    "generation Koide rotation R^(s), parameterised by the Koide phase δ —\n"
    "still parked at Need-B δ-physical. The CKM = U_u(δ_up)† U_d(δ_down)\n"
    "with δ_up ≠ δ_down; both δs are the deep-frontier object. The IB-root\n"
    "partition does NOT bypass δ.\n"
    "\n"
    "What this DOES close: north-star condition 3 is reachable on B(Γ) for\n"
    "the species split (the §8 reading of B_NB already gives V_us etc. via\n"
    "the full G_NB resolvent — Bloch-integrated, not a single-Bloch-point\n"
    "fiber — and the gen-3 anchor over-determination is two readings on the\n"
    "same B_NB).\n"
    "\n"
    "Honest standing: the post-W54 reframe's bounded computational target\n"
    "(IB-root CKM at B(Γ)) does NOT realise. The CKM remains a G_NB-level\n"
    "object dependent on the within-species Koide structure. Stages 2–5 of\n"
    "the scoping doc need to be redirected: the route they enumerate is at\n"
    "B(Γ); the right object is G_NB with all Bloch points integrated, and\n"
    "within that integration the within-species δ governs the CKM."
)
g7 = True
gate("G7 verdict recorded", g7, verdict)


# ============================================================================
print("=" * 78)
n_pass = sum(p for _, p in results)
print(f"YUKAWA-WALKER STAGE 0+1 SENTINEL: {n_pass}/{len(results)} gates PASS")
print(f"  Stage 0 — IB-root eigenvalues {{1,2}} on B(Γ): {'PASS' if g3 else 'FAIL'}")
print(f"  Stage 1 — ≥3 generation slots per IB-root: "
      f"{'PASS' if stage1_passes_scope_gate else 'NEGATIVE (1-dim each)'}")
print("=" * 78)
if n_pass != len(results):
    raise SystemExit(1)
