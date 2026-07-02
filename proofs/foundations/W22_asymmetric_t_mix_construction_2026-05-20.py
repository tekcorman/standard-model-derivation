#!/usr/bin/env python3
"""
W22 — Asymmetric T_mix using the W21 broken-vacuum orientation
==============================================================

Date: 2026-05-20
Predecessor: W21 (`W21_higgs_vev_srs_to_srsz_lift_2026-05-20.py`) supplied the
explicit per-edge lift of ⟨h⁰⟩ from srs's K_4 to srs-z's BD(K_4)=Q_3, with
σ_combined = σ_swap × σ_mirror sign-flipping the configuration.

W22 is Step 2 of the W20 forward path: construct the asymmetric T_mix that
USES the W21 orientation, and verify it breaks the chi_tilde 2026-05-01 EOD
"χ̃-pair mass degeneracy preserved" structural obstruction.

This probe is SELF-CONTAINED — uses the same explicit K_4 / BD(K_4) construction
as W21 (no RCSR parsing, no external data files). The Hashimoto operator is
constructed from the explicit graph; Bloch decomposition isn't needed because
the χ̃-anticommutation argument is k-independent on bipartite graphs.

CHI_TILDE 2026-05-01 BASELINE (chi_tilde memory P2.3 NEGATIVE result):
  T_cover = bipartite-cover projection π : Q_3 → K_4 (sheet-blind)
  Σ|⟨srs|T_cover|χ̃=+1⟩|² = Σ|⟨srs|T_cover|χ̃=−1⟩|²
  → χ̃-pair mass degeneracy PRESERVED. T_mix sterile for χ̃-breaking.

W21 PROVIDED: a sheet-DEPENDENT input (uniform ⟨h⁰⟩ · e_1 with σ_mirror
sign-flip giving sheet A = +v/√2, sheet B = −v/√2 in the LH frame).

W22 HYPOTHESIS: T_oriented = T_cover + g_Y · T_yukawa, where
  T_yukawa[i, j] = (⟨h⁰⟩ / v) · χ̃[j] · T_cover[i, j] = (1/√2) · (±1) · T_cover[i, j]
is the W21-Higgs-VEV-weighted, χ̃-graded cover term. Then
  |T_oriented|² = T_cover² · (1 + (g_Y / √2) · χ̃)²
which yields χ̃-ASYMMETRIC sums:
  Σ_{χ̃=+1} |T_oriented|² = (1 + g_Y/√2)² · S_cover/2
  Σ_{χ̃=−1} |T_oriented|² = (1 − g_Y/√2)² · S_cover/2
  difference = g_Y · √2 · S_cover
where S_cover = Σ_all |T_cover|² is the total cover transition strength.

PRE-DECLARED GATE CHECKS:
  G1. T_cover reproduces chi_tilde 2026-05-01 EOD's χ̃-symmetric Σ|T|² baseline.
  G2. T_yukawa is well-defined.
  G3. T_oriented is Hermitian.
  G4. T_oriented produces χ̃-ASYMMETRIC Σ|T|² for g_Y ≠ 0 with leading-order
      linear scaling g_Y · √2 · S_cover.
  G5. The asymmetry sign tracks the sign of g_Y.
  G6. At g_Y = 0, T_oriented = T_cover (baseline recovered).
  G7. Second-order mass-splitting between χ̃ = ±1 sectors on srs-z is non-zero
      under T_oriented and sign-flips with g_Y.

DELIBERATE NON-CLAIMS:
  • g_Y is treated as a FREE PARAMETER. W22 does NOT derive y_t = 1.
  • Per-sector Yukawa values come from Step 3 (n_free per (sector, generation)).
  • W22 only verifies that the MECHANISM connecting W21's orientation to
    χ̃-sector-breaking exists and behaves as predicted at leading order.

USAGE:
    python3 proofs/foundations/W22_asymmetric_t_mix_construction_2026-05-20.py
"""

from __future__ import annotations
import numpy as np

EXPECTED = {
    "G1_cover_chi_symmetric":         True,
    "G2_yukawa_well_defined":         True,
    "G3_oriented_hermitian":          True,
    "G4_oriented_chi_asymmetric":     True,
    "G5_asymmetry_sign_tracks_gY":    True,
    "G6_gY_zero_recovers_baseline":   True,
    "G7_2nd_order_splitting_nonzero": True,
}
RESULTS = {}

print("=" * 78)
print("W22 — asymmetric T_mix using the W21 broken-vacuum orientation")
print("=" * 78)


# ============================================================================
# Step A — K_4 (srs primitive) and BD(K_4)=Q_3 (srs-z primitive), self-contained
# ============================================================================
# Same construction as W21 — explicit, no RCSR dependence.
N_V_K4 = 4
K4_edges = [(u, v) for u in range(N_V_K4) for v in range(u + 1, N_V_K4)]   # 6 edges
N_V_BD = 8                                                                  # = 2 × 4
def encode(u, sheet): return u + sheet * N_V_K4

bd_edges = []                # 12 undirected edges of BD(K_4)
cover_pairs = []             # (alpha_idx, beta_idx) per K_4 edge
for u, v in K4_edges:
    alpha = (encode(u, 0), encode(v, 1))
    beta  = (encode(v, 0), encode(u, 1))
    bd_edges.append(alpha)
    bd_edges.append(beta)
    cover_pairs.append((len(bd_edges) - 2, len(bd_edges) - 1))
assert len(bd_edges) == 12

# Directed arcs: each undirected edge contributes 2 directed arcs.
# K_4 has 12 directed arcs; BD(K_4) has 24.
def directed_arcs(edges):
    arcs = []
    for ei, (u, v) in enumerate(edges):
        arcs.append((u, v, ei))   # tail=u, head=v, underlying_edge=ei
        arcs.append((v, u, ei))   # reverse direction
    return arcs

K4_arcs = directed_arcs(K4_edges)   # 12 arcs on K_4
BD_arcs = directed_arcs(bd_edges)   # 24 arcs on BD(K_4)
N_ARCS_K4 = len(K4_arcs)
N_ARCS_BD = len(BD_arcs)
print(f"\nStep A — explicit K_4 and BD(K_4)=Q_3 (no external data)")
print(f"  K_4: |V| = {N_V_K4}, |E| = {len(K4_edges)}, |arcs| = {N_ARCS_K4}")
print(f"  BD(K_4)=Q_3: |V| = {N_V_BD}, |E| = {len(bd_edges)}, |arcs| = {N_ARCS_BD}")


# ============================================================================
# Step B — Hashimoto operators (no Bloch needed; abstract graph version)
# ============================================================================
def hashimoto(arcs):
    """B[a', a] = 1 if a' continues a (NB: head(a) = tail(a') and a' ≠ reverse(a))."""
    n = len(arcs)
    B = np.zeros((n, n), dtype=complex)
    for i_p, (t_p, h_p, e_p) in enumerate(arcs):
        for i, (t, h, e) in enumerate(arcs):
            if h == t_p and e_p != e:   # head of a matches tail of a'; NB excludes reverse
                B[i_p, i] = 1.0
    return B

B_K4 = hashimoto(K4_arcs)         # 12×12
B_BD = hashimoto(BD_arcs)         # 24×24
print(f"\nStep B — Hashimoto operators on abstract graphs (k=Γ)")
print(f"  B_K4  shape = {B_K4.shape}, row-sum (= k-1 = 2): {set(int(s.real) for s in B_K4.sum(axis=1))}")
print(f"  B_BD  shape = {B_BD.shape}, row-sum (= k-1 = 2): {set(int(s.real) for s in B_BD.sum(axis=1))}")


# ============================================================================
# Step C — χ̃ on BD(K_4) walker (tail-side diagonal)
# ============================================================================
# Side A = sheet 0 (vertices 0..3); Side B = sheet 1 (vertices 4..7).
side_label = {idx: (+1 if idx < N_V_K4 else -1) for idx in range(N_V_BD)}
chi_BD_diag = np.array([side_label[t] for (t, _, _) in BD_arcs], dtype=float)
chi_BD = np.diag(chi_BD_diag).astype(complex)
n_plus = int((chi_BD_diag > 0).sum())
n_minus = int((chi_BD_diag < 0).sum())
print(f"\nStep C — χ̃ on BD(K_4) walker (tail-side grading)")
print(f"  χ̃ = +1 sector (tail in sheet A): {n_plus} arcs")
print(f"  χ̃ = -1 sector (tail in sheet B): {n_minus} arcs")

# Confirm anti-commutation {χ̃, B_BD} = 0 (load-bearing for the structural baseline).
anticomm = chi_BD @ B_BD + B_BD @ chi_BD
print(f"  ||{{χ̃, B_BD}}||_F = {np.linalg.norm(anticomm):.4e}  (expect 0 on bipartite Hashimoto)")
assert np.linalg.norm(anticomm) < 1e-12, "χ̃ must anticommute with B_BD"


# ============================================================================
# Step D — Cover projection π : BD(K_4) → K_4, sheet-blind
# ============================================================================
# Each BD(K_4) vertex (u, sheet) projects to K_4 vertex u (forgetting sheet).
# π on arcs: (tail, head, edge_idx_BD) → (π(tail), π(head), edge_idx_K4) where
# edge_idx_K4 is the K_4 edge underlying the BD edge.
def pi_vertex(v_bd): return v_bd % N_V_K4

# For each BD edge, its underlying K_4 edge index can be recovered from
# (pi(u), pi(v)) lookup, since each BD edge connects different K_4 vertices.
K4_edge_lookup = {frozenset(e): i for i, e in enumerate(K4_edges)}
bd_edge_to_k4_edge = []
for (u_bd, v_bd) in bd_edges:
    bd_edge_to_k4_edge.append(K4_edge_lookup[frozenset((pi_vertex(u_bd), pi_vertex(v_bd)))])

# T_cover[a_k4, a_bd] = 1 if π takes a_bd to a_k4
K4_arc_idx = {(t, h, e): i for i, (t, h, e) in enumerate(K4_arcs)}
T_cover = np.zeros((N_ARCS_K4, N_ARCS_BD), dtype=complex)
for j, (t_bd, h_bd, e_bd) in enumerate(BD_arcs):
    t_k = pi_vertex(t_bd)
    h_k = pi_vertex(h_bd)
    e_k = bd_edge_to_k4_edge[e_bd]
    i = K4_arc_idx[(t_k, h_k, e_k)]
    T_cover[i, j] = 1.0
print(f"\nStep D — cover-projection T_cover (chi_tilde 2026-05-01 baseline)")
print(f"  T_cover shape = {T_cover.shape}")
# Each K_4 arc receives 2 BD(K_4) arc preimages (one per cover sheet).
print(f"  Preimage count per K_4 arc: {set(int(s) for s in (T_cover != 0).sum(axis=1))}  (expect {{2}})")

# Boltzmann weight from M2a structural-DL audit (kept for parity with chi_tilde probe)
DELTA_DL = 3.25
w_srsz = 2.0 ** (-DELTA_DL)
amp = np.sqrt(w_srsz)
T_cover_w = amp * T_cover

# G1: χ̃-symmetric Σ|T_cover|²
chi_plus_mask  = chi_BD_diag > 0
chi_minus_mask = chi_BD_diag < 0
S_cover_plus  = float(np.sum(np.abs(T_cover_w[:, chi_plus_mask])  ** 2))
S_cover_minus = float(np.sum(np.abs(T_cover_w[:, chi_minus_mask]) ** 2))
S_cover_total = S_cover_plus + S_cover_minus
print(f"  Boltzmann amp √w = {amp:.4f}")
print(f"  Σ|T_cover_w|² over χ̃ = +1: {S_cover_plus:.6f}")
print(f"  Σ|T_cover_w|² over χ̃ = -1: {S_cover_minus:.6f}")
print(f"  S_cover total: {S_cover_total:.6f}")
G1 = abs(S_cover_plus - S_cover_minus) < 1e-12
print(f"  G1: chi_tilde 2026-05-01 baseline reproduced (χ̃-symmetric): {G1}")
RESULTS["G1_cover_chi_symmetric"] = bool(G1)


# ============================================================================
# Step E — T_yukawa: W21-Higgs-VEV-weighted, χ̃-graded cover
# ============================================================================
# Per W21: ⟨h⁰⟩/v = 1/√2 on every BD edge, with σ_combined sign-flipping it
# between sheet A and sheet B. The per-arc dimensionless weight is
#     w_yukawa[j] = (⟨h⁰⟩ / v) · χ̃[j] = (1/√2) · (±1)
hzero_over_v = 1.0 / np.sqrt(2.0)
T_yukawa = np.zeros_like(T_cover_w)
for j in range(N_ARCS_BD):
    T_yukawa[:, j] = hzero_over_v * chi_BD_diag[j] * T_cover_w[:, j]

nonzero_match = np.all((T_cover_w != 0) | (T_yukawa == 0))
print(f"\nStep E — T_yukawa")
print(f"  T_yukawa[:, j] = (1/√2) · χ̃[j] · T_cover_w[:, j]")
print(f"  T_yukawa zero where T_cover zero: {nonzero_match}")
print(f"  ||T_yukawa||_F = {np.linalg.norm(T_yukawa):.4f}")
RESULTS["G2_yukawa_well_defined"] = bool(nonzero_match)


# ============================================================================
# Step F — Joint walker T_oriented and Hermiticity
# ============================================================================
N_JOINT = N_ARCS_K4 + N_ARCS_BD

def build_T_mix(g_Y: float):
    T_off = T_cover_w + g_Y * T_yukawa
    T = np.zeros((N_JOINT, N_JOINT), dtype=complex)
    T[:N_ARCS_K4, N_ARCS_K4:] = T_off
    T[N_ARCS_K4:, :N_ARCS_K4] = T_off.conj().T
    return T, T_off

T_mix_unit, T_off_unit = build_T_mix(g_Y=1.0)
herm_res = np.linalg.norm(T_mix_unit - T_mix_unit.conj().T)
print(f"\nStep F — joint walker T_oriented")
print(f"  N_joint = {N_JOINT}")
print(f"  ||T_oriented - T_oriented†||_F (at g_Y=1) = {herm_res:.2e}")
G3 = herm_res < 1e-12
print(f"  G3: T_oriented Hermitian: {G3}")
RESULTS["G3_oriented_hermitian"] = bool(G3)


# ============================================================================
# Step G — Asymmetry sweep over g_Y
# ============================================================================
def sector_sums(T_off):
    s_p = float(np.sum(np.abs(T_off[:, chi_plus_mask])  ** 2))
    s_m = float(np.sum(np.abs(T_off[:, chi_minus_mask]) ** 2))
    return s_p, s_m

predicted_diff = lambda gY: gY * np.sqrt(2.0) * S_cover_total
print(f"\nStep G — asymmetry sweep")
print(f"  Leading-order prediction: difference = g_Y · √2 · S_cover_total = g_Y · {np.sqrt(2.0)*S_cover_total:.6f}")
print(f"  {'g_Y':>8s} {'S(χ̃=+1)':>14s} {'S(χ̃=-1)':>14s} {'difference':>14s}   {'predicted':>14s}   {'residual':>12s}")
print(f"  " + "-" * 80)

asymmetry_table = []
for gY in [-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0]:
    _, T_off = build_T_mix(g_Y=gY)
    s_p, s_m = sector_sums(T_off)
    diff = s_p - s_m
    pred = predicted_diff(gY)
    res = abs(diff - pred)
    asymmetry_table.append((gY, s_p, s_m, diff, pred, res))
    print(f"  {gY:>+8.3f} {s_p:>14.6f} {s_m:>14.6f} {diff:>+14.6f}   {pred:>+14.6f}   {res:>12.2e}")

G4 = all(
    res < 1e-10 and (abs(diff) > 1e-10 if gY != 0 else abs(diff) < 1e-10)
    for gY, _, _, diff, _, res in asymmetry_table
)
print(f"  G4: χ̃-asymmetric at g_Y ≠ 0 with leading-order match: {G4}")
RESULTS["G4_oriented_chi_asymmetric"] = bool(G4)

G5 = all(
    (diff > 0 if gY > 0 else (diff < 0 if gY < 0 else abs(diff) < 1e-10))
    for gY, _, _, diff, _, _ in asymmetry_table
)
print(f"  G5: asymmetry sign tracks sign of g_Y: {G5}")
RESULTS["G5_asymmetry_sign_tracks_gY"] = bool(G5)

gY0_diff = next(diff for gY, _, _, diff, _, _ in asymmetry_table if gY == 0.0)
G6 = abs(gY0_diff) < 1e-12
print(f"  G6: g_Y=0 recovers baseline (asymmetry vanishes): {G6}")
RESULTS["G6_gY_zero_recovers_baseline"] = bool(G6)


# ============================================================================
# Step H — Sector-resolved Feshbach self-energy on H_BD
# ============================================================================
# χ̃ is diagonal in the arc basis but ANTI-commutes with B_BD — B_BD eigenstates
# are χ̃-mixed (their χ̃-overlap is zero). The natural sector resolution lives
# in the ARC BASIS, where χ̃ = diag(±1) sorts the 24 arcs into two 12-dim sectors.
#
# Feshbach self-energy on H_BD (24-dim):
#     Σ^(2)(E_ref) = T_off^† · (E_ref · I − B_K4)^(−1) · T_off            [24×24]
# Sector-resolved trace:
#     Tr_{χ̃=±}( Σ^(2) ) = Σ_{n with χ̃[n]=±1}  Σ^(2)[n, n]
#
# G7: Tr_{χ̃=+}( Σ^(2) ) ≠ Tr_{χ̃=−}( Σ^(2) ) for g_Y ≠ 0; equal at g_Y = 0;
#     sign-flips with sign of g_Y.

# Pick an E_ref outside B_K4's spectrum so the resolvent is well-defined.
eigs_K4 = np.linalg.eigvals(B_K4)
# B_K4 spectrum is {+2, +1×5, -1×4, ...} numerically; pick E_ref = 5 to be safe.
spec_strs_K4 = [(f"{e.real:+.3f}" if abs(e.imag) < 1e-9 else f"{e.real:+.3f}{e.imag:+.3f}i") for e in eigs_K4]
E_REF = 5.0
resolvent = np.linalg.inv(E_REF * np.eye(N_ARCS_K4) - B_K4)

print(f"\nStep H — sector-resolved Feshbach self-energy")
print(f"  B_K4 spectrum: {spec_strs_K4[:6]}...")
print(f"  E_ref = {E_REF} (chosen outside B_K4 spectrum so resolvent is non-singular)")
print(f"  ||resolvent||_F = {np.linalg.norm(resolvent):.4f}")
print()
print(f"  {'g_Y':>8s} {'Tr_χ̃=+(Σ²)':>14s} {'Tr_χ̃=-(Σ²)':>14s} {'splitting':>14s}")
print(f"  " + "-" * 60)
splittings = []
for gY in [0.0, 0.1, 0.5, 1.0, -1.0]:
    _, T_off = build_T_mix(g_Y=gY)
    Sigma2 = T_off.conj().T @ resolvent @ T_off          # 24×24 on H_BD
    tr_p = float(np.trace(Sigma2[np.ix_(np.where(chi_plus_mask)[0],
                                        np.where(chi_plus_mask)[0])]).real)
    tr_m = float(np.trace(Sigma2[np.ix_(np.where(chi_minus_mask)[0],
                                        np.where(chi_minus_mask)[0])]).real)
    split = tr_p - tr_m
    splittings.append((gY, tr_p, tr_m, split))
    print(f"  {gY:>+8.3f} {tr_p:>+14.6f} {tr_m:>+14.6f} {split:>+14.6f}")

split_zero = next(s for gY, _, _, s in splittings if gY == 0.0)
split_pos1 = next(s for gY, _, _, s in splittings if gY == 1.0)
split_neg1 = next(s for gY, _, _, s in splittings if gY == -1.0)
G7 = (
    abs(split_zero) < 1e-10
    and abs(split_pos1) > 1e-6
    and abs(split_neg1) > 1e-6
    and np.sign(split_pos1) == -np.sign(split_neg1)
)
print(f"  G7: sector-resolved Feshbach self-energy χ̃-asymmetric (sign-flips with g_Y): {G7}")
RESULTS["G7_2nd_order_splitting_nonzero"] = bool(G7)


# ============================================================================
# Step I — Verdict
# ============================================================================
print("\n" + "=" * 78)
print("W22 VERDICT — Gate Check")
print("=" * 78)
all_pass = True
for k, expected in EXPECTED.items():
    actual = RESULTS.get(k)
    status = "PASS" if actual == expected else "FAIL"
    if actual != expected:
        all_pass = False
    print(f"  {status}  {k:35s}  expected={expected}, got={actual}")

print()
if all_pass:
    print("  ALL CHECKS PASS.")
    print()
    print("  W22 establishes Step 2 of the W20 forward path:")
    print()
    print("    - T_cover (chi_tilde 2026-05-01 baseline) is sheet-blind and χ̃-symmetric")
    print("      in Σ|T|². Confirmed (G1).")
    print("    - T_yukawa = (⟨h⁰⟩/v) · χ̃ · T_cover encodes the W21 broken-vacuum")
    print("      orientation as a per-arc, sheet-dependent weight on the cover bridge.")
    print("    - T_oriented(g_Y) = T_cover + g_Y · T_yukawa is Hermitian (G3).")
    print("    - For g_Y ≠ 0, T_oriented produces χ̃-ASYMMETRIC Σ|T|² with leading-order")
    print("      linear scaling g_Y · √2 · S_cover (G4, G5).")
    print("    - At g_Y = 0, T_oriented reduces to T_cover; baseline recovered (G6).")
    print("    - Second-order Feshbach residue on χ̃-paired states is non-zero for")
    print("      g_Y ≠ 0 and sign-flips with g_Y (G7). χ̃-pair mass degeneracy IS BROKEN")
    print("      by the W21 orientation — contradicts chi_tilde 2026-05-01 EOD's verdict,")
    print("      which was about g_Y = 0 only.")
    print()
    print("  STATUS: Step 2 closed at the mechanism level. The asymmetric T_mix exists,")
    print("  is well-defined on the joint walker, and produces the predicted χ̃-asymmetric")
    print("  second-order Feshbach residue. g_Y is a FREE PARAMETER in W22 — Step 3")
    print("  (n_free per (sector, generation) → g_Y per channel) is the next bounded")
    print("  research target.")
    print()
    print("  Honest caveats:")
    print("    - g_Y is a placeholder for the Yukawa coupling in the substrate picture;")
    print("      whether g_Y = y exactly requires Step 3.")
    print("    - The 2nd-order shift uses Hashimoto B_BD on abstract BD(K_4) (k=Γ) as the")
    print("      bare Hamiltonian; the full y_t derivation will need B at the K-rational")
    print("      saddle and the full Bloch-decomposed framework apparatus.")
    print("    - W22 verifies the MECHANISM. Quantitative per-channel Yukawa values are")
    print("      NOT outputs of W22 — that's Step 3.")
else:
    print("  ONE OR MORE CHECKS FAILED. Re-examine the construction.")

print()
print("=" * 78)
