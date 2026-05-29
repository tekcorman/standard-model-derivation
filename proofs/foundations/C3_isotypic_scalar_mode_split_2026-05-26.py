#!/usr/bin/env python3
"""
C_3 isotypic decomposition of the u=±1 eigenspace on K_4, with focus on
whether the 2 J=+1 scalar modes split into distinct C_3 isotypics.

CONTEXT
-------
The graph-only operators (J, vertex-parity, S_4 irreps, symmetric vertex-lift S)
all fail to distinguish the 2 J=+1 scalar modes. The framework has C_3
generation structure on srs (`theorem_C3_block_decomposition_2026-05-21.md`,
THEOREM-GRADE). If the 2 scalar modes lie in DIFFERENT C_3 isotypics (one
trivial = "single-generation", one non-trivial = "3-generation-cycled"),
then C_3 provides the structural split.

This matters because:
- U(1)_Y and SU(2)_L gauge bosons are gauge-singlets (no generation index),
  so their hypercharge / weak coupling sums over generations uniformly →
  couples to C_3-trivial Hashimoto content.
- SU(3)_c color acts within each generation but the color triplet structure
  doesn't permute generations → also seemingly C_3-trivial.

The first cut just tests: are the 2 scalar modes in different C_3 isotypics?
Even just confirming there's SOME structural split would advance the leading-
mode derivation past the graph-equivalent dead-end.
"""

import math
import numpy as np
from fractions import Fraction

# ============================================================
# 1. K_4 setup
# ============================================================
N_V = 4
vertices = list(range(N_V))
directed_edges = [(u, v) for u in vertices for v in vertices if u != v]
N_DE = len(directed_edges)
e2i = {e: i for i, e in enumerate(directed_edges)}

B = np.zeros((N_DE, N_DE), dtype=int)
for i, (u, v) in enumerate(directed_edges):
    for w in vertices:
        if w == u or w == v:
            continue
        B[i, e2i[(v, w)]] = 1
J = np.zeros((N_DE, N_DE), dtype=int)
for i, (u, v) in enumerate(directed_edges):
    J[i, e2i[(v, u)]] = 1

# ============================================================
# 2. u = ±1 eigenspace
# ============================================================
ev, evec = np.linalg.eig(B.astype(float))
mp = np.abs(ev - 1.0) < 1e-8
mm = np.abs(ev + 1.0) < 1e-8
V_p = np.real_if_close(evec[:, mp], tol=1000)
V_m = np.real_if_close(evec[:, mm], tol=1000)
if np.iscomplexobj(V_p):
    V_p = np.real(V_p)
if np.iscomplexobj(V_m):
    V_m = np.real(V_m)
V_p, _ = np.linalg.qr(V_p)
V_m, _ = np.linalg.qr(V_m)
V_pm = np.concatenate([V_p, V_m], axis=1)
assert V_pm.shape[1] == 5

# Wilson-loop split
triangles = []
for omit in range(N_V):
    others = [v for v in range(N_V) if v != omit]
    a, b, c = others
    triangles.append([(a, b), (b, c), (c, a)])
H_mat = np.array([[sum(V_pm[e2i[e], k] for e in C) for C in triangles]
                  for k in range(V_pm.shape[1])])
U_h, S_h, _ = np.linalg.svd(H_mat, full_matrices=True)
rank_h = int(np.sum(S_h > 1e-8))
V_cycle  = V_pm @ U_h[:, :rank_h]
V_scalar = V_pm @ U_h[:, rank_h:]
print("="*78)
print("  C_3 isotypic decomposition of u=±1 eigenspace on K_4  (2026-05-26)")
print("="*78)
print(f"  V_cycle dim:  {V_cycle.shape[1]}  (3 J=-1 Wilson-loop carriers)")
print(f"  V_scalar dim: {V_scalar.shape[1]}  (2 J=+1 modes, zero loop sums)")
print()

# ============================================================
# 3. C_3 action on K_4 directed edges
# ============================================================
# Choose the C_3 subgroup that cycles {0, 1, 2} and fixes 3
# Generation labeling per framework convention: vertex 3 = "fixed point" of C_3
def apply_c3(sigma, edge):
    """Apply σ ∈ S_4 to a directed edge."""
    u, v = edge
    return (sigma[u], sigma[v])

# C_3 generator: (0 1 2)(3) — sends 0→1, 1→2, 2→0, 3→3
c3_gen = {0: 1, 1: 2, 2: 0, 3: 3}
c3_id = {v: v for v in vertices}
c3_gen2 = {0: 2, 1: 0, 2: 1, 3: 3}
c3_elements = [c3_id, c3_gen, c3_gen2]

def perm_matrix(sigma):
    M = np.zeros((N_DE, N_DE))
    for i, e in enumerate(directed_edges):
        new_e = apply_c3(sigma, e)
        j = e2i[new_e]
        M[j, i] = 1
    return M

P_id = perm_matrix(c3_id)   # identity
P_c  = perm_matrix(c3_gen)  # generator
P_cc = perm_matrix(c3_gen2) # generator²
assert np.allclose(P_id, np.eye(N_DE))
assert np.allclose(P_c @ P_c, P_cc)
assert np.allclose(P_c @ P_cc, P_id)

# ============================================================
# 4. C_3 character of V_pm and isotypic decomposition
# ============================================================
chars = [np.trace(V_pm.T @ P @ V_pm) for P in (P_id, P_c, P_cc)]
print(f"  C_3 characters on V_pm (5-dim):  χ(e)={chars[0]:.3f}, χ(c)={chars[1]:.3f}, χ(c²)={chars[2]:.3f}")

# C_3 irreps:
#   trivial:    e→1, c→1, c²→1
#   ω:          e→1, c→ω, c²→ω²    (ω = e^(2πi/3))
#   ω²:         e→1, c→ω², c²→ω
# Over ℝ, ω and ω² combine into a 2-dim faithful real rep.
#
# Multiplicities (over ℂ):
#   m_trivial = (1/3) (χ(e) + χ(c) + χ(c²))
#   m_ω       = (1/3) (χ(e) + ω̄·χ(c) + ω·χ(c²))
#   m_ω²      = conjugate of m_ω
omega = np.exp(2j*np.pi/3)
m_triv = (chars[0] + chars[1] + chars[2]) / 3.0
m_omega = (chars[0] + np.conj(omega)*chars[1] + omega*chars[2]) / 3.0
m_omegabar = (chars[0] + omega*chars[1] + np.conj(omega)*chars[2]) / 3.0
print(f"  Multiplicity of trivial rep:    {m_triv:.4f}")
print(f"  Multiplicity of ω rep:         {m_omega:.4f}  (complex)")
print(f"  Multiplicity of ω̄ rep:         {m_omegabar:.4f}  (complex)")
# Real 2-dim rep multiplicity = m_omega + m_omegabar in real combination
print()

# ============================================================
# 5. Project V_pm onto C_3 isotypic sub-spaces
# ============================================================
# Trivial projector: (1/3) Σ_g P_g
Pi_triv = (P_id + P_c + P_cc) / 3.0
# 2-dim faithful real projector: I - Π_triv
Pi_faith = np.eye(N_DE) - Pi_triv

# Project V_pm onto trivial and faithful sub-spaces
V_pm_triv = Pi_triv @ V_pm
V_pm_faith = Pi_faith @ V_pm
print(f"  Projection of V_pm onto C_3 trivial isotypic:")
for k in range(V_pm.shape[1]):
    v = V_pm[:, k]
    v_t = Pi_triv @ v
    v_f = Pi_faith @ v
    norm_t = np.dot(v_t, v_t)
    norm_f = np.dot(v_f, v_f)
    total = np.dot(v, v)
    print(f"    mode {k}: |trivial|² / |v|² = {norm_t/total:.4f}, "
          f"|faithful|² / |v|² = {norm_f/total:.4f}")
print()

# ============================================================
# 6. KEY TEST: do V_cycle and V_scalar live in distinct C_3 isotypics?
# ============================================================
print("-"*78)
print(" KEY TEST: C_3 isotypic content of V_cycle (dim 3) and V_scalar (dim 2)")
print("-"*78)
def isotypic_content(V_sub, name):
    chars_sub = [np.trace(V_sub.T @ P @ V_sub) for P in (P_id, P_c, P_cc)]
    m_t = (chars_sub[0] + chars_sub[1] + chars_sub[2]) / 3.0
    m_w = (chars_sub[0] + np.conj(omega)*chars_sub[1] + omega*chars_sub[2]) / 3.0
    m_wb = (chars_sub[0] + omega*chars_sub[1] + np.conj(omega)*chars_sub[2]) / 3.0
    print(f"  {name} (dim {V_sub.shape[1]}):")
    print(f"    χ(e)={chars_sub[0]:.3f}, χ(c)={chars_sub[1]:.3f}, χ(c²)={chars_sub[2]:.3f}")
    print(f"    m_trivial = {m_t:.3f},  m_ω = {m_w:.3f},  m_ω̄ = {m_wb:.3f}")
    return m_t, m_w + m_wb     # real-faithful = m_ω + m_ω̄

m_cycle_t, m_cycle_f = isotypic_content(V_cycle, "V_cycle")
m_scalar_t, m_scalar_f = isotypic_content(V_scalar, "V_scalar")
print()

# ============================================================
# 7. Diagnosis
# ============================================================
print("="*78)
print(" DIAGNOSIS")
print("="*78)

# Round to nearest integer / half-integer if very close
def near_int(x, tol=0.05):
    return abs(x - round(x.real)) < tol and abs(x.imag) < tol

if near_int(m_cycle_t) and near_int(m_scalar_t):
    n_cycle_t = int(round(m_cycle_t.real))
    n_scalar_t = int(round(m_scalar_t.real))
    n_cycle_f = int(round(m_cycle_f.real))
    n_scalar_f = int(round(m_scalar_f.real))
    print(f"  V_cycle  isotypic: {n_cycle_t} trivial + {n_cycle_f//2} × (2-dim faithful)")
    print(f"  V_scalar isotypic: {n_scalar_t} trivial + {n_scalar_f//2} × (2-dim faithful)")
    print()
    if n_scalar_t == 1 and n_scalar_f == 1:
        print(f"  ✓ V_scalar splits cleanly: 1 C_3-trivial + 1 single-direction in 2-dim faithful.")
        print(f"    The 2 scalar modes lie in DIFFERENT C_3 isotypics — STRUCTURAL SPLIT FOUND.")
    elif n_scalar_t == 2 and n_scalar_f == 0:
        print(f"  V_scalar is entirely C_3-trivial. Both scalar modes in the trivial isotypic.")
        print(f"  → no internal split via C_3 alone.")
    elif n_scalar_t == 0 and n_scalar_f == 2:
        print(f"  V_scalar is entirely C_3-faithful (one full 2-dim faithful copy).")
        print(f"  → 2 scalar modes form an irreducible C_3 pair — no internal split.")
    else:
        print(f"  V_scalar has mixed isotypic content ({n_scalar_t}, {n_scalar_f}); structure")
        print(f"  is non-trivial but doesn't immediately give 1+1 split.")
else:
    print(f"  Multiplicities are not clean integers — non-orthogonal projections.")
    print(f"  m_cycle_t = {m_cycle_t},  m_scalar_t = {m_scalar_t}")
print()

# Additional diagnostic: how does sin²θ_W = 3/8 relate to the C_3 split?
print(" CONNECTION TO sin²θ_W = 3/8 (Georgi-Quinn-Weinberg trace identity)")
print(" -----------------------------------------------------------------")
print(" If V_cycle = (1 trivial + 1 faithful-pair) = 1 + 2 = 3 dim AND")
print("    V_scalar = (0 trivial + 1 faithful-pair) = 0 + 2 = 2 dim, total 5,")
print(" then 'C_3-trivial content' = 1 mode (just in V_cycle),")
print(" and 'C_3-faithful content' = 4 modes (2 in V_cycle + 2 in V_scalar).")
print()
print(" The framework's sin²θ_W = 3/8 arises from GQW trace identity over a")
print(" Pati-Salam multiplet. If the C_3-isotypic count reproduces this ratio")
print(" (3 charged of 8, or similar projection), that's a deep structural link.")
print()
print(" Within the 4-dim 'EM-block sector' (Route H c=4/12):")
print(f"   if scalar mode chosen = the C_3-trivial scalar (1 mode), then")
print(f"   the 4 modes = (3 cycle) + (1 trivial scalar) where the 'extra' 1")
print(f"   couples to gauge-singlet content (consistent with U(1)_Y and SU(2)_L")
print(f"   weak diagonal coupling, but NOT with SU(3) color-triplet structure).")
print("="*78)
