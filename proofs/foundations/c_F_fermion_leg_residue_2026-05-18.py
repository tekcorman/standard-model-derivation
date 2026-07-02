#!/usr/bin/env python3
"""
>>> SUPERSEDED-VERDICT NOTICE (2026-05-21 spring-cleaning) <<<
This probe's verdict (REJECT — w_F = 1/144) is NOT the framework's current
position on c_F and should not be cited as such. It applies the delta_r
gauge-singlet residue formula verbatim to the fermion leg; the framework
subsequently judged that a channel mismatch and re-derived c_F through the
channel_select MDL gate — see `c_F_channel_select_waterfilling_2026-05-18.py`
(-> 1/12) and `docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md`
(Family-D section, ~line 125). c_F is currently graded THEOREM-GRADE-STRUCTURAL,
*conditional on that channel argument*. This file and that one record an open,
documented methodological dispute over the fermion-leg channel — read both.
>>> end notice <<<

proofs/foundations/c_F_fermion_leg_residue_2026-05-18.py

EXECUTION of the pre-registered W1 probe
(docs/audits/preregistrations/c_F_fermion_leg_residue_2026-05-18.md,
committed 788bb45 BEFORE this file was written).

Computes the Family-D per-fermion-leg residue weight w_F by the IDENTICAL
formula used for the proven δ_r coefficient, with channel (B1=Perron) and
normalization (B2=δ_r verbatim) frozen by the prereg. Applies the
pre-registered decision rule. No m_τ value is consulted.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proofs" / "foundations"))
sys.path.insert(0, str(REPO / "predictions"))

from proofs.common import K_STAR, N_ATOMS  # noqa: E402
# Frozen input A1 — reuse the EXISTING B_NB(srs) construction verbatim.
from nb_two_vertex_generations_probe import (  # noqa: E402
    directed_edges, nb_operator, rev_index,
)

k_star, N = K_STAR, N_ATOMS
GAMMA = (0.0, 0.0, 0.0)

print("=" * 78)
print("  W1 c_F fermion-leg residue — pre-registered execution (prereg 788bb45)")
print("=" * 78)

de = directed_edges()
rev = rev_index(de)
two_E = len(de)
B_G = nb_operator(GAMMA, de, rev)

# Frozen input A2 — Perron projector P_P = |1><1|/<1|1>; B_NB·1=(k*-1)·1.
ones = np.ones(two_E, dtype=complex)
assert np.allclose(B_G @ ones, (k_star - 1) * ones, atol=1e-9), \
    "Perron/uniform check failed (A2)"
P_Perron = np.outer(ones, ones.conj()) / (ones.conj() @ ones)

print(f"\n  2|E| = {two_E}   Perron eigval = k*-1 = {k_star-1}   "
      f"handshake 2|E|=N·k* : {two_E}={N}·{k_star}={N*k_star}")

# ---- D.6 sanity: reproduce δ_r's c_S=1/12 via the SAME code path ----------
# unified_oblique_one_resolvent_2026-05-16.py:160 verbatim formula.
s_hat = ones / np.sqrt(two_E)                       # unit gauge-singlet
c_S = float(np.real(s_hat.conj() @ P_Perron @ s_hat)) / two_E
print(f"\n  [D.6 sanity] gauge-singlet c_S = <ŝ|P_P|ŝ>/(2|E|) = {c_S:.10f}")
print(f"               1/12 = {1/12:.10f}   match: {abs(c_S - 1/12) < 1e-12}")
assert abs(c_S - 1.0 / 12.0) < 1e-12, "D.6 FAILED — not the δ_r formula path"

# ---- B1+B2: per-fermion-leg residue, δ_r formula verbatim ----------------
# B1: Perron channel.  B2: w = <V̂|P_channel|V̂>/(2|E|), V̂ = unit |e_a>.
# Frozen input A3: V̂_F is a CAR single-directed-edge unit basis vector.
# D.3: must be identical for ALL 12 directed-edge choices.
w_F_vals_num = []
w_F_vals_exact = []
bare_vals_exact = []   # <e_a|P_P|e_a> WITHOUT the /(2|E|) (transparency only)
for a in range(two_E):
    e_a = np.zeros(two_E, dtype=complex)
    e_a[a] = 1.0                                    # unit single-edge vector
    bare = np.real(e_a.conj() @ P_Perron @ e_a)     # <e_a|P_P|e_a>
    w_F = bare / two_E                              # B2 verbatim δ_r formula
    w_F_vals_num.append(w_F)
    # exact: <e_a|P_P|e_a> = |<e_a|1>|^2/<1|1> = 1/(2|E|); /(2|E|) -> 1/(2|E|)^2
    bare_vals_exact.append(Fraction(1, two_E))
    w_F_vals_exact.append(Fraction(1, two_E * two_E))

all_same = (len(set(w_F_vals_exact)) == 1
            and max(w_F_vals_num) - min(w_F_vals_num) < 1e-12)
print(f"\n  [D.3] all {two_E} edges identical w_F : {all_same}")
assert all_same, "D.3 FAILED — w_F edge-dependent ⇒ REJECT (ill-defined)"

w_F_exact = w_F_vals_exact[0]
bare_exact = bare_vals_exact[0]
print(f"\n  <e_a|P_P|e_a>            = {float(bare_exact):.10f}  (exact {bare_exact})")
print(f"  w_F = <e_a|P_P|e_a>/(2|E|) = {w_F_vals_num[0]:.10f}  (exact {w_F_exact})")
print(f"\n  (transparency) bare matrix element without the /(2|E|) "
      f"= {bare_exact} = 1/12  ← the value that WOULD strengthen,")
print(f"  but B2 (δ_r formula verbatim, with the explicit /(2|E|)) gives "
      f"w_F = {w_F_exact}.")

# ---- Pre-registered decision rule (C) ------------------------------------
target_strengthen = Fraction(1, 12)
print("\n" + "=" * 78)
print("  PRE-REGISTERED DECISION (rule C; exact equality)")
print("=" * 78)
if w_F_exact == target_strengthen:
    verdict = "STRENGTHEN"
    print(f"  w_F = {w_F_exact} == 1/12  →  STRENGTHEN")
    print("  c_F derived by the δ_r-class Perron residue; Family-D fermion")
    print("  sector graduates; the F-1/F-2 'two routes' are replaced by this.")
else:
    verdict = "REJECT"
    print(f"  w_F = {w_F_exact}  ≠  1/12  →  REJECT")
    print("  The Family-D claim c_F = −α₁²/(N·k*) = −α₁²/12 is NOT what the")
    print("  pre-registered δ_r-class residue gives for a CAR single-edge")
    print("  fermion coupling.  Family-D fermion sector is falsified by its")
    print("  own claimed mechanism.  m_τ/m_e/m_μ must be recomputed with the")
    print(f"  residue-true w_F = {w_F_exact} (= 1/(2|E|)²), and reported at")
    print("  honest σ_PDG.  This is the pre-authorized clean negative.")

print(f"\n  VERDICT: {verdict}")
print("=" * 78)
