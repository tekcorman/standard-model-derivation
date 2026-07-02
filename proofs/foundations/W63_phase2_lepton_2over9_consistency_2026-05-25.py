#!/usr/bin/env python3
"""
W63 — Phase 2: lepton 2/9 consistency test for Candidate C
       (variance-of-per-gen-Landauer-entropy = ε²; some moment = δ)

Companion to W62 Phase 1 setup. Tests whether the per-generation Landauer
entropies at the lepton sector (P-saddle, color singlet, chir-5/3 walker)
reduce — under ANY defensible S_j identification — to the observed Koide
parameters (ε² = 2, δ = 2/9).

DISCIPLINE (per the W58 feedback an internal note):
  - Try MULTIPLE defensible S_j choices.
  - Compute Var_j, Skew_j, and C_3-Fourier |S^(1)|, arg(S^(1)) for each.
  - Report ALL results.
  - If NO choice gives (ε² = 2, δ = 2/9) WITHOUT FITTING, declare AB2 fires.
  - If a choice DOES match, the match must come with structural motivation
    independently of the observation — otherwise it's pattern-fitting.

PRE-DECLARED GATES:
  G1: identify ≥ 4 defensible S_j choices (different naturalentropy quantities
      on the per-isotypic walker).
  G2: for each choice, compute (Var_j, Skew_j, |S^(1)|, arg(S^(1))).
  G3: at least one choice gives ε² candidate within ~5% of 2 AND
      δ candidate within ~5% of 2/9 WITHOUT requiring an unmotivated
      rescaling. (5% is the W58 threshold against which patterns get tested.)
  G4: the matching choice's structural motivation is independent of the
      target (ε²=2, δ=2/9) — not derived FROM the target.

PRE-DECLARED ABORTS:
  AB2 (W60 §"Pre-declared aborts"): if Phase 2 cannot reproduce δ_lepton =
       2/9 from the Landauer mechanism alone, mechanism is WRONG, abort.
       Triggers if G3 fails for all defensible choices.

The Phase 2 verdict — PASS or AB2 fires — determines whether the W60 arc
proceeds (to Phase 3, with quark predictions) or pivots to Candidate D
(Berry phase) per W61.
"""

from __future__ import annotations
import math
import sys
import os
import numpy as np
from numpy import linalg as la

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from proofs.common import omega3, find_bonds
from proofs.foundations.theorem_B5_3_core import (
    K_P, build_directed_edges, bloch_hashimoto,
    build_c3_on_directed_edges,
)

np.set_printoptions(precision=4, suppress=True, linewidth=140)
TOL = 1e-8

# Lepton sector targets (theorem-grade values)
EPS2_LEPTON = 2.0           # exact (Koide identity at Q=2/3)
DELTA_LEPTON = 2.0/9.0      # exact (Bernoulli variance Q(1-Q) at Q=2/3 — algebraic identity)

# Tolerance for declaring a match (per W58: 5% match against framework primitives
# is the "easy to find" threshold). Use tight 2% as the actual threshold for
# Phase 2 closure.
TOL_PCT = 2.0


# ------------------------------------------------------------------------
# Build B(P), C_3, decompose into isotypics (same as W62 setup)
# ------------------------------------------------------------------------
print("=" * 78)
print("W63 — Phase 2: lepton 2/9 consistency test (Candidate C of W61)")
print("=" * 78)
print()
print(f"  Target (theorem-grade lepton anchor): ε² = 2, δ = 2/9 ≈ {DELTA_LEPTON:.5f}")
print(f"  Match threshold: {TOL_PCT}%")
print(f"  AB2 fires if NO S_j choice matches both targets within {TOL_PCT}%.")
print()

bonds = find_bonds()
directed = build_directed_edges(bonds)
B_P = bloch_hashimoto(K_P, directed)
U_C3 = build_c3_on_directed_edges(directed)
n = B_P.shape[0]

evals_C3, evecs_C3 = la.eig(U_C3)
isotypic_idx = {
    "trivial": [i for i in range(n) if abs(evals_C3[i] - 1.0) < 1e-6],
    "omega":   [i for i in range(n) if abs(evals_C3[i] - omega3) < 1e-6],
    "omegab":  [i for i in range(n) if abs(evals_C3[i] - omega3**2) < 1e-6],
}

def restrict(B, evecs, idx):
    V = evecs[:, idx]
    Q, _ = la.qr(V)
    V = Q[:, :len(idx)]
    return V.conj().T @ B @ V


B_iso = {label: restrict(B_P, evecs_C3, idx) for label, idx in isotypic_idx.items()}
print(f"  isotypic dims: trivial={B_iso['trivial'].shape[0]}, "
      f"ω={B_iso['omega'].shape[0]}, ω̄={B_iso['omegab'].shape[0]}")
print(f"  (lepton concentrates at P per §4(B); each isotypic is one C_3-gen slot)")
print()


# ------------------------------------------------------------------------
# Multiple defensible S_j candidates
# ------------------------------------------------------------------------
def get_eigvals(B):
    return la.eigvals(B)

def S_max_log(B):
    """S = −log(max|eig|). The naive b1-anchor extension."""
    evs = get_eigvals(B)
    return -math.log(max(abs(e) for e in evs))

def S_mean_log_abs(B):
    """S = −mean(log|eig|). The geometric-mean amplitude entropy."""
    evs = get_eigvals(B)
    log_mags = [math.log(abs(e)) for e in evs if abs(e) > 1e-10]
    return -np.mean(log_mags)

def S_spectral_entropy(B):
    """S = Shannon entropy of normalized |eig|² distribution."""
    evs = get_eigvals(B)
    p = np.abs(evs)**2
    p_norm = p / np.sum(p)
    return -np.sum(p_norm * np.log(np.maximum(p_norm, 1e-10)))

def S_log_dim(B):
    """S = log(dim). Trivial "maximum entropy on uniform distribution" baseline."""
    return math.log(B.shape[0])

def S_trace_log(B):
    """S = −log(|trace(B)/dim| + 1e-10). Mean amplitude entropy."""
    return -math.log(abs(np.trace(B))/B.shape[0] + 1e-10)

S_choices = {
    "max_log":          (S_max_log,         "S = −log(max|eig|) [naive b1 extension]"),
    "mean_log_abs":     (S_mean_log_abs,    "S = −mean(log|eig|) [geometric-mean amp entropy]"),
    "spectral_entropy": (S_spectral_entropy,"S = Shannon entropy of |eig|² distribution"),
    "log_dim":          (S_log_dim,         "S = log(dim) [max-entropy baseline]"),
    "trace_log":        (S_trace_log,       "S = −log(|trace(B)/dim|) [mean amp entropy]"),
}


# ------------------------------------------------------------------------
# For each S_j choice, compute moments + Fourier components
# ------------------------------------------------------------------------
def compute_statistics(S_vals):
    """Compute mean, variance, sample skewness, and C_3 Fourier components."""
    S_arr = np.array(S_vals)
    mean = np.mean(S_arr)
    var = np.var(S_arr)
    # Skewness (population skewness)
    if var > 1e-12:
        sigma = math.sqrt(var)
        skew = np.mean((S_arr - mean)**3) / sigma**3
    else:
        skew = 0.0
    # C_3 Fourier: S^(k) = (1/3) Σ_j ω^(jk) S_j
    S_F0 = (S_arr[0] + S_arr[1] + S_arr[2]) / 3.0
    S_F1 = (S_arr[0] + omega3 * S_arr[1] + omega3**2 * S_arr[2]) / 3.0
    S_F2 = (S_arr[0] + omega3**2 * S_arr[1] + omega3 * S_arr[2]) / 3.0
    return {
        "mean": mean,
        "var": var,
        "skew": skew,
        "S_F0": S_F0,
        "S_F1": S_F1,
        "S_F2": S_F2,
        "|S_F1|": abs(S_F1),
        "arg(S_F1)": np.angle(S_F1),
    }


print("=" * 78)
print("For each S_j choice, compute statistics + check against (ε²=2, δ=2/9)")
print("=" * 78)
print()

# Pre-declared mappings to test (the "natural" maps from {S_j} statistics
# to (ε², δ)). For each S_j choice and each mapping, record whether
# (ε², δ) candidates match (within TOL_PCT).
# NO POST-HOC RESCALING ALLOWED — these mappings are pre-declared.

mappings = [
    ("ε² = Var, δ = Skew",           lambda s: (s["var"], s["skew"])),
    ("ε² = Var, δ = arg(S_F1)",      lambda s: (s["var"], s["arg(S_F1)"])),
    ("ε² = |S_F1|², δ = arg(S_F1)",  lambda s: (s["|S_F1|"]**2, s["arg(S_F1)"])),
    ("ε² = 2·|S_F1|², δ = arg(S_F1)",lambda s: (2*s["|S_F1|"]**2, s["arg(S_F1)"])),
    ("ε² = var/var_max, δ = skew/3", lambda s: (s["var"]/(s["mean"]**2) if s["mean"] != 0 else float('inf'), s["skew"]/3)),
]

any_match = False
match_log = []

for s_label, (s_fn, s_desc) in S_choices.items():
    print(f"--- S_j choice: {s_label} ({s_desc}) ---")
    S_trivial = s_fn(B_iso["trivial"])
    S_omega = s_fn(B_iso["omega"])
    S_omegab = s_fn(B_iso["omegab"])
    print(f"  S_j: trivial = {S_trivial:.4f}, ω = {S_omega:.4f}, ω̄ = {S_omegab:.4f}")
    stats = compute_statistics([S_trivial, S_omega, S_omegab])
    print(f"  stats: mean={stats['mean']:.4f}, var={stats['var']:.4f}, "
          f"skew={stats['skew']:.4f}, |S_F1|={stats['|S_F1|']:.4f}, "
          f"arg(S_F1)={stats['arg(S_F1)']:.4f}")
    for m_label, m_fn in mappings:
        eps2_cand, delta_cand = m_fn(stats)
        eps2_err = abs(eps2_cand - EPS2_LEPTON) / EPS2_LEPTON * 100 if EPS2_LEPTON > 0 else float('inf')
        delta_err = abs(delta_cand - DELTA_LEPTON) / DELTA_LEPTON * 100 if abs(DELTA_LEPTON) > 0 else float('inf')
        both_match = eps2_err < TOL_PCT and delta_err < TOL_PCT
        if both_match:
            any_match = True
            match_log.append((s_label, m_label, eps2_cand, delta_cand, eps2_err, delta_err))
        flag = " <-- BOTH MATCH" if both_match else ""
        print(f"    [{m_label:48s}]  ε²={eps2_cand:+.4f} ({eps2_err:+.1f}%), "
              f"δ={delta_cand:+.4f} ({delta_err:+.1f}%){flag}")
    print()


# ------------------------------------------------------------------------
# Verdict
# ------------------------------------------------------------------------
print("=" * 78)
print("W63 PHASE 2 VERDICT — Candidate C consistency at lepton anchor")
print("=" * 78)
print()

if not any_match:
    print("AB2 FIRES — HONEST NEGATIVE on Candidate C.")
    print()
    print(f"Across {len(S_choices)} defensible S_j choices and {len(mappings)} "
          f"pre-declared moment-mapping candidates ({len(S_choices)*len(mappings)} "
          f"total combinations), NONE reproduce both lepton targets")
    print(f"(ε² = 2, δ = 2/9) within {TOL_PCT}%.")
    print()
    print("The per-generation Landauer entropy statistical moments at the")
    print("lepton sector (P-saddle, color singlet) do NOT reduce to the Koide")
    print("(ε², δ) parameters under any naive identification tested.")
    print()
    print("STRUCTURAL READING (W61 §):")
    print("  The W62-confirmed per-gen Landauer entropies have very specific")
    print("  values at P-saddle determined by V_Ram's (4, 2, 2) C_3-isotypic")
    print("  decomposition. Their statistical moments are SUBSTRATE-DERIVED")
    print("  numbers, BUT they don't match the lepton anchor's (ε², δ)")
    print("  parameters under any natural map.")
    print()
    print("IMPLICATION FOR W60 ARC:")
    print("  Candidate C (variance-of-Landauer-entropy = ε²) is FALSIFIED")
    print("  at the lepton sector. Per W61's pre-declared aborts, this")
    print("  means the §6(i) Landauer-route Candidate B (I-projection")
    print("  deviation) is unlikely to close — the per-gen Landauer entropy")
    print("  quantity-by-itself doesn't carry the Koide (ε², δ) structure.")
    print()
    print("  Per W61, fallback is Candidate D (Berry/geometric phase). The")
    print("  W60 arc should pivot or close-negative.")
else:
    print(f"POSSIBLE MATCHES FOUND ({len(match_log)} combinations match within {TOL_PCT}%):")
    for s_label, m_label, eps2_cand, delta_cand, eps2_err, delta_err in match_log:
        print(f"  {s_label:18s} + {m_label:48s}")
        print(f"    ε² = {eps2_cand:+.4f} ({eps2_err:+.1f}% off target 2)")
        print(f"    δ  = {delta_cand:+.4f} ({delta_err:+.1f}% off target 2/9)")
    print()
    print("HONEST CAUTION (per W58 feedback discipline):")
    print(f"  Among {len(S_choices)*len(mappings)} combinations tested, {len(match_log)} ")
    print(f"  match within {TOL_PCT}%. The probability of finding a {TOL_PCT}%-level")
    print("  match by coincidence depends on the density of candidate")
    print("  identifications.")
    print()
    print("  For this match to qualify as a Phase 2 PASS rather than as")
    print("  pattern-fitting, an A-PRIORI STRUCTURAL motivation for the")
    print("  specific S_j choice and the specific moment mapping must be")
    print("  provided — INDEPENDENTLY of the lepton anchor it matches.")
    print()
    print("  Without that motivation, the match should be treated as")
    print("  candidate-found-but-not-derived, NOT a Phase 2 closure.")

print()
print("=" * 78)
sentinel = "AB2 fires" if not any_match else f"{len(match_log)} match candidate(s)"
print(f"W63 sentinel: {sentinel}")
print("=" * 78)
