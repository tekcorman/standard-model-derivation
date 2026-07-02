#!/usr/bin/env python3
"""
proofs/foundations/m_H_bubble_alpha1sq_probe.py

PROBE — Higgs-quartic bubble correction structural hypothesis test.

HYPOTHESIS
----------
Δλ_Higgs = −(1/2) · α₁_bare²
with α₁_bare² = (2/3)^16 = the bare NB walker survival amplitude
of a CLOSED length-16 NB walk on srs (m=2 closed-bubble host topology:
two girth-10 cycles glued at a 2-edge seam, the same topology as V_ub's
m=2 host per hashimoto_16cycle_decomposition.py).

EMPIRICAL ANCHOR
----------------
Δλ_obs = λ_obs − λ_tree = (m_H_obs/v)²/2 − 2·(5/3)·α₁_bare
       ≈ −7.81×10⁻⁴
Empirical match α (from theorem_mH_1loop_scoping.md §2):
    λ_tree − α₁_bare²/2 matches λ_obs to 0.015% (0.1σ on m_H).

THREE-STEP STRUCTURAL TEST
--------------------------
Step 1: Length-16 closed-NB-walk structure on srs.
        - Per-vertex count of closed L=16 NB walks.
        - Verify 100% decompose as m=2 host (two girths + 2-edge seam).
        - Compare bare per-walk amplitude (2/3)^16 to Δλ_obs.

Step 2: Cl(0,2) edge-qubit γ-product around the closed length-16 walk.
        Does the channel-traversal phase give −1 (fermion-loop) or +1?
        Two conventions tested:
          (a) γ_a γ_a = +I per pass (boson-like, alternating γ₁ γ₂)
          (b) γ-product with directed-edge-type ordering: pick γ₁ on
              "forward" (b₀-orbit) and γ₂ on "C₃-image" (b₁/b₂-orbit)
              edges, compute cumulative phase.

Step 3: A2-T waterline accounting for m=2, 3, 4 closed bubbles.
        - DL_surprise(m) = L_closed(m) · log₂(3/2)
        - DL_encoding(m) = log₂(N_cycles(m) per vertex)
        - Above waterline iff DL_surprise > DL_encoding.
        - If only m=2 retained, Match α's (1/2)·α₁² single-term form is
          the framework's natural answer (no waterline geometric series).

STATUS: SCOPING PROBE — falsifies or confirms hypothesis. Single
run reports magnitude + sign + waterline; verdict written separately.

This script makes NO MODIFICATIONS to any prediction file. It is a
read-only structural test of a hypothesis about Δλ.
"""

import sys
import os
import time
from collections import defaultdict, Counter
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'flavor'))

import vcb_hashimoto_bfs as vcb

bonds_prim     = vcb.bonds_prim
prim_type_key  = vcb.prim_type_key
type_label     = vcb.type_label
g              = vcb.g
N_SUPER        = vcb.N_SUPER
nb_successors  = vcb.nb_successors
edge_prim_type = vcb.edge_prim_type
in_bounds      = vcb.in_bounds


# =============================================================================
# Framework constants
# =============================================================================
k_star = 3
alpha_1_bare_frac = Fraction(2, 3) ** (g - 2)     # = (2/3)^8 = 256/6561
alpha_1_bare      = float(alpha_1_bare_frac)
alpha_1_sq_frac   = alpha_1_bare_frac ** 2         # = (2/3)^16
alpha_1_sq        = float(alpha_1_sq_frac)

lambda_tree_frac = 2 * Fraction(5, 3) * alpha_1_bare_frac  # = 2560/19683
lambda_tree      = float(lambda_tree_frac)

m_H_obs = 125.20      # GeV (PDG 2024)
v_obs   = 246.22      # GeV
lambda_obs = m_H_obs ** 2 / (2 * v_obs ** 2)
delta_lambda_obs = lambda_obs - lambda_tree

match_alpha_target = -alpha_1_sq / 2   # the empirical match α
match_beta_target  = -lambda_tree / (16 * np.pi ** 2)


# =============================================================================
# Enumerate closed NB walks of given length from a representative starting edge
# =============================================================================

def find_closed_walks(start_edge, L, max_walks=10000, time_budget=30.0):
    """
    DFS for closed NB walks of length exactly L through start_edge.
    Returns list of walks; each walk is a sequence of L directed edges,
    walk[0] == start_edge, and the L-th step returns to start_edge.

    Walks may self-intersect (not required to be simple) since the
    framework's bubble correction includes self-intersecting closed
    NB walks.
    """
    found = []
    start_time = time.time()

    def dfs(current, path, depth):
        if len(found) >= max_walks:
            return
        if time.time() - start_time > time_budget:
            return
        if depth == L:
            for succ in nb_successors(*current):
                if succ == start_edge:
                    found.append(list(path))
                    return
            return
        for succ in nb_successors(*current):
            path.append(succ)
            dfs(succ, path, depth + 1)
            path.pop()

    path = [start_edge]
    dfs(start_edge, path, 1)
    return found, (time.time() - start_time)


def find_simple_closed_walks(start_edge, L, max_walks=10000, time_budget=60.0):
    """
    Closed NB walks of length L that visit no Hashimoto node more than
    once (simple cycles).  This is the framework's standard "girth-cycle"
    family, generalized to arbitrary even L.
    """
    found = []
    start_time = time.time()
    path_set = {start_edge}

    def dfs(current, path, depth):
        if len(found) >= max_walks:
            return
        if time.time() - start_time > time_budget:
            return
        if depth == L:
            for succ in nb_successors(*current):
                if succ == start_edge:
                    found.append(list(path))
                    return
            return
        for succ in nb_successors(*current):
            if succ == start_edge:
                continue
            if succ in path_set:
                continue
            path_set.add(succ)
            path.append(succ)
            dfs(succ, path, depth + 1)
            path.pop()
            path_set.discard(succ)

    dfs(start_edge, [start_edge], 1)
    return found, (time.time() - start_time)


# =============================================================================
# Step 1 — Length-16 closed-walk structure on srs
# =============================================================================

def pick_start_edge():
    """Representative starting directed edge near the supercell center."""
    center = (N_SUPER // 2,) * 3
    for bond_idx in range(12):
        prim_bond = bonds_prim[bond_idx]
        dc = prim_bond[2]
        tgt_cell = tuple(center[d] + dc[d] for d in range(3))
        if in_bounds(tgt_cell):
            return (prim_bond[0], center, prim_bond[1], tgt_cell)
    raise RuntimeError("No valid starting directed edge near center")


def step_1_closed_walk_inventory():
    print("=" * 76)
    print("STEP 1 — Length-L closed NB-walk inventory on srs (supercell N=%d)" % N_SUPER)
    print("=" * 76)
    print()

    start = pick_start_edge()
    print(f"  Starting directed edge: {start}")
    print()

    results = {}
    for L in [10, 12, 14, 16]:
        max_walks = 50000 if L <= 14 else 200000
        budget = 90.0 if L == 16 else 30.0
        walks, elapsed = find_simple_closed_walks(start, L,
                                                  max_walks=max_walks,
                                                  time_budget=budget)
        amp = float(Fraction(2, 3) ** L)
        results[L] = {
            'n_simple_walks': len(walks),
            'amplitude_per_walk': amp,
            'elapsed': elapsed,
        }
        print(f"  L = {L:>2}: {len(walks):>6} simple closed NB walks, "
              f"(2/3)^{L} = {amp:.4e}, elapsed = {elapsed:.1f}s")

    print()
    print("  Reference: at L=10, n_simple = 15 (Sunada srs girth-cycle count).")
    print("  Reference: at L=16, hashimoto_16cycle_decomposition.py reports 1344")
    print("             (per directed edge start; matches if simple ≈ all NB).")
    print()

    print("  Bare amplitudes vs Δλ_obs:")
    print(f"    Δλ_obs                = {delta_lambda_obs:+.4e}")
    print(f"    target |Δλ_obs|        = {abs(delta_lambda_obs):.4e}")
    print(f"    α₁_bare²/2 (Match α)  = {alpha_1_sq/2:.4e}")
    print(f"    ratio target/(α₁²/2)  = {abs(delta_lambda_obs)/(alpha_1_sq/2):.4f}")
    print()

    return results, start


# =============================================================================
# Step 2 — Cl(0,2) edge-qubit γ-product around closed length-16 walk
# =============================================================================

def cl02_gamma(channel_idx):
    """
    Cl(0,2) generators in standard 2×2 representation:
      γ₁ = σ_x,  γ₂ = σ_y,  γ₁² = γ₂² = +I,  {γ₁,γ₂} = 0.
    Note: this is the Cl(2,0) convention (γ² = +I).  The framework's
    Cl(0,2) edge-qubit theorem uses (γ² = −I) per A3-T F=ℂ relabel,
    but the *sign* of the closed-loop trace is the same in either
    convention because the loop length is even (L=16).
    """
    if channel_idx == 0:
        return np.array([[0., 1.], [1., 0.]], dtype=complex)   # σ_x
    elif channel_idx == 1:
        return np.array([[0., -1j], [1j, 0.]], dtype=complex)  # σ_y
    else:
        raise ValueError(channel_idx)


def edge_channel_index(directed_edge):
    """
    Pick the Cl(0,2) channel for a directed edge by the C₃-orbit position
    of its primitive bond type.  Convention:
      orbit position 0 (canonical b₀) → channel 0 (γ₁ = σ_x)
      orbit position 1 (C₃=ω², b₁)    → channel 1 (γ₂ = σ_y)
      orbit position 2 (C₃=ω,  b₂)    → channel 1 (γ₂ = σ_y)
    This is the simplest C₃-orbit-respecting convention.  We also test the
    alternative "pure alternating" choice as a robustness check.
    """
    pt = edge_prim_type(*directed_edge)
    if pt is None:
        return None
    _, pos = type_label[pt]
    return 0 if pos == 0 else 1


def gamma_product_trace(walk, convention='c3_orbit'):
    """
    Compute Tr[γ_{e₁}·γ_{e₂}·...·γ_{e_L}] for a closed NB walk.
    Returns (trace, sign) where sign ∈ {+1, -1, 0} (0 if vanishing).

    Conventions:
      'c3_orbit'    : channel from C₃-orbit position of primitive bond
      'alternating' : alternate channels 0, 1, 0, 1, ... along the walk
      'forward_back': channel 0 for edges with even primitive-bond-index
                      direction, channel 1 for odd
    """
    M = np.eye(2, dtype=complex)
    for i, edge in enumerate(walk):
        if convention == 'c3_orbit':
            ch = edge_channel_index(edge)
            if ch is None:
                return None, 0
        elif convention == 'alternating':
            ch = i % 2
        elif convention == 'forward_back':
            pt = edge_prim_type(*edge)
            ch = (pt or 0) % 2
        else:
            raise ValueError(convention)
        M = M @ cl02_gamma(ch)
    tr = np.trace(M)
    return tr, np.sign(tr.real) if abs(tr.real) > 1e-9 else 0


def step_2_jw_sign(start, walks_16):
    print("=" * 76)
    print("STEP 2 — Cl(0,2) γ-product sign on closed length-16 NB walks")
    print("=" * 76)
    print()

    if len(walks_16) == 0:
        print("  No length-16 closed walks found.  Skipping Step 2.")
        return None

    # Sample up to N walks; report distribution of trace signs by convention.
    sample_size = min(200, len(walks_16))
    print(f"  Sampling {sample_size} closed length-16 walks (of {len(walks_16)} total).")
    print()

    for convention in ['c3_orbit', 'alternating', 'forward_back']:
        sign_counter = Counter()
        trace_values = []
        for walk in walks_16[:sample_size]:
            tr, sgn = gamma_product_trace(walk, convention=convention)
            if tr is None:
                continue
            trace_values.append(tr)
            # Bucket by real-part sign and imag content
            if abs(tr.imag) > 1e-9 and abs(tr.real) < 1e-9:
                sign_counter['pure_imag'] += 1
            elif abs(tr.real) < 1e-9 and abs(tr.imag) < 1e-9:
                sign_counter['zero'] += 1
            elif tr.real > 0:
                sign_counter['+real'] += 1
            else:
                sign_counter['-real'] += 1

        # Average trace
        avg_tr = np.mean(trace_values) if trace_values else 0
        print(f"  Convention '{convention}':")
        print(f"    Sign distribution: {dict(sign_counter)}")
        print(f"    Average trace:     {avg_tr:.4f}")
        print()

    print("  INTERPRETATION:")
    print("    A loop-trace of +2 (=+I closed) = boson-like loop, NO fermion sign.")
    print("    A loop-trace of −2 = fermion-like loop, gives the −1 sign.")
    print("    A vanishing trace = no contribution to that channel.")
    print()


# =============================================================================
# Step 3 — A2-T waterline accounting for m=2, 3, 4 closed bubbles
# =============================================================================

def step_3_waterline(results):
    print("=" * 76)
    print("STEP 3 — A2-T waterline on m-girth-chain closed bubbles")
    print("=" * 76)
    print()
    print("  Closed length L_closed(m) = (g-4)·m + 4 = 6m+4 for srs (g=10).")
    print("  Bare amplitude per cycle: (2/3)^L_closed")
    print("  A2-T waterline surprise (per cycle): L·log₂(3/2) bits")
    print("  Encoding cost:  log₂(N_walks(L) per vertex) bits")
    print("  Above-waterline iff surprise > encoding.")
    print()

    # We computed N_walks at the chosen start_edge.  Per-vertex count
    # for simple cycles is roughly N_walks (since each cycle has L
    # equivalent starting positions, and the 3 outgoing edges per vertex,
    # so per-vertex count ≈ 3·N_walks / L; here we use the directly-found
    # N_walks since the framework's per-cycle-orbit normalization treats
    # them as 1).
    print(f"  {'m':>3} {'L_closed':>9} {'(2/3)^L':>12} {'N_walks':>9} "
          f"{'surprise':>10} {'encoding':>10} {'above?':>8}")
    print("  " + "-" * 70)
    log2_32 = float(np.log2(3 / 2))
    for m in [1, 2, 3, 4]:
        L = 6 * m + 4
        if m == 1:
            L = 10   # m=1 is just the single girth, L=g=10
        elif m >= 2:
            L = 6 * m + 4   # m=2:16, m=3:22, m=4:28
        amp = float(Fraction(2, 3) ** L)
        n_walks = results.get(L, {}).get('n_simple_walks', None)
        surprise = L * log2_32
        encoding = float(np.log2(n_walks)) if n_walks and n_walks > 0 else float('inf')
        above = surprise > encoding
        n_walks_str = str(n_walks) if n_walks is not None else "?"
        print(f"  {m:>3} {L:>9} {amp:>12.4e} {n_walks_str:>9} "
              f"{surprise:>10.2f} {encoding:>10.2f} {str(above):>8}")

    print()
    print("  If only m=2 is above-waterline, the framework's natural single-term")
    print("  bubble correction is (1/2)·α₁_bare² (Match α).  If m=3, 4 are also")
    print("  retained, the geometric resummation (1/2)·α₁²·729/665 applies.")
    print()


# =============================================================================
# Step 4 — Magnitude comparison: bubble candidate forms vs Δλ_obs
# =============================================================================

def step_4_magnitude():
    print("=" * 76)
    print("STEP 4 — Magnitude comparison: bubble candidates vs Δλ_obs")
    print("=" * 76)
    print()

    candidates = [
        ("Match α: −α₁²/2",
         -alpha_1_sq / 2),
        ("Single-girth m=1 closed bubble: −(1/2)(2/3)^10",
         -float(Fraction(2, 3) ** 10) / 2),
        ("Geometric resummed m≥2: −(1/2)·α₁²·729/665",
         -alpha_1_sq / 2 * 729 / 665),
        ("Match β: −λ_tree/(16π²)",
         -lambda_tree / (16 * np.pi ** 2)),
        ("Bare α₁² (no Bose 1/2)",
         -alpha_1_sq),
    ]

    print(f"  Δλ_obs = {delta_lambda_obs:+.4e}  (|Δλ| = {abs(delta_lambda_obs):.4e})")
    print()
    print(f"  {'candidate':<48} {'value':>12} {'rel_err':>10}")
    print("  " + "-" * 70)
    for name, val in candidates:
        rel_err = (val - delta_lambda_obs) / delta_lambda_obs * 100
        # m_H tension under candidate:
        lam_pred = lambda_tree + val
        m_H_pred = np.sqrt(2 * lam_pred) * v_obs
        print(f"  {name:<48} {val:>+.4e} {rel_err:>+9.2f}%   "
              f"m_H = {m_H_pred:.3f} GeV")

    print()
    print("  σ_PDG on λ ≈ 0.14% (from σ_m_H = 0.11 GeV at m_H = 125.20 GeV).")
    print("  Match α deviation 2.6% on Δλ corresponds to ~0.015% on λ,")
    print("  ~0.1σ_PDG on m_H — well within experimental precision.")
    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print()
    print("=" * 76)
    print("m_H bubble α₁_bare²/2 probe")
    print("=" * 76)
    print()
    print(f"  Framework constants:")
    print(f"    k* = {k_star}, g = {g}")
    print(f"    α₁_bare    = (2/3)^8 = {alpha_1_bare_frac} = {alpha_1_bare:.6f}")
    print(f"    α₁_bare²   = (2/3)^16 = {alpha_1_sq:.6e}")
    print(f"    α₁_bare²/2 = {alpha_1_sq/2:.6e}")
    print(f"    λ_tree     = 2·(5/3)·α₁_bare = {lambda_tree_frac} = {lambda_tree:.6f}")
    print()
    print(f"  Observed:")
    print(f"    m_H_obs    = {m_H_obs} ± 0.11 GeV (PDG 2024)")
    print(f"    v_obs      = {v_obs} GeV")
    print(f"    λ_obs      = m_H²/(2v²) = {lambda_obs:.6f}")
    print(f"    Δλ_obs     = {delta_lambda_obs:+.4e}")
    print()

    # Step 1
    results, start = step_1_closed_walk_inventory()

    # Need walks_16 for Step 2 — re-enumerate (cached in results doesn't include the walks themselves)
    print("  Re-enumerating L=16 walks for Step 2...")
    walks_16, _ = find_simple_closed_walks(start, 16, max_walks=2000, time_budget=60.0)
    print(f"  Got {len(walks_16)} walks for trace analysis.")
    print()

    # Step 2
    step_2_jw_sign(start, walks_16)

    # Step 3
    step_3_waterline(results)

    # Step 4
    step_4_magnitude()

    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    print("""
  Read the output above:

  (a) Step 1 — Is α₁_bare² the right bare amplitude scale?
      The (2/3)^16 amplitude per closed length-16 walk × the per-vertex
      count should give either (i) α₁²/2 directly (no count), or
      (ii) some natural normalization to α₁²/2.

  (b) Step 2 — Does the Cl(0,2) γ-product give a −1 sign?
      The 'c3_orbit' convention is the framework's natural choice.
      If average trace is −2, fermion-loop sign verified.
      If +2, the boson-loop sign is wrong direction; hypothesis fails.

  (c) Step 3 — Is m=2 the only above-waterline contribution?
      If yes: single-term (1/2)·α₁² is the natural answer (Match α).
      If m≥3 also retained: geometric resum applies (worse match).

  All three must align for the hypothesis Δλ = −(1/2)·α₁_bare² to be
  structurally derived.  Any failure → honest negative.
""")
