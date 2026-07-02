#!/usr/bin/env python3
"""
y_τ sub-leading Drude-style apparatus — Phase 1A probe.

CONTEXT
-------
Per an internal working note,
Direction 1's working hypothesis was: "the +0.13% un-derived residual is
the BZ-integrated correction beyond the leading Bloch saddle approximation
in y_τ_tree = α_1_full / k*² = (5/3)(2/3)^8 / 9."

UPDATE post-scoping (2026-05-01 evening): per `predictions/alpha_1_full.py`
graduation note 2026-04-20 (session 4), the (5/3) factor is now
structurally derived as n_g_edge/k* = 5/3 (combinatorial graph invariant),
NOT as tan²(arg h(k_P)). The earlier identification with tan²(arg h)
was reclassified as an A2-graduation step. So both factors in α_1_full
are EXACT structural integers; tree-level y_τ has no obvious "saddle
approximation" to be sub-leading-corrected against.

This probe nonetheless asks an empirical question that distinguishes two
possibilities:

  Q. Does the Bloch-saddle interpretation tan²(arg h) coincide with the
     combinatorial n_g_edge/k* = 5/3 IDENTICALLY across the BZ, or only
     at k_P (a "k=3 numerical coincidence" like ε_CP and A_hemispherical
     per `docs/theorems/theorem_class_A_audit.md`)?

If ⟨tan²(arg h(k))⟩_BZ = 5/3 across BZ: the two interpretations are
genuinely equivalent (no saddle approximation lurking).

If ⟨tan²(arg h(k))⟩_BZ ≠ 5/3: the (5/3) factor has Bloch-spectrum
sub-structure that the combinatorial derivation doesn't see. Whether this
relates to the y_τ +0.13% residual is a separate question.

Either result is informative. The probe is a one-shot empirical
check, not a direct y_τ closure attempt. Per
an internal note: derive mechanism
first, compute value second, compare to PDG third.

This script DOES NOT compare to PDG y_τ. It only reports the BZ-averaged
tan²(arg h) and the deviation from 5/3.

Method
------
1. Build the directed-edge Hashimoto operator B(k) on the srs primitive
   cell (12-dim, per `theorem_walker_dynamics.py`).
2. At each BZ point k, find the eigenvalue h(k) closest to the
   Ramanujan-saturating value |h|² = k-1 = 2 (the "Perron-walker"
   eigenvalue at general k).
3. Compute tan²(arg h(k)) at each k.
4. BZ-average uniformly.
5. Compare to 5/3.

References
----------
- `predictions/alpha_1_full.py` graduation note 2026-04-20
- `docs/theorems/theorem_class_A_audit.md` (k=3 numerical coincidences)
- `proofs/foundations/theorem_walker_dynamics.py` (Hashimoto operator)
"""

import math
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from theorem_walker_dynamics import bloch_hashimoto, build_directed_edges
from t_v_eigenstructure import find_bonds


# Build the canonical directed-edge list once.
BONDS = find_bonds()
DIRECTED = build_directed_edges(BONDS)

K_STAR = 3                 # coordination number
RAM_SQ = K_STAR - 1        # |h|² Ramanujan saturation = 2
TAN_SQ_TARGET = 5 / 3      # tan²(arg h) at k_P; n_g_edge/k* combinatorial


def find_perron_walker_eigenvalue(k_frac):
    """At Bloch momentum k_frac (fractional), find the eigenvalue h(k) of
    the Hashimoto operator B(k) closest to Ramanujan saturation |h|² = 2.

    Returns h (complex). At k_P this is exactly (√3+i√5)/2.
    """
    B = bloch_hashimoto(k_frac, DIRECTED)
    eigs = np.linalg.eigvals(B)
    # Pick eigenvalue with |h|² closest to RAM_SQ = 2.
    abs_sq = np.abs(eigs) ** 2
    idx = np.argmin(np.abs(abs_sq - RAM_SQ))
    return eigs[idx]


def tan_sq_arg_h(k_frac):
    """tan²(arg h(k)) = (Im h / Re h)² for the Perron-walker eigenvalue."""
    h = find_perron_walker_eigenvalue(k_frac)
    re = h.real
    im = h.imag
    if abs(re) < 1e-12:
        return float('inf')
    return (im / re) ** 2


def k_p_check():
    """Sanity: at k_P (the framework's canonical P-point), h should be
    (√3+i√5)/2 and tan²(arg h) should be exactly 5/3."""
    # Canonical k_P in fractional coordinates per multiple framework files
    # (theorem_P1_ramanujan_support.py, feshbach_ifeshbach_k4_closure.py, ...)
    k_p = np.array([0.25, 0.25, 0.25])
    h = find_perron_walker_eigenvalue(k_p)
    tan_sq = tan_sq_arg_h(k_p)
    expected_h_sq = 2.0
    expected_tan_sq = 5 / 3
    print(f"  k_P (1/2, 1/2, 1/2) sanity check:")
    print(f"    h(k_P) = {h.real:+.6f} + {h.imag:+.6f}i")
    print(f"    |h(k_P)|² = {abs(h)**2:.6f}  (expected 2.000000)")
    print(f"    tan²(arg h(k_P)) = {tan_sq:.6f}  (expected 5/3 = {5/3:.6f})")
    print(f"    h_match: {'✓' if abs(abs(h)**2 - expected_h_sq) < 1e-3 else '✗'}; "
          f"tan_sq_match: {'✓' if abs(tan_sq - expected_tan_sq) < 1e-3 else '✗'}")


def bz_average_tan_sq(N=12, exclude_distant=True, ram_sq_tol=0.5):
    """BZ-average tan²(arg h(k)) on a uniform N×N×N grid in BCC fractional
    coordinates.

    Parameters
    ----------
    N : int
        Grid size per dimension. Total k-points = N³.
    exclude_distant : bool
        If True, exclude k where |h(k)|² is more than ram_sq_tol away
        from the Ramanujan saturation value 2 (these are k-points where
        the framework's "Perron-walker" identification breaks down — the
        leading eigenvalue isn't Ramanujan-saturating).
    ram_sq_tol : float
        Tolerance for "Ramanujan-like" eigenvalue selection.

    Returns
    -------
    avg : float
        BZ-average of tan²(arg h(k)) across selected k-points.
    n_used : int
        Number of k-points included in the average.
    n_total : int
        Total k-points in the grid.
    """
    grid = np.linspace(0, 1, N, endpoint=False) + 1.0 / (2 * N)  # MP shift
    total = 0.0
    count = 0
    skipped = 0
    skipped_inf = 0
    for k1 in grid:
        for k2 in grid:
            for k3 in grid:
                k_frac = np.array([k1, k2, k3])
                h = find_perron_walker_eigenvalue(k_frac)
                if exclude_distant and abs(abs(h) ** 2 - RAM_SQ) > ram_sq_tol:
                    skipped += 1
                    continue
                if abs(h.real) < 1e-12:
                    skipped_inf += 1
                    continue
                tan_sq = (h.imag / h.real) ** 2
                total += tan_sq
                count += 1
    return (total / count if count > 0 else float('nan')), count, N**3, skipped, skipped_inf


def main():
    print("=" * 78)
    print("y_τ Drude apparatus — Phase 1A probe")
    print("Question: does ⟨tan²(arg h(k))⟩_BZ = 5/3 across the BZ?")
    print("=" * 78)
    print()
    k_p_check()
    print()

    print("BZ averaging tan²(arg h(k)) on uniform grids:")
    print(f"  Target value (combinatorial n_g_edge/k*): {TAN_SQ_TARGET:.6f}")
    print()
    print(f"  {'N':>4s} {'avg':>14s} {'avg-5/3':>12s} {'rel %':>10s} {'used':>8s} {'skip_far':>10s} {'skip_inf':>10s}")
    print("  " + "-" * 76)
    for N in [6, 8, 10, 12]:
        avg, used, total, skipped, skipped_inf = bz_average_tan_sq(N=N)
        diff = avg - TAN_SQ_TARGET
        rel_pct = (diff / TAN_SQ_TARGET) * 100 if TAN_SQ_TARGET != 0 else float('nan')
        print(f"  {N:>4d} {avg:>14.6f} {diff:>+12.6f} {rel_pct:>+9.3f}% "
              f"{used:>4d}/{total:<3d} {skipped:>10d} {skipped_inf:>10d}")

    print()
    print("=" * 78)
    print("EMPIRICAL FINDING (run 2026-05-01 evening)")
    print("=" * 78)
    print()
    print("  k_P sanity check passes: h(k_P) = (√3+i√5)/2 (up to sign), tan² = 5/3 ✓")
    print()
    print("  BZ-averaged tan²(arg h(k)) is NOT close to 5/3 — values range from")
    print("  ~16 to ~143 depending on grid size, with no smooth convergence. The")
    print("  function tan²(arg h(k)) does not behave as a smooth scalar field on")
    print("  the BZ. Root cause: at general k, the Hashimoto operator's eigenvalue")
    print("  with |h|² closest to the Ramanujan saturation 2 has Re h that varies")
    print("  wildly (often near zero, giving huge tan²). The framework's natural")
    print("  identification h = (√3+i√5)/2 at k_P uses additional discrete structure")
    print("  (C_3 stability, parity pairing, ...) that doesn't extend smoothly off")
    print("  k_P.")
    print()
    print("  STRUCTURAL CONCLUSION:")
    print("    (5/3) = tan²(arg h(k_P)) is genuinely a k_P-specific value, not the")
    print("    saddle-evaluation of a smooth BZ-integrated object. A5(a)'s")
    print("    'Ramanujan eigenvalues = SM mass spectrum' selects discrete")
    print("    high-symmetry k-points, not a continuous BZ region. There is no")
    print("    'BZ-integrated tan²(arg h)' for Drude-style apparatus to compute.")
    print()
    print("    Direction 1's saddle-vs-BZ hypothesis on tan²(arg h) is INAPPLICABLE.")
    print("    The +0.13% y_τ residual cannot live in this object.")
    print()
    print("  BROADER IMPLICATIONS for Direction 1:")
    print("    The Drude APPARATUS itself (Kubo on Bloch operator with vertex")
    print("    insertion) may still apply to a different Yukawa-related quantity")
    print("    — but not via the simple tan²(arg h) interpretation. Possible")
    print("    pivots within Direction 1: a vertex-specific matrix Y(k) that's")
    print("    structurally non-trivial (not just identity-proportional contact)")
    print("    AND smoothly defined across BZ.")
    print()
    print("    Cleaner pivots OUT of Direction 1: Direction 2 (sub-leading cycle")
    print("    amplitudes — super-girth contributions to α_1 beyond the geometric")
    print("    series), or Direction 4 (Feshbach analog of 5/12 for the vertex).")
    print()
    print("  Phase 1A produced a useful negative result. No predictions changes.")


if __name__ == '__main__':
    main()
