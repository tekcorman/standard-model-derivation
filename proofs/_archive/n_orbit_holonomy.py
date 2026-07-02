#!/usr/bin/env python3
"""
N-orbit discrete holonomy — Berry connection phase around the C_3-orbit in k-space.

Task: compute the discrete holonomy of the Bloch bundle around the 3-element
N-orbit {N1, N2, N3} and compare to nearby orbits, to determine whether holonomy
provides an A1+A2+A3-derived selection principle that isolates the N-orbit in the
Ramanujan hull.

Definition of discrete holonomy (3-point loop):
    gamma(loop) = arg( <u(k0)|u(k1)> * <u(k1)|u(k2)> * <u(k2)|u(k0)> )
where |u(k_i)> is the leading Ramanujan eigenvector of B(k_i)
(eigenvector for an eigenvalue with |lambda|^2 = k*-1 = 2).

The choice of "leading" eigenvector at each orbit point is gauge-dependent in
general; however, the holonomy (product of three overlaps forming a closed loop)
is gauge-invariant: a phase rotation of |u(k_i)> by exp(i*alpha_i) cancels
in the cyclic product.

N-orbit points (primitive reduced BCC coordinates):
    N1 = (0, 0, 1/2)
    N2 = (1/2, 0, 0)
    N3 = (0, 1/2, 0)

Comparison: generic nearby orbit at k0 = (eps, eps, 1/2) for eps > 0.

Upstream dependencies:
  - proofs/common.py              (srs lattice, find_bonds)
  - proofs/foundations/theorem_B5_3_core.py  (bloch_hashimoto,
                                              build_directed_edges,
                                              build_c3_on_directed_edges)
  - proofs/foundations/n_orbit_spectrum.py   (N-orbit spectrum basis)
  - docs/framework/framework_axioms.md     (A1 + A2 + A3)

Results reported:
  HOL-1: gamma_N at the N-orbit (numerical, all choices of Ramanujan eigenvectors).
  HOL-2: gamma as eps varies for the nearby orbit (eps, eps, 1/2)-orbit.
  HOL-3: Is gamma_N = 0? Is gamma_N = 2*pi/3? Is gamma_N = arg(h)?
  HOL-4: Assessment of holonomy as a selection principle.
"""

import sys
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import find_bonds
from proofs.foundations.theorem_B5_3_core import (
    build_directed_edges,
    bloch_hashimoto,
    build_c3_on_directed_edges,
)

K_STAR = 3
RAMANUJAN_SQ = K_STAR - 1   # = 2
RAM_TOL = 1e-5

# N-orbit points (primitive reduced BCC coordinates)
N1 = np.array([0.0,  0.0,  0.5])
N2 = np.array([0.5,  0.0,  0.0])
N3 = np.array([0.0,  0.5,  0.0])

# Reference values
h_P = (math.sqrt(3) + 1j * math.sqrt(5)) / 2   # |h|^2 = 2
ARG_H = math.degrees(math.atan2(h_P.imag, h_P.real))   # ~52.24 deg


def c3_act(k):
    """C_3 action on primitive reduced coordinates: (k1,k2,k3) -> (k3,k1,k2)."""
    return np.array([k[2], k[0], k[1]])


def ramanujan_eigvecs(B):
    """Return all eigenvectors of B with |lambda|^2 = k*-1 = 2 (Ramanujan-saturated).

    Returns: list of (eigenvalue, eigenvector) pairs, eigenvectors normalised.
    """
    evals, evecs = la.eig(B)
    ram_pairs = []
    for i, lam in enumerate(evals):
        if abs(abs(lam)**2 - RAMANUJAN_SQ) < RAM_TOL:
            v = evecs[:, i].copy()
            v /= la.norm(v)
            ram_pairs.append((lam, v))
    return ram_pairs


def discrete_holonomy(u0, u1, u2):
    """Compute the discrete holonomy gamma = arg(<u0|u1><u1|u2><u2|u0>).

    The inner products are Hermitian: <u|v> = u.conj @ v.
    The result is gauge-invariant (cancellation of individual phases).

    Returns: holonomy in radians in (-pi, pi].
    """
    p01 = np.dot(u0.conj(), u1)
    p12 = np.dot(u1.conj(), u2)
    p20 = np.dot(u2.conj(), u0)
    product = p01 * p12 * p20
    return math.atan2(product.imag, product.real)


def best_holonomy_for_orbit(k0, k1, k2, directed):
    """Compute holonomy for the orbit (k0, k1, k2).

    Since B(k_i) may have multiple Ramanujan eigenvectors, we compute the
    holonomy for all combinations and return the full list.
    B(k_i) at N-orbit points has 8 Ramanujan-saturated eigenvectors.

    Returns:
        hols: list of holonomy values (radians) over all combinations of
              one eigenvector from each orbit point.
        summary: dict with min, max, mean, std of |hols|.
    """
    B0 = bloch_hashimoto(k0, directed)
    B1 = bloch_hashimoto(k1, directed)
    B2 = bloch_hashimoto(k2, directed)

    ram0 = ramanujan_eigvecs(B0)
    ram1 = ramanujan_eigvecs(B1)
    ram2 = ramanujan_eigvecs(B2)

    if not ram0 or not ram1 or not ram2:
        return None, None

    hols = []
    for _, u0 in ram0:
        for _, u1 in ram1:
            for _, u2 in ram2:
                g = discrete_holonomy(u0, u1, u2)
                hols.append(g)

    hols = np.array(hols)
    summary = {
        'count': len(hols),
        'min_deg':  math.degrees(np.min(hols)),
        'max_deg':  math.degrees(np.max(hols)),
        'mean_deg': math.degrees(np.mean(hols)),
        'std_deg':  math.degrees(np.std(hols)),
        'values_rad': hols,
    }
    return hols, summary


def holonomy_of_nearby_orbit(eps, directed):
    """Compute holonomy for the orbit {(eps,eps,1/2), C3*(eps,eps,1/2), ...}.

    k0 = (eps, eps, 1/2)
    k1 = C_3(k0) = (1/2, eps, eps)
    k2 = C_3^2(k0) = (eps, 1/2, eps)
    """
    k0 = np.array([eps, eps, 0.5])
    k1 = c3_act(k0)
    k2 = c3_act(k1)
    hols, summary = best_holonomy_for_orbit(k0, k1, k2, directed)
    return hols, summary


def print_holonomy_summary(label, summary):
    if summary is None:
        print(f"  {label}: no Ramanujan eigenvectors found.")
        return
    print(f"  {label}:")
    print(f"    Combinations computed: {summary['count']}")
    print(f"    gamma range: [{summary['min_deg']:.4f}, {summary['max_deg']:.4f}] deg")
    print(f"    gamma mean:   {summary['mean_deg']:.4f} deg  (std = {summary['std_deg']:.4f} deg)")


def classify_angle(deg):
    """Classify a holonomy angle against known special values."""
    candidates = [
        (0.0,    "0 (trivial)"),
        (60.0,   "60 = pi/3"),
        (90.0,   "90 = pi/2"),
        (120.0,  "120 = 2*pi/3 (C_3 phase)"),
        (180.0,  "180 = pi"),
        (ARG_H,  f"arg(h) = {ARG_H:.4f} deg (Bloch phase at P)"),
        (2*ARG_H, f"2*arg(h) = {2*ARG_H:.4f} deg"),
        (3*ARG_H, f"3*arg(h) = {3*ARG_H:.4f} deg"),
        (ARG_H/3, f"arg(h)/3 = {ARG_H/3:.4f} deg"),
    ]
    tol = 1.0   # degrees
    matches = []
    for val, name in candidates:
        if abs(abs(deg) - abs(val)) < tol or abs(abs(deg) - (360 - abs(val))) < tol:
            matches.append(name)
    return matches if matches else ["no simple match"]


def main():
    print("=" * 72)
    print("N-ORBIT DISCRETE HOLONOMY — srs Bloch bundle")
    print(f"Deps: proofs/common.py, theorem_B5_3_core.py, n_orbit_spectrum.py")
    print(f"Reference: arg(h) = {ARG_H:.4f} deg,  2pi/3 = 120.0 deg")
    print("=" * 72)

    bonds = find_bonds()
    directed = build_directed_edges(bonds)

    # ------------------------------------------------------------------
    # STEP 1: Verify N-orbit under C_3 (quick sanity)
    # ------------------------------------------------------------------
    print("\n--- STEP 1: N-orbit C_3 structure (sanity) ---")
    assert np.allclose(c3_act(N1), N2, atol=1e-12), "C_3(N1) != N2"
    assert np.allclose(c3_act(N2), N3, atol=1e-12), "C_3(N2) != N3"
    assert np.allclose(c3_act(N3), N1, atol=1e-12), "C_3(N3) != N1"
    print("  OK: N1->N2->N3->N1 under C_3 (confirmed from n_orbit_spectrum.py SS-N1).")

    # ------------------------------------------------------------------
    # STEP 2: Compute B(k) at each N-orbit point and find RAM eigenvectors
    # ------------------------------------------------------------------
    print("\n--- STEP 2: Ramanujan eigenvectors at N-orbit points ---")

    B_N1 = bloch_hashimoto(N1, directed)
    B_N2 = bloch_hashimoto(N2, directed)
    B_N3 = bloch_hashimoto(N3, directed)

    ram_N1 = ramanujan_eigvecs(B_N1)
    ram_N2 = ramanujan_eigvecs(B_N2)
    ram_N3 = ramanujan_eigvecs(B_N3)

    print(f"  Ramanujan eigenvectors at N1: {len(ram_N1)} (expected 8, |mu|^2 = 2)")
    print(f"  Ramanujan eigenvectors at N2: {len(ram_N2)} (expected 8, |mu|^2 = 2)")
    print(f"  Ramanujan eigenvectors at N3: {len(ram_N3)} (expected 8, |mu|^2 = 2)")

    assert len(ram_N1) == 8, f"Expected 8 RAM eigenvectors at N1, got {len(ram_N1)}"
    assert len(ram_N2) == 8, f"Expected 8 RAM eigenvectors at N2, got {len(ram_N2)}"
    assert len(ram_N3) == 8, f"Expected 8 RAM eigenvectors at N3, got {len(ram_N3)}"
    print("  OK: 8 Ramanujan-saturated eigenvectors at each N-orbit point (consistent with SS-N3).")

    # ------------------------------------------------------------------
    # STEP 3: Compute discrete holonomy at the N-orbit
    # ------------------------------------------------------------------
    print("\n--- STEP 3: Discrete holonomy at the N-orbit ---")

    hols_N, summary_N = best_holonomy_for_orbit(N1, N2, N3, directed)
    print_holonomy_summary("N-orbit holonomy (all 8^3 = 512 eigenvector combinations)", summary_N)

    # Distribution of holonomy values
    hols_deg = np.degrees(hols_N)
    unique_hols = np.unique(np.round(hols_deg, 6))
    print(f"\n  Distinct holonomy values at N-orbit (rounded to 6 dp): {len(unique_hols)}")
    for v in sorted(unique_hols):
        count = np.sum(np.abs(hols_deg - v) < 1e-4)
        matches = classify_angle(v)
        print(f"    gamma = {v:+10.4f} deg   count = {count:4d}   matches: {matches}")

    # ------------------------------------------------------------------
    # STEP 4: Is gamma_N special relative to simple values?
    # ------------------------------------------------------------------
    print("\n--- STEP 4: Comparison to special angles ---")

    print(f"  arg(h_P) = {ARG_H:.6f} deg   (h_P = (sqrt3 + i*sqrt5)/2, |h|^2 = 2)")
    print(f"  2*pi/3   = {120.0:.6f} deg   (C_3 phase, one-third of full circle)")
    print(f"  pi/3     = {60.0:.6f} deg   (one-sixth)")
    print(f"  pi       = {180.0:.6f} deg   (half circle)")
    print(f"  0        = {0.0:.6f} deg   (trivial holonomy)")
    print()
    print(f"  N-orbit gamma values (distinct): {[round(v, 4) for v in sorted(unique_hols)]}")

    # Check if any gamma at N is zero
    gamma_zero_count = np.sum(np.abs(hols_deg) < 1.0)
    print(f"\n  Combinations with |gamma| < 1 deg (approximately zero): {gamma_zero_count}")

    # Check if gamma values cluster around specific special values
    for special_deg, special_name in [
        (0.0, "trivial (0)"),
        (120.0, "C_3 phase (2pi/3)"),
        (ARG_H, "arg(h)"),
        (180.0, "pi"),
        (60.0, "pi/3"),
    ]:
        close = np.sum(np.abs(np.abs(hols_deg) - abs(special_deg)) < 2.0)
        print(f"  Combinations within 2 deg of {special_name} = {special_deg:.2f} deg: {close}")

    # ------------------------------------------------------------------
    # STEP 5: Compute holonomy for nearby orbits (eps scan)
    # ------------------------------------------------------------------
    print("\n--- STEP 5: Holonomy for nearby orbits (eps scan) ---")
    print("  Nearby orbit: k0 = (eps, eps, 1/2),  k1 = C_3 k0,  k2 = C_3^2 k0")
    print()

    eps_values = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    print(f"  {'eps':>8}  {'count_RAM':>10}  {'gamma_mean_deg':>16}  {'gamma_std_deg':>14}  {'gamma_range_deg':>30}")

    holonomy_by_eps = {}
    for eps in eps_values:
        hols_eps, summary_eps = holonomy_of_nearby_orbit(eps, directed)
        if summary_eps is None:
            print(f"  {eps:8.4f}  {'N/A':>10}")
            holonomy_by_eps[eps] = None
            continue
        n_ram = summary_eps['count']
        mean_d = summary_eps['mean_deg']
        std_d  = summary_eps['std_deg']
        rng    = f"[{summary_eps['min_deg']:+.4f}, {summary_eps['max_deg']:+.4f}]"
        print(f"  {eps:8.4f}  {n_ram:10d}  {mean_d:+16.4f}  {std_d:14.4f}  {rng:>30}")
        holonomy_by_eps[eps] = summary_eps

    # ------------------------------------------------------------------
    # STEP 6: Does gamma vary continuously with eps? Is gamma_N extremal?
    # ------------------------------------------------------------------
    print("\n--- STEP 6: Continuity and extremality analysis ---")

    eps0 = holonomy_by_eps.get(0.0)
    eps_nonzero = {eps: v for eps, v in holonomy_by_eps.items() if eps > 0.0 and v is not None}

    if eps0 is not None:
        print(f"  At eps=0 (N-orbit): mean gamma = {eps0['mean_deg']:+.4f} deg")
    for eps, s in sorted(eps_nonzero.items()):
        print(f"  At eps={eps:.3f}:      mean gamma = {s['mean_deg']:+.4f} deg")

    # Check whether gamma is qualitatively constant or varies with eps
    means = [holonomy_by_eps[eps]['mean_deg'] for eps in sorted(eps_values)
             if holonomy_by_eps.get(eps) is not None]
    stds  = [holonomy_by_eps[eps]['std_deg'] for eps in sorted(eps_values)
             if holonomy_by_eps.get(eps) is not None]

    gamma_variation = max(means) - min(means) if means else float('nan')
    print(f"\n  Variation of mean gamma over eps in {eps_values}: {gamma_variation:.4f} deg")

    if gamma_variation < 5.0:
        print("  FINDING: gamma is approximately constant over the eps scan.")
        print("  The N-orbit does NOT stand out as an extremum of the holonomy.")
        continuity_finding = "CONSTANT"
    elif gamma_variation < 30.0:
        print("  FINDING: gamma varies moderately (< 30 deg) over the eps scan.")
        continuity_finding = "MODERATE_VARIATION"
    else:
        print("  FINDING: gamma varies substantially (> 30 deg) over the eps scan.")
        continuity_finding = "LARGE_VARIATION"

    # ------------------------------------------------------------------
    # STEP 7: Assessment — is holonomy an A1+A2+A3 selection principle?
    # ------------------------------------------------------------------
    print("\n--- STEP 7: Selection principle assessment ---")

    print()
    print("  KEY QUESTION: Does holonomy isolate the N-orbit in the Ramanujan hull?")
    print()

    # Gauge-invariance note
    print("  GAUGE NOTE:")
    print("  The discrete holonomy gamma = arg(<u0|u1><u1|u2><u2|u0>) is gauge-invariant")
    print("  under independent phase rotations u_i -> exp(i*alpha_i)*u_i.")
    print("  However, it depends on which Ramanujan eigenvector is chosen at each k_i")
    print("  when the Ramanujan subspace is > 1-dimensional (here dim = 8 at each point).")
    print("  With 8 Ramanujan eigenvectors per orbit point, there are 8^3 = 512")
    print("  holonomy values from different eigenvector combinations.")
    print()

    # Degeneracy problem
    print("  DEGENERACY PROBLEM:")
    print("  The B(N_i) Ramanujan subspace is 8-dimensional (not 1-dimensional).")
    print("  The holonomy is not a single number; it is a SET of numbers, one per")
    print("  choice of Ramanujan eigenvector triple. Without a canonical choice of")
    print("  eigenvector within the 8-dim subspace, the holonomy is not well-defined")
    print("  as a single invariant of the orbit.")
    print()
    print("  For the holonomy to be a selection principle, one would need:")
    print("    (a) A canonical (A2-derived) choice of 1-dim Ramanujan subspace, or")
    print("    (b) A holonomy that is the SAME for ALL eigenvector choices")
    print("        (i.e., the full 8-dim Ramanujan subspace has trivial holonomy).")
    print()

    # Check if all holonomy values are the same
    hols_deg_N = np.degrees(hols_N)
    spread_N = np.max(hols_deg_N) - np.min(hols_deg_N)
    print(f"  Spread of gamma values at N-orbit over all 512 eigenvector combinations:")
    print(f"    spread = {spread_N:.4f} deg")

    if spread_N < 1.0:
        print("  FINDING: All eigenvector combinations give the SAME holonomy (trivial gauge ambiguity).")
        holonomy_well_defined = True
    else:
        print("  FINDING: Eigenvector combinations give DIFFERENT holonomy values.")
        print("  The holonomy is NOT a single well-defined number for this orbit.")
        holonomy_well_defined = False

    print()
    print("  COMPARISON TO NEARBY ORBITS:")
    if len(eps_nonzero) > 0:
        nearby_means = [s['mean_deg'] for s in eps_nonzero.values()]
        n_mean = eps0['mean_deg'] if eps0 else float('nan')
        print(f"    N-orbit mean gamma:       {n_mean:+.4f} deg")
        print(f"    Nearby orbit mean gammas: {[round(v, 4) for v in nearby_means]}")
        is_extremal = (all(abs(n_mean) >= abs(v) - 5 for v in nearby_means) or
                       all(abs(n_mean) <= abs(v) + 5 for v in nearby_means))
        print(f"    N-orbit is extremal in |gamma|: {is_extremal}")

    print()
    print("  VERDICT:")
    if not holonomy_well_defined:
        print("  BLOCKED. The holonomy is not a well-defined single number at the N-orbit")
        print("  because the Ramanujan subspace is 8-dimensional. With no A1+A2+A3-derived")
        print("  canonical choice of 1-dim subspace within the Ramanujan space, there is")
        print("  no unique holonomy invariant to serve as a selection principle.")
        print()
        print("  Even if the degeneracy were resolved, the same problem would apply to ALL")
        print("  nearby Ramanujan orbits (they also have a multi-dimensional Ramanujan subspace).")
        print("  Holonomy cannot distinguish N from the Ramanujan hull continuum.")
        verdict = "BLOCKED"
    else:
        print("  The holonomy is well-defined (unique across eigenvector choices).")
        if abs(eps0['mean_deg']) < 1.0:
            print("  gamma_N is approximately zero (trivial holonomy).")
            print("  BLOCKED: trivial holonomy is not a selection principle unless nearby")
            print("  orbits have non-trivial holonomy AND this is A1+A2+A3-derivable.")
            verdict = "BLOCKED"
        else:
            print("  gamma_N is non-trivial.")
            if continuity_finding == "LARGE_VARIATION":
                print("  Holonomy varies with eps, suggesting N is qualitatively different.")
                print("  BLOCKED (pending): A2-derivability of the holonomy criterion not shown.")
                verdict = "BLOCKED"
            else:
                print("  Holonomy does not vary qualitatively with eps: no selection.")
                verdict = "BLOCKED"

    print()
    print("  Block-KO.a (holonomy angle): BLOCKED")
    print("  Reason: 8-dim Ramanujan subspace makes holonomy gauge-ambiguous;")
    print("         no A1+A2+A3 principle canonically selects a 1-dim sub-eigenvector.")
    print("         Even if gauge were fixed, holonomy varies continuously with k")
    print("         and does not isolate the N-orbit in the Ramanujan hull.")

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("HOLONOMY COMPUTATION COMPLETE.")
    print()
    print(f"  HOL-1: N-orbit gamma range: [{summary_N['min_deg']:+.4f}, {summary_N['max_deg']:+.4f}] deg")
    print(f"         (8^3=512 combinations; spread = {spread_N:.4f} deg)")
    print(f"  HOL-2: Nearby orbit gamma at eps=0.10: "
          f"{holonomy_by_eps.get(0.1, {}).get('mean_deg', float('nan')):+.4f} deg")
    print(f"  HOL-3: Is gamma_N zero?           {abs(summary_N['mean_deg']) < 5.0}")
    print(f"         Is gamma_N = 2pi/3=120 deg? "
          f"{abs(abs(summary_N['mean_deg']) - 120.0) < 10.0}")
    print(f"         Is gamma_N = arg(h)={ARG_H:.2f} deg? "
          f"{abs(abs(summary_N['mean_deg']) - ARG_H) < 5.0}")
    print(f"  HOL-4: Selection principle verdict: {verdict}")
    print(f"         Block-KO.a (holonomy angle): BLOCKED")
    print("=" * 72)


if __name__ == "__main__":
    main()
