#!/usr/bin/env python3
"""
α₂₁ = 162° from band crossing topology on the Γ-P line of the srs BCC BZ.

HYPOTHESIS: The ω band rises from E=-1 (Γ, triplet) to E=+√3 (P, upper pair),
crossing through the trivial band. This crossing is a topological feature.
The Berry phase accumulated through/around this crossing may encode α₂₁ = 162°.

Known exact formula: 10·arctan(2-√3) + π/15 = 9π/10 = 162°
where:
  10 = girth of srs
  2-√3 = spectral gap λ₁ of srs Laplacian
  15 = number of 10-cycles per vertex

We check:
  1. Exact crossing point on Γ-P line
  2. Berry phase of ω band from Γ to P (through the crossing)
  3. Berry phase of ω² band from Γ to P
  4. Wilson loops enclosing the crossing point
  5. Phase combinations involving arctan(√7), arctan(2-√3), girth=10, etc.
"""

import numpy as np
from numpy import linalg as la
from itertools import product
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.dirname(os.path.abspath(__file__))
np.set_printoptions(precision=10, linewidth=120)

# ══════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE (from srs_generation_c3.py)
# ══════════════════════════════════════════════════════════════════════

omega3 = np.exp(2j * np.pi / 3)
NN_DIST = np.sqrt(2) / 4
A_PRIM = np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])

ATOMS = np.array([
    [1/8, 1/8, 1/8],
    [3/8, 7/8, 5/8],
    [7/8, 5/8, 3/8],
    [5/8, 3/8, 7/8],
])
N_ATOMS = 4

C3_PERM = np.zeros((4, 4), dtype=complex)
C3_PERM[0, 0] = 1
C3_PERM[3, 1] = 1
C3_PERM[1, 2] = 1
C3_PERM[2, 3] = 1


def find_bonds():
    tol = 0.02
    bonds = []
    for i in range(N_ATOMS):
        ri = ATOMS[i]
        for j in range(N_ATOMS):
            for n1, n2, n3 in product(range(-2, 3), repeat=3):
                rj = ATOMS[j] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
                dist = la.norm(rj - ri)
                if dist < tol:
                    continue
                if abs(dist - NN_DIST) < tol:
                    bonds.append((i, j, (n1, n2, n3)))
    return bonds


def bloch_H(k_frac, bonds):
    H = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
    k = np.asarray(k_frac, dtype=float)
    for src, tgt, cell in bonds:
        phase = np.exp(2j * np.pi * np.dot(k, cell))
        H[tgt, src] += phase
    return H


def diag_H(k_frac, bonds):
    H = bloch_H(k_frac, bonds)
    evals, evecs = la.eigh(H)
    idx = np.argsort(np.real(evals))
    return np.real(evals[idx]), evecs[:, idx]


def c3_decompose(k_frac, bonds, degen_tol=1e-8):
    evals, evecs = diag_H(k_frac, bonds)
    groups = []
    i = 0
    while i < N_ATOMS:
        grp = [i]
        while i + 1 < N_ATOMS and abs(evals[i+1] - evals[i]) < degen_tol:
            i += 1
            grp.append(i)
        groups.append(grp)
        i += 1

    new_evecs = evecs.copy()
    c3_diag = np.zeros(N_ATOMS, dtype=complex)

    for grp in groups:
        if len(grp) == 1:
            b = grp[0]
            c3_diag[b] = np.conj(evecs[:, b]) @ C3_PERM @ evecs[:, b]
        else:
            sub = evecs[:, grp]
            C3_sub = np.conj(sub.T) @ C3_PERM @ sub
            c3_evals, c3_evecs = la.eig(C3_sub)
            order = np.argsort(np.angle(c3_evals))
            c3_evals = c3_evals[order]
            c3_evecs = c3_evecs[:, order]
            new_sub = sub @ c3_evecs
            for ig, b in enumerate(grp):
                new_evecs[:, b] = new_sub[:, ig]
                c3_diag[b] = c3_evals[ig]

    return evals, new_evecs, c3_diag


def label_c3(c3_val):
    if abs(c3_val - 1.0) < 0.3:
        return '1'
    elif abs(c3_val - omega3) < 0.3:
        return 'w'
    elif abs(c3_val - omega3**2) < 0.3:
        return 'w2'
    return '?'


# ══════════════════════════════════════════════════════════════════════
# 1. FIND EXACT CROSSING POINT ON Γ-P LINE
# ══════════════════════════════════════════════════════════════════════

def find_crossing(bonds, tol=1e-12, max_iter=100):
    """
    Find t* on Γ-P line where E(ω band) = E(upper trivial band).

    k(t) = t * (1/4, 1/4, 1/4), t in [0,1].
    At Γ: all three non-singlet bands at E=-1 (ω below trivial_hi).
    At P: ω at E=+√3, trivial_hi at E=+√3 (they're the upper pair).

    Actually at P, the upper pair is {trivial, ω} both at +√3.
    So the "crossing" is where they approach each other.

    More precisely: track which bands have which C₃ labels and find
    where the ω band energy equals a trivial band energy.
    """
    print("=" * 70)
    print("  1. FINDING EXACT CROSSING POINT ON Gamma-P LINE")
    print("=" * 70)

    # First scan to find approximate crossing region
    n_scan = 2000
    ts = np.linspace(0.001, 0.999, n_scan)
    E_w = np.zeros(n_scan)
    E_w2 = np.zeros(n_scan)
    E_t_lo = np.zeros(n_scan)
    E_t_hi = np.zeros(n_scan)

    for i, t in enumerate(ts):
        k = [t * 0.25, t * 0.25, t * 0.25]
        evals, evecs, c3d = c3_decompose(k, bonds)
        trivials = []
        for b in range(N_ATOMS):
            lab = label_c3(c3d[b])
            if lab == 'w':
                E_w[i] = evals[b]
            elif lab == 'w2':
                E_w2[i] = evals[b]
            elif lab == '1':
                trivials.append(evals[b])
        if len(trivials) == 2:
            trivials.sort()
            E_t_lo[i] = trivials[0]
            E_t_hi[i] = trivials[1]

    # Check: does ω cross trivial_hi?
    diff_w_thi = E_w - E_t_hi
    cross_indices = []
    for i in range(1, n_scan):
        if diff_w_thi[i-1] * diff_w_thi[i] < 0:
            cross_indices.append(i)

    # Also check ω crossing trivial_lo
    diff_w_tlo = E_w - E_t_lo
    cross_lo_indices = []
    for i in range(1, n_scan):
        if diff_w_tlo[i-1] * diff_w_tlo[i] < 0:
            cross_lo_indices.append(i)

    # And ω² crossing trivials
    diff_w2_thi = E_w2 - E_t_hi
    cross_w2_hi = []
    for i in range(1, n_scan):
        if diff_w2_thi[i-1] * diff_w2_thi[i] < 0:
            cross_w2_hi.append(i)

    diff_w2_tlo = E_w2 - E_t_lo
    cross_w2_lo = []
    for i in range(1, n_scan):
        if diff_w2_tlo[i-1] * diff_w2_tlo[i] < 0:
            cross_w2_lo.append(i)

    print(f"\n  Scan: {n_scan} points on t in (0,1)")
    print(f"  omega crosses trivial_hi at {len(cross_indices)} points: t ~ {[ts[i] for i in cross_indices]}")
    print(f"  omega crosses trivial_lo at {len(cross_lo_indices)} points: t ~ {[ts[i] for i in cross_lo_indices]}")
    print(f"  omega2 crosses trivial_hi at {len(cross_w2_hi)} points: t ~ {[ts[i] for i in cross_w2_hi]}")
    print(f"  omega2 crosses trivial_lo at {len(cross_w2_lo)} points: t ~ {[ts[i] for i in cross_w2_lo]}")

    # Report band energies at key points
    for t_val, label in [(0.0, "Gamma"), (0.5, "midpoint"), (1.0, "P")]:
        k = [t_val * 0.25, t_val * 0.25, t_val * 0.25]
        evals, _, c3d = c3_decompose(k, bonds)
        labs = [label_c3(c3d[b]) for b in range(N_ATOMS)]
        print(f"\n  At {label} (t={t_val}):")
        for b in range(N_ATOMS):
            print(f"    Band {b}: E={evals[b]:+.8f}  C3={labs[b]}")

    # If no actual crossing found, check closest approach
    all_crossings = []

    if cross_indices:
        for ci in cross_indices:
            # Bisection to find exact crossing
            t_lo, t_hi = ts[ci-1], ts[ci]
            for _ in range(max_iter):
                t_mid = (t_lo + t_hi) / 2
                k = [t_mid * 0.25] * 3
                evals, _, c3d = c3_decompose(k, bonds)
                e_w, e_th = None, None
                trivs = []
                for b in range(N_ATOMS):
                    lab = label_c3(c3d[b])
                    if lab == 'w':
                        e_w = evals[b]
                    elif lab == '1':
                        trivs.append(evals[b])
                if trivs:
                    e_th = max(trivs)
                if e_w is not None and e_th is not None:
                    if (e_w - e_th) * diff_w_thi[ci-1] > 0:
                        t_lo = t_mid
                    else:
                        t_hi = t_mid
                if abs(t_hi - t_lo) < tol:
                    break
            t_cross = (t_lo + t_hi) / 2
            all_crossings.append(('w_x_thi', t_cross))
            print(f"\n  EXACT crossing (w vs trivial_hi): t* = {t_cross:.15f}")
            # Get energy at crossing
            k = [t_cross * 0.25] * 3
            evals, _, c3d = c3_decompose(k, bonds)
            for b in range(N_ATOMS):
                lab = label_c3(c3d[b])
                print(f"    Band {b}: E={evals[b]:+.12f}  C3={lab}")

    if not cross_indices:
        # Find closest approach
        min_diff = np.min(np.abs(diff_w_thi))
        min_idx = np.argmin(np.abs(diff_w_thi))
        print(f"\n  No crossing found. Closest approach: |E(w)-E(trivial_hi)| = {min_diff:.8f} at t={ts[min_idx]:.6f}")

    # Also find where ω band = 0 (crosses zero energy)
    zero_cross = []
    for i in range(1, n_scan):
        if E_w[i-1] * E_w[i] < 0:
            zero_cross.append(ts[i])
    if zero_cross:
        print(f"\n  omega band crosses E=0 at t ~ {zero_cross}")

    return ts, E_w, E_w2, E_t_lo, E_t_hi, all_crossings


# ══════════════════════════════════════════════════════════════════════
# 2. BERRY PHASE ALONG Γ-P LINE
# ══════════════════════════════════════════════════════════════════════

def berry_phase_line(bonds, band_label='w', n_pts=10000):
    """
    Compute Berry phase accumulated by a specific C₃-labeled band
    along the Γ-P line.

    φ_Berry = -Im Σ log <ψ(k_i)|ψ(k_{i+1})>

    This is the discretized version of the Berry connection integral.
    We track the band by its C₃ label at each k-point.
    """
    print(f"\n  Berry phase for {band_label} band along Gamma -> P...")

    ts = np.linspace(0.001, 0.999, n_pts)
    states = []

    for t in ts:
        k = [t * 0.25] * 3
        evals, evecs, c3d = c3_decompose(k, bonds)
        for b in range(N_ATOMS):
            if label_c3(c3d[b]) == band_label:
                states.append(evecs[:, b].copy())
                break
        else:
            # Fallback: use closest C₃ eigenvalue
            if band_label == 'w':
                target = omega3
            elif band_label == 'w2':
                target = omega3**2
            else:
                target = 1.0
            dists = [abs(c3d[b] - target) for b in range(N_ATOMS)]
            best = np.argmin(dists)
            states.append(evecs[:, best].copy())

    # Compute Berry phase
    phase = 0.0
    for i in range(len(states) - 1):
        overlap = np.dot(np.conj(states[i]), states[i+1])
        # Gauge fix: make overlap real and positive where possible
        phase -= np.imag(np.log(overlap))

    return phase, np.degrees(phase)


def berry_phase_GP(bonds):
    """Berry phases for all C₃-labeled bands along Γ-P."""
    print("\n" + "=" * 70)
    print("  2. BERRY PHASES ALONG Gamma-P LINE")
    print("=" * 70)

    results = {}
    for label in ['w', 'w2', '1']:
        phase_rad, phase_deg = berry_phase_line(bonds, label, n_pts=10000)
        results[label] = (phase_rad, phase_deg)
        print(f"    {label:4s}: phi = {phase_rad:+.10f} rad = {phase_deg:+.6f} deg")

    # Check key values
    target = 162.0
    print(f"\n  Target: alpha_21 = {target} deg = {np.radians(target):.10f} rad")
    for label, (pr, pd) in results.items():
        diff = abs(abs(pd) - target)
        print(f"    |phi({label})| - 162 = {abs(pd):.6f} - 162 = {diff:+.6f} deg")

    # Also check differences
    if 'w' in results and 'w2' in results:
        diff_phase = results['w'][1] - results['w2'][1]
        print(f"\n  phi(w) - phi(w2) = {diff_phase:+.6f} deg")
        print(f"  |phi(w) - phi(w2)| - 162 = {abs(diff_phase) - 162:+.6f} deg")

    return results


# ══════════════════════════════════════════════════════════════════════
# 3. WILSON LOOPS AROUND CROSSING POINT
# ══════════════════════════════════════════════════════════════════════

def wilson_loop_around_crossing(bonds, t_center, radius, band_label='w', n_pts=500):
    """
    Wilson loop in the plane perpendicular to (111) at the crossing point.

    The crossing point is at k = t_center * (1/4,1/4,1/4) in fractional coords.
    We construct a circular loop in the plane perpendicular to (1,1,1)
    at this k-point.

    Perpendicular directions in fractional space:
      e1 = (1,-1,0)/sqrt(2)
      e2 = (1,1,-2)/sqrt(6)
    """
    # Center in fractional coords
    k_c = np.array([t_center * 0.25] * 3)

    # Perpendicular directions (fractional)
    e1 = np.array([1, -1, 0]) / np.sqrt(2)
    e2 = np.array([1, 1, -2]) / np.sqrt(6)

    thetas = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    states = []
    for theta in thetas:
        k = k_c + radius * (np.cos(theta) * e1 + np.sin(theta) * e2)
        evals, evecs, c3d = c3_decompose(k, bonds)

        # Find band by C₃ label
        found = False
        for b in range(N_ATOMS):
            if label_c3(c3d[b]) == band_label:
                states.append(evecs[:, b].copy())
                found = True
                break

        if not found:
            # Off the C₃ axis: C₃ not a good quantum number
            # Track by energy continuity instead
            if states:
                overlaps = [abs(np.dot(np.conj(states[-1]), evecs[:, b]))**2
                            for b in range(N_ATOMS)]
                best = np.argmax(overlaps)
                states.append(evecs[:, best].copy())
            else:
                # First point: use closest C₃ eigenvalue
                target_c3 = omega3 if band_label == 'w' else (omega3**2 if band_label == 'w2' else 1.0)
                dists = [abs(c3d[b] - target_c3) for b in range(N_ATOMS)]
                best = np.argmin(dists)
                states.append(evecs[:, best].copy())

    # Wilson loop phase
    phase = 0.0
    for i in range(len(states)):
        j = (i + 1) % len(states)
        overlap = np.dot(np.conj(states[i]), states[j])
        phase -= np.imag(np.log(overlap))

    return phase, np.degrees(phase)


def wilson_loops_scan(bonds, crossings):
    """Wilson loops at multiple radii around each crossing point."""
    print("\n" + "=" * 70)
    print("  3. WILSON LOOPS AROUND CROSSING POINTS")
    print("=" * 70)

    # If no crossings found, use key points along the line
    if not crossings:
        # Use the point where ω band crosses E=0 and several other points
        test_points = [0.25, 0.5, 0.75]
        print(f"  No crossings found. Testing Wilson loops at t = {test_points}")
    else:
        test_points = [c[1] for c in crossings]

    radii = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    target = 162.0

    all_results = {}

    for t_c in test_points:
        print(f"\n  Wilson loops around t = {t_c:.6f}:")
        for band_label in ['w', 'w2']:
            print(f"    Band: {band_label}")
            for r in radii:
                phase_rad, phase_deg = wilson_loop_around_crossing(
                    bonds, t_c, r, band_label=band_label, n_pts=500)
                diff = abs(abs(phase_deg) - target)
                marker = " <<<" if diff < 1.0 else ""
                print(f"      r={r:.4f}: phi = {phase_deg:+.4f} deg  "
                      f"(|phi|-162 = {diff:+.4f}){marker}")
                all_results[(t_c, band_label, r)] = (phase_rad, phase_deg)

    # Also try non-Abelian Wilson loop (2x2 for the ω,ω² pair)
    print(f"\n  Non-Abelian Wilson loop (w,w2 pair):")
    for t_c in test_points:
        for r in [0.01, 0.05, 0.1, 0.2]:
            W = non_abelian_wilson(bonds, t_c, r, n_pts=500)
            eig_W = la.eigvals(W)
            phases_W = np.degrees(np.angle(eig_W))
            print(f"    t={t_c:.4f}, r={r:.4f}: Wilson eigenphases = {phases_W}")
            for p in phases_W:
                if abs(abs(p) - target) < 2.0:
                    print(f"      ^^^ NEAR 162 deg!")

    return all_results


def non_abelian_wilson(bonds, t_center, radius, n_pts=500):
    """
    Non-Abelian Wilson loop for the {ω, ω²} two-band subspace.
    Returns 2x2 Wilson loop matrix.
    """
    k_c = np.array([t_center * 0.25] * 3)
    e1 = np.array([1, -1, 0]) / np.sqrt(2)
    e2 = np.array([1, 1, -2]) / np.sqrt(6)

    thetas = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    # Collect 2-band subspace at each point
    subspaces = []
    for theta in thetas:
        k = k_c + radius * (np.cos(theta) * e1 + np.sin(theta) * e2)
        evals, evecs, c3d = c3_decompose(k, bonds)

        # Find ω and ω² bands
        w_vec = None
        w2_vec = None
        for b in range(N_ATOMS):
            lab = label_c3(c3d[b])
            if lab == 'w':
                w_vec = evecs[:, b].copy()
            elif lab == 'w2':
                w2_vec = evecs[:, b].copy()

        if w_vec is None or w2_vec is None:
            # Off-axis: track by overlap with previous
            if subspaces:
                prev = subspaces[-1]
                overlaps = np.abs(np.conj(prev.T) @ evecs)**2
                # Best match for each previous band
                for col in range(2):
                    best = np.argmax(overlaps[col])
                    if col == 0:
                        w_vec = evecs[:, best].copy()
                    else:
                        w2_vec = evecs[:, best].copy()
            else:
                # Use C₃ eigenvalue proximity
                w_dists = [abs(c3d[b] - omega3) for b in range(N_ATOMS)]
                w2_dists = [abs(c3d[b] - omega3**2) for b in range(N_ATOMS)]
                w_vec = evecs[:, np.argmin(w_dists)].copy()
                w2_vec = evecs[:, np.argmin(w2_dists)].copy()

        subspaces.append(np.column_stack([w_vec, w2_vec]))

    # Compute Wilson loop: W = prod_i P(i+1)^dag P(i)
    W = np.eye(2, dtype=complex)
    for i in range(len(subspaces)):
        j = (i + 1) % len(subspaces)
        overlap = np.conj(subspaces[j].T) @ subspaces[i]  # 2x2
        W = overlap @ W

    return W


# ══════════════════════════════════════════════════════════════════════
# 4. PHASE COMBINATIONS AND NUMBER THEORY
# ══════════════════════════════════════════════════════════════════════

def phase_analysis(berry_results, wilson_results, crossings, bonds):
    """
    Check if any computed phases match α₂₁ = 162° via known algebraic relations.
    """
    print("\n" + "=" * 70)
    print("  4. PHASE COMBINATION ANALYSIS")
    print("=" * 70)

    # Key constants
    arctan_sqrt7 = np.degrees(np.arctan(np.sqrt(7)))  # Ihara K₄ triplet phase
    arctan_2ms3 = np.degrees(np.arctan(2 - np.sqrt(3)))  # spectral gap phase = π/12 = 15°
    girth = 10
    cycles_per_vertex = 15
    target = 162.0  # = 9π/10

    print(f"\n  Reference values:")
    print(f"    arctan(sqrt(7)) = {arctan_sqrt7:.10f} deg")
    print(f"    arctan(2-sqrt(3)) = {arctan_2ms3:.10f} deg  (= pi/12 = 15 deg)")
    print(f"    girth = {girth}")
    print(f"    10-cycles per vertex = {cycles_per_vertex}")
    print(f"    target: 162 = 10 * 15 + 180/15 = 150 + 12 = 162")
    print(f"    verify: 10*arctan(2-sqrt(3)) + pi/15 = {girth * arctan_2ms3 + 180/cycles_per_vertex:.10f} deg")

    # Collect all computed phases
    all_phases = []
    for label, (pr, pd) in berry_results.items():
        all_phases.append((f"Berry({label})", pd))

    for (tc, bl, r), (pr, pd) in wilson_results.items():
        all_phases.append((f"Wilson(t={tc:.3f},{bl},r={r:.3f})", pd))

    # Check each phase against target
    print(f"\n  Direct matches with 162 deg:")
    for name, phase in all_phases:
        diff = abs(abs(phase) - target)
        if diff < 5.0:
            print(f"    {name}: {phase:+.6f} deg  (off by {diff:.6f})")

    # Check combinations of phases
    print(f"\n  Combination checks:")
    berry_phases = {k: v[1] for k, v in berry_results.items()}

    if 'w' in berry_phases and 'w2' in berry_phases:
        diff_ww2 = berry_phases['w'] - berry_phases['w2']
        print(f"    Berry(w) - Berry(w2) = {diff_ww2:.6f} deg")

        # Check if diff_ww2 relates to arctan values
        ratio1 = diff_ww2 / arctan_sqrt7
        ratio2 = diff_ww2 / arctan_2ms3
        print(f"    / arctan(sqrt7) = {ratio1:.6f}")
        print(f"    / arctan(2-sqrt3) = {ratio2:.6f}")
        print(f"    / (pi/3 = 60) = {diff_ww2 / 60:.6f}")
        print(f"    / (pi/6 = 30) = {diff_ww2 / 30:.6f}")

    # Check crossing point parameter t*
    if crossings:
        for name, t_c in crossings:
            print(f"\n  Crossing at t* = {t_c:.15f}")
            k_cross = t_c * 0.25
            print(f"    k_cross (fractional) = {k_cross:.15f}")
            # Energy at crossing
            k = [k_cross] * 3
            evals, _, _ = c3_decompose(k, bonds)
            E_cross = evals[1]  # approximate
            print(f"    E at crossing ~ {evals}")

            # Check if t* has algebraic significance
            print(f"    t* / (2-sqrt(3)) = {t_c / (2-np.sqrt(3)):.10f}")
            print(f"    arctan(t*) = {np.degrees(np.arctan(t_c)):.10f} deg")
            print(f"    arctan(E_cross) = {np.degrees(np.arctan(E_cross)):.10f} deg")

    # Berry phase from specific sub-segments
    print(f"\n  Sub-segment Berry phases:")
    if crossings:
        t_cross = crossings[0][1]
        # Gamma to crossing
        phase_to_cross = berry_phase_segment(bonds, 0.001, t_cross, 'w', 5000)
        # Crossing to P
        phase_from_cross = berry_phase_segment(bonds, t_cross, 0.999, 'w', 5000)
        print(f"    Berry(w, Gamma->crossing): {np.degrees(phase_to_cross):+.6f} deg")
        print(f"    Berry(w, crossing->P):     {np.degrees(phase_from_cross):+.6f} deg")
        print(f"    Sum:                        {np.degrees(phase_to_cross + phase_from_cross):+.6f} deg")

        diff1 = abs(abs(np.degrees(phase_to_cross)) - target)
        diff2 = abs(abs(np.degrees(phase_from_cross)) - target)
        if diff1 < 5:
            print(f"    *** Gamma->crossing segment near 162! (off by {diff1:.4f})")
        if diff2 < 5:
            print(f"    *** crossing->P segment near 162! (off by {diff2:.4f})")


def berry_phase_segment(bonds, t_start, t_end, band_label, n_pts):
    """Berry phase for a C₃-labeled band over a segment of the Γ-P line."""
    ts = np.linspace(t_start, t_end, n_pts)
    states = []

    for t in ts:
        k = [t * 0.25] * 3
        evals, evecs, c3d = c3_decompose(k, bonds)
        target_c3 = omega3 if band_label == 'w' else (omega3**2 if band_label == 'w2' else 1.0)
        # Find best match
        best = None
        for b in range(N_ATOMS):
            if label_c3(c3d[b]) == band_label:
                best = b
                break
        if best is None:
            dists = [abs(c3d[b] - target_c3) for b in range(N_ATOMS)]
            best = np.argmin(dists)
        states.append(evecs[:, best].copy())

    phase = 0.0
    for i in range(len(states) - 1):
        overlap = np.dot(np.conj(states[i]), states[i+1])
        phase -= np.imag(np.log(overlap))

    return phase


# ══════════════════════════════════════════════════════════════════════
# 5. BAND CROSSING GEOMETRY AND GIRTH ENCODING
# ══════════════════════════════════════════════════════════════════════

def crossing_geometry(bonds, crossings):
    """
    Investigate the geometry of the band crossing.

    At a band crossing of two bands with different C₃ quantum numbers,
    the crossing is PROTECTED (bands don't hybridize because they carry
    different irrep labels). This is a symmetry-protected crossing.

    Check: does the crossing angle or energy encode girth=10,
    spectral gap 2-√3, or 15 ten-cycles?
    """
    print("\n" + "=" * 70)
    print("  5. BAND CROSSING GEOMETRY")
    print("=" * 70)

    # Compute band velocities (dE/dt) at crossing
    dt = 1e-6
    if crossings:
        for name, t_c in crossings:
            print(f"\n  Crossing: {name} at t* = {t_c:.12f}")

            # Velocities
            k_m = [(t_c - dt) * 0.25] * 3
            k_p = [(t_c + dt) * 0.25] * 3
            evals_m, _, c3d_m = c3_decompose(k_m, bonds)
            evals_p, _, c3d_p = c3_decompose(k_p, bonds)

            # Identify bands by C₃ label
            for label in ['w', 'w2', '1']:
                E_m, E_p = None, None
                trivials_m, trivials_p = [], []
                for b in range(N_ATOMS):
                    lm = label_c3(c3d_m[b])
                    lp = label_c3(c3d_p[b])
                    if lm == label and label != '1':
                        E_m = evals_m[b]
                    if lp == label and label != '1':
                        E_p = evals_p[b]
                    if lm == '1':
                        trivials_m.append(evals_m[b])
                    if lp == '1':
                        trivials_p.append(evals_p[b])

                if label == '1' and trivials_m and trivials_p:
                    trivials_m.sort()
                    trivials_p.sort()
                    for idx, sub in enumerate(['lo', 'hi']):
                        if idx < len(trivials_m):
                            vel = (trivials_p[idx] - trivials_m[idx]) / (2 * dt)
                            print(f"    dE/dt({label}_{sub}) = {vel:+.6f}")
                elif E_m is not None and E_p is not None:
                    vel = (E_p - E_m) / (2 * dt)
                    print(f"    dE/dt({label}) = {vel:+.6f}")

            # Crossing angle between ω and trivial
            # Get velocities of the two crossing bands
            v_w, v_t = None, None
            for b in range(N_ATOMS):
                if label_c3(c3d_m[b]) == 'w':
                    e_w_m = evals_m[b]
                if label_c3(c3d_p[b]) == 'w':
                    e_w_p = evals_p[b]
            v_w = (e_w_p - e_w_m) / (2 * dt)

            trivials_m.sort()
            trivials_p.sort()
            if len(trivials_m) >= 2 and len(trivials_p) >= 2:
                v_t = (trivials_p[-1] - trivials_m[-1]) / (2 * dt)

                if v_w is not None and v_t is not None:
                    # Crossing angle
                    angle = np.degrees(np.arctan(abs(v_w - v_t)))
                    print(f"\n    Band velocities at crossing:")
                    print(f"      v(w) = {v_w:+.8f}")
                    print(f"      v(trivial_hi) = {v_t:+.8f}")
                    print(f"      v_rel = {abs(v_w - v_t):.8f}")
                    print(f"      crossing angle = arctan(v_rel) = {angle:.6f} deg")

                    # Check against known values
                    print(f"      angle / arctan(2-sqrt3) = {angle / np.degrees(np.arctan(2-np.sqrt(3))):.6f}")
                    print(f"      angle / arctan(sqrt7) = {angle / np.degrees(np.arctan(np.sqrt(7))):.6f}")

    # Energy at crossing vs algebraic values
    print(f"\n  Key algebraic energies:")
    print(f"    sqrt(3) = {np.sqrt(3):.10f}")
    print(f"    2-sqrt(3) = {2-np.sqrt(3):.10f}")
    print(f"    sqrt(7) = {np.sqrt(7):.10f}")

    # Check: what is E at the crossing?
    if crossings:
        t_c = crossings[0][1]
        k = [t_c * 0.25] * 3
        evals, _, _ = c3_decompose(k, bonds)
        print(f"\n    E at crossing (t={t_c:.8f}):")
        for b, e in enumerate(evals):
            print(f"      Band {b}: {e:+.10f}")
            # Check ratios
            if abs(e) > 0.01:
                print(f"        E/sqrt(3) = {e/np.sqrt(3):.10f}")
                print(f"        arctan(E) = {np.degrees(np.arctan(e)):.6f} deg")


# ══════════════════════════════════════════════════════════════════════
# 6. ADDITIONAL BERRY PHASE CALCULATIONS
# ══════════════════════════════════════════════════════════════════════

def berry_phase_closed_loops(bonds):
    """
    Berry phases for closed loops in the BZ.

    Key loops:
    1. Full Γ-P-Γ' (along 111, period = 4 in t)
    2. Loop around the P point in the perpendicular plane
    3. Triangular loop Γ-P-H-Γ (encloses part of BZ)
    """
    print("\n" + "=" * 70)
    print("  6. BERRY PHASES FOR CLOSED BZ LOOPS")
    print("=" * 70)

    target = 162.0

    # Loop 1: Along (111) from Γ to Γ' (full period)
    # k = t*(1/4,1/4,1/4), t in [0,4] wraps the BZ
    print(f"\n  Loop along (111): Gamma -> P -> Gamma' (full period)")
    n_pts = 20000
    ts = np.linspace(0, 4.0, n_pts, endpoint=False)

    for band_label in ['w', 'w2']:
        states = []
        for t in ts:
            k = [t * 0.25] * 3
            evals, evecs, c3d = c3_decompose(k, bonds)
            target_c3 = omega3 if band_label == 'w' else omega3**2
            # Track by overlap
            if states:
                overlaps = [abs(np.dot(np.conj(states[-1]), evecs[:, b]))**2
                            for b in range(N_ATOMS)]
                best = np.argmax(overlaps)
            else:
                dists = [abs(c3d[b] - target_c3) for b in range(N_ATOMS)]
                best = np.argmin(dists)
            states.append(evecs[:, best].copy())

        # Close the loop
        phase = 0.0
        for i in range(len(states)):
            j = (i + 1) % len(states)
            overlap = np.dot(np.conj(states[i]), states[j])
            phase -= np.imag(np.log(overlap))

        phase_deg = np.degrees(phase)
        diff = abs(abs(phase_deg) - target)
        marker = " <<<" if diff < 2.0 else ""
        print(f"    {band_label}: phi = {phase_deg:+.6f} deg  (|phi|-162 = {diff:+.4f}){marker}")

    # Loop 2: Hexagonal loop in perpendicular plane at P
    print(f"\n  Hexagonal loops perpendicular to (111) at P:")
    k_P = np.array([0.25, 0.25, 0.25])
    e1 = np.array([1, -1, 0]) / np.sqrt(2)
    e2 = np.array([1, 1, -2]) / np.sqrt(6)

    for r in [0.05, 0.1, 0.15, 0.2, 0.25]:
        for band_label in ['w', 'w2']:
            n_hex = 600
            thetas = np.linspace(0, 2*np.pi, n_hex, endpoint=False)
            states = []

            for theta in thetas:
                k = k_P + r * (np.cos(theta) * e1 + np.sin(theta) * e2)
                evals, evecs, c3d = c3_decompose(k, bonds)

                if states:
                    overlaps = [abs(np.dot(np.conj(states[-1]), evecs[:, b]))**2
                                for b in range(N_ATOMS)]
                    best = np.argmax(overlaps)
                else:
                    target_c3 = omega3 if band_label == 'w' else omega3**2
                    dists = [abs(c3d[b] - target_c3) for b in range(N_ATOMS)]
                    best = np.argmin(dists)
                states.append(evecs[:, best].copy())

            phase = 0.0
            for i in range(len(states)):
                j = (i + 1) % len(states)
                overlap = np.dot(np.conj(states[i]), states[j])
                phase -= np.imag(np.log(overlap))

            phase_deg = np.degrees(phase)
            diff = abs(abs(phase_deg) - target)
            marker = " <<<" if diff < 2.0 else ""
            print(f"    r={r:.2f}, {band_label}: phi = {phase_deg:+.6f} deg{marker}")

    # Loop 3: Great circle through P in the BZ
    print(f"\n  Great circles through P:")

    # Circle in the (k1=k2=k3) plane but tilted
    for tilt in [0, np.pi/6, np.pi/4, np.pi/3]:
        for band_label in ['w']:
            n_gc = 1000
            thetas = np.linspace(0, 2*np.pi, n_gc, endpoint=False)
            states = []

            for theta in thetas:
                # Parameterize a circle that passes through P
                k1 = 0.25 + 0.25 * np.cos(theta)
                k2 = 0.25 + 0.25 * np.cos(theta + tilt)
                k3 = 0.25 + 0.25 * np.sin(theta)
                k = [k1, k2, k3]

                evals, evecs, c3d = c3_decompose(k, bonds)
                if states:
                    overlaps = [abs(np.dot(np.conj(states[-1]), evecs[:, b]))**2
                                for b in range(N_ATOMS)]
                    best = np.argmax(overlaps)
                else:
                    dists = [abs(c3d[b] - omega3) for b in range(N_ATOMS)]
                    best = np.argmin(dists)
                states.append(evecs[:, best].copy())

            phase = 0.0
            for i in range(len(states)):
                j = (i + 1) % len(states)
                overlap = np.dot(np.conj(states[i]), states[j])
                phase -= np.imag(np.log(overlap))

            phase_deg = np.degrees(phase)
            diff = abs(abs(phase_deg) - target)
            marker = " <<<" if diff < 2.0 else ""
            print(f"    tilt={np.degrees(tilt):.0f} deg, w: phi = {phase_deg:+.6f} deg{marker}")

    return


# ══════════════════════════════════════════════════════════════════════
# 7. ZETA FUNCTION / SPECTRAL APPROACH
# ══════════════════════════════════════════════════════════════════════

def spectral_analysis(bonds, crossings):
    """
    Check if the crossing point encodes spectral gap / girth information.

    The Ihara zeta function of srs encodes girth=10 and spectral gap 2-√3.
    If the band crossing is related, its position should encode these.
    """
    print("\n" + "=" * 70)
    print("  7. SPECTRAL / ZETA FUNCTION ANALYSIS")
    print("=" * 70)

    lambda1 = 2 - np.sqrt(3)  # spectral gap
    girth = 10

    if crossings:
        t_c = crossings[0][1]
    else:
        t_c = 0.5  # fallback

    # Compute the full spectrum along Γ-P and look for spectral gap encoding
    n_pts = 2000
    ts = np.linspace(0, 1, n_pts)
    E_all = np.zeros((n_pts, N_ATOMS))

    for i, t in enumerate(ts):
        k = [t * 0.25] * 3
        evals, _, _ = c3_decompose(k, bonds)
        E_all[i] = evals

    # Bandwidth of each band
    for b in range(N_ATOMS):
        bw = E_all[:, b].max() - E_all[:, b].min()
        print(f"  Band {b}: width = {bw:.8f}, range = [{E_all[:, b].min():.6f}, {E_all[:, b].max():.6f}]")

    # Ratio of bandwidths
    widths = [E_all[:, b].max() - E_all[:, b].min() for b in range(N_ATOMS)]
    for i in range(N_ATOMS):
        for j in range(i+1, N_ATOMS):
            ratio = widths[i] / widths[j] if widths[j] > 0 else float('inf')
            print(f"  width[{i}]/width[{j}] = {ratio:.8f}")
            if abs(ratio - lambda1) < 0.1:
                print(f"    ^^^ NEAR spectral gap 2-sqrt(3) = {lambda1:.8f}")

    # Check the crossing energy
    if crossings:
        t_c = crossings[0][1]
        k = [t_c * 0.25] * 3
        evals, _, _ = c3_decompose(k, bonds)
        E_cross = sorted(evals)[1]  # second lowest

        print(f"\n  E at crossing = {E_cross:.10f}")
        print(f"  E / (2-sqrt3) = {E_cross / lambda1:.10f}")
        print(f"  E / sqrt(3) = {E_cross / np.sqrt(3):.10f}")
        print(f"  arctan(E) * 10 = {np.degrees(np.arctan(E_cross)) * 10:.6f} deg")
        print(f"  arctan(E) * 10 + 12 = {np.degrees(np.arctan(E_cross)) * 10 + 12:.6f} deg")

    # Phase from BZ integration (Chern-like)
    print(f"\n  BZ-averaged quantities:")

    # Berry curvature integrated over BZ
    # For a 1D line, this is just the total Berry phase
    # For a 2D slice, compute Berry curvature on a grid

    # 2D Berry curvature in the (e1, e2) plane at various heights along (111)
    print(f"\n  2D Berry curvature slices perpendicular to (111):")
    e1 = np.array([1, -1, 0]) / np.sqrt(2)
    e2 = np.array([1, 1, -2]) / np.sqrt(6)
    e3 = np.array([1, 1, 1]) / np.sqrt(3)  # along (111)

    for t_slice in [0.0, 0.25, 0.5, 0.75, 1.0]:
        k_center = np.array([t_slice * 0.25] * 3)
        # Compute Berry curvature at center
        dk = 0.001
        # F_12 = Im[log( <u(0,0)|u(dk,0)> <u(dk,0)|u(dk,dk)> <u(dk,dk)|u(0,dk)> <u(0,dk)|u(0,0)> )]
        corners = [
            k_center,
            k_center + dk * e1,
            k_center + dk * e1 + dk * e2,
            k_center + dk * e2,
        ]

        for band_label in ['w']:
            states_c = []
            for kc in corners:
                evals, evecs, c3d = c3_decompose(kc, bonds)
                target_c3 = omega3
                dists = [abs(c3d[b] - target_c3) for b in range(N_ATOMS)]
                best = np.argmin(dists)
                states_c.append(evecs[:, best].copy())

            # Plaquette product
            prod = 1.0 + 0j
            for i in range(4):
                j = (i + 1) % 4
                prod *= np.dot(np.conj(states_c[i]), states_c[j])
            F12 = -np.imag(np.log(prod)) / dk**2
            print(f"    t={t_slice:.2f}: F_12(w) = {F12:.6f}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  SRS alpha_21 = 162 deg FROM BAND CROSSING TOPOLOGY")
    print("  Gamma-P line crossing analysis")
    print("=" * 70)

    bonds = find_bonds()
    print(f"  {len(bonds)} bonds found.")

    # 1. Find crossing
    ts, E_w, E_w2, E_tlo, E_thi, crossings = find_crossing(bonds)

    # 2. Berry phases along Γ-P
    berry_results = berry_phase_GP(bonds)

    # 3. Wilson loops
    wilson_results = wilson_loops_scan(bonds, crossings)

    # 4. Phase analysis
    phase_analysis(berry_results, wilson_results, crossings, bonds)

    # 5. Crossing geometry
    crossing_geometry(bonds, crossings)

    # 6. Closed BZ loops
    berry_phase_closed_loops(bonds)

    # 7. Spectral analysis
    spectral_analysis(bonds, crossings)

    # ═══════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)

    target = 162.0
    print(f"\n  Target: alpha_21 = {target} deg = 9*pi/10")
    print(f"  Formula: 10*arctan(2-sqrt3) + pi/15 = 150 + 12 = 162")

    # Collect ALL phases computed
    best_match = None
    best_diff = 999.0

    for label, (pr, pd) in berry_results.items():
        diff = abs(abs(pd) - target)
        if diff < best_diff:
            best_diff = diff
            best_match = f"Berry({label}) = {pd:.6f} deg"

    for (tc, bl, r), (pr, pd) in wilson_results.items():
        diff = abs(abs(pd) - target)
        if diff < best_diff:
            best_diff = diff
            best_match = f"Wilson(t={tc:.3f},{bl},r={r:.3f}) = {pd:.6f} deg"

    print(f"\n  Closest match: {best_match}")
    print(f"  Distance from 162: {best_diff:.6f} deg")

    if best_diff < 0.5:
        print(f"\n  >>> POTENTIAL HIT: within 0.5 deg of 162! <<<")
    elif best_diff < 2.0:
        print(f"\n  >>> NEAR MISS: within 2 deg of 162 <<<")
    else:
        print(f"\n  >>> NO MATCH: band crossing topology does not directly give 162 <<<")
        print(f"  This is mechanism #12 killed.")

    # Plot
    plot_results(ts, E_w, E_w2, E_tlo, E_thi, crossings, berry_results)


def plot_results(ts, E_w, E_w2, E_tlo, E_thi, crossings, berry_results):
    """Summary plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(ts, E_w, 'r-', lw=2, label='omega')
    ax.plot(ts, E_w2, 'b-', lw=2, label='omega^2')
    ax.plot(ts, E_tlo, 'k--', lw=1, label='trivial (lo)')
    ax.plot(ts, E_thi, 'k:', lw=1, label='trivial (hi)')
    for name, t_c in crossings:
        ax.axvline(t_c, color='green', ls='--', alpha=0.7, label=f'crossing t={t_c:.4f}')
    ax.set_xlabel('t (Gamma=0, P=1)')
    ax.set_ylabel('Energy')
    ax.set_title('Bands along Gamma-P (C3-labeled)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Berry phase accumulation along the line
    n_pts = 500
    ts_bp = np.linspace(0.01, 0.99, n_pts)
    for band_label, color in [('w', 'red'), ('w2', 'blue')]:
        cum_phase = np.zeros(n_pts)
        states = []
        for i, t in enumerate(ts_bp):
            k = [t * 0.25] * 3
            evals, evecs, c3d = c3_decompose(k, bonds=find_bonds())
            target_c3 = omega3 if band_label == 'w' else omega3**2
            dists = [abs(c3d[b] - target_c3) for b in range(N_ATOMS)]
            best = np.argmin(dists)
            states.append(evecs[:, best].copy())
            if i > 0:
                overlap = np.dot(np.conj(states[-2]), states[-1])
                cum_phase[i] = cum_phase[i-1] - np.imag(np.log(overlap))
            else:
                cum_phase[i] = 0

        ax.plot(ts_bp, np.degrees(cum_phase), color=color, lw=2, label=band_label)

    ax.axhline(162, color='green', ls='--', alpha=0.5, label='162 deg')
    ax.axhline(-162, color='green', ls='--', alpha=0.5)
    ax.set_xlabel('t (Gamma=0, P=1)')
    ax.set_ylabel('Accumulated Berry phase (deg)')
    ax.set_title('Berry phase along Gamma-P')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'srs_alpha21_crossing.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {path}")


if __name__ == '__main__':
    main()
