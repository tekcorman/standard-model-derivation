#!/usr/bin/env python3
"""
Survey Hashimoto walker spectrum at all BCC high-symmetry points (Γ, H, P, N).

CONTEXT
=======
The framework currently uses h_walker eigenvalue at the P-point — h = (√3 + i√5)/2,
|h|² = k* - 1 = 2 (Ramanujan-saturating) — for ALL sector predictions (lepton + quark).
Per `docs/theorems/theorem_bloch_lift_mu.md` §"Scope of upgrade":

    "The identification of k_P as the physically relevant Bloch momentum for the SM
    mass spectrum remains under A5(a) and is NOT closed by this theorem."

I.e., the choice of P-point is currently ADOPTED. R-14 closure path (b) per residue
register asks whether DIFFERENT Bloch points distinguish quark vs lepton sectors —
i.e., is P uniquely picked, or could quarks live at one Ramanujan-saturating point
and leptons at another?

QUESTION
========
Is P the ONLY high-symmetry point whose Bloch-Hashimoto operator has ALL eigenvalues
Ramanujan-saturated (|h|² = k* - 1 = 2)? Equivalently: is |E_n(k)| ≤ 2√(k*-1) = 2√2
for every scalar adjacency eigenvalue E_n at every high-symmetry point k?

The Ihara-Bass quadratic h² - E·h + (k*-1) = 0 has complex (Ramanujan-saturating) roots
iff E² ≤ 4(k*-1) = 8 ⇔ |E| ≤ 2√2 ≈ 2.828. Real (non-Ramanujan) roots iff |E| > 2√2.

WHAT THIS PROBE COMPUTES
========================
For each k ∈ {Γ, H, P, N}:
  - Scalar adjacency eigenvalues E_n(k) (4 per point, since H(k) is 4×4 for srs)
  - Hashimoto eigenvalues h via Ihara-Bass per E_n
  - Whether each h satisfies Ramanujan saturation
  - Total count of Ramanujan-saturated vs non-saturated eigenvalues per point

OUTCOME FOR R-14 PATH (b)
=========================
- If P is the ONLY uniformly-Ramanujan point: structural argument that no Bloch-point
  alternative exists → R-14 (b) closes as NEGATIVE (no sector splitting via this route).
- If multiple points are uniformly-Ramanujan: candidates exist; need additional structure
  to assign sectors to specific Bloch points (research-level).
"""

from __future__ import annotations

import math
import numpy as np
from numpy import linalg as la

K_STAR = 3
K_MINUS_ONE = K_STAR - 1  # = 2
RAMANUJAN_BOUND = 2 * math.sqrt(K_MINUS_ONE)  # = 2√2 ≈ 2.828

# Same bond list as proofs/foundations/lorentz_sig_g_sub_lichnerowicz_closure.py
CELL_EDGES = [
    (0, 1, (1, 1, 1)),
    (0, 2, (1, 1, 1)),
    (0, 3, (1, 1, 1)),
    (1, 2, (-1, 0, 0)),
    (1, 3, (0, 1, 0)),
    (2, 3, (0, 0, -1)),
]

DIRECTED_BONDS = []
for s, t, c in CELL_EDGES:
    DIRECTED_BONDS.append((s, t, c))
    DIRECTED_BONDS.append((t, s, tuple(-x for x in c)))

N_ATOMS = 4
N_EDGES = 6  # undirected
N_DIRECTED = 12  # = 2|E| = dim of B(k)


def scalar_bloch_H(k1: float, k2: float, k3: float) -> np.ndarray:
    """4×4 scalar adjacency A(k) at fractional k."""
    H = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
    for s, t, c in DIRECTED_BONDS:
        H[t, s] += np.exp(2j * np.pi * (c[0] * k1 + c[1] * k2 + c[2] * k3))
    return H


def hashimoto_eigenvalues_from_scalar(scalar_eigs: np.ndarray) -> list:
    """For each scalar eigenvalue E, return both Hashimoto roots h± from
    h² - E·h + (k*-1) = 0. Returns list of (E, h_+, h_-, |h|²_+, |h|²_-)."""
    out = []
    for E in scalar_eigs:
        # Quadratic h² - E·h + (k*-1) = 0
        disc = E**2 - 4 * K_MINUS_ONE
        sqrt_disc = np.lib.scimath.sqrt(disc)  # handles disc < 0 → imag
        h_plus = (E + sqrt_disc) / 2
        h_minus = (E - sqrt_disc) / 2
        out.append({
            'E': E,
            'h_plus': h_plus,
            'h_minus': h_minus,
            'abs2_plus': abs(h_plus)**2,
            'abs2_minus': abs(h_minus)**2,
            'ramanujan': abs(disc) < 1e-10 or disc.real < 0,  # complex root or boundary
        })
    return out


def survey_point(name: str, k_frac: tuple) -> dict:
    """Survey a single Bloch high-symmetry point."""
    A = scalar_bloch_H(*k_frac)
    # A should be Hermitian for scalar Bloch
    herm_err = la.norm(A - A.conj().T)
    eigs = la.eigvalsh(A)  # use eigvalsh for guaranteed-real output
    eigs_real = sorted(eigs.tolist())  # sorted for readability

    # Compute Hashimoto pairs per scalar eigenvalue
    hashimoto = hashimoto_eigenvalues_from_scalar(np.array(eigs_real))

    # Tally
    n_ram = sum(2 for h in hashimoto if h['ramanujan'])
    n_nonram = sum(2 for h in hashimoto if not h['ramanujan'])
    uniform_ramanujan = n_nonram == 0

    return {
        'name': name,
        'k_frac': k_frac,
        'hermiticity_err': herm_err,
        'scalar_eigs': eigs_real,
        'hashimoto': hashimoto,
        'n_ramanujan': n_ram,
        'n_non_ramanujan': n_nonram,
        'uniform_ramanujan': uniform_ramanujan,
    }


# ============================================================================
# 1. Survey all BCC high-symmetry points
# ============================================================================
HIGH_SYM_POINTS = [
    ('Γ', (0.0,    0.0,   0.0  )),
    ('H', (-0.5,   0.5,   0.5  )),
    ('P', (0.25,   0.25,  0.25 )),
    ('N', (0.0,    0.0,   0.5  )),
]

print("=" * 78)
print("BCC high-symmetry-point Hashimoto-walker spectrum survey")
print("=" * 78)
print(f"  k* = {K_STAR}, k* - 1 = {K_MINUS_ONE}, Ramanujan |h| bound = √(k*-1) = √2 ≈ {math.sqrt(K_MINUS_ONE):.4f}")
print(f"  Ihara-Bass quadratic: h² - E·h + 2 = 0, complex roots iff |E| ≤ 2√2 ≈ {RAMANUJAN_BOUND:.4f}")
print()

results = {}
for name, k_frac in HIGH_SYM_POINTS:
    r = survey_point(name, k_frac)
    results[name] = r

    print(f"--- {name} = {k_frac} ---")
    print(f"  Hermiticity check: ||A - A†|| = {r['hermiticity_err']:.2e}")
    print(f"  Scalar eigenvalues E_n(k): {[f'{e:+.4f}' for e in r['scalar_eigs']]}")
    print()
    print(f"  Hashimoto eigenvalues per scalar root (h± from h² - Eh + 2 = 0):")
    print(f"    {'E_n':>9}  {'h_+':>22}  {'|h_+|²':>9}  {'h_-':>22}  {'|h_-|²':>9}  Ram?")
    for h in r['hashimoto']:
        E = h['E']
        hp = h['h_plus']
        hm = h['h_minus']
        ram = '✓' if h['ramanujan'] else '✗'
        if abs(hp.imag) < 1e-10:
            hp_str = f"{hp.real:+.4f}"
        else:
            hp_str = f"{hp.real:+.4f}{'+' if hp.imag>=0 else '-'}{abs(hp.imag):.4f}i"
        if abs(hm.imag) < 1e-10:
            hm_str = f"{hm.real:+.4f}"
        else:
            hm_str = f"{hm.real:+.4f}{'+' if hm.imag>=0 else '-'}{abs(hm.imag):.4f}i"
        print(f"    {E:>+9.4f}  {hp_str:>22}  {h['abs2_plus']:>9.4f}  {hm_str:>22}  {h['abs2_minus']:>9.4f}  {ram}")

    print()
    print(f"  Tally: {r['n_ramanujan']} Ramanujan / {r['n_non_ramanujan']} non-Ramanujan (out of 8)")
    print(f"  Uniformly Ramanujan? {'YES' if r['uniform_ramanujan'] else 'NO'}")
    print()

# ============================================================================
# 2. Summary: which points are uniformly Ramanujan
# ============================================================================
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print()
print(f"  {'point':>6}  {'k':>22}  {'scalar eigvals':>30}  {'uniform Ram?':>14}")
print(f"  {'-'*6}  {'-'*22}  {'-'*30}  {'-'*14}")
for name, k_frac in HIGH_SYM_POINTS:
    r = results[name]
    eig_str = ', '.join(f'{e:+.3f}' for e in r['scalar_eigs'])
    uniform = 'YES' if r['uniform_ramanujan'] else 'NO'
    print(f"  {name:>6}  {str(k_frac):>22}  {eig_str:>30}  {uniform:>14}")

uniform_pts = [n for n, r in results.items() if r['uniform_ramanujan']]
print()
print(f"  Uniformly-Ramanujan high-symmetry points: {uniform_pts}")
print()

# ============================================================================
# 3. Structural diagnosis
# ============================================================================
print("=" * 78)
print("STRUCTURAL DIAGNOSIS for R-14 path (b)")
print("=" * 78)
print()
if len(uniform_pts) == 1:
    print(f"""  Only ONE high-symmetry point ({uniform_pts[0]}) has all Hashimoto eigenvalues
  Ramanujan-saturated. The framework's choice of {uniform_pts[0]}-point as 'the' relevant
  Bloch momentum for SM mass spectrum is then STRUCTURALLY UNIQUE within the
  high-symmetry set — no alternative point exists.

  ⇒ R-14 path (b) "Bloch-point sector identification" CLOSES AS NEGATIVE.
    Quark/lepton differentiation cannot come from sector-specific high-symmetry
    Bloch-point assignment, because no alternative Ramanujan-saturating point
    is available. R-14 closure must come from path (a) or (c) instead.
""")
elif len(uniform_pts) > 1:
    print(f"""  MULTIPLE high-symmetry points ({uniform_pts}) have all Hashimoto eigenvalues
  Ramanujan-saturated. This opens a candidate mechanism for R-14 path (b):
  quark and lepton sectors could live at DIFFERENT uniformly-Ramanujan points,
  giving sector-specific h_walker eigenvalues.

  ⇒ R-14 path (b) is OPEN with candidates. Next: identify which alternative
    point pairs with which sector via additional structural argument.

  Specific h_walker eigenvalues at each:""")
    for name in uniform_pts:
        r = results[name]
        # Print one Hashimoto root per scalar eigenvalue (the +imag one for chirality)
        h_list = []
        for h in r['hashimoto']:
            hp = h['h_plus']
            if abs(hp.imag) < 1e-10:
                h_list.append(f'{hp.real:+.4f}')
            else:
                h_list.append(f'{hp.real:+.4f}+{abs(hp.imag):.4f}i')
        h_str = ', '.join(h_list)
        print(f"    {name}: h_+ values = {h_str}")
else:
    print("""  NO high-symmetry points are uniformly Ramanujan-saturated. This is unexpected
  given the framework's prior identification of P as Ramanujan-saturating; check
  Ihara-Bass implementation.
""")

print("=" * 78)
print("END")
print("=" * 78)
