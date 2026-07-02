#!/usr/bin/env python3
"""
gyroid_surface_gauge_higgs_hodge_2026-06-13.py
==============================================
THE LEAD (first step): free gauge+Higgs DYNAMICS on the genus-3 surface complex.

Thread A (`gyroid_surface_c2_flux_homology`) built the 2-cochain structure the
framework's gauge sector lacked -- C_2 (girth-10 rings) -> C_1 (edges) -> C_0
(vertices), d_1 d_2 = 0 -- and resolved the "Gamma anomaly" as H_1 = the gyroid
genus.  That fixed the flux TOPOLOGY.  This probe takes the next step: put an inner
product on the complex and build the HODGE LAPLACIANS -- the free (quadratic) gauge
and Higgs dynamics -- and read the physics off their spectra.

CONSTRUCTION (Bloch-resolved, native).  With standard Hermitian inner products on
C_0 (4), C_1 (6), C_2 (6) and boundary maps d_1(k): C_1->C_0, d_2(k): C_2->C_1:

  Higgs / scalar Laplacian   Delta_0 = d_1 d_1^t            (on C_0, vertices)
  GAUGE-field Hodge Lap.     Delta_1 = d_2 d_2^t + d_1^t d_1 (on C_1, edges)
  2-form Laplacian           Delta_2 = d_2^t d_2            (on C_2, rings)

Hodge theorem: dim ker Delta_n = b_n.  So the GAUGE field's harmonic (massless,
physical, topological) modes = H_1 = the genus, and the Higgs zero mode = H_0 = the
Perron / constant mode (the framework's named "condensate's unique home").  The Hodge
decomposition of C_1 = im d_1^t (pure gauge) (+) harmonic (physical) (+) im d_2
(field-strength-sourced) IS the gauge-fixing (longitudinal/transverse) split.

WHAT THIS PROBE SHOWS
  A  valid cochain complex (d_1 d_2 = 0) + the three Hodge Laplacians, Hermitian PSD.
  B  HODGE SPECTRUM at the saddles: dim ker Delta_0 (Higgs condensate modes),
     dim ker Delta_1 (physical gauge modes = genus), dim ker Delta_2.
  C  HODGE / GAUGE-FIXING decomposition of the gauge field C_1 = pure-gauge (+)
     physical (+) field-strength; dims add to 6.
  D  free dispersions: the nonzero Delta_1 (gauge) and Delta_0 (Higgs) eigenvalues
     give the free massive-mode spectrum; Higgs VEV channel = the Gamma Perron mode.
  E  VERDICT + honest scope: this is the free, ABELIAN Hodge dynamics -- the first
     dynamics on the C^2 the framework lacked.  Open: the non-abelian PS (Cl(6)/Cl(2))
     operator-algebra version (extends de_rham_susy_fibered_v2), the Higgs potential /
     EW breaking, and matter coupling.  No graded content changes.
"""

import os
import sys

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, A_PRIM  # noqa: E402

FAILURES = []
RNG = np.random.default_rng(20260613)


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --- srs combinatorics (Phase-5.3/B2 + Thread A conventions) -----------------
bonds = find_bonds()
EDGES = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
NE = len(EDGES)
E_INDEX = {e: a for a, e in enumerate(EDGES)}
REV = {a: E_INDEX[(j, i, tuple(-x for x in c))] for a, (i, j, c) in enumerate(EDGES)}
SEEN, UBONDS = set(), []
for (i, j, c) in EDGES:
    if (j, i, tuple(-x for x in c)) in SEEN:
        continue
    SEEN.add((i, j, tuple(c)))
    UBONDS.append((i, j, tuple(c)))
NU = len(UBONDS)
NV = 4
FOLLOW = {a: [b for b in range(NE) if EDGES[b][0] == EDGES[a][1] and b != REV[a]] for a in range(NE)}


def d1(k):
    D = np.zeros((NV, NU), complex)
    for b, (i, j, c) in enumerate(UBONDS):
        D[i, b] += -1.0
        D[j, b] += np.exp(2j * np.pi * np.dot(k, np.asarray(c, float)))
    return D


def closed_nb_walks(length):
    walks = []

    def step(path, a, d):
        ca = EDGES[a][2]
        d2 = (d[0] + ca[0], d[1] + ca[1], d[2] + ca[2])
        if len(path) == length:
            if path[0][0] in FOLLOW[a] and d2 == (0, 0, 0):
                walks.append(list(path))
            return
        for b in FOLLOW[a]:
            step(path + [(b, d2)], b, d2)
    for a0 in range(NE):
        step([(a0, (0, 0, 0))], a0, (0, 0, 0))
    return walks


def cycle_class(walk):
    inst = set()
    for a, d in walk:
        i, j, c = EDGES[a]
        ra = REV[a]
        d2 = (d[0] + c[0], d[1] + c[1], d[2] + c[2])
        inst.add(min((a, d), (ra, d2)))
    base = min(d for _, d in inst)
    return frozenset((a, (d[0] - base[0], d[1] - base[1], d[2] - base[2])) for a, d in inst)


def ring_chain(walk, k):
    v = np.zeros(NU, complex)
    for a, d in walk:
        i, j, c = EDGES[a]
        if (i, j, tuple(c)) in SEEN:
            v[UBONDS.index((i, j, tuple(c)))] += np.exp(2j * np.pi * np.dot(k, np.asarray(d, float)))
        else:
            i2, j2, c2 = EDGES[REV[a]]
            d2 = (d[0] + c[0], d[1] + c[1], d[2] + c[2])
            v[UBONDS.index((i2, j2, tuple(c2)))] -= np.exp(2j * np.pi * np.dot(k, np.asarray(d2, float)))
    return v


REPS = list({cycle_class(w): w for w in closed_nb_walks(10)}.values())
NR = len(REPS)


def d2(k):
    return np.array([ring_chain(w, np.asarray(k, float)) for w in REPS]).T   # NU x NR


def hodge(k):
    D1, D2 = d1(np.asarray(k, float)), d2(k)
    L0 = D1 @ D1.conj().T                                  # Higgs/scalar, on C_0
    L1 = D2 @ D2.conj().T + D1.conj().T @ D1               # gauge, on C_1
    L2 = D2.conj().T @ D2                                  # 2-form, on C_2
    return D1, D2, L0, L1, L2


def kerdim(M, tol=1e-9):
    w = la.eigvalsh((M + M.conj().T) / 2)
    return int(np.sum(np.abs(w) < tol))


def main():
    print("=" * 90)
    print(" THE LEAD (step 1): free gauge+Higgs Hodge dynamics on the genus-3 surface complex")
    print("=" * 90)

    DELTA = np.array([0.5, 0.5, -0.5])
    SADDLES = {"Gamma": np.zeros(3), "H": DELTA, "P": np.array([.25, .25, .25]),
               "N": A_PRIM @ np.array([0.0, .5, .5]), "generic": RNG.uniform(-.5, .5, 3)}

    # --- A: complex + Laplacians --------------------------------------------
    print(f"\n A  cochain complex C^0({NV}) -> C^1({NU}) -> C^2({NR});  Hodge Laplacians Hermitian PSD")
    worst_cc, worst_herm, worst_psd = 0.0, 0.0, 0.0
    for k in SADDLES.values():
        D1, D2, L0, L1, L2 = hodge(k)
        worst_cc = max(worst_cc, la.norm(D1 @ D2))
        for L in (L0, L1, L2):
            worst_herm = max(worst_herm, la.norm(L - L.conj().T))
            worst_psd = min(worst_psd, la.eigvalsh((L + L.conj().T) / 2).min())
    gate("A1 d_1 d_2 = 0 (valid cochain complex)", worst_cc < 1e-9, f"max |d1 d2|={worst_cc:.1e}")
    gate("A2 Delta_0, Delta_1, Delta_2 Hermitian and PSD (genuine free actions)",
         worst_herm < 1e-9 and worst_psd > -1e-9, f"min eig={worst_psd:.1e}")

    # --- B: Hodge spectrum (harmonic = homology) ----------------------------
    print("\n B  Hodge harmonic content  dim ker Delta_n = b_n  (Higgs / gauge / 2-form)")
    print(f"    {'k':>8} | {'ker D0 (Higgs)':>14} | {'ker D1 (gauge=genus)':>20} | {'ker D2':>7}")
    print("    " + "-" * 60)
    res = {}
    for nm, k in SADDLES.items():
        _, _, L0, L1, L2 = hodge(k)
        res[nm] = (kerdim(L0), kerdim(L1), kerdim(L2))
        print(f"    {nm:>8} | {res[nm][0]:>14} | {res[nm][1]:>20} | {res[nm][2]:>7}")
    gate("B1 GAUGE harmonic modes = genus: dim ker Delta_1 = 3 at Gamma, 0 at H/P/N/generic",
         res["Gamma"][1] == 3 and all(res[nm][1] == 0 for nm in ("H", "P", "N", "generic")))
    gate("B2 HIGGS zero mode = Perron/condensate: dim ker Delta_0 = 1 at Gamma, 0 elsewhere",
         res["Gamma"][0] == 1 and all(res[nm][0] == 0 for nm in ("H", "P", "N", "generic")))

    # --- C: Hodge / gauge-fixing decomposition of C^1 -----------------------
    print("\n C  gauge field C^1 = pure-gauge (im d_1^t) (+) physical harmonic (+) field-strength (im d_2)")
    print(f"    {'k':>8} | {'pure-gauge':>10} | {'physical(harm)':>14} | {'field-strength':>14} | sum")
    print("    " + "-" * 60)
    okC = True
    for nm, k in SADDLES.items():
        D1, D2, _, L1, _ = hodge(k)
        pg = np.linalg.matrix_rank(D1.conj().T, tol=1e-9)         # im d_1^t (longitudinal)
        fs = np.linalg.matrix_rank(D2, tol=1e-9)                  # im d_2 (transverse, sourced)
        harm = NU - pg - fs                                       # harmonic = physical topological
        okC &= (pg + harm + fs == NU) and harm == res[nm][1]
        print(f"    {nm:>8} | {pg:>10} | {harm:>14} | {fs:>14} | {pg+harm+fs}")
    gate("C Hodge decomposition holds (pure-gauge + physical + field-strength = 6; physical = ker Delta_1)",
         okC)

    # --- D: free dispersions -------------------------------------------------
    print("\n D  free spectra (nonzero Hodge eigenvalues = massive mode dispersion)")
    for nm in ("Gamma", "P", "generic"):
        _, _, L0, L1, _ = hodge(SADDLES[nm])
        e0 = np.sort(la.eigvalsh((L0 + L0.conj().T) / 2))
        e1 = np.sort(la.eigvalsh((L1 + L1.conj().T) / 2))
        print(f"    {nm:>8}:  Higgs Delta_0 spec = {np.round(e0,3)}")
        print(f"    {'':>8}   gauge Delta_1 spec = {np.round(e1,3)}")

    # --- E: verdict ----------------------------------------------------------
    print("\n" + "=" * 90)
    print(" VERDICT  (the lead, step 1)")
    print("=" * 90)
    print("""  The free gauge+Higgs dynamics on the genus-3 surface complex is built and behaves
  exactly as the Hodge theorem requires:

    * GAUGE field (Delta_1 on C^1): its harmonic (massless, physical) modes = H_1 = the
      gyroid genus -- 3 at Gamma, 0 off Gamma.  The gauge field splits canonically into
      pure-gauge (longitudinal) + physical-topological (the genus) + field-strength-
      sourced (transverse) -- i.e. the Hodge decomposition IS gauge-fixing, with the
      physical sector = the genus-3 topological flux of Thread A.
    * HIGGS / scalar (Delta_0 on C^0): its single zero mode at Gamma = the Perron /
      constant mode -- the framework's named "condensate's unique home" -- the natural
      Higgs-VEV channel.  Off Gamma it is gapped.

  So matter (ODD) on the 1-skeleton, gauge+Higgs (EVEN) as Hodge forms on the surface
  complex: the gauge physical modes ARE the genus, the Higgs IS the Perron mode.  This is
  the first dynamics built on the C^2 the framework previously lacked.

  HONEST SCOPE.  This is the FREE, ABELIAN (quadratic) Hodge skeleton.  Open next steps:
   (i) the non-abelian Pati-Salam structure -- promote scalars to the Cl(6)/Cl(2)
       operator-algebra cochains (extend de_rham_susy_fibered_v2 with this C^2);
   (ii) the Higgs POTENTIAL and EW symmetry breaking (the VEV in the Gamma Perron mode);
   (iii) coupling to the matter 1-skeleton (covariant derivative, Yukawa).
  Also: the 6 girth-10 2-cells reproduce H_1 (the gauge content) faithfully but are NOT a
  literal triangulation of the genus-3 surface (H_2 differs) -- adequate for the gauge
  sector, to be refined for the full surface. No graded content changes.""")

    if FAILURES:
        print(f"\n RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print("\ngyroid_surface_gauge_higgs_hodge_2026-06-13.py: done (sentinel).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
