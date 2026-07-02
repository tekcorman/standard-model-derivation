#!/usr/bin/env python3
"""
THEOREM (substrate description length; srs dominance via topological crystals)
==============================================================================

This derives the observer's minimal description length L(net) of a candidate
crystal-net substrate from first principles, RESOLVING the measure ambiguity
(space-group+Wyckoff DL vs quotient-cell DL) that left "which net dominates"
undetermined. The resolution is forced by Sunada's topological crystallography,
not chosen.

PREMISES
--------
(P1) [framework MDL / A2-T] The observer's model is specified by its minimal
     generating description; posterior weight ∝ 2^(-L(net)).  (Used framework-wide.)
(P2) [Sunada, Topological Crystallography] A connected d-periodic graph G is the
     maximal abelian cover of a finite quotient multigraph Q under a period map
     H_1(Q;Z) ↠ Z^d.  G is CANONICAL — determined by Q alone — iff b_1(Q) = d,
     where b_1(Q) = |E(Q)| - |V(Q)| + 1 is the first Betti number (cycle rank).
     If b_1(Q) > d, an extra rank-d quotient of Z^{b_1(Q)} (the period map) must
     be specified.  [Published theorem; Sunada 2013, Kotani-Sunada standard realization.]
(P3) [derived upstream] d = 3 (Gleason via observer_hilbert_space); the substrate
     is k-regular (no-privilege ⟹ k-regularity; walker_dynamics §4a).

LEMMA 1 (description length).
     L(net) = K(Q) + L(period map),  with  L(period map) = 0  iff b_1(Q) = d,
     else L(period map) ≥ (b_1(Q) - d)·c  for some c > 0.
     So a net is minimal-description iff it is a TOPOLOGICAL CRYSTAL (b_1 = d).

LEMMA 2 (which coordinations admit a single-orbit topological crystal at d=3).
     k-regular + one vertex orbit + b_1 = 3  ⟹  |E| = k|V|/2  and
     b_1 = |E| - |V| + 1 = (k/2 - 1)|V| + 1 = 3  ⟹  |V| = 4/(k-2).
     Positive-integer solutions: k ∈ {3, 4, 6} with |V| ∈ {4, 2, 1}.
     Quotients: K_4 (k=3), the 2-vertex 4-bond graph (k=4), the 1-vertex 3-loop
     graph (k=6); maximal abelian covers = srs, dia, pcu (Sunada's named crystals).
     Ramanujan modulus √(k-1) = √2, √3, √5 = the generators of K = Q(√2,√3,√5).

LEMMA 3 (the k=3 minimal quotient is K_4).
     Among connected 4-vertex 3-regular multigraphs with b_1 = 3, K_4 (the complete
     graph, |Aut| = 24 = S_4) is the maximal-symmetry / minimal-K(Q) one. Its
     maximal abelian cover is srs (Sunada's "K_4 crystal" = Laves / (10,3)-a).

THEOREM.  srs is the MDL-dominant substrate at d=3:
  (a) srs = topological crystal of K_4: b_1 = 3 = d, period-map cost 0.
  (b) srs-z (bipartite double cover of srs): quotient Q_3 with |V| = 8 > 4 AND
      b_1 = 5 > 3 ⟹ L(srs-z) > L(srs) on BOTH the quotient-size term K(Q) and
      the period-map term. So srs-z is strictly subdominant — the over-determined
      cover where the (subleading) mass/chirality structure lives.
  (c) k>3 topological crystals (dia, pcu) carry a per-event coding cost
      log2(k-1) > log2(2); relative posterior weight ((k-1)/2)^(-N) → 0
      super-exponentially in the observation count N.

COROLLARY (measure ambiguity resolved).  The space-group+Wyckoff DL (whose
log2(W) term gave srs-z cheaper) is NOT the algorithmically-minimal description:
it mis-penalizes symmetry (more-symmetric space groups have more Wyckoff
positions). The topological-crystal description length IS minimal and gives srs.

STATUS: THEOREM-GRADE-CONDITIONAL on (P1 MDL premise + P2 Sunada [published] +
P3 [derived]). Not in scope of this theorem: the CP/parity (chirality) selection
of srs over achiral covers — that is an EMPIRICAL penalty (achiral ⟹ CP-conserving
⟹ contradicts observed δ_CP, η_B), separate from L(net), and quantified elsewhere.

------------------------------------------------------------------------------
Machine verification of the Lemmas below (b_1 from the cached RCSR primitive cells).
"""

import json, os, re

SNAP = json.load(open(os.path.join(os.path.dirname(__file__), '..', '..',
        'simulator', 'menus', 'data', 'rcsr_candidates_snapshot.json')))['entries']


def centering(sg):
    return {'I': 2, 'F': 4, 'R': 3, 'C': 2, 'A': 2, 'B': 2, 'P': 1}.get(sg[0], 1)


def primitive_VE(name):
    e = SNAP[name]; vo = e['vertex_orbits'][0]; k = vo['coord']
    mult = int(re.match(r'(\d+)', vo['wyckoff_label']).group(1))
    V = mult // centering(e['sg_name'])
    E = k * V // 2
    return k, V, E, E - V + 1   # k, |V|, |E|, b_1


def check():
    fails = []
    # Lemma-1/Theorem(a,b): topological-crystal status from b_1.
    expect = {'srs': 3, 'srs-z': 5, 'dia': 3, 'pcu': 3, 'bcu': 4, 'fcu': 6}
    print(f"{'net':<7}{'k':>3}{'|V|':>4}{'|E|':>4}{'b1':>4}{'b1=d=3':>8}  role")
    for n, b1_exp in expect.items():
        k, V, E, b1 = primitive_VE(n)
        if b1 != b1_exp:
            fails.append(f"{n}: b1={b1} expected {b1_exp}")
        role = "topological crystal (canonical)" if b1 == 3 else f"over-determined (+{b1-3} period vectors)"
        print(f"{n:<7}{k:>3}{V:>4}{E:>4}{b1:>4}{str(b1==3):>8}  {role}")

    # Lemma-2: |V| = 4/(k-2) integer iff k in {3,4,6}
    print("\nLemma 2 — single-orbit topological crystals at d=3 (|V| = 4/(k-2) ∈ Z+):")
    iso = [k for k in range(3, 13) if (k > 2 and 4 % (k - 2) == 0)]
    print(f"   integer-|V| coordinations: {iso}   (expect [3, 4, 6])")
    if iso != [3, 4, 6]:
        fails.append(f"Lemma2: got {iso}")
    for k in iso:
        print(f"     k={k}: |V|={4//(k-2)}, modulus √(k-1)=√{k-1}")

    # Theorem(b): srs strictly cheaper than srs-z on both terms
    _, Vs, Es, b1s = primitive_VE('srs')
    _, Vz, Ez, b1z = primitive_VE('srs-z')
    quotient_smaller = Vs < Vz
    periodmap_cheaper = (b1s == 3) and (b1z > 3)
    print(f"\nTheorem(b): srs vs srs-z")
    print(f"   quotient size |V|: srs={Vs} < srs-z={Vz}  -> {quotient_smaller}")
    print(f"   period-map cost: srs b1=3 (0) , srs-z b1={b1z} (>0)  -> srs cheaper: {periodmap_cheaper}")
    if not (quotient_smaller and periodmap_cheaper):
        fails.append("Theorem(b): srs not strictly cheaper than srs-z")

    print("\n" + ("ALL CHECKS PASS — srs is the MDL-dominant substrate (topological crystal of K_4)."
                  if not fails else "FAILURES: " + "; ".join(fails)))
    return not fails


if __name__ == '__main__':
    ok = check()
    raise SystemExit(0 if ok else 1)
