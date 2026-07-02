#!/usr/bin/env python3
"""
gyroid_mdl_vs_symmetry_optimal_2026-06-13.py
============================================
Thread B: is the MDL-optimal net provably the symmetry/geometry optimum?  i.e. does
"MDL-minimum => minimal-surface (L^3/V) minimum" hold as a clean implication?

ANSWER: NO -- but with a precise positive core.  Decomposing the framework's own MDL
functional (dl_comparison) across the candidate nets shows MDL = [a transitivity-floor
part, shared with the geometric L^3/V handle via edge+vertex transitivity] PLUS
[naming + chirality terms that are description-costs, absent from any geometric
functional].  So MDL and L^3/V are NOT the same functional; they converge on srs only
through the shared transitivity, and srs actually PAYS MDL bits (chirality, a richer
Wyckoff) that pure symmetry/geometry would not charge.

This sharpens exploit #3 / Item 1 a second time:
  exploit #3 : "three independent handles"          (overstated)
  Item 1     : "co-extremised by maximal symmetry"  (closer)
  Thread B   : "co-extremised by edge+vertex TRANSITIVITY specifically; MDL is the
               transitivity-floor PLUS non-geometric naming+chirality terms, so
               'MDL => minimal-surface' is NOT a functional implication."

WHAT THIS PROBE SHOWS (native, from dl_comparison's real bit-breakdowns)
  A  the MDL terms split into TRANSITIVITY-FLOOR {n_orbits, coordinates, edges},
     NAMING {space_group, wyckoff}, and CHIRALITY.
  B  srs uniquely hits the transitivity floor (orbits=1, coords=0, edges=0) -- the
     SAME edge+vertex transitivity that forces the 120-deg vertex and L^3/V=27/sqrt2
     (Item 1).  This is the genuine common core.
  C  srs PAYS the chirality bit (1.0 vs 0.0) and a HIGHER Wyckoff naming cost than ths
     (3.17 vs 3.00); space_group is constant (7.85, non-discriminating).  These terms
     are in MDL but NOT in L^3/V, so MDL != a geometric functional, and srs is NOT the
     description-cheapest in them.
  D  VERDICT: "MDL => minimal-surface" is FALSE as a functional implication.  The
     common core is edge+vertex transitivity (drives both the MDL floor and L^3/V);
     srs's full selection additionally needs chirality (R-12, physics -- MDL would
     happily drop the bit) and the Sunada strong-isotropy theorem (a symmetry result,
     not MDL) to exclude any centrosymmetric edge-transitive competitor.  No graded
     content changes.
"""

import os
import sys
from math import sqrt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.foundations.dl_comparison import dl_srs, dl_ths, dl_eta, dl_utj  # noqa: E402

FAILURES = []
FLOOR = ("n_orbits", "coordinates", "edges")
NAMING = ("space_group", "wyckoff")


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main():
    print("=" * 90)
    print(" THREAD B: does MDL-minimum imply the geometric (L^3/V) minimum?  Decompose and decide.")
    print("=" * 90)

    nets = {"srs": dl_srs()[1], "ths": dl_ths()[1], "eta": dl_eta()[1], "utj": dl_utj()[1]}

    # --- A: term split -------------------------------------------------------
    print("\n A  MDL term decomposition (bits)")
    cols = ["space_group", "wyckoff", "n_orbits", "coordinates", "edges", "chirality"]
    print(f"    {'net':>5} | " + " ".join(f"{c[:9]:>9}" for c in cols) + " |   total")
    print("    " + "-" * 92)
    for nm, b in nets.items():
        tot = sum(b.values())
        print(f"    {nm:>5} | " + " ".join(f"{b[c]:>9.2f}" for c in cols) + f" | {tot:>7.2f}")

    # --- B: transitivity floor (the common core with L^3/V) ------------------
    print("\n B  transitivity-floor terms {n_orbits, coordinates, edges} -- shared cause with L^3/V")
    floor = {nm: sum(b[k] for k in FLOOR) for nm, b in nets.items()}
    print("    floor(orbits+coords+edges):  " + ",  ".join(f"{nm}={floor[nm]:.0f}" for nm in nets))
    gate("B srs uniquely hits the transitivity floor (orbits=1, coords=0, edges=0)",
         nets["srs"]["n_orbits"] == 1 and nets["srs"]["coordinates"] == 0 and nets["srs"]["edges"] == 0
         and floor["srs"] == min(floor.values()) and list(floor.values()).count(floor["srs"]) == 1,
         f"srs floor={floor['srs']:.0f} (= edge+vertex transitivity = the 120-deg vertex = L^3/V min)")

    # --- C: MDL's non-geometric terms (srs PAYS) -----------------------------
    print("\n C  non-geometric MDL terms (absent from L^3/V): srs PAYS them")
    sg_const = len({round(b["space_group"], 6) for b in nets.values()}) == 1
    gate("C1 space_group bits constant across nets (7.85) -- non-discriminating naming",
         sg_const, "all pay log2(230)")
    gate("C2 srs PAYS chirality = 1.0 while centrosymmetric ths pays 0.0 (chiral = lower point symmetry)",
         nets["srs"]["chirality"] == 1.0 and nets["ths"]["chirality"] == 0.0)
    gate("C3 srs PAYS a HIGHER Wyckoff naming cost than ths (richer I4_132: W=9)",
         nets["srs"]["wyckoff"] > nets["ths"]["wyckoff"],
         f"srs={nets['srs']['wyckoff']:.2f} > ths={nets['ths']['wyckoff']:.2f}")
    win = sum(nets["ths"].values()) - sum(nets["srs"].values())
    d_edges = nets["ths"]["edges"] - nets["srs"]["edges"]
    d_chir = nets["ths"]["chirality"] - nets["srs"]["chirality"]
    d_wyck = nets["ths"]["wyckoff"] - nets["srs"]["wyckoff"]
    print(f"    srs beats ths by {win:+.2f} bits = edges {d_edges:+.0f} (save)  chirality {d_chir:+.2f} (srs pays)  "
          f"wyckoff {d_wyck:+.2f} (srs pays)")
    gate("C4 srs's win is the edge-transitivity saving NET of its chirality+Wyckoff costs",
         abs(win - (d_edges + d_chir + d_wyck)) < 1e-9 and d_edges > 0 and d_chir < 0 and d_wyck < 0)

    # --- D: verdict ----------------------------------------------------------
    print("\n" + "=" * 90)
    print(" VERDICT  (Thread B)")
    print("=" * 90)
    print(f"""  "MDL-minimum => minimal-surface (L^3/V) minimum" is NOT a clean functional implication.

  MDL  =  [ transitivity-floor: n_orbits + coordinates + edges ]      <- shared with L^3/V
        +  [ naming: space_group + wyckoff ]  +  [ chirality ]        <- NOT in L^3/V

  * The transitivity floor is the genuine COMMON CORE: edge+vertex transitivity zeroes
    srs's coordinate/edge bits (B) AND forces the 120-deg vertex => L^3/V = 27/sqrt(2)
    (Item 1).  That is why MDL and L^3/V both pick srs.
  * But MDL also carries naming + chirality terms a geometric functional does not.  srs
    is NOT the description-cheapest there: it PAYS +1 bit for chirality and +0.17 for a
    richer Wyckoff (C).  MDL alone would happily drop the chirality bit -- so MDL does
    NOT force chirality; R-12 (physics) does.

  So srs is the maximally-symmetric / minimal-description / tightest CHIRAL net, and the
  three handles converge on it only WITHIN the chiral class.  The clean "least description
  = least area" theorem fails (different functionals); what is true is the narrower:

       edge+vertex transitivity  =>  {{ MDL coordinate/edge floor }}  AND  {{ L^3/V minimum }}.

  srs's full selection = transitivity (common core) + chirality (R-12, physics) + Sunada
  strong-isotropy (a symmetry theorem, to exclude any centrosymmetric edge-transitive
  competitor -- MDL itself does not do this).  Honest second refinement of Item 1.
  No graded content changes.""")

    if FAILURES:
        print(f"\n RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        return 1
    print("\ngyroid_mdl_vs_symmetry_optimal_2026-06-13.py: done (sentinel).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
