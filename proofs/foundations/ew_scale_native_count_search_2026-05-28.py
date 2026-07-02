"""
proofs/foundations/ew_scale_native_count_search_2026-05-28.py

Honest search: do the EW-scale gauge observables have native substrate-
symmetry-count forms, the way the UNIFICATION values do?

Reference (unification, clean structural counts):
  α_GUT      = 1/24  = 1/|Aut(K_4)|
  sin²θ_W    = 3/8   = k*²/(|V|·|E|) = 9/24

Question: does sin²θ_W(M_Z), M_Z/m_W, α_s(M_Z), α_EM(M_Z) have an equally
clean count — or do the only "matches" require non-structural integers
(numerology)?

DISCIPLINE: a match counts ONLY if numerator AND denominator are in the
structural set. Report cherry-picked matches (needing 13, 7, 17, ...) as
NUMEROLOGY, not as findings. NO forcing.
"""

from __future__ import annotations

import itertools


def banner(title, char="="):
    print(char * 100)
    print(title)
    print(char * 100)


# Structural substrate integers (with their meanings)
STRUCTURAL = {
    1: "unit",
    2: "k*-1 (NB survival numerator) / |E|/|V|",
    3: "k* (coordination)",
    4: "|V| (atoms/cell) = k*+1",
    5: "k*+2 / (g-2)/... ",       # 5 is borderline; flag it
    6: "|E| (edges)",
    8: "g-2 / |V|·2",
    9: "k*² ",
    10: "g (girth)",
    12: "2|E| (directed edges)",
    24: "|Aut(K_4)| = |V|·|E| = 4!",
}
# The genuinely clean ones (appear directly in derivations):
CLEAN = {2, 3, 4, 6, 8, 9, 10, 12, 24}
# 5 and 1 are weaker — flag if a match needs them.


# Observed EW-scale values (PDG 2024)
OBSERVED = {
    "sin²θ_W(M_Z) MS-bar":   0.23121,
    "sin²θ_W on-shell (1-m_W²/M_Z²)": 1 - (80.369/91.1876)**2,
    "α_s(M_Z)":              0.1179,
    "α_EM(M_Z)":             1/127.951,
    "m_W/M_Z":               80.369/91.1876,
    "(m_W/M_Z)²":            (80.369/91.1876)**2,
}

# Reference unification values (should match cleanly — sanity check)
REFERENCE = {
    "α_GUT = 1/24":          1/24,
    "sin²θ_W(unif) = 3/8":   3/8,
}


def search_ratios(target, tol=0.01):
    """Find all p/q with p,q in structural integers matching target within tol.
    Returns list of (p, q, value, rel_err, is_clean)."""
    matches = []
    ints = sorted(STRUCTURAL.keys())
    for p in ints:
        for q in ints:
            val = p / q
            if abs(val - target) / target < tol:
                is_clean = (p in CLEAN) and (q in CLEAN)
                matches.append((p, q, val, abs(val - target) / target, is_clean))
    # also p/(q+r) type? NO — that's where numerology starts. Keep to p/q only.
    matches.sort(key=lambda m: m[3])
    return matches


def main():
    banner("EW-scale native symmetry-count search — honest, no forcing", "#")
    print()
    print("Structural integer set (clean):", sorted(CLEAN))
    print("Weaker integers (flag if needed): 1, 5")
    print()

    banner("SANITY CHECK — unification values (should match cleanly)")
    print()
    for name, val in REFERENCE.items():
        ms = search_ratios(val, tol=0.005)
        print(f"  {name} = {val:.5f}")
        for p, q, v, err, clean in ms[:3]:
            flag = "✓ CLEAN" if clean else "✗ needs non-clean int"
            print(f"      {p}/{q} = {v:.5f}  (err {err*100:.3f}%)  [{flag}]")
        print()

    banner("EW-SCALE OBSERVABLES — do they have clean counts?")
    print()
    for name, val in OBSERVED.items():
        ms = search_ratios(val, tol=0.01)
        print(f"  {name} = {val:.5f}")
        if not ms:
            print(f"      NO structural ratio p/q within 1%.")
        else:
            for p, q, v, err, clean in ms[:4]:
                flag = "✓ CLEAN" if clean else "✗ needs non-clean int"
                print(f"      {p}/{q} = {v:.5f}  (err {err*100:.3f}%)  [{flag}]")
        print()

    banner("HONEST VERDICT")
    print()
    print("Reading guide:")
    print("  - A CLEAN match (both ints in {2,3,4,6,8,9,10,12,24}) within ~0.5% with")
    print("    a UNIQUE structural form = candidate native count, worth deriving.")
    print("  - Multiple competing matches, or matches needing non-clean ints (5,7,13,..),")
    print("    or matches only at >0.5% = NUMEROLOGY, not a finding. Reject.")
    print()
    print("  The unification values (1/24, 3/8) are UNIQUE clean counts — that's why")
    print("  they're theorem-grade. The test for the EW values is whether they have")
    print("  the SAME uniqueness, or whether forcing a count is just curve-fitting.")
    print()


if __name__ == "__main__":
    main()
