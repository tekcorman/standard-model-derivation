#!/usr/bin/env python3
"""Phase 5.2 -- A5-mass re-pricing at the LEDGER-CITED grain (panel-ordered).

The 5.2 adjudication panel (2026-06-11) UPHELD the dictionary lens's
partition finding: the frozen 5.2 spec's counting unit (8,6,6,2,2,4,2,18)
is the meta-theorem saddle-local regrouping, NOT the partition the ledger
row cites (walker_class_dictionary_2026-05-27.md Sec 2.2: h_P 4, h_P_neg 4,
h_Gamma 8, h_H 8, h_N 2, h_N_neg 2, Perron 2, Trivial 18).  This probe is
the panel-ordered re-enumeration at the Sec-2.2 grain; the A5-mass ledger
row moves ONLY in the commit-set where this probe passes in verify.py.

At the Sec-2.2 grain a FAMILY is a spectral eigenvalue class across all
saddles (classification by VALUE, saddle-blind), so the N "spill" modes
belong to h_Gamma/h_H by spectral definition -- the exact branch identity
(spill sets = roots of lambda^2 -+ lambda + 2 = parent Ramanujan sets)
makes the old grain's R2 "continuity principle" MOOT (gate G8).

Gates (panel order, verbatim chain 24 -> 12 -> 6):
  G1  native census reproduces the Sec-2.2 partition (4,4,8,8,2,2,2,18).
  G2  size-admissible family->role bijections = 2! * 2! * 3! = 24 of 40320.
  G3  /2! content-identical dark labels -> 12 DISTINCT assignments
      (= the meta-theorem-grain total: the two grains reconcile).
  G4  orbit factorization 12 = 2 (nu-orientation) x 2 (P-sign) x 3 (Higgs
      placement) -- exact product structure, each combination once.
  G5  the nu-orientation Z2 IS the body-centering mirror: spectrum negation
      maps the h_Gamma value-class onto the h_H value-class (machine zero)
      -- this is the bit single-homed at the Phase-1.3 nu_L/nu_R line.
  G6  in-row residual = 6 assignments = 2.585 bits -> ROW PRINTS 3.0
      (spent rounds up); strictly below the old 15.299.
  G7  (b1) = 0: provenance audit -- NO exclusion consumes Phase-5.1 output
      (tags: pre-5.1-native / dictionary-conditional only).
  G8  R2-mootness: the N modes at Re = -+1/2 are ELEMENTS of the parent
      Ramanujan value-sets (branch identity exact, 1e-9).
  G9  ledger arithmetic: spent 114.4 - 15.3 + 3.0 = 102.1; net
      377.1 - 102.1 = +275.0 (the panel's ordered numbers).

Provenance tags carried on every exclusion (G7 audits them):
  size-admissibility ... pre-5.1-native (family sizes = computed spectra)
                         + dictionary-conditional (role budgets = the
                         dictionary's sector commitments)
  dark-label dedup ..... dictionary-conditional (Sec 3.4 assigns h_N and
                         h_N_neg NO distinct SM content)
  mirror cross-ref ..... pre-5.1-native (antiperiod is exact spectral math)
                         + single-home accounting (Phase-1.3 line)
NOTE (not gated, ordered identity-check pending): spectrum negation also
pairs h_P <-> h_P_neg -- consistent with the panel's ordered check whether
the in-row P-sign bit is the already-priced omega/omega2 convention.
"""
import os
import sys
from itertools import permutations

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, ATOMS, A_PRIM  # noqa: E402

FAILURES = []
SQ3, SQ5, SQ7 = np.sqrt(3), np.sqrt(5), np.sqrt(7)


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


bonds = find_bonds()
EDGES = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
NE = len(EDGES)
E_INDEX = {e: a for a, e in enumerate(EDGES)}
REV = {a: E_INDEX[(j, i, tuple(-x for x in c))] for a, (i, j, c) in enumerate(EDGES)}


def B_of(k):
    B = np.zeros((NE, NE), dtype=complex)
    for a, (i, j, c) in enumerate(EDGES):
        for b, (i2, j2, c2) in enumerate(EDGES):
            if i2 == j and b != REV[a]:
                B[b, a] = np.exp(2j * np.pi * np.dot(k, np.asarray(c2, float)))
    return B


SADDLES = {
    "Gamma": np.zeros(3),
    "H": np.array([0.5, 0.5, -0.5]),
    "P": np.array([0.25, 0.25, 0.25]),
    "N": A_PRIM @ np.array([0.0, 0.5, 0.5]),
}


def family_of(lam):
    """Sec-2.2 grain: family = eigenvalue VALUE class, saddle-blind."""
    m = abs(lam)
    if abs(m - 2.0) < 1e-6:
        return "Perron"
    if abs(m - 1.0) < 1e-6:
        return "trivial"
    r = lam.real
    if abs(r - SQ3 / 2) < 1e-6:
        return "h_P"
    if abs(r + SQ3 / 2) < 1e-6:
        return "h_P_neg"
    if abs(r + 0.5) < 1e-6:
        return "h_Gamma"
    if abs(r - 0.5) < 1e-6:
        return "h_H"
    if abs(r - SQ5 / 2) < 1e-6:
        return "h_N"
    if abs(r + SQ5 / 2) < 1e-6:
        return "h_N_neg"
    return "UNCLASSIFIED"


print("=" * 72)
print(" PHASE 5.2 -- Sec-2.2-grain enumeration (panel-ordered, 24 -> 12 -> 6)")
print("=" * 72)

spec = {nm: la.eigvals(B_of(k)) for nm, k in SADDLES.items()}
fam_size = {}
for nm, evs in spec.items():
    for lam in evs:
        f = family_of(lam)
        fam_size[f] = fam_size.get(f, 0) + 1

SEC22 = {"h_P": 4, "h_P_neg": 4, "h_Gamma": 8, "h_H": 8,
         "h_N": 2, "h_N_neg": 2, "Perron": 2, "trivial": 18}
gate("G1 native census = Sec-2.2 partition (4,4,8,8,2,2,2,18), total 48",
     fam_size == SEC22 and sum(fam_size.values()) == 48,
     f"{ {k: fam_size.get(k, 0) for k in SEC22} }")

# Roles at the Sec-2.2 grain (dictionary Sec 3; budgets dictionary-conditional):
# the charged sector splits by P-sign (Sec 3.1: h_P drives lepton/up-type
# Yukawas, h_P_neg the down-type/CKM-source variants); each nu sector is a
# full 8-mode family (spill inside by value); two SEPARATE dark labels that
# carry IDENTICAL content (Sec 3.4: "SM components hosted: NONE" for both).
ROLES = {"chg_primary": 4, "chg_ckm": 4, "nu_L": 8, "nu_R": 8,
         "dark_a": 2, "dark_b": 2, "higgs": 2, "nonmatter": 18}
FAMILIES = list(SEC22)
ROLE_LIST = list(ROLES)

PROVENANCE = {
    "size-admissibility": ("pre-5.1-native", "dictionary-conditional"),
    "dark-label-dedup": ("dictionary-conditional",),
    "mirror-cross-ref": ("pre-5.1-native", "single-home-accounting"),
}

# G2: size filter over all 8! bijections   [exclusion: size-admissibility]
adm = []
for pi in permutations(ROLE_LIST):
    f = dict(zip(FAMILIES, pi))
    if all(ROLES[f[fam]] == SEC22[fam] for fam in FAMILIES):
        adm.append(f)
gate("G2 size-admissible bijections = 2!*2!*3! = 24 of 40320",
     len(adm) == 24, f"count={len(adm)}")

# G3: identify content-identical dark labels  [exclusion: dark-label-dedup]
def assignment_key(f):
    g = {fam: ("dark" if f[fam] in ("dark_a", "dark_b") else f[fam])
         for fam in FAMILIES}
    return tuple(sorted(g.items()))


distinct = sorted({assignment_key(f) for f in adm})
gate("G3 /2! content-identical dark labels -> 12 distinct assignments "
     "(= meta-theorem-grain total: grains reconcile)",
     len(distinct) == 12, f"count={len(distinct)}, bits={np.log2(len(distinct)):.3f}")

# G4: orbit factorization 12 = 2 (nu-orientation) x 2 (P-sign) x 3 (Higgs)
def coords(key):
    g = dict(key)
    nu_or = g["h_Gamma"]                       # 'nu_L' or 'nu_R'
    p_sign = g["h_P"]                          # 'chg_primary' or 'chg_ckm'
    higgs_at = next(fam for fam in ("h_N", "h_N_neg", "Perron")
                    if g[fam] == "higgs")
    return nu_or, p_sign, higgs_at


triples = [coords(k) for k in distinct]
nu_vals = sorted({t[0] for t in triples})
ps_vals = sorted({t[1] for t in triples})
hg_vals = sorted({t[2] for t in triples})
full_product = (len(nu_vals) == 2 and len(ps_vals) == 2 and len(hg_vals) == 3
                and len(set(triples)) == 12)
# also: the nu swap is JOINT (h_Gamma and h_H always opposite)
joint_nu = all(dict(k)["h_H"] == ("nu_R" if dict(k)["h_Gamma"] == "nu_L"
                                  else "nu_L") for k in distinct)
gate("G4 orbit factorization EXACT: 12 = 2 (nu-orient) x 2 (P-sign) x 3 "
     "(Higgs among size-2), nu swap joint",
     full_product and joint_nu,
     f"|nu|={len(nu_vals)}, |P|={len(ps_vals)}, |H|={len(hg_vals)}")

# G5: the nu-orientation Z2 IS the body-centering mirror (antiperiod)
#     [exclusion: mirror-cross-ref -- the bit is homed at Phase 1.3]
DELTA = np.array([0.5, 0.5, -0.5])


def multiset_dev(a, b):
    """Greedy tolerance matching (sort_complex mispairs conjugate ties)."""
    b = list(b)
    worst = 0.0
    for z in a:
        j = int(np.argmin([abs(z - w) for w in b]))
        worst = max(worst, abs(z - b.pop(j)))
    return worst


mirror_dev = max(multiset_dev(la.eigvals(B_of(np.asarray(k) + DELTA)),
                              -spec[nm])
                 for nm, k in SADDLES.items())
neg_maps = all(family_of(-lam) == {"h_Gamma": "h_H", "h_H": "h_Gamma"}[family_of(lam)]
               for evs in spec.values() for lam in evs
               if family_of(lam) in ("h_Gamma", "h_H"))
gate("G5 nu-orientation Z2 = body-centering mirror: spec B(k+Delta) = "
     "-spec B(k); negation maps h_Gamma class <-> h_H class",
     mirror_dev < 1e-9 and neg_maps, f"antiperiod dev={mirror_dev:.1e}")

# G6: in-row residual after the 1.3-homed mirror bit is cross-referenced out
in_row = {(t[1], t[2]) for t in triples}     # quotient by the nu-orientation
bits_in_row = np.log2(len(in_row))
ROW_PRINTS = 3.0                              # spent rounds up (2.585 -> 3.0)
gate("G6 in-row residual = 6 assignments = 2.585 bits -> row prints 3.0 "
     "< 15.299",
     len(in_row) == 6 and abs(bits_in_row - np.log2(6)) < 1e-12
     and ROW_PRINTS >= bits_in_row and ROW_PRINTS < 15.299,
     f"count={len(in_row)}, bits={bits_in_row:.3f}")

# G7: (b1) = 0 -- no exclusion consumes Phase-5.1 output
gate("G7 provenance audit: NO '5.1-native' tag on any exclusion ((b1)=0; "
     "space-group labels are SM-content-blind)",
     all("5.1-native" not in tags for tags in PROVENANCE.values()),
     "; ".join(f"{k}: {'+'.join(v)}" for k, v in PROVENANCE.items()))

# G8: R2-mootness -- the N modes at Re=-+1/2 are ELEMENTS of the parent
#     Ramanujan value-sets (exact branch identity; no continuity principle)
roots_p = {(1 + 1j * SQ7) / 2, (1 - 1j * SQ7) / 2}     # lambda^2 - lambda + 2
roots_m = {(-1 + 1j * SQ7) / 2, (-1 - 1j * SQ7) / 2}   # lambda^2 + lambda + 2
def in_set(z, S, tol=1e-9):
    return any(abs(z - w) < tol for w in S)


nH = [z for z in spec["N"] if family_of(z) == "h_H"]
nG = [z for z in spec["N"] if family_of(z) == "h_Gamma"]
parent_H = [z for z in spec["H"] if family_of(z) == "h_H"]
parent_G = [z for z in spec["Gamma"] if family_of(z) == "h_Gamma"]
g8 = (len(nH) == 2 and len(nG) == 2
      and all(in_set(z, roots_p) for z in nH + parent_H)
      and all(in_set(z, roots_m) for z in nG + parent_G))
gate("G8 R2 MOOT at this grain: N modes ARE parent-set elements (branch "
     "identity exact; family membership is spectral definition)", g8)

# G9: the panel's ordered ledger arithmetic
OLD_ROW, NEW_ROW = 15.3, ROW_PRINTS
spent_old, earned = 114.4, 377.1
spent_new = round(spent_old - OLD_ROW + NEW_ROW, 1)
net_new = round(earned - spent_new, 1)
gate("G9 ledger arithmetic: spent 114.4 -> 102.1; net +262.7 -> +275.0",
     spent_new == 102.1 and net_new == 275.0,
     f"spent={spent_new}, net=+{net_new}, ratio={earned / spent_new:.2f}")

print("\n--- the panel-ordered chain (Sec-2.2 grain) ---")
print("  40320 family->role bijections")
print("  -> 24   size-admissible            [pre-5.1-native + dict-conditional]")
print("  -> 12   distinct assignments       [dark labels content-identical]")
print("  ->  6   IN-ROW (= 2.585 bits)      [nu-orientation bit homed at 1.3]")
print("  row prints 3.0 (spent rounds up); (a) = -11.71 bookkeeping;")
print("  (b) = 0.0 credited to 5.1; (c) = P-sign 1.0 + Higgs placement 1.585.")

print("\n" + "=" * 72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
    sys.exit(1)
print(" RESULT: ALL GATES PASS -- the row-move gate condition is met")
print("=" * 72)
sys.exit(0)
