#!/usr/bin/env python3
"""Phase 5.1 S4 -- independent cross-check of the native little-group result.

Spec: docs/scoping/phase5_1_ebr_spec_2026-06-11.md (S4; HALT RULE: any
disagreement with the native probe stops the arc until resolved).

spgrep (spglib team, v0.6) enumerates small representations of space-group
little groups on the fly -- irrep machinery sharing NO code with the native
probe phase5_1_little_groups_saddle_irreps_2026-06-11.py. Division of
labor: the symmetry operations are constructed NATIVELY from the crystal
(same solved-from-atoms+bonds route as the S1 probe, duplicated here so the
committed S1 probe stays frozen); spgrep supplies only the independent
irrep enumeration via its low-level API
get_spacegroup_irreps_from_primitive_symmetry (the high-level cell API
mis-standardizes this primitive setting -- it re-bases k; documented here,
worked around). spglib's space-group identification is gated as X1.
IMPORT USED AS A FALSIFIABLE REFERENCE (standing discipline), never as the
computation of record.

Gates:
  X1 spglib identifies the srs cell as I4_132 (No. 214).
  X2 little co-group orders at Gamma/H/P/N = 24/24/12/4 (native G4a).
  X3 small-irrep dimension menus from spgrep:
       Gamma: [1,1,2,3,3]   (ordinary O -- native F-GH)
       H:     [1,1,2,3,3]   (ordinary O -- native F-GH)
       P:     [2,2,2]       (ALL EVEN -> independent confirmation of the
                             NONTRIVIAL factor system -- native F-P)
       N:     [1,1,1,1]     (trivial D2 -- native F-N)
     and the native probe's observed block dims are a subset of each menu.
  X4 sum(d^2) = |little co-group| at every saddle (menu completeness).
"""
import os
import sys
from itertools import permutations, product

import numpy as np
from numpy import linalg as la

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds, ATOMS, A_PRIM  # noqa: E402

import spglib  # noqa: E402
from spgrep.core import get_spacegroup_irreps_from_primitive_symmetry  # noqa: E402

FAILURES = []
M_CART = A_PRIM.T
M_INV = la.inv(M_CART)


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --- native space-group construction (as S1 probe; duplicated, frozen there)
bonds = find_bonds()
EDGES = [(i, j, tuple(int(x) for x in c)) for (i, j, c) in bonds]
E_INDEX = {e: a for a, e in enumerate(EDGES)}


def mod1(v):
    w = np.asarray(v, float) % 1.0
    w[np.abs(w - 1.0) < 1e-9] = 0.0
    return w


def is_bcc(v):
    w = np.asarray(v, float)
    if np.max(np.abs(w - np.round(w))) < 1e-9:
        return True
    w2 = w - 0.5
    return np.max(np.abs(w2 - np.round(w2))) < 1e-9


def canon_tau(v):
    cands = [mod1(v), mod1(np.asarray(v, float) + 0.5)]
    keys = [tuple(np.round(c, 6)) for c in cands]
    return cands[keys.index(min(keys))]


def atom_of(pos):
    for j in range(4):
        L = pos - ATOMS[j]
        if is_bcc(L):
            return j, L
    return None, None


def prim_int(v_cart):
    d = M_INV @ np.asarray(v_cart, float)
    di = np.round(d)
    assert np.max(np.abs(d - di)) < 1e-9
    return di.astype(int)


def op_preserves(R, tau):
    for i in range(4):
        if atom_of(R @ ATOMS[i] + tau)[0] is None:
            return False
    for (i, j, c) in EDGES:
        i2, Li = atom_of(R @ ATOMS[i] + tau)
        j2, Lj = atom_of(R @ (ATOMS[j] + M_CART @ np.asarray(c, float)) + tau)
        if (i2, j2, tuple(prim_int(Lj) - prim_int(Li))) not in E_INDEX:
            return False
    return True


OPS = []
for perm in permutations(range(3)):
    for signs in product((1, -1), repeat=3):
        R = np.zeros((3, 3))
        for row, (col, s) in enumerate(zip(perm, signs)):
            R[row, col] = s
        if abs(la.det(R) - 1.0) > 1e-9:
            continue
        seen = set()
        for j in range(4):
            tau = canon_tau(ATOMS[j] - R @ ATOMS[0])
            key = tuple(np.round(tau, 6))
            if key in seen:
                continue
            seen.add(key)
            if op_preserves(R, tau):
                OPS.append((R, tau))
assert len(OPS) == 24, f"native construction found {len(OPS)} ops"

ROT_PRIM = np.array([np.round(M_INV @ R @ M_CART).astype(int) for R, _ in OPS])
TRANS_PRIM = np.array([mod1(M_INV @ t) for _, t in OPS])


def main():
    print("=" * 72)
    print(" PHASE 5.1 S4 -- spgrep cross-check (independent irrep engine)")
    print("=" * 72)

    cell = (A_PRIM, (ATOMS @ la.inv(A_PRIM)) % 1.0, [1, 1, 1, 1])
    sg = spglib.get_spacegroup(cell, symprec=1e-6)
    gate("X1 spglib space group = I4_132 (214)", "214" in sg, sg)

    saddles = {
        "Gamma": np.zeros(3),
        "H": np.array([0.5, 0.5, -0.5]),
        "P": np.array([0.25, 0.25, 0.25]),
        "N": A_PRIM @ np.array([0.0, 0.5, 0.5]),
    }
    expected_orders = {"Gamma": 24, "H": 24, "P": 12, "N": 4}
    expected_menus = {
        "Gamma": [1, 1, 2, 3, 3],
        "H": [1, 1, 2, 3, 3],
        "P": [2, 2, 2],
        "N": [1, 1, 1, 1],
    }
    native_observed = {        # irrep-block dims seen in the 12-dim fiber
        "Gamma": [1, 2, 3, 3],
        "H": [1, 2, 3, 3],
        "P": [2, 2, 2],
        "N": [1, 1, 1, 1],
    }

    for nm, k in saddles.items():
        irreps, mapping = get_spacegroup_irreps_from_primitive_symmetry(
            ROT_PRIM, TRANS_PRIM, kpoint=np.asarray(k, float))
        order = len(mapping)
        dims = sorted(ir.shape[1] for ir in irreps)
        gate(f"X2 [{nm}] little co-group order = {expected_orders[nm]}",
             order == expected_orders[nm], f"spgrep={order}")
        gate(f"X3 [{nm}] small-irrep dim menu = {expected_menus[nm]}",
             dims == expected_menus[nm], f"spgrep={dims}")
        gate(f"X3' [{nm}] native observed dims subset of menu",
             all(d in dims for d in set(native_observed[nm])),
             f"native={native_observed[nm]}")
        gate(f"X4 [{nm}] sum(d^2) = |co-group|",
             sum(d * d for d in dims) == expected_orders[nm],
             f"{sum(d * d for d in dims)}")
        if nm == "P":
            gate("X3'' [P] menu ALL EVEN -> nontrivial factor system "
                 "independently confirmed", all(d % 2 == 0 for d in dims))

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}")
        print(" HALT RULE: native-vs-reference disagreement -- arc stops here.")
        return 1
    print(" RESULT: ALL GATES PASS -- native result independently confirmed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
