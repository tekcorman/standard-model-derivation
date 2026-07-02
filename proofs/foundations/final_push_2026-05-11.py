"""
proofs/foundations/final_push_2026-05-11.py

One more comprehensive push:
  §A. Hashimoto eigenvector parity (P, T, C symmetry labels)
  §B. Sub-leading Hashimoto content beyond Ramanujan saddles
  §C. Algebraic identities catalog: ALL combinations of substrate constants
  §D. Specific test: arg(h_H) vs R-14 candidate arccos(1/3) — are they
      different substrate objects or related?
"""

import math
import sys
import itertools
from pathlib import Path
from fractions import Fraction

import numpy as np
from numpy import linalg as la
from collections import Counter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from simulator.srs_engine.srs_substrate import SrsSubstrate

substrate = SrsSubstrate()


# ============================================================
# §A. Hashimoto eigenvector parity
# ============================================================

def hashimoto_parity():
    print("=" * 100)
    print("§A. Hashimoto eigenvector parity (under directed-edge reversal)")
    print("=" * 100)
    print()
    print("  Parity P: directed edge e = (src, tgt, cell) → rev(e) = (tgt, src, -cell)")
    print("  P is an involution on the 12-dim directed-edge space.")
    print("  Each Hashimoto eigenmode has P eigenvalue ±1 (parity label).")
    print()

    bonds = substrate.bonds
    nB = len(bonds)
    # Build parity permutation matrix
    P = np.zeros((nB, nB), dtype=complex)
    for e_idx, (src, tgt, cell) in enumerate(bonds):
        rev_cell = tuple(-c for c in cell)
        rev_e = (tgt, src, rev_cell)
        for f_idx, (fsrc, ftgt, fcell) in enumerate(bonds):
            if (fsrc, ftgt, fcell) == rev_e:
                P[f_idx, e_idx] = 1
                break

    # Verify P² = I
    print(f"  P² = I deviation: {la.norm(P @ P - np.eye(nB)):.4e}")
    P_evals = la.eigvals(P)
    pos = sum(1 for e in P_evals if abs(e - 1) < 0.01)
    neg = sum(1 for e in P_evals if abs(e + 1) < 0.01)
    print(f"  P has eigenvalues: {pos} × (+1), {neg} × (-1)")
    print(f"  (6 P-even + 6 P-odd subspaces, total 12 = 2|E|)")
    print()

    for k_name in ['Gamma', 'P', 'N', 'H']:
        B = substrate.hashimoto_at_k(k_name)
        evals, evecs = la.eig(B)

        # Sort by |λ|
        order = np.argsort(-np.abs(evals))
        evals = evals[order]
        evecs = evecs[:, order]

        # P eigenvalue for each eigenvector
        labels = []
        for i, v in enumerate(evecs.T):
            v_P = P @ v
            # P preserves the subspace if v_P is parallel to v
            if la.norm(v) > 1e-10:
                coeff = np.vdot(v, v_P) / np.vdot(v, v)
                if abs(coeff - 1) < 0.05:
                    label = '+1'
                elif abs(coeff + 1) < 0.05:
                    label = '-1'
                else:
                    label = '?'
                labels.append((evals[i], label, coeff))

        # Group by |λ|
        print(f"  --- k = {k_name} ---")
        by_mag = {}
        for e, lab, c in labels:
            mag = round(abs(e), 4)
            by_mag.setdefault(mag, []).append(lab)
        for mag, labs in sorted(by_mag.items(), reverse=True):
            c = Counter(labs)
            c_str = ", ".join(f"{l}×{n}" for l, n in c.items())
            print(f"    |λ| = {mag}: P-labels {c_str}")


# ============================================================
# §B. Sub-leading Hashimoto content (|λ| ≠ √2)
# ============================================================

def sub_leading():
    print()
    print("=" * 100)
    print("§B. Sub-leading Hashimoto eigenvalues (beyond Ramanujan saddles)")
    print("=" * 100)
    print()
    print("  At |λ| ≠ √2 (the Ramanujan bound), what does the substrate contain?")
    print("  Look at |λ| = 1 (unit modulus) eigenmodes at each k-point.")
    print()

    for k_name in ['Gamma', 'P', 'N', 'H']:
        B = substrate.hashimoto_at_k(k_name)
        evals = la.eigvals(B)
        # |λ| = 1 eigenmodes
        unit = [e for e in evals if abs(abs(e) - 1) < 0.001]
        print(f"  --- k = {k_name} ---")
        print(f"    {len(unit)} eigenmodes with |λ| = 1 (unit modulus)")
        # Group by exact arg
        args = Counter()
        for e in unit:
            arg = round(math.degrees(math.atan2(e.imag, e.real)), 2)
            args[arg] += 1
        for arg, count in sorted(args.items()):
            print(f"      arg = {arg:+.2f}°, mult = {count}")


# ============================================================
# §C. Algebraic identities catalog
# ============================================================

def algebraic_identities():
    print()
    print("=" * 100)
    print("§C. Exhaustive algebraic identities catalog")
    print("=" * 100)
    print()

    # Catalog of substrate constants (real values, exact where possible)
    constants = {
        # Counts
        'k*': 3,
        '|V|': 4,
        '|E|': 6,
        'g': 10,
        'g-2': 8,
        'k*²': 9,
        '|V|·k*': 12,
        '|V|·g': 40,
        '2^k*': 8,
        '|V|·|E|': 24,
        # Rationals
        '(k-1)/k': Fraction(2, 3),
        '1/k*': Fraction(1, 3),
        '1/|V|': Fraction(1, 4),
        '1/g': Fraction(1, 10),
        '5/12': Fraction(5, 12),
        '5/3': Fraction(5, 3),
        '3/5': Fraction(3, 5),
        '3/8': Fraction(3, 8),
        '9/40': Fraction(9, 40),
        '1/24': Fraction(1, 24),
        # Powers of 2/3
        '(2/3)^2': Fraction(4, 9),
        '(2/3)^4': Fraction(16, 81),
        '(2/3)^6': Fraction(64, 729),
        '(2/3)^8': Fraction(256, 6561),
        '(2/3)^10': Fraction(1024, 59049),
        # Sqrts
        '√3': math.sqrt(3),
        '√5': math.sqrt(5),
        '√7': math.sqrt(7),
        '√2': math.sqrt(2),
    }

    # Look for triple identities A·B = C or A+B = C
    print(f"  Searching for triple identities A · B = C and A + B = C:")
    print()

    found_products = []
    found_sums = []
    found_quotients = []
    items_list = list(constants.items())
    for (name_a, val_a) in items_list:
        for (name_b, val_b) in items_list:
            if name_a > name_b: continue  # avoid double-counting
            v_a = float(val_a)
            v_b = float(val_b)
            for (name_c, val_c) in items_list:
                if name_c == name_a or name_c == name_b: continue
                v_c = float(val_c)
                # Product
                if abs(v_a * v_b - v_c) < 1e-9 and v_c > 0.001:
                    found_products.append((name_a, name_b, name_c, v_c))
                # Sum
                if abs(v_a + v_b - v_c) < 1e-9 and v_c > 0.001:
                    found_sums.append((name_a, name_b, name_c, v_c))
                # Quotient
                if abs(v_b) > 1e-10 and abs(v_a / v_b - v_c) < 1e-9 and v_c > 0.001:
                    found_quotients.append((name_a, name_b, name_c, v_c))

    print(f"  Products A · B = C (showing first 20 distinct):")
    seen = set()
    for a, b, c, v in found_products[:50]:
        key = tuple(sorted([a, b]))
        if (key, c) in seen: continue
        seen.add((key, c))
        print(f"    {a} · {b} = {c}  (value {v:.6f})")
    print()
    print(f"  Sums A + B = C (showing first 20 distinct):")
    seen = set()
    for a, b, c, v in found_sums[:20]:
        key = tuple(sorted([a, b]))
        if (key, c) in seen: continue
        seen.add((key, c))
        print(f"    {a} + {b} = {c}  (value {v:.6f})")


# ============================================================
# §D. arg(h_H) vs R-14 arccos(1/3)
# ============================================================

def arg_h_H_vs_R14():
    print()
    print("=" * 100)
    print("§D. arg(h_H) = arctan(√7) ≈ 69.30° vs R-14 candidate arccos(1/3) ≈ 70.53°")
    print("=" * 100)
    print()
    print("  Both are framework candidates for CKM δ_CP. Are they related?")
    print()
    arg_h_H = math.degrees(math.atan(math.sqrt(7)))
    arccos_1_3 = math.degrees(math.acos(1/3))
    print(f"  arg(h_H) = arctan(√7) = {arg_h_H:.6f}°")
    print(f"  arccos(1/3) = {arccos_1_3:.6f}°")
    print(f"  Difference: {arccos_1_3 - arg_h_H:.6f}°")
    print()
    print(f"  cos(arg(h_H)) = cos(arctan(√7)) = 1/√(1+7) = 1/√8 = 1/(2√2) = √2/4 ≈ {math.cos(math.atan(math.sqrt(7))):.6f}")
    print(f"  cos(arccos(1/3)) = 1/3 ≈ {1/3:.6f}")
    print()
    print(f"  These give DIFFERENT substrate objects:")
    print(f"    R-14: cos(δ_CKM) = T_{{B-L}} eigenvalue = +1/3 (color sector)")
    print(f"    h_H:  cos(δ_?) = √2/4 = 1/(2√2)")
    print()
    print(f"  Are these structurally related? Try identities:")
    print(f"    1/(2√2) · 2 · √2 = 1   ← trivial")
    print(f"    (1/3)² = 1/9, (1/(2√2))² = 1/8 — close but different")
    print(f"    cos² → eigenvalues of T_BL: 1/9 vs 1/8 — no clean relation")
    print()
    print(f"  Both within CKM γ = 65.9° ± 3.5°:")
    print(f"    R-14 candidate: 70.53° (Δ = +4.63° ≈ 1.3σ)")
    print(f"    h_H candidate:  69.30° (Δ = +3.40° ≈ 0.97σ)")
    print(f"    Both within 2σ; h_H is slightly closer to PDG central.")
    print()
    print(f"  These are TWO DIFFERENT substrate candidates for CKM δ_CP.")
    print(f"  Both are framework-derivable. Neither has been shown to be")
    print(f"  the canonical one. R-14's identification predates h_H finding.")


def main():
    print("Final push: parity, sub-leading, identities, R-14 cross-check")
    print()
    hashimoto_parity()
    sub_leading()
    algebraic_identities()
    arg_h_H_vs_R14()


if __name__ == "__main__":
    main()
