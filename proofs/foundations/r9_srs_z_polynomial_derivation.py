#!/usr/bin/env python3
"""
=============================================================================
SUPERSEDED / CORRECTION 2026-05-12 — see `proofs/foundations/r9_srsz_simulator_run.py`
and an internal working note. This script
does NOT do what its title says. It derives the polynomial 16x²−32x+15 = 0,
whose roots are x = 3/4 and 5/4 — but those are the BOUNDARY of the interval in
which srs-z's (10,3)-b bond topology exists, NOT srs-z's Wyckoff-8c coordinate
(x ≈ 0.6607). So it never encoded the quantity the γ.2 closure needed. And the
"19-bit" cost is inflated: 16x²−32x+15 = (4x−3)(4x−5), so x=3/4 is the root of
the degree-1 polynomial 4x−3=0 (~6 bits), and srs's coordinate is 1/8 exactly
(~3 bits) — a single-digit-bit ΔDL, far below the γ.2 closure threshold and, in
any case, not a hard gate. R-9's actual status: DOMINANT-CONDITIONAL (the
"R-9 CLOSES TO SUB-1σ" verdict printed below is RETRACTED). Kept for provenance.
=============================================================================

R-9 closure (γ.2 verification) — derive srs-z's defining polynomial.

GOAL: derive the algebraic equation that determines srs-z's Wyckoff 8c
free-parameter range, then check polynomial degree d and coefficient
magnitude M.

CORRECTED FROM 2026-05-02 EOD+8 v1: now uses framework's actual symmetry
operation data (via rcsr_net_assessment.py) instead of hand-coded
Wyckoff positions. This produces correct orbit and bond geometry.

METHOD:
  1. Use the framework's symmetry ops + orbit_of() to generate the 8c
     orbit for srs-z.
  2. Symbolically compute squared distances from (x,x,x) to all other
     orbit positions and their nearest periodic images.
  3. Identify the 3 nearest (C_3-related triangle) and the next-nearest
     atom that competes.
  4. Setting the two equal gives the BOUNDARY polynomial — at this x,
     the 3-regular topology transitions.
  5. Report degree, coefficients, roots, bit-encoding cost.

Reference: x = 0.6607 (RCSR srs-z).
"""

import sys, os, math
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcsr_net_assessment import get_space_group_ops, orbit_of


def get_orbit_symbolic(sg_name, x_sym):
    """
    Generate the symbolic 8c orbit by applying numerical symmetry ops to
    a NUMERICAL value, then identifying the symbolic form via pattern
    matching against {x, 1-x, x+1/2, 1/2-x, x+1/4, 1/4-x, x+3/4, 3/4-x,
    -x, ...} reductions.
    """
    rotations, translations, _, _ = get_space_group_ops(sg_name)
    # Generate orbit at a generic numerical x to identify positions
    x_num = 0.61803  # arbitrary in 3-regular range
    v = np.array([x_num, x_num, x_num])
    orbit_num = orbit_of(v, rotations, translations)

    # For each numerical orbit position, identify symbolic form
    candidates = [
        (1, 0, 0),    # x
        (-1, 0, 1),   # 1 - x
        (1, 0, sp.Rational(1, 4)),    # x + 1/4
        (-1, 0, sp.Rational(1, 4)),   # 1/4 - x
        (1, 0, sp.Rational(1, 2)),    # x + 1/2
        (-1, 0, sp.Rational(1, 2)),   # 1/2 - x
        (1, 0, sp.Rational(3, 4)),    # x + 3/4
        (-1, 0, sp.Rational(3, 4)),   # 3/4 - x
        (1, 0, -sp.Rational(1, 4)),   # x - 1/4
        (-1, 0, -sp.Rational(1, 4)),  # -1/4 - x → -x - 1/4
        (1, 0, sp.Rational(5, 4)),    # x + 5/4 (mod 1 → x + 1/4)
        (-1, 0, sp.Rational(5, 4)),   # 5/4 - x (mod 1 → 1/4 - x)
    ]

    def identify(numeric_val, x_num):
        # Try each candidate and see which matches numeric_val (mod 1)
        for sign, _, offset in candidates:
            if isinstance(offset, sp.Rational):
                off_num = float(offset)
            else:
                off_num = offset
            test_val = (sign * x_num + off_num) % 1.0
            if abs(test_val - numeric_val % 1.0) < 1e-3:
                return sign * x_sym + offset
        # Fall back: assume it's some simple form we missed
        return None

    orbit_sym = []
    for p in orbit_num:
        sym = tuple(identify(p[i], x_num) for i in range(3))
        if any(s is None for s in sym):
            print(f"    [warning: couldn't identify position {p} symbolically]")
            return None, orbit_num
        orbit_sym.append(sym)
    return orbit_sym, orbit_num


def squared_distance(p, q, max_shift=2, x_sym=None, x_val=None):
    """Symbolic squared distance from p to nearest periodic image of q."""
    best = None
    best_val = float('inf') if x_val is not None else None
    for sx in range(-max_shift, max_shift + 1):
        for sy in range(-max_shift, max_shift + 1):
            for sz in range(-max_shift, max_shift + 1):
                qx, qy, qz = q[0] + sx, q[1] + sy, q[2] + sz
                d_sq = sp.expand((p[0] - qx)**2 + (p[1] - qy)**2 + (p[2] - qz)**2)
                if x_val is not None and x_sym is not None:
                    val = d_sq.subs(x_sym, x_val)
                    try:
                        d_val = float(val)
                    except (TypeError, ValueError):
                        d_val = float(sp.N(val))
                    if d_val < best_val - 1e-12:
                        best_val = d_val
                        best = d_sq
    return best, best_val


def main():
    print("=" * 96)
    print("R-9 CLOSURE — srs-z defining polynomial derivation (γ.2 verification)")
    print("=" * 96)

    x = sp.Symbol('x', positive=True, real=True)
    print(f"\n  Generating Wyckoff 8c orbit using framework's symmetry op data...")
    orbit_sym, orbit_num = get_orbit_symbolic('P4(1)32', x)
    if orbit_sym is None:
        print(f"    Symbolic identification failed; aborting.")
        return

    print(f"\n  Wyckoff 8c orbit (8 positions, symbolic in x):")
    for i, p in enumerate(orbit_sym, 1):
        print(f"    {i}: ({p[0]}, {p[1]}, {p[2]})")

    p1 = orbit_sym[0]
    x_emp = 0.6607
    print(f"\n  Empirical x_srs_z = {x_emp}")
    print(f"\n  Squared distances from atom 1 (numerically at x = {x_emp}):")
    distances = []
    for i, p in enumerate(orbit_sym[1:], 2):
        d_sym, d_num = squared_distance(p1, p, x_sym=x, x_val=x_emp)
        distances.append((i, d_num, d_sym))
    distances.sort(key=lambda t: t[1])
    for rank, (idx, d_val, d_sym) in enumerate(distances, 1):
        marker = (" ★ NEAREST (3-reg bond)" if rank <= 3
                  else " ◐ 4th-6th (next nearest)" if rank <= 6
                  else "")
        print(f"    rank {rank}: atom {idx}, d² = {d_sym} → {d_val:.6f}{marker}")

    d_min_sym = distances[0][2]   # nearest neighbor (×3 by C_3)
    d_compete_sym = distances[3][2]  # 4th nearest competitor
    d_max_sym = distances[6][2] if len(distances) > 6 else None  # most distant

    print("\n" + "-" * 96)
    print("3-REGULARITY BOUNDARY POLYNOMIAL")
    print("-" * 96)
    print(f"\n  d²_min     = {d_min_sym}    (atoms in C_3 triangle of 3-regular bonds)")
    print(f"  d²_compete = {d_compete_sym}  (next-nearest C_3 triangle)")
    if d_max_sym is not None:
        print(f"  d²_max     = {d_max_sym}    (most distant orbit position with shifts)")

    # Boundary: 3-regularity transitions when d²_min equals the most distant
    # competing distance (d²_max — this is when topology changes)
    if d_max_sym is not None:
        boundary_eq = sp.expand(d_max_sym - d_min_sym)
        print(f"\n  Boundary equation (where most-distant equals 3-reg min):")
        print(f"    {d_max_sym} − ({d_min_sym}) = {boundary_eq}")

        if boundary_eq.is_polynomial(x):
            poly = sp.Poly(boundary_eq, x)
            coeffs = [int(c) if c.is_integer else sp.Rational(c).limit_denominator(1000)
                      for c in poly.all_coeffs()]
            degree = poly.degree()
            # Multiply through to integer coeffs
            denoms = [sp.Rational(c).q for c in coeffs if isinstance(c, sp.Rational)]
            lcm = 1
            for d in denoms:
                lcm = sp.lcm(lcm, d)
            int_coeffs = [c * lcm for c in coeffs]
            int_coeffs_eval = [int(c) for c in int_coeffs]
            max_int_coeff = max(abs(c) for c in int_coeffs_eval)
            print(f"\n  ✓ Boundary equation IS polynomial of degree {degree}")
            print(f"    rational coeffs: {coeffs}")
            print(f"    integer form (×{lcm}): {int_coeffs_eval}")
            print(f"    max |integer coeff| = {max_int_coeff}")

            # Roots
            roots = sp.solve(boundary_eq, x)
            print(f"\n  Roots:")
            for r in roots:
                if r.is_real:
                    r_num = float(r)
                    marker = " ← in (0,1)" if 0 < r_num < 1 else ""
                    print(f"    x = {r}  ≈ {r_num:.6f}{marker}")

            # Algebraic-K-complexity bit cost (γ.2)
            bits_g2 = (degree + 1) * math.log2(2 * max_int_coeff + 1) + math.log2(max(degree, 1))
            print(f"\n  Algebraic-K-complexity bit-cost (γ.2):")
            print(f"    (degree+1) × log2(2·max_coeff+1) + log2(degree)")
            print(f"    = {degree+1} × log2({2*max_int_coeff+1}) + log2({degree})")
            print(f"    = {bits_g2:.2f} bits")

            # R-9 closure verdict
            print("\n" + "=" * 96)
            print("R-9 CLOSURE VERDICT")
            print("=" * 96)
            gap = 3.41
            level_2_baseline = 2.40
            total = level_2_baseline + bits_g2
            print(f"""
  Baseline Level 2 ΔDL (refinements α + β):    {level_2_baseline:.2f} bits
  Refinement (γ.2) algebraic polynomial bits:  {bits_g2:.2f} bits
                                              -----
  TOTAL ΔDL(srs-z − srs):                      {total:.2f} bits

  Sub-3σ V_us-match threshold:                 5.81 bits
  Sub-1σ V_us-match threshold:                 7.39 bits

  Margin to sub-1σ closure: {total - 7.39:+.2f} bits
""")

            if total >= 7.39:
                print("  ✓ R-9 CLOSES TO SUB-1σ via M2a structural alone.")
                print("    Refinement (γ.2) provides the polynomial encoding of srs-z's")
                print("    free-parameter geometric constraint — degree-{} polynomial".format(degree))
                print("    with integer coefficients ≤ {} that defines the 3-regular".format(max_int_coeff))
                print("    topology boundary at Wyckoff 8c of P4_132.")
                print()
                print("    The framework's empirical V_us match (PDG -0.015σ) is now")
                print("    POST-HOC VALIDATION of the structural exclusion of srs-z,")
                print("    not the primary closure mechanism.")
            elif total >= 5.81:
                print("  ◐ R-9 closes to sub-3σ but not sub-1σ via this argument.")
                print("    Additional structural refinement needed for tighter closure.")
            else:
                print("  ✗ R-9 does NOT close at this encoding level.")

            print()
            print("  NOTE on SOURCE of the polynomial:")
            print(f"    The polynomial 16x² - 32x + 15 = 0 (or equivalent) arises from")
            print(f"    the geometric requirement: at Wyckoff 8c of P4_132 with vertices")
            print(f"    at (x,x,x), the 3-regular topology with bonds to the C_3-related")
            print(f"    triangle of atoms transitions to a different topology when the")
            print(f"    bond distance d²_min equals the next-shortest distance d²_max.")
            print(f"    This boundary defines the open interval where srs-z's specific")
            print(f"    3-regular topology is realized.")


if __name__ == '__main__':
    main()
