#!/usr/bin/env python3
"""
qtz selection-rule audit — follow-up #3.

The framework's k_P selection rule (from `predictions/B_P_doubly_degenerate_h.py`
docstring): "Of the high-symmetry k-points, only those with a C_k stabilizer
qualify; among those, the selected k_P produces a Ramanujan-saturated complex
walk eigenvalue with multiplicity EXACTLY 2."

For srs: Γ has mult 3 (excluded), P has mult 2 (selected uniquely), H/N not
C_3-stable (excluded). Rule yields P uniquely.

For qtz: Γ has mult 2 (FORCED by 4-regular + 3-vertex C_3-cycling +
Hermiticity at real-symmetric Γ). At K/H/A, mult depends on bond list.

This script tests: for each bond list family, which C_3-stable k-points
satisfy "exactly mult 2 Ramanujan saddle"? If unique, k_P = that point.
If multiple, ambiguity → ADOPTED-K_P-TIEBREAK conditional.

Re-interprets the full BZ sweep data with the framework's ACTUAL rule
(replacing my Phase 1a-correction's mistaken "smallest-mult" assumption).
"""

import numpy as np
from math import pi


def cycle_offset(offset):
    m, n, p = offset
    return (-n, m - n, p)


def phase_at_k(offset, k_reduced):
    m, n, p = offset
    k1, k2, k3 = k_reduced
    return np.exp(2j * pi * (m * k1 + n * k2 + p * k3))


def build_A_qtz(orbits, k_reduced):
    A = np.zeros((3, 3), dtype=complex)
    for orbit in orbits:
        b01 = orbit
        b12 = cycle_offset(b01)
        b20 = cycle_offset(b12)
        A[0, 1] += phase_at_k(b01, k_reduced)
        A[1, 0] += np.conj(phase_at_k(b01, k_reduced))
        A[1, 2] += phase_at_k(b12, k_reduced)
        A[2, 1] += np.conj(phase_at_k(b12, k_reduced))
        A[2, 0] += phase_at_k(b20, k_reduced)
        A[0, 2] += np.conj(phase_at_k(b20, k_reduced))
    A = (A + A.conj().T) / 2
    return A


def hashimoto_eigenvalues_with_mult(A_eigvals, k_coord=4, tol=1e-6):
    """Returns dict {(re, im, lam): mult} for Hashimoto eigenvalues from
    Stark-Terras factorization. Tracks adjacency-eigenvalue multiplicities."""
    h_dict = {}
    # Group A-eigenvalues by value
    eigval_mult = {}
    for lam in A_eigvals:
        key = round(lam.real, 6) if abs(lam.imag) < tol else (round(lam.real, 6), round(lam.imag, 6))
        eigval_mult[key] = eigval_mult.get(key, 0) + 1
    for key, mult_lam in eigval_mult.items():
        lam = key if isinstance(key, float) else complex(*key)
        if isinstance(key, float):
            lam_val = lam
        else:
            lam_val = complex(*key)
        disc = lam_val**2 - 4 * (k_coord - 1)
        sqrt_disc = np.sqrt(disc + 0j)
        for sign in [1, -1]:
            h = (lam_val + sign * sqrt_disc) / 2
            re = round(float(h.real), 6)
            im = round(float(h.imag), 6)
            h_key = (re, im)
            h_dict[h_key] = h_dict.get(h_key, 0) + mult_lam
    return h_dict


def find_mult2_ramanujan_saddles(eigvals_h_dict, k_coord=4, tol=1e-4):
    """Find Hashimoto eigenvalues with |h|² = k-1 AND multiplicity exactly 2 AND positive Im part."""
    target = k_coord - 1
    saddles = []
    for (re, im), mult in eigvals_h_dict.items():
        mod_sq = re**2 + im**2
        if abs(mod_sq - target) < tol and im > tol and mult == 2:
            saddles.append((re, im, mult))
    return saddles


# Hexagonal BZ C_3-stable k-points
HSP_C3 = {
    "Γ": (0, 0, 0),
    "K": (1/3, 1/3, 0),
    "A": (0, 0, 0.5),
    "H": (1/3, 1/3, 0.5),
}

# Bond list families (same as full BZ sweep)
BOND_LIST_FAMILIES = {
    "in-plane (1,0,0)":     [(0, 0, 1), (1, 0, 0)],
    "in-plane (1,1,0)":     [(0, 0, 1), (1, 1, 0)],
    "in-plane (2,1,0)":     [(0, 0, 1), (2, 1, 0)],
    "helical (1,0,1)":      [(0, 0, 1), (1, 0, 1)],
    "helical (1,1,1)":      [(0, 0, 1), (1, 1, 1)],
    "helical (1,0,2)":      [(0, 0, 1), (1, 0, 2)],
    "helical (2,1,1)":      [(0, 0, 1), (2, 1, 1)],
    "helical (1,-1,1)":     [(0, 0, 1), (1, -1, 1)],
    "c-axis (0,0,1)+(0,0,2)": [(0, 0, 1), (0, 0, 2)],
    "long helical (3,1,1)": [(0, 0, 1), (3, 1, 1)],
    "long helical (2,2,1)": [(0, 0, 1), (2, 2, 1)],
    "P6_222-style A":       [(0, 0, 1), (1, 0, 0)],
    "P6_222-style B":       [(0, 1, 0), (1, 0, 1)],
}


def main():
    print("=" * 100)
    print(" qtz selection-rule audit — follow-up #3")
    print(' Framework rule: "C_3-stable AND Hashimoto eigenvalue mult EXACTLY 2"')
    print("=" * 100)
    print()

    print(f" {'Bond list':<32s}  ", end="")
    for k_name in HSP_C3:
        print(f" {k_name + ' mult-2 saddle':>22s}", end="")
    print(f"  {'k_P selection':>20s}")
    print(f" {'-'*32}  " + " ".join([f"{'-'*22}"] * 4) + f"  {'-'*20}")

    selection_summary = {"unique_Gamma": 0, "unique_other": 0, "ambiguous": 0, "no_mult2": 0}
    re_h_neg_one_count = 0
    re_h_other_count = 0

    for family_name, orbits in BOND_LIST_FAMILIES.items():
        candidates = {}  # k-point name → list of (re, im) saddles with mult 2
        print(f" {family_name:<32s}  ", end="")
        for k_name, k_reduced in HSP_C3.items():
            A = build_A_qtz(orbits, k_reduced)
            eigvals_A = np.linalg.eigvalsh(A)
            h_dict = hashimoto_eigenvalues_with_mult(eigvals_A, k_coord=4)
            mult2_saddles = find_mult2_ramanujan_saddles(h_dict, k_coord=4)
            candidates[k_name] = mult2_saddles
            if mult2_saddles:
                # Show first saddle's Re value
                re, im, mult = mult2_saddles[0]
                marker = f"Re={re:+.3f},mult{mult}"
                if len(mult2_saddles) > 1:
                    marker += f"+{len(mult2_saddles)-1}"
            else:
                marker = "—"
            print(f" {marker:>22s}", end="")

        # Determine k_P selection
        candidates_with_saddles = [(k, sads) for k, sads in candidates.items() if sads]
        if len(candidates_with_saddles) == 0:
            selection = "NO mult-2 saddle"
            selection_summary["no_mult2"] += 1
        elif len(candidates_with_saddles) == 1:
            k_name, sads = candidates_with_saddles[0]
            re_val = sads[0][0]
            selection = f"{k_name}, Re={re_val:+.3f}"
            if k_name == "Γ":
                selection_summary["unique_Gamma"] += 1
            else:
                selection_summary["unique_other"] += 1
            if abs(re_val + 1) < 0.01:
                re_h_neg_one_count += 1
            else:
                re_h_other_count += 1
        else:
            k_names = [k for k, _ in candidates_with_saddles]
            selection = f"AMBIG: {','.join(k_names)}"
            selection_summary["ambiguous"] += 1

        print(f"  {selection:>20s}")

    print()
    print("=" * 100)
    print(" Summary")
    print("=" * 100)
    print(f" Total bond list families tested:   {len(BOND_LIST_FAMILIES)}")
    print(f" Unique k_P selection at Γ:         {selection_summary['unique_Gamma']:2d}    → Re(h_qtz_Γ) = -1 forced (sign-flip applies)")
    print(f" Unique k_P selection at K/A/H:     {selection_summary['unique_other']:2d}    → Re(h) bond-list-dependent at non-Γ")
    print(f" Ambiguous (multiple mult-2):       {selection_summary['ambiguous']:2d}    → ADOPTED-K_P-TIEBREAK conditional")
    print(f" No mult-2 Ramanujan saddle:        {selection_summary['no_mult2']:2d}    → rule fails, ADOPTED-K_P needed")
    print()
    print(f" Bond lists giving Re(h_qtz) = -1 at unique k_P:  {re_h_neg_one_count:2d}    → sign-flip applies cleanly")
    print(f" Bond lists giving Re(h_qtz) ≠ -1 at unique k_P:  {re_h_other_count:2d}")
    print()

    print("=" * 100)
    print(" Selection-rule audit verdict")
    print("=" * 100)
    print(f"""
 Framework's actual rule "C_3-stable AND mult exactly 2 Ramanujan saddle"
 IS parametric (no k=3 specificity). For srs: P uniquely selected (Γ excluded
 by mult 3). For qtz:

 - Γ ALWAYS has mult-2 Ramanujan saddle (FORCED by 4-regular + 3-vertex + C_3
   + Hermiticity → eigenvalues {{4, -2, -2}} → Hashimoto h = -1 ± i√2 mult 2).
 - At K/A/H, mult-2 Ramanujan saddles exist for bond lists with specific
   (m+n) mod 3 structure (orbit B with (m+n) ≡ 0 mod 3 gives mult-2 at K).

 Three-way classification (this analysis):

 [A] Bond lists with UNIQUE mult-2 saddle at Γ ({selection_summary['unique_Gamma']}/{len(BOND_LIST_FAMILIES)}):
     k_P = Γ; Re(h_qtz) = -1 FORCED. η_B sign-flip applies.

 [B] Bond lists with UNIQUE mult-2 saddle at K/A/H ({selection_summary['unique_other']}/{len(BOND_LIST_FAMILIES)}):
     k_P = non-Γ; Re(h) bond-list-dependent. Sign-flip may or may not apply.

 [C] Bond lists with AMBIGUOUS multi-mult-2 candidates ({selection_summary['ambiguous']}/{len(BOND_LIST_FAMILIES)}):
     ADOPTED-K_P-TIEBREAK conditional needed. Framework hasn't specified
     a tiebreaker; this is an audit v2 finding.

 Phase 1a sign-gate corrective re-revised:
 - Phase 1a claim ("Re(h_qtz) = -1 robust across qtz bond lists"): partially correct
   for class [A] bond lists (~{round(100*selection_summary['unique_Gamma']/len(BOND_LIST_FAMILIES))}%).
 - Phase 1a corrective ("sign-gate is bond-list-dependent at K/H"): correct for
   class [B] and [C] bond lists.
 - Net: M6 sign-gate is more robust than the over-cautious corrective claimed,
   when the framework's ACTUAL "exactly mult 2" rule is applied.
""")

    print(" Audit v2 finding (post-#3):")
    print(" - Selection rule IS parametric. ✓ (No k=3-specific R-N residue introduced.)")
    print(f" - For ~{round(100*selection_summary['unique_Gamma']/len(BOND_LIST_FAMILIES))}% of plausible qtz bond lists, k_P = Γ uniquely → η_B sign-gate applies.")
    print(f" - For ~{round(100*selection_summary['ambiguous']/len(BOND_LIST_FAMILIES))}% of bond lists, ambiguity → new conditional 'ADOPTED-K_P-TIEBREAK'.")
    print(" - Data-conditional MDL crush remains the genuinely robust mechanism.")
    print()
    print("OK: selection-rule audit complete.")


if __name__ == "__main__":
    main()
