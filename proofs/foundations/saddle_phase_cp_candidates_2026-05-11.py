"""
proofs/foundations/saddle_phase_cp_candidates_2026-05-11.py

Test whether substrate saddle args + cross-saddle phase differences match
ANY observed CP-violating phase or mixing angle in the SM.

Numerical observation: arg(h_H) = arctan(√7) ≈ 69.30°. CKM CP phase
γ ≈ 65.9° ± 3.5° (unitarity triangle). Close match? Worth checking.

Also: arg(h_N) = arctan(√(3/5)) ≈ 37.76°. Various mixing angles.
"""

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    print("=" * 100)
    print("Substrate saddle phases vs observed CP / mixing angles")
    print("=" * 100)
    print()

    # All saddle arguments
    saddle_args = {
        'arg(h_P) = arctan(√5/√3) = arctan(√(5/3))': math.degrees(math.atan(math.sqrt(5/3))),
        'arg(h_N) = arctan(√3/√5) = arctan(√(3/5))': math.degrees(math.atan(math.sqrt(3/5))),
        'arg(h_H) = arctan(√7)': math.degrees(math.atan(math.sqrt(7))),
        'arg(h_Γ) = 180 − arctan(√7)': 180 - math.degrees(math.atan(math.sqrt(7))),
    }

    # Cross-saddle phase differences
    arg_P = math.degrees(math.atan(math.sqrt(5/3)))
    arg_N = math.degrees(math.atan(math.sqrt(3/5)))
    arg_H = math.degrees(math.atan(math.sqrt(7)))
    arg_G = 180 - arg_H

    cross_args = {
        'arg(h_P) − arg(h_N)': arg_P - arg_N,
        'arg(h_P) − arg(h_H)': arg_P - arg_H,
        'arg(h_P) − arg(h_Γ)': arg_P - arg_G,
        'arg(h_N) − arg(h_H)': arg_N - arg_H,
        'arg(h_N) − arg(h_Γ)': arg_N - arg_G,
        'arg(h_H) − arg(h_Γ)': arg_H - arg_G,
        # Sums
        'arg(h_P) + arg(h_N)': arg_P + arg_N,
        'arg(h_P) + arg(h_H)': arg_P + arg_H,
        'arg(h_N) + arg(h_H)': arg_N + arg_H,
        # Half-angles
        'arg(h_P) / 2': arg_P / 2,
        'arg(h_N) / 2': arg_N / 2,
        'arg(h_H) / 2': arg_H / 2,
        # Twice
        '2·arg(h_P)': 2 * arg_P,
        '2·arg(h_N)': 2 * arg_N,
        '2·arg(h_H)': 2 * arg_H,
    }

    # Observed CP/mixing angles (PDG/NuFIT/lattice)
    observed = {
        'CKM δ_CP (γ unitarity triangle, PDG 2024)': (65.9, 3.5),  # ± degrees
        'CKM α angle': (84.0, 4.5),
        'CKM β angle': (22.5, 0.5),
        'PMNS δ_CP (NuFIT 6.0)': (177.0, 20.0),
        'PMNS θ_12': (33.45, 0.7),
        'PMNS θ_13': (8.57, 0.11),
        'PMNS θ_23': (49.7, 1.3),
        'CKM θ_12 (Cabibbo)': (13.04, 0.05),
        'CKM θ_13': (0.20, 0.01),  # very small
        'CKM θ_23': (2.36, 0.06),
        'Weinberg θ_W(M_Z) sin θ_W ≈ 0.481 → θ_W ≈ 28.7°': (28.7, 0.1),
        'TBM θ_12 = arctan(1/√2) ≈ 35.26°': (35.26, 0),
        'TBM θ_23 = 45°': (45.0, 0),
        'arccos(1/3) ≈ 70.53° (K_4 dihedral, framework δ_CKM)': (70.53, 0),
        'Cabibbo-Zenczykowski angle ≈ 70.53°': (70.53, 0),
    }

    print("Direct saddle arguments:")
    print(f"  {'name':<55} {'value°':>10}  {'matches'}")
    print(f"  {'-'*55} {'-'*10}  {'-'*60}")
    for name, val in saddle_args.items():
        matches = []
        for obs_name, (obs_val, obs_err) in observed.items():
            diff = abs(val - obs_val)
            if obs_err > 0 and diff < 3 * obs_err:
                matches.append(f"{obs_name} ({obs_val}±{obs_err}, Δ={diff:.2f}°)")
            elif obs_err == 0 and diff < 2:
                matches.append(f"{obs_name} (Δ={diff:.2f}°)")
        match_str = "; ".join(matches[:2]) if matches else "—"
        print(f"  {name:<55} {val:>+10.4f}  {match_str}")

    print()
    print("Cross-saddle phase differences:")
    print(f"  {'name':<35} {'value°':>10}  {'matches'}")
    print(f"  {'-'*35} {'-'*10}  {'-'*70}")
    for name, val in cross_args.items():
        # Normalize to (-180, 180]
        val_norm = ((val + 180) % 360) - 180
        matches = []
        for obs_name, (obs_val, obs_err) in observed.items():
            diff = abs(val_norm - obs_val)
            # Also try abs (some phases are signed)
            diff_abs = abs(abs(val_norm) - obs_val)
            min_diff = min(diff, diff_abs)
            if obs_err > 0 and min_diff < 2 * obs_err:
                matches.append(f"{obs_name} (Δ={min_diff:.2f}°)")
            elif obs_err == 0 and min_diff < 1:
                matches.append(f"{obs_name} (Δ={min_diff:.2f}°)")
        match_str = "; ".join(matches[:2]) if matches else "—"
        print(f"  {name:<35} {val_norm:>+10.4f}  {match_str}")

    print()
    print("=" * 100)
    print("Notable near-matches (< 3 PDG σ or < 2° from a clean angle)")
    print("=" * 100)
    print()

    # Compile candidates
    candidates = []
    for name, val in {**saddle_args, **cross_args}.items():
        val_norm = ((val + 180) % 360) - 180 if name in cross_args else val
        for obs_name, (obs_val, obs_err) in observed.items():
            diff = abs(val_norm - obs_val)
            diff_abs = abs(abs(val_norm) - obs_val)
            min_diff = min(diff, diff_abs)
            tol = 3 * obs_err if obs_err > 0 else 2.0
            if min_diff < tol:
                candidates.append((name, val_norm, obs_name, obs_val, min_diff, tol))

    print(f"  {'substrate quantity':<45} {'value°':>10}  {'observed':<60} {'Δ°':>8}")
    print(f"  {'-'*45} {'-'*10}  {'-'*60} {'-'*8}")
    for substrate_name, sub_val, obs_name, obs_val, diff, tol in candidates:
        print(f"  {substrate_name[:43]:<45} {sub_val:>+10.4f}  {obs_name[:58]:<60} {diff:>8.4f}")

    if not candidates:
        print("  (none found within stated tolerances)")

    # Specific noteworthy patterns
    print()
    print("Closest specific matches worth follow-up:")
    print(f"  arg(h_H) = 69.30° vs CKM δ_CP (γ) = 65.9° ± 3.5°: Δ = {abs(arg_H - 65.9):.2f}°")
    print(f"    (within 1σ of PDG; identifies arg(h_H) = arctan(√7) as CKM CP candidate)")
    print()
    print(f"  arg(h_H) = 69.30° vs arccos(1/3) = 70.53° (framework's existing δ_CKM identification):")
    print(f"    Δ = {abs(arg_H - 70.53):.2f}° (close, not equal — separate substrate object)")
    print()
    print(f"  2·arg(h_P) = 104.48° vs PMNS δ_CP = 177° ± 20°:")
    print(f"    Δ = {abs(2*arg_P - 177):.2f}° (within 4σ — possible)")
    print()
    print(f"  arg(h_N) + arg(h_H) = 107.06° vs PMNS δ_CP = 177° ± 20°:")
    print(f"    Δ = {abs(arg_N + arg_H - 177):.2f}° (within 4σ)")
    print()
    print(f"  arg(h_P) - arg(h_N) = 14.48° vs CKM Cabibbo θ_12 = 13.04° ± 0.05°:")
    print(f"    Δ = {abs((arg_P - arg_N) - 13.04):.2f}° (within ~30σ)")
    print()

    # Numerical relationships
    print("=" * 100)
    print("Numerical algebraic patterns")
    print("=" * 100)
    print()
    print(f"  arctan(√7) · 2 = 2·arg(h_H) = {2*arg_H:.4f}°")
    print(f"  arctan(√7) + arctan(√(5/3)) = {arg_H + arg_P:.4f}°")
    print(f"  arctan(√7) − arctan(√(5/3)) = {arg_H - arg_P:.4f}°")
    print()
    print(f"  Note: arctan(√(5/3)) + arctan(√(3/5)) = 90° (complementary)")
    print(f"  Verified: arg(h_P) + arg(h_N) = {arg_P + arg_N:.4f}° (≈ 90.00°)")
    print()
    print(f"  arg(h_N) = arctan(√(3/5)) ≈ 37.76°")
    print(f"  90° − arg(h_N) = arg(h_P) = arctan(√(5/3)) ≈ 52.24°")
    print(f"  This is the substrate's 'h_P / h_N complementarity' — the R/I swap is")
    print(f"  the 90°-complement of the argument.")

if __name__ == "__main__":
    main()
