#!/usr/bin/env python3
"""
Investigation #2-followup — ANALYTICAL FESHBACH at the Ramanujan-circle
saddle, derived via Sokhotski-Plemelj outside-radial limit.

DERIVATION:
  S(h) = ∫_0^{2π} ρ(φ) / (h - √2 e^{iφ}) dφ
       = (1/√2) ∫ ρ(φ)/(z_h - e^{iφ}) dφ  with z_h = e^{i·arg h}

  Substituting z = e^{iφ}, dz = iz dφ, and Fourier-expanding
  ρ(φ) = (1/2π) Σ_n M_n e^{inφ}, each per-mode integral becomes:

    I_n = (1/(2πi)) ∮_|z|=1 z^{n-1} dz / (z_h - z)

  In the OUTSIDE-RADIAL limit (z_h on contour, take radial limit |z_h|>1):
    n ≥ 1 :  pole at z=z_h is outside contour       → I_n = 0
    n = 0 :  pole at z=0 inside, residue 1/z_h       → I_0 = 1/z_h
    n = -m (m≥1): pole at z=0 of order m+1, res 1/z_h^{m+1}
                                                     → I_{-m} = 1/z_h^{m+1}

  For real density (M_{-m} = M_m), summing gives the closed form:

      ┌─────────────────────────────────────────────────────────┐
      │                                                         │
      │    Σ(h) = (α_1/h) · [ M_0 + Σ_{m≥1} M_m · e^{-imα} ]    │
      │                                                         │
      │    where α = arg h, |h| = √2 = Ramanujan radius         │
      │                                                         │
      └─────────────────────────────────────────────────────────┘

CONSEQUENCES:

  1. Leading (M_0=1, M_n=0 for n≥1) — universal, substrate-invariant:
        Σ_lead = α_1/h = α_1 · h̄/|h|² = α_1 · (√3 - i√5)/4
        Re/α_1 = √3/4 ≈ 0.433  (= Re(h)/|h|²)
        -Im/α_1 = √5/4 ≈ 0.559  (= Im(h)/|h|² = m_ν dark coefficient ★)

  2. Subleading (M_n ≠ 0 for n≥1) — substrate-specific:
        ΔΣ = (α_1/h) · Σ_{m≥1} M_m e^{-imα}
        This is the substrate-dependent CORRECTION to the universal
        leading term.

  3. The Sokhotski-Plemelj P.V. value would be α_1/(2h) = HALF of a separate private derivation by the author. The OUTSIDE-RADIAL limit α_1/h IS the correct physical
     prescription (causal +iε for a saddle approaching spectrum from
     outside). This is the convention a separate private derivation by the author uses.

  4. Discrete Feshbach sum is K-noisy approximation: at finite K-grid,
     different eigenvalues happen to lie within ±tolerance of |λ|²=2,
     and the resulting weighted sum doesn't converge cleanly (cf. Inv
     #2 K-convergence study). The analytical formula above is the
     well-defined limit.

This script:
  - Implements the closed-form Σ(h) given {M_n}
  - Verifies leading = framework's √5/4 (m_ν) and √3/4 (Re part)
  - Computes substrate-specific subleading predictions from empirical M_n
  - Cross-checks against framework's existing dark coefficients
"""

import sys, os, math
import numpy as np

H_SADDLE = complex(math.sqrt(3) / 2, math.sqrt(5) / 2)
ALPHA_1_BARE = (2 / 3) ** 8


def sigma_analytical(M_n_list, h=H_SADDLE, alpha_1=ALPHA_1_BARE):
    """
    Closed-form Σ(h) at Ramanujan-circle boundary (outside-radial limit).
    M_n_list: complex array M[0], M[1], ..., M[N-1]. For Hermitian density,
              M_n is real with M_{-n} = M_n; only positive-mode list needed.
    """
    alpha = math.atan2(h.imag, h.real)
    bracket = M_n_list[0] + sum(M_n_list[m] * complex(math.cos(m*alpha), -math.sin(m*alpha))
                                for m in range(1, len(M_n_list)))
    return alpha_1 * bracket / h


def main():
    print("=" * 96)
    print("INVESTIGATION #2-followup — ANALYTICAL FESHBACH at Ramanujan-circle saddle")
    print("=" * 96)
    alpha = math.atan2(H_SADDLE.imag, H_SADDLE.real)
    print(f"\n  Saddle:     h = (√3 + i√5)/2 = {H_SADDLE}")
    print(f"  |h|² = {abs(H_SADDLE)**2:.6f} (= 2 = Ramanujan radius²)")
    print(f"  arg h = α = {alpha:.6f} rad = {math.degrees(alpha):.4f}°")
    print(f"  cos α = √3/√8 = {math.cos(alpha):.6f}")
    print(f"  sin α = √5/√8 = {math.sin(alpha):.6f}")
    print(f"  α_1 = (2/3)^8 = {ALPHA_1_BARE:.6f}")

    # ---------- Step 1: Verify leading reproduces a separate private derivation by the author α_1/h ----------
    print("\n" + "-" * 96)
    print("STEP 1 — Leading (M_0=1, all M_n>0 = 0): does it reproduce framework dark coefficients?")
    print("-" * 96)
    M_uniform = [1.0] + [0.0] * 15
    sig_lead = sigma_analytical(M_uniform)
    re_norm = sig_lead.real / ALPHA_1_BARE
    im_norm = sig_lead.imag / ALPHA_1_BARE
    print(f"\n  Σ_lead = (α_1/h) · M_0 = {sig_lead.real:+.6f} {sig_lead.imag:+.6f}i")
    print(f"  Re(Σ_lead)/α_1 = {re_norm:+.6f}  (analytical: √3/4 = {math.sqrt(3)/4:.6f})")
    print(f"  Im(Σ_lead)/α_1 = {im_norm:+.6f}  (analytical: -√5/4 = {-math.sqrt(5)/4:.6f})")
    print(f"  Framework dark coeff -Im/α_1 = {-im_norm:.6f}  (target: √5/4 = m_ν dark)")
    leading_re_match = abs(re_norm - math.sqrt(3)/4) < 1e-10
    leading_im_match = abs(im_norm - (-math.sqrt(5)/4)) < 1e-10
    print(f"\n  ✓ Leading EXACTLY reproduces √3/4 (Re) and √5/4 (-Im) "
          f"({'YES' if (leading_re_match and leading_im_match) else 'NO'})")

    # ---------- Step 2: M_2 = -0.27 modulation predicted subleading correction ----------
    print("\n" + "-" * 96)
    print("STEP 2 — M_2 modulation (cos(2φ) substrate density structure, M_2 = -0.27 from Inv #3)")
    print("-" * 96)
    M_with_M2 = [1.0, 0.0, -0.27] + [0.0] * 13
    sig_M2 = sigma_analytical(M_with_M2)
    re_M2 = sig_M2.real / ALPHA_1_BARE
    im_M2 = sig_M2.imag / ALPHA_1_BARE
    delta_re = re_M2 - re_norm
    delta_im = im_M2 - im_norm
    print(f"\n  Σ_M2 = (α_1/h) · [1 + M_2 e^{{-2iα}}] = {sig_M2.real:+.6f} {sig_M2.imag:+.6f}i")
    print(f"  Re(Σ_M2)/α_1   = {re_M2:+.6f}   (Δ from leading: {delta_re:+.4f}, "
          f"{delta_re/re_norm*100:+.1f}%)")
    print(f"  -Im(Σ_M2)/α_1  = {-im_M2:+.6f}   (Δ from leading: {-delta_im:+.4f}, "
          f"{-delta_im/(-im_norm)*100:+.1f}%)")
    print(f"\n  M_2 PREDICTION:")
    print(f"    cos(2φ) modulation INCREASES Re(Σ)/α_1 by ~40%")
    print(f"    cos(2φ) modulation DECREASES |Im(Σ)|/α_1 by ~13.5% (from 0.559 to 0.484)")

    # ---------- Step 3: Per-substrate analytical Σ from empirical M_n ----------
    # M_n values from Inv #3 (q_space_extended_probe.py at K=6)
    # Index: M[0], M[2], M[3], M[4], M[6], M[8], M[10], M[12]
    EMPIRICAL_MN = {
        # name: dict of {n: M_n}
        'srs-z':   {0: 1.0, 2: -0.2606, 3: 0.0,    4: -0.1361, 6: -0.0760, 8: -0.0502, 10: -0.0451, 12: -0.0223},
        'srs-c4':  {0: 1.0, 2: -0.2640, 3: 0.0,    4: -0.1460, 6: -0.1011, 8: -0.1066, 10: +0.3077, 12: -0.2597},
        'hcb-c4':  {0: 1.0, 2: -0.2929, 3: 0.0,    4: -0.1893, 6: +0.2054, 8: -0.2616, 10: +0.0085, 12: -0.3440},
        'srs-c27': {0: 1.0, 2: -0.2802, 3: 0.0,    4: -0.1593, 6: -0.1089, 8: -0.1007, 10: +0.3538, 12: -0.1876},
        'srs':     {0: 1.0, 2: -0.2802, 3: 0.0,    4: -0.1593, 6: -0.1089, 8: -0.1007, 10: +0.3538, 12: -0.1876},
        'lou':     {0: 1.0, 2: -0.2548, 3: 0.0012, 4: -0.1301, 6: -0.0688, 8: -0.0398, 10: -0.0281, 12: -0.0269},
    }

    print("\n" + "-" * 96)
    print("STEP 3 — Per-substrate analytical Σ(h) from empirical M_n (Inv #3 data, K_GRID=6)")
    print("-" * 96)
    print(f"\n  {'name':<10s}  {'Re(Σ)/α_1':>10s}  {'-Im(Σ)/α_1':>11s}  {'Δ Re vs lead':>13s}  {'Δ -Im vs lead':>14s}")
    print(f"  {'(leading)':<10s}  {math.sqrt(3)/4:>10.4f}  {math.sqrt(5)/4:>11.4f}  {'(reference)':>13s}  {'(reference)':>14s}")
    for name, mn_dict in EMPIRICAL_MN.items():
        # Build M_n list up to n=12, filling missing with 0
        M_list = [mn_dict.get(n, 0.0) for n in range(13)]
        sig = sigma_analytical(M_list)
        re_n = sig.real / ALPHA_1_BARE
        im_n = sig.imag / ALPHA_1_BARE
        d_re = re_n - math.sqrt(3)/4
        d_im_neg = (-im_n) - math.sqrt(5)/4
        print(f"  {name:<10s}  {re_n:>+10.4f}  {-im_n:>+11.4f}  "
              f"{d_re:>+13.4f}  {d_im_neg:>+14.4f}")

    # ---------- Step 4: Sub-leading match against framework % corrections ----------
    print("\n" + "-" * 96)
    print("STEP 4 — Predicted % shift on framework dark corrections from M_2 modulation")
    print("-" * 96)
    print(f"""
  Framework's leading dark corrections use Im(Σ)/α_1 = √5/4 ≈ 0.559 (m_ν, V_us, β family).
  M_2 = -0.27 modulation predicts -Im(Σ)/α_1 → 0.484 (DECREASE by 13.5%).

  Framework predictions affected (if M_n correction is the right physics):
    m_ν2:  uses (1 + √5/4·α_1).  M_2 correction → (1 + 0.484·α_1)
           Shift: (0.484-0.559)·α_1 = -0.0029. m_ν2 changes by -0.29%.
    V_us:  uses similar (1 + 5/12·α_1/(1-α_1)) form via different derivation.
    β cosmic birefringence: uses Im(Σ)/α_1 ≈ √5/4 family.

  Question: does the framework's m_ν2 prediction match PDG more closely
  WITH or WITHOUT this -13.5% shift on the dark coefficient?

  Currently m_ν2 predicts ratio m_ν2/m_ν3 within ~1% of PDG. A -0.29% shift
  is at the precision boundary — could improve or degrade the match.

  SUBLEADING TEST: this is a specific, falsifiable prediction.
""")


if __name__ == '__main__':
    main()
