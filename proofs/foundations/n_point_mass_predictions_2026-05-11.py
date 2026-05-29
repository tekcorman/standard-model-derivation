"""
proofs/foundations/n_point_mass_predictions_2026-05-11.py

Test whether N-point apparatus reproduces ANY observed mass observable
that the framework's P-point apparatus doesn't.

The framework's existing P-point derivation:
  V_Ram at P has C_3 multiplicities (4, 2, 2)
  Born-rule Koide-like ratio:
    Q = (Σ p_j) / (Σ √p_j)² where p_j are multiplicities
    With (4,2,2):  Q = 8 / (2 + √2 + √2)² = ?
  But framework's derivation gives Q = 2/3 via SPECIFIC mass identification
  using h_P = (√3+i√5)/2 in arg/cos manipulation.

For N-point:
  V_Ram at N has C_3 multiplicities (2, 0, 0)
  Only trivial isotype — no ω or ω̄.
  CANNOT construct 3-generation Koide from (2,0,0) alone.

But N has 4 adjacency eigenvalues: {-√5, -1, +1, +√5} with one labeled
'1' under C_3. The (2,0,0) V_Ram refers to the doubled-multiplicity
form. The actual mass content is the 1 trivial isotype.

Let's compute the saddle h_N's contribution to mass-like observables.
"""

import math
import sys
from pathlib import Path
from fractions import Fraction

import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from simulator.srs_engine.srs_substrate import SrsSubstrate

substrate = SrsSubstrate()


def main():
    print("=" * 100)
    print("N-point mass-prediction test — does h_N + ±√5 reproduce any observable?")
    print("=" * 100)
    print()

    h_P_re = math.sqrt(3) / 2
    h_P_im = math.sqrt(5) / 2
    h_N_re = math.sqrt(5) / 2
    h_N_im = math.sqrt(3) / 2

    print("Framework's existing apparatus (P-point):")
    print(f"  V_Ram multiplicities (4, 2, 2)")
    print(f"  h_P = (√3 + i√5)/2,  Re/Im ratio = √3/√5 = √(3/5) = 0.7746")
    print(f"  arg(h_P) = arctan(√5/√3) = {math.degrees(math.atan(math.sqrt(5/3))):.4f}°")
    print(f"  Class-2 mass² factor: tan²(arg h_P) = 5/3")
    print()
    print("Candidate apparatus (N-point):")
    print(f"  V_Ram multiplicities (2, 0, 0) — DIFFERENT from P (no ω content)")
    print(f"  h_N = (√5 + i√3)/2,  Re/Im ratio = √5/√3 = √(5/3) = 1.2910")
    print(f"  arg(h_N) = arctan(√3/√5) = {math.degrees(math.atan(math.sqrt(3/5))):.4f}°")
    print(f"  Class-2 mass² factor: tan²(arg h_N) = 3/5")
    print()
    print("  KEY: V_Ram (2,0,0) at N has NO ω, ω̄ components → cannot construct")
    print("  3-generation Koide structure analogous to P. N hosts a different sector.")
    print()

    # ============================================================
    # Class-1 amplitude predictions: substitute h_N for h_P
    # ============================================================
    print("=" * 100)
    print("Class-1 amplitude observable: ν_amp = |Im(h)| / |h|² at each saddle")
    print("=" * 100)
    print()
    print(f"  Class-1 amplitude is used in framework's CKM elements (V_us, V_ub, V_cb).")
    print(f"  Framework formula: m_ν correction factor = √5/4 via h_P amplitude.")
    print()
    print(f"  At h_P: ν_amp = √5/4 = {math.sqrt(5)/4:.6f}")
    print(f"  At h_N: ν_amp = √3/4 = {math.sqrt(3)/4:.6f}")
    print(f"  At h_H: ν_amp = √7/4 = {math.sqrt(7)/4:.6f}")
    print(f"  At h_Γ: ν_amp = √7/4 = {math.sqrt(7)/4:.6f}")
    print()
    print(f"  Substituting h_N for h_P in V_us derivation:")
    print(f"    Framework V_us = 9/40 = 0.225 (uses P-point apparatus, dark factor 5/12)")
    # If we use h_N's amplitude correction instead of h_P's:
    # Framework's V_us correction is c·α₁/(1−α₁) where c is the Class-2 (5/12 or 5/3) factor
    # depending on the channel.
    print(f"    With h_N's dark factor instead of h_P's:")
    alpha_bare = (2/3)**8
    alpha_full_P = (5/3) * alpha_bare
    alpha_full_N = (3/5) * alpha_bare
    print(f"      α_full_P = (5/3)(2/3)^8 = {alpha_full_P:.10f}")
    print(f"      α_full_N = (3/5)(2/3)^8 = {alpha_full_N:.10f}")
    print()

    # ============================================================
    # Direct comparison to observed quark masses (running, MS-bar)
    # ============================================================
    print("=" * 100)
    print("Observed mass ratios + framework's existing predictions")
    print("=" * 100)
    print()

    # Lepton masses (PDG 2024, MeV)
    m_e = 0.5109989  # MeV
    m_mu = 105.6583755  # MeV
    m_tau = 1776.86  # MeV

    # Quark masses (PDG MS-bar at 2 GeV, MeV)
    m_u = 2.16  # MeV
    m_d = 4.67  # MeV
    m_s = 93.4  # MeV
    m_c = 1273  # MeV
    m_b = 4180  # MeV
    m_t = 172570  # MeV (pole mass, MeV)

    print(f"  Lepton masses (PDG, MeV):")
    print(f"    m_e = {m_e}, m_μ = {m_mu}, m_τ = {m_tau}")
    print(f"    Ratios: m_μ/m_e = {m_mu/m_e:.4f}, m_τ/m_μ = {m_tau/m_mu:.4f}")
    print()
    print(f"  Quark masses (PDG MS-bar, MeV):")
    print(f"    m_u = {m_u}, m_d = {m_d}, m_s = {m_s}, m_c = {m_c}, m_b = {m_b}, m_t = {m_t}")
    print(f"    Up-type ratios: m_c/m_u = {m_c/m_u:.2f}, m_t/m_c = {m_t/m_c:.4f}")
    print(f"    Down-type ratios: m_s/m_d = {m_s/m_d:.2f}, m_b/m_s = {m_b/m_s:.2f}")
    print()

    # Framework's lepton Koide ratio
    Q_obs_lepton = (m_e + m_mu + m_tau) / (math.sqrt(m_e) + math.sqrt(m_mu) + math.sqrt(m_tau))**2
    print(f"  Lepton Q_Koide observed: {Q_obs_lepton:.6f} (framework prediction: 2/3 = 0.6667)")

    # Quark Koide ratios
    Q_obs_quark_up = (m_u + m_c + m_t) / (math.sqrt(m_u) + math.sqrt(m_c) + math.sqrt(m_t))**2
    Q_obs_quark_down = (m_d + m_s + m_b) / (math.sqrt(m_d) + math.sqrt(m_s) + math.sqrt(m_b))**2
    print(f"  Up-quark Q_Koide observed: {Q_obs_quark_up:.6f}")
    print(f"  Down-quark Q_Koide observed: {Q_obs_quark_down:.6f}")
    print()

    # ============================================================
    # The N-point hypothesis test
    # ============================================================
    print("=" * 100)
    print("HYPOTHESIS TEST: does N-point apparatus reproduce a quark Koide?")
    print("=" * 100)
    print()
    print(f"  Framework's lepton Q = 2/3 derives from V_Ram(P) (4,2,2)")
    print(f"  + Born rule.")
    print()
    print(f"  V_Ram(N) = (2, 0, 0) does NOT support 3-generation Koide structure")
    print(f"  (no ω, ω̄ components). So the simple substitution doesn't work.")
    print()
    print(f"  BUT: the N-point has 4 adjacency eigenvalues {{-√5, -1, +1, +√5}}.")
    print(f"  Could these themselves be a 4-particle spectrum?")
    print()
    print(f"  Treating |λ_i|² as 'mass²' values: |√5|² = 5, |1|² = 1, |1|² = 1, |√5|² = 5")
    print(f"  Sum of squared mass² = {(5+1+1+5)}, sum of mass = {2*math.sqrt(5) + 2}")
    print()
    print(f"  If we treat (m_1, m_2, m_3, m_4) = (√5, 1, 1, √5):")
    masses = [math.sqrt(5), 1, 1, math.sqrt(5)]
    sum_m = sum(masses)
    sum_sqrt = sum(math.sqrt(m) for m in masses)
    Q4 = sum_m / sum_sqrt**2
    print(f"    Σ m = {sum_m:.4f}, Σ √m = {sum_sqrt:.4f}, (Σ √m)² = {sum_sqrt**2:.4f}")
    print(f"    Q_4 = {Q4:.6f}")
    print()
    print(f"  Quark sector has 6 masses (3 up + 3 down), not 4. So direct mass-as-eigenvalue")
    print(f"  reading doesn't match either.")
    print()

    # ============================================================
    # Try eigenvalue RATIOS at each k-point
    # ============================================================
    print("=" * 100)
    print("Eigenvalue ratios at each k-point — compare to observed mass ratios")
    print("=" * 100)
    print()
    for k_name in ['Gamma', 'P', 'N', 'H']:
        A = substrate.adjacency_at_k(k_name)
        evals = sorted(la.eigvals(A).real, key=abs, reverse=True)
        nonzero = [e for e in evals if abs(e) > 1e-6]
        if not nonzero:
            continue
        print(f"  {k_name}: eigenvalues sorted by |λ|: {[f'{e:+.3f}' for e in evals]}")
        # Compute ratios
        biggest = max(abs(e) for e in evals)
        for e in evals:
            if abs(e) > 1e-6:
                ratio = e / biggest
                print(f"    {e:+.4f} / max_|λ| = {ratio:+.4f}")
        print()

    # Compare to mass ratios
    print(f"  Observed mass ratios (m_lower/m_higher) for comparison:")
    print(f"    Up-quark sector:   m_u/m_t = {m_u/m_t:.6f}")
    print(f"    Charm/top:         m_c/m_t = {m_c/m_t:.6f}")
    print(f"    Down/bottom:       m_d/m_b = {m_d/m_b:.6f}")
    print(f"    Strange/bottom:    m_s/m_b = {m_s/m_b:.6f}")
    print(f"    e/τ:               m_e/m_τ = {m_e/m_tau:.6e}")
    print(f"    μ/τ:               m_μ/m_τ = {m_mu/m_tau:.6e}")
    print()
    print(f"  NONE of the substrate eigenvalue ratios match the mass ratios.")
    print(f"  Mass hierarchies span MANY orders of magnitude; substrate ratios are O(1).")
    print(f"  → Direct eigenvalue-as-mass reading does NOT close quark Yukawa hierarchy.")
    print()

    # ============================================================
    # h_N · h_P* algebraic structure
    # ============================================================
    print("=" * 100)
    print("Cross-saddle products — possible quark-lepton bridge?")
    print("=" * 100)
    print()
    h_P = complex(math.sqrt(3)/2, math.sqrt(5)/2)
    h_N = complex(math.sqrt(5)/2, math.sqrt(3)/2)
    h_H = complex(0.5, math.sqrt(7)/2)
    h_G = complex(-0.5, math.sqrt(7)/2)
    saddles = {'h_P': h_P, 'h_N': h_N, 'h_H': h_H, 'h_Γ': h_G}

    print(f"  Pairwise products h_i · h_j^*:")
    print(f"  {'pair':<15} {'product':<25} {'|prod|':>8} {'arg°':>10}")
    print(f"  {'-'*15} {'-'*25} {'-'*8} {'-'*10}")
    for name_i, h_i in saddles.items():
        for name_j, h_j in saddles.items():
            if name_i >= name_j:
                continue
            p = h_i * h_j.conjugate()
            arg_p = math.degrees(math.atan2(p.imag, p.real))
            print(f"  {name_i}·{name_j}*{'':>5} {p.real:+.4f}{p.imag:+.4f}i      {abs(p):>8.4f} {arg_p:>+10.2f}")

    print()
    print(f"  All pairwise |h_i · h_j*| = 2 = k*-1 → all saddles live on Ramanujan circle.")
    print()
    print(f"  Net: cross-saddle products give NEW arg values (14.48°, -17.06°, etc.)")
    print(f"  not currently in framework. Each could correspond to a phase observable.")

    # ============================================================
    # Conclusion
    # ============================================================
    print()
    print("=" * 100)
    print("Conclusion")
    print("=" * 100)
    print("""
  Direct substitutions tested:
  - h_N → quark sector: no clean match (V_Ram structure incompatible with
    3-generation Koide; mass-ratio scales don't match)
  - Eigenvalue-as-mass: substrate ratios all O(1), observed hierarchies O(10^5)
    → no match.

  What this rules out (HONEST):
  - The naive "h_P → leptons, h_N → quarks" hypothesis at the direct
    Class-2 mass² substitution level.
  - Eigenvalue ratios as mass ratios.

  What this does NOT rule out:
  - h_N corresponds to a SUB-LEADING correction (not Class-2 directly).
  - h_N is the framework's amplitude-class with different sign convention.
  - h_N is used in a different Born-rule context (not 3-gen Koide).
  - h_H, h_Γ may map to different observables (gauge-sector running, etc.).

  Direction worth investigating next:
  - Treat each saddle as a DIFFERENT CLASS (Class-1 amplitude, Class-2 mass²,
    Class-3 edge-local, Class-4 NEW).
  - h_P → Class-2 (framework's existing)
  - h_N → Class-1? (substitute into V_us, V_ub, etc.)
  - h_H, h_Γ → Class-3 or Class-4 (new amplitude type)
  - This is a structural re-reading of the framework's dark-extraction map.
""")


if __name__ == "__main__":
    main()
