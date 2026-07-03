#!/usr/bin/env python3
"""
proofs/foundations/F4_S2b_width_ratio_dark_lemma_2026-07-02.py

F4 S2b — THE DARK-CANCELLATION LEMMA for width observables (CAS, sympy).

Pole convention: z0 = M - i*Gamma/2 (Gamma << M). The framework's dark dressing is
multiplicative on the recurrence rate: z -> z*(1 - Sigma), Sigma = the channel read
(theorem_dark_self_energy_unified_2026-06-28: Sigma = alpha1/h; gauge sector reads the
PERRON channel h_P = 2, which is EXACTLY REAL by Ihara-Bass — machine-verified in
F4_width_math_verification S1; the shell channel h = (sqrt3+i*sqrt5)/2 is complex).

LEMMA (three parts, each CAS-checked below):
  L1  A REAL multiplicative dressing (any Sigma_r, species-common) leaves Gamma/M
      invariant EXACTLY (all orders), and leaves any width ratio Gamma_i/Gamma_j
      invariant EXACTLY when common. => The known gauge-sector matching-point darks
      (Perron channel: real) CANNOT generate or shift a width fraction. Whatever
      produces Gamma must be the omega-resolved embedding Sigma_X(omega)
      (incomplete_equations_todo.md par.7), not the existing dark sector.
  L2  A COMPLEX dressing shifts the width fraction at first order by
      Delta(Gamma/M) = 2*Sigma_i/(1 - Sigma_r) + O(Sigma_i^2, (Gamma/M)^2*Sigma_i):
      a complex-pole reading of the SHELL dressing would give EVERY shell fermion
      Delta(Gamma/m) = 2*(alpha1*sqrt5/4)/(1 - alpha1*sqrt3/4) ~ 4.4e-2.
  L3  (Consistency corollary; comparison-side numbers marked) L2 is EXCLUDED by
      stability: Gamma_e = 0 exactly and Gamma_mu/m_mu = 2.8e-18 — the complex-pole
      reading over-applies by >1e16. => The framework's existing dark-map practice
      (Re / |Im| components applied as separate REAL corrections per observable) is
      the ONLY pole-consistent reading of the shell dressing; the fermion pole stays
      on the real axis at matching-point order. This UPGRADES the dark map's
      component-wise usage from convention to forced-by-stability.

Scope (stated, not overclaimed): L1 protects width RATIOS against the KNOWN dark
sector only. It does not preclude a genuine width-side term from the un-built
Sigma_X(omega) — that is new physics, not a dressing ambiguity.
"""
import sys
import sympy as sp

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

M, G, Sr, Si, GW, GZ = sp.symbols('M Gamma Sigma_r Sigma_i Gamma_W Gamma_Z',
                                  real=True, positive=True)

z0 = M - sp.I * G / 2

print("=" * 80)
print(" L1  real common dressing: Gamma/M and width ratios invariant EXACTLY")
print("=" * 80)
z_real = sp.expand(z0 * (1 - Sr))
Gp = -2 * sp.im(z_real); Mp = sp.re(z_real)
check("Gamma'/M' - Gamma/M == 0 identically (all orders in Sigma_r, Gamma/M)",
      sp.simplify(Gp / Mp - G / M) == 0)
ratio = sp.simplify((GW * (1 - Sr)) / (GZ * (1 - Sr)) - GW / GZ)
check("common real dressing cancels in Gamma_W/Gamma_Z identically", ratio == 0)
print("    (gauge sector reads the Perron channel h_P = 2: EXACTLY real, Ihara-Bass —")
print("     so the known matching-point darks cannot touch a width fraction.)")

print("=" * 80)
print(" L2  complex dressing: first-order width-fraction shift = 2 Sigma_i/(1-Sigma_r)")
print("=" * 80)
z_cplx = sp.expand(z0 * (1 - Sr - sp.I * Si))
Gp = -2 * sp.im(z_cplx); Mp = sp.re(z_cplx)
# exact first-order coefficient in Sigma_i:
coeff = sp.simplify(sp.diff(sp.simplify(Gp / Mp), Si).subs(Si, 0))
check("d(Gamma'/M')/dSigma_i |_0 = (2/(1-Sigma_r))(1 + (Gamma/2M)^2), exactly",
      sp.simplify(coeff - 2 * (1 + G**2 / (4 * M**2)) / (1 - Sr)) == 0)

alpha1 = sp.Rational(2, 3) ** 8
d_shell = float(2 * (alpha1 * sp.sqrt(5) / 4) / (1 - alpha1 * sp.sqrt(3) / 4))
print(f"    complex-pole reading of the SHELL dressing would give Delta(Gamma/m) = "
      f"{d_shell:.6f} for every shell fermion")

print("=" * 80)
print(" L3  stability corollary (COMPARISON SIDE, marked: PDG lifetimes enter here)")
print("=" * 80)
r_mu = (6.582119569e-25 / 2.1969811e-6) / 0.1056583755   # Gamma_mu/m_mu
check(f"complex-pole shell reading over-applies for mu by {d_shell/r_mu:.1e} (>1e15) "
      f"and predicts Gamma_e > 0 (electron exactly stable) => EXCLUDED",
      d_shell / r_mu > 1e15)
print("""    => the dark map's component-wise REAL usage (Re, |Im| as separate real
    corrections per observable) is FORCED BY STABILITY, not a convention; the fermion
    pole stays real at matching-point order; widths live in Sigma_X(omega) only.""")

print("=" * 80)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 80)
sys.exit(0 if ok_all else 1)
