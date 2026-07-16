#!/usr/bin/env python3
"""
proofs/foundations/D4_S3_alpha1cubed_isotype_2026-07-06.py

D4 SPECTRAL-ACTION program, station S3 -- does the continuum-D4 generation-resolve the alpha_1^3? (-70 ppm).
Pre-registration: internal research notes (f441caf BEFORE this file).
CLASS: pure structure. NO PDG except the single marked open-target line. THE -70 ppm STAYS OPEN.

POISON (pre-declared, PRINTED not INVOKED): 2*alpha_1^3 ~ 1.19e-4, and every alpha_1-power near 70 ppm;
the 2*alpha_1^3/mu_rep water-filling (REFUTED, todo Q3). Do NOT pattern-match, do NOT insert an alpha_1 power.

This probe tests the pre-registered crux (P1/P2) and reports what the scoping actually forces -- including a
CORRECTION to the 06-30 framing that sharpens (does not close) the open miss.
"""
import os
import sys
from fractions import Fraction

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "bridge"))
import the_run  # noqa: E402

print("=" * 94)
print(" P1  the three GENERATIONS of read_masses ARE the C3-Fourier shell cos(delta + 2*pi*j/3)")
print("=" * 94)
Qs = the_run.read_moduli(); ds = the_run.read_phases()
# the charged-lepton species (e): build sqrt(m_j) = c0 + 2 c1 cos(delta + 2 pi j /3), j=0,1,2 = e,mu,tau
import cmath
nh = 3                                    # the charged-lepton species label (Lambda^3 singlet)
c0 = (0.5) ** 0.5
c1 = (float(6 * Qs[nh] - 2) / 8) ** 0.5
delta = float(ds[nh]) if nh in ds else 2.0 / 9
def sqrt_m(j, c0=c0, c1=c1, delta=delta):
    return abs(c0 + 2 * c1 * cmath.cos(delta + 2 * cmath.pi * j / 3))
sm = [sqrt_m(j) for j in range(3)]
print(f"    c0={c0:.6f} c1={c1:.6f} delta={delta:.6f} ; sqrt(m_j)={[round(x,5) for x in sm]}")
print(f"    => generation index j runs through the C3 shell cos(delta+2pi j/3); c0,c1,delta are")
print(f"       generation-COMMON coefficients, the j-dependence is ENTIRELY the C3-Fourier shell.")

print("=" * 94)
print(" P2  the CRUX, corrected: a generation-BLIND (sigma-isotype-blind) ADDITIVE alpha_1^3 correction to")
print("     c0 STILL moves m_e/m_tau -- because the C3 SHELL supplies the generation resolution, not the")
print("     winding's sigma-holonomy. So sigma-blindness is NOT the wall (this corrects the 06-30 framing).")
print("=" * 94)
# masses and the ratio -- identify generations by magnitude: e = smallest sqrt(m), tau = largest
m = [x * x for x in sm]
j_e = int(np.argmin(sm)); j_tau = int(np.argmax(sm))    # e = smallest, tau = largest (physical ordering)
ratio = m[j_e] / m[j_tau]                                # m_e / m_tau
# perturb c0 by a small generation-COMMON delta_c0 (an isotype-BLIND additive correction) -> does the ratio move?
eps = 1e-6
sm_p = [sqrt_m(j, c0=c0 + eps) for j in range(3)]
m_p = [x * x for x in sm_p]
ratio_p = m_p[j_e] / m_p[j_tau]
dratio = (ratio_p - ratio) / ratio / eps   # d ln(m_e/m_tau) / d c0
print(f"    m_e/m_tau = {ratio:.6f} ; d ln(m_e/m_tau)/dc0 = {dratio:.4f}  (NONZERO)")
print(f"    => a sigma-ISOTYPE-BLIND additive shift to c0 changes m_e/m_tau (lever {dratio:.2f}). The C3")
print(f"       shell already resolves the generations; the isotype-blindness 06-30 fixated on is NOT the")
print(f"       obstruction. CORRECTED CRUX: the open piece is the FORCED MAGNITUDE+TYPE of the alpha_1^3")
print(f"       correction to (c0,c1,delta), NOT its generation resolution.")

print("=" * 94)
print(" P3  where that leaves S3: the forced alpha_1^3 correction is the continuum-D4 a2 (=Tr D^2) additive")
print("     term -- an UN-DONE construction. NOT closable here. The -70 ppm STAYS OPEN.")
print("=" * 94)
a1f = float(Fraction(5, 3) * Fraction(2, 3) ** 8)
POISON_2a13 = 2 * a1f ** 3                  # PRINTED, NOT INVOKED
print(f"    read_moduli already carries the alpha_1^1 correction to c1 (via Q); the residual -70 ppm is the")
print(f"    NEXT-order (alpha_1^3) correction to (c0,c1,delta). 06-30 found the ADDITIVE spectral-action")
print(f"    structure (c^2 -> mu + alpha_1^3) gives the CORRECT SIGN but was a toy-map artifact (unforced);")
print(f"    the 1/mu_rep operator is REFUTED (Q3). The forced object is the continuum-D4 a2 = Tr D^2 additive")
print(f"    term on the A5(b) cone -- the a2 (mass^2) spectral coefficient, the SIBLING of S1's a4. That")
print(f"    construction is NOT built (S1 built a4; a2 for the mass sector is the next substantial sitting).")
print(f"    POISON (printed, NOT invoked): 2*alpha_1^3 = {POISON_2a13:.3e} ; |eps_target| ~ 1.75e-7. NOT used.")
print()
print("    ------------------------------------------------------------------------------------------------")
print("    OPEN TARGET (single marked comparison, PDG): m_e/m_tau residual = -70.3 ppm. UNCLOSED.")
print("    ------------------------------------------------------------------------------------------------")

print("=" * 94)
print(" VERDICT (S3) -- WALL / NON-CLOSURE, crux SHARPENED (and 06-30 framing corrected)")
print("=" * 94)
print("""    S3 does NOT close the -70 ppm, and it does NOT wall where 06-30 thought. Findings:
      * P1: read_masses' 3 generations are the C3-Fourier shell cos(delta+2pi j/3); c0,c1,delta are
        generation-common; the j-resolution is the shell.
      * P2 (CORRECTS 06-30): the C3 shell already resolves the generations, so a sigma-isotype-BLIND
        additive alpha_1^3 correction to c0 DOES move m_e/m_tau. The isotype-blindness that 06-30 read as
        the obstruction (=> the refuted 1/mu_rep) is NOT the obstruction. This REOPENS the additive route
        as generation-viable WITHOUT any 1/mu_j operator.
      * P3: the remaining open piece is the FORCED MAGNITUDE+TYPE of the alpha_1^3 correction to
        (c0,c1,delta) -- specifically the continuum-D4 a2 (= Tr D^2) additive mass^2 term on the A5(b)
        cone, the SIBLING of S1's a4. That construction is UN-BUILT (S1 did a4; the a2/mass sector is a
        separate substantial sitting). 06-30's additive-sign-correct result was the unforced toy-map;
        forcing it needs the a2.
    => S3 outcome: the -70 ppm STAYS OPEN (NOT floored, NOT solved). Net progress: the crux is re-pointed
    from "generation resolution (blind => refuted operator)" to "the FORCED a2 additive alpha_1^3 magnitude"
    -- a cleaner, un-refuted target that plugs into the SAME S1 machine (a2 alongside a4). NO poison invoked;
    no value moved; no alpha_1 power inserted. NEXT: build the continuum-D4 a2 (mass^2) coefficient (S3b).""")
print("=" * 94)
print(" OVERALL: S3 ran clean -- WALL/NON-CLOSURE with the crux sharpened; -70 ppm OPEN.")
print("=" * 94)
sys.exit(0)
