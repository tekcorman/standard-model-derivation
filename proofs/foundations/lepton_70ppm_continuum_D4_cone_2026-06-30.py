#!/usr/bin/env python3
"""
proofs/foundations/lepton_70ppm_continuum_D4_cone_2026-06-30.py

THE CONTINUUM-D4 CONE attack on the open -70 ppm lepton miss — probe 3 of
an internal working note (the ONLY remaining route to
operator-FORCE the 1/mu_rep MDL allocation; "gates no value — a grade question").

SETUP (forced, established):
  - generations = C3 isotypes; squared return weights c_t^2 = |h_t|^2 = mu_rep = (4,2,2)
    (read_flavor / Probe 1, forced two ways).  h_0=2 (Perron, real), h_{1,2}=(-1+-i√7)/2 (shell).
  - the dark budget alpha1^3 = (2/3)^24 = 59.4 ppm is a SCALAR survival, isotype-blind
    (cycle C3-decomp uniform (5,5,5), verified).
  - the spin-1 cone (lambda=-1 triple) is the continuum limit H=v(k.S); its spectral
    action a2 = Tr(D^2) = sum of squared spectral weights is ADDITIVE (Seeley-DeWitt).

THE DECISIVE DICHOTOMY (this is the result):
  * MULTIPLICATIVE (the framework's native dark, resolvent Sigma=alpha1/h; the SAME
    object that closes m_b/m_t as m_q.(1-alpha1/h_P^p)): per-isotype it gives
        kappa_t = 2 alpha1^3 Re(h_t)/mu_t
    -> shells get Re(h)=-1/2 -> NEGATIVE.  (probe 2's wrong sign, reconfirmed.)
  * ADDITIVE (the spectral-action a2: c_t^2 -> c_t^2 + alpha1^3, uniform):
        delta m_t/m_t = + alpha1^3/(2 mu_t)
    -> the +1/mu_rep allocation with the CORRECT (+) sign FALLS OUT of the sqrt.

VERDICT (honest, NOT a closure):
  The additive spectral-action structure is what WOULD force +1/mu with the right sign.
  BUT the framework's dark is MULTIPLICATIVE (heavy_quark_anchor_dark, m_q.(1-alpha1/h_P^p)
  -- and it WORKS for m_b/m_t at the real Perron h=2).  Adopting the additive form for the
  lepton shells *because* it gives the right sign would be a FIT, and it CONTRADICTS the
  working heavy-quark dark.  So the continuum cone does NOT operator-force the allocation.
  It SHARPENS the frontier to one precise question: why would the lepton alpha1^3
  generation-ALLOCATION dark be additive (spectral-action a2) when the heavy-quark alpha1
  single-channel dark is multiplicative (resolvent)?  The MDL ceiling STANDS; the -70 ppm
  stays OPEN.  (Consistent with march12's gauge-side "+4 imported, not derived".)
  New, durable: the sign failure is now understood as the additive-vs-multiplicative
  dichotomy, and the additive a2 is identified as the structure that would close it.
"""
import numpy as np

mu = {0: 4.0, 1: 2.0, 2: 2.0}                                  # (4,2,2) squared return weights
h  = {0: 2.0+0j, 1: -0.5+0.5j*np.sqrt(7), 2: -0.5-0.5j*np.sqrt(7)}
alpha1 = (2/3)**8
a13 = alpha1**3

if __name__ == "__main__":
    print("=" * 78)
    print("  Continuum-D4 cone: does the 1/mu_rep MDL allocation operator-force? (probe 3)")
    print("=" * 78)
    print(f"\n  alpha1^3 = (2/3)^24 = {a13*1e6:.2f} ppm  (the -70 ppm dark scale; isotype-blind scalar)")
    print(f"  c_t^2 = |h_t|^2 = mu_rep = {tuple(round(abs(h[t])**2) for t in (0,1,2))}  (forced)")
    print(f"  Re(h_t) = {tuple(float(h[t].real) for t in (0,1,2))}  (Perron +2, shells -1/2)")

    print("\n  (A) MULTIPLICATIVE  (framework's native dark, resolvent; closes m_b/m_t):")
    for t in (0, 1, 2):
        k = 2*a13*h[t].real/mu[t]
        print(f"      t={t} (mu={mu[t]:.0f}): kappa = 2a1^3 Re(h)/mu = {k*1e6:+7.2f} ppm  "
              f"[{'OK' if t == 0 else 'WRONG SIGN: shell, data is +'}]")

    print("\n  (B) ADDITIVE  (spectral-action a2: c^2 -> c^2 + alpha1^3, uniform):")
    for t in (0, 1, 2):
        dmm = 0.5*a13/mu[t]
        print(f"      t={t} (mu={mu[t]:.0f}): delta m/m = +a1^3/(2 mu) = {dmm*1e6:+7.3f} ppm  [+, the 1/mu allocation]")
    print(f"      ratio triv:shell = 1/mu = {1/mu[0]:.2f}:{1/mu[1]:.2f}  -> the MDL (1/4,1/2,1/2) FALLS OUT of the sqrt")

    print("\n" + "=" * 78)
    print("  VERDICT (honest — NOT a closure; the -70 ppm stays OPEN)")
    print("=" * 78)
    print("""  The ADDITIVE spectral-action structure gives the +1/mu_rep allocation with the
  CORRECT (+) sign and resolves probe 2's sign failure.  But the framework's dark is
  MULTIPLICATIVE (heavy_quark_anchor_dark: m_q.(1-alpha1/h_P^p), which WORKS for m_b/m_t
  at the real Perron h=2).  Switching to additive for the lepton shells *because* it gives
  the right sign would be a FIT and contradicts the working heavy-quark dark.

  => The continuum-D4 cone does NOT operator-force the 1/mu_rep allocation.  The MDL
     ceiling STANDS (as the prep doc predicted: a grade frontier, itself open; consistent
     with march12's gauge-side "+4 imported, not derived").  The -70 ppm is OPEN.

  WHAT IS NEW (durable): the sign failure is now understood as the additive(a2)-vs-
  multiplicative(resolvent) dichotomy; the ADDITIVE a2 is the identified structure that
  WOULD close it.  The frontier is sharpened to ONE question: why would the lepton
  generation-ALLOCATION dark (alpha1^3, a budget shared across the (4,2,2) spinor states)
  be additive, when the heavy-quark single-channel dark (alpha1) is multiplicative?
  No fit was made; the number was never in doubt; the operator-forcing remains open.""")
    print("=" * 78)
