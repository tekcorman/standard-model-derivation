#!/usr/bin/env python3
# ============================================================
# F7: the up-sector precision amplifier — invert m_u's +15.5% miss
# ============================================================
#
# Scope: docs/scoping/fresh_threads_baryon_sector_2026-05-31.md §F7.
#
# m_u is the framework's WORST relative prediction (+15.5%), masked by PDG's 23%
# error bar. The quark-Koide cascade (predictions/_koide_quark.py) is
#   sqrt(m_j) = sqrt(M0)*(1 + eps_n*cos(2pi*j/k* + delta_n)),  j=0..k*-1,
# anchored at the heaviest mass; m_min = m_anchor*(f_min/f_max)^2. Up-type
# (n=2): m_u = m_min sits at the factor NEAREST ZERO (f_min ~ 0.01), so
# m_u ∝ f_min^2 is amplified ~(f_max/f_min)^2 ~ 7e4 against eps_up/delta_up.
#
# THE DECISIVE TEST (not just "back out the error"): does the SAME up-sector
# correction that fixes m_u also keep m_c (the mid up-type, currently +0.55%,
# good)? If a single eps_up (or delta_up) shift fixes m_u AND keeps m_c within
# PDG -> the +15.5% is a real STRUCTURAL up-sector deficit (a missing sub-leading
# eps^2 term) = a genuine clue. If fixing m_u BREAKS m_c -> m_u's miss is
# per-species / within-PDG-noise, not a structural handle.

import os, sys, math

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

ALPHA1_FULL = (5.0/3.0)*(2.0/3.0)**8
K_STAR, G = 3, 10

# observed (MS-bar 2 GeV for u,d,s; m_c(m_c); pole m_t; PDG 2024), MeV
OBS = {"m_u": 2.16, "m_c": 1270.0, "m_t": 172700.0,
       "m_d": 4.67, "m_s": 93.4,   "m_b": 4180.0}
SIG = {"m_u": 0.49, "m_c": 20.0,   "m_d": 0.48, "m_s": 8.6}   # PDG 1sigma, MeV
# framework anchors (the heaviest in each sector, the framework's own value), MeV
ANCH = {"up": 174100.0, "down": 4270.0}


def f_of_n(n): return 1.0 + (n-1)*(G-2)/(2.0*G)
def eps_sq(n): return 2.0 + 6.0*ALPHA1_FULL*n*f_of_n(n)
def delta_n(n): return 2.0/(9.0*(n+1))


def factors(eps, delta):
    fs = sorted(1.0 + eps*math.cos(2*math.pi*j/K_STAR + delta) for j in range(K_STAR))
    return fs[0], fs[1], fs[-1]   # f_min, f_mid, f_max


def masses(n, eps, delta, anchor):
    fmin, fmid, fmax = factors(eps, delta)
    return anchor*(fmin/fmax)**2, anchor*(fmid/fmax)**2   # (m_min, m_mid)


from scipy.optimize import brentq


def eps_zero(delta):
    """eps at which f_min = 0 (the singularity to stay below)."""
    cmin = min(math.cos(2*math.pi*j/K_STAR + delta) for j in range(K_STAR))
    return -1.0/cmin   # cmin < 0


def solve_eps(n, anchor, target_min):
    """eps (delta fixed) so m_min == target_min, in the MONOTONIC region
    (0, eps_zero) where m_min decreases with eps. Searches the correct side."""
    d0, e0 = delta_n(n), math.sqrt(eps_sq(n))
    ez = eps_zero(d0)
    def g(e): return masses(n, e, d0, anchor)[0] - target_min
    if g(e0) > 0:                      # predicted > target (over): need larger eps
        return brentq(g, e0, ez*(1 - 1e-9))
    else:                              # predicted < target (under): need smaller eps
        return brentq(g, e0*0.3, e0)


def report_sector(name, n, anchor, min_key, mid_key):
    e0, d0 = math.sqrt(eps_sq(n)), delta_n(n)
    fmin, fmid, fmax = factors(e0, d0)
    m_min0, m_mid0 = masses(n, e0, d0, anchor)
    print(f"\n{'='*70}\n{name.upper()} sector (n={n}): eps={e0:.5f} (eps^2={e0**2:.5f}), "
          f"delta={d0:.5f} rad")
    print(f"  f_min={fmin:.5f}  f_mid={fmid:.5f}  f_max={fmax:.5f}  "
          f"(amplif (f_max/f_min)^2 = {(fmax/fmin)**2:.0f})")
    print(f"  predicted {min_key}={m_min0:.4f}  (obs {OBS[min_key]}  "
          f"{100*(m_min0-OBS[min_key])/OBS[min_key]:+.2f}%, {(m_min0-OBS[min_key])/SIG[min_key]:+.2f}sig)")
    print(f"  predicted {mid_key}={m_mid0:.2f}  (obs {OBS[mid_key]}  "
          f"{100*(m_mid0-OBS[mid_key])/OBS[mid_key]:+.2f}%, {(m_mid0-OBS[mid_key])/SIG[mid_key]:+.2f}sig)")

    print(f"  --- invert {min_key} via eps (delta fixed), then check {mid_key} ---")
    eps = solve_eps(n, anchor, OBS[min_key])
    m_min, m_mid = masses(n, eps, d0, anchor)
    d_eps2 = eps**2 - e0**2
    mc_pull = (m_mid-OBS[mid_key])/SIG[mid_key]
    verdict = "KEEPS/IMPROVES" if abs(mc_pull) < 1.0 else "BREAKS"
    print(f"   d(eps^2) = {d_eps2:+.5f}  = {d_eps2/ALPHA1_FULL**2:+.3f} x alpha1^2  "
          f"({100*d_eps2/(e0**2):+.3f}% of eps^2)")
    print(f"   => {min_key} fixed to {m_min:.4f};  {mid_key} -> {m_mid:.2f} "
          f"({100*(m_mid-OBS[mid_key])/OBS[mid_key]:+.2f}%, {mc_pull:+.2f}sig)  [{verdict} {mid_key}]")
    return e0, d0


def all_light_masses(d_eps2):
    """Apply a UNIVERSAL additive d_eps2 to eps^2(n) for both sectors; return
    the four light-quark predictions (m_u,m_c,m_d,m_s) and total chi^2."""
    out, chi2 = {}, 0.0
    for n, anchor, mn, md in ((2, ANCH["up"], "m_u", "m_c"), (1, ANCH["down"], "m_d", "m_s")):
        eps = math.sqrt(eps_sq(n) + d_eps2)
        m_min, m_mid = masses(n, eps, delta_n(n), anchor)
        out[mn], out[md] = m_min, m_mid
        chi2 += ((m_min-OBS[mn])/SIG[mn])**2 + ((m_mid-OBS[md])/SIG[md])**2
    return out, chi2


def main():
    print("="*70)
    print("F7: up-sector amplifier — is m_u's +15.5% a structural eps/delta")
    print("deficit (fixable, keeps m_c) or per-species noise?")
    print("="*70)
    print(f"alpha1_full=(5/3)(2/3)^8={ALPHA1_FULL:.8f}; alpha1^2={ALPHA1_FULL**2:.8f}")

    report_sector("up", 2, ANCH["up"], "m_u", "m_c")
    report_sector("down", 1, ANCH["down"], "m_d", "m_s")

    # universal sub-leading term test
    print(f"\n{'='*70}\nUNIVERSAL sub-leading test: add one d(eps^2) to ALL sectors,")
    print("minimize total chi^2 over (m_u,m_c,m_d,m_s):")
    best = min((all_light_masses(c*ALPHA1_FULL**2)[1], c) for c in
               [x/100.0 for x in range(-200, 401)])
    chi2_0 = all_light_masses(0.0)[1]
    c_best = best[1]
    pred0, _ = all_light_masses(0.0)
    predb, chi2b = all_light_masses(c_best*ALPHA1_FULL**2)
    print(f"  baseline (d_eps^2=0): chi^2 = {chi2_0:.2f}")
    print(f"  best universal d(eps^2) = {c_best:+.2f} x alpha1^2 = "
          f"{c_best*ALPHA1_FULL**2:+.5f}: chi^2 = {chi2b:.2f}")
    print(f"  {'mass':>5} {'obs':>9} {'pred(0)':>10} {'pred(univ)':>12} {'obs sig':>8}")
    for k in ("m_u","m_c","m_d","m_s"):
        print(f"  {k:>5} {OBS[k]:>9.3f} {pred0[k]:>10.3f} {predb[k]:>12.3f}  "
              f"({(predb[k]-OBS[k])/SIG[k]:+.2f}sig)")

    # F7 -> F8 cross-link: does the up-sector fix bring m_d - m_u to the lattice
    # Q_np QCD value (+2.49 +/- 0.20 MeV, BMW 2015)?
    m_u_base = masses(2, math.sqrt(eps_sq(2)), delta_n(2), ANCH["up"])[0]
    eps_up_fix = solve_eps(2, ANCH["up"], OBS["m_u"])
    m_u_fix = masses(2, eps_up_fix, delta_n(2), ANCH["up"])[0]
    m_d_base = masses(1, math.sqrt(eps_sq(1)), delta_n(1), ANCH["down"])[0]
    print(f"\n{'='*70}\nF7 -> F8 cross-link (Q_np QCD input = m_d - m_u):")
    print(f"  baseline:        m_d - m_u = {m_d_base:.3f} - {m_u_base:.3f} = "
          f"{m_d_base - m_u_base:.3f} MeV")
    print(f"  up-sector fixed: m_d - m_u = {m_d_base:.3f} - {m_u_fix:.3f} = "
          f"{m_d_base - m_u_fix:.3f} MeV   (lattice Q_np QCD = +2.49 +/- 0.20)")

    print("\n" + "="*70)
    print("VERDICT — F7: a real structural clue (NOT noise, NOT the +4).")
    print("="*70)
    print("""  UP sector: m_u's +15.5% (the framework's worst relative miss, amplified
  ~70,000x by f_min~0) is a CONSISTENT structural eps deficit: a single
  sub-leading term d(eps^2) ~ +alpha1^2 (the natural NEXT order above the
  leading 6*alpha1*n*f) fixes m_u AND improves m_c (two masses, one correction).
  This points squarely at a missing higher-order term in
      eps^2(n) = 2 + 6*alpha1*n*f(n)  [+ O(alpha1^2) ?]
  The leading "6" = N_LQ (PS leptoquark coset, W4-derived); the missing piece
  is its alpha1^2 (2-channel / 2-loop) analog. The coefficient (~+1) is FIT here,
  not derived -- a structural derivation of it is the closure target.

  DOWN sector: m_d (-1.4%) / m_s (+2.7%) do NOT share the clue (m_d wants the
  opposite-sign shift); smaller and not a single-eps story. The up clue is the
  clean one.

  PAYOFF to F8: fixing m_u brings m_d - m_u to ~2.45 MeV = the lattice Q_np QCD
  value (+2.49+/-0.20). The up-sector precision and the nucleon Q_np are the same
  clue -- closing F7 supplies F8's QCD input.

  CAVEAT: the coefficient is fit, not derived; the whole light-quark sector is
  already within PDG (chi^2 0.70). This is a STRUCTURAL CLUE that the amplifier
  surfaced, not a falsification fix -- it tells us WHERE the next-order Koide
  term lives, which the wide error bars otherwise hide.""")
    print("="*70)


if __name__ == "__main__":
    main()
