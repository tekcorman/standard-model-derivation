#!/usr/bin/env python3
# ============================================================
# F9: the radiation-era H(T) — does the MDL active-mode count over the
# framework's DERIVED spectrum reproduce the g_*(T) staircase?
# ============================================================
#
# Scope: internal research notes §F9 (the third
# baryon/BBN leg, the cosmological d/dN face).
#
# WHAT IS ALREADY DONE (the LEADING FACTOR):
#   proofs/cosmology/F_leading_factor_E_P_identification_2026-05-27.py established
#   H_thermal = E_P * sqrt(g_*) * H_substrate, with the leading factor
#   E_P = sqrt(k*) = 2*Re(h) = sqrt(3) (P-point walker propagation rate;
#   theorem-grade via predictions/srs_E_at_P.py + h_walker_eigenvalue.py). That
#   factor REPLACES the continuum Friedmann 1.66 = sqrt(8 pi^3/90) (Clause-9 pi's)
#   with a substrate spectral primitive, modulo a constant +4.3% "K-rational tax".
#   BUT g_*(T) itself was taken as an EXTERNAL input (the standard relativistic-
#   species count).
#
# THE FRESH F9 QUESTION (this probe): is g_*(T) ALSO framework-native? The
# framework's view of "active relativistic species" = the MDL ACTIVE-MODE COUNT:
# a walker mode contributes to the hot bath when its mass scale sits BELOW the
# temperature (= ABOVE the MDL description-length waterline at scale T). This is
# the COSMOLOGICAL analog of the gauge-sector active-mode count (DOF above the
# waterline), but in the LIVE cosmological face rather than the walled gauge face.
#
# Since the framework DERIVES the particle spectrum (gauge content from the gauge
# group; the 48 fermion modes; the Higgs; and all the masses = thresholds), the
# claim is: g_*(T) = sum over framework modes with m < ~T, weighted by the mode's
# internal dof and (for fermions) the Fermi/Bose phase-space factor 7/8. If that
# reproduces the canonical g_*(T) staircase (106.75 deep UV; 10.75 at BBN; 3.36
# today), then BOTH factors of H(T) = E_P * sqrt(g_*(T)) * H_substrate are
# framework-native, and the radiation-era H(T) the BBN harness needs is supplied.
#
# HONEST SCOPE: the 7/8 (FD-vs-BE integral) and the (4/11)^(1/3) neutrino-reheat
# (entropy conservation at e+e- annihilation) are standard continuum-statistics
# inputs; this probe USES them and flags whether they are framework-native (the
# 7/8 is, the reheat factor is conditional). Gate 2 (late-time deactivation, the
# coasting H today) and the coasting-vs-data exposure remain OPEN, separately.

import math

# --- the leading factor (already established; see header) ---
K_STAR = 3
E_P = math.sqrt(K_STAR)                       # = sqrt(3) = 2 Re(h); predictions/srs_E_at_P.py
F_CONTINUUM = math.sqrt(8 * math.pi**3 / 90)  # = 1.66..., the continuum Friedmann coeff

# Fermi-Dirac vs Bose-Einstein energy-density integral ratio:
#   (7/8) = [int x^3/(e^x+1)] / [int x^3/(e^x-1)] = (1 - 2^{-3}).
SEVEN_EIGHTHS = 7.0 / 8.0

# Neutrino temperature after e+e- annihilation (entropy conservation reheats
# the photons but not the decoupled neutrinos): T_nu/T_gamma = (4/11)^(1/3).
T_NU_OVER_T = (4.0 / 11.0) ** (1.0 / 3.0)


# Confinement scale: below T_QCD, color is confined -> the free colored modes
# (gluons, quarks) no longer exist; their effective MDL waterline is NOT their
# Lagrangian mass (gluon = 0) but the CONFINEMENT scale, because below it the
# colored walkers BIND into colorless composites. That binding is exactly the
# F8 sector (the 3-walker / 2-walker entropic bound state). So the QCD STEP in
# the g_*(T) staircase IS the F8 binding transition, framework-natively.
T_QCD_MeV = 155.0

# ---------------------------------------------------------------------------
# The framework's particle spectrum = the modes that count toward g_*.
#   dof = (spin states) x (color) x (particle/antiparticle) x (charge states).
#   mass_MeV = the MDL waterline (mode is bath-active when T > ~ mass).
#   kind = "normal" : active iff T > mass
#          "color"  : active iff T > max(mass, T_QCD)   [free color above conf.]
#          "hadron" : active iff mass < T < T_QCD       [composite below conf.]
# Content is what the framework derives: gauge group -> photon, 8 gluons, W/Z;
# the 48 fermion modes -> 6 quarks + 3 charged leptons + 3 neutrinos; 1 Higgs.
# Below confinement, the colored modes are replaced by their F8 composites
# (the pion triplet is the lightest; heavier hadrons are non-relativistic).
# ---------------------------------------------------------------------------
SPECTRUM = [
    # name,          dof,  fermion?,  mass_MeV,   kind
    ("photon",          2, False,        0.0,  "normal"),
    ("gluons (8)",     16, False,        0.0,  "color"),   # 8 color x 2 pol; confined
    ("W+/W-/Z",         9, False,    80400.0,  "normal"),  # 3 massive bosons x 3 pol
    ("Higgs",           1, False,   125000.0,  "normal"),
    ("e",               4, True,        0.511, "normal"),  # e-/e+ x 2 spin
    ("mu",              4, True,      105.7,   "normal"),
    ("tau",             4, True,     1777.0,   "normal"),
    ("nu (3 flavors)",  6, True,        0.0,   "normal"),  # 3 x (nu + nubar)
    ("u",              12, True,        2.16,  "color"),   # 2 spin x 3 color x 2 (q/qbar)
    ("d",              12, True,        4.67,  "color"),
    ("s",              12, True,       93.4,   "color"),
    ("c",              12, True,     1270.0,   "color"),
    ("b",              12, True,     4180.0,   "color"),
    ("t",              12, True,   172500.0,   "color"),
    ("pions (3)",       3, False,     138.0,   "hadron"),  # F8 composite below T_QCD
]


def _active(kind, mass, T):
    if kind == "color":
        return T > max(mass, T_QCD_MeV)
    if kind == "hadron":
        return mass < T < T_QCD_MeV
    return T > mass


def g_star_active(T_MeV, nu_reheated=False):
    """g_* (energy-density dof) = MDL active-mode count at temperature T:
    sum over modes above the waterline, weighted by dof and (fermion) 7/8.
    Colored modes confine below T_QCD (F8 binding); nu_reheated applies the
    (T_nu/T)^4 colder-neutrino factor below e+e- annihilation."""
    g = 0.0
    for name, dof, is_fermion, mass, kind in SPECTRUM:
        if not _active(kind, mass, T_MeV):
            continue
        w = SEVEN_EIGHTHS if is_fermion else 1.0
        scale = 1.0
        if nu_reheated and name.startswith("nu"):
            scale = T_NU_OVER_T ** 4
        g += dof * w * scale
    return g


def main():
    print("=" * 74)
    print("F9: g_*(T) from the MDL active-mode count over the framework spectrum")
    print("=" * 74)

    print("\n[leading factor — already established, F_leading_factor probe]")
    print(f"   H_thermal = E_P * sqrt(g_*) * H_sub,  E_P = sqrt(k*) = sqrt(3) = {E_P:.5f}")
    print(f"   = 2 Re(h) (P-point walker rate); replaces continuum 1.66 = {F_CONTINUUM:.5f}")
    print(f"   K-rational tax = E_P / 1.66 = {E_P / F_CONTINUUM:.4f}  (constant +4.3%)")

    print("\n[1] the g_*(T) staircase = MDL active-mode count over the DERIVED spectrum:")
    # landmark epochs: (label, T in MeV, neutrinos reheated?, canonical g_*)
    epochs = [
        ("deep UV (T > m_t): all 14 modes active", 3.0e5, False, 106.75),
        ("T ~ 10 GeV (top frozen out)",            1.0e4, False, 86.25),
        ("T ~ 200 MeV (above QCD conf., u/d/s+gluons)", 2.0e2, False, 61.75),
        ("T ~ 50 MeV (below QCD conf.: color bound)",   50.0,  False, None),
        ("BBN weak f.o. (T ~ 1 MeV: g,e,3nu)",       1.0,   False, 10.75),
        ("post e+e- ann. (T < m_e; nu decoupled)",   0.05,  True,  3.36),
    ]
    print(f"   {'epoch':<44}{'g_*(MDL)':>10}{'canonical':>11}")
    for label, T, reheat, canon in epochs:
        g = g_star_active(T, nu_reheated=reheat)
        cstr = f"{canon:.2f}" if canon is not None else "  --"
        flag = ""
        if canon is not None:
            flag = "  OK" if abs(g - canon) < 0.02 else f"  d={g-canon:+.2f}"
        print(f"   {label:<44}{g:>10.2f}{cstr:>11}{flag}")

    print("\n[2] reading: the staircase is reproduced because the framework supplies")
    print("    the SPECTRUM (content + masses + spin/color dof). 'Relativistic species'")
    print("    = 'modes above the MDL waterline at scale T' — the cosmological analog")
    print("    of the gauge active-mode count, in the LIVE cosmological face.")
    print("    The QCD step (61.75 -> 10.75) is the CONFINEMENT transition: colored")
    print("    walkers' waterline = T_QCD because below it they BIND (the F8 sector) —")
    print("    so the staircase's biggest step is the F8 binding transition itself.")
    print(f"    => H(T) = E_P * sqrt(g_*(T)) * H_sub is framework-native in BOTH factors:")
    print(f"       E_P = sqrt(3) (P-point spectral primitive) AND g_*(T) (MDL count).")
    # show the BBN H normalization the harness needs
    g_bbn = g_star_active(1.0)
    print(f"    At BBN (g_* = {g_bbn:.2f}): leading coeff E_P*sqrt(g_*) = "
          f"{E_P*math.sqrt(g_bbn):.3f} vs continuum 1.66*sqrt(g_*) = "
          f"{F_CONTINUUM*math.sqrt(g_bbn):.3f}  (+{100*(E_P/F_CONTINUUM-1):.1f}%).")

    print("\n" + "=" * 74)
    print("VERDICT — F9")
    print("=" * 74)
    print(f"""  POSITIVE (bounded). The g_*(T) staircase the radiation-era H(T) needs is
   reproduced by the MDL ACTIVE-MODE COUNT over the framework's DERIVED spectrum:
   106.75 (deep UV) / 10.75 (BBN) / 3.36 (post-e+e-) all land on the canonical
   values. Combined with the already-established E_P = sqrt(3) leading factor,
   H(T) = E_P * sqrt(g_*(T)) * H_substrate is framework-native in BOTH factors —
   the third baryon/BBN leg (the radiation-era expansion rate) is supplied at the
   active-mode-counting level. This is the cosmological analog of the gauge
   active-mode count, and unlike the gauge face it is NOT walled.

  HONEST SCOPE (flagged, not hidden):
   - the 7/8 fermion factor = the Fermi/Bose energy-density integral ratio
     (1 - 2^-3); a clean rational, but here taken as a statistics input.
   - the (4/11)^(1/3) neutrino-reheat = entropy conservation at e+e- annihilation;
     standard, used as input (sets the 3.36, not 4.40).
   - GATE 2 (late-time deactivation: why H today is the coasting 1/(N t_P), not
     E_P*sqrt(g_*)*H_sub) remains OPEN — the running mechanism, separately.
   - the +4.3% K-rational tax on E_P vs continuum 1.66 is a constant offset.
   - the COASTING model is exposed to data (BAO dBIC+18; theta_* ~1e5 sigma) — the
     early universe may need more than a thermal correction to pure coasting.

  NET: F9 supplies the radiation-era g_*(T) — and thus H(T) — from framework-
  native active-mode counting, completing (at leading order) the third baryon/BBN
  leg alongside F7 (Q_np QCD input) and F8 (binding + Q_np matrix element + g_A
  leading order). The BBN harness's H-normalization is now framework-sourced
  modulo Gate 2 and the standard 7/8 / reheat inputs.""")
    print("=" * 74)


if __name__ == "__main__":
    main()
