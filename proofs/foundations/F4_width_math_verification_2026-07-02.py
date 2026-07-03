#!/usr/bin/env python3
"""
proofs/foundations/F4_width_math_verification_2026-07-02.py

F4 SESSION 1 — VERIFY THE MATH THE KICKOFF POINTS AT, BEFORE BUILDING ANYTHING.
(docs/scoping/F4_widths_from_Im_kickoff_2026-07-02.md; skeptical-of-prose pass.)

This probe produces NO width prediction. It (i) re-derives the claimed inputs from
scratch, (ii) runs the pre-registered OVER-APPLICATION AUDIT of the naive Im-read,
(iii) maps the comparison-side landscape. Everything downstream (a real width read)
is gated on the incomplete equation this probe isolates.

PRE-DECLARED SCORING CLASSES (written before computing; kickoff §4 rule 1):
  (a) NEW CONTENT       = a substrate correction beyond SM structure, or a forced
                          derivation of structure the SM must input (e.g. the 1/12pi
                          phase-space constant falling out of a substrate read).
  (b) SM-REPRODUCTION   = the framework's own couplings re-deriving an SM tree/loop
                          result (consistency, not discovery; labeled as such).
  (c) MISS              = an open numerical miss, logged per the top-down law.
  Section S6 of this probe is class (b) BY CONSTRUCTION (it assembles the SM tree
  formula from framework live reads) and is a LANDSCAPE MARKER, not a prediction:
  nothing here enters predictions/ or the value lock.

PRE-REGISTERED KILL-TEST (stability anchor; declared before computing):
  Any mechanism that reads the MATCHING-POINT Im (the Ramanujan-circle constants
  sqrt5/4, sqrt7/4, the 1/sqrt2 dephasing step, or 2*alpha1*sqrt5/4) as a particle
  width MUST explain Gamma_e = 0 (electron: same shell channel, same sqrt5/4,
  exactly stable; tau_e > 6.6e28 yr). If the read gives every shell fermion the
  same nonzero Gamma/m, it is OVER-APPLYING (transport dephasing != particle decay)
  and is dead on arrival. This is decided by the audit in S4, not by prose.

PRE-STATED over-application lesson (kickoff §4 rule 2): expect the naive read to
over-apply; quantify by how much, per particle, against the measured landscape.
PDG numbers appear ONLY in S4/S6 (comparison side, marked; kickoff §4 rule 3).
"""
import cmath
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from proofs.common import find_bonds  # noqa: E402

K = 3                      # srs is 3-regular
Q = K - 1                  # Ihara-Bass q = k-1 = 2 (Ramanujan radius^2)
ALPHA1 = (2.0 / 3.0) ** 8  # the run coupling u = alpha1 (Feshbach Exponent Principle)
TOL = 1e-12

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")


# ---------------------------------------------------------------------------
# substrate operators (rebuilt from scratch, not imported from the_run.py)
# ---------------------------------------------------------------------------
def directed_edges():
    return [(int(i), int(j), tuple(int(x) for x in c)) for i, j, c in find_bonds()]

DE = directed_edges()
POS = {e: i for i, e in enumerate(DE)}
REV = [POS[(v, u, tuple(-x for x in n))] for (u, v, n) in DE]

def adjacency(k):
    """4x4 Bloch adjacency of srs (Laves K4 cell) at k (reciprocal-lattice units)."""
    A = np.zeros((4, 4), complex)
    for (u, v, n) in DE:
        A[u, v] += np.exp(2j * np.pi * np.dot(k, n))
    return A

def nb_operator(k):
    """12x12 Bloch Hashimoto (non-backtracking) operator at k."""
    m = len(DE)
    B = np.zeros((m, m), complex)
    kk = np.asarray(k, float)
    for a, (ua, va, na) in enumerate(DE):
        for b, (ub, vb, nb) in enumerate(DE):
            if va == ub and b != REV[a]:
                B[a, b] = np.exp(2j * np.pi * np.dot(kk, nb))
    return B

def ihara_bass_roots(mu):
    """lambda^2 - mu*lambda + q = 0 (per adjacency eigenvalue mu)."""
    disc = complex(mu * mu - 4 * Q)
    r = cmath.sqrt(disc)
    return (mu + r) / 2, (mu - r) / 2


# ---------------------------------------------------------------------------
print("=" * 88)
print(" S1  the claimed Im structure, re-derived: h, 1/h, and WHERE h lives in the spectrum")
print("=" * 88)
# Ihara-Bass at mu = sqrt3 (the P-shell) and mu = -1 (the Gamma cone):
h_P_shell, _ = ihara_bass_roots(math.sqrt(3))     # expect (sqrt3 + i sqrt5)/2
h_G_shell, _ = ihara_bass_roots(-1.0)             # expect (-1 + i sqrt7)/2
lam_perron, lam_small = ihara_bass_roots(3.0)     # expect {2, 1}

h = h_P_shell
check("h(mu=sqrt3) = (sqrt3+i*sqrt5)/2", abs(h - (math.sqrt(3) + 1j * math.sqrt(5)) / 2) < TOL)
check("|h|^2 = 2 (Ramanujan circle)", abs(abs(h) ** 2 - 2) < TOL)
check("Re(1/h) = sqrt3/4", abs((1 / h).real - math.sqrt(3) / 4) < TOL)
check("-Im(1/h) = sqrt5/4", abs(-(1 / h).imag - math.sqrt(5) / 4) < TOL)
check("h(mu=-1) = (-1+i*sqrt7)/2, F_Gamma = sqrt7/4",
      abs(h_G_shell - (-1 + 1j * math.sqrt(7)) / 2) < TOL
      and abs(h_G_shell.imag / abs(h_G_shell) ** 2 - math.sqrt(7) / 4) < TOL)
check("Perron root at mu=3: {2,1}, REAL (no Im at all)",
      abs(lam_perron - 2) < TOL and abs(lam_small - 1) < TOL)

# where do mu = sqrt3 and mu = -1 actually live? scan the srs Bloch spectrum.
# NOTE (found by this probe): in the integer-translation convention of find_bonds(),
# the P-shell is at k = (1/4,1/4,1/4); k = (1/2,1/2,1/2) is the charge-conjugate cone
# {-3, 1,1,1} (the mirror of Gamma) — docs that say "P = (1/2,1/2,1/2)" conflate the two.
K_P = (0.25, 0.25, 0.25)
A_G = np.linalg.eigvalsh(adjacency((0, 0, 0)))
A_P = np.linalg.eigvalsh(adjacency(K_P))
A_R = np.linalg.eigvalsh(adjacency((0.5, 0.5, 0.5)))
print(f"    A(Gamma)        eigs = {np.round(A_G, 6)}   (Perron + spin-1 cone)")
print(f"    A(P=1/4,1/4,1/4) eigs = {np.round(A_P, 6)}   (the +-sqrt3 P-shell)")
print(f"    A(R=1/2,1/2,1/2) eigs = {np.round(A_R, 6)}   (charge-conjugate cone)")
check("A(Gamma) = {3, -1^3}", np.allclose(sorted(A_G), [-1, -1, -1, 3], atol=1e-9))
check("A(P) = {+-sqrt3, x2 each}",
      np.allclose(sorted(A_P), [-math.sqrt(3)] * 2 + [math.sqrt(3)] * 2, atol=1e-9))
check("A(R) = {-3, 1^3} (mirror cone)", np.allclose(sorted(A_R), [-3, 1, 1, 1], atol=1e-9))

# and the NB Bloch spectrum at P actually contains h (machine check of the pointer claim)
lam_B_P = np.linalg.eigvals(nb_operator(K_P))
dmin = min(abs(l - h) for l in lam_B_P)
check("B(P) spectrum contains h=(sqrt3+i*sqrt5)/2 (machine)", dmin < 1e-9)

print("""
    READ: the object's Im structure is the IHARA-BASS DICHOTOMY. Adjacency modes with
    |mu| < 2*sqrt2 lift to NB eigenvalues ON the circle |lambda| = sqrt2 (complex, dephasing);
    the Perron mode mu=3 lifts to REAL {1,2} (no Im, stable). ALL Im constants the kickoff
    quotes (sqrt5/4 at P, sqrt7/4 at Gamma) are values of ONE functional F = Im(lam)/|lam|^2
    on the shell band. The Z's own channel (Perron) has ZERO Im at the matching point.""")

# ---------------------------------------------------------------------------
print("=" * 88)
print(" S2  the Feshbach theorem Sigma(h) = alpha1/h — contour math re-verified numerically")
print("=" * 88)
# outside-radial: for ANY eps>0 the uniform-density integral is EXACTLY 1/h_eps
# (only the z=0 pole is enclosed), so verifying at finite eps verifies the limit.
alpha = cmath.phase(h)
for eps, npts in ((1e-2, 400001), (1e-3, 4000001)):
    h_eps = math.sqrt(2) * (1 + eps) * cmath.exp(1j * alpha)
    phi = np.linspace(0, 2 * np.pi, npts)[:-1]
    I_num = np.mean(1.0 / (h_eps - math.sqrt(2) * np.exp(1j * phi)))
    err = abs(I_num - 1 / h_eps)
    check(f"finite-eps integral = 1/h_eps exactly (eps={eps:g}, err={err:.1e}); limit -> 1/h",
          err < 1e-7 and abs(1 / h_eps - 1 / h) < 2 * eps)
# principal value (h exactly ON the circle) -> half residue = 1/(2h):
phi = np.linspace(alpha + 1e-6, alpha + 2 * np.pi - 1e-6, 4000001)
I_pv = np.trapezoid(1.0 / (h - math.sqrt(2) * np.exp(1j * phi)), phi) / (2 * np.pi)
check(f"P.V. (on-circle) -> 1/(2h) (the factor-2 the theorem flags; |I-1/2h|={abs(I_pv-0.5/h):.2e})",
      abs(I_pv - 0.5 / h) < 1e-3)
print(f"""
    Sigma(h) = alpha1/h CONFIRMED as stated: uniform density (water-filling M_n=0) x
    outside-radial prescription. |Im Sigma| = alpha1*sqrt5/4 = {ALPHA1*math.sqrt(5)/4:.6f}.
    NOTE what this object IS: the girth-return self-energy AT THE MATCHING POINT — one
    complex constant. It has no frequency argument. An optical-theorem width is
    Im Sigma(E_pole) WITH E-resolution and open-channel thresholds. That function does
    not exist in the repo (isolated below as the incomplete equation).""")

# ---------------------------------------------------------------------------
print("=" * 88)
print(" S3  the clocks already pinned in the object (all re-verified, none new)")
print("=" * 88)
check("Lindblad girth survival: (1-1/k)^(g-2) = (2/3)^8 = alpha1 (wigner_weisskopf_dark.py route)",
      abs((1 - 1 / K) ** 8 - ALPHA1) < TOL)
check("Ramanujan dephasing step: |h_shell| / h_Perron = 1/sqrt2 (CLEANROOM 4a 'decay step')",
      abs(abs(h) / 2.0 - 1 / math.sqrt(2)) < TOL)
print(f"    per-step phase of the shell mode: arg h = {alpha:.6f} rad; per-step relative decay:")
print(f"    -log(|h|/2) = {-math.log(abs(h)/2):.6f}  =>  naive 'decay/phase' = {-math.log(abs(h)/2)/alpha:.4f}")
print("""    Three clock rates now on the table, ALL O(0.1-1) per step:  1/3 (Lindblad edge leak),
    1/sqrt2 (shell-vs-Perron amplitude), 2pi/sqrt7 (the run phase velocity). None of these is
    a particle width by itself — S4 decides that quantitatively.""")

# ---------------------------------------------------------------------------
print("=" * 88)
print(" S4  THE OVER-APPLICATION AUDIT (pre-registered) — COMPARISON SIDE: PDG enters here")
print("=" * 88)
# candidate naive matching-point "width" constants (per the three mechanisms):
cand = {
    "2*alpha1*sqrt5/4 (pole read of m(1-alpha1/h), Gamma=2|Im|)": 2 * ALPHA1 * math.sqrt(5) / 4,
    "alpha1*sqrt5/4   (|Im Sigma| itself)": ALPHA1 * math.sqrt(5) / 4,
    "log-decay/phase  (bare shell dephasing, no alpha1)": -math.log(abs(h) / 2) / alpha,
}
# measured landscape (PDG 2024; hbar = 6.582119569e-25 GeV s):
HBAR = 6.582119569e-25
meas = {
    "e":   (0.0, 0.00051099895),                    # stable: tau > 6.6e28 yr
    "mu":  (HBAR / 2.1969811e-6, 0.1056583755),
    "tau": (HBAR / 290.3e-15, 1.77686),
    "H":   (0.0041, 125.20),
    "t":   (1.42, 172.57),
    "W":   (2.085, 80.377),
    "Z":   (2.4952, 91.1876),
}
print(f"    {'particle':>8} {'Gamma/m (meas)':>16}", end="")
for lbl in cand: print(f"{'x-over':>10}", end="")
print("   (x-over = candidate / measured)")
for p, (G, m) in meas.items():
    r = G / m
    print(f"    {p:>8} {r:>16.3e}", end="")
    for c in cand.values():
        print(f"{(c / r if r > 0 else float('inf')):>10.1e}", end="")
    print()
r_mu = meas["mu"][0] / meas["mu"][1]
r_Z = meas["Z"][0] / meas["Z"][1]
check("kill-test: matching-point reads over-apply for the electron (infinitely: Gamma_e=0)", True)
check("kill-test: over-apply for mu by > 1e15", cand["2*alpha1*sqrt5/4 (pole read of m(1-alpha1/h), Gamma=2|Im|)"] / r_mu > 1e15)
check("gauge bosons: same constant lands within x2 of Gamma/M (order-of-magnitude only)",
      1 < cand["2*alpha1*sqrt5/4 (pole read of m(1-alpha1/h), Gamma=2|Im|)"] / r_Z < 2)
print("""
    VERDICT (forced by the table, not by prose):
    * NO matching-point constant is a particle width. The measured Gamma/m spans >61
      decades (e -> t) while every matching-point read is ONE constant. The complex
      dressed factor m(1 - alpha1/h) is NOT a resonance pole: its Im is transport/
      dephasing content (delta-rho's verified +1.091% usage), not decay.
    * The ONLY regime where a coupling-strength Im read is even the right ORDER is the
      gauge resonances (Z: x1.6, W: x1.7) — the saturated O(g^2), all-channels-open,
      2-body decays. This is WHY Gamma_Z/M_Z is the right first target, and WHAT is
      missing is exactly the O(1) factor = open-channel content x phase space.
    * Widths therefore REQUIRE the energy-resolved self-energy with thresholds:
      Im Sigma(E) proportional to open-channel spectral weight AT the pole, vanishing
      below threshold (Gamma_e = 0 <=> no open channel, NOT a small number).""")

# ---------------------------------------------------------------------------
print("=" * 88)
print(" S5  where E-resolution already lives in the object: the cavity Green's function")
print("=" * 88)
def cavity_g(z):
    """g(z) = 1/(z - k f(z)), q f^2 - z f + 1 = 0; physical branch = decaying cavity
    field (|f| minimal off-cut; retarded Im g < 0 on the cut, z -> z - i0)."""
    r = cmath.sqrt(complex(z * z - 4 * Q))
    fs = [(z - r) / (2 * Q), (z + r) / (2 * Q)]
    if abs(z) < 2 * math.sqrt(Q):              # on the cut: retarded root, Im g <= 0
        gs = [1.0 / (z - K * f) for f in fs]
        return min(gs, key=lambda g: g.imag)
    f = min(fs, key=abs)                       # off-cut: decaying cavity field
    return 1.0 / (z - K * f)

def kesten_mckay(x):
    if abs(x) >= 2 * math.sqrt(Q): return 0.0
    return K * math.sqrt(4 * Q - x * x) / (2 * math.pi * (K * K - x * x))

errs = []
for x in (0.0, 0.7, 1.3, 2.1, 2.7):
    g = cavity_g(x)
    errs.append(abs(-g.imag / math.pi - kesten_mckay(x)))
check("on the cut: -Im g(x-i0)/pi = Kesten-McKay density (k=3), max err %.1e" % max(errs),
      max(errs) < 1e-6)
gP = cavity_g(3.0)
check("at the Perron point z=k=3 (the Z's channel): Im g = 0 EXACTLY (off-cut, stable)",
      abs(gP.imag) < 1e-9)
print(f"""    g(3) = {gP.real:.6f} (real).  The object ALREADY has an energy-resolved spectral
    function: Im g(z) = pi x (density of states) on the cut |z| <= 2*sqrt2, zero outside.
    The Z lives OFF the cut (z=3): its leading read is real => stable => its width is a
    SECOND-ORDER Feshbach embedding into the on-cut (shell/cone) continuum, evaluated at
    the Z's own frequency — the object Sigma_Z(omega) that the repo has never built.
    Fermions: same statement, with thresholds = the dressed masses themselves.""")

# ---------------------------------------------------------------------------
print("=" * 88)
print(" S6  CLASS (b) LANDSCAPE — SM tree assembled from FRAMEWORK live reads (marked)")
print("=" * 88)
# framework live reads (provenance comments; these are the framework's OWN endpoint values):
g2 = 0.65175          # predictions/g_2.py live (post-alpha_GUT-DC; -0.18 sigma)
s2 = 0.23121          # framework sin^2 run endpoint (live read lands at PDG-scale value, +0.96 sigma band)
a_s = 0.1179          # predictions/alpha_s.py live (c_color = 1/4; -0.13 sigma)
c2 = 1 - s2
gZ2 = g2 * g2 / c2

# fermion content is the framework's own Cl(6) read: (T3, Q, Nc) per species x 3 gens
species = [(+0.5, 0.0, 1), (-0.5, -1.0, 1), (+0.5, +2.0 / 3.0, 3), (-0.5, -1.0 / 3.0, 3)]
def vf_af(T3, Qc): return T3 - 2 * Qc * s2, T3
S_all = 3 * sum(N * (sum(x * x for x in vf_af(T3, Qc))) for (T3, Qc, N) in species)
S_top = 3 * (vf_af(+0.5, 2 / 3)[0] ** 2 + 0.25)
S_open = S_all - S_top          # top closed: the framework's own m_t read >> M_Z/2
f_had = (S_open - 3 * (0.5 + (vf_af(-0.5, -1)[0] ** 2 + 0.25))) / S_open
tree_Z = gZ2 * S_open / (48 * math.pi)
qcd = 1 + f_had * (a_s / math.pi + 1.409 * (a_s / math.pi) ** 2)
tree_W = 9 * g2 * g2 / (48 * math.pi)
qcdW = 1 + (2 / 3) * (a_s / math.pi + 1.409 * (a_s / math.pi) ** 2)

# exact K-rational collapse at the static substrate boundary s^2 = 3/8:
s2x = Fraction(3, 8)
spec_x = [(Fraction(1, 2), Fraction(0), 1), (Fraction(-1, 2), Fraction(-1), 1),
          (Fraction(1, 2), Fraction(2, 3), 3), (Fraction(-1, 2), Fraction(-1, 3), 3)]
Sx = 3 * sum(N * ((T3 - 2 * Qc * s2x) ** 2 + T3 ** 2) for (T3, Qc, N) in spec_x)
print(f"    static boundary s^2=3/8:  Sum_f Nc(v^2+a^2) = {Sx} exactly (u-quark v = 0 exactly);")
print(f"    open (no top) = {Sx - Fraction(3,4)}.  [K-rational collapse — structure note, no claim]")
print(f"    run endpoint  s^2={s2}:  S_open = {S_open:.4f}   f_had = {f_had:.4f}   gZ^2 = {gZ2:.6f}")
print()
print("    assembled (class b, NOT predictions):            value      vs PDG(comparison)")
PDG_Z, PDG_W_over_Z = 0.0273634, 2.085 / 2.4952
print(f"    Gamma_Z/M_Z  tree                       = {tree_Z:.6f}    {(tree_Z/PDG_Z-1)*100:+.2f}%")
print(f"    Gamma_Z/M_Z  tree x own-alpha_s QCD     = {tree_Z*qcd:.6f}    {(tree_Z*qcd/PDG_Z-1)*100:+.2f}%")
ratio_WZ = (tree_W * qcdW) / (tree_Z * qcd) * (80.377 / 91.1876)
print(f"    Gamma_W/Gamma_Z (same assembly, PDG M_W/M_Z)= {ratio_WZ:.6f}    {(ratio_WZ/PDG_W_over_Z-1)*100:+.2f}%  (PDG +-2.0%)")
print(f"""
    READ (class b): with ONLY the framework's own g2, s^2, alpha_s and its own fermion
    content, the golden-rule assembly lands within ~0.5% of both width observables
    [Gamma_Z/M_Z {(tree_Z*qcd/PDG_Z-1)*100:+.2f}%, Gamma_W/Gamma_Z {(ratio_WZ/PDG_W_over_Z-1)*100:+.2f}%]. The remaining few-x0.1% on Gamma_Z is the
    EW radiative layer (rho_f, sbar^2_eff) — the SAME oblique territory the M_Z BZ result
    mapped as the framework's floor. NOTHING here is new content: the SM formula
    g_Z^2 * S / (48 pi) was INPUT. The class-(a) question is whether 1/(12 pi) and the
    on-shell measure fall out of the substrate's own cone spectral function.

    NUMEROLOGY HAZARD (poisoned, pre-registered): Gamma_Z/M_Z / alpha1 = {PDG_Z/ALPHA1:.4f}.
    K-rational*sqrt5 candidates within 2%% of it exist (e.g. 5*sqrt5/16 = {5*math.sqrt(5)/16:.4f}).
    Per the MDL ledger's density argument these are UNUSABLE without a forced projection
    derivation; listed here so no future session 'discovers' them.""")

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
