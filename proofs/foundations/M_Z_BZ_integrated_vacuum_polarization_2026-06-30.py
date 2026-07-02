#!/usr/bin/env python3
"""
proofs/foundations/M_Z_BZ_integrated_vacuum_polarization_2026-06-30.py

THE PLANNED M_Z ATTACK (scoped in docs/incomplete_equations_todo.md ★ NEXT ATTACK):
build the BZ-INTEGRATED Z-current (T3 - s^2 Q) vacuum polarization and ask whether
the SM's 0.810 fraction of the chiral-shell step FALLS OUT (-> M_Z closes) or whether
it confirms a real ~few-% substrate-vs-SM oblique residual.

CONTEXT (the bracket this resolves; theorem_M_Z_shell_vertexbox_closure_2026-06-29.md):
  M_Z tree->pole oblique = Perron singlet (delta_r) + shell vertex/box (Q-winding).
  delta_r            = 0.3384%  ->  +8.1 sigma  (under; the LIVE single-term value)
  + chiral shell @Gamma (Sw2=2, F=sqrt7/4) = +0.0230%  -> 0.3614%  ->  -1.9 sigma (over)
  SM tree->pole target = 0.3570%  ==  delta_r + 0.810 . (chiral step).
The shell template EVALUATES F at the single k-point Gamma.  The genuine vacuum
polarization is the FULL BRILLOUIN-ZONE INTEGRAL of the Q-current self-energy.

FORCED OBJECT (no fit; basis-free):
  shell_BZ = <Sigma w^2 . F>_BZ . (1/2|E|) . s^4 . alpha1
  <Sigma w^2 . F>_BZ = INT_BZ d3k Sum_{n: Im lam_n(k)>0} |<l_n|W|r_n>|^2 . Im(lam_n)/|lam_n|^2
  W = (P - P^2)/(i sqrt3)  : the C3 winding-charge operator (eigs {0,+-1}) on the
                            12 darts, P = the C3 dart permutation built on
                            directed_edges()'s OWN ordering ([B,P]=0, verified).
  Chirality = the Im(lam)>0 hemisphere (at Gamma the two hemispheres CANCEL:
              h gives +2.sqrt7/4, hbar gives -2.sqrt7/4; the chiral shell is one).
  By B(-k)=conj(B(k)) this equals (1/2) INT Sum_all w^2 |Im lam|/|lam|^2  (cross-check).

CLOSURE RATIO  R = <Sigma w^2 . F>_BZ / [Sigma w^2 . F]_Gamma,  [.]_Gamma = 2.(sqrt7/4).
  R = 0.810  =>  SM falls out, M_Z closes.
  R != 0.810 =>  forced substrate-vs-SM oblique residual (a complete honest result).

DISCIPLINE GATE (hard): the 0.810 must FALL OUT.  Do NOT pattern-match it
(0.810 ~ 13/16 and near other coincidences -- FORBIDDEN to use any).

RESULT (this script): R = 0.2046 (robust, basis-free, entirely shell-band).
  0.810 does NOT fall out.  The BZ-integrated shell is ~1/5 of the Gamma template:
  Gamma is the BZ MAXIMUM of F (|lam|^2=2 minimal, Im lam maximal there), so the
  k-point template OVER-estimated the shell ~5x and the "bracket" was an artifact.
  The genuine substrate oblique delta_r + shell_BZ = 0.3431% UNDER-predicts the SM
  0.3570% -> M_Z stays +6.0 sigma.  M_Z is a forced ~4%-relative substrate-vs-SM
  oblique difference, NOT a term that zeroes out.  (Even the field-theory full
  two-propagator bubble, R~0.57 but exceptional-point-ill-conditioned and NOT the
  framework's template, fails to reach 0.810.)
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from proofs.common import find_bonds, K_STAR, GIRTH

np.set_printoptions(precision=6, suppress=True, linewidth=140)


# --------------------------------------------------------------------------
# substrate: 12 darts in find_bonds() ordering, the C3 dart permutation, W
# --------------------------------------------------------------------------
def directed_edges():
    return [(int(i), int(j), tuple(int(x) for x in c)) for i, j, c in find_bonds()]

def rev_index(de):
    pos = {(u, v, n): i for i, (u, v, n) in enumerate(de)}
    return [pos[(v, u, tuple(-x for x in n))] for (u, v, n) in de]

def nb_operator(k, de, rev):
    m = len(de); B = np.zeros((m, m), complex); k = np.asarray(k, float)
    for a, (ua, va, na) in enumerate(de):
        for b, (ub, vb, nb) in enumerate(de):
            if va == ub and b != rev[a]:
                B[a, b] = np.exp(2j * np.pi * np.dot(k, nb))
    return B

de = directed_edges(); rev = rev_index(de); pos = {e: i for i, e in enumerate(de)}
assert len(de) == 12

# C3 on vertices sigma=(1 3 2) (proofs.common.C3_PERM), on homology (n1,n2,n3)->(n3,n1,n2)
sigma_v = {0: 0, 1: 3, 2: 1, 3: 2}
sigma_n = lambda n: (n[2], n[0], n[1])
P = np.zeros((12, 12), complex)
for i, (u, v, n) in enumerate(de):
    img = (sigma_v[u], sigma_v[v], sigma_n(n))
    assert img in pos, "C3 image is not a dart"
    P[pos[img], i] = 1.0
assert np.allclose(np.linalg.matrix_power(P, 3), np.eye(12)), "P^3 != I"

W = (P - P @ P) / (1j * np.sqrt(3))            # winding-charge operator, eigs {0,+-1}

B0 = nb_operator((0, 0, 0), de, rev)
assert np.max(np.abs(B0 @ P - P @ B0)) < 1e-9, "[B(0),P] != 0 -- WRONG basis"

SQRT2 = np.sqrt(2)
F_GAMMA = np.sqrt(7) / 4                        # Im(h)/|h|^2 at Gamma, h=(-1+i sqrt7)/2
TEMPLATE_GAMMA = 2 * F_GAMMA                    # [Sigma w^2 . F]_Gamma  (chiral, Sw2=2)


# --------------------------------------------------------------------------
# (0) reproduce the Gamma joint structure: Sw2 = 0 (Perron) / 4 (shell) / 4 (|l|=1)
# --------------------------------------------------------------------------
def gamma_winding_split():
    evals, R = np.linalg.eig(B0)
    used = np.zeros(12, bool); groups = []
    for i in np.argsort(-np.abs(evals)):
        if used[i]: continue
        grp = [j for j in range(12) if not used[j] and abs(evals[j] - evals[i]) < 1e-6]
        for j in grp: used[j] = True
        groups.append((evals[i], grp))
    omega = np.exp(2j * np.pi / 3)
    def wof(z): return 0 if abs(z-1) < .3 else (+1 if abs(z-omega) < .3 else -1)
    split = {}
    for lam, grp in groups:
        Q, _ = np.linalg.qr(R[:, grp])
        pv = np.linalg.eigvals(Q.conj().T @ P @ Q)
        sw2 = sum(wof(z) ** 2 for z in pv)
        split[round(abs(lam), 4)] = split.get(round(abs(lam), 4), 0) + sw2
    return split


# --------------------------------------------------------------------------
# (1) the BZ integrand and the integral
# --------------------------------------------------------------------------
def integrand(k):
    """primary: Sum_{Im lam>0} |<l|W|r>|^2 . Im(lam)/|lam|^2  (diagonal per-mode F);
       also returns the basis-free cross-check (1/2)Sum_all w^2 |Im lam|/|lam|^2."""
    B = nb_operator(k, de, rev)
    lam, R = np.linalg.eig(B)
    if np.linalg.cond(R) > 1e9:
        return None
    L = np.linalg.inv(R)
    wnn = np.diag(L @ W @ R)
    chiral = 0.0; halfabs = 0.0
    for n in range(12):
        F = lam[n].imag / (abs(lam[n]) ** 2)
        w2 = abs(wnn[n]) ** 2
        halfabs += 0.5 * w2 * abs(F)
        if lam[n].imag > 1e-9:
            chiral += w2 * F
    return chiral, halfabs

def bz_ratio(ngrid):
    pts = (np.arange(ngrid) + 0.5) / ngrid      # offset MP grid, avoids Gamma
    acc = np.zeros(2); cnt = 0
    for a in pts:
        for b in pts:
            for c in pts:
                v = integrand((a, b, c))
                if v is None: continue
                acc += v; cnt += 1
    return acc / cnt / TEMPLATE_GAMMA, cnt


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 78)
    print("  M_Z via the BZ-integrated Z-current (T3 - s^2 Q) vacuum polarization")
    print("=" * 78)

    split = gamma_winding_split()
    print("\n(0) Gamma winding split (joint diag of B(0),P):")
    print(f"    Perron |l|=2   : Sw2 = {split.get(2.0,0)}   (expect 0)")
    print(f"    shell  |l|=sqrt2: Sw2 = {split.get(round(SQRT2,4),0)}   (expect 4; chiral half = 2)")
    print(f"    |l|=1          : Sw2 = {split.get(1.0,0)}   (expect 4)")
    print(f"    total          : Sw2 = {sum(split.values())}   (= Tr W^2 = 8)")
    assert split.get(2.0,0) == 0 and split.get(round(SQRT2,4),0) == 4 and sum(split.values()) == 8

    # Gamma bracket (live constants)
    two_E = 12; s2 = 0.231; alpha1 = (2/3)**8; c_S = 1.0/two_E
    delta_r = c_S * alpha1 / (1 - alpha1)
    shell_step = 2 * c_S * s2**2 * F_GAMMA * alpha1         # chiral Gamma shell, +0.0230%
    SM = 0.003570
    print(f"\n(1) Gamma template bracket:")
    print(f"    delta_r                 = {delta_r*100:+.4f}%   (+8.1 sigma)")
    print(f"    chiral shell @Gamma     = {shell_step*100:+.4f}%   -> total {(delta_r+shell_step)*100:.4f}% (-1.9 sigma)")
    print(f"    SM tree->pole target    = {SM*100:.4f}%   = delta_r + {((SM-delta_r)/shell_step):.4f}.(chiral step)")

    print(f"\n(2) BZ integral  R = <Sw2.F>_BZ / [Sw2.F]_Gamma   (closure target: R=0.810):")
    print(f"    {'ngrid':>6} {'R(diag chiral)':>15} {'R(1/2 Sum|F|)':>15}")
    Rfinal = None
    for ng in (12, 20, 28, 36, 44):
        (rch, rha), cnt = bz_ratio(ng)
        print(f"    {ng:>6} {rch:>15.5f} {rha:>15.5f}")
        Rfinal = rch
    R = Rfinal

    print(f"\n(3) VERDICT:")
    print(f"    R = {R:.4f}.  0.810 does NOT fall out (off by {(R-0.810)/0.810*100:.0f}%).")
    shell_bz = R * shell_step
    total_bz = delta_r + shell_bz
    gap = SM - total_bz
    M_Z, sig = 91.1876, 0.0021
    sigma_after_dr   = (SM - delta_r) * M_Z / sig
    sigma_after_bz   = gap * M_Z / sig
    print(f"    BZ shell           = R . {shell_step*100:.4f}% = {shell_bz*100:+.4f}%")
    print(f"    substrate oblique  = delta_r + shell_BZ = {total_bz*100:.4f}%   (SM {SM*100:.4f}%)")
    print(f"    => substrate UNDER-predicts the SM oblique by {gap*100:.4f}% ({gap/SM*100:.1f}% relative)")
    print(f"    M_Z residual: after delta_r alone  = {sigma_after_dr:+.1f} sigma")
    print(f"                  after + BZ shell     = {sigma_after_bz:+.1f} sigma   (NOT closed)")
    print()
    print("    => The Gamma 'bracket' was an ARTIFACT of evaluating F at its BZ")
    print("       maximum (Gamma).  The genuine BZ-integrated shell is ~5x smaller and")
    print("       does NOT bracket -- the substrate UNDER-predicts the EW oblique.")
    print("       M_Z is a FORCED substrate-vs-SM oblique difference (a real ~4%-relative")
    print("       residual), the framework's intrinsic precision floor on the oblique.")
    print("       0.810 is NOT forced by T3 - s^2 Q.  Honest result, not a fit, not a close.")
    print("=" * 78)
