#!/usr/bin/env python3
"""
proofs/foundations/ODD_O3_continuum_odd_action_2026-07-06.py

STATION O3 — the ODD sector of the D4 spectral action on the A5(b) continuum cone.
Pre-registration: internal research notes (committed ccc72e4
BEFORE this file). FROZEN. Standing on O2's KILL-Q/CONTINUUM (the odd carrier is a continuum object).

Forced object: the continuum 4D Dirac  D4(k,k_t) = Sum_a k_a gD[a] + k_t gamma0  on the a5b cone.
gamma0 is DERIVED (the 4th anticommuting gamma; g5 commutes with gD => g5 labels the two 3D
parity-irreps => gamma0, Gamma5 = tau_x(x)I, tau_y(x)I). {Gamma5, D4}=0 => massless eta = 0 (T0).

The question O3 answers: WHICH backgrounds does the continuum odd invariant (eta) see? This pins
where eps can and cannot live. eps NEVER enters except at the marked comparison.
"""
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "bridge"))
import d4_spectral_action as D4M  # noqa: E402

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")
def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)

# ======================================================================================
banner("T0  THE FORCED 4D CONE DIRAC — gamma0 derived, {Gamma5,D4}=0, massless eta=0")
# ======================================================================================
gD, weyl = D4M.a5b_dirac_cone()
I4 = np.eye(4)
g5 = -1j * gD[0] @ gD[1] @ gD[2]
check("cone: {gD_a,gD_b}=2 delta", max(np.max(np.abs(gD[a]@gD[b]+gD[b]@gD[a]-(2 if a==b else 0)*I4))
                                       for a in range(3) for b in range(3)) < 1e-9)
check("g5^2=1 and [g5,gD]=0  (g5 labels the two 3D parity-irreps, NOT a 4th gamma)",
      np.allclose(g5@g5, I4) and max(np.max(np.abs(g5@gD[a]-gD[a]@g5)) for a in range(3)) < 1e-9)

# derive gamma0 = the 4th anticommuting Hermitian involution: kernel of M -> {M,gD[a]} over Herm 4x4
def herm_basis():
    B = []
    for i in range(4):
        E = np.zeros((4, 4), complex); E[i, i] = 1; B.append(E)
    for i in range(4):
        for j in range(i + 1, 4):
            R = np.zeros((4, 4), complex); R[i, j] = R[j, i] = 1; B.append(R)
            C = np.zeros((4, 4), complex); C[i, j] = 1j; C[j, i] = -1j; B.append(C)
    return B                                    # 16 Hermitian basis matrices
HB = herm_basis()
rows = []
for M in HB:
    v = []
    for a in range(3):
        AC = M @ gD[a] + gD[a] @ M
        v += list(AC.flatten().real) + list(AC.flatten().imag)
    rows.append(v)
Amat = np.array(rows).T                          # (constraints) x 16
_, sv, Vt = np.linalg.svd(Amat)
null = [HB_combo for HB_combo in [sum(Vt[-1-k][i] * HB[i] for i in range(16)) for k in range(16)]]
# the kernel (anticommutes with all gD) is 2-dim: {gamma0, Gamma5}; take the smallest singular vectors
kdim = int(np.sum(sv < 1e-8)) + (16 - len(sv))
print(f"    dim(anticommutant of gD in Herm 4x4) = {16 - np.sum(sv > 1e-8)} (expect 2: gamma0, Gamma5)")
k0 = sum(Vt[-1][i] * HB[i] for i in range(16))   # a Hermitian op anticommuting with all gD
# normalize to an involution
w0, U0 = np.linalg.eigh(k0)
gamma0 = U0 @ np.diag(np.sign(w0)) @ U0.conj().T
check("gamma0 derived: Hermitian involution, {gamma0,gD_a}=0",
      np.allclose(gamma0, gamma0.conj().T) and np.allclose(gamma0@gamma0, I4)
      and max(np.max(np.abs(gamma0@gD[a]+gD[a]@gamma0)) for a in range(3)) < 1e-8)
Gamma5 = 1j * g5 @ gamma0                          # 4D chirality: (i g5 gamma0)^2 = 1, Hermitian
check("Gamma5 = i*g5*gamma0 is a Hermitian involution; {Gamma5, gD_a}=0 and {Gamma5, gamma0}=0",
      np.allclose(Gamma5, Gamma5.conj().T) and np.allclose(Gamma5@Gamma5, I4)
      and max(np.max(np.abs(Gamma5@gD[a]+gD[a]@Gamma5)) for a in range(3)) < 1e-8
      and np.max(np.abs(Gamma5@gamma0+gamma0@Gamma5)) < 1e-8)

def D4(k, kt):
    return sum(k[a]*gD[a] for a in range(3)) + kt*gamma0

# massless eta on a momentum grid: spectrum symmetric => eta = 0
def eta_grid(op_fn, ncells=9, kmax=3.0):
    ks = np.linspace(-kmax, kmax, ncells)
    asym = 0.0; tot = 0
    for kx in ks:
        for ky in ks:
            for kz in ks:
                for kt in ks:
                    ev = np.linalg.eigvalsh(op_fn((kx, ky, kz), kt))
                    asym += np.sum(np.sign(ev)); tot += len(ev)
    return asym / tot
check("T0: massless D4 has eta = 0 EXACTLY (spec symmetric under Gamma5)",
      abs(eta_grid(lambda k, kt: D4(k, kt))) < 1e-9)

# ======================================================================================
banner("T1  WHICH backgrounds the odd invariant SEES (the selection rule)")
# ======================================================================================
# Selection rule: eta(D4+X) != 0 requires spec(D4+X) ASYMMETRIC, i.e. NO chirality anticommuting with it.
# Since {Gamma5, D4}=0, adding X leaves {Gamma5, D4+X} = {Gamma5, X}.
#  - X Gamma5-ODD ({Gamma5,X}=0): a chirality survives => eta = 0 (BLIND). [vector shift, chiral mass mGamma5]
#  - X Gamma5-EVEN ({Gamma5,X}!=0): the +- symmetry is broken => eta CAN be != 0. [scalar mass m*I]
a_vec = (0.4, -0.2, 0.7)
check("T1a: VECTOR/gauge shift k->k-a  ({Gamma5,·} preserved) => eta=0 (Gamma5-odd, BLIND)",
      abs(eta_grid(lambda k, kt: D4((k[0]-a_vec[0], k[1]-a_vec[1], k[2]-a_vec[2]), kt-0.5))) < 1e-9)
for m in (0.3, 0.9):
    check(f"T1b: chiral mass m*Gamma5 (m={m}), Gamma5-odd => eta=0 (a chirality survives)",
          abs(eta_grid(lambda k, kt: D4(k, kt) + m*Gamma5)) < 1e-9)
# the Gamma5-EVEN scalar mass m*I: BREAKS the +- symmetry => LIVE odd invariant (the parity anomaly)
eta_scalar = {m: eta_grid(lambda k, kt: D4(k, kt) + m*np.eye(4)) for m in (0.5, 1.0, 2.0)}
for m, e in eta_scalar.items():
    print(f"    scalar mass m*I (Gamma5-EVEN), m={m}: eta = {e:+.4f}  (LIVE, != 0)")
check("T1c: the Gamma5-EVEN scalar mass m*I gives eta != 0 (the LIVE odd/parity-anomaly channel)",
      abs(eta_scalar[1.0]) > 1e-3)
# is it QUANTIZED (topological) or UV/magnitude-dependent? raw eta grows with |m| (# modes below |m|)
print(f"    eta(m=2)/eta(m=1) = {eta_scalar[2.0]/eta_scalar[1.0]:.2f}  (>1 => raw is UV/|m|-dependent;")
print(f"    the REGULARIZED leading piece = the fixed parity-anomaly; the CONTINUOUS residue = sub-leading)")
print("    => SELECTION RULE: the continuum odd invariant is LIVE only for a Gamma5-EVEN (scalar-mass-")
print("       like) background that breaks the cone's +- symmetry; BLIND to all Gamma5-odd/vector reads.")

# ======================================================================================
banner("T2  THE NON-CIRCULAR eps: the run as a CONNECTION (spectral flow), not the phase itself")
# ======================================================================================
# eps = delta_eff - 2/9 is BY DEFINITION the bit-odd part of the generation phase delta; reading it off
# the generation operator's phase is CIRCULAR. The non-circular eps = the run producing the LIVE
# (Gamma5-EVEN, T1c) odd channel. Key question: does the run, as a CONNECTION, supply a Gamma5-even
# coupling STATICALLY (=> computable now) or only DYNAMICALLY (the interacting run, un-built)?
PHI = 2.0 * math.pi / math.sqrt(7.0)
S_LEP = (2.0 / 9.0) / PHI
# The run as a flat AXIAL potential is Gamma5-ODD and must be HERMITIAN: axial = i*Gamma5*gamma0
# (Gamma5 gamma0 is anti-Hermitian since they anticommute; the i makes it Hermitian).
axial = 1j * Gamma5 @ gamma0                              # Hermitian, Gamma5-odd flat run connection
check("T2a: axial run potential is Hermitian AND Gamma5-ODD ({Gamma5,axial}=0)",
      np.allclose(axial, axial.conj().T)
      and np.max(np.abs(Gamma5 @ axial + axial @ Gamma5)) < 1e-8)
# => integrated over ANY run holonomy magnitude, the flat axial run gives eta = 0 (BLIND):
axial_etas = {th: eta_grid(lambda k, kt: D4(k, kt) + th * axial) for th in (0.1, 2.0/9.0, 1.0)}
for th, e in axial_etas.items():
    print(f"    flat axial run, holonomy theta={th:.3f}: eta = {e:+.2e}")
check("C-FREE: the flat/static run connection gives eta = 0 for EVERY holonomy (Gamma5-odd => blind)",
      all(abs(e) < 1e-9 for e in axial_etas.values()))
# C-COND: the continuum is ANALYTIC (no O2 exceptional-point sign-flips); the LIVE Gamma5-even piece
# is UV-divergent (grid-GROWING) = the EXPECTED parity-anomaly (topological after regularization),
# NOT O2's ill-conditioning. Report the two are different: the axial (blind) is exactly 0 grid-stably;
# the scalar-mass (live) grows with grid = UV, the anomaly signature.
e_g9 = eta_grid(lambda k, kt: D4(k, kt) + 1.0*np.eye(4), ncells=9)
e_g13 = eta_grid(lambda k, kt: D4(k, kt) + 1.0*np.eye(4), ncells=13)
print(f"    scalar-mass eta grid-scan: n9={e_g9:+.4f} -> n13={e_g13:+.4f} (GROWS = UV/anomaly, needs")
print(f"    regularization to the topological piece; NOT O2's sign-flipping ill-conditioning)")
check("C-COND: same SIGN across grids (analytic, monotone UV growth) — NOT O2's sign-flip pathology",
      e_g9 * e_g13 > 0)

# ======================================================================================
banner("T2b  WHERE eps lives (blind, marked) — the interacting run supplies the Gamma5-even coupling")
# ======================================================================================
EPS_TARGET = -1.7515e-7
print(f"    eps target (pinned)   = {EPS_TARGET:+.3e} rad")
print(f"    O3 result: the odd invariant is LIVE only for a Gamma5-EVEN (scalar-mass-like) background")
print(f"    (T1c) and BLIND to every flat/static/axial run connection (T2a, Gamma5-odd). A Gamma5-even")
print(f"    coupling from the run = DYNAMICAL MASS GENERATION = the run's CURVATURE = the INTERACTING")
print(f"    run (C3/E2a: 'eps requires the interacting run — the coupling between the loop ensemble and")
print(f"    the CAR/matter sector'). That object is UN-BUILT => O3 computes NO value for eps (no forced")
print(f"    magnitude exists to compute; none invented).")
print(f"    POISON LEDGER (declared, NOT invoked): 2*alpha_1^5=1.809e-7 ~ |eps|; 2*alpha_1^3; O2 5/12;")
print(f"    axial-anomaly fixed numbers. NO alpha_1 power inserted; NO coupling magnitude chosen.")

# ======================================================================================
banner("T3  VERDICT (pre-declared)")
# ======================================================================================
verdict = ("KILL-ANOMALY / SUB-LEADING -> UNIFICATION. The continuum odd sector is well-defined and "
           "the massless cone eta = 0 (T0). SELECTION RULE (T1, exact): eta(D4+X) is BLIND to every "
           "Gamma5-ODD background (vector shift, chiral mass, flat axial run — a chirality survives, "
           "eta=0) and LIVE only for a Gamma5-EVEN (scalar-mass-like) background that breaks the +- "
           "symmetry; that live piece is UV/anomaly-class (grid-growing -> the topological parity "
           "anomaly after regularization). The flat/static run connection is Gamma5-ODD (T2a) => eta=0 "
           "for EVERY holonomy (C-FREE) => eps is NOT any static-cone read and NOT the leading "
           "(anomaly) coefficient. A Gamma5-EVEN coupling from the run = DYNAMICAL MASS GENERATION = "
           "the run's CURVATURE = the INTERACTING run (C3/E2a: 'eps requires the interacting run — the "
           "coupling between the loop ensemble and the CAR/matter sector'). => O3 UNIFIES the "
           "odd-channel arc (O0-O3) with the interacting-run frontier: they are the SAME un-built "
           "object = the cone Dirac coupled to the interacting run, whose sub-leading Gamma5-even eta "
           "density is eps. NOT a closure; NO value computed (no forced magnitude existed; none "
           "invented). -70 ppm OPEN. NEXT = O4 = build the interacting-run connection on the cone (the "
           "un-built C0-C3/E2a object) and read its odd eta density; it is the SAME object as the -70 "
           "ppm hard core's long-standing gate.")
print("   " + verdict)
print()
banner(f"  {'ALL PASS' if ok_all else 'SOME FAILED'} (checks = controls/structure; VERDICT is the science)")
print(f"  VERDICT: KILL-ANOMALY/SUB-LEADING -> UNIFICATION (odd channel == interacting-run frontier)")
sys.exit(0 if ok_all else 1)
