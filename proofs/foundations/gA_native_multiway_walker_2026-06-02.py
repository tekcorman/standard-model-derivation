#!/usr/bin/env python3
# ============================================================
# g_A reduction — the NATIVE MULTIWAY computation. Measure the axial-charge
# reduction by RUNNING the framework's walker dynamics, not by importing the
# Melosh formula.
# ============================================================
#
# Methodology (the native loop, per coupling_from_multiway_simulation_2026-05-19:
# "THE SIMULATION the user has been asking for"): EVOLVE the walker on the
# substrate -> OBSERVE -> MEASURE the quantity as an emergent property, with
# correctness gates. Here the quantity is the axial-charge (chirality) RETENTION.
#
# WHY THIS IS NATIVE (vs the prior g_A arc). The prior probes took D(k) STATICALLY
# (a band average) and imported the relativistic Melosh factor rho=1/3+(2/3)m/E and
# the SU(6) wavefunction. Here NOTHING is imported: the framework states (walker_
# dynamics.py W1-W4) that the observer's data is a NON-BACKTRACKING walk on srs and
# the 1-step amplitude operator is the Hashimoto B; the axial current IS the
# chirality operator gamma_chiral = -i*g1..g6 (substrate_state.py op_5_9); and
# arg(h) of the walker eigenvalue h=(sqrt3+i sqrt5)/2 carries chirality. So the
# axial-charge reduction = how much chirality POLARIZATION a walker RETAINS as it
# propagates under the actual non-backtracking dynamics with the Clifford transport.
#
# THE OBJECT: the spinor-lifted non-backtracking walk. Directed-edge Hashimoto
# (the framework's amplitude operator) lifted with the per-edge Clifford generator
# gamma^{e} (the same gamma's that build D=sum gamma^e L_e). B_spin propagates the
# 8-dim spinor along each non-backtracking step. We evolve a chirality-polarized
# walker |psi_0> (a +1 eigenstate of gamma_chiral), form |psi_L> = B_spin^L |psi_0>
# (the MULTIWAY coherent sum over all length-L NB walks), and MEASURE the retained
# chirality polarization R(L) = <psi_L|gamma_chiral|psi_L> / <psi_L|psi_L>.
# g_A reduction (emergent) = R at the nucleon's characteristic walk length (girth).
#
# HARD GATES (probe VOID if G0 fails):
#   G0  the SCALAR Hashimoto B reproduces the theorem-grade eigenvalue
#       h=(sqrt3+i sqrt5)/2 at the P-point (engine is correct).
#   G1  L=0 retention = 1 (a chirality eigenstate starts fully polarized).
#   G2  the -1 chirality sector mirrors the +1 sector (no spurious bias).
#
# FLAGGED construction choices (honest, not hidden): (i) gamma^{e} shared by both
# directions of undirected edge e (matches D=sum gamma^e L_e); (ii) Bloch phase per
# step; the real-space walker = BZ average over k (multiway sum over cells); (iii)
# B_spin is non-Hermitian (Hashimoto) so the state is renormalized each step -- we
# track POLARIZATION, not magnitude; (iv) the "characteristic length" = srs girth 10
# (the F1/F8 bound-state scale), reported alongside the full R(L) curve.

import os, sys
import numpy as np
from itertools import product

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from proofs.common import find_bonds  # noqa: E402

# ---- Clifford gamma^1..gamma^6 (8x8) + chirality (same convention as substrate_state op_5_9) ----
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sm, sp = (sx - 1j * sy) / 2, (sx + 1j * sy) / 2


def kron_chain(*m):
    out = m[0]
    for x in m[1:]:
        out = np.kron(out, x)
    return out


def site(j, op, N=3):
    return kron_chain(*[sz if k < j else (op if k == j else I2) for k in range(N)])


# JW fermions -> Cl(6) Majorana gammas (exactly substrate_state op_5_6/5_8)
c = [site(j, sm) for j in range(3)]
cd = [site(j, sp) for j in range(3)]
GAMMAS = []
for j in range(3):
    GAMMAS.append(c[j] + cd[j])          # gamma_{2j-1}
    GAMMAS.append(1j * (c[j] - cd[j]))   # gamma_{2j}
# chirality gamma_chiral = -i gamma1..gamma6
prod = np.eye(8, dtype=complex)
for g in GAMMAS:
    prod = prod @ g
GAMMA_CHIRAL = -1j * prod
assert np.allclose(GAMMA_CHIRAL @ GAMMA_CHIRAL, np.eye(8), atol=1e-9), "gamma_chiral^2 != I"
assert np.allclose(GAMMA_CHIRAL, GAMMA_CHIRAL.conj().T, atol=1e-9), "gamma_chiral not Hermitian"


# ---- srs undirected edges (6) and directed edges (12) ----
def undirected_edges():
    seen = {}
    for s, t, cell in find_bonds():
        cell = tuple(int(x) for x in cell)
        key = (s, t, cell) if s < t else (t, s, tuple(-x for x in cell))
        seen[key] = True
    return sorted(seen.keys())


UE = undirected_edges()              # 6 undirected edges (a,b,n) with a<b
assert len(UE) == 6
# directed edges: (tail, head, shift_cell, undirected_index)
DE = []
for e_idx, (a, b, n) in enumerate(UE):
    DE.append((a, b, np.array(n), e_idx))            # a->b, shift +n
    DE.append((b, a, -np.array(n), e_idx))           # b->a, shift -n
NDE = len(DE)                                        # 12


def hashimoto_scalar(k):
    """Scalar Hashimoto B(k): B[i,j]=phase if head(j)==tail(i) and i is not reverse of j."""
    B = np.zeros((NDE, NDE), dtype=complex)
    for i, (ti, hi, ni, ei) in enumerate(DE):
        for j, (tj, hj, nj, ej) in enumerate(DE):
            if hj != ti:
                continue
            # reverse = same undirected edge, opposite direction
            if ei == ej and tj == hi and hj == ti and np.array_equal(nj, -ni):
                continue
            B[i, j] = np.exp(2j * np.pi * np.dot(k, ni))   # phase for the step taken (edge i)
    return B


def hashimoto_spinor(k, transport="single"):
    """Spinor-lifted Hashimoto: each NB step j->i transports the 8-spinor.
    transport='single'  : gamma^{e(i)}                 (chirality-FLIPPING: odd grade)
    transport='bilinear' : gamma^{e(i)} gamma^{e(j)}   (chirality-PRESERVING: even grade,
                           the natural 'turn' operator from in-edge j to out-edge i)."""
    Bsp = np.zeros((NDE * 8, NDE * 8), dtype=complex)
    for i, (ti, hi, ni, ei) in enumerate(DE):
        ph = np.exp(2j * np.pi * np.dot(k, ni))
        for j, (tj, hj, nj, ej) in enumerate(DE):
            if hj != ti:
                continue
            if ei == ej and tj == hi and hj == ti and np.array_equal(nj, -ni):
                continue
            g = GAMMAS[ei] if transport == "single" else GAMMAS[ei] @ GAMMAS[ej]
            Bsp[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = ph * g
    return Bsp


def chiral_op_full():
    """gamma_chiral on the full (directed-edge x spinor) space = I_NDE (x) gamma_chiral."""
    return np.kron(np.eye(NDE), GAMMA_CHIRAL)


def retention_curve(kpts, Lmax, sector=+1, transport="single"):
    """Evolve chirality-polarized walkers under B_spin^L, BZ-averaged; return R(L)."""
    GC = chiral_op_full()
    # projector onto the chosen chirality sector (on the spinor factor)
    P = np.kron(np.eye(NDE), (np.eye(8) + sector * GAMMA_CHIRAL) / 2)
    dim = NDE * 8
    R = np.zeros(Lmax + 1)
    norms = np.zeros(Lmax + 1)
    for k in kpts:
        Bsp = hashimoto_spinor(np.array(k, float), transport=transport)
        # start from a chirality-polarized, edge-uniform real walker, projected to sector
        psi0 = P @ (np.ones(dim, dtype=complex) / np.sqrt(dim))
        psi = psi0.copy()
        for L in range(Lmax + 1):
            nrm = np.vdot(psi, psi).real
            if nrm < 1e-300:
                break
            pol = np.vdot(psi, GC @ psi).real / nrm
            R[L] += pol * nrm           # weight by amplitude (multiway path count)
            norms[L] += nrm
            psi = Bsp @ psi
            # renormalize to track polarization, not blow-up
            psi = psi / np.sqrt(np.vdot(psi, psi).real)
    return R / np.maximum(norms, 1e-300)


def main():
    print("=" * 78)
    print(" g_A REDUCTION — native MULTIWAY walker computation (no imported Melosh)")
    print("=" * 78)

    # ---- GATE G0: scalar Hashimoto reproduces h=(sqrt3+i sqrt5)/2 at P ----
    h_true = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
    BP = hashimoto_scalar(np.array([0.25, 0.25, 0.25]))
    ev = np.linalg.eigvals(BP)
    closest = ev[np.argmin(np.abs(ev - h_true))]
    g0 = abs(closest - h_true) < 1e-6
    print(f"\n[G0] scalar Hashimoto B(P) eigenvalue nearest h=(sqrt3+i sqrt5)/2={h_true:.4f}:")
    print(f"     found {closest:.6f}   |diff|={abs(closest-h_true):.2e}   -> {'PASS' if g0 else 'FAIL'}")
    if not g0:
        print("     ABORT: edge/Hashimoto construction wrong; everything downstream void.")
        # show spectrum to debug
        print("     spectrum:", np.round(np.sort_complex(ev), 3))
        return
    # |h|^2 = 2 (Ramanujan), arg(h) the chirality phase
    print(f"     |h|^2 = {abs(closest)**2:.4f} (Ramanujan saturation = 2);  "
          f"arg(h) = {np.degrees(np.angle(closest)):.2f} deg (chirality phase)")

    # ---- BZ grid for the multiway (real-space) walker ----
    n = 4
    ks = [(i + 0.5) / n for i in range(n)]
    kpts = list(product(ks, repeat=3))
    Lmax = 14

    # ---- GATES G1/G2 + the measurement, for BOTH natural transports ----
    gA_LO = 5.0 / 3.0
    results = {}
    for tr in ("single", "bilinear"):
        Rp = retention_curve(kpts, Lmax, sector=+1, transport=tr)
        Rm = retention_curve(kpts, Lmax, sector=-1, transport=tr)
        results[tr] = (Rp, Rm)

    Rp_s, Rm_s = results["single"]
    print(f"\n[G1] L=0 retention (full polarization): R+(0)={Rp_s[0]:+.4f}  "
          f"-> {'PASS' if abs(abs(Rp_s[0])-1) < 1e-6 else 'FAIL'}")
    print(f"[G2] -1 sector mirrors +1: R-(0)={Rm_s[0]:+.4f}  "
          f"-> {'PASS' if abs(Rp_s[0]+Rm_s[0]) < 1e-6 else 'FAIL'}")

    print(f"\n[measurement] chirality-polarization retention |R(L)| (multiway-summed, BZ-avg):")
    print(f"     L  | single-gamma (flip)  | bilinear-gamma (turn) |")
    for L in range(Lmax + 1):
        mark = "  <- girth" if L == 10 else ""
        print(f"    {L:2d}  |  R={results['single'][0][L]:+.3f}  |R|={abs(results['single'][0][L]):.3f}"
              f"  |  R={results['bilinear'][0][L]:+.3f}  |R|={abs(results['bilinear'][0][L]):.3f}{mark}")

    for tr in ("single", "bilinear"):
        Rp = results[tr][0]
        R_girth, R_tail = abs(Rp[10]), np.mean(np.abs(Rp[11:Lmax + 1]))
        print(f"\n[emergent g_A | {tr}]  |R| girth-10 = {R_girth:.4f}; tail = {R_tail:.4f}"
              f"  ->  g_A = (5/3)|R| = {gA_LO*R_girth:.4f} (girth), {gA_LO*R_tail:.4f} (tail)")
    print(f"     (observed g_A = 1.2723; static-arc band average gave 1.44)")

    print("\n" + "=" * 78)
    print(" VERDICT — the native walk does NOT dilute the axial charge (honest result)")
    print("=" * 78)
    print(f"""  G0 PASSES to machine precision: the Hashimoto B reproduces the theorem-grade
  h=(sqrt3+i sqrt5)/2 -- the engine is the real framework walk.

  THE MEASUREMENT: the chirality POLARIZATION magnitude |R(L)| = 1.000 for ALL L,
  for BOTH natural Clifford transports:
    - single gamma^e (odd grade): each NB step is a CLEAN chirality FLIP (R alternates
      +-1), magnitude preserved.
    - bilinear gamma^e gamma^f (even grade, the 'turn' operator): each step PRESERVES
      chirality (R=+1), magnitude preserved.
  Either way the axial charge is NOT diluted -> native g_A = (5/3)*1 = 5/3 EXACTLY.
  No reduction emerges from single-walker non-backtracking propagation.

  WHY: a pure-grade Clifford transport (single OR bilinear) maps chirality
  eigenstates to chirality eigenstates. DILUTION needs a MIXED-grade step (both a
  chirality-preserving AND a flipping part at once) -- which neither natural NB
  transport has. The NB walk conserves |chirality|.

  WHAT THIS EXPOSES (the point of going native): the static-arc g_A ~ 1.44 was NOT
  a framework-dynamics result -- it came from the IMPORTED Melosh formula
  rho=1/3+(2/3)m/E plus a band average. The framework's OWN walker dynamics gives
  NO reduction. So the g_A reduction is NOT a single-walker propagation effect; it
  must live in (a) the multi-walker (3-quark) COMPOSITE chirality combination, or
  (b) genuinely outside the native walker dynamics -- consistent with the earlier
  finding that the binding/interaction is chirality-blind. Either way, the honest
  native answer is: the framework substrate dynamics, run, conserves the axial
  charge and does not by itself produce the 5/3 -> 1.27 reduction.

  FLAGGED: gamma^e per-edge transport (the two natural pure-grade choices both
  tested); BZ-averaged real-space walker; girth=10 characteristic length. The
  3-walker composite is the named next native step (not done here).""")
    print("=" * 78)


if __name__ == "__main__":
    main()
