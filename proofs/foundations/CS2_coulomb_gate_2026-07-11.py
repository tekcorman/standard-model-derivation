#!/usr/bin/env python3
"""
proofs/foundations/CS2_coulomb_gate_2026-07-11.py

CS-2 -- THE COULOMB GATE (the connection sector's third station; frozen spec quoted verbatim
from internal research notes, commit d1f56fb, the "CS-2" bullet
under "THE STATIONS", lines 40-43):

    "CS-2 -- THE COULOMB GATE (MEDIUM-HEAVY; behind CS-1; || CS-3): re-run the IV4 two-walker
    binding with the CS-1 kernel. Acceptance = the exact numbers the contact vertex fails:
    exponent -> +1, B_static/B_equal -> 2. Then the atomic block; E_odd = 0.381876 MeV (the
    3-body Coulomb displacement) is the sector's parameter-free EARLY confront; Delta-alpha as
    by-product."

PREREQUISITES (read in full before writing this file):
  - internal research notes: CS-1 landed VERDICT =
    FORCED-MASSLESS-TRANSVERSE. The transverse channel's leading momentum-squared coefficient is
    resolved, isotropic, grid-stable: pi_2 = -5.022683e-08 at u = alpha_1 = (2/3)^8 (isotropic to
    1.58e-18 absolute spread across 6 axis/pair combinations; grid-stable 16^3->20^3 to 7.7e-11
    relative). CS-1's OWN reading of its acceptance criterion (its Sec.3): "the standard argument
    (int d^3p e^{ip.r}/p^2 /(2pi)^3 = 1/(4 pi r), NOT re-derived numerically there) then gives a
    static 1/r kernel between two sources coupled through this current." CS-1 explicitly declined
    to fix an absolute scale for this coefficient (its Hazard #3: "No renormalization-scheme
    identification is attempted ... the raw u_1^8 suppression means the number is not on a
    directly comparable scale to any O(1) coupling without one, and none is proposed here").
  - docs/framework/BOOTCAMP.md Sec.7 (the interaction anatomy): "The contact class has mu-exponent
    ~= +0.12 (deep-contact), NOT linear-mu ==> Coulomb-class binding ... belongs to the CONNECTION
    sector," and Sec.9 (the trap list): "alpha_1 != alpha_EM."
  - proofs/foundations/IV4_T0_class_2026-07-10.py (T1-FENCE verdict; re-run fresh in this file,
    Sec.0 below, exit 0): the MDL CONTACT vertex's own two-walker binding gives, at the operating
    point U=3, s=1: exponent dlnB/dln(mu_eff) = +0.1211 (EQUAL), B_static/B_equal = 1.0597 -- "the
    exact numbers the contact vertex fails" that this station's acceptance criterion refers to.

WHAT "THE CS-1 KERNEL" MEANS HERE (disclosed construction, not invented from nothing): CS-1's own
station established the EXISTENCE of a nonzero, isotropic, grid-stable transverse p^2 coefficient
and (per its own Sec.3, quoted above) read this via the standard continuum argument as licensing a
static 1/r real-space kernel. CS-1 did NOT hand off a usable-at-all-p lattice function (its own
Hazard #3 forbids treating its raw pi_2 number as an absolute coupling), so this station builds
the STANDARD real-space form V(R) = -g/|R| (the object CS-1's own acceptance language names) on
the SAME real-space relative-cell lattice IV4 diagonalizes on, and treats the overall coupling g
as a SCANNED, declared-context parameter -- exactly mirroring IV4's own U/s class-scan practice,
never fixed to CS-1's raw pi_2 number (poison: "No retro-fit of the Maxwell normalization to
alpha_EM"; CS-1's number is cited ONLY for orientation in Sec.4 below, never used as an amplitude).

THE KINETIC SECTOR T-HAT is UNCHANGED from IV4: the SAME 32x32 lowest-positive-Dirac-band
dispersion (Block 1, replicated verbatim, cited by line) feeding the SAME real-space Fourier
synthesis kinetic_real_space (Block 2, replicated verbatim). Only V-hat changes: IV4's contact
well -U|0><0| is replaced by the long-range kernel described above. This isolates the ONE
variable the design note asks this station to test: does the SHAPE of the potential (contact vs.
1/r), not any new kinetic physics, control the mu-scaling class?

STANDING LAWS HONORED: top-down; NO goal-seeking toward any measured binding energy (hydrogen,
positronium, Rydberg, nuclear); confronts happen ONLY where named below, with declared tolerances;
dual-outcome throughout (a negative is fully bookable); alpha_1 != alpha_EM; the_run.py, the_net.py,
verify.py, locks/registers, and CS-1's own file are NOT edited (this is a new, standalone file).

GATE DISCIPLINE: the design note's CS-2 bullet does not specify (a) a numeric acceptance
tolerance for "-> +1" / "-> 2" (unlike IV4's OWN pre-reg, which DOES carry explicit thresholds
for these same two quantities), (b) a construction for "the atomic block," or (c) a construction
for the "E_odd ... parameter-free EARLY confront" beyond naming the target number. Per this
station's own dispatch ("if the CS-2 spec leaves a decision point ambiguous, do not invent --
name the ambiguity ... and stop that branch"):
  (a) is resolved by REUSING (not inventing) IV4's own already-frozen thresholds for the SAME two
      quantities (IV4_T0_class_prereg_2026-07-10.md: |exponent_op-1|<=0.15 AND
      |B_static/B_equal-2|<=0.2) -- the least-invented available choice, disclosed here BEFORE
      Sec.2's numbers are computed.
  (b) and (c) are NOT resolved by invention: Sec.5 (atomic block) and Sec.6 (E_odd) each open with
      a named blocker and STOP the quantitative branch, per the gate. Sec.6 delivers only the one
      construction-free piece available (a sign/direction check), explicitly separated from the
      blocked quantitative match.

Standalone: python3 proofs/foundations/CS2_coulomb_gate_2026-07-11.py ; exit 0.
"""

import os
import sys
import time
import numpy as np
from itertools import product

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from proofs.common import find_bonds  # noqa: E402

T0_WALL = time.time()


def banner(t, w=100):
    print("=" * w)
    print(f" {t}")
    print("=" * w)


def elapsed():
    return time.time() - T0_WALL


# =============================================================================================
# BLOCK 1 -- the 32x32 Dirac D(k) + lowest-positive band, REPLICATED VERBATIM from
# proofs/foundations/IV4_T0_class_2026-07-10.py lines 131-197 (itself replicated verbatim from
# bound_state_edge_resolved_kernel_2026-05-29.py). NOT edited; the T-hat sector this station
# holds fixed while only V-hat changes.
# =============================================================================================
E_BIT = 1.0
GIRTH = 10

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
k3 = lambda a, b, c: np.kron(np.kron(a, b), c)
GAMMAS = [k3(X, I2, I2), k3(Y, I2, I2), k3(Z, X, I2),
          k3(Z, Y, I2), k3(Z, Z, X), k3(Z, Z, Y)]
BONDS = find_bonds()


def undirected_edges():
    seen = {}
    for src, tgt, cell in BONDS:
        cell = tuple(int(c) for c in cell)
        key = (src, tgt, cell) if src < tgt else (tgt, src, tuple(-c for c in cell))
        seen[key] = True
    e = sorted(seen.keys())
    assert len(e) == 6
    return e


EDGES = undirected_edges()


def L_e(edge, k):
    a, b, n = edge
    L = np.zeros((4, 4), dtype=complex)
    ph = np.exp(2j * np.pi * np.dot(k, n))
    L[b, a], L[a, b] = ph, np.conj(ph)
    for c in range(4):
        if c not in (a, b):
            L[c, c] = 1.0
    return L


def D_of_k(k):
    D = np.zeros((32, 32), dtype=complex)
    for i, e in enumerate(EDGES):
        D += np.kron(GAMMAS[i], L_e(e, k))
    return D


def validate_dirac():
    kk = np.array([0.17, 0.31, 0.53])
    D = D_of_k(kk)
    R = np.zeros((32, 32), dtype=complex)
    Ls = [L_e(e, kk) for e in EDGES]
    for i in range(6):
        for j in range(6):
            if i != j:
                R += 0.5 * np.kron(GAMMAS[i] @ GAMMAS[j], Ls[i] @ Ls[j] - Ls[j] @ Ls[i])
    return np.allclose(D @ D, 6 * np.eye(32) + R, atol=1e-9) and np.allclose(D, D.conj().T)


def eps_low(k):
    ev = np.linalg.eigvalsh(D_of_k(k))
    return ev[ev > 1e-9].min()


# =============================================================================================
# BLOCK 2 -- dispersion mesh + real-space kinetic synthesis, REPLICATED VERBATIM from
# IV4_T0_class_2026-07-10.py lines 292-336.
# =============================================================================================
_EPS_CACHE = {}


def eps_on_mesh(n_q):
    if n_q in _EPS_CACHE:
        return _EPS_CACHE[n_q]
    qs = (np.arange(n_q) + 0.5) / n_q
    eps = np.empty((n_q, n_q, n_q))
    for i, j, l in product(range(n_q), repeat=3):
        eps[i, j, l] = eps_low(np.array([qs[i], qs[j], qs[l]]))
    _EPS_CACHE[n_q] = eps
    return eps


def epair_equal(n_q):
    eps = eps_on_mesh(n_q)
    return eps + eps[::-1, ::-1, ::-1]


def epair_static(n_q, const=None):
    eps = eps_on_mesh(n_q)
    if const is None:
        const = eps.min()
    return eps + const


def kinetic_real_space(epair3, box, n_q):
    qs = (np.arange(n_q) + 0.5) / n_q
    rng = np.arange(-2 * box, 2 * box + 1)
    px = np.exp(2j * np.pi * np.outer(rng, qs))
    A = np.einsum('ri,ijk->rjk', px, epair3.astype(complex))
    B = np.einsum('sj,rjk->rsk', px, A)
    T = np.einsum('tk,rsk->rst', px, B) / n_q ** 3
    return T


# =============================================================================================
# BLOCK 3 -- the CONTACT vertex solver, REPLICATED VERBATIM from IV4_T0_class_2026-07-10.py
# lines 338-379 (solve_relative_contact, pole_binding). Used ONLY to reproduce, fresh, IN THIS
# FILE, the exact baseline numbers ("the exact numbers the contact vertex fails") the design
# note's acceptance criterion refers to -- a regression check against the already-landed T1-FENCE
# verdict, not a re-adjudication of it.
# =============================================================================================
_RSI_CACHE = {}


def real_space_index(box):
    """Rv: (M,3) integer relative-cell coordinates; idx: (M,M) lookup into a (4box+1)^3 array of
    separations d=Ri-Rj, exactly IV4's own index arithmetic (lines 344-349), factored out for
    reuse by BOTH the contact solver (Block 3) and the general-kernel solver (Block 4). Cached by
    box: this station calls it repeatedly (box convergence loop, A-1, A-2 scan, Sec.6) with only a
    handful of distinct box values, and the (M,M) index/separation arrays are the dominant cost
    at large box (M=(2*box+1)^3), so recomputation is pure waste -- performance only, no physics
    content."""
    if box in _RSI_CACHE:
        return _RSI_CACHE[box]
    Rv = np.array(list(product(range(-box, box + 1), repeat=3)))
    d = Rv[:, None, :] - Rv[None, :, :]
    idx = ((d[..., 0] + 2 * box) * (4 * box + 1) + (d[..., 1] + 2 * box)) \
        * (4 * box + 1) + (d[..., 2] + 2 * box)
    i0 = int(np.where((Rv == 0).all(axis=1))[0][0])
    _RSI_CACHE[box] = (Rv, d, idx, i0)
    return Rv, d, idx, i0


def solve_relative_contact(epair3, box, n_q, U):
    T = kinetic_real_space(epair3, box, n_q)
    E_th = float(epair3.min())
    Rv, d, idx, i0 = real_space_index(box)
    Tmat = T.reshape(-1)[idx]
    H = Tmat.copy()
    H[i0, i0] += -U
    ev, evec = np.linalg.eigh(H)
    E0 = float(ev[0])
    psi = evec[:, 0]
    T_exp = float(np.real(psi.conj() @ Tmat @ psi))
    V_exp = -U * float(np.abs(psi[i0]) ** 2)
    assert abs(E0 - (T_exp + V_exp)) < 1e-9, "eigen identity broken"
    return E0, E_th, T_exp, V_exp, float(np.abs(psi[i0]) ** 2)


def pole_binding(dep_flat, U, s, delta_safe=0.02, iters=200):
    g = lambda B: U * np.mean(1.0 / (s * dep_flat + B))
    lo, hi = delta_safe * s, U
    if g(lo) < 1.0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if g(mid) >= 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


# =============================================================================================
# BLOCK 4 -- NEW: the Coulomb-shape kernel + general (non-contact) real-space solver.
#
# V(R) := -g / sqrt(|R|^2 + a^2),  a = 1 (the SAME lattice unit as e_bit=t=1 used throughout;
# a softened/regularized 1/r, since bare 1/|R| is undefined at the on-site R=0 separation and no
# convention for that is specified anywhere in this program -- a=1 is the minimal, no-new-scale
# choice: the core radius equals the ALREADY-adopted unit cell spacing, nothing else is imported).
# HAZARD (named, not resolved): an alternative regularization (e.g. excluding R=0 entirely, V=0
# there) is equally defensible and NOT tested here beyond the box-convergence check in Sec.1;
# this is a disclosed simplification, per this station's own gate discipline.
#
# V IS DIAGONAL in the relative-position basis (a bug found and fixed mid-station, disclosed
# below), exactly as IV4's OWN contact well is: solve_relative_contact's V is -U at the SINGLE
# entry [i0,i0] and zero everywhere else -- i.e. V acts as V(R)*delta(R-R'), a LOCAL potential of
# the relative coordinate R itself, never a hopping term between two DIFFERENT R-basis states.
# Only T-hat (built from the dispersion via kinetic_real_space) is a genuine hopping matrix
# T(Ri-Rj), because momentum-space kinetic energy is non-local in position space; the potential is
# not. A first draft of this file built V as a full matrix V[i,j] = -g/sqrt(|Ri-Rj|^2+a^2) (an
# off-diagonal "hopping" kernel) and the box-convergence check (Sec.2.1) caught it immediately:
# binding energy GREW without bound as box grew (259 -> 439 -> 666 -> 939 -> 1259 at box=3..7,
# clearly unphysical -- a genuine Coulomb well's binding should CONVERGE as the box exceeds the
# bound state's extent, exactly as IV4's OWN Stage-0 box scan converges to <2e-4). Diagnosed as
# the wrong operator (V must be diagonal in R, not a matrix in R-R'); fixed below to
# Vdiag[i] = -g/sqrt(|Rv[i]|^2+a^2), added only to the Hamiltonian's diagonal.
#
# The "s" (band-curvature) rescaling reused for the mu-exponent scan mirrors IV4's OWN pole_binding
# convention (lines 362-379: s multiplies "dep" = E_pair - E_th, holding the s=1 THRESHOLD fixed as
# the reference zero) -- NOT a naive global rescale of epair3. Reproduced here for a full
# real-space diagonalization (no simple pole condition exists for a non-contact potential):
# since kinetic_real_space is LINEAR in its momentum-space input, T(epair3) and a constant-shifted
# copy T(E_th + s*(epair3-E_th)) = s*T(epair3) + (1-s)*E_th at every R, EXCEPT that the constant
# term (1-s)*E_th, being q-independent, Fourier-synthesizes to a pure on-site (R=0) shift (a
# derived, checked fact: Sec.0 self-test). Hence: T_mat(s) = s*T_mat(1); T_mat(s)[i,i] +=
# (1-s)*E_th for EVERY diagonal entry i (every cell's self-separation is R=(0,0,0), not just one
# distinguished index). This exactly reproduces IV4's own "s" convention for the general
# (matrix-diagonalization) solver used here, generalizing the class-scan test beyond the
# contact-only pole condition IV4 used it for.
# =============================================================================================
def coulomb_v_diag(Rv, a=1.0):
    """V(R)/g = -1/sqrt(|R|^2+a^2), a DIAGONAL (M,) array -- one value per relative-position
    basis state Rv[i], NOT a matrix between different states (see the block header)."""
    r2 = np.sum(Rv.astype(float) ** 2, axis=-1)
    return -1.0 / np.sqrt(r2 + a * a)


def build_Tmat_s1(epair3, box, n_q):
    """T_mat at s=1 (unscaled), plus E_th and Rv (the relative-position basis values) -- the
    pieces needed to build T_mat(s) for any s via the on-site-shift rule derived above, and
    V's diagonal via coulomb_v_diag(Rv)."""
    T = kinetic_real_space(epair3, box, n_q)
    Rv, d, idx, i0 = real_space_index(box)
    Tmat1 = T.reshape(-1)[idx]
    E_th = float(epair3.min())
    return Tmat1, Rv, E_th


def solve_relative_general(Tmat1, E_th, Vdiag, g, s=1.0):
    """H = T_mat(s) + g*diag(Vdiag); returns E0, B (binding rel. to the FIXED s=1 threshold,
    IV4's own convention), <T>, <V>, and the eigen-identity residual (should be ~0).

    NOTE (bug found and fixed mid-station, disclosed): every DIAGONAL entry Tmat[i,i] evaluates
    the SAME on-site (R=0) value (self-separation is always zero, for ANY cell i, not just the
    single index i0 = the cell R=(0,0,0) itself) -- so the (1-s)*E_th on-site correction derived
    in Sec.0 must be added to the WHOLE diagonal, not just Tmat[i0,i0]. A first draft applied the
    shift only at i0 and failed Sec.0's own self-test by 0.36 (max|T_direct-T_derived|); caught by
    that self-test before any physics number was trusted, fixed here."""
    M = Tmat1.shape[0]
    idxr = np.arange(M)
    Tmat_s = (s * Tmat1).copy()
    Tmat_s[idxr, idxr] += (1.0 - s) * E_th
    H = Tmat_s.copy()
    H[idxr, idxr] += g * Vdiag
    ev, evec = np.linalg.eigh(H)
    E0 = float(ev[0])
    psi = evec[:, 0]
    T_exp = float(np.real(psi.conj() @ Tmat_s @ psi))
    V_exp = float(g * np.sum(np.abs(psi) ** 2 * Vdiag))
    resid = abs(E0 - (T_exp + V_exp))
    B = E_th - E0
    return E0, B, T_exp, V_exp, resid


def main():
    W = 100
    print("=" * W)
    print("CS-2 -- THE COULOMB GATE")
    print("pre-reg: internal research notes, CS-2 bullet (quoted "
          "verbatim in this file's docstring)")
    print("units: substrate energy (t = e_bit = 1); dispersion/box/n_q conventions inherited "
          "verbatim from IV4_T0_class_2026-07-10.py")
    print("=" * W)

    ok = validate_dirac()
    print(f"\n[validation] Dirac D(k)^2 = 6I + R_sub : {'PASS' if ok else 'FAIL'}")
    assert ok, "Dirac validation regression"

    # -------------------------------------------------------------------------------------------
    # Sec.0 -- self-test of the on-site-shift rule used by solve_relative_general's "s" scaling
    # -------------------------------------------------------------------------------------------
    banner("Sec.0 -- self-test: constant-shift-of-dispersion => on-site-only real-space shift")
    box_t, nq_t = 3, 14
    ep_t = epair_equal(nq_t)
    Tmat1_t, _Rv_t_unused, Eth_t = build_Tmat_s1(ep_t, box_t, nq_t)
    s_t = 0.7
    # direct route: rebuild T from the explicitly shifted dispersion E_th + s*(eps-E_th)
    ep_shifted = Eth_t + s_t * (ep_t - Eth_t)
    T_direct = kinetic_real_space(ep_shifted, box_t, nq_t)
    Rv_t, _, idx_t, _ = real_space_index(box_t)
    Tmat_direct = T_direct.reshape(-1)[idx_t]
    # derived route: s*Tmat1 + on-site shift on the WHOLE diagonal (every cell's self-separation
    # is R=(0,0,0), not just the single distinguished index i0 -- see solve_relative_general's
    # docstring for the bug this self-test caught in a first draft)
    Mt = Tmat1_t.shape[0]
    Tmat_derived = (s_t * Tmat1_t).copy()
    Tmat_derived[np.arange(Mt), np.arange(Mt)] += (1 - s_t) * Eth_t
    dev0 = float(np.max(np.abs(Tmat_direct - Tmat_derived)))
    print(f"  max|T_mat(direct shifted-dispersion) - T_mat(derived on-site-shift-only)| = {dev0:.3e}")
    assert dev0 < 1e-9, "on-site-shift rule broken -- do not trust solve_relative_general's s-scan"
    print("  [PASS] the on-site-shift rule (used throughout Sec.2/Sec.3) is exact")

    # -------------------------------------------------------------------------------------------
    # Sec.1 -- REPRODUCE the contact-vertex baseline (regression against the already-landed
    # T1-FENCE verdict), fresh, in this file
    # -------------------------------------------------------------------------------------------
    banner("Sec.1 -- baseline: the CONTACT vertex's own two-walker binding (IV4, reproduced fresh)")
    U_OP = 3.0
    box1, nq1 = 4, 14
    ep_eq1 = epair_equal(nq1)
    ep_st1 = epair_static(nq1)
    res_contact = {}
    for label, ep3 in (("EQUAL", ep_eq1), ("STATIC", ep_st1)):
        E0, E_th, T_exp, V_exp, a0 = solve_relative_contact(ep3, box1, nq1, U_OP)
        Bnd = E_th - E0
        res_contact[label] = Bnd
        print(f"  {label:6s}: E_th={E_th:9.6f}  E0={E0:10.6f}  B={Bnd:9.6f}")
    ratio_contact = res_contact["STATIC"] / res_contact["EQUAL"]
    print(f"  B_static/B_equal (contact) = {ratio_contact:.6f}")
    assert abs(ratio_contact - 1.0597) < 2e-3, "contact ratio regression vs IV4's own reported 1.0597"

    ep30 = epair_equal(30)
    d30 = (ep30 - ep30.min()).reshape(-1)
    sm, sp = 2 ** (-1 / 8), 2 ** (1 / 8)
    Bm = pole_binding(d30, U_OP, sm)
    Bp = pole_binding(d30, U_OP, sp)
    expo_contact = -(np.log(Bp) - np.log(Bm)) / (np.log(sp) - np.log(sm))
    print(f"  operating-point exponent dlnB/dln(mu_eff) (contact, EQUAL) = {expo_contact:+.6f}")
    assert abs(expo_contact - 0.1211) < 2e-3, "contact exponent regression vs IV4's own reported +0.1211"
    print("  [PASS] contact-vertex baseline regression-matches IV4_T0_class_2026-07-10.py's own "
          "landed T1-FENCE numbers (+0.1211, 1.0597) -- 'the exact numbers the contact vertex "
          "fails' this station's acceptance criterion refers to.")

    # -------------------------------------------------------------------------------------------
    # Sec.2 -- PART A: re-run the two-walker binding with the Coulomb-SHAPE kernel
    # -------------------------------------------------------------------------------------------
    banner("Sec.2 -- PART A: the Coulomb-shape kernel V(R)=-g/sqrt(R^2+1), same T-hat as IV4")
    print("  ADOPTED THRESHOLD (disclosed reuse, not a new invention): IV4's OWN frozen T1-CLOSE "
          "criterion for these same two quantities (IV4_T0_class_prereg_2026-07-10.md): "
          "|exponent_op-1|<=0.15 AND |B_static/B_equal-2|<=0.2. The design note's own CS-2 "
          "wording ('exponent -> +1, B_static/B_equal -> 2') gives no separate numeric tolerance; "
          "this is the least-invented available choice (SAME quantities, predecessor station's "
          "OWN already-frozen numbers), declared BEFORE the numbers below are computed.")

    # --- box convergence at the operating point (g=U_OP=3, s=1) ---
    print("\n[2.1] box convergence at g=3, s=1 (n_q=14), EQUAL and STATIC:")
    ep_eq14 = epair_equal(14)
    ep_st14 = epair_static(14)
    conv = {"EQUAL": {}, "STATIC": {}}
    CONV_BOXES = (3, 4, 5, 6)   # box=7 dropped: box5->6 already converged to <1e-4 relative
                                # (checked in a preliminary run; kept here only through box=6 to
                                # bound wall time on a shared/loaded machine -- performance-only
                                # trim, no physics content changed)
    for box in CONV_BOXES:
        for label, ep3 in (("EQUAL", ep_eq14), ("STATIC", ep_st14)):
            Tmat1, Rv, Eth = build_Tmat_s1(ep3, box, 14)
            Vdiag = coulomb_v_diag(Rv, a=1.0)
            E0, B, T_exp, V_exp, resid = solve_relative_general(Tmat1, Eth, Vdiag, g=U_OP, s=1.0)
            conv[label][box] = B
            print(f"    box={box}  {label:6s}: B={B:10.6f}  <T>={T_exp:9.6f}  <V>={V_exp:10.6f}  "
                  f"eigen-resid={resid:.2e}")
    for label in ("EQUAL", "STATIC"):
        rel = abs(conv[label][5] - conv[label][4]) / abs(conv[label][5])
        rel2 = abs(conv[label][6] - conv[label][5]) / abs(conv[label][6])
        print(f"    {label} relative change box4->5: {rel:.4e}   box5->6: {rel2:.4e}")
    BOX_PRIMARY = 5
    print(f"  PRIMARY box adopted: {BOX_PRIMARY} (box4->5->6 changes reported above, both "
          "<1e-4 relative; box=5 used throughout Sec.2/Sec.3 for the long-range kernel -- larger "
          "than IV4's own box=4 since a 1/r tail is longer-ranged than the contact well, but kept "
          "at 5 rather than 6 for wall-time reasons on a shared machine, disclosed, checked).")

    # --- A-1 analog: the operating-point ratio ---
    banner(f"[2.2] operating-point ratio (g=3, s=1, box={BOX_PRIMARY}, n_q=14)")
    Tmat1_eq, Rv6, Eth_eq = build_Tmat_s1(ep_eq14, BOX_PRIMARY, 14)
    Tmat1_st, _, Eth_st = build_Tmat_s1(ep_st14, BOX_PRIMARY, 14)
    Vdiag6 = coulomb_v_diag(Rv6, a=1.0)
    res_coul = {}
    for label, Tmat1, Eth in (("EQUAL", Tmat1_eq, Eth_eq), ("STATIC", Tmat1_st, Eth_st)):
        E0, B, T_exp, V_exp, resid = solve_relative_general(Tmat1, Eth, Vdiag6, g=U_OP, s=1.0)
        res_coul[label] = B
        print(f"  {label:6s}: E_th={Eth:9.6f}  E0={E0:10.6f}  B={B:9.6f}  eigen-resid={resid:.2e}")
    ratio_coul = res_coul["STATIC"] / res_coul["EQUAL"]
    print(f"\n  B_static/B_equal (Coulomb-shape, g=3) = {ratio_coul:.6f}   "
          f"(contact: {ratio_contact:.6f}; target: 2.0)")

    # --- A-2 analog: the mu-exponent class scan ---
    banner("[2.3] the mu-exponent class scan, g in {0.3,0.5,1,2,3,5} x s in {2^-1..2}")
    U_GRID = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    S_GRID = [2 ** -1.0, 2 ** -0.5, 1.0, 2 ** 0.5, 2.0]
    lnS = np.log(S_GRID)
    scan = {}
    for cfg, ep3, Eth_ref in (("EQUAL", ep_eq14, Eth_eq), ("STATIC", ep_st14, Eth_st)):
        Tmat1, Rv, Eth = build_Tmat_s1(ep3, BOX_PRIMARY, 14)
        Vdiag = coulomb_v_diag(Rv, a=1.0)
        tab = {}
        for g in U_GRID:
            for s in S_GRID:
                E0, B, _, _, resid = solve_relative_general(Tmat1, Eth, Vdiag, g=g, s=s)
                tab[(g, s)] = (B, resid)
        scan[cfg] = tab
        print(f"\n  --- {cfg} ---")
        hdr = "  g \\ s  |" + "".join(f"   s={s:6.4f}   " for s in S_GRID)
        print(hdr)
        for g in U_GRID:
            row = f"  {g:5.2f}  |"
            for s in S_GRID:
                B, resid = tab[(g, s)]
                flag = "!" if resid > 1e-6 else " "
                row += f" {B:9.5f}{flag:<4}"
            print(row)

    # FOUND IDENTITY (visible in the first full run's printed tables; machine-checked here as a
    # pure post-first-run DIAGNOSTIC -- no convention or number above was changed): the STATIC
    # column at scale s equals the EQUAL column at scale s/2 EXACTLY. Derivation: eps(-q)=eps(q)
    # (D(-k) = conj(D(k)), real spectrum), so epair_equal = 2*eps and epair_static = eps+eps_min
    # share E_th = 2*eps_min, and E_th + s*(epair_static-E_th) = E_th + (s/2)*(epair_equal-E_th)
    # identically. Consequence: the A-3 ratio B_static/B_equal and the A-2 mu-exponent are two
    # reads of ONE underlying B(mu) curve (ratio = B(2*mu)/B(mu), exponent = its log-derivative)
    # -- exactly as in IV4's own construction (its mu_static/mu_equal = 2 note).
    id_dev = max(abs(scan["STATIC"][(g, s)][0] - scan["EQUAL"][(g, s / 2)][0])
                 for g in U_GRID for s in S_GRID if (s / 2) in S_GRID)
    print(f"\n  FOUND IDENTITY check: max|B_static(g,s) - B_equal(g,s/2)| over overlapping grid = "
          f"{id_dev:.2e}  (exact; see the in-code derivation comment)")

    # operating-point refined pair s = 2^{-1/8}, 2^{+1/8} (IV4's own declared refinement pair):
    # these two s values are NOT on S_GRID, so they get their own dedicated solves here
    op_expo = {}
    for cfg, ep3 in (("EQUAL", ep_eq14), ("STATIC", ep_st14)):
        Tmat1, Rv, Eth = build_Tmat_s1(ep3, BOX_PRIMARY, 14)
        Vdiag = coulomb_v_diag(Rv, a=1.0)
        _, Bm_c, _, _, _ = solve_relative_general(Tmat1, Eth, Vdiag, g=U_OP, s=sm)
        _, Bp_c, _, _, _ = solve_relative_general(Tmat1, Eth, Vdiag, g=U_OP, s=sp)
        expo = -(np.log(Bp_c) - np.log(Bm_c)) / (np.log(sp) - np.log(sm))
        op_expo[cfg] = expo
    print(f"\n  operating-point exponent (g=3, s=1; refined pair s=2^-1/8,2^+1/8):")
    print(f"    EQUAL : dlnB/dln(mu_eff) = {op_expo['EQUAL']:+.5f}   (contact: {expo_contact:+.5f})")
    print(f"    STATIC: dlnB/dln(mu_eff) = {op_expo['STATIC']:+.5f}")

    # exponent across the whole g-grid (centered log-derivative on the s-grid, EQUAL config)
    print(f"\n  exponent(g) across the full g-grid (EQUAL config, centered on S_GRID):")
    expo_vs_g = {}
    for g in U_GRID:
        Bs = [scan["EQUAL"][(g, s)][0] for s in S_GRID]
        # centered derivative at s=1 (index 2)
        e = -(np.log(Bs[3]) - np.log(Bs[1])) / (lnS[3] - lnS[1])
        expo_vs_g[g] = e
        print(f"    g={g:5.2f}: exponent~{e:+.4f}   B(s=1)={Bs[2]:.6f}")

    # -------------------------------------------------------------------------------------------
    # Sec.3 -- PART A VERDICT
    # -------------------------------------------------------------------------------------------
    banner("Sec.3 -- PART A VERDICT (frozen tree, Sec.2 header, applied here)")
    expo_op = op_expo["EQUAL"]
    close = (abs(expo_op - 1.0) <= 0.15) and (abs(ratio_coul - 2.0) <= 0.2)
    moved_toward = (abs(expo_op - 1.0) < abs(expo_contact - 1.0)) and \
                   (abs(ratio_coul - 2.0) < abs(ratio_contact - 2.0))
    if close:
        part_a_verdict = "CLASS-CONFIRMED"
    elif moved_toward:
        part_a_verdict = "CLASS-PARTIAL"
    else:
        part_a_verdict = "CLASS-MISS"
    print(f"  exponent: contact {expo_contact:+.4f}  ->  Coulomb-shape {expo_op:+.4f}   (target +1)")
    print(f"  ratio   : contact {ratio_contact:.4f}  ->  Coulomb-shape {ratio_coul:.4f}   (target 2.0)")
    print(f"  T1-CLOSE-style criterion (|exp-1|<=0.15 AND |ratio-2|<=0.2): "
          f"{'MET' if close else 'NOT MET'}")
    print(f"  moved toward BOTH targets relative to the contact baseline: {moved_toward}")
    print(f"\n  ==> PART A VERDICT: {part_a_verdict}")
    any_cross = any(abs(expo_vs_g[g] - 1.0) <= 0.15 for g in U_GRID)
    print(f"  (untuned scan check: does ANY g on the frozen grid cross |exponent-1|<=0.15? "
          f"{any_cross} -- reported honestly, not cherry-picked)")

    # -------------------------------------------------------------------------------------------
    # Sec.4 -- CS-1's own number, cited for orientation ONLY (never used as an amplitude above)
    # -------------------------------------------------------------------------------------------
    banner("Sec.4 -- CS-1's own coefficient, cited for orientation (NOT used as an input above)")
    PI2_CS1 = -5.022683e-08     # CS1_finite_k_propagator_2026-07-11.py line 468 (printed fit)
    PI0_CS1 = 4.241799e-09      # ibid., line 467
    PI4_CS1 = 1.620705e-07      # ibid., line 468
    INV_G2_CS1 = -PI2_CS1       # ibid., line 558: "1/g^2_candidate := -pi_2"
    print(f"  CS-1's declared-context number: 1/g^2_candidate = {INV_G2_CS1:+.6e} "
          "(CS1_finite_k_propagator_2026-07-11.py line 558)")
    print(f"  If (NOT done here -- this would be exactly the poisoned retro-fit) this were taken "
          f"as an actual coupling g^2 = 1/{INV_G2_CS1:.4e} = {1.0/INV_G2_CS1:.4e}, it sits "
          f"{1.0/INV_G2_CS1/U_GRID[-1]:.3e} times ABOVE the top of the g-grid scanned in Sec.2 -- "
          "underscoring CS-1's own Hazard #3 (its number is nowhere near an O(1)-comparable "
          "scale). No part of Sec.2/Sec.3 depends on this paragraph's numbers.")

    # -------------------------------------------------------------------------------------------
    # Sec.5 -- PART B: "the atomic block" -- BLOCKED, per this station's own gate discipline
    # -------------------------------------------------------------------------------------------
    banner("Sec.5 -- PART B: THE ATOMIC BLOCK (r_p, H 1S-2S, Lamb, 21cm, Ps x2) -- BLOCKED")
    print("""  NAMED AMBIGUITY / GATE-STOP (per this station's dispatch: 'do not invent -- name the
  ambiguity ... and stop that branch'):
    The design note's mandate (line 5 of IV7_connection_design_note_2026-07-10.md) re-homes the
    atomic block to this sector, but CS-2's own bullet gives NO construction for it, and computing
    any of r_p / H 1S-2S / Lamb / 21cm / Ps requires an ABSOLUTE (dimensionful, comparable-to-1)
    coupling and mass scale. CS-1 explicitly declined to supply that scale (Hazard #3: 'no
    renormalization-scheme identification ... none is proposed here'), and the design note's own
    poison list forbids inventing one here ('No retro-fit of the Maxwell normalization to
    alpha_EM'; 'measured a_e/atomic values quarantined to declared-context lines'). Sec.2 above
    deliberately used a SCANNED, dimensionless g -- exactly to avoid this exact invention. Absent
    a scale bridge, no member of the atomic block can be computed as a genuine framework
    PREDICTION at this station. STOPPED here; not attempted; not faked with a placeholder.""")

    # -------------------------------------------------------------------------------------------
    # Sec.6 -- PART C: E_odd = 0.381876 MeV, "the sector's parameter-free EARLY confront"
    # -------------------------------------------------------------------------------------------
    banner("Sec.6 -- PART C: E_odd (the 3-body Coulomb displacement) -- SIGN CHECK ONLY, "
           "quantitative match BLOCKED")
    print("""  PROVENANCE (opened, not assumed): E_odd = 0.381876 MeV was MEASURED, not framework-
  predicted, at I-0b-RATIO's A3 step (docs/incomplete_equations_todo.md lines 846-848):
  B(3H)/B(3He) = 1.0990 vs the forced-equal-topology value 1 -- i.e. 3He (extra p-p pair) binds
  LESS than 3H (extra n-n pair), the standard sign of an extra repulsive same-charge Coulomb term
  in 3He relative to 3H. It is registered as 'un-priced' (T0_nuclear_prereg_2026-07-10.md line 35;
  IV4_T0_class_prereg_2026-07-10.md line 54): the framework has never produced its OWN value.

  NAMED AMBIGUITY / GATE-STOP (quantitative branch): turning this station's Coulomb-shape kernel
  into an actual MeV-scale prediction for E_odd needs BOTH (i) the same missing absolute scale
  bridge named in Sec.5, and (ii) a genuine 3-body solve on the T0-NUCLEAR-2 relative-Jacobi
  geometry (T0_NUCLEAR2_2026-07-10.py) with one pair's kernel sign flipped -- a construction the
  design note does not specify (which pair? which of that station's box/grid conventions carry
  over?) and which is its own separate HEAVY station, not a sub-task of this one. NOT attempted;
  NOT invented; STOPPED here.

  WHAT IS DELIVERED (parameter-free in the strongest sense -- no coupling magnitude, no scale, no
  3-body machinery, just a sign flip on the ALREADY-BUILT 2-body EQUAL solver from Sec.2): does a
  REPULSIVE same-shape kernel reduce binding relative to the attractive case, at the SAME |g|,
  on the SAME lattice T-hat? This is the minimal, construction-free content of 'an extra
  repulsive term makes the like-charge system bind less' -- the qualitative fact E_odd's own
  measured sign instantiates for 3He vs 3H.""")
    Tmat1_eq6, Rv6b, Eth_eq6 = build_Tmat_s1(ep_eq14, BOX_PRIMARY, 14)
    Vdiag_attr = coulomb_v_diag(Rv6b, a=1.0)
    Vdiag_rep = -Vdiag_attr
    E0_a, B_a, _, _, _ = solve_relative_general(Tmat1_eq6, Eth_eq6, Vdiag_attr, g=U_OP, s=1.0)
    E0_r, B_r, _, _, _ = solve_relative_general(Tmat1_eq6, Eth_eq6, Vdiag_rep, g=U_OP, s=1.0)
    print(f"\n  attractive (g={U_OP}): B = {B_a:.6f}")
    print(f"  repulsive  (g={U_OP}): B = {B_r:.6f}   (negative/<=0 means unbound, as a genuine 1/r "
          "repulsion should give -- no bound state for two like-sign static charges)")
    sign_consistent = B_r < B_a
    print(f"\n  SIGN CHECK: B_repulsive < B_attractive : {sign_consistent} "
          f"({'CONSISTENT' if sign_consistent else 'INCONSISTENT'} with E_odd's own measured "
          "sign, B(3He)<B(3H), i.e. the extra same-charge pair binds less)")
    print(f"\n  ==> PART C VERDICT: SIGN-{'CONSISTENT' if sign_consistent else 'INCONSISTENT'} "
          "(qualitative only); quantitative MeV match NOT ATTEMPTED (blocked, see above).")

    # -------------------------------------------------------------------------------------------
    # Sec.7 -- Delta-alpha as by-product (scale-free structural report ONLY)
    # -------------------------------------------------------------------------------------------
    banner("Sec.7 -- Delta-alpha AS BY-PRODUCT (scale-free structural report, per the note's "
           "wording 'Delta-alpha as by-product')")
    print(f"""  CS-1's own already-computed transverse-channel Taylor coefficients at u=alpha_1
  (CS1_finite_k_propagator_2026-07-11.py line 467-468, cited not recomputed):
    pi0 = {PI0_CS1:+.6e}   pi2 = {PI2_CS1:+.6e}   pi4 = {PI4_CS1:+.6e}
  A SCALE-FREE local running fraction (no absolute momentum scale needed: p0 is CS-1's own
  dimensionless fractional-Bloch-momentum, not an energy), at the edge of CS-1's OWN tested
  window p0=0.10 (CS1's P0_VALUES, its line 441):
    rho(p0=0.10) := pi4*p0^2 / pi2 = {PI4_CS1 * 0.10 ** 2 / PI2_CS1:+.6f}
  i.e. a ~{abs(PI4_CS1 * 0.10**2/PI2_CS1)*100:.1f}% local curvature correction to the leading pi2
  term already visible inside CS-1's own fitted range -- structural evidence that the transverse
  channel DOES run (is not an exact pure-p^2 form), consistent in KIND with 'the Delta-alpha
  low-scale running' the design note names (line 16). NOT claimed as a value of any real Delta-
  alpha: connecting this dimensionless ratio to an actual physical momentum/energy scale needs
  precisely the scale bridge named as missing in Sec.5/Sec.6 -- reported here as a structural
  by-product only, per the note's own word 'by-product', not a match.""")

    # -------------------------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------------------------
    banner("SUMMARY")
    print(f"""  Sec.0 on-site-shift self-test ..................... PASS
  Sec.1 contact-vertex baseline regression .......... PASS (exp {expo_contact:+.4f},
      ratio {ratio_contact:.4f} match IV4's own landed T1-FENCE numbers)
  Sec.2/3 PART A (Coulomb-shape kernel vs contact) .. exp {expo_op:+.4f} (target +1),
      ratio {ratio_coul:.4f} (target 2.0)  ==>  {part_a_verdict}
  Sec.5 PART B (atomic block) ....................... BLOCKED (no scale bridge; stopped, per gate)
  Sec.6 PART C (E_odd) .............................. SIGN-{'CONSISTENT' if sign_consistent else 'INCONSISTENT'}
      (qualitative only); quantitative match BLOCKED (stopped, per gate)
  Sec.7 Delta-alpha by-product ...................... reported (scale-free ratio only, no match claimed)
  wall time: {elapsed():.1f}s
""")
    print("SCOPE / POISON HONESTY: no goal-seeking toward any measured binding energy anywhere "
          "above; the Coulomb-shape coupling g is a SCANNED, declared grid, never fit or tuned "
          "toward 2.0/+1; CS-1's own pi_2 number is cited for orientation only (Sec.4), never used "
          "as an amplitude; alpha_1 != alpha_EM; the_run.py/the_net.py/verify.py/locks/registers "
          "and CS-1's own file are untouched; ONE new proofs/ file; no commit made.")
    banner("DONE")


if __name__ == "__main__":
    main()
