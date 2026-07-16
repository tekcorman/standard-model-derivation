#!/usr/bin/env python3
"""
proofs/foundations/B3_MZ_oblique_cone_recompute_2026-07-07.py

PHASE-1 B3 (full public push roadmap): recompute the M_Z oblique two-point on the
A5(b) Cl(3,1) CONTINUUM CONE, not the lattice D3.  BLIND probe; pre-registered at
internal research notes (blinding timestamp
commit 6463e3d; construction scoping 8dd7542).  The SM target 0.3570% is consulted
only in the final block, AFTER the cone tensor is computed.

WHAT THE LATTICE DID (M_Z_BZ_integrated_vacuum_polarization_2026-06-30.py):
  substrate oblique = delta_r (Perron singlet, uniform) + chiral winding shell (anisotropic).
  delta_r        = (1/2|E|) . alpha1/(1-alpha1) = 0.3384%   (dominant; +8.1 sigma alone)
  winding shell  = <Sum w^2 . F>_BZ , F = Im(lam)/|lam|^2 on the NON-normal B(k)  (small)
  total 0.3431% vs SM 0.3570% -> M_Z +6.1 sigma.  The shell is the ONLY lattice-vs-cone
  distinct piece; delta_r is a global algebraic normalization (2|E|=12 darts, alpha1).

THE CONE OBSERVABLE (pre-reg frozen computation):  the Z-current two-point <J_Z J_Z>
integrated over the cone momentum measure = the interband current-current polarization
tensor Pi_ab of the cone Dirac H(k)=Sum k_a gD[a].  The gauge factor (T3 - s^2 Q) is a
per-species scalar -> it multiplies Pi_ab but does NOT change its TENSOR structure, so the
oblique-DEVIATION question is entirely: is Pi_ab isotropic (proportional to delta_ab, the
standard relativistic transverse form) or anisotropic (a substrate oblique deviation)?

ANALYTIC CORE (confirmed numerically below):  with band projectors
P_-(k)=(I - khat.gD)/2 (filled), P_+(k)=(I + khat.gD)/2 (empty), velocity v_a = gD[a],
    Tr[P_- gD[a] P_+ gD[b]]  =  2 (delta_ab - khat_a khat_b)      (EXACT transverse projector)
using {gD_a,gD_b}=2 delta and Tr[gc ga gd gb]=4(dca ddb - dcd dab + dcb dad).  The angular
integral of (delta_ab - khat_a khat_b) = (8pi/3) delta_ab -> Pi_ab is EXACTLY isotropic.
The anisotropic (oblique-deviation) part is ZERO by the exact emergent SO(3) that A5(b) locks.

PRE-DECLARED OUTCOME reached (see prereg): CONFIRM-FLOOR (refined) -- the cone carries NO
oblique deviation; it cannot close M_Z; the lattice +6 sigma residual STANDS as the physical
prediction (its anisotropic part = genuine substrate discreteness with no continuum analogue;
its dominant part = a global algebraic normalization).  M_Z stays OPEN.  No value moved.
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "derivation_topdown" / "bridge"))
from d4_spectral_action import a5b_dirac_cone  # noqa: E402

np.set_printoptions(precision=6, suppress=True, linewidth=140)


def band_projectors(gD, khat):
    """P_-(filled, E=-|k|) and P_+(empty, E=+|k|) for H(khat)=khat.gD (|k|=1)."""
    Hh = sum(khat[a] * gD[a] for a in range(3))
    Pm = (np.eye(4) - Hh) / 2.0
    Pp = (np.eye(4) + Hh) / 2.0
    return Pm, Pp


def current_kernel(gD, khat):
    """M_ab(khat) = Tr[P_- gD[a] P_+ gD[b]]  (the interband current-current kernel, |k|=1)."""
    Pm, Pp = band_projectors(gD, khat)
    M = np.zeros((3, 3), complex)
    for a in range(3):
        for b in range(3):
            M[a, b] = np.trace(Pm @ gD[a] @ Pp @ gD[b])
    return M


def lebedev_like_grid(n_cos=16, n_phi=32):
    """Exact-for-low-degree angular grid: Gauss-Legendre in u=cos(theta), uniform (periodic
    trapezoid, spectrally exact) in phi. Integrates k_a k_b (degree 2) to machine precision."""
    u, wu = np.polynomial.legendre.leggauss(n_cos)          # nodes in [-1,1], sum(wu)=2
    phis = (np.arange(n_phi)) * 2 * np.pi / n_phi
    wphi = 2 * np.pi / n_phi
    pts, wts = [], []
    for uk, wk in zip(u, wu):
        s = np.sqrt(max(0.0, 1 - uk * uk))
        for ph in phis:
            pts.append(np.array([s * np.cos(ph), s * np.sin(ph), uk]))
            wts.append(wk * wphi)
    return np.array(pts), np.array(wts)


def main():
    print("=" * 82)
    print("  B3: the M_Z oblique two-point on the A5(b) Cl(3,1) CONTINUUM CONE (blind)")
    print("=" * 82)

    # ---- C-CONE: the imported cone passes its own structure gate ----
    gD, weyl = a5b_dirac_cone()
    anti = max(np.max(np.abs(gD[a] @ gD[b] + gD[b] @ gD[a] - (2.0 if a == b else 0) * np.eye(4)))
               for a in range(3) for b in range(3))
    ktest = np.array([0.3, -0.7, 1.1])
    Htest = sum(ktest[a] * gD[a] for a in range(3))
    h2 = np.max(np.abs(Htest @ Htest - (ktest @ ktest) * np.eye(4)))
    print(f"\n[C-CONE] {{gD_a,gD_b}}=2delta residual = {anti:.2e} ; H^2=|k|^2 residual = {h2:.2e}")
    assert anti < 1e-9 and h2 < 1e-9, "cone structure gate FAILED"

    # ---- C-CONVERGE / normality: H(k) is Hermitian -> normal, no exceptional point ----
    conds = []
    for _ in range(200):
        v = np.random.randn(3); v /= np.linalg.norm(v)
        Hh = sum(v[a] * gD[a] for a in range(3))
        w, U = np.linalg.eigh(Hh)  # Hermitian eig
        conds.append(np.linalg.cond(U))
    print(f"[normality] max cond(eigvecs) over 200 random k-hat = {max(conds):.3e}   "
          f"(lattice non-normal B was ill-conditioned >1e9; cone is NORMAL)")
    assert max(conds) < 10, "cone eig unexpectedly ill-conditioned"

    # ---- CORE: the interband kernel M_ab(khat) = 2(delta_ab - khat_a khat_b) exactly ----
    print("\n(1) interband current-current kernel  M_ab(khat) = Tr[P_- gD_a P_+ gD_b]")
    max_dev = 0.0
    max_imag = 0.0
    for _ in range(500):
        v = np.random.randn(3); v /= np.linalg.norm(v)
        M = current_kernel(gD, v)
        analytic = 2.0 * (np.eye(3) - np.outer(v, v))
        max_dev = max(max_dev, np.max(np.abs(M.real - analytic)))
        max_imag = max(max_imag, np.max(np.abs(M.imag)))
    print(f"    max |Re M_ab - 2(delta_ab - khat_a khat_b)| over 500 dirs = {max_dev:.2e}")
    print(f"    max |Im M_ab|                                            = {max_imag:.2e}")
    print("    => M_ab is EXACTLY the transverse projector 2(delta - khat khat). Analytic confirmed.")
    assert max_dev < 1e-9 and max_imag < 1e-9

    # ---- integrate over angles: Pi_ab(shape) = INT dOmega M_ab  (radial factor is an isotropic scalar) ----
    pts, wts = lebedev_like_grid()
    Pi = np.zeros((3, 3))
    for khat, w in zip(pts, wts):
        Pi += w * current_kernel(gD, khat).real
    iso = np.trace(Pi) / 3.0
    aniso = Pi - iso * np.eye(3)
    print("\n(2) angular integral  INT dOmega M_ab  (the cone polarization tensor shape):")
    print(f"    Pi_ab =\n{Pi}")
    print(f"    isotropic part (tr/3) = {iso:.6f}   (analytic 16pi/3 = {16*np.pi/3:.6f})")
    print(f"    ||anisotropic part|| / isotropic = {np.max(np.abs(aniso))/abs(iso):.2e}")
    assert np.max(np.abs(aniso)) / abs(iso) < 1e-6, "cone polarization is NOT isotropic (unexpected)"

    # ---- basis-free cross-check: rotational invariance of the kernel under a random SO(3) ----
    from numpy.linalg import qr
    G = np.random.randn(3, 3); Rrot, _ = qr(G); Rrot *= np.sign(np.linalg.det(Rrot))
    Pi_rot = np.zeros((3, 3))
    for khat, w in zip(pts, wts):
        Pi_rot += w * current_kernel(gD, Rrot @ khat).real
    print(f"\n[C-BASIS] ||R Pi R^T - Pi|| (SO(3) invariance) = {np.max(np.abs(Rrot @ Pi @ Rrot.T - Pi)):.2e}")

    # ---- VERDICT (blind target consulted only now) ----
    two_E = 12; s2 = 0.231; alpha1 = (2/3)**8
    delta_r = (1.0/two_E) * alpha1/(1-alpha1)     # global algebraic normalization (NOT a cone-spectral object)
    SM = 0.003570                                  # <-- blind target, first look
    M_Z, sig = 91.1876, 0.0021
    print("\n" + "=" * 82)
    print("(3) VERDICT")
    print("=" * 82)
    print(f"    Cone Z-current polarization Pi_ab is EXACTLY isotropic & transverse")
    print(f"    (anisotropy {np.max(np.abs(aniso))/abs(iso):.1e}, forced by emergent SO(3)).")
    print(f"    => the cone carries NO anisotropic oblique DEVIATION. The lattice winding-shell")
    print(f"       (the only lattice-vs-cone-distinct oblique piece) has ZERO continuum analogue.")
    print(f"    => the cone eig is NORMAL (cond ~ {max(conds):.1f}); the lattice non-normal-B")
    print(f"       exceptional-point ill-conditioning does NOT occur here (outcome (c) excluded).")
    print(f"    delta_r (dominant piece) = (1/2|E|).alpha1/(1-alpha1) = {delta_r*100:.4f}%  is a GLOBAL")
    print(f"       algebraic normalization (2|E|=12 darts), NOT a cone-spectral object -> unchanged.")
    print(f"    delta_r alone -> M_Z {(SM-delta_r)*M_Z/sig:+.1f} sigma (the cone offers NO closing term).")
    print()
    print(f"    OUTCOME = CONFIRM-FLOOR (refined). The oblique residual is CONFIRMED intrinsic from an")
    print(f"    independent continuum route: its anisotropic part is genuine substrate discreteness")
    print(f"    with no continuum analogue; its dominant part is global-algebraic. The cone CANNOT")
    print(f"    close M_Z. The lattice ~4%/+6 sigma oblique residual STANDS as the physical prediction.")
    print(f"    M_Z pole stays OPEN. No value moved; nothing relabeled as artifact (the residual is a")
    print(f"    real forced substrate read -- the discrete substrate IS physical).")
    print("=" * 82)


if __name__ == "__main__":
    main()
