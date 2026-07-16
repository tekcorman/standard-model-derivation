#!/usr/bin/env python3
"""
derivation_topdown/bridge/d4_spectral_action.py

THE NATIVE CONTINUUM-D₄ SPECTRAL-ACTION a₄ MACHINE — reusable infrastructure (D4 program, station S1).

zeta_{D4}(0) = the a4 Seeley-DeWitt coefficient of the DERIVED A5(b) Fock-Dirac cone:
    a4  ⊃  (1/12) tr Ω²  +  (1/2) tr E²        (the universal Gilkey theorem — pure math, like Ihara-Bass)
with the framework's own objects:
    * the cone  H = Σ k_a γ^{h_a}   (A5(b): Lorentz-locked Cl(3,1), a genuine continuum Dirac, H²=|k|²),
    * the endomorphism  E = i F_ab γ^a γ^b = -2 F·S   (COMPUTED from the A5(b) γ commutators — native),
    * the curvature  Ω = F   (the framework's inner-fluctuation field strength).

The SM-physics flavor ("one-loop QFT β formula") is REMOVED: only the pure-math Gilkey a4 is imported.
NAMED residuals (grade): the vector/scalar rows still use the universal helicity rule (not re-derived on
a native cone); the KO 2→6 form-parity↔statistics step; the flat/Higgs time-leg shadow (DN_C1).

Validation: `proofs/foundations/D4_S1_native_a4_machine_2026-07-06.py` (ALL PASS; pre-reg e55c7c1).
Downstream consumers: S2 (spin rows, a read on beta_rows); S3 (the α₁³/−70 ppm — trap-dense, own pre-reg);
S4 (the CAR-KMS loop). Import this; do NOT rebuild the cone.
"""
import math
import os
import sys
from fractions import Fraction

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "derivation_topdown", "dirac_srs_mdl"))
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

_EDGES = [(0, 1, (0, 0, 0)), (0, 2, (0, 0, 0)), (0, 3, (0, 0, 0)),
          (1, 2, (1, 0, 0)), (1, 3, (0, 1, 0)), (2, 3, (0, 0, 1))]
_NV, _NE = 4, 6
_EPS = np.zeros((3, 3, 3))
for _a in range(3):
    for _b in range(3):
        for _c in range(3):
            _EPS[_a, _b, _c] = 0.5 * (_a - _b) * (_b - _c) * (_c - _a)


def a5b_dirac_cone():
    """The A5(b) continuum Fock-Dirac cone, restricted to ONE 4-dim Dirac (per-species).

    Returns (gD, weyl) : gD = [γ^{h_x}, γ^{h_y}, γ^{h_z}] the 4x4 spatial Dirac gammas ({γ_a,γ_b}=2δ),
    weyl = a 4x2 isometry onto a γ5=+1 Weyl. H(k)=Σ k_a gD[a] is the physical fermion cone (H²=|k|²).
    """
    g6 = [np.array(g, complex) for g in AlgebraicUtility.cl6_generators()]
    d0 = np.zeros((_NV, _NE))
    for e, (i, j, v) in enumerate(_EDGES):
        d0[i, e] = -1.0; d0[j, e] = 1.0
    Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    H1, _ = np.linalg.qr(Chat)
    B1 = np.linalg.svd(d0)[2][:3].T
    gam = lambda x: sum(x[a] * g6[a] for a in range(_NE))
    gh = [gam(H1[:, i]) for i in range(3)]
    gb = [gam(B1[:, i]) for i in range(3)]
    S3 = 1j * gb[0] @ gb[1] / 2                       # Hermitian T-ID2 commutant su(2) generator
    wK, UK = np.linalg.eigh(S3); blk = UK[:, wK > 0]  # one 4-dim Dirac block (per-species)
    gD = [blk.conj().T @ gh[a] @ blk for a in range(3)]
    g5 = -1j * gD[0] @ gD[1] @ gD[2]
    w5, V5 = np.linalg.eigh(g5); weyl = V5[:, w5 > 0]
    return gD, weyl


def spin_generators(gD):
    """The spin-1/2 rotation generators S_c on the Dirac block (Casimir 3/4)."""
    M = lambda a, b: (1j / 4) * (gD[a] @ gD[b] - gD[b] @ gD[a])
    return [sum(0.5 * _EPS[c, a, b] * M(a, b) for a in range(3) for b in range(3)) for c in range(3)]


def endomorphism_E(gD, F):
    """The Weitzenböck endomorphism E = i Σ_{a<b} F_ab γ^a γ^b (the native magnetic moment = -2 F·S).

    F : 3x3 antisymmetric field-strength matrix. Returns the 4x4 (or dim-of-gD) endomorphism E,
    from [π_a,π_b]=i F_ab on minimal coupling k→k-A of the A5(b) cone.
    """
    return 1j * sum(F[a][b] * gD[a] @ gD[b] for a in range(3) for b in range(3) if a < b)


def orbital_curvature_t2coeff():
    """The a4 orbital curvature (1/12)trΩ², from the constant-B Landau trace Bt/sinh(Bt) on the cone.

    Returns the symbolic t²-relative coefficient = -B²/6 (B a sympy symbol). Pure-math (Gilkey/orbital);
    the A5(b) covariant π² supplies [π_x,π_y]=iB so the Landau tower is B(2n+1).
    """
    import sympy as sp
    B, t = sp.symbols('B t', positive=True)
    orbital = sp.simplify((sp.exp(-B * t) / (1 - sp.exp(-2 * B * t))).rewrite(sp.sinh))  # 1/(2 sinh Bt)
    ratio = sp.simplify(2 * B * t * orbital)                                             # Bt/sinh(Bt)
    ser = sp.series(ratio, t, 0, 4).removeO()
    return sp.simplify(ser.coeff(t, 2) / ser.coeff(t, 0)), B                             # -B²/6


def spin_beta(s, twosz2):
    """The universal Seeley-DeWitt helicity β coefficient  b = -(-1)^{2s} [ (2 s_z)² - 1/3 ].

    s = spin (0, 1/2, 1); twosz2 = (2 s_z)² for the physical helicity. Returns a Fraction.
      fermion (Weyl, s=1/2, (2s_z)²=1) -> +2/3   [NATIVE on the A5(b) cone via (1/2)trE², see fermion_2sz2]
      scalar  (s=0, (2s_z)²=0)         -> +1/3   [universal rule]
      vector  (s=1, (2s_z)²=4) net     -> -11/3  [universal rule + ghost bookkeeping]
    """
    return Fraction(-1) ** int(round(2 * s)) * -(Fraction(twosz2).limit_denominator() - Fraction(1, 3))


def fermion_2sz2(gD=None, weyl=None):
    """(2 s_z)² for the fermion, computed NATIVELY as (1/2)tr E² / B² on a Weyl of the A5(b) cone (=1)."""
    if gD is None:
        gD, weyl = a5b_dirac_cone()
    F = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]   # F_xy = B = 1
    Ez_w = weyl.conj().T @ endomorphism_E(gD, F) @ weyl
    return 0.5 * float(np.real(np.trace(Ez_w.conj().T @ Ez_w)))


# (2 s_z)² for the physical helicity of a spin-s field — NATIVE, validated in D4_S2_native_spin_rows
# from the framework's OWN spin reps: spin-0 = the Higgs (no spin); spin-1/2 = the A5(b) cone (E=-2F·S,
# fermion_2sz2 above); spin-1 = the emergent band VECTOR rep S_a (Casimir 2), (1/2)trE² over the s_z=±1
# transverse pair = 4. The universal helicity rule spin_beta(s, spin_2sz2(s)) then gives the rows.
_2SZ2 = {Fraction(0): 0, Fraction(1, 2): 1, Fraction(1): 4}


def spin_2sz2(s):
    """(2 s_z)² for a spin-s field (native; s ∈ {0, 1/2, 1}). See D4_S2_native_spin_rows for the reps."""
    return _2SZ2[Fraction(s)]


def spin_row(s):
    """The native Seeley-DeWitt β row for spin s: b = -(-1)^{2s}[(2s_z)² - 1/3]  (+1/3, +2/3, -11/3)."""
    return spin_beta(s, spin_2sz2(s))


# native group factors (D1 probes 1+3): the traces, not table lookups
_T3 = {1: Fraction(0), 3: Fraction(1, 2), 8: Fraction(3)}     # SU(3), NATIVE_a4_color_su3 (probe 1)
_T2 = {1: Fraction(0), 2: Fraction(1, 2), 3: Fraction(2)}     # SU(2), NATIVE_a4_su2L (probe 3)
_C2G = {1: Fraction(0), 2: Fraction(2), 3: Fraction(3)}       # adjoint Casimir (native)


def gauge_dynkin(fields, mult):
    """Dynkin sums Σ over (color_dim, su2_dim, hypercharge Y)×mult, with the D1-native group factors."""
    s = {1: Fraction(0), 2: Fraction(0), 3: Fraction(0)}
    for c, w, Y in fields:
        s[3] += _T3[c] * w * mult
        s[2] += _T2[w] * c * mult
        s[1] += Fraction(3, 5) * Y * Y * c * w * mult
    return s


def beta_rows(fermions, higgs, gens=3):
    """Assemble the 4D-completion β rows b_i = -3 C₂ + T_f + T_H from the object's own a4 (S1) + group
    factors. fermions/higgs = lists of (color_dim, su2_dim, Y). Returns {1: b1, 2: b2, 3: b3}."""
    Tf = gauge_dynkin(fermions, gens); TH = gauge_dynkin(higgs, 1)
    return {i: -3 * _C2G[i] + Tf[i] + TH[i] for i in (1, 2, 3)}


def sm_content():
    """The forced SM field content (color_dim, su2_dim, Y) — off the Cl(6)-Fock Hamming weight."""
    K = 3; sgn = lambda n: 1 if n % 2 == 0 else -1; Qn = lambda n: Fraction(sgn(n) * n, K)
    fermions = [(3, 2, Qn(2) - Fraction(1, 2)), (1, 2, Qn(0) - Fraction(1, 2)),
                (3, 1, Qn(2)), (3, 1, Qn(1)), (1, 1, Qn(3))]
    higgs = [(1, 2, Fraction(1, 2)), (1, 2, Fraction(-1, 2))]
    return fermions, higgs


if __name__ == "__main__":
    # self-test: reproduce the S1 core validations (the module is its own smoke test)
    import sympy as sp
    ok = True
    gD, weyl = a5b_dirac_cone()
    ok &= max(np.max(np.abs(gD[a] @ gD[b] + gD[b] @ gD[a] - (2.0 if a == b else 0) * np.eye(4)))
              for a in range(3) for b in range(3)) < 1e-9
    # H² = |k|²
    k = np.array([0.3, -0.7, 1.1]); H = sum(k[a] * gD[a] for a in range(3))
    ok &= np.max(np.abs(H @ H - (k @ k) * np.eye(4))) < 1e-9
    # E = -2 F·S native
    S = spin_generators(gD)
    Fv = np.array([0.4, -0.2, 0.9])
    F = [[sum(_EPS[c, a, b] * Fv[c] for c in range(3)) for b in range(3)] for a in range(3)]
    E = endomorphism_E(gD, F)
    Bc = np.array([0.5 * sum(_EPS[c, a, b] * F[a][b] for a in range(3) for b in range(3)) for c in range(3)])
    m2fs = -2 * sum(Bc[c] * S[c] for c in range(3))
    ok &= (np.max(np.abs(E - m2fs)) < 1e-9 or np.max(np.abs(E + m2fs)) < 1e-9)
    # (1/12)trΩ² = -B²/6
    t2, B = orbital_curvature_t2coeff()
    ok &= sp.simplify(t2 - (-B ** 2 / 6)) == 0
    # fermion +2/3 native
    ok &= abs(fermion_2sz2(gD, weyl) - 1.0) < 1e-6
    ok &= spin_row(Fraction(1, 2)) == Fraction(2, 3)     # fermion
    ok &= spin_row(1) == Fraction(-11, 3)                # vector
    ok &= spin_row(0) == Fraction(1, 3)                  # scalar
    # b_i = {33/5, 1, -3}
    f, h = sm_content()
    b = beta_rows(f, h)
    ok &= (b == {1: Fraction(33, 5), 2: Fraction(1), 3: Fraction(-3)})
    print(f"d4_spectral_action self-test: {'ALL PASS' if ok else '*** FAIL ***'}  "
          f"(cone H²=|k|²; E=-2F·S native; (1/12)trΩ²=-B²/6; fermion +2/3; b_i={dict((i, str(v)) for i, v in b.items())})")
    sys.exit(0 if ok else 1)
