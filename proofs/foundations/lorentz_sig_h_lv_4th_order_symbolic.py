#!/usr/bin/env python3
"""
Symbolic 4th-order Feshbach-Löwdin / Rayleigh-Schrödinger derivation of the
dim-6 Lorentz-violation coefficients of the scalar Bloch H Perron band on srs.

Establishes THEOREM-GRADE closed-form values:

    D_H        = 1/16
    D4_iso^H   = -1/1024
    D4_aniso^H = +1/1536
    eta^H_NB   = D4_aniso^H / D_H^2 = 1/6

upgrading the existing high-precision numerical extraction
(`proofs/foundations/lorentz_sig_h_lv_coefficients.py`) from
`mathematically-complete pending symbolic` to `theorem-grade symbolic`
under the parameter_linter hard quality gate.

Method
------
Feshbach-Löwdin partition of the 4x4 Bloch Hamiltonian H(k) relative to
the non-degenerate Perron eigenstate psi_0 = (1,1,1,1)/2 at lambda_0 = 3.

    [P-block]: H_PP(k) = <psi_0 | H(k) | psi_0>            (1x1)
    [Q-block]: H_QQ(k) = Q H(k) Q                           (3x3, Q = I - P)
    [off-diag]: H_PQ(k) = <psi_0 | H(k) Q,  H_QP(k) = adj   (1x3, 3x1)

The Perron eigenvalue lambda_0(k) satisfies the exact equation

    lambda_0(k) = H_PP(k) + H_PQ(k) * [lambda_0 I_3 - H_QQ(k)]^{-1} * H_QP(k).

Setting delta(k) = lambda_0(k) - 3 and noting H_QQ(0) = -I_3 so that
[lambda_0 I_3 - H_QQ(0)]^{-1} = (1/4) I_3, we expand the resolvent as a
Neumann series around (4 + delta) I_3 - w_QQ(k) where w_QQ = H_QQ + I_3
is O(k). The fixed-point equation

    delta = p(k) + H_PQ * [(4 + delta) I_3 - w_QQ]^{-1} * H_QP

with p(k) = H_PP(k) - 3 is iterated symbolically, truncating to total
degree 4 in (k_x, k_y, k_z) at each step. Converges in <=4 iterations.

Gauge
-----
Physical-displacement gauge with exact-rational atom positions (Wyckoff
8a, x = 1/8) and BCC primitive vectors. Phases are exp(i k_cart * r_disp)
with k = (k_x, k_y, k_z) a symbolic 3-vector and r_disp the physical
nearest-neighbour displacement (an exact rational vector). This matches
the convention of `proofs/foundations/lorentz_sig_h_lv_coefficients.py`.

Cited theorems / framework grounding
------------------------------------
- A1 (binary self-inverse toggle): srs adjacency, H(k) is the scalar
  Bloch fibre on the 4-atom primitive cell.
- A3 (complex Hilbert): H(k) is a 4x4 Hermitian operator over C.
- Biggs 1993, Algebraic Graph Theory 2nd ed., Cambridge, sec. 2.2:
  spec(A_{K_4}) = {3, -1, -1, -1}; non-degenerate Perron at +3.
- Kato 1980, Perturbation Theory for Linear Operators 2nd ed.,
  Springer Grundlehren 132, sec. II.4 + II.5: Feshbach-Löwdin formula
  and Rayleigh-Schrödinger expansion for non-degenerate eigenvalues.
- Sunada 2012, Topological Crystallography, Springer: srs identification
  as the maximum-symmetry 3-regular crystal net (input atom positions).

Cross-checks
------------
- D_H = 1/16 matches `predictions/srs_bloch_dispersion_gamma.py` (Kato + S3,
  already symbolic).
- D4_iso^H = -1/1024 and D4_aniso^H = +1/1536 reproduce the 25+ digit
  numerical extraction of `lorentz_sig_h_lv_coefficients.py` exactly.
- eta^H_NB = 1/6 confirms `predictions/srs_bloch_lv_dim6.py`.
- Ihara cross-walker theorem (`lorentz_sig_ihara_lv_relation.py`) then
  lifts D_NB = 1/8, D4_aniso^NB = 1/768, D4_iso^NB = 3/512,
  eta_NB = 1/12 to theorem-grade as algebraic corollaries.
"""

from __future__ import annotations
import sys
import time
from fractions import Fraction

import sympy as sp


# -----------------------------------------------------------------------------
# 1. Geometry: exact-rational Wyckoff 8a atom positions and BCC primitive vecs
#    (matches proofs/foundations/lorentz_sig_h_lv_coefficients.py and
#    proofs/lorentz/hashimoto_dispersion_symbolic.py).
# -----------------------------------------------------------------------------
ATOMS_EXACT = [
    (sp.Rational(1, 8), sp.Rational(1, 8), sp.Rational(1, 8)),
    (sp.Rational(3, 8), sp.Rational(7, 8), sp.Rational(5, 8)),
    (sp.Rational(7, 8), sp.Rational(5, 8), sp.Rational(3, 8)),
    (sp.Rational(5, 8), sp.Rational(3, 8), sp.Rational(7, 8)),
]
A_PRIM_EXACT = [
    (sp.Rational(-1, 2), sp.Rational(1, 2), sp.Rational(1, 2)),
    (sp.Rational(1, 2), sp.Rational(-1, 2), sp.Rational(1, 2)),
    (sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(-1, 2)),
]
N_ATOMS = 4
NN_DIST_SQ_EXACT = sp.Rational(1, 8)  # (sqrt(2)/4)^2 = 2/16 = 1/8


def find_bonds_exact():
    """
    Identify NN bonds in the primitive cell by exact-rational distance check.
    Returns list of (src, tgt, (n1, n2, n3)) tuples (12 directed bonds).
    """
    bonds = []
    for i in range(N_ATOMS):
        ri = ATOMS_EXACT[i]
        for j in range(N_ATOMS):
            for n1 in range(-2, 3):
                for n2 in range(-2, 3):
                    for n3 in range(-2, 3):
                        rj_x = ATOMS_EXACT[j][0] + n1*A_PRIM_EXACT[0][0] + n2*A_PRIM_EXACT[1][0] + n3*A_PRIM_EXACT[2][0]
                        rj_y = ATOMS_EXACT[j][1] + n1*A_PRIM_EXACT[0][1] + n2*A_PRIM_EXACT[1][1] + n3*A_PRIM_EXACT[2][1]
                        rj_z = ATOMS_EXACT[j][2] + n1*A_PRIM_EXACT[0][2] + n2*A_PRIM_EXACT[1][2] + n3*A_PRIM_EXACT[2][2]
                        dx = rj_x - ri[0]
                        dy = rj_y - ri[1]
                        dz = rj_z - ri[2]
                        d_sq = dx*dx + dy*dy + dz*dz
                        if d_sq == 0:
                            continue
                        if d_sq == NN_DIST_SQ_EXACT:
                            bonds.append((i, j, (n1, n2, n3)))
    # Each atom should have exactly 3 NN.
    counts = [0] * N_ATOMS
    for s, _, _ in bonds:
        counts[s] += 1
    assert all(c == 3 for c in counts), f"Bad NN count: {counts}"
    assert len(bonds) == 12, f"Expected 12 directed bonds; got {len(bonds)}"
    return bonds


def disp_exact(src, tgt, cell):
    """Exact rational displacement r_tgt + cell*A_PRIM - r_src."""
    out = []
    for d in range(3):
        v = ATOMS_EXACT[tgt][d] - ATOMS_EXACT[src][d]
        for i in range(3):
            v += cell[i] * A_PRIM_EXACT[i][d]
        out.append(v)
    return tuple(out)


# -----------------------------------------------------------------------------
# 2. Symbolic Bloch Hamiltonian, Taylor-truncated to total degree 4 in k.
# -----------------------------------------------------------------------------

def truncate_total_degree(expr, vars_, max_total_degree):
    """Truncate polynomial expression to total degree <= max_total_degree in vars_."""
    expr = sp.expand(expr)
    if expr == 0:
        return sp.S.Zero
    # Use sp.Poly for fast monomial enumeration.
    poly = sp.Poly(expr, *vars_)
    out = sp.S.Zero
    for monom, coef in poly.terms():
        if sum(monom) <= max_total_degree:
            term = coef
            for i, v in enumerate(vars_):
                if monom[i] > 0:
                    term *= v ** monom[i]
            out += term
    return sp.expand(out)


def truncate_matrix(M, vars_, max_total_degree):
    rows, cols = M.shape
    out = sp.zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = truncate_total_degree(M[i, j], vars_, max_total_degree)
    return out


def build_bloch_H_taylor(kx, ky, kz, bonds, max_order):
    """
    4x4 H(k) with each entry truncated to total degree <= max_order in (kx,ky,kz).
    Phase exp(i k * r_disp) -> sum_{m=0}^{max_order} (i k.r)^m / m!.
    """
    H = sp.zeros(N_ATOMS, N_ATOMS)
    I_unit = sp.I
    vars_ = (kx, ky, kz)
    for src, tgt, cell in bonds:
        r = disp_exact(src, tgt, cell)
        kr = r[0] * kx + r[1] * ky + r[2] * kz
        # Truncate exp(i kr) to total degree max_order in (kx,ky,kz).
        phase = sp.S.Zero
        ikr_pow = sp.S.One
        fact = 1
        for m in range(max_order + 1):
            phase += ikr_pow / fact
            ikr_pow = sp.expand(ikr_pow * I_unit * kr)
            ikr_pow = truncate_total_degree(ikr_pow, vars_, max_order)
            fact *= (m + 1)
        H[tgt, src] += phase
    H = truncate_matrix(H, vars_, max_order)
    return H


# -----------------------------------------------------------------------------
# 3. Block decomposition + Feshbach-Löwdin iteration.
# -----------------------------------------------------------------------------

def perron_basis():
    """Orthonormal {psi_0, g_1, g_2, g_3} basis with psi_0 = (1,1,1,1)/2."""
    psi0 = sp.Matrix([sp.Rational(1, 2)] * 4)  # 4x1
    g1 = sp.Matrix([1, -1, 0, 0]) / sp.sqrt(2)
    g2 = sp.Matrix([1, 1, -2, 0]) / sp.sqrt(6)
    g3 = sp.Matrix([1, 1, 1, -3]) / sp.sqrt(12)
    G = sp.Matrix.hstack(g1, g2, g3)  # 4x3
    U = sp.Matrix.hstack(psi0, G)     # 4x4 unitary
    return psi0, G, U


def feshbach_lowdin(H, vars_, max_order):
    """
    Compute lambda_0(k) - 3 to total degree max_order via Feshbach-Löwdin.

    H must be 4x4 Hermitian, expressed as polynomials in vars_ truncated to
    total degree max_order.

    Iterates the fixed-point equation
        delta = p(k) + H_PQ * [(4 + delta) I_3 - w_QQ]^{-1} * H_QP
    where p = H_PP - 3 and w_QQ = H_QQ + I_3.
    """
    psi0, G, U = perron_basis()
    Hb = U.H * H * U
    Hb = truncate_matrix(Hb, vars_, max_order)

    H_PP = Hb[0, 0]                    # scalar
    H_PQ = Hb[0:1, 1:4]                # 1x3
    H_QP = Hb[1:4, 0:1]                # 3x1
    H_QQ = Hb[1:4, 1:4]                # 3x3

    p_k = sp.expand(H_PP - 3)
    p_k = truncate_total_degree(p_k, vars_, max_order)
    w_QQ = H_QQ + sp.eye(3)
    w_QQ = truncate_matrix(w_QQ, vars_, max_order)

    I3 = sp.eye(3)
    delta = sp.S.Zero
    # Iterate fixed-point equation.  Each iteration converges one more order
    # of accuracy.  Five iterations suffice for max_order = 4.
    for it in range(6):
        # Build resolvent M = [(4 + delta) I - w_QQ]^{-1}
        # = (1/(4+delta)) * sum_{n=0}^infty (w_QQ / (4+delta))^n
        # Truncate Neumann series to enough terms that w_QQ^n is at order
        # > max_order in k.  Since w_QQ is at lowest order k (linear), we
        # need n up to max_order.  Take n = 0..max_order.
        # Expand 1/(4+delta) in powers of delta. delta starts at order k^2,
        # so for max_order = 4 we need delta up to second power.
        max_delta_power = max_order // 2 + 1
        prefactor = sp.S.Zero
        for m in range(max_delta_power + 1):
            prefactor += sp.Rational(1, 4) * (-delta / 4) ** m
        prefactor = truncate_total_degree(sp.expand(prefactor), vars_, max_order)

        M = sp.zeros(3, 3)
        wp = I3
        for n in range(max_order + 1):
            # Term: prefactor^(n+1) * w_QQ^n
            coef = prefactor ** (n + 1)
            coef = truncate_total_degree(sp.expand(coef), vars_, max_order)
            term = coef * wp
            term = truncate_matrix(term, vars_, max_order)
            M += term
            # advance wp <- wp * w_QQ for next iteration
            wp = wp * w_QQ
            wp = truncate_matrix(wp, vars_, max_order)
            # Early exit if wp becomes zero or pure-higher-order.
            if all(wp[i, j] == 0 for i in range(3) for j in range(3)):
                break
        M = truncate_matrix(M, vars_, max_order)

        sandwich = (H_PQ * M * H_QP)[0, 0]
        sandwich = truncate_total_degree(sp.expand(sandwich), vars_, max_order)

        new_delta = truncate_total_degree(sp.expand(p_k + sandwich), vars_, max_order)

        if sp.expand(new_delta - delta) == 0:
            return new_delta, it + 1
        delta = new_delta
    return delta, max_order + 2  # didn't converge in budget; return anyway


# -----------------------------------------------------------------------------
# 4. Coefficient extraction in Cartesian (kx, ky, kz).
# -----------------------------------------------------------------------------

def extract_lv_coefficients(delta_expr, kx, ky, kz):
    """
    Given delta(kx, ky, kz) = lambda_0 - 3 truncated to total degree 4,
    extract D_H, D4_iso^H, D4_aniso^H from the convention
        delta = -D_H |k|^2 - D4_iso |k|^4 - D4_aniso (kx^4 + ky^4 + kz^4) + O(k^6)
    where |k|^2 = kx^2 + ky^2 + kz^2.

    Equivalently the |k|^4 part decomposes as
        -(D4_iso + D4_aniso * f4(khat)) * |k|^4  with  f4(khat) = (kx^4+ky^4+kz^4)/|k|^4.
    """
    delta_expr = sp.expand(delta_expr)
    poly = sp.Poly(delta_expr, kx, ky, kz)
    # Verify there is no O(k) or O(k^3) term (must vanish by symmetry).
    for monom, coef in poly.terms():
        deg = sum(monom)
        if deg in (1, 3):
            assert coef == 0, f"Nonzero odd-degree term {monom}: {coef}"
    # O(k^2): coefficient of k_i k_j must satisfy
    #   k_x^2: -D_H,  k_y^2: -D_H,  k_z^2: -D_H,  k_i k_j (i!=j): 0
    cxx = poly.coeff_monomial(kx**2)
    cyy = poly.coeff_monomial(ky**2)
    czz = poly.coeff_monomial(kz**2)
    cxy = poly.coeff_monomial(kx * ky)
    cxz = poly.coeff_monomial(kx * kz)
    cyz = poly.coeff_monomial(ky * kz)
    assert cxx == cyy == czz, f"Anisotropic O(k^2): {cxx}, {cyy}, {czz}"
    assert cxy == 0 and cxz == 0 and cyz == 0, \
        f"Cross O(k^2) terms nonzero: {cxy}, {cxz}, {cyz}"
    D_H = -cxx
    # O(k^4): general cubic-symmetric form is
    #   alpha (kx^2+ky^2+kz^2)^2 + beta (kx^4+ky^4+kz^4)
    # Coeff of kx^4   = alpha + beta
    # Coeff of kx^2 ky^2 = 2 alpha
    # By cubic symmetry we expect kx^4 = ky^4 = kz^4 and ki^2 kj^2 (i!=j) all equal.
    cxxxx = poly.coeff_monomial(kx**4)
    cyyyy = poly.coeff_monomial(ky**4)
    czzzz = poly.coeff_monomial(kz**4)
    cxxyy = poly.coeff_monomial(kx**2 * ky**2)
    cxxzz = poly.coeff_monomial(kx**2 * kz**2)
    cyyzz = poly.coeff_monomial(ky**2 * kz**2)
    assert cxxxx == cyyyy == czzzz, f"O(k^4) ki^4 anisotropic: {cxxxx}, {cyyyy}, {czzzz}"
    assert cxxyy == cxxzz == cyyzz, f"O(k^4) ki^2 kj^2 anisotropic"
    # All odd k_i k_j k_k k_l terms must vanish by inversion + cubic symmetry
    for monom, coef in poly.terms():
        if sum(monom) == 4:
            # Acceptable: ki^4 or ki^2 kj^2 with no other free indices
            counts_per_axis = monom  # (n_x, n_y, n_z)
            if all(n in (0, 2, 4) for n in counts_per_axis):
                continue
            assert coef == 0, f"Disallowed quartic term {monom}: {coef}"
    alpha = sp.Rational(1, 2) * cxxyy
    beta = cxxxx - alpha
    # delta = ... - D4_iso |k|^4 - D4_aniso (kx^4+ky^4+kz^4)
    # so alpha = -D4_iso, beta = -D4_aniso
    D4_iso = -alpha
    D4_aniso = -beta
    return D_H, D4_iso, D4_aniso


# -----------------------------------------------------------------------------
# 5. Driver.
# -----------------------------------------------------------------------------

def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main():
    header("Symbolic 4th-order Feshbach-Löwdin / Rayleigh-Schrödinger derivation")
    print("  Target: D_H, D4_iso^H, D4_aniso^H of srs scalar Bloch Perron band")
    print("  Method: Feshbach-Löwdin partition relative to Perron eigenstate")
    print("  Gauge:  physical-displacement (Wyckoff 8a + BCC primitive vectors)")

    t0 = time.time()
    bonds = find_bonds_exact()
    print(f"\n  bonds: {len(bonds)} directed (expected 12)")
    print(f"  bonds = {bonds}")

    kx, ky, kz = sp.symbols('kx ky kz', real=True)
    vars_ = (kx, ky, kz)

    header("Step 1: build Taylor-truncated Bloch H(k) to total degree 4")
    H = build_bloch_H_taylor(kx, ky, kz, bonds, max_order=4)
    t1 = time.time()
    print(f"  built in {t1 - t0:.2f}s")
    # Sanity: H(0) should equal K_4 adjacency matrix.
    H0 = H.subs({kx: 0, ky: 0, kz: 0})
    H0 = sp.simplify(H0)
    K4_adj = sp.ones(4, 4) - sp.eye(4)
    assert H0 == K4_adj, f"H(0) != K_4 adjacency: H(0) = {H0}"
    print(f"  sanity: H(0) = K_4 adjacency  OK")

    header("Step 2: Feshbach-Löwdin iteration -> delta(k) = lambda_0(k) - 3")
    delta_expr, n_iter = feshbach_lowdin(H, vars_, max_order=4)
    t2 = time.time()
    print(f"  converged in {n_iter} iterations, {t2 - t1:.2f}s")

    header("Step 3: extract LV coefficients in Cartesian (kx, ky, kz)")
    print(f"  delta(k) = {sp.simplify(delta_expr)}")

    D_H, D4_iso_H, D4_aniso_H = extract_lv_coefficients(delta_expr, kx, ky, kz)

    print(f"\n  D_H        = {D_H}")
    print(f"  D4_iso^H   = {D4_iso_H}")
    print(f"  D4_aniso^H = {D4_aniso_H}")

    eta_H_NB = D4_aniso_H / (D_H ** 2)
    eta_H_NB = sp.simplify(eta_H_NB)
    print(f"  eta^H_NB   = D4_aniso^H / D_H^2 = {eta_H_NB}")

    header("Step 4: assert exact-rational targets")
    expected_D_H = sp.Rational(1, 16)
    expected_D4_iso = sp.Rational(-1, 1024)
    expected_D4_aniso = sp.Rational(1, 1536)
    expected_eta = sp.Rational(1, 6)

    print(f"  D_H        : {D_H}  vs  expected 1/16        ", "OK" if D_H == expected_D_H else "FAIL")
    print(f"  D4_iso^H   : {D4_iso_H}  vs  expected -1/1024 ", "OK" if D4_iso_H == expected_D4_iso else "FAIL")
    print(f"  D4_aniso^H : {D4_aniso_H}  vs  expected 1/1536 ", "OK" if D4_aniso_H == expected_D4_aniso else "FAIL")
    print(f"  eta^H_NB   : {eta_H_NB}  vs  expected 1/6      ", "OK" if eta_H_NB == expected_eta else "FAIL")

    assert D_H == expected_D_H, f"D_H mismatch"
    assert D4_iso_H == expected_D4_iso, f"D4_iso^H mismatch"
    assert D4_aniso_H == expected_D4_aniso, f"D4_aniso^H mismatch"
    assert eta_H_NB == expected_eta, f"eta^H_NB mismatch"

    header("Step 5: Ihara cross-walker corollaries (algebraic)")
    # h(lambda) = (lambda + sqrt(lambda^2 - 8))/2,  h(3)=2, h'(3)=2, h''(3)=-4
    # D_NB           = h'(3) D_H                = 2 * (1/16) = 1/8
    # D4_aniso^NB    = h'(3) D4_aniso^H          = 2 * (1/1536) = 1/768
    # D4_iso^NB      = h'(3) D4_iso^H - (1/2) h''(3) D_H^2
    #                = 2 * (-1/1024) - (1/2)*(-4)*(1/16)^2
    #                = -1/512 + 1/128 = +3/512
    # eta_NB         = D4_aniso^NB / D_NB^2     = (1/768) / (1/8)^2 = 1/12
    h_p_3 = sp.Integer(2)
    h_pp_3 = sp.Integer(-4)
    D_NB = h_p_3 * D_H
    D4_aniso_NB = h_p_3 * D4_aniso_H
    D4_iso_NB = h_p_3 * D4_iso_H - sp.Rational(1, 2) * h_pp_3 * D_H**2
    eta_NB = D4_aniso_NB / D_NB**2
    print(f"  D_NB        = h'(3) D_H            = {D_NB}    (expected 1/8)")
    print(f"  D4_aniso^NB = h'(3) D4_aniso^H      = {D4_aniso_NB}   (expected 1/768)")
    print(f"  D4_iso^NB   = h'(3) D4_iso^H - h''(3)/2 D_H^2 = {D4_iso_NB} (expected 3/512)")
    print(f"  eta_NB      = D4_aniso^NB / D_NB^2   = {eta_NB}    (expected 1/12)")
    assert D_NB == sp.Rational(1, 8)
    assert D4_aniso_NB == sp.Rational(1, 768)
    assert D4_iso_NB == sp.Rational(3, 512)
    assert eta_NB == sp.Rational(1, 12)

    header("THEOREM VERIFIED")
    print("  All four scalar-Bloch coefficients match closed-form rationals.")
    print("  Ihara cross-walker theorem then forces all four Hashimoto coefficients.")
    print(f"\n  Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
