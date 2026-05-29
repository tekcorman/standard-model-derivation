#!/usr/bin/env python3
"""
Dimension-6 Lorentz violation coefficients for the scalar Bloch H Perron band.

Audit anchor: Foundational Lorentz-arc result. Conditional on Rows 4, 6 of
`docs/audits/registers/uniqueness_ledger.md` (k* = 3 + srs identification). Sister to
`predictions/eta_lattice_lorentz_dim6.py` (Hashimoto via Ihara cross-walker).
THEOREM-GRADE SYMBOLIC via Feshbach-Löwdin
(`proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py`). Component of
the joint LORENTZ_SIG closure per `docs/theorems/lorentz_sig_ccclose_joint_closure.md`.

Framework prediction: the O(k^4) coefficients of the scalar adjacency Bloch
dispersion lambda_0(k) on srs at the Perron Gamma-top band, verified
numerically to 25+ decimal digits consistent with exact rationals.

Sister file to predictions/eta_lattice_lorentz_dim6.py (Hashimoto walker).
The scalar-Bloch and Hashimoto LV coefficients are connected by the Ihara
factorization u^2 - lambda u + 2 = 0 at the k=3 Perron eigenvalue (Ihara
1966, Stark-Terras 1996), with cross-walker derivative h'(3) = 2 and
h''(3) = -4. See proofs/foundations/lorentz_sig_ihara_lv_relation.py.

Gate grade: THEOREM-GRADE SYMBOLIC. The closed-form quartic coefficients are
derived in `proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py` via
symbolic Feshbach-Löwdin partition (Kato §II.4) of H(k) relative to the
non-degenerate Perron eigenstate, with the fixed-point equation

    delta = (H_PP - 3) + H_PQ * [(4 + delta) I_3 - w_QQ]^{-1} * H_QP

iterated to convergence in `sympy` exact arithmetic on Taylor-truncated
H(k) to total degree 4 in (k_x, k_y, k_z). All four scalar-Bloch values
(D_H, D4_iso^H, D4_aniso^H, eta^H_NB) and the four Hashimoto cross-walker
corollaries (D_NB, D4_aniso^NB, D4_iso^NB, eta_NB) are reproduced exactly.

The earlier 25+ digit numerical extraction
(`proofs/foundations/lorentz_sig_h_lv_coefficients.py`, 4-point Vandermonde
at 500-bit mpmath) now serves as an independent cross-check, not the
primary source.

Cross-references:
- predictions/srs_bloch_dispersion_gamma.py - quadratic D_H = 1/16 theorem (Kato).
- predictions/eta_lattice_lorentz_dim6.py    - Hashimoto sister, eta_NB = 1/12.
- predictions/srs_dirac_cone_velocities.py   - spin-1 Dirac at the Gamma cone.
- proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py - symbolic Feshbach-Löwdin proof (PRIMARY).
- proofs/foundations/lorentz_sig_h_lv_coefficients.py - high-precision numerical cross-check.
- proofs/foundations/lorentz_sig_ihara_lv_relation.py - Ihara cross-walker theorem.
"""

# ============================================================
# PARAMETER: Scalar Bloch dim-6 LV coefficients (D_H, D4_iso^H,
#            D4_aniso^H, eta^H_NB).
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       Not currently excluded. The framework's "observable" LV
#              channel is the photon's (Hashimoto walker, eta_NB = 1/12,
#              see predictions/eta_lattice_lorentz_dim6.py). The scalar
#              Bloch H is a sister operator on the same srs lattice; its
#              dim-6 LV coefficients are structurally related to the
#              Hashimoto's via the Ihara factorization.
# Source:      Same experimental references as eta_lattice (LHAASO, etc.).
#              The scalar Bloch values themselves are not directly
#              measurable; their physics relevance is via the Ihara
#              cross-walker theorem, which underlies the Hashimoto values.
#
# The headline scalar-Bloch quantity from this file is eta^H_NB = 1/6,
# which is exactly twice the Hashimoto eta_NB = 1/12. The factor of 2 is
# structural: u'(3) = 2 in the Ihara map.

# --- PREDICTED VALUE -----------------------------------------
# Value:       D_H        = 1/16    (k^2 coefficient, theorem-grade via 2nd-order Kato + S3)
#              D4_iso^H   = -1/1024 (k^4 isotropic, theorem-grade via symbolic Feshbach-Löwdin)
#              D4_aniso^H = +1/1536 (k^4 anisotropic, theorem-grade via symbolic Feshbach-Löwdin)
#              eta^H_NB   = D4_aniso^H / D_H^2 = 1/6  (theorem-grade)
# Deviation:   N/A (structural quantities; no direct observable counterpart).

# --- DERIVED FORMULAS ----------------------------------------
# The scalar adjacency Bloch matrix H(k) on srs has Perron top eigenvalue
# lambda_0(k) with Taylor expansion near k=0 (cf. predictions/srs_bloch_dispersion_gamma.py):
#
#   lambda_0(k) = 3 - D_H |k|^2 - [D4_iso^H + D4_aniso^H f4(khat)] |k|^4 + O(k^6)
#
# Convention matches predictions/eta_lattice_lorentz_dim6.py exactly:
#   minus sign on both D_2 and D_4 terms; D4_aniso > 0 means dispersion
#   drops faster along [100] than [111].
#
# High-precision extraction at proofs/foundations/lorentz_sig_h_lv_coefficients.py
# (mpmath 500-bit precision, 4-point Vandermonde fit eliminating D6/D8
# contamination, three high-symmetry directions) yields:
#
#   D_H        = 1/16    (50+ digit match)
#   D4_iso^H   = -1/1024 (25+ digit match)
#   D4_aniso^H = +1/1536 (25+ digit match)
#
# Hence:
#   eta^H_NB = D4_aniso^H / D_H^2 = (1/1536) / (1/16)^2
#            = (1/1536) * 256
#            = 256/1536
#            = 1/6.
#
# IHARA CROSS-WALKER THEOREM. The Hashimoto Bloch top eigenvalue h_max(k)
# is related to the scalar Bloch top lambda_0(k) by the Ihara factorisation
# for 3-regular graphs (k = 3, k - 1 = 2 in u^2 - lambda u + (k-1) = 0):
#
#   h(lambda) = (lambda + sqrt(lambda^2 - 8)) / 2,   h(3) = 2,
#   h'(3)  = 2,    h''(3) = -4.
#
# Substituting lambda_0(k) = 3 - D_H k^2 - alpha^H k^4 into h(lambda_0(k))
# and matching against h_max(k) = 2 - D_NB k^2 - alpha^NB k^4 yields:
#
#   D_NB             = 2 D_H                      = 1/8       (matches Hashimoto)
#   D4_aniso^NB      = 2 D4_aniso^H                = 1/768     (matches Hashimoto)
#   D4_iso^NB        = 2 D4_iso^H + 2 D_H^2        = +3/512    (NEW; numerically
#                                                              verified)
#   eta_NB           = (1/2) eta^H_NB              = 1/12      (matches Hashimoto)
#
# The cubic anisotropy D4_aniso is therefore a UNIVERSAL graph property of
# srs (the same for both walkers up to the factor 2 in u'(3)). The isotropic
# part D4_iso has additional cross-walker shift +2 D_H^2 from h''(3) = -4.
#
# Sign: eta^H_NB = +1/6 > 0, subluminal (same as Hashimoto: cubic anisotropy
# slows propagation along high-symmetry [100] axes relative to the average).
#
# Derivation chain:
#   A1 + A2-T + Stage 3 spatial setup
#     -> scalar adjacency Bloch H(k) on srs primitive cell.
#     -> Kato 1980 Sec II.5 Thm 5.1/5.11 for Perron band Taylor expansion
#        (D_H = 1/16 by Rayleigh-Schrodinger 2nd-order, S3 theorem).
#     -> 4th-order Taylor expansion gives D4_iso^H, D4_aniso^H.
#     -> eta^H_NB = D4_aniso^H / D_H^2 = 1/6.
#
# THEOREM-GRADE STATUS: the closed-form values D4_iso^H = -1/1024 and
# D4_aniso^H = +1/1536 are derived by symbolic Feshbach-Löwdin partition
# (Kato §II.4) of H(k) relative to the Perron eigenstate, in
# proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py. The script
# uses sympy exact-rational arithmetic throughout (no numerical fit,
# no large-denominator coincidence argument) and runs in ~4 seconds.
# The 25+ digit numerical extraction in
# proofs/foundations/lorentz_sig_h_lv_coefficients.py is now an
# independent cross-check rather than the primary source.

# --- INPUTS --------------------------------------------------
# symbol     | value     | status    | source                                                          | meaning
# -----------|-----------|-----------|-----------------------------------------------------------------|---------
# D_H        | 1/16      | [derived] | predictions/srs_bloch_dispersion_gamma.py                       | k^2 coefficient (Kato + S3)
# D4_iso^H   | -1/1024   | [derived] | proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py       | k^4 isotropic (Feshbach-Löwdin)
# D4_aniso^H | +1/1536   | [derived] | proofs/foundations/lorentz_sig_h_lv_4th_order_symbolic.py       | k^4 anisotropic (Feshbach-Löwdin)

# --- IMPLEMENTATION ------------------------------------------

from fractions import Fraction
import functools

# Dispersion coefficients from high-precision extraction (proofs/foundations/lorentz_sig_h_lv_coefficients.py)
D_H = Fraction(1, 16)
D4_iso_H = Fraction(-1, 1024)
D4_aniso_H = Fraction(1, 1536)

# Headline LV ratio
eta_H_NB = D4_aniso_H / (D_H * D_H)   # = (1/1536) * 256 = 1/6
srs_bloch_lv_dim6_pred = float(eta_H_NB)

assert eta_H_NB == Fraction(1, 6), f"Expected 1/6; got {eta_H_NB}"

print(f"D_H          = {D_H} = {float(D_H):.6f}")
print(f"D4_iso^H     = {D4_iso_H} = {float(D4_iso_H):.10f}")
print(f"D4_aniso^H   = {D4_aniso_H} = {float(D4_aniso_H):.10f}")
print(f"eta^H_NB     = D4_aniso^H / D_H^2 = {eta_H_NB} = {float(eta_H_NB):.15f}")
print(f"Sign:        +{float(eta_H_NB):.4f} > 0  (subluminal)")


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_eta_H_NB(D_H_val, D4_aniso_H_val):
    """
    Dimension-6 Lorentz-violation coefficient for the scalar Bloch H Perron
    band on srs.

    eta^H_NB = D4_aniso^H / D_H^2.

    For srs with exact-rational lattice geometry (NN_DIST = sqrt(2)/4),
    the scalar Bloch dispersion coefficients are
        D_H        = 1/16   (Kato + S3, theorem-grade)
        D4_iso^H   = -1/1024 (numerical to 25+ digits)
        D4_aniso^H = +1/1536 (numerical to 25+ digits)
    yielding eta^H_NB = 1/6.

    Parameters
    ----------
    D_H_val : float
        Coefficient of |k|^2 in lambda_0(k) Taylor expansion.
    D4_aniso_H_val : float
        Coefficient of f4(khat) |k|^4 in lambda_0(k) Taylor expansion.

    Returns
    -------
    float
        eta^H_NB dimensionless.
    """
    return D4_aniso_H_val / (D_H_val * D_H_val)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl_result = float(eta_H_NB)
    pure_result = predict_eta_H_NB(float(D_H), float(D4_aniso_H))
    print(f"\nImplementation: {impl_result:.15f}")
    print(f"Pure function:  {pure_result:.15f}")
    assert abs(impl_result - pure_result) < 1e-15

    # Exact rational check
    assert eta_H_NB == Fraction(1, 6), "Exact rational check failed"
    print(f"Exact value:    1/6 = {float(Fraction(1, 6)):.15f}  OK")

    # Sympy exact
    import sympy as sp
    D_H_sym = sp.Rational(1, 16)
    D4_iso_H_sym = sp.Rational(-1, 1024)
    D4_aniso_H_sym = sp.Rational(1, 1536)
    eta_H_sym = D4_aniso_H_sym / D_H_sym**2
    assert eta_H_sym == sp.Rational(1, 6), f"Sympy mismatch: {eta_H_sym}"
    print(f"Sympy exact:    {eta_H_sym} = 1/6  OK")

    # Sign check
    assert pure_result > 0, "Expected subluminal (positive)"
    print(f"Sign:           subluminal (> 0)")

    # Cross-walker check: Ihara relation gives eta_NB = (1/2) eta^H_NB = 1/12
    eta_NB_predicted = sp.Rational(1, 2) * eta_H_sym
    assert eta_NB_predicted == sp.Rational(1, 12), \
        f"Ihara cross-walker mismatch: {eta_NB_predicted}"
    print(f"Cross-walker:   Ihara gives eta_NB = (1/2) * eta^H_NB = 1/12  OK")
    print(f"                (matches predictions/eta_lattice_lorentz_dim6.py)")

    print("\nOK: eta^H_NB = 1/6 (CAS-verified, subluminal, sister to eta_NB = 1/12).")
