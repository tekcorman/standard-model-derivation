#!/usr/bin/env python3
"""
qtz Bloch-Hashimoto spectrum at Γ — analytic + numerical probe (Phase 0d).

Audit v2 Phase 0d (M6 operator-wave spectrum) check: compute qtz's
Hashimoto operator spectrum at the Γ-point of the trigonal Brillouin zone
without requiring the full bond list, by exploiting only:
  (a) 4-regular: row sums of A_qtz(Γ) = k = 4.
  (b) 3-vertex primitive cell with vertex-transitive C_3 action that
      cycles all three vertices (3-cycle, no fixed vertex).
  (c) Stark-Terras factorization (verified generic in Phase 0b):
      det(uI - B) = (u^2-1)^(|E|-|V|) · Π_{λ ∈ σ(A)} (u^2 - λu + (k-1)).

Compares the resulting (Re(h), Im(h)/|h|²) at qtz Γ against srs's
(Re(h_srs_P), Im(h_srs_P)/|h_srs_P|²) = (√3/2, √5/4) at srs's k_P.

Output: confirms qtz IS Ramanujan at Γ (|h_qtz_Γ|² = k-1 = 3); but
Re(h_qtz_Γ) = -1 is NEGATIVE (vs srs's +√3/2 at k_P_srs), and
Im(h_qtz_Γ)/|h_qtz_Γ|² = √2/3 ≠ √5/4. The sign-flip of Re(h) is the
structural finding: any observable that uses Re(h) as a kinematic
factor (e.g., η_B Sakharov tree amplitude) would flip sign on qtz.

NOTE: This script handles the Γ-point only. qtz's analog of srs's k_P
(the unique selected saddle for η_B-style observables) might be a
different high-symmetry k-point (K, M, A, ...). Full Phase 0d closure
needs all qtz high-symmetry k-points. This Γ probe establishes that
M6 surfaces structurally distinct numerical content on qtz at the
first non-trivial k-point, sufficient to prove M6 is differential
(not generic-without-content); not yet sufficient to gate qtz on η_B
(needs identification of qtz's k_P analog).
"""

import sympy as sp


# ---- Inputs ----
k_qtz = 4  # qtz coordination
n_V_qtz = 3  # vertices per primitive cell
n_E_qtz = n_V_qtz * k_qtz // 2  # = 6 by handshake


# ---- Step 1: A_qtz(Γ) by 3-cycle vertex-transitive symmetry ----
# At Γ, all Bloch phases are 1; A_qtz(Γ) is real symmetric.
# Vertex-transitive C_3 cycling all 3 vertices forces A to be circulant
# with first row (0, c, c) where c counts undirected bonds between any
# two distinct vertices in the primitive cell + lattice-summed off-cell
# bonds (which at Γ all contribute with phase 1).
# Row sum = 2c = 4 (4-regular) → c = 2.
A_qtz_Gamma = sp.Matrix([
    [0, 2, 2],
    [2, 0, 2],
    [2, 2, 0],
])

# Verify: row sums = 4, trace = 0 (no self-loops), Hermitian.
for i in range(3):
    row_sum = sum(A_qtz_Gamma[i, j] for j in range(3))
    assert row_sum == k_qtz, f"Row {i} sum = {row_sum}, expected {k_qtz}"
assert A_qtz_Gamma.trace() == 0, "Trace must be 0 (no self-loops)"
assert A_qtz_Gamma == A_qtz_Gamma.T, "A must be symmetric"


# ---- Step 2: A eigenvalues ----
lam = sp.symbols('lam')
char_poly = sp.factor((lam * sp.eye(3) - A_qtz_Gamma).det())
print(f"A_qtz(Γ) characteristic polynomial: {char_poly}")
# Expect (λ-4)(λ+2)² → eigenvalues {4, -2, -2}
A_eigs = sp.solve(char_poly, lam)
print(f"A_qtz(Γ) eigenvalues: {A_eigs}")
assert sp.simplify(char_poly - (lam - 4) * (lam + 2) ** 2) == 0, \
    f"Unexpected char poly: {char_poly}"


# ---- Step 3: Stark-Terras factorization for B_qtz(Γ) ----
u = sp.symbols('u')
# det(uI - B) = (u² - 1)^(|E|-|V|) · Π_λ (u² - λu + (k-1))
prefactor = (u ** 2 - 1) ** (n_E_qtz - n_V_qtz)
inner = sp.expand(
    (u ** 2 - 4 * u + (k_qtz - 1))
    * (u ** 2 + 2 * u + (k_qtz - 1)) ** 2  # λ = -2 with mult 2
)
det_full = sp.expand(prefactor * inner)
print(f"\ndet(uI - B_qtz(Γ)) = (u²-1)^{n_E_qtz - n_V_qtz} · (u²-4u+3) · (u²+2u+3)²")

# Check: degree should be 2·|E| = 12
deg = sp.degree(det_full, u)
assert deg == 2 * n_E_qtz, f"Expected degree {2 * n_E_qtz}, got {deg}"


# ---- Step 4: Hashimoto eigenvalue extraction ----
# Tree (u²-1)³ → u = ±1 each with mult 3.
# (u²-4u+3) = (u-1)(u-3) → u = 1 (Perron-trivial) and u = 3 (Perron).
# (u²+2u+3)² → u = -1 ± i√2, each with multiplicity 2.

# Solve the inner factor for the complex saddle.
quadratic = u ** 2 + 2 * u + 3
roots = sp.solve(quadratic, u)
print(f"\nInner factor u²+2u+3 roots (the Ramanujan-saturated saddle): {roots}")

# Pick the upper root (positive imaginary part) by convention.
h_qtz_Gamma = next(r for r in roots if sp.im(sp.simplify(r)) > 0)
print(f"h_qtz(Γ) = {h_qtz_Gamma}")

# Multiplicity from the squared inner factor.
mult_h_qtz_Gamma = 2
print(f"Multiplicity of h_qtz(Γ) in B = {mult_h_qtz_Gamma}")


# ---- Step 5: Properties of h_qtz(Γ) ----
re_h = sp.re(sp.simplify(h_qtz_Gamma))
im_h = sp.im(sp.simplify(h_qtz_Gamma))
mod_sq_h = sp.simplify(h_qtz_Gamma * sp.conjugate(h_qtz_Gamma))
mod_h = sp.sqrt(mod_sq_h)
im_over_modsq = sp.simplify(im_h / mod_sq_h)
re_over_mod = sp.simplify(re_h / mod_h)

print(f"\nProperties of h_qtz(Γ):")
print(f"  Re(h_qtz_Γ)               = {re_h}")
print(f"  Im(h_qtz_Γ)               = {im_h}")
print(f"  |h_qtz_Γ|²                = {mod_sq_h}      (Ramanujan bound k-1 = {k_qtz - 1})")
print(f"  Im(h_qtz_Γ)/|h_qtz_Γ|²    = {im_over_modsq} ≈ {float(im_over_modsq):.4f}")
print(f"  sin(arg h_qtz_Γ) = Im(h)/|h| = {sp.simplify(im_h / mod_h)} ≈ {float(im_h / mod_h):.4f}")

# Ramanujan saturation check.
assert mod_sq_h == k_qtz - 1, \
    f"|h_qtz_Γ|² = {mod_sq_h}, expected Ramanujan bound {k_qtz - 1}"


# ---- Step 6: Comparison with srs at k_P ----
re_h_srs = sp.sqrt(3) / 2
im_h_srs = sp.sqrt(5) / 2
mod_sq_h_srs = re_h_srs ** 2 + im_h_srs ** 2  # = 2 = k_srs - 1
im_over_modsq_srs = sp.simplify(im_h_srs / mod_sq_h_srs)

print(f"\nReference: srs at k_P:")
print(f"  Re(h_srs_P)               = {re_h_srs} ≈ {float(re_h_srs):.4f}")
print(f"  Im(h_srs_P)               = {im_h_srs} ≈ {float(im_h_srs):.4f}")
print(f"  |h_srs_P|²                = {mod_sq_h_srs}      (Ramanujan k-1 = 2)")
print(f"  Im(h_srs_P)/|h_srs_P|²    = {im_over_modsq_srs} ≈ {float(im_over_modsq_srs):.4f}")

print(f"\n--- Comparison ---")
print(f"  Re(h_qtz_Γ) - Re(h_srs_P)   = {sp.simplify(re_h - re_h_srs)}")
print(f"    sign: qtz is NEGATIVE, srs is POSITIVE — STRUCTURAL SIGN FLIP")
print(f"  |h|² ratio (qtz/srs)        = {mod_sq_h}/{mod_sq_h_srs} = {sp.Rational(mod_sq_h, mod_sq_h_srs) if mod_sq_h_srs else 'na'}")
print(f"  Im(h)/|h|² ratio (qtz/srs)  = {sp.simplify(im_over_modsq / im_over_modsq_srs)}")


# ---- Step 7: Implications for η_B Sakharov closure ----
# η_B_srs = ε_CP · Re(h_P_srs) · α₁_srs^M_srs = (1/5) · (√3/2) · (2/3)^48
# η_B_qtz_Γ = ε_CP_qtz · Re(h_qtz_Γ) · α₁_qtz^M_qtz
#           = ε_CP_qtz · (-1) · (3/4)^(M_qtz · 4)
#
# The Re(h) sign flip means qtz at Γ predicts NEGATIVE η_B. Observed
# η_B is positive (matter-excess universe). Per the η_B closure, |Re(h)|
# is the kinematic factor with sign determined by the substrate's
# saddle. If qtz's selected k-point analog is Γ, qtz predicts the
# WRONG SIGN of η_B — a hard structural gate (sign-falsification).

eta_B_srs_re_h_factor = re_h_srs            # +√3/2
eta_B_qtz_Gamma_re_h_factor = re_h          # -1

print(f"\nImplication for η_B (Sakharov-Hashimoto-Bass closure):")
print(f"  η_B_srs uses Re(h_P_srs) = +√3/2 (positive matter excess)")
print(f"  η_B_qtz at Γ uses Re(h_qtz_Γ) = -1 (NEGATIVE → antimatter excess)")
print(f"  Sign mismatch with observed positive η_B if qtz_k_P_analog = Γ.")
print(f"  Caveat: full Phase 0d needs to verify qtz's selected k-point.")
print(f"  Phase 0d may need K, M, A, L, H k-point analyses too.")


# ---- Numerical cross-check ----
import numpy as np
A_num = np.array(A_qtz_Gamma.tolist(), dtype=float)
A_eigs_num = np.linalg.eigvals(A_num)
print(f"\nNumerical A_qtz(Γ) eigenvalues: {sorted(A_eigs_num.real, reverse=True)}")
A_eigs_expected = sorted([4.0, -2.0, -2.0], reverse=True)
A_eigs_sorted = sorted(A_eigs_num.real, reverse=True)
for got, expected in zip(A_eigs_sorted, A_eigs_expected):
    assert abs(got - expected) < 1e-12, \
        f"Numerical eigenvalue mismatch: got {got}, expected {expected}"


# ---- Summary ----
print(f"\n{'='*60}")
print(f"PHASE 0d Γ-POINT FINDING")
print(f"{'='*60}")
print(f"qtz IS Ramanujan at Γ (|h|² = k-1 = 3 saturated).")
print(f"Multiplicity 2 — analogous to srs at k_P (mult 2).")
print(f"BUT Re(h_qtz_Γ) = -1 vs Re(h_srs_P) = +√3/2: STRUCTURAL SIGN FLIP.")
print(f"Im(h_qtz_Γ)/|h_qtz_Γ|² = √2/3 ≈ 0.471 vs srs's √5/4 ≈ 0.559.")
print(f"")
print(f"M6 verdict (partial, Γ-point only):")
print(f"  Mechanism is GENERIC (qtz has Ramanujan saddles too).")
print(f"  Numerical content DIFFERS structurally:")
print(f"    - Re(h) sign flip → could falsify η_B sign on qtz at Γ.")
print(f"    - Im(h)/|h|² ratio → 0.471/0.559 = 0.84 differential.")
print(f"  Full Phase 0d needs K, M, A, L, H k-point spectra to identify")
print(f"  qtz's analog of srs's k_P (the framework-selected saddle).")
print(f"")
print(f"OK: qtz Γ-point structural finding established.")
