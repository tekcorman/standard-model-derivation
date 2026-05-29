#!/usr/bin/env python3
"""
Scratch probe: survey dispersions at Gamma, H, P, N of the srs scalar Bloch
adjacency A(k) and identify all Compton-like (quadratic, isolated-from-zero)
modes. Working sheet for the fermion-mass-source attempt.

Results (summary at end of script):
  * Gamma: +3 simple (Perron, gamma=1/16, photon). -1 triplet: degenerate.
    Under C_3 decomposition (trivial + omega + omega-bar), only the
    C_3-trivial mode has a QUADRATIC dispersion (E2 = -pi^2/3 sum dk_a^2).
    The omega + omega-bar modes form a Dirac cone with linear first-order
    shifts +/- 2 pi sqrt(3) t along the body-diagonal.
  * H: -3 simple quadratic (gamma = -1/16, negative = antiphoton/hole).
    +1 triplet: degenerate, no C_3 stabilizer at H, generic splitting.
  * P: +/-sqrt(3) each doubly degenerate, Dirac cones (per prior work).
  * N: 4 simple eigenvalues. +1 and -1 have INDEFINITE (hyperbolic)
    quadratic dispersions (E2 = -2 pi^2 dk1 dk3 etc. with zero diagonal).
    +/-sqrt(5) have complicated anisotropic quadratic dispersions.

Key finding for fermion mass source:
  The Gamma-point -1 triplet decomposes under C_3 as (trivial, omega,
  omega-bar). ONLY the trivial isotypic gives a quadratic (Compton-like)
  dispersion. The omega/omega-bar pair gives a Dirac cone. So three
  generations cannot be read off as three Compton modes at Gamma.
"""

import sympy as sp
import numpy as np
from numpy import linalg as la

# ------------------------------------------------------------------
# Build A(k) symbolically
# ------------------------------------------------------------------

k1, k2, k3 = sp.symbols('k1 k2 k3', real=True)
A_k = sp.zeros(4, 4)


def _add(tgt, src, cell):
    A_k[tgt, src] += sp.exp(sp.I * 2 * sp.pi * (cell[0] * k1 + cell[1] * k2 + cell[2] * k3))


_add(1, 0, (-1, -1, -1)); _add(0, 1, (1, 1, 1))
_add(2, 0, (-1, -1, -1)); _add(0, 2, (1, 1, 1))
_add(3, 0, (-1, -1, -1)); _add(0, 3, (1, 1, 1))
_add(2, 1, (1, 0, 0));    _add(1, 2, (-1, 0, 0))
_add(3, 1, (0, -1, 0));   _add(1, 3, (0, 1, 0))
_add(3, 2, (0, 0, 1));    _add(2, 3, (0, 0, -1))

# Hermiticity sanity
assert sp.simplify(A_k - A_k.H) == sp.zeros(4, 4)


def eval_A(k_red):
    """Return A(k_red) as a sympy matrix in simplified closed form."""
    return sp.simplify(A_k.subs({k1: k_red[0], k2: k_red[1], k3: k_red[2]}))


def A_numeric(k_red):
    A = np.array(A_k.subs({k1: float(k_red[0]), k2: float(k_red[1]), k3: float(k_red[2])})
                 .evalf().tolist(), dtype=complex)
    return (A + A.conj().T) / 2


GAMMA = (sp.Integer(0), sp.Integer(0), sp.Integer(0))
H_pt  = (sp.Rational(-1, 2), sp.Rational(1, 2), sp.Rational(1, 2))
P_pt  = (sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 4))
N_pt  = (sp.Integer(0), sp.Integer(0), sp.Rational(1, 2))

points = {'Gamma': GAMMA, 'H': H_pt, 'P': P_pt, 'N': N_pt}


# ------------------------------------------------------------------
# Step A. Spectrum at each high-symmetry point.
# ------------------------------------------------------------------
print("=" * 70)
print("STEP A: eigenvalues at Gamma, H, P, N")
print("=" * 70)
for label, k0 in points.items():
    A0 = eval_A(k0)
    eigs = A0.eigenvals()
    spec_str = ", ".join([f"{sp.simplify(ev)} x {m}" for ev, m in eigs.items()])
    print(f"  {label:6s}: {{{spec_str}}}")


# ------------------------------------------------------------------
# Step B. For each simple (non-degenerate) eigenvalue, compute the
# second-order Rayleigh-Schrodinger quadratic dispersion coefficient.
# ------------------------------------------------------------------

def rs_nondegenerate_E2(k0, E_target):
    """Return E2(dk) for a simple eigenvalue E_target of A(k0)."""
    A0 = eval_A(k0)
    dk1, dk2, dk3 = sp.symbols('dk1 dk2 dk3', real=True)
    # Perturbation Taylor terms
    H1 = sp.zeros(4, 4)
    for ka, dka in zip((k1, k2, k3), (dk1, dk2, dk3)):
        H1 = H1 + sp.diff(A_k, ka).subs({k1: k0[0], k2: k0[1], k3: k0[2]}) * dka
    H2 = sp.zeros(4, 4)
    for ka, dka in zip((k1, k2, k3), (dk1, dk2, dk3)):
        for kb, dkb in zip((k1, k2, k3), (dk1, dk2, dk3)):
            d2 = sp.diff(A_k, ka, kb).subs({k1: k0[0], k2: k0[1], k3: k0[2]})
            H2 = H2 + sp.Rational(1, 2) * d2 * dka * dkb
    # Build eigenvectors
    eigs = A0.eigenvals()
    v0 = None
    others = []  # list of (En, v_orthonormal)
    for ev, m in eigs.items():
        ns = (A0 - ev * sp.eye(4)).nullspace()
        # Gram-Schmidt
        orth = []
        for v in ns:
            for u in orth:
                v = v - (u.H * v)[0, 0] * u
            nrm = sp.sqrt((v.H * v)[0, 0])
            if sp.simplify(nrm) == 0:
                continue
            orth.append(sp.simplify(v / nrm))
        if sp.simplify(ev - E_target) == 0:
            assert len(orth) == 1
            v0 = orth[0]
        else:
            for v in orth:
                others.append((ev, v))
    assert v0 is not None
    # Second-order E2
    diag = sp.expand((v0.H * H2 * v0)[0, 0])
    cross = sp.S.Zero
    for En, vn in others:
        amp = (vn.H * H1 * v0)[0, 0]
        amp_c = (v0.H * H1 * vn)[0, 0]
        cross = cross + sp.expand(amp * amp_c / (E_target - En))
    return sp.simplify(diag + cross)


print()
print("=" * 70)
print("STEP B: quadratic dispersion E2(dk) for each simple eigenvalue")
print("=" * 70)

# Wrap in a helper that converts dk (reduced) E2 to physical q^2 coefficient
def E2_to_gamma_phys(E2_expr):
    """
    E2(dk) is a quadratic form in (dk1, dk2, dk3). Physical |q|^2
    = (2 pi)^2 * dk^T G dk with G = diag(2) + off-diag(1).
    If E2 = -const * |q|^2, return const; else return None + a direction-
    dependent gamma map.
    """
    dk1, dk2, dk3 = sp.symbols('dk1 dk2 dk3', real=True)
    G = sp.Matrix([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    # |q|^2 / (2 pi)^2:
    q_sq_over_4pi2 = 2 * (dk1**2 + dk2**2 + dk3**2) + 2 * (dk1*dk2 + dk1*dk3 + dk2*dk3)
    # If E2 = -gamma_phys * (2 pi)^2 * (that form), then
    #   gamma_phys = -E2 / ( (2 pi)^2 * q_sq_over_4pi2 )
    # This is direction-invariant only if the ratio is a constant.
    # Test along several directions.
    gammas = {}
    for label, subs_dict in [
        ('along (1,0,0)', {dk1: 1, dk2: 0, dk3: 0}),
        ('along (0,1,0)', {dk1: 0, dk2: 1, dk3: 0}),
        ('along (0,0,1)', {dk1: 0, dk2: 0, dk3: 1}),
        ('along (1,1,0)', {dk1: 1, dk2: 1, dk3: 0}),
        ('along (1,0,1)', {dk1: 1, dk2: 0, dk3: 1}),
        ('along (0,1,1)', {dk1: 0, dk2: 1, dk3: 1}),
        ('along (1,1,1)', {dk1: 1, dk2: 1, dk3: 1}),
        ('along (1,-1,0)', {dk1: 1, dk2: -1, dk3: 0}),
        ('along (1,-2,1)', {dk1: 1, dk2: -2, dk3: 1}),
    ]:
        E2_d = E2_expr.subs(subs_dict)
        qsq_d = q_sq_over_4pi2.subs(subs_dict)
        if qsq_d == 0:
            gammas[label] = 'null direction (|q|=0)'
        else:
            g = sp.simplify(-E2_d / ((2 * sp.pi)**2 * qsq_d))
            gammas[label] = g
    return gammas


for label, k0 in points.items():
    A0 = eval_A(k0)
    eigs = A0.eigenvals()
    for ev, mult in eigs.items():
        ev_s = sp.simplify(ev)
        if mult == 1:
            E2 = rs_nondegenerate_E2(k0, ev_s)
            print(f"\n  {label}, E = {ev_s} (simple):")
            print(f"    E2(dk) = {sp.expand(E2)}")
            # Directional gamma values
            gammas = E2_to_gamma_phys(E2)
            for dir_lbl, g in gammas.items():
                print(f"      gamma {dir_lbl:22s} = {g}")


# ------------------------------------------------------------------
# Step C. Gamma-point -1 triplet: degenerate RS with C_3 decomposition.
# The -1 subspace at Gamma is 3-dim. C_3 acts on v_0 fixed and (v_1, v_2, v_3)
# as a 3-cycle. Decompose -1 subspace under C_3.
# ------------------------------------------------------------------

print()
print("=" * 70)
print("STEP C: Gamma-point -1 triplet, C_3 decomposition")
print("=" * 70)

A_G = eval_A(GAMMA)
# Build P_sigma
P_sigma = sp.Matrix([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0],
])
assert sp.simplify(P_sigma * A_G - A_G * P_sigma) == sp.zeros(4, 4)
assert sp.simplify(P_sigma**3 - sp.eye(4)) == sp.zeros(4, 4)

omega = sp.exp(2 * sp.pi * sp.I / 3)
omega_bar = sp.exp(-2 * sp.pi * sp.I / 3)

# C_3 isotypic eigenvectors of P_sigma:
# On v_0: trivial
# On {v_1, v_2, v_3}: trivial (sum), omega, omega-bar.
u_triv_fixed = sp.Matrix([1, 0, 0, 0])
u_triv_cycle = sp.Matrix([0, 1, 1, 1]) / sp.sqrt(3)
u_omega = sp.Matrix([0, 1, omega, omega**2]) / sp.sqrt(3)
u_omegabar = sp.Matrix([0, 1, omega**2, omega]) / sp.sqrt(3)

# Verify eigenvalues under P_sigma
assert sp.simplify(P_sigma * u_triv_fixed - u_triv_fixed) == sp.zeros(4, 1)
assert sp.simplify(P_sigma * u_triv_cycle - u_triv_cycle) == sp.zeros(4, 1)

def is_zero_vec(v):
    # simplify with exp-rewrite
    out = []
    for i in range(v.shape[0]):
        e = v[i].rewrite(sp.exp)
        e = sp.simplify(sp.expand_complex(e))
        out.append(e)
    return all(sp.simplify(x) == 0 for x in out)

assert is_zero_vec(P_sigma * u_omega - omega * u_omega)
assert is_zero_vec(P_sigma * u_omegabar - omega_bar * u_omegabar)

# -1 eigenspace of A_G = orthogonal complement of Perron v_P = (1,1,1,1)/2.
# Inside this, C_3 isotypic decomp:
# trivial: orthogonal to Perron (sqrt(3)*u_triv_fixed - u_triv_cycle) / 2 (unnormalised)
v_triv_m1 = sp.sqrt(3) * u_triv_fixed - u_triv_cycle
v_triv_m1 = v_triv_m1 / sp.sqrt((v_triv_m1.H * v_triv_m1)[0, 0])
v_triv_m1 = sp.simplify(v_triv_m1)
assert sp.simplify(A_G * v_triv_m1 - (-1) * v_triv_m1) == sp.zeros(4, 1)
assert sp.simplify(P_sigma * v_triv_m1 - v_triv_m1) == sp.zeros(4, 1)

# omega and omega-bar auto in -1 eigenspace (orthogonal to (1,1,1,1))
assert is_zero_vec(A_G * u_omega - (-1) * u_omega)
assert is_zero_vec(A_G * u_omegabar - (-1) * u_omegabar)

print("  -1 eigenspace isotypic basis built:")
print(f"    trivial (v_triv_m1):   sqrt(3)*e_0/2 - (e_1+e_2+e_3)/(2*sqrt(3))")
print(f"    omega (u_omega):       (0, 1, omega, omega^2) / sqrt(3)")
print(f"    omega-bar (u_omegabar): (0, 1, omega^2, omega) / sqrt(3)")


# ------------------------------------------------------------------
# First-order perturbation matrix M1 on the -1 triplet, in isotypic basis
# ------------------------------------------------------------------

dk1_s, dk2_s, dk3_s = sp.symbols('dk1 dk2 dk3', real=True)
H1 = sp.zeros(4, 4)
for ka, dka in zip((k1, k2, k3), (dk1_s, dk2_s, dk3_s)):
    H1 = H1 + sp.diff(A_k, ka).subs({k1: 0, k2: 0, k3: 0}) * dka

basis3 = [v_triv_m1, u_omega, u_omegabar]
M1 = sp.zeros(3, 3)
for i, vi in enumerate(basis3):
    for j, vj in enumerate(basis3):
        val = (vi.H * H1 * vj)[0, 0]
        val = val.rewrite(sp.exp)
        val = sp.simplify(sp.expand_complex(val))
        M1[i, j] = val

print("\n  First-order M1 matrix in isotypic basis (trivial, omega, omega-bar):")
sp.pprint(M1)

# Check block structure: trivial sector must be scalar, omega <-> omega
# sector can have off-diagonal mixing along C_3-breaking directions.
print(f"\n  M1[0,0] (trivial self-shift) = {sp.simplify(M1[0, 0])}")
print(f"  M1[1,1] (omega self-shift)    = {sp.simplify(M1[1, 1])}")
print(f"  M1[2,2] (omega-bar self-shift)= {sp.simplify(M1[2, 2])}")
print(f"  M1[1,2] (omega <-> omega-bar) = {sp.simplify(M1[1, 2])}")
print(f"  M1[0,1] (trivial <-> omega)   = {sp.simplify(M1[0, 1])}")


# ------------------------------------------------------------------
# Along C_3-preserving direction dk = t*(1,1,1), M1 should be block-
# diagonal in the C_3 isotypic basis (no mixing). Compute eigenvalues.
# ------------------------------------------------------------------

print("\n  Along body-diagonal direction dk = t*(1,1,1):")
t = sp.symbols('t', real=True, positive=True)
M1_bd = M1.subs({dk1_s: t, dk2_s: t, dk3_s: t})
M1_bd_simp = sp.Matrix(3, 3, lambda i, j: sp.simplify(M1_bd[i, j]))
print(f"    M1[0,0] = {M1_bd_simp[0, 0]}")
print(f"    M1[1,1] = {M1_bd_simp[1, 1]}")
print(f"    M1[2,2] = {M1_bd_simp[2, 2]}")
print(f"    M1[1,2] = {M1_bd_simp[1, 2]}   (off-diag omega<->omega-bar)")
print(f"    M1[2,1] = {M1_bd_simp[2, 1]}")

# The omega and omega-bar diagonal entries might be complex (first-order
# shift = diagonal of M1 in eigenbasis of H1).
# Is M1_bd Hermitian? Check.
M1_bd_H = M1_bd_simp.H
diff = sp.Matrix(3, 3, lambda i, j: sp.simplify(M1_bd_simp[i, j] - M1_bd_H[i, j]))
print(f"\n    M1_bd - M1_bd^dag = {diff}")

# Compute eigenvalues of M1_bd to get the actual first-order shifts
M1_bd_eigs = M1_bd_simp.eigenvals()
print(f"\n    First-order shifts (eigenvalues of M1_bd) along (1,1,1):")
for ev, m in M1_bd_eigs.items():
    print(f"      {sp.simplify(ev)}  (mult {m})")


# ------------------------------------------------------------------
# Second-order RS for the -1 triplet.
# Because M1 is non-zero at first order, the proper degenerate RS
# treatment is: (i) diagonalise M1 to find the leading-order
# eigenvectors that split the degeneracy, (ii) within each split
# subspace (now non-degenerate at first order), compute second-order
# correction using eigenstates of the external subspace (here just
# the Perron +3 eigenvector).
#
# For the C_3-preserving direction, M1_bd is already block-diagonal
# in the isotypic basis, so the trivial / omega / omega-bar basis
# vectors ARE the leading-order eigenvectors. Their eigenvalues of
# M1_bd are the first-order shifts.
# ------------------------------------------------------------------

print()
print("  Second-order RS per isotypic component at Gamma (-1 triplet),")
print("  using the C_3-preserving isotypic basis (valid in C_3-preserving")
print("  directions; not the leading-eigenvector basis in generic directions):")

H2 = sp.zeros(4, 4)
for ka, dka in zip((k1, k2, k3), (dk1_s, dk2_s, dk3_s)):
    for kb, dkb in zip((k1, k2, k3), (dk1_s, dk2_s, dk3_s)):
        d2 = sp.diff(A_k, ka, kb).subs({k1: 0, k2: 0, k3: 0})
        H2 = H2 + sp.Rational(1, 2) * d2 * dka * dkb

v_perron = sp.Matrix([1, 1, 1, 1]) / 2
assert sp.simplify(A_G * v_perron - 3 * v_perron) == sp.zeros(4, 1)

def simplify_scalar(x):
    e = x.rewrite(sp.exp)
    e = sp.simplify(sp.expand_complex(e))
    return e

for label, v_alpha in zip(['trivial', 'omega', 'omega-bar'], basis3):
    diag = simplify_scalar((v_alpha.H * H2 * v_alpha)[0, 0])
    amp = simplify_scalar((v_perron.H * H1 * v_alpha)[0, 0])
    amp_c = simplify_scalar((v_alpha.H * H1 * v_perron)[0, 0])
    cross = simplify_scalar(amp * amp_c / (-1 - 3))
    E2_alpha = simplify_scalar(diag + cross)
    print(f"    {label:10s} E2(dk) = {sp.expand(E2_alpha)}")

# ------------------------------------------------------------------
# Step D. Compute the full 3-eigenvalue branch at Gamma (-1) along
# body-diagonal direction numerically, to confirm which branches are
# quadratic vs linear.
# ------------------------------------------------------------------

print()
print("=" * 70)
print("STEP D: numerical probe of Gamma (-1 triplet) along (1,1,1) direction")
print("=" * 70)
print("  t         | eigenvalues (sorted) | (ev - (-1))/t       | (ev - (-1))/t^2")
for t_val in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
    M = A_numeric((t_val, t_val, t_val))
    eigs = np.sort(la.eigvalsh(M))
    # Isolate the three -1 branches and the +3 branch
    m1_branches = eigs[0:3]
    shifts = m1_branches + 1
    print(f"  {t_val:8.1e} | {m1_branches} | {shifts/t_val} | {shifts/t_val**2}")


# ------------------------------------------------------------------
# Step E. Same probe along C_3-breaking direction (1,0,0).
# ------------------------------------------------------------------

print()
print("=" * 70)
print("STEP E: numerical probe of Gamma (-1 triplet) along (1,0,0) direction")
print("=" * 70)
print("  t         | eigenvalues (sorted) | shifts/t           | shifts/t^2")
for t_val in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
    M = A_numeric((t_val, 0.0, 0.0))
    eigs = np.sort(la.eigvalsh(M))
    m1_branches = eigs[0:3]
    shifts = m1_branches + 1
    print(f"  {t_val:8.1e} | {m1_branches} | {shifts/t_val} | {shifts/t_val**2}")


# ------------------------------------------------------------------
# Step E2. Full M1 eigenvalue analysis along (1,0,0)
# ------------------------------------------------------------------
print()
print("=" * 70)
print("STEP E2: full M1 eigenvalues along (1,0,0) direction")
print("=" * 70)
M1_100 = M1.subs({dk1_s: t, dk2_s: 0, dk3_s: 0})
M1_100 = sp.Matrix(3, 3, lambda i, j: simplify_scalar(M1_100[i, j]))
print(f"  M1 along (1,0,0):")
sp.pprint(M1_100)
print(f"\n  Eigenvalues of M1 along (1,0,0):")
eigs_M1_100 = M1_100.eigenvals()
for ev, m in eigs_M1_100.items():
    print(f"    {sp.simplify(ev)}")
# Numerically verify
M1_100_num = np.array(M1_100.subs({t: 1.0}).evalf().tolist(), dtype=complex)
numerical_eigs = np.sort(np.real(la.eigvals(M1_100_num)))
print(f"\n  Numerical eigenvalues (at t=1): {numerical_eigs}")
print(f"  Predicted 2*2*pi*sqrt(?): depends on structure along (1,0,0).")
print(f"  From numerical data in Step E: shifts/t = +/- 4.44, i.e. first-order")
print(f"  splits linearly at slope 4.44. Note 4*sqrt(3)*pi/(2 sqrt(3)) = ... ")
# Check: +-2 sqrt(3) pi / sqrt(3) = +-2 pi (if the body-diag slope is
# 2 sqrt(3) pi and the (1,0,0) slope scales as 1/sqrt(3))
# Actually predicted along (1,0,0): M1 eigenvalues at t=1:
# Expected slope: likely 2 sqrt(3) pi / sqrt(3) = 2 pi ~ 6.28, but data
# shows 4.44 ~ sqrt(2) * pi ? No, 4.44 / pi = 1.414 = sqrt(2).
# So slope = sqrt(2) pi. Interesting - not a simple form.


# ------------------------------------------------------------------
# Step F. H-point +1 triplet: is C_3 a stabilizer? No (H has
# 4-fold orbit). But can we still decompose under some subgroup?
# Check the true little group of H.
# ------------------------------------------------------------------

print()
print("=" * 70)
print("STEP F: H-point +1 triplet behaviour")
print("=" * 70)
print("  H = (-1/2, 1/2, 1/2) in primitive reduced coords.")
print("  C_3 (k1,k2,k3) -> (k3,k1,k2) sends H -> (1/2, -1/2, 1/2).")
print("  These are congruent modulo Z^3? (1/2) - (-1/2) = 1, integer; yes.")
# Actually check if (1/2, -1/2, 1/2) == H + G for some reciprocal lattice vec
# But in reduced coords the BZ is at [-1/2, 1/2]^3; H and its image differ
# by (1, -1, 0), which is a reciprocal lattice vector.
# So actually C_3 MAY stabilize H modulo reciprocal lattice.
# Test: is A(H) == A(C3(H))?

A_H = eval_A(H_pt)
H_rotated = (sp.Rational(1, 2), sp.Rational(-1, 2), sp.Rational(1, 2))
A_H_rotated = eval_A(H_rotated)
diff = sp.simplify(A_H - A_H_rotated)
print(f"  A(H) - A(C3 H) = zero? {diff == sp.zeros(4, 4)}")

# If A is the same (modulo basis), then there is an extended symmetry.
# Let's test eigenvalues directly
print(f"  A(H) eigenvals = {A_H.eigenvals()}")
print(f"  A(C3*H) eigenvals = {A_H_rotated.eigenvals()}")

# Numerical probe
print("  Numerical probe along (1,0,0) at H:")
print("  t         | eigenvalues (sorted)")
for t_val in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
    M = A_numeric((-0.5 + t_val, 0.5, 0.5))
    eigs = np.sort(la.eigvalsh(M))
    print(f"  {t_val:8.1e} | {eigs}")


# ------------------------------------------------------------------
# Step G. N-point: 4 simple eigenvalues. Each has its own quadratic
# dispersion. Are they Compton-like?
# ------------------------------------------------------------------

print()
print("=" * 70)
print("STEP G: N-point simple eigenvalues, full quadratic dispersions")
print("=" * 70)
for ev in [sp.Integer(1), sp.Integer(-1), sp.sqrt(5), -sp.sqrt(5)]:
    E2 = rs_nondegenerate_E2(N_pt, ev)
    print(f"\n  N, E = {ev}:")
    print(f"    E2 = {sp.expand(E2)}")
    # Signature of the quadratic form
    dk = sp.symbols('dk1 dk2 dk3', real=True)
    Q = sp.zeros(3, 3)
    E2_exp = sp.expand(E2)
    for i, a in enumerate(dk):
        for j, b in enumerate(dk):
            if i == j:
                Q[i, j] = E2_exp.coeff(a, 2)
            else:
                c = E2_exp.coeff(a).coeff(b)
                Q[i, j] = c / 2
    # Eigenvalues of Q reveal the signature
    eigs_Q = Q.eigenvals()
    print(f"    Q tensor eigenvalues: {[sp.simplify(e) for e in eigs_Q]}")
    # For definite negative (Compton-massive) signature, need all eigs < 0.


# ------------------------------------------------------------------
# Final summary
# ------------------------------------------------------------------

print()
print("=" * 70)
print("SUMMARY OF COMPTON-LIKE (QUADRATIC) MODES ACROSS BZ")
print("=" * 70)
print("""
Gamma point (C_3 stabilizer):
  +3 simple (Perron): QUADRATIC. Isotropic in physical q.
     E2 = -|q|^2 / 16. Photon-like (gamma_phys = 1/16).
  -1 triplet:
    * C_3-trivial isotypic: QUADRATIC. E2 = -(pi^2/3)(dk1^2+dk2^2+dk3^2).
      Anisotropic in physical q.
    * C_3-omega / omega-bar pair: LINEAR (Dirac cone) along C_3 directions.
      First-order shift +- 2 pi sqrt(3) t along (1,1,1).
  Only ONE Compton mode in the -1 triplet (the trivial), not three.

H point (C_3 stabilizes H mod reciprocal lattice):
  -3 simple: QUADRATIC. E2 = +(pi^2/2)(dk1^2+...+dk2 dk3). Negative effective
    mass (gamma negative). Antiphoton-like.
  +1 triplet: degenerate, behavior mirrors Gamma (-1) triplet by particle-hole
    analogy. Expect one quadratic (trivial) + Dirac cone (omega/omega-bar).

P point (C_3 stabilizer):
  +/- sqrt(3) each doubly degenerate. Dirac cones.
  (per an internal working note)

N point (smaller little group; no C_3):
  1 simple: E2 = -2 pi^2 (dk1 dk3 + dk2 dk3 + dk3^2). Indefinite Q.
  -1 simple: E2 = -2 pi^2 dk1 dk2. HYPERBOLIC (saddle).
  +sqrt(5), -sqrt(5): complicated anisotropic E2, may be indefinite.
  No clean 3-fold grouping available.

Total Compton-massive (gamma DEFINITE) modes identified:
  Gamma +3 (Perron): photon, gamma=1/16 isotropic.
  Gamma -1 trivial isotypic: gamma_tensor = pi^2/3 * I, anisotropic (since
    |q|^2 is not proportional to sum dk_a^2).
  H -3 simple: gamma = pi^2/2 diag + pi^2/4 off, photon mirror.
  (H +1 trivial presumably also a candidate by symmetry.)

Total linear/Dirac modes: Gamma -1 omega/omegabar, P +/-sqrt(3), H +1 omega/omegabar.

Total indefinite/saddle modes: N +1, -1, +/-sqrt(5).

BEST CANDIDATE for fermion mass source: NONE produces three distinct
Compton masses. The Gamma -1 triplet produces ONE Compton mode (trivial
isotypic) and TWO Dirac cone modes (omega/omega-bar), not three masses.
""")
