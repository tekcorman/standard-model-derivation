#!/usr/bin/env python3
"""
Refine: numerically determine the true second-order E2 coefficient for the
Compton branch of Gamma (-1 triplet) along various directions, to check
whether E2 really does vanish along (1,0,0) as STEP E of the main probe
suggested (truly flat to machine precision).
"""

import sympy as sp
import numpy as np
from numpy import linalg as la

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


def A_numeric(k_red):
    A = np.array(A_k.subs({k1: float(k_red[0]), k2: float(k_red[1]), k3: float(k_red[2])})
                 .evalf().tolist(), dtype=complex)
    return (A + A.conj().T) / 2


print("Refined probe: -1 triplet at Gamma in several directions, small t.")
print("Compton branch = middle eigenvalue after sorting.")
print()
print("Direction | t        | Compton E2/t^2 (after subtracting -1)")
print("-" * 70)

directions = [
    ('(1,0,0)',       np.array([1.0, 0.0, 0.0])),
    ('(0,1,0)',       np.array([0.0, 1.0, 0.0])),
    ('(0,0,1)',       np.array([0.0, 0.0, 1.0])),
    ('(1,1,0)',       np.array([1.0, 1.0, 0.0])),
    ('(1,1,1)',       np.array([1.0, 1.0, 1.0])),
    ('(1,-1,0)',      np.array([1.0, -1.0, 0.0])),
    ('(2,1,0)',       np.array([2.0, 1.0, 0.0])),
    ('(1,1,-1)',      np.array([1.0, 1.0, -1.0])),
]

for label, d in directions:
    # Normalize to unit reduced length, but keep direction structure
    for t in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        dk = t * d
        M = A_numeric(dk)
        eigs = np.sort(la.eigvalsh(M))
        m1_branches = eigs[0:3]
        shifts = m1_branches + 1
        # middle is the near-zero one
        idx = np.argmin(np.abs(shifts))
        compton_shift = shifts[idx]
        coef = compton_shift / (t**2)
        print(f"{label:10s} | {t:8.1e} | coef = {coef:+.6e}")
    print()

# Focus on (1,0,0): is it truly zero, or is there a very small coefficient
# that only shows up at smaller t where other branches dominate?
print()
print("(1,0,0) direction VERY small t (double precision limits):")
for t in [1e-5, 1e-6, 1e-7, 1e-8]:
    M = A_numeric((t, 0, 0))
    eigs = np.sort(la.eigvalsh(M))
    m1_branches = eigs[0:3]
    shifts = m1_branches + 1
    idx = np.argmin(np.abs(shifts))
    compton_shift = shifts[idx]
    coef = compton_shift / (t**2)
    print(f"t = {t:.0e}: Compton shift = {compton_shift:+.3e}, coef = {coef:+.3e}")


# The 3x3 M_1 matrix along (1,0,0): eigenvalues are (0, +sqrt(2) pi, -sqrt(2) pi).
# The zero-eigenvalue eigenvector v_0 mixes trivial/omega/omega-bar.
# Second-order RS on this eigenvector involves (a) <v_0|H_2|v_0> and (b)
# sum over the OTHER -1 eigenvectors (the +/- sqrt(2) pi eigenvectors) is
# suppressed because those are IN the degenerate subspace... actually in
# degenerate RS, after first-order split the non-zero-first-order eigen-
# values are at finite separation at order O(t), so their contribution to
# E_0^(2) is O(t^2 / t) = O(t), giving an effective O(t^3) term in
# lambda_0. At O(t^2) only the <v_0|H_2|v_0> + Perron contribution remains.

# Compute <v_0|H_2|v_0> + |<v_Perron|H_1|v_0>|^2 / (-4) along (1,0,0)
# numerically by building v_0 from M_1's kernel.

print()
print("Symbolic check of the E2 along (1,0,0) on the proper Compton branch:")
from sympy import Matrix, sqrt, I as SpI, pi, symbols, simplify, zeros, eye

# Build M_1 at (1,0,0), find zero eigenvector, compute E2 directly.
# M_1 from fermion_mass_probe.py:
t_s = symbols('t', real=True, positive=True)
M1_10 = Matrix([
    [0, pi*t_s*(-sqrt(3) - 3*SpI)/6, pi*t_s*(sqrt(3) - 3*SpI)/6],
    [pi*t_s*(-sqrt(3) + 3*SpI)/6, 2*sqrt(3)*pi*t_s/3, 0],
    [pi*t_s*(sqrt(3) + 3*SpI)/6, 0, -2*sqrt(3)*pi*t_s/3],
])
print("M_1 along (1,0,0):")
sp.pprint(M1_10)

eigs = M1_10.eigenvects()
print()
print("Eigenvectors of M_1:")
for ev, m, vs in eigs:
    print(f"  eigenvalue: {simplify(ev)}")
    for v in vs:
        vs_simp = Matrix([simplify(v[i, 0]) for i in range(3)])
        print(f"    vec: {vs_simp.T}")
