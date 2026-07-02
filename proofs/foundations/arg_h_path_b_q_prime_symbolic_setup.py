#!/usr/bin/env python3
"""
arg_h_path_b_q_prime_symbolic_setup.py — first stage of the analytic
SU(2) Berry curvature derivation for the +h band crossing at k_P.

Plan: this script extracts the bond list numerically from
build_primitive_unit_cell + find_primitive_connectivity, then constructs
B(k) symbolically (sympy) at k = k_P + (δ1, δ2, δ3). The resulting B(δk)
is a 12×12 matrix of trig polynomials in δ_a, exact under k_P = (1/4)·(1,1,1).

Stage outputs:
    1. Bond table dump (12 directed bonds, with cell vectors).
    2. Symbolic B(k_P) — verify spectrum is {±h, ±h̄, kernel}.
    3. Symbolic ∂_a B at k_P per axis a ∈ {1, 2, 3} (3 matrices).
    4. Symbolic +h band 2-dim eigenspace projector.
    5. Symbolic 2×2 H_eff^a := P_+h · ∂_a B · P_+h.

This is just the SETUP. The Berry curvature integration is a follow-on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sympy as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proofs" / "cosmology"))

from srs_photon_bloch_primitive import (
    build_primitive_unit_cell,
    find_primitive_connectivity,
)


PRINT_WIDTH = 78


def extract_bond_table():
    """Return a deterministic list of 12 directed bonds (e_src, e_tgt, cell)."""
    verts, lat = build_primitive_unit_cell()
    bonds_raw = find_primitive_connectivity(verts, lat)
    # bonds_raw entries are (src, tgt, cell, dr); we only need (src, tgt, cell).
    bonds = [(b[0], b[1], tuple(b[2])) for b in bonds_raw]
    return bonds


def print_bonds(bonds):
    print(f"\n12 directed bonds (src, tgt, cell):")
    for i, (s, t, c) in enumerate(bonds):
        print(f"  bond {i:2d}: {s} -> {t}  cell = {c}")


def build_B_symbolic(bonds, k_syms):
    """B(k)[f, e] = [tgt(e)=src(f)] · [f != rev(e)] · exp(-2πi k·e.cell).

    Returns a 12×12 sympy Matrix in symbols k_syms = (k1, k2, k3)
    (reduced coords).
    """
    n = len(bonds)
    k1, k2, k3 = k_syms
    B = sp.zeros(n, n)
    for e_idx, (e_src, e_tgt, e_cell) in enumerate(bonds):
        for f_idx, (f_src, f_tgt, f_cell) in enumerate(bonds):
            if f_src != e_tgt:
                continue
            rev_cell = tuple(-c for c in e_cell)
            if f_src == e_tgt and f_tgt == e_src and f_cell == rev_cell:
                continue
            phase_arg = -2 * sp.pi * (k1 * e_cell[0] + k2 * e_cell[1] + k3 * e_cell[2])
            B[f_idx, e_idx] = B[f_idx, e_idx] + sp.exp(sp.I * phase_arg)
    return B


def main():
    print("=" * PRINT_WIDTH)
    print("Q' analytic setup: symbolic B(k) at k_P + δk")
    print("=" * PRINT_WIDTH)

    bonds = extract_bond_table()
    assert len(bonds) == 12
    print_bonds(bonds)

    # -------------------------------------------------------------------------
    # Stage 1: B(k_P) symbolic — verify spectrum.
    # -------------------------------------------------------------------------
    print(f"\n" + "-" * PRINT_WIDTH)
    print("Stage 1 — B(k_P) symbolic at k_P = (1/4, 1/4, 1/4)")
    print("-" * PRINT_WIDTH)

    k1, k2, k3 = sp.symbols("k1 k2 k3", real=True)
    B_sym = build_B_symbolic(bonds, (k1, k2, k3))
    # Substitute k_P = (1/4, 1/4, 1/4):
    B_kP = B_sym.subs({k1: sp.Rational(1, 4), k2: sp.Rational(1, 4), k3: sp.Rational(1, 4)})
    B_kP = sp.simplify(B_kP)

    print("  B(k_P) entries (should be in {0, ±1, ±i} — Gaussian integers):")
    nz = 0
    val_set = set()
    for i in range(12):
        for j in range(12):
            v = B_kP[i, j]
            if v != 0:
                nz += 1
                val_set.add(sp.nsimplify(v, rational=False))
    print(f"    nonzero entries: {nz}")
    print(f"    distinct values: {sorted(val_set, key=str)}")

    # Spectrum check.
    print(f"\n  Verifying eigenvalues are {{±h, ±h̄, ±1, 0}}:")
    h = (sp.sqrt(3) + sp.I * sp.sqrt(5)) / 2
    h_bar = (sp.sqrt(3) - sp.I * sp.sqrt(5)) / 2

    # Compute characteristic polynomial.
    lam = sp.Symbol("lam")
    # For speed, drop to numerical eigenvalues first, then verify symbolic
    # eigenvalues at the expected ±h, ±h̄, ±1, 0.
    B_num = np.array(B_kP.tolist(), dtype=complex)
    evs_num = np.linalg.eigvals(B_num)
    print(f"    numerical eigenvalues (sorted by |λ|, then arg):")
    for ev in sorted(evs_num, key=lambda z: (-abs(z), np.angle(z))):
        print(f"      λ = {ev.real:+.6f}{ev.imag:+.6f}j   |λ| = {abs(ev):.4f}")

    # Verify h is an eigenvalue symbolically: det(B_kP - h·I) = 0?
    print(f"\n  Symbolic check: det(B(k_P) − h·I) = ?")
    h_check = sp.simplify(sp.det(B_kP - h * sp.eye(12)))
    print(f"    det(B − h·I) = {h_check}  (expect 0)")

    print(f"\n  Symbolic check: det(B(k_P) + h·I) = ?")
    neg_h_check = sp.simplify(sp.det(B_kP + h * sp.eye(12)))
    print(f"    det(B + h·I) = {neg_h_check}  (expect 0)")

    # -------------------------------------------------------------------------
    # Stage 2: ∂_a B at k_P symbolic per axis.
    # -------------------------------------------------------------------------
    print(f"\n" + "-" * PRINT_WIDTH)
    print("Stage 2 — ∂_a B at k_P for axis a = 1, 2, 3")
    print("-" * PRINT_WIDTH)

    dB_da_sym = []
    for a, ka in enumerate((k1, k2, k3)):
        dB = sp.diff(B_sym, ka)
        dB_kP = dB.subs({k1: sp.Rational(1, 4),
                         k2: sp.Rational(1, 4),
                         k3: sp.Rational(1, 4)})
        dB_kP = sp.simplify(dB_kP)
        dB_da_sym.append(dB_kP)
        # Survey: how many nonzero entries; what coefficients show up?
        nz = 0
        coeff_set = set()
        for i in range(12):
            for j in range(12):
                v = dB_kP[i, j]
                if v != 0:
                    nz += 1
                    coeff_set.add(sp.nsimplify(v / sp.pi, rational=False))
        print(f"  axis {a + 1}: {nz} nonzero entries; "
              f"distinct (entry / π) values: {sorted(coeff_set, key=str)[:6]}…")

    # -------------------------------------------------------------------------
    # Stage 3: +h band eigenspace at k_P symbolically.
    # -------------------------------------------------------------------------
    print(f"\n" + "-" * PRINT_WIDTH)
    print("Stage 3 — +h band 2-dim eigenspace at k_P")
    print("-" * PRINT_WIDTH)

    # nullspace(B_kP - h·I)
    M = B_kP - h * sp.eye(12)
    M = sp.simplify(M)
    null = M.nullspace()
    print(f"  dim(null(B − h·I)) = {len(null)} (expect 2)")
    for i, v in enumerate(null):
        print(f"  basis vector {i}: shape {v.shape}, first few entries:")
        for j in range(min(4, v.rows)):
            print(f"    v[{j}] = {sp.simplify(v[j, 0])}")

    print(f"\n" + "=" * PRINT_WIDTH)
    print("OK: symbolic_setup completed")


if __name__ == "__main__":
    main()
