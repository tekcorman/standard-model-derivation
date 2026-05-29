#!/usr/bin/env python3
"""
G_sub session 5: finite-difference probe of GS energy under finite strain.

Settles the static-elastic sign-convention question of
an internal working note by computing
∂²E_GS/∂u^{ab}∂u^{cd}|_{u=0} directly via finite-difference of the
GS energy at finite small strain — bypassing perturbation theory entirely.

Method
------
For deformation gradient E^{ac} (3×3 real matrix), the deformed bond
displacement is

  r_b(E)^μ = r_b^μ + E^{μc} r_b^c    (sum over c)

and the deformed Bloch Hamiltonian is

  H[E](k)_{βα} = Σ_bonds exp(i k · r_b(E))

(per the convention of `lorentz_sig_strain_perturbation.py`, where
δH/δE^{ac} = i k_a r_b^c × phase).

The matter ground-state energy at half-filling (μ = 0) is

  E_GS(E; k) = Σ_{bands n with eigval(H[E](k))_n < 0} eigval(H[E](k))_n

BZ-averaged: ⟨E_GS⟩_BZ(E). For small finite u, this is a smooth function;
compute its second derivative numerically by finite-differencing.

Compare to the perturbative computations of
`lorentz_sig_g_sub_elastic_moduli.py`:

  Hypothesis (subtractive, metallic f-sum cancellation):
    ∂²⟨E_GS⟩_BZ / ∂u² = K_dia_script - K_para_script  ≈ +0.19

  Alternative (additive):
    ∂²⟨E_GS⟩_BZ / ∂u² = K_dia_script + K_para_script  ≈ +35

The finite-difference computation is the GROUND TRUTH (no sign-convention
ambiguity).

Status
------
Numerical settlement of the elastic-modulus sign convention. Theorem-grade
once the result lands cleanly on one combination; closes step 1 of the
session-5 entry-point closure plan.
"""
from __future__ import annotations

import numpy as np

# srs primitive cell (matching lorentz_sig_g_sub_elastic_moduli.py)
ATOMS = np.array([
    [1/8, 1/8, 1/8],
    [3/8, 7/8, 5/8],
    [7/8, 5/8, 3/8],
    [5/8, 3/8, 7/8],
])

A_PRIM = np.array([
    [-1/2,  1/2,  1/2],
    [ 1/2, -1/2,  1/2],
    [ 1/2,  1/2, -1/2],
])

CELL_EDGES = [
    (0, 1, (1, 1, 1)),
    (0, 2, (1, 1, 1)),
    (0, 3, (1, 1, 1)),
    (1, 2, (-1, 0, 0)),
    (1, 3, (0, 1, 0)),
    (2, 3, (0, 0, -1)),
]

DIRECTED_BONDS = []
for s, t, c in CELL_EDGES:
    DIRECTED_BONDS.append((s, t, np.array(c)))
    DIRECTED_BONDS.append((t, s, -np.array(c)))


def bond_displacement(src, tgt, cell):
    return ATOMS[tgt] - ATOMS[src] + cell @ A_PRIM


BOND_DISPLACEMENTS = [
    (s, t, bond_displacement(s, t, c)) for s, t, c in DIRECTED_BONDS
]


def H_bloch_strained(k_cart, E):
    """4×4 Bloch H at Cartesian k under deformation gradient E (3×3 real).

    Convention: r_b → r_b + E·r_b (so script's E^{ac} matches strain-
    perturbation's E^{ac} = ∂_c u_a, with deformed r_b^a = r_b^a + E^{ac} r_b^c).
    """
    H = np.zeros((4, 4), dtype=complex)
    for s, t, rb in BOND_DISPLACEMENTS:
        rb_def = rb + E @ rb
        phase = np.exp(1j * np.dot(k_cart, rb_def))
        H[t, s] += phase
    return (H + H.conj().T) / 2


def gs_energy_at_k(k_cart, E, mu=0.0):
    """Sum of eigenvalues below mu at given strain E, given k."""
    H = H_bloch_strained(k_cart, E)
    eigvals = np.linalg.eigvalsh(H).real
    return float(np.sum(eigvals[eigvals < mu]))


def bz_average_gs_energy(E, N_grid=12, half_extent=2*np.pi, mu=0.0):
    """BZ-average of GS energy over [-half_extent, half_extent]³ grid."""
    ks = np.linspace(-half_extent, half_extent, N_grid, endpoint=False)
    total = 0.0
    n = 0
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                total += gs_energy_at_k(np.array([k1, k2, k3]), E, mu)
                n += 1
    return total / n


def finite_diff_2nd_deriv(coord_pair, N_grid=12, h=1e-3, half_extent=2*np.pi):
    """Symmetric central difference for ∂²⟨E_GS⟩/∂E^{ab} ∂E^{cd}.

    coord_pair = ((a, b), (c, d)). For diagonal (ab == cd), uses 1D 2nd
    derivative; for off-diagonal, mixed partial via 4-point formula.
    """
    (a, b), (c, d) = coord_pair

    def E_with(*entries):
        E = np.zeros((3, 3))
        for (i, j, val) in entries:
            E[i, j] = val
        return E

    if (a, b) == (c, d):
        # ∂²f/∂x² ≈ [f(+h) - 2 f(0) + f(-h)] / h²
        f_plus = bz_average_gs_energy(E_with((a, b, +h)), N_grid, half_extent)
        f_zero = bz_average_gs_energy(np.zeros((3, 3)), N_grid, half_extent)
        f_minus = bz_average_gs_energy(E_with((a, b, -h)), N_grid, half_extent)
        return (f_plus - 2 * f_zero + f_minus) / h**2

    # mixed partial: ∂²f/∂x∂y ≈ [f(+h,+h) - f(+h,-h) - f(-h,+h) + f(-h,-h)] / (4h²)
    f_pp = bz_average_gs_energy(E_with((a, b, +h), (c, d, +h)), N_grid, half_extent)
    f_pm = bz_average_gs_energy(E_with((a, b, +h), (c, d, -h)), N_grid, half_extent)
    f_mp = bz_average_gs_energy(E_with((a, b, -h), (c, d, +h)), N_grid, half_extent)
    f_mm = bz_average_gs_energy(E_with((a, b, -h), (c, d, -h)), N_grid, half_extent)
    return (f_pp - f_pm - f_mp + f_mm) / (4 * h**2)


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def voigt_iso_mu(C_11, C_12, C_44):
    """Voigt isotropic shear modulus from cubic (C_11, C_12, C_44)."""
    return (C_11 - C_12 + 3 * C_44) / 5


def main():
    header("G_sub session 5: finite-difference probe of ∂²E_GS/∂u²")
    print()
    print("  Computes ∂²⟨E_GS⟩_BZ/∂E^{ab}∂E^{cd}|_{E=0} by direct finite-")
    print("  difference of the BZ-averaged ground-state energy under deformation")
    print("  gradient E. No perturbation-theory sign convention enters.")
    print()

    N = 12
    half_extent = 2 * np.pi
    h = 1e-3

    print(f"  Grid: {N}³ k-points over [-2π, 2π]³  (proper BCC fundamental domain)")
    print(f"  Strain step: h = {h}")

    # Compute the symmetric-strain elastic constants C_11, C_12, C_44.
    # C_11 = ∂²E/∂E^{xx}∂E^{xx}
    # C_12 = ∂²E/∂E^{xx}∂E^{yy}
    # For shear: the symmetric strain u_{xy} = (E^{xy} + E^{yx})/2 enters; the
    # antisymmetric ω_{xy} = (E^{xy} - E^{yx})/2 enters separately as rotation.
    # To extract C_44 (symmetric-shear modulus), use a SYMMETRIC strain:
    #   E^{xy} = E^{yx} = u → ε^{xy} = u (Voigt 2ε^{xy}=2u; convention-dep)
    # Compute C_44 directly via mixed second derivative ∂²E/∂E^{xy}∂E^{xy}
    # at antisymmetric strain (rotation), which should give zero, vs symmetric.

    header("Step 1: pure-stretch second derivatives")
    print()
    print("  Compute ∂²⟨E_GS⟩/∂E^{ab}² for various (a, b).")
    print()
    h_stretch = 1e-3
    diagonal = {}
    for (a, b), label in [((0,0), 'xx'), ((1,1), 'yy'), ((2,2), 'zz'),
                          ((0,1), 'xy'), ((1,0), 'yx'),
                          ((0,2), 'xz'), ((2,0), 'zx'),
                          ((1,2), 'yz'), ((2,1), 'zy')]:
        d2 = finite_diff_2nd_deriv(((a, b), (a, b)), N_grid=N, h=h_stretch,
                                     half_extent=half_extent)
        diagonal[(a, b)] = d2
        print(f"    ∂²E_GS/(∂E^{{{label}}})² = {d2:+.6f}")

    header("Step 2: symmetric vs antisymmetric shear")
    print()
    print("  ε^{xy} = (E^{xy}+E^{yx})/2 (symmetric strain)")
    print("  ω^{xy} = (E^{xy}-E^{yx})/2 (rotation)")
    print()
    print("  Apply E^{xy} = E^{yx} = u (symmetric pure shear):")
    print("    ∂²E_GS/∂u²|_{sym shear} should give 4 × (C^{xyxy}_{(sym)})")
    print()

    # Symmetric pure shear: E^{xy} = E^{yx} = u
    f_zero = bz_average_gs_energy(np.zeros((3,3)), N, half_extent)
    h_shear = 1e-3
    E_plus = np.zeros((3, 3)); E_plus[0,1] = E_plus[1,0] = +h_shear
    E_minus = np.zeros((3, 3)); E_minus[0,1] = E_minus[1,0] = -h_shear
    f_plus_sym = bz_average_gs_energy(E_plus, N, half_extent)
    f_minus_sym = bz_average_gs_energy(E_minus, N, half_extent)
    sym_2nd = (f_plus_sym - 2 * f_zero + f_minus_sym) / h_shear**2
    print(f"    Symmetric shear E^{{xy}}=E^{{yx}}=u: ∂²E_GS/∂u² = {sym_2nd:+.6f}")
    print(f"    Decomposition: 2(∂²E/∂E^{{xy}}²) + 2(∂²E/∂E^{{xy}}∂E^{{yx}})")
    print(f"                 = 2 × {diagonal[(0,1)]:.4f} + 2 × (mixed)")

    # mixed E^{xy}, E^{yx}
    mixed_xy_yx = finite_diff_2nd_deriv(((0,1), (1,0)), N_grid=N, h=h_shear,
                                          half_extent=half_extent)
    print(f"    ∂²E_GS/∂E^{{xy}}∂E^{{yx}} = {mixed_xy_yx:+.6f}")
    print(f"    Predicted symmetric 2nd: {2*diagonal[(0,1)] + 2*mixed_xy_yx:+.6f}")

    # Antisymmetric pure shear (rotation): E^{xy} = -E^{yx} = u
    E_plus = np.zeros((3, 3)); E_plus[0,1] = +h_shear; E_plus[1,0] = -h_shear
    E_minus = np.zeros((3, 3)); E_minus[0,1] = -h_shear; E_minus[1,0] = +h_shear
    f_plus_anti = bz_average_gs_energy(E_plus, N, half_extent)
    f_minus_anti = bz_average_gs_energy(E_minus, N, half_extent)
    anti_2nd = (f_plus_anti - 2 * f_zero + f_minus_anti) / h_shear**2
    print(f"    Antisymmetric shear E^{{xy}}=-E^{{yx}}=u: ∂²E_GS/∂u² = {anti_2nd:+.6f}")
    print(f"    (rotation should give 0 if rotational invariance holds — sanity)")

    header("Step 3: cubic Voigt elastic constants from finite-diff (symmetrized)")
    print()
    # C_11 = ∂²E/∂(ε^xx)² where ε^xx = E^xx (symmetric strain has only diagonal entries)
    # C_12 = ∂²E/∂ε^xx ∂ε^yy
    # C_44 = (1/4) × ∂²E/∂(ε^xy)² where ε^xy = (E^xy + E^yx)/2
    #         applying symmetric strain E^xy = E^yx = u → ε^xy = u → ∂²E/∂u² = 4 × C^{xyxy}
    #         but Voigt: C_44 = C^{xyxy}
    # Following standard cubic elasticity convention: U = (1/2) C^{abcd} ε_{ab} ε_{cd}
    # for symmetric ε. C_44 in Voigt = C^{xyxy}.

    # Average over 3 cubic axes for stretch:
    C_11_fd = (diagonal[(0,0)] + diagonal[(1,1)] + diagonal[(2,2)]) / 3
    print(f"    C_11 = ⟨∂²E/∂(E^aa)²⟩ avg over a∈{{x,y,z}}  = {C_11_fd:+.6f}")

    # C_12 = ∂²E/∂E^xx ∂E^yy (and cyclic):
    C_xx_yy = finite_diff_2nd_deriv(((0,0), (1,1)), N_grid=N, h=h_stretch,
                                      half_extent=half_extent)
    C_yy_zz = finite_diff_2nd_deriv(((1,1), (2,2)), N_grid=N, h=h_stretch,
                                      half_extent=half_extent)
    C_xx_zz = finite_diff_2nd_deriv(((0,0), (2,2)), N_grid=N, h=h_stretch,
                                      half_extent=half_extent)
    C_12_fd = (C_xx_yy + C_yy_zz + C_xx_zz) / 3
    print(f"    C_12 = ⟨∂²E/∂E^aa ∂E^bb⟩ a≠b           = {C_12_fd:+.6f}")

    # C_44 from symmetric shear: ∂²E/∂u²|_{ε^xy=u} = 2 × 2 × C^{xyxy}
    # because U = (1/2) C^{abcd} ε_{ab} ε_{cd} and symmetric shear ε^xy = ε^yx = u
    # gives U = (1/2) × 2 × (C^{xyxy} u² + C^{xyyx} u²) × 2 (count both pairs)
    # ... let me work it out: U = (1/2) Σ C^{abcd} ε_{ab} ε_{cd}.
    # With ε^{xy} = ε^{yx} = u, only 4 nonzero entries of ε; U = (1/2)×4×C^{xyxy}×u²
    #                                            = 2 C^{xyxy} u²
    # So ∂²E/∂u²|_{sym shear} = 4 C^{xyxy} = 4 C_44.
    sym_xy_2nd = sym_2nd
    sym_xz_E_plus = np.zeros((3,3)); sym_xz_E_plus[0,2]=sym_xz_E_plus[2,0]=+h_shear
    sym_xz_E_minus= np.zeros((3,3)); sym_xz_E_minus[0,2]=sym_xz_E_minus[2,0]=-h_shear
    sym_xz_2nd = (bz_average_gs_energy(sym_xz_E_plus, N, half_extent)
                  - 2*f_zero
                  + bz_average_gs_energy(sym_xz_E_minus, N, half_extent)) / h_shear**2
    sym_yz_E_plus = np.zeros((3,3)); sym_yz_E_plus[1,2]=sym_yz_E_plus[2,1]=+h_shear
    sym_yz_E_minus= np.zeros((3,3)); sym_yz_E_minus[1,2]=sym_yz_E_minus[2,1]=-h_shear
    sym_yz_2nd = (bz_average_gs_energy(sym_yz_E_plus, N, half_extent)
                  - 2*f_zero
                  + bz_average_gs_energy(sym_yz_E_minus, N, half_extent)) / h_shear**2
    sym_avg = (sym_xy_2nd + sym_xz_2nd + sym_yz_2nd) / 3
    C_44_fd = sym_avg / 4
    print(f"    sym shear ∂²E/∂u²: xy={sym_xy_2nd:+.6f}, xz={sym_xz_2nd:+.6f}, yz={sym_yz_2nd:+.6f}")
    print(f"    avg = {sym_avg:+.6f}, C_44 = avg/4 = {C_44_fd:+.6f}")

    mu_iso_fd = voigt_iso_mu(C_11_fd, C_12_fd, C_44_fd)
    print()
    print(f"    Voigt isotropic μ: μ_iso = (C_11 - C_12 + 3 C_44)/5 = {mu_iso_fd:+.6f}")

    header("Step 4: comparison with K_dia + K_para and K_dia - K_para")
    print()

    # Import and compare
    from lorentz_sig_g_sub_elastic_moduli import bz_average_full, voigt_components

    K_para, K_dia, K_full_add = bz_average_full(N_grid=N, mu=0.0, half_extent=half_extent)
    K_full_sub = K_dia - K_para

    v_para = voigt_components(K_para)
    v_dia = voigt_components(K_dia)
    v_add = voigt_components(K_full_add)
    v_sub = voigt_components(K_full_sub)

    mu_para = voigt_iso_mu(v_para['C_11'], v_para['C_12'], v_para['C_44'])
    mu_dia = voigt_iso_mu(v_dia['C_11'], v_dia['C_12'], v_dia['C_44'])
    mu_add = voigt_iso_mu(v_add['C_11'], v_add['C_12'], v_add['C_44'])
    mu_sub = voigt_iso_mu(v_sub['C_11'], v_sub['C_12'], v_sub['C_44'])

    print(f"  Finite-difference μ_iso (truth):                     {mu_iso_fd:+.6f}")
    print()
    print(f"  K_para script-convention μ_iso:                       {mu_para:+.6f}")
    print(f"  K_dia  script-convention μ_iso:                       {mu_dia:+.6f}")
    print(f"  K_dia + K_para  (additive: 'gapped Iorio')  μ_iso:    {mu_add:+.6f}")
    print(f"  K_dia - K_para  (subtractive: 'metallic')   μ_iso:    {mu_sub:+.6f}")
    print()
    diff_add = abs(mu_iso_fd - mu_add)
    diff_sub = abs(mu_iso_fd - mu_sub)
    print(f"  |fd - additive|     = {diff_add:.6f}")
    print(f"  |fd - subtractive|  = {diff_sub:.6f}")
    print()

    if diff_sub < diff_add:
        print(f"  → SUBTRACTIVE wins. K_dia - K_para is the correct combination.")
        print(f"    Physical interpretation: metallic f-sum-rule cancellation;")
        print(f"    para and dia individually large, residual ~0.19 is physics.")
        candidate = mu_iso_fd
        print()
        print(f"  Closure candidate: μ_iso = {candidate:+.6f}")
        print(f"  Compare 15/(8π²) = {15/(8*np.pi**2):.6f}  "
              f"  (ratio fd/(15/8π²) = {candidate / (15/(8*np.pi**2)):.6f})")
        # Try the entry-doc's relation 1/(16π G) = μ_iso
        if candidate > 0:
            G_candidate = 1.0 / (16 * np.pi * candidate)
            print()
            print(f"  Under 1/(16π G_sub) = μ_iso:")
            print(f"    G_sub = 1/(16π × {candidate:.4f}) = {G_candidate:+.6f}")
            print(f"    Compare π/30      = {np.pi/30:.6f}")
            print(f"    Compare 4(√3-1)/27 = {4*(np.sqrt(3)-1)/27:.6f}")
    else:
        print(f"  → ADDITIVE wins. K_dia + K_para is the correct combination.")
        print(f"    Physical interpretation: stable Iorio-elastic solid;")
        print(f"    paramagnetic + diamagnetic both contribute positive ∂²E.")

    header("CONCLUSION")
    print()
    print(f"  Finite-difference μ_iso = {mu_iso_fd:.6f} on {N}³ grid.")
    print(f"  This is the GROUND TRUTH for the substrate's static elastic shear modulus.")
    print(f"  Settles the K_dia ± K_para sign question of g_sub_session5_entry_point.md.")


if __name__ == "__main__":
    main()
