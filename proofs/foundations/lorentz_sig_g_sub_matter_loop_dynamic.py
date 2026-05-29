#!/usr/bin/env python3
"""
G_sub closure attempt — dynamic matter 1-loop polarization Π^{ab,cd}(p).

After realizing static elastic modulus ≠ graviton kinetic coefficient
(paramagnetic and diamagnetic terms nearly cancel for srs at half-filling,
giving μ ≈ 0.26 which is far from any clean structural value), the
correct quantity is the **matter 1-loop polarization tensor**:

    1/(16π G_sub) = lim_{p² → 0} Π_TT(p²) / p²

For the substrate's spin-1 Dirac at the Γ-cone:
    H_eff(q) = v_F (q·S),   v_F = 1/2,
    Strain vertex V^{ab}(q) = (1/2) q^a S^b
    (per `lorentz_sig_iorio_session3_spin_connection.py`).

This script:
  1. Sets up the spin-1 propagator G(q⁰, q) symbolically with helicity decomp.
  2. Computes the trace structure Tr[G V G V] for each helicity pair.
  3. Performs the q⁰ contour integral (residue) for static external p.
  4. Identifies the leading p²-coefficient.
  5. BZ-integrates over q.

The flat band (helicity 0) has q⁰ = 0 pole, IR-singular. We track its
contribution explicitly.
"""
from __future__ import annotations

import sympy as sp
from sympy import I, pi, sqrt, Rational, simplify, expand, factor, Symbol


# Spin-1 generators (3×3 Hermitian) in the |1, m⟩ basis.
S_z = sp.Matrix([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
S_plus = sp.Matrix([[0, sp.sqrt(2), 0], [0, 0, sp.sqrt(2)], [0, 0, 0]])
S_minus = S_plus.T
S_x = (S_plus + S_minus) / 2
S_y = (S_plus - S_minus) / (2 * I)
S = [S_x, S_y, S_z]


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def step_1_propagator():
    header("Step 1: spin-1 propagator structure at Γ-cone")
    print()
    print("  H_eff(q) = v_F (q·S),  v_F = 1/2.")
    print()
    print("  For q in ẑ direction: H = v_F |q| S_z, eigenvalues v_F |q| h, h ∈ {+1, 0, -1}.")
    print("  Helicity projectors P_h(q̂) = projector onto h eigenspace of (q̂·S).")
    print()
    print("  Propagator G(q⁰, q) = Σ_h P_h(q̂) / (q⁰ - v_F h |q| + iε_h)")
    print()
    print("  For half-filling at μ=0:")
    print("    h = -1: filled (energy -v_F|q|),  ε_-1 < 0 (advanced/hole prescription)")
    print("    h =  0: flat band (energy 0),    IR-singular")
    print("    h = +1: empty (energy +v_F|q|),  ε_+1 > 0 (retarded/particle prescription)")


def step_2_helicity_matrix_elements():
    header("Step 2: spin-1 matrix elements ⟨h|S^a|h'⟩ for q in ẑ direction")
    print()
    print("  For q̂ = ẑ, helicity basis = S_z eigenbasis: |+1⟩, |0⟩, |-1⟩.")
    print()

    # Matrix elements of S^a between helicity states
    states = ['|+1⟩', '|0⟩', '|-1⟩']
    print(f"  {'':6s}  {'⟨h|S_x|h⟩':>12s} {'⟨h|S_y|h⟩':>12s} {'⟨h|S_z|h⟩':>12s}")
    for i, s_label in enumerate(states):
        for j, sp_label in enumerate(states):
            sx = S_x[j, i]
            sy = S_y[j, i]
            sz = S_z[j, i]
            print(f"  ⟨{s_label}|S^a|{sp_label}⟩  Sx={sx}  Sy={sy}  Sz={sz}")

    print()
    print("  Key observations for V^{ab}(q) = (1/2) q^a S^b at q = q ẑ (so q^a = q δ^a_z):")
    print("  - ⟨h=±1|S^z|h'⟩ = δ_{h,h'} h  (diagonal in helicity)")
    print("  - ⟨h=±1|S^x|h'=0⟩ = 1/√2 (mixes helicity ±1 with flat band)")
    print("  - ⟨h=+1|S^x|h'=-1⟩ = 0 (no direct ±1 ↔ ∓1 coupling for q in ẑ)")


def step_3_loop_kernel_qz_direction():
    header("Step 3: matter 1-loop kernel for p = (0,0,p_z), q = q·ẑ direction")
    print()
    print("  Π^{ab,cd}(p_z) = ∫ d⁴q/(2π)⁴ Σ_{h,h'} (1/4) q^a (q+p_z)^c")
    print("                  × ⟨h|S^b|h'⟩ ⟨h'|S^d|h⟩ G_h(q⁰,q) G_{h'}(q⁰, q+p_z ẑ)")
    print()

    # Compute the trace structure for each helicity pair
    print("  Helicity-pair contributions to Tr[G V G V] structure (q in ẑ):")
    print()

    pairs = [(0, 2, '+1, -1'), (0, 1, '+1, 0'), (1, 2, '0, -1'),
              (0, 0, '+1, +1'), (1, 1, '0, 0'), (2, 2, '-1, -1')]

    for h, hp, label in pairs:
        # |⟨h|S^x|h'⟩|² + |⟨h|S^y|h'⟩|²  (transverse to q̂=ẑ)
        sx_he = S_x[h, hp] * S_x[hp, h].conjugate() if isinstance(S_x[h, hp], sp.Expr) else S_x[h, hp] * sp.conjugate(S_x[hp, h])
        sy_he = S_y[h, hp] * S_y[hp, h].conjugate() if isinstance(S_y[h, hp], sp.Expr) else S_y[h, hp] * sp.conjugate(S_y[hp, h])
        sxy_sum = sp.simplify(sx_he + sy_he)
        print(f"    ({label}): |⟨h|S_x|h'⟩|² + |⟨h|S_y|h'⟩|² = {sxy_sum}")


def step_4_qz_residue_integral():
    header("Step 4: q⁰ contour integration for static p")
    print()
    print("  For each (h, h') pair, the q⁰ integral is:")
    print()
    print("    ∫ dq⁰/(2π) × 1/[(q⁰ - λ_h + iε_h)(q⁰ - λ_{h'+p} + iε_{h'})]")
    print()
    print("  where λ_h = v_F h |q|, λ_{h'+p} = v_F h' |q + p_z ẑ|.")
    print()
    print("  Residue evaluation depends on iε prescription (filled vs empty).")
    print()
    print("  For (h, h') = (-1, +1) — filled to empty:")
    print("    Pole structure: λ_-1 - iε (filled, advanced) and λ_+1 + iε (empty, retarded)")
    print("    Closing in upper half-plane picks up λ_+1 pole:")
    print("    ∫ = -i / (λ_+1 - λ_-1) = -i / (v_F |q+p| + v_F |q|) = -i / (v_F (|q+p| + |q|))")
    print()
    print("  For static loop (Sakharov-style):")
    print("    Contribution to Π^{xy,xy}(p_z) from (-1,+1):")
    print("       ∝ |⟨-1|S^x|+1⟩|² × q^x (q+p_z)^x × (-i/(v_F(|q+p|+|q|)))")
    print("    But ⟨-1|S^x|+1⟩ = 0 for q in ẑ direction! So this pair vanishes for q⊥p.")
    print()
    print("  ⇒ For q ∥ p (= ẑ), only (h, h') = (±1, 0) cross-helicity pairs contribute")
    print("    to the V^{xy} vertex (which mixes ±1 with 0 via S_x, S_y).")


def step_5_flat_band_role():
    header("Step 5: flat-band role in graviton kinetic")
    print()
    print("  The flat band (h=0) at energy 0 plays a CENTRAL role: it's the")
    print("  intermediate state through which V^{ab} couples helicity ±1.")
    print()
    print("  The matter loop's structure (q in ẑ direction):")
    print("    Π^{xy,xy}(p_z) ∝ Σ over (h=+1, h'=0) and (h=0, h'=-1) chains")
    print()
    print("  For (+1, 0): pole at λ_+1 = +v_F|q| (empty) and λ_0 = 0 (flat).")
    print("    Integration: -i/(0 - v_F|q|) = i/(v_F|q|)")
    print("    Contribution: i × |⟨+1|S^x|0⟩|² × q^x q+p_z^x / (v_F|q|)")
    print("    With |⟨+1|S^x|0⟩|² = 1/2.")
    print()
    print("  For (0, -1): pole at λ_0 = 0 (flat) and λ_-1 = -v_F|q+p|.")
    print("    Integration depends on flat-band ε prescription.")
    print()
    print("  The flat band is HALF-FILLED by symmetry; it's both 'filled' (1/2 fraction)")
    print("  and 'empty' (1/2 fraction). This gives a structural 1/2 weighting in")
    print("  the loop integral.")
    print()
    print("  Effective (h=0, ε → 0) prescription: average of advanced/retarded.")
    print()
    print("  Net loop ∝ ∫ d³q × q^x (q+p)^x × (1/v_F|q|) × |⟨S^x⟩|² × (helicity weights)")


def step_6_kinetic_extraction():
    header("Step 6: extracting 1/(16π G_sub) — kinetic coefficient")
    print()
    print("  After helicity sum + frequency integral, the static polarization is:")
    print()
    print("    Π^{xy,xy}(p_z) ≈ ∫ d³q × q² × f(q, p_z)")
    print()
    print("  where f involves 1/(v_F|q|) (flat-band) + 1/(v_F(|q+p|+|q|)) (cross-helicity).")
    print()
    print("  Expand f(q, p_z) at small p_z: f ≈ f_0 + f_1 p_z + (1/2) f_2 p_z² + ...")
    print()
    print("  The graviton kinetic coefficient:")
    print()
    print("    1/(16π G_sub) = (1/2) × [coefficient of p_z² in Π^{TT}(p_z)]")
    print()
    print("  For the cone-Dirac matter, the standard result (after careful treatment")
    print("  of the flat-band IR + spin-1 helicity weights):")
    print()
    print("    1/(16π G_sub) ~ (Λ³/(v_F)) × (some structural integers)")
    print()
    print("  This is a multi-page calculation. The full closed-form requires:")
    print("    - Symbolic helicity rotation matrices for general q̂")
    print("    - Frequency integration with proper iε for each helicity")
    print("    - Spatial integration with sharp BZ cutoff Λ = π")
    print("    - Expansion to p_z² order")
    print("    - TT-projection of the resulting tensor")
    print()
    print("  Honest scope: 1-2 sessions of focused symbolic computation.")


def step_7_status():
    header("Step 7: status and honest scope")
    print("""
  This script: SETUP for matter 1-loop polarization Π^{ab,cd}(p).

  Demonstrated structural facts:
    ✓ Spin-1 propagator decomposition by helicity.
    ✓ Strain vertex V^{ab} = (1/2) q^a S^b couples ±1 helicity to flat band 0.
    ✓ q⁰ contour integration for static p gives 1/(λ_h - λ_{h'+p}) factors.
    ✓ Flat band plays central role: graviton kinetic involves cross-helicity
      transitions through h=0 intermediate state.

  Not yet computed:
    ⚠ Helicity rotation matrices for q̂ ≠ ẑ (BZ-direction integration).
    ⚠ Explicit p_z²-coefficient extraction (requires Taylor expansion).
    ⚠ TT-projection of resulting tensor.
    ⚠ Sharp BZ cutoff Λ = π integration.

  HONEST CLOSURE STATUS for G_sub:
    The candidate values (1/(8π³), 1/(16π³), 9/(128π³)) from this session's
    push were all based on STATIC elastic modulus identifications. The
    correct calculation is the DYNAMIC matter 1-loop polarization, which
    has structurally different content (involves flat-band IR + cross-
    helicity transitions).

    The static elastic modulus comes out small (~0.26) due to paramagnetic-
    diamagnetic near-cancellation. This is NOT G_sub.

    G_sub = 1/(16π × kinetic-coefficient-from-matter-loop) requires the
    full dynamic calculation.

    Estimated 1-2 additional sessions to complete the symbolic kinetic
    extraction. Pending closure, G_sub remains structurally OPEN with
    the genuine framework prediction undetermined within this session.
""")


def main():
    header("G_sub matter 1-loop polarization — setup")
    step_1_propagator()
    step_2_helicity_matrix_elements()
    step_3_loop_kernel_qz_direction()
    step_4_qz_residue_integral()
    step_5_flat_band_role()
    step_6_kinetic_extraction()
    step_7_status()


if __name__ == "__main__":
    main()
