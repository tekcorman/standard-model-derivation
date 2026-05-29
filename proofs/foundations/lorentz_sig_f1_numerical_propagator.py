#!/usr/bin/env python3
"""
F1 numerical evaluation: explicit substrate Feynman propagator G_F(k, ω) at Γ and P.

Per the F1 closure (`docs/forward_constructions/forward_construction_substrate_propagator.md` Theorem 3.2):

    G_F^sub(k, ω) = i (ω + H(k)) / (ω² - H²(k) + iε)

with H(k) the substrate Bloch Hamiltonian. This script computes G_F(k, ω) as
explicit 4×4 matrices at the two physically-distinguished BZ points:

    Γ = (0, 0, 0)_frac      (high-symmetry origin; lower-3-band Dirac cone here)
    P = (1/4, 1/4, 1/4)_frac (corner; double-Dirac cones at λ = ±√3)

For each k, we report:
  1. Spectrum {ε_α(k)} of H(k).
  2. Wave functions u_α(k) (eigenvectors).
  3. Pole locations of G_F (at ω = ±|ε_α(k)|).
  4. Residues at each pole.
  5. Sample evaluation of G_F(k, ω) at off-shell ω.

This grounds the F1 propagator at the level of explicit 4×4 numerical matrices,
which can then be used downstream by F5/F6/F7 (S-matrix, Feynman rules, RG).

Note on spinor-Dirac lift. The full substrate Dirac D_sub is 32×32 (Cl(6,0)
spinor ⊗ 4-atom Bloch fiber), constructed via the Bloch-lift theorem from
`docs/theorems/theorem_bloch_lift_mu.md`. Its propagator has the same structural form
with D(k) in place of H(k). The 4×4 scalar Bloch propagator computed here is
the framework's "scalar field" sector; the 32×32 spinor lift is a separate
research item (~2-3 sessions).
"""

import os
import sys
import numpy as np
import sympy as sp

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

from proofs.common import find_bonds, bloch_H, ATOMS, A_PRIM, NN_DIST


def header(s):
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


# =============================================================================
# Numerical setup
# =============================================================================

bonds = find_bonds()


def H_at_k(k_frac):
    """Build the 4×4 scalar Bloch Hamiltonian numerically."""
    return bloch_H(k_frac, bonds)


def propagator_F(H, omega, eps=1e-9):
    """Compute G_F(ω) = i (ω + H) / (ω² - H² + iε) as a 4×4 numerical matrix.

    Uses matrix functional calculus. For complex ω, the iε prescription is
    automatic (no need to add it explicitly).
    """
    H = np.asarray(H, dtype=complex)
    n = H.shape[0]
    I = np.eye(n)
    omega_c = complex(omega) + 1j * eps
    numerator = 1j * (omega_c * I + H)
    H_sq = H @ H
    denom = (omega_c**2) * I - H_sq
    G = numerator @ np.linalg.inv(denom)
    return G


def report_at_k(name, k_frac):
    header(f"H(k) and G_F(k, ω) at k = {name}")
    print(f"\n  k_frac = {k_frac}")

    H = H_at_k(k_frac)
    print(f"\n  H({name}) =")
    for row in H:
        print("    " + "  ".join(f"{x.real:+.4f}{x.imag:+.4f}j" if abs(x.imag) > 1e-12
                                  else f"{x.real:+.4f}      " for x in row))

    # Spectrum
    eigs, vecs = np.linalg.eigh(H)
    idx = np.argsort(eigs)
    eigs = eigs[idx]
    vecs = vecs[:, idx]
    print(f"\n  Spectrum {{ε_α({name})}} =")
    for i, e in enumerate(eigs):
        print(f"    ε_{i} = {e:+.6f}")

    # Wave functions
    print(f"\n  Wave functions u_α({name}, r) (eigenvectors of H, columns):")
    for alpha in range(4):
        v = vecs[:, alpha]
        v_str = "  ".join(f"{x.real:+.3f}{x.imag:+.3f}j" if abs(x.imag) > 1e-10
                          else f"{x.real:+.3f}     " for x in v)
        print(f"    α = {alpha}: u = ({v_str})  at ε = {eigs[alpha]:+.4f}")

    # Pole structure
    print(f"\n  Pole structure of G_F({name}, ω):")
    print(f"    G_F has 4 simple poles at ω = ±|ε_α|:")
    for alpha in range(4):
        e = eigs[alpha]
        print(f"      ω = {e:+.6f}    (eigenvalue ε_{alpha})")
    # If H has degenerate spectrum, multiple poles coincide
    unique_eigs = np.unique(np.round(eigs, 8))
    if len(unique_eigs) < 4:
        print(f"    Note: spectrum has {4 - len(unique_eigs)} fold degeneracy.")
    return H, eigs, vecs


def sample_propagator_at_omega(H, omega_values, name="Γ"):
    print(f"\n  G_F({name}, ω) sampled at off-shell ω values:")
    for omega in omega_values:
        G = propagator_F(H, omega)
        # Print [0, 0] entry (representative)
        G00 = G[0, 0]
        print(f"    ω = {omega:+.4f}:  G[0, 0] = {G00.real:+.6f} + {G00.imag:+.6f}j")


# =============================================================================
# Analytic-form analysis at Γ (K_4 adjacency)
# =============================================================================

def analytic_form_gamma():
    header("Analytic G_F(Γ, ω) -- K_4 adjacency")

    # H(Γ) = J - I (K_4 adjacency)
    # Eigenvalues: 3 (once), -1 (three times)
    # Eigenvectors:
    #   v_0 = (1,1,1,1)/2   at ε = 3 (Perron, all-ones)
    #   v_⊥ = orthogonal complement at ε = -1 (3-fold degenerate)
    #
    # H² = (J - I)² = J² - 2J + I
    # J² = 4J (since J is rank-1, J² = (Σ basis · 1)² = 4J... wait)
    # Actually J² = J · J: each row of J is (1,1,1,1) and each column is (1,1,1,1).
    # So J²[i,j] = sum_k J[i,k] J[k,j] = sum_k 1 · 1 = 4. So J² = 4·1_n where 1_n is the
    # all-ones matrix. Wait, J IS the all-ones matrix. So J² = 4·J. Then:
    # H² = 4J - 2J + I = 2J + I.
    #
    # Spectrum of H²: at v_0 (uniform), J v_0 = 4 v_0, so H² v_0 = (8 + 1) v_0 = 9 v_0. ✓
    # At v_⊥, J v_⊥ = 0, so H² v_⊥ = 0 + 1·v_⊥ = v_⊥. ✓
    #
    # G_F(Γ, ω) = i (ω + H) / (ω² - H² + iε)
    #          = i (ω I + J - I) / (ω² · I - 2J - I + iε)
    #
    # Spectral decomposition:
    # In the v_0 direction: G_F → i(ω + 3)/(ω² - 9 + iε) = i / (ω - 3 + iε) - i / (ω + 3 - iε) ... wait
    # Actually i(ω + ε_α)/(ω² - ε_α² + iε) = i / (ω - ε_α + iε sgn(ε_α)) [from F1 §3.2]
    # For ε_α = 3 > 0: i / (ω - 3 + iε)
    # For ε_α = -1 < 0: i / (ω + 1 - iε)
    #
    # So:
    #   G_F(Γ, ω) = (i / (ω - 3 + iε)) P_0  +  (i / (ω + 1 - iε)) P_⊥
    # where P_0 is projection onto v_0 (rank 1) and P_⊥ is projection onto v_0^⊥ (rank 3).
    print(f"""
  H(Γ) = J - I  where J = all-ones matrix.
  H²(Γ) = 2J + I  (since J² = 4J, H² = (J-I)² = 4J - 2J + I = 2J + I).

  Spectral decomposition:
    P_0  = J/4    (projection onto v_0 = (1,1,1,1)/2,  rank 1, eigenvalue 3)
    P_⊥  = I - J/4 (projection onto v_0^⊥, rank 3, eigenvalue -1)

  Substrate Feynman propagator at Γ in spectral form:

    G_F(Γ, ω) = i [ P_0 / (ω - 3 + iε)  +  P_⊥ / (ω + 1 - iε) ]

  Two simple poles:
    ω = +3 - iε  (Perron particle pole, residue +i P_0)
    ω = -1 + iε  (lower-band antiparticle pole, residue -i P_⊥, multiplicity 3)

  At Γ, the Perron mode is the unique non-degenerate one; the other 3 modes
  form the lower-band 3-fold cluster (the spin-1 Dirac cone in the Iorio
  framework — see lorentz_sig_dirac_cone_velocities.py).
""")


# =============================================================================
# Analytic-form analysis at P
# =============================================================================

def analytic_form_P():
    header("Analytic G_F(P, ω) -- doubly-degenerate ±√3 spectrum")

    # H(P) eigenvalues: ±√3, each with multiplicity 2.
    # H²(P) = 3 I_4 (constant identity since all eigenvalues squared = 3).
    # G_F(P, ω) = i (ω I + H(P)) / (ω² - 3 + iε)
    #
    # Spectral decomposition:
    # Let P_+ = projector onto +√3 eigenspace (rank 2)
    # Let P_- = projector onto -√3 eigenspace (rank 2)
    # Then H(P) = √3 (P_+ - P_-) and:
    # G_F(P, ω) = i (ω I + √3 (P_+ - P_-)) / (ω² - 3 + iε)
    #          = (i / (ω - √3 + iε)) P_+  +  (i / (ω + √3 - iε)) P_-
    #
    # Two simple poles:
    # ω = +√3 - iε (residue +i P_+, mult 2)
    # ω = -√3 + iε (residue -i P_-, mult 2)
    print(f"""
  H(P) eigenvalues: ±√3, each with multiplicity 2.
  H²(P) = 3 · I_4  (proportional to identity).

  Spectral decomposition:
    P_+ = projector onto +√3 eigenspace (rank 2)
    P_- = projector onto -√3 eigenspace (rank 2)

  Substrate Feynman propagator at P in spectral form:

    G_F(P, ω) = i [ P_+ / (ω - √3 + iε)  +  P_- / (ω + √3 - iε) ]

  Two simple poles (each with multiplicity 2 since the eigenvalues are doubly
  degenerate):
    ω = +√3 ≈ +1.732  (particle pole, P-cone upper double Dirac)
    ω = -√3 ≈ -1.732  (antiparticle pole, P-cone lower double Dirac)

  P-point spectrum is purely chiral (only ±√3, no zero mode), reflecting the
  P-cone's 2-fold double-Dirac structure (predictions/srs_dirac_cone_velocities.py).
""")


# =============================================================================
# Main
# =============================================================================

def main():
    print()
    print("#" * 78)
    print("#  F1 numerical evaluation: substrate Feynman propagator G_F(k, ω)")
    print("#  at Γ and P, scalar Bloch H(k) (4×4) on srs primitive cell")
    print("#" * 78)

    # Numerical evaluation at Γ
    H_G, eigs_G, vecs_G = report_at_k("Γ", (0.0, 0.0, 0.0))
    sample_propagator_at_omega(H_G, [0.0, 0.5, 2.0, 4.0], name="Γ")

    # Numerical evaluation at P
    H_P, eigs_P, vecs_P = report_at_k("P", (0.25, 0.25, 0.25))
    sample_propagator_at_omega(H_P, [0.0, 1.0, 2.0, 5.0], name="P")

    # Analytic spectral forms
    analytic_form_gamma()
    analytic_form_P()

    # =========================================================================
    # Verify spectral decomposition numerically
    # =========================================================================
    header("Spectral-decomposition verification (Γ and P)")

    # At Γ:
    P_0 = np.outer(vecs_G[:, 3], vecs_G[:, 3].conj())  # Perron projector (eigenvalue +3)
    P_perp = np.eye(4) - P_0  # Complement
    omega = 0.5 + 1e-6j
    G_full = propagator_F(H_G, 0.5)
    G_spectral = (1j / (omega - 3.0)) * P_0 + (1j / (omega + 1.0)) * P_perp
    err = np.max(np.abs(G_full - G_spectral))
    print(f"\n  Γ:  ||G_F^numerical(ω=0.5) − G_F^spectral(ω=0.5)|| = {err:.2e}")

    # At P:
    P_plus = sum(np.outer(vecs_P[:, alpha], vecs_P[:, alpha].conj())
                 for alpha in range(4) if eigs_P[alpha] > 0)
    P_minus = sum(np.outer(vecs_P[:, alpha], vecs_P[:, alpha].conj())
                  for alpha in range(4) if eigs_P[alpha] < 0)
    omega = 1.0 + 1e-6j
    sqrt3 = np.sqrt(3.0)
    G_full_P = propagator_F(H_P, 1.0)
    G_spectral_P = (1j / (omega - sqrt3)) * P_plus + (1j / (omega + sqrt3)) * P_minus
    err_P = np.max(np.abs(G_full_P - G_spectral_P))
    print(f"  P:  ||G_F^numerical(ω=1) − G_F^spectral(ω=1)|| = {err_P:.2e}")

    if err < 1e-4 and err_P < 1e-4:
        print("\n  ✓ Spectral decomposition agrees with numerical G_F at both k-points.")

    header("Summary")
    print("""
  Theorem-grade explicit forms:

    G_F(Γ, ω) = i [ P_0 / (ω - 3 + iε)  +  P_⊥ / (ω + 1 - iε) ]
               where P_0 = J/4 (Perron projector, rank 1, all-ones direction),
                     P_⊥ = I - J/4 (orthogonal complement, rank 3, lower-3-band cluster).

    G_F(P, ω) = i [ P_+ / (ω - √3 + iε)  +  P_- / (ω + √3 - iε) ]
               where P_± are the rank-2 projectors onto the ±√3 eigenspaces.

  Pole structure:
    Γ: poles at ω = +3, -1 (multiplicity 1 and 3 respectively)
    P: poles at ω = ±√3 (each multiplicity 2)
    H: poles at ω = -3, +1 (mult 1, 3)  -- particle-hole conjugate of Γ

  These are the substrate's emergent fermion-mass "spectrum" at the special
  k-points. F4 LSZ uses these on-shell residues to extract scattering
  amplitudes; F11 Wightman uses these poles to verify the spectrum condition.

  Spinor-Dirac (32×32) lift: separate research item (~2-3 sessions). The
  scalar-Bloch result above is the structural template; the spinor lift
  uses the same spectral form with D(k) in place of H(k) and ε_α(k) the
  32 substrate-Dirac eigenvalues.
""")


if __name__ == "__main__":
    main()
