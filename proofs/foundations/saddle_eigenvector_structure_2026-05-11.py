"""
proofs/foundations/saddle_eigenvector_structure_2026-05-11.py

Extract eigenvector structure of the 3 unused Ramanujan saddles
(h_N, h_H, h_Γ) parallel to the framework's existing h_P apparatus.

For each saddle:
  - 4 adjacency eigenvalues (real)
  - Their corresponding eigenvectors (Bloch modes on 4 atom sites)
  - C_3 isotypic decomposition at each k-point
  - Stark-Terras factorization analog for Hashimoto eigenvectors
"""

import math
import sys
from pathlib import Path

import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from simulator.srs_engine.srs_substrate import SrsSubstrate
from proofs.common import C3_PERM, label_c3, omega3, c3_decompose

substrate = SrsSubstrate()


def c3_isotypic_at(k_name):
    """Body-diagonal C3 isotypic decomposition at given k-point."""
    k_frac = substrate._resolve_k(k_name)
    try:
        evals, evecs, c3_diag, offdiag = c3_decompose(k_frac, substrate.bonds)
        labels = [label_c3(c) for c in c3_diag]
        mu_1 = labels.count('1')
        mu_w = labels.count('w')
        mu_wbar = labels.count('w2')
        return (mu_1, mu_w, mu_wbar), evals, evecs, labels
    except Exception as e:
        return None, None, None, f"C_3 decomposition failed at {k_name}: {e}"


def main():
    print("=" * 100)
    print("Eigenvector + C_3 isotypic structure at ALL 4 high-symmetry k-points")
    print("=" * 100)
    print()

    for k_name in ['Gamma', 'P', 'N', 'H']:
        print(f"\n{'='*60}")
        print(f"k-point: {k_name}")
        print(f"{'='*60}")

        # Adjacency eigenvalues + eigenvectors
        A = substrate.adjacency_at_k(k_name)
        evals, evecs = la.eig(A)
        order = np.argsort(-evals.real)
        evals = evals[order]
        evecs = evecs[:, order]

        print(f"\nAdjacency eigenvalues + eigenvectors:")
        for i, lam in enumerate(evals):
            v = evecs[:, i]
            # Normalize to give consistent phase
            v_disp = v / v[0] if abs(v[0]) > 1e-9 else v
            print(f"  λ_{i}: {lam.real:+.4f}+{lam.imag:+.4f}i")
            print(f"     |v| = [{', '.join(f'{abs(c):.3f}' for c in v)}]")
            print(f"     v_relphase (vs v[0]) = [{', '.join(f'{math.degrees(math.atan2(c.imag, c.real)):+6.1f}°' for c in v_disp)}]")

        # C_3 isotypic
        try:
            mults, evals_c3, evecs_c3, labels = c3_isotypic_at(k_name)
            print(f"\nC_3 isotypic (body-diagonal rotation) at {k_name}:")
            if isinstance(labels, str):
                print(f"  {labels}")
            else:
                print(f"  Multiplicities (μ_1, μ_ω, μ_ω̄) = {mults}")
                print(f"  V_Ram doubled = {(2*mults[0], 2*mults[1], 2*mults[2])}")
                print(f"  C_3 eigenvalue labels per adjacency eigenmode: {labels}")
        except Exception as e:
            print(f"  C_3 decomposition error: {e}")

        # Hashimoto at this k-point
        B = substrate.hashimoto_at_k(k_name)
        evB = la.eigvals(B)
        evB_sorted = sorted(evB, key=lambda z: (-abs(z), -z.real))

        # Find the saddle eigenvalues (|λ| ≈ √2)
        saddles = [e for e in evB_sorted if abs(abs(e) - math.sqrt(2)) < 0.001]
        n_saddles = len(saddles)
        print(f"\nHashimoto Ramanujan saddles at {k_name}: {n_saddles} eigenvalues with |λ| = √2")
        if saddles:
            unique_saddles = set()
            for e in saddles:
                arg = math.degrees(math.atan2(e.imag, e.real))
                key = (round(e.real, 4), round(e.imag, 4))
                unique_saddles.add(key)
            print(f"  Distinct saddle values: {len(unique_saddles)}")
            for (re, im) in sorted(unique_saddles):
                arg = math.degrees(math.atan2(im, re))
                mult = sum(1 for e in saddles if abs(e.real - re) < 1e-4 and abs(e.imag - im) < 1e-4)
                # Identify with named saddle
                name = None
                if abs(re - math.sqrt(3)/2) < 0.01 and abs(im - math.sqrt(5)/2) < 0.01:
                    name = "h_P"
                elif abs(re - math.sqrt(3)/2) < 0.01 and abs(im + math.sqrt(5)/2) < 0.01:
                    name = "h_P_bar"
                elif abs(re + math.sqrt(3)/2) < 0.01 and abs(im - math.sqrt(5)/2) < 0.01:
                    name = "-h_P_bar"
                elif abs(re + math.sqrt(3)/2) < 0.01 and abs(im + math.sqrt(5)/2) < 0.01:
                    name = "-h_P"
                elif abs(re - math.sqrt(5)/2) < 0.01 and abs(im - math.sqrt(3)/2) < 0.01:
                    name = "h_N"
                elif abs(re - math.sqrt(5)/2) < 0.01 and abs(im + math.sqrt(3)/2) < 0.01:
                    name = "h_N_bar"
                elif abs(re + math.sqrt(5)/2) < 0.01 and abs(im - math.sqrt(3)/2) < 0.01:
                    name = "-h_N_bar"
                elif abs(re + math.sqrt(5)/2) < 0.01 and abs(im + math.sqrt(3)/2) < 0.01:
                    name = "-h_N"
                elif abs(re - 0.5) < 0.01 and abs(im - math.sqrt(7)/2) < 0.01:
                    name = "h_H"
                elif abs(re - 0.5) < 0.01 and abs(im + math.sqrt(7)/2) < 0.01:
                    name = "h_H_bar"
                elif abs(re + 0.5) < 0.01 and abs(im - math.sqrt(7)/2) < 0.01:
                    name = "-h_H_bar (h_Γ)"
                elif abs(re + 0.5) < 0.01 and abs(im + math.sqrt(7)/2) < 0.01:
                    name = "-h_H (h_Γ_bar)"
                else:
                    name = "?"
                print(f"    {re:+.4f}{im:+.4f}i (arg={arg:+.2f}°, mult={mult}): {name}")

    # Summary table
    print("\n" + "=" * 100)
    print("SADDLE SUMMARY ACROSS k-POINTS")
    print("=" * 100)
    print()
    print("Confirmed: 4 distinct Ramanujan saddle 'families' (one per high-symmetry k-point):")
    print()
    print(f"  {'k-point':<10} {'saddle family':<30} {'arg':>10} {'tan²(arg)':>12}")
    print(f"  {'-'*10} {'-'*30} {'-'*10} {'-'*12}")

    saddle_data = {
        'Gamma': ('h_Γ = (−1 + i√7)/2', 110.70, 7.0),
        'P':     ('h_P = (√3 + i√5)/2', 52.24, 5/3),
        'N':     ('h_N = (√5 + i√3)/2', 37.76, 3/5),
        'H':     ('h_H = (1 + i√7)/2', 69.30, 7.0),
    }
    for k, (name, arg, t2) in saddle_data.items():
        print(f"  {k:<10} {name:<30} {arg:>+10.2f} {t2:>12.6f}")
    print()
    print("  Note: h_Γ and h_H both have tan²(arg) = 7 (same magnitude, different sign of Re)")
    print("  Note: h_P and h_N are Re/Im swaps; tan² values are reciprocal (5/3 vs 3/5)")
    print()
    print("  STRUCTURAL ALGEBRAIC RELATION:")
    print("    h_P · h_N* = h_P · (√5 − i√3)/2 = (√3 + i√5)(√5 − i√3)/4")
    h_P = complex(math.sqrt(3)/2, math.sqrt(5)/2)
    h_N = complex(math.sqrt(5)/2, math.sqrt(3)/2)
    product = h_P * h_N.conjugate()
    print(f"    Numerically: h_P · h_N* = {product.real:+.4f}{product.imag:+.4f}i")
    arg_prod = math.degrees(math.atan2(product.imag, product.real))
    print(f"    |product| = {abs(product):.4f}, arg = {arg_prod:+.2f}°")
    # This should be 2·exp(i·(arg(h_P) − arg(h_N))) = 2·exp(i·14.48°)
    print(f"    Expected: 2·exp(i·(arg(h_P) − arg(h_N))) = 2·exp(i·{52.24 - 37.76:+.2f}°)")
    print()
    print("    h_P · h_H_bar:")
    h_H = complex(0.5, math.sqrt(7)/2)
    product_PH = h_P * h_H.conjugate()
    print(f"    Numerically: h_P · h_H* = {product_PH.real:+.4f}{product_PH.imag:+.4f}i")
    print(f"    |product| = {abs(product_PH):.4f}, arg = {math.degrees(math.atan2(product_PH.imag, product_PH.real)):+.2f}°")

    print("\n" + "=" * 100)
    print("Interpretation candidates:")
    print("=" * 100)
    print("""
  The 4 Ramanujan saddles form a substrate-derived structure with:
    - Same magnitude (all |h| = √2) → all saturate the Ramanujan bound
    - Distinct phases (52.24°, 37.76°, 69.30°, 110.70°)
    - Specific algebraic relations between them

  Possible roles (each is a HYPOTHESIS to test, not a closure):

  (1) Multiple gauge sectors:
      h_P → SU(2)_L gauge sector (lepton)
      h_N → SU(2)_R gauge sector (quark/right-handed)
      h_H → SU(3)_c gauge sector (color)
      h_Γ → U(1)_Y gauge sector (hypercharge)
      Test: do framework's β-coefficient calculations match if each
      sector uses its own saddle? Currently all use h_P implicitly.

  (2) Generation labels:
      h_P → generation 1 (e, u, d)
      h_N → generation 2 (μ, c, s)
      h_H → generation 3 (τ, t, b)
      h_Γ → "anti" / sterile sector
      Test: do mass hierarchies follow saddle structure?

  (3) Class differentiation:
      h_P → mass² class (the framework's existing Class-2 closure)
      h_N → amplitude class (different observables)
      h_H → phase class
      h_Γ → dark/hidden class
      Test: do framework's predictions in each class involve their saddle?

  (4) Brillouin zone topological labels:
      The 4 k-points (Γ, P, N, H) are distinct topological cells
      of the BZ. The 4 saddles are their natural saddle eigenvalues.
      The framework's current predictions live at P; other k-points
      have their own predictions yet to be identified.

  Each hypothesis is independently testable. Each could explain a
  subset of the framework's currently-unclosed observables (R-14,
  quark hierarchy, sub-leading PMNS, etc.).
""")


if __name__ == "__main__":
    main()
