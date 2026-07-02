#!/usr/bin/env python3
"""
n2_directed_walk_chirality_split_2026-05-19.py

TEST: is the up-type / down-type (n=2 / n=1) Yukawa-anchor splitting carried by
the parity-odd handedness of the COMPLEX DIRECTED non-backtracking walk,
restricted to the n=2 two-directed-edge sector?

WHY THIS PROBE EXISTS
---------------------
The R1 isotypic obstruction (`R14_R1_C3_isotypic_yukawa_2026-05-05.md`) proved
that *symmetry-character* operators cannot tell n=1 (down) from n=2 (up): Λ¹ and
Λ² have identical C₃ isotypic content (Hodge duality at k*=3), so any
character-only operator gives y_b = y_t. That obstruction is real but NARROW —
it is blind to ORIENTATION. A single directed edge (n=1) is a polar object with
no internal winding; an ordered pair of directed edges (n=2) is an oriented
object with a circulation sense (a handedness). The non-backtracking (Hashimoto)
walk B is intrinsically directed and its P-saddle eigenvalue is COMPLEX,
h = (√3 + i√5)/2 — the imaginary part IS the handedness. The framework's whole
dark-correction mechanism is already a parity-odd projection of this complex h
(master doc §1); every prior application used the *symmetric* (sign-blind
tan²(arg h)) reading. The asymmetric reading is the master doc's NAMED-OPEN
"Family E / R_asymmetric / propagator custodial-breaking" slot.

CLAIM UNDER TEST
----------------
Decompose the level-n directed walk B^n (n = number of directed edges =
resolvent power in G_NB = (I-uB)^-1 = Σ uⁿBⁿ, the over-determination test's
central object) by the handedness grading χ_Im = sign(Im B) (the R-15 Session 1
construction). Then:

    n=1 (single directed edge, POLAR):   Σ_+ == Σ_-   (handedness-symmetric)
    n=2 (two directed edges, ORIENTED):  Σ_+ != Σ_-   (handedness-broken)

i.e. the up/down split is the n=2 directed-walk chirality, absent at n=1.

PRE-DECLARED ABORTS / FALSIFICATION (written BEFORE any number is computed)
--------------------------------------------------------------------------
  A1  VALIDITY.   χ_Im must be a conserved Z₂ on V_Ram: ||[χ_Im,B]|| ~ 0 and
                  χ_Im² = I. If [χ_Im,B] is not ~0 the decomposition is
                  ill-defined  ->  ABORT (probe invalid, NOT a result).
  A2  n=1 CONTROL. If the n=1 (single directed edge) handedness asymmetry is
                  NOT ~0, the asymmetry is not n=2-specific  ->  hypothesis
                  FALSIFIED ("n=2's natural chirality" is not the mechanism).
  A3  n=2 SIGNAL. If the n=2 handedness asymmetry IS ~0 (same as n=1), the
                  directed n=2 walk does NOT break handedness  ->  FALSIFIED.
  A4  CHIRALITY-SPECIFICITY. Repeat with χ_Re (Hermitian, conserved,
                  = particle/antiparticle, NOT handedness). If χ_Re reproduces
                  the same n=2 asymmetry as χ_Im, the effect is a generic
                  grading artifact, not chirality  ->  NOT SUPPORTED.
  A5  PARITY-ODD. The n=2 asymmetry must be parity-ODD: under Im(h) -> -Im(h)
                  (complex conjugation of the walk) it must flip sign / swap
                  the +/- sectors. If invariant, it is not parity-odd  ->
                  NOT SUPPORTED (not the claimed chirality mechanism).
  A6  SMUGGLE.    Every constant traces to a prior closure (enumerated in the
                  provenance dict). None is tuned to pass. By construction.

PASS  = A1 holds AND A2 holds (n=1 symmetric) AND A3 holds (n=2 asymmetric)
        AND A4 holds (χ_Im-specific) AND A5 holds (parity-odd).
Any other outcome is an HONEST NEGATIVE and is reported as such.

HONEST SCOPE (printed in the verdict, not hidden)
  A PASS establishes the MECHANISM EXISTS (the directed n=2 walk breaks
  handedness where n=1 does not). It does NOT produce y_t, the y_t/y_b ≈ 41,
  the y_t/y_τ ≈ 97, or ANY number. It is a mechanism-existence / channel test
  (same standing as the over-determination test). The known open caveat —
  χ_Im is non-Hermitian on V_Ram because B is non-normal there (R-15 Session
  1; the pseudo-Hermitian inner-product question) — is REPORTED, not closed.

Constants & machinery — all bound to existing theorem-grade objects:
  - srs net / bonds            proofs.common.find_bonds
  - directed Hashimoto B, K_P,
    H_EXACT, C₃ operator       proofs.foundations.theorem_B5_3_core
  - V_Ram (8-dim Ramanujan)    proofs.foundations.cocycle_check_vram
  - χ_Im / χ_Re construction   R15_session_1_trivial_C3_chirality_decomp.py
                               (spectral-projector sign(Im)/sign(Re), lifted
                                from dim-4 trivial-C₃ to the full 8-dim V_Ram)
  - n = resolvent power Bⁿ     theorem_unified_oblique.md §8 / the 2026-05-16
                               over-determination test (G_NB = Σ uⁿ Bⁿ)
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from numpy import linalg as la

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import find_bonds
from proofs.foundations.theorem_B5_3_core import (
    K_P, H_EXACT, build_directed_edges, bloch_hashimoto,
)
from proofs.foundations.cocycle_check_vram import find_vram_basis

TOL_ZERO = 1e-6          # "~0" threshold for aborts A1/A2/A3
np.set_printoptions(precision=5, suppress=True, linewidth=140)


# --------------------------------------------------------------------------
# Build the Ramanujan-restricted directed walk and the Z₂ gradings
# --------------------------------------------------------------------------
def build_B_VR():
    """B restricted to the 8-dim Ramanujan subspace V_Ram at the P-saddle.

    Exactly the R-15 Session 1 construction (no k-grid sampling, no tolerance
    fuzz: at K_P the Ramanujan modes are exactly {±h, ±h̄})."""
    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    B_P = bloch_hashimoto(K_P, directed)
    V_raw = find_vram_basis(B_P, H_EXACT)
    Q, _ = la.qr(np.asarray(V_raw))
    V_Ram = Q[:, :8]
    B_VR = V_Ram.conj().T @ B_P @ V_Ram
    return B_VR


def build_sign_grading(B_VR, part):
    """χ = sign(part(eigenvalue)) as a conserved Z₂ on V_Ram, built from the
    spectral projectors of B (R-15 Session 1 Part C, lifted dim-4 -> dim-8).

    part = 'im'  -> handedness grading χ_Im   (the hypothesis's grading)
    part = 're'  -> particle/antiparticle χ_Re (the Hermitian control, A4)
    """
    evals, evecs = la.eig(B_VR)
    if part == 'im':
        signs = np.array([1.0 if ev.imag > 0 else -1.0 for ev in evals])
    elif part == 're':
        signs = np.array([1.0 if ev.real > 0 else -1.0 for ev in evals])
    else:
        raise ValueError(part)
    P = evecs
    chi = P @ np.diag(signs.astype(complex)) @ la.inv(P)
    return chi


def level_n_chi_split(B_VR, chi, n):
    """Handedness-graded level-n directed-walk amplitude on V_Ram.

    P_± = (I ± χ)/2 are the two grading projectors. The level-n directed
    object is Bⁿ (n directed edges = the n-th resolvent power, the
    over-determination test's object). The grading-g normalized n-step
    return amplitude is

        Σ^(n)_g = Tr( P_g · Bⁿ · P_g ) / Tr(P_g).

    Returns (Σ_+, Σ_-) as complex scalars.
    """
    I = np.eye(B_VR.shape[0], dtype=complex)
    P_plus = (I + chi) / 2.0
    P_minus = (I - chi) / 2.0
    Bn = la.matrix_power(B_VR, n)
    tr_pp = np.trace(P_plus)
    tr_pm = np.trace(P_minus)
    s_plus = np.trace(P_plus @ Bn @ P_plus) / tr_pp
    s_minus = np.trace(P_minus @ Bn @ P_minus) / tr_pm
    return complex(s_plus), complex(s_minus)


def asym(s_plus, s_minus):
    """Scalar handedness asymmetry: |Σ_+ − Σ_-| (and its parts)."""
    d = s_plus - s_minus
    return abs(d), d.real, d.imag


# --------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------
def main():
    print("=" * 84)
    print("n=2 DIRECTED-WALK CHIRALITY SPLIT — does handedness split up from down?")
    print("=" * 84)

    B_VR = build_B_VR()
    h = complex(H_EXACT)
    print(f"\n  P-saddle eigenvalue h = ({h.real:.6f} + {h.imag:.6f}i)   "
          f"|h|² = {abs(h)**2:.6f}  (= k*-1 = 2)")
    evals = np.sort_complex(la.eigvals(B_VR))
    print(f"  B|_V_Ram spectrum (8 Ramanujan modes): "
          f"{[f'{e:.3f}' for e in evals]}")

    chi_Im = build_sign_grading(B_VR, 'im')   # subject: handedness
    chi_Re = build_sign_grading(B_VR, 're')   # control A4: particle/antipart.

    fails = []

    # ---- A1 VALIDITY: χ_Im conserved Z₂ -------------------------------------
    comm = la.norm(chi_Im @ B_VR - B_VR @ chi_Im)
    sq = la.norm(chi_Im @ chi_Im - np.eye(8))
    herm = la.norm(chi_Im - chi_Im.conj().T)
    print("\n" + "-" * 84)
    print("A1  VALIDITY — χ_Im a conserved Z₂ on V_Ram?")
    print("-" * 84)
    print(f"  ||[χ_Im, B]||  = {comm:.2e}   (need ~0: conserved)")
    print(f"  ||χ_Im² − I||  = {sq:.2e}   (need ~0: Z₂)")
    print(f"  ||χ_Im − χ_Im†|| = {herm:.3f}   "
          f"(non-zero EXPECTED — the known open inner-product caveat, R-15 S1)")
    a1_ok = comm < TOL_ZERO and sq < TOL_ZERO
    if not a1_ok:
        fails.append("A1 (χ_Im not a conserved Z₂ — decomposition ill-defined)")

    # ---- compute the level-n splits (χ_Im subject, χ_Re control) ------------
    print("\n" + "-" * 84)
    print("LEVEL-n HANDEDNESS SPLIT   Σ^(n)_g = Tr(P_g Bⁿ P_g)/Tr(P_g)")
    print("-" * 84)
    res = {}
    for tag, chi in (("χ_Im (handedness, subject)", chi_Im),
                     ("χ_Re (partcl/antiprt, control A4)", chi_Re)):
        print(f"\n  grading = {tag}")
        for n in (1, 2):
            sp, sm = level_n_chi_split(B_VR, chi, n)
            mag, dre, dim_ = asym(sp, sm)
            res[(tag, n)] = (sp, sm, mag)
            print(f"    n={n}:  Σ_+ = {sp:+.6f}   Σ_- = {sm:+.6f}")
            print(f"          |Σ_+ − Σ_-| = {mag:.6e}  "
                  f"(ΔRe={dre:+.3e}, ΔIm={dim_:+.3e})")

    im1 = res[("χ_Im (handedness, subject)", 1)][2]
    im2 = res[("χ_Im (handedness, subject)", 2)][2]
    re2 = res[("χ_Re (partcl/antiprt, control A4)", 2)][2]

    # ---- A2 n=1 must be symmetric ------------------------------------------
    print("\n" + "-" * 84)
    print("A2  n=1 CONTROL — single directed edge (POLAR) must be symmetric")
    print("-" * 84)
    a2_ok = im1 < TOL_ZERO
    print(f"  n=1 χ_Im asymmetry = {im1:.3e}   "
          f"({'~0  PASS' if a2_ok else 'NONZERO  FALSIFIES n=2-specificity'})")
    if not a2_ok:
        fails.append("A2 (n=1 already handedness-asymmetric — not n=2-specific)")

    # ---- A3 n=2 must be asymmetric -----------------------------------------
    print("\n" + "-" * 84)
    print("A3  n=2 SIGNAL — two directed edges (ORIENTED) must break handedness")
    print("-" * 84)
    a3_ok = im2 > TOL_ZERO
    print(f"  n=2 χ_Im asymmetry = {im2:.3e}   "
          f"({'NONZERO  PASS' if a3_ok else '~0  FALSIFIES the hypothesis'})")
    if not a3_ok:
        fails.append("A3 (n=2 directed walk does NOT break handedness)")

    # ---- A4 chirality-specificity ------------------------------------------
    print("\n" + "-" * 84)
    print("A4  CHIRALITY-SPECIFICITY — χ_Re (Hermitian, NOT handedness) control")
    print("-" * 84)
    ratio = im2 / re2 if re2 > 1e-15 else float('inf')
    a4_ok = (im2 > TOL_ZERO) and (re2 < TOL_ZERO or ratio > 2.0)
    print(f"  n=2 χ_Im asym = {im2:.3e}   n=2 χ_Re asym = {re2:.3e}   "
          f"ratio = {ratio:.2f}")
    print(f"  need χ_Im n=2 effect distinct from χ_Re (ratio>2 or χ_Re~0): "
          f"{'PASS' if a4_ok else 'NOT SUPPORTED (generic grading artifact)'}")
    if not a4_ok:
        fails.append("A4 (n=2 asymmetry not χ_Im-specific — generic artifact)")

    # ---- A5 parity-odd: Im(h) -> -Im(h) (conjugate the walk) ---------------
    print("\n" + "-" * 84)
    print("A5  PARITY-ODD — under Im(h) → −Im(h) the n=2 split must flip/swap")
    print("-" * 84)
    B_conj = B_VR.conj()
    chi_Im_c = build_sign_grading(B_conj, 'im')
    sp2, sm2 = level_n_chi_split(B_VR, chi_Im, 2)
    spc, smc = level_n_chi_split(B_conj, chi_Im_c, 2)
    # parity-odd signature: the +/- imaginary parts swap under conjugation
    swap_err = abs((sp2 - sm2) + (spc - smc))   # odd  -> ~0 ; even -> ~2|Δ|
    even_ref = abs((sp2 - sm2))
    a5_ok = (even_ref > TOL_ZERO) and (swap_err < 0.25 * even_ref)
    print(f"  Δ(n=2, h)        = {sp2 - sm2:+.6e}")
    print(f"  Δ(n=2, conj h)   = {spc - smc:+.6e}")
    print(f"  ||Δ + Δ_conj||   = {swap_err:.3e}   (parity-ODD ⇒ ~0; "
          f"need < 25% of |Δ| = {0.25*even_ref:.3e})")
    print(f"  parity-odd: {'PASS' if a5_ok else 'NOT SUPPORTED (asym not parity-odd)'}")
    if not a5_ok:
        fails.append("A5 (n=2 asymmetry is not parity-odd)")

    # ---- A6 smuggle audit ---------------------------------------------------
    provenance = {
        "h_P=(√3+i√5)/2": "theorem_B5_3_core.H_EXACT (Row P52, Ramanujan at P)",
        "K_P=(1/4,1/4,1/4)": "theorem_B5_3_core.K_P (P-saddle k-point)",
        "srs bonds / directed B": "proofs.common.find_bonds + theorem_B5_3_core",
        "V_Ram (8-dim)": "cocycle_check_vram.find_vram_basis (Ramanujan subspace)",
        "χ_Im=sign(Im B), χ_Re=sign(Re B)": "R15_session_1 Part C construction",
        "n = Bⁿ (resolvent power)": "over-determination test G_NB=Σ uⁿBⁿ (b741e08)",
    }
    print("\n" + "-" * 84)
    print(f"A6  SMUGGLE AUDIT — {len(provenance)} constants, each from a prior closure")
    print("-" * 84)
    for k, v in provenance.items():
        print(f"    {k:<34} <- {v}")
    print("  zero constants tuned to pass (by construction).")

    # ---- verdict ------------------------------------------------------------
    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    if not fails:
        print(f"""
  PASS — all pre-declared aborts cleared.

    n=1 (single directed edge, polar):  handedness-SYMMETRIC  ({im1:.2e})
    n=2 (two directed edges, oriented): handedness-BROKEN     ({im2:.2e})
    effect is χ_Im-specific (vs χ_Re control) and parity-ODD (flips under
    Im(h) → −Im(h)).

  ⇒ The up-type/down-type (n=2/n=1) split IS carried by the parity-odd
    chirality of the complex DIRECTED walk on the n=2 two-directed-edge
    sector — exactly the master doc's named-open Family E / R_asymmetric
    slot, and consistent with the 2026-05-17 §6(i) "mass ∝ 1/inverse-
    propagator" structural theorem (the dynamical-mass principle this
    channel would feed).

  HONEST SCOPE — what this does NOT do:
   • No number. It does NOT produce y_t, y_t/y_b ≈ 41, or y_t/y_τ ≈ 97.
     It establishes the CHANNEL EXISTS (mechanism-existence test, same
     standing as the over-determination test), not the coupling value.
   • The χ_Im non-Hermiticity on V_Ram (B non-normal — R-15 S1) is the
     standing open caveat: identifying χ_Im with PHYSICAL chirality still
     needs the pseudo-Hermitian inner product (Mostafazadeh) OR γ₅ imported
     from the B3 spinor structure. REPORTED, not closed.
   • Re-locates "σ₊-nilpotent, no route" → "the directed n=2 walk in a
     pseudo-Hermitian inner product" — a sharp, named, attackable seam.
""")
        rc = 0
    else:
        print("\n  HONEST NEGATIVE — pre-declared abort(s) hit:\n")
        for f in fails:
            print(f"    ✗ {f}")
        print("""
  The hypothesis "up/down split = parity-odd chirality of the directed n=2
  walk" is, as stated, NOT supported by this probe. This is a disciplined
  route-elimination (Need-B-style), not a failure: it sharpens where the
  mechanism cannot be. No number was produced or claimed either way.
""")
        rc = 1
    print("=" * 84)
    return rc


if __name__ == "__main__":
    sys.exit(main())
