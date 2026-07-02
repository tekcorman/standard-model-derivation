"""
Spectral-triple choice-axis menu — framework-slice NCG variant enumeration.

Per the M-arc theory writeup,
the framework's spectral triple (A_F, H_F, D_F) is a KO-dim 0 GNS spectral
triple (NOT a standard Chamseddine-Connes almost-commutative SM model), and
several explicit choices remain to be made BEFORE the CC reductions M3-M5
can be carried out unambiguously.

This module enumerates the choices ONLY.  Hard-constraint filtering
(Lie-algebra closure, gauge equivariance with D_F, distinct gauge factors
commute, CC spectral-triple axioms for J) lives in
`simulator.gating.spectral_consistency`.

Choice axes
-----------
(1) J real-structure variant      — J^(α): X ↦ X̄, J^(β): X ↦ X†
(2) Basis on Cl(6) Fock per vertex — Brauer-Weyl, B3 species, C_3 eigenbasis,
                                     PS tensor-product (not yet built)
(3) SU(3) embedding in SU(4)       — standard Gell-Mann upper-3×3 (acts on
                                     basis vectors e_1,e_2,e_3, fixes e_4),
                                     aligned-with-PS-tensor-product (requires
                                     basis #4 above; flagged "not built").
(4) U(1)_Y formula                 — Y = T_3R + (B−L)/2  (Pati-Salam standard)
                                     Y = (3/5)·(B−L) + T_3R  (SU(5) GUT
                                     normalisation, per
                                     theorem_sin2_theta_W_unification §11).
(5) Inner-fluctuation reduction    — KO-dim 0 sign (JΦJ = +Φ) or KO-dim 6
    convention                       sign (JΦJ = −Φ).  Determines how the
                                     1-form module Ω_D^1 (= 1536-dim raw) is
                                     reduced to physical Higgs scalars.

The five axes together generate 2 × 4 × 2 × 2 × 2 = 64 candidate tuples
(2 × 3 × 2 × 2 × 2 = 48 if the "PS tensor-product" basis is skipped on the
first pass).  Each tuple goes through C1-C4 in the gating layer.

NB: enumeration is PURE — no numpy operator construction here.  The gate
builds operators from the (choice, framework data) pair and checks hard
constraints at machine precision.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Choice axis: J real-structure
# ---------------------------------------------------------------------------

J_VARIANTS = ('alpha', 'beta')
J_VARIANT_DESC = {
    'alpha': 'J^(α)(X) = X̄  (entrywise complex conjugate)',
    'beta':  'J^(β)(X) = X†  (Hermitian adjoint)',
}


# ---------------------------------------------------------------------------
# Choice axis: basis on Cl(6) Fock per vertex
# ---------------------------------------------------------------------------

BASES = ('brauer_weyl', 'b3_species', 'c3_eigenbasis', 'ps_tensor_product')
BASIS_DESC = {
    'brauer_weyl':       'Brauer-Weyl: Γ_a per theorem_B3_spinor_fermion.py '
                         '(Cl(6) Fock = ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2 qubit factors)',
    'b3_species':        'B3 species: simultaneous eigenbasis of '
                         '(T_1, T_2, Y) Spin(6) Cartan triple',
    'c3_eigenbasis':     'C_3 eigenbasis: simultaneous eigenbasis of '
                         'body-diagonal C_3 (per theorem_B3_B6_reconciliation)',
    'ps_tensor_product': 'Pati-Salam tensor product: Cl(6) Fock = ℂ^4_{SU(4)} '
                         '⊗ ℂ^2_{SU(2)_L} (NOT yet built; M2.refined deliverable)',
}
BASIS_BUILT = {
    'brauer_weyl':       True,
    'b3_species':        True,
    'c3_eigenbasis':     True,
    'ps_tensor_product': False,
}


# ---------------------------------------------------------------------------
# Choice axis: SU(3) embedding in SU(4) = Spin(6)
# ---------------------------------------------------------------------------

SU3_EMBEDDINGS = ('gell_mann_upper3', 'aligned_with_ps_tensor')
SU3_EMBEDDING_DESC = {
    'gell_mann_upper3':      'Standard λ^a/2 on (e_1, e_2, e_3) of 4-rep, '
                             'fixes e_4 ("lepton")',
    'aligned_with_ps_tensor': 'SU(3) acts on the SU(4) factor of the PS '
                              'tensor-product basis; commutes with SU(2)_L by '
                              'construction (REQUIRES ps_tensor_product basis)',
}


# ---------------------------------------------------------------------------
# Choice axis: U(1)_Y formula
# ---------------------------------------------------------------------------

U1Y_FORMULAS = ('t3r_plus_bminusL_over_2', 'three_fifths_bminusL_plus_t3r')
U1Y_FORMULA_DESC = {
    't3r_plus_bminusL_over_2':     'Y = T_3R + (B−L)/2  (Pati-Salam standard)',
    'three_fifths_bminusL_plus_t3r': 'Y = (3/5)·(B−L) + T_3R  (SU(5) GUT, '
                                     'theorem_sin2_theta_W_unification §11)',
}


# ---------------------------------------------------------------------------
# Choice axis: inner-fluctuation reduction convention
# ---------------------------------------------------------------------------

REDUCTIONS = ('ko_dim_0', 'ko_dim_6')
REDUCTION_DESC = {
    'ko_dim_0': 'KO-dim 0 sign: JΦJ = +Φ  (framework-natural per M1)',
    'ko_dim_6': 'KO-dim 6 sign: JΦJ = −Φ  (CC SM standard)',
}


# ---------------------------------------------------------------------------
# Tuple dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralTripleChoice:
    """One candidate tuple of NCG choices for the framework's spectral triple.

    Attributes:
        j_variant      : 'alpha' | 'beta'  (see J_VARIANTS)
        basis          : 'brauer_weyl' | 'b3_species' | 'c3_eigenbasis' |
                         'ps_tensor_product'
        su3_embedding  : 'gell_mann_upper3' | 'aligned_with_ps_tensor'
        u1y_formula    : 't3r_plus_bminusL_over_2' |
                         'three_fifths_bminusL_plus_t3r'
        reduction      : 'ko_dim_0' | 'ko_dim_6'
    """
    j_variant: str
    basis: str
    su3_embedding: str
    u1y_formula: str
    reduction: str

    @property
    def name(self) -> str:
        return (f'(J={self.j_variant}, basis={self.basis}, '
                f'SU3={self.su3_embedding}, '
                f'Y={self.u1y_formula}, red={self.reduction})')

    @property
    def is_constructable(self) -> bool:
        """Whether the gate has the explicit operator construction for this tuple.

        False if the basis is `ps_tensor_product` (not yet built) OR if the
        SU(3) embedding is `aligned_with_ps_tensor` (requires that basis).
        """
        if not BASIS_BUILT.get(self.basis, False):
            return False
        if self.su3_embedding == 'aligned_with_ps_tensor' \
                and self.basis != 'ps_tensor_product':
            return False
        return True


# ---------------------------------------------------------------------------
# Enumerators
# ---------------------------------------------------------------------------

def enumerate_j_variants() -> list[str]:
    return list(J_VARIANTS)


def enumerate_bases(include_unbuilt: bool = True) -> list[str]:
    if include_unbuilt:
        return list(BASES)
    return [b for b in BASES if BASIS_BUILT[b]]


def enumerate_su3_embeddings() -> list[str]:
    return list(SU3_EMBEDDINGS)


def enumerate_u1y_formulas() -> list[str]:
    return list(U1Y_FORMULAS)


def enumerate_reductions() -> list[str]:
    return list(REDUCTIONS)


def enumerate_full_menu(include_unbuilt_basis: bool = False
                        ) -> list[SpectralTripleChoice]:
    """Cross-product of the five choice axes.

    With `include_unbuilt_basis=False` (default), the `ps_tensor_product`
    basis is excluded and the cross-product is 2 × 3 × 2 × 2 × 2 = 48.
    With `include_unbuilt_basis=True` we get 2 × 4 × 2 × 2 × 2 = 64; the
    extra 16 candidates are flagged `not_constructable` in the gating layer.
    """
    out: list[SpectralTripleChoice] = []
    bases = enumerate_bases(include_unbuilt=include_unbuilt_basis)
    for jv in J_VARIANTS:
        for b in bases:
            for su3 in SU3_EMBEDDINGS:
                for y in U1Y_FORMULAS:
                    for red in REDUCTIONS:
                        out.append(SpectralTripleChoice(
                            j_variant=jv,
                            basis=b,
                            su3_embedding=su3,
                            u1y_formula=y,
                            reduction=red,
                        ))
    return out


# ---------------------------------------------------------------------------
# Convenience: human-readable summary of a choice tuple
# ---------------------------------------------------------------------------

def describe_choice(c: SpectralTripleChoice) -> str:
    return (
        f'  J real-structure: {J_VARIANT_DESC[c.j_variant]}\n'
        f'  Basis:            {BASIS_DESC[c.basis]}\n'
        f'  SU(3) embedding:  {SU3_EMBEDDING_DESC[c.su3_embedding]}\n'
        f'  U(1)_Y formula:   {U1Y_FORMULA_DESC[c.u1y_formula]}\n'
        f'  Reduction:        {REDUCTION_DESC[c.reduction]}'
    )
