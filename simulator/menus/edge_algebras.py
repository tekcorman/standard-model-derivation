"""
Edge qubit-algebra menu — candidate algebras at each substrate directed edge.

Per Task B (commit 7748658, 2026-05-07): the edge qubit carries a local
algebra distinct from the vertex algebra. The framework's dominant retention
at the edge is Cl(0, 2) ≅ ℍ (theorem-grade per `theorem_g2_edge_qubit_su2.md`,
Aut(ℍ) ⊃ SU(2)_L × SU(2)_R of the Pati-Salam left/right doublets).

Alternatives at the edge layer (plurally co-retained at framework scale per
the Task B / cooling-cascade enumeration):
  - Cl(0, p) for p = 0, 1, 2, 3, 4, … with signature (0, p)
  - Cayley-Dickson at the edge level (𝕆, etc.)

NB: enumerator only. MDL gating in simulator.gating. The numeric
N_attest data mirror proofs/foundations/sector_cooling_cascade_audit.py (edge_data).
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EdgeAlgebra:
    """One candidate edge-qubit algebra.

    Attributes:
        name           : 'Cl(0,2) ≅ ℍ', 'Cl(0,1) ≅ ℂ', …
        signature      : (p, q) — Clifford signature
        dim_real       : real dimension
        dim_rep        : minimal faithful complex representation dim
        automorphism   : 'Sp(1)×Sp(1)' (ℍ), 'U(1)' (ℂ), 'G_2' (𝕆), …
        n_attest       : worldline length at which the edge algebra's
                         structure constants become attested
        notes          : provenance / cross-reference notes
    """
    name: str
    signature: tuple
    dim_real: int
    dim_rep: int
    automorphism: str
    n_attest: int = 1
    notes: str = ''

    @property
    def description_bits(self) -> float:
        """L(E) — small per-layer L(M) contribution, ~ log₂(dim_real)."""
        return max(1.0, math.log2(max(2, self.dim_real)))


# Cl(0,p) real dimensions: 2^p. Minimal complex rep dims and N_attest data
# from sector_cooling_cascade_audit.py (edge_data).
_CL0P = {
    0: ('Cl(0,0) ≅ ℝ edge', 1, 1, '{1}', 4),
    1: ('Cl(0,1) ≅ ℂ edge', 2, 1, 'U(1)', 4),
    2: ('Cl(0,2) ≅ ℍ edge', 4, 2, 'Sp(1)×Sp(1) ⊃ SU(2)_L×SU(2)_R', 4),
    3: ('Cl(0,3) ≅ ℍ⊕ℍ edge', 8, 2, 'Sp(1)×Sp(1)', 9),
    4: ('Cl(0,4) ≅ M_2(ℍ) edge', 16, 4, 'Sp(2)', 16),
}


def enumerate_clifford(p_max: int = 4) -> list[EdgeAlgebra]:
    """Cl(0, p) for p = 0..p_max. Real dim 2^p."""
    out = []
    for p in range(0, p_max + 1):
        name, dim_real, dim_rep, auto, n_att = _CL0P.get(
            p, (f'Cl(0,{p}) edge', 2 ** p, 2 ** ((p + 1) // 2), 'Pin(p)', (2 ** p) ** 2))
        out.append(EdgeAlgebra(
            name=name, signature=(0, p), dim_real=dim_real, dim_rep=dim_rep,
            automorphism=auto, n_attest=n_att,
            notes=('framework dominant edge retention (G_2 theorem)' if p == 2 else '')))
    return out


def enumerate_full_menu(p_max: int = 4) -> list[EdgeAlgebra]:
    """All edge-algebra candidates plurally retained at framework scale.

    Cl(0,p) for p ≤ p_max, plus the octonionic edge candidate (𝕆) used in
    some Layer-1-escape subdominant slices.
    """
    out = enumerate_clifford(p_max)
    out.append(EdgeAlgebra(
        name='𝕆 (octonion) edge', signature=(0, 0), dim_real=8, dim_rep=8,
        automorphism='G_2', n_attest=512,
        notes='octonionic edge — Layer-1 escape candidate (audited NEGATIVE)'))
    return out


def framework_dominant() -> EdgeAlgebra:
    """Cl(0, 2) ≅ ℍ — framework's dominant edge retention (G2 theorem)."""
    name, dim_real, dim_rep, auto, n_att = _CL0P[2]
    return EdgeAlgebra(
        name=name, signature=(0, 2), dim_real=dim_real, dim_rep=dim_rep,
        automorphism=auto, n_attest=n_att,
        notes='framework dominant edge retention; theorem_g2_edge_qubit_su2.md')
