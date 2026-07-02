"""
Vertex local-algebra menu — candidate algebras at each substrate vertex.

For a substrate vertex with k incoming generators, the local algebra is the
algebraic structure on the Fock / spinor representation. Per Tasks A
(commit 2c2a624, 2026-05-07) + the cooling-cascade audit, the menu includes:

  Clifford family       Cl(2k, 0) — associative, dim 2^(2k), Fock dim 2^k.
                        Cl(6,0) is the framework's dominant retention (k=3).
  Cayley-Dickson tower  ℝ (d=0), ℂ (d=1), ℍ (d=2), 𝕆 (d=3, non-assoc),
                        sedenion (d=4), … ; real dim 2^d.
  Hurwitz normed        Subset of Cayley-Dickson with norm composition;
                        only ℝ, ℂ, ℍ, 𝕆 qualify (Hurwitz 1898).
  Magic-square Lie      Tits-Freudenthal R⊗𝕆 = F_4 (52), ℂ⊗𝕆 = E_6 (78),
                        ℍ⊗𝕆 = E_7 (133), 𝕆⊗𝕆 = E_8 (248), plus the rest of
                        the 4×4 magic square.

Enumerator scope:
- enumerate_clifford       — Cl(2k, 0) for k ≤ k_max
- enumerate_cayley_dickson — depth d ≤ d_max
- enumerate_hurwitz        — closed list {ℝ, ℂ, ℍ, 𝕆}
- enumerate_magic_square   — 16 entries of the magic square
- enumerate_full_menu      — union of the above (deduplicated)

NB: this module enumerates ONLY. MDL gating (which retention dominates at
each k, each N) lives in simulator.gating. The numeric N_attest /
description-length data mirror proofs/foundations/sector_cooling_cascade_audit.py
(vertex_data) and the Task A scoring.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VertexAlgebra:
    """One candidate vertex local algebra.

    Attributes:
        name           : 'Cl(6,0)', '𝕆 (octonion)', 'ℂ⊗𝕆 = E_6', …
        family         : 'Clifford' | 'CayleyDickson' | 'Hurwitz' | 'MagicSquare'
        dim_real       : real dimension of the algebra
        dim_fock       : Fock / spinor representation dim (2^k for Cl(2k,0))
        k_compat       : tuple of vertex-coordination values this algebra is
                         compatible with (Cl(6,0) requires k=3)
        associative    : True for Cl, ℝ, ℂ, ℍ; False for 𝕆 and higher CD
        normed         : True iff Hurwitz; False otherwise
        automorphism   : 'Spin(2k)' | 'G_2' | 'F_4' | 'E_6' | 'E_7' | 'E_8' | …
        n_attest       : worldline length at which the algebra's structure
                         constants become attested (cooling-cascade audit)
        notes          : provenance / cross-reference notes
    """
    name: str
    family: str
    dim_real: int
    dim_fock: int
    k_compat: tuple
    associative: bool
    normed: bool
    automorphism: str
    n_attest: int = 1
    notes: str = ''

    @property
    def description_bits(self) -> float:
        """L(V) — a small per-layer L(M) contribution the gating layer sums in.

        The local algebra's structure-constant table grows with the algebra
        dimension; we charge ~ log₂(dim_real) bits (the same Elias-style cost
        order the Coxeter layer uses for its relation matrix).
        """
        return max(1.0, math.log2(max(2, self.dim_real)))


# Cooling-cascade-audit vertex N_attest data (sector_cooling_cascade_audit.py).
_CLIFFORD_N_ATTEST = {1: 4, 2: 16, 3: 36, 4: 64, 5: 100, 6: 144, 7: 196, 8: 256}
_CD_N_ATTEST = {0: 4, 1: 4, 2: 16, 3: 512, 4: 1048576, 5: 1048576 * 4}
_MAGIC_DIM = {
    'R⊗R': 3, 'R⊗C': 3, 'R⊗H': 8, 'R⊗O = F_4 (52)': 52,
    'C⊗R': 3, 'C⊗C': 8, 'C⊗H': 11, 'C⊗O = E_6 (78)': 78,
    'H⊗R': 8, 'H⊗C': 11, 'H⊗H': 16, 'H⊗O = E_7 (133)': 133,
    'O⊗R = F_4 (52)': 52, 'O⊗C = E_6 (78)': 78, 'O⊗H = E_7 (133)': 133,
    'O⊗O = E_8 (248)': 248,
}
_MAGIC_N_ATTEST = {
    'R⊗R': 4, 'R⊗C': 4, 'R⊗H': 16, 'R⊗O = F_4 (52)': 512,
    'C⊗R': 4, 'C⊗C': 8, 'C⊗H': 32, 'C⊗O = E_6 (78)': 4096,
    'H⊗R': 16, 'H⊗C': 32, 'H⊗H': 64, 'H⊗O = E_7 (133)': 32768,
    'O⊗R = F_4 (52)': 512, 'O⊗C = E_6 (78)': 4096, 'O⊗H = E_7 (133)': 32768,
    'O⊗O = E_8 (248)': 262144,
}


def enumerate_clifford(k_max: int = 8) -> list[VertexAlgebra]:
    """Cl(2k, 0) for k = 1..k_max. dim 2^(2k), Fock dim 2^k.

    The framework's dominant retention is k=3 ⇒ Cl(6, 0) at the trivalent
    srs vertex (theorem-grade; Cl(6,0) Fock rep dim 8, Aut = Spin(6) = SU(4)).
    """
    out = []
    for k in range(1, k_max + 1):
        out.append(VertexAlgebra(
            name=f'Cl({2*k},0)', family='Clifford',
            dim_real=2 ** (2 * k), dim_fock=2 ** k, k_compat=(k,),
            associative=True, normed=(2 * k <= 2), automorphism=f'Spin({2*k})',
            n_attest=_CLIFFORD_N_ATTEST.get(k, (2 * k) ** 2),
            notes=('framework dominant retention at k=3' if k == 3 else '')))
    return out


def enumerate_cayley_dickson(d_max: int = 5) -> list[VertexAlgebra]:
    """Cayley-Dickson algebras at depth d ∈ {0, …, d_max}. real dim = 2^d.

    d=0 ℝ, d=1 ℂ, d=2 ℍ, d=3 𝕆 (non-assoc, alternative, normed),
    d=4 sedenion (non-alt, non-normed, zero divisors), d=5 trigintaduonion.
    """
    names = {0: 'ℝ', 1: 'ℂ', 2: 'ℍ (quaternion)', 3: '𝕆 (octonion)',
             4: 'sedenion', 5: 'trigintaduonion'}
    auto = {0: '{1}', 1: 'Z/2', 2: 'SO(3)', 3: 'G_2', 4: 'G_2 × ...',
            5: 'G_2 × ...'}
    out = []
    for d in range(0, d_max + 1):
        out.append(VertexAlgebra(
            name=names.get(d, f'CD(d={d})'), family='CayleyDickson',
            dim_real=2 ** d, dim_fock=2 ** d, k_compat=tuple(),
            associative=(d <= 2), normed=(d <= 3),
            automorphism=auto.get(d, 'G_2 × ...'),
            n_attest=_CD_N_ATTEST.get(d, (2 ** d) ** 2),
            notes=('Hurwitz normed division algebra' if d <= 3 else
                   'non-alternative; zero divisors')))
    return out


def enumerate_hurwitz() -> list[VertexAlgebra]:
    """{ℝ, ℂ, ℍ, 𝕆} — the four normed division algebras (Hurwitz 1898)."""
    return enumerate_cayley_dickson(3)


def enumerate_magic_square() -> list[VertexAlgebra]:
    """16 entries of the Tits-Freudenthal magic square: R/C/H/O ⊗ R/C/H/O.

    Diagonal-of-interest (·⊗𝕆): F_4 (52), E_6 (78), E_7 (133), E_8 (248) from
    R/C/H/O ⊗ 𝕆 — candidate vertex algebras for Layer-1-escape subdominant zoo
    slices (memory 2026-05-06+1; all audited NEGATIVE via M1-M7).
    """
    out = []
    seen = set()
    for key, dim in _MAGIC_DIM.items():
        if key in seen:
            continue
        seen.add(key)
        is_lie = '=' in key  # exceptional Lie entries carry '= F_4/E_6/...'
        out.append(VertexAlgebra(
            name=(key if is_lie else f'{key} (dim {dim})'),
            family='MagicSquare', dim_real=dim, dim_fock=dim, k_compat=tuple(),
            associative=False, normed=False,
            automorphism=(key.split('=')[-1].strip() if is_lie else 'Lie'),
            n_attest=_MAGIC_N_ATTEST.get(key, dim ** 2),
            notes='Tits-Freudenthal magic square; Layer-1 escape candidate'))
    return out


def enumerate_full_menu(k_max: int = 8, d_max: int = 5) -> list[VertexAlgebra]:
    """Union of Clifford + Cayley-Dickson + magic-square menus (deduplicated)."""
    menu = (enumerate_clifford(k_max) + enumerate_cayley_dickson(d_max)
            + enumerate_magic_square())
    seen, out = set(), []
    for va in menu:
        if va.name in seen:
            continue
        seen.add(va.name)
        out.append(va)
    return out


def framework_dominant() -> VertexAlgebra:
    """The framework's dominant vertex algebra: Cl(6, 0) at k=3 (trivalent srs)."""
    return VertexAlgebra(
        name='Cl(6,0)', family='Clifford', dim_real=64, dim_fock=8,
        k_compat=(3,), associative=True, normed=False,
        automorphism='Spin(6) = SU(4)', n_attest=_CLIFFORD_N_ATTEST[3],
        notes='framework dominant retention; Aut = SU(4) ⊃ Pati-Salam color')
