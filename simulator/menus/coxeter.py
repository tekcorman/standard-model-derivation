"""
Coxeter quotient menu — substrate-side candidate enumeration.

A Coxeter system M = (S, m) is specified by |S| generators and a symmetric
matrix m: S × S → ℤ_≥2 ∪ {∞} of relation orders, with the relations
    (s_i s_j)^{m_ij} = 1.

The substrate menu is the set of Coxeter quotients (Z/2)^{*|E|} / {relations}
that the framework's A2-T waterline keeps plurally retained at framework scale.
Per memory 2026-05-06+2 + 2026-05-07 (Tasks A-E), this menu spans:
- Finite Coxeter for |E|=2..8 (including E_8, F_4, H_3, H_4, etc.)
- Affine Coxeter (Ã_2, C̃_2, G̃_2, …) — 2D/3D crystal tilings
- Bounded-m multi-generator (Path B "hyperbolic") enumeration
- Free baseline (no relations) as compression reference

Enumerator scope, in order of growth-class:
- enumerate_finite       — closed finite Coxeter groups
- enumerate_affine       — affine extensions
- enumerate_hyperbolic   — multi-generator Path B (bounded m_max)
- enumerate_free         — free baselines
- enumerate_full_menu    — union of the above

NB: this module enumerates ONLY. Compression value Φ(M, N), description
length L(M), frequency factor, and combined weight live in
simulator.gating.mdl. The numeric data mirror
  proofs/foundations/sector_coxeter_freq_weighted_audit.py
  proofs/foundations/sector_coxeter_full_menu_ranking_audit.py
  proofs/foundations/sector_path_B_multi_gen_audit.py
"""

from dataclasses import dataclass, field
from typing import Optional


# Framework-scale frequency cutoff: log₂(N_hub) ≈ log₂(10^60) ≈ 199.3 → 200
# (commit 30b4bd7 / sector_coxeter_freq_weighted_audit.py lines 225-246).
N_HUB_LOG2 = 200.0


@dataclass(frozen=True)
class CoxeterSystem:
    """One Coxeter system M = (S, m) — a substrate-menu candidate.

    Attributes:
        name           : human-readable identifier ('A_3 = S_4', 'srs ~ H_3-like', …)
        generators     : |S| = number of involutive generators (= |E|)
        m_pairs        : dict (i, j) → m_ij ∈ ℤ_≥2 ∪ {∞} ; absent ⇒ m=2 (commute).
                         For 'hyperbolic'/multi-gen entries m_pairs is empty and
                         the single multi-generator relator is (multi_gen_K,
                         multi_gen_m): (T_1…T_K)^m = id.
        order          : |W(M)| if finite, None for affine/hyperbolic/free
        growth_class   : 'finite' | 'affine' | 'hyperbolic' | 'free'
        rank           : Coxeter rank (= generators for irreducible finite;
                         = finite Weyl rank for affine; = K for multi-gen)
        finite_order   : for affine systems, |W_finite| of the underlying finite
                         Weyl group (polynomial-growth coefficient); None otherwise
        multi_gen_K    : for 'hyperbolic'/multi-gen entries, the relator arity K
        multi_gen_m    : for 'hyperbolic'/multi-gen entries, the relator exponent m
        notes          : provenance / cross-reference notes
    """
    name: str
    generators: int
    m_pairs: dict
    order: Optional[int]
    growth_class: str
    rank: int = field(default=0)
    finite_order: Optional[int] = None
    multi_gen_K: Optional[int] = None
    multi_gen_m: Optional[int] = None
    notes: str = ''

    # ---- derived quantities the gating layer needs ----------------------

    @property
    def E(self) -> int:
        """Alias for `generators` (|E| in the substrate-stream language)."""
        return self.generators

    @property
    def max_relation_length(self) -> int:
        """Length of the longest defining relator.

        For a pairwise relation (T_iT_j)^m it is 2·max(m_ij) (default m=2 ⇒ 4).
        For a multi-generator relator (T_1…T_K)^m it is K·m. Free baseline = 0.
        """
        if self.growth_class == 'free':
            return 0
        if self.multi_gen_K is not None and self.multi_gen_m is not None:
            return self.multi_gen_K * self.multi_gen_m
        max_m = 2
        for m in self.m_pairs.values():
            if m == float('inf'):
                continue
            if m > max_m:
                max_m = m
        return 2 * max_m


# ---------------------------------------------------------------------------
# Diagram helpers (linear / branched Dynkin-style m-pairs)
# ---------------------------------------------------------------------------

def _path_pairs(E: int, m: int = 3) -> dict:
    """Linear A_n-style diagram on E nodes, all braids m."""
    return {(i, i + 1): m for i in range(1, E)}


def _Bn_pairs(E: int, m: int = 4) -> dict:
    """B_n / C_n: m=4 on the first bond, m=3 elsewhere."""
    p = {(i, i + 1): 3 for i in range(2, E)}
    p[(1, 2)] = m
    return p


def _Dn_pairs(E: int) -> dict:
    """D_n: linear chain of length E-1 with a fork at the (E-2)th node."""
    p = {(i, i + 1): 3 for i in range(1, E - 1)}
    p[(E - 2, E)] = 3
    return p


def _En_pairs(E: int) -> dict:
    """E_n exceptional diagram: a length-(E-1) chain with a leg off node 3."""
    p = {(i, i + 1): 3 for i in range(1, E - 1)}
    p[(3, E)] = 3
    return p


# ---------------------------------------------------------------------------
# Finite Coxeter / Weyl groups, |E| = 2..8
# (mirrors sector_coxeter_full_menu_ranking_audit.py + freq_weighted_audit.py)
# ---------------------------------------------------------------------------

def enumerate_finite(E_max: int = 8) -> list[CoxeterSystem]:
    """Enumerate the closed finite Coxeter groups with |E| ≤ E_max.

    Classical families (A_n, B_n/C_n, D_n) plus exceptional finite Coxeter
    (E_6, E_7, E_8, F_4, H_3, H_4, I_2(p) for finite p). Orders are exact.
    """
    out: list[CoxeterSystem] = []

    def add(name, E, m_pairs, order, rank=None, notes=''):
        if E <= E_max:
            out.append(CoxeterSystem(
                name=name, generators=E, m_pairs=dict(m_pairs), order=order,
                growth_class='finite', rank=(rank if rank is not None else E),
                notes=notes))

    # |E|=2 : I_2(p) dihedrals
    for p in [2, 3, 4, 5, 6, 8, 12, 16, 24]:
        names = {2: 'V_4 = (Z/2)²', 3: 'S_3 = D_3', 4: 'D_4', 5: 'H_2 = D_5',
                 6: 'G_2 = D_6'}
        nm = names.get(p, f'I_2({p}) = D_{p}')
        add(f'{nm}', 2, {(1, 2): p}, 2 * p, rank=2, notes='dihedral')

    # |E|=3
    add('(Z/2)³', 3, {(1, 2): 2, (1, 3): 2, (2, 3): 2}, 8, notes='abelian')
    add('A_3 = S_4 (tetrahedral)', 3, {(1, 2): 3, (2, 3): 3}, 24)
    add('B_3 (octahedral)', 3, {(1, 2): 4, (2, 3): 3}, 48)
    add('H_3 (icosahedral)', 3, {(1, 2): 5, (2, 3): 3}, 120,
        notes='srs sits in this icosahedral region (memory 2026-05-07)')

    # |E|=4
    add('(Z/2)⁴', 4, {}, 16, notes='abelian')
    add('A_4 = S_5', 4, _path_pairs(4, 3), 120)
    add('B_4', 4, _Bn_pairs(4), 384)
    add('D_4', 4, _Dn_pairs(4), 192)
    add('F_4 (rank-4 exceptional)', 4, {(1, 2): 3, (2, 3): 4, (3, 4): 3}, 1152)
    add('H_4 (icosahedral×)', 4, {(1, 2): 5, (2, 3): 3, (3, 4): 3}, 14400)

    # |E|=5
    add('A_5 = S_6', 5, _path_pairs(5, 3), 720)
    add('B_5', 5, _Bn_pairs(5), 3840)
    add('D_5', 5, _Dn_pairs(5), 1920)

    # |E|=6
    add('A_6 = S_7', 6, _path_pairs(6, 3), 5040)
    add('B_6', 6, _Bn_pairs(6), 46080)
    add('D_6', 6, _Dn_pairs(6), 23040)
    add('E_6 (exceptional)', 6, _En_pairs(6), 51840)

    # |E|=7
    add('A_7 = S_8', 7, _path_pairs(7, 3), 40320)
    add('D_7', 7, _Dn_pairs(7), 322560)
    add('E_7 (exceptional)', 7, _En_pairs(7), 2903040)

    # |E|=8
    add('A_8 = S_9', 8, _path_pairs(8, 3), 362880)
    add('B_8', 8, _Bn_pairs(8), 10321920)
    add('D_8', 8, _Dn_pairs(8), 5160960)
    add('E_8 (THE exceptional)', 8, _En_pairs(8), 696729600)

    return out


# ---------------------------------------------------------------------------
# Affine Coxeter, |E| = rank + 1, polynomial growth |W(M,N)| ~ |W_fin|·N^r
# ---------------------------------------------------------------------------

def enumerate_affine(E_max: int = 8) -> list[CoxeterSystem]:
    """Enumerate affine Coxeter systems with |E| ≤ E_max.

    Ã_n, B̃_n, C̃_n, D̃_n, plus exceptional Ẽ_6, Ẽ_7, F̃_4, G̃_2. Affine
    groups have polynomial growth |W(N)| ~ |W_finite|·N^r where r = finite
    Weyl rank. (Tables: Humphreys 1990.)
    """
    out: list[CoxeterSystem] = []

    def add(name, E, m_pairs, finite_order, finite_rank, notes=''):
        if E <= E_max:
            out.append(CoxeterSystem(
                name=name, generators=E, m_pairs=dict(m_pairs), order=None,
                growth_class='affine', rank=finite_rank,
                finite_order=finite_order, notes=notes))

    # |E|=3 affine : rank-2 finite Weyl, 2D crystal tilings
    add('Ã_2 (triangular tiling)', 3, {(1, 2): 3, (2, 3): 3, (1, 3): 3}, 6, 2,
        notes='2D triangular crystal net')
    add('C̃_2 (square tiling)', 3, {(1, 2): 4, (2, 3): 4}, 8, 2,
        notes='2D square crystal net')
    add('G̃_2 (kagome)', 3, {(1, 2): 6, (2, 3): 3}, 12, 2,
        notes='2D kagome crystal net')
    # |E|=4 affine : rank-3 finite Weyl, 3D crystal tilings
    add('Ã_3 (3D triangular)', 4, {(1, 2): 3, (2, 3): 3, (3, 4): 3, (1, 4): 3}, 24, 3)
    add('B̃_3', 4, {(1, 2): 4, (2, 3): 3, (3, 4): 4}, 48, 3)
    add('C̃_3', 4, {(1, 2): 4, (2, 3): 3, (3, 4): 4}, 48, 3)
    # |E|=5 affine
    add('D̃_4', 5, {(1, 5): 3, (2, 5): 3, (3, 5): 3, (4, 5): 3}, 192, 4,
        notes='4-valent star (D_4 affine)')
    add('Ã_4', 5, {(1, 2): 3, (2, 3): 3, (3, 4): 3, (4, 5): 3, (1, 5): 3}, 120, 4)
    add('F̃_4', 5, {(1, 2): 3, (2, 3): 4, (3, 4): 3, (4, 5): 3}, 1152, 4)
    # |E|=7 affine
    add('Ẽ_6', 7, {(1, 2): 3, (2, 3): 3, (3, 4): 3, (4, 5): 3, (3, 6): 3, (6, 7): 3},
        51840, 6)
    # |E|=8 affine
    add('Ẽ_7', 8, {(1, 2): 3, (2, 3): 3, (3, 4): 3, (4, 5): 3, (5, 6): 3,
                   (3, 7): 3, (7, 8): 3}, 2903040, 7)

    return out


# ---------------------------------------------------------------------------
# Multi-generator (Path B) "hyperbolic" cells: (T_1…T_K)^m = id atop F_inv(|E|)
# (mirrors sector_path_B_multi_gen_audit.py — no closed-form |W|, freq axis only)
# ---------------------------------------------------------------------------

_PATH_B_E = [2, 3, 4, 5, 6, 7, 8]
_PATH_B_K = [3, 4, 5, 6, 7, 8]
_PATH_B_M = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32]


def enumerate_hyperbolic(E_max: int = 8, m_max: int = 32) -> list[CoxeterSystem]:
    """Enumerate bounded-m multi-generator (Path B) systems.

    Single-relator family R(K, m): (T_1…T_K)^m = id with 3 ≤ K ≤ |E|, m ≤ m_max.
    These generically give infinite, non-classifiable quotients (order=None);
    the gating layer scores them on the frequency axis only (Path B is a
    strictly weaker retention criterion than the Coxeter Φ−L+freq audit).
    Mirrors sector_path_B_multi_gen_audit.py: |E|≤8, K∈3..8, m∈{1..32}.
    """
    out: list[CoxeterSystem] = []
    for E in _PATH_B_E:
        if E > E_max:
            continue
        for K in _PATH_B_K:
            if K > E:
                continue
            for m in _PATH_B_M:
                if m > m_max:
                    continue
                out.append(CoxeterSystem(
                    name=f'PathB R(K={K}, m={m}) on F_inv({E})',
                    generators=E, m_pairs={}, order=None,
                    growth_class='hyperbolic', rank=K,
                    multi_gen_K=K, multi_gen_m=m,
                    notes='multi-generator single relator; freq-axis scoring only'))
    return out


def enumerate_free(E_max: int = 8) -> list[CoxeterSystem]:
    """Free baselines F_inv(|E|): m=∞ on every pair, no relations.

    Compression reference — Φ(M_free, N) = 0 by definition.
    """
    return [
        CoxeterSystem(name=f'F_inv({E}) free baseline', generators=E,
                      m_pairs={}, order=None, growth_class='free', rank=E,
                      notes='compression reference: Φ ≡ 0')
        for E in range(2, E_max + 1)
    ]


def enumerate_full_menu(E_max: int = 8, m_max: int = 32) -> list[CoxeterSystem]:
    """Union of finite + affine + bounded-m multi-gen + free baselines.

    Returns the substrate Coxeter-quotient menu the A2-T waterline operates
    over at observation length N. The N-dependence of retention is delegated
    to simulator.gating.{mdl, cooling}.
    """
    return (enumerate_finite(E_max)
            + enumerate_affine(E_max)
            + enumerate_hyperbolic(E_max, m_max)
            + enumerate_free(E_max))


def srs_equivalent() -> CoxeterSystem:
    """The Coxeter system corresponding to srs in the menu.

    srs (the (10,3)-a / K4 crystal net, |E| = k* = 3) is edge-transitive on
    a 3-regular 3-connected net; in the Coxeter-menu language its compression
    margin and N_attest profile coincide with the H_3 (icosahedral) |E|=3
    entry (cooling-cascade audit, substrate row 'srs (|E|=3 + ad-trans)').
    This helper returns that distinguished entry, tagged as the srs slice.
    """
    return CoxeterSystem(
        name='srs ~ H_3-like (|E|=3, edge-transitive 3-regular crystal net)',
        generators=3, m_pairs={(1, 2): 5, (2, 3): 3}, order=120,
        growth_class='finite', rank=3,
        notes=('srs / (10,3)-a / K4 crystal net; |E| = k* = 3; Sunada-unique '
               'edge-transitive 3-regular 3-connected net; N_attest profile '
               '= H_3 (cooling-cascade audit). Substrate-only MDL does NOT '
               'select this slice — see gating/cooling.dominant_slice docstring.'))
