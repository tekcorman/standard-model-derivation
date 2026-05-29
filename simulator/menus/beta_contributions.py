"""
β-contribution candidate menu — walk-based search for Δb = (+5/2, +25/6, +4).

Per scoping doc `walk_based_delta_b_search_scoping_2026-05-14.md`: enumerate
candidate mechanisms by which substrate-native walks/channels/sectors
contribute to the gauge-coupling β-coefficient deltas Δb_i needed to flow
from the framework's substrate-native boundary (α_GUT⁻¹ = 24, sin²θ_W = 3/8
at M_unif) down to PDG values at M_Z.

Each candidate is parameterised by:
  - particle-content vector: list of (rep_3, rep_2, hypercharge_Y, statistics,
    multiplicity) entries.  Each entry is a substrate-derivable BOSONIC or
    FERMIONIC bundle of states with specific transformation properties.
  - origin description: which substrate primitives (walk length, channel count,
    h-power, edge orbit, etc.) generated each multiplicity.

The candidate produces Δb_i for i ∈ {1, 2, 3} (U(1)_Y, SU(2)_L, SU(3)_c) via
the standard one-loop β formula:

  Δb_i = Σ_{p ∈ particles}  (stat_factor_p) × T_i(rep_p) × mult_p

where:
  - stat_factor_p = 2/3 for a Weyl fermion, 1/3 for a real scalar,
                    1/6 for a complex-scalar pair (treating as 2 reals)
                    [In MSSM conventions: per real-DOF; complex doublet = 4 real]
  - T_i(rep) = Dynkin index of rep under gauge factor i
  - mult_p = number of copies

The Δb tuple is then a triple of rationals to be matched against the MSSM
target (+5/2, +25/6, +4).

NB: enumerator ONLY.  Match testing + structural-criterion gating in
`simulator.gating.delta_b_match`.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional


# ---------------------------------------------------------------------------
# Particle bundle dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParticleBundle:
    """One bosonic-or-fermionic bundle with specific gauge transformation.

    Attributes:
        label     : human-readable identifier ('squark-Q', 'gluino', ...)
        rep_3     : SU(3) Casimir dim (1 = singlet, 3 = fundamental, 8 = adjoint)
        rep_2     : SU(2) Casimir dim (1 = singlet, 2 = doublet, 3 = triplet/adjoint)
        Y         : hypercharge (Fraction)
        statistics: 'fermion' (Weyl) or 'scalar' (real DOF count)
        n_real    : number of real DOF carried per copy (e.g. complex doublet = 4)
        mult      : multiplicity (number of copies)
        origin    : free-form note explaining substrate origin
    """
    label: str
    rep_3: int
    rep_2: int
    Y: Fraction
    statistics: str
    n_real: int
    mult: int = 1
    origin: str = ''


# ---------------------------------------------------------------------------
# Gauge β-function contribution formulas
# ---------------------------------------------------------------------------

def dynkin_su2(rep_dim: int) -> Fraction:
    """Dynkin index T(R) of SU(2) rep with given dim.

    SU(2) irreps: 1 (singlet, T=0), 2 (fundamental, T=1/2), 3 (adjoint, T=2),
    4 (T=5), etc.  Formula: T(j) = j(j+1)(2j+1)/3 with rep dim 2j+1.
    """
    if rep_dim == 1: return Fraction(0)
    if rep_dim == 2: return Fraction(1, 2)
    if rep_dim == 3: return Fraction(2)
    if rep_dim == 4: return Fraction(5)
    # General formula
    j = Fraction(rep_dim - 1, 2)
    return j * (j + 1) * (2 * j + 1) / 3


def dynkin_su3(rep_dim: int) -> Fraction:
    """Dynkin index T(R) of SU(3) rep with given dim.

    Standard: T(1)=0, T(3)=T(3̄)=1/2, T(6)=T(6̄)=5/2, T(8)=3.
    """
    table = {1: Fraction(0), 3: Fraction(1, 2), 6: Fraction(5, 2),
             8: Fraction(3), 10: Fraction(15, 2), 15: Fraction(10)}
    return table.get(rep_dim, Fraction(rep_dim) / 2)  # rough default


def u1_contribution(Y: Fraction, n_real: int) -> Fraction:
    """U(1)_Y β contribution from a particle with hypercharge Y and n_real DOF.

    In SU(5)/GUT normalisation:  Δb_1 contribution = (3/5) × Y² × n_real_factor
    where n_real_factor = 1 per real scalar DOF, or 2 per Weyl fermion (which
    has 2 real Lorentz DOF on shell).  We're computing the bare Δb_1 (= Σ Y²
    × stat × mult × 3/5) in GUT normalisation.
    """
    return Fraction(3, 5) * Y * Y * n_real


def beta_factor(statistics: str) -> Fraction:
    """Pre-factor in one-loop β:
      complex scalar / 6 real DOF per   →  1/6 per real
      Weyl fermion / 2 real DOF per     →  1/3 per real (= 2/3 per Weyl)
    For Δb computation: contribution = (factor) × (sum over reps weighted by Dynkin/Y²) × n_real
    Using uniform "per real DOF" convention:
      scalar: factor = 1/6 per real
      Weyl:   factor = 1/3 per real (= 2/3 × Weyl rep counted as 2 real)
    """
    if statistics == 'scalar':
        return Fraction(1, 6)
    if statistics == 'fermion':
        return Fraction(1, 3)
    raise ValueError(f'unknown statistics: {statistics}')


def bundle_delta_b(bundle: ParticleBundle) -> tuple[Fraction, Fraction, Fraction]:
    """Δb_i contribution of one ParticleBundle to (b_1, b_2, b_3).

    Δb_3 = factor × T_3(rep_3) × (rep_2) × n_real × mult
    Δb_2 = factor × T_2(rep_2) × (rep_3) × n_real × mult
    Δb_1 = factor × (3/5) × Y² × (rep_3 × rep_2) × n_real × mult

    Convention: T_i(rep) × dim of OTHER factors gives the total Dynkin-weighted
    count for one bundle.
    """
    bf = beta_factor(bundle.statistics)
    db_3 = bf * dynkin_su3(bundle.rep_3) * bundle.rep_2 * bundle.n_real * bundle.mult
    db_2 = bf * dynkin_su2(bundle.rep_2) * bundle.rep_3 * bundle.n_real * bundle.mult
    db_1 = bf * Fraction(3, 5) * bundle.Y * bundle.Y * bundle.rep_3 * bundle.rep_2 * bundle.n_real * bundle.mult
    return (db_1, db_2, db_3)


# ---------------------------------------------------------------------------
# Candidate dataclass + Δb computation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BetaContributionCandidate:
    """One candidate β-contribution mechanism = a set of ParticleBundles
    plus an origin description tying the multiplicities to substrate primitives.
    """
    name: str
    bundles: tuple
    origin_description: str
    description_bits: float = 0.0  # MDL cost (fill via gating)

    def delta_b(self) -> tuple[Fraction, Fraction, Fraction]:
        d1, d2, d3 = Fraction(0), Fraction(0), Fraction(0)
        for b in self.bundles:
            x1, x2, x3 = bundle_delta_b(b)
            d1 += x1; d2 += x2; d3 += x3
        return (d1, d2, d3)


# ---------------------------------------------------------------------------
# Substrate primitives (constants used by the enumeration)
# ---------------------------------------------------------------------------

K_STAR = 3                      # srs degree
G_GIRTH = 10                    # srs girth
N_ATOMS = 4                     # srs primitive-cell vertex count
N_EDGES = 6                     # srs primitive-cell edge count
N_CHANNELS = 2                  # Cl(0,2) min faithful complex rep dim
N_GEN = 3                       # framework's 3 generations (C_3 outer)
N_COLOR = 3                     # SU(3)_c fundamental dim (= k*)
ALPHA_GUT_INV = 24              # framework α_GUT⁻¹
ALPHA1_BARE_NUM = 256           # (k*-1)^(g-2) = 2^8
ALPHA1_BARE_DEN = 6561          # (k*)^(g-2) = 3^8
TAN_SQ_ARG_H_NUM = 5            # tan²(arg h) = 5/3
TAN_SQ_ARG_H_DEN = 3
COS_SQ_ARG_H_NUM = 3            # cos²(arg h) = 3/8 = sin²θ_W(M_unif)
COS_SQ_ARG_H_DEN = 8


# ---------------------------------------------------------------------------
# MSSM target (for reference, used by the gating layer)
# ---------------------------------------------------------------------------

MSSM_DELTA_B = (Fraction(5, 2), Fraction(25, 6), Fraction(4))


# ---------------------------------------------------------------------------
# Reference catalog: MSSM extras as ParticleBundles (sanity check)
# ---------------------------------------------------------------------------

def mssm_extras_catalog() -> list[ParticleBundle]:
    """Standard MSSM extras (sfermions, gauginos, Higgsinos, extra Higgs).

    Used to verify our Δb formula reproduces MSSM Δb = (5/2, 25/6, 4).
    """
    out: list[ParticleBundle] = []
    # 3 generations of sfermions (complex scalars = 2 real DOF per scalar)
    for gen in range(3):
        out.append(ParticleBundle(
            label=f'Q̃_L (gen {gen+1})', rep_3=3, rep_2=2, Y=Fraction(1, 6),
            statistics='scalar', n_real=2, mult=1,
            origin='MSSM sfermion: SU(2)_L quark doublet'))
        out.append(ParticleBundle(
            label=f'ũ_R^c (gen {gen+1})', rep_3=3, rep_2=1, Y=Fraction(-2, 3),
            statistics='scalar', n_real=2, mult=1,
            origin='MSSM sfermion: up-quark singlet (conjugate)'))
        out.append(ParticleBundle(
            label=f'd̃_R^c (gen {gen+1})', rep_3=3, rep_2=1, Y=Fraction(1, 3),
            statistics='scalar', n_real=2, mult=1,
            origin='MSSM sfermion: down-quark singlet (conjugate)'))
        out.append(ParticleBundle(
            label=f'L̃_L (gen {gen+1})', rep_3=1, rep_2=2, Y=Fraction(-1, 2),
            statistics='scalar', n_real=2, mult=1,
            origin='MSSM sfermion: lepton doublet'))
        out.append(ParticleBundle(
            label=f'ẽ_R^c (gen {gen+1})', rep_3=1, rep_2=1, Y=Fraction(1),
            statistics='scalar', n_real=2, mult=1,
            origin='MSSM sfermion: electron singlet (conjugate)'))
    # 1 extra Higgs doublet (MSSM has Hu + Hd; SM has only H — extra is 1 doublet)
    out.append(ParticleBundle(
        label='extra Higgs doublet', rep_3=1, rep_2=2, Y=Fraction(1, 2),
        statistics='scalar', n_real=2, mult=1,
        origin='MSSM: 2nd Higgs doublet (Hd) vs SM single Higgs'))
    # Higgsinos (2 Weyl fermions, both doublets, Y = ±1/2)
    out.append(ParticleBundle(
        label='H̃u', rep_3=1, rep_2=2, Y=Fraction(1, 2),
        statistics='fermion', n_real=2, mult=1,
        origin='MSSM Higgsino (Weyl)'))
    out.append(ParticleBundle(
        label='H̃d', rep_3=1, rep_2=2, Y=Fraction(-1, 2),
        statistics='fermion', n_real=2, mult=1,
        origin='MSSM Higgsino (Weyl)'))
    # Gauginos: bino (singlet), wino (SU(2) adjoint), gluino (SU(3) adjoint)
    out.append(ParticleBundle(
        label='B̃ (bino)', rep_3=1, rep_2=1, Y=Fraction(0),
        statistics='fermion', n_real=2, mult=1,
        origin='MSSM gaugino: U(1)_Y'))
    out.append(ParticleBundle(
        label='W̃ (wino)', rep_3=1, rep_2=3, Y=Fraction(0),
        statistics='fermion', n_real=2, mult=1,
        origin='MSSM gaugino: SU(2)_L adjoint'))
    out.append(ParticleBundle(
        label='g̃ (gluino)', rep_3=8, rep_2=1, Y=Fraction(0),
        statistics='fermion', n_real=2, mult=1,
        origin='MSSM gaugino: SU(3)_c adjoint'))
    return out


def mssm_baseline_candidate() -> BetaContributionCandidate:
    """The MSSM matter content as a baseline candidate.  Used to verify
    bundle_delta_b reproduces Δb = (5/2, 25/6, 4) under our conventions.
    """
    return BetaContributionCandidate(
        name='MSSM_baseline',
        bundles=tuple(mssm_extras_catalog()),
        origin_description='Standard MSSM (sfermions, gauginos, Higgsinos, '
                          '2 Higgs doublets) as ground truth.',
    )


# ---------------------------------------------------------------------------
# Substrate-derived multiplicity vectors
# ---------------------------------------------------------------------------
#
# The framework's substrate provides specific INTEGERS that could play the
# multiplicity role in a bosonic-extra construction:
#
#   N_CHANNELS = 2 (Cl(0,2) min faithful rep dim)  →  could give "2 Higgs doublets"
#   N_GEN = 3 (C_3 outer)                          →  could give "3 generations"
#   N_ATOMS = 4 (srs cell)                         →  could give "4 something"
#   N_COLOR = 3 (k*)                               →  could give "3 colors"
#   N_EDGES = 6 (srs cell)                         →  could give "6 something"
#
# Enumeration: build candidates by combining substrate-native multiplicities
# with substrate-native rep-content (per-vertex Cl(6) Fock = 4+4̄ of SU(4)_PS).

def enumerate_substrate_candidates() -> list[BetaContributionCandidate]:
    """Enumerate β-contribution candidates motivated by substrate primitives.

    Each candidate names ONE substrate-derived multiplicity per particle bundle.
    The structural P1 criterion (three-from-one) requires the gauge-factor
    split to come from a single structural origin (e.g., per-vertex Cl(6) Fock
    decomposition naturally splits into SU(2)_L × SU(2)_R × U(1)_Y).
    """
    out: list[BetaContributionCandidate] = []

    # Candidate 1: MSSM baseline (for ground-truth Δb check)
    out.append(mssm_baseline_candidate())

    # Candidate 2: "Only extra Higgs doublet" (no sfermions/gauginos)
    out.append(BetaContributionCandidate(
        name='extra_Higgs_doublet_only',
        bundles=(
            ParticleBundle(
                label='extra Higgs doublet', rep_3=1, rep_2=2, Y=Fraction(1, 2),
                statistics='scalar', n_real=2, mult=1,
                origin='from n_channels = 2 (Cl(0,2)) → 2 Higgs doublets total'),
        ),
        origin_description='Single extra Higgs doublet from n_channels-1.',
    ))

    # Candidate 3: 2HDM (= 2 Higgs doublets in SM) — but this is just SM, not enough for MSSM Δb
    # (Already included above as "extra Higgs doublet only" since SM has 1 H, 2HDM has 2 → 1 extra)

    # Candidate 4: sfermions only (3 gens, no gauginos/higgsinos)
    sf_only_bundles = [b for b in mssm_extras_catalog()
                       if 'sfermion' in b.origin]
    out.append(BetaContributionCandidate(
        name='sfermions_only_3gen',
        bundles=tuple(sf_only_bundles),
        origin_description='3 gens of sfermions from N_GEN=3 (C_3 outer); '
                          'no gauginos / higgsinos.',
    ))

    # Candidate 5: gauginos only
    g_only_bundles = [b for b in mssm_extras_catalog()
                       if 'gaugino' in b.origin]
    out.append(BetaContributionCandidate(
        name='gauginos_only',
        bundles=tuple(g_only_bundles),
        origin_description='Only gauginos (B̃, W̃, g̃) from edge-sector '
                          '12 = 8+3+1 structure.',
    ))

    # Candidate 6: gauginos + sfermions, no Higgsinos
    no_higgs_bundles = [b for b in mssm_extras_catalog()
                        if 'Higgsino' not in b.origin
                        and 'Higgs doublet' not in b.origin]
    out.append(BetaContributionCandidate(
        name='sfermions_plus_gauginos',
        bundles=tuple(no_higgs_bundles),
        origin_description='Sfermions (3 gens) + gauginos; no extra Higgs.',
    ))

    # Candidate 7: gauginos + Higgsinos, no sfermions
    light_bundles = [b for b in mssm_extras_catalog()
                     if 'gaugino' in b.origin or 'Higgsino' in b.origin
                     or 'Higgs doublet' in b.origin]
    out.append(BetaContributionCandidate(
        name='gauginos_plus_higgsinos_plus_extraH',
        bundles=tuple(light_bundles),
        origin_description='Gauginos + Higgsinos + extra Higgs doublet; '
                          'no sfermions (sfermions might decouple at heavier scale).',
    ))

    # Candidate 8: 1 generation of sfermions only (per cell? Per srs unit cell has 1 set of fermions
    # per vertex via B3; if cell has N_ATOMS = 4 vertices, that's 4 "generations" per cell)
    one_gen_sfermions = []
    for b in mssm_extras_catalog():
        if 'sfermion' in b.origin and '1' in b.label:
            one_gen_sfermions.append(b)
    out.append(BetaContributionCandidate(
        name='one_gen_sfermions',
        bundles=tuple(one_gen_sfermions),
        origin_description='1 gen of sfermions (debug case: would not match Δb).',
    ))

    # Candidate 9: 4 generations of sfermions (motivated by N_ATOMS = 4)
    n_atoms_sfermions = []
    rep_only = ['Q̃_L', 'ũ_R^c', 'd̃_R^c', 'L̃_L', 'ẽ_R^c']
    for sf_label in rep_only:
        for gen in range(N_ATOMS):
            for b in mssm_extras_catalog():
                if 'sfermion' in b.origin and sf_label in b.label and '(gen 1)' in b.label:
                    out_bundle = ParticleBundle(
                        label=f'{sf_label} (cell-pos {gen+1})',
                        rep_3=b.rep_3, rep_2=b.rep_2, Y=b.Y,
                        statistics=b.statistics, n_real=b.n_real, mult=b.mult,
                        origin=f'sfermion at vertex {gen+1} of srs cell (N_ATOMS=4)')
                    n_atoms_sfermions.append(out_bundle)
    out.append(BetaContributionCandidate(
        name='N_ATOMS_gens_sfermions',
        bundles=tuple(n_atoms_sfermions),
        origin_description='Sfermions × N_ATOMS=4 (1 per srs cell vertex, '
                          'NOT C_3-outer-generation count).',
    ))

    return out


# ---------------------------------------------------------------------------
# Systematic enumeration over (n_gen × sfermion_subset × gauginos × Higgsinos
# × extra Higgs doublets), substrate-bounded
# ---------------------------------------------------------------------------

_SFERMION_TEMPLATES = [
    ('Q̃_L',    3, 2, Fraction(1, 6)),
    ('ũ_R^c',  3, 1, Fraction(-2, 3)),
    ('d̃_R^c',  3, 1, Fraction(1, 3)),
    ('L̃_L',    1, 2, Fraction(-1, 2)),
    ('ẽ_R^c',  1, 1, Fraction(1)),
    ('ν̃_R^c',  1, 1, Fraction(0)),  # right-handed sneutrino (gauge singlet)
]


def _make_sfermion(label: str, r3: int, r2: int, Y: Fraction, gen: int) -> ParticleBundle:
    return ParticleBundle(
        label=f'{label} (gen {gen+1})', rep_3=r3, rep_2=r2, Y=Y,
        statistics='scalar', n_real=2, mult=1,
        origin=f'sfermion gen {gen+1}')


def _make_gaugino(name: str, r3: int, r2: int, Y: Fraction) -> ParticleBundle:
    return ParticleBundle(
        label=name, rep_3=r3, rep_2=r2, Y=Y,
        statistics='fermion', n_real=2, mult=1,
        origin=f'gaugino for gauge factor')


def _make_higgsino(name: str, Y: Fraction) -> ParticleBundle:
    return ParticleBundle(
        label=name, rep_3=1, rep_2=2, Y=Y,
        statistics='fermion', n_real=2, mult=1, origin='Higgsino doublet')


def _make_extra_higgs(name: str, Y: Fraction) -> ParticleBundle:
    return ParticleBundle(
        label=name, rep_3=1, rep_2=2, Y=Y,
        statistics='scalar', n_real=2, mult=1, origin='extra Higgs doublet')


_GAUGINO_TEMPLATES = [
    ('B̃ (bino)',     1, 1, Fraction(0)),
    ('W̃ (wino)',     1, 3, Fraction(0)),
    ('g̃ (gluino)',   8, 1, Fraction(0)),
]


def enumerate_systematic(
        n_gen_options: list[int] = None,
        sfermion_subset_sizes: list[int] = None,
        gaugino_options: list[bool] = None,
        n_higgsino_pair_options: list[int] = None,
        n_extra_higgs_options: list[int] = None,
        include_ν_R: bool = False,
) -> list[BetaContributionCandidate]:
    """Cross-product enumeration over substrate-bounded multiplicities.

    Defaults:
      n_gen ∈ {1, 2, 3, 4}              (substrate: N_GEN=3, N_ATOMS=4)
      sfermion subset: all 5 (or 6 with ν_R)
      gauginos: each of {B̃, W̃, g̃} can be in or out independently
      Higgsino pairs: 0 or 1 pair = 2 Higgsinos
      extra Higgs doublets: 0 or 1
    """
    if n_gen_options is None:
        n_gen_options = [1, 2, 3, 4]
    if gaugino_options is None:
        gaugino_options = [False, True]
    if n_higgsino_pair_options is None:
        n_higgsino_pair_options = [0, 1]
    if n_extra_higgs_options is None:
        n_extra_higgs_options = [0, 1]

    sf_templates = _SFERMION_TEMPLATES[:5]  # exclude ν_R
    if include_ν_R:
        sf_templates = _SFERMION_TEMPLATES

    # Sfermion subset sizes: 0 (no sfermions) up to len(sf_templates) (all)
    if sfermion_subset_sizes is None:
        sfermion_subset_sizes = list(range(0, len(sf_templates) + 1))

    candidates: list[BetaContributionCandidate] = []
    for n_gen in n_gen_options:
        # Enumerate sfermion subsets by size — for each size, the canonical
        # subset is "first k templates".  (Could enumerate all C(5,k) subsets;
        # we restrict to size for now to keep menu compact.)
        sf_subset_choices = []
        if 0 in sfermion_subset_sizes:
            sf_subset_choices.append(())
        for k in sfermion_subset_sizes:
            if k == 0 or k > len(sf_templates):
                continue
            # All 5 sfermion types together (canonical MSSM)
            sf_subset_choices.append(tuple(sf_templates[:k]))
        # Always include the "all" subset (size 5)
        if (tuple(sf_templates),) not in [(s,) for s in sf_subset_choices]:
            sf_subset_choices.append(tuple(sf_templates))
        # Dedup
        sf_subset_choices = list(set(sf_subset_choices))

        for sf_subset in sf_subset_choices:
            for bino in gaugino_options:
                for wino in gaugino_options:
                    for gluino in gaugino_options:
                        for nh in n_higgsino_pair_options:
                            for nxh in n_extra_higgs_options:
                                bundles = []
                                for gen in range(n_gen):
                                    for tmpl in sf_subset:
                                        bundles.append(_make_sfermion(
                                            tmpl[0], tmpl[1], tmpl[2], tmpl[3], gen))
                                if bino:
                                    bundles.append(_make_gaugino(*_GAUGINO_TEMPLATES[0]))
                                if wino:
                                    bundles.append(_make_gaugino(*_GAUGINO_TEMPLATES[1]))
                                if gluino:
                                    bundles.append(_make_gaugino(*_GAUGINO_TEMPLATES[2]))
                                if nh >= 1:
                                    bundles.append(_make_higgsino('H̃u', Fraction(1, 2)))
                                    bundles.append(_make_higgsino('H̃d', Fraction(-1, 2)))
                                if nxh >= 1:
                                    bundles.append(_make_extra_higgs(
                                        'extra H doublet', Fraction(1, 2)))
                                name = (f'gen={n_gen}/sf={len(sf_subset)}/'
                                        f'B={int(bino)}/W={int(wino)}/'
                                        f'g={int(gluino)}/H̃={nh}/Hxs={nxh}')
                                if not bundles:
                                    continue
                                candidates.append(BetaContributionCandidate(
                                    name=name,
                                    bundles=tuple(bundles),
                                    origin_description='Systematic enumeration'))
    return candidates


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def enumerate_exotic_reps(
        max_color_rep: int = 8,
        max_weak_rep: int = 3,
        Y_options: list[Fraction] = None,
        max_mult: int = 4,
        max_total_extra_DOF: int = 50,
) -> list[BetaContributionCandidate]:
    """Enumerate single-bundle exotic-rep candidates to test whether ANY
    non-MSSM-like content can match Δb = (+5/2, +25/6, +4).

    Each candidate has ONE bundle with arbitrary (rep_3, rep_2, Y, statistics,
    mult).  This is a single-source structural test: can a SINGLE substrate
    structure reproduce the target?  (Almost certainly not, but tests P1.)
    """
    if Y_options is None:
        Y_options = [Fraction(0), Fraction(1, 6), Fraction(-1, 6),
                     Fraction(1, 3), Fraction(-1, 3),
                     Fraction(1, 2), Fraction(-1, 2),
                     Fraction(2, 3), Fraction(-2, 3),
                     Fraction(1), Fraction(-1)]
    color_reps = [1, 3, 6, 8] if max_color_rep >= 8 else [1, 3, 6][:max_color_rep // 3 + 1]
    weak_reps = list(range(1, max_weak_rep + 1))
    out: list[BetaContributionCandidate] = []
    for r3 in color_reps:
        for r2 in weak_reps:
            for Y in Y_options:
                for stat in ('scalar', 'fermion'):
                    for mult in range(1, max_mult + 1):
                        n_real = 2  # complex scalar or Weyl fermion → 2 real DOF
                        total_dof = r3 * r2 * n_real * mult
                        if total_dof > max_total_extra_DOF:
                            continue
                        b = ParticleBundle(
                            label=f'({r3},{r2},Y={Y})[{stat}]×{mult}',
                            rep_3=r3, rep_2=r2, Y=Y,
                            statistics=stat, n_real=n_real, mult=mult,
                            origin=f'single-bundle exotic: ({r3},{r2},{Y})')
                        out.append(BetaContributionCandidate(
                            name=f'exotic_({r3},{r2},Y={Y})_{stat}_x{mult}',
                            bundles=(b,),
                            origin_description=f'Single bundle: ({r3},{r2},Y={Y}) '
                                              f'{stat} × {mult}'))
    return out


def enumerate_full_menu() -> list[BetaContributionCandidate]:
    """Full β-contribution candidate menu.

    Combines hand-curated substrate-motivated candidates with systematic
    cross-product enumeration over substrate-bounded (gen, sfermion-subset,
    gaugino, Higgsino, extra-Higgs) tuples, plus single-bundle exotic
    reps as a structural sanity check.
    """
    return (enumerate_substrate_candidates()
            + enumerate_systematic()
            + enumerate_exotic_reps())
