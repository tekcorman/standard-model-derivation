"""
Crystal-net realization menu — the framework's ACTUAL substrate candidate set.

Two distinct MDL axes determine the substrate; this module is the second one:

  Axis A — COXETER-QUOTIENT (relation structure of the toggle stream).
    Which quotient of F_inv(|E|) compresses the substrate stream. Enumerated
    in `menus/coxeter.py`, scored in `gating/mdl.py`. Raw substrate-only MDL
    at framework scale picks a high-|E| exceptional quotient — i.e. it does
    NOT single out k*=3 (the `sector_coxeter_full_menu_ranking_audit.py`
    "skeptical bridge probe": k*=3 is observer-side, not substrate-only-MDL).

  Axis B — CRYSTAL-NET REALIZATION (the spatial substrate).  ← THIS MODULE
    The substrate is srs (the (10,3)-a / Laves / K₄ net, space group I4_132,
    girth 10), forced STRUCTURALLY — NOT by a DL tiebreak:
      (A) self-containment ⟹ no privileged spatial direction or edge-orientation
        ⟹ the walker's causal state is a directed edge (Shalizi-Crutchfield) the
        observer's model must treat all directed edges as equivalent ⟹ the model
        is strongly isotropic (arc-transitive)
      ⟹ by substrate-agnosticism (`theorem_substrate_agnosticism.md`) the SUBSTRATE
        is strongly isotropic
      ⟹ Sunada 2012 (Notices AMS 59(2) 208-215): the UNIQUE strongly-isotropic
        3-regular 3-connected ℝ³ crystal net is srs (up to handedness). With
        k*=3, d=3 (`k_star_derivation.md`, `d_spatial_derivation.md`) ⟹ srs.
    (R-9 CLOSED — STRUCTURAL, 2026-05-12; see the residue register R-9 entry +
    `walker_dynamics_derivation.md` Step 4b + `g_girth_derivation.md` Step 2.
    "Strong isotropy" is (A)'s no-privilege applied to spatial labels — derived,
    not adopted.) The DL comparison (`dl_comparison.py`) is now a CONSISTENCY
    CHECK — srs is also DL-minimum and uniquely specifiable by symmetry alone;
    the 8 V+E-transitive-but-not-strongly-isotropic candidates (srs-z, srs-c4,
    srs-c8, srs-c27, lou, lov, okw, hcb-c4) each pay extra description bits (≥2
    arc-orbits = "which-arc-type" structure the directionless observer cannot
    justify). The substrate-net is the MDL-minimum HYPOTHESIS (full DL_model +
    DL_data — Kolmogorov-minimal description of the data), NOT a `channel_select`
    above-waterline ensemble (`channel_select` is for distinct channels, not
    competing whole-substrate hypotheses). The earlier "R-9 = open srs-vs-srs-z
    2.56-bit gap, near-closure via Wyckoff-x≈0.6607 encoding" framing is RETRACTED
    (the γ.2 polynomial was the wrong object; the M2a +3.25-bit refinements were
    cherry-picked add-ons; the closure does not use the data term at all).
    srs-z is NOT a competing substrate — it is the BIPARTITE DOUBLE COVER of srs
    (8-atom Q₃ quotient = bipartite double of srs's K₄ quotient), carrying the
    Witten-SUSY-QM χ̃ grading: the substrate-level home of the framework's
    adopted SUSY/MSSM structure. ⇒ R-9 and the MSSM-adoption gap are the SAME
    question (quotient vs cover) — see frontier item "MSSM as adoption".

DATA: vendored. `data/rcsr_candidates_snapshot.json` is a date-stamped parsed
snapshot (via `proofs.foundations.rcsr_net_assessment.parse_rcsr_3dall`) of the
RCSR crystal-net candidates the substrate apparatus references — the 9
V+E-transitive 3-c chiral 3D cubic nets, the achiral / hexagonal 3-regular nets
the A2-T program references, and a set of non-3-regular reference nets for
DL/coordination comparison. Regenerate with `data/_refresh_rcsr_snapshot.py`
(see its docstring). So Axis-B fingerprints work with NO network dependency.

LOGIC: delegated (for now). The per-net structural+spectral+DL FINGERPRINT
computation and the DL minimization stay in the mature `proofs/foundations/`
apparatus — this module loads the vendored data and calls into them:
  - rcsr_net_assessment.py            build the net + Bloch operators + spectra
  - rcsr_per_substrate_fingerprint.py uniform per-net fingerprint
  - rcsr_candidate_sweep.py           χ̃ / bipartiteness sweep across the 9 chiral candidates
  - dl_comparison.py                  DL(net) = DL(SG)+DL(Wyckoff)+DL(coords)+DL(edges) → srs (consistency check)
  - srs_vs_srs_z_dl_audit.py, qtz_vs_srs_dl_comparison.py, lov_dl_audit.py  pairwise DL audits
  - r9_srsz_simulator_run.py + simulator/srsz_substrate.py  the srs-z double-cover characterization
  - substrate_lattice_waterfilling_batch.py + an internal working note
                                      A2-T (Boltzmann-weighted-MDL) gating, channel-by-channel

╔══════════════════════════════════════════════════════════════════════════╗
║ ABSORB TARGET (option c — R-9 has landed, 2026-05-12, so this is now      ║
║ UNBLOCKED). The eventual clean architecture: this module OWNS the         ║
║ crystal-net machinery, and the `proofs/foundations/` probes above become  ║
║ thin wrappers over it (correct dependency direction — the substrate       ║
║ computer is upstream of the audits). What moves in: (1) the STRUCTURAL    ║
║ selector — (A) ⟹ no privileged direction ⟹ arc-transitive substrate ⟹    ║
║ Sunada uniqueness ⟹ srs (the new load-bearing argument; see              ║
║ `framework_substrate_selection`); (2) `assess_net`'s fingerprint          ║
║ computation (Bloch-Hashimoto build, high-symmetry-k spectra, Ramanujan-   ║
║ doublet detection, γ_7^A lift); (3) `dl_comparison.py`'s DL accounting    ║
║ AS A CONSISTENCY CHECK (NOT the M2a +3.25-bit refinements nor the γ.2     ║
║ Wyckoff-free-parameter encoding — both RETRACTED by the R-9 closure);     ║
║ (4) the A2-T channel-waterfilling (→ a `gating/waterfilling.py` module).  ║
║ What STAYS: the public interface here (`enumerate_candidates`, `get_net`, ║
║ `framework_substrate`, `framework_substrate_selection`,                   ║
║ `chirality_channel_contributors`, `rcsr_fingerprint`, `dl_comparison`,    ║
║ `CANDIDATE_NETS`, `CrystalNet`) — callers don't change. The seam is the   ║
║ `_backend_*` functions below; option (c) = replace their bodies           ║
║ (delegate → in-house). Coordinate with whatever the bg linter is in       ║
║ `proofs/foundations/` doing next (post-R-9 it moved to M_R waterfilling). ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Channels (from an internal working note §2a).
CHANNELS = ('C1_spectral', 'C2_combinatorial', 'C3_chirality',
            'C4_dark_cosmo', 'C5_liv', 'C6_gauge')

_ALL_CHANNELS = CHANNELS
_NONCHIRAL = ('C1_spectral', 'C2_combinatorial', 'C4_dark_cosmo', 'C5_liv')  # no C3; partial C6


@dataclass(frozen=True)
class CrystalNet:
    """One crystal-net substrate-realization candidate (or reference net).

    Static metadata mirroring rcsr_candidate_sweep_2026-05-01.md +
    substrate_a2t_waterfilling_program.md + the R-9 prep doc. The live
    structural/spectral fingerprint is `rcsr_fingerprint(name)`.

    Attributes:
        name           : RCSR symbol ('srs', 'srs-z', 'ths', 'qtz', …)
        space_group    : Hermann-Mauguin space group (RCSR notation)
        chiral         : True iff a Sohncke (chiral) space group (no inversion center)
        coordination   : vertex coordination k*
        kind           : 'framework_substrate' | 'chiral_cubic_candidate' |
                         'achiral_or_noncubic_3regular' | 'reference_other_coord'
        n_atoms_prim   : primitive-cell vertex count |V|_prim (None if not catalogued here)
        n_edges_prim   : primitive-cell edge count |E|_prim (multigraph for I-centered)
        girth          : crystal-lattice girth (drives α_1 = q_NB^(g−2)); None if unknown
        bipartite      : 'BIPARTITE' | 'NOT_BIPARTITE' | 'DISCONNECTED' | None
        dl_struct_bits : Convention-B Level-2 structural description length (bits); None if unknown
        arc_transitive : True iff strongly isotropic (crystallographic automorphism group
                         transitive on (vertex, directed-edge) pairs ⇒ ONE arc-orbit). srs is the
                         UNIQUE 3-reg 3-conn ℝ³ crystal net with this property (Sunada 2012) — which
                         is WHY srs is the substrate (R-9 closure: it's forced by (A)'s no-privilege
                         applied to spatial labels, not picked by a DL tiebreak). All other entries
                         here are at best V+E-transitive (≥2 arc-orbits = "which-arc-type" structure
                         a directionless observer cannot justify ⇒ extra description bits).
        channels       : tuple of channels this candidate contributes to (others hard-gated);
                         () for reference nets (not substrate candidates)
        in_framework_candidate_set : True iff in R-9's V+E-transitive 3-c chiral 3D cubic set
                                     OR the A2-T-referenced non-chiral-channel set
        is_framework_substrate : True for srs only
        notes          : provenance / cross-reference
    """
    name: str
    space_group: str
    chiral: bool
    coordination: int
    kind: str
    n_atoms_prim: Optional[int] = None
    n_edges_prim: Optional[int] = None
    girth: Optional[int] = None
    bipartite: Optional[str] = None
    dl_struct_bits: Optional[float] = None
    channels: tuple = ()
    in_framework_candidate_set: bool = False
    is_framework_substrate: bool = False
    arc_transitive: bool = False
    notes: str = ''


# ---------------------------------------------------------------------------
# Curated candidate set (the substrate candidates the framework's apparatus uses)
# ---------------------------------------------------------------------------

CANDIDATE_NETS = [
    # The 9 V+E-transitive 3-connected chiral 3D CUBIC candidate nets
    # (rcsr_net_assessment.py:592 / rcsr_candidate_sweep_2026-05-01.md). srs is
    # the ONLY strongly-isotropic (arc-transitive) one — that's the R-9 closure:
    # (A) no privileged direction ⟹ arc-transitive substrate model ⟹ (Sunada
    # 2012) srs unique. The other 8 are V+E-transitive but have ≥2 arc-orbits
    # (= extra "which-arc-type" structure), so they're NOT competing substrates;
    # srs-z in particular is the bipartite DOUBLE COVER of srs (the χ̃/SUSY layer).
    CrystalNet('srs', 'I4_132', True, 3, 'framework_substrate', 4, 12, 10, 'NOT_BIPARTITE', 12.17,
               _ALL_CHANNELS, in_framework_candidate_set=True, is_framework_substrate=True,
               arc_transitive=True,
               notes='(10,3)-a / Laves / K₄ net; the UNIQUE strongly-isotropic 3-reg 3-conn ℝ³ '
                     'crystal net (Sunada 2012) ⇒ THE framework substrate, forced by (A) ⟹ '
                     'no privileged direction ⟹ arc-transitive substrate model ⟹ srs. R-9 CLOSED '
                     '— STRUCTURAL (2026-05-12); also DL-minimum / uniquely symmetry-specifiable '
                     '(a consistency check, not the selector).'),
    CrystalNet('srs-z', 'P4_132', True, 3, 'chiral_cubic_candidate', 8, 12, 10, 'BIPARTITE', 12.17,
               _ALL_CHANNELS, in_framework_candidate_set=True, arc_transitive=False,
               notes='(10,3)-b. NOT a competing substrate — it is the BIPARTITE DOUBLE COVER of '
                     'srs (8-atom Q₃ quotient = bipartite double of srs\'s K₄ quotient; adjacency '
                     'spectrum = ±srs\'s, doubled; h=(√3+i√5)/2 mult 4 vs srs\'s 2 at the BZ '
                     'corner). 1/2-arc-transitive (2 arc-orbits) ⇒ extra description bits. Carries '
                     'the Witten-SUSY-QM χ̃ Z₂ grading (the srs_z_chi_* operator): the substrate-'
                     'level home of the framework\'s adopted SUSY/MSSM structure. ⇒ R-9 ≡ the '
                     'MSSM-adoption question (quotient vs cover). The earlier "DL ties srs / 2.56-'
                     'bit gap / Wyckoff-x≈0.6607 encoding closes it" framing is RETRACTED.'),
    CrystalNet('srs-c4', 'P4_232', True, 3, 'chiral_cubic_candidate', 4, 6, None, 'NOT_BIPARTITE', None,
               _ALL_CHANNELS, in_framework_candidate_set=True, arc_transitive=False,
               notes='catenated srs ×4; |E|_prim=6; V+E-transitive but ≥2 arc-orbits ⇒ extra bits (excluded by R-9)'),
    CrystalNet('srs-c8', 'I432', True, 3, 'chiral_cubic_candidate', 4, 12, None, 'NOT_BIPARTITE', None,
               _ALL_CHANNELS, in_framework_candidate_set=True,
               notes='catenated srs ×8; non-uniform conventional degree sequence — flagged'),
    CrystalNet('srs-c27', 'I4_132', True, 3, 'chiral_cubic_candidate', 4, 12, None, 'NOT_BIPARTITE', None,
               _ALL_CHANNELS, in_framework_candidate_set=True, notes='catenated srs ×27'),
    CrystalNet('lou', 'I4_132', True, 3, 'chiral_cubic_candidate', 12, 36, None, 'NOT_BIPARTITE', None,
               _ALL_CHANNELS, in_framework_candidate_set=True,
               notes='12-atom primitive; uses RCSR "Eq" aux edge orbit (parser-fix dependent)'),
    CrystalNet('lov', 'I4_132', True, 3, 'chiral_cubic_candidate', 12, 36, None, 'BIPARTITE', None,
               _ALL_CHANNELS, in_framework_candidate_set=True,
               notes='12-atom primitive (6+6); NEW bipartite-cover-shadow partner of srs-z '
                     '(γ_7^A → −χ̃; {χ̃, B(k)}=0 at every k) — rcsr_candidate_sweep_2026-05-01.md'),
    CrystalNet('okw', 'I4_132', True, 3, 'chiral_cubic_candidate', 12, 36, None, 'NOT_BIPARTITE', None,
               _ALL_CHANNELS, in_framework_candidate_set=True,
               notes='12-atom primitive; uses RCSR "Eq" aux edge orbit'),
    CrystalNet('hcb-c4', 'P4_332', True, 3, 'chiral_cubic_candidate', 8, 12, None, 'DISCONNECTED', None,
               _ALL_CHANNELS, in_framework_candidate_set=True,
               notes='catenated honeycomb ×4; disconnected primitive (2 of 8 BFS-reachable) — anomalous'),
    # Achiral / non-cubic 3-regular nets the A2-T program references for the
    # non-chiral channels (substrate_a2t_waterfilling_program.md §2b):
    CrystalNet('ths', 'I4_1/amd', False, 3, 'achiral_or_noncubic_3regular', None, None, None, None, 13.85,
               _NONCHIRAL, in_framework_candidate_set=True,
               notes='ThSi_2 net (R-7); centrosymmetric ⇒ hard-gated out of the chirality channel; '
                     'dark/cosmo + spectral + combinatorial channels only'),
    CrystalNet('ths-z', 'I4_1/amd', False, 3, 'achiral_or_noncubic_3regular', None, None, None, None, None,
               _NONCHIRAL, notes='ThSi_2-z variant; centrosymmetric — same channel gating as ths'),
    CrystalNet('eta', 'P6_222', True, 3, 'achiral_or_noncubic_3regular', None, None, None, None, None,
               _NONCHIRAL, in_framework_candidate_set=True,
               notes='hexagonal 3-regular net; A2-T-referenced; outside R-9\'s CUBIC chiral '
                     'enumeration ⇒ not asserted as a chiral-channel substrate — non-chiral channels only'),
    CrystalNet('utj', 'P4_2/nbc', False, 3, 'achiral_or_noncubic_3regular', None, None, None, None, None,
               _NONCHIRAL, in_framework_candidate_set=True,
               notes='centrosymmetric 3-regular net; A2-T-referenced (dark channel); chirality-gated'),
    # Non-3-regular REFERENCE nets — NOT substrate candidates (the framework
    # requires k*=3); kept for DL / coordination comparison (k_star_derivation,
    # qtz_vs_srs_dl_comparison, dia=R-8, etc.).
    CrystalNet('qtz', 'P6_222', True, 4, 'reference_other_coord', None, None, None, None, None,
               (), notes='α-quartz net; DL-comparison reference (qtz_vs_srs_dl_comparison.py)'),
    CrystalNet('dia', 'Fd-3m', False, 4, 'reference_other_coord', None, None, None, None, None,
               (), notes='diamond net (R-8); 4-coordinated — coordination-comparison reference'),
    CrystalNet('dia-c', 'Pn-3m', False, 4, 'reference_other_coord', notes='catenated diamond — reference'),
    CrystalNet('pcu', 'Pm-3m', False, 6, 'reference_other_coord', notes='primitive cubic — reference'),
    CrystalNet('nbo', 'Im-3m', False, 4, 'reference_other_coord', notes='NbO net — reference'),
    CrystalNet('bcu', 'Im-3m', False, 8, 'reference_other_coord', notes='body-centered cubic — reference'),
    CrystalNet('fcu', 'Fm-3m', False, 12, 'reference_other_coord', notes='face-centered cubic — reference'),
    CrystalNet('sod', 'Im-3m', False, 4, 'reference_other_coord', notes='sodalite net — reference'),
    CrystalNet('rho', 'Im-3m', False, 4, 'reference_other_coord', notes='RHO zeolite net — reference'),
    CrystalNet('lvt', 'I4_1/amd', False, 4, 'reference_other_coord', notes='lvt net — reference'),
    CrystalNet('cds', 'P4_2/mmc', False, 4, 'reference_other_coord', notes='CdSO4 net — reference'),
    CrystalNet('crs', 'Fd-3m', False, 6, 'reference_other_coord', notes='C(rystobalite)-related — reference'),
    CrystalNet('unc', 'P4_122', False, 4, 'reference_other_coord', notes='reference'),
    CrystalNet('und', 'I4_1/acd', False, 4, 'reference_other_coord', notes='reference'),
    CrystalNet('une', 'R-3', False, 4, 'reference_other_coord', notes='reference'),
    CrystalNet('unj', 'P6_122', True, 4, 'reference_other_coord', notes='reference'),
]

# Substrate candidates the framework's apparatus actually uses (k*=3; in R-9's
# enumeration or the A2-T non-chiral-channel set). Reference nets excluded.
_FRAMEWORK_CANDIDATE_NAMES = [c.name for c in CANDIDATE_NETS if c.in_framework_candidate_set]

# Note on d>3 substrates: the A2-T program also references R-4 / R-5 (dimension
# > 3 substrates) for the dark/cosmological channel. Those are NOT RCSR 3D nets
# and have no `3dall.txt` entry — not vendorable here. They live in the dark-
# sector buildup machinery (predictions/H_multiway_dim_count.py etc.).
D_GT_3_REFERENCED_NOT_VENDORABLE = ('R-4 (d=4 substrate)', 'R-5 (higher-d substrate)')


def _by_name():
    return {c.name: c for c in CANDIDATE_NETS}


# ---------------------------------------------------------------------------
# Public menu API (stable across the eventual option-(c) absorb)
# ---------------------------------------------------------------------------

def enumerate_candidates(include_reference: bool = False) -> list[CrystalNet]:
    """The crystal-net substrate-realization candidate set.

    By default, the framework's substrate candidates only (k*=3; R-9's
    V+E-transitive 3-c chiral 3D cubic set + the A2-T non-chiral-channel nets).
    With `include_reference=True`, also the non-3-regular reference nets used
    only for DL / coordination comparison.
    """
    if include_reference:
        return list(CANDIDATE_NETS)
    return [c for c in CANDIDATE_NETS if c.in_framework_candidate_set]


def get_net(name: str) -> CrystalNet:
    """Look up one net by RCSR symbol (candidate or reference)."""
    nets = _by_name()
    if name not in nets:
        raise ValueError(f"crystal_nets.get_net: no net {name!r}; have {sorted(nets)}")
    return nets[name]


def framework_substrate() -> CrystalNet:
    """srs — the framework's substrate.

    Forced STRUCTURALLY (R-9 closure, 2026-05-12), NOT by a DL tiebreak — see
    `framework_substrate_selection()` for the chain. srs is the unique
    strongly-isotropic (arc-transitive) 3-regular 3-connected ℝ³ crystal net
    (Sunada 2012); arc-transitivity is forced by axiom (A)'s no-privilege
    applied to spatial labels. (It is also DL-minimum and uniquely
    symmetry-specifiable — a consistency check, not the selector.)
    """
    return get_net('srs')


def framework_substrate_selection() -> dict:
    """The structural argument that selects srs (the R-9 closure).

    Every step is (A), a published theorem, or a derived framework theorem — no
    adopted lattice property, no cherry-picked bit-count, no data fit:

      1. (A) self-containment ⟹ no privileged spatial direction or edge-
         orientation (a "which-way" datum (A) forbids supplying — the same
         no-privilege that forces the uniform substrate measure,
         `theorem_toggle_from_self_containment.md` Step 1, and the absent
         inter-generator commutation, Step 7).
      2. The walker's causal state is a directed edge (Shalizi-Crutchfield 2001;
         `walker_dynamics_derivation.md` Step 5) ⟹ by (1) the observer's model
         treats all directed edges as equivalent ⟹ the model is strongly
         isotropic (crystallographic automorphism group transitive on (vertex,
         directed-edge) pairs — arc-transitive).
      3. By substrate-agnosticism (`theorem_substrate_agnosticism.md` — the
         substrate IS the observer's description-length-minimal canonical model)
         ⟹ the substrate is strongly isotropic. So strong isotropy is DERIVED
         from (A), not adopted.
      4. Sunada 2012 (Notices AMS 59(2) 208-215): the UNIQUE 3-regular
         3-connected ℝ³ crystal net that is strongly isotropic is srs (the
         Laves / K₄ / (10,3)-a net), up to handedness. With k* = 3, d = 3
         (`k_star_derivation.md`, `d_spatial_derivation.md`) ⟹ the substrate is srs.

    Front-end derivation: `walker_dynamics_derivation.md` Step 4b +
    `g_girth_derivation.md` Step 2. Residue register: R-9 (CLOSED — STRUCTURAL,
    2026-05-12). The 8 V+E-transitive-but-not-strongly-isotropic candidates
    (srs-z, srs-c4, srs-c8, srs-c27, lou, lov, okw, hcb-c4) each pay extra
    description bits (≥2 arc-orbits) and cannot be specified by symmetry alone.
    """
    return {
        'substrate': 'srs',
        'closure': 'R-9 — CLOSED, STRUCTURAL (2026-05-12)',
        'chain': [
            '(A) self-containment ⟹ no privileged spatial direction / edge-orientation',
            'walker causal state = directed edge ⟹ all directed edges equivalent ⟹ model is arc-transitive (strongly isotropic)',
            'substrate-agnosticism ⟹ the substrate is strongly isotropic (derived from (A), not adopted)',
            'Sunada 2012 ⟹ srs is the unique strongly-isotropic 3-reg 3-conn ℝ³ crystal net; with k*=3, d=3 ⟹ substrate = srs',
        ],
        'front_end_docs': ['predictions/walker_dynamics_derivation.md (Step 4b)',
                           'predictions/g_girth_derivation.md (Step 2)',
                           'docs/audits/registers/structural_residue_register.md (R-9)'],
        'dl_role': ('consistency check — srs is also DL-minimum and uniquely '
                    'symmetry-specifiable; the M2a +3.25-bit refinements and the '
                    'γ.2 Wyckoff-x≈0.6607 polynomial encoding are RETRACTED as load-bearing'),
        'srs_z_role': ('NOT a competing substrate — the bipartite double cover of srs, '
                       'carrying the Witten-SUSY-QM χ̃ grading; R-9 ≡ the MSSM-adoption question'),
    }


def chirality_channel_contributors() -> list[CrystalNet]:
    """Candidates that contribute to the chirality channel (chiral cubic nets).

    Per audit-v2 Phase 1d + Sunada 2012, only srs survives the chirality +
    Bloch-decomposable filters for the *chiral* observable channels (the other
    chiral-cubic candidates are hard-gated by additional structural arguments —
    odd-cycle obstruction, etc.); non-chiral channels see the centrosymmetric /
    d>3 ensemble at small Boltzmann weight. See `substrate_lattice_waterfilling_batch.py`.
    Returns the chiral-cubic candidate set (srs + the other 8); the achiral /
    non-cubic 3-regular nets (ths, eta, utj, …) are chirality-gated.
    """
    return [c for c in CANDIDATE_NETS if c.chiral and c.kind in
            ('framework_substrate', 'chiral_cubic_candidate')]


# ---------------------------------------------------------------------------
# Vendored snapshot loader
# ---------------------------------------------------------------------------

_SNAPSHOT_PATH = Path(__file__).resolve().parent / 'data' / 'rcsr_candidates_snapshot.json'
_snapshot_cache = None


def snapshot() -> dict:
    """The vendored parsed RCSR snapshot ({'_meta': …, 'entries': {name: …}}).

    Loaded once, cached. Regenerate with `data/_refresh_rcsr_snapshot.py`.
    """
    global _snapshot_cache
    if _snapshot_cache is None:
        with open(_SNAPSHOT_PATH) as f:
            _snapshot_cache = json.load(f)
    return _snapshot_cache


def snapshot_meta() -> dict:
    """The snapshot's `_meta` block (source URL, SHA-256, fetch date, net list)."""
    return snapshot()['_meta']


def snapshot_net_names() -> list[str]:
    """All RCSR net symbols available in the vendored snapshot."""
    return sorted(snapshot()['entries'])


def snapshot_entry(name: str) -> Optional[dict]:
    """The vendored parsed RCSR entry for `name`, or None if not in the snapshot."""
    return snapshot()['entries'].get(name)


# ---------------------------------------------------------------------------
# Computation backends — DELEGATE for now; option (c) replaces these bodies.
# ---------------------------------------------------------------------------

_RCSR_DATA_FILE = '/tmp/rcsr_3d_current.txt'


def _repo_on_path():
    import sys
    repo = Path(__file__).resolve().parents[2]   # .../simulator/menus/crystal_nets.py → repo
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def _backend_fingerprint(name: str) -> dict:
    """Compute the per-net structural+spectral fingerprint.

    OPTION-(c) SEAM. Currently: load the parsed RCSR entry (vendored snapshot
    first, then the live `/tmp` file) and call
    `proofs.foundations.rcsr_net_assessment.assess_net` on it. Option (c) =
    replace this body with the in-house fingerprint computation. Returns
    {'source': 'vendored_snapshot' | 'live_rcsr_file', 'fingerprint': <dict>}
    or {'source': None, 'reason': <str>} on failure.
    """
    entry = snapshot_entry(name)
    source = 'vendored_snapshot'
    if entry is None:
        # fall back to the live RCSR file
        import os
        if not os.path.exists(_RCSR_DATA_FILE):
            return {'source': None, 'reason': (
                f'{name!r} not in vendored snapshot and {_RCSR_DATA_FILE} absent — '
                f'add it to data/_refresh_rcsr_snapshot.py and regenerate, or fetch '
                f'`curl -sL https://rcsr.anu.edu.au/data/3dall.txt -o {_RCSR_DATA_FILE}`')}
        _repo_on_path()
        try:
            from proofs.foundations.rcsr_net_assessment import parse_rcsr_3dall
            entry = parse_rcsr_3dall(_RCSR_DATA_FILE, [name]).get(name)
            source = 'live_rcsr_file'
        except Exception as e:
            return {'source': None, 'reason': f'parse_rcsr_3dall failed: {type(e).__name__}: {e}'}
        if entry is None:
            return {'source': None, 'reason': f'{name!r} not found in {_RCSR_DATA_FILE}'}
    _repo_on_path()
    try:
        from proofs.foundations.rcsr_net_assessment import assess_net
        fp = assess_net(entry, verbose=False)
    except Exception as e:  # parser/spglib drift etc. — degrade gracefully
        return {'source': source, 'fingerprint': None, 'snapshot_entry': entry,
                'reason': f'assess_net delegation raised: {type(e).__name__}: {e}'}
    if not fp or 'adj_eigenvalues' not in fp:
        # assess_net's bond reconstruction couldn't build this net — happens for
        # non-cubic / hexagonal RCSR entries (the current probe targets the cubic
        # 3-regular candidate set; the raw parsed entry is still returned).
        return {'source': source, 'fingerprint': None, 'snapshot_entry': entry,
                'reason': ('assess_net could not build this net (likely a non-cubic / '
                           'hexagonal RCSR entry the current probe does not handle); '
                           'raw parsed entry is in snapshot_entry')}
    return {'source': source, 'fingerprint': fp, 'snapshot_entry': entry}


def _backend_dl() -> dict:
    """Crystal-net description-length comparison — a CONSISTENCY CHECK.

    Post-R-9: the DL comparison is no longer the substrate SELECTOR (that is the
    (A) ⟹ arc-transitive ⟹ Sunada ⟹ srs chain — `framework_substrate_selection`).
    It's the cross-check that srs is *also* DL-minimum / uniquely
    symmetry-specifiable, and that the V+E-transitive-but-not-arc-transitive
    competitors pay extra description bits. The M2a +3.25-bit refinements
    (`srs_vs_srs_z_dl_audit.py`) and the γ.2 Wyckoff-x≈0.6607 polynomial
    encoding (`r9_srs_z_free_parameter_audit.py`) are RETRACTED as load-bearing.

    OPTION-(c) SEAM. Currently delegates to `proofs.foundations.dl_comparison`.
    Option (c) = replace with the in-house DL accounting (kept as a consistency
    check; no M2a/γ.2 add-ons).
    """
    static = {c.name: c.dl_struct_bits for c in CANDIDATE_NETS if c.dl_struct_bits is not None}
    _repo_on_path()
    try:
        from proofs.foundations import dl_comparison as _dlc
        return {'available': True, 'module': _dlc.__name__, 'static_dl_struct_bits': static,
                'role': 'consistency check (NOT the selector — see framework_substrate_selection)',
                'note': ('run `python proofs/foundations/dl_comparison.py` for the full '
                         'srs-vs-all-3-regular-nets DL minimization. srs-vs-srs-z: DL ties at '
                         '12.17 bits (M2a-only) — but R-9 is CLOSED structurally regardless '
                         '(arc-transitivity + Sunada); srs-z is the double cover, not a rival.')}
    except Exception as e:
        return {'available': False, 'reason': f'{type(e).__name__}: {e}',
                'static_dl_struct_bits': static,
                'role': 'consistency check (NOT the selector)'}


# ---------------------------------------------------------------------------
# Public delegation API (stable across option (c))
# ---------------------------------------------------------------------------

def rcsr_data_available() -> bool:
    """True iff a live RCSR `3dall.txt` is present (in addition to the vendored snapshot)."""
    import os
    return os.path.exists(_RCSR_DATA_FILE)


def rcsr_fingerprint(name: str) -> dict:
    """Per-net structural+spectral+DL fingerprint.

    Returns a dict with:
      - the static metadata from `CANDIDATE_NETS` (when `name` is catalogued)
      - `raw_snapshot_entry` — the vendored parsed RCSR entry (SG, cell, vertex
        & edge orbits, Wyckoff data, coordination sequence, vertex symbol)
      - `fingerprint_source` ∈ {'vendored_snapshot', 'live_rcsr_file', None}
      - `snapshot_meta` — the snapshot's fetch date + source SHA-256
      - `live_fingerprint` + `available: True` — the `assess_net` output (built
        net + adjacency spectrum + Bloch spectra at high-symmetry k-points +
        Ramanujan doublets + bonds) WHEN `assess_net` can build the net (it
        targets the cubic 3-regular set; for non-cubic / hexagonal RCSR entries
        it can't, so `available: False` + `reason` + the raw parsed entry stand in).

    Works for any net in the vendored snapshot (the 9 chiral cubic candidates,
    the achiral/hexagonal 3-regular nets, and the non-3-regular reference nets).
    """
    nets = _by_name()
    static_d = {}
    if name in nets:
        c = nets[name]
        static_d = {
            'name': c.name, 'space_group': c.space_group, 'chiral': c.chiral,
            'coordination': c.coordination, 'kind': c.kind,
            'n_atoms_prim': c.n_atoms_prim, 'n_edges_prim': c.n_edges_prim,
            'girth': c.girth, 'bipartite': c.bipartite, 'dl_struct_bits': c.dl_struct_bits,
            'channels': c.channels, 'in_framework_candidate_set': c.in_framework_candidate_set,
            'is_framework_substrate': c.is_framework_substrate, 'notes': c.notes,
        }
    elif name not in snapshot()['entries']:
        raise ValueError(f"rcsr_fingerprint: {name!r} is not a catalogued net and "
                         f"not in the vendored snapshot ({snapshot_net_names()})")
    else:
        static_d = {'name': name, 'kind': 'snapshot-only (not in CANDIDATE_NETS)',
                    'notes': 'no curated metadata — vendored snapshot entry only'}
    res = _backend_fingerprint(name)
    common = {**static_d, 'fingerprint_source': res.get('source'),
              'raw_snapshot_entry': res.get('snapshot_entry'),
              'snapshot_meta': {'fetched_or_refreshed': snapshot_meta()['fetched_or_refreshed'],
                                'source_sha256': snapshot_meta()['source_sha256']}}
    if res.get('fingerprint') is not None:
        return {**common, 'live_fingerprint': res['fingerprint'], 'available': True}
    return {**common, 'live_fingerprint': None, 'available': False, 'reason': res.get('reason')}


def dl_comparison() -> dict:
    """Crystal-net description-length comparison — a CONSISTENCY CHECK, not the selector.

    Post-R-9 the substrate is selected structurally ((A) ⟹ arc-transitive ⟹
    Sunada ⟹ srs — `framework_substrate_selection()`); this is the cross-check
    that srs is *also* DL-minimum. Returns the static DL_struct values (srs ties
    srs-z at 12.17 bits under M2a-only accounting — fine, since the structural
    closure doesn't use the data term; ths = 13.85) + `role` + a pointer to
    `proofs.foundations.dl_comparison`, or `{'available': False, …}` if that
    module isn't importable.
    """
    return _backend_dl()
