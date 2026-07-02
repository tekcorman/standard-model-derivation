"""
S1 — gauge-structure tuples (Tasks A-E: the gauge zoo).

A *gauge tuple* is a (substrate, vertex-algebra, edge-algebra) triple together
with the gauge group it generates: the vertex algebra Cl(2k*) gives Spin(2k*)
(at the trivalent srs, Cl(6) ⟹ Spin(6) = SU(4) — Pati-Salam color); the edge
qubit algebra Cl(0,2) ≅ ℍ gives SU(2)_L × SU(2)_R (the Pati-Salam left/right
doublets, via `theorem_g2_edge_qubit_su2.md`). So the framework's gauge tuple
is (srs, Cl(6,0), Cl(0,2) ≅ ℍ) ⟹ **SU(4) × SU(2)_L × SU(2)_R** (Pati-Salam),
which embeds into SO(10) / E_6 / … (and breaks to the SM via the K_4 quotient).

After R-9 closed (srs forced) + the observer conditioning (k* = 3 ⟹ vertex
algebra Cl(2·3) = Cl(6); G_2 edge-qubit theorem ⟹ edge algebra Cl(0,2)), the
framework's gauge tuple is fixed. The OTHER tuples here are plurally co-retained
per A2-T (different vertex/edge algebras on srs, or different substrates) and
produce different gauge groups — including the Layer-1-escape candidates
(𝕆 vertex ⟹ G_2; Tits-Freudenthal magic square ⟹ F_4 / E_6 / E_7 / E_8) which
are audited NEGATIVE/UNCONNECTED through every channel M1-M7 (`frontier.layer1_escapes`):
the saturated symmetry zoo is formally rich but observably barren.

Source: Tasks A-E (commits 2c2a624 / 7748658 / a648f98 / 51edbc8 / d5bdc45,
2026-05-07) — `sector_{local_algebra, edge_algebra, combined_gauge}_zoo_audit.py`,
`sector_cooling_cascade_audit.py`, `sector_zoo_framework_connection_audit.py`,
`sector_saturation_state_scoping_audit.py`; the PS realization
`proofs/gauge/k4_pati_salam_cl8.py` + `cl8_verification.py`; the GUT embeddings
`proofs/gauge/{k5_gut_cl10, srs_so10_embedding}.py`.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GaugeTuple:
    """One (substrate, vertex-algebra, edge-algebra) → gauge-group tuple.

    Attributes:
        substrate      : crystal-net name (the framework's is 'srs')
        vertex_algebra : vertex local-algebra name (Cl(6,0) for the framework)
        edge_algebra   : edge qubit-algebra name (Cl(0,2) ≅ ℍ for the framework)
        gauge_group    : the gauge group the tuple generates
        combined_n_attest : worldline length at which the tuple is attested (= max of per-layer)
        kind           : 'framework_dominant' | 'subdominant_clifford' |
                         'subdominant_substrate' | 'layer1_escape_octonion' | 'layer1_escape_magic_square'
        n_attest_computed : True iff combined_n_attest was recomputed from the live menus
                            (only the srs-substrate tuples), else it's the cooling-cascade-audit static value
        notes          : provenance / audit verdict
    """
    substrate: str
    vertex_algebra: str
    edge_algebra: str
    gauge_group: str
    combined_n_attest: int
    kind: str
    n_attest_computed: bool = False
    notes: str = ''


# Static tuple table — mirrors sector_cooling_cascade_audit.py's representative_tuples.
# (The srs-substrate tuples' combined N_attest is recomputed below from the live menus.)
_STATIC_TUPLES = [
    ('srs', 'Cl(6,0)',          'Cl(0,2) ≅ ℍ edge', 'SU(4) × SU(2)_L × SU(2)_R (Pati-Salam)', 59049,
     'framework_dominant',
     'THE framework gauge tuple. Cl(6) ⟹ Spin(6) = SU(4) (color); Cl(0,2) ≅ ℍ ⟹ SU(2)_L×SU(2)_R '
     '(theorem_g2_edge_qubit_su2.md). Embeds SO(10)/E_6; breaks to the SM via the K_4 quotient. '
     'k4_pati_salam_cl8.py + cl8_verification.py + srs_so10_embedding.py.'),
    ('srs', 'Cl(8,0)',          'Cl(0,2) ≅ ℍ edge', 'Spin(8) × SU(2)_L × SU(2)_R', 59049,
     'subdominant_clifford', 'k* = 4 vertex (NOT the trivalent srs vertex) — incompatible with k* = 3; '
     'listed for the zoo. Spin(8) triality is the curiosity here.'),
    ('srs', 'Cl(10,0)',         'Cl(0,2) ≅ ℍ edge', 'Spin(10) GUT × SU(2)_L × SU(2)_R', 59049,
     'subdominant_clifford', 'k* = 5 vertex; incompatible with k* = 3. Spin(10) is the standard GUT '
     'group — the framework reaches it as a SUBGROUP of the embedding, not as the vertex algebra.'),
    ('srs', '𝕆 (octonion)',     'Cl(0,2) ≅ ℍ edge', 'G_2 × SU(2)_L × SU(2)_R', 59049,
     'layer1_escape_octonion', 'Aut(𝕆) = G_2 at the vertex. LAYER-1 ESCAPE CANDIDATE — audited '
     'NEGATIVE/UNCONNECTED via M1-M7 (22/24 I4_132 elements violate the octonion Φ; f_3 saturates '
     'at ~0.80 with ~10^-60 suppression by the GUT epoch). See frontier.layer1_escapes.'),
    ('srs', 'R⊗O = F_4 (52)',   'Cl(0,2) ≅ ℍ edge', 'F_4 × SU(2)_L × SU(2)_R', 59049,
     'layer1_escape_magic_square', 'Tits-Freudenthal magic square (ℝ⊗𝕆 = F_4). Layer-1 escape — '
     'M5/M6 require framework extensions; audited barren via every channel. frontier.layer1_escapes.'),
    ('srs', 'C⊗O = E_6 (78)',   'Cl(0,2) ≅ ℍ edge', 'E_6 × SU(2)_L × SU(2)_R', 59049,
     'layer1_escape_magic_square', 'ℂ⊗𝕆 = E_6. Layer-1 escape; M2 (E_6 → PS valid for ONE generation, '
     'subdominant). frontier.layer1_escapes.'),
    ('srs', 'H⊗O = E_7 (133)',  'Cl(0,2) ≅ ℍ edge', 'E_7 × SU(2)_L × SU(2)_R', 59049,
     'layer1_escape_magic_square', 'ℍ⊗𝕆 = E_7. Layer-1 escape; M6 (does ℍ⊗𝕆 = E_7 live on srs-z, the '
     'double cover?) — flagged OPEN/UNCONNECTED in M_mechanisms_synthesis_2026-05-07.md. frontier.{layer1_escapes, mssm_as_adoption}.'),
    ('srs', 'O⊗O = E_8 (248)',  'Cl(0,2) ≅ ℍ edge', 'E_8 × SU(2)_L × SU(2)_R', 262144,
     'layer1_escape_magic_square', '𝕆⊗𝕆 = E_8 (the vertex-magic curiosity). Layer-1 escape; attests '
     'later (N ≈ 2.6×10^5) but still well below framework scale. frontier.layer1_escapes.'),
    # Tuples on a DIFFERENT substrate (NOT srs — excluded by R-9, kept for the zoo / cooling cascade):
    ('A_3 = S_4 (|E|=3)', 'Cl(6,0)', 'Cl(0,2) ≅ ℍ edge', 'SU(4) × SU(2)² (Pati-Salam, S_4 substrate)', 729,
     'subdominant_substrate', 'S_4 substrate instead of srs — NOT arc-transitive, excluded by R-9. '
     'Same gauge structure but the substrate is wrong.'),
    ('A_4 = S_5 (|E|=4)', 'Cl(8,0)', 'Cl(0,2) ≅ ℍ edge', 'Spin(8) × SU(2)²', 65536,
     'subdominant_substrate', 'Higher-|E| substrate (excluded by d=3 ⇒ |E|=3 conditioning) + k*=4 vertex.'),
    ('E_6 (|E|=6)', 'Cl(12,0)', 'Cl(0,2) ≅ ℍ edge', 'Spin(12) × SU(2)²', 46656,
     'subdominant_substrate', 'E_6 Coxeter substrate (|E|=6, excluded by |E|=3) + k*=6 vertex.'),
    ('E_8 (|E|=8)', 'Cl(16,0)', 'Cl(0,2) ≅ ℍ edge', 'Spin(16) × SU(2)²', 262144,
     'subdominant_substrate', 'E_8 Coxeter substrate (|E|=8, excluded) + k*=8 vertex. The Spin(16) ⊂ E_8 curiosity.'),
]


def _srs_coxeter_n_attest() -> int:
    """N_attest of the srs slice's Coxeter side (= the H_3-equivalent system, 3^10)."""
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from simulator.menus import coxeter
    from simulator.gating import mdl
    return int(mdl.n_attest(coxeter.srs_equivalent()))


def _algebra_n_attest(vertex_name: str, edge_name: str):
    """(vertex.n_attest, edge.n_attest) from the live vertex/edge menus; (None, None) if not found."""
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from simulator.menus import vertex_algebras as va, edge_algebras as ea
    v = next((x.n_attest for x in va.enumerate_full_menu() if x.name == vertex_name), None)
    e = next((x.n_attest for x in ea.enumerate_full_menu() if x.name == edge_name), None)
    return v, e


def _build_tuples() -> list[GaugeTuple]:
    out = []
    srs_cox_na = None
    for substrate, vert, edge, group, static_na, kind, notes in _STATIC_TUPLES:
        na, computed = static_na, False
        if substrate == 'srs':
            if srs_cox_na is None:
                srs_cox_na = _srs_coxeter_n_attest()
            v_na, e_na = _algebra_n_attest(vert, edge)
            if v_na is not None and e_na is not None:
                na, computed = max(srs_cox_na, int(v_na), int(e_na)), True
        out.append(GaugeTuple(substrate=substrate, vertex_algebra=vert, edge_algebra=edge,
                              gauge_group=group, combined_n_attest=na, kind=kind,
                              n_attest_computed=computed, notes=notes))
    return out


GAUGE_TUPLES = _build_tuples()


def enumerate_tuples() -> list[GaugeTuple]:
    """All representative gauge tuples (the gauge zoo)."""
    return list(GAUGE_TUPLES)


def framework_gauge_tuple() -> GaugeTuple:
    """(srs, Cl(6,0), Cl(0,2) ≅ ℍ) ⟹ SU(4) × SU(2)_L × SU(2)_R (Pati-Salam) — the framework's."""
    return next(t for t in GAUGE_TUPLES if t.kind == 'framework_dominant')


def subdominant_tuples() -> list[GaugeTuple]:
    """Plurally-co-retained tuples that aren't the framework one (incl. Layer-1 escapes)."""
    return [t for t in GAUGE_TUPLES if t.kind != 'framework_dominant']


def layer1_escape_tuples() -> list[GaugeTuple]:
    """The Layer-1-escape candidates (octonion / magic-square vertex) — audited barren (frontier.layer1_escapes)."""
    return [t for t in GAUGE_TUPLES if t.kind.startswith('layer1_escape')]


def cooling_cascade_order() -> list[GaugeTuple]:
    """Tuples sorted by combined N_attest (the order they enter the attested zoo as N grows)."""
    return sorted(GAUGE_TUPLES, key=lambda t: t.combined_n_attest)


def summary() -> dict:
    fw = framework_gauge_tuple()
    return {
        'framework_gauge_group': fw.gauge_group,
        'framework_tuple': (fw.substrate, fw.vertex_algebra, fw.edge_algebra),
        'framework_n_attest': fw.combined_n_attest,
        'framework_n_attest_computed': fw.n_attest_computed,
        'n_tuples': len(GAUGE_TUPLES),
        'n_layer1_escapes': len(layer1_escape_tuples()),
        'layer1_verdict': 'audited NEGATIVE/UNCONNECTED via M1-M7 — see frontier.layer1_escapes',
        'source': 'Tasks A-E (2026-05-07) + sector_cooling_cascade_audit.py + k4_pati_salam_cl8.py + srs_so10_embedding.py',
    }
