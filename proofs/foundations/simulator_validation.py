"""
simulator_validation.py

Validation probe for the enumerate-then-MDL-gate simulator rebuild
(simulator/ — the architecture that replaces the cherry-picked
dominant-slice simulator/ + match/).

Tests the menu + gating + zoo layers against the existing one-off audits
they consolidate:
  1. menus.coxeter      — finite/affine/multi-gen/free counts vs the
                          sector_coxeter_*_audit.py + sector_path_B_*.py menus
  2. gating.mdl         — L_elias, description_length, free_word_log_count,
                          compression_value, freq_factor, n_attest,
                          combined_weight vs sector_coxeter_freq_weighted_audit.py
                          (machine precision)
  3. gating.mdl Stage 2 — channel_select / canonical_encoding behaviour
                          (and parity with simulator/kernel.py.channel_select)
  4. gating.cooling     — saturated_zoo / dominant_slice ranking;
                          honest finding: raw substrate-only MDL does NOT
                          single out srs / |E|=3 (matches
                          sector_coxeter_full_menu_ranking_audit.py's verdict)
  5. zoo + substrate    — framework_slice() = srs × Cl(6,0) × Cl(0,2);
                          structural counts via the live SrsSubstrate
  6. kernel             — MDL primitives delegate to gating.mdl; counting
                          primitives delegate to the live simulator for the
                          framework slice (and raise for others)

This is the acceptance-criteria check for the rebuild's menu/gating/zoo
core (README criteria 1-4, partial 5-6). Counting/observables wiring for
non-framework slices (per-Coxeter Cayley-graph builders) is a TODO and is
asserted to raise NotImplementedError, not silently return.
"""

import sys
import math
from pathlib import Path
from fractions import Fraction

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from simulator import CountingKernel, Substrate, zoo
from simulator.menus import coxeter as cm
from simulator.menus import vertex_algebras as vm
from simulator.menus import edge_algebras as em
from simulator.gating import mdl
from simulator.gating import cooling

# the live simulator's audit functions, for parity checks
from proofs.foundations import sector_coxeter_freq_weighted_audit as AUDIT


class TestStats:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def check(self, name, condition, detail=""):
        if condition:
            print(f"  ✓ {name}")
            self.passed += 1
        else:
            print(f"  ✗ {name} — {detail}")
            self.failed.append((name, detail))

    def summary(self):
        total = self.passed + len(self.failed)
        print(f"\n  RESULT: {self.passed}/{total} passed")
        for n, d in self.failed:
            print(f"    FAILED: {n} — {d}")
        return not self.failed


# ---------------------------------------------------------------------------

def test_coxeter_menu(stats):
    print("\n[1] menus.coxeter — enumerators")
    fin = cm.enumerate_finite()
    aff = cm.enumerate_affine()
    hyp = cm.enumerate_hyperbolic()
    free = cm.enumerate_free()
    full = cm.enumerate_full_menu()
    stats.check("enumerate_finite returns ≥ 25 systems", len(fin) >= 25, f"got {len(fin)}")
    stats.check("finite menu spans |E| 2..8",
                {c.generators for c in fin} == {2, 3, 4, 5, 6, 7, 8},
                f"got {sorted({c.generators for c in fin})}")
    stats.check("E_8 (THE exceptional) present with order 696729600",
                any(c.name.startswith('E_8') and c.order == 696729600 for c in fin))
    stats.check("H_3 icosahedral present with order 120",
                any('H_3' in c.name and c.order == 120 and c.generators == 3 for c in fin))
    stats.check("affine menu present (Ã_2, G̃_2, …)", len(aff) >= 8, f"got {len(aff)}")
    stats.check("affine systems carry finite_order + rank",
                all(c.finite_order and c.rank for c in aff))
    # Path B multi-gen sweep: |E|≤8, K∈3..8, m∈{1..32}; matches
    # sector_path_B_multi_gen_audit.py's cell count = Σ over (E,K with K≤E) of 11.
    expect_hyp = sum(11 for E in range(2, 9) for K in range(3, 9) if K <= E)
    stats.check(f"multi-gen sweep has {expect_hyp} cells (matches sector_path_B_multi_gen_audit)",
                len(hyp) == expect_hyp, f"got {len(hyp)}")
    stats.check("free baselines for |E| 2..8", len(free) == 7, f"got {len(free)}")
    stats.check("full menu = finite ∪ affine ∪ multi-gen ∪ free",
                len(full) == len(fin) + len(aff) + len(hyp) + len(free))
    srs = cm.srs_equivalent()
    stats.check("srs_equivalent: |E|=3, order 120, finite",
                srs.generators == 3 and srs.order == 120 and srs.growth_class == 'finite')
    # max_relation_length / n_attest match the freq-weighted audit's max_relation_length
    h3 = next(c for c in fin if 'H_3' in c.name)
    stats.check("H_3 max_relation_length = 10 (= 2·max m = 2·5)",
                h3.max_relation_length == 10)
    e8 = next(c for c in fin if c.name.startswith('E_8'))
    stats.check("E_8 max_relation_length = 6 (= 2·3, all braids m=3)",
                e8.max_relation_length == 6)


def test_vertex_edge_menus(stats):
    print("\n[2] menus.vertex_algebras / edge_algebras")
    cl = vm.enumerate_clifford()
    cd = vm.enumerate_cayley_dickson()
    ms = vm.enumerate_magic_square()
    stats.check("Clifford menu Cl(2..16,0) i.e. k=1..8", len(cl) == 8, f"got {len(cl)}")
    cl6 = next(c for c in cl if c.name == 'Cl(6,0)')
    stats.check("Cl(6,0): dim_real 64, dim_fock 8, k_compat (3,), Aut Spin(6)",
                cl6.dim_real == 64 and cl6.dim_fock == 8 and cl6.k_compat == (3,)
                and cl6.automorphism == 'Spin(6)')
    stats.check("Cayley-Dickson tower ℝ..trigintaduonion (d=0..5)", len(cd) == 6)
    stats.check("𝕆 (octonion): non-associative, normed, Aut G_2",
                any(c.name.startswith('𝕆') and not c.associative and c.normed
                    and c.automorphism == 'G_2' for c in cd))
    stats.check("magic square has the 4 exceptional Lie diagonal entries",
                sum(1 for m in ms if any(x in m.name for x in ('F_4', 'E_6', 'E_7', 'E_8'))) >= 4)
    vdom = vm.framework_dominant()
    stats.check("vertex framework_dominant = Cl(6,0), Aut SU(4)",
                vdom.name == 'Cl(6,0)' and 'SU(4)' in vdom.automorphism)
    ec = em.enumerate_clifford()
    stats.check("edge Clifford menu Cl(0,0..4)", len(ec) == 5)
    edom = em.framework_dominant()
    stats.check("edge framework_dominant = Cl(0,2) ≅ ℍ, SU(2)_L×SU(2)_R",
                edom.signature == (0, 2) and 'SU(2)_L' in edom.automorphism)


def test_mdl_parity(stats):
    print("\n[3] gating.mdl — parity with sector_coxeter_freq_weighted_audit.py")
    # L_elias parity
    for m in [2, 3, 4, 5, 8, 16, 24, float('inf')]:
        stats.check(f"L_elias({m}) = audit.L_elias({m})",
                    mdl.L_elias(m) == AUDIT.L_elias(m), f"{mdl.L_elias(m)} vs {AUDIT.L_elias(m)}")
    # Reproduce a few of the audit's `systems` rows exactly.
    for sysrow in AUDIT.systems:
        E = sysrow['E']
        gc = {'finite': 'finite', 'free': 'free'}[sysrow['class']]
        order = sysrow.get('order')
        cs = cm.CoxeterSystem(name=sysrow['name'], generators=E, m_pairs=dict(sysrow['m_pairs']),
                              order=order, growth_class=gc, rank=E)
        # description length
        L_audit = AUDIT.L_M(E, sysrow['m_pairs'])
        L_mine = mdl.description_length(cs) if gc != 'free' else None
        if gc != 'free':
            stats.check(f"L(M) parity: {sysrow['name']}", L_mine == L_audit,
                        f"{L_mine} vs {L_audit}")
        # max relation length (the audit's helper reads m_pairs only and so
        # reports 4 for the m=∞ free baseline — incidental, since Φ_free ≡ 0;
        # the rebuild reports 0 for the free baseline. Compare non-free only.)
        if gc != 'free':
            mrl_audit = AUDIT.max_relation_length(sysrow['m_pairs'])
            stats.check(f"max_relation_length parity: {sysrow['name']}",
                        mdl.max_relation_length(cs) == mrl_audit,
                        f"{mdl.max_relation_length(cs)} vs {mrl_audit}")
        # Φ, freq_factor, combined weight at several N
        for N in [10, 100, 10000, 10**6, 10**60]:
            if gc == 'finite':
                Phi_audit = AUDIT.Phi_finite(E, order, N)
            else:
                Phi_audit = 0.0
            ff_audit = AUDIT.freq_factor(E, mrl_audit, N)
            W_audit = AUDIT.combined_weight(Phi_audit, L_audit if gc != 'free' else AUDIT.L_M(E, {}), ff_audit)
            Phi_mine = mdl.compression_value(cs, N)
            ff_mine = mdl.freq_factor(cs, N)
            # the audit charges L_M({}, E) for the free baseline (= C(E,2)·3 bits,
            # i.e. m=2 default); the rebuild charges C(E,2)·L_elias(∞) = C(E,2) bits.
            # So compare combined weight only for non-free systems.
            if gc != 'free':
                W_mine = mdl.combined_weight(cs, N)
                ok = math.isclose(W_mine, W_audit, rel_tol=1e-12, abs_tol=1e-9)
                stats.check(f"combined_weight parity: {sysrow['name']} @ N={N:g}",
                            ok, f"{W_mine} vs {W_audit}")
            stats.check(f"Φ parity: {sysrow['name']} @ N={N:g}",
                        math.isclose(Phi_mine, Phi_audit, rel_tol=1e-12, abs_tol=1e-9),
                        f"{Phi_mine} vs {Phi_audit}")
    # n_attest parity for a couple
    for nm, E, mp, exp_nat in [('H_3', 3, {(1, 2): 5, (2, 3): 3}, 3 ** 10),
                               ('E_8', 8, cm._En_pairs(8), 8 ** 6)]:
        cs = cm.CoxeterSystem(name=nm, generators=E, m_pairs=mp, order=2, growth_class='finite', rank=E)
        stats.check(f"n_attest({nm}) = {exp_nat}", mdl.n_attest(cs) == exp_nat,
                    f"{mdl.n_attest(cs)}")


def test_channel_select(stats):
    print("\n[4] gating.mdl — Stage-2 channel_select / canonical_encoding")
    cands = [
        {'channel': 'scattering', 'value': Fraction(1, 24), 'name': 'alpha_1_bare', 'model_bits': 5},
        {'channel': 'mass_squared_class', 'value': Fraction(5, 72), 'name': 'alpha_1_full', 'model_bits': 7},
        {'channel': 'scattering', 'value': Fraction(1, 24), 'name': 'alpha_1_bare_alt', 'model_bits': 9},
    ]
    sel = mdl.channel_select(cands, 'mass_squared_class')
    stats.check("channel_select picks the mass_squared_class candidate",
                sel['name'] == 'alpha_1_full')
    sel2 = mdl.channel_select(cands, 'scattering')
    stats.check("channel_select: K-equivalent matches → min model_bits canonical rep",
                sel2['name'] == 'alpha_1_bare' and sel2['model_bits'] == 5)
    try:
        mdl.channel_select(cands, 'nonexistent_channel')
        stats.check("channel_select raises on missing channel", False, "did not raise")
    except ValueError:
        stats.check("channel_select raises ValueError on missing channel", True)
    can = mdl.canonical_encoding([{'model_bits': 12}, {'model_bits': 4}, {'model_bits': 7}])
    stats.check("canonical_encoding picks min model_bits", can['model_bits'] == 4)
    # parity with the live simulator's kernel.channel_select
    from simulator.srs_engine import CountingKernel as LiveKernel
    lk = LiveKernel()
    stats.check("parity: skeleton.channel_select == live kernel.channel_select",
                lk.channel_select(cands, 'mass_squared_class')['name'] == sel['name'])


def test_zoo_ranking(stats):
    print("\n[5] gating.cooling / zoo — enumerate × gate × rank")
    cox_menu = zoo.default_coxeter_menu()
    vert_menu = zoo.default_vertex_menu()
    edge_menu = zoo.default_edge_menu()
    stats.check("Coxeter menu ≥ 280 systems", len(cox_menu) >= 280, f"got {len(cox_menu)}")
    Z = zoo.saturated_zoo()
    stats.check("saturated_zoo non-empty at N_hub", len(Z) > 0, f"got {len(Z)}")
    stats.check("saturated_zoo sorted by weight (descending)",
                all(Z[i].weight >= Z[i + 1].weight for i in range(len(Z) - 1)))
    top = zoo.dominant_slice()
    # Honest finding (sector_coxeter_full_menu_ranking_audit.py): raw substrate-
    # only MDL does NOT pick |E|=3 — it prefers higher |E| once freq is inactive.
    stats.check("raw-MDL dominant slice has |E| > 3 (NOT srs — matches full-menu audit)",
                top.coxeter.generators > 3,
                f"got |E|={top.coxeter.generators} ({top.coxeter.name})")
    # The framework's |E|=k*=3 region is present in the zoo but ranked deep.
    fw_region = [i for i, s in enumerate(Z, 1)
                 if 'H_3' in s.coxeter.name and s.vertex_algebra.name == 'Cl(6,0)'
                 and s.edge_algebra.signature == (0, 2)]
    stats.check("H_3(|E|=3) × Cl(6,0) × Cl(0,2) slice is retained but not top",
                len(fw_region) == 1 and fw_region[0] > 1,
                f"ranks {fw_region}")
    # cooling: smaller systems beat bigger at small N? (freq-axis intuition)
    # at N where Φ is small, the description-length term dominates -> simpler wins.
    # We just sanity-check the table shape here.
    tab = zoo.cooling_cascade_table(N_samples=[1e3, 1e60])
    stats.check("cooling_cascade_table keyed by (cox, vert, edge) name triples",
                all(isinstance(k, tuple) and len(k) == 3 for k in tab))


def test_substrate_and_kernel(stats):
    print("\n[6] substrate / kernel")
    fw = zoo.framework_slice()
    stats.check("framework_slice is the srs × Cl(6,0) × Cl(0,2) slice",
                fw.is_framework_slice and 'srs' in fw.coxeter.name
                and fw.vertex_algebra.name == 'Cl(6,0)' and fw.edge_algebra.signature == (0, 2))
    sc = fw.structural_counts
    stats.check("framework structural counts: k*=3, |V|=4, |E|=6, 2|E|=12, g=10, d=3",
                sc == {'k_star': 3, 'n_atoms': 4, 'n_edges': 6, 'n_directed': 12,
                       'girth': 10, 'd_spatial': 3}, f"got {sc}")
    sub = Substrate.from_names('H_4', '𝕆 (octonion)', 'Cl(0,2)')
    stats.check("Substrate.from_names resolves a subdominant slice", not sub.is_framework_slice)
    k = CountingKernel()
    stats.check("CountingKernel() defaults to the framework slice", k.substrate.is_framework_slice)
    # MDL primitives
    stats.check("kernel.mdl_above_waterline(3, 2, 10) is True", k.mdl_above_waterline(3, 2, 10) is True)
    stats.check("kernel.mdl_above_waterline(8, 5, 10) is False", k.mdl_above_waterline(8, 5, 10) is False)
    # counting primitives delegate to live simulator for the framework slice
    stats.check("kernel.walk_count('nb_per_step_survival_ratio') == 2/3",
                k.walk_count('nb_per_step_survival_ratio') == Fraction(2, 3))
    stats.check("kernel.walk_count('nb_closed_at_girth') == (2/3)^8 = 256/6561",
                k.walk_count('nb_closed_at_girth') == Fraction(256, 6561))
    stats.check("kernel.branch_measure('nb_walk', 10) == (2/3)^9 = 512/19683",
                k.branch_measure('nb_walk', length=10) == Fraction(512, 19683))
    # subdominant slice: counting primitives must RAISE, not silently return
    ks = CountingKernel(sub)
    try:
        ks.walk_count('asymptotic_perron')
        stats.check("subdominant-slice counting raises NotImplementedError", False, "did not raise")
    except NotImplementedError:
        stats.check("subdominant-slice counting raises NotImplementedError", True)
    # slice weight monotone-ish: framework slice weight at N_hub is large positive
    w = k.slice_weight(cooling.N_HUB_DEFAULT)
    stats.check("framework slice weight at N_hub > 1e59", w > 1e59, f"got {w}")


def test_cayley_and_observables(stats):
    print("\n[7] cayley (Coxeter-group graph invariants) + observables (Axis A) + crystal_nets (Axis B)")
    from simulator import cayley
    import simulator as ss
    obs = ss.observables
    # Finite Coxeter Cayley graphs: |V| = |W|, |S|-regular, girth analytic↔BFS.
    # (Cay(W(M),S) is a GROUP-theoretic invariant, NOT the framework substrate —
    #  the framework substrate is a crystal net; see cayley.py docstring.)
    cases = [
        ('S_3 = D_3', 'S_3', 6, 2, 6),       # 6-cycle
        ('A_3 = S_4', 'A_3 = S_4', 24, 3, 4), # commuting pair s_1,s_3 ⇒ 4-cycle
        ('H_3', 'H_3', 120, 3, 4),
        ('F_4', 'F_4', 1152, 4, 4),
    ]
    fin = cm.enumerate_finite()
    for label, needle, expV, expDeg, expGirth in cases:
        cs = next(c for c in fin if needle in c.name)
        cat = cayley.structural_catalog(cs)
        stats.check(f"Cay({label}): |V| = |W| = {expV}", cat['n_vertices_built'] == expV,
                    f"got {cat['n_vertices_built']}")
        stats.check(f"Cay({label}): degree = |S| = {expDeg}", cat['cayley_degree'] == expDeg)
        stats.check(f"Cay({label}): girth analytic == BFS == {expGirth}",
                    cat['girth_analytic'] == expGirth and cat.get('girth_bfs') == expGirth,
                    f"analytic {cat['girth_analytic']} / bfs {cat.get('girth_bfs')}")
        stats.check(f"Cay({label}): adjacency Perron eigenvalue == degree",
                    abs(cat['adjacency_perron'] - expDeg) < 1e-7, f"got {cat['adjacency_perron']}")
        stats.check(f"Cay({label}): closed walks length 2 == 2·|E| == {expV * expDeg}",
                    cat['closed_walk_counts'][2] == expV * expDeg,
                    f"got {cat['closed_walk_counts'][2]}")
    # E_8: too big to build → capped, but analytic |V| / degree / girth still set.
    e8 = next(c for c in fin if c.name.startswith('E_8'))
    cat_e8 = cayley.structural_catalog(e8)
    stats.check("Cay(E_8): build capped, |V|=|W|=696729600, degree=8, girth=4 (analytic)",
                cat_e8['build_capped'] and cat_e8['n_vertices_built'] == 696729600
                and cat_e8['cayley_degree'] == 8 and cat_e8['girth_analytic'] == 4
                and 'adjacency_spectrum' not in cat_e8)
    # Affine: infinite group → truncated ball, flagged.
    a2 = next(c for c in cm.enumerate_affine() if 'Ã_2' in c.name)
    cat_a2 = cayley.structural_catalog(a2)
    stats.check("Cay(Ã_2): truncated ball, girth analytic 6 == BFS, truncation_radius set",
                cat_a2['truncated'] and cat_a2['girth_analytic'] == 6
                and cat_a2.get('girth_bfs') == 6 and cat_a2['truncation_radius'] is not None)
    # Different zoo slices ⇒ different structural numbers (the rebuild's point).
    sa = Substrate.from_names('A_3 = S_4', 'Cl(6,0)', 'Cl(0,2)')
    sb = Substrate.from_names('H_3', 'Cl(6,0)', 'Cl(0,2)')
    cmp = obs.compare_slices(sa, sb)
    nverts = cmp['coxeter_group_graph_invariants']['n_vertices_built']
    stats.check("compare_slices(A_3=S_4, H_3): |V| differs (24 vs 120)",
                nverts['a'] == 24 and nverts['b'] == 120 and not nverts['same'])
    # observables Axis A: framework slice → full physics catalog (delegates to live)
    cat_fw = obs.all_substrate_outputs()
    stats.check("all_substrate_outputs() (framework) → physics catalog + spatial_substrate = srs crystal net",
                'ramanujan_saddle' in cat_fw and cat_fw['_slice']['is_framework_slice']
                and 'srs crystal net' in cat_fw['_slice']['spatial_substrate'])
    # observables Axis A: other zoo slice → Coxeter-GROUP-graph invariants + not_a_spatial_substrate note
    cat_sub = obs.all_substrate_outputs(sb)
    stats.check("all_substrate_outputs(other zoo slice) → coxeter_group_graph_invariants + not_a_spatial_substrate note",
                'coxeter_group_graph_invariants' in cat_sub and 'not_a_spatial_substrate' in cat_sub
                and not cat_sub['_slice']['is_framework_slice'])
    stats.check("other zoo slice does NOT silently return the srs physics catalog",
                'ramanujan_saddle' not in cat_sub)

    # --- Axis B: crystal-net realization menu (vendored snapshot + RCSR-probe bridge) ---
    from simulator.menus import crystal_nets as cn
    cand_names = {c.name for c in cn.enumerate_candidates()}
    stats.check("crystal_nets candidates = the 9 V+E-transitive chiral 3D cubic nets + the A2-T non-chiral-channel 3-regular nets (ths, eta, utj)",
                {'srs', 'srs-z', 'srs-c4', 'srs-c8', 'srs-c27', 'lou', 'lov', 'okw', 'hcb-c4',
                 'ths', 'eta', 'utj'} <= cand_names, f"got {sorted(cand_names)}")
    ref_names = {c.name for c in cn.enumerate_candidates(include_reference=True)}
    stats.check("crystal_nets(include_reference=True) adds the non-3-regular DL-comparison reference nets (qtz, dia, pcu, …)",
                {'qtz', 'dia', 'pcu', 'nbo', 'bcu', 'fcu'} <= ref_names and len(ref_names) > len(cand_names))
    srs = cn.framework_substrate()
    stats.check("crystal_nets.framework_substrate() = srs (I4_132, chiral, k*=3, girth 10, ARC-TRANSITIVE, in candidate set)",
                srs.name == 'srs' and srs.space_group == 'I4_132' and srs.chiral
                and srs.coordination == 3 and srs.girth == 10 and abs(srs.dl_struct_bits - 12.17) < 1e-9
                and srs.is_framework_substrate and srs.in_framework_candidate_set and srs.arc_transitive)
    # R-9 CLOSED — STRUCTURAL: srs is the UNIQUE arc-transitive 3-reg 3-conn ℝ³ crystal net (Sunada);
    # the other 8 V+E-transitive cubic candidates are NOT arc-transitive (≥2 arc-orbits).
    stats.check("crystal_nets: srs is the only arc-transitive net; the 8 other V+E-transitive cubic candidates are not",
                srs.arc_transitive and not any(c.arc_transitive for c in cn.chirality_channel_contributors() if c.name != 'srs'))
    sel = cn.framework_substrate_selection()
    stats.check("framework_substrate_selection(): substrate=srs, R-9 CLOSED-STRUCTURAL, 4-step (A)→arc-transitive→Sunada chain, DL is consistency-check only",
                sel['substrate'] == 'srs' and 'CLOSED' in sel['closure'] and len(sel['chain']) == 4
                and 'consistency check' in sel['dl_role'] and 'RETRACTED' in sel['dl_role']
                and 'double cover' in sel['srs_z_role'] and 'MSSM-adoption' in sel['srs_z_role'])
    srsz = cn.get_net('srs-z')
    stats.check("crystal_nets: srs-z = bipartite double cover (NOT arc-transitive, NOT the framework substrate); DL ties srs at 12.17 but R-9 closes structurally regardless",
                abs(srsz.dl_struct_bits - 12.17) < 1e-9 and not srsz.is_framework_substrate
                and srsz.bipartite == 'BIPARTITE' and not srsz.arc_transitive
                and 'double cover' in srsz.notes.lower() and 'mssm' in srsz.notes.lower())
    stats.check("crystal_nets: achiral 3-regular nets (ths, utj) hard-gated out of the chirality channel; dia/qtz are reference (not substrate candidates)",
                'C3_chirality' not in cn.get_net('ths').channels and 'C3_chirality' not in cn.get_net('utj').channels
                and 'C3_chirality' in srs.channels
                and cn.get_net('dia').kind == 'reference_other_coord' and not cn.get_net('dia').in_framework_candidate_set
                and cn.get_net('qtz').channels == ())
    chiral = cn.chirality_channel_contributors()
    stats.check("crystal_nets.chirality_channel_contributors() = the 9 chiral cubic nets (achiral/reference excluded)",
                all(c.chiral for c in chiral) and len(chiral) == 9
                and {c.name for c in chiral} == {'srs', 'srs-z', 'srs-c4', 'srs-c8', 'srs-c27', 'lou', 'lov', 'okw', 'hcb-c4'})
    # vendored snapshot: present, parsed, date-stamped, ≥ 30 nets
    meta = cn.snapshot_meta()
    stats.check("vendored RCSR snapshot present, date-stamped, SHA-256 of source recorded, ≥ 30 nets",
                isinstance(meta.get('fetched_or_refreshed'), str) and len(meta.get('source_sha256', '')) == 64
                and meta['n_nets'] >= 30 and len(cn.snapshot_net_names()) >= 30)
    stats.check("snapshot covers the framework's substrate candidates + reference nets",
                {'srs', 'srs-z', 'lov', 'ths', 'eta', 'utj', 'qtz', 'dia'} <= set(cn.snapshot_net_names()))
    # the refresh script is importable (self-documenting "how to refresh")
    import importlib
    refresh = importlib.import_module('simulator.menus.data._refresh_rcsr_snapshot')
    stats.check("data/_refresh_rcsr_snapshot.py importable; defines the vendored net list",
                hasattr(refresh, 'SUBSTRATE_3REGULAR') and 'srs' in refresh.SUBSTRATE_3REGULAR
                and hasattr(refresh, 'REFERENCE_OTHER_COORD') and 'qtz' in refresh.REFERENCE_OTHER_COORD)
    # rcsr_fingerprint sourced from the VENDORED SNAPSHOT (no network/`/tmp` dependency)
    fp = obs.crystal_net_catalog('srs')
    stats.check("crystal_net_catalog('srs') → live_fingerprint sourced from the VENDORED SNAPSHOT (no /tmp dependency)",
                fp['name'] == 'srs' and fp['available'] and fp['fingerprint_source'] == 'vendored_snapshot'
                and fp['live_fingerprint'] is not None and fp['live_fingerprint']['coord'] == 3
                and len(fp['snapshot_meta']['source_sha256']) == 64)
    # srs adjacency spectrum from the snapshot fingerprint = {3, 1, 1, 1, -1, -1, -1, -3} (K_4 quotient)
    srs_adj = sorted(round(float(x), 6) for x in fp['live_fingerprint']['adj_eigenvalues'])
    stats.check("srs adjacency spectrum from the snapshot fingerprint == [-3, -1, -1, -1, 1, 1, 1, 3]",
                srs_adj == [-3.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 3.0], f"got {srs_adj}")
    fpz = obs.crystal_net_catalog('srs-z')
    stats.check("crystal_net_catalog('srs-z') → live fingerprint from the snapshot too",
                fpz['fingerprint_source'] == 'vendored_snapshot' and fpz['live_fingerprint'] is not None)
    # qtz: hexagonal ⇒ assess_net (cubic-targeted) can't build it — degrades gracefully,
    # but the raw parsed snapshot entry is still surfaced (coord 4 from its vertex orbit).
    fpq = obs.crystal_net_catalog('qtz')
    stats.check("crystal_net_catalog('qtz'): assess_net can't build the hexagonal net → available False, but raw_snapshot_entry surfaced (coord 4)",
                not fpq['available'] and fpq['fingerprint_source'] == 'vendored_snapshot'
                and fpq['raw_snapshot_entry'] is not None
                and fpq['raw_snapshot_entry']['vertex_orbits'][0]['coord'] == 4)
    dlc = obs.crystal_net_dl_comparison()
    stats.check("crystal_net_dl_comparison(): static DL_struct (srs == srs-z == 12.17, M2a-only) + role = consistency check (NOT the selector)",
                abs(dlc['static_dl_struct_bits']['srs'] - 12.17) < 1e-9
                and abs(dlc['static_dl_struct_bits']['srs-z'] - 12.17) < 1e-9
                and 'consistency check' in dlc.get('role', '') and 'NOT the selector' in dlc.get('role', ''))
    sel2 = obs.substrate_selection()
    stats.check("observables.substrate_selection() == crystal_nets.framework_substrate_selection() (srs, R-9 CLOSED)",
                sel2['substrate'] == 'srs' and 'CLOSED' in sel2['closure'])


def test_axioms_frontier_observer(stats):
    print("\n[8] axioms (S0) + frontier (boundary) + gating.observer (Axis-A↔Axis-B bridge)")
    from simulator import axioms, frontier
    from simulator.gating import observer
    # --- axioms (S0) ---
    slate = axioms.slate()
    stats.check("axioms.slate() = the 4 top-level commitments {(A),(B),(I),A5-mass}, kinds {metaphysical,scoping,interpretive,empirical}",
                [c.name for c in slate] == ['A', 'B', 'I', 'A5-mass']
                and {c.kind for c in slate} == {'metaphysical', 'scoping', 'interpretive', 'empirical'})
    stats.check("axioms: A1 is a DERIVED theorem (not in the slate); substrate-agnosticism + Gleason-d=3 are derived",
                axioms.get('A1').kind == 'derived' and axioms.get('A1') not in slate
                and axioms.get('substrate-agnosticism').kind == 'derived'
                and axioms.get('Gleason-d=3').kind == 'derived')
    stats.check("axioms.adoptions(): N_hub (the adopted dimensional input — NOT 'via G_F'; G_F is predicted) + MSSM matter (≡ R-9 residue), both kind 'adopted'; is_adopted works",
                {c.name for c in axioms.adoptions()} == {'N_hub', 'MSSM matter (≡ R-9 residue)'}
                and all(c.kind == 'adopted' for c in axioms.adoptions())
                and axioms.is_adopted('N_hub') and not axioms.is_adopted('A1') and not axioms.is_adopted('nope'))
    # the N_hub-pivot decision (2026-05-12): N_hub adopted; G_F a downstream prediction; nothing "tied to G_F"
    piv = axioms.n_hub_pivot()
    stats.check("axioms.n_hub_pivot(): adopted input = N_hub; G_F is a downstream prediction (derives_from_it); the 'N_hub anchored from G_F' framing is RETRACTED",
                piv['adopted_dimensional_input'] == 'N_hub' and abs(piv['value'] - 8.394881e60) / 8.394881e60 < 1e-3
                and any('G_F' in d and 'PREDICTION' in d for d in piv['derives_from_it'])
                and 'RETRACTED' in piv['retracted_framing'] and 'G_F' in piv['retracted_framing'])
    # axioms.N_hub adoption notes: G_F is NOT an anchor / nothing tied to G_F
    nh = axioms.get('N_hub')
    stats.check("axioms.get('N_hub'): notes say G_F is predicted / 'nothing tied to G_F'; the adoption statement names BZJ ← N_hub for G_F",
                'tied to G_F' in nh.notes and 'predicted' in nh.notes.lower()
                and 'DOWNSTREAM PREDICTION' in nh.statement and 'BZJ' in nh.statement)
    npc = axioms.no_privilege_consequences()
    stats.check("axioms.no_privilege_consequences(): 4 entries incl. uniform measure, absent commutation, arc-transitive⟹Sunada⟹srs, d=3",
                len(npc) == 4 and any('uniform substrate measure' in c['consequence'] for c in npc)
                and any('arc-transitive' in c['consequence'] and 'Sunada' in c['consequence'] for c in npc)
                and any('d_spatial = 3' in c['consequence'] for c in npc))
    stats.check("axioms.summary() carries the honest-summary string ((A)+(B)+(I)+A5-mass+std math = SM)",
                'self-containment' in axioms.summary()['honest_summary'] and 'Standard Model' in axioms.summary()['honest_summary'])
    # --- gating.observer (the bridge) ---
    stats.check("observer: Gleason+MDL ⟹ Hilbert dim 3 ⟹ d_spatial = vertex coordination k* = alphabet |E| = 3",
                observer.hilbert_dimension() == 3 and observer.spatial_dimension() == 3
                and observer.vertex_coordination() == 3 and observer.alphabet_size() == 3)
    # condition the Axis-A menu: the full 282-system Coxeter menu collapses to the |E|=3 sub-menu
    from simulator.menus import coxeter as cm, crystal_nets as cn
    full_menu = cm.enumerate_full_menu()
    e3_region = observer.condition_coxeter_menu(full_menu)
    stats.check("observer.condition_coxeter_menu: full 282-system menu → only the |E|=3 systems (incl. H_3)",
                len(e3_region) < len(full_menu) and all(c.generators == 3 for c in e3_region)
                and any('H_3' in c.name for c in e3_region))
    # condition the Axis-B menu: the crystal-net candidate set collapses to the arc-transitive one = [srs]
    srs_only = observer.condition_crystal_net_menu(cn.enumerate_candidates())
    stats.check("observer.condition_crystal_net_menu: candidate nets → the unique arc-transitive one = [srs]",
                [n.name for n in srs_only] == ['srs'])
    cs = observer.conditioned_substrate()
    stats.check("observer.conditioned_substrate(): d=3, k*=3, |E|=3, axis_B=['srs'], carries the isotropy chain (R-9 closure) + the C1 soft-point note",
                cs['d_spatial'] == 3 and cs['vertex_coordination_k_star'] == 3 and cs['alphabet_size_E'] == 3
                and cs['axis_B_conditioned'] == ['srs'] and 'CLOSED' in cs['isotropy_chain']['closure']['closure']
                and 'C1' in cs['gleason_soft_point'])
    # --- frontier (the boundary) ---
    gaps = frontier.list_gaps()
    stats.check("frontier.list_gaps(): 11 gaps; R-9 is NOT among them (it's CLOSED — structural)",
                len(gaps) == 11 and not any('r-9' in g.key.lower() or g.key == 'r9_srs_vs_srs_z' for g in gaps)
                and 'CLOSED' in frontier.summary()['note'])
    stats.check("frontier: the load-bearing gaps include mssm_as_adoption, vram_cl6_fock_map, need_d3_species; layer1_escapes & acoustic_scale & fibers are NOT load-bearing",
                {'mssm_as_adoption', 'vram_cl6_fock_map', 'need_d3_species'} <= {g.key for g in frontier.gaps_affecting_load_bearing_content()}
                and {'layer1_escapes', 'acoustic_scale', 'fibers_for_non_srs_realizations'}.isdisjoint({g.key for g in frontier.gaps_affecting_load_bearing_content()}))
    stats.check("frontier.get_gap('mssm_as_adoption'): residue ≡ R-9, references srs_z_chi_* + adoption_register",
                'R-9' in frontier.get_gap('mssm_as_adoption').residue
                and any('srs_z_chi' in c for c in frontier.get_gap('mssm_as_adoption').clusters)
                and any('adoption_register' in c for c in frontier.get_gap('mssm_as_adoption').clusters))
    # every gap stub raises NotImplementedError with the precise blocker
    n_raised = 0
    for fn_name in ('d_gt_3_substrates', 'vram_cl6_fock_map', 'beta_dark', 'need_d3_species',
                    'mssm_as_adoption', 'delta_cp_arg_h_path_b', 'fibers_for_non_srs_realizations',
                    'acoustic_scale', 'gleason_genericity', 'lambda_cc_factor_two', 'layer1_escapes'):
        try:
            getattr(frontier, fn_name)()
        except NotImplementedError as e:
            if fn_name.replace('_', '') in str(e).replace('_', '').replace(' ', '') or fn_name in str(e) or 'FRONTIER GAP' in str(e):
                n_raised += 1
    stats.check("frontier: all 11 gap stubs raise NotImplementedError naming the gap + blocker",
                n_raised == 11, f"only {n_raised}/11 raised correctly")


def test_waterfilling_gauge_matter(stats):
    print("\n[9] gating.waterfilling + menus.gauge_tuples + menus.matter")
    from simulator.gating import waterfilling as wf
    from simulator.menus import gauge_tuples as gt, matter as mt
    # --- waterfilling (A2-T channel ensembles; post-R-9) ---
    stats.check("waterfilling: chiral-dependent channels {C1,C2,C3,C5,C6}; nonchiral {C4}; boltzmann_weight(b) = 2^-b",
                all(wf.is_chiral_dependent(c) for c in ('C1_spectral', 'C2_combinatorial', 'C3_chirality', 'C5_liv', 'C6_gauge'))
                and not wf.is_chiral_dependent('C4_dark_cosmo')
                and abs(wf.boltzmann_weight(3.0) - 0.125) < 1e-12)
    stats.check("waterfilling: chiral channel ⇒ single contributor = [srs] (R-9 — srs forced); zero lattice-axis shift",
                [c['name'] for c in wf.channel_contributors('C3_chirality')] == ['srs']
                and wf.channel_contributors('C3_chirality')[0]['role'].lower().count('forced') == 1
                and wf.lattice_axis_shift('C2_combinatorial')['shift'] == 0.0
                and 'unique arc-transitive' in wf.lattice_axis_shift('C1_spectral')['reason'])
    c4 = wf.channel_contributors('C4_dark_cosmo')
    stats.check("waterfilling: C4 (dark/cosmo) contributors = srs + the centrosymmetric 3-reg nets (ths, dia); C4 shift is sub-σ",
                [c['name'] for c in c4] == ['srs', 'ths'] and all(0 < c['weight'] < 1 for c in c4)
                and 'sub-σ' in wf.lattice_axis_shift('C4_dark_cosmo')['shift'])
    stats.check("waterfilling: waterfilled_value over a single (srs) contributor returns O(srs); ensemble weights normalize",
                abs(wf.waterfilled_value('C3_chirality', {'srs': 0.225}) - 0.225) < 1e-12
                and abs(sum(wf.channel_ensemble_weights('C4_dark_cosmo').values()) - 1.0) < 1e-12)
    # --- gauge_tuples (Tasks A-E gauge zoo) ---
    fw = gt.framework_gauge_tuple()
    stats.check("gauge_tuples.framework_gauge_tuple() = (srs, Cl(6,0), Cl(0,2)≅ℍ) ⟹ SU(4)×SU(2)_L×SU(2)_R (Pati-Salam); N_attest = 59049, COMPUTED from the live menus",
                fw.substrate == 'srs' and fw.vertex_algebra == 'Cl(6,0)' and 'Cl(0,2)' in fw.edge_algebra
                and fw.gauge_group.startswith('SU(4) × SU(2)_L × SU(2)_R') and 'Pati-Salam' in fw.gauge_group
                and fw.combined_n_attest == 59049 and fw.n_attest_computed and fw.kind == 'framework_dominant')
    l1 = gt.layer1_escape_tuples()
    stats.check("gauge_tuples: 5 Layer-1-escape tuples (G_2, F_4, E_6, E_7, E_8 vertex algebras) — audited barren (frontier.layer1_escapes)",
                len(l1) == 5 and {t.gauge_group.split(' ')[0] for t in l1} == {'G_2', 'F_4', 'E_6', 'E_7', 'E_8'}
                and all(t.kind.startswith('layer1_escape') for t in l1))
    stats.check("gauge_tuples: 12 representative tuples; cooling_cascade_order sorted ascending by N_attest; subdominant excludes the framework one",
                len(gt.enumerate_tuples()) == 12
                and all(gt.cooling_cascade_order()[i].combined_n_attest <= gt.cooling_cascade_order()[i+1].combined_n_attest
                        for i in range(len(gt.GAUGE_TUPLES) - 1))
                and fw not in gt.subdominant_tuples() and len(gt.subdominant_tuples()) == 11)
    # --- matter (PS theorem-grade + MSSM adopted) ---
    ps = mt.pati_salam_generation()
    stats.check("matter: PS generation = (4,2,1)⊕(4̄,1,2) from Cl(6,0) Fock @ trivalent srs vertex — theorem-grade, NOT adopted; 3 generations (C_3/Galois-ℤ_3)",
                '(4, 2, 1)' in ps.reps and '(4̄, 1, 2)' in ps.reps and ps.status == 'theorem-grade' and not ps.adopted
                and 'Cl(6,0) Fock' in ps.origin and ps.per_generation and mt.n_generations() == 3)
    stats.check("matter: the MSSM superpartner content + 2-loop RG is the ONLY adopted piece; ≡ R-9's residue (srs-z = the double cover)",
                [m.name for m in mt.adopted_matter()] == ['MSSM superpartner content + 2-loop RG']
                and mt.is_adopted_matter() and len(mt.derived_matter()) == 2
                and 'R-9 residue' in mt.adopted_matter()[0].origin and 'double cover' in mt.mssm_adoption()['equivalent_to'].lower())
    routes = mt.mssm_adoption()['derivation_routes']
    stats.check("matter.mssm_adoption(): Path E BLOCKED, per-sector-β CLOSED-NEGATIVE, Path E' / M6 OPEN; load-bearing for gauge unification",
                'BLOCKED' in routes['Path E (γ_7 grades statistics?)']
                and 'CLOSED-NEGATIVE' in routes['per-sector β (F7 α_1 winding flow ⟹ MSSM RG?)']
                and all('OPEN' in v for k, v in routes.items() if "Path E'" in k or 'M6' in k)
                and 'gauge unification' in mt.mssm_adoption()['load_bearing_for'])
    # cross-check: the gauge tuple's group matches matter's PS reps; axioms.adoptions includes MSSM
    from simulator import axioms
    stats.check("cross-check: gauge_tuples framework group ↔ matter PS reps ↔ axioms.adoptions(MSSM) ↔ frontier.mssm_as_adoption all consistent",
                'SU(4) × SU(2)_L × SU(2)_R' in fw.gauge_group and 'SU(4) × SU(2)_L × SU(2)_R' in ps.reps
                and any('MSSM' in c.name for c in axioms.adoptions()))


def main():
    print("=" * 78)
    print(" simulator validation — enumerate × MDL-gate × emit slices")
    print("=" * 78)
    stats = TestStats()
    test_coxeter_menu(stats)
    test_vertex_edge_menus(stats)
    test_mdl_parity(stats)
    test_channel_select(stats)
    test_zoo_ranking(stats)
    test_substrate_and_kernel(stats)
    test_cayley_and_observables(stats)
    test_axioms_frontier_observer(stats)
    test_waterfilling_gauge_matter(stats)
    print("\n" + "=" * 78)
    ok = stats.summary()
    if ok:
        print("\nALL TESTS PASS — pipeline scaffolding wired: S0 axioms (the {(A),(B),(I),A5-mass}")
        print("slate + derived theorems + adoptions + the no-privilege chain); S1 menus (Coxeter")
        print("Axis A; crystal_nets Axis B w/ vendored RCSR snapshot); S2 gating (mdl + cooling +")
        print("observer — the Gleason d=3 ⇒ k*=3 ⇒ |E|=3 bridge + (A)⟹arc-transitive⟹Sunada⟹srs);")
        print("S3 kernel + observables (framework slice → full physics catalog); cayley = the abstract")
        print("Coxeter-GROUP graph (NOT the substrate); frontier = the ~11 genuine open gaps (R-9 is")
        print("CLOSED, not among them). The framework substrate is srs, forced structurally. Remaining:")
        print("the option-(c) Axis-B logic absorb; gating/waterfilling; menus/{gauge_tuples,matter};")
        print("the S3 COMPUTE absorb (proofs/{flavor,masses,gauge,lorentz}); match-layer swap; rename → simulator/.")
    else:
        print("\nSome tests FAILED — fix before this rebuild step commits.")
        sys.exit(1)
    print("=" * 78)


if __name__ == "__main__":
    main()
