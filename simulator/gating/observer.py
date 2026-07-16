"""
S2 — observer-side conditioning: the bridge that turns the substrate-only menus
into the framework's substrate.

Two distinct MDL stages (`gating/mdl.py` + `gating/cooling.py`) gate the
*substrate-only* menus (Axis A: Coxeter quotients; Axis B: crystal-net
realizations). But the framework's substrate is fixed by ALSO conditioning on
two observer-side facts — both consequences of axiom (A)'s no-privilege
principle (`axioms.no_privilege_consequences()`), not extra inputs:

  1. **d_spatial = 3** — Gleason 1957 (Born-rule frame functions are unique iff
     Hilbert dim ≥ 3) + the MDL minimum-cost viable dimension ⟹ the observer's
     Hilbert space is 3-dimensional ⟹ d_spatial = 3 ⟹ vertex coordination
     k* = 3 ⟹ the substrate alphabet |E| = 3. This **collapses Axis A's
     high-|E| raw-MDL argmax onto the |E| = 3 region** (`sector_coxeter_full_menu_ranking_audit.py`'s
     "k*=3 is observer-side" finding, made operational). Probes:
     `sector_C1_gleason_genericity_audit.py` (the C1 soft point — see
     `frontier.gleason_genericity`), `observer_hilbert_space_construction.py`,
     `theorem8_penrose_kolmogorov_resolution.py` + `theorem8_per_class_scaling_verification.py`,
     `predictions/observer_dim_three_derivation.md` / `predictions/k_star_derivation.md`.

  2. **Strong isotropy (arc-transitivity)** — the walker's causal state is a
     directed edge (Shalizi-Crutchfield 2001); a directionless observer must
     treat all directed edges as equivalent ⟹ the model is arc-transitive ⟹
     (substrate-agnosticism) the substrate is arc-transitive ⟹ (Sunada 2012)
     **srs is selected among the |E| = 3 ℝ³ crystal nets by STRONG ISOTROPY**
     (full local S₃ stabilizer), the R-9 closure's actual discriminator — NOT
     bare arc-transitivity: srs-z is ALSO arc-transitive (one arc orbit,
     edge-reversible) but has only a C₃ stabilizer, so it is filtered by
     strong isotropy, not by this stage (R-9 SUPERSESSION, corrected
     2026-06-15/07-10; see `docs/audits/registers/structural_residue_
     register.md` ~line 160). This **collapses Axis B onto srs**. Front-end:
     `walker_dynamics_derivation.md` Step 4b + `g_girth_derivation.md` Step 2;
     chain object: `menus.crystal_nets.framework_substrate_selection()`.

So `gating/observer.py` is the connector: `apply(coxeter_menu, crystal_net_menu)`
takes the substrate-only menus and returns the observer-conditioned slice
(|E| = 3 Coxeter region; srs on Axis B). The Layer-1-escape audits M1-M7
(`M{1,3,4,7}_*`, `theorem9_f3_quantification_on_srs.py`) live on this side too —
they ask whether any subdominant slice survives observer conditioning to produce
observed physics; the answer (NEGATIVE/UNCONNECTED via every audited channel) is
recorded in `frontier.layer1_escapes`.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Stage: Hilbert dimension / d_spatial / k* via Gleason + MDL
# ---------------------------------------------------------------------------

GLEASON_MIN_DIM = 3   # Gleason 1957: Born-rule frame functions unique iff dim ≥ 3


def hilbert_dimension(max_n: int = 8) -> int:
    """The observer's Hilbert-space dimension, by Gleason 1957 + MDL minimum cost.

    Candidate dimension n: model cost = n² − 1 (density-matrix free parameters
    on ℂⁿ); viable iff n ≥ GLEASON_MIN_DIM (for n < 3 frame functions are
    non-unique ⟹ no canonical Born-rule extension ⟹ unbounded waterline
    penalty). MDL picks the minimum-cost viable candidate ⟹ 3. Mirrors
    `simulator.kernel.CountingKernel.mdl_select_hilbert_dimension` (the first
    mechanical per-observable MDL invocation; previously `d_spatial` prose-argued
    d=3 then returned 3 hardcoded).
    """
    candidates = [{'n': n, 'model_bits': n * n - 1, 'viable': n >= GLEASON_MIN_DIM}
                  for n in range(1, max_n)]
    viable = [c for c in candidates if c['viable']]
    if not viable:
        raise ValueError("observer.hilbert_dimension: no viable candidate ≤ max_n")
    return min(viable, key=lambda c: c['model_bits'])['n']


def spatial_dimension() -> int:
    """d_spatial = 3. The observer's Hilbert dim is 3 (Gleason + MDL); the
    framework's d_spatial = Hilbert dim (`predictions/d_spatial_derivation.md`).
    """
    return hilbert_dimension()


def vertex_coordination() -> int:
    """k* = 3 — the substrate's per-vertex coordination. Tied to d_spatial = 3
    via the crystal-net constraint (`predictions/k_star_derivation.md`:
    toggle-arity / 3-regular). This is the Coxeter alphabet size |E| on Axis A.
    """
    return spatial_dimension()


def alphabet_size() -> int:
    """|E| = k* = 3 — the number of involutive generators of F_inv(|E|)."""
    return vertex_coordination()


# ---------------------------------------------------------------------------
# Stage: strong isotropy ⟹ srs (the R-9 closure front-end)
# ---------------------------------------------------------------------------

def isotropy_requirement() -> dict:
    """The arc-transitivity requirement on the substrate model — (A)-derived.

    Returns the chain (delegates to `axioms.no_privilege_consequences()` for the
    (A)→arc-transitivity step and `menus.crystal_nets.framework_substrate_selection()`
    for the arc-transitivity→Sunada→srs step).
    """
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from simulator import axioms
    from simulator.menus import crystal_nets
    np_chain = next((c for c in axioms.no_privilege_consequences()
                     if 'arc-transitive' in c['consequence']), None)
    return {
        'requirement': 'the substrate model is strongly isotropic (arc-transitive)',
        'from': '(A) no-privilege applied to spatial labels — derived, not adopted',
        'no_privilege_step': np_chain,
        'closure': crystal_nets.framework_substrate_selection(),
    }


# ---------------------------------------------------------------------------
# The connector: apply observer conditioning to the substrate-only menus
# ---------------------------------------------------------------------------

def condition_coxeter_menu(coxeter_menu: list) -> list:
    """Collapse the Axis-A Coxeter-quotient menu onto |E| = k* = 3.

    The substrate-only raw MDL on the full menu prefers high |E| (the "skeptical
    bridge" finding); observer conditioning (d_spatial = 3 ⟹ k* = 3 ⟹ |E| = 3)
    restricts to the |E| = 3 sub-menu, within which the framework's relation
    structure (the H_3-region / srs-equivalent system) sits.
    """
    k = alphabet_size()
    return [c for c in coxeter_menu if getattr(c, 'generators', None) == k]


def condition_crystal_net_menu(crystal_net_menu: list) -> list:
    """Collapse the Axis-B crystal-net menu onto the arc-transitive ones.

    Among the k* = 3 ℝ³ crystal nets, srs AND srs-z are both arc-transitive
    (R-9 SUPERSESSION, corrected 2026-06-15/07-10 — see
    `docs/audits/registers/structural_residue_register.md` ~line 160; the
    stronger, framework-selecting discriminator is STRONG ISOTROPY, which srs
    alone satisfies — see `isotropy_requirement()`/`framework_substrate_
    selection()`). `crystal_net_menu` entries are `menus.crystal_nets.
    CrystalNet`; returns the ones with `arc_transitive` and
    `coordination == k*` (= [srs, srs-z] — bare arc-transitivity alone does
    NOT collapse to srs; that collapse needs the strong-isotropy stage below).
    """
    k = vertex_coordination()
    return [c for c in crystal_net_menu
            if getattr(c, 'arc_transitive', False) and getattr(c, 'coordination', None) == k]


def conditioned_substrate() -> dict:
    """The observer-conditioned substrate: (|E| = 3 Coxeter region, srs crystal net).

    The single function that ties Axis A and Axis B together via the two
    observer-side facts. Returns the d/k*/|E| trio + the srs identification +
    the chain references.
    """
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from simulator.menus import coxeter, crystal_nets
    cox_region = condition_coxeter_menu(coxeter.enumerate_full_menu())
    nets = condition_crystal_net_menu(crystal_nets.enumerate_candidates())
    return {
        'd_spatial': spatial_dimension(),
        'vertex_coordination_k_star': vertex_coordination(),
        'alphabet_size_E': alphabet_size(),
        'hilbert_dim': hilbert_dimension(),
        'axis_A_conditioned_region': f'{len(cox_region)} Coxeter systems with |E| = {alphabet_size()} '
                                     f'(incl. the H_3-region / srs-equivalent system)',
        'axis_B_conditioned': [n.name for n in nets],   # → ['srs', 'srs-z'] (bare arc-transitivity;
                                                         # srs alone is picked out by strong isotropy,
                                                         # see isotropy_chain below)
        'gleason_soft_point': 'C1 genericity — see frontier.gleason_genericity',
        'isotropy_chain': isotropy_requirement(),
        'note': ('this is observer-side conditioning, not an extra input — both '
                 'd=3 (Gleason+MDL) and arc-transitivity (no-privilege) follow '
                 'from the slate; see axioms.no_privilege_consequences().'),
    }
