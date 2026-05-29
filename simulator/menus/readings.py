"""
Reading-class + walk-class enumeration.

Per the unification consolidation doc
`docs/forward_constructions/forward_construction_one_B_many_readings.md`:

  Reading classes (R1-R7) — observable's substrate-side QN selects which:
    R1 amplitude     Im[Σ] coupling, √5/4·α₁ (off-diagonal C₃ obs)
    R2 mass²         Im²/Re², 5/3·α₁ (mass-mixing diag)
    R3 edge-local    bare 1·α₁ (C₃-symmetric vertex, Tr σ_x = 0)
    R4 h-functional  sin/cos/arg of h (direct walker phase)
    R5 Born / comb   |amp|² or count ratios (combinatorial)
    R6 character     trace ratios, polytope dihedrals
    R7 Bloch-Taylor  k² / k⁴ dispersion coefficients

  Walk classes (W1-W10) — which class of NB walks contributes:
    W1 (n_fixed=2)   NB closed cycle, in/out pinned
    W2 (n_fixed=0)   NB closed cycle, fully closed loop
    W3 (n_fixed=1)   NB transition, one pinned edge
    W4 (geom series) Σ over windings
    W5 (multi-cycle) Σ over Hashimoto host topology
    W6 (site-stab)   k* indistinguishable edge slots
    W7 (coupling-pair) k*² per girth cycle
    W8 (Cl(6) Fock slots) 2^k* × k*
    W9 (Hashimoto modes) 2|E|
    W10 (cycle space marginal) 2(|E|-|V|)+1

CHANNEL LABELS (post-2026-05-12 channel_select migration). Each
(reading-class, walk-class) combination — together with the fiber and
mode-pair — defines a `channel` string. At the prediction layer (match
package), an observable's substrate definition fixes its channel string;
`kernel.channel_select(candidates, channel)` picks the candidate matching
it from the above-waterline set. Examples of channel strings already in
use across match/sm_predictions/:
  'scattering' (n_fixed=2 / W1)            — α₁_bare
  'mass_squared_class' (R2)                — α₁_full, y_τ, λ_H
  'dark_class_3_edge_local' (R3)           — θ_12 / θ_13 PMNS
  'mssm_one_loop_beta_running'             — g_1/g_2/g_3/α_s/α_EM
  'k4_minus_eigenspace_dihedral' (R6)      — δ_CP_CKM
  'walker_phase_winding_n_eq_g' (R4 / W4)  — α_21 PMNS
  'dirac_cone_at_p' / 'dirac_cone_at_gamma' — v_F_P / v_F_Γ
  'minimum_spanning_coordination'          — k* = 3
  'edge_transitive_3d_3reg_3conn_crystal_net' — g = 10 (srs Sunada uniqueness)

NB: this module is enum only. The classes (and channel labels) apply
universally across the substrate zoo; what changes per slice is which
numerical values they produce.
"""

from enum import Enum


class ReadingClass(str, Enum):
    AMPLITUDE       = 'R1'  # Im[Σ], √5/4·α₁ — channel: dark_class_1_amplitude
    MASS_SQUARED    = 'R2'  # Im²/Re², 5/3·α₁ — channel: dark_class_2_mass_squared
    EDGE_LOCAL      = 'R3'  # 1·α₁ — channel: dark_class_3_edge_local
    H_FUNCTIONAL    = 'R4'  # sin/cos/arg of h — channel: walker_phase_*
    BORN_COMBINATORIAL = 'R5'  # |amp|², count ratios — channel: combinatorial / spectral
    CHARACTER       = 'R6'  # traces, polytope dihedrals — channel: *_eigenspace_dihedral, z3_holonomy, *
    BLOCH_TAYLOR    = 'R7'  # k², k⁴ coefficients — channel: scalar_bloch_taylor_order_*


class WalkClass(str, Enum):
    NB_CLOSED_PINNED_2  = 'W1'   # n_fixed=2: (k-1/k)^(g-2) — channel: scattering
    NB_CLOSED_LOOP      = 'W2'   # n_fixed=0: (k-1/k)^g — channel: self_energy
    NB_TRANSITION_1     = 'W3'   # n_fixed=1: (k-1/k)^(g-1) — channel: transition
    NB_GEOMETRIC_SUM    = 'W4'   # Σ_n (k-1/k)^(8n) — channel: nb_geometric_sum
    MULTI_CYCLE_HOST    = 'W5'   # Σ_{m≥2} multi-host — channel: multi_cycle_host_sum
    SITE_STABILIZER     = 'W6'   # 1/k* — channel: site_stabilizer_orbit
    COUPLING_PAIR       = 'W7'   # k*² per girth cycle — channel: coupling_pair_per_girth_cycle
    CL6_FOCK_SLOTS      = 'W8'   # 2^k* × k* — channel: cl6_fock_label_slots
    HASHIMOTO_MODES     = 'W9'   # 2|E| — channel: hashimoto_modes
    CYCLE_SPACE_MARGINAL = 'W10' # 2(|E|-|V|)+1 — channel: cycle_space_marginal_modes
