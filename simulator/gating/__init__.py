"""
simulator.gating — MDL waterline + frequency support + cooling cascade + channel selection.

TWO-STAGE gating (post-2026-05-12 MDL cleanup):

  Stage 1 — WATERLINE THRESHOLD (mdl.above_waterline / cooling.retained_at).
    For any menu of substrate-slice candidates (Coxeter × vertex_alg ×
    edge_alg), or — at the prediction layer — physical realizations of an
    observable, A2-T retains the subset clearing the waterline at observation
    length N:

      W(M, N) = Φ(M, N) − L(M) + min(freq_factor(M, N), 0)  ≥  threshold

    where Φ is compression value, L is description length, and freq_factor
    penalizes models whose rarest defining relation isn't yet attested in N
    samples. ALL retained candidates are PHYSICALLY REALIZED — no single
    minimum-cost winner is picked at this stage.

  Stage 2 — CHANNEL SELECT (mdl.channel_select / kernel.channel_select).
    For ONE specific observable, which retained candidate it reads is fixed
    by a STRUCTURAL argument (the observable's substrate definition) — a
    `channel` string fixed BEFORE candidates are enumerated. channel_select
    picks the matching one; if several K-equivalently match, the min-bit-cost
    canonical representative. This is the WATERFILLING-CORRECT selection.
    The retired `mdl_select` (argmin over total bit cost) wrongly collapsed
    Stages 1+2 into a global minimum — RETRACTED 2026-05 per
    feedback_waterline_not_minimum_canonical_distinction.

Submodules:
- mdl:          L(M), Φ(M, N), freq_factor, combined_weight, N_attest (Stage 1)
                + channel_select, canonical_encoding (Stage 2 wrappers)
- cooling:      retention-vs-N profile, saturated-zoo enumeration (Stage 1)
- observer:     observer-side conditioning — Gleason d=3 ⇒ k*=3 ⇒ |E|=3 (the
                Axis-A↔Axis-B bridge) + (A)'s no-privilege ⇒ arc-transitive ⇒
                Sunada ⇒ srs (the R-9 closure front-end). NOT a substrate-only
                gate — it consumes the substrate-only menus and returns the
                framework's conditioned slice.
- waterfilling: A2-T Boltzmann-weighted channel ensembles. Post-R-9: degenerate
                for chiral-dependent channels (srs forced ⇒ single contributor);
                nontrivial only for C4 dark/cosmo (sub-σ shift from ths/dia + the
                d>3 placeholders + the dim-count dark partition).
- associativity: F(E)-associativity gate (NA-4 Phase 1). Classifies closed
                predictions as F(E)-associative vs substrate-Layer-1 based on
                which load-bearing primitives their derivation chain uses.
                The catalog is permanently useful as scoping infrastructure
                — every future closure attempt can check whether it depends
                on associativity, independent of whether Need-D-3 / R-15
                ever close.
"""

from . import (mdl, cooling, observer, waterfilling, spectral_consistency,
               delta_b_match, associativity)
