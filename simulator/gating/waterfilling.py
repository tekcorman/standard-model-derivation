"""
S2 — A2-T Boltzmann-weighted channel ensembles ("waterfilling").

A2-T retains EVERY encoding with L(M) + L(data|M) < L(raw), plurally weighted
by compression savings. For an observable O, the prediction is, in principle, a
Boltzmann-weighted sum over the above-waterline ensemble that contributes to O's
*channel*:

    O_pred = Σ_C  w(C) · O(C) · 𝟙[C ∈ channel(O)]  /  Z_channel(O),
    w(C) = 2^{-DL_struct(C)},   Z_channel(O) = Σ_C w(C) · 𝟙[C ∈ channel(O)].

Channels (an internal working note §2a):
  C1 Spectral · C2 Combinatorial · C3 Chirality · C4 Dark/Cosmo · C5 LIV · C6 Gauge.

⚠️ POST-R-9 (2026-05-12). R-9 closing changes what "ensemble" means here:

  - The *substrate net* is NOT a channel-select ensemble — it is the
    MDL-minimum HYPOTHESIS (full DL_model + DL_data; `theorem_substrate_agnosticism.md`),
    and it is forced STRUCTURALLY to be srs ((A) ⟹ no privileged direction ⟹
    arc-transitive ⟹ Sunada ⟹ srs; see `axioms.no_privilege_consequences()` /
    `crystal_nets.framework_substrate_selection()`). So the "naive Boltzmann
    ensemble over whole-substrate hypotheses breaks PDG" finding (program-doc
    §(l)) used the wrong object; under the right one, w(srs-z)/w(srs) ≈ 0 and
    w(any other arc-transitive 3-reg ℝ³ net) = 0 (there is no other).

  - So for the CHIRAL channels (anything requiring a chiral substrate — C3, and
    the chirality-derived structural factors that feed C1/C2: V_us, η_B, the
    fermion identifications, β cosmic birefringence, the dark-correction sign,
    …) the substrate ensemble has a SINGLE member, srs. O_pred = O(srs).
    ZERO lattice-axis shift, uniformly. (Pre-R-9 this was "srs dominates the
    Boltzmann sum"; post-R-9 it's "srs is forced; no sum.")

  - The only place "waterfilling" remains NONTRIVIAL is the C4 dark/cosmo
    channel — the dark-sector buildup: the dim-count partition of F_inv(E)'s
    word ensemble into visible (reduced words) + dark (cancellable strings)
    (`predictions/H_multiway_dim_count.py` ⟹ Ω_DM/Ω_m), plus the d>3 substrate
    placeholders (R-4/R-5 — `frontier.d_gt_3_substrates`; not RCSR 3D nets, so
    NOT enumerated here) and the centrosymmetric 3-reg nets (ths, dia) which DO
    contribute to C4 (no chirality referenced). Those contribute at tiny
    Boltzmann weight (2^{-14} … 2^{-19}); only Ω_DM/Ω_m has been computed
    quantitatively: +0.002 shift, below the 1.6-1.9% PDG sensitivity. The
    actual numbers live in S3 / `cosmology.py` (absorbing `proofs/cosmology/*`
    + `substrate_lattice_waterfilling_omega_dm.py`); this module is the
    *gating-side* machinery (which substrates contribute, with what weight).

Probes this absorbs / references: `substrate_lattice_waterfilling_batch.py`,
`substrate_lattice_waterfilling_{omega_dm, v_us, v_cb, R_nu}.py`,
`beta_c1_waterfilling_audit.py`, an internal working note.
"""

import math
from typing import Optional


# Channels (mirrors menus.crystal_nets.CHANNELS).
CHANNELS = ('C1_spectral', 'C2_combinatorial', 'C3_chirality',
            'C4_dark_cosmo', 'C5_liv', 'C6_gauge')

# Which channels REQUIRE a chiral substrate (or chirality-derived structural
# factors). Per the program doc §2a: C3 obviously; and C1/C2 observables that
# use chirality-derived structure (V_us = 9/40 uses g·N_atoms on the chiral
# srs; the fermion identifications; etc.) — in practice, since srs is the
# unique arc-transitive 3-reg ℝ³ net (R-9), ALL of C1/C2/C3/C5/C6 collapse to
# the srs-only value. C4 (dark/cosmo) is the exception.
_CHIRAL_DEPENDENT = ('C1_spectral', 'C2_combinatorial', 'C3_chirality', 'C5_liv', 'C6_gauge')
_NONCHIRAL = ('C4_dark_cosmo',)


def boltzmann_weight(dl_struct_bits: float) -> float:
    """w(C) = 2^{-DL_struct(C)}."""
    return 2.0 ** (-dl_struct_bits)


def is_chiral_dependent(channel: str) -> bool:
    """True iff the channel's observables require a chiral substrate (⟹ srs only, R-9)."""
    if channel not in CHANNELS:
        raise ValueError(f"waterfilling: unknown channel {channel!r}; have {CHANNELS}")
    return channel in _CHIRAL_DEPENDENT


def channel_contributors(channel: str) -> list[dict]:
    """The substrate-realization members that contribute to `channel`, with weights.

    Returns list of {name, dl_struct_bits, weight, role}. For chiral-dependent
    channels: [srs] only (R-9 — srs is the unique arc-transitive 3-reg ℝ³ net,
    forced). For C4 (dark/cosmo): srs (dominant) + the centrosymmetric 3-reg
    nets ths/dia (R-7/R-8 — they DON'T carry chirality but DO contribute to C4)
    at small weight + a note that the d>3 substrates (R-4/R-5) and the
    dim-count dark partition contribute too but are NOT enumerated here
    (`frontier.d_gt_3_substrates`; the dark partition is in cosmology.py).
    """
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from simulator.menus import crystal_nets

    srs = crystal_nets.get_net('srs')
    srs_entry = {'name': 'srs', 'dl_struct_bits': srs.dl_struct_bits,
                 'weight': boltzmann_weight(srs.dl_struct_bits), 'role': 'the substrate (forced — R-9)'}
    if is_chiral_dependent(channel):
        return [srs_entry]
    # C4 dark/cosmo — non-chiral; centrosymmetric 3-reg nets contribute too.
    out = [srs_entry]
    for nm in ('ths', 'dia'):
        try:
            c = crystal_nets.get_net(nm)
            if c.dl_struct_bits is not None:
                out.append({'name': nm, 'dl_struct_bits': c.dl_struct_bits,
                            'weight': boltzmann_weight(c.dl_struct_bits),
                            'role': f'centrosymmetric 3-reg net (R-{7 if nm == "ths" else 8}); '
                                    'C4 only (chirality-gated elsewhere)'})
        except ValueError:
            pass
    return out


def channel_ensemble_weights(channel: str) -> dict:
    """Normalized Boltzmann weights {name: w/Z} over the channel's contributors."""
    contribs = channel_contributors(channel)
    Z = sum(c['weight'] for c in contribs)
    return {c['name']: (c['weight'] / Z if Z > 0 else 0.0) for c in contribs}


def waterfilled_value(channel: str, per_realization_value: dict) -> float:
    """O_pred = Σ w(C)/Z · O(C) over the channel's contributors.

    `per_realization_value` maps realization name → O(C). For chiral-dependent
    channels this is just O(srs); for C4 it's the (tiny-shift) weighted sum.
    Missing realizations are dropped from both numerator and Z.
    """
    contribs = [c for c in channel_contributors(channel) if c['name'] in per_realization_value]
    if not contribs:
        raise ValueError(f"waterfilled_value: no per-realization values supplied for any "
                         f"contributor of {channel!r} ({[c['name'] for c in channel_contributors(channel)]})")
    Z = sum(c['weight'] for c in contribs)
    return sum(c['weight'] * per_realization_value[c['name']] for c in contribs) / Z


def lattice_axis_shift(channel: str) -> dict:
    """Summary of the substrate-lattice-axis Boltzmann shift for `channel`.

    Chiral-dependent channels: ZERO shift (srs is the unique contributor,
    forced — R-9). C4: a tiny shift from the centrosymmetric / d>3 / dim-count-
    dark contributions — only Ω_DM/Ω_m computed quantitatively (+0.002, below
    the ~1.6-1.9% PDG sensitivity). See `proofs/foundations/substrate_lattice_waterfilling_*.py`
    + an internal working note.
    """
    if is_chiral_dependent(channel):
        return {'channel': channel, 'shift': 0.0, 'reason': (
            'srs is the unique arc-transitive 3-reg ℝ³ crystal net (R-9 closure: '
            '(A) ⟹ arc-transitive ⟹ Sunada ⟹ srs) ⟹ single ensemble member ⟹ '
            'O_pred = O(srs), no lattice-axis shift'), 'contributors': ['srs']}
    return {'channel': channel, 'shift': 'sub-σ (only Ω_DM/Ω_m computed: +0.002, below PDG sensitivity)',
            'reason': ('C4 is non-chiral ⇒ the centrosymmetric 3-reg nets (ths/dia) '
                       'and the d>3 substrate placeholders (R-4/R-5) and the dim-count '
                       'dark partition contribute at tiny Boltzmann weight (2^{-14}…2^{-19})'),
            'contributors': [c['name'] for c in channel_contributors(channel)] + ['R-4/R-5 (not enumerated — frontier.d_gt_3_substrates)', 'dim-count dark partition (cosmology.py)'],
            'numbers_in': 'simulator.cosmology / predictions/Omega_DM_over_Omega_m.py'}


def summary() -> dict:
    return {
        'channels': list(CHANNELS),
        'chiral_dependent_channels': list(_CHIRAL_DEPENDENT),
        'nonchiral_channels': list(_NONCHIRAL),
        'post_r9': ('the substrate is srs (forced — not a Boltzmann ensemble over '
                    'whole-substrate hypotheses); chiral-dependent channels ⇒ single '
                    'contributor ⇒ zero lattice-axis shift; only C4 has a (sub-σ) shift'),
        'shifts': {ch: lattice_axis_shift(ch) for ch in CHANNELS},
        'absorbs': ['substrate_lattice_waterfilling_batch.py', 'substrate_lattice_waterfilling_{omega_dm,v_us,v_cb,R_nu}.py',
                    'beta_c1_waterfilling_audit.py', 'an internal working note'],
    }
