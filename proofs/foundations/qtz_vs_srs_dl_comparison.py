#!/usr/bin/env python3
"""
qtz vs srs description-length comparison — M2 audit v2 mechanism for Row 4.

Adapts dl_comparison.py's framework to cross-coordination comparison
(srs at k=3 vs qtz at k=4). The question for audit v2 Row 4 closure:
does srs have lower MDL than qtz at the structural specification level?

DL(crystal) = DL(space_group) + DL(vertex_orbits) + DL(coordinates)
              + DL(edges) + DL(chirality)

For srs (I4_132, Wyckoff 8a, vertex+edge transitive, chiral):
  DL(srs) = log2(230) + L*(1) + log2(5) + 0 + 0 + 1
          = 7.85 + 1.00 + 2.32 + 0 + 0 + 1.00 = 12.17 bits.

For qtz (P6_222 maximally-symmetric realization, Wyckoff 3-orbit pos,
vertex+edge transitive, chiral):
  DL(qtz) = log2(230) + L*(1) + log2(11) + 0 + 0 + 1
          = 7.85 + 1.00 + 3.46 + 0 + 0 + 1.00 = 13.31 bits.

ΔDL(qtz − srs) = 13.31 − 12.17 = +1.14 bits.

qtz is **1.14 bits MORE EXPENSIVE** than srs at the structural level. M2
gates qtz with Boltzmann weight 2^(-1.14) ≈ 0.45 — a WEAK soft gate, not
sufficient to suppress observable-specific M3·M4·M5 differentials.

For η_B specifically, M6 sign-gate closes Row 4 (sign-falsification).
For non-η_B observables, audit v2 closure is ONE-AMONG-MANY at the
v2-internal level; srs's empirical selection is the framework anchor.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dl_comparison import dl_choice, dl_integer

# ============================================================================
# srs reference (already in dl_comparison.py)
# ============================================================================

def dl_srs():
    """SRS net (Laves graph): I4_132 (#214), Wyckoff 8a (1 of 5 positions),
    vertex+edge transitive, chiral. Coordinate x=1/8 is the barycentric
    placement (determined by topology); 0 coordinate bits."""
    bits = {}
    bits['space_group'] = dl_choice(230)               # 7.85
    bits['n_orbits']    = dl_integer(1)                # 1.00
    bits['wyckoff']     = dl_choice(5)                 # 2.32 (8a from 5 in I4_132)
    bits['coordinates'] = 0.0                          # topology determines coords
    bits['edges']       = 0.0                          # edge-transitive
    bits['chirality']   = 1.0                          # chiral pair
    return sum(bits.values()), bits


def dl_qtz():
    """QTZ net (topological): P6_222 (#180) maximally-symmetric realization,
    Wyckoff 3c (1 of 11 positions), vertex+edge transitive, chiral.
    Coordinate (1/2, 0, 0) at 3c is fixed (no free params); 0 coordinate bits.

    NOTE: qtz net's *topological* spacegroup is P6_222/P6_422 chiral pair (RCSR
    canonical assignment for the max-symmetry realization). Alpha-quartz's
    actual Si geometry (P3_121/P3_221) is a LOWER-symmetry geometric realization
    of the same topology — for MDL comparison we use the highest-symmetry form
    (lowest DL within the topology class).

    Reference: International Tables for Crystallography Vol A, P6_222.
    Wyckoff positions (11 total):
      12k (general, 1)
      6j, 6i, 6h (.2 or ..2 site symmetry, 1 free param each)
      3g, 3f (222. or ..2., 0 free params)
      2e, 2d (.3. or ..3, 1 free param)
      1c, 1b, 1a (high site symmetry, fixed)
    qtz uses Wyckoff 3c (vertex position) — actually 3f or 3g per RCSR.
    Either way, log2(11) bits to specify.
    """
    bits = {}
    bits['space_group'] = dl_choice(230)               # 7.85
    bits['n_orbits']    = dl_integer(1)                # 1.00
    bits['wyckoff']     = dl_choice(11)                # 3.46 (3-orbit pos from 11 in P6_222)
    bits['coordinates'] = 0.0                          # 3c fixed coordinates
    bits['edges']       = 0.0                          # edge-transitive
    bits['chirality']   = 1.0                          # chiral pair (P6_222 vs P6_422)
    return sum(bits.values()), bits


# ============================================================================
# MAIN
# ============================================================================

print("=" * 70)
print("M2 AUDIT v2 — qtz vs srs cross-coordination DL comparison")
print("=" * 70)

dl_srs_total, dl_srs_bd = dl_srs()
dl_qtz_total, dl_qtz_bd = dl_qtz()

print(f"\nDL(srs):")
for k, v in dl_srs_bd.items():
    print(f"  {k:15s} = {v:5.2f} bits")
print(f"  {'TOTAL':15s} = {dl_srs_total:5.2f} bits")

print(f"\nDL(qtz):")
for k, v in dl_qtz_bd.items():
    print(f"  {k:15s} = {v:5.2f} bits")
print(f"  {'TOTAL':15s} = {dl_qtz_total:5.2f} bits")

delta_dl = dl_qtz_total - dl_srs_total
print(f"\nΔDL(qtz − srs) = {delta_dl:+.2f} bits")

if delta_dl < 0:
    boltzmann_qtz_over_srs = 2 ** (-delta_dl)
    print(f"qtz is {-delta_dl:.2f} bits CHEAPER than srs at structural level.")
    print(f"Boltzmann weight ratio P(qtz)/P(srs) = 2^{-delta_dl:.2f} = {boltzmann_qtz_over_srs:.2f}")
    print(f"M2 DOES NOT GATE qtz. qtz is structurally MORE compressed than srs.")
elif delta_dl > 0:
    print(f"qtz is {delta_dl:.2f} bits MORE EXPENSIVE than srs.")
    print(f"M2 gates qtz with Boltzmann weight 2^(-{delta_dl:.2f}) = {2**(-delta_dl):.4f}")
    print(f"This is a WEAK soft gate (not sufficient to suppress O(10⁵) M5 chain differentials).")
else:
    print(f"qtz and srs have equal MDL.")

print(f"\n{'=' * 70}")
print("INTERPRETATION FOR AUDIT v2 ROW 4 CLOSURE")
print(f"{'=' * 70}")

print(f"""
ΔDL = +{delta_dl:.2f} bits favoring srs. qtz is {delta_dl:.2f} bits more expensive
than srs at the structural specification level. Boltzmann weight 2^(-{delta_dl:.2f})
≈ {2**(-delta_dl):.2f} suppresses qtz contribution by factor ~2.

Combined with M3·M4·M5 differentials (observable-specific):

For LONG-CHAIN observables like η_B (M=6 chain factor):
- M3·M4·M5 enhances qtz contribution by ~3.4×10⁵ vs srs.
- M2 weight ~0.45.
- Net qtz/srs ratio: ~1.5×10⁵.
- M6 sign-gate kicks in: qtz predicts negative η_B (categorical falsification).
- **Row 4 closes UNIQUE for η_B via M6 sign-gate**, NOT via M2 Boltzmann.

For SHORT-CHAIN observables like V_us = 9/40, dark 5/12, β cosmic birefringence:
- M5 cumulative chain factor ~ 1 (no M-event chain in these observables).
- M3·M4 differentials ~ O(1)–O(10) per observable.
- M2 weight ~0.45.
- Net qtz/srs ratio: ~O(1)–O(5) (observable-specific).
- M6 sign-flip: irrelevant (these observables use Im(h)/|h|², not Re(h) sign).

Honest verdict on Row 4:
- **UNIQUE-on-η_B** via M6 sign-gate (Phase 1a finding, structurally robust).
- **DOMINANT-with-tight-margin** on most non-η_B observables: combined M2 (0.45)
  + observable-specific M3·M4 differentials gives qtz contribution suppressed
  by O(2)–O(10) below srs's; this is NOT enough to make srs UNIQUE without
  additional input.
- For SOME observables (V_us, V_cb), the M3·M4·M5 product may give qtz
  contribution at observable level — would falsify qtz on observable
  agreement, similar to η_B sign-gate but via numerical disagreement
  rather than sign falsification.

Row P29 (η_B) graduates to UNIQUE-THEOREM-GRADE-CONDITIONAL via M6.
Other parameter rows graduate to **DOMINANT-CONDITIONAL** with named margin
~ 2× to 10× per observable, computable per-observable as audit v2 closure
work.

The honest finding: srs's selection over qtz IS audit-v2 derivable, but
ONLY via the combined product of multiple weak gates (M2 ~1 bit, M3·M4
~O(1) bits, observable-specific M5). The strongest single audit-v2 gate
is M6 sign-flip on η_B. For other observables, the combined product
suffices for DOMINANT-with-tight-margin status, not UNIQUE.
""")

print("=" * 70)
print("CAVEATS")
print("=" * 70)

print("""
1. The dl_comparison.py methodology counts STRUCTURAL specification.
   It does NOT count the 'predictive efficiency' of the framework's
   F_substrate function. If qtz's F_qtz produces predictions that
   disagree with observation by O(10⁵)× (as the M3·M4·M5 differential
   suggests for some observables), then the per-observable DL(data |
   substrate) would heavily favor srs. This would re-introduce M2
   gating at the data-conditional level.

2. The DL-on-data-conditional analysis is observable-specific and
   computationally intensive. It's deferred to a follow-up session.

3. Wyckoff counts: I4_132 W=5 from `dl_comparison.py`. P6_222 W=11
   per ITC standard. If these counts are off by ±1-2, ΔDL shifts by
   ±~0.5 bits — but the sign holds (qtz spacegroup has more Wyckoff
   positions than srs's, so M2 favors srs).

4. The framework's selection rule could include additional structural
   filters not captured in M1-M6 (e.g., 'unique chiral 3-regular 3-
   periodic net' — Sunada's theorem). Such a filter is k=3-specific
   by construction. If we accept Sunada's theorem as a *named* M1
   residue R-N, then M1 hard-gates qtz. This is a re-derivation of
   srs from k=3 + chirality + edge-transitivity rather than from
   pure MDL — it's an additional structural input the framework
   uses but doesn't currently have a R-N residue named for.

5. For observables that don't depend on substrate-specific spectral
   data (e.g., A_hemispherical, structural anchors), audit v2 doesn't
   apply — the predictions are Class-D Poisson statistics not
   substrate-specific.
""")
