#!/usr/bin/env python3
"""
A1 dark-correction propagation probe (2026-05-25).

User push: "we derived a theorem grade correction. why isn't that propagating?
does it help here?"

The theorem-grade correction is α_GUT_obs = α_GUT_bare × (1 - (1/k*)·α₁/(1-α₁))
= 1/24.329 (vs bare 1/24).

A1's current formula uses BARE quantities:
  M_unif = α_GUT_bare × α_1_bare × M_Pl                    (uses bare)
  N_GUT  = N_hub × (v_today/M_unif)^4                       (depends on M_unif)
  T_GUT  = M_unif × c_S × K_per_GeV       (c_S = 1/12 bare Perron-residue)
  T_today = T_GUT × √(N_GUT/N_hub)                          (α=1/2 propagation)

  → 2.954 K vs 2.725 K observed (+8.36%)

This probe tests several propagation choices:

  (P1) α_GUT_obs in M_unif only
  (P2) (1 - α₁/(1-α₁)) applied to T_GUT directly with c=1
  (P3) (1 - α₁/(1-α₁)) applied to T_GUT directly with c=1/k* (α_GUT's c_g)
  (P4) (1 - α₁/(1-α₁)) applied to T_today (i.e., to the propagation, c=1)
  (P5) (1 - α₁/(1-α₁)) applied TWICE — once at anchor, once at propagation

The honest answer: does any STRUCTURAL choice close the 8% residual, or does
the dark-correction template (with its known c_g values) not happen to fit
this observable?
"""

from __future__ import annotations
import math
from fractions import Fraction

# Framework primitives
k_B = 1.380649e-23
GeV = 1.602176634e-10
K_per_GeV = GeV / k_B

N_hub = 8.394881e60
v_today = 246.22
T_CMB_observed = 2.7255

# Substrate primitives
k_star = 3
N_atoms = 4
two_E = N_atoms * k_star    # = 12

# Bare values
alpha_GUT_bare = Fraction(1, 24)               # = 1/(2^k* × k*) substrate counting
alpha_1_bare = Fraction(2, 3) ** 8             # = 256/6561 NB walker survival
c_S_bare = Fraction(1, two_E)                  # = 1/12 Perron-residue singlet

# Theorem-grade dark correction
c_alpha_GUT = Fraction(1, k_star)              # = 1/3 from Routes H + C
waterline = alpha_1_bare / (1 - alpha_1_bare)  # = 256/6305 ≈ 0.04060
alpha_GUT_obs = alpha_GUT_bare * (1 - c_alpha_GUT * waterline)  # = 18659/453960 ≈ 1/24.329

# M_Pl in GeV (CODATA)
M_Pl_GeV = 1.220890e19

print("=" * 72)
print("A1 DARK-CORRECTION PROPAGATION PROBE")
print("=" * 72)

print(f"\nBare quantities:")
print(f"  α_GUT_bare    = {alpha_GUT_bare} = {float(alpha_GUT_bare):.8f}  →  1/α_GUT_bare = {1/float(alpha_GUT_bare):.4f}")
print(f"  α_1_bare      = (2/3)^8 = {float(alpha_1_bare):.8f}")
print(f"  c_S_bare      = 1/12 = {float(c_S_bare):.8f}")
print(f"  waterline     = α₁/(1-α₁) = {float(waterline):.8f}")

print(f"\nTheorem-grade dark correction (α_GUT):")
print(f"  c_α_GUT = 1/k* = {c_alpha_GUT}")
print(f"  α_GUT_obs = α_GUT_bare × (1 - (1/k*)·α₁/(1-α₁)) = {alpha_GUT_obs} ≈ 1/{1/float(alpha_GUT_obs):.4f}")
print(f"  Ratio α_GUT_obs / α_GUT_bare = {float(alpha_GUT_obs/alpha_GUT_bare):.6f}")


def compute_T_today(M_unif_GeV, c_S_value, dark_at_propagation=1.0):
    """T_today = (M_unif × c_S × K_per_GeV) × √(N_GUT/N_hub) × dark_at_propagation
       where N_GUT = N_hub × (v/M_unif)^4."""
    N_GUT = N_hub * (v_today / M_unif_GeV) ** 4
    T_GUT_K = M_unif_GeV * c_S_value * K_per_GeV
    return T_GUT_K * math.sqrt(N_GUT / N_hub) * dark_at_propagation


# ---------------------------------------------------------------------------
# Baseline (all bare, current corrected A1 framing)
# ---------------------------------------------------------------------------
M_unif_bare = float(alpha_GUT_bare * alpha_1_bare) * M_Pl_GeV
T_baseline = compute_T_today(M_unif_bare, float(c_S_bare))

print(f"\n{'='*72}")
print("BASELINE (all bare — current A1 corrected framing)")
print('='*72)
print(f"  M_unif_bare = {M_unif_bare:.4e} GeV")
print(f"  T_today = {T_baseline:.4f} K  vs observed {T_CMB_observed} K")
print(f"  Deviation = {(T_baseline-T_CMB_observed)/T_CMB_observed*100:+.2f}%")


# ---------------------------------------------------------------------------
# P1 — α_GUT_obs in M_unif only
# ---------------------------------------------------------------------------
M_unif_P1 = float(alpha_GUT_obs * alpha_1_bare) * M_Pl_GeV
T_P1 = compute_T_today(M_unif_P1, float(c_S_bare))

print(f"\n{'='*72}")
print("P1: α_GUT_obs propagated into M_unif (M_unif → M_unif × 24/24.329)")
print('='*72)
print(f"  M_unif_P1 = {M_unif_P1:.4e} GeV (was {M_unif_bare:.4e})")
print(f"  N_GUT shifts: was N_hub × (v/M_unif_bare)^4, now × (v/M_unif_P1)^4")
print(f"  T_today = {T_P1:.4f} K")
print(f"  Deviation = {(T_P1-T_CMB_observed)/T_CMB_observed*100:+.2f}%")
print(f"  Δ vs baseline = {(T_P1-T_baseline):.4f} K ({(T_P1-T_baseline)/T_baseline*100:+.2f}%)")
print(f"  NOTE: M_unif decreases by factor 0.9865, but N_GUT increases by 1.0556")
print(f"        (since N_GUT ∝ 1/M_unif^4); net effect: T_today INCREASES slightly")


# ---------------------------------------------------------------------------
# P2 — (1 - waterline) factor applied to T_GUT (c=1 at the anchor)
# ---------------------------------------------------------------------------
dark_factor_c1 = 1 - float(waterline)
T_P2 = compute_T_today(M_unif_bare, float(c_S_bare) * dark_factor_c1)

print(f"\n{'='*72}")
print("P2: dark factor (c=1) applied to c_S at anchor → c_S → c_S × (1 - α₁/(1-α₁))")
print('='*72)
print(f"  dark_factor = (1 - α₁/(1-α₁)) = {dark_factor_c1:.6f}")
print(f"  T_today = {T_P2:.4f} K")
print(f"  Deviation = {(T_P2-T_CMB_observed)/T_CMB_observed*100:+.2f}%")


# ---------------------------------------------------------------------------
# P3 — c=1/k* applied to c_S (matching α_GUT's c_g)
# ---------------------------------------------------------------------------
dark_factor_inv_kstar = 1 - float(c_alpha_GUT) * float(waterline)
T_P3 = compute_T_today(M_unif_bare, float(c_S_bare) * dark_factor_inv_kstar)

print(f"\n{'='*72}")
print("P3: c=1/k* applied to c_S (analog of α_GUT's c_g)")
print('='*72)
print(f"  dark_factor = (1 - (1/k*)·α₁/(1-α₁)) = {dark_factor_inv_kstar:.6f}")
print(f"  T_today = {T_P3:.4f} K")
print(f"  Deviation = {(T_P3-T_CMB_observed)/T_CMB_observed*100:+.2f}%")


# ---------------------------------------------------------------------------
# P4 — dark factor applied to propagation (T_today × (1-w))
# ---------------------------------------------------------------------------
T_P4 = compute_T_today(M_unif_bare, float(c_S_bare), dark_at_propagation=dark_factor_c1)

print(f"\n{'='*72}")
print("P4: dark factor (c=1) applied to T_today via propagation")
print('='*72)
print(f"  T_today = T_baseline × (1 - α₁/(1-α₁)) = {T_P4:.4f} K")
print(f"  Deviation = {(T_P4-T_CMB_observed)/T_CMB_observed*100:+.2f}%")


# ---------------------------------------------------------------------------
# P5 — dark factor applied TWICE (anchor + propagation)
# ---------------------------------------------------------------------------
T_P5 = compute_T_today(M_unif_bare, float(c_S_bare) * dark_factor_c1,
                       dark_at_propagation=dark_factor_c1)

print(f"\n{'='*72}")
print("P5: dark factor (c=1) applied TWICE — anchor + propagation")
print('='*72)
print(f"  T_today = T_baseline × (1 - α₁/(1-α₁))² = {T_P5:.4f} K")
print(f"  Deviation = {(T_P5-T_CMB_observed)/T_CMB_observed*100:+.2f}%")


# ---------------------------------------------------------------------------
# Required correction factor to close
# ---------------------------------------------------------------------------
required_factor = T_CMB_observed / T_baseline
required_c_if_single_anchor = (1 - required_factor) / float(waterline)
required_c_if_squared = (1 - math.sqrt(required_factor)) / float(waterline)

print(f"\n{'='*72}")
print("REQUIRED DARK CORRECTION TO CLOSE")
print('='*72)
print(f"  Need T_today × X = 2.7255 where T_baseline = {T_baseline:.4f}")
print(f"  X = {required_factor:.6f}  (or equivalently a -{(1-required_factor)*100:.2f}% factor)")
print(f"\n  If single application X = (1 - c·waterline):")
print(f"    → c = (1-X)/waterline = {required_c_if_single_anchor:.4f}")
print(f"  If squared application X = (1 - c·waterline)²:")
print(f"    → c = {required_c_if_squared:.4f}")
print(f"\n  Comparison with structural c_g values:")
print(f"    1/k* = 1/3 = 0.333")
print(f"    c_v = 5/12 = 0.417")
print(f"    1   = 1.000")
print(f"    2   = 2.000")
print(f"    N_atoms/k* = 4/3 = 1.333")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*72}")
print("SUMMARY")
print('='*72)
print(f"\n  Baseline (all bare):                    T_today = {T_baseline:.4f} K  ({(T_baseline-T_CMB_observed)/T_CMB_observed*100:+.2f}%)")
print(f"  P1 (α_GUT_obs in M_unif):               T_today = {T_P1:.4f} K  ({(T_P1-T_CMB_observed)/T_CMB_observed*100:+.2f}%)  ← slightly WORSE")
print(f"  P2 (c=1 on c_S at anchor):              T_today = {T_P2:.4f} K  ({(T_P2-T_CMB_observed)/T_CMB_observed*100:+.2f}%)")
print(f"  P3 (c=1/k* on c_S at anchor):           T_today = {T_P3:.4f} K  ({(T_P3-T_CMB_observed)/T_CMB_observed*100:+.2f}%)")
print(f"  P4 (c=1 on T_today propagation):        T_today = {T_P4:.4f} K  ({(T_P4-T_CMB_observed)/T_CMB_observed*100:+.2f}%)")
print(f"  P5 (c=1 applied TWICE):                 T_today = {T_P5:.4f} K  ({(T_P5-T_CMB_observed)/T_CMB_observed*100:+.2f}%)  ← CLOSEST")
print(f"\n  Observed CMB:                           T_CMB    = {T_CMB_observed} K")

print(f"""
FINDING:
  - P1 (the user's natural reading: propagate α_GUT_obs into M_unif) is
    NOT the answer. M_unif decreases by a factor 0.9865, but N_GUT
    increases by 1/0.9865^4 = 1.056, and √(N_GUT/N_hub) increases by 1.027.
    Net: T_today goes UP slightly (worse, not better).

  - P5 (c=1 dark correction applied at BOTH anchor and propagation)
    closes the 8% residual to ~-0.2%. This is structurally interesting:
    the form is T_today = M_unif × c_S × (1 - α₁/(1-α₁))² × K_per_GeV × √(N_GUT/N_hub).

  - The structural meaning of "c=1 applied twice": if the dark correction
    enters once when the substrate's Perron projector hits the GUT-energy
    reservoir (anchor) and once when the thermal-photon channel decoheres
    along propagation (each e-fold sees the dark waterline). The factor
    of (1-w)² = (1 - 2w + w²) ≈ 1 - 2w for small w explains why a
    single c=2 correction at anchor (P-something) gives nearly the same
    numerical result.

OPEN QUESTION: is the c=1-applied-twice STRUCTURAL, or pattern-hunting?
  - The first c=1 has a natural reading: the Perron projector at Γ on B_NB
    picks up the dark waterline as the substrate's surviving NB share
    (same waterline factor that appears in δ_r).
  - The second c=1 is the harder one: it would need to be a propagation-
    level statement about how the thermal scale picks up the dark waterline
    along each e-fold of horizon expansion. Not yet derived.

Before claiming this as a structural finding: would need an explicit
derivation of why the dark waterline enters the propagation, not just
the anchor.
""")
print("=" * 72)
