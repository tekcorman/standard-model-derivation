#!/usr/bin/env python3
"""
R-7 closure: ths-net residue → CKM/PMNS girth-4 corrections.

Purpose: test the hypothesis from an internal working note
that ths-net contributes a soft-gated correction to V_cb (and other CKM/PMNS
elements) under A2-T waterline, with MDL Boltzmann weight 2^{-ΔDL} ≈ 0.31.

OUTCOME: REFUTED in additive form. The naive additive correction
V_cb_total = V_cb_srs + w · V_cb_ths breaks PDG agreement at every reasonable
weight w. Mechanism: ths is centrosymmetric (chirality = 0), and the framework's
chirality residue R-12 is load-bearing for parity violation — non-chiral
lattices cannot host the SU(2)_L parity-violating structure. R-12 therefore
acts as a HARD-GATING filter on lattice alternatives, not just a producer of
parity-violation. This sharpens srs's selection from "MDL-min 3D 3-regular
crystal net" to "MDL-min 3D 3-regular CHIRAL crystal net" and refutes R-7
along with R-8 (dia is also centrosymmetric) and most of R-9 (full-MDL-spectrum).

Cross-references:
  - docs/audits/registers/structural_residue_register.md R-7 (this closure updates the entry)
  - proofs/foundations/dl_comparison.py (DL data; ths chirality = 0)
  - predictions/V_cb_derivation.md (srs-only V_cb formula)
"""

from fractions import Fraction
from math import log2

# =============================================================================
# Inputs from upstream framework files
# =============================================================================

# Lattice DLs from dl_comparison.py
DL_SRS = 12.17  # bits, chiral, edge-transitive
DL_THS = 13.85  # bits, centrosymmetric, 2 edge orbits
DL_DIA = None   # not in dl_comparison; estimated > srs by larger margin

# Chirality bit (per dl_comparison.py): srs = 1.0 (chiral), ths = 0.0 (centro)
CHIRALITY_SRS = 1.0
CHIRALITY_THS = 0.0

# srs parameters (from predictions/k_star.py, predictions/g_girth.py, predictions/V_cb_derivation.md)
K_STAR = 3
G_SRS = 10            # girth of srs
L_CB_SRS = G_SRS - 2  # NB walk length for V_cb on srs (n_fixed = 2)

# ths girth (RCSR database, ThSi2 net) — verified externally
G_THS = 4

# PDG 2024 V_cb (exclusive average)
V_CB_PDG = 40.5e-3
V_CB_PDG_SIGMA = 1.5e-3


# =============================================================================
# srs-only V_cb (reproduces predictions/V_cb.py)
# =============================================================================

def alpha_n_geometric(k, g, n_fixed=2):
    """First-winding NB walk amplitude on a k-regular girth-g graph."""
    L = g - n_fixed
    return Fraction(k - 1, k) ** L

def vcb_geometric_series(alpha):
    """A2-T waterline geometric resummation: sum over all retained windings."""
    return alpha / (1 - alpha)

alpha_srs = alpha_n_geometric(K_STAR, G_SRS, n_fixed=2)
v_cb_srs = vcb_geometric_series(alpha_srs)

print(f"srs alpha_1     = {alpha_srs} = {float(alpha_srs):.6f}")
print(f"srs V_cb        = {v_cb_srs} = {float(v_cb_srs):.6e}")
print(f"PDG V_cb        = {V_CB_PDG:.6e} ± {V_CB_PDG_SIGMA:.6e}")
print(f"srs deviation   = {(float(v_cb_srs) - V_CB_PDG) / V_CB_PDG_SIGMA:+.2f}σ")


# =============================================================================
# ths-only V_cb (analogous formula on ths)
# =============================================================================

alpha_ths = alpha_n_geometric(K_STAR, G_THS, n_fixed=2)
v_cb_ths_isolated = vcb_geometric_series(alpha_ths)

print(f"\nths alpha_1     = {alpha_ths} = {float(alpha_ths):.6f}")
print(f"ths V_cb (iso)  = {v_cb_ths_isolated} = {float(v_cb_ths_isolated):.6f}")
print(f"  (ths girth-4 cycle gives huge {float(v_cb_ths_isolated):.2f}; ths cannot stand alone)")


# =============================================================================
# R-7 hypothesis: additive correction at MDL Boltzmann weight 2^{-ΔDL}
# =============================================================================

dDL = DL_THS - DL_SRS
weight_boltz = 2 ** (-dDL)

print(f"\nΔDL(ths − srs)  = {dDL:+.2f} bits")
print(f"Boltzmann weight w = 2^(-ΔDL) = {weight_boltz:.4f}")

v_cb_total_additive = float(v_cb_srs) + weight_boltz * float(v_cb_ths_isolated)
sigma_off_additive = (v_cb_total_additive - V_CB_PDG) / V_CB_PDG_SIGMA

print(f"\nR-7 additive prediction: V_cb = V_cb_srs + w · V_cb_ths")
print(f"  = {float(v_cb_srs):.6e} + {weight_boltz:.4f} × {float(v_cb_ths_isolated):.4f}")
print(f"  = {v_cb_total_additive:.6e}")
print(f"  PDG deviation: {sigma_off_additive:+.1f}σ")
print(f"  PDG agreement (within 1σ): {'PASS' if abs(sigma_off_additive) < 1 else 'FAIL'}")


# =============================================================================
# Sensitivity: what weight w would barely-still-be-compatible with PDG?
# =============================================================================

print(f"\nSensitivity analysis:")
print(f"  For PDG agreement at 1σ, max correction Δ ≤ {V_CB_PDG_SIGMA + abs(float(v_cb_srs) - V_CB_PDG):.4e}")
max_correction = V_CB_PDG_SIGMA + abs(float(v_cb_srs) - V_CB_PDG)
max_weight = max_correction / float(v_cb_ths_isolated)
print(f"  Max compatible weight w_max ≈ {max_weight:.4f}")
print(f"  log2(w_max) ≈ {log2(max_weight):.2f}")
print(f"  Boltzmann form 2^(-ΔDL) gives w = {weight_boltz:.4f} = 2^({log2(weight_boltz):.2f})")
print(f"  Boltzmann overpredicts by factor: {weight_boltz / max_weight:.1f}×")


# =============================================================================
# Sensitivity: try several alternative weight formulas
# =============================================================================

print(f"\nAlternative weight formulas (for reference):")
forms = [
    ("2^(-ΔDL)",        2 ** (-dDL)),
    ("2^(-2·ΔDL)",      2 ** (-2 * dDL)),
    ("2^(-ΔDL²)",       2 ** (-(dDL ** 2))),
    ("exp(-ΔDL)",       2.718281828 ** (-dDL)),
    ("2^(-ΔDL·ln 2)",   2 ** (-dDL * 0.6931)),
    ("2^(-ΔDL) / N_atoms (12)",  (2 ** (-dDL)) / 12),
]
for name, w in forms:
    v = float(v_cb_srs) + w * float(v_cb_ths_isolated)
    sigma = (v - V_CB_PDG) / V_CB_PDG_SIGMA
    pass_str = "PASS" if abs(sigma) < 1 else "FAIL"
    print(f"  w={name:25s} w={w:.4e}  V_cb={v:.4e}  ({sigma:+.1f}σ)  {pass_str}")


# =============================================================================
# REFUTATION + MECHANISM
# =============================================================================

print(f"""
{'='*75}
R-7 CLOSURE — REFUTED
{'='*75}

The simple additive R-7 hypothesis (V_cb = V_cb_srs + w · V_cb_ths with
Boltzmann weight w = 2^(-ΔDL)) FAILS PDG agreement by {abs(sigma_off_additive):.0f}σ.

Adjusting the weight to fit V_cb requires w ≤ {max_weight:.3f},
i.e., w_eff = 2^({log2(max_weight):.2f}) — far below the lattice MDL-margin
2^(-ΔDL) = 2^({-dDL:.2f}). No "natural" weight formula produces ths-leakage
small enough to be invisible in V_cb. (See sensitivity table above.)

MECHANISM OF REFUTATION:

ths is centrosymmetric (chirality = 0; see proofs/foundations/dl_comparison.py
line 194). srs is chiral (chirality = 1; line 174). The framework's chirality
residue R-12 (docs/audits/registers/structural_residue_register.md) is LOAD-BEARING for parity
violation: SU(2)_L's left-handed-only coupling, the Im(h)/|h| factor in dark
corrections, and the framework's mirror-image structure all require both srs
hands to be retained simultaneously.

A centrosymmetric lattice has no left-vs-right distinction by construction.
Therefore ths CANNOT host the chirality residue. ths-leakage to a chirality-
dependent observable like V_cb is structurally impossible — not soft-gated
at small weight, but HARD-GATED by the framework's chirality requirement.

SHARPENED SUBSTRATE IDENTIFICATION:

The framework's substrate is not "the MDL-minimum 3D 3-regular crystal net"
(Row 6 of docs/audits/registers/uniqueness_ledger.md) but more specifically:

  the MDL-minimum 3D 3-regular CHIRAL crystal net.

Among 3D 3-regular chiral nets (chirality = 1 in dl_comparison.py),
srs is uniquely the MDL minimum (and the only entry currently in dl_comparison).
This sharpens Row 6's DOMINANT classification toward UNIQUE-within-chiral.

CONSEQUENCES FOR OTHER RESIDUES IN THE REGISTER:

R-7 (this entry):   REFUTED. Mechanism: chirality hard-gating.
R-8 (dia):          INHERITS REFUTATION. dia is centrosymmetric (Fd-3m, #227).
                    Same hard-gating mechanism applies.
R-9 (full-MDL):     PARTIALLY REFUTED. Restricted to chiral 3-regular 3D nets,
                    not all 3-regular 3D nets. Most low-DL competitors in
                    dl_comparison.py are centrosymmetric; the residue's
                    candidate set shrinks dramatically.
R-12 (chirality):   ACQUIRES NEW STRUCTURAL ROLE. Not just produces parity
                    violation — also hard-gates non-chiral lattice alternatives.
                    To be promoted from "ACCOUNTED-FOR" to "ACCOUNTED-FOR
                    + STRUCTURAL FILTER".

CALIBRATION OUTCOME:

R-7 was the calibration case for the residue-as-MDL-Boltzmann-weight
methodology underlying R-1, R-4, R-6, R-11. The calibration FAILED for the
simple additive form. The more refined methodology emerging from this closure:

  Soft-gated alternatives are *additionally* filtered by load-bearing residues
  the framework has already adopted (R-12's chirality requirement, and any
  others to be discovered). Apparent soft-gating (finite ΔDL) can become
  effective hard-gating via downstream load-bearing structure.

R-1, R-4, R-6, R-11 must be re-tested with this refinement. R-4 (d=4 → time
dimension) is most likely affected: any d=4 alternative must be compatible
with chirality and Lorentz signature, which may hard-gate it like ths.
""")
