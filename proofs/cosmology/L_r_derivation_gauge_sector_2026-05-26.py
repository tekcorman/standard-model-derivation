#!/usr/bin/env python3
"""
L_r derivation probe — gauge-sector F-fiber transitions (GUT, EWSB).

Scoping: an internal working note

Tests two candidate L_r derivations from FRAMEWORK-INTERNAL structure
(no fitting):

  L_r at GUT (PS retention) — TWO independent readings:
    R1 = number of layers in combined-gauge tuple  (substrate + vertex + edge = 3)
    R2 = length of PS Lie algebra commutator       ([T_A, T_B] = i f_ABC T_C = 3)
    AB1 PASS if R1 == R2 == 3 (no parameter choice).

  L_r at EWSB (PS → SM breaking) — MULTIPLE candidate readings:
    R3 = multiplet dimension                       (16 = dim of (4,2,2))
    R4 = multiplet dim + closure step              (16 + 1 = 17)
    R5 = multiplet dim + breaking direction        (16 + 1 = 17, different basis)
    R6 = log2 encoding of multiplet                (log2(16) = 4)
    R7 = number of SM sub-multiplets               (6 per generation)
    R8 = log2(SM sub-multiplets)                   (log2(6) ≈ 3)
    R9 = number of PS Lie generators               (21 = 15 + 3 + 3)
    R10 = generators × commutator length          (21 × 3 = 63)
    AB2 PASS only if ALL candidate readings give the SAME L_r without
    appeal to which value matches EWSB. Otherwise the choice is post-hoc.

INPUTS (framework-internal):
  - PS tuple: substrate srs (3 edges) × vertex Cl(6,0) (8-dim spinor) × edge Cl(0,2) (4-dim)
  - PS Lie algebra: su(4) ⊕ su(2)_L ⊕ su(2)_R, dim 15 + 3 + 3 = 21
  - PS fermion multiplet: (4, 2, 2)_PS, dim 16 per generation
  - SM sub-multiplet count per generation: 6 (Q_L, u_R, d_R, L_L, e_R, ν_R)
  - 96 combined generators, alpha = 1/2, T_P = 1.221e19 GeV

PRE-DECLARED ABORTS:
  AB1: L_r=3 readings disagree (R1 != R2) → fitting required. STOP.
  AB2: L_r=17 requires choosing one specific reading among R3..R10 based on
       which matches EWSB → post-hoc, not principled. STOP. Downgrade L_r=17.
  AB3: both AB1 and AB2 fire → combined-gauge cannot derive L_r from structure.
       Report Outcome C. STOP.
  AB4: no fitted parameters.
  AB5: NO pattern-fitting on (3, 17, 20, 22, 29) — if only L_r=3 survives,
       post-EWSB L_r values are reported as OPEN, not extrapolated.

Reports outcome A (both clear), B (only L_r=3 clears), or C (neither clears).
"""
import math

# ----------------------------------------------------------------------
# Framework-internal inputs (no fitted constants)
# ----------------------------------------------------------------------
T_P_GEV = 1.221e19
ALPHA = 0.5
N_GEN_COMBINED = 96  # 3 × 8 × 4 = substrate × vertex × edge

# PS structure
PS_LAYERS = 3  # substrate, vertex, edge
PS_COMMUTATOR_LENGTH = 3  # [T_A, T_B] = i f_ABC T_C, length 3
PS_LIE_DIM_su4 = 15
PS_LIE_DIM_su2L = 3
PS_LIE_DIM_su2R = 3
PS_LIE_DIM_TOTAL = PS_LIE_DIM_su4 + PS_LIE_DIM_su2L + PS_LIE_DIM_su2R  # 21
PS_FERMION_MULTIPLET_DIM = 4 * 2 * 2  # (4, 2, 2)_PS = 16
SM_SUB_MULTIPLETS_PER_GEN = 6  # Q_L, u_R, d_R, L_L, e_R, ν_R

# Target physics scales
T_GUT_GEV = 1.0e16
T_EWSB_GEV = 1.0e2


def T_phys_of_N(N):
    return T_P_GEV * N**(-ALPHA)


def N_of_T_phys(T):
    return (T_P_GEV / T) ** (1.0 / ALPHA)


def L_r_for_T(T_target):
    """Inverse: what L_r maps to T_target under N_attest = 96^L_r?"""
    N = N_of_T_phys(T_target)
    return math.log(N) / math.log(N_GEN_COMBINED)


def log_distance_dec(L_r, T_target):
    """Log10 distance in T_phys between 96^L_r mapping and T_target."""
    T_pred = T_phys_of_N(N_GEN_COMBINED ** L_r)
    return abs(math.log10(T_pred) - math.log10(T_target))


print("=" * 100)
print("L_r DERIVATION PROBE — gauge-sector F-fiber transitions")
print("=" * 100)
print()
print(f"Framework inputs: 96 combined generators (= 3 × 8 × 4), α=1/2, T_P={T_P_GEV:.3e} GeV")
print(f"PS Lie algebra dim = 15 + 3 + 3 = 21")
print(f"PS fermion multiplet (4,2,2)_PS dim = 16")
print(f"SM sub-multiplets per generation = 6 (Q_L, u_R, d_R, L_L, e_R, ν_R)")
print()
print(f"Target physics scales: GUT = {T_GUT_GEV:.2e} GeV; EWSB = {T_EWSB_GEV:.2e} GeV")
print(f"Implied L_r (inverse N_attest): GUT → L_r = {L_r_for_T(T_GUT_GEV):.3f}; "
      f"EWSB → L_r = {L_r_for_T(T_EWSB_GEV):.3f}")
print()

# ----------------------------------------------------------------------
# Test 1: L_r at GUT — two independent readings
# ----------------------------------------------------------------------
print("=" * 100)
print("TEST 1 — L_r at GUT (PS retention)")
print("=" * 100)
print()

R1_GUT = PS_LAYERS  # number of layers in combined-gauge tuple
R2_GUT = PS_COMMUTATOR_LENGTH  # PS Lie algebra commutator length

print(f"R1 (number of layers in combined-gauge tuple):     {R1_GUT}")
print(f"R2 (PS Lie commutator [T_A, T_B] = i f_ABC T_C):   {R2_GUT}")
print()

GUT_AB1_pass = (R1_GUT == R2_GUT)
if GUT_AB1_pass:
    L_r_GUT_derived = R1_GUT
    T_pred = T_phys_of_N(N_GEN_COMBINED ** L_r_GUT_derived)
    dec_dist = log_distance_dec(L_r_GUT_derived, T_GUT_GEV)
    print(f"AB1 PASS: both readings give L_r = {L_r_GUT_derived} without parameter choice.")
    print(f"  T_phys at L_r=3: {T_pred:.3e} GeV (target GUT: {T_GUT_GEV:.0e} GeV)")
    print(f"  Distance: {dec_dist:.3f} decades")
    GUT_CLEARS = (dec_dist < 1.0)
    print(f"  Verification (< 1 decade): {'PASS' if GUT_CLEARS else 'FAIL'}")
else:
    L_r_GUT_derived = None
    print(f"AB1 FAIL: R1 ({R1_GUT}) != R2 ({R2_GUT}). Disagreement requires parameter choice.")
    GUT_CLEARS = False
print()


# ----------------------------------------------------------------------
# Test 2: L_r at EWSB — multiple candidate readings
# ----------------------------------------------------------------------
print("=" * 100)
print("TEST 2 — L_r at EWSB (PS → SM breaking)")
print("=" * 100)
print()

EWSB_readings = {
    'R3: multiplet dim (16)':                      PS_FERMION_MULTIPLET_DIM,
    'R4: multiplet dim + closure (16+1)':          PS_FERMION_MULTIPLET_DIM + 1,
    'R5: multiplet dim + breaking dir (16+1)':     PS_FERMION_MULTIPLET_DIM + 1,
    'R6: log2(multiplet dim) (log2 16)':           round(math.log2(PS_FERMION_MULTIPLET_DIM)),
    'R7: SM sub-multiplet count (6)':              SM_SUB_MULTIPLETS_PER_GEN,
    'R8: log2(SM sub-multiplets) (≈3)':            math.ceil(math.log2(SM_SUB_MULTIPLETS_PER_GEN)),
    'R9: PS Lie generators (21)':                  PS_LIE_DIM_TOTAL,
    'R10: PS Lie × commutator (21×3)':             PS_LIE_DIM_TOTAL * 3,
    'R11: dim of PS Higgs (1,2,2)_PS (4)':         1 * 2 * 2,
    'R12: bilinear PS×PS_bar word (16+16+1)':      2 * PS_FERMION_MULTIPLET_DIM + 1,
}

print(f"{'reading':<46} {'L_r':>6} {'T_phys':<22} {'dist EWSB (dec)':>16}")
print("-" * 100)

EWSB_unique_match = None
EWSB_matches_within_0_5_dec = []
for name, L_r in EWSB_readings.items():
    T_pred = T_phys_of_N(N_GEN_COMBINED ** L_r)
    dist = log_distance_dec(L_r, T_EWSB_GEV)
    flag = ""
    if dist < 0.5:
        flag = "  <- match within 0.5 dec"
        EWSB_matches_within_0_5_dec.append((name, L_r, dist))
    print(f"{name:<46} {L_r:>6} {T_pred:.3e} GeV{'':<6} {dist:>16.3f}{flag}")

print()

# AB2 evaluation: principled requires either ONE reading dominates, or all
# agree. If multiple readings give different L_r and only specific values
# match EWSB, the choice is post-hoc.
unique_L_r_values = set(EWSB_readings.values())
print(f"Distinct L_r values across {len(EWSB_readings)} candidate readings: {sorted(unique_L_r_values)}")
print()

EWSB_AB2_pass = False
EWSB_principled_L_r = None

if len(unique_L_r_values) == 1:
    # All readings agree
    L_r_only = unique_L_r_values.pop()
    EWSB_principled_L_r = L_r_only
    EWSB_AB2_pass = True
    print(f"AB2 PASS: all readings give L_r = {L_r_only} (no choice required).")
elif len(EWSB_matches_within_0_5_dec) == 1:
    # Only one reading matches; is there an independent reason to prefer it?
    name, L_r, dist = EWSB_matches_within_0_5_dec[0]
    # The reading must be uniquely framework-natural without appeal to EWSB match.
    # Since multiple readings exist, choosing this one requires JUSTIFICATION
    # beyond just "it matches" — that justification doesn't exist.
    print(f"AB2 FAIL: only '{name.split(':')[0]}' matches within 0.5 dec, but multiple")
    print(f"         framework-natural readings exist with different L_r values.")
    print(f"         Selecting this one requires post-hoc justification.")
else:
    if len(EWSB_matches_within_0_5_dec) > 1:
        print(f"AB2 FAIL: {len(EWSB_matches_within_0_5_dec)} readings match within 0.5 dec — "
              f"ambiguous which is 'the' framework-natural reading.")
    else:
        print(f"AB2 FAIL: no reading matches within 0.5 dec — EWSB does NOT come out of "
              f"any of these framework-natural readings.")

print()
if not EWSB_AB2_pass:
    print("  Implication: L_r = 17 (the value matching EWSB in the local-algebra")
    print("  probe) does NOT have a unique framework-internal derivation. Multiple")
    print("  framework-natural readings give DIFFERENT L_r values. The L_r=17")
    print("  choice from the local-algebra regression was post-hoc.")
EWSB_CLEARS = EWSB_AB2_pass
print()


# ----------------------------------------------------------------------
# AB5: post-EWSB regime — explicitly REFUSE pattern-fitting
# ----------------------------------------------------------------------
print("=" * 100)
print("AB5 — Post-EWSB regime status (QCD, BBN, Recomb)")
print("=" * 100)
print()
print("Per AB5 of scoping doc §4: NO pattern-fitting on the L_r sequence")
print("(3, 17, 20, 22, 29) from the local-algebra probe regression.")
print()
print(f"L_r values from local-algebra regression for post-EWSB scales:")
for label, T_target, L_r_regression in [
    ('QCD',           0.2,      20),
    ('BBN',           1.0e-3,   22),
    ('Recombination', 2.6e-10,  29),
]:
    print(f"  {label:<18} regression L_r = {L_r_regression}  (T_phys at L_r = "
          f"{T_phys_of_N(N_GEN_COMBINED**L_r_regression):.3e} GeV)")
print()
print("These values DO NOT have framework-internal derivations under combined-gauge.")
print("Post-EWSB transitions are at the BOUND-STATE level (hadron / nucleon / atom)")
print("which is NOT directly captured by gauge-tuple multiplet structure.")
print()
print("AB5 ENFORCED: post-EWSB L_r values reported as OPEN structural gaps.")
print("              Bound-state mechanism is required, not algebraic word-length.")
print()


# ----------------------------------------------------------------------
# Outcome determination
# ----------------------------------------------------------------------
print("=" * 100)
print("OUTCOME DETERMINATION")
print("=" * 100)
print()
if GUT_CLEARS and EWSB_CLEARS:
    outcome = "A"
elif GUT_CLEARS and not EWSB_CLEARS:
    outcome = "B"
elif not GUT_CLEARS and not EWSB_CLEARS:
    outcome = "C"
else:
    outcome = "?"  # only EWSB without GUT — unexpected

print(f"OUTCOME: {outcome}")
print()
if outcome == "A":
    print("  Both L_r=3 (GUT) and L_r=17 (EWSB) derive structurally without fitting.")
    print("  Combined-gauge L_r selection rule established for gauge sector.")
    print("  Bound-state regime (QCD/BBN/Recomb) reported as open gap.")
elif outcome == "B":
    print("  L_r=3 for GUT derives cleanly (R1 == R2 == 3, independent readings agree).")
    print("  L_r=17 for EWSB FAILS AB2 — multiple framework-natural readings give")
    print("  different L_r values; the L_r=17 choice from local-algebra regression")
    print("  was post-hoc.")
    print()
    print("  STRUCTURAL FINDING: combined-gauge has a principled L_r at GUT but NOT")
    print("  at EWSB. The L_r selection rule cannot be derived from algebraic")
    print("  word-length alone for post-GUT transitions. Additional structure")
    print("  required: the multiway-DAG event structure of the parent scoping doc")
    print("  §1, where F-fiber transitions are EVENTS with their own structural L_r.")
elif outcome == "C":
    print("  Neither L_r=3 nor L_r=17 derives structurally.")
    print("  Combined-gauge cannot derive L_r per F-fiber transition from algebraic")
    print("  word-length. The propagation cascade reframe needs a structurally")
    print("  different mechanism (e.g., multiway-DAG event structure, or a")
    print("  fundamentally new layer).")
print()


print("=" * 100)
print(f"PROBE COMPLETE — OUTCOME {outcome}")
print("=" * 100)
