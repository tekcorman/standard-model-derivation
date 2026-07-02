#!/usr/bin/env python3
"""
Alphabet-direct θ* derivation (Path 1) — probe.

Scoping: an internal working note

Tests four candidate derivations of θ* = 1/|alphabet| rad as the framework's
smallest MDL-resolvable angular feature at N_recomb:

  A: 1 nat ↔ 1 rad identification (information-geometric convention)
  B: Beta-Bernoulli σ coarse-grained to alphabet symbols
  C: standard ΛCDM r_s/D_A coincidence (no derivation, observational)
  D: Wigner D-matrix angular mode counting (spherical harmonics)

GOAL: identify whether any candidate gives θ* = 1/|alphabet| rad at
theorem-grade WITHOUT convention choice or numerical coincidence.

PRE-DECLARED ABORTS:
  AB1: no candidate gives θ*=1/96 rad framework-natural without convention → Path 1 fails honestly.
  AB2: candidates that require nat-rad identification → CANDIDATE not THEOREM.
  AB3: no fitted parameters.
  AB4: 0.06% match alone doesn't promote.
  AB5: factoring 96 = 24×4 = 3×8×4 isn't derivation, it's decomposition.
"""
import math

# Framework inputs (theorem-grade)
N_ALPHABET = 96       # combined-gauge tuple
N_LOCAL = 24          # 2^k* × k* = 8 × 3 (parent reframe decomposition)
N_ATOMS = 4           # atoms per srs primitive cell
K_STAR = 3            # substrate valence
K_EDGE = 2            # edge qubit dimension

# Recombination N (under α = 25/48 cumulative-Perron, posterior metric session)
T_P_GEV = 1.221e19
ALPHA = 25.0/48.0
T_recomb_GEV = 3.242e-10
N_RECOMB = (T_P_GEV / T_recomb_GEV) ** (1.0 / ALPHA)

# Planck observation
THETA_STAR_PLANCK = 0.0104108
THETA_STAR_SIGMA = 0.0000031


print("=" * 100)
print("ALPHABET-DIRECT θ* DERIVATION (Path 1) — probe")
print("=" * 100)
print()
print(f"Framework primitives (theorem-grade):")
print(f"  |alphabet| = {N_ALPHABET} = 3 × 8 × 4 = substrate × vertex Fock × edge qubit")
print(f"  N_local    = {N_LOCAL} = 2^k* × k* = 8 × 3 (per-node gauge events)")
print(f"  N_atoms    = {N_ATOMS} (per srs primitive cell)")
print(f"  k*         = {K_STAR}, k_edge = {K_EDGE}")
print(f"  N_recomb   = {N_RECOMB:.3e} (under α = 25/48)")
print()
print(f"Target: θ* = {THETA_STAR_PLANCK:.10f} ± {THETA_STAR_SIGMA:.0e} rad (Planck 2018)")
print()


# ----------------------------------------------------------------------
# Test all candidate angular-resolution readings
# ----------------------------------------------------------------------
print("=" * 100)
print("Candidate angular-resolution readings")
print("=" * 100)
print()
print(f"{'Reading':<48} {'Formula':<28} {'Value':>14} {'σ from Planck':>14}")
print("-" * 110)


def sigma_dev(value):
    return (value - THETA_STAR_PLANCK) / THETA_STAR_SIGMA


readings = [
    ('Linear / direct',                f"1/|alphabet|",                    1.0/N_ALPHABET),
    ('Full circle (1D)',                f"2π/|alphabet|",                   2*math.pi/N_ALPHABET),
    ('Spherical patch (2D)',            f"√(4π/|alphabet|)",                math.sqrt(4*math.pi/N_ALPHABET)),
    ('Spherical patch (square)',        f"√(4π)/|alphabet|",               math.sqrt(4*math.pi)/N_ALPHABET),
    ('Fisher-saturated',                f"1/√N_recomb",                   1.0/math.sqrt(N_RECOMB)),
    ('Per-atom event (parent reframe)', f"(1/N_local)/N_atoms = 1/96",      (1.0/N_LOCAL)/N_ATOMS),
    ('Wigner ℓ_max ~ √|alphabet|',      f"π/√|alphabet|",                  math.pi/math.sqrt(N_ALPHABET)),
    ('k*-scaled linear',                f"k*/|alphabet|",                  K_STAR/N_ALPHABET),
    ('N_atoms-scaled',                  f"1/(|alphabet|·N_atoms/k*)",     1.0/(N_ALPHABET*N_ATOMS/K_STAR)),
]

for name, formula, value in readings:
    sigma = sigma_dev(value)
    match = "✓" if abs(sigma) < 5 else ""
    print(f"{name:<48} {formula:<28} {value:>14.6f} {sigma:>14.2f} {match}")
print()


# ----------------------------------------------------------------------
# Candidate A: 1 nat ↔ 1 rad identification
# ----------------------------------------------------------------------
print("=" * 100)
print("Candidate A — 1 nat ↔ 1 rad identification (information-geometric)")
print("=" * 100)
print()
print("Reading: in MDL/info geometry, 1 nat (natural unit of information) corresponds")
print("to 1 radian (natural unit of angular extent) via the natural identification")
print("inherent to Beta-Bernoulli posteriors on directional parameters.")
print()
print("Under this convention:")
print(f"  Alphabet of 96 distinguishable types → 96 nats of distinguishing info")
print(f"  → 1 nat per alphabet symbol")
print(f"  → 1 rad per alphabet symbol")
print(f"  → 1/96 rad per smallest angular feature")
print()
print(f"  θ* = 1/96 = {1/96:.6f} rad → match Planck within 0.06%")
print()
print("GRADE EVALUATION:")
print("  This is a CONVENTION choice (1 nat = 1 rad). The framework hasn't")
print("  rigorously derived this identification — it's natural under specific")
print("  parameterizations but ambiguous in general.")
print()
print("  Specifically, for a Beta-Bernoulli posterior on p ∈ [0, 1]:")
print("    1 nat of info corresponds to a posterior CONCENTRATION at unit Fisher")
print("    Mapping p → angle via p = (1 + cos θ)/2 gives 1 nat → ~1 rad ONLY at")
print("    p ≈ 1/2; at other p values the conversion factor differs by ~2-fold.")
print()
print("  Verdict: CANDIDATE-STRUCTURAL (matches but convention-dependent)")
print()


# ----------------------------------------------------------------------
# Candidate B: Beta-Bernoulli σ coarse-grained to alphabet symbols
# ----------------------------------------------------------------------
print("=" * 100)
print("Candidate B — Beta-Bernoulli σ coarse-grained to alphabet symbols")
print("=" * 100)
print()
print("Reading: at N_recomb, posterior σ ∝ 1/√N (full Fisher resolution). The")
print("framework coarse-grains to alphabet level: the smallest distinguishable")
print("angular extent is 1 alphabet symbol's worth of the parameter range.")
print()
print(f"  Full Fisher resolution: 1/√N_recomb = {1/math.sqrt(N_RECOMB):.3e} rad")
print(f"  Alphabet coarse-graining: 1/|alphabet| = {1/N_ALPHABET:.6f} rad")
print(f"  Ratio: {N_ALPHABET/math.sqrt(N_RECOMB):.3e} (alphabet much COARSER than Fisher)")
print()
print("GRADE EVALUATION:")
print("  The coarse-graining IS a real framework feature — the alphabet limits")
print("  resolution. But the SPECIFIC form '1/|alphabet| rad' (not 2π/|alphabet|,")
print("  not √(4π/|alphabet|)) requires the linear identification which is what")
print("  Candidate A also needs.")
print()
print("  Verdict: REDUCES TO CANDIDATE A (still needs nat-rad identification)")
print()


# ----------------------------------------------------------------------
# Candidate C: standard ΛCDM r_s/D_A coincidence
# ----------------------------------------------------------------------
print("=" * 100)
print("Candidate C — standard ΛCDM r_s/D_A coincidence")
print("=" * 100)
print()
print("Reading: standard cosmology's r_s/D_A ≈ 1/97 by observational coincidence,")
print("matching framework's |alphabet| = 96 within 1%. The framework doesn't derive")
print("Ω_m, Ω_Λ from primitives, so this match is observational/numerical, not")
print("structural.")
print()
print(f"  ΛCDM r_s ≈ 144 Mpc, D_A ≈ 14 Gpc → r_s/D_A ≈ 0.0103 ≈ 1/97")
print(f"  Framework |alphabet| = 96 → 1/96 = 0.01042")
print(f"  Match to within 1%, both ≈ Planck θ* = 0.0104108")
print()
print("GRADE EVALUATION:")
print("  This is OBSERVATIONAL COINCIDENCE, not derivation. The framework doesn't")
print("  predict Ω_m, Ω_Λ from primitives.")
print()
print("  IF the framework PREDICTED r_s and D_A to match standard ΛCDM (which")
print("  the propagation reframe DOES NOT — see probe 10 finding r_s ≈ 300 Gpc),")
print("  then the match would be structural.")
print()
print("  Verdict: NOT A DERIVATION (observational coincidence)")
print()


# ----------------------------------------------------------------------
# Candidate D: Wigner D-matrix angular modes
# ----------------------------------------------------------------------
print("=" * 100)
print("Candidate D — Wigner D-matrix angular mode counting")
print("=" * 100)
print()
print("Reading: the CMB sky's angular modes are indexed by spherical harmonics")
print("Y_ℓm. The number of modes at multipole ℓ ≤ ℓ_max is Σ(2ℓ+1) ≈ ℓ_max². For")
print("|alphabet| = 96 distinct angular modes, ℓ_max ≈ √96 ≈ 9.8. Angular resolution")
print("at ℓ_max ≈ 10 is θ ≈ π/ℓ_max ≈ 0.314 rad.")
print()
ell_max_eq = math.sqrt(N_ALPHABET)
theta_wigner = math.pi / ell_max_eq
print(f"  ℓ_max ≈ √|alphabet| = √96 = {ell_max_eq:.2f}")
print(f"  θ_Wigner = π/ℓ_max = {theta_wigner:.4f} rad")
print(f"  Match to Planck θ* = 0.0104: factor {theta_wigner/THETA_STAR_PLANCK:.1f} off")
print()
print("GRADE EVALUATION:")
print("  This reading gives θ ≈ 0.31 rad, NOT 0.01 rad. The Wigner D-matrix")
print("  counting does NOT match the framework's 1/|alphabet| claim.")
print()
print("  Verdict: DOES NOT MATCH (0.31 rad vs 0.01 rad)")
print()


# ----------------------------------------------------------------------
# Honest summary
# ----------------------------------------------------------------------
print("=" * 100)
print("HONEST SUMMARY — Path 1 promotion attempt")
print("=" * 100)
print()
print("Of the four candidate derivations:")
print()
print("  A (1 nat ↔ 1 rad):       MATCHES numerically (0.06%) but rests on")
print("                            convention identification not independently justified.")
print()
print("  B (σ coarse-grained):     MATCHES IF Reading A's identification holds.")
print("                            Reduces to A.")
print()
print("  C (ΛCDM coincidence):     MATCHES numerically (1%) but is observational")
print("                            coincidence — framework doesn't derive Ω_m, Ω_Λ.")
print()
print("  D (Wigner D-matrix):      DOES NOT MATCH (0.31 rad vs 0.01 rad).")
print()
print("AB1 verdict: no candidate gives θ* = 1/96 rad at THEOREM-GRADE without")
print("             convention choice or observational coincidence.")
print()
print("AB2 verdict: Candidate A requires 1 nat ↔ 1 rad identification, which IS a")
print("             convention choice. Under different parameterizations the factor")
print("             can differ by 2-fold.")
print()
print("AB3 verdict: no fitted parameters introduced. PASS.")
print()
print("AB4 verdict: 0.06% match alone doesn't promote. ENFORCED.")
print()
print("AB5 verdict: parent reframe's 96 = 24 × 4 = 8 × 3 × 4 is decomposition")
print("             of |alphabet|, NOT a derivation of why 1/|alphabet| = θ* in rad.")
print()
print("PATH 1 OUTCOME: HONEST CANDIDATE-STRUCTURAL.")
print()
print("The framework's θ* = 1/96 reading:")
print("  - Has STRUCTURAL alphabet derivation (|alphabet| = 96 theorem-grade)")
print("  - Lacks RIGOROUS unit derivation (why rad, not 2π·rad or √sr)")
print("  - Matches Planck at 0.06%, structurally suggestive but not theorem-grade")
print()
print("Path 1 cannot promote θ* to STRUCTURAL via this attempt. Remains CANDIDATE.")
print()


# ----------------------------------------------------------------------
# What this implies
# ----------------------------------------------------------------------
print("=" * 100)
print("Implications for L6 closure")
print("=" * 100)
print()
print("Path 1 (alphabet-direct) fails to promote θ* to STRUCTURAL via convention-")
print("independent derivation. The 0.06% match remains a TANTALIZING NUMERICAL")
print("COINCIDENCE that hints at structural connections but isn't theorem-grade.")
print()
print("Path 2 (W3 freeze-out window for r_s/D_A) would be the next attempt:")
print("  - Show Saha freeze-out width corresponds to W3 narrow integration")
print("  - Would give r_s ≈ 0.61 Gpc within the freeze-out window")
print("  - Independent framework derivation if Saha narrowness is framework-natural")
print("  - Risk: higher post-hoc concern than Path 1")
print()
print("Path 3 (Ω_m, Ω_Λ from framework): predict ΛCDM cosmological parameters")
print("from framework primitives so that standard r_s/D_A = 1/96 emerges. This is")
print("a deep cosmological reform requiring 5-10+ sessions.")
print()
print("Honest L6 status: θ* = 1/96 remains CANDIDATE-STRUCTURAL via alphabet-direct.")
print("Multi-session work would aim at one of the three paths above.")
print()
print("=" * 100)
print("ALPHABET-DIRECT θ* PROBE COMPLETE")
print("=" * 100)
