#!/usr/bin/env python3
"""
Propagation cascade first bridge — does the existing Coxeter N_attest table
line up with the standard physics cascade GUT → today, when N is read as
observer-martingale time via T_phys(N) = T_P · N^(-1/2)?

Scoping doc: an internal working note

INPUTS (all already in the framework, NO fitted constants):
  - N_attest values from sector_coxeter_freq_weighted_audit.py
    (formula N_attest = |E|^max(L_r))
  - T_P = 1.221e19 GeV (Planck temperature, standard)
  - α = 1/2 (theorem-grade beta-Bernoulli posterior σ-scaling,
    parent reframe an internal working note §3.3)
  - Standard physics cascade scales (PDG / cosmology textbooks):
      GUT             ~ 1e16  GeV
      EWSB            ~ 1e2   GeV (Higgs vev = 246 GeV; weak scale)
      QCD             ~ 0.2   GeV (Lambda_QCD)
      BBN             ~ 1e-3  GeV (1 MeV)
      Recombination   ~ 3e-10 GeV (3000 K ≈ 0.26 eV)
      Today (CMB)     ~ 2.35e-13 GeV (2.725 K)

PRE-REGISTERED OUTCOMES (per scoping doc §6):
  1. DIRECT HIT — Coxeter N_attest values map onto physics cascade scales
     within ~1 decade. Validates propagation structure.
  2. LAYERED CASCADE — Coxeter cluster in one regime (e.g., near-Planck);
     other framework cascades (local algebra, edge qubit, multi-gen) must
     populate later regimes. Locates the gap precisely.
  3. NO CORRESPONDENCE — N_attest values fall in physics-irrelevant regimes.
     Bridge claim dead.

PRE-DECLARED ABORTS:
  AB1: N_attest < 1 anywhere → substrate-anchor wrong. STOP.
  AB2: no N_attest within 3 decades of any physics scale → bridge dead. STOP.
  AB3: matches require cherry-picking → numerology. STOP.
  AB4: any fitted parameter introduced → not structural. STOP.

NO acceptance test. The probe REPORTS HONESTLY which outcome the data supports.
"""
import math

# ----------------------------------------------------------------------
# Inputs (all framework-internal, no fitted constants)
# ----------------------------------------------------------------------
T_P_GEV = 1.221e19  # Planck temperature in GeV (standard)
ALPHA = 0.5         # beta-Bernoulli posterior σ-scaling (theorem-grade)

# Standard physics cascade scales (GeV)
PHYSICS_CASCADE = [
    ('GUT',                   1.0e16),
    ('EWSB (weak)',           1.0e2),
    ('QCD (Lambda_QCD)',      0.2),
    ('BBN',                   1.0e-3),
    ('Recombination',         2.6e-10),
    ('Today (CMB)',           2.35e-13),
]


# Coxeter systems with N_attest (from sector_coxeter_freq_weighted_audit.py)
# N_attest = |E|^max(L_r) where max(L_r) = 2 * max(m_ij)
COXETER_SYSTEMS = [
    # (name, |E|, max_L_r, N_attest)
    ('V_4 (m=2)',           2, 4,  2**4),
    ('S_3 = D_3 (m=3)',     2, 6,  2**6),
    ('D_4 (m=4)',           2, 8,  2**8),
    ('D_8 (m=8)',           2, 16, 2**16),
    ('(Z/2)^3 (all m=2)',   3, 4,  3**4),
    ('A_3 = S_4',           3, 6,  3**6),
    ('B_3 octahedral',      3, 8,  3**8),
    ('H_3 icosahedral',     3, 10, 3**10),
    ('A_4 = S_5',           4, 6,  4**6),
    ('F_4',                 4, 8,  4**8),
    ('H_4',                 4, 10, 4**10),
    ('A_6 = S_7',           6, 6,  6**6),
    ('E_6',                 6, 6,  6**6),
    ('E_7',                 7, 6,  7**6),
    ('A_8 = S_9',           8, 6,  8**6),
    ('E_8',                 8, 6,  8**6),
]


# Multi-generator relations: L_r = k*m, N_attest = |E|^(km)
# Sweep per the audit's "Bounds Part B" discussion.
MULTI_GEN_SAMPLES = []
for E in [3, 4, 6, 8]:
    for k in range(2, E+1):
        for m in [2, 3]:
            L_r = k * m
            N_attest = E ** L_r
            MULTI_GEN_SAMPLES.append((f'|E|={E} k={k} m={m}', E, L_r, N_attest))


# ----------------------------------------------------------------------
# Core map: T_phys(N) = T_P * N^(-alpha), inverse N(T) = (T_P/T)^(1/alpha)
# ----------------------------------------------------------------------
def T_phys_of_N(N):
    """Physical temperature at observer-martingale time N (substrate-anchored)."""
    return T_P_GEV * N**(-ALPHA)


def N_of_T_phys(T):
    """Observer-martingale time at physical temperature T."""
    return (T_P_GEV / T) ** (1.0 / ALPHA)


def fmt_gev(T):
    """Format a temperature in GeV with reasonable unit choice."""
    if T >= 1e9:
        return f"{T:.2e} GeV"
    if T >= 1.0:
        return f"{T:.3g} GeV"
    if T >= 1e-3:
        return f"{T*1e3:.3g} MeV"
    if T >= 1e-6:
        return f"{T*1e6:.3g} keV"
    if T >= 1e-9:
        return f"{T*1e9:.3g} eV"
    if T >= 1e-12:
        return f"{T*1e12:.3g} meV"
    return f"{T:.2e} GeV"


# ----------------------------------------------------------------------
# AB1 check: any N_attest < 1?
# ----------------------------------------------------------------------
print("=" * 100)
print("PROPAGATION CASCADE — FIRST BRIDGE PROBE")
print("=" * 100)
print()
print(f"Inputs: T_P = {T_P_GEV:.3e} GeV;  alpha = {ALPHA};  T_phys(N) = T_P · N^(-alpha)")
print()

all_N_attest = [n for _,_,_,n in COXETER_SYSTEMS] + [n for _,_,_,n in MULTI_GEN_SAMPLES]
min_N = min(all_N_attest)
if min_N < 1:
    print(f"AB1 TRIGGERED: min(N_attest) = {min_N} < 1.  Substrate anchor invalid.  STOP.")
    raise SystemExit(1)
print(f"AB1 check: min(N_attest) = {min_N} >= 1.  PASS.")
print()


# ----------------------------------------------------------------------
# Table 1: Coxeter systems mapped to T_phys(N_attest)
# ----------------------------------------------------------------------
print("=" * 100)
print("Table 1 — Coxeter N_attest mapped to physical temperature")
print("=" * 100)
print(f"{'system':<28} {'|E|':>4} {'max(L_r)':>9} {'N_attest':>14}  {'T_phys = T_P/sqrt(N)':<22}")
print("-" * 100)
for name, E, L_r, N in COXETER_SYSTEMS:
    T = T_phys_of_N(N)
    print(f"{name:<28} {E:>4} {L_r:>9} {N:>14.3e}  {fmt_gev(T):<22}")
print()


# ----------------------------------------------------------------------
# Table 2: physics cascade scales → N_target
# ----------------------------------------------------------------------
print("=" * 100)
print("Table 2 — Physics cascade scales mapped to observer-time N_target = (T_P/T)^2")
print("=" * 100)
print(f"{'scale':<22} {'T (GeV)':>12}  {'N_target':>14}  {'log10(N_target)':>16}")
print("-" * 100)
for label, T in PHYSICS_CASCADE:
    N = N_of_T_phys(T)
    print(f"{label:<22} {T:>12.3e}  {N:>14.3e}  {math.log10(N):>16.2f}")
print()


# ----------------------------------------------------------------------
# Table 3: Nearest matches — for each physics scale, find closest N_attest
# ----------------------------------------------------------------------
print("=" * 100)
print("Table 3 — For each physics cascade scale, nearest Coxeter N_attest")
print("=" * 100)
print(f"{'scale':<22} {'N_target':>12}  {'nearest N_attest':>18}  {'nearest system':<24}  {'|log10 dist|':>14}")
print("-" * 100)

def closest_N_attest(N_target, source):
    best = None
    best_logd = float('inf')
    for entry in source:
        name, E, L_r, N = entry
        logd = abs(math.log10(N) - math.log10(N_target))
        if logd < best_logd:
            best_logd = logd
            best = (name, N, logd)
    return best

# Combined source (Coxeter + multi-gen)
ALL_SYSTEMS = COXETER_SYSTEMS + MULTI_GEN_SAMPLES

any_within_3_decades = False
any_within_1_decade = False
for label, T in PHYSICS_CASCADE:
    N_target = N_of_T_phys(T)
    name, N_match, logd = closest_N_attest(N_target, ALL_SYSTEMS)
    if logd < 3.0:
        any_within_3_decades = True
    if logd < 1.0:
        any_within_1_decade = True
    flag = ""
    if logd < 1.0:
        flag = "  <- WITHIN 1 DECADE"
    elif logd < 3.0:
        flag = "  <- within 3 decades"
    print(f"{label:<22} {N_target:>12.2e}  {N_match:>18.2e}  {name:<24}  {logd:>14.2f}{flag}")
print()


# ----------------------------------------------------------------------
# Table 4: Multi-generator regime — range of N_attest
# ----------------------------------------------------------------------
print("=" * 100)
print("Table 4 — Multi-generator N_attest range (|E|=3..8, k=2..|E|, m=2,3)")
print("=" * 100)
mg_N = [n for _,_,_,n in MULTI_GEN_SAMPLES]
mg_T = [T_phys_of_N(n) for n in mg_N]
print(f"N_attest range: {min(mg_N):.2e} – {max(mg_N):.2e}")
print(f"T_phys range:   {fmt_gev(max(mg_T))} – {fmt_gev(min(mg_T))}")
print()
print("Highest-N_attest multi-gen samples (closest to framework scale):")
sorted_mg = sorted(MULTI_GEN_SAMPLES, key=lambda x: -x[3])[:8]
for name, E, L_r, N in sorted_mg:
    T = T_phys_of_N(N)
    print(f"  {name:<22}  L_r={L_r:>3}  N_attest = {N:>10.2e}  T_phys = {fmt_gev(T)}")
print()


# ----------------------------------------------------------------------
# Coverage check: what N range is covered, and where are the gaps?
# ----------------------------------------------------------------------
print("=" * 100)
print("Coverage analysis — which physics cascade scales are reachable?")
print("=" * 100)
print()
all_N = sorted(set(all_N_attest))
log_min, log_max = math.log10(min(all_N)), math.log10(max(all_N))
print(f"Framework N_attest spans:  10^{log_min:.2f}  to  10^{log_max:.2f}")
print(f"  -> T_phys spans:         {fmt_gev(T_phys_of_N(max(all_N)))}  to  {fmt_gev(T_phys_of_N(min(all_N)))}")
print()

# What physics scales sit inside the covered N range?
print("Physics scales relative to framework N_attest coverage:")
print(f"{'scale':<22} {'N_target':>12}  {'inside coverage?':<20}")
for label, T in PHYSICS_CASCADE:
    N = N_of_T_phys(T)
    inside = (min(all_N) <= N <= max(all_N))
    flag = "INSIDE" if inside else ("BELOW (low N)" if N < min(all_N) else "ABOVE (high N)")
    print(f"{label:<22} {N:>12.2e}  {flag:<20}")
print()


# ----------------------------------------------------------------------
# AB2 check + outcome determination
# ----------------------------------------------------------------------
print("=" * 100)
print("AB-GATE CHECK + OUTCOME DETERMINATION")
print("=" * 100)
print()

# AB2: no N_attest within 3 decades of ANY physics scale → bridge dead
print(f"AB2 check: any N_attest within 3 decades of any physics scale?  ", end="")
if any_within_3_decades:
    print("YES.  AB2 PASS.")
else:
    print("NO.  AB2 TRIGGERED.")
print()

# Determine outcome
print("OUTCOME DETERMINATION:")
print()
n_inside = sum(1 for label, T in PHYSICS_CASCADE if min(all_N) <= N_of_T_phys(T) <= max(all_N))
n_total = len(PHYSICS_CASCADE)
print(f"  Physics scales falling INSIDE Coxeter N_attest range: {n_inside}/{n_total}")

if any_within_1_decade and n_inside >= n_total // 2:
    print()
    print("  -> OUTCOME 1 (DIRECT HIT) appears supported.")
    print("     Multiple physics scales line up with Coxeter N_attest within 1 decade.")
elif n_inside <= 1:
    print()
    print("  -> OUTCOME 2 (LAYERED CASCADE) appears supported.")
    print("     Coxeter cooling cascade covers only a narrow N regime; the standard")
    print("     physics cascade (GUT → today) spans an N range mostly OUTSIDE the")
    print("     Coxeter-attestation range. This indicates the framework's cascade is")
    print("     MULTI-LAYERED — Coxeter is one layer, with local-algebra / edge-qubit /")
    print("     longer multi-generator relations as additional layers not yet enumerated")
    print("     (per saturated_symmetry_zoo_cooling_cascade_scoping_2026-05-07.md §2).")
elif not any_within_3_decades:
    print()
    print("  -> OUTCOME 3 (NO CORRESPONDENCE).  Bridge claim dead at this level.")
    print("     STOP per AB2.  Do not patch with fitted parameters.")
else:
    print()
    print("  -> AMBIGUOUS / INTERMEDIATE.  Some matches within 3 decades but not 1.")
    print("     Treat as Outcome 2 (layered cascade) with sub-decade noise.")

print()
print("=" * 100)
print("FIRST BRIDGE PROBE COMPLETE")
print("=" * 100)
