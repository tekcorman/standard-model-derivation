#!/usr/bin/env python3
"""
Propagation cascade — local-algebra layer N_attest enumeration.

Second probe in the propagation cascade reframe (see scoping doc
an internal working note and first-bridge
verdict an internal working note).

The first bridge probe established Outcome 2: Coxeter cooling cascade naturally
sits at Planck → ~10^8 GeV (top of cascade matches GUT within 0.05 decades).
The post-GUT cascade (EWSB → today, spanning 10^2 GeV → 10^-13 GeV) must come
from OTHER framework layers — flagged as Tasks A/B/C in
saturated_symmetry_zoo_cooling_cascade_scoping_2026-05-07.md §2.

This probe enumerates the LOCAL-ALGEBRA layer (Task A): N_attest for canonical
local algebras at vertex.

PRINCIPLED N_attest FORMULA (carried over from Coxeter case):
    N_attest = |generators|^max(L_r)
where max(L_r) = length of the longest defining relation in the algebra's
presentation. This is the frequency-support threshold at which the rarest
distinguishing word becomes attested ≥1× in a uniform random length-N stream.

ALGEBRA FAMILIES (canonical, framework-internal):

1. Cl(2k, 0) Clifford algebras (intrinsic, anti-commutator only):
   generators: 2k gamma matrices
   max(L_r): 4 (anti-commutator γ_i γ_j γ_i γ_j = -1 for i≠j)
   N_attest = (2k)^4

2. Cl(2k, 0) Fock-structured (with chirality / top-rank element):
   generators: 2k
   max(L_r): 2k (chirality element γ_FIVE = γ_1 γ_2 ... γ_{2k})
   N_attest = (2k)^(2k)
   This is the framework's actual usage (lepton/quark chirality essential).

3. Cayley-Dickson tower at depth d (dim 2^d, real Cayley-Dickson):
   generators: 2^d - 1 imaginary units
   max(L_r): grows with d (d=1: i^2=-1 length 2; d=2: ijk=-1 length 3;
                          d=3: Moufang length 4; d=4: zero divisors length 4)
   For depth d ≥ 2: max(L_r) ≈ d + 1 (heuristic from sedenion analyses).
   N_attest = (2^d - 1)^(d+1) for d ≥ 2

4. Exceptional Lie algebras (Serre presentation):
   generators: rank
   max(L_r): 6 (typical Serre relation ad(e_i)^2(e_j) = 0 → length 3 nested
              commutator → length 6 word in associative envelope)
   N_attest = rank^6
   Same as Coxeter analog (Cartan-Coxeter duality).

5. Combined-gauge tuple (substrate × vertex × edge):
   For Pati-Salam dominant tuple at framework scale: srs (3 generators) ×
   Cl(6,0) at vertex (8-dim spinor) × Cl(0,2)≅ℍ at edge (4-dim).
   Combined generators: 3 × 8 × 4 = 96 (the gauge-mediated multiplet alphabet).
   max(L_r): structural, depends on the gauge multiplet's natural word length.
   NOTE: combined-gauge is FLAGGED in this probe as needing a principled
   max(L_r) selection rule. We report a range, not a single value.

PRE-REGISTERED OUTCOMES:
  1. Local-algebra spans post-GUT — closes the bridge with no further layers needed.
  2. Local-algebra spans only Planck → ~few decades below GUT — confirms multi-layered
     cascade; further layers (edge-qubit + combined-gauge) needed below.
  3. Local-algebra adds nothing beyond Coxeter coverage — the layer doesn't help.

PRE-DECLARED ABORTS:
  AB1: any N_attest < 1 → algebra-internal formula broken. STOP.
  AB2: combined-gauge match requires cherry-picking word length → numerology. STOP.
  AB3: no algebra-internal coverage extension beyond Coxeter at all → bridge dead. STOP.
  AB4: no fitted parameters. All inputs are framework-internal (T_P, α=1/2, algebra
       presentations). If patching is needed, STOP.
"""
import math

# ----------------------------------------------------------------------
# Inputs (framework-internal, no fitted constants)
# ----------------------------------------------------------------------
T_P_GEV = 1.221e19
ALPHA = 0.5

PHYSICS_CASCADE = [
    ('GUT',                   1.0e16),
    ('EWSB (weak)',           1.0e2),
    ('QCD (Lambda_QCD)',      0.2),
    ('BBN',                   1.0e-3),
    ('Recombination',         2.6e-10),
    ('Today (CMB)',           2.35e-13),
]

# Reference: Coxeter coverage from first bridge probe
COXETER_N_MAX = 4.72e21  # top of Coxeter cooling cascade (|E|=8 k=8 m=3)
COXETER_N_MIN = 16        # bottom (V_4)


def T_phys_of_N(N):
    return T_P_GEV * N**(-ALPHA)


def N_of_T_phys(T):
    return (T_P_GEV / T) ** (1.0 / ALPHA)


def fmt_gev(T):
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


print("=" * 100)
print("PROPAGATION CASCADE — LOCAL-ALGEBRA LAYER N_attest ENUMERATION")
print("=" * 100)
print()
print(f"Inputs: T_P = {T_P_GEV:.3e} GeV;  alpha = {ALPHA};  T_phys(N) = T_P · N^(-alpha)")
print(f"Coxeter coverage reference (from first bridge): N ∈ [{COXETER_N_MIN}, {COXETER_N_MAX:.2e}]")
print(f"                                                 T ∈ [{fmt_gev(T_phys_of_N(COXETER_N_MAX))}, "
      f"{fmt_gev(T_phys_of_N(COXETER_N_MIN))}]")
print()


# ----------------------------------------------------------------------
# Family 1: Cl(2k, 0) intrinsic (anti-commutator only)
# ----------------------------------------------------------------------
print("=" * 100)
print("Family 1 — Cl(2k, 0) Clifford algebras, intrinsic (max L_r = 4)")
print("=" * 100)
print(f"{'algebra':<24} {'|gen|':>6} {'max(L_r)':>9} {'N_attest':>14} {'T_phys':<18}")
print("-" * 100)
cl_intrinsic = []
for k in range(1, 9):
    n_gen = 2 * k
    L_r = 4
    N = n_gen ** L_r
    T = T_phys_of_N(N)
    cl_intrinsic.append((f"Cl({n_gen},0) intrinsic", n_gen, L_r, N, T))
    print(f"{'Cl('+str(n_gen)+',0) intrinsic':<24} {n_gen:>6} {L_r:>9} {N:>14.2e} {fmt_gev(T):<18}")
print()


# ----------------------------------------------------------------------
# Family 2: Cl(2k, 0) Fock-structured (with chirality element)
# ----------------------------------------------------------------------
print("=" * 100)
print("Family 2 — Cl(2k, 0) Fock-structured (chirality element γ_5 length 2k)")
print("=" * 100)
print(f"{'algebra':<24} {'|gen|':>6} {'max(L_r)':>9} {'N_attest':>14} {'T_phys':<18}")
print("-" * 100)
cl_fock = []
for k in range(1, 9):
    n_gen = 2 * k
    L_r = 2 * k
    N = n_gen ** L_r
    T = T_phys_of_N(N)
    cl_fock.append((f"Cl({n_gen},0) Fock", n_gen, L_r, N, T))
    print(f"{'Cl('+str(n_gen)+',0) Fock':<24} {n_gen:>6} {L_r:>9} {N:>14.2e} {fmt_gev(T):<18}")
print()


# ----------------------------------------------------------------------
# Family 3: Cayley-Dickson tower
# ----------------------------------------------------------------------
print("=" * 100)
print("Family 3 — Cayley-Dickson tower at depth d (max L_r = d+1 heuristic)")
print("=" * 100)
print(f"{'algebra':<24} {'|gen|':>6} {'max(L_r)':>9} {'N_attest':>14} {'T_phys':<18}")
print("-" * 100)
cd_names = {1: 'C complex', 2: 'H quaternion', 3: 'O octonion', 4: 'S sedenion',
            5: 'pathion (d=5)', 6: 'chingon (d=6)', 7: 'routon (d=7)', 8: 'voudon (d=8)'}
cd_tower = []
for d in range(1, 9):
    n_gen = (2**d) - 1
    L_r = d + 1  # heuristic: c=length-2, h=length-3, o=length-4, s=length-5, ...
    if d == 1:
        L_r = 2  # i^2 = -1
    N = n_gen ** L_r if n_gen > 0 else 1
    T = T_phys_of_N(max(N, 1))
    cd_tower.append((cd_names[d], n_gen, L_r, N, T))
    print(f"{cd_names[d]:<24} {n_gen:>6} {L_r:>9} {N:>14.2e} {fmt_gev(T):<18}")
print()
print("NOTE: Cayley-Dickson at d ≥ 4 has zero divisors; framework's *natural use*")
print("stops at d=3 (octonion, Hurwitz bound for normed division algebras).")
print("Higher d entries shown for spectrum completeness but ARE NOT load-bearing.")
print()


# ----------------------------------------------------------------------
# Family 4: Exceptional Lie algebras (magic square)
# ----------------------------------------------------------------------
print("=" * 100)
print("Family 4 — Exceptional Lie algebras (Serre relations, max L_r = 6)")
print("=" * 100)
print(f"{'algebra':<24} {'|gen|=rank':>10} {'max(L_r)':>9} {'N_attest':>14} {'T_phys':<18}")
print("-" * 100)
exc_lie = []
for name, rank in [('F_4', 4), ('E_6', 6), ('E_7', 7), ('E_8', 8)]:
    L_r = 6
    N = rank ** L_r
    T = T_phys_of_N(N)
    exc_lie.append((name, rank, L_r, N, T))
    print(f"{name:<24} {rank:>10} {L_r:>9} {N:>14.2e} {fmt_gev(T):<18}")
print()


# ----------------------------------------------------------------------
# Family 5: Combined-gauge tuple (Pati-Salam dominant)
# ----------------------------------------------------------------------
print("=" * 100)
print("Family 5 — Combined-gauge tuple (Pati-Salam: srs × Cl(6,0) × Cl(0,2))")
print("=" * 100)
print()
print("Combined generators: substrate (3 srs edges) × vertex (8-dim Cl(6) spinor)")
print("                     × edge (4-dim Cl(0,2)≅ℍ) = 3 × 8 × 4 = 96 generators.")
print()
print("NOTE (per probe pre-registration): max(L_r) for combined-gauge has no")
print("principled framework-internal value YET. We sweep L_r = 2..30 and report")
print("the full spectrum; specific match-claims require independent justification")
print("of the chosen L_r (AB2 gate).")
print()
print(f"{'L_r':>4} {'N_attest':>14} {'T_phys':<22}  {'nearest physics scale':<24}  {'|log10 dist|':>12}")
print("-" * 100)

N_GEN_PS = 96
combined_gauge = []
ps_near_matches = {}  # collect closest matches per physics scale

for L_r in range(2, 31):
    N = N_GEN_PS ** L_r
    T = T_phys_of_N(N)
    # Find nearest physics scale
    best_label = None
    best_dist = float('inf')
    for label, T_target in PHYSICS_CASCADE:
        d = abs(math.log10(T) - math.log10(T_target))
        if d < best_dist:
            best_dist = d
            best_label = label
    combined_gauge.append((L_r, N, T, best_label, best_dist))
    flag = ""
    if best_dist < 0.5:
        flag = "  <- match"
    print(f"{L_r:>4} {N:>14.2e} {fmt_gev(T):<22}  {best_label:<24}  {best_dist:>12.2f}{flag}")
    if best_label not in ps_near_matches or best_dist < ps_near_matches[best_label][1]:
        ps_near_matches[best_label] = (L_r, best_dist)
print()


# ----------------------------------------------------------------------
# Coverage analysis: where does each family span?
# ----------------------------------------------------------------------
print("=" * 100)
print("Coverage analysis — what T_phys range does each family span?")
print("=" * 100)
print()

def span_of(family, exclude_below=None):
    Ns = [row[3] for row in family]
    if not Ns:
        return None
    valid = [n for n in Ns if (exclude_below is None or n >= exclude_below) and n > 0]
    if not valid:
        return None
    Nmin, Nmax = min(valid), max(valid)
    return (Nmin, Nmax, T_phys_of_N(Nmax), T_phys_of_N(Nmin))


for label, family in [
    ('Cl intrinsic (k=1..8)', cl_intrinsic),
    ('Cl Fock (k=1..8)', cl_fock),
    ('Cayley-Dickson (d=1..3, framework-natural)', cd_tower[:3]),
    ('Cayley-Dickson (d=1..8, full)', cd_tower),
    ('Exceptional Lie', exc_lie),
    ('Combined-gauge PS (L=2..30, unrestricted)',
     [(r[0], None, r[0], r[1], r[2]) for r in combined_gauge]),
]:
    sp = span_of(family)
    if sp is None:
        print(f"{label:<48}  (empty)")
        continue
    Nmin, Nmax, Tmax_T, Tmin_T = sp
    print(f"{label:<48}  N ∈ [{Nmin:.2e}, {Nmax:.2e}]")
    print(f"{'':<48}  T ∈ [{fmt_gev(Tmax_T)}, {fmt_gev(Tmin_T)}]")
print()


# ----------------------------------------------------------------------
# Check which physics scales fall in which family's coverage
# ----------------------------------------------------------------------
print("=" * 100)
print("Per-scale coverage — which framework layer reaches each physics scale?")
print("=" * 100)
print()
print(f"{'physics scale':<22} {'N_target':>12} {'Coxeter?':<11} {'Cl Fock?':<11} {'CD (d≤3)?':<12} {'Exc Lie?':<11} {'PS combined?':<14}")
print("-" * 100)

def covers(family, N_target):
    Ns = [row[3] for row in family if row[3] > 0]
    if not Ns:
        return False
    return min(Ns) <= N_target <= max(Ns)


for label, T in PHYSICS_CASCADE:
    N_target = N_of_T_phys(T)
    cox = COXETER_N_MIN <= N_target <= COXETER_N_MAX
    cl_f = covers(cl_fock, N_target)
    cd_3 = covers(cd_tower[:3], N_target)
    exc = covers(exc_lie, N_target)
    ps = covers([(r[0], None, r[0], r[1], r[2]) for r in combined_gauge], N_target)
    def mark(b): return 'YES' if b else 'no'
    print(f"{label:<22} {N_target:>12.2e} {mark(cox):<11} {mark(cl_f):<11} "
          f"{mark(cd_3):<12} {mark(exc):<11} {mark(ps):<14}")
print()


# ----------------------------------------------------------------------
# AB gate checks
# ----------------------------------------------------------------------
print("=" * 100)
print("AB-GATE CHECK")
print("=" * 100)
print()

# AB1: any N_attest < 1?
all_N = ([row[3] for row in cl_intrinsic] + [row[3] for row in cl_fock] +
         [row[3] for row in cd_tower if row[3] > 0] + [row[3] for row in exc_lie] +
         [row[1] for row in combined_gauge])
ab1_pass = min(all_N) >= 1
print(f"AB1 (N_attest >= 1):     {'PASS' if ab1_pass else 'FAIL'} (min = {min(all_N)})")

# AB2: combined-gauge match requires cherry-picked L_r?
# Look at which physics scales have a combined-gauge match within < 0.5 decades
# and whether multiple L_r values match different scales (= regression / numerology)
ps_matches = {label: dist for label, (L, dist) in ps_near_matches.items() if dist < 0.5}
ab2_concern = len(ps_matches) >= 2
print(f"AB2 (combined-gauge cherry-pick risk): "
      f"{'CONCERN' if ab2_concern else 'OK'} ({len(ps_matches)} scales with <0.5 dec match)")
if ab2_concern:
    print("    -> Combined-gauge can hit multiple physics scales by varying L_r alone.")
    print("       Without a principled selection rule per scale, this is parameter regression.")
    print("       Combined-gauge claims REQUIRE independent justification of L_r per scale.")
    print(f"    -> Near matches: {ps_matches}")

# AB3: does local-algebra extend post-GUT coverage at all?
post_gut_coverage_local = False
for label, T in PHYSICS_CASCADE[1:]:  # everything EXCEPT GUT
    N_target = N_of_T_phys(T)
    if covers(cl_fock, N_target) or covers(cd_tower[:3], N_target) or covers(exc_lie, N_target):
        post_gut_coverage_local = True
        break
ab3_pass = post_gut_coverage_local
print(f"AB3 (local-algebra extends post-GUT): {'PASS' if ab3_pass else 'FAIL'}")
if not ab3_pass:
    print("    -> Local-algebra (Cl Fock + Cayley-Dickson d≤3 + Exc Lie) covers ONLY the")
    print("       same Planck → GUT regime as Coxeter. No new T_phys territory.")

# AB4: no fitted parameters
print(f"AB4 (no fitted parameters): PASS (only T_P, α=1/2, and standard algebra presentations)")
print()


# ----------------------------------------------------------------------
# Outcome determination
# ----------------------------------------------------------------------
print("=" * 100)
print("OUTCOME DETERMINATION")
print("=" * 100)
print()

# Where do local-algebra families reach?
cl_fock_Nmax = max(row[3] for row in cl_fock)
cd_d3_Nmax = max(row[3] for row in cd_tower[:3])
local_Nmax = max(cl_fock_Nmax, cd_d3_Nmax, max(row[3] for row in exc_lie))
local_Tmin = T_phys_of_N(local_Nmax)
print(f"Local-algebra (framework-natural) reaches down to T = {fmt_gev(local_Tmin)}")
print(f"                                       at N = {local_Nmax:.2e}")
print()

if local_Tmin < 1.0:
    print("OUTCOME 1: Local-algebra layer ALONE reaches into post-GUT regime.")
    print("           Bridge essentially complete with this single additional layer.")
elif local_Tmin < 1e10:
    print("OUTCOME 2 (partial): Local-algebra extends coverage modestly below GUT")
    print("                     but does NOT reach EWSB. More layers needed below.")
elif local_Tmin < 1e15:
    print("OUTCOME 2: Local-algebra adds a few decades below GUT but the gap to")
    print("           EWSB remains ~10+ decades in N. The next layer (edge-qubit +")
    print("           combined-gauge with principled L_r selection) is required.")
else:
    print("OUTCOME 3: Local-algebra adds essentially no new coverage beyond Coxeter.")
    print("           The post-GUT cascade cannot be reached via canonical local algebras.")
    print("           The framework needs a structurally different layer.")
print()

# Comment on combined-gauge
if ab2_concern:
    print("COMBINED-GAUGE WARNING (AB2 concern):")
    print("  The PS combined-gauge tuple with sweepable L_r can hit MULTIPLE physics")
    print("  scales but does so via L_r selection alone. Without an independent")
    print("  framework-internal derivation of which L_r corresponds to which physical")
    print("  phase transition, combined-gauge cannot be claimed as a structural")
    print("  bridge. The structural priority is: derive L_r per phase transition")
    print("  from the multiway-DAG event structure (the F-fiber transitions of")
    print("  scoping doc §1).")
print()

print("=" * 100)
print("LOCAL-ALGEBRA LAYER ENUMERATION COMPLETE")
print("=" * 100)
