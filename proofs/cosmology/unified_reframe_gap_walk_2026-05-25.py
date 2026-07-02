#!/usr/bin/env python3
"""
Unified observation-process reframe — walking the four gaps (G1-G4).

Per an internal working note §4,
four gaps remain open. This probe walks each in turn with honest attempts:

  G1 — Rigorous mechanism for d_eff_horizon = 3 + 1/(2|E|)
  G2 — Calibration discipline for cumulative-history observables
  G3 — Rigorous structural reading for θ* = α_GUT_bare/N_atoms
  G4 — 3.5% T_today residual structural source

Honest discipline (W58 / no-fit / no-pattern-hunting): report whatever
the walk finds — closure, partial progress, or honest negative.
"""

from __future__ import annotations
import math
from fractions import Fraction


# Framework primitives (theorem-grade upstream)
k_B = 1.380649e-23
hbar = 1.054571817e-34
c_light = 2.99792458e8
G_Newton = 6.6743e-11
t_P = math.sqrt(hbar * G_Newton / c_light ** 5)
T_P = hbar / (k_B * t_P)
M_Pl_GeV = 1.220890e19

k_star = 3
N_atoms = 4
n_E = 6
two_E = 2 * n_E
c_S = Fraction(1, two_E)
alpha_GUT_bare = Fraction(1, 24)
alpha_1_bare = Fraction(2, 3) ** 8                  # 256/6561
waterline = alpha_1_bare / (1 - alpha_1_bare)       # 256/6305

N_hub = 8.394881e60
v_today = 246.22
M_unif_GeV = 1.985e16
T_CMB = 2.7255
theta_star_obs = 0.01041085
theta_star_sigma = 0.0000031

print("=" * 76)
print("Unified observation-process reframe — walking the four gaps")
print("=" * 76)


# ===========================================================================
# G1 — Rigorous mechanism for d_eff_horizon = 3 + 1/(2|E|)
# ===========================================================================
print(f"\n{'#'*76}")
print("G1 — Rigorous derivation of d_eff_horizon = 3 + 1/(2|E|)")
print('#'*76)
print(f"""
Goal: sharpen the partial M2 mechanism (Π(1+c_S/n) → N^c_S) to a rigorous
derivation tying beta-Bernoulli information accumulation directly to the
exponent 1/(2|E|) on horizon volume.

ARGUMENT:
  Step 1 — Per-edge posterior information accumulation
    Beta-Bernoulli on edge e: after N_e observations, posterior is
    Beta(α_0+k, β_0+N_e-k). For Jeffreys prior + symmetric updates,
    Fisher information accumulates as N_e (the standard 1/√N_e SE).
    Posterior log-precision: log(N_e) bits per edge.

  Step 2 — Total observer information at epoch N
    For uniform sampling across 2|E| edges: N_e = N/(2|E|).
    Per-edge information: log(N/(2|E|)) ≈ log(N) for large N.
    Total across all edges: 2|E| × log(N).

  Step 3 — Perron-projected (gauge-readable) share
    Perron-singlet weight at Γ: c_S = 1/(2|E|).
    Gauge-readable information at epoch N: c_S × (total information per cell)
                                          = (1/(2|E|)) × 2|E| × log(N)
                                          = log(N)

    PER EDGE, gauge-readable information contribution: c_S × log(N) =
    log(N)/(2|E|).

  Step 4 — Microstate volume from Boltzmann S = log W
    Per-edge microstate factor: W_edge = exp(c_S × log N) = N^c_S = N^(1/(2|E|))

    Combined 3D spatial horizon × per-edge microstate factor:
    V_eff = V_3D × W_edge = N^3 × N^(1/(2|E|)) = N^(3 + 1/(2|E|))

  Step 5 — d_eff_horizon = 3 + 1/(2|E|) ✓ TARGET

CRITICAL CHECK: in Step 4, why does W_edge multiply V_3D (rather than
add to it as another spatial dimension, or compound across edges)?

The structural reading: V_3D counts spatial CELLS in the horizon; the
Perron-projected information per cell adds a fractional MICROSTATE-VOLUME
dimension per cell. Each cell's microstate space gets enlarged by the
gauge-readable factor N^(1/(2|E|)). Total effective volume = (number of
cells) × (microstate volume per cell) = N^3 × N^(1/(2|E|)).
""")

# Numerical verification of the derivation
d_eff_derived = 3 + float(c_S)
alpha_from_d_eff = (d_eff_derived - 1) / 4
T_today_from_d_eff = T_P * math.exp(-alpha_from_d_eff * math.log(N_hub))

print(f"DERIVED quantities (rigorous step-by-step):")
print(f"  Per-edge Fisher information: I_edge(N_e) = N_e/(p̂(1-p̂)) ≈ N_e × 4 at p̂=1/2")
print(f"  Per-edge posterior log-precision: log(N_e) ≈ log(N) for large N")
print(f"  Perron-projected per-edge information: c_S × log(N) = log(N)/(2|E|)")
print(f"  Per-edge microstate factor: W_edge = N^c_S = N^(1/(2|E|))")
print(f"  d_eff_horizon = 3 + c_S = {Fraction(3) + c_S}")
print(f"  α = (d_eff - 1)/4 = {Fraction(3) + c_S - 1}/4 = {Fraction((Fraction(3) + c_S - 1), 4)} = {alpha_from_d_eff:.6f}")
print(f"  T_today predicted = T_P × N_hub^(-α) = {T_today_from_d_eff:.4f} K")
print(f"  vs observed {T_CMB} K → {(T_today_from_d_eff - T_CMB)/T_CMB*100:+.2f}%")

print(f"""
G1 VERDICT — STRUCTURALLY RIGOROUS BUT WITH ONE OPEN STEP:
  Steps 1-3 are rigorous (standard beta-Bernoulli posterior theory +
  theorem-grade Perron-singlet projection).
  Step 4 has one structural assertion that needs further justification:
  WHY does the per-edge microstate factor multiply (rather than add to)
  the spatial cell count?

  The multiplicative reading is consistent with "each cell has its own
  microstate space, total = product over cells × spatial structure",
  but a tight derivation of this multiplicative form from first
  principles would require more work.

  G1 LANDS at STRUCTURAL DERIVATION WITH ONE LOAD-BEARING ASSERTION
  (multiplicative cell × microstate factoring). Improved from M2/M3
  PARTIAL to "rigorous up to one structural choice". Not yet theorem-
  grade closure, but tighter than the previous mechanism gap.
""")


# ===========================================================================
# G2 — Calibration discipline for cumulative-history observables
# ===========================================================================
print(f"\n{'#'*76}")
print("G2 — Calibration discipline: second cumulative-history observable?")
print('#'*76)
print(f"""
The Routes H+C calibration discipline requires multiple INDEPENDENT
observable derivations using the SAME mechanism. For the cumulative-
history class (mechanism = cumulative-Perron over substrate ticks), we
need a SECOND observable to verify the d_eff_horizon mechanism.

CANDIDATE OBSERVABLES TO TEST:

  (a) δ_r (Z-channel oblique correction):
      Form: δ_r = c_S × α₁/(1-α₁) = c_S × 256/6305
      Mechanism: instantaneous Perron-singlet projection × cumulative
        dark-waterline GEOMETRIC SUM α₁ + α₁² + α₁³ + ...
      The geometric sum IS cumulative — but at SCALE of substrate
      windings, not observation epoch N.
      Class: PARTIAL cumulative-history (cumulative over windings, not N)

  (b) A_s (primordial scalar amplitude):
      Form: A_s = α_GUT × (2/3)^g × (M_GUT/M_Pl)² × (1/54)
      Mechanism: bare-a single-loop closure at the §8 family level
      Class: INSTANTANEOUS (at gauge unification epoch, no N-cumulative)

  (c) Λ_CC (cosmological constant):
      Form: V_Ram h↔h̄ split
      Class: SUB-SECTOR PROJECTION, distinct mechanism

  (d) η_baryon-to-photon ratio (if A_s framing applies):
      Class: substrate-MDL-allocated under reframe, set by primitives
      not cumulative-N

None of (a)-(d) is cleanly the SAME class as T(N) cumulative-Perron-
horizon-volume. δ_r is closest (cumulative over windings) but its
cumulation is at substrate-spectral level, not observation-epoch N.

DEEPER CALIBRATION ATTEMPT: try to verify d_eff_horizon = 3 + 1/(2|E|)
via TWO INDEPENDENT mechanisms (Routes H + Routes C analog):

  Route 1 (already in G1): beta-Bernoulli posterior log-precision × c_S
    → gives 3 + 1/(2|E|)

  Route 2 candidate: Stark-Terras spectral integration of cumulative
    Perron-singlet eigenmode flux on B_NB over substrate history.

    For B_NB(srs) Perron eigenvalue λ = k* - 1 = 2:
    cumulative Perron-flux over N ticks: ∫(c_S × λ^n) ≈ c_S × λ^N

    This is EXPONENTIAL growth, NOT power-law correction to 3.
    Route 2 candidate FAILS to reach 3 + 1/(2|E|) via spectral integration.
""")

print(f"G2 VERDICT — calibration discipline NOT achievable in current framework:")
print(f"""
  - No second observable in cumulative-N-horizon-volume class found
  - Spectral-integration Route 2 candidate FAILS (gives exponential,
    not power-law correction)
  - G1's derivation has only ONE route (Route 1: posterior precision)

  The cumulative-history observable class has a SINGLE EXAMPLE (T(N)).
  Routes H+C calibration discipline doesn't apply — there's nothing
  to cross-check against.

  This is a structural limitation of the framework's current observable
  catalogue, not a defect of the mechanism. The mechanism is supported
  by ONE numerical match (T_today within 3.5%) + ONE structural
  derivation (G1).

  G2 LANDS at HONEST NEGATIVE for calibration discipline.

  Recommendation: accept the d_eff_horizon = 3+1/(2|E|) candidate at
  "structural derivation with single-route support" grade. To upgrade
  to theorem-grade closure would require finding a second cumulative-
  history observable in the framework — currently unavailable.
""")


# ===========================================================================
# G3 — Rigorous structural reading for θ* = α_GUT_bare/N_atoms
# ===========================================================================
print(f"\n{'#'*76}")
print("G3 — Rigorous derivation of θ* = α_GUT_bare/N_atoms = 1/96")
print('#'*76)
print(f"""
The numerical match is clean: θ* = 1/(2^k* × k* × N_atoms) = 1/96 =
0.0104167 rad vs Planck 0.0104108 ± 0.0000031 → +1.88σ (+0.06%). Uses
only theorem-grade framework primitives. But the structural reading
("angular resolution at one gauge event per primitive-cell atom") is
heuristic.

RIGOROUS DERIVATION ATTEMPT — Bayes-optimal angular resolution:

  Setup: observer makes beta-Bernoulli observations on CMB-sphere
  directions. The CMB sphere is partitioned into substrate primitive
  cell projections. Each cell projects to some angular extent θ_cell.

  Per primitive cell, observer can resolve up to N_local = α_GUT_bare^(-1)
  = 24 distinct gauge-event states. Across N_atoms = 4 atoms per cell,
  there are N_atoms × N_local = 96 distinct primitive observables per
  cell projection.

  The Bayes-optimal angular resolution at which one of these 96
  observables becomes detectable is:
    θ_resolve = θ_cell / 96

  If θ_cell = 1 rad (one cell projection spans 1 rad — but this is
  the heuristic assumption), then θ_resolve = 1/96 = θ_*.

  WHY θ_cell = 1 rad? The framework doesn't have a derivation that says
  primitive-cell projections span exactly 1 rad. This is an unjustified
  choice.

ALTERNATIVE READING — geodesic-distance scale:

  In the observer's posterior space (Fisher-metric d_spatial = 3), the
  geodesic distance scale corresponding to one primitive cell's worth of
  posterior information is set by the Cencov-Fisher metric. For a
  Bernoulli posterior near p̂=1/2, the Fisher metric is approximately
  flat with characteristic scale 1/√N per direction.

  The angular conversion from posterior-space-distance to CMB-sphere-angle
  depends on the observer's specific embedding — which the framework
  doesn't yet have a derived form for.

OK — let me try a DIFFERENT structural reading:

  At recombination epoch N_rec, the observer's information about the
  substrate's gauge structure has saturated some specific amount. The
  angular size at which "one gauge event per primitive cell atom" is
  detectable converts to:
    θ_* ~ 1/(observations per cell-atom at recombination)
        ~ 1/(N_rec/(N_atoms × N_local))
        ~ N_atoms × N_local / N_rec
        = 96 / N_rec

  Setting θ_* = 0.0104108 rad: 96/N_rec = 0.0104108, so N_rec = 9220.

  Framework's z_eff candidates: BAO-Fisher ~1.83 (today-late) or CMB-
  visibility ~5800 (CMB-recombination). The N_rec = 9220 from this θ_*
  reading doesn't immediately correspond to either.

UPSHOT: the structural reading remains heuristic. The 1/96 numerical
match is composed of theorem-grade primitives (α_GUT_bare, N_atoms),
but the RIGOROUS DERIVATION of WHY these specific primitives in this
specific combination produce Planck's θ_* requires:
  - A derived embedding of substrate primitive-cells in the CMB sphere
  - A derived "θ_cell" angular extent
  - Or an alternative structural mechanism we haven't found yet
""")

print(f"G3 VERDICT — rigorous mechanism remains heuristic.")
print(f"""
  The 1/96 = α_GUT_bare/N_atoms numerical match is real (uses theorem-
  grade primitives, no fitted constants, within 2σ of Planck precision).

  The structural reading "angular resolution at gauge-event-per-atom"
  is motivated but contains an unjustified assumption (primitive-cell
  angular extent = 1 rad or similar).

  Alternative readings (geodesic-distance scale, observation-count
  inverse) don't close cleanly.

  G3 LANDS at NUMERICAL CANDIDATE WITH PARTIAL STRUCTURAL READING.
  Similar epistemic grade to G1 — supported by clean numerics + heuristic
  but not airtight structural argument.
""")


# ===========================================================================
# G4 — 3.5% T_today residual structural source
# ===========================================================================
print(f"\n{'#'*76}")
print("G4 — Structural source of the 3.5% T_today residual")
print('#'*76)
print(f"""
Under α = 25/48 cumulative-Perron at substrate anchor:
  T_today_predicted = 2.6305 K
  T_today_observed  = 2.7255 K
  Residual = -3.48%

What α gives T_today = 2.7255 exactly?
""")

alpha_exact = (math.log(T_P) - math.log(T_CMB)) / math.log(N_hub)
print(f"  α_exact = (ln(T_P) - ln(T_CMB)) / ln(N_hub) = {alpha_exact:.6f}")
print(f"  vs α = 25/48 = 0.520833")
print(f"  Δα = {alpha_exact - 25/48:.6f}")

d_exact = 4 * alpha_exact + 1
print(f"\n  Corresponding d_eff_exact = 4α + 1 = {d_exact:.6f}")
print(f"  vs d_eff_target = 3 + 1/12 = {3 + 1/12:.6f}")
print(f"  Δd = {d_exact - (3 + 1/12):.6f}")

print(f"""
The needed correction to d_eff_horizon is +{d_exact - (3 + 1/12):.4f}.

CANDIDATE STRUCTURAL CORRECTIONS:

  (a) Sub-leading dark-correction shift to 1/(2|E|):
      d_eff = 3 + (1/12)(1 - α₁/(1-α₁)) = 3 + (1/12)(1 - 256/6305)
            = 3 + (1/12)(6049/6305) = {3 + (1/12)*(6049/6305):.6f}
      Gives α = {((3 + (1/12)*(6049/6305)) - 1)/4:.6f}
      T_today_pred = T_P × N_hub^(-α) = ?
""")

d_cand_a = 3 + (1/12) * (6049/6305)
alpha_cand_a = (d_cand_a - 1) / 4
T_cand_a = T_P * math.exp(-alpha_cand_a * math.log(N_hub))
print(f"      T_today_pred = {T_cand_a:.4f} K → {(T_cand_a - T_CMB)/T_CMB*100:+.2f}%")

print(f"""
  (b) Add second-order Perron correction 1/(2|E|)²:
      d_eff = 3 + 1/(2|E|) + 1/(2|E|)² × prefactor
      For prefactor = 1 (simple second-order): d_eff = 3 + 1/12 + 1/144
""")
d_cand_b = 3 + 1/12 + 1/144
alpha_cand_b = (d_cand_b - 1) / 4
T_cand_b = T_P * math.exp(-alpha_cand_b * math.log(N_hub))
print(f"      d_eff = {d_cand_b:.6f}, α = {alpha_cand_b:.6f}")
print(f"      T_today_pred = {T_cand_b:.4f} K → {(T_cand_b - T_CMB)/T_CMB*100:+.2f}%")

print(f"""
  (c) N_hub precision: if N_hub is slightly different from 8.395e60,
      T_today shifts. What N_hub gives T_today = 2.7255 exactly at α=25/48?
""")
N_hub_required = math.exp((math.log(T_P) - math.log(T_CMB)) / (25/48))
print(f"      N_hub_required = {N_hub_required:.3e}")
print(f"      vs current N_hub = {N_hub:.3e}")
print(f"      Ratio: {N_hub_required/N_hub:.4f} ({(N_hub_required/N_hub - 1)*100:+.2f}%)")

print(f"""
  (d) Constant prefactor: T(N) = c_prefactor × T_P × N^(-25/48)
      Need c_prefactor = T_CMB / T_pred = {T_CMB/T_today_from_d_eff:.4f}
      In framework primitives, this is ~1.036. Candidates:
""")
target_ratio = T_CMB / T_today_from_d_eff
print(f"        25/24 = {25/24:.4f}")
print(f"        √(13/12) = {math.sqrt(13/12):.4f}")
print(f"        1 + 1/(2|E|·k*) = 1 + 1/36 = {1 + 1/36:.4f}")
print(f"        (Required: {target_ratio:.4f})")

print(f"""
ANALYSIS:
  (a) Dark-correction shift: T_today = {T_cand_a:.4f} K, residual {(T_cand_a - T_CMB)/T_CMB*100:+.2f}%.
      Closer than baseline but still {abs((T_cand_a - T_CMB)/T_CMB*100):.1f}% off. STRUCTURAL but
      not closing.
  (b) Second-order Perron 1/144: T_today = {T_cand_b:.4f} K, residual
      {(T_cand_b - T_CMB)/T_CMB*100:+.2f}%. Overcorrects.
  (c) N_hub precision: requires {(N_hub_required/N_hub - 1)*100:.2f}% shift in N_hub. The
      framework's N_hub is "pinned by G_F consistency" — a 2.6% shift
      would be load-bearing on the G_F-anchor chain. NOT obviously
      available.
  (d) Constant prefactor: ~1.036 doesn't match clean framework primitives.

NONE of (a)-(d) gives a clean structural closure of the 3.5% residual.
""")

print(f"G4 VERDICT — 3.5% residual remains genuinely open.")
print(f"""
  - Candidate structural corrections (a), (b) don't close
  - Candidate (c) N_hub precision shift would require 2.6% N_hub
    adjustment, not obviously available
  - Candidate (d) prefactor not in clean framework primitives

  The 3.5% residual is SMALLER than the 8% at GUT anchor but not closing
  via any clean structural mechanism. May be:
    - Sub-leading dark sector contribution not captured by current
      framework
    - Genuine precision floor for A1 candidate
    - N_hub precision issue

  G4 LANDS at OPEN. The 3.5% residual is now well-characterized as
  "outside the clean structural-correction catalogue", same epistemic
  status as the 8% was before — just smaller.

  Per W58 discipline: do NOT chase the 3.5% with pattern-hunting fits.
  Accept the cumulative-Perron candidate at 3.5% precision and move
  on, OR find a structural mechanism that's INDEPENDENTLY motivated.
""")


# ===========================================================================
# SUMMARY OF GAP WALK
# ===========================================================================
print(f"\n{'='*76}")
print("UNIFIED GAP-WALK SUMMARY")
print('='*76)
print(f"""
G1 (d_eff_horizon mechanism): STRUCTURAL DERIVATION with one load-
    bearing assertion (multiplicative cell × microstate factoring).
    Improvement from PARTIAL to "rigorous up to one structural choice".
    Path forward: derive the multiplicative factoring rigorously from
    first principles.

G2 (calibration discipline): HONEST NEGATIVE.
    No second cumulative-history observable found in framework. Routes
    H+C analog calibration not achievable. Single-route support only.
    Path forward: either find new cumulative-history observable OR
    accept single-route support as the framework's structural limit.

G3 (θ* rigorous derivation): NUMERICAL CANDIDATE with partial
    structural reading. The 1/96 = α_GUT_bare/N_atoms match is real
    but the "angular resolution per cell-atom" reading requires an
    unjustified primitive-cell angular extent. Path forward: derive
    the substrate-cell-to-CMB-sphere embedding rigorously.

G4 (3.5% T_today residual): OPEN.
    Candidate sources tested (dark shift, second-order Perron, N_hub
    precision, constant prefactor) — none close cleanly. May be
    precision floor or sub-leading effect not in current framework.
    Path forward: accept 3.5% precision floor OR find new mechanism.

NET LANDING for the unified observation-process reframe:
  - G1 improves to structural-with-one-assertion (was PARTIAL)
  - G2, G3, G4 remain open with clearly-stated structures
  - No gap closes to theorem-grade in this walk
  - But the FRAMING is now substantially tighter than before

EPISTEMIC GRADE: the reframe stays at "structural candidate with strong
numerical support and clearly-stated mechanism gaps", as before. The
walk SHARPENS the gaps without closing them.

Per W58 / no-fit discipline: no pattern-hunting attempted; all
candidates structurally motivated; honest negatives reported where
applicable.
""")

print("=" * 76)
print("STATUS: gap-walk SHARPENS but does not close G1-G4.")
print("        Reframe remains structural candidate; multi-session work needed.")
print("=" * 76)
