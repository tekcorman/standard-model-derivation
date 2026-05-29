#!/usr/bin/env python3
"""
proofs/cosmology/cS_2sphere_boundary_reopener_2026-05-28.py

PURSUE the one named reopener of the c_S factor-of-2 (gravity coupling):
  is the GRAVITATING horizon entropy S_grav (entering the Cai-Kim first law on the
  apparent-horizon 2-sphere) FORCED by boundary geometry to be 2x the observer's
  worldline information record S_record = N — closing G_eff = G?

Parents:
  - cS_extent_vs_flux_2026-05-28.py (FORCED-1: worldline accounting gives c_S=1 ->
    G_eff=2G; named this 2-sphere reopener as the only escape)
  - holographic_identification_conjecture_first_analysis_2026-05-28.py (Bekenstein
    check: S_record = N is LINEAR, ~N below the area bound A/4G ~ N^2)

THE TWO ROLES ANY "ENTROPY" PLAYS (the reconciliation constraint)
-----------------------------------------------------------------
  Role A (Cai-Kim coupling):   G_eff = 1/(kappa * c_S * M_Pl),  S_grav = c_S R_A M_Pl
  Role B (holographic source): rho_sub = E_obs/V, E_obs = kappa * S_record,
                               volume-matching the cascade clock fixes kappa = M_Pl/2.
The verdict that named this reopener worried S_grav and S_record must be the SAME
object. This probe FIRST settles that (B-A1): what exactly must be true for c_S=2?

BLIND DISCIPLINE
----------------
Decide whether boundary geometry FORCES the doubling, without assuming it to land
G_eff=G. Pre-registered:
  FORCED-2     a geometric factor forces S_grav = 2 S_record -> flagship closes.
  NOT-FORCED   no clean geometric factor 2; the doubling would have to be adopted
               -> gravity stays form-level (the investigation of the factor is then
               exhaustive across all routes tried).

sympy + numeric. References (tracked): predictions/N_hub_derivation.md,
predictions/k_star.py, docs/theorems/theorem_observer_energy_functional.md,
Cai-Kim 2005 (JHEP 0502:050), Hayward 1998 (dynamical apparent-horizon T).
"""
from __future__ import annotations
import math
import sys

import sympy as sp

FAIL = []


def abort(tag, msg):
    print(f"\n  X ABORT [{tag}] — HONEST NEGATIVE\n    {msg}")
    FAIL.append(tag)


def head(s):
    print("\n" + "=" * 78 + f"\n  {s}\n" + "=" * 78)


print(__doc__)

# ----------------------------------------------------------------------
# B-A1 — what closure REQUIRES: S_grav = 2 * S_record  (sympy)
# ----------------------------------------------------------------------
head("B-A1 — closure requires the gravitating entropy = 2x the worldline record")

kappa, cS, N, H, M, R = sp.symbols("kappa c_S N H M_Pl R_A", positive=True)

# Role B (holographic + volume scheme): fixes kappa from the cascade clock.
# rho_sub = kappa*S_record/V, V=(4pi/3)R_H^3, standard Friedmann, S_record = N:
#   H = 1/(2 G kappa N) ; cascade H = 1/(N t_P) = M_Pl/N ; G=1/M_Pl^2
#   => kappa = M_Pl/2   (N cancels)
kappa_B = M / 2
print(f"  Role B (holographic source, volume scheme): kappa = M_Pl/2  [cascade clock]")

# Role A (Cai-Kim coupling): G_eff = 1/(kappa*c_S*M_Pl); S_grav = c_S*R_A*M_Pl.
G_eff = 1 / (kappa * cS * M)
G_newton = 1 / M**2
cS_for_Geff_G = sp.solve(sp.Eq(G_eff.subs(kappa, kappa_B), G_newton), cS)[0]
print(f"  Role A (Cai-Kim coupling): G_eff = G with kappa=M_Pl/2  =>  c_S = {cS_for_Geff_G}")

# So S_grav/S_record = c_S * R_A * M_Pl / (R_A * M_Pl) = c_S = 2.
ratio_required = cS_for_Geff_G
print(f"  S_grav / S_record = c_S = {ratio_required}")
print(f"  i.e. CLOSURE <=> the apparent-horizon thermodynamic entropy is EXACTLY")
print(f"  TWICE the observer's worldline information record (E_obs = kappa*N).")
# and the converse: if S_grav = S_record (the framework's current identification):
G_eff_if_equal = sp.simplify(G_eff.subs({kappa: kappa_B, cS: 1}) / G_newton)
print(f"  Converse: if S_grav = S_record (c_S=1, current identification) => G_eff = {G_eff_if_equal}G.")
ba1 = (ratio_required == 2) and (G_eff_if_equal == 2)
if not ba1:
    abort("B-A1", "reconciliation algebra wrong.")
else:
    print("  -> the reopener must FORCE the factor 2 between two a-priori-distinct")
    print("     entropies; this is a non-trivial structural requirement.  OK")

# ----------------------------------------------------------------------
# B-A2 — the structural cost: this SPLITS the 'OEF = Clausius' asset
# ----------------------------------------------------------------------
head("B-A2 — the cost: S_grav != S_record breaks the 'OEF IS the Clausius relation'")

print("""  The framework's prized gravity asset is that its OEF theorem
  delta E_obs = kappa * dS_total IS the horizon Clausius relation delta Q = T dS
  (T <-> kappa), DERIVED not postulated. That identification uses ONE entropy:
  S_total = S_record = N.

  But B-A1 shows closure needs the gravitating entropy S_grav = 2N, distinct from
  the OEF record S_record = N. So the reopener can only close by DECLARING two
  different entropies:
    - S_record = N : the observer's worldline information (sets E_obs, role B);
    - S_grav  = 2N : the apparent-horizon thermodynamic entropy (role A).
  Then the OEF relation is NOT directly the gravitational Clausius relation — it is
  the SOURCE term, while a separate boundary entropy supplies the coupling. The
  factor 2 must come from the geometric relation between these two, not be adopted.
  => The reopener is viable ONLY if geometry FORCES S_grav = 2 S_record (B-A3).""")
ba2 = True  # statement of cost; no numeric claim

# ----------------------------------------------------------------------
# B-A3 — does boundary geometry FORCE S_grav = 2 S_record? (enumerate + measure)
# ----------------------------------------------------------------------
head("B-A3 — geometric factor between worldline record and 2-sphere boundary")

k_star = 3
# The observer worldline paints S_record = N = R_A*M_Pl (radial/temporal extent).
# Candidate geometric relations S_grav/S_record the 2-sphere boundary could supply:
geom = {}

# (1) radial extent only (the worldline itself) -> ratio 1
geom["radial worldline extent (current id.)"] = sp.Integer(1)

# (2) chord/diameter through observer (antipode-to-antipode = 2 R_A) -> ratio 2,
#     BUT the observer's causal AGE is R_A (one-way), not 2R_A; the far half lies
#     outside the realized worldline history.
geom["diameter 2R_A vs radius R_A"] = sp.Integer(2)

# (3) full 4pi sphere vs visible 2pi hemisphere -> ratio 2, BUT a central observer
#     sees the FULL 4pi sphere (no hemisphere restriction in space).
geom["4pi sphere / 2pi hemisphere"] = sp.Integer(2)

# (4) area count (Bekenstein) -> ratio ~ N (breaks linearity / coasting), NOT 2.
geom["area count A/4G (~N, breaks linear)"] = sp.Symbol("~N")

# (5) ingoing + outgoing null congruences on the apparent horizon -> naively 2,
#     BUT the horizon entropy A/4G is one surface; standard accounting does NOT
#     double for the two congruences.
geom["ingoing+outgoing null (A/4G single)"] = sp.Integer(1)

# (6) k*-coordination edge factor -> tracks k*=3, not 2.
geom["edges per node (k* coordination)"] = sp.Rational(k_star, 1)

print("  candidate S_grav/S_record from boundary geometry:")
forced = []
for name, val in geom.items():
    is2 = (val == 2)
    # 'forced' means the factor 2 is geometrically NECESSARY, not merely available.
    # Mark which 2-valued candidates are forced vs which are defeated by a physical
    # objection (noted below).
    print(f"    {str(val):>6}   {name}")
print(f"""
  Assessment of the two candidates that give 2:
    - diameter 2R_A : DEFEATED. The observer's causal age (hence N) is R_A (one
      light-crossing, one-way). The antipodal half of any chord lies outside the
      realized worldline; counting it doubles a length the observer never traversed.
    - 4pi vs 2pi    : DEFEATED. A central observer perceives the FULL 4pi 2-sphere;
      there is no spatial hemisphere restriction. (A past-vs-future split is
      temporal, already encoded in 'one-way age = R_A'.)
  The remaining candidates give 1 (radial / single-surface), ~N (area, breaks the
  linear/coasting structure), or k*=3 (edge counting) -- none is a forced 2.""")
forced_two = []  # no geometric candidate forces the factor 2
print(f"  geometric candidates that FORCE S_grav = 2 S_record: "
      f"{forced_two if forced_two else 'NONE'}")
ba3_forced = len(forced_two) > 0

# ----------------------------------------------------------------------
# B-A4 — the one principled, non-defeated route (named, not forced)
# ----------------------------------------------------------------------
head("B-A4 — mutual information across the horizon (the residual candidate)")

print("""  One reading is NOT defeated by a geometric objection but is also NOT forced:
    horizon MUTUAL INFORMATION.  For a globally pure state split by the horizon
    into interior A and exterior B, I(A:B) = S(A)+S(B)-S(AB) = 2 S(A) (since
    S(AB)=0, S(A)=S(B)). If the GRAVITATING entropy were the boundary mutual
    information (2 S_record) while the observer's RECORD is S(A)=S_record, the
    factor 2 would follow.

  But this is a CHOICE, not a forcing:
    - standard holography gravitates the ENTANGLEMENT entropy S(A) (=A/4G), i.e.
      ONE x S_record, not the mutual information -> that gives back c_S=1, G_eff=2G;
    - asserting 'gravity sees the mutual information (2x)' to land G_eff=G is exactly
      the goal-seeking the blind protocol forbids.
  Also: a dynamical-apparent-horizon temperature carries its OWN forced O(1) in
  coasting -- Hayward T = (1/2 pi R_A)(1 - dR_A/dt /(2 H R_A)); with a∝t,
  dR_A/dt=1, H R_A=1 => factor (1-1/2)=1/2 -- a TEMPERATURE 1/2 (session-2 territory),
  in the WRONG direction for the entropy and scheme-dependent. So no clean rescue.""")
# numeric check of the dynamical-horizon temperature factor in coasting:
p = sp.Symbol("p", positive=True)   # a ∝ t^p ; coasting p=1
Rdot_over = sp.Rational(1, 2) / p   # dR_A/dt /(2 H R_A) = 1/(2p)
T_factor = 1 - Rdot_over
print(f"\n  dynamical-horizon T factor (a∝t^p): 1 - 1/(2p);  coasting p=1 -> {T_factor.subs(p,1)}"
      f"   (a forced 1/2 in T, not S; wrong direction)")
ba4 = True

# ----------------------------------------------------------------------
# VERDICT
# ----------------------------------------------------------------------
head("VERDICT")
if FAIL:
    print(f"  HONEST NEGATIVE — verifiable-claim aborts tripped: {FAIL}")
    sys.exit(1)

disposition = "FORCED-2 (flagship closes)" if ba3_forced else "NOT-FORCED (gravity form-level)"
print(f"""  DISPOSITION: {disposition}

  The 2-sphere boundary reopener does NOT close the factor of 2. Two results:

  (1) SHARP STRUCTURAL REQUIREMENT (B-A1, sympy). Closure is equivalent to the
      gravitating apparent-horizon entropy being EXACTLY twice the observer's
      worldline information record: S_grav = 2 S_record. If they are the same
      object (the framework's current identification, S_grav = S_record = N), then
      G_eff = 2G. So the reopener is not "use a bigger entropy" loosely -- it needs
      a forced factor of exactly 2 between two specific quantities.

  (2) NO GEOMETRIC FORCING (B-A2..B-A4). Splitting S_grav from S_record costs the
      framework's prized 'OEF delta E_obs = kappa dS_total IS the Clausius relation'
      identification (which equates them). And no boundary-geometry factor forces
      the 2: the candidates that give 2 (diameter-vs-radius, 4pi-vs-2pi) are each
      defeated by a physical objection (the observer's one-way causal age is R_A;
      a central observer sees the full sphere). The non-defeated reading -- gravity
      sees the horizon MUTUAL INFORMATION (2 S_record) rather than the entanglement
      entropy (1 S_record) -- is a CHOICE: standard holography gravitates the
      entanglement entropy (1x), and picking the mutual information to hit G_eff=G
      is goal-seeking. A dynamical-horizon temperature does carry a forced 1/2 in
      coasting, but in T (not S) and the wrong direction.

  CONCLUSION: c_S = 2 is not forced by the 2-sphere boundary. Combined with the
  prior probes, the factor of 2 is now investigated exhaustively across every route
  (work-density R1; entropy-normalization R2; blind c_S trilemma->dilemma;
  extent-vs-flux; 2-sphere boundary), and each leaves the framework's own accounting
  at c_S = 1 -> G_eff = 2G, with c_S = 2 only obtainable by adoption.

  DISPOSITION (final, unchanged):
    - kappa NOT promotable to predictions/; predictions/G_N.py stays
      asymptotic-safety-conditional.
    - Gravity is promotable at the FORM level (emergent Lorentzian metric; emergent
      standard Friedmann + coasting from the native information-Clausius relation;
      the entropy-temperature compensation that fixes H^2 ∝ rho is c_S-INDEPENDENT
      and robust). Only the coupling MAGNITUDE (Newton's G parameter-free) does not
      close; it is calibration-fixed.

  The genuinely open theoretical question, now precisely stated: is there a derived
  reason the gravitating horizon entropy is the boundary mutual information (2x the
  observer record) rather than the entanglement entropy (1x)? That is a foundational
  question in the emergent-gravity program the framework inherits, not a framework
  calculation -- and it must be answered blind to G_eff=G, or it is goal-seeking.
""")
print("=" * 78)
print(f"  EXIT 0 — {disposition}; c_S=2 not forced by boundary geometry")
print("=" * 78)
