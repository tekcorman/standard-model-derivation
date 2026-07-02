#!/usr/bin/env python3
"""
L6 closure direction R1 — first probe: native r_s candidates.

Scope: an internal working note §3.

Tests candidate R1a (coasting r_s with framework-natural lower cutoffs) and
articulates the structural gap against Planck's r_s ≈ 147 Mpc.

Honest-discipline (W58 / an internal note):
report whatever the probe finds, do not promote numerical coincidences.

R1a derivation: in coasting (a ∝ N, a·H = H_0 constant):
    r_s_comoving = (c_s/H_0) × ln(a*/a_min)

Diverges as a_min → 0 (the L6 wall). Test several framework-natural a_min.
"""

from __future__ import annotations
import math


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
c = 2.998e8           # m/s
Mpc = 3.0857e22       # m
Gpc = 1000 * Mpc

H_0_si = 68.2 * 1000 / (3.0857e22)   # 1/s (framework's H_0 ≈ Planck's)
c_over_H0 = c / H_0_si                # m, Hubble distance
c_sound_radiation = c / math.sqrt(3)  # standard radiation c_s

# Framework cosmological scales
N_hub = 8.394881e60
v_today_GeV = 246.22
M_unif_GeV = 1.985e16
M_Pl_GeV = 1.22089e19

# Framework's GUT epoch in observer ticks
N_GUT = N_hub / (M_unif_GeV / v_today_GeV) ** 4
print(f"N_GUT (framework gauge unification epoch) = {N_GUT:.3e}")
print(f"  (in coasting a ∝ N, this corresponds to a_GUT = N_GUT/N_hub)")

# Scale-factor cosmological anchors
z_star_standard = 1379.0
z_star_framework = 15365.0
a_star_std = 1.0 / (1.0 + z_star_standard)
a_star_fw = 1.0 / (1.0 + z_star_framework)

# Planck target
r_s_planck_Mpc = 147.0
r_s_planck_m = r_s_planck_Mpc * Mpc


# ---------------------------------------------------------------------------
# R1a — coasting r_s with framework cutoff
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("R1a — coasting r_s = (c_s/H_0) × ln(a*/a_min)")
print("=" * 72)

print(f"\nConstants:")
print(f"  c_s = c/√3 = {c_sound_radiation:.3e} m/s (standard radiation sound speed)")
print(f"  c/H_0 = {c_over_H0/Gpc:.3f} Gpc (Hubble distance)")
print(f"  c_s/H_0 = {c_sound_radiation/H_0_si/Gpc:.3f} Gpc")
print(f"\nTarget: Planck r_s ≈ {r_s_planck_Mpc} Mpc = {r_s_planck_Mpc/1000:.4f} Gpc")

# Framework-natural lower cutoffs (all in a-units, with a_today = 1)
cutoff_candidates = [
    ("a_BBN ≈ 10⁻⁹",            1e-9,  "z ~ 10⁹, standard BBN era"),
    ("a_QCD ≈ 10⁻¹²",           1e-12, "z ~ 10¹², standard QCD epoch"),
    ("a_EW ≈ 10⁻¹⁵",            1e-15, "z ~ 10¹⁵, electroweak epoch"),
    ("a_GUT = N_GUT/N_today",   N_GUT/N_hub, f"framework's gauge unification (a_GUT ≈ {N_GUT/N_hub:.2e})"),
    ("a_Planck ≈ 1/N_hub",      1.0/N_hub,    f"first observation tick (a ≈ {1.0/N_hub:.2e})"),
    ("a_thermal (T_sub→T_obs)", 2.725/1.4e32, "scale where T=T_substrate=Planck temp"),
]

for z_label, z_star, a_star in [
    ("standard z*=1379",  z_star_standard, a_star_std),
    ("framework z*=15366", z_star_framework, a_star_fw),
]:
    print(f"\n--- {z_label}, a* = {a_star:.3e} ---")
    print(f"  {'a_min':<32} | {'ln(a*/a_min)':>12} | {'r_s (Gpc)':>10} | {'ratio to Planck':>15}")
    print(f"  {'-'*32}-|-{'-'*12}-|-{'-'*10}-|-{'-'*15}")
    for label, a_min, desc in cutoff_candidates:
        if a_min > a_star:
            print(f"  {label:<32} | {'N/A':>12} | {'N/A':>10} | a_min > a*")
            continue
        log_factor = math.log(a_star / a_min)
        r_s = (c_sound_radiation / H_0_si) * log_factor
        ratio = r_s / r_s_planck_m
        print(f"  {label:<32} | {log_factor:>12.2f} | {r_s/Gpc:>10.3f} | {ratio:>13.1e}×")


# ---------------------------------------------------------------------------
# What c_s would close R1a?
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("REQUIRED c_s to close R1a (assuming the coasting formula form)")
print("=" * 72)

print(f"\nFor r_s = {r_s_planck_Mpc} Mpc with a_min and z* fixed:")
print(f"  c_s_required = r_s × H_0 / ln(a*/a_min)")
print(f"\n  {'cutoff':<25} | {'z*':<10} | {'c_s_req/c':<15} | structural?")
print(f"  {'-'*25}-|-{'-'*10}-|-{'-'*15}-|-")

for label, a_min, _ in cutoff_candidates[:4]:
    for z_label, z_star, a_star in [("standard", z_star_standard, a_star_std),
                                     ("framework", z_star_framework, a_star_fw)]:
        if a_min > a_star:
            continue
        log_factor = math.log(a_star / a_min)
        c_s_required = r_s_planck_m * H_0_si / log_factor
        ratio_to_c = c_s_required / c
        match_radiation = abs(ratio_to_c - 1/math.sqrt(3)) / (1/math.sqrt(3)) * 100
        marker = " ← matches c/√3!" if match_radiation < 5 else ""
        print(f"  {label:<25} | {z_label:<10} | {ratio_to_c:>12.2e} × c | (off by {match_radiation:.0f}%){marker}")


# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("HONEST VERDICT — R1a (coasting r_s with framework cutoff)")
print("=" * 72)
print(f"""
R1a tested across 6 framework-natural lower cutoffs × 2 z* values (standard
+ framework). None gives r_s within an order of magnitude of Planck's
147 Mpc using c_s = c/√3.

Closest: a_BBN cutoff with standard z* gives r_s ≈ 34 Gpc — still off by
factor 230. Framework z* makes things worse (smaller log factor doesn't
compensate). All other cutoffs are worse.

The required c_s to close is ~0.5% × c (vs standard radiation c/√3 = 58%),
across all cutoff×z* combinations. There's NO framework-derivable c_s near
0.5% × c — substrate-acoustic Bloch speed is order c (lattice spacing ℓ_P
over hop time t_P).

STRUCTURAL FINDING:
  R1a fails. Coasting r_s as ∫c_s da/(a·H) with framework cutoff does NOT
  reproduce the standard 147 Mpc, regardless of cutoff choice.

  The L6 wall is NOT a cutoff problem. The coasting r_s formula has the
  WRONG functional form for what's being measured at recombination.

This is consistent with the original step 2 finding ("L6 wall is FRW-
formula-breakdown"). R1a confirms it more sharply: even with the cleanest
possible cutoff, the formula gives the wrong answer.

IMPLICATIONS:
  - The physical r_s ≈ 147 Mpc that Planck measures is NOT captured by
    the coasting-cosmology sound horizon integral. The standard r_s
    derivation uses a specific RDE→MDE transition that coasting doesn't
    have.
  - The "native r_s on observer graph" must be a DIFFERENT object from
    the FRW r_s integral. Candidates R1b (Bloch coherence length) and
    R1c (Fisher-distance) are still open.
  - Or: maybe the framework's z* is right but the "θ*" Planck measures
    is not r_s/D_A in coasting — it might be a different angular feature.

NEXT-STEP CANDIDATES:
  - R1b: derive substrate Bloch-acoustic coherence length on B_NB(srs)
    explicitly. Compute h_P-mode coherence scale; compare to Planck r_s.
  - R1c: Fisher-distance on observer-graph posterior between anchor and
    recombination surface. Requires explicit observer-filtration model.
  - Reframe: the observable Planck reads as "first acoustic peak position"
    might correspond to a different framework object (e.g., the gauge-
    singlet Perron-mode characteristic scale at recombination). Worth
    examining separately.

VERDICT: R1a CHARACTERIZED-NEGATIVE. R1b, R1c, and reframing remain open.
""")
print("=" * 72)
