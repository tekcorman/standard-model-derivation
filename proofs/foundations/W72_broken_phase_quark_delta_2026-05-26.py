#!/usr/bin/env python3
"""
W72 (extended) — Broken-phase quark δ via V_triv axis rotation + κ alignment

Per user correction after W70/W71: stay in BROKEN PHASE; use the
framework's V_triv axis rotation between lepton-axis e_0 and quark-axis
(e_1+e_2+e_3)/√3. Tests:
  (a) V_triv projection of chir-7 girth phase gives δ per species
  (b) κ broken-phase coupling from rotation alignment + W52's φ
  (c) Lepton's parallel mechanism via chir-5/3 at P-fiber
  (d) Up-quark with γ_7 = +1
  (e) Full 3-generation consistency check (m_b extraction alone is
      MISLEADING — need to verify δ predicts m_s and m_d too)

PRE-DECLARED GATES (extended):
  G1: V_triv basis orthonormal
  G2: chir-7 axis = √3/2·ê_0 − 1/2·ê_cycled
  G3: arg(h_Γ^g) = 27° at girth-10 chir-7
  G4: lepton |proj − δ_lepton| (sanity)
  G5: up vs down projections differ via γ_7
  G6: κ candidates in sensible range
  G7: κ near V_us factor-of-2
  G8 (NEW): predicted m_s, m_d from W72 δ_down match empirical
  G9 (NEW): predicted m_c, m_u from W72 δ_up match empirical
  G10 (NEW): chir-5/3 projection at P gives lepton δ ≈ 2/9
  G11 (NEW): full 3-generation consistency for empirical δ extraction

Per W58: enumerate and report; no reverse-fitting.
"""

from __future__ import annotations
import math
import cmath
import numpy as np

gates = []
def gate(name, passed, detail=""):
    gates.append((name, bool(passed)))
    flag = "PASS" if passed else "FAIL"
    print(f"  [{flag}] {name}")
    if detail:
        for line in detail.strip("\n").split("\n"):
            print(f"         {line}")
    print()


print("=" * 78)
print("W72 (extended) — Broken-phase quark δ via V_triv axis rotation")
print("=" * 78)
print()


# ──────────────────────────────────────────────────────────────────
# §1 — V_triv basis vectors at Γ
# ──────────────────────────────────────────────────────────────────
e_0 = np.array([1, 0, 0, 0], dtype=float)
e_cycled = np.array([0, 1, 1, 1], dtype=float) / math.sqrt(3)

g1_pass = (abs(np.dot(e_0, e_0) - 1) < 1e-12 and
           abs(np.dot(e_cycled, e_cycled) - 1) < 1e-12 and
           abs(np.dot(e_0, e_cycled)) < 1e-12)
gate("G1 V_triv basis vectors orthonormal", g1_pass)


# ──────────────────────────────────────────────────────────────────
# §2 — A(Γ) V_triv eigenvectors at λ=+3 (real h) and λ=−1 (chir 7)
# ──────────────────────────────────────────────────────────────────
v_plus3 = np.array([1, 1, 1, 1], dtype=float) / 2
v_minus1 = np.array([3, -1, -1, -1], dtype=float) / (2 * math.sqrt(3))

proj_lepton_chir7 = np.dot(e_0, v_minus1)
proj_quark_chir7 = np.dot(e_cycled, v_minus1)
g2_pass = (abs(proj_lepton_chir7 - math.sqrt(3) / 2) < 1e-12 and
           abs(proj_quark_chir7 - (-0.5)) < 1e-12)
gate("G2 chir-7 axis = √3/2·ê_0 − 1/2·ê_cycled", g2_pass)


# ──────────────────────────────────────────────────────────────────
# §3 — Chir-7 girth-g holonomy phase
# ──────────────────────────────────────────────────────────────────
h_Gamma = complex(-1, math.sqrt(7)) / 2
arg_h_Gamma = cmath.phase(h_Gamma)
GIRTH = 10
phase_chir7_g = ((arg_h_Gamma * GIRTH + math.pi) % (2 * math.pi)) - math.pi

g3_pass = abs(math.degrees(phase_chir7_g) - 27) < 1.0
gate("G3 arg(h_Γ^g) ≈ 27°", g3_pass,
     f"got {math.degrees(phase_chir7_g):+.4f}°")


# ──────────────────────────────────────────────────────────────────
# §4 — Projections per species
# ──────────────────────────────────────────────────────────────────
phase_proj_lepton_rad = math.radians(math.degrees(phase_chir7_g)) * proj_lepton_chir7
phase_proj_quark_rad = math.radians(math.degrees(phase_chir7_g)) * proj_quark_chir7

# γ_7 split: up has γ_7 = +1, down has γ_7 = −1
phase_proj_up_rad = +1 * phase_proj_quark_rad
phase_proj_down_rad = -1 * phase_proj_quark_rad

delta_lepton_target = 2 / 9

print(f"§4 — Per-species projections (W72 mechanism):")
print(f"  Lepton (ê_0):      proj = cos(30°) × 27°  = {math.degrees(phase_proj_lepton_rad):+.4f}°")
print(f"  Up-quark (γ_7=+1): proj = cos(120°) × 27° = {math.degrees(phase_proj_up_rad):+.4f}°")
print(f"  Down-quark (γ_7=−1): proj = -cos(120°) × 27° = {math.degrees(phase_proj_down_rad):+.4f}°")
print(f"  Framework δ_lepton target = 2/9 = {math.degrees(delta_lepton_target):.4f}°")
print()

g4_pass = abs(phase_proj_lepton_rad - delta_lepton_target) < 0.2
gate("G4 lepton projection within 0.2 rad of 2/9",
     g4_pass,
     f"|proj − 2/9| = {abs(phase_proj_lepton_rad - delta_lepton_target):.4f} rad "
     f"({math.degrees(abs(phase_proj_lepton_rad - delta_lepton_target)):.2f}°)")

g5_pass = abs(phase_proj_up_rad - phase_proj_down_rad) > 0.05
gate("G5 up vs down differ structurally", g5_pass)


# ──────────────────────────────────────────────────────────────────
# §5 — Full 3-generation Koide consistency for empirical sectors
# ──────────────────────────────────────────────────────────────────
print("=" * 78)
print("§5 — Full 3-generation Koide consistency check (the load-bearing test)")
print("=" * 78)
print()

def koide_self_consistent_delta(masses, label):
    """
    Extract self-consistent (M_0, ε², δ) from 3 masses + verify
    Koide identity. Returns (M_0, eps_sq, delta_rad) or None if fit fails.
    """
    sqrt_m = [math.sqrt(m) for m in masses]
    sum_sqrt = sum(sqrt_m)
    sum_m = sum(masses)
    Q = sum_m / sum_sqrt**2  # Koide ratio (RG-invariant)
    eps_sq_from_Q = 6 * Q - 2  # ε² = 6Q − 2 (from Q = (1+ε²/2)/3)
    M_0 = sum_sqrt / 3

    # Extract δ: ratios r_j = (√m_j / M_0 − 1) / ε should = cos(2πj/3 + δ)
    eps = math.sqrt(abs(eps_sq_from_Q))
    if eps < 1e-6:
        return None
    ratios = [(sm / M_0 - 1) / eps for sm in sqrt_m]
    # Identity: Σ ratios = 0 (should hold)
    sum_ratios = sum(ratios)

    # For (m_j[0], m_j[1], m_j[2]) → (cos(δ), cos(δ+120°), cos(δ+240°)):
    # Use atan2 over Σ_j r_j × e^(i·2πj/3) which gives e^(-iδ) × 3/2
    sum_complex = sum(r * cmath.exp(2j * math.pi * j / 3) for j, r in enumerate(ratios))
    # sum_complex = (3/2) × e^(-iδ)  →  δ = -arg(sum_complex)
    if abs(sum_complex) < 1e-6:
        return None
    delta_rad = -cmath.phase(sum_complex)

    print(f"  {label} sector:")
    print(f"    masses: {masses}")
    print(f"    √m's:   {[f'{s:.4f}' for s in sqrt_m]}")
    print(f"    M_0 = {M_0:.4f}, Q = {Q:.6f}")
    print(f"    ε² (from Q via 6Q − 2) = {eps_sq_from_Q:.4f}")
    print(f"    ratios (√m/M_0 − 1)/ε = {[f'{r:+.4f}' for r in ratios]}")
    print(f"    Σ ratios = {sum_ratios:+.4f} (Koide identity says =0)")
    print(f"    extracted δ = {delta_rad:+.4f} rad = {math.degrees(delta_rad):+.4f}°")
    print()
    return (M_0, eps_sq_from_Q, delta_rad, ratios)


# Charged lepton (PDG pole masses, in MeV)
print(f"--- Charged lepton ---")
lepton_data = koide_self_consistent_delta(
    [0.511, 105.66, 1777.0], "Charged lepton")

# Down sector — using m_b(m_b)=4180, m_s(2GeV)=93.4, m_d(2GeV)=4.67
# NOTE: mixed schemes (m_b at m_b, m_s and m_d at 2 GeV).
# δ extraction is scheme-dependent; we use this as a representative point
print(f"--- Down sector (mixed scheme: m_b at m_b MS-bar; m_s, m_d at 2 GeV) ---")
down_data = koide_self_consistent_delta(
    [4.67, 93.4, 4180.0], "Down quark")

# Down sector — alternative: all at 2 GeV scale (m_b ≈ 4.888 GeV)
print(f"--- Down sector (consistent scheme: all at 2 GeV) ---")
down_data_2GeV = koide_self_consistent_delta(
    [4.67, 93.4, 4888.0], "Down quark @2GeV")

# Up sector — m_u(2GeV)=2.16, m_c(m_c)=1270, m_t(pole)=172690
# Mixed-scheme; up sector is messy due to pole vs MS-bar
print(f"--- Up sector (mixed scheme: pole-mass top) ---")
up_data = koide_self_consistent_delta(
    [2.16, 1270.0, 172690.0], "Up quark")


# ──────────────────────────────────────────────────────────────────
# §6 — Compare W72 predictions to self-consistent empirical δ
# ──────────────────────────────────────────────────────────────────
print("=" * 78)
print("§6 — Compare W72 V_triv projection predictions vs self-consistent empirical δ")
print("=" * 78)
print()

W72_pred = {
    "lepton": math.degrees(phase_proj_lepton_rad),
    "down": math.degrees(phase_proj_down_rad),
    "up": math.degrees(phase_proj_up_rad),
}

empirical = {}
if lepton_data:
    empirical["lepton"] = math.degrees(lepton_data[2])
if down_data:
    empirical["down (m_b scheme)"] = math.degrees(down_data[2])
if down_data_2GeV:
    empirical["down (2 GeV)"] = math.degrees(down_data_2GeV[2])
if up_data:
    empirical["up (mixed)"] = math.degrees(up_data[2])

print(f"  Sector              | W72 prediction | empirical (self-consistent) | Δ")
print(f"  {'-' * 18}-+-{'-' * 14}-+-{'-' * 27}-+-{'-' * 8}")
for sector, emp in empirical.items():
    base = sector.split()[0]
    pred = W72_pred.get(base, "—")
    if isinstance(pred, float):
        delta = pred - emp
        # Account for Koide-cosine 2π/3 ambiguity: equivalent δ differ by 120° shifts
        # Reduce to smallest |Δ| modulo 120°
        delta_reduced = (delta + 60) % 120 - 60
        print(f"  {sector:<18} | {pred:+8.4f}°     | {emp:+9.4f}°                  | "
              f"{delta:+7.2f}° (reduced mod 120: {delta_reduced:+.2f}°)")
    else:
        print(f"  {sector:<18} | {pred:<14} | {emp:+9.4f}°                  | —")
print()

# G8: down-sector match (any scheme within 5° = 0.087 rad)
g8_match_any_down = False
if "down (m_b scheme)" in empirical:
    delta_check = abs(W72_pred["down"] - empirical["down (m_b scheme)"])
    delta_reduced = abs((W72_pred["down"] - empirical["down (m_b scheme)"] + 60) % 120 - 60)
    if delta_reduced < 5:
        g8_match_any_down = True
if "down (2 GeV)" in empirical:
    delta_check = abs(W72_pred["down"] - empirical["down (2 GeV)"])
    delta_reduced = abs((W72_pred["down"] - empirical["down (2 GeV)"] + 60) % 120 - 60)
    if delta_reduced < 5:
        g8_match_any_down = True
gate("G8 W72 down prediction matches empirical δ_down within 5° (any scheme)",
     g8_match_any_down)

# G9: up-sector match
g9_match = False
if "up (mixed)" in empirical:
    delta_reduced = abs((W72_pred["up"] - empirical["up (mixed)"] + 60) % 120 - 60)
    if delta_reduced < 5:
        g9_match = True
gate("G9 W72 up prediction matches empirical δ_up within 5°", g9_match)


# ──────────────────────────────────────────────────────────────────
# §7 — Lepton parallel test: chir-5/3 at P-fiber
# ──────────────────────────────────────────────────────────────────
print("=" * 78)
print("§7 — Lepton parallel: chir-5/3 at P-fiber projection")
print("=" * 78)
print()

# h_P = (√3 + i√5)/2, |h_P|² = 2, chir 5/3
h_P = complex(math.sqrt(3), math.sqrt(5)) / 2
arg_h_P = cmath.phase(h_P)

# Lepton walker is Type III, L = g − 2 = 8
L_lepton = GIRTH - 2  # 8

# Phase at L=8: arg(h_P) × 8
phase_chir5over3_L8 = arg_h_P * L_lepton
phase_chir5over3_L8 = ((phase_chir5over3_L8 + math.pi) % (2 * math.pi)) - math.pi

# At P-fiber, V_triv eigenvalues are ±√3
# λ=+√3 eigenvector (analogous structure to Γ but with different coefficients)
# Since V_triv basis (ê_0, ê_cycled) is the same, the eigenvector for λ=+√3
# at P is the analogue of λ=+3 at Γ.
# At P, A|_V_triv = [[0, √3·c_P], [√3·c_P, ?]] with c_P some Bloch phase factor
# For simplicity (and matching the framework's K_4-quotient framing), use the
# K_4 V_triv eigenvector at λ=+√3 in the same basis form: linear combination
# of ê_0 and ê_cycled with coefficients (a, b) satisfying eigenvalue equation.

# In K_4 with A|_V_triv = [[0, √3], [√3, 2]], the eigenvectors are:
#   λ=+3: (1/2, √3/2) — fully sym, contains both color sectors
#   λ=−1: (√3/2, −1/2) — orthogonal, the chir-7 sector
# At P, the structure shifts but the (ê_0, ê_cycled) basis is fixed.
# For now, use the analogous v_P+ = (1/2)·ê_0 + (√3/2)·ê_cycled (same as Γ's λ=+3 vec)
# and check projection.

# Per the framework's theorem_color_singlet_P_concentration, lepton at P uses
# v_P+ (the symmetric V_triv combination), projected onto ê_0 (lepton-axis).
v_P_plus = 0.5 * e_0 + (math.sqrt(3) / 2) * e_cycled
proj_lepton_P = np.dot(e_0, v_P_plus)  # = 1/2

phase_proj_lepton_P = math.radians(math.degrees(phase_chir5over3_L8)) * proj_lepton_P

print(f"  h_P = (√3 + i√5)/2; arg(h_P) = {math.degrees(arg_h_P):+.4f}°")
print(f"  Lepton Type III, L = g − 2 = {L_lepton}")
print(f"  arg(h_P^L) = {math.degrees(phase_chir5over3_L8):+.4f}°")
print(f"  v_P+ = 1/2·ê_0 + √3/2·ê_cycled (V_triv at P, analogue of λ=+3 at Γ)")
print(f"  Projection of lepton on v_P+ = ⟨ê_0|v_P+⟩ = {proj_lepton_P:.4f}")
print(f"  Predicted δ_lepton = arg(h_P^L) × proj = {math.degrees(phase_proj_lepton_P):+.4f}°")
print(f"  Framework δ_lepton target = 2/9 = {math.degrees(delta_lepton_target):.4f}°")
print()

g10_pass = abs(phase_proj_lepton_P - delta_lepton_target) < 0.2
gate("G10 chir-5/3 projection at P gives δ_lepton ≈ 2/9 within 0.2 rad",
     g10_pass,
     f"|Δ| = {abs(math.degrees(phase_proj_lepton_P) - math.degrees(delta_lepton_target)):.4f}°")


# ──────────────────────────────────────────────────────────────────
# §8 — κ candidates from rotation alignment (test b)
# ──────────────────────────────────────────────────────────────────
print("=" * 78)
print("§8 — κ broken-phase coupling from rotation alignment + W52's φ")
print("=" * 78)

phi_K4 = math.acos(1.0 / 3.0)
kappa_candidates = {
    "κ_a = |proj_q|·sin(φ_K4)": abs(proj_quark_chir7) * math.sin(phi_K4),
    "κ_b = |proj_q|·(1−cos φ_K4)/2": abs(proj_quark_chir7) * (1 - math.cos(phi_K4)) / 2,
    "κ_c = |proj_q|·tan(φ_K4/2)": abs(proj_quark_chir7) * math.tan(phi_K4 / 2),
    "κ_d = |proj_q|·sin(φ_K4/2)": abs(proj_quark_chir7) * math.sin(phi_K4 / 2),
}

V_us_target = 9 / 40
g6_pass = all(0.001 < k < 1.0 for k in kappa_candidates.values())
g7_pass = any(0.5 < k / V_us_target < 2.0 for k in kappa_candidates.values())
print(f"  φ_K4 = arccos(1/3) = {math.degrees(phi_K4):.4f}°")
print(f"  V_us framework target = 9/40 = {V_us_target}")
print()
for label, k in kappa_candidates.items():
    print(f"  {label} = {k:.6f}  (ratio to V_us: {k/V_us_target:.4f})")
print()
gate("G6 κ candidates in [0.001, 1.0]", g6_pass)
gate("G7 some κ within factor-2 of V_us = 9/40", g7_pass)


# ──────────────────────────────────────────────────────────────────
# §9 — Predicted m_s, m_d from W72 δ_down vs empirical
# ──────────────────────────────────────────────────────────────────
print("=" * 78)
print("§9 — Predict m_s, m_d from W72 δ_down and compare to PDG")
print("=" * 78)
print()

# W72 says δ_down ≈ +13.52°. With ε² = 5/2 (framework W53 pin):
delta_W72_down = math.radians(13.5241)
eps_down = math.sqrt(5/2)

# M_0 from the LARGEST mass (m_b at m_b scale 4180 MeV):
m_b_obs = 4180.0
sqrt_mb = math.sqrt(m_b_obs)
# √m_b = M_0 (1 + ε cos(δ))
M_0_down = sqrt_mb / (1 + eps_down * math.cos(delta_W72_down))

# Predict m_s, m_d from j=1, j=2 (with m_b at j=0)
def koide_mass(j, M_0, eps, delta_rad):
    sqrt_m_signed = M_0 * (1 + eps * math.cos(2 * math.pi * j / 3 + delta_rad))
    return sqrt_m_signed ** 2  # take square regardless of sign

# Try different generation labelings to find best match
print(f"  Inputs: ε² = 5/2 (W53 pin), δ_down (W72) = +13.5241°")
print(f"  Anchor m_b = {m_b_obs} MeV (at m_b MS-bar scale)")
print(f"  M_0 (down) from m_b anchor: {M_0_down:.4f}")
print()
print(f"  Trial labelings (j=0 carries the m_b anchor):")
for label, j_strange, j_down in [("m_b=j0, m_s=j1, m_d=j2", 1, 2),
                                  ("m_b=j0, m_s=j2, m_d=j1", 2, 1)]:
    m_s_pred = koide_mass(j_strange, M_0_down, eps_down, delta_W72_down)
    m_d_pred = koide_mass(j_down, M_0_down, eps_down, delta_W72_down)
    print(f"    {label}:")
    print(f"      m_s predicted: {m_s_pred:.2f} MeV (PDG ≈ 93.4 MeV)")
    print(f"      m_d predicted: {m_d_pred:.4f} MeV (PDG ≈ 4.67 MeV)")
    print(f"      m_s ratio to PDG: {m_s_pred/93.4:.4f}")
    print(f"      m_d ratio to PDG: {m_d_pred/4.67:.4f}")
print()


# ──────────────────────────────────────────────────────────────────
# §10 — Verdict
# ──────────────────────────────────────────────────────────────────
print("=" * 78)
print("W72 — Final Verdict")
print("=" * 78)
n_pass = sum(1 for _, p in gates if p)
n_total = len(gates)
print(f"  {n_pass}/{n_total} gates pass")
for name, p in gates:
    print(f"  [{'PASS' if p else 'FAIL'}] {name}")
print()
print(f"  Per-sector δ comparison (W72 prediction vs empirical self-consistent):")
for sector, emp in empirical.items():
    base = sector.split()[0]
    pred = W72_pred.get(base, "—")
    if isinstance(pred, float):
        delta_reduced = (pred - emp + 60) % 120 - 60
        match = "MATCH" if abs(delta_reduced) < 5 else "MISS"
        print(f"    {sector}: W72={pred:+.2f}° vs empirical={emp:+.2f}° → reduced Δ={delta_reduced:+.2f}° [{match}]")
print()
