#!/usr/bin/env python3
"""
proofs/foundations/BR4_session5_chirality_flip_holonomy_2026-05-27.py

BR4 Session 5 — Chirality-flip / girth-ring holonomy intertwiner candidate
                (direction γ per Session 3-4 recommendations)

PURPOSE
-------
Final BR4 candidate test (per entry-point §11 threshold). M_persistence
(`theorem_fermion_mass_operator_persistence_2026-05-21.md`) frames mass
as girth-ring holonomy of L↔R chirality oscillation on the srs↔srs-z
double cover. The framework's per-species walker types use specific
Bloch saddles:

  Walker III (charged lepton, L=g-2=8): chir-5/3 at P, h_P = (√3+i√5)/2
  Walker IV (down-type, L=g=10):       chir-5/3 at P
  Walker II (up-type, L=0):            saturation, no walker traversal
  Walker I (neutrino, asymptotic):     chir-7 at Γ/H, h_Γ = (-1+i√7)/2

W73's lepton 0.3% near-match: δ_lepton = (π - arg(h_P))/g matches 2/9 rad
to 0.3%. This session tests whether the same chirality-flip rule provides
a STRUCTURAL derivation of δ_quark for the down-sector (where W73 gave
~10-15% match) and the up-sector (where W73 had no clean formula).

STRUCTURAL CANDIDATE (direction γ):
-----------------------------------
For a circulant Hermitian M_gen on C³_obs:
  M_gen = M_0·(I + ε·X + ε·X²·e^(i·δ))
where X is the cyclic shift and δ is the Koide phase.

Eigenvalues: m_k = M_0·|1 + ε·e^(iδ + 2πi(k-1)/3)|² (Koide-cosine form).

Candidate: δ_species comes from the chirality-flip walker phase per
girth step. For chir-5/3 species, the walker eigenvalue is h_P = √2·e^(iφ)
where φ = arctan(√(5/3)). After L=g steps, total phase is g·φ. The
"per-step deficit from π" — i.e., (π - φ) — divided by g, gives the
within-species δ:
  δ_species = (π - φ)/g_eff(species)
where g_eff(lepton) = g - 0 (use g=10 for the W73 formula) or some other
species-dependent factor.

PRE-FLIGHT CHECKS
-----------------
- AB5: does the candidate reduce to substrate C_3-equivariant?
       The chirality-flip operator on srs↔srs-z is by construction
       NOT C_3-equivariant — it's a holonomy, sensitive to walker
       path orientation, broken under the C_3 anti-action via γ_7.
       So AB5 should PASS in principle.

- AB6: does W (= circulant M_gen with chirality-flip-derived δ)
       commute with σ_C3? Yes — circulant M_gen commutes with σ_C3 by
       construction. This IS the W75 obstruction.

       BUT: the W75 obstruction was about C_3 ISOTYPIC decomposition
       killing complex-conjugate-pair closure. For WITHIN-species M_gen
       on C³_obs, circulant structure is REQUIRED by the M_gen
       non-degeneracy theorem (Galois-invariant Hermitian on C³_obs ARE
       circulant). The δ phase is the off-diagonal phase, encoded as the
       imaginary part of the circulant parameter b.

       So circulant M_gen is the RIGHT structural form for within-species
       — it gives 3 distinct masses with Koide-cosine pattern.

- AB2: do the eigenvalues match Koide pattern with framework ε² values?
       Test this directly by computing m_k from circulant M_gen with
       (ε², δ) inputs.

Run with:
    python3 proofs/foundations/BR4_session5_chirality_flip_holonomy_2026-05-27.py
"""

import numpy as np

TOL = 1e-9


# ---------------------------------------------------------------------------
# Framework Ramanujan saddle phases (theorem-grade)
# ---------------------------------------------------------------------------

# P-saddle (chir-5/3, used for charged lepton + down + V_us, V_cb, V_ub)
h_P = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
phi_P = np.angle(h_P)                       # arctan(√5/√3) = arctan(√(5/3))

# Γ/H saddle (chir-7, used for neutrino)
h_Gamma = (-1 + 1j * np.sqrt(7)) / 2
phi_Gamma = np.angle(h_Gamma)               # = π - arctan(√7) for h_Γ

# N-saddle (chir-3/5, different sector)
h_N = (np.sqrt(5) + 1j * np.sqrt(3)) / 2
phi_N = np.angle(h_N)

# Framework girth
g = 10
omega = np.exp(2j * np.pi / 3)


# ---------------------------------------------------------------------------
# Test 1: W73-style chirality-flip rule for δ_species
# ---------------------------------------------------------------------------

print("=" * 76)
print("BR4 Session 5 — Chirality-flip holonomy candidate (direction γ)")
print("=" * 76)
print()
print("Test 1 — W73 chirality-flip rule δ = (π - arg(h_saddle)) / g:")
print()
print(f"  Framework saddles:")
print(f"    h_P (chir-5/3):     {h_P:+.4f}    arg = {np.degrees(phi_P):.4f}°")
print(f"    h_Γ (chir-7):       {h_Gamma:+.4f}   arg = {np.degrees(phi_Gamma):.4f}°")
print(f"    h_N (chir-3/5):     {h_N:+.4f}    arg = {np.degrees(phi_N):.4f}°")
print()

# Lepton: chir-5/3, formula gives 12.78° vs framework 2/9 rad = 12.73°
delta_lepton_W73 = (np.pi - phi_P) / g
delta_lepton_framework = 2/9   # Bernoulli identity
print(f"  Charged lepton (Walker III, L=g-2=8, chir-5/3):")
print(f"    Candidate δ = (π - arg(h_P))/g = {np.degrees(delta_lepton_W73):.4f}°")
print(f"    Framework δ_lepton = 2/9 rad   = {np.degrees(delta_lepton_framework):.4f}°")
print(f"    Δ = {np.degrees(delta_lepton_W73 - delta_lepton_framework):+.4f}° "
      f"({100*(delta_lepton_W73 - delta_lepton_framework)/delta_lepton_framework:+.3f}%)")
print()

# Down: chir-5/3, formula gives same value 12.78° but empirical δ_down ~ 5-7°
delta_down_W73 = (np.pi - phi_P) / g
print(f"  Down quark (Walker IV, L=g=10, chir-5/3):")
print(f"    Candidate δ = (π - arg(h_P))/g = {np.degrees(delta_down_W73):.4f}°")

# Empirical down δ via Koide cosine fit to PDG (m_d, m_s, m_b)
m_d_PDG = 4.67   # MeV (2 GeV MS-bar)
m_s_PDG = 93.4
m_b_PDG = 4180


def koide_delta_from_masses(m1, m2, m3):
    """Extract Koide δ from 3 within-species masses via cos parametrization.
    m_i = M_0·(1 + ε·cos(δ + 2π(i-1)/3))²
    M_0 + ε combine via 3-mass identity; extract δ via phase of complex
    moment Σ_i √m_i · ω^(i-1)."""
    sqrtm = np.array([np.sqrt(m1), np.sqrt(m2), np.sqrt(m3)])
    moment = sum(sqrtm[i] * omega**i for i in range(3))
    # δ is the phase of this complex moment (up to convention)
    return np.angle(moment)


delta_down_empirical = koide_delta_from_masses(m_d_PDG, m_s_PDG, m_b_PDG)
print(f"    Empirical δ_down (Koide fit) = {abs(np.degrees(delta_down_empirical)):.4f}°")
print(f"    Candidate vs empirical: {np.degrees(delta_down_W73):.4f}° vs "
      f"{abs(np.degrees(delta_down_empirical)):.4f}°")
print()

# Up: Walker II saturation, no chir-5/3 walker traversal; W73 had no formula
m_u_PDG = 2.16   # MeV
m_c_PDG = 1270
m_t_PDG = 172570

delta_up_empirical = koide_delta_from_masses(m_u_PDG, m_c_PDG, m_t_PDG)
print(f"  Up quark (Walker II, L=0 saturation):")
print(f"    No chir-5/3 walker traversal → W73 rule N/A for δ_up.")
print(f"    Empirical δ_up (Koide fit) = {abs(np.degrees(delta_up_empirical)):.4f}°")
print()


# ---------------------------------------------------------------------------
# Test 2: Compute Koide masses with W73-derived δ_down + framework ε²_down
# ---------------------------------------------------------------------------
# Framework: ε²_down = 5/2 (W53 Type IV walker)
# Use M_0 anchored to m_b (gen 3) and δ_down = (π - arg(h_P))/g
# Predict m_d, m_s and compare to PDG

print("Test 2 — Koide reconstruction with framework ε² + W73 δ:")
print()


def koide_masses(M_0, eps, delta):
    """Generate 3 masses from Koide-cosine parametrization."""
    return [M_0 * (1 + eps * np.cos(delta + 2*np.pi*k/3))**2 for k in range(3)]


# Down sector: ε²_down = 5/2 → ε_down = √(5/2)
eps_down = np.sqrt(5/2)
delta_down_candidate = (np.pi - phi_P) / g

# Anchor M_0 to reproduce m_b (gen 3)
# m_b = M_0·(1 + ε·cos(δ + 4π/3))²
# But which generation index corresponds to which physical mass?
# Largest mass should be cos closest to +1, i.e., cos(δ + 2π·k/3) max.

# Try all 3 mass-orderings to find which gives best fit
best_dev = np.inf
best_k_b = None
for k_b in range(3):
    M_0_trial = m_b_PDG / (1 + eps_down * np.cos(delta_down_candidate + 2*np.pi*k_b/3))**2
    masses_trial = koide_masses(M_0_trial, eps_down, delta_down_candidate)
    # Match (m_d, m_s, m_b) to the 3 generated masses (in some order)
    masses_sorted = sorted(masses_trial)
    # PDG order m_d < m_s < m_b
    pdg_sorted = sorted([m_d_PDG, m_s_PDG, m_b_PDG])
    dev = sum((masses_sorted[i] - pdg_sorted[i])**2 / pdg_sorted[i]**2
              for i in range(3))
    if dev < best_dev:
        best_dev = dev
        best_k_b = k_b
        best_M_0 = M_0_trial

masses_pred = koide_masses(best_M_0, eps_down, delta_down_candidate)
masses_pred_sorted = sorted(masses_pred)
pdg_sorted = sorted([m_d_PDG, m_s_PDG, m_b_PDG])

print(f"  Down-sector with ε²=5/2, δ=(π-arg(h_P))/g (W73 candidate):")
print(f"    δ_down candidate = {np.degrees(delta_down_candidate):.4f}°")
print(f"    M_0 (anchor)      = {best_M_0:.4f} MeV (chose gen index k_b = {best_k_b})")
print()
print(f"    {'mass':>10s} {'PDG':>12s} {'predicted':>12s} {'dev':>10s}")
print(f"    {'m_d':>10s} {pdg_sorted[0]:>12.4f} {masses_pred_sorted[0]:>12.4f} "
      f"{100*(masses_pred_sorted[0] - pdg_sorted[0])/pdg_sorted[0]:>+9.2f}%")
print(f"    {'m_s':>10s} {pdg_sorted[1]:>12.4f} {masses_pred_sorted[1]:>12.4f} "
      f"{100*(masses_pred_sorted[1] - pdg_sorted[1])/pdg_sorted[1]:>+9.2f}%")
print(f"    {'m_b':>10s} {pdg_sorted[2]:>12.4f} {masses_pred_sorted[2]:>12.4f} "
      f"{100*(masses_pred_sorted[2] - pdg_sorted[2])/pdg_sorted[2]:>+9.2f}%")
print()


# ---------------------------------------------------------------------------
# Test 3: AB5/AB6 on circulant Hermitian M_gen
# ---------------------------------------------------------------------------

print("Test 3 — AB5/AB6 on circulant Hermitian M_gen:")
print()

# Circulant Hermitian M_gen on C³_obs with a real diagonal and complex off-diag
a = best_M_0 * (1 + eps_down**2/2)
b = best_M_0 * eps_down * np.exp(1j * delta_down_candidate)
M_gen = a * np.eye(3, dtype=complex)
sigma_C3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
sigma_C3_sq = sigma_C3 @ sigma_C3
M_gen += b * sigma_C3 + b.conjugate() * sigma_C3_sq

# Verify it's Hermitian (Hermitian circulant)
hermitian = np.allclose(M_gen, M_gen.conj().T, atol=TOL)
print(f"  M_gen built as a·I + b·σ + b*·σ² (circulant Hermitian)")
print(f"  Hermitian: {hermitian}")

# Eigenvalues — should give 3 real values matching down sector
eigvals_M = np.linalg.eigvalsh(M_gen)
print(f"  M_gen eigenvalues: {sorted(np.real(eigvals_M))}")
print(f"  These should match (m_d, m_s, m_b) = {pdg_sorted}")
print()

# AB5/AB6 checks
def isotypic_proj_3(omega_power):
    P = np.zeros((3, 3), dtype=complex)
    for k in range(3):
        P += (omega ** (-k * omega_power)) * np.linalg.matrix_power(sigma_C3, k)
    return P / 3


P_1 = isotypic_proj_3(0)
P_omega = isotypic_proj_3(1)
P_omegabar = isotypic_proj_3(2)

comm_sigma = M_gen @ sigma_C3 - sigma_C3 @ M_gen
comm_P1 = M_gen @ P_1 - P_1 @ M_gen
comm_Pom = M_gen @ P_omega - P_omega @ M_gen
comm_Pombar = M_gen @ P_omegabar - P_omegabar @ M_gen

print(f"  AB5/AB6 commutation checks (within-species M_gen on C³_obs):")
print(f"    ‖[M_gen, σ_C3]‖_∞ = {np.abs(comm_sigma).max():.3e}")
print(f"    ‖[M_gen, P_1]‖_∞  = {np.abs(comm_P1).max():.3e}")
print(f"    ‖[M_gen, P_ω]‖_∞  = {np.abs(comm_Pom).max():.3e}")
print(f"    ‖[M_gen, P_ω̄]‖_∞  = {np.abs(comm_Pombar).max():.3e}")
print()

if np.abs(comm_sigma).max() < 1e-9:
    print("  M_gen IS circulant Hermitian → commutes with σ_C3 (by construction).")
    print()
    print("  STRUCTURAL OBSERVATION:")
    print("    The framework's M_gen non-degeneracy theorem REQUIRES circulant")
    print("    structure (Galois-invariant Hermitian on C³_obs are circulant).")
    print("    'Commutes with σ_C3' is therefore a STRUCTURAL FEATURE for")
    print("    within-species M_gen, not a bug.")
    print()
    print("    AB6 was framed for the BR4 W-vertex intertwiner (cross-species)")
    print("    where circulant W would fail. For the WITHIN-species M_gen,")
    print("    circulant is by structural requirement.")
    print()
    print("    The δ_quark phase is encoded in the COMPLEX ARGUMENT of the")
    print("    off-diagonal b parameter. The structural question is what fixes")
    print("    arg(b) = δ_species.")


# ---------------------------------------------------------------------------
# Summary verdict
# ---------------------------------------------------------------------------

print()
print("=" * 76)
print("VERDICT")
print("=" * 76)
print()

print("  Direction (γ) chirality-flip via M_persistence chir-5/3 status:")
print()
print(f"  1. Lepton match (W73 δ=12.78° vs framework δ_lepton=12.73°): 0.3% — recorded")
print(f"  2. Down match (W73 δ=12.78° vs empirical δ_down "
      f"= {abs(np.degrees(delta_down_empirical)):.2f}°): NOT clean")
print(f"  3. Up match: no formula (Walker II saturation)")
print()
print(f"  Down-sector mass reconstruction with ε²=5/2 + W73 δ:")
print(f"    m_d off by {100*(masses_pred_sorted[0] - pdg_sorted[0])/pdg_sorted[0]:+.1f}%")
print(f"    m_s off by {100*(masses_pred_sorted[1] - pdg_sorted[1])/pdg_sorted[1]:+.1f}%")
print(f"    m_b anchored")
print()
print("  AB checks (for within-species M_gen, not cross-species W-vertex):")
print("    M_gen is circulant Hermitian by framework structural requirement.")
print("    AB5/AB6 reframed: within-species M_gen is C_3-equivariant by")
print("    design (M_gen non-degeneracy theorem). The δ phase emerges from")
print("    the off-diagonal COMPLEX argument, not from breaking circulant")
print("    structure.")
print()
print("  STRUCTURAL FINDING: The 'chirality-flip / girth-ring holonomy'")
print("  framing reproduces δ_lepton at 0.3% (W73). For δ_quark, the same")
print("  formula gives the lepton value (12.78°) but empirical δ_down is")
print(f"  ~{abs(np.degrees(delta_down_empirical)):.0f}° — the rule does not generalize across species.")
print()
print("  The framework's M_persistence theorem (§3.2 'M = shape ∘ dynamics')")
print("  identifies δ_species as encoded in the WITHIN-SPECIES walker phase")
print("  pattern, but the SPECIFIC species-dependent δ_quark formula remains")
print("  OPEN at the chirality-flip framing.")
print()
print("  CONCLUSION: direction (γ) provides the structural FRAMEWORK")
print("  (chirality-flip on srs↔srs-z double cover) but not the specific")
print("  derivation of δ_quark per species.")
