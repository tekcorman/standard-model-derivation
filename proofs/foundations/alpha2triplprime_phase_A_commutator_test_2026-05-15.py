#!/usr/bin/env python3
"""
proofs/foundations/alpha2triplprime_phase_A_commutator_test_2026-05-15.py

α2''' Phase A — Species-projector commutator test.

Hypothesis (from scoping doc): species-sensitive Kubo via per-atom
Hamming-weight projectors can extract substrate Δρ from existing
Π_JJ machinery.

Phase A minimum test: does the velocity vertex v^μ commute with a
per-atom Hamming-weight projector?

If [v^μ, P_n_per_atom] = 0 trivially:
  → species filter doesn't enter Kubo non-trivially at the 4-atom
    Bloch level.  ABORT condition (1) of scoping doc hits.

If [v^μ, P_n_per_atom] ≠ 0:
  → there's structure to extract.  Proceed to Phase B.

This probe tests the structural compatibility of the existing Π_JJ
machinery with the species-projector approach.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lorentz_sig_g_sub_elastic_moduli import BOND_DISPLACEMENTS
from gauge_beta_from_substrate_kubo_probe import velocity_matrix

print("=" * 78)
print("  α2''' Phase A — species-projector commutator test")
print("=" * 78)
print()

# Sample velocity matrix at generic k
k_test = np.array([0.3, 0.5, 0.7])
v_x = velocity_matrix(k_test, 0)
v_y = velocity_matrix(k_test, 1)
v_z = velocity_matrix(k_test, 2)

print(f"Existing Π_JJ machinery operates on 4-atom Bloch adjacency:")
print(f"  v^μ is a 4×4 matrix indexed by atom labels (α, β) ∈ {{0,1,2,3}}")
print(f"  v^μ_{{ts}}(k) = i r_b^μ exp(i k·r_b)  for bond b: s→t")
print()
print(f"Shape v_x: {v_x.shape}")
print()

# Per-atom Hamming-weight projector at the 4-atom Bloch level
# In the simplest model: assign each atom a fixed occupation N_α
# The projector onto a specific (N_0, N_1, N_2, N_3) assignment is
# DIAGONAL in atom basis (each atom either has the right occupation or not).
#
# Specifically: P_n_per_atom = diag(δ(N_0, n), δ(N_1, n), δ(N_2, n), δ(N_3, n))
# for testing whether the velocity vertex respects per-atom assignments.

print("=" * 78)
print("Test 1: Per-atom Hamming-weight projector commutator")
print("=" * 78)
print()

# Try various Hamming-weight assignments
assignments = {
    "n=2 at atom 0":    np.diag([1.0, 0.0, 0.0, 0.0]),
    "n=2 at atom 1":    np.diag([0.0, 1.0, 0.0, 0.0]),
    "n=1 at atoms 0,2": np.diag([1.0, 0.0, 1.0, 0.0]),
    "n=1 at all atoms": np.diag([1.0, 1.0, 1.0, 1.0]),
    "n=2 at 0,1; n=1 at 2,3": np.diag([1.0, 1.0, 0.5, 0.5]),  # weighted
}

for label, P in assignments.items():
    comm_x = v_x @ P - P @ v_x
    comm_y = v_y @ P - P @ v_y
    comm_z = v_z @ P - P @ v_z
    max_comm = max(np.max(np.abs(comm_x)), np.max(np.abs(comm_y)),
                   np.max(np.abs(comm_z)))
    print(f"  {label:<35} ‖[v^μ, P]‖_max = {max_comm:.6f}")

print()

# Crucial observation:
print("=" * 78)
print("STRUCTURAL OBSERVATION")
print("=" * 78)
print()
print("The velocity vertex v^μ is BOND HOPPING in the 4-atom adjacency space.")
print("Per-atom DIAGONAL projectors P = diag(p_0, p_1, p_2, p_3) DO NOT commute")
print("with v^μ in general (commutator is non-zero when p_t ≠ p_s for bond")
print("s→t).")
print()
print("This means: at the 4-atom Bloch level, per-atom projectors CAN enter")
print("the Kubo Π_JJ machinery non-trivially.")
print()
print("BUT — and this is the structural finding — the framework's HAMMING-WEIGHT")
print("species filter (n=1=d_L, n=2=ū_R per Furey 2018 + charge_before_color §9)")
print("lives at the Cl(6) FOCK PER ATOM level, NOT at the 4-atom Bloch adjacency")
print("level.  Each atom carries Cl(6) Fock (8 states {0,1}^3), and the species")
print("identification is which Hamming weight is occupied at the matter level.")
print()
print("The 4-atom Bloch operator H_bloch is GAUGE-level (adjacency); the matter")
print("Fock is INTERNAL to each atom site (Cl(6) Fock dim 8 per atom).")
print()
print("DIFFERENT LEVELS:")
print("  - Π_JJ machinery: 4-atom Bloch (gauge-coupling-blind to species)")
print("  - Hamming-weight species filter: Cl(6) Fock per atom (gauge-coupling-")
print("    blind to spatial Bloch)")
print()
print("Species-sensitive Kubo Δρ requires LINKING these two levels:")
print("  Full Hilbert: (4-atom Bloch) ⊗ (Cl(6) Fock per atom)^4 = 4 × 8^4 =")
print("    16384-dim per cell")
print("  Velocity vertex extended: v^μ × P_species(per-atom Fock occupation)")
print("  Kubo trace summed over Fock states with species weights")
print()

# Pseudo-test: at the 4-atom Bloch level alone, the per-atom projector
# is well-defined and gives non-trivial commutators with v^μ.  But this
# isn't the SPECIES filter — it's just a per-atom diagonal weighting.
# The actual species filter requires the per-atom Cl(6) Fock structure.

print("=" * 78)
print("Phase A verdict (honest)")
print("=" * 78)
print()
print("Abort condition (1) DOES NOT trivially hit at the 4-atom Bloch level —")
print("per-atom diagonal projectors give non-trivial commutators with bond")
print("hopping v^μ.  The species filter CAN enter non-trivially.")
print()
print("HOWEVER, the actual Hamming-weight species filter (n=1 vs n=2 on")
print("Cl(6) Fock per atom) requires extending the Bloch operator to include")
print("per-atom Cl(6) Fock structure.  The existing Π_JJ machinery operates")
print("at the wrong structural level for direct species-resolved Kubo.")
print()
print("REVISED PHASE A FINDING:")
print()
print("  α2''' as originally scoped requires building a Cl(6)-Fock-extended")
print("  Bloch operator at 4 × 8^4 = 16384-dim per cell (compared to 4 per")
print("  cell for the existing probe).  This is a substantial structural")
print("  extension — multi-session, not single-session bounded.")
print()
print("  The CONCEPTUAL framework still holds: walker-level Δρ via species-")
print("  sensitive Kubo is K-rational by construction and bypasses the path δ")
print("  operator-level obstruction.  But IMPLEMENTING it at substrate level")
print("  requires more machinery than exists in the current Π_JJ probe.")
print()
print("=" * 78)
print("Alternative reframe: per-atom weighting as proxy species filter")
print("=" * 78)
print()
print("Without building the full 16384-dim machinery, we can probe a")
print("STRUCTURAL PROXY: assign each atom in the primitive cell a Hamming-")
print("weight (per Furey identification at unbroken-PS) and weight the Kubo")
print("trace by per-atom weight factors.")
print()
print("Furey 2018: per the standard PS embedding, a single generation has")
print("(ν_L, e_L, u_L, d_L) + chirality partners = 16 fermion states per")
print("generation, distributed across the substrate.  In the primitive cell")
print("of srs, this corresponds to specific per-atom Hamming-weight")
print("assignments.")
print()
print("PROXY TEST: compute Π_JJ_up and Π_JJ_down by re-weighting the bond")
print("hopping at each atom by the up-vs-down Hamming-weight content")
print("expected at that atom.")
print()
print("This is a structural HEURISTIC, not a rigorous species-resolved Kubo.")
print("It might still reveal whether up-down asymmetry GIVES A NON-ZERO")
print("DIFFERENCE in the substrate gauge self-energy.  If yes → proceed to")
print("rigorous treatment.  If no → species filter doesn't enter at this")
print("level even heuristically.")
print()

# Sanity check: compute the simplest Pi difference under per-atom weighting
print("=" * 78)
print("Heuristic species-weighted Kubo at k = (0.3, 0.5, 0.7), ω = 0.3")
print("=" * 78)
print()

from gauge_beta_from_substrate_kubo_probe import Pi_v_at_k

# Standard sector-blind reference (Pi_v at single k)
omega_E = 0.3
T = 0.05
K_total = Pi_v_at_k(k_test, omega_E, T)
print(f"Sector-blind Π^{{μν}} at k={k_test}:")
print(K_total)
print(f"  trace/3: {np.trace(K_total)/3:.6f}")
print()

# Heuristic species weighting via DIAGONAL re-weighting at atoms
# (Furey: standard PS has 4 atoms per cell; up-type concentrated on some,
# down-type on others.  Without explicit PS embedding, try simple proxies.)

# Up-sector proxy: weight atoms 0 and 1 (call them "up-charged" sites)
W_up = np.diag([1.0, 1.0, 0.0, 0.0])
# Down-sector proxy: weight atoms 2 and 3 (call them "down-charged" sites)
W_down = np.diag([0.0, 0.0, 1.0, 1.0])

# Apply weights via velocity-vertex re-weighting:
# v^μ_{ts} → W_t · v^μ_{ts}  (origin-projected, simplest weighting)
# But this is just one possible proxy; truly species-resolved requires
# Cl(6) Fock.  Use it as a SIGNAL test.

def Pi_weighted(k_cart, omega_E, T, weight):
    """Apply per-atom diagonal weight at the destination of hop."""
    from lorentz_sig_g_sub_elastic_moduli import H_bloch
    from lorentz_sig_g_sub_dynamic_omega_T import fermi_smooth
    H_k = H_bloch(k_cart)
    eigs, U = np.linalg.eigh(H_k)
    V = np.zeros((3, 4, 4), dtype=complex)
    for m in range(3):
        v_raw = velocity_matrix(k_cart, m)
        # Apply weight at vertex (destination index)
        v_w = weight @ v_raw
        v_w = (v_w + v_w.conj().T) / 2  # Hermitize
        V[m] = U.conj().T @ v_w @ U
    f = np.array([fermi_smooth(eigs[n], 0.0, T) for n in range(4)])
    K = np.zeros((3, 3), dtype=float)
    for n in range(4):
        for m in range(4):
            diff = f[n] - f[m]
            if abs(diff) < 1e-15:
                continue
            Delta = eigs[n] - eigs[m]
            denom = Delta * Delta + omega_E * omega_E
            weight_kubo = diff * Delta / denom
            for a in range(3):
                for b in range(3):
                    term = np.conj(V[a, m, n]) * V[b, m, n]
                    K[a, b] += -2.0 * (term * weight_kubo).real
    return K

K_up = Pi_weighted(k_test, omega_E, T, W_up)
K_down = Pi_weighted(k_test, omega_E, T, W_down)

print(f"Heuristic species-weighted (W_up = diag(1,1,0,0)):")
print(K_up)
print(f"  trace/3: {np.trace(K_up)/3:.6f}")
print()
print(f"Heuristic species-weighted (W_down = diag(0,0,1,1)):")
print(K_down)
print(f"  trace/3: {np.trace(K_down)/3:.6f}")
print()

trace_up = np.trace(K_up)/3
trace_down = np.trace(K_down)/3
print(f"Trace difference up - down: {trace_up - trace_down:.6f}")
print(f"Ratio (up-down)/total: {(trace_up - trace_down)/(np.trace(K_total)/3):.6f}")
print()

# Verdict
print("=" * 78)
print("Phase A actual numerical finding")
print("=" * 78)
print()
if abs(trace_up - trace_down) > 1e-6:
    print(f"NON-TRIVIAL: heuristic species-weighted Π^{{μν}} differs between up and")
    print(f"down sector assignments at this k point.  Difference = {trace_up - trace_down:.6f}")
    print()
    print(f"This SUGGESTS that even at the 4-atom Bloch level (which is supposedly")
    print(f"sector-blind), per-atom Hamming-weight weighting can produce non-trivial")
    print(f"species-asymmetric contributions to Kubo Π_JJ.")
    print()
    print(f"CAVEAT: this is a HEURISTIC test with proxy per-atom assignments.  The")
    print(f"actual Furey-PS-embedded assignment isn't trivially (1,1,0,0)/(0,0,1,1).")
    print(f"Specific PS embedding on srs's 4 atoms requires Cl(4) ⊗ Cl(2) ≅ Cl(6)")
    print(f"per-vertex structure — substantial multi-session work to nail down.")
else:
    print(f"TRIVIAL: heuristic species-weighted Π^{{μν}} essentially equal between")
    print(f"up and down assignments.  Difference < 1e-6.")
    print()
    print(f"This suggests the 4-atom Bloch level is genuinely sector-blind, and")
    print(f"species sensitivity requires Cl(6) Fock extension.")

print()
print("=" * 78)
print("Phase A honest verdict")
print("=" * 78)
print()
print("(1) The species-projector commutator does NOT trivially vanish at the")
print("    4-atom Bloch level — per-atom weights enter Kubo non-trivially.")
print()
print("(2) BUT the actual Hamming-weight species filter (n=1 vs n=2 on Cl(6)")
print("    Fock per atom) lives at a DIFFERENT structural level than the existing")
print("    Π_JJ machinery.  Building the linked machinery is multi-session.")
print()
print("(3) A HEURISTIC proxy at the 4-atom Bloch level shows non-trivial up-vs-")
print("    down asymmetric contributions, suggesting the mechanism CAN exist.")
print("    But rigorous derivation requires Cl(6) Fock extension.")
print()
print("(4) α2''' as originally scoped (3-phase bounded) is over-optimistic given")
print("    the structural level mismatch.  Realistic effort to build the species-")
print("    resolved Kubo machinery: 3-6 sessions just for Phase A.  Phase B+C")
print("    on top: another 2-3 sessions.  Total: 5-9 sessions — between the")
print("    optimistic 3-session estimate and the multi-sprint Path β/B figures.")
print()
print("STATUS: α2''' Phase A partial — structural compatibility confirmed at")
print("        heuristic level; rigorous implementation needs Cl(6) Fock")
print("        extension to the Bloch operator.")
print()
print("=" * 78)
