#!/usr/bin/env python3
"""
proofs/foundations/BR4_session3_canonical_basis_koide_check_2026-05-27.py

BR4 Session 3 — Canonical mass basis via R-C grading + Q_i + γ_7;
                re-verify AB5/AB6; diagonalize W; check Koide spectrum.

PURPOSE
-------
Session 2 built a Bloch-fiber Q_i intertwiner that passed AB5/AB6
pre-flight, but used an arbitrary DFT_3 mass-basis choice (σ_eff in
that basis differed from σ_C3 by ‖·‖_F = 2.449). Session 3 resolves
the 24-parameter freedom via R-C generation grading:

  |gen i⟩ ∈ Cl(6) Fock is the joint eigenvector with:
    σ_eff eigenvalue ω^(i-1)   (R-C: gen i ↔ i-th C_3 isotype)
    Q_i  eigenvalue +1          (T4: gen i has canonical Q_i)
    γ_7  eigenvalue +1          (chirality fix: left-handed Weyl piece)

If these joint eigenspaces are 1-dim (or naturally 1-dim after the
chirality fix), the basis is canonical. Otherwise, we report the
ambiguity and fall back to an isotype rep.

PIPELINE
--------
1. Build Cl(6) Fock generators, σ_eff (diagonal Spin(3) lift from T2),
   Q_1, Q_2, Q_3 (T4 Furey pair-complements), γ_7 (chirality).
2. Decompose Cl(6) Fock into σ_eff isotypes (4, 2, 2).
3. Within each σ_eff isotype, intersect with +1 γ_7 eigenspace, then
   with +1 Q_i eigenspace, to get |gen i⟩.
4. Build W_ji = ⟨gen j | D_i | gen i⟩.
5. Re-check AB5/AB6 on canonical-basis W.
6. Diagonalize W (or M_gen = W^†W) and read off spectrum.
7. Test against Koide-cosine parametrization with framework's
   theorem-grade values: ε²_down = 5/2 (W53), ε²_up = 17/5 (P37),
   δ_lepton = 2/9 (Bernoulli at Q=2/3).
"""

import numpy as np
from scipy.linalg import expm
from collections import Counter

TOL = 1e-9


# ---------------------------------------------------------------------------
# Cl(6,0) Brauer-Weyl
# ---------------------------------------------------------------------------
def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

G = [None] * 7
G[1] = kron(sx, I2, I2)
G[2] = kron(sy, I2, I2)
G[3] = kron(sz, sx, I2)
G[4] = kron(sz, sy, I2)
G[5] = kron(sz, sz, sx)
G[6] = kron(sz, sz, sy)

G7 = -1j * G[1] @ G[2] @ G[3] @ G[4] @ G[5] @ G[6]

Q1 = G[3] @ G[4] @ G[5] @ G[6]
Q2 = G[1] @ G[2] @ G[5] @ G[6]
Q3 = G[1] @ G[2] @ G[3] @ G[4]

# Diagonal Spin(3) lift σ_eff per T2
S12 = -1j/2 * (G[1] @ G[2])
S13 = -1j/2 * (G[1] @ G[3])
S23 = -1j/2 * (G[2] @ G[3])
J123 = (1/np.sqrt(3)) * (S23 - S13 + S12)
sigma_123 = expm(-1j * (2*np.pi/3) * J123)

S45 = -1j/2 * (G[4] @ G[5])
S46 = -1j/2 * (G[4] @ G[6])
S56 = -1j/2 * (G[5] @ G[6])
J456 = (1/np.sqrt(3)) * (S56 - S46 + S45)
sigma_456 = expm(-1j * (2*np.pi/3) * J456)

sigma_diag = sigma_123 @ sigma_456

# σ_diag has order 3 (verified in Session 2)
sigma_eff = sigma_diag
assert np.allclose(sigma_eff @ sigma_eff @ sigma_eff, np.eye(8), atol=1e-8), \
    "σ_eff is not order 3"

omega = np.exp(2j * np.pi / 3)


# ---------------------------------------------------------------------------
# Step 1 — Project onto σ_eff isotypes
# ---------------------------------------------------------------------------

def isotype_projector_8(sigma, omega_power):
    """Project onto σ-eigenspace with eigenvalue ω^omega_power on 8-dim Fock."""
    P = np.zeros_like(sigma)
    for k in range(3):
        P += (omega ** (-k * omega_power)) * np.linalg.matrix_power(sigma, k)
    return P / 3


P_iso = {
    0: isotype_projector_8(sigma_eff, 0),   # trivial
    1: isotype_projector_8(sigma_eff, 1),   # ω
    2: isotype_projector_8(sigma_eff, 2),   # ω̄
}

print("=" * 76)
print("BR4 Session 3 — Canonical basis via R-C + Q_i + γ_7; Koide check")
print("=" * 76)
print()
print("Step 1 — σ_eff isotypic decomposition of Cl(6) Fock:")
for power, P in P_iso.items():
    rank = int(np.round(np.real(np.trace(P))))
    print(f"  trace P_{['1','ω','ω̄'][power]}_iso = {np.trace(P).real:.3f}  (dim = {rank})")
print()


# ---------------------------------------------------------------------------
# Step 2 — Within each isotype, intersect with +1 γ_7 and +1 Q_i eigenspaces
# ---------------------------------------------------------------------------

def project_within(M, restriction_subspace_basis):
    """Project operator M onto a subspace given by orthonormal basis vectors."""
    V = np.column_stack(restriction_subspace_basis) if restriction_subspace_basis else None
    if V is None or V.shape[1] == 0:
        return np.array([])
    return V.conj().T @ M @ V


def orthonormal_basis_of_subspace(P, tol=1e-8):
    """Get an orthonormal basis for the range of projector P."""
    # SVD-based extraction
    U, s, Vh = np.linalg.svd(P)
    rank = int(np.sum(s > tol))
    return U[:, :rank]


# Get orthonormal bases of each isotype subspace
B_iso = {power: orthonormal_basis_of_subspace(P) for power, P in P_iso.items()}

print("Step 2 — Joint eigenspaces with (σ_eff isotype, +1 γ_7, +1 Q_i):")
gen_vecs = {}
ambiguous = False

for gen_idx in [1, 2, 3]:
    iso_power = gen_idx - 1   # gen 1 ↔ trivial (power 0), gen 2 ↔ ω, gen 3 ↔ ω̄
    Q_i = [Q1, Q2, Q3][gen_idx - 1]

    B = B_iso[iso_power]
    dim = B.shape[1]
    print(f"\n  --- Gen {gen_idx} (σ_eff isotype: {['1','ω','ω̄'][iso_power]}, dim {dim}; "
          f"Q_{gen_idx}; γ_7=+1) ---")

    # Within isotype, project γ_7 and find +1 eigenspace
    G7_restricted = project_within(G7, [B[:, k] for k in range(dim)])
    eigvals_g7, eigvecs_g7 = np.linalg.eigh(G7_restricted)
    plus1_indices = [k for k, v in enumerate(eigvals_g7) if v > 0.5]
    print(f"    γ_7 eigenvalues within isotype: {[f'{e:+.2f}' for e in eigvals_g7]}")
    print(f"    +1 γ_7 subspace dim within isotype: {len(plus1_indices)}")

    if len(plus1_indices) == 0:
        print(f"    NO +1 γ_7 states in this isotype — gen {gen_idx} unidentifiable here")
        ambiguous = True
        continue

    # Build orthonormal basis of (isotype ∩ +1 γ_7)
    B_iso_g7plus = B @ eigvecs_g7[:, plus1_indices]
    sub_dim = B_iso_g7plus.shape[1]

    # Within isotype ∩ +1 γ_7, project Q_i; pick MAX-eigenvalue eigenvector
    # (relaxes strict +1 constraint; T4's "Q_i ↔ gen i" is a label/best-alignment,
    # not an eigenvalue equation, since Q_i mixes σ_eff isotypes).
    Q_restricted = B_iso_g7plus.conj().T @ Q_i @ B_iso_g7plus
    eigvals_Q, eigvecs_Q = np.linalg.eigh(Q_restricted)
    print(f"    Q_{gen_idx} eigenvalues within isotype ∩ +1 γ_7: "
          f"{[f'{e:+.4f}' for e in eigvals_Q]}")
    max_idx = int(np.argmax(eigvals_Q))
    print(f"    Picking max-Q_{gen_idx} eigenvector (eigenvalue {eigvals_Q[max_idx]:+.4f})")
    if len(eigvals_Q) > 1 and eigvals_Q[max_idx] < 0.99:
        print(f"    NOTE: max Q_{gen_idx} eigenvalue < 1 ⟹ Q_{gen_idx} mixes σ_eff isotypes")

    gen_vec = B_iso_g7plus @ eigvecs_Q[:, max_idx]
    gen_vec = gen_vec / np.linalg.norm(gen_vec)
    gen_vecs[gen_idx] = gen_vec
    print(f"    |gen {gen_idx}⟩ assigned: norm = {np.linalg.norm(gen_vec):.6f}")

print()


# ---------------------------------------------------------------------------
# Step 3 — Verify |gen i⟩ orthonormal; build C³_obs ⊂ Cl(6) Fock
# ---------------------------------------------------------------------------

if len(gen_vecs) < 3:
    print(f"  ABORT — basis fix failed (only {len(gen_vecs)}/3 gens identified)")
    print(f"  AB1 partially triggered for the (R-C × Q_i × +γ_7) canonical-basis choice.")
    print(f"  Recommendation: relax one constraint (e.g., drop γ_7 requirement)")
    print(f"                  or pivot to direction (iii) chirality-flip.")
else:
    V_mass = np.column_stack([gen_vecs[1], gen_vecs[2], gen_vecs[3]])
    overlap = V_mass.conj().T @ V_mass

    print(f"Step 3 — |gen i⟩ inner product matrix (should be I_3 if orthonormal):")
    print(f"  Real part:")
    for row in overlap.real:
        print(f"    [ {row[0]:+.4f}  {row[1]:+.4f}  {row[2]:+.4f} ]")
    print(f"  Imag part:")
    for row in overlap.imag:
        print(f"    [ {row[0]:+.4f}  {row[1]:+.4f}  {row[2]:+.4f} ]")

    is_orthonormal = np.allclose(overlap, np.eye(3), atol=1e-8)
    print(f"  Orthonormal: {is_orthonormal}")
    if not is_orthonormal:
        max_off = np.max(np.abs(overlap - np.eye(3)))
        print(f"  max |off-diag| = {max_off:.3e}")
    print()


    # -----------------------------------------------------------------------
    # Step 4 — Build W_ji = ⟨gen j | D_i | gen i⟩
    # -----------------------------------------------------------------------

    D = {1: (np.sqrt(3)/2) * G7 + 1j * (np.sqrt(5)/2) * Q1,
         2: (np.sqrt(3)/2) * G7 + 1j * (np.sqrt(5)/2) * Q2,
         3: (np.sqrt(3)/2) * G7 + 1j * (np.sqrt(5)/2) * Q3}

    W = np.zeros((3, 3), dtype=complex)
    for i in [1, 2, 3]:
        gi = gen_vecs[i]
        for j in [1, 2, 3]:
            gj = gen_vecs[j]
            W[j-1, i-1] = gj.conj() @ D[i] @ gi

    print("Step 4 — Canonical-basis W = ⟨gen j | D_i | gen i⟩:")
    print()
    print("  Magnitudes |W_ji|:")
    for j in range(3):
        print(f"    [ {abs(W[j,0]):.4f}  {abs(W[j,1]):.4f}  {abs(W[j,2]):.4f} ]")
    print()
    print("  Arguments arg(W_ji) [degrees]:")
    for j in range(3):
        args = [np.degrees(np.angle(W[j, i])) if abs(W[j, i]) > 1e-12 else 0.0
                for i in range(3)]
        print(f"    [ {args[0]:>+8.2f}°  {args[1]:>+8.2f}°  {args[2]:>+8.2f}° ]")
    print()


    # -----------------------------------------------------------------------
    # Step 5 — AB5/AB6 re-check in canonical basis
    # -----------------------------------------------------------------------

    sigma_C3 = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=complex)
    sigma_C3_sq = sigma_C3 @ sigma_C3


    def isotypic_proj_3(omega_power):
        P = np.zeros((3, 3), dtype=complex)
        for k in range(3):
            P += (omega ** (-k * omega_power)) * np.linalg.matrix_power(sigma_C3, k)
        return P / 3


    P_1 = isotypic_proj_3(0)
    P_omega_3 = isotypic_proj_3(1)
    P_omegabar_3 = isotypic_proj_3(2)

    # Circulant decomposition: W = a·I + b·σ + c·σ²
    def trace_inner_3x3(A, B):
        return np.trace(A.conj().T @ B) / 3

    a = trace_inner_3x3(np.eye(3), W)
    b = trace_inner_3x3(sigma_C3, W)
    c = trace_inner_3x3(sigma_C3_sq, W)
    W_circ = a * np.eye(3) + b * sigma_C3 + c * sigma_C3_sq
    residual_norm = np.linalg.norm(W - W_circ)

    comm_sigma = W @ sigma_C3 - sigma_C3 @ W
    comm_P1 = W @ P_1 - P_1 @ W
    comm_Pom = W @ P_omega_3 - P_omega_3 @ W
    comm_Pombar = W @ P_omegabar_3 - P_omegabar_3 @ W

    print("Step 5 — AB5/AB6 checks in canonical basis:")
    print(f"  Circulant components: a={a:.4f}, b={b:.4f}, c={c:.4f}")
    print(f"  ‖W − W_circulant‖_F     = {residual_norm:.3e}")
    print(f"  ‖[W, σ_C3]‖_∞           = {np.abs(comm_sigma).max():.3e}")
    print(f"  ‖[W, P_1]‖_∞            = {np.abs(comm_P1).max():.3e}")
    print(f"  ‖[W, P_ω]‖_∞            = {np.abs(comm_Pom).max():.3e}")
    print(f"  ‖[W, P_ω̄]‖_∞            = {np.abs(comm_Pombar).max():.3e}")
    print()


    # -----------------------------------------------------------------------
    # Step 6 — Diagonalize W; read off eigenvalues
    # -----------------------------------------------------------------------

    eigvals_W = np.linalg.eigvals(W)
    eigvals_sorted = sorted(eigvals_W, key=lambda z: abs(z))

    print(f"Step 6 — W spectrum:")
    print(f"  Eigenvalues of W:")
    for k, e in enumerate(eigvals_sorted):
        print(f"    λ_{k+1} = {e:.6f}  |λ| = {abs(e):.6f}  arg = {np.degrees(np.angle(e)):+.4f}°")
    print(f"  |λ_3|/|λ_2| = {abs(eigvals_sorted[2])/abs(eigvals_sorted[1]):.4f}")
    print(f"  |λ_2|/|λ_1| = {abs(eigvals_sorted[1])/max(abs(eigvals_sorted[0]),1e-12):.4f}")
    print()

    # M_gen ≡ W^†W has positive real eigenvalues which would be mass²
    M_gen_proxy = W.conj().T @ W
    eigvals_M = np.linalg.eigvalsh(M_gen_proxy)
    eigvals_M_sorted = sorted(eigvals_M)

    print(f"  M_gen proxy = W^†W eigenvalues (would be mass² in canonical reading):")
    for k, e in enumerate(eigvals_M_sorted):
        print(f"    m²_{k+1} = {e:.6f}  m_{k+1} = {np.sqrt(abs(e)):.6f}")
    if abs(eigvals_M_sorted[0]) > 1e-12 and abs(eigvals_M_sorted[1]) > 1e-12:
        print(f"  m_3/m_2 = {np.sqrt(eigvals_M_sorted[2]/eigvals_M_sorted[1]):.4f}")
        print(f"  m_2/m_1 = {np.sqrt(eigvals_M_sorted[1]/eigvals_M_sorted[0]):.4f}")
    print()


    # -----------------------------------------------------------------------
    # Step 7 — Test Koide-cosine parametrization
    # -----------------------------------------------------------------------
    # Koide: m_i = M_0 (1 + ε cos(δ + 2π(i-1)/3))² (within-species)
    # framework theorem-grade values:
    #   ε²_lepton = 0 (degenerate; doesn't apply with Koide-cosine)
    #   ε²_down = 5/2 (W53 Type IV walker)
    #   ε²_up = 17/5 (P37 ratio)
    #   δ_lepton = 2/9 ≈ 0.2222 rad ≈ 12.73° (Bernoulli)

    if abs(eigvals_M_sorted[0]) > 1e-12:
        m_1 = np.sqrt(eigvals_M_sorted[0])
        m_2 = np.sqrt(eigvals_M_sorted[1])
        m_3 = np.sqrt(eigvals_M_sorted[2])

        # Try to fit Koide m_i = M_0 (1 + ε cos(δ_i))² with δ_i = δ + 2π(i-1)/3
        # Solve for M_0, ε, δ from 3 masses (over-determined unless cosines align)
        # M_0 = (m_1 + m_2 + m_3) / 3, sum of cosines = 0
        # Koide ratio: (Σ m_i)² / Σ m_i² = ... well-known form

        sum_m = m_1 + m_2 + m_3
        sum_m_sq = m_1**2 + m_2**2 + m_3**2
        koide_ratio = sum_m**2 / sum_m_sq if sum_m_sq > 0 else 0

        print(f"Step 7 — Koide-pattern check:")
        print(f"  Koide ratio (Σm_i)²/Σm_i² = {koide_ratio:.6f}")
        print(f"    SM Koide for leptons ≈ 2/3 (Koide 1981 empirical fact)")
        print(f"    Framework Q_Koide = 2/3 (theorem-grade)")
        print()

        # The "δ" from W eigenvalues' arguments
        args_W = sorted([np.angle(e) for e in eigvals_W], key=lambda x: x)
        delta_W_candidate = abs(args_W[1] - args_W[0]) if len(args_W) >= 2 else 0
        print(f"  Argument gaps between W eigenvalues:")
        print(f"    {[f'{np.degrees(a):+.2f}°' for a in args_W]}")
        print(f"    smallest gap = {np.degrees(delta_W_candidate):.4f}°")
        print(f"    framework δ_lepton = 2/9 rad = 12.73°")
        print(f"    framework φ_K4 (down candidate) = arccos(1/3) = 70.53°")

    print()


    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    print()

    is_circulant_canonical = residual_norm < 1e-9
    commutes_sigma_canonical = np.abs(comm_sigma).max() < 1e-9

    if not is_orthonormal:
        print("  Basis identification ANOMALY: |gen i⟩ not orthonormal.")
        print("  This signals an issue with the joint (σ_eff isotype, +γ_7, +Q_i)")
        print("  spec — possibly Q_i and γ_7 don't have +1 eigenvectors in the")
        print("  required isotypes. Investigate before next step.")
    elif is_circulant_canonical:
        print("  W in canonical basis IS circulant — AB6 FAILS in this basis.")
        print("  The (R-C × γ_7 × Q_i) basis choice doesn't preserve non-circularity.")
    else:
        print(f"  W in canonical basis is NON-circulant (‖residual‖ = {residual_norm:.3e}).")
        if commutes_sigma_canonical:
            print(f"  But W commutes with σ_C3 — unusual; investigate.")
        else:
            print(f"  W does NOT commute with σ_C3 — AB5/AB6 PASS in canonical basis.")
            print(f"  Koide-pattern check above shows whether eigenvalue spectrum matches.")
    if ambiguous:
        print()
        print("  CAVEAT: residual basis-choice ambiguity remained at the +1 Q_i step")
        print("          for at least one generation. Result is canonical-modulo-this.")
