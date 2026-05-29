#!/usr/bin/env python3
"""
β closure — Reading B: cross-sector self-energy Σ_γ.

The photon at ω²=36 lives in the walker −1 sector (verified in
srs_photon_walker_correspondence.py).  Walker −1 has zero direct
chirality.  But the walker B doesn't preserve the symmetric subspace
(=photon subspace); applying B to the photon mode produces a vector with
amplitude in OTHER walker sectors, including the chiral ones (±h, ±h*).

This "leak" is the cross-sector coupling Σ_γ.  We compute it explicitly
and check whether its restriction to L⊕R has the right form for c=1.

Strategy
--------
1. Take photon basis Q (orthonormal in undirected-edge space).
2. Lift to directed bonds: γ_dir = π_sym · Q.
3. Apply B: B·γ_dir.  This stays in directed-bond space but generally
   leaves the symmetric subspace.
4. Decompose B·γ_dir into walker eigensectors P_λ.  Track:
   - The symmetric piece: B_sym·γ_dir = (π_sym · π_sym†) · B · γ_dir.
     (This stays in walker −1 sector by hypothesis.)
   - The antisymmetric piece: B_anti·γ_dir = (I − π_sym·π_sym†)·B·γ_dir.
     This is the LEAK into other sectors.
5. The chirality content of the leak is what enters Σ_γ at second order
   in perturbation theory:
       Σ_γ = (B_anti·γ_dir)† · (1/(B − ω_γ)) · (B_anti·γ_dir)
   where ω_γ = −1 (the photon's walker eigenvalue).
6. Restrict Σ_γ to the L/R photon basis and read off the chirality
   eigenvalue.
"""

import os
import sys
import math
import numpy as np
from numpy import linalg as la

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from srs_photon_bloch_primitive import (
    build_primitive_unit_cell,
    find_primitive_connectivity,
    canonical_edges_primitive,
    incidence_matrix_primitive,
)
from srs_photon_hodge import build_d1, build_edge_lookup
from srs_cycle_enumerator import enumerate_simple_cycles
from srs_photon_c3_chainmap import build_C3_edge, build_delta_1, K_P_RED
from srs_photon_chirality_coefficient import (
    build_pi_projector,
    build_C3_directed,
    build_B_directed,
)


def main():
    print("=" * 72)
    print("β closure — cross-sector Σ_γ self-energy on photon ω²=36 mode")
    print("=" * 72)

    verts, lat = build_primitive_unit_cell()
    bonds = find_primitive_connectivity(verts, lat)
    edges = canonical_edges_primitive(bonds)
    edge_lookup = build_edge_lookup(edges)
    cycles = enumerate_simple_cycles(bonds, max_length=10)
    k_red = K_P_RED

    d = incidence_matrix_primitive(k_red, edges, len(verts))
    d1 = build_d1(cycles, edge_lookup, k_red, len(edges))
    Delta_1 = build_delta_1(d, d1)
    B = build_B_directed(bonds, k_red)
    C3_e = build_C3_edge(edges, k_red)
    pi_sym = build_pi_projector(bonds, edges, k_red)

    # ------- Photon basis at ω²=36 -------
    eigs, vecs = la.eig(Delta_1)
    order = np.argsort(eigs.real)
    eigs, vecs = eigs[order], vecs[:, order]
    mask = np.abs(eigs.real - 36) < 1e-6
    photon_basis = vecs[:, mask]
    Q, _ = la.qr(photon_basis)         # 6 × 2

    # L, R in the photon basis
    omega = np.exp(2j * math.pi / 3)
    omega2 = omega.conjugate()
    C3_photon = Q.conj().T @ C3_e @ Q
    ev_C3, vec_C3 = la.eig(C3_photon)
    L_in_Q = vec_C3[:, np.argmin(np.abs(ev_C3 - omega))]
    R_in_Q = vec_C3[:, np.argmin(np.abs(ev_C3 - omega2))]
    L_in_Q /= la.norm(L_in_Q)
    R_in_Q /= la.norm(R_in_Q)

    # Lift to directed bonds
    L_dir = pi_sym @ Q @ L_in_Q
    R_dir = pi_sym @ Q @ R_in_Q
    print(f"\nPhoton L lifted to directed bonds: norm = {la.norm(L_dir):.6f}")
    print(f"Photon R lifted to directed bonds: norm = {la.norm(R_dir):.6f}")

    # ------- Walker eigendecomposition -------
    Bevs, Bvecs = la.eig(B)
    for j in range(Bvecs.shape[1]):
        Bvecs[:, j] /= la.norm(Bvecs[:, j])

    h = complex(math.sqrt(3)/2, math.sqrt(5)/2)
    targets = {"+h": h, "+h*": h.conjugate(), "-h": -h, "-h*": -h.conjugate(),
               "+1": 1+0j, "-1": -1+0j}
    proj = {}
    for label, target in targets.items():
        idx = [i for i, ev in enumerate(Bevs) if abs(ev - target) < 1e-6]
        V = Bvecs[:, idx]
        Q_w, _ = la.qr(V)
        proj[label] = (target, Q_w, Q_w @ Q_w.conj().T)

    # ------- Apply B to photon and decompose -------
    print(f"\n--- 1. Apply B to L_dir, decompose into walker sectors ---")
    BL = B @ L_dir
    BR = B @ R_dir
    print(f"  ⟨L|B|L⟩ = {np.vdot(L_dir, BL):+.6f}  (should be -1 if L pure walker −1)")
    print(f"  ⟨R|B|R⟩ = {np.vdot(R_dir, BR):+.6f}  (should be -1)")

    # Project BL onto each walker sector
    print(f"\n  Walker-sector decomposition of B·L_dir = (-1)·L_dir + leak:")
    print(f"    {'sector':>8}  {'amplitude (norm)':>18}  {'arg(λ)':>10}  sin(arg)")
    leak_chirality_L = 0.0
    leak_chirality_R = 0.0
    for label, (target, Q_w, P) in proj.items():
        amp_L = la.norm(P @ BL)        # mass on this sector
        amp_R = la.norm(P @ BR)
        sin_arg = math.sin(np.angle(target))
        # Subtract the diagonal (-1)·γ piece for the −1 sector
        if label == "-1":
            # The "leak" is what's beyond the diagonal action
            BL_proj = P @ BL              # ≈ -L_dir on the −1 sector
            extra = BL_proj - (-1.0) * (P @ L_dir)
            print(f"    {label:>8}  {amp_L:>18.6f}  ({la.norm(extra):.2e} extra beyond diagonal)")
        else:
            print(f"    {label:>8}  {amp_L:>18.6f}  {math.degrees(np.angle(target)):+.2f}°  "
                  f"{sin_arg:+.6f}")
        # Accumulate chirality contribution from leak
        leak_chirality_L += amp_L**2 * sin_arg
        leak_chirality_R += amp_R**2 * sin_arg

    print(f"\n  Chirality of B·L_dir (leak): Σ |amp|²·sin(arg λ) = {leak_chirality_L:+.6f}")
    print(f"  Chirality of B·R_dir (leak):                       = {leak_chirality_R:+.6f}")

    # ------- Second-order perturbative Σ_γ -------
    print(f"\n--- 2. Second-order Σ_γ on the photon ω_γ = -1 ---")
    print(f"  Σ_γ = ⟨γ|B† · (1/(B−ω_γ)) · B|γ⟩  with ω_γ = -1")
    print(f"  (Σ over walker sectors λ ≠ -1 of |⟨γ|B|λ⟩|² / (λ − ω_γ).)")

    omega_gamma = -1.0 + 0.0j
    Sigma_LL = 0.0 + 0.0j
    Sigma_RR = 0.0 + 0.0j
    Sigma_LR = 0.0 + 0.0j
    chir_diag = 0.0

    for label, (target, Q_w, P) in proj.items():
        if label == "-1":
            continue
        denom = target - omega_gamma
        # ⟨L|B†|λ⟩ = ⟨B·L|λ⟩ = (Q_w† · BL)
        BL_proj = Q_w.conj().T @ BL                     # column vector in λ-eigspace
        BR_proj = Q_w.conj().T @ BR
        Sigma_LL += np.vdot(BL_proj, BL_proj) / denom
        Sigma_RR += np.vdot(BR_proj, BR_proj) / denom
        Sigma_LR += np.vdot(BL_proj, BR_proj) / denom
        # Chirality content per sector
        sin_arg = math.sin(np.angle(target))
        # Chirality contribution from this sector to Σ_LL
        contrib = (np.vdot(BL_proj, BL_proj) / denom) * sin_arg
        # (Diagnostic only; chirality is not the same as Σ_γ)

    print(f"\n  Σ_LL = ⟨L|B† · G_Q · B|L⟩ = {Sigma_LL:+.6f}")
    print(f"  Σ_RR = ⟨R|B† · G_Q · B|R⟩ = {Sigma_RR:+.6f}")
    print(f"  Σ_LR = ⟨L|B† · G_Q · B|R⟩ = {Sigma_LR:+.6f}")
    print(f"  (Σ_LL − Σ_RR)/2 = {(Sigma_LL - Sigma_RR)/2:+.6f}")
    sin_arg_h = math.sqrt(5/8)
    print(f"  sin(arg h) = {sin_arg_h:+.6f}")
    print(f"  ratio (Σ_LL − Σ_RR)/(2·sin(arg h)) = "
          f"{((Sigma_LL - Sigma_RR)/2 / sin_arg_h):+.6f}")

    # Imaginary part of Σ (chirality readout)
    Sigma_chir = (Sigma_LL - Sigma_RR) / 2
    print(f"\n  Im[(Σ_LL − Σ_RR)/2] = {Sigma_chir.imag:+.6f}  "
          f"← this is the chirality content of cross-sector self-energy")
    print(f"  Re[(Σ_LL − Σ_RR)/2] = {Sigma_chir.real:+.6f}")

    # ------- Comparison with sin(arg h) -------
    print(f"\n--- 3. Coefficient extraction ---")
    print(f"  If β = c·sin(arg h)·α_EM with c·α_EM matching cross-sector Σ_γ:")
    print(f"    c·α_EM = Σ_chir / sin(arg h) (interpretation)")
    print(f"    Σ_chir = {Sigma_chir}")
    print(f"    Σ_chir / sin(arg h) = {Sigma_chir/sin_arg_h}")
    print(f"  For c=1 to hold, Σ_chir / sin(arg h) should be α_EM-like in magnitude.")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
