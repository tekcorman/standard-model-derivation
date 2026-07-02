#!/usr/bin/env python3
"""
Koide delta phase computation.

Computes the simultaneous eigenvectors of B(P) and U_{C_3} within the
Ramanujan subspace V_Ram, extracts the relative phase between the trivial and
omega C_3-isotypic sectors, and compares to arg(h)/4 and delta_obs.

Claim under test:
  Does the relative phase phi_1 - phi_0 between the omega and trivial sectors
  in the Koide Fourier sum equal delta_obs ~ 12.74 deg, arg(h)/4 ~ 13.06 deg,
  or zero?

The Koide amplitude at generation j is:
  amp_j = c_0 + c_1 * omega^j + c_2 * omega^{-j}
where c_0 = sqrt(4) * exp(i*phi_0), c_1 = sqrt(2) * exp(i*phi_1).
The Koide phase delta = phi_1 - phi_0.

Methodology:
  1. Build B(P) (12x12) and U_{C_3} (12x12).
  2. Extract V_Ram (8-dim Ramanujan subspace).
  3. Within V_Ram, find a basis that simultaneously diagonalizes B(P) and U_{C_3}.
  4. Separate eigenvectors by C_3-isotypic sector (trivial vs omega vs omega^2).
  5. For each sector, compute the phase of the corresponding B(P) eigenvalue.
  6. The relative phase delta = arg(c_1) - arg(c_0) where c_alpha = sqrt(mu_alpha)
     * exp(i*phi_alpha), with phi_alpha the phase carried by the sector.
  7. Compare to arg(h) ~ 52.24 deg, arg(h)/4 ~ 13.06 deg, delta_obs ~ 12.74 deg.

Key question:
  The B(P) eigenvalues on V_Ram are {h, h*, -h, -h*}, each with mult 2.
  The phase of the B(P) eigenvalue is:
    - arg(h)   ~ +52.24 deg  for the h-eigenspace
    - arg(h*)  ~ -52.24 deg  for the h*-eigenspace
    - arg(-h)  ~ +180-52.24 = 127.76 deg for the -h-eigenspace
    - arg(-h*) ~ -127.76 deg for the -h*-eigenspace
  Each of these four eigenspaces is 2-dimensional.
  The C_3-isotypic structure on V_Ram is (4, 2, 2) = trivial + omega + omega^2.
  The question is: which B(P) eigenvalues appear in which C_3-isotypic sectors?

  If the trivial sector (mult 4) contains two B(P) eigenvalues from different
  eigenspaces (e.g. one h and one h*), the phase of c_0 is the average or
  combination of arg(h) and arg(h*) = -(arg(h)). This would give phi_0 = 0.
  If the omega sector (mult 2) contains one h-eigenspace eigenvector, phi_1 = arg(h).
  Then delta = phi_1 - phi_0 = arg(h) != 0.

  Alternatively: if the phase structure is determined by the simultaneous
  eigenvalues, we compute directly from the eigenvector projections.

Rigor status:
  STRICT-SOLID (numerical computation reusing verified B(P) infrastructure).
  All structural inputs (B(P) eigenvalues, C_3 action, (4,2,2) isotypic dims)
  are already verified in ../../predictions/B_P_doubly_degenerate_h_derivation.md and
  docs/theorem_B5_3_core.md.
"""

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from numpy import linalg as la

from proofs.common import find_bonds, C3_PERM, omega3

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

H_EXACT = (math.sqrt(3) + 1j * math.sqrt(5)) / 2   # h = (sqrt(3)+i*sqrt(5))/2
ARG_H = math.atan2(math.sqrt(5), math.sqrt(3))      # arg(h) in radians
ARG_H_DEG = math.degrees(ARG_H)                      # ~52.24 deg
DELTA_OBS = 12.735                                    # PDG delta_obs in degrees
K_P = (0.25, 0.25, 0.25)                             # P-point


# -----------------------------------------------------------------------
# Infrastructure: reused from t_v_eigenstructure.py
# -----------------------------------------------------------------------

def build_directed_edges(bonds):
    directed = [tuple(b) for b in bonds]
    assert len(directed) == 12
    return directed


def bloch_hashimoto(k_frac, directed):
    """12x12 Bloch Hashimoto B(k)."""
    n = len(directed)
    B = np.zeros((n, n), dtype=complex)
    k = np.asarray(k_frac, dtype=float)
    for i_p, (src_p, tgt_p, cell_p) in enumerate(directed):
        for i_e, (src_e, tgt_e, cell_e) in enumerate(directed):
            if tgt_e != src_p:
                continue
            is_reverse = (tgt_p == src_e and
                          tuple(np.array(cell_p) + np.array(cell_e)) == (0, 0, 0))
            if is_reverse:
                continue
            phase = np.exp(2j * np.pi * np.dot(k, cell_p))
            B[i_p, i_e] += phase
    return B


def c3_vertex_perm():
    perm = {}
    for i in range(4):
        for j in range(4):
            if abs(C3_PERM[i, j] - 1.0) < 1e-12:
                perm[j] = i
    assert perm == {0: 0, 1: 3, 2: 1, 3: 2}
    return perm


def c3_cell_perm(cell):
    return (cell[2], cell[0], cell[1])


def build_c3_on_directed_edges(directed):
    vp = c3_vertex_perm()
    n = len(directed)
    edge_to_idx = {de: i for i, de in enumerate(directed)}
    U = np.zeros((n, n), dtype=complex)
    for i, (src, tgt, cell) in enumerate(directed):
        new_edge = (vp[src], vp[tgt], c3_cell_perm(cell))
        j = edge_to_idx.get(new_edge)
        if j is None:
            raise RuntimeError(f"C_3 mapped {(src, tgt, cell)} -> {new_edge} not found")
        U[j, i] = 1.0
    return U


# -----------------------------------------------------------------------
# Simultaneous diagonalization within a subspace
# -----------------------------------------------------------------------

def simultaneous_diag_within_subspace(B_P, U, evecs_B_full, b_evals_full, ram_idx, tol=1e-5):
    """
    Within V_Ram (indexed by ram_idx from the full B_P eigdecomposition),
    find a basis that simultaneously diagonalizes B_P and U_{C_3}.

    Since [B_P, U] = 0 and B_P is already diagonalized, we only need to
    diagonalize U within each degenerate B_P eigenspace. The B_P eigenvalues
    on V_Ram are {h, h*, -h, -h*}, each with multiplicity 2.

    Parameters
    ----------
    B_P : (12, 12) complex array
    U   : (12, 12) complex array, U_{C_3}
    evecs_B_full : (12, 12) complex array, all B_P eigenvectors
    b_evals_full : (12,) complex array, all B_P eigenvalues
    ram_idx      : list of 8 indices into the Ramanujan subspace
    tol          : tolerance for grouping degenerate B_P eigenvalues

    Returns
    -------
    evecs_joint : (12, 8) complex array, simultaneous eigenvectors (rows = full 12-dim)
    b_evals     : (8,) complex array, B_P eigenvalues for each joint eigenvector
    u_evals     : (8,) complex array, U_{C_3} eigenvalues (in {1, omega, omega^2})
    """
    evals_ram = b_evals_full[ram_idx]
    evecs_ram = evecs_B_full[:, ram_idx]   # 12 x 8

    # Group the 8 Ramanujan eigenvectors by B_P eigenvalue (4 groups of 2)
    # Sort by (real, imag) for reproducibility
    sort_key = lambda z: (round(z.real, 5), round(z.imag, 5))
    order = np.argsort([sort_key(e)[0] + 1e-4 * sort_key(e)[1] for e in evals_ram])
    evals_ram = evals_ram[order]
    evecs_ram = evecs_ram[:, order]

    # Build groups of degenerate B_P eigenvalues
    groups = []
    i = 0
    while i < 8:
        grp = [i]
        while i + 1 < 8 and abs(evals_ram[i + 1] - evals_ram[i]) < tol:
            i += 1
            grp.append(i)
        groups.append(grp)
        i += 1

    # For each group, diagonalize U within the degenerate B_P subspace
    result_evecs = np.zeros((12, 8), dtype=complex)
    result_b_evals = np.zeros(8, dtype=complex)
    result_u_evals = np.zeros(8, dtype=complex)

    col = 0
    for grp in groups:
        sub_evecs = evecs_ram[:, grp]   # 12 x len(grp)
        # Orthonormalize (numerical eigenvectors may not be perfectly orthogonal)
        Q_sub, _ = la.qr(sub_evecs)
        Q_sub = Q_sub[:, :len(grp)]

        if len(grp) == 1:
            # Non-degenerate: U eigenvalue from expectation
            v = Q_sub[:, 0]
            u_ev = v.conj() @ U @ v
            result_evecs[:, col] = v
            result_b_evals[col] = evals_ram[grp[0]]
            result_u_evals[col] = u_ev
            col += 1
        else:
            # Degenerate: project U into this 2-dim subspace and diagonalize
            U_restricted = Q_sub.conj().T @ U @ Q_sub  # 2x2
            u_ev_degen, u_ev_vecs = la.eig(U_restricted)
            # Sort by angle for reproducibility
            u_order = np.argsort(np.angle(u_ev_degen))
            u_ev_degen = u_ev_degen[u_order]
            u_ev_vecs = u_ev_vecs[:, u_order]
            for k_idx in range(len(grp)):
                result_evecs[:, col] = Q_sub @ u_ev_vecs[:, k_idx]
                result_b_evals[col] = evals_ram[grp[0]]   # same for all in group
                result_u_evals[col] = u_ev_degen[k_idx]
                col += 1

    return result_evecs, result_b_evals, result_u_evals


# -----------------------------------------------------------------------
# Phase extraction
# -----------------------------------------------------------------------

def classify_c3_sector(u_eval, tol=0.15):
    """Classify a U_{C_3} eigenvalue as trivial (0), omega (1), or omega^2 (2)."""
    if abs(u_eval - 1.0) < tol:
        return 0
    elif abs(u_eval - omega3) < tol:
        return 1
    elif abs(u_eval - omega3 ** 2) < tol:
        return 2
    else:
        raise ValueError(f"Cannot classify U eigenvalue {u_eval}")


def extract_sector_b_evals(b_evals, u_evals, sector):
    """Extract B(P) eigenvalues for a given C_3 sector (0, 1, or 2)."""
    return [b_evals[i] for i in range(len(b_evals)) if classify_c3_sector(u_evals[i]) == sector]


# -----------------------------------------------------------------------
# Main computation
# -----------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Koide delta phase computation")
    print("Simultaneous eigenvectors of B(P) and U_{C_3} in V_Ram")
    print("=" * 72)

    # --- Build operators ---
    bonds = find_bonds()
    directed = build_directed_edges(bonds)
    B_P = bloch_hashimoto(K_P, directed)
    U = build_c3_on_directed_edges(directed)

    print(f"\nh = (sqrt(3)+i*sqrt(5))/2")
    print(f"arg(h)   = {ARG_H_DEG:.4f} deg")
    print(f"arg(h)/4 = {ARG_H_DEG / 4:.4f} deg")
    print(f"delta_obs (PDG) = {DELTA_OBS:.4f} deg")

    # Verify commutation
    comm = la.norm(B_P @ U - U @ B_P)
    assert comm < 1e-10, f"[B(P), U] nonzero: {comm}"
    print(f"\n[B(P), U_{{C_3}}] = 0 verified (norm = {comm:.2e})")

    # --- Extract V_Ram ---
    evals_B, evecs_B = la.eig(B_P)
    ram_idx = [i for i, ev in enumerate(evals_B) if abs(abs(ev) ** 2 - 2.0) < 1e-6]
    assert len(ram_idx) == 8, f"Expected 8 Ramanujan eigenvectors, got {len(ram_idx)}"
    evecs_ram = evecs_B[:, ram_idx]
    print(f"\nV_Ram: {len(ram_idx)} eigenvectors (B eigenvalues with |mu|^2 = 2)")

    # Print the B(P) eigenvalues in V_Ram
    evals_ram = evals_B[ram_idx]
    print("B(P) eigenvalues in V_Ram:")
    for ev in sorted(evals_ram, key=lambda z: (round(z.real, 4), round(z.imag, 4))):
        print(f"  {ev.real:+.6f} {ev.imag:+.6f}i   |mu|={abs(ev):.6f}  arg={math.degrees(np.angle(ev)):.4f} deg")

    # --- Simultaneous diagonalization within V_Ram ---
    print("\nSimultaneous diagonalization of B(P) and U_{C_3} within V_Ram...")
    evecs_joint, b_evals_joint, u_evals_joint = simultaneous_diag_within_subspace(
        B_P, U, evecs_B, evals_B, ram_idx, tol=1e-5
    )

    print("\nJoint (B_P-eval, U_{C_3}-eval) pairs in V_Ram:")
    print(f"  {'Index':>5}  {'B_P eval':>24}  {'arg(B_P eval) deg':>18}  {'U eval':>24}  {'C_3 sector':>12}")
    for i in range(len(b_evals_joint)):
        b_ev = b_evals_joint[i]
        u_ev = u_evals_joint[i]
        b_arg = math.degrees(np.angle(b_ev))
        sector = classify_c3_sector(u_ev)
        sector_name = ['trivial', 'omega', 'omega^2'][sector]
        print(f"  {i:>5}  {b_ev.real:+.6f}{b_ev.imag:+.6f}i  {b_arg:>18.4f}  {u_ev.real:+.6f}{u_ev.imag:+.6f}i  {sector_name:>12}")

    # --- Sector classification ---
    trivial_b_evals = extract_sector_b_evals(b_evals_joint, u_evals_joint, sector=0)
    omega_b_evals = extract_sector_b_evals(b_evals_joint, u_evals_joint, sector=1)
    omega2_b_evals = extract_sector_b_evals(b_evals_joint, u_evals_joint, sector=2)

    print(f"\nC_3-isotypic structure on V_Ram:")
    print(f"  trivial  sector (mult {len(trivial_b_evals)}): B(P) eigenvalues = {[f'{z.real:+.4f}{z.imag:+.4f}i' for z in trivial_b_evals]}")
    print(f"  omega    sector (mult {len(omega_b_evals)}): B(P) eigenvalues = {[f'{z.real:+.4f}{z.imag:+.4f}i' for z in omega_b_evals]}")
    print(f"  omega^2  sector (mult {len(omega2_b_evals)}): B(P) eigenvalues = {[f'{z.real:+.4f}{z.imag:+.4f}i' for z in omega2_b_evals]}")

    assert len(trivial_b_evals) == 4, f"trivial mult = {len(trivial_b_evals)}, expected 4"
    assert len(omega_b_evals) == 2, f"omega mult = {len(omega_b_evals)}, expected 2"
    assert len(omega2_b_evals) == 2, f"omega^2 mult = {len(omega2_b_evals)}, expected 2"

    # --- Phase extraction ---
    # The Koide Fourier sum is:
    #   amp_j = sum_alpha c_alpha * omega^{j*alpha}
    # with c_0 (trivial sector, alpha=0) and c_1 (omega sector, alpha=1).
    # The amplitude c_alpha = sqrt(mu_alpha) * exp(i*phi_alpha).
    # The phase phi_alpha is what we want to compute.
    #
    # The phase comes from the B(P) eigenvalues in each sector. The B(P) eigenvalue
    # on a simultaneous eigenvector with C_3-sector alpha is the "flux eigenvalue"
    # that governs the amplitude in that sector.
    #
    # For the trivial sector (mult 4): the four B(P) eigenvalues are two pairs
    # (h, h*) or (h, -h*) etc. The coherent sum of their phases determines phi_0.
    # For the omega sector (mult 2): the two B(P) eigenvalues determine phi_1.
    #
    # The Koide reading rule uses |c_alpha| = sqrt(mu_alpha), not the complex phase.
    # The delta = 0 result in the real-phase case arises because with all real phases,
    # the trivial sector's contribution is symmetric (equal h and h* contributions),
    # giving phi_0 = 0, and the omega sector's phase phi_1 is also zero.
    #
    # However, the B(P) eigenvalues in each sector are complex. The phase of the
    # GEOMETRIC MEAN of B(P) eigenvalues within a sector is a natural definition of
    # the sector phase.

    print("\n--- Phase analysis ---")

    def sector_mean_arg(b_evals_sector):
        """
        Compute the geometric mean argument of the B(P) eigenvalues in a sector.
        For a sector with eigenvalues mu_1, mu_2, ..., the geometric mean argument
        is arg(mu_1 * mu_2 * ... * mu_n)^{1/n} = (sum arg(mu_i)) / n.
        """
        args = [np.angle(z) for z in b_evals_sector]
        return sum(args) / len(args)

    def sector_product_arg(b_evals_sector):
        """
        Compute the argument of the product of B(P) eigenvalues in a sector.
        This is the total accumulated phase.
        """
        prod = 1.0 + 0j
        for z in b_evals_sector:
            prod *= z
        return np.angle(prod)

    phi_0_mean = sector_mean_arg(trivial_b_evals)
    phi_1_mean = sector_mean_arg(omega_b_evals)
    phi_2_mean = sector_mean_arg(omega2_b_evals)

    phi_0_prod = sector_product_arg(trivial_b_evals)
    phi_1_prod = sector_product_arg(omega_b_evals)
    phi_2_prod = sector_product_arg(omega2_b_evals)

    print(f"\nGeometric mean arg per sector:")
    print(f"  trivial sector phi_0 (mean arg) = {math.degrees(phi_0_mean):+.6f} deg")
    print(f"  omega   sector phi_1 (mean arg) = {math.degrees(phi_1_mean):+.6f} deg")
    print(f"  omega^2 sector phi_2 (mean arg) = {math.degrees(phi_2_mean):+.6f} deg")
    print(f"  relative phase phi_1 - phi_0 (mean) = {math.degrees(phi_1_mean - phi_0_mean):+.6f} deg")

    print(f"\nProduct arg per sector:")
    print(f"  trivial sector phi_0 (prod arg) = {math.degrees(phi_0_prod):+.6f} deg")
    print(f"  omega   sector phi_1 (prod arg) = {math.degrees(phi_1_prod):+.6f} deg")
    print(f"  omega^2 sector phi_2 (prod arg) = {math.degrees(phi_2_prod):+.6f} deg")
    print(f"  relative phase phi_1 - phi_0 (prod) = {math.degrees(phi_1_prod - phi_0_prod):+.6f} deg")

    # --- Individual B(P) eigenvalue args in each sector ---
    print(f"\nIndividual B(P) eigenvalue args by sector:")
    print(f"  trivial sector:")
    for z in trivial_b_evals:
        print(f"    {z.real:+.6f}{z.imag:+.6f}i  arg = {math.degrees(np.angle(z)):+.6f} deg")
    print(f"  omega sector:")
    for z in omega_b_evals:
        print(f"    {z.real:+.6f}{z.imag:+.6f}i  arg = {math.degrees(np.angle(z)):+.6f} deg")
    print(f"  omega^2 sector:")
    for z in omega2_b_evals:
        print(f"    {z.real:+.6f}{z.imag:+.6f}i  arg = {math.degrees(np.angle(z)):+.6f} deg")

    # --- Check whether the trivial sector contains paired h and h* ---
    # If trivial sector = {h, h*, -h, -h*} then the product is |h|^4 = 4 (real),
    # giving phi_0_prod = 0.
    prod_trivial = 1.0 + 0j
    for z in trivial_b_evals:
        prod_trivial *= z
    print(f"\nProduct of trivial sector B(P) evals: {prod_trivial.real:+.6f}{prod_trivial.imag:+.6f}i")
    print(f"  |product| = {abs(prod_trivial):.6f}  arg = {math.degrees(np.angle(prod_trivial)):+.6f} deg")
    print(f"  Is product real? {abs(prod_trivial.imag) < 1e-10}")

    prod_omega = 1.0 + 0j
    for z in omega_b_evals:
        prod_omega *= z
    print(f"Product of omega sector B(P) evals: {prod_omega.real:+.6f}{prod_omega.imag:+.6f}i")
    print(f"  |product| = {abs(prod_omega):.6f}  arg = {math.degrees(np.angle(prod_omega)):+.6f} deg")

    # --- Summary comparison ---
    print("\n" + "=" * 72)
    print("SUMMARY: Relative phase comparison")
    print("=" * 72)
    delta_from_mean = math.degrees(phi_1_mean - phi_0_mean)
    delta_from_prod = math.degrees(phi_1_prod - phi_0_prod)
    print(f"\n  arg(h)        = {ARG_H_DEG:.4f} deg")
    print(f"  arg(h)/4      = {ARG_H_DEG / 4:.4f} deg")
    print(f"  delta_obs     = {DELTA_OBS:.4f} deg (PDG)")
    print(f"\n  phi_1 - phi_0 (mean-arg method)    = {delta_from_mean:+.4f} deg")
    print(f"  phi_1 - phi_0 (prod-arg method)    = {delta_from_prod:+.4f} deg")
    print()

    # Check which h-type eigenvalues appear in which sector
    h_targets = {
        'h':    H_EXACT,
        'h*':   H_EXACT.conjugate(),
        '-h':  -H_EXACT,
        '-h*': -H_EXACT.conjugate(),
    }

    print("B(P) eigenvalue classification by sector:")
    for sector_name, sector_evals in [
        ('trivial', trivial_b_evals),
        ('omega',   omega_b_evals),
        ('omega^2', omega2_b_evals),
    ]:
        labels = []
        for z in sector_evals:
            matched = [name for name, t in h_targets.items() if abs(z - t) < 1e-5]
            labels.append(matched[0] if matched else f"?{z:.4f}")
        print(f"  {sector_name:>8}: {labels}")

    # --- Structural analysis ---
    print("\n--- Structural analysis ---")
    print()
    print("The trivial C_3-sector (mult 4) contains B(P) eigenvalues from")
    print("multiple eigenspaces. If it contains both h and h* (conjugate pair),")
    print("their phases cancel: arg(h) + arg(h*) = 0, so phi_0 = 0 (mean-arg).")
    print()
    print("The omega C_3-sector (mult 2) contains B(P) eigenvalues from one")
    print("or two B(P)-eigenspaces. The mean arg of those eigenvalues is phi_1.")
    print()
    print("The relative phase delta = phi_1 - phi_0 is the Koide delta.")

    return {
        'phi_0_mean_deg': math.degrees(phi_0_mean),
        'phi_1_mean_deg': math.degrees(phi_1_mean),
        'delta_mean_deg': delta_from_mean,
        'phi_0_prod_deg': math.degrees(phi_0_prod),
        'phi_1_prod_deg': math.degrees(phi_1_prod),
        'delta_prod_deg': delta_from_prod,
        'arg_h_deg': ARG_H_DEG,
        'arg_h_over_4_deg': ARG_H_DEG / 4,
        'delta_obs_deg': DELTA_OBS,
        'trivial_b_evals': trivial_b_evals,
        'omega_b_evals': omega_b_evals,
        'omega2_b_evals': omega2_b_evals,
    }


if __name__ == "__main__":
    results = main()
