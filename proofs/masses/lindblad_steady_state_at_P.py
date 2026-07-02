#!/usr/bin/env python3
"""
Canonical prediction file for the Lindblad steady state on the visible
Bloch fibre at the P-point of the srs Hashimoto walker.

NOTE (post-A3, 2026-04-18): Under the three-axiom framework (A1+A2+A3;
docs/framework/framework_axioms.md), G.1 (Hilbert-space structure) and G.5 (complex
field) are DERIVED via CDP 2011 Theorem 25
(predictions/observer_hilbert_space.py). The Hilbert-space premise of
Lindblad dynamics is therefore no longer assumed. The Lindblad-form
derivation from A1+A2+A3 (vs adoption) remains a separate open workstream.

Setup (open-quantum-system reading; cf. an internal Markov-vs-unitary classification audit
"Lindblad" appearance and recommendation; an internal working note):

    The visible sector is the 2|E| = 12-dim Hilbert space of directed
    edges per srs primitive cell, carrying the Hashimoto Bloch operator
    B(P) (../../predictions/B_P_doubly_degenerate_h_derivation.md). The dark sector is a
    probability/measure space (per the framework reading: dark has
    uncompressible chaotic dynamics and is NOT a Hilbert space). The
    correct mathematical formalism for Hilbert+Hilbert decoherence
    coupled to a stochastic environment is a Lindblad master equation.

Construction (Lindblad 1976; Gorini-Kossakowski-Sudarshan 1976;
Breuer & Petruccione 2002 Ch. 3):

    Hamiltonian H on the visible fibre at P:
        H := (B(P) + B(P)^dag)/2     (Hermitian symmetrization)

    Jump operators L_e for each directed edge e in {0, ..., 11}:
        L_e := sqrt(1/k*) * P_e
    where P_e := |e><e| is the rank-1 projector onto directed edge e and
    1/k* = 1/3 is the W4 cancellation rate of theorem_walker_dynamics.md
    Step 4 (the probability that a given toggle reduces the reduced word
    is 1/k* per step). This is the "dephasing" Lindblad model: each W4
    cancellation event erases coherence between directed edges without
    transporting probability between them.

    Probability conservation: sum_e L_e^dag L_e = (1/k*) sum_e P_e
        = (1/k*) * I_12  (since the P_e are a complete orthogonal basis).

    Lindblad equation:
        L(rho) = -i [H, rho]
                 + sum_e ( L_e rho L_e^dag - 0.5 {L_e^dag L_e, rho} ).

Result (closed-form):

    1) Steady state on the 12-dim Bloch fibre at P:
           rho_ss = I/12   (uniformly mixed; unique)
       This follows because sum_e L_e^dag L_e is proportional to the
       identity, so the only fixed point of L is the maximally mixed
       state. Verified numerically by direct kernel computation of the
       144-dim vectorized Lindblad superoperator.

    2) Population on the h-eigenspace at P (Tr P_h = 2,
       ../../predictions/B_P_doubly_degenerate_h_derivation.md Step 7):
           Tr(P_h rho_ss) = Tr(P_h)/12 = 2/12 = 1/6.

    3) Channel-summed jump rate at the h-eigenspace:
           m_h := sum_e Tr(L_e^dag L_e P_h)
                = (1/k*) sum_e (P_h)_{ee}
                = (1/k*) Tr(P_h)
                = 2/k* = 2/3.
       This is the candidate mass scale of the Lindblad / mass-as-flux
       reading at the h-eigenmode (an internal working note
       framing statement).

    4) Bidirectional flux on the h-eigenmode (gain rate from MDL-uncompressed
       toggles + loss rate from cancellation events):
           Phi^bi_h = 2 m_h = 4/k* = 4/3.

Open: the small-q dispersion of the dissipator's spectral density is
|q|^0 (q-independent) because the jump operators L_e = sqrt(1/k*) P_e
are q-independent projectors onto directed-edge basis vectors. Under
the standard cosmological convention <zeta(q) zeta(-q)> ~ |q|^{n_s - 4},
this gives n_s = 4 (worse than the |q|^2 / n_s = 2 stall of
an internal working note). The dissipator therefore
does NOT supply the q-power needed to move n_s toward 0.965. See
an internal working note for the focused scoping of what
does not close.
"""

# ============================================================
# PARAMETER: lindblad_steady_state_at_P
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Value:       rho_ss = I/12 (maximally mixed on the 12-dim Bloch fibre)
#              Tr(P_h rho_ss) = 1/6
#              m_h = 2/k* = 2/3   (channel-summed jump rate on h-subspace)
#              Phi^bi_h = 4/k* = 4/3   (bidirectional flux on h-subspace)
# Source:      Structural prediction of the Lindblad equation on the
#              visible Bloch fibre at P. "Observation" = numerical
#              kernel of the 144-dim vectorized Lindblad superoperator.
# PDG edition: n/a

# --- PREDICTED VALUE -----------------------------------------
# Value:       rho_ss = I/12         (closed form, exact)
#              Tr(P_h rho_ss) = 1/6  (closed form, exact)
#              m_h = 2/k* = 2/3      (closed form, exact)
#              Phi^bi_h = 4/k* = 4/3 (closed form, exact)
# Deviation:   ||L(rho_ss)|| ~ 1.3e-16 (machine precision); smallest
#              singular value of vectorized L is ~ 1.4e-15 (unique
#              steady state).

# --- DERIVED FORMULA -----------------------------------------
# Full proof in predictions/lindblad_steady_state_at_P_derivation.md.
# Skeleton:
#
#   1. Upstream: k* = 3, d = 3 -> srs = I4_132 Wyckoff 8a
#                                       [predictions/k_star.py,
#                                        predictions/d_spatial.py,
#                                        predictions/g_girth_derivation.md §2]
#   2. Upstream: walker dynamics on srs = NB walks; Hashimoto B is
#      the 1-step amplitude operator on the 12-dim directed-edge space;
#      W4 cancellation rate per step = 1/k*.
#                                       [../../predictions/walker_dynamics_derivation.md
#                                        Steps 4, 6, 7]
#   3. Upstream: B(P) has h-eigenspace of multiplicity 2.
#                                       [predictions/B_P_doubly_degenerate_h.py;
#                                        ../../predictions/B_P_doubly_degenerate_h_derivation.md]
#   4. Hamiltonian H = (B(P) + B(P)^dag)/2 is Hermitian by construction
#      (H - H^dag = 0 holds symbolically and numerically).
#   5. Jump operators L_e = sqrt(1/k*) P_e where P_e is the projector
#      onto directed edge e (rank-1 diagonal matrix). The W4 cancellation
#      rate 1/k* is derived per step in walker_dynamics Step 4.
#   6. Probability conservation: sum_e L_e^dag L_e = (1/k*) I_12.
#      Consequence: the maximally mixed state I/12 is the unique steady
#      state (general result for unital Lindbladians; Wolf 2012
#      "Quantum Channels and Operations" Theorem 6.1, or Breuer-
#      Petruccione 2002 §3.2.4).
#   7. Vectorize L: dim 144 = 12 x 12; null space has dimension 1
#      (verified by SVD); null vector is vec(I/12).
#   8. Mass-scale on h-subspace:
#         m_h = sum_e Tr(L_e^dag L_e P_h^orth)
#             = (1/k*) sum_e (P_h^orth)_{ee}
#             = (1/k*) Tr(P_h^orth)
#             = 2/k*  (since P_h^orth is the orthogonal projector onto
#                      the 2-dim h-eigenspace, hence Tr(P_h^orth) = 2).
#   9. Bidirectional flux on h-subspace:
#         Phi^bi_h = (gain rate to h) + (loss rate from h)
#                  = 2 * m_h = 4/k*.
#
# The orthogonal projector P_h^orth is built numerically by running QR
# on the eigenvectors of B(P) at h; the closed-form trace identity is
# basis-independent because Tr is invariant under change of basis.

# --- INPUTS --------------------------------------------------
# symbol      | value             | status    | predictions/ file                            | meaning
# ------------|-------------------|-----------|----------------------------------------------|--------
# k_star      | 3                 | [derived] | predictions/k_star.py                        | coordination number; W4 cancellation rate = 1/k*
# d_spatial   | 3                 | [derived] | predictions/d_spatial.py                     | spatial dimension; selects 3D srs
# srs embed   | I4_132 Wyckoff 8a | [derived] | predictions/g_girth_derivation.md §2         | space group + bond list
# B(P)        | 12x12 complex     | [derived] | predictions/B_P_doubly_degenerate_h.py       | Hashimoto Bloch at P
# h, mult 2   | (sqrt3+i sqrt5)/2 | [derived] | predictions/B_P_doubly_degenerate_h.py       | h-eigenspace dim
# Lindblad    | gen. quantum dyn. | [cited]   | Lindblad 1976; GKS 1976                      | unitarity-preserving CP semigroup

# --- IMPLEMENTATION ------------------------------------------
# Numerical construction of the 144-dim vectorized Lindblad
# superoperator on the 12-dim Bloch fibre at P, with closed-form
# verification of rho_ss = I/12 and the four trace identities.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import sympy as sp

from k_star import predict_k_star
from d_spatial import predict_d_spatial

# Upstream framework values.
d = predict_d_spatial()
k_star = predict_k_star(d)

# Build B(P) using the same primitive-cell bond list as
# predictions/B_P_doubly_degenerate_h.py (I4_132 Wyckoff 8a srs realisation).
from proofs.common import find_bonds  # noqa: E402
import functools

bonds = find_bonds()
directed = [(s, t, c) for (s, t, c) in bonds]
N = len(directed)
assert N == 2 * 6, f"Unexpected directed-edge count {N}, expected 12 for srs primitive cell."


def bloch_hashimoto(k_frac, directed):
    """Bloch Hashimoto matrix at fractional k = (k1, k2, k3)."""
    n = len(directed)
    B = np.zeros((n, n), dtype=complex)
    k = np.asarray(k_frac, dtype=float)
    for ip, (jp_src, jp_tgt, jp_cell) in enumerate(directed):
        for ie, (ie_src, ie_tgt, ie_cell) in enumerate(directed):
            if ie_tgt != jp_src:
                continue
            is_reverse = (jp_tgt == ie_src
                          and tuple(np.array(jp_cell) + np.array(ie_cell)) == (0, 0, 0))
            if is_reverse:
                continue
            phase = np.exp(2j * np.pi * np.dot(k, jp_cell))
            B[ip, ie] += phase
    return B


# ---- B(P) and h-eigenspace ----
P_pt = (0.25, 0.25, 0.25)
B_P = bloch_hashimoto(P_pt, directed)
ev_B, V_B = np.linalg.eig(B_P)

h_target = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
mask_h = np.abs(ev_B - h_target) < 1e-6
h_indices = np.where(mask_h)[0]
assert len(h_indices) == 2, f"Expected mult(h) = 2; got {len(h_indices)}."

V_h_raw = V_B[:, h_indices]
V_h, _ = np.linalg.qr(V_h_raw)  # orthonormal basis of h-eigenspace
P_h = V_h @ V_h.conj().T  # orthogonal projector onto h-eigenspace
trace_P_h = np.trace(P_h).real
assert abs(trace_P_h - 2.0) < 1e-10, f"Tr(P_h) = {trace_P_h}, expected 2."

# ---- Hamiltonian: H = (B + B^dag) / 2 (Hermitian) ----
H = (B_P + B_P.conj().T) / 2
hermitian_residual = np.max(np.abs(H - H.conj().T))
assert hermitian_residual < 1e-12, f"H not Hermitian; residual = {hermitian_residual}."


# ---- Jump operators: L_e = sqrt(1/k*) P_e ----
def projector(e, dim=N):
    P = np.zeros((dim, dim), dtype=complex)
    P[e, e] = 1.0
    return P


gamma = 1.0 / k_star  # W4 cancellation rate per step (walker_dynamics Step 4)
L_ops = [np.sqrt(gamma) * projector(e) for e in range(N)]

# Probability conservation: sum_e L_e^dag L_e = gamma * I_N.
S_check = sum(L.conj().T @ L for L in L_ops)
err_S = np.max(np.abs(S_check - gamma * np.eye(N)))
assert err_S < 1e-12, f"sum_e L_e^dag L_e != gamma * I; residual = {err_S}."

# ---- Vectorized Lindblad superoperator (144 x 144) ----
I_N = np.eye(N, dtype=complex)
L_super = -1j * (np.kron(I_N, H) - np.kron(H.T, I_N))
for L in L_ops:
    LdL = L.conj().T @ L
    L_super = L_super + np.kron(L.conj(), L) - 0.5 * (np.kron(I_N, LdL) + np.kron(LdL.T, I_N))

# ---- Steady-state via SVD: smallest singular value ----
U_sv, S_sv, Vh_sv = np.linalg.svd(L_super)
assert S_sv[-1] < 1e-10, f"No null vector found; smallest sv = {S_sv[-1]}."
assert S_sv[-2] > 1e-3, f"Steady state not unique; second-smallest sv = {S_sv[-2]}."

vec_ss = Vh_sv.conj().T[:, -1]
rho_ss = vec_ss.reshape(N, N)
rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)
rho_ss = rho_ss / np.trace(rho_ss).real

# Verify rho_ss = I/N.
err_rho = np.max(np.abs(rho_ss - np.eye(N) / N))
assert err_rho < 1e-10, f"rho_ss != I/N; residual = {err_rho}."

# Verify L(rho_ss) = 0 directly.
L_of_rho = (
    -1j * (H @ rho_ss - rho_ss @ H)
    + sum(
        L @ rho_ss @ L.conj().T - 0.5 * (L.conj().T @ L @ rho_ss + rho_ss @ L.conj().T @ L)
        for L in L_ops
    )
)
err_L = np.max(np.abs(L_of_rho))
assert err_L < 1e-12, f"L(rho_ss) != 0; residual = {err_L}."

# ---- Trace identities ----
pop_h = np.trace(P_h @ rho_ss).real  # = Tr(P_h)/N = 2/12 = 1/6
mass_h = sum(np.trace(L.conj().T @ L @ P_h).real for L in L_ops)  # = gamma * Tr(P_h) = 2/3
flux_h = 2 * mass_h  # = 4/3

# Closed-form expectations.
pop_h_expected = float(sp.Rational(1, 6))
mass_h_expected = float(sp.Rational(2, 3))
flux_h_expected = float(sp.Rational(4, 3))

assert abs(pop_h - pop_h_expected) < 1e-10, f"pop_h: {pop_h} vs {pop_h_expected}"
assert abs(mass_h - mass_h_expected) < 1e-10, f"mass_h: {mass_h} vs {mass_h_expected}"
assert abs(flux_h - flux_h_expected) < 1e-10, f"flux_h: {flux_h} vs {flux_h_expected}"

print(f"k* = {k_star}, d = {d}")
print(f"Bloch fibre dim N = {N}")
print(f"Hermitian H = (B(P) + B(P)^dag)/2 verified (residual {hermitian_residual:.1e}).")
print(f"Jump operators: 12 dephasing channels L_e = sqrt(1/{k_star}) P_e.")
print(f"sum_e L_e^dag L_e = (1/{k_star}) I  (probability conservation, residual {err_S:.1e}).")
print()
print(f"Vectorized Lindblad superoperator: {L_super.shape}")
print(f"Smallest singular value (steady state):       {S_sv[-1]:.3e}")
print(f"Second-smallest singular value (uniqueness):  {S_sv[-2]:.3e}")
print(f"Steady state rho_ss = I/{N} (maximally mixed); residual {err_rho:.1e}.")
print(f"||L(rho_ss)|| = {err_L:.1e}")
print()
print(f"h-eigenspace at P: dim 2, Tr(P_h) = {trace_P_h:.6f}")
print(f"  Tr(P_h rho_ss)               = {pop_h:.6f}  (expected 1/6 = {pop_h_expected:.6f})")
print(f"  m_h = sum_e Tr(L_e^dag L_e P_h) = {mass_h:.6f}  (expected 2/k* = {mass_h_expected:.6f})")
print(f"  Phi^bi_h = 2 m_h              = {flux_h:.6f}  (expected 4/k* = {flux_h_expected:.6f})")


# --- PURE FUNCTION -------------------------------------------
# Inputs: k_star only. The pure function rebuilds B(P), constructs the
# Lindblad superoperator, and returns the four scalar identities. The
# bond list is forced by k_star = 3 + d_spatial = 3 (g_girth derivation
# §2) so no other physical literals enter.

@functools.lru_cache(maxsize=None)
def predict_lindblad_steady_state_at_P(k_star):
    """
    Constructs the Lindblad master equation on the visible Bloch fibre
    at the P-point of the srs Hashimoto walker, computes the unique
    steady state, and returns four scalar invariants.

    The visible fibre is 12-dim (2|E| directed edges, |E| = 6 for srs
    primitive cell). The Hamiltonian is H = (B(P) + B(P)^dag)/2 with
    B(P) the 12x12 Bloch Hashimoto matrix. The jump operators are 12
    dephasing channels L_e = sqrt(1/k_star) P_e where P_e is the rank-1
    projector onto directed edge e and 1/k_star is the W4 cancellation
    rate per step (walker_dynamics Step 4). The Lindblad equation

        L(rho) = -i [H, rho] + sum_e (L_e rho L_e^dag - 0.5 {L_e^dag L_e, rho})

    has the unique steady state rho_ss = I/12 (because the dissipator
    is unital: sum_e L_e^dag L_e = (1/k_star) I_12).

    Parameters
    ----------
    k_star : int
        Coordination number. Theorem established for k_star = 3 (srs).

    Returns
    -------
    dict with keys:
        'rho_ss_diagonal' : float
            Diagonal entry of rho_ss; equal to 1/12 = 1/N for k_star=3.
        'population_on_h' : float
            Tr(P_h rho_ss); equals Tr(P_h)/N = 2/12 = 1/6 for k_star=3.
        'mass_h' : float
            Channel-summed jump rate at h-subspace; equals 2/k_star.
        'flux_bi_h' : float
            Bidirectional flux on h-subspace; equals 4/k_star.
    """
    if k_star != 3:
        raise ValueError(
            f"lindblad_steady_state_at_P established for k_star = 3 only. "
            f"Got k_star = {k_star}."
        )

    import sys as _sys
    import os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    repo = _os.path.dirname(here)
    if repo not in _sys.path:
        _sys.path.insert(0, repo)
    from proofs.common import find_bonds as _find_bonds

    bonds_local = _find_bonds()
    directed_local = [(s, t, c) for (s, t, c) in bonds_local]
    n_local = len(directed_local)
    if n_local != 12:
        raise RuntimeError(f"Unexpected directed-edge count: {n_local}")

    # B(P) at P = (1/4, 1/4, 1/4).
    P_loc = (0.25, 0.25, 0.25)
    B_loc = np.zeros((n_local, n_local), dtype=complex)
    k_arr = np.asarray(P_loc, dtype=float)
    for ip, (jp_src, jp_tgt, jp_cell) in enumerate(directed_local):
        for ie, (ie_src, ie_tgt, ie_cell) in enumerate(directed_local):
            if ie_tgt != jp_src:
                continue
            is_reverse = (jp_tgt == ie_src
                          and tuple(np.array(jp_cell) + np.array(ie_cell)) == (0, 0, 0))
            if is_reverse:
                continue
            phase = np.exp(2j * np.pi * np.dot(k_arr, jp_cell))
            B_loc[ip, ie] += phase

    # h-eigenspace projector.
    h_loc = (np.sqrt(3) + 1j * np.sqrt(5)) / 2
    ev_loc, V_loc = np.linalg.eig(B_loc)
    mask = np.abs(ev_loc - h_loc) < 1e-6
    idx = np.where(mask)[0]
    if len(idx) != 2:
        raise RuntimeError(f"h-eigenspace mult = {len(idx)} (expected 2).")
    Vh_orth, _ = np.linalg.qr(V_loc[:, idx])
    Ph_loc = Vh_orth @ Vh_orth.conj().T

    # Hermitian Hamiltonian.
    H_loc = (B_loc + B_loc.conj().T) / 2

    # Jump operators.
    gamma_loc = 1.0 / k_star
    L_loc = []
    for e in range(n_local):
        Pe = np.zeros((n_local, n_local), dtype=complex)
        Pe[e, e] = 1.0
        L_loc.append(np.sqrt(gamma_loc) * Pe)

    # Vectorized Lindblad and steady state.
    Iloc = np.eye(n_local, dtype=complex)
    Lsup = -1j * (np.kron(Iloc, H_loc) - np.kron(H_loc.T, Iloc))
    for Lk in L_loc:
        LdL = Lk.conj().T @ Lk
        Lsup = Lsup + np.kron(Lk.conj(), Lk) - 0.5 * (np.kron(Iloc, LdL) + np.kron(LdL.T, Iloc))

    _, sv, Vt = np.linalg.svd(Lsup)
    if sv[-1] > 1e-8 or sv[-2] < 1e-3:
        raise RuntimeError(f"Steady-state degeneracy unexpected: {sv[-3:]}")

    rho_ss_loc = Vt.conj().T[:, -1].reshape(n_local, n_local)
    rho_ss_loc = 0.5 * (rho_ss_loc + rho_ss_loc.conj().T)
    rho_ss_loc = rho_ss_loc / np.trace(rho_ss_loc).real

    pop_h_loc = float(np.trace(Ph_loc @ rho_ss_loc).real)
    mass_h_loc = float(sum(np.trace(Lk.conj().T @ Lk @ Ph_loc).real for Lk in L_loc))
    flux_loc = 2 * mass_h_loc

    return {
        'rho_ss_diagonal': float(np.diag(rho_ss_loc).real[0]),
        'population_on_h': pop_h_loc,
        'mass_h': mass_h_loc,
        'flux_bi_h': flux_loc,
    }


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    impl = {
        'rho_ss_diagonal': 1.0 / N,
        'population_on_h': pop_h,
        'mass_h': mass_h,
        'flux_bi_h': flux_h,
    }
    pure = predict_lindblad_steady_state_at_P(k_star)

    print()
    print("Implementation:")
    for k, v in impl.items():
        print(f"  {k}: {v:.10f}")
    print("Pure function:")
    for k, v in pure.items():
        print(f"  {k}: {v:.10f}")

    for key in impl:
        diff = abs(impl[key] - pure[key])
        assert diff < 1e-10, f"Mismatch for {key}: {impl[key]} vs {pure[key]}"
    print()
    print("OK: outputs agree.")
    print(f"Closed form: rho_ss = I/12; m_h = 2/k* = 2/3; Phi^bi_h = 4/k* = 4/3.")
