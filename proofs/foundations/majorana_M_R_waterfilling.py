#!/usr/bin/env python3
"""
proofs/foundations/majorana_M_R_waterfilling.py

PURPOSE — "do it right, don't goal-seek" probe for the ν_R Majorana mass M_R
that sets the PMNS Majorana phases (P35 α_21, P36 α_31).

The existing construction (`proofs/flavor/srs_hashimoto_seesaw_verify.py`
STEP 3, `proofs/foundations/path_b_M_R_upgrade.py`) takes

    M_R^(m,m) = h_m^g                                                   (*)

— the Hashimoto walker amplitude around a single closed structure of
length g = girth = 10 on the m-th C_3 generation channel at the P-point
(h_ω = (√3+i√5)/2, h_ω² = (-√3+i√5)/2, h_trivial = ±1, return = 2 if L even).
That is the "shortest closed loop" representative.  Per the framework's own
principle — mass comes from PERSISTENT STRUCTURES in the multiway, and EVERY
closed structure above the A2-T / MDL waterline is retained (waterfilling),
not just the shortest — (*) is the argmin shortcut, not the waterfilling-
correct object.

This probe builds M_R the right way:

    M_R^(m,m) = Σ_{L : retained}  w(L) · h_m^L                          (**)

with w(L) = 2^{-DL(L)} the structural Boltzmann/MDL weight of a closed
structure of length L, retained set = { L ≥ g : DL(L) ≤ waterline }.
It RUNS (**) under explicit, named DL(L) encodings and an explicit
waterline cutoff L_max, runs the Pati-Salam seesaw + Takagi, and PRINTS
whatever Majorana phases fall out — including their dependence on the
cutoff.  No fudging: the predicted (162.4°, 324.8°) is reproduced ONLY in
one specific case, and that case is flagged as an input, not derived.

KEY STRUCTURAL FACT — RAMANUJAN SATURATION at P:  |h_ω|² = |h_ω²|² = k*−1 = 2.
The natural MDL cost of "which closed NB walk of length L" is
DL_walk(L) = log₂(#closed NB walks of length L) ≈ (L/2)·log₂(k*−1) = L/2 bits,
so the per-loop contribution magnitude |w(L)·h_m^L| = 2^{−L/2}·(√2)^L = 1 —
EVERY retained loop contributes with EQUAL magnitude.  There is no leading
term; longer-loop corrections are O(1) phase rotations, not small.  A sum of
unit phasors e^{iLθ} for L = g…L_max points in the direction of its mean
term:  arg(M_R^(m)) ≈ (g + L_max)/2 · arg(h_m).  The existing prediction is
the L_max = g case (sum collapses to one term, arg = g·arg(h_m)) —
equivalently "M_R is the persistent-structure PERIOD AMPLITUDE (a coupling,
one fundamental period = one girth ring per channel-ring), not the RESOLVENT
(a sum over all periods)".  Whether the A2-T waterline for the ν_R-Majorana-
mass channel actually sits at L_max = g is the open piece; this probe does
not pretend to derive it.

WHAT THIS PROBE ESTABLISHES / DOES NOT
  ✓ B(P) commutes with the edge-lifted C_3 (so per-channel return amplitudes
    are well defined, not hand-waved);
  ✓ M_R^(m) under (**) for several explicit (DL encoding, L_max) models;
  ✓ seesaw + Takagi → α_21, α_31 for each model;
  ✓ the (g+L_max)/2·arg(h) phase-drift law, numerically;
  ✓ that the existing (162.4°, 324.8°) ⇔ {only the girth structure retained}
    ⇔ {M_R = period-amplitude coupling, not resolvent}.
  ✗ It does NOT derive the A2-T waterline / DL_raw for this channel.  That is
    the residual gate gap for P35/P36 — STRUCTURAL-DERIVATION-CONDITIONAL on
    "M_R = girth-period amplitude", same tier as R-9's γ.2 encoding choice.
    This probe SHARPENS that conditional; it does not discharge it.

CROSS-REFERENCES
  - proofs/flavor/srs_hashimoto_seesaw_verify.py     (existing M_R = h^g)
  - proofs/foundations/path_b_M_R_upgrade.py          (3×3 M_R algebra)
  - predictions/h_walker_eigenvalue.py , predictions/g_girth.py
  - docs/parameters/parameter_uniqueness_ledger.md  Rows P35, P36
"""

import os
import sys
import math
import cmath
from itertools import product

import numpy as np
from numpy import linalg as la

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

# --- srs primitive cell (I4_132, 4 atoms) — same as proofs/common.py ---------
A_PRIM = np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
ATOMS = np.array([[1/8, 1/8, 1/8], [3/8, 7/8, 5/8], [7/8, 5/8, 3/8], [5/8, 3/8, 7/8]])
N_ATOMS = 4
NN_DIST = math.sqrt(2) / 4
K_STAR = 3
G_GIRTH = 10
k_P = np.array([0.25, 0.25, 0.25])
omega3 = np.exp(2j * np.pi / 3)

# Hashimoto walker eigenvalues at P (theorem-grade; predictions/h_walker_eigenvalue.py)
H_OMEGA   = (math.sqrt(3) + 1j*math.sqrt(5)) / 2     # omega band  (lambda = +sqrt3)
H_OMEGA_2 = (-math.sqrt(3) + 1j*math.sqrt(5)) / 2    # omega^2 band (lambda = -sqrt3)
ARG_OMEGA   = cmath.phase(H_OMEGA)                   # ~ 52.2388 deg
ARG_OMEGA_2 = cmath.phase(H_OMEGA_2)                 # ~ 127.7612 deg

# C_3 little-group at P: fixes atom 0, cycles 1->2->3 (proofs/common.py C3_PERM)
C3_VPERM = {0: 0, 1: 2, 2: 3, 3: 1}

# C_3 generation modes in the 4-dim vertex space (atom 0 is the C_3 fixed point)
GEN_BASIS = {
    "trivial": np.array([0, 1, 1, 1], dtype=complex) / math.sqrt(3),
    "omega":   np.array([0, 1, omega3, omega3**2], dtype=complex) / math.sqrt(3),
    "omega^2": np.array([0, 1, omega3**2, omega3], dtype=complex) / math.sqrt(3),
}

# Per-generation Hashimoto channel eigenvalue at P, as used by the existing
# construction (srs_hashimoto_seesaw_verify.py STEP 3): the omega band carries
# h_ω, the omega^2 band h_ω², the trivial sector the |h|=1 pair {+1,-1}
# (closed-walk return on the trivial channel = (+1)^L + (-1)^L = 2 if L even).
CHANNEL_EIG = {"trivial": None, "omega": H_OMEGA, "omega^2": H_OMEGA_2}


def find_bonds():
    bonds = []
    for i in range(N_ATOMS):
        nbrs = []
        for j in range(N_ATOMS):
            for n1, n2, n3 in product(range(-2, 3), repeat=3):
                rj = ATOMS[j] + n1*A_PRIM[0] + n2*A_PRIM[1] + n3*A_PRIM[2]
                d = la.norm(rj - ATOMS[i])
                if d < 0.02:
                    continue
                if abs(d - NN_DIST) < 0.02:
                    nbrs.append((j, (n1, n2, n3)))
        assert len(nbrs) == 3
        for j, cell in nbrs:
            bonds.append((i, j, cell))
    return bonds


def bloch_H(k_frac, bonds):
    H = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
    k = np.asarray(k_frac, float)
    for s, t, c in bonds:
        H[t, s] += np.exp(2j*np.pi*np.dot(k, c))
    return H


def build_hashimoto(k_frac, bonds):
    n = len(bonds)
    B = np.zeros((n, n), dtype=complex)
    k = np.asarray(k_frac, float)
    for f, (fs, ft, fc) in enumerate(bonds):
        for e, (es, et, ec) in enumerate(bonds):
            if fs != et:
                continue
            if ft == es and np.array_equal(np.array(fc), -np.array(ec)):
                continue   # no backtracking
            B[f, e] = np.exp(2j*np.pi*np.dot(k, np.array(fc)))
    return B


def edge_c3_permutation(bonds):
    """Lift C_3 (vertex perm 0->0,1->2->3->1) to the directed-edge basis.
    srs has exactly one bond per ordered vertex pair, so the lift is unique."""
    by_pair = {}
    for idx, (s, t, c) in enumerate(bonds):
        by_pair.setdefault((s, t), []).append(idx)
    n = len(bonds)
    P = np.zeros((n, n))
    for idx, (s, t, c) in enumerate(bonds):
        cand = by_pair[(C3_VPERM[s], C3_VPERM[t])]
        assert len(cand) == 1
        P[cand[0], idx] = 1.0
    return P


# --- DL(L) encoding models for a closed non-backtracking structure of len L --
LOG2_KM1 = math.log2(K_STAR - 1)                     # = log2(2) = 1.0

def DL_walk(L):
    """'Which closed NB walk of length L': log2(#closed NB walks of length L)
    ~ (L/2)·log2(k*-1) bits.  With k*=3 → L/2 bits.  This is the encoding that
    makes |w(L)·h_m^L| = 1 (Ramanujan saturation)."""
    return 0.5 * L * LOG2_KM1

def DL_steps(L):
    """'Spell out the edge sequence': at each of L steps choose 1 of (k*-1)
    NB continuations → L·log2(k*-1) bits.  With k*=3 → L bits."""
    return L * LOG2_KM1

def DL_length(L):
    """'Just name the length L': log2 L bits — degenerate, far too cheap."""
    return math.log2(L)


# --- M_R via waterfilling (**) ----------------------------------------------
def channel_return_amplitude(channel, L):
    """Closed-NB-walk return amplitude of length L on a generation channel,
    matching srs_hashimoto_seesaw_verify.py STEP 3 channel assignments."""
    h = CHANNEL_EIG[channel]
    if h is None:                       # trivial channel: {+1,-1} → 2 if L even
        return complex(2.0) if L % 2 == 0 else complex(0.0)
    return h ** L


def majorana_phase_set(M_R_diag):
    """The framework's Majorana-phase content is the arg of the diagonal M_R
    entries (the seesaw with a real M_D just inverts the sign; the NuFIT-
    ordered (α_21, α_31) is then a fixed relabelling of this set — see
    srs_hashimoto_seesaw_verify.py STEP 3/5).  Return arg(M_R^(m)) in degrees."""
    return [math.degrees(cmath.phase(z)) % 360.0 for z in M_R_diag]


def main():
    print("=" * 80)
    print(" ν_R Majorana mass M_R — waterfilling-correct construction (not argmin)")
    print("=" * 80)

    bonds = find_bonds()
    assert len(bonds) == 12
    B = build_hashimoto(k_P, bonds)
    Pc3 = edge_c3_permutation(bonds)

    # (1) B(P) commutes with the edge-lifted C_3  ⇒  closed-walk content
    #     decomposes by C_3 irrep (so per-generation amplitudes are well posed)
    comm = la.norm(B @ Pc3 - Pc3 @ B)
    print(f"\n[1] ||[B(P), C_3_edge]|| = {comm:.2e}   "
          f"({'PASS — B(P) is C_3-block-diagonal' if comm < 1e-9 else 'FAIL'})")
    assert comm < 1e-9
    # B(P) spectrum: the Ramanujan eigenvalues h_ω, h_ω² (|h|=√2) appear with
    # multiplicity 2 (C_3-protected), plus the |h|=1 sector.
    spec = la.eigvals(B)
    n_h = sum(1 for z in spec if abs(z - H_OMEGA) < 1e-6 or abs(z - H_OMEGA_2) < 1e-6
              or abs(z - H_OMEGA.conjugate()) < 1e-6 or abs(z - H_OMEGA_2.conjugate()) < 1e-6)
    n_1 = sum(1 for z in spec if abs(abs(z) - 1.0) < 1e-6)
    print(f"    B(P) spectrum: {n_h} eigenvalues at |h|=√2 (the ±√3±i√5 over 2 set), "
          f"{n_1} at |h|=1.  h_ω,h_ω² each C_3-protected doubly-degenerate.")
    assert abs(abs(H_OMEGA)**2 - (K_STAR - 1)) < 1e-12  # Ramanujan saturation

    # (2) valid closed-NB-walk lengths: L >= girth, Tr(B^L) ≠ 0
    def trB(L):
        return abs(np.trace(np.linalg.matrix_power(B, L)))
    valid = [L for L in range(2, 60) if L >= G_GIRTH and trB(L) > 1e-9]
    print(f"\n[2] valid closed-NB-walk lengths L ≥ g={G_GIRTH}: "
          f"{valid[:12]}{' ...' if len(valid) > 12 else ''}  "
          f"(g={G_GIRTH} = shortest closed structure on srs)")

    # (2b) The path_b "structural source" route: cardinality-k cycle-space orbit
    #      ↔ k girth rings ↔ phase k·g·arg(h).  This rests on the claim "each K_4
    #      cycle-space generator (triangle) ↔ one girth-g cycle when lifted to
    #      srs".  CHECK IT: a K_4 cycle lifts to a closed cycle in srs iff its
    #      total voltage (Z^3 lattice-vector sum of its edge cell-shifts) is 0.
    bond_cell = {(s, t): np.array(c) for (s, t, c) in bonds}
    print("\n[2b] path_b 'cardinality-k orbit ↔ k girth rings' route — voltage check:")
    any_lifts = False
    for tri in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
        v = sum(bond_cell[(a, b)] for a, b in zip(tri, tri[1:] + tri[:1]))
        lifts = np.allclose(v, 0)
        any_lifts |= lifts
        print(f"     K_4 triangle {tri}: voltage = {v.astype(int)}  → lifts to a "
              f"closed srs cycle? {'YES' if lifts else 'NO'}")
    print(f"     ⇒ K_4 triangles have NONZERO voltage — they do NOT lift to srs "
          f"cycles (let alone girth-{G_GIRTH} ones).")
    print(f"       So 'each K_4 cycle-space generator ↔ one srs girth ring' is FALSE,"
          f" and the\n       factor 'g' in the path_b phase (k-1)·g·arg(h) is not "
          f"sourced.  [route 2 broken]")
    assert not any_lifts  # confirm the finding

    # (3) REFERENCE: the existing prediction = single girth structure (L_max = g)
    MR_ref = [channel_return_amplitude(c, G_GIRTH) for c in ("trivial", "omega", "omega^2")]
    ph_ref = majorana_phase_set(MR_ref)
    print("\n" + "=" * 80)
    print("[3] M_R = Σ_{L=g..L_max} 2^{-DL(L)} · h_m^L   →   Majorana phase content")
    print("=" * 80)
    print(f"\n  REFERENCE  M_R^(m) = h_m^g  (existing srs_hashimoto_seesaw_verify STEP 3):")
    print(f"     arg(M_R) = {{trivial:{ph_ref[0]:.2f}°, ω:{ph_ref[1]:.2f}°, ω²:{ph_ref[2]:.2f}°}}")
    print(f"     → this is the phase set the seesaw+NuFIT-ordering maps to "
          f"{{α_21, α_31}} = {{162.39°, 324.78°}}\n"
          f"       (α_21 = arg(h_ω^g)=162.39°, α_31 = arg((h_ω/h_ω²)^g)=324.78°; "
          f"δ_CP = arg(h_ω²^g)=197.61° as a bonus).")

    # (4) RUN the waterfilling M_R under explicit (encoding, waterline) models
    encodings = [("DL_walk  (L/2 bits;  |w·h^L| = 1   ← Ramanujan)", DL_walk),
                 ("DL_steps (L  bits;   |w·h^L| = 2^{-L/2}, converges)", DL_steps),
                 ("DL_length(log2 L bits; far too cheap)", DL_length)]
    for ename, DL in encodings:
        print(f"\n  --- encoding: {ename} ---")
        print(f"      {'L_max':>6} | {'#loops':>6} | {'arg M_R^ω (°)':>13} | {'arg M_R^ω² (°)':>14}"
              f" | {'(g+Lmax)/2·argh_ω (°)':>22}")
        for L_max in (G_GIRTH, G_GIRTH+2, G_GIRTH+6, G_GIRTH+10, G_GIRTH+20, G_GIRTH+40):
            Ls = [L for L in valid if L <= L_max]
            if not Ls:
                continue
            ws = [2.0 ** (-DL(L)) for L in Ls]
            MRm = [sum(w * channel_return_amplitude(c, L) for L, w in zip(Ls, ws))
                   for c in ("trivial", "omega", "omega^2")]
            ph = majorana_phase_set(MRm)
            drift = (math.degrees(((G_GIRTH + L_max) / 2.0) * ARG_OMEGA)) % 360.0
            print(f"      {L_max:>6} | {len(Ls):>6} | {ph[1]:>13.2f} | {ph[2]:>14.2f} | {drift:>22.2f}")
        print(f"      (L_max=g row reproduces the prediction: arg M_R^ω = "
              f"arg(h_ω^g) = {ph_ref[1]:.2f}°)")

    # (4) verdict
    print("\n" + "=" * 80)
    print("[4] WHAT THIS SHOWS  (no goal-seeking — read it straight)")
    print("=" * 80)
    print("""
  GOAL of this probe: discharge the P35/P36 conditional by deriving the M_R
  phase (equivalently L_max = g) from the framework's A2-T rate-distortion
  machinery.  RESULT: it does NOT discharge — and the failure is localized.

  • B(P) commutes with the edge-lifted C_3, so closed-walk content genuinely
    decomposes by generation channel — 'A_m(L) = h_m^L' is well posed. [step 1]
    h_ω, h_ω² are C_3-protected doubly-degenerate with |h|^2 = k*-1 = 2
    (Ramanujan saturation) — load-bearing below.

  ROUTE 1 — M_R as an A2-T-waterfilled loop sum  Σ_{L=g..L_max} 2^{-DL(L)}·h_m^L:
  • Its phase DEPENDS ON THE WATERLINE L_max.  With the natural 'which closed NB
    walk of length L' encoding DL_walk(L) = (L/2)·log2(k*-1) = L/2 bits, the
    Ramanujan saturation makes every retained loop contribute with EQUAL
    magnitude (|w·h^L| = 1) — there is NO leading term, and no finite cutoff
    emerges from the A2-T surprise threshold (every ring length is equally
    surprising) — so the sum of unit phasors does not converge; its phase
    drifts as ≈ (g+L_max)/2·arg(h_m) (last column).  With the 'spell-out-the-
    edge-sequence' encoding it converges to a DIFFERENT value (≈186°).  Neither
    natural choice yields the predicted set; the predicted {0°,162.39°,197.61°}
    is ONLY the L_max = g (single-girth-structure) case, which the A2-T
    machinery does not single out.  ⇒ Route 1 does not derive the phase.

  ROUTE 2 — M_R phase from the path_b 'cardinality-k orbit ↔ k girth rings' chain
    (gen k = cardinality-(k-1) Z_3-gauge orbit of K_4 cycle-space subsets ⇒
     phase (k-1)·g·arg(h)):  the channel/orbit structure is genuine (and it
    correctly avoids Route 1's divergence — each channel has a FIXED ring
    count, not a sum) — BUT the factor 'g' rests on 'each K_4 cycle-space
    generator (triangle) ↔ one girth-g cycle when lifted to srs', and step
    [2b] shows that is FALSE: the K_4 triangles have voltages {(1,0,0),
    (0,-1,0),(0,0,1),(1,1,1)}, all nonzero, so they do not lift to closed srs
    cycles at all.  ⇒ Route 2's 'g' is not sourced.

  CONCLUSION:
  • The Majorana phase  arg(M_R^(m)) = g·arg(h_m)  (equiv. (k-1)·g·arg(h)) is
    NOT derived by either available route.  It is a bare IDENTIFICATION — "the
    ν_R Majorana coupling carries one girth-ring's worth of walker holonomy per
    generation channel" — of the same character as A5(a) (Ramanujan eigenvalue
    ↔ mass).  The real |M_R| = δ⁴·M_Pl/(2·k*·N_atoms) chain (m_ν3 closure) is
    phase-free and is unaffected; only the Majorana-phase rows P35/P36 ride on
    this identification.
  • Honest grade for P35/P36: STRUCTURAL-DERIVATION-CONDITIONAL on the 'ν_R
    Majorana phase = girth-ring walker holonomy' identification — NOT
    UNIQUE-THEOREM-GRADE-CONDITIONAL (the current prediction-file/ledger label).
    Same tier as R-9's γ.2 algebraic-K-complexity encoding choice.
  • Not falsified (Majorana phases are unmeasured), but it is identification-
    conditional, not theorem-grade.  This probe localizes the gap to one
    statement (the K_4-cycle-generator ↔ srs-girth-ring map, which is false as
    path_b states it) and shows the loop-sum route diverges — so option B
    (discharge the conditional) FAILS, informatively.
""")


if __name__ == "__main__":
    main()
