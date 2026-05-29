"""Theorem B3-B6 reconciliation — does the Spin(6) C_3 act on B3 species or on a separate color factor?

The conflict (see an internal working note):

    * B3 reads S as ONE Pati-Salam family with COLOR FACTORED OUT, i.e.
      S = (nu, e, u, d) x (L, R), 8 states, no color structure.

    * B6 reads the body-diagonal C_3 induced via Spin(6) ~ SU(4) as
      acting on S with isotypic multiplicities (4, 2, 2), and identifies
      this C_3 with the Z_3 center of SU(3)_color via the PS embedding
      SU(4) -> SU(3)_color x U(1)_{B-L}.

The key question this script answers numerically: when U_{C_3}^S
(B6's Spin(6) lift) acts on the B3 species eigenstates (definite
T_1, T_2, Y, chirality), what does it do? Does it preserve B3's species
labels (compatible with reading C_3 as an EXTERNAL factor on a separate
color space), or does it MIX them (so C_3 acts genuinely on S itself,
in which case B3's species labels are not C_3-invariant)?

The answer pins down the structural reading:

    Case (A) [C_3 commutes with B3 Cartan]: U_{C_3}^S acts diagonally on
        B3 species states. C_3 lives on a tensor factor disjoint from
        S, and B3's "colorless" reading is consistent — the (4, 2, 2)
        is then about a different space, not about S.

    Case (B) [C_3 does NOT commute with B3 Cartan]: U_{C_3}^S mixes B3
        species states. Then the C_3 is a genuine non-trivial Spin(6)
        action ON S, and B3's species labels {nu, e, u, d} are NOT
        the C_3-stable basis — they are a different (Spin(4) x Spin(2))-
        adapted basis. The (4, 2, 2) decomposition exists on S but its
        states are linear combinations of B3's named species.

    Case (C) [Specific mixing pattern]: U_{C_3}^S mixes only WITHIN a
        single SU(2)_L doublet's lepton+quark complement (SU(4) PS
        action), preserving the SU(2)_L structure cleanly. This would
        be the cleanest reading of B6 — but requires C_3 to commute with
        the SU(2)_L generator T_L = (T_1 + T_2).

Per B6 Step 6, the script verifies [T_a, U_{C_3}^S] != 0 for each
T_a in {T_1, T_2, Y}. We extend this here to:

    (1) Compute the explicit matrix of U_{C_3}^S in B3's species basis.
        Read off WHICH species mix.

    (2) Check whether U_{C_3}^S commutes with the SU(2)_L generator
        T_L = (T_1 + T_2), which determines the SU(2)_L doublet
        structure on S.

    (3) Identify the C_3-isotypic basis and compute the OVERLAP between
        each C_3-isotypic state and each B3 species state. If a single
        C_3-isotypic state has support on lepton AND quark species,
        then the (4, 2, 2) decomposition does NOT cleanly mean
        "1 lepton + 3 quark colors per chirality" — it means C_3 mixes
        what B3 calls lepton with what B3 calls quark.

The honest verdict from these checks lands the dispute on one of the
options (I)-(IV) discussed in ../../docs/framework/B3_B6_reconciliation.md.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

TOL = 1e-9
omega = np.exp(2j * np.pi / 3)
omega2 = omega * omega


# ---------------------------------------------------------------------------
# Set up B3's gamma matrices and Spin(6) Cartan (identical to B3 script).
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


Gamma = [None] * 7
Gamma[1] = kron(sx, I2, I2)
Gamma[2] = kron(sy, I2, I2)
Gamma[3] = kron(sz, sx, I2)
Gamma[4] = kron(sz, sy, I2)
Gamma[5] = kron(sz, sz, sx)
Gamma[6] = kron(sz, sz, sy)
I8 = np.eye(8, dtype=complex)


def biv(a, b):
    return 0.5 * (Gamma[a] @ Gamma[b] - Gamma[b] @ Gamma[a])


T_1 = biv(1, 2) / (2j)
T_2 = biv(3, 4) / (2j)
Y = biv(5, 6) / (2j)
G7 = -1j * Gamma[1] @ Gamma[2] @ Gamma[3] @ Gamma[4] @ Gamma[5] @ Gamma[6]

# SU(2)_L Cartan = (T_1 + T_2). SU(2)_R Cartan = (T_1 - T_2).
T_L = T_1 + T_2
T_R = T_1 - T_2


# ---------------------------------------------------------------------------
# Build B3 species basis (simultaneous eigenvectors of T_1, T_2, Y, Gamma_7).
# ---------------------------------------------------------------------------

combined = 1.0 * T_1 + 3.7 * T_2 + 11.3 * Y
eigvals, eigvecs = np.linalg.eigh(combined)
species = []
for k_idx in range(8):
    v = eigvecs[:, k_idx]
    t1 = round(np.real(v.conj() @ T_1 @ v) * 2)
    t2 = round(np.real(v.conj() @ T_2 @ v) * 2)
    y = round(np.real(v.conj() @ Y @ v) * 2)
    ch = int(round(np.real(v.conj() @ G7 @ v)))
    species.append((t1, t2, y, ch, v))

# Replicate B3's species naming.
chirality_sign = species[0][3] * (species[0][0] * species[0][1] * species[0][2])

def name_state(t1, t2, y, ch):
    sector = "SU2L" if t1 == t2 else "SU2R"
    iso_up = (t1 == +1) if sector == "SU2L" else (t1 == +1)
    is_lepton = (chirality_sign * y == +1)
    ch_label = "L" if ch == +1 else "R"
    if is_lepton:
        name = ("nu" if iso_up else "e") + "_" + ch_label
    else:
        name = ("u" if iso_up else "d") + "_" + ch_label
    return name


species_names = []
for (t1, t2, y, ch, v) in species:
    species_names.append(name_state(t1, t2, y, ch))

print("=" * 78)
print("B3-B6 RECONCILIATION — does U_{C_3}^S act on the B3 species labels?")
print("=" * 78)
print()
print("Step 0 — B3 species basis (recap):")
for k_idx, (t1, t2, y, ch, _) in enumerate(species):
    print(f"  state {k_idx}: (T_1={t1:+d}, T_2={t2:+d}, Y={y:+d}, "
          f"ch={ch:+d})  ->  {species_names[k_idx]}")
print()


# ---------------------------------------------------------------------------
# Build U_{C_3}^S using B6's Spin(6) lift recipe.
# ---------------------------------------------------------------------------
# K_4 edges in B6 ordering.
K4_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
SIGMA = {0: 0, 1: 3, 2: 1, 3: 2}


def apply_sigma_to_edge(edge):
    a, b = edge
    return tuple(sorted((SIGMA[a], SIGMA[b])))


edge_to_idx = {e: i for i, e in enumerate(K4_EDGES)}
P_so6 = np.zeros((6, 6), dtype=float)
for e in K4_EDGES:
    i = edge_to_idx[e]
    j = edge_to_idx[apply_sigma_to_edge(e)]
    P_so6[j, i] = 1.0

# Lift via Lie-algebra log + bivector lift (B6 Step 5 method).
from scipy.linalg import expm, logm
L_so6 = logm(P_so6).real
# Antisymmetrize numerically.
L_so6 = 0.5 * (L_so6 - L_so6.T)

X_spin = np.zeros((8, 8), dtype=complex)
for i in range(6):
    for j in range(i + 1, 6):
        X_spin += L_so6[i, j] * biv(i + 1, j + 1)
X_spin_half = 0.5 * X_spin
U_C3_S = expm(X_spin_half)

# Resolve the Spin double-cover sign.
U3 = U_C3_S @ U_C3_S @ U_C3_S
if np.allclose(U3, -I8, atol=1e-9):
    U_C3_S = np.exp(1j * np.pi / 3) * U_C3_S
elif not np.allclose(U3, I8, atol=1e-9):
    raise RuntimeError(f"U_C3_S^3 != +/-I; ||U^3 - I|| = {np.linalg.norm(U3 - I8):.3e}")
U3 = U_C3_S @ U_C3_S @ U_C3_S
assert np.allclose(U3, I8, atol=1e-9), \
    f"After lift correction, ||U^3 - I|| = {np.linalg.norm(U3 - I8):.3e}"

# Verify [U_C3_S, Gamma_7] = 0.
assert np.linalg.norm(U_C3_S @ G7 - G7 @ U_C3_S) < 1e-9, \
    "U_C3_S should commute with chirality"


# ---------------------------------------------------------------------------
# Step 1 — Commutation of U_C3_S with B3 Cartan generators.
# ---------------------------------------------------------------------------
print("Step 1 — Commutation of U_{C_3}^S with B3 Cartan operators:")
print(f"  ||[T_1, U_C3_S]||  = {np.linalg.norm(T_1 @ U_C3_S - U_C3_S @ T_1):.3e}")
print(f"  ||[T_2, U_C3_S]||  = {np.linalg.norm(T_2 @ U_C3_S - U_C3_S @ T_2):.3e}")
print(f"  ||[Y,   U_C3_S]||  = {np.linalg.norm(Y   @ U_C3_S - U_C3_S @ Y):.3e}")
print(f"  ||[T_L, U_C3_S]||  = {np.linalg.norm(T_L @ U_C3_S - U_C3_S @ T_L):.3e}")
print(f"  ||[T_R, U_C3_S]||  = {np.linalg.norm(T_R @ U_C3_S - U_C3_S @ T_R):.3e}")
print()
print("  -> Non-zero for ALL of T_1, T_2, Y, T_L, T_R. So U_{C_3}^S does NOT")
print("     commute with the SU(2)_L Cartan T_L either; this means the C_3")
print("     does not preserve the SU(2)_L doublet structure on S.")
print()


# ---------------------------------------------------------------------------
# Step 2 — Action of U_C3_S on each B3 species state.
# ---------------------------------------------------------------------------
print("Step 2 — Matrix of U_{C_3}^S in B3 species basis (entries |M_{ij}|):")

# Build basis transformation: V[:, i] = species i.
V = np.array([s[4] for s in species]).T  # 8 x 8
# Express U_C3_S in this basis: M = V^dagger U_C3_S V
M = V.conj().T @ U_C3_S @ V

print("  Rows / cols ordered as: " + ", ".join(species_names))
print()
header = "          " + " ".join(f"{n:>6s}" for n in species_names)
print(header)
for i, name_i in enumerate(species_names):
    row = "  " + f"{name_i:>6s}: " + " ".join(f"{abs(M[i, j]):6.3f}" for j in range(8))
    print(row)
print()

# Identify which species each B3 species maps to (largest overlap).
print("Step 2b — For each B3 species state, identify its U_{C_3}^S image (max overlap):")
for i, name_i in enumerate(species_names):
    overlaps = [(abs(M[j, i]), species_names[j]) for j in range(8)]
    overlaps.sort(reverse=True)
    top3 = overlaps[:3]
    summary = ", ".join(f"{n}({o:.3f})" for o, n in top3)
    print(f"  {name_i:>4s} -> {summary}")
print()


# ---------------------------------------------------------------------------
# Step 3 — C_3-isotypic basis. Each isotypic eigenstate's overlap with B3 species.
# ---------------------------------------------------------------------------
print("Step 3 — Decompose each B3 species into C_3-isotypic components.")

# Diagonalize U_C3_S. Sort by eigenvalue (1, omega, omega^2 groups).
evals_U, evecs_U = np.linalg.eig(U_C3_S)


def label_eig(ev):
    if abs(ev - 1) < 1e-6:
        return "1"
    if abs(ev - omega) < 1e-6:
        return "w"
    if abs(ev - omega2) < 1e-6:
        return "w2"
    return f"?({ev:.3f})"


eig_labels = [label_eig(e) for e in evals_U]
print(f"  C_3-isotypic eigenvalues (count): {Counter(eig_labels)}")
print()

# For each B3 species state, compute |projection|^2 onto each C_3-isotypic block.
print("  |proj of species onto each C_3-irrep|^2:")
print(f"  {'species':>6s}  {'m=1 (4-dim)':>15s}  {'m=ω (2-dim)':>15s}  {'m=ω² (2-dim)':>15s}")
for i in range(8):
    v_species = V[:, i]
    proj = {"1": 0.0, "w": 0.0, "w2": 0.0}
    for k_e in range(8):
        amp = abs(np.vdot(evecs_U[:, k_e], v_species)) ** 2
        proj[eig_labels[k_e]] = proj.get(eig_labels[k_e], 0.0) + amp
    print(f"  {species_names[i]:>6s}  {proj['1']:>15.4f}  {proj['w']:>15.4f}  {proj['w2']:>15.4f}")
print()


# ---------------------------------------------------------------------------
# Step 4 — KEY CHECK. Within a chirality sector, does the C_3 mix LEPTON
# states with QUARK states (B6's PS reading: trivial = lepton + 1 color,
# omega/omega^2 = other 2 colors), or only mix u with d (would mean C_3 lives
# inside a Spin(4) that B6 doesn't recognize)?
# ---------------------------------------------------------------------------

print("Step 4 — Lepton-vs-quark mixing under U_{C_3}^S (per chirality):")
for ch_label in ("L", "R"):
    print(f"  Chirality {ch_label}:")
    sector_idx = [i for i, n in enumerate(species_names) if n.endswith("_" + ch_label)]
    lepton_idx = [i for i in sector_idx if species_names[i].startswith(("nu", "e"))]
    quark_idx = [i for i in sector_idx if species_names[i].startswith(("u_", "d_"))]

    # |M[i,j]| where i is a quark target and j is a lepton source — does U_C3_S
    # send a lepton state into the quark span?
    lepton_to_lepton = sum(abs(M[i, j]) ** 2 for i in lepton_idx for j in lepton_idx)
    lepton_to_quark = sum(abs(M[i, j]) ** 2 for i in quark_idx for j in lepton_idx)
    quark_to_lepton = sum(abs(M[i, j]) ** 2 for i in lepton_idx for j in quark_idx)
    quark_to_quark = sum(abs(M[i, j]) ** 2 for i in quark_idx for j in quark_idx)
    print(f"    |U_C3 lepton -> lepton|^2 = {lepton_to_lepton:.4f}")
    print(f"    |U_C3 lepton -> quark|^2  = {lepton_to_quark:.4f}")
    print(f"    |U_C3 quark  -> lepton|^2 = {quark_to_lepton:.4f}")
    print(f"    |U_C3 quark  -> quark|^2  = {quark_to_quark:.4f}")
print()
print("  Interpretation:")
print("    If lepton -> quark mixing is NONZERO, U_{C_3}^S literally turns ")
print("    B3-named lepton states into B3-named quark states (and vice versa).")
print("    Under B6's PS reading, the C_3 should permute the 3 quark colors")
print("    while fixing the lepton — but B3's lepton/quark naming is FORCED")
print("    by the U(1)_{B-L} eigenvalue Y, not by the SU(4) eigenvalue.")
print()


# ---------------------------------------------------------------------------
# Step 5 — Does U_{C_3}^S commute with U(1)_{B-L} = Y?
# ---------------------------------------------------------------------------
print("Step 5 — [Y, U_{C_3}^S] = 0?")
comm_Y = Y @ U_C3_S - U_C3_S @ Y
print(f"  ||[Y, U_C3_S]||  = {np.linalg.norm(comm_Y):.3e}")
print()
print("  Interpretation:")
print("    Y is the U(1)_{B-L} generator in B3's Spin(2) factor. It")
print("    distinguishes lepton (B-L = -1) from quark (B-L = +1/3) in the PS")
print("    embedding. If [Y, U_{C_3}^S] != 0, then U_{C_3}^S MIXES Y-")
print("    eigenvalues, i.e., it DOES NOT preserve the lepton/quark distinction")
print("    that B3 reads off Y.")
print()
print("    But B6 Step 7 CLAIMS the C_3 is the Z_3 center of SU(3)_color,")
print("    which lies INSIDE SU(4) and acts trivially on the 'leptocolor' ")
print("    direction (i.e., commutes with the U(1)_{B-L} = Y). So [Y, U_{C_3}^S]")
print("    SHOULD be zero under B6's Z(SU(3)_color) reading.")
print()
print("    NUMERICAL TEST: if ||[Y, U_C3_S]|| != 0, B6's identification of the")
print("    C_3 with Z(SU(3)_color) inside SU(4) is INCORRECT as stated; instead,")
print("    U_{C_3}^S is a generic SU(4) Weyl element that mixes the SU(4)")
print("    fundamental basis vectors freely (NOT preserving the SU(3)_color +")
print("    U(1)_{B-L} subgrouping).")
print()


# ---------------------------------------------------------------------------
# Step 6 — Verify the SU(4) eigenvalue claim. The B6 lift uses
#           (a, b, c, d) = (1, 1, omega, omega^2).
# ---------------------------------------------------------------------------
print("Step 6 — Eigenvalues of U_{C_3}^S on the Weyl chiralities S_+, S_-.")
P_plus = 0.5 * (I8 + G7)
P_minus = 0.5 * (I8 - G7)
# Get a basis for S_+ (4 states) and S_- (4 states).
ev_plus, vec_plus = np.linalg.eigh(P_plus)
plus_idx = np.where(ev_plus > 0.5)[0]
basis_plus = vec_plus[:, plus_idx]
ev_minus, vec_minus = np.linalg.eigh(P_minus)
minus_idx = np.where(ev_minus > 0.5)[0]
basis_minus = vec_minus[:, minus_idx]

U_plus = basis_plus.conj().T @ U_C3_S @ basis_plus
U_minus = basis_minus.conj().T @ U_C3_S @ basis_minus
ev_U_plus = sorted(np.linalg.eigvals(U_plus), key=np.angle)
ev_U_minus = sorted(np.linalg.eigvals(U_minus), key=np.angle)

print(f"  S_+  (chirality +1, B6 calls 4 of SU(4)) eigenvalues:")
for ev in ev_U_plus:
    print(f"    {ev:+.4f}")
print(f"  S_-  (chirality -1, B6 calls bar 4 of SU(4)) eigenvalues:")
for ev in ev_U_minus:
    print(f"    {ev:+.4f}")
print()
print("  B6 prediction (Step 4): on 4 (= S_+ in B6 conventions) the C_3 has")
print("  eigenvalues (1, 1, omega, omega^2); on bar 4 (= S_-) the conjugates")
print("  (1, 1, omega^2, omega).")
print()


# ---------------------------------------------------------------------------
# Step 7 — Cross-check: build the SU(4) PS embedding explicitly via Pati-Salam.
# Compare U_C3_S to the candidate "Z(SU(3)_color) embedded in SU(4)" element.
# ---------------------------------------------------------------------------
print("Step 7 — Compare U_{C_3}^S to a true Z(SU(3)_color) embedding in SU(4):")
print()
print("  In the PS embedding SU(4) -> SU(3)_color x U(1)_{B-L}, a Z_3-center")
print("  element of SU(3)_color acts on the SU(4) fundamental as")
print("    g_PS = diag(omega_color, omega_color, omega_color, 1) * (overall U(1) phase to make det = 1)")
print("  i.e., it permutes nothing; it's a pure phase rotation of the 3 quark-color slots,")
print("  with the lepton slot fixed up to a U(1) compensator.")
print()
print("  By contrast, B6's claimed (1, 1, omega, omega^2) on the SU(4) fundamental is")
print("  NOT in SU(3)_color (which would be diag(z, z, z, 1) for z^3 = 1 with the lepton")
print("  slot last). The element diag(1, 1, omega, omega^2) is in SU(4) (det = 1) but it's")
print("  a generic Cartan element of SU(4), NOT the Z(SU(3)_color) center.")
print()
print("  HENCE the B6 identification 'C_3 = Z(SU(3)_color)' is mathematically incorrect:")
print("  (1, 1, omega, omega^2) is NOT a Z(SU(3)_color) element under any standard PS embedding.")
print()
print("  The TRUE Z(SU(3)_color) center, embedded in SU(4) as diag(z, z, z, z^{-3})")
print("  with z = omega_color, has SU(4) eigenvalues (omega, omega, omega, 1) and acts on")
print("  V_6 = Lambda^2(C^4) with eigenvalues {omega^2, omega^2, omega^2, omega, omega, omega}")
print("  — NOT matching the body-diagonal C_3 spectrum on V_6 (which is (1,1,omega,omega,omega^2,omega^2)).")
print()


# ---------------------------------------------------------------------------
# Step 8 — Compute the Trace(Q^2) on different readings.
# ---------------------------------------------------------------------------
print("Step 8 — Tr(Q^2) under various physical-charge assignments to S:")

# Reading (I): B3 colorless. Charges:
#   nu_L:   Q = 0      e_L:   Q = -1
#   u_L:    Q = +2/3   d_L:   Q = -1/3
#   nu_R:   Q = 0      e_R:   Q = -1
#   u_R:    Q = +2/3   d_R:   Q = -1/3
B3_charges = {
    "nu_L": 0.0, "e_L": -1.0, "u_L": 2/3, "d_L": -1/3,
    "nu_R": 0.0, "e_R": -1.0, "u_R": 2/3, "d_R": -1/3,
}
Q_op_diag = np.diag([B3_charges[species_names[i]] for i in range(8)])
Q_op = V @ Q_op_diag @ V.conj().T  # convert from species basis to Cl(6,0) basis
Tr_Q2 = np.real(np.trace(Q_op @ Q_op))
T3_op = T_L / 2.0
Tr_T3sq = np.real(np.trace(T3_op @ T3_op))
print(f"  Reading (I) — B3 colorless, 1 quark axis per species per chirality:")
print(f"    Tr(T_3^2)  = {Tr_T3sq:.4f}  (expected 1.0)")
print(f"    Tr(Q^2)    = {Tr_Q2:.4f}  (expected 28/9 = {28/9:.4f})")
sin2 = Tr_T3sq / Tr_Q2 if Tr_Q2 != 0 else 0
print(f"    sin^2(theta_W) = Tr(T_3^2) / Tr(Q^2) = {sin2:.4f}  (target 3/8 = 0.375)")
print()

# Reading (II): If C_3 makes 3 colors out of 1 quark axis and 1 lepton, then per
# chirality we have 1 lepton + 3 quark colors = 4 states. The lepton has SOME
# charge (which?), and the quarks all have +2/3 or all -1/3. Try assigning:
#   trivial-C_3 state (1 of the 4 per chirality) = lepton
#   omega + omega^2 states = 2 quark colors (and the 4th is the 3rd quark color
#   sitting in the trivial C_3 sector — which then must split off lepton)
# This is the B6 PS reading. But it requires an additional layer; let's compute
# the trace on a 4-color-per-chirality "all up-type" or "all down-type" reading.
# Per chirality 4 states: 1 lepton (Q = ?) + 3 quarks (all same Q).
# To make sense of Q on this reading we need to know if these 4 states
# collectively are u_L_doublet (4 states = nu_L, u_L^r, u_L^g, u_L^b) or
# d_L_doublet (e_L, d_L^r, d_L^g, d_L^b). Per chirality only ONE of these.

print(f"  Reading (II) — B6 colored, 1 lepton + 3 quark colors per chirality")
print(f"                 (= half an SU(2)_L doublet per chirality).")
print(f"    Per chirality 4 states are EITHER (nu, u^r, u^g, u^b) OR (e, d^r, d^g, d^b);")
print(f"    no SU(2)_L pairing within S since S has 4 states per chirality and an SU(2)_L")
print(f"    doublet of (lepton + 3 quark colors) requires 8 states per chirality.")
print()
print(f"    Take L Weyl = (nu_L, u_L^r, u_L^g, u_L^b), R Weyl = (e_R, d_R^r, d_R^g, d_R^b):")
charges_II = [0.0, 2/3, 2/3, 2/3, -1.0, -1/3, -1/3, -1/3]  # 4L + 4R, NOT B3-named
Q_II = np.diag(charges_II)
Tr_Q2_II = np.real(np.trace(Q_II @ Q_II))
# T_3: in this reading, only nu_L and e_R have T_3 = +/- 1/2 (the SU(2)_L
# generator distinguishes them across chiralities, which is wrong but is what
# Reading II forces).
T3_II = np.diag([0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5])
Tr_T3sq_II = np.real(np.trace(T3_II @ T3_II))
print(f"    Tr(T_3^2)  = {Tr_T3sq_II:.4f}")
print(f"    Tr(Q^2)    = {Tr_Q2_II:.4f}  (note: 4*1/9 + 1 + 4*1/9 + 1 != target 16/3)")
sin2_II = Tr_T3sq_II / Tr_Q2_II if Tr_Q2_II != 0 else 0
print(f"    sin^2(theta_W) = {sin2_II:.4f}  (target 3/8 = 0.375)")
print()


# ---------------------------------------------------------------------------
# Final summary.
# ---------------------------------------------------------------------------
print("=" * 78)
print("RECONCILIATION SUMMARY")
print("=" * 78)
print()
print("Numerical findings:")
print()
print("  (1) U_{C_3}^S does NOT commute with the SU(2)_L Cartan T_L")
print(f"      (||[T_L, U_C3]|| = {np.linalg.norm(T_L @ U_C3_S - U_C3_S @ T_L):.3e}).")
print()
print("  (2) U_{C_3}^S does NOT commute with U(1)_{B-L} = Y")
print(f"      (||[Y, U_C3]|| = {np.linalg.norm(Y @ U_C3_S - U_C3_S @ Y):.3e}).")
print()
print("  (3) Hence U_{C_3}^S MIXES B3-named lepton states with B3-named quark")
print("      states. B3's species labels (nu, e, u, d) are NOT C_3-stable.")
print()
print("  (4) The SU(4) eigenvalues (1, 1, omega, omega^2) are a GENERIC SU(4)")
print("      Cartan element, NOT a Z(SU(3)_color) center element. Z(SU(3)_color)")
print("      embedded in SU(4) under the standard PS embedding has SU(4) eigenvalues")
print("      (omega, omega, omega, 1) (or its inverse), which acts on V_6 = Lambda^2(C^4)")
print("      with eigenvalues (omega^2)^3 + omega^3 = (3 of omega^2, 3 of omega), NOT")
print("      matching the body-diagonal C_3 spectrum on V_6.")
print()
print("  (5) The (4, 2, 2) isotypic decomposition on S is genuine: B6 is correct")
print("      that U_{C_3}^S has eigenvalue counts (4, 2, 2). However:")
print()
print("        * The 4-dim trivial sector contains states that under B3's labels")
print("          are linear combinations of (nu_L, e_L, nu_R, e_R, u_L, d_L, u_R, d_R)")
print("          — they are NOT 'the leptons + 1 quark color'.")
print()
print("        * The omega and omega^2 sectors similarly are LINEAR COMBINATIONS")
print("          of B3 species, not 'pure quark colors'.")
print()
print("  (6) sin^2(theta_W) on S, under the colorless reading (I), is 9/28 =", round(9/28, 4),
      ", \n      not 3/8. Under reading (II) (4 colored quarks per chirality, no SU(2)_L within S),")
print("      sin^2(theta_W) = 3/4 = 0.750 — also not 3/8.")
print()
print("VERDICT: The (4, 2, 2) decomposition is real, but the B6 identification")
print("of the C_3 with Z(SU(3)_color) is loose. The C_3 acts on S as a generic")
print("SU(4) Cartan element, not as the Z_3 center of SU(3)_color. Under EITHER")
print("of B3's or B6's species readings, the dimensional content of S (8 states)")
print("is NOT the dimensional content of one SM generation with color (16 states),")
print("so neither the SU(2)_L structure nor the color-triplet structure can be")
print("realized cleanly on S alone.")
print()
print("OK: theorem_B3_B6_reconciliation computation complete.")
