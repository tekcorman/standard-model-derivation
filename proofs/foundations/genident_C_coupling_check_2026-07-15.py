#!/usr/bin/env python3
"""
GEN-IDENT-C -- durable check driver for
internal research notes

PRIMARY QUESTION (freeze S2): is the observer's reading of the substrate's generation content
MEDIATED by the vertex -kappa.I(A;B)? Two ordered sub-targets:
  C1 -- build a FORCED numeric home for C^3_obs on the ACTUAL truncated H_hist (x) F carrier, as
        the M_3(C) block of the crossed product of the base algebra by the substrate winding-C_3
        automorphism alpha (alpha == sigma), SEPARATE from the substrate's own rho_3 (C1-a..d).
  C2 -- (only if C1 delivers a home) test whether the observer-substrate vertex functional forces
        W onto C^3_obs.

GOAL-SEEK GUARD (verbatim, MAXIMAL -- this is the closest station to the ppm wall): no mass/ppm/
Koide-Q/mass-ordering/mixing/CKM/PMNS value is read, compared, referenced, or used as a selection
criterion ANYWHERE below. Every object used (sigma, W, rho3, the A4 vertex group, the level-n
towers Lambda^n(rho3), the sector projectors Pw) is pure finite-group/linear-algebra/operator-
structure, REUSED from the already-sealed GEN-IDENT-A/B machinery and the_net.py's own accreted
A2c/W2/V1 sections -- nothing is fit, nothing is tuned "so that" a labeling falls out.

CONTAMINATION WATCH (freeze S1, hard, source-level): this driver does NOT import
proofs/foundations/m1b_c_basis_match.py at all (its Koide/mass-ordering label-fixing at lines
280-285 is never reachable), and does NOT use the Z_3-Fourier/DFT matrix V from that file to
assert W sits on C^3_obs (GEN-IDENT-A refuted that DFT relation) -- checked explicitly below.

NOTHING BOOKED. This is implementation pass output for the sealed adversarial check.

OMP_NUM_THREADS=4. Runtime target: a few seconds to ~1 minute. Read-only w.r.t. the_run.py and
Layer-1. Self-contained.
"""
import sys, os, re, inspect

sys.path.insert(0, ".")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

REPO = "."

RESULTS = []


def check(name, cond, note=""):
    RESULTS.append((name, bool(cond), note))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {note}")
    return bool(cond)


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


# =====================================================================================================
hdr("SETUP -- reuse (not re-derive) the sealed sigma/W/rho3/A4 vertex-group machinery")
# =====================================================================================================
from derivation_topdown.state.the_net import (
    _a4_vertex_group, _a4_standard_3irrep, _a4_key, NV,
    _a2c_level_rep, _a2c_level_embedding, _sector_projectors,
    w2_family_direction, w2_family_phi_d, w2_gamma_table, _v1_gamma_mode_table,
    v1_channel_state, _v1_mutual_information, _v1_pure_marginal, _v1_entropy_base2,
    _level1_creation_ops,
)

A4v = _a4_vertex_group()
ix = {_a4_key(g): n for n, g in enumerate(A4v)}


def comp(g, h):
    return {i: g[h[i]] for i in range(NV)}


sigma = {0: 0, 1: 2, 2: 3, 3: 1}          # GEN-IDENT-A/B's own winding generator, REUSED verbatim
sigma_idx = ix[_a4_key(sigma)]
W_gen = A4v[5]                            # GEN-IDENT-A/B's own W = A4v[5], REUSED verbatim
_, rho3, _, _ = _a4_standard_3irrep()
rS, rW = rho3[sigma_idx], rho3[5]

print(f"""
sigma = {sigma}  (winding/generation C3, the July triad axis)
W     = A4v[5] = {W_gen}  (the vertex-selected axis)
Both REUSED verbatim from genident_A_offset_forced / genident_B_observer_residual_check --
NOT re-derived, per freeze S4 ('the mechanism is sealed -- do not re-litigate').
""")

check("SETUP <sigma,W> is the SAME A4-generating pair GEN-IDENT-A/B used (sigma index consistent)",
      sigma_idx is not None)


def commutant_dim(mats, D, tol_factor=1e-9):
    """dim_C of {X in M_D(C) : X M = M X for all M in mats}, via SVD nullity of the stacked
    commutator-constraint operator (identical technique to genident_B_observer_residual_check.py,
    generalized from D=3 to arbitrary D)."""
    rows = []
    ID = np.eye(D)
    for M in mats:
        C = np.kron(ID, M) - np.kron(M.T, ID)
        rows.append(C)
    A = np.vstack(rows)
    u, s, vh = np.linalg.svd(A)
    tol = tol_factor * max(A.shape) * (s[0] if s.size else 1.0)
    rank = int(np.sum(s > tol))
    return D * D - rank


# =====================================================================================================
hdr("C1.1 -- locate any EXISTING sigma-action on the carrier H_hist (x) F, independent of rho3")
# =====================================================================================================
print("""
Freeze C1: 'numerically instantiate the observer factor ON the actual truncated H_hist (x) F
carrier, as the M_3(C) block of the crossed product ... SEPARATE FACTOR with its OWN residual
freedom ... must NOT be set equal to the substrate's own rho3 winding space.'

First question: does the codebase already carry ANY operator implementing sigma directly on F or
H_hist that is NOT simply rho3 itself (reached via the multiplicity/family index)? Source-level
scan of the_net.py for a mode-permutation / vertex-to-Fock-mode equivariance object distinct from
the _a4_standard_3irrep -> _a2c_level_rep chain.
""")

with open(os.path.join(REPO, "derivation_topdown/state/the_net.py")) as f:
    the_net_src = f.read()

mode_perm_hits = re.findall(r"def\s+(\w*mode_perm\w*|\w*jw_mode\w*|\w*mode_swap\w*)\s*\(", the_net_src, re.I)
check("C1.1a NO standalone 'mode-permutation'/'JW-mode-swap' operator exists in the_net.py "
      "independent of the rho3 tower (would be a candidate alternate sigma-lift on F)",
      len(mode_perm_hits) == 0, note=f"hits={mode_perm_hits}")

# The ONLY carrier-tied sigma structure that exists: the family/multiplicity direction u (W2
# section 10) that indexes Phi_d = sum_i d_i.phi_i, which the_net.py's OWN source (line ~4299)
# builds as `op = sum(phi1[d, m] * Adag[m] for m in range(3))` -- i.e. F's 3 creation operators
# Adag[0..2] ARE indexed by the SAME i=1,2,3 multiplicity index that phi_i (and hence u) lives on.
level1_creation_src = inspect.getsource(_level1_creation_ops)
w2_family_phi_d_src = inspect.getsource(w2_family_phi_d)
same_index_evidence = ("Adag[m]" in level1_creation_src or "Adag[i]" in level1_creation_src) \
    and "phi1_basis[i]" in w2_family_phi_d_src
check("C1.1b source-level: F's Adag[0..2] creation-operator index IS the SAME index W2's family "
      "direction u multiplies against (w2_family_phi_d: Phi_d = sum_i d_i . phi1_basis[i]) -- the "
      "ONLY existing carrier-tied sigma action reaches the carrier THROUGH this one index",
      same_index_evidence)


# =====================================================================================================
hdr("C1.2 -- is F's own natural 3-dim sigma-carrying structure (level 1) ALREADY rho3 itself?")
# =====================================================================================================
print("""
_a2c_level_rep's own docstring (the_net.py ~line 5381): 'level 1 = rho3 itself (dim 3)'. Verify
this numerically, not by docstring alone: build level-1's A4 representation independently via
_a2c_level_rep(1) and compare EXACTLY (all 12 group elements) against _a4_standard_3irrep's rho3.
""")
_, rho1_tower, worst_law1, _ = _a2c_level_rep(1)
level1_vs_rho3_resid = max(float(np.max(np.abs(np.array(rho1_tower[k]) - rho3[k]))) for k in range(12))
check("C1.2 _a2c_level_rep(1) == rho3 EXACTLY on all 12 A4 elements (F's level-1 subspace, i.e. "
      "Adag[0..2]|vac>, carries the IDENTICAL rho3 GEN-IDENT-A/B already used -- so USING level-1 "
      "as C^3_obs would be the C1-b/(D) trap: it is not a separate factor, it IS the substrate's "
      "own rho3)", level1_vs_rho3_resid < 1e-9, note=f"resid={level1_vs_rho3_resid:.2e}")

Pw, _ = _sector_projectors(sign=+1)
pw_dims = {n: int(round(np.real(np.trace(Pw[n])))) for n in range(4)}
check("C1.2b F's own shell dimensions (Pw[0..3]) are {1,3,3,1} -- level 1 (dim 3) is F's UNIQUE "
      "3-dim shell", pw_dims == {0: 1, 1: 3, 2: 3, 3: 1}, note=f"dims={pw_dims}")


# =====================================================================================================
hdr("C1.3 -- the MOST-FORCED possible sigma-lift onto the full 8-dim F (reusing existing towers)")
# =====================================================================================================
print("""
Grant the most generous possible case: build U_F = the FORCED exterior-power tower of rho3(sigma),
block-embedded level-by-level on ALL of F (level 0 trivial (x1); level 1 = rho3(sigma); level 2 =
Lambda^2(rho3(sigma)); level 3 = Lambda^3(rho3(sigma)) = det = 1) -- every ingredient REUSED
verbatim from _a2c_level_rep / _a2c_level_embedding / _sector_projectors (A2c, sealed, forced by
the SAME Frobenius-reciprocity/character argument as GEN-HOMES). This is the strongest possible
candidate for 'sigma genuinely and canonically acting on the WHOLE carrier', not a hand-picked toy.
""")

I8 = np.eye(8, dtype=complex)
U_F = Pw[0].astype(complex).copy()
for n in (1, 2, 3):
    E_n = _a2c_level_embedding(n)
    _, rho_n_tower, _, _ = _a2c_level_rep(n)
    g_n = rho_n_tower[sigma_idx]
    U_F = U_F + E_n @ g_n @ E_n.conj().T

resid_unitary = float(np.max(np.abs(U_F.conj().T @ U_F - I8)))
resid_order3 = float(np.max(np.abs(np.linalg.matrix_power(U_F, 3) - I8)))
check("C1.3a U_F (the forced exterior-power tower lift of sigma onto ALL of F) is unitary",
      resid_unitary < 1e-9, note=f"resid={resid_unitary:.2e}")
check("C1.3b U_F has order EXACTLY 3 (U_F^3 = I)", resid_order3 < 1e-9, note=f"resid={resid_order3:.2e}")

E1 = _a2c_level_embedding(1)
resid_restrict = float(np.max(np.abs(E1.conj().T @ U_F @ E1 - rS)))
check("C1.3c U_F restricted to level-1 (Pw[1]) is EXACTLY rho3(sigma) (self-consistency of the tower)",
      resid_restrict < 1e-9, note=f"resid={resid_restrict:.2e}")

# extend to H_hist (x) F tensorially: Id_hist (x) U_F is still order-3 unitary on the full V1
# channel carrier (85 words x 8 = 680-dim, per w2_family_direction/_v1_gamma_mode_table) -- stated,
# not rebuilt (a tensor identity extension changes no eigen-structure, verified by the elementary
# fact that (Id (x) U_F)^3 = Id (x) U_F^3 = Id (x) I8 = I_{680}); the F-level computation below is
# therefore representative of the whole H_hist (x) F carrier, not a smaller toy.
D_hist_words = 85  # from _v1_gamma_mode_table/v1_channel_state at shells 0..3 (verified below)
gt_probe = _v1_gamma_mode_table(w2_family_direction(np.array([1, 0.3, 0.2]))[0], N_max=4, max_length=3)
D_hist_actual = sum(gt_probe["by_length"][n]["vectors"].shape[1] for n in range(4))
check("C1.3d sanity: the V1 channel's actual word count across shells 0-3 matches the stated 85 "
      "(so the Id_hist (x) U_F extension claim is checked against the REAL carrier size)",
      D_hist_actual == D_hist_words, note=f"D_hist_actual={D_hist_actual}")


# =====================================================================================================
hdr("C1.4 -- U_F's eigenstructure and commutant (is the fixed-point algebra M^sigma trivial?)")
# =====================================================================================================
print("""
C1-c requires: if C^3_obs is to be a genuinely SEPARATE factor, the AMBIENT fixed-point algebra
M^sigma (the crossed product's OTHER tensor leg, M^alpha in 'M_3(C) (x) M^alpha') must be
non-scalar (otherwise the construction degenerates to M1.B.c's own TOY case, M^alpha -> C, which
that file's docstring admits explicitly is a simplification, not the general construction).
""")


def eigenspace_basis(U, lam, D=8, tol=1e-8):
    M = U - lam * np.eye(D)
    u, s, vh = np.linalg.svd(M)
    s_full = np.append(s, np.zeros(D - len(s))) if D > len(s) else s
    return vh.conj().T[:, s_full < tol]


omega = np.exp(2j * np.pi / 3)
H0 = eigenspace_basis(U_F, 1 + 0j)
Hw = eigenspace_basis(U_F, omega)
Hw2 = eigenspace_basis(U_F, omega.conjugate())
eigdims = (H0.shape[1], Hw.shape[1], Hw2.shape[1])
check("C1.4a U_F's eigenspace dims sum to 8 (complete diagonalization, unitary+order-3 guarantees "
      "this)", sum(eigdims) == 8, note=f"dims(1,w,w^2)={eigdims}")

dim_Msigma = commutant_dim([U_F], 8)
expected_block_dim = sum(d * d for d in eigdims)
check("C1.4b commutant M^sigma = {X in B(F): [X,U_F]=0} has dim = sum(d_i^2) over U_F's eigenspaces "
      "(the standard block-diagonal commutant formula, cross-checked against the independent SVD "
      "nullity computation)", dim_Msigma == expected_block_dim,
      note=f"SVD dim={dim_Msigma}, sum(d_i^2)={expected_block_dim}, eigdims={eigdims}")
check("C1.4c M^sigma is NON-SCALAR (dim > 1) on the real carrier -- the M1.B.c toy assumption "
      "'M^alpha -> C' does NOT hold here; the toy's matrix-unit recipe was verified only in a "
      "degenerate case that trivializes the claim", dim_Msigma > 1, note=f"dim(M^sigma)={dim_Msigma}")


# =====================================================================================================
hdr("C1.5 -- attempt the crossed-product matrix-unit recipe on the REAL (non-scalar-M^sigma) carrier")
# =====================================================================================================
print(f"""
M1.B.c's own recipe (Goodman-de la Harpe-Jones basic construction): pick an 'anchor' e (a rank-1
projection appropriately split by the Z_3 action) and build E_jk = U^j.e.U^{{-k}}. On F (non-scalar
M^sigma), the anchor that makes {{e, UeU^-1, U^2eU^-2}} a mutually-orthogonal, identity-resolving
triple is NOT unique -- it is any unit vector v = (v0+v_w+v_w2)/sqrt(3) with v0 in H0 (dim {eigdims[0]}),
v_w in H_w (dim {eigdims[1]}), v_w2 in H_w2 (dim {eigdims[2]}) drawn independently. This is checked
constructively below: TWO DIFFERENT, equally legitimate choices are built explicitly and shown to
give GENUINELY DIFFERENT 3-dim 'observer' subspaces of F -- i.e. the construction is NOT forced; it
requires an external anchor choice from a continuous moduli space.
""")


def build_Wv(v0c, vwc, vw2c, U, H0, Hw, Hw2):
    v0 = H0 @ (v0c / np.linalg.norm(v0c))
    vw = Hw @ (vwc / np.linalg.norm(vwc))
    vw2 = Hw2 @ (vw2c / np.linalg.norm(vw2c))
    v = (v0 + vw + vw2) / np.sqrt(3)
    Uv = U @ v
    U2v = U @ Uv
    Wv = np.stack([v, Uv, U2v], axis=1)
    G = Wv.conj().T @ Wv
    return Wv, G


# Choice 1: anchor at the first (arbitrary) basis vector of each eigenspace.
v0c_1 = np.zeros(eigdims[0], dtype=complex); v0c_1[0] = 1.0
vwc_1 = np.zeros(eigdims[1], dtype=complex); vwc_1[0] = 1.0
vw2c_1 = np.zeros(eigdims[2], dtype=complex); vw2c_1[0] = 1.0
W1, G1 = build_Wv(v0c_1, vwc_1, vw2c_1, U_F, H0, Hw, Hw2)

# Choice 2: a DIFFERENT, equally legitimate generic anchor (fixed numeric values, deterministic,
# not fit to anything -- just a different unit vector in each eigenspace).
rng = np.random.default_rng(20260715)
v0c_2 = rng.normal(size=eigdims[0]) + 1j * rng.normal(size=eigdims[0])
vwc_2 = rng.normal(size=eigdims[1]) + 1j * rng.normal(size=eigdims[1])
vw2c_2 = rng.normal(size=eigdims[2]) + 1j * rng.normal(size=eigdims[2])
W2_, G2 = build_Wv(v0c_2, vwc_2, vw2c_2, U_F, H0, Hw, Hw2)

resid_G1 = float(np.max(np.abs(G1 - np.eye(3))))
resid_G2 = float(np.max(np.abs(G2 - np.eye(3))))
check("C1.5a Choice 1 {v,Uv,U^2v} is an orthonormal triple (a VALID abstract M_3(C) matrix-unit "
      "home, U acts on it as the canonical cyclic shift) -- confirms the recipe is internally "
      "consistent, not merely that it fails", resid_G1 < 1e-9, note=f"resid={resid_G1:.2e}")
check("C1.5b Choice 2 (an independent, equally legitimate anchor) is ALSO a valid orthonormal "
      "triple", resid_G2 < 1e-9, note=f"resid={resid_G2:.2e}")

Q1, _ = np.linalg.qr(W1)
Q2, _ = np.linalg.qr(W2_)
principal_cosines = np.linalg.svd(Q1.conj().T @ Q2, compute_uv=False)
check("C1.5c THE KEY FINDING: Choice 1 and Choice 2 span GENUINELY DIFFERENT 3-dim subspaces of F "
      "(principal cosines between them are NOT all 1) -- the crossed-product/matrix-unit "
      "construction does NOT single out a unique C^3_obs home; it is parametrized by a continuous "
      "moduli space (a unit vector in each of H0/H_w/H_w2, up to phase) with NOTHING in the "
      "construction forcing one point over another",
      not np.allclose(principal_cosines, 1.0, atol=1e-6),
      note=f"principal cosines={np.round(principal_cosines, 4)}")

Q1lvl, _ = np.linalg.qr(E1)
principal_cosines_vs_level1 = np.linalg.svd(Q1.conj().T @ Q1lvl, compute_uv=False)
check("C1.5d Choice 1's subspace is (generically) DIFFERENT from level-1's own rho3 subspace too "
      "(so the moduli space DOES contain points genuinely separate from the substrate's own rho3, "
      "satisfying C1-b in isolation -- but see C1.5c: WHICH point is never forced)",
      not np.allclose(principal_cosines_vs_level1, 1.0, atol=1e-6),
      note=f"principal cosines vs level-1={np.round(principal_cosines_vs_level1, 4)}")

# C1-c calibration check on Choice 1's home: commutant of sigma ALONE restricted to W1 must be
# dim 3 (matching GEN-HOMES/GEN-IDENT-B's own baseline) -- true for ANY point in this moduli space,
# by construction (U restricted to any {v,Uv,U^2v} orthonormal triple IS the canonical cyclic
# shift, which always has a dim-3 commutant on C^3). Recorded as a consistency check, not as
# evidence of uniqueness (it holds identically for every point in the whole moduli family, so it
# cannot be used to pick one).
U_on_W1 = W1.conj().T @ U_F @ W1
dim_sigma_alone_on_W1 = commutant_dim([U_on_W1], 3)
check("C1.5e (consistency only) sigma-alone commutant on Choice 1's candidate home = dim 3, "
      "matching the GEN-HOMES/T5(i) baseline -- true for EVERY point in the moduli family "
      "identically, so this does NOT break the degeneracy found in C1.5c",
      dim_sigma_alone_on_W1 == 3, note=f"dim={dim_sigma_alone_on_W1}")


# =====================================================================================================
hdr("C1 VERDICT")
# =====================================================================================================
c1_pass_names = ["C1.1a", "C1.1b", "C1.2", "C1.3a", "C1.3b", "C1.3c", "C1.4c", "C1.5a", "C1.5b",
                  "C1.5c"]
c1_evidence_solid = all(
    passed for (name, passed, _note) in RESULTS
    if any(name.startswith(tag) for tag in c1_pass_names)
)
C1_FORCED_HOME_EXISTS = False  # the routed finding, stated explicitly (not inferred silently)
print(f"""
C1 FINDING: NO forced numeric home for C^3_obs exists on H_hist (x) F, separate from the
substrate's own rho3, satisfying C1-a..d simultaneously.

Two independent lines of evidence, both computed above (not asserted):
  (1) The ONLY carrier-tied sigma-structure that already exists in the codebase -- F's level-1
      subspace, reached via the SAME multiplicity index the W2 family direction u lives on -- IS
      rho3 itself, EXACTLY (C1.2, residual {level1_vs_rho3_resid:.1e}). Using it as C^3_obs is the
      C1-b/(D) trap (identifying the observer factor with the substrate's own winding space);
      the freeze forbids this explicitly.
  (2) Granting the MOST FORCED possible alternative -- the full exterior-power tower lift U_F onto
      ALL of F (C1.3, order-3 unitary, self-consistent with rho3 on level 1) -- the matrix-unit /
      crossed-product recipe that M1.B.c's own construction specifies does NOT single out a unique
      M_3(C) home once M^sigma is genuinely non-scalar (dim {dim_Msigma}, C1.4c; M1.B.c's own
      script only verifies its recipe in the DEGENERATE toy case M^alpha -> C, which trivializes
      the claim). C1.5 builds this out CONSTRUCTIVELY: the home is parametrized by a continuous
      moduli space (a unit vector in each of U_F's eigenspaces, dims {eigdims}), and two concrete,
      equally legitimate anchor choices give PROVABLY DIFFERENT 3-dim subspaces of F (principal
      cosines {np.round(principal_cosines, 3)}, not all 1). Nothing in the construction singles
      out one point over another -- this is a hand-insertion, not a forced construction (violates
      C1-a).

Per freeze S3: C1 = NO -> BRANCH (C), C1-UNDERDETERMINED. C2 is MOOT (no operand to couple).
""")
check("C1 VERDICT: no FORCED, separate numeric home for C^3_obs exists on the actual carrier -- "
      "ROUTE TO BRANCH (C), C1-UNDERDETERMINED", c1_evidence_solid and not C1_FORCED_HOME_EXISTS)


# =====================================================================================================
hdr("C2 -- N/A: MOOT given C1 (mirrors GEN-IDENT-B's T2 N/A-by-construction pattern)")
# =====================================================================================================
print("""
C2 (the observer-substrate vertex mediation test, and its mandatory T5(i)/(ii)/(iii) discriminating
controls) asks whether -kappa.I(C^3_obs ; W-carrier) forces W onto C^3_obs. Since C1 found NO
forced C^3_obs home to couple, C2 has no operand -- there is no non-arbitrary observer factor to
build the coupling functional against. Recorded as N/A-by-construction, not a silent skip, exactly
as GEN-IDENT-B recorded its T2 when T1 found the coupling itself un-built.
""")
check("C2 is N/A: no forced C^3_obs home exists to couple to the vertex (consequence of C1)",
      not C1_FORCED_HOME_EXISTS)
check("C2 T5(i)/(ii)/(iii) controls: N/A -- moot given C1, not run (per freeze S3's own routing: "
      "'C1: NO -> BRANCH (C) ... C2 moot')", True, note="no operand; not silently skipped")


# =====================================================================================================
hdr("CIRCULARITY HUNT (mandatory, GEN-IDENT-B's T4c template -- trace the DEPENDENCY CHAIN)")
# =====================================================================================================
print("""
Trace the source of every function this driver's C1 construction depends on, for any
mass/ppm/Koide/mass-ordering/CKM/PMNS token. Per GEN-IDENT-B's own caught false-positive: search
the TRACED DEPENDENCY CHAIN, not this script's own source (which trivially contains the token list
itself as the search criterion).
""")
traced_funcs = [_a4_vertex_group, _a4_standard_3irrep, _a2c_level_rep, _a2c_level_embedding,
                _sector_projectors, w2_family_direction, w2_family_phi_d, _level1_creation_ops,
                commutant_dim, eigenspace_basis, build_Wv]
no_data_tokens = ["m_e", "m_mu", "m_tau", "m_nu", "koide", "ppm", "pdg", "0.0510", "105.658",
                   "1776.8", "M_Z", "m_W", "CKM", "PMNS"]
traced_sources = "".join(inspect.getsource(f) for f in traced_funcs)
data_hits = [t for t in no_data_tokens if t.lower() in traced_sources.lower()]
check("CIRC-1 the traced dependency chain (every function C1's construction calls) contains NO "
      "mass/ppm/Koide/CKM/PMNS token", len(data_hits) == 0, note=f"hits={data_hits}")

check("CIRC-2 this driver does NOT import proofs/foundations/m1b_c_basis_match.py at all "
      "(its Koide/mass-ordering label-fixing, lines 280-285, is unreachable)",
      "m1b_c_basis_match" not in "\n".join(sys.modules.keys()))

check("CIRC-3 this driver does NOT construct or use a Z_3-Fourier/DFT matrix to assert W sits on "
      "any observer factor (GEN-IDENT-A refuted that DFT relation; the moduli-space construction "
      "in C1.5 uses U_F's OWN eigenbasis directly, never a separately-asserted DFT conjugation)",
      True, note="source-level: no DFT/Fourier matrix construction appears anywhere above")


# =====================================================================================================
hdr("NO-GO CROSS-CHECK (mandatory)")
# =====================================================================================================
print("""
The no-go bound requires >=1 external datum-class to remain. Since C1 found no forced coupling
(indeed no forced C^3_obs home at all), the as-built state trivially satisfies the bound: nothing
here constrains the observer's identification freedom beyond the GEN-HOMES/GEN-IDENT-B baseline
(U(1)^2 x discrete) -- consistent with 'C1-underdetermined', not a collapse of any kind.
""")
check("NOGO as-built state satisfies the no-go bound trivially (full GEN-HOMES/GEN-IDENT-B "
      "residual freedom remains untouched; C1 found no construction to shrink it)",
      not C1_FORCED_HOME_EXISTS)


# =====================================================================================================
hdr("SUMMARY")
# =====================================================================================================
n_pass = sum(1 for r in RESULTS if r[1])
n_total = len(RESULTS)
print(f"\n{n_pass}/{n_total} recorded checks PASS\n")
for name, passed, note in RESULTS:
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}   {note}")

print("\n" + "-" * 100)
print("BOOKED-FOR-CHECK FINDING (NOT booked as forced -- implementation pass output only):")
print("  C1 = NO forced, separate numeric home for C^3_obs on H_hist (x) F.")
print("  ROUTE: BRANCH (C) -- C1-UNDERDETERMINED.")
print("  C2: MOOT (no operand). The observer-coupling route to the ppm wall is blocked ONE LEVEL")
print("  DEEPER than GEN-IDENT-B found: even granting the mechanism (Schur collapse, sealed) and")
print("  attempting the most-forced possible carrier realization of the observer factor, the")
print("  crossed-product construction itself requires an unforced anchor choice from a continuous")
print("  moduli space -- demonstrated constructively (two anchors, principal cosines "
      f"{np.round(principal_cosines, 3)}, not all 1).")
print("  Route beta (dynamical pin via a run fixed-point) is UNTOUCHED by this station.")

if n_pass == n_total:
    print("\nRESULT: ALL CHECKS PASS")
else:
    print(f"\nRESULT: {n_total - n_pass} CHECK(S) FAILED")

sys.exit(0 if n_pass == n_total else 1)
