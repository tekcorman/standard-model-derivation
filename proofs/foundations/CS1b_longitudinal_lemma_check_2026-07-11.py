#!/usr/bin/env python3
"""
proofs/foundations/CS1b_longitudinal_lemma_check_2026-07-11.py

CS-1b -- THEOREM-HUNT: proof (and machine-check) of CS-1's own S-3c/Hazard-#2 finding
(internal research notes): for external momentum p purely along
one axis mu, the Kubo bubble's LONGITUDINAL component Pi_mu,mu(p) (u = alpha_1) is found to be
EXACTLY independent of p over the tested range -- reported there as "not proved from first
principles" (Hazard #2). This file machine-checks the closed derivation written up in
docs/theorems/theorem_cs1_longitudinal_p_independence.md. It does NOT edit CS1_finite_k_propagator
_2026-07-11.py (read-only precedent) and does not edit any engine file.

THE MECHANISM (see the theorem note for the full derivation; summarized here for the reader):
  1. srs.hashimoto(k)'s own source is a per-target-dart DIAGONAL reweighting of a k-independent
     mask: B(k) = D(k) M. For p along a single axis mu, D(p) is EXACTLY a rank-2 diagonal update
     of the identity (D(p) - I has rank 2), supported on exactly the 2 darts of the mu-cotree
     edge -- the SAME 2 darts the vertex V_mu(k) = 2*pi*i*diag(v^mu)*B(k) is itself supported on
     (V_mu has rank <= 2, per CS-1's own S-0/S-1 disclosure). This coincidence of supports is the
     structural fact a GENERIC (dense-homology) toy operator does not share (CS-1's own S-1c
     control already showed this numerically; here it is shown to be the reason).
  2. Consequence A (checked at L-2): for the TRANSVERSE vertex V_nu, nu != mu, D(p) acts as the
     EXACT IDENTITY on V_nu(k) (their supports are disjoint dart sets) -- so V_nu(k+p) = V_nu(k)
     EXACTLY, for ANY p along axis mu. The transverse vertex is completely p-shift-INSENSITIVE;
     all of the transverse channel's p-dependence must come from G(k+p) alone, with no
     vertex-side cancellation available -- this is exactly why the argument below does NOT also
     kill the transverse channel (the load-bearing scope check).
  3. Consequence B (checked at L-3/L-4): for the LONGITUDINAL vertex V_mu, the SAME rank-2
     coincidence lets the bubble integrand collapse via one Sherman-Morrison/Woodbury step into
     an EXACT 2x2-matrix trace, Tr[V_mu(kmid) G(k) V_mu(kmid) G(k+p)] = -4*pi^2*Tr[X R(k) X
     R(k+p)], where R(k):=F0^T G(k) E (2x2, F0/E the constant mu-cotree-dart mask/basis vectors)
     and X = diag(x1,x2) with x1*x2 EXACTLY = -1 (independent of both k and p) -- this is a pure
     rank-2 linear-algebra identity, pointwise in k, no k-average needed yet.
  4. Consequence C (checked at L-5): R(k) itself obeys a SECOND Woodbury/Mobius identity in k_mu
     alone (same rank-2 mechanism, now treating k_mu as the "shift" off a k_mu=0 reference):
     R(k) = R0(k_perp) * [I + C(k_mu) R0(k_perp)]^{-1}, C(k_mu) diagonal in e^{2*pi*i*k_mu}.
  5. THE LEMMA (checked at L-6): for a 2x2 Mobius family R(z) = R0(I+C(z)R0)^{-1} of THIS exact
     form, the k_mu-integral of Tr[X R(z) X R(zt)] (t := e^{2*pi*i*p0}) is EXACTLY independent of
     t, for ANY 2x2 matrix R0 and any u -- a residue-calculus fact (3 poles survive inside the
     unit disk: z=0 with residue EXACTLY 1/u^2 always, plus a t-independent/t-reciprocal pair
     whose t-dependence cancels in the sum -- verified here to high numerical precision for a
     generic random R0, reproducing the theorem note's own exact symbolic (sympy) residue
     computation for representative concrete instances).
  6. L-7 reconciles the whole chain against CS-1's OWN reported brute-force Pi_00(p) numbers.

HARD RULES (restated, binding): CS1_finite_k_propagator_2026-07-11.py is NOT edited (read-only
citation only); no engine/proofs/verify.py/lock file touched; this is the ONE new file; no
goal-seeking (every number below is computed, not assumed); numbers only from running code.
"""
import os
import sys
import time

import numpy as np

T0 = time.time()
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402 -- the engine, unmodified

np.set_printoptions(precision=6, suppress=True, linewidth=120)
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    cond = bool(cond)
    ok_all = ok_all and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return cond


def banner(t):
    print("=" * 100)
    print(f" {t}")
    print("=" * 100)


def elapsed():
    return time.time() - T0


ALPHA1 = (2.0 / 3.0) ** 8


def main():
    banner("CS-1b -- PROOF/MACHINE-CHECK OF THE LONGITUDINAL p-INDEPENDENCE (CS-1 S-3c / Hazard #2)")
    print("Target claim (internal research notes, S-3c + Hazard #2): "
          "for p||axis mu, Pi_mu,mu(p;u=alpha1) is EXACTLY p-independent. Proof note: "
          "docs/theorems/theorem_cs1_longitudinal_p_independence.md. Precedent (read-only): "
          "proofs/foundations/CS1_finite_k_propagator_2026-07-11.py.")

    # =======================================================================================
    banner("L-0  REBUILD THE OBJECT (same construction as CS1_finite_k_propagator, regression-checked)")
    # =======================================================================================
    EDGES = srs.EDGES
    DARTS = []
    for i, j, v in EDGES:
        DARTS += [(i, j, np.array(v, float)), (j, i, -np.array(v, float))]
    ND = len(DARTS)
    V_HOM = np.array([d[2] for d in DARTS])
    MASK = np.zeros((ND, ND), dtype=bool)
    for b, (tb, hb, vb) in enumerate(DARTS):
        for a, (ta, ha, va) in enumerate(DARTS):
            if ha == tb and not (hb == ta and np.array_equal(vb, -va)):
                MASK[b, a] = True
    print(f"NE={len(EDGES)}  NV={srs.NV}  ND={ND} darts")

    def B_batch(K):
        K = np.asarray(K, float)
        PH = np.exp(2j * np.pi * (K @ V_HOM.T))
        return MASK[None, :, :] * PH[:, :, None]

    def V_batch(K, mu):
        return (2j * np.pi * V_HOM[:, mu])[None, :, None] * B_batch(K)

    rng0 = np.random.default_rng(0)
    ktest = rng0.random((4, 3))
    worst0 = max(np.max(np.abs(B_batch(ktest[i:i + 1])[0] - srs.hashimoto(ktest[i]))) for i in range(4))
    check("L-0 B_batch(k) == srs.hashimoto(k) exactly at 4 random k", worst0 < 1e-13, f"{worst0:.1e}")

    # per-axis mu-cotree dart indices d+, d- (v_b = +e_mu, -e_mu exactly)
    DPLUS = {}
    DMINUS = {}
    for mu in range(3):
        e_mu = np.zeros(3); e_mu[mu] = 1.0
        DPLUS[mu] = [b for b in range(ND) if np.allclose(V_HOM[b], e_mu)][0]
        DMINUS[mu] = [b for b in range(ND) if np.allclose(V_HOM[b], -e_mu)][0]
    print(f"  mu-cotree dart pairs (d+, d-): {[(DPLUS[m], DMINUS[m]) for m in range(3)]}")

    def grid(N):
        g1 = np.arange(N) / N
        return np.stack(np.meshgrid(g1, g1, g1, indexing="ij"), axis=-1).reshape(-1, 3)

    # =======================================================================================
    banner("L-1  IDENTITY B(k+p) = D(p) B(k)  (exact, pointwise in k -- the master fact)")
    # =======================================================================================
    rng1 = np.random.default_rng(1)
    k0 = rng1.random(3)
    p_rand = rng1.random(3) * 0.8
    Dp_rand = np.exp(2j * np.pi * (V_HOM @ p_rand))
    dev_L1 = np.max(np.abs(B_batch((k0 + p_rand)[None, :])[0] - Dp_rand[:, None] * B_batch(k0[None, :])[0]))
    check("L-1 B(k+p) == D(p) B(k) exactly (D(p):=diag(exp(2*pi*i*p.v_b)), generic random p)",
          dev_L1 < 1e-12, f"dev={dev_L1:.2e}")

    # =======================================================================================
    banner("L-2  VERTEX SHIFT LAWS -- longitudinal V_mu(k+p)=D(p)V_mu(k); TRANSVERSE V_nu(k+p)=V_nu(k) exactly")
    print(" This is the scope-defining fact: D(p) (rank-2, supported on the mu-cotree darts) acts as")
    print(" the IDENTITY on any matrix whose nonzero rows lie OUTSIDE that support -- and V_nu (nu!=mu)")
    print(" is supported EXACTLY on the (disjoint) nu-cotree darts, per srs.EDGES's own structure.")
    # =======================================================================================
    p0val = 0.31
    worst_long = 0.0
    worst_trans = 0.0
    for axis_p in range(3):
        p = np.zeros(3); p[axis_p] = p0val
        Dp = np.exp(2j * np.pi * (V_HOM @ p))
        Vmu_shift = V_batch((k0 + p)[None, :], axis_p)[0]
        Vmu_pred = Dp[:, None] * V_batch(k0[None, :], axis_p)[0]
        worst_long = max(worst_long, np.max(np.abs(Vmu_shift - Vmu_pred)))
        for nu in range(3):
            if nu == axis_p:
                continue
            Vnu_shift = V_batch((k0 + p)[None, :], nu)[0]
            Vnu_unshift = V_batch(k0[None, :], nu)[0]
            worst_trans = max(worst_trans, np.max(np.abs(Vnu_shift - Vnu_unshift)))
    check("L-2a LONGITUDINAL: V_mu(k+p) == D(p) V_mu(k) exactly (all 3 axes)", worst_long < 1e-12,
          f"worst dev={worst_long:.2e}")
    check("L-2b TRANSVERSE:   V_nu(k+p) == V_nu(k) EXACTLY for nu != axis(p) (all 6 combinations) "
          "-- the transverse vertex is p-shift-INSENSITIVE; this is WHY the cancellation below "
          "cannot also apply to the transverse channel (load-bearing scope check)",
          worst_trans < 1e-12, f"worst dev={worst_trans:.2e}")

    # =======================================================================================
    banner("L-3  WOODBURY REDUCTION -- Tr[V_mu(kmid) G(k) V_mu(kmid) G(k+p)] == -4pi^2 Tr[X R(k) X R(k+p)]")
    print(" R(k) := F0^T G(k) E  (2x2; F0, E built from the constant mu-cotree dart mask/basis).")
    print(" X := diag(lambda*e^{i*pi*p0}, -lambda^{-1}*e^{-i*pi*p0}), lambda=e^{2*pi*i*k_mu}.")
    print(" Checked POINTWISE at one random k (no k-average) -- a pure rank-2 linear-algebra identity.")
    # =======================================================================================
    u = ALPHA1
    I_ND = np.eye(ND, dtype=complex)

    def G_of(k):
        return np.linalg.inv(I_ND - u * B_batch(k[None, :])[0])

    def _F0E(mu):
        dp, dm = DPLUS[mu], DMINUS[mu]
        F0 = np.zeros((ND, 2))
        F0[:, 0] = MASK[dp, :].astype(float)
        F0[:, 1] = MASK[dm, :].astype(float)
        E = np.zeros((ND, 2))
        E[dp, 0] = 1.0
        E[dm, 1] = 1.0
        return F0, E

    def R_of(k, mu):
        """R(k) := F0^T G(k) E, the 2x2 reduced propagator at the mu-cotree dart pair."""
        F0, E = _F0E(mu)
        return F0.T @ G_of(k) @ E

    worst_woodbury = 0.0
    worst_4term = 0.0
    for axis_p in range(3):
        p = np.zeros(3); p[axis_p] = p0val
        kmid = k0 + 0.5 * p
        Gk = G_of(k0)
        Gkp = G_of(k0 + p)
        Vmu = V_batch(kmid[None, :], axis_p)[0]
        brute = np.trace(Vmu @ Gk @ Vmu @ Gkp)

        Rk = R_of(k0, axis_p)
        Rkp = R_of(k0 + p, axis_p)
        lam = np.exp(2j * np.pi * k0[axis_p])
        # X = diag(x1,x2), x1 := lambda*e^{i*pi*p0}, x2 := -lambda^{-1}*e^{-i*pi*p0} (theorem note L-3)
        x1 = lam * np.exp(1j * np.pi * p0val)
        x2 = -1.0 / lam * np.exp(-1j * np.pi * p0val)
        Xd = np.diag([x1, x2])
        woodbury = -4 * np.pi ** 2 * np.trace(Xd @ Rk @ Xd @ Rkp)
        worst_woodbury = max(worst_woodbury, abs(brute - woodbury))

        t = np.exp(2j * np.pi * p0val)
        term4 = (t * lam ** 2 * Rk[0, 0] * Rkp[0, 0] + (1.0 / (t * lam ** 2)) * Rk[1, 1] * Rkp[1, 1]
                 - Rk[0, 1] * Rkp[1, 0] - Rk[1, 0] * Rkp[0, 1])
        four_term = -4 * np.pi ** 2 * term4
        worst_4term = max(worst_4term, abs(brute - four_term))

    check("L-3 Woodbury 2x2 reduction reproduces the brute-force trace EXACTLY (pointwise, 3 axes)",
          worst_woodbury < 1e-9, f"worst dev={worst_woodbury:.2e}")
    check("L-4 the explicit 4-term formula (x1*x2==-1 identically collapsed) also matches exactly",
          worst_4term < 1e-9, f"worst dev={worst_4term:.2e}")

    # =======================================================================================
    banner("L-5  SECOND WOODBURY -- R(k) == R0(k_perp) [I + C(k_mu) R0(k_perp)]^{-1}")
    print(" (same rank-2 mechanism applied to k_mu itself, treating k_mu=0 as reference)")
    # =======================================================================================
    axis_test = 0
    k_perp = np.array([0.37, 0.61])

    def R0_ref(k_perp, mu):
        k_ref = np.zeros(3)
        others = [a for a in range(3) if a != mu]
        k_ref[others[0]] = k_perp[0]
        k_ref[others[1]] = k_perp[1]
        return R_of(k_ref, mu)

    R0mat = R0_ref(k_perp, axis_test)
    worst_L5 = 0.0
    for kmu in (0.05, 0.23, 0.5, 0.77, 0.91):
        kfull = np.zeros(3)
        others = [a for a in range(3) if a != axis_test]
        kfull[axis_test] = kmu
        kfull[others[0]] = k_perp[0]
        kfull[others[1]] = k_perp[1]
        R_direct = R_of(kfull, axis_test)
        z = np.exp(2j * np.pi * kmu)
        Cmat = np.diag([-u * (z - 1), -u * (1.0 / z - 1)])
        R_pred = R0mat @ np.linalg.inv(np.eye(2) + Cmat @ R0mat)
        worst_L5 = max(worst_L5, np.max(np.abs(R_direct - R_pred)))
    check("L-5 R(k) == R0(k_perp)[I+C(k_mu)R0(k_perp)]^{-1} exactly (5 k_mu values, fixed k_perp)",
          worst_L5 < 1e-9, f"worst dev={worst_L5:.2e}")

    # =======================================================================================
    banner("L-6  THE ABSTRACT 2x2 LEMMA -- the k_mu-integral of Tr[X R(z) X R(zt)] is EXACTLY "
           "t-independent, for a GENERIC 2x2 R0 (not tied to srs at all)")
    print(" This is the pure linear-algebra fact the whole theorem rests on. Checked to high")
    print(" precision via a fine k_mu quadrature (N=4000), for both a random abstract R0 AND the")
    print(" actual srs-derived R0(k_perp) above -- reconciling the abstract lemma with the real object.")
    # =======================================================================================

    def R_of_z_abstract(z, R0, uu):
        C11 = -uu * (z - 1)
        C22 = -uu * (1.0 / z - 1)
        M11, M12 = 1 + C11 * R0[0, 0], C11 * R0[0, 1]
        M21, M22 = C22 * R0[1, 0], 1 + C22 * R0[1, 1]
        detM = M11 * M22 - M12 * M21
        Minv = np.array([[M22, -M12], [-M21, M11]]) / detM
        return R0 @ Minv

    def T_of_kmu(kmu, p0, R0, uu):
        z = np.exp(2j * np.pi * kmu)
        t = np.exp(2j * np.pi * p0)
        lam = z
        R = R_of_z_abstract(z, R0, uu)
        Q = R_of_z_abstract(z * t, R0, uu)
        return (t * lam ** 2 * R[0, 0] * Q[0, 0] + (1.0 / (t * lam ** 2)) * R[1, 1] * Q[1, 1]
                - R[0, 1] * Q[1, 0] - R[1, 0] * Q[0, 1])

    def kmu_integral(p0, R0, uu, N=4000):
        kmus = np.arange(N) / N
        vals = np.array([T_of_kmu(km, p0, R0, uu) for km in kmus])
        return np.mean(vals)

    rng2 = np.random.default_rng(42)
    R0_abstract = rng2.normal(size=(2, 2)) + 1j * rng2.normal(size=(2, 2))
    u_abstract = 0.13
    p0_scan = (0.0, 0.05, 0.15, 0.3, 0.45, 0.7)
    vals_abstract = [kmu_integral(p0, R0_abstract, u_abstract) for p0 in p0_scan]
    spread_abstract = max(abs(v - vals_abstract[0]) for v in vals_abstract)
    check("L-6a ABSTRACT lemma: k_mu-integral independent of p0 for a GENERIC random 2x2 R0 "
          f"(spread {spread_abstract:.2e} vs scale {abs(vals_abstract[0]):.2e})",
          spread_abstract < 1e-9 * max(abs(vals_abstract[0]), 1.0))

    vals_srs = [kmu_integral(p0, R0mat, u) for p0 in p0_scan]
    spread_srs = max(abs(v - vals_srs[0]) for v in vals_srs)
    check("L-6b SAME lemma applied to the ACTUAL srs-derived R0(k_perp) matrix (this reduction is "
          "not just abstract -- it is the real object)", spread_srs < 1e-9 * max(abs(vals_srs[0]), 1e-15),
          f"spread {spread_srs:.2e} vs scale {abs(vals_srs[0]):.2e}")

    # residue-at-z=0 general fact: Res_{z=0} T(z,t)/z == 1/u^2, independent of R0 and t (see theorem
    # note for the symbolic (sympy) derivation; checked numerically here via a tiny-contour estimate)
    def residue_at_zero(p0, R0, uu, eps=1e-6, n_ang=64):
        # (1/2pi i) oint_{|z|=eps} T(z,t)/z dz  ~=  average of T(z,t) over a tiny circle (z->0 limit
        # of T itself, since T(z,t)/z has a SIMPLE pole at 0 <=> T(z,t) -> const as z->0)
        angs = np.arange(n_ang) / n_ang * 2 * np.pi
        zs = eps * np.exp(1j * angs)
        t = np.exp(2j * np.pi * p0)
        vals = []
        for z in zs:
            lam = z
            R = R_of_z_abstract(z, R0, uu)
            Q = R_of_z_abstract(z * t, R0, uu)
            vals.append(t * lam ** 2 * R[0, 0] * Q[0, 0] + (1.0 / (t * lam ** 2)) * R[1, 1] * Q[1, 1]
                        - R[0, 1] * Q[1, 0] - R[1, 0] * Q[0, 1])
        return np.mean(vals)

    res0_est = residue_at_zero(0.2, R0_abstract, u_abstract)
    res0_pred = 1.0 / u_abstract ** 2
    check("L-6c general fact Res_{z=0} = 1/u^2 (independent of R0, t) -- small-circle numeric estimate",
          abs(res0_est.real - res0_pred) < 1e-3 * res0_pred, f"est={res0_est:.6f}  1/u^2={res0_pred:.6f}")

    # =======================================================================================
    banner("L-7  RECONCILIATION -- reproduce CS-1's OWN reported S-3c brute-force Pi_00(p) numbers")
    # =======================================================================================

    def Pi_00_bruteforce(p0, N=16):
        K = grid(N)
        p = np.zeros(3); p[0] = p0
        Kmid = K + 0.5 * p
        Bk = B_batch(K)
        Bkp = B_batch(K + p)
        Gk = np.linalg.inv(I_ND[None, :, :] - u * Bk)
        Gkp = np.linalg.inv(I_ND[None, :, :] - u * Bkp)
        Va = V_batch(Kmid, 0)
        VaG = np.einsum("nij,njk->nik", Va, Gk)
        VaGp = np.einsum("nij,njk->nik", Va, Gkp)
        return np.mean(np.einsum("nij,nji->n", VaG, VaGp)).real

    cs1_reported = {0.0: 4.24180382422406e-09, 0.45: 4.24180382424947e-09}
    worst_reconcile = 0.0
    for p0, reported in cs1_reported.items():
        mine = Pi_00_bruteforce(p0)
        dev = abs(mine - reported) / abs(reported)
        worst_reconcile = max(worst_reconcile, dev)
        print(f"    p0={p0:.2f}: this script={mine:.14e}  CS-1 return={reported:.14e}  rel_dev={dev:.2e}")
    check("L-7 reproduces CS-1's own reported Pi_00(p) endpoints (S-3c table) to high precision",
          worst_reconcile < 1e-6, f"worst rel dev={worst_reconcile:.2e}")

    # =======================================================================================
    banner("L-8  CONTRAST -- brute-force TRANSVERSE channel genuinely depends on p (from the SAME "
           "code path), confirming the theorem's scope claim is not an over-cancellation bug")
    # =======================================================================================

    def Pi_11_bruteforce(p0, N=16):
        K = grid(N)
        p = np.zeros(3); p[0] = p0
        Kmid = K + 0.5 * p
        Bk = B_batch(K)
        Bkp = B_batch(K + p)
        Gk = np.linalg.inv(I_ND[None, :, :] - u * Bk)
        Gkp = np.linalg.inv(I_ND[None, :, :] - u * Bkp)
        Va = V_batch(Kmid, 1)
        VaG = np.einsum("nij,njk->nik", Va, Gk)
        VaGp = np.einsum("nij,njk->nik", Va, Gkp)
        return np.mean(np.einsum("nij,nji->n", VaG, VaGp)).real

    long_vals = [Pi_00_bruteforce(p0) for p0 in (0.0, 0.05, 0.15, 0.30, 0.45)]
    trans_vals = [Pi_11_bruteforce(p0) for p0 in (0.0, 0.05, 0.15, 0.30, 0.45)]
    long_relspread = (max(long_vals) - min(long_vals)) / abs(long_vals[0])
    trans_relspread = (max(trans_vals) - min(trans_vals)) / abs(trans_vals[0])
    print(f"    longitudinal Pi_00(p||e0) rel spread over p0 in [0,0.45]: {long_relspread:.3e}")
    print(f"    transverse   Pi_11(p||e0) rel spread over p0 in [0,0.45]: {trans_relspread:.3e}")
    check("L-8 longitudinal channel is p-independent to numerical-floor precision", long_relspread < 1e-9)
    check("L-8 transverse channel is GENUINELY p-dependent (spread >> floor) -- the theorem's scope "
          "exclusion is not an accident", trans_relspread > 1e-4)

    # =======================================================================================
    banner("SUMMARY")
    # =======================================================================================
    print(f"""  L-0 object regression ............................ {'PASS' if worst0 < 1e-13 else 'FAIL'}
  L-1 B(k+p)=D(p)B(k) ............................... {'PASS' if dev_L1 < 1e-12 else 'FAIL'}
  L-2a longitudinal vertex shift law ................ {'PASS' if worst_long < 1e-12 else 'FAIL'}
  L-2b transverse vertex shift-INSENSITIVITY ........ {'PASS' if worst_trans < 1e-12 else 'FAIL'}
  L-3 Woodbury 2x2 reduction (pointwise) ............ {'PASS' if worst_woodbury < 1e-9 else 'FAIL'}
  L-4 explicit 4-term formula ....................... {'PASS' if worst_4term < 1e-9 else 'FAIL'}
  L-5 second Woodbury (R(k) Mobius form) ............ {'PASS' if worst_L5 < 1e-9 else 'FAIL'}
  L-6a abstract 2x2 lemma (generic R0) .............. {'PASS' if spread_abstract < 1e-9*max(abs(vals_abstract[0]),1.0) else 'FAIL'}
  L-6b same lemma on the ACTUAL srs R0(k_perp) ...... {'PASS' if spread_srs < 1e-9*max(abs(vals_srs[0]),1e-15) else 'FAIL'}
  L-6c general Res_{{z=0}}=1/u^2 ....................... {'PASS' if abs(res0_est.real-res0_pred) < 1e-3*res0_pred else 'FAIL'}
  L-7 reconciles CS-1's own reported numbers ........ {'PASS' if worst_reconcile < 1e-6 else 'FAIL'}
  L-8 scope contrast (longitudinal vs transverse) ... {'PASS' if (long_relspread < 1e-9 and trans_relspread > 1e-4) else 'FAIL'}
  wall time: {elapsed():.1f}s
""")
    print("RESULT:", "ALL CHECKS PASS" if ok_all else "*** A CHECK FAILED -- see FAIL lines above ***")
    banner("DONE")
    return ok_all


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
