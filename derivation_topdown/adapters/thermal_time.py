#!/usr/bin/env python3
"""
derivation_topdown/adapters/thermal_time.py

G5a ADAPTER -- the CONNES-ROVELLI THERMAL-TIME contract suite on the tick sector of the run.
Pre-registered in internal research notes (frozen BEFORE this file).
Companion charter: internal research notes (G5a = "thermal_time");
protocol: internal research notes (this file = pipeline step 3,
IMPLEMENTATION). Sibling adapters (G1-G4, G5b, G5c, G6) are scaffolded in adapters/README.md.

WHAT THIS FILE IS: an ADAPTER, not a new derivation. It imports the reused M0-2R station
recipes (the run marginal, the criticality/currency theorems) and the master net object
(state/the_net.py, for the tick anchor) and asserts, on the object the framework already
built, a CONSTRUCTIVE INSTANTIATION of the thermal-time hypothesis:

    Connes, A. & Rovelli, C., "Von Neumann algebra automorphisms and time-thermodynamics
        relation in generally covariant quantum theories", Class. Quantum Grav. 11 (1994) 2899.
    Tomita, M. & Takesaki, M. -- Tomita-Takesaki modular theory (the modular automorphism group).
    Kubo, R. (1957); Martin, P.C. & Schwinger, J. (1959) -- the KMS condition.

The thermal-time hypothesis says: physical time is not a primitive external parameter but the
MODULAR FLOW of the state one happens to be in. This suite makes that concrete and falsifiable
on the framework's own object: the run's tick-count subalgebra, in the run's own (pure, globally)
state omega, is claimed to be a Gibbs/KMS state with respect to the TICK-NUMBER operator N-hat --
i.e. thermal time (the modular flow of omega) coincides with the physical tick flow.

THE KEY DISCIPLINE POINT (restated from the pre-reg, binding throughout this file): the identity
<A sigma_ibeta(B)> = <BA> built with sigma FROM RHO ITSELF is an algebraic tautology (cyclicity of
trace) -- it holds for ANY invertible rho and proves nothing. The content-bearing claim is that
the MODULAR generator equals the physical TICK generator, K_mod ~ N-hat. Every KMS check below
therefore conjugates by e^{-beta_eff * N-hat} (the independently-defined tick-counting operator),
never by rho itself.

CLAIM = INSTANTIATION, NOT EQUIVALENCE: a green suite means "the run's tick marginal, restricted
to the N-hat subalgebra, satisfies the KMS boundary condition w.r.t. the physical tick flow at the
derived beta_eff = 2 log(u_c/alpha_1); run it and see." It does not claim the tick algebra is a
von Neumann algebra of any particular type, nor that a crossed-product/observer construction
exists (those are KMS-6's declared scope, deferred to G5b/G5c).

THE CONTRACTS (frozen tolerances; see internal research notes verbatim):
  KMS-0  ANCHOR (regression)         -- net.anchor_tick_2pi() True (N-hat integer spectrum;
                                         minimal period exactly 2pi).
  KMS-1  THE RUN MARGINAL IS
         GEOMETRIC                  -- rebuild p_n via the M0_2R_T1 marginal(u,Bmat,lamP,N,seed)
                                         recipe (u=alpha_1=(2/3)^8, B=Hashimoto at Gamma, same
                                         seed/truncation as that file). p_{n+1}/p_n constant over
                                         all computed n, relative std < 1e-12; ratio ==
                                         (alpha_1/u_c)^2 with u_c=1/(k-1)=1/2 (< 1e-12).
  KMS-2  THE MODULAR GENERATOR IS
         THE TICK (affine)          -- -log p_n = beta_eff*n + const, max residual < 1e-9; fitted
                                         slope == derived beta_eff = 2*log(u_c/alpha_1) (< 1e-9).
  KMS-3  GIBBS FORM (matrix-level)  -- on the tick truncation, rho_run := diag(p_n) equals
                                         e^{-beta_eff*N-hat}/Tr(e^{-beta_eff*N-hat}) with max-abs
                                         matrix distance < 1e-9.
  KMS-4  TWO-POINT KMS w.r.t. THE
         TICK FLOW                  -- sigma^tick_i(B) := e^{-beta_eff*N-hat} B e^{+beta_eff*N-hat}
                                         (the z=i analytic point); for all ordered pairs (A,B) in
                                         the frozen observable set {S, S^dag, S^2, S^dag2, P_0,
                                         P_1, N-hat, 3 pseudo-random Hermitians seed(0)}:
                                         |Tr(rho_run.A.sigma_i(B)) - Tr(rho_run.B.A)| < 1e-9.
  KMS-5  THE CURRENCY POINT
         (symbolic regression)      -- sympy-only (no floats): 2^{-L} = e^{-beta*kappa*L} for all
                                         L forces beta*kappa = ln 2 exactly; at that point the
                                         per-tick Boltzmann factor 2^{-b_edge} = u_c exactly.
  KMS-6  SCOPE DECLARATION          -- printed, not computed; never gates PASS/FAIL.

REUSE MAP (zero physics added; every physics-bearing symbol below is copied/re-expressed from
the named prior-art file, not re-derived):
  proofs/foundations/M0_2R_T1_run_kms_tick_2026-07-07.py
      lines ~48-61 (S0): k=srs.DEG, q=k-1, u_c=1/q, alpha_1=(q/k)^(10-2), B0=srs.hashimoto(Gamma),
      lam_P = Perron eigenvalue of B0 -- copied verbatim (KMS-1 S0 block below).
      lines ~67-80: shell_norms(u,Bmat,lamP,N,seed) and marginal(u,Bmat,lamP,N,seed) -- copied
      verbatim (the Perron-normalised, non-normality-safe recipe) (KMS-1).
      lines ~86-124 (T1a/T1b): the geometric-ratio check and the affine-fit (-log p_n) check --
      re-expressed here as KMS-1 and KMS-2 under the frozen tolerances of this pre-reg.
  proofs/foundations/M0_2R_T2_T3_arrow_criticality_currency_2026-07-07.py
      lines ~226-243 (T3): the sympy currency derivation (2^{-L}=e^{-beta*kappa*L} forces
      beta*kappa=ln2; per-tick factor at that point = u_c) -- copied recipe, re-expressed
      SYMBOLICALLY THROUGHOUT (no float cast) per the pre-reg's KMS-5 discipline (KMS-5).
  proofs/foundations/M0_convention_control_2026-07-07.py
      lines ~248-263 (C-KMS): the two-point KMS template <a_i sigma(a_j^dag)> = <a_j^dag a_i> --
      ADAPTED per the pre-reg: sigma here conjugates by e^{-beta_eff*N-hat} (the TICK generator),
      never by rho (KMS-4).
  derivation_topdown/state/the_net.py
      anchor_tick_2pi() -- imported and called directly, unmodified (KMS-0).

POISONS (binding, per pre-reg): no engine/proofs edits (bridge/the_run.py, state/the_net.py,
proofs/ are untouched; only this one new file is created); no new physics; no constants beyond
alpha_1=(2/3)^8 and u_c=1/(k-1) (both engine-derived, not hand-typed -- read off srs.DEG below),
declared tolerances, and np.random.seed(0); the observable set and tolerances are frozen exactly
as pre-registered; NO substituting the rho-tautology for the tick-flow KMS check (the flow
generator is ALWAYS N-hat, never rho, in every sigma(.) built below). A failing contract is
reported as a finding, never loosened or reworded.
"""
import math
import os
import sys
import time

import numpy as np
import sympy as sp

_T0 = time.time()

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
import the_net as net  # noqa: E402

np.set_printoptions(precision=6, suppress=True)
ok_all = True          # gates KMS-0..KMS-5 only (KMS-6 is a declaration, never a gate)


def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond


def banner(t):
    print("=" * 88)
    print(f" {t}")
    print("=" * 88)


print("=" * 88)
print(" G5a ADAPTER -- Connes-Rovelli thermal-time contract suite on the tick sector")
print(" (Connes-Rovelli CQG 11 (1994) 2899; Tomita-Takesaki; Kubo-Martin-Schwinger)")
print("=" * 88)

# ===========================================================================
banner("KMS-0  ANCHOR  (regression: the tick modular flow is a compact U(1) of period 2pi)")
# ===========================================================================
a_tick = net.anchor_tick_2pi()
check("KMS-0 net.anchor_tick_2pi(): N-hat integer spectrum, minimal period exactly 2pi",
      a_tick, detail=f"anchor_tick_2pi() = {a_tick}")

# ===========================================================================
banner("KMS-1  THE RUN MARGINAL IS GEOMETRIC  (M0_2R_T1 S0/T1a recipe, verbatim reuse)")
# ===========================================================================
# S0 (verbatim from M0_2R_T1_run_kms_tick_2026-07-07.py lines ~48-61): the forced constants,
# read off the object -- nothing hand-typed beyond the tick truncation N_max and the frozen
# tolerances.
k = srs.DEG                                   # coordination number (READ) = 3
q = k - 1                                     # NB branching = continuations per dart
u_c = 1.0 / q                                 # path-gas critical fugacity (Perron combinatorics)
alpha1 = (q / k) ** (10 - 2)                  # the run's operating fugacity ((2/3)^8), the_run.py
B0 = srs.hashimoto((0, 0, 0)).real            # 12x12 NB step at Gamma (real, integer)
ND = B0.shape[0]
lam_P = max(abs(np.linalg.eigvals(B0)))       # Perron eigenvalue of B(Gamma)
print(f"    k={k} q=k-1={q}  u_c=1/(k-1)={u_c}  alpha_1=(2/3)^8={alpha1:.10f}  "
      f"lam_P(B(Gamma))={lam_P:.10f}")
check("KMS-1 pre-req: Perron eigenvalue of B(Gamma) = k-1 (=> u_c = 1/(k-1) is the correct read)",
      abs(lam_P - q) < 1e-9, detail=f"lam_P={lam_P:.10f}, q={q}")

PERRON = np.ones(ND) / math.sqrt(ND)          # the stationary (Perron/equilibrium) run seed


def shell_norms(u, Bmat, lamP, N, seed):
    """||shell n||^2 = u^{2n} ||B^n seed||^2, Perron-normalised to avoid overflow (verbatim
    M0_2R_T1 recipe). ||v||^2 uses vdot (Hermitian norm) -- correct for complex fibers."""
    Bh = Bmat / lamP
    v = seed.astype(complex).copy()
    out = []
    for n in range(N + 1):
        out.append(((u * lamP) ** (2 * n)) * float(np.vdot(v, v).real))
        v = Bh @ v
    return np.array(out)


def marginal(u, Bmat, lamP, N, seed):
    """The N-hat marginal p_n (verbatim M0_2R_T1 recipe)."""
    w = shell_norms(u, Bmat, lamP, N, seed)
    return w / w.sum()


N_max = 40                                    # SAME truncation as M0_2R_T1's T1(a) equilibrium test
p = marginal(alpha1, B0, lam_P, N_max, PERRON)
ratios = p[1:] / p[:-1]
r_mean = float(np.mean(ratios))
r_rel_std = float(np.std(ratios) / r_mean)
born2 = (alpha1 / u_c) ** 2

# geometric tail bound: p decays as ~r_mean^n; print the truncation depth and the bound it buys.
tail_bound = float(p[-1])
print(f"    truncation N_max = {N_max};  geometric ratio r = (alpha_1/u_c)^2 = {born2:.10f}")
print(f"    geometric tail bound: p_(N_max) = {tail_bound:.3e}  (measured directly on the "
      f"normalised truncated marginal -- the mass carried by the last computed shell)")
check("KMS-1a truncation is deep enough: p_(N_max) < 1e-15 (boundary mass negligible)",
      tail_bound < 1e-15, detail=f"p_(N_max) = {tail_bound:.3e}")
check("KMS-1b the run marginal is GEOMETRIC: p_(n+1)/p_n constant over all n, relative std < 1e-12",
      r_rel_std < 1e-12, detail=f"mean ratio = {r_mean:.12f}, relative std = {r_rel_std:.3e}")
check("KMS-1c ratio == (alpha_1/u_c)^2 (u_c=1/(k-1)=1/2, Perron combinatorics) to < 1e-12",
      abs(r_mean - born2) < 1e-12, detail=f"|ratio - (alpha_1/u_c)^2| = {abs(r_mean - born2):.3e}")

# ===========================================================================
banner("KMS-2  THE MODULAR GENERATOR IS THE TICK  (affine fit: -log p_n = beta_eff*n + const)")
# ===========================================================================
nn = np.arange(N_max + 1, dtype=float)
Mgen = -np.log(p)                             # the single-particle modular generator eigenvalues
Afit = np.vstack([nn, np.ones(N_max + 1)]).T
slope, intercept = np.linalg.lstsq(Afit, Mgen, rcond=None)[0]
resid = float(np.max(np.abs(Mgen - (slope * nn + intercept))))
beta_eff = 2 * math.log(u_c / alpha1)         # the DERIVED inverse tick-temperature
print(f"    affine fit over n=0..{N_max}: slope = {slope:.10f}, intercept = {intercept:.6f}")
print(f"    derived beta_eff = 2*log(u_c/alpha_1) = {beta_eff:.10f}")
check("KMS-2a -log p_n is EXACTLY AFFINE in n (max residual < 1e-9) => modular flow ~ N-hat",
      resid < 1e-9, detail=f"max residual = {resid:.3e}")
check("KMS-2b fitted slope == derived beta_eff = 2*log(u_c/alpha_1) (< 1e-9)",
      abs(slope - beta_eff) < 1e-9, detail=f"|slope - beta_eff| = {abs(slope - beta_eff):.3e}")

# ===========================================================================
banner("KMS-3  GIBBS FORM (matrix-level)  -- rho_run == e^{-beta_eff N-hat}/Z as matrices")
# ===========================================================================
dim = N_max + 1
Nhat = np.diag(nn)                            # the tick-count operator on the truncation
rho_run = np.diag(p)                          # the run's tick marginal AS a density matrix
gibbs_unnorm = np.exp(-beta_eff * nn)
Z_gibbs = float(gibbs_unnorm.sum())
gibbs = np.diag(gibbs_unnorm / Z_gibbs)
mat_dist = float(np.max(np.abs(rho_run - gibbs)))
print(f"    truncation N_max = {N_max} (dim = {dim});  geometric tail bound p_(N_max) = "
      f"{tail_bound:.3e} (same bound as KMS-1: the boundary carries negligible mass)")
print(f"    Z = Tr(e^{{-beta_eff N-hat}}) = {Z_gibbs:.10f};  max|rho_run - e^{{-beta_eff N-hat}}/Z| "
      f"= {mat_dist:.3e}")
check("KMS-3 rho_run := diag(p_n) == e^{-beta_eff*N-hat}/Tr(e^{-beta_eff*N-hat}) as matrices "
      "(max-abs distance < 1e-9)", mat_dist < 1e-9, detail=f"max-abs distance = {mat_dist:.3e}")

# ===========================================================================
banner("KMS-4  TWO-POINT KMS w.r.t. THE TICK FLOW  (sigma built from N-hat, NEVER from rho)")
# ===========================================================================
# BOUNDARY CARE (per pre-reg): the shift operators S, S^dag are TRUNCATED on this finite space
# (S|0>=0, S^dag|N_max>=0). Any edge effect this introduces is weighted by the state's mass near
# n=N_max, i.e. by p_(N_max) ~ r^(N_max) -- already shown above to be < 1e-15, far below the 1e-9
# contract tolerance. This justifies using this truncation directly (no larger N_max needed):
print(f"    BOUNDARY-CARE bound: p_(N_max) = {tail_bound:.3e} < 1e-15  =>  any truncation-edge "
      f"artifact in the (2,B)/(A,2) sandwich is bounded by this state mass, two orders of "
      f"magnitude below the 1e-9 contract tolerance. The observable set is NOT restricted.")

rho_c = rho_run.astype(complex)

S = np.zeros((dim, dim), dtype=complex)       # the tick LOWERING shift: S|n> = |n-1>, S|0> = 0
for n_ in range(1, dim):
    S[n_ - 1, n_] = 1.0
Sdag = S.conj().T                             # the tick RAISING shift (truncated: S^dag|N_max>=0)
S2 = S @ S
Sdag2 = Sdag @ Sdag
P0 = np.zeros((dim, dim), dtype=complex); P0[0, 0] = 1.0
P1 = np.zeros((dim, dim), dtype=complex); P1[1, 1] = 1.0
Nhat_c = Nhat.astype(complex)

np.random.seed(0)                             # FROZEN seed; observable set is frozen, not tuned


def _random_hermitian(d):
    Mrand = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    return (Mrand + Mrand.conj().T) / 2.0


H1 = _random_hermitian(dim)
H2 = _random_hermitian(dim)
H3 = _random_hermitian(dim)

OBS = {"S": S, "S_dag": Sdag, "S^2": S2, "S_dag^2": Sdag2,
       "P_0": P0, "P_1": P1, "N_hat": Nhat_c, "H1(seed0)": H1, "H2(seed0)": H2, "H3(seed0)": H3}
names = list(OBS.keys())
print(f"    frozen observable set ({len(names)}): {names}")


def sigma_tick_i(Bop, beta, nvec):
    """sigma^tick_z(B) at the analytic point z=i: conjugation by e^{-beta_eff*N-hat}, i.e.
    sigma_i(B) = e^{-beta*N-hat} . B . e^{+beta*N-hat}. N-hat is diagonal, so this is the
    elementwise scaling sigma[i,j] = exp(-beta*i) * B[i,j] * exp(+beta*j) -- computed WITHOUT
    ever referencing rho (the tick-flow generator N-hat only, per the key discipline point)."""
    dm = np.exp(-beta * nvec)
    dp = np.exp(+beta * nvec)
    return (dm[:, None] * Bop) * dp[None, :]


worst_diff = 0.0
worst_pair = None
n_fail = 0
n_pairs = 0
for na in names:
    for nb in names:
        A = OBS[na]; B = OBS[nb]
        sigB = sigma_tick_i(B, beta_eff, nn)
        lhs = np.trace(rho_c @ A @ sigB)
        rhs = np.trace(rho_c @ B @ A)
        diff = abs(lhs - rhs)
        n_pairs += 1
        if diff > worst_diff:
            worst_diff = diff
            worst_pair = (na, nb)
        if diff >= 1e-9:
            n_fail += 1
print(f"    checked {n_pairs} ordered pairs (A,B); worst |Tr(rho.A.sigma_i(B)) - Tr(rho.B.A)| = "
      f"{worst_diff:.3e} at pair {worst_pair}; {n_fail} pairs at/above tolerance")
check("KMS-4 two-point KMS w.r.t. the TICK flow: |Tr(rho_run.A.sigma_i(B)) - Tr(rho_run.B.A)| "
      "< 1e-9 for ALL ordered pairs in the frozen observable set",
      worst_diff < 1e-9 and n_fail == 0,
      detail=f"worst = {worst_diff:.3e} at {worst_pair}, {n_fail}/{n_pairs} failing")

# ===========================================================================
banner("KMS-5  THE CURRENCY POINT  (symbolic regression, sympy only -- no floats)")
# ===========================================================================
# Verbatim re-expression of M0_2R_T2_T3's T3 (lines ~226-243), kept SYMBOLIC end-to-end per the
# pre-reg's KMS-5 discipline (no float() cast anywhere in this block).
beta_s, kappa_s, L_s = sp.symbols('beta kappa L', positive=True)
lhs_sym = 2 ** (-L_s)                          # amplitude representation: p = 2^{-L}
rhs_sym = sp.exp(-beta_s * kappa_s * L_s)      # energy representation: E = kappa*L, Gibbs e^{-beta E}
sol = sp.solve(sp.Eq(-sp.log(2), -beta_s * kappa_s), beta_s * kappa_s)
bk = sol[0]
print(f"    require 2^(-L) = e^(-beta*kappa*L) for all L  =>  beta*kappa = {bk}")
check("KMS-5a currency consistency FORCES beta*kappa = ln 2 EXACTLY (symbolic, no floats)",
      sp.simplify(bk - sp.log(2)) == 0, detail=f"beta*kappa = {bk}")

q_int = sp.Integer(k - 1)                      # q = k-1 = 2, read off srs.DEG above (as an int)
b_edge_sym = sp.log(q_int, 2)                  # b_edge = log2(q); sympy exact-simplifies to 1
u_c_sym = sp.Rational(1, k - 1)                # u_c = 1/(k-1), EXACT rational
per_tick_sym = sp.exp(-sp.log(2) * b_edge_sym)  # the per-tick Boltzmann factor at beta*kappa=ln2
print(f"    b_edge = log2(k-1) = {b_edge_sym} (exact);  per-tick factor e^(-ln2*b_edge) = "
      f"{per_tick_sym};  u_c = 1/(k-1) = {u_c_sym}")
check("KMS-5b per-tick Boltzmann factor at the Landauer point = 2^(-b_edge) = u_c EXACTLY "
      "(symbolic equality, no floats)",
      sp.simplify(per_tick_sym - u_c_sym) == 0,
      detail=f"per_tick = {per_tick_sym}, u_c = {u_c_sym}")

# ===========================================================================
banner("KMS-6  SCOPE DECLARATION  (printed, NOT computed; never gates PASS/FAIL)")
# ===========================================================================
print("""  This suite does NOT claim, and none of KMS-0..KMS-5 establishes:
    (i)   The von Neumann TYPE of the tick algebra (deferred to a future station, G5b).
    (ii)  The crossed-product / observer construction relating the modular flow to an emergent
          observer algebra (deferred to a future station, G5c).
    (iii) KMS on SPATIAL regions / causal diamonds (that is ML-1/G4 territory -- a DIFFERENT
          modular flow, the diamond entanglement Hamiltonian, not the global tick generator
          used here).
    (iv)  Any thermodynamic-limit / continuum statement. The truncation is FINITE (N_max = 40);
          the geometric tail bound p_(N_max) = %.3e (< 1e-15) justifies treating this truncation
          as exact at the declared 1e-9 tolerance, but no N_max -> infinity limit is taken or
          claimed.
  These remain OPEN and are carried into adapters/README.md as declared, unclaimed scope.""" % tail_bound)

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
elapsed = time.time() - _T0
kms1_ok = (tail_bound < 1e-15 and r_rel_std < 1e-12 and abs(r_mean - born2) < 1e-12)
kms2_ok = (resid < 1e-9 and abs(slope - beta_eff) < 1e-9)
kms3_ok = (mat_dist < 1e-9)
kms4_ok = (worst_diff < 1e-9 and n_fail == 0)
kms5_ok = (sp.simplify(bk - sp.log(2)) == 0 and sp.simplify(per_tick_sym - u_c_sym) == 0)
print(f"""  KMS-0  anchor (tick U(1), period 2pi)      : {'PASS' if a_tick else 'FAIL'}
  KMS-1  run marginal is geometric            : {'PASS' if kms1_ok else 'FAIL'}  (ratio={r_mean:.6e}, rel_std={r_rel_std:.2e})
  KMS-2  modular generator == tick (affine)    : {'PASS' if kms2_ok else 'FAIL'}  (beta_eff={beta_eff:.6f}, resid={resid:.2e})
  KMS-3  Gibbs form (matrix-level)             : {'PASS' if kms3_ok else 'FAIL'}  (max-abs dist={mat_dist:.2e})
  KMS-4  two-point KMS w.r.t. tick flow        : {'PASS' if kms4_ok else 'FAIL'}  (worst={worst_diff:.2e} over {n_pairs} pairs)
  KMS-5  currency point (symbolic)             : {'PASS' if kms5_ok else 'FAIL'}  (beta*kappa=ln2 exact; per-tick=u_c exact)
  KMS-6  scope declaration                     : printed above (declaration only, not a gate)
  wall time: {elapsed:.1f}s""")
print("RESULT:", "ALL KMS-0..KMS-5 CONTRACTS PASS (Connes-Rovelli thermal-time instantiation "
      "at the derived beta_eff)" if ok_all else
      "AT LEAST ONE CONTRACT FAILED -- see per-contract detail above (a finding, not a bug)")
sys.exit(0 if ok_all else 1)
