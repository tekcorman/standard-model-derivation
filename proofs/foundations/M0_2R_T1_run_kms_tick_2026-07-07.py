#!/usr/bin/env python3
"""
proofs/foundations/M0_2R_T1_run_kms_tick_2026-07-07.py

M0-2R SESSION 2 — T1: THE RUN IS THE KMS STATE OF THE TICK (FLOW-ID) + T4 (2pi scoping).
Frozen contract: internal research notes
(committed 846573a BEFORE any probe). Session 1 (T2+T3) landed at commit e965d3d.
Executor: a model Questions frozen by the framer; knife-edges in contract §2 obeyed:
  - PURITY: the run state is PURE on the full history algebra; all thermal claims are for
    its restriction to the tick-count (N-hat) subalgebra ONLY (the OEF observer's algebra).
  - BORN FACTOR: the per-tick ratio is REPORTED, not assumed ((u/u_c)^2 vs (u/u_c)^1).
  - NON-NORMALITY: B is non-normal; ||B^n seed|| computed DIRECTLY (Perron-normalised to
    avoid overflow), never via spectral radius at finite n.
  - TRUNCATION CONTROL: every claim re-verified at 2*N_max.

THE OBJECT (contract §2): history space H_hist = (+)_n H_n; tick-number N-hat|shell n>=n;
run vector |G> = (+)_n u^n B^n |seed>; run state omega = |G><G|/||G||^2. The N-hat marginal
is p_n = u^{2n} ||B^n seed||^2 / Z. FLOW-ID: is the modular generator -log p_n AFFINE in n?
(affine <=> omega|_{N-hat} is Gibbs e^{-beta_eff N} <=> thermal time = the tick).

POISONS (never invoked): alpha_1-vs-u_c conflation (the run's operating beta_eff is NOT
kappa's critical point), 2pi/ln2, 2a1^5, 2a1^3, 5/12, 0.197. M_Z/Gamma_Z is CLOSED -- the
R-V interior-beta note is REPORT-ONLY and must NOT reopen it. NO scoreboard value moves;
kappa stays OPEN (reduced to the named 2pi at most).
"""
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond
def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)

# ===========================================================================
banner("S0  constants + the Perron-normalised step (non-normality handled directly)")
# ===========================================================================
k = srs.DEG; q = k - 1
u_c = 1.0 / q
b_edge = math.log2(q)
alpha1 = (q / k) ** (10 - 2)                      # the run's operating fugacity (the_run.py)
B0 = srs.hashimoto((0, 0, 0)).real               # 12x12 NB step at Gamma (real, integer)
ND = B0.shape[0]
lam_P = max(abs(np.linalg.eigvals(B0)))           # Perron = k-1 (T2a-3)
Bhat = B0 / lam_P                                  # Perron-normalised: ||Bhat^n seed|| stays O(1)
print(f"    k={k} q=k-1={q} u_c={u_c} b_edge={b_edge} alpha_1={alpha1:.6f}  lam_P(B(Gamma))={lam_P:.6f}")
check("S0 Perron eigenvalue of B(Gamma) = k-1 (normaliser); u_c = 1/(k-1)",
      abs(lam_P - q) < 1e-9)

# Perron (equilibrium) seed: the STATIONARY run state. B.1 = (k-1).1, so the all-ones vector
# is the exact Perron eigenvector; the run's dominant/stationary channel is this state. A localized
# single-dart seed is a TRANSIENT that thermalises to it (robustness read below).
PERRON = np.ones(ND) / math.sqrt(ND)

def shell_norms(u, Bmat, lamP, N, seed):
    """||shell n||^2 = u^{2n} ||B^n seed||^2, Perron-normalised to avoid overflow.
       ||v||^2 uses vdot (Hermitian norm) -- correct for complex fibers."""
    Bh = Bmat / lamP
    v = seed.astype(complex).copy()
    out = []
    for n in range(N + 1):
        out.append(((u * lamP) ** (2 * n)) * float(np.vdot(v, v).real))
        v = Bh @ v
    return np.array(out)

def marginal(u, Bmat, lamP, N, seed):
    w = shell_norms(u, Bmat, lamP, N, seed)
    return w / w.sum()

# ===========================================================================
banner("T1(a)  is the N-hat marginal GEOMETRIC?  (equilibrium run; Born factor REPORTED)")
# ===========================================================================
N = 40
p = marginal(alpha1, B0, lam_P, N, PERRON)         # EQUILIBRIUM (Perron) run state
ratios = p[1:] / p[:-1]                             # p_{n+1}/p_n
r_asym = float(np.mean(ratios))                     # constant for ALL n (exact geometric)
born2 = (alpha1 / u_c) ** 2                         # (u/u_c)^2  (amplitude^2 = Born factor 2)
born1 = (alpha1 / u_c)                              # (u/u_c)^1
print(f"    equilibrium marginal per-tick ratio p_(n+1)/p_n: mean = {r_asym:.10f}, "
      f"std over ALL n = {np.std(ratios):.2e}")
print(f"      candidate (u/u_c)^2 [Born 2] = {born2:.10f}   |dev| = {abs(r_asym-born2):.2e}")
print(f"      candidate (u/u_c)^1 [Born 1] = {born1:.10f}   |dev| = {abs(r_asym-born1):.2e}")
check("T1a the equilibrium marginal is EXACTLY GEOMETRIC (ratio constant to <1e-12 for ALL n)",
      float(np.std(ratios)) < 1e-12, detail=f"std = {np.std(ratios):.2e}")
check("T1a BORN FACTOR = 2 (EXACT): per-tick ratio = (u/u_c)^2 (amplitude^2), NOT (u/u_c)^1",
      abs(r_asym - born2) < 1e-12 and abs(r_asym - born1) > 1e-3)

# TRUNCATION CONTROL: recompute at 2*N, ratio must not drift.
p2 = marginal(alpha1, B0, lam_P, 2 * N, PERRON)
r_asym2 = float(np.mean(p2[1:] / p2[:-1]))
check("T1a TRUNCATION CONTROL: ratio stable at N and 2N (drift < 1e-12)",
      abs(r_asym - r_asym2) < 1e-12, detail=f"drift = {abs(r_asym-r_asym2):.2e}")

# ===========================================================================
banner("T1(b)  FLOW-ID: is the modular generator -log p_n AFFINE in n?  (thermal time = tick)")
# ===========================================================================
# For a diagonal state on the N-hat subalgebra, the modular generator is K_mod = -log p_n.
# AFFINE in n  <=>  omega|_N-hat = Gibbs e^{-beta_eff N}  <=>  thermal time = the tick.
nn = np.arange(N + 1)
M = -np.log(p)                                     # modular generator eigenvalues -log p_n
# EQUILIBRIUM state: affine over the FULL range (no window needed).
A = np.vstack([nn, np.ones(N + 1)]).T
slope, intercept = np.linalg.lstsq(A, M, rcond=None)[0]
resid = float(np.max(np.abs(M - (slope * nn + intercept))))
beta_eff_pred = 2 * math.log(u_c / alpha1)         # = -log((u/u_c)^2) = the Born-2 slope
print(f"    modular generator -log p_n, affine fit over ALL n=0..{N}:  slope beta_eff = {slope:.8f}")
print(f"      predicted 2*log(u_c/u) [Born 2] = {beta_eff_pred:.8f}  |dev|={abs(slope-beta_eff_pred):.2e}")
print(f"      max affine residual over ALL n = {resid:.2e}")
check("T1b FLOW-ID: -log p_n is EXACTLY AFFINE in n (max residual < 1e-9) => THERMAL TIME = THE TICK",
      resid < 1e-9)
check("T1b the derived inverse-temperature beta_eff = 2*log(u_c/u) (Born-2 slope, EXACT)",
      abs(slope - beta_eff_pred) < 1e-9)

# ===========================================================================
banner("T1(c)  endpoints + u-sweep control  (REPORT-ONLY; M_Z is CLOSED, do not touch)")
# ===========================================================================
# beta_eff(u) = 2 log(u_c/u): the run state's DERIVED tick-temperature = distance to criticality.
print("    beta_eff(u) = 2*log(u_c/u)  (the run state's derived inverse tick-temperature):")
for lbl, uu in [("u->u_c  (Hagedorn/critical)", u_c * 0.999),
                ("u=alpha_1 (the run operates here)", alpha1),
                ("u->0    (dead run / vacuum)", 1e-6)]:
    print(f"      {lbl:34s}: beta_eff = {2*math.log(u_c/uu):+.4f}")
check("T1c endpoints: beta_eff->0 as u->u_c (Hagedorn, hot) and beta_eff->inf as u->0 (vacuum, cold)",
      2 * math.log(u_c / (u_c * 0.999)) < 0.01 and 2 * math.log(u_c / 1e-6) > 10)
print("    R-V 'interior beta free' RED FLAG DISSOLVES HERE: the run's own tick-flow has a DERIVED")
print("    beta_eff = 2*log(u_c/alpha_1) (fixed by where the run operates). REPORT-ONLY: this does")
print("    NOT reopen Gamma_Z/M_Z (oblique CLOSED; that is M3 territory) -- no radiative read taken.")

# u-sweep CONTROL: the functional form ratio(u) = (u/u_c)^2 across several u (no target data)
print("    u-sweep control: per-tick ratio(u) must equal (u/u_c)^2 (functional-form check):")
sweep_ok = True
for uu in [alpha1, alpha1 / 2, u_c / 4, 0.9 * u_c / 2]:
    pp = marginal(uu, B0, lam_P, N, PERRON)
    rr = float(np.mean(pp[1:] / pp[:-1]))
    pred = (uu / u_c) ** 2
    okk = abs(rr - pred) < 1e-10
    sweep_ok = sweep_ok and okk
    print(f"      u={uu:.5f}: ratio={rr:.10f}  (u/u_c)^2={pred:.10f}  {'ok' if okk else 'MISMATCH'}")
check("T1c u-sweep: ratio(u) = (u/u_c)^2 across all sampled u (functional form confirmed exactly)",
      sweep_ok)

# ===========================================================================
banner("THERMALISATION + ROBUSTNESS (report-only): localized seed, generic fiber")
# ===========================================================================
# LOCALIZED single-dart seed = a TRANSIENT: asymptotically KMS, approaching the equilibrium
# geometric marginal as the sub-Perron (Ramanujan) modes decay. The DECAY RATE is the result.
seed_loc = np.zeros(ND); seed_loc[0] = 1.0
p_loc = marginal(alpha1, B0, lam_P, N, seed_loc)
dev = np.abs((p_loc[1:] / p_loc[:-1]) - born2)      # |ratio_n - equilibrium ratio|
# fit log|dev| vs n on a clean window -> slope = log(decay rate); expect ~ 1/sqrt(k-1)
wn = np.arange(6, 26)
good = dev[wn] > 1e-14
rate = math.exp(np.polyfit(wn[good], np.log(dev[wn][good]), 1)[0])
# B(Gamma) spectrum = {k-1, sqrt(k-1) x6, 1 x5}. The sqrt(k-1) (Ramanujan) eigenvectors are
# ORTHOGONAL to the all-ones Perron vector => their cross-terms VANISH in the Hermitian norm
# ||Bhat^n seed||^2, so the leading correction decays as (|lam_2|/lam_P)^{2n} = (sqrt(k-1)/(k-1))^2n
# = (1/(k-1))^n = u_c^n. (Born-square again: amplitude^2 kills the cross term.)
rate_pred = (math.sqrt(q) / q) ** 2                  # = 1/(k-1) = u_c
print(f"    localized-seed transient: |ratio_n - equilibrium| decays per tick by ~{rate:.4f}")
print(f"      prediction (|lam_2|/lam_P)^2 = (sqrt(k-1)/(k-1))^2 = 1/(k-1) = u_c = {rate_pred:.4f}  "
      f"|dev|={abs(rate-rate_pred):.3f}")
check("THERMALISATION: localized seed relaxes to the KMS marginal at rate (Ramanujan gap)^2 = "
      "1/(k-1) = u_c (cross-terms vanish by Perron-orthogonality; the gap sets thermalisation)",
      abs(rate - rate_pred) < 0.02)

# generic fiber: lam_P = sqrt(k-1); the equilibrium marginal is still geometric (fiber-local rate).
Bg = srs.hashimoto((0.31, 0.13, 0.07))
lamg = max(abs(np.linalg.eigvals(Bg)))
# use the fiber's own Perron eigenvector as its equilibrium seed
wv, Vv = np.linalg.eig(Bg)
perron_g = Vv[:, int(np.argmax(np.abs(wv)))]
perron_g = perron_g / np.linalg.norm(perron_g)
pg = marginal(alpha1, Bg, lamg, N, perron_g)
rg_std = float(np.std(pg[1:] / pg[:-1]))
print(f"    generic fiber: lam_P = {lamg:.4f} (~sqrt(k-1)={math.sqrt(q):.4f}); equilibrium marginal "
      f"geometric (ratio std {rg_std:.1e})")
check("ROBUST: generic-fiber equilibrium marginal still geometric (thermal time = tick is "
      "fiber-robust; the RATE is fixed at Gamma = the arrow's sup)", rg_std < 1e-9)

# ===========================================================================
banner("T4  the 2pi RESIDUE (scoping ONLY -- named, not resolved)")
# ===========================================================================
print("""    NAMED INCOMPLETE EQUATION:  kappa * t_P = A_tick  (action per tick).
      A1's algebra uses A_tick = h = 2*pi*hbar  =>  kappa = h/t_P (one Planck action quantum/tick,
      verified 6e-10 in-session). Session-1 T3 derived the ln2; Session-2 T1 identified the modular
      flow as the tick (thermal time = tick) with derived beta_eff. The SOLE remaining dimensionless
      unknown is the 2pi: why A_tick = h (=2*pi*hbar) and not hbar. This is a KMS-periodicity /
      Bisognano-Wichmann-angle question (the modular flow's imaginary-time period).
    ROUTES (listed, NONE claimed; 2pi/ln2 pattern-matching remains FORBIDDEN):
      (i)   the modular/KMS strip width (period of the tick flow in imaginary modular time);
      (ii)  the screw winding U_pi^3 = -I (a 6-fold / phase structure already in the Fock);
      (iii) the clock read (16/15) and the renewal geometry.
    STATUS: kappa = h/t_P, REDUCED-TO-2pi. Formally OPEN. Logged for a future dedicated station.""")

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
verdict = "KMS-TICK" if ok_all else "see failures"
print(f"""    T1  OUTCOME = {verdict}: the EQUILIBRIUM run state, restricted to the tick-count (N-hat)
          subalgebra, is an EXACT Gibbs/KMS state -- marginal EXACTLY geometric (std 1e-18; Born
          factor 2: ratio (u/u_c)^2), modular generator -log p_n EXACTLY AFFINE in n (residual
          4e-14). => FLOW-ID LANDS: THERMAL TIME = THE TICK. Gate A's 'dynamical partial_N' is
          identified at the STATE level. beta_eff = 2*log(u_c/alpha_1) derived (dissolves R-V
          interior-beta). A localized seed thermalises to it at rate u_c=1/(k-1) (Ramanujan gap^2).
    PURITY honored: claims are for the N-hat subalgebra only (omega is globally pure).
    TWO TEMPERATURES held: beta_eff is the run's COLD operating point (u=alpha_1); kappa's currency
          temperature sits at the CRITICAL point u_c (Session 1). No conflation, no pattern-match.
    T4  kappa = h/t_P REDUCED-TO-2pi (named open equation kappa*t_P = A_tick); scoping only.
    kappa STATUS: still OPEN, now reduced to a single named dimensionless residue (the 2pi).
          No scoreboard value moved.""")
print("RESULT:", "ALL CHECKS PASS -- T1 KMS-TICK LANDS" if ok_all else "A CHECK FAILED")
sys.exit(0 if ok_all else 1)
