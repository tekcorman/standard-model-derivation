#!/usr/bin/env python3
"""
proofs/foundations/I2_matsumoto_confront_2026-07-10.py

STATION I-2 -- THE MATSUMOTO CONFRONT (ASSESSMENT station, MEDIUM effort; Ring-2 key,
internal research notes:44-45: "the tick's KMS temperature vs the
subshift theorem beta = log(branching entropy) -- external ratification or honest mismatch of
the ln 2 currency"). Computes NO observable; touches NO scoreboard row; edits NO register --
the architect books any adoption-downgrade after review.

THE QUESTION (frozen before any number below was computed):
  Is the framework's Landauer lock  beta*kappa = ln 2  (M0-2R T3, the currency theorem) the
  SAME mathematical statement as the graph-C*-algebra theorem (Olesen-Pedersen for O_n;
  Enomoto-Fujii-Watatani for Cuntz-Krieger O_A; Matsumoto-Watatani-Yoshida for subshift
  algebras): "for the gauge action on the algebra of a suitable irreducible subshift, the
  UNIQUE KMS state occurs at beta = log(spectral radius) = topological entropy of the shift"
  -- i.e. is our ln 2 the log of the NB-walk's growth rate, making T3 a RATIFICATION-BY-IMPORT
  -- or a DIFFERENT statement (different algebra / action / temperature normalization)?

METHOD (in-file pre-reg; all comparison numbers computed from the ENGINE matrix, imported):
  S0  read the repo's own KMS/tick objects (file:line citations; nothing re-derived);
  S1  machine-check the hypotheses on the NB dart shift: EFW's (irreducible 0-1 matrix),
      plus primitivity, plus the SEPARATE Cuntz-Krieger condition-(I)/simplicity check
      "not a permutation" (re-attributed per the adversarial check; see S1b);
  S2  machine-check spectral radius r(B0) and h_top = ln r(B0) (matrix + exact word counts);
  S3  machine-check the CK-KMS fixed-point equation (the computational core of the theorem):
      a gauge-KMS diagonal state needs a POSITIVE vector x with A_sft x = e^beta x -- Perron
      forces beta = ln r uniquely; plus the shell-mass trichotomy (state on O_A iff beta = h_top;
      Toeplitz-only iff beta > h_top; nothing iff beta < h_top);
  S4  machine-check the temperature-normalization dictionary: THEIR generator = the integer
      grading (per-tick energy 1) => beta_CK = h_top = ln(k-1); OUR Hamiltonian = kappa*L =
      kappa*b_edge*N-hat => the SAME per-tick weight iff beta*kappa = h_top/b_edge = ln 2,
      an IDENTITY for every k >= 3 (checked k = 3..12);
  S5  print the object-by-object mapping table;
  S6  print the VERDICT.

VERDICT CRITERIA (frozen FIRST; dual-outcome, no goal-seek):
  SAME-OBJECT      = the subshift, the flow generator, the critical temperature (after the
                     declared unit dictionary), and the critical state all match at the object
                     level, machine-checked where checkable;
  DIFFERENT-OBJECT = a precise mismatch in algebra, action, or temperature that no declared
                     unit conversion removes;
  UNDETERMINED     = a named missing piece blocks the identification.
  The three rows are adjudicated SEPARATELY (a qualified verdict is allowed and expected);
  the honest prior is stated in the contract: growth rate k-1 = 2 gives log 2, but whether OUR
  flow is THEIR gauge action needs the object-level check -- do not force it.

THE REPO'S OWN OBJECTS (S0 citations; read, not re-derived):
  * The engine matrix: srs.hashimoto (derivation_topdown/dirac_srs_mdl/srs.py:42-49), re-exported
    as the geodesic generator by derivation_topdown/bridge/the_run.py:46. IMPORTED here via
    the_run (task discipline: import, never rebuild). Convention: B[b,a] = 1 iff dart a -> dart b
    is an admissible non-backtracking step (column = source), so the SFT transition matrix with
    row = source is A_sft = B0^T.
  * The history algebra + state: H_hist = (+)_n H_n, tick number N-hat, run vector
    |G> = (+)_n u^n B^n |seed>, marginal p_n = u^{2n}||B^n seed||^2/Z
    (proofs/foundations/M0_2R_T1_run_kms_tick_2026-07-07.py:16-19,67-80).
  * The flow: FLOW-ID -- the modular generator of the run state restricted to the N-hat
    subalgebra is AFFINE in n, K_mod = beta_eff*N-hat, beta_eff = 2 ln(u_c/alpha_1)
    (M0_2R_T1:107-124); the tick flow's U(1) has minimal period 2pi because spec(N-hat) = Z>=0
    consecutive (M0_2R_T4_twopi_2026-07-07.py:79-100).
  * The currency theorem: p = 2^{-L} and E = kappa*L jointly consistent with Gibbs e^{-beta E}
    IFF beta*kappa = ln 2 (M0_2R_T2_T3_arrow_criticality_currency_2026-07-07.py:224-237,
    T3-1); at that point the per-tick Boltzmann factor is 2^{-b_edge} = u_c = 1/(k-1)
    (same file:239-250, T3-2), with b_edge = log2(k-1), u_c = 1/(k-1) (same file:56-63).
  * The arrow: forward run converges iff u < u_c (T2c, same file:181-199); the gas partition
    function Z(u) = 1/(1-(k-1)u) with pole at u_c (T2b, same file:148-169).
  * The two-point KMS check w.r.t. the tick flow (conjugation by e^{-beta_eff*N-hat}):
    derivation_topdown/adapters/thermal_time.py:235-304 (G5a KMS-4; beta_eff = 5.1011473686,
    DEFINED at thermal_time.py:209 as beta_eff = 2 ln(u_c/alpha_1)).

THE LITERATURE SIDE (originally stated from knowledge with EXACT theorem numbers FLAGGED for
verification; the load-bearing statements have SINCE BEEN CONFIRMED by the adversarial checker
-- see the LITERATURE VERIFICATION block below and the adjudicated list in S6):
  * Olesen-Pedersen (~1978): the gauge action on the Cuntz algebra O_n admits exactly one
    KMS state, at beta = log n.
  * Enomoto-Fujii-Watatani (Math. Japon. 29(4) (1984), 607-619): for O_A with A an
    irreducible 0-1 matrix, the gauge action gamma_z(S_i) = z S_i admits a UNIQUE KMS state,
    at beta = log r(A) (= h_top of the subshift sigma_A, Parry); its restriction to the
    diagonal is the Parry measure (per-symbol fugacity 1/r(A)). [The "not a permutation"
    clause originally recalled here is the CK condition-(I)/simplicity hypothesis, NOT an
    EFW-KMS hypothesis -- see the LITERATURE VERIFICATION block and S1b.]
  * Matsumoto-Watatani-Yoshida (~1998): the same statement for the C*-algebras of general
    irreducible subshifts, beta = h_top.
  * Laca-Neshveyev (J. Funct. Anal. 211 (2004), as recalled) / Exel-Laca: on the TOEPLITZ
    extension (the Fock-type algebra with the vacuum defect Sum S_i S_i^* = 1 - P_vac), KMS_beta
    states EXIST for all beta >= log r(A) (a simplex, parametrized by boundary/trace data,
    seed-dependent) and none below; at beta = log r(A) the state is unique and factors through
    the Cuntz-Krieger quotient O_A.
  * Parry (1964): h_top(SFT_A) = log r(A); the Parry measure is the unique measure of maximal
    entropy.

LITERATURE VERIFICATION (adversarial checker, 2026-07-10 -- CONFIRMED, with sources):
  * Enomoto-Fujii-Watatani, Math. Japon. 29(4) (1984) 607-619 -- CONFIRMED: unique gauge-KMS
    at beta = ln rho(A); KMS states <-> normalized positive Perron eigenvectors. The
    "not a permutation" clause is NOT an EFW hypothesis (it is CK condition-(I)/simplicity);
    accessible recaps require only irreducible 0-1, and uniqueness survives the permutation
    case at beta = 0 (aHLRS Remark 4.5).
  * Olesen-Pedersen, Math. Scand. 42 (1978) 111-118 -- CONFIRMED: O_n, unique KMS at
    beta = log n.
  * an Huef-Laca-Raeburn-Sims, "KMS states on the C*-algebras of finite graphs",
    arXiv:1205.2194, Theorems 3.1 + 4.3 (built on Laca-Neshveyev, J. Funct. Anal. 211 (2004)
    457-482) -- CONFIRMED: Thm 4.3(a) unique KMS_{ln rho} on the Toeplitz algebra,
    phi(s_mu s_nu*) = delta_{mu,nu} rho^{-|mu|} x_{s(mu)}, factoring through C*(E);
    Thm 4.3(c) no KMS below ln rho; Remark 4.4 irreducibility crucial.
  * Matsumoto-Watatani-Yoshida, Math. Z. 228 (1998) 489-509 -- EXISTS, not load-bearing here
    (our case is an SFT, which reduces to EFW).

POISONS (binding): no goal-seek (r(B0) is computed from the imported engine matrix -- ln 2
falls out or it does not); the 2pi/ln2 pattern-match stays FORBIDDEN (this confront is
object-level: fixed-point equations, not numerology); the TWO-TEMPERATURES guard is held
(the run's operating alpha_1 and kappa's critical u_c are never conflated -- S4 maps each
separately); the Born-factor-2 layer of the run marginal is FENCED as framework-specific,
never silently identified with the power-1 Parry measure; NO existing file edited; ONE new
file; runtime seconds.
"""
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))

from derivation_topdown.bridge import the_run  # noqa: E402  the ENGINE (task: import, don't rebuild)
import srs                                     # noqa: E402  constants only (DEG); same walled-off module

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond
def banner(t):
    print("=" * 92); print(f" {t}"); print("=" * 92)

# ===========================================================================
banner("S0  the repo's objects, READ (constants + the engine matrix; citations in docstring)")
# ===========================================================================
k = srs.DEG                       # coordination (READ; = 3)
q = k - 1                         # NB branching per dart
u_c = 1.0 / q                     # path-gas critical fugacity  (T2b pole; T3-2 per-tick factor)
b_edge = math.log2(q)             # description cost per edge, bits (T2/T3: b_edge = log2(k-1))
alpha1 = (q / k) ** (10 - 2)      # the run's OPERATING fugacity (the_run girth-10 renewal; T1)
B0 = the_run.hashimoto((0, 0, 0)).real   # the geodesic generator at Gamma (the_run.py:46 -> srs.py:42)
ND = B0.shape[0]
Bi = np.rint(B0).astype(int)      # exact integer copy (entries verified {0,1} in S1)
A_sft = Bi.T                      # SFT transition matrix, row = source (srs.hashimoto: B[b,a] = a->b)
print(f"    k = srs.DEG = {k}; q = k-1 = {q}; u_c = 1/(k-1) = {u_c}; b_edge = log2(k-1) = {b_edge}")
print(f"    alpha_1 (run operating fugacity) = (2/3)^8 = {alpha1:.6f}")
print(f"    B0 = the_run.hashimoto(Gamma).real: {ND}x{ND} on darts (2|E| of the K4 cell)")
check("S0 engine matrix imported intact: B0 is exactly integer 0/1 (max |B0 - round(B0)| = 0)",
      float(np.max(np.abs(B0 - Bi))) < 1e-12 and set(Bi.flatten().tolist()) == {0, 1})

# ===========================================================================
banner("S1  EFW/CK hypotheses on the NB dart shift (the subshift IS the multiway path set)")
# ===========================================================================
# The one-sided SFT: symbols = the 12 darts; word d0 d1 ... d_{n-1} admissible iff every step
# d_i -> d_{i+1} is a legal NB step, i.e. A_sft[d_i, d_{i+1}] = 1. These words ARE the run's
# multiway histories (T2's path gas configurations) -- checked by enumeration in S2.
rows = A_sft @ np.ones(ND, dtype=int)
cols = A_sft.T @ np.ones(ND, dtype=int)
check("S1a A_sft is a 0-1 matrix, (k-1)-out-regular AND (k-1)-in-regular (all row/col sums = 2)",
      set(rows.tolist()) == {q} and set(cols.tolist()) == {q},
      detail=f"row sums {sorted(set(rows.tolist()))}, col sums {sorted(set(cols.tolist()))}")
check("S1b A_sft is NOT a permutation matrix (row sums = 2 != 1) -- the Cuntz-Krieger "
      "condition-(I)/simplicity hypothesis, RE-ATTRIBUTED: NOT an EFW-KMS hypothesis "
      "(accessible EFW recaps require only irreducible 0-1; uniqueness survives the "
      "permutation case at beta = 0, aHLRS Rem 4.5)",
      not np.array_equal(A_sft @ A_sft.T, np.eye(ND, dtype=int)))
# irreducibility: some power connects every ordered pair
R = np.zeros((ND, ND), dtype=int); M = np.eye(ND, dtype=int)
for _ in range(ND):
    M = M @ A_sft
    R += (M > 0).astype(int)
check("S1c A_sft IRREDUCIBLE (every dart reaches every dart within 12 steps)", (R > 0).all())
# primitivity (aperiodicity): minimal m with A^m entrywise positive
M = A_sft.copy(); m_prim = 1
while not (M > 0).all() and m_prim < 60:
    M = M @ A_sft; m_prim += 1
check("S1d A_sft PRIMITIVE (A^m > 0 entrywise; K4's 3- and 4-cycles force gcd of periods = 1)",
      (M > 0).all(), detail=f"minimal m = {m_prim}")
print("    => the EFW hypotheses (irreducible 0-1) HOLD for the engine's own matrix; 'not a permutation'")
print("       holds too but is the CK condition-(I)/simplicity hypothesis, NOT an EFW-KMS hypothesis (S1b).")

# ===========================================================================
banner("S2  spectral radius + topological entropy of the shift (matrix AND exact word counts)")
# ===========================================================================
ev = np.linalg.eigvals(B0)
rho = float(max(abs(ev)))
h_top = math.log(rho)
print(f"    r(B0) = {rho:.12f};  h_top = ln r = {h_top:.12f};  ln 2 = {math.log(2):.12f}")
check("S2a spectral radius r(B0) = k-1 = 2 (the NB growth rate; Perron)", abs(rho - q) < 1e-9)
check("S2b h_top(edge shift) = ln r(B0) = ln 2  (|h_top - ln 2| < 1e-9)",
      abs(h_top - math.log(2)) < 1e-9)
# exact word counts two ways: (i) integer matrix powers, (ii) explicit enumeration on the dart
# digraph -- their agreement PINS the convention and certifies "the SFT = the run's history set".
one = np.ones(ND, dtype=object)
Ao = A_sft.astype(object)
counts_matrix = []
P = np.eye(ND, dtype=object)
for n in range(1, 9):
    counts_matrix.append(int(one @ (P @ one)))     # words of n darts = 1^T A^{n-1} 1
    P = P @ Ao
succ = [np.nonzero(A_sft[a])[0].tolist() for a in range(ND)]
frontier = [(a,) for a in range(ND)]
counts_enum = [len(frontier)]
for n in range(2, 9):
    frontier = [w + (b,) for w in frontier for b in succ[w[-1]]]
    counts_enum.append(len(frontier))
expected = [ND * q ** (n - 1) for n in range(1, 9)]
print(f"    words of length n (matrix)      : {counts_matrix}")
print(f"    words of length n (enumerated)  : {counts_enum}")
print(f"    12*(k-1)^(n-1)                  : {expected}")
check("S2c word counts: matrix power == explicit enumeration == 12*(k-1)^(n-1) EXACT (n = 1..8)",
      counts_matrix == counts_enum == expected)
check("S2d entropy from counts: (1/n) ln W_n -> ln 2 (within 1/n prefactor drift at n = 8)",
      abs(math.log(counts_enum[-1]) / 8 - math.log(2)) < math.log(ND / q) / 8 + 1e-12,
      detail=f"(1/8) ln W_8 = {math.log(counts_enum[-1])/8:.6f}")
print("    => THE FIRST NUMBER OF THE CONFRONT: the NB shift's topological entropy IS ln 2 (nats/tick).")

# ===========================================================================
banner("S3  the CK-KMS fixed point (the computational core of EFW/Matsumoto), on OUR matrix")
# ===========================================================================
# A gauge-KMS_beta state on the graph algebra, restricted to the diagonal, must satisfy
#   psi(S_mu S_mu^*) = e^{-beta} * Sum_{d: mu d admissible} psi(S_{mu d} S_{mu d}^*)  ... (*)
# With the ansatz psi(S_mu S_mu^*) = c * e^{-beta |mu|} x_{last(mu)}, (*) reads
#   x_a = e^{-beta} Sum_b A_sft[a,b] x_b   <=>   A_sft x = e^{beta} x,   x > 0.
# Perron-Frobenius on the irreducible A_sft: a POSITIVE eigenvector exists ONLY for the Perron
# eigenvalue => e^{beta} = r(A) forced => beta = h_top = ln 2, x = the all-ones vector here.
evA, VA = np.linalg.eig(A_sft.astype(float))
positive_lams = []
for i in range(ND):
    v = VA[:, i]
    v = v / v[np.argmax(np.abs(v))]
    if np.all(np.abs(v.imag) < 1e-9) and np.all(v.real > 1e-9):
        positive_lams.append(float(evA[i].real))
print(f"    eigenvalues of A_sft admitting an entrywise-POSITIVE eigenvector: {positive_lams}")
check("S3a the fixed point (*) has a positive solution ONLY at e^beta = r(A) = 2 (Perron) "
      "=> beta = ln 2 FORCED", len(positive_lams) == 1 and abs(positive_lams[0] - q) < 1e-9)
check("S3b the Perron vector is the all-ones vector EXACTLY (A_sft.1 = 2.1; out-regularity)",
      np.array_equal(A_sft @ np.ones(ND, dtype=int), q * np.ones(ND, dtype=int)))
# Shell-mass trichotomy: with x = 1, the total mass of the length-n diagonal projections is
#   M_n(beta) = W_n e^{-beta n} c,  ratio M_{n+1}/M_n = (k-1) e^{-beta} = u/u_c  with u := e^{-beta}.
# In O_A, Sum_{|mu|=n} S_mu S_mu^* = 1 for EVERY n  => a state needs M_n CONSTANT => beta = h_top.
# On the Toeplitz extension, Sum_{|mu|=n} S_mu S_mu^* = 1 - (paths that died into the vacuum),
# so M_n may DECREASE (beta > h_top: the deficit is the vacuum/seed weight) but never increase.
print("    shell-mass ratio M_{n+1}/M_n = (k-1) e^{-beta} = u/u_c  (u := e^{-beta}):")
tri_ok = True
for lbl, beta in [("beta = ln 2       (CK point)     ", math.log(2)),
                  ("beta = -ln alpha_1 (run's gas pt) ", -math.log(alpha1)),
                  ("beta = 0.40       (< h_top)      ", 0.40)]:
    ratio = q * math.exp(-beta)
    regime = ("CONSTANT -> state on O_A (unique CK point)" if abs(ratio - 1) < 1e-12 else
              "DECREASING -> Toeplitz-only KMS (vacuum defect > 0)" if ratio < 1 else
              "INCREASING -> NO state anywhere (supercritical)")
    tri_ok &= abs(ratio - (math.exp(-beta) / u_c)) < 1e-12
    print(f"      {lbl}: ratio = {ratio:.6f}  => {regime}")
check("S3c the trichotomy variable IS the framework's u/u_c (ratio = e^{-beta}/u_c exactly): "
      "state-on-O_A <=> u = u_c; Toeplitz-KMS <=> u < u_c (T2's ARROW); none <=> u > u_c", tri_ok)
check("S3d the CK point's per-symbol fugacity e^{-h_top} = 1/(k-1) = u_c = 2^{-b_edge} "
      "(= T3-2's per-tick Boltzmann factor, EXACT)",
      abs(math.exp(-h_top) - u_c) < 1e-12 and abs(u_c - 2 ** (-b_edge)) < 1e-15)
print("    => the unique-KMS fixed-point equation and T2b's pole of Z(u) = 1/(1-(k-1)u) are the")
print("       SAME Perron condition; the CK critical state's fugacity IS the framework's u_c.")

# ===========================================================================
banner("S4  the temperature dictionary: beta_CK = h_top (per tick)  <=>  beta*kappa = ln 2 (per bit)")
# ===========================================================================
# THEIR flow: the gauge action gamma_z(S_d) = z S_d; generator = the integer grading by word
# length = the tick count N-hat (T1's FLOW-ID identifies the run's modular flow with exactly this
# grading; T4b's 2pi = the same integrality that makes gamma a circle action). Per-tick energy 1
# => the unique KMS point sits at beta_CK = h_top = ln(k-1).
# OUR Hamiltonian: E = kappa*L with L = b_edge*N (bits) => per-tick Gibbs exponent beta*kappa*b_edge.
# EQUAL per-tick weights  <=>  beta*kappa*b_edge = h_top  <=>  beta*kappa = h_top/b_edge
#                          =  ln(k-1) / log2(k-1)  =  ln 2   IDENTICALLY, for every k >= 3.
print("    beta*kappa = h_top/b_edge = ln(k-1)/log2(k-1), evaluated for k = 3..12:")
dict_ok = True
for kk in range(3, 13):
    qq = kk - 1
    val = math.log(qq) / math.log2(qq) if qq > 1 else float("nan")
    dict_ok &= abs(val - math.log(2)) < 1e-14
    print(f"      k = {kk:2d}:  h_top = ln {qq} = {math.log(qq):.6f} nats/tick;  b_edge = {math.log2(qq):.6f} "
          f"bits/tick;  h_top/b_edge = {val:.15f}")
check("S4a beta*kappa = h_top/b_edge = ln 2 for EVERY k (the Landauer lock is the k-INDEPENDENT, "
      "per-bit form of 'unique KMS at topological entropy')", dict_ok)
check("S4b at the repo's k = 3 the two ln 2's coincide NUMERICALLY because b_edge = 1 bit/tick "
      "exactly (h_top = ln 2 nats/tick AND beta*kappa = ln 2 per bit)",
      abs(b_edge - 1.0) < 1e-15 and abs(h_top - math.log(2)) < 1e-9)
# TWO-TEMPERATURES guard, mapped: the run's OPERATING point is u = alpha_1 < u_c, i.e. gas-level
# beta_gas = -ln(alpha_1) > h_top -- strictly INSIDE the Toeplitz existence half-line, NOT at the
# CK point. kappa's currency temperature sits AT the CK point u_c. No conflation.
beta_gas = -math.log(alpha1)
beta_eff = 2.0 * math.log(u_c / alpha1)
check("S4c two-temperatures guard mapped: run operating beta_gas = -ln(alpha_1) = 8 ln(3/2) > h_top "
      "(Toeplitz interior); kappa's point = the CK boundary beta = h_top",
      beta_gas > h_top + 0.1, detail=f"beta_gas = {beta_gas:.6f} vs h_top = {h_top:.6f}")
check("S4d [DICTIONARY-ONLY] the Born-2 fence: beta_eff = 2*(beta_gas - h_top) is DEFINITIONALLY "
      "EXACT -- a rewrite of thermal_time.py:209's own definition beta_eff = 2 ln(u_c/alpha_1) "
      "using h_top = -ln u_c; NEVER an independent corroboration of G5a (5.1011473686 recovered "
      "by construction). The run MARGINAL is the amplitude-SQUARED object, measured from the CK "
      "point; a DISTINCT, framework-specific layer",
      abs(beta_eff - 2.0 * (beta_gas - h_top)) < 1e-12 and abs(beta_eff - 5.1011473686) < 1e-9,
      detail=f"beta_eff = {beta_eff:.10f}")

# ===========================================================================
banner("S5  OBJECT-MAPPING TABLE (their object <-> repo object <-> adjudication)")
# ===========================================================================
table = [
    ("subshift of finite type Sigma_A (symbols,\n     admissible words; A irreducible 0-1)",
     "the NB dart histories of the srs cell = the multiway\n     path gas (T2); 12 darts, A_sft = B0^T "
     "(srs.py:42, via\n     the_run.py:46)",
     "SAME (S1+S2c: hypotheses hold; enumerated words\n     == matrix counts == 12*2^(n-1))"),
    ("gauge action gamma_z(S_d) = z S_d; generator =\n     integer grading by word length; 2pi-periodic",
     "the tick flow e^{-i theta N-hat} (T1 FLOW-ID: modular\n     flow = tick flow; T4b: N-hat integer "
     "=> minimal period\n     2pi, M0_2R_T4:79-100)",
     "SAME GENERATOR (grading = N-hat); THEIR gauge\n     circle's 2pi = T4b's circle (free corroboration)"),
    ("unique KMS_beta at beta = log r(A) = h_top\n     (EFW/Matsumoto; per-tick energy 1)",
     "T3-1 Landauer lock beta*kappa = ln 2 with H = kappa*\n     b_edge*N-hat "
     "(M0_2R_T2_T3:224-237)",
     "SAME TEMPERATURE, different UNITS: beta*kappa =\n     h_top/b_edge = ln 2 for ALL k (S4a, machine-checked)"),
    ("the KMS state's diagonal = Parry measure;\n     per-symbol fugacity e^{-h_top} = 1/r(A)",
     "u_c = 1/(k-1) = 2^{-b_edge} = T3-2's per-tick factor\n     (M0_2R_T2_T3:239-250); T2b's pole of Z(u)",
     "SAME NUMBER + SAME fixed-point equation (S3a/S3d)"),
    ("Toeplitz-side KMS geography (aHLRS arXiv:1205.2194\n     Thms 3.1 + 4.3, built on Laca-Neshveyev, JFA 211\n"
     "     (2004) 457-482): states iff beta >= h_top; simplex\n     above, unique at h_top factoring through O_A",
     "T2c ARROW = sub-criticality u < u_c (existence of the\n     convergent run; seed-dependent transients, "
     "M0_2R_T2_T3:\n     181-199); criticality = the seed-free Perron point",
     "SAME GEOGRAPHY, VERIFIED (S3c trichotomy; aHLRS\n     Thm 4.3(a): unique KMS_{ln rho} on the Toeplitz "
     "algebra,\n     phi(s_mu s_nu*) = delta_{mu,nu} rho^{-|mu|} x_{s(mu)}, factoring\n     through C*(E); "
     "4.3(c): no KMS below ln rho; Rem 4.4:\n     irreducibility crucial)"),
    ("the full C*-algebra O_A (S_d partial isometries,\n     Sum S_d S_d^* = 1)",
     "NOT CONSTRUCTED as such: H_hist carries a Fock-type\n     rep WITH a seed/vacuum "
     "(M0_2R_T1:16-19) => the\n     TOEPLITZ extension (Sum S_d S_d^* = 1 - P_vac), not O_A",
     "UNDETERMINED-AS-BUILT: the ONE named missing\n     construction (represent S_d on H_hist; check the\n"
     "     defect; expected resolution = Toeplitz, which\n     STRENGTHENS the geography row)"),
    ("(no analogue: their state is the power-1\n     path measure)",
     "the run MARGINAL p_n ~ u^{2n}||B^n seed||^2 (Born\n     factor 2, M0_2R_T1:83-98) at operating "
     "alpha_1;\n     beta_eff = 2*(beta_gas - h_top) = 5.1011473686\n     (S4d, DICTIONARY-ONLY)",
     "FRAMEWORK-SPECIFIC (amplitude-squared layer;\n     NOT covered by the import; fence held)"),
    ("(dimensionless theory: beta in units of the\n     gauge generator)",
     "kappa's MAGNITUDE = h/t_P (T4: the 2pi from tick\n     integrality; t_P the dimensional anchor); "
     "the currency\n     identification E = kappa*L itself",
     "FRAMEWORK-SPECIFIC (the import ratifies only the\n     dimensionless lock beta*kappa = ln 2)"),
]
for i, (theirs, ours, status) in enumerate(table, 1):
    print(f"  [{i}] THEIRS: {theirs}")
    print(f"      OURS  : {ours}")
    print(f"      STATUS: {status}")
    print()

# ===========================================================================
banner("S6  VERDICT (printed prose; the machine checks above are the evidence)")
# ===========================================================================
print("""    VERDICT: SAME-OBJECT (QUALIFIED) -- RATIFICATION-BY-IMPORT at the critical point
    (full-algebra construction pending -- SAME-OBJECT booked for the subshift,
    grading/generator, critical temperature, and the critical state's diagonal);
    the literature list is now VERIFIED (header block).

    Adjudicated per the frozen criteria, row by row:

    (i) SUBSHIFT + FLOW GENERATOR + CRITICAL TEMPERATURE + CRITICAL STATE: SAME-OBJECT,
        machine-checked here. The NB dart shift of the engine's own matrix satisfies the
        EFW hypotheses (S1); its topological entropy is ln 2 (S2); the gauge-KMS fixed-point
        equation is the SAME Perron condition as T2b's pole of Z(u), forcing beta = ln 2
        with the critical fugacity u_c = e^{-h_top} = T3-2's per-tick factor (S3); and the
        temperature dictionary is an IDENTITY, not a fit: beta*kappa = h_top/b_edge = ln 2
        for every k >= 3 (S4a). The Landauer lock beta*kappa = ln 2 IS the graph-algebra
        statement 'unique KMS at beta = topological entropy', written in per-bit units.
        The two ln 2's -- h_top = ln(k-1) nats/tick (k-dependent) and beta*kappa = ln 2 per
        bit (k-independent) -- coincide numerically at k = 3 because b_edge = 1 exactly;
        the identity h_top/b_edge = ln 2 disambiguates them and shows the coincidence is
        unit-mediated, NOT numerology (S4b). This is the answer to the confront question:
        our ln 2 IS the log of the NB-walk's growth rate, expressed per bit.

    (ii) THE FULL C*-ALGEBRA: UNDETERMINED-AS-BUILT, with the missing piece NAMED. The
        framework's history space carries a Fock-type representation WITH a seed (vacuum);
        that is the TOEPLITZ extension of the Cuntz-Krieger algebra, not O_A itself. The
        clean import for the framework as built is therefore the Toeplitz-side KMS
        classification, VERIFIED as an Huef-Laca-Raeburn-Sims, arXiv:1205.2194, Thms 3.1 +
        4.3 (built on Laca-Neshveyev, JFA 211 (2004) 457-482): KMS states exist exactly on
        beta >= h_top (a simplex above); Thm 4.3(a): the UNIQUE KMS_{ln rho} state on the
        Toeplitz algebra, phi(s_mu s_nu*) = delta_{mu,nu} rho^{-|mu|} x_{s(mu)}, factors
        through C*(E); Thm 4.3(c): no KMS below ln rho; Rem 4.4: irreducibility crucial.
        That does not merely ratify -- it gives the framework's OWN two-temperature
        geography an external theorem: T2's arrow (= sub-criticality u < u_c) IS the
        KMS-existence half-line, and T3's Landauer = criticality IS the unique CK boundary
        point (S3c exhibits the trichotomy on our matrix, with u/u_c as the exact
        trichotomy variable).
        COMPLETION STEP (one station, not run here; DESIGN REQUIREMENTS per the adversarial
        check): represent the dart partial isometries S_d on H_hist and machine-check
        Sum S_d S_d^* = 1 - P_seed (Toeplitz-CK relations). THEN respect the proven trap:
        the run vector |G> = Sum_n u^n B^n |seed> has NONZERO off-degree correlations, so
        the run state is NOT gauge-invariant on the dart algebra -- and every KMS_beta
        state with beta != 0 is flow-invariant (its off-degree two-point functions vanish),
        so a NAIVE two-point KMS check on words S_mu S_nu^* will FALSE-FAIL. The station
        must therefore (a) gauge-average the run state / restrict to the diagonal
        (|mu| = |nu|) FIRST, and (b) pre-register the tested dart-algebra temperature:
        candidate beta' = beta_eff + h_top = 2*beta_gas - h_top ~= 5.7943 -- NOT beta_eff
        (a ladder-algebra number, the N-hat marginal's) and NOT beta_gas (the power-1 gas
        exponent). (This extends G5a's KMS-4 beyond the abelian N-hat subalgebra.)

    (iii) FRAMEWORK-SPECIFIC RESIDUES (never importable, correctly so): kappa's MAGNITUDE
        kappa = h/t_P (T4's 2pi + the t_P anchor); the currency identification E = kappa*L
        (that a description length IS an energy -- the import makes the ln 2 in Landauer's
        k_B T ln 2 the shift's entropy, but the identification premise stays the
        framework's adopted axiom, exactly as T3's adoption-downgrade analysis already
        held); and the Born-factor-2 run-marginal layer (p_n ~ u^{2n}: the amplitude-
        squared state at the operating point alpha_1, whose beta_eff = 2*(beta_gas -
        h_top) = 5.1011473686 measures Born-squared distance from the CK point -- S4d,
        DICTIONARY-ONLY: definitionally exact, a rewrite of thermal_time.py:209's own
        definition via h_top = -ln u_c, never an independent corroboration of G5a).

    WHAT THE IMPORT ADDS NOW THAT THE LITERATURE IS VERIFIED (the value of the
    ratification; contingent only on the (ii) construction landing):
      * UNIQUENESS at criticality: T3 proved consistency FORCES the point; EFW (verified)
        adds that on the full noncommutative dart algebra the point is the ONLY KMS point
        of the tick/gauge flow -- upgrading 'Landauer = criticality' from a consistency to
        a uniqueness theorem.
      * The measure-of-maximal-entropy identification (Parry): the critical path gas is
        the MME of the NB shift -- an information-theoretic characterization the currency
        layer can cite instead of re-derive.
      * An external name for the two-temperature geography (Toeplitz half-line vs CK
        point), retiring it as a framework idiosyncrasy.

    LITERATURE STATUS (the VERIFY-BEFORE-IMPORT list, now ADJUDICATED by the adversarial
    checker; sources in the header LITERATURE VERIFICATION block):
      1. VERIFIED  Olesen-Pedersen, Math. Scand. 42 (1978) 111-118: O_n gauge action,
         unique KMS at beta = log n.
      2. VERIFIED  Enomoto-Fujii-Watatani, Math. Japon. 29(4) (1984) 607-619: A irreducible
         0-1, unique gauge-KMS at beta = ln rho(A); KMS states <-> normalized positive
         Perron eigenvectors. The 'not a permutation' clause is RE-ATTRIBUTED to CK
         condition-(I)/simplicity (not an EFW-KMS hypothesis); uniqueness survives the
         permutation case at beta = 0 (aHLRS Rem 4.5).
      3. VERIFIED (exists; not load-bearing)  Matsumoto-Watatani-Yoshida, Math. Z. 228
         (1998) 489-509: our case is an SFT, which reduces to EFW.
      4. VERIFIED  an Huef-Laca-Raeburn-Sims, 'KMS states on the C*-algebras of finite
         graphs', arXiv:1205.2194, Thms 3.1 + 4.3 (built on Laca-Neshveyev, J. Funct.
         Anal. 211 (2004) 457-482): the TOEPLITZ-side classification -- the statement the
         framework as built actually instantiates; its hypotheses gate the (ii)
         completion step.
      5. OPEN (textbook-standard)  Parry (1964): h_top = log r(A), MME uniqueness -- not
         re-verified by the checker; standard symbolic-dynamics material.

    STATUS: ASSESSMENT ONLY. No scoreboard row moved; no register edited; the adoption-
    downgrade opportunity (T3's ln 2 gains an external uniqueness theorem; the currency
    premise E = kappa*L stays adopted) is REPORTED for the architect to book or reject
    after the (ii) completion station (the literature verification is DONE -- header block).""")

# the S6(ii) pre-registered dart-algebra candidate temperature, machine-pinned (design req. (b))
beta_prime = beta_eff + h_top
check("S6a pre-registered dart-algebra temperature: beta' = beta_eff + h_top = 2*beta_gas - h_top "
      "(identity, machine-pinned; the (ii) station tests THIS, not beta_eff and not beta_gas)",
      abs(beta_prime - (2.0 * beta_gas - h_top)) < 1e-12,
      detail=f"beta' = {beta_prime:.10f} (~= 5.7943)")

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
print(f"""    Machine-checked: EFW hypothesis (irreducible 0-1) + primitivity (m={m_prim}) + the SEPARATE
      CK condition-(I)/simplicity check (not a permutation; re-attributed, S1b) on the engine
      matrix; r(B0) = {rho:.9f} = k-1; h_top = ln 2 (dev {abs(h_top-math.log(2)):.1e});
      words == enumeration == 12*2^(n-1) (n<=8); positive-eigenvector uniqueness => the CK-KMS
      fixed point sits at beta = ln 2 ONLY; per-symbol fugacity at it = u_c = {u_c}; the shell-mass
      trichotomy variable = u/u_c exactly; beta*kappa = h_top/b_edge = ln 2 for k = 3..12 (1e-14);
      beta_eff = 2*(beta_gas - h_top) = {beta_eff:.10f} (DICTIONARY-ONLY: a definitional rewrite
      of thermal_time.py:209 via h_top = -ln u_c, never an independent corroboration of G5a; the
      Born-2 fence); pre-registered dart-algebra beta' = {beta_prime:.10f} (S6a).
    VERDICT: SAME-OBJECT (QUALIFIED) -- the Landauer lock is the per-bit form of unique-KMS-at-
      topological-entropy (full-algebra construction pending -- SAME-OBJECT booked for the
      subshift, grading/generator, critical temperature, and the critical state's diagonal).
      Literature VERIFIED (EFW + Olesen-Pedersen + aHLRS arXiv:1205.2194 Thms 3.1/4.3; header
      block); ratification-by-import now CONDITIONAL only on the named Toeplitz/CK construction
      on H_hist (S6(ii) design requirements: gauge-average/diagonal first; test beta').
      DIFFERENT-OBJECT is REJECTED (no unremovable mismatch found); full SAME-OBJECT is NOT
      claimed (algebra row undetermined).""")
print("RESULT:", "ALL MACHINE CHECKS PASS -- I-2 CONFRONT COMPLETE: SAME-OBJECT (QUALIFIED) "
      "(full-algebra construction pending -- SAME-OBJECT booked for the subshift, "
      "grading/generator, critical temperature, and the critical state's diagonal)"
      if ok_all else "A MACHINE CHECK FAILED -- verdict void")
sys.exit(0 if ok_all else 1)
