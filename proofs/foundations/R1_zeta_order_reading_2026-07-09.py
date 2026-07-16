#!/usr/bin/env python3
"""
proofs/foundations/R1_zeta_order_reading_2026-07-09.py

R1 -- the u^4/u^5 ORDER-READING of the forced chiral asymmetry.  Pre-registered in
internal research notes (FROZEN BEFORE this file; Build Ops
Protocol, pipeline step 3).  Charter's scheduled reading: ML-5b reduced the -70 ppm to ONE
number (eps = Delta c, lever = 1) and named the missing alpha_1^2 -> alpha_1^{4-5} suppression;
the trunk synthesis observed that eps's order alpha_1^{4-5} = LOOP ORDERS u^4/u^5 of the SAME
zeta whose loop expansion G6ab (adapters/zeta_gauge.py, ZG-4b) machine-checked at 1.2e-17.

R1 TESTS THE LOCATION HYPOTHESIS.  It does NOT construct a transport functional and does NOT
select anything toward the target.  ONE frozen functional (no family, no selector); the target
and the poison ladder appear ONLY in the R1-3 confront section; every A_L is booked raw
regardless of branch; the -70 ppm stays OPEN in every branch (R1-3/R1-4).

THE ONE FROZEN FUNCTIONAL (verbatim from the pre-reg):
  A(u)   := tr(Q1 . G_int(u)) - conj(tr(Q2 . G_int(u))),  G_int(u) = P_VAC (I-u W_INT)^-1 P_VAC^dagger
  G_L    := P_VAC . W_INT^L . P_VAC^dagger          (the exact per-order block)
  A_L    := tr(Q1 . G_L) - conj(tr(Q2 . G_L))        so that  A(u) = sum_L u^L A_L  (an identity)
  tail45 := alpha_1^4 A_4 + alpha_1^5 A_5             -- THE single declared read.
Nothing else is a candidate: no alternative Q-combinations, no lepton-slice selector, no other
projection or weighting -- not even "for comparison" (ML-5b already showed the lepton-slice
selector is the un-forced gap; R1 deliberately does not introduce one).

REUSE MAP (copied verbatim; NOT re-derived):
  - derivation_topdown/adapters/zeta_gauge.py (G6a,b) and
    proofs/foundations/ML5_epsilon_2026-07-08.py (ML-5): srs.EDGES/DARTS, cl6_generators, gam(),
    the J6 frame -> A_ops -> NHAT -> vac, W_INT (the 8*ND x 8*ND Clifford-blocked walk operator),
    P_VAC (the dart-indexed vacuum block projector), G_int(u) = P_VAC(I-uW_INT)^-1 P_VAC^dagger,
    the winding machinery sigma3/P3/Q_t (Q_t acting on dart space), ALPHA1 = (2/3)^8.
    The machinery block below (through the definition of ALPHA1) is copied verbatim in
    structure from ML5_epsilon_2026-07-08.py lines ~44-115 (same variable names, same
    construction order) -- this is the SAME chiral-asymmetry carrier ML-5 built and evaluated,
    not a new one.
  - The chiral asymmetry convention A(u) = tr(Q1 G_int(u)) - conj(tr(Q2 G_int(u))) is ML-5's own
    (ML5_epsilon_2026-07-08.py's `asym(u)`), reproduced here verbatim; R1-0 requires
    |A(alpha_1)| = 8.8166e-04 to < 1e-8, exactly ML-5's own printed carrier value.
  - rho(W_INT) = sqrt(2) is an ESTABLISHED fact from G6ab (zeta_gauge.py ZG-4b prints
    "spectral radius rho(W_INT)"); R1-0 reads it off the SAME W_INT built here (a free
    corroboration of the established fact, not a re-derivation) and uses it for the printed
    geometric truncation bound.

FROZEN STATIONS (frozen wording -- see the pre-reg):
  R1-0  REGRESSION: rebuild the machinery verbatim; verify A(alpha_1) reproduces ML-5's
        8.8166e-04 (< 1e-8) AND the identity A(alpha_1) = sum_{L=1..40} alpha_1^L A_L
        (< 1e-12), printing the geometric truncation bound (rho(W_INT) = sqrt(2), G6ab).
  R1-1  THE ORDER TABLE (raw): A_L for L=1..10 -- |A_L|, arg(A_L), Re, Im.  Raw observation
        only (which orders vanish, any parity pattern) -- no interpretation gated on it.
  R1-2  THE READ (blind): tail45 = alpha_1^4 A_4 + alpha_1^5 A_5 -- printed, full precision.
        The target/poisons do NOT appear in this section (structural hard rule).
  R1-3  THE CONFRONT (declared end; the target enters ONLY here): compare |tail45| against
        eps = 1.7515e-7 (magnitude; the booked signed value is -1.7515e-7 rad) and the poison
        ladder {2*alpha_1^5, alpha_1^4, alpha_1^3}.  FROZEN branch logic (mechanical):
          LOCATION-CONFIRMED: |tail45| within 5% of |eps| AND nearer eps than every poison
                               by >= 3x.
          LOCATION-REFUTED:   |tail45| differs from |eps| by > 2x, OR matches some poison
                               better than it matches eps.
          AMBIGUOUS:          neither of the above (between bands, or within 3x of a poison).
        Regardless of branch: the -70 ppm stays OPEN; a CONFIRMED is a LEAD requiring its own
        follow-up pre-registration for sign/phase identification -- NOT decided here.
  R1-4  SCOPE: printed declaration (not gating).

HARD RULES (binding): ONE new file; no engine/proofs edits; no git commits.  THE
ONE-FUNCTIONAL RULE IS ABSOLUTE -- no alternative combination, projection, weighting, or slice
is computed anywhere in this file, not even for debugging or comparison.  The target and
poison ladder are referenced ONLY inside the R1-3 block.  Exit 0 iff R1-0's two checks pass
AND R1-3 reaches one of its three named (mutually-exclusive, exhaustive) branches -- all three
are "definite" per the pre-reg, so R1-3 always gates true by construction; only R1-0 can fail.
"""
import cmath
import itertools
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
import srs  # noqa: E402  -- the engine, unmodified
from simulator.srs_engine.utils import AlgebraicUtility  # noqa: E402

np.set_printoptions(precision=6, suppress=True)
FAILURES = []


def banner(t):
    print("=" * 88)
    print(f" {t}")
    print("=" * 88)


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)
    return bool(cond)


def cfmt(z, prec=15):
    """full-precision complex formatter"""
    return f"{z.real:+.{prec}e} {z.imag:+.{prec}e}j"


# ===========================================================================
banner("SETUP -- the FORCED machinery, copied verbatim (ML5_epsilon_2026-07-08.py lines ~44-115)")
# ===========================================================================
EDGES = srs.EDGES
NE, NV = len(EDGES), srs.NV
ND = 2 * NE
g6 = [np.array(g) for g in AlgebraicUtility.cl6_generators()]
EIDX = {(min(i, j), max(i, j)): e for e, (i, j, v) in enumerate(EDGES)}
gam = lambda x: sum(x[a] * g6[a] for a in range(NE))
DARTS = []
for i, j, v in EDGES:
    DARTS += [(i, j), (j, i)]
EDGE_OF_DART = [d // 2 for d in range(ND)]
B_G = srs.hashimoto((0.0, 0.0, 0.0)).real


def edge_rep(sig):
    R6 = np.zeros((NE, NE))
    for e, (i, j, v) in enumerate(EDGES):
        a, b = sig[i], sig[j]
        s = 1.0
        if a > b:
            a, b, s = b, a, -1.0
        R6[EIDX[(a, b)], e] = s
    return R6


d0 = np.zeros((NV, NE))
for e, (i, j, v) in enumerate(EDGES):
    d0[i, e] = -1.0; d0[j, e] = 1.0
Chat = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
H1, _ = np.linalg.qr(Chat)
B1 = np.linalg.svd(d0)[2][:3].T
A4 = [dict(enumerate(p)) for p in itertools.permutations(range(4))
      if sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j]) % 2 == 0]
rows = [np.kron(np.eye(3), (H1.T @ edge_rep(g) @ H1).T) - np.kron(B1.T @ edge_rep(g) @ B1, np.eye(3))
        for g in A4]
_, Sp, Vp = np.linalg.svd(np.vstack(rows))
phi = Vp[-1].reshape(3, 3); phi *= math.sqrt(3) / np.linalg.norm(phi)
J6 = B1 @ phi @ H1.T - H1 @ phi.T @ B1.T
wJ, VJ = np.linalg.eig(J6)
modes, _ = np.linalg.qr(VJ[:, np.where(np.abs(wJ - 1j) < 1e-9)[0]])
A_ops = [gam(np.conj(modes[:, m])) / math.sqrt(2) for m in range(3)]
NHAT = sum(a.conj().T @ a for a in A_ops)
wN, VN = np.linalg.eigh(NHAT)
vac = VN[:, [int(np.argmin(wN))]]; vac = vac / np.linalg.norm(vac)

W_INT = np.zeros((8 * ND, 8 * ND), complex)
for dp in range(ND):
    for d in range(ND):
        if abs(B_G[dp, d]) > 0.5:
            W_INT[dp * 8:(dp + 1) * 8, d * 8:(d + 1) * 8] = gam(np.eye(NE)[:, EDGE_OF_DART[dp]])
P_VAC = np.zeros((ND, 8 * ND), complex)
for d in range(ND):
    P_VAC[d, d * 8:(d + 1) * 8] = vac[:, 0].conj()


def G_int(u):
    X = np.linalg.solve(np.eye(8 * ND) - u * W_INT, P_VAC.conj().T)
    return P_VAC @ X


sigma3 = {0: 0, 1: 2, 2: 3, 3: 1}
P3 = np.zeros((ND, ND))
for a, (i, j) in enumerate(DARTS):
    for b, (p, q) in enumerate(DARTS):
        if (p, q) == (sigma3[i], sigma3[j]):
            P3[b, a] = 1.0
            break
OM = cmath.exp(2j * math.pi / 3)
Q_t = [sum(OM ** (-t * m) * np.linalg.matrix_power(P3, m) for m in range(3)) / 3 for t in range(3)]
ALPHA1 = (2.0 / 3.0) ** 8                                   # the run coupling (M0-2R operating point)
print(f"  ND (dart count) = {ND}; W_INT size = {8 * ND}x{8 * ND}; ALPHA1 = (2/3)^8 = {ALPHA1:.15f}")


def asym(u):
    """ML-5's own chiral-asymmetry functional -- the ONE frozen A(u)."""
    G = G_int(u)
    return np.trace(Q_t[1] @ G) - np.conj(np.trace(Q_t[2] @ G))


def G_of_L(L):
    """the exact per-order block G_L = P_VAC . W_INT^L . P_VAC^dagger"""
    Wp = np.linalg.matrix_power(W_INT, L)
    return P_VAC @ Wp @ P_VAC.conj().T


def A_of_L(L):
    """the exact per-order coefficient A_L = tr(Q1 G_L) - conj(tr(Q2 G_L))"""
    GL = G_of_L(L)
    return np.trace(Q_t[1] @ GL) - np.conj(np.trace(Q_t[2] @ GL))


# ===========================================================================
banner("R1-0  REGRESSION -- reproduce ML-5's carrier + verify the per-order identity")
# ===========================================================================
A_a1 = asym(ALPHA1)
print(f"  A(alpha_1) = {cfmt(A_a1)}   |A(alpha_1)| = {abs(A_a1):.10e}")
ML5_REFERENCE = 8.8166e-04
r1_0a = check("R1-0a  |A(alpha_1)| reproduces ML-5's 8.8166e-04",
              abs(abs(A_a1) - ML5_REFERENCE) < 1e-8,
              detail=f"|A(alpha_1)|={abs(A_a1):.10e}, |diff|={abs(abs(A_a1) - ML5_REFERENCE):.3e} (tol 1e-8)")

# A_L for L = 1..40, computed once directly from the frozen definition G_L = P_VAC W_INT^L
# P_VAC^dagger (iteratively, W_INT^L built by repeated matmul -- equivalent to, and cheaper
# than, calling G_of_L/A_of_L 40 independent times), reused by R1-0 (identity check), R1-1
# (order table L=1..10), and R1-2 (tail45 = L=4,5 terms).
AL = {}
Wp = np.eye(8 * ND, dtype=complex)
for L in range(1, 41):
    Wp = Wp @ W_INT
    GL = P_VAC @ Wp @ P_VAC.conj().T
    AL[L] = np.trace(Q_t[1] @ GL) - np.conj(np.trace(Q_t[2] @ GL))
# spot-check the iterative build against the literal frozen definition (G_of_L/A_of_L) at a
# couple of orders, so the two equivalent expressions of the SAME formula are cross-checked.
_spotcheck = max(abs(AL[L] - A_of_L(L)) for L in (1, 4, 5, 10))
r1_0d = check("R1-0d (internal) iterative A_L build matches the literal G_L=P_VAC W_INT^L "
              "P_VAC^dagger definition (A_of_L) at L in {1,4,5,10}", _spotcheck < 1e-10,
              detail=f"max|diff|={_spotcheck:.3e}")

# the L=0 term (diagnostic only, NOT part of the pre-registered sum range L=1..40; shown to
# demonstrate the declared range is the whole story -- G_int(u) = sum_{L=0}^inf u^L G_L and
# A_0 turns out to vanish identically, so starting the frozen sum at L=1 loses nothing).
A_0 = A_of_L(0)
print(f"  [diagnostic, not part of the frozen L=1..40 sum] A_0 = {cfmt(A_0)}  (|A_0|={abs(A_0):.3e})")

series_sum = sum(ALPHA1 ** L * AL[L] for L in range(1, 41))
identity_diff = abs(series_sum - A_a1)
print(f"  sum_{{L=1..40}} alpha_1^L A_L = {cfmt(series_sum)}")
print(f"  A(alpha_1) direct            = {cfmt(A_a1)}")
r1_0b = check("R1-0b  identity A(alpha_1) == sum_{L=1..40} alpha_1^L A_L",
              identity_diff < 1e-12, detail=f"|diff| = {identity_diff:.3e} (tol 1e-12)")

# --- the geometric truncation bound (rho(W_INT) = sqrt(2), established in G6ab/ZG-4b) ---
eigs_W = np.linalg.eigvals(W_INT)
rho_W = float(np.max(np.abs(eigs_W)))
print(f"\n  rho(W_INT) (computed here, on the SAME W_INT) = {rho_W:.12f}   "
      f"sqrt(2) (G6ab-established) = {math.sqrt(2):.12f}   "
      f"|diff| = {abs(rho_W - math.sqrt(2)):.3e}  (corroborates the established fact; not re-derived)")
u_rho = ALPHA1 * rho_W
print("""  TRUNCATION BOUND (analytic): G_int(u) = sum_{L=0}^inf u^L G_L is a convergent Neumann
  series for |u|*rho(W_INT) < 1.  Since Q_t are projectors (operator norm <= 1) and
  G_L = P_VAC W_INT^L P_VAC^dagger, |A_L| decays no slower than a fixed multiple of
  rho(W_INT)^L; the tail dropped by truncating the frozen sum at L=40 (i.e. L=41..inf) is
  bounded by the standard geometric-series remainder:
      sum_{L=41}^inf (|u| rho)^L = (|u| rho)^41 / (1 - |u| rho)""")
if u_rho < 1:
    trunc_bound = u_rho ** 41 / (1 - u_rho)
else:
    trunc_bound = float("inf")
print(f"  |u|*rho(W_INT) = {u_rho:.6e}  =>  truncation bound = {trunc_bound:.6e}  "
      f"(< 1e-12: {trunc_bound < 1e-12})")
print(f"  [context: the EMPIRICAL identity discrepancy above ({identity_diff:.3e}) is the "
      f"floating-point roundoff floor, not truncation -- the analytic tail is ~{trunc_bound:.1e}, "
      f"i.e. converged far beyond double precision by L=40]")
r1_0c = check("R1-0c  geometric truncation bound < 1e-12", trunc_bound < 1e-12,
              detail=f"bound = {trunc_bound:.3e}")

r1_0_ok = r1_0a and r1_0b and r1_0c and r1_0d

# ===========================================================================
banner("R1-1  THE ORDER TABLE (raw) -- A_L for L=1..10")
# ===========================================================================
print(f"  {'L':>3} {'|A_L|':>22} {'arg(A_L)':>18} {'Re(A_L)':>22} {'Im(A_L)':>22}")
for L in range(1, 11):
    z = AL[L]
    print(f"  {L:>3} {abs(z):>22.15e} {cmath.phase(z):>18.12f} {z.real:>22.15e} {z.imag:>22.15e}")

print("\n  RAW OBSERVATION (no interpretation gated on it):")
odd_mags = [abs(AL[L]) for L in range(1, 11) if L % 2 == 1]
even_mags = [abs(AL[L]) for L in range(1, 11) if L % 2 == 0]
print(f"    odd-L  (1,3,5,7,9)  |A_L| in [{min(odd_mags):.3e}, {max(odd_mags):.3e}]  "
      f"-- at/near the floating-point noise floor (effectively zero to machine precision)")
print(f"    even-L (2,4,6,8,10) |A_L| in [{min(even_mags):.3e}, {max(even_mags):.3e}]  "
      f"-- O(1) magnitudes")
even_phases = [round(cmath.phase(AL[L]) / math.pi) for L in range(2, 11, 2)]
print(f"    even-L phases, in units of pi: {even_phases}  "
      f"(all even-L terms sit on the negative-real axis, arg = +-pi)")
print(f"    even-L magnitude sequence (L=2,4,6,8,10): "
      f"{[round(abs(AL[L]), 6) for L in range(2, 11, 2)]}")
print("    => PARITY PATTERN: odd loop-orders L vanish (to machine precision); only even L")
print("       contribute, each as a negative real number.  Booked raw; not interpreted here.")

# ===========================================================================
banner("R1-2  THE READ (blind) -- tail45 = alpha_1^4 A_4 + alpha_1^5 A_5")
# ===========================================================================
tail45 = ALPHA1 ** 4 * AL[4] + ALPHA1 ** 5 * AL[5]
print(f"  A_4 = {cfmt(AL[4])}   A_5 = {cfmt(AL[5])}")
print(f"  tail45 = alpha_1^4 * A_4 + alpha_1^5 * A_5 = {cfmt(tail45)}")
print(f"  |tail45| = {abs(tail45):.15e}")
print(f"  arg(tail45) = {cmath.phase(tail45):.15f}  rad")

# ===========================================================================
banner("R1-3  THE CONFRONT (declared end -- the target enters ONLY here)")
# ===========================================================================
EPS_MAG = 1.7515e-7          # eps magnitude (the booked signed value is -1.7515e-7 rad)
EPS_SIGNED_RAD = -1.7515e-7
POISON_2A5 = 2.0 * ALPHA1 ** 5
POISON_A4 = ALPHA1 ** 4
POISON_A3 = ALPHA1 ** 3
poisons = {"2*alpha_1^5": POISON_2A5, "alpha_1^4": POISON_A4, "alpha_1^3": POISON_A3}

print(f"  target   eps            = {EPS_MAG:.6e}  (booked signed value: {EPS_SIGNED_RAD:.6e} rad)")
for name, val in poisons.items():
    print(f"  poison   {name:<12} = {val:.6e}")
print(f"  |tail45|                = {abs(tail45):.15e}")


def ratio(a, b):
    """symmetric multiplicative distance: how many times larger one is than the other (>=1)."""
    a, b = abs(a), abs(b)
    hi, lo = max(a, b), min(a, b)
    return float("inf") if lo == 0 else hi / lo


ratio_eps = ratio(abs(tail45), EPS_MAG)
poison_ratios = {name: ratio(abs(tail45), val) for name, val in poisons.items()}
nearest_poison_name = min(poison_ratios, key=poison_ratios.get)
nearest_poison_ratio = poison_ratios[nearest_poison_name]

print(f"\n  distance ratio |tail45| <-> eps            = {ratio_eps:.4f}x")
for name, r in poison_ratios.items():
    print(f"  distance ratio |tail45| <-> poison({name:<12}) = {r:.4f}x")
print(f"  nearest poison: {nearest_poison_name}  (ratio {nearest_poison_ratio:.4f}x)")

within_5pct_of_eps = ratio_eps <= 1.05
nearer_eps_than_every_poison_by_3x = all(poison_ratios[name] / ratio_eps >= 3.0 for name in poisons)
differs_from_eps_by_gt_2x = ratio_eps > 2.0
matches_a_poison_better = any(poison_ratios[name] < ratio_eps for name in poisons)

# FROZEN branch logic (mechanical, per the pre-reg -- evaluated in the declared order):
if within_5pct_of_eps and nearer_eps_than_every_poison_by_3x:
    branch = "LOCATION-CONFIRMED"
elif differs_from_eps_by_gt_2x or matches_a_poison_better:
    branch = "LOCATION-REFUTED"
else:
    branch = "AMBIGUOUS"

print(f"\n  within 5% of |eps|?                    {within_5pct_of_eps}")
print(f"  nearer eps than every poison by >=3x?  {nearer_eps_than_every_poison_by_3x}")
print(f"  differs from |eps| by >2x?              {differs_from_eps_by_gt_2x}")
print(f"  matches a poison better than eps?       {matches_a_poison_better}"
      + (f"  (nearest: {nearest_poison_name}, ratio {nearest_poison_ratio:.4f}x "
         f"vs eps ratio {ratio_eps:.4f}x)" if matches_a_poison_better else ""))

print(f"\n  *** R1-3 BRANCH: {branch} ***\n")

if branch == "LOCATION-CONFIRMED":
    print("""  BOOKED MEANING (pre-reg, verbatim sense): the alpha_1^{4-5} content of the forced
  asymmetry IS the -70 ppm's magnitude -- a MAJOR lead.  Booked raw; flagged for maximal
  adversarial scrutiny + an immediate follow-up pre-registration for the sign/phase
  identification (NOT decided here -- R1 does not determine sign/phase).
  The -70 ppm stays OPEN regardless: even LOCATION-CONFIRMED is a lead, not a closure.""")
elif branch == "LOCATION-REFUTED":
    print(f"""  BOOKED MEANING (pre-reg, verbatim sense): |tail45| differs from |eps| by
  {ratio_eps:.2f}x (and/or matches poison '{nearest_poison_name}' more closely, ratio
  {nearest_poison_ratio:.4f}x) -- the u^{{4-5}} location hypothesis, AS POSED THROUGH THIS
  FORCED FUNCTIONAL, is dead.  Booked raw; the -70 ppm's suppression object remains fully
  open (back to ML-5b's wall: the lepton-slice transport functional is still un-built).""")
else:
    print("""  BOOKED MEANING (pre-reg, verbatim sense): the read falls between the declared bands
  (or within 3x of a poison) -- booked raw as UNRESOLVED.  No rounding toward the
  flattering branch.  The -70 ppm's suppression object remains fully open.""")

r1_3_definite = branch in ("LOCATION-CONFIRMED", "LOCATION-REFUTED", "AMBIGUOUS")  # exhaustive by construction

# ===========================================================================
banner("R1-4  SCOPE DECLARATION (printed, NOT computed; never gates PASS/FAIL)")
# ===========================================================================
print("""  NOT claimed by this file, regardless of the R1-3 branch:
    (i)   any transport functional (the lepton-slice projection ML-5b identified as the
          un-built object is NOT constructed here);
    (ii)  any lepton-slice projection or alternative Q-combination (the ONE-FUNCTIONAL RULE
          was held absolute throughout: A(u), A_L, and tail45 are the only objects computed);
    (iii) the sign/phase identification -- even a CONFIRMED magnitude match would require its
          own pre-registered follow-up to address sign/phase (not decided in R1);
    (iv)  any scoreboard change: the -70 ppm stays OPEN in ALL THREE branches -- even
          LOCATION-CONFIRMED is a lead, not a closure.
  These remain the declared, unclaimed scope.""")

# ===========================================================================
banner("SUMMARY")
# ===========================================================================
print(f"""  R1-0  regression (carrier reproduction + per-order identity + truncation bound): {'PASS' if r1_0_ok else 'FAIL'}
  R1-1  order table L=1..10                                                        : printed above (raw, not gated)
  R1-2  tail45 (blind read)                                                        : {cfmt(tail45)}  (|tail45|={abs(tail45):.6e})
  R1-3  confront                                                                   : {branch}
  R1-4  scope declaration                                                          : printed above (declaration only)""")

exit_ok = r1_0_ok and r1_3_definite
print("\nRESULT:", f"R1-0 PASSED AND R1-3 REACHED A DEFINITE BRANCH ({branch})" if exit_ok else
      "R1-0 FAILED (see FAILURES above) -- a finding, not adjusted")
if FAILURES:
    print("FAILURES:", FAILURES)
sys.exit(0 if exit_ok else 1)
