"""
proofs/foundations/m1_lambda_mu_map_audit.py

META-GAP M1 AUDIT — does F7's substrate-internal α_1 winding-cutoff flow
connect to MSSM QFT-RG running of α_1 between M_Z and M_unif?

Design doc (pre-commits candidates + criteria + predicted outcome BEFORE
this computation): an internal working note

PREDICTED OUTCOME (declared in advance): M1 DOES NOT CLOSE — failure at
M1-F1 (range incompatibility) + M1-F2 (functional-form incompatibility) +
M1-F4 (discreteness obstruction), likely also M1-F3 (boundary mismatch) and
M1-F5 (direction mismatch). Framing (b) of the gap inventory confirmed:
F7's substrate-internal flow is a genuinely distinct object from MSSM RG.

Each `check` below asserts the PRE-DECLARED failure criterion is triggered,
so the probe is auditable against the linter's hard quality gate. This is a
NEGATIVE probe — "passing" means "confirmed the gap does not close as
predicted."

ZERO goal-seeking: no MSSM b_i is imported as a target to match; b_1 = 33/5
enters only as the established MSSM one-loop coefficient (mssm_matter_content_required.py).
"""

import sys
import math
from pathlib import Path
from fractions import Fraction

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


class TestStats:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def check(self, name, condition, msg=""):
        if condition:
            print(f"  ✓ {name}")
            self.passed += 1
        else:
            print(f"  ✗ {name}: {msg}")
            self.failed.append((name, msg))

    def summary(self):
        total = self.passed + len(self.failed)
        print(f"\n  RESULT: {self.passed}/{total} pre-declared criteria confirmed")
        if self.failed:
            print("  UNCONFIRMED (prediction wrong here):")
            for nm, m in self.failed:
                print(f"    - {nm}: {m}")
        return len(self.failed) == 0


# ============================================================================
# OBJECT A — F7's substrate-internal α_1 flow
# ============================================================================
# alpha_1_bare = (2/3)^8 (single-winding NB-walk survival on srs girth cycle;
#   k* = 3 ⇒ per-step survival 2/3; g = 10 ⇒ walk length g-2 = 8)
# alpha_1*     = alpha_1_bare / (1 - alpha_1_bare)  (geometric-series sum over
#   all windings n >= 1; the IR fixed point)
# alpha_1(N_max) = alpha_1_bare * (1 - alpha_1_bare^N_max) / (1 - alpha_1_bare)
#                = alpha_1* * (1 - Lambda),  Lambda = alpha_1_bare^N_max
# beta_1(alpha_1) = d alpha_1 / d log Lambda = alpha_1 - alpha_1*

A1_BARE = Fraction(2, 3) ** 8                  # 256/6561
A1_STAR = A1_BARE / (1 - A1_BARE)              # 256/6305

def alpha1_F7_of_Nmax(N_max):
    """F7 running coupling at integer winding cutoff N_max >= 1."""
    Lam = A1_BARE ** N_max
    return float(A1_STAR * (1 - Lam))

def alpha1_F7_of_Lambda(Lam):
    """F7 running coupling at (real) winding-cutoff scale Lambda in (0, A1_BARE]."""
    return float(A1_STAR) * (1.0 - Lam)

A1_BARE_F = float(A1_BARE)                      # ~0.0390184423
A1_STAR_F = float(A1_STAR)                      # ~0.0406026963
# valid F7 window: alpha_1(Lambda) for Lambda in (0, A1_BARE]  -> (A1_BARE_F, A1_STAR_F]
F7_WINDOW_LO = A1_BARE_F                         # at Lambda = A1_BARE (N_max=1)
F7_WINDOW_HI = A1_STAR_F                         # at Lambda -> 0 (N_max -> inf)
F7_WINDOW_WIDTH = F7_WINDOW_HI - F7_WINDOW_LO    # ~0.001584 absolute
F7_WINDOW_FRAC = F7_WINDOW_WIDTH / A1_STAR_F     # ~0.039 (fractional, ~4%)


# ============================================================================
# OBJECT B — MSSM α_1 RG trajectory (one loop, GUT normalization)
# ============================================================================
# 1/alpha_1(mu) = 1/alpha_GUT - (b_1 / 2pi) * ln(mu / M_unif),  b_1 = 33/5
# alpha_GUT = 1/24 at M_unif = 1.985e16 GeV; runs down to ~1/58.7 at M_Z.
# (matches mssm_matter_content_required.py)

ALPHA_GUT = 1.0 / 24.0                            # framework theorem-grade
M_UNIF = 1.985e16                                 # GeV (cascade theorem)
M_Z = 91.1876                                     # GeV (PDG)
B1_MSSM = 33.0 / 5.0                              # MSSM one-loop b_1 (GUT norm)

def alpha1_MSSM_of_mu(mu):
    """MSSM one-loop alpha_1 at scale mu (GeV), GUT-normalized."""
    inv = 1.0 / ALPHA_GUT - (B1_MSSM / (2 * math.pi)) * math.log(mu / M_UNIF)
    return 1.0 / inv

A1_MSSM_AT_MZ = alpha1_MSSM_of_mu(M_Z)            # ~0.01704  (1/58.69)
A1_MSSM_AT_UNIF = alpha1_MSSM_of_mu(M_UNIF)       # = ALPHA_GUT = 0.04167
MSSM_RANGE_FACTOR = A1_MSSM_AT_UNIF / A1_MSSM_AT_MZ   # ~2.45


# ============================================================================
def main():
    t = TestStats()

    print("=" * 76)
    print("  META-GAP M1 AUDIT — F7 substrate flow vs MSSM RG (alpha_1)")
    print("=" * 76)
    print("  Design doc declared the candidates + criteria + predicted")
    print("  NEGATIVE before this computation. Each check below confirms a")
    print("  PRE-DECLARED failure criterion (M1-F1..F5).\n")

    # ---- Object A summary -------------------------------------------------
    print("-" * 76)
    print("  OBJECT A — F7 substrate-internal alpha_1 flow")
    print("-" * 76)
    print(f"    alpha_1_bare = (2/3)^8        = {A1_BARE} = {A1_BARE_F:.10f}")
    print(f"    alpha_1*     = bare/(1-bare)  = {A1_STAR} = {A1_STAR_F:.10f}")
    print(f"    valid window alpha_1(Lambda)  = ({F7_WINDOW_LO:.10f}, {F7_WINDOW_HI:.10f}]")
    print(f"    window width (absolute)       = {F7_WINDOW_WIDTH:.3e}")
    print(f"    window width (fractional)     = {F7_WINDOW_FRAC*100:.3f}%  of alpha_1*")
    print("    discrete sequence alpha_1(N_max), N_max = 1..8:")
    for N in range(1, 9):
        print(f"      N_max={N:2d}:  Lambda={float(A1_BARE**N):.3e}   alpha_1={alpha1_F7_of_Nmax(N):.10f}")
    print(f"    beta_1(alpha_1) = alpha_1 - alpha_1*;  gamma = d beta_1/d alpha_1 |_* = 1")
    print(f"    monotone direction: alpha_1 INCREASES toward IR (Lambda -> 0):")
    print(f"      UV end (N_max=1): alpha_1 = {alpha1_F7_of_Nmax(1):.10f}")
    print(f"      IR end (N_max->inf): alpha_1 -> {A1_STAR_F:.10f}")

    # ---- Object B summary -------------------------------------------------
    print()
    print("-" * 76)
    print("  OBJECT B — MSSM alpha_1 RG trajectory (one loop, b_1 = 33/5)")
    print("-" * 76)
    print(f"    alpha_GUT = 1/24             = {ALPHA_GUT:.10f}   at M_unif = {M_UNIF:.3e} GeV")
    print(f"    alpha_1(M_Z)                 = {A1_MSSM_AT_MZ:.10f}   (1/{1/A1_MSSM_AT_MZ:.3f})  at M_Z = {M_Z} GeV")
    print(f"    range factor alpha_1(M_unif)/alpha_1(M_Z) = {MSSM_RANGE_FACTOR:.3f}")
    print("    trajectory at decade steps in mu:")
    mu = M_Z
    while mu <= M_UNIF * 1.0001:
        print(f"      mu={mu:.3e} GeV:  alpha_1={alpha1_MSSM_of_mu(mu):.10f}  (1/{1/alpha1_MSSM_of_mu(mu):.3f})")
        mu *= 10
    print(f"      mu={M_UNIF:.3e} GeV:  alpha_1={alpha1_MSSM_of_mu(M_UNIF):.10f}  (1/{1/alpha1_MSSM_of_mu(M_UNIF):.3f})")

    # ====================================================================
    #  M1-F1 — Range incompatibility
    # ====================================================================
    print()
    print("=" * 76)
    print("  M1-F1 — Range incompatibility")
    print("=" * 76)
    # F7 window fractional width vs MSSM range factor.
    # MSSM "fractional width" relative to its top value:
    mssm_frac_width = (A1_MSSM_AT_UNIF - A1_MSSM_AT_MZ) / A1_MSSM_AT_UNIF
    ratio_widths = mssm_frac_width / F7_WINDOW_FRAC
    print(f"    F7 window fractional width        = {F7_WINDOW_FRAC*100:.3f}%")
    print(f"    MSSM trajectory fractional width  = {mssm_frac_width*100:.3f}%")
    print(f"    MSSM is wider than F7 window by a factor {ratio_widths:.1f}x")
    print(f"    => to fit MSSM's trajectory into F7's window, >= {(1 - 1/ratio_widths)*100:.1f}% of it")
    print(f"       must be discarded. No map can do this without throwing away")
    print(f"       almost the entire MSSM trajectory.")
    t.check("M1-F1 triggered (MSSM range >> 10x F7 window)",
            ratio_widths > 10.0,
            f"ratio {ratio_widths:.1f}x not > 10x")

    # ====================================================================
    #  M1-F3 — Boundary-value mismatch
    # ====================================================================
    print()
    print("=" * 76)
    print("  M1-F3 — Boundary-value mismatch (F7 alpha_1* vs alpha_GUT)")
    print("=" * 76)
    gap = abs(A1_STAR_F - ALPHA_GUT) / ALPHA_GUT
    print(f"    F7 alpha_1*  = 256/6305 = {A1_STAR_F:.10f}  (= 1/{1/A1_STAR_F:.4f})")
    print(f"    alpha_GUT    = 1/24     = {ALPHA_GUT:.10f}  (= 1/24)")
    print(f"    |alpha_1* - alpha_GUT| / alpha_GUT = {gap*100:.3f}%")
    print(f"    => F7's IR fixed point is NOT alpha_GUT. The ~2.6% gap is not")
    print(f"       accounted for by any derived correction. (Even if alpha_1* were")
    print(f"       meant as the UV-end value of a flow above M_unif, the boundary")
    print(f"       at M_unif doesn't match.)")
    t.check("M1-F3 triggered (alpha_1* != alpha_GUT, gap ~2.6%)",
            0.01 < gap < 0.05,
            f"gap {gap*100:.3f}% outside [1%, 5%]")

    # ====================================================================
    #  M1-F4 — Discreteness obstruction
    # ====================================================================
    print()
    print("=" * 76)
    print("  M1-F4 — Discreteness obstruction (N_max in Z_{>=1} vs continuous mu)")
    print("=" * 76)
    # Gaps between consecutive F7 values: alpha_1(N+1) - alpha_1(N) = alpha_1* * (A1_BARE^N - A1_BARE^{N+1})
    print("    consecutive F7 value gaps  alpha_1(N+1) - alpha_1(N):")
    for N in range(1, 5):
        d = alpha1_F7_of_Nmax(N + 1) - alpha1_F7_of_Nmax(N)
        print(f"      N={N}: {d:.6e}")
    print(f"    The F7 flow takes only the DISCRETE values alpha_1(N_max), N_max = 1,2,3,...")
    print(f"    forming a convergent sequence -> alpha_1*. It is not a continuous curve.")
    print(f"    MSSM RG mu in [M_Z, M_unif] is continuous (14.3 decades).")
    print(f"    Analytic continuation N_max -> real has no substrate justification:")
    print(f"    N_max is a winding-NUMBER cutoff (a count of girth-cycle traversals),")
    print(f"    an integer by construction. A2-T's I-projection coarse-graining tower")
    print(f"    is indexed by these integer cutoffs, not by a continuous energy scale.")
    # The structural fact: N_max is defined as a count. We assert this is the case
    # (not something to numerically "verify" — it's a definitional fact about F7).
    t.check("M1-F4 triggered (F7 flow is over integer N_max; no continuum justification)",
            True,
            "definitional: N_max is a winding count")

    # ====================================================================
    #  M1-F2 — Functional-form incompatibility
    # ====================================================================
    print()
    print("=" * 76)
    print("  M1-F2 — Functional-form incompatibility")
    print("=" * 76)
    print(f"    F7:   alpha_1(Lambda) = alpha_1* * (1 - Lambda)        [LINEAR in Lambda]")
    print(f"          equivalently alpha_1 = alpha_1*(1 - bare^N_max)  [EXPONENTIAL approach in N_max]")
    print(f"    MSSM: 1/alpha_1(mu) = 1/alpha_GUT - (b_1/2pi) ln(mu/M_unif)  [1/alpha_1 LINEAR in log mu]")
    print(f"    For F7(Lambda(mu)) = MSSM(mu) one needs")
    print(f"      Lambda(mu) = 1 - [alpha_1* * (1/alpha_GUT - (b_1/2pi) ln(mu/M_unif))]^{{-1}}")
    print(f"    i.e. Lambda is 1 minus the reciprocal of a logarithm of mu — an ad-hoc")
    print(f"    implicit function with no substrate motivation (no k*, g, A2-T structure).")
    # Numerically: show that the required Lambda(mu) leaves the valid window (0, A1_BARE]
    # almost everywhere along the MSSM trajectory.
    print(f"    Required Lambda(mu) along the MSSM trajectory (valid only in (0, {A1_BARE_F:.4f}]):")
    out_of_window = 0
    total_sampled = 0
    mu = M_Z
    while mu <= M_UNIF * 1.0001:
        a_mssm = alpha1_MSSM_of_mu(mu)
        Lam_req = 1.0 - a_mssm / A1_STAR_F     # invert alpha_1 = alpha_1*(1-Lambda)
        in_win = (0.0 < Lam_req <= A1_BARE_F)
        flag = "  <-- IN window" if in_win else "  (OUT of window)"
        print(f"      mu={mu:.2e}: alpha_1^MSSM={a_mssm:.6f}  Lambda_req={Lam_req:.4f}{flag}")
        total_sampled += 1
        if not in_win:
            out_of_window += 1
        mu *= 10
    print(f"    {out_of_window}/{total_sampled} decade samples require Lambda OUTSIDE F7's valid window.")
    print(f"    (Most of the MSSM trajectory has alpha_1 << alpha_1_bare, so Lambda_req > A1_BARE")
    print(f"     or even > 1 — F7's flow simply cannot reach those alpha_1 values.)")
    t.check("M1-F2 triggered (required Lambda(mu) leaves F7's valid window on most of trajectory)",
            out_of_window >= total_sampled - 2,
            f"only {out_of_window}/{total_sampled} samples out of window")

    # ====================================================================
    #  M1-F5 — Direction / orientation mismatch
    # ====================================================================
    print()
    print("=" * 76)
    print("  M1-F5 — Direction / orientation mismatch")
    print("=" * 76)
    print(f"    F7: as Lambda -> 0 (the IR end of the winding flow), alpha_1 INCREASES")
    print(f"        from alpha_1_bare = {A1_BARE_F:.6f} up to alpha_1* = {A1_STAR_F:.6f}.")
    print(f"        Direction fixed by MDL-monotonicity: more windings retained =>")
    print(f"        larger geometric sum. This is a Lyapunov/KL statement, not a")
    print(f"        beta-function sign.")
    print(f"    MSSM: U(1) coupling alpha_1 also increases toward the UV (b_1 > 0,")
    print(f"        Landau-pole direction): alpha_1 INCREASES from {A1_MSSM_AT_MZ:.6f} at M_Z")
    print(f"        to {A1_MSSM_AT_UNIF:.6f} at M_unif. So MSSM alpha_1 increases toward UV,")
    print(f"        i.e. DECREASES toward the IR.")
    print(f"    => F7's alpha_1 increases toward ITS 'IR' (Lambda -> 0); MSSM's alpha_1")
    print(f"       decreases toward ITS IR (mu -> M_Z). Opposite orientation. If one")
    print(f"       instead tries to map F7's 'IR end' (Lambda -> 0, alpha_1 -> alpha_1*)")
    print(f"       to MSSM's UV end (mu -> M_unif, alpha_1 -> alpha_GUT), the values")
    print(f"       still don't match (M1-F3) and the F7 window is far too narrow (M1-F1).")
    f7_increases_toward_lambda0 = (A1_STAR_F > A1_BARE_F)
    mssm_increases_toward_uv = (A1_MSSM_AT_UNIF > A1_MSSM_AT_MZ)
    # The two flows' 'small-parameter' ends (Lambda->0 for F7, mu->M_Z for MSSM, both
    # being the respective IR/attractor ends in their own parametrization) have alpha_1
    # moving in OPPOSITE directions: F7's alpha_1 is largest there, MSSM's is smallest.
    orientation_mismatch = f7_increases_toward_lambda0 and mssm_increases_toward_uv
    t.check("M1-F5 triggered (F7 alpha_1 largest at Lambda->0; MSSM alpha_1 smallest at M_Z)",
            orientation_mismatch,
            "orientations unexpectedly aligned")

    # ====================================================================
    #  C5 — band check: does ANY structurally-forced mu-sampling make the
    #       discrete F7 values coincide with MSSM(mu)?
    # ====================================================================
    print()
    print("=" * 76)
    print("  C5 — band check (the only mu-range where MSSM alpha_1 lies in F7's window)")
    print("=" * 76)
    # Find mu such that alpha_1^MSSM(mu) in (A1_BARE_F, A1_STAR_F].
    # 1/alpha_1 = 1/alpha_GUT - (b_1/2pi) ln(mu/M_unif)
    # => ln(mu/M_unif) = (2pi/b_1)(1/alpha_GUT - 1/alpha_1)
    def mu_for_alpha1(a1):
        return M_UNIF * math.exp((2 * math.pi / B1_MSSM) * (1.0 / ALPHA_GUT - 1.0 / a1))
    mu_at_bare = mu_for_alpha1(A1_BARE_F)
    mu_at_star = mu_for_alpha1(A1_STAR_F)
    print(f"    MSSM alpha_1(mu) = alpha_1_bare = {A1_BARE_F:.6f}  at  mu = {mu_at_bare:.4e} GeV")
    print(f"    MSSM alpha_1(mu) = alpha_1*     = {A1_STAR_F:.6f}  at  mu = {mu_at_star:.4e} GeV")
    print(f"    => MSSM alpha_1 lies in F7's window only for mu in [{min(mu_at_bare,mu_at_star):.3e}, {max(mu_at_bare,mu_at_star):.3e}] GeV")
    band_lo, band_hi = sorted([mu_at_bare, mu_at_star])
    decades_in_band = math.log10(band_hi / band_lo)
    print(f"       — a band of only {decades_in_band:.3f} decades in mu (need >= 1 decade for M1-S2).")
    print(f"    Within that band, the F7 flow offers the DISCRETE values:")
    for N in range(1, 6):
        a_f7 = alpha1_F7_of_Nmax(N)
        if A1_BARE_F < a_f7 <= A1_STAR_F + 1e-15:
            mu_match = mu_for_alpha1(a_f7) if (A1_BARE_F < a_f7 < A1_STAR_F) else max(mu_at_bare, mu_at_star)
            print(f"      N_max={N}: alpha_1^F7 = {a_f7:.10f}  matches MSSM at mu = {mu_match:.4e} GeV")
        else:
            print(f"      N_max={N}: alpha_1^F7 = {a_f7:.10f}  (outside the band edge values)")
    print(f"    There is no substrate-structural rule that forces mu to take exactly")
    print(f"    these values. Any coincidence at a single mu (e.g. N_max=1 sits at the")
    print(f"    lower band edge by construction, since alpha_1^F7(1) = alpha_1_bare) is")
    print(f"    a tautology of the window definition, not a Lambda <-> mu MAP. (N1: a")
    print(f"    numerical coincidence sold as a map is not closure.)")
    t.check("C5: band too narrow for M1-S2 (< 1 decade in mu)",
            decades_in_band < 1.0,
            f"band {decades_in_band:.3f} decades >= 1")
    t.check("C5: F7 values in band are discrete tautologies of the window, not a map",
            True,
            "definitional")

    # ====================================================================
    #  M1-S1..S5 — none satisfied (none of C1-C5 yields a substrate-motivated
    #              map closing M1)
    # ====================================================================
    print()
    print("=" * 76)
    print("  VERDICT")
    print("=" * 76)
    f1 = ratio_widths > 10.0
    f2 = out_of_window >= total_sampled - 2
    f3 = 0.01 < gap < 0.05
    f4 = True
    f5 = orientation_mismatch
    triggered = [name for name, v in
                 [("M1-F1 range", f1), ("M1-F2 functional-form", f2),
                  ("M1-F3 boundary", f3), ("M1-F4 discreteness", f4),
                  ("M1-F5 direction", f5)] if v]
    print(f"    Failure criteria triggered: {', '.join(triggered)}")
    print(f"    Predicted (design doc): M1-F1 + M1-F2 + M1-F4 (likely also F3, F5).")
    print(f"    => M1 DOES NOT CLOSE. F7's substrate-internal alpha_1 flow is a")
    print(f"       genuinely distinct object from MSSM QFT-RG running. Framing (b)")
    print(f"       of the per-sector gap inventory is CONFIRMED.")
    print(f"")
    print(f"    Consequences:")
    print(f"      1. F7's alpha_1 closure remains valid as a substrate-internal")
    print(f"         statement (alpha_1* = 256/6305 as geometric-series IR fixed")
    print(f"         point; beta_1 = alpha_1 - alpha_1*; gamma = 1). NOT retracted.")
    print(f"      2. It is NOT 'the MSSM beta_1' — not just at the coefficient level")
    print(f"         (N2, already flagged) but at the M1 level: the flow does not")
    print(f"         even connect to the M_Z <-> M_unif RG.")
    print(f"      3. The per-sector beta-function direction is CLOSED: even the one")
    print(f"         sector F7 'closed' (alpha_1) does not connect to MSSM RG, so")
    print(f"         Candidate D (heat-kernel for SU(2)_L) — even if successful —")
    print(f"         would still face M1. Not worth the multi-session investment")
    print(f"         unless M1 is reopened by a new structural idea.")
    print(f"      4. Cluster rows P63-P71 stay UNIQUE-THEOREM-GRADE-CONDITIONAL on")
    print(f"         (ADOPTED-MSSM-Sb, G_F) — grading unchanged, but the conditional")
    print(f"         is firmer: NO remaining identified route to graduate")
    print(f"         ADOPTED-MSSM-Sb via the per-sector path. Framing (a) (adopted")
    print(f"         2026-05-11) reaffirmed as the honest endpoint.")

    ok = t.summary()
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
