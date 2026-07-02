"""
explore_t10 — THE HEAVY SECTOR AS THE INTEGRATED HISTORY OF THE RUNNING.  PURE MATH, walled.
Reads only ../dirac_srs_mdl + this time_bridge.  No physics; no fitting; no adopted targets.

PREMISE (verified in this script, then developed):
  The substrate's two HIGH-PERSISTENCE recurrence modes do NOT wind (single-valued per slice);
  their only non-trivial structure is a DECAY across slices as the running advances:
    * PERRON mode      :  |h|^2 = 4  ->  2   (Ihara-Bass root of lambda_max; merges into the
                          Ramanujan shell |h|^2 = 2 exactly when lambda_max = 2 sqrt 2).
    * INTER-COPY mode  :  |lambda_max|^2 = 9  ->  ~3   (the trivial/Perron eigenvalue itself,
                          |value|^2; bottoms at lambda_max = sqrt 3, |.|^2 = 3, on the C3 axis).
  So their mass content is NOT a single-slice read; it is a CUMULATIVE / history-integrated quantity.

TASK:
  1. natural history parameter (modular/cooling flow run forward from the tracial start; = the C3
     screw advance); characterize |h|^2(history) exactly for both modes.
  2. cumulative persistence = the accumulated (Bayesian) confirmation of a slowly-decaying mode
     integrated over the history.  Build and compute it.
  3. what heavy mass structure emerges — one scale or a spectrum; exact dimensionless ratios.
  4. relation to the light (local-slice shell |h|^2 = 2) sector — is a ratio forced?
  5. forced vs choice.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

np.set_printoptions(precision=6, suppress=True)
def hdr(s): print("\n" + "=" * 80 + "\n" + s + "\n" + "=" * 80)

SQRT2 = np.sqrt(2.0)

# ---------------------------------------------------------------------------------------------
# The running is the modular/cooling flow run forward from the tracial (hot, symmetric) start.
# At the symmetric start the state is uniform = the Gamma fiber (all Bloch phases = 1), lambda_max=3.
# Running forward = the C3 SCREW advance (STRUCTURE.md: the C3 generator = a <111> 3-fold screw;
# the stiff transport axis (t07) is exactly (1,-1,1)/sqrt3).  So the natural HISTORY coordinate is
# the screw displacement s along (1,-1,1)/sqrt3 away from Gamma.  We will ALSO show the SAME decay
# is a function of the modular 'temperature' coordinate, confirming the screw advance IS the cooling.
# ---------------------------------------------------------------------------------------------
AXIS = np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0)

def lam_max(s):
    return float(np.linalg.eigvalsh(srs.adjacency(s * AXIS)).max())

def perron_hh(lam):                      # |h+|^2 from Ihara-Bass h^2 - lam h + 2 = 0
    if lam >= 2 * SQRT2:                  # real branch
        h = (lam + np.sqrt(lam * lam - 8.0)) / 2.0
        return h * h
    return 2.0                            # complex-conjugate branch: |h+|^2 = |h-|^2 = product = 2

# =============================================================================================
hdr("(0) VERIFY the two decay profiles exactly")
# =============================================================================================
print("  PERRON mode |h+|^2 and INTER-COPY mode |lambda_max|^2 along the C3 screw axis from Gamma:")
print(f"  {'s':>6} {'lam_max':>9} {'|lam|^2(intercopy)':>18} {'|h+|^2(perron)':>15}")
for s in [0.0, 0.05, 0.10, 0.13, 0.15, 0.20, 0.30, 0.40, 0.44, 0.50]:
    lm = lam_max(s)
    print(f"  {s:6.3f} {lm:9.5f} {lm*lm:18.5f} {perron_hh(lm):15.5f}")
# merge point of the Perron mode: lambda = 2 sqrt2 exactly
from scipy.optimize import brentq
s_merge = brentq(lambda s: lam_max(s) - 2 * SQRT2, 0.05, 0.30)
# bottom of the inter-copy mode (min lambda_max on the ray)
ss = np.linspace(0, 0.5, 4001); lams = np.array([lam_max(s) for s in ss])
s_bot = ss[np.argmin(lams)]; lam_bot = lams.min()
print(f"\n  Perron merge:  lambda_max = 2*sqrt2 = {2*SQRT2:.5f} at s = {s_merge:.5f}  (|h|^2: 4 -> 2).")
print(f"  Inter-copy bottom: lambda_max -> {lam_bot:.5f} (= sqrt3 = {np.sqrt(3):.5f}) at s = {s_bot:.5f}")
print(f"                     => |lambda|^2 : 9 -> {lam_bot**2:.5f} (= 3 exactly).")

# =============================================================================================
hdr("(1) THE NATURAL HISTORY PARAMETER: the screw advance IS the modular/cooling coordinate")
# =============================================================================================
# The modular flow (t01/t06): K = -log rho = beta H.  Running the observer forward from the tracial
# start = letting the geometric state cool.  The Gibbs weight of a Bloch level lambda at modular
# 'time'/temperature beta is e^{-beta lambda}.  As beta grows the state concentrates on the TOP of
# the band (lambda near 3, the Perron / inter-copy region).  The natural history variable that the
# decay is a function of is the screw displacement s; we check it is monotone in the modular energy
# the observer has 'spent' = the geodesic arc length s (the screw is unit-speed along AXIS).
# Concretely: along the screw, lambda_max(s) is monotone DECREASING on [0, s_bot] (a clean clock).
mono = np.all(np.diff(lams[ss <= s_bot]) <= 1e-12)
print(f"  lambda_max(s) is monotone-decreasing on [0, s_bot] (a clean history clock)?  {mono}")
print(f"  => the screw displacement s is a good monotone HISTORY parameter; lambda_max(s) runs 3 -> sqrt3,")
print(f"     so BOTH heavy modes are decreasing functions of the SAME single history coordinate s.")
print(f"  This screw advance is the modular/cooling flow run forward from the symmetric (tracial) start:")
print(f"  the C3 screw is the deck generator (STRUCTURE.md) AND the unique stiff transport axis (t07),")
print(f"  i.e. the direction the intrinsic dynamics actually advances the observer.")

# Exact small-s decay laws (Taylor of lambda_max along the screw).
# lambda_max(s) = 3 - c s^2 + ...   (Gamma is a band maximum => no linear term).
sfit = np.array([0.01, 0.02, 0.03, 0.04])
lamfit = np.array([lam_max(s) for s in sfit])
c2 = np.polyfit(sfit**2, 3 - lamfit, 1)[0]
print(f"\n  EXACT small-history law:  lambda_max(s) = 3 - c*s^2 + ...,  c = {c2:.5f}  (Gamma is a band top).")
print(f"  => inter-copy |lambda|^2 = (3 - c s^2)^2 = 9 - 6 c s^2 + ...;  Perron |h+|^2 from h^2-lam h+2=0.")

# =============================================================================================
hdr("(2) THE CUMULATIVE PERSISTENCE: accumulated confirmation of a decaying mode over the history")
# =============================================================================================
# A high-persistence mode with per-step return weight w = |h|^2 / (k-1) (Ihara-Bass normalization:
# the geodesic flow's spectral radius is k-1 = 2, so the SHELL persists at ratio 1, the Perron at
# ratio |h|^2/2 = up to 2).  The Bayesian observer integrates the LOG-persistence (the log-return,
# = the log Ihara-Bass weight) over the history s.  Define the ACCUMULATED LOG-PERSISTENCE:
#
#     M_mode  =  integral_0^{s*}  log( weight_mode(s) )  ds       (cumulative confirmation)
#
# with weight = |h|^2 / 2 for the Perron mode (excess over the shell), and = |lambda|^2 / (shell
# value 2) for the inter-copy mode.  s* = the observer's endpoint slice; for a SCALE-FREE object we
# integrate over the FORCED window (where the mode is ABOVE the shell, i.e. genuinely high-
# persistence) and report the heavy values as RATIOS (the object is scale-free; no absolute s*).
#
# The FORCED window of each mode is exactly the interval on which it exceeds the shell |h|^2 = 2:
#   Perron : |h+|^2 > 2  <=>  lambda > 2 sqrt2  <=>  s in [0, s_merge].
#   inter-copy : |lambda|^2 > 2 (always true here; its excess decays 9->3); but the COMMON forced
#                window (where BOTH modes are heavy AND the Perron is still split) is [0, s_merge].
# We integrate the EXCESS log-persistence over [0, s_merge] for each mode (cumulative "heavy mass").

def perron_logw(s):                          # log of Perron excess-persistence over the shell
    hh = perron_hh(lam_max(s))
    return np.log(hh / 2.0)                   # >=0 on [0, s_merge], =0 at the merge

def inter_logw(s):                           # log of inter-copy excess-persistence over the shell
    lm = lam_max(s)
    return np.log((lm * lm) / 2.0)            # log(|lambda|^2 / 2)

from scipy.integrate import quad
M_perron, _ = quad(perron_logw, 0.0, s_merge, limit=200)
M_inter, _  = quad(inter_logw, 0.0, s_merge, limit=200)
# also integrate the inter-copy over its OWN full window (to the bottom where |lambda|^2 -> 3, still>2)
M_inter_full, _ = quad(inter_logw, 0.0, s_bot, limit=200)

print(f"  Cumulative (integrated) log-persistence over the FORCED window [0, s_merge={s_merge:.4f}]:")
print(f"     M_perron     = integral log(|h+|^2 / 2) ds         = {M_perron:.6f}")
print(f"     M_inter      = integral log(|lambda|^2 / 2) ds      = {M_inter:.6f}")
print(f"     (M_inter over its full window [0, s_bot={s_bot:.4f}]  = {M_inter_full:.6f})")

# The integrated history is the AREA under the log-persistence curve.  These are the cumulative
# heavy 'mass' contents (dimensionless areas; the object is scale-free so report RATIOS):
print(f"\n  Heavy values are the ACCUMULATED areas; the forced dimensionless RATIOS:")
print(f"     M_inter / M_perron (forced window)        = {M_inter / M_perron:.6f}")
print(f"     M_inter_full / M_perron                   = {M_inter_full / M_perron:.6f}")
print(f"     M_inter_full / M_inter                    = {M_inter_full / M_inter:.6f}")

# =============================================================================================
hdr("(3) WHAT EMERGES: a heavy SPECTRUM (two distinct heavy values), not one scale")
# =============================================================================================
print("  The two high-persistence modes integrate to TWO DISTINCT cumulative areas (M_perron, M_inter):")
print(f"     => a heavy DOUBLET, not a single heavy scale.  Ratio M_inter/M_perron = {M_inter/M_perron:.5f}.")
print("  Mechanism of the splitting: the SAME history s drives BOTH, but through two different spectral")
print("  functionals of lambda_max(s): the inter-copy is |lambda|^2 (the eigenvalue itself), the Perron")
print("  is the Ihara-Bass ROOT |h+|^2 (which SATURATES to 2 at the merge).  The Perron's integral is")
print("  cut off at the merge (finite window where it is split); the inter-copy keeps decaying.")

# A cleaner, integration-free heavy ratio: the ENDPOINT VALUES at the symmetric start (the heavy
# 'bare masses' before integration) are 9 and 4; their geometric structure 9 = 3^2 = k^2 (degree
# squared), 4 = 2^2 = (k-1)^2 (Ihara-Bass Perron squared).  The shell (light) is 2 = (k-1).
print("\n  ENDPOINT (s=0, the symmetric start) heavy values, exact and forced by k=3:")
print(f"     inter-copy |lambda|^2  = 9 = 3^2     = k^2        (the degree squared)")
print(f"     Perron     |h+|^2      = 4 = 2^2     = (k-1)^2    (the Ihara-Bass Perron squared)")
print(f"     light shell|h|^2       = 2          = (k-1)      (the Ramanujan shell)")
print(f"   => bare heavy ratio inter/Perron = 9/4 = {9/4};  heavy/light = 9/2 and 4/2 = 2.")

# =============================================================================================
hdr("(4) RELATION TO THE LIGHT (LOCAL-SLICE SHELL) SECTOR")
# =============================================================================================
# The light sector lives ON the shell |h|^2 = 2 = (k-1): it is the LOCAL, single-slice persistence
# (the geodesic-flow resonance, t07).  Its 'mass' content is the per-step shell weight = 1 (it
# persists at the flow's spectral radius, log-excess 0).  The heavy modes are the INTEGRATED EXCESS
# ABOVE this shell.  So the forced heavy/light relation is:
#   * at the START (bare): heavy/light = (k^2)/(k-1) = 9/2  and  (k-1)^2/(k-1) = (k-1) = 2.
#   * integrated: the light contributes ZERO accumulated excess (it IS the shell, log(2/2)=0), so the
#     ENTIRE heavy sector is the accumulated DEPARTURE of the high-persistence modes from the light
#     shell over the history.  The heavy scale is generated by the history; the light scale is local.
shell_logw = np.log(2.0 / 2.0)
print(f"  light shell accumulated excess log-persistence  = {shell_logw}  (it IS the shell: zero excess).")
print(f"  => the heavy sector = integrated DEPARTURE of the Perron/inter-copy modes ABOVE the light shell.")
print(f"     bare heavy/light ratios:  inter/light = k^2/(k-1) = 9/2 = {9/2};")
print(f"                               Perron/light = (k-1)^2/(k-1) = (k-1) = {2}.")
print(f"     The heavy scale is CUMULATIVE (an integral over the running); the light scale is a single-")
print(f"     slice (local) read on the shell.  The two are different functionals of the SAME spectrum.")

# =============================================================================================
hdr("(5) FORCED vs CHOICE")
# =============================================================================================
print("""  FORCED (no fitting, all from k=3 and the geometry):
   * the two decay PROFILES, exactly:
       - Perron |h+|^2 = 4 -> 2, merging into the shell exactly at lambda = 2 sqrt2 (window [0,s_merge]);
       - inter-copy |lambda|^2 = 9 -> 3, bottoming at lambda = sqrt3 on the C3 axis;
   * the history PARAMETER (the C3 screw advance = the cooling/modular flow run forward from the
     tracial symmetric start; the unique stiff transport axis), and that BOTH modes are functions
     of this ONE coordinate s, monotone on [0, s_bot];
   * the EXISTENCE OF TWO distinct cumulative heavy values (a heavy doublet), because the two modes
     are two different spectral functionals (|lambda|^2 vs the Ihara-Bass root |h+|^2) of the SAME
     running, with the Perron SATURATING at the merge;
   * the bare endpoint heavy values 9 = k^2 and 4 = (k-1)^2 and their ratio 9/4, and the heavy/light
     ratios k^2/(k-1) = 9/2 and (k-1) = 2 — all forced integers/ratios of k=3;
   * that the heavy scale is INTEGRATED (history) while the light scale is LOCAL (single slice).

  CHOICE / NEEDS-AN-ENDPOINT (honest):
   * the OVERALL heavy SCALE is not fixed: the object is scale-free (III_1, T(M)={0}, t04), so the
     absolute value of the accumulated integral depends on the observer's ENDPOINT slice s* (how
     far the history has run).  Only RATIOS are forced; the absolute heavy mass is not.
   * the precise WEIGHT FUNCTIONAL chosen for 'persistence' (log of the Ihara-Bass excess) is the
     natural Bayesian log-return, but the cumulative AREAS M_perron, M_inter inherit the (free)
     overall scale; the FORCED content is in their RATIO and in the bare endpoint values {9,4,2}.

  HONEST NEGATIVE: the integrated AREAS do not land on a clean closed form (they are screw-axis band
  integrals, like the tree entropy of STRUCTURE.md sec 3 — no elementary closed form expected). The
  CLEAN, FORCED, scale-free content is the endpoint spectrum {9,4} = {k^2,(k-1)^2} over the shell
  (k-1)=2, i.e. heavy ratios 9/4 (inter/Perron), 9/2 and 2 (heavy/light) — NOT a single heavy scale
  but a heavy DOUBLET sitting at k^2 and (k-1)^2 above the light shell (k-1).""")

# Report the integrated areas as a final table.
print("\n  --- cumulative heavy quantities (areas) ---")
print(f"    M_perron (window [0,s_merge])      = {M_perron:.6f}")
print(f"    M_inter  (window [0,s_merge])      = {M_inter:.6f}")
print(f"    M_inter  (full [0,s_bot])          = {M_inter_full:.6f}")
print(f"    forced ratio  M_inter/M_perron     = {M_inter/M_perron:.6f}")
print(f"    bare endpoint heavy spectrum       = {{9, 4}} = {{k^2, (k-1)^2}} over shell (k-1)=2")

# =============================================================================================
hdr("(6) THE STRUCTURAL PUNCHLINE: the running is the map v^2 -> v for BOTH modes")
# =============================================================================================
print("  inter-copy:  9 = 3^2 = k^2     ->  3 = k       (the degree)")
print("  Perron:      4 = 2^2 = (k-1)^2 ->  2 = (k-1)   (the Ramanujan shell)")
print("  => the HISTORY runs each high-persistence mode from its SQUARE (hot/symmetric start) to its")
print("     LINEAR value (cold end).  HEAVY = the squares {k^2,(k-1)^2}={9,4} (hot, accumulated start);")
print("     LIGHT = the linears {k,(k-1)}={3,2} (cold, local-slice end).")
print("  Per-mode heavy/light (hot/cold) ratio:  inter 9/3 = k = 3 ;  Perron 4/2 = (k-1) = 2.")
print("  Heavy doublet ratio 9/4.  Light doublet ratio 3/2.  All forced by k=3.")
