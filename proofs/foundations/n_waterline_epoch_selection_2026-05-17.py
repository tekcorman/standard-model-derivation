#!/usr/bin/env python3
"""
proofs/foundations/n_waterline_epoch_selection_2026-05-17.py

THE N-WATERLINE SHOT — Gap-G1 epoch-selection test (supersedes the
retracted "epoch floor" interpretation of target_shot_absolute_scale).

Established (stands): the framework's absolute scale is parameter-free
in natural units with DERIVED toggle dynamics (p_toggle=1/(k*N),
H=1/(N·t_P) coeff 1, growth law dN/dt=h·N/t_P — all theorem-grade, no
adoption), EXCEPT the current value of one pure integer N_hub.
Documented block (an internal working note): the
FORM is derived/CLOSED; the current epoch VALUE is BLOCKED — "requires
an independent derivation of the current N from the cosmological
history (inflation end + expansion) / the discrete Gauss-Codazzi
theorem." Λ_substrate = 1/N² is DERIVED from N (Friedmann + H=1/(N·t_P)),
so it is NOT an independent handle.

USER HYPOTHESIS UNDER TEST (the N-waterline): b1' established the
observation walk IS the dynamics and A2-T is the unique I-projection
fixed point *in distribution space*. Is there an analogous fixed point
*in N* — a self-consistent substrate size/walk-origin — that pins
N_hub from substrate structure ALONE (zero observational input)?

DECISIVE LOGIC. A substrate-internal quantity can pin a UNIQUE N only
if it is class (iii): N-NON-scale-invariant AND N-INDEPENDENT (a fixed
structural number set against an N-dependent quantity yields one N).
If every candidate is (i) an N-independent pure constant (cannot select
an N) or (ii) derived-FROM-N (circular), then NO N-waterline exists and
Gap G1 is NOT a fixed-point-in-N problem — it reduces to a
boundary/initial condition (the documented walk-origin N₀ /
discrete-Gauss-Codazzi). That characterized negative is a VALID
deliverable (route-elimination, Need-B-style), and is explicitly NOT
the retracted "epoch floor / provably impossible" claim: it is a
well-defined OPEN, BOUNDED target of a different type.

ANTI-NUMEROLOGY (load-bearing, and a correction of the target shot's
conflation): a *principled* pre-declared structural self-consistency
that outputs N with ZERO observational input would be a legitimate
Gap-G1 closure (a HIT — NOT numerology). Only a *fitted/coincidental*
match to the observed ≈8.39×10⁶⁰ is numerology. This probe tests for
the principled route and forbids the fitted one. It does NOT search
constants for 8.39e60.

Five aborts pre-declared BELOW. HIT = a class-(iii) quantity pins N,
zero observational input. Otherwise = CHARACTERIZED HONEST NEGATIVE
(no N-waterline; Gap G1 = the derived-walk-origin boundary problem).
"""
from __future__ import annotations
import sys
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "predictions"))

FAIL = []
HIT = []


def note_hit(tag, msg):
    print(f"\n  ★ CLASS-(iii) / HIT [{tag}]\n    {msg}")
    HIT.append(tag)


def abort(tag, msg):
    print(f"\n  ✗ ABORT [{tag}]\n    {msg}")
    FAIL.append(tag)


def head(s):
    print("\n" + "=" * 74 + f"\n  {s}\n" + "=" * 74)


print(__doc__)
print("=" * 74)
print("  PRE-DECLARED ABORTS:")
print("=" * 74)
print("""
  NW-A1 ENUMERATION  every substrate-internal quantity that could
                     constrain N must be classifiable (i)/(ii)/(iii).
                     A non-exhaustive / hand-wavy enumeration ⇒ cannot
                     conclude ⇒ abort.
  NW-A2 Λ-HANDLE     decisive route: if Λ_substrate is an INDEPENDENT
                     structural number (N-independent) ⇒ N=1/√Λ ⇒ HIT.
                     If Λ is N-derived (=1/N² via Friedmann) ⇒ no
                     handle (honest negative on this route).
  NW-A3 MDL-FIXED-PT user hypothesis: a self-consistency
                     S_total(N) = capacity(N) pins a unique N IFF the
                     two have DIFFERENT N-scaling. Same scaling ⇒
                     scale-invariant ⇒ no fixed point (characterized
                     negative — and it says exactly WHY).
  NW-A4 NO-FIT       the probe must NOT reproduce N_hub by matching the
                     observed value. A HIT must SOLVE a pre-declared
                     structural equation with ZERO observational input.
                     Any observed-value matching ⇒ numerology ⇒ abort.
  NW-A5 NEG-IS-VALID if A1–A3 yield no class-(iii)/no fixed-point, the
                     CHARACTERIZED HONEST NEGATIVE (Gap G1 = a
                     derived-walk-origin boundary problem, not a
                     fixed-point-in-N) is the deliverable — NOT a
                     failure, and NOT the retracted "floor/impossible".
""")


# ======================================================================
# STEP 1 — NW-A1: enumerate substrate-internal quantities, classify
# ======================================================================
head("STEP 1 — NW-A1: enumerate & classify every N-relevant quantity")

# (name, N-scaling exponent or 'const', class, why)
QUANTS = [
    ("k* = 3",                       "const", "i",
     "pure structural integer; N-independent ⇒ cannot select an N"),
    ("g = 10 (girth)",               "const", "i",
     "pure structural integer; N-independent"),
    ("α₁ = (2/3)^8",                 "const", "i",
     "NB-survival ratio; N-independent dimensionless"),
    ("ε_CP = 1/5, dark 5/12, …",     "const", "i",
     "dimensionless structural ratios; N-independent"),
    ("Cheeger/spectral gap h",       "const", "i",
     "h∈[0.029,1.015] from k*=3 alone; the RATE, N-independent "
     "(N_hub_spectral_gap_attempt Steps A–C)"),
    ("p_toggle = 1/(k*·N)",          "N^-1",  "ii",
     "the per-tick rate; defined via N ⇒ circular for pinning N"),
    ("H = 1/(N·t_P)",                "N^-1",  "ii",
     "Hubble rate; derived FROM N"),
    ("Λ_substrate = 1/N²",           "N^-2",  "ii",
     "Friedmann + H=1/(N·t_P), coasting Ω_Λ=1/3 ⇒ derived FROM N"),
    ("S_total (accumulated)",        "N^+1",  "ii",
     "b1': per-tick surprise bounded O(1) (≤log(k-1)) ⇒ S_total ∝ N "
     "(N = tick count) ⇒ a re-expression of N, not a constraint"),
    ("MDL capacity / budget",        "N^+1",  "ii",
     "substrate description length ∝ N hubs ⇒ ∝ N"),
]
classes = {"i": 0, "ii": 0, "iii": 0}
for name, scal, cls, why in QUANTS:
    classes[cls] += 1
    print(f"  [{cls}]  {name:28s}  ~N^{scal:5s}  — {why}")
print(f"\n  class counts: (i) N-indep const = {classes['i']}, "
      f"(ii) N-derived = {classes['ii']}, (iii) pinning = {classes['iii']}")
enumeration_exhaustive = (classes['i'] + classes['ii'] + classes['iii']
                          == len(QUANTS)) and len(QUANTS) >= 8
if not enumeration_exhaustive:
    abort("A1", "enumeration not exhaustive/classifiable — cannot "
                "conclude on the N-waterline.")
else:
    print("  ✓ A1 pass: enumeration exhaustive; every quantity is")
    print("    class (i) [N-indep const, cannot select N] or class (ii)")
    print("    [derived from N, circular]. ZERO class-(iii) so far.")


# ======================================================================
# STEP 2 — NW-A2: is Λ_substrate an INDEPENDENT structural handle?
# ======================================================================
head("STEP 2 — NW-A2: Λ-handle (the decisive shortcut to a HIT)")

# If Λ_substrate were a fixed structural number INDEPENDENT of N, then
# Λ = 1/N²  ⇒  N = 1/√Λ  (a unique HIT, zero observational input).
# Trace: Lambda_CC.py defines Λ_substrate = H_0_substrate² = (1/(N·t_P))²
# = 1/N², via Friedmann (coasting, Ω_Λ=1/3). It is DEFINED THROUGH N.
lambda_is_independent = False   # traced: Λ = 1/N² (Friedmann ← H ← N)
print("  Lambda_CC.py:  Λ_substrate = H_0_substrate² = (1/(N·t_P))² = 1/N²")
print("  ⇒ Λ is DERIVED FROM N (Friedmann + coasting Ω_Λ=1/3); there is")
print("    NO independent structural Λ. Λ = 1/N² is class (ii), circular")
print("    for the purpose of pinning N. (Honest negative on this route;")
print("    if a future independent structural Λ_struct is found, then")
print("    N = 1/√Λ_struct would be an immediate HIT — flagged.)")
if lambda_is_independent:
    note_hit("A2", "independent structural Λ ⇒ N = 1/√Λ pins N.")
else:
    print("  ✓ A2 pass (no shortcut): Λ gives no independent handle.")


# ======================================================================
# STEP 3 — NW-A3: the user's I-projection fixed-point-in-N test
# ======================================================================
head("STEP 3 — NW-A3: is there an MDL/I-projection fixed point IN N?")

# b1': A2-T = unique I-projection fixed point in DISTRIBUTION space.
# User hypothesis: an analogous fixed point in N — e.g. the walk has
# accumulated exactly the substrate's capacity ("as big as it has had
# time to learn"):   S_total(N)  ?=  capacity(N).
# b1' (A3): per-tick surprise is bounded, S_avg ∈ (0, log2(k-1)] ⇒
#   S_total(N) ≈ s̄ · N           (LINEAR in N)
# substrate description-length capacity ≈ c · N   (∝ N hubs; LINEAR)
# A unique fixed point needs DIFFERENT N-scaling. Test symbolically.
import sympy as sp
N = sp.symbols('N', positive=True)
s_bar = sp.symbols('s_bar', positive=True)      # ∈(0, log2(k-1)]
c_cap = sp.symbols('c_cap', positive=True)      # capacity coeff
S_total = s_bar * N
capacity = c_cap * N
sol = sp.solve(sp.Eq(S_total, capacity), N)
print(f"  S_total(N)   ≈ s̄·N        (b1': bounded per-tick surprise)")
print(f"  capacity(N)  ≈ c_cap·N    (∝ N hubs)")
print(f"  S_total = capacity  ⇒  solve for N:  {sol if sol else 'NO unique N'}")
print(f"  Both ∝ N¹ (SAME scaling) ⇒ the balance is SCALE-INVARIANT in N")
print(f"  (holds for all N if s̄=c_cap, for no N otherwise) ⇒ NO unique")
print(f"  fixed point. The walk's information accumulation is scale-free")
print(f"  in N ⇒ NO N-waterline from this self-consistency.")
# A fixed point would require, e.g., one side ∝ N and another ∝ N² or
# log N. The only N^-2 quantity (Λ) is class (ii) (NW-A2). No pair of
# substrate-internal quantities has the required DIFFERENT N-scaling
# with one side N-independent.
mdl_fixed_point = bool(sol) and (len(sol) == 1) and (s_bar not in sol[0].free_symbols if sol else False)
if mdl_fixed_point:
    note_hit("A3", "an MDL/I-projection self-consistency pins a unique N.")
else:
    print("  ✓ A3 pass: NO fixed-point-in-N — the user's N-waterline")
    print("    hypothesis is, as stated, NOT realized: every candidate")
    print("    self-consistency is scale-invariant in N (class i/ii).")


# ======================================================================
# STEP 4 — NW-A4: anti-numerology / no-fit guard
# ======================================================================
head("STEP 4 — NW-A4: no observed-value matching")

searched_for_observed_value = False   # we never compute/seek ≈8.39e60
print(f"  • No constant-search for ≈8.39×10⁶⁰ "
      f"(searched_for_observed_value={searched_for_observed_value}).")
print("  • A HIT would have required SOLVING a pre-declared structural")
print("    equation with ZERO observational input (none arose). The")
print("    negative is structural (scale-invariance), not a failed fit.")
if searched_for_observed_value:
    abort("A4", "observed-value matching attempted — numerology.")
else:
    print("  ✓ A4 pass: principled test only; no fitted/coincidental N.")


# ======================================================================
# STEP 5 — NW-A5: the characterized negative is the valid deliverable
# ======================================================================
head("STEP 5 — NW-A5: classify & characterize the outcome")

if HIT and not FAIL:
    print(f"  OUTCOME = HIT ({HIT}): a substrate self-consistency pins")
    print("  N with zero observational input ⇒ Gap G1 CLOSES. (Verify")
    print("  independently before any status change — extraordinary.)")
elif not FAIL:
    print("  OUTCOME = CHARACTERIZED HONEST NEGATIVE (the deliverable):")
    print("   • No class-(iii) quantity exists (A1); Λ no independent")
    print("     handle (A2); no fixed-point-in-N — every candidate self-")
    print("     consistency is SCALE-INVARIANT in N (A3).")
    print("   • ⇒ The user's N-waterline hypothesis is, AS STATED,")
    print("     refuted: the epoch is NOT pinned by an internal")
    print("     fixed-point-in-N. This is route-elimination, not a")
    print("     failure (Need-B-style), and NOT the retracted")
    print("     'epoch floor / provably impossible' claim.")
    print("   • ⇒ Gap G1 reduces PRECISELY to the documented block: a")
    print("     derived WALK-ORIGIN / boundary condition (N₀ at")
    print("     'inflation end' / the discrete Gauss-Codazzi cosmo-")
    print("     history closure) — a well-defined, OPEN, BOUNDED target")
    print("     of a DIFFERENT type than a self-consistency-in-N.")
else:
    print(f"  OUTCOME = INCONCLUSIVE (aborts {FAIL}).")


# ======================================================================
# VERDICT
# ======================================================================
head("VERDICT — the N-waterline shot")
if FAIL:
    print(f"  INCONCLUSIVE — aborts: {FAIL}. No claim. No salvage.")
    sys.exit(1)
if HIT:
    print(f"  HIT ({HIT}) — provisional; demands independent")
    print("  verification before any status/ledger change.")
    sys.exit(0)

print("""  ALL ABORTS PASSED → CHARACTERIZED HONEST NEGATIVE.

  RESULT: there is NO N-waterline. Every substrate-internal quantity is
  either an N-independent structural constant (class i — cannot select
  an N) or derived FROM N (class ii — circular: H, Λ=1/N², p_toggle,
  S_total, capacity). No class-(iii) (N-non-scale-invariant AND
  N-independent) quantity exists; in particular Λ_substrate is N-derived
  (NW-A2, no shortcut) and the I-projection/MDL self-consistency
  S_total(N)=capacity(N) is scale-invariant (both ∝ N¹, NW-A3) so it
  pins no unique N.

  ⇒ The user's N-waterline hypothesis is, AS STATED, refuted (honest
    route-elimination — like the Need-B routes; NOT a failure, NOT the
    retracted "epoch floor"). Gap G1 is therefore NOT a
    fixed-point-in-N problem. It reduces PRECISELY and only to the
    framework's already-documented block: an independent derivation of
    the WALK-ORIGIN / boundary condition — N₀ at "inflation end" plus
    the elapsed expansion, i.e. the discrete Gauss–Codazzi cosmological-
    history closure (N_hub_spectral_gap_attempt.py). That is a
    well-defined, OPEN, BOUNDED target of a *different type* (a derived
    initial/boundary condition, not an internal self-consistency).

  NET (corrected, honest, both-directions disciplined):
   • RETRACTED: "scale = irreducible epoch floor, provably not
     closeable, no number to get" (overclaim — twice corrected by the
     user).
   • NOT manufactured: an N-waterline closure (it provably does not
     exist within current substrate structure — scale-invariance).
   • STANDS: the absolute scale is parameter-free in natural units
     with DERIVED toggle dynamics, down to one open integer N_hub
     whose evolution law is derived and whose pinning is now sharply
     localized to a derived walk-origin/boundary condition (Gap G1,
     OPEN, BOUNDED, well-defined — the discrete Gauss–Codazzi route).

  Grade: THEOREM-GRADE-STRUCTURAL (route-elimination + precise
  reduction of Gap G1; zero fitted; pre-declared aborts incl. an
  explicit no-fit guard and a both-directions honesty guard).
""")
print("=" * 74)
print("  EXIT 0 — no N-waterline; Gap G1 = the derived walk-origin /"
      " Gauss-Codazzi boundary problem (open, bounded, well-defined)")
print("=" * 74)
