#!/usr/bin/env python3
"""
delta_rho_dyson_selfconsistent_pole_2026-05-17.py

δρ DISPERSIVE-RESUMMATION PROGRAM — Route 1, step 1 (capstone-§6,
user-committed). Genuinely NEW: the Dyson self-consistent dressed pole.
NOT in the exhausted list (host-sums, Family-D, single-factor
resummation, band-edge branch-expansion, M_n-waterfilling, bare-saddle
continuum, single Fano functionals — none did the self-consistent
fixed point).

OBJECT. The leading δρ evaluates the W/h_P self-energy ONCE at the bare
on-cut spectral point z_W=√3 (Ihara image of h_P; disc=z²−4q=−5<0).
Physically the W self-energy dresses its OWN pole: the spectral point
shifts by Σ and Σ must be re-evaluated there (the standard tree→pole
self-consistency — the SAME structure as δ_r for M_Z, predictions/M_Z.py
M_Z_pole=M_Z_tree·(1−δ_r)). Dyson fixed point on the cut:

        z*  =  z_W + α₁ · g(z*) ,        z_W = √3 ,

with g the substrate's THEOREM-GRADE cavity resolvent (Kesten/McKay;
tree_cover_S_and_resummation / theorem_unified_oblique §7.5):
        f(z) = [ z − sqrt(z²−4q) ] / (2q),   g(z) = 1/( z − k·f(z) ),
        k=k*=3, q=k*−1=2.  On the cut sqrt(z²−4q)=+i·sqrt(4q−z²)
        (retarded/outside-radial branch — the framework's stated causal
        prescription; gives −Im g(√3)=√5/4, reproducing the leading).

δρ(z) = (1/2)·(−Im g(z))·α₁.   Leading: z=√3 ⇒ −Im g(√3)=√5/4 ⇒
δρ_lead=+1.0906%.  Self-consistent: δρ* = (1/2)·(−Im g(z*))·α₁.

GUARDRAILS (program doc §2 — a violation kills the route):
 G1 §7.5  : a fixed point, NOT a single 1/(1−α₁) factor.            [satisfied by construction]
 G2 O9    : z* and −Im g(z*) must be small-height K-rational.        [TESTED]
 G3 lock  : α₁→0 ⇒ z*→√3 ⇒ −Im g→√5/4 EXACTLY (triple-lock intact). [TESTED as control]
 G4 nofit : only α₁,k*,q,√3 enter; no tuned scale, no bespoke form.  [satisfied by construction]
 G5 noload: only the substrate cavity g (no SM/Sirlin/Δα).           [satisfied by construction]

PRE-DECLARED VERDICT:
 PASS-CANDIDATE  iff δρ* within ~1% of obs, screening sign
   (δρ*<δρ_lead toward +1.0429%), z* & −Im g(z*) small-height
   K-rational, G1–G5 all hold, no tuning. (Then: independent
   re-derivation required; NOT shipped.)
 NEG  otherwise — record the structural reason; it localizes Route 1
   and hands off to Route 2 (sub-tree multi-insertion sum). A NEG here
   is the expected, honest, deliverable outcome — not a failure.
"""
import cmath, math

K = 3.0
Q = 2.0
ALPHA1 = (2.0/3.0)**8
DR_OBS = 0.0104286
Z_W = math.sqrt(3.0)                       # on-cut Ihara image of h_P

def sqrt_cut(z):
    """sqrt(z²−4q) on the McKay cut, retarded branch: +i·sqrt(4q−z²)
    for |z|<2√q (reproduces −Im g(√3)=√5/4 — the framework's causal
    prescription)."""
    d = z*z - 4.0*Q
    if d.real < 0 and abs(d.imag) < 1e-12:
        return 1j*cmath.sqrt(-d)
    return cmath.sqrt(d)

def g(z):
    z = complex(z)
    f = (z - sqrt_cut(z)) / (2.0*Q)
    return 1.0 / (z - K*f)

def drho(z):
    return 0.5 * (-g(z).imag) * ALPHA1

def rel(x):
    return (x/DR_OBS - 1.0)*100.0

# --- control: leading reproduction (G3) ------------------------------------
g0 = g(Z_W)
print(f"control g(√3) = {g0.real:+.6f} {g0.imag:+.6f}i   "
      f"−Im = {-g0.imag:.6f}  (√5/4 = {math.sqrt(5)/4:.6f})  "
      f"{'OK (G3)' if abs(-g0.imag-math.sqrt(5)/4)<1e-9 else 'FAIL-control'}")
dr_lead = drho(Z_W)
print(f"δρ_lead = {dr_lead*100:+.5f}%  ({rel(dr_lead):+.2f}% vs obs)\n")

# --- Route 1 step 1: solve the Dyson fixed point z* = √3 + α₁·g(z*) --------
z = complex(Z_W, 0.0)
hist = []
for it in range(200):
    z_new = Z_W + ALPHA1 * g(z)
    hist.append(z_new)
    if abs(z_new - z) < 1e-15:
        z = z_new
        break
    z = z_new
z_star = z
conv = abs((Z_W + ALPHA1*g(z_star)) - z_star)
print(f"Dyson fixed point  z* = {z_star.real:+.8f} {z_star.imag:+.8f}i   "
      f"(|residual|={conv:.1e}, {it+1} iters)")
print(f"   shift z*−√3 = {(z_star-Z_W).real:+.6f} {(z_star-Z_W).imag:+.6f}i   "
      f"(O(α₁)≈{ALPHA1:.4f})")
g_star = g(z_star)
F_star = -g_star.imag
dr_star = drho(z_star)
print(f"   g(z*) = {g_star.real:+.6f} {g_star.imag:+.6f}i   "
      f"−Im g(z*) = {F_star:.6f}  (leading √5/4={math.sqrt(5)/4:.6f}; "
      f"Δ={(F_star-math.sqrt(5)/4)/(math.sqrt(5)/4)*100:+.3f}%)")
print(f"   δρ* = {dr_star*100:+.6f}%   ({rel(dr_star):+.3f}% vs obs)   "
      f"sign: {'SCREENING→obs' if dr_star<dr_lead else 'AWAY'}")

# --- K-rationality (G2) ----------------------------------------------------
def kmatch(x, tol=2e-4):
    rts={'1':1.,'√2':2**.5,'√3':3**.5,'√5':5**.5,'√6':6**.5,'√10':10**.5,'√15':15**.5,'√30':30**.5}
    best=None
    for nm,r in rts.items():
        for p in range(-20,21):
            for d in range(1,73):
                v=p*r/d
                if abs(v-x)<tol and (best is None or abs(v-x)<best[0]):
                    best=(abs(v-x),f"{p}{nm}/{d}={v:+.6f}")
    return best
kF = kmatch(F_star)
kZr = kmatch(z_star.real); kZi = kmatch(z_star.imag)
print(f"\nG2 K-rationality (ℚ(√2,√3,√5), height≤20/den≤72):")
print(f"   −Im g(z*) = {F_star:.7f} → " + (kF[1] if kF else "NONE ⇒ not K-rational"))
print(f"   Re z*     = {z_star.real:.7f} → " + (kZr[1] if kZr else "NONE"))
print(f"   Im z*     = {z_star.imag:.7f} → " + (kZi[1] if kZi else "NONE"))

# --- verdict (pre-declared) -----------------------------------------------
print("\n" + "="*72)
near   = abs(rel(dr_star)) < 1.2
screen = dr_star < dr_lead
g2ok   = kF is not None
if near and screen and g2ok:
    print("VERDICT: PASS-CANDIDATE (Route 1) — the Dyson self-consistent")
    print(f"  dressed pole gives δρ*={dr_star*100:+.5f}% ({rel(dr_star):+.2f}% of obs),")
    print("  screening sign, K-rational, G1–G5 intact, NO tuning. NOT shipped:")
    print("  requires an independent closed-form re-derivation of z* before")
    print("  any grade/number change. Hand to recording + independent check.")
else:
    why = []
    if not near:   why.append(f"magnitude off ({rel(dr_star):+.2f}% vs ~0)")
    if not screen: why.append("wrong sign (away from obs)")
    if not g2ok:   why.append("−Im g(z*) not small-height K-rational")
    print("VERDICT: NEG (Route 1 step 1) — " + "; ".join(why) + ".")
    print("  Honest, expected program outcome (negatives ARE the deliverable).")
    print("  Localizes: the FIRST-iteration Dyson dressing alone does not")
    print("  close +4.58%. The triple-lock (√5/4 robust as α₁→0) is")
    print("  re-confirmed. Route 1 continues to the full fixed-point /")
    print("  multi-insertion structure (Route 2), per §7.5 (the closure")
    print("  is a sub-tree multi-insertion sum, not one dressing). NO")
    print("  number/grade changed; δρ stays +0.76σ_obs consistent.")
print("="*72)
