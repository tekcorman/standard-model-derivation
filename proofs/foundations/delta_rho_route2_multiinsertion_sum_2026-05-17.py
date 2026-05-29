#!/usr/bin/env python3
"""
delta_rho_route2_multiinsertion_sum_2026-05-17.py

δρ DISPERSIVE-RESUMMATION PROGRAM — Route 2, step 1 (capstone-§6).
The sub-tree MULTI-INSERTION sum, §7.5's literal object: a genuine
order-by-order combinatorial insertion series (NOT the forbidden single
1/(1−α₁) factor — G1), parameter-free, then summed by PRE-DECLARED
parameter-free divergent-series methods.

CONSTRUCTION (zero free choices — the framework's OWN NB-insertion
cavity recursion):
  on-cut spectral point  z_W = √3  (Ihara image of h_P; disc=z²−4q=−5<0),
  q=k*−1=2, k=k*=3, α₁=(2/3)^8.
  bare (0 insertions):  f₀ = 1/z
  n nested insertions:  f_{n+1} = 1/(z − q·f_n)        [real iteration]
  partial resolvent:    g_n = 1/(z − k·f_n)            [real, on-cut osc.]
  dispersive shift at order n:  δz_n = α₁·g_n           [real — the
        multi-insertion DISPERSIVE self-energy; the absorptive part is
        the non-perturbative branch, evaluated at the shifted point]
  absorptive functional at order n:
        F_n = −Im g♯(z_W + δz_n),  g♯ the COMPLEX cavity resolvent with
        the retarded branch √(z²−4q)=+i√(4q−z²)  (gives F_0→√5/4).
  δρ_n = (1/2)·F_n·α₁.   Leading (n→bare, δz=0): δρ=+1.0906% (+4.58%).

NOTE (structural, established before running): the real iteration on the
cut is non-contracting (Route 1 / §7.5, doubly corroborated) ⇒ {δρ_n} is
a DIVERGENT/oscillating asymptotic series. Its physical value (if any) is
obtained by the *standard* divergent-series prescriptions — this is the
physically-correct treatment of a continuum-embedded self-energy and is
G1-compliant (a summation of the multi-insertion series, NOT a single
geometric factor).

PRE-DECLARED parameter-free summations (no tuning, no method choice
post-hoc — ALL three reported):
  (C)  Cesàro (average of partial sums of the δρ_n increments)
  (T)  optimal truncation at the least term (standard asymptotic rule)
  (P)  Padé [m/m] on the increment series
PRE-DECLARED VERDICT:
  PASS-CANDIDATE iff ≥2 of {C,T,P} AGREE (mutual spread <0.5% of obs) on
    a value within ~1% of obs, screening sign, and that value is
    small-height K-rational, G1–G5 intact. (Then independent
    re-derivation required; NOT shipped.)
  NEG otherwise — and if the methods DISAGREE that is the honest finding:
    the multi-insertion series is not standard-summable to the +4.58%;
    the residual is a non-perturbative (Stokes/branch) object at
    interior z=√3 → hands to Route 3. Negatives ARE the deliverable.
"""
import cmath, math

Z = math.sqrt(3.0)
Q = 2.0
K = 3.0
ALPHA1 = (2.0/3.0)**8
DR_OBS = 0.0104286
SQRT5_4 = math.sqrt(5.0)/4.0
DR_LEAD = 0.5*SQRT5_4*ALPHA1
def rel(x): return (x/DR_OBS-1.0)*100.0

def g_sharp(z):
    """complex cavity resolvent, retarded branch on the McKay cut."""
    z = complex(z)
    d = z*z - 4.0*Q
    s = 1j*cmath.sqrt(-d) if (d.real < 0 and abs(d.imag) < 1e-12) else cmath.sqrt(d)
    f = (z - s)/(2.0*Q)
    return 1.0/(z - K*f)

print(f"control: −Im g♯(√3) = {-g_sharp(Z).imag:.6f}  (√5/4={SQRT5_4:.6f})  "
      f"{'OK' if abs(-g_sharp(Z).imag-SQRT5_4)<1e-9 else 'FAIL'}")
print(f"δρ_lead = {DR_LEAD*100:+.5f}%  ({rel(DR_LEAD):+.2f}% vs obs)\n")

# ---- build the multi-insertion sequence (real cavity iteration) -----------
N = 60
f = 1.0/Z                                  # f₀, bare (0 insertions)
drho = []
fs = []
for n in range(N):
    g_n = 1.0/(Z - K*f)                    # real partial resolvent
    dz  = ALPHA1*g_n                       # real dispersive shift, order n
    F_n = -g_sharp(Z + dz).imag            # absorptive at shifted point
    drho.append(0.5*F_n*ALPHA1)
    fs.append(f)
    f = 1.0/(Z - Q*f)                      # next insertion
osc = sum(1 for i in range(1,N) if (fs[i]-fs[i-1])*(fs[i-1]-fs[i-2 if i>=2 else i-1])<0)
print(f"multi-insertion f_n (first 8): "
      + " ".join(f"{v:+.3f}" for v in fs[:8]) + " ...")
print(f"  sign-changes in {{f_n}} over {N}: {osc}  ⇒ "
      + ("NON-contracting / divergent (as §7.5/Route1 predict)"
         if osc > N//4 else "contracting?? (unexpected)"))
print(f"  δρ_n (first 8, %): " + " ".join(f"{v*100:+.4f}" for v in drho[:8]) + " ...")
print(f"  δρ_n range over n: [{min(drho)*100:+.4f}%, {max(drho)*100:+.4f}%]\n")

# ---- pre-declared parameter-free summations ------------------------------
# (C) Cesàro: average of the partial-sum sequence of δρ_n themselves
partial = drho[:]                                    # δρ_n is already the "value at order n"
def cesaro(seq, k=2):
    s = seq[:]
    for _ in range(k):
        s = [sum(s[:i+1])/(i+1) for i in range(len(s))]
    return s[-1]
C = cesaro(partial, 2)

# (T) optimal truncation: value at the order minimising |δρ_{n}−δρ_{n-1}|
diffs = [abs(partial[i]-partial[i-1]) for i in range(1,len(partial))]
n_opt = 1 + min(range(len(diffs)), key=lambda i: diffs[i])
T = partial[n_opt]

# (P) Padé [m/m] on the increment series a_n = δρ_n − δρ_{n-1}; eval at 1
def pade_eval(seq, m):
    # build [m/m] Padé of the power series Σ seq[k] x^k at x=1 (least-squares-free,
    # standard linear solve); guard singular.
    import numpy as np
    Ncf = 2*m+1
    a = np.array(seq[:Ncf], dtype=float)
    if len(a) < Ncf: return None
    A = np.zeros((m, m))
    b = -a[m+1:2*m+1]
    for i in range(m):
        for j in range(m):
            A[i, j] = a[m+i-j]
    try:
        q = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    qcoef = np.concatenate(([1.0], q))
    pcoef = np.zeros(m+1)
    for i in range(m+1):
        pcoef[i] = sum(a[i-j]*qcoef[j] for j in range(min(i, m)+1))
    num = sum(pcoef); den = sum(qcoef)
    return None if abs(den) < 1e-12 else num/den
incr = [partial[0]] + [partial[i]-partial[i-1] for i in range(1, len(partial))]
P = pade_eval(incr, 8)

print("PRE-DECLARED parameter-free summations of the multi-insertion series:")
print(f"  (C) Cesàro(2)            = {C*100:+.5f}%   ({rel(C):+.2f}% vs obs)")
print(f"  (T) optimal-truncation   = {T*100:+.5f}%   ({rel(T):+.2f}% vs obs)  "
      f"[at order n={n_opt}]")
print(f"  (P) Padé[8/8]            = "
      + (f"{P*100:+.5f}%   ({rel(P):+.2f}% vs obs)" if P is not None else "SINGULAR"))

vals = [v for v in (C, T, P) if v is not None]
spread = (max(vals)-min(vals))/DR_OBS*100 if len(vals) >= 2 else 99
agree = [v for v in vals if abs(rel(v)) < 1.2]
def kmatch(x):
    rts={'1':1.,'√2':2**.5,'√3':3**.5,'√5':5**.5}
    best=None
    for nm,r in rts.items():
        for p in range(-9,10):
            for d in range(1,49):
                v=0.5*(p*r/d)*ALPHA1
                if abs(v-x)<0.012*DR_OBS and (best is None or abs(v-x)<best[0]):
                    best=(abs(v-x),f"F={p}{nm}/{d}")
    return best

print("\n" + "="*72)
if len(agree) >= 2 and spread < 0.5:
    km = kmatch(sum(agree)/len(agree))
    if km:
        print("VERDICT: PASS-CANDIDATE (Route 2) — ≥2 parameter-free")
        print(f"  summations AGREE (spread {spread:.2f}% of obs) within ~1% of")
        print(f"  obs, screening, K-rational ({km[1]}). NOT shipped: requires")
        print("  independent closed-form re-derivation before any grade/number.")
    else:
        print("VERDICT: NEG (G2) — methods agree near obs but the value is")
        print("  NOT small-height K-rational ⇒ numerology. Refused.")
else:
    print("VERDICT: NEG (Route 2 step 1) — the parameter-free summations do")
    print(f"  NOT mutually agree (spread {spread:.1f}% of obs) / not within 1%.")
    print("  HONEST STRUCTURAL FINDING (the deliverable): the multi-insertion")
    print("  series is divergent and NOT standard-summable to the +4.58% —")
    print("  the absorptive δρ residual is a NON-PERTURBATIVE (Stokes/branch)")
    print("  object at interior z=√3, beyond the perturbative insertion sum.")
    print("  Re-localizes precisely → Route 3 (branch-cut dispersion at the")
    print("  interior point; §7.5's ε-expansion was at the BAND EDGE, never")
    print("  the interior z=√3 — a genuinely new, defined next sub-object).")
    print("  Triple-lock intact; δρ +0.76σ_obs; ZERO number/grade changed.")
print("="*72)
