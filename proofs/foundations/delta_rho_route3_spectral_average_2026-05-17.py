#!/usr/bin/env python3
"""
delta_rho_route3_spectral_average_2026-05-17.py

δρ DISPERSIVE-RESUMMATION PROGRAM — Route 3, step 1 (capstone-§6).
The genuine bottom of the well: the NON-PERTURBATIVE object. Route 2
proved the +4.58% is not any perturbative insertion sum; the absorptive
√5/4 is the non-perturbative branch value. Route 3 tests the one
physically-correct non-perturbative observable for a state EMBEDDED IN
A CONTINUUM:

  the leading δρ uses the self-energy's absorptive part at the SINGLE
  on-shell point z_W=√3:  −Im g♯(√3) = √5/4  (the peak value).
  But a continuum-embedded state's PHYSICAL observable is the
  SPECTRAL-DENSITY-WEIGHTED AVERAGE over the whole cut, not the peak:

     ⟨F⟩ = ∫_cut (−Im g♯(μ))·w(μ) dμ ,   w = the substrate's OWN
            universal-cover spectral measure (Kesten–McKay), normalised.

  single-point-peak vs spectral-average IS exactly the
  perturbative-vs-non-perturbative distinction here. Parameter-free
  (g♯ = the theorem-grade cavity resolvent; w = ρ_KM, the substrate's
  own measure — both K-rational, already derived). §7.5-compliant
  (a spectral integral, NOT a single factor or a perturbative series).
  NOT in any prior probe (all used the single saddle point).

  q=k*−1=2, k=k*=3, α₁=(2/3)^8, cut μ∈[−2√q,2√q]=[−2√2,2√2],
  retarded branch √(μ²−4q)=+i√(4q−μ²) (gives −Im g♯(√3)=√5/4).
  Kesten–McKay (3-regular tree, the universal cover):
     ρ_KM(μ) = (k/2π)·√(4q−μ²)/(k²−μ²) = (3/2π)·√(8−μ²)/(9−μ²).
  Note Sokhotski–Plemelj: −Im g♯(μ) = π·ρ_NB(μ) is itself the spectral
  density ⇒ ⟨F⟩ is a clean spectral moment (peak vs mean of the SAME
  density) — no ext ernal weight invented.

  δρ_lead = (1/2)(√5/4)α₁ = +1.0906% (+4.58% vs obs +1.0429%).
  δρ_avg  = (1/2)·⟨F⟩·α₁.

PRE-DECLARED VERDICT (G1–G5 from the program doc):
  PASS-CANDIDATE iff δρ_avg within ~1% of obs, screening sign
    (δρ_avg<δρ_lead toward obs), ⟨F⟩ small-height K-rational, no
    tuning. (Then independent re-derivation required; NOT shipped.)
  NEG otherwise — honest deliverable: localizes further or establishes
    the residual is beyond closed form (⇒ Phase-3 adoption decision).
"""
import cmath, math
import numpy as np

def _trap(y, x):
    """version-safe trapezoid (np.trapz removed in newer numpy)."""
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else \
           float(np.sum((y[1:]+y[:-1])*0.5*np.diff(x)))

Q = 2.0
K = 3.0
ALPHA1 = (2.0/3.0)**8
DR_OBS = 0.0104286
SQRT5_4 = math.sqrt(5.0)/4.0
DR_LEAD = 0.5*SQRT5_4*ALPHA1
EDGE = 2.0*math.sqrt(Q)                       # 2√2, McKay band edge
def rel(x): return (x/DR_OBS - 1.0)*100.0

def g_sharp(z):
    z = complex(z)
    d = z*z - 4.0*Q
    s = 1j*cmath.sqrt(-d) if (d.real < 0 and abs(d.imag) < 1e-12) else cmath.sqrt(d)
    f = (z - s)/(2.0*Q)
    return 1.0/(z - K*f)

# control: on-shell peak value
F_peak = -g_sharp(math.sqrt(3.0)).imag
print(f"control: −Im g♯(√3) = {F_peak:.8f}  (√5/4={SQRT5_4:.8f})  "
      f"{'OK' if abs(F_peak-SQRT5_4)<1e-9 else 'FAIL'}")
print(f"δρ_lead (single-point peak) = {DR_LEAD*100:+.5f}%  "
      f"({rel(DR_LEAD):+.2f}% vs obs)\n")

# ---- full-cut spectral average (parameter-free) --------------------------
# integrate on the OPEN cut; endpoints are integrable (√ vanishes / KM √).
NMU = 2_000_001
mu = np.linspace(-EDGE, EDGE, NMU)[1:-1]      # drop exact edges (branch pts)
absF = np.array([-g_sharp(m).imag for m in mu])     # = π·ρ_NB(μ) ≥ 0

# (A) self-weighted spectral average: weight = the spectral density itself
#     (−Im g♯ ∝ ρ_NB) — the intrinsic "mean vs peak" of the resonance's
#     own spectral function. No external/invented weight.
wA = absF / _trap(absF, mu)
F_avgA = _trap(absF*wA, mu)

# (B) Kesten–McKay-weighted average (the universal cover's own measure)
rho_km = (K/(2*math.pi))*np.sqrt(np.clip(4*Q-mu**2,0,None))/(K*K-mu**2)
wB = rho_km/_trap(rho_km, mu)
F_avgB = _trap(absF*wB, mu)

for tag,Fa in (("A self-weighted ⟨−Im g♯⟩_{ρ_NB}",F_avgA),
               ("B Kesten–McKay ⟨−Im g♯⟩_{ρ_KM} ",F_avgB)):
    dr = 0.5*Fa*ALPHA1
    print(f"{tag}: ⟨F⟩={Fa:.6f}  (peak √5/4={SQRT5_4:.6f}; "
          f"Δ={(Fa-SQRT5_4)/SQRT5_4*100:+.2f}%)  "
          f"δρ={dr*100:+.5f}% ({rel(dr):+.2f}% vs obs)  "
          f"sign:{'SCREEN→obs' if dr<DR_LEAD else 'AWAY'}")

# ---- K-rationality of the averages (G2) ----------------------------------
def kmatch(x, tol=2.5e-4):
    rts={'1':1.,'√2':2**.5,'√3':3**.5,'√5':5**.5,'√6':6**.5,'√10':10**.5,
         '√15':15**.5,'π':math.pi,'1/π':1/math.pi}
    best=None
    for nm,r in rts.items():
        for p in range(-15,16):
            for d in range(1,61):
                v=p*r/d
                if abs(v-x)<tol and (best is None or abs(v-x)<best[0]):
                    best=(abs(v-x),f"{p}{nm}/{d}={v:+.6f}")
    return best
for tag,Fa in (("A",F_avgA),("B",F_avgB)):
    km=kmatch(Fa)
    print(f"   G2  ⟨F⟩_{tag} = {Fa:.7f} → "
          + (km[1] if km else "NONE small-height ⇒ not K-rational"))

# ---- verdict (pre-declared) ----------------------------------------------
print("\n" + "="*72)
res=[]
for tag,Fa in (("A",F_avgA),("B",F_avgB)):
    dr=0.5*Fa*ALPHA1
    res.append((tag,Fa,dr,abs(rel(dr))<1.2, dr<DR_LEAD, kmatch(Fa) is not None))
winner=[r for r in res if r[3] and r[4] and r[5]]
if winner:
    tag,Fa,dr,_,_,_=winner[0]
    print(f"VERDICT: PASS-CANDIDATE (Route 3) — the {tag} full-cut spectral")
    print(f"  average closes δρ to {rel(dr):+.2f}% of obs, screening sign,")
    print(f"  K-rational, parameter-free, §7.5-compliant. NOT shipped:")
    print(f"  requires independent closed-form re-derivation of ⟨F⟩ before")
    print(f"  any grade/number change. Hand to recording + independent check.")
else:
    print("VERDICT: NEG (Route 3 step 1) — the full-cut spectral average does")
    print("  NOT close +4.58% with screening sign + K-rational + no tuning.")
    print("  HONEST DELIVERABLE: the single-point→spectral-average")
    print("  (perturbative→non-perturbative) distinction is NOT the +4.58%.")
    print("  Combined with Routes 1–2 (all perturbative routes NEG) and the")
    print("  triple-lock, this exhausts the program's bounded/parameter-free")
    print("  routes: the +4.58% is a non-perturbative object with NO closed")
    print("  form reachable by the framework's substrate-native machinery —")
    print("  the honest terminal finding ⇒ Phase-3 adoption decision")
    print("  (A5(b)-class), the residual being a named, fully-characterized,")
    print("  parameter-free-unreachable non-perturbative correction. δρ")
    print("  stays +0.76σ_obs consistent; triple-lock intact; 0 number/grade.")
print("="*72)
