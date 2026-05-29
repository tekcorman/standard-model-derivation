#!/usr/bin/env python3
"""
proofs/foundations/majorana_phase_deltaL2_perron_2026-05-19.py

PROBE 9 — discharge-or-terminus, AND honest test of the factorization
bridge I overstated in the 2026-05-19 commit.

HONEST PREMISE CORRECTION: probes 8/8a amplitudes are i^Y * omega^(wY) —
purely finite-group, NO arg(h) in them. So the committed claim
"Majorana phase = [discrete DeltaL=2 holonomy] x [spectral arg(h) factor]"
is an UNVERIFIED BRIDGE, not a computed result. The locked discrete
DeltaL=2 holonomy is real; its relation to the PHYSICAL value 162.39
(= g*arg(h), the actual P35 prediction) is what this probe tests.

OBJECT: the DeltaL=2-constrained non-backtracking transfer operator =
the Y-twisted Bloch-Hashimoto B(t) := A_H(k=(t,t,t)), where Y=cx+cy+cz
and the body-diagonal Bloch phase e^{2pi i t Y} is the Donsker-Varadhan
tilt conjugate to the hypercharge Y (Type-3 citable: large-deviation
tilting of a transfer operator; Ihara-Bass for the per-t spectrum). The
U-wall = Ramanujan degeneracy of the LEADING |eig| at the untwisted
point (no isolated Perron mode -> length-summed phase drifts). HYPOTHESIS
(user): the DeltaL=2 constraint LIFTS that degeneracy -> a unique Perron
mode whose argument is arg(h), cutoff-free.

THE TEST: scan t; find where (if anywhere) the leading eigenvalue is
ISOLATED (|mu_max| strictly > |mu_2|, a real spectral gap) AND
arg(mu_max) = arg(h) = 52.2388 deg; check whether that t is the
structurally-forced P-point t=1/4 and/or the DeltaY=+-2 saddle
(d|mu_max|/dt = 0). Then test the BRIDGE: does
   g*arg(mu_max(t*))  combine with the 8a-locked discrete phase
to reproduce the physical alpha_21=162.39 / alpha_31=324.78?

CORRECTNESS GATES (VOID if fail):
  G1  at t=1/4, k=(1/4,1/4,1/4)=P: reproduce the known P-point leading
      eigen-args 52.239 / 127.761 (prior probes) -> operator built right.
  G2  the UNTWISTED-type regime must exhibit the Ramanujan band
      (>=2 eigenvalues with |mu| ~ sqrt(k*-1)=sqrt2) -> the U-wall is
      really there to be (or not be) lifted.

PRE-DECLARED OUTCOMES:
  DISCHARGE : there is a STRUCTURALLY-FORCED t* (P-point and/or the
              DeltaY=+-2 saddle) where mu_max is ISOLATED (clean gap,
              Ramanujan degeneracy lifted) and g*arg(mu_max(t*)) composed
              with the 8a-locked discrete phase = 162.39/324.78 within
              ~1 deg => the bridge HOLDS; arg(h) IS the DeltaL=2 Perron
              eigenphase; ADOPTED-NU-MAJ-PHASE fully discharged.
  TERMINAL  : no such isolated t* (Ramanujan degeneracy survives the
   + CORRECT   DeltaL=2 constraint) OR the composition != physical value
   DOCS       => the factorization BRIDGE FAILS. The locked discrete
              DeltaL=2 holonomy stands alone but does NOT compose to the
              physical 162.39; the committed "x spectral arg(h)" wording
              MUST be corrected to "discrete DeltaL=2 holonomy locked;
              physical-value bridge OPEN; spectral factor fully adopted."
Ships no number into predictions/; changes no ledger row (the doc
correction, if triggered, is a separate explicit step).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import ATOMS, A_PRIM, N_ATOMS, find_bonds

np.set_printoptions(precision=5, suppress=True)
SQRT3,SQRT5=np.sqrt(3.),np.sqrt(5.)
h=(SQRT3+1j*SQRT5)/2.; g=10; k_star=3
ARGH=np.degrees(np.angle(h))%360                 # 52.2388
A21=(g*ARGH)%360                                  # 162.388
print(f"arg(h)={ARGH:.4f}  g*arg(h)=alpha_21={A21:.3f}  |h|={abs(h):.6f} "
      f"(sqrt(k*-1)={np.sqrt(k_star-1):.6f}; Ramanujan saturation)")

bonds=find_bonds(); n=len(bonds)
def AH(k):
    M=np.zeros((n,n),complex)
    for j,(sj,tj,dcj) in enumerate(bonds):
        for i,(si,ti,dci) in enumerate(bonds):
            if sj!=ti: continue
            if tj==si and tuple(int(dci[d])+int(dcj[d]) for d in range(3))==(0,0,0): continue
            M[j,i]=np.exp(2j*np.pi*np.dot(k,dci))
    return M

def spec(t):
    ev=np.linalg.eigvals(AH((t,t,t)))
    idx=np.argsort(-np.abs(ev))
    ev=ev[idx]
    mags=np.abs(ev)
    gap=mags[0]-mags[1]
    return ev, mags, gap

# ---- GATES ----
print("\n"+"="*72)
print("GATES — G1 reproduce P-point eigen-args at t=1/4; G2 Ramanujan band")
print("="*72)
evP,magP,gapP=spec(0.25)
argsP=sorted({round(np.degrees(np.angle(e))%360,3) for e in evP if abs(e)>1e-6})
print(f"  t=1/4 (=P): |mu| top4={magP[:4]}  leading-gap={gapP:.4e}")
print(f"             distinct eig-args (deg) = {argsP}")
g1 = any(abs(a-52.239)<0.5 for a in argsP) and any(abs(a-127.761)<0.5 for a in argsP)
# Ramanujan band: at a generic tilt, >=2 eigenvalues near |mu|=sqrt2
evg,magg,gapg=spec(0.137)
band=int(np.sum(np.abs(magg-np.sqrt(2))<0.05))
g2 = band>=2
print(f"  G1(P-args 52.24 & 127.76 present)={g1}   "
      f"G2(>=2 |mu|~sqrt2 at generic t: band={band})={g2}")
if not (g1 and g2):
    print("\n  ** GATE FAILED — operator/frame unsound. VOID. **"); sys.exit(0)
print("  GATES PASSED.")

# ---- scan: where is mu_max ISOLATED, and what is its arg? ----
print("\n"+"="*72)
print("SPECTRUM vs Y-tilt t  (k=(t,t,t); t=1/4 is the P-point)")
print("  isolated <=> leading |mu|-gap not ~0 (Ramanujan degeneracy lifted)")
print("="*72)
ts=np.linspace(0.0,0.5,26)
rows=[]
for t in ts:
    ev,mg,gp=spec(t)
    amax=np.degrees(np.angle(ev[0]))%360
    rows.append((t,mg[0],gp,amax))
    iso = "ISOLATED" if gp>1e-3 else "degenerate"
    star=""
    if abs(t-0.25)<1e-9: star=" <- P-point"
    if abs(amax-ARGH)<0.5 or abs(amax-(360-ARGH))<0.5 or abs(amax-127.761)<0.5:
        star+=" [arg~+-arg(h)]"
    print(f"  t={t:5.3f}: |mu_max|={mg[0]:.5f} gap={gp:8.2e} {iso:10s} "
          f"arg(mu_max)={amax:8.3f}{star}")

# ---- DeltaY=+-2 saddle: extremum of |mu_max(t)| ----
fine=np.linspace(1e-4,0.5-1e-4,400)
mm=np.array([spec(t)[1][0] for t in fine])
isad=int(np.argmax(mm)); tsad=fine[isad]
ev_s,mg_s,gp_s=spec(tsad)
arg_s=np.degrees(np.angle(ev_s[0]))%360
print("\n"+"="*72)
print("DeltaY=+-2 SADDLE (stationary |mu_max|, L->inf microcanonical pt)")
print("="*72)
print(f"  t_saddle={tsad:.5f}  |mu_max|={mg_s[0]:.5f}  gap={gp_s:.3e}  "
      f"({'ISOLATED' if gp_s>1e-3 else 'DEGENERATE'})  arg(mu_max)={arg_s:.3f}")
print(f"  P-point t=1/4: gap={gapP:.3e} "
      f"({'ISOLATED' if gapP>1e-3 else 'DEGENERATE'})  "
      f"arg(mu_max@P)={np.degrees(np.angle(evP[0]))%360:.3f}")

# ---- THE BRIDGE TEST ----
print("\n"+"="*72)
print("BRIDGE TEST — does g*arg(mu_max) at a structurally-forced t")
print("              reproduce the physical alpha_21=162.39 (P35)?")
print("="*72)
disch=False
for label,t,ev_,gp_ in [("P-point t=1/4",0.25,evP,gapP),
                        (f"DeltaY-saddle t={tsad:.4f}",tsad,ev_s,gp_s)]:
    am=np.degrees(np.angle(ev_[0]))%360
    comp=(g*am)%360
    isol = gp_>1e-3
    d21=abs((comp-A21+180)%360-180)
    print(f"  {label}: arg(mu_max)={am:.4f}  g*arg={comp:.3f}  "
          f"|gap|={gp_:.2e} {'ISOLATED' if isol else 'DEGENERATE'}  "
          f"vs alpha_21={A21:.3f} (off {d21:.3f})")
    if isol and d21<1.0: disch=True

print("\n"+"="*72); print("  VERDICT"); print("="*72)
if disch:
    V=("DISCHARGE — a structurally-forced t* has an ISOLATED Perron mode "
       "with g*arg(mu_max)=alpha_21; arg(h) IS the DeltaL=2 Perron "
       "eigenphase; bridge HOLDS; ADOPTED-NU-MAJ-PHASE dischargeable.")
else:
    V=("TERMINAL + MUST-CORRECT-DOCS — no structurally-forced isolated "
       "Perron t* reproduces alpha_21 (Ramanujan degeneracy survives the "
       "DeltaL=2 constraint and/or composition != physical value). The "
       "factorization BRIDGE FAILS. Committed 'x spectral arg(h)' wording "
       "must be corrected: locked discrete DeltaL=2 holonomy stands alone; "
       "physical-value bridge OPEN; spectral factor fully adopted.")
print("  "+V)
print("  Ships no number; changes no ledger row.")
print("="*72)
