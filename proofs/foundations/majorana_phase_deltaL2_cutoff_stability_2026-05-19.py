#!/usr/bin/env python3
"""
proofs/foundations/majorana_phase_deltaL2_cutoff_stability_2026-05-19.py

PROBE 8a — lock (or break) the derived discrete factor.

Probe 8 found the DeltaL=2 (DeltaY=+-2, Y=Vx+Vy+Vz the C3-invariant U(1))
class holonomy is class-constant & enantiomer-signed at length g=10 -> a
candidate cutoff-free discrete factor of the Majorana phase. This probe
tests whether that survives the cutoff: compute it at L=g AND L=2g.

KEY DESIGN: the full summed amplitude carries BOTH the discrete DeltaL=2
factor AND the universal spectral arg(h)-per-step rotation (Y-independent).
To isolate the DISCRETE factor we test Y-class DIFFERENCES in which the
universal spectral rotation cancels:
  D_enantio(L)  = arg S(+2) - arg S(-2)        (enantiomer; spectral cancels)
  D_majdirac(L) = arg S(+2) - arg S(0)         (Majorana vs Dirac; ditto)
A genuine cutoff-free discrete holonomy => D_*(g) == D_*(2g).

Memory-light: accumulate per-(Y,channel) complex sums on the fly (no walk
storage). Channels: abelian Bloch, C3-omega, C3-omega^2.

CORRECTNESS GATE (VOID if fails): the on-the-fly accumulator must
reproduce probe 8's L=10 result: Y=+2 omega-arg ~ 60, Y=-2 ~ 300,
abelian-Bloch 180 for |Y|=2 / 0 for Y=0. Verify before trusting L=20.

PRE-DECLARED:
  LOCK     : D_enantio and D_majdirac are length-invariant (g vs 2g,
             within 1 deg) AND class structure preserved => the derived
             discrete DeltaL=2 / enantiomer factor is genuinely cutoff-
             free; the U-wall escape is real; REDUCTION stands & is locked.
  COLLAPSE : D_* drift with length => length-g class-constancy was a
             coincidence; DeltaL=2 sector does NOT escape the U-wall;
             REDUCTION downgrades toward NEGATIVE (honest retraction).
Ships no number into predictions/; changes no ledger row.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds

g=10
P=np.array([0.25,0.25,0.25])
om=np.exp(2j*np.pi/3)
bonds=find_bonds()
adj={}
for (s,t,c) in bonds: adj.setdefault(s,[]).append((t,tuple(c)))

def accumulate(L, SUP):
    """Sum over length-L non-backtracking walks atom0->atom0.
       Returns dict: acc[Y][ch] = complex summed amplitude,
       ch in ('ab','w1','w2'); Y = net voltage component sum."""
    acc={}
    def add(Y, ph, w):
        d=acc.setdefault(Y,{'ab':0j,'w1':0j,'w2':0j})
        d['ab']+=ph
        d['w1']+=ph*w
        d['w2']+=ph*np.conj(w)
    # DFS carrying: atom, cell, prev(atom,cell), depth,
    #   bloch (complex prod of exp(2pi i P.c)),
    #   wpow (omega^(sum c mod 3) as complex), Ysum (int)
    def dfs(a, cell, prev, d, bloch, wpow, Ysum):
        if d==L:
            if a==0:
                add(Ysum, bloch, wpow)
            return
        for (t,c) in adj.get(a,()):
            nc=(cell[0]+c[0],cell[1]+c[1],cell[2]+c[2])
            if any(abs(x)>SUP for x in nc): continue
            if prev==(t,nc): continue
            sc=c[0]+c[1]+c[2]
            dfs(t, nc, (a,cell), d+1,
                bloch*np.exp(2j*np.pi*(P[0]*c[0]+P[1]*c[1]+P[2]*c[2])),
                wpow*(om**(sc%3)),
                Ysum+sc)
    dfs(0,(0,0,0),None,0,1.0+0j,1.0+0j,0)
    return acc

def argd(z): return (np.degrees(np.angle(z))%360) if abs(z)>1e-9 else float('nan')

# ---- correctness gate at L=g ----
print("="*70)
print(f"CORRECTNESS GATE — reproduce probe 8 at L=g={g}")
print("="*70)
ag=accumulate(g, SUP=4)
for Y in (-2,0,2):
    if Y in ag:
        d=ag[Y]
        print(f"  Y={Y:+d}: n-amp |ab|={abs(d['ab']):.3f} "
              f"ab-arg={argd(d['ab']):7.2f}  w1-arg={argd(d['w1']):7.2f}  "
              f"w2-arg={argd(d['w2']):7.2f}")
gate = (2 in ag and -2 in ag and 0 in ag
        and abs(argd(ag[2]['ab'])-180)<1 and abs(argd(ag[0]['ab'])-0)<1
        and abs(argd(ag[2]['w1'])-60)<1 and abs(argd(ag[-2]['w1'])-300)<1)
print(f"  GATE = {'PASS' if gate else 'FAIL'} "
      f"(expect ab|Y=2|=180, ab|Y=0|=0, w1 Y=+2~60 / Y=-2~300)")
if not gate:
    print("\n  ** GATE FAILED — accumulator disagrees with probe 8. VOID. **")
    sys.exit(0)

def diffs(acc, ch):
    if not all(k in acc for k in (2,-2,0)): return None
    De=(argd(acc[2][ch])-argd(acc[-2][ch]))%360
    Dm=(argd(acc[2][ch])-argd(acc[0][ch]))%360
    Dr=argd(acc[2][ch]/acc[-2][ch]) if abs(acc[-2][ch])>1e-9 else float('nan')
    return De,Dm,Dr

print("\n"+"="*70)
print(f"CUTOFF TEST — discrete factor at L=g={g} vs L=2g={2*g}")
print("  (Y-class differences cancel the universal spectral rotation)")
print("="*70)
a2=accumulate(2*g, SUP=6)
rows=[]
for Lname,acc in [(f"L=g={g}",ag),(f"L=2g={2*g}",a2)]:
    line=f"  {Lname:10s}: "
    cls=sorted(acc)
    line+=f"Y-classes={cls[:3]}...{cls[-3:]}  "
    for ch in ('w1','w2'):
        r=diffs(acc,ch)
        if r: line+=f"[{ch}] D_enantio={r[0]:7.2f} D_majdirac={r[1]:7.2f}  "
    rows.append((Lname,acc,line))
    print(line)

# compare g vs 2g
print("\n"+"="*70); print("  STABILITY VERDICT"); print("="*70)
stable=True; detail=[]
for ch in ('w1','w2','ab'):
    rg=diffs(ag,ch); r2=diffs(a2,ch)
    if rg is None or r2 is None:
        detail.append(f"  {ch}: class missing at one length"); stable=False; continue
    dDe=abs((rg[0]-r2[0]+180)%360-180)
    dDm=abs((rg[1]-r2[1]+180)%360-180)
    ok=dDe<1.0 and dDm<1.0
    stable&=ok
    detail.append(f"  {ch}: D_enantio g={rg[0]:.2f} 2g={r2[0]:.2f} (d={dDe:.2f}) | "
                  f"D_majdirac g={rg[1]:.2f} 2g={r2[1]:.2f} (d={dDm:.2f})  "
                  f"{'STABLE' if ok else 'DRIFT'}")
for d in detail: print(d)
V = "LOCK (discrete DeltaL=2/enantiomer factor is genuinely cutoff-free)" if stable \
    else "COLLAPSE (discrete factor drifts with cutoff; U-wall not escaped)"
print(f"\n  GATE=PASS  -> {V}")
print("  Ships no number; changes no ledger row.")
print("="*70)
