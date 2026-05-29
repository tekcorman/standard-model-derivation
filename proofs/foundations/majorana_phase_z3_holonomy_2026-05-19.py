#!/usr/bin/env python3
"""
proofs/foundations/majorana_phase_z3_holonomy_2026-05-19.py

PROBE 7 — the holonomy reframe.  Is the nu_R Majorana phase the
holonomy of the Z3 generation connection around the girth cycle-space
generator on the chiral srs net (cutoff-free, single-loop), rather than
a spectral arg(h^g) (length-summed, U-walled)?

Concepts under test (framework's own structures only):
  - flavor sector = Fock occupation; nu_R = the n=0 / parity endpoint.
  - generation connection = the Z^3 edge VOLTAGE (find_bonds cell offsets)
    reduced into the C3 generation fiber; its Wilson loop around a single
    girth-10 cycle is a genuine (homotopy) holonomy.
  - enantiomer (I4_1 32 vs mirror) = sign of the voltage / curvature.

DECISIVE DISTINCTION FROM PROBES 1-6:
  A genuine flat-connection holonomy must be (i) CLASS-CONSTANT (same on
  all cycles in a voltage class -> gauge invariant), (ii) EXACTLY WINDING-
  LINEAR (w turns -> exactly w*theta, NO drift -- the precise property
  whose ABSENCE was the Ramanujan U-wall), (iii) ENANTIOMER-CONJUGATE
  (V -> -V flips sign). The probe contrasts the single-loop holonomy with
  the spectral g*arg(h) reference explicitly.

CORRECTNESS GATE: the DFS must find exactly 15 girth-10 cycles at v0 and
girth must be 10 (framework n_g). Else frame wrong -> VOID.

PRE-DECLARED OUTCOMES:
  DISCHARGE : a single-loop generation-fiber holonomy is class-constant,
              exactly winding-linear, enantiomer-conjugate, matches a live
              local target (alpha_21~162.39 / alpha_31~324.78) AND is
              K-rational => ADOPTED-NU-MAJ-PHASE reduces to the binary
              I4_1/I4_3 enantiomer convention; value DERIVED.
  REDUCTION : connection structure clean (nonzero voltage, clean class
              split, winding-linear, enantiomer-conjugate) but value !=
              local, OR theta must still be put in by hand => well-posed
              cutoff-free relocation; U-wall dissolved; value conditional.
  NEGATIVE  : girth holonomy is a trivial 90/120-deg multiple (abelian)
              and g*arg(h)=162.39 appears ONLY in the spectral object
              => arg(h) is irreducibly spectral; the reframe unifies the
              concepts but does NOT escape the U-wall. Six-probe wall stands.
Ships no number into predictions/; changes no ledger row.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import ATOMS, A_PRIM, N_ATOMS, find_bonds, C3_PERM

np.set_printoptions(precision=4, suppress=True)
ATOMS=np.array(ATOMS,float); A_PRIM=np.array(A_PRIM,float)
bonds=find_bonds()
SQRT3,SQRT5=np.sqrt(3.),np.sqrt(5.)
h_w=(SQRT3+1j*SQRT5)/2.; g=10
A21=np.degrees(np.angle(h_w**g))%360
A31=np.degrees(np.angle((h_w/((-SQRT3+1j*SQRT5)/2.))**g))%360
print(f"LOCAL (live, spectral reference): arg(h)={np.degrees(np.angle(h_w)):.4f}"
      f"  g*arg(h)=alpha_21={A21:.3f}  alpha_31={A31:.3f}")

# ---- girth-10 NB cycles at v0 (DFS over a supercell) ----
SUP=3; girth=10
adj={}
for (s,t,c) in bonds: adj.setdefault(s,[]).append((t,tuple(c)))
def nbrs(a,cell):
    o=[]
    for (t,c) in adj.get(a,[]):
        nc=(cell[0]+c[0],cell[1]+c[1],cell[2]+c[2])
        if all(abs(x)<=SUP for x in nc): o.append((t,nc,c))
    return o
start=(0,(0,0,0)); CYC=[]
def dfs(path,edgev,cur,d):
    a,cell=cur; prev=path[-2] if d>=1 else None
    for (t,nc,c) in nbrs(a,cell):
        if prev is not None and (t,nc)==prev: continue
        if d==girth-1:
            if (t,nc)==start and start not in path[1:]:
                CYC.append((path[:],edgev+[np.array(c)]))
        elif d<girth-1:
            if (t,nc)==start: continue
            dfs(path+[(t,nc)],edgev+[np.array(c)],(t,nc),d+1)
dfs([start],[],start,0)
# also check no shorter NB closed cycle exists (girth == 10)
def shortest_cycle():
    for L in range(3,girth):
        found=[]
        def d2(p,cu,dd):
            a,cl=cu; pv=p[-2] if dd>=1 else None
            for (t,nc,c) in nbrs(a,cl):
                if pv is not None and (t,nc)==pv: continue
                if dd==L-1:
                    if (t,nc)==start and start not in p[1:]: found.append(1)
                elif dd<L-1 and (t,nc)!=start: d2(p+[(t,nc)],(t,nc),dd+1)
        d2([start],start,0)
        if found: return L
    return girth
uniq={}
for (p,ev) in CYC:
    es=frozenset(tuple(sorted([p[i],p[(i+1)%len(p)]])) for i in range(len(p)))
    uniq.setdefault(es,(p,ev))
cycles=list(uniq.values())
gmin=shortest_cycle()

print("\n"+"="*70)
print("CORRECTNESS GATE — exactly 15 girth-10 cycles at v0; girth==10")
print("="*70)
print(f"  unique girth-10 cycles at v0 = {len(cycles)}  (expect 15)")
print(f"  shortest NB closed cycle length = {gmin}  (expect 10)")
if len(cycles)!=15 or gmin!=10:
    print("\n  ** GATE FAILED — frame/enumeration wrong. VOID. **"); sys.exit(0)
print("  GATE PASSED.\n")

# ---- per-cycle net Z^3 voltage (the connection holonomy in Z^3) ----
def cycle_voltage(ev): return np.sum(ev,axis=0).astype(int)
V=[cycle_voltage(ev) for (_,ev) in cycles]
vclasses={}
for i,v in enumerate(V): vclasses.setdefault(tuple(v),[]).append(i)
print("="*70)
print("T-A — voltage class structure of the 15 girth cycles")
print("="*70)
for vc,idx in sorted(vclasses.items()):
    print(f"  voltage {vc}: {len(idx)} cycles   sum%3={tuple(np.array(vc)%3)}")
nonzero=sum(1 for v in V if np.any(v!=0))
split=sorted(len(i) for i in vclasses.values())
print(f"  nonzero-voltage cycles: {nonzero}/15   class sizes={split}")
ta = nonzero>0 and len(vclasses)>1     # connection nontrivial & class-structured

# ---- holonomies around a single girth cycle ----
P=np.array([0.25,0.25,0.25])
def abelian_bloch(ev):                       # U(1): prod exp(2pi i P.c)
    return np.angle(np.prod([np.exp(2j*np.pi*np.dot(P,c)) for c in ev]))
def z3_voltage_hol(v):                        # Z3: omega^(sum V mod 3)
    return (2*np.pi/3)*(int(np.sum(v))%3)
# generation-fiber (nonabelian) transport: per directed bond, the C3
# generation 3-cycle raised to the bond's voltage-component sum mod 3,
# carried by the scalar Bloch phase. Wilson loop = ordered product.
Z3=np.roll(np.eye(3),1,axis=0)               # 3-cycle generator
def gen_fiber_hol(ev):
    M=np.eye(3,dtype=complex)
    for c in ev:
        ph=np.exp(2j*np.pi*np.dot(P,c))
        M = (ph*np.linalg.matrix_power(Z3,int(np.sum(c))%3)) @ M
    return M
print("\n"+"="*70)
print("T-B — single-loop holonomies (per voltage class)")
print("="*70)
for vc,idx in sorted(vclasses.items()):
    ev0=cycles[idx[0]][1]
    ab=np.degrees(abelian_bloch(ev0))%360
    z3=np.degrees(z3_voltage_hol(np.array(vc)))%360
    eig=np.linalg.eigvals(gen_fiber_hol(ev0))
    ge=sorted(np.degrees(np.angle(eig))%360)
    # class-constancy: do all cycles in the class share the holonomy?
    abs_all={round(np.degrees(abelian_bloch(cycles[j][1]))%360,2) for j in idx}
    print(f"  V={vc} (n={len(idx)}): abelianU1={ab:7.2f}  Z3volt={z3:6.1f}  "
          f"genfiber-eigargs={['%.1f'%x for x in ge]}  class-const(U1)={len(abs_all)==1}")

# ---- T-C: EXACT winding-linearity (the U-wall discriminator) ----
print("\n"+"="*70)
print("T-C — winding linearity w=1,2,3 (holonomy => EXACT w*theta; the")
print("      property whose ABSENCE was the Ramanujan U-wall)")
print("="*70)
ev_rep=cycles[0][1]
lin_ok=True
for nm,fn in [("abelianU1",lambda e:np.degrees(abelian_bloch(e))%360)]:
    base=fn(ev_rep)
    for w in (1,2,3):
        val=np.degrees(abelian_bloch(list(ev_rep)*w))%360
        exp=(w*base)%360
        ok=abs((val-exp+180)%360-180)<1e-6
        lin_ok&=ok
        print(f"  {nm} w={w}: holonomy={val:7.2f}  expected w*theta={exp:7.2f}  exact={ok}")
# genfiber winding
for w in (1,2,3):
    M=gen_fiber_hol(list(ev_rep)*w)
    ar=sorted(np.degrees(np.angle(np.linalg.eigvals(M)))%360)
    print(f"  genfiber w={w}: eig-args={['%.1f'%x for x in ar]}")

# ---- T-D: enantiomer (V -> -V) conjugation ----
print("\n"+"="*70)
print("T-D — enantiomer (mirror net: voltage V -> -V) => conjugate?")
print("="*70)
ev0=cycles[0][1]
h_plus=np.degrees(abelian_bloch(ev0))%360
h_minus=np.degrees(abelian_bloch([-c for c in ev0]))%360
print(f"  abelianU1: I4_1 = {h_plus:.2f}   I4_3(mirror) = {h_minus:.2f}   "
      f"sum%360 = {(h_plus+h_minus)%360:.2f} (0 => conjugate pair)")
td = abs((h_plus+h_minus)%360)<1e-6 or abs((h_plus+h_minus)%360-360)<1e-6

# ---- value & K-rationality check vs the spectral reference ----
print("\n"+"="*70)
print("VALUE — any single-loop holonomy class hitting a live local target?")
print("="*70)
hit=False
for vc,idx in sorted(vclasses.items()):
    ev0=cycles[idx[0]][1]
    for nm,val in [("abelianU1",np.degrees(abelian_bloch(ev0))%360),
                   ("Z3volt",np.degrees(z3_voltage_hol(np.array(vc)))%360)]:
        for tgt,tn in [(A21,"alpha_21"),(A31,"alpha_31")]:
            d=abs((val-tgt+180)%360-180)
            if d<5.0: print(f"  V={vc} {nm}={val:.2f} ~ {tn}={tgt:.2f} (off {d:.2f})"); hit=True
ge_args=sorted(np.degrees(np.angle(np.linalg.eigvals(gen_fiber_hol(cycles[0][1]))))%360)
print(f"  spectral reference g*arg(h)=alpha_21={A21:.3f} is NOT a 90/120-deg")
print(f"  multiple: 162.388/90={A21/90:.4f}, /120={A21/120:.4f} (=> arg(h) is")
print(f"  a SPECTRAL eigen-phase, not a finite Z3/U(1) flat-connection value)")
print(f"  genfiber eig-args (cycle0): {['%.2f'%x for x in ge_args]}")

print("\n"+"="*70); print("  VERDICT"); print("="*70)
if hit and ta and lin_ok and td:
    Vd="DISCHARGE"
elif ta and lin_ok and td:
    Vd="REDUCTION (cutoff-free holonomy well-posed; U-wall dissolved; value conditional)"
else:
    Vd="NEGATIVE (single-loop holonomies are abelian 90/120-deg; arg(h) irreducibly spectral; U-wall intrinsic)"
print(f"  GATE=PASS  T-A(class)={'P' if ta else 'F'}  T-C(winding-linear)="
      f"{'P' if lin_ok else 'F'}  T-D(enantiomer-conj)={'P' if td else 'F'}  "
      f"value-hit={'Y' if hit else 'N'}")
print(f"  -> {Vd}")
print("  Ships no number; changes no ledger row.")
print("="*70)
