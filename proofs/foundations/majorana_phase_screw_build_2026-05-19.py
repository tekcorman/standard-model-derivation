#!/usr/bin/env python3
"""
proofs/foundations/majorana_phase_screw_build_2026-05-19.py

THE BUILD (user picked option 2).  I4_1 32 4_1-screw action on the
framework's OWN substrate (proofs.common: ATOMS are CARTESIAN, conventional
a=1; bond offset = c.A_PRIM; bcc I-centering).  Screw generators = ITA #214
general positions (from delta_dynamical.full_41_generators), supplied as
explicit (linear O, translation tau) pairs — NOT finite-differenced through
a %1.0 wrap (that bug was caught by the C3 gate in the prior revision).

Tests the two structural facts the whole thread reduced to:
  T1  CHIRALITY: exactly one screw handedness (4_1) is a graph automorphism
      of srs; the enantiomorph (4_3, same rotation, screw translation 3/4)
      is NOT.  That asymmetry IS the substrate-derived chirality.
  T2  GENUINE girth-10 chirality invariant via 4_1 screw-axis helicity,
      where the naive winding measure is EXACTLY ZERO noise
      (srs_girth_chirality_split.py).
  T3  SCREW-PERIOD <-> GIRTH: helical 10-ring axial span vs the 1/4-cell
      screw step (integer => L=g screw-fixed => would dissolve U).

CORRECTNESS GATE (linter discipline; I have erred this session): the probe
must first reproduce the KNOWN C3 [111] automorphism (x,y,z)->(z,x,y),
atom perm = C3_PERM = [0,3,1,2].  If C3 fails, frame is wrong => VOID, no
screw result reported.

PRE-DECLARED OUTCOMES:
  DISCHARGE : C3 gate PASS + T1 + T2 clean bimodal + T3 integer.
  REDUCTION : C3 gate PASS + T1 + T2 clean, T3 fails (chirality substrate-
              derived; length U survives).
  NEGATIVE  : C3 gate FAIL (VOID) | T1 fail | T2 ~zero.
Ships no number into predictions/; changes no ledger row.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import ATOMS, A_PRIM, N_ATOMS, find_bonds

np.set_printoptions(precision=4, suppress=True)
TOL = 0.03
ATOMS = np.array(ATOMS, float)             # CARTESIAN (conv. a=1)
A_PRIM = np.array(A_PRIM, float)           # bcc primitive (Cartesian)
bonds = find_bonds()

def in_bcc(d):
    """Cartesian d is a bcc lattice vector: d ≡ 0 or (1/2,1/2,1/2) (mod 1)."""
    f = (d + 0.5) % 1.0 - 0.5
    return np.all(np.abs(f) < TOL) or np.all(np.abs(np.abs(f) - 0.5) < TOL)

# ---- operators as explicit (O linear, tau) — ITA #214 (delta_dynamical) ----
OPS = {
 "C3 [111] (gate)": (np.array([[0,0,1],[1,0,0],[0,1,0]],float), np.zeros(3)),
 "4_1 || z":  (np.array([[0,-1,0],[1,0,0],[0,0,1]],float), np.array([0.5,0,0.25])),
 "4_1 || x":  (np.array([[1,0,0],[0,0,-1],[0,1,0]],float), np.array([0.25,0.5,0])),
 "4_1 || y":  (np.array([[0,0,1],[0,1,0],[-1,0,0]],float), np.array([0,0.25,0.5])),
 "4_3 || z (enantiomorph)":
              (np.array([[0,-1,0],[1,0,0],[0,0,1]],float), np.array([0.5,0,0.75])),
}

def atom_perm(O, tau):
    perm=[-1]*N_ATOMS
    for i in range(N_ATOMS):
        img=O@ATOMS[i]+tau
        for j in range(N_ATOMS):
            if in_bcc(img-ATOMS[j]): perm[i]=j; break
    return perm if sorted(p for p in perm if p>=0)==list(range(N_ATOMS)) else None

# precompute, for each atom, its NN bond displacement vectors (Cartesian)
nn_disp = {a:[] for a in range(N_ATOMS)}
for (s,t,c) in bonds:
    nn_disp[s].append((t, ATOMS[t] + np.array(c,float)@A_PRIM - ATOMS[s]))

def is_automorphism(O, tau):
    perm=atom_perm(O,tau)
    if perm is None: return False, None
    for s in range(N_ATOMS):
        for (t, d) in nn_disp[s]:
            Dp = O@d                                   # image displacement
            s2, t2 = perm[s], perm[t]
            ok=False
            for (tt, dd) in nn_disp[s2]:               # bond from image source?
                if in_bcc(tt - t2 if False else np.zeros(3)):  # (atom id exact)
                    pass
                if tt==t2 and np.linalg.norm(Dp-dd)<TOL: ok=True; break
            if not ok:                                  # try reverse bond
                for (tt, dd) in nn_disp[perm[t]]:
                    if tt==s2 and np.linalg.norm(Dp+dd)<TOL: ok=True; break
            if not ok: return False, perm
    return True, perm

print("="*72)
print("CORRECTNESS GATE — known C3 [111] must be a graph automorphism")
print("="*72)
O,tau = OPS["C3 [111] (gate)"]
c3_ok, c3_perm = is_automorphism(O,tau)
print(f"  C3 automorphism = {c3_ok}   atom-perm = {c3_perm}  (expect [0,3,1,2])")
if not (c3_ok and c3_perm==[0,3,1,2]):
    print("\n  ** GATE FAILED — frame still wrong. Screw verdict VOID. **")
    print("="*72); sys.exit(0)
print("  GATE PASSED.\n")

print("="*72); print("T1 — substrate-derived CHIRALITY"); print("="*72)
res={}
for nm in ["4_1 || z","4_1 || x","4_1 || y","4_3 || z (enantiomorph)"]:
    O,tau=OPS[nm]; ok,perm=is_automorphism(O,tau); res[nm]=ok
    print(f"  {nm:26s}: automorphism = {ok}   atom-perm={perm}")
t1 = (res["4_1 || z"] or res["4_1 || x"] or res["4_1 || y"]) and not res["4_3 || z (enantiomorph)"]
print(f"\n  T1 = {'PASS (unique handedness: 4_1 yes, 4_3 no)' if t1 else 'FAIL'}")

# ---- girth-10 enumeration at v0 (srs_ten_cycle DFS), SUP=3 ----
SUP=3; girth=10
def vcart(a,cell): return ATOMS[a]+np.array(cell,float)@A_PRIM   # CARTESIAN
adj={}
for (s,t,c) in bonds: adj.setdefault(s,[]).append((t,c))
def nbrs(a,cell):
    out=[]
    for (t,c) in adj.get(a,[]):
        nc=(cell[0]+c[0],cell[1]+c[1],cell[2]+c[2])
        if all(abs(x)<=SUP for x in nc): out.append((t,nc))
    return out
start=(0,(0,0,0)); C=[]
def dfs(path,cur,d):
    a,cell=cur; prev=path[-2] if d>=1 else None
    for nx in nbrs(a,cell):
        if prev is not None and nx==prev: continue
        if d==girth-1:
            if nx==start and start not in path[1:]: C.append(path[:])
        elif d<girth-1:
            if nx==start: continue
            dfs(path+[nx],nx,d+1)
dfs([start],start,0)
uniq={}
for p in C:
    es=frozenset(tuple(sorted([p[i],p[(i+1)%len(p)]])) for i in range(len(p)))
    uniq.setdefault(es,p)
cycles=list(uniq.values())

print("\n"+"="*72)
print(f"T2 — 4_1 screw-axis helicity on {len(cycles)} girth-10 rings")
print("     (naive winding here is EXACTLY 0 — srs_girth_chirality_split)")
print("="*72)
zc=np.array([0,0,1.0])                                  # 4_1||z screw axis
naive=[]; sh=[]
for p in cycles:
    Pp=[vcart(a,c) for (a,c) in p]; n=len(Pp); o=np.mean(Pp,0)
    s=0.0; w=0.0
    for i in range(n):
        v1=Pp[i]-o; v2=Pp[(i+1)%n]-o
        p1=v1-np.dot(v1,zc)*zc; p2=v2-np.dot(v2,zc)*zc
        ang=np.arctan2(np.dot(np.cross(p1,p2),zc), np.dot(p1,p2)+1e-30)
        dz=np.dot(Pp[(i+1)%n]-Pp[i],zc)
        s+=ang*dz
        e1=Pp[(i+1)%n]-Pp[i]; e2=Pp[(i+2)%n]-Pp[(i+1)%n]
        w+=np.dot(np.cross(e1,e2),e1+e2)
    naive.append(w); sh.append(s)
naive=np.array(naive); sh=np.array(sh)
pos=int((sh>1e-4).sum()); neg=int((sh<-1e-4).sum()); fl=int((np.abs(sh)<=1e-4).sum())
print(f"  naive winding  max|.| = {np.max(np.abs(naive)):.2e}   (expect ~0)")
print(f"  screw helicity min|.| = {np.min(np.abs(sh)):.4f}  max|.| = {np.max(np.abs(sh)):.4f}")
print(f"  screw-helicity split  +{pos} / -{neg} / flat {fl}   (of {len(cycles)})")
t2 = np.min(np.abs(sh))>1e-3 and pos>0 and neg>0 and fl==0
print(f"  T2 = {'CLEAN bimodal (genuine invariant exists)' if t2 else 'NOT clean'}")

print("\n"+"="*72); print("T3 — screw-period <-> girth length"); print("="*72)
qstep=0.25                                              # 1/4-cell screw step (a=1)
spans=np.array([ (max(np.dot(vcart(a,c),zc) for (a,c) in p)
                 -min(np.dot(vcart(a,c),zc) for (a,c) in p))/qstep for p in cycles])
ni=np.all(np.abs(spans-np.round(spans))<0.1)
print(f"  axial span / (1/4-cell): min={spans.min():.3f} max={spans.max():.3f} "
      f"mean={spans.mean():.3f}")
print(f"  integer # of screw steps: {ni}  set={sorted(set(np.round(spans).astype(int)))}")
t3=bool(ni)

print("\n"+"="*72); print("  VERDICT"); print("="*72)
V=("DISCHARGE" if (t1 and t2 and t3) else
   "REDUCTION (chirality substrate-derived; length U survives)" if (t1 and t2) else
   "NEGATIVE (screw route does not help; blocker stands)")
print(f"  C3-gate=PASS  T1={'P' if t1 else 'F'}  T2={'P' if t2 else 'F'}  "
      f"T3={'P' if t3 else 'F'}  ->  {V}")
print("  Ships no number; changes no ledger row.")
print("="*72)
