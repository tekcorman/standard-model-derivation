#!/usr/bin/env python3
"""
proofs/foundations/majorana_phase_chirality_projected_2026-05-19.py

CORRECTED PROBE — the 2026-05-19 global-zeta probe
(majorana_phase_global_zeta_probe_2026-05-19.py) found Tr[(A_H^(m)(P))^g]
EXACTLY REAL (-240.000) and concluded "global => phase-free".  User pushback:
that reality is the CONJUGATE-PAIRING signature of summing BOTH chirality
sectors (for every NB eigenvalue mu there is mu_bar; Sigma mu^g is real BY
CONSTRUCTION), i.e. the documented "+Im-chirality convention" step in
srs_hashimoto_seesaw_verify.py — NOT a verdict on the physics.

A nu_R Majorana mass nu_R^T C nu_R is built from ONE chirality component
(Type-3 QFT structure).  The correct global object is therefore the
CHIRALITY-PROJECTED waterfilling sum: EVERY retained adjacency mode lambda
contributes (full waterfilling — NOT single-leading-mode), but with ONE
Ihara-Bass root per mode (the +Im branch of mu^2 - lambda mu + (k*-1) = 0).
This is NOT the retracted argmin/leading-mode move; nothing subleading is
dropped — only the conjugate (opposite-chirality) root per mode.

ANTI-GOAL-SEEK — PRE-DECLARED OUTCOMES (before any number):
  Local targets live from h=(±sqrt3+i sqrt5)/2, g=10.  Chirality projection
  done THREE structurally-distinct ways (C1 Im-sign half-spectrum; C2
  per-IB-pair +Im root; C3 framework-native: +Im root of the adjacency
  Bloch quadratic).  Phase checked for LENGTH-DRIFT within one chirality
  (g, 2g, 3g) — the real Ramanujan test — and BZ-convergence.

  R  DISCHARGE  : chirality-projected cutoff-free phase is STABLE across
                  length, BZ-res AND the 3 projection schemes, and matches
                  a live local target (alpha_21~162.39 / alpha_31~324.78).
                  => P35/P36 derivable; residual collapses to the DISCRETE
                     +/- chirality SIGN label only (documented weak
                     convention) — a major reduction of ADOPTED-NU-MAJ-PHASE.
                     User is right.
  D  DIFFERENT  : stable across all but != local => different cutoff-free
                  prediction (162/325 were single-leading-mode artefacts).
  UD UNDERDET   : C1/C2/C3 disagree => chirality projection is
                  scheme-dependent; the adoption survives in the choice.
  U  DRIFT      : arg drifts with length even within one chirality =>
                  Ramanujan phase-drift survives chirality projection;
                  the original concern stands chirality-resolved.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS

np.set_printoptions(precision=6, suppress=True)
SQRT3, SQRT5 = np.sqrt(3.0), np.sqrt(5.0)
g = 10
k_star = 3
h_w  = ( SQRT3 + 1j*SQRT5)/2.0
h_w2 = (-SQRT3 + 1j*SQRT5)/2.0
A21 = np.degrees(np.angle(h_w**g)) % 360                 # 162.388 (P35)
A31 = np.degrees(np.angle((h_w/h_w2)**g)) % 360          # 324.775 (P36)
DLT = np.degrees(np.angle(h_w2**g)) % 360                # 197.612 (other band)
print(f"LOCAL (live): alpha_21={A21:.3f}  alpha_31={A31:.3f}  other-band={DLT:.3f}")

# ---- twist machinery (verbatim from vus_ihara_zeta_c3twisted.py) ----
bonds = find_bonds(); n = len(bonds)
C3C = np.array([[0,0,1],[1,0,0],[0,1,0]], float)
c3a = {i:int(np.argmax(C3_PERM[:,i])) for i in range(N_ATOMS)}
def disp(s,t,c): return (np.array(ATOMS[t])+c[0]*np.array(A_PRIM[0])
        +c[1]*np.array(A_PRIM[1])+c[2]*np.array(A_PRIM[2])-np.array(ATOMS[s]))
pd = [disp(s,t,c) for s,t,c in bonds]
def c3b(i):
    s0,_,_=bonds[i]; ns=c3a[s0]; r=C3C@pd[i]
    for j,(s,t,c) in enumerate(bonds):
        if s==ns and np.allclose(pd[j],r,atol=1e-8): return j
    raise ValueError
cm=[c3b(i) for i in range(12)]; vis=[False]*12; orb=[]
for st in range(12):
    if vis[st]: continue
    b0,b1,b2=st,cm[st],cm[cm[st]]; assert cm[b2]==b0 and len({b0,b1,b2})==3
    orb.append((b0,b1,b2)); vis[b0]=vis[b1]=vis[b2]=True
pos={}
for (b0,b1,b2) in orb: pos[b0],pos[b1],pos[b2]=0,1,2
pa=np.array([pos[i] for i in range(n)])
def W(om): p=om**pa; return np.outer(p,p.conj())
def AH(k):
    M=np.zeros((n,n),complex)
    for j,(sj,tj,dcj) in enumerate(bonds):
        for i,(si,ti,dci) in enumerate(bonds):
            if sj!=ti: continue
            if tj==si and tuple(int(dci[d])+int(dcj[d]) for d in range(3))==(0,0,0): continue
            M[j,i]=np.exp(2j*np.pi*np.dot(k,dci))
    return M
OM=np.exp(2j*np.pi/3); OM2=np.exp(4j*np.pi/3)
W1,W2=W(OM),W(OM2)
P=np.array([0.25,0.25,0.25])

def chir_sum(Aop, power, scheme):
    """Sum mu^power over the +chirality projection of operator Aop.
       scheme C1: Im(mu)>tol half-spectrum
       scheme C2: per conjugate-partner pair, the Im>0 member
       (C3 handled separately on the adjacency Bloch quadratic)."""
    ev = np.linalg.eigvals(Aop)
    tol = 1e-7
    if scheme == "C1":
        sel = ev[ev.imag > tol]
    elif scheme == "C2":
        sel, used = [], np.zeros(len(ev), bool)
        for a in range(len(ev)):
            if used[a]: continue
            # find conjugate partner
            b = np.argmin([abs(ev[a]-np.conj(ev[c])) if c!=a and not used[c]
                           else 1e9 for c in range(len(ev))])
            if abs(ev[a]-np.conj(ev[b]))<1e-5 and b!=a:
                used[a]=used[b]=True
                sel.append(ev[a] if ev[a].imag>=ev[b].imag else ev[b])
            else:
                used[a]=True
                if ev[a].imag>tol: sel.append(ev[a])
        sel=np.array(sel)
    return np.sum(sel**power) if len(sel) else 0j

# ---- C3: framework-native — +Im root of the ADJACENCY Bloch quadratic ----
def adj_bloch(k):
    """4x4 C3-twisted adjacency Bloch operator (srs primitive cell)."""
    A=np.zeros((N_ATOMS,N_ATOMS),complex)
    for (s,t,c) in bonds:
        ph=np.exp(2j*np.pi*np.dot(k,c))
        tw=OM**((pos.get  (0,0)))  # placeholder; twist applied below
    # build untwisted then C3-character-weight via bond orbit position
    A=np.zeros((N_ATOMS,N_ATOMS),complex)
    for bi,(s,t,c) in enumerate(bonds):
        A[s,t]+=np.exp(2j*np.pi*np.dot(k,c))*(OM**pa[bi])
    return A
def chir_sum_C3(k, power):
    lam=np.linalg.eigvals(adj_bloch(k))
    tot=0j
    for L in lam:
        disc=L*L-4*(k_star-1)
        r=np.sqrt(complex(disc))
        m1,m2=(L+r)/2,(L-r)/2
        mu=m1 if m1.imag>=m2.imag else m2          # +Im branch
        tot+=mu**power
    return tot

print("\n[A] CHIRALITY-PROJECTED girth-length amplitude at P (all modes, +chir)")
for lbl,Wt in (("omega",W1),("omega^2",W2)):
    Aop=Wt*AH(P)
    for sc in ("C1","C2"):
        s=chir_sum(Aop,g,sc)
        print(f"   {lbl:7s} {sc}: arg Sigma_+ mu^g = {np.degrees(np.angle(s))%360:8.3f} deg "
              f"|S|={abs(s):.3e}")
    s3=chir_sum_C3(P,g)
    print(f"   {lbl:7s} C3: arg = {np.degrees(np.angle(s3))%360:8.3f} deg  "
          f"(adjacency-quadratic +Im root, all 4 modes)")
print(f"   compare local: alpha_21={A21:.3f}  other-band={DLT:.3f}  alpha_31={A31:.3f}")

print("\n[B] LENGTH-DRIFT within ONE chirality  (the real Ramanujan test:")
print("    does arg drift with length g,2g,3g once conjugates are removed?)")
for lbl,Wt in (("omega",W1),("omega^2",W2)):
    Aop=Wt*AH(P)
    row=[f"{np.degrees(np.angle(chir_sum(Aop,L,'C1')))%360:8.3f}" for L in (g,2*g,3*g)]
    rowC3=[f"{np.degrees(np.angle(chir_sum_C3(P,L)))%360:8.3f}" for L in (g,2*g,3*g)]
    print(f"   {lbl:7s} C1 arg @L=g,2g,3g: {' '.join(row)}")
    print(f"   {lbl:7s} C3 arg @L=g,2g,3g: {' '.join(rowC3)}")

print("\n[C] BZ-AVERAGED chirality-projected amplitude phase (cutoff-free in k;")
print("    fixed forced length g; convergence-tested)")
for NB in (12,20,30):
    acc={1:0j,2:0j}
    for i1 in range(NB):
        for i2 in range(NB):
            for i3 in range(NB):
                kk=np.array([i1,i2,i3])/NB
                acc[1]+=chir_sum(W1*AH(kk),g,"C1")
                acc[2]+=chir_sum(W2*AH(kk),g,"C1")
    nk=NB**3
    p1=np.degrees(np.angle(acc[1]/nk))%360
    p2=np.degrees(np.angle(acc[2]/nk))%360
    p31=np.degrees(np.angle((acc[1])/(acc[2])))%360
    print(f"   NB={NB:2d}: arg<Sigma_+ omega>={p1:8.3f}  arg<omega^2>={p2:8.3f}  "
          f"arg(ratio)={p31:8.3f}  (loc a21={A21:.2f} a31={A31:.2f})")

# ---- mechanical verdict ----
def near(x,t,tol=8.0):
    d=abs((x-t+180)%360-180); return d<=tol,d
print("\n"+"="*74+"\n  VERDICT (pre-declared; nothing searched for a target)\n"+"="*74)
sP=chir_sum(W1*AH(P),g,"C1"); s2P=chir_sum(W2*AH(P),g,"C1")
aw=np.degrees(np.angle(sP))%360; aw2=np.degrees(np.angle(s2P))%360
ar=np.degrees(np.angle(sP/s2P))%360
for nm,val,tg in (("arg<+chir omega>^g vs alpha_21",aw,A21),
                   ("                  vs other-band",aw,DLT),
                   ("arg<+chir omega2>^g vs alpha_21",aw2,A21),
                   ("ratio arg          vs alpha_31",ar,A31)):
    ok,d=near(val,tg); print(f"   {nm:34s}: {val:8.3f}  {'MATCH' if ok else 'no'} (off {d:.2f})")
print("""
   Then read against PRE-DECLARED R / D / UD / U using [A] scheme-agreement,
   [B] length-stability, [C] BZ-stability.  Ships no number; no ledger row.""")
print("="*74)
