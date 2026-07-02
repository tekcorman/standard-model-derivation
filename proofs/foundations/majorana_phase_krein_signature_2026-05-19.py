#!/usr/bin/env python3
"""
proofs/foundations/majorana_phase_krein_signature_2026-05-19.py

DEEPEST CHIRALITY TEST — Krein-signature reading of the nu_R Majorana phase.

Probes 1-2 (2026-05-19) showed: democratic waterfilling Sigma mu^g is phase-
free (conjugate pairing), and chirality-as-eigenvalue-selection is scheme-
dependent / length-drifting.  This probe tests the structurally-deepest
version of the user's chirality intuition:

  A Majorana neutrino (its own antiparticle, chiral) is built with the
  INDEFINITE metric (C / gamma^0), not the Euclidean v^dag v.  The canonical
  indefinite metric on the non-backtracking edge space is the EDGE-REVERSAL
  involution J (e -> e_bar), which is exactly the operator implementing the
  Ihara-zeta functional equation zeta(u) <-> zeta(1/((k*-1)u)).  The
  Hashimoto operator is J-self-adjoint.  The KREIN SIGNATURE of mode i is
  s_i = sign Re( v_i^dag J v_i ).  For a J-self-adjoint operator complex-
  conjugate eigenvalues carry OPPOSITE Krein sign, so the Krein-weighted
  waterfilling amplitude

        M_R^(m) ~ Sigma_i  s_i * mu_i^g          (ALL modes; weight = the
                                                   intrinsic Krein signature)

  is NOT forced real by conjugate pairing.  This is full waterfilling
  (nothing subleading dropped) weighted by a framework-INTRINSIC invariant
  (the indefinite-metric signature = the substrate's particle/antiparticle =
  chirality), NOT the retracted single-leading-mode pick and NOT an
  eigenvalue selection by hand.

ANTI-GOAL-SEEK — PRE-DECLARED OUTCOMES:
  Local targets live from h=(+-sqrt3+i sqrt5)/2, g=10.  J built canonically
  (edge reversal). Krein form tested two ways: Hermitian-indefinite v^dag J v
  (K-H) and complex-symmetric v^T J v (K-S).  Length-drift g,2g,3g and BZ
  convergence checked.  Structurally-pinned points only (P-point, girth g).

  R  DISCHARGE : Krein-weighted phase STABLE across length, BZ-res AND both
                 Krein forms, and matches a live local target
                 (alpha_21~162.39 / alpha_31~324.78).  => P35/P36 derivable
                 via Krein signature; ADOPTED-NU-MAJ-PHASE collapses to (at
                 most) the discrete overall Krein-orientation sign (a
                 particle/antiparticle convention).  Chirality intuition
                 VINDICATED at the deepest level.
  D  DIFFERENT : stable but != local  => different cutoff-free prediction.
  UD UNDERDET  : K-H vs K-S disagree, or many Krein-neutral modes => the
                 Krein structure is ill-posed / scheme-dependent here.
  U  DRIFT     : nonzero but drifts with length => the Ramanujan phase-
                 drift survives even the Krein weighting; original concern
                 stands at the deepest level.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS

np.set_printoptions(precision=6, suppress=True)
SQRT3, SQRT5 = np.sqrt(3.0), np.sqrt(5.0)
g, k_star = 10, 3
h_w  = ( SQRT3 + 1j*SQRT5)/2.0
h_w2 = (-SQRT3 + 1j*SQRT5)/2.0
A21 = np.degrees(np.angle(h_w**g)) % 360
A31 = np.degrees(np.angle((h_w/h_w2)**g)) % 360
DLT = np.degrees(np.angle(h_w2**g)) % 360
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
    b0,b1,b2=st,cm[st],cm[cm[st]]; assert cm[b2]==b0
    orb.append((b0,b1,b2)); vis[b0]=vis[b1]=vis[b2]=True
posd={}
for (b0,b1,b2) in orb: posd[b0],posd[b1],posd[b2]=0,1,2
pa=np.array([posd[i] for i in range(n)])
def Wmat(om): p=om**pa; return np.outer(p,p.conj())
def AH(k):
    M=np.zeros((n,n),complex)
    for j,(sj,tj,dcj) in enumerate(bonds):
        for i,(si,ti,dci) in enumerate(bonds):
            if sj!=ti: continue
            if tj==si and tuple(int(dci[d])+int(dcj[d]) for d in range(3))==(0,0,0): continue
            M[j,i]=np.exp(2j*np.pi*np.dot(k,dci))
    return M
OM, OM2 = np.exp(2j*np.pi/3), np.exp(4j*np.pi/3)
W1, W2 = Wmat(OM), Wmat(OM2)
P=np.array([0.25,0.25,0.25])

# ---- canonical indefinite metric J = edge reversal (e -> e_bar) ----
def cell_of(c): return tuple(int(round(x)) for x in c)
rev = [None]*n
for i,(s,t,c) in enumerate(bonds):
    tgt=(t,s,cell_of([-c[0],-c[1],-c[2]]))
    for j,(s2,t2,c2) in enumerate(bonds):
        if s2==tgt[0] and t2==tgt[1] and cell_of(c2)==tgt[2]:
            rev[i]=j; break
assert all(r is not None for r in rev), f"reversal incomplete: {rev}"
assert all(rev[rev[i]]==i for i in range(n)), "J not an involution"
J = np.zeros((n,n));
for i in range(n): J[i,rev[i]]=1.0
print(f"\nJ = edge-reversal: involution OK, fixed pts={sum(1 for i in range(n) if rev[i]==i)},"
      f" signature(eig J)={sorted(np.round(np.linalg.eigvals(J).real).astype(int))}")

def krein_weighted(Aop, power, form):
    """Sigma_i s_i mu_i^power, s_i = sign Re(<v_i, J v_i>).
       form 'KH': v^dag J v (Hermitian-indefinite);  'KS': v^T J v."""
    w, V = np.linalg.eig(Aop)
    s = np.zeros(len(w)); neutral=0
    for i in range(len(w)):
        v=V[:,i]
        q = (v.conj()@J@v) if form=="KH" else (v@J@v)
        qr=q.real
        if abs(qr) < 1e-9*max(1.0,abs(v).max()**2):
            neutral+=1; s[i]=0.0
        else:
            s[i]=np.sign(qr)
    return np.sum(s*(w**power)), neutral, w

print("\n[A] KREIN-WEIGHTED girth amplitude at P (all modes, Krein-sign weight)")
for lbl,Wt in (("omega",W1),("omega^2",W2)):
    Aop=Wt*AH(P)
    for form in ("KH","KS"):
        val,neu,_=krein_weighted(Aop,g,form)
        a=np.degrees(np.angle(val))%360 if abs(val)>1e-9 else float('nan')
        print(f"   {lbl:7s} {form}: arg = {a:8.3f} deg  |M|={abs(val):.3e}  "
              f"Krein-neutral modes={neu}/{n}")
print(f"   compare local: alpha_21={A21:.3f}  other-band={DLT:.3f}  alpha_31={A31:.3f}")

print("\n[B] LENGTH-DRIFT within the Krein-weighted sum (g,2g,3g) — the")
print("    decisive cutoff-free test (stable => meaningful; drifts => U)")
for lbl,Wt in (("omega",W1),("omega^2",W2)):
    Aop=Wt*AH(P)
    for form in ("KH","KS"):
        rr=[]
        for L in (g,2*g,3*g):
            val,_,_=krein_weighted(Aop,L,form)
            rr.append(f"{(np.degrees(np.angle(val))%360) if abs(val)>1e-9 else float('nan'):8.3f}")
        print(f"   {lbl:7s} {form} arg @L=g,2g,3g: {' '.join(rr)}")

print("\n[C] inter-channel ratio (alpha_31 analog) + BZ-average (cutoff-free")
print("    in k, forced length g, convergence-tested), Hermitian Krein form")
vP1,_,_=krein_weighted(W1*AH(P),g,"KH"); vP2,_,_=krein_weighted(W2*AH(P),g,"KH")
if abs(vP2)>1e-9:
    print(f"   P-point arg(M_omega/M_omega2) = {np.degrees(np.angle(vP1/vP2))%360:8.3f}"
          f"   (local alpha_31={A31:.3f})")
for NB in (12,20,30):
    a1=a2=0j
    for i1 in range(NB):
        for i2 in range(NB):
            for i3 in range(NB):
                kk=np.array([i1,i2,i3])/NB
                v1,_,_=krein_weighted(W1*AH(kk),g,"KH")
                v2,_,_=krein_weighted(W2*AH(kk),g,"KH")
                a1+=v1; a2+=v2
    nk=NB**3
    pr1=np.degrees(np.angle(a1/nk))%360 if abs(a1)>1e-12 else float('nan')
    prr=np.degrees(np.angle(a1/a2))%360 if abs(a2)>1e-12 else float('nan')
    print(f"   NB={NB:2d}: arg<M_omega>={pr1:8.3f}  arg<M_om/M_om2>={prr:8.3f}"
          f"   (loc a21={A21:.2f} a31={A31:.2f})")

# ---- mechanical verdict ----
def near(x,t,tol=8.0):
    if np.isnan(x): return False,float('nan')
    d=abs((x-t+180)%360-180); return d<=tol,d
print("\n"+"="*74+"\n  VERDICT (pre-declared; nothing searched for a target)\n"+"="*74)
vw,_,_=krein_weighted(W1*AH(P),g,"KH"); vw2,_,_=krein_weighted(W2*AH(P),g,"KH")
aw =np.degrees(np.angle(vw ))%360 if abs(vw )>1e-9 else float('nan')
aw2=np.degrees(np.angle(vw2))%360 if abs(vw2)>1e-9 else float('nan')
ar =np.degrees(np.angle(vw/vw2))%360 if abs(vw2)>1e-9 else float('nan')
for nm,val,tg in (("Krein<omega>^g  vs alpha_21",aw,A21),
                  ("                vs other-band",aw,DLT),
                  ("Krein<omega2>^g vs alpha_21",aw2,A21),
                  ("ratio           vs alpha_31",ar,A31)):
    ok,d=near(val,tg)
    print(f"   {nm:32s}: {val:8.3f}  {'MATCH' if ok else 'no'} "
          f"(off {d:.2f})" if not np.isnan(val) else f"   {nm:32s}: nan")
print("""
   Read against PRE-DECLARED R/D/UD/U using [A] KH-vs-KS agreement &
   Krein-neutral count, [B] length-stability, [C] BZ-stability.
   Ships no number; changes no ledger row.""")
print("="*74)
