#!/usr/bin/env python3
"""
proofs/foundations/majorana_phase_screw_exchange_verify_2026-05-19.py

STRUCTURE-VERIFICATION (not a phase prediction).  User's argument:
  (1) edge-reversal J commutes with Re(A_H), anticommutes with Im(A_H)
      <=> J A_H(k) J = A_H(k)*  ;
  (2) hence v |-> J v* maps the h-eigenstate to the h*-eigenstate
      (Theta = J.K is an antiunitary symmetry of A_H, Theta^2 = +I);
  (3) the I4_1 32 4_1 screw handedness then selects ONE of {h, h*}.

This probe RIGOROUSLY checks (1) and (2) (cheap, decisive — the load-
bearing algebra).  It does NOT test (3): the 4_1 screw operator is not
built anywhere in the codebase (only C3_PERM exists), and the framework's
own srs_ten_cycle_chirality_diagnostic.py shows the naive measure does NOT
classify the rings (6 FLAT) — the genuine screw measure is the open item.
(3) is scoped as the decisive NEXT probe, not claimed here.

PRE-DECLARED OUTCOMES (verification probe):
  CONFIRMED : (1) holds to ~1e-12 for untwisted AND C3-twisted A_H at
              multiple k incl. P, AND (2) holds (J v* is an A_H eigenvector
              with eigenvalue = conj of the original) => the user's
              exchange MECHANISM is correct; ADOPTED-NU-MAJ-PHASE reduces
              to the (un-built) 4_1-screw sign + (open) screw-period<->girth
              length identity.  A genuine reduction, NOT a closure.
  PARTIAL   : (1) holds untwisted but the C3 twist spoils it (the twist
              does not commute with reversal) => the exchange is rep-
              dependent; selection still entangled with the C3 channel.
  REFUTED   : (1) fails => J does not implement complex conjugation of
              A_H; the whole J.K exchange argument is wrong.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from proofs.common import find_bonds, ATOMS, A_PRIM, C3_PERM, N_ATOMS

np.set_printoptions(precision=4, suppress=True)
bonds = find_bonds(); n = len(bonds)

# ---- C3 twist machinery (verbatim from vus_ihara_zeta_c3twisted.py) ----
C3C = np.array([[0,0,1],[1,0,0],[0,1,0]], float)
c3a = {i:int(np.argmax(C3_PERM[:,i])) for i in range(N_ATOMS)}
def disp(s,t,c): return (np.array(ATOMS[t])+c[0]*np.array(A_PRIM[0])
        +c[1]*np.array(A_PRIM[1])+c[2]*np.array(A_PRIM[2])-np.array(ATOMS[s]))
pd=[disp(s,t,c) for s,t,c in bonds]
def c3b(i):
    s0,_,_=bonds[i]; ns=c3a[s0]; r=C3C@pd[i]
    for j,(s,t,c) in enumerate(bonds):
        if s==ns and np.allclose(pd[j],r,atol=1e-8): return j
    raise ValueError
cm=[c3b(i) for i in range(12)]; vis=[False]*12; orb=[]
for st in range(12):
    if vis[st]: continue
    b0,b1,b2=st,cm[st],cm[cm[st]]
    orb.append((b0,b1,b2)); vis[b0]=vis[b1]=vis[b2]=True
posd={}
for (b0,b1,b2) in orb: posd[b0],posd[b1],posd[b2]=0,1,2
pa=np.array([posd[i] for i in range(n)])
def Wmat(om): p=om**pa; return np.outer(p,p.conj())
OM=np.exp(2j*np.pi/3); W1=Wmat(OM)

def AH(k):
    M=np.zeros((n,n),complex)
    for j,(sj,tj,dcj) in enumerate(bonds):
        for i,(si,ti,dci) in enumerate(bonds):
            if sj!=ti: continue
            if tj==si and tuple(int(dci[d])+int(dcj[d]) for d in range(3))==(0,0,0): continue
            M[j,i]=np.exp(2j*np.pi*np.dot(k,dci))
    return M

# ---- J = edge reversal (e -> e_bar) ----
def cof(c): return tuple(int(round(x)) for x in c)
rev=[None]*n
for i,(s,t,c) in enumerate(bonds):
    for j,(s2,t2,c2) in enumerate(bonds):
        if s2==t and t2==s and cof(c2)==cof([-c[0],-c[1],-c[2]]): rev[i]=j;break
assert all(r is not None for r in rev) and all(rev[rev[i]]==i for i in range(n))
J=np.zeros((n,n))
for i in range(n): J[i,rev[i]]=1.0

P=np.array([0.25,0.25,0.25])
KPTS=[("Gamma",[0,0,0]),("P",[0.25,0.25,0.25]),("gen",[0.13,0.27,0.41])]

print("="*70)
print("[1]  J A_H(k) J  ==  conj(A_H(k))   [<=> [J,ReA]=0, {J,ImA}=0]")
print("="*70)
ok1=True
for tag,A_op,lbl in [("untwisted",lambda k:AH(k),""),
                     ("C3-omega-twisted",lambda k:W1*AH(k),"")]:
    for kn,kv in KPTS:
        A=A_op(kv)
        e_conj = np.max(np.abs(J@A@J - np.conj(A)))
        e_re   = np.max(np.abs(J@A.real - A.real@J))      # [J,ReA]
        e_im   = np.max(np.abs(J@A.imag + A.imag@J))      # {J,ImA}
        print(f"  {tag:18s} k={kn:6s}: |JAJ-conjA|={e_conj:.2e}  "
              f"|[J,ReA]|={e_re:.2e}  |{{J,ImA}}|={e_im:.2e}")
        ok1 &= (e_conj<1e-10 and e_re<1e-10 and e_im<1e-10)

print("\n"+"="*70)
print("[2]  v -> J conj(v) maps h-eigenstate to h*-eigenstate at P")
print("="*70)
ok2=True
for tag,Aop in [("untwisted",AH(P)),("C3-omega",W1*AH(P))]:
    w,V=np.linalg.eig(Aop)
    # pick the leading-|.| complex eigenpair
    idx=np.argmax(np.abs(w)+1e6*(np.abs(w.imag)<1e-6))
    h=w[idx]; v=V[:,idx]
    u=J@np.conj(v)
    resid=np.linalg.norm(Aop@u - np.conj(h)*u)/ (np.linalg.norm(u)+1e-30)
    # is conj(h) actually in the spectrum?
    inspec=np.min(np.abs(w-np.conj(h)))
    print(f"  {tag:9s}: h={h:.4f}  conj(h)={np.conj(h):.4f}  "
          f"||A(Jv*)-h* (Jv*)||/||.||={resid:.2e}  dist(conj h,spec)={inspec:.2e}")
    ok2 &= (resid<1e-8 and inspec<1e-8)

print("\n"+"="*70)
print("[3]  Theta = J.K antiunitary: Theta^2=+I and [Theta,A_H]=0")
print("     (Theta A Theta^-1 = J conj(A) J ; check == A)")
print("="*70)
ok3=True
for tag,Aop in [("untwisted",AH(P)),("C3-omega",W1*AH(P))]:
    th2=np.max(np.abs(J@np.conj(J)-np.eye(n)))           # (JK)^2 = J J* = J^2 = I
    comm=np.max(np.abs(J@np.conj(Aop)@J - Aop))          # Theta A Theta^-1 - A
    print(f"  {tag:9s}: |Theta^2 - I|={th2:.2e}   |Theta A Theta^-1 - A|={comm:.2e}")
    ok3 &= (th2<1e-12 and comm<1e-10)

print("\n"+"="*70)
print("[4]  C3-sign-swap <-> chirality (B3 Open-Q4): does Theta send the")
print("     omega-twist h-sector to the omega^2-twist h*-sector?")
print("="*70)
W2=Wmat(np.exp(4j*np.pi/3))
Aw, Aw2 = W1*AH(P), W2*AH(P)
ww,Vw=np.linalg.eig(Aw); idx=np.argmax(np.abs(ww)+1e6*(np.abs(ww.imag)<1e-6))
hw=ww[idx]; vw=Vw[:,idx]; u=J@np.conj(vw)
r_self = np.linalg.norm(Aw @u-np.conj(hw)*u)/(np.linalg.norm(u)+1e-30)
r_cross= np.linalg.norm(Aw2@u-np.conj(hw)*u)/(np.linalg.norm(u)+1e-30)
print(f"  omega h={hw:.4f}; Theta v lands in: omega-sector resid={r_self:.2e}"
      f"   omega^2-sector resid={r_cross:.2e}")
print(f"  => Theta keeps the C3 channel ({'same' if r_self<1e-8 else '?'}); the")
print(f"     4_1-screw sign (NOT built here) is what must pick h vs h* WITHIN it.")

print("\n"+"="*70)
print("  VERDICT")
print("="*70)
v = "CONFIRMED" if (ok1 and ok2 and ok3) else ("PARTIAL" if ok1 else "REFUTED")
print(f"  [1]={'OK' if ok1 else 'FAIL'}  [2]={'OK' if ok2 else 'FAIL'}  "
      f"[3]={'OK' if ok3 else 'FAIL'}   ->  {v}")
print("""  Reading: CONFIRMED => the J.K exchange mechanism is correct &
  rep-robust; ADOPTED-NU-MAJ-PHASE reduces to (a) the un-built 4_1-screw
  enantiomorph sign and (b) the still-open screw-period<->girth-10 length
  identity (srs_ten_cycle_chirality_diagnostic.py). A REDUCTION, not a
  closure. Ships no number; changes no ledger row.""")
print("="*70)
