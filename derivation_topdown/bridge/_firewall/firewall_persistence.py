"""FALSIFIED / RETRACTED reading (kept as a record). This firewall WRONGLY treated tau (the spanning-tree
branching COUNT) as THE MASS. That is a regression: the established mass mechanism is the
recurrence-under-running (the (4,2,2) Born weights + running phase delta=2pi/sqrt7 + resolvent
G_NB=(I-uB)^-1, a=(2/3)^8), which already gives the leptons to 0.008%. tau is NOT the mass; the
branching/tilt is at most a SECTOR DIFFERENTIATOR that must feed the recurrence mass operator, not replace it.
Do NOT use tau-as-mass. See the parent CLOSURE / memory.

(original note) Mapping it tried: config=matter state; P=tau*D=mass-scale; protected(a.V=0)=persistent."""
import sys, os, itertools, numpy as np
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dirac_srs_mdl"))
import srs
E = srs.EDGES                                   # [(i,j,(v1,v2,v3)), ...]  6 edges
axis = np.array([1.,-1.,1.])/np.sqrt(3)
def tau(occ):
    if not occ: return 0
    L = np.zeros((4,4))
    for (i,j,v) in occ:
        L[i,i]+=1; L[j,j]+=1; L[i,j]-=1; L[j,i]-=1
    ev = np.sort(np.linalg.eigvalsh(L))[1:]
    return int(round(np.prod(ev)/4))
rows=[]
for r in range(7):
    for occ in itertools.combinations(E, r):
        V = np.sum([np.array(v) for (_,_,v) in occ], axis=0) if occ else np.zeros(3)
        rows.append((len(occ), tau(list(occ)), abs(float(axis.dot(V)))<1e-9))
prot=[x for x in rows if x[2]]; drift=[x for x in rows if not x[2]]
print("FORCED PERSISTENCE STRUCTURE  (P=tau*D; protected: P=tau, drifting: P->0)")
print(f"  64 configs | protected(persistent, a.V=0)={len(prot)}  drifting={len(drift)}")
print(f"  PERSISTENT tau-spectrum (P=tau)  {sorted(Counter(p[1] for p in prot).items())}")
print(f"    distinct persistent masses (tau>0): {sorted(set(p[1] for p in prot if p[1]>0))}")
print(f"  drifting tau-spectrum            {sorted(Counter(d[1] for d in drift).items())}")
print(f"  whole tau-spectrum               {sorted(set(p[1] for p in rows))}")
print()
pv = sorted(set(p[1] for p in prot if p[1]>0))
print(f"  => FORCED persistent mass-RATIOS (if P=mass): {':'.join(str(x) for x in pv)}  (small integers)")
print(f"     drifting hierarchy knob: geodesic decay 1/sqrt(k-1)=1/sqrt2 per step (k=3 forced).")
print()
print("OBSERVED (firewall, my side):")
print("  charged-fermion masses span ~5 orders (m_t/m_e~3.4e5); even within one ladder m_mu/m_e=206.8, m_tau/m_e=3477.")
print("  => small integer ratios {1,3,4,8} do NOT match the fermion hierarchy as direct masses.")
print("  the 1/sqrt2 geodesic decay IS the existing framework's walker-decay mechanism (resolvent (I-uB)^-1).")
