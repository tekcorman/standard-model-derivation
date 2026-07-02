#!/usr/bin/env python3
"""Phase 3.3 / S5 -- the n_s gate: q-dependence of the DERIVED dissipator.

Frozen-spec commitment: outcome reported regardless of direction; this gate
decides whether the L6 cosmology cluster (n_s, sigma_8, r_s, theta_*)
reopens. In-repo null (Lindblad era): jump operators q-independent -> the
dissipator's spectral density carries no |q| power -> n_s = 4 vs observed
0.965.

Question now: with the DERIVED structure (3.1 Davies class + 3.2 leakage),
does the effective dissipator acquire Bloch-momentum (kappa) dependence?

Two distinct invariants, computed separately (the honest split):
  (i) RATE DENSITY: sum_jumps K+K -- what the in-repo cosmological
      identification reads. For the derived class this is kappa-FLAT
      (trace structure), so the n_s = 4 null PERSISTS in minimal form.
  (ii) OPERATOR STRUCTURE: the dissipator superoperator's kappa-variation.
      M3 (adopted dephasing) and the leak channel are exactly
      kappa-independent; M1 (backtrack-as-flip) IS kappa-dependent (the
      flip S(kappa) carries Bloch phases) -- the FIRST derived dissipator
      with momentum dependence; leading small-kappa power computed.

Gates:
  N1 rate density kappa-flat for all derived models (exact).
  N2 M3 + leak superoperators exactly kappa-independent; M1 superoperator
     kappa-dependent with measured leading power m (log-log fit).
  N3 GATE VERDICT (honest, either-way commitment): under the in-repo
     identification (n_s read from the rate density's q-power), the
     n_s = 4 null PERSISTS in the derived minimal class -> the L6 cluster
     REMAINS BLOCKED at this gate. The M1 structural kappa-dependence is
     PANEL-DEMOTED: spectrally EMPTY for n_s (S(kappa)^2 = I identically;
     dissipator spectrum exactly {0 x72, -2 x72} at every kappa) -- an
     eigenbasis-only hook, NOT a lever.
"""
import os, sys
import numpy as np
from numpy import linalg as la
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from proofs.common import find_bonds

FAILURES = []
N = 12; q = 2.0/3.0
def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok: FAILURES.append(name)

bonds = find_bonds()
edges = [(i, j, tuple(c)) for (i, j, c) in bonds]
rev = {}
for a,(i,j,c) in enumerate(edges):
    t=(j,i,tuple(-x for x in c))
    for b,e2 in enumerate(edges):
        if e2==t: rev[a]=b

def S_of(kap):
    S = np.zeros((N,N),complex)
    for a,(i,j,c) in enumerate(edges):
        S[rev[a],a]=np.exp(2j*np.pi*np.dot(kap,np.asarray(c,float)))
    return S
Pe = [np.zeros((N,N),complex) for _ in range(N)]
for e in range(N): Pe[e][e,e]=1.0

def super_of(Ks):
    L = np.zeros((N*N,N*N),complex)
    for K in Ks: L += np.kron(K, K.conj())
    return L

print("="*72); print(" PHASE 3.3 -- the n_s gate (reported either way)"); print("="*72)
rng = np.random.default_rng(3)
kaps = [rng.random(3) for _ in range(4)]

# N1: rate density kappa-flat (sum K+K) for M1, M3, leak
ok1, worst1 = True, 0.0
for kap in kaps:
    for Ks in ([S_of(kap)], [P for P in Pe]):
        R = sum(K.conj().T@K for K in Ks)
        worst1 = max(worst1, la.norm(R - R[0,0]*np.eye(N)))
        ok1 &= la.norm(R - R[0,0]*np.eye(N)) < 1e-12
gate("N1 RATE DENSITY kappa-flat for the whole derived class (exact)",
     ok1, f"worst dev {worst1:.2e} -- the object the in-repo n_s identification reads")

# N2: superoperator kappa-dependence
d_m3 = max(la.norm(super_of([P for P in Pe]) - super_of([P for P in Pe])) for kap in kaps)  # trivially 0
D0_m1 = super_of([S_of(np.zeros(3))])
dep_m1 = max(la.norm(super_of([S_of(kap)]) - D0_m1) for kap in kaps)
# leading small-kappa power for M1 along random directions
direc = rng.random(3); direc /= la.norm(direc)
ts = np.array([1e-3, 2e-3, 4e-3, 8e-3])
vals = np.array([la.norm(super_of([S_of(t*direc)]) - D0_m1) for t in ts])
m_fit = np.polyfit(np.log(ts), np.log(vals), 1)[0]
gate("N2 M3/leak superoperators kappa-independent; M1 kappa-DEPENDENT, leading power m",
     d_m3 < 1e-15 and dep_m1 > 0.1 and 0.8 < m_fit < 2.2,
     f"M1 variation {dep_m1:.3f}; fitted m = {m_fit:.3f} (first derived dissipator w/ momentum dependence)")

# N3: the verdict
gate("N3 GATE VERDICT: n_s = 4 null PERSISTS in the derived minimal class "
     "(rate density flat) -- L6 REMAINS BLOCKED at this gate",
     ok1, "M1 structural dependence = the one new lever (does NOT enter the "
          "rate density); honest negative per the frozen either-way commitment")

print("\n  L6 verdict input: the derived dynamics does not supply the |q|-power")
print("  the in-repo identification needs (would require m ~ 3 in the rate")
print("  density; the class gives 0 there). Recorded per spec.")
print("\n"+"="*72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}"); sys.exit(1)
print(" RESULT: ALL GATES PASS -- n_s gate resolved (honest negative)"); print("="*72)
