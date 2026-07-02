#!/usr/bin/env python3
"""Phase 3.2 / S1 -- the step isometry: the record kept explicit.

Bet spec: docs/scoping/phase3_2_bet_spec_2026-06-11.md (frozen, hashed).
S1 builds the Stinespring isometry of the observer step with the discarded
record explicit, and gates consistency with the Phase 3.1 channels:

    V|psi> = sqrt(q) (U|psi>)|0>_rec + sqrt(1-q) sum_e (P_e|psi>)|e>_rec

(record space: |0> = "coherent NB step", |e> = "cancellation at edge e").

Gates:
  S1a V is an exact isometry (V+V = I_12).
  S1b Tr_rec V rho V+ = Phi_M3(rho) (the 3.1 adopted-class channel) on a
      basis of density matrices -- the 3.1 dynamics is the visible marginal
      of a genuinely kept record.
  S1c the record marginal is a valid state and its |0>-weight is q = 2/3
      on the maximally mixed input (the cancellation record carries 1/k).
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
P = np.array([0.25,0.25,0.25])
C = np.zeros((N,N),complex)
for a,(i,j,c) in enumerate(edges):
    for b,(i2,j2,c2) in enumerate(edges):
        if i2==i: C[b,a]=2/3-(1 if b==a else 0)
S = np.zeros((N,N),complex)
for a,(i,j,c) in enumerate(edges):
    S[rev[a],a]=np.exp(2j*np.pi*np.dot(P,np.asarray(c,float)))
U = S@C
Pe = [np.zeros((N,N),complex) for _ in range(N)]
for e in range(N): Pe[e][e,e]=1.0

R = N+1  # record dim: |0> + |e>, e=1..12
V = np.zeros((N*R, N), complex)
for col in range(N):
    psi = np.zeros(N,complex); psi[col]=1
    out = np.zeros((N,R),complex)
    out[:,0] = np.sqrt(q)*(U@psi)
    for e in range(N):
        out[:,1+e] = np.sqrt(1-q)*(Pe[e]@psi)
    V[:,col] = out.reshape(-1)

print("="*72); print(" PHASE 3.2/S1 -- step isometry with explicit record"); print("="*72)
gate("S1a V is an exact isometry (V+V = I)", la.norm(V.conj().T@V - np.eye(N)) < 1e-12,
     f"||V+V - I|| = {la.norm(V.conj().T@V - np.eye(N)):.2e}")

def vis_marginal(rho):
    big = V@rho@V.conj().T
    T = big.reshape(N,R,N,R)
    return np.einsum('irjr->ij', T)
def rec_marginal(rho):
    big = V@rho@V.conj().T
    T = big.reshape(N,R,N,R)
    return np.einsum('iris->rs', T)
def Phi3(rho):
    return q*U@rho@U.conj().T + (1-q)*sum(Pp@rho@Pp for Pp in Pe)

rng = np.random.default_rng(7); worst=0
for _ in range(6):
    A = rng.standard_normal((N,N)) + 1j*rng.standard_normal((N,N))
    rho = A@A.conj().T; rho /= np.trace(rho)
    worst = max(worst, la.norm(vis_marginal(rho)-Phi3(rho)))
gate("S1b Tr_rec V rho V+ = Phi_M3(rho) (3.1 channel = visible marginal)",
     worst < 1e-12, f"worst {worst:.2e}")

rr = rec_marginal(np.eye(N)/N)
gate("S1c record marginal valid; coherent weight = q = 2/3 on mixed input",
     abs(np.trace(rr)-1) < 1e-12 and abs(rr[0,0].real - q) < 1e-12
     and la.eigvalsh((rr+rr.conj().T)/2).min() > -1e-12,
     f"rec[0,0] = {rr[0,0].real:.6f}")

print("\n  S1 established: the 3.1 dynamics is the visible marginal of a kept")
print("  record. S2 next: the waterline split of the record (dark face).")
print("\n"+"="*72)
if FAILURES:
    print(f" RESULT: {len(FAILURES)} GATE(S) FAILED: {FAILURES}"); sys.exit(1)
print(" RESULT: ALL GATES PASS -- S1 isometry established"); print("="*72)
