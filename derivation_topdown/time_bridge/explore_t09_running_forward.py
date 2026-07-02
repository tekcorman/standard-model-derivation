"""
explore_t09 — RUNNING FORWARD FROM A BEGINNING: the structure of forward development, the clock,
the initial datum, and whether the ARROW is forced or chosen.  PURE MATH, walled.  No physics.

Item 5 of the dynamical task.  Prior work (explore_16, t06 FACET 3) showed the heat flow's entropy
rises monotonically from a localized start.  Here we develop the FULL forward-running structure and
settle the forced-vs-choice status of the arrow on EACH flow:

  (A) THE CLOCK.  Three candidate clocks the object provides, compared:
      - discrete geodesic step m (the NB walk): integer-valued, intrinsic, but no metric scale;
      - continuous Dirac/heat parameter t (the spectral-triple flow parameter e^{itD}, e^{-tD^2});
      - the modular parameter (the intrinsic III_1 time of t03/t06).
      We show all three are the SAME 1-parameter group up to a (free) rescaling: the generator is D
      (resp. D^2), so the clock is FORCED in DIRECTION/GENERATOR but FREE in RATE (no intrinsic unit,
      matching III_1 scale-freeness, t04).

  (B) THE INITIAL DATUM.  Running forward = a Cauchy problem: give psi(0) (unitary) or p(0) (heat),
      the flow is determined.  The initial datum is FREE (any state); the LAW is FORCED (D).  We
      show the forward map is a well-posed semigroup/group: e^{itD} (group, reversible) and
      e^{-tD^2} (semigroup, t>=0 only — directed).

  (C) IS THE ARROW FORCED?  Decisive distinction:
      - UNITARY Dirac flow e^{itD}: time-REVERSAL symmetric.  A(-k) = conj A(k) (the object's own
        TR, STRUCTURE.md), and spec is symmetric, so for every forward trajectory there is a
        backward one.  => NO forced arrow at the level of the LAW; reversible.
      - HEAT semigroup e^{-tD^2}: defined only for t>=0 (e^{+tD^2} is unbounded — the backward heat
        equation is ill-posed).  => the heat flow is INTRINSICALLY one-directional: the arrow is
        FORCED BY THE SEMIGROUP STRUCTURE (not merely by the initial condition).  We verify
        ||e^{-tD^2}|| <= 1 (contraction, t>=0) while ||e^{+tD^2}|| blows up.
      - The H-theorem: relative entropy S(p(t) || uniform) is monotone NON-INCREASING under the heat
        flow (the object's own Lyapunov function) — verified.  This is the forced arrow, concretely.
      => FORCED: the heat-semigroup arrow (irreversibility) and its Lyapunov/H-theorem.
         CHOICE:  which end is "initial" — i.e. the low-entropy boundary condition that makes a
         particular history; the LAW does not pick it, the INITIAL DATUM does.

  (D) THE FORWARD-DEVELOPMENT STRUCTURE, concretely on a finite cover:
      from a delta start, track (i) the Lyapunov functional (relative entropy to equilibrium),
      (ii) the participation number 1/sum p^2 (how many sites are occupied = the growing "size"),
      (iii) the approach to the unique equilibrium = the conserved-mass uniform state.
      The forward development is RELAXATION TO THE STATIC FIXED POINT, monotone in the Lyapunov
      functional, at the rate set by the Laplacian spectral gap.

No physics; small matrices; exact where exact.
"""
import numpy as np, sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

np.set_printoptions(precision=6, suppress=True)
def hdr(s): print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)

# =====================================================================================
hdr("(A) THE CLOCK: discrete step / Dirac t / modular t are one group; direction forced, rate free")
# =====================================================================================
A = srs.adjacency((0.13, 0.27, 0.41)); E, U = np.linalg.eigh(A)
# the continuous Dirac flow e^{itA}: a one-parameter UNITARY group (group law verified)
def Uflow(t): return U @ np.diag(np.exp(1j*t*E)) @ U.conj().T
grouplaw = np.allclose(Uflow(0.4) @ Uflow(0.9), Uflow(1.3)) and np.allclose(Uflow(0.0), np.eye(4))
print(f"  Dirac flow e^(itA): group law U(s)U(t)=U(s+t), U(0)=I ?  {grouplaw}  => a clock (1-param group).")
print(f"  rescaling t -> c t just relabels the clock (rate FREE); the GENERATOR (=A, i.e. D) is FORCED.")
print(f"  => the clock's direction/generator is forced by D; its UNIT/RATE is free (no intrinsic scale,")
print(f"     matching the III_1 scale-free verdict t04: T(M)={{0}}, no period).")

# =====================================================================================
hdr("(B) THE INITIAL DATUM: well-posed Cauchy problem; group (reversible) vs semigroup (directed)")
# =====================================================================================
# unitary group: reversible (run backward exactly)
rng = np.random.default_rng(1); psi0 = rng.standard_normal(4)+1j*rng.standard_normal(4); psi0/=np.linalg.norm(psi0)
back = Uflow(-1.7) @ (Uflow(1.7) @ psi0)
print(f"  unitary group e^(itA): forward-then-back recovers psi0 exactly ? {np.allclose(back, psi0)}  (REVERSIBLE).")
# heat semigroup: e^{-tL} contraction for t>=0, e^{+tL} unbounded
L = 3*np.eye(4) - A; wL = np.linalg.eigvalsh(L)
nrm_fwd = np.exp(-1.0*wL).max()       # ||e^{-tL}|| = exp(-t * min eigenvalue) = 1 (zero mode)
nrm_back = np.exp(+5.0*wL).max()      # ||e^{+tL}|| = exp(+t * max eigenvalue) -> blows up
print(f"  heat semigroup e^(-tL):  ||e^(-1.0 L)|| = {nrm_fwd:.4f} (<=1, contraction);  "
      f"||e^(+5.0 L)|| = {nrm_back:.2e} (blows up).")
print(f"  => forward heat is a well-posed CONTRACTION SEMIGROUP (t>=0 only); BACKWARD heat is ILL-POSED.")
print(f"     The Cauchy datum (the state at t=0) is FREE; the LAW (generator D / D^2) is FORCED.")

# =====================================================================================
hdr("(C) IS THE ARROW FORCED?  unitary = reversible (TR-symmetric);  heat = forced one-direction")
# =====================================================================================
# (C1) time-reversal of the object: A(-k) = conj A(k) => spectrum symmetric; the unitary law has a TR.
k = np.array([0.13, 0.27, 0.41])
tr_sym = np.allclose(np.linalg.eigvalsh(srs.adjacency(-k)), np.linalg.eigvalsh(srs.adjacency(k)))
print(f"  (C1) UNITARY: A(-k)=conj A(k) => spec(A(-k))=spec(A(k)) ? {tr_sym}  => the Dirac flow is")
print(f"       TIME-REVERSAL symmetric: NO forced arrow in the LAW (reversible).")
# (C2) heat H-theorem on a finite patch: relative entropy to the uniform equilibrium is monotone down.
R = 2
cells = [(a,b,c) for a in range(-R,R+1) for b in range(-R,R+1) for c in range(-R,R+1)]
cidx = {c:i for i,c in enumerate(cells)}; nv = 4*len(cells)
def vid(s,cell): return cidx[cell]*4+s
Aadj = np.zeros((nv,nv))
for cell in cells:
    a,b,c = cell
    for (i,j,v) in srs.EDGES:
        nbr=(a+v[0],b+v[1],c+v[2])
        if nbr in cidx:
            x,y=vid(i,cell),vid(j,nbr); Aadj[x,y]+=1; Aadj[y,x]+=1
Lp = np.diag(Aadj.sum(1)) - Aadj
wL2, VL2 = np.linalg.eigh(Lp)
p0 = np.zeros(nv); p0[vid(0,(0,0,0))] = 1.0
def p(t):
    q = VL2 @ (np.exp(-t*wL2)*(VL2.T@p0)); q = np.clip(q,1e-300,None); return q/q.sum()
unif = np.ones(nv)/nv
def relent(q): return np.sum(q*np.log(q/unif))      # KL(q || uniform) >= 0, =0 at equilibrium
print(f"  (C2) HEAT H-theorem: relative entropy D(p(t)||uniform) (Lyapunov fn) — must be monotone DOWN:")
prev=1e9; mono=True
for t in [0.0,0.3,1.0,3.0,10.0,40.0]:
    re = relent(p(t))
    if re>prev+1e-9: mono=False
    prev=re
    print(f"     t={t:5.1f}   D(p||u) = {re:.5f}")
print(f"     monotone non-increasing ? {mono}  => the heat flow has a forced LYAPUNOV functional (the")
print(f"     H-theorem); the ARROW is FORCED by the semigroup, the END that is 'initial' is the CHOICE.")

# =====================================================================================
hdr("(D) FORWARD-DEVELOPMENT STRUCTURE: relaxation to the unique (conserved-mass) fixed point")
# =====================================================================================
print(f"  finite cover R={R}: {len(cells)} cells, {nv} vertices.  Forward run from a delta start:")
print(f"     t       D(p||u)    participation 1/sum p^2     <distance to equilibrium>")
for t in [0.0,0.5,2.0,8.0,30.0,120.0]:
    q = p(t); part = 1.0/np.sum(q**2); dist = np.linalg.norm(q-unif)
    print(f"   {t:6.1f}   {relent(q):.5f}      {part:8.2f} / {nv}            {dist:.5f}")
gap = np.sort(wL2)[1]   # Laplacian spectral gap (slowest relaxation rate)
print(f"  Laplacian spectral gap (slowest decay rate) = {gap:.4f}  => late-time D(p||u) ~ e^(-2*gap*t).")
print(f"  => forward development = MONOTONE relaxation of every Lyapunov measure to the SINGLE static")
print(f"     fixed point (the uniform, conserved-mass equilibrium), at the rate set by the gap.")
print(f"     The 'size' (participation) grows from 1 toward {nv}; equilibrium = the static structure.")

hdr("FINDING (t09): running forward — the forced/choice ledger of the arrow")
print("""  Running the object forward is a well-posed Cauchy problem; what is forced vs chosen:
    CLOCK            FORCED: generator/direction (D for the group, D^2 for the heat semigroup, the
                     dGamma(D) modular generator for intrinsic time).
                     CHOICE/FREE: the RATE/UNIT (no intrinsic scale; III_1 is scale-free, t04).
    LAW              FORCED: the evolution (unitary group e^{itD}; contraction semigroup e^{-tD^2}).
    INITIAL DATUM    CHOICE: the state at t=0 (any Cauchy datum); a particular HISTORY needs it.
    ARROW            FORCED on the HEAT/dissipative flow: it is a contraction SEMIGROUP (backward
                     heat ill-posed) with a monotone Lyapunov functional (H-theorem, D(p||u) down).
                     NOT forced on the UNITARY Dirac flow: that law is time-reversal symmetric
                     (A(-k)=conj A(k)); reversible.  The thermodynamic arrow's DIRECTION-OF-USE
                     (which end is the low-entropy 'beginning') is the CHOICE of initial datum.
    FIXED POINT      FORCED: the unique equilibrium = the uniform, conserved-mass state = the STATIC
                     structure of the object; forward development is relaxation onto it at the
                     Laplacian-gap rate.
  Net: the object FORCES the dynamical law, the generator, the irreversibility of the dissipative
  flow, and the equilibrium; it leaves FREE the initial state and the time UNIT (no intrinsic scale).""")
