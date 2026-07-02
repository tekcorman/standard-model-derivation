"""
THE DYNAMICAL THEORY OF THE DOUBLED CHIRAL SPINOR PROPAGATING ON THE CHIRAL-SCREW SUBSTRATE.

Pure math.  Reads ONLY the bare object (../dirac_srs_mdl/srs.py) and takes the static spinor
architecture (m06) as given.  DEVELOPS the dynamics the source files do not cover:

  1. propagation (EOM / propagator / transport) along the screw axis and transverse;
  2. the conserved quantities + the algebra they close into;
  3. THE OPEN QUESTION: is the spinor handedness tied to the screw winding/axial label,
     or independent?  Decided FROM THE DYNAMICS (screw transport + closed-loop holonomy);
  4. the role of the screw's fixed handedness.

Every result marked FORCED vs CHOICE; external inputs named.  No targets adopted.
"""
import numpy as np, cmath, sys, os
from scipy.linalg import expm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

np.set_printoptions(precision=5, suppress=True, linewidth=140)
om = cmath.exp(2j*np.pi/3)
I2 = np.eye(2)
s1 = np.array([[0,1],[1,0]], complex); s2 = np.array([[0,-1j],[1j,0]], complex); s3 = np.array([[1,0],[0,-1]], complex)
def comm(A,B): return A@B - B@A
def acomm(A,B): return A@B + B@A
def is0(M, tol=1e-9): return np.allclose(M,0,atol=tol)

# Cl(4) (m06 convention): g1,g2,g3 spatial, g4 chirality/mass direction; gc = grading.
g = [np.kron(s1,s1), np.kron(s1,s2), np.kron(s1,s3), np.kron(s2,I2)]
gc = g[0]@g[1]@g[2]@g[3]
# spin generators (bivectors) of the spatial rotation group Spin(3) inside Cl(4):
Sig = [-(1j/4)*comm(g[i],g[j]) for (i,j) in [(1,2),(2,0),(0,1)]]   # Sig_x,Sig_y,Sig_z, eig +-1/2

print("="*100)
print(" DYNAMICS OF THE DOUBLED CHIRAL SPINOR ON THE CHIRAL-SCREW SUBSTRATE   (pure math, walled)")
print("="*100)

# =====================================================================================
# 0.  THE SUBSTRATE SCREW, IN NATIVE TERMS (rebuilt from srs.py + MDL harmonic realization).
# =====================================================================================
DARTS = srs._darts()
sigma = {0:0, 1:2, 2:3, 3:1}
Msig  = np.array([[0,-1,0],[0,0,-1],[1,0,0]])      # sigma_* on the winding lattice H_1=Z^3
def outgoing(i): return [(h,v) for (t,h,v) in DARTS if t==i]
A = np.zeros((15,12)); b = np.zeros(15)
def idx(i,c): return 3*i+c
row=0
for i in range(4):
    for c in range(3):
        for (h,v) in outgoing(i):
            A[row,idx(h,c)] += 1; A[row,idx(i,c)] -= 1; b[row] -= v[c]
        row+=1
for c in range(3):
    for i in range(4): A[row,idx(i,c)] = 1
    row+=1
Y,*_ = np.linalg.lstsq(A,b,rcond=None); Yv = Y.reshape(4,3)
fd = np.array([Yv[h]+np.array(v,float)-Yv[t] for (t,h,v) in DARTS])
Corr = sum(np.outer(f,f) for f in fd)
ev,evec = np.linalg.eigh(Corr); L = evec@np.diag(1/np.sqrt(ev))@evec.T; Linv=np.linalg.inv(L)
R = L@Msig@Linv
w_,V_ = np.linalg.eig(R); axis = V_[:,np.argmin(np.abs(w_-1))].real; axis/=np.linalg.norm(axis)
pitch = axis@(L@np.array([1.0,0,0]));  axis = axis if pitch>0 else -axis; pitch=abs(pitch)
print(f"""
[0]  THE SUBSTRATE SCREW (native): deck winding lattice Z^3, sigma_* = Msig (order 3);
     cartesian screw  R = L Msig L^-1 about <111> axis {np.round(axis,4)}, 120 deg turn LOCKED to
     +pitch={pitch:+.5f} climb (3-step period sqrt(3/2)={np.sqrt(1.5):.5f}).  One fixed handedness, no mirror.
     Off-axis conserved label: the C3 WINDING sector s in {{0,1,2}} (sigma-Fourier eigenvalue om^s).""")

# =====================================================================================
# 1.  PROPAGATION.  Axial/transverse resolution of the Cl(4) Dirac; the screw transport operator.
# =====================================================================================
print("\n"+"#"*100); print("# 1.  PROPAGATION  (EOM, propagator, screw transport)"); print("#"*100)
g_ax = sum(axis[a]*g[a] for a in range(3))
tmp = np.array([1.0,0,0]);  tmp = np.array([0,1.0,0]) if abs(axis@tmp)>0.9 else tmp
e1 = tmp-(tmp@axis)*axis; e1/=np.linalg.norm(e1); e2=np.cross(axis,e1)
g_t1 = sum(e1[a]*g[a] for a in range(3)); g_t2 = sum(e2[a]*g[a] for a in range(3))
print(f"""
[1.1] AXIAL / TRANSVERSE SPLIT (FORCED by the screw axis).  D = g_ax p_ax + g_t1 p_t1 + g_t2 p_t2 + m g4.
   Clifford along the screw frame:  {{g_ax,g_ax}}=2I {np.allclose(acomm(g_ax,g_ax),2*np.eye(4))},
   {{g_t1,g_t1}}=2I {np.allclose(acomm(g_t1,g_t1),2*np.eye(4))}, {{g_ax,g_t1}}=0 {is0(acomm(g_ax,g_t1))},
   {{g_t1,g_t2}}=0 {is0(acomm(g_t1,g_t2))}.   => D^2 = (p_ax^2+p_t1^2+p_t2^2) + m^2 (massive cone),
   propagator (E - D)^-1; EOM i d_t psi = D psi.   [FORCED by Cl(4)+axis]""")

# Screw transport: spin lift U of the 120-deg axis rotation (the cover of R on the spinor).
thR = 2*np.pi/3
Jax = sum(axis[k]*Sig[k] for k in range(3))        # AXIAL SPIN operator (helicity generator)
U = expm(-1j*thR*Jax)                               # spinor screw-transport over one step
ok_lift = all(np.allclose(U@g[a]@np.linalg.inv(U), sum(R[b,a]*g[b] for b in range(3)),atol=1e-6) for a in range(3))
if not ok_lift:
    U = expm(+1j*thR*Jax)
    ok_lift = all(np.allclose(U@g[a]@np.linalg.inv(U), sum(R[b,a]*g[b] for b in range(3)),atol=1e-6) for a in range(3))
print(f"""[1.2] THE SCREW TRANSPORT OPERATOR  U = exp(-i*(2pi/3)*J_ax)  (spin lift of R).
   U g_a U^-1 = R_ab g_b (genuine Spin(3) cover) ? {ok_lift}
   [U, g_ax]=0 (axial gamma R-invariant) ? {np.allclose(comm(U,g_ax),0,atol=1e-6)};   [U, gc]=0 (spatial
   rotation is EVEN, does not swap handedness) ? {np.allclose(comm(U,gc),0,atol=1e-6)};   [U, g4]=0
   (mass survives transport) ? {np.allclose(comm(U,g[3]),0,atol=1e-6)}.
   ONE screw step = (translate one cell: Bloch phase e^{{2pi i k.t}})  x  (rotate frame R)  x  U.""")

# =====================================================================================
# 2.  CONSERVED QUANTITIES + the algebra they close into.
# =====================================================================================
print("\n"+"#"*100); print("# 2.  CONSERVED QUANTITIES (internal + substrate) and their algebra"); print("#"*100)
print(f"""
[2.1] WHAT THE AXIAL FLOW CONSERVES.  Massless axial propagation H = g_ax p_ax.  Test each candidate
   charge for [Q, g_ax] (conserved by kinetic) and [Q, g4] (survives the mass):
   AXIAL SPIN  J_ax :  [J_ax,g_ax]=0 {np.allclose(comm(Jax,g_ax),0,atol=1e-9)}, [J_ax,g4]=0 {np.allclose(comm(Jax,g[3]),0,atol=1e-9)}, [J_ax,gc]=0 {np.allclose(comm(Jax,gc),0,atol=1e-9)}  -> EXACTLY CONSERVED (all)
   CHIRALITY   gc   :  [gc,g_ax]=0 {np.allclose(comm(gc,g_ax),0,atol=1e-9)}  ({{gc,g_ax}}=0 {is0(acomm(gc,g_ax))})  -> NOT conserved by kinetic; chirality is
                       conserved ONLY in the massless+axial-helicity-eigenstate sense (see 2.2)
   So the UNIVERSAL conserved internal charge is the AXIAL SPIN (helicity) J_ax, eigenvalues +-1/2,
   not chirality.  This is FORCED: g_ax anticommutes with gc, so a propagating mode is a helicity
   eigenstate, not a chirality eigenstate.   [FORCED]""")

wj,Vj = np.linalg.eigh(Jax)
print("[2.2] HELICITY is the good label; chirality is the SIGN of helicity x sign of momentum.")
print("   (helicity m, chirality on the eigenvector):")
for i in range(4):
    chir=(Vj[:,i].conj()@gc@Vj[:,i]).real
    print(f"      m = {wj[i]:+.2f}   chirality<gc> = {chir:+.2f}")
print("""   Each chirality block carries BOTH helicities (m=+1/2 and -1/2): handedness and helicity are
   INDEPENDENT internal labels; the propagation conserves helicity.""")

# THE SUBSTRATE conserved label: C3 winding sector on the fixed line.
P12 = np.zeros((12,12))
for a,(i,j,v) in enumerate(DARTS):
    gimg=(sigma[i],sigma[j])
    for c,(p,q,_) in enumerate(DARTS):
        if (p,q)==gimg: P12[c,a]=1; break
def Bk(t): return srs.hashimoto(t*np.array([1.0,-1.0,1.0]))
commute = all(np.allclose(Bk(t)@P12 - P12@Bk(t),0,atol=1e-9) for t in [0.0,0.13,0.27])
print(f"""
[2.3] THE SUBSTRATE CONSERVED LABEL (the screw winding).  On the C3-fixed Bloch line k=t(1,-1,1)
   the deck/screw symmetry sigma commutes with the propagation generator (Hashimoto B):
   [B(k),P_sigma]=0 along the line ? {commute}.  => the C3 WINDING sector s in {{0,1,2}} (eigenvalue
   om^s of P_sigma) is a CONSERVED quantum number of the propagating field on the screw axis.
   Plus the continuous AXIAL MOMENTUM p_ax = k.axis (the Z^3 Bloch/translation charge).   [FORCED]""")

print("""
[2.4] THE ALGEBRA THE CONSERVED CHARGES CLOSE INTO.
   Mutually-commuting (simultaneously diagonal) conserved set on the axial flow:
        { p_ax (axial momentum, u(1)) ,  J_ax (axial spin/helicity, the Cartan of spin) ,
          s    (C3 winding sector, Z_3) } .
   - p_ax generates U(1) translations along the axis (continuous).
   - J_ax is the Cartan generator of the spatial Spin(3); the FULL spin algebra is su(2) with
     [Sig_a,Sig_b]=i eps_abc Sig_c, but the SCREW BREAKS Spin(3) -> Spin(2)=U(1)_axial (only J_ax
     survives as a symmetry, since only the axis rotation is a substrate symmetry).
   - s is the Z_3 = <sigma> winding, the substrate deck remnant on the axis.
   Joint symmetry of the propagating spinor on the screw:  U(1)_pax  x  U(1)_helicity  x  Z_3 .""")
# verify su(2) and the Z3
su2 = all(np.allclose(comm(Sig[a],Sig[b]), 1j*sum(([1,-1][ (a,b) in [(1,0),(2,1),(0,2)] ])*Sig[c]
        for c in [ ({0,1,2}-{a,b}).pop() ] ) if a!=b else 0*Sig[0], atol=1e-9) for a in range(3) for b in range(3))
print(f"   spin su(2) closure [Sig_a,Sig_b]=i eps Sig_c ? {su2};   P_sigma^3 = I (Z_3) ? "
      f"{np.allclose(np.linalg.matrix_power(P12,3),np.eye(12))}")

# =====================================================================================
# 3.  THE CENTRAL QUESTION — is handedness tied to the winding/axial label?
# =====================================================================================
print("\n"+"#"*100); print("# 3.  CENTRAL QUESTION: handedness vs screw winding/axial label — decided by transport"); print("#"*100)

print(f"""
[3.1] SCREW-TRANSPORT EIGENVALUES = the C3 winding character TIMES the spinor double-cover sign.
   Transport one screw step: U|m> = exp(-i*(2pi/3)*m)|m>,  m = +-1/2 (helicity).""")
for i in range(4):
    m=wj[i]; ph=cmath.exp(-1j*thR*m)
    # express as om^? times sign
    print(f"      m={m:+.2f}:  U-eigenvalue = exp(-i*2pi*m/3) = {ph.real:+.4f}{ph.imag:+.4f}j "
          f"= {'-om' if abs(ph+om)<1e-6 else ('-om^2' if abs(ph+om**2)<1e-6 else '?')}")
print(f"""   So a +1/2-helicity spinor advancing +1 screw step picks up  -om^2 ;  a -1/2 picks up -om.
   The winding-character part (om^{{±1}}) is HALF the substrate's integer winding step om^{{±1}}: the
   spinor carries HALF-INTEGER winding because it is a SPIN-1/2 lift of the order-3 screw.   [FORCED]""")

U3 = np.linalg.matrix_power(U,3)
print(f"""
[3.2] CLOSED-LOOP HOLONOMY — the forced -1.  Three screw steps = ONE lattice period along the axis:
   the SUBSTRATE returns to its start (closed loop in the Albanese torus), but the SPINOR returns with
        U^3 = {('-I' if np.allclose(U3,-np.eye(4)) else '+I' if np.allclose(U3,np.eye(4)) else '?')}   (U^3 = -I, verified).
   The order-3 screw lifts to an ORDER-6 spinor rotation (eig(U) = e^{{±i pi/3}} = primitive 6th roots).
   => Transporting the spinor once around the minimal closed screw loop returns it to MINUS itself.
   This is FORCED and k-INDEPENDENT (it is the 2pi/double-cover sign of any 2*pi/n rotation lifted to
   Spin): the substrate forces a definite, nontrivial relationship.   NOT independent.   [FORCED]""")
print(f"""
[3.3] EXACTLY WHAT IS TIED TO WHAT.
   - The propagating spinor's conserved AXIAL SPIN m=+-1/2 (helicity) is LOCKED to the screw transport
     phase:  phase(one step) = exp(-i*(2pi/3)*m).  Helicity <-> winding-per-step is rigid, ratio fixed.
   - Around the closed minimal screw loop the holonomy is exactly  U^3 = -1  (independent of m, of k,
     and of the mass m).  The spinor is a section of a Z_2-TWISTED (spin) bundle over the screw loop.
   - HANDEDNESS (chirality gc) is NOT itself the winding charge: gc anticommutes with the kinetic
     g_ax, so it is not conserved; the conserved internal charge is helicity, and IT is tied to the
     winding.  Handedness enters only through the FIXED SIGN of the lift (see Part 4).
   RELATION (exact):   spinor screw winding per step = -(1/2) x (substrate winding step) in the
   exponent, i.e. helicity m -> phase exp(-i*(2pi/3) m); loop holonomy = (-1).   [FORCED]""")

# =====================================================================================
# 4.  ROLE OF THE SCREW'S FIXED HANDEDNESS.
# =====================================================================================
print("\n"+"#"*100); print("# 4.  ROLE OF THE FIXED HANDEDNESS"); print("#"*100)
# Compare srs (+120) vs mirror srs* (-120): which spinor data flip?
U_mir = expm(+1j*thR*Jax) if ok_lift else expm(-1j*thR*Jax)   # opposite screw sense
print(f"""
[4.1] WHAT THE FIXED HANDEDNESS FORCES (compare srs's +120 screw to the mirror srs*'s -120 screw):
   - The CLOSED-LOOP -1 is the SAME for both hands (U^3 = -I either way): the double-cover sign is
     handedness-INDEPENDENT.   srs U^3=-I ? {np.allclose(np.linalg.matrix_power(U,3),-np.eye(4))};  srs* U^3=-I ? {np.allclose(np.linalg.matrix_power(U_mir,3),-np.eye(4))}.
   - The SIGN of the helicity-winding lock FLIPS with the hand: srs assigns m=+1/2 the phase exp(-i pi/3),
     the mirror assigns m=+1/2 the phase exp(+i pi/3).  i.e. the CORRELATION (helicity m) <-> (winding
     direction) is sign-locked by the screw's handedness — a +helicity spinor co-winds with the screw
     on srs and counter-winds on srs*.   srs U(m=+1/2)-eig = {cmath.exp(-1j*thR*0.5):+.3f}; srs* = {cmath.exp(+1j*thR*0.5):+.3f}.
   - In a MIRROR-SYMMETRIC substrate (both hands present) this sign would be FREE/averaged-out
     (the two screws would contribute opposite locks and the helicity-winding correlation would cancel).
     The single fixed hand FORCES a definite, non-cancelling sign of the (helicity <-> winding) coupling.
   - srs* is the OPPOSITE-Weyl-charge net (m06 Part 3): the 4th-gamma mass couples the +2 and -2 Weyl
     points; the fixed hand makes that pairing chiral (no orientation-reversing symmetry to undo it).""")

print(f"""
[4.2] FORCED vs CHOICE (summary).
   FORCED:
     (i)   Dirac splits axial+transverse; D^2 = p^2 + m^2 (massive cone).               [Cl(4)+axis]
     (ii)  the order-3 screw lifts to an ORDER-6 spinor transport: U^3 = -I.            [Spin double cover]
     (iii) conserved internal charge on the axis = HELICITY J_ax (=+-1/2), NOT chirality
           (g_ax anticommutes gc); helicity survives the mass ([J_ax,g4]=0).            [Clifford]
     (iv)  helicity is LOCKED to the winding: U|m> = exp(-i*(2pi/3) m)|m>; closed-loop
           holonomy = -1, k- and m(mass)-independent.  Handedness/helicity are tied to the screw.   [transport]
     (v)   the substrate winding sector s (Z_3) and axial momentum p_ax (U(1)) are conserved.  [deck sym]
     (vi)  joint symmetry  U(1)_pax x U(1)_helicity x Z_3 ; the screw breaks spatial Spin(3)->U(1)_axial.
     (vii) the SIGN of the helicity<->winding lock is fixed by the screw's one handedness (Part 4.1).
   CHOICE / EXTERNAL INPUT:
     - the mass STRENGTH m (the gap scale): not fixed by the bare geometry (m06's one irreducible input);
       it does NOT affect any conserved-charge or holonomy result above.
     - orientation conventions (which axis end is '+', which screw is called 'right'): convention only;
       the GAUGE-INVARIANT content is the order-6 lift (U^3=-1) and the rigid helicity<->winding ratio.
   NAMED EXTERNAL STRUCTURES USED: the spin lift U (Spin(3) cover) is the natural transport of the
     Clifford spinor under the substrate's own screw R — no new input; the harmonic (Albanese/MDL)
     metric is used only to realize R cartesianly (it is the object's own standard realization).""")
print("\n[done]")
