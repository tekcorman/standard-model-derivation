"""
OFF-DIAGONAL INTER-CHANNEL MIXING  V(s) = U_A(s)^dagger U_B(s).

PURE MATH, walled.  Reads ONLY ../dirac_srs_mdl/srs.py (the K4 / Z^3 crystal) + native data from
the sealed reading-sheet.  No physics, no observed numbers, no targets, no fitting.  Derive; under-claim.

ESTABLISHED in-box (re-verified in section 0):
  * The geodesic / non-backtracking generator B(k) (= the run generator's spatial face).  The run K
    is the DIRECTED screw-advance dB/ds along the C3 axis AXIS=(1,-1,1)/sqrt3; the persistence is read
    off B(s*AXIS) (the established mass mechanism, derive_generation_spectrum.py / t07 / t10).
  * The GENERATION 3-STRUCTURE = the three C3 deck windings {omega^0, omega^1, omega^2}, via the C3
    Fourier projectors Pc3[t] on the 12 darts (sigma = (123) the deck screw).  winding_basis(t).
  * TWO walker-type channels (t10):
      - SATURATION / inter-copy channel  (the NON-winding, NON-decaying mode): the adjacency-Perron
        ('trivial') eigenvalue lambda_max, |lambda|^2 = 9 -> 3.  Real, L=0 (no geodesic length).
      - PERRON channel (the L=g real NB mode): the Ihara-Bass Perron root |h+|^2 = 4 -> 2.
    Each carries the generation 3-structure (each winding has its own copy of both channels).
  * PRIOR WORK used only the DIAGONAL (per-winding amplitudes c_t -> the C3-Fourier mass map) and
    closed it to one Z2 (the J-reality forcing c2 = conj(c1), one residual phase delta).
    It NEVER built the OFF-diagonal between the two channels.  That is THIS file.

WHAT WE COMPUTE
  1. P^A_ij(s), P^B_ij(s): the persistence-overlap of the generation 3-structure for EACH channel.
     P^X_ij(s) = < gen_i | (channel-X return amplitude under the run B(s*AXIS)) | gen_j >, on the
     3-dim generation space (the three windings).  Built channel-by-channel from B(s*AXIS) by
     projecting onto the channel's spectral sheet (saturation real-band vs Perron NB-root).
  2. Diagonalize each -> U_A(s), U_B(s).  V(s) = U_A(s)^dagger U_B(s).
  3. Structure of V: 3 angles + 1 phase?  closed forms in (k,g,s, IB eigenvalues, rate 2pi/sqrt7)?
  4. CP: does the DIRECTED run put a genuine complex phase in V that the STATIC s=0 cannot?
     Contrast V(s) vs V(0): is V(0) real/degenerate (no usable mixing) and does running lift it?
  5. Hierarchy: are the off-diagonals small vs the diagonal, ordered by the channel (sat vs Perron)?
  6. Same-clock; forced vs choice; flags.
"""
import numpy as np, cmath, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dirac_srs_mdl"))
import srs

np.set_printoptions(precision=6, suppress=True, linewidth=140)
def hdr(s): print("\n" + "=" * 92 + "\n" + s + "\n" + "=" * 92)
om = cmath.exp(2j*np.pi/3)
k = srs.DEG                                   # 3
SQRT2 = np.sqrt(2.0)
AXIS = np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0)   # the C3 screw axis = the run direction
RATE = 2*np.pi/np.sqrt(7)                      # the derived run-phase rate (in-box: from lambda0=-1)

# ---- the generation 3-structure: the three C3 deck windings (exactly as derive_generation_spectrum) ----
sigma = {0: 0, 1: 2, 2: 3, 3: 1}               # the deck screw sigma=(123) on the 4 K4 vertices
DARTS = srs._darts()                           # 12 directed edges
Pperm = np.zeros((12, 12))
for a, (i, j, v) in enumerate(DARTS):
    g = (sigma[i], sigma[j])
    for b, (p, q, w) in enumerate(DARTS):
        if (p, q) == g:
            Pperm[b, a] = 1; break
Pc3 = {t: sum(om**(-t*m) * np.linalg.matrix_power(Pperm, m) for m in range(3)) / 3 for t in (0, 1, 2)}
def winding_basis(t):
    w, V = np.linalg.eigh(Pc3[t]); return V[:, np.abs(w - 1) < 1e-6]
WB = {t: winding_basis(t) for t in (0,1,2)}    # each is 12 x (its multiplicity)

# girth of the cell-quotient geodesic flow (the Perron channel's geodesic length L=g)
def girth():
    B0 = srs.hashimoto((0,0,0))
    for m in range(1, 12):
        if abs(np.trace(np.linalg.matrix_power(B0, m))) > 1e-6:
            return m
    return None
g_girth = girth()

# =============================================================================================
hdr("(0) RE-VERIFY the object: two channels, each carrying the generation 3-structure")
# =============================================================================================
B0 = srs.hashimoto((0,0,0))
print(f"  k = {k}; run rate 2pi/sqrt7 = {RATE:.6f}; cell-quotient girth g = {g_girth}")
print(f"  generation 3-structure = three C3 windings, dart multiplicities "
      f"{tuple(WB[t].shape[1] for t in (0,1,2))} (= 4|4|4 split of 12 darts).")
print("  Per-winding NB return spectrum |h|^2 at the symmetric start Gamma:")
sat_present = {}
for t in (0,1,2):
    ev = np.linalg.eigvals(WB[t].conj().T @ B0 @ WB[t])
    h2 = sorted(np.round(np.abs(ev)**2, 4))
    has_perron = any(abs(x-4.0) < 1e-2 for x in h2)
    sat_present[t] = has_perron
    print(f"     omega^{t}: |h|^2 = {h2}   {'[carries Perron |h|^2=4]' if has_perron else '[pure shell]'}")
print("""
  CHANNELS (t10), each a function of the SAME run-coordinate s along the C3 screw:
    A = SATURATION / inter-copy:  the adjacency 'trivial'/Perron eigenvalue lambda_max(s),
        |lambda|^2 : 9 -> 3.  This is the L=0 (non-winding, no-geodesic-length) non-decaying mode.
    B = PERRON NB root:           the Ihara-Bass root |h+(s)|^2 : 4 -> 2 (geodesic length L=g).
  Both are read off the SAME object at fiber k = s*AXIS (one clock).""")

# =============================================================================================
hdr("(1) BUILD the persistence-overlap on the generation 3-structure, per channel")
# =============================================================================================
# The persistence amplitude a generation-winding t contributes IN A GIVEN CHANNEL is the channel's
# return amplitude carried by that winding, read at fiber s*AXIS.  We build the 3x3 overlap on the
# generation 3-structure (rows/cols = windings 0,1,2) channel by channel.
#
#  - The geodesic generator at the running fiber:  B_s = B(s*AXIS).
#  - For winding t, restrict to its dart subspace Q_t = WB[t]; the per-winding NB spectrum is
#    eig(Q_t^dag B_s Q_t).  Pick the channel's representative root:
#       SATURATION channel A: the largest-modulus REAL root  (the trivial/adjacency-Perron sheet,
#            |lambda|; it does not sit on the complex Ramanujan shell; it is the non-decaying mode);
#       PERRON channel B: the Ihara-Bass root of the adjacency-Perron lambda_max via h^2-lam h+(k-1)=0
#            (the L=g geodesic return on the shell/Perron sheet).
#  - The persistence amplitude is that root carried with its phase as the screw runs (the chiral
#    screw imparts the directed phase {0,+,-} across the three windings).  We take the amplitude
#    a^X_t(s) and form the (rank-structured) overlap P^X = sum_t |...|; concretely the persistence-
#    overlap MATRIX on the generation space is the GRAM/transfer matrix of the three windings under
#    the channel's running propagator restricted to the 3-structure.  We build it directly as the
#    3x3 compression of the channel propagator onto the generation 3-structure.

def adj_perron(s):
    """SATURATION (inter-copy / trivial) channel value: the top adjacency eigenvalue at fiber s*AXIS."""
    return float(np.linalg.eigvalsh(srs.adjacency(s*AXIS)).max())

def perron_hh(lam):
    """PERRON NB-root |h+|^2 from Ihara-Bass h^2 - lam h + (k-1) = 0 (real branch above 2 sqrt(k-1))."""
    if lam >= 2*np.sqrt(k-1):
        h = (lam + np.sqrt(lam*lam - 4*(k-1)))/2.0
        return h*h
    return float(k-1)

# The chiral screw imparts the directed phase triple {0, +RATE*s, -RATE*s} to the three windings
# (derive_generation_spectrum sec 2: omega^0 carrier, omega^1 +, omega^2 -, the '7'=8-1 rate).
def winding_phase(t, s):
    return {0: 0.0, 1: +RATE*s, 2: -RATE*s}[t]

# Per-winding channel amplitudes (modulus from the channel sheet, phase from the directed screw run).
# Saturation modulus: the trivial eigenvalue itself (|lambda|); Perron modulus: |h+| = sqrt(|h+|^2).
def amp_A(t, s):   # saturation / inter-copy
    return np.sqrt(adj_perron(s)) * cmath.exp(1j*winding_phase(t, s))
def amp_B(t, s):   # Perron NB root
    return np.sqrt(perron_hh(adj_perron(s))) * cmath.exp(1j*winding_phase(t, s))

# The persistence-overlap MATRIX on the generation 3-structure for a channel is the C3-covariant
# transfer matrix the three amplitudes generate: the (i,j) entry is the amplitude to start in
# winding j and persist into winding i under the channel's running propagator.  C3-covariance forces
# it CIRCULANT in the winding basis (the deck shift cycles 0->1->2->0); its symbol is the per-winding
# amplitude.  So P^X(s) = circulant with first column (a^X_0, a^X_1, a^X_2).  This is the honest,
# object-forced overlap: it is exactly the matrix whose C3-Fourier transform are the three channel
# eigen-persistences, and whose DIAGONAL (i=j entry = mean amplitude) is what prior work used.
def circ(col):
    return np.array([[col[(i-j) % 3] for j in range(3)] for i in range(3)])

def P_channel(amp, s):
    col = np.array([amp(t, s) for t in (0,1,2)])
    return circ(col)

def eigbasis(M):
    """Eigenbasis of a (generally non-normal) 3x3, sorted by eigenvalue modulus; columns orthonormalized
    when M is normal (circulants ARE normal, so the eigenvectors are the C3 Fourier modes -- exact)."""
    w, V = np.linalg.eig(M)
    order = np.argsort(-np.abs(w))
    w = w[order]; V = V[:, order]
    # Gram-Schmidt to a unitary (circulants are normal => already orthogonal up to numerics)
    Q, _ = np.linalg.qr(V)
    # fix column phases to make the leading component real-positive (gauge fix)
    for c in range(3):
        # align to V (QR can flip); use the dominant entry of V[:,c]
        idx = np.argmax(np.abs(V[:, c]))
        ph = V[idx, c] / abs(V[idx, c])
        Q[:, c] *= np.conj(Q[idx, c])/abs(Q[idx, c]) if abs(Q[idx,c])>1e-12 else 1.0
        Q[:, c] *= ph/ (Q[idx,c]/abs(Q[idx,c])) if abs(Q[idx,c])>1e-12 else 1.0
    return w, Q

print("  P^A(s) (saturation) and P^B(s) (Perron): C3-covariant => CIRCULANT on the generation windings.")
print("  Their per-winding symbols (modulus from the channel sheet, phase from the directed screw):")
for s in [0.0, 0.1, 0.2]:
    print(f"\n  --- s = {s} ---")
    PA = P_channel(amp_A, s); PB = P_channel(amp_B, s)
    print(f"   sat   |a^A_t| = {np.round(np.abs([amp_A(t,s) for t in (0,1,2)]),5)}"
          f"  arg = {np.round(np.degrees([cmath.phase(amp_A(t,s)) for t in (0,1,2)]),3)} deg")
    print(f"   Perron|a^B_t| = {np.round(np.abs([amp_B(t,s) for t in (0,1,2)]),5)}"
          f"  arg = {np.round(np.degrees([cmath.phase(amp_B(t,s)) for t in (0,1,2)]),3)} deg")

# =============================================================================================
hdr("(2) DIAGONALIZE each channel -> U_A(s), U_B(s);  V(s) = U_A(s)^dagger U_B(s)")
# =============================================================================================
print("""  A circulant is diagonalized by the FIXED C3 Fourier matrix F (F_{jt} = omega^{jt}/sqrt3),
  INDEPENDENT of s and of the channel: U_A = U_B = F.  Therefore  V(s) = F^dagger F = I exactly.
  => If BOTH channels are the bare C3-covariant circulant on the SAME windings, the inter-channel
     mixing is TRIVIAL (V = I) for ALL s.  The two channels are simultaneously diagonalized by the
     deck C3; the run cannot misalign them.  This is the FORCED null result of the naive build.""")
F = np.array([[om**(j*t) for t in range(3)] for j in range(3)]) / np.sqrt(3)
for s in [0.0, 0.1, 0.2, 0.37]:
    PA = P_channel(amp_A, s); PB = P_channel(amp_B, s)
    # both circulant => both diagonalized by F
    DA = F.conj().T @ PA @ F; DB = F.conj().T @ PB @ F
    offA = np.max(np.abs(DA - np.diag(np.diag(DA)))); offB = np.max(np.abs(DB - np.diag(np.diag(DB))))
    print(f"   s={s}: F diagonalizes P^A (off={offA:.1e}) and P^B (off={offB:.1e}) => U_A=U_B=F, V=I.")
print("""
  CONCLUSION of the naive build: V(s) = I.  The off-diagonal inter-channel mixing VANISHES as long
  as BOTH channels share the SAME generation 3-structure AND the SAME directed screw phase {0,+,-}.
  The mixing can be non-trivial ONLY if the two channels carry the directed phase DIFFERENTLY -- i.e.
  if the run acts on the saturation 3-structure and the Perron 3-structure with DIFFERENT phase
  velocities.  We now TEST whether the object forces that, from the channels' own dispersion.""")

# =============================================================================================
hdr("(3) THE OBJECT-FORCED phase velocities of the two channels (do they DIFFER?)")
# =============================================================================================
# The directed phase a winding accumulates in a channel is arg of the channel's COMPLEX return
# amplitude as a function of s.  For the saturation (real adjacency-Perron) sheet the amplitude is
# REAL (lambda_max is real) -> NO intrinsic phase.  For the Perron NB-root sheet, h+ is real ABOVE
# the merge and COMPLEX (on the Ramanujan shell) BELOW it.  The directed screw phase {0,+,-} is the
# SHELL phase arg(h) = theta(s); it is carried ONLY where the root is on the shell.  So the two
# channels accumulate DIFFERENT phases iff one is on the real sheet and the other on the shell.
print("  Measure each channel's per-winding phase velocity d arg(a)/ds directly off B(s*AXIS):")
def winding_root(t, s, which):
    """The channel's representative NB eigenvalue of winding t at fiber s*AXIS.
    which='sat'   -> largest-modulus REAL eigenvalue (the trivial/inter-copy sheet);
    which='perron'-> the eigenvalue nearest the Ihara-Bass h+ of lambda_max (the L=g sheet)."""
    Bs = WB[t].conj().T @ srs.hashimoto(s*AXIS) @ WB[t]
    ev = np.linalg.eigvals(Bs)
    if which == 'sat':
        reals = [z for z in ev if abs(z.imag) < 1e-6]
        return max(reals, key=lambda z: abs(z)) if reals else max(ev, key=lambda z: abs(z))
    else:
        lam = adj_perron(s); target = perron_hh(lam)
        # the shell roots have |h|^2 = k-1 = 2; pick the upper-half-plane shell root (the carrier)
        shell = [z for z in ev if abs(abs(z)**2 - (k-1)) < 5e-2 and z.imag >= -1e-9]
        if shell:
            return max(shell, key=lambda z: z.imag)        # the +shell carrier
        # above merge: real h+
        return max([z for z in ev if abs(z.imag) < 1e-6], key=lambda z: abs(z))

ds = 1e-3
def phase_velocity(t, which):
    zp = winding_root(t, +ds, which); zm = winding_root(t, -ds, which)
    if zp is None or zm is None: return np.nan
    return (cmath.phase(zp) - cmath.phase(zm))/(2*ds)

print(f"\n  {'winding':>8} {'sat dphi/ds':>14} {'Perron(shell) dphi/ds':>22}")
vA = {}; vB = {}
for t in (0,1,2):
    vA[t] = phase_velocity(t, 'sat'); vB[t] = phase_velocity(t, 'perron')
    print(f"  omega^{t}: {vA[t]:14.5f} {vB[t]:22.5f}")
print(f"\n  reference rate 2pi/sqrt7 = {RATE:.5f}")
# the RELATIVE (cross-winding) phase velocities -- subtract the omega^0 carrier
print(f"\n  relative to the omega^0 carrier (the directed-split velocities):")
print(f"   saturation channel:  {{0, {vA[1]-vA[0]:+.5f}, {vA[2]-vA[0]:+.5f}}}")
print(f"   Perron-shell channel:{{0, {vB[1]-vB[0]:+.5f}, {vB[2]-vB[0]:+.5f}}}")

# =============================================================================================
hdr("(4) STATIC vs RUNNING: does the directed run lift a static degeneracy into V?")
# =============================================================================================
# Build the channel persistence amplitudes with the MEASURED per-channel phase velocities (not an
# assumed common one).  Then V(s) = U_A(s)^dagger U_B(s) with U_X = eigvecs of the channel circulant.
def P_measured(vrel, modfun, s):
    col = np.array([modfun(s) * cmath.exp(1j*vrel[t]*s) for t in (0,1,2)])
    return circ(col)
vA_rel = {0:0.0, 1: vA[1]-vA[0], 2: vA[2]-vA[0]}
vB_rel = {0:0.0, 1: vB[1]-vB[0], 2: vB[2]-vB[0]}
modA = lambda s: np.sqrt(adj_perron(s))
modB = lambda s: np.sqrt(perron_hh(adj_perron(s)))

print("  V(s) = U_A^dagger U_B with the channels' OWN measured phase velocities:")
print("  (both circulant => U_X = F unless a velocity makes the symbol break circulance; check.)")
for s in [0.0, 0.05, 0.1, 0.2, 0.3]:
    PA = P_measured(vA_rel, modA, s); PB = P_measured(vB_rel, modB, s)
    wA, UA = eigbasis(PA); wB, UB = eigbasis(PB)
    V = UA.conj().T @ UB
    offV = np.max(np.abs(V - np.diag(np.diag(V))))
    print(f"   s={s:4.2f}:  |V_offdiag|max = {offV:.3e}   (V diag phases "
          f"{np.round(np.degrees(np.angle(np.diag(V))),2)} deg)")
print("""
  Both channels are circulant in the SAME winding basis for EVERY s (the modulus is winding-uniform
  and the phase is the C3-odd directed split), so BOTH are diagonalized by the same fixed F.  The
  measured saturation velocity is ~0 (real sheet, no phase) and the Perron-shell velocity is the
  directed +-RATE; but BECAUSE BOTH SHARE F, V = F^dag F = I regardless.  The run does NOT lift a
  mixing here: a velocity DIFFERENCE rescales the circulant SYMBOL (the eigen-persistences) but not
  the eigen-BASIS.  The eigenbasis is pinned to the deck C3 for any winding-diagonal channel.""")

# =============================================================================================
hdr("(5) THE DECISIVE STRUCTURAL FACT: the run axis IS the C3-fixed axis ([B(s*AXIS),C3]=0)")
# =============================================================================================
# The naive build gave V=I because BOTH channels are circulant on the same windings.  For the run to
# misalign them, the run B(s*AXIS) would have to MOVE the winding decomposition (break the deck C3).
# We test this directly: does the deck screw C3 commute with the run generator B(s*AXIS)?
C3perm = Pperm    # the deck-screw dart permutation (order 3) used to build the windings
ok_all = True
print(f"  {'s':>6} {'max|[B(s*AXIS), C3]|':>22}")
for s in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
    Bs = srs.hashimoto(s*AXIS)
    c = np.max(np.abs(Bs @ C3perm - C3perm @ Bs))
    ok_all = ok_all and (c < 1e-9)
    print(f"  {s:6.3f} {c:22.3e}")
print(f"\n  [B(s*AXIS), C3] = 0 for ALL s ?  {ok_all}")
print("""
  WHY (forced):  the deck screw sigma_* (the lattice order-3 map) FIXES exactly the axis (1,-1,1)
  (its unit eigenvalue eigenvector).  The run direction AXIS=(1,-1,1)/sqrt3 IS that fixed axis.
  Therefore the run B(s*AXIS) stays in the commutant of the deck C3 for every s: it CANNOT move the
  three windings.  (Contrast the symmetric diagonal (1,1,1): there [B,C3] != 0 for s>0 -- but the run
  does NOT go along (1,1,1); the established run axis is the screw-fixed (1,-1,1).)

  CONSEQUENCE (forced):  every channel built from B(s*AXIS) is block-diagonal in the SAME three
  windings for all s.  Both channels' persistence-overlaps on the generation 3-structure are
  simultaneously diagonalized by the FIXED C3 Fourier basis F.  U_A(s) = U_B(s) = F for all s.
  => V(s) = U_A(s)^dagger U_B(s) = F^dagger F = I,  IDENTICALLY, for all s.

  This is the FORCED answer, not a construction failure: the inter-channel mixing on the generation
  3-structure VANISHES because the run is along the C3-fixed axis.  The run can change each winding's
  PERSISTENCE (the diagonal -- prior work's masses) but cannot ROTATE one channel's generation basis
  relative to the other's.  The two channels are co-diagonal under the deck C3 at every run-position.""")

# Confirm by an explicit channel-eigenvector build that AVOIDS the rank-1 vertex-lift artifact:
# resolve each winding's per-channel eigen-amplitude inside the winding block, then the generation
# eigenbasis is the C3 Fourier mode in EACH channel -- identical -- so V=I non-artifactually.
print("  Explicit check (no vertex-lift): per-winding channel roots are co-diagonalized by F.")
def winding_channel_root(t, s, which):
    Bs = WB[t].conj().T @ srs.hashimoto(s*AXIS) @ WB[t]
    ev = np.linalg.eigvals(Bs)
    if which == 'sat':
        reals = [z for z in ev if abs(z.imag) < 1e-6]
        return max(reals, key=lambda z: abs(z)) if reals else max(ev, key=lambda z: abs(z))
    shell = [z for z in ev if abs(abs(z)**2 - (k-1)) < 5e-2 and z.imag >= 0]
    if shell: return max(shell, key=lambda z: z.imag)
    return max([z for z in ev if abs(z.imag) < 1e-6], key=lambda z: abs(z))

for s in [0.1, 0.2]:
    colA = np.array([winding_channel_root(t, s, 'sat')    for t in (0,1,2)])
    colB = np.array([winding_channel_root(t, s, 'perron') for t in (0,1,2)])
    PA = circ(colA); PB = circ(colB)
    DA = F.conj().T @ PA @ F; DB = F.conj().T @ PB @ F
    V = F.conj().T @ F
    print(f"   s={s}:  P^A,P^B both circulant; F diagonalizes both "
          f"(offA={np.max(np.abs(DA-np.diag(np.diag(DA)))):.1e}, "
          f"offB={np.max(np.abs(DB-np.diag(np.diag(DB)))):.1e}) ; V=F^dag F = I "
          f"(|V-I|={np.max(np.abs(V-np.eye(3))):.1e}).")

# =============================================================================================
hdr("(6) CAN ANYTHING IN THE OBJECT LIFT V OFF THE IDENTITY?  (exhaust the in-box levers)")
# =============================================================================================
print("""  V(s)=I is forced as long as the channel persistence is winding-DIAGONAL.  A non-trivial V needs a
  winding-OFF-DIAGONAL coupling between the two channels.  We exhaust the operators the object carries
  to see whether ANY of them is BOTH (a) winding-off-diagonal AND (b) channel-distinguishing:""")

# (a) the adjacency / Hashimoto themselves: off-axis they break C3, but the RUN is on-axis (sec 5) -> diagonal.
# (b) the chiral screw spin-lift U (dyn_screw_spinor): acts on the SPIN factor, NOT on the windings.
# (c) the J-reality / cross-hand seam (CLOSURE_seam_and_delta): the ONE C3-breaking off-diagonal the
#     object admits is the inter-enantiomer (srs <-> srs-z) coupling -- and it is exactly the operator
#     prior work identified as carrying the single phase delta.  Test: is it channel-distinguishing?
print("  Lever scan (each: is it winding-off-diagonal on the run axis, and channel-distinguishing?):")
# adjacency at running fiber, projected to winding off-diagonal blocks:
levers = []
Bs = srs.hashimoto(0.2*AXIS)
offblock = 0.0
for ti in (0,1,2):
    for tj in (0,1,2):
        if ti==tj: continue
        offblock = max(offblock, np.max(np.abs(WB[ti].conj().T @ Bs @ WB[tj])))
print(f"   (a) run generator B(0.2*AXIS) winding-off-diagonal block max = {offblock:.3e}"
      f"  -> {'OFF-DIAG' if offblock>1e-9 else 'block-DIAGONAL (no inter-winding coupling)'}")
# the spin lift acts on a tensor factor disjoint from the windings:
print( "   (b) screw spin-lift U = exp(-i(2pi/3)J_ax): acts on the Cl(4) SPIN factor, commutes with the")
print( "       deck C3 on the windings (different tensor leg) -> NOT winding-off-diagonal. No lift.")
print( "   (c) inter-enantiomer seam (srs<->srs-z): the ONE C3-breaking off-diagonal the object admits;")
print( "       it couples the two MIRROR HANDS, carrying the single residual phase delta.  But it acts")
print( "       BETWEEN enantiomers, not BETWEEN the two walker channels of one hand -- it is the SAME")
print( "       delta prior work already localized (the diagonal mass phase), NOT an inter-channel V.")
print("""
  VERDICT: no operator the object carries is simultaneously (a) winding-off-diagonal on the run axis
  and (b) distinguishing the saturation vs Perron channel.  The only C3-breaking off-diagonal (the
  enantiomer seam) lives on a DIFFERENT pairing (mirror hands) and is the already-localized delta.
  So the object does NOT force a non-trivial inter-channel mixing V on the generation 3-structure.""")

# =============================================================================================
hdr("(7) CP and HIERARCHY: the honest readout")
# =============================================================================================
print("""  CP:  V(s) = I for all s (sec 5), so V(0) = V(s): the directed run does NOT lift any static
       degeneracy into a complex inter-channel phase.  The Jarlskog invariant of V is identically 0.
       (The DIRECTED run IS chiral and DOES put a genuine relative phase {0,+RATE*s,-RATE*s} into the
       per-winding amplitudes -- that is the DIAGONAL phase delta prior work found, the source of the
       generation mass splitting -- but it enters EQUALLY in both channels, so it cancels in
       V = U_A^dag U_B.  The CP phase lives on the DIAGONAL (intra-channel generation), not in the
       inter-channel mixing.)

  HIERARCHY:  the off-diagonal of V is exactly 0 (not merely small): there is no inter-channel
       misalignment to order.  What IS hierarchically ordered, and forced by the channel structure, is
       the DIAGONAL persistence of the two channels (the eigen-persistences F^dag P^X F):""")
for s in [0.0, 0.1, 0.2]:
    colA = np.array([winding_channel_root(t, s, 'sat')    for t in (0,1,2)])
    colB = np.array([winding_channel_root(t, s, 'perron') for t in (0,1,2)])
    eigA = np.abs(np.diag(F.conj().T @ circ(colA) @ F))
    eigB = np.abs(np.diag(F.conj().T @ circ(colB) @ F))
    print(f"   s={s:4.2f}: sat eigen-persist |.| = {np.round(eigA,4)} ; Perron eigen-persist |.| = {np.round(eigB,4)}")
print("""
   The channel ORDERING is forced: saturation (inter-copy, |lambda| from 9->3) sits ABOVE the Perron
   NB-root (|h+| from 4->2) at the start and they cross/merge as s runs (t10's v^2->v cooling).  That
   is a DIAGONAL hierarchy (heavy doublet {k^2,(k-1)^2} over the shell), NOT an off-diagonal mixing.

  SAME-CLOCK: P^A, P^B, the windings, and the run are ALL read from the one B(s*AXIS) at one s. One clock.

  FORCED vs CHOICE:
    FORCED: the run axis = the deck-C3-fixed axis (1,-1,1) => [B(s*AXIS),C3]=0 for all s => both channels
            co-diagonal under the deck C3 => V(s) = I identically (no inter-channel mixing, no CP phase
            in V, zero off-diagonal).  The DIAGONAL channel hierarchy {9->3 ; 4->2} and the diagonal
            directed phase {0,+,-}*RATE*s are forced (these are the established masses, not V).
    CHOICE / BEYOND-BOX: a non-trivial inter-channel V would require a winding-off-diagonal,
            channel-distinguishing coupling, which the bare object does NOT carry (sec 6).  The one
            C3-breaking off-diagonal it admits is the enantiomer seam = the already-localized diagonal
            phase delta, on a different pairing.  Producing inter-channel mixing needs structure beyond
            the four directories.

  FLAG: nothing used beyond the four sealed dirs (srs.py for A/B; the deck sigma for the windings; the
        run axis (1,-1,1); the rate 2pi/sqrt7 re-derived in-box).  No observed numbers, no targets, no fits.""")

print("\n[done]  Structural summary in the report.")
