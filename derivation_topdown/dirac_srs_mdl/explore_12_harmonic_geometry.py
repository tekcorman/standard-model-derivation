"""
explore_12 -- the KOTANI-SUNADA STANDARD REALIZATION (harmonic embedding) of srs.
Pure math, walled off (srs.py + numpy/stdlib only).

The standard realization is the harmonic (balanced) map of the maximal abelian cover into
its Albanese torus R^3/L Z^3.  It is THE canonical geometric incarnation of the net and is
the realization Kotani-Sunada single out by energy minimization (the "MDL" of the net).

(1) HARMONIC EQUILIBRIUM.
    Place the 4 cell-vertices at x_0..x_3 in R^3, lattice = column-span of a 3x3 period
    matrix L.  For an edge (i,j,vec) the head image is x_j + L*vec; the dart displacement is
        w(i->j,vec) = x_j + L*vec - x_i .
    HARMONIC (balanced) condition: at every vertex the outgoing darts sum to zero,
        sum_{darts at i} w = 0   for all i .
    ALBANESE (standard-realization) condition: the metric is fixed by requiring the building
    blocks to be ISOTROPIC,
        sum_{darts} w w^T  proportional to  I_3 .
    Solve for the fractional vertex positions and the period Gram matrix G = L^T L; report.

(2) GEOMETRY.  Edge vectors at a vertex, the BOND ANGLE between them (srs has equal angles),
    and a check that all vertices and all edges are equivalent (maximal symmetry).

(3) HELIX / CHIRALITY.  srs is built from Coxeter's girth-ten 10_3 helices.  Follow a chain
    of edges, report the screw axis / handedness, confirm the geometry is chiral.

(4) GYROID.  srs is the labyrinth graph of the gyroid minimal surface; report a linked quantity.
"""
import numpy as np, itertools, srs
np.set_printoptions(precision=5, suppress=True)

# Darts (outgoing half-edges) and the established sigma=(123) action on H_1 = Z^3 (explore_06).
DARTS = srs._darts()                                  # 12 darts (tail, head, vec)
sigma = {0: 0, 1: 2, 2: 3, 3: 1}                      # the 3-cycle (1 2 3); fixes vertex 0
Msig  = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]) # sigma_* on H_1 (e1->e3, e2->-e1, e3->-e2)

print("="*86)
print("(1)  HARMONIC EQUILIBRIUM  +  ALBANESE METRIC  (the standard realization)")
print("="*86)

# -------------------------------------------------------------------------------------------
# 1a. HARMONIC positions in FRACTIONAL (lattice) coordinates.
#     Write x_i = sum of lattice basis * (fractional coords y_i in R^3).  Working in fractional
#     coords, an edge displacement reads  w_frac(i->j,vec) = y_j + vec - y_i  (L drops out of
#     the *balance* equation since L is invertible: L w_frac = 0  <=>  w_frac = 0 summed).
#     Balance at vertex i:  sum_{darts at i} (y_head + vec - y_i) = 0.
#     This is the graph-Laplacian system  Laplacian * Y = -(net voltage at i),  i.e. the
#     harmonic-map equation.  Gauge: fix the centroid sum_i y_i = 0 (translation freedom).
# -------------------------------------------------------------------------------------------
def outgoing(i):
    """all darts leaving vertex i as (head, vec)."""
    return [(h, v) for (t, h, v) in DARTS if t == i]

# Build and solve the balance equations for the 4 fractional positions y_0..y_3 in R^3.
# Unknowns: 4*3 = 12.  Equations: balance at each vertex (3 comps each) + centroid gauge.
A = np.zeros((4*3 + 3, 4*3))
b = np.zeros(4*3 + 3)
def idx(i, c): return 3*i + c
row = 0
for i in range(4):
    darts_i = outgoing(i)
    for c in range(3):
        # sum_darts (y_head[c] + vec[c] - y_i[c]) = 0
        for (h, v) in darts_i:
            A[row, idx(h, c)] += 1.0
            A[row, idx(i, c)] -= 1.0
            b[row] -= v[c]
        row += 1
# centroid gauge: sum_i y_i = 0 (each component)
for c in range(3):
    for i in range(4):
        A[row, idx(i, c)] = 1.0
    b[row] = 0.0
    row += 1

Y, *_ = np.linalg.lstsq(A, b, rcond=None)
Yv = Y.reshape(4, 3)
print("\n[1a] Harmonic equilibrium positions in FRACTIONAL (Albanese-lattice) coordinates")
print("     (centroid fixed at 0; the period lattice basis is the metric of 1b):")
for i in range(4):
    print(f"     y_{i} = {Yv[i]}")

# verify balance exactly
resid = 0.0
for i in range(4):
    s = np.zeros(3)
    for (h, v) in outgoing(i):
        s += Yv[h] + np.array(v, float) - Yv[i]
    resid = max(resid, np.max(np.abs(s)))
print(f"     max |balance residual| (sum of outgoing dart vectors per vertex) = {resid:.2e}")
print(f"     => harmonic (balanced) condition satisfied: {resid < 1e-9}")

# -------------------------------------------------------------------------------------------
# 1b. ALBANESE METRIC.  The standard realization's metric makes the cover maximally symmetric.
#     The Albanese inner product on H_1 = Z^3 is  G_alb = (sum over a TRANSVERSAL of darts of
#     the harmonic 1-forms)  -- concretely the harmonic projection of the homology.  We compute
#     it from the ISOTROPY requirement: choose the period Gram matrix G = L^T L so that the dart
#     displacements w(e) = L(y_j + vec - y_i) satisfy  sum_e w w^T  ~ I_3  (isotropic blocks).
#     Let f(e) = y_head + vec - y_tail in R^3 (fractional dart vectors, from 1a).  Then
#         sum_e w w^T = L (sum_e f f^T) L^T  =!  c * I  =>  G_correlator := sum_e f f^T,
#     and L is chosen with  L G_correlator L^T = I, i.e.  L = G_correlator^{-1/2}  (up to O(3)).
#     The Albanese period Gram matrix is then  G_alb = L^T L = G_correlator^{-1}  (the canonical
#     dual/Albanese metric).  Report both the correlator and the resulting Albanese Gram matrix.
# -------------------------------------------------------------------------------------------
fracdarts = []
for (t, h, v) in DARTS:
    fracdarts.append(Yv[h] + np.array(v, float) - Yv[t])
fracdarts = np.array(fracdarts)                       # 12 x 3 (the 12 dart vectors, fractional)

Corr = np.zeros((3, 3))
for f in fracdarts:
    Corr += np.outer(f, f)
print("\n[1b] Dart-correlator  C = sum_darts f f^T  (fractional):")
print(Corr)
# isotropy of the correlator itself (in fractional coords): is it already proportional to I?
iso_frac = np.allclose(Corr, np.trace(Corr)/3*np.eye(3))
print(f"     correlator proportional to I already (fractional coords isotropic): {iso_frac}")

# The Albanese realization rescales space by L = C^{-1/2} so that L C L^T = I.
evals, evecs = np.linalg.eigh(Corr)
L = evecs @ np.diag(1.0/np.sqrt(evals)) @ evecs.T     # symmetric C^{-1/2}
G_alb = L.T @ L                                       # = C^{-1}
print("\n     Albanese period transform  L = C^{-1/2}  (so that  L C L^T = I_3):")
print(L)
print("     check  L C L^T = I :", np.allclose(L @ Corr @ L.T, np.eye(3)))
print("\n     ALBANESE period Gram matrix  G_alb = L^T L = C^{-1}  (metric of the period vectors):")
print(G_alb)
# The defining isotropy is on the BOND blocks (sum w w^T ~ I), NOT on the period Gram: the
# period basis e1,e2,e3 is a PRIMITIVE (non-orthogonal) basis of the Albanese lattice, so G_alb
# need not be diagonal.  Identify the lattice properly below (1b').
diag_equal = np.allclose(np.diag(G_alb), G_alb[0, 0])
off_equal  = np.allclose(np.abs(G_alb[np.triu_indices(3, 1)]), abs(G_alb[0, 1]))
print(f"     all period vectors equal length (|a_m|^2 = {G_alb[0,0]} for all m): {diag_equal}")
print(f"     all primitive-pair angles equal in magnitude (|G_ij| const off-diag): {off_equal}")

# 1b'. IDENTIFY the Albanese lattice.  The maximally-symmetric srs realization is known to have
#      a BODY-CENTRED CUBIC (bcc) Albanese lattice.  Test: do the period vectors (cols of L)
#      generate a bcc lattice?  A primitive bcc basis has each |a|^2 equal and the conventional
#      cube edge recovered as combinations.  Check by forming the dual/conventional cell.
print("\n[1b'] Identify the Albanese lattice (period vectors are a PRIMITIVE basis):")
acart = L.T                                            # rows = cartesian period vectors a_1,a_2,a_3
# A standard primitive bcc basis: a_i = (s/2)(-e_i + e_j + e_k) variants, |a_i|^2 = 3 s^2/4,
# pairwise dot = -s^2/4.  Our G_alb has diag 1.5, off |0.5| -> ratio diag/|off| = 3, matching bcc
# (3s^2/4) / (s^2/4) = 3.  Report the ratio (lattice fingerprint).
ratio = G_alb[0, 0]/abs(G_alb[0, 1])
print(f"     Gram diag/|off-diag| ratio = {ratio:.4f}   (bcc primitive basis fingerprint = 3.0)")
print(f"     => Albanese lattice is BODY-CENTRED CUBIC (bcc): {np.isclose(ratio, 3.0)}")
# conventional cubic cell volume vs primitive: det of primitive basis
volp = abs(np.linalg.det(acart))
print(f"     primitive cell volume |det L| = {volp:.5f}   (bcc: 1/2 of the conventional cube)")

# CARTESIAN harmonic coordinates of the 4 vertices in the Albanese realization: X_i = L y_i.
Xv = (L @ Yv.T).T
print("\n[1c] CARTESIAN harmonic vertex coordinates in the Albanese realization  X_i = L y_i:")
for i in range(4):
    print(f"     X_{i} = {Xv[i]}")
# Cartesian period vectors = columns of L (images of e1,e2,e3):
print("     Albanese period vectors (cartesian, = columns of L):")
for m in range(3):
    print(f"       a_{m+1} = {L[:, m]}   |a_{m+1}| = {np.linalg.norm(L[:, m]):.5f}")

# =====================================================================================
print("\n" + "="*86)
print("(2)  GEOMETRY:  bond angle, edge/vertex equivalence (maximal symmetry)")
print("="*86)

# Cartesian dart vectors  w(e) = L f(e).
wdarts = (L @ fracdarts.T).T                           # 12 x 3
lens = np.linalg.norm(wdarts, axis=1)
print("\n[2a] Cartesian dart (bond) vectors and lengths:")
for d, (t, h, v) in enumerate(DARTS):
    vs = str(tuple(int(x) for x in v))
    print(f"     {t}->{h:1} vec={vs:>10}:  w = {wdarts[d]}   |w| = {lens[d]:.5f}")
print(f"\n     all bond lengths equal: {np.allclose(lens, lens[0])}   (common length = {lens[0]:.5f})")

# Bond angles at vertex 0 (the three outgoing darts).
print("\n[2b] BOND ANGLES at vertex 0 (between its three outgoing bonds):")
out0_idx = [d for d, (t, h, v) in enumerate(DARTS) if t == 0]
vecs0 = wdarts[out0_idx]
angs = []
for a, c in itertools.combinations(range(3), 2):
    u1, u2 = vecs0[a], vecs0[c]
    ca = (u1 @ u2)/(np.linalg.norm(u1)*np.linalg.norm(u2))
    ang = np.degrees(np.arccos(np.clip(ca, -1, 1)))
    angs.append(ang)
    print(f"     angle(bond {a}, bond {c}) = {ang:.4f} deg     cos = {ca:+.5f}")
print(f"\n     all three bond angles equal: {np.allclose(angs, angs[0])}")
print(f"     bond angle = {angs[0]:.4f} deg   (cos = {np.cos(np.radians(angs[0])):+.5f})")
# identify the exact cosine
print(f"     exact?  cos = -1/2 (120 deg): {np.isclose(np.cos(np.radians(angs[0])), -0.5)}")

# All vertices equivalent: each vertex's 3 outgoing darts have the SAME multiset of pairwise
# angles and the SAME length -> local geometry identical (combined with the A4 vertex-transitivity
# established in explore_10, this is maximal symmetry / strong isotropy).
print("\n[2c] VERTEX / EDGE EQUIVALENCE (maximal symmetry / strong isotropy):")
ok_vertices = True
ref = None
for i in range(4):
    oi = [d for d, (t, h, v) in enumerate(DARTS) if t == i]
    vv = wdarts[oi]
    la = sorted(np.linalg.norm(vv, axis=1))
    aa = sorted(np.degrees(np.arccos(np.clip(
        (vv[a] @ vv[c])/(np.linalg.norm(vv[a])*np.linalg.norm(vv[c])), -1, 1)))
        for a, c in itertools.combinations(range(3), 2))
    sig = (np.round(la, 6).tolist(), np.round(aa, 4).tolist())
    if ref is None: ref = sig
    elif sig != ref: ok_vertices = False
    print(f"     vertex {i}: bond lengths {np.round(la,5)}  angles {np.round(aa,3)}")
print(f"     all four vertices geometrically identical (same lengths & angles): {ok_vertices}")
print("     (with A4 vertex- and edge-transitivity from explore_10 => the net is strongly")
print("      isotropic: ONE vertex orbit, ONE edge orbit, equal bond length, equal bond angle.)")

# =====================================================================================
print("\n" + "="*86)
print("(3)  HELIX / SCREW STRUCTURE  +  CHIRALITY")
print("="*86)

# -------------------------------------------------------------------------------------------
# 3a. The C3 screw AXIS.  srs is a packing of 10_3 helices winding about the <111> body-diagonal
#     axes (these are the only proper rotation axes of the net: Aut+(K4)=A4, the tetrahedral
#     rotation group, has 3-fold axes along <111> and 2-fold axes only).  sigma=(123) acts on
#     H_1 by Msig and permutes vertices by sigma{}; in the Albanese frame Msig is the cartesian
#     rotation  R = L Msig L^{-1}.
# -------------------------------------------------------------------------------------------
Linv = np.linalg.inv(L)
R = L @ Msig @ Linv
print("\n[3a] Cartesian realization of sigma=(123) (a C3 net axis):  R = L Msig L^{-1}")
print(R)
print(f"     R orthogonal (a rotation): {np.allclose(R @ R.T, np.eye(3))}")
print(f"     det R = {np.linalg.det(R):+.4f}   (+1 => proper rotation)")
print(f"     R^3 = I: {np.allclose(np.linalg.matrix_power(R, 3), np.eye(3))}   (order 3)")
w_, V_ = np.linalg.eig(R)
axis = V_[:, np.argmin(np.abs(w_ - 1))].real
axis = axis/np.linalg.norm(axis)
rot_angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1)/2, -1, 1)))
print(f"     C3 axis direction (cartesian) = {axis}   (a <111> body diagonal)")
print(f"     rotation angle about the axis = {rot_angle:.3f} deg  (= 120 deg)")

# cover vertex position (cartesian): X(s,cell) = X_s + L @ cell.
def pos(s, cell):
    return Xv[s] + L @ np.array(cell, float)

# -------------------------------------------------------------------------------------------
# 3b. The genuine SCREW (helix generator).  A net automorphism g:(s,c)->(sigma[s], Msig c + t),
#     for ANY integer translation t, is realised cartesianly by the affine screw  P -> R P + L t.
#     Its intrinsic pitch (translation along the rotation axis, independent of the axis-point)
#     is  pitch(t) = axis . (L t).  We pick the minimal-pitch t = (1,0,0); the ORBIT of an
#     off-axis vertex under g is then a true HELIX.  (A single edge gives ZERO axial climb here
#     -- the bonds lie in planes nearly transverse to <111> -- so the helix must be exhibited as
#     this screw orbit, NOT as one bond stepped by R.  This is the honest construction.)
# -------------------------------------------------------------------------------------------
print("\n[3b] The screw / helix generator g:(s,c)->(sigma[s], Msig c + t),  t=(1,0,0):")
bond = lens[0]
tcell = np.array([1, 0, 0])
# orient the axis along the direction of POSITIVE climb so 'handedness' is well-defined.
pitch_signed = axis @ (L @ tcell.astype(float))
if pitch_signed < 0:
    axis = -axis; pitch_signed = -pitch_signed
pitch = pitch_signed
print(f"     intrinsic pitch  = axis . (L t)        = {pitch:+.5f}   (axial climb per 120 deg step)")
print(f"     full period along axis (3 steps)       = {3*pitch:.5f}  (= sqrt(3/2) = {np.sqrt(1.5):.5f},")
print(f"        the <111> lattice period of the bcc Albanese lattice): {np.isclose(3*pitch, np.sqrt(1.5))}")

# Trace the helix = screw orbit of an OFF-AXIS vertex (vertex 1).
s, cell = 1, np.array([0, 0, 0]); orbit = [(s, tuple(cell))]
for _ in range(9):
    cell = Msig @ cell + tcell; s = sigma[s]; orbit.append((s, tuple(cell)))
pts = np.array([pos(o[0], o[1]) for o in orbit])
axcoord = pts @ axis
print("\n     helix (screw orbit of vertex 1) -- monotone climb along the axis:")
for o, P, z in zip(orbit, pts, axcoord):
    print(f"       {o[0]} cell={o[1]}  cart={np.round(P,4)}  axial={z:+.4f}")
rises = np.diff(axcoord)
print(f"     axial climb constant per step: {np.allclose(rises, pitch)}   (= {pitch:+.5f})")
# transverse return after 3 steps (=> genuine 3-fold screw, a 3_1 or 3_2 helix):
perp = pts - np.outer(axcoord, axis)
print(f"     transverse position returns after 3 steps (3-fold screw): {np.allclose(perp[3], perp[0], atol=1e-9)}")

# HANDEDNESS: orient axis along +climb (done above); the rotation R is right-handed (3_1) iff
# it is COUNTER-clockwise about +axis (right-hand rule), i.e. sign of (u x Ru).axis = +1.
u = wdarts[out0_idx[0]].copy(); u = u - (u @ axis)*axis; u = u/np.linalg.norm(u)
sense = np.sign(np.dot(np.cross(u, R @ u), axis))
handed = "RIGHT-handed" if sense > 0 else "LEFT-handed"
print(f"\n     rotation sense about the +climb axis (sign of (u x Ru).axis) = {sense:+.0f}")
print(f"     => with the climb oriented +axis, the +120 deg rotation turns {('CCW' if sense>0 else 'CW')};")
print(f"        this traced helix is {handed} (a 3-fold/3_1-type screw orbit).")
print("     HONEST SCOPE: the handedness LABEL of one screw element is convention-dependent")
print("     (climb-sense x rotation-sense), and the cover's Z^3 translations let you reach a")
print("     screw of either reduced pitch about a given <111> axis; so a single screw's hand is")
print("     NOT by itself the chirality invariant.  The rigorous chirality statement is 3c.")
print("     Coxeter 10_3 (girth ten): the shortest CLOSED circuit has 10 bonds and winds these")
print("     <111> 3-fold helices; girth 10 was verified in explore_10.")

# -------------------------------------------------------------------------------------------
# 3c. CHIRALITY -- the RIGOROUS invariant: NO orientation-reversing isometry exists.  This is
#     proven combinatorially in explore_10 (no IMPROPER graph automorphism -- odd permutation x
#     signed lattice map -- realises the homology inversion k->-k; scanned over all 24 perms x
#     48 signed maps).  We re-exhibit it geometrically here: the net's symmetries are R-rotations
#     + lattice translations (orientation-PRESERVING, det=+1); applying the inversion -I to the
#     realization produces the enantiomorph srs* (the opposite-hand net), which is NOT
#     superposable on srs by any proper motion.  det(-I)=-1 is realised by no symmetry => chiral.
# -------------------------------------------------------------------------------------------
print("\n[3c] CHIRALITY (the rigorous invariant -- no improper symmetry):")
# all net point-symmetries in this frame have det = +1 (proper); show inversion is not among them.
dets = []
for Mrot in [np.eye(3, dtype=int), Msig, Msig @ Msig]:
    dets.append(int(round(np.linalg.det(L @ Mrot @ Linv))))
print(f"     det of the realized rotations {{I, sigma, sigma^2}} = {dets}  (all +1, proper)")
print(f"     inversion -I has det = {int(round(np.linalg.det(-np.eye(3))))}  (-1, improper) and is NOT a symmetry")
print(f"        of srs (no improper graph automorphism exists -- explore_10, exhaustive scan).")
print("     => the standard realization is CHIRAL: srs and its mirror srs* are distinct, non-")
print("        superposable nets (maximal space group I4_1 32; single-net P4_1 32 / P4_3 32 are")
print("        the enantiomorphic pair).  The inversion maps the traced 3-fold helix to its")
print("        opposite-hand partner, exhibiting the two enantiomorphs.")

# =====================================================================================
print("\n" + "="*86)
print("(4)  GYROID  CONNECTION  (noted, not constructed)")
print("="*86)
print("""
   srs is the LABYRINTH (skeletal) graph of the GYROID, Schoen's triply-periodic minimal
   surface (TPMS).  The gyroid divides space into two interpenetrating congruent labyrinths;
   the graph threading EACH labyrinth is an srs net, and the two are enantiomorphs (one srs,
   one srs*) -- the same chirality this script just measured.  Linkable geometric quantities:
     * The two srs nets sit on the two sides of the gyroid; their mutual chirality = the
       chirality of the gyroid itself (the gyroid is a chiral TPMS, space group I4_1 32).
     * Bond angle / coordination: srs is 3-coordinate with the equal {bond angle} above; the
       gyroid channels follow the same C3 (3_1/3_2) screw axes found in (3).
     * The cubic Albanese lattice of (1b) is the natural cubic cell in which the gyroid is
       usually written (level set  sin x cos y + sin y cos z + sin z cos x = 0).""")
g_const = np.cos(np.radians(angs[0]))
print(f"   Reported linkable quantity: srs bond angle cos = {g_const:+.4f} (= -1/2, i.e. 120 deg),")
print(f"   the tetrahedral-complement angle carried by the gyroid's 3-coordinate channel network.")

print("\n" + "="*86)
print("DONE.")
print("="*86)
