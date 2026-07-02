"""
Charge algebra of the conserved quantities on ONE common matter space.  Pure math; no physics.

COMMON MATTER SPACE (native terms):
  H = (chiral 4-spinor  C^4 = C^2_+ (+) C^2_-)  (x)  (internal C3-weight module  C^3, weights {-1,0,+1}).
  dim H = 12.

  Spinor index = (chi, hel),  chi in {+,-} = chirality grading eigenvalue (m06's gamma_c),
                              hel in {+,-} = the two states inside a chiral C^2 (the Weyl/helicity index).
  The srs (+) srs-z doubling (m06 Part 3) IS the chirality doubling: the two C^2 halves are the two
  enantiomeric nets, opposite Weyl handedness; the 4th gamma couples them.
  Internal index = C3 weight in {-1,0,+1} (the C3 character weight on the A4 module; m05 / explore_06).

OPERATORS (all Hermitian, all built explicitly; we then ASK what the math gives, adopting nothing):
  g   = chirality grading            (spectrum {+1,-1})
  h   = helicity                     (spectrum {+1,-1})           [the Weyl index in each chiral half]
  V   = internal C3 weight           (spectrum {-1,0,+1}), SAME on both chiral halves  (vector-like)
  W   = (C3 turn) + (handedness x helicity)   -- reconstructed per the prompt's recipe; coefficients
        determined by demanding its stated spectrum {-3,-1,+1,+3} (sixth-units), reported honestly
  T   = diagonal (Cartan) generator of the chiral SU(2) of the doubling, acting on ONE chirality half.

We then: (1) print the joint charge table; (2) compute the linear span (with I); (3) write every exact
linear relation and judge FORCED vs free-normalization.
"""
import numpy as np
import itertools

np.set_printoptions(precision=4, suppress=True, linewidth=160)

# --- 2x2 pieces ---
I2 = np.eye(2)
sz = np.diag([1.0, -1.0])                 # SU(2)/Cartan diag(+1,-1)
Pp = np.diag([1.0, 0.0])                  # projector onto the "+" state of a C^2

# Spinor C^4 = C^2(chi) (x) C^2(hel).  Order: (++),(+-),(-+),(--) with (chi,hel).
g4   = np.kron(sz, I2)                     # chirality      diag(+,+,-,-)
h4   = np.kron(I2, sz)                     # helicity       diag(+,-,+,-)
# handedness of the screw = which enantiomer = the chirality block sign (srs=+, srs-z=-):
hand4 = np.kron(sz, I2)                    # == g4  (handedness IS the chirality/copy sign here)

# internal C^3, C3 weight diag:
I3   = np.eye(3)
Vw   = np.diag([-1.0, 0.0, 1.0])          # the C3 character weight, spectrum {-1,0,+1}

I4 = np.eye(4)

# --- lift everything to H = C^4 (x) C^3 (dim 12) ---
G = np.kron(g4, I3)                        # chirality
H = np.kron(h4, I3)                        # helicity
V = np.kron(I4, Vw)                        # internal C3 weight (same on both chiral halves) -> vector-like

# T: Cartan of the chiral SU(2) of the doubling, acting on ONE chirality half.
# The SU(2) of the doubling rotates the 2-dim helicity (Weyl) index WITHIN a single chiral block.
# "Acts on one chirality half" => support on the chi=+ block only; Cartan = (1/2) sz on helicity.
T4 = np.kron(Pp, 0.5 * sz)                 # on chi=+ block: (1/2)diag(+1,-1) in helicity; zero on chi=-
T  = np.kron(T4, I3)

# --- W = (C3 turn) + (handedness x helicity).  Reconstruct, do not assert coefficients. ---
# Pieces (each in its own natural unit):
turn4 = I4                                 # placeholder; the C3 turn lives in the internal factor:
TURN  = np.kron(I4, Vw)                    # the internal C3 turn  (== V as an operator)
HXH   = np.kron(hand4 @ h4, I3)            # handedness x helicity, eigenvalues +-1

print("=" * 100)
print(f" COMMON MATTER SPACE  H = C^4 (x) C^3,  dim = {G.shape[0]}")
print("=" * 100)

# ---------------------------------------------------------------------------
# DETERMINE W's coefficients by DEMANDING the stated spectrum {-3,-1,+1,+3} (sixth-units).
# W = a*TURN + b*HXH.  Search small integer/half-integer a,b; report which (if any) give the spectrum,
# and whether the choice is forced (unique simple) or a free normalization.
# ---------------------------------------------------------------------------
target = sorted([-3, -1, 1, 3])
print("\n--- reconstructing W = a*(C3 turn) + b*(handedness x helicity); demanding spec = {-3,-1,+1,+3} ---")
hits = []
for a in [1, 2, 3]:
    for b in [1, 2, 3]:
        Wtry = a * TURN + b * HXH
        spec = sorted(set(np.round(np.linalg.eigvalsh(Wtry), 6).tolist()))
        if spec == target:
            hits.append((a, b))
        print(f"   a={a}, b={b}:  distinct spectrum = {spec}")
print("   integer (a,b) reproducing exactly {-3,-1,+1,+3}:", hits)

# Adopt the simplest hit if unique; else report ambiguity.
if hits:
    a, b = sorted(hits)[0]
else:
    a, b = 3, 1
W = a * TURN + b * HXH
print(f"\n   => W = {a}*(C3 turn) + {b}*(handedness x helicity).")
print(f"      spec(W) = {sorted(set(np.round(np.linalg.eigvalsh(W),6).tolist()))}")

OPS = {"g (chirality)": G, "h (helicity)": H, "V": V, "W": W, "T": T}

# ---------------------------------------------------------------------------
# (0) Check mutual commutativity (they must be simultaneously diagonal to be "charges").
# ---------------------------------------------------------------------------
print("\n--- mutual commutators (should all vanish for simultaneous charges) ---")
names = list(OPS)
allcomm = True
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        A, Bm = OPS[names[i]], OPS[names[j]]
        c = np.max(np.abs(A @ Bm - Bm @ A))
        if c > 1e-9:
            allcomm = False
            print(f"   [{names[i]}, {names[j]}] != 0  (max |.| = {c:.3g})")
print("   all five commute ?", allcomm)

# ---------------------------------------------------------------------------
# (1) JOINT CHARGE TABLE.  All ops diagonal in the product basis -> read diagonals.
# ---------------------------------------------------------------------------
labels = []
for ci, cs in [(0, "+"), (1, "-")]:          # chirality
    for hi, hs in [(0, "+"), (1, "-")]:      # helicity
        for wi, ws in [(0, "-1"), (1, "0"), (2, "+1")]:  # C3 weight
            idx = (ci * 2 + hi) * 3 + wi
            labels.append((idx, cs, hs, ws))

print("\n" + "=" * 100)
print(" (1)  JOINT CHARGE TABLE  (one row per basis state of H)")
print("=" * 100)
hdr = f"{'state (chi,hel,wt)':>20s} | " + " | ".join(f"{n:>14s}" for n in OPS)
print(hdr); print("-" * len(hdr))
for idx, cs, hs, ws in labels:
    vals = [OPS[n][idx, idx].real for n in OPS]
    print(f"  chi={cs} hel={hs} wt={ws:>2s}     | " + " | ".join(f"{v:>14.3f}" for v in vals))

# ---------------------------------------------------------------------------
# (2) LINEAR SPAN of {I, g, h, V, W, T}.  Dimension and dependencies.
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print(" (2)  LINEAR SPAN of {I, g, h, V, W, T}  (as 12x12 Hermitian operators)")
print("=" * 100)
basis_ops = {"I": np.eye(12), **OPS}
# flatten the diagonals (all diagonal) into vectors and rank them.
vecs = {n: np.real(np.diag(M)) for n, M in basis_ops.items()}
order = list(basis_ops)
Mstack = np.array([vecs[n] for n in order])      # 6 x 12
rank_all = np.linalg.matrix_rank(Mstack, tol=1e-9)
print(f"   operators: {order}")
print(f"   rank of the 6 diagonals  = dim span = {rank_all}")

# Which subsets are independent: incremental rank.
print("\n   incremental independence (add one at a time):")
chosen = []
cur = np.zeros((0, 12))
for n in order:
    test = np.vstack([cur, vecs[n]])
    r0 = np.linalg.matrix_rank(cur, tol=1e-9) if cur.shape[0] else 0
    r1 = np.linalg.matrix_rank(test, tol=1e-9)
    indep = r1 > r0
    print(f"     + {n:>14s} : {'INDEPENDENT' if indep else 'DEPENDENT on the previous'}")
    if indep:
        cur = test
        chosen.append(n)
print(f"   a maximal independent set (greedy): {chosen}")

# ---------------------------------------------------------------------------
# (3) EXACT linear relations among {I,g,h,V,W,T}.  Null space of the 6x12 stack.
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print(" (3)  EXACT LINEAR RELATIONS  (null space of the diagonal stack)")
print("=" * 100)
# null space of Mstack^T-relations: we want coefficient vectors c with sum_n c_n * vec_n = 0.
# i.e. left null space of Mstack (6 x 12): vectors c (length 6) with c @ Mstack = 0.
U, S, Vt = np.linalg.svd(Mstack)
tol = 1e-9
null_mask = S < tol
# left null space = rows of U corresponding to zero singular values:
nullc = U[:, np.where(np.append(S, [0]*(6-len(S))) < tol)[0]] if False else None
# Proper: left null space vectors are the left-singular vectors with zero singular value.
# Mstack is 6x12, rank r. Left null space dim = 6 - r.
rank = np.linalg.matrix_rank(Mstack, tol=tol)
print(f"   stack is 6x12, rank {rank}  ->  {6-rank} independent linear relation(s) among the 6 operators.\n")

# Find the left null space explicitly via SVD of Mstack (6x12): U is 6x6.
U6, S6, _ = np.linalg.svd(Mstack, full_matrices=True)
# singular values padded to length 6
sv = np.zeros(6); sv[:len(S6)] = S6
relvecs = U6[:, sv < tol].T                      # each row = coeffs over `order`
def fmt_rel(c):
    terms = []
    for coef, n in zip(c, order):
        if abs(coef) > 1e-6:
            terms.append(f"{coef:+.4f}*{n}")
    return "  ".join(terms)
for r, c in enumerate(relvecs):
    # normalize so the largest-magnitude coeff is +-1 (to expose simple ratios)
    cc = c / c[np.argmax(np.abs(c))]
    print(f"   relation {r+1}:   0 = " + fmt_rel(cc))
    # try to rationalize
    from fractions import Fraction
    rats = [Fraction(x).limit_denominator(12) for x in cc]
    print(f"               rationalized coeffs over {order}:")
    print("               " + ", ".join(f"{n}:{rt}" for n, rt in zip(order, rats) if rt != 0))

print("\n[done]")
