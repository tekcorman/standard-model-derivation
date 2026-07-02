"""
srs = the maximal abelian (Z^3) cover of the complete graph K_4 (Sunada's K_4 crystal).
WALLED-OFF clean room: no imports from outside this directory. Pure math.

K_4: 4 vertices, 6 edges, 3-regular.  b_1(K_4) = 6 - 4 + 1 = 3  =>  deck group Z^3.
Choose spanning tree {01,02,03} (vectors 0); the 3 cotree edges {12,13,23} carry the
Z^3 basis vectors e_1,e_2,e_3.  Bloch/Floquet variable k in [0,1)^3 (fractional).
"""
import numpy as np

NV = 4
DEG = 3
# (tail, head, homology vector in Z^3)
EDGES = [(0, 1, (0, 0, 0)), (0, 2, (0, 0, 0)), (0, 3, (0, 0, 0)),
         (1, 2, (1, 0, 0)), (1, 3, (0, 1, 0)), (2, 3, (0, 0, 1))]

def adjacency(k):
    """Bloch adjacency A(k): NV x NV Hermitian."""
    k = np.asarray(k, float); A = np.zeros((NV, NV), complex)
    for i, j, v in EDGES:
        p = np.exp(2j*np.pi*(k @ np.array(v))); A[i, j] += p; A[j, i] += np.conj(p)
    return A

def incidence(k):
    """Oriented incidence d(k): NV x |E|, Bloch-phased.  d(e_{ij}) = head - tail."""
    k = np.asarray(k, float); d = np.zeros((NV, len(EDGES)), complex)
    for e, (i, j, v) in enumerate(EDGES):
        p = np.exp(2j*np.pi*(k @ np.array(v))); d[i, e] = -1.0; d[j, e] = p
    return d

def hodge_dirac(k):
    """D = [[0, d],[d*, 0]] on C0 (+) C1 ;  D^2 |_{C0} = graph Laplacian 3I - A."""
    d = incidence(k); n0, n1 = NV, len(EDGES)
    return np.block([[np.zeros((n0, n0)), d], [d.conj().T, np.zeros((n1, n1))]])

def _darts():
    D = []
    for i, j, v in EDGES:
        D += [(i, j, np.array(v)), (j, i, -np.array(v))]
    return D

def hashimoto(k):
    """Non-backtracking (Hashimoto) operator B(k) on directed edges: 2|E| x 2|E|."""
    k = np.asarray(k, float); D = _darts(); n = len(D); B = np.zeros((n, n), complex)
    for b, (tb, hb, vb) in enumerate(D):
        for a, (ta, ha, va) in enumerate(D):
            if ha == tb and not (hb == ta and np.array_equal(vb, -va)):
                B[b, a] = np.exp(2j*np.pi*(k @ vb))
    return B

def ihara_zeta_inv(u, k):
    """Bass determinant: zeta(u)^{-1} = (1-u^2)^{|E|-|V|} det(I - uA + (DEG-1)u^2 I)."""
    A = adjacency(k); I = np.eye(NV)
    return (1 - u**2)**(len(EDGES) - NV) * np.linalg.det(I - u*A + (DEG-1)*u**2*I)
