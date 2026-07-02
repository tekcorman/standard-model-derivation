"""
explore_17 — run the dynamics and examine the SOLUTION STRUCTURE. Pure math.

The flow is linear, so every solution is  psi(t) = sum_{n,k} c_{n,k} e^{-i E_n(k) t} |n,k>.
The solution structure IS the dispersion E_n(k) and its eigenmodes. The striking part: near the
band-touchings the low-energy solutions are governed by an EMERGENT first-order (Weyl-type) generator
k·L — a linear cone — with a pseudospin set by the node's degeneracy.
"""
import numpy as np, srs

def spec(k): return np.sort(np.linalg.eigvalsh(srs.adjacency(k)))

print("PART A — the solution spectrum: dispersion E_n(k)")
for nm, k in [("Gamma", (0, 0, 0)), ("P (1/4,1/4,1/4)", (.25, .25, .25)),
              ("(1/4,3/4,1/4)", (.25, .75, .25)), ("H", (.5, .5, .5))]:
    print(f"  {nm:16}: {np.round(spec(k), 4)}")
grid = [spec((a/10, b/10, c/10)) for a in range(10) for b in range(10) for c in range(10)]
g = np.array(grid)
print(f"  4 bands, spectrum in [{g.min():.3f}, {g.max():.3f}]; touchings at the nodes below.")

def grad(k0, eps=1e-5):
    e = np.eye(3)
    return [(srs.adjacency(np.add(k0, eps*e[a])) - srs.adjacency(np.subtract(k0, eps*e[a])))/(2*eps)
            for a in range(3)]

def effective(k0, tol=1e-4):
    w, V = np.linalg.eigh(srs.adjacency(k0))
    vals, cnts = np.unique(np.round(w, 4), return_counts=True)
    E = vals[np.argmax(cnts)]; m = cnts.max()
    P = V[:, np.abs(w - E) < tol]
    return E, m, [P.conj().T @ ga @ P for ga in grad(k0)]

print("\nPART B — emergent relativistic (Weyl-type) structure at the band touchings:")
for nm, k0 in [("Gamma", (0, 0, 0)), ("H", (.5, .5, .5)), ("(1/4,3/4,1/4)", (.25, .75, .25))]:
    E, m, Ha = effective(k0)
    print(f"\n  node {nm:14} at E = {E:+.3f}, degeneracy {m}:")
    for kd in [(1, 0, 0), (0, 1, 0), (1, 1, 1)]:
        u = np.array(kd, float); u /= np.linalg.norm(u)
        eig = np.sort(np.linalg.eigvalsh(sum(u[a]*Ha[a] for a in range(3))).real)
        print(f"    q-dir {str(kd):9}: H_eff eigenvalues = {np.round(eig, 4)}")
    if m == 3:
        print(f"    => two linear cones + a flat band ({{-v,0,+v}}), ANISOTROPIC (v varies by direction:")
        print(f"       ~4.44 axial vs ~3.63 along (111)).  A charge-2 'double-Weyl' node (Berry charge 2,")
        print(f"       rigorous in explore_09); pseudospin-1-TYPE, but not the isotropic so(3) spin-1.")
    if m == 2:
        print(f"    => a single linear (anisotropic) Weyl cone {{-v,+v}}; Berry charge 1 (explore_09): Weyl point.")

print("\nPART C — structure of a general solution")
print("  psi(t) = sum_{n,k} c_{n,k} e^{-i E_n(k) t} |n,k>.")
print("  Conserved: the energy distribution |c_{n,k}|^2 (phases only rotate) and crystal momentum k.")
print("  A wave packet at k0 propagates at the group velocity v_g = grad_k E_n(k0) — the ballistic")
print("  light-cone of explore_16.  Stationary solutions: the zero modes (b_1 harmonic forms / cohomology).")
print("  => the full solution manifold = {extended Bloch modes (relativistic near the nodes)}")
print("     plus the finite-dim stationary (harmonic) sector; nothing else.")
