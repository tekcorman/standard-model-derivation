"""
explore_16 — the dynamics, made concrete: evolve flows on a finite srs patch from a localized
(low-entropy) initial state, and show the arrow of time. Pure math.

LAW (forced by the object): heat/diffusion  d_t p = -L p  (L = graph Laplacian = D^2);
                            wave/Dirac       d_t psi = i A psi  (unitary tight-binding flow).
INITIAL CONDITION (free initial data): a localized state at the origin (low entropy).
What we show: heat -> monotone entropy increase (irreversible arrow) + diffusive spreading <r^2>~t;
              wave -> ballistic spreading <r^2>~t^2 (the geodesic light-cone), unitary.
"""
import numpy as np, srs, math

R = 3
cells = [(a, b, c) for a in range(-R, R+1) for b in range(-R, R+1) for c in range(-R, R+1)]
cidx = {c: i for i, c in enumerate(cells)}
ncell = len(cells); nv = 4*ncell
def vid(s, cell): return cidx[cell]*4 + s

A = np.zeros((nv, nv))
for cell in cells:
    a, b, c = cell
    for (i, j, v) in srs.EDGES:
        nbr = (a+v[0], b+v[1], c+v[2])
        if nbr in cidx:
            x, y = vid(i, cell), vid(j, nbr); A[x, y] += 1; A[y, x] += 1
deg = A.sum(1); L = np.diag(deg) - A
posv = np.zeros((nv, 3))
for cell in cells:
    for s in range(4): posv[vid(s, cell)] = cell
r2 = (posv**2).sum(1)
print(f"finite srs patch: R={R}  ->  {ncell} cells, {nv} vertices (interior degree {int(deg.max())})")

psi0 = np.zeros(nv); psi0[vid(0, (0, 0, 0))] = 1.0
wL, VL = np.linalg.eigh(L)
wA, VA = np.linalg.eigh(A)
def heat(t):    p = VL@(np.exp(-t*wL)*(VL.T@psi0)); p = np.clip(p, 0, None); return p/p.sum()
def wave(t):    psi = VA@(np.exp(1j*t*wA)*(VA.T@psi0)); p = np.abs(psi)**2; return p/p.sum()

print("\nHEAT flow  d_t p = -L p   from the localized state  (THE ARROW):")
print("   t        entropy S(t)            peak p_max      <r^2>(t)")
for t in [0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
    p = heat(t); S = -np.sum(p[p > 1e-15]*np.log(p[p > 1e-15]))
    print(f"  {t:6.1f}    {S:.4f}  (max {math.log(nv):.4f})    {p.max():.4f}        {np.sum(p*r2):.3f}")
print("  => S(t) increases monotonically from 0 (localized) toward log(nv) (uniform equilibrium).")
print("     The equilibrium is the STATIC fixed point; the arrow comes from the low-entropy start.")

print("\nWAVE/DIRAC flow  d_t psi = i A psi  (unitary):  <r^2>(t)")
for t in [0.3, 0.6, 1.0, 1.5, 2.0]:
    print(f"  t={t:4.1f}   <r^2> = {np.sum(wave(t)*r2):.3f}")

# fit in the INTERMEDIATE window: after the intra-cell transient (the origin's neighbors are all in
# the same cell, r^2=0), before the wavefront reaches the boundary.
th = np.array([1.0, 2.0, 3.0]); ph = [np.sum(heat(t)*r2) for t in th]
tw = np.array([0.6, 0.9, 1.2]); pw = [np.sum(wave(t)*r2) for t in tw]
print(f"\n  spreading exponent  <r^2> ~ t^p  (intermediate window, past the intra-cell transient):")
print(f"    heat  p = {np.polyfit(np.log(th), np.log(ph), 1)[0]:.2f}   (diffusive, expect 1)   [t={list(th)}]")
print(f"    wave  p = {np.polyfit(np.log(tw), np.log(pw), 1)[0]:.2f}   (ballistic, expect 2)    [t={list(tw)}]")
print("\n  Same law D, two regimes: heat = irreversible (entropy up, the arrow); wave = unitary,")
print("  ballistic (the geodesic light-cone, the NB-walk flow). Both need only the initial condition.")
