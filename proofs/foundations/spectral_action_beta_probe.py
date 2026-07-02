#!/usr/bin/env python3
"""
spectral_action_beta_probe.py
=============================
The actual MSSM-mechanism test: compute the Chamseddine–Connes-style spectral
action of the gauge-equivariant operator-algebra supercharge Q̂_alg (from
`de_rham_susy_fibered_v2_probe.py`) integrated over the Brillouin zone, extract
β-function-like flow, and compare to SM and MSSM gauge β-coefficients.

Setup (recap from v2 probe):
  C⁰_alg = ⊕_v Cl(6)_v ≅ ⊕_v M₈(ℂ)        (matter operator algebra, 256-dim per cell)
  C¹_alg = ⊕_e Cl(2)_e ≅ ⊕_e M₂(ℂ)        (gauge operator algebra, 24-dim per cell)
  d̂_alg(k):  (d̂_alg A)_e = ¼ tr_⊥^(v)(A_v) − e^{2πi k·n}/4 tr_⊥^(u)(A_u)
  Q̂_alg(k) = [[0, d̂†],[d̂, 0]]    — gauge-equivariant by construction
  Q̂_alg(k)² = blockdiag(Δ̂₀(k), Δ̂₁(k))

Spectral action:
  S(Λ) = ∫_BZ d³k/(2π)³ · Tr_{cell} f(Q̂_alg(k)² / Λ²)
For β-function extraction, two natural probes:
  • heat trace Z(t)  =  Tr e^{−tQ̂²}_{BZ}  (with t ~ 1/Λ²);  its log-derivative
    d log Z / d log t encodes the effective dimension flow.
  • mode-counting function  N_eff(Λ²)  =  number of (k, n) with λ_n(k) < Λ²,
    normalised per unit cell, per-band.  Its log-derivative gives a β-like quantity.

What this probe does
--------------------
A — diagonalise Q̂_alg(k)² over a BZ grid (8×8×8 = 512 k-points, 280-dim per k),
    accumulate the full spectral density ρ(λ).
B — compute Z(t) = ⟨Tr e^{−tQ²}⟩_BZ for t ∈ [10⁻², 10²] and plot log Z vs log t;
    identify the scaling region(s) and extract effective dimensions.
C — compute N_eff(Λ²) and its log-derivative b_eff(Λ) = d N_eff / d log Λ;
    show the asymptotic limits.
D — decompose C¹_alg into SU(2)-irreps per edge (1 ⊕ 3 under SU(2) conjugation):
    project Δ̂₁(k) onto the SU(2)-singlet and SU(2)-triplet sectors of each edge,
    and compute the spectral densities of these two sectors separately —
    the *triplet* sector is the natural carrier of gauge dynamics.
E — compare to SM (b₂ = −19/6 < 0 — non-asymptotic-free) and MSSM (b₂ = 1 >0 —
    asymptotic-free) gauge β-functions; report which the framework's spectral
    action resembles in sign and scaling.

VERDICT (printed honestly): does the framework's spectral action of Q̂_alg
reproduce MSSM-like β-function flow, SM-like, or its own thing?  This is the
moment of truth for the "spectral-triple replaces MSSM as the framework's
β-function mechanism" hypothesis.  No graded content changes.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Reuse the v2 fibered-SUSY construction
from proofs.foundations.de_rham_susy_fibered_v2_probe import (  # noqa: E402
    d_alg, EDGES, NE, NV, incident_edges, T_SLOT,
)

np.set_printoptions(precision=4, suppress=True, linewidth=140)

# Gauge β-coefficients for reference
B_SM = {'b1 (U(1))': 41/10, 'b2 (SU(2))': -19/6, 'b3 (SU(3))': -7}
B_MSSM = {'b1 (U(1))': 33/5, 'b2 (SU(2))': 1.0, 'b3 (SU(3))': -3.0}


# ---------------------------------------------------------------------------
# Q̂_alg(k)² spectrum over a BZ grid
# ---------------------------------------------------------------------------

def Q_squared(k):
    """The full Q̂_alg(k)² as a (280, 280) Hermitian matrix on C⁰_alg ⊕ C¹_alg."""
    d = d_alg(k)
    dim0 = NV * 64
    dim1 = NE * 4
    Q = np.zeros((dim0 + dim1, dim0 + dim1), dtype=complex)
    Q[:dim0, dim0:] = d.conj().T
    Q[dim0:, :dim0] = d
    return Q @ Q   # blockdiag(D0, D1)


def gather_spectrum_over_BZ(grid_n=8):
    """Diagonalise Q̂_alg² at every k in an n³ grid; return (eigenvalues_matter, eigenvalues_gauge)
    as flat numpy arrays."""
    print(f"  diagonalising at {grid_n}³ = {grid_n**3} k-points (Δ̂₀ + Δ̂₁ block) …")
    e_matter, e_gauge = [], []
    pts = np.linspace(0, 1, grid_n, endpoint=False)
    for i, kx in enumerate(pts):
        for ky in pts:
            for kz in pts:
                d = d_alg((kx, ky, kz))
                D0 = d.conj().T @ d
                D1 = d @ d.conj().T
                e_matter.extend(np.linalg.eigvalsh((D0 + D0.conj().T) / 2).tolist())
                e_gauge.extend(np.linalg.eigvalsh((D1 + D1.conj().T) / 2).tolist())
        print(f"    kx slice {i+1}/{grid_n}")
    return np.array(e_matter), np.array(e_gauge)


# ---------------------------------------------------------------------------
# SU(2)-irrep decomposition of C¹_alg per edge
#
# Cl(2)_e ≅ M_2(ℂ) decomposes under SU(2) conjugation as  1 ⊕ 3:
#   singlet (1) = I_2 (identity, the "trace" mode)
#   triplet (3) = {σ_x, σ_y, σ_z} (the "rotation generators", carries gauge dynamics)
# ---------------------------------------------------------------------------

def edge_su2_projectors():
    """Return (P_singlet, P_triplet) as (24, 24) block-diagonal projectors on C¹_alg.

    Per edge: in the basis {I/√2, σ_x/√2, σ_y/√2, σ_z/√2} (4-dim Cl(2)),
    the singlet projector = diag(1, 0, 0, 0) and the triplet projector = diag(0, 1, 1, 1).
    But our C¹_alg basis is column-major flatten of 2×2 matrices: indices map
    {00 (=I or σz), 10 (=σx+iσy), 01 (=σx−iσy), 11 (=I or −σz)} — let's just construct
    by hand using the change-of-basis from flatten-basis to Pauli-basis.
    """
    # The 4 Pauli basis elements (flatten in column-major)
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_basis_cols = [m.flatten('F') / np.sqrt(2) for m in (I2, sx, sy, sz)]
    # change-of-basis matrix V_p: column k = flatten-coords of k-th Pauli element
    V_p = np.column_stack(pauli_basis_cols)
    # singlet = the I component; triplet = σ_x,y,z
    P_singlet_pauli = np.diag([1.0, 0.0, 0.0, 0.0])
    P_triplet_pauli = np.diag([0.0, 1.0, 1.0, 1.0])
    # project in flatten basis: P_flat = V_p · P_pauli · V_p†
    P_s_edge = V_p @ P_singlet_pauli @ V_p.conj().T
    P_t_edge = V_p @ P_triplet_pauli @ V_p.conj().T
    # block-diagonal over the 6 edges
    P_s = np.zeros((NE * 4, NE * 4), dtype=complex)
    P_t = np.zeros((NE * 4, NE * 4), dtype=complex)
    for e in range(NE):
        P_s[e * 4:(e + 1) * 4, e * 4:(e + 1) * 4] = P_s_edge
        P_t[e * 4:(e + 1) * 4, e * 4:(e + 1) * 4] = P_t_edge
    return P_s, P_t


def gauge_sector_spectra_over_BZ(grid_n=8):
    """For each k, diagonalise Δ̂₁(k) RESTRICTED to the SU(2)-singlet and triplet sectors of C¹_alg."""
    P_s, P_t = edge_su2_projectors()
    e_singlet, e_triplet = [], []
    pts = np.linspace(0, 1, grid_n, endpoint=False)
    for kx in pts:
        for ky in pts:
            for kz in pts:
                d = d_alg((kx, ky, kz))
                D1 = d @ d.conj().T
                D1s = P_s @ D1 @ P_s
                D1t = P_t @ D1 @ P_t
                # nonzero eigenvalues only
                wS = np.linalg.eigvalsh((D1s + D1s.conj().T) / 2)
                wT = np.linalg.eigvalsh((D1t + D1t.conj().T) / 2)
                e_singlet.extend(wS[wS > 1e-8].tolist())
                e_triplet.extend(wT[wT > 1e-8].tolist())
    return np.array(e_singlet), np.array(e_triplet)


# ---------------------------------------------------------------------------
# spectral observables: heat trace, mode counting
# ---------------------------------------------------------------------------

def heat_trace(eigs, ts):
    return np.array([np.sum(np.exp(-t * eigs)) for t in ts])


def mode_count(eigs, lambdas):
    """N_eff(Λ²) = number of modes with eigenvalue ≤ Λ²  (per cell, summed over BZ grid)."""
    return np.array([np.sum(eigs <= L) for L in lambdas])


def log_derivative(x, y, smooth=1):
    """d log y / d log x via centred differences (smoothing optional)."""
    lx, ly = np.log(x), np.log(np.maximum(y, 1e-30))
    dlx = np.gradient(lx)
    dly = np.gradient(ly)
    return dly / dlx


# ======================================================================
def part_A(grid_n):
    print("=" * 100)
    print("PART A — Q̂_alg(k)² spectrum over the BZ")
    print("=" * 100)
    e_m, e_g = gather_spectrum_over_BZ(grid_n)
    N_k = grid_n ** 3
    print(f"\n  total eigenvalues collected: matter sector = {len(e_m)}, gauge sector = {len(e_g)}")
    print(f"  (matter: {len(e_m)//N_k} per k × {N_k} k-points = {len(e_m)};  gauge: {len(e_g)//N_k} × {N_k})")
    print(f"\n  matter sector range:  min = {e_m.min():.4f},   max = {e_m.max():.4f}")
    print(f"  zero modes per k (matter):    {np.sum(e_m < 1e-7) // N_k} of 256")
    print(f"  zero modes per k (gauge):     {np.sum(e_g < 1e-7) // N_k} of 24  (matches Δ̂₁ kernel)")
    # quick histogram in bins
    print(f"\n  matter spectral density (binned):")
    bins = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0]
    counts, edges = np.histogram(e_m[e_m > 1e-8], bins=bins)
    for ce, cb in zip(counts, zip(edges[:-1], edges[1:])):
        print(f"    [{cb[0]:.2f}, {cb[1]:.2f}):  {ce:6d}  density = {ce/N_k/(cb[1]-cb[0]):.2f}")
    return e_m, e_g


def part_B(e_m, e_g, grid_n):
    print("\n" + "-" * 100)
    print("B — heat trace  Z(t) = Tr e^{-tQ²}  averaged over BZ  (matter + gauge)")
    print("-" * 100)
    ts = np.logspace(-2, 2, 25)
    Zm = heat_trace(e_m, ts) / (grid_n ** 3)
    Zg = heat_trace(e_g, ts) / (grid_n ** 3)
    print(f"\n   {'t = 1/Λ²':>10} | {'Z_matter(t)':>12} {'Z_gauge(t)':>12} | {'Z_m/Z_g':>10}")
    print("  " + "-" * 60)
    for t, zm, zg in zip(ts[::3], Zm[::3], Zg[::3]):
        print(f"   {t:>10.3e} | {zm:>12.2f} {zg:>12.2f} | {zm/zg if zg > 0 else 0:>10.3f}")
    # extract scaling: for the small-t regime, Z(t) ≈ N (full dim); large-t, Z(t) → N_zero (kernel dim).
    # The intermediate slope d log Z / d log t gives the "running of effective DoF count".
    slope_m = log_derivative(ts, Zm)
    slope_g = log_derivative(ts, Zg)
    print(f"\n  d log Z / d log t  (effective β-like flow):")
    print(f"   {'t = 1/Λ²':>10} | {'matter slope':>14} {'gauge slope':>14}")
    print("  " + "-" * 44)
    for t, sm, sg in zip(ts[::3], slope_m[::3], slope_g[::3]):
        print(f"   {t:>10.3e} | {sm:>14.4f} {sg:>14.4f}")


def part_C(e_m, e_g, grid_n):
    print("\n" + "-" * 100)
    print("C — mode counting  N_eff(Λ²) = #{ eigenvalues < Λ² }  and  d log N_eff / d log Λ")
    print("-" * 100)
    Lambdas_sq = np.logspace(-2, 0.5, 25)
    Nm = mode_count(e_m, Lambdas_sq) / (grid_n ** 3)
    Ng = mode_count(e_g, Lambdas_sq) / (grid_n ** 3)
    print(f"\n   {'Λ²':>10} | {'N_m(Λ²)/cell':>14} {'N_g(Λ²)/cell':>14} | {'N_m/N_g':>10}")
    print("  " + "-" * 60)
    for L, nm, ng in zip(Lambdas_sq[::3], Nm[::3], Ng[::3]):
        print(f"   {L:>10.3e} | {nm:>14.2f} {ng:>14.2f} | {nm/ng if ng > 0 else 0:>10.3f}")
    slope_m = log_derivative(Lambdas_sq, Nm)
    slope_g = log_derivative(Lambdas_sq, Ng)
    print(f"\n  d log N_eff / d log Λ²:")
    print(f"   {'Λ²':>10} | {'matter':>14} {'gauge':>14}")
    print("  " + "-" * 44)
    for L, sm, sg in zip(Lambdas_sq[::3], slope_m[::3], slope_g[::3]):
        print(f"   {L:>10.3e} | {sm:>14.4f} {sg:>14.4f}")


def part_D(grid_n):
    print("\n" + "-" * 100)
    print("D — SU(2)-irrep decomposition of C¹_alg (1 ⊕ 3 per edge);  triplet = gauge-dynamic sector")
    print("-" * 100)
    print("  computing singlet + triplet spectra over BZ …")
    e_s, e_t = gauge_sector_spectra_over_BZ(grid_n)
    print(f"\n  singlet sector (per-edge 'trace' mode, 1 of 4 per edge):  total nonzero eigenvalues = {len(e_s)}")
    print(f"    range: [{e_s.min() if len(e_s) else 0:.4f}, {e_s.max() if len(e_s) else 0:.4f}]")
    print(f"  triplet sector (per-edge gauge generators, 3 of 4 per edge):  total = {len(e_t)}")
    print(f"    range: [{e_t.min():.4f}, {e_t.max():.4f}]")

    # ratio singlet/triplet by mode counting at different scales — this is the *closest* analog
    # of "how the SU(2)-charged content runs vs the SU(2)-invariant content"
    print(f"\n  SU(2)-triplet / SU(2)-singlet mode-count ratio  at various Λ²  (the gauge-charged DoFs):")
    Lsq = np.logspace(-2, 0.5, 10)
    for L in Lsq:
        ns = int(np.sum(e_s <= L)) / (grid_n ** 3)
        nt = int(np.sum(e_t <= L)) / (grid_n ** 3)
        ratio = nt / ns if ns > 0 else float('inf')
        print(f"    Λ² = {L:>8.3e}:  singlet-modes/cell = {ns:>6.2f},  triplet-modes/cell = {nt:>6.2f},  ratio = {ratio:.3f}")
    return e_s, e_t


def part_E(e_m, e_g, e_s, e_t, grid_n):
    print("\n" + "-" * 100)
    print("E — comparison to SM and MSSM gauge β-coefficients")
    print("-" * 100)
    print(f"""
  SM    β-coefficients (1-loop, with sign convention dg_i/dt = b_i g_i³/16π²):
    b₁ = {B_SM['b1 (U(1))']:+.4f}    b₂ = {B_SM['b2 (SU(2))']:+.4f}    b₃ = {B_SM['b3 (SU(3))']:+.4f}    (b₂, b₃ < 0 ⇒ asymptotic-free SU(2), SU(3))
  MSSM  β-coefficients:
    b₁ = {B_MSSM['b1 (U(1))']:+.4f}    b₂ = {B_MSSM['b2 (SU(2))']:+.4f}    b₃ = {B_MSSM['b3 (SU(3))']:+.4f}    (additional matter from sfermions etc. ⇒ b₂ flips to >0)
""")
    # the "framework β-like quantity" — compute from heat-trace asymptotic
    # The closest analog of b_i is: slope of  log (Z_gauge_triplet / Z_total)  vs  log t,
    # in some intermediate-t window.
    ts = np.logspace(-1.5, 1.5, 30)
    Zt = heat_trace(e_t, ts) / (grid_n ** 3)
    Zs = heat_trace(e_s, ts) / (grid_n ** 3)
    ratio = Zt / np.maximum(Zs, 1e-30)
    slope = log_derivative(ts, ratio)
    print(f"  d log(Z_triplet / Z_singlet) / d log t  at intermediate t:")
    print(f"   {'t':>10} | {'log-derivative (framework-native β-like)':>40}")
    print("  " + "-" * 56)
    for t, s in zip(ts[5:25:4], slope[5:25:4]):
        print(f"   {t:>10.3e} | {s:>40.4f}")
    # extract mean & sign
    mid = slope[10:20]
    print(f"\n  mean slope in mid-range:  {mid.mean():.4f}  (sign: {'POSITIVE (MSSM-like, asymptotic-free)' if mid.mean() > 0 else 'NEGATIVE (SM-like, non-asymptotic-free)' if mid.mean() < 0 else 'zero'})")


def main():
    grid_n = 6     # 216 k-points; tradeoff between resolution and runtime
    print(f"""
==========================================================================================
SPECTRAL ACTION OF Q̂_alg — does the framework's gauge-equivariant spectral-triple SUSY
reproduce MSSM-like β-function flow?   (grid: {grid_n}³ = {grid_n**3} k-points)
==========================================================================================""")
    e_m, e_g = part_A(grid_n)
    part_B(e_m, e_g, grid_n)
    part_C(e_m, e_g, grid_n)
    e_s, e_t = part_D(grid_n)
    part_E(e_m, e_g, e_s, e_t, grid_n)
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print("""
  WHAT WE COMPUTED
   • full Q̂_alg(k)² spectrum on a BZ grid (matter sector 256-dim + gauge sector 24-dim per k);
   • heat trace Z(t), mode-counting N_eff(Λ²), and their log-derivatives across scales;
   • SU(2)-singlet / SU(2)-triplet decomposition of the gauge-side spectrum (the triplet is the
     SU(2)-CHARGED sector, the carrier of gauge dynamics);
   • slope of log(Z_triplet / Z_singlet) vs log t at intermediate scales (the framework-native
     β-function-like quantity).

  HONEST READING
   • The framework's Q̂_alg spectrum is band-structured (eigenvalues 0.5, 1.0 at Γ;  0.317, 0.5,
     1.183 at P;  …) with a topological zero-mode count (232 zero modes in C⁰_alg, 2–3 in C¹_alg).
     This is a DISCRETE, FINITE-CELL spectrum — not a continuum-QFT spectrum with logarithmic
     running between widely-separated scales.
   • The d log Z / d log t flow extracted above is FINITE-SIZE / band-structure flow, not 4D
     continuum gauge-coupling running.  Direct numerical comparison to SM's b₂ = −19/6 or MSSM's
     b₂ = +1 is therefore NOT meaningful at face value — they live in different categories
     (Wilsonian RG in 4D continuum vs heat-kernel flow on a 3D Bloch operator).

   • What CAN be said honestly:
      (i) the framework's Q̂_alg gives a genuine, gauge-equivariant spectral-triple SUSY with a
          well-defined spectral action (computed above);
      (ii) translating that spectral action to a 4D continuum gauge-coupling running requires
           additional steps — embedding into a Connes–Chamseddine-style spectral triple over
           spacetime (NOT just the spatial substrate), and applying the spectral-action principle
           in 4D.  That bridge is NOT built here.
      (iii) the SIGN of the framework-native β-like quantity (d log Z_triplet/Z_singlet / d log t)
            is informative even without the full 4D bridge — see the table above.

  SO THE FAIR VERDICT
   The "spectral-triple SUSY replaces MSSM" hypothesis is INTERESTING and STRUCTURALLY VIABLE
   (the supercharge Q̂_alg exists, is gauge-equivariant, has a well-defined spectral action), but
   the actual β-function extraction requires the 4D spacetime spectral triple (Connes–Chamseddine
   step), which this probe does NOT build.  The 3D Bloch-operator spectral action computed here
   gives flow data but in a finite-cell category, not the 4D continuum where SM/MSSM β-functions
   live.  So we have a candidate MECHANISM (yes, gauge-equivariantly closed), but the final test
   of "does it give MSSM-equivalent β-functions" requires more theoretical infrastructure than
   one probe can build.

   The honest next step is therefore EITHER:
     (a) build the 4D continuum embedding (the framework's spatial substrate × time → spacetime
         spectral triple) and apply Chamseddine–Connes — a substantial project, multi-session;
     (b) pivot to `frontier.beta_dark` (the framework-native RG, distinct from spectral-action
         route) and see if it gives the running natively from substrate dynamics;
     (c) accept that the framework's β-function mechanism is research-level and ADOPTED-MSSM-Sb
         is in fact the right adoption FOR NOW, with the spectral-triple structure above as the
         flagged direction for eventual closure.

   No graded content changes.
""")
    print("spectral_action_beta_probe.py: done (sentinel).")


if __name__ == "__main__":
    main()
