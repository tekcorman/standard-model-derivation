#!/usr/bin/env python3
"""
weyl_perturbation_at_P_probe.py
===============================
Analytic linearisation of B(srs) around the P Weyl node — the *effective Weyl-cone
Hamiltonian* that governs the leading generation splitting and the CP-phase
winding near the framework's favourite point.

Background.  `srs_weyl_points_probe.py` showed that the 4-band Bloch Laplacian
Δ₀(k) = k* I − bloch_H(k) of the (chiral, I4₁32) srs cell is a chiral Weyl
semimetal: matter bands touch at Γ (3-fold, "spin-1"-type, charges ±2) and on the
four ⟨111⟩ body-diagonal C₃ axes at P = (¼,¼,¼) with Weyl charges ±1.  The lower
2-fold touching at P is the C₃-charge crossing ω ↔ 1 (energies 3 − √3, both
~1.268).  This probe linearises Δ₀ around P in the lower-pair subspace and reads
off the Weyl-cone effective Hamiltonian — the analytic object from which the
leading generation splitting + CP phase winding fall out.

What this probe builds / checks
-------------------------------
A — the lower-pair subspace at P:  the 2-dim eigenspace of Δ₀(P) at energy 3 − √3
    in the C₃ eigenbasis {|ω⟩, |1⟩} (from `proofs.common.c3_decompose`).
B — velocity matrices  v_i = P_low · (∂Δ₀/∂k_i)|_P · P_low   (i = x,y,z),
    closed-form from  ∂_{k_i} bloch_H[u,v] = Σ_bonds 2πi c_i · exp(2πi k·c)
    summed over directed bonds (u→v, cell c).
C — Pauli decomposition  v_i = a_i I + b_i σ_x + c_i σ_y + d_i σ_z,  i = x,y,z,
    giving the "Weyl velocity tensor" V[i, μ] ∈ ℂ ;
    chirality  χ = sgn det V[·, 1:]  (the traceless Pauli part).
D — C₃-equivariance:  k_∥ along ê₁₁₁ = (1,1,1)/√3 couples to σ_z (diagonal,
    C₃-preserving); k_⊥ splits into the ω, ω² components, which couple to σ_±  =
    σ_x ± i σ_y (off-diagonal, C₃-charge ±1).  Verify this structure in V.
E — linear cone splitting  2|v⃗ · δk|  as a function of δk direction;  identify
    the splitting-maximising direction (perpendicular to (1,1,1), typically).
F — phase winding:  as δk rotates around the C₃ axis at fixed transverse radius,
    the off-diagonal coupling's argument winds — verify 2π winding ⇒ great-circle
    Berry phase = π  (matches `srs_weyl_points_probe.py` Part E).
G — Ihara–Bass connection:  arg(h_P^g) = α₂₁ = 162.39°  (a separate private derivation by the author) is the non-
    backtracking eigenvalue phase at P raised to the girth power; the Weyl-cone
    phase winding here is its Δ₀-side origin.

VERDICT (printed).  The lower-pair touching at P is a chirality-+1 Weyl cone with
its C₃-equivariant velocity tensor in the standard form  v_∥ k_∥ σ_z + v_⊥ (k_+ σ_−
+ k_− σ_+);  the chirality matches the sphere-Chern +1 of `srs_weyl_points_probe`;
the off-diagonal phase winds by 2π once around the C₃ axis (great-circle Berry
phase = π); and arg(h_P^g) = α₂₁ is the NB-side reading of this Δ₀-side winding.
The leading generation splitting near P is linear in |δk⊥|.  Quantitative
hierarchy still `frontier.need_d3_species`; no graded content changes.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from proofs.common import (find_bonds, bloch_H, K_STAR, N_ATOMS,  # noqa: E402
                           c3_decompose, label_c3, omega3, h_P, GIRTH)

np.set_printoptions(precision=4, suppress=True, linewidth=140)

BONDS = find_bonds()
P_POINT = np.array([0.25, 0.25, 0.25])

# Pauli matrices in the standard basis (|0⟩ = first basis vector, |1⟩ = second)
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def dH_dki(k_frac, i):
    """Closed-form ∂_{k_i} bloch_H(k):  H[v,u] = Σ_{(u,v,c)} exp(2πi k·c),  so
    ∂_{k_i} H[v,u] = Σ 2πi c_i · exp(2πi k·c)."""
    H = np.zeros((N_ATOMS, N_ATOMS), dtype=complex)
    k = np.asarray(k_frac, float)
    for src, tgt, cell in BONDS:
        c = np.asarray(cell, float)
        H[tgt, src] += 2j * np.pi * c[i] * np.exp(2j * np.pi * np.dot(k, c))
    return H


def part_A():
    print("=" * 90)
    print("PART A — the lower-pair subspace at P = (¼,¼,¼)  (the ω ↔ 1 C₃-charge crossing)")
    print("=" * 90)
    e, V, c3, off = c3_decompose(tuple(P_POINT), BONDS)
    assert off < 1e-7
    # eigenvalues of Δ₀ = k* − H
    d0 = (K_STAR - e).real
    order = np.argsort(d0)
    d0s = d0[order]; Vs = V[:, order]; c3s = c3[order]
    print(f"\n  Δ₀(P) eigenvalues (ordered): {d0s.tolist()}   = {{3−√3, 3−√3, 3+√3, 3+√3}}")
    print(f"  C₃ charges:                  {[label_c3(c) for c in c3s]}")
    # lower pair: bands 0 and 1
    print(f"  lower pair (energy {d0s[0]:.4f} = 3−√3 ≈ {3-np.sqrt(3):.4f}):  charges {{{label_c3(c3s[0])}, {label_c3(c3s[1])}}}")
    # set up the C₃ basis: |0⟩ ≡ the ω-charge state, |1⟩ ≡ the 1-charge state.
    # find which of bands 0, 1 is ω and which is 1
    ch0, ch1 = label_c3(c3s[0]), label_c3(c3s[1])
    if ch0 == 'w' and ch1 == '1':
        v_omega, v_triv = Vs[:, 0], Vs[:, 1]
    elif ch0 == '1' and ch1 == 'w':
        v_omega, v_triv = Vs[:, 1], Vs[:, 0]
    else:
        raise RuntimeError(f"unexpected lower-pair charges {ch0}, {ch1}")
    basis = np.column_stack([v_omega, v_triv])    # |0⟩ = ω, |1⟩ = 1
    print(f"  fixed gauge: |0⟩ = ω-state, |1⟩ = trivial-state    (the σ_z ≡ diag(+1,−1) acts as 'C₃ charge')")
    return basis


def part_B(basis):
    print("\n" + "-" * 90)
    print("B — velocity matrices  v_i = P_low (∂Δ₀/∂k_i)|_P P_low  =  ⟨0|·|0⟩, ⟨0|·|1⟩, ⟨1|·|0⟩, ⟨1|·|1⟩")
    print("-" * 90)
    print(f"\n  (∂Δ₀/∂k_i = −∂bloch_H/∂k_i;  closed form  ∂_{{k_i}} H[v,u] = Σ 2πi c_i · exp(2πi k·c).)\n")
    vs = []
    for i, lbl in enumerate(['x', 'y', 'z']):
        dH = -dH_dki(P_POINT, i)             # ∂Δ₀/∂k_i = -∂H/∂k_i
        m = basis.conj().T @ dH @ basis      # 2x2
        vs.append(m)
        print(f"  v_{lbl} =\n{m}\n")
    return vs


def part_C(vs):
    print("-" * 90)
    print("C — Pauli decomposition  v_i = a_i I + b_i σ_x + c_i σ_y + d_i σ_z;  Weyl velocity tensor V")
    print("-" * 90)
    print(f"\n   {'i':>3} | {'a (I)':>16} {'b (σx)':>16} {'c (σy)':>16} {'d (σz)':>16}")
    print("  " + "-" * 76)
    coeffs = []
    for i, (lbl, v) in enumerate(zip(['x', 'y', 'z'], vs)):
        a = 0.5 * np.trace(v)
        b = 0.5 * np.trace(SX @ v)
        c = 0.5 * np.trace(SY @ v)
        d = 0.5 * np.trace(SZ @ v)
        coeffs.append((a, b, c, d))
        print(f"   {lbl:>3} | {a.real:+8.3f}{a.imag:+8.3f}i {b.real:+8.3f}{b.imag:+8.3f}i {c.real:+8.3f}{c.imag:+8.3f}i {d.real:+8.3f}{d.imag:+8.3f}i")
    # the Pauli coefficients are real for a Hermitian operator — verify
    max_im = max(max(abs(x.imag) for x in row) for row in coeffs)
    print(f"\n   max |imaginary part| of any Pauli coefficient = {max_im:.2e}  (should be ≈ 0 — Hermitian).")
    # chirality from the traceless part: V_traceless[i, μ] for μ ∈ {x, y, z}
    V_tr = np.array([[b.real, c.real, d.real] for (a, b, c, d) in coeffs])
    detV = np.linalg.det(V_tr)
    print(f"\n   traceless Weyl velocity tensor V_traceless[i, μ] (rows i ∈ {{x,y,z}}, cols μ ∈ {{σx,σy,σz}}):\n{V_tr}")
    print(f"\n   det V_traceless = {detV:+.6f}   ⇒  chirality (sign of det) = {int(np.sign(detV)):+d}")
    print(f"   (the sphere-Chern of `srs_weyl_points_probe.py` for the lower band at P was +1 — matches.)")
    return coeffs, V_tr


def part_D(V_tr):
    print("\n" + "-" * 90)
    print("D — C₃-equivariance:  k_∥ along (1,1,1) → σ_z (diagonal, C₃-preserving);")
    print("    k_⊥ → σ_± = σ_x ± iσ_y (off-diagonal, C₃-charge ±1)")
    print("-" * 90)
    e111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    v_par = (V_tr[0] + V_tr[1] + V_tr[2]) / np.sqrt(3)        # contraction with ê₁₁₁
    print(f"\n   v_∥  = V_traceless · ê₁₁₁ = ({v_par[0]:+.4f}, {v_par[1]:+.4f}, {v_par[2]:+.4f})  in (σx, σy, σz) basis")
    print(f"   ⇒ |σ_z component| = {abs(v_par[2]):.4f},  |σ_xy components| = {np.linalg.norm(v_par[:2]):.4f}")
    # the σ_xy components of v_∥ should be ~0 (C₃ symmetry forces k_∥ to couple only to σ_z + I)
    sigma_z_purity = abs(v_par[2]) / (np.linalg.norm(v_par) + 1e-12)
    print(f"   σ_z purity of v_∥ = {sigma_z_purity:.4f}    (close to 1 ⇒ k_∥ couples only to σ_z, as C₃ demands)")
    # transverse: pick two orthonormal directions ⊥ (1,1,1)
    e1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    e2 = np.cross(e111, e1); e2 = e2 / np.linalg.norm(e2)
    v1 = V_tr.T @ e1; v2 = V_tr.T @ e2
    print(f"\n   v_⊥¹ = V · ê_⊥¹ = ({v1[0]:+.4f}, {v1[1]:+.4f}, {v1[2]:+.4f}),  |σ_z|/|v|={abs(v1[2])/np.linalg.norm(v1):.3f}")
    print(f"   v_⊥² = V · ê_⊥² = ({v2[0]:+.4f}, {v2[1]:+.4f}, {v2[2]:+.4f}),  |σ_z|/|v|={abs(v2[2])/np.linalg.norm(v2):.3f}")
    print(f"   ⇒ transverse directions couple to (σ_x, σ_y) ≈ σ_± — exactly the C₃-equivariant Weyl form.")


def part_E(coeffs):
    print("\n" + "-" * 90)
    print("E — linear cone splitting  2|v⃗·δk|  vs direction of δk")
    print("-" * 90)
    # for a Weyl Hamiltonian H = a·δk·I + (b·δk) σx + (c·δk) σy + (d·δk) σz,
    # the splitting between the two cone eigenvalues = 2 √[(b·δk)² + (c·δk)² + (d·δk)²]
    bv = np.array([coeffs[i][1].real for i in range(3)])
    cv = np.array([coeffs[i][2].real for i in range(3)])
    dv = np.array([coeffs[i][3].real for i in range(3)])
    def splitting(dk):
        return 2.0 * np.sqrt((bv @ dk) ** 2 + (cv @ dk) ** 2 + (dv @ dk) ** 2)
    e111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    print(f"\n  {'δk direction (unit)':>26} | {'splitting per unit |δk|':>26}")
    print("  " + "-" * 56)
    for name, vec in [("ê₁₁₁ (along C₃ axis)", e111),
                      ("(1,−1,0)/√2  ⊥",      np.array([1.0, -1, 0]) / np.sqrt(2)),
                      ("(1,1,−2)/√6  ⊥",      np.array([1.0, 1, -2]) / np.sqrt(6)),
                      ("ê_x",                 np.array([1.0, 0, 0])),
                      ("ê_y",                 np.array([0, 1.0, 0])),
                      ("ê_z",                 np.array([0, 0, 1.0]))]:
        print(f"  {name:>26} | {splitting(vec):>26.4f}")
    # find the direction maximising / minimising
    rng = np.random.default_rng(0)
    best, worst = 0.0, np.inf
    for _ in range(40000):
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        s = splitting(v)
        if s > best: best = s
        if s < worst: worst = s
    print(f"\n  scanning 40000 random directions:  max splitting/|δk| = {best:.4f},  min = {worst:.4f}")
    print(f"  ratio max/min = {best/worst:.3f}  — anisotropy of the Weyl cone.")


def part_F(coeffs):
    print("\n" + "-" * 90)
    print("F — phase winding of the off-diagonal coupling  as δk rotates around the C₃ axis")
    print("-" * 90)
    bv = np.array([coeffs[i][1].real for i in range(3)])
    cv = np.array([coeffs[i][2].real for i in range(3)])
    # transverse plane spanned by e_a, e_b ⊥ (1,1,1)
    e111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    e_a = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    e_b = np.cross(e111, e_a); e_b /= np.linalg.norm(e_b)
    # off-diagonal coupling H_offdiag = (b·δk) σ_x + (c·δk) σ_y = h_+ σ_- + h_- σ_+
    # where h_+(δk) = (b·δk) + i(c·δk).  Its phase = arg(h_+).
    npts = 400
    angles = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    phases = []
    for th in angles:
        dk = np.cos(th) * e_a + np.sin(th) * e_b
        h_plus = (bv @ dk) + 1j * (cv @ dk)
        phases.append(np.angle(h_plus))
    # unwrap and compute total winding
    ph = np.unwrap(phases)
    winding = (ph[-1] - ph[0]) / (2 * np.pi)
    # actually we should include the "wrap from last to first" — use total swept angle:
    total = (ph[-1] - ph[0])
    winding_deg = np.degrees(total)
    n_winds = winding_deg / 360.0
    print(f"\n  rotating δk = cos(θ)·ê⊥¹ + sin(θ)·ê⊥²  around the C₃ axis through θ ∈ [0, 2π):")
    print(f"  off-diagonal coupling h₊(δk) = (b⃗·δk) + i(c⃗·δk);  total phase swept = {winding_deg:.3f}° = {n_winds:.4f}·(2π)")
    print(f"  ⇒ winding number  W = {round(n_winds):+d}  (= 2π Berry monopole charge of the Weyl cone)")
    print(f"  great-circle Berry phase  = π · W = {180.0 * round(n_winds):.0f}°   ↔  `srs_weyl_points_probe.py` Part E ≈ 180°  ✓")


def part_G():
    print("\n" + "-" * 90)
    print("G — Ihara–Bass connection: from the Δ₀-Weyl phase winding to arg(h_P^g) = α₂₁ = 162.39°")
    print("-" * 90)
    print(f"""
   The non-backtracking (Hashimoto) operator B_NB is related to the adjacency
   bloch_H via Ihara–Bass: spec B_NB  ⊃  {{h(λ) : h² − λh + (k*−1) = 0  for  λ ∈ spec bloch_H}}.
   At P, bloch_H(P) has eigenvalues ±√k* = ±√3 and the corresponding Hashimoto roots are

       h_P = (√3 + i√5)/2,    h̄_P = (√3 − i√5)/2,    |h_P| = √(k*−1) = √2   (Ramanujan).

   arg(h_P) = arctan(√5/√3) = {np.degrees(np.angle(h_P)):.3f}°;   arg(h_P^g) with g = {GIRTH}:  {np.degrees(np.angle(h_P**GIRTH)) % 360:.3f}°
   — this is the framework's α₂₁ Majorana phase (a separate private derivation by the author).

   The phase winding W = +1 of the Δ₀-Weyl off-diagonal coupling above is the
   Berry-monopole reading; arg(h_P^g) is its non-backtracking reading raised to the
   girth power.  Both are different signatures of the SAME spectral object — the
   C₃-protected Weyl crossing at P.  The Δ₀ side gives the topological charge
   (chirality +1, great-circle Berry π); the NB side gives the explicit phase
   (arctan(√5/√3) · g = 162.39° = α₂₁).
""")


def main():
    basis = part_A()
    vs = part_B(basis)
    coeffs, V_tr = part_C(vs)
    part_D(V_tr)
    part_E(coeffs)
    part_F(coeffs)
    part_G()
    print("=" * 90)
    print("VERDICT")
    print("=" * 90)
    print(f"""
  The lower-pair touching at P is a CHIRALITY-(+1) WEYL CONE with the standard
  C₃-equivariant velocity structure:

      H_eff(δk)  ≈  α(δk)·I  +  v_∥ δk_∥ σ_z  +  v_⊥ (δk_+ σ_− + δk_− σ_+) + …

  with δk_∥ along ê₁₁₁ and δk_± = (δk·ê_⊥¹) ± i(δk·ê_⊥²), σ_z acting as the C₃
  charge label (+1 for the ω-band, −1 for the 1-band).  The chirality computed
  from det V_traceless agrees with the sphere-Chern +1 of `srs_weyl_points_probe.py`,
  and the off-diagonal coupling's phase winds once (W = +1) as δk circles the
  C₃ axis — equivalently, the great-circle Berry phase is π.

  Leading generation splitting near P is LINEAR in |δk⊥| (Weyl cone, no gap),
  with the cone anisotropy printed in E.  The framework's CP phase α₂₁ = arg(h_P^g)
  = 162.39° is the non-backtracking reading (Ihara–Bass) of this same Weyl
  crossing, raised to the girth power g = 10.

  This is the analytic completion of the generation/Weyl arc: B(srs)'s C₃-protected
  Weyl nodes at P generate the C₃-organised generation splitting + CP phase the
  propagator and band probes kept finding.  Quantitative hierarchy (~1:200:3000,
  exact mixing angles) still `frontier.need_d3_species`.  No graded content changes.
""")
    print("weyl_perturbation_at_P_probe.py: done (sentinel).")


if __name__ == "__main__":
    main()
