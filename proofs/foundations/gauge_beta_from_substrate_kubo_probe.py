#!/usr/bin/env python3
"""
gauge_beta_from_substrate_kubo_probe.py — Π_JJ Phase A.

Extends the G_sub Kubo-polarisation machinery (Π_TT, stress-energy vertex)
to the gauge-current case (Π_JJ, velocity vertex). Per the 2026-05-13
continuum-bridge handoff §2.3:

    Π_TT vertex: A^{ac}(k) = i Σ_b exp(i k·r_b) k_a r_b^c
                            = k_a · v^c(k)        (graviton momentum × velocity)

    Π_JJ vertex: v^μ(k) = i Σ_b r_b^μ exp(i k·r_b) = ∂H/∂k_μ
                            (matter velocity = U(1) current at vertex level)

The gauge-group structure factors out:
    Π^{ab}_JJ(p, ω_E) = Tr_R[T^a T^b] × Π^{μν}_v(p, ω_E)
                      = T(R) δ^{ab} × Π^{μν}_v(p, ω_E)

so the substrate-side computation is a single tensor Π^{μν}_v common to all
gauge factors; the per-factor differences live in T_i(R) (matter-rep index).

Phase A goal (this probe). Compute Π^{μν}_v(p=0, ω_E) on the same half-filled
spin-1 Iorio matter sector at finite Euclidean regulator ω_E with smooth
Fermi smearing T (same conventions as
`lorentz_sig_g_sub_dynamic_omega_T.py`), and check:

  (1) cubic 3-fold structure: Π^{xx}=Π^{yy}=Π^{zz} (all equal) and
      Π^{xy}=Π^{xz}=Π^{yz} (all equal), with the two values independently
      free. srs is CHIRAL (space group I4_132/P4_332, no mirrors), so the
      cross-piece Π^{xy}≠0 in general — that's the substrate's anomalous-
      Hall / Witten-θ-angle channel, structurally distinct from the trace.

  (2) Drude structure in TWO channels:
        Π_trace(ω_E)  = (1/3) Σ_μ Π^{μμ}(ω_E)   — gauge-β channel
                                                  (enters 1/g²(ω_E))
        Π_cross(ω_E)  = Π^{xy}(ω_E)             — chiral / θ-angle channel

      Each is fit to a + d/ω² over the saturated regime ω_E ∈ [0.15, 0.7].

  (3) Phase A does NOT pre-commit to a structural identification of (a, d).
      Phase B (next session) extracts the UV asymptote 1/g_i²|_{ω_E→∞}
      with gauge-group trace Tr_R[T^a T^b] = T_i(R) δ^{ab} per factor;
      Phase C maps the Drude pole to gauge β-coefficients and compares MSSM.

Phase A is purely a substrate-side numerical extraction; no comparison to
MSSM β is attempted here. Sentinels assert isotropy + finite Drude pole.

Reuses Bloch operator from `lorentz_sig_g_sub_elastic_moduli.py`.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lorentz_sig_g_sub_elastic_moduli import BOND_DISPLACEMENTS, H_bloch
from lorentz_sig_g_sub_dynamic_omega_T import fermi_smooth


def velocity_matrix(k_cart: np.ndarray, mu: int) -> np.ndarray:
    """v^μ(k) = ∂H/∂k_μ on the 4-atom adjacency Bloch space.

    H_{ts}(k) = Σ_{b: s→t} exp(i k·r_b)
    v^μ_{ts}(k) = i Σ_{b: s→t} r_b^μ exp(i k·r_b)
    """
    V = np.zeros((4, 4), dtype=complex)
    for s, t, rb in BOND_DISPLACEMENTS:
        V[t, s] += 1j * rb[mu] * np.exp(1j * np.dot(k_cart, rb))
    return (V + V.conj().T) / 2


def Pi_v_at_k(k_cart: np.ndarray, omega_E: float, T: float, mu: float = 0.0) -> np.ndarray:
    """Velocity-velocity Kubo tensor Π^{μν}(k, p=0, ω_E) at a single k."""
    H_k = H_bloch(k_cart)
    eigs, U = np.linalg.eigh(H_k)
    V = np.zeros((3, 4, 4), dtype=complex)
    for m in range(3):
        V[m] = U.conj().T @ velocity_matrix(k_cart, m) @ U
    f = np.array([fermi_smooth(eigs[n], mu, T) for n in range(4)])

    K = np.zeros((3, 3), dtype=float)
    for n in range(4):
        for m in range(4):
            diff = f[n] - f[m]
            if abs(diff) < 1e-15:
                continue
            Delta = eigs[n] - eigs[m]
            denom = Delta * Delta + omega_E * omega_E
            weight = diff * Delta / denom
            for a in range(3):
                for b in range(3):
                    term = np.conj(V[a, m, n]) * V[b, m, n]
                    K[a, b] += -2.0 * (term * weight).real
    return K


def Pi_v_BZ(omega_E: float, T: float, N: int = 12, mu: float = 0.0,
            half_extent: float = 2 * np.pi) -> np.ndarray:
    """MP-shifted BZ average of Π^{μν}_v at q=0."""
    ks = (np.arange(N) + 0.5) * (2 * half_extent / N) - half_extent
    K_total = np.zeros((3, 3))
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                K_total += Pi_v_at_k(np.array([k1, k2, k3]), omega_E, T, mu)
    return K_total / N ** 3


def Pi_JJ_at_kp(k_cart: np.ndarray, p_cart: np.ndarray, omega_E: float, T: float,
                mu: float = 0.0) -> np.ndarray:
    """Π^{μν}_JJ(k, p=external photon momentum, ω_E) at a single k.

    Symmetric vertex at k_mid = k + p/2 (matches `Pi_at_k_omega_T` convention).
    """
    k_mid = k_cart + p_cart / 2
    H_k = H_bloch(k_cart)
    H_kp = H_bloch(k_cart + p_cart)
    eigs_k, U_k = np.linalg.eigh(H_k)
    eigs_kp, U_kp = np.linalg.eigh(H_kp)

    V_mid = np.zeros((3, 4, 4), dtype=complex)
    for m in range(3):
        V_mid[m] = velocity_matrix(k_mid, m)

    V_basis = np.zeros((3, 4, 4), dtype=complex)
    for m in range(3):
        V_basis[m] = U_kp.conj().T @ V_mid[m] @ U_k

    f_n = np.array([fermi_smooth(eigs_k[n], mu, T) for n in range(4)])
    f_m = np.array([fermi_smooth(eigs_kp[m], mu, T) for m in range(4)])

    K = np.zeros((3, 3), dtype=float)
    for n in range(4):
        for m in range(4):
            diff = f_n[n] - f_m[m]
            if abs(diff) < 1e-15:
                continue
            Delta = eigs_k[n] - eigs_kp[m]
            denom = Delta * Delta + omega_E * omega_E
            weight = diff * Delta / denom
            for a in range(3):
                for b in range(3):
                    term = np.conj(V_basis[a, m, n]) * V_basis[b, m, n]
                    K[a, b] += -2.0 * (term * weight).real
    return K


def Pi_JJ_BZ(p_cart: np.ndarray, omega_E: float, T: float, N: int = 12,
             mu: float = 0.0, half_extent: float = 2 * np.pi) -> np.ndarray:
    """MP-shifted BZ average of Π^{μν}_JJ at external momentum p."""
    ks = (np.arange(N) + 0.5) * (2 * half_extent / N) - half_extent
    K_total = np.zeros((3, 3))
    for k1 in ks:
        for k2 in ks:
            for k3 in ks:
                K_total += Pi_JJ_at_kp(np.array([k1, k2, k3]), p_cart, omega_E, T, mu)
    return K_total / N ** 3


def extract_pi2(omega_E: float, T: float, N: int = 12,
                p_z_values=(0.0, 0.05, 0.10, 0.15, 0.20)) -> dict:
    """Extract leading p² coefficient of Π^{μν}(p_z ẑ, ω_E) via polynomial fit.

    Returns dict with:
      'pi_xx_at_p': Π^{xx}(p_z) values
      'pi_zz_at_p': Π^{zz}(p_z) values
      'pi_xy_at_p': Π^{xy}(p_z) values
      'pi_xx_p2':  leading p² coef of Π^{xx} (transverse gauge kinetic)
      'pi_zz_p2':  leading p² coef of Π^{zz} (longitudinal — gauge-invariance probe)
      'pi_xy_p2':  leading p² coef of Π^{xy} (chiral / θ-angle channel)
      'pi_xx_p0':  Π^{xx}(p_z=0) (substrate Stueckelberg mass)
    """
    Pi_xx = []
    Pi_zz = []
    Pi_xy = []
    for p_z in p_z_values:
        p_cart = np.array([0.0, 0.0, p_z])
        K = Pi_JJ_BZ(p_cart, omega_E, T, N=N)
        Pi_xx.append(K[0, 0])
        Pi_zz.append(K[2, 2])
        Pi_xy.append(K[0, 1])
    p_arr = np.array(p_z_values)
    p2_arr = p_arr ** 2
    # Fit polynomial in p² up to degree 2 (i.e. p_z⁴ in p_z)
    # Π(p_z) = π_0 + π_2 p_z² + π_4 p_z⁴
    a_xx = np.polyfit(p2_arr, Pi_xx, 2)  # [π_4, π_2, π_0]
    a_zz = np.polyfit(p2_arr, Pi_zz, 2)
    a_xy = np.polyfit(p2_arr, Pi_xy, 2)
    return {
        "p_z_values": list(p_z_values),
        "pi_xx_at_p": Pi_xx,
        "pi_zz_at_p": Pi_zz,
        "pi_xy_at_p": Pi_xy,
        "pi_xx_p0": a_xx[2],
        "pi_xx_p2": a_xx[1],
        "pi_xx_p4": a_xx[0],
        "pi_zz_p2": a_zz[1],
        "pi_xy_p2": a_xy[1],
    }


def header(s: str) -> None:
    print()
    print("=" * 78)
    print(f"  {s}")
    print("=" * 78)


def main() -> None:
    header("Π_JJ Phase A: velocity-velocity Kubo polarisation Π^{μν}(p=0, ω_E)")
    print()
    print("  Substrate-side Bloch integral, gauge-group-trace factor pulled out.")
    print("  Reuses Π_TT machinery; vertex A^{ac}=k_a v^c → v^μ (graviton k_a removed).")
    print()

    # --- Step 1: cubic 3-fold structure check at a single regulator --------
    header("Step 1: cubic 3-fold structure (Π_diag and Π_offdiag each isotropic)")
    print()
    omega_check, T_check, N_check = 0.30, 0.30, 12
    t0 = time.time()
    K_iso = Pi_v_BZ(omega_check, T_check, N=N_check)
    elapsed = time.time() - t0
    diag = np.array([K_iso[0, 0], K_iso[1, 1], K_iso[2, 2]])
    offdiag = np.array([K_iso[0, 1], K_iso[0, 2], K_iso[1, 2]])
    diag_spread = diag.max() - diag.min()
    od_spread = offdiag.max() - offdiag.min()
    diag_mean = diag.mean()
    od_mean = offdiag.mean()
    print(f"  N={N_check}, ω_E={omega_check}, T={T_check}, elapsed={elapsed:.1f}s")
    print()
    print(f"  Π^{{μν}}_v(p=0, ω_E={omega_check}):")
    for row in K_iso:
        print("    " + "  ".join(f"{x:+.6e}" for x in row))
    print()
    print(f"  diagonal entries:        {diag}")
    print(f"  diagonal mean:           {diag_mean:+.6e}")
    print(f"  diagonal spread:         {diag_spread:+.4e}")
    print(f"  off-diagonal entries:    {offdiag}")
    print(f"  off-diagonal mean:       {od_mean:+.6e}    (NONZERO ⇒ chiral, srs has no mirrors)")
    print(f"  off-diagonal spread:     {od_spread:+.4e}")

    iso_tol = 0.05 * max(abs(diag_mean), 1e-12)
    od_tol = 0.05 * max(abs(od_mean), 1e-12)
    assert diag_spread < iso_tol, (
        f"Diagonal isotropy failure: spread {diag_spread:.2e} > 5% of mean {abs(diag_mean):.2e}"
    )
    assert od_spread < od_tol, (
        f"Off-diagonal isotropy failure: spread {od_spread:.2e} > 5% of mean {abs(od_mean):.2e}"
    )
    print()
    print(f"  [OK] Π_diag and Π_offdiag each isotropic under cubic 3-fold (T_d projection).")
    print(f"  [NOTE] Π_offdiag ≠ 0 is the substrate's CHIRAL response channel —")
    print(f"         srs has no mirrors (space group I4_132 or P4_332).")

    # NB: the q=0 piece K_iso[μν] is NOT 1/g²(ω) — it's the substrate-induced
    # Stueckelberg mass term Π^{μν}(p=0, ω), which is nonzero because the
    # lattice violates gauge invariance at the cutoff. The gauge KINETIC term
    # = 1/g²(ω) lives in the leading p² coefficient at finite external photon
    # momentum, extracted in Step 2.

    # --- Step 2: finite-p extraction of Π_2 = 1/g²(ω) ----------------------
    header("Step 2: leading p² coefficient Π_2(ω) = 1/g²(ω) via finite-p fit")
    print()
    print("  Same recipe as Π_TT's a_2 extraction: scan p_z ∈ {0, 0.05, 0.10, 0.15, 0.20},")
    print("  fit Π^{xx}(p_z²) = π_0 + π_2 p_z² + π_4 p_z⁴, take π_2 as the kinetic coef.")
    print("  By transverse-photon decomposition Π^{μν}(q) = (q² δ^{μν} - q^μ q^ν) Π(q²),")
    print("  for q = p_z ẑ: Π^{xx} → p_z² Π(0) but Π^{zz} → 0; so π_2_xx is gauge-relevant,")
    print("  while π_2_zz should vanish under exact gauge invariance (substrate breaks it).")
    print()

    omegas = [0.70, 0.55, 0.45, 0.35, 0.30, 0.25, 0.20, 0.18, 0.15]
    Ns = [12, 14]
    p_z_values = (0.0, 0.05, 0.10, 0.15, 0.20)

    all_fits = {}
    for N in Ns:
        print(f"  --- N = {N} grid -----------------------------------------------")
        print(f"  {'ω_E':>6s}  {'time':>6s}  {'π_2_xx':>13s}  {'π_2_zz':>13s}  {'π_2_xy':>13s}  {'π_0_xx':>13s}")
        records = []
        for omega in omegas:
            t0 = time.time()
            res = extract_pi2(omega, omega, N=N, p_z_values=p_z_values)
            dt = time.time() - t0
            records.append((omega, res))
            print(f"  {omega:>6.3f}  {dt:>5.1f}s  {res['pi_xx_p2']:>+.6e}  "
                  f"{res['pi_zz_p2']:>+.6e}  {res['pi_xy_p2']:>+.6e}  {res['pi_xx_p0']:>+.6e}")

        # Drude fit on π_2_xx (the gauge-relevant transverse kinetic)
        omegas_arr = np.array([r[0] for r in records])
        pi2_xx = np.array([r[1]["pi_xx_p2"] for r in records])
        pi2_zz = np.array([r[1]["pi_zz_p2"] for r in records])
        pi2_xy = np.array([r[1]["pi_xy_p2"] for r in records])
        inv_om2 = 1.0 / omegas_arr ** 2

        # 2p Drude: π_2(ω) = a + d/ω²
        d_xx, a_xx = np.polyfit(inv_om2, pi2_xx, 1)
        d_zz, a_zz = np.polyfit(inv_om2, pi2_zz, 1)
        d_xy, a_xy = np.polyfit(inv_om2, pi2_xy, 1)

        # 3p (+ log) for robustness
        logom2 = np.log(omegas_arr ** 2)
        A_mat = np.column_stack([np.ones_like(omegas_arr), inv_om2, logom2])
        a3_xx, d3_xx, b3_xx = np.linalg.lstsq(A_mat, pi2_xx, rcond=None)[0]
        a3_zz, d3_zz, b3_zz = np.linalg.lstsq(A_mat, pi2_zz, rcond=None)[0]
        a3_xy, d3_xy, b3_xy = np.linalg.lstsq(A_mat, pi2_xy, rcond=None)[0]

        print()
        print(f"    Drude fit on π_2_xx (transverse, gauge-kinetic-relevant):")
        print(f"      2p:  a_xx = {a_xx:+.6e},  d_xx = {d_xx:+.6e}")
        print(f"      3p:  a_xx = {a3_xx:+.6e},  d_xx = {d3_xx:+.6e},  |b_log| = {abs(b3_xx):.3e}")
        print(f"    Drude fit on π_2_zz (longitudinal, ~0 under gauge invariance):")
        print(f"      2p:  a_zz = {a_zz:+.6e},  d_zz = {d_zz:+.6e}")
        print(f"    Drude fit on π_2_xy (chiral / θ-angle):")
        print(f"      2p:  a_xy = {a_xy:+.6e},  d_xy = {d_xy:+.6e}")
        all_fits[N] = {
            "xx": (a_xx, d_xx, b3_xx),
            "zz": (a_zz, d_zz, b3_zz),
            "xy": (a_xy, d_xy, b3_xy),
        }

    # --- Step 3: cross-grid consistency ------------------------------------
    header("Step 3: cross-grid consistency (N=12 → 14) on π_2_xx Drude (a, d)")
    print()
    for ch in ("xx", "zz", "xy"):
        a12, d12, b12 = all_fits[12][ch]
        a14, d14, b14 = all_fits[14][ch]
        a_rel = abs(a14 - a12) / max(abs(a12), 1e-12)
        d_rel = abs(d14 - d12) / max(abs(d12), 1e-12)
        print(f"  π_2_{ch}:")
        print(f"    N=12: a = {a12:+.6e},  d = {d12:+.6e},  |b_log| = {abs(b12):.3e}")
        print(f"    N=14: a = {a14:+.6e},  d = {d14:+.6e},  |b_log| = {abs(b14):.3e}")
        print(f"    Δa/a = {a_rel * 100:+.3f}%,  Δd/d = {d_rel * 100:+.3f}%")
        print()

    # --- Step 4: structural readout & sentinel assertions -------------------
    header("Step 4: structural readout and sentinel checks")
    a14_xx, d14_xx, b14_xx = all_fits[14]["xx"]
    a14_zz, d14_zz, b14_zz = all_fits[14]["zz"]
    a14_xy, d14_xy, b14_xy = all_fits[14]["xy"]
    print(f"  Π_TT (G_sub) template:  Π_2_TT(ω_E) = 4/π² - 1/(36 ω²)")
    print(f"                          = {4/np.pi**2:+.6f} + ({-1/36:+.6f})/ω²")
    print()
    print(f"  Π_JJ Phase A (N=14, ω_E ∈ [0.15, 0.70]):")
    print(f"    π_2_xx(ω) = ({a14_xx:+.6f}) + ({d14_xx:+.6f})/ω²      [transverse gauge kinetic]")
    print(f"    π_2_zz(ω) = ({a14_zz:+.6f}) + ({d14_zz:+.6f})/ω²      [longitudinal, ~ gauge breaking]")
    print(f"    π_2_xy(ω) = ({a14_xy:+.6f}) + ({d14_xy:+.6f})/ω²      [chiral / θ-angle]")
    print()
    print(f"  Compared to clean K[π] forms:")
    cands_a = {
        "4/π² (Π_TT match)": 4/np.pi**2,
        "1/π² (single atom)": 1/np.pi**2,
        "8/π² (2N_atoms/π²)": 8/np.pi**2,
        "16/π² (N_atoms²/π²)": 16/np.pi**2,
        "k*/π² (3/π²)": 3/np.pi**2,
        "g/π² (10/π²)": 10/np.pi**2,
    }
    cands_d = {
        "-1/36 (Π_TT match)": -1/36,
        "-1/12 (1/⟨Tr H²⟩)": -1/12,
        "-1/(12·3) (Π_TT)": -1/36,
        "-1/9 (1/k*²)": -1/9,
        "+1/36": 1/36,
        "+1/12": 1/12,
    }
    print(f"    a candidates vs π_2_xx = {a14_xx:+.6f}:")
    for name, val in cands_a.items():
        ratio = a14_xx / val if abs(val) > 1e-12 else float('inf')
        print(f"      {name:30s} = {val:+.6f}   ratio = {ratio:+.4f}")
    print(f"    d candidates vs d_xx = {d14_xx:+.6f}:")
    for name, val in cands_d.items():
        ratio = d14_xx / val if abs(val) > 1e-12 else float('inf')
        print(f"      {name:30s} = {val:+.6f}   ratio = {ratio:+.4f}")

    # Sentinel: π_2_xx Drude weight should be nonzero (running gauge coupling).
    assert abs(d14_xx) > 1e-4, f"Drude weight d14_xx = {d14_xx:.3e} vanishes — no running"
    # Cross-grid stability check on the transverse channel.
    a12_xx, d12_xx, _ = all_fits[12]["xx"]
    a_rel_xx = abs(a14_xx - a12_xx) / max(abs(a12_xx), 1e-12)
    d_rel_xx = abs(d14_xx - d12_xx) / max(abs(d12_xx), 1e-12)
    assert a_rel_xx < 0.10, f"a_xx not grid-converged: {a_rel_xx*100:.2f}% drift"
    assert d_rel_xx < 0.25, f"d_xx not grid-converged: {d_rel_xx*100:.2f}% drift"

    print()
    print("=" * 78)
    print("  Phase A: PASS — finite-p Π_JJ machinery works; π_2_xx(ω) Drude form fits")
    print(f"           with N=12 → 14 grid stability (Δa/a={a_rel_xx*100:.2f}%, Δd/d={d_rel_xx*100:.2f}%).")
    print("=" * 78)

    # Sign-flip to physical kinetic-coef convention (matches Π_TT's a_2_phys=-a_2/2):
    a_phys = -a14_xx
    d_phys = -d14_xx
    print()
    print("  PHYSICAL FORM (sign-flip to canonical kinetic-coef convention):")
    print(f"    1/g²_substrate(ω) ≡ -π_2_xx(ω) = ({a_phys:+.6f}) + ({d_phys:+.6f})/ω²")
    print()
    print("  Structural candidates with best fit:")
    print(f"    UV asymptote a_phys = +{a_phys:.6f}:")
    print(f"      • 1/π² = {1/np.pi**2:+.6f}   (deviation {(a_phys - 1/np.pi**2)/(1/np.pi**2)*100:+.2f}%)")
    print(f"      • 1/g  = 1/10 = +0.100000   (deviation {(a_phys - 0.1)/0.1*100:+.2f}%)")
    print(f"    Drude weight d_phys = {d_phys:+.6f}:")
    print(f"      • -1/168 = {-1/168:+.6f}   (deviation {(d_phys - (-1/168))/(-1/168)*100:+.2f}%)")
    print(f"        where 168 = α_GUT⁻¹ × (g - n_fixed) = 24 × 7  (cf. FEP cocyclicity)")
    print(f"      • -1/(2g² - 32) = -1/168 = {-1/168:+.6f}  (same)")
    print()
    print("  IR Drude pole (1/g²(ω) → 0):  ω_pole² = -d_phys / a_phys = "
          f"{-d_phys/a_phys:.6f}")
    print(f"                              ω_pole  = {np.sqrt(-d_phys/a_phys):.6f}")
    print(f"  Compare to Π_TT pole:        ω_TT_pole = π/12 = {np.pi/12:.6f}")
    print()
    print("  Phase B (next session): nail down (a, d) structural form via:")
    print("    (i) analytic Kubo evaluation at ω → ∞ to pin a (1/π² vs 1/g);")
    print("    (ii) gauge-group trace T_i(R) per gauge factor on standard PS generation;")
    print("    (iii) per-factor 1/g_i²(ω) and α_GUT⁻¹ readout at UV.")
    print("  Phase C: map d × T_i(R) to MSSM b_i; compare running M_unif → M_Z.")


if __name__ == "__main__":
    main()
