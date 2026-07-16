#!/usr/bin/env python3
# ============================================================
# Chirality-dependent kernel — STAGE 2: scale-free OVER-DETERMINATION (make-or-break).
# ============================================================
#
# Scope: internal research notes (§7 Stage 2).
# Stage 0 (GREEN, c1e6d55): the OEF vertex E_int=-kappa*I(A;B) IS spin-dependent
# once the walker spinor is kept; the singlet binds deeper than the triplet (QCD
# sign). Stage 2 is the decisive test: ONE functional (the OEF mutual information
# / total correlation of the SPIN state) must reproduce, with NO free parameter
# and NO scale, the RATIO of the two hyperfine splittings:
#
#     R = (m_Delta - m_N) / (m_rho - m_pi)        [observed = 293/637 = 0.460]
#
# kappa cancels in the ratio, so this is a clean falsifiable prediction. It either
# over-determines the data (PASS) or it does not (FAIL/PARTIAL). No tuning.
#
# THE FUNCTIONAL (framework-derived, not chosen here):
#   2-body (mesons): E_int = -kappa * I(A;B)   (two_subsystem_oef_vertex)
#   n-body (baryons): E_int = -kappa * C       (n_body_oef_vertex; C = total
#       correlation = sum_i S(rho_i) - S(rho_joint), Watanabe).
#   Both evaluated on the SPIN state of the hadron. The spin-INDEPENDENT binding
#   cancels in each splitting, so the splitting = -kappa * (Delta of the spin
#   information between the two channels).
#
# THE STATES (the physics that makes this non-trivial):
#   - mesons pi/rho: same flavor (q qbar), differ only in spin -> PURE spin
#     singlet (pi) / triplet (rho). No flavor entanglement.
#   - baryons N/Delta: spin-flavor ENTANGLED (SU(6) 56). The spin state the OEF
#     sees is the FLAVOR-TRACED (mixed) state. Delta spin factorizes (pure S=3/2);
#     the nucleon spin is genuinely mixed. We build the real SU(6) proton (the F8
#     wavefunction) and Delta, trace flavor, and compute C on the actual spin rho.
#
# HONEST FLAGS: (i) the OEF n-body binding = total correlation C (framework-
# derived) is compared to a PAIRWISE-sigma.sigma quark model — they need NOT
# agree, which is the point (falsifiable). (ii) spatial<->spin factorization
# (s-wave), so spin-independent pieces cancel cleanly in the splitting. (iii)
# rotational invariance: each hadron = its full spin MULTIPLET (mixed state).
# (iv) magnitude kappa stays walled -> only the RATIO is claimed. (v) Stage-1
# rigor (actual Cl(6) junction spinors vs SU(6) constituent spins) deferred; here
# the spins are the standard SU(6) constituent spin-1/2 (the F8 spin content).

import numpy as np
from itertools import product

# ---------- qubit / entropy utilities ----------
def vn(rho):
    ev = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log2(ev)))


def marg(rho_n, n, keep):
    """Single-site (keep) reduced density matrix of an n-qubit density matrix."""
    t = rho_n.reshape([2] * n + [2] * n)
    traced = list(range(n))
    traced.remove(keep)
    # trace out every site except `keep`
    for s in sorted(traced, reverse=True):
        t = np.trace(t, axis1=s, axis2=s + (t.ndim // 2))
    return t.reshape(2, 2)


def total_correlation(rho_n, n):
    return sum(vn(marg(rho_n, n, i)) for i in range(n)) - vn(rho_n)


def marg_qudit(rho, n, d, keep):
    """Single-site reduced density matrix of an n-qudit (dim d) density matrix."""
    t = rho.reshape([d] * n + [d] * n)
    traced = [i for i in range(n) if i != keep]
    for s in sorted(traced, reverse=True):
        t = np.trace(t, axis1=s, axis2=s + (t.ndim // 2))
    return t.reshape(d, d)


def total_corr_qudit(rho, n, d):
    return sum(vn(marg_qudit(rho, n, d, i)) for i in range(n)) - vn(rho)


def sf_rho_from_baryon(psi):
    """Full spin-FLAVOR walker density matrix: NO trace (each quark is 4-dim)."""
    return np.outer(psi, psi.conj())


def mutual_info_2(rho2):
    return vn(marg(rho2, 2, 0)) + vn(marg(rho2, 2, 1)) - vn(rho2)


# ---------- meson spin states (pure; flavor is a product, irrelevant) ----------
def meson_spin_rhos():
    up = np.array([1, 0], complex); dn = np.array([0, 1], complex)
    k = lambda a, b: np.kron(a, b)
    singlet = (k(up, dn) - k(dn, up)) / np.sqrt(2)
    tplus, tzero, tminus = k(up, up), (k(up, dn) + k(dn, up)) / np.sqrt(2), k(dn, dn)
    rho_pi = np.outer(singlet, singlet.conj())                      # S=0 multiplet (1 state)
    rho_rho = sum(np.outer(s, s.conj()) for s in (tplus, tzero, tminus)) / 3.0  # S=1 multiplet
    return rho_pi, rho_rho


# ---------- baryon spin states from the REAL SU(6) wavefunctions ----------
# single-quark index = flavor(0=u,1=d)*2 + spin(0=up,1=dn); 3 quarks -> 64-dim.
def ket_from_terms(terms):
    """terms: list of (coeff, ((f,s),(f,s),(f,s))) with f in {+1=u,-1=d}, s in {+1,-1}."""
    psi = np.zeros(4 ** 3, dtype=complex)
    enc = lambda f, s: ((0 if f == +1 else 1) * 2 + (0 if s == +1 else 1))
    for c, st in terms:
        idx = 0
        for (f, s) in st:
            idx = idx * 4 + enc(f, s)
        psi[idx] += c
    return psi / np.linalg.norm(psi)


def spin_rho_from_baryon(psi):
    """Trace out the 3 flavor qubits -> 8x8 spin density matrix (3 spin qubits)."""
    t = psi.reshape(2, 2, 2, 2, 2, 2)         # (f1,s1,f2,s2,f3,s3)
    # trace the 3 flavor indices: rho_spin = sum_f t[...] conj(t[...]) over f1,f2,f3
    rho = np.tensordot(t, t.conj(), axes=([0, 2, 4], [0, 2, 4]))   # -> (s1,s2,s3,s1',s2',s3')
    return rho.reshape(8, 8)


def proton_up_terms():
    uu, ud = (+1, +1), (+1, -1)
    du, dd = (-1, +1), (-1, -1)
    return [(2, (uu, uu, dd)), (-1, (uu, ud, du)), (-1, (ud, uu, du)),
            (2, (uu, dd, uu)), (-1, (uu, du, ud)), (-1, (ud, du, uu)),
            (2, (dd, uu, uu)), (-1, (du, uu, ud)), (-1, (du, ud, uu))]


def proton_dn_terms():
    # spin-flip of proton-up (m=-1/2): flip every quark spin
    return [(c, tuple((f, -s) for (f, s) in st)) for c, st in proton_up_terms()]


def delta_pp_terms(m):
    """Delta++ = uuu, pure spin S=3/2 multiplet member m in {3,1,-1,-3} (units 1/2)."""
    uu, ud = (+1, +1), (+1, -1)
    if m == 3:
        return [(1, (uu, uu, uu))]
    if m == 1:   # symmetric (uud-spin) /sqrt3
        return [(1, (uu, uu, ud)), (1, (uu, ud, uu)), (1, (ud, uu, uu))]
    if m == -1:
        return [(1, (uu, ud, ud)), (1, (ud, uu, ud)), (1, (ud, ud, uu))]
    if m == -3:
        return [(1, (ud, ud, ud))]


def main():
    print("=" * 78)
    print(" CHIRALITY KERNEL — STAGE 2: scale-free over-determination (make-or-break)")
    print("=" * 78)
    print("   ONE functional (OEF spin information), NO scale: predict")
    print("   R = (m_Delta - m_N)/(m_rho - m_pi).  Observed = 293/637 = 0.460.\n")

    # ----- mesons -----
    rho_pi, rho_rho = meson_spin_rhos()
    I_pi, I_rho = mutual_info_2(rho_pi), mutual_info_2(rho_rho)
    split_meson = I_pi - I_rho      # = -(E_pi - E_rho)/kappa = (m_rho - m_pi)/kappa
    print("[mesons] OEF mutual information of the spin state:")
    print(f"    I(pi, S=0) = {I_pi:.4f};  I(rho, S=1) = {I_rho:.4f}")
    print(f"    (m_rho - m_pi)/kappa = I_pi - I_rho = {split_meson:.4f}")

    # ----- baryons (real SU(6) states, flavor traced) -----
    # nucleon spin multiplet (m=+1/2 and -1/2), flavor-traced, averaged
    rN_up = spin_rho_from_baryon(ket_from_terms(proton_up_terms()))
    rN_dn = spin_rho_from_baryon(ket_from_terms(proton_dn_terms()))
    rho_N = (rN_up + rN_dn) / 2.0
    C_N = total_correlation(rho_N, 3)
    # Delta spin multiplet (pure S=3/2), flavor-traced (factorizes -> pure spin)
    rD = np.zeros((8, 8), dtype=complex)
    for m in (3, 1, -1, -3):
        psiD = ket_from_terms(delta_pp_terms(m))
        rD += spin_rho_from_baryon(psiD)
    rho_Delta = rD / 4.0
    C_Delta = total_correlation(rho_Delta, 3)
    split_baryon = C_N - C_Delta    # = (m_Delta - m_N)/kappa
    print("\n[baryons, RECIPE 1: spin-only C, flavor TRACED] (the diagnostic):")
    print(f"    rank(rho_spin): N = {np.linalg.matrix_rank(rho_N, tol=1e-9)}, "
          f"Delta = {np.linalg.matrix_rank(rho_Delta, tol=1e-9)}")
    print(f"    C(N) = {C_N:.4f};  C(Delta) = {C_Delta:.4f};  split = {split_baryon:.4f}")
    print(f"    -> ZERO: tracing flavor sends both to the maximally-mixed S-sector")
    print(f"       (both rank 4) -> total correlation can't see S=1/2 vs S=3/2.")
    print(f"       Tracing flavor discards the SU(6) spin-flavor correlation = WRONG recipe.")

    # RECIPE 2 (principled): total correlation of the FULL spin-flavor walker (no trace;
    # color & spatial are common to N and Delta, so they cancel in the splitting).
    sfN = sum(sf_rho_from_baryon(ket_from_terms(t))
              for t in (proton_up_terms(), proton_dn_terms())) / 2.0
    C_sf_N = total_corr_qudit(sfN, 3, 4)
    sfD = sum(sf_rho_from_baryon(ket_from_terms(delta_pp_terms(m))) for m in (3, 1, -1, -3)) / 4.0
    C_sf_D = total_corr_qudit(sfD, 3, 4)
    split_baryon_sf = C_sf_N - C_sf_D
    print("\n[baryons, RECIPE 2: full spin-FLAVOR walker C (no trace) — the principled one]:")
    print(f"    C_sf(N) = {C_sf_N:.4f};  C_sf(Delta) = {C_sf_D:.4f}")
    print(f"    (m_Delta - m_N)/kappa = C_sf(N) - C_sf(Delta) = {split_baryon_sf:.4f}")
    # use recipe 2 as the headline baryon splitting
    split_baryon = split_baryon_sf

    # ----- the scale-free ratio -----
    R_pred = split_baryon / split_meson
    R_obs = (1232 - 939) / (775.3 - 138.0)
    print("\n[over-determination] kappa cancels in the ratio:")
    print(f"    R_pred = (C_N - C_Delta)/(I_pi - I_rho) = {R_pred:.4f}")
    print(f"    R_obs  = (m_Delta - m_N)/(m_rho - m_pi) = {R_obs:.4f}")
    print(f"    deviation = {100*(R_pred - R_obs)/R_obs:+.1f}%")

    sign_ok = split_meson > 0 and split_baryon > 0
    print(f"\n    signs: rho>pi {'OK' if split_meson>0 else 'WRONG'}, "
          f"Delta>N {'OK' if split_baryon>0 else 'WRONG'}")

    # ---- confound check: is the ratio even scale-free in QCD? (the sigma.sigma model) ----
    print("\n[confound] is (Delta-N)/(rho-pi) a scale-free SPIN-ALGEBRA ratio in QCD?")
    ss_meson = 1 - (-3)        # <sigma.sigma> rho - pi  = +1 -(-3) = 4
    ss_baryon = 3 - (-3)       # <sum sigma_i.sigma_j> Delta - N = +3 -(-3) = 6
    print(f"    universal-coefficient sigma.sigma: (Delta-N)/(rho-pi) = {ss_baryon}/{ss_meson}"
          f" = {ss_baryon/ss_meson:.2f}")
    print(f"    but R_obs = {R_obs:.2f}.  The mismatch ({ss_baryon/ss_meson:.2f} vs {R_obs:.2f})")
    print(f"    is real QCD physics: the meson and baryon hyperfine SCALES differ")
    print(f"    (|psi(0)|^2, reduced masses). So a UNIVERSAL-kappa functional CANNOT")
    print(f"    reproduce 0.46 by spin content alone -> this cross-sector ratio is a")
    print(f"    CONFOUNDED test, not a clean make-or-break.")

    print("\n" + "=" * 78)
    print(" VERDICT — STAGE 2: the over-determination does NOT close (honest negative)")
    print("=" * 78)
    print(f"""  Three findings, reported without spin:

  1. RECIPE 1 (spin-only total correlation, flavor traced) gives ZERO N-Delta
     splitting: tracing flavor sends both N and Delta to the maximally-mixed
     total-spin sector (rank 4), and the total correlation sees only sector
     dimension, not S=1/2 vs S=3/2. The SU(6) spin-flavor correlation -- where the
     nucleon's structure lives -- is exactly what gets discarded.

  2. RECIPE 2 (full spin-flavor walker C) restores the right SIGN (Delta>N, rho>pi)
     but R_pred = {R_pred:.2f} vs R_obs = {R_obs:.2f}. And C_sf(N)={C_sf_N:.1f} >> C_sf(Delta)={C_sf_D:.1f}
     largely because the proton's SU(6) state is far MORE spin-flavor entangled than
     uuu -- i.e. the number is dominated by entanglement CONTENT, not a clean
     hyperfine. The OEF binding is an ENTROPIC correlation measure, NOT the
     sigma_1.sigma_2 OPERATOR; they share the meson sign (Stage 0) but diverge
     quantitatively for baryons.

  3. The cross-sector ratio is CONFOUNDED anyway: even the universal-coefficient
     sigma.sigma quark model gives {ss_baryon/ss_meson:.2f}, not {R_obs:.2f} -- the real meson/baryon
     hyperfine scale difference is non-spin physics no universal-kappa functional
     has. So this was a flawed make-or-break: the ratio is not scale-free.

  DISPOSITION: Stage 0 STANDS (a spin-dependent OEF term exists, with the correct
  meson singlet-below-triplet sign). Stage 2 is a NEGATIVE for the strong claim:
  the OEF information-hyperfine does NOT quantitatively reproduce QCD color-magnetism
  (it is an entropy, not a sigma.sigma operator), and the cross-sector ratio cannot
  cleanly test it. The g_A residual is therefore NOT closed by this mechanism: the
  OEF spin term has the right SIGN but is not demonstrably the right MAGNITUDE or
  OPERATOR. Honest next steps would need a within-sector (common-scale) observable
  or the walled magnitude kappa -- not another recipe (further recipe-hunting on a
  confounded ratio would be fishing). sqrt(phi) and band-selection stay foreclosed;
  g_A stands at LO 5/3 + relativistic ~1.44, residual genuinely open.""")
    print("=" * 78)


if __name__ == "__main__":
    main()
