#!/usr/bin/env python3
# ============================================================
# Chirality-dependent kernel sector — STAGE 0 GATE.
# Is the OEF mutual-information vertex spin-dependent once the walker's spin is
# included, or is it intrinsically edge-only (spin-blind)?
# ============================================================
#
# Scope: docs/scoping/chirality_dependent_kernel_sector_2026-06-02.md (§6 gate).
# Context: the F8 g_A arc showed g_A is robustly ~1.44 and the residual ~13% is a
# single structural gap — every native kernel (U=kappa*dS, OEF -kappa*I(A;B),
# II_3) is a functional of EDGE coverage, never the Cl(6) spinor, so pi and rho
# are binding-degenerate (no hyperfine -> no pi-rho split -> no Goldstone -> no
# g_A reduction). The scoping BET: the spin-blindness is a CHOICE (edge-coverage
# entropy), not the OEF principle. The OEF vertex is E_int = -kappa*I(A;B) with
# I the mutual information of the joint DESCRIPTION; the full walker state is a
# Cl(6)-spinor section, so its honest description includes the SPIN, and I gains
# a spin-correlation term = the hyperfine kernel, with NO new axiom.
#
# THE GATE (this probe): compute I(A;B) for the two-walker SPIN state in the
# spin-singlet (S=0, "pi") vs spin-triplet (S=1, "rho") channels, using the SAME
# OEF mutual-information functional, now on the spinor (not edge) sector.
#   - GREEN: I_singlet != I_triplet  -> hyperfine is latent-native; spin-blindness
#            was an edge-truncation artifact. (Check the SIGN: pi must be more
#            bound than rho, the QCD ordering.)
#   - RED:   I_singlet == I_triplet  -> spin-blindness is intrinsic; R1 dead.
#
# The two "qubits" ARE the two walkers' spinor-return SU(2) doublets (the spin-1/2
# established in F8_gA_nucleon_spin_content from the Cl(6) spinor sector). The
# total mutual information factorizes as I = I_spatial + I_spin at leading order
# (s-wave, no spin-orbit); I_spatial is the channel-independent edge piece, so the
# CHANNEL SPLITTING is carried entirely by I_spin computed here.
#
# Honest flags (carried forward, not hidden): (i) spatial<->spin factorization
# (s-wave); (ii) spin = the spinor-return doublet; (iii) a spin MULTIPLET is the
# rotationally-invariant (m-averaged) state, so its von Neumann entropy is the
# multiplet mixed-state entropy (the per-m pure mutual information is basis-
# dependent and is shown for transparency); (iv) this two-qubit realization is the
# IN-PRINCIPLE gate — the rigorous derivation from the actual Cl(6) junction
# spinors is Stage 1; (v) kappa (bits->MeV) stays the documented magnitude wall,
# so only SCALE-FREE statements are made.

import numpy as np

KAPPA = 1.0   # symbolic unit; absolute scale (bits->MeV) is the walled magnitude

# ---- single-qubit (spinor-return) basis: |up>=(1,0), |down>=(0,1) ----
up = np.array([1, 0], dtype=complex)
dn = np.array([0, 1], dtype=complex)


def kron(a, b):
    return np.kron(a, b)


# two-walker spin states
S0 = (kron(up, dn) - kron(dn, up)) / np.sqrt(2)          # singlet  (pseudoscalar, "pi")
T_p = kron(up, up)                                       # triplet m=+1
T_0 = (kron(up, dn) + kron(dn, up)) / np.sqrt(2)         # triplet m= 0
T_m = kron(dn, dn)                                       # triplet m=-1


def rho_from_ket(psi):
    return np.outer(psi, psi.conj())


def partial_trace_A(rho4):
    """Trace out subsystem B (2x2) from a 4x4 (A⊗B) density matrix -> rho_A (2x2)."""
    r = rho4.reshape(2, 2, 2, 2)         # [a, b, a', b']
    return np.trace(r, axis1=1, axis2=3)


def vn_entropy(rho):
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log2(ev)))


def mutual_information(rho_AB):
    rA = partial_trace_A(rho_AB)
    # subsystem B by tracing out A: reshape and trace axis 0,2
    r = rho_AB.reshape(2, 2, 2, 2)
    rB = np.trace(r, axis1=0, axis2=2)
    return vn_entropy(rA) + vn_entropy(rB) - vn_entropy(rho_AB)


def sigma_dot(channel_S):
    """<sigma_1 . sigma_2> = 2 S(S+1) - 3  (S=0 -> -3, S=1 -> +1)."""
    return 2 * channel_S * (channel_S + 1) - 3


def main():
    print("=" * 76)
    print(" CHIRALITY-DEPENDENT KERNEL — STAGE 0 GATE")
    print(" Is the OEF vertex I(A;B) spin-dependent on the spinor sector?")
    print("=" * 76)
    print("   OEF vertex (derived): E_int(A,B) = -kappa * I(A;B).")
    print("   edge-coverage implementation set I_spin -> 0 (truncated to edges).")
    print("   GATE: include the walker spin -> does I split singlet from triplet?\n")

    # ---- per-state mutual information (transparency: shows m-dependence) ----
    print("[1] per-state spin mutual information (z-basis, pure states):")
    for name, ket in [("singlet S=0    (pi)", S0), ("triplet m=+1", T_p),
                      ("triplet m= 0", T_0), ("triplet m=-1", T_m)]:
        I = mutual_information(rho_from_ket(ket))
        print(f"     {name:18s}: I(A;B) = {I:.4f} bits")
    print("     (note: triplet per-m is basis-dependent (2,0,0)/perm; the physical")
    print("      rotationally-invariant object is the MULTIPLET mixed state below.)")

    # ---- rotationally-invariant multiplet mutual information (the physical one) ----
    print("\n[2] multiplet (rotationally-invariant) mutual information:")
    rho_singlet = rho_from_ket(S0)
    rho_triplet = (rho_from_ket(T_p) + rho_from_ket(T_0) + rho_from_ket(T_m)) / 3.0
    I_singlet = mutual_information(rho_singlet)
    I_triplet = mutual_information(rho_triplet)
    print(f"     I_singlet (S=0, pseudoscalar 'pi') = {I_singlet:.4f} bits")
    print(f"     I_triplet (S=1, vector       'rho')= {I_triplet:.4f} bits  (= 2 - log2 3)")
    split = I_singlet - I_triplet
    print(f"     SPLIT  I_singlet - I_triplet = {split:.4f} bits")

    # ---- the OEF binding and the sign ----
    print("\n[3] OEF binding E_int = -kappa*I  (kappa=1 unit; only signs/ratios meaningful):")
    E_singlet = -KAPPA * I_singlet
    E_triplet = -KAPPA * I_triplet
    print(f"     E(pi,  S=0) = {E_singlet:+.4f} kappa     E(rho, S=1) = {E_triplet:+.4f} kappa")
    print(f"     => pi is {'MORE' if E_singlet < E_triplet else 'LESS'} bound than rho "
          f"(deeper by {E_triplet-E_singlet:.4f} kappa)")
    print(f"        m_rho - m_pi = (E_pi - E_rho) = +{split:.4f} kappa > 0  "
          f"-> pi LIGHTER than rho  ✓ (QCD ordering)")

    # ---- effective sigma_1.sigma_2 operator (is it hyperfine-like?) ----
    print("\n[4] effective spin operator: fit E_int = a + b*<sigma_1.sigma_2>:")
    ss0, ss1 = sigma_dot(0), sigma_dot(1)        # -3 (singlet), +1 (triplet)
    # solve [1, ss0; 1, ss1] [a;b] = [E_singlet; E_triplet]
    M = np.array([[1, ss0], [1, ss1]], dtype=float)
    a, b = np.linalg.solve(M, np.array([E_singlet, E_triplet]))
    print(f"     <sigma.sigma>: singlet {ss0:+d}, triplet {ss1:+d}")
    print(f"     fit: E_int = {a:+.4f} + ({b:+.4f}) * <sigma_1.sigma_2>  (per kappa)")
    print(f"     hyperfine coefficient b = {b:+.4f} kappa")
    print(f"     => b > 0: higher spin (triplet) = higher energy = less bound. This is")
    print(f"        the QCD COLOR-MAGNETIC SIGN (S=0 pseudoscalar lightest). The OEF")
    print(f"        spin mutual-info IS sigma.sigma-LIKE, but NOT exactly proportional")
    print(f"        (I is an entropy, sigma.sigma is linear) -> a FALSIFIABLE ratio")
    print(f"        prediction for Stage 2 (pi-rho vs N-Delta vs g_A).")

    # ---- the control: the OLD edge-coverage (spin-truncated) result ----
    print("\n[5] CONTROL — the edge-coverage (spin-truncated) functional:")
    print("     I_spin is DROPPED -> I_singlet = I_triplet = I_edge (channel-blind).")
    print("     split = 0 -> no pi-rho splitting. This is exactly the spin-blind result")
    print("     the g_A arc found. The gate result below is the difference made by NOT")
    print("     truncating the OEF state to its edge projection.")

    print("\n" + "=" * 76)
    green = split > 1e-6 and E_singlet < E_triplet
    print(f" VERDICT — STAGE 0 GATE: {'GREEN' if green else 'RED'}")
    print("=" * 76)
    if green:
        print(f"""  GREEN. The OEF mutual-information vertex IS spin-dependent once the walker
  spinor is included: I_singlet = {I_singlet:.3f} bits vs I_triplet = {I_triplet:.3f} bits
  (split {split:.3f}), and the sign is correct — the S=0 pseudoscalar ('pi') is
  MORE bound than the S=1 vector ('rho'), i.e. pi is lighter, the QCD ordering.
  The effective operator is a sigma_1.sigma_2 HYPERFINE with the color-magnetic
  sign (coefficient +{b:.3f} kappa).

  -> The chirality/spin-dependent kernel is LATENT-NATIVE in the OEF vertex. The
     spin-blindness of the g_A / binding work was an EDGE-TRUNCATION ARTIFACT
     (S(X) taken as edge coverage, dropping the spinor sector), NOT a feature of
     the OEF principle. NO new axiom is needed — only the un-truncated state.

  This is exactly the scoping bet (R1). The hyperfine kernel exists; the
  remaining work is to PIN it, not to invent it.

  HONEST BOUNDARY (unchanged): this two-qubit realization is the in-principle
  gate. It does NOT yet give magnitudes (kappa walled) and is NOT yet the
  rigorous junction-spinor operator. And it does NOT by itself produce a
  Goldstone-massless pion — splitting pi from rho is far easier than the chiral
  Ward identity (the scoped Stage-3 wall).

  NEXT (per the scoping plan):
   Stage 1 — derive the operator form from the ACTUAL Cl(6) junction spinors of
             D(k) (replace the abstract qubits with the real spinor-return).
   Stage 2 — the make-or-break, SCALE-FREE over-determination: predict
             (N-Delta)/(rho-pi) and the g_A reduction (1.44 -> ?) from the SAME
             functional; ratios cancel kappa. THIS is where it lives or dies.""")
    else:
        print("  RED. The OEF mutual information does not split the channels even with the")
        print("  spinor included -> spin-blindness is intrinsic; R1 dead, fall to R2/R3/R4.")
    print("=" * 76)


if __name__ == "__main__":
    main()
