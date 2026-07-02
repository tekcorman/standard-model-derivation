#!/usr/bin/env python3
# ============================================================
# SCALE BRIDGE, binding-ENERGY closure attempt: is binding-energy = κ × (full
# formation description length) PARAMETER-FREE, or does it route into the QCD wall?
# ============================================================
#
# Scope: the runnable-simulation scale bridge. The pin-T probe
# (scale_bridge_pin_T_epoch) found U=κ·ΔS_bare lands at the right ORDER at the
# formation epoch but is low by the Saha log; the proposed closure was
#   binding-energy = κ × (full formation DL),   full DL = bare overlap ΔS ⊕ Saha log,
# with the framework's Boltzmann-Saha bridge + η_B supplying the log. This probe
# tests whether that closure is PREDICTIVE (gives B parameter-free) or CIRCULAR
# (reproduces the Saha condition, B in -> B out). Honest, no tuning.
#
# Uses the REAL framework machinery: bbn_network's deuterium bottleneck (the
# Boltzmann-Saha freeze-out) + the framework's derived η_B.

import os
import sys
import math

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import proofs.cosmology.lib.bbn_network as bbn

LN2 = math.log(2.0)
ETA_B = 6.11e-10          # framework-derived (predictions/eta_B_derivation.md)
DS_BARE = 3               # the 2-body MDL overlap (binding information), this arc
V_HIGGS_MeV = 245.68e3    # framework v (N_hub chain), for the fixed-scale test
T_TODAY_MeV = 2.7255 * 8.617333e-5 * 1e-6   # ~2.35e-10 MeV


def main():
    print("=" * 78)
    print(" SCALE BRIDGE: is binding-ENERGY = κ × full-formation-DL parameter-free?")
    print("=" * 78)

    # real framework Saha bottleneck for the deuteron
    B_D = bbn.B_D_MeV
    T_D = bbn.deuterium_bottleneck_T_MeV(ETA_B)     # solves T = B_D/ln[(mT/2π)^1.5/(η n_γ)]
    L_saha = B_D / T_D                               # nats
    kappa = T_D * LN2                                # κ = k_B T_D ln2  (k_B=1, MeV)
    print(f"\n  deuteron B_D = {B_D:.4f} MeV;  framework bottleneck T_D = {T_D:.5f} MeV")
    print(f"  Saha log L = B_D/T_D = {L_saha:.2f} nats = {L_saha/LN2:.1f} bits "
          f"(= log₂ phase-space dilution; η_B + thermal volume)")

    # ---- (1) the proposed closure: full DL = L/ln2 ----
    print("\n[1] the proposed closure  B = κ × DL_full,  DL_full = L_saha/ln2:")
    DL_full = L_saha / LN2
    B_pred = kappa * DL_full
    print(f"    DL_full = {DL_full:.1f} bits;  κ × DL_full = {B_pred:.4f} MeV  (== B_D)")
    print(f"    -> reproduces B_D EXACTLY. BUT this is CIRCULAR: T_D was SOLVED from")
    print(f"       B_D (the bottleneck condition T_D = B_D/L), so B_D enters on the")
    print(f"       right via T_D. It restates the Saha condition (B = L·k_B·T_form),")
    print(f"       it does NOT predict B. B remains the INPUT.")

    # ---- (2) the bare vertex: off by the Saha log ----
    print("\n[2] the bare vertex  U = κ × ΔS_bare:")
    U_bare = kappa * DS_BARE
    print(f"    U = {U_bare:.4f} MeV;  B_D/U = {B_D/U_bare:.1f}× (= L_saha/(ln2·ΔS) = "
          f"{L_saha/(LN2*DS_BARE):.1f}) -> the Saha-log gap, as before.")

    # ---- (3) can ANY fixed framework-intrinsic T give B parameter-free? ----
    print("\n[3] fixed framework-intrinsic T (no formation-epoch circularity)?")
    for name, T in (("today (N_hub)", T_TODAY_MeV), ("v_Higgs scale", V_HIGGS_MeV)):
        U = T * LN2 * DS_BARE
        print(f"    T = {name:<14} ({T:.3e} MeV): U=κ·ΔS = {U:.3e} MeV "
              f"-> B_D/U = {B_D/U:.2e}×  ({'too small' if U<B_D else 'too big'})")
    print(f"    -> no fixed framework scale gives nuclear B_D (~MeV) from ΔS=3:")
    print(f"       today is ~1e10 too small, v is ~1e5 too big. Only the FORMATION")
    print(f"       epoch lands near B_D — and that is circular (B in -> B out).")

    print("\n" + "=" * 78)
    print(" VERDICT — binding-ENERGY value routes into the QCD/Clause-9 wall")
    print("=" * 78)
    print(f"""  HONEST NEGATIVE on a parameter-free binding ENERGY. The proposed closure
  B = κ × (full formation DL) is CIRCULAR: with DL_full = L_saha/ln2 it reproduces
  B_D exactly, but only because the formation temperature T_D was itself SOLVED
  from B_D (the Saha bottleneck T_D = B_D/L). It is the Saha condition in OEF
  clothing — consistent, but it takes B as INPUT and returns B; it predicts nothing.

  And no FIXED framework-intrinsic temperature rescues it: 'today' undershoots by
  ~1e10, the v_Higgs scale overshoots by ~1e5, and only the formation epoch lands
  near B_D (circularly). So κ·ΔS cannot deliver the nuclear MeV value.

  WHAT THIS LOCATES (the honest boundary): the framework natively supplies
   * the binding INFORMATION — ΔS (the dimensionless 2-/n-body MDL vertex), and
   * the SCALE LEDGER — one dimensional input N_hub (the binding adds none),
  but NOT the binding-ENERGY VALUE. The MeV/eV magnitude is set by the constituent
  COUPLING scale (nuclear/QCD for B_D), which is the documented Clause-9 / gauge-
  running-keystone wall — the SAME wall as Q_np's QCD part and the g_A 0.76
  reduction (this session). The binding-energy value is QCD-external, by the same
  boundary that walls the rest of the baryon sector.

  NET (the scale bridge, fully resolved): a dimensionful runnable simulation needs
  NO new scale (one input N_hub + the F9 T-identification — established). The
  binding INFORMATION and the formation SCALE are native and self-consistent with
  the framework's Saha structure. But specific binding-ENERGY VALUES (B_D, 13.6 eV,
  …) are NOT parameter-free here — they require the constituent-coupling (QCD/EM)
  dynamics, which is Clause-9-walled. The scale-free predictions stand (binding
  quantized in κ; ratios = integer ΔS ratios); the absolute MeV values do not.""")
    print("=" * 78)


if __name__ == "__main__":
    main()
