#!/usr/bin/env python3
# ============================================================
# SCALE BRIDGE, numeric closure attempt: pin the OEF temperature T via the F9
# epoch-T(N) identification, then compute the binding U = kappa*dS in physical
# units and test against real bound states (deuteron, hydrogen).
# ============================================================
#
# Scope: the runnable-simulation scale bridge. scale_bridge_binding_2026-06-01.py
# showed binding costs NO new dimensional input (it inherits the single input
# N_hub); the one open piece was the NAMED IDENTIFICATION of the OEF temperature
# T (kappa = k_B T ln2; uncalibrated in theorem_observer_energy_functional) with
# its N_hub-chain value. The user's directive: pin T via the F9 epoch-T(N).
#
# THE IDENTIFICATION:
#   F9 / freeze_out_thermal_trajectory: the radiation-era temperature falls as
#   T(N) = T_anchor * sqrt(N_anchor / N)   (thermal energy per horizon-area ~ 1/N²,
#   horizon area ~ (c N t_P)²). Anchor at today: T_today, N = N_hub.
#   OEF: kappa(N) = k_B T(N) ln2  ->  binding U(N) = kappa(N) * dS.
#
# THE TEST (honest, two evaluations because the EPOCH matters):
#   (a) T = T_today (observer = now): does U = kappa*dS reproduce a binding energy?
#   (b) T = T_formation (the bound state forms when the bath cools to ~ its scale,
#       the framework's own Saha/deuterium-bottleneck epoch): does it then?
#   Real targets: deuteron (B=2.224 MeV, forms at BBN) and hydrogen (B=13.6 eV,
#   forms at recombination). dS_bare = the 2-body MDL overlap = 3 bits.
#
# DISCIPLINE: this is a closure ATTEMPT, reported honestly whether it lands or
# walls. No tuning of dS or T to hit the answer.

import math

K_B_eV_per_K = 8.617333e-5     # Boltzmann constant, eV/K
LN2 = math.log(2.0)

# --- F9 epoch-T(N): T(N) = T_today * sqrt(N_hub / N) ---
T_TODAY_K = 2.7255
T_TODAY_eV = T_TODAY_K * K_B_eV_per_K       # ~2.35e-4 eV
N_HUB = 8.394881e60


def T_of_N(N):
    return T_TODAY_eV * math.sqrt(N_HUB / N)   # eV


def N_of_T(T_eV):
    return N_HUB * (T_TODAY_eV / T_eV) ** 2     # invert


def kappa(T_eV):
    return T_eV * LN2      # kappa = k_B T ln2; T already carried in eV (k_B absorbed)


DS_BARE = 3        # the 2-body MDL overlap (bits), from the interaction-layer arc


def main():
    print("=" * 78)
    print(" SCALE BRIDGE numeric: pin T via F9 epoch-T(N), test U=κ·ΔS vs binding")
    print("=" * 78)
    print(f"\n  T(N) = T_today·√(N_hub/N),  T_today = {T_TODAY_eV:.3e} eV ({T_TODAY_K} K),")
    print(f"  N_hub = {N_HUB:.3e};  κ(N) = k_B T(N) ln2;  U = κ·ΔS_bare, ΔS_bare = {DS_BARE}.")

    # ---- (a) T = today ----
    print("\n[a] observer = TODAY (N = N_hub):")
    k_today = kappa(T_TODAY_eV)
    U_today = k_today * DS_BARE
    print(f"    κ_today = {k_today:.3e} eV/bit;  U = κ·ΔS = {U_today:.3e} eV")
    print(f"    vs deuteron 2.224e6 eV: off by {2.224e6/U_today:.1e}×  -> ABSURD.")
    print(f"    => the LITERAL 'T = now' reading fails: binding would be ~1e-4 eV.")
    print(f"       Binding cannot be the present-epoch temperature (it is fixed, not")
    print(f"       falling as the universe cools). The epoch must be FORMATION.")

    # ---- (b) T = formation epoch (Saha/bottleneck), for two real bound states ----
    print("\n[b] observer = FORMATION epoch (bath cools to ~ the binding scale):")
    print(f"    {'state':<10}{'B (eV)':>11}{'T_form (eV)':>13}{'N_form':>11}"
          f"{'U=κ·ΔS':>11}{'B/U':>8}")
    cases = [
        # name, B (eV), T_formation (eV) [Saha bottleneck], Saha log L = B/T_form
        ("deuteron", 2.224e6, 2.224e6 / 26.0),   # BBN deuterium bottleneck, log~26
        ("hydrogen", 13.6,    13.6 / 42.0),       # recombination, log~42
    ]
    for name, B, T_form in cases:
        Nf = N_of_T(T_form)
        U = kappa(T_form) * DS_BARE
        print(f"    {name:<10}{B:>11.3e}{T_form:>13.3e}{Nf:>11.2e}{U:>11.3e}{B/U:>8.1f}")
    print(f"    => RIGHT ORDER OF MAGNITUDE at formation, but systematically LOW by")
    print(f"       ~12-20× (deuteron 12.5×, hydrogen 20×).")

    # ---- the diagnosis: the gap IS the Saha log / (ln2·ΔS) ----
    print("\n[c] the gap is the Saha log, not noise:")
    print(f"    U = κ·ΔS = (k_B T_form ln2)·ΔS, and T_form = B / L_Saha (the formation")
    print(f"    condition). So B/U = L_Saha / (ln2·ΔS_bare).")
    for name, B, T_form in cases:
        L = B / T_form
        print(f"      {name}: L_Saha = {L:.0f},  ln2·ΔS = {LN2*DS_BARE:.2f}"
              f"  ->  B/U = {L/(LN2*DS_BARE):.1f}  (matches column above)")
    DS_needed = 26.0 / LN2
    print(f"    To close exactly: ΔS = L_Saha/ln2 ~ {DS_needed:.0f} bits (deuteron), NOT 3.")
    print(f"    i.e. the binding ENERGY needs the FULL formation description length")
    print(f"    (incl. the phase-space/η dilution = the Saha log), which the framework")
    print(f"    HAS (theorem_mdl_boltzmann_saha_bridge), but which is NOT the bare")
    print(f"    2-walker overlap ΔS=3 (that is the binding INFORMATION, not the energy).")

    print("\n" + "=" * 78)
    print(" VERDICT — T pinned; binding SCALE right at formation; bare-ΔS map walls")
    print("=" * 78)
    print(f"""  HONEST PARTIAL. The F9 epoch-T(N) identification DOES pin T (κ = k_B T(N) ln2,
  T(N)=T_today√(N_hub/N), anchored on N_hub — no new scale, as the scale-ledger
  analysis required). And it makes a real, non-trivial statement: a bound state's
  energy scale is the temperature of the epoch at which it forms — which is the
  framework's OWN Saha/bottleneck condition (a composite attests when the bath
  cools to ~ its binding scale; cf. the minimal-assembly attestation waterline).

  BUT the clean numeric map U = κ·ΔS_bare does NOT close:
   * 'T = today' fails by ~1e10 (binding is fixed; the epoch temperature falls).
     So T must be the FORMATION epoch, not the present — an honest constraint on
     the identification (T is local-to-formation, not global).
   * At formation, U = κ·ΔS_bare lands at the RIGHT ORDER but systematically LOW
     by the Saha log: B/U = L_Saha/(ln2·ΔS) ~ 12.5× (deuteron), 20× (hydrogen).
     The gap is structural (the Saha freeze-out log ~26-42), the SAME object the
     framework already derives — closing it needs the FULL formation description
     length (~ L_Saha/ln2 ~ 38 bits), not the bare overlap ΔS=3.

  READING: the bare 2-walker overlap ΔS=3 is the binding INFORMATION (correct,
  dimensionless, from the vertex); the binding ENERGY in physical units = κ ×
  (full formation description length), and the difference is exactly the Saha
  phase-space/η log the Boltzmann-Saha bridge supplies. So the numeric scale
  bridge is reachable but routes through the Saha log, NOT through κ·ΔS_bare.

  NET: T is pinned via F9 epoch-T(N) (scale ledger stays at one input N_hub); the
  binding-energy magnitude comes out at the right scale at formation and is tied
  to the framework's Saha structure; but the simple κ·ΔS_bare = binding-energy
  identity is OFF by the Saha log (~12-20×, systematic, sector-dependent). The
  honest closure target is binding-energy = κ × full-formation-DL (= bare overlap
  ⊕ the Saha log), a concrete next step, not the one-line vertex value.

  CAVEAT: T_form uses the standard Saha logs (26 deuteron / 42 hydrogen) as inputs
  here; deriving those from the framework's own bottleneck (Boltzmann-Saha bridge +
  η_B, both in-framework) is what would make this fully parameter-free.""")
    print("=" * 78)


if __name__ == "__main__":
    main()
