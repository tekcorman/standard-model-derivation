#!/usr/bin/env python3
# ============================================================
# HORN (iii): is the observer's register-MDL running a GENUINELY
# different function from the QFT beta-function — one that escapes
# the trilemma — or does it coincide with standard RG where we have data?
# ============================================================
#
# Scope: internal research notes (trilemma; horn iii
# is "is standard RG the right N->scale map?" — where the d/dN thesis lives or dies).
#
# THE THESIS'S LAST HOPE: wedge-3's Landau pole assumed register-MDL running ==
# standard logarithmic RG (linear-in-log, slope = mode count). If the observer's
# register-MDL running is a DIFFERENT function (e.g. bounded/saturating MDL-
# probability growth), the 2HDM Landau pole might be an artifact, and the trilemma
# could dissolve without the +4.
#
# THE TEST that settles it: the gauge couplings' running is MEASURED. Confront the
# thesis with data. At 1-loop, register-MDL running and QFT beta COINCIDE (both
# linear-in-log, slope = active-mode count) — that is why they look alike. The
# question is whether there is any ROOM for them to differ WHERE WE HAVE DATA.
#
# Active-mode count at observed scales = the SM/2HDM content (no sparticles seen);
# register-MDL b_3 = -7. We run the measured couplings up and ask: (1) does the
# substrate's own 2HDM content reproduce the OBSERVED running (yes -> no room for a
# novel running law at observed scales); (2) does it deliver the framework's
# structural unification alpha_GUT^-1 = 24 (no -> that is MSSM-only).

import math

# observed 1/alpha_i(M_Z), GUT-normalized (PDG-ish)
INV_A1_MZ = 59.0      # U(1)_Y (GUT-normalized, x5/3)
INV_A2_MZ = 29.6      # SU(2)_L
INV_A3_MZ = 8.47      # SU(3)_c  (alpha_s ~ 0.118)
M_Z = 91.1876
M_GUT = 2.0e16
ALPHA_GUT_INV_STRUCT = 24.0   # framework structural value = 2^k* . k*

# one-loop b_i, GUT-normalized. Convention: 1/alpha(M_high) = 1/alpha(M_Z) - (b/2pi) ln(M_high/M_Z)
# SM / 2HDM (the SUBSTRATE's own content; = register-MDL active-mode count):
B_SM   = (41/10, -19/6, -7.0)     # (b1, b2, b3)
# 2HDM (two Higgs doublets) — same b3, slightly different b1,b2:
B_2HDM = (21/5,  -3.0,  -7.0)
# MSSM (the ADOPTED +4 content):
B_MSSM = (33/5,   1.0,  -3.0)


def run_up(inv_a_mz, b):
    return inv_a_mz - (b / (2 * math.pi)) * math.log(M_GUT / M_Z)


def main():
    print("=" * 72)
    print("HORN (iii): is register-MDL running a different function from QFT beta?")
    print("Confront it with the MEASURED gauge-coupling running.")
    print("=" * 72)

    L = math.log(M_GUT / M_Z)
    print(f"\nln(M_GUT/M_Z) = {L:.2f}; observed 1/alpha_i(M_Z) = "
          f"({INV_A1_MZ}, {INV_A2_MZ}, {INV_A3_MZ})")

    print("\n[1] 1-loop coincidence: register-MDL running (linear-in-log, slope =")
    print("    active-mode count) IS standard RG at 1-loop. At OBSERVED scales the")
    print("    active modes = SM content (no sparticles seen) -> register-MDL b_3 = -7")
    print("    = the SM/2HDM value = what is MEASURED. So where we have data, the")
    print("    register-MDL running and QFT beta are the SAME function. No room for a")
    print("    'different running law' at observed scales.")

    print("\n[2] Run the measured couplings UP to M_GUT under each content:")
    for name, b in (("SM/2HDM (substrate = register-MDL)", B_SM),
                    ("2HDM", B_2HDM),
                    ("MSSM (adopted, +4)", B_MSSM)):
        vals = [run_up(INV_A1_MZ, b[0]), run_up(INV_A2_MZ, b[1]), run_up(INV_A3_MZ, b[2])]
        spread = max(vals) - min(vals)
        unify = spread < 1.5
        print(f"    {name:36}: 1/alpha_i(M_GUT) = "
              f"({vals[0]:.1f}, {vals[1]:.1f}, {vals[2]:.1f})  "
              f"spread {spread:.1f}  {'UNIFY ~' + str(round(sum(vals)/3,1)) if unify else 'NO unification'}")

    print(f"\n    -> MSSM unifies at ~24 (= the framework's structural 2^k*.k* = 24).")
    print(f"       SM/2HDM (the substrate's OWN content) does NOT unify and lands far")
    print(f"       from 24. The famous SM non-unification.")

    print("\n[3] So where is the inconsistency, really?")
    print("    - The RUNNING LAW is not the problem: register-MDL = 2HDM = the OBSERVED")
    print("      running at the scales we measure. Horn (iii)'s hope of a novel running")
    print("      law that escapes the trilemma finds NO room where we have data.")
    print("    - The load-bearing claim is the framework's STRUCTURAL unification")
    print("      alpha_GUT^-1 = 24, which holds ONLY under MSSM (+4) extrapolation in")
    print("      the UNOBSERVED TeV->GUT range — content the substrate (2HDM) lacks.")

    print("\n" + "=" * 72)
    print("VERDICT — horn (iii): CLOSED as an escape; the keystone is the unification")
    print("=" * 72)
    print(f"""  Horn (iii) does NOT yield a register-MDL running law that dissolves the
  trilemma. At every scale we can MEASURE, the register-MDL running coincides
  with standard RG using the substrate's own 2HDM content — which is exactly
  what is observed. There is no room for a 'different running law' where we have
  data; the 1-loop coincidence is forced by the data, not assumed.

  So the trilemma does not dissolve via the running; it SHARPENS to its real
  keystone — the framework's flagship structural result alpha_GUT^-1 = 24
  (= 2^k* . k*). That value is the MSSM unification point. The substrate's own
  content (2HDM) reproduces the observed low-energy running but does NOT unify at
  24 (SM non-unification). So the headline gauge unification is CONDITIONAL on the
  adopted MSSM (+4) — it is the framework's single most load-bearing adoption, and
  it is one of the four Paper-0 headline predictions (alpha_GUT^-1 = 24).

  HORN (iii)'s sole surviving redoubt: register-MDL running could differ from
  standard RG only in the UNOBSERVED TeV->GUT range, unifying at 24 with 2HDM
  content via a register-native running coefficient. That is exactly the
  recurrence-count program — bounded, register-MDL, and already stalled on the
  +4. It is testable but not open-ended.

  HONEST BOTTOM OF THE RABBIT HOLE: the d/dN diagnosis is correct (off predictions
  = missing dynamics), but the thesis's boldest escape (a genuinely novel observer
  running law) does not survive the measured running. The framework's dynamical
  inconsistency is real; its keystone is the alpha_GUT^-1 = 24 unification claim,
  which rests on the adopted MSSM +4 (closed-negative in the substrate). The
  flagship gauge unification is the load-bearing adoption — that, not any single
  off prediction, is what the whole constellation has been pointing at.""")
    print("=" * 72)


if __name__ == "__main__":
    main()
