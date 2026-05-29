#!/usr/bin/env python3
"""
proofs/foundations/nu_mass_half_sided_persistence_2026-05-19.py

ON-TASK probe (neutrino MASS scale). Tests whether a persistence model
DERIVES the disclosed dominant adoption y_ν = 1 in m_ν3 = y_ν²·v²/M_R.

*** SMUGGLE CORRECTION (user, mid-probe) ***
The first version enumerated "½ per-chirality occupancy / ½ amplitude"
readings. Those SMUGGLE a per-chirality probability. Per the framework's
observer-MDL posture (theorem_observer_persistence_closure_IC_amplitude.md
/ P1'): physical observables are functionals of the OBSERVER's compressed
model; the observer retains the TOTAL persistence of a structure, NOT
which chirality it occupies instant-to-instant (that is high-DL detail
the compressed model discards). A per-chirality occupancy is therefore a
quantity the observer does not compress — injecting it is a smuggled
parameter (parameter_linter cardinal violation). Those readings are VOID
by construction, not "value-falsified".

VALID object only: the TOTAL probability of persistence of the
observer-compressed neutrino mass-bearing structure. No chirality split.

CORRECTNESS GATE (VOID if fail): at y_ν=1 the chain reproduces the LIVE
predictions/m_nu3.py value (≈50.565 meV, +0.87%, +2.18σ).

PRE-DECLARED (no value-fishing; this is a methodology + single-object
probe, NOT an enumeration):
  RE-POSED   : the only non-smuggling reading is y_ν = total
               observer-compressed persistence. The framework's OWN
               result "m_ν3 is δ-INDEPENDENT — unlike the δ-Koide
               charged leptons" says the observer compresses the neutrino
               as the unique hierarchy-FREE mode (no compressible
               decoration). Total persistence of a hierarchy-free
               structure = the un-suppressed natural amplitude ⇒ y_ν=1 is
               re-posed as "the total observer-compressed persistence of
               the δ-independent fermion" — sharper than a hand-wave, but
               NOT yet derived (forcing total-persistence=1 needs the
               observer-MDL/P1' theorem applied to the ν structure —
               NAMED, NOT done here). +0.87% stays N_hub↔G_F-gated.
  DERIVED    : ONLY if a non-smuggling computation forces y_ν exactly
               (=1 or =0.9957) from the observer-MDL machinery — not
               attempted here; would be a separate gated probe.
Ships no number into predictions/; changes no ledger row.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'predictions'))
import numpy as np, subprocess
from k_star import predict_k_star
from g_girth import predict_g_girth
from d_spatial import predict_d_spatial

d=predict_d_spatial(); k=predict_k_star(d); g=predict_g_girth(k,d)
alpha1=((k-1)/k)**(g-2)
print(f"k*={k} g={g}  master-mass total survival/persistence α₁=(2/3)^{g-2}={alpha1:.6f}")
print("(α₁ is a TOTAL persistence probability — NO chirality occupancy in it.)")

REPO=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..')
out=subprocess.run([sys.executable,"predictions/m_nu3.py"],capture_output=True,
                   text=True,cwd=REPO)
base=next(float(l.split("=")[1].split("meV")[0]) for l in out.stdout.splitlines()
          if "m_ν₃ predicted" in l)
OBS,SIG=50.1298,0.1995

print("\n"+"="*70+"\nCORRECTNESS GATE — live m_nu3.py base (y_ν=1)\n"+"="*70)
gate=abs(base-50.565)<0.05
print(f"  base={base:.4f} meV  reproduces 50.565 (+0.87%,+2.18σ)? {gate}")
if not gate:
    print("  ** GATE FAILED. VOID. **"); sys.exit(0)
print("  GATE PASSED.")

print("\n"+"="*70+"\nSMUGGLE FLAG — readings that inject a non-compressed quantity\n"+"="*70)
for tag in ["½ per-chirality time-occupancy (y_ν²=½)",
            "symmetric ½ chirality amplitude (y_ν=1/√2)",
            "y_ν = survival amplitude α₁ / √α₁"]:
    print(f"  VOID (smuggled): {tag}")
    print("     -> injects a per-chirality / per-instant probability the")
    print("        observer-MDL model does NOT compress. Not a value test.")

print("\n"+"="*70+"\nVALID object — total observer-compressed persistence\n"+"="*70)
print("  Framework's own m_ν3 derivation: m_ν3 is δ-INDEPENDENT — 'a clean")
print("  structural distinction between neutrinos and charged leptons (the")
print("  latter carry δ-dependent Koide hierarchies)'. ⇒ the observer")
print("  compresses the neutrino as the unique hierarchy-FREE mode: no")
print("  compressible decoration ⇒ TOTAL persistence = the un-suppressed")
print(f"  natural amplitude ⇒ y_ν = 1  ⇒ m_ν3 = {base:.4f} meV "
      f"[+0.87%, +2.18σ vs NuFIT {OBS}].")
print(f"  Data would require y_ν=√(obs/base)={np.sqrt(OBS/base):.5f} "
      f"(0.43% near-unity deficit — within the N_hub anchor band, NOT a")
print("  persistence effect).")

print("\n"+"="*70+"\n  VERDICT (pre-declared: RE-POSED, not DERIVED)\n"+"="*70)
print("""  The per-chirality readings are VOID (smuggled — user-caught). The
  only non-smuggling reading is y_ν = total observer-compressed
  persistence; the framework's δ-independence of m_ν3 RE-POSES the
  adopted y_ν=1 as 'the total persistence of the unique hierarchy-free
  observer-compressed fermion' — sharper than the hand-wave, but NOT a
  derivation: forcing total-persistence=1 requires the observer-MDL/P1'
  theorem (theorem_observer_persistence_closure_IC_amplitude.md /
  theorem_p1_prime_derived_from_a1.md) applied to the ν mass structure
  — a NAMED, separate, gated step, NOT done here. The half-sided-
  persistence model does NOT derive a sub-unity y_ν and does NOT close
  the scale. +0.87%/+2.18σ remains gated by the N_hub↔G_F circularity,
  untouched. No status change; nothing shipped.""")
print("="*70)
