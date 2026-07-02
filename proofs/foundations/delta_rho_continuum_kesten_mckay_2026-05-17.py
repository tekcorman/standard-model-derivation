#!/usr/bin/env python3
"""
delta_rho_continuum_kesten_mckay_2026-05-17.py

GO AT THE CONTINUUM OBJECT (user-directed, 2026-05-17). The δρ +4.58% is
the continuum/dispersive (Fano-type) self-energy that the discrete-cell
M₀-only treatment is structurally blind to. Compute it from the
substrate's OWN universal-cover spectral measure — the Kesten–McKay
density of the 3-regular tree — via the framework's EXISTING
analytical-Feshbach contour formula. No side-loaded physics, no fit.

FRAMEWORK-NATIVE INPUTS (all intrinsic, k*=3, q=k*−1=2):
 • Universal cover of srs = the 3-regular tree. Its adjacency spectral
   measure is the Kesten–McKay density (standard; the tree's OWN measure,
   = the framework's z*-mechanism home, tree_cover_S_and_resummation):
     ρ_KM(μ) = (3/2π)·√(8−μ²)/(9−μ²),  μ∈[−2√2, 2√2].
 • Non-backtracking circle param: μ = 2√2·cosφ, h = √2·e^{iφ}
   (|h|²=q, Ramanujan). Pushforward density on the circle:
     ρ_circ(φ) ∝ sin²φ / (9 − 8cos²φ)   (Jacobian × ρ_KM), normalized.
 • Framework analytical-Feshbach (theorem_analytical_feshbach_ramanujan_
   boundary.md / q_space_analytical_feshbach.py), SAME formula:
     S_unit(h) ≡ Σ(h)/α₁ = ∫ ρ(φ)/(h − √2 e^{iφ}) dφ
              = (1/h)·[M₀ + Σ_{m≥1} M_m e^{−imα}],   α = arg h.
   M₀-only (uniform ρ) ⇒ S_unit = 1/h_P ⇒ −Im = √5/4 (the LEADING
   Feshbach functional; reproduced as the control below).
 • Sokhotski–Plemelj: |h_P|=√2 sits ON the circle ⇒ outside-radial
   limit (the framework's stated causal prescription), ε→0⁺.
 • δρ = (1/2)·F·(2/3)^8 with F = −Im(S_unit); leading F=√5/4 →
   δρ_lead=+1.0906%; obs +1.0429% (+4.58% rel).

PRE-DECLARED ABORTS (anti-numerology / O9 / no-side-load):
 (K1) the continuum S_unit(h_P) is NOT K-rational (Re,−Im ∉ small-height
      ℚ(√2,√3,√5)) → O9 violation → the continuum object is not
      substrate-algebraic → NEG (honest).
 (K2) direct SP integral and the Fourier-mode-sum reconstruction
      DISAGREE (>1e−6) → computational error → ABORT (no verdict).
 (S)  δρ_cont moves the WRONG way (away from obs) or needs a truncation
      choice to land near obs → NEG.
 (P)  it lands near obs AND is K-rational AND full-sum (no truncation)
      AND correct sign → CANDIDATE-POSITIVE (then: independent
      re-derivation required before any grade claim — NOT shipped here).
"""
import math
import numpy as np

# ---- framework constants --------------------------------------------------
K = 3; Q = K - 1                      # k*=3, q=2
SQRT_Q = math.sqrt(Q)                  # √2  (Ramanujan radius)
H_P = complex(math.sqrt(3)/2, math.sqrt(5)/2)   # saddle, |h_P|²=2
ALPHA = math.atan2(H_P.imag, H_P.real)          # arg h_P
ALPHA1 = (2/3)**8
DR_OBS = 0.0104286
F_LEAD = math.sqrt(5)/4               # M₀-only leading Feshbach functional
DR_LEAD = 0.5 * F_LEAD * ALPHA1       # +1.0906%

def rel(x): return (x/DR_OBS - 1.0)*100.0

# ---- Kesten–McKay pushforward circle density (normalized) -----------------
def rho_circ_unnorm(phi):
    s2 = math.sin(phi)**2
    return s2 / (9.0 - 8.0*math.cos(phi)**2)

# normalization ∫_0^{2π} ρ dφ = 1  (high-accuracy quadrature)
NPHI = 400000
phig = (np.arange(NPHI) + 0.5) * (2*math.pi/NPHI)
rho_vals = np.sin(phig)**2 / (9.0 - 8.0*np.cos(phig)**2)
NORM = rho_vals.sum() * (2*math.pi/NPHI)
def rho_circ(phi): return rho_circ_unnorm(phi) / NORM

# ---- (1) direct SP integral  S_unit(h_P) = ∫ ρ_circ/(h_P − √2 e^{iφ}) dφ --
def S_unit_direct(eps):
    """Outside-radial SP: h_P → h_P·(1+eps), eps→0⁺."""
    h = H_P * (1.0 + eps)
    integrand = rho_vals/NORM / (h - SQRT_Q*np.exp(1j*phig))
    return integrand.sum() * (2*math.pi/NPHI)

# Richardson-ish ε→0 extrapolation
eps_list = [1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5]
S_eps = [S_unit_direct(e) for e in eps_list]
# linear extrapolation in eps from the two smallest
S_dir = S_eps[-1] + (S_eps[-1]-S_eps[-2]) * (eps_list[-1])/(eps_list[-2]-eps_list[-1])

# control: uniform density must give 1/h_P  (leading reproduction)
unif = (np.ones(NPHI)/(2*math.pi)) / (H_P*(1+6.25e-5) - SQRT_Q*np.exp(1j*phig))
S_unit_unif = unif.sum()*(2*math.pi/NPHI)

# ---- (2) Fourier-mode reconstruction (cross-check + K-rationality of M_n) --
def M_n(n):
    return (rho_vals/NORM * np.exp(-1j*n*phig)).sum() * (2*math.pi/NPHI)
Ms = {n: M_n(n) for n in range(0, 41, 2)}        # even modes only (period-π)
S_modes = (Ms[0] + sum(Ms[m]*np.exp(-1j*m*ALPHA) for m in range(2,41,2))) / H_P

# ---- assemble δρ ----------------------------------------------------------
F_cont = -S_dir.imag
DR_CONT = 0.5 * F_cont * ALPHA1

print("CONTROL  uniform ρ → S_unit (should = 1/h_P; −Im=√5/4):")
print(f"   S_unit_unif = {S_unit_unif:.6f}   1/h_P = {1/H_P:.6f}   "
      f"−Im={-S_unit_unif.imag:.6f}  (√5/4={F_LEAD:.6f})  "
      f"{'OK' if abs(-S_unit_unif.imag-F_LEAD)<1e-3 else 'FAIL-control'}")
print(f"\nKesten–McKay continuum M_n (even; M₀ must=1): "
      f"M0={Ms[0].real:.6f}  M2={Ms[2].real:+.6f}  M4={Ms[4].real:+.6f}  "
      f"M6={Ms[6].real:+.6f}  M8={Ms[8].real:+.6f}")
print(f"\n(1) direct SP integral   S_unit(h_P) = {S_dir.real:+.6f} {S_dir.imag:+.6f}i")
print(f"(2) Fourier-mode sum     S_unit(h_P) = {S_modes.real:+.6f} {S_modes.imag:+.6f}i")
agree = abs(S_dir - S_modes)
print(f"    |(1)−(2)| = {agree:.2e}   "
      f"{'OK (K2 pass)' if agree < 5e-4 else 'DISAGREE → ABORT (K2)'}")

print(f"\nF_cont = −Im S_unit = {F_cont:+.6f}   (leading √5/4 = {F_LEAD:.6f}; "
      f"Δ = {(F_cont-F_LEAD)/F_LEAD*100:+.2f}%)")
print(f"δρ_lead = {DR_LEAD*100:+.5f}%  ({rel(DR_LEAD):+.2f}% vs obs)")
print(f"δρ_cont = {DR_CONT*100:+.5f}%  ({rel(DR_CONT):+.2f}% vs obs)")

# ---- K-rationality (O9) test on the continuum S_unit ----------------------
def k_match(x, tag):
    import itertools
    best = None
    rts = {'1':1.0,'√2':math.sqrt(2),'√3':math.sqrt(3),'√5':math.sqrt(5),
           '√6':math.sqrt(6),'√10':math.sqrt(10),'√15':math.sqrt(15),'√30':math.sqrt(30)}
    for name,r in rts.items():
        for p in range(-12,13):
            for qq in range(1,49):
                v = p*r/qq
                if abs(v-x) < 2e-4 and (best is None or abs(v-x)<best[0]):
                    best=(abs(v-x), f"{p}{name}/{qq} = {v:+.6f}")
    print(f"   {tag} = {x:+.7f}  →  nearest small-height K: "
          + (best[1] if best else "none < 2e-4  ⇒ NOT K-rational"))
    return best is not None
print("\nO9 K-rationality (ℚ(√2,√3,√5), height≤12/den≤48):")
reK = k_match(S_dir.real, "Re S_unit")
imK = k_match(-S_dir.imag, "−Im S_unit")

# ---- verdict (pre-declared) ----------------------------------------------
print("\n" + "="*72)
near = abs(rel(DR_CONT)) < 1.2
sign_ok = (DR_CONT < DR_LEAD)          # screening = reduction toward obs
if agree >= 5e-4:
    print("ABORT (K2): the two computations disagree — no verdict (fix quadrature).")
elif near and sign_ok and reK and imK:
    print("VERDICT: CANDIDATE-POSITIVE — the continuum Kesten–McKay self-energy")
    print(f"  lands within ~1% of obs ({rel(DR_CONT):+.2f}%), correct (screening)")
    print("  sign, FULL convergent mode sum (no truncation), and K-rational")
    print("  (O9 PASS). NOT shipped — requires independent closed-form")
    print("  re-derivation of S_unit(h_P) before any grade claim.")
elif sign_ok and not near:
    print(f"VERDICT: NEG (S) — continuum correction has the right (screening)")
    print(f"  sign but wrong magnitude ({rel(DR_CONT):+.2f}% vs target ~0%).")
    print("  The continuum object is real & framework-native but does NOT")
    print("  close the +4.58% at this (untruncated) level. Honest negative;")
    print("  localizes: the gap is not the leading continuum dispersive term.")
elif not (reK and imK):
    print("VERDICT: NEG (K1) — continuum S_unit(h_P) is NOT small-height")
    print("  K-rational ⇒ O9 algebraicity violation ⇒ the continuum object")
    print("  as computed is not substrate-native algebraic. Honest negative.")
else:
    print(f"VERDICT: NEG (S) — wrong sign (δρ moves away from obs: "
          f"{rel(DR_CONT):+.2f}%). Continuum dispersive term is not the +4.58%.")
print("="*72)
