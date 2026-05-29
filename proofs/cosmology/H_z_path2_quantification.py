#!/usr/bin/env python3
"""
proofs/cosmology/H_z_path2_quantification.py

PATH 2 STAGE 1.0: Quantify the H(z) shape Pantheon+ data demands.

QUESTION
--------
Pure coasting (cascade theorem H = 1/(N · t_P), N(t) = t/t_P) gives
H(z) = H_0·(1+z) at all redshifts. Pantheon+ SNe Ia data is fit by
ΛCDM with H(z) = H_0·√(Ω_m(1+z)³ + Ω_Λ). These have different SHAPE,
not just different H_0.

Path 1 fixed the SH0ES low-z H_0 question (partial closure ~1.4σ).
Path 2 must address the residual ~3σ AND the high-z shape mismatch
flagged by `coasting_sn1a_comparison.py`.

This script computes:
1. ΛCDM "truth" H(z) at H_0_LCDM = 73.04, Ω_m = 0.315 (Pantheon+ best-fit)
2. Coasting H(z) at framework H_0 = 68.19 and at coast-best-fit H_0 = 71.5
3. f(z) ≡ H_coast(z) / H_LCDM(z) — the ratio path 2 must produce structurally
4. Δμ(z) per redshift bin — the systematic residual after best-fit H_0
5. Magnitude of "effective f(z)" needed: what structural correction f(z)
   to N(t) = (t/t_P) · f(t) would close the shape gap?

RESULT (preview)
----------------
- f(z) is bell-shaped: 1.000 at z=0, peaks at ~1.15 around z=0.5,
  declines past z=1
- Pure coasting + constant rescaling (DHS) cannot reproduce the shape;
  it shifts the curve uniformly
- Pure coasting + branch-mean inhomogeneity (BMI) with z-growing
  variance has the right qualitative behavior but needs σ_N(z) to
  grow then plateau — non-trivial shape constraint
- Best candidate: cascade-theorem refinement giving N(t) = (t/t_P)·f(t)
  with f(t) > 1 at intermediate t (branch proliferation factor)
"""

import math
from scipy import integrate, optimize


# --- constants ---
c_km_s = 2.99792458e5


def H_LCDM(z, H0, Om=0.315, OL=0.685):
    """ΛCDM Hubble rate."""
    return H0 * math.sqrt(Om * (1.0 + z)**3 + OL)


def H_coast(z, H0):
    """Coasting Hubble rate (a ∝ t)."""
    return H0 * (1.0 + z)


def d_L(z, H_func, **kwargs):
    """Luminosity distance from Hubble rate function."""
    if z <= 0:
        return 0.0
    integrand = lambda zp: 1.0 / H_func(zp, **kwargs)
    chi, _ = integrate.quad(integrand, 0.0, z, epsabs=1e-12, epsrel=1e-12)
    return c_km_s * (1.0 + z) * chi


def mu(d_L_Mpc):
    return 5.0 * math.log10(d_L_Mpc) + 25.0


# --- truth model ---
H0_LCDM = 73.04
Om_LCDM = 0.315


# --- framework prediction ---
H0_coast_framework = 68.19


# --- z bins of interest ---
z_bins = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00]


# -----------------------------------------------------------------------
# 1. f(z) ≡ H_coast(z) / H_LCDM(z) at SAME H_0
# -----------------------------------------------------------------------
# This is the structural shape correction path 2 must produce.
# (At equal H_0 to isolate shape from amplitude.)

H0_common = 73.0  # arbitrary pivot for ratio comparison
print("=" * 72)
print("  PATH 2 STAGE 1.0: H(z) data demand quantification")
print("=" * 72)
print()
print(f"  Truth model:   ΛCDM, Ω_m = {Om_LCDM}, H_0 = {H0_LCDM} km/s/Mpc")
print(f"  Framework:     coasting (a ∝ t), H_0 = {H0_coast_framework} km/s/Mpc")
print()
print("--- 1. Shape ratio f(z) = H_coast(z) / H_LCDM(z), same H_0 ---")
print(f"  (At same H_0 = {H0_common}, isolates shape)")
print()
print(f"  {'z':>5}  {'H_coast':>9}  {'H_LCDM':>9}  {'f(z) = ratio':>14}")
print("  " + "-" * 50)
f_values = []
for z in z_bins:
    Hc = H_coast(z, H0_common)
    Hl = H_LCDM(z, H0_common, Om_LCDM, 1 - Om_LCDM)
    f  = Hc / Hl
    f_values.append((z, f))
    print(f"  {z:>5.2f}  {Hc:>9.2f}  {Hl:>9.2f}  {f:>14.4f}")

# Find the peak of f(z)
def neg_f(z):
    return -H_coast(z, H0_common) / H_LCDM(z, H0_common, Om_LCDM, 1 - Om_LCDM)
res = optimize.minimize_scalar(neg_f, bounds=(0.01, 3.0), method='bounded')
z_peak = res.x
f_peak = -res.fun
print()
print(f"  Peak of f(z): f({z_peak:.3f}) = {f_peak:.4f}")
print(f"    Δf = f_peak - 1 = {f_peak - 1.0:+.4f} = {(f_peak-1.0)*100:+.1f}%")


# -----------------------------------------------------------------------
# 2. Δμ residual at fixed (best-fit) coasting H_0
# -----------------------------------------------------------------------
print()
print("--- 2. Per-bin Δμ residual after best-fit coasting H_0 ---")
print("  (μ_coast(z, H_best) - μ_LCDM(z, 73.04))")
print()

# Best-fit coasting H_0 to match ΛCDM truth, on Pantheon+ z range 0.05-1.5
z_panth = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50]
mu_truth = [mu(d_L(z, H_LCDM, H0=H0_LCDM, Om=Om_LCDM, OL=1-Om_LCDM)) for z in z_panth]

def chi2_panth(H_c, weights=None):
    if weights is None:
        weights = [1.0] * len(z_panth)
    return sum(weights[i] * (mu(d_L(z, H_coast, H0=H_c)) - mu_truth[i])**2
               for i, z in enumerate(z_panth))

# Several weightings to bracket realistic Pantheon+ leverage:
weightings = {
    'uniform-z (toy)':      [1.0] * len(z_panth),
    'low-z dominant (SH0ES-like)':
                            [10.0, 8.0, 5.0, 3.0, 1.5, 0.8, 0.4, 0.2, 0.1],
    'realistic Pantheon+ (low-z heavy, tail to z~1.5)':
                            [4.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.6, 0.2],
    'high-z dominant (volume z²)':
                            [z**2 for z in z_panth],
}

print(f"  Best-fit coasting H_0 across Pantheon+ z range:")
for label, w in weightings.items():
    res = optimize.minimize_scalar(lambda H: chi2_panth(H, w),
                                    bounds=(50, 95), method='bounded',
                                    options={'xatol': 1e-5})
    print(f"    {label:55} → H_0 = {res.x:.2f} km/s/Mpc")
H0_best_panth = optimize.minimize_scalar(
    lambda H: chi2_panth(H, weightings['realistic Pantheon+ (low-z heavy, tail to z~1.5)']),
    bounds=(50, 95), method='bounded', options={'xatol': 1e-5}).x
print(f"  Path 1 (SH0ES range only):                       71.56 km/s/Mpc")
print()
print(f"  Per-bin residual Δμ at best-fit coasting (Pantheon+ range):")
print(f"  {'z':>5}  {'μ_LCDM':>8}  {'μ_coast':>8}  {'Δμ':>8}  significance")
print("  " + "-" * 55)
sys_floor = 0.06  # mag
for z in z_panth:
    m_t = mu(d_L(z, H_LCDM, H0=H0_LCDM, Om=Om_LCDM, OL=1-Om_LCDM))
    m_c = mu(d_L(z, H_coast, H0=H0_best_panth))
    dmu = m_c - m_t
    sig = abs(dmu) / sys_floor
    flag = " **" if abs(dmu) > sys_floor else ""
    print(f"  {z:>5.2f}  {m_t:>8.3f}  {m_c:>8.3f}  {dmu:>+8.4f}  {sig:>4.1f}σ_sys{flag}")


# -----------------------------------------------------------------------
# 3. What structural correction f(t) to N(t) = (t/t_P)·f(t) reproduces ΛCDM?
# -----------------------------------------------------------------------
print()
print("--- 3. Structural correction f(t) interpretation ---")
print("  If H_framework(z) = (1/t)·1/f(t) instead of 1/t (pure coasting),")
print("  and we want H_framework(z) = H_LCDM(z), then:")
print("      f(t) = H_coast(z) / H_LCDM(z)  [at same H_0]")
print()
print("  i.e., f(t) is the inverse of the matter contribution to the")
print("  effective expansion rate.")
print()
print(f"  Numerically: f peaks at z ≈ {z_peak:.2f} with Δf = {(f_peak-1)*100:+.1f}%")
print(f"  This is a small (~{(f_peak-1)*100:.0f}%) positive bump in f(t),")
print("  not a constant rescaling.")
print()
print("  Implications for mechanism candidates:")
print("    DHS (constant ξ):           CANNOT produce z-dependent f")
print("    BMI (variance Var(N)):      f = 1 + Var(N)/⟨N⟩², need Var(N)/⟨N⟩²")
print(f"                                ≈ {f_peak-1:.3f} at z = {z_peak:.2f},")
print(f"                                vs ≈ 0 at z = 0 → variance grows with z")
print("    Branch-proliferation f(t):  natural shape match if f(t) tracks")
print("                                multi-tick observable count")


# -----------------------------------------------------------------------
# 4. Quantitative magnitude budget for path 2
# -----------------------------------------------------------------------
print()
print("--- 4. Path-2 magnitude budget ---")
print()
print("  SH0ES H_0 question:")
print(f"    Need H_local(z=0) ≈ {73.0/H0_coast_framework:.4f} × H_global(z=0)")
print(f"      = (1 + {(73.0/H0_coast_framework - 1)*100:.1f}%) × H_framework")
print(f"    Mechanism: ξ_local ≈ 0.93 in DHS terms")
print()
print("  High-z shape question:")
print(f"    Need f_max(z={z_peak:.2f}) ≈ {f_peak:.4f}")
print(f"      = (1 + {(f_peak-1)*100:.1f}%) bump over coasting")
print(f"    Mechanism: z-dependent BMI variance, OR f(t) cascade correction")
print()
print("  Residual after path 2 if both addressed:")
print("    SH0ES tension:    closes")
print("    High-z SN shape:  closes within Pantheon+ systematic floor")
print()
print("  Residual if only DHS (constant ξ):")
print("    SH0ES tension:    closes")
print("    High-z SN shape:  shape mismatch persists (DHS doesn't fix shape)")
print()
print("  This suggests DHS is INSUFFICIENT alone. BMI or branch-proliferation")
print("  is required for the high-z piece.")


# -----------------------------------------------------------------------
# 5. Are SH0ES and high-z gaps the SAME phenomenon or DECOUPLED?
# -----------------------------------------------------------------------
print()
print("--- 5. Are the two gaps the same phenomenon? ---")
print()
print(f"  At z → 0: f(z) → 1 (no shape correction needed)")
print(f"  At z = 0: SH0ES needs ~7% boost in measured H_0")
print()
print("  These don't naturally couple via a single shape function f(z).")
print("  f(z=0) = 1 means no z-dependent mechanism boosts H_0 locally.")
print()
print("  Resolution candidates:")
print("    (a) SH0ES has a true distance-ladder calibration systematic")
print("        (framework H_0 = 68 is correct globally AND locally)")
print("    (b) The 'local boost' is observer-specific (DHS), distinct from")
print("        the high-z shape correction (BMI or f(t))")
print("    (c) Both effects come from the same multiway-branch ensemble,")
print("        with a non-trivial scale-dependence we haven't identified")
print()
print("  STRATEGIC IMPLICATION: SH0ES residual and high-z SN tension may")
print("  need to be addressed by DIFFERENT mechanisms, OR (a) absorbs the")
print("  SH0ES residual entirely and only the high-z piece is structural.")


# -----------------------------------------------------------------------
# 6. Search for structural framework numbers near f_peak ≈ 1.15
# -----------------------------------------------------------------------
print()
print("--- 6. Framework-natural numbers near f_peak ~ 1.15 ---")
print()
candidates = [
    ("(k*-1)/k* + 1 = 5/3·...",  (3-1)/3 + 1.0/3.0/2.0),  # placeholder
    ("1 + 1/(2·k*)",             1 + 1/(2*3)),             # 7/6 = 1.167
    ("1 + 1/(2·g)",              1 + 1/(2*10)),            # 11/10 = 1.10
    ("g/(g-1)",                  10/9),                    # 1.111
    ("k*/(k*-1) - 1/g",          3/2 - 1/10),              # 1.40 — too big
    ("1 + 1/k* - 1/g",           1 + 1/3 - 1/10),          # 1.233 — too big
    ("(k*+1)/(k*+0.5)",          4/3.5),                   # 1.143 — close!
    ("(g+1)/g",                  11/10),                   # 1.10
    ("1 + 2/(k*·g)",             1 + 2/30),                # 1.067 — too small
    ("Ω_Λ + 1 = 4/3",            4/3),                     # 1.333 — too big
    ("(k* + 1/g)/(k* - 0)",      (3 + 0.1)/3),             # 1.033 — too small
    ("1 + 1/N_atoms·something",  1 + 1.0/N_atoms_ if (N_atoms_ := 4) else 4),  # placeholder
]

print(f"  Target: f_peak = {f_peak:.4f}")
print(f"  {'expression':30}  {'value':>9}  {'gap from target':>18}")
for name, val in candidates:
    gap = (val - f_peak) / f_peak * 100
    print(f"  {name:30}  {val:>9.4f}  {gap:>+15.2f}%")
print()
print("  None are clean K-rational matches. f_peak is not yet identified")
print("  as a structural number — Stage 2 derivation work needed.")
