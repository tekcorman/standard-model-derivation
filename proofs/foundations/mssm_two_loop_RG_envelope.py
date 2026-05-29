"""
proofs/foundations/mssm_two_loop_RG_envelope.py

MSSM two-loop RG running diagnostic — how much each gauge-coupling
cluster observable at M_Z varies as the SUSY-breaking scale M_SUSY is
scanned across [M_Z, 10 TeV], with two-loop MSSM β-functions above
M_SUSY and SM β-functions below.

This is a sensitivity diagnostic only. The earlier framing of the envelope
as a theoretical-uncertainty band widening Clause 8 PASS/FAIL was retracted —
predictions are compared to PDG against σ_PDG alone; this script reports the
M_SUSY-scan spread without claiming it absorbs deviations.

Convention: x_i ≡ 1/α_i (GUT-normalized for i=1). t ≡ ln(µ).
  dx_i/dt = −b_i/(2π) − Σ_j b_ij/(8π²) · (1/x_j)         (two-loop RGE)

References:
  Martin, "A Supersymmetry Primer" (hep-ph/9709356) §6.5.2-3 (MSSM b_i, b_ij)
  Machacek & Vaughn, Nucl. Phys. B222 (1983) 83 (two-loop SM β-functions)
  Peskin & Schroeder §16 (one-loop SM β-functions, threshold matching)
"""

import math
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.integrate import solve_ivp
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


# ============================================================================
# β-function coefficients
# ============================================================================
# Ordering: index 0 = U(1)_Y (GUT-normalized, α_1 = (5/3) α_Y),
#           index 1 = SU(2)_L, index 2 = SU(3)_c.

# --- MSSM ---
B_MSSM = np.array([33.0/5.0, 1.0, -3.0])
B_IJ_MSSM = np.array([
    [199.0/25.0, 27.0/5.0, 88.0/5.0],
    [9.0/5.0,    25.0,     24.0    ],
    [11.0/5.0,   9.0,      14.0    ],
])

# --- SM (3 generations + 1 Higgs doublet) ---
B_SM = np.array([41.0/10.0, -19.0/6.0, -7.0])
# Two-loop SM b_ij (Machacek-Vaughn; gauge part only, Yukawa contributions
# omitted — they're <0.1% and irrelevant for the envelope). GUT-normalized b_1.
B_IJ_SM = np.array([
    [199.0/50.0, 27.0/10.0, 44.0/5.0],
    [9.0/10.0,   35.0/6.0,  12.0    ],
    [11.0/10.0,  9.0/2.0,  -26.0    ],
])


def rge_rhs(t, x, b, b_ij, two_loop=True):
    """RHS of dx_i/dt where x_i = 1/α_i, t = ln(µ)."""
    x = np.asarray(x, dtype=float)
    alpha = 1.0 / x
    dxdt = -b / (2.0 * math.pi)
    if two_loop:
        dxdt -= (b_ij @ alpha) / (8.0 * math.pi ** 2)
    return dxdt


# ============================================================================
# Running
# ============================================================================

def run_couplings(x_start, t_start, t_end, b, b_ij, two_loop=True, n_steps=2000):
    """Integrate the RGE from t_start to t_end. Returns x at t_end.

    Uses scipy solve_ivp if available, else a fixed-step RK4.
    """
    if t_start == t_end:
        return np.asarray(x_start, dtype=float)
    if _HAVE_SCIPY:
        sol = solve_ivp(
            lambda t, x: rge_rhs(t, x, b, b_ij, two_loop),
            (t_start, t_end), x_start,
            method='RK45', rtol=1e-10, atol=1e-12, dense_output=False,
        )
        return sol.y[:, -1]
    # Fallback RK4
    x = np.asarray(x_start, dtype=float)
    h = (t_end - t_start) / n_steps
    t = t_start
    for _ in range(n_steps):
        k1 = rge_rhs(t, x, b, b_ij, two_loop)
        k2 = rge_rhs(t + h/2, x + h*k1/2, b, b_ij, two_loop)
        k3 = rge_rhs(t + h/2, x + h*k2/2, b, b_ij, two_loop)
        k4 = rge_rhs(t + h, x + h*k3, b, b_ij, two_loop)
        x = x + h*(k1 + 2*k2 + 2*k3 + k4)/6
        t += h
    return x


# ============================================================================
# Full pipeline: M_unif → (MSSM) → M_SUSY → (SM) → M_Z
# ============================================================================

ALPHA_GUT = 1.0 / 24.0           # framework theorem
M_UNIF_GEV = 1.985e16            # framework theorem (M_unif = 32/k*^(g-1) · M_Pl)
M_Z_GEV = 91.1876                # PDG (running endpoint; envelope ~insensitive
                                 # to whether 91.19 or framework's 91.97)
HYPERCHARGE_NORM = 3.0 / 5.0     # α_Y = (3/5) α_1_GUT (SU(5) embedding)
ALPHA_EM_0 = 1.0 / 137.035999    # IR fine-structure constant (M_SUSY-insensitive)
M_E_MEV = 0.51099895             # electron mass (theorem-grade in framework)


def cluster_observables_at_MZ(M_SUSY_GeV, two_loop=True):
    """Run from M_unif to M_Z through the M_SUSY threshold; return observables."""
    t_unif = math.log(M_UNIF_GEV)
    t_susy = math.log(M_SUSY_GeV)
    t_MZ = math.log(M_Z_GEV)

    # Start at M_unif: all three GUT-normalized couplings equal α_GUT.
    x_unif = np.array([1.0/ALPHA_GUT, 1.0/ALPHA_GUT, 1.0/ALPHA_GUT])

    # MSSM running M_unif → M_SUSY
    x_susy = run_couplings(x_unif, t_unif, t_susy, B_MSSM, B_IJ_MSSM, two_loop)

    # Threshold matching at M_SUSY: continuity (one-loop matching; finite
    # threshold corrections O(α/4π) are O(<0.1%) and not modeled here).
    x_susy_sm = x_susy.copy()

    # SM running M_SUSY → M_Z
    x_MZ = run_couplings(x_susy_sm, t_susy, t_MZ, B_SM, B_IJ_SM, two_loop)

    inv_a1, inv_a2, inv_a3 = x_MZ
    a1 = 1.0/inv_a1
    a2 = 1.0/inv_a2
    a3 = 1.0/inv_a3
    aY = HYPERCHARGE_NORM * a1
    sin2_W = aY / (a2 + aY)
    alpha_EM = a2 * sin2_W                         # = (1/α_2 + 1/α_Y)^{-1}
    g_1 = math.sqrt(4*math.pi*a1)
    g_2 = math.sqrt(4*math.pi*a2)
    g_3 = math.sqrt(4*math.pi*a3)
    # M_Z from framework formula uses v_higgs and (α_2, α_1); the M_SUSY
    # sensitivity enters through α_2, α_1. Use the framework's structural
    # form M_Z ∝ √(α_2 + (3/5)α_1) (the v_higgs prefactor is M_SUSY-independent).
    MZ_factor = math.sqrt(a2 + HYPERCHARGE_NORM*a1)   # ∝ M_Z up to fixed prefactor
    # m_W = M_Z · cos θ_W = M_Z · √(1 − sin²θ_W).
    mW_factor = MZ_factor * math.sqrt(1.0 - sin2_W)
    # R∞ ∝ α_EM(0)² · m_e — α_EM(0) is M_SUSY-INSENSITIVE.
    R_inf_factor = ALPHA_EM_0**2

    return {
        'sin2_theta_W_MZ': sin2_W,
        'alpha_EM_MZ': alpha_EM,
        'alpha_s_MZ': a3,
        'g_1_MZ': g_1,
        'g_2_MZ': g_2,
        'g_3_MZ': g_3,
        'M_Z_factor': MZ_factor,        # M_Z ∝ this (fixed prefactor √π·v)
        'm_W_factor': mW_factor,        # m_W ∝ this
        'R_inf_factor': R_inf_factor,   # M_SUSY-flat
        'inv_alpha_1': inv_a1,
        'inv_alpha_2': inv_a2,
        'inv_alpha_3': inv_a3,
    }


# ============================================================================
# Envelope scan
# ============================================================================

def compute_envelope(verbose=True):
    """Scan M_SUSY ∈ [M_Z, 10 TeV]; return half-spread per observable."""
    M_SUSY_grid_GeV = [
        M_Z_GEV,        # 91.19 GeV (lower edge — pure MSSM)
        200.0,
        500.0,
        1000.0,         # 1 TeV — literature-convention central value
        2000.0,
        5000.0,
        10000.0,        # 10 TeV (upper edge)
    ]

    observables = ['sin2_theta_W_MZ', 'alpha_EM_MZ', 'alpha_s_MZ',
                   'g_1_MZ', 'g_2_MZ', 'g_3_MZ', 'M_Z_factor', 'm_W_factor']

    # 1-loop and 2-loop both scanned; envelope = spread over (M_SUSY × loop-order)
    results = {}  # (M_SUSY, loop) -> obs dict
    for M_SUSY in M_SUSY_grid_GeV:
        for two_loop in [False, True]:
            results[(M_SUSY, two_loop)] = cluster_observables_at_MZ(M_SUSY, two_loop)

    if verbose:
        print("=" * 110)
        print("M_SUSY threshold sensitivity scan (MSSM above M_SUSY, SM below)")
        print("=" * 110)
        print()
        print(f"  RGE: x_i = 1/α_i, dx_i/dt = −b_i/(2π) − Σ_j b_ij/(8π²)·(1/x_j)")
        print(f"  MSSM above M_SUSY (b = {list(B_MSSM)}); SM below ({list(B_SM)}).")
        print(f"  Start: 1/α_i(M_unif) = 24 (all GUT-normalized, α_GUT = 1/24).")
        print(f"  M_unif = {M_UNIF_GEV:.3e} GeV; M_Z = {M_Z_GEV} GeV.")
        print()
        print(f"  Scan grid: M_SUSY ∈ {{{', '.join(f'{m:g}' for m in M_SUSY_grid_GeV)}}} GeV × {{1-loop, 2-loop}}")
        print()

        # Table: observable values across the scan
        print(f"  {'M_SUSY (GeV)':>12} {'loop':>6} | "
              + " ".join(f"{o.replace('_MZ',''):>14}" for o in ['sin2_theta_W', 'alpha_EM', 'alpha_s', 'g_3']))
        print("  " + "-" * 100)
        for M_SUSY in M_SUSY_grid_GeV:
            for two_loop in [False, True]:
                r = results[(M_SUSY, two_loop)]
                ll = "2-loop" if two_loop else "1-loop"
                print(f"  {M_SUSY:>12g} {ll:>6} | "
                      + f"{r['sin2_theta_W_MZ']:>14.6f} {r['alpha_EM_MZ']:>14.7f} "
                      + f"{r['alpha_s_MZ']:>14.6f} {r['g_3_MZ']:>14.6f}")
        print()

    # Central value: M_SUSY = 1 TeV, two-loop
    central = results[(1000.0, True)]

    # Half-spread per observable over the M_SUSY × loop-order scan
    half_spread = {}
    for obs in observables:
        vals = [results[(M, ll)][obs] for M in M_SUSY_grid_GeV for ll in [False, True]]
        envelope_width = max(vals) - min(vals)
        half_spread[obs] = envelope_width / 2.0

    # R∞: α_EM(0) is M_SUSY-insensitive, so the M_SUSY-threshold spread is ~0.
    half_spread['R_inf'] = 0.0

    if verbose:
        print("=" * 110)
        print("M_SUSY-scan half-spread per observable (diagnostic only)")
        print("=" * 110)
        print()
        print(f"  {'observable':<22} {'central (M_SUSY=1TeV, 2-loop)':>30} "
              + f"{'half-spread (abs)':>20} {'half-spread (rel)':>20}")
        print("  " + "-" * 96)
        for obs in observables:
            c = central[obs]
            st = half_spread[obs]
            rel = st / abs(c) if c != 0 else 0.0
            label = obs.replace('_MZ', '(M_Z)').replace('_factor', ' (∝)')
            print(f"  {label:<22} {c:>30.7f} {st:>20.7f} {100*rel:>18.3f}%")
        print()
        print(f"  Note: this is a sensitivity diagnostic. It does NOT widen Clause 8")
        print(f"  tolerances — predictions are compared to PDG against σ_PDG alone.")
        print(f"  α_s(M_Z) shows the largest M_SUSY dependence in this scan;")
        print(f"  α_EM(0) (and R∞) are M_SUSY-insensitive.")

    return central, half_spread


# ============================================================================
# Comparison to PDG (central values only — σ_PDG-only deviation reporting)
# ============================================================================

PDG = {
    'sin2_theta_W_MZ': (0.23121, 0.00004),
    'alpha_EM_MZ':     (1.0/127.944, 0.014/127.944**2),
    'alpha_s_MZ':      (0.1180, 0.0009),
    'g_2_MZ':          (0.6520, 0.0001),
    'g_3_MZ':          (1.218, 0.005),
}


def derive_g1_pdg():
    aEM = 1.0/127.944
    s2 = 0.23121
    aY = aEM/(1.0 - s2)
    a1 = (5.0/3.0)*aY
    return math.sqrt(4*math.pi*a1)


def pdg_comparison(verbose=True):
    """Compare framework central (M_SUSY=1TeV, 2-loop) to PDG.

    Reports raw deviation and Nσ against σ_PDG only — no envelope absorption.
    """
    central, _ = compute_envelope(verbose=False)
    pdg = dict(PDG)
    pdg['g_1_MZ'] = (derive_g1_pdg(), 0.0001)

    if verbose:
        print()
        print("=" * 96)
        print("Framework vs PDG (M_SUSY=1TeV, 2-loop central; deviation in σ_PDG only)")
        print("=" * 96)
        print()
        print(f"  {'observable':<20} {'predicted':>12} {'PDG':>12} {'σ_PDG':>10} "
              + f"{'Δ':>12} {'Nσ_PDG':>10}")
        print("  " + "-" * 90)

    rows = [
        ('sin2_theta_W_MZ', 'sin²θ_W(M_Z)'),
        ('g_1_MZ',          'g_1(M_Z) GUTn'),
        ('g_2_MZ',          'g_2(M_Z)'),
        ('g_3_MZ',          'g_3(M_Z)'),
        ('alpha_s_MZ',      'α_s(M_Z)'),
        ('alpha_EM_MZ',     'α_EM(M_Z)'),
    ]
    out = {}
    for key, label in rows:
        pred = central[key]
        pdg_central, sigma_pdg = pdg[key]
        delta = pred - pdg_central
        n_sigma_pdg = delta / sigma_pdg if sigma_pdg > 0 else float('inf')
        out[key] = (pred, pdg_central, sigma_pdg, delta, n_sigma_pdg)
        if verbose:
            print(f"  {label:<20} {pred:>12.6f} {pdg_central:>12.6f} {sigma_pdg:>10.6f} "
                  + f"{delta:>+12.6f} {n_sigma_pdg:>+9.2f}σ")

    return out


# ============================================================================
# Main
# ============================================================================

def main():
    compute_envelope(verbose=True)
    pdg_comparison(verbose=True)


if __name__ == "__main__":
    main()
