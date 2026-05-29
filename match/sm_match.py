"""
match.sm_match — σ-deviation reporter pairing substrate outputs to PDG.

Architectural role: this is the bridge between
    simulator.observables — physics-free substrate output catalog
and
    Standard-Model identifications + PDG observed values.

Each row of the master match table specifies:
    (SM observable name, substrate-output reading, PDG value, PDG σ)

Calling sm_match_table() returns a list of MatchRecord with the framework
prediction, the PDG observation, the σ-deviation, and a brief description
of which substrate reading the prediction comes from.

This is the ONLY file in the project that directly compares framework
output to PDG: simulator never imports from here, and canonical
predictions/ never imports from here. sm_match.py is a downstream
consumer of both.
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional, Callable, List

from .sm_predictions import (
    V_us, V_cb, V_ub, V_cd, V_cs, V_td, V_ts, V_tb, V_ud, J_CKM,
    sin2_theta_W, alpha_GUT, sin2_theta_W_MZ,
    y_tau, lambda_H, alpha_1_bare, Q_Koide,
    m_tau, m_mu, m_e, m_H, M_Z, m_W,
    g_1, g_2, g_3, alpha_s, alpha_EM, M_unif, R_infinity,
    m_nu2, m_nu3, R_nu_splitting,
    theta_12_PMNS, theta_13_PMNS, theta_23_PMNS,
    delta_CP_CKM, theta_QCD, alpha_21_PMNS, alpha_31_PMNS,
    epsilon_CP, A_hemispherical, delta_CP_PMNS, beta_cosmic_birefringence,
    eta_lattice, eta_NB_H, D_H,
    H_0, t_0, Lambda_CC, w_DE, Omega_DM_over_Omega_m, eta_B, A_s,
)
from .anchors import G_F


@dataclass
class MatchRecord:
    """One row of the SM-match table."""
    sm_name: str                # SM observable name (e.g., 'V_us', 'm_τ')
    pred: Any                   # framework prediction (Fraction, float, complex)
    obs: Optional[float]        # PDG observed value (None if not directly observed)
    sigma: Optional[float]      # PDG 1-σ uncertainty (None if not observed)
    units: str = ''             # observation units string
    sigma_dev: Optional[float] = field(init=False, default=None)
    rel_err: Optional[float] = field(init=False, default=None)
    note: str = ''

    def __post_init__(self):
        if self.obs is None or self.sigma is None or self.pred is None:
            return
        try:
            p = float(self.pred)
        except Exception:
            return
        diff = p - self.obs
        self.sigma_dev = diff / self.sigma if self.sigma > 0 else None
        if self.obs != 0:
            self.rel_err = diff / self.obs


# ============================================================================
# Master match table — every observable, one record
# ============================================================================

# PDG 2024 values (from docs/parameters/target_parameters.md and individual
# prediction files). Entries with obs=None are framework-internal or not
# directly observed.
_TABLE_SPEC = [
    # Gauge / mixing
    ('V_us',                lambda: float(V_us()),                  0.22501,         0.00068,    '',           'CKM'),
    ('V_cb',                lambda: float(V_cb()),                  0.04182,         0.00085,    '',           'CKM'),
    ('V_ub',                lambda: float(V_ub()),                  0.00382,         0.00020,    '',           'CKM'),
    ('V_ud',                lambda: float(V_ud()),                  0.97435,         0.00016,    '',           'CKM'),
    ('V_cd',                lambda: float(V_cd()),                  0.22487,         0.00068,    '',           'CKM'),
    ('V_cs',                lambda: float(V_cs()),                  0.97349,         0.00016,    '',           'CKM'),
    ('V_td',                lambda: float(V_td()),                  0.00854,         0.00023,    '',           'CKM'),
    ('V_ts',                lambda: float(V_ts()),                  0.04110,         0.00083,    '',           'CKM'),
    ('V_tb',                lambda: float(V_tb()),                  0.99912,         0.00003,    '',           'CKM'),
    ('J_CKM',               lambda: float(J_CKM()),                 3.08e-5,         0.05e-5,    '',           'CKM'),
    ('δ_CP_CKM (deg)',      lambda: float(delta_CP_CKM()),          68.5,            3.0,        'deg',        'CP phase'),
    ('θ_QCD',               lambda: float(theta_QCD()),             0,               1e-10,      '',           'strong CP'),

    # Higgs / mass cascade
    ('m_τ (GeV)',           lambda: float(m_tau()),                 1.77686,         0.00012,    'GeV',        'Koide'),
    ('m_μ (GeV)',           lambda: float(m_mu()),                  0.10566,         5e-7,       'GeV',        'Koide'),
    ('m_e (GeV)',           lambda: float(m_e()),                   0.000511,        1e-7,       'GeV',        'Koide'),
    ('m_H (GeV)',           lambda: float(m_H()),                   125.20,          0.11,       'GeV',        'EW cascade'),
    ('M_Z (GeV)',           lambda: float(M_Z()),                   91.1876,         0.0021,     'GeV',        'EW cascade'),
    ('m_W (GeV)',           lambda: float(m_W()),                   80.369,          0.013,      'GeV',        'EW cascade'),
    ('y_τ',                 lambda: float(y_tau()),                 7.226e-3,        1e-5,       '',           'Yukawa'),
    ('λ_H (Higgs quartic)', lambda: float(lambda_H()),              0.1294,          0.0006,     '',           'Higgs quartic'),

    # Gauge couplings
    ('sin²θ_W (M_Z)',       lambda: float(sin2_theta_W_MZ()),       0.23121,         0.00004,    '',           'EW mixing'),
    ('sin²θ_W (M_unif)',    lambda: float(sin2_theta_W()),          0.375,           0.05,       '',           'GUT'),
    ('α_GUT',               lambda: float(alpha_GUT()),             1.0/24.3,        1.0/24/40,  '',           'GUT'),
    ('g_1 (M_Z)',           lambda: float(g_1()),                   0.4614,          0.0001,     '',           'gauge'),
    ('g_2 (M_Z)',           lambda: float(g_2()),                   0.6520,          0.0001,     '',           'gauge'),
    ('g_3 (M_Z)',           lambda: float(g_3()),                   1.218,           0.005,      '',           'gauge'),
    ('α_s (M_Z)',           lambda: float(alpha_s()),               0.118,           0.0009,     '',           'strong'),
    ('α_EM⁻¹ (M_Z)',        lambda: 1.0/float(alpha_EM()),          127.944,         0.014,      '',           'EM'),
    ('M_unif (GeV)',        lambda: float(M_unif()),                2.0e16,          0.5e16,     'GeV',        'GUT scale'),
    ('R_∞ (m⁻¹)',           lambda: float(R_infinity()),            1.0973731568e7,  2.1e-3,     'm⁻¹',        'atomic'),

    # Neutrinos
    ('m_ν₂ (eV)',           lambda: float(m_nu2()),                 8.65e-3,         5e-5,       'eV',         'neutrino'),
    ('m_ν₃ (eV)',           lambda: float(m_nu3()),                 50.13e-3,        0.20e-3,    'eV',         'neutrino'),
    ('R_ν = Δm²₃₁/Δm²₂₁',    lambda: float(R_nu_splitting()),        32.576,          0.5,        '',           'neutrino splitting'),
    ('θ_12 PMNS (deg)',     lambda: float(theta_12_PMNS()),         33.41,           0.75,       'deg',        'PMNS'),
    ('θ_13 PMNS (deg)',     lambda: float(theta_13_PMNS()),         8.57,            0.11,       'deg',        'PMNS'),
    ('θ_23 PMNS (deg)',     lambda: float(theta_23_PMNS()),         49.2,            1.3,        'deg',        'PMNS'),
    ('δ_CP PMNS (deg)',     lambda: float(delta_CP_PMNS()),         177.0,           20.0,       'deg',        'PMNS'),
    ('α_21 PMNS (deg)',     lambda: float(alpha_21_PMNS()),         None,            None,       'deg',        'Majorana (unconstrained)'),
    ('α_31 PMNS (deg)',     lambda: float(alpha_31_PMNS()),         None,            None,       'deg',        'Majorana (unconstrained)'),

    # Cosmology
    ('H_0 (km/s/Mpc)',      lambda: float(H_0()),                   67.4,            0.5,        'km/s/Mpc',   'cosmology'),
    ('t_0 (Gyr)',            lambda: float(t_0()),                   13.79,           0.02,       'Gyr',        'cosmology'),
    ('Λ_CC',                lambda: float(Lambda_CC()),             2.85e-122,       3e-124,     '(Planck)',   'cosmology'),
    ('w_DE',                lambda: float(w_DE()),                  -1.03,           0.03,       '',           'dark energy'),
    ('Ω_DM/Ω_m',            lambda: float(Omega_DM_over_Omega_m()), 0.846,           0.016,      '',           'dark matter'),
    ('η_B (baryon/photon)', lambda: float(eta_B()),                 6.12e-10,        0.04e-10,   '',           'baryogenesis'),
    ('A_s primordial',      lambda: float(A_s()),                   2.10e-9,         0.03e-9,    '',           'primordial'),
    ('β cosmic birefringence (deg)', lambda: float(beta_cosmic_birefringence()), 0.342, 0.094,   'deg',        'CMB EB'),

    # Lorentz / dim-6 LV (substrate predictions; observed bounds are very loose)
    ('η_lattice (dim-6 LV)', lambda: float(eta_lattice()),           None,            None,       '',           'LV (LHAASO bound)'),
    ('η_NB^H (scalar Bloch)', lambda: float(eta_NB_H()),             None,            None,       '',           'LV substrate sister'),
    ('D_H',                 lambda: float(D_H()),                   None,            None,       '',           'Bloch dispersion'),

    # Bayesian / dark
    ('ε_CP (Bayesian asymmetry)', lambda: float(epsilon_CP()),       None,            None,       '',           'birth/death asymmetry'),
    ('A_hemispherical CMB', lambda: float(A_hemispherical()),       0.07,            0.02,       '',           'CMB hemispherical'),

    # Anchors
    ('G_F (GeV⁻²)',         lambda: float(G_F()),                   1.1663787e-5,    6e-12,      'GeV⁻²',      'EXTERNAL ANCHOR'),

    # Structural identities (no PDG comparison; framework-internal)
    ('Q_Koide',             lambda: float(Q_Koide()),                None,            None,       '',           'Koide identity'),
    ('α₁_bare',             lambda: float(alpha_1_bare()),           None,            None,       '',           'Feshbach n_fixed=2'),
]


def sm_match_table() -> List[MatchRecord]:
    """Build the master match table — one MatchRecord per SM observable.

    Suppresses noisy stdout from canonical prediction files.
    """
    import io as _io
    import contextlib as _contextlib
    records = []
    for sm_name, pred_fn, obs, sigma, units, note in _TABLE_SPEC:
        try:
            with _contextlib.redirect_stdout(_io.StringIO()):
                pred = pred_fn()
        except Exception as e:
            pred = None
            note = f"{note} | ERROR: {e}"
        records.append(MatchRecord(
            sm_name=sm_name, pred=pred, obs=obs, sigma=sigma,
            units=units, note=note,
        ))
    return records


def print_match_table(records: Optional[List[MatchRecord]] = None):
    """Pretty-print the match table for inspection.

    Format: SM name | predicted | observed | σ-dev | rel-err | note.
    """
    if records is None:
        records = sm_match_table()
    print("=" * 110)
    print(f"{'SM observable':32s} {'predicted':>16s} {'observed (PDG)':>16s} "
          f"{'σ-dev':>8s} {'rel-err':>10s}  note")
    print("-" * 110)
    for r in records:
        pred_str = f"{r.pred}" if r.pred is None else (
            f"{r.pred:.4g}" if isinstance(r.pred, (int, float)) else str(r.pred)
        )
        obs_str = f"{r.obs:.4g}" if r.obs is not None else "—"
        sd_str = f"{r.sigma_dev:+.2f}σ" if r.sigma_dev is not None else "—"
        re_str = f"{r.rel_err:+.2%}" if r.rel_err is not None else "—"
        print(f"{r.sm_name:32s} {pred_str:>16s} {obs_str:>16s} "
              f"{sd_str:>8s} {re_str:>10s}  {r.note}")
    print("=" * 110)
    n_total = len(records)
    n_obs = sum(1 for r in records if r.obs is not None)
    n_within_3sig = sum(
        1 for r in records
        if r.sigma_dev is not None and abs(r.sigma_dev) < 3
    )
    print(f"\n  {n_total} observables in match table; {n_obs} with PDG observation")
    print(f"  {n_within_3sig}/{n_obs} within 3σ_PDG")


def summary_within_3sigma() -> dict:
    """Quick summary: how many predictions are within 3σ of PDG?"""
    records = sm_match_table()
    n_total = len(records)
    n_obs = sum(1 for r in records if r.obs is not None)
    n_within_1sig = sum(
        1 for r in records
        if r.sigma_dev is not None and abs(r.sigma_dev) < 1
    )
    n_within_3sig = sum(
        1 for r in records
        if r.sigma_dev is not None and abs(r.sigma_dev) < 3
    )
    return {
        'n_total': n_total,
        'n_with_observation': n_obs,
        'n_within_1sigma': n_within_1sig,
        'n_within_3sigma': n_within_3sig,
        'frac_within_3sigma': n_within_3sig / n_obs if n_obs > 0 else 0,
    }


if __name__ == "__main__":
    print_match_table()
    print()
    print("Summary:", summary_within_3sigma())
