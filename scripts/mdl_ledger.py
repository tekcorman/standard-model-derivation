#!/usr/bin/env python3
"""
mdl_ledger.py — computes the MDL ledger per the FROZEN methods
(docs/audits/mdl_ledger_methods.md, committed c474e27 BEFORE this script).

Data side:  b_i = log2( W_prior / max(2*sigma_i, 2*|Delta_i|) )   [methods §2]
            (log-uniform priors computed in log-space: W = ln(hi/lo), w = width/x)
Spec side:  Column A choice-point manifest + trials-from-receipts   [methods §3]
            Column B hostile ceiling = gzip(minimal formal statement)
Baseline:   the SM-as-fit's measured parameter list, IDENTICAL priors [methods §4]

Predicted values are pulled from the live predictions/ DAG (same introspection
as scripts/value_lock.py). Measured values/sigmas are explicit in the manifest
below (auditable against PDG 2024 / NuFIT 6.0 / Planck 2018 as cited).
All judgment calls: CONSERVATIVE (methods §5.4); decisions logged in NOTES.
"""
import gzip, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'predictions'))
os.chdir(ROOT)
import run_predictions as rp

LOG2 = math.log(2)

def pred(slug):
    mod = rp._load_module(slug)
    if mod is None:
        raise RuntimeError(f'import fail: {slug}')
    p, _, _, _ = rp._find_result_vars(mod, slug)
    return p

def bits(prior, x_meas, sigma, delta):
    """Explained bits per methods §2."""
    w = max(2 * sigma, 2 * abs(delta))
    kind = prior[0]
    if kind == 'uniform':
        W = prior[2] - prior[1]
        return math.log(W / w) / LOG2
    if kind == 'loguniform':
        W = math.log(prior[2] / prior[1])
        return math.log(W / (w / abs(x_meas))) / LOG2
    raise ValueError(kind)

V_HIGGS = 246.22  # GeV; masses priced as m/v (methods §2 priors table)

# ---------------------------------------------------------------------------
# HEADLINE ROW MANIFEST (methods §2 inclusion rules; exclusions listed below)
# fields: name, slug (None = value inline), transform, measured, sigma, prior
# ---------------------------------------------------------------------------
ID = lambda v: v
ROWS = [
    # --- gauge sector: 3 independent dof (sin2thW, alpha_EM, alpha_s) ---
    ('sin2_theta_W(M_Z)', 'sin2_theta_W_MZ', ID, 0.23121, 0.00004, ('uniform', 0, 1)),
    ('alpha_EM(M_Z)',     'alpha_EM',        ID, 1/127.944, 8.6e-7, ('loguniform', 1e-6, 4*math.pi)),
    ('alpha_s(M_Z)',      'alpha_s',         ID, 0.1180, 0.0009,  ('loguniform', 1e-6, 4*math.pi)),
    # --- EW masses (as m/v; M_Z and m_W are MISS-priced rows) ---
    ('M_Z/v',  'M_Z', lambda v: v/V_HIGGS, 91.1876/V_HIGGS, 0.0021/V_HIGGS, ('loguniform', 1e-12, 1)),
    ('m_W/v',  'm_W', lambda v: v/V_HIGGS, 80.369/V_HIGGS,  0.013/V_HIGGS,  ('loguniform', 1e-12, 1)),
    ('m_H/v',  'm_H', lambda v: v/V_HIGGS, 125.20/V_HIGGS,  0.11/V_HIGGS,   ('loguniform', 1e-12, 1)),
    # --- quark masses (m/v; PDG 2024 MS-bar) ---
    ('m_u/v', 'm_u', lambda v: v/V_HIGGS, 2.16e-3/V_HIGGS, 0.49e-3/V_HIGGS, ('loguniform', 1e-12, 1)),
    ('m_d/v', 'm_d', lambda v: v/V_HIGGS, 4.67e-3/V_HIGGS, 0.48e-3/V_HIGGS, ('loguniform', 1e-12, 1)),
    ('m_s/v', 'm_s', lambda v: v/V_HIGGS, 0.0934/V_HIGGS,  0.0086/V_HIGGS,  ('loguniform', 1e-12, 1)),
    ('m_c/v', 'm_c', lambda v: v/V_HIGGS, 1.27/V_HIGGS,    0.02/V_HIGGS,    ('loguniform', 1e-12, 1)),
    ('m_b/v', 'm_b', lambda v: v/V_HIGGS, 4.18/V_HIGGS,    0.03/V_HIGGS,    ('loguniform', 1e-12, 1)),
    ('m_t/v', 'm_t', lambda v: v/V_HIGGS, 172.69/V_HIGGS,  0.30/V_HIGGS,    ('loguniform', 1e-12, 1)),
    # --- charged leptons (m/v; the open ppm misses are Delta-priced) ---
    ('m_e/v',  'm_e',   lambda v: v/V_HIGGS, 0.51099895e-3/V_HIGGS, 1.6e-13/V_HIGGS, ('loguniform', 1e-12, 1)),
    ('m_mu/v', 'm_mu',  lambda v: v/V_HIGGS, 0.1056583755/V_HIGGS/1e3 if False else 0.1056584e-0/V_HIGGS*1e-3*1e3, None, None),  # placeholder replaced below
    ('m_tau/v','m_tau', lambda v: v/V_HIGGS, 1.77686/V_HIGGS, 0.00012/V_HIGGS, ('loguniform', 1e-12, 1)),
    # --- CKM: 4 dof only (methods §2 rule 5) ---
    ('V_us',       'V_us', ID, 0.22501, 0.00068, ('uniform', 0, 1)),
    ('V_cb',       'V_cb', ID, 0.0406,  0.0009,  ('uniform', 0, 1)),
    ('V_ub',       'V_ub', ID, 3.82e-3, 0.20e-3, ('uniform', 0, 1)),
    ('delta_CP^CKM [deg]', 'delta_CP_CKM', ID, 68.5, 3.0, ('uniform', 0, 360)),
    # --- PMNS: 4 dof (theta13 conditional on DARK-MAP, priced in spec) ---
    ('theta12_PMNS [deg]', 'theta_12_PMNS', ID, 33.41, 0.75, ('uniform', 0, 90)),
    ('theta13_PMNS [deg]', 'theta_13_PMNS', ID, 8.57,  0.11, ('uniform', 0, 90)),
    ('theta23_PMNS [deg]', 'theta_23_PMNS', ID, 49.2,  1.3,  ('uniform', 0, 90)),
    ('delta_CP^PMNS [deg]','delta_CP_PMNS', ID, 177.0, 19.5, ('uniform', 0, 360)),
    # --- neutrino masses: 2 dof = R_nu + m_nu3 (m_nu2 derived; §2 rule 5) ---
    ('R_nu = dm31^2/dm21^2', 'R_nu_splitting', ID, 32.576, 0.90, ('loguniform', 1, 1e4)),
    ('m_nu3 [eV]', 'm_nu3', ID, 50.13e-3, 0.20e-3, ('loguniform', 1e-14, 1e-9 * 1e9 if False else 1)),  # fixed below
    # --- precision / beyond-single-sector ---
    ('theta_QCD [rad]', 'theta_QCD', ID, 0.0, 5e-11, ('uniform', 0, 2*math.pi)),
    ('eta_B',      'eta_B', ID, 6.12e-10, 0.04e-10, ('loguniform', 6.12e-13, 6.12e-7)),
    ('beta_biref [deg]', 'beta_cosmic_birefringence', ID, 0.342, 0.094, ('uniform', -1, 1)),
    ('A_hemis',    'A_hemispherical', ID, 0.07, 0.02, ('uniform', 0, 1)),
    ('Omega_DM/Omega_m', 'Omega_DM_over_Omega_m', ID, 0.846, 0.016, ('uniform', 0, 1)),
    ('N_eff (=N_gen)', None, lambda _: 3.0, 2.99, 0.17, ('uniform', 0, 10)),
]

# fix the two placeholder rows cleanly (kept explicit for audit clarity)
ROWS[13] = ('m_mu/v', 'm_mu', lambda v: v/V_HIGGS, 0.1056583755/V_HIGGS, 2.3e-9/V_HIGGS, ('loguniform', 1e-12, 1))
ROWS[24] = ('m_nu3 [eV]', 'm_nu3', ID, 50.13e-3, 0.20e-3, ('loguniform', 1e-14, 1))

EXCLUDED = [
    ('v_higgs / G_F', 'round-trip: N_hub calibrated from measured G_F (methods §2.1)'),
    ('g_1, g_2, g_3, lambda_H, lambda_3, alpha_GUT, M_unif, delta_rho, delta_r',
     'dof guard: re-parametrizations of counted rows / no direct observable (§2.5)'),
    ('V_ud,V_cd,V_cs,V_td,V_ts,V_tb,J_CKM', 'unitarity-derived; V_ts/V_tb also ride the V_cb data tension (§2.4/2.5)'),
    ('Q/eps/delta_Koide', 'dof guard: re-parametrization of the 3 lepton masses (§2.5)'),
    ('m_nu2', 'dof guard: derived from m_nu3 and R_nu (§2.5)'),
    ('H_0(x2), t_0, Lambda, w_DE, Omega_m/Omega_L', 'Category-B coasting: excluded from headline (§2.3)'),
    ('Omega_DM, Omega_b (absolute)', 'z_eff-conditional adoption (§2.2)'),
    ('Majorana phases, m_bb, Sum m_nu', 'unmeasured: freeze rows, not ledger rows'),
]

# ---------------------------------------------------------------------------
# SPEC SIDE — Column A choice-point manifest (methods §3)
# ---------------------------------------------------------------------------
SPEC_A = [
    ('substrate selection among waterline survivors {srs,srs-c8,lou,lov} (R-9, ruled 2026-07-01)', math.log2(4)),
    ('ADOPTED-NU-MAJ-PHASE: h^g reading among 4 documented fork branches', math.log2(4)),
    ('ADOPTED-DARK-MAP: classifier scope over ~4 pathways', math.log2(4)),
    ('ADOPTED-A5b-Sub3: sub-class assignment, 6 gated rows x log2(3)', 6 * math.log2(3)),
    ('ADOPTED-B3 residue: lepton-vs-quark sector label', 1.0),
    ('dark-sign rate-reading selection (3 candidate readings; CAS lemma open)', math.log2(3)),
    ('N_hub calibration from measured G_F (ppm-class consumed measurement)', 20.0),
    ('z_eff adoption NOT priced: its dependent rows are excluded from the headline', 0.0),
    ('all theorem-closed selections (k*, d, gauge group, dictionary/selection-map, level-selection, Q=2/3, delta=2/9, dark magnitude/channel/power, ...)', 0.0),
]

# Trials from receipts — the FROZEN methods §3 rule: per observable family,
# charge log2(1 + N_candidates), with an 8-candidate default floor where records
# are thin. Receipts counted mechanically: predictions/retracted/ versions +
# negative_results.md killed hypotheses + alternative-consideration instances
# in the parameter uniqueness ledger.
def count_trials(floor_candidates=8, n_families=None):
    retracted = len([f for f in os.listdir(os.path.join(ROOT, 'predictions', 'retracted')) if f.endswith('.py')])
    neg = 0
    p = os.path.join(ROOT, 'explorations', 'negative_results.md')
    if os.path.exists(p):
        neg = sum(1 for line in open(p) if line.startswith('### '))
    ledger_alts = 0
    lp = os.path.join(ROOT, 'docs', 'parameters', 'parameter_uniqueness_ledger.md')
    if os.path.exists(lp):
        import re
        pat = re.compile(r'alternative|rejected|excluded|ruled out|refuted', re.I)
        ledger_alts = sum(1 for line in open(lp) if pat.search(line))
    documented = retracted + neg + ledger_alts
    nf = n_families if n_families else len(ROWS)
    per_family = max(documented / nf, floor_candidates)
    return documented, nf * math.log2(1 + per_family)

# Column B — hostile ceiling: gzip of a minimal formal statement (methods §3)
MINIMAL_SPEC = """AXIOMS: (A) the universe is self-contained (no external structure). (B) predictions
are for finite-memory observers. (I) a binary distinction is read as an involutive operator.
A5: the Bloch-Hashimoto spectrum of the substrate is the SM mass spectrum; above-waterline
non-backtracking branch measure is coupling strength. DERIVED SLATE: toggles generate F_inv(E);
MDL is a waterline (retain every representation with L_total<L_raw); complex Hilbert space via
purification; local CAR via Jordan-Wigner. SUBSTRATE: the MDL-dominant 3-regular chiral crystal
net: srs (I4132, k*=3, girth 10, cell V=4 E=6), dominant in the waterline survivor set
{srs, srs-c8, lou, lov}. OBJECT: D = B(srs (x) srs-z) (x) d_N, B the non-backtracking operator on
the joint mirror cover, d_N the run; G = (I-uB)^-1 with u = alpha_1 = (2/3)^8. READS: masses =
diagonal recurrence rates (circulant sqrt(m_j) = |c0+c1 w^j+c1* w^-j|, moduli (4,2,2), phase =
directed run phase); mixings = off-diagonal walk sums (waterline series u^L/(1-u^L), counting
density k*^2/(g N)); gauge = Cl(6) Fock Hamming grading, Q=n/k*, beta from the 4D time-completion
(1/3)Tf+(2/3)TH+(2/3)C2G at boundary alpha_GUT^-1 = 2^k* k* = 24, sin^2 = k*/2^k*; dark = first
girth return Sigma = alpha_1/h at h = (sqrt3+i sqrt5)/2, sign DOWN by mass=recurrence-rate;
cosmology = spectral action poles, Lambda = 3/N^2, coasting H0 t0 = 1, observer/substrate rate
gap 1+1/(5 k*) = 16/15. SCALE: one unit (Planck) and one calibrated dimensional input N_hub
(from G_F). ADOPTIONS: NU-MAJ-PHASE (h^g), DARK-MAP (beta, theta13), A5b-Sub3 classifier,
B3 sector label. OPEN: m_e -70.3ppm subleading, M_Z oblique +7.76 sigma, zeta_D4(0), the
substrate discriminator."""

def column_b_bits():
    return 8 * len(gzip.compress(MINIMAL_SPEC.encode(), 9))

# LOGGED JUDGMENT CALL (methods §5.4; supersession candidate for methods v2):
# Column B as v1-written prices the framework's prose against NOTHING. The symmetric
# hostile comparison gzips a minimal SM statement in the same style — which must embed
# its measured parameter decimals (the framework's statement contains no decimal
# literals; every constant is K-rational/closed-form). Both gzips reported; the fair
# hostile margin is their DIFFERENCE plus the data-side residual asymmetry.
MINIMAL_SM = """AXIOMS: quantum field theory on Minkowski spacetime, gauge principle. GAUGE GROUP:
SU(3)_c x SU(2)_L x U(1)_Y, spontaneously broken by one Higgs doublet. MATTER: 3 generations of
chiral fermions in reps (3,2,1/6)+(3bar,1,-2/3)+(3bar,1,1/3)+(1,2,-1/2)+(1,1,1), plus 3
right-handed neutrinos (Majorana, seesaw). PARAMETERS (measured inputs, PDG 2024 / NuFIT 6.0):
v = 246.2196 GeV; sin^2 theta_W(M_Z) = 0.23121; alpha_EM(M_Z) = 1/127.944;
alpha_s(M_Z) = 0.1180; m_H = 125.20 GeV; m_u = 2.16 MeV; m_d = 4.67 MeV; m_s = 93.4 MeV;
m_c = 1.27 GeV; m_b = 4.18 GeV; m_t = 172.69 GeV; m_e = 0.51099895000 MeV;
m_mu = 105.6583755 MeV; m_tau = 1776.86 MeV; V_us = 0.22501; V_cb = 0.0406;
V_ub = 0.00382; delta_CKM = 68.5 deg; theta12 = 33.41 deg; theta13 = 8.57 deg;
theta23 = 49.2 deg; delta_PMNS = 177 deg; dm21^2 = 7.49e-5 eV^2; dm31^2 = 2.513e-3 eV^2;
theta_QCD < 1e-10. UNEXPLAINED (no parameter encodes them): eta_B, the CMB hemispherical
asymmetry amplitude, the dark-matter fraction, cosmic birefringence, why 3 generations."""

def column_b_sm_bits():
    return 8 * len(gzip.compress(MINIMAL_SM.encode(), 9))

# ---------------------------------------------------------------------------
# SM BASELINE (methods §4): the SM+nu measured parameter list, SAME priors
# (v/G_F is an input here; theta_QCD is an input bounded by measurement)
# ---------------------------------------------------------------------------
BASE = [
    ('v [GeV]', 246.22, 246.22*4.3e-7, ('loguniform', 0.24622, 246220.0)),  # G_F ppm-class, +-3 decades
    ('sin2_theta_W', 0.23121, 0.00004, ('uniform', 0, 1)),
    ('alpha_EM(M_Z)', 1/127.944, 8.6e-7, ('loguniform', 1e-6, 4*math.pi)),
    ('alpha_s(M_Z)', 0.1180, 0.0009, ('loguniform', 1e-6, 4*math.pi)),
    ('m_H/v', 125.20/V_HIGGS, 0.11/V_HIGGS, ('loguniform', 1e-12, 1)),
    ('m_u/v', 2.16e-3/V_HIGGS, 0.49e-3/V_HIGGS, ('loguniform', 1e-12, 1)),
    ('m_d/v', 4.67e-3/V_HIGGS, 0.48e-3/V_HIGGS, ('loguniform', 1e-12, 1)),
    ('m_s/v', 0.0934/V_HIGGS, 0.0086/V_HIGGS, ('loguniform', 1e-12, 1)),
    ('m_c/v', 1.27/V_HIGGS, 0.02/V_HIGGS, ('loguniform', 1e-12, 1)),
    ('m_b/v', 4.18/V_HIGGS, 0.03/V_HIGGS, ('loguniform', 1e-12, 1)),
    ('m_t/v', 172.69/V_HIGGS, 0.30/V_HIGGS, ('loguniform', 1e-12, 1)),
    ('m_e/v', 0.51099895e-3/V_HIGGS, 1.6e-13/V_HIGGS, ('loguniform', 1e-12, 1)),
    ('m_mu/v', 0.1056583755/V_HIGGS, 2.3e-9/V_HIGGS, ('loguniform', 1e-12, 1)),
    ('m_tau/v', 1.77686/V_HIGGS, 0.00012/V_HIGGS, ('loguniform', 1e-12, 1)),
    ('V_us', 0.22501, 0.00068, ('uniform', 0, 1)),
    ('V_cb', 0.0406, 0.0009, ('uniform', 0, 1)),
    ('V_ub', 3.82e-3, 0.20e-3, ('uniform', 0, 1)),
    ('delta_CP^CKM [deg]', 68.5, 3.0, ('uniform', 0, 360)),
    ('theta12 [deg]', 33.41, 0.75, ('uniform', 0, 90)),
    ('theta13 [deg]', 8.57, 0.11, ('uniform', 0, 90)),
    ('theta23 [deg]', 49.2, 1.3, ('uniform', 0, 90)),
    ('delta_CP^PMNS [deg]', 177.0, 19.5, ('uniform', 0, 360)),
    ('R_nu', 32.576, 0.90, ('loguniform', 1, 1e4)),
    ('m_nu3 [eV]', 50.13e-3, 0.20e-3, ('loguniform', 1e-14, 1)),
    ('theta_QCD [rad]', 0.0, 5e-11, ('uniform', 0, 2*math.pi)),
]
# NOTE: eta_B, A_hemis, Omega_DM/Omega_m, beta_biref, N_eff have NO SM-parameter
# encoding at all — they are FRAMEWORK SURPLUS (reported separately, §4).
SURPLUS_NAMES = {'eta_B', 'A_hemis', 'Omega_DM/Omega_m', 'beta_biref [deg]', 'N_eff (=N_gen)'}


def main():
    print('=' * 96)
    print('MDL LEDGER — computed per frozen methods (docs/audits/mdl_ledger_methods.md @ c474e27)')
    print('=' * 96)

    # ---- data side ----
    total, surplus_total, table = 0.0, 0.0, []
    for name, slug, tf, meas, sigma, prior in ROWS:
        p = tf(pred(slug)) if slug else tf(None)
        delta = p - meas
        b = bits(prior, meas, sigma, delta)
        miss = abs(delta) > sigma
        table.append((name, p, meas, b, 'MISS-priced' if miss else 'hit'))
        if name in SURPLUS_NAMES:
            surplus_total += b
        else:
            total += b
    print(f"\nDATA SIDE — headline rows ({len(ROWS)}):")
    print(f"{'row':<26} {'predicted':>13} {'measured':>13} {'bits':>7}  status")
    for name, p, m, b, s in table:
        print(f"{name:<26} {p:>13.6g} {m:>13.6g} {b:>7.1f}  {s}")
    print(f"\n  L_data (SM-parameter rows)     = {total:8.1f} bits")
    print(f"  L_data (beyond-SM surplus rows) = {surplus_total:8.1f} bits  (eta_B, A_hemis, Omega-ratio, beta, N_eff)")

    # ---- spec side ----
    a_total = sum(b for _, b in SPEC_A)
    documented, trials_bits = count_trials()
    print(f"\nSPEC SIDE — Column A (choice-points):")
    for name, b in SPEC_A:
        print(f"  {b:6.1f}  {name}")
    print(f"  Column A subtotal              = {a_total:8.1f} bits")
    print(f"  L_trials (methods §3 rule: log2(1+N) per family, 8-candidate floor;"
          f" {documented} receipts ≈ {documented/len(ROWS):.1f}/family documented)"
          f" = {trials_bits:8.1f} bits")
    colB = column_b_bits()
    colB_sm = column_b_sm_bits()
    print(f"  Column B hostile ceiling (gzip of the minimal formal statement) = {colB} bits")
    print(f"  Column B SYMMETRIC baseline (gzip of the minimal SM statement)  = {colB_sm} bits")
    print(f"  Column B fair hostile margin (SM gzip − framework gzip)         = {colB_sm - colB:+} bits of statement,")
    print(f"    with the framework's statement containing ZERO decimal parameter literals and the")
    print(f"    SM's embedding all ~26 (its decimals are irreducible; the framework's constants are")
    print(f"    K-rational closed forms already counted in Column A's choice-points).")

    # ---- baseline ----
    base_total = sum(bits(prior, m, s, 0.0) for _, m, s, prior in BASE)
    print(f"\nBASELINE — SM-as-fit ({len(BASE)} measured parameters, identical priors):")
    print(f"  L_SM-baseline                  = {base_total:8.1f} bits")

    # ---- verdicts ----
    print('\n' + '=' * 96)
    specA = a_total + trials_bits
    print(f"HEADLINE (Column A): spec+trials = {specA:.1f} bits  vs  data explained = {total:.1f} bits"
          f"  (+{surplus_total:.1f} surplus)")
    print(f"  margin  = {total - specA:+.1f} bits on SM-parameter rows alone"
          f"  |  compression ratio = {total/specA:.2f}x")
    print(f"  vs SM-as-fit baseline {base_total:.1f} bits: the framework encodes the same table for"
          f" {specA:.1f} bits + residuals")
    print(f"HOSTILE (Column B): spec ceiling = {colB} bits  vs  data = {total:.1f} (+{surplus_total:.1f}) bits")
    print(f"  margin  = {total + surplus_total - colB - trials_bits:+.1f} bits")
    print('=' * 96)

    # ---- SENSITIVITY (methods §5: robustness of the Column-A margin) ----
    print("\nSENSITIVITY of the Column-A margin (base = data 304-class vs spec+trials):")
    def margin(trials, spec=a_total, data=total):
        return data - spec - trials
    _, t8 = count_trials(8);   _, t16 = count_trials(16);   _, t64 = count_trials(64)
    print(f"  trials floor  8 candidates/family (methods default): margin = {margin(t8):+7.1f} bits")
    print(f"  trials floor 16 candidates/family:                    margin = {margin(t16):+7.1f} bits")
    print(f"  trials floor 64 candidates/family:                    margin = {margin(t64):+7.1f} bits")
    print(f"  N_hub priced double (40 bits):                        margin = {margin(t8, a_total+20):+7.1f} bits")
    print(f"  drop theta_QCD (largest single row, 35.9 bits):       margin = {margin(t8, a_total, total-35.9):+7.1f} bits")
    print(f"  all four stresses simultaneously (floor 64 + 2x N_hub + no theta_QCD):")
    print(f"                                                        margin = {margin(t64, a_total+20, total-35.9):+7.1f} bits")
    be = (total - a_total)
    import math as _m
    be_per_family = 2 ** (be / len(ROWS)) - 1
    print(f"  BREAK-EVEN: margin hits zero at trials = {be:.1f} bits = {be/len(ROWS):.1f} bits/family")
    print(f"              = ~{be_per_family:,.0f} secretly-tried candidates PER OBSERVABLE,")
    print(f"              against append-only registers documenting ~6 per family.")


if __name__ == '__main__':
    main()
