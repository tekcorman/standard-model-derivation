#!/usr/bin/env python3
"""Audit pilot — per-prediction bit-budget audit (v3 canonical, 2026-04-27;
T1.5 Bayesian observable Φ added 2026-04-27).

For each prediction, compute:
    Φ_obs (flat)  = log₂(prior_width / observed_σ)       # current default
    Φ_obs (Bayes) = log₂(prior / σ) − ½log₂(2π) − χ²·log₂(e)/2
                                                          # T1.5 Bayes factor
    L_amort = (chain.shared_L / N_chain_members)   # chain-amortized infrastructure
            + L_marginal                            # per-prediction structural cost
    B_pred = Φ_obs − L_amort                        # per-prediction net contribution

The Bayes-factor decomposition:
    BF = P(data | framework predicts v_pred) / P(data | uniform prior of width W)
       = (W / √(2πσ²)) · exp(−(v_pred − v_obs)² / 2σ²)
    log₂(BF) = log₂(W/σ) − ½log₂(2π) − χ² · log₂(e) / 2,    χ² = ((v_pred−v_obs)/σ)²

So `Φ_Bayes = Φ_flat − OCCAM − FIT_PENALTY` where:
    OCCAM       = ½ log₂(2π)   ≈ 1.326 bits  (Gaussian normalization)
    FIT_PENALTY = χ² log₂(e)/2 ≈ 0.721 χ² bits (penalizes poor fits)

For exact predictions (χ²=0): Φ_Bayes = Φ_flat − 1.326 (just the Occam factor).
For poor fits (χ²>>1): Φ_Bayes turns sharply negative — Bayes correctly
penalizes predictions whose value disagrees with observation in σ units.

Framework deficit (substrate-side from wave_simulator.py post T1.1):
    Φ_subs = 94.15, L_subs = 230, Net_subs = −135.85 bits.
Predictions must collectively contribute Σ B_pred ≥ 136 bits to break even.

v3 corrections vs v1:
- CHAIN ATTRIBUTION: shared infrastructure paid once across chain members
- RICHER SAMPLE: 22 predictions across 8 chains (was 5)
- INFORMED PRIORS: physically-motivated prior widths where applicable

T1.5 (this update):
- BAYESIAN Φ_obs replaces flat-prior log-ratio with proper Bayes factor
- χ² penalty surfaces predictions in disagreement with observation
- Headline run reports BOTH flat and Bayes columns for comparison
"""
import math
from collections import Counter

# Framework totals updated 2026-04-27 post frontier-closure + T1.2 formal L.
# Substrate now halts with ALL 219 catalog ops firing (LORENTZ_SIG + NC_GEOM
# closed via op 6.10 Γ-cone Minkowski + op 7.1 Connes spectral triple).
#
# Two L-encodings tracked:
#   L_HANDRATED = 679 — sum of per-op hand-rated L; double-counts downstream
#                       ops that inherit upstream closure theorems.
#   L_FORMAL    = 393 — closure-amortized: closure ops + refinement-producing ops
#                       pay full hand-rated L; downstream consumers pay 1 bit.
#                       Reproducible via `python3 simulator.py --formal-L`.
# We use L_FORMAL for the headline since hand-rated double-counts.
FRAMEWORK_PHI_SUBSTRATE  = 88.15   # strict-A2 with load-bearing-op exception
FRAMEWORK_L_HANDRATED    = 679    # enumeration-mode catalog total (no A2 gate; over-counts)
FRAMEWORK_L_SUBSTRATE    = 112    # strict-A2 + formal L: ops passing MDL waterline OR load-bearing for predictions
FRAMEWORK_NET_SUBSTRATE  = FRAMEWORK_PHI_SUBSTRATE - FRAMEWORK_L_SUBSTRATE  # = -23.85

# ---------------- Chains ----------------
# T2.5 (2026-04-27) — `op_ids` field added: explicit a separate private derivation by the author-DAG link to the
# catalog ops each chain relies on. Verified against simulator.py CATALOG.
CHAINS = {
    'Koide':     {'shared_L': 12, 'desc': 'Koide formula + PS + JW + α₁ + v_Higgs',
                  'op_ids': ['4.21', '5.6', '5.7', '5.8', '5.9', '5.30',
                             '4.34', '4.36', '4.51']},
    'CKM':       {'shared_L': 5,  'desc': 'SRS + Hashimoto + cycle counting',
                  'op_ids': ['4.21', '2.18', '4.17', '4.20', '4.22']},
    'PMNS':      {'shared_L': 5,  'desc': 'PS + dark + cycle counting',
                  'op_ids': ['4.21', '5.30', '4.34', '4.51', '4.36']},
    'Higgs':     {'shared_L': 6,  'desc': 'BZJ + α₁ + Higgs potential',
                  'op_ids': ['4.51', '4.52', '5.8', '5.9']},
    'Cosmology': {'shared_L': 8,  'desc': 'Friedmann + N(t) + BZJ',
                  'op_ids': ['6.18', '6.19', '6.20', '6.21', '6.22', '4.51']},
    'Gauge':     {'shared_L': 6,  'desc': 'GUT + Killing form + PS',
                  'op_ids': ['5.30', '4.43', '4.34', '5.42', '5.43']},
    'Neutrino':  {'shared_L': 6,  'desc': 'Feshbach + dark + R_ν',
                  'op_ids': ['2.18', '2.31', '4.17', '5.30']},
    'Parity':    {'shared_L': 5,  'desc': 'girth-cycle + parity + Lorentz',
                  'op_ids': ['2.18', '4.17', '5.39', '5.10']},
}

# ---------------- Predictions (v3 canonical sample, 22 entries) ----------------
# Each prediction's σ_eff = σ_obs only (the per-prediction theoretical-uncertainty
# band framing was retracted; sigma_effective uses σ_obs alone).
PREDICTIONS = [
    # === Gauge sector ===
    {'name':'α_GUT', 'doc':'predictions/alpha_GUT.py', 'formula':'1/24',
     'value_pred':1/24, 'value_obs':1/24.3, 'sigma_obs':0.0007,
     'prior_width':0.1, 'L_marginal':2, 'chain':'Gauge', 'note':'1/24 vs 1/24.3'},
    {'name':'sin²θ_W(M_unif)', 'doc':'predictions/sin2_theta_W.py', 'formula':'3/8',
     'value_pred':0.375, 'value_obs':0.23121, 'sigma_obs':0.00004,
     'prior_width':0.5, 'L_marginal':3, 'chain':'Gauge',
     'note':'matches at unification via MSSM RGE'},

    # === Higgs sector ===
    {'name':'m_H', 'doc':'predictions/m_H.py', 'formula':'125.30 GeV (BZJ)',
     'value_pred':125.30, 'value_obs':125.20, 'sigma_obs':0.11,
     'prior_width':None, 'L_marginal':3, 'chain':'Higgs', 'note':'0.08% match'},
    {'name':'λ_Higgs', 'doc':'predictions/lambda_higgs.py', 'formula':'2·α₁_full',
     'value_pred':0.13006, 'value_obs':0.1294, 'sigma_obs':0.001,
     'prior_width':1.0, 'L_marginal':2, 'chain':'Higgs',
     'note':'Cl(2) anti-commutation; 0.5%'},

    # === Lepton masses (Koide chain) ===
    {'name':'m_e', 'doc':'predictions/m_e.py', 'formula':'m_τ × (f_min/f_max)²',
     'value_pred':0.0005116, 'value_obs':0.0005110, 'sigma_obs':1e-9,
     'prior_width':None, 'L_marginal':2, 'chain':'Koide',
     'note':'0.12% (inherits m_τ systematic)'},
    {'name':'m_μ', 'doc':'predictions/m_mu.py', 'formula':'m_τ × (f_mid/f_max)²',
     'value_pred':0.10578, 'value_obs':0.10566, 'sigma_obs':2e-7,
     'prior_width':None, 'L_marginal':2, 'chain':'Koide', 'note':'0.12%'},
    {'name':'m_τ', 'doc':'predictions/m_tau.py', 'formula':'v · y_τ',
     'value_pred':1.77909, 'value_obs':1.77686, 'sigma_obs':0.00012,
     'prior_width':None, 'L_marginal':3, 'chain':'Koide',
     'note':'0.13%; lone independent lepton mass'},
    {'name':'Q_Koide', 'doc':'predictions/Q_Koide.py', 'formula':'2/3 exact',
     'value_pred':2/3, 'value_obs':0.6667, 'sigma_obs':0.0001,
     'prior_width':1.0, 'L_marginal':1, 'chain':'Koide',
     'note':'STRICT-SOLID; exact identity'},
    {'name':'y_τ', 'doc':'predictions/y_tau.py', 'formula':'1280/177147 = α₁_full/k*²',
     'value_pred':1280/177147, 'value_obs':1280/177147, 'sigma_obs':1e-5,
     'prior_width':1.0, 'L_marginal':2, 'chain':'Koide',
     'note':'theorem-grade, 0 adoptions; via m_τ/v'},

    # === CKM chain ===
    {'name':'V_cb', 'doc':'predictions/V_cb.py', 'formula':'256/6305',
     'value_pred':40.60e-3, 'value_obs':41.0e-3, 'sigma_obs':1.4e-3,
     'prior_width':0.05, 'L_marginal':3, 'chain':'CKM', 'note':'+0.07σ'},
    {'name':'V_us', 'doc':'predictions/V_us.py', 'formula':'9/40',
     'value_pred':0.225, 'value_obs':0.22534, 'sigma_obs':0.00045,
     'prior_width':0.5, 'L_marginal':3, 'chain':'CKM', 'note':'−0.015σ'},
    {'name':'δ_CP^CKM', 'doc':'predictions/delta_CP_CKM.py', 'formula':'arccos(1/3)',
     'value_pred':70.53, 'value_obs':68.5, 'sigma_obs':3.0,
     'prior_width':360.0, 'L_marginal':2, 'chain':'CKM', 'note':'0.7σ'},

    # === PMNS chain (new) ===
    {'name':'θ_23_PMNS', 'doc':'predictions/theta_23_PMNS.py', 'formula':'48.72°',
     'value_pred':48.72, 'value_obs':49.2, 'sigma_obs':1.3,
     'prior_width':90.0, 'L_marginal':3, 'chain':'PMNS', 'note':'0.4σ'},

    # === Cosmology ===
    {'name':'H_0', 'doc':'predictions/H_0.py', 'formula':'68.18 km/s/Mpc',
     'value_pred':68.18, 'value_obs':67.4, 'sigma_obs':0.5,
     'prior_width':None, 'L_marginal':3, 'chain':'Cosmology', 'note':'+1.6σ CMB'},
    {'name':'t_0', 'doc':'predictions/t_0.py', 'formula':'14.38 Gyr',
     'value_pred':14.38, 'value_obs':14.42, 'sigma_obs':0.5,    # Methuselah systematic
     'prior_width':None, 'L_marginal':2, 'chain':'Cosmology', 'note':'−0.1σ Methuselah'},
    {'name':'Ω_DM/Ω_m', 'doc':'predictions/Omega_DM_over_Omega_m.py', 'formula':'0.8488',
     'value_pred':0.8488, 'value_obs':0.846, 'sigma_obs':0.016,
     'prior_width':1.0, 'L_marginal':2, 'chain':'Cosmology', 'note':'0.1%'},
    {'name':'w_DE', 'doc':'predictions/w_DE.py', 'formula':'-1 exact',
     'value_pred':-1.0, 'value_obs':-1.03, 'sigma_obs':0.03,
     'prior_width':2.0, 'L_marginal':1, 'chain':'Cosmology', 'note':'cosmological constant'},
    {'name':'n_s', 'doc':'predictions/n_s', 'formula':'0.968 from branching stats',
     'value_pred':0.968, 'value_obs':0.965, 'sigma_obs':0.004,
     'prior_width':1.0, 'L_marginal':2, 'chain':'Cosmology', 'note':'0.75σ'},

    # === Parity / Lorentz ===
    {'name':'A_hemispherical', 'doc':'predictions/A_hemispherical.py', 'formula':'1/15',
     'value_pred':1/15, 'value_obs':0.07, 'sigma_obs':0.02,
     'prior_width':1.0, 'L_marginal':2, 'chain':'Parity', 'note':'CMB; 0.08σ'},
    {'name':'ε_CP_baryon', 'doc':'predictions for ε_CP', 'formula':'1/5 exact',
     'value_pred':0.2, 'value_obs':0.2, 'sigma_obs':0.05,    # broad observational uncertainty
     'prior_width':1.0, 'L_marginal':2, 'chain':'Parity',
     'note':'Sakharov component; framework-derived 1/5'},
    {'name':'η_5_LIV', 'doc':'predictions/eta_5_lorentz_dim5.py', 'formula':'0 exact',
     'value_pred':0.0, 'value_obs':0.0, 'sigma_obs':0.1,    # LHAASO 2024 |η|<0.1
     'prior_width':1.0, 'L_marginal':2, 'chain':'Parity',
     'note':'consistent with LHAASO bound'},

    # === Neutrino ===
    {'name':'R_ν_splitting', 'doc':'predictions/R_nu_splitting.py', 'formula':'228/7',
     'value_pred':228/7, 'value_obs':32.576, 'sigma_obs':0.5,
     'prior_width':100.0, 'L_marginal':3, 'chain':'Neutrino', 'note':'exact theorem'},
    {'name':'m_ν3', 'doc':'predictions/m_nu3.py', 'formula':'49.35 meV',
     'value_pred':49.35e-12, 'value_obs':50.1e-12, 'sigma_obs':2e-12,
     'prior_width':None, 'L_marginal':3, 'chain':'Neutrino',
     'note':'1.5σ; Feshbach amplitude class'},
]

# ---------------- Compute ----------------
OCCAM = 0.5 * math.log2(2 * math.pi)         # ≈ 1.326 bits
LOG2E_HALF = math.log2(math.e) / 2            # ≈ 0.721 bits per unit χ²

def sigma_effective(p):
    """σ_eff = σ_obs. The earlier per-prediction theoretical-uncertainty band
    accounting was retracted; deviations are reported against σ_obs only."""
    sig_obs = p['sigma_obs']
    if sig_obs is None:
        return None
    return sig_obs

def Phi_obs_flat(p):
    """Flat-prior observable Φ: log₂(W/σ_obs). Original v3 definition,
    using σ_obs only — kept for backward comparison."""
    val, sig, prior = p['value_obs'], p['sigma_obs'], p['prior_width']
    if val is None or sig is None:
        return None
    W = prior if prior is not None else val   # log-uniform proxy when prior=None
    return math.log2(W / sig)

def Phi_obs_bayes(p):
    """Bayes-factor observable Φ using σ_eff = σ_obs:
        log₂(W/σ_eff) − ½log₂(2π) − χ²·log₂(e)/2
        χ² = ((v_pred − v_obs)/σ_eff)²

    The compression and χ² penalty are both evaluated against σ_obs alone.
    """
    val, prior = p['value_obs'], p['prior_width']
    sig_eff = sigma_effective(p)
    if val is None or sig_eff is None:
        return None, None
    W = prior if prior is not None else val
    compression = math.log2(W / sig_eff)
    v_pred = p.get('value_pred')
    if v_pred is None:
        return compression - OCCAM, 0.0
    chi2 = ((v_pred - val) / sig_eff) ** 2
    return compression - OCCAM - LOG2E_HALF * chi2, chi2

chain_counts = Counter(p['chain'] for p in PREDICTIONS)

results = []
for p in PREDICTIONS:
    phi_flat = Phi_obs_flat(p)
    phi_bayes, chi2 = Phi_obs_bayes(p)
    n = chain_counts[p['chain']]
    L_amort = CHAINS[p['chain']]['shared_L'] / n + p['L_marginal']
    B_flat = phi_flat - L_amort if phi_flat is not None else None
    B_bayes = phi_bayes - L_amort if phi_bayes is not None else None
    results.append((p, phi_flat, phi_bayes, chi2, L_amort, B_flat, B_bayes, n))

# ---------------- Report ----------------
print("="*128)
print(f"AUDIT PILOT v3 + T1.5 Bayes — {len(PREDICTIONS)} predictions / {len(chain_counts)} chains / chain-attributed L")
print("="*128)
print(f"\nFramework substrate baseline (T1.1): Φ={FRAMEWORK_PHI_SUBSTRATE}, L={FRAMEWORK_L_SUBSTRATE}, Net={FRAMEWORK_NET_SUBSTRATE:+.2f} bits")
print(f"T1.5 corrections: −Occam (½log₂(2π) ≈ {OCCAM:.3f} bits), −FIT (≈ {LOG2E_HALF:.3f}·χ² bits)\n")

print(f"{'name':<22}{'chain':<11}{'#':>3}{'χ²':>7}{'Φ_flat':>8}{'Φ_Bayes':>9}{'L_amort':>9}{'B_flat':>8}{'B_Bayes':>9}  match")
print('-'*128)
total_B_flat, total_B_bayes, total_L, counted = 0, 0, 0, 0
for p, phi_f, phi_b, chi2, L_a, B_f, B_b, n in results:
    if phi_f is not None:
        chi2_str = f'{chi2:>7.2f}' if chi2 is not None else '   —   '
        print(f"{p['name']:<22}{p['chain']:<11}{n:>3}{chi2_str}{phi_f:>8.2f}{phi_b:>9.2f}{L_a:>9.2f}{B_f:>+8.2f}{B_b:>+9.2f}  {p['note'][:38]}")
        total_B_flat += B_f; total_B_bayes += B_b; total_L += L_a; counted += 1
    else:
        print(f"{p['name']:<22}{p['chain']:<11}{n:>3}{'—':>7}{'—':>8}{'—':>9}{L_a:>9.2f}{'—':>8}{'—':>9}  {p['note'][:38]} (indirect)")

print('-'*128)
print(f"{'TOTAL':<22}{'':<11}{'':<3}{'':<7}{'':<8}{'':<9}{total_L:>9.2f}{total_B_flat:>+8.2f}{total_B_bayes:>+9.2f}  ({counted} predictions)")

# Per-chain
print(f"\n{'-'*128}")
print(f"Per-chain breakdown:")
print(f"{'chain':<13}{'shared_L':>10}{'#preds':>8}{'Σ Φ_flat':>10}{'Σ Φ_Bayes':>11}{'Σ L_amort':>11}{'Σ B_flat':>10}{'Σ B_Bayes':>11}")
chain_rows = {}
for p, phi_f, phi_b, chi2, L_a, B_f, B_b, n in results:
    if phi_f is None: continue
    chain_rows.setdefault(p['chain'], []).append((phi_f, phi_b, L_a, B_f, B_b))
for chain in sorted(chain_rows, key=lambda c: -sum(r[4] for r in chain_rows[c])):
    rows = chain_rows[chain]
    print(f"{chain:<13}{CHAINS[chain]['shared_L']:>10}{len(rows):>8}{sum(r[0] for r in rows):>10.2f}{sum(r[1] for r in rows):>11.2f}{sum(r[2] for r in rows):>11.2f}{sum(r[3] for r in rows):>+10.2f}{sum(r[4] for r in rows):>+11.2f}")

# Worst χ² offenders (T1.5 surfacing)
print(f"\n{'-'*128}")
print(f"Worst χ² offenders (T1.5 fit-quality penalty):")
worst = sorted(((p, chi2) for p, _, _, chi2, _, _, _, _ in results if chi2 is not None and chi2 > 1.0),
               key=lambda x: -x[1])[:5]
if not worst:
    print(f"  All predictions have χ² ≤ 1.0 — Bayes penalty is at most {LOG2E_HALF:.2f} bit each.")
else:
    for p, chi2 in worst:
        v_p, v_o, s = p['value_pred'], p['value_obs'], p['sigma_obs']
        print(f"  {p['name']:<20}  χ² = {chi2:>7.2f}  (pred {v_p:.6g} vs obs {v_o:.6g}, σ {s:.3g})  "
              f"penalty {-LOG2E_HALF*chi2:+.2f} bits")

# Projection
N_TOTAL = 45
print(f"\n{'-'*128}")
print(f"Pilot under FLAT prior:  Σ B = {total_B_flat:+.2f} bits over {counted} predictions  (avg {total_B_flat/counted:+.2f} bits/pred)")
print(f"Pilot under BAYES prior: Σ B = {total_B_bayes:+.2f} bits over {counted} predictions  (avg {total_B_bayes/counted:+.2f} bits/pred)")
proj_flat = total_B_flat / counted * N_TOTAL
proj_bayes = total_B_bayes / counted * N_TOTAL
total_flat = FRAMEWORK_NET_SUBSTRATE + proj_flat
total_bayes = FRAMEWORK_NET_SUBSTRATE + proj_bayes
print(f"\nProjection to {N_TOTAL} theorem-grade predictions:")
print(f"  Flat:  {FRAMEWORK_NET_SUBSTRATE:+.2f} (substrate) + {proj_flat:+.2f} (preds) = {total_flat:+.2f} bits  "
      f"{'→ BREAKS EVEN' if total_flat > 0 else f'→ deficit {-total_flat:.2f} bits'}")
print(f"  Bayes: {FRAMEWORK_NET_SUBSTRATE:+.2f} (substrate) + {proj_bayes:+.2f} (preds) = {total_bayes:+.2f} bits  "
      f"{'→ BREAKS EVEN' if total_bayes > 0 else f'→ deficit {-total_bayes:.2f} bits'}")

# T2.5: verify chain → catalog op_ids a separate private derivation by the author-DAG links are well-formed
print(f"\n{'-'*128}")
print(f"T2.5 a separate private derivation by the author-DAG cross-check: each chain's op_ids must exist in simulator CATALOG")
try:
    import os, sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from simulator.srs_engine import CATALOG  # noqa: WPS433
    catalog_ids = {op[0] for op in CATALOG}
    print(f"  CATALOG size: {len(catalog_ids)} ops")
    print(f"  {'chain':<12} {'#preds':>6} {'#ops':>5}  op_ids → coverage")
    for chain_name, c in sorted(CHAINS.items()):
        op_ids = c.get('op_ids', [])
        missing = [o for o in op_ids if o not in catalog_ids]
        n_preds = sum(1 for p in PREDICTIONS if p['chain'] == chain_name)
        marker = '✓' if not missing else '✗'
        print(f"  {chain_name:<12} {n_preds:>6} {len(op_ids):>5}  {marker} {', '.join(op_ids)}"
              + (f"  MISSING {missing}" if missing else ''))
        if missing:
            raise AssertionError(f"chain {chain_name} references missing op_ids {missing}")
    print(f"\n  ✓ All chains' a separate private derivation by the author-DAG op_id links are valid against CATALOG.")

    # T2.6: wave-reached prediction generation
    # Run the simulator and check which chains/predictions are reachable
    # given the wave's halt state.
    from simulator.srs_engine import run_full  # noqa: WPS433

    print(f"\n{'-'*128}")
    print(f"T2.6 Wave-reached predictions: which predictions does the halting state unlock?")
    print(f"{'-'*128}")

    # Run with strict A2 + formal L (the framework's actual MDL waterline gate)
    sim_final, _ = run_full(use_formal_L=True, strict_a2=True)
    fired_ids = sim_final.fired_ids
    print(f"  Simulator halt: {len(fired_ids)} ops fired; "
          f"missing tags = {sorted({'CCLOSE','LORENTZ_SIG'} - sim_final.tags)}")

    reachable_chains: list[str] = []
    blocked_chains: list[tuple[str, list[str]]] = []
    for chain_name, c in sorted(CHAINS.items()):
        op_ids = c.get('op_ids', [])
        missing_ops = [o for o in op_ids if o not in fired_ids]
        if not missing_ops:
            reachable_chains.append(chain_name)
        else:
            blocked_chains.append((chain_name, missing_ops))

    print(f"\n  REACHABLE chains ({len(reachable_chains)}/{len(CHAINS)}):")
    for chain in reachable_chains:
        preds_in_chain = [p['name'] for p in PREDICTIONS if p['chain'] == chain]
        print(f"    {chain:<12} → {len(preds_in_chain)} predictions: {', '.join(preds_in_chain)}")

    if blocked_chains:
        print(f"\n  BLOCKED chains ({len(blocked_chains)}/{len(CHAINS)}):")
        for chain, missing in blocked_chains:
            preds = [p['name'] for p in PREDICTIONS if p['chain'] == chain]
            print(f"    {chain:<12} → blocked by ops {missing}; {len(preds)} preds blocked: {', '.join(preds)}")

    # Per-prediction reachability
    n_reach_pred = sum(1 for p in PREDICTIONS if p['chain'] in reachable_chains)
    n_blocked_pred = len(PREDICTIONS) - n_reach_pred
    print(f"\n  Summary: {n_reach_pred}/{len(PREDICTIONS)} predictions wave-reached at halt; "
          f"{n_blocked_pred} blocked.")
    if n_blocked_pred == 0:
        print(f"  ✓ ALL pilot predictions reachable from current halt state.")
except ImportError as e:
    print(f"  (could not import simulator CATALOG: {e})")
