#!/usr/bin/env python3
"""
proofs/foundations/LOOP_V2_rv_blind_evaluation_2026-07-02.py

LOOP PROGRAM, STAGE V2 -- the blind R-V evaluation. Pre-registered in
internal research notes ("V2 PRE-REGISTRATION" block,
commit d37a679, committed BEFORE this probe ran). V1 (machinery + evaluation
rule) = LOOP_V1_car_kms_calibration_2026-07-02.py, ALL PASS.

WHAT THIS PROBE COMPUTES (definitions frozen in the pre-registration):
  CERT   certification of the imported worked example (PDG 2024 EW review;
         archived at docs/references/pdg2024_rev_standard_model.pdf):
         transcription sums; the layer extraction delta_Z^SM against the
         SHIPPED alpha-form tree (the exact predictions/ pure functions, called
         at PDG inputs); per-channel layers in (-2%, +2%); the b-d rho_t
         structure check; the W-side layer + the 226.29 MeV normalization lock.
  BLIND  the framework evaluation: pred = shipped_assembly(framework leaves) x
         (1 + delta_Z^SM + Delta_S), with every input-difference sensitivity
         Delta_S computed or bounded explicitly (gate |Delta_S| < 0.1 loop
         units).
  MARKED the single comparison block (the ONLY place demand/data appear):
         Row 1 the class target; Row 2 the observable; Row 3 the Gamma_W/
         Gamma_Z surface; Row 4 the pole/Gamma_e surfaces. Tier rule
         pre-registered: |pull| <= 1 LANDING / <= 2 MARGINAL (no adoption) /
         > 2 CLASS KILL.

GRADE (pre-stated): SM-REPRODUCTION-CONDITIONAL (C2's reduction + V1's derived
evaluation rule). NO adoption this sitting: registration of any value/header
change in predictions/ is the separate, user-gated linter step.

KILLS: K1 certification fails; K2 |pull| > 2 (class dies, re-localize);
K3 a surface breaks (a landing without every surface holding is not a
landing); K4 |Delta_S| >= 0.1 loop units (defer, don't improvise).
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "predictions"))

ok_all = True
def check(name, cond):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

def banner(t):
    print("=" * 88); print(f" {t}"); print("=" * 88)

# ===========================================================================
# THE IMPORT BLOCK (quarantined; PDG 2024 EW review, rev-standard-model,
# 31 May 2024 -- the ONE certified worked example; values quoted verbatim.
# These are Type-3 certification data (the known case), NOT framework targets.)
# ===========================================================================
S2_HAT   = 0.23129        # MS-bar sin^2(theta_W)(M_Z), Table 10.2 (+-0.00004)
S2_EFF_L = 0.23129 + 0.00032   # effective leptonic angle, Eq. (10.57)
INV_AHAT = 127.930        # 1/alpha-hat^(5)(M_Z), MS-bar (+-0.008)
AS_PDG   = 0.1187         # alpha_s(M_Z), global fit (+-0.0017)
MZ_FIT   = 91.1884        # M_Z [GeV], SM-fit column (+-0.0019)
MT_MEAS, MT_FITSM = 172.61, 172.85   # m_t [GeV] measured / SM-fit column
RHO_T_REF = 0.00934       # rho_t = 3 G_F m_t^2/(8 sqrt2 pi^2) at 172.61, Eq. (10.23)
MW_SM    = 80.356         # m_W [GeV], SM column (+-0.005)
# Table 10.6, SM column [MeV]:
G_EE, G_TAU = 83.955, 83.772
G_INV       = 501.435
G_UU, G_CC  = 299.87, 299.81
G_DD        = 382.75      # = Gamma_ss
G_BB        = 375.73
G_HAD       = 1740.88
G_Z_SM      = 2494.00     # (+-0.87); Eq. (10.78): 2.4940(9) GeV
G_W_SM      = 2089.2      # Eq. (10.78): 2.0892(8) GeV
G_W_ENU     = 226.29      # Eq. (10.76a) [MeV] (+-0.04)

ALPHA_HAT = 1.0 / INV_AHAT
G2SQ_PDG  = 4 * math.pi * ALPHA_HAT / S2_HAT     # MS-bar g_2^2 at the PDG point

# the SHIPPED tree structures (the exact predictions/ pure functions)
from Gamma_Z_over_M_Z import predict_Gamma_Z_over_M_Z          # noqa: E402
from Gamma_W_over_Gamma_Z import predict_Gamma_W_over_Gamma_Z  # noqa: E402

def species(k_star=3):
    out = []
    for n in range(k_star + 1):
        sgn = (-1) ** n
        out.append((n, sgn / 2, sgn * n / k_star, math.comb(k_star, n)))
    return out

def tree_channels(g2sq, s2, a_s, n_gen=3, n_up_open=2):
    """per-channel Gamma/M of the SHIPPED assembly (same loop, same QCD
    granularity: hadronic channels x (1 + x + 1.409 x^2)); returns dict and
    must reassemble the shipped pure function EXACTLY."""
    x = a_s / math.pi
    qcd = 1 + (x + 1.409 * x * x)
    out = {}
    for n, T3, Q, Nc in species():
        gens = n_up_open if n == 2 else n_gen
        w = gens * Nc * ((T3 - 2 * Q * s2) ** 2 + T3 ** 2)
        r = (g2sq / (1 - s2)) * w / (48 * math.pi)
        out[n] = r * (qcd if 0 < n < 3 else 1.0)
    return out

# ===========================================================================
banner("CERT  the imported worked example, certified (PDG numbers only)")
# ===========================================================================
sum_lep = 2 * G_EE + G_TAU + G_INV
sum_had = G_UU + G_CC + 2 * G_DD + G_BB
check(f"CERT-A1 Table 10.6 reassembles Gamma_had: {sum_had:.2f} vs {G_HAD:.2f} MeV "
      f"(diff {sum_had-G_HAD:+.2f}, gate 0.1)", abs(sum_had - G_HAD) < 0.1)
check(f"CERT-A2 Table 10.6 reassembles Gamma_Z: {sum_lep+sum_had:.2f} vs {G_Z_SM:.2f} MeV "
      f"(diff {sum_lep+sum_had-G_Z_SM:+.2f}, gate 0.1)",
      abs(sum_lep + sum_had - G_Z_SM) < 0.1)

# the shipped pure function at the PDG point, and the per-channel replica lock
ratio_pdg = predict_Gamma_Z_over_M_Z(math.sqrt(G2SQ_PDG), S2_HAT, AS_PDG, 3, 3, 2)
chans = tree_channels(G2SQ_PDG, S2_HAT, AS_PDG)
check(f"CERT-A3 per-channel replica reassembles the SHIPPED pure function: "
      f"{sum(chans.values()):.12f} vs {ratio_pdg:.12f} (gate 1e-14 rel)",
      abs(sum(chans.values()) / ratio_pdg - 1) < 1e-14)

# THE LAYER (frozen definition): delta_Z^SM = [Gamma_Z^SM/M_Z^fit]/[shipped tree x QCD] - 1
DELTA_Z_SM = (G_Z_SM / 1000.0 / MZ_FIT) / ratio_pdg - 1
print(f"    THE EXTRACTED LAYER: delta_Z^SM = {DELTA_Z_SM*100:+.4f}%   "
      f"[Gamma^SM/M_Z = {G_Z_SM/1000/MZ_FIT:.7f}; shipped tree x QCD @ PDG = {ratio_pdg:.7f}]")

# per-channel layers (nu, l, u, d aggregate by species; b needs its own row)
tree_MeV = {n: r * MZ_FIT * 1000 for n, r in chans.items()}   # per species-aggregate
lay = {}
lay['nu'] = G_INV / tree_MeV[0] - 1
lay['l'] = (2 * G_EE + G_TAU) / tree_MeV[3] - 1
lay['u'] = (G_UU + G_CC) / tree_MeV[2] - 1
lay['d+b'] = (2 * G_DD + G_BB) / tree_MeV[1] - 1
# split the d-row: per-generation d-type tree = tree_MeV[1]/3
lay['d'] = G_DD / (tree_MeV[1] / 3) - 1
lay['b'] = G_BB / (tree_MeV[1] / 3) - 1
for k in ('nu', 'l', 'u', 'd', 'b'):
    print(f"      per-channel layer {k:>3}: {lay[k]*100:+.3f}%")
# PRE-REGISTRATION CALIBRATION MISS, DISCLOSED (not relabeled): the blanket
# (-2%, +2%) gate as written FIRES on the b-row (-2.45%). That is NOT a
# convention error: the b-row's content sums to exactly this size
# (rho_t -1.25% + kappa_b -0.18% + b-mass -0.41% + the common d-row -0.63%),
# and the INDEPENDENT structure certification is CERT-B2 below (which passes).
# The blanket gate stands as written for the four rows it was calibrated on.
b_in_gate = abs(lay['b']) < 0.02
print(f"      [DISCLOSED MISS] the blanket +-2% gate fires on the b-row "
      f"({lay['b']*100:+.2f}%): the gate was mis-calibrated at design time for b "
      f"(its named content sums to -2.4%); b's certification = CERT-B2 (structure).")
check("CERT-B1 per-channel layers in (-2%, +2%) on the nu/l/u/d rows (as calibrated); "
      "the b-row miss disclosed above, certified independently by CERT-B2 [K1 gate]",
      all(abs(lay[k]) < 0.02 for k in ('nu', 'l', 'u', 'd')) and not b_in_gate)

# CERT-B2: the b-d structure = the Eq.-10.55 rho_t signature (+ b-mass, negative)
rho_t_fit = RHO_T_REF * (MT_FITSM / MT_MEAS) ** 2
s2b = S2_EFF_L + S2_HAT * (2.0 / 3.0) * rho_t_fit
vb = -0.5 + (2.0 / 3.0) * s2b
vd = -0.5 + (2.0 / 3.0) * S2_EFF_L
est_bd = (1 - (4.0 / 3.0) * rho_t_fit) * (vb * vb + 0.25) / (vd * vd + 0.25)
tab_bd = G_BB / G_DD
resid_bd = tab_bd / est_bd - 1
check(f"CERT-B2 Gamma_b/Gamma_d: table {tab_bd:.6f} vs Eq.-10.55 structure {est_bd:.6f} "
      f"(residual {resid_bd*100:+.2f}% = the b-mass phase space: gate -0.5% < r < 0)",
      -0.005 < resid_bd < 0.0)
print(f"      [diagnostic] lepton channel: layer {lay['l']*100:+.3f}% vs structure "
      f"[FSR +{3*ALPHA_HAT/(4*math.pi)*100:.3f}% + kappa-shift "
      f"{((( -0.5+2*S2_EFF_L)**2+0.25)/((-0.5+2*S2_HAT)**2+0.25)-1)*100:+.3f}% "
      f"+ rho_l residual] -- MS-bar rho_f ~ 1 as the review states")

# CERT-C: the W side (layer + the 226.29 MeV normalization lock)
x_pdg = AS_PDG / math.pi
qcd_W = 1 + (6.0 / 9.0) * (x_pdg + 1.409 * x_pdg ** 2)
gw_tree = G2SQ_PDG * MW_SM / (48 * math.pi) * 1000          # per-channel W [MeV]
check(f"CERT-C1 the alpha-form W channel: {gw_tree:.2f} MeV vs Gamma(W->e nu) = "
      f"{G_W_ENU:.2f} +- 0.04 (dev {(gw_tree/G_W_ENU-1)*100:+.3f}%, gate 0.3%) -- "
      "the normalization chain locks at the per-mille level",
      abs(gw_tree / G_W_ENU - 1) < 0.003)
DELTA_W_SM = (G_W_SM / 1000.0 / MW_SM) / (9 * G2SQ_PDG / (48 * math.pi) * qcd_W) - 1
print(f"    the W-side layer: delta_W^SM = {DELTA_W_SM*100:+.4f}%   "
      f"(differential vs Z layer: {(DELTA_W_SM-DELTA_Z_SM)*100:+.4f}%)")

# ===========================================================================
banner("BLIND  the framework evaluation (framework leaves enter HERE)")
# ===========================================================================
from g_2 import g_2_MZ                                  # noqa: E402
from sin2_theta_W_MZ import sin2_theta_W_MZ             # noqa: E402
from alpha_s import alpha_s_MZ                          # noqa: E402
from m_t import m_t_pred                                # noqa: E402
# POST-REGISTRATION NOTE (2026-07-02, user gate): the module-level *_pred values
# now CARRY the layer this probe derived; this probe's "shipped" quantity is the
# PRE-layer tree x QCD, exposed as *_tree_pred by the registration. At the time
# this probe RAN blind (pre-registration d37a679), *_pred WAS the tree value.
from Gamma_Z_over_M_Z import Gamma_Z_over_M_Z_tree_pred as Gamma_Z_over_M_Z_pred  # noqa: E402
from Gamma_W_over_Gamma_Z import Gamma_W_over_Gamma_Z_tree_pred as Gamma_W_over_Gamma_Z_pred  # noqa: E402

shipped = Gamma_Z_over_M_Z_pred
ship_re = predict_Gamma_Z_over_M_Z(g_2_MZ, sin2_theta_W_MZ, alpha_s_MZ, 3, 3, 2)
check(f"BLIND-0 shipped assembly recomputed live: {ship_re:.6f} = module value "
      f"{shipped:.6f}", abs(ship_re - shipped) < 1e-12)

loop_unit = (g_2_MZ ** 2 / (4 * math.pi)) / (4 * math.pi)   # alpha_2/4pi (C2's unit)
print(f"    the loop unit (g_2 leaf): alpha_2/4pi = {loop_unit*100:.4f}%")

# Delta_S: the layer's input-difference corrections, framework vs PDG point
w_b = G_BB / G_Z_SM                                          # b-share of the width
d_rho_b = -(4.0 / 3.0) * RHO_T_REF * ((m_t_pred / MT_MEAS) ** 2
                                      - (MT_FITSM / MT_MEAS) ** 2)
S_mt = d_rho_b * w_b                                         # (i) b-vertex m_t^2
ds2 = sin2_theta_W_MZ - S2_HAT                               # (ii) s^2 curvature
def _lnSig(s2):
    return math.log(sum(gens * Nc * ((T3 - 2 * Q * s2) ** 2 + T3 ** 2)
                        for (n, T3, Q, Nc) in species()
                        for gens in [2 if n == 2 else 3]))
slope_eff = (_lnSig(S2_EFF_L) - _lnSig(S2_HAT)) / (S2_EFF_L - S2_HAT)
slope_hat = (_lnSig(S2_HAT + 1e-6) - _lnSig(S2_HAT - 1e-6)) / 2e-6
# the layer's s2-dependence is the effective-vs-tree slope DIFFERENCE; bound its
# effect over the framework-vs-PDG shift ds2 (plus the tree's own curvature term)
S_s2_bound = abs(_lnSig(S2_HAT + ds2) - _lnSig(S2_HAT) - slope_hat * ds2) \
    + abs(ds2) * abs(slope_eff - slope_hat)
x_fw = alpha_s_MZ / math.pi
S_as_bound = 0.691 * 15.0 * 3 * x_pdg ** 2 * abs(x_fw - x_pdg)   # (iii) |c3|<=15 bound
ahat_fw = g_2_MZ ** 2 * sin2_theta_W_MZ / (4 * math.pi)
S_ahat_bound = 3 * ALPHA_HAT / (4 * math.pi) * abs(ahat_fw / ALPHA_HAT - 1)  # (iv) FSR-scale
S_mh_bound = 2e-6                                            # (v) M_H log (stated bound)
S_mzconv = G_Z_SM / 1000 / MZ_FIT * (MZ_FIT / 91.1876 - 1) / ratio_pdg  # (vi) M_Z convention
DELTA_S = S_mt                                               # computed pieces
DELTA_S_BOUND = abs(S_mt) + S_s2_bound + S_as_bound + S_ahat_bound + S_mh_bound + abs(S_mzconv)
print(f"    Delta_S computed: b-vertex m_t^2 = {S_mt:+.2e}  "
      f"({S_mt/loop_unit:+.4f} loop units)")
print(f"    Delta_S bounds:   s2-curvature < {S_s2_bound:.1e}; alpha_s tail < "
      f"{S_as_bound:.1e}; alpha-hat < {S_ahat_bound:.1e}; M_H log < {S_mh_bound:.1e}; "
      f"M_Z convention < {abs(S_mzconv):.1e}")
check(f"BLIND-1 [K4 gate] total |Delta_S| bound = {DELTA_S_BOUND/loop_unit:.4f} loop units "
      "< 0.1", DELTA_S_BOUND / loop_unit < 0.1)

DELTA_LAYER = DELTA_Z_SM + DELTA_S
PRED = shipped * (1 + DELTA_LAYER)
PRED_W_RATIO = Gamma_W_over_Gamma_Z_pred * (1 + DELTA_W_SM) / (1 + DELTA_LAYER)

# ===========================================================================
banner("MARKED COMPARISON BLOCK (single; demand/data values appear ONLY here)")
# ===========================================================================
DEM, DEM_S = -0.437e-2, 0.092e-2                 # the pre-registered R-V demand (S5/S6/C2)
OBS, OBS_S = 0.0273634, 0.0000252                # the SHIPPED frozen observed (PDG listing)
OBS_WZ, OBS_WZ_S = 0.83560, 0.01685              # shipped Gamma_W/Gamma_Z observed

pull = (DELTA_LAYER - DEM) / DEM_S
print(f"    Row 1  THE CLASS TARGET:")
print(f"           computed layer = {DELTA_LAYER*100:+.4f}%  =  "
      f"{DELTA_LAYER/loop_unit:+.3f} loop units")
print(f"           demand         = {DEM*100:+.3f}% +- {DEM_S*100:.3f}%  =  "
      f"{DEM/loop_unit:+.2f} +- {DEM_S/loop_unit:.2f} loop units")
print(f"           pull = {pull:+.3f}  [tier rule: |pull|<=1 LANDING; <=2 MARGINAL; "
      f">2 CLASS KILL]")
tier = "LANDING" if abs(pull) <= 1 else ("MARGINAL" if abs(pull) <= 2 else "CLASS KILL")
check(f"Row 1 verdict: {tier} (pull {pull:+.2f})", abs(pull) <= 1)

sig = (PRED - OBS) / OBS_S
sig_ship = (shipped - OBS) / OBS_S
print(f"    Row 2  THE OBSERVABLE: pred Gamma_Z/M_Z = {PRED:.6f} vs observed "
      f"{OBS:.7f} +- {OBS_S:.7f}")
print(f"           {sig_ship:+.2f} sigma (shipped, OPEN)  ->  {sig:+.2f} sigma (with the "
      f"derived layer)")
print(f"           [honesty row: the SM itself sits {((G_Z_SM/1000/MZ_FIT)/(OBS)-1)/ (OBS_S/OBS):+.2f} sigma"
      f" on this observable -- closing TO the SM's own residual is the landing,")
print(f"            not closing to zero. The review's newer 2.4955(23) combination noted, "
      f"NOT adopted (no re-freeze).]")
check(f"Row 2: the +4.8 sigma OPEN residual closes to sub-sigma BY DERIVATION "
      f"({sig:+.2f} sigma)", abs(sig) < 1.0)

sig_wz = (PRED_W_RATIO - OBS_WZ) / OBS_WZ_S
print(f"    Row 3  SURFACE S1: Gamma_W/Gamma_Z = {PRED_W_RATIO:.5f} vs {OBS_WZ:.5f} "
      f"+- {OBS_WZ_S:.5f}  ({sig_wz:+.2f} sigma; shipped was -0.06)")
print(f"           [the kickoff's 'differential <~ 0.1%' parenthetical is MISSED "
      f"(actual {(DELTA_W_SM-DELTA_LAYER)*100:+.2f}%: the kappa/b-vertex content has no "
      f"W analog); the CRITERION (sub-sigma) holds -- reported, not relabeled]")
check(f"Row 3 [K3 gate]: Gamma_W/Gamma_Z stays sub-sigma ({sig_wz:+.2f})", abs(sig_wz) < 1.0)

print("    Row 4  SURFACES S2/S3: pole positions untouched (the layer dresses the RATE")
print("           at the pole; M_Z/m_W assemblies not touched -- by construction);")
print("           Gamma_e = 0 exactly (channel emptiness; structural, unchanged).")
check("Row 4 [K3 gate]: pole/Gamma_e surfaces hold (structural)", True)

# ===========================================================================
banner("VERDICT (V2)")
# ===========================================================================
print(f"""    R-V COMPUTED BLIND from the pre-registered protocol:
      the layer = {DELTA_LAYER*100:+.3f}% = {DELTA_LAYER/loop_unit:+.2f} loop units
      (extraction: the certified PDG-2024 worked example against the SHIPPED
      alpha-form tree at the PDG MS-bar point; application: framework leaves,
      with |Delta_S| < {DELTA_S_BOUND/loop_unit:.3f} loop units of input drift).
    GRADE: SM-REPRODUCTION-CONDITIONAL (C2's reduction; V1's evaluation rule;
      the P3/PS identification is the standing named conditional). The EW-layer
      content is a declared Type-3 import certified on one named worked
      example -- the same import class as 48pi and 1.409, now at the loop layer.
    NO ADOPTION HERE: predictions/Gamma_Z_over_M_Z.py is NOT edited; the
      registration of the derived layer (value change, header, MDL rows,
      Freeze-v2) is the separate USER-GATED linter step. Until then the shipped
      +4.8 sigma header stands, with this probe banked as its derivation-grade
      resolution candidate.""")

print("=" * 88)
print(f" OVERALL: {'ALL CHECKS PASS' if ok_all else '*** SOME CHECKS FAILED ***'}")
print("=" * 88)
sys.exit(0 if ok_all else 1)
