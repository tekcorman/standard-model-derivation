# ============================================================
# PARAMETER: Γ_Z / M_Z — the Z resonance width fraction
# ============================================================
# Companion of predictions/Gamma_W_over_Gamma_Z.py (F4 S3, 2026-07-02).
# EW radiative layer REGISTERED 2026-07-02 (user gate) — see below.
#
# --- OBSERVED VALUE ------------------------------------------
# Value:       Γ_Z/M_Z = 0.0273634 ± 0.0000252   (±0.092%)
# Source:      PDG 2024: Γ_Z = 2.4952 ± 0.0023 GeV, M_Z = 91.1876 ± 0.0021 GeV
# PDG edition: 2024  (the EW-review's newer Γ_Z = 2.4955 ± 0.0023 combination
#              is NOTED, not adopted — no re-freeze of the S3 comparison)
#
# --- PREDICTED VALUE -----------------------------------------
# Value:       0.027350   (tree×QCD 0.027484 × (1 + δ_Z), δ_Z = −0.4864%)
# Deviation:   −0.049% relative, −0.55σ_PDG — **Clause 8c PASS**, equal to
#              the SM's own −0.53σ residual on this observable (the honest
#              content of SM-REPRODUCTION grade: the framework closes TO
#              the SM, not to zero).
#
# --- GRADE (parameter_linter.md vocabulary) ------------------
# STRUCTURAL-DERIVATION-CONDITIONAL per Clause 9(9b) + Clause 10c (the
# golden-rule 1/(48π) and the EW layer are both continuum-loop Type-3
# content; K-rationality BROKEN and acknowledged; the row can NEVER be
# promoted past bridge-conditional until the native loop derivation
# lands). Clause 8: PASS (8c, −0.55σ ≤ 1σ_PDG) — upgraded 2026-07-02 from
# FAIL(8d) by the LOOP PROGRAM's derived layer, NOT by assembly tuning:
#   the layer was named in the frozen assembly BEFORE comparison (S3),
#   its class was selected by pre-registered probe (C2, 2188fbe), its
#   evaluation rule derived (V1, a5287f4: the retarded vacuum loop is the
#   forced evaluation — the arrow selects it; thermality = statistics
#   only), and its value computed BLIND under a pre-registered tier rule
#   with all falsification surfaces gated (V2, d37a679: pull −0.54,
#   LANDING; Γ_W/Γ_Z stays sub-σ; poles untouched; Γ_e = 0). The +4.8σ
#   residual served as the loop program's pre-registered falsification
#   target and the class LANDED. USER GATE 2026-07-02.
# Resolution of the 2026-07-02 user ruling ("no Type-3 import ever closes
# Γ_Z/M_Z — importing a value that moves a value is an oxymoron"): the
# ruling barred a VALUE import with no derivation chain (the retracted
# f878f82 pattern). What is registered here is the OUTPUT of the derived
# loop class evaluated per the derived rule, with the import surface
# reduced to the standard-EW loop-formula content certified on one named
# worked example — the same Type-3 class as 48π/1.409 already in this
# assembly. The row grade stays bridge-conditional exactly because that
# import surface is nonzero. Native closure (the interacting sector
# coupling / walk↔Fock dictionary at theorem grade) remains OPEN in
# incomplete_equations_todo.md §7 — the grade ceiling, not this row's
# numerical status.
#
# A5(b) CLOSURE 2026-07-05 (A5b_closure_kahler_dirac_reduction, LOCK): the
# P3/PS current identification that the "standard-EW content" reduction was
# conditional on is now DERIVED (the physical current IS the Clifford γ^μ;
# Lorentz-locked Cl(3,1); a₄ locks 2/2/0). That lifts the current-ID
# conditionality — but NOT the grade ceiling: the loop COEFFICIENT stays the
# Type-3 import (48π/1.409 class), and native closure (the interacting sector
# coupling) is still OPEN (§7). No value moves; the −0.55σ number is unchanged.
#
# --- WIDTH-SIDE DARK: NONE, BY THEOREM -----------------------
# F4_S2b_width_ratio_dark_lemma (CAS): real Perron dressing shifts Γ and M
# together (Γ/M invariant exactly); complex-pole shell reading is
# stability-excluded. Applying dark here is FORBIDDEN, not omitted.
# (Clause 10a: the registered EW layer is NOT a dark dressing — it is the
# rate-side loop content the R3 rate clause assigns to widths; pole
# positions keep their static dressings untouched, so M_Z's own oblique
# residual (+6σ-class) is NOT touched by this registration.)
#
# --- ASSEMBLY (frozen S3 tree×QCD + the REGISTERED derived layer) -----
# tree × own-α_s QCD (frozen 2026-07-02 BEFORE comparison) × (1 + δ_Z).
# δ_Z BUNDLES the entire S3 stated-not-applied family in one certified
# object (EW ρ_f/s̄²_eff/Z-bb̄ ≈ −0.4% dominant; per-channel QED FSR;
# fermion-mass phase space; QCD 3rd order) — extracted from the certified
# PDG-2024 worked example against THIS file's own tree at the PDG MS̄
# point, applied at framework leaves with input drift |ΔS| < 0.012 loop
# units (single source: predictions/ew_width_layer.py; probe:
# proofs/foundations/LOOP_V2_rv_blind_evaluation_2026-07-02.py).
# The in-file gates assert BOTH the pre-layer residual's PRESENCE (the
# 10b anti-stale tripwire: the tree×QCD assembly still shows its +4.8σ
# deficit) AND the post-layer Clause-8 PASS — a silent vanish of either
# fails loudly.
#
# --- DERIVED FORMULA -----------------------------------------
# Γ_Z/M_Z = [ g₂²/c² · Σ_open(s²) / (48π) ]
#           × [1 + f_had·(α_s/π + 1.409(α_s/π)²)] × (1 + δ_Z)
#   Σ_open, content, closure: identical Cl(6)-Fock read as the companion
#   file (Q = (−1)ⁿ n/k*, T₃ = (−1)ⁿ/2, N(n) = C(k*,n); top closed by own
#   m_t). M_Z's VALUE does not enter (massless-channel ratio); only
#   channel openness uses the framework's own m_t ≫ M_Z/2. G_F unused
#   (Clause 10d audit: nothing here consumes G_F or v). Non-circular.
#
# --- INPUTS --------------------------------------------------
# symbol   | value      | status    | predictions/ file            | meaning
# ---------|------------|-----------|------------------------------|-----------------
# g₂(M_Z)  | 0.65175    | [derived] | g_2.py                       | SU(2)_L coupling at M_Z
# s²(M_Z)  | 0.23125    | [derived] | sin2_theta_W_MZ.py           | weak mixing angle at M_Z
# α_s(M_Z) | 0.1179     | [derived] | alpha_s.py                   | strong coupling at M_Z
# m_t      | 172.41     | [derived] | m_t.py                       | top mass (channel closure only)
# M_Z      | 91.2039    | [derived] | M_Z.py                       | (closure comparison only)
# k*       | 3          | [derived] | k_star.py                    | coordination (Cl(6) Fock)
# n_gen    | 3          | [derived] | R3_observer_c3_generation.py | generation count
# δ_Z      | −0.004864  | [derived-bridge] | ew_width_layer.py     | the registered EW layer
#          |            |           |                              | ([external] certified PDG-2024
#          |            |           |                              | worked example inside the leaf)
# Formula-structure constants (48π, 2Q s², 1.409) = the declared Type-3 import.

import functools
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from g_2 import g_2_MZ                                  # noqa: E402
from sin2_theta_W_MZ import sin2_theta_W_MZ            # noqa: E402
from alpha_s import alpha_s_MZ                          # noqa: E402
from M_Z import M_Z_GeV                                 # noqa: E402
from m_t import m_t_pred                                # noqa: E402
from k_star import predict_k_star                       # noqa: E402
from d_spatial import predict_d_spatial                 # noqa: E402
from R3_observer_c3_generation import R3_observer_c3_generation_pred  # noqa: E402
import ew_width_layer as _ewl                           # noqa: E402

# --- IMPLEMENTATION ------------------------------------------
_k_star = predict_k_star(predict_d_spatial())
_n_gen = int(R3_observer_c3_generation_pred)
assert m_t_pred > M_Z_GeV / 2, "top closure premise failed"
_n_up_open = _n_gen - 1


def _implementation():
    s2 = sin2_theta_W_MZ
    tot, had = 0.0, 0.0
    for n in range(_k_star + 1):
        sgn = (-1) ** n
        T3, Q, Nc = sgn / 2, sgn * n / _k_star, math.comb(_k_star, n)
        gens = _n_up_open if n == 2 else _n_gen
        w = gens * Nc * ((T3 - 2 * Q * s2) ** 2 + T3 ** 2)
        tot += w
        if 0 < n < _k_star:
            had += w
    x = alpha_s_MZ / math.pi
    tree = (g_2_MZ ** 2 / (1 - s2)) * tot / (48 * math.pi)
    tree_qcd = tree * (1 + (had / tot) * (x + 1.409 * x * x))
    return tree_qcd * (1 + _ewl.ew_width_layer_Z_pred), tree_qcd, tree


# --- PURE FUNCTION -------------------------------------------
@functools.lru_cache(maxsize=None)
def predict_Gamma_Z_over_M_Z(g2_MZ, s2_MZ, a_s_MZ, k_star, n_gen, n_up_open):
    """
    Γ_Z/M_Z from the frozen tree×QCD golden-rule assembly (Type-3 structure)
    with all numerical content from framework reads. PRE-LAYER value: the
    registered EW layer is applied by predict_Gamma_Z_over_M_Z_dressed.

    Parameters
    ----------
    g2_MZ : float      SU(2)_L coupling at M_Z (framework)
    s2_MZ : float      sin²θ_W at M_Z (framework RG endpoint)
    a_s_MZ : float     α_s at M_Z (framework)
    k_star : int       coordination number (Cl(6) Fock rank)
    n_gen : int        generation count
    n_up_open : int    open up-type generations at M_Z

    Returns
    -------
    float : tree×QCD Γ_Z/M_Z (pre-layer)
    """
    tot, had = 0.0, 0.0
    for n in range(k_star + 1):
        sgn = (-1) ** n
        T3, Q, Nc = sgn / 2, sgn * n / k_star, math.comb(k_star, n)
        gens = n_up_open if n == 2 else n_gen
        w = gens * Nc * ((T3 - 2 * Q * s2_MZ) ** 2 + T3 ** 2)
        tot += w
        if 0 < n < k_star:
            had += w
    x = a_s_MZ / math.pi
    return (g2_MZ ** 2 / (1 - s2_MZ)) * tot / (48 * math.pi) \
        * (1 + (had / tot) * (x + 1.409 * x * x))


@functools.lru_cache(maxsize=None)
def predict_Gamma_Z_over_M_Z_dressed(tree_qcd_ratio, ew_layer_Z):
    """the registered full prediction: tree×QCD × (1 + δ_Z)."""
    return tree_qcd_ratio * (1 + ew_layer_Z)


# --- ANTI-DRIFT WELD -----------------------------------------
# the layer leaf carries a REPLICA of this file's tree (needed to extract
# δ_Z at the PDG MS̄ point without a circular import); assert the replica
# and this file's pure function agree EXACTLY at both evaluation points.
assert abs(_ewl._tree_ratio_alpha_form(_ewl.G2SQ_PDG, _ewl.S2_HAT_PDG, _ewl.ALPHA_S_FIT)
           / predict_Gamma_Z_over_M_Z(math.sqrt(_ewl.G2SQ_PDG), _ewl.S2_HAT_PDG,
                                      _ewl.ALPHA_S_FIT, 3, 3, 2) - 1) < 1e-14, \
    "layer-leaf tree replica drifted from the shipped assembly (PDG point)"
assert abs(_ewl._tree_ratio_alpha_form(g_2_MZ ** 2, sin2_theta_W_MZ, alpha_s_MZ)
           / predict_Gamma_Z_over_M_Z(g_2_MZ, sin2_theta_W_MZ, alpha_s_MZ, 3, 3, 2)
           - 1) < 1e-14, \
    "layer-leaf tree replica drifted from the shipped assembly (framework point)"

# --- VALIDATION ----------------------------------------------
Gamma_Z_over_M_Z_obs = 2.4952 / 91.1876
Gamma_Z_over_M_Z_sigma = Gamma_Z_over_M_Z_obs * math.sqrt(
    (0.0023 / 2.4952) ** 2 + (0.0021 / 91.1876) ** 2)
Gamma_Z_over_M_Z_pred, Gamma_Z_over_M_Z_tree_pred, _tree_only = _implementation()

if __name__ == "__main__":
    impl = Gamma_Z_over_M_Z_pred
    pure = predict_Gamma_Z_over_M_Z_dressed(
        predict_Gamma_Z_over_M_Z(g_2_MZ, sin2_theta_W_MZ, alpha_s_MZ,
                                 _k_star, _n_gen, _n_up_open),
        _ewl.ew_width_layer_Z_pred)
    dev = impl / Gamma_Z_over_M_Z_obs - 1
    sig = (impl - Gamma_Z_over_M_Z_obs) / Gamma_Z_over_M_Z_sigma
    sig_tree = (Gamma_Z_over_M_Z_tree_pred - Gamma_Z_over_M_Z_obs) / Gamma_Z_over_M_Z_sigma
    print("Γ_Z/M_Z — Z width fraction (F4 S3 assembly + the LOOP-program EW layer,")
    print("          registered 2026-07-02 by user gate; Clause 9b bridge-conditional)")
    print(f"  Implementation: {impl:.6f}   (tree×QCD {Gamma_Z_over_M_Z_tree_pred:.6f}; "
          f"δ_Z = {_ewl.ew_width_layer_Z_pred*100:+.4f}%)")
    print(f"  Pure function:  {pure:.6f}")
    assert abs(impl - pure) < 1e-12, f"Mismatch: {impl} vs {pure}"
    print(f"  Observed:       {Gamma_Z_over_M_Z_obs:.7f} ± {Gamma_Z_over_M_Z_sigma:.7f}")
    print(f"  Deviation:      {dev*100:+.3f}%  ({sig:+.2f}σ_PDG)  [pre-layer: "
          f"{sig_tree:+.2f}σ — the located deficit, still present in the tree]")
    # honest gates:
    # (1) the 10b anti-stale tripwire — the PRE-layer assembly must still show
    #     its located +4.8σ-class deficit; if it silently vanishes, re-audit:
    assert sig_tree > 3.0, ("pre-layer residual vanished — the S3 accounting is stale; "
                            "re-audit before trusting the layer")
    # (2) the registered layer must match the V2-banked value (regression guard):
    assert abs(_ewl.ew_width_layer_Z_pred - (-0.4864e-2)) < 2e-5, \
        "δ_Z drifted from the V2-banked value — re-run the V2 probe before shipping"
    # (3) Clause 8c: the dressed prediction is sub-σ:
    assert abs(sig) < 1.0, ("left the Clause-8c band — data or leaf moved; "
                            "re-audit the registration")
    print("OK: Clause 8c PASS (−0.55σ) at bridge-conditional grade; the pre-layer")
    print("    deficit (+4.8σ-class) remains asserted-present (10b tripwire); native")
    print("    derivation of the layer stays OPEN (todo §7 — the grade ceiling).")
