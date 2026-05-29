"""
Gauge sector predictions — counting-first queries.

Covers:
- CKM matrix elements (V_us, V_cb, V_ub, plus unitarity-derived rest)
- Jarlskog invariant J_CKM
- Weak mixing angle sin²θ_W at unification (3/8)
- Unified gauge coupling α_GUT (1/24)
- Standard hypercharge assignments
"""

import math
import cmath
from fractions import Fraction
from simulator.srs_engine.kernel import CountingKernel
from match.pati_salam import PatiSalamUtility


# ============================================================================
# CKM MATRIX
# ============================================================================

def V_us(kernel=None):
    """V_us = k*²/(g·|V|) = 9/40 — Cabibbo angle. CHANNEL-SELECTED.

    Waterfilling-correct derivation: srs supports three structurally
    distinct substrate-counting MECHANISMS, called Levels in the framework
    (`predictions/V_us_derivation.md`, `theorem_A5b_level_prescription.md`):

        Level 1 ↔ channel `bare_amplitude`            — single NB walk (α_1_bare)
        Level 2 ↔ channel `srs_crystal_coupling_density` — pair-density on srs
        Level 3 ↔ channel `hashimoto_winding_sum`     — geometric series (V_cb)

    ALL THREE LEVELS ARE PHYSICALLY REALIZED — different observables
    occupy them:
      - Level 1 (`bare_amplitude`):           α_1_bare = (2/3)^8
      - Level 2 (`srs_crystal_coupling_density`): V_us = k*²/(g·|V|) = 9/40
      - Level 3 (`hashimoto_winding_sum`):    V_cb = Σ_n (k*-1/k*)^L = 256/6305

    V_us is the (1,2) CKM mixing magnitude — a coupling-pair density
    between off-diagonal quark species. Its substrate channel is the
    Level-2 `srs_crystal_coupling_density`, NOT Level 1 (which is a
    single-amplitude observable) or Level 3 (which is a winding-sum
    observable). channel_select picks Level 2 by channel-name match.

    The structural identification (V_us = coupling-pair density at
    Level 2) is now NAMED at the call site as
    `channel='srs_crystal_coupling_density'`. Previously this was
    implicit in the wrapper writing `Fraction(coupling_pairs, ...)`
    without justifying why that formula (vs Level 1 or Level 3) for
    V_us. Per `feedback_waterline_not_minimum_canonical_distinction.md`,
    above-waterline candidates in other channels are NOT discarded;
    they're physically realized as other observables.

    Memory record: nine Level-3 approaches to V_us were systematically
    falsified before the Level-2 mechanism was identified (per
    an internal note). Each falsified Level-3 approach
    was attempting to use the wrong channel for this observable — exactly
    the kind of channel-mismatch the explicit channel naming now prevents
    at the call site.

    Canonical proof: `proofs/flavor/vus_l2_density.py` (8 PASSED).
    """
    kernel = kernel or CountingKernel()
    # Compute each Level's substrate value (all are valid kernel queries;
    # they coexist above the waterline as physically realized counts).
    coupling_pairs = kernel.equiv_class_count('coupling_pair_per_girth_cycle')
    girth_slots = kernel.walk_count('girth_cycle_per_atom')
    n_atoms = kernel.orbit_count('lattice_atoms')

    candidates = [
        {
            'name': 'Level 1 (bare NB-walk amplitude)',
            'channel': 'bare_amplitude',
            'value': kernel.walk_count('nb_closed_at_girth'),  # (2/3)^8
        },
        {
            'name': 'Level 2 (srs coupling-pair density)',
            'channel': 'srs_crystal_coupling_density',
            'value': Fraction(coupling_pairs, girth_slots * n_atoms),  # 9/40
        },
        {
            'name': 'Level 3 (Hashimoto winding geometric sum)',
            'channel': 'hashimoto_winding_sum',
            'value': kernel.branch_measure('nb_walk_geometric_sum', length=9),  # 256/6305
        },
    ]

    # V_us's substrate definition: off-diagonal CKM coupling-pair density
    # on the srs crystal net (Level 2). The channel is fixed BEFORE the
    # kernel queries are made — not selected post-hoc by closest-to-PDG.
    selected = kernel.channel_select(
        candidates,
        channel='srs_crystal_coupling_density',
    )
    return selected['value']


def V_cb(kernel=None):
    """V_cb = α_1_bare / (1 − α_1_bare) = 256/6305. CHANNEL-SELECTED.

    Waterfilling-correct derivation: srs supports three Levels of substrate
    counting (same trinity as V_us), each physically realized for a
    distinct CKM element:

        Level 1 ↔ channel `bare_amplitude`               (single NB walk)
        Level 2 ↔ channel `srs_crystal_coupling_density` (pair density)
        Level 3 ↔ channel `hashimoto_winding_sum`        (geometric Σ over n)

    V_cb = (2,3) CKM mixing magnitude = b → c transition amplitude. The
    substrate channel is the multi-winding Hashimoto-walker geometric
    sum (Level 3): per `predictions/V_cb.py` and
    `proofs/flavor/vcb_hashimoto_bfs.py`, V_cb arises as a sum over all
    winding numbers n ≥ 1 of girth-cycle NB walk survival:
        V_cb = Σ_{n≥1} (α_1_bare)^n = α_1_bare / (1 − α_1_bare).

    Level 1 (single winding, α_1_bare alone) and Level 2 (pair density)
    are physically realized for OTHER observables (α_1_bare, V_us); they
    are not discarded — they're above-waterline counts for those
    observables. channel_select picks Level 3 by channel-name match.
    """
    kernel = kernel or CountingKernel()
    coupling_pairs = kernel.equiv_class_count('coupling_pair_per_girth_cycle')
    girth_slots = kernel.walk_count('girth_cycle_per_atom')
    n_atoms = kernel.orbit_count('lattice_atoms')

    candidates = [
        {
            'name': 'Level 1 (bare NB-walk amplitude)',
            'channel': 'bare_amplitude',
            'value': kernel.walk_count('nb_closed_at_girth'),  # (2/3)^8
        },
        {
            'name': 'Level 2 (srs coupling-pair density)',
            'channel': 'srs_crystal_coupling_density',
            'value': Fraction(coupling_pairs, girth_slots * n_atoms),  # 9/40
        },
        {
            'name': 'Level 3 (Hashimoto winding geometric sum)',
            'channel': 'hashimoto_winding_sum',
            'value': kernel.branch_measure('nb_walk_geometric_sum', length=9),  # 256/6305
        },
    ]

    selected = kernel.channel_select(
        candidates,
        channel='hashimoto_winding_sum',
    )
    return selected['value']


def V_ub(kernel=None):
    """V_ub = Σ_{m≥2} (2/3)^(6m+2) / (1 - (2/3)^(6m+2)) ≈ 3.767e-3. DERIVED.

    Counting query (M1 multi-cycle walk-rep sum, UNIQUE-THEOREM-GRADE for
    amplitude per commit 753f4cf, 2026-04-30):

      α_m = ((k*-1)/k*)^L_eff with L_eff = m·g - 2(m-1)·s_seam - n_fixed
      V_ub = Σ_{m=2}^{m_max} α_m / (1 - α_m)

    where:
      k*       = 3   (substrate)
      g        = 10  (girth)
      s_seam   = 2   (CAS-verified seam length on m=2 hosts)
      n_fixed  = 2   (1 b-type + 1 u-type pinning)
      m_max    = 10  (truncation; converges to ~14 digits)

    For m=2, L_eff = 2·10 - 0 - 2 = 18... but the framework's "L = 6m+2"
    closed form uses (m·g - 2(m-1)·s_seam - n_fixed) which gives 6m+2 only
    when g=10 and s_seam=2 (substituting: 10m - 4(m-1) - 2 = 6m + 2 ✓).
    """
    kernel = kernel or CountingKernel()
    k = kernel.substrate.K_STAR
    g = kernel.substrate.GIRTH
    s_seam = 2  # CAS-verified for m=2 host (proofs/flavor/hashimoto_16cycle_decomposition)
    n_fixed = 2  # 1 b-type + 1 u-type causal-state pinning
    m_max = 10  # truncation; series converges geometrically
    survival = Fraction(k - 1, k)
    total = Fraction(0)
    for m in range(2, m_max + 1):
        L_eff = m * g - 2 * (m - 1) * s_seam - n_fixed
        a = survival ** L_eff
        total += a / (1 - a)
    return total


# ============================================================================
# CKM remainder via 3x3 unitarity (Type-4 closure)
# ============================================================================

def _ckm_via_unitarity(kernel=None):
    """Build the full 3x3 CKM matrix from V_us, V_cb, V_ub + δ_CP via unitarity.

    DERIVED via standard CKM parametrization (Wolfenstein-equivalent):
      - λ = V_us
      - A = V_cb / λ²
      - ρ + iη from V_ub and δ_CP

    Returns a 3x3 numpy array of complex entries (CKM matrix).
    """
    import numpy as np

    kernel = kernel or CountingKernel()

    # Take the three independent magnitudes from substrate
    v_us = float(V_us(kernel))      # 9/40
    v_cb = float(V_cb(kernel))      # 256/6305
    v_ub = float(V_ub(kernel))      # 128/32805

    # CP phase from polytope dihedral — call cp_phases at site (no duplication)
    from .cp_phases import delta_CP_CKM as _delta_CP_CKM
    delta_cp = math.radians(_delta_CP_CKM(kernel))

    # Standard PDG parametrization with three rotation angles + phase
    # sin(θ_12) = V_us, sin(θ_13) = V_ub, sin(θ_23) = V_cb (approximately)
    s12 = v_us
    s13 = v_ub
    s23 = v_cb
    c12 = math.sqrt(1 - s12 ** 2)
    c13 = math.sqrt(1 - s13 ** 2)
    c23 = math.sqrt(1 - s23 ** 2)

    # CKM matrix in the standard parametrization
    V = np.zeros((3, 3), dtype=complex)
    V[0, 0] = c12 * c13                                                       # V_ud
    V[0, 1] = s12 * c13                                                       # V_us
    V[0, 2] = s13 * cmath.exp(-1j * delta_cp)                                 # V_ub
    V[1, 0] = -s12 * c23 - c12 * s23 * s13 * cmath.exp(1j * delta_cp)         # V_cd
    V[1, 1] = c12 * c23 - s12 * s23 * s13 * cmath.exp(1j * delta_cp)          # V_cs
    V[1, 2] = s23 * c13                                                       # V_cb
    V[2, 0] = s12 * s23 - c12 * c23 * s13 * cmath.exp(1j * delta_cp)          # V_td
    V[2, 1] = -c12 * s23 - s12 * c23 * s13 * cmath.exp(1j * delta_cp)         # V_ts
    V[2, 2] = c23 * c13                                                       # V_tb
    return V


def V_cd(kernel=None):
    """V_cd via 3x3 CKM unitarity. DERIVED."""
    V = _ckm_via_unitarity(kernel)
    return abs(V[1, 0])


def V_cs(kernel=None):
    """V_cs via 3x3 CKM unitarity. DERIVED."""
    V = _ckm_via_unitarity(kernel)
    return abs(V[1, 1])


def V_td(kernel=None):
    """V_td via 3x3 CKM unitarity. DERIVED."""
    V = _ckm_via_unitarity(kernel)
    return abs(V[2, 0])


def V_ts(kernel=None):
    """V_ts via 3x3 CKM unitarity. DERIVED."""
    V = _ckm_via_unitarity(kernel)
    return abs(V[2, 1])


def V_tb(kernel=None):
    """V_tb via 3x3 CKM unitarity. DERIVED."""
    V = _ckm_via_unitarity(kernel)
    return abs(V[2, 2])


def V_ud(kernel=None):
    """V_ud via 3x3 CKM unitarity. DERIVED."""
    V = _ckm_via_unitarity(kernel)
    return abs(V[0, 0])


def J_CKM(kernel=None):
    """Jarlskog invariant J_CKM via standard parametrization. DERIVED.

    J_CKM = c_12 c_23 c_13² s_12 s_23 s_13 sin(δ_CP)

    where s_ij, c_ij are the CKM mixing angle sines/cosines, derived from
    V_us, V_cb, V_ub. Returns ≈ 3.16e-5.
    """
    kernel = kernel or CountingKernel()
    s12 = float(V_us(kernel))
    s23 = float(V_cb(kernel))
    s13 = float(V_ub(kernel))
    c12 = math.sqrt(1 - s12 ** 2)
    c23 = math.sqrt(1 - s23 ** 2)
    c13 = math.sqrt(1 - s13 ** 2)
    from .cp_phases import delta_CP_CKM as _delta_CP_CKM
    delta_cp = math.radians(_delta_CP_CKM(kernel))
    return c12 * c23 * c13 ** 2 * s12 * s23 * s13 * math.sin(delta_cp)


# ============================================================================
# GAUGE COUPLINGS
# ============================================================================

def sin2_theta_W(kernel=None):
    """sin²θ_W = 3/8 at M_unif — weak mixing angle (Family 4 PS rep theory).

    Counting query: Σ Tr(T_3L²) / Σ Tr(Q²) on PS reps (4,2,1) + (4̄,1,2).
    """
    kernel = kernel or CountingKernel()
    return PatiSalamUtility.sin2_theta_W()  # 3/8


def alpha_GUT(kernel=None):
    """α_GUT = 1/24 — unified gauge coupling at M_unif (Family 8).

    Counting query: 1 / cl6_fock_label_slots = 1/(2^k* · k*) = 1/24.
    """
    kernel = kernel or CountingKernel()
    return PatiSalamUtility.alpha_GUT(kernel)  # 1/24


def hypercharge(particle_label, kernel=None):
    """Standard hypercharge assignments Y from PS embedding.

    Args:
        particle_label: SM particle label ('q_L', 'u_R', 'd_R', 'l_L',
                       'e_R', 'higgs')
    """
    return PatiSalamUtility.hypercharge_Y(particle_label)
