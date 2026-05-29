#!/usr/bin/env python3
"""
NA-4 Phase 3 Path A Step 2 — octonion substrate Hashimoto probe on srs.

CONTEXT
=======
Per an internal working note:
Path A Step 1 (bit-count probe) verified finite non-associative substrates
clear the A2-MDL waterline at N_hub.  Step 2 is the FIRST observable-level
test: does the octonion substrate (smallest finite candidate, |W|=16) admit
a Hashimoto-style spectral structure on srs that reproduces the framework's
three calibrating constraints?

CALIBRATING CONSTRAINTS (per Family D master doc §8 rule 2 + handoff §3)
========================================================================
The octonion-substrate Hashimoto, evaluated on the srs lattice, must reproduce:

  (a) Ramanujan saturation at BZ corner P:
      |h|² = k* − 1 = 2 with same eigenvalue multiplicity as associative.
      Associative baseline: 8 out of 12 eigenvalues of B(P) at |λ|² = 2.

  (b) NB walker survival α₁_bare = (q_NB)^(g−2) = (2/3)^8.
      This is a per-step combinatorial rate determined by (k*, g) = (3, 10).
      The octonion substrate preserves the srs lattice combinatorics
      (same |V|, |E|, k*, g) so (b) is preserved BY CONSTRUCTION.  The
      check is operational: confirm the octonion-Hashimoto spectral radius
      is k* − 1 = 2 (Perron eigenvalue of B(Γ)) — which is the spectral
      manifestation of (b).

  (c) Marginal-mode fraction = 5/12 (v_Higgs calibration).
      Associative: 5 of 12 eigenvalues of B(Γ) have |λ| = 1.
      Octonion lift: (5×8)/(8×12) = 40/96 expected if octonion structure
      preserves marginal sector dimension.

CONSTRUCTION
============
The octonion-valued Hashimoto operator B_𝕆 acts on the 12-dim space of
directed edges, each edge carrying an octonion-valued state ψ_e ∈ 𝕆.

Edge labeling:  each of the 6 undirected srs edges carries an imaginary
unit octonion {e_1, …, e_6}; reverse edges carry the negation.  Octonion
unit e_7 is unused (only 6 undirected edges).

Hashimoto entry: B_𝕆[e', e] = u_{e'} · ψ(c_{e'})  if e → e' is NB, else 0
where u_{e'} is the octonion label of the FORWARD edge and ψ(c) =
exp(2πi k·c) is the Bloch phase, embedded in the complex subalgebra
ℂ ≅ ⟨1, e_1⟩ ⊂ 𝕆.

Lift to ℝ^96 via left-multiplication representation L: 𝕆 → End_ℝ(ℝ^8).
Each octonion entry u acts on the 8-real-dim fibre via L_u(x) = u · x.
B_𝕆 lifted to M ∈ ℝ^{96×96} via block structure:
    M[(e', a), (e, b)] = L(B_𝕆[e', e])[a, b]

NON-TRIVIAL TEST
================
If u_e = 1 (trivial labeling) then M = B ⊗ I_8 (associative spectrum ×
8-fold degeneracy) — trivially preserves (a), (b), (c).  The probe
uses NON-TRIVIAL octonion labels to test whether the non-associative
structure preserves the calibrating spectrum.

Specifically: left-multiplication operators L_a and L_b on 𝕆 satisfy
L_a · L_b ≠ L_{a·b} when a, b don't lie in a common associative subalgebra
(per Moufang identity).  So a path-product over a closed NB-cycle involves
the FULL non-associative structure, not just the cycle's combinatorial
length.

VERDICT
=======
PASS  if all three calibrations match the associative baseline up to the
      8-fold ℝ^8 fibre degeneracy (a: ≥8×8 = 64 eigs at |λ|²=2 at P;
      b: spectral radius = k*−1 = 2; c: marginal-mode dim = 5×8 = 40 at Γ).
      Path A Step 2 closes positively; octonion substrate retained as
      Hashimoto-spectral candidate.  Proceed to Path A Step 3 (test
      observable-level reproduction of α₁ and c_λ on octonion substrate).

FAIL  if ANY of the three calibrations diverges from the expected lift.
      Octonion substrate ruled out as Hashimoto-spectral candidate; per
      handoff §3, next session targets sedenion substrate (|W|=32).

REFERENCES
==========
- `proofs/foundations/NA4_phase3_path_A_finite_substrate_bitcount_probe.py` (Step 1)
- `proofs/foundations/sector_P2_1_edge_octonion_formalization.py` (octonion algebra)
- `proofs/wave_engine/dark_5_12_spectral.py` (associative 5/12 baseline)
- `proofs/foundations/srs_p_point_algebra.py` (associative Hashimoto at P)
- `docs/theorems/theorem_substrate_feshbach_dark_corrections_master.md` §8 rule 2
  (v_Higgs calibration discipline)
"""

from __future__ import annotations
import os
import sys

import numpy as np

# Allow imports from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from proofs.common import find_bonds, N_ATOMS, K_STAR


TOL_RAM = 1e-8       # Ramanujan |λ|²=2 tolerance
TOL_MARGINAL = 1e-8  # |λ|=1 tolerance


# ============================================================================
# §1 Octonion algebra (Cayley-Dickson from ℍ)
# ============================================================================
# Re-implemented here for self-containment; identical to
# proofs/foundations/sector_P2_1_edge_octonion_formalization.py.

def H_mult(p, q):
    a0, a1, a2, a3 = p
    b0, b1, b2, b3 = q
    return np.array([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0,
    ])


def H_conj(p):
    return np.array([p[0], -p[1], -p[2], -p[3]])


def O_mult(p, q):
    """Octonion multiplication via Cayley-Dickson on ℍ × ℍ."""
    p_a, p_b = p[:4], p[4:]
    q_a, q_b = q[:4], q[4:]
    out_a = H_mult(p_a, q_a) - H_mult(H_conj(q_b), p_b)
    out_b = H_mult(q_b, p_a) + H_mult(p_b, H_conj(q_a))
    return np.concatenate([out_a, out_b])


def O_conj(p):
    """Octonion conjugate: (a + ℓb)* = a* − ℓb."""
    a, b = p[:4], p[4:]
    return np.concatenate([H_conj(a), -b])


def L_oct(u):
    """Left-multiplication operator L_u(x) = u · x as 8×8 real matrix.

    Columns are u · e_j for j=0..7 (action on basis of 𝕆 ≅ ℝ^8).
    """
    M = np.zeros((8, 8))
    for j in range(8):
        e_j = np.zeros(8); e_j[j] = 1.0
        M[:, j] = O_mult(u, e_j)
    return M


def octonion_unit(idx, sign=+1):
    """Return ±e_idx ∈ ℝ^8 (octonion-unit vector)."""
    u = np.zeros(8); u[idx] = float(sign)
    return u


# ============================================================================
# §2 Canonical edge labeling on srs
# ============================================================================
# 6 undirected edges of srs's primitive cell map to 6 imaginary octonion
# units {e_1, …, e_6}.  Reverse edges carry the negation.
#
# Bond list (from proofs.common.find_bonds()):
#   bond[0]:  0 -> 1  cell=(-1,-1,-1)    --- undirected edge (0,1)
#   bond[1]:  0 -> 2  cell=(-1,-1,-1)    --- undirected edge (0,2)
#   bond[2]:  0 -> 3  cell=(-1,-1,-1)    --- undirected edge (0,3)
#   bond[3]:  1 -> 0  cell=( 1, 1, 1)    --- reverse of (0,1)
#   bond[4]:  1 -> 2  cell=( 1, 0, 0)    --- undirected edge (1,2)
#   bond[5]:  1 -> 3  cell=( 0,-1, 0)    --- undirected edge (1,3)
#   bond[6]:  2 -> 0  cell=( 1, 1, 1)    --- reverse of (0,2)
#   bond[7]:  2 -> 1  cell=(-1, 0, 0)    --- reverse of (1,2)
#   bond[8]:  2 -> 3  cell=( 0, 0, 1)    --- undirected edge (2,3)
#   bond[9]:  3 -> 0  cell=( 1, 1, 1)    --- reverse of (0,3)
#   bond[10]: 3 -> 1  cell=( 0, 1, 0)    --- reverse of (1,3)
#   bond[11]: 3 -> 2  cell=( 0, 0,-1)    --- reverse of (2,3)
#
# Labelings tested by the probe (sensitivity check).
# Each LabelingScheme maps an undirected pair (a,b) with a<b to an octonion
# unit index (0=real, 1..7=imaginary).  Sign convention: +u for src<tgt,
# −u for tgt<src; reverse edges carry the negation.

LABELING_TRIVIAL = {  # u_e = +1 for all undirected edges (8-fold trivial lift)
    (0, 1): 0, (0, 2): 0, (0, 3): 0,
    (1, 2): 0, (1, 3): 0, (2, 3): 0,
}

LABELING_QUATERNIONIC = {  # u_e ∈ {e_1, e_2, e_3} — restricted to ℍ ⊂ 𝕆
    (0, 1): 1, (0, 2): 2, (0, 3): 3,
    (1, 2): 1, (1, 3): 2, (2, 3): 3,
}

LABELING_CANONICAL_OCTONION = {  # u_e ∈ {e_1,…,e_6} — full octonion
    (0, 1): 1, (0, 2): 2, (0, 3): 3,
    (1, 2): 4, (1, 3): 5, (2, 3): 6,
}

LABELING_FANO_TRIPLE_DRIVEN = {  # Adjacent edges per atom use a Fano triple
    # Fano triples e_1 e_2 = e_3, e_1 e_4 = e_5, e_1 e_6 = e_7
    # Atom 0 edges {(0,1),(0,2),(0,3)} → e_1, e_2, e_3 (Fano triple)
    # Atom 1 edges {(0,1),(1,2),(1,3)} → e_1, e_4, e_5 (must agree on (0,1)=e_1)
    # Atom 2 edges {(0,2),(1,2),(2,3)} → e_2, e_4, e_6 (must agree on (0,2)=e_2, (1,2)=e_4)
    # Atom 3 edges {(0,3),(1,3),(2,3)} → e_3, e_5, e_6 (must agree on (0,3)=e_3, (1,3)=e_5, (2,3)=e_6)
    (0, 1): 1, (0, 2): 2, (0, 3): 3,
    (1, 2): 4, (1, 3): 5, (2, 3): 6,
}


def make_edge_octonion_label(label_map):
    """Return a function (directed_bond) -> octonion ∈ ℝ^8 for the given label_map.

    Convention: reverse-edge label = octonion CONJUGATE of forward label.
    For imaginary units e_α (α ≥ 1): conj(e_α) = −e_α (negation).
    For real unit e_0 = 1: conj(1) = +1 (no flip).
    This is the natural unit-octonion phase convention: reversing the
    edge corresponds to inverting the phase, and u_α^(−1) = u_α^* / |u_α|² = u_α^*.
    """
    def edge_label(directed_bond):
        src, tgt, _cell = directed_bond
        a, b = (src, tgt) if src < tgt else (tgt, src)
        idx = label_map[(a, b)]
        u_forward = octonion_unit(idx, sign=+1)
        if src < tgt:
            return u_forward
        else:
            return O_conj(u_forward)
    return edge_label


# ============================================================================
# §3 Octonion-Hashimoto operator at a k-point
# ============================================================================

def build_octonion_hashimoto(k_frac, bonds, label_map):
    """Build the lifted octonion-Hashimoto M ∈ ℝ^{96×96} at wavevector k.

    Construction:
      - 12 directed edges, each carries an octonion-unit label ±e_α
        determined by `label_map` (undirected pair -> imaginary unit index).
      - NB transition condition: tgt(e) = src(e'), and e' is not the reverse of e.
      - Entry B_𝕆[e', e] = u_{e'} · ψ(c_{e'}) where ψ ∈ ℂ ⊂ 𝕆 via {1, e_1}.
      - Lift to M[(e', a), (e, b)] = L_{B_𝕆[e', e]}[a, b].

    At Γ (k=0), ψ = 1 — entries are pure octonion-unit labels.
    """
    n_E = len(bonds)
    M = np.zeros((8 * n_E, 8 * n_E))
    k = np.asarray(k_frac, dtype=float)
    e_1 = octonion_unit(1)
    edge_label = make_edge_octonion_label(label_map)

    for ip, (sp, tp, cp) in enumerate(bonds):
        u_p = edge_label(bonds[ip])
        # Bloch phase ψ(c_{e'}) = cos(θ) + sin(θ) e_1, θ = 2π k·c_{e'}
        theta = 2.0 * np.pi * np.dot(k, cp)
        c_real = float(np.cos(theta))
        c_imag = float(np.sin(theta))
        # ψ as octonion (real-part on e_0, imag-part on e_1)
        psi = c_real * octonion_unit(0) + c_imag * e_1
        # Phased label = u_{e'} · ψ
        u_phased = O_mult(u_p, psi)
        L_block = L_oct(u_phased)

        for ie, (se, te, ce) in enumerate(bonds):
            # NB condition: tgt(e) = src(e') and e' is not the reverse of e
            if te != sp:
                continue
            is_reverse = (sp == te and tp == se and
                          tuple(np.array(cp) + np.array(ce)) == (0, 0, 0))
            if is_reverse:
                continue
            # Write block at (ip-row, ie-col)
            M[ip*8:(ip+1)*8, ie*8:(ie+1)*8] += L_block

    return M


def build_complex_hashimoto(k_frac, bonds):
    """For sanity comparison: the framework's associative complex Hashimoto.

    Matches `proofs/foundations/srs_p_point_algebra.bloch_hashimoto`.
    """
    n_E = len(bonds)
    B = np.zeros((n_E, n_E), dtype=complex)
    k = np.asarray(k_frac, dtype=float)
    for ip, (sp, tp, cp) in enumerate(bonds):
        for ie, (se, te, ce) in enumerate(bonds):
            if te != sp:
                continue
            is_reverse = (sp == te and tp == se and
                          tuple(np.array(cp) + np.array(ce)) == (0, 0, 0))
            if is_reverse:
                continue
            phase = np.exp(2j * np.pi * np.dot(k, cp))
            B[ip, ie] += phase
    return B


# ============================================================================
# §4 Spectral classification
# ============================================================================

def classify_eigenvalues(evs):
    """Bucket eigenvalues by |λ|² into Perron / oscillatory / marginal / other.

    Returns dict with counts.
    """
    counts = {
        'perron':    0,  # |λ|² ≈ k*-1+1 = (k*-1)^? — for B at Γ, |λ|=k*-1=2 (largest real)
        'ramanujan': 0,  # |λ|² ≈ k*-1 = 2  (oscillatory)
        'marginal':  0,  # |λ| ≈ 1
        'zero':      0,
        'other':     0,
    }
    for ev in evs:
        absq = float(abs(ev)**2)
        if absq < TOL_MARGINAL:
            counts['zero'] += 1
        elif abs(absq - 4.0) < TOL_RAM:
            counts['perron'] += 1
        elif abs(absq - 2.0) < TOL_RAM:
            counts['ramanujan'] += 1
        elif abs(absq - 1.0) < TOL_MARGINAL:
            counts['marginal'] += 1
        else:
            counts['other'] += 1
    return counts


# ============================================================================
# §5 Probe — calibrations (a), (b), (c)
# ============================================================================

def run_labeling(label_name, label_map, bonds):
    """Run all three calibration tests under one labeling. Returns dict."""
    print("-" * 78)
    print(f"Labeling: {label_name}")
    print(f"  map: {label_map}")
    print("-" * 78)

    results = {}
    for kname, k in [('Γ', np.array([0.0, 0.0, 0.0])),
                     ('P', np.array([0.25, 0.25, 0.25]))]:
        M = build_octonion_hashimoto(k, bonds, label_map)
        evs = np.linalg.eigvals(M)
        c = classify_eigenvalues(evs)
        c['total'] = len(evs)
        c['spec_radius'] = float(max(abs(e) for e in evs))
        results[kname] = c
        moduli_sq = sorted([abs(e)**2 for e in evs], reverse=True)
        # Top-5 |λ|² + bin counts
        top5 = [f'{m2:.4f}' for m2 in moduli_sq[:5]]
        print(f"  {kname}:  perron={c['perron']:3d}  ram={c['ramanujan']:3d}  "
              f"marg={c['marginal']:3d}  other={c['other']:3d}  "
              f"|λ|_max={c['spec_radius']:.4f}  top5 |λ|²={top5}")

    cg = results['Γ']
    cP = results['P']

    expected_ram_at_P = 8 * 8     # 8 ramanujan eigs × 8 fibre
    expected_marginal_at_G = 5 * 8
    pass_a = cP['ramanujan'] == expected_ram_at_P
    pass_b = abs(cg['spec_radius'] - (K_STAR - 1)) < 1e-6
    pass_c = cg['marginal'] == expected_marginal_at_G

    print(f"  Calibrations:  (a) Ram at P {cP['ramanujan']:3d}/64  "
          f"{'PASS' if pass_a else 'FAIL'}  "
          f"(b) spec rad Γ={cg['spec_radius']:.4f}/{K_STAR-1}  "
          f"{'PASS' if pass_b else 'FAIL'}  "
          f"(c) Marg at Γ {cg['marginal']:3d}/40  "
          f"{'PASS' if pass_c else 'FAIL'}")
    print()
    return {
        'label_name': label_name,
        'Γ': cg, 'P': cP,
        'pass_a': pass_a, 'pass_b': pass_b, 'pass_c': pass_c,
        'pass_all': pass_a and pass_b and pass_c,
    }


def main():
    print("=" * 78)
    print("NA-4 Phase 3 Path A Step 2 — octonion Hashimoto on srs")
    print("=" * 78)
    print()

    bonds = find_bonds()
    print(f"srs lattice: |V|={N_ATOMS}, |E_directed|={len(bonds)}, k*={K_STAR}")
    print(f"Octonion lift: ℝ^{8*len(bonds)} (8 real fibres × 12 edges)")
    print()

    # --- Associative baseline (sanity check) -------------------------------
    print("-" * 78)
    print("Associative complex baseline (sanity check)")
    print("-" * 78)
    for name, k in [('Γ', np.array([0.0, 0.0, 0.0])),
                    ('P', np.array([0.25, 0.25, 0.25]))]:
        B_assoc = build_complex_hashimoto(k, bonds)
        evs_assoc = np.linalg.eigvals(B_assoc)
        c = classify_eigenvalues(evs_assoc)
        spec_radius = max(abs(e) for e in evs_assoc)
        print(f"  {name}:  spec radius = {spec_radius:.6f}   "
              f"(perron|λ|²=4: {c['perron']}, ram|λ|²=2: {c['ramanujan']}, "
              f"marg|λ|²=1: {c['marginal']})")
    print()
    print("Expected from theorem_dark_5_12_spectral and srs_p_point_algebra:")
    print("  Γ:  perron 1, marginal 5 (5/12), oscillatory complex 6")
    print("  P:  ramanujan 8, marginal 4")
    print()

    # --- Sensitivity sweep across labelings --------------------------------
    print("=" * 78)
    print("Octonion-substrate Hashimoto sensitivity sweep")
    print("=" * 78)
    print()
    print("(Expected lift: each associative eigenvalue acquires an 8-fold")
    print(" ℝ^8 fibre degeneracy.  Calibration preserved ⇒ ramanujan@P=64,")
    print(" marginal@Γ=40, spec_radius@Γ=2.)")
    print()

    sweep = [
        ('trivial (u_e = 1; sanity check)', LABELING_TRIVIAL),
        ('quaternionic (u_e ∈ {e_1, e_2, e_3} ⊂ ℍ)', LABELING_QUATERNIONIC),
        ('canonical octonion (u_e ∈ {e_1,…,e_6} ⊂ 𝕆)',
            LABELING_CANONICAL_OCTONION),
    ]
    sweep_results = []
    for label_name, label_map in sweep:
        sweep_results.append(run_labeling(label_name, label_map, bonds))

    # --- Verdict ----------------------------------------------------------
    print("=" * 78)
    print("PATH A STEP 2 VERDICT")
    print("=" * 78)
    print()
    print(f"  {'Labeling':<55s}  {'a':>4s}  {'b':>4s}  {'c':>4s}  {'all':>4s}")
    print("  " + "-" * 76)
    for r in sweep_results:
        print(f"  {r['label_name']:<55s}  "
              f"{'PASS' if r['pass_a'] else 'FAIL':>4s}  "
              f"{'PASS' if r['pass_b'] else 'FAIL':>4s}  "
              f"{'PASS' if r['pass_c'] else 'FAIL':>4s}  "
              f"{'PASS' if r['pass_all'] else 'FAIL':>4s}")
    print()

    # Read the structural finding off the sweep
    trivial = sweep_results[0]
    quaternionic = sweep_results[1]
    octonion = sweep_results[2]

    if trivial['pass_all'] and not octonion['pass_all']:
        print("STRUCTURAL FINDING:")
        print()
        print("  - Trivial lift (u_e = 1) PRESERVES calibration, as expected:")
        print("    the lifted Hashimoto is B ⊗ I_8 (8-fold degenerate version of")
        print("    associative spectrum) so Ramanujan-saturation at P and the")
        print("    5/12 marginal fraction at Γ inherit unchanged.")
        print()
        ass_q = quaternionic['pass_all']
        ass_o = octonion['pass_all']
        if ass_q and not ass_o:
            print("  - Quaternionic sublabeling (u_e ∈ ℍ ⊂ 𝕆) PRESERVES calibration:")
            print("    the labels lie in the ASSOCIATIVE subalgebra ℍ, so the path-")
            print("    products of left-multiplications commute with their algebra")
            print("    products (L_a L_b = L_{ab} on the 𝕆 fibre when a,b ∈ ℍ).")
            print("    The spectrum still factors and inherits associative calibration.")
            print()
            print("  - Full octonion labeling (u_e ∈ 𝕆 \\ ℍ for some edges) FAILS")
            print("    calibration on (a) Ramanujan at P and (c) marginal at Γ.")
            print("    Reason: L_a L_b ≠ L_{ab} when a,b don't lie in a common")
            print("    associative subalgebra (alternative law of octonions falls")
            print("    short of full associativity).  The path-product spectral")
            print("    structure is DISRUPTED by octonion non-associativity.")
        elif not ass_q and not ass_o:
            print("  - Both quaternionic and full octonion labelings FAIL — the")
            print("    8-fold degeneracy is broken by any non-trivial edge-labeling")
            print("    (even associative ℍ sub-labels).")
        elif ass_q and ass_o:
            print("  - All labelings preserve calibration — the result was")
            print("    insensitive to labeling choice (contradicts initial finding).")
            print("    Treat as anomalous; investigate.")
        print()
        if not ass_o:
            print("CONCLUSION: Octonion-substrate Hashimoto on srs does NOT preserve")
            print("            the framework's calibrating constraints under non-")
            print("            trivial labelings.  Per Family D master doc §8 rule 2")
            print("            (v_Higgs 5/12 calibration), octonion substrate is")
            print("            RULED OUT as a Hashimoto-spectral substrate replacement.")
            print()
            print("            Path A Step 2 closes NEGATIVE for octonion.")
            print("            Per scoping doc §5, next bounded step is sedenion")
            print("            substrate (|W|=32, |W|=16-imag); however sedenion has")
            print("            STRICTLY WEAKER algebra (power-associative only, zero")
            print("            divisors, no norm composition) and the non-associative")
            print("            structure is MORE pronounced — calibration is highly")
            print("            unlikely to survive.  See verdict doc for guidance on")
            print("            whether to attempt sedenion or to bound the negative.")
    elif octonion['pass_all']:
        print("VERDICT: PASS — octonion preserves calibration.  Proceed to Step 3.")
    else:
        print("VERDICT: anomalous (trivial lift also failed).  Investigate probe.")
    print()
    print("=" * 78)

    # Sentinel: trivial labeling MUST preserve calibration
    assert trivial['pass_all'], (
        "Sentinel failure: trivial labeling (u_e=1) should give B ⊗ I_8 "
        "and trivially preserve calibration."
    )
    print("Sentinel: trivial labeling preserves calibration ✓")
    return octonion['pass_all']


if __name__ == '__main__':
    main()
