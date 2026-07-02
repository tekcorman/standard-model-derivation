#!/usr/bin/env python3
"""
Edge-algebra zoo enumeration at directed edge (Task B of saturated-symmetry-zoo project).

Methodology — saturated symmetries cooled (per memory 2026-05-06+2 + commits
80f8c2c, f1e395c, 30b4bd7), parallel to Task A vertex local algebra
(`sector_local_algebra_zoo_audit.py`, commit 2c2a624).

EDGE LAYER STRUCTURE
====================

At each directed edge of the substrate's Cayley graph, the observer has a
local 2-state structure (qubit-like): the edge can be "occupied" or
"unoccupied" by the substrate's toggle event. The edge's local algebra
acts on this 2D Hilbert.

The framework's theorem-grade choice (theorem_g2_edge_qubit_su2.md):
  Cl(0, 2) ≅ ℍ (4-dim, associative, normed division).
  ℍ's Sp(1) × Sp(1) = Spin(4) action via left × right multiplication
  gives SU(2)_L × SU(2)_R (Pati-Salam factor).

Under saturated zoo: enumerate all 2-letter-CAR-compatible algebras at the
edge layer + their automorphism groups.

CANDIDATES AT EDGE LAYER
========================

CLIFFORD FAMILY at edge (varying signature p+q ≤ small):
  Cl(0,1) ≅ ℂ                — 2-dim, normed division.
  Cl(1,0) ≅ ℝ ⊕ ℝ            — 2-dim, NOT division.
  Cl(0,2) ≅ ℍ                — 4-dim, normed division. THE FRAMEWORK'S CHOICE.
  Cl(1,1) ≅ M_2(ℝ)           — 4-dim, NOT division.
  Cl(2,0) ≅ M_2(ℝ)           — 4-dim, NOT division.
  Cl(0,3) ≅ ℍ ⊕ ℍ            — 8-dim, NOT division.
  Cl(0,4) ≅ M_2(ℍ)           — 16-dim, NOT division (ℍ-valued matrices).

Hurwitz hard gate at edge: only Cl(0,1)=ℂ and Cl(0,2)=ℍ are normed
division. Higher-dim Cl members (Cl(0,3) etc.) have zero divisors.

CAYLEY-DICKSON TOWER at edge:
  ℝ (d_CD=0, 1-dim).
  ℂ ≅ Cl(0,1) (d_CD=1, 2-dim).
  ℍ ≅ Cl(0,2) (d_CD=2, 4-dim).
  𝕆 (d_CD=3, 8-dim) — alternative non-associative; doesn't naturally embed
    as 2D-qubit-acting algebra but enters the zoo as "alternative edge
    algebra" at depth 3.
  sedenion (d_CD=4): 16-dim, loses normed division + alternativity.

The Cayley-Dickson edge-tower OVERLAPS with low-dim Cl family:
  ℂ = Cl(0,1), ℍ = Cl(0,2). After ℍ, the families diverge (Cl(0,3) ≠ 𝕆).

TITS-FREUDENTHAL MAGIC SQUARE at edge:
  Same magic-square Lie algebras as Task A apply at the edge × edge or
  edge × vertex tensor combinations. Specifically of interest:
    ℍ ⊗ {ℝ, ℂ, ℍ, 𝕆} → sp(3), su(6), so(12), E_7
    These describe edge-paired symmetry structures that enter the
    saturated zoo.

DOMINANT EDGE-LAYER RETENTION
==============================

Per framework's theorem_g2_edge_qubit_su2.md: Cl(0,2) ≅ ℍ is theorem-grade
at the dominant slice. Sp(1) × Sp(1) action gives SU(2)_L × SU(2)_R, the
Pati-Salam left/right factor.

Other edge-layer alternatives are plurally retained at A2-T waterline but
suppressed by either Hurwitz hard gates (non-division algebras) or by
relation-cost factors at framework scale.

DAG: pure edge-algebra menu enumeration. No new framework structure.
"""

import math


# ----------------------------------------------------------------------------
# Encoding primitives (matching freq-weighted audit conventions)
# ----------------------------------------------------------------------------

def L_elias(m):
    """Elias-gamma encoding cost for positive integer m."""
    if m == float('inf'):
        return 1.0
    if m < 1:
        return float('inf')
    return 1 + 2 * math.floor(math.log2(m))


def L_M_clifford_signature(p, q):
    """
    L for Cl(p, q): specify p, q signature.
    """
    return L_elias(p + 1) + L_elias(q + 1)


def L_M_cayley_dickson(d_CD):
    base = L_elias(d_CD + 1)
    if d_CD >= 4:
        # Sedenion zero divisors / lost properties specifier
        base += L_elias(84)
    return base


# ----------------------------------------------------------------------------
# Algebra dimension and properties
# ----------------------------------------------------------------------------

def cliff_dim_signature(p, q):
    return 2 ** (p + q)


def cliff_division_status(p, q):
    """
    Cl(p, q) is a division algebra iff:
      Cl(0,0) = R, Cl(0,1) = C, Cl(0,2) = H.
    Otherwise NOT a division algebra (has zero divisors / matrix-algebra structure).
    """
    if (p, q) == (0, 0):
        return 'R div'
    if (p, q) == (0, 1):
        return 'C div'
    if (p, q) == (0, 2):
        return 'H div'
    return 'NOT div'


def cliff_normed_division(p, q):
    """Cl(p,q) normed-division: only (0,0)=R, (0,1)=C, (0,2)=H."""
    return (p, q) in {(0, 0), (0, 1), (0, 2)}


# ----------------------------------------------------------------------------
# Edge-layer alphabet and relation length
# ----------------------------------------------------------------------------

def edge_alphabet_clifford(p, q):
    """Number of Clifford generators."""
    return p + q


def edge_alphabet_cd(d_CD):
    """Cayley-Dickson at depth d_CD: 2^d_CD basis (or imaginary units)."""
    return 2 ** d_CD


def max_L_r_clifford():
    """Cl: 2-letter relations."""
    return 2


def max_L_r_cd(d_CD):
    if d_CD <= 2:
        return 2
    if d_CD == 3:
        return 3
    return d_CD + 1


def max_L_r_magic(d1, d2):
    return max(max_L_r_cd(d1), max_L_r_cd(d2))


def magic_square_dim(d1, d2):
    return (2 ** d1) * (2 ** d2)


# ----------------------------------------------------------------------------
# Frequency factor
# ----------------------------------------------------------------------------

def freq_factor_local(alphabet, max_L_r, N):
    if N <= 0 or alphabet < 2:
        return float('inf')
    return math.log2(N) - max_L_r * math.log2(alphabet)


def N_attest_local(alphabet, max_L_r):
    if alphabet < 2:
        return 1
    return alphabet ** max_L_r


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main():
    N_hub = 10 ** 60

    print("=" * 105)
    print(" Edge-algebra zoo at directed edge — saturated retention at N_hub = 10^60")
    print(" (Task B of saturated symmetry zoo + cooling cascade project)")
    print("=" * 105)
    print()

    rows = []

    # ---- CLIFFORD FAMILY at edge (low-dim, varying signature) ----
    print(" CLIFFORD FAMILY at edge — low-dim, varying signature p + q:")
    print()
    print(f"   {'algebra':<28} {'p':>2} {'q':>2} {'dim':>5} {'iso':<20} {'L(M)':>6} "
          f"{'L_r':>4} {'N_attest':>10} {'ff@10^60':>10} {'div?':<9} {'verdict':>10}")
    print("   " + "-" * 110)
    cl_signatures = [
        (0, 0, 'R'),
        (1, 0, 'R⊕R'),
        (0, 1, 'C'),
        (2, 0, 'M_2(R)'),
        (1, 1, 'M_2(R) split'),
        (0, 2, 'H'),                    # FRAMEWORK'S CHOICE
        (3, 0, 'M_2(R)⊕M_2(R)'),
        (0, 3, 'H⊕H'),
        (2, 2, 'M_2(R)⊗M_2(R)=M_4(R)'),
        (4, 0, 'M_4(R)'),
        (0, 4, 'M_2(H)'),
    ]
    for p, q, iso in cl_signatures:
        dim = cliff_dim_signature(p, q)
        L_M = L_M_clifford_signature(p, q)
        L_r = max_L_r_clifford()
        alphabet = max(edge_alphabet_clifford(p, q), 2)
        n_att = N_attest_local(alphabet, L_r)
        ff = freq_factor_local(alphabet, L_r, N_hub)
        div = cliff_division_status(p, q)
        verdict = 'FREQ-OK' if ff >= 0 else 'SUPPRESSED'

        # Mark framework's choice
        marker = ' ★' if (p, q) == (0, 2) else ''

        rows.append({
            'class': 'Clifford-edge',
            'name': f'Cl({p},{q})' + marker,
            'p': p,
            'q': q,
            'dim': dim,
            'iso': iso,
            'L_M': L_M,
            'L_r': L_r,
            'alphabet': alphabet,
            'N_attest': n_att,
            'ff': ff,
            'division': cliff_normed_division(p, q),
            'verdict': verdict,
        })
        print(f"   {f'Cl({p},{q})' + marker:<28} {p:>2} {q:>2} {dim:>5} {iso:<20} "
              f"{L_M:>6.1f} {L_r:>4} {n_att:>10} {ff:>10.1f} {div:<9} {verdict:>10}")
    print()
    print("   ★ = framework's theorem-grade dominant edge algebra (theorem_g2_edge_qubit_su2.md).")
    print()

    # ---- CAYLEY-DICKSON TOWER at edge ----
    print(" CAYLEY-DICKSON TOWER at edge:")
    print()
    print(f"   {'algebra':<22} {'d_CD':>4} {'dim':>5} {'L(M)':>6} {'L_r':>4} "
          f"{'N_attest':>10} {'ff@10^60':>10} {'normed div?':<14} {'verdict':>10}")
    print("   " + "-" * 95)
    cd_data = [
        (0, 'R',         'normed div'),
        (1, 'C',         'normed div'),
        (2, 'H',         'normed div'),
        (3, 'O',         'normed div alt'),    # normed division + alternative
        (4, 'sedenion S', 'NOT normed div'),
        (5, 'trigintaduonion', '—'),
    ]
    for d_CD, name, prop in cd_data:
        dim = 2 ** d_CD
        L_M = L_M_cayley_dickson(d_CD)
        L_r = max_L_r_cd(d_CD)
        alphabet = max(edge_alphabet_cd(d_CD), 2)
        n_att = N_attest_local(alphabet, L_r)
        ff = freq_factor_local(alphabet, L_r, N_hub)
        verdict = 'FREQ-OK' if ff >= 0 else 'SUPPRESSED'

        rows.append({
            'class': 'Cayley-Dickson-edge',
            'name': name,
            'd_CD': d_CD,
            'dim': dim,
            'L_M': L_M,
            'L_r': L_r,
            'alphabet': alphabet,
            'N_attest': n_att,
            'ff': ff,
            'division': prop == 'normed div' or prop == 'normed div alt',
            'verdict': verdict,
        })
        print(f"   {name:<22} {d_CD:>4} {dim:>5} {L_M:>6.1f} {L_r:>4} "
              f"{n_att:>10} {ff:>10.1f} {prop:<14} {verdict:>10}")
    print()

    # ---- MAGIC SQUARE at edge layer (ℍ-paired) ----
    print(" TITS-FREUDENTHAL MAGIC SQUARE — ℍ-paired (edge layer × second factor):")
    print()
    print(f"   {'magic square':<20} {'dim':>5} {'Lie algebra':<18} {'L(M)':>6} "
          f"{'L_r':>4} {'N_attest':>10} {'ff@10^60':>10} {'verdict':>10}")
    print("   " + "-" * 100)
    magic_h_pairs = [
        ('H ⊗ R', 2, 0, 'sp(3) (21)'),
        ('H ⊗ C', 2, 1, 'su(6) (35)'),
        ('H ⊗ H', 2, 2, 'so(12) (66)'),
        ('H ⊗ O', 2, 3, 'E_7 (133)'),
    ]
    for label, d1, d2, lie in magic_h_pairs:
        dim = magic_square_dim(d1, d2)
        L_M = L_M_cayley_dickson(d1) + L_M_cayley_dickson(d2) + 2.0
        L_r = max_L_r_magic(d1, d2)
        alphabet = max((2 ** d1) * (2 ** d2), 2)
        n_att = N_attest_local(alphabet, L_r)
        ff = freq_factor_local(alphabet, L_r, N_hub)
        verdict = 'FREQ-OK' if ff >= 0 else 'SUPPRESSED'

        rows.append({
            'class': 'Magic-edge',
            'name': label,
            'lie': lie,
            'dim': dim,
            'L_M': L_M,
            'L_r': L_r,
            'alphabet': alphabet,
            'N_attest': n_att,
            'ff': ff,
            'verdict': verdict,
        })
        print(f"   {label:<20} {dim:>5} {lie:<18} {L_M:>6.1f} {L_r:>4} "
              f"{n_att:>10} {ff:>10.1f} {verdict:>10}")
    print()

    n_total = len(rows)
    n_freq_ok = sum(1 for r in rows if r['verdict'] == 'FREQ-OK')
    print(f"   Total enumerated: {n_total}.  FREQ-OK at N_hub: {n_freq_ok}.  SUPPRESSED: {n_total - n_freq_ok}.")
    print()

    # ---- COOLING CASCADE ----
    print("=" * 105)
    print(" COOLING CASCADE — per-system N_attest thresholds (smallest first)")
    print("=" * 105)
    print()
    print(f"   {'system':<28} {'class':<22} {'N_attest':>10} {'log₂':>8}")
    print("   " + "-" * 75)
    rows_sorted = sorted(rows, key=lambda r: r['N_attest'])
    for r in rows_sorted:
        log_attest = math.log2(r['N_attest']) if r['N_attest'] > 1 else 0
        print(f"   {r['name']:<28} {r['class']:<22} {r['N_attest']:>10} {log_attest:>8.2f}")
    print()
    print(" All retained at framework scale 10^60 (log₂ ≈ 200).")
    print()

    # ---- VERDICT ----
    print("=" * 105)
    print(" SATURATED EDGE-ALGEBRA ZOO at N_hub")
    print("=" * 105)
    print()
    print(" Clifford family at edge (signature p+q ≤ 4): 11 members enumerated.")
    print("   - Cl(0,0) = ℝ trivial.")
    print("   - Cl(0,1) = ℂ: 2-dim normed division.")
    print("   - Cl(0,2) = ℍ: 4-dim normed division. ★ FRAMEWORK'S DOMINANT EDGE")
    print("     ALGEBRA per theorem_g2_edge_qubit_su2.md (theorem-grade).")
    print("   - Cl(2,0), Cl(1,1) = M_2(ℝ): 4-dim NOT division. Subdominant.")
    print("   - Higher-signature Cl: progressively non-division. All retained.")
    print()
    print(" Cayley-Dickson tower at edge: ℝ, ℂ, ℍ, 𝕆, sedenion, trigintaduonion.")
    print("   - ℝ, ℂ, ℍ: shared with Cl-family at low dim.")
    print("   - 𝕆 at edge: 8-dim, alternative non-associative. Subdominant zoo entry.")
    print("   - sedenion+: lose normed-division + alternativity properties.")
    print()
    print(" ℍ-paired magic-square Lie algebras at edge:")
    print("   - ℍ ⊗ ℝ = sp(3) (21-dim).")
    print("   - ℍ ⊗ ℂ = su(6) (35-dim).")
    print("   - ℍ ⊗ ℍ = so(12) (66-dim).")
    print("   - ℍ ⊗ 𝕆 = E_7 (133-dim) — exceptional Lie algebra at edge × 𝕆.")
    print()
    print(" The dominant slice at framework saturation:")
    print("   - Edge: Cl(0,2) ≅ ℍ (theorem-grade).")
    print("   - Vertex: Cl(6,0) (per Task A).")
    print("   - Substrate: srs at |E|=3 (per Theorem 8).")
    print("   - Combined gauge: SU(4) × SU(2)_L × SU(2)_R = Pati-Salam.")
    print()
    print(" Subdominant zoo entries plurally retained per A2-T waterline:")
    print("   - Edge alternatives: M_2(ℝ), Cl(0,3)=ℍ⊕ℍ, etc.")
    print("   - 𝕆 at edge: Layer-1 octonion at edge layer (parallel to vertex 𝕆).")
    print("   - ℍ ⊗ 𝕆 = E_7 at edge × octonion: exceptional Lie symmetry candidate.")
    print()
    print(" Tasks remaining:")
    print("   A: Vertex local-algebra zoo                 — DONE (commit 2c2a624).")
    print("   B: Edge qubit algebra zoo                   — THIS PROBE.")
    print("   C: Combined gauge-structure tuples          — pending (substrate × vertex × edge).")
    print("   D: Cooling cascade across all layers        — pending.")
    print("   E: Connect to existing framework apparatus  — pending (PS = dominant tuple).")

    return 0


if __name__ == "__main__":
    main()
