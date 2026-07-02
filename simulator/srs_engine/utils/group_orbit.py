"""
Group-orbit utility — enumerated automorphism groups, C_3, Galois Z_3.

Per the counting-first audit: symmetry groups are enumerated automorphism
groups of the substrate. Aut(srs), C_3 (at high-symmetry k-points), and
Galois Z_3 (operator-algebraic generation symmetry from M ⋊ Z_3 ≅ M_3(ℂ) ⊗ M^α)
are all FINITE enumerable groups, not abstract Lie groups.

This utility wraps the kernel's orbit_count primitive with explicit group
table operations needed for predictions.
"""

import numpy as np
from functools import lru_cache


class GroupOrbitUtility:
    """Group-orbit operations on top of the counting kernel.

    Provides:
    - C_3 cyclic group: characters, irreducible decompositions
    - Galois Z_3: operator-algebraic generation symmetry
    - PS embedding chain: Spin(6) → Spin(4) × Spin(2) → SU(2)_L × SU(2)_R × U(1)_{B-L}
                        → SU(2)_L × U(1)_Y × SU(3)_c
    - Aut(srs) on K_4 quotient (cubic point group action on Wyckoff 8a)
    """

    OMEGA = np.exp(2j * np.pi / 3)  # ω = e^{2πi/3}, primitive cube root of 1

    @staticmethod
    def c3_characters():
        """C_3 character table.

        C_3 has 3 irreducible reps: trivial (1), ω, ω̄.
        Returns dict {irrep_name: character_values}.
        """
        omega = GroupOrbitUtility.OMEGA
        return {
            'trivial': [1, 1, 1],          # χ(e), χ(c), χ(c²)
            'omega':   [1, omega, omega ** 2],
            'omega_bar': [1, omega ** 2, omega],
        }

    @staticmethod
    def c3_isotypic_amplitude(multiplicities, generation_index):
        """Compute amp_j for C_3-isotypic decomposition with given multiplicities.

        Per Q_Koide derivation: amp_j = √(μ_trivial) + √(μ_ω)·ω^j + √(μ_ω̄)·ω^(-j)
        where j is the generation index in {0, 1, 2}.

        For srs at P-point with (μ_trivial, μ_ω, μ_ω̄) = (4, 2, 2):
          amp_0 = 2 + 2√2 ≈ 4.83
          amp_1 = 2 - √2 ≈ 0.59
          amp_2 = 2 - √2 ≈ 0.59

        Args:
            multiplicities: tuple (μ_trivial, μ_ω, μ_ω̄)
            generation_index: integer j ∈ {0, 1, 2}

        Returns:
            complex amplitude amp_j
        """
        mu_trivial, mu_omega, mu_omega_bar = multiplicities
        omega = GroupOrbitUtility.OMEGA
        j = generation_index
        return (
            np.sqrt(mu_trivial)
            + np.sqrt(mu_omega) * omega ** j
            + np.sqrt(mu_omega_bar) * omega ** (-j)
        )

    @staticmethod
    def koide_q_from_isotypic(multiplicities):
        """Compute Q_Koide from C_3-isotypic multiplicities.

        Q = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)²
        where m_j = |amp_j|² for j = 0, 1, 2.

        For (4, 2, 2): Q = 2/3 (theorem-grade, color-sector identity).
        """
        masses = []
        for j in range(3):
            amp_j = GroupOrbitUtility.c3_isotypic_amplitude(multiplicities, j)
            masses.append(abs(amp_j) ** 2)

        sum_m = sum(masses)
        sum_sqrt_m = sum(np.sqrt(m) for m in masses)
        return sum_m / (sum_sqrt_m ** 2)

    @staticmethod
    def ps_subgroup_chain():
        """Pati-Salam subgroup chain enumeration.

        Returns the chain Spin(6) → Spin(4)×Spin(2) → SU(2)_L×SU(2)_R×U(1)_{B-L}
                              → SU(2)_L × U(1)_Y × SU(3)_c
        as a list of (group_name, dimensions, factor_dims).
        """
        return [
            ('Spin(6)', 15, [15]),  # so(6) Lie algebra dim
            ('Spin(4) × Spin(2)', 7, [6, 1]),  # so(4) ⊕ so(2)
            ('SU(2)_L × SU(2)_R × U(1)_{B-L}', 7, [3, 3, 1]),
            ('SU(2)_L × U(1)_Y × SU(3)_c', 12, [3, 1, 8]),  # SM gauge group
        ]

    @staticmethod
    def galois_z3_generation_orbit():
        """Galois Z_3 of M^α ⊂ M ⊂ M ⋊_α Z_3 ≅ M_3(ℂ) ⊗ M^α.

        From M1.B closure: the Galois Z_3 acts on observer C³_obs, giving
        3 generations as the Z_3 orbit. Returns the 3 cyclic permutations.
        """
        return [
            np.eye(3, dtype=complex),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex),
            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex),
        ]

    @staticmethod
    def n_generations():
        """Number of fermion generations = 3 (Galois Z_3 orbit dim)."""
        return 3
