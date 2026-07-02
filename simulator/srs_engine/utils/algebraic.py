"""
Algebraic utility — Cl(6), Cl(0,2) representations from counted generators.

Per the counting-first audit: Clifford algebras are derived from counts
(6 anticommuting involutive generators give 2^(6/2) = 8-dim irreducible
representation; 2 such generators give 2-dim).

This utility constructs the explicit matrix representations needed for
predictions involving the substrate's algebraic structure (vertex
enumeration, fermion content, edge qubit, etc.).
"""

import numpy as np
from functools import lru_cache


class AlgebraicUtility:
    """Clifford algebra constructions on top of the counting kernel.

    Provides explicit matrix representations of:
    - Cl(6) generators γ^a (8x8 complex matrices) at trivalent srs node
    - Cl(0,2) generators f_1, f_2 (2x2 complex matrices) at edge qubit
    - Chirality operator γ_5
    - Bivectors γ^{ab} ∈ so(6) Lie algebra
    """

    @staticmethod
    @lru_cache(maxsize=None)
    def cl6_generators():
        """6 Cl(6) generators γ^a as 8×8 complex matrices.

        Standard Pauli-decomposition representation:
          γ^1 = σ_x ⊗ I  ⊗ I
          γ^2 = σ_y ⊗ I  ⊗ I
          γ^3 = σ_z ⊗ σ_x ⊗ I
          γ^4 = σ_z ⊗ σ_y ⊗ I
          γ^5 = σ_z ⊗ σ_z ⊗ σ_x
          γ^6 = σ_z ⊗ σ_z ⊗ σ_y

        Satisfies {γ^a, γ^b} = 2 δ^{ab} I (Euclidean Cl(6) signature).
        """
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)

        return tuple([
            np.kron(np.kron(sx, I2), I2),
            np.kron(np.kron(sy, I2), I2),
            np.kron(np.kron(sz, sx), I2),
            np.kron(np.kron(sz, sy), I2),
            np.kron(np.kron(sz, sz), sx),
            np.kron(np.kron(sz, sz), sy),
        ])

    @staticmethod
    @lru_cache(maxsize=None)
    def cl6_chirality():
        """γ_5 = γ^1·γ^2·γ^3·γ^4·γ^5·γ^6 — chirality operator.

        Squares to ±I (sign depends on convention). Splits the 8-dim Cl(6)
        spinor into 4_L ⊕ 4_R chirality eigenspaces.
        """
        gens = AlgebraicUtility.cl6_generators()
        g5 = gens[0]
        for i in range(1, 6):
            g5 = g5 @ gens[i]
        return g5

    @staticmethod
    @lru_cache(maxsize=None)
    def cl6_bivectors():
        """15 bivectors γ^{ab} = (1/2)[γ^a, γ^b] forming so(6) Lie algebra.

        Returns dict {(a, b): matrix} for a < b in {1, ..., 6}.
        15 = C(6, 2) bivectors total.
        """
        gens = AlgebraicUtility.cl6_generators()
        bivectors = {}
        for a in range(6):
            for b in range(a + 1, 6):
                bivectors[(a + 1, b + 1)] = 0.5 * (gens[a] @ gens[b] - gens[b] @ gens[a])
        return bivectors

    @staticmethod
    @lru_cache(maxsize=None)
    def cl02_generators():
        """2 Cl(0,2) generators as 2×2 complex matrices on the edge qubit.

        From G2 theorem: edge has spatial orientation f_1 and causal
        direction f_2. After A3-T complexification:
          f_1 → γ^1 = i·σ_x  (signature -1)
          f_2 → γ^2 = i·σ_z  (signature -1)
        Satisfies γ^a² = -I, {γ^1, γ^2} = 0 (Cl(0,2) over ℝ ≅ ℍ).
        Minimal faithful complex rep is 2-dimensional → SU(2) doublet.
        """
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        return (1j * sx, 1j * sz)

    @staticmethod
    def cl02_quaternion_basis():
        """Quaternion basis {I, i, j, k} on the edge qubit.

        i = γ^1 of Cl(0,2)
        j = γ^2 of Cl(0,2)
        k = γ^1·γ^2 (the bivector)
        Satisfies i² = j² = k² = -I, ij = k.
        """
        I2 = np.eye(2, dtype=complex)
        gens = AlgebraicUtility.cl02_generators()
        i_gen, j_gen = gens
        k_gen = i_gen @ j_gen
        return {'1': I2, 'i': i_gen, 'j': j_gen, 'k': k_gen}

    @staticmethod
    def verify_cl6_anticommutation():
        """Verify {γ^a, γ^b} = 2 δ^{ab} I for the 6 Cl(6) generators."""
        gens = AlgebraicUtility.cl6_generators()
        I8 = np.eye(8, dtype=complex)
        for a in range(6):
            for b in range(a, 6):
                anti = gens[a] @ gens[b] + gens[b] @ gens[a]
                expected = 2 * I8 if a == b else np.zeros((8, 8), dtype=complex)
                if not np.allclose(anti, expected):
                    return False
        return True

    @staticmethod
    def verify_cl02_anticommutation():
        """Verify Cl(0,2) signature: f_a² = -I, {f_1, f_2} = 0."""
        gens = AlgebraicUtility.cl02_generators()
        I2 = np.eye(2, dtype=complex)
        for a in range(2):
            sq = gens[a] @ gens[a]
            if not np.allclose(sq, -I2):
                return False
        anti = gens[0] @ gens[1] + gens[1] @ gens[0]
        if not np.allclose(anti, np.zeros((2, 2), dtype=complex)):
            return False
        return True
