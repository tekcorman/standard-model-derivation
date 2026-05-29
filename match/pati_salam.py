"""
Pati-Salam utility — rep-theoretic computations on PS embedding.

Per the counting-first audit: PS rep theory is enumerated subgroup chain
on Cl(6) bivectors. Trace identities (Tr(T_3L²), Tr(Q²)) are sums of
eigenvalue counts.

This utility provides the most-needed PS rep-theoretic computations for
predictions like sin²θ_W, hypercharge assignments, and gauge coupling
ratios.
"""

from fractions import Fraction


class PatiSalamUtility:
    """Pati-Salam rep-theoretic utilities.

    PS = SU(4) × SU(2)_L × SU(2)_R
    Per generation, fermion content is (4, 2, 1) + (4̄, 1, 2):
      (4, 2, 1) = LH SU(4) quartet × SU(2)_L doublet × SU(2)_R singlet
      (4̄, 1, 2) = RH SU(4)bar × SU(2)_L singlet × SU(2)_R doublet
    SU(4) decomposes as 3 colors + 1 lepton under SU(3)_c × U(1)_{B-L}.
    """

    # Standard PS fermion content per generation
    PS_REPS_PER_GENERATION = [(4, 2, 1), (4, 1, 2)]  # (LH, RH); (4̄ ≡ 4 here)

    @staticmethod
    def trace_T3L_squared(rep):
        """Compute Tr(T_3L²) for a PS rep (n_color, n_L, n_R).

        T_3L is the third generator of SU(2)_L. After PS → SM breaking,
        T_3R becomes part of hypercharge Y = T_3R + (B-L)/2.

        For SU(2)_L doublet: Tr(T_3L²) per doublet = (1/2)² + (-1/2)² = 1/2.
        For SU(2)_L singlet: Tr(T_3L²) = 0.

        Multiply by SU(4) color × SU(2)_R dimensions.
        """
        n_color, n_L, n_R = rep
        if n_L == 2:
            return Fraction(1, 2) * n_color * n_R
        return Fraction(0)

    @staticmethod
    def trace_T3R_squared(rep):
        """Compute Tr(T_3R²) for a PS rep.

        Symmetric to T_3L: only contributes for SU(2)_R doublets.
        After PS → SM breaking, T_3R is absorbed into hypercharge Y.
        """
        n_color, n_L, n_R = rep
        if n_R == 2:
            return Fraction(1, 2) * n_color * n_L
        return Fraction(0)

    @staticmethod
    def trace_Q_squared(rep):
        """Compute Tr(Q²) for a PS rep.

        DERIVED from Q = T_3L + Y where Y = T_3R + (B-L)/2 in PS embedding.

        SU(4) decomposes under SU(3)_c × U(1)_{B-L} as:
          4 → 3_quark (B-L = +1/3) + 1_lepton (B-L = -1)
        For 4̄: opposite signs.

        For each PS rep (n_color=4, n_L, n_R), enumerate the SU(2)_L and
        SU(2)_R weights (T_3L = ±1/2 for doublet, 0 for singlet), then
        compute Q = T_3L + T_3R + (B-L)/2 for each constituent state and
        sum Q² across all states.
        """
        n_color, n_L, n_R = rep
        if n_color != 4:
            raise NotImplementedError(f"Only 4 of SU(4) supported; got {n_color}")

        # T_3L weights: ±1/2 if doublet, 0 if singlet
        if n_L == 2:
            t3L_weights = [Fraction(1, 2), Fraction(-1, 2)]
        else:
            t3L_weights = [Fraction(0)]

        # T_3R weights: ±1/2 if doublet, 0 if singlet
        if n_R == 2:
            t3R_weights = [Fraction(1, 2), Fraction(-1, 2)]
        else:
            t3R_weights = [Fraction(0)]

        # SU(4) decomposition: 4 → 3 quark colors (B-L = +1/3) + 1 lepton (B-L = -1)
        # For (4,2,1) and (4,1,2) we use SU(4) (not 4̄) — the PS labeling
        # convention here treats both as the standard 4-rep
        # B-L values for the 4 constituents
        n_quark_colors = 3
        bL_quark = Fraction(1, 3)   # quark B-L
        bL_lepton = Fraction(-1)    # lepton B-L

        # For RH partners (4̄,1,2) the convention flips B-L sign
        if n_R == 2:
            bL_quark = -bL_quark
            bL_lepton = -bL_lepton

        # Enumerate all states: SU(4) constituent × SU(2)_L weight × SU(2)_R weight
        sum_Q_sq = Fraction(0)
        for t3L in t3L_weights:
            for t3R in t3R_weights:
                # 3 quark colors with B-L
                for color in range(n_quark_colors):
                    Q = t3L + t3R + bL_quark / 2
                    sum_Q_sq += Q ** 2
                # 1 lepton with its B-L
                Q = t3L + t3R + bL_lepton / 2
                sum_Q_sq += Q ** 2

        return sum_Q_sq

    @staticmethod
    def sin2_theta_W(reps=None):
        """sin²θ_W at unification = Σ Tr(T_3L²) / Σ Tr(Q²) on PS reps.

        For PS_REPS_PER_GENERATION = [(4,2,1), (4̄,1,2)]:
          Σ Tr(T_3L²) = 2 + 0 = 2 (only LH doublet contributes)
          Σ Tr(Q²) = 8/3 + 8/3 = 16/3 (both contribute)
          sin²θ_W = 2 / (16/3) = 3/8 (exact rational at M_unif)

        Args:
            reps: list of PS reps; defaults to standard per-generation content

        Returns:
            Fraction — exact value 3/8 at unification
        """
        if reps is None:
            reps = PatiSalamUtility.PS_REPS_PER_GENERATION

        sum_T3L_sq = sum(PatiSalamUtility.trace_T3L_squared(r) for r in reps)
        sum_Q_sq = sum(PatiSalamUtility.trace_Q_squared(r) for r in reps)
        return sum_T3L_sq / sum_Q_sq

    @staticmethod
    def alpha_GUT(kernel):
        """α_GUT = 1/(2^k* · k*) — unified gauge coupling at unification.

        From kernel: equiv_class_count('cl6_fock_label_slots') = 24 = 1/α_GUT.
        For k* = 3: α_GUT = 1/24 ≈ 0.04167.
        """
        slots = kernel.equiv_class_count('cl6_fock_label_slots')
        return Fraction(1, slots)

    # Particle quantum-number table: (T_3R, B-L) per standard SM particle label.
    # These are the irreducible PS labels; Y is DERIVED from them via the
    # embedding formula Y = T_3R + (B-L)/2.
    # Note: the table here is the irreducible (T_3R, B-L) assignment under PS,
    # not the derived Y. This minimizes hardcoded content.
    _PS_QUANTUM_NUMBERS = {
        # Particle | T_3R | B-L
        'q_L':  (Fraction(0),    Fraction(1, 3)),    # quark doublet, SU(2)_R singlet
        'u_R':  (Fraction(1, 2), Fraction(1, 3)),    # up-type RH, SU(2)_R doublet
        'd_R':  (Fraction(-1, 2), Fraction(1, 3)),   # down-type RH, SU(2)_R doublet
        'l_L':  (Fraction(0),    Fraction(-1)),      # lepton doublet, SU(2)_R singlet
        'e_R':  (Fraction(-1, 2), Fraction(-1)),     # electron RH, SU(2)_R doublet
        'higgs': (Fraction(1, 2), Fraction(0)),      # Higgs (T_3R=+1/2 from PS Higgs identification)
    }

    @staticmethod
    def hypercharge_Y(rep_label):
        """Hypercharge Y for standard SM particle labels.

        DERIVED from PS embedding formula:  Y = T_3R + (B-L)/2

        The (T_3R, B-L) assignment per particle is the irreducible PS labeling;
        Y is computed via the embedding formula, not hardcoded.
        """
        if rep_label not in PatiSalamUtility._PS_QUANTUM_NUMBERS:
            return None
        t3R, bL = PatiSalamUtility._PS_QUANTUM_NUMBERS[rep_label]
        return t3R + bL / 2  # PS embedding formula

    @staticmethod
    def fermion_states_per_generation():
        """Total fermion states per generation per chirality = 8.

        From Cl(6) spinor dim = 2^(6/2) = 8.
        Decomposes as: 2 quarks × 3 colors + 2 leptons = 6 + 2 = 8.
        """
        return 8
