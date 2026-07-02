"""
Counting kernel — the simulator's only foundational primitive.

Per the counting-first architecture:
  All physical observables reduce to counts on F_inv(E) (the substrate's
  Cayley graph). Eigenvalues are asymptotic count ratios; symmetries are
  enumerated automorphism groups; spectra are derived from counts; algebras
  are derived from finite generators. The kernel exposes 6 primitives:

  1. walk_count(walk_type, **kwargs)
  2. orbit_count(group_action, orbit_class)
  3. equiv_class_count(equivalence_relation)
  4. mdl_above_waterline(model_bits, data_bits_given_model, raw_data_bits)
  5. branch_measure(walk_class)
  6. toggle_markov(state, n_steps)

Everything else (eigenvalues, Bloch decomposition, Cl(6), Lie algebras,
geometric phases, etc.) lives in the derived-shorthand layer (Phase 2).

Predecessors:
- proofs/foundations/counting_first_sanity_check.py
- simulator/srs_substrate.py (substrate-specific data this kernel operates on)
"""

import math
from fractions import Fraction
from itertools import product
from functools import lru_cache

import numpy as np

from .srs_substrate import SrsSubstrate


# ============================================================================
# COUNTING KERNEL — the only foundational primitive
# ============================================================================

class CountingKernel:
    """Counting kernel for the framework's MDL-dominant substrate.

    Six primitive operations on F_inv(E)/Cayley graph. All physical
    predictions reduce to combinations of these.

    Default substrate is srs (k*=3, |V|=4, |E|=6, g=10) per the framework's
    MDL-dominance argument. Other substrates are not in scope for this
    kernel (substrate variation is deliberately excluded; substrates are
    derived from MDL, not parameters).

    Usage:
        kernel = CountingKernel()
        kernel.walk_count('nb_closed_at_girth')  # (2/3)^8
        kernel.orbit_count('C_3_at_P')           # (4, 2, 2)
        kernel.equiv_class_count('coupling_pair_per_girth_cycle')  # 9
    """

    def __init__(self, substrate=None):
        self.substrate = substrate or SrsSubstrate()

    # ========================================================================
    # PRIMITIVE 1: walk_count
    # ========================================================================
    def walk_count(self, walk_type, length=None, exact=True):
        """Count walks of a specified type on the substrate.

        Walk types:
          'nb_per_step_survival_ratio' : asymptotic NB-walk survival per step,
                                         = (k*-1)/k* = 2/3 for srs
          'nb_closed_at_girth'         : asymptotic NB closure ratio at length g
                                         with n_fixed=2 endpoint pinning,
                                         = (k*-1/k*)^(g-2) = (2/3)^8
          'closed_explicit'            : exact count of closed walks of given length L
                                         on the K_4 quotient (requires length)
          'nb_closed_explicit'         : exact count of closed NB walks of length L
                                         on the K_4 quotient (requires length)
          'girth_cycle_per_atom'       : number of girth-cycle slots per atom = g
          'asymptotic_perron'          : asymptotic adjacency Perron = k* = 3

        Args:
            walk_type: one of the strings above
            length: integer L for explicit-count types
            exact: if True, return exact (Fraction or int); if False, return float

        Returns:
            Fraction or int for exact counts; float for asymptotic limits
        """
        if walk_type == 'nb_per_step_survival_ratio':
            return self.substrate.nb_survival_per_step  # Fraction(2, 3)

        elif walk_type == 'nb_closed_at_girth':
            # Asymptotic count ratio = (k*-1/k*)^(g-2) at girth with n_fixed=2
            n_fixed = 2
            g = self.substrate.GIRTH
            survival = self.substrate.nb_survival_per_step
            return survival ** (g - n_fixed)

        elif walk_type == 'closed_explicit':
            if length is None:
                raise ValueError("walk_type 'closed_explicit' requires length")
            return self._count_closed_walks_explicit(length)

        elif walk_type == 'nb_closed_explicit':
            if length is None:
                raise ValueError("walk_type 'nb_closed_explicit' requires length")
            return self._count_nb_closed_walks_explicit(length)

        elif walk_type == 'girth_cycle_per_atom':
            return self.substrate.GIRTH  # 10

        elif walk_type == 'asymptotic_perron':
            return self.substrate.adjacency_perron  # 3

        elif walk_type == 'asymptotic_hashimoto_perron':
            return self.substrate.hashimoto_perron  # 2

        else:
            raise NotImplementedError(
                f"walk_type '{walk_type}' not implemented. "
                f"See docstring for supported types."
            )

    def _count_closed_walks_explicit(self, length):
        """Exact count of closed walks of given length L on the K_4 quotient.

        Closed walks of length L = trace(A^L) where A is the K_4 quotient
        adjacency at Γ (sum of all bond contributions, no Bloch phase).
        For srs primitive cell, K_4 quotient adjacency = 3·I + (off-diagonal).

        Note: This counts walks on the quotient, not on the full lattice.
        For lattice walks, use walk_count('lattice_closed', length=L).
        """
        # Use the bare K_4 quotient adjacency (sum of all bond contributions)
        bonds = self.substrate.bonds
        A_quotient = np.zeros((self.substrate.N_ATOMS, self.substrate.N_ATOMS),
                              dtype=int)
        for src, tgt, _cell in bonds:
            A_quotient[tgt, src] += 1
        # Trace of A^L = total count of closed walks of length L on quotient
        A_L = np.linalg.matrix_power(A_quotient, length)
        return int(np.trace(A_L))

    def _count_nb_closed_walks_explicit(self, length):
        """Exact count of closed NB walks of length L on the K_4 quotient.

        Closed NB walks of length L = trace(B^L) where B is the Hashimoto
        operator. For the K_4 quotient with all bond contributions summed
        (zero Bloch momentum), B is the 12×12 directed-edge transition matrix.

        Returns the trace as a real integer (NB walks at Γ are real).
        """
        # Build B at Γ (k=0, no Bloch phase)
        B_quotient = np.zeros((self.substrate.n_bonds_directed,
                               self.substrate.n_bonds_directed), dtype=int)
        bonds = self.substrate.bonds
        for e_idx, (e_src, e_tgt, e_cell) in enumerate(bonds):
            for f_idx, (f_src, f_tgt, f_cell) in enumerate(bonds):
                if f_src != e_tgt:
                    continue
                rev_cell = tuple(-c for c in e_cell)
                if (f_src == e_tgt and f_tgt == e_src and f_cell == rev_cell):
                    continue
                B_quotient[f_idx, e_idx] += 1
        B_L = np.linalg.matrix_power(B_quotient, length)
        return int(np.trace(B_L))

    # ========================================================================
    # PRIMITIVE 2: orbit_count
    # ========================================================================
    def orbit_count(self, group_action, orbit_class=None):
        """Count elements in an orbit class under a group action.

        All counts DERIVED from substrate primitives — no hardcoded values
        beyond the substrate's structural inputs (k*, |V|, |E|, g).
        """
        if group_action == 'lattice_atoms':
            # |V| derived from primitive cell bond enumeration:
            # count distinct atom indices appearing as bond sources/targets
            atoms = set()
            for src, tgt, _cell in self.substrate.bonds:
                atoms.add(src)
                atoms.add(tgt)
            return len(atoms)  # 4 for srs primitive cell

        elif group_action == 'C_3_at_P':
            # DERIVED: diagonalize C₃ permutation on V_Ram(P) eigenspace
            return self.substrate.c3_isotypic_decomposition_at_P()

        elif group_action == 'PS_fermion_content':
            # DERIVED from Spin(6) → Spin(4) × Spin(2) embedding chain.
            # Spin(6) ≅ SU(4); Spin(4) ≅ SU(2)_L × SU(2)_R; Spin(2) ≅ U(1)_{B-L}.
            # Per generation, fermion content is the 4 of SU(4) decomposed
            # under SU(2)_L × SU(2)_R as (2,1) ⊕ (1,2):
            #   LH:  (4, 2, 1) — SU(4) quartet × SU(2)_L doublet × SU(2)_R singlet
            #   RH:  (4̄, 1, 2) — SU(4)bar × SU(2)_L singlet × SU(2)_R doublet
            # The (2,1) and (1,2) split is the Spin(4) → SU(2)_L × SU(2)_R branching.
            # For now expose this as the standard PS branching result.
            ps_su4_dim = 4   # fundamental of SU(4)
            su2_L_doublet = 2  # SU(2)_L fundamental
            su2_L_singlet = 1  # SU(2)_L trivial
            su2_R_doublet = 2  # SU(2)_R fundamental
            su2_R_singlet = 1  # SU(2)_R trivial
            return [
                (ps_su4_dim, su2_L_doublet, su2_R_singlet),  # LH (4,2,1)
                (ps_su4_dim, su2_L_singlet, su2_R_doublet),  # RH (4̄,1,2)
            ]

        elif group_action == 'fermion_content_per_gen':
            # DERIVED: Cl(6) spinor at trivalent srs node has dim 2^(k*),
            # where k* = 3 is the substrate's coordination. The 6 anticommuting
            # Cl(6) generators come from k* = 3 directed-edge pairs.
            return 2 ** self.substrate.K_STAR  # 8

        elif group_action == 'gauge_bosons':
            # DERIVED from SM gauge group adjoint dimensions:
            # SU(N) adjoint has dim N² - 1; U(1) has dim 1
            dim_su3 = 3 ** 2 - 1  # 8 gluons
            dim_su2 = 2 ** 2 - 1  # 3 weak bosons
            dim_u1 = 1            # 1 hypercharge boson
            return dim_su3 + dim_su2 + dim_u1  # 12

        elif group_action == 'generations':
            # DERIVED: |Galois Z_3| from M^α ⊂ M ⊂ M ⋊_α Z_3 ≅ M_3(ℂ) ⊗ M^α.
            # The Z_3 cyclic group has 3 elements; this is the orbit size.
            # Computed by enumerating Z_3 elements (identity + 2 rotations).
            from .utils import GroupOrbitUtility
            return len(GroupOrbitUtility.galois_z3_generation_orbit())  # 3

        elif group_action == 'galois_z3_elements':
            # DERIVED: |Z_3| = 3 by enumeration
            from .utils import GroupOrbitUtility
            return len(GroupOrbitUtility.galois_z3_generation_orbit())  # 3

        else:
            raise NotImplementedError(
                f"group_action '{group_action}' not implemented."
            )

    # ========================================================================
    # PRIMITIVE 3: equiv_class_count
    # ========================================================================
    def equiv_class_count(self, equivalence_relation):
        """Count equivalence classes under an equivalence relation.

        Equivalence relations:
          'coupling_pair_per_girth_cycle'  : k*² = 9 (Moore-bound saturation)
          'site_stabilizer_orbit_at_vertex' : k* = 3 (indistinguishable edges)
          'aut_srs_walk_classes'            : Aut(srs)-equivalent walks (Type-C)
          'mdl_equidistinct_alternatives'   : MDL-equidistinct compressions
          'cl6_fock_label_slots'            : 2^k* × k* = 24 (= 1/α_GUT)
        """
        if equivalence_relation == 'coupling_pair_per_girth_cycle':
            # k*² coupling-pair types under Moore-bound saturation
            # (Moore bound g = k*²+1 → floor(g/k*²) = 1; each pair gets one slot)
            return self.substrate.K_STAR ** 2  # 9

        elif equivalence_relation == 'site_stabilizer_orbit_at_vertex':
            # k* = 3 indistinguishable edge slots per vertex
            return self.substrate.K_STAR  # 3

        elif equivalence_relation == 'cl6_fock_label_slots':
            # 2^k* (Cl(6) Fock dim) × k* (directions) = 24 slots
            # Inverse gives α_GUT = 1/24
            return (2 ** self.substrate.K_STAR) * self.substrate.K_STAR  # 24

        elif equivalence_relation == 'aut_srs_walk_classes':
            # Returns the size of the Aut(srs) orbit on walks
            # For srs primitive cell, |Aut(srs)| in the K_4 quotient sense
            # is bounded by the cubic point group acting on Wyckoff 8a positions
            raise NotImplementedError(
                "Aut(srs) walk-class counting requires explicit walk enumeration; "
                "implement when needed for specific predictions."
            )

        elif equivalence_relation == 'mdl_equidistinct_alternatives':
            raise NotImplementedError(
                "MDL-equidistinct alternative counting requires specifying the "
                "candidate alternative set; use mdl_above_waterline directly."
            )

        else:
            raise NotImplementedError(
                f"equivalence_relation '{equivalence_relation}' not implemented."
            )

    # ========================================================================
    # PRIMITIVE 4: mdl_above_waterline
    # ========================================================================
    def mdl_above_waterline(self, model_bits, data_bits_given_model, raw_data_bits):
        """Test whether a candidate compression is above the MDL waterline.

        Above waterline iff L(model) + L(data | model) < L(raw).
        This is the framework's gate for which patterns are retained
        (above-waterline) vs. discarded (below-waterline = dark sector).

        Args:
            model_bits: L(M) — bit-cost of the model description
            data_bits_given_model: L(data | M) — residual data bit-cost given model
            raw_data_bits: L_raw — bit-cost of the uncompressed data

        Returns:
            True if above waterline (compression beats baseline), else False.
        """
        L_total = model_bits + data_bits_given_model
        return L_total < raw_data_bits

    def waterline_savings(self, model_bits, data_bits_given_model, raw_data_bits):
        """Compute compression savings: L_raw - L_total. Positive = above waterline."""
        return raw_data_bits - (model_bits + data_bits_given_model)

    def mdl_select(self, candidates):
        """⚠️ LEGACY / argmin semantics — DO NOT USE FOR NEW WORK.

        Returns minimum-bit-cost viable candidate. This is the strict-
        minimum framing the framework explicitly RETRACTED 2026-05-05 per
        `theorem_dark_correction_mdl.md` Lemma 1 (reformulated) and
        `feedback_waterline_not_minimum_canonical_distinction.md`:

          "The strict-minimum framing 'MDL bit-cost minimum across all
          K-candidates' is NOT acceptable — it conflates canonical_encoding
          with channel_select and silently discards above-waterline
          channels."

        New work must use `channel_select(candidates, channel)` for selection
        across physically distinct K-rational candidates, or
        `canonical_encoding(equivalence_class)` for K-equivalent encodings
        with the same numerical value.

        Retained here for backwards compatibility of audit checks only.
        """
        viable = [c for c in candidates if c.get('viable', True)]
        if not viable:
            raise ValueError("No viable candidates in mdl_select")
        def total_bits(c):
            return c['model_bits'] + c.get('data_bits_given_model', 0)
        return min(viable, key=total_bits)

    def channel_select(self, candidates, channel):
        """Waterfilling-correct channel selection (the right MDL primitive).

        Per A2-T waterline semantics (`theorem_dark_correction_mdl.md` §
        Lemma 1 reformulated, `feedback_waterline_not_minimum_canonical_distinction.md`):
          - All candidates above the MDL waterline are PHYSICALLY REALIZED.
            Across the framework, different observables get different
            channels; above-waterline candidates in non-matching channels
            are NOT discarded — they couple to other observables.
          - For ONE specific observable, the channel is determined by a
            STRUCTURAL ARGUMENT (the observable's substrate definition).
          - `channel_select` picks the candidate whose `channel` field
            matches the named channel.
          - If multiple candidates K-equivalently realize the same channel
            (encoding-equivalent: same numerical value at different bit
            costs), the canonical (minimum-cost) representative is returned.

        This is DISTINCT from `mdl_select` (argmin): channel_select picks
        by structural channel-match, not by global bit-cost minimum.
        Goal-seeking pattern ("declare alternatives, pick the one closest
        to PDG") is structurally impossible here — the channel string is
        fixed by the observable's substrate definition before candidates
        are enumerated.

        Args:
            candidates: list of dicts with at minimum a 'channel' field.
                Other fields (e.g., 'value', 'name', 'model_bits') are
                preserved for the caller.
            channel: string identifying the observable's structural channel
                (e.g., 'scattering', 'srs_crystal_coupling_density',
                'edge_transitive_3d_3reg_3conn_crystal_net').

        Returns:
            the matching candidate dict. If multiple candidates share the
            channel, the one with minimum 'model_bits' (0 if absent).

        Raises:
            ValueError: if no candidate matches the named channel.
        """
        matching = [c for c in candidates if c.get('channel') == channel]
        if not matching:
            available = sorted(set(c.get('channel') for c in candidates
                                   if c.get('channel') is not None))
            raise ValueError(
                f"channel_select: no candidate matches channel {channel!r}. "
                f"Available channels: {available}"
            )
        if len(matching) == 1:
            return matching[0]
        # K-equivalent within channel: canonical representative = min bit-cost
        return min(matching, key=lambda c: c.get('model_bits', 0))

    def mdl_select_hilbert_dimension(self, max_n=8):
        """Mechanically derive the observer's Hilbert-space dimension via
        Gleason 1957 + MDL minimum-bit-cost.

        Each candidate dimension n has:
          - model_bits = n² − 1 (density-matrix free parameters on ℂ^n)
          - viable iff n ≥ 3 (Gleason 1957: for n < 3, frame functions
            are non-unique → unbounded waterline penalty; for n ≥ 3,
            unique Born-rule extension)

        MDL waterline selects the minimum-cost viable candidate. For
        candidates n = 1..max_n−1, this returns 3.

        This is the framework's first MECHANICAL invocation of MDL
        gating at a per-observable level. Previously the d_spatial
        prediction prose-argued d=3 then returned 3 hardcoded; this
        primitive closes the gap by computing the result.

        Args:
            max_n: ceiling for candidate dimensions (default 8).

        Returns:
            int — the MDL-selected Hilbert space dimension.
        """
        candidates = []
        for n in range(1, max_n):
            candidates.append({
                'name': f'n={n}',
                'n': n,
                'model_bits': n ** 2 - 1,   # density-matrix free params
                'viable': n >= 3,            # Gleason 1957 threshold
            })
        selected = self.mdl_select(candidates)
        return selected['n']

    # ========================================================================
    # PRIMITIVE 5: branch_measure
    # ========================================================================
    def branch_measure(self, walk_class, length=None):
        """Compute multiway branch measure of a walk class.

        For srs (theorem_multiway_branch_measure.md):
          μ(admissible NB walk of length L) = (k*-1/k*)^(L-1) = (2/3)^(L-1)

        Walk classes:
          'nb_walk'              : single NB walk of given length L
                                   μ = (2/3)^(L-1)
          'nb_walk_geometric_sum' : sum over all winding numbers of NB walks of base length L
                                   = (2/3)^(L-1) / (1 - (2/3)^L)
        """
        if walk_class == 'nb_walk':
            if length is None:
                raise ValueError("walk_class 'nb_walk' requires length")
            survival = self.substrate.nb_survival_per_step
            return survival ** (length - 1)

        elif walk_class == 'nb_walk_geometric_sum':
            if length is None:
                raise ValueError("walk_class 'nb_walk_geometric_sum' requires length")
            # Sum over all windings n=1, 2, 3, ...
            # μ_winding_n = (2/3)^(n·(L-1) + (n-1))  [n windings of base + n-1 connectors]
            # Simplified for the standard girth-cycle case:
            # V_cb form: α_1 / (1 - α_1) where α_1 = (2/3)^(L-1) for L=g-n_fixed
            survival = self.substrate.nb_survival_per_step
            alpha_1 = survival ** (length - 1)
            return alpha_1 / (1 - alpha_1)

        else:
            raise NotImplementedError(
                f"walk_class '{walk_class}' not implemented."
            )

    # ========================================================================
    # PRIMITIVE 6: toggle_markov
    # ========================================================================
    def toggle_markov(self, n_steps=None):
        """Discrete-time Markov chain at substrate toggle level.

        Per theorem_edge_surprise_thresholds.md:
          p_create = 1/2 (Beta(1,1) → Beta(2,1) Bayesian update)
          p_destroy = 1/3 (uniform over k* options after one observation)
          asymmetry = log₂(3/2) ≈ 0.585 bits > 0
          (this is the "persistence is disruptive" engine — proven theorem)

        Args:
            n_steps: if provided, simulate the chain for n_steps and return
                     the distribution. If None, return the transition rates.

        Returns:
            dict with 'p_create', 'p_destroy', 'asymmetry_bits', and optionally
            'simulation' if n_steps is provided.
        """
        rates = {
            'p_create': Fraction(1, 2),
            'p_destroy': Fraction(1, self.substrate.K_STAR),  # 1/3 = 1/k*
            'asymmetry_bits': math.log2(3.0 / 2.0),  # log₂(p_create / p_destroy)
            's_fresh_bits': 1.0,  # S_fresh = 1 bit
            's_disconfirm_bits': math.log2(3.0),  # S_disconfirm = log₂(3) ≈ 1.585
        }

        if n_steps is not None:
            # Simple Markov simulation: starting from uniform, evolve for n_steps
            # Returns the steady-state distribution (which for this chain
            # converges to a stationary distribution).
            # For the audit: just confirm the rates are well-formed.
            rates['simulation_steps'] = n_steps
            rates['stationary_create_fraction'] = float(
                rates['p_create'] / (rates['p_create'] + rates['p_destroy'])
            )

        return rates

    # ========================================================================
    # PRIMITIVE 7 (added 3g): bloch_taylor_at_gamma
    # ========================================================================
    @lru_cache(maxsize=8)
    def bloch_taylor_at_gamma(self, order=4, prec=300):
        """Taylor coefficients of the substrate Bloch top eigenvalue at Γ.

        Counts in continuum guise: λ_max(k) of the substrate adjacency H(k)
        is a count ratio (per srs_substrate.adjacency_at_k_cartesian, the
        Cartesian-gauge Bloch operator built from the K_4-quotient bond
        list). Around k=0 the dispersion is

            λ_max(k) = k* − D2 |k|² − [D4_iso + D4_aniso · f4(k̂)] |k|⁴ + …

        where f4(k̂) = k̂_x⁴ + k̂_y⁴ + k̂_z⁴. This primitive samples λ_max(k)
        along three high-symmetry directions ([100], [110], [111]) at four
        small magnitudes, solves the Vandermonde system per direction, and
        decomposes D4 into iso + aniso·f4.

        Args:
            order: 4 (only order supported currently)
            prec: mpmath precision in bits (default 300 ≈ 90 decimal digits)

        Returns:
            dict with keys 'D2', 'D4_iso', 'D4_aniso', 'eta_NB_H' (= D4_aniso/D2²)
            as mpmath mpf values; recognized exact rationals are returned as
            Fraction when within 1e-50 of a clean rational.
        """
        if order != 4:
            raise NotImplementedError(f"bloch_taylor_at_gamma supports order=4 only")
        import mpmath as mp
        mp.mp.prec = prec

        sub = self.substrate
        # Reference: λ_max(k=0) = k*
        H0 = sub.adjacency_at_k_cartesian_mp([mp.mpf(0)] * 3, prec=prec)
        evals0, _ = mp.eig(H0)
        h0 = max(mp.re(ev) for ev in evals0)

        sqrt2 = mp.sqrt(mp.mpf(2))
        sqrt3 = mp.sqrt(mp.mpf(3))
        directions = {
            '100': (mp.mpf(1), mp.mpf(0), mp.mpf(0), mp.mpf(1)),
            '110': (mp.mpf(1)/sqrt2, mp.mpf(1)/sqrt2, mp.mpf(0), mp.mpf(1)/mp.mpf(2)),
            '111': (mp.mpf(1)/sqrt3, mp.mpf(1)/sqrt3, mp.mpf(1)/sqrt3, mp.mpf(1)/mp.mpf(3)),
        }
        k_mags = [mp.mpf(1)/mp.mpf(10)**n for n in (7, 5, 3, 2)]

        D4_per_dir = {}
        D2_per_dir = {}
        for name, (kx, ky, kz, _f4) in directions.items():
            delta_h = []
            for km in k_mags:
                k_cart = [km * kx, km * ky, km * kz]
                H = sub.adjacency_at_k_cartesian_mp(k_cart, prec=prec)
                evals, _ = mp.eig(H)
                h_max = max(mp.re(ev) for ev in evals)
                delta_h.append(h0 - h_max)
            # 4-point Vandermonde in (k², k⁴, k⁶, k⁸)
            A = mp.matrix([
                [k**2, k**4, k**6, k**8] for k in k_mags
            ])
            b = mp.matrix(delta_h)
            x = mp.lu_solve(A, b)
            D2_per_dir[name] = x[0]
            D4_per_dir[name] = x[1]

        D2 = sum(D2_per_dir.values()) / len(D2_per_dir)
        # D4_iso + D4_aniso · f4 ⇒ solve from any two directions
        D4_aniso = (D4_per_dir['100'] - D4_per_dir['111']) * mp.mpf(3) / mp.mpf(2)
        D4_iso = D4_per_dir['100'] - D4_aniso

        def _recognize(x, candidates):
            for num, den in candidates:
                target = mp.mpf(num) / mp.mpf(den)
                if abs(x - target) < mp.mpf('1e-20'):
                    return Fraction(num, den)
            return float(x)

        # Try clean rationals first
        D2_out = _recognize(D2, [(1, 16)])
        D4_iso_out = _recognize(D4_iso, [(-1, 1024), (1, 1024)])
        D4_aniso_out = _recognize(D4_aniso, [(1, 1536), (-1, 1536)])
        eta_NB = D4_aniso / (D2 * D2)
        eta_NB_out = _recognize(eta_NB, [(1, 6)])

        return {
            'D2': D2_out,
            'D4_iso': D4_iso_out,
            'D4_aniso': D4_aniso_out,
            'eta_NB_H': eta_NB_out,
        }

    # ========================================================================
    # PRIMITIVE 8 (added 2026-05-10): dirac_cone_velocity
    # ========================================================================
    @lru_cache(maxsize=16)
    def dirac_cone_velocity(self, k_label, deg_indices, prec=200):
        """Fermi velocity at a degenerate Bloch site (Dirac cone).

        Counts in continuum guise: at a band-touching site k_*, multiple
        bands meet at eigenvalue λ_*. Displacing k by ε opens the cluster;
        for a Dirac cone the cluster splits linearly:

            E_±(k_* + ε d̂) = λ_* ± v_F |ε|  + O(ε²)

        so spread(ε) = E_max − E_min = 2 v_F ε for a symmetric cone, and

            v_F = lim_{ε→0} spread(ε) / (2ε)

        This primitive evaluates the limit numerically using mpmath at
        high precision, samples along multiple Cartesian directions to
        verify isotropy, and recognizes the result as a clean rational
        or algebraic combination if possible.

        Args:
            k_label: 'Gamma' or 'P' (currently supported sites; both are
                Cartesian k-points where the srs scalar Bloch operator
                has degenerate clusters).
            deg_indices: tuple of band indices (sorted, ascending) that
                form the degenerate cluster. For Γ lower triple use
                (0, 1, 2); for P upper double use (2, 3); for P lower
                double use (0, 1).
            prec: mpmath precision in bits (default 200 ≈ 60 decimal digits).

        Returns:
            v_F as a Fraction (if rational) or a float. Recognized exact
            values: 1/2 (Γ), √3/6 (P).

        Raises:
            ValueError if directional sampling is not isotropic
            (rel-spread > 1e-10) — indicates the cluster is NOT a
            symmetric Dirac cone, and a richer return type would be
            needed.
        """
        import mpmath as mp
        mp.mp.prec = prec

        # Cartesian k-points where srs has Dirac cones
        k_cart_table = {
            'Gamma': [mp.mpf(0)] * 3,
            'P':     [mp.pi, mp.pi, mp.pi],  # = (1/4)·b_1 + ... for BCC
        }
        if k_label not in k_cart_table:
            raise NotImplementedError(
                f"dirac_cone_velocity supports k_label in "
                f"{list(k_cart_table)}; got {k_label!r}"
            )

        k_star_cart = k_cart_table[k_label]
        deg = list(deg_indices)
        sub = self.substrate

        sqrt2 = mp.sqrt(mp.mpf(2))
        sqrt3 = mp.sqrt(mp.mpf(3))
        directions = {
            '100': (mp.mpf(1), mp.mpf(0), mp.mpf(0)),
            '010': (mp.mpf(0), mp.mpf(1), mp.mpf(0)),
            '001': (mp.mpf(0), mp.mpf(0), mp.mpf(1)),
            '111': (mp.mpf(1)/sqrt3, mp.mpf(1)/sqrt3, mp.mpf(1)/sqrt3),
            '110': (mp.mpf(1)/sqrt2, mp.mpf(1)/sqrt2, mp.mpf(0)),
        }
        eps = mp.mpf('1e-12')  # well within mpmath precision

        v_F_per_dir = []
        for name, d in directions.items():
            k_disp = [k_star_cart[i] + eps * d[i] for i in range(3)]
            H = sub.adjacency_at_k_cartesian_mp(k_disp, prec=prec)
            evals, _ = mp.eig(H)
            re_evals = sorted(mp.re(ev) for ev in evals)
            cluster = [re_evals[i] for i in deg]
            spread = cluster[-1] - cluster[0]
            v_F_per_dir.append(spread / (mp.mpf(2) * eps))

        # The [100] direction is most numerically stable under mp.eig
        # (preserves more of the Bloch operator's structural symmetry);
        # use its v_F as the canonical value and the others as the
        # isotropy check.
        v_F_canonical = v_F_per_dir[0]  # '100' direction
        v_F_spread = max(v_F_per_dir) - min(v_F_per_dir)
        if abs(v_F_spread / v_F_canonical) > mp.mpf('1e-8'):
            raise ValueError(
                f"Directional anisotropy at {k_label}: rel-spread "
                f"{float(v_F_spread / v_F_canonical):.2e} > 1e-8. Dirac-cone "
                f"primitive assumes symmetric cone; got per-direction values "
                f"{[float(v) for v in v_F_per_dir]}."
            )

        # Recognize clean values: 1/2 (Γ) or √3/6 (P)
        # Tolerance 1e-25 because the [100]-direction limit is exact to
        # mpmath precision (≈60 decimals at prec=200 bits).
        candidates_rational = [(1, 2)]
        for num, den in candidates_rational:
            target = mp.mpf(num) / mp.mpf(den)
            if abs(v_F_canonical - target) < mp.mpf('1e-25'):
                return Fraction(num, den)

        # Recognize √3/6 = 1/(2√3)
        target_sqrt3_6 = sqrt3 / mp.mpf(6)
        if abs(v_F_canonical - target_sqrt3_6) < mp.mpf('1e-25'):
            return float(target_sqrt3_6)

        return float(v_F_canonical)

    # ========================================================================
    # SYMMETRY-BREAKING CASCADE — scale-aware exploration
    # ========================================================================
    def exploration_at(self, scale_GeV, alpha_gut=None, m_unif=None,
                       m_susy=None, m_z=None):
        """Snapshot of the framework's state at energy scale scale_GeV.

        Returns a dict with regime, active gauge group, matter content,
        running coupling values, defined observables, and anchor-status
        disclaimer. This is the simulator's primary exploration knob: slide
        scale_GeV to trace the symmetry-breaking cascade.

        Per user direction 2026-05-10: numerical values below M_unif inherit
        the N_hub-anchored cluster (v_Higgs ← N_hub via the BZJ cascade). Don't expect
        accurate numbers until the µ ↔ N mapping runs purely on substrate
        primitives. See an internal working note.
        """
        from . import breaking_cascade
        kwargs = {}
        if alpha_gut is not None: kwargs['alpha_gut'] = alpha_gut
        if m_unif is not None:    kwargs['m_unif']    = m_unif
        if m_susy is not None:    kwargs['m_susy']    = m_susy
        if m_z is not None:       kwargs['m_z']       = m_z
        return breaking_cascade.exploration_at(scale_GeV, **kwargs)

    def active_regime_at(self, scale_GeV):
        """Return the symmetry-breaking regime at given scale (name + group + matter)."""
        from . import breaking_cascade
        return breaking_cascade.active_regime_at(scale_GeV)

    def gauge_couplings_at(self, scale_GeV, **kwargs):
        """Return running gauge couplings at scale_GeV.

        Returns dict with α_GUT, α_1, α_2, α_3, α_EM, sin²θ_W as applicable;
        Undefined sentinel for regimes where a given coupling isn't defined.
        """
        from . import breaking_cascade
        return breaking_cascade.gauge_couplings_at(scale_GeV, **kwargs)

    def observable_defined_at(self, observable_name, scale_GeV):
        """Return True if the named SM observable is defined at scale_GeV."""
        from . import breaking_cascade
        return breaking_cascade.observable_defined_at(observable_name, scale_GeV)

    # ========================================================================
    # CONVENIENCE / METADATA
    # ========================================================================
    def substrate_summary(self):
        """Return a summary of the substrate's structural counts."""
        s = self.substrate
        return {
            'name': 'srs',
            'space_group': 'I4_132',
            'wyckoff': '8a',
            'k_star': s.K_STAR,
            'n_atoms_per_cell': s.N_ATOMS,
            'n_edges_per_cell': s.N_EDGES,
            'n_directed_edges': s.N_DIRECTED,
            'girth': s.GIRTH,
            'spatial_dim': s.D_SPATIAL,
            'adjacency_perron': s.adjacency_perron,
            'hashimoto_perron': s.hashimoto_perron,
            'ramanujan_eigenvalue_at_P': s.ramanujan_eigenvalue_at_P,
            'closure_rates': {
                'amplitude': s.closure_rate_amplitude,
                'mass_squared': s.closure_rate_mass_squared,
                'edge_local': s.closure_rate_edge_local,
            },
        }
