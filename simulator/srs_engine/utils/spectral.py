"""
Spectral utility — asymptotic count limits as eigenvalues.

Per the counting-first audit: eigenvalues of substrate operators are
asymptotic limits of count ratios. This utility provides the spectral
shortcuts that physicists invented to avoid direct counting when counts
become intractable.

For srs (small structure), exact spectral data is computable directly;
for general use, the asymptotic limit IS the eigenvalue.
"""

import math
from fractions import Fraction
import numpy as np


class SpectralUtility:
    """Spectral shortcuts on top of the counting kernel.

    All methods take a kernel and return spectral observables derived from
    the kernel's count primitives.
    """

    @staticmethod
    def adjacency_perron_eigenvalue(kernel):
        """λ_max(A) = k* (asymptotic count ratio of closed walks).

        From walk_count('asymptotic_perron') = k*.
        """
        return kernel.walk_count('asymptotic_perron')

    @staticmethod
    def hashimoto_perron_eigenvalue(kernel):
        """λ_max(B) = k* − 1 (asymptotic count ratio of NB-walks).

        From walk_count('asymptotic_hashimoto_perron') = k* − 1.
        """
        return kernel.walk_count('asymptotic_hashimoto_perron')

    @staticmethod
    def nb_survival_per_step(kernel):
        """Per-step NB walk survival = (k* − 1)/k* (Perron ratio).

        Asymptotic limit of count(NB walks of length L) / count(all walks of L).
        """
        return kernel.walk_count('nb_per_step_survival_ratio')

    @staticmethod
    def adjacency_spectrum_at_k(kernel, k_point):
        """Sorted real eigenvalues of A(k) at the given k-point.

        For the kernel's substrate (srs primitive cell, K_4 quotient),
        this is computed via direct diagonalization of the 4x4 Bloch matrix.
        """
        return kernel.substrate.adjacency_spectrum_at_k(k_point)

    @staticmethod
    def hashimoto_eigenvalue_at_P(kernel):
        """The framework's Ramanujan saddle eigenvalue h = (√3 + i√5)/2.

        For srs at P-point, h is the dominant complex eigenvalue of B(P).
        """
        return kernel.substrate.ramanujan_eigenvalue_at_P

    @staticmethod
    def asymptotic_count_ratio(kernel, walk_type, length):
        """Generic asymptotic count ratio for a walk type at length L.

        Falls back to direct enumeration for small L; uses Perron eigenvalue
        for large L (asymptotic regime).
        """
        if length <= 12:
            # Direct enumeration tractable
            if walk_type == 'closed':
                return Fraction(kernel.walk_count('closed_explicit', length=length))
            elif walk_type == 'nb_closed':
                return Fraction(kernel.walk_count('nb_closed_explicit', length=length))
            else:
                raise NotImplementedError(f"walk_type {walk_type} for asymptotic ratio")
        else:
            # Asymptotic regime: λ^L
            if walk_type == 'closed':
                lam = SpectralUtility.adjacency_perron_eigenvalue(kernel)
            elif walk_type == 'nb_closed':
                lam = SpectralUtility.hashimoto_perron_eigenvalue(kernel)
            else:
                raise NotImplementedError(f"walk_type {walk_type} for asymptotic ratio")
            return lam ** length
