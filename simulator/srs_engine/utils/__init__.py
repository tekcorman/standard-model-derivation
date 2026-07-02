"""
Derived-shorthand utilities for the counting-first substrate computer.

Per the architecture: counting kernel is the only foundational primitive;
all other apparatus (eigenvalues, algebras, group theory, geometric phases,
etc.) is derived shorthand built on top of the kernel.

These utilities are PHYSICS-FREE — they compute substrate quantities
(eigenvalues, irreducible representations, dihedral angles, holonomy,
character traces) without naming any SM observable. The match/ package
consumes these outputs and pairs them with SM observables.

- SpectralUtility: asymptotic count limits → eigenvalues
- AlgebraicUtility: Cl(6), Cl(0,2) representations from counted generators
- GroupOrbitUtility: substrate orbits, C₃ at P, Galois Z_3
- GeometricPhaseUtility: continuous angles, dihedrals, Berry phases

PatiSalamUtility was moved to match/pati_salam.py — Pati-Salam group
theory is a physics-naming layer over the underlying Spin(6) algebra
(which lives in AlgebraicUtility); it does not belong in the substrate-
computer simulator.
"""

from .spectral import SpectralUtility
from .algebraic import AlgebraicUtility
from .group_orbit import GroupOrbitUtility
from .geometric_phase import GeometricPhaseUtility

__all__ = [
    'SpectralUtility',
    'AlgebraicUtility',
    'GroupOrbitUtility',
    'GeometricPhaseUtility',
]
