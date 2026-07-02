"""
Geometric-phase utility — continuous angles from count-determined data.

Per the counting-first audit: geometric phase VALUES (dihedrals, polytope
angles, Berry windings) are continuous functions of count-determined data.
The kernel provides the count data; this utility computes the continuous-
shorthand quantities WITHOUT physics naming.

Method names describe substrate operations only (closure rates by class,
polytope dihedrals on K_4 (-1)-eigenspace, Z_3 holonomy on cycles, walker
phase windings). The match/ package consumes these outputs and pairs them
with SM observable names (δ_CP_CKM, α_21 PMNS, θ_QCD, etc.).
"""

import math
import numpy as np


class GeometricPhaseUtility:
    """Continuous geometric quantities derived from count data.

    NB: this is the substrate-side. Match-layer (match/) names are listed in
    each docstring as cross-references only.
    """

    # ------------------------------------------------------------------------
    # Closure rates ν at the Ramanujan saddle, by reading-class
    # ------------------------------------------------------------------------

    @staticmethod
    def closure_rate_amplitude(kernel):
        """ν_amp = |Im(h)|/|h|² = √5/4 — amplitude-class closure rate.

        Substrate: at the Ramanujan saddle h = (√3+i√5)/2,
        Im(h)/|h|² = (√5/2) / 2 = √5/4.

        Match-layer use: dark-extraction map Class 1 (off-diagonal C₃
        observables — V_us, m_ν₂, m_ν₃ family).
        """
        return kernel.substrate.closure_rate_amplitude

    @staticmethod
    def closure_rate_mass_squared(kernel):
        """ν_m² = tan²(arg h) = 5/3 — mass²-class closure rate.

        Substrate: arg(h) = arctan(√5/√3); tan²(arg h) = 5/3.

        Match-layer use: dark-extraction map Class 2 (mass-mixing
        diagonalization — y_τ, λ_H, θ_23 PMNS family).
        """
        return kernel.substrate.closure_rate_mass_squared

    @staticmethod
    def closure_rate_edge_local(kernel):
        """ν_edge = 1 — edge-local closure rate.

        Substrate: Tr(σ_x) = 0 at C₃-symmetric vertex kills the
        Im-channel enhancement; only the bare α₁ survives.

        Match-layer use: dark-extraction map Class 3 (θ_13 PMNS,
        V_cb edge-local family).
        """
        return kernel.substrate.closure_rate_edge_local

    @staticmethod
    def closure_rate(kernel, observable_class):
        """Class-dependent closure rate at the Ramanujan saddle.

        Args:
            observable_class: 'amplitude', 'mass_squared', or 'edge_local'
        """
        if observable_class == 'amplitude':
            return GeometricPhaseUtility.closure_rate_amplitude(kernel)
        elif observable_class == 'mass_squared':
            return GeometricPhaseUtility.closure_rate_mass_squared(kernel)
        elif observable_class == 'edge_local':
            return GeometricPhaseUtility.closure_rate_edge_local(kernel)
        else:
            raise ValueError(f"Unknown observable_class: {observable_class}")

    # ------------------------------------------------------------------------
    # Polytope dihedrals on K_4 (-1)-eigenspace
    # ------------------------------------------------------------------------

    @staticmethod
    def k4_minus_eigenspace_dihedral():
        """Regular-tetrahedron dihedral angle = arccos(1/3) on K_4 adjacency
        (-1)-eigenspace.

        Substrate: the K_4 adjacency has spectrum {+3, -1, -1, -1}; the
        three (-1)-eigenvectors form the corners of a regular tetrahedron
        in 3-D, with face-to-face dihedral cos(β) = 1/3 (= 1/k* on srs).

        Match-layer use: δ_CP_CKM ≈ 70.53° (CP-violating phase from
        polytope geometry).
        """
        cos_value = 1.0 / 3.0
        radians = math.acos(cos_value)
        degrees = math.degrees(radians)
        return {
            'cos_value': cos_value,
            'radians': radians,
            'degrees': degrees,  # ≈ 70.53°
        }

    # ------------------------------------------------------------------------
    # Walker phase windings at the Ramanujan saddle
    # ------------------------------------------------------------------------

    @staticmethod
    def arg_h_at_P(kernel):
        """arg(h) = arctan(√5/√3) — polar angle of the Ramanujan eigenvalue.

        Used as the unit phase increment for walker-windings on closed cycles.

        Match-layer use: PMNS Majorana phases α_21 = g·arg(h),
        α_31 = 2g·arg(h) where g is the Galois winding factor (= girth on srs).
        """
        h = kernel.substrate.ramanujan_eigenvalue_at_P
        return math.atan2(h.imag, h.real)

    @staticmethod
    def walker_phase_winding(kernel, winding_number=1):
        """n × arg(h) mod 2π — walker phase accumulated over n windings.

        Args:
            winding_number: integer n ≥ 1.

        Returns dict with phase in radians and degrees, both mod 2π.

        Match-layer use:
          n=g (= girth = 10) → α_21 PMNS Majorana phase ≈ 162.39°
          n=2g               → α_31 PMNS Majorana phase ≈ 324.78°
        """
        arg_h = GeometricPhaseUtility.arg_h_at_P(kernel)
        rad = (winding_number * arg_h) % (2 * math.pi)
        deg = math.degrees(rad) % 360
        return {
            'radians': rad,
            'degrees': deg,
            'arg_h_degrees': math.degrees(arg_h),
            'winding_number': winding_number,
        }

    # ------------------------------------------------------------------------
    # Z_3 holonomy on cycles
    # ------------------------------------------------------------------------

    @staticmethod
    def z3_holonomy_flat():
        """Z_3 holonomy on srs cycles is FLAT (= identity).

        Substrate: per R3 refutation
        (`proofs/flavor/z3_holonomy_cycles.py`), all gauge-invariant
        Z_3 holonomies on girth-10, 12, 14 cycles vanish identically.
        The bundle is globally trivializable.

        Match-layer use: θ_QCD = 0 (no strong CP violation).
        """
        return {
            'holonomy': 0.0,
            'phase_rad': 0.0,
            'note': 'Z_3 connection on srs is flat; trivializable bundle',
        }
