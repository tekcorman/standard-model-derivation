"""
Fiber menu — candidate Bloch fibers per Coxeter system.

For each Coxeter system in the substrate menu, the corresponding Cayley
graph supports a Bloch decomposition. The relevant fibers are the
high-symmetry k-points of the Brillouin zone (stabilized by site
symmetries) plus the linear-dispersion limit fibers (Dirac cones).

For the framework's dominant slice (srs ~ H_3-like):
- Γ-point: zero momentum, trivial site stabilizer
- P-point (1/4, 1/4, 1/4): Ramanujan saddle, C_3-stabilized
- N, H points: standard BCC primitive cell high-symmetry
- Γ-cone, P-cone: linear-dispersion limits (Dirac cones)

For subdominant slices (other Coxeter quotients), the high-symmetry
points are different — distinct stabilizer subgroups, distinct fibers.

NB: enumerator only. The fiber set is determined by the Coxeter system's
group action on the Cayley graph; MDL gating is at the Coxeter level,
not the fiber level. The srs entries mirror the high-symmetry k-points
used in proofs/wave_engine/ and simulator/srs_substrate.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fiber:
    """One Bloch fiber on a Cayley graph.

    Attributes:
        name           : 'Gamma' | 'P' | 'N' | 'H' | 'Gamma_cone' | 'P_cone' | …
        k_fractional   : k in fractional BZ coordinates
        stabilizer     : site-symmetry group at this k-point (Schoenflies)
        dispersion     : 'quadratic' | 'linear (Dirac)' | 'flat' | …
        notes          : provenance
    """
    name: str
    k_fractional: tuple
    stabilizer: str
    dispersion: str
    notes: str = ''


# Dominant-slice srs high-symmetry k-points (cubic I4_132 / (10,3)-a net).
_SRS_FIBERS = [
    Fiber('Gamma', (0.0, 0.0, 0.0), 'O_h (full point group)', 'quadratic',
          notes='zero momentum; bottom of the band; v_F_Γ Dirac-cone limit lives here'),
    Fiber('P', (0.25, 0.25, 0.25), 'C_3 (body-diagonal 3-fold)', 'linear (Dirac)',
          notes='Ramanujan saddle; C_3-stabilized; arg(h_P), v_F_P, sin²θ_W readings'),
    Fiber('N', (0.5, 0.0, 0.0), 'C_2', 'quadratic',
          notes='BCC zone-face midpoint'),
    Fiber('H', (0.5, -0.5, 0.5), 'O_h', 'quadratic',
          notes='BCC zone corner'),
    Fiber('Gamma_cone', (0.0, 0.0, 0.0), 'O_h', 'linear (Dirac)',
          notes='linear-dispersion limit at Γ — v_F_Γ'),
    Fiber('P_cone', (0.25, 0.25, 0.25), 'C_3', 'linear (Dirac)',
          notes='linear-dispersion limit at P — v_F_P'),
]


def enumerate_for_coxeter(coxeter_name: str) -> list[Fiber]:
    """Enumerate high-symmetry fibers for a given Coxeter system's Cayley graph.

    Skeleton: only the srs / H_3-like dominant slice is wired in. For other
    Coxeter systems the fiber set is determined by the space group + Wyckoff
    positions of the corresponding Cayley graph; that table is not yet
    transcribed (TODO in the rebuild).
    """
    if 'srs' in coxeter_name or 'H_3' in coxeter_name:
        return list(_SRS_FIBERS)
    raise NotImplementedError(
        f"fibers for Coxeter system {coxeter_name!r} not yet transcribed; "
        "only the srs / H_3-like dominant slice is wired in. Use a space-group "
        "+ Wyckoff-position table for the corresponding Cayley graph."
    )


def srs_fibers() -> list[Fiber]:
    """The dominant-slice srs fibers: Γ, P (Ramanujan), N, H + dispersion-cone limits."""
    return list(_SRS_FIBERS)
