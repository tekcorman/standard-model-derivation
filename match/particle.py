"""
Particle profile aggregator — F4 of the universal simulator build.

The Particle dataclass packages mass + electroweak quantum numbers + sector
+ derivation status into a single queryable object. It does NOT introduce
new predictions — every field is sourced from the existing simulator/predictions
modules. The point is API consolidation: "pick a particle, get its profile"
rather than threading 8-12 separate prediction calls.

Usage:
    from simulator.srs_engine.particle import get_particle, list_particles

    e = get_particle('electron')
    print(e.mass_GeV)           # 0.000511619...  (from m_e())
    print(e.Q_em)               # Fraction(-1)
    print(e.Y_hyp)              # Fraction(-1) (e_R hypercharge)

    leptons = list_particles(sector='lepton')   # e, μ, τ
    bosons  = list_particles(sector='gauge')    # γ, W, Z, gluon
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Tuple, List

from simulator.srs_engine.kernel import CountingKernel
from .sm_predictions import (
    m_e, m_mu, m_tau,
    m_nu2, m_nu3,
    m_H, M_Z, m_W,
    v_higgs, y_tau, lambda_H,
    hypercharge,
)
from .anchors import m_top
from .pati_salam import PatiSalamUtility


# ============================================================================
# Particle dataclass — one frozen record per SM particle
# ============================================================================

@dataclass(frozen=True)
class Particle:
    """Aggregated profile for one Standard Model particle.

    All fields are sourced from the simulator's existing predictions; this
    class is consolidation, not new derivation.

    Status flags:
      mass_status ∈ {
        'theorem-grade',                # exact derivation, no anchors
        'theorem-grade-conditional',    # derivation conditional on standing axioms
        'in-progress',                  # 🟡 — known derivation gap
        'downgraded',                   # ❌ — explicitly retracted
        'structural',                   # ⚙️ definitional / massless
        'anchor',                       # external (e.g., G_F-anchored chain input)
      }
    """
    name: str
    symbol: str
    sector: str            # 'lepton' | 'neutrino' | 'quark' | 'gauge' | 'scalar'
    generation: Optional[int]   # 1, 2, 3 for fermions; None for bosons
    spin: Fraction         # 1/2 fermion, 1 vector, 0 scalar

    # Mass
    mass_GeV: Optional[float]   # None if not derivable from substrate alone
    mass_status: str

    # Standard-Model quantum numbers
    Q_em: Fraction
    T_3L: Fraction
    color: str             # 'singlet' | 'triplet' | 'antitriplet' | 'octet'
    chirality: Optional[str]   # 'L' | 'R' | 'Dirac' | None

    # Pati-Salam embedding (where defined)
    T_3R: Optional[Fraction] = None
    B_minus_L: Optional[Fraction] = None
    Y_hyp: Optional[Fraction] = None

    # Couplings (where the framework gives them)
    yukawa: Optional[float] = None

    # 1-line derivation summary
    derivation: str = ""


# ============================================================================
# Catalog construction — pulls from existing predictions
# ============================================================================

def _Y_from_PS(label):
    """Hypercharge via PS embedding Y = T_3R + (B-L)/2."""
    return PatiSalamUtility.hypercharge_Y(label)


def _build_catalog(kernel=None):
    kernel = kernel or CountingKernel()
    cat = {}

    # ---- Charged leptons (Koide ratio chain, theorem-grade-conditional) ----
    cat['electron'] = Particle(
        name='electron', symbol='e', sector='lepton', generation=1,
        spin=Fraction(1, 2),
        mass_GeV=m_e(kernel),  # already in GeV (~5.1e-4)
        mass_status='theorem-grade-conditional',
        Q_em=Fraction(-1), T_3L=Fraction(-1, 2),
        color='singlet', chirality='Dirac',
        T_3R=Fraction(-1, 2), B_minus_L=Fraction(-1),
        Y_hyp=_Y_from_PS('e_R'),
        derivation='Koide chain m_e = m_τ · (f_min/f_max)²; abs scale via v_H',
    )
    cat['muon'] = Particle(
        name='muon', symbol='μ', sector='lepton', generation=2,
        spin=Fraction(1, 2),
        mass_GeV=m_mu(kernel),  # GeV
        mass_status='theorem-grade-conditional',
        Q_em=Fraction(-1), T_3L=Fraction(-1, 2),
        color='singlet', chirality='Dirac',
        T_3R=Fraction(-1, 2), B_minus_L=Fraction(-1),
        Y_hyp=_Y_from_PS('e_R'),
        derivation='Koide chain m_μ = m_τ · (f_mid/f_max)²',
    )
    cat['tau'] = Particle(
        name='tau', symbol='τ', sector='lepton', generation=3,
        spin=Fraction(1, 2),
        mass_GeV=m_tau(kernel),
        mass_status='theorem-grade-conditional',
        Q_em=Fraction(-1), T_3L=Fraction(-1, 2),
        color='singlet', chirality='Dirac',
        T_3R=Fraction(-1, 2), B_minus_L=Fraction(-1),
        Y_hyp=_Y_from_PS('e_R'),
        yukawa=float(y_tau(kernel)),
        derivation='m_τ = v · y_τ with y_τ = α_1_full/k*² theorem-grade',
    )

    # ---- Neutrinos (m_ν1 = 0 convention; m_ν2, m_ν3 derived) ----
    cat['nu_1'] = Particle(
        name='nu_1', symbol='ν₁', sector='neutrino', generation=1,
        spin=Fraction(1, 2),
        mass_GeV=0.0,
        mass_status='structural',  # convention (NuFIT 6.0 NO lightest-massless)
        Q_em=Fraction(0), T_3L=Fraction(1, 2),
        color='singlet', chirality='L',
        T_3R=Fraction(0), B_minus_L=Fraction(-1),
        Y_hyp=_Y_from_PS('l_L'),
        derivation='Convention-anchored 0; framework m_ν1 derivation R-15 open',
    )
    cat['nu_2'] = Particle(
        name='nu_2', symbol='ν₂', sector='neutrino', generation=2,
        spin=Fraction(1, 2),
        mass_GeV=m_nu2(kernel) * 1e-9,  # m_nu2 returns eV; → GeV
        mass_status='theorem-grade-conditional',
        Q_em=Fraction(0), T_3L=Fraction(1, 2),
        color='singlet', chirality='L',
        T_3R=Fraction(0), B_minus_L=Fraction(-1),
        Y_hyp=_Y_from_PS('l_L'),
        derivation='m_ν₂ = m_ν₃ / √R via R = 228/7',
    )
    cat['nu_3'] = Particle(
        name='nu_3', symbol='ν₃', sector='neutrino', generation=3,
        spin=Fraction(1, 2),
        mass_GeV=m_nu3(kernel) * 1e-9,
        mass_status='theorem-grade-conditional',
        Q_em=Fraction(0), T_3L=Fraction(1, 2),
        color='singlet', chirality='L',
        T_3R=Fraction(0), B_minus_L=Fraction(-1),
        Y_hyp=_Y_from_PS('l_L'),
        derivation='m_ν₃ = (k*·N_atoms)·M_Pl·N_hub^(-1/2) global formula',
    )

    # ---- Quarks (m_t DOWNGRADED; m_u..m_b 🟡 in-progress per target_parameters) ----
    quark_specs = [
        ('up',      'u', 1, Fraction( 2, 3), Fraction( 1, 2), 'q_L', 'in-progress'),
        ('down',    'd', 1, Fraction(-1, 3), Fraction(-1, 2), 'q_L', 'in-progress'),
        ('charm',   'c', 2, Fraction( 2, 3), Fraction( 1, 2), 'q_L', 'in-progress'),
        ('strange', 's', 2, Fraction(-1, 3), Fraction(-1, 2), 'q_L', 'in-progress'),
        ('top',     't', 3, Fraction( 2, 3), Fraction( 1, 2), 'q_L', 'downgraded'),
        ('bottom',  'b', 3, Fraction(-1, 3), Fraction(-1, 2), 'q_L', 'in-progress'),
    ]
    for nm, sym, gen, Q, T3L, ps_label, status in quark_specs:
        # Top mass returns None per anchors.m_top (DOWNGRADED 2026-05-04 EOD+3)
        mass = m_top(kernel) if nm == 'top' else None  # m_u..m_b not yet derived
        cat[f'{nm}_quark'] = Particle(
            name=f'{nm}_quark', symbol=sym, sector='quark', generation=gen,
            spin=Fraction(1, 2),
            mass_GeV=mass,
            mass_status=status,
            Q_em=Q, T_3L=T3L,
            color='triplet', chirality='Dirac',
            T_3R=Fraction(1, 2) if Q == Fraction(2, 3) else Fraction(-1, 2),
            B_minus_L=Fraction(1, 3),
            Y_hyp=_Y_from_PS('u_R' if Q == Fraction(2, 3) else 'd_R'),
            derivation=(
                'm_t DOWNGRADED 2026-05-04 EOD+3 (PDG inputs needed)'
                if nm == 'top'
                else 'quark mass cascade pending R-14 closure'
            ),
        )

    # ---- Gauge bosons ----
    cat['photon'] = Particle(
        name='photon', symbol='γ', sector='gauge', generation=None,
        spin=Fraction(1),
        mass_GeV=0.0, mass_status='structural',
        Q_em=Fraction(0), T_3L=Fraction(0),
        color='singlet', chirality=None,
        T_3R=Fraction(0), B_minus_L=Fraction(0), Y_hyp=Fraction(0),
        derivation='Massless U(1)_em gauge boson; structural after EW breaking',
    )
    cat['W_boson'] = Particle(
        name='W_boson', symbol='W±', sector='gauge', generation=None,
        spin=Fraction(1),
        mass_GeV=m_W(kernel),
        mass_status='theorem-grade-conditional',
        Q_em=Fraction(1), T_3L=Fraction(1),  # |Q| = 1; T_3L = ±1
        color='singlet', chirality=None,
        derivation='m_W = M_Z·cos(θ_W) ≈ 80.4 GeV via 5-stage gauge closure',
    )
    cat['Z_boson'] = Particle(
        name='Z_boson', symbol='Z⁰', sector='gauge', generation=None,
        spin=Fraction(1),
        mass_GeV=M_Z(kernel),
        mass_status='theorem-grade-conditional',
        Q_em=Fraction(0), T_3L=Fraction(0),
        color='singlet', chirality=None,
        derivation='Self-consistent EW matching M_Z = √π·v·√(α_2 + (3/5)·α_1)',
    )
    cat['gluon'] = Particle(
        name='gluon', symbol='g', sector='gauge', generation=None,
        spin=Fraction(1),
        mass_GeV=0.0, mass_status='structural',
        Q_em=Fraction(0), T_3L=Fraction(0),
        color='octet', chirality=None,
        derivation='Massless SU(3)_c gauge boson; 8 generators',
    )

    # ---- Higgs ----
    cat['higgs'] = Particle(
        name='higgs', symbol='H⁰', sector='scalar', generation=None,
        spin=Fraction(0),
        mass_GeV=m_H(kernel),
        mass_status='theorem-grade',
        Q_em=Fraction(0), T_3L=Fraction(-1, 2),
        color='singlet', chirality=None,
        T_3R=Fraction(1, 2), B_minus_L=Fraction(0),
        Y_hyp=_Y_from_PS('higgs'),
        yukawa=float(lambda_H(kernel)),  # λ_H Higgs self-coupling (not strictly Yukawa)
        derivation='m_H = √(2·λ_H)·v with λ_H = 2·α_1_full theorem-grade',
    )

    return cat


# Module-level cache (recomputed only on demand)
_CATALOG_CACHE = None


def _catalog():
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        _CATALOG_CACHE = _build_catalog()
    return _CATALOG_CACHE


# ============================================================================
# Public API
# ============================================================================

def get_particle(name: str) -> Particle:
    """Look up a particle by canonical name.

    Names: 'electron', 'muon', 'tau', 'nu_1', 'nu_2', 'nu_3',
           'up_quark', 'down_quark', 'charm_quark', 'strange_quark',
           'top_quark', 'bottom_quark', 'photon', 'W_boson', 'Z_boson',
           'gluon', 'higgs'.

    Raises KeyError if name not in catalog.
    """
    cat = _catalog()
    if name not in cat:
        raise KeyError(f"Particle '{name}' not in catalog. "
                       f"Use list_particles() to see available names.")
    return cat[name]


def list_particles(sector: Optional[str] = None) -> List[Particle]:
    """List all particles (or filter by sector).

    Sectors: 'lepton', 'neutrino', 'quark', 'gauge', 'scalar'.
    """
    cat = _catalog()
    if sector is None:
        return list(cat.values())
    return [p for p in cat.values() if p.sector == sector]


def particle_names() -> List[str]:
    """Return all canonical particle names in the catalog."""
    return list(_catalog().keys())
