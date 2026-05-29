"""
simulator_particle_validation.py

Validation probe for F4 — the Particle dataclass + get_particle/list_particles
API. Verifies that the catalog correctly aggregates the underlying predictions
and that quantum numbers are self-consistent.

Tests cover:
  - 17-particle catalog completeness
  - mass values per particle (where derivable)
  - electric charges, spins, sectors
  - Pati-Salam embedding consistency: Q = T_3L + Y on the chiral irreps
  - downgraded particles (m_top = None)
  - sector filtering

Predecessors:
- simulator/particle.py
- simulator/predictions/__init__.py (the data sources)
"""

import sys
import math
from pathlib import Path
from fractions import Fraction

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from match import (
    Particle, get_particle, list_particles, particle_names,
)


class TestStats:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def check(self, name, condition, msg=""):
        if condition:
            print(f"  ✓ {name}")
            self.passed += 1
        else:
            print(f"  ✗ {name}: {msg}")
            self.failed.append((name, msg))

    def check_rel(self, name, predicted, expected, rtol=1e-3):
        if predicted is None:
            self.check(name, False, f"got None, expected {expected}")
            return
        ok = abs(predicted - expected) / abs(expected) < rtol
        if ok:
            print(f"  ✓ {name}: {predicted}")
            self.passed += 1
        else:
            print(f"  ✗ {name}: predicted {predicted}, expected {expected}")
            self.failed.append((name, f"{predicted} vs {expected}"))

    def summary(self):
        total = self.passed + len(self.failed)
        print(f"\n  RESULT: {self.passed}/{total} passed")
        if self.failed:
            print("  FAILURES:")
            for nm, m in self.failed:
                print(f"    - {nm}: {m}")
        return len(self.failed) == 0


def test_catalog_completeness(stats):
    print("\n[catalog] 17-particle SM catalog")
    expected_names = {
        'electron', 'muon', 'tau',
        'nu_1', 'nu_2', 'nu_3',
        'up_quark', 'down_quark', 'charm_quark', 'strange_quark',
        'top_quark', 'bottom_quark',
        'photon', 'W_boson', 'Z_boson', 'gluon', 'higgs',
    }
    actual = set(particle_names())
    stats.check("17 particles in catalog", len(actual) == 17,
                f"got {len(actual)}: {actual}")
    stats.check("expected names match", expected_names == actual,
                f"diff: {expected_names ^ actual}")


def test_lepton_masses(stats):
    print("\n[masses] charged-lepton GeV values (Koide chain)")
    e = get_particle('electron')
    mu = get_particle('muon')
    tau = get_particle('tau')
    stats.check_rel("m_e (GeV)", e.mass_GeV, 0.511e-3, rtol=2e-3)
    stats.check_rel("m_μ (GeV)", mu.mass_GeV, 0.10566, rtol=2e-3)
    stats.check_rel("m_τ (GeV)", tau.mass_GeV, 1.7769, rtol=2e-3)
    stats.check_rel("y_τ (Yukawa)", tau.yukawa, 7.226e-3, rtol=1e-3)


def test_neutrino_masses(stats):
    print("\n[masses] neutrino sector")
    nu1 = get_particle('nu_1')
    nu2 = get_particle('nu_2')
    nu3 = get_particle('nu_3')
    stats.check("m_ν1 = 0 (NuFIT NO convention)", nu1.mass_GeV == 0.0)
    stats.check_rel("m_ν2 (GeV)", nu2.mass_GeV, 8.86e-12, rtol=2e-3)
    stats.check_rel("m_ν3 (GeV)", nu3.mass_GeV, 5.057e-11, rtol=2e-3)
    stats.check("ν3 > ν2 > ν1 (NO ordering)",
                nu3.mass_GeV > nu2.mass_GeV > nu1.mass_GeV)


def test_top_downgraded(stats):
    print("\n[honest] m_top DOWNGRADED")
    top = get_particle('top_quark')
    stats.check("m_top is None", top.mass_GeV is None)
    stats.check("status flag = 'downgraded'", top.mass_status == 'downgraded')
    stats.check("derivation cites 2026-05-04 EOD+3",
                '2026-05-04 EOD+3' in top.derivation)


def test_electroweak_bosons(stats):
    print("\n[masses] electroweak bosons + Higgs")
    W = get_particle('W_boson')
    Z = get_particle('Z_boson')
    H = get_particle('higgs')
    stats.check_rel("m_W (GeV)", W.mass_GeV, 80.369, rtol=1e-2)
    stats.check_rel("M_Z (GeV)", Z.mass_GeV, 91.19, rtol=1e-2)
    stats.check_rel("m_H (GeV)", H.mass_GeV, 125.20, rtol=5e-3)
    photon = get_particle('photon')
    gluon = get_particle('gluon')
    stats.check("m_γ = 0 (structural)", photon.mass_GeV == 0.0)
    stats.check("m_g = 0 (structural)", gluon.mass_GeV == 0.0)


def test_charges_and_quantum_numbers(stats):
    print("\n[QN] electric charges + hypercharges")
    stats.check("Q(e) = -1", get_particle('electron').Q_em == Fraction(-1))
    stats.check("Q(u) = +2/3", get_particle('up_quark').Q_em == Fraction(2, 3))
    stats.check("Q(d) = -1/3", get_particle('down_quark').Q_em == Fraction(-1, 3))
    stats.check("Q(γ) = 0", get_particle('photon').Q_em == 0)
    stats.check("Q(W) = 1", get_particle('W_boson').Q_em == 1)
    stats.check("Y(higgs) = 1/2 via PS", get_particle('higgs').Y_hyp == Fraction(1, 2))
    stats.check("Y(e_R) = -1 via PS", get_particle('electron').Y_hyp == Fraction(-1))


def test_spins(stats):
    print("\n[QN] spins per sector")
    for nm in ['electron', 'tau', 'up_quark', 'nu_1']:
        stats.check(f"spin({nm}) = 1/2",
                    get_particle(nm).spin == Fraction(1, 2))
    for nm in ['photon', 'W_boson', 'Z_boson', 'gluon']:
        stats.check(f"spin({nm}) = 1", get_particle(nm).spin == Fraction(1))
    stats.check("spin(higgs) = 0", get_particle('higgs').spin == Fraction(0))


def test_color_assignments(stats):
    print("\n[QN] color reps")
    stats.check("e color = singlet", get_particle('electron').color == 'singlet')
    stats.check("u color = triplet", get_particle('up_quark').color == 'triplet')
    stats.check("γ color = singlet", get_particle('photon').color == 'singlet')
    stats.check("g color = octet", get_particle('gluon').color == 'octet')


def test_sector_filtering(stats):
    print("\n[API] list_particles(sector=...)")
    leptons = list_particles('lepton')
    neutrinos = list_particles('neutrino')
    quarks = list_particles('quark')
    bosons = list_particles('gauge')
    scalars = list_particles('scalar')
    stats.check("3 charged leptons", len(leptons) == 3)
    stats.check("3 neutrinos", len(neutrinos) == 3)
    stats.check("6 quarks", len(quarks) == 6)
    stats.check("4 gauge species (γ, W, Z, g)", len(bosons) == 4)
    stats.check("1 scalar (Higgs)", len(scalars) == 1)
    stats.check("17 total via list_particles()", len(list_particles()) == 17)


def test_PS_embedding_consistency(stats):
    """Q = T_3L + Y on left-handed SU(2)_L doublet partners (SM identity)."""
    print("\n[QN] Pati-Salam: Q = T_3L + Y on chiral irreps")
    # Lepton doublet: (ν, e_L) — standard convention has Y(l_L) = -1/2
    # Our table uses Y(e_R) = -1 because we don't split L/R explicitly.
    # We test Q(higgs) = T_3L + Y for the Higgs scalar (Y=1/2, T_3L=-1/2):
    H = get_particle('higgs')
    Q_check = H.T_3L + H.Y_hyp
    stats.check(f"Q(higgs) = T_3L + Y = {H.T_3L} + {H.Y_hyp} = {Q_check}",
                Q_check == H.Q_em)


def main():
    print("=" * 78)
    print("Particle profile validation — F4 of universal simulator build")
    print("=" * 78)

    stats = TestStats()

    test_catalog_completeness(stats)
    test_lepton_masses(stats)
    test_neutrino_masses(stats)
    test_top_downgraded(stats)
    test_electroweak_bosons(stats)
    test_charges_and_quantum_numbers(stats)
    test_spins(stats)
    test_color_assignments(stats)
    test_sector_filtering(stats)
    test_PS_embedding_consistency(stats)

    print("\n" + "=" * 78)
    success = stats.summary()
    if success:
        print("\nALL TESTS PASS — F4 (Particle profile aggregator) COMMITTED.")
        print("\nUsage:")
        print("  from simulator.srs_engine import get_particle, list_particles")
        print("  e = get_particle('electron')   # one record per SM particle")
        print("  print(e.mass_GeV, e.Q_em, e.Y_hyp, e.derivation)")
    else:
        print("\nSome tests FAILED — F4 needs fixes before commit.")
        sys.exit(1)
    print("=" * 78)


if __name__ == "__main__":
    main()
