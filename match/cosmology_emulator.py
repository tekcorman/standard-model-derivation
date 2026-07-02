"""
Cosmology emulator — counting-first integration of the existing cosmology library.

Wraps proofs/cosmology/lib/ (~3290 lines: ontology, cosmography, bias_functions,
distances, forward_models, fisher, lcdm_fitter, multi_dataset) under a clean
simulator API consistent with the rest of simulator/.

Provides:
- Basic LCDM observables (H_0, t_0, Λ_CC, w_DE, Ω_DM, η_B) — already in
  simulator.predictions.cosmology, exposed here at object level
- Cosmography: H(z), distances at z, age at z
- LCDM emulator: bias function family with z_eff conditional
- Frame-aware ontology: SUBSTRATE / OBSERVER / LCDM_EXTRACTED

Per the cosmology architecture,
this is an LCDM-fit emulator (NOT a substrate fluid simulator). Substrate physics
enters only as project-native primitives (cascade theorem H(z), substrate Ω
partition); LCDM parameters are extracted via the bias function family.

Out of scope (multi-audit BLOCKED):
- Primordial spectrum tilt n_s
- Tensor-to-scalar ratio r
- σ_8 matter clustering amplitude
- Sound horizon r_s, acoustic angular scale θ_*
- Native CMB power spectrum C_l

Predecessors:
- docs/theorems/theorem_cosmology_bias_function_family.md (bias function theorem)
- proofs/cosmology/lib/*.py (existing 3290-line cosmology library)
- simulator/predictions/cosmology.py (basic observables already wrapped)
"""

import sys
import math
from pathlib import Path
from functools import lru_cache, cached_property

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from proofs.cosmology.lib.ontology import Frame, Tagged
from proofs.cosmology.lib.cosmography import (
    H_coasting, H_flat_LCDM, H_flat_wCDM,
    coasting, flat_LCDM,
)
from proofs.cosmology.lib.bias_functions import (
    local_friedmann_two_component,
    Omega_m_local,
    Omega_L_local,
    solve_z_eff_for_Omega_m,
    w_local_at_fixed_Omega_m,
    Omega_m_local_coasting_closed_form,
    w_local_at_fixed_Omega_m_coasting_closed_form,
)
from proofs.cosmology.lib.distances import (
    comoving_distance,
    angular_diameter_distance,
    luminosity_distance,
)

from simulator.srs_engine.kernel import CountingKernel
from .sm_predictions.cosmology import (
    H_0 as _H_0,
    t_0 as _t_0,
    Lambda_CC as _Lambda_CC,
    w_DE as _w_DE,
    Omega_DM_over_Omega_m as _Omega_DM_over_Omega_m,
    eta_B as _eta_B,
    N_HUB,
)


# ============================================================================
# COSMOLOGY EMULATOR — the user-facing class
# ============================================================================

class CosmologyEmulator:
    """LCDM-fit emulator on top of the framework's substrate.

    Wraps the existing cosmology library (proofs/cosmology/lib/) under a
    clean simulator API. Handles substrate-frame, observer-frame, and
    LCDM-extracted quantities with frame-aware ontology.

    Usage:
        cosmo = CosmologyEmulator()

        # Basic observables
        cosmo.H_0_observer()                # 68.18 km/s/Mpc
        cosmo.age()                         # 14.38 Gyr
        cosmo.lambda_cc()                   # 3 / N_hub²

        # Cosmography
        cosmo.H_at(z=1.0, frame='observer')  # H(z) in observer frame
        cosmo.distance(z=1.0, kind='comoving')  # comoving distance

        # LCDM extraction via bias function family
        cosmo.lcdm_extracted(z_eff=1.916)   # all LCDM parameters at z_eff
    """

    # Standard z_eff anchor (data-side, from Planck Ω_m inversion)
    Z_EFF_PLANCK = 1.916

    def __init__(self, kernel=None):
        self.kernel = kernel or CountingKernel()
        # Substrate-frame and observer-frame coasting cosmographies
        # H_0_observer = 68.18 km/s/Mpc; substrate frame is (15/16) of observer
        self._H_0_observer = _H_0()  # 68.18
        self._H_0_substrate = self._H_0_observer * 15.0 / 16.0  # cascade D2-extended

    # ========================================================================
    # BASIC OBSERVABLES (LCDM cascade)
    # ========================================================================

    def H_0_observer(self):
        """H_0 = 68.18 km/s/Mpc — observer-frame Hubble rate.

        UNIQUE-THEOREM-GRADE post G1b R2 closure.
        """
        return self._H_0_observer

    def H_0_substrate(self):
        """H_0 substrate-frame = (15/16) · H_0_observer ≈ 63.92 km/s/Mpc.

        Per cascade D2-extended observer rate gap (theorem-grade).
        """
        return self._H_0_substrate

    def age(self):
        """t_0 = 14.38 Gyr — age of universe in LCDM coasting limit."""
        return _t_0()

    def lambda_cc(self):
        """Λ_CC = 3/N_hub² ≈ 4.26e-122 — cosmological constant in Planck units."""
        return _Lambda_CC()

    def w_dark_energy(self):
        """w_DE = -1 — dark energy equation of state (LCDM-consistent)."""
        return _w_DE()

    def omega_dm_over_omega_m(self):
        """Ω_DM/Ω_m = 0.849 — dark matter fraction (Family 7 MDL waterline)."""
        return _Omega_DM_over_Omega_m()

    def baryon_to_photon_ratio(self):
        """η_B ≈ 6.1e-10 — baryon-to-photon ratio (Family 6 Sakharov)."""
        return _eta_B()

    @property
    def n_hub(self):
        """N_hub ≈ 8.4e60 — total toggle count (cosmology cascade anchor)."""
        return N_HUB

    # ========================================================================
    # COSMOGRAPHY — H(z), distances, age at z
    # ========================================================================

    def H_at(self, z, frame='observer'):
        """Hubble rate at redshift z in the specified frame.

        Frame:
          'observer': H(z) = H_0_observer · (1+z) (coasting, project-native)
          'substrate': H(z) = H_0_substrate · (1+z) (coasting, substrate frame)
        """
        if frame == 'observer':
            return H_coasting(z, self._H_0_observer)
        elif frame == 'substrate':
            return H_coasting(z, self._H_0_substrate)
        else:
            raise ValueError(f"Unknown frame: {frame}")

    # Speed of light (km/s) — needed for distance integrations
    C_KM_S = 299792.458

    def distance(self, z, kind='comoving', frame='observer'):
        """Distance at redshift z.

        Args:
            z: redshift
            kind: 'comoving' / 'angular_diameter' / 'luminosity'
            frame: 'observer' / 'substrate'

        Returns:
            Tagged float (Mpc) — distance with frame ontology attached
        """
        H_0 = self._H_0_observer if frame == 'observer' else self._H_0_substrate
        frame_obj = Frame.OBSERVER if frame == 'observer' else Frame.SUBSTRATE
        cosmography = coasting(H_0, frame_obj)

        if kind == 'comoving':
            return comoving_distance(z, cosmography, self.C_KM_S)
        elif kind == 'angular_diameter':
            return angular_diameter_distance(z, cosmography, self.C_KM_S)
        elif kind == 'luminosity':
            return luminosity_distance(z, cosmography, self.C_KM_S)
        else:
            raise ValueError(f"Unknown distance kind: {kind}")

    def age_at(self, z, frame='observer'):
        """Cosmic age at redshift z (Gyr).

        For coasting H(z) = H_0 (1+z): t(z) = 1 / (H_0 · (1+z)).
        Returns age in Gyr.
        """
        H_0 = self._H_0_observer if frame == 'observer' else self._H_0_substrate

        # Convert H_0 from km/s/Mpc to Gyr^-1
        # 1 km/s/Mpc = 1.022e-12 yr^-1 = 1.022e-3 Gyr^-1
        H_0_per_Gyr = H_0 * 1.022e-3

        # For coasting: t(z) = 1 / (H_0 (1+z))
        return 1.0 / (H_0_per_Gyr * (1.0 + z))

    # ========================================================================
    # LCDM EMULATOR — bias function family
    # ========================================================================

    def lcdm_extracted(self, z_eff=None):
        """LCDM parameters that an external Friedmann fitter would recover.

        Per the bias function theorem: framework predicts coasting H(z);
        humans extract LCDM by fitting Friedmann's two-component class.
        At a single shared conditional z_eff, all extracted LCDM parameters
        are determined.

        Args:
            z_eff: redshift anchor (default: Z_EFF_PLANCK = 1.916, from
                   inverting Planck Ω_m = 0.3153)

        Returns:
            dict with Ω_m_LCDM, Ω_Λ_LCDM, w_LCDM, factor_2_ratio,
            H_0_LCDM (extracted Hubble rate)
        """
        if z_eff is None:
            z_eff = self.Z_EFF_PLANCK

        # Construct project-native coasting cosmography in observer frame
        native = coasting(self._H_0_observer, Frame.OBSERVER)

        # Apply bias function: extract Ω_m_LCDM at z_eff
        Omega_m_LCDM_val = Omega_m_local(native, z_eff)
        Omega_L_LCDM_val = Omega_L_local(native, z_eff)

        # w_DE LCDM-extracted at fixed Omega_m
        w_LCDM_val = w_local_at_fixed_Omega_m(native, z_eff, Omega_m_LCDM_val)

        # Λ ratio LCDM/substrate (the factor-of-2)
        # Λ_LCDM = 3 H_0_LCDM² Ω_Λ_LCDM = (16/15)² · 3 H_0_substrate² · Ω_Λ_LCDM
        # Λ_substrate = 3 H_0_substrate² · Ω_Λ_substrate(z=0) = 3 H_0_substrate² · (1/3) = H_0_substrate²
        # So ratio = (16/15)² · 3 · Ω_Λ_LCDM
        H_0_ratio_sq = (16.0 / 15.0) ** 2
        Omega_L_substrate = 1.0 / 3.0  # 1/k* = 1/3 for k*=3
        lambda_ratio = (H_0_ratio_sq * Omega_L_LCDM_val) / Omega_L_substrate

        return {
            'z_eff': z_eff,
            'Omega_m_LCDM': float(Omega_m_LCDM_val),
            'Omega_Lambda_LCDM': float(Omega_L_LCDM_val),
            'w_DE_LCDM': float(w_LCDM_val),
            'Lambda_LCDM_over_Lambda_substrate': lambda_ratio,
            'H_0_LCDM': self._H_0_observer,
        }

    def bias_function(self, parameter, z):
        """Bias function value: what LCDM parameter at z reproduces native H(z).

        Args:
            parameter: 'Omega_m', 'Omega_Lambda', or 'w_DE'
            z: redshift

        Returns:
            float — the LCDM parameter value at z
        """
        native = coasting(self._H_0_observer, Frame.OBSERVER)

        if parameter == 'Omega_m':
            return float(Omega_m_local(native, z))
        elif parameter == 'Omega_Lambda':
            return float(Omega_L_local(native, z))
        elif parameter == 'w_DE':
            Om = Omega_m_local(native, z)
            return float(w_local_at_fixed_Omega_m(native, z, Om))
        else:
            raise ValueError(f"Unknown parameter: {parameter}")

    def solve_z_eff_from_observation(self, parameter, observed_value):
        """Inverse: find z_eff such that bias_function(parameter, z_eff) = observed.

        Used to anchor z_eff against an empirical Planck (or other) value.
        """
        native = coasting(self._H_0_observer, Frame.OBSERVER)

        if parameter == 'Omega_m':
            return solve_z_eff_for_Omega_m(native, observed_value)
        else:
            raise NotImplementedError(f"solve_z_eff for {parameter} not implemented")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    def summary(self):
        """Return a summary of the cosmology emulator's predictions."""
        lcdm = self.lcdm_extracted()
        return {
            'cosmology_anchor': {
                'N_hub': self.n_hub,
                'H_0_observer_km_s_Mpc': self._H_0_observer,
                'H_0_substrate_km_s_Mpc': self._H_0_substrate,
                'age_Gyr': self.age(),
                'Lambda_CC_planck_units': self.lambda_cc(),
            },
            'matter_content': {
                'w_DE': self.w_dark_energy(),
                'Omega_DM_over_Omega_m': self.omega_dm_over_omega_m(),
                'eta_B': self.baryon_to_photon_ratio(),
            },
            'LCDM_extracted_at_z_eff': lcdm,
            'cosmography_examples': {
                'H_at_z_1': self.H_at(z=1.0),
                'comoving_distance_at_z_1': self.distance(z=1.0, kind='comoving'),
                'age_at_z_1_Gyr': self.age_at(z=1.0),
            },
        }
