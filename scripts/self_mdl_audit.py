#!/usr/bin/env python3
"""Self-MDL audit: is the framework shorter than the data it predicts?

Computes the framework's net compression in bits:

    net = earned - spent

  earned : information content of the measured quantities the framework
           closes, scored only down to the precision the framework ACTUALLY
           achieves (achieved = max(sigma_exp, |residual|)).
  spent  : bits to specify the framework itself -- every discrete choice
           among operator-permitted alternatives, every adoption, every
           continuous calibration, and the A5 labeling.

Methodology is FROZEN in docs/audits/registers/self_mdl_ledger.md.
Conservative-direction rule: when uncertain, earned rounds DOWN and
spent rounds UP. Output is PROVISIONAL v0 pending the adversarial
methodology review (ultracode gate) and full P-row coverage.

Run from repo root:  python3 scripts/self_mdl_audit.py
"""
from math import log2, log, pi

# ---------------------------------------------------------------------------
# Prior conventions (FROZEN -- see self_mdl_ledger.md "Prior classes")
# ---------------------------------------------------------------------------
W_MASS = log(1.22e28 / 1e-3)      # log-uniform mass prior [1e-3 eV, M_Pl], nats
W_COUPLING = log(4 * pi / 1e-6)   # log-uniform coupling prior [1e-6, 4*pi], nats
W_RATIO = log(1e3 / 1e-3)         # log-uniform O(1) ratio prior [1e-3, 1e3], nats
W_SMALL = log(1.0 / 1e-15)        # log-uniform small-ratio prior [1e-15, 1], nats


def uniform(width, achieved):
    """Uniform prior of total width `width`; achieved absolute width."""
    return max(0.0, log2(width / achieved))


def loguni(w_nats, rel_achieved):
    """Log-uniform prior of width w_nats; achieved relative width."""
    return max(0.0, log2(w_nats / rel_achieved))


# ---------------------------------------------------------------------------
# EARNED -- independent generating set only (no derived combos double-counted)
# Each row: (name, bits, justification). Residuals from the live scoreboard;
# achieved = max(sigma_exp, |residual|). Rows the framework does NOT close
# (n_s, sigma_8, r_s, theta_*) earn nothing and are listed as zero for honesty.
# ---------------------------------------------------------------------------
EARNED = [
    # CKM (4 independent parameters of the standard parameterization)
    ("V_us",        uniform(1, 6.8e-4),        "uniform(0,1); sigma_exp 6.8e-4, residual -0.01sigma"),
    ("V_cb",        uniform(1, 9.0e-4),        "uniform(0,1); sigma_exp 9.0e-4, residual 0.00sigma"),
    ("V_ub",        uniform(1, 2.0e-4),        "uniform(0,1); sigma_exp 2.0e-4, residual -0.26sigma"),
    ("delta_CKM",   uniform(360, 3.0),         "phase deg; achieved max(3.0 sigma_exp, 2.0 resid)"),
    # PMNS
    ("theta12_PMNS", uniform(90, 0.75),        "angle deg; sigma_exp 0.75"),
    ("theta13_PMNS", uniform(90, 0.12),        "angle deg; sigma_exp 0.12"),
    ("theta23_PMNS", uniform(90, 1.3),         "angle deg; achieved max(1.3 sigma_exp, 0.48 resid)"),
    ("delta_PMNS",  uniform(360, 20.0),        "phase deg; sigma_exp ~20"),
    ("R_nu",        loguni(W_RATIO, 0.01),     "Dm2 ratio 228/7; achieved ~1% (NuFIT Dm2 precisions)"),
    # Strong CP
    ("theta_QCD",   uniform(2 * pi, 1e-10),    "exact-zero vs bound 1e-10 over [0,2pi)"),
    # Charged fermion masses (9) + 2 neutrino scales (m_nu1=0 counted at bound)
    ("m_t",   loguni(W_MASS, 8.2e-3),  "achieved |resid| 0.82% (> sigma_exp)"),
    ("m_b",   loguni(W_MASS, 2.15e-2), "achieved |resid| 2.15%"),
    ("m_c",   loguni(W_MASS, 1.6e-2),  "within 1sigma; sigma_exp ~1.6%"),
    ("m_s",   loguni(W_MASS, 5.0e-2),  "within 1sigma; sigma_exp ~5%"),
    ("m_d",   loguni(W_MASS, 1.0e-1),  "within 1sigma; sigma_exp ~10% (conservative)"),
    ("m_u",   loguni(W_MASS, 3.0e-1),  "within 1sigma; sigma_exp ~30% (conservative)"),
    ("m_tau", loguni(W_MASS, 6.8e-5),  "within sigma_exp 6.8e-5 (residual -0.19sigma)"),
    ("m_mu",  loguni(W_MASS, 1.0e-2),  "ESTIMATE-CONSERVATIVE 1% pending M_persistence row audit"),
    ("m_e",   loguni(W_MASS, 1.0e-2),  "ESTIMATE-CONSERVATIVE 1% pending M_persistence row audit"),
    ("m_nu2", loguni(W_MASS, 3.0e-2),  "8.86 meV; achieved ~ oscillation precision 3%"),
    ("m_nu3", loguni(W_MASS, 1.5e-2),  "50.57 meV; achieved ~ oscillation precision 1.5%"),
    # Higgs sector (v EXCLUDED: it is the N_hub<->G_F calibration, paid in SPENT)
    ("m_H",      loguni(W_MASS, 8.8e-4),     "sigma_exp 0.11/125.2; residual -0.05sigma"),
    ("lambda_H", loguni(W_COUPLING, 1e-2),   "CONSERVATIVE 1% (quote -0.05sigma)"),
    # Gauge sector at M_Z (conditional on paid ADOPTED-MSSM-Sb)
    ("alpha_s",     loguni(W_COUPLING, 8e-3),  "achieved ~0.8%"),
    ("sin2thetaW",  uniform(1, 1e-3),          "ESTIMATE-CONSERVATIVE 1e-3 (EW ppm floor NOT credited)"),
    ("alpha_EM",    loguni(W_COUPLING, 1e-3),  "ESTIMATE-CONSERVATIVE 1e-3 (Delta-alpha gap; ppm NOT credited)"),
    # Cosmology
    ("eta_B",        loguni(W_SMALL, 6.5e-3),  "sigma_exp 0.04/6.12; residual -0.20sigma"),
    ("Omega_DM/m",   uniform(1, 1.6e-2),       "uniform(0,1); sigma_exp 0.016"),
    ("H_0",          loguni(W_MASS, 1.16e-2),  "achieved |resid| 1.16% vs Planck (t_0 excluded: derived)"),
    ("w_DE",         uniform(2, 3e-2),         "uniform[-2,0]; sigma_exp 0.03"),
    ("Lambda_CC",    8.8,                      "ESTIMATE: 130-decade log prior vs factor-2 achieved"),
    ("A_s",          8.0,                      "ESTIMATE-CONSERVATIVE prefactor ~few %"),
    ("beta_biref",   9.0,                      "ESTIMATE: prior ~1 rad vs sigma_exp 0.094 deg"),
    ("A_hemis",      uniform(1, 2e-2),         "uniform(0,1); sigma_exp 0.02"),
    # Open cluster earns NOTHING (honesty rows)
    ("n_s",     0.0, "OPEN (L6) -- earns 0"),
    ("sigma_8", 0.0, "OPEN (L6) -- earns 0"),
    ("r_s",     0.0, "OPEN (L6) -- earns 0"),
    ("theta_*", 0.0, "OPEN (L6) -- earns 0"),
]

# ---------------------------------------------------------------------------
# SPENT -- every choice point, adoption, calibration, labeling.
# Sources: uniqueness_ledger.md (25 rows), adoption_register.md (7 ACTIVE),
# A5 clauses. UNIQUE-unconditional rows pay 0 and are not listed.
# DEFAULT = frozen conservative menu size 8 (3 bits) when row does not
# enumerate its alternatives.
# ---------------------------------------------------------------------------
SPENT = [
    ("axiom-slate (A)+(B)+(I)",      10.0, "CONVENTION: flat charge for the commitment slate"),
    ("k* coordination (O1 joints)",   1.0, "Gleason+MDL derivation; 1 bit contingency while O1 spec open"),
    ("substrate srs (Row 6)",         2.0, "superposition reframe: strong-isotropy UNDISCHARGED; srs/srs-z/higher-k"),
    ("A4 CAR (fermionic)",            1.0, "fermionic vs bosonic READING of the edge algebra (selection bit; ordering-free global-CAR CONSTRUCTIBILITY closed by 5.3 B3 bridge + panel, tree/star/cyclic clusters; the commuting toggle reading remains available in the same ambient; even-subalgebra-physicality/parity-superselection covered by this bit until derived; discharge route = MDL-Fock A2 comparison, unexecuted)"),
    ("A5-mass labeling",              2.0, "5.2 PANEL + ORDERED CHECK 2026-06-11: P-sign bit = omega/omega2 channel-labeling convention (probe phase5_2_psign_omega_identity I1-I5; single-homed at its priced ~1-bit Majorana-panel ledger line, R3/M1.B+B3 complex); in-row = Higgs placement 1.585 -> 2.0 up; nu-orientation at the 1.3 line; (a) -11.71 was a v0 over-price, NOT an EBR result; remaining sensitivity: R1 ratified -> 0 in-row"),
    ("A5(b) channel levels",         19.0, "12 observables x log2(3) level choices (TODO: refine vs Clause-7 defenses)"),
    ("ADOPTED-MSSM-Sb",               3.0, "DEFAULT(8): matter-content/RG-scheme menu"),
    ("ADOPTED-NU-MAJ-PHASE",          3.0, "DEFAULT(8): M_R phase identification"),
    ("ADOPTED-PS-SCALE",              3.0, "DEFAULT(8): nu_R bare-scale source"),
    ("ADOPTED-DARK-MAP",              3.0, "DEFAULT(8): tan^2(arg h) correction form"),
    ("ADOPTED-K_P-TIEBREAK",          2.0, "k_P selection tiebreaker"),
    ("N_hub <-> G_F calibration",    27.1, "continuous: log2(W_MASS / 5e-7) -- one full measured constant"),
    ("4.3 sector-anchor bridge",      1.0, "PHASE-4 PANEL 2026-06-12: dictionary-licensed triple-H -> 3 generations; carries the absolute b3=-7 (native trace 8 -> -17/3 = no target); carved out of the buffer, cross-ref A5-mass chain"),
    ("Phi-condensate Dirac-class",    1.6, "PHASE-4 PANEL: the mirror-even condensate is D2-SPECIFIC (D3 mirror-odd, D1 mixed); log2(3); Higgs/mirror rows only; -> 0 if scheme->D2 forcing ratified"),
    ("JAJ^-1 scheme-truncation",      1.0, "PHASE-4 PANEL: frozen fluctuation scheme ran A-only; gauge-normalization rows uncertified; DISCHARGEABLE by erratum probe E3"),
    ("c_S cut-defect identification", 1.0, "PHASE-4 PANEL: gravitating entropy := the cut-localized additivity-defect object (role assignment beyond I2); adoption-class; single-homed at the gravity final-state doc"),
    ("form-selection residual",      15.0, "1 bit x ~15 earned rows graded non-UNIQUE (frozen v0 rule)"),
    ("open-joint contingency",       10.0, "CONVENTION: flat buffer for unenumerated freedom (adversarial review target)"),
]


def main():
    print("=" * 74)
    print("SELF-MDL AUDIT v0 (PROVISIONAL -- pending adversarial methodology review)")
    print("=" * 74)
    print("\n-- EARNED (compression achieved, conservative) " + "-" * 26)
    e_total = 0.0
    for name, bits, why in EARNED:
        e_total += bits
        print(f"  {name:<14} {bits:7.1f}  {why}")
    print(f"  {'EARNED TOTAL':<14} {e_total:7.1f}")
    print("\n-- SPENT (framework specification, conservative) " + "-" * 24)
    s_total = 0.0
    for name, bits, why in SPENT:
        s_total += bits
        print(f"  {name:<28} {bits:6.1f}  {why}")
    print(f"  {'SPENT TOTAL':<28} {s_total:6.1f}")
    net = e_total - s_total
    print("\n" + "=" * 74)
    print(f"  NET COMPRESSION = {e_total:.1f} - {s_total:.1f} = {net:+.1f} bits")
    print(f"  ratio earned/spent = {e_total / s_total:.2f}")
    print("=" * 74)
    print("""
  Reading: net > 0 means the framework's specification is shorter than the
  information it recovers from measurement -- the MDL standard of evidence.
  v0 caveats: (a) ESTIMATE rows on both sides; (b) the SPENT side is only as
  complete as the enumerated choice points -- the adversarial review's job is
  to find UNCOUNTED freedom (reading rules, post-hoc form choices). A positive
  net survives only if that review converges.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
