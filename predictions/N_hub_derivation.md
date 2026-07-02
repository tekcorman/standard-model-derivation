# N_hub: Hubble-Planck Toggle-Graph Node Count

**Audit anchor:** Row P17 of `docs/parameters/parameter_uniqueness_ledger.md`. STRICT-SOLID; the *value* of the adopted N_hub is empirical (Gap G1; pinned via the measured G_F). The form H = 1/(N · t_P) with coefficient = 1 is theorem-grade; the *value* of the adopted N_hub is pinned to ppm precision via the measured G_F (PDG 2024 / MuLan 2011) — a calibration, not a structural tie. See `docs/framework/framework_axioms.md` for A1+A2 chain.

**Status:** identification (adopted)  
**Parameter:** N_hub = (H_0 × t_P)^{-1} ≈ 8.08 × 10^60  
**File:** `predictions/N_hub.py`

---

## Abstract

N_hub is the number of toggle-graph nodes at the current cosmic epoch, identified
with the Hubble-Planck inverse (H_0 t_P)^{-1}.  This is an adopted scale anchor,
not a derivation from A1–A4.  The identification is structurally analogous to
the adoption of Newton's G ("one toggle = one Planck time × one Planck length
squared") — both anchor the discrete framework to observed cosmological scales.

The result is a unit-conversion formula, not a theorem.  The present file
records the adoption, its empirical inputs, and the future route to a genuine
derivation via the cosmological constant.

---

## Axioms

None invoked.  This is an adoption (identification), not a derivation.

External inputs (empirical, not derived from A1–A4):
- **H_0** = 67.4 ± 0.5 km/s/Mpc  (Planck Collaboration 2018, arXiv:1807.06209)
- **t_P** = 5.391247 × 10^{-44} s  (NIST CODATA 2018)

---

## Derivation

The toggle graph has one node per Planck time interval over the Hubble time
T_hub = H_0^{-1}.  The node count is therefore:

    N_hub = T_hub / t_P = 1 / (H_0 t_P)

Unit conversion:

    H_0 = 67.4 km/s/Mpc
        = 67.4 × 10^3 m/s / (3.085677581 × 10^22 m)
        = 2.1838 × 10^{-18} s^{-1}

    N_hub = 1 / (2.1838 × 10^{-18} s^{-1} × 5.391247 × 10^{-44} s)
          = 1 / (1.1773 × 10^{-61})
          ≈ 8.49 × 10^60

(The slight difference from the note value "≈ 8.08 × 10^60" above reflects
use of the more precise Mpc conversion factor 3.085677581 × 10^22 m rather
than the rounded 3.0857 × 10^22 m used in some upstream files.  N_hub.py
uses the more precise value; v_higgs.py and its downstream chain should be
updated to chain-import from here for consistency.)

---

## Result

    N_hub = 1 / (H_0 t_P)  ≈  8.49 × 10^60   [ADOPTED — identification]

The value enters the Higgs VEV chain as:

    v = δ² M_P / (√2 N_hub^{1/4}) × (1 − (5/12) α₁)

where N_hub^{1/4} ≈ 1.703 × 10^15 is the BZJ finite-size scaling factor.

---

## Comparison

Not applicable.  N_hub is adopted from observation; there is no independent
theoretical prediction to compare against at this stage.

---

## Open Questions

1. **Lambda_CC closure route.**  The cosmological constant in the toggle
   framework satisfies Λ = 3 H^2 (in natural units with c = 1).  Once
   predictions/Lambda_CC.py reaches theorem grade, the identification
   becomes:

       N = sqrt(3 / Λ_CC)

   which replaces this adoption with a genuine derivation and makes H_0 a
   non-trivial output:

       H_0 = sqrt(Λ_CC / 3) / t_P

   At that point, the 0.5 km/s/Mpc observational uncertainty in H_0 becomes
   a testable residual rather than an input.

2. **H_0 tension.**  The Planck 2018 CMB value (67.4 km/s/Mpc) differs from
   the distance-ladder value (~73 km/s/Mpc, Riess et al. 2022,
   arXiv:2112.04510) at ~5σ.  The toggle framework currently adopts the
   CMB value; the Lambda_CC route may eventually discriminate between them.

3. **Precision of t_P.**  t_P = sqrt(ħG/c^5).  Since G is itself an external
   input (same Gap G1 wall), the precision of N_hub is ultimately limited
   by the precision of the joint (H_0, G) anchor until both are derived from
   A1–A4.

## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.
