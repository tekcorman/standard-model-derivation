#!/usr/bin/env python3
"""
Append audit v2 (Clause 7) footers to UNIQUE-THEOREM-GRADE prediction derivations.

For each predictions/*_derivation.md mapped to a UNIQUE-THEOREM-GRADE
parameter ledger row, append a short audit v2 footer per the §5 template
in an internal working note.

The footer cites the index doc + the relevant inheritance closure.
Skips files that already have an "## Audit v2 (Clause 7) status" section.
"""

import os
import re
from pathlib import Path

PREDICTIONS_DIR = Path(__file__).parent.parent.parent / "predictions"

# Map each UNIQUE-THEOREM-GRADE prediction's _derivation.md basename to its
# inheritance category. "row4" = standard Row 4 inheritance (most predictions).
# "row3" = Row 3 (d_spatial). "row1" = Row 1 (p_toggle). "class_A" = Class A
# audit pattern (P5 c=5/12, P28 ε_CP=1/5). "a5b" = ADOPTED-A5b-Sub3 family
# (V_ub, θ_12, θ_13, δ_CP_CKM). "eta_B" = already has full footer.
# "blocked" / "external" = NOT UNIQUE-THEOREM-GRADE; skip.

# Only include UNIQUE-THEOREM-GRADE rows (per parameter_uniqueness_ledger.md
# 2026-04-30 EOD). Excludes BLOCKED, ADOPTED, RETRACTED, and ADVANCED-only.
INHERITANCE = {
    # Row 4 inheritance (k* = 3) — most predictions
    "alpha_1_derivation.md": "row4",
    "alpha_1_full_derivation.md": "row4",
    "V_cb_derivation.md": "row4",
    "V_us_derivation.md": "row4",
    "sin2_theta_W_derivation.md": "row4",
    "y_tau_derivation.md": "row4",
    "Q_Koide_derivation.md": "row4",
    "epsilon_Koide_derivation.md": "row4",
    "delta_Koide_derivation.md": "row4",
    "v_higgs_derivation.md": "row4",
    "m_tau_derivation.md": "row4",
    "m_mu_derivation.md": "row4",
    "m_e_derivation.md": "row4",
    "m_H_derivation.md": "row4",
    "lambda_higgs_derivation.md": "row4",
    "theta_23_PMNS_derivation.md": "row4",
    "theta_QCD_derivation.md": "row4",
    "N_hub_derivation.md": "row4",
    "H_0_derivation.md": "row4",
    "t_0_derivation.md": "row4",
    "w_DE_derivation.md": "row4",
    "Omega_DM_over_Omega_m_derivation.md": "row4",
    "A_hemispherical_derivation.md": "row4",
    "R_nu_splitting_derivation.md": "row4",
    "alpha_GUT_derivation.md": "row4",
    "eta_5_lorentz_dim5_derivation.md": "row4",
    "eta_lattice_lorentz_dim6_derivation.md": "row4",
    "beta_cosmic_birefringence_derivation.md": "row4",
    "feshbach_exponent_principle_derivation.md": "row4",
    "h_walker_eigenvalue_derivation.md": "row4",
    "B_P_doubly_degenerate_h_derivation.md": "row4",
    "srs_E_at_P_derivation.md": "row4",
    "srs_cubic_moment_derivation.md": "row4",
    "srs_bloch_dispersion_gamma_derivation.md": "row4",
    "g_girth_derivation.md": "row4",
    "k_star_derivation.md": "row4",
    "uniform_Q_density_derivation.md": "row4",
    "koide_quark_ratio_derivation.md": "row4",
    # Row 3 inheritance
    "d_spatial_derivation.md": "row3",
    # Row 1 inheritance (p=2 toggle arity)
    "p_toggle_derivation.md": "row1",
    # Class A audit pattern (downgraded to DOMINANT-CONDITIONAL)
    # No specific dark-coefficient _derivation.md exists; 5/12 is in
    # docs/theorems/theorem_dark_correction_mdl.md. ε_CP also has no dedicated file.
    # ADOPTED-A5b-Sub3 family (STRICT-SOLID-CONDITIONAL)
    "V_ub_derivation.md": "a5b",
    "delta_CP_CKM_geometry_derivation.md": "a5b",
    # η_B already has full footer
    "eta_B_derivation.md": "eta_B",
    # m_ν (P31) is STRICT-SOLID-CONDITIONAL on ADOPTED-PS, not Row 4 cleanly
    "m_nu2_derivation.md": "ps",
    "m_nu3_derivation.md": "ps",
}


FOOTER_ROW4 = """
## Audit v2 (Clause 7) status

This prediction inherits Row 4 (k* = 3) audit v2 closure. The §3 multi-mechanism
table (M1-M6 vs qtz at k=4) is consolidated in
an internal working note §2.1; closure doc:
an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 4 audit v2 closure (DOMINANT-with-named-margins; UNIQUE-on-η_B).
- **Named margin** (from index §2.1): combined M2 (~0.45 Boltzmann) × observable-specific M3·M4·M5 product.
- **Inherits structurally:** Row 4 v2 closure protects against qtz at k=4 alternative via combined mechanism product. M6 sign-flip is irrelevant for this observable (uses Im(h)/|h|² or magnitude, not Re(h) sign) unless the observable uses Re(h_P) directly (η_B specifically; see `predictions/eta_B_derivation.md` for that case).
- **Conditionals deferred:** RCSR-vetted qtz bond list verification; selection-rule audit; data-conditional MDL.
"""

FOOTER_ROW3 = """
## Audit v2 (Clause 7) status

This prediction inherits Row 3 (d = 3) audit v2 closure. See
an internal working note §3 (foundational rows)
and an internal working note.

- **Status (post-audit-v2):** UNIQUE-THEOREM-GRADE-CONDITIONAL on Row 3 audit v2 (DOMINANT via Gleason d≥3 + R-4/R-5 empirical anchors + M5 no new amplification).
- **Named conditional:** M1 R-N hard-gates at d-axis (Gleason d=2, R-4 d=4, R-5 d≥5). M5 non-local amplification check returns no new amplification for d=4 alternatives (η_5 = 0 empirical anchor stands).
"""

FOOTER_ROW1 = """
## Audit v2 (Clause 7) status

This prediction inherits Row 1 (p = 2 toggle arity) audit v2 closure. See
an internal working note §3 and
an internal working note Phase 1b.

- **Status (post-audit-v2):** DOMINANT (margin +1 bit/state) — A1 axiom written for p=2 (T·T=I); alternative p≥3 doesn't fit A1 cleanly. M2 ΔDL +1 bit confirms.
- **Named margin:** +1 bit/state vs alternative arities.
"""

FOOTER_A5B = """
## Audit v2 (Clause 7) status

This prediction inherits Row 4 audit v2 closure + ADOPTED-A5b-Sub3 conditional.
See an internal working note and
an internal working note.

- **Status (post-audit-v2):** STRICT-SOLID-CONDITIONAL on (ADOPTED-A5b-Sub3 + Row 4 audit v2). The amplitude / geometric formula is sound; substrate-side mass-eigenstate identification rests on un-graduated framework adoption (see an internal working note).
- **Named margin:** observed within ~0.5σ of best-measured value; ADOPTED-A5b-Sub3 is the dominant un-graduated conditional, Row 4 audit v2 is parallel conditional.
"""

FOOTER_PS = """
## Audit v2 (Clause 7) status

This prediction inherits Row 4 audit v2 closure + ADOPTED-PS-SCALE conditional.
See an internal working note.

- **Status (post-audit-v2):** STRICT-SOLID-CONDITIONAL on (ADOPTED-PS-SCALE + Row 4 audit v2). The dark-correction form Im(h)/|h|² is theorem-grade (per `docs/theorems/theorem_m_nu_dark_correction_uniqueness_closure.md`); the bare-scale m_{ν3}^bare derivation remains the un-graduated open input.
- **Named margin:** ADOPTED-PS-SCALE is the dominant un-graduated conditional.
"""

FOOTER_ETAB_NOTE = "ETA_B_ALREADY_HAS_FOOTER"


def already_has_footer(content):
    """Check if the file already has an audit v2 footer."""
    return "## Audit v2 (Clause 7) status" in content or "## Audit v2 (Clause 7)" in content


def insert_footer(content, footer):
    """Insert footer before '## Cross-references' or '## References' if present, else append."""
    # Try to insert before existing cross-references / references / files-referenced section
    for marker in ["## Cross-references", "## References", "## Files referenced", "## Footnote"]:
        if marker in content:
            return content.replace(marker, footer.rstrip() + "\n\n" + marker, 1)
    # Else append at end
    if not content.endswith("\n"):
        content += "\n"
    return content + footer


FOOTERS = {
    "row4": FOOTER_ROW4,
    "row3": FOOTER_ROW3,
    "row1": FOOTER_ROW1,
    "a5b": FOOTER_A5B,
    "ps": FOOTER_PS,
    "eta_B": FOOTER_ETAB_NOTE,
}


def main():
    skipped_already = []
    skipped_eta_b = []
    skipped_missing = []
    updated = []

    for filename, category in INHERITANCE.items():
        filepath = PREDICTIONS_DIR / filename
        if not filepath.exists():
            skipped_missing.append(filename)
            continue
        content = filepath.read_text()
        if already_has_footer(content):
            skipped_already.append(filename)
            continue
        if category == "eta_B":
            skipped_eta_b.append(filename)
            continue
        footer = FOOTERS[category]
        new_content = insert_footer(content, footer)
        filepath.write_text(new_content)
        updated.append((filename, category))

    print(f"Updated {len(updated)} files:")
    for f, cat in updated:
        print(f"  [{cat:6s}] {f}")
    print()
    if skipped_already:
        print(f"Skipped (already has footer): {len(skipped_already)}")
        for f in skipped_already:
            print(f"  {f}")
    if skipped_eta_b:
        print(f"Skipped (eta_B already has full footer): {len(skipped_eta_b)}")
    if skipped_missing:
        print(f"Skipped (file missing): {len(skipped_missing)}")
        for f in skipped_missing:
            print(f"  {f}")


if __name__ == "__main__":
    main()
