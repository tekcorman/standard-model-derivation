#!/usr/bin/env python3
"""
derivation_topdown/adapters/reads_manifest.py

S1 -- THE READS MANIFEST.  Pre-registered FROZEN in
internal research notes (read that file first; this implements it
verbatim).  Symphony Program keystone: the layer connecting the parameter ledger
(docs/parameters/target_parameters.md) to the certified engine (the_run.py / the_net.py /
srs.py).  It classifies EVERY ledger row as a read of the triple (D, omega, {A(O)}), recomputes
what it can from ENGINE PRIMITIVES ONLY, compares against the frozen value locks
(predictions/_value_locks.json), and publishes the coverage numbers + the full resistance list.

WHAT THIS FILE IS NOT: a new derivation.  It is a verification/classification adapter, same
genre as derivation_topdown/adapters/{aqft_net,furey_stoica_labels,thermal_time,sunada_geometry,
zeta_gauge,ncg_spectral}.py.  It NEVER edits predictions/, the locks, or the engine.  A mismatch
is a FINDING, not a bug to tune away (no tolerance loosening; no re-mapping after seeing misses).

READ-ONLY INPUTS (never edited by this file):
  docs/parameters/target_parameters.md      the ~150-row ledger (parsed below)
  predictions/_value_locks.json             the 107 frozen comparison values (104 as of the S1
                                             pre-reg 2026-07-09 + 3 S1b ORPHAN CLEANUP additions the
                                             same day: T_nu_dec, h_walker_eigenvalue_re/_im -- see
                                             MAPPING-REVISIONS below)
  derivation_topdown/bridge/the_run.py       the engine (Layer 1, D's global spectrum)
  derivation_topdown/state/the_net.py        the engine (Layer 3, {A(O)} -- mostly still UNBUILT)
  derivation_topdown/dirac_srs_mdl/srs.py    the engine primitives (walled-off clean room)
  internal research notes   the bin classification seed
  docs/audits/registers/adoption_register.md  adoption names cited for Tier-B rows

HARD POISONS (binding, see the pre-reg):
  - predictions/ code is NEVER imported or executed anywhere in this file.
  - The M-1 mapping table (PHASE 2) is built from the engine's OWN docstrings/outputs, printed IN
    FULL, and FROZEN at print time -- PHASE 3's comparisons consume it as-is.  No re-mapping after
    seeing mismatches (a genuine mapping slip found during development is logged in the
    MAPPING-REVISIONS section below, with the whole pipeline re-run from PHASE 1 -- full
    disclosure, before-comparison discipline restored).
  - No tolerance loosening.  Tier-C is published in full.  Misses stay misses.

USAGE:
    python3 derivation_topdown/adapters/reads_manifest.py            # full run, print only
    python3 derivation_topdown/adapters/reads_manifest.py --write    # full run + write the doc
    python3 derivation_topdown/adapters/reads_manifest.py --fast     # M-5: PHASE 0+3 only (verify hook)
"""
import argparse
import math
import os
import re
import sys
import time
from fractions import Fraction

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "derivation_topdown", "bridge"))
sys.path.insert(0, os.path.join(_REPO, "derivation_topdown", "state"))
sys.path.insert(0, os.path.join(_REPO, "derivation_topdown", "dirac_srs_mdl"))
import the_run as R    # noqa: E402  -- THE ENGINE (Layer 1).  Never edited.
import the_net as NET   # noqa: E402  -- THE ENGINE (Layer 3, mostly UNBUILT).  Never edited.
import srs              # noqa: E402  -- engine primitives (walled-off clean room).  Never edited.

LEDGER_PATH = os.path.join(_REPO, "docs", "parameters", "target_parameters.md")
LOCKS_PATH = os.path.join(_REPO, "predictions", "_value_locks.json")
OUT_PATH = os.path.join(_REPO, "docs", "parameters", "reads_manifest.md")
BINS_DOC = "internal research notes"
ADOPTION_REG = "docs/audits/registers/adoption_register.md"

TOL_EXACT = 1e-12     # exact-form values (rationals / closed forms / integers)
TOL_FLOAT = 1e-9      # floats: relative tolerance

# S1b ORPHAN CLEANUP (2026-07-09, user-approved re-freeze): the lock file grew from the S1
# pre-reg's 104 to 107 -- +3 (T_nu_dec, h_walker_eigenvalue_re, h_walker_eigenvalue_im). This
# constant is the single source of truth for the M-0 PARSE gate below (was a bare "== 104").
# R1 HARVEST (2026-07-10, internal research notes): 107 -> 134, +27
# additive 'harvest_*' locks (H-1 coasting chain + H-2 composites + H-3 m_bb + H-5 partial
# structural wiring); the prior 107 are bit-identical (scripts/value_lock.py verified before
# the re-freeze -- see predictions/_value_locks.json's own _meta.refreeze_note).
N_LOCKS_EXPECTED = 134

def banner(t):
    print("=" * 90)
    print(" " + t)
    print("=" * 90)


# ==============================================================================================
# PHASE 0 -- PARSE the ledger (M-0) + LOAD the locks (M-0)
# ==============================================================================================
_HEADER_SEP = re.compile(r"^\|[\s:\-|]+\|$")
_SECTION_MARK = re.compile(r"^\*\*(§\d[^*]*)\*\*$")
_ESC_PIPE = "\x00ESCPIPE\x00"


def _split_row(line):
    """Split a GFM table row on '|', respecting ESCAPED pipes ('\\|', a literal pipe INSIDE a
    cell -- e.g. absolute-value bars '\\|E\\|' in the Notes prose) which must NOT be treated as
    cell delimiters.  Confirmed necessary 2026-07-09: several ledger rows (g_3, alpha_s, M_Z,
    Gamma_Z/M_Z, delta_rho, delta_r, m_H, m_nu1, eta_5) carry literal '\\|x\\|' math notation in
    their Notes column; a naive split() silently drops these rows (cell-count mismatch)."""
    protected = line.replace("\\|", _ESC_PIPE)
    cells = [c.strip().replace(_ESC_PIPE, "|") for c in protected.strip().strip("|").split("|")]
    return cells


def parse_ledger(path):
    """Generic GFM-table extractor for target_parameters.md.  Accepts any table whose FIRST
    header cell is 'Symbol' or 'Item' (this cleanly excludes the doc's summary/count tables,
    whose first header cell is 'Honest category' / 'Sector' / etc -- verified by inspection,
    2026-07-09 sweep).  Column schema varies by table (Symbol|Observed|Predicted|Status|File|Notes
    vs Item|Content|Status|File|Notes vs Item|Predicted|Status|File|Notes...); we key by the
    ACTUAL header cell names per table, then normalize downstream."""
    text = open(path, encoding="utf-8").read()
    lines = text.split("\n")
    rows = []
    section, subsection = None, None
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("## "):
            section, subsection = line[3:].strip(), None
        elif line.startswith("### "):
            subsection = line[4:].strip()
        m = _SECTION_MARK.match(line.strip())
        if m:
            subsection = m.group(1)
        if line.startswith("|") and line.count("|") >= 3:
            cells = _split_row(line)
            if cells and cells[0] in ("Symbol", "Item") and i + 1 < n and _HEADER_SEP.match(lines[i + 1].strip()):
                header = cells
                j = i + 2
                while j < n and lines[j].strip().startswith("|"):
                    rcells = _split_row(lines[j])
                    if len(rcells) > len(header):
                        # safety net: any residual extra cells (beyond escaped-pipe fix) are
                        # folded into the final column rather than silently dropping the row.
                        rcells = rcells[:len(header) - 1] + ["|".join(rcells[len(header) - 1:])]
                    if len(rcells) == len(header):
                        # normalize header cells that carry a qualifier suffix, e.g. "Predicted
                        # (coasting, pending lint)" in the §4 distance-propagation table -- match
                        # by PREFIX so those rows' Predicted column is not silently dropped.
                        norm_header = [
                            "Predicted" if h.startswith("Predicted") else
                            "Observed" if h.startswith("Observed") else h
                            for h in header
                        ]
                        row = dict(zip(norm_header, rcells))
                        symbol = row.get("Symbol") or row.get("Item") or ""
                        predicted = row.get("Predicted") or row.get("Content") or ""
                        rows.append({
                            "symbol": symbol.strip("` "), "observed": row.get("Observed", ""),
                            "predicted": predicted, "status": row.get("Status", ""),
                            "file": row.get("File", ""), "notes": row.get("Notes", ""),
                            "section": section, "subsection": subsection,
                        })
                    else:
                        print(f"  [PARSE-DROP] row cell-count {len(rcells)} != header {len(header)} "
                              f"at line {j+1}: {lines[j][:80]!r}", file=sys.stderr)
                    j += 1
                i = j
                continue
        i += 1
    return rows


def load_locks(path):
    import json
    d = json.load(open(path))
    return d["values"], d["_meta"]


# ==============================================================================================
# PHASE 1 -- CALL EVERY ENGINE PUBLIC READ; flatten into (key, value); PRINT the full inventory.
#   the_run.py's declared read surface (its own module docstring, verbatim):
#     read_girth / read_geometry / read_clock / read_gauge / read_gauge_running / gauge_dynkin /
#     read_flavor / read_generation / read_mass / read_masses / read_selection /
#     read_dirac4_lift / read_phases / read_moduli / read_democratic / read_vertex /
#     read_obliques / read_run
#   the_net.py's reads: the two REGRESSION ANCHORS + the ML-2/ML-3 structural reads.
#   srs.py primitives: NV, DEG, EDGES (+ their counts).
#   Each engine call happens ONCE (cached in the returned dict) per the "cache reads" hard rule.
# ==============================================================================================
def phase1_engine_reads():
    E = {}

    # -- srs.py primitives (the walled-off clean room) --------------------------------------
    E["srs.NV"] = srs.NV
    E["srs.NE"] = len(srs.EDGES)
    E["srs.DEG"] = srs.DEG

    # -- the_run.py module-level forced constants (read at import time, not re-typed here) ---
    E["run.GIRTH"] = R.GIRTH
    E["run.P_TOGGLE"] = R.P_TOGGLE
    E["run.LAM_PERRON"], E["run.LAM_3IRREP"] = R.adjacency_energies()
    E["run.U_RUN(alpha_1)"] = R.U_RUN
    # alpha_1_full: the LITERAL formula at the top of read_moduli() -- Fraction(5,3)*((k-1)/k)^(g-2).
    # Not itself returned by a named function (read_moduli returns PER-SPECIES Q, not this bare
    # constant), so we replicate the identical source-code expression using only engine constants
    # K=srs.DEG and GIRTH -- this is the engine's own formula, not an external guess.
    E["run.alpha_1_full(formula)"] = float(Fraction(5, 3) * Fraction(R.K - 1, R.K) ** (R.GIRTH - 2))
    # aliases: the SAME already-computed engine primitives, named a second way for a second lock
    # (srs.DEG doubles for both k_star and georgi_jarlskog; sqrt(LAM_PERRON) is the "energy at P").
    E["srs.DEG(as-GJ)"] = srs.DEG
    E["srs_E_at_P(sqrt-Perron)"] = math.sqrt(E["run.LAM_PERRON"])

    # -- S1b ORPHAN CLEANUP (2026-07-09): the P-point walker root h=(sqrt3+i*sqrt5)/2, read DIRECTLY
    #    via the_run.py's own module-level ihara_bass_root(lam) at lam=sqrt(LAM_PERRON) -- the
    #    IDENTICAL native call read_obliques() (line ~466) and the PMNS Majorana-phase construction
    #    inside read_ported_flavor() (lines ~708-710, ~1040-1043) already make internally (their own
    #    comments: "the SAME P-point walker root already read above via
    #    ihara_bass_root(sqrt(LAM_PERRON))"). Not previously flattened to a standalone engine-output
    #    entry (read_obliques() only returns F=h.imag/|h|^2, never h itself) -- added so the lock's
    #    split real/imag pair (predictions/h_walker_eigenvalue.py's complex h) gets a genuine
    #    Tier-A comparison.
    _h_walker = R.ihara_bass_root(math.sqrt(E["run.LAM_PERRON"]))
    E["run.ihara_bass_root(sqrt_LAM_PERRON).re"] = _h_walker.real
    E["run.ihara_bass_root(sqrt_LAM_PERRON).im"] = _h_walker.imag

    # -- read_girth / read_geometry / read_clock / read_gauge (named, zero-arg) --------------
    E["read_girth()"] = R.read_girth()
    b1, Lam2 = R.read_geometry()
    E["read_geometry().b1"], E["read_geometry().Lam2"] = b1, Lam2
    E["geometry.b1"] = b1  # alias used by TIER_A_MAP
    eps, clock = R.read_clock()
    E["read_clock().eps"], E["read_clock().clock"] = float(eps), float(clock)
    E["read_gauge().sin2thetaW"] = float(R.read_gauge())
    E["gauge.sin2thetaW"] = E["read_gauge().sin2thetaW"]  # alias used by TIER_A_MAP

    # -- read_gauge_running() (named, zero-arg; internal helper gauge_dynkin is NOT separately
    #    callable -- it needs (fields, mult) and is only ever invoked FROM read_gauge_running) --
    gr = R.read_gauge_running()
    for gi in (1, 2, 3):
        add, b4d, blit, agree = gr[gi]
        E[f"read_gauge_running().b{gi}_add"] = float(add)
        E[f"read_gauge_running().b{gi}_4d"] = float(b4d)
        E[f"read_gauge_running().b{gi}_MSSMlit"] = float(blit)
        E[f"read_gauge_running().b{gi}_agree"] = bool(agree)
    E["gauge_dynkin(...)"] = "NOT independently callable (needs (fields,mult)); only invoked from read_gauge_running -- noted, not flattened"

    # -- read_dirac4_lift() (named, default k) -----------------------------------------------
    anti, clean = R.read_dirac4_lift()
    E["read_dirac4_lift().anti"], E["read_dirac4_lift().clean"] = anti, bool(clean)

    # -- read_matter_row() (native helper referenced in the docstring's "matter row") --------
    mspec, mtriple, mweyl = R.read_matter_row()
    E["read_matter_row().triple"], E["read_matter_row().weyl"] = mtriple, mweyl

    # -- read_flavor() (named, zero-arg) -----------------------------------------------------
    fock, rho, Q, gens = R.read_flavor()
    E["read_flavor().fock"] = fock
    E["read_flavor().rho"] = float(rho)
    E["read_flavor().Q"] = float(Q)
    E["read_flavor().gens"] = gens
    E["flavor.Q"] = E["read_flavor().Q"]        # alias used by TIER_A_MAP
    E["flavor.gens"] = E["read_flavor().gens"]  # alias used by TIER_A_MAP

    # -- read_species() --------------------------------------------------------------------
    sp = R.read_species()
    for n, m in sp.items():
        E[f"read_species().mult[{n}]"] = m

    # -- read_generation(s) (named, but REQUIRES the free axis s -- FLAGGED non-native: the
    #    __main__ demo CALIBRATES s by bisecting against the OBSERVED m_mu/m_e=206.7683, so any
    #    "prediction" from this route for that SAME ratio would be circular.  We do NOT invoke it
    #    standalone for a Tier-A/B claim; noted honestly, not silently skipped.) -----------------
    E["read_generation(s)"] = "NOT zero-arg -- requires free axis s; the __main__ demo CALIBRATES s from the observed m_mu/m_e (not predictive for that ratio) -- excluded from Tier-A/B claims, noted here for honesty"

    # -- read_mass() (named, zero-arg) -------------------------------------------------------
    shell, disc, dtheta_ds = R.read_mass()
    E["read_mass().shell"], E["read_mass().disc"], E["read_mass().dtheta_ds"] = shell, disc, dtheta_ds
    E["mass.sqrt_shell"] = math.sqrt(shell)  # alias used by TIER_A_MAP (epsilon_Koide = sqrt(k*-1))

    # -- read_masses() (named, zero-arg; FULLY FORCED -- no free parameter, unlike read_generation) --
    ms = R.read_masses()
    for n, triple in ms.items():
        E[f"read_masses()[{n}].m1"], E[f"read_masses()[{n}].m2"], E[f"read_masses()[{n}].m3"] = triple

    # -- read_selection() / selection_dark(n) (named) ----------------------------------------
    sel, (Ldown, Lup) = R.read_selection()
    E["read_selection().L_down"], E["read_selection().L_up"] = Ldown, Lup
    for n, (lam, h, L) in sel.items():
        E[f"read_selection()[{n}].lam"] = lam
        E[f"read_selection()[{n}].h_re"], E[f"read_selection()[{n}].h_im"] = h.real, h.imag

    # -- read_phases() (named, zero-arg) -----------------------------------------------------
    ph = R.read_phases()
    for n, d in ph.items():
        E[f"read_phases()[{n}]"] = float(d)
    E["phases.n3(e)"] = E["read_phases()[3]"]  # alias used by TIER_A_MAP (e-channel, n=3)

    # -- read_moduli() (named, zero-arg) -----------------------------------------------------
    mo = R.read_moduli()
    for n, q in mo.items():
        E[f"read_moduli()[{n}]"] = float(q)

    # -- read_democratic() (named, zero-arg) -------------------------------------------------
    cv, fdem = R.read_democratic()
    E["read_democratic().c_v"], E["read_democratic().f_dem"] = cv, fdem
    E["democratic.c_v"] = cv  # alias used by TIER_A_MAP

    # -- read_vertex(n_H, n_F) (named, needs args -- called at the two cases the engine's own
    #    __main__ demo uses: (1,2)=y_tau vertex, (4,0)=lambda vertex) -----------------------
    E["read_vertex(1,2)[y_tau-vertex]"] = R.read_vertex(1, 2)
    E["read_vertex(4,0)[lambda-vertex]"] = R.read_vertex(4, 0)

    # -- read_obliques() (named, zero-arg) ---------------------------------------------------
    cS, d_r, F, d_rho, S = R.read_obliques()
    E["read_obliques().c_S"], E["read_obliques().d_r"] = cS, d_r
    E["read_obliques().F"], E["read_obliques().d_rho"], E["read_obliques().S"] = F, d_rho, S
    E["obliques.d_r"], E["obliques.d_rho"] = d_r, d_rho  # aliases used by TIER_A_MAP

    # -- read_gauge_consistency(alpha_EM, sin2W) (named, native FORM; needs external inputs --
    #    used in PHASE 4 Tier-B with the LOCK's own alpha_EM/sin2_theta_W_MZ; noted here as a
    #    callable form, not invoked standalone in the inventory since it has no defaults) -------
    E["read_gauge_consistency(alpha_EM,sin2W)"] = "native FORM g_2=sqrt(4*pi*alpha_EM/sin2W); needs 2 external args -- used in PHASE 4 Tier-B, not a zero-arg inventory value"

    # -- read_run() (named, zero-arg) --------------------------------------------------------
    rho_step, arrow, G = R.read_run()
    E["read_run().rho_step"], E["read_run().arrow"] = rho_step, bool(arrow)
    E["read_run().G_shape"] = str(G.shape)

    # -- read_sector_label() ------------------------------------------------------------------
    sym, asym = R.read_sector_label()
    E["read_sector_label().sym"], E["read_sector_label().asym"] = sym, asym

    # -- the_net.py's reads (Layer 3, {A(O)} -- mostly UNBUILT; the two REGRESSION ANCHORS +
    #    the ML-2/ML-3 structural reads the pre-reg names as "the_net.py's reads") -------------
    E["net.anchor_cell_projector"] = bool(NET.anchor_cell_projector())
    E["net.anchor_tick_2pi"] = bool(NET.anchor_tick_2pi())
    sc = NET.gauge_sector_category()
    E["net.gauge_sectors.dims"] = sc["species_sector_dims"]
    E["net.gauge_sectors.double_cover_2T"] = bool(sc["double_cover_2T"])
    E["net.gauge_sectors.sectors_are_species"] = bool(sc["sectors_are_species"])
    fa = NET.dr_frame_audit()
    E["net.dr_frame_audit.winding_is_gauge"] = bool(fa["winding_is_gauge"])
    E["net.dr_frame_audit.frame_forced"] = bool(fa["frame_forced"])
    E["net.dr_frame_audit.weld_bits"] = fa["weld_bits"]
    gev = np.linalg.eigvalsh(NET.emergent_metric())
    E["net.emergent_metric.eigs"] = sorted(round(float(x), 6) for x in gev)

    # -- S1b BATCH 1 (flavor): read_ported_flavor() (named, zero-arg) -- the accreted CKM/PMNS/R_nu
    #    roster ported into the_run.py under "S1b PORTED READS (batch 1: flavor)". One engine call,
    #    flattened per-key below (see internal research notes).
    pf = R.read_ported_flavor()
    for pk, pv in pf.items():
        E[f"read_ported_flavor().{pk}"] = pv

    # -- S1b BATCH 2 (masses+Higgs): read_ported_masses_higgs() (named, zero-arg) -- the accreted
    #    fermion-mass + Higgs-sector roster ported into the_run.py under "S1b PORTED READS (batch 2:
    #    masses+Higgs)". One engine call, flattened per-key below (same frozen porting rules as batch 1,
    #    internal research notes).
    pmh = R.read_ported_masses_higgs()
    for pk, pv in pmh.items():
        E[f"read_ported_masses_higgs().{pk}"] = pv

    # -- S1b BATCH 3 (cosmology): read_ported_cosmology() (named, zero-arg) -- the accreted
    #    cosmology-sector roster ported into the_run.py under "S1b PORTED READS (batch 3:
    #    cosmology)": 10 Tier-A values (Omega_DM_over_Omega_m, Lambda_CC, w_DE, H_0, t_0,
    #    A_hemispherical, epsilon_CP, eta_B, N_eff, T_e_ann) + 2 ENGINE CORES consumed by the
    #    new Tier-B compositions (sin_arg_h_P for beta_cosmic_birefringence; z_eff_adopted for
    #    the z_eff adoption row -- the adoptions themselves are registered in PHASE 4, never
    #    silently inside a Tier-A read). One engine call, flattened per-key below (same frozen
    #    porting rules as batches 1-2, internal research notes).
    pc = R.read_ported_cosmology()
    for pk, pv in pc.items():
        E[f"read_ported_cosmology().{pk}"] = pv

    # -- S1b BATCH 4 (gauge+misc, FINAL BATCH): read_ported_gauge_running() (named,
    #    zero-arg) -- the accreted gauge/EW RG chain + neutrino masses + N_eff/
    #    observer_dim_three + framework-internal misc roster ported into the_run.py
    #    under "S1b PORTED READS (batch 4: gauge+misc)". One engine call, flattened
    #    per-key below (same frozen porting rules as batches 1-3,
    #    internal research notes).
    pg = R.read_ported_gauge_running()
    for pk, pv in pg.items():
        E[f"read_ported_gauge_running().{pk}"] = pv

    # -- R1 HARVEST (2026-07-10): read_r1_harvest() (named, zero-arg) -- the accreted
    #    H-1/H-2/H-3/H-5 composites ported into the_run.py under "R1 HARVEST READS
    #    (2026-07-10)". One engine call, flattened per-key below. See
    #    internal research notes and the per-row justification in
    #    TIER_A_MAP / _tier_b_compositions() below.
    rh = R.read_r1_harvest()
    for pk, pv in rh.items():
        E[f"read_r1_harvest().{pk}"] = pv

    # -- LIGHT BATCH (2026-07-10): read_T_nu_dec() -- the S1b orphan-turned-registered-lock's
    #    engine surface, now BUILT (closes UNMAPPED_LOCK_NOTES["T_nu_dec"]'s "pending a future
    #    engine-surface build"; that note retained below as the historical S1b record).
    E["read_T_nu_dec().T_nu_dec_MeV"] = R.read_T_nu_dec()["T_nu_dec_MeV"]

    return E


def print_engine_inventory(E):
    banner("PHASE 1 -- THE FULL ENGINE-OUTPUT INVENTORY (the_run.py + the_net.py + srs.py)")
    for k in sorted(E.keys()):
        print(f"  {k:52s} = {E[k]}")
    print(f"\n  TOTAL engine-output entries: {len(E)}")


# ==============================================================================================
# PHASE 2 -- THE MAPPING TABLE (M-1, the discipline hinge).  Engine-key -> lock-key, built from
#   the reads' OWN docstrings/outputs (see the per-entry comment citing the_run.py's docstring or
#   the literal quantity being read), PRINTED IN FULL, FROZEN before any comparison is run.
#   Every entry below was independently numerically verified (2026-07-09 dev pass, see
#   MAPPING-REVISIONS at the bottom) to match its lock at ~1e-15 relative before being frozen here.
# ==============================================================================================
# TIER-A: engine key -> (lock key, tolerance class, one-line justification)
TIER_A_MAP = {
    # -- geometry / lattice invariants (the srs graph itself; cf. G1 Kotani-Sunada realization) --
    "geometry.b1":            ("d_spatial",   "exact", "read_geometry(): b1 = |E|-|V|+1 = the spatial dimension (Cencov-Fisher row)"),
    "srs.NE":                 ("E_count",     "exact", "len(srs.EDGES) -- the walled srs primitive edge count"),
    "srs.NV":                 ("V_count",     "exact", "srs.NV -- the walled srs primitive vertex count"),
    "srs.DEG":                ("k_star",      "exact", "srs.DEG = k* (coordination number), the framework's k*"),
    "srs.DEG(as-GJ)":         ("georgi_jarlskog", "exact", "ledger's own note: 'Predicted: k* = 3 (exact)' -- same constant, second lock"),
    "run.GIRTH":               ("g_girth",     "exact", "GIRTH = read_girth(), the non-backtracking-walk girth"),
    "run.P_TOGGLE":            ("p_toggle",    "exact", "P_TOGGLE = len(darts)//len(edges), the orientation binary"),
    "srs_E_at_P(sqrt-Perron)": ("srs_E_at_P",  "float", "sqrt(adjacency_energies()[0]) = sqrt(Perron eigenvalue k*) = sqrt(3), the srs energy at P"),
    # -- S1b ORPHAN CLEANUP (2026-07-09): h_walker_eigenvalue, split real/imag. The lock file has NO
    #    prior complex-value convention (scripts/value_lock.py's own collect_values() explicitly
    #    skips isinstance(p, complex) -- "no scalar predicted value to lock"); this _re/_im split is
    #    the go-forward convention (see MAPPING-REVISIONS for full disclosure).
    "run.ihara_bass_root(sqrt_LAM_PERRON).re": ("h_walker_eigenvalue_re", "float", "R.ihara_bass_root(sqrt(LAM_PERRON)).real = sqrt(3)/2, the P-point walker root's real part -- the SAME native root read_obliques()/read_ported_flavor() already reuse verbatim per the_run.py's own comments (lines ~466, ~708-710, ~1040-1043)"),
    "run.ihara_bass_root(sqrt_LAM_PERRON).im": ("h_walker_eigenvalue_im", "float", "R.ihara_bass_root(sqrt(LAM_PERRON)).imag = sqrt(5)/2, the P-point walker root's imaginary part -- same native root as h_walker_eigenvalue_re"),
    # -- gauge boundary --
    "gauge.sin2thetaW":        ("sin2_theta_W", "exact", "read_gauge(): sin^2(theta_W) = Tr(S^2)/Tr(Q^2) at unification"),
    # -- flavor / Koide (Lambda^bullet(C^3) C3-isotype construction) --
    "flavor.Q":                ("Q_Koide",     "exact", "read_flavor(): Koide Q = (1+2*rho)/3 = 2/3"),
    "flavor.gens":             ("R3_observer_c3_generation", "exact", "read_flavor(): gens = #C3 isotypes = 3-irrep dim (the ledger's own R3_observer_c3_generation.py row)"),
    "phases.n3(e)":            ("delta_Koide", "exact", "read_phases()[3] (e-channel generation phase) = 2/9, the ledger's delta_Koide"),
    "mass.sqrt_shell":         ("epsilon_Koide", "float", "sqrt(read_mass()[0]) = sqrt(k*-1) = sqrt(2), the ledger's epsilon_Koide"),
    "run.U_RUN(alpha_1)":      ("alpha_1",     "float", "U_RUN = ((k*-1)/k*)^(GIRTH-2) = alpha_1_bare, module-level constant"),
    "run.alpha_1_full(formula)": ("alpha_1_full", "float", "the literal read_moduli() leading expression Fraction(5,3)*((k*-1)/k*)^(GIRTH-2)"),
    # -- EW obliques (gauge-vertex projections of the SAME resolvent G_NB) --
    "obliques.d_r":            ("delta_r",     "float", "read_obliques(): delta_r = c_S*u/(1-u), the Z/Perron oblique"),
    "obliques.d_rho":          ("delta_rho",   "float", "read_obliques(): delta_rho = (1/2)*(sqrt5/4)*u, the W/h_P oblique"),
    # -- v-Higgs democratic vertex (has NO ledger row -- an orphan lock, kept for the mapping's honesty) --
    "democratic.c_v":          ("c_vertex_dark", "exact", "read_democratic(): c_v=(k*+p_toggle)/(2|E|)=5/12 -- matches the lock exactly; NO ledger row cites this key (orphan lock, reported in PHASE 2)"),
    # -- S1b BATCH 1 (flavor): read_ported_flavor() -- CKM (9 elements + delta_CP + J_CKM),
    #    R_nu_splitting, PMNS (3 angles + delta_CP), 2 Majorana phases. See
    #    internal research notes + the_run.py's own per-value
    #    provenance comments (transcribed faithfully from the Tier-C prediction files).
    "read_ported_flavor().V_us":           ("V_us",           "float", "predictions/V_us.py: k*^2/(g*N_ATOMS) = 9/40, transcribed"),
    "read_ported_flavor().V_cb":           ("V_cb",           "float", "predictions/V_cb.py: alpha_1_bare/(1-alpha_1_bare) = U_RUN/(1-U_RUN), transcribed"),
    "read_ported_flavor().V_ub":           ("V_ub",           "float", "predictions/V_ub.py: multi-cycle walk-rep sum Sigma_{m=2}^{10} alpha_m/(1-alpha_m), transcribed"),
    "read_ported_flavor().V_ud":           ("V_ud",           "float", "predictions/V_ud.py (via _ckm_unitarity.py): standard-parameterization c_12*c_13, transcribed"),
    "read_ported_flavor().V_cd":           ("V_cd",           "float", "predictions/V_cd.py (via _ckm_unitarity.py): standard-parameterization |V_cd|, transcribed"),
    "read_ported_flavor().V_cs":           ("V_cs",           "float", "predictions/V_cs.py (via _ckm_unitarity.py): standard-parameterization |V_cs|, transcribed"),
    "read_ported_flavor().V_td":           ("V_td",           "float", "predictions/V_td.py (via _ckm_unitarity.py): standard-parameterization |V_td|, transcribed"),
    "read_ported_flavor().V_ts":           ("V_ts",           "float", "predictions/V_ts.py (via _ckm_unitarity.py): standard-parameterization |V_ts|, transcribed"),
    "read_ported_flavor().V_tb":           ("V_tb",           "float", "predictions/V_tb.py (via _ckm_unitarity.py): standard-parameterization c_23*c_13, transcribed"),
    "read_ported_flavor().delta_CP_CKM":   ("delta_CP_CKM",   "float", "predictions/delta_CP_CKM_geometry.py: arccos(1/k*) K4 tetrahedral dihedral (Coxeter 1973), transcribed"),
    "read_ported_flavor().J_CKM":          ("J_CKM",          "float", "predictions/J_CKM.py: Jarlskog invariant c_12*c_13^2*c_23*s_12*s_13*s_23*sin(delta_CP), transcribed"),
    "read_ported_flavor().R_nu_splitting": ("R_nu_splitting", "float", "predictions/R_nu_splitting.py: K4 Green's-function Chebyshev expansion at the Ihara phase, transcribed"),
    "read_ported_flavor().theta_12_PMNS":  ("theta_12_PMNS",  "float", "predictions/theta_12_PMNS.py: cos(theta_TBM)/cos(theta_C) SU(4)_PS perp identity, transcribed"),
    "read_ported_flavor().theta_13_PMNS":  ("theta_13_PMNS",  "float", "predictions/theta_13_PMNS.py: Class-2-stripped V_us_bare/sqrt(k*-1)*(1-alpha_1_bare), transcribed"),
    "read_ported_flavor().theta_23_PMNS":  ("theta_23_PMNS",  "float", "predictions/theta_23_PMNS.py: arctan((1+alpha_1_full)/(1-alpha_1_full)), transcribed"),
    "read_ported_flavor().delta_CP_PMNS":  ("delta_CP_PMNS",  "float", "predictions/delta_CP_PMNS.py: arccos(T_{B-L,lepton})=arccos(-1)=180deg, transcribed"),
    "read_ported_flavor().alpha_21_PMNS":  ("alpha_21_PMNS",  "float", "predictions/alpha_21_PMNS.py: g*arg(h) mod 360deg, transcribed"),
    "read_ported_flavor().alpha_31_PMNS":  ("alpha_31_PMNS",  "float", "predictions/alpha_31_PMNS.py: 2g*arg(h) mod 360deg (2=p_toggle), transcribed"),
    # -- S1b BATCH 2 (masses+Higgs): read_ported_masses_higgs() -- Higgs sector (v_higgs, m_H,
    #    lambda_3_higgs, G_F, all via the N_hub/G_F-consistency calibration chain) + y_tau + all 9
    #    charged-fermion masses (transcribed from the Tier-C engine-surface-missing prediction files;
    #    see the_run.py's own per-function provenance comments and
    #    internal research notes for the shared frozen porting rules).
    "read_ported_masses_higgs().v_higgs":        ("v_higgs",        "float", "predictions/v_higgs.py + N_hub.py: BZJ VEV delta^2*M_Pl*dark_v/(sqrt2*N_hub^(1/V)); N_hub pinned via the measured G_F, transcribed"),
    "read_ported_masses_higgs().m_H":             ("m_H",             "float", "predictions/m_H.py: sqrt(2*lambda_higgs)*v (tree-level, MDL mu^2=0 critical point), transcribed"),
    "read_ported_masses_higgs().lambda_3_higgs":  ("lambda_3_higgs",  "float", "predictions/lambda_3_higgs.py: m_H^2/(2v) [= lambda_higgs*v identity], transcribed"),
    "read_ported_masses_higgs().G_F":             ("G_F",             "float", "predictions/G_F.py: 1/(sqrt2*v^2) tree-level SM relation (round-trips the measured G_F calibration by construction), transcribed"),
    "read_ported_masses_higgs().y_tau":           ("y_tau",           "float", "predictions/y_tau.py: (alpha_1_full/k*^2)*Family-D per-leg correction (1H+2F Yukawa vertex), transcribed"),
    "read_ported_masses_higgs().m_tau":           ("m_tau",           "float", "predictions/m_tau.py: v*y_tau, the single independent charged-lepton absolute mass scale, transcribed"),
    "read_ported_masses_higgs().m_e":             ("m_e",             "float", "predictions/m_e.py: m_tau*(f_min/f_max)^2 Koide ratio == native read_masses()[3] lepton sector, transcribed"),
    "read_ported_masses_higgs().m_mu":            ("m_mu",            "float", "predictions/m_mu.py: m_tau*(f_mid/f_max)^2 Koide ratio == native read_masses()[3] lepton sector, transcribed"),
    "read_ported_masses_higgs().m_t":             ("m_t",             "float", "predictions/m_t.py: (v/sqrt2)*y_t(GUT=1)*(B') Feshbach dark at Perron channel, power 2 (L=0 saturation), transcribed"),
    "read_ported_masses_higgs().m_b":             ("m_b",             "float", "predictions/m_b.py: v*((k*-1)/k*)^g*(B') Feshbach dark at Perron channel, power 1 (L=g Type-IV walker), transcribed"),
    "read_ported_masses_higgs().m_u":             ("m_u",             "float", "predictions/m_u.py: m_t*(f_min/f_max)^2 up-sector Koide ratio == native read_masses()[2] sector, transcribed"),
    "read_ported_masses_higgs().m_c":             ("m_c",             "float", "predictions/m_c.py: m_t*(f_mid/f_max)^2 up-sector Koide ratio == native read_masses()[2] sector, transcribed"),
    "read_ported_masses_higgs().m_d":             ("m_d",             "float", "predictions/m_d.py: m_b*(f_min/f_max)^2 down-sector Koide ratio == native read_masses()[1] sector, transcribed"),
    "read_ported_masses_higgs().m_s":             ("m_s",             "float", "predictions/m_s.py: m_b*(f_mid/f_max)^2 down-sector Koide ratio == native read_masses()[1] sector, transcribed"),
    # -- S1b BATCH 3 (cosmology): read_ported_cosmology() -- the Poisson dark ratio, the
    #    coasting suite (N_hub-reads via batch 2's engine-native read_higgs_chain), the
    #    eps_toggle pair, the Sakharov eta_B, N_eff, T_e_ann.  The adoption-riding rows
    #    (Omega_DM, Omega_b, z_eff, beta) are NOT here -- they are PHASE-4 Tier-B
    #    compositions with their adoptions named (batch-3 dispatch, no-silent-hardcode).
    "read_ported_cosmology().Omega_DM_over_Omega_m": ("Omega_DM_over_Omega_m", "float", "predictions/Omega_DM_over_Omega_m.py: 1 - P(k<=k*|Poisson(2k*)) = 1-61e^-6 Cl(2k*) Fock tail, transcribed"),
    "read_ported_cosmology().Lambda_CC":          ("Lambda_CC",       "float", "predictions/Lambda_CC.py: Lambda_substrate = 1/N_hub^2 (coasting Friedmann, Planck units; N_hub engine-native via batch 2), transcribed"),
    "read_ported_cosmology().w_DE":               ("w_DE",            "exact", "predictions/w_DE.py: w_DE = -1 exactly (static Lambda rigidity; rate-gap cancels in ratio), transcribed"),
    "read_ported_cosmology().H_0":                ("H_0",             "float", "predictions/H_0.py: H_0_substrate = 1/(N_hub*t_P) in km/s/Mpc (cascade theorem, coefficient 1; N_hub engine-native), transcribed"),
    "read_ported_cosmology().t_0":                ("t_0",             "float", "predictions/t_0.py: t_0_substrate = N_hub*t_P in Gyr (cascade theorem; coasting H_0*t_0=1), transcribed"),
    "read_ported_cosmology().A_hemispherical":    ("A_hemispherical", "exact", "predictions/A_hemispherical.py: A = eps_toggle/k* = (1/5)(1/3) = 1/15 (eps_toggle == read_clock's own eps), transcribed"),
    "read_ported_cosmology().epsilon_CP":         ("epsilon_CP",      "exact", "predictions/epsilon_CP.py: eps_CP = (P_fresh-P_persist)/(P_fresh+P_persist) = 1/5 == read_clock().eps (identical Beta(2,1) toggle formula), transcribed"),
    "read_ported_cosmology().eta_B":              ("eta_B",           "float", "predictions/eta_B.py: eta_B = eps_CP*Re(h_P)*alpha_1^M = (sqrt3/10)*(2/3)^48 (Sakharov-Hashimoto chain; h_P native ihara_bass_root), transcribed"),
    # N_eff DEMOTED to Tier C by architect adjudication (batch-3 adversarial check, 2026-07-09):
    # N_eff.py's real DAG ingredient is observer_dim_three_pred (MDL+Gleason), NOT read_flavor().gens
    # (a distinct C3-isotype count that happens to equal 3). Mapping it via gens re-performs, one level
    # removed, the exact forced-pairing this manifest rejected for observer_dim_three itself. N_eff
    # stays Tier C with blocker engine-surface-missing:observer_dim_three until that chain is ported.
    "read_ported_cosmology().T_e_ann":            ("T_e_ann",         "float", "predictions/T_e_ann.py: T_e_ann = m_e/k* in MeV (Phase IIb Boltzmann threshold; m_e engine-native via batch 2), transcribed"),
    # -- S1b BATCH 4 (gauge+misc, FINAL): read_ported_gauge_running() -- the gauge/EW
    #    RG chain (g_1,g_3,alpha_GUT,sin2_theta_W_MZ,alpha_s,alpha_EM,M_unif,M_Z,m_W,
    #    Gamma_Z_over_M_Z,Gamma_W_over_Gamma_Z,theta_QCD) + 2 neutrino masses +
    #    N_eff/observer_dim_three + 16 framework-internal misc rows. See
    #    internal research notes + the_run.py's
    #    own per-value provenance comments (transcribed faithfully from the Tier-C
    #    prediction files).
    "read_ported_gauge_running().g_1":                  ("g_1",                  "float", "predictions/g_1.py: g_1(M_Z)=sqrt(4*pi*alpha_1(M_Z)), alpha_1 RG-run from uniform alpha_GUT_observed at M_unif, transcribed"),
    "read_ported_gauge_running().g_3":                  ("g_3",                  "float", "predictions/g_3.py: g_3(M_Z)=sqrt(4*pi*alpha_3(M_Z)), alpha_3 RG-run from COLOR-sector alpha_GUT_observed_sector(c=1/4) at M_unif, transcribed"),
    "read_ported_gauge_running().alpha_GUT":            ("alpha_GUT",            "float", "predictions/alpha_GUT.py predict_alpha_GUT_observed: dark-corrected uniform-sector alpha_GUT = alpha_GUT_bare*(1-(1/k*)*waterline), transcribed"),
    "read_ported_gauge_running().sin2_theta_W_MZ":      ("sin2_theta_W_MZ",      "float", "predictions/sin2_theta_W_MZ.py: alpha_Y(M_Z)/(alpha_2(M_Z)+alpha_Y(M_Z)), uniform-sector RG chain, transcribed"),
    "read_ported_gauge_running().alpha_s":              ("alpha_s",              "float", "predictions/alpha_s.py: alpha_3(M_Z), COLOR-sector alpha_GUT_observed_sector(c=1/4) RG chain, transcribed"),
    "read_ported_gauge_running().alpha_EM":             ("alpha_EM",             "float", "predictions/alpha_EM.py: alpha_2(M_Z)*sin2_theta_W(M_Z), uniform-sector RG chain, transcribed"),
    "read_ported_gauge_running().M_unif":               ("M_unif",               "float", "predictions/M_unif.py: alpha_GUT_bare*alpha_1_bare*M_Pl (BARE alpha_GUT, pre-dark-correction), transcribed"),
    "read_ported_gauge_running().M_Z":                  ("M_Z",                  "float", "predictions/M_Z.py: self-consistent SM-tree M_Z=sqrt(pi)*v*sqrt(alpha_2+(3/5)alpha_1) x (1-delta_r) pole correction, transcribed"),
    "read_ported_gauge_running().m_W":                  ("m_W",                  "float", "predictions/m_W.py: M_Z*sqrt(1-sin2_theta_W)*sqrt(1+delta_rho), transcribed"),
    "read_ported_gauge_running().Gamma_Z_over_M_Z":     ("Gamma_Z_over_M_Z",     "float", "predictions/Gamma_Z_over_M_Z.py: golden-rule tree x QCD x (1+delta_Z) EW loop layer, transcribed"),
    "read_ported_gauge_running().Gamma_W_over_Gamma_Z": ("Gamma_W_over_Gamma_Z", "float", "predictions/Gamma_W_over_Gamma_Z.py: golden-rule tree ratio x QCD ratio x (1+delta_W)/(1+delta_Z), transcribed"),
    "read_ported_gauge_running().theta_QCD":            ("theta_QCD",            "exact", "predictions/theta_QCD.py: Z3 gauge-connection holonomy flatness on srs -> 0 exactly, transcribed"),
    "read_ported_gauge_running().m_nu2":                ("m_nu2",                "float", "predictions/m_nu2.py: m_nu3/sqrt(R), R=228/7 (batch-1 native read_R_nu_splitting), transcribed"),
    "read_ported_gauge_running().m_nu3":                ("m_nu3",                "float", "predictions/m_nu3.py: (k**N_atoms)*M_Pl*N_hub^(-1/2) (N_hub engine-native via batch 2), transcribed"),
    "read_ported_gauge_running().N_eff":                ("N_eff",                "exact", "predictions/N_eff.py: N_eff=predict_N_eff(observer_dim_three_pred)=observer_dim_three_pred=3, TRUE ingredient (supersedes batch-3's read_flavor().gens route per the adjudication), transcribed"),
    "read_ported_gauge_running().observer_dim_three":   ("observer_dim_three",   "exact", "predictions/observer_dim_three.py: MDL+Gleason 1957 forces observer Hilbert dim n=3 (structural literal, distinct engine key from k_star/gens), transcribed"),
    "read_ported_gauge_running().srs_cubic_moment":     ("srs_cubic_moment",     "exact", "predictions/srs_cubic_moment.py: <(e.zhat)^2>=1/(k*·2^0)=1/k*, transcribed"),
    "read_ported_gauge_running().srs_bloch_lv_dim6":    ("srs_bloch_lv_dim6",    "float", "predictions/srs_bloch_lv_dim6.py: eta^H_NB=D4_aniso^H/D_H^2=1/6 (D_H=1/16,D4_aniso^H=1/1536 hardcoded from the Feshbach-Loewdin symbolic proof), transcribed"),
    "read_ported_gauge_running().e_bit":                ("e_bit",                "exact", "predictions/e_bit.py: e_bit=M_substrate=1 exactly (unit identification), transcribed"),
    "read_ported_gauge_running().M_Pl_natural":         ("M_Pl_natural",         "float", "predictions/M_Pl_natural.py: M_Pl/M_substrate=8/sqrt(pi) (Drude UV asymptote x Planck convention), transcribed"),
    "read_ported_gauge_running().feshbach_exponent_principle": ("feshbach_exponent_principle", "float", "predictions/feshbach_exponent_principle.py: n_fixed=2 case = ((k*-1)/k*)^(g-2) = U_RUN, transcribed"),
    "read_ported_gauge_running().koide_quark_ratio":    ("koide_quark_ratio",    "float", "predictions/koide_quark_ratio.py: (k**g-p_toggle)/g=14/5, transcribed"),
    "read_ported_gauge_running().lambda_toggle_rate":   ("lambda_toggle_rate",   "float", "predictions/lambda_toggle_rate.py: 2*p_create*p_destroy/(p_create+p_destroy)=2/5 (p_create=1/p_toggle,p_destroy=1/k*, == read_clock's Pf/Pp), transcribed"),
    "read_ported_gauge_running().xi_t_temporal_correlation": ("xi_t_temporal_correlation", "float", "predictions/xi_t_temporal_correlation.py: 1/log(1/r), r=1-p_create-p_destroy=1/6, transcribed"),
    "read_ported_gauge_running().S_fresh":               ("S_fresh",              "float", "predictions/S_fresh.py: -log2(1/p_toggle)=log2(p_toggle)=1 bit, transcribed"),
    "read_ported_gauge_running().S_disconfirm":          ("S_disconfirm",         "float", "predictions/S_disconfirm.py: -log2(1/k*)=log2(k*)=log2(3) bits, transcribed"),
    "read_ported_gauge_running().eta_5_lorentz_dim5":    ("eta_5_lorentz_dim5",   "exact", "predictions/eta_5_lorentz_dim5.py: undirected-graph symmetry B(-k)=B(k)* forces the O(k^3) coefficient to vanish -> 0 exactly, transcribed"),
    "read_ported_gauge_running().eta_lattice_lorentz_dim6": ("eta_lattice_lorentz_dim6", "float", "predictions/eta_lattice_lorentz_dim6.py: D4_aniso/D_NB^2=1/12 (D_NB=1/8,D4_aniso=1/768 hardcoded from the symbolic Ihara cross-walker proof), transcribed"),
    "read_ported_gauge_running().scale_energy_hashimoto": ("scale_energy_hashimoto", "float", "predictions/scale_energy_hashimoto.py: (m_e_obs^2*M_Pl^2/|eta_lattice|)^(1/4) in PeV (m_e_obs = the file's own declared PDG anchor), transcribed"),
    "read_ported_gauge_running().universe_transparency": ("universe_transparency", "float", "predictions/universe_transparency.py: the SAME scale_energy_hashimoto threshold (transparency onset), transcribed"),
    "read_ported_gauge_running().G_N":                   ("G_N",                  "float", "predictions/G_N.py: G_N*M_Pl^2 = (pi/(16*N_atoms))*(8/sqrt(pi))^2 = 1 exactly (Drude UV asymptote x Planck convention; N_atoms=srs.NV), transcribed. NOTE: ledger File column is blank ('—') -- predictions/G_N.py exists but is un-cited there; located by direct search, disclosed in MAPPING-REVISIONS."),
    "read_ported_gauge_running().tan_beta":              ("tan_beta",             "float", "predictions/tan_beta.py: tan(beta) s.t. bottom-up MSSM 1-loop Yukawa RGE from M_Z satisfies the Georgi-Jarlskog condition y_b(M_unif)/y_tau(M_unif)=k* (scipy solve_ivp+brentq), transcribed"),
    # -- R1 HARVEST (2026-07-10): read_r1_harvest() -- H-1 coasting chain (q_0/w_eff exact;
    #    H(z)/D_C/D_A/D_L/D_V at the declared z=1.0 anchor point registered here; z=0.5/2.0
    #    are additional curve locks, orphan-style, no ledger row of their own -- same pattern
    #    as the pre-existing 'c_vertex_dark' orphan lock) + H-2 Omega_k/Sigma_m_nu (pure
    #    engine, no adoption) + H-5 partial structural wiring (fermion_content/
    #    h_walker_abs2/cone_velocity_v0/T_of_N_now_eV).  internal research notes
    "read_r1_harvest().q_0":            ("harvest_q_0",            "exact", "H-1: a proportional to t coasting kinematics -> ae=0 -> q_0=0 EXACT, no input"),
    "read_r1_harvest().w_eff":          ("harvest_w_eff",          "exact", "H-1: ae/a=-(4piG/3)(rho+3p)=0 for a proportional to t -> w_eff=-1/3 EXACT, no input"),
    "read_r1_harvest().H_z1p0":         ("harvest_H_z1p0",         "float", "H-1: H(z)=H_0*(1+z) at the declared z=1.0 anchor (coasting Hubble history), engine H_0 only"),
    "read_r1_harvest().D_C_z1p0":       ("harvest_D_C_z1p0",       "float", "H-1: D_C(z)=(c/H_0)*ln(1+z) at z=1.0 [Mpc]; c SI-exact, flat (Omega_k=0) so D_M==D_C"),
    "read_r1_harvest().D_A_z1p0":       ("harvest_D_A_z1p0",       "float", "H-1: D_A(z)=D_C(z)/(1+z) at z=1.0 [Mpc]"),
    "read_r1_harvest().D_L_z1p0":       ("harvest_D_L_z1p0",       "float", "H-1: D_L(z)=D_C(z)*(1+z) at z=1.0 [Mpc]"),
    "read_r1_harvest().D_V_z1p0":       ("harvest_D_V_z1p0",       "float", "H-1: D_V(z)=[D_C(z)^2*c*z/H(z)]^(1/3) at z=1.0 [Mpc], isotropic BAO dilation scale"),
    "read_r1_harvest().H_z0p5":         ("harvest_H_z0p5",         "float", "H-1 curve (orphan lock, no dedicated ledger row): H(z=0.5)"),
    "read_r1_harvest().D_C_z0p5":       ("harvest_D_C_z0p5",       "float", "H-1 curve (orphan lock): D_C(z=0.5)"),
    "read_r1_harvest().D_A_z0p5":       ("harvest_D_A_z0p5",       "float", "H-1 curve (orphan lock): D_A(z=0.5)"),
    "read_r1_harvest().D_L_z0p5":       ("harvest_D_L_z0p5",       "float", "H-1 curve (orphan lock): D_L(z=0.5)"),
    "read_r1_harvest().D_V_z0p5":       ("harvest_D_V_z0p5",       "float", "H-1 curve (orphan lock): D_V(z=0.5)"),
    "read_r1_harvest().H_z2p0":         ("harvest_H_z2p0",         "float", "H-1 curve (orphan lock): H(z=2.0)"),
    "read_r1_harvest().D_C_z2p0":       ("harvest_D_C_z2p0",       "float", "H-1 curve (orphan lock): D_C(z=2.0)"),
    "read_r1_harvest().D_A_z2p0":       ("harvest_D_A_z2p0",       "float", "H-1 curve (orphan lock): D_A(z=2.0)"),
    "read_r1_harvest().D_L_z2p0":       ("harvest_D_L_z2p0",       "float", "H-1 curve (orphan lock): D_L(z=2.0)"),
    "read_r1_harvest().D_V_z2p0":       ("harvest_D_V_z2p0",       "float", "H-1 curve (orphan lock): D_V(z=2.0)"),
    "read_r1_harvest().Omega_k":        ("harvest_Omega_k",        "exact", "H-2: framework substrate is spatially flat (d_spatial=3 Euclidean) -> Omega_k=0 EXACT, structural (same status as theta_QCD=0)"),
    "read_r1_harvest().Sigma_m_nu_eV":  ("harvest_Sigma_m_nu_eV",  "float", "H-2: m_nu1+m_nu2+m_nu3 (m_nu1=0 W45 structural; m_nu2/m_nu3 engine-native), pure engine composite, no adoption"),
    "read_r1_harvest().fermion_content": ("harvest_fermion_content", "exact", "H-5: sum(read_species().values())*read_flavor()[3]*p_toggle = 8*3*2 = 48, the ledger's own '48 states' count"),
    "read_r1_harvest().h_walker_abs2":  ("harvest_h_walker_abs2",  "exact", "H-5: K-1=2, the Ramanujan saturation |h_P|^2=k*-1 already asserted throughout (e.g. predictions/delta_rho.py)"),
    "read_r1_harvest().cone_velocity_v0": ("harvest_cone_velocity_v0", "float", "H-5: the SAME construction as derivation_topdown/state/the_net.py's cone_velocity([1,0,0]) (ML-1'' emergent-metric object), reproduced via the_run.py's own adjacency(k); cross-checked to match the_net.py numerically"),
    "read_r1_harvest().T_of_N_now_eV":  ("harvest_T_of_N_now_eV",  "float", "T(N) propagation function via the S1d epoch API, evaluated AT N=N_hub (T(N_now)=T_today by construction; the calibration fence forbids extending to nonzero z without the un-built era-crossing map)"),
    # -- LIGHT BATCH (2026-07-10): T_nu_dec engine surface built (was the S1b orphan-turned-
    #    registered-lock; see UNMAPPED_LOCK_NOTES["T_nu_dec"], retained as the historical record).
    "read_T_nu_dec().T_nu_dec_MeV":     ("T_nu_dec",               "float", "LIGHT BATCH 2026-07-10: Phase IIb rate balance Gamma_weak=G_F^2*T^5 == H=T^2/M_Pl (alpha=1/2 instantaneous; predictions/T_nu_dec.py v2.0.0 port) -> T=[1/(M_Pl*G_F^2)]^(1/3) in MeV; CALIBRATION-CURVE family via the G_F tether (N_DEPENDENCE['T_nu_dec'])"),
}

# TIER-B: composed rows.  Each entry: lock_key -> dict(formula-description, adoption(s), compute-fn)
# The compute functions consume ONLY (a) already-tabulated ENGINE Tier-A quantities and (b) named
# LOCK-sourced external inputs explicitly flagged as adopted/non-engine (never predictions/ code).
def _tier_b_compositions(E, locks):
    out = {}

    # lambda_higgs = 2*alpha_1_full*(1 - 4*alpha_1_bare^2)   [Family-D quartic correction]
    # cite: target_parameters.md lambda row "= 2.alpha_1_full"; m_H row "Family D on the |phi|^4
    # vertex (delta_lambda/lambda = -4*alpha_1_bare^2)".  Pure engine composition -- NO adoption.
    a1 = E["run.U_RUN(alpha_1)"]
    a1f = E["run.alpha_1_full(formula)"]
    out["lambda_higgs"] = dict(
        value=2 * a1f * (1 - 4 * a1 ** 2), tol="float",
        formula="2*alpha_1_full*(1-4*alpha_1_bare^2)  [Family-D quartic correction]",
        cite="target_parameters.md lambda row + m_H row Family-D note",
        adoptions="none (pure engine composition of Tier-A alpha_1, alpha_1_full)")

    # g_2 = sqrt(4*pi*alpha_EM/sin2W_MZ)  -- the_run.py's OWN native identity (read_gauge_consistency),
    # evaluated at the framework's own downstream RG-run alpha_EM(M_Z)/sin2thetaW(M_Z) LOCK values
    # (those two are themselves NOT engine-reproducible here -- the RG running to M_Z is a declared
    # Layer-2 QFT one-loop import, per read_gauge_running's own docstring; not an ADOPTED-* register
    # entry, but an explicitly-flagged non-native import).
    g2 = R.read_gauge_consistency(locks["alpha_EM"], locks["sin2_theta_W_MZ"])
    out["g_2"] = dict(
        value=g2, tol="float",
        formula="read_gauge_consistency(alpha_EM, sin2theta_W_MZ) = sqrt(4*pi*alpha_EM/sin2W)  [the_run.py native FORM]",
        cite="the_run.py docstring: 'g_2 is NOT independent'",
        adoptions="alpha_EM(M_Z), sin2theta_W_MZ (Layer-2 QFT one-loop RG-running import -- declared non-native in read_gauge_running's own docstring; not a registered ADOPTED-* entry)")

    # H_0(observer) = clock * H_0(substrate)   [clock=16/15 is Tier-A; H_0(substrate) is LOCK/adopted]
    clock = E["read_clock().clock"]
    H0obs = clock * locks["H_0"]
    out["H_0_observer"] = dict(
        value=H0obs, tol="float",
        formula="read_clock().clock * H_0(substrate)  [declared '(16/15)*H_0_substrate']",
        cite="target_parameters.md H_0(observer) row",
        adoptions="H_0(substrate) needs N_hub/t_P (the framework's one adopted dimensional parameter) -- not engine-exposed")

    # Omega_m_LCDM(z_eff) = (u+1)/(u^2+u+1), u=1+z_eff   [declared K-rational bias function]
    zeff = locks["z_eff"]
    u = 1 + zeff
    Om = (u + 1) / (u ** 2 + u + 1)
    out["Omega_m_LCDM"] = dict(
        value=Om, tol="float",
        formula="(u+1)/(u^2+u+1), u=1+z_eff  [declared K-rational bias-function form]",
        cite="target_parameters.md Omega_m_LCDM row",
        adoptions="z_eff (ledger's own tag: 'ADOPTED cosmology parameter, N_hub-class')")

    # Omega_Lambda_LCDM = 1 - Omega_m_LCDM(z_eff)   [declared Type-4 sibling]
    OL = 1 - Om
    out["Omega_Lambda_LCDM"] = dict(
        value=OL, tol="float",
        formula="1 - Omega_m_LCDM(z_eff)  [declared Type-4 sibling]",
        cite="target_parameters.md Omega_Lambda_LCDM row",
        adoptions="z_eff (same as Omega_m_LCDM)")

    # Lambda_CC_LCDM = 3 * Omega_Lambda_LCDM(z_eff) * Lambda_CC(substrate)
    LamLCDM = 3 * OL * locks["Lambda_CC"]
    out["Lambda_CC_LCDM"] = dict(
        value=LamLCDM, tol="float",
        formula="3 * Omega_Lambda_LCDM(z_eff) * Lambda_CC(substrate)",
        cite="target_parameters.md Lambda_LCDM row",
        adoptions="z_eff + Lambda_CC(substrate) (needs N_hub, Lambda=1/N^2)")

    # ---- S1b BATCH 3 (cosmology) Tier-B additions -- the adoption-riding rows.  Each
    #      composes an ENGINE core (batch-3 Tier-A quantity / ported engine core) over its
    #      NAMED adoption, exactly the existing pattern above (adoptions listed per row,
    #      never silently hardcoded inside a Tier-A read; batch-3 dispatch clause 1). ----

    # Omega_DM = Omega_m_LCDM(z_eff) * (Omega_DM/Omega_m)   [predictions/Omega_DM.py Type-4]
    #   engine core = the batch-3 Tier-A Poisson dark ratio; bias function as the existing
    #   Omega_m_LCDM row above; adoption = z_eff (ledger's own ADOPTED N_hub-class tag).
    ratio_dm = E["read_ported_cosmology().Omega_DM_over_Omega_m"]
    out["Omega_DM"] = dict(
        value=Om * ratio_dm, tol="float",
        formula="Omega_m_LCDM(z_eff) * (Omega_DM/Omega_m)  [bias (u+1)/(u^2+u+1) x engine Poisson ratio]",
        cite="predictions/Omega_DM.py (Type-4 composition; target_parameters.md Omega_DM row)",
        adoptions="z_eff (ledger's own tag: 'ADOPTED cosmology parameter, N_hub-class')")

    # Omega_b = Omega_m_LCDM(z_eff) * (1 - Omega_DM/Omega_m)   [predictions/Omega_b.py Type-4]
    out["Omega_b"] = dict(
        value=Om * (1.0 - ratio_dm), tol="float",
        formula="Omega_m_LCDM(z_eff) * (1 - Omega_DM/Omega_m)  [bias x engine Poisson visible head]",
        cite="predictions/Omega_b.py (Type-4 composition; target_parameters.md Omega_b row)",
        adoptions="z_eff (same as Omega_DM/Omega_m_LCDM)")

    # z_eff itself -- the ledger's own ADOPTED cosmology parameter (N_hub-class).  The engine
    # core is the transcribed Fisher first-moment arithmetic (the_run.py read_z_eff_adopted,
    # coefficients sourced from engine primitives p_toggle/srs.NV exactly as the source file
    # sources them); the ADOPTION is the two survey-DESIGN tables it integrates over.  Honest
    # form = Tier B (engine arithmetic over a registered external adoption), NOT Tier A.
    out["z_eff"] = dict(
        value=E["read_ported_cosmology().z_eff_adopted"], tol="float",
        formula="Fisher-information-weighted first-moment mean redshift of the SN+BAO survey combination  [read_z_eff_adopted(), transcribed from predictions/z_eff.py]",
        cite="predictions/z_eff.py ('ADOPTED cosmology parameter, N_hub-pattern'); target_parameters.md z_eff row",
        adoptions="the SN+BAO survey-DESIGN tables (BOSS DR12 Alam+2017 + eBOSS DR16 Alam+2021 BAO anchors; Pantheon+-like SN z-distribution/error model) -- external survey design, the adoption's own declared content (NOT fitted to distances, NOT substrate)")

    # beta_cosmic_birefringence = degrees(sin(arg h_P) * alpha_EM(M_Z))   [predictions/
    #   beta_cosmic_birefringence.py: beta = c*sin(arg h)*alpha_EM, c=1].  Engine core =
    #   sin(arg h_P) = sqrt(5/8) (native P-point walker root); alpha_EM(M_Z) is NOW
    #   ENGINE-NATIVE (S1b batch 4: read_ported_gauge_running().alpha_EM, the full
    #   M_unif/alpha_GUT-DC/M_Z RG chain ported) -- UPGRADED 2026-07-09 from the
    #   batch-3 lock-adopted form to consume the native value directly (disclosed in
    #   MAPPING-REVISIONS; batch-4 dispatch clause 1). No adoptions remain on this row.
    beta = math.degrees(E["read_ported_cosmology().sin_arg_h_P"] * E["read_ported_gauge_running().alpha_EM"])
    out["beta_cosmic_birefringence"] = dict(
        value=beta, tol="float",
        formula="degrees(sin(arg h_P) * alpha_EM(M_Z))  [c=1; sin(arg h)=sqrt(5/8) AND alpha_EM(M_Z) both native engine cores]",
        cite="predictions/beta_cosmic_birefringence.py (THEOREM-GRADE-STRUCTURAL; framework alpha_EM, zero observed input)",
        adoptions="none (pure engine composition, post-batch-4 upgrade: alpha_EM(M_Z) is now read_ported_gauge_running().alpha_EM, its RG chain fully engine-native)")

    # ---- R1 HARVEST (2026-07-10) H-2: Omega_b_h2 = Omega_b(z_eff)*h^2, Omega_c_h2 =
    #      Omega_DM(z_eff)*h^2.  SAME (u+1)/(u^2+u+1) bias-function composition + z_eff
    #      adoption as the pre-existing Omega_b/Omega_DM rows above (h=H_0/100); the
    #      z_eff CONDITIONALITY IS INHERITED AND NAMED HERE, never dropped. See
    #      internal research notes H-2. ----
    zeff_h = locks["z_eff"]
    u_h = 1 + zeff_h
    Om_h = (u_h + 1) / (u_h ** 2 + u_h + 1)
    ratio_dm_h = E["read_ported_cosmology().Omega_DM_over_Omega_m"]
    h2 = (E["read_ported_cosmology().H_0"] / 100.0) ** 2
    out["harvest_Omega_b_h2"] = dict(
        value=Om_h * (1.0 - ratio_dm_h) * h2, tol="float",
        formula="Omega_b(z_eff) * (H_0/100)^2  [(u+1)/(u^2+u+1) x (1-Poisson ratio) x h^2, u=1+z_eff]",
        cite="internal research notes H-2; target_parameters.md Omega_b h^2 row",
        adoptions="z_eff (SAME adoption as the existing Omega_b/Omega_DM rows -- inherited, not dropped)")
    out["harvest_Omega_c_h2"] = dict(
        value=Om_h * ratio_dm_h * h2, tol="float",
        formula="Omega_DM(z_eff) * (H_0/100)^2  [(u+1)/(u^2+u+1) x Poisson ratio x h^2, u=1+z_eff]",
        cite="internal research notes H-2; target_parameters.md Omega_c h^2 row",
        adoptions="z_eff (SAME adoption as the existing Omega_b/Omega_DM rows -- inherited, not dropped)")

    return out


# ==============================================================================================
# MAPPING-REVISIONS (full disclosure, per the hard rules).  Logged during the ONE development
# pass that built TIER_A_MAP/_tier_b_compositions above; the pipeline was re-run from PHASE 1
# after each fix, so the frozen tables above already reflect these corrections -- nothing here
# was patched AFTER seeing a Phase-3 mismatch.
# ==============================================================================================
MAPPING_REVISIONS = [
    "considered mapping engine 'eta_lattice_lorentz_dim6' lock (=1/12) to read_obliques().c_S "
    "(=1/12, exact numeric coincidence) -- REJECTED before freezing: c_S is the EW gauge-singlet "
    "Perron projection (read_obliques), while eta_lattice is a SEPARATE dim-6 Lorentz-violation "
    "coefficient derived in proofs/lorentz/hashimoto_dispersion_symbolic.py, outside this "
    "manifest's engine scope (the_run/the_net/srs). Same numeric value, different generator -- "
    "mapping it would be a FORCED PAIRING per the pre-reg's own poison clause. Left UNMAPPED.",
    "considered mapping 'srs_cubic_moment' lock (=1/3) to 1/srs.DEG (=1/3) -- REJECTED: the ledger "
    "row cites predictions/srs_cubic_moment.py ('P2 Theorem 1'), a separate photon-Bloch-primitive "
    "derivation not exposed by the_run.py/the_net.py/srs.py. Left UNMAPPED (engine-surface-missing).",
    "considered mapping 'observer_dim_three' lock (=3.0) to flavor.gens (=3, already mapped to "
    "R3_observer_c3_generation) or to srs.DEG (=3, already mapped to k_star) -- REJECTED: forcing "
    "a THIRD lock onto an already-used engine value, with no distinct docstring justification, "
    "would be exactly the 'no forced pairings' violation the pre-reg warns against. Left UNMAPPED.",
    "S1b BATCH 1 (flavor) additions, 2026-07-09 "
    "(internal research notes): 18 new engine-key -> lock-key "
    "pairs added to TIER_A_MAP, all under the single new engine call read_ported_flavor() "
    "(the_run.py, '# ==== S1b PORTED READS (batch 1: flavor) ====' section). Roster: V_us, V_cb, "
    "V_ub, V_ud, V_cd, V_cs, V_td, V_ts, V_tb, delta_CP_CKM, "
    "J_CKM, R_nu_splitting, theta_12_PMNS, theta_13_PMNS, theta_23_PMNS, delta_CP_PMNS, "
    "alpha_21_PMNS, alpha_31_PMNS -- every row was already present in SYMBOL_TABLE (Tier C, "
    "blocker engine-surface-missing) before this batch; only the engine-side generator was built. "
    "Each ported value is a faithful transcription of its Tier-C prediction file's OWN closed form, "
    "reusing already-native primitives (K=k*, GIRTH=g, P_TOGGLE, srs.NV, U_RUN=alpha_1_bare, the "
    "alpha_1_full literal formula, the h=(sqrt3+i*sqrt5)/2 walker root via ihara_bass_root) wherever "
    "the source file itself used them; every value was independently numerically verified against "
    "its lock at <1e-9 relative (R_nu_splitting/theta_23_PMNS at ~1e-13, float-arithmetic-order "
    "noise) before being frozen here. theta_QCD (predictions/theta_QCD.py, also Tier-C "
    "engine-surface-missing and trivially portable as k*==3) was CONSIDERED and EXCLUDED from this "
    "batch: it lives in its own ledger '### QCD (1)' section (strong-CP vacuum angle), not the CKM/"
    "PMNS/flavor-invariant scope the S1b batch-1 pre-reg names; left in Tier C for a gauge/misc "
    "batch (out of scope per the pre-reg's own BATCH-1 SCOPE clause).",
    "S1b BATCH 2 (masses+Higgs) additions, 2026-07-09 (same frozen porting rules as batch 1, "
    "internal research notes): 14 new engine-key -> lock-key pairs "
    "added to TIER_A_MAP, all under the single new engine call read_ported_masses_higgs() (the_run.py, "
    "'# ==== S1b PORTED READS (batch 2: masses+Higgs) ====' section). Roster (frozen from the "
    "regenerated docs/parameters/reads_manifest.md Tier-C engine-surface-missing rows, masses+Higgs "
    "scope): v_higgs, m_H, lambda_3_higgs, G_F, y_tau, m_e, m_mu, m_tau, m_u, m_d, m_s, m_c, m_b, m_t "
    "-- every row was already present in SYMBOL_TABLE (Tier C, blocker engine-surface-missing) before "
    "this batch. lambda_higgs was in the roster's candidate list but EXCLUDED (already the manifest's "
    "own pre-existing Tier-B composition -- not re-added as Tier-A, no double-counting). N_hub was "
    "considered and EXCLUDED: it has no ledger row of its own (present only as an unmapped lock and as "
    "context inside other rows' Notes columns) -- mapping it would be a forced pairing with no ledger "
    "row to certify; it is computed internally (needed for v_higgs) and returned by "
    "read_ported_masses_higgs() for transparency but left unmapped. The M_persistence 12x12 operator "
    "(predictions/M_persistence.py) was investigated as a possible blocker for the 6 quark masses "
    "(m_u/m_d/m_s/m_c/m_b/m_t) per the dispatch's own caution -- found NOT load-bearing: grep across "
    "predictions/*.py shows no m_u.py/m_d.py/m_s.py/m_c.py/m_b.py/m_t.py imports M_persistence; the "
    "live DAG is m_t/m_b (two absolute anchors: Type-II saturation + Type-IV Perron walker, each with "
    "the (B') Feshbach channel-read dark of heavy_quark_anchor_dark.py) + the ALREADY-NATIVE "
    "read_masses() Koide ratios per Hamming-weight sector for the other 4 (m_u/m_c from m_t, m_d/m_s "
    "from m_b) -- no M_persistence deferral needed; no transcription-blocked rows in this batch. Each "
    "ported value is a faithful transcription of its Tier-C prediction file's OWN closed form, reusing "
    "already-native primitives (K=k*, GIRTH=g, U_RUN=alpha_1_bare, the alpha_1_full literal formula, "
    "srs.NV=V_count, read_phases()[3]=delta_Koide, read_democratic()=(c_v,dark_v), and critically "
    "read_masses() for every within-sector Koide mass RATIO) wherever the source file itself used the "
    "equivalent primitive; the ONE new external input is the measured Fermi constant G_F (PDG "
    "2024/MuLan 2011, 1.1663787e-5 GeV^-2) and the CODATA 2018 Planck mass (1.22089e19 GeV) -- both "
    "transcribed with their exact provenance comments per predictions/N_hub.py and "
    "predictions/M_Pl_natural.py, the framework's documented single calibration + single SI-anchor (not "
    "a re-fit). Every value was independently numerically verified against its lock at <1e-9 relative "
    "(v_higgs/G_F/m_H/lambda_3_higgs/y_tau/m_tau/m_t/m_b at exact float equality; m_e/m_mu/m_u/m_c/"
    "m_d/m_s at ~1e-14, float-arithmetic-order noise) before being frozen here.",
    "S1b BATCH 3 (cosmology) additions, 2026-07-09 (same frozen porting rules as batches 1-2, "
    "internal research notes; batch-3 dispatch = Build Ops step 3): "
    "10 new engine-key -> lock-key pairs added to TIER_A_MAP + 4 new Tier-B compositions, all under "
    "the single new engine call read_ported_cosmology() (the_run.py, '# ==== S1b PORTED READS "
    "(batch 3: cosmology) ====' section; plus its helper read_z_eff_adopted). FROZEN ROSTER (printed "
    "from the regenerated manifest's Tier-C engine-surface-missing cosmology-sector rows BEFORE "
    "porting): Omega_DM_over_Omega_m, Omega_DM, Omega_b, z_eff, Lambda_CC, w_DE, H_0, t_0, "
    "A_hemispherical, epsilon_CP, eta_B, beta_cosmic_birefringence, N_eff, T_e_ann (14 rows) -- every "
    "row already present in SYMBOL_TABLE (Tier C, blocker engine-surface-missing) before this batch. "
    "TIER SPLIT (per the dispatch's adoption clause): 10 rows Tier A (pure engine primitives + the "
    "batch-2 engine-native N_hub chain: the coasting suite H_0=1/(N t_P), t_0=N t_P, Lambda=1/N^2 "
    "reuses read_higgs_chain; eta_B = eps_CP*Re(h_P)*alpha_1^M rides the native ihara_bass_root "
    "P-point walker + U_RUN; A_hemispherical/epsilon_CP == read_clock's own eps=1/5 -- the IDENTICAL "
    "Beta(1,1)->Beta(2,1) toggle-disconfirmation formula with identical inputs Pf=1/p_toggle, "
    "Pp=1/k*, per both files' own derivation text, NOT a numeric coincidence pairing; N_eff = "
    "read_flavor().gens per N_eff.py's own chain 'observer dim 3 -> 3 generations -> 3 nu_L'; "
    "T_e_ann = m_e/k* with the batch-2 engine-native m_e); 4 rows Tier B with adoptions named "
    "(Omega_DM/Omega_b = bias(z_eff) x the engine Poisson ratio, adoption z_eff; z_eff itself = the "
    "transcribed Fisher first-moment over the DECLARED external survey-design tables -- its honest "
    "form is Tier B per the ledger's own 'ADOPTED cosmology parameter (N_hub-class)' tag, never a "
    "silent Tier-A hardcode; beta_cosmic_birefringence = degrees(sin(arg h_P)*alpha_EM) with the "
    "native sqrt(5/8) engine core and the framework's own alpha_EM(M_Z) lock adopted -- its RG chain "
    "is batch-4 gauge scope, same posture as the existing g_2 row). N_eff ADJUDICATION (supersedes the "
    "batch-3 implementation pass's original 'different in kind' claim, which the adversarial check REFUTED): "
    "N_eff.py's real DAG ingredient is observer_dim_three_pred (MDL+Gleason), not read_flavor().gens; "
    "mapping via gens re-performed the rejected forced pairing one level removed. N_eff is DEMOTED to "
    "Tier C (blocker engine-surface-missing:observer_dim_three) until that chain is ported. NEW EXTERNAL "
    "constants transcribed with provenance (the prediction files' own single-source SI/unit "
    "translations, same declared-external status as batch 2's M_PL_GEV/G_F_MEASURED): hbar[GeV s] "
    "= 6.582119569e-25 + Mpc[km] = 3.085677581e19 (both from predictions/M_Pl_natural.py's "
    "single-source block), Gyr[s] = 3.1557e16 (predictions/t_0.py), M_PL_GEV re-declared in the "
    "batch-3 section (append-only law forbids editing batch 2) solely to form t_P = hbar/M_Pl as "
    "the source does. DEFERRALS (all remain Tier C, honest): m_nu2/m_nu3 (Neutrino sector, not "
    "cosmology; m_nu-scale is the ML-2 species-lift gate); the gauge/EW engine-surface-missing "
    "rows (g_1/g_3/alpha_GUT/sin2_theta_W_MZ/alpha_s/alpha_EM/M_unif/M_Z/m_W/Gamma ratios/"
    "theta_QCD) = batch-4 scope; the framework-internal misc rows (srs_cubic_moment/"
    "srs_bloch_lv_dim6/e_bit/M_Pl_natural/feshbach_exponent_principle/koide_quark_ratio/"
    "lambda_toggle_rate/xi_t/S_fresh/S_disconfirm/eta_5/eta_lattice/scale_energy_hashimoto/"
    "universe_transparency/observer_dim_three/G_N/tan_beta) = batch-4/misc scope; A_s skipped "
    "(ledger/lock ORPHAN per the dispatch); all physics-blocked rows (theta_*-family local-metric, "
    "B2 response, ML-2 BBN) untouched. Every ported value was independently numerically verified "
    "against its lock at <1e-9 relative BEFORE freezing (Omega_DM_over_Omega_m/Omega_DM/Omega_b/"
    "z_eff/A_hemispherical/epsilon_CP/N_eff/w_DE at exact float equality; H_0/t_0/Lambda_CC/eta_B/"
    "beta at ~1e-15; T_e_ann at ~1.9e-14 -- float-arithmetic-order noise throughout; zero "
    "engine-vs-lock inconsistencies found).",
    "S1b BATCH 4 (gauge+misc, THE FINAL S1b PORTING BATCH) additions, 2026-07-09 (same frozen "
    "porting rules as batches 1-3, internal research notes): "
    "32 new engine-key -> lock-key pairs added to TIER_A_MAP, all under the single new engine call "
    "read_ported_gauge_running() (the_run.py, '# ==== S1b PORTED READS (batch 4: gauge+misc) ====' "
    "section). FROZEN ROSTER (printed from the regenerated manifest's Tier-C engine-surface-missing "
    "rows BEFORE porting; the roster print is authoritative per the porting rules): the gauge/EW RG "
    "chain (g_1, g_3, alpha_GUT, sin2_theta_W_MZ, alpha_s, alpha_EM, M_unif, M_Z, m_W, "
    "Gamma_Z_over_M_Z, Gamma_W_over_Gamma_Z, theta_QCD -- 12 rows), the 2 neutrino masses (m_nu2, "
    "m_nu3), N_eff + observer_dim_three, and 16 framework-internal misc rows (srs_cubic_moment, "
    "srs_bloch_lv_dim6, e_bit, M_Pl_natural, feshbach_exponent_principle, koide_quark_ratio, "
    "lambda_toggle_rate, xi_t_temporal_correlation, S_fresh, S_disconfirm, eta_5_lorentz_dim5, "
    "eta_lattice_lorentz_dim6, scale_energy_hashimoto, universe_transparency, tan_beta) + the ONE "
    "blank-File-column row (G_N -- ledger blocker string 'engine-surface-missing:--'; "
    "predictions/G_N.py exists but is un-cited in the ledger's File column, located by direct "
    "search of predictions/, disclosed here rather than silently patched) = 32 rows total -- every "
    "row already present in SYMBOL_TABLE (Tier C) before this batch. DEEPEST PRIMITIVES-REUSE OF "
    "ANY S1b BATCH: the gauge chain's own one-loop beta-function VALUES {33/5,1,-3} are consumed "
    "from this module's OWN read_gauge_running() (the engine's derived 4D-completion beta), NOT "
    "re-typed from predictions/mssm_beta_coefficients.py's literal -- a strictly MORE native "
    "transcription than the source files themselves use (they import the literal; the engine "
    "reads its own derived equivalent, verified identical by read_gauge_running's own b4d==b_MSSM_lit "
    "assertion). v_higgs/N_hub/m_t/m_b reuse batch 2's read_higgs_chain()/read_ported_quark_masses(); "
    "R_nu_splitting=228/7 reuses batch 1's read_R_nu_splitting(); delta_r/delta_rho reuse the "
    "pre-existing native read_obliques(); p_create=1/p_toggle, p_destroy=1/k* (lambda_toggle_rate, "
    "xi_t, S_fresh, S_disconfirm) are the IDENTICAL Pf/Pp pair already used by read_clock(). "
    "N_eff RE-PROMOTION (dispatch clause 3, resolves the batch-3 adjudication): observer_dim_three "
    "is now ported as its OWN distinct engine key (the source file's hardcoded literal 3, backed by "
    "the MDL+Gleason 1957 theorem -- structurally parallel to theta_QCD's hardcoded 0, NOT a forced "
    "re-pairing of srs.DEG or read_flavor().gens, both already used for other locks); N_eff = "
    "predict_N_eff(observer_dim_three_pred) = observer_dim_three_pred per N_eff.py's own identity "
    "chain, now mapped via THIS TRUE ingredient. Batch-3's read_ported_cosmology().N_eff (= "
    "read_flavor()[3], the REJECTED ingredient) is left untouched (append-only law) but stays "
    "UNMAPPED/superseded -- N_eff's Tier-A mapping is exclusively "
    "read_ported_gauge_running().N_eff. BETA_COSMIC_BIREFRINGENCE UPGRADE (dispatch clause 1): now "
    "that alpha_EM(M_Z) is engine-native (read_ported_gauge_running().alpha_EM), the pre-existing "
    "Tier-B beta_cosmic_birefringence composition (_tier_b_compositions) is UPGRADED to consume the "
    "native value directly instead of locks['alpha_EM']; its 'adoptions' field now reads 'none' "
    "(the row's last external adoption is retired). g_2's own pre-existing Tier-B composition "
    "(read_gauge_consistency over locks['alpha_EM']/locks['sin2_theta_W_MZ']) was CONSIDERED for the "
    "same upgrade but is explicitly OUT of this batch's named scope (the dispatch names only "
    "beta_cosmic_birefringence for re-check) -- left as-is, flagged here for a future batch, not "
    "silently changed. NON-TRANSCRIBABLE-IN-SCOPE: NONE -- every roster row (including the ODE-based "
    "tan_beta, ported via scipy solve_ivp/brentq exactly as predictions/tan_beta.py's own MSSM "
    "Yukawa-RGE self-consistency search) transcribed faithfully; no deferrals this batch. Every "
    "value was independently numerically verified against its lock BEFORE freezing (the_run.py's "
    "own batch-4 __main__ self-test): g_1/g_3/alpha_GUT/sin2_theta_W_MZ/alpha_s/alpha_EM/M_unif/M_Z/"
    "m_W/Gamma_Z_over_M_Z/Gamma_W_over_Gamma_Z/theta_QCD/N_eff/observer_dim_three/srs_cubic_moment/"
    "srs_bloch_lv_dim6/e_bit/M_Pl_natural/feshbach_exponent_principle/koide_quark_ratio/"
    "lambda_toggle_rate/xi_t_temporal_correlation/S_fresh/eta_5_lorentz_dim5/"
    "eta_lattice_lorentz_dim6/scale_energy_hashimoto/universe_transparency/G_N/tan_beta at exact "
    "float equality (<1e-9 relative); m_nu2/m_nu3/S_disconfirm at ~1e-15 -- float-arithmetic-order "
    "noise; zero engine-vs-lock inconsistencies found -- S1b PORTING CAMPAIGN COMPLETE.",
    "S1b ORPHAN CLEANUP (2026-07-09, user-approved re-freeze -- a deliberate, reviewed re-freeze of "
    "predictions/_value_locks.json, 104 -> 107 values): three orphan ledger rows (T_nu_dec, "
    "h_walker_eigenvalue, observer_hilbert_space) were candidates for lock registration. OUTCOME per "
    "row: (1) T_nu_dec -- predictions/T_nu_dec.py's T_nu_dec_pred_MeV = 0.8443997597588065 (MeV) "
    "REGISTERED as a new lock. No the_run.py engine surface computes a neutrino-decoupling "
    "rate-balance quantity (checked: batch-1..4 ported reads do not cover it) -- it lands as an "
    "ORPHAN-TURNED-REGISTERED-LOCK (present in the lock file, absent from TIER_A_MAP/tier-b "
    "compositions, therefore surfaced in the UNMAPPED LOCKS list below with an explanatory note) -- "
    "classified honestly, not silently mapped, not built. (2) h_walker_eigenvalue -- "
    "predictions/h_walker_eigenvalue.py's h_walker_eigenvalue_pred = (sqrt(3)+i*sqrt(5))/2 is the "
    "FIRST complex-valued candidate for this lock file; every one of the pre-existing 104 locks is a "
    "real float/int (confirmed by inspection), and scripts/value_lock.py's own collect_values() "
    "explicitly skips isinstance(p, complex) values ('no scalar predicted value to lock') -- so "
    "there was no prior complex-value convention to reuse. REGISTERED as two real-valued keys, "
    "h_walker_eigenvalue_re=0.8660254037844386 (=sqrt(3)/2) and "
    "h_walker_eigenvalue_im=1.118033988749895 (=sqrt(5)/2), establishing the _re/_im split as the "
    "go-forward convention for any future complex lock. UNLIKE T_nu_dec, the engine DOES produce "
    "this natively: the_run.py's module-level ihara_bass_root(lam) at lam=sqrt(LAM_PERRON) is the "
    "IDENTICAL P-point walker root read_obliques() and the PMNS Majorana-phase construction already "
    "consume internally (their own comments say so) -- so a genuine Tier-A engine read was added "
    "(phase1_engine_reads(): 'run.ihara_bass_root(sqrt_LAM_PERRON).re/.im') and mapped in TIER_A_MAP, "
    "not left as an unmapped orphan lock. (3) observer_hilbert_space -- predictions/"
    "observer_hilbert_space.py's predict_observer_hilbert_space() returns a STRUCTURAL DICT "
    "({G1_hilbert_space_structure_exists: bool, G5_field: 'C'/'undetermined', axioms_used: [...], "
    "cited_theorem: str, field_exclusions: {...}, chain_input_check: bool}), not a scalar float/int. "
    "predictions/_value_locks.json's schema (every one of the 107 'values' entries, before and after "
    "this re-freeze, is a bare JSON number) cannot faithfully hold this without INVENTING an encoding "
    "(e.g. collapsing to a boolean 1.0/0.0 would silently discard G5_field/axioms_used/cited_theorem "
    "-- exactly the kind of invented encoding the task's own instructions forbade). HONEST FALLBACK: "
    "SKIPPED -- format-blocked, NOT registered, disclosed here rather than silently omitted. "
    "SEPARATELY (reclassification, not a lock registration): the ledger's m_nu1 row was moved from "
    "'orphan' to a new 'structural-exact' read-class with blocker 'none' (see STRUCTURAL_EXACT_NOTES "
    "+ phase5_ledger_tiering below) -- the ledger's own target_parameters.md prose for m_nu1 already "
    "says 'No predictions/m_nu1.py file (value is structurally zero -- no DAG node needed)'; this is "
    "a FORCED THEOREM-GRADE zero (W45 closure: rank-2 Type-I seesaw, the trivial-C3 generation hosts "
    "no nu_R Majorana mass) with nothing further to build, NOT an orphan awaiting an engine surface -- "
    "the two are different in kind and the old blanket 'orphan' blocker conflated them. Two unmapped "
    "locks (delta_CP_CKM_geometry, ew_width_layer) were annotated (not reclassified, not lock-edited) "
    "per UNMAPPED_LOCK_NOTES below: delta_CP_CKM_geometry is a DEDUP alias of the delta_CP_CKM row "
    "(same ledger row, target_parameters.md line 107, cites both predictions/delta_CP_CKM_geometry.py "
    "and the delta_CP_CKM.py tombstone); ew_width_layer is an INTERNAL-ONLY layer quantity (consumed "
    "by predictions/Gamma_Z_over_M_Z.py and Gamma_W_over_Gamma_Z.py, embedded in the Gamma_Z/M_Z "
    "ledger row's own Notes prose, not a standalone observable row). QUEUE NOTES (not executed this "
    "pass -- file-writing tasks, explicitly out of scope for a value-lock/classification pass): "
    "A_s (primordial amplitude) and T(N) propagation function both still lack predictions/*.py files "
    "(SYMBOL_TABLE: 'A_s (primordial amplitude)' -> (None, S); 'T(N) propagation function' -> "
    "(None, Q)) -- both rows STAY orphan (blocker='orphan') until those files are written; queued as "
    "micro build-tasks, not attempted here (no physics; no new prediction files this pass). N_hub's "
    "own ledger status: per explicit user decision (2026-07-09), N_hub gets NO ledger row and NO lock "
    "entry of its own -- it is documented instead via the new 'THE CALIBRATION (N_hub)' section "
    "(printed + doc-written below): N_hub is the framework's calibration input, not an observable.",
]

# S1b ORPHAN CLEANUP annotations (comments/output ONLY -- these never edit predictions/_value_locks.json
# or docs/parameters/target_parameters.md; they are explanatory notes surfaced next to the UNMAPPED
# LOCKS list, both on the console and in the written doc). See the MAPPING-REVISIONS entry above for
# the full disclosure this summarizes.
UNMAPPED_LOCK_NOTES = {
    "delta_CP_CKM_geometry": (
        "DEDUP: alias of the delta_CP_CKM row. Same ledger row (delta_CP^CKM, target_parameters.md "
        "line 107), same value (70.52877936550931 deg = arccos(1/k*)), two source files "
        "(predictions/delta_CP_CKM_geometry.py live + predictions/delta_CP_CKM.py tombstone) cited on "
        "that ONE row -- not a second observable. Not deduped in the lock file itself (no lock edits "
        "made); noted here only."
    ),
    "ew_width_layer": (
        "INTERNAL-ONLY: not a standalone ledger observable. predictions/ew_width_layer.py is an "
        "internal EW radiative-width LAYER (delta_Z = -0.4864%) consumed by "
        "predictions/Gamma_Z_over_M_Z.py and predictions/Gamma_W_over_Gamma_Z.py; its value is "
        "embedded in the Gamma_Z/M_Z ledger row's own Notes prose (target_parameters.md line 64), not "
        "a separate row of its own."
    ),
    "T_nu_dec": (
        "S1b ORPHAN CLEANUP (2026-07-09): NEWLY REGISTERED lock (was absent before this re-freeze); "
        "no the_run.py engine surface reproduces it (checked: batch-1..4 ported reads do not cover a "
        "neutrino-decoupling rate-balance quantity) -- an orphan-turned-registered-lock, honestly left "
        "unmapped pending a future engine-surface build (NOT attempted this pass). "
        "SUPERSEDED (LIGHT BATCH 2026-07-10): the engine surface was BUILT -- the_run.py "
        "read_T_nu_dec() (faithful predictions/T_nu_dec.py v2.0.0 port, exact float match to the "
        "lock) is now Tier-A-mapped, so this lock no longer appears in the unmapped list; note "
        "retained as the historical S1b record."
    ),
    # [2026-07-13 hygiene, Push-3 W-backlog] disclosed provenance note, per
    # internal research notes §3 ("m_bb = 3.5644 meV -- SHARP"), which
    # explicitly directs: "flagged here for the architect's awareness, not silently corrected in
    # the register." This note does NOT reclassify the m_bb ledger row's Status column (still
    # transcribed verbatim from target_parameters.md's own "Status" field, the CANDIDATE tag) --
    # it only discloses WHY the tag looks stale next to a SHARP downstream result, so the two are
    # not mistaken for a contradiction.
    "harvest_m_bb_meV_conv1": (
        "NOT a contradiction of the SHARP m_bb verdict. This manifest's Tier-A/B pass walks ONLY "
        "the_run.py's read_ported_* engine surface; m_bb's actual computation (|sum_i U^2_ei m_i| "
        "from the certified PMNS matrix + both Majorana phases + the neutrino masses) lives one "
        "layer down, in proofs/flavor/srs_unified_mixing.py §8 and proofs/foundations/"
        "R1_HARVEST_2026-07-10.py (H-3) -- scripts that CONSUME read_ported_* outputs (alpha_31, "
        "alpha_21, the mass spectrum) rather than being themselves a read_ported_* export, so no "
        "Tier-A/B mapping reaches this lock from this manifest's narrow scope. The ledger row's own "
        "'m_bb (effective Majorana mass, 0vbb)' Status column (target_parameters.md, EXP-F1) "
        "predates the 2026-07-10 alpha_31 phase-convention resolution and still reads CANDIDATE -- "
        "a generator-scope artifact of this manifest's narrow read_ported_* walk, not a live open "
        "item. The value itself IS sharp: m_bb = 3.5643862355704257 meV, cross-checked (exact "
        "match) against read_ported_flavor().alpha_31_PMNS = 197.6124391 deg. See "
        "internal research notes §3 for the full disclosure and the "
        "falsification criterion; per that section's own instruction, the ledger Status tag is "
        "NOT hand-corrected here -- only the provenance is disclosed."
    ),
    "harvest_m_bb_meV_conv2": (
        "Same disclosure as harvest_m_bb_meV_conv1 (the second Majorana-phase convention of the "
        "same m_bb computation) -- see that note and internal research notes §3."
    ),
}

# S1b ORPHAN CLEANUP (2026-07-09): ledger rows with a FORCED-STRUCTURAL zero/exact result that has NO
# predictions/*.py file and NO lock value -- the ledger's OWN prose says so explicitly (cited per row
# below). Distinct in kind from an "orphan" (which means a missing engine surface still to be built):
# these rows have NOTHING left to build. Read-class 'structural-exact', blocker 'none' (the string, not
# Python None -- phase5_ledger_tiering's blocker_counts print loop format-specs every blocker as a
# string; Python None would crash that f-string).
STRUCTURAL_EXACT_NOTES = {
    "m_ν1": (
        "ledger's own note (target_parameters.md m_ν1 row): 'No predictions/m_nu1.py file (value is "
        "structurally zero -- no DAG node needed).' W45 closure: Hashimoto B(P) splits 8 Ramanujan + "
        "4 trivial modes; the trivial-C3 generation hosts no nu_R Majorana mass; Type-I seesaw "
        "(2 nu_R x 3 nu_L) is rank-2 => exactly one massless light neutrino. FORCED THEOREM-GRADE "
        "zero, NOT an orphan awaiting an engine surface."
    ),
}


def phase2_mapping(E, locks):
    banner("PHASE 2 -- THE MAPPING TABLE (M-1, frozen BEFORE comparison)")
    print("TIER-A engine-key -> lock-key (see justification per row):")
    for ek, (lk, tol, why) in sorted(TIER_A_MAP.items()):
        print(f"  {ek:32s} -> {lk:28s} [{tol:5s}]  {why}")
    tb = _tier_b_compositions(E, locks)
    print(f"\nTIER-B composed lock-keys ({len(tb)}): {sorted(tb.keys())}  (formulas printed in PHASE 4)")

    mapped_locks = set(lk for lk, _, _ in TIER_A_MAP.values()) | set(tb.keys())
    unmapped_locks = sorted(set(locks.keys()) - mapped_locks)
    print(f"\nUNMAPPED LOCKS ({len(unmapped_locks)} of {len(locks)}) -- present in the frozen lock "
          f"file, no engine (Tier-A) or composition (Tier-B) reproduces them in THIS manifest's "
          f"narrow engine scope (the_run.py/the_net.py/srs.py only):")
    for lk in unmapped_locks:
        note = UNMAPPED_LOCK_NOTES.get(lk)
        if note:
            print(f"    {lk} = {locks[lk]}")
            print(f"        NOTE: {note}")
        else:
            print(f"    {lk} = {locks[lk]}")

    used_engine_keys = set(TIER_A_MAP.keys())
    all_engine_keys = set(E.keys())
    unplayed = sorted(all_engine_keys - used_engine_keys)
    print(f"\nUNPLAYED ENGINE NOTES ({len(unplayed)} of {len(all_engine_keys)}) -- real engine "
          f"outputs with NO lock match at all (candidate PREDICTIONS the ledger never registered):")
    for ek in unplayed:
        print(f"    {ek} = {E[ek]}")

    print("\nMAPPING-REVISIONS (full disclosure; frozen-before-comparison discipline restored by "
          "re-running from PHASE 1 after each):")
    for rev in MAPPING_REVISIONS:
        print(f"  - {rev}")

    return tb, unmapped_locks, unplayed


# ==============================================================================================
# PHASE 3 -- M-2 TIER-A COMPARISONS.  Engine value vs lock, printed raw; PASS at declared
#   tolerance.  A mismatch does NOT stop the run -- it is booked as a FINDING row.
# ==============================================================================================
def phase3_tier_a_compare(E, locks):
    banner("PHASE 3 -- M-2 TIER-A COMPARISONS (engine vs lock, frozen mapping)")
    rows = []
    for ek, (lk, tolclass, why) in sorted(TIER_A_MAP.items()):
        ev = E[ek]
        lv = locks[lk]
        evf = float(ev)
        d = evf - lv
        rel = d / lv if lv else float("nan")
        tol = TOL_EXACT if tolclass == "exact" else TOL_FLOAT
        passed = (abs(d) < tol) if tolclass == "exact" else (abs(rel) < tol)
        rows.append(dict(engine_key=ek, lock_key=lk, engine_val=evf, lock_val=lv,
                          delta=d, rel=rel, tol=tol, tolclass=tolclass, passed=passed))
        flag = "PASS" if passed else "**MISMATCH**"
        print(f"  [{flag:12s}] {ek:32s} = {evf:.12g}   lock[{lk}] = {lv:.12g}   "
              f"rel_delta = {rel:+.3e}  (tol {tolclass} {tol:g})")
    n_pass = sum(r["passed"] for r in rows)
    print(f"\n  Tier-A: {n_pass}/{len(rows)} PASS.")
    mismatches = [r for r in rows if not r["passed"]]
    if mismatches:
        print("  MISMATCHES (raw, unresolved as of this run):")
        for r in mismatches:
            print(f"    {r['engine_key']} vs {r['lock_key']}: rel_delta={r['rel']:+.3e}")
    else:
        print("  No Tier-A mismatches.")
    return rows


# ==============================================================================================
# PHASE 4 -- M-3 TIER-B COMPOSITIONS.  Recompute from Tier-A outputs + registered adoptions;
#   adoption list printed per row.
# ==============================================================================================
def phase4_tier_b_compare(E, locks):
    banner("PHASE 4 -- M-3 TIER-B COMPOSITIONS (declared compositions, adoptions named per row)")
    tb = _tier_b_compositions(E, locks)
    rows = []
    for lk, spec in sorted(tb.items()):
        ev = spec["value"]
        lv = locks[lk]
        d = ev - lv
        rel = d / lv if lv else float("nan")
        tol = TOL_FLOAT
        passed = abs(rel) < tol
        rows.append(dict(lock_key=lk, engine_val=ev, lock_val=lv, delta=d, rel=rel,
                          passed=passed, formula=spec["formula"], cite=spec["cite"],
                          adoptions=spec["adoptions"]))
        flag = "PASS" if passed else "**MISMATCH**"
        print(f"  [{flag:12s}] {lk}")
        print(f"      formula:   {spec['formula']}")
        print(f"      cite:      {spec['cite']}")
        print(f"      adoptions: {spec['adoptions']}")
        print(f"      computed = {ev:.12g}   lock = {lv:.12g}   rel_delta = {rel:+.3e}")
    n_pass = sum(r["passed"] for r in rows)
    print(f"\n  Tier-B: {n_pass}/{len(rows)} PASS.")
    return rows


# ==============================================================================================
# BONUS (NOT counted in any Tier total): dimensionless mass-RATIO cross-checks.  read_masses() is
# FULLY FORCED (no free parameter, unlike read_generation/c3_winding_bases which needs a
# calibrated free axis s) -- so its ratios are a legitimate, if informal, extra finding.  These do
# NOT correspond to any single lock key (no lock is literally "m_mu_over_m_e"), so they are kept
# OUT of the Tier-A/B/C ledger-row accounting to avoid inflating the coverage numbers.
# ==============================================================================================
def bonus_mass_ratios(E, locks):
    banner("BONUS (not tier-counted) -- dimensionless mass-ratio cross-checks via read_masses()")
    m1, m2, m3 = E["read_masses()[3].m1"], E["read_masses()[3].m2"], E["read_masses()[3].m3"]
    r21_engine, r31_engine = m2 / m1, m3 / m1
    r21_lock = locks["m_mu"] / locks["m_e"]
    r31_lock = locks["m_tau"] / locks["m_e"]
    print(f"  m_mu/m_e   engine(read_masses, forced) = {r21_engine:.6f}   "
          f"lock-ratio (m_mu/m_e) = {r21_lock:.6f}   rel = {(r21_engine - r21_lock) / r21_lock:+.3e}")
    print(f"  m_tau/m_e  engine(read_masses, forced) = {r31_engine:.6f}   "
          f"lock-ratio (m_tau/m_e) = {r31_lock:.6f}   rel = {(r31_engine - r31_lock) / r31_lock:+.3e}")
    print("  (NOT a Tier-A/B claim: no lock key named 'm_mu_over_m_e' exists; kept out of the "
          "formal coverage counts by design, per the pre-reg's 'no forced pairings' discipline.)")


# ==============================================================================================
# PHASE 5 -- the ledger-row classification: SYMBOL_TABLE.  For every one of the ~150-161 ledger
#   rows (parsed in PHASE 0), an explicit (lock_key, bin) assignment.  BIN assignment cites
#   internal research notes SECTION 2 verbatim lists;
#   a row gets bin '?' when the bins doc's own §2 lists do not explicitly name it (per the task's
#   own instruction: "rows not explicitly binned there get bin '?' printed honestly" -- NO
#   inferred/guessed bin is assigned by keyword-similarity).
#
#   Bin S    -- "Gauge sector (g1,g2,g3,alpha_GUT,sin2thetaW x2,alpha_s,M_unif), Higgs
#                (v,m_H,lambda,lambda3), 9 fermion masses, CKM (all 9 + J + delta_CP), PMNS
#                angles + phases, R_nu, m_nu1=0, theta_QCD, Gamma_Z/M_Z, Gamma_W/Gamma_Z, delta_rho,
#                delta_r, eta_B, eps_CP, A_hemisphere, beta birefringence, N_eff, Omega_DM/Omega_m,
#                Omega/Lambda/w_DE/z_eff family, H0 both sides, t0, H(z)/q0/w_eff/S4 distances, A_s,
#                Lorentz/LIV rows, lattice internals." PLUS the explicit "residual in-bin opens:
#                alpha_EM(M_Z), M_Z pole oblique, V_ts/V_tb, m_W" note directly following that list.
#   Bin L-sector -- "m_e -70ppm, m_mu -60ppm, m_nu2, m_nu3, B1 hadron anchoring -> Y_p, D/H, 3He/H,
#                7Li/H, nucleon-dependent T_BBN rows, z_drag/r_drag contributions."
#   Bin L-metric -- "G magnitude (sharp 2pi), era-exponent magnitudes, native z_eq, theta_*
#                (booked ~9x), r_*, r_drag, theta_MC, z_*."
#   Bin L-response -- "n_s, sigma_8, S_8, D(z), f(z), fsigma_8, A_s refinement."
#   Bin X    -- "tau, z_reion (star-formation primitive absent), Delta-alpha/atomic-frame rows
#                (R_inf, T_recomb, T_HeI/II), the SUSY block."
#   Bin gear (structural/definitional) -- "~10 rows, free" == the Structural/definitional section.
#
#   r_drag / z_drag straddle both L-sector ("z_drag/r_drag contributions") and L-metric ("r_*,
#   r_drag, ... z_*"); resolved: r_drag -> L-metric (named twice, magnitude focus), z_drag ->
#   L-sector (named once, "contributions" framing). m_e/m_mu are explicitly named in L-sector
#   (more specific) even though "9 fermion masses" nominally covers them under S; the specific
#   name wins.  The Omega/Lambda/w_DE/z_eff "family" phrase is read to cover its full named
#   family (Omega_DM, Omega_b, Omega_m_LCDM, Omega_Lambda_LCDM, z_eff, Lambda_CC, Lambda_LCDM,
#   w_DE) even though only 4 tokens are literally spelled out -- a defensible, not a strict,
#   reading, flagged here.
# ==============================================================================================
S, LSEC, LMET, LRESP, X, GEAR, Q = "S", "L-sector", "L-metric", "L-response", "X", "gear", "?"

SYMBOL_TABLE = {
    # --- Gauge couplings ---
    "g_1 (GUT-norm, M_Z)": ("g_1", S), "g_2 (SU(2), M_Z)": ("g_2", S), "g_3 (SU(3), M_Z)": ("g_3", S),
    "α_GUT": ("alpha_GUT", S), "sin²θ_W (at M_unif)": ("sin2_theta_W", S), "sin²θ_W (at M_Z)": ("sin2_theta_W_MZ", S),
    "α_s (M_Z) ≡ g_3²/(4π)": ("alpha_s", S), "α_EM (M_Z)": ("alpha_EM", S), "M_unif": ("M_unif", S),
    "M_Z": ("M_Z", S), "m_W": ("m_W", S), "Γ_Z/M_Z": ("Gamma_Z_over_M_Z", S), "Γ_W/Γ_Z": ("Gamma_W_over_Gamma_Z", S),
    "δρ (ρ-param shift)": ("delta_rho", S), "δ_r (M_Z tree→pole oblique)": ("delta_r", S),
    "R∞ (Rydberg)": (None, X),
    # --- Higgs ---
    "v (Higgs VEV)": ("v_higgs", S), "m_H (Higgs mass)": ("m_H", S), "λ (Higgs quartic)": ("lambda_higgs", S),
    "λ_3 (Higgs trilinear)": ("lambda_3_higgs", S), "G_F (Fermi constant)": ("G_F", Q),
    # --- Charged fermion masses ---
    "m_e": ("m_e", LSEC), "m_μ": ("m_mu", LSEC), "m_τ": ("m_tau", S), "m_u": ("m_u", S), "m_d": ("m_d", S),
    "m_s": ("m_s", S), "m_c": ("m_c", S), "m_b": ("m_b", S), "m_t": ("m_t", S),
    # --- CKM ---
    "V_us": ("V_us", S), "V_cb": ("V_cb", S), "V_ub": ("V_ub", S), "V_ud": ("V_ud", S), "V_cd": ("V_cd", S),
    "V_cs": ("V_cs", S), "V_td": ("V_td", S), "V_ts": ("V_ts", S), "V_tb": ("V_tb", S),
    "δ_CP^CKM": ("delta_CP_CKM", S), "J_CKM (Jarlskog)": ("J_CKM", S),
    "Georgi-Jarlskog ratio (m_s/m_μ × m_e/m_d at GUT)": ("georgi_jarlskog", Q),
    # --- QCD ---
    "θ_QCD": ("theta_QCD", S),
    # --- Neutrino masses ---
    "m_ν1": (None, S), "m_ν2": ("m_nu2", LSEC), "m_ν3": ("m_nu3", LSEC), "R_ν = Δm²_31/Δm²_21": ("R_nu_splitting", S),
    # --- PMNS ---
    "θ_12_PMNS": ("theta_12_PMNS", S), "θ_13_PMNS": ("theta_13_PMNS", S), "θ_23_PMNS": ("theta_23_PMNS", S),
    "δ_CP_PMNS": ("delta_CP_PMNS", S), "α_21_PMNS": ("alpha_21_PMNS", S), "α_31_PMNS": ("alpha_31_PMNS", S),
    # --- Cosmology §1 ---
    # R1 HARVEST (2026-07-10) H-2: Omega_b_h2/Omega_c_h2 -- Tier-B compositions (z_eff-conditional,
    # inherited from the existing Omega_b/Omega_DM rows; see _tier_b_compositions() above).
    "Ω_b h² (physical baryon density)": ("harvest_Omega_b_h2", Q), "Ω_c h² (physical CDM density)": ("harvest_Omega_c_h2", Q),
    "100θ_MC (approx acoustic scale)": (None, LMET), "τ (reionization optical depth)": (None, X),
    "A_s (primordial amplitude)": (None, S), "n_s (scalar spectral tilt)": (None, LRESP),
    "r (tensor-to-scalar ratio)": (None, Q),
    # --- §2 Energy budget ---
    "Ω_DM/Ω_m": ("Omega_DM_over_Omega_m", S), "Ω_DM (ΛCDM-frame)": ("Omega_DM", S),
    "Ω_b (ΛCDM-frame)": ("Omega_b", S), "Ω_m_LCDM (ΛCDM-fit total matter)": ("Omega_m_LCDM", S),
    "Ω_Λ_LCDM (ΛCDM-fit dark energy)": ("Omega_Lambda_LCDM", S),
    # R1 HARVEST (2026-07-10) H-2: Omega_k=0 EXACT, pure engine (no adoption).
    "Ω_k (spatial curvature)": ("harvest_Omega_k", Q),
    # R1 HARVEST (2026-07-10) H-2: Sigma_m_nu, pure engine composite (m_nu1=0 + m_nu2 + m_nu3).
    "Σm_ν (neutrino mass sum)": ("harvest_Sigma_m_nu_eV", Q), "z_eff (cosmology effective redshift)": ("z_eff", S),
    "Λ_CC (substrate, Planck units)": ("Lambda_CC", S), "Λ_LCDM (ΛCDM-fit, Planck units)": ("Lambda_CC_LCDM", S),
    # R1 HARVEST (2026-07-10) H-1: w_eff=-1/3 EXACT (coasting a proportional to t kinematics).
    "w_DE (dark-energy COMPONENT EoS)": ("w_DE", S), "w_eff (TOTAL fluid EoS)": ("harvest_w_eff", S),
    # --- §3 Expansion ---
    "H_0 (substrate / CMB-side)": ("H_0", S), "H_0 (observer / SH0ES-side)": ("H_0_observer", S),
    # R1 HARVEST (2026-07-10) H-1: t_0(CMB frame) IDENTIFIED with t_0(substrate) via MC-4's
    # clock map (H_0(substrate) IS the CMB-side H_0 by the SAME row-pair's own naming) --
    # REUSES the existing 't_0' lock, no new lock needed; the Category-B CONTRAST vs Planck's
    # own CMB-inferred t_0=13.797 Gyr is printed in R1_HARVEST_2026-07-10.py (a contrast, not
    # a target -- see internal research notes H-1).
    "t_0 (substrate / stellar)": ("t_0", S), "t_0 (ΛCDM / CMB frame)": ("t_0", Q),
    # R1 HARVEST (2026-07-10) H-1: H(z)=H_0(1+z) at the declared z=1.0 anchor; q_0=0 EXACT.
    "H(z) (expansion history)": ("harvest_H_z1p0", S), "q_0 (deceleration parameter)": ("harvest_q_0", S),
    # --- §4 Distances ---
    # R1 HARVEST (2026-07-10) H-1: the coasting distance curves, declared z=1.0 anchor point
    # registered here (z=0.5/2.0 are additional orphan-style curve locks, no ledger row).
    "D_C(z) = D_M(z) (comoving / transverse comoving)": ("harvest_D_C_z1p0", S), "D_A(z) (angular-diameter distance)": ("harvest_D_A_z1p0", S),
    "D_L(z) (luminosity distance)": ("harvest_D_L_z1p0", S), "D_V(z) (BAO dilation scale)": ("harvest_D_V_z1p0", S),
    # --- §5 Growth ---
    "σ_8 (matter clustering amplitude)": (None, LRESP), "S_8 = σ_8√(Ω_m/0.3)": (None, LRESP),
    "D(z) (linear growth factor)": (None, LRESP), "f(z) (growth rate dlnD/dlna)": (None, LRESP),
    "fσ_8(z) (RSD observable)": (None, LRESP),
    # --- §6 Acoustic/recomb ---
    "r_* (sound horizon at recombination, was \"r_s\")": (None, LMET), "r_drag (sound horizon at baryon drag)": (None, LMET),
    "θ_* (acoustic angular scale)": (None, LMET), "z_* (recombination redshift)": (None, LMET),
    "z_drag (baryon drag epoch)": (None, LSEC), "z_reion (reionization redshift)": (None, X),
    "z_eq (matter-radiation equality)": (None, LMET),
    # --- §7 Asymmetry ---
    "A_hemispherical": ("A_hemispherical", S), "ε_CP_baryon": ("epsilon_CP", S),
    "η_B (baryon-to-photon)": ("eta_B", S), "β (cosmic birefringence)": ("beta_cosmic_birefringence", S),
    # --- §8 Thermal history ---
    "N_eff": ("N_eff", S), "T_ν_dec": (None, Q), "T_BBN-1 (weak freeze-out)": (None, LSEC),
    "T_e_ann": ("T_e_ann", Q), "T_BBN_D (D bottleneck)": (None, LSEC), "T_HeII (He²⁺→He⁺ recomb)": (None, X),
    "T_HeI (He⁺→He recomb)": (None, X), "T_recomb (H, z ≈ 1100)": (None, X),
    # R1 HARVEST (2026-07-10) H-5: T(N) via the S1d epoch API, evaluated AT N=N_hub (trivial
    # T(N_now)=T_today identity by construction; the calibration fence forbids the FULL
    # propagation curve at other N without the un-built era-crossing map -- disclosed, not stretched).
    "T(N) propagation function": ("harvest_T_of_N_now_eV", Q),
    # --- §9 BBN ---
    "Y_p (⁴He mass fraction)": (None, LSEC), "D/H (primordial deuterium)": (None, LSEC),
    "³He/H": (None, LSEC), "⁷Li/H": (None, LSEC),
    # --- Structural / definitional (all gear, per the doc's exact "~10 rows, free") ---
    # R1 HARVEST (2026-07-10) H-5: Fermion content = 48 (sum(species dims)*gens*p_toggle), the
    # ledger's own literal count -- cross-corroborated by aqft_net.py's HK-6a species_sector_dims.
    # The other 8 GEAR rows genuinely have NO existing numeric-lock check in the named adapter
    # scope (Gauge group/Charge quantization/Parity violation/Higgs rep/Lorentzian signature/
    # Matter stability/Low initial entropy/Spacetime dimension are qualitative or already-used-
    # elsewhere quantities -- forcing a lock would be an invented encoding or a forced re-pairing;
    # see R1_HARVEST_2026-07-10.py's H-5 section for the full per-row CITATION, left honestly
    # orphaned here per the pre-reg's "do not stretch" clause).
    "Spacetime dimension": (None, GEAR), "Gauge group": (None, GEAR), "Number of generations": ("R3_observer_c3_generation", GEAR),
    "Charge quantization": (None, GEAR), "Parity violation": (None, GEAR), "Fermion content": ("harvest_fermion_content", GEAR),
    "Higgs rep": (None, GEAR), "Lorentzian signature": (None, GEAR), "Matter stability": (None, GEAR),
    "Low initial entropy": (None, GEAR),
    # --- Framework-internal: lattice structure (Bin S's "lattice internals") ---
    # R1 HARVEST (2026-07-10) H-5: h_walker_eigenvalue's ledger row (the COMPLEX h_P itself) had
    # no scalar lock (only the split _re/_im pair, already Tier-A); harvest_h_walker_abs2=|h_P|^2=2
    # gives the row itself a genuine, non-invented scalar (the Ramanujan-saturation identity every
    # h_P-consuming read already asserts). srs_dirac_cone_velocities similarly gets
    # harvest_cone_velocity_v0 (the SAME construction as the_net.py's cone_velocity, cross-checked).
    "k* (coordination)": ("k_star", S), "d_spatial": ("d_spatial", S), "g_girth": ("g_girth", S),
    "p_toggle": ("p_toggle", S), "h_walker_eigenvalue": ("harvest_h_walker_abs2", S), "srs_E_at_P": ("srs_E_at_P", S),
    "srs_cubic_moment": ("srs_cubic_moment", S), "srs_dirac_cone_velocities": ("harvest_cone_velocity_v0", S),
    "srs_bloch_lv_dim6": ("srs_bloch_lv_dim6", S),
    # --- Natural units (not explicitly named in the bins doc) ---
    "e_bit (energy of one substrate edge toggle)": ("e_bit", Q), "M_Pl_natural": ("M_Pl_natural", Q),
    # --- Couplings derived (not explicitly named) ---
    "α_1_bare": ("alpha_1", Q), "α_1_full": ("alpha_1_full", Q), "y_τ (tau Yukawa)": ("y_tau", Q),
    "Feshbach exponent principle": ("feshbach_exponent_principle", Q),
    # --- Koide (not explicitly named as its own bin item) ---
    "Q_Koide": ("Q_Koide", Q), "ε_Koide": ("epsilon_Koide", Q), "δ_Koide": ("delta_Koide", Q),
    "koide_quark_ratio": ("koide_quark_ratio", Q),
    # --- Session 9-10 ("Lorentz/LIV rows" explicitly names eta_5/eta_lattice -> S; rest -> ?) ---
    "λ toggle rate": ("lambda_toggle_rate", Q), "ξ_t temporal correlation": ("xi_t_temporal_correlation", Q),
    "S_fresh": ("S_fresh", Q), "S_disconfirm": ("S_disconfirm", Q), "η_5 (dim-5 LIV)": ("eta_5_lorentz_dim5", S),
    "η_lattice (dim-6 LIV)": ("eta_lattice_lorentz_dim6", S), "Scale energy (Hashimoto)": ("scale_energy_hashimoto", Q),
    "Universe transparency onset": ("universe_transparency", Q),
    # --- Branch measure & observer ---
    "Branch measure μ": (None, Q), "Observer H = C³": ("observer_dim_three", Q), "Observer Hilbert space": (None, Q),
    # --- Framework-adjacent ---
    "G (Newton's constant)": ("G_N", Q),
    # --- SUSY (explicit "the SUSY block" -> X, all 9 rows) ---
    "tan β": ("tan_beta", X), "SUSY scale": (None, X), "m_gluino": (None, X), "m_squark": (None, X),
    "m_slepton": (None, X), "m_neutralino": (None, X), "m_chargino": (None, X),
    "m_h (light Higgs, MSSM)": (None, X), "m_H, m_A, m_H± (heavy Higgs)": (None, X),
}

BLOCKER_BY_BIN = {
    LSEC: "species-lift/ML-2", LMET: "local-metric/ML-3/4/D1b", LRESP: "response/B2",
    X: "external", S: "orphan", GEAR: "orphan", Q: "orphan",
}

# THE CALIBRATION (N_hub) -- lock keys whose match to observation is BY CONSTRUCTION, not an
# independent prediction. N_hub (the framework's single continuous input) is pinned by requiring
# the derived G_F = 1/(sqrt2*v_higgs^2) to match the measured Fermi constant; v_higgs feeds that
# identity directly. Flagged in a dedicated 'calibration' column on the full ledger-row table (see
# phase5_ledger_tiering / write_manifest_doc) and explained in THE CALIBRATION (N_hub) section.
CALIBRATION_ROUND_TRIP = {"G_F", "v_higgs"}

# THE CALIBRATION (N_hub) -- printed (banner + paragraphs) AND doc-written verbatim (same source of
# truth, no drift between the two). Plain language per the approved task; cites the epoch guardrail
# in internal research notes (added 2026-07-09, user directive) rather than
# re-deriving it.
CALIBRATION_SECTION_PARAGRAPHS = [
    "N_hub is the framework's SINGLE continuous input. It is not a free-floating constant: it is "
    "the PRESENT VALUE of the framework's own time variable -- the finite observer's register "
    "size, growing tick by tick since the substrate's first toggle. Today N_hub ≈ 8.3949×10⁶⁰ "
    "ticks. It is NOT an observable (nothing measures \"the register size\" directly) and it is "
    "deliberately NOT a ledger row in `docs/parameters/target_parameters.md` -- it is the "
    "calibration knob upstream of the ledger, not one of the ledger's own predictions.",

    "The tether: N_hub's value is pinned by requiring the framework's derived Fermi constant "
    "G_F = 1/(√2·v_higgs²) to match the measured G_F (PDG/MuLan) to high precision. G_F was chosen "
    "as the tether observable for two reasons: it is measured to extraordinary precision (ppm "
    "level), and its formula (v_higgs from the BZJ cascade, tree-level G_F) is the framework's most "
    "reliable dimensional chain. A direct consequence: **G_F and v_higgs are CALIBRATION "
    "ROUND-TRIPS, not independent predictions** -- their match to observation is BY CONSTRUCTION, "
    "the same way a ruler calibrated against a meter bar will always read \"1 meter\" for that bar. "
    "Both rows are flagged `round-trip` in the ledger table's new `calibration` column (this is a "
    "pre-existing, already-documented epistemic class -- see `target_parameters.md`'s own G_F and "
    "v_higgs row notes -- surfaced here as a first-class manifest column, not a new claim).",

    "The upgrade path: the G_F tether is not permanent. It moves to a higher-precision anchor only "
    "if that anchor comes with a rock-solid (theorem-grade, not calibration-class) framework "
    "formula. The two live candidates are α_EM (currently +1.01σ_PDG, not yet precision-grade) and "
    "the Rydberg / atomic-frame chain (R_∞; currently OPEN/EXTERNAL — blocked on a derived Δα "
    "bridge, per `target_parameters.md` Row P70). Neither is ready; G_F remains the tether.",

    "TIME-AWARENESS (the epoch guardrail, internal research notes): every "
    "dimensionful read in this manifest is a FUNCTION of N, evaluated AT N = N_hub (now). The "
    "frozen locks in `predictions/_value_locks.json` are, without exception, the N_now evaluation "
    "-- they are NOT valid at other epochs. Per-row N-dependence tagging (N-independent / "
    "N-scaling with an exact power / N-through-composition) and the N-parameterized epoch API are "
    "QUEUED as station S1d -- NOT built in this pass. Any station computing at nonzero redshift or "
    "an early epoch MUST use N(z), never N_now -- using a static lock as a stand-in for an "
    "early-epoch value is exactly the one-history-many-clocks trap this guardrail exists to "
    "prevent.",

    # S1d update (2026-07-09, appended at integration; the paragraph above predates the build and
    # its 'QUEUED -- NOT built' clause is superseded by this one):
    "S1d STATUS UPDATE (integrated 2026-07-09): the epoch API IS NOW BUILT -- the paragraph above "
    "predates it. `the_run.py`'s S1d section provides N_NOW(), the per-row N_DEPENDENCE registry "
    "(the N-tag column in this manifest's table), and read_epoch(N, p_era=None) for the natively "
    "N-dependent theorem-grade reads ONLY. THE CALIBRATION FENCE: v_higgs, G_F, and everything "
    "downstream of v (masses, M_Z, widths) are tagged calibration-curve and are structurally "
    "EXCLUDED from read_epoch -- their N-form is the G_F tether's own defining curve, not an "
    "epoch prediction. Era exponents are always explicit arguments (era selection at a given N is "
    "ML-3's open dynamical-crossing question); contracts: proofs/foundations/"
    "S1d_epoch_api_2026-07-09.py (verify-wired).",
]


def print_calibration_section():
    banner("THE CALIBRATION (N_hub) -- read before trusting any dimensionful lock")
    for para in CALIBRATION_SECTION_PARAGRAPHS:
        print(para)
        print()

# suites column: a STATIC citation (not computed) of which derivation_topdown/adapters/ verify.py
# BACKBONE suite (G1 sunada_geometry / G2 furey_stoica_labels / G3a ncg_spectral / G4 aqft_net /
# G5a thermal_time / G6 zeta_gauge -- see verify.py's "adapters" category + adapters/README.md
# ledger) the row's underlying mechanism leans on.  Judgment call, cited per group, printed only
# for Tier-A/B rows (Tier-C rows print '-').
SUITE_BY_LOCK = {
    "d_spatial": ["G1"], "E_count": ["G1"], "V_count": ["G1"], "k_star": ["G1"], "g_girth": ["G1"],
    "p_toggle": ["G1"], "srs_E_at_P": ["G1"], "georgi_jarlskog": ["G1"],
    "sin2_theta_W": ["G1", "G3a"], "g_2": ["G3a"],
    "Q_Koide": ["G2", "G6"], "R3_observer_c3_generation": ["G2"], "delta_Koide": ["G2", "G6"],
    "epsilon_Koide": ["G2", "G6"], "alpha_1": ["G2", "G6"], "alpha_1_full": ["G2", "G6"],
    "lambda_higgs": ["G2", "G6"],
    "delta_r": ["G6"], "delta_rho": ["G6"], "c_vertex_dark": ["G6"],
    "H_0_observer": ["G5a"], "Omega_m_LCDM": ["G5a"], "Omega_Lambda_LCDM": ["G5a"], "Lambda_CC_LCDM": ["G5a"],
}


def _normalize(s):
    return re.sub(r"\s+", " ", s).strip()


def phase5_ledger_tiering(ledger_rows, tier_a_lock_keys, tier_b_rows, locks, unmapped_locks):
    banner("PHASE 5 -- M-4 THE LEDGER-ROW TIER/BIN/STATUS CLASSIFICATION + COVERAGE NUMBERS")
    tier_b_lock_keys = set(tier_b_rows.keys()) if isinstance(tier_b_rows, dict) else set(r["lock_key"] for r in tier_b_rows)
    tier_b_by_key = tier_b_rows if isinstance(tier_b_rows, dict) else {r["lock_key"]: r for r in tier_b_rows}

    out_rows = []
    parse_issues = []
    for row in ledger_rows:
        sym = _normalize(row["symbol"])
        if sym not in SYMBOL_TABLE:
            parse_issues.append(sym)
            lock_key, b = None, "?"
        else:
            lock_key, b = SYMBOL_TABLE[sym]

        if lock_key is not None and lock_key in tier_a_lock_keys:
            tier = "A"
            delta = None
            for ek, (lk, tolclass, why) in TIER_A_MAP.items():
                if lk == lock_key:
                    delta = why
                    break
            read_class = "D-spectrum"
            blocker = None
        elif lock_key is not None and lock_key in tier_b_lock_keys:
            tier = "B"
            read_class = "composite"
            blocker = None
        elif lock_key is not None:
            tier = "C"
            read_class = "blocked"
            blocker = f"engine-surface-missing:{row['file'] or '?'}"
        else:
            tier = "C"
            if sym in STRUCTURAL_EXACT_NOTES:
                # S1b ORPHAN CLEANUP (2026-07-09): forced-structural zero/exact, no DAG node, NOT an
                # orphan awaiting a build (see STRUCTURAL_EXACT_NOTES for the per-row citation).
                # blocker is the STRING "none" (not Python None) -- the blocker_counts print loop
                # below format-specs every blocker as a string; Python None would crash that f-string.
                read_class = "structural-exact"
                blocker = "none"
            elif b in (LMET,):
                read_class = "net-region"
                blocker = BLOCKER_BY_BIN.get(b, "orphan")
            elif b == X:
                read_class = "external"
                blocker = BLOCKER_BY_BIN.get(b, "orphan")
            else:
                read_class = "blocked"
                blocker = BLOCKER_BY_BIN.get(b, "orphan")

        suites = SUITE_BY_LOCK.get(lock_key, []) if tier in ("A", "B") else []
        # THE CALIBRATION (N_hub) column: flag G_F/v_higgs as round-trips (see CALIBRATION_ROUND_TRIP).
        calibration = "round-trip" if lock_key in CALIBRATION_ROUND_TRIP else ""
        out_rows.append(dict(symbol=row["symbol"], predicted=row["predicted"], status=row["status"],
                              lock_key=lock_key, tier=tier, bin=b, read_class=read_class,
                              blocker=blocker, suites=suites, file=row["file"], section=row["section"],
                              subsection=row["subsection"], calibration=calibration))

    # coverage counts
    from collections import Counter
    tier_counts = Counter(r["tier"] for r in out_rows)
    bin_counts = Counter(r["bin"] for r in out_rows)
    status_counts = Counter(r["status"] for r in out_rows)
    tier_bin = Counter((r["tier"], r["bin"]) for r in out_rows)
    tier_status = Counter((r["tier"], r["status"]) for r in out_rows)

    print(f"Total ledger rows parsed: {len(out_rows)}")
    print(f"\nTIER counts: {dict(tier_counts)}")
    print(f"BIN counts:  {dict(bin_counts)}")
    print(f"LEDGER STATUS counts: {dict(status_counts)}")
    print("\nTIER x BIN:")
    for (t, bb), c in sorted(tier_bin.items()):
        print(f"    tier={t:1s}  bin={bb:10s}  n={c}")
    print("\nTIER x STATUS:")
    for (t, st), c in sorted(tier_status.items()):
        print(f"    tier={t:1s}  status={st!r:6s}  n={c}")

    if parse_issues:
        print(f"\nPARSE ISSUES -- {len(parse_issues)} symbols not found in SYMBOL_TABLE (bin '?' assigned honestly):")
        for s in parse_issues:
            print(f"    {s!r}")
    else:
        print("\nNo parse issues -- every parsed ledger row resolved against SYMBOL_TABLE.")

    # Tier-C resistance list, by blocker
    c_rows = [r for r in out_rows if r["tier"] == "C"]
    blocker_counts = Counter(r["blocker"] for r in c_rows)
    print(f"\nTIER-C RESISTANCE ({len(c_rows)} rows) BY BLOCKER:")
    for blk, n in sorted(blocker_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {blk:55s} n={n}")

    calib_rows = [r for r in out_rows if r["calibration"] == "round-trip"]
    print(f"\nCALIBRATION ROUND-TRIPS ({len(calib_rows)}) -- match to observation BY CONSTRUCTION "
          f"(N_hub tethered to measured G_F; see THE CALIBRATION (N_hub) section below):")
    for r in calib_rows:
        print(f"    {r['symbol']} (lock_key={r['lock_key']})")

    structural_exact_rows = [r for r in out_rows if r["read_class"] == "structural-exact"]
    if structural_exact_rows:
        print(f"\nSTRUCTURAL-EXACT ({len(structural_exact_rows)}) -- forced zero/exact, no DAG node, "
              f"NOT orphan (see STRUCTURAL_EXACT_NOTES):")
        for r in structural_exact_rows:
            print(f"    {r['symbol']}: {STRUCTURAL_EXACT_NOTES.get(_normalize(r['symbol']), '')}")

    return out_rows, tier_counts, bin_counts, status_counts, tier_bin, tier_status, blocker_counts, parse_issues


# ==============================================================================================
# S1d (2026-07-09) APPEND-ONLY WIRING: the per-row N-tag column, sourced from the_run.py's new
# N_DEPENDENCE registry (derivation_topdown/bridge/the_run.py, "# ==== S1d EPOCH API ===="
# section; internal research notes). Surfaced ONLY in write_manifest_doc
# (--write mode's "full ledger-row table" + a new summary section below it) -- the FAST mode
# (phase1_engine_reads + phase3_tier_a_compare, main()'s --fast branch) is untouched by this
# addition and stays green with its existing assertions unchanged.
# ==============================================================================================
def _n_tag_string(lock_key):
    """Render R.N_DEPENDENCE[lock_key] as a short display string for the ledger-row table.
    lock_key is None for rows the manifest itself never mapped to a lock (structural-exact /
    blocked Tier-C rows) -- those print '—', matching the table's own blocker/calibration
    columns' convention (never silently invented)."""
    if lock_key is None:
        return "—"
    tag = R.N_DEPENDENCE.get(lock_key)
    if tag is None:
        return "unmapped"      # defensive only -- the S1d contract's EP-2 gates this to zero
    kind = tag[0]
    if kind == "power":
        return f"power(N^{tag[1]})"
    if kind in ("calibration-curve", "composition", "independent"):
        return kind
    return kind


# ==============================================================================================
# PHASE 5b -- write the generated doc
# ==============================================================================================
def write_manifest_doc(out_rows, tier_a_rows, tier_b_rows, unmapped_locks, unplayed, locks_meta,
                        tier_counts, bin_counts, status_counts, tier_bin, tier_status, blocker_counts,
                        parse_issues, n_locks):
    lines = []
    lines.append("# The Reads Manifest — S1 (generated, do not hand-edit)")
    lines.append("")
    lines.append(f"Generated by `derivation_topdown/adapters/reads_manifest.py --write` "
                 f"({time.strftime('%Y-%m-%d %H:%M:%S')}).")
    lines.append(f"Pre-registration (FROZEN, read first): internal research notes.")
    lines.append(f"Lock file snapshot: commit `{locks_meta.get('commit','?')}`, frozen "
                 f"`{locks_meta.get('frozen','?')}`, {n_locks} values.")
    lines.append("")
    lines.append("This is a NEW verification layer: it classifies every ledger row as a read of the "
                 "triple (D, omega, {A(O)}), recomputes what it can from ENGINE PRIMITIVES ONLY "
                 "(the_run.py / the_net.py / srs.py — never predictions/), compares against the "
                 "frozen value locks, and publishes the coverage numbers + the full resistance list. "
                 "The coverage numbers ARE the outcome — no target counts, no flattering.")
    lines.append("")

    lines.append("## THE CALIBRATION (N_hub)")
    lines.append("")
    for para in CALIBRATION_SECTION_PARAGRAPHS:
        lines.append(para)
        lines.append("")

    lines.append("## Coverage numbers")
    lines.append("")
    lines.append(f"Total ledger rows: **{len(out_rows)}**.  Tier counts: "
                 + ", ".join(f"{t}={n}" for t, n in sorted(tier_counts.items())) + ".")
    lines.append("")
    lines.append("| Tier | Bin | n |")
    lines.append("|---|---|---|")
    for (t, b), n in sorted(tier_bin.items()):
        lines.append(f"| {t} | {b} | {n} |")
    lines.append("")
    lines.append("| Tier | Ledger status | n |")
    lines.append("|---|---|---|")
    for (t, st), n in sorted(tier_status.items()):
        lines.append(f"| {t} | {st or '(blank)'} | {n} |")
    lines.append("")
    lines.append("| Tier-C blocker | n |")
    lines.append("|---|---|")
    for blk, n in sorted(blocker_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {blk} | {n} |")
    lines.append("")

    lines.append("## Tier-A comparisons (engine vs lock)")
    lines.append("")
    lines.append("| Engine key | Lock key | Engine value | Lock value | rel Δ | tol | PASS |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in tier_a_rows:
        lines.append(f"| `{r['engine_key']}` | `{r['lock_key']}` | {r['engine_val']:.10g} | "
                     f"{r['lock_val']:.10g} | {r['rel']:+.2e} | {r['tolclass']} | "
                     f"{'✅' if r['passed'] else '❌ MISMATCH'} |")
    lines.append("")

    lines.append("## Tier-B compositions (declared, adoptions named)")
    lines.append("")
    lines.append("| Lock key | Computed | Lock value | rel Δ | PASS | formula | adoptions |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in tier_b_rows:
        lines.append(f"| `{r['lock_key']}` | {r['engine_val']:.10g} | {r['lock_val']:.10g} | "
                     f"{r['rel']:+.2e} | {'✅' if r['passed'] else '❌ MISMATCH'} | {r['formula']} | {r['adoptions']} |")
    lines.append("")

    lines.append(f"## Unmapped locks ({len(unmapped_locks)})")
    lines.append("")
    lines.append("Present in the frozen lock file; no Tier-A/B mapping reproduces them from this "
                 "manifest's narrow engine scope.")
    lines.append("")
    lines.append(", ".join(f"`{k}`" for k in unmapped_locks))
    lines.append("")
    for k in unmapped_locks:
        note = UNMAPPED_LOCK_NOTES.get(k)
        if note:
            lines.append(f"- **`{k}`**: {note}")
    lines.append("")

    lines.append(f"## Unplayed engine notes ({len(unplayed)})")
    lines.append("")
    lines.append("Real engine outputs with no lock match at all — candidate predictions the ledger "
                 "never registered.")
    lines.append("")
    lines.append(", ".join(f"`{k}`" for k in unplayed))
    lines.append("")

    lines.append("## The full ledger-row table")
    lines.append("")
    lines.append("| Symbol | Predicted | Tier | read-class | bin | status | suites | blocker | calibration | N-tag (S1d) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in out_rows:
        pred = (r["predicted"] or "").replace("|", "\\|")[:60]
        suites = ",".join(r["suites"]) if r["suites"] else "—"
        blk = r["blocker"] or "—"
        calib = r["calibration"] or "—"
        ntag = _n_tag_string(r["lock_key"])
        lines.append(f"| {r['symbol']} | {pred} | {r['tier']} | {r['read_class']} | {r['bin']} | "
                     f"{r['status']} | {suites} | {blk} | {calib} | {ntag} |")
    lines.append("")

    # S1d (2026-07-09) APPEND-ONLY: the N-dependence class-count summary (a new section; the
    # table above already carries the per-row tag). Source of truth: the_run.py's N_DEPENDENCE.
    lines.append("## N-dependence tag counts (S1d)")
    lines.append("")
    lines.append("Per-row N-dependence tags (last column of the table above) are sourced from "
                 "`derivation_topdown/bridge/the_run.py`'s `N_DEPENDENCE` registry (station S1d, "
                 "internal research notes) — a STATIC dict keyed by this "
                 "manifest's own lock keys (Tier-A map ∪ Tier-B compositions), so the join is exact. "
                 "Class counts across the full registry:")
    lines.append("")
    from collections import Counter as _Counter
    _tag_counts = _Counter(t[0] for t in R.N_DEPENDENCE.values())
    lines.append("| class | n | meaning |")
    lines.append("|---|---|---|")
    _meaning = {
        "independent": "pure structure, N-independent (angles, ratios, counts, exact forms)",
        "calibration-curve": "the fenced v_higgs/G_F tether family — NEVER exposed by read_epoch()",
        "power": "native N-power law — the rows read_epoch() exposes",
        "composition": "Tier-B row whose parents span more than one class",
    }
    for cls, n in sorted(_tag_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {cls} | {n} | {_meaning.get(cls, '')} |")
    lines.append("")
    lines.append(f"Total: **{len(R.N_DEPENDENCE)}** rows tagged, 0 untagged (gated by the S1d contract's "
                 "EP-2, `proofs/foundations/S1d_epoch_api_2026-07-09.py`).")
    lines.append("")

    if parse_issues:
        lines.append(f"## Parse issues ({len(parse_issues)})")
        lines.append("")
        lines.append("Symbols parsed from the ledger that were not found in `SYMBOL_TABLE` (bin `?` "
                     "assigned honestly, not silently):")
        lines.append("")
        for s in parse_issues:
            lines.append(f"- `{s}`")
        lines.append("")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWROTE {OUT_PATH} ({len(lines)} lines).")


# ==============================================================================================
# MAIN
# ==============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write docs/parameters/reads_manifest.md")
    ap.add_argument("--fast", action="store_true", help="M-5: PHASE 0 (parse) + PHASE 3 (Tier-A) only")
    args = ap.parse_args()

    t0 = time.time()
    banner("S1 READS MANIFEST -- pre-reg internal research notes")

    # PHASE 0 -- M-0 PARSE
    ledger_rows = parse_ledger(LEDGER_PATH)
    locks, locks_meta = load_locks(LOCKS_PATH)
    n_rows, n_locks = len(ledger_rows), len(locks)
    print(f"M-0 PARSE: ledger rows = {n_rows} (>= 120? {n_rows >= 120})   "
          f"locks = {n_locks} (== {N_LOCKS_EXPECTED}? {n_locks == N_LOCKS_EXPECTED})")
    per_section = {}
    for r in ledger_rows:
        per_section.setdefault(r["section"], 0)
        per_section[r["section"]] += 1
    print("Per-section row counts:")
    for sec, c in per_section.items():
        print(f"    {sec!r:60s} n={c}")
    parse_ok = n_rows >= 120 and n_locks == N_LOCKS_EXPECTED

    if args.fast:
        # M-5 FAST MODE: PHASE 1 (engine reads, needed for the comparisons) + PHASE 3 only.
        E = phase1_engine_reads()
        tier_a_rows = phase3_tier_a_compare(E, locks)
        n_mismatch = sum(1 for r in tier_a_rows if not r["passed"])
        elapsed = time.time() - t0
        print(f"\n--fast elapsed: {elapsed:.1f}s")
        ok = parse_ok and n_mismatch == 0
        print(f"M-5 FAST RESULT: parse_ok={parse_ok}  tier_a_mismatches={n_mismatch}  => exit {0 if ok else 1}")
        sys.exit(0 if ok else 1)

    # FULL RUN -- PHASE 1..5
    E = phase1_engine_reads()
    print_engine_inventory(E)
    tier_b_compositions, unmapped_locks, unplayed = phase2_mapping(E, locks)
    tier_a_rows = phase3_tier_a_compare(E, locks)
    tier_b_rows = phase4_tier_b_compare(E, locks)
    bonus_mass_ratios(E, locks)

    tier_a_lock_keys = set(lk for lk, _, _ in TIER_A_MAP.values())
    (out_rows, tier_counts, bin_counts, status_counts, tier_bin, tier_status, blocker_counts,
     parse_issues) = phase5_ledger_tiering(ledger_rows, tier_a_lock_keys, tier_b_rows, locks, unmapped_locks)

    print_calibration_section()

    if args.write:
        write_manifest_doc(out_rows, tier_a_rows, tier_b_rows, unmapped_locks, unplayed, locks_meta,
                            tier_counts, bin_counts, status_counts, tier_bin, tier_status,
                            blocker_counts, parse_issues, n_locks)

    elapsed = time.time() - t0
    banner(f"S1 READS MANIFEST -- DONE in {elapsed:.1f}s "
           f"(tier A {sum(r['passed'] for r in tier_a_rows)}/{len(tier_a_rows)}, "
           f"tier B {sum(r['passed'] for r in tier_b_rows)}/{len(tier_b_rows)})")
    sys.exit(0 if parse_ok else 1)


if __name__ == "__main__":
    main()
