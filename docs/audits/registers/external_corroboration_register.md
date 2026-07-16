# External Corroboration Register — append-only (opened 2026-07-10, station CB-2)

**Standing rule (binding).** Rows in this register are **NON-LOAD-BEARING**. They support or strain the
*interpretation* of claims made in this repo; they NEVER move a number, a lock, a verify gate, or the status
of any derivation. A TENSION row obligates a raw finding in `docs/incomplete_equations_todo.md` where stated
below; it never obligates (or permits) a change to the claim's own proof.

**Format rules.** Append-only: a landed row is never edited — append a superseding row instead. Every result
row must cite the producing repo, commit hash, and script path. Status vocabulary: `CORROBORATES` /
`TENSION` / `NULL` / `PENDING`.

**Provenance.** Created per the CB track roadmap (internal research notes,
station CB-2). The prior inline pattern ("**Independent corroboration:** …" lines, e.g.
`docs/parameters/derivations.md:53` gyroid photonics) remains valid for literature corroborations; this
register exists for *experimental results produced by sibling programs* (currently: the the upstream engine PB track,
`~/projects/the upstream engine/core/PHYSICS_BRIDGE_SIDE_QUEST_2026_07_09.md`).

---

## §1 FROZEN INTAKE MAP — committed BEFORE any source experiment has run

The verdict spaces below are the the upstream engine PB track's own pre-registered verdicts. The bookings are frozen NOW so
that no outcome can be reinterpreted after the fact. Any outcome not listed books as `NULL` + a note.

### 1a. Source: the upstream engine PB-2 — the lift criticality sweep
Target claim: **M0-2R T2** (`proofs/foundations/M0_2R_T2_T3_arrow_criticality_currency_2026-07-07.py`) —
arrow = sub-criticality u < u_c = 1/(k−1) of the branching non-backtracking path gas. The *proof* is
srs-specific and machine-checked; what PB-2 probes is the **genericity clause** (that this is the stability
law of register-growth processes as a class).

| PB-2 verdict | Booking (frozen 2026-07-10) |
|---|---|
| TRANSITION-AT-PREDICTED (transition located in measured u_eff within the pre-registered window around 1/(k̄−1), k̄ measured before unblinding) | `CORROBORATES` — scope line mandatory: *system-level corroboration of T2's genericity in a non-physics substrate; not a physics-parameter claim; non-load-bearing.* |
| TRANSITION-ELSEWHERE | `TENSION` + raw finding in `docs/incomplete_equations_todo.md`: the genericity clause of T2 is challenged; the srs-instance theorem is untouched. |
| NO-TRANSITION | `NULL` — the lift admission process is not a realization of a branching path gas; no inference either way. |
| HARNESS-MISMATCH (declared by the the upstream engine prereg's own criteria) | `NULL` — same as above, booked with the mismatch reason. |
| PILOT-ONLY (any run on the prohibited legacy lift_v0 gate) | **Not bookable.** No row. |

### 1b. Source: the upstream engine PB-5 — the srs-convergence experiment (learning-native MDL objective)
Target claim: **`proofs/foundations/dl_comparison.py`** — MDL selects srs uniquely (13.02 bits; nearest 3D
rival ths +0.83) *in the crystallographic vocabulary* (space group / Wyckoff terms). What PB-5 probes is
**frame-independence of the substrate selection** — the objection "the minimum is an artifact of the chosen
encoding vocabulary" cannot be answered from inside this repo, because every internal re-derivation shares
the frame.

| PB-5 verdict | Booking (frozen 2026-07-10) |
|---|---|
| SRS-EMERGES (unique minimum under the learning-native objective, crystallographic vocabulary banned, k priced not fixed) | `CORROBORATES` — the selection survives a foreign vocabulary built by a different program. Record the learning-frame margin in bits (the crystal-frame margin was +0.83; the new margin is new information). |
| OTHER-WINS | `TENSION` — bounds the *interpretation* "the physics substrate is the learning-optimal memory." Explicit scope line: does NOT touch the crystallographic selection theorem, which stands on its own frozen objective. Winner and margin recorded. |
| DEGENERATE (ties within 1 bit) | `NULL` — the learning-native objective lacks the resolution to select; recorded with the tie set. |

---

## §2 RESULTS (append below this line; newest last)

*(no rows yet — both source experiments PENDING as of 2026-07-10)*
