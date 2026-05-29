#!/usr/bin/env python3
"""
Need-D-3 — Stage 3: attempt the V_Ram ≅ Cl(6)-Fock identification (the γ₇↔
           walker-type rule). HONEST OUTCOME — reduced, NOT closed.

Stage 2 reduced north_star condition 2 to one probe-grade step: lift the §4(C)
γ₇↔walker-type rule to theorem-grade. Stage 3 attempts it. The honest result:
the step reduces to the framework's own acknowledged DEEPEST piece — §4(D)'s
MDL-waterline derivation of the walk length L(n) — which Stage 3 does NOT
close. It strips Need-D-3 down to exactly that one bounded computation.

PRE-DECLARED GATES (G2–G6 honest-record; the verdict is "reduced, not closed"):
  G1  Reduce the target: γ₇↔h-root  ≡  γ₇↔(L=0 vs L=g).
  G2  The over-determination is the TEST, not the derivation.
  G3  The triplet residual: the candidate L(n) and the §4(C)/§5 sketch —
      honest about its grade.
  G4  The mechanism gap: §4(D)'s MDL-waterline L(n), "the deepest piece".
  G5  The conditionality is now clean and unconditional.
  G6  Honest status of Need-D-3 after Stages 1–3.
  G7  Verdict.
"""

from fractions import Fraction as F

results = []


def gate(name, passed, detail=""):
    results.append((name, bool(passed)))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if detail:
        for line in detail.strip("\n").split("\n"):
            print(f"         {line}")
    print()


k_star, g = 3, 10
q_NB = F(k_star - 1, k_star)


# ======================================================================
print("=" * 72)
print("G1 — reduce the target: γ₇↔h-root  ≡  γ₇↔(L=0 vs L=g)")
print("=" * 72)
# Gamma trivial lambda=3: Ihara-Bass h^2 - 3h + 2 = 0 -> h in {1,2}.
ib_roots = [h for h in range(-3, 4) if h*h - 3*h + 2 == 0]
# at L=0 both roots give h^0 = 1 -> the h-choice is DEGENERATE at L=0.
degenerate_at_L0 = (1**0 == 1 and 2**0 == 1)
g1 = (sorted(ib_roots) == [1, 2] and degenerate_at_L0)
gate("G1 the γ₇↔h-root rule reduces to γ₇↔(L=0 / L=g)", g1,
     f"Γ trivial λ=3 Ihara-Bass roots: h ∈ {sorted(ib_roots)}\n"
     f"at L=0:  h^0 = 1 for BOTH h=1 and h=2  ⇒  the IB-root choice is\n"
     f"DEGENERATE at L=0 (W40 finding; walker theorem §4.2).\n"
     "So 'h=1 saturation vs h=2 Perron' is not an independent choice — it IS\n"
     "the choice L=0 (saturation; roots degenerate, y=h⁰=1) vs L=g (the Perron\n"
     "h=2 walker runs the full girth). Stage 3's real target: derive L(n) for\n"
     "the colour triplet — n=2 (u) → L=0, n=1 (d) → L=g.")


# ======================================================================
print("=" * 72)
print("G2 — the over-determination is the TEST, not the derivation")
print("=" * 72)
g2 = True   # honest-record
gate("G2 §8/W55 tests a fixed L(n); it does not derive it", g2,
     "W55 showed the Type III entry (y_τ, L=g−2) over-determines with §8's\n"
     "survival amplitude a = q_NB^(g−2) — that is the north_star DIAGNOSTIC\n"
     "(one object, masses and couplings forced to agree). The triplet entries\n"
     "y_t (L=0) and y_b (L=g) would be tested the SAME way once L(n) is fixed.\n"
     "But the over-determination is an ACCEPTANCE TEST — it confirms a correct\n"
     "L(n); it does not by itself derive which L attaches to which n. The\n"
     "derivation is a separate, structural task (G3/G4).")


# ======================================================================
print("=" * 72)
print("G3 — the triplet residual: candidate L(n) + the §4(C)/§5 sketch")
print("=" * 72)
# candidate, colour-triplet sector n ∈ {1,2}, n_max = 2:
#   L = g·(n_max − n)   →   d (n=1) → L=g ,  u (n=2) → L=0
L_candidate = {n: g * (2 - n) for n in (1, 2)}
matches = (L_candidate[1] == g and L_candidate[2] == 0)
g3 = True   # honest-record — reports the candidate AND its honest grade
gate("G3 candidate L = g·(n_max − n) for the triplet — SKETCH grade", g3,
     f"colour-triplet sector n ∈ {{1,2}}, n_max = 2:\n"
     f"  L(d, n=1) = g·(2−1) = {L_candidate[1]}   → Type IV (Perron, full girth)\n"
     f"  L(u, n=2) = g·(2−2) = {L_candidate[2]}    → Type II (saturation)\n"
     f"matches the §4(C) split: {matches}\n"
     "Reading: the MAXIMAL Hamming weight in the triplet sector saturates\n"
     "(L=0); this is exactly walker theorem §5's words — 'n=2 maximal edge\n"
     "occupation → all girth-cycle modes MDL-retained → saturation'.\n"
     "** HONEST GRADE: this is a 2-point formula + the §5 word-sketch. It is\n"
     "   NOT a derivation. The rigorous derivation is G4. **")


# ======================================================================
print("=" * 72)
print("G4 — the mechanism gap: §4(D)'s MDL-waterline L(n)")
print("=" * 72)
g4 = True   # honest-record
gate("G4 the rigorous L(n) is §4(D) — 'the deepest piece, still sketch'", g4,
     "The mechanical content: n occupied Cl(6)-Fock edge-modes at the\n"
     "trivalent vertex → a shift of the A2-T MDL waterline threshold → the\n"
     "retained non-backtracking walk length L. The walker theorem\n"
     "(theorem_walker_length_MDL_waterline §9) lists this — '§4(D) Hamming\n"
     "weight → walker length L via MDL waterline' — explicitly as 'the\n"
     "DEEPEST piece', under 'Successor pieces (still SKETCH / open)'.\n"
     "The TOOL exists: A2-T (the MDL waterline) is a derived theorem. The\n"
     "open task is the computation — the waterline shift as a function of n.\n"
     "Stage 3 does NOT perform it. This is the genuine residual.")


# ======================================================================
print("=" * 72)
print("G5 — the conditionality is now clean (and unconditional)")
print("=" * 72)
g5 = True
gate("G5 the residual is an unconditional derivation, not a conditional", g5,
     "§4(D)'s species map was graded 'conditional on Need-D-3'. Stage 1 showed\n"
     "that 'Need-D-3' was the dead CKM Y_u/Y_d wall (or, read as 'conditional\n"
     "on the selection map', circular). Either way the conditional is void.\n"
     "The genuine residual is therefore UNCONDITIONAL: derive L(n) from the\n"
     "MDL waterline. Not a wall, not multi-sprint, not a CKM problem — one\n"
     "bounded MDL-waterline computation.")


# ======================================================================
print("=" * 72)
print("G6 — honest status of Need-D-3 after Stages 1–3")
print("=" * 72)
status = {
    "Stage 1": "the '9+ attacks / Path B / multi-sprint wall' was the CKM "
        "Y_u/Y_d eigenbasis problem — dead (W49 symmetric-phase; §8/W55 "
        "over-determination). The selection map was never behind it.",
    "Stage 2": "the selection map is a FORCED unique bijection (ν→I, d→IV, "
        "u→II, e→III); 3 of 4 entries theorem-grade; the open piece is the "
        "d/u split's grade.",
    "Stage 3": "that split reduces to §4(D)'s MDL-waterline L(n) — 'the "
        "deepest piece'. Stage 3 gives the candidate L=g·(n_max−n) (maximal "
        "Hamming weight saturates) at sketch grade, and the over-determination "
        "as its acceptance test — but does NOT derive it.",
    "net": "Need-D-3 is no longer a wall, a CKM problem, or multi-sprint. It "
        "is ONE precisely-stated, bounded, unconditional derivation: L(n) from "
        "the A2-T MDL waterline. Everything else around it is closed or "
        "theorem-grade. That single derivation is the genuine open frontier.",
}
g6 = True
gate("G6 Need-D-3 = one bounded MDL derivation; everything else cleared", g6,
     "\n".join(f"{k}: {v}" for k, v in status.items()))


# ======================================================================
print("=" * 72)
print("G7 — verdict")
print("=" * 72)
verdict = (
    "Stage 3 — REDUCED, NOT CLOSED. The γ₇↔walker-type rule reduces to "
    "γ₇↔L(n) (G1); the over-determination is its acceptance test, not its "
    "derivation (G2); the candidate L=g·(n_max−n) sits at sketch grade (G3); "
    "the rigorous derivation is §4(D)'s MDL-waterline L(n), the framework's "
    "acknowledged deepest piece (G4), and Stage 3 does not perform it. The "
    "disciplined path delivered Stages 1–2 (cleared a 6-day-old wall, forced "
    "the bijection) and Stage 3 reaches the irreducible core — a single "
    "bounded MDL computation — and hands it off honestly. north_star "
    "condition 2 is NOT met; it is reduced to that one derivation."
)
g7 = g1
gate("G7 verdict — Stage 3 reduces Need-D-3 to one MDL derivation; not closed",
     g7, verdict)


# ======================================================================
print("=" * 72)
n_pass = sum(p for _, p in results)
print(f"STAGE 3 SENTINEL: {n_pass}/{len(results)} gates "
      f"(G2–G6 honest-record; the VERDICT is 'reduced, not closed')")
print("=" * 72)
print("""
Need-D-3 Stage 3 — HONEST: reduced, not closed.

The γ₇↔walker-type rule = the choice L=0 (saturation) vs L=g (Perron), since
the two Ihara-Bass roots are degenerate at L=0. So the residual is L(n) for
the colour triplet. The candidate — L = g·(n_max−n), the maximal-Hamming-
weight species saturates — matches the §4(C)/§5 sketch, but at SKETCH grade
(2 points + words). The rigorous derivation is §4(D)'s MDL-waterline L(n),
the framework's own 'deepest piece, still sketch'. Stage 3 does not perform it.

NET of the disciplined path: Need-D-3 is no longer a Tier-3 multi-sprint wall
or a CKM problem. Stages 1–2 cleared the dead wall and forced the selection
map; Stage 3 strips the residual to ONE bounded, unconditional computation —
L(n) from the A2-T MDL waterline. That is the genuine open frontier, and it is
where a focused derivation session should now go.
""")
if n_pass != len(results):
    raise SystemExit(1)
