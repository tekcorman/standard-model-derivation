# Standard Model from First Principles

!!! quote ""
    **The same substrate object, read 12 different ways, matches PDG observations on all 12 — with zero fitted constants.**

The 12 observables are 7 quark-sector (y_t, y_b, V_us, V_cb, V_ub, δ_r, δρ), 4 lepton/PMNS (y_τ, θ_12, θ_13, θ_23), and the A_s cosmological prefactor — all read from the **non-backtracking resolvent** $G_{NB} = (I - u \cdot B_{NB}(\mathrm{srs}))^{-1}$ with one argument $a = (2/3)^8$ and zero fitted constants. **Then 79 more parameters fall out of the same substrate.**

## In one sentence

Three meta-commitments — self-containment of the universe, finite observer, active reading of binary distinctions — plus standard published mathematics, force a substrate (the **srs crystal net**) whose spectral content is the Standard Model. One empirical labeling rule (A5-mass: which substrate eigenvalues are which observed masses) attaches contact with experiment. There are no further inputs.

## Where to go next

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } __Read the story__

    ---

    A narrative arc from *what can exist?* to the Standard Model. Best for first-time readers.

    [:octicons-arrow-right-24: Start the story](story/index.md)

-   :material-test-tube:{ .lg .middle } __Run the verifier__

    ---

    `python3 verify.py` — 25 backbone proofs in ~10 seconds. If any fail, the framework is wrong, full stop.

    [:octicons-arrow-right-24: Repository on GitHub](https://github.com/tekcorman/standard-model-derivation)

-   :material-target:{ .lg .middle } __What would falsify it__

    ---

    Specific numerical predictions that, if measured against, refute the framework.

    [:octicons-arrow-right-24: Falsification criteria](falsification.md)

-   :material-checkbox-multiple-marked-circle:{ .lg .middle } __The 12-way over-determination__

    ---

    One resolvent. Twelve readouts. Zero fitted constants. All match PDG.

    [:octicons-arrow-right-24: See the cross-validation](over-determination.md)

</div>

## Status (2026-05-26)

Across **123 tracked targets**: **91 ✅ closed**, **9 🟡 in progress**, **13 ❌ open or out-of-scope**, **10 ⚙️ structural definitional**. The `predictions/` directory of the GitHub repo is the source of truth; this site is the narrative + visual layer for human readers.

For the rigorous research material:

- [docs/honest_assessment.md](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/honest_assessment.md) — what's proven, what's adopted, what's open, what would falsify
- [docs/parameters/target_parameters.md](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/parameters/target_parameters.md) — every tracked parameter with current grade
- [docs/master_plan.md](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/master_plan.md) — current frontier
- [docs/framework/framework_axioms.md](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/framework/framework_axioms.md) — canonical foundation statement

See the [Reference](reference.md) page for the full pointer map.

---

!!! note "On this site"
    This is the **explainer**: the narrative + visual layer for readers who want the story, not the journal-grade rigor. The corresponding rigorous research material lives in the [`docs/`](https://github.com/tekcorman/standard-model-derivation/tree/main/docs) tree of the GitHub repo. Both serve different readers; neither replaces the other.

    The site is auto-deployed from the [`explainer/`](https://github.com/tekcorman/standard-model-derivation/tree/main/explainer) directory of the repo on every push. Animations and interactive visualizations land progressively.
