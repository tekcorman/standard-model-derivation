# Reference — pointers into the rigorous research tree

This site is the narrative + visual layer. The rigorous research material — every derivation, every audit, every theorem — lives in the [`docs/`](https://github.com/tekcorman/standard-model-derivation/tree/main/docs) tree of the GitHub repo.

## Where to look for X

| Question | Look at |
|---|---|
| What does this framework claim? | [`README.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/README.md) |
| Show me one result in 5 minutes | [`docs/quickstart.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/quickstart.md) |
| What's actually proven? | [`docs/honest_assessment.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/honest_assessment.md) |
| Status of every parameter | [`docs/parameters/target_parameters.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/parameters/target_parameters.md) |
| Numerical predictions vs PDG | `predicted_parameters.md` at repo root (auto-generated; gitignored — run `python3 run_predictions.py` to refresh) |
| What's the current frontier? | [`docs/master_plan.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/master_plan.md) |
| What are the axioms? | [`docs/framework/framework_axioms.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/framework/framework_axioms.md) |
| What's the layered architecture? | [`docs/framework/framework_architecture.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/framework/framework_architecture.md) |
| What's the conceptual story? | [`docs/framework/narrative_spine.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/framework/narrative_spine.md) |
| The finish-line goal | [`docs/north_star.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/north_star.md) |
| Repository layout + conventions | [`docs/orientation.md`](https://github.com/tekcorman/standard-model-derivation/blob/main/docs/orientation.md) |
| Per-parameter derivations | [`predictions/*.py`](https://github.com/tekcorman/standard-model-derivation/tree/main/predictions) + paired `*_derivation.md` |
| Closed theorem statements | [`docs/theorems/`](https://github.com/tekcorman/standard-model-derivation/tree/main/docs/theorems) (~92 files) |
| Audit registers (live) | [`docs/audits/registers/`](https://github.com/tekcorman/standard-model-derivation/tree/main/docs/audits/registers) — uniqueness ledger, residue register, adoption register |
| Retracted derivations | [`predictions/retracted/`](https://github.com/tekcorman/standard-model-derivation/tree/main/predictions/retracted) — honest history of failed re-audits |

## Run the framework yourself

```bash
git clone https://github.com/tekcorman/standard-model-derivation.git
cd standard-model-derivation
pip install numpy scipy sympy matplotlib

python3 verify.py                       # 25 backbone proofs in ~10 seconds
python3 run_predictions.py              # regenerate predicted_parameters.md at repo root
python3 scripts/validate_citations.py   # citation discipline check
```

If `verify.py` fails on any backbone proof, the framework is wrong, full stop.

## Glossary (selected)

| Term | Meaning |
|---|---|
| **srs** | The Laves / (10,3)-a crystal net; the MDL-optimal substrate; forced unique by Sunada 2012 (R-9 closure 2026-05-12). Space group I4₁32, $k_* = 3$, girth $g = 10$. |
| **srs-z** | The bipartite double cover of srs. Carries the chirality grading; hosts the mass operator. |
| **K₄** | The complete graph on 4 vertices; the smallest quotient of srs carrying the same substrate content. |
| **F_inv(E)** | The free involutive monoid on alphabet $E$; the substrate as an algebra. Cayley graph of $F_{\mathrm{inv}}(E)$ is the substrate as a graph. |
| **Hashimoto operator $B_{NB}$** | The non-backtracking walk operator on directed edges of a graph. Carries the substrate's recurrence content. |
| **h** | The Ramanujan eigenvalue $h = (\sqrt{3} + i\sqrt{5})/2$ at the $P$ point of the BCC Brillouin zone of srs. Saturates the Alon–Boppana bound. |
| **MDL waterline** | The framework's reading of Minimum Description Length: every compression that *pays for itself* (positive savings $L_{\mathrm{total}} < L_{\mathrm{raw}}$) is retained, not just the strict minimum. |
| **A5-mass** | The framework's empirical labeling: which Bloch-Hashimoto eigenvalues correspond to which Standard Model masses. The only empirical anchor. |
| **R-9 closure** | The proof (2026-05-12) that srs is forced uniquely as the substrate-net via Sunada 2012 + (A)'s no-privilege principle. |
| **12-observable §8 family** | 12 distinct observables read from the same resolvent $G_{NB}$; the framework's strongest over-determination claim. |
| **M_persistence** | The 12×12 fermion mass operator on srs-z (shipped 2026-05-26); produces all 12 fermion mass eigenvalues + the $m_{\nu_1} = 0$ kernel. |
| **Family-D** | The Higgs-quartic-vertex dark correction $\delta\lambda / \lambda = -4 \alpha_1^{\text{bare 2}}$; propagates into $m_H = 125.20$ GeV. |
| **K-rationality bright-line** | Parameter-linter Clause 9: SM-import constants must lie in $K = \mathbb{Q}(\sqrt{2}, \sqrt{3}, \sqrt{5})$ or be flagged as non-derivable. |
