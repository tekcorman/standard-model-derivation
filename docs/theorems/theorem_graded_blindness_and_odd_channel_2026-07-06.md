# Graded-blindness theorem + the odd-channel classification — theorem (O0 + O1)

**Date:** 2026-07-06 (a model, following internal research notes).
**Status:** **THEOREM-GRADE-STRUCTURAL** — a structural identity on the srs objects, machine-verified. It has **no numerical target** and is compared to nothing (the −70 ppm number ε never appears). This is CONSOLIDATION: it proves that four separately-banked "chirality is blind" results are one theorem, and it isolates the unique un-probed functional class. It moves **no value**.
**Verification:** `../../proofs/foundations/ODD_O0_graded_blindness_theorem_2026-07-06.py` (ALL PASS). No pre-registration required (no blind comparison — this is a structural identity, not a probe of ε; the pre-registration discipline is reserved for Station O2, which does compare to ε).
**Scope:** the charged-lepton −70 ppm (§1 of `../incomplete_equations_todo.md`), the neutrino on-cut subleading (§8), and the whole route-exhaustion history. Explains WHY ~15 routes walled; it does **not** compute ε (that is Station O2).

---

## 1. Theorem statement (graded blindness)

**Setup (the object, verbatim from `derivation_topdown/bridge/the_run.py:199-214`).** The 4D master object is the graded sum

$$D_4 \;=\; A + B, \qquad A := D_3 \otimes \mathbf 1, \qquad B := \gamma_t \otimes \partial_N,$$

where `D_3` = the srs Hodge–Dirac operator on `0-forms ⊕ 1-forms` (10×10 at each Bloch point, Hermitian), `γ_t` = the form-degree grading `diag(+1 on 0-forms, −1 on 1-forms)`, and `∂_N` = the run (`read_run`). The Hodge–Dirac operator is form-**odd**, so

$$\{D_3,\ \gamma_t\} = 0 \quad(\text{exact, verified: anti} < 10^{-11}) \;\Longrightarrow\; \{A, B\} = 0 \;\Longrightarrow\; \boxed{D_4^2 = A^2 + B^2 = D_3^2 + \partial_N^2}$$

— the "clean split" with **no cross term**, for **every** `∂_N` (the identity is structural, verified on three independent random Hermitian run operators to machine precision, not an accident of one `∂_N`).

**The bit σ.** Let σ be the Z₂ operation `A ↦ A, B ↦ −B` — the relative sign/phase of the two graded pieces. It is realized physically as the arrow/run reversal `∂_N → −∂_N` (equivalently `γ_t → −γ_t`); by the one-bit theorem (`theorem`/`TID2_D_chirality_bit_2026-07-02`) the enantiomer J-flip carries `{J, γ_t, γ⁵, handedness}` coherently, so σ **is** the chiral/enantiomer bit. Then

$$\sigma:\ D_4 = A+B \ \longmapsto\ A - B, \qquad (A-B)^2 = A^2 + B^2 = D_4^2 \quad(\text{since } \{A,B\}=0).$$

**Theorem (graded blindness).**
1. **(EVEN clause)** `D_4^2` is **exactly σ-invariant**. Hence every functional that factors through `D_4^2` — the spectrum `{λ²}`, the moduli `|λ|`, the heat coefficients `a_k` / `ζ(0)`, resolvent traces `Tr(D_4^2 − z)^{-1}`, and the eigenprojectors/Berry connections of `D_4^2` — is σ-invariant, i.e. **bit-even = chirality-blind**, by the object's own clean split.
2. **(ODD clause)** Functionals **linear in `D_4`**, `Tr(D_4\, g(D_4^2))` (the η-invariant, the odd heat trace `Tr(D_4 e^{-tD_4^2})`, the spectral flow of `s ↦ D_4(s)`), are σ-**odd**. Their chiral content is exactly the **B-term**

$$\operatorname{Tr}\!\big(D_4\, g(D_4^2)\big) = \underbrace{\operatorname{Tr}\!\big(A\, g(D_4^2)\big)}_{\text{σ-even (vector)}} + \underbrace{\operatorname{Tr}\!\big(B\, g(D_4^2)\big)}_{\text{σ-odd (chiral)}}, \qquad \text{chiral part} = \operatorname{Tr}\!\big((\gamma_t\otimes\partial_N)\, g(D_4^2)\big).$$

The chiral carrier `Tr(B g(D_4^2))` is **real** and generically **nonzero** (verified: −0.234 at t=0.7 for the test run) — the channel is **live**, and it is the **unique** σ-odd functional linear in `D_4`.

**Proof.** `(A−B)^2 = A^2 − AB − BA + B^2 = A^2 + B^2` because `AB + BA = 0`. So `σ(D_4^2) = D_4^2`; any function of `D_4^2` (equivalently of its spectrum/eigenspaces) is therefore σ-invariant — the EVEN clause. For the ODD clause, under σ: `Tr(D_4 g(D_4^2)) → Tr((A−B) g(D_4^2))`, and since `g(D_4^2)` is σ-invariant, the A-term is fixed and the B-term flips sign; the σ-odd projection `½[Tr(D_4 g) − Tr(σD_4 g)] = Tr(B g(D_4^2))`. ∎

---

## 2. The four scattered walls are corollaries of this one theorem

| # | banked wall | probe | it is the theorem's… |
|---|---|---|---|
| C1 | **Q3 conjugation** — every isotype-multiplicity correction is chirality-blind (μ_ω = μ_ω̄) | `OMEGA_S2_Q3_isotype_allocation_2026-07-02.py` | EVEN clause: multiplicities are functions of `spec(D²)`-graded sectors ⟹ σ-invariant |
| C2 | **E2c bit-parity** — the mass read's 1st-order invariant δ is bit-EVEN; the iJ channel feeds only χ (mass-2nd-order) | `LOOP_E2c_read_projection_2026-07-02.py` | the EVEN/ODD split of `Tr(D_4 g)`: δ is a modulus datum `|c_j|` (even), χ the phase-sum (odd) |
| C3 | **W2 seed** ⟨0\|U_π²\|0⟩ = i/2: Re = 0 democratic, Im = ½ all-chirality | `LOOP_A5_winding_weld_W2_2026-07-04.py` | the theorem on a single matrix element: Re = σ-even part (blind), Im = σ-odd part (chiral) |
| C4 | **Perron-null** — perron_frame(+J) == perron_frame(−J); the democratic sector carries zero chiral holonomy | `LOOP_A5_magnitude_relative_berry_2026-07-05.py` | the lemma *σ-odd operator ⟹ zero expectation in any σ-invariant sector* (proved in O0.4) |

**The C4 lemma (pure linear algebra, proved in the probe).** For a unitary involution `S` (`S²=1`), a σ-invariant projector `P` (`SPS = P`), and a σ-odd operator `O` (`SOS = −O`): `Tr(PO) = Tr(SPOS) = Tr(P·SOS) = −Tr(PO) ⟹ Tr(PO) = 0`. The Perron/democratic sector is σ-invariant (it is the bit-even eigenspace — C3's Re-seed = 0 democracy); its chiral (σ-odd) holonomy is therefore exactly zero.

**Why this ties the bow on the route-exhaustion history.** The ~15 documented dead routes to the −70 ppm (`../incomplete_equations_todo.md` §1) — non-adiabatic transport (×3), band/modulus curvature, resolvent/cycle trace, joint cover, enantiomer twist, scale/N_hub, cosmic cascade, degenerate-PT, closed-loop Berry, eigenvalue-rate, eigenvector-Berry (shell + relative-Perron), off-diagonal/non-abelian geometric, a₂-additive — are **all** either (i) even functionals (factor through `D²`: spectra, moduli, a₂/a₄, resolvents, Berry-of-`D²`) or (ii) `D²`-eigenstate projections (state-block reads, weld-gated). By the theorem they are **blind to ε by the object's own clean split** — the walls were not mysterious, they were structural. The a₂-additive route's same-day S3 correction ("a₂ moves only the SOFT direction; the hard crux is a chiral PHASE") is the EVEN clause caught in the act: an `a₂` coefficient is an even functional and *cannot* carry ε.

---

## 3. The odd-channel classification (O1) — the unique un-probed class

The `DN_CHIRAL_A_route_reaudit_2026-07-02.py` classification of chiral routes, completed:

| class | chiral-sensitive? | continuous? | needs weld descent? | status |
|---|---|---|---|---|
| **R1** conjugation-symmetric (even) | no (σ-even) | — | — | DEAD by the EVEN clause (exact zero in the pinned direction) |
| **R2** topological-odd (quantized) | yes | **no** (Chern ∓2, Berry −π, Z₂) | no | DEAD: quantization no-go, carries no continuous ~60 ppm |
| **R3** dynamical-odd, state-projected | yes | yes | **yes** (which coupled state to transport = ADOPTED-WINDING-WELD) | over-applies ×10³–10⁶ and/or weld-gated |
| **R4** **continuous-odd spectral TRACE** (η / spectral flow / odd heat trace) | **yes** (σ-odd) | **yes** | **no** (a trace is basis-free) | **UN-PROBED** |

**R4 is the unique remaining class**: it is the only functional that is simultaneously σ-odd (escapes R1), continuous (escapes R2's quantization no-go), and a **trace** — so it needs **no** winding-weld descent (the ADOPTED-WINDING-WELD seam gates state *projections*, not traces; the number dissolves the weld obstruction even though the adoption itself stands as a labeling map, per §3/T2 of the handoff).

**Inventory (verified 2026-07-06, a model).** No repo probe computes a continuous odd spectral invariant. The 19 probes that touch odd objects all compute **quantized** ones (`supertrace Str e^{−tD₃²} = χ = −2` per fiber in `OMEGA_T1`; the Γ-triple Cherns ∓2; the Perron-winding Z₂ holonomy −π). The continuous odd invariant is **named but never run**: `b4_a4_dirac_index_probe.py:344` lists it as its explicit sequel — *"η-invariant (spectral asymmetry, can be nonzero without zero modes)"* — and `forward_construction_substrate_atiyah_singer.md` has sat unconnected since April 2026. (grep witnesses in the probe output; the "spectral flow" hits in `oblique_S_U_kappa` = the Γ→P Perron mode-tracking, an even object; `dark_eps_cp_spectral` "spectral asymmetry" = a naming of the ratio 1/5, not a computed η.)

---

## 4. Consequence + the seam back to the master object

- **Gate A's number-face is re-posed** (label only; the gate does not move, nothing closes, the −70 ppm stays OPEN): not "build the a₂ mass-sector coefficient" (an even functional — S3 proved it cannot carry ε) but **build the ODD sector of the D₄ spectral action** — `Tr((γ_t⊗∂_N) g(D_4^2))` / η / the spectral flow of the run family. This is the ONE functional class the theorem does not render blind.
- **Integration seam** (`the_run.py` READ 2″/3″): the odd sector is a **new read of the same object**, sitting alongside `ζ_{D₄}(0)` (which is `a₄` = an even coefficient). It requires **no new operator** — `γ_t`, `∂_N`, and `D_4` are all already in `the_run.py`. The neutrino on-cut subleading (§8) integrates through the same seam. Any flag/wording change in `the_run.py` is USER-gated.
- **Sideways hardening** (structure, not a claim): the §4 dark-sign reading "mass = recurrence rate ⟹ DOWN" is a spectral-asymmetry sign selection; the one-bit theorem makes η *the natural continuous function of the one bit*. A computed odd-channel asymmetry with a definite sign would give the rate-reading independent structural support — but that is downstream of O2 and is **not** claimed here.

**Next station:** O2 — the relative-η probe, `η(full run family) − η(constant-φ leading family)`, its own pre-registration committed **before** the probe, blind against ε = −1.7515×10⁻⁷ ± 3.9×10⁻¹⁰ rad, poisons binding (2α₁⁵, 2α₁³, no inserted power, no bracket interpolation). A KILL (the odd trace is quantized-or-zero on this family) closes the entire R4 class and is equally decisive. See internal research notes §2.
