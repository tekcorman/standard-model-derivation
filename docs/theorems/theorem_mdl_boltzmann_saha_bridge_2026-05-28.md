# MDL → Boltzmann/Saha bridge — theorem

**Date:** 2026-05-28.
**Status:** THEOREM-GRADE-STRUCTURAL-CONDITIONAL. The Boltzmann distribution and the freeze-out log-transcendence are derived from A2-T (MDL) composed with `theorem_observer_energy_functional.md` (E_obs = κ·S) and Jaynes/Shannon maximum-entropy, all standard published math (Type 3) or upstream theorem-grade (Type 4). CONDITIONAL on two interpretive identifications, named in §6: (I1) the A2-T-retained microstate ensemble is the maximum-entropy ensemble under the finite observer's single retained macro-constraint ⟨E_obs⟩; (I2) the observer-energy functional's E=κ·S relation applies to a thermal configuration's description length (not only to the temporal accumulated-surprise of its original construction).
**Scope:** Derives (a) the Boltzmann/Gibbs realization weight p_i ∝ exp(−E_i/k_BT) for substrate configurations, and (b) the Phase-III freeze-out "log-transcendence" T_fo = E_bind / N_thermal with N_thermal = −log p = the MDL description length of the rare configuration. **Purpose: replace the reverted "Axiom F"** — the freeze-out log was once postulated; here it is derived.
**Out of scope:** the Saha PREFACTOR (2π)^{3/2} phase-space normalization (the π is irreducible per `../../proofs/cosmology/session_A_substrate_partition_function_2026-05-27.py`; it lives INSIDE ln(prefactor) and is a separate object); precision BBN/recombination numerics (gated on the nucleon sector, `../../proofs/cosmology/nucleon_sector_BBN_gate_scoping_2026-05-28.py`); calibrating the Landauer reference temperature T to a specific value (inherited from the OEF theorem's same out-of-scope clause).
**Probe:** `../../proofs/cosmology/A2T_mdl_to_boltzmann_saha_bridge_2026-05-28.py` (numerics: code length affine in energy to machine precision; freeze-out log = description length; recombination sanity check).

**Post-2026-05-08 axiom slate note.** A2-T (cited throughout) is a derived theorem of (A) self-containment + (B) finite observer + standard math + (I) active reading, per `theorem_A2_mdl_from_finite_register.md`. References to "A2-T" remain semantically valid; see `../framework/framework_axioms.md` §10.

---

## 1. Theorem statement

**Theorem (MDL → Boltzmann/Saha).** Let an observer under A2-T (MDL waterline; `theorem_A2_mdl_from_finite_register.md`) retain a set of substrate configurations {i}, each with a minimal description length L_i (bits) under the observer's optimal code, and an associated observer energy E_i = κ·S_i (κ = k_B T ln2, S_i = L_i·ln2 nats = the configuration's self-information) per `theorem_observer_energy_functional.md`. Suppose the finite observer (B) retains a single macroscopic constraint on this ensemble: the mean energy ⟨E⟩ = U. Then:

**(Boltzmann)** The MDL-canonical (maximum-entropy / least-committal) realized distribution over the retained configurations is the Gibbs distribution
$$p_i = \frac{e^{-\beta E_i}}{Z},\qquad Z = \sum_i e^{-\beta E_i},$$
with β the Lagrange multiplier dual to ⟨E⟩=U. Equivalently the optimal code length is **affine in energy**, $L_i = (\beta/\ln 2)\,E_i + \log_2 Z$, and the realized weight is the Kraft–McMillan probability $p_i = 2^{-L_i}$.

**(Temperature identification)** β = 1/(k_B T) with T the OEF Landauer reference temperature; equivalently $p_i = 2^{-S_i} = e^{-E_i/k_BT}$, consistent with E_i = κ S_i and κ = k_B T ln2.

**(Freeze-out log-transcendence corollary)** For a bound configuration of binding energy E_bind in equilibrium with an η-diluted background, the bound↔free balance gives
$$\beta\,E_{\text{bind}} = \log\!\big(\text{prefactor}\cdot\eta^{-1}\big) =: N_{\text{thermal}},\qquad T_{\text{fo}} = \frac{E_{\text{bind}}}{N_{\text{thermal}}}.$$
Here N_thermal = −log p_rare is **identically the MDL description length** of the rare bound configuration relative to the diluted background. The freeze-out "log-transcendence" is the description length — derived, not postulated.

---

## 2. Axioms and cited upstream

**Framework axioms / derived theorems:**
- **A2-T** (`theorem_A2_mdl_from_finite_register.md`; `../framework/framework_axioms.md` §3) — MDL waterline with selective retention. Its stated foundation is Shannon's Rate-Distortion theorem (A-IT5) + Rissanen 1978/1983 MDL.
- **B** (finite observer; `../framework/framework_axioms.md` §1) — supplies the "single retained macro-constraint" clause (a finite observer cannot retain the full microstate; it retains coarse aggregates, here the mean energy).

**Type 4 upstream (theorem-grade framework files):**
- `theorem_observer_energy_functional.md` — E_obs = κ·S_total, κ = k_B T ln2 (Landauer), S = −log₂ P (Shannon self-information). Provides the energy↔information relation E = κ·S. (I2 applies it to a configuration's description length.)

**Type 3 citations (standard published math):**
- **Shannon, C. E.** (1948). Source coding: the optimal prefix code assigns length L_i = −log₂ p_i; Kraft–McMillan inequality Σ 2^{−L_i} ≤ 1 (equality for a complete code) ⇒ realized weight p_i = 2^{−L_i}.
- **Jaynes, E. T.** (1957). *Information theory and statistical mechanics.* Phys. Rev. 106, 620. Maximum-entropy distribution under a mean-value constraint ⟨E⟩=U is the Gibbs distribution p_i ∝ exp(−βE_i).
- **Rissanen, J.** (1978/1983) — MDL; the A2-T foundation. The least-committal model = the one minimizing expected description length of the residual unknown = the MaxEnt distribution.

No fabricated citations. No post-hoc fitting.

---

## 3. Setup — the retained ensemble and its energy

By A2-T the observer retains every above-waterline representation; `../framework/framework_axioms.md` §3 states all retained representations are "physically realized simultaneously," and `../orientation.md` §(residue register) states soft-gated retained alternatives "carry non-zero Boltzmann-style weight." This theorem makes that weight exact.

Each retained configuration i has a minimal description length L_i (bits) under the observer's optimal code (A2-T + Shannon source coding). Its self-information is S_i = L_i ln2 (nats). By `theorem_observer_energy_functional.md`, the observer assigns it energy E_i = κ S_i with κ = k_B T ln2 (Landauer-scaled information; I2).

The finite observer (B) cannot track the full joint microstate; it retains the mean energy ⟨E⟩ = Σ p_i E_i = U as its single macroscopic handle on the ensemble (I1).

---

## 4. Proof

**(Boltzmann.)** Among all distributions {p_i} on the retained configurations with Σ p_i = 1 and Σ p_i E_i = U, the MDL-canonical (least-committal) choice is the one maximizing Shannon entropy H(p) = −Σ p_i ln p_i — because the max-entropy distribution is precisely the one requiring the shortest expected description of the residual unknown given the constraint (Rissanen/Jaynes duality; A2-T's MDL character applied at the ensemble level, I1). By Jaynes 1957 (Lagrange multipliers β for ⟨E⟩, α for normalization),
$$\frac{\partial}{\partial p_i}\Big[-\sum_j p_j\ln p_j - \beta\big(\textstyle\sum_j p_j E_j - U\big) - \alpha\big(\textstyle\sum_j p_j - 1\big)\Big] = 0$$
$$\Rightarrow\quad -\ln p_i - 1 - \beta E_i - \alpha = 0 \quad\Rightarrow\quad p_i = e^{-\beta E_i}/Z,\ \ Z = \sum_j e^{-\beta E_j}.$$
Taking −log₂: $L_i = -\log_2 p_i = (\beta/\ln 2) E_i + \log_2 Z$ — **affine in E_i** (Type 2). The probe verifies this to machine precision (residual ~1e-15) on a toy spectrum, with slope = β/ln2 and intercept = log₂Z.

**(Temperature identification.)** The realized weight p_i = 2^{−L_i} (Kraft–McMillan, Type 3) must equal the Gibbs weight: 2^{−L_i} = 2^{−S_i/ln2}... more directly, p_i = e^{−βE_i} and E_i = κS_i = k_B T ln2 · S_i, while p_i = e^{−S_i} (since S_i = −ln p_i is the nat self-information). Consistency forces βκ = 1, i.e. **β = 1/(k_B T)** with T the OEF Landauer temperature. The "temperature" is the bits-per-energy conversion rate already fixed by the OEF theorem; it is not a new constant.

**(Freeze-out corollary.)** A bound configuration's equilibrium abundance ratio to the η-diluted free/photon background is, by the Boltzmann result, ∝ exp(βE_bind)/(multiplicity·η^{-1}). Freeze-out (abundance balance, or rate ~ H) occurs when this ratio is O(1):
$$\beta E_{\text{bind}} = \log(\text{prefactor}\cdot\eta^{-1}) = N_{\text{thermal}}.$$
But N_thermal = −log p_rare is, by definition, the self-information = MDL description length of the rare bound state against the diluted background. Hence T_fo = E_bind / N_thermal with N_thermal a quantity NATIVE to A2-T. ∎

The probe's recombination sanity check: E_bind=13.6 eV, T_recomb≈0.32 eV ⇒ N_thermal=42.5, decomposing as ln(1/η)≈21 (baryon-dilution description length) + ln(prefactor)≈21 — reproducing the standard Saha recombination log, now read as a description length.

---

## 5. Relation to the reverted "Axiom F"

A prior session implemented the Phase-III freeze-out log T_fo = E_bind/log(prefactor·η^{-1}) as an **axiom** (Axiom F) and reverted it (axiom-elimination ethos). This theorem **discharges that axiom**: the log is the MDL description length (§4 corollary), and the Boltzmann factor it inverts is the MaxEnt-canonical realization weight of the A2-T-retained ensemble (§4 Boltzmann), with energy from the OEF theorem. No new axiom is added; the freeze-out structure becomes a consequence of A2-T + OEF + Jaynes/Shannon.

---

## 6. What is and is NOT closed (honest residue)

**Closed (given the two interpretive identifications below):** the exp(−E/T) Boltzmann form, the affine-in-energy code length, and the freeze-out T_fo = E_bind/N_thermal log-transcendence.

**The two conditional identifications (named, not hidden):**
- **(I1)** the A2-T-retained ensemble is MaxEnt-distributed under the single retained constraint ⟨E_obs⟩=U. Grounded in A2-T's own MDL character (least-committal = MaxEnt, Rissanen/Jaynes duality) + the finite-observer clause (B), but it is an interpretation of A2-T at the ensemble level, not pure algebra.
- **(I2)** the OEF relation E=κ·S applies to a thermal configuration's description length, not only to the temporal accumulated-surprise of the OEF theorem's original construction. Grounded in the Landauer universality of E=κS, but an application beyond OEF's literal scope.

**NOT closed (separate objects):**
- The Saha PREFACTOR (2π)^{3/2}: the π is irreducible (Session A) and sits INSIDE ln(prefactor); this theorem derives the OUTER exp/log structure, not the prefactor.
- Precision BBN/recombination numerics: gated on the nucleon sector (Q_np, g_A; Stream 3).

---

## 7. Downstream use

Stream 3's BBN reaction network (once the nucleon sector is built) may use this MDL-grounded Boltzmann factor for its detailed-balance / Saha abundance ratios, rather than importing it from statistical mechanics. The η-diluted freeze-out logs throughout Phase III (recombination, deuterium bottleneck, e+e− annihilation) are all instances of the §4 corollary.
