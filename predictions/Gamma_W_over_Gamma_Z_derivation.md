# Γ_W/Γ_Z — Derivation (F4 S3, 2026-07-02)

## 1. Abstract

We assemble the ratio of the W and Z total decay widths from the framework's own derived
electroweak endpoint values and its own fermion content, through the standard tree-level
golden-rule width formulas dressed by the QCD correction series. The result, Γ_W/Γ_Z = 0.83460,
agrees with PDG 2024 (0.83560 ± 0.01685) at −0.06σ. This is the framework's **first width
observable**. The derivation is non-trivial for three reasons: (i) every numerical input is a
framework read (no PDG constant feeds the computation; G_F is unused); (ii) a CAS-proved lemma
guarantees the ratio is exactly immune to the framework's known dark sector, so the comparison is
unambiguous; (iii) the assembly was **frozen (pre-registered) before comparison**, so no term
selection against the data was possible. The grade is capped below theorem-grade by Clause 9: the
width formula's 1/(48π) is a continuum loop factor (Type-3 SM import).

## 2. Framework results invoked

- **Electroweak endpoints** (each with its own derivation file): sin²θ_W(M_Z) = 0.23125
  (`sin2_theta_W_MZ.py`), α_s(M_Z) = 0.1179 (`alpha_s.py`), m_W = 80.4010 GeV (`m_W.py`),
  M_Z = 91.2039 GeV (`M_Z.py`), m_t = 172.41 GeV (`m_t.py`). All THEOREM-GRADE-CONDITIONAL
  (α_GUT = 1/24 + M_unif + 1-loop RG chain).
- **Fermion content read** (Cl(6)-Fock, `the_run.py read_species`, structural): for node
  occupation n ∈ {0,…,k*}: Q = (−1)ⁿ·n/k*, T₃ = (−1)ⁿ/2, multiplicity N(n) = C(k*,n)
  (n = 0: ν, 1: d, 2: u, 3: e; C(3,1) = C(3,2) = 3 = the color triplet). k* = 3 (`k_star.py`),
  n_gen = 3 (`R3_observer_c3_generation.py`).
- **The dark-cancellation lemma** (CAS, `proofs/foundations/F4_S2b_width_ratio_dark_lemma
  _2026-07-02.py`): a species-common REAL multiplicative dressing cancels identically in width
  ratios; the gauge sector's matching-point dark reads the exactly-real Perron channel
  (Ihara–Bass λ ∈ {1,2} at μ = 3); a complex-pole shell dressing is stability-excluded
  (over-applies ×1.6·10¹⁶ against Γ_μ/m_μ; contradicts Γ_e = 0). **Therefore no dark correction
  may be applied to this ratio — forbidden, not omitted.**

## 3. Derivation

**Step 1 (Type-3 structure, declared).** Tree-level partial widths for massless final states
(Peskin & Schroeder ch. 20; PDG EW review):

$$\Gamma(W\to f\bar f') = \frac{g^2\,m_W}{48\pi}\,N_c, \qquad
\Gamma(Z\to f\bar f) = \frac{g^2/c^2\,M_Z}{48\pi}\,N_c\,(v_f^2+a_f^2),$$

with $v_f = T_3 - 2Qs^2$, $a_f = T_3$. QCD dressing of quark channels
$1 + a_s/\pi + 1.409\,(a_s/\pi)^2$ (Chetyrkin–Kühn–Kwiatkowski 1996). These formulas are the
**declared standard-QFT import** — their $1/(48\pi)$ is transcendental over K
(Lindemann 1882), so per Clause 9(9b) the row is bridge-conditional; the native derivation of the
per-channel phase space is the open equation (`incomplete_equations_todo.md` §7; the
band-geometric route was closed by computation — `F4_cone_spectral_function_2026-07-02.py`).

**Step 2 (open channels from the framework's own spectrum).** W: 3 lepton doublets + quark
doublets with CKM row unitarity; the top row is closed since the framework's own
m_t = 172.41 > m_W. Hence $N_W = n_{gen} + N_c\,(n_{gen}-1) = 3 + 3\cdot 2 = 9$. Z: all fermions
except top (m_t > M_Z/2):

$$\Sigma_Z(s^2) = \sum_{f\ \mathrm{open}} N_c\,(v_f^2+a_f^2) = 7.3009 \ \ \text{at}\ s^2 = 0.23125 .$$

**Step 3 (the ratio; algebra).** g² cancels between numerator and denominator:

$$\frac{\Gamma_W}{\Gamma_Z} = \frac{N_W\,c^2}{\Sigma_Z(s^2)}\cdot\frac{m_W}{M_Z}\cdot
\frac{1+\tfrac{6}{9}\,\delta_{QCD}}{1+f_{had}^Z\,\delta_{QCD}},\qquad
\delta_{QCD} = \frac{a_s}{\pi}+1.409\Big(\frac{a_s}{\pi}\Big)^2 .$$

The QCD factors nearly cancel ($6/9 = 0.667$ vs $f^Z_{had} = 0.691$). The mass ratio is the
framework's own $80.4010/91.2039 = 0.881553$.

**Step 4 (numerics).** $9\times 0.76875/7.3009 \times 0.881553 \times 0.99906 = 0.83460$.

## 4. Result

$$\boxed{\;\Gamma_W/\Gamma_Z = 0.83460\;}$$

## 5. Comparison with experiment

PDG 2024: Γ_W = 2.085 ± 0.042 GeV, Γ_Z = 2.4952 ± 0.0023 GeV ⟹ observed 0.83560 ± 0.01685.
Deviation: −0.00100 absolute, **−0.12% relative, −0.06σ** (Clause 8c numerical PASS). Grade:
**MATHEMATICALLY COMPLETE / bridge-conditional (Clause 9b)** — capped by the Type-3 π-import,
not by any numerical or input gap.

## 6. Open questions

1. **The native phase space** (the honest gap): 1/(48π) per channel is imported. The
   band-geometric derivation is CLOSED (kill branch: the substrate cones are chirally warped
   spin-1 multifolds with q²-dark pair channels — probe above); the open route is the Clifford
   γ-trace layer (Cl(6)⊗Cl(0,2) vertex forms). Until derived, this row cannot rise above
   bridge-conditional.
2. Stated-not-applied refinements (all ≪ ±2% measurement, largely cancelling in the ratio):
   QED FSR, finite m_b/m_c/m_τ phase space, EW ρ_f/vertex layer, QCD 3rd order.
   *(The EW layer is now REGISTERED — §7; its "largely cancelling" estimate was wrong on size.)*
3. The companion Γ_Z/M_Z carries the +4.8σ_exp EW-radiative-layer residual — see its derivation
   file; the ratio here is insensitive to that layer at the current measurement precision.
   *(Resolved — §7 and the companion's §7.)*

---

## 7. ADDENDUM (2026-07-02, user gate) — the EW-layer DIFFERENTIAL registered

The LOOP program's derived EW layer (`ew_width_layer.py`; chain C2 2188fbe → V1 a5287f4 → V2
d37a679; companion derivation file §7) applies per width; in this ratio the common
normalization cancels and the DIFFERENTIAL survives:

$$\frac{\Gamma_W}{\Gamma_Z}\Big|_{\rm reg} = 0.83460\times\frac{1+\delta_W}{1+\delta_Z}
= 0.83460\times\frac{1-0.000787}{1-0.004864} = \boxed{0.83802}.$$

Deviation: **+0.29%, +0.14σ_PDG** (pre-layer −0.06σ) — Clause 8c PASS, comfortably inside the
±2.0% measurement. **Honesty note (recorded in the V2 probe and the loop-kickoff banner):** the
pre-registration's size estimate for this differential ("≲0.1%, largely common") was WRONG —
the actual differential is +0.41%, because the κ̂ (effective-angle) shift and the Z→bb̄ vertex
have no W analog. The pre-registered falsification CRITERION (the ratio stays sub-σ) held; the
estimate is corrected here, not relabeled. Grade unchanged: MATHEMATICALLY COMPLETE /
bridge-conditional (Clause 9b — now carrying the EW layer in the same Type-3 class as 1/(48π)).
