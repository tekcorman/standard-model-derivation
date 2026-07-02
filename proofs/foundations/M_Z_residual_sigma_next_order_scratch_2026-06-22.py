#!/usr/bin/env python3
"""
proofs/foundations/M_Z_residual_sigma_next_order_scratch_2026-06-22.py   [_scratch]

QUESTION (firewall, constructive): does the Σ = α₁/h dark-correction read
framework — the one that closes the heavy quarks at the Perron eigenvalue
(m_b: −α₁/2; m_t: −α₁/4) — account for M_Z's remaining +0.018% (+7.76σ_PDG),
the same way it closed m_b/m_t?

This is a DIAGNOSIS scratch.  It modifies nothing.  It reads the live
predictions/ DAG and the committed proofs, then tests each candidate
next-order Σ term against the residual.  Honest verdict at the bottom.

ESTABLISHED reads (cited, not re-derived):
  - Σ = α₁/h  : resolvent read of 1/h at the channel eigenvalue, ×girth
                amplitude α₁=(2/3)^8.  Closes heavy quarks at h_P=2.
  - δ_r       : predictions/delta_r.py = c_S·α₁/(1−α₁), c_S=1/(2|E|)=1/12,
                the B_NB(srs) Perron-residue gauge-SINGLET projection
                (unified_oblique_one_resolvent_2026-05-16.py).
  - the M_Z tree→pole oblique IS a real SM effect (Δr family) — confirmed
                INTRINSIC with EXACT PDG inputs:
                M_Z_residual_is_tree_vs_pole_oblique_2026-05-15.py.
"""
from __future__ import annotations
from fractions import Fraction

# ── forced reads ────────────────────────────────────────────────────────
k, g, V = 3, 10, 4
two_E   = V * k                         # = 12 = 2|E| (handshake 2|E|=N·k*)
alpha1  = float(Fraction(2, 3) ** 8)    # girth amplitude (k-1)/k ^(g-2)
h_P     = 2.0                           # Perron eigenvalue = k*-1
c_S     = 1.0 / two_E                   # = 1/12 gauge-singlet projection

# ── live numbers (from predictions/M_Z.py run + the exact-PDG oblique probe) ──
M_Z_obs       = 91.1876
sigma         = 0.0021
M_Z_tree_res  = 0.0035742713696660894   # framework-input tree-over-pole residual
delta_r       = c_S * alpha1 / (1 - alpha1)
pole_res      = 0.0001786195923646016   # remaining AFTER δ_r  (+0.0179%, the target)
pole_sigma    = pole_res * M_Z_obs / sigma
full_oblique  = 0.003925                # SM tree-vs-pole with EXACT PDG inputs
fw_tree_res   = 0.003573                # framework-input tree residual

print("=" * 78)
print("  M_Z residual vs the Σ=α₁/h framework — constructive diagnosis")
print("=" * 78)

# ─────────────────────────────────────────────────────────────────────────
print("\n── Q1.  Is δ_r the Σ=α₁/h read? ──────────────────────────────────")
Sigma_hP = alpha1 / h_P                  # the m_b read at the Perron eigenvalue
print(f"  Σ = α₁/h_P = α₁/2          = {Sigma_hP*100:+.4f}%   (the m_b dark read)")
print(f"  δ_r = c_S·α₁/(1−α₁)        = {delta_r*100:+.4f}%   (the applied M_Z oblique)")
print(f"  ratio Σ/δ_r               = {Sigma_hP/delta_r:.3f}")
print("  VERDICT: δ_r is NOT the Σ=α₁/h read.  Both read the SAME resolvent")
print("  G_NB=(I−uB)⁻¹ at the SAME α₁, but they are DIFFERENT residues:")
print("    • Σ=α₁/h_P uses the bare pole residue 1/h (=1/2 at Perron) — a")
print("      SINGLE insertion, no resummation. Right for a Yukawa anchor.")
print("    • δ_r uses the gauge-SINGLET projection weight c_S=1/(2|E|)=1/12")
print("      of the Perron residue, ×the FULLY-RESUMMED geometric ladder")
print("      1/(1−α₁).  The 1/12 is the Z-vertex projection onto the uniform")
print("      Perron eigenvector; the 1/2 is the bare 1/h. Different objects.")
print("  So δ_r and the m_b/m_t Σ-read are SIBLING resolvent reads (one B),")
print("  not the same read at a different eigenvalue.")

# ─────────────────────────────────────────────────────────────────────────
print("\n── Q2.  What is the remaining +0.018%? ───────────────────────────")
print(f"  target = pole residual after δ_r = {pole_res*100:+.4f}%  ({pole_sigma:+.2f}σ)")
print()
print("  (a) next-order in the δ_r channel:  ALREADY RESUMMED.")
print(f"      δ_r = c_S·α₁·(1+α₁+α₁²+…); the /(1−α₁) sums ALL orders.")
print(f"      The α₁² sub-term ({c_S*alpha1*alpha1*100:+.5f}%) is INSIDE δ_r already —")
print(f"      there is no leftover next order in this channel.")
print()
two_insert = (delta_r) ** 2
print(f"  (b) a second independent insertion (δ_r)²  = {two_insert*100:+.6f}%")
print(f"      → 16× too small to be the +{pole_res*100:.4f}% residual.")
print()
print(f"  (c) Σ=α₁/h_P as a SECOND oblique = {Sigma_hP*100:+.4f}%")
print(f"      → 100× too BIG. Not the residual.")
print()
print("  (d) THE ACTUAL DECOMPOSITION (this is the honest answer):")
print(f"      full SM tree→pole oblique (EXACT PDG inputs)  = {full_oblique*100:+.4f}%")
print(f"      δ_r captures                                  = {delta_r*100:+.4f}%  ({delta_r/full_oblique*100:.0f}% of it)")
missing_oblique = full_oblique - delta_r
fw_input_err    = fw_tree_res - full_oblique     # framework g_2 is 0.038% LOW ⇒ negative
print(f"      missing oblique (true−δ_r)                    = {missing_oblique*100:+.4f}%  ({(1-delta_r/full_oblique)*100:.0f}% of the oblique δ_r misses)")
print(f"      framework INPUT error (fw−exactPDG, g_2 0.038% low) = {fw_input_err*100:+.4f}%")
print(f"      SUM (missing oblique + input error)           = {(missing_oblique+fw_input_err)*100:+.4f}%")
print(f"      observed pole residual                        = {pole_res*100:+.4f}%   ✓ matches")
print()
print("  ⇒ the +0.018% is NOT a single clean sub-leading oblique.  It is the")
print("    SUM of (i) the ~14% of the SM oblique that δ_r's leading Perron-")
print("    singlet term does not capture (+0.054%) and (ii) an UPSTREAM")
print("    framework input error (−0.035%, the framework g_2 running 0.038%")
print("    low).  They partially CANCEL, leaving +0.018%.  Mixed origin.")

# ─────────────────────────────────────────────────────────────────────────
print("\n── Q3.  Does Σ=α₁/h close the +0.018% without tuning? ─────────────")
print("  Tempting near-miss:  pole_res/α₁ =", f"{pole_res/alpha1:.5f}", "≈ 1/216 = 1/6³.")
cand = alpha1 / 216
cur_pole = M_Z_obs * (1 + pole_res)
new_pole = cur_pole * (1 - cand)
print(f"    a 2nd oblique = α₁/216 → M_Z {cur_pole:.4f} → {new_pole:.4f}  ({(new_pole-M_Z_obs)/sigma:+.2f}σ)")
print("    216 = 6³ = |E|³, and 1/216 = c_S/18.  BUT this is REVERSE-")
print("    ENGINEERED from the residual, and the target itself (Q2.d) is a")
print("    MIXED quantity (oblique + input error) — so any single clean")
print("    coefficient that hits +0.018% is fitting the CANCELLATION, not a")
print("    forced second residue.  No resolvent channel forces 1/216 here:")
print("    the second Perron sub-residue is the (b) α₁²-ladder term (already")
print("    in δ_r), and the h_P channel CANCELS in the neutral-Z singlet (it")
print("    is the W/custodial piece → δρ, by construction of δ_r).")
print("  → HONEST: Σ=α₁/h does NOT have a forced, untuned next term that")
print("    closes the +0.018%.  The residual is not even purely an oblique.")

# ─────────────────────────────────────────────────────────────────────────
print("\n── Q4.  m_W consequence ──────────────────────────────────────────")
mW_pred, mW_obs, mW_sig = 80.40104706812734, 80.3692, 0.0133
mW_rel = (mW_pred - mW_obs) / mW_obs
mW_if_MZ_exact = mW_pred * (M_Z_obs / (M_Z_obs * (1 + pole_res)))
print(f"  m_W (live)              = {mW_pred:.4f}  ({(mW_pred-mW_obs)/mW_sig:+.2f}σ),  rel {mW_rel*100:+.4f}%")
print(f"  M_Z rel residual passes through linearly (m_W ∝ M_Z):  {pole_res*100:+.4f}%")
print(f"  if M_Z were exact, m_W → {mW_if_MZ_exact:.4f}  ({(mW_if_MZ_exact-mW_obs)/mW_sig:+.2f}σ)")
own = mW_rel - pole_res
print(f"  decomposition of m_W's {mW_rel*100:+.4f}%:")
print(f"    M_Z-inherited  = {pole_res*100:+.4f}%  ({pole_res/mW_rel*100:.0f}%)")
print(f"    m_W-OWN (cosθ_W/δρ sector) = {own*100:+.4f}%  ({own/mW_rel*100:.0f}%)")
print("  ⇒ m_W is only ~45% M_Z-inherited.  Closing M_Z moves m_W +2.39σ →")
print("    +1.31σ (helps, NOT a full close); the other ~55% is m_W's own")
print("    sin²θ_W/δρ residual.  So 'if M_Z closes, m_W follows' is only")
print("    HALF true.")

print("\n" + "=" * 78)
print("OVERALL VERDICT")
print("=" * 78)
print("""  Σ=α₁/h (and its next order) does NOT close M_Z's +7.76σ the way it
  closed m_b/m_t — and the reason is structural, not a missing term:

  1. δ_r IS the framework's Σ-style oblique read (Perron-singlet residue,
     c_S=1/12, fully resummed). It is a SIBLING of the m_b/m_t α₁/h read
     from the ONE resolvent G_NB — but a DIFFERENT residue (1/12 singlet
     projection + resummation, not the bare 1/h). It already captures 86%
     of the true SM tree→pole oblique.

  2. The +0.018% that remains is NOT a clean sub-leading oblique. It is a
     MIXTURE: ~+0.054% of genuine SM oblique that δ_r's leading term misses,
     PARTIALLY CANCELLED by a ~−0.035% UPSTREAM framework input error (g_2
     runs 0.038% low). There is no single forced Σ term that targets a
     cancellation of two different-origin pieces.

  3. The honest location of the residual: PART oblique-higher-order (a real
     sub-leading Δr piece the framework's single leading Perron-singlet term
     does not resum), PART upstream √(α_2+(3/5)α_1) running precision (g_2).
     M_Z is 2.3 ppm — a +0.018% structural prediction is the framework's
     intrinsic precision floor (NOT a free parameter, NOT a missing read).

  4. m_W: only ~45% M_Z-inherited. Closing M_Z → +1.31σ (not 0). The rest
     is m_W's own cosθ_W/δρ sector.

  No forcing, no fit.  Σ=α₁/h is the right FAMILY (and δ_r is its M_Z member),
  but the +7.76σ does not close to ≤1σ_PDG from a next-order Σ term: the
  residual is a mixed oblique-floor + input-precision effect at the 2.3-ppm
  PDG floor, not a single forced correction.""")
print("=" * 78)
