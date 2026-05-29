# Retracted gauge derivation scripts

**Retracted 2026-05-04 EOD+1** — superseded by the M_unif-anchored Tier 1 cluster predictions in `predictions/`.

## Files

- `g_1_derivation.py`
- `g_2_derivation.py`
- `g_3_derivation.py`

## Why retracted

These scripts used inputs that are now retracted or stale:

- **sin²θ_W = 3/13** — RETRACTED formula (per master plan §2.1). The framework's theorem-grade derivation gives sin²θ_W(M_unif) = 3/8 (`predictions/sin2_theta_W.py`, GQW trace argument, Class C theorem-grade).
- **α_GUT = 1/24.1** — should be theorem-grade 1/24 (`predictions/alpha_GUT.py`).
- **External M_GUT = 2×10¹⁶ GeV** — replaced by framework-derived M_unif = (32/k*^(g-1))·M_Pl (Row P62, THEOREM-GRADE-CONDITIONAL post-Stage-5).

The g_3 derivation specifically gave α_s(M_Z) ≈ 0.155 (+31% off PDG) due to these stale inputs. The current `predictions/alpha_s.py` gives 0.121 (+2.8% from PDG; ~+3.7σ_PDG).

## Replacements

| Stale | Replacement |
|---|---|
| `g_1_derivation.py` | `predictions/g_1.py` + `predictions/g_1_derivation.md` |
| `g_2_derivation.py` | `predictions/g_2.py` + `predictions/g_2_derivation.md` |
| `g_3_derivation.py` | `predictions/g_3.py` + `predictions/g_3_derivation.md` |

All replacements ship at THEOREM-GRADE-CONDITIONAL inheriting from M_unif (Row P62) and M_Z (Row P64). See ledger Rows P65-P70.

## Historical record

The scripts are preserved here for record. Do not import from this directory — the imports are now broken (the `_mssm_rge.py` helper still exists at `proofs/gauge/_mssm_rge.py` but the conventions used by these scripts are stale).
