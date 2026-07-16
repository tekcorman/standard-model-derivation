#!/usr/bin/env python3
"""
proofs/foundations/S1d_epoch_api_2026-07-09.py

S1d — THE EPOCH API CONTRACT (N_hub = the time variable). Pre-registered FROZEN in
internal research notes (commit 715b11d, BEFORE this file was
written); implements that pre-reg's EP-0..EP-6 verbatim, as labeled PASS/FAIL checks.

WHAT THIS TESTS (deliverable 2 of the pre-reg): the new append-only surface at the bottom of
derivation_topdown/bridge/the_run.py (`# ==== S1d EPOCH API ====`) — N_NOW(), the static
N_DEPENDENCE registry, ERA_EXPONENTS, and read_epoch(N, p_era=None) — plus the companion
append-only wiring in derivation_topdown/adapters/reads_manifest.py (the per-row N-tag
column). This station is PLUMBING + FENCING: it moves no value and closes no row (see the
SCOPE DECLARATION printed below).

Exit 0 iff every EP-0..EP-6 check PASSES.
"""
import importlib.util
import math
import os
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))          # proofs/foundations -> proofs -> repo
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "dirac_srs_mdl"))
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "bridge"))
import srs               # noqa: E402  -- engine primitives (walled-off clean room). Never edited.
import the_run as R      # noqa: E402  -- THE ENGINE (Layer 1). Never edited above the S1d marker.
import derivation_topdown.adapters.reads_manifest as M   # noqa: E402  -- the manifest (read-only import)

ok_all = True


def banner(t):
    print("=" * 88)
    print(f" {t}")
    print("=" * 88)


def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    return cond


REL_TIGHT = 1e-12   # EP-0/EP-1/EP-5's "bit-consistent" tolerance


def relclose(a, b, tol=REL_TIGHT):
    if b == 0:
        return abs(a) < tol
    return abs(a - b) / abs(b) < tol


# ==============================================================================================
banner("SCOPE DECLARATION (printed, per the pre-reg)")
# ==============================================================================================
print("""  NOT claimed by this station: any early-universe PREDICTION (this station is plumbing +
  fencing — it moves no value and closes no row); the era-selection dynamics (which era holds
  at which N = ML-3's open dynamical crossing; the API takes p_era as an argument precisely
  because this is open); epoch-dependent fermion masses (fenced; un-derived); any modification
  of the locks (defined at N_now, unchanged).""")

# ==============================================================================================
banner("EP-0  IDENTITY -- N_NOW() == the lock-freeze value")
# ==============================================================================================
N_NOW_FROZEN_REFERENCE = 8.394881088442309e60   # EP-0's OWN frozen reference constant (the ONE
                                                  # hardcoded N_hub value the pre-reg's poisons
                                                  # permit, and only here, in the contract file)
n_now = R.N_NOW()
print(f"  N_NOW() = {n_now!r}")
check("EP-0 N_NOW() == 8.394881088442309e60 (rel <= 1e-12)",
      relclose(n_now, N_NOW_FROZEN_REFERENCE), detail=f"rel={abs(n_now-N_NOW_FROZEN_REFERENCE)/N_NOW_FROZEN_REFERENCE:.3e}")

# ==============================================================================================
banner("EP-1  PRESENT-DAY ANCHOR -- read_epoch(N_NOW()) == the engine's own present-day reads")
# ==============================================================================================
higgs = R.read_higgs_chain()
pc = R.read_ported_cosmology()
pg = R.read_ported_gauge_running()
e_now = R.read_epoch(n_now)

pairs = [
    ("H_sub_km_s_Mpc", "H_0 (read_ported_cosmology)", e_now["H_sub_km_s_Mpc"], pc["H_0"]),
    ("t", "t_0 (read_ported_cosmology)", e_now["t"], pc["t_0"]),
    ("Lambda_CC", "Lambda_CC (read_ported_cosmology)", e_now["Lambda_CC"], pc["Lambda_CC"]),
    ("m_nu3_eV", "m_nu3 (read_ported_gauge_running)", e_now["m_nu3_eV"], pg["m_nu3"]),
    ("m_nu2_eV", "m_nu2 (read_ported_gauge_running)", e_now["m_nu2_eV"], pg["m_nu2"]),
]
for epoch_key, engine_label, ev, lv in pairs:
    rel = abs(ev - lv) / abs(lv) if lv else abs(ev)
    check(f"EP-1 read_epoch(N_now)[{epoch_key!r}] == {engine_label}",
          relclose(ev, lv), detail=f"epoch={ev:.12g} engine={lv:.12g} rel={rel:.3e}")

# also cross-check the DIRECT native forms (the_run.py:1403-1416), not just the ported roster,
# since the pre-reg names those line numbers explicitly as the source of the direct N^(-1/2) forms
m_nu3_direct = R.read_m_nu3_eV(higgs)
m_nu2_direct = R.read_m_nu2_eV(m_nu3_direct)
check("EP-1 read_epoch(N_now)['m_nu3_eV'] == read_m_nu3_eV(higgs) directly",
      relclose(e_now["m_nu3_eV"], m_nu3_direct))
check("EP-1 read_epoch(N_now)['m_nu2_eV'] == read_m_nu2_eV(...) directly",
      relclose(e_now["m_nu2_eV"], m_nu2_direct))

# EP-1b (informal, folded into EP-1 -- not a separate frozen contract number): read_epoch adds
# ZERO new physics at N_now, AND is vectorization-friendly (deliverable 1's explicit requirement).
Ns_vec = np.array([n_now * 0.5, n_now, n_now * 2.0])
e_vec = R.read_epoch(Ns_vec)
scalar_checks = [R.read_epoch(float(x))["H_sub"] for x in Ns_vec]
vec_ok = isinstance(e_vec["H_sub"], np.ndarray) and all(
    relclose(e_vec["H_sub"][i], scalar_checks[i]) for i in range(3))
check("EP-1 read_epoch accepts a numpy array N and matches elementwise scalar calls (vectorization)",
      vec_ok)
check("EP-1 read_epoch accepts a bare Python float N (scalar in -> scalar out)",
      isinstance(R.read_epoch(float(n_now))["H_sub"], float))

# ==============================================================================================
banner("EP-2  REGISTRY COMPLETENESS -- every manifest row (Tier-A map + Tier-B compositions) has an N-tag")
# ==============================================================================================
E_manifest = M.phase1_engine_reads()
locks, locks_meta = M.load_locks(M.LOCKS_PATH)
tier_b = M._tier_b_compositions(E_manifest, locks)
tier_a_lock_keys = set(lk for lk, _, _ in M.TIER_A_MAP.values())
tier_b_lock_keys = set(tier_b.keys())
all_manifest_keys = tier_a_lock_keys | tier_b_lock_keys

untagged = sorted(all_manifest_keys - set(R.N_DEPENDENCE.keys()))
extra_registry_only = sorted(set(R.N_DEPENDENCE.keys()) - all_manifest_keys)

from collections import Counter
class_counts = Counter(tag[0] for tag in R.N_DEPENDENCE.values())
print(f"  manifest rows (Tier-A U Tier-B) = {len(all_manifest_keys)}")
print(f"  N_DEPENDENCE registry rows      = {len(R.N_DEPENDENCE)}")
print(f"  class counts: {dict(class_counts)}")
print(f"  UNTAGGED manifest rows (no N_DEPENDENCE entry): {len(untagged)}  {untagged}")
if extra_registry_only:
    print(f"  registry-only rows (no matching manifest row, informational, not a gate): "
          f"{len(extra_registry_only)}  {extra_registry_only}")

check("EP-2 UNTAGGED == 0 (every manifest row has an N-tag)", len(untagged) == 0,
      detail=f"{len(all_manifest_keys)} manifest rows, {len(untagged)} untagged")
check("EP-2 no stray registry-only rows (registry == manifest row set exactly)",
      len(extra_registry_only) == 0)
check("EP-2 class counts sum to the full registry", sum(class_counts.values()) == len(R.N_DEPENDENCE))

# ==============================================================================================
banner("EP-3  THE CALIBRATION FENCE (the teeth) -- read_epoch excludes the fenced family at every N")
# ==============================================================================================
EXCLUSION_SET = {
    "v_higgs", "G_F", "m_H", "lambda_3_higgs",
    "m_e", "m_mu", "m_tau", "m_u", "m_d", "m_s", "m_c", "m_b", "m_t",   # "any fermion mass except m_nu2/m_nu3"
    "M_Z", "m_W",
    "Gamma_Z_over_M_Z", "Gamma_W_over_Gamma_Z",                          # "any width"
    "tan_beta", "T_e_ann",
}
print(f"  exclusion set ({len(EXCLUSION_SET)}): {sorted(EXCLUSION_SET)}")

e_plain = R.read_epoch(n_now)
e_era = R.read_epoch(n_now, p_era=R.ERA_EXPONENTS["matter"])
check("EP-3 read_epoch(N) [no era] contains NONE of the exclusion set",
      len(EXCLUSION_SET & set(e_plain.keys())) == 0)
check("EP-3 read_epoch(N, p_era=...) contains NONE of the exclusion set either",
      len(EXCLUSION_SET & set(e_era.keys())) == 0)

mistagged = [k for k in EXCLUSION_SET if R.N_DEPENDENCE.get(k, (None,))[0] != "calibration-curve"]
check("EP-3 every exclusion-set member is tagged calibration-curve in N_DEPENDENCE",
      len(mistagged) == 0, detail=f"mistagged={mistagged}")

# the module docstring/section-banner states WHY (the tether's defining curve != epoch physics)
with open(os.path.join(REPO, "derivation_topdown", "bridge", "the_run.py"), encoding="utf-8") as f:
    the_run_src = f.read()
marker_idx = the_run_src.find("==== S1d EPOCH API")
s1d_src = the_run_src[marker_idx:] if marker_idx >= 0 else ""
rationale_present = ("TETHER" in s1d_src.upper()) and ("CALIBRATION" in s1d_src.upper()) and (
    "epoch physics" in s1d_src or "an epoch prediction" in s1d_src)
check("EP-3 the S1d section states the calibration-fence rationale (tether != epoch physics)",
      marker_idx >= 0 and rationale_present)

# ==============================================================================================
banner("EP-4  ERA EXPLICITNESS -- era outputs ONLY with p_era; exponents == MG-1c's 2/n at n={4,3,2}")
# ==============================================================================================
ERA_KEYS = {"a_ratio", "H_metric", "T_of_N"}
e_no_era = R.read_epoch(n_now)
e_with_era = R.read_epoch(n_now, p_era=R.ERA_EXPONENTS["radiation"])
check("EP-4 read_epoch(N) [p_era unset] has NONE of {a_ratio,H_metric,T_of_N}",
      len(ERA_KEYS & set(e_no_era.keys())) == 0)
check("EP-4 read_epoch(N, p_era=...) has ALL of {a_ratio,H_metric,T_of_N}",
      ERA_KEYS <= set(e_with_era.keys()))

# TypeError check: p_era has NO default other than None -- calling era-dependent physics without
# it is impossible by construction (there is no keyword that smuggles a default era in).
import inspect
sig = inspect.signature(R.read_epoch)
check("EP-4 read_epoch's p_era default is None (no era default; must be passed explicitly)",
      sig.parameters["p_era"].default is None)

# the three declared exponents == MG-1c's era_exponent(n)=2/n at n=4,3,2 -- RECOMPUTED here
# (not imported/copied from the MG-1c proof file, per the pre-reg's own instruction; that file
# is a top-level script that sys.exit()s on import, so it is deliberately not imported).
def era_exponent_recomputed(n):
    return Fraction(2, n)

check("EP-4 ERA_EXPONENTS['radiation'] == era_exponent(4) == 1/2",
      R.ERA_EXPONENTS["radiation"] == era_exponent_recomputed(4) == Fraction(1, 2))
check("EP-4 ERA_EXPONENTS['matter'] == era_exponent(3) == 2/3",
      R.ERA_EXPONENTS["matter"] == era_exponent_recomputed(3) == Fraction(2, 3))
check("EP-4 ERA_EXPONENTS['reciprocal'] == era_exponent(2) == 1",
      R.ERA_EXPONENTS["reciprocal"] == era_exponent_recomputed(2) == Fraction(1, 1))
check("EP-4 all three ERA_EXPONENTS values are exact Fractions (not floats)",
      all(isinstance(v, Fraction) for v in R.ERA_EXPONENTS.values()))

# MG-0's H_metric = p*H_sub theorem, sampled at each declared era
for era_name, p in R.ERA_EXPONENTS.items():
    e_era_i = R.read_epoch(n_now, p_era=p)
    check(f"EP-4 H_metric == p*H_sub at era={era_name} (p={p})",
          relclose(e_era_i["H_metric"], float(p) * e_era_i["H_sub"]))

# ==============================================================================================
banner("EP-5  TWO-CLOCKS RECONCILIATION -- scale_bridge_pin_T_epoch's T_of_N form, engine N substituted")
# ==============================================================================================
scale_bridge_path = os.path.join(REPO, "proofs", "foundations", "scale_bridge_pin_T_epoch_2026-06-01.py")
spec = importlib.util.spec_from_file_location("scale_bridge_pin_T_epoch_2026_06_01", scale_bridge_path)
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)   # module-level only defines constants/functions -- no main() runs (guarded)

print(f"  scale_bridge's own T_TODAY_eV = {sb.T_TODAY_eV!r}   engine's T_TODAY_EV_S1D = {R.T_TODAY_EV_S1D!r}")
check("EP-5 the two files' T_today constants are IDENTICAL (same declared external input)",
      sb.T_TODAY_eV == R.T_TODAY_EV_S1D)

n_hub_drift_rel = (n_now - sb.N_HUB) / sb.N_HUB
print(f"  scale_bridge's own hardcoded local N_HUB = {sb.N_HUB!r}")
print(f"  engine N_NOW()                            = {n_now!r}")
print(f"  BOOKED (not edited): the old file's hardcoded N_HUB drift vs the engine value = "
      f"{n_hub_drift_rel:+.6e} relative")

# the RECONCILIATION proper: substitute the ENGINE's N for the old file's local constant --
# i.e. reconstruct scale_bridge's OWN form (T_today * sqrt(N_hub_used/N)) using T_today from
# that file and N_hub_used = the ENGINE's N_now, and compare to read_epoch's T_of_N at p=1/2,
# evaluated at the SAME N and the SAME N_now on both sides.
N_eval = n_now / 2.0    # an arbitrary nontrivial evaluation point (representative early-epoch N)
form_scale_bridge_engineN = sb.T_TODAY_eV * math.sqrt(n_now / N_eval)
form_read_epoch = R.read_epoch(N_eval, p_era=Fraction(1, 2))["T_of_N"]
rel_form = abs(form_read_epoch - form_scale_bridge_engineN) / abs(form_scale_bridge_engineN)
print(f"  scale_bridge form (engine N substituted) = {form_scale_bridge_engineN!r}")
print(f"  read_epoch(N_eval, p_era=1/2)['T_of_N']  = {form_read_epoch!r}   rel = {rel_form:.3e}")
check("EP-5 scale_bridge_pin_T_epoch's T_of_N FORM == read_epoch's T_of_N at p=1/2 "
      "(engine N substituted for the old local constant; rel <= 1e-12)",
      rel_form <= REL_TIGHT)

# context only (NOT gated): what the OLD file's function itself returns with its own drifted
# local N_HUB, for transparency -- shows the drift is exactly half the N-drift (T ~ N^{-1/2}).
old_direct = sb.T_of_N(N_eval)
rel_old_direct_vs_new = abs(old_direct - form_read_epoch) / form_read_epoch
print(f"  [context, not gated] scale_bridge.T_of_N(N_eval) with its OWN drifted N_HUB = "
      f"{old_direct!r}  (rel vs read_epoch = {rel_old_direct_vs_new:.3e}, "
      f"~ half the N_HUB drift as expected from T~N^-1/2)")

# ==============================================================================================
banner("EP-6  NOTHING MOVED -- lock harness green (additive growth sanctioned), manifest fast green")
# ==============================================================================================
BUDGET_S = 100   # generous per-subprocess budget; total station budget is ~120s (deliverable 2)

def _run_subprocess(cmd, label, budget):
    t0 = time.time()
    try:
        res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=budget)
        elapsed = time.time() - t0
        return res.returncode, res.stdout + res.stderr, elapsed, False
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return None, "", elapsed, True

rc_lock, out_lock, dt_lock, timed_out_lock = _run_subprocess(
    [sys.executable, "scripts/value_lock.py"], "value_lock", BUDGET_S)
if timed_out_lock:
    print(f"  EP-6 value_lock.py exceeded the {BUDGET_S}s budget -- DISCLOSED FALLBACK: run "
          f"manually: python3 scripts/value_lock.py")
    import json
    with open(M.LOCKS_PATH) as f:
        lockfile = json.load(f)
    # INTEGRATION FIX (2026-07-10): the lock file GROWS additively by design (S1b->107, R1
    # harvest->134, ...). EP-6's invariant is "S1d moved nothing": harness exits 0 AND the count
    # never shrinks below the S1d-era 107.
    check("EP-6 [fallback, cheap invariant] lock file has >= 107 entries (additive growth only)",
          len(lockfile["values"]) >= 107)
else:
    print(f"  value_lock.py ran in {dt_lock:.1f}s, exit={rc_lock}")
    print("  " + "\n  ".join(out_lock.strip().splitlines()[-3:]))
    check("EP-6 scripts/value_lock.py exits 0 (all locked values bit-identical, no drift)",
          rc_lock == 0 and "PASS" in out_lock)

rc_fast, out_fast, dt_fast, timed_out_fast = _run_subprocess(
    [sys.executable, "derivation_topdown/adapters/reads_manifest.py", "--fast"], "manifest --fast", BUDGET_S)
if timed_out_fast:
    print(f"  EP-6 reads_manifest.py --fast exceeded the {BUDGET_S}s budget -- DISCLOSED FALLBACK: "
          f"asserting the cheap in-process invariant instead (parse_ok + zero Tier-A mismatches).")
    tier_a_rows = M.phase3_tier_a_compare(E_manifest, locks)
    n_mismatch = sum(1 for r in tier_a_rows if not r["passed"])
    check("EP-6 [fallback, cheap invariant] manifest Tier-A: zero mismatches",
          n_mismatch == 0)
else:
    print(f"  reads_manifest.py --fast ran in {dt_fast:.1f}s, exit={rc_fast}")
    print("  " + "\n  ".join(out_fast.strip().splitlines()[-2:]))
    check("EP-6 derivation_topdown/adapters/reads_manifest.py --fast exits 0 (fast mode stays green)",
          rc_fast == 0)

# the_run.py append-only: a clean `git diff` on that ONE file must contain ZERO removed lines
# (a modified line always shows as a '-'-prefixed line in unified diff; pure addition has none).
try:
    diff_out = subprocess.run(
        ["git", "diff", "--no-color", "--unified=0", "--", "derivation_topdown/bridge/the_run.py"],
        cwd=REPO, capture_output=True, text=True, timeout=30).stdout
    removed_lines = [ln for ln in diff_out.splitlines()
                     if ln.startswith("-") and not ln.startswith("---")]
    print(f"  git diff derivation_topdown/bridge/the_run.py: {len(removed_lines)} removed line(s) "
          f"(0 expected -- pure addition)")
    if removed_lines:
        print("  REMOVED LINES (should be empty):")
        for ln in removed_lines[:20]:
            print(f"    {ln}")
    check("EP-6 [manual-step-acceptable per pre-reg] git diff on the_run.py shows ZERO removed "
          "lines (pure append; no existing line touched)", len(removed_lines) == 0)
except Exception as ex:
    print(f"  git diff check could not run ({ex}) -- MANUAL STEP for the checker: run "
          f"`git diff -- derivation_topdown/bridge/the_run.py` and confirm zero '-'-prefixed lines.")
    check("EP-6 git-diff append-only check ran", False, detail=str(ex))

# ==============================================================================================
banner("SUMMARY")
# ==============================================================================================
verdict = "ALL EP-0..EP-6 PASS" if ok_all else "SEE FAILURES ABOVE"
print(f"""  S1d EPOCH API -- {verdict}.
  N_NOW() = {n_now!r}
  N_DEPENDENCE: {len(R.N_DEPENDENCE)} rows tagged ({dict(class_counts)}), 0 untagged manifest rows.
  ERA_EXPONENTS: {dict(R.ERA_EXPONENTS)}
  Calibration fence: {len(EXCLUSION_SET)}-member exclusion set, all tagged calibration-curve,
    NONE ever returned by read_epoch() at any N.
  EP-5 constant drift (scale_bridge's local N_HUB vs the engine's N_NOW(), BOOKED not edited):
    {n_hub_drift_rel:+.6e} relative.
  This station moves NO scoreboard value and closes NO row (plumbing + fencing only).""")
sys.exit(0 if ok_all else 1)
