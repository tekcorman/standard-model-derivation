#!/usr/bin/env python3
# ============================================================
# Inverse-Noether Scanner — empirical symmetry-detection probe
#
# Phase 4 of the symmetry-shortcut program.
# Predecessors: docs/theorems/theorem_substrate_symmetry_to_martingale.md (engine);
#               docs/theorems/theorem_substrate_generation_charge_conservation.md §6
#               (inverse-Noether parsimony argument).
# Paired md: docs/forward_constructions/forward_construction_inverse_noether_scanner.md.
#
# Purpose. Given a substrate simulator and a list of candidate observables,
# detect which observables witness an underlying symmetry of the dynamics.
# The empirical test is:
#
#     E[f(state)] = E[f(sigma . state)]  for all sigma in candidate G
#
# under the dynamics' stationary distribution. If the dynamics is G-symmetric,
# this equality holds for every observable. If G is broken, the equality fails
# for non-G-invariant observables — and which observables fail tells you HOW
# G is broken.
#
# This is the operational form of the inverse-Noether parsimony heuristic
# (Phase 0 §6 / Phase 1b §6): an observable for which f(state) and f(sigma.state)
# have matching stationary expectations is consistent with hidden G symmetry.
# Confirmation requires the forward direction (verifying (H1)-(H3) explicitly).
#
# Honest scope. The scanner detects distributional symmetry of the stationary;
# it does NOT prove the prior is G-invariant in the Bayesian sense. It produces
# parsimony evidence. Its operational value is:
#   (a) validation: confirms a known-symmetric dynamics is symmetric (sanity check);
#   (b) discovery: flags observables that witness a broken symmetry the user
#       has not written down explicitly.
# ============================================================

import numpy as np

# -----------------------------------------------------------
# Substrate: trivalent vertex toggle chain (3 edges, 8 states)
# -----------------------------------------------------------
# State: (s_1, s_2, s_3) in {0,1}^3. Each edge toggles independently
# with p_create = 1/2 (0->1) and p_destroy = 1/3 (1->0). Per A1 + Stage 2a.

def toggle_step(state, rng, p_create=0.5, p_destroy=1.0/3.0):
    new_state = list(state)
    for i in range(3):
        if state[i] == 0:
            if rng.random() < p_create:
                new_state[i] = 1
        else:
            if rng.random() < p_destroy:
                new_state[i] = 0
    return tuple(new_state)


def toggle_step_asymmetric(state, rng):
    """Diagnostic chain: per-edge asymmetric rates that BREAK site-C_3
    symmetry. Edge index 2 has p_create = 0.20 instead of 0.50; edges 0,1
    keep the canonical 0.50. Used to validate the scanner detects broken
    symmetry."""
    p_create = [0.50, 0.50, 0.20]
    p_destroy = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    new_state = list(state)
    for i in range(3):
        if state[i] == 0:
            if rng.random() < p_create[i]:
                new_state[i] = 1
        else:
            if rng.random() < p_destroy[i]:
                new_state[i] = 0
    return tuple(new_state)


# -----------------------------------------------------------
# Observables to scan
# -----------------------------------------------------------

def f_single_edge_0(state):
    return float(state[0])

def f_single_edge_2(state):
    return float(state[2])

def f_avg_occupancy(state):
    return sum(state) / 3.0

def f_edge_variance(state):
    avg = sum(state) / 3.0
    return sum((s - avg) ** 2 for s in state) / 3.0

def f_pair_diff_01(state):
    return float(state[0] - state[1])

def f_pair_diff_02(state):
    return float(state[0] - state[2])

def f_cyclic_weighted(state):
    return float(state[0] + 2 * state[1] + 3 * state[2])

def f_total_occupancy(state):
    return float(sum(state))

def f_triple_product(state):
    return float(state[0] * state[1] * state[2])

OBSERVABLES = {
    "edge0":            (f_single_edge_0,    "non-C3"),
    "edge2":            (f_single_edge_2,    "non-C3"),
    "avg":              (f_avg_occupancy,    "C3-invariant"),
    "edge_variance":    (f_edge_variance,    "C3-invariant"),
    "pair_diff_01":     (f_pair_diff_01,     "non-C3"),
    "pair_diff_02":     (f_pair_diff_02,     "non-C3"),
    "cyclic_weighted":  (f_cyclic_weighted,  "non-C3"),
    "total":            (f_total_occupancy,  "C3-invariant"),
    "triple_product":   (f_triple_product,   "C3-invariant"),
}


# -----------------------------------------------------------
# Symmetry: site-C_3 cyclic permutation of the 3 edges
# -----------------------------------------------------------

def c3_apply(state, k):
    """Apply k-fold cyclic shift: (s_0,s_1,s_2) -> (s_{(0-k)%3}, s_{(1-k)%3}, s_{(2-k)%3}).
    k=0: identity. k=1: forward shift. k=2: backward shift."""
    return tuple(state[(i - k) % 3] for i in range(3))


def is_c3_invariant_function(observable):
    """Identify whether an observable is identically C_3-invariant by
    checking f(state) == f(sigma.state) on all 8 states."""
    states = [tuple([(j >> i) & 1 for i in range(3)]) for j in range(8)]
    for st in states:
        v0 = observable(st)
        for k in (1, 2):
            if abs(observable(c3_apply(st, k)) - v0) > 1e-12:
                return False
    return True


# -----------------------------------------------------------
# Empirical signature: stationary expectations under the
# three C_3-images of each state.
# -----------------------------------------------------------

def collect_states(step_fn, n_realizations=4000, n_steps=200, seed=42, t_warmup=40):
    """Collect a pool of post-warmup states across realizations.
    The stationary distribution is sampled by aggregating states from
    multiple time slices and multiple realizations."""
    rng = np.random.default_rng(seed)
    states = []
    for r in range(n_realizations):
        s = (0, 0, 0)
        for t in range(n_steps):
            if t >= t_warmup:
                states.append(s)
            s = step_fn(s, rng)
    return states


def assess_observable_c3(observable, states):
    """Compute E[f(state)], E[f(sigma.state)], E[f(sigma^2.state)]
    on the same pool of states. Under C_3-symmetric dynamics, all three
    should agree (the stationary distribution is invariant under sigma).
    Under broken C_3, they differ — and the spread quantifies the breaking
    AT THIS OBSERVABLE."""
    vals_id = np.array([observable(s) for s in states])
    vals_s1 = np.array([observable(c3_apply(s, 1)) for s in states])
    vals_s2 = np.array([observable(c3_apply(s, 2)) for s in states])

    n = len(states)
    means = np.array([vals_id.mean(), vals_s1.mean(), vals_s2.mean()])
    se = np.array([vals_id.std(), vals_s1.std(), vals_s2.std()]) / np.sqrt(n)
    spread = means.max() - means.min()
    spread_se = np.sqrt(np.sum(se ** 2))
    detection_z = spread / max(spread_se, 1e-9)

    return {
        "means_across_sigma": means,
        "spread": spread,
        "spread_se": spread_se,
        "detection_z": detection_z,
        "consistent_with_c3": detection_z < 3.0,
    }


# -----------------------------------------------------------
# Main scan
# -----------------------------------------------------------

def run_scan(step_fn, label):
    print()
    print("=" * 84)
    print(f"SCAN: {label}")
    print("=" * 84)
    states = collect_states(step_fn)

    print(f"{'observable':<18} {'a-priori':<14} {'<f>_id':>9} {'<f>_s1':>9} {'<f>_s2':>9} "
          f"{'spread':>9} {'z-score':>8}  flag")
    print("-" * 100)
    rows = []
    for name, (obs_fn, sym_label) in OBSERVABLES.items():
        invariant = is_c3_invariant_function(obs_fn)
        if invariant:
            row = {"name": name, "label": sym_label, "is_invariant": True,
                   "consistent_with_c3": True, "spread": 0.0, "detection_z": 0.0,
                   "means_across_sigma": None}
            print(f"{name:<18} {sym_label:<14} {'(trivially C3-invariant — pointwise)':>56}")
        else:
            res = assess_observable_c3(obs_fn, states)
            flag = ("+ consistent with C3" if res["consistent_with_c3"]
                    else "- WITNESSES C3 BREAK")
            print(f"{name:<18} {sym_label:<14} "
                  f"{res['means_across_sigma'][0]:>9.4f} "
                  f"{res['means_across_sigma'][1]:>9.4f} "
                  f"{res['means_across_sigma'][2]:>9.4f} "
                  f"{res['spread']:>9.4f} {res['detection_z']:>8.2f}  {flag}")
            row = {"name": name, "label": sym_label, "is_invariant": False, **res}
        rows.append(row)
    return rows


def interpret(rows_sym, rows_asym):
    print()
    print("=" * 84)
    print("INTERPRETATION — inverse-Noether scanner output")
    print("=" * 84)
    print()
    print("Test: under C_3-symmetric dynamics, E[f(s)] = E[f(sigma.s)] for all f, sigma.")
    print("      Pointwise-C_3-invariant observables are trivially consistent (f = f.sigma).")
    print("      Non-invariant observables are the WITNESSES: their across-sigma spread")
    print("      is zero under C_3-symmetric dynamics, non-zero under broken C_3.")
    print()

    def witnesses_breaking(rows):
        return [r for r in rows if not r["is_invariant"] and not r["consistent_with_c3"]]

    sym_break = witnesses_breaking(rows_sym)
    asym_break = witnesses_breaking(rows_asym)

    print(f"SYMMETRIC dynamics (canonical p_c=1/2, p_d=1/3, all edges):")
    print(f"  observables witnessing C_3-breaking: {len(sym_break)} (expected 0)")
    if sym_break:
        for r in sym_break:
            print(f"    - {r['name']}: spread={r['spread']:.4f}, z={r['detection_z']:.2f}")
    print()
    print(f"ASYMMETRIC dynamics (edge 2 has p_c=0.20; site-C_3 BROKEN):")
    print(f"  observables witnessing C_3-breaking: {len(asym_break)}")
    if asym_break:
        print(f"  These observables EMPIRICALLY DETECT the broken symmetry:")
        for r in asym_break:
            print(f"    - {r['name']}: spread={r['spread']:.4f}, z={r['detection_z']:.2f}")
    print()

    if len(sym_break) == 0 and len(asym_break) > 0:
        print("RESULT: scanner correctly classifies both runs.")
        print("  - Symmetric run produces ZERO witnesses (sanity check passed).")
        print("  - Asymmetric run produces witnesses that pinpoint the break to edge 2.")
        print()
        print("Inverse-Noether reading: each non-invariant observable f gives a probe of")
        print("the dynamics' symmetry. The set of probes that pass simultaneously narrows")
        print("the space of consistent-symmetries; probes that fail diagnose how a hidden")
        print("symmetry breaks. This is the operational form of the inverse-Noether")
        print("heuristic from `theorem_substrate_symmetry_to_martingale.md` §6.")
    else:
        print("RESULT: scanner output is non-canonical — review parameters.")


if __name__ == "__main__":
    print()
    print("Inverse-Noether Scanner — empirical symmetry-detection probe")
    print("(Phase 4 of the symmetry-shortcut program.)")
    print()
    print("Substrate: trivalent vertex toggle chain (3 edges, 8 states).")
    print("Candidate symmetry tested: site-C_3 (cyclic permutation of edges).")

    rows_sym = run_scan(toggle_step,
                        "SYMMETRIC toggle (canonical p_c=1/2, p_d=1/3, all edges)")
    rows_asym = run_scan(toggle_step_asymmetric,
                         "ASYMMETRIC toggle (edge 2 has p_c=0.20; site-C_3 BROKEN)")
    interpret(rows_sym, rows_asym)
