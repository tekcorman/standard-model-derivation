#!/usr/bin/env python3
"""Categorical refinement walker — A2-greedy enumeration of compressible
refinements of F(E)'s word partition, no catalog consulted.

The architecture (per the session's bit-budget closure):
    state := partition of F(E)'s length-N words
    candidate refinement := normal-closure quotient by a relator r ∈ F(E)*
    Φ(r) := log₂(|classes_before| / |classes_after|)
    L(r) := bits to specify r in a fixed prefix-free generator grammar
    A2-pass := Φ(r) ≥ L(r)
    walk := greedy A2-passing chain; halt when no relator passes

This v0 prototype:
- Enumerates relators of length 2..k_max
- Applies each via orbit-BFS-canonicalization on the existing partition
- Reports A2-passing relators ranked by Φ - L
- Walks greedily and halts

Independent rediscovery test: starting from F(E) with no relations imposed,
verify the walk picks (T_e² = id) as the first canonical refinement, then
follows the lean cascade. If it picks something the catalog doesn't have,
that's a discovery.
"""
from __future__ import annotations
from dataclasses import replace
from typing import Iterable
import math, itertools, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from substrate_state import (  # noqa: E402
    SubstrateState, all_words, initial_state, refine_partition,
)

# ---------------- Relator type + canonicalization ----------------

Relator = tuple   # tuple of generator indices

def canonical_relator(r: Relator) -> Relator:
    """Lex-min over cyclic rotations of r and of its reverse.
    Two relators that share a canonical form generate the same normal
    closure (since cyclic conjugation and inversion both fix <<r>>)."""
    if not r:
        return r
    candidates = set()
    for w in (tuple(r), tuple(reversed(r))):
        for i in range(len(w)):
            candidates.add(w[i:] + w[:i])
    return min(candidates)

def is_proper_relator(r: Relator) -> bool:
    """Reject trivial/redundant relators:
    - empty (no relation)
    - length 1 (r = T_e = id collapses entire substrate; uninteresting)
    - already its own canonical form (avoid enumerating duplicates)
    """
    if len(r) < 2:
        return False
    return canonical_relator(r) == tuple(r)

def enumerate_relators(E: int, max_length: int) -> Iterable[Relator]:
    """Enumerate canonical relators over E generators, length 2..max_length."""
    seen: set[Relator] = set()
    for k in range(2, max_length + 1):
        for r in itertools.product(range(E), repeat=k):
            r_can = canonical_relator(r)
            if r_can in seen:
                continue
            if not is_proper_relator(r_can):
                continue
            seen.add(r_can)
            yield r_can

# ---------------- Orbit canonicalization ----------------

def orbit_canonical(word: tuple, r: Relator, n_max: int,
                     max_orbit: int = 5000,
                     cache: dict | None = None) -> tuple:
    """Lex-min of the orbit of `word` under {insert/remove any cyclic rotation
    or reverse of r, anywhere}. Optional `cache` shared across calls memoizes
    word → canonical so each orbit is BFS'd once.
    """
    if cache is not None and word in cache:
        return cache[word]
    L = len(r)
    bound = n_max + L
    rotations = set()
    for w in (tuple(r), tuple(reversed(r))):
        for i in range(len(w)):
            rotations.add(w[i:] + w[:i])
    rotations = list(rotations)

    visited = {word}
    queue = [word]
    while queue and len(visited) < max_orbit:
        w = queue.pop()
        wlen = len(w)
        for rvar in rotations:
            for i in range(wlen - L + 1):
                if w[i:i+L] == rvar:
                    w2 = w[:i] + w[i+L:]
                    if w2 not in visited:
                        visited.add(w2); queue.append(w2)
            if wlen + L <= bound:
                for i in range(wlen + 1):
                    w2 = w[:i] + rvar + w[i:]
                    if w2 not in visited:
                        visited.add(w2); queue.append(w2)
    canonical = min(visited)
    if cache is not None:
        for v in visited:
            cache[v] = canonical
    return canonical

# ---------------- Formal L (prefix-free generator grammar) ----------------

# Schema: each relator is encoded as
#   (length k) prefix-free, log₂(MAX_K + 1) bits via Elias gamma-style ceiling
#   + k · log₂(E) bits for letter sequence
# Plus an additive constant for "this is a relator declaration" (call it 1 bit).
DECLARATION_BITS = 1.0

def formal_L(r: Relator, E: int, max_length: int = 8) -> float:
    """Bits to specify relator r given alphabet size E.
    Length encoded with log₂(max_length+1); each letter with log₂(E)."""
    return DECLARATION_BITS + math.log2(max_length + 1) + len(r) * math.log2(E)

# ---------------- Apply a relator to the partition ----------------

def apply_relator(state: SubstrateState, r: Relator,
                  max_orbit: int = 5000) -> SubstrateState:
    """Refine state by quotienting under the normal closure of r.
    Each word is mapped to the lex-min of its orbit under r-rewrites.
    Uses a shared cache so each orbit is BFS'd once across all input words."""
    E = state.E
    n_max = state.n_max
    cache: dict = {}

    def equiv_fn(w):
        return orbit_canonical(w, r, n_max, max_orbit, cache=cache)

    return refine_partition(
        state,
        equiv_fn=equiv_fn,
        ref_label=f'rel{r}',
        L=int(round(formal_L(r, E))),
        emits=f'relator {r} imposed',
    )

# ---------------- enumerate_minimal_epis at a given state ----------------

def enumerate_minimal_epis(state: SubstrateState, max_relator_length: int,
                            max_orbit: int = 5000) -> list[dict]:
    """Return list of candidate refinements, one per relator that strictly
    refines `state`. Each entry: {r, Phi, L, n_after, passes_a2}."""
    candidates = []
    for r in enumerate_relators(state.E, max_relator_length):
        # Skip if relator is too long for orbit BFS to stay tractable
        if len(r) > state.n_max:
            # Orbit may be unbounded; skip for v0 (extend later)
            continue
        new_state = apply_relator(state, r, max_orbit=max_orbit)
        n_before = state.n_classes
        n_after = new_state.n_classes
        if n_after >= n_before:
            continue   # no refinement
        Phi = math.log2(n_before / n_after)
        L = formal_L(r, state.E)
        candidates.append({
            'r': r,
            'Phi': Phi,
            'L': L,
            'net': Phi - L,
            'n_before': n_before,
            'n_after': n_after,
            'passes_a2': Phi >= L,
            'new_state': new_state,
        })
    candidates.sort(key=lambda c: -c['net'])
    return candidates

# ---------------- E-symmetric schema (one move imposes E sibling relators) ----------------

def is_e_symmetric_orbit(r: Relator, E: int) -> bool:
    """Check whether r is the canonical form of an E-orbit class — i.e., does
    permuting all generators e → π(e) for some π ∈ Sym(E) give the same
    canonical r? If so, the schema 'r' has E! / |stabilizer| siblings."""
    # Easy sufficient case: r consists of a single repeated letter
    return len(set(r)) == 1

def apply_symmetric_schema(state: SubstrateState, r_template: Relator,
                            max_orbit: int = 5000) -> tuple[SubstrateState, list[Relator]]:
    """Apply the E-symmetric schema where r_template's letters are abstract;
    impose all E concrete instantiations simultaneously (e.g., template (a,a)
    expands to (0,0), (1,1), ..., (E-1,E-1)).

    Returns (new_state, list_of_concrete_relators_imposed).
    """
    E = state.E
    n_max = state.n_max
    distinct_slots = sorted(set(r_template))
    n_slots = len(distinct_slots)
    relators: list[Relator] = []
    if n_slots == 1:
        # Template like (a,a,...) → instantiate a ↦ each e ∈ {0,..,E-1}
        for e in range(E):
            r_concrete = tuple(e for _ in r_template)
            relators.append(canonical_relator(r_concrete))
    else:
        # Template like (a,b,a,b) → instantiate a, b distinctly; permutations of E choose n_slots
        from itertools import permutations
        for perm in permutations(range(E), n_slots):
            assignment = dict(zip(distinct_slots, perm))
            r_concrete = tuple(assignment[s] for s in r_template)
            relators.append(canonical_relator(r_concrete))
    relators = sorted(set(relators))

    # Apply all relators simultaneously: equivalence under union of orbits.
    # Each relator generates rewrites for all cyclic rotations + reverse.
    # Flatten to a single rewrite-variant list for efficiency.
    all_variants = []
    for r in relators:
        for w in (tuple(r), tuple(reversed(r))):
            for i in range(len(w)):
                all_variants.append(w[i:] + w[:i])
    all_variants = list(set(all_variants))   # dedupe
    bound = n_max + max(len(r) for r in relators)  # tighter bound

    # Orbit-level cache: word → its orbit's canonical representative.
    # This is the dominant optimization: BFS once per orbit, not once per
    # input word. With ~1300 input words and orbits often hundreds of words
    # large, this is a 100×+ speedup for length-4 schemas.
    canon_cache: dict[tuple, tuple] = {}

    def equiv_fn(w):
        if w in canon_cache:
            return canon_cache[w]
        visited = {w}
        queue = [w]
        while queue and len(visited) < max_orbit:
            x = queue.pop()
            xlen = len(x)
            for rvar in all_variants:
                Lr = len(rvar)
                # Removals
                for i in range(xlen - Lr + 1):
                    if x[i:i+Lr] == rvar:
                        y = x[:i] + x[i+Lr:]
                        if y not in visited:
                            visited.add(y); queue.append(y)
                # Insertions, length-bounded
                if xlen + Lr <= bound:
                    for i in range(xlen + 1):
                        y = x[:i] + rvar + x[i:]
                        if y not in visited:
                            visited.add(y); queue.append(y)
        canonical = min(visited)
        # Cache: every word in this orbit shares the same canonical
        for v in visited:
            canon_cache[v] = canonical
        return canonical

    # Schema L: one template + log₂(E) bits per distinct slot for "for all" vs "exists"
    schema_L = (DECLARATION_BITS + math.log2(8 + 1) +
                 len(r_template) * math.log2(2 + 1) +   # log_2(#distinct slots+1) per letter
                 n_slots * math.log2(E))                 # log_2(E) per slot for "for all"
    new_state = refine_partition(
        state,
        equiv_fn=equiv_fn,
        ref_label=f'schema{r_template}',
        L=int(round(schema_L)),
        emits=f'schema {r_template} imposed via {len(relators)} concrete relators',
    )
    return new_state, relators

def schema_L(r_template: Relator, E: int) -> float:
    """Description bits for an E-symmetric schema."""
    n_slots = len(set(r_template))
    return (DECLARATION_BITS + math.log2(8 + 1) +
            len(r_template) * math.log2(n_slots + 1) +
            n_slots * math.log2(E))

# ---------------- Schema enumeration ----------------

def enumerate_schemas(max_length: int) -> Iterable[Relator]:
    """Enumerate canonical schema templates (relator templates over abstract
    letters, where template (a,a) covers all (e,e) instantiations).

    Templates use letters 0,1,2,... in order of first appearance to canonicalize.
    """
    seen: set[Relator] = set()
    for k in range(2, max_length + 1):
        # Enumerate all length-k templates with letters from {0, ..., k-1}
        for letters in itertools.product(range(k), repeat=k):
            # Canonicalize: relabel so first appearance order is 0,1,2,...
            relabel = {}
            counter = 0
            canon = []
            for x in letters:
                if x not in relabel:
                    relabel[x] = counter
                    counter += 1
                canon.append(relabel[x])
            t = tuple(canon)
            # Also canonicalize cyclic + reversal
            t_can = canonical_relator(t)
            # Re-canonicalize the slot labeling after cyclic shift
            relabel2 = {}
            counter2 = 0
            recanon = []
            for x in t_can:
                if x not in relabel2:
                    relabel2[x] = counter2
                    counter2 += 1
                recanon.append(relabel2[x])
            t_final = tuple(recanon)
            if t_final in seen:
                continue
            if not is_proper_relator(t_final):
                continue
            seen.add(t_final)
            yield t_final

def enumerate_minimal_schema_epis(state: SubstrateState, max_length: int,
                                    max_orbit: int = 5000,
                                    max_distinct_slots: int = 2) -> list[dict]:
    """Enumerate symmetric-schema candidates (more compressive than single
    relators because they impose E sibling relations at once with shared L).

    `max_distinct_slots` caps the template's number of distinct abstract
    letters (default 2). Higher slot counts produce E·(E-1)·(E-2)·...
    concrete relators — combinatorially expensive. Slot count 2 covers
    involutivity (a,a), (a,b), abelianization (a,b,a,b), commutators, etc."""
    out = []
    for tmpl in enumerate_schemas(max_length):
        if len(tmpl) > state.n_max:
            continue
        if len(set(tmpl)) > max_distinct_slots:
            continue
        new_state, relators_used = apply_symmetric_schema(state, tmpl, max_orbit)
        n_before = state.n_classes
        n_after = new_state.n_classes
        if n_after >= n_before:
            continue
        Phi = math.log2(n_before / n_after)
        L = schema_L(tmpl, state.E)
        out.append({
            'template': tmpl,
            'concrete_relators': relators_used,
            'Phi': Phi,
            'L': L,
            'net': Phi - L,
            'n_before': n_before,
            'n_after': n_after,
            'passes_a2': Phi >= L,
            'new_state': new_state,
        })
    out.sort(key=lambda c: -c['net'])
    return out

# ---------------- Conjugation move (cyclic-rotation quotient) ----------------

def cyclic_canonical(word: tuple) -> tuple:
    """Lex-min over cyclic rotations of `word`. The conjugation quotient
    identifies words that differ by inner-automorphism action: u ~ v iff
    there exists g such that v = g·u·g⁻¹. For free involutive products,
    this is exactly cyclic equivalence (up to free reduction)."""
    if not word:
        return word
    rotations = [word[i:] + word[:i] for i in range(len(word))]
    return min(rotations)

def apply_conjugation(state: SubstrateState) -> SubstrateState:
    """Quotient state by cyclic-rotation equivalence (conjugation by F_inv(E)
    elements). This is a Move of a different categorical type than relator
    quotients — it's a quotient by inner-automorphism action, not by a
    normal closure of an explicit relator."""
    return refine_partition(
        state,
        equiv_fn=cyclic_canonical,
        ref_label='conjugation',
        L=int(round(conjugation_L())),
        emits='conjugation/inner-automorphism quotient',
    )

def conjugation_L() -> float:
    """Description bits for the conjugation move. It's a single named
    operation parameterized only by 'apply cyclic-rotation canonicalization' —
    no E-dependent parameter, no length parameter. Just one declaration."""
    return DECLARATION_BITS + 2.0   # ~3 bits ("conjugation" as a primitive move)

def evaluate_conjugation(state: SubstrateState) -> dict:
    """Return a candidate dict in the same format as enumerate_minimal_schema_epis."""
    new_state = apply_conjugation(state)
    n_before, n_after = state.n_classes, new_state.n_classes
    if n_after >= n_before:
        return None
    Phi = math.log2(n_before / n_after)
    L = conjugation_L()
    return {
        'kind': 'conjugation',
        'template': 'CONJ',
        'concrete_relators': [],
        'Phi': Phi,
        'L': L,
        'net': Phi - L,
        'n_before': n_before,
        'n_after': n_after,
        'passes_a2': Phi >= L,
        'new_state': new_state,
    }

# ---------------- Unified move enumerator ----------------

def enumerate_all_moves(state: SubstrateState, max_relator_length: int,
                         max_orbit: int = 5000) -> list[dict]:
    """Enumerate ALL categorical move types at this state:
      - Relator-schema quotients (T_e T_f ... = id, parameterized templates)
      - Conjugation quotient (cyclic-rotation equivalence)

    Returns list sorted by Φ − L descending (so caller can pick max).
    Each entry has 'kind' field: 'schema' | 'conjugation'.
    """
    moves: list[dict] = []
    # Relator schemas
    schema_cands = enumerate_minimal_schema_epis(state, max_relator_length, max_orbit)
    for c in schema_cands:
        c['kind'] = 'schema'
    moves.extend(schema_cands)
    # Conjugation
    if 'conjugation' not in state.refinements:
        c = evaluate_conjugation(state)
        if c is not None:
            moves.append(c)
    moves.sort(key=lambda c: -c['net'])
    return moves

# ---------------- Lookahead evaluation ----------------

def lookahead_value(state: SubstrateState, max_relator_length: int,
                     max_depth: int, max_orbit: int = 5000) -> float:
    """Recursively walk from `state` greedily by max-(Φ-L) up to `max_depth`
    steps, return the resulting (Φ_total - L_total) at halt.

    This is the value of starting at `state` and walking optimally; it lets
    the caller compare candidate moves by their global walk-value, not just
    immediate Φ-L. With max_depth large enough to reach halt, this gives
    the true global value (subject to the local-greedy approximation).
    """
    s = state
    for _ in range(max_depth):
        cands = enumerate_all_moves(s, max_relator_length, max_orbit)
        if not cands:
            break
        s = cands[0]['new_state']
    return s.Phi_total - s.L_total

def categorical_walk_lookahead(initial: SubstrateState,
                                 max_relator_length: int,
                                 max_steps: int,
                                 lookahead_depth: int = 3,
                                 max_orbit: int = 5000,
                                 verbose: bool = True
                                ) -> tuple[SubstrateState, list[SubstrateState]]:
    """Walk with lookahead: at each step, evaluate each candidate by
    simulating greedy downstream walks to depth `lookahead_depth`. Pick
    the candidate whose continuation gives max total (Φ-L) at halt.
    """
    state = initial
    history = [state]
    for step in range(max_steps):
        cands = enumerate_all_moves(state, max_relator_length, max_orbit)
        if not cands:
            if verbose:
                print(f"\nStep {step+1}: no candidate refinement. HALT.")
            break

        # Evaluate each candidate by lookahead
        for c in cands:
            after = c['new_state']
            c['lookahead_net'] = lookahead_value(after, max_relator_length,
                                                  lookahead_depth, max_orbit)

        best = max(cands, key=lambda c: c['lookahead_net'])

        if verbose:
            kind = best.get('kind', 'schema')
            label = best.get('template') or best.get('r') or kind
            print(f"\nStep {step+1}: applying [{kind}] {label}")
            print(f"  immediate Φ={best['Phi']:.3f}  L={best['L']:.3f}  Δ={best['net']:+.3f}")
            print(f"  lookahead net at halt = {best['lookahead_net']:+.3f}")
            print(f"  classes: {best['n_before']} → {best['n_after']}")
            # Show alternatives ranked by lookahead
            print(f"  alternatives (immediate-greedy vs lookahead):")
            cands_sorted_local = sorted(cands, key=lambda c: -c['net'])
            for c in cands_sorted_local[:3]:
                k = c.get('kind', 'schema')
                lbl = str(c.get('template') or c.get('r') or k)
                marker = '←' if c is best else ' '
                print(f"    {marker} [{k}] {lbl:<20}  immediate Δ={c['net']:+.3f}  "
                      f"lookahead={c['lookahead_net']:+.3f}")
        state = best['new_state']
        history.append(state)
    return state, history

def categorical_walk(initial: SubstrateState,
                      max_relator_length: int = 4,
                      max_steps: int = 20,
                      use_schemas: bool = True,
                      verbose: bool = True,
                      strict_a2_per_step: bool = False) -> SubstrateState:
    """Greedy walk on the partition lattice starting from `initial`.

    Two modes:
      strict_a2_per_step=True:  fire only if Φ_marg ≥ L_marg (per-step gate).
                                Halts early at finite n if Φ < L locally.
      strict_a2_per_step=False: fire the candidate with max Φ - L regardless
                                of sign (greedy max-compression). Halts when
                                no candidate refines further. Global A2 is
                                checked at halt.
    """
    state = initial
    history = [state]
    for step in range(max_steps):
        if use_schemas:
            cands = enumerate_all_moves(state, max_relator_length)
        else:
            cands = enumerate_minimal_epis(state, max_relator_length)
        if not cands:
            if verbose:
                print(f"\nStep {step + 1}: no candidate refinement available. HALT.")
            break
        if strict_a2_per_step:
            passing = [c for c in cands if c['passes_a2']]
            if not passing:
                if verbose:
                    print(f"\nStep {step + 1}: no A2-passing refinement (per-step). HALT.")
                    print(f"  (top non-passing: Φ={cands[0]['Phi']:.3f} < L={cands[0]['L']:.3f})")
                break
            best = passing[0]
        else:
            best = cands[0]   # greedy max-(Φ − L), regardless of sign
        if verbose:
            kind = best.get('kind', 'schema')
            label = best.get('template') or best.get('r') or kind
            print(f"\nStep {step + 1}: applying [{kind}] {label}  "
                  f"Φ={best['Phi']:.3f}  L={best['L']:.3f}  Δ={best['net']:+.3f}")
            print(f"  classes: {best['n_before']} → {best['n_after']}")
            if best.get('concrete_relators'):
                concrete = best['concrete_relators']
                preview = concrete[:3]
                more = f" ... ({len(concrete) - 3} more)" if len(concrete) > 3 else ''
                print(f"  concrete relators: {preview}{more}")
        state = best['new_state']
        history.append(state)
    return state, history

# ---------------- Main ----------------

if __name__ == '__main__':
    print("=" * 100)
    print("Categorical refinement walker — A2-greedy from F(E) with NO catalog consulted")
    print("=" * 100)

    E = 6
    n_max = 4
    s0 = initial_state(E=E, n_max=n_max)
    print(f"\nInitial state: F({E}) restricted to length-{n_max} words")
    print(f"  classes: {s0.n_classes} (= {E}^{n_max} = {E**n_max} raw words)")
    print(f"  no relations imposed yet")

    # ============ Step 0: unified move enumeration (ranked) ============
    print(f"\n--- Step 0: enumerate all categorical moves (schemas + conjugation, length ≤ 4) ---")
    import time as _time
    t0 = _time.time()
    cands = enumerate_all_moves(s0, max_relator_length=4)
    print(f"  enumeration time: {_time.time() - t0:.2f}s")
    print(f"\n{len(cands)} candidate moves yielded a proper refinement.")
    print(f"\nTop by Φ − L:")
    print(f"  {'kind':<12}{'label':<22}{'Φ':>8}{'L':>8}{'Δ':>9}{'classes':>15}")
    print('  ' + '-' * 75)
    for c in cands[:10]:
        kind = c.get('kind', 'schema')
        label = str(c.get('template') or c.get('r') or kind)
        print(f"  {kind:<12}{label:<22}{c['Phi']:>8.3f}{c['L']:>8.3f}{c['net']:>+9.3f}"
              f"{c['n_before']:>7} →{c['n_after']:>5}")

    # ============ Walk ============
    print(f"\n{'='*100}")
    print(f"Greedy walk (schemas + conjugation, max_relator_length=2):")
    print(f"{'='*100}")
    final, hist = categorical_walk(s0, max_relator_length=4, max_steps=6,
                                    use_schemas=True, verbose=True,
                                    strict_a2_per_step=False)

    print(f"\n{'='*100}")
    print(f"Greedy halt state:")
    print(f"  classes:       {final.n_classes}")
    print(f"  Φ_total:       {final.Phi_total:.4f} bits")
    print(f"  L_total:       {final.L_total} bits")
    print(f"  Net Φ − L:     {final.Phi_total - final.L_total:+.4f}")
    print(f"  refinements:   {final.refinements}")
    print(f"  steps walked:  {len(hist) - 1}")

    # ============ Walk WITH LOOKAHEAD ============
    print(f"\n{'='*100}")
    print(f"Walk WITH LOOKAHEAD (depth=3): each candidate scored by simulating")
    print(f"downstream greedy walk; pick the one with max (Φ-L) at lookahead halt.")
    print(f"{'='*100}")
    final_la, hist_la = categorical_walk_lookahead(
        s0, max_relator_length=4, max_steps=6,
        lookahead_depth=3, verbose=True,
    )

    print(f"\n{'='*100}")
    print(f"Lookahead halt state:")
    print(f"  classes:       {final_la.n_classes}")
    print(f"  Φ_total:       {final_la.Phi_total:.4f} bits")
    print(f"  L_total:       {final_la.L_total} bits")
    print(f"  Net Φ − L:     {final_la.Phi_total - final_la.L_total:+.4f}")
    print(f"  refinements:   {final_la.refinements}")
    print(f"  steps walked:  {len(hist_la) - 1}")

    # Comparison
    print(f"\n{'='*100}")
    print(f"Greedy vs lookahead comparison:")
    print(f"  Greedy:    halt at {final.n_classes} classes, Net = {final.Phi_total - final.L_total:+.4f}")
    print(f"             refinements: {final.refinements}")
    print(f"  Lookahead: halt at {final_la.n_classes} classes, Net = {final_la.Phi_total - final_la.L_total:+.4f}")
    print(f"             refinements: {final_la.refinements}")
    if final.refinements == final_la.refinements:
        print(f"  → confluent: greedy and lookahead reach the same halt.")
    else:
        print(f"  → DIVERGENT: lookahead's global value differs from greedy's local value.")

    # ============ Test: is the catalog's lean cascade halt a walker fixed point? ============
    print(f"\n{'='*100}")
    print(f"Lean cascade fixed-point test:")
    print(f"  Apply catalog's op 0.4 → 1.8 → 1.10 (involutivity + conjugation + abelianization),")
    print(f"  then ask: does the walker find any further refinement, or halt immediately?")
    print(f"{'='*100}")
    from substrate_state import (
        op_0_4_involutive, op_1_8_conjugation, op_1_10_abelianization,
    )
    s_lean = initial_state(E=E, n_max=n_max)
    s_lean = op_0_4_involutive(s_lean)
    s_lean = op_1_8_conjugation(s_lean)
    s_lean = op_1_10_abelianization(s_lean)
    print(f"\n  Lean cascade halt: {s_lean.n_classes} classes, "
          f"Φ={s_lean.Phi_total:.3f}, L={s_lean.L_total}, "
          f"Net = {s_lean.Phi_total - s_lean.L_total:+.3f}")
    print(f"  refinements: {s_lean.refinements}")

    cands_post = enumerate_all_moves(s_lean, max_relator_length=4)
    refining = [c for c in cands_post if c['n_after'] < c['n_before']]
    a2_passing = [c for c in refining if c['passes_a2']]

    if refining:
        print(f"\n  Walker finds {len(refining)} candidate refinement(s) at lean cascade halt:")
        for c in refining[:5]:
            kind = c.get('kind', 'schema')
            label = str(c.get('template') or c.get('r') or kind)
            a2_flag = '✓' if c['passes_a2'] else '✗'
            print(f"    [{kind}] {label}: {c['n_before']} → {c['n_after']} classes, "
                  f"Φ={c['Phi']:.3f}, L={c['L']:.3f}, Δ={c['net']:+.3f}, A2={a2_flag}")

    if not a2_passing:
        print(f"\n  ✓ NO A2-passing refinement at the lean cascade halt.")
        print(f"    Under strict A2 (Φ ≥ L), the catalog's halt IS a fixed point of the walker.")
        if refining:
            print(f"    (The {len(refining)} non-passing candidates are BFS-truncation artifacts")
            print(f"     or sub-A2 micro-compressions; they don't pay for their description cost.)")
    else:
        print(f"\n  ⚠ DISCOVERY: {len(a2_passing)} A2-passing refinement(s) at lean cascade halt.")
        print(f"    Catalog missed compressions that genuinely pay for themselves.")
