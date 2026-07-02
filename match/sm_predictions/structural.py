"""
Structural predictions — Family 9 (graph facts).

These are the irreducible counts that the substrate IS.
Every other prediction depends on them, but they themselves are inputs.

Per the framework, srs is the MDL-dominant substrate at framework scale
N_hub ~ 10^60. Subdominant substrates (different graph, different lattice)
are below waterline at framework scale and don't generate competing predictions.
"""

from fractions import Fraction
from simulator.srs_engine.kernel import CountingKernel


def k_star(kernel=None, max_k=8):
    """k* = 3 — coordination number per vertex on srs. CHANNEL-SELECTED.

    Waterfilling-correct derivation: for a d-dimensional crystal net,
    coordination numbers k ≥ d are PHYSICALLY VALID lattices (they span
    R^d). Each k corresponds to a distinct lattice family with different
    structural properties. The framework's substrate is the
    minimum-coordination spanning lattice — k = d, which is the
    `minimum_spanning_coordination` channel.

    Above-waterline candidates k = 4, 5, ... correspond to OTHER lattices
    with redundant edges (Fisher rank still d but with k − d extra
    directed-edge equivalence classes). They are not discarded —
    they're the framework's "alternative substrates" that lose at
    Phase 0 MDL waterline (per `inventory_state_skeptical_digest_2026-05-06.md`
    Coxeter-quotient menu). For srs specifically, k = d = 3.

    Note: this is NOT argmin over k. The substrate identification
    "minimum spanning coordination" is the channel; alternative channels
    (overspecified lattices) are physically realized but for different
    substrates. channel_select picks the channel-matching candidate.
    """
    kernel = kernel or CountingKernel()
    d = d_spatial(kernel)
    candidates = []
    for k in range(1, max_k):
        if k < d:
            # Below Gleason / spanning waterline — not physically a viable
            # d-dimensional crystal net at all
            continue
        if k == d:
            channel = 'minimum_spanning_coordination'  # srs-class lattice
        else:
            channel = f'overspecified_coordination_{k - d}_redundant_edges'
        candidates.append({
            'name': f'k={k}',
            'k': k,
            'channel': channel,
        })
    selected = kernel.channel_select(
        candidates,
        channel='minimum_spanning_coordination',
    )
    assert selected['k'] == kernel.substrate.K_STAR, (
        f"channel-selected k={selected['k']} disagrees with "
        f"substrate.K_STAR ({kernel.substrate.K_STAR})"
    )
    return selected['k']


def d_spatial(kernel=None):
    """d_spatial = 3 — spatial dimensions. CHANNEL-SELECTED.

    Waterfilling-correct derivation: among Hilbert space dimensions
    n ∈ {1, 2, 3, 4, ...}:
      - n < 3: BELOW WATERLINE (Gleason 1957 — frame functions on ℂ^n
        for n < 3 are non-unique; unbounded waterline penalty;
        discarded as physically unrealizable observers)
      - n ≥ 3: above waterline, all physically realizable as observers
        of n-dimensional crystal nets

    The substrate's primitive cell has rank-3 edge displacement matrix
    (3D crystal net), so the observer matching THIS substrate has
    Hilbert dimension n = 3. Channel: `match_substrate_rank_3`.

    Above-waterline candidates n = 4, 5, ... are NOT discarded — they
    represent observers of higher-dimensional substrates (which the
    framework does not select at framework scale per the Coxeter-
    quotient menu MDL). channel_select picks n = 3 by substrate-rank
    match, not by global bit-cost argmin.
    """
    kernel = kernel or CountingKernel()
    substrate_rank = kernel.substrate.D_SPATIAL  # 3 (BCC primitive cell rank)
    candidates = []
    for n in range(1, 8):
        if n < 3:
            continue  # below Gleason waterline; not a viable observer
        candidates.append({
            'name': f'n={n}',
            'n': n,
            'channel': f'match_substrate_rank_{n}',
        })
    selected = kernel.channel_select(
        candidates,
        channel=f'match_substrate_rank_{substrate_rank}',
    )
    return selected['n']


def g_girth(kernel=None):
    """g = 10 — girth of srs lattice. CHANNEL-SELECTED (Sunada uniqueness).

    Waterfilling-correct derivation: among 3D 3-regular 3-connected crystal
    nets (the Coxeter-quotient menu at framework scale; see
    `inventory_state_skeptical_digest_2026-05-06.md`), Sunada 2012 proves
    srs is the UNIQUE vertex+edge-transitive entry. This is a sharp peak:
    edge-transitivity is a discrete property (yes/no) and only one net
    in the catalog satisfies it.

    Channel: `edge_transitive_3d_3reg_3conn_crystal_net`. Above-waterline
    candidates in OTHER channels (non-edge-transitive 3D 3-regular 3-conn
    nets — lou, lov, okw, srs-c4, etc., per `rcsr_candidate_sweep.py`) are
    physically realized as alternative crystal nets but are not the
    framework's substrate (they exceed Sunada's LB on DL(edges)).

    channel_select picks the unique entry in this channel: srs. Returns
    srs's girth = 10 (mathematical property of the (3,10)-cage).
    """
    kernel = kernel or CountingKernel()
    candidates = [
        {
            'name': 'srs',
            'channel': 'edge_transitive_3d_3reg_3conn_crystal_net',
            'girth': 10,  # mathematical property of srs (RCSR)
        },
        # Above-waterline alternative crystal nets — physically realized
        # but in non-edge-transitive channels (Sunada LB)
        {
            'name': 'non-edge-transitive nets (lou, lov, okw, ...)',
            'channel': 'non_edge_transitive_3d_3reg_3conn_crystal_net',
            'girth': None,  # depends on specific net
        },
    ]
    selected = kernel.channel_select(
        candidates,
        channel='edge_transitive_3d_3reg_3conn_crystal_net',
    )
    assert selected['girth'] == kernel.substrate.GIRTH, (
        f"channel-selected girth ({selected['girth']}) disagrees with "
        f"substrate.GIRTH ({kernel.substrate.GIRTH})"
    )
    return selected['girth']  # 10


def fermion_states_per_gen(kernel=None):
    """8 fermion states per generation per chirality (Cl(6) spinor dim).

    Derived from: 6 anticommuting involutive generators (Cl(6) at trivalent
    srs node) give 2^(6/2) = 8-dim irreducible representation.
    Decomposes as: 2 quarks × 3 colors + 2 leptons = 8.
    """
    kernel = kernel or CountingKernel()
    return kernel.orbit_count('fermion_content_per_gen')  # 8


def n_generations(kernel=None):
    """N_gen = 3 — number of fermion generations.

    Derived from: Galois Z_3 orbit of M^α ⊂ M ⊂ M ⋊_α Z_3 ≅ M_3(ℂ) ⊗ M^α
    on observer C³_obs. Jones index = 3.
    """
    kernel = kernel or CountingKernel()
    return kernel.orbit_count('generations')  # 3


def n_gauge_bosons(kernel=None):
    """N_gauge = 12 — number of SM gauge bosons.

    Counting: 8 gluons (SU(3)_c adjoint) + 3 SU(2)_L + 1 U(1)_Y = 12.
    """
    kernel = kernel or CountingKernel()
    return kernel.orbit_count('gauge_bosons')  # 12


def dark_feshbach_c(kernel=None):
    """c = 5/12 — dark Feshbach amplitude (Family 1, Class A spectral).

    Counting query: dim(marginal sector) / dim(B) = (2(|E|−|V|)+1) / (2|E|)
                  = (2·2 + 1) / 12 = 5/12.

    The marginal eigenmodes of Hashimoto B are the dark sector;
    their fraction of B's full spectrum is c = 5/12.
    """
    kernel = kernel or CountingKernel()
    n_E = kernel.substrate.N_EDGES
    n_V = kernel.substrate.N_ATOMS
    marginal_dim = 2 * (n_E - n_V) + 1
    total_dim = 2 * n_E
    return Fraction(marginal_dim, total_dim)  # 5/12
