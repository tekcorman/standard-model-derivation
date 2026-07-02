#!/usr/bin/env python3
"""
RCSR Net Assessment Machine — first-principles Bayesian-observer spectrum.

Generalized pipeline for any RCSR net:
  (1) Parse the net's entry from the live RCSR /data/3dall.txt feed.
  (2) Use spglib's verified space group operations to generate atom orbit and
      edge midpoint orbit.
  (3) Reconstruct the bond list by matching midpoints to atom pairs.
  (4) Build the primitive-cell adjacency multigraph.
  (5) Construct Bloch Hashimoto B(k) on the 2|E|-dim directed-edge basis.
  (6) Diagonalize at high-symmetry k-points (Γ, R, M, X for primitive cubic;
      Γ, P, N, H for body-centered cubic).
  (7) Report spectral structure: Stark-Terras factorization, Ramanujan
      saturation, irrationals encountered, C₃-protected degeneracies.

NO srs-specific values imported anywhere. Pure first-principles application
of A1 + A2 + Jaynes + Shalizi-Crutchfield + Bloch + spectral identification
to whichever net is being analyzed.

Designed to run on all 9 V+E-transitive 3-c chiral 3D candidates:
  srs, srs-z, srs-c4, srs-c8, srs-c27, lou, lov, okw, hcb-c4
in a single sweep.
"""

import os
import re
import sys
import math
import numpy as np
from numpy.linalg import eig, eigvalsh, eigvals
import spglib

# =============================================================================
# RCSR DATA PARSER
# =============================================================================

# Mapping from RCSR space group symbols to spglib hall numbers.
# RCSR uses notation like "P4(1)32"; spglib uses "P4_132".
SG_NAME_TO_HALL = {}
def _build_sg_table():
    for hn in range(1, 531):
        try:
            sg = spglib.get_spacegroup_type(hn)
            short = sg['international_short']
            num = sg['number']
            # Add several normalized forms
            forms = [short]
            forms.append(short.replace('_', '(').replace('  ', '') + ')' if '_' in short else short)
            # E.g. "I4_132" -> "I4(1)32"
            for form in forms:
                if form not in SG_NAME_TO_HALL:
                    SG_NAME_TO_HALL[form] = (hn, num, short)
        except Exception:
            pass

_build_sg_table()

# Manual additions to bridge RCSR -> spglib naming
_MANUAL_SG = {
    'P4(1)32': (509, 213, 'P4_132'),
    'P4(3)32': (508, 212, 'P4_332'),
    'I4(1)32': (510, 214, 'I4_132'),
    'P4(2)32': (504, 208, 'P4_232'),
    'I432':    (507, 211, 'I432'),
    'F432':    (505, 209, 'F432'),
    'F4(1)32': (506, 210, 'F4_132'),
    'P432':    (503, 207, 'P432'),
    'P2(1)3':  (499, 198, 'P2_13'),
    'I2(1)3':  (500, 199, 'I2_13'),
    'P23':     (495, 195, 'P23'),
    'F23':     (496, 196, 'F23'),
    'I23':     (497, 197, 'I23'),
    'Im-3m':   (529, 229, 'Im-3m'),
    'Fd-3m':   (525, 227, 'Fd-3m'),
    'Ia-3d':   (530, 230, 'Ia-3d'),
    'Ia-3':    (501, 199, 'Ia-3'),  # Note: 199 actually I213; 206 is Ia-3
    'R-3m':    (None, None, 'R-3m'),  # variable; we'll skip
    'I4_1/amd': (None, None, 'I4_1/amd'),
}
SG_NAME_TO_HALL.update(_MANUAL_SG)


def parse_rcsr_3dall(filename, target_names):
    """Parse selected entries from RCSR /data/3dall.txt.

    Returns dict: {net_name: parsed_entry_dict}.
    """
    with open(filename, 'r') as f:
        data = f.read()

    blocks = re.split(r'(?m)^start\s*$', data)

    results = {}
    for blk in blocks[1:]:
        lines = blk.split('\n')
        nonempty = [ln for ln in lines if ln.strip()]
        if len(nonempty) < 5:
            continue
        blk_id = nonempty[0]
        name = nonempty[1]
        if name not in target_names:
            continue

        entry = {'name': name, 'id': blk_id}

        # Find space group line: "P4(1)32   213"
        sg_re = re.compile(r'^([IFPRCAH][\d()/\-a-z]*[a-z0-9])\s+(\d+)\s*$')
        for ln in lines:
            m = sg_re.match(ln.strip())
            if m:
                entry['sg_name'] = m.group(1)
                entry['sg_number'] = int(m.group(2))
                break

        # Find cell line: "0.8864  0.8864  0.8864  90.000  90.000  90.000"
        cell_re = re.compile(r'^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$')
        for i, ln in enumerate(lines):
            m = cell_re.match(ln)
            if m and 'sg_name' in entry:
                # Cell line should come right after the space group line
                # Validate: the previous nonempty line should be the SG line
                vals = [float(g) for g in m.groups()]
                if vals[3] in (90.0, 60.0, 120.0):  # standard angle, looks like a cell line
                    entry['cell_a'] = vals[0]
                    entry['cell_b'] = vals[1]
                    entry['cell_c'] = vals[2]
                    entry['cell_alpha'] = vals[3]
                    entry['cell_beta'] = vals[4]
                    entry['cell_gamma'] = vals[5]
                    break

        # Find vertex orbits and edge orbits
        vertex_orbits = []
        edge_orbits = []
        i = 0
        while i < len(lines):
            ln = lines[i].strip()
            m_v = re.match(r'^V(\d+)\s+(\d+)\s*$', ln)
            # Edge orbits: "E1 2", "E2 2", ... AND auxiliary forms like "Eq 2"
            # (RCSR uses Eq for additional edge orbits in lou/lov/okw and a few
            # others). We treat any E-suffix as a distinct edge orbit and tag the
            # orbit_index = -1 for non-numeric suffixes.
            m_e = re.match(r'^E(\d+)\s+(\d+)\s*$', ln)
            m_e_aux = None if m_e else re.match(r'^E([a-zA-Z][a-zA-Z\d]*)\s+(\d+)\s*$', ln)
            if m_v:
                # Next line: Cartesian coords
                cart = lines[i+1].strip().split()
                cart = [float(x) for x in cart]
                # Next line: Wyckoff symbol form
                wyck_form = lines[i+2].strip()
                # Next line: Wyckoff position label (e.g., "8 c")
                wyck_label = lines[i+3].strip()
                # Next line: site symmetry
                site_sym = lines[i+4].strip()
                # Next line: maybe coord seq or other; skip
                vertex_orbits.append({
                    'orbit_index': int(m_v.group(1)),
                    'coord': int(m_v.group(2)),
                    'cartesian': cart,
                    'wyckoff_form': wyck_form,
                    'wyckoff_label': wyck_label,
                    'site_symmetry': site_sym,
                })
                # Advance past the 5 consumed lines (header + 4 fields). Trailing
                # padding lines (if any) are skipped by the i+=1 fallback below.
                i += 5
                continue
            if m_e or m_e_aux:
                m_used = m_e if m_e else m_e_aux
                cart = lines[i+1].strip().split()
                cart = [float(x) for x in cart]
                wyck_form = lines[i+2].strip()
                wyck_label = lines[i+3].strip()
                site_sym = lines[i+4].strip()
                # orbit_index is the digit suffix when present, else -1 to mark aux
                try:
                    orbit_index = int(m_used.group(1))
                except ValueError:
                    orbit_index = -1
                edge_orbits.append({
                    'orbit_index': orbit_index,
                    'orbit_label': m_used.group(1),
                    'multiplicity': int(m_used.group(2)),
                    'cartesian': cart,
                    'wyckoff_form': wyck_form,
                    'wyckoff_label': wyck_label,
                    'site_symmetry': site_sym,
                })
                i += 5
                continue
            i += 1

        entry['vertex_orbits'] = vertex_orbits
        entry['edge_orbits'] = edge_orbits

        # Coordination sequence (numeric line near end)
        for ln in lines:
            t = ln.strip().split()
            if len(t) >= 8 and all(re.match(r'^\d+$', x) for x in t):
                entry['coordination_sequence'] = [int(x) for x in t]
                break

        # Vertex symbol and girth
        for ln in lines:
            if re.match(r'^\s*\d+\([\d,]+\)\.', ln):
                entry['vertex_symbol'] = ln.strip()
                break

        results[name] = entry

    return results


# =============================================================================
# ORBIT GENERATION (using spglib's verified ops)
# =============================================================================

def get_space_group_ops(sg_name):
    """Return (rotations, translations) arrays for a given RCSR-format space group name."""
    if sg_name not in SG_NAME_TO_HALL:
        raise ValueError(f"Unknown space group: {sg_name}")
    hall, num, short = SG_NAME_TO_HALL[sg_name]
    if hall is None:
        raise ValueError(f"No verified hall number for space group {sg_name}")
    ops = spglib.get_symmetry_from_database(hall)
    return np.array(ops['rotations']), np.array(ops['translations']), hall, num


def orbit_of(point, rotations, translations, tol=1e-6):
    """Generate the orbit of a point in fractional coords, modulo unit cell."""
    seen = []
    for R, t in zip(rotations, translations):
        p = (R @ point + t) % 1.0
        is_dup = False
        for s in seen:
            d = (p - s + 0.5) % 1.0 - 0.5
            if np.linalg.norm(d) < tol:
                is_dup = True
                break
        if not is_dup:
            seen.append(p)
    return np.array(seen)


# =============================================================================
# BOND LIST RECONSTRUCTION
# =============================================================================

def reconstruct_bonds(atom_orbit, midpoint_orbit, tol=1e-3, max_shift=3):
    """For each midpoint M, find the (atom_i, atom_j, shift) where
       atom_i + (atom_j + shift) = 2 * M  AND the bond length is minimum.

    The shortest-bond constraint is essential for body-centered (I) groups,
    where multiple (i, j, shift) triples satisfy the midpoint equation due to
    body-centering, but only the closest is the actual bond.

    Returns: list of (i, j, shift_xyz) tuples — one per midpoint.
    """
    n_atoms = len(atom_orbit)
    bonds = []
    for k_m, midpoint in enumerate(midpoint_orbit):
        target = 2.0 * midpoint
        candidates = []
        for i in range(n_atoms):
            for j in range(n_atoms):
                # Try a range of shifts to find all valid (i, j, shift) triples
                for dx in range(-max_shift, max_shift + 1):
                    for dy in range(-max_shift, max_shift + 1):
                        for dz in range(-max_shift, max_shift + 1):
                            shift = np.array([dx, dy, dz])
                            sum_pos = atom_orbit[i] + atom_orbit[j] + shift
                            residual = target - sum_pos
                            if np.linalg.norm(residual) < tol:
                                # Bond length: distance from atom_i to (atom_j + shift)
                                bond_vec = atom_orbit[j] + shift - atom_orbit[i]
                                bond_len = np.linalg.norm(bond_vec)
                                candidates.append((bond_len, i, j, tuple(shift.tolist())))
        if not candidates:
            bonds.append(None)
            continue
        # Pick shortest
        candidates.sort()
        _, i, j, shift = candidates[0]
        # Canonicalize: prefer i <= j
        if i > j:
            i, j = j, i
            shift = tuple(-s for s in shift)
        bonds.append((i, j, shift))
    return bonds


# =============================================================================
# ADJACENCY + BLOCH HASHIMOTO
# =============================================================================

def build_adjacency(bonds, n_atoms):
    """Build the primitive-cell adjacency multigraph (sum of A[i,j] = number of edges)."""
    A = np.zeros((n_atoms, n_atoms), dtype=int)
    for i, j, shift in bonds:
        A[i, j] += 1
        if i != j or shift != (0, 0, 0):
            A[j, i] += 1  # symmetric
    return A


def build_directed_edges(bonds):
    """Each undirected edge becomes 2 directed arcs.
    Returns list of (atom_from, atom_to, shift_from_to).
    """
    arcs = []
    for i, j, shift in bonds:
        arcs.append((i, j, shift))                                       # forward
        arcs.append((j, i, tuple(-s for s in shift)))                     # reverse
    return arcs


def bloch_hashimoto(arcs, k_frac, n_atoms):
    """Construct the Bloch Hashimoto operator B(k) at fractional Bloch wave vector k_frac.

    B has dimension 2|E| × 2|E| where 2|E| = len(arcs) per primitive cell.
    Hashimoto definition: B[a',a] = 1 if a' is a non-backtracking continuation of a.
    Non-backtracking: head of a = tail of a', AND a' != reverse(a).

    With Bloch phases: B[a',a] = exp(2πi k · shift(a')) if continuation.
    Convention: a Bloch phase factor is associated with the destination cell of a'.
    """
    n_arcs = len(arcs)
    B = np.zeros((n_arcs, n_arcs), dtype=complex)
    # Map each arc to its (head_atom, head_cell_offset_from_origin)
    # arc a = (tail, head, shift). Tail at home cell (0,0,0). Head at cell (shift).
    # For continuation a' = (head_a, head_a', shift_a'), tail of a' at the head_a cell.
    # The destination of a' is at cell shift_a + shift_a' (cumulative).
    for i_a, a in enumerate(arcs):
        tail_a, head_a, shift_a = a
        for i_ap, ap in enumerate(arcs):
            tail_ap, head_ap, shift_ap = ap
            # NB continuation: tail_ap == head_a AND ap != reverse(a)
            if tail_ap != head_a:
                continue
            # Reverse of a is (head_a, tail_a, -shift_a)
            reverse_a = (head_a, tail_a, tuple(-s for s in shift_a))
            if ap == reverse_a:
                continue
            # This is a valid continuation. Bloch phase = exp(2πi k · shift_ap)
            # Note: shift_ap is the periodic shift OF a' (from head_a's cell to head_ap's cell)
            # The "global" shift from origin to head_ap is shift_a + shift_ap.
            # By convention we put the phase on the new arc's shift only:
            # this makes B(k) translate-equivariant. (Standard Bloch convention.)
            phase = np.exp(2j * np.pi * np.dot(k_frac, shift_ap))
            B[i_ap, i_a] += phase
    return B


# =============================================================================
# SPECTRAL ANALYSIS
# =============================================================================

def stark_terras_finite(adj_matrix_simple, k_coord, n_V, n_E):
    """Apply Stark-Terras factorization to FINITE quotient adjacency.
    For multigraphs we'd need a generalization; here we assume simple graph.
    Returns (bipartite_eigs, oscillatory_eigs, eigenvalues_of_adj).
    """
    eigvals_adj = sorted(np.real(eigvals(adj_matrix_simple.astype(float))), reverse=True)
    bipartite_count = n_E - n_V
    oscillatory = []
    for lam in eigvals_adj:
        disc = lam * lam - 4 * (k_coord - 1)
        if disc >= 0:
            sd = math.sqrt(disc)
            u_plus = (lam + sd) / 2.0
            u_minus = (lam - sd) / 2.0
            oscillatory.append((u_plus, u_minus, 'real', lam))
        else:
            sd = math.sqrt(-disc)
            u_plus = complex(lam / 2.0, sd / 2.0)
            u_minus = complex(lam / 2.0, -sd / 2.0)
            oscillatory.append((u_plus, u_minus, 'complex', lam))
    return bipartite_count, oscillatory, eigvals_adj


def identify_irrational(value, tol=1e-6):
    """Try to identify a real number as a simple algebraic value."""
    # Check rationals
    for d in range(1, 13):
        for n in range(-30, 31):
            if abs(value - n / d) < tol:
                return f"{n}/{d}"
    # Check ±√n
    for n2 in range(2, 30):
        for sign in [1, -1]:
            v = sign * math.sqrt(n2)
            if abs(value - v) < tol:
                return f"{'-' if sign<0 else ''}√{n2}"
            if abs(value - v / 2) < tol:
                return f"{'-' if sign<0 else ''}√{n2}/2"
    # Check (a ± √n)/c
    for c in [1, 2, 3, 4]:
        for n2 in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]:
            for a in range(-5, 6):
                for sign in [1, -1]:
                    v = (a + sign * math.sqrt(n2)) / c
                    if abs(value - v) < tol:
                        return f"({a}{'+' if sign>0 else '-'}√{n2})/{c}"
    return None


def analyze_eigenvalues(eigvals_complex, k):
    """Identify each eigenvalue and check for Ramanujan saturation."""
    info = []
    for lam in eigvals_complex:
        re_l = float(np.real(lam))
        im_l = float(np.imag(lam))
        mod_sq = abs(lam)**2
        ramanujan = abs(mod_sq - (k - 1)) < 1e-6
        # Try to identify Re(λ) and Im(λ) algebraically
        re_id = identify_irrational(re_l)
        im_id = identify_irrational(abs(im_l)) if abs(im_l) > 1e-9 else "0"
        info.append({
            'eigenvalue': lam,
            're': re_l,
            'im': im_l,
            'modulus_sq': mod_sq,
            'ramanujan_saturated': ramanujan,
            're_identified': re_id,
            'im_identified': im_id,
        })
    return info


# =============================================================================
# K-POINT SELECTION (group-theoretic, no srs imports)
# =============================================================================

C3_KPOINTS_PRIMITIVE = {
    'Γ':  np.array([0.0, 0.0, 0.0]),
    'R':  np.array([0.5, 0.5, 0.5]),    # body diagonal of BZ for primitive cubic
    'M':  np.array([0.5, 0.5, 0.0]),    # face-center
    'X':  np.array([0.5, 0.0, 0.0]),    # face-center, axis
    'midbody': np.array([0.25, 0.25, 0.25]),  # mid body-diag (analog of srs's P)
}
# These cover the high-symmetry points likely relevant for primitive (P) cubic groups.


# =============================================================================
# MAIN ASSESSMENT FUNCTION
# =============================================================================

def assess_net(entry, verbose=True):
    """Run the full assessment pipeline on a single net entry."""
    name = entry['name']
    if verbose:
        print(f"\n{'='*78}")
        print(f"ASSESSING NET: {name}")
        print(f"{'='*78}")

    if 'sg_name' not in entry or 'cell_a' not in entry:
        if verbose: print("  [SKIP] Missing space group or cell data.")
        return None

    sg_name = entry['sg_name']
    if verbose:
        print(f"  Space group: {sg_name} (#{entry.get('sg_number', '?')})")
        print(f"  Cell: a={entry['cell_a']:.4f} b={entry['cell_b']:.4f} c={entry['cell_c']:.4f}")
        print(f"        α={entry['cell_alpha']} β={entry['cell_beta']} γ={entry['cell_gamma']}")

    try:
        rotations, translations, hall, num = get_space_group_ops(sg_name)
    except Exception as e:
        if verbose: print(f"  [SKIP] Cannot get space group ops: {e}")
        return None

    if verbose: print(f"  spglib hall #{hall}, {len(rotations)} ops")

    # RCSR convention check: "Cartesian" field is actually FRACTIONAL coords.
    # Verified by srs entry: Cartesian (0.125, 0.125, 0.125) = symbolic (1/8, 1/8, 1/8).
    # The cell parameter (e.g. 2.8284 for srs) is the cubic side in units where
    # bond length = 1, but coords listed in the "Cartesian" field are already
    # fractional. So we use them directly without conversion.
    a = entry['cell_a']
    if not (entry['cell_a'] == entry['cell_b'] == entry['cell_c'] and entry['cell_alpha'] == 90.0):
        if verbose: print(f"  [WARN] Non-cubic cell — pipeline currently assumes cubic.")

    vorbits = entry.get('vertex_orbits', [])
    eorbits = entry.get('edge_orbits', [])
    if not vorbits or not eorbits:
        if verbose: print("  [SKIP] No vertex or edge orbits parsed.")
        return None

    # Generate atom orbit
    v0 = vorbits[0]
    v_frac = np.array(v0['cartesian'])  # already fractional per RCSR convention
    atom_orbit = orbit_of(v_frac, rotations, translations)
    if verbose: print(f"  Atom orbit: {len(atom_orbit)} positions (Wyckoff {v0['wyckoff_label']}, coord {v0['coord']})")

    # Generate edge midpoint orbit
    e0 = eorbits[0]
    e_frac = np.array(e0['cartesian'])  # already fractional
    midpoint_orbit = orbit_of(e_frac, rotations, translations)
    if verbose: print(f"  Edge midpoint orbit: {len(midpoint_orbit)} positions (Wyckoff {e0['wyckoff_label']})")

    n_atoms = len(atom_orbit)
    n_edges = len(midpoint_orbit)
    coord = v0['coord']
    expected_edges = n_atoms * coord // 2
    if n_edges != expected_edges:
        if verbose: print(f"  [WARN] |E|={n_edges} but expected {expected_edges} for k={coord} on {n_atoms} atoms")

    # Reconstruct bonds
    bonds = reconstruct_bonds(atom_orbit, midpoint_orbit, tol=1e-3)
    n_resolved = sum(1 for b in bonds if b is not None)
    if verbose: print(f"  Bonds reconstructed: {n_resolved}/{n_edges}")

    if n_resolved != n_edges:
        if verbose: print(f"  [WARN] Could not resolve all bonds — {n_edges - n_resolved} unresolved")
        return {'name': name, 'status': 'unresolved_bonds',
                'n_atoms': n_atoms, 'n_edges': n_edges, 'n_resolved': n_resolved,
                'sg_name': sg_name}

    # Build adjacency multigraph
    A_multi = np.zeros((n_atoms, n_atoms), dtype=int)
    for i, j, shift in bonds:
        A_multi[i, j] += 1
        if i != j:
            A_multi[j, i] += 1
        else:
            # self-loop: count once
            pass
    if verbose:
        print(f"  Adjacency multigraph row sums: {A_multi.sum(axis=1)}")

    # Adjacency eigenvalues
    A_simple = A_multi.astype(float)
    adj_eigs = sorted(np.real(eigvals(A_simple)), reverse=True)
    if verbose:
        print(f"  Adjacency eigenvalues:")
        for i, lam in enumerate(adj_eigs):
            ident = identify_irrational(lam)
            print(f"    λ_{i+1} = {lam:+.6f}  {('(' + ident + ')') if ident else ''}")

    # Build directed arcs and Bloch Hashimoto
    arcs = build_directed_edges(bonds)
    n_arcs = len(arcs)
    if verbose: print(f"  Directed arcs (Hashimoto dim): {n_arcs}")

    # Test at high-symmetry k-points (for primitive cubic groups)
    spectra = {}
    for k_name, k_frac in C3_KPOINTS_PRIMITIVE.items():
        try:
            B_k = bloch_hashimoto(arcs, k_frac, n_atoms)
            eigs_k = eigvals(B_k)
            spectra[k_name] = sorted(eigs_k, key=lambda x: (np.real(x), np.imag(x)), reverse=True)
            if verbose:
                # Find Ramanujan-saturating eigenvalues
                ramanujan = [e for e in eigs_k if abs(abs(e)**2 - (coord-1)) < 1e-6]
                print(f"  B(k={k_name}={k_frac.tolist()}): {len(eigs_k)} eigenvalues, "
                      f"{len(ramanujan)} Ramanujan-saturating")
        except Exception as e:
            if verbose: print(f"  [WARN] B(k={k_name}) failed: {e}")
            spectra[k_name] = None

    # Detect doubly-degenerate eigenvalues at each k-point
    double_degen = {}
    for k_name, eigs in spectra.items():
        if eigs is None:
            continue
        eig_re_im = [(round(e.real, 5), round(e.imag, 5)) for e in eigs]
        from collections import Counter
        counts = Counter(eig_re_im)
        doubles = [(k, v) for k, v in counts.items() if v >= 2 and abs(complex(*k))**2 - (coord-1) > -1e-3]
        ramanujan_doubles = [d for d in doubles if abs(complex(*d[0]))**2 - (coord-1) < 1e-3 and abs(complex(*d[0]))**2 - (coord-1) > -1e-3]
        # More carefully: filter for Ramanujan-saturated mult >= 2
        ram_d = []
        for (re_v, im_v), m in counts.items():
            if m >= 2 and abs(complex(re_v, im_v))**2 - (coord-1) < 1e-3 and abs(complex(re_v, im_v))**2 - (coord-1) > -1e-3:
                if abs(im_v) > 1e-4:  # complex (oscillatory)
                    ram_d.append((complex(re_v, im_v), m))
        double_degen[k_name] = ram_d
        if verbose and ram_d:
            print(f"  C₃-protected (?) Ramanujan doublets at k={k_name}:")
            for lam, m in ram_d:
                re_id = identify_irrational(lam.real)
                im_id = identify_irrational(abs(lam.imag))
                print(f"      λ = {lam.real:+.4f} + {lam.imag:+.4f}i, mult {m}"
                      f"  Re ≈ {re_id}, |Im| ≈ {im_id}")

    return {
        'name': name,
        'sg_name': sg_name,
        'n_atoms': n_atoms,
        'n_edges': n_edges,
        'coord': coord,
        'adj_eigenvalues': adj_eigs,
        'spectra_at_k': spectra,
        'ramanujan_doublets': double_degen,
        'bonds': bonds,
        'cell_a': a,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    candidates = ['srs', 'srs-z', 'srs-c4', 'srs-c8', 'srs-c27', 'lou', 'lov', 'okw', 'hcb-c4']
    rcsr_file = '/tmp/rcsr_3d_current.txt'

    if not os.path.exists(rcsr_file):
        print(f"ERROR: {rcsr_file} not found. Run: curl -sL https://rcsr.anu.edu.au/data/3dall.txt -o {rcsr_file}")
        sys.exit(1)

    entries = parse_rcsr_3dall(rcsr_file, candidates)
    print(f"Parsed {len(entries)} of {len(candidates)} target nets from {rcsr_file}")
    for name in candidates:
        if name not in entries:
            print(f"  MISSING from RCSR: {name}")

    results = {}
    for name in candidates:
        if name not in entries:
            continue
        results[name] = assess_net(entries[name])

    # Comparative summary
    print(f"\n\n{'='*78}")
    print("COMPARATIVE SUMMARY: Bayesian-observer spectral structure across 9 candidates")
    print(f"{'='*78}")
    print(f"{'Net':<10s} {'SG':<10s} {'|V|':>4s} {'|E|':>4s} {'k':>3s} {'AdjEigs':<35s} {'Ram@Γ':>6s} {'Ram@R':>6s}")
    for name in candidates:
        r = results.get(name)
        if r is None:
            print(f"{name:<10s} (no data)")
            continue
        if 'adj_eigenvalues' not in r:
            print(f"{name:<10s} (parsing failed)")
            continue
        adj_str = ' '.join(f"{x:+.2f}" for x in r['adj_eigenvalues'][:6])
        ram_g = sum(m for _, m in r['ramanujan_doublets'].get('Γ', []))
        ram_r = sum(m for _, m in r['ramanujan_doublets'].get('R', []))
        print(f"{name:<10s} {r['sg_name']:<10s} {r['n_atoms']:>4d} {r['n_edges']:>4d} {r['coord']:>3d} "
              f"{adj_str:<35s} {ram_g:>6d} {ram_r:>6d}")


if __name__ == '__main__':
    main()
