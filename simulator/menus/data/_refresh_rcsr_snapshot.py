#!/usr/bin/env python3
"""
Regenerate `rcsr_candidates_snapshot.json` — the vendored, date-stamped parsed
snapshot of the RCSR crystal-net candidates the framework's substrate apparatus
references.

WHY VENDORED. `simulator.menus.crystal_nets` (Axis B — the substrate-
realization menu) needs the parsed RCSR entries to compute per-net fingerprints.
The live RCSR data is a one-off network fetch (`/tmp/rcsr_3d_current.txt`); a
fresh checkout / CI box won't have it, and the framework's substrate-uniqueness
claims should be reproducible against a *frozen, citable* input rather than an
ephemeral download. So we vendor a parsed snapshot. (Option (b) of the rebuild
plan — decouple the DATA dependency now; the fingerprint/DL/A2-T-waterfilling
LOGIC still lives in `proofs/foundations/` and is delegated to, pending the
later option (c) absorb.)

HOW TO REFRESH:
  1. curl -sL https://rcsr.anu.edu.au/data/3dall.txt -o /tmp/rcsr_3d_current.txt
  2. python simulator/menus/data/_refresh_rcsr_snapshot.py
  3. Review the diff to rcsr_candidates_snapshot.json (the `_meta` block records
     the fetch date + a SHA-256 of the source file so drift is visible).

The set of vendored nets = every coord-3 (3-regular) RCSR net the substrate
apparatus references — the 9 V+E-transitive 3-c chiral 3D candidates (srs,
srs-z, srs-c4, srs-c8, srs-c27, lou, lov, okw, hcb-c4) plus the achiral /
other-symmetry 3-regular nets (ths, ths-z, eta, etc, etd, utj) — plus a set of
non-3-regular REFERENCE nets the DL-comparison / k*-derivation work compares
against (qtz, dia, dia-c, pcu, nbo, bcu, fcu, sod, rho, lvt, cds, crs, unc, und,
une, unj). Reference nets are NOT substrate candidates (the framework requires
k* = 3); they're in the snapshot so DL/coordination comparisons are self-contained.
"""

import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]   # .../simulator/menus/data → repo root
_SRC = '/tmp/rcsr_3d_current.txt'
_OUT = _HERE / 'rcsr_candidates_snapshot.json'

# Coord-3 (3-regular) substrate candidates + achiral/other-symmetry 3-regular nets.
SUBSTRATE_3REGULAR = [
    'srs', 'srs-z', 'srs-c4', 'srs-c8', 'srs-c27', 'lou', 'lov', 'okw', 'hcb-c4',
    'ths', 'ths-z', 'eta', 'etc', 'etd', 'utj',
]
# Non-3-regular reference nets (DL / coordination comparison points; NOT substrate candidates).
REFERENCE_OTHER_COORD = [
    'qtz', 'dia', 'dia-c', 'pcu', 'nbo', 'bcu', 'fcu', 'sod', 'rho', 'lvt',
    'cds', 'crs', 'unc', 'und', 'une', 'unj',
]
ALL_NETS = SUBSTRATE_3REGULAR + REFERENCE_OTHER_COORD


def main() -> int:
    if not os.path.exists(_SRC):
        print(f"ERROR: {_SRC} not found. Fetch it first:\n"
              f"  curl -sL https://rcsr.anu.edu.au/data/3dall.txt -o {_SRC}")
        return 1
    sys.path.insert(0, str(_REPO))
    from proofs.foundations.rcsr_net_assessment import parse_rcsr_3dall

    with open(_SRC, 'rb') as f:
        src_bytes = f.read()
    src_sha = hashlib.sha256(src_bytes).hexdigest()

    entries = parse_rcsr_3dall(_SRC, ALL_NETS)
    missing = sorted(set(ALL_NETS) - set(entries))
    if missing:
        print(f"WARNING: {len(missing)} requested nets not found in {_SRC}: {missing}")

    snapshot = {
        '_meta': {
            'description': ('Vendored parsed RCSR crystal-net snapshot for '
                            'simulator.menus.crystal_nets (Axis-B substrate-'
                            'realization menu). Regenerate with _refresh_rcsr_snapshot.py.'),
            'source_url': 'https://rcsr.anu.edu.au/data/3dall.txt',
            'source_file': _SRC,
            'source_sha256': src_sha,
            'fetched_or_refreshed': date.today().isoformat(),
            'parser': 'proofs.foundations.rcsr_net_assessment.parse_rcsr_3dall',
            'n_nets': len(entries),
            'substrate_3regular': [n for n in SUBSTRATE_3REGULAR if n in entries],
            'reference_other_coord': [n for n in REFERENCE_OTHER_COORD if n in entries],
            'missing_from_source': missing,
        },
        'entries': {name: entries[name] for name in ALL_NETS if name in entries},
    }
    with open(_OUT, 'w') as f:
        json.dump(snapshot, f, indent=1, sort_keys=True)
        f.write('\n')
    print(f"Wrote {_OUT} — {len(entries)} nets, source SHA-256 {src_sha[:12]}…, "
          f"date {snapshot['_meta']['fetched_or_refreshed']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
