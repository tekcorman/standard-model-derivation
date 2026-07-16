#!/usr/bin/env python3
"""
proofs/foundations/FOCK2_selftest_wire_2026-07-14.py

[PUSH 3, L9 hygiene -- verify.py wiring batch, 2026-07-14]

Thin FAST wrapper for the_net.py's fock2_selftest_2026_07_14 (the FOCK-2 section-12 self-test:
dimension-1 sectors exactly blind by dimension count on a random direction, the per-shell
decomposition reconstructing v1_F2_F3's own omega-weighted aggregate, and the per-sector read's
gauge-invariance across the triad's own headline coordinates). That function was already written
self-contained and fast (< 120s verify per-entry timeout law, measured ~3.9s) but explicitly NOT
wired into verify.py at write time ("integration batch, L9"). This file is that wiring: read-only
import, one call, exit by its own return value.

Does NOT re-derive or adjudicate anything; the_net.py is not modified.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
import the_net as net  # noqa: E402  the ONE master Layer-3 object; nothing rebuilt here

if __name__ == "__main__":
    ok = net.fock2_selftest_2026_07_14(verbose=True)
    sys.exit(0 if ok else 1)
