#!/usr/bin/env python3
"""
proofs/foundations/W2_selftest_wire_2026-07-13.py

[PUSH 3, L9 hygiene -- verify.py wiring batch, 2026-07-14]

Thin FAST wrapper for the_net.py's w2_selftest_2026_07_13 (the W2 station regression: welded-state
construction, the EXACT level-1 Schur mechanism + numeric level-2 confirmation, direction
independence, T1/T2 well-posedness, the honesty clause, and the PAIR read's J_F-conjugation
identity). That function was already written self-contained and fast (< 120s verify per-entry
timeout law, measured ~0.6s) but explicitly NOT wired into verify.py at write time ("integration
batch, L9"). This file is that wiring: read-only import, one call, exit by its own return value.

Does NOT re-derive or adjudicate anything; the_net.py is not modified.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
import the_net as net  # noqa: E402  the ONE master Layer-3 object; nothing rebuilt here

if __name__ == "__main__":
    ok = net.w2_selftest_2026_07_13(verbose=True)
    sys.exit(0 if ok else 1)
