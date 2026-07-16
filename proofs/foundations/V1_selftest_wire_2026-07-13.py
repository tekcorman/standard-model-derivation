#!/usr/bin/env python3
"""
proofs/foundations/V1_selftest_wire_2026-07-13.py

[PUSH 3, L9 hygiene -- verify.py wiring batch, 2026-07-14]

Thin FAST wrapper for the_net.py's v1_selftest_2026_07_13 (the V1 station regression: the two
module anchors, occupation-basis transform exactness, channel-state exactness (norm / Schur-blind
field marginal / purity identity), the copy-overlap level lemma + Holevo mechanism identity,
global-phase drop and conj-invariance of the pair functionals, a small-sample step-0 spread
contrast, and one pinned regression value). That function was already written self-contained and
fast (< 120s verify per-entry timeout law, measured ~9.8s) but explicitly NOT wired into verify.py
at write time ("integration batch, L9"). This file is that wiring: read-only import, one call,
exit by its own return value.

Does NOT re-derive or adjudicate anything; the_net.py is not modified.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "derivation_topdown", "state"))
import the_net as net  # noqa: E402  the ONE master Layer-3 object; nothing rebuilt here

if __name__ == "__main__":
    ok = net.v1_selftest_2026_07_13(verbose=True)
    sys.exit(0 if ok else 1)
