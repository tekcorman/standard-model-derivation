#!/usr/bin/env python3
"""
predictions/E_count.py — |E| = 6, edge count of the K_4 quotient graph
(equivalently, edge count of the srs primitive cell on the directed-edge
representation when collapsed).

|E| follows from the handshake lemma applied to the K_4 quotient:

    Σ_{v∈V} deg(v) = 2 |E|     ⇒     |E| = |V| · k_star / p_toggle

For srs / K_4 quotient: |V| = 4, k_star = 3, p_toggle = 2 (incidence
multiplicity per edge: each undirected edge has 2 endpoints) ⇒
|E| = 4·3/2 = 6.

This is the same |E| that appears in the Ihara-Bass identity for K_4
(|E| = 6 with k = 3 giving the 6-dim |E|-(|V|-1) = 3-dim residual at
u = ±1 in the NB walk operator factorization). The Hashimoto operator
B at the P-point has dim 2|E| = 12 = N·k (handshake), and the marginal
modes (u² = 1 prefactor) span the (|E|−|V|+1) = 3-dim Wilson-loop
H¹(K_4) sector that drives the framework's α_GUT dark correction.

Companion leaves: predictions/V_count.py (= 4), predictions/k_star.py
(= 3), predictions/p_toggle.py (= 2), predictions/g_girth.py (= 10).
"""

# ============================================================
# PARAMETER: |E| (edge count of K_4 quotient / srs primitive cell)
# ============================================================

# --- OBSERVED VALUE ------------------------------------------
# Not a directly observed quantity. The framework's prediction is a
# structural integer (handshake lemma + Sunada 2012 srs primitive cell).

# --- PREDICTED VALUE -----------------------------------------
# Value:       |E| = 6 (proven for k_star = 3, |V| = 4, p_toggle = 2)
# Deviation:   N/A (structural integer)

# --- DERIVED FORMULA -----------------------------------------
# Handshake lemma (Euler):
#   Σ_{v∈V} deg(v) = 2·|E|
# For a k-regular graph, deg(v) = k_star for all v, so:
#   |V|·k_star = 2·|E|
#   ⇒  |E| = |V|·k_star / p_toggle   (where p_toggle = 2 = endpoints/edge)
#
# For k_star = 3, |V| = 4: |E| = 4·3/2 = 6.

# --- INPUTS --------------------------------------------------
# symbol   | value | status    | predictions/ file        | meaning
# ---------|-------|-----------|--------------------------|--------
# k_star   | 3     | [derived] | predictions/k_star.py    | coordination number
# V_count  | 4     | [derived] | predictions/V_count.py   | |V| of K_4 / srs primitive cell
# p_toggle | 2     | [derived] | predictions/p_toggle.py  | edge endpoint count (handshake)

# --- IMPLEMENTATION ------------------------------------------

import functools


# --- PURE FUNCTION -------------------------------------------

@functools.lru_cache(maxsize=None)
def predict_E_count(k_star, V_count, p_toggle):
    """
    Return |E| = |V|·k_star/p_toggle (handshake lemma).

    Parameters
    ----------
    k_star : int
        Coordination number (predict_k_star).
    V_count : int
        Vertex count (predict_V_count).
    p_toggle : int
        Toggle arity / endpoints per edge (predict_p_toggle).

    Returns
    -------
    int
        |E| = edge count of the K_4 quotient.
    """
    return V_count * k_star // p_toggle


# --- INTROSPECTION (for run_predictions.py) ------------------
from k_star import predict_k_star
from d_spatial import predict_d_spatial
from V_count import predict_V_count
from p_toggle import predict_p_toggle

_d = predict_d_spatial()
_k = predict_k_star(_d)
_V = predict_V_count(_k, _d)
_p = predict_p_toggle()
E_count_pred = predict_E_count(_k, _V, _p)


# --- VALIDATION ----------------------------------------------

if __name__ == "__main__":
    print("=" * 68)
    print("  |E|  --  edge count of K_4 quotient (srs primitive cell)")
    print("=" * 68)
    print(f"  k_star       = {_k}     (predictions/k_star.py)")
    print(f"  |V|          = {_V}     (predictions/V_count.py)")
    print(f"  p_toggle     = {_p}     (predictions/p_toggle.py)")
    print(f"  |E| = V·k/p = {E_count_pred}  (handshake lemma)")
    print()
    print("  Companion: 2|E| = V·k = 12 = N·k (cell-NB Hashimoto dim).")
