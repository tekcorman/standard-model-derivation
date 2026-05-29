#!/usr/bin/env python3
"""§C closure experiment (HISTORICAL — superseded by joint-closure 2026-04-27).

Run the wave simulator twice — once with CCLOSE in initial tags (closure
hypothetically achieved), once without (baseline). Diff the firing sets
to see what §C closure unlocks mechanically.

Original result (run 2026-04-26):
  Baseline:  173/195 ops fire,  Φ=183.34, L=522, Net=-339
  Closure:   195/195 ops fire,  Φ=189.34, L=593, Net=-404
  Δ:         +22 ops unlocked, +6 Φ, +71 L, -65 Net
  → §C closure unlocks the GR/cosmology/CFT machinery (smooth manifold,
    Riemann, Ricci, Einstein, Friedmann, FLRW, Hubble, ...) but only
    1 of the 22 ops contributes substrate-counting Φ (6.8 de Rham).
    The compression payoff lives at the prediction layer, not at
    catalog construction.

POST-CLOSURE (2026-04-27): CCLOSE has been replaced by NC_GEOM in the
simulator (per docs/theorems/lorentz_sig_ccclose_joint_closure.md). NC_GEOM is
established by op 7.1 (spectral triple); LORENTZ_SIG is established by
op 6.10 (Lorentzian metric). All 219 catalog ops now fire; this script
will show "Δ = 0 unlocked" since CCLOSE no longer gates anything. The
script is preserved as historical record of the pre-closure analysis.
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulator as ws
importlib.reload(ws)

final_open, hist_open = ws.run_full()

def run_with_cclose():
    state = ws.WaveState(
        refinements=frozenset(),
        tags=set(ws.INITIAL_TAGS) | {'CCLOSE'},
        fired=[], fired_ids=set(),
        refinements_used=set(),
        templates_used=set(),
        Phi_total=0.0, L_total=0,
        objects=[],
    )
    history = [state]
    while True:
        nxt = ws.step_cascade(state)
        if nxt is None: break
        state = nxt
        history.append(state)
    return state, history

final_closed, hist_closed = run_with_cclose()
unlocked = final_closed.fired_ids - final_open.fired_ids

print("="*100)
print("§C CLOSURE EXPERIMENT")
print("="*100)
print(f"\nBaseline (CCLOSE open):    {len(final_open.fired_ids):3d}/195 fired,  Φ={final_open.Phi_total:7.2f}  L={final_open.L_total:4d}  Net={final_open.Net:+7.2f}")
print(f"Closure  (CCLOSE in tags):  {len(final_closed.fired_ids):3d}/195 fired,  Φ={final_closed.Phi_total:7.2f}  L={final_closed.L_total:4d}  Net={final_closed.Net:+7.2f}")
print(f"\nUnlocked: {len(unlocked)} ops ({len(unlocked)/len(ws.CATALOG)*100:.1f}% of catalog)")
print(f"Δ Φ: {final_closed.Phi_total - final_open.Phi_total:+.2f}  Δ L: {final_closed.L_total - final_open.L_total:+d}  Δ Net: {final_closed.Net - final_open.Net:+.2f}")

print(f"\nNewly-fired ops:")
print(f"{'op':>5} L{'#':<1} {'name':<50} {'Φ':>6} {'L':>3} {'Net':>7}")
for op in final_closed.fired:
    op_id, layer, name, tmpl, L, extras, ref = op
    if op_id in unlocked:
        idx = next(i for i, st in enumerate(hist_closed[1:], 1) if st.fired[-1][0] == op_id)
        Phi = hist_closed[idx].Phi_total - hist_closed[idx-1].Phi_total
        print(f"{op_id:>5} L{layer}   {name:<50} {Phi:>6.2f} {L:>3} {Phi-L:>+7.2f}")
