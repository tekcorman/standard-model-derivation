# proofs/_archive/

42 probe scripts retired here by the 2026-05-21 orphan triage. Each was cited by
no documentation and, on being read and cross-checked against the live
framework, found to be one of:

- **superseded** — a later probe or theorem does the same job and is the one
  actually used (e.g. the `vus_*` / `vcb_*` spectral and Feshbach routes,
  replaced by `vus_l2_density.py` / `vcb_nfixed_proof.py`);
- **dead-end** — an exploratory probe or negative result that was not carried
  forward and has no successor.

They are kept, not deleted: a dead-end probe is the record of *why* a mechanism
was tried and abandoned, which is part of the framework's honest trail. Nothing
here is live — no live code imports these, no current doc cites them. Do not
cite a script in this directory as evidence for a current claim.

Scripts still cited by archived session docs, or still imported as utilities by
live probes, were **not** moved here — they remain under `proofs/<sector>/`.
See `proofs/README.md` for the full relevance tiering.
