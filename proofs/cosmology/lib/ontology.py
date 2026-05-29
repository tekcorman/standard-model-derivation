"""
ontology.py — Frame tagging and substrate/observer/LCDM translation.

WHY THIS MODULE EXISTS
----------------------
The single biggest source of confusion in framework cosmology work is
conflating substrate-side, observer-side, and LCDM-extracted quantities.
The 2026-05-07 IC closure made observer-MDL primary; the 2026-05-08 (γ)
parametric-class-translation framing made the three-frame distinction
unavoidable. Encoding the distinction in code at the API level prevents
silent drift.

THREE FRAMES
------------
Frame.SUBSTRATE
    Substrate-native rate. H(z) follows graph-growth at substrate level;
    pure coasting H_substrate = 1/(N(z) * t_P). Time is substrate Planck
    ticks. No observer-side rate gap applied.

Frame.OBSERVER
    Framework's compressed observer-MDL reading. H_observer = (16/15) *
    H_substrate at z=0 (theorem-grade-conditional per
    theorem_cascade_D2_extended_observer_rate.md). Time is observer-
    proper. Observables in this frame are what the framework predicts the
    observer reads.

Frame.LCDM_EXTRACTED
    Standard-cosmology fit-class extraction (LambdaCDM Friedmann two-
    component, or with extensions). The "Omega_m = 0.315, Omega_L = 0.685"
    of Planck data is in this frame. NOT a framework-native quantity; it
    is the observer's translation under an assumed parametric class.

Mixing these without explicit translation is a structural error. The
library's translate(...) primitive forces the translation to be named.

REFERENCES
----------
docs/theorems/theorem_cascade_D2_extended_observer_rate.md
an internal working note
an internal working note
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Callable


class Frame(enum.Enum):
    """Cosmological frame tag.

    SUBSTRATE        — substrate-native rate, pre-observer-MDL gap.
    OBSERVER         — framework's observer-MDL reading.
    LCDM_EXTRACTED   — humans' Friedmann-class extraction (LCDM fit).
    """

    SUBSTRATE = "substrate"
    OBSERVER = "observer"
    LCDM_EXTRACTED = "lcdm_extracted"


# ---------------------------------------------------------------------------
# Frame-translation primitives
# ---------------------------------------------------------------------------
# Every primitive takes a numerical value with an INPUT frame tag and an
# explicit named correction; it returns the value in an OUTPUT frame tag.
# The correction is a pure number — its derivation is the caller's
# responsibility (cite the theorem document; the library does not fit it).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tagged:
    """A numerical value carrying a Frame tag.

    Carrying the tag in a small wrapper makes mis-composition fail loudly
    at the call site rather than silently. Use .value to extract the float
    once a translation has been performed.

    The value may be any floating-point quantity. The tag is immutable.
    """

    value: float
    frame: Frame

    def expect(self, frame: Frame) -> float:
        """Return the underlying value, asserting the frame matches.

        Raises ValueError if the tag is not the expected frame. Use this at
        a call site that is contractually frame-specific (e.g., a forward
        model that requires observer-frame H_0).
        """
        if self.frame is not frame:
            raise ValueError(
                f"Frame mismatch: expected {frame.value!r}, "
                f"got {self.frame.value!r}. Translate first."
            )
        return self.value


def translate(
    tagged: Tagged,
    target: Frame,
    factor: float,
    citation: str,
) -> Tagged:
    """Translate a tagged value to a different frame by a stated factor.

    The factor is named and cited; the library does not derive it. This
    forces the caller to make the translation explicit and traceable.

    Parameters
    ----------
    tagged : Tagged
        Source value with its frame tag.
    target : Frame
        Destination frame.
    factor : float
        Multiplicative factor: target_value = factor * source_value.
        For SUBSTRATE -> OBSERVER on H_0, this is 16/15 per
        theorem_cascade_D2_extended_observer_rate.md.
    citation : str
        Non-empty string identifying where `factor` is derived. Forces the
        caller to be explicit about what they are invoking.

    Returns
    -------
    Tagged
        Same numerical value scaled by `factor`, tagged with `target`.
    """
    if tagged.frame is target:
        raise ValueError(
            f"translate(): source and target frames are both "
            f"{target.value!r}; nothing to translate."
        )
    if not citation:
        raise ValueError(
            "translate(): citation must be a non-empty string identifying "
            "the derivation of `factor`. Untraceable frame translations "
            "are a structural error."
        )
    return Tagged(value=tagged.value * factor, frame=target)


# ---------------------------------------------------------------------------
# Convenience: lift a callable so it propagates a Frame tag through a
# pure transformation that does not change the frame (e.g., taking H(z)
# from H_0 in observer frame).
# ---------------------------------------------------------------------------


def in_frame(frame: Frame, fn: Callable[..., float]) -> Callable[..., Tagged]:
    """Wrap a pure function so its output is Tagged with `frame`.

    The wrapped function takes the same arguments as `fn` and returns a
    Tagged whose .value is fn(...) and whose .frame is `frame`. Use only
    when the operation is intrinsically frame-preserving (e.g., evaluating
    H(z) at a given z, where z is dimensionless and the frame inheres in
    H_0 itself).
    """

    def wrapped(*args, **kwargs) -> Tagged:
        return Tagged(value=fn(*args, **kwargs), frame=frame)

    return wrapped


# ---------------------------------------------------------------------------
# Self-test — runs under `python -m proofs.cosmology.lib.ontology`.
# Not a prediction; a sanity check that the module behaves as documented.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Substrate-frame H_0 (numerical example: 68.18 km/s/Mpc per
    # predictions/H_0.py; cited at use site, not hardcoded here).
    h0_sub = Tagged(value=68.18, frame=Frame.SUBSTRATE)

    # Translate to observer frame using the (16/15) cascade D2-extended
    # rate gap — caller must cite. Library does not assert correctness of
    # the factor; it only forces it to be named.
    h0_obs = translate(
        h0_sub,
        target=Frame.OBSERVER,
        factor=16.0 / 15.0,
        citation="theorem_cascade_D2_extended_observer_rate.md",
    )

    print("Substrate H_0:", h0_sub)
    print("Observer  H_0:", h0_obs)

    # Frame mismatch errors loudly:
    try:
        h0_sub.expect(Frame.OBSERVER)
    except ValueError as e:
        print("Caught expected frame mismatch:", e)

    # Untraceable translations fail:
    try:
        translate(h0_sub, target=Frame.OBSERVER, factor=1.0, citation="")
    except ValueError as e:
        print("Caught untraceable translate:", e)
